import copy
import numpy as np
import tensorflow as tf
from motornet.nets.losses import PositionLoss, L2xDxActivationLoss, L2xDxRegularizer, CompoundedLoss
from motornet.tasks import Task
from typing import Union
import time


class CentreOutFF(Task):
    def __init__(
            self,
            network,
            name: str = 'CentreOutReach',
            angular_step: float = 15,
            catch_trial_perc: float = 50,
            reaching_distance: float = 0.1,
            start_position: Union[list, tuple, np.ndarray] = None,
            deriv_weight: float = 0.,
            go_cue_range: Union[list, tuple, np.ndarray] = (0.05, 0.25),
            **kwargs
    ):

        super().__init__(network, name=name, **kwargs)

        self.angular_step = angular_step
        self.catch_trial_perc = catch_trial_perc
        self.reaching_distance = reaching_distance
        self.start_position = start_position
        if not self.start_position:
            # start at the center of the workspace
            lb = np.array(self.network.plant.pos_lower_bound)
            ub = np.array(self.network.plant.pos_upper_bound)
            self.start_position = lb + (ub - lb) / 2
        self.start_position = np.array(self.start_position).reshape(1, -1)

        muscle_loss = L2xDxActivationLoss(
            max_iso_force=self.network.plant.muscle.max_iso_force,
            dt=self.network.plant.dt,
            deriv_weight=deriv_weight
        )
        self.add_loss('muscle state',       loss_weight=5.0, loss=muscle_loss)          # 5.0

        gru_loss = L2xDxRegularizer(       deriv_weight=0.05, dt=self.network.plant.dt) # 0.05
        self.add_loss('gru_hidden_0',       loss_weight=0.1, loss=gru_loss)             # 0.1

        self.add_loss('cartesian position', loss_weight=2.0, loss=PositionLoss())       # 1.0

        go_cue_range = np.array(go_cue_range) / self.network.plant.dt
        self.go_cue_range = [int(go_cue_range[0]), int(go_cue_range[1])]
        self.delay_range = self.go_cue_range

        # CW by default
        self.FF_matvel = tf.convert_to_tensor(kwargs.get('FF_matvel', np.array([[0,1],[-1,0]])), dtype=tf.float32)
        self.FF_matvel_x = tf.slice(self.FF_matvel, [0, 0], [1, -1])
        self.FF_matvel_y = tf.slice(self.FF_matvel, [1, 0], [1, -1])

        self.apply_ff = tf.keras.layers.Lambda(
            lambda x: x[0] * tf.concat(
                [
                    tf.reduce_sum(self.FF_matvel_x * x[1], axis=1, keepdims=True),
                    tf.reduce_sum(self.FF_matvel_y * x[1], axis=1, keepdims=True),
                ], axis=1),
            name="apply_force_field"
        )

    def generate(self, batch_size, n_timesteps, condition="test", ff_coefficient: float = 0.):
        """
        condition = "train": learn to reach to random targets in workspace in a NF
                    "test" : centre-out reaches to each target in a given FF/NF
                    "adapt": re-learn centre-out reaches in a given FF/NF
        """
        catch_trial = np.zeros(batch_size, dtype='float32')
        if (condition=="train"): # train net to reach to random targets in workspace in a NF
            init_states   = self.get_initial_state(batch_size=batch_size)
            goal_states_j = self.network.plant.draw_random_uniform_states(batch_size=batch_size)
            goal_states   = self.network.plant.joint2cartesian(goal_states_j)
            p             = int(np.floor(batch_size * self.catch_trial_perc / 100))
            catch_trial[np.random.permutation(catch_trial.size)[:p]] = 1.
            
        elif (condition=="test"): # centre-out reaches to each target in a given FF/NF
            angle_set   = np.deg2rad(np.arange(0, 360, self.angular_step))
            reps        = int(np.ceil(batch_size / len(angle_set)))
            angle       = np.tile(angle_set, reps=reps)
            batch_size  = reps * len(angle_set)
            start_jpv   = np.concatenate([self.start_position, np.zeros_like(self.start_position)], axis=1)
            start_cpv   = self.network.plant.joint2cartesian(start_jpv)
            end_cp      = self.reaching_distance * np.stack([np.cos(angle), np.sin(angle)], axis=-1)
            init_states = self.network.get_initial_state(batch_size=batch_size, inputs=start_jpv)
            goal_states = start_cpv + np.concatenate([end_cp, np.zeros_like(end_cp)], axis=-1)
            catch_trial = np.zeros(batch_size, dtype='float32')

        elif (condition=="adapt"): # re-learn centre-out reaches in a given FF/NF
            angle_set   = np.deg2rad(np.arange(0, 360, self.angular_step))
            reps        = int(np.ceil(batch_size / len(angle_set)))
            angle       = np.tile(angle_set, reps=reps)
            batch_size  = reps * len(angle_set)
            start_jpv   = np.concatenate([self.start_position, np.zeros_like(self.start_position)], axis=1)
            start_cpv   = self.network.plant.joint2cartesian(start_jpv)
            end_cp      = self.reaching_distance * np.stack([np.cos(angle), np.sin(angle)], axis=-1)
            init_states = self.network.get_initial_state(batch_size=batch_size, inputs=start_jpv)
            goal_states = start_cpv + np.concatenate([end_cp, np.zeros_like(end_cp)], axis=-1)
            p             = int(np.floor(batch_size * self.catch_trial_perc / 100))
            catch_trial[np.random.permutation(catch_trial.size)[:p]] = 1.

        startpos     = self.network.plant.joint2cartesian(init_states[0][:, :])
        go_cue       = np.ones([batch_size, n_timesteps, 1])
        targets      = self.network.plant.state2target(state=goal_states, n_timesteps=n_timesteps).numpy()
        inputs_targ  = copy.deepcopy(targets[:, :, :self.network.plant.space_dim])
        tmp          = np.repeat(startpos[:, np.newaxis, :self.network.plant.space_dim], n_timesteps, axis=1)
        inputs_start = copy.deepcopy(tmp)
        for i in range(batch_size):
            if ((condition=="train") or (condition=="adapt")):
                go_cue_time = int(np.random.uniform(self.go_cue_range[0], self.go_cue_range[1]))
            elif (condition=="test"):
                go_cue_time = int(self.go_cue_range[0])
            if catch_trial[i] > 0.:
                targets[i, :, :] = startpos[i, np.newaxis, :]
            else:
                targets[i, :go_cue_time, :] = startpos[i, np.newaxis, :]
                inputs_start[i, go_cue_time + self.network.visual_delay:, :] = 0.
                go_cue[i, go_cue_time + self.network.visual_delay:, 0] = 0.

        inputs = {
            "inputs": np.concatenate([inputs_start, inputs_targ, go_cue], axis=-1),
            "ff_coefficient": ff_coefficient * np.ones((batch_size, n_timesteps, 1)),
        }
        return [inputs, self.convert_to_tensor(targets), init_states]

    def recompute_inputs(self, inputs, states):
        b = inputs['ff_coefficient']

        _, cstate, _, _ = self.network.unpack_plant_states(states)
        vel = cstate[:, 2:]

        inputs['endpoint_load'] = self.apply_ff((b, vel))  # [2x2] x [2xbatch] = [2xbatch]
        return inputs

