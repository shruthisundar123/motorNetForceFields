import os
import sys
import json
import copy
import numpy as np
import tensorflow as tf
from motornet.tasks import Task
from motornet.plants import RigidTendonArm26
from motornet.plants.muscles import RigidTendonHillMuscleThelen
from motornet.nets.layers import GRUNetwork
from motornet.nets.models import MotorNetModel
from motornet.nets.losses import L2xDxActivationLoss, L2xDxRegularizer, PositionLoss
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.losses import LossFunctionWrapper

auto_reduction = losses_utils.ReductionV2.AUTO
file_dir = sys.path[0]


class ClippedPositionLoss(LossFunctionWrapper):

    def __init__(self, target_size: float, name: str = 'position', reduction=auto_reduction):
        super().__init__(_clipped_position_loss, name=name, reduction=reduction, target_size=target_size)


def _clipped_position_loss(y_true, y_pred, target_size):
    true_pos, _ = tf.split(y_true, 2, axis=-1)
    pred_pos, _ = tf.split(y_pred, 2, axis=-1)
    err = true_pos - pred_pos
    l1 = tf.abs(err)
    l2 = tf.reduce_sum(tf.sqrt(err ** 2), axis=-1, keepdims=True)
    return tf.reduce_mean(tf.where(l2 < target_size, 0., l1))


class CustomTask(Task):

    def __init__(
            self,
            network,
            name: str = 'CustomTask',
            angular_step: float = 45,
            catch_trial_perc: float = 50,
            reaching_distance: float = 0.1,
            start_position=None,
            go_cue_range=(0.05, 0.30),
            perturbation_range=(0.05, 0.25),
            **kwargs
    ):

        super().__init__(network, name=name, **kwargs)

        self.training_n_timesteps = 80
        self.angular_step = angular_step
        self.catch_trial_perc = catch_trial_perc
        self.reaching_distance = reaching_distance
        self.start_position = start_position
        if not self.start_position:
            # start at the center of the workspace
            lb = np.array(self.network.plant.pos_lower_bound)
            ub = np.array(self.network.plant.pos_upper_bound)
            self.start_position = lb + (ub - lb) / 2
        self.start_position = np.array(self.start_position).reshape(-1).tolist()

        max_iso_force = self.network.plant.muscle.max_iso_force
        muscle_loss = L2xDxActivationLoss(max_iso_force=max_iso_force, dt=self.network.plant.dt, deriv_weight=0.)
        gru_loss = L2xDxRegularizer(deriv_weight=0.05, dt=self.network.plant.dt)
        self.add_loss('gru_hidden_0', loss_weight=0.1, loss=gru_loss)
        self.add_loss('muscle state', loss_weight=5., loss=muscle_loss)
        self.add_loss('cartesian position', loss_weight=2., loss=ClippedPositionLoss(target_size=0.01))

        go_cue_range = np.array(go_cue_range) / self.network.plant.dt
        self.go_cue_range = [int(go_cue_range[0]), int(go_cue_range[1])]

        perturbation_range = np.array(perturbation_range) / self.network.plant.dt
        self.perturbation_range = [int(perturbation_range[0]), int(perturbation_range[1])]
        self.delay_range = [x + y for x, y in zip(self.perturbation_range, self.go_cue_range)]

        # self.perturbations = [0., 4.]

        self.perturbations_validation = [3., 6.]
        self.perturbations_validation.extend([-1 * perturbation for perturbation in self.perturbations_validation])
        self.perturbations_validation.extend([0.])

    def generate(self, batch_size, n_timesteps, validation: bool = False):

        catch_trial = np.zeros(batch_size, dtype='float32')
        condition_hold = np.mod(np.random.permutation(batch_size), 2)

        if validation:
            n_timesteps = 200
            angle_set = np.deg2rad(np.arange(0, 360, self.angular_step))
            batch_size = 2 * len(angle_set) * len(self.perturbations_validation)
            angle = np.tile(angle_set, reps=int(batch_size / len(angle_set)))

            perturbation = list(np.sort(self.perturbations_validation * len(angle_set))) * 2
            catch_trial = np.zeros(batch_size, dtype='float32')  # must be re-drawn to match new batch size
            condition_hold = np.sort(np.mod(np.random.permutation(batch_size), 2))  # same

            start_jpv = np.concatenate([self.start_position, np.zeros_like(self.start_position)])[np.newaxis, :]
            start_cpv = self.network.plant.joint2cartesian(start_jpv)
            end_cp = self.reaching_distance * np.stack([np.cos(angle), np.sin(angle)], axis=-1)
            init_states = self.network.get_initial_state(batch_size=batch_size, inputs=start_jpv)
            goal_states = start_cpv + np.concatenate([end_cp, np.zeros_like(end_cp)], axis=-1)

        else:
            init_states = self.get_initial_state(batch_size=batch_size)
            goal_states_j = self.network.plant.draw_random_uniform_states(batch_size=batch_size)
            goal_states = self.network.plant.joint2cartesian(goal_states_j)

            half_batch = int(np.floor(batch_size / 2))
            catch_trial[np.random.permutation(batch_size)[:half_batch]] = 1.

            # to_flip = np.random.permutation(batch_size)[:half_batch]
            # to_clip = np.random.permutation(batch_size)[:half_batch]
            perturbation = np.zeros(batch_size)
            # perturbation[to_flip] = -1. * perturbation[to_flip]
            # perturbation[to_clip] = 0.

        center = self.network.plant.joint2cartesian(init_states[0][:, :])
        go_cue = np.ones([batch_size, n_timesteps, 1])
        targets = self.network.plant.state2target(state=goal_states, n_timesteps=n_timesteps).numpy()
        inputs_targ = copy.deepcopy(targets[:, :, :self.network.plant.space_dim])
        tmp = np.repeat(center[:, np.newaxis, :self.network.plant.space_dim], n_timesteps, axis=1)
        inputs_start = copy.deepcopy(tmp)

        endpoint_load = np.zeros([batch_size, n_timesteps, 2])

        for i in range(batch_size):

            # track condition
            is_catch_trial = catch_trial[i] > 0.
            is_condition_hold = condition_hold[i] > 0.

            # define time cues
            if validation:
                go_cue_time = 10
                perturbation_time = go_cue_time + self.network.visual_delay + 7
                # if is_condition_hold:
                #     go_cue_time = int(self.training_n_timesteps / 2)
                #     perturbation_time = 10 + self.network.visual_delay + 7
                # else:
                #     go_cue_time = 10
                #     perturbation_time = go_cue_time + self.network.visual_delay + 7
            else:
                go_cue_time = int(np.random.uniform(0, self.training_n_timesteps))
                perturbation_time = int(np.random.uniform(0, self.training_n_timesteps))

            # compute perturbation forces
            # if validation:
            #     start = np.array(center[i, :2]).reshape((1, 2))
            #     end = np.array(targets[i, -1, :2]).reshape((1, 2))
            # else:
            #     a = np.random.uniform(0., np.pi * 2)
            #     start = np.zeros((1, 2))
            #     end = np.stack([np.cos(a), np.sin(a)], axis=-1).reshape((1, 2))
            if is_condition_hold:
                perturbation[i] = 2. * perturbation[i]
            start = np.array(center[i, :2]).reshape((1, 2))
            end = np.array(targets[i, -1, :2]).reshape((1, 2))
            force = perturbation[i] * self.orthogonal_perturbation(start, end)

            # compute targets
            if is_catch_trial or is_condition_hold:
                targets[i, :, :] = center[i, np.newaxis, :]
            else:
                targets[i, :go_cue_time, :] = center[i, np.newaxis, :]
                inputs_start[i, go_cue_time + self.network.visual_delay:, :] = 0.
                go_cue[i, go_cue_time + self.network.visual_delay:, 0] = 0.

            endpoint_load[i, perturbation_time:perturbation_time+9, :] = force[:, np.newaxis, :]

        inputs = {
            "inputs": np.concatenate([inputs_start, inputs_targ, go_cue], axis=-1),
            "endpoint_load": endpoint_load,
            "perturbation": np.tile(np.array(perturbation).reshape((-1, 1, 1)), [1, n_timesteps, 1])
        }
        return [inputs, self.convert_to_tensor(targets), init_states]

    @staticmethod
    def orthogonal_perturbation(start, end):
        a = end - start
        den = np.sqrt(np.sum((end - start) ** 2, axis=1))
        rot = a / den[:, np.newaxis]
        force = rot[:, ::-1] * [1, -1]
        return force


def build(verbose=0):
    plant = RigidTendonArm26(muscle_type=RigidTendonHillMuscleThelen(), proprioceptive_delay=0.02, visual_delay=0.05,
                             excitation_noise_sd=10 ** -4)
    cell = GRUNetwork(plant=plant, n_units=110, kernel_regularizer=10 ** -6, name='cell', hidden_noise_sd=10 ** -3)
    task = CustomTask(network=cell, start_position=[np.pi/4, np.pi/2])

    inputs = task.get_input_dict_layers()
    state0 = task.get_initial_state_layers()

    # wrap cell in an RNN layer
    rnn = tf.keras.layers.RNN(cell=cell, return_sequences=True, name='RNN')
    states_out = rnn(inputs, initial_state=state0)
    control_rnn = MotorNetModel(inputs=[inputs, state0], outputs=states_out, name='controller', task=task)

    # and compile
    control_rnn.compile(optimizer=tf.optimizers.Adam(clipnorm=1.), loss=task.losses, loss_weights=task.loss_weights)
    if verbose == 1:
        control_rnn.summary()
    return control_rnn


def create(active_dir, verbose=0):

    models_dir = os.path.join(active_dir, "models")
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)

    n_repeats = 1

    protocol_table = {
        "n_repeats": n_repeats,
        "activedir path": active_dir,
        "repeat #": [],
        "filename": [],
    }

    for r in range(n_repeats):
        nn = build(verbose=verbose)
        filename = "arm26" + "__r" + str(r)
        file = os.path.join(models_dir, filename)
        nn.save_model(file)
        protocol_table["repeat #"].append(r)
        protocol_table["filename"].append(filename)

    with open(os.path.join(active_dir, 'protocol_table.json'), 'w+') as f:
        json.dump(protocol_table, f)


if __name__ == '__main__':
    create(file_dir, verbose=1)
    print("\nModels built successfully.\n")