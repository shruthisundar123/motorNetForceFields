import os
import sys
import time
import json
import pickle
import numpy as np
from joblib import Parallel, delayed
from argparse import ArgumentParser

# -------------------------------
# Sort out arguments
# -------------------------------
parser = ArgumentParser()

help_d = "Full path of the directory hosting the pipeline data."
help_a = "Action to perform, 'train' or 'test'."
help_n = "Number of examples per epoch. If smaller than batch_size for training, then batch_size is used."
help_b = "[OPTIONAL, DEFAULT=32] Batch size."
help_e = "[OPTIONAL, DEFAULT=1] Number of epochs." +\
         "Each epoch re-generate new examples." +\
         "Ignored if action is not training."

parser.add_argument("-d", "--dir", dest="active_dir", required=True, help=help_d)
parser.add_argument("-a", "--action", dest="action", required=True, help=help_a)
parser.add_argument("-n", "--n_examples", dest="n_examples", required=True, type=int, help=help_n)
parser.add_argument("-b", "--batch_size", dest="batch_size", nargs='?', const=1, type=int, default=32, help=help_b)
parser.add_argument("-e", "--epochs", dest="epochs", nargs='?', const=1, type=int, default=1, help=help_e)

active_dir = parser.parse_args().active_dir
action = parser.parse_args().action
n_examples = parser.parse_args().n_examples
batch_size = parser.parse_args().batch_size
epochs = parser.parse_args().epochs

# find root directory and add to path
file_dir = sys.path[0]
root_dir = os.path.join(file_dir[:file_dir.find('MotorNet')], 'MotorNet')
sys.path.append(root_dir)
sys.path.append(active_dir)

if action != 'train' and action != 'test':
    raise ValueError('the action specified must be train or test')

models_dir = os.path.join(active_dir, "models")
weights_dir = os.path.join(active_dir, "weights")
data_dir = os.path.join(active_dir, "data")
log_dir = os.path.join(active_dir, "log")

run_list = [file[:-5] for file in os.listdir(models_dir) if (file.endswith(".json"))]
if not run_list:
    raise ValueError('No configuration files found')


def fn(run_iter):
    import tensorflow as tf
    from motornet.nets.callbacks import TensorflowFix, BatchLogger
    from build import build
    # print('tensorflow version: ' + tf.__version__)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    weight_subdir = os.path.join(weights_dir, run_list[run_iter])
    weight_file = os.path.join(weight_subdir, run_list[run_iter])
    log_file = os.path.join(log_dir, run_list[run_iter] + ".json")
    model_file = os.path.join(models_dir, run_list[run_iter] + ".json")
    protocol_file = os.path.join(active_dir, "protocol_table.json")

    with open(model_file, 'r') as file:
        cfg = json.load(file)

    control_rnn = build()
    task = control_rnn.task
    n_t = cfg['Task']['training_n_timesteps']

    tensorflowfix_callback = TensorflowFix()
    batchlog_callback = BatchLogger()

    # load trained weights if any
    if os.path.isfile(weight_file + '.h5'):
        control_rnn.load_weights(weight_file + '.h5')
    elif os.path.isfile(weight_file + '.index'):
        control_rnn.load_weights(weight_file).expect_partial()
    else:
        if not os.path.exists(weights_dir):
            os.mkdir(weights_dir)
        if not os.path.exists(weight_subdir):
            os.mkdir(weight_subdir)

    if action == 'train':

        n_batches = int(max(n_examples / batch_size, 1))
        task.set_training_params(batch_size=batch_size, n_timesteps=n_t)
        [inputs, targets, init_states] = task.generate(n_timesteps=n_t, batch_size=n_batches * batch_size)

        control_rnn.fit(x=[inputs, init_states],
                        y=targets,
                        verbose=1,
                        epochs=1,
                        batch_size=batch_size,
                        callbacks=[tensorflowfix_callback, batchlog_callback],
                        shuffle=False)

        # save weights of the model
        for file in os.listdir(weight_subdir):
            # side-step all previous save files and flag them with a "tmp-" prefix
            old = os.path.join(weight_subdir, file)
            new = os.path.join(weight_subdir, 'tmp-' + file)
            os.replace(src=old, dst=new)
        # create new save file
        control_rnn.save_weights(weight_file + '.h5', save_format='h5')
        # remove all old files that were "tmp-"-flagged above
        _ = [os.remove(os.path.join(weight_subdir, f)) for f in os.listdir(weight_subdir) if f.startswith('tmp-')]

        # add any info generated during the training process
        if os.path.isfile(log_file):
            with open(log_file, 'r') as file:
                training_log = json.load(file)
            for key, value in training_log.items():
                training_log[key] += batchlog_callback.history[key]
        else:
            if not os.path.exists(log_dir):
                os.mkdir(log_dir)
            training_log = batchlog_callback.history

        with open(log_file, 'w') as file:
            json.dump(training_log, file)

    elif action == 'test':

        [inputs, targets, init_states] = task.generate(n_timesteps=n_t, batch_size=n_examples, validation=True)
        print(inputs['inputs'].shape)

        inputs_np = {key: np.array(val) for key, val in inputs.items()}
        results = control_rnn([inputs, init_states], training=False)
        results_np = {key: np.array(val) for key, val in results.items()}

        # retrieve training history
        with open(log_file, 'r') as file:
            training_log = json.load(file)

        # save run as .mat file
        data_file = os.path.join(data_dir, run_list[run_iter] + ".pickle")
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)

        with open(data_file, 'wb') as file:
            pickle.dump(obj={'results': results_np,
                             'inputs': inputs_np,
                             'targets': targets.numpy(),
                             'weights': control_rnn.get_weights(),
                             'training_log': training_log,
                             'task': cfg['Task'],
                             'controller': cfg['Network'],
                             'plant': cfg['Plant'],
                             },
                        file=file)
            file.close()

if __name__ == '__main__':
    run_indexes = range(len(run_list))
    n_jobs = 4
    repeats = 1

    if action == 'train':
        repeats = epochs

    print(run_list)

    for i in range(repeats):
        print('repeat#' + str(i))
        run_backlog = run_indexes

        while len(run_backlog) > 0:
            these_runs = run_backlog[:n_jobs]
            run_backlog = run_backlog[n_jobs:]

            if len(these_runs) == 1:
                this_run = these_runs[0]
                fn(this_run)
            else:
                _ = Parallel(n_jobs=len(these_runs))(delayed(fn)(this_run) for this_run in these_runs)