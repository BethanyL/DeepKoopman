import datetime
import pickle
import time

import numpy as np
import tensorflow as tf


def stack_data(data, num_shifts, len_time):
    nd = data.ndim
    if nd > 1:
        n = data.shape[1]
    else:
        data = (np.asmatrix(data)).getT()
        n = 1
    num_traj = data.shape[0] / len_time

    new_len_time = len_time - num_shifts

    data_tensor = np.zeros([num_shifts + 1, num_traj * new_len_time, n])

    for j in np.arange(num_shifts + 1):
        for count in np.arange(num_traj):
            data_tensor_range = np.arange(count * new_len_time, new_len_time + count * new_len_time)
            data_tensor[j, data_tensor_range, :] = data[count * len_time + j: count * len_time + j + new_len_time, :]

    return data_tensor


def choose_optimizer(params, regularized_loss, trainable_var):
    if params['opt_alg'] == 'adam':
        optimizer = tf.train.AdamOptimizer(params['learning_rate']).minimize(regularized_loss, var_list=trainable_var)
    elif params['opt_alg'] == 'adadelta':
        if params['decay_rate'] > 0:
            optimizer = tf.train.AdadeltaOptimizer(params['learning_rate'], params['decay_rate']).minimize(
                regularized_loss,
                var_list=trainable_var)
        else:
            # defaults 0.001, 0.95
            optimizer = tf.train.AdadeltaOptimizer(params['learning_rate']).minimize(regularized_loss,
                                                                                     var_list=trainable_var)
    elif params['opt_alg'] == 'adagrad':
        # also has initial_accumulator_value parameter
        optimizer = tf.train.AdagradOptimizer(params['learning_rate']).minimize(regularized_loss,
                                                                                var_list=trainable_var)
    elif params['opt_alg'] == 'adagradDA':
        # Be careful when using AdagradDA for deep networks as it will require careful initialization of the gradient
        # accumulators for it to train.
        optimizer = tf.train.AdagradDAOptimizer(params['learning_rate'], tf.get_global_step()).minimize(
            regularized_loss,
            var_list=trainable_var)
    elif params['opt_alg'] == 'ftrl':
        # lots of hyperparameters: learning_rate_power, initial_accumulator_value,
        # l1_regularization_strength, l2_regularization_strength
        optimizer = tf.train.FtrlOptimizer(params['learning_rate']).minimize(regularized_loss, var_list=trainable_var)
    elif params['opt_alg'] == 'proximalGD':
        # can have built-in reg.
        optimizer = tf.train.ProximalGradientDescentOptimizer(params['learning_rate']).minimize(regularized_loss,
                                                                                                var_list=trainable_var)
    elif params['opt_alg'] == 'proximalAdagrad':
        # initial_accumulator_value, reg.
        optimizer = tf.train.ProximalAdagradOptimizer(params['learning_rate']).minimize(regularized_loss,
                                                                                        var_list=trainable_var)
    elif params['opt_alg'] == 'RMS':
        # momentum, epsilon, centered (False/True)
        if params['decay_rate'] > 0:
            optimizer = tf.train.RMSPropOptimizer(params['learning_rate'], params['decay_rate']).minimize(
                regularized_loss,
                var_list=trainable_var)
        else:
            # default decay_rate 0.9
            optimizer = tf.train.RMSPropOptimizer(params['learning_rate']).minimize(regularized_loss,
                                                                                    var_list=trainable_var)
    else:
        raise ValueError("chose invalid opt_alg %s in params dict" % params['opt_alg'])
    return optimizer


def check_progress(start, best_error, params):
    finished = 0
    save_now = 0

    current_time = time.time()

    if not params['been5min']:
        # only check 5 min progress once
        if current_time - start > 5 * 60:
            if best_error > params['min_5min']:
                print("too slowly improving in first five minutes: err %.15f" % best_error)
                params['stop_condition'] = 'too slowly improving in first 5 min'
                finished = 1
                return finished, save_now
            else:
                print("been 5 minutes, err = %.15f < %.15f" % (best_error, params['min_5min']))
                params['been5min'] = 1
    if not params['been20min']:
        # only check 20 min progress once
        if current_time - start > 20 * 60:
            if best_error > params['min_20min']:
                print("too slowly improving in first 20 minutes: err %.15f" % best_error)
                params['stop_condition'] = 'too slowly improving in first 20 min'
                finished = 1
                return finished, save_now
            else:
                print("been 20 minutes, err = %.15f < %.15f" % (best_error, params['min_20min']))
                params['been20min'] = 1
    if not params['been40min']:
        # only check 40 min progress once
        if current_time - start > 40 * 60:
            if best_error > params['min_40min']:
                print("too slowly improving in first 40 minutes: err %.15f" % best_error)
                params['stop_condition'] = 'too slowly improving in first 40 min'
                finished = 1
                return finished, save_now
            else:
                print("been 40 minutes, err = %.15f < %.15f" % (best_error, params['min_40min']))
                params['been40min'] = 1
    if not params['been1hr']:
        # only check 1 hr progress once
        if current_time - start > 60 * 60:
            if best_error > params['min_1hr']:
                print("too slowly improving in first hour: err %.15f" % best_error)
                params['stop_condition'] = 'too slowly improving in first hour'
                finished = 1
                return finished, save_now
            else:
                print("been 1 hour, err = %.15f < %.15f" % (best_error, params['min_1hr']))
                save_now = 1
                params['been1hr'] = 1
    if not params['been2hr']:
        # only check 2 hr progress once
        if current_time - start > 2 * 60 * 60:
            if best_error > params['min_2hr']:
                print("too slowly improving in first two hours: err %.15f" % best_error)
                params['stop_condition'] = 'too slowly improving in first two hours'
                finished = 1
                return finished, save_now
            else:
                print("been 2 hours, err = %.15f < %.15f" % (best_error, params['min_2hr']))
                save_now = 1
                params['been2hr'] = 1
    if not params['been3hr']:
        # only check 3 hr progress once
        if current_time - start > 3 * 60 * 60:
            if best_error > params['min_3hr']:
                print("too slowly improving in first three hours: err %.15f" % best_error)
                params['stop_condition'] = 'too slowly improving in first three hours'
                finished = 1
                return finished, save_now
            else:
                print("been 3 hours, err = %.15f < %.15f" % (best_error, params['min_3hr']))
                save_now = 1
                params['been3hr'] = 1
    if not params['been4hr']:
        # only check 4 hr progress once
        if current_time - start > 4 * 60 * 60:
            if best_error > params['min_4hr']:
                print("too slowly improving in first four hours: err %.15f" % best_error)
                params['stop_condition'] = 'too slowly improving in first four hours'
                finished = 1
                return finished, save_now
            else:
                print("been 4 hours, err = %.15f < %.15f" % (best_error, params['min_4hr']))
                save_now = 1
                params['been4hr'] = 1

    if not params['beenHalf']:
        # only check halfway progress once
        if current_time - start > params['max_time'] / 2:
            if best_error > params['min_halfway']:
                print("too slowly improving 1/2 of way in: val err %.15f" % best_error)
                params['stop_condition'] = 'too slowly improving halfway in'
                finished = 1
                return finished, save_now
            else:
                print("Halfway through time, err = %.15f < %.15f" % (best_error, params['min_halfway']))
                params['beenHalf'] = 1

    if current_time - start > params['max_time']:
        params['stop_condition'] = 'past max time'
        finished = 1
        return finished, save_now

    return finished, save_now


def save_files(sess, saver, csv_path, train_val_error, params, weights, biases):
    np.savetxt(csv_path, train_val_error, delimiter=',')

    for key, value in weights.iteritems():
        np.savetxt(csv_path.replace('error', key), np.asarray(sess.run(value)), delimiter=',')
    for key, value in biases.iteritems():
        np.savetxt(csv_path.replace('error', key), np.asarray(sess.run(value)), delimiter=',')
    params['minTrain'] = np.min(train_val_error[:, 0])
    params['minTest'] = np.min(train_val_error[:, 1])
    params['minRegTrain'] = np.min(train_val_error[:, 2])
    params['minRegTest'] = np.min(train_val_error[:, 3])
    print "min train: %.12f, min val: %.12f, min reg. train: %.12f, min reg. val: %.12f" % (
        params['minTrain'], params['minTest'], params['minRegTrain'], params['minRegTest'])
    save_params(params)


def save_params(params):
    with open(params['model_path'].replace('ckpt', 'pkl'), 'wb') as f:
        pickle.dump(params, f, pickle.HIGHEST_PROTOCOL)


def set_defaults(params):
    # defaults related to dataset
    if 'data_name' not in params:
        raise KeyError("Error: must give data_name as input to main")
    if 'len_time' not in params:
        raise KeyError("Error, must give len_time as input to main")
    if 'data_train_len' not in params:
        raise KeyError("Error, must give data_train_len as input to main")
    if 'delta_t' not in params:
        raise KeyError("Error, must give delta_t as input to main")

    # defaults related to saving results
    if 'folder_name' not in params:
        print("setting default: using folder named 'results'")
        params['folder_name'] = 'results'
    if 'exp_suffix' not in params:
        print("setting default name of experiment")
        params['exp_suffix'] = '_' + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    if 'model_path' not in params:
        print("setting default path for model")
        exp_name = params['data_name'] + params['exp_suffix']
        params['model_path'] = "./%s/%s_model.ckpt" % (params['folder_name'], exp_name)

    # defaults related to network architecture
    if 'widths' not in params:
        raise KeyError("Error, must give widths as input to main")
    if 'widths_omega' not in params:
        raise KeyError("Error, must give widths for omega net")
    if 'act_type' not in params:
        print("setting default: activation function is ReLU")
        params['act_type'] = 'relu'
    if 'batch_flag' not in params:
        print("setting default: no batch normalization")
        params['batch_flag'] = 0

    params['d'] = len(params['widths'])  # d must be calculated like this
    params['do'] = len(params['widths_omega'])  # do must be calculated like this
    print params['widths']
    print params['widths_omega']

    # defaults related to initialization of parameters
    if 'dist_weights' not in params:
        print("setting default: distribution for weights on main net is dl")
        params['dist_weights'] = 'dl'
    if 'dist_weights_omega' not in params:
        print("setting default: distribution for weights on omega net is dl")
        params['dist_weights_omega'] = 'dl'
    if 'dist_biases' not in params:
        print("setting default: initialize biases for main net to 0")
        params['dist_biases'] = 0
    if 'dist_biases_omega' not in params:
        print("setting default: initialize biases for omega net to 0")
        params['dist_biases_omega'] = 0

    if 'first_guess' not in params:
        print("setting default: no first guess for main network")
        params['first_guess'] = 0
    if 'first_guess_omega' not in params:
        print("setting default: no first guess for omega net")
        params['first_guess_omega'] = 0

    if 'scale' not in params:
        print("setting default: scale for weights in main net is 0.1 (applies to tn distribution)")
        params['scale'] = 0.1
    if 'scale_omega' not in params:
        print("setting default: scale for weights in omega net is 0.1 (applies to tn distribution)")
        params['scale_omega'] = 0.1

    if isinstance(params['dist_weights'], basestring):
        params['dist_weights'] = [params['dist_weights']] * (len(params['widths']) - 1)
    if isinstance(params['dist_biases'], int):
        params['dist_biases'] = [params['dist_biases']] * (len(params['widths']) - 1)
    if isinstance(params['dist_weights_omega'], basestring):
        params['dist_weights_omega'] = [params['dist_weights_omega']] * (len(params['widths_omega']) - 1)
    if isinstance(params['dist_biases_omega'], int):
        params['dist_biases_omega'] = [params['dist_biases_omega']] * (len(params['widths_omega']) - 1)

    # defaults related to loss function
    if 'relative_loss' not in params:
        print("setting default: loss is not relative")
        params['relative_loss'] = 0

    if 'shifts' not in params:
        print("setting default: penalty on all shifts from 1 to num_shifts")
        params['shifts'] = np.arange(params['num_shifts']) + 1
    if 'shifts_middle' not in params:
        print("setting default: penalty on all middle shifts from 1 to num_shifts_middle")
        params['shifts_middle'] = np.arange(params['num_shifts_middle']) + 1
    params['num_shifts'] = len(params['shifts'])  # must be calculated like this
    params['num_shifts_middle'] = len(params['shifts_middle'])  # must be calculated like this

    if 'recon_lam' not in params:
        print("setting default: weight on reconstruction is 1.0")
        params['recon_lam'] = 1.0
    if 'mid_shift_lam' not in params:
        print("setting default: weight on loss3 is 1.0")
        params['mid_shift_lam'] = 1.0
    if 'L1_lam' not in params:
        print("setting default: L1_lam is .00001")
        params['L1_lam'] = .00001
    if 'L2_lam' not in params:
        print("setting default: no L2 regularization")
        params['L2_lam'] = 0.0
    if 'Linf_lam' not in params:
        print("setting default: no L_inf penalty")
        params['Linf_lam'] = 0.0

    # defaults related to training
    if 'num_passes_per_file' not in params:
        print("setting default: 1000 passes per training file")
        params['num_passes_per_file'] = 1000
    if 'num_steps_per_batch' not in params:
        print("setting default: 1 step per batch before moving to next training file")
        params['num_steps_per_batch'] = 1
    if 'num_steps_per_file_pass' not in params:
        print("setting default: up to 1000000 steps per training file before moving to next one")
        params['num_steps_per_file_pass'] = 1000000
    if 'learning_rate' not in params:
        print("setting default learning rate")
        params['learning_rate'] = .003
    if 'opt_alg' not in params:
        print("setting default: use Adam optimizer")
        params['opt_alg'] = 'adam'
    if 'decay_rate' not in params:
        print("setting default: decay_rate is 0 (applies to some optimizer algorithms)")
        params['decay_rate'] = 0
    if 'batch_size' not in params:
        print("setting default: no batches (use whole training file at once)")
        params['batch_size'] = 0
    if 'dropout_rate' not in params:
        print("setting default: no dropout")
        params['dropout_rate'] = 1.0

    # setting defaults related to keeping track of training time and progress
    if 'max_time' not in params:
        print("setting default: run up to 6 hours")
        params['max_time'] = 6 * 60 * 60  # 6 hours
    if 'min_5min' not in params:
        params['min_5min'] = 10 ** (-2)
        print("setting default: must reach %f in 5 minutes" % params['min_5min'])
    if 'min_20min' not in params:
        params['min_20min'] = 10 ** (-3)
        print("setting default: must reach %f in 20 minutes" % params['min_20min'])
    if 'min_40min' not in params:
        params['min_40min'] = 10 ** (-4)
        print("setting default: must reach %f in 40 minutes" % params['min_40min'])
    if 'min_1hr' not in params:
        params['min_1hr'] = 10 ** (-5)
        print("setting default: must reach %f in 1 hour" % params['min_1hr'])
    if 'min_2hr' not in params:
        params['min_2hr'] = 10 ** (-5.25)
        print("setting default: must reach %f in 2 hours" % params['min_2hr'])
    if 'min_3hr' not in params:
        params['min_3hr'] = 10 ** (-5.5)
        print("setting default: must reach %f in 3 hours" % params['min_3hr'])
    if 'min_4hr' not in params:
        params['min_4hr'] = 10 ** (-5.75)
        print("setting default: must reach %f in 4 hours" % params['min_4hr'])
    if 'min_halfway' not in params:
        params['min_halfway'] = 10 ** (-4)
        print("setting default: must reach %f in first half of time allotted" % params['min_halfway'])

    # initializing trackers for how long the training has run
    params['been5min'] = 0
    params['been20min'] = 0
    params['been40min'] = 0
    params['been1hr'] = 0
    params['been2hr'] = 0
    params['been3hr'] = 0
    params['been4hr'] = 0
    params['beenHalf'] = 0


def num_shifts_in_stack(params):
    max_shifts_to_stack = 1
    if params['num_shifts']:
        max_shifts_to_stack = max(max_shifts_to_stack, max(params['shifts']))
    if params['num_shifts_middle']:
        max_shifts_to_stack = max(max_shifts_to_stack, max(params['shifts_middle']))

    return max_shifts_to_stack
