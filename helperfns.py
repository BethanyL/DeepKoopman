import tensorflow as tf
import numpy as np
import os
import time
import datetime
import pickle
import networkarch as net


def stack_data(data, num_shifts, lenT):
    nd = data.ndim
    if nd > 1:
        n = data.shape[1]
    else:
        data = (np.asmatrix(data)).getT()
        n = 1
    num_traj = data.shape[0] / lenT

    newLenT = lenT - num_shifts

    data_tensor = np.zeros([num_shifts + 1, num_traj * newLenT, n])

    for j in np.arange(num_shifts + 1):
        for count in np.arange(num_traj):
            data_tensor[j, count * newLenT: newLenT + count * newLenT, :] = data[
                                                                            count * lenT + j: count * lenT + j + newLenT,
                                                                            :]

    return data_tensor


def ChooseOptimizer(params, regularized_loss, trainable_var):
    if params['optalg'] == 'adam':
        optimizer = tf.train.AdamOptimizer(params['lr']).minimize(regularized_loss, var_list=trainable_var)
    elif params['optalg'] == 'adadelta':
        if params['decay_rate'] > 0:
            optimizer = tf.train.AdadeltaOptimizer(params['lr'], params['decay_rate']).minimize(regularized_loss,
                                                                                                var_list=trainable_var)
        else:
            # defaults 0.001, 0.95
            optimizer = tf.train.AdadeltaOptimizer(params['lr']).minimize(regularized_loss, var_list=trainable_var)
    elif params['optalg'] == 'adagrad':
        # also has initial_accumulateor_value parameter
        optimizer = tf.train.AdagradOptimizer(params['lr']).minimize(regularized_loss, var_list=trainable_var)
    elif params['optalg'] == 'adagradDA':
        # Be careful when using AdagradDA for deep networks as it will require careful initialization of the gradient accumulators for it to train.
        optimizer = tf.train.AdagradDAOptimizer(params['lr'], tf.get_global_step()).minimize(regularized_loss,
                                                                                             var_list=trainable_var)
    #	elif optalg == 'momentum':
    # need to pass in momentum
    #		optimizer = tf.train.MomentumOptimizer(params['lr']).minimize(regularized_loss)
    elif params['optalg'] == 'ftrl':
        # lots of hyperparameters: learning_rate_power, initial_accumulator_value,
        # l1_regularization_strength, l2_regularization_strength
        optimizer = tf.train.FtrlOptimizer(params['lr']).minimize(regularized_loss, var_list=trainable_var)
    elif params['optalg'] == 'proximalGD':
        # can have built-in reg.
        optimizer = tf.train.ProximalGradientDescentOptimizer(params['lr']).minimize(regularized_loss,
                                                                                     var_list=trainable_var)
    elif params['optalg'] == 'proximalAdagrad':
        # initial_accumulator_value, reg.
        optimizer = tf.train.ProximalAdagradOptimizer(params['lr']).minimize(regularized_loss, var_list=trainable_var)
    elif params['optalg'] == 'RMS':
        # momentum, epsilon, centered (False/True)
        if params['decay_rate'] > 0:
            optimizer = tf.train.RMSPropOptimizer(params['lr'], params['decay_rate']).minimize(regularized_loss,
                                                                                               var_list=trainable_var)
        else:
            # default decay_rate 0.9
            optimizer = tf.train.RMSPropOptimizer(params['lr']).minimize(regularized_loss, var_list=trainable_var)
    else:
        print "chose invalid optalg %s" % params['optalg']
    return optimizer


def DefineLoss(x, y, g_list, g_list_omega, params, trainable_var):
    # Minimize the mean squared errors.
    # subtraction and squaring element-wise, then average over both dimensions
    # n columns
    # average of each row (across columns), then average the rows
    den_nonzero = 10 ** (-5)

    # autoencoder loss
    if params['relative_loss']:
        loss1den = tf.reduce_mean(tf.reduce_mean(tf.square(tf.squeeze(x[0, :, :])), 1)) + den_nonzero
    else:
        loss1den = tf.to_double(1.0)
    loss1 = params['recon_lam'] * tf.truediv(
        tf.reduce_mean(tf.reduce_mean(tf.square(y[0] - tf.squeeze(x[0, :, :])), 1)), loss1den)

    # gets dynamics
    loss2 = tf.zeros([1, ], dtype=tf.float64)
    if params['num_shifts'] > 0:
        for j in np.arange(params['num_shifts']):
            # xk+1, xk+2, xk+3
            shift = params['shifts'][j]
            if params['relative_loss']:
                loss2den = tf.reduce_mean(tf.reduce_mean(tf.square(tf.squeeze(x[shift, :, :])), 1)) + den_nonzero
            else:
                loss2den = tf.to_double(1.0)
            loss2 = loss2 + params['recon_lam'] * tf.truediv(
                tf.reduce_mean(tf.reduce_mean(tf.square(y[j + 1] - tf.squeeze(x[shift, :, :])), 1)), loss2den)
        loss2 = loss2 / params['num_shifts']

    # K linear
    loss3 = tf.zeros([1, ], dtype=tf.float64)
    countSM = 0
    if params['num_shifts_middle'] > 0:
        next_step = net.varying_multiply(g_list[0], g_list_omega[0], params['deltat'])
        for j in np.arange(max(params['shifts_middle'])):
            if ((j + 1) in params['shifts_middle']):
                # muliply g_list[0] by L (j+1) times
                # next_step = tf.matmul(g_list[0], L_pow)
                if params['relative_loss']:
                    loss3den = tf.reduce_mean(
                        tf.reduce_mean(tf.square(tf.squeeze(g_list[countSM + 1])), 1)) + den_nonzero
                else:
                    loss3den = tf.to_double(1.0)
                loss3 = loss3 + params['mid_shift_lam'] * tf.truediv(
                    tf.reduce_mean(tf.reduce_mean(tf.square(next_step - g_list[countSM + 1]), 1)), loss3den)
                countSM += 1
            # hopefully still on correct traj, so same omegas as before
            next_step = net.varying_multiply(next_step, g_list_omega[j + 1], params['deltat'])
        loss3 = loss3 / params['num_shifts_middle']

    if params['reg_lam']:
        l1_regularizer = tf.contrib.layers.l1_regularizer(scale=params['reg_lam'], scope=None)
        # TODO: don't include biases? use weights dict instead?
        loss_L1 = tf.contrib.layers.apply_regularization(l1_regularizer, weights_list=trainable_var)
    else:
        loss_L1 = tf.zeros([1, ], dtype=tf.float64)

    # tf.nn.l2_loss returns number
    l2_regularizer = tf.add_n([tf.nn.l2_loss(v) for v in trainable_var if 'b' not in v.name])
    loss_L2 = params['l2_lam'] * l2_regularizer

    # inf norm on autoencoder error
    linf1_den = tf.norm(tf.norm(tf.squeeze(x[0, :, :]), axis=1, ord=np.inf), ord=np.inf) + den_nonzero
    linf2_den = tf.norm(tf.norm(tf.squeeze(x[1, :, :]), axis=1, ord=np.inf), ord=np.inf) + den_nonzero
    regularization_penalty_Linf_1 = tf.truediv(
        tf.norm(tf.norm(y[0] - tf.squeeze(x[0, :, :]), axis=1, ord=np.inf), ord=np.inf), linf1_den)
    regularization_penalty_Linf_2 = tf.truediv(
        tf.norm(tf.norm(y[1] - tf.squeeze(x[1, :, :]), axis=1, ord=np.inf), ord=np.inf), linf2_den)
    loss_Linf = params['reg_inf_lam'] * (regularization_penalty_Linf_1 + regularization_penalty_Linf_2)

    loss = loss1 + loss2 + loss3 + loss_Linf
    regularized_loss = loss + loss_L1 + loss_L2

    return loss1, loss2, loss3, loss_Linf, loss_L1, loss_L2, loss, regularized_loss


def CheckProgress(start, start_file, bestErr, timeBestErr, params):
    finished = 0
    saveNow = 0

    currentTime = time.time()
    if (currentTime - max(timeBestErr, start_file) > 5 * 60):
        print("too long since last improvement")
        params['stop_condition'] = 'too long since last improvement'
        finished = 1
        return finished, saveNow

    if not params['been5min']:
        # only check 5 min progress once
        if (currentTime - start > 5 * 60):
            if bestErr > params['min5min']:
                print("too slowly improving in first five minutes: err %.15f" % bestErr)
                params['stop_condition'] = 'too slowly improving in first 5 min'
                finished = 1
                return finished, saveNow
            else:
                print("been 5 minutes, err = %.15f < %.15f" % (bestErr, params['min5min']))
                params['been5min'] = 1
    if not params['been20min']:
        # only check 20 min progress once
        if (currentTime - start > 20 * 60):
            if bestErr > params['min20min']:
                print("too slowly improving in first 20 minutes: err %.15f" % bestErr)
                params['stop_condition'] = 'too slowly improving in first 20 min'
                finished = 1
                return finished, saveNow
            else:
                print("been 20 minutes, err = %.15f < %.15f" % (bestErr, params['min20min']))
                params['been20min'] = 1
    if not params['been40min']:
        # only check 40 min progress once
        if (currentTime - start > 40 * 60):
            if bestErr > params['min40min']:
                print("too slowly improving in first 40 minutes: err %.15f" % bestErr)
                params['stop_condition'] = 'too slowly improving in first 40 min'
                finished = 1
                return finished, saveNow
            else:
                print("been 40 minutes, err = %.15f < %.15f" % (bestErr, params['min40min']))
                params['been40min'] = 1
    if not params['been1hr']:
        # only check 1 hr progress once
        if (currentTime - start > 60 * 60):
            if bestErr > params['min1hr']:
                print("too slowly improving in first hour: err %.15f" % bestErr)
                params['stop_condition'] = 'too slowly improving in first hour'
                finished = 1
                return finished, saveNow
            else:
                print("been 1 hour, err = %.15f < %.15f" % (bestErr, params['min1hr']))
                saveNow = 1
                params['been1hr'] = 1
    if not params['been2hr']:
        # only check 2 hr progress once
        if (currentTime - start > 2 * 60 * 60):
            if bestErr > params['min2hr']:
                print("too slowly improving in first two hours: err %.15f" % bestErr)
                params['stop_condition'] = 'too slowly improving in first two hours'
                finished = 1
                return finished, saveNow
            else:
                print("been 2 hours, err = %.15f < %.15f" % (bestErr, params['min2hr']))
                saveNow = 1
                params['been2hr'] = 1
    if not params['been3hr']:
        # only check 3 hr progress once
        if (currentTime - start > 3 * 60 * 60):
            if bestErr > params['min3hr']:
                print("too slowly improving in first three hours: err %.15f" % bestErr)
                params['stop_condition'] = 'too slowly improving in first three hours'
                finished = 1
                return finished, saveNow
            else:
                print("been 3 hours, err = %.15f < %.15f" % (bestErr, params['min3hr']))
                saveNow = 1
                params['been3hr'] = 1
    if not params['been4hr']:
        # only check 4 hr progress once
        if (currentTime - start > 4 * 60 * 60):
            if bestErr > params['min4hr']:
                print("too slowly improving in first four hours: err %.15f" % bestErr)
                params['stop_condition'] = 'too slowly improving in first four hours'
                finished = 1
                return finished, saveNow
            else:
                print("been 4 hours, err = %.15f < %.15f" % (bestErr, params['min4hr']))
                saveNow = 1
                params['been4hr'] = 1

    if not params['beenHalf']:
        # only check halfway progress once
        if (currentTime - start > params['max_time'] / 2):
            if bestErr > params['minHalfway']:
                print("too slowly improving 1/2 of way in: test err %.15f" % bestErr)
                params['stop_condition'] = 'too slowly improving halfway in'
                finished = 1
                return finished, saveNow
            else:
                print("Halfway through time, err = %.15f < %.15f" % (bestErr, params['minHalfway']))
                params['beenHalf'] = 1

    if (currentTime - start > params['max_time']):
        params['stop_condition'] = 'past max time'
        finished = 1
        return finished, saveNow

    return finished, saveNow


def TryNet(n, data_train_len, data_test, exp_name, params):
    ## SET UP NETWORK
    phase = tf.placeholder(tf.bool, name='phase')
    keep_prob = tf.placeholder(tf.float64, shape=[], name='keep_prob')
    x, y, g_list, weights, biases, g_list_omega = net.CreateKoopmanNet(n, phase, keep_prob, params)

    max_shifts_to_stack = 1
    if len(params['shifts']):
        max_shifts_to_stack = max(max_shifts_to_stack, max(params['shifts']))
    if len(params['shifts_middle']):
        max_shifts_to_stack = max(max_shifts_to_stack, max(params['shifts_middle']))

    ## DEFINE LOSS FUNCTION
    trainable_var = tf.trainable_variables()
    loss1, loss2, loss3, loss_Linf, loss_L1, loss_L2, loss, regularized_loss = DefineLoss(x, y, g_list, g_list_omega,
                                                                                          params, trainable_var)

    ## CHOOSE OPTIMIZATION ALGORITHM
    optimizer = ChooseOptimizer(params, regularized_loss, trainable_var)

    ## LAUNCH GRAPH AND INITIALIZE
    sess = tf.Session()
    saver = tf.train.Saver()

    # Before starting, initialize the variables.  We will 'run' this first.
    if not params['modelpath']:
        params['modelpath'] = "./%s/%s_model.ckpt" % (params['foldername'], exp_name)
    init = tf.global_variables_initializer()
    sess.run(init)

    csvpath = params['modelpath'].replace('model', 'error')
    csvpath = csvpath.replace('ckpt', 'csv')
    print csvpath

    num_saved = np.floor(
        (params['num_steps_per_file_pass'] / 20 + 1) * data_train_len * params['num_passes_per_file']).astype(int)
    train_test_error = np.zeros([num_saved, 16])
    count = 0
    bestErr = 10000
    bestErrB = 10000
    timeBestErr = time.time()

    data_test_tensor = stack_data(data_test, max_shifts_to_stack, params['lenT'])

    if params['max_time']:
        start = time.time()
    finished = 0
    save_path = saver.save(sess, params['modelpath'])

    flagLossOnlyPart = 0
    ## TRAINING
    # loop over training data files
    for f in range(data_train_len * params['num_passes_per_file']):
        if finished:
            break
        filenum = (f % data_train_len) + 1  # 1...data_train_len

        if (data_train_len > 1) or (f == 0):
            # don't keep reloading data if always same
            data_train = np.loadtxt(('./data/%s_train%d_x.csv' % (params['data_name'], filenum)), delimiter=',')
            data_train_tensor = stack_data(data_train, max_shifts_to_stack, params['lenT'])
            num_examples = data_train_tensor.shape[1]
            ind = np.arange(num_examples)
            np.random.shuffle(ind)
            data_train_tensor = data_train_tensor[:, ind, :]

            num_batches = int(np.floor(num_examples / params['batch_size']))
        start_file = time.time()

        # loop over batches in this file
        for step in range(params['num_steps_per_batch'] * num_batches):

            if params['batch_size'] < data_train_tensor.shape[1]:
                offset = (step * params['batch_size']) % (num_examples - params['batch_size'])
            else:
                offset = 0

            batch_data_train = data_train_tensor[:, offset:(offset + params['batch_size']), :]

            feed_dict_train = {x: batch_data_train, phase: 1, keep_prob: params['dropout_rate']}
            feed_dict_train_loss = {x: batch_data_train, phase: 1, keep_prob: 1.0}
            feed_dict_test = {x: data_test_tensor, phase: 0, keep_prob: 1.0}

            sess.run(optimizer, feed_dict=feed_dict_train)

            if step % 20 == 0:
                trainerr = sess.run(loss, feed_dict=feed_dict_train_loss)
                testerr = sess.run(loss, feed_dict=feed_dict_test)

                if (testerr < (bestErr - bestErr * (10 ** (-5)))):
                    saveNewLoss = 1
                    if saveNewLoss:
                        bestErr = testerr.copy()
                        save_path = saver.save(sess, params['modelpath'])
                        timeBestErr = time.time()
                        print("New best test error %f" % bestErr)

                train_test_error[count, 0] = trainerr
                train_test_error[count, 1] = testerr
                train_test_error[count, 2] = sess.run(regularized_loss, feed_dict=feed_dict_train_loss)
                train_test_error[count, 3] = sess.run(regularized_loss, feed_dict=feed_dict_test)
                train_test_error[count, 4] = sess.run(loss1, feed_dict=feed_dict_train_loss)
                train_test_error[count, 5] = sess.run(loss1, feed_dict=feed_dict_test)
                train_test_error[count, 6] = sess.run(loss2, feed_dict=feed_dict_train_loss)
                train_test_error[count, 7] = sess.run(loss2, feed_dict=feed_dict_test)
                train_test_error[count, 8] = sess.run(loss3, feed_dict=feed_dict_train_loss)
                train_test_error[count, 9] = sess.run(loss3, feed_dict=feed_dict_test)
                train_test_error[count, 10] = sess.run(loss_Linf, feed_dict=feed_dict_train_loss)
                train_test_error[count, 11] = sess.run(loss_Linf, feed_dict=feed_dict_test)
                train_test_error[count, 12] = sess.run(loss_L1, feed_dict=feed_dict_train_loss)
                train_test_error[count, 13] = sess.run(loss_L1, feed_dict=feed_dict_test)
                train_test_error[count, 14] = sess.run(loss_L2, feed_dict=feed_dict_train_loss)
                train_test_error[count, 15] = sess.run(loss_L2, feed_dict=feed_dict_test)

                np.savetxt(csvpath, train_test_error, delimiter=',')
                if params['max_time']:
                    finished, saveNow = CheckProgress(start, start_file, bestErr, timeBestErr, params)
                    if saveNow:
                        train_test_error_trunc = train_test_error[range(count), :]
                        savefiles(sess, saver, csvpath, train_test_error_trunc, params, weights, biases, 0)
                    if finished:
                        break
                count = count + 1

            if step > params['num_steps_per_file_pass']:
                params['stop_condition'] = 'reached num_steps_per_file_pass'
                break

    ## SAVE RESULTS
    train_test_error = train_test_error[range(count), :]
    print(train_test_error)
    params['time_exp'] = time.time() - start
    savefiles(sess, saver, csvpath, train_test_error, params, weights, biases, 1)
    return np.min(train_test_error[:, 0])


def savefiles(sess, saver, csvpath, train_test_error, params, weights, biases, endflag):
    np.savetxt(csvpath, train_test_error, delimiter=',')
    if endflag:
        saver.restore(sess, params['modelpath'])

    for key, value in weights.iteritems():
        try:
            np.savetxt(csvpath.replace('error', key), np.asarray(sess.run(value)), delimiter=',')
        except:
            np.savetxt(csvpath.replace('error', key), np.asarray(value), delimiter=',')
    for key, value in biases.iteritems():
        np.savetxt(csvpath.replace('error', key), np.asarray(sess.run(value)), delimiter=',')
    params['minTrain'] = np.min(train_test_error[:, 0])
    params['minTest'] = np.min(train_test_error[:, 1])
    params['minRegTrain'] = np.min(train_test_error[:, 2])
    params['minRegTest'] = np.min(train_test_error[:, 3])
    print "min train: %.12f, min test: %.12f, min reg. train: %.12f, min reg. test: %.12f" % (
    params['minTrain'], params['minTest'], params['minRegTrain'], params['minRegTest'])
    save_params(params)


def save_params(params):
    with open(params['modelpath'].replace('ckpt', 'pkl'), 'wb') as f:
        pickle.dump(params, f, pickle.HIGHEST_PROTOCOL)


def set_defaults(params):
    if 'relative_loss' not in params:
        params['relative_loss'] = 0
    if 'recon_lam' not in params:
        params['recon_lam'] = 1.0
    if 'first_guess_omega' not in params:
        params['first_guess_omega'] = 0
    if 'widths_omega' not in params:
        raise KeyError, "Error, must give widths for omega net"
    if 'distributionW_omega' not in params:
        params['distributionW_omega'] = 'tn'
    if 'distributionB_omega' not in params:
        params['distributionB_omega'] = 0
    if 'scale_omega' not in params:
        params['scale_omega'] = 0.1
    # if 'omega_lam' not in params:
    #	params['omega_lam'] = 1.0
    if 'shifts' not in params:
        params['shifts'] = np.arange(params['num_shifts']) + 1
    if 'shifts_middle' not in params:
        params['shifts_middle'] = np.arange(params['num_shifts_middle']) + 1
    if 'first_guess' not in params:
        params['first_guess'] = 0
    if 'num_passes_per_file' not in params:
        params['num_passes_per_file'] = 1000
    if 'num_steps_per_batch' not in params:
        params['num_steps_per_batch'] = 1
    if 'num_steps_per_file_pass' not in params:
        params['num_steps_per_file_pass'] = 1000000
    if 'data_name' not in params:
        raise KeyError, "Error: must give data_name as input to main"
    if 'exp_suffix' not in params:
        params['exp_suffix'] = '_' + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    if 'widths' not in params:
        raise KeyError, "Error, must give widths as input to main"
    if 'lr' not in params:
        params['lr'] = .003
    if 'd' not in params:
        params['d'] = len(params['widths'])
    if 'act_type' not in params:
        params['act_type'] = 'relu'
    # leaving out xname and yname
    if 'mid_act_type' not in params:
        params['mid_act_type'] = params['act_type']
    if 'foldername' not in params:
        params['foldername'] = 'results'
    if 'reg_lam' not in params:
        params['reg_lam'] = .00001
    if 'reg_inf_lam' not in params:
        params['reg_inf_lam'] = 0.0
    if 'w' not in params:
        params['w'] = []
    if 'distributionW' not in params:
        params['distributionW'] = 'tn'
    if 'distributionB' not in params:
        params['distributionB'] = 0
    if 'scale' not in params:
        params['scale'] = 0.1
    if 'batch_flag' not in params:
        params['batch_flag'] = 0
    if 'optalg' not in params:
        params['optalg'] = 'adam'
    if 'decay_rate' not in params:
        params['decay_rate'] = 0
    if 'num_shifts' not in params:
        params['num_shifts'] = len(params['shifts'])
    if 'num_shifts_middle' not in params:
        params['num_shifts_middle'] = len(params['shifts_middle'])
    if 'batch_size' not in params:
        params['batch_size'] = 0
    if 'lenT' not in params:
        raise KeyError, "Error, must give lenT as input to main"
    if 'max_time' not in params:
        params['max_time'] = 0
    if 'l2_lam' not in params:
        params['l2_lam'] = 0.0
    if 'dropout_rate' not in params:
        params['dropout_rate'] = 1.0
    if 'modelpath' not in params:
        params['modelpath'] = 0
    if 'min5min' not in params:
        params['min5min'] = 10 ** (-2)
    if 'min20min' not in params:
        params['min20min'] = 10 ** (-3)
    if 'min40min' not in params:
        params['min40min'] = 10 ** (-4)
    if 'min1hr' not in params:
        params['min1hr'] = 10 ** (-5)
    if 'min2hr' not in params:
        params['min2hr'] = 10 ** (-5.25)
    if 'min3hr' not in params:
        params['min3hr'] = 10 ** (-5.5)
    if 'min4hr' not in params:
        params['min4hr'] = 10 ** (-5.75)
    if 'minHalfway' not in params:
        params['minHalfway'] = 10 ** (-4)
    if 'mid_shift_lam' not in params:
        params['mid_shift_lam'] = 1.0
    params['been5min'] = 0
    params['been20min'] = 0
    params['been40min'] = 0
    params['been1hr'] = 0
    params['been2hr'] = 0
    params['been3hr'] = 0
    params['been4hr'] = 0
    params['beenHalf'] = 0

    return params


def main_exp(params):
    # type: (object) -> object

    params = set_defaults(params)

    if not os.path.exists(params['foldername']):
        os.makedirs(params['foldername'])

    # Load x and y from files, generated by dyn system
    # want example x feature
    exp_name = params['data_name'] + params['exp_suffix']

    data_test = np.genfromtxt(('./data/%s_val_x.csv' % (params['data_name'])), delimiter=',')

    # data is num_steps x num_examples x n
    # n is width of input to network
    n = data_test.shape[1]
    if len(params['w']) == 0:
        params['w'] = np.ones(n)

    # can pass list of lists (widths for each layer) or list of ints (all layers have same width)
    if isinstance(params['widths'], int):
        m = params['widths']
        params['widths'] = np.repeat(m, params['d'])
    print params['widths']
    if isinstance(params['distributionW'], basestring):
        print("making list of distributions for weights W")
        params['distributionW'] = [params['distributionW']] * (len(params['widths']) - 1)
        print params['distributionW']
    if isinstance(params['distributionB'], int):
        print("making list of distributions for biases b")
        params['distributionB'] = [params['distributionB']] * (len(params['widths']) - 1)
        print params['distributionB']
    if isinstance(params['distributionW_omega'], basestring):
        print("making list of distributions for weights W in omega net")
        params['distributionW_omega'] = [params['distributionW_omega']] * (len(params['widths_omega']) - 1)
        print params['distributionW_omega']
    if isinstance(params['distributionB_omega'], int):
        print("making list of distributions for biases b in omega net")
        params['distributionB_omega'] = [params['distributionB_omega']] * (len(params['widths_omega']) - 1)
        print params['distributionB_omega']
    err = TryNet(n, params['data_train_len'], data_test, exp_name, params)

    tf.reset_default_graph()

    return err
