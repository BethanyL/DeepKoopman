import os
import time

import numpy as np
import tensorflow as tf

import helperfns as h
import networkarch as net


def define_loss(x, y, g_list, g_list_omega, params):
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
    count_shifts_middle = 0
    if params['num_shifts_middle'] > 0:
        next_step = net.varying_multiply(g_list[0], g_list_omega[0], params['delta_t'])
        for j in np.arange(max(params['shifts_middle'])):
            if (j + 1) in params['shifts_middle']:
                # multiply g_list[0] by L (j+1) times
                # next_step = tf.matmul(g_list[0], L_pow)
                if params['relative_loss']:
                    loss3den = tf.reduce_mean(
                        tf.reduce_mean(tf.square(tf.squeeze(g_list[count_shifts_middle + 1])), 1)) + den_nonzero
                else:
                    loss3den = tf.to_double(1.0)
                loss3 = loss3 + params['mid_shift_lam'] * tf.truediv(
                    tf.reduce_mean(tf.reduce_mean(tf.square(next_step - g_list[count_shifts_middle + 1]), 1)), loss3den)
                count_shifts_middle += 1
            # hopefully still on correct traj, so same omegas as before
            next_step = net.varying_multiply(next_step, g_list_omega[j + 1], params['delta_t'])
        loss3 = loss3 / params['num_shifts_middle']

    # inf norm on autoencoder error
    Linf1_den = tf.norm(tf.norm(tf.squeeze(x[0, :, :]), axis=1, ord=np.inf), ord=np.inf) + den_nonzero
    Linf2_den = tf.norm(tf.norm(tf.squeeze(x[1, :, :]), axis=1, ord=np.inf), ord=np.inf) + den_nonzero
    Linf1_penalty = tf.truediv(
        tf.norm(tf.norm(y[0] - tf.squeeze(x[0, :, :]), axis=1, ord=np.inf), ord=np.inf), Linf1_den)
    Linf2_penalty = tf.truediv(
        tf.norm(tf.norm(y[1] - tf.squeeze(x[1, :, :]), axis=1, ord=np.inf), ord=np.inf), Linf2_den)
    loss_Linf = params['Linf_lam'] * (Linf1_penalty + Linf2_penalty)

    loss = loss1 + loss2 + loss3 + loss_Linf

    return loss1, loss2, loss3, loss_Linf, loss


def define_regularization(params, trainable_var, loss):
    if params['L1_lam']:
        l1_regularizer = tf.contrib.layers.l1_regularizer(scale=params['L1_lam'], scope=None)
        # TODO: don't include biases? use weights dict instead?
        loss_L1 = tf.contrib.layers.apply_regularization(l1_regularizer, weights_list=trainable_var)
    else:
        loss_L1 = tf.zeros([1, ], dtype=tf.float64)

    # tf.nn.l2_loss returns number
    l2_regularizer = tf.add_n([tf.nn.l2_loss(v) for v in trainable_var if 'b' not in v.name])
    loss_L2 = params['L2_lam'] * l2_regularizer

    regularized_loss = loss + loss_L1 + loss_L2

    return loss_L1, loss_L2, regularized_loss


def try_net(data_train_len, data_val, params):
    # SET UP NETWORK
    phase = tf.placeholder(tf.bool, name='phase')
    keep_prob = tf.placeholder(tf.float64, shape=[], name='keep_prob')
    x, y, g_list, weights, biases, g_list_omega = net.create_koopman_net(phase, keep_prob, params)

    max_shifts_to_stack = 1
    if len(params['shifts']):
        max_shifts_to_stack = max(max_shifts_to_stack, max(params['shifts']))
    if len(params['shifts_middle']):
        max_shifts_to_stack = max(max_shifts_to_stack, max(params['shifts_middle']))

    # DEFINE LOSS FUNCTION
    trainable_var = tf.trainable_variables()
    loss1, loss2, loss3, loss_Linf, loss = define_loss(x, y, g_list, g_list_omega, params)
    loss_L1, loss_L2, regularized_loss = define_regularization(params, trainable_var, loss)

    # CHOOSE OPTIMIZATION ALGORITHM
    optimizer = h.choose_optimizer(params, regularized_loss, trainable_var)

    # LAUNCH GRAPH AND INITIALIZE
    sess = tf.Session()
    saver = tf.train.Saver()

    init = tf.global_variables_initializer()
    sess.run(init)

    csv_path = params['model_path'].replace('model', 'error')
    csv_path = csv_path.replace('ckpt', 'csv')
    print csv_path

    num_saved = np.floor(
        (params['num_steps_per_file_pass'] / 20 + 1) * data_train_len * params['num_passes_per_file']).astype(int)
    train_val_error = np.zeros([num_saved, 16])
    count = 0
    best_error = 10000

    data_val_tensor = h.stack_data(data_val, max_shifts_to_stack, params['len_time'])

    start = time.time()
    finished = 0
    saver.save(sess, params['model_path'])

    # TRAINING
    # loop over training data files
    for f in range(data_train_len * params['num_passes_per_file']):
        if finished:
            break
        file_num = (f % data_train_len) + 1  # 1...data_train_len

        if (data_train_len > 1) or (f == 0):
            # don't keep reloading data if always same
            data_train = np.loadtxt(('./data/%s_train%d_x.csv' % (params['data_name'], file_num)), delimiter=',')
            data_train_tensor = h.stack_data(data_train, max_shifts_to_stack, params['len_time'])
            num_examples = data_train_tensor.shape[1]
            ind = np.arange(num_examples)
            np.random.shuffle(ind)
            data_train_tensor = data_train_tensor[:, ind, :]

            num_batches = int(np.floor(num_examples / params['batch_size']))

        # loop over batches in this file
        for step in range(params['num_steps_per_batch'] * num_batches):

            if params['batch_size'] < data_train_tensor.shape[1]:
                offset = (step * params['batch_size']) % (num_examples - params['batch_size'])
            else:
                offset = 0

            batch_data_train = data_train_tensor[:, offset:(offset + params['batch_size']), :]

            feed_dict_train = {x: batch_data_train, phase: 1, keep_prob: params['dropout_rate']}
            feed_dict_train_loss = {x: batch_data_train, phase: 1, keep_prob: 1.0}
            feed_dict_val = {x: data_val_tensor, phase: 0, keep_prob: 1.0}

            sess.run(optimizer, feed_dict=feed_dict_train)

            if step % 20 == 0:
                train_error = sess.run(loss, feed_dict=feed_dict_train_loss)
                val_error = sess.run(loss, feed_dict=feed_dict_val)

                if val_error < (best_error - best_error * (10 ** (-5))):
                    best_error = val_error.copy()
                    saver.save(sess, params['model_path'])
                    print("New best val error %f" % best_error)

                train_val_error[count, 0] = train_error
                train_val_error[count, 1] = val_error
                train_val_error[count, 2] = sess.run(regularized_loss, feed_dict=feed_dict_train_loss)
                train_val_error[count, 3] = sess.run(regularized_loss, feed_dict=feed_dict_val)
                train_val_error[count, 4] = sess.run(loss1, feed_dict=feed_dict_train_loss)
                train_val_error[count, 5] = sess.run(loss1, feed_dict=feed_dict_val)
                train_val_error[count, 6] = sess.run(loss2, feed_dict=feed_dict_train_loss)
                train_val_error[count, 7] = sess.run(loss2, feed_dict=feed_dict_val)
                train_val_error[count, 8] = sess.run(loss3, feed_dict=feed_dict_train_loss)
                train_val_error[count, 9] = sess.run(loss3, feed_dict=feed_dict_val)
                train_val_error[count, 10] = sess.run(loss_Linf, feed_dict=feed_dict_train_loss)
                train_val_error[count, 11] = sess.run(loss_Linf, feed_dict=feed_dict_val)
                train_val_error[count, 12] = sess.run(loss_L1, feed_dict=feed_dict_train_loss)
                train_val_error[count, 13] = sess.run(loss_L1, feed_dict=feed_dict_val)
                train_val_error[count, 14] = sess.run(loss_L2, feed_dict=feed_dict_train_loss)
                train_val_error[count, 15] = sess.run(loss_L2, feed_dict=feed_dict_val)

                np.savetxt(csv_path, train_val_error, delimiter=',')
                finished, save_now = h.check_progress(start, best_error, params)
                if save_now:
                    train_val_error_trunc = train_val_error[range(count), :]
                    h.save_files(sess, saver, csv_path, train_val_error_trunc, params, weights, biases, 0)
                if finished:
                    break
                count = count + 1

            if step > params['num_steps_per_file_pass']:
                params['stop_condition'] = 'reached num_steps_per_file_pass'
                break

    # SAVE RESULTS
    train_val_error = train_val_error[range(count), :]
    print(train_val_error)
    params['time_exp'] = time.time() - start
    h.save_files(sess, saver, csv_path, train_val_error, params, weights, biases, 1)
    return np.min(train_val_error[:, 0])


def main_exp(params):
    h.set_defaults(params)

    if not os.path.exists(params['folder_name']):
        os.makedirs(params['folder_name'])

    # data is num_steps x num_examples x n
    data_val = np.genfromtxt(('./data/%s_val_x.csv' % (params['data_name'])), delimiter=',')
    err = try_net(params['data_train_len'], data_val, params)
    tf.reset_default_graph()
    return err
