import tensorflow as tf
import numpy as np


def weight_variable(shape, varname, distribution='tn', scale=0.1, first_guess=0):
    if distribution == 'tn':
        initial = tf.truncated_normal(shape, stddev=scale, dtype=tf.float64) + first_guess
    elif distribution == 'xavier':
        scale = 4 * np.sqrt(6.0 / (shape[0] + shape[1]))
        initial = tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float64)
    elif distribution == 'dl':
        # see page 295 of Goodfellow et al's DL book
        # divide by sqrt of m, where m is number of inputs
        scale = 1.0 / np.sqrt(shape[0])
        initial = tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float64)
    elif distribution == 'glorot_bengio':
        # see page 295 of Goodfellow et al's DL book
        scale = np.sqrt(6.0 / (shape[0] + shape[1]))
        initial = tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float64)
    else:
        initial = np.loadtxt(distribution, delimiter=',', dtype=np.float64)
        if (initial.shape[0] != shape[0]) or (initial.shape[1] != shape[1]):
            raise ValueError(
                'Initialization for %s is not correct shape. Expecting (%d,%d), but find (%d,%d) in %s.' % (
                    varname, shape[0], shape[1], initial.shape[0], initial.shape[1], distribution))
    return tf.Variable(initial, name=varname)


def bias_variable(shape, varname, distribution=''):
    if distribution:
        initial = np.genfromtxt(distribution, delimiter=',', dtype=np.float64)
    else:
        initial = tf.constant(0.1, shape=shape, dtype=tf.float64)
    return tf.Variable(initial, name=varname)


def encoder(widths, dist_weights, dist_biases, scale, num_shifts_max, first_guess):
    x = tf.placeholder(tf.float64, [num_shifts_max + 1, None, widths[0]])
    # nx1 patch, number of input channels, number of output channels (features)
    # m = number of hidden units

    weights = dict()
    biases = dict()

    for i in np.arange(len(widths) - 1):
        weights['WE%d' % (i + 1)] = weight_variable([widths[i], widths[i + 1]], varname='WE%d' % (i + 1),
                                                    distribution=dist_weights[i], scale=scale,
                                                    first_guess=first_guess)
        # TODO: first guess for biases too (and different ones for different weights)
        biases['bE%d' % (i + 1)] = bias_variable([widths[i + 1], ], varname='bE%d' % (i + 1),
                                                 distribution=dist_biases[i])
    return x, weights, biases


def encoder_apply(x, weights, biases, act_type, batch_flag, phase, out_flag, shifts_middle, keep_prob, name='E',
                  num_encoder_weights=1):
    y = []
    num_shifts_middle = len(shifts_middle)
    for j in np.arange(num_shifts_middle + 1):
        if j == 0:
            shift = 0
        else:
            shift = shifts_middle[j - 1]
        x_shift = tf.squeeze(x[shift, :, :])
        y.append(
            encoder_apply_one_shift(x_shift, weights, biases, act_type, batch_flag, phase, out_flag, keep_prob, name,
                                    num_encoder_weights))
    return y


def encoder_apply_one_shift(prev_layer, weights, biases, act_type, batch_flag, phase, out_flag, keep_prob, name='E',
                            num_encoder_weights=1):
    for i in np.arange(num_encoder_weights - 1):
        h1 = tf.matmul(prev_layer, weights['W%s%d' % (name, i + 1)]) + biases['b%s%d' % (name, i + 1)]
        if batch_flag:
            h1 = tf.contrib.layers.batch_norm(h1, is_training=phase)
        if act_type == 'sigmoid':
            h1 = tf.sigmoid(h1)
        elif act_type == 'relu':
            h1 = tf.nn.relu(h1)
        elif act_type == 'elu':
            h1 = tf.nn.elu(h1)
        prev_layer = tf.cond(keep_prob < 1.0, lambda: tf.nn.dropout(h1, keep_prob), lambda: h1)

    if len(weights) == 1:
        i = -1

    final = tf.matmul(prev_layer, weights['W%s%d' % (name, i + 2)]) + biases['b%s%d' % (name, i + 2)]
    if (not out_flag) and batch_flag:
        final = tf.contrib.layers.batch_norm(final, is_training=phase)

    return final


def decoder(widths, dist_weights, dist_biases, scale, name='D', first_guess=0):
    weights = dict()
    biases = dict()
    for i in np.arange(len(widths) - 1):
        ind = i + 1
        weights['W%s%d' % (name, ind)] = weight_variable([widths[i], widths[i + 1]], varname='W%s%d' % (name, ind),
                                                         distribution=dist_weights[ind - 1], scale=scale,
                                                         first_guess=first_guess)
        biases['b%s%d' % (name, ind)] = bias_variable([widths[i + 1], ], varname='b%s%d' % (name, ind),
                                                      distribution=dist_biases[ind - 1])
    return weights, biases


def decoder_apply(prev_layer, weights, biases, act_type, batch_flag, phase, keep_prob, num_decoder_weights):
    for i in np.arange(num_decoder_weights - 1):
        h1 = tf.matmul(prev_layer, weights['WD%d' % (i + 1)]) + biases['bD%d' % (i + 1)]
        if batch_flag:
            h1 = tf.contrib.layers.batch_norm(h1, is_training=phase)
        if act_type == 'sigmoid':
            h1 = tf.sigmoid(h1)
        elif act_type == 'relu':
            h1 = tf.nn.relu(h1)
        elif act_type == 'elu':
            h1 = tf.nn.elu(h1)
        prev_layer = tf.cond(keep_prob < 1.0, lambda: tf.nn.dropout(h1, keep_prob), lambda: h1)

    final = tf.matmul(prev_layer, weights['WD%d' % (i + 2)]) + biases['bD%d' % (i + 2)]

    return final


def form_L_stack(omega_output, deltat):
    # encoded_layer is [None, 2]
    # omega_output is [None, 1]
    if omega_output.shape[1] == 1:
        entry11 = tf.cos(omega_output * deltat)
        entry12 = tf.sin(omega_output * deltat)
        row1 = tf.concat([entry11, -entry12], axis=1)  # [None, 2]
        row2 = tf.concat([entry12, entry11], axis=1)  # [None, 2]

    elif omega_output.shape[1] == 2:
        scale = tf.exp(omega_output[:, 1] * deltat)
        entry11 = tf.multiply(scale, tf.cos(omega_output[:, 0] * deltat))
        entry12 = tf.multiply(scale, tf.sin(omega_output[:, 0] * deltat))
        row1 = tf.stack([entry11, -entry12], axis=1)  # [None, 2]
        row2 = tf.stack([entry12, entry11], axis=1)  # [None, 2]
    # TODO: generalize this function
    Lstack = tf.stack([row1, row2], axis=2)  # [None, 2, 2] put one row below other
    return Lstack


def varying_multiply(y, omegas, deltat):
    # multiply on the left: y*omegas

    # y is [None, 2] and omegas is [None, 1]
    ystack = tf.stack([y, y], axis=2)  # [None, 2, 2] put one row below other
    Lstack = form_L_stack(omegas, deltat)  # [None, 2, 2]
    elmtwise_prod = tf.multiply(ystack, Lstack)
    # add middle dimension (across "columns") i.e. cos(omega*deltat)*y1 + sin(omega*deltat)*y2
    output = tf.reduce_sum(elmtwise_prod, 1)  # [None, 2]
    return output


def create_omega_net(phase, keep_prob, params, x):
    weights, biases = decoder(params['widths_omega'], dist_weights=params['dist_weights_omega'],
                              dist_biases=params['dist_biases_omega'], scale=params['scale_omega'], name='O',
                              first_guess=params['first_guess_omega'])
    g_list = encoder_apply(x, weights, biases, params['act_type'], params['batch_flag'], phase, out_flag=0,
                           shifts_middle=params['shifts_middle'], keep_prob=keep_prob, name='O',
                           num_encoder_weights=len(weights))

    return g_list, weights, biases


def create_koopman_net(phase, keep_prob, params):
    depth = (params['d'] - 4) / 2  # i.e. 10 or 12 -> 3 or 4

    max_shifts_to_stack = 1
    if len(params['shifts']):
        max_shifts_to_stack = max(max_shifts_to_stack, max(params['shifts']))
    if len(params['shifts_middle']):
        max_shifts_to_stack = max(max_shifts_to_stack, max(params['shifts_middle']))

    encoder_widths = params['widths'][0:depth + 2]  # n ... nl
    x, weights, biases = encoder(encoder_widths, dist_weights=params['dist_weights'][0:depth + 1],
                                 dist_biases=params['dist_biases'][0:depth + 1], scale=params['scale'],
                                 num_shifts_max=max_shifts_to_stack, first_guess=params['first_guess'])
    params['num_encoder_weights'] = len(weights)
    g_list = encoder_apply(x, weights, biases, params['act_type'], params['batch_flag'], phase, out_flag=0,
                           shifts_middle=params['shifts_middle'], keep_prob=keep_prob,
                           num_encoder_weights=params['num_encoder_weights'])

    # g_list_omega is list of omegas, one entry for each middle_shift of x (like g_list)
    g_list_omega, weights_omega, biases_omega = create_omega_net(phase, keep_prob, params, x)
    params['num_omega_weights'] = len(weights_omega)
    weights.update(weights_omega)
    biases.update(biases_omega)

    num_widths = len(params['widths'])
    decoder_widths = params['widths'][depth + 2:num_widths]  # nl ... n
    weights_decoder, biases_decoder = decoder(decoder_widths, dist_weights=params['dist_weights'][depth + 2:],
                                              dist_biases=params['dist_biases'][depth + 2:],
                                              scale=params['scale'])
    weights.update(weights_decoder)
    biases.update(biases_decoder)

    y = []
    # y[0] is x[0,:,:] encoded and then decoded (no stepping forward)
    encoded_layer = g_list[0]
    params['num_decoder_weights'] = depth + 1
    y.append(decoder_apply(encoded_layer, weights, biases, params['act_type'], params['batch_flag'], phase, keep_prob,
                           params['num_decoder_weights']))

    # g_list_omega[0] is for x[0,:,:], pairs with g_list[0]=encoded_layer
    advanced_layer = varying_multiply(encoded_layer, g_list_omega[0], params['deltat'])

    for j in np.arange(max(params['shifts'])):  # loops 0, 1, ...
        # considering penalty on subset of yk+1, yk+2, yk+3, ... yk+20
        if (j + 1) in params['shifts']:
            y.append(decoder_apply(advanced_layer, weights, biases, params['act_type'], params['batch_flag'], phase,
                                   keep_prob, params['num_decoder_weights']))

        advanced_layer = varying_multiply(advanced_layer, g_list_omega[j + 1], params['deltat'])

    if len(y) != (len(params['shifts']) + 1):
        print "messed up looping over shifts! %r" % params['shifts']
        raise ValueError(
            'length(y) not proper length: check create_koopman_net code and how defined params[shifts] in experiment')

    return x, y, g_list, weights, biases, g_list_omega
