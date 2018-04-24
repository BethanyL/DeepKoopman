import numpy as np
import tensorflow as tf

import helperfns


def weight_variable(shape, var_name, distribution='tn', scale=0.1, first_guess=0):
    """Create a variable for a weight matrix.

    Arguments:
        shape -- array giving shape of output weight variable
        var_name -- string naming weight variable
        distribution -- string for which distribution to use for random initialization
        scale -- (for tn distribution): standard deviation of normal distribution before truncation
        first_guess -- (for tn distribution): array of first guess for weight matrix, added to tn dist.

    Returns:
        a TensorFlow variable for a weight matrix

    Side effects:
        None
    """
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
    elif distribution == 'he':
        # from He, et al. ICCV 2015 (referenced in Andrew Ng's class)
        # divide by m, where m is number of inputs
        scale = np.sqrt(2.0 / shape[0])
        initial = tf.random_normal(shape, mean=0, stddev=scale, dtype=tf.float64)
    elif distribution == 'glorot_bengio':
        # see page 295 of Goodfellow et al's DL book
        scale = np.sqrt(6.0 / (shape[0] + shape[1]))
        initial = tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float64)
    else:
        initial = np.loadtxt(distribution, delimiter=',', dtype=np.float64)
        if (initial.shape[0] != shape[0]) or (initial.shape[1] != shape[1]):
            raise ValueError(
                'Initialization for %s is not correct shape. Expecting (%d,%d), but find (%d,%d) in %s.' % (
                    var_name, shape[0], shape[1], initial.shape[0], initial.shape[1], distribution))
    return tf.Variable(initial, name=var_name)


def bias_variable(shape, var_name, distribution=''):
    """Create a variable for a bias vector.

    Arguments:
        shape -- array giving shape of output bias variable
        var_name -- string naming bias variable
        distribution -- string for which distribution to use for random initialization (file name)

    Returns:
        a TensorFlow variable for a bias vector

    Side effects:
        None
    """
    if distribution:
        initial = np.genfromtxt(distribution, delimiter=',', dtype=np.float64)
    else:
        initial = tf.constant(0.0, shape=shape, dtype=tf.float64)
    return tf.Variable(initial, name=var_name)


def encoder(widths, dist_weights, dist_biases, scale, num_shifts_max, first_guess):
    """Create an encoder network: an input placeholder x, dictionary of weights, and dictionary of biases.

    Arguments:
        widths -- array or list of widths for layers of network
        dist_weights -- array or list of strings for distributions of weight matrices
        dist_biases -- array or list of strings for distributions of bias vectors
        scale -- (for tn distribution of weight matrices): standard deviation of normal distribution before truncation
        num_shifts_max -- number of shifts (time steps) that losses will use (max of num_shifts and num_shifts_middle)
        first_guess -- (for tn dist. of weight matrices): array of first guess for weight matrix, added to tn dist.

    Returns:
        x -- placeholder for input
        weights -- dictionary of weights
        biases -- dictionary of biases

    Side effects:
        None
    """
    x = tf.placeholder(tf.float64, [num_shifts_max + 1, None, widths[0]])

    weights = dict()
    biases = dict()

    for i in np.arange(len(widths) - 1):
        weights['WE%d' % (i + 1)] = weight_variable([widths[i], widths[i + 1]], var_name='WE%d' % (i + 1),
                                                    distribution=dist_weights[i], scale=scale,
                                                    first_guess=first_guess)
        # TODO: first guess for biases too (and different ones for different weights)
        biases['bE%d' % (i + 1)] = bias_variable([widths[i + 1], ], var_name='bE%d' % (i + 1),
                                                 distribution=dist_biases[i])
    return x, weights, biases


def encoder_apply(x, weights, biases, act_type, batch_flag, phase, shifts_middle, keep_prob, name='E',
                  num_encoder_weights=1):
    """Apply an encoder to data x.

    Arguments:
        x -- placeholder for input
        weights -- dictionary of weights
        biases -- dictionary of biases
        act_type -- string for activation type for nonlinear layers (i.e. sigmoid, relu, or elu)
        batch_flag -- 0 if no batch_normalization, 1 if batch_normalization
        phase -- boolean placeholder for dropout: training phase or not training phase
        shifts_middle -- number of shifts (steps) in x to apply encoder to for linearity loss
        keep_prob -- probability that weight is kept during dropout
        name -- string for prefix on weight matrices, default 'E' for encoder
        num_encoder_weights -- number of weight matrices (layers) in encoder network, default 1

    Returns:
        y -- list, output of encoder network applied to each time shift in input x

    Side effects:
        None
    """
    y = []
    num_shifts_middle = len(shifts_middle)
    for j in np.arange(num_shifts_middle + 1):
        if j == 0:
            shift = 0
        else:
            shift = shifts_middle[j - 1]
        if isinstance(x, (list,)):
            x_shift = x[shift]
        else:
            x_shift = tf.squeeze(x[shift, :, :])
        y.append(
            encoder_apply_one_shift(x_shift, weights, biases, act_type, batch_flag, phase, keep_prob, name,
                                    num_encoder_weights))
    return y


def encoder_apply_one_shift(prev_layer, weights, biases, act_type, batch_flag, phase, keep_prob, name='E',
                            num_encoder_weights=1):
    """Apply an encoder to data for only one time step (shift).

    Arguments:
        prev_layer -- input for a particular time step (shift)
        weights -- dictionary of weights
        biases -- dictionary of biases
        act_type -- string for activation type for nonlinear layers (i.e. sigmoid, relu, or elu)
        batch_flag -- 0 if no batch_normalization, 1 if batch_normalization
        phase -- boolean placeholder for dropout: training phase or not training phase
        keep_prob -- probability that weight is kept during dropout
        name -- string for prefix on weight matrices, default 'E' (for "encoder")
        num_encoder_weights -- number of weight matrices (layers) in encoder network, default 1

    Returns:
        final -- output of encoder network applied to input prev_layer (a particular time step / shift)

    Side effects:
        None
    """
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

    # apply last layer without any nonlinearity
    final = tf.matmul(prev_layer, weights['W%s%d' % (name, num_encoder_weights)]) + biases[
        'b%s%d' % (name, num_encoder_weights)]

    if batch_flag:
        final = tf.contrib.layers.batch_norm(final, is_training=phase)

    return final


def decoder(widths, dist_weights, dist_biases, scale, name='D', first_guess=0):
    """Create a decoder network: a dictionary of weights and a dictionary of biases.

    Arguments:
        widths -- array or list of widths for layers of network
        dist_weights -- array or list of strings for distributions of weight matrices
        dist_biases -- array or list of strings for distributions of bias vectors
        scale -- (for tn distribution of weight matrices): standard deviation of normal distribution before truncation
        name -- string for prefix on weight matrices, default 'D' (for "decoder")
        first_guess -- (for tn dist. of weight matrices): array of first guess for weight matrix, added to tn dist.

    Returns:
        weights -- dictionary of weights
        biases -- dictionary of biases

    Side effects:
        None
    """
    weights = dict()
    biases = dict()
    for i in np.arange(len(widths) - 1):
        ind = i + 1
        weights['W%s%d' % (name, ind)] = weight_variable([widths[i], widths[i + 1]], var_name='W%s%d' % (name, ind),
                                                         distribution=dist_weights[ind - 1], scale=scale,
                                                         first_guess=first_guess)
        biases['b%s%d' % (name, ind)] = bias_variable([widths[i + 1], ], var_name='b%s%d' % (name, ind),
                                                      distribution=dist_biases[ind - 1])
    return weights, biases


def decoder_apply(prev_layer, weights, biases, act_type, batch_flag, phase, keep_prob, num_decoder_weights):
    """Apply a decoder to data prev_layer

    Arguments:
        prev_layer -- input to decoder network
        weights -- dictionary of weights
        biases -- dictionary of biases
        act_type -- string for activation type for nonlinear layers (i.e. sigmoid, relu, or elu)
        batch_flag -- 0 if no batch_normalization, 1 if batch_normalization
        phase -- boolean placeholder for dropout: training phase or not training phase
        keep_prob -- probability that weight is kept during dropout
        num_decoder_weights -- number of weight matrices (layers) in decoder network

    Returns:
        output of decoder network applied to input prev_layer

    Side effects:
        None
    """
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

    # apply last layer without any nonlinearity
    return tf.matmul(prev_layer, weights['WD%d' % num_decoder_weights]) + biases['bD%d' % num_decoder_weights]


def form_complex_conjugate_block(omegas, delta_t):
    """Form a 2x2 block for a complex conj. pair of eigenvalues, but for each example, so dimension [None, 2, 2]

    2x2 Block is
    exp(mu * delta_t) * [cos(omega * delta_t), -sin(omega * delta_t)
                         sin(omega * delta_t), cos(omega * delta_t)]

    Arguments:
        omegas -- array of parameters for blocks. first column is freq. (omega) and 2nd is scaling (mu), size [None, 2]
        delta_t -- time step in trajectories from input data

    Returns:
        stack of 2x2 blocks, size [None, 2, 2], where first dimension matches first dimension of omegas

    Side effects:
        None
    """
    scale = tf.exp(omegas[:, 1] * delta_t)
    entry11 = tf.multiply(scale, tf.cos(omegas[:, 0] * delta_t))
    entry12 = tf.multiply(scale, tf.sin(omegas[:, 0] * delta_t))
    row1 = tf.stack([entry11, -entry12], axis=1)  # [None, 2]
    row2 = tf.stack([entry12, entry11], axis=1)  # [None, 2]
    return tf.stack([row1, row2], axis=2)  # [None, 2, 2] put one row below other


def varying_multiply(y, omegas, delta_t, num_real, num_complex_pairs):
    """Multiply y-coordinates on the left by matrix L, but let matrix vary.

    Arguments:
        y -- array of shape [None, k] of y-coordinates, where L will be k x k
        omegas -- list of arrays of parameters for the L matrices
        delta_t -- time step in trajectories from input data
        num_real -- number of real eigenvalues
        num_complex_pairs -- number of pairs of complex conjugate eigenvalues

    Returns:
        array same size as input y, but advanced to next time step

    Side effects:
        None
    """
    k = y.shape[1]
    complex_list = []

    # first, Jordan blocks for each pair of complex conjugate eigenvalues
    for j in np.arange(num_complex_pairs):
        ind = 2 * j
        ystack = tf.stack([y[:, ind:ind + 2], y[:, ind:ind + 2]], axis=2)  # [None, 2, 2]
        L_stack = form_complex_conjugate_block(omegas[j], delta_t)
        elmtwise_prod = tf.multiply(ystack, L_stack)
        complex_list.append(tf.reduce_sum(elmtwise_prod, 1))

    if len(complex_list):
        # each element in list output_list is shape [None, 2]
        complex_part = tf.concat(complex_list, axis=1)

    # next, diagonal structure for each real eigenvalue
    # faster to not explicitly create stack of diagonal matrices L
    real_list = []
    for j in np.arange(num_real):
        ind = 2 * num_complex_pairs + j
        temp = y[:, ind]
        real_list.append(tf.multiply(temp[:, np.newaxis], tf.exp(omegas[num_complex_pairs + j] * delta_t)))

    if len(real_list):
        real_part = tf.concat(real_list, axis=1)
    if len(complex_list) and len(real_list):
        return tf.concat([complex_part, real_part], axis=1)
    elif len(complex_list):
        return complex_part
    else:
        return real_part


def create_omega_net(phase, keep_prob, params, ycoords):
    """Create the auxiliary (omega) network(s), which have ycoords as input and output omegas (parameters for L).

    Arguments:
        phase -- boolean placeholder for dropout: training phase or not training phase
        keep_prob -- probability that weight is kept during dropout
        params -- dictionary of parameters for experiment
        ycoords -- array of shape [None, k] of y-coordinates, where L will be k x k

    Returns:
        omegas -- list, output of omega (auxiliary) network(s) applied to input ycoords
        weights -- dictionary of weights
        biases -- dictionary of biases

    Side effects:
        Adds 'num_omega_weights' key to params dict
    """
    weights = dict()
    biases = dict()

    for j in np.arange(params['num_complex_pairs']):
        temp_name = 'OC%d_' % (j + 1)
        create_one_omega_net(params, temp_name, weights, biases, params['widths_omega_complex'])

    for j in np.arange(params['num_real']):
        temp_name = 'OR%d_' % (j + 1)
        create_one_omega_net(params, temp_name, weights, biases, params['widths_omega_real'])

    params['num_omega_weights'] = len(params['widths_omega_real']) - 1

    omegas = omega_net_apply(phase, keep_prob, params, ycoords, weights, biases)

    return omegas, weights, biases


def create_one_omega_net(params, temp_name, weights, biases, widths):
    """Create one auxiliary (omega) network for one real eigenvalue or a pair of complex conj. eigenvalues.

    Arguments:
        params -- dictionary of parameters for experiment
        temp_name -- string for prefix on weight matrices, i.e. OC1 or OR1
        weights -- dictionary of weights
        biases -- dictionary of biases
        widths -- array or list of widths for layers of network

    Returns:
        None

    Side effects:
        Updates weights and biases dictionaries
    """
    weightsO, biasesO = decoder(widths, dist_weights=params['dist_weights_omega'],
                                dist_biases=params['dist_biases_omega'], scale=params['scale_omega'], name=temp_name,
                                first_guess=params['first_guess_omega'])
    weights.update(weightsO)
    biases.update(biasesO)


def omega_net_apply(phase, keep_prob, params, ycoords, weights, biases):
    """Apply the omega (auxiliary) network(s) to the y-coordinates.

    Arguments:
        phase -- boolean placeholder for dropout: training phase or not training phase
        keep_prob -- probability that weight is kept during dropout
        params -- dictionary of parameters for experiment
        ycoords -- array of shape [None, k] of y-coordinates, where L will be k x k
        weights -- dictionary of weights
        biases -- dictionary of biases

    Returns:
        omegas -- list, output of omega (auxiliary) network(s) applied to input ycoords

    Side effects:
        None
    """
    omegas = []
    for j in np.arange(params['num_complex_pairs']):
        temp_name = 'OC%d_' % (j + 1)
        ind = 2 * j
        omegas.append(
            omega_net_apply_one(phase, keep_prob, params, ycoords[:, ind:ind + 2], weights, biases, temp_name))
    for j in np.arange(params['num_real']):
        temp_name = 'OR%d_' % (j + 1)
        ind = 2 * params['num_complex_pairs'] + j
        omegas.append(omega_net_apply_one(phase, keep_prob, params, ycoords[:, ind], weights, biases, temp_name))

    return omegas


def omega_net_apply_one(phase, keep_prob, params, ycoords, weights, biases, name):
    """Apply one auxiliary (omega) network for one real eigenvalue or a pair of complex conj. eigenvalues.

    Arguments:
        phase -- boolean placeholder for dropout: training phase or not training phase
        keep_prob -- probability that weight is kept during dropout
        params -- dictionary of parameters for experiment
        ycoords -- array of shape [None, k] of y-coordinates, where L will be k x k
        weights -- dictionary of weights
        biases -- dictionary of biases
        name -- string for prefix on weight matrices, i.e. OC1 or OR1

    Returns:
        omegas - output of one auxiliary (omega) network to input ycoords

    Side effects:
        None
    """
    if len(ycoords.shape) == 1:
        ycoords = ycoords[:, np.newaxis]

    if ycoords.shape[1] == 2:
        # complex conjugate pair
        input = tf.reduce_sum(tf.square(ycoords), axis=1, keep_dims=True)

    else:
        input = ycoords

    omegas = encoder_apply_one_shift(input, weights, biases, params['act_type'], params['batch_flag'], phase,
                                     keep_prob=keep_prob, name=name,
                                     num_encoder_weights=params['num_omega_weights'])
    return omegas


def create_koopman_net(phase, keep_prob, params):
    """Create a Koopman network that encodes, advances in time, and decodes.

    Arguments:
        phase -- boolean placeholder for dropout: training phase or not training phase
        keep_prob -- probability that weight is kept during dropout
        params -- dictionary of parameters for experiment

    Returns:
        x -- placeholder for input
        y -- list, output of decoder applied to each shift: g_list[0], K*g_list[0], K^2*g_list[0], ..., length num_shifts + 1
        g_list -- list, output of encoder applied to each shift in input x, length num_shifts_middle + 1
        weights -- dictionary of weights
        biases -- dictionary of biases

    Side effects:
        Adds more entries to params dict: num_encoder_weights, num_omega_weights, num_decoder_weights
    """
    depth = int((params['d'] - 4) / 2)

    max_shifts_to_stack = helperfns.num_shifts_in_stack(params)

    encoder_widths = params['widths'][0:depth + 2]  # n ... k
    x, weights, biases = encoder(encoder_widths, dist_weights=params['dist_weights'][0:depth + 1],
                                 dist_biases=params['dist_biases'][0:depth + 1], scale=params['scale'],
                                 num_shifts_max=max_shifts_to_stack, first_guess=params['first_guess'])
    params['num_encoder_weights'] = len(weights)
    g_list = encoder_apply(x, weights, biases, params['act_type'], params['batch_flag'], phase,
                           shifts_middle=params['shifts_middle'], keep_prob=keep_prob,
                           num_encoder_weights=params['num_encoder_weights'])

    # g_list_omega is list of omegas, one entry for each middle_shift of x (like g_list)
    omegas, weights_omega, biases_omega = create_omega_net(phase, keep_prob, params, g_list[0])
    # params['num_omega_weights'] = len(weights_omega) already done inside create_omega_net
    weights.update(weights_omega)
    biases.update(biases_omega)

    num_widths = len(params['widths'])
    decoder_widths = params['widths'][depth + 2:num_widths]  # k ... n
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
    advanced_layer = varying_multiply(encoded_layer, omegas, params['delta_t'], params['num_real'],
                                      params['num_complex_pairs'])

    for j in np.arange(max(params['shifts'])):
        # considering penalty on subset of yk+1, yk+2, yk+3, ...
        if (j + 1) in params['shifts']:
            y.append(decoder_apply(advanced_layer, weights, biases, params['act_type'], params['batch_flag'], phase,
                                   keep_prob, params['num_decoder_weights']))

        omegas = omega_net_apply(phase, keep_prob, params, advanced_layer, weights, biases)
        advanced_layer = varying_multiply(advanced_layer, omegas, params['delta_t'], params['num_real'],
                                          params['num_complex_pairs'])

    if len(y) != (len(params['shifts']) + 1):
        print("messed up looping over shifts! %r" % params['shifts'])
        raise ValueError(
            'length(y) not proper length: check create_koopman_net code and how defined params[shifts] in experiment')

    return x, y, g_list, weights, biases
