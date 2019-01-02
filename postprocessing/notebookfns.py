import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def PlotErrors(errors, logInd):
    print("note that these errors include the weights alpha_1, alpha_2, alpha_3")
    print("red is training error, blue is validation error")
    for j in logInd:
        errors[:, j] = np.log10(errors[:, j])
    f, ax = plt.subplots(3, 3)
    f.set_figwidth(13)
    f.set_figheight(9)
    numErrs = errors.shape[0]
    ax[0, 0].scatter(np.arange(numErrs), errors[:, 0], c='r')
    ax[0, 0].scatter(np.arange(numErrs), errors[:, 1], c='b')
    ax[0, 0].set_ylabel('log10(pre-reg. error)')
    ax[0, 1].scatter(np.arange(numErrs), errors[:, 2], c='r')
    ax[0, 1].scatter(np.arange(numErrs), errors[:, 3], c='b')
    ax[0, 1].set_ylabel('log10(reg. error)')

    ax[1, 0].scatter(np.arange(numErrs), errors[:, 4], c='r')
    ax[1, 0].scatter(np.arange(numErrs), errors[:, 5], c='b')
    ax[1, 0].set_ylabel('log10(loss1: reconstruction)')
    ax[1, 1].scatter(np.arange(numErrs), errors[:, 6], c='r')
    ax[1, 1].scatter(np.arange(numErrs), errors[:, 7], c='b')
    ax[1, 1].set_ylabel('log10(loss2: prediction)')
    ax[1, 2].scatter(np.arange(numErrs), errors[:, 8], c='r')
    ax[1, 2].scatter(np.arange(numErrs), errors[:, 9], c='b')
    ax[1, 2].set_ylabel('log10(loss3: linearity)')

    ax[2, 0].scatter(np.arange(numErrs), errors[:, 10], c='r')
    ax[2, 0].scatter(np.arange(numErrs), errors[:, 11], c='b')
    ax[2, 0].set_ylabel('log10(L_inf)')
    ax[2, 1].scatter(np.arange(numErrs), errors[:, 12], c='r')
    ax[2, 1].scatter(np.arange(numErrs), errors[:, 13], c='b')
    ax[2, 1].set_ylabel('log10(L1)')
    ax[2, 2].scatter(np.arange(numErrs), errors[:, 14], c='r')
    ax[2, 2].scatter(np.arange(numErrs), errors[:, 15], c='b')
    ax[2, 2].set_ylabel('log10(L2)')
    plt.tight_layout()


def load_weights(fname, numWeights, type='E'):
    W = dict()
    b = dict()
    for j in range(numWeights):
        W1 = np.matrix(np.genfromtxt(fname.replace("model.pkl", "W%s%d.csv" % (type, j + 1)), delimiter=','))
        b1 = np.matrix(np.genfromtxt(fname.replace("model.pkl", "b%s%d.csv" % (type, j + 1)), delimiter=','))
        if j > 0:
            if W1.shape[0] != lastSize:
                if W1.shape[0] == 1:
                    # need to transpose?
                    if W1.shape[1] == lastSize:
                        W1 = np.transpose(W1)
                    else:
                        print("error: sizes %d and %r" % (lastSize, W1.shape))
                else:
                    print("error: sizes %d and %r" % (lastSize, W1.shape))
        lastSize = W1.shape[1]
        W['W%s%d' % (type, j + 1)] = W1
        b['b%s%d' % (type, j + 1)] = b1

    return W, b


def load_weights_koopman(fname, numWeights, numWeightsOmega, num_real, num_complex_pairs):
    d = int((numWeights - 1) / 2)

    weights, biases = load_weights(fname, d, 'E')
    W, b = load_weights(fname, d, 'D')
    weights.update(W)
    biases.update(b)

    for j in np.arange(num_complex_pairs):
        W, b = load_weights(fname, numWeightsOmega, 'OC%d_' % (j + 1))
        weights.update(W)
        biases.update(b)

    for j in np.arange(num_real):
        W, b = load_weights(fname, numWeightsOmega, 'OR%d_' % (j + 1))
        weights.update(W)
        biases.update(b)

    return weights, biases


def num_shifts_in_stack(params):
    max_shifts_to_stack = 1
    if params['num_shifts']:
        max_shifts_to_stack = max(max_shifts_to_stack, max(params['shifts']))
    if params['num_shifts_middle']:
        max_shifts_to_stack = max(max_shifts_to_stack, max(params['shifts_middle']))

    return max_shifts_to_stack


def stack_data(data, num_shifts, len_time):
    nd = data.ndim
    if nd > 1:
        n = data.shape[1]
    else:
        data = (np.asmatrix(data)).getT()
        n = 1
    num_traj = int(data.shape[0] / len_time)

    newlen_time = len_time - num_shifts

    data_tensor = np.zeros((num_shifts + 1, num_traj * newlen_time, n))

    for j in np.arange(num_shifts + 1):
        for count in np.arange(num_traj):
            data_tensor[j, count * newlen_time: newlen_time + count * newlen_time, :] = data[
                                                                                        count * len_time + j: count * len_time + j + newlen_time,
                                                                                        :]

    return data_tensor, num_traj


def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def encoder_apply(x, weights, biases, name, num_weights, act_type='relu'):
    prev_layer = x.copy()

    for i in np.arange(num_weights - 1):
        h1 = np.dot(prev_layer, weights['W%s%d' % (name, i + 1)]) + biases['b%s%d' % (name, i + 1)]

        if act_type == 'sigmoid':
            h1 = sigmoid(h1)
        elif act_type == 'relu':
            h1 = relu(h1)

        prev_layer = h1.copy()
    final = np.dot(prev_layer, weights['W%s%d' % (name, num_weights)]) + biases['b%s%d' % (name, num_weights)]

    return final


def omega_net_apply_one(ycoords, W, b, name, num_weights, act_type):
    if ycoords.shape[1] == 2:
        input = np.sum(np.square(ycoords), axis=1)
    else:
        input = ycoords

    # want input to be [?, 1]
    if len(input.shape) == 1:
        input = input[:, np.newaxis]

    return encoder_apply(input, W, b, name, num_weights, act_type)


def omega_net_apply(ycoords, W, b, num_real, num_complex_pairs, num_weights, act_type='relu'):
    omegas = []

    for j in np.arange(num_complex_pairs):
        temp_name = 'OC%d_' % (j + 1)
        ind = 2 * j
        omegas.append(omega_net_apply_one(ycoords[:, ind:ind + 2], W, b, temp_name, num_weights, act_type))

    for j in np.arange(num_real):
        temp_name = 'OR%d_' % (j + 1)
        ind = 2 * num_complex_pairs + j
        omegas.append(omega_net_apply_one(ycoords[:, ind], W, b, temp_name, num_weights, act_type))

    return omegas


def varying_multiply(y, omegas, delta_t, num_real, num_complex_pairs):
    k = y.shape[1]

    complex_list = []

    for j in np.arange(num_complex_pairs):
        ind = 2 * j
        ystack = np.stack([np.asarray(y[:, ind:ind + 2]), np.asarray(y[:, ind:ind + 2])], axis=2)
        L_stack = FormComplexConjugateBlock(omegas[j], delta_t)
        elmtwise_prod = np.multiply(ystack, L_stack)
        complex_list.append(np.sum(elmtwise_prod, axis=1))

    if len(complex_list):
        complex_part = np.concatenate(complex_list, axis=1)

    real_list = []
    for j in np.arange(num_real):
        ind = 2 * num_complex_pairs + j
        temp_y = y[:, ind]
        if len(temp_y.shape) == 1:
            temp_y = temp_y[:, np.newaxis]
        temp_omegas = omegas[num_complex_pairs + j]
        evals = np.exp(temp_omegas * delta_t)
        item_real_list = np.multiply(temp_y, evals)
        real_list.append(item_real_list)

    if len(real_list):
        real_part = np.concatenate(real_list, axis=1)

    if len(complex_list) and len(real_list):
        return np.concatenate([complex_part, real_part], axis=1)

    elif len(complex_list):
        return complex_part

    else:
        return real_part


def FormComplexConjugateBlock(omegas, delta_t):
    omegas = np.array(omegas)
    scale = np.exp(omegas[:, 1] * delta_t)
    entry11 = np.multiply(scale, np.cos(omegas[:, 0] * delta_t))
    entry12 = np.multiply(scale, np.sin(omegas[:, 0] * delta_t))
    row1 = np.stack([entry11, -entry12], axis=1)  # [None, 2]
    row2 = np.stack([entry12, entry11], axis=1)  # [None, 2]
    Lstack = np.stack([row1, row2], axis=2)  # [None, 2, 2] put one row below other
    return Lstack


def decoder_apply(prev_layer, weights, biases, name, num_weights, act_type='relu'):
    for i in np.arange(num_weights - 1):
        h1 = np.dot(prev_layer, weights['W%s%d' % (name, i + 1)]) + biases['b%s%d' % (name, i + 1)]

        if act_type == 'sigmoid':
            h1 = sigmoid(h1)
        elif act_type == 'relu':
            h1 = relu(h1)
        prev_layer = h1.copy()
    final = np.dot(prev_layer, weights['W%s%d' % (name, num_weights)]) + biases['b%s%d' % (name, num_weights)]

    return final


def ApplyKoopmanNetOmegas(x, W, b, delta_t, num_real, num_complex_pairs, num_encoder_weights, num_omega_weights,
                          num_decoder_weights):
    # print x.shape

    yk = encoder_apply(x, W, b, 'E', num_encoder_weights)
    omegas = omega_net_apply(yk, W, b, num_real, num_complex_pairs, num_omega_weights)

    ykplus1 = varying_multiply(yk, omegas, delta_t, num_real, num_complex_pairs)
    omegas = omega_net_apply(ykplus1, W, b, num_real, num_complex_pairs, num_omega_weights)
    ykplus2 = varying_multiply(ykplus1, omegas, delta_t, num_real, num_complex_pairs)
    omegas = omega_net_apply(ykplus2, W, b, num_real, num_complex_pairs, num_omega_weights)
    ykplus3 = varying_multiply(ykplus2, omegas, delta_t, num_real, num_complex_pairs)

    xk_recon = decoder_apply(yk, W, b, 'D', num_decoder_weights)
    xkplus1 = decoder_apply(ykplus1, W, b, 'D', num_decoder_weights)
    xkplus2 = decoder_apply(ykplus2, W, b, 'D', num_decoder_weights)
    xkplus3 = decoder_apply(ykplus3, W, b, 'D', num_decoder_weights)

    return yk, ykplus1, ykplus2, ykplus3, xk_recon, xkplus1, xkplus2, xkplus3


def EigenfunctionPlot(cmapname, data, outline_x1, outline_y1, outline_x2, outline_y2,
                      X, Y, filename='', cbFlag=True, levels=np.arange(-.3, .45, .15),
                      cbTicks=[-.3, 0, .3], climits=None):
    fig = plt.figure()
    plt.figure(figsize=(16 / 3 * 2, 16 / 3 * 2))
    # fig.set_figheight(16/3*2)
    im = plt.imshow(data, interpolation='bilinear', origin='lower',
                    cmap=cmapname, extent=(np.min(X), np.max(X), np.min(Y), np.max(Y)))
    print(levels)

    if climits:
        plt.clim(climits[0], climits[1])

    colors = [(1, 1, 1), (.4, .4, .4), (1, 1, 1)]
    mymapWBW = LinearSegmentedColormap.from_list('white_black_white', colors, N=7)

    CS = plt.contour(data, levels,
                     origin='lower',
                     linewidths=2,
                     extent=(np.min(X), np.max(X), np.min(Y), np.max(Y)), cmap=mymapWBW)

    # plt.clabel(CS, levels,  #levels[1::2],  # label every second level
    #           inline=1,
    #           fmt='%1.2f',
    #           fontsize=30)
    plt.axis('off')
    plt.plot(outline_x1, outline_y1, c='k', linewidth=4)
    plt.plot(outline_x2, outline_y2, c='k', linewidth=4)
    # make a colorbar for the contour lines
    # CB = plt.colorbar(CS, shrink=0.8, extend='both')

    if cbFlag:
        CBI = plt.colorbar(im, orientation='horizontal', shrink=.7, ticks=cbTicks)
        # CBI = plt.colorbar(im, orientation='horizontal', shrink=.8, ticks = [0, .6, 1.2])
        # CBI.ax.set_xticklabels(['-0.1', '0.0', '0.1'])
        CBI.ax.set_xticklabels(['', '', ''])

    if filename:
        plt.savefig(filename, dpi=200, transparent=True)
        # plt.savefig(filename, dpi=200, transparent=False)


def PredictKoopmanNetOmegas(x, weights, biases, delta_t, num_steps, num_real, num_complex_pairs, num_encoder_weights,
                            num_omega_weights, num_decoder_weights):
    n = weights['WE1'].shape[0]
    if len(x.shape) == 1:
        x = x[np.newaxis, :]
    num_examples = x.shape[0]
    prediction = np.zeros((num_examples, num_steps, n))

    ypred = encoder_apply(x, weights, biases, 'E', num_encoder_weights)
    for j in np.arange(num_steps):
        omegas = omega_net_apply(ypred, weights, biases, num_real, num_complex_pairs, num_omega_weights)
        ypred = varying_multiply(ypred, omegas, delta_t, num_real, num_complex_pairs)
        prediction[:, j, :] = decoder_apply(ypred, weights, biases, 'D', num_decoder_weights)

    return prediction


def ApplyKoopmanNetOmegasFull(x_stacked, W, b, delta_t, num_shifts, num_shifts_middle, num_real,
                              num_complex_pairs, num_encoder_weights, num_omega_weights, num_decoder_weights):
    g_list = []
    y = []

    for shift in np.arange(num_shifts_middle + 1):
        x = np.squeeze(x_stacked[shift, :, :])
        yk = encoder_apply(x, W, b, 'E', num_encoder_weights)
        g_list.append(yk)

    encoded_layer = g_list[0]
    y.append(decoder_apply(encoded_layer, W, b, 'D', num_decoder_weights))
    omegas = omega_net_apply(encoded_layer, W, b, num_real, num_complex_pairs, num_omega_weights)
    advanced_layer = varying_multiply(encoded_layer, omegas, delta_t, num_real, num_complex_pairs)
    for shift in np.arange(num_shifts):
        y.append(decoder_apply(advanced_layer, W, b, 'D', num_decoder_weights))
        omegas = omega_net_apply(advanced_layer, W, b, num_real, num_complex_pairs, num_omega_weights)
        advanced_layer = varying_multiply(advanced_layer, omegas, delta_t, num_real, num_complex_pairs)

    return y, g_list


def define_loss(x, y, g_list, params, W, b):
    # Minimize the mean squared errors.
    # subtraction and squaring element-wise, then average over both dimensions
    # n columns
    # average of each row (across columns), then average the rows

    num_real = params['num_real']
    num_complex_pairs = params['num_complex_pairs']
    num_omega_weights = params['num_omega_weights']

    # autoencoder loss
    mean_squared_error1 = np.mean(np.mean(np.square(y[0] - np.squeeze(x[0, :, :])), axis=1))
    loss1 = params['recon_lam'] * mean_squared_error1

    # gets dynamics
    loss2 = 0
    if params['num_shifts'] > 0:
        for j in np.arange(params['num_shifts']):
            # xk+1, xk+2, xk+3
            shift = params['shifts'][j]
            mean_squared_error2 = np.mean(np.mean(np.square(y[j + 1] - np.squeeze(x[shift, :, :])), axis=1))
            loss2 = loss2 + params['recon_lam'] * mean_squared_error2
        loss2 = loss2 / params['num_shifts']

    # K linear
    loss3 = 0
    count_shifts_middle = 0
    if params['num_shifts_middle'] > 0:
        omegas = omega_net_apply(g_list[0], W, b, num_real, num_complex_pairs, num_omega_weights)
        next_step = varying_multiply(g_list[0], omegas, params['delta_t'], params['num_real'],
                                     params['num_complex_pairs'])
        for j in np.arange(max(params['shifts_middle'])):
            if (j + 1) in params['shifts_middle']:
                # multiply g_list[0] by L (j+1) times
                # next_step = tf.matmul(g_list[0], L_pow)
                mean_squared_error3 = np.mean(np.mean(np.square(next_step - g_list[count_shifts_middle + 1]), axis=1))
                loss3 = loss3 + params['mid_shift_lam'] * mean_squared_error3
                count_shifts_middle += 1
            omegas = omega_net_apply(next_step, W, b, num_real, num_complex_pairs, num_omega_weights)
            next_step = varying_multiply(next_step, omegas, params['delta_t'], params['num_real'],
                                         params['num_complex_pairs'])
        loss3 = loss3 / params['num_shifts_middle']

    # inf norm on autoencoder error
    Linf1_penalty = np.linalg.norm(np.linalg.norm(y[0] - np.squeeze(x[0, :, :]), axis=1, ord=np.inf), ord=np.inf)
    Linf2_penalty = np.linalg.norm(np.linalg.norm(y[1] - np.squeeze(x[1, :, :]), axis=1, ord=np.inf), ord=np.inf)
    loss_Linf = params['Linf_lam'] * (Linf1_penalty + Linf2_penalty)

    loss = loss1 + loss2 + loss3 + loss_Linf

    return loss1, loss2, loss3, loss_Linf, loss


def L1_loss_fn(W, b):
    loss = 0
    for key in W:
        loss += np.sum(np.abs(W[key]))
    for key in b:
        loss += np.sum(np.abs(b[key]))
    return loss


def L2_loss_fn(W):
    loss = 0
    for key in W:
        loss += np.sum(np.square(W[key])) / 2
    return loss


def define_regularization(params, W, b, loss):
    if params['L1_lam']:
        loss_L1 = params['L1_lam'] * L1_loss_fn(W, b)
    else:
        loss_L1 = 0

    loss_L2 = params['L2_lam'] * L2_loss_fn(W)

    regularized_loss = loss + loss_L1 + loss_L2

    return loss_L1, loss_L2, regularized_loss


def loss_training(params, max_shifts_to_stack, W, b):
    total_loss1 = 0
    total_loss2 = 0
    total_loss3 = 0
    total_loss_Linf = 0
    total_loss = 0
    total_regularized_loss = 0
    data_train_len = params['data_train_len']
    total_num_traj = 0
    for j in np.arange(data_train_len):
        X = np.loadtxt('%s_train%d_x.csv' % (params['data_name'], j + 1), delimiter=',')
        X_stacked, num_traj = stack_data(X, max_shifts_to_stack, params['len_time'])
        print("file %d has %d trajectories" % (j+1, num_traj))
        total_num_traj += num_traj
        y, g_list = ApplyKoopmanNetOmegasFull(X_stacked, W, b, params['delta_t'], params['num_shifts'],
                                              params['num_shifts_middle'], params['num_real'],
                                              params['num_complex_pairs'],
                                              params['num_encoder_weights'], params['num_omega_weights'],
                                              params['num_decoder_weights'])
        loss1, loss2, loss3, loss_Linf, loss = define_loss(X_stacked, y, g_list, params, W, b)
        total_loss1 += loss1
        total_loss2 += loss2
        total_loss3 += loss3
        total_loss_Linf += loss_Linf
        total_loss += loss
        loss_L1, loss_L2, regularized_loss = define_regularization(params, W, b, loss)
        total_regularized_loss += regularized_loss
    av_loss1 = total_loss1 / data_train_len
    av_loss2 = total_loss2 / data_train_len
    av_loss3 = total_loss3 / data_train_len
    av_loss_Linf = total_loss_Linf / data_train_len
    av_loss = total_loss / data_train_len
    av_regularized_loss = total_regularized_loss / data_train_len
    return av_loss1, av_loss2, av_loss3, av_loss_Linf, av_loss, av_regularized_loss, total_num_traj


def loss_test(params, max_shifts_to_stack, W, b, suffix='test'):

    X = np.loadtxt('%s_%s_x.csv' % (params['data_name'], suffix), delimiter=',')
    X_stacked, num_traj_test = stack_data(X, max_shifts_to_stack, params['len_time'])
    y, g_list = ApplyKoopmanNetOmegasFull(X_stacked, W, b, params['delta_t'], params['num_shifts'],
                                          params['num_shifts_middle'], params['num_real'], params['num_complex_pairs'],
                                          params['num_encoder_weights'], params['num_omega_weights'],
                                          params['num_decoder_weights'])
    loss1, loss2, loss3, loss_Linf, loss = define_loss(X_stacked, y, g_list, params, W, b)
    loss_L1, loss_L2, regularized_loss = define_regularization(params, W, b, loss)
    print("test file has %d trajectories" % num_traj_test)

    return loss1, loss2, loss3, loss_Linf, loss, regularized_loss
