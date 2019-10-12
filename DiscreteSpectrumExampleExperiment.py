import copy
import numpy as np
import random as r

import training

params = {}

# settings related to dataset
params['data_name'] = 'DiscreteSpectrumExample'
params['len_time'] = 51
n = 2  # dimension of system (and input layer)
num_initial_conditions = 5000  # per training file
params['delta_t'] = 0.02

# settings related to saving results
params['folder_name'] = 'exp1'

# settings related to network architecture
params['num_real'] = 2
params['num_complex_pairs'] = 0
params['num_evals'] = 2
k = params['num_evals']  # dimension of y-coordinates

# settings related to loss function
params['num_shifts'] = 30
params['num_shifts_middle'] = params['len_time'] - 1
max_shifts = max(params['num_shifts'], params['num_shifts_middle'])
num_examples = num_initial_conditions * (params['len_time'] - max_shifts)
params['recon_lam'] = .1
params['L1_lam'] = 0.0

# settings related to the training
params['num_passes_per_file'] = 15 * 6 * 10
params['num_steps_per_batch'] = 2
params['learning_rate'] = 10 ** (-3)

# settings related to the timing
params['max_time'] = 4 * 60 * 60  # 4 hours
params['min_5min'] = .5
params['min_20min'] = .0004
params['min_40min'] = .00008
params['min_1hr'] = .00003
params['min_2hr'] = .00001
params['min_3hr'] = .000006
params['min_halfway'] = .000006

for count in range(200):  # loop to do random experiments
    params['data_train_len'] = r.randint(1, 3)
    params['batch_size'] = int(2 ** (r.randint(7, 9)))
    steps_to_see_all = num_examples / params['batch_size']
    params['num_steps_per_file_pass'] = (int(steps_to_see_all) + 1) * params['num_steps_per_batch']
    if r.random() < .5:
        params['auto_first'] = 1
    else:
        params['auto_first'] = 0

    params['L2_lam'] = 10 ** (-r.randint(13, 15))
    if r.random() < .5:
        params['Linf_lam'] = 0.0
    else:
        params['Linf_lam'] = 10 ** (-r.randint(6, 10))

    d = r.randint(1, 4)
    if d == 1:
        wopts = np.arange(50, 160, 10)
        w = wopts[r.randint(0, len(wopts) - 1)]
        params['widths'] = [2, w, k, k, w, 2]
    elif d == 2:
        wopts = np.arange(15, 45, 5)
        w = wopts[r.randint(0, len(wopts) - 1)]
        params['widths'] = [2, w, w, k, k, w, w, 2]
    elif d == 3:
        wopts = np.arange(10, 25, 5)
        w = wopts[r.randint(0, len(wopts) - 1)]
        params['widths'] = [2, w, w, w, k, k, w, w, w, 2]
    elif d == 4:
        wopts = np.arange(10, 20, 5)
        w = wopts[r.randint(0, len(wopts) - 1)]
        params['widths'] = [2, w, w, w, w, k, k, w, w, w, w, 2]

    do = r.randint(1, 4)
    if do == 1:
        wopts = np.arange(20, 110, 10)
        wo = wopts[r.randint(0, len(wopts) - 1)]
        params['hidden_widths_omega'] = [wo, ]
    elif do == 2:
        wopts = np.arange(10, 25, 5)
        wo = wopts[r.randint(0, len(wopts) - 1)]
        params['hidden_widths_omega'] = [wo, wo]
    elif do == 3:
        wopts = np.arange(5, 20, 5)
        wo = wopts[r.randint(0, len(wopts) - 1)]
        params['hidden_widths_omega'] = [wo, wo, wo]
    elif do == 4:
        wopts = np.arange(5, 15, 5)
        wo = wopts[r.randint(0, len(wopts) - 1)]
        params['hidden_widths_omega'] = [wo, wo, wo, wo]

    training.main_exp(copy.deepcopy(params))
