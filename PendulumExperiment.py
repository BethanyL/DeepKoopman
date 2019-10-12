import copy
import numpy as np
import random as r

import training

params = {}

# settings related to dataset
params['data_name'] = 'Pendulum'
params['len_time'] = 51
n = 2  # dimension of system (and input layer)
num_initial_conditions = 5000  # per training file
params['delta_t'] = 0.02

# settings related to saving results
params['folder_name'] = 'exp2'

# settings related to network architecture
params['num_real'] = 0
params['num_complex_pairs'] = 1
params['num_evals'] = 2
k = params['num_evals']  # dimension of y-coordinates

# defaults related to initialization of parameters
params['dist_weights'] = 'dl'
params['dist_weights_omega'] = 'dl'

# settings related to loss function
params['num_shifts'] = 30
params['num_shifts_middle'] = params['len_time'] - 1
max_shifts = max(params['num_shifts'], params['num_shifts_middle'])
num_examples = num_initial_conditions * (params['len_time'] - max_shifts)
params['recon_lam'] = .001
params['L1_lam'] = 0.0
params['auto_first'] = 1

# settings related to training
params['num_passes_per_file'] = 15 * 6 * 50
params['num_steps_per_batch'] = 2
params['learning_rate'] = 10 ** (-3)

# settings related to timing
params['max_time'] = 6 * 60 * 60  # 6 hours
params['min_5min'] = .25
params['min_20min'] = .02
params['min_40min'] = .002
params['min_1hr'] = .0002
params['min_2hr'] = .00002
params['min_3hr'] = .000004
params['min_4hr'] = .0000005
params['min_halfway'] = 1

for count in range(200):  # loop to do random experiments
    params['data_train_len'] = r.randint(3, 6)
    params['batch_size'] = int(2 ** (r.randint(7, 9)))
    steps_to_see_all = num_examples / params['batch_size']
    params['num_steps_per_file_pass'] = (int(steps_to_see_all) + 1) * params['num_steps_per_batch']
    params['L2_lam'] = 10 ** (-r.randint(13, 14))
    params['Linf_lam'] = 10 ** (-r.randint(7, 10))

    d = r.randint(1, 2)
    if d == 1:
        wopts = np.arange(100, 200, 5)
        w = wopts[r.randint(0, len(wopts) - 1)]
        params['widths'] = [n, w, k, k, w, n]
    elif d == 2:
        wopts = np.arange(30, 90, 5)
        w = wopts[r.randint(0, len(wopts) - 1)]
        params['widths'] = [n, w, w, k, k, w, w, n]

    do = r.randint(1, 2)
    if do == 1:
        wopts = np.arange(140, 190, 5)
        wo = wopts[r.randint(0, len(wopts) - 1)]
        params['hidden_widths_omega'] = [wo, ]
    elif do == 2:
        wopts = np.arange(10, 55, 5)
        wo = wopts[r.randint(0, len(wopts) - 1)]
        params['hidden_widths_omega'] = [wo, wo]

    training.main_exp(copy.deepcopy(params))
