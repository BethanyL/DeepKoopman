import copy
import numpy as np
import random as r

import training

params = {}

# settings related to dataset
params['data_name'] = 'FluidFlowBox'
params['len_time'] = 101
n = 3  # dimension of system (and input layer)
num_initial_conditions = 5000  # per training file
params['delta_t'] = 0.01

# settings related to saving results
params['folder_name'] = 'exp4'

# settings related to network architecture
params['num_real'] = 1
params['num_complex_pairs'] = 1
params['num_evals'] = 3
k = params['num_evals']  # dimension of y-coordinates

# defaults related to initialization of parameters
params['dist_weights'] = 'dl'
params['dist_weights_omega'] = 'dl'

# settings related to loss function
params['num_shifts'] = 30
params['num_shifts_middle'] = params['len_time'] - 1
max_shifts = max(params['num_shifts'], params['num_shifts_middle'])
num_examples = num_initial_conditions * (params['len_time'] - max_shifts)
params['recon_lam'] = .1
params['L1_lam'] = 0.0
params['auto_first'] = 1

# settings related to training
params['num_passes_per_file'] = 15 * 6 * 10
params['num_steps_per_batch'] = 2
params['learning_rate'] = 10 ** (-3)

# settings related to timing
params['max_time'] = 6 * 60 * 60  # 6 hours
params['min_5min'] = .45
params['min_20min'] = .005
params['min_40min'] = .0005
params['min_1hr'] = .00025
params['min_2hr'] = .00005
params['min_3hr'] = .000007
params['min_4hr'] = .000005
params['min_halfway'] = 1

for count in range(200):  # loop to do random experiments
    params['data_train_len'] = r.randint(1, 4)
    params['batch_size'] = int(2 ** (r.randint(7, 8)))
    steps_to_see_all = num_examples / params['batch_size']
    params['num_steps_per_file_pass'] = (int(steps_to_see_all) + 1) * params['num_steps_per_batch']
    params['L2_lam'] = 10 ** (-r.randint(13, 14))
    params['Linf_lam'] = 10 ** (-r.randint(7, 9))

    d = 1
    wopts = np.arange(50, 200, 5)
    w = wopts[r.randint(0, len(wopts) - 1)]
    params['widths'] = [n, w, k, k, w, n]

    do = r.randint(1, 2)
    if do == 1:
        wopts = np.arange(20, 135, 5)
        wo = wopts[r.randint(0, len(wopts) - 1)]
        params['hidden_widths_omega'] = [wo, ]
    elif do == 2:
        wopts = np.arange(10, 30, 5)
        wo = wopts[r.randint(0, len(wopts) - 1)]
        params['hidden_widths_omega'] = [wo, wo]

    training.main_exp(copy.deepcopy(params))
