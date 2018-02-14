import copy
import random as r

import training as t

params = {}
params['data_name'] = 'DiscreteSpectrumExample'
params['folder_name'] = 'exp1'
params['distributionW'] = 'dl'
params['distributionW_omega'] = 'dl'
params['batch_flag'] = 0
params['opt_alg'] = 'adam'
params['dropout_rate'] = 1.0

params['num_passes_per_file'] = 15 * 4
params['num_steps_per_batch'] = 2
params['len_time'] = 51
params['max_time'] = 4 * 60 * 60  # 4 hours
params['delta_t'] = 0.02
L = 2

params['num_shifts'] = 3
params['mid_shift_lam'] = 1.0
params['data_train_len'] = 10
numICs = 29400  # per training file (10 training data files)
params['minHalfway'] = 10 ** (-5)

for count in range(200):
    params['num_shifts_middle'] = r.randint(5, params['len_time'] - 1)
    max_shifts = max(params['num_shifts'], params['num_shifts_middle'])
    num_examples = numICs * (params['len_time'] - max_shifts)
    params['batch_size'] = int(2 ** (r.uniform(6, 9)))
    steps_to_see_all = num_examples / params['batch_size']
    params['num_steps_per_file_pass'] = (int(steps_to_see_all) + 1) * params['num_steps_per_batch']
    params['L2_lam'] = 10 ** (-r.uniform(12, 18))
    params['Linf_lam'] = 10 ** (-6)
    params['L1_lam'] = 10 ** (-r.uniform(14, 18))
    params['lr'] = 10 ** (-3)
    w = 100
    params['widths'] = [2, w, w, w, L, L, w, w, w, 2]
    params['widths_omega'] = [2, w, w, w, 1]
    params['d'] = len(params['widths'])

    t.main_exp(copy.deepcopy(params))
