import copy
import random as r

import training

params = {}

# settings related to dataset
params['data_name'] = 'DiscreteSpectrumExample'
params['len_time'] = 51
n = 2  # dimension of system (and input layer)
params['data_train_len'] = 10
num_initial_conditions = 29400  # per training file (10 training data files)
params['delta_t'] = 0.02

# settings related to saving results
params['folder_name'] = 'exp1'

# settings related to network architecture
w = 100  # width of hidden layers in encoder and decoder
k = 2  # dimension of y-coordinates 
params['widths'] = [n, w, w, w, k, k, w, w, w, n]
params['widths_omega'] = [2, w, w, w, 1]

# settings related to loss function
params['num_shifts'] = 3
params['Linf_lam'] = 10 ** (-6)

# settings related to the training
params['num_passes_per_file'] = 15 * 4
params['num_steps_per_batch'] = 2
params['learning_rate'] = 10 ** (-3)

# settings related to the timing
params['max_time'] = 4 * 60 * 60  # 4 hours
params['min_halfway'] = 10 ** (-5)

for count in range(200): # loop to do random experiments
    params['num_shifts_middle'] = r.randint(5, params['len_time'] - 1)
    max_shifts = max(params['num_shifts'], params['num_shifts_middle'])
    num_examples = num_initial_conditions * (params['len_time'] - max_shifts)
    params['batch_size'] = int(2 ** (r.uniform(6, 9)))
    steps_to_see_all = num_examples / params['batch_size']
    params['num_steps_per_file_pass'] = (int(steps_to_see_all) + 1) * params['num_steps_per_batch']

    params['L2_lam'] = 10 ** (-r.uniform(12, 18))
    params['L1_lam'] = 10 ** (-r.uniform(14, 18))

    training.main_exp(copy.deepcopy(params))
