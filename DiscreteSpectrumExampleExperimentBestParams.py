import copy

import training

params = {}

# settings related to dataset
params['data_name'] = 'DiscreteSpectrumExample'
params['data_train_len'] = 1
params['len_time'] = 51
n = 2  # dimension of system (and input layer)
num_initial_conditions = 5000  # per training file
params['delta_t'] = 0.02

# settings related to saving results
params['folder_name'] = 'exp1_best'

# settings related to network architecture
params['num_real'] = 2
params['num_complex_pairs'] = 0
params['num_evals'] = 2
k = params['num_evals']  # dimension of y-coordinates
w = 30
params['widths'] = [2, w, w, k, k, w, w, 2]
wo = 10
params['hidden_widths_omega'] = [wo, wo, wo]

# settings related to loss function
params['num_shifts'] = 30
params['num_shifts_middle'] = params['len_time'] - 1
max_shifts = max(params['num_shifts'], params['num_shifts_middle'])
num_examples = num_initial_conditions * (params['len_time'] - max_shifts)
params['recon_lam'] = .1
params['Linf_lam'] = 10 ** (-7)
params['L1_lam'] = 0.0
params['L2_lam'] = 10 ** (-15)
params['auto_first'] = 0

# settings related to the training
params['num_passes_per_file'] = 15 * 6 * 10
params['num_steps_per_batch'] = 2
params['learning_rate'] = 10 ** (-3)
params['batch_size'] = 256
steps_to_see_all = num_examples / params['batch_size']
params['num_steps_per_file_pass'] = (int(steps_to_see_all) + 1) * params['num_steps_per_batch']

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
    training.main_exp(copy.deepcopy(params))
