import copy
import random as r

import training

params = {}

# settings related to dataset
params['data_name'] = 'Pendulum'
params['len_time'] = 51
n = 2
params['data_train_len'] = 10
num_initial_conditions = 29400  # per training file (10 training data files)
params['delta_t'] = 0.02

# settings related to saving results
params['folder_name'] = 'exp2'

# settings related to network architecture
L = 2

# settings related to loss function
params['num_shifts'] = 3
params['recon_lam'] = .001

# settings related to training
params['num_passes_per_file'] = 15 * 6
params['num_steps_per_batch'] = 2
params['learning_rate'] = 10 ** (-3)

# settings related to timing
params['max_time'] = 6 * 60 * 60  # 6 hours
params['min_5min'] = 10 ** (-2)
params['min_20min'] = 10 ** (-2.2)
params['min_40min'] = 10 ** (-3)
params['min_1hr'] = 10 ** (-4)
params['min_2hr'] = 10 ** (-6.4)
params['min_3hr'] = 10 ** (-6.7)
params['min_4hr'] = 10 ** (-7.0)
params['min_halfway'] = 10 ** (-6.7)

for count in range(200):
    params['num_shifts_middle'] = r.randint(48, 50)
    max_shifts = max(params['num_shifts'], params['num_shifts_middle'])
    num_examples = num_initial_conditions * (params['len_time'] - max_shifts)
    params['batch_size'] = int(2 ** (r.uniform(6, 8.5)))
    steps_to_see_all = num_examples / params['batch_size']
    params['num_steps_per_file_pass'] = (int(steps_to_see_all) + 1) * params['num_steps_per_batch']

    params['L2_lam'] = 10 ** (-r.uniform(12, 16))
    params['Linf_lam'] = 10 ** (-r.uniform(7, 10))
    params['L1_lam'] = 10 ** (-r.uniform(14, 18))

    w = r.randint(40, 100)
    params['widths'] = [n, w, w, w, w, L, L, w, w, w, w, n]
    params['widths_omega'] = [2, w, w, w, w, 1]

    training.main_exp(copy.deepcopy(params))
