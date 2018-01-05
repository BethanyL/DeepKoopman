import helperfns as h
import random as r
import copy

params = {}
params['data_name'] = 'PendulumRing6Long'
params['foldername'] = 'exp167'
params['distributionW'] = 'dl'
params['distributionW_omega'] = 'dl'
params['batch_flag'] = 0
params['optalg'] = 'adam'
params['dropout_rate'] = 1.0

params['num_passes_per_file'] = 15*6
params['num_steps_per_batch'] = 2
params['lenT'] = 51
params['max_time'] = 6 * 60 * 60 # 6 hours
deltat = 0.02
params['deltat'] = deltat
l = 2

params['num_shifts'] = 3
params['mid_shift_lam'] = 1.0
params['data_train_len'] = 10
numICs = 29400 # per training file (10 training data files)

params['min5min'] = 10**(-2)
params['min20min'] = 10**(-2.2)
params['min40min'] = 10**(-3)
params['min1hr'] = 10**(-4)
params['min2hr'] = 10**(-6.4)
params['min3hr'] = 10**(-6.7)
params['min4hr'] = 10**(-7.0)
params['minHalfway'] = 10**(-6.7)

params['recon_lam'] = .001

for count in range(200):
	params['num_shifts_middle'] = r.randint(48,50)
	max_shifts = max(params['num_shifts'], params['num_shifts_middle'])
	num_examples = numICs * (params['lenT'] - max_shifts)
	params['batch_size'] = int(2**(r.uniform(6,8.5)))
	steps_to_see_all = num_examples/params['batch_size']
	params['num_steps_per_file_pass'] = (int(steps_to_see_all)+1)*params['num_steps_per_batch']
	params['l2_lam'] = 10**(-r.uniform(12,16))
	params['reg_inf_lam'] = 10**(-r.uniform(7,10))
	params['reg_lam'] = 10**(-r.uniform(14,18))
	params['lr'] = 10**(-3)
	w = r.randint(40,100)
	params['widths'] = [2, w, w, w, w, l, l, w, w, w, w, 2]
	params['widths_omega'] = [2, w, w, w, w, 1]
	params['d'] = len(params['widths'])


	h.main_exp(copy.deepcopy(params))
