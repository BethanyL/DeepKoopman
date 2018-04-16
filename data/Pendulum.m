
numICs = 5000;
filenamePrefix = 'Pendulum';

x1range = [-3.1,3.1];
x2range = [-2, 2];
tSpan = 0:0.02:1;

max_potential = .99;

seed = 1;
X_test = PendulumFn(x1range, x2range, round(.1*numICs), tSpan, seed, max_potential);
filename_test = strcat(filenamePrefix, '_test_x.csv');
dlmwrite(filename_test, X_test, 'precision', '%.14f')

seed = 2;
X_val = PendulumFn(x1range, x2range, round(.2*numICs), tSpan, seed, max_potential);
filename_val = strcat(filenamePrefix, '_val_x.csv');
dlmwrite(filename_val, X_val, 'precision', '%.14f')

for j = 1:6
	seed = 2+j;
	X_train = PendulumFn(x1range, x2range, round(.7*numICs), tSpan, seed, max_potential);
	filename_train = strcat(filenamePrefix, sprintf('_train%d_x.csv', j));
	dlmwrite(filename_train, X_train, 'precision', '%.14f')
end
