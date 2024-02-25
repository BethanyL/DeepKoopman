
numICs = 5000;
filenamePrefix = 'DiscreteSpectrumExample';

x1range = [-.5, .5];
x2range = x1range;
tSpan = 0:0.02:1;
mu = -0.05;
lambda = -1;

seed = 1;
X_test = DiscreteSpectrumExampleFn(x1range, x2range, round(.1*numICs), tSpan, mu, lambda, seed);
filename_test = strcat(filenamePrefix, '_test_x.csv');
dlmwrite(filename_test, X_test, 'precision', '%.14f')

seed = 2;
X_val = DiscreteSpectrumExampleFn(x1range, x2range, round(.2*numICs), tSpan, mu, lambda, seed);
filename_val = strcat(filenamePrefix, '_val_x.csv');
dlmwrite(filename_val, X_val, 'precision', '%.14f')

for j = 1:3
	seed = 2+j;
	X_train = DiscreteSpectrumExampleFn(x1range, x2range, round(.7*numICs), tSpan, mu, lambda, seed);
	filename_train = strcat(filenamePrefix, sprintf('_train%d_x.csv',j));
	dlmwrite(filename_train, X_train, 'precision', '%.14f')
end


