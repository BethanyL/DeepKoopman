
numTest_ICs = 5000;
filenamePrefix = 'FluidFlowBox';

x1range = [-1.1,1.1];
x2range = [-1.1,1.1];
x3range = [0,2.42];
tSpan = 0:0.01:1;
max_x3 = 2.5;

seed = 1;
X_test = FluidFlowBoxFn(x1range, x2range, x3range, numTest_ICs, tSpan, seed, max_x3);
filename_test = strcat(filenamePrefix, '_test_x.csv');
dlmwrite(filename_test, X_test, 'precision', '%.14f')

seed = 2;
X_val = FluidFlowBoxFn(x1range, x2range, x3range, numTest_ICs, tSpan, seed, max_x3);
filename_val = strcat(filenamePrefix, '_val_x.csv');
dlmwrite(filename_val, X_val, 'precision', '%.14f')

for j = 1:4
	seed = 2+j;
	X_train = FluidFlowBoxFn(x1range, x2range, x3range, numTest_ICs, tSpan, seed, max_x3);
	filename_train = strcat(filenamePrefix, sprintf('_train%d_x.csv', j));
	dlmwrite(filename_train, X_train, 'precision', '%.14f')
end
