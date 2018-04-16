
numTest_ICs = 5000;
filenamePrefix = 'FluidFlowOnAttractor';

Rrange = [0,1.1];
Trange = [0,2*pi];
tSpan = 0:0.05:6;


seed = 1;
X_test = FluidFlowOnAttractorFn(Rrange, Trange, numTest_ICs, tSpan, seed);
filename_test = strcat(filenamePrefix, '_test_x.csv');
dlmwrite(filename_test, X_test, 'precision', '%.14f')

seed = 2;
X_val = FluidFlowOnAttractorFn(Rrange, Trange, numTest_ICs, tSpan, seed);
filename_val = strcat(filenamePrefix, '_val_x.csv');
dlmwrite(filename_val, X_val, 'precision', '%.14f')

for j = 1:3
	seed = 2+j;
	X_train = FluidFlowOnAttractorFn(Rrange, Trange, numTest_ICs, tSpan, seed);
	filename_train = strcat(filenamePrefix, sprintf('_train%d_x.csv', j));
	dlmwrite(filename_train, X_train, 'precision', '%.14f')
end
