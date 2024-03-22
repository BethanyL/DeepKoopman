import torch
import simpleKoopmanAutoencoder as kp
import dataloader as dl
import matplotlib.pyplot as plt

params = {}

#Settings related to dataset
params['lenTime'] = 51
params['deltaT'] = 0.02

#Settings related to loss function
params['numShifts'] = 30
params['reconLam'] = .1
params['LinfLam'] = 10 ** (-7)
params['L2Lam'] = 10 ** (-15)

#Settings related to Network Architecture
params['inputSize'] = 2
params['hiddenSize'] = 30
params['hiddenLayer'] = 2
params['latentSize'] = 2


testDataset = dl.TrajectoryDataset('Koopman (Local)/data/DiscreteSpectrumExample_train1_x.csv')


trajectoryX = [tensor[0].tolist() for tensor in testDataset[0][1]]
trajectoryY = [tensor[1].tolist() for tensor in testDataset[0][1]]

plt.plot(trajectoryX, trajectoryY)
plt.show()

testModel = kp.SimpleKoopmanNeuralNetwork(params)

testModel.load_state_dict(torch.load('trainedModel.pt'))
testModel = testModel.to(torch.float64)

trajectoryPrediction, latentTrajectoryPrediction = testModel(testDataset[0][0])

trajectoryX = [tensor[0].tolist() for tensor in trajectoryPrediction]
trajectoryY = [tensor[1].tolist() for tensor in trajectoryPrediction]

plt.plot(trajectoryX, trajectoryY)
plt.show()

testLoss = kp.LossFunction(params)

print("Loss = ", testLoss(testDataset[0][1], testModel))

testTrajectoryPrediction, latentTrajectoryPrediction = testModel(torch.randn(2, dtype= torch.float64))

testTrajectoryX = [tensor[0].tolist() for tensor in trajectoryPrediction]
testTrajectoryY = [tensor[1].tolist() for tensor in trajectoryPrediction]

plt.plot(testTrajectoryX, testTrajectoryY, label='Line')
plt.scatter(testTrajectoryX, testTrajectoryY, color='red')
plt.show()