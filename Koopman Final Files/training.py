import simpleKoopmanAutoencoder as kp
import dataloader as dl
import torch


def pytorchTraining(koopmanModel, koopmanLoss, optimizer, numEpochs, dataloader):
  
  koopmanModel.train()
  
  
  for epoch in range(numEpochs):

    runningLoss = 0
    
    for input, target in dataloader:
      optimizer.zero_grad()


      loss = koopmanLoss(target, koopmanModel)

      loss.backward()
      optimizer.step()

      runningLoss += loss.item()

    print(f'Epoch {epoch}, loss = {runningLoss}')

  torch.save(koopmanModel.state_dict(), 'trainedModel.pt')


if __name__ == '__main__':
#Initializing the parameters dictionary
    params = {}

    #Settings related to dataset
    params['lenTime'] = 51
    params['deltaT'] = 0.02

    #Settings related to loss function
    params['numShifts'] = 30
    params['reconLam'] = .1
    params['LinfLam'] = 0
    params['L2Lam'] = 0

    #Settings related to Network Architecture
    params['inputSize'] = 2
    params['hiddenSize'] = 30
    params['hiddenLayer'] = 2
    params['latentSize'] = 2


    testKoopmanModel = kp.SimpleKoopmanNeuralNetwork(params)

    testKoopmanModel = testKoopmanModel.to(torch.float64)

    testDataset = dl.TrajectoryDataset('Koopman (Local)/data/DiscreteSpectrumExample_train1_x.csv')

    testDataloader = dl.getDataLoader(testDataset)

    testLoss = kp.LossFunction(params)

    pytorchTraining(testKoopmanModel, testLoss, torch.optim.Adam(testKoopmanModel.parameters(), lr= 0.001), 100, testDataloader)