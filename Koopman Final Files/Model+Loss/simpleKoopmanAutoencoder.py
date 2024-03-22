import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import dataloader

class SimpleKoopmanNeuralNetwork(nn.Module):
    def __init__(self, params):
        super(SimpleKoopmanNeuralNetwork, self).__init__()
        
        self.params = params

        inputSize, hiddenSize, hiddenLayer, latentSize = self.params['inputSize'], self.params['hiddenSize'], self.params['hiddenLayer'], self.params['latentSize']

        #Input layer of Encoder
        encoderLayers = [nn.Linear(inputSize, hiddenSize), nn.ReLU()]

        ###Define Encoder Layer
        for _ in range(hiddenLayer):
            encoderLayers.append(nn.Linear(hiddenSize, hiddenSize))
            encoderLayers.append(nn.ReLU())

        #Output layer of Encoder
        encoderLayers.append(nn.Linear(hiddenSize, latentSize))

        #Creating the Encoder Network
        self.Encoder = nn.Sequential(*encoderLayers)

        #Input layer of Decoder
        decoderLayers = [nn.Linear(latentSize, hiddenSize), nn.ReLU()]

        ###Define Decoder Layer
        for _ in range(hiddenLayer):
            decoderLayers.append(nn.Linear(hiddenSize, hiddenSize))
            decoderLayers.append(nn.ReLU())

        #Output layer of Decoder
        decoderLayers.append(nn.Linear(hiddenSize, inputSize))

        #Creating the Decoder Network
        self.Decoder = nn.Sequential(*decoderLayers)

        #Simple Koopman Auto-Encoder (Without Auxiliary Network)
        self.K = nn.Linear(latentSize, latentSize, bias=False)

    def forward(self, initialInput):
        #Take as input a 2 dimension tensor (initial State)

        #get the first encoded state (y1)
        encodedInitialInput = self.Encoder(initialInput)

        #First element of the latent trajectory is encoded input
        latentTrajectoryPrediction = [encodedInitialInput]

        #Alongside the trajectory, we multiply by the Koopman operator
        for m in range(49):
            latentTrajectoryPrediction.append(self.K(latentTrajectoryPrediction[-1]))

        #Decoding the trajectory
        trajectoryPrediction = [self.Decoder(latentState) for latentState in latentTrajectoryPrediction]
        
        #We get both the encoded trajectory and the decoded trajectory
        return trajectoryPrediction, latentTrajectoryPrediction
    
    

class LossFunction(nn.Module):
    def __init__(self, params):
        super().__init__()

        # We intialize the loss based on the parameters in the dictionnary
        self.alpha1 = params['reconLam']
        self.alpha2 = params['LinfLam']
        self.alpha3 = params['L2Lam']

        self.numShifts = params['numShifts']

    def forward(self, trajectoryInput, koopmanModel):
        
        #We get the prediction from the model we initialize the loss function with
        trajectoryPrediction, latentTrajectoryPrediction = koopmanModel(trajectoryInput[0])

        #We compute the auto-encoder loss for the initial state
        lossRecon = F.mse_loss(trajectoryInput[0], trajectoryPrediction[0])

        #We compute the Prediction loss
        lossPred = 0

        for m in range(self.numShifts):
            lossPred += F.mse_loss(trajectoryInput[m+1], trajectoryPrediction[m+1])

        lossPred *= (1/self.numShifts)

        #We compute the Linear loss on the whole trajectory using the encoder of the model
        lossLin = 0

        for m in range(49):
            lossLin += F.mse_loss(koopmanModel.Encoder(trajectoryInput[m+1]), latentTrajectoryPrediction[m+1])

        lossLin *= (1/49)

        #We compute the infinite loss
        lossInf = torch.linalg.vector_norm(trajectoryInput[0] - trajectoryPrediction[0]) + torch.linalg.vector_norm(trajectoryInput[1] - trajectoryPrediction[1]) 

        return self.alpha1*(lossRecon + lossPred) + lossLin + self.alpha2*lossInf
    


if __name__ == '__main__':

    #Initializing the parameters dictionary
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


    testKoopmanModel = SimpleKoopmanNeuralNetwork(params)

    testKoopmanModel = testKoopmanModel.to(torch.float64)

    testDataset = dataloader.TrajectoryDataset('Koopman (Local)/data/DiscreteSpectrumExample_train1_x.csv')

    testLoss = LossFunction(params)

    print(testLoss(testDataset[0][1], testKoopmanModel))

    print(testKoopmanModel(testDataset[0][0]))