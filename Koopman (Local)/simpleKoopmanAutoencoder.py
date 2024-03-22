import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SimpleKoopmanNeuralNetwork(nn.Module):
    def __init__(self, inputSize, hiddenSize, hiddenLayer, latentSize):
        super(SimpleKoopmanNeuralNetwork, self).__init__()
        
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

    def forward(self, x):

        xEncoded = self.Decoder(self.Encoder(x))

        y = self.Encoder(x)
        yNext = self.K(y)
        xNext = self.Decoder(yNext)

        return y, yNext, xNext, xEncoded
    
    def multiForward(self, x, numShift):
        xNextList = []

        for _ in range(numShift):
            _, _, xNext, _ = self.forward(x)
            xNextList.append(xNext.clone())
            x = xNext

        return xNextList
    
    

class SimpleLossFunction(nn.Module):
    def __init__(self, alpha1, alpha2, alpha3):
        super().__init__()
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3

    def lossRecon(self, x, xEncoded):
        
        return F.mse_loss(x, xEncoded)
    
    def lossPred(self, x, xNextList):

        return torch.mean(torch.stack([F.mse_loss(x, xNext) for xNext in xNextList]))

    def lossInf(self, x, xEncoded, xNext, xNextEncoded):

        return torch.linalg.vector_norm(x - xEncoded, ord = np.inf) + torch.linalg.vector_norm(xNext - xNextEncoded, ord = np.inf)

    def lossLin(self, x, xNextList):
        
        return torch.mean(torch.stack([F.mse_loss(x, xNext) for xNext in xNextList]))


    def lossWeight(self, model):

        lossWeight = 0

        for key, item in model.state_dict().items():
            parts = key.split(".")

            if parts[0] in ["Encoder", "Decoder"] and parts[-1] == "weight":

                lossWeight += torch.linalg.matrix_norm(item)

        return lossWeight

    def forward(self, model, x, xEncoded, xNextList, xNext, xNextEncoded):

        return self.alpha1 * (self.lossRecon(x, xEncoded) + self.lossPred(x, xNextList)) + self.alpha2 * self.lossInf(x, xEncoded, xNext, xNextEncoded) + self.alpha3 * self.lossWeight(model)