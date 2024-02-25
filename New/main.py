import torch
import torch.nn as nn
import torch.nn.functional as F


class KoopmanNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, auxiliary_size):
        super(KoopmanNeuralNetwork, self).__init__()
        
        ###Define Encoder
        self.Encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_size)
        )

        ###Define Decoder
        self.Decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)
        )

        ###Define Koopman Auxiliary Network
        #self.Auxiliary = nn.Sequential(
        #    nn.Linear(latent_size, auxiliary_size),
        #    nn.ReLU(),
        #    nn.Linear(auxiliary_size, auxiliary_size),
        #    nn.ReLU(),
        #    nn.Linear(auxiliary_size, latent_size),
        #)

        #Without Auxiliary Network
        self.K = nn.Linear(latent_size, latent_size, bias=False)

    def forward(self, x_k):
        y_k = self.Encoder(x_k)
        y_k1 = self.K(y_k)
        x_k1 = self.Decoder(y_k1)

        return x_k, y_k, y_k1, x_k1
    

class Loss(nn.Module):
    def __init__(self, alpha1, alpha2, alpha3):
        super().__init__()
    
    def forward(self, x_k, x_k1, y_k, y_k1):

        #lossRecon : loss MSE entre x_1 et  
        lossRecon = 

        #lossPred : moyenne des loss MSE entre x_1 et x_m+1
        lossPred = 

        #lossLin : moyenne des loss MSE entre ym+1 et K^m*y
        lossLin = 

        #lossInf : norm inf entre x_1 et ^x_1 + norm inf entre x_2 et ^x_2
        lossInf = torch.linalg.vector_norm(,ord=inf)
        
        #Somme des loss
        lossSum = self.alpha1*(lossRecon + lossPred) + lossLin + self.alpha2*lossInf + self.alpha3 * 



        return lossSum