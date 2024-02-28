import torch
import torch.nn as nn


from simpleKoopmanAutoencoder import *

c = SimpleKoopmanNeuralNetwork(2, 30, 3, 3)


input = torch.randn(2)

_, _, cNext, cEncoded = c(input)

fun = SimpleLossFunction(1,1,1)


xNextList = c.multiForward(input,2)


#print(c.state_dict())

print(fun(c, input, cEncoded, xNextList, cNext, c(cNext)[3]))
