import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#Data
x_train = torch.FloatTensor([[1],[2],[3]])
y_train = torch.FloatTensor([[1],[2],[3]])

#Model initialize
W=torch.zeros(1)
#Learning rate
lr=0.1

#repeat
nb_epochs=10
for epoch in range(nb_epochs+1):
    #Calc H(x)
    hypo = x_train*W

    #Calc cost gradient
    cost = torch.mean((hypo-y_train)**2)
    gradient = torch.sum((W*x_train-y_train)*x_train)

    print('Epoch {:4d}/{} W: {:.3f}, Cost: {:.6f}'.format(epoch, nb_epochs, W.item(), cost.item()))

    #improve H(x) with cost gradient
    W -= lr*gradient
