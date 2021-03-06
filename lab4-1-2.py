import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear=nn.Linear(3,1)
    def forward(self,x):
        return self.linear(x)

#data
x1_train=torch.FloatTensor([[73],[93],[89],[96], [73]])
x2_train=torch.FloatTensor([[80],[88],[91],[98],[66]])
x3_train=torch.FloatTensor([[75],[93],[90],[100],[70]])
x_train=torch.cat([x1_train,x2_train,x3_train],dim=1)
print(x_train)
y_train=torch.FloatTensor([[152],[185],[180],[196],[142]])

#model initialize
# W=torch.zeros((3,1),requires_grad=True)
# b=torch.zeros(1,requires_grad=True)
model=MultivariateLinearRegressionModel()

#set optimizer
#optimizer=optim.SGD([W,b], lr=1e-5)
optimizer=optim.SGD(model.parameters(),lr=1e-5)

nb_epochs=10000
for epoch in range(nb_epochs+1):
    #calc H(x)
    #hypothesis=x_train.matmul(W)+b
    prediction=model(x_train)

    #calc Cost
    #cost=torch.mean((hypothesis-y_train)**2)
    cost=F.mse_loss(prediction,y_train)

    #imporve H(x) with cost
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    print('Epoch {:4d}/{} Cost: {:.6f}'.format(epoch, nb_epochs, cost.item()))

