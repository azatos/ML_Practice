import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class BinaryClassifier(nn.Module):
    def __inti__(self):
        super().__init__()
        self.linear=nn.Linear(8,1)
        self.sigmoid=nn.Sigmoid()
    def forward(self,x):
        return self.sigmoid(self.linear(x))

model=BinaryClassifier()

xy=np.loadtxt('./data/data-03-diabetes.csv',delimiter=',',dtype=np.float32)
x_data=xy[:,0:-1]
y_data=xy[:,[-1]]
x_train=torch.FloatTensor(x_data)
y_train=torch.FloatTensor(y_data)

W=torch.zeros((8,1),requires_grad=True)
b=torch.zeros(1, requires_grad=True)

optimizer=optim.SGD([W,b],lr=0.01)
nb_epochs=50000
for epoch in range(nb_epochs+1):
    hypothesis=torch.sigmoid(x_train.matmul(W) + b)
    cost=F.binary_cross_entropy(hypothesis,y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch%1000==0:
        prediction=hypothesis>=torch.FloatTensor([0.5])
        correct_prediction=prediction.float()==y_train
        accuracy=correct_prediction.sum().item()/len(correct_prediction)
        print('Epoch:{:3d}/{} Cost:{:.6f} Accuracy : {:2.2f}%'.format(epoch, nb_epochs, cost.item(), accuracy*100))
