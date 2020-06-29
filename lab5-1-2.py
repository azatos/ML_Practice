import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

xy=np.loadtxt('./data/data-03-diabetes.csv',delimiter=',',dtype=np.float32)
x_data=xy[:,0:-1]
y_data=xy[:,[-1]]
x_train=torch.FloatTensor(x_data)
y_train=torch.FloatTensor(y_data)

W=torch.zeros((8,1),requires_grad=True)
b=torch.zeros(1, requires_grad=True)

optimizer=optim.SGD([W,b],lr=1)
nb_epochs=10000
for epoch in range(nb_epochs+1):
    hypothesis=torch.sigmoid(x_train.matmul(W)+b)
    cost=F.binary_cross_entropy(hypothesis,y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch%1000==0:
        print('Epoch{:4d}/{} Cost:{:6f}'.format(epoch,nb_epochs,cost.item()))

hypothesis=torch.sigmoid(x_train.matmul(W)+b)
print(hypothesis[:5])
prediction=hypothesis>=torch.FloatTensor([0.5])
print(prediction[:5],y_train)
correct_prediction=prediction.float()==y_train
print(correct_prediction[:5])
accuracy=correct_prediction.sum().item()/len(correct_prediction)
print('Accuracy : {:2.2f}%'.format(accuracy*100))
