import numpy as np
import torch

m1=torch.FloatTensor([1,4])
m2=torch.FloatTensor([2,5])
m3=torch.FloatTensor([3,6])

print(torch.cat([m1,m2,m3],dim=0))
print(torch.stack([m1,m2,m3]))

