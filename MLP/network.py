import numpy as np 
import torch as t 
import torch.nn as nn 
from torch.nn.modules import Sequential

class PAMLP(nn.Module):
    def __init__(self,K):
        super(PAMLP,self).__init__()
        self.hidden_layer_1 = nn.Linear(K**2,200)
        self.hidden_layer_2 = nn.Linear(200,200)
        self.hidden_layer_3 = nn.Linear(200,K)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self,csi):
        csi = csi.view(csi.size(0),-1) #平坦化 
        z = self.hidden_layer_1(csi)
        z = self.relu(z)
        z = self.hidden_layer_2(z)
        z = self.relu(z)
        z = self.hidden_layer_3(z)
        out = self.sigmoid(z)

        return out



