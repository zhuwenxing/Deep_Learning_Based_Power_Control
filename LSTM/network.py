import numpy as np 

import torch as t 
import torch.nn as nn 
from torch.nn.modules import Sequential

class RNN(nn.Module):
    def __init__(self,K):
        super(RNN,self).__init__()
        self.K = K
        self.lstm = nn.LSTM(
            input_size = self.K,
            hidden_size = 100,
            num_layers= 2,
            batch_first = True
        )
        self.fc = nn.Linear(100,self.K)
        self.sigmoid = nn.Sigmoid()
    def forward(self,csi):
        output,(h_n,c_n) = self.lstm(csi,None)

        z = output[:,-1,:]
        z = self.fc(z)
        z = self.sigmoid(z)
        return z