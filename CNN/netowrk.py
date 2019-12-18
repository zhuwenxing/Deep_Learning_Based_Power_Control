import numpy as np 

import torch as t 
import torch.nn as nn 
from torch.nn.modules import Sequential

class ConvLayer(nn.Module):
    def __init__(self, in_channels,out_channels,kernel_size,stride):
        super(ConvLayer,self).__init__()
        zero_padding = int(np.floor(kernel_size / 2))
        self.zero_pad = nn.ZeroPad2d(zero_padding)
        self.conv2d = nn.Conv2d(in_channels,out_channels,kernel_size,stride)
        self.relu = nn.ReLU()
    def forward(self,x):
        out = self.zero_pad(x)
        out = self.conv2d(out)
        out = self.relu(out)
        return out

class CNN_Net(nn.Module):
    def __init__(self,K):
        super(CNN_Net,self).__init__()
        self.K = K
        self.con_block = nn.Sequential(
            ConvLayer(1,8,kernel_size=3,stride=1),
            *(ConvLayer(8,8,kernel_size=3,stride=1) for i in range(5)),
            ConvLayer(8,1,kernel_size=3,stride=1)
        )
        self.fc = nn.Linear(self.K**2,self.K)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,csi):

        z = self.con_block(csi)
        z = z.view(z.size(0),-1)
        z = self.fc(z)
        z = self.sigmoid(z)

        return z

