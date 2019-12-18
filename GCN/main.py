import os 

import torch as t 
import torch.nn as nn
import torch.optim as optim
from network import *
from loss import RateLoss
from torch.utils.data import DataLoader
from dataset import MyDataset
from utils import *

# 参数定义

K = 20
num_H = 2000
var_db = 0
var = 1 / 10 ** (var_db / 10)
layer_num = 5
batch_size = 50
epochs = 2000
lr = 0.001

# 计算测试集速率
test_data = sio.loadmat("dataset/test_20.mat")
X = test_data["X"]
Y = test_data["Y"]
A = test_data["A"]
rate_wmmse = np_sum_rate(X, Y, A)
print(f"rate_wmmse: {rate_wmmse}")
X = t.from_numpy(X.astype(np.float32))
Y = t.from_numpy(Y.astype(np.float32))

# 加载训练数据
train_data_path = "dataset/feature_20.mat"
train_data = MyDataset(train_data_path)
train_dataloader = DataLoader(train_data, batch_size)

# 加载测试数据
test_data_path = "dataset/feature_test_20.mat"
test_data = MyDataset(test_data_path)
test_dataloader = DataLoader(train_data, 500)
for i, data in enumerate(test_dataloader):
    test_data = data
    break

# 构建网络模型和准则
model = IGCNet(K,layer_num)
criterion = RateLoss(K, var)
optimizer = optim.Adam(model.parameters(),lr = lr)

for epoch in range(epochs):
    for i, data in enumerate(train_dataloader):
        direct_H, inter_to, inter_from, other_H, alpha, abs_H= data
        # 构建所需输入数据
        Xinterf = t.unsqueeze(inter_from,-1)
        Xintert = t.unsqueeze(inter_to,-1)
        Xdiag = t.unsqueeze(direct_H,-1)
        Xdiag_o = t.unsqueeze(other_H,-1)
        intensity = t.ones(batch_size, K, 1)
        w_alpha = t.unsqueeze(alpha,-1)
        optimizer.zero_grad()
        pred = model(Xinterf,Xintert,Xdiag,Xdiag_o,intensity,w_alpha)
        loss = criterion(abs_H, pred)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print(f"loss: {loss.item()}")
            direct_H, inter_to, inter_from, other_H, alpha, abs_H= test_data
            Xinterf = t.unsqueeze(inter_from,-1)
            Xintert = t.unsqueeze(inter_to,-1)
            Xdiag = t.unsqueeze(direct_H,-1)
            Xdiag_o = t.unsqueeze(other_H,-1)
            intensity = t.ones(500, K, 1)
            w_alpha = t.unsqueeze(alpha, -1)
            pred_test = model(Xinterf,Xintert,Xdiag,Xdiag_o,intensity,w_alpha)
            loss_test = criterion(abs_H, pred_test)
            rate_nn = -loss_test.item()
            print(f"rate_wmmse: {rate_wmmse}  rate_nn: {rate_nn}")
