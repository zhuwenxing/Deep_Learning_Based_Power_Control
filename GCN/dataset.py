import os
import numpy as np
import scipy.io as sio 
from torch.utils.data import Dataset,DataLoader


class MyDataset(Dataset):
    def __init__(self, filename):
        super(MyDataset).__init__()
        features = sio.loadmat(filename)
        self.direct_H = features["direct_H"]
        self.inter_to = features["inter_to"]
        self.inter_from = features["inter_from"]
        self.other_H = features["other_H"]
        self.alpha = features["alpha"]
        self.abs_H = features["abs_H"]
    def __len__(self):
        return len(self.direct_H)

    def __getitem__(self, idx):
        direct_H = self.direct_H[idx]
        inter_to = self.inter_to[idx]
        inter_from = self.inter_from[idx]
        other_H = self.other_H[idx]
        alpha = self.alpha[idx]
        abs_H = self.abs_H[idx]
        return (direct_H.astype(np.float32), inter_to.astype(np.float32), inter_from.astype(np.float32), other_H.astype(np.float32), alpha.astype(np.float32),abs_H.astype(np.float32))




## Testing
# 2019-12-18 测试成功
if __name__ == "__main__":
    filename = "dataset/feature_20.mat"
    traindatset = MyDataset(filename)
    traindataloader = DataLoader(traindatset, 50)
    for i, data in enumerate(traindataloader):
        print(data[4])
