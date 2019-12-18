#导入模块
import numpy as np
import scipy.io as sio





# 定义参数
K = 20  #
num_H = 500


# 加载数据
data = sio.loadmat("dataset/test_20.mat")
Xtrain = data["X"]
Atrain = data["A"]

def extract_features(H,n,L,alpha):
    direct_H = np.zeros((n,L))
    inter_to = np.zeros((n,L,L))
    inter_from = np.zeros((n,L,L))
    other_H = np.zeros((n, L, L))
    abs_H = H
    for ii in range(n):
        diag_H = np.diag(H[ii,:,:])
        for jj in range(L):
            direct_H[ii,jj] = H[ii,jj,jj]
            inter_to[ii,jj,:] = H[ii,:,jj].T
            inter_to[ii,jj,jj] = 0
            inter_from[ii,jj,:] = H[ii,jj,:]
            inter_from[ii,jj,jj] = 0
            other_H[ii,jj,:] = diag_H
            other_H[ii,jj,jj] = 0
    return direct_H, inter_to, inter_from, other_H, alpha, abs_H

features = extract_features(Xtrain, num_H, K, Atrain)

# 输入参数中比较还有一个intensity 初始化时 为全1向量

sio.savemat(f"dataset/feature_test_{K}.mat", {"direct_H": features[0], "inter_to": features[1], "inter_from": features[2], "other_H": features[3], "alpha": features[4],"abs_H":features[5]})





## test

# import numpy as np
# import scipy.io as sio
# features = sio.loadmat("dataset/feature_20.mat")
