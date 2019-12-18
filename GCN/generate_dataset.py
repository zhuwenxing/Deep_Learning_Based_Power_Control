import scipy.io as sio                     # import scipy.io for .mat file I/O 
import numpy as np                         # import numpy
import matplotlib.pyplot as plt            # import matplotlib.pyplot for figure plotting
import function_wmmse_powercontrol as wf
import tensorflow as tf

K = 20                  # number of users
num_H = 2000           # number of training samples
num_test = 500            # number of testing  samples
training_epochs = 50      # number of training epochs
trainseed = 0              # set random seed for training set
testseed = 7               # set random seed for test set
print('Gaussian IC Case: K=%d, Total Samples: %d, Total Iterations: %d\n'%(K, num_H, training_epochs))
var_db = 0
var = 1 / 10 ** (var_db / 10)

import time
def generate_wGaussian(K, num_H, var_noise=1, Pmin=0, seed=2017):
    print('Generate Data ... (seed = %d)' % seed)
    np.random.seed(seed)
    Pmax = 1
    Pini = Pmax*np.ones(K)
    #alpha = np.random.rand(num_H,K)
    alpha = np.ones((num_H,K))
    #var_noise = 1
    X=np.zeros((K**2,num_H))
    Y=np.zeros((K,num_H))
    total_time = 0.0
    for loop in range(num_H):
        CH = 1/np.sqrt(2)*(np.random.randn(K,K)+1j*np.random.randn(K,K))
        H=abs(CH)
        X[:,loop] = np.reshape(H, (K**2,), order="F")
        H=np.reshape(X[:,loop], (K,K), order="F")
        mid_time = time.time()
        Y[:,loop] = wf.WMMSE_sum_rate2(Pini, alpha[loop,:], H, Pmax, var_noise)
        total_time = total_time + time.time() - mid_time
    
    # print("wmmse time: %0.2f s" % total_time)
    return X, Y, alpha, total_time

Xtrain, Ytrain, Atrain, wtime = generate_wGaussian(K, num_H, seed=trainseed, var_noise = var)
X, Y, A, wmmsetime = generate_wGaussian(K, num_test, seed=testseed, var_noise = var)
Xtrain = Xtrain.transpose()
X = X.transpose()
Ytrain = Ytrain.transpose()
Y = Y.transpose()
print(Xtrain.shape, Ytrain.shape)

Xtrain = Xtrain.reshape((-1,K,K))
X = X.reshape((-1, K, K))

#保存为mat文件

sio.savemat(f"dataset/train_{K}.mat", {"Xtrain": Xtrain, "Ytrain": Ytrain, "Atrain": Atrain})
sio.savemat(f"dataset/test_{K}.mat", {"X": X, "Y": Y, "A": A})


# 测试

