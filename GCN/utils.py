import numpy as np
import scipy.io as sio

def IC_sum_rate(H, alpha, p, var_noise):
    H = np.square(H)
    fr = np.diag(H)*p
    ag = np.dot(H,p) + var_noise - fr
    y = np.sum(alpha * np.log2(1+fr/ag) )
    return y
def np_sum_rate(X,Y,alpha):
    avg = 0
    n = X.shape[0]
    for i in range(n):
        avg += IC_sum_rate(X[i,:,:],alpha[i,:],Y[i,:],1)/n
    return avg




if __name__ == "__main__":
    test_data = sio.loadmat("dataset/test_20.mat")
    X = test_data["X"]
    Y = test_data["Y"]
    A = test_data["A"]
    rate = np_sum_rate(X, Y, A)
    print(rate)
