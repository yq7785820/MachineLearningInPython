import numpy as np
import scipy.io as sio


data = sio.loadmat(r'G:\2014斯坦福大学机器学习mkv视频\机器学习课程2014源代码\mlclass-ex3-jin\ex3data1.mat')
X = data['X']
y = data['y']
y[np.where(y == 10)] = 0
initial_theta=np.zeros(400)


def sigma(z):
    return 1 / (1 + np.e ** -z)


# 输入时每个class的theta为行向量，每个样本的X为行向量，labelY为行向量
def IrCostFunction(theta, X, y, lam):
    m = len(y)
    theta=theta.reshape(-1,1)
    h = sigma(np.dot(X, theta.reshape(-1,1)))
    temp = theta.copy()
    temp[0] = 0

    J = (-np.dot(y.T, np.log(h)) - np.dot((1 - y).T, np.log(1 - h))) / m + lam * np.dot(temp.T, temp) / 2 / m

    return J[0,0]


def Grad(theta, X, y, lam):
    m = len(y)
    theta = theta.reshape(-1, 1)
    temp = theta.copy()
    temp[0] = 0
    h = sigma(np.dot(X, theta))
    grad = np.dot(X.T, (h - y)) / m + lam * temp / m

    return grad.T[0]
