import numpy as np
import oneVsAll as ova
import scipy.io as sio

data = sio.loadmat(r'G:\2014斯坦福大学机器学习mkv视频\机器学习课程2014源代码\mlclass-ex3-jin\ex3data1.mat')
X = data['X']
y = data['y']
y[np.where(y==10)]=0

all_theta=ova.oneVsAll(X,y,10,100)

def sigma(z):
    return 1/(1+np.e**-z)

def predictOneVsAll(all_theta,X):
    m=X.shape[0]

    X=np.c_[np.ones(m),X]

    return np.argmax(sigma(np.dot(all_theta,X.reshape(-1,1))),0),(sigma(np.dot(all_theta,X.reshape(-1,1))),0)

predictOneVsAll(all_theta,X[[1],:])

