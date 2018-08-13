import numpy as np
import scipy.optimize as so
import IrCostFunction as icf

def oneVsAll(X,y,num_labels,lam):
    m,n=X.shape

    all_theta=np.zeros((num_labels,n+1))

    X=np.c_[np.ones(m),X]
    initial_theta=np.zeros(n+1)
    for i in range(num_labels):
        all_theta[i,:]=so.fmin_bfgs(icf.IrCostFunction,initial_theta,fprime=icf.Grad,args=(X,np.int32(y == i),lam))

    return all_theta