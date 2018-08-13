# MachineLearningInPython
### 一对多的逻辑回归（One vs All）
样本是像素为20*20px的数字，共5000个样本，对应数字为0-9


#### 显示图像
首先是读取数据，对于原始的mat格式文件，可以用scipy.io中的loadmat函数读取，并随机选择其中的100个样本进行展示：
```
data = sio.loadmat('ex3data1.mat')
X = data['X']
y = data['y']
xShow = X[np.random.random_integers(0,4999,100), :]
```
然后定义一个函数，用于将样本数据格式整理为可以被imshow函数接收的格式：

```
def displayData(X, example_width, display_rows,display_columns, example_height ):
'''
参数分别为样本数据、每个样本整理后的宽、高，总随机集的宽、高。
'''
    _pad = 1
    h1 = example_height + _pad
    w1 = example_width + _pad
    display_array = -np.ones((_pad + display_rows * (example_height + _pad), _pad + display_columns * (example_width + pad)))
    curr_ex = 0
    for j in range(display_rows):
        for i in range(display_columns):
            if (curr_ex > m - 1):
                break
            max_val = np.max(np.abs(X[curr_ex, :]))
            display_array[(pad + j * h1):(pad + j * h1 + example_height),(pad + i * w1):(pad + i * w1 + example_width )] = np.reshape(X[curr_ex, :], (example_height,example_width)) / max_val
            curr_ex+=1
        if (curr_ex > m-1):
            break
    return display_array
```
最后用matplotlib.pyplot的imshow函数即可将之显示：

```
display_array=displayData(xShow, 20，20，10，10)
plt.imshow(display_array.T)
plt.show()
```

#### 定义损失函数及其梯度函数
sigma函数：

```
def sigma(z):
    return 1 / (1 + np.e ** -z)
```

损失函数：

```
#输入时每个样本的theta为行向量，每个样本的X为行向量，labelY为行向量，下面的梯度函数同
def IrCostFunction(theta, X, y, lam):
    m = len(y)
    theta=theta.reshape(-1,1)
    h = sigma(np.dot(X, theta.reshape(-1,1)))
    temp = theta.copy()
    #bias无需参与正则化，将其对应的正则项置零，下面的梯度函数同
    temp[0] = 0
    J = (-np.dot(y.T, np.log(h)) - np.dot((1 - y).T, np.log(1 - h))) / m + lam * np.dot(temp.T, temp) / 2 / m
    return J[0,0]
    #返回的损失函数值应当是一个标量
```
梯度函数

```
def Grad(theta, X, y, lam):
    m = len(y)
    theta = theta.reshape(-1, 1)
    temp = theta.copy()
    temp[0] = 0
    h = sigma(np.dot(X, theta))
    grad = np.dot(X.T, (h - y)) / m + lam * temp / m
    return grad.T[0]
    #返回所有权重参数关于损失函数的梯度
```
#### 定义通过梯度下降算法求解theta的函数
> 如何利用逻辑回归解决多分类的问题，OneVsAll就是把当前某一类看成一类，其他所有类别看作一类，这样有成了二分类的问题了

我们知道需要解决的是0-9这十个数字的分类问题。
所以我们可以通过寻找10组不同的theta，来满足问题的需要。
对于确定某一数字n的分类时，我们可以讲等于n的labelY记为1，其余为0，这样就相当于将问题转化为二元分类问题：
```
def oneVsAll(X,y,num_labels,lam):
    m,n=X.shape
    all_theta=np.zeros((num_labels,n+1))
    X=np.c_[np.ones(m),X]
    initial_theta=np.zeros(n+1)
    for i in range(num_labels):
        all_theta[i,:]=so.fmin_bfgs(IrCostFunction,initial_theta,fprime=Grad,args=(X,np.int32(y == i),lam))
    return all_theta
```
#### 进行预测
每个数据都可根据不同的theta，来确定其属于某个数字的概率值，返回最大概率值对应的数字，作为预测值
```
def predictOneVsAll(all_theta,X):
    m=X.shape[0]
    X=np.c_[np.ones(m),X]
    return np.argmax(sigma(np.dot(all_theta,X.reshape(-1,1))),0)
```
