import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

data = sio.loadmat(r'G:\2014斯坦福大学机器学习mkv视频\机器学习课程2014源代码\mlclass-ex3-jin\ex3data1.mat')
X = data['X']
y = data['y']

xShow = X[np.random.random_integers(0,4999,100), :]


def displayData(X, example_width):
    (m, n) = X.shape
    example_height = int(n / example_width)
    display_rows = int(np.floor(np.sqrt(m)))
    display_columns = int(np.ceil(m / display_rows))
    pad = 1
    h1 = example_height + pad
    w1 = example_width + pad
    display_array = -np.ones((pad + display_rows * (example_height + pad), pad + display_columns * (example_width + pad)))
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
display_array=displayData(xShow, 20)
plt.imshow(display_array.T)
plt.show()
