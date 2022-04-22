import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame,Series
from sklearn.linear_model import LinearRegression

# 读取数据
datafile = './dataset/housing.xlsx'
data = pd.read_excel(datafile)
examDf = np.mat(data)
m = len(examDf)
min = examDf.min(axis=0)
max = examDf.max(axis=0)
# 标准化
for i in range(0,14):
    examDf[:, i] = (examDf[:, i] - min[0,i]) / (max[0,i] - min[0,i])

#分离X和Y
X = examDf[:,0:13]
Y = examDf[:,13:14]

# print(X)
# print(Y)

# 代价函数
def computeCostDeri(X, Y, theta):
    inner = np.dot(X, theta.T) - Y
    return np.sum(inner) / len(X)
def computeCost(X,Y,theta):
    inner = np.power(np.dot(X, theta.T) - Y, 2)
    return np.sum(inner) / (2 * len(X))
def SGD(X, Y, theta, alpha, turn):
    error = 0
    m = len(X)
    ret = np.zeros(turn)
    for t in range(0, turn):
        row = random.randint(0, m - 1)
        cur = X[row:row+1,:]
        hx = np.dot(cur, theta.T)
        ret[t] = np.power(hx - Y[row, 0], 2) / 2
        error = ret[t];
        for i in range(0, 13):
            theta[0,i] -= (hx - Y[row,0]) * cur[0,i] * alpha
        if error <= 1e-9:
            break
    return ret

alpha = 0.001
theta = np.zeros((1, 13))
turn = 5000
ret = SGD(X, Y, theta, alpha, turn)
print(ret)
# SGD图表
numx = []
numy = []
step = 50
for i in range(0, 100):
    numx.append(i * step)
    cur = ret[i * step]
    numy.append(cur)
plt.xlabel('iteration')
plt.ylabel('cost')
plt.plot(numx, numy)
plt.title('SGD')
plt.show()