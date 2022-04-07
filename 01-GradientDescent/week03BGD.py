import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame,Series
from sklearn.linear_model import LinearRegression

# 读取数据
datafile = 'D:\\yuecheng\\College\\pythonMachineLearning\\dataset\\housing.xlsx'
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

# 代价函数，求每个样本方程计算的值与实际值的差值
def computeCostDeri(X, Y, theta):
    return np.dot(X, theta) - Y
#代价函数cost，用于计算每轮迭代的代价
def computeCost(inner):
    inner = np.power(inner, 2)
    return np.sum(inner) / (2 * m)
# 批量梯度下降BGD
def BGD(X, Y, theta, alpha, turn):
    ret = np.zeros(turn)            #ret保存每次迭代计算的代价cost
    error = 0
    for t in range(0, turn):
        # 每步循环，求各样本损失函数inner
        inner = computeCostDeri(X,Y,theta)
        error1 = computeCost(inner)
        ret[t] = error1
        cost = np.zeros((13,1))
        for i in range(0, m):
            cost += inner[i,0] * X[i,:].T
        theta = theta - cost * alpha / m
        dis = float(error - error1)
        if abs(dis) <= 1e-15:
            break
        error = error1
    return ret


alpha = 0.001
theta = np.zeros((13, 1))
turn = 1000
ret = BGD(X, Y, theta, alpha, turn)
print(ret)

# BGD图表
numx = []
numy = []
step = 50
for i in range(0, 20):
    numx.append(i * step)
    cur = ret[i * step]
    numy.append(cur)
plt.xlabel('iteration')
plt.ylabel('cost')
plt.plot(numx, numy)
plt.title('BGD')
plt.show()
