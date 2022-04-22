import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

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

validation_size = 100
seed = 10
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y,test_size=validation_size, random_state=seed)
knn = MLPRegressor()
knn.fit(X_train,Y_train)
K_pred = knn.predict(X_validation)
score = r2_score(Y_validation, K_pred)

plt.title("net")
plt.plot(np.arange(len(K_pred)),Y_validation,'go-',label = 'true value')
plt.plot(np.arange(len(K_pred)),K_pred,'ro-',label = 'predict value')


plt.legend()
plt.show()