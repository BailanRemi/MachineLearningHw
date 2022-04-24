import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import OneHotEncoder

class ELM:
    # x：训练集
    def __init__(self, x, w, b):
        self.w = w      # 输入权重w
        self.b = b      # 隐层偏差b
        h = self.sigmoid(np.dot(x, self.w) + self.b)
        self.H_ = np.linalg.pinv(h)     #隐藏层输出
    # 激活函数sigmoid
    @staticmethod
    def sigmoid(x):
        return 1.0 / (1 + np.exp(x))

    # 分类器训练
    # T：训练集标签
    def classify_train(self, T):
        en_one = OneHotEncoder()
        T = en_one.fit_transform(T.reshape(-1, 1)).toarray()  # 独热编码,   Kecimen 0;  Besni 1
        self.beta = np.dot(self.H_, T)  #最小二乘
        return self.beta            #返回输出权重beta

    # 用训练好的分类器预测
    def classify_predict(self, test_x):
        b_row = test_x.shape[0]
        h = self.sigmoid(np.dot(test_x, self.w) + self.b[:b_row, :])
        result = np.dot(h, self.beta)
        result =np.argmax(result,axis=1)
        return result