import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import time

class network(object):
    # input：输入层节点数，即特征数
    # hidden：隐藏层节点数
    # output：输出层节点数
    # alpha：学习率
    # iter：迭代次数
    def __init__(self, input, hidden, output, alpha, iter):
        # 设置节点数
        self.input = input
        self.hidden = hidden
        self.output = output
        self.wi = np.random.normal(-1e-4, 1e-4, (self.hidden, self.input))
        self.wo = np.random.normal(-1e-4, 1e-4, (self.output, self.hidden))
        self.b1 = np.random.normal(-1e-4, 1e-4, (self.hidden, 1))
        self.b2 = np.random.normal(-1e-4, 1e-4, (self.output, 1))
        self.alpha = alpha
        self.costArr = []
        self.iter = iter

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def d_sigmoid(self, output):
        return np.multiply(output, 1 - output)

    def train(self, X_train, Y_train):
        hin = np.dot(self.wi, X_train) - self.b1
        hout = self.sigmoid(hin)
        outin = np.dot(self.wo, hout) - self.b2
        output = self.sigmoid(outin)
        index = np.argmax(np.transpose(output), axis=1)
        result = np.zeros((self.output, 1))
        result[index] = 1
        # 计算代价
        cost = np.sum(np.power(Y_train - result, 2)) / 2
        # 误差计算
        output_err = Y_train - output
        hidden_err = np.dot(self.wo.T, np.multiply(self.d_sigmoid(output), output_err))
        # 更新权重和阈值
        delt_wo = self.alpha * np.dot(np.multiply(output_err, self.d_sigmoid(output)), np.transpose(hout))
        delt_wi = self.alpha * np.dot(np.multiply(hidden_err, self.d_sigmoid(hout)), np.transpose(X_train))
        self.b2 -= self.alpha * np.multiply(output_err, self.d_sigmoid(output))
        self.b1 -= self.alpha * np.multiply(hidden_err, self.d_sigmoid(hout))
        self.wo = self.wo + delt_wo
        self.wi = self.wi + delt_wi

        return cost

    def train_batch(self, x, y):
        row = len(x)
        Y = np.zeros((row, self.output))
        for i in range(0, row):
            index = y[i]
            Y[i, index] = 1
        for i in range(0, self.iter):
            cost = 0
            for j in range(0, row):
                X_train = np.mat(x[j, :]).T
                Y_train = np.mat(Y[j, :]).T
                cost = cost + self.train(X_train, Y_train)
            cost = cost / row
            if cost <= 1e-9:
                break
            self.costArr.append(cost)
        return self.costArr

    def train_single(self, x, y):
        row = len(x)
        Y = np.zeros((row, self.output))
        for i in range(0, row):
            index = y[i]
            Y[i, index] = 1
        for i in range(0, iter):
            cur = random.randint(0, row-1)
            X_train = np.mat(x[cur, :]).T
            Y_train = np.mat(Y[cur, :]).T
            cost = self.train(X_train, Y_train)
            self.costArr.append(cost)
        return self.costArr

    def predict(self, X_test, Y_test):
        X_test = np.transpose(X_test)
        hin = np.dot(self.wi, X_test) - self.b1
        hout = self.sigmoid(hin)
        outin = np.dot(self.wo, hout) - self.b2
        output = self.sigmoid(outin)
        output = np.transpose(output)
        output = np.argmax(output, axis=1)
        acc = 0
        for i in range(0, len(output)):
            if output[i] == Y_test[i]:
                acc = acc + 1
        return acc


if __name__ == '__main__':
    # 读取数据集
    data = pd.read_csv("lris.csv", header=None)
    data = np.array(data)
    label = data[:, 4]
    data = data[:, :4]
    # 标签用数字表示
    dic = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    for i in range(0, len(label)):
        label[i] = dic[label[i]]
    # 归一化
    from sklearn.preprocessing import MinMaxScaler

    sc = MinMaxScaler(feature_range=[0, 1])
    data = sc.fit_transform(data)
    # 分划30%作为测试集（45）
    from sklearn.model_selection import train_test_split

    X_train, X_test, Y_train, Y_test = train_test_split(data, label, test_size=0.3, random_state=7777)
    m = X_train.shape[0]
    n = X_train.shape[1]
    input = 4
    hidden = 10
    output = 3
    alpha = 0.9
    iter = 500
    xarr = []
    for i in range(0, iter):
        xarr.append(i)

    model_batch = network(input, hidden, output, alpha, iter)
    starttime = time.time()
    cost_batch = model_batch.train_batch(X_train, Y_train)
    endtime = time.time()
    time_batch = endtime - starttime

    model_single = network(input, hidden, output, alpha, iter)
    starttime = time.time()
    cost_single = model_single.train_single(X_train, Y_train)
    endtime = time.time()
    time_single = endtime - starttime

    acc_batch = model_batch.predict(X_test, Y_test) / len(X_test)
    acc_single = model_single.predict(X_test, Y_test) / len(X_test)
    print("Batch Accuracy：%.2f %%" % (acc_batch * 100))
    print("Single Accuracy：%.2f %%" % (acc_single * 100))
    print("-------------------------------")
    print("Batch time：", time_batch)
    print("Single time：", time_single)
    plt.figure()
    plt.plot(xarr, cost_batch)
    plt.figure()
    plt.plot(xarr, cost_single)
    plot_x = ["Batch", "Single"]
    plot_acc = [acc_batch*100, acc_single*100]
    plot_time = [time_batch, time_single]
    plt.figure()
    plt.bar(plot_x, plot_acc, 0.5)
    plt.ylabel("Accuracy")
    for i, j in zip(plot_x, plot_acc):  # 柱子上的数字显示
        plt.text(i, j, '%.2f%%' % (j), ha='center', va='bottom', fontsize=15)
    plt.figure()
    plt.plot(plot_x, plot_time)
    plt.ylabel("time")
    for i, j in zip(plot_x, plot_time):  # 折线上的数字显示
        plt.text(i, j, '%.4fs' % (j), ha='center', va='bottom', fontsize=15)
    plt.show()