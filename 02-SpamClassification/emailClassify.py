# 共2893文本，481垃圾邮件，2412非垃圾邮件
# 分为10份。1份测试集，9份训练集，交叉验证k=9
# 1.生成字典 2.特征提取 3.分类

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from sklearn import svm
from sklearn.model_selection import KFold
# 预处理：根据前9个part创建字典
def createDic(filePath):
    cnt = 0
    wordDic = {}  # 字典
    uselessSet = set(["the", "and", "for", "you", "that", "are", "this", "with", "will"])  # 一些英文表达中的常用词，这些词出现频率很高，且对分类没有帮助
    for part in os.listdir(filePath):  # 得到每个part
        if part == "part10":  # 跳过测试集
            continue
        partPath = os.path.join(filePath, part)
        for info in os.listdir(partPath):  # 得到每个数据记录
            dataPath = os.path.join(partPath, info)
            dataPath = open(dataPath, 'r')
            cnt = cnt + 1
            i = 0
            for line in dataPath.readlines():  # 按行处理
                i = i + 1
                if i <= 2:    continue  # 跳过非正文的前两行
                words = line.split(" ")  # 获取所有单词
                for word in words:
                    if len(word) < 3:      continue  # 字符串小于3的不加入字典
                    if word in uselessSet: continue  # 常用词不加入字典
                    wordDic[word] = wordDic.get(word, 0) + 1
    return [wordDic, cnt]


# 数据特征提取
def processData(wordList, filePath, trainSum, testSum):
    X_train = np.zeros((trainSum, 1000))
    Y_train = np.zeros((trainSum, 1))
    X_test = np.zeros((testSum, 1000))
    Y_test = np.zeros((testSum, 1))
    wordIndexDic = {}
    for i in range(0, len(wordList)):
        wordIndexDic[wordList[i][0]] = i
    cur = 0
    # 训练集
    for part in os.listdir(filePath):  # 得到每个part
        if part == "part10":  # 测试集最后再处理
            continue
        partPath = os.path.join(filePath, part)
        for info in os.listdir(partPath):  # 得到每个数据记录
            if info.find("spmsg") != -1:  # 垃圾邮件
                Y_train[cur, 0] = 0
            else:
                Y_train[cur, 0] = 1
            dataPath = os.path.join(partPath, info)
            dataPath = open(dataPath, 'r')
            i = 0
            for line in dataPath.readlines():
                i = i + 1
                if i <= 2:  continue  # 跳过非正文
                words = line.split(" ")  # 获取所有单词
                for word in words:
                    if word in wordIndexDic.keys():
                        index = wordIndexDic.get(word)
                        X_train[cur, index] = X_train[cur, index] + 1
            cur = cur + 1
    # 测试集
    cur = 0
    filePath = "./dataset/lingspam_public/part10"
    for info in os.listdir(filePath):
        if info.find("spmsgc") != -1:  # 垃圾邮件
            Y_test[cur, 0] = 0
        else:
            Y_test[cur, 0] = 1
        dataPath = os.path.join(filePath, info)
        dataPath = open(dataPath, 'r')
        i = 0
        for line in dataPath.readlines():
            i = i + 1
            if i <= 2:  continue  # 跳过非正文
            words = line.split(" ")  # 获取所有单词
            for word in words:
                if word in wordIndexDic.keys():
                    index = wordIndexDic.get(word)
                    X_test[cur, index] = X_test[cur, index] + 1
        cur = cur + 1
    return [X_train, Y_train, X_test, Y_test]

#交叉验证可视化
def foldVision(kfold, X, y, K):
    fig, ax = plt.subplots(figsize=(10, 5))

    for ii, (tr, tt) in enumerate(kfold.split(X, y)):
        p1 = ax.scatter(tr, [ii] * len(tr), c="#221f1f", marker="_", lw=8)
        p2 = ax.scatter(tt, [ii] * len(tt), c="#b20710", marker="_", lw=8)
        ax.set(
            title="Kfold (K = " + str(K) + ")",
            xlabel="Index",
            ylabel="Iteration",
            ylim=[kfold.n_splits, -1],
        )
        ax.legend([p1, p2], ["Training", "Validation"])

    return plt

#交叉验证
def cross_validation(kfold, X_train, Y_train, trainSum):
    bestC = 0
    cmin = 1
    cmax = 5
    cstep = 0.2
    bestAcc = 0
    while cmin <= cmax:
        sumAcc = 0
        for i, j in kfold.split(X_train, Y_train):
            train_data, train_label = X_train[i, :], Y_train[i]
            valid_data, valid_label = X_train[j, :], Y_train[j]
            model = svm.SVC(decision_function_shape="ovo", kernel="rbf", C=cmin)
            model.fit(train_data, train_label.ravel())
            pred_y = model.predict(valid_data)
            acc = 0
            for i in range(0, len(valid_label)):
                if pred_y[i] == Y_test[i]:
                    acc = acc + 1
            sumAcc = sumAcc + acc / len(valid_label)
        avgAcc = sumAcc / 9
        print("==========================")
        print("C = ", cmin, "\naccuracy = ", "{:.2f}".format(avgAcc*100), "%")
        if avgAcc > bestAcc:
            bestAcc = avgAcc
            bestC = cmin
        cmin = cmin + cstep
    return bestC


filePath = "./dataset/lingspam_public"
[wordDic, trainSum] = createDic(filePath)
# 从字典中选取出现次数最多的几个单词作为特征，选取1000个
wordList = Counter.most_common(wordDic, 1000)  # 返回list对象
print(wordList)
testSum = len(os.listdir("./dataset/lingspam_public/part10"))
print("训练集样本数：", trainSum, "\n测试集样本数：", testSum)
[X_train, Y_train, X_test, Y_test] = processData(wordList, filePath, trainSum, testSum)
# print(X_train)

#交叉验证
kfold = KFold(n_splits=9, shuffle=False)
bestC = cross_validation(kfold, X_train, Y_train, trainSum)

#SVM分类器
model =svm.SVC(decision_function_shape="ovo", kernel="rbf", C=bestC)
model.fit(X_train, Y_train.ravel())
pred_y =model.predict(X_test)
acc = 0
for i in range(0, len(Y_test)):
    if pred_y[i] == Y_test[i]:
        acc = acc + 1
acc = acc / testSum
print("\n\n")
print("bestC = ", bestC, "\n")
print("测试集准确率：", "{:.2f}".format(acc*100), "%")

plt_cv = foldVision(kfold, X_train, Y_train, 9)
plt_cv.show()

