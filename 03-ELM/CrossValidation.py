#用交叉验证来验证分类器效果
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np
from elm_class import ELM

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
    #图表保存在本地
    plt.savefig('./results/CrossValidation.png')

def cross_validation_klm(kfold, data, labels):
    sumAcc = 0
    num = 125       #神经元个数
    for i, j in kfold.split(data, labels):
        train_data, train_label = data[i, :], labels[i]
        valid_data, valid_label = data[j, :], labels[j]
        #ELM分类
        row = train_data.shape[0]
        columns = train_data.shape[1]
        rnd = np.random.RandomState(4396)
        # 随机产生输入权重w 和隐层偏差b
        w = rnd.uniform(-1, 1, (columns, num))
        b = np.zeros([row, num], dtype=float)
        for i in range(num):
            rand_b = rnd.uniform(-0.4, 0.4)
            for j in range(row):
                b[j, i] = rand_b
        model = ELM(train_data, w, b)
        model.classify_train(train_label)
        predict = model.classify_predict(valid_data)
        acc = np.sum(predict == valid_label) / len(predict)
        sumAcc = sumAcc + acc
    return sumAcc / 10

def cross_validation_svm(kfold, data, labels):
    sumAcc = 0
    for i, j in kfold.split(data, labels):
        train_data, train_label = data[i, :], labels[i]
        valid_data, valid_label = data[j, :], labels[j]
        model = svm.SVC(decision_function_shape="ovo", kernel="rbf", C=2)
        model.fit(train_data, train_label.ravel())
        predict = model.predict(valid_data)
        acc = np.sum(predict == valid_label) / len(predict)
        sumAcc = sumAcc + acc
    return sumAcc / 10

def cross_validation_tree(kfold, data, labels):
    sumAcc = 0
    for i, j in kfold.split(data, labels):
        train_data, train_label = data[i, :], labels[i]
        valid_data, valid_label = data[j, :], labels[j]
        model = DecisionTreeClassifier(criterion='entropy')
        model.fit(train_data, train_label.ravel())
        predict = model.predict(valid_data)
        acc = np.sum(predict == valid_label) / len(predict)
        sumAcc = sumAcc + acc
    return sumAcc / 10

def cross_validation_knn(kfold, data, labels):
    sumAcc = 0
    for i, j in kfold.split(data, labels):
        train_data, train_label = data[i, :], labels[i]
        valid_data, valid_label = data[j, :], labels[j]
        model = KNeighborsClassifier()
        model.fit(train_data, train_label.ravel())
        predict = model.predict(valid_data)
        acc = np.sum(predict == valid_label) / len(predict)
        sumAcc = sumAcc + acc
    return sumAcc / 10