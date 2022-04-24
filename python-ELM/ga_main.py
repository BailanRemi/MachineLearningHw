from ga_class import GA
import geatpy as ea
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from elm_class import ELM
#数据读取和预处理
data = pd.read_excel('./Raisin_Dataset.xlsx')       #读取数据集
data = np.array(data)
data = shuffle(data)                                #打乱顺序
# 分划数据集和测试集
labels = data[:,7]
data = data[:,:7]
# 数据预处理、归一化
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=[0,1])
data = sc.fit_transform(data)

model_ga = GA(data, labels)

algorithm = ea.soea_DE_rand_1_bin_templet(model_ga,
                                          ea.Population(Encoding='RI', NIND=20),
                                          MAXGEN = 30,  # 最大进化代数。
                                          logTras = 1,  # 表示每隔多少代记录一次日志信息，0表示不记录。
                                          trappedValue = 1e-6,  # 单目标优化陷入停滞的判断阈值。
                                          maxTrappedCount = 10)  # 进化停滞计数器最大上限值，如果连续10次迭代，差值都不大于1e-6，视为收敛
#求解
res = ea.optimize(algorithm, verbose=True, drawing=1, outputMsg=True, drawLog=False, saveFlag=True, dirName='result')
num = res['Vars'][0,0]
sumAcc = 0
kfold = KFold(n_splits=10, shuffle=False)
for i, j in kfold.split(data, labels):
    train_data, train_label = data[i, :], labels[i]
    valid_data, valid_label = data[j, :], labels[j]
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
print("KLM_GA Accuracy：%.2f %%" % (sumAcc / 10 * 100))
