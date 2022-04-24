from sklearn.utils import shuffle
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time

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

#保存各分类器的分类结果
plot_x = ["ELM", "SVM", "TREE", "KNN"]
plot_acc = []           #准确率
plot_time = []          #运行时间
# K-折交叉验证
import CrossValidation
K = 10
kfold = KFold(n_splits=10, shuffle=False)
# 交叉验证可视化
CrossValidation.foldVision(kfold, data, labels, K)

# 极限学习机ELM
starttime = time.time()
acc_klm = CrossValidation.cross_validation_klm(kfold, data, labels)
endtime = time.time()
print("KLM Accuracy：%.2f %%" % (acc_klm * 100), "\ttime：%.4fs" %(endtime - starttime))
plot_acc.append(acc_klm)
plot_time.append(endtime-starttime)

#SVM
starttime = time.time()
acc_svm = CrossValidation.cross_validation_svm(kfold, data, labels)
endtime = time.time()
print("SVM Accuracy：%.2f %%" % (acc_svm * 100), "\ttime：%.4fs" %(endtime - starttime))
plot_acc.append(acc_svm)
plot_time.append(endtime-starttime)

#决策树
starttime = time.time()
acc_tree = CrossValidation.cross_validation_tree(kfold, data, labels)
endtime = time.time()
print("TREE Accuracy：%.2f %%" % (acc_tree * 100), "\ttime：%.4fs" %(endtime - starttime))
plot_acc.append(acc_tree)
plot_time.append(endtime-starttime)

#KNN
starttime = time.time()
acc_knn = CrossValidation.cross_validation_knn(kfold, data, labels)
endtime = time.time()
print("KNN Accuracy：%.2f %%" % (acc_knn * 100), "\ttime：%.4fs" %(endtime - starttime))
plot_acc.append(acc_knn)
plot_time.append(endtime-starttime)

# 结果对比
# 准确率
plt.figure()
for i in range(0, len(plot_acc)):
    plot_acc[i] = plot_acc[i]*100
plt.bar(plot_x, plot_acc, 0.5)
plt.ylabel("Accuracy")
plt.title('Classification')
for i, j in zip(plot_x, plot_acc):   #柱子上的数字显示
    plt.text(i,j,'%.2f%%'%(j),ha='center',va='bottom',fontsize=15)
plt.savefig('./results/compare_acc.png')
# 时间
plt.figure()
plt.plot(plot_x, plot_time)
plt.ylabel("Time")
for i, j in zip(plot_x, plot_time):   #折线上的数字显示
    plt.text(i,j,'%.4fs'%(j),ha='center',va='bottom',fontsize=15)
plt.savefig('./results/compare_time.png')

plt.show()