import geatpy as ea
import numpy as np
from sklearn.model_selection import KFold
from elm_class import ELM
from multiprocessing.dummy import Pool as ThreadPool
# 遗传算法工具箱geatpy单目标优化模板
class GA(ea.Problem):  # 继承Problem父类
    def __init__(self, data, labels):
        name = 'KLM_GA'  # 函数名称，可以随意设置
        M = 1  # 初始化M（目标维数）
        maxormins = [-1]  # 目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标
        Dim = 1  # 初始化Dim（决策变量维数）
        varTypes = [1]  # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        lb = [100] * Dim  # 决策变量下界
        ub = [500] * Dim  # 决策变量上界
        lbin = [1] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
        # 属性集和标签集
        self.data = data
        self.dataTarget = labels

    def aimFunc(self, pop):  # 目标函数，采用多线程加速计算
        Vars = pop.Phen  # 得到决策变量矩阵
        pop.ObjV = np.zeros((pop.sizes, 1))  # 初始化种群个体目标函数值列向量

        def subAimFunc(index):
            num = Vars[index, 0]
            # 计算交叉验证的得分
            sumAcc = 0
            kfold = KFold(n_splits=10, shuffle=False)
            for i, j in kfold.split(self.data, self.dataTarget):
                train_data, train_label = self.data[i, :], self.dataTarget[i]
                valid_data, valid_label = self.data[j, :], self.dataTarget[j]
                # ELM分类
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
            pop.ObjV[index] = sumAcc/10  # 把交叉验证的平均得分作为目标函数值

        pool = ThreadPool(2)  # 设置池的大小
        pool.map(subAimFunc, list(range(pop.sizes)))
