# coding=gbk
import numpy as np
import geatpy as ea

from algorithms.objective import get_objective


class TaskScheduling(ea.Problem):
    def __init__(self, task_list):
        name = 'TaskScheduling'
        M = 3  # 目标维数
        maxormins = [-1, 1, 1]  # 目标的最大最小化标记，1表示最小化，-1表示最大化
        Dim = len(task_list)  # 决策变量维数
        varTypes = [1] * Dim  # 决策变量的类型，0表示实数型，1表示整数型
        lb = [0] * Dim  # 决策变量下界
        ub = [Dim - 1] * Dim  # 决策变量上界
        lbin = [1] * Dim  # 决策变量下边界是否包含（1：是，0：否）
        ubin = [1] * Dim  # 决策变量上边界是否包含（1：是，0：否）

        self.task_list = task_list

        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def evalVars(self, X):  # 目标函数
        N = X.shape[0]
        ObjV = []  # 存储目标函数值矩阵
        for i in range(N):
            objective = get_objective(self.task_list, X[i, :])
            ObjV.append([objective.resource_utilization, objective.running_time, objective.waiting_time])
        f = np.array(ObjV)
        return f


def get_idx_list(task_list, dataset):
    """
    使用geatpy库提供的模板构造NSGAII算法
    初始种群数：50
    迭代轮数：300
    """
    algorithm = ea.moea_NSGA2_templet(
        problem=TaskScheduling(task_list),
        population=ea.Population(Encoding='P', NIND=50),
        MAXGEN=300,
        logTras=0
    )
    res = ea.optimize(algorithm,
                      verbose=False,
                      drawing=0,
                      outputMsg=False,
                      drawLog=False,
                      saveFlag=False)
    """
    最后生成了一个帕累托最优解集
    要在解集里选一个作为最终结果
    此处是选择解集中位置相对居中的解
    """
    first_column = res['ObjV'][:, 0]
    median = np.median(first_column)
    median_index = np.argmin(np.abs(first_column - median))
    return res['Vars'][median_index]
