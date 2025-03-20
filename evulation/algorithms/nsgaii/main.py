# coding=gbk
import numpy as np
import geatpy as ea

from algorithms.objective import get_objective


class TaskScheduling(ea.Problem):
    def __init__(self, task_list):
        name = 'TaskScheduling'
        M = 3  # Ŀ��ά��
        maxormins = [-1, 1, 1]  # Ŀ��������С����ǣ�1��ʾ��С����-1��ʾ���
        Dim = len(task_list)  # ���߱���ά��
        varTypes = [1] * Dim  # ���߱��������ͣ�0��ʾʵ���ͣ�1��ʾ������
        lb = [0] * Dim  # ���߱����½�
        ub = [Dim - 1] * Dim  # ���߱����Ͻ�
        lbin = [1] * Dim  # ���߱����±߽��Ƿ������1���ǣ�0����
        ubin = [1] * Dim  # ���߱����ϱ߽��Ƿ������1���ǣ�0����

        self.task_list = task_list

        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def evalVars(self, X):  # Ŀ�꺯��
        N = X.shape[0]
        ObjV = []  # �洢Ŀ�꺯��ֵ����
        for i in range(N):
            objective = get_objective(self.task_list, X[i, :])
            ObjV.append([objective.resource_utilization, objective.running_time, objective.waiting_time])
        f = np.array(ObjV)
        return f


def get_idx_list(task_list, dataset):
    """
    ʹ��geatpy���ṩ��ģ�幹��NSGAII�㷨
    ��ʼ��Ⱥ����50
    ����������300
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
    ���������һ�����������Ž⼯
    Ҫ�ڽ⼯��ѡһ����Ϊ���ս��
    �˴���ѡ��⼯��λ����Ծ��еĽ�
    """
    first_column = res['ObjV'][:, 0]
    median = np.median(first_column)
    median_index = np.argmin(np.abs(first_column - median))
    return res['Vars'][median_index]
