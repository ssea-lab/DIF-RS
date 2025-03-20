# coding=gbk
import numpy as np

from algorithms.objective import duration_idx


def get_type_1_2_tasks(server, task):
    type_1_task_list, type_2_task_list = [], []
    for running_task in server:
        if 1 / running_task[duration_idx] > 1 / task[duration_idx]:
            type_1_task_list.append(running_task)
        else:
            type_2_task_list.append(running_task)
    return type_1_task_list, type_2_task_list


def get_server_wrt(type_1_task_list, type_2_task_list, task):
    part_1 = 1 * np.sum(type_1_task_list, axis=0)[duration_idx] if len(type_1_task_list) > 0 else 0
    part_2 = 1 * task[duration_idx]
    part_3 = task[duration_idx] * 1 if len(type_2_task_list) > 0 else 0
    server_wrt = part_1 + part_2 + part_3
    return server_wrt


def dispatching_policy(task, server_list):
    """
    dispatching policy
    :param task: ���·�������
    :param server_list: �������б��������֮ǰ�Ѿ��·���task
    :return: task��Ҫ�·�����Ŀ�����������
    """
    server_wrt_list = []
    for server in server_list:
        type_1_task_list, type_2_task_list = get_type_1_2_tasks(server, task)
        server_wrt = get_server_wrt(type_1_task_list, type_2_task_list, task)
        server_wrt_list.append(server_wrt)
    min_wrt_server_idx = np.argmin(server_wrt_list)
    return min_wrt_server_idx


def scheduling_policy(task_list):
    """
    scheduling policy
    :param task_list: �����ȵ�����
    :return: �����������У������������ִ��˳��
    """
    weight_density_list = 1 / task_list[:, duration_idx]
    idx_list = np.argsort(weight_density_list)[::-1]
    return idx_list


def get_idx_list(task_list, dataset):
    """
    ���´���û��Դ������OnDisc�������Ҹ����Լ�����⸴�ֵ� :(
    ���������Ż�Ŀ�겻ͬ�����ò���ԭ���Ĵ����߼������˸��� :)
    ���ķ�Ϊdispatching policy��scheduling policy
    ����dispatching policy���ھ���������·����ĸ���������scheduling policy���ھ��������ڷ������ڵ�ִ��˳��
    ��������ֻ���õ�scheduling policy����������ܽ�����ж����ӽ������������õ�����dispatching policy
    ͬʱ����������Ϊ����������ġ�ʱ�������ԡ�������û�У������ó���1��ʾ
    """

    return scheduling_policy(task_list)
