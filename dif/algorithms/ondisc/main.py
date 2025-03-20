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
    :param task: 待下发的任务
    :param server_list: 服务器列表，里面存着之前已经下发的task
    :return: task将要下发到的目标服务器索引
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
    :param task_list: 待调度的任务
    :return: 任务索引序列，代表了任务的执行顺序
    """
    weight_density_list = 1 / task_list[:, duration_idx]
    idx_list = np.argsort(weight_density_list)[::-1]
    return idx_list


def get_idx_list(task_list, dataset):
    """
    文章代码没开源，所以OnDisc代码是我根据自己的理解复现的 :(
    而且由于优化目标不同，不得不对原来的处理逻辑进行了改造 :)
    论文分为dispatching policy和scheduling policy
    其中dispatching policy用于决策任务该下发到哪个服务器，scheduling policy用于决策任务在服务器内的执行顺序
    所以这里只能用到scheduling policy，如果后期能将任务卸载添加进来，或许还能用到它的dispatching policy
    同时，文章中人为定义了任务的“时延敏感性”，这里没有，所以用常数1表示
    """

    return scheduling_policy(task_list)
