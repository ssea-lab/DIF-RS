# coding=gbk
import numpy as np

# [cpu,io,bandwidth,ram,timestamp,duration,latitude,longitude]
cpu_idx = 0
io_idx = 1
bandwidth_idx = 2
ram_idx = 3
timestamp_idx = 4
duration_idx = 5
time_remain_idx = 4
resource_n = 4


class Objective:
    def __init__(self, task_n, resource_utilization, running_time, waiting_time):
        self.task_n = task_n
        self.resource_utilization = resource_utilization
        self.running_time = running_time
        self.waiting_time = waiting_time
        self.alpha = 1
        self.beta = 1 / self.task_n
        self.gama = 1 / self.task_n
        self.reward = self.get_reward()
        self.efficiency = -1

    def get_reward(self):
        return self.alpha * (1 - self.resource_utilization) + \
               self.beta * self.running_time + \
               self.gama * self.waiting_time

    def set_efficiency(self, efficiency):
        """
        ��¼�㷨��ִ��Ч��
        """
        self.efficiency = efficiency


def get_objective(task_list, idx_list):
    """
    :param
    task_list: (task_n, 8), input tasks
               [cpu,io,bandwidth,ram,timestamp,duration,latitude,longitude]
    idx_list: (task_n), predicted tours
    :returns
    resource_utilization, running_time, waiting_time
    """
    sorted_task_list = []
    for idx in idx_list:
        sorted_task_list.append(task_list[idx])

    sum_resource_utilization = 0
    running_time = 0
    sum_waiting_time = 0
    server_resource_remain = np.array([1, 1, 1, 1], dtype=float)  # [CPU, IO, BANDWIDTH, RAM]
    # record tasks being executed within the server
    server_status_map = np.empty((0, 5))  # [CPU, IO, BANDWIDTH, RAM, TIME_REMAIN]

    for task in sorted_task_list:
        task_resource_require = np.array(task[:resource_n])  # [CPU, IO, BANDWIDTH, RAM]
        # insufficient resources, running until resources are sufficient
        while np.any(np.greater(task_resource_require, server_resource_remain)):
            resource_utilization = np.sum(server_status_map[:, cpu_idx])
            min_time_remain = np.min(server_status_map[:, time_remain_idx])
            delete_rows = server_status_map[server_status_map[:, time_remain_idx] == min_time_remain]
            # revert server resources
            server_resource_remain += np.sum(delete_rows[:, :resource_n], axis=0)
            # remove tasks
            server_status_map = np.delete(server_status_map,
                                          np.where(server_status_map[:, time_remain_idx] == min_time_remain),
                                          axis=0)
            # update resource utilization
            sum_resource_utilization += resource_utilization * min_time_remain
            # update running time
            running_time += min_time_remain
            # update remaining time
            if server_status_map.size > 0:
                server_status_map[:, time_remain_idx] -= min_time_remain
        # sufficient resources, directly placing task on the server
        task_in_server = np.concatenate((task[:resource_n], [task[duration_idx]]))
        server_status_map = np.vstack((server_status_map, task_in_server))
        # update waiting time
        sum_waiting_time += running_time + task[duration_idx] + task[timestamp_idx]
        # occupation of server resources
        server_resource_remain -= task_resource_require

    # complete remaining tasks
    while server_status_map.size > 0:
        resource_utilization = np.sum(server_status_map[:, cpu_idx])
        min_time_remain = np.min(server_status_map[:, time_remain_idx])
        # remove tasks
        server_status_map = np.delete(server_status_map,
                                      np.where(server_status_map[:, time_remain_idx] == min_time_remain),
                                      axis=0)
        # update resource utilization
        sum_resource_utilization += resource_utilization * min_time_remain
        # update running time
        running_time += min_time_remain
        # update remaining time
        if server_status_map.size > 0:
            server_status_map[:, time_remain_idx] -= min_time_remain

    # resource utilization
    resource_utilization = sum_resource_utilization / running_time
    # average waiting time
    task_n = len(task_list)
    waiting_time = sum_waiting_time / task_n

    return Objective(task_n, resource_utilization, running_time, waiting_time)
