import importlib
import sys
import time
import numpy as np
from algorithms.objective import get_objective
from data import getData

from algorithms.rlpnet.config import Config
import logging


def check_modules_in_file(file_path):
    algorithm_list = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('#'):
                continue
            module_name = '.' + line.strip() + '.main'
            algorithm = importlib.import_module(module_name, package='algorithms')
            algorithm_list.append(algorithm)
    return algorithm_list


def start_exp(task_list, algorithm, dataset):
    start_time = int(time.time() * 1000)
    idx_list = algorithm.get_idx_list(task_list, dataset)
    # print(idx_list)
    end_time = int(time.time() * 1000)
    objective = get_objective(task_list, idx_list)
    objective.set_efficiency(end_time - start_time)
    return objective

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    algorithms_file_path = '/ssd_data/lzw/DIF-RS/evulation/algs'
    try:
        algorithm_list = check_modules_in_file(algorithms_file_path)
        # print('load algorithms:')
        # for algorithm in algorithm_list:
        #     print('\t' + algorithm.__name__)
    except ImportError as e:
        raise e

    data_file = "/ssd_data/lzw/DIF-RS/data/test/google_cluster_trace.txt"
    dataset = 'eua_dataset'
    task_n = 100
    data = getData(data_file=data_file,task_n=task_n)
    print(len(data))
    # print(data[1])
    resource_utilization = [0]*len(algorithm_list)
    running_time = [0]*len(algorithm_list)
    waiting_time = [0]*len(algorithm_list)
    efficiency = [0]*len(algorithm_list)
    cost = [0]*len(algorithm_list)
    print(len(algorithm_list))
    for data_index,item in enumerate(data):
        data_index += 1
        task_list = np.array(item)
        task_list[:, 4] = abs(task_list[:, 4] - task_list[-1][4])
        log = ''
        for index,algorithm in enumerate(algorithm_list):
            objective = start_exp(task_list, algorithm, dataset)
            resource_utilization[index] += objective.resource_utilization
            running_time[index] += objective.running_time
            waiting_time[index] += objective.waiting_time
            efficiency[index] += objective.efficiency
            cost[index] += objective.reward
            log+='\t{}: resource_utilization: {}, running_time: {}, waiting_time: {}, algorithm efficiency: {},cost: {}\n'.format(
                algorithm.__name__.split('.')[1], resource_utilization[index]/data_index, running_time[index]/data_index, waiting_time[index]/data_index,
                efficiency[index]/data_index,cost[index]/data_index)
        print(f'\r第{data_index}个：\n{log}')