import numpy as np

from algorithms.objective import timestamp_idx, duration_idx


def get_idx_list(task_list, dataset):
    ratio_list = (task_list[:, timestamp_idx] + task_list[:, duration_idx]) / task_list[:, duration_idx]
    idx_list = np.argsort(-ratio_list)
    return idx_list
