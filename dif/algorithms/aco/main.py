# coding=gbk
import numpy as np

from algorithms.objective import get_objective


class AntColonyOptimization:
    def __init__(self, task_list):
        self.task_list = task_list
        self.ant_n = 30
        self.iteration_n = 200
        self.alpha = 1.0
        self.beta = 1.0
        self.rho = 0.5
        self.q = 100
        self.task_n = task_list.shape[0]
        self.pheromone_matrix = np.ones((self.task_n, self.task_n)) / self.task_n
        self.best_path = None
        self.best_reward = np.inf

    def run(self):
        for _ in range(self.iteration_n):
            ant_paths = self.construct_ant_paths()
            self.update_pheromone_matrix(ant_paths)
            self.update_best_path(ant_paths)
        return self.best_path, self.best_reward

    def construct_ant_paths(self):
        ant_paths = []
        for ant in range(self.ant_n):
            mask = [False] * self.task_n
            start_idx = np.random.randint(self.task_n)
            mask[start_idx] = True
            path = [start_idx]
            for _ in range(self.task_n - 1):
                next_idx = self.select_next_idx(path[-1], mask)
                mask[next_idx] = True
                path.append(next_idx)
            ant_paths.append(path)
        return ant_paths

    def select_next_idx(self, current_idx, mask):
        unvisited_idx_list = np.where(np.logical_not(mask))[0]
        pheromone_values = self.pheromone_matrix[current_idx, unvisited_idx_list]
        heuristic_values = 1.0 / abs(current_idx - unvisited_idx_list)
        probabilities = pheromone_values ** self.alpha * heuristic_values ** self.beta
        probabilities /= np.sum(probabilities)
        next_idx = np.random.choice(unvisited_idx_list, p=probabilities)
        return next_idx

    def update_pheromone_matrix(self, ant_paths):
        self.pheromone_matrix *= (1 - self.rho)
        for path in ant_paths:
            path_reward = self.calculate_path_reward(path)
            delta_pheromone = self.q / path_reward
            for i in range(len(path) - 1):
                idx_i = path[i]
                idx_j = path[i + 1]
                self.pheromone_matrix[idx_i, idx_j] += delta_pheromone
                self.pheromone_matrix[idx_j, idx_i] += delta_pheromone

    def calculate_path_reward(self, path):
        objective = get_objective(self.task_list, path)
        return objective.reward

    def update_best_path(self, ant_paths):
        for path in ant_paths:
            path_reward = self.calculate_path_reward(path)
            if path_reward < self.best_reward:
                self.best_reward = path_reward
                self.best_path = path


def get_idx_list(task_list, dataset):
    """
    蚁群算法
    初始种群数：30
    迭代轮数：200
    """
    aco = AntColonyOptimization(task_list)
    best_path, _ = aco.run()
    return best_path
