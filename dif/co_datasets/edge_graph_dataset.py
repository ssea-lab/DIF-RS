"""TSP (Traveling Salesman Problem) Graph Dataset"""

import numpy as np
import torch
# import os
from sklearn.neighbors import KDTree
from torch_geometric.data import Data as GraphData
# import pymysql
# import yaml
from tqdm import tqdm
# import random
import types

# from co_datasets.psnet.config import Config
# from co_datasets.psnet.sampling import get_idx_list
# from concurrent.futures import ThreadPoolExecutor
# import time
# import concurrent.futures

class EDGEGraphDataset(torch.utils.data.Dataset):
  def __init__(self,arg = None, data_file=None,sparse_factor=-1,tag=None):
      self.data_file = data_file
      self.file_lines = []
      self.sparse_factor = sparse_factor
      with open(self.data_file, 'r', encoding='utf-8') as file:
        instance = []
        lines = file.readlines()
        lines_len = len(lines)
        lines_len = int(arg.dataset_size * lines_len)
        lines_len = (lines_len // (arg.task_n+1)) * (arg.task_n+1)
        # lines = lines[0:int(arg.dataset_size * len(lines))]
        for index in tqdm(range(lines_len), desc="Processing lines", total=lines_len):
          line = lines[index]
          instance.append([float(item) for item in line.strip().split(' ')])  # 去掉行末的换行符
          if (index + 1) % (arg.task_n+1) == 0:
              temp = instance[0][4]  # 减去第一个任务的时间，获得每个任务的相对时间
              for i in range(len(instance)-1):
                  instance[i][4] = abs(instance[i][4] - temp)
              # print(instance[50])
              instance[arg.task_n] = [int(i) for i in instance[arg.task_n]]
              self.file_lines.append(instance)
              instance = []
        # 处理最后一组不满50行的情况
        # if instance:
        #   self.file_lines.append(instance)
      print(f'Loaded "{self.data_file}" with {len(self.file_lines)} tasks each has {arg.task_n} task')
   
    # with open(self.data_file, 'r', encoding='utf-8') as file:
    #   instance = []
    #   lines = file.readlines()
    #   lines_len = len(lines)
    #   lines_len = int(arg.dataset_size * lines_len)
    #   lines_len = (lines_len // arg.task_n) * arg.task_n
    #   # lines = lines[0:int(arg.dataset_size * len(lines))]
    #   for index in tqdm(range(lines_len), desc="Processing lines", total=lines_len):
    #     line = lines[index]
    #     instance.append([float(item) for item in line.strip().split(' ')])  # 去掉行末的换行符
    #     if (index + 1) % arg.task_n == 0:
    #         temp = instance[0][4]  # 减去第一个任务的时间，获得每个任务的相对时间
    #         for i, _ in enumerate(instance):
    #             instance[i][4] = abs(instance[i][4] - temp)
    #         idx_list = get_idx_list(instance, arg.dataset)
    #         instance.append(idx_list)
    #         self.file_lines.append(instance)
    #         instance = []
    #   # 处理最后一组不满50行的情况
    #   if instance:
    #     self.file_lines.append(instance)
    # self.db = EDGEGraphDataset.connect()
    # self.cursor =  self.db.cursor()
    # self.dataset = arg.dataset
    # self.file_lines = []
    # self.sparse_factor = sparse_factor
    # for _ in tqdm(range(arg.batch_size * arg.num_epochs)):
    #   instance = self.get_data_instance(arg)
    #   instance = instance.squeeze(0).tolist()
    #   idx_list = get_idx_list(instance,self.dataset)
    #   # if(tag == "val"):
    #   #   print(idx_list)
    #   instance.append(idx_list)
    #   self.file_lines.append(instance)
      
  
  def __len__(self):
    return len(self.file_lines)

  def get_example(self, idx):
    # print("self.file_lines len :" + str(len(self.file_lines)) +"-------- idx :"+str(idx))
    # Select sample
    line = self.file_lines[idx]

    # Extract points 最后一个放置了标签
    points = np.array(line[:-1])

    tour = np.array(line[-1])

    return points, tour

  def __getitem__(self, idx):
    points, tour = self.get_example(idx)
    if self.sparse_factor <= 0:
      # Return a densely connected graph
      adj_matrix = np.zeros((points.shape[0], points.shape[0]))
      for i in range(tour.shape[0]):  
          adj_matrix[i, tour[i]] = 1      
      # return points, adj_matrix, tour
      return (
          torch.LongTensor(np.array([idx], dtype=np.int64)),
          torch.from_numpy(points).float(),
          torch.from_numpy(adj_matrix).float(),
          torch.from_numpy(tour).long(),
      )
    else:
      # Return a sparse graph where each node is connected to its k nearest neighbors
      # k = self.sparse_factor
      sparse_factor = self.sparse_factor
      kdt = KDTree(points, leaf_size=30, metric='euclidean')
      dis_knn, idx_knn = kdt.query(points, k=sparse_factor, return_distance=True)

      edge_index_0 = torch.arange(points.shape[0]).reshape((-1, 1)).repeat(1, sparse_factor).reshape(-1)
      edge_index_1 = torch.from_numpy(idx_knn.reshape(-1))

      edge_index = torch.stack([edge_index_0, edge_index_1], dim=0)

      tour_edges = np.zeros(points.shape[0], dtype=np.int64)
      tour_edges[tour[:-1]] = tour[1:]
      tour_edges = torch.from_numpy(tour_edges)
      tour_edges = tour_edges.reshape((-1, 1)).repeat(1, sparse_factor).reshape(-1)
      tour_edges = torch.eq(edge_index_1, tour_edges).reshape(-1, 1)
      graph_data = GraphData(x=torch.from_numpy(points).float(),
                             edge_index=edge_index,
                             edge_attr=tour_edges)

      point_indicator = np.array([points.shape[0]], dtype=np.int64)
      edge_indicator = np.array([edge_index.shape[1]], dtype=np.int64)
      return (
          torch.LongTensor(np.array([idx], dtype=np.int64)),
          graph_data,
          torch.from_numpy(point_indicator).long(),
          torch.from_numpy(edge_indicator).long(),
          torch.from_numpy(tour).long(),
      )
if __name__ == '__main__':
  self  = types.SimpleNamespace()
  arg =  types.SimpleNamespace()
  arg.dataset = "alibaba_cluster_trace"
  arg.task_n = 50
  arg.num_epochs = 10
  arg.batch_size = 5
    
  # test_dataset = EDGEGraphDataset(arg,
  #       sparse_factor=-1,
  #   )
  # # line = test_dataset.file_lines[0]
  # # idx_list = get_idx_list(line,arg.dataset)
  # # print(idx_list)
  # print(test_dataset)
  # print(test_dataset.file_lines[0][0])
  # print(type(test_dataset.file_lines[0][0]))

  self.file_lines= []
  with open("/data/coding/DIF-RS/data/train/alibaba_cluster_trace_50.txt", 'r', encoding='utf-8') as file:
      instance = []
      lines = file.readlines()
      lines_len = len(lines)
      # lines_len = int(arg.dataset_size * lines_len)
      # lines_len = (lines_len // arg.task_n) * arg.task_n
      # lines = lines[0:int(arg.dataset_size * len(lines))]
      for index in tqdm(range(lines_len), desc="Processing lines", total=lines_len):
        line = lines[index]
        instance.append([float(item) for item in line.strip().split(' ')])  # 去掉行末的换行符
        if (index + 1) % (arg.task_n+1) == 0:
            temp = instance[0][4]  # 减去第一个任务的时间，获得每个任务的相对时间
            for i in range(len(instance)-1):
                instance[i][4] = abs(instance[i][4] - temp)
            instance[50] = [int(i) for i in instance[50]]
            print(instance[50])
            self.file_lines.append(instance)
            self.file_lines.append(instance)
            instance = []
      # 处理最后一组不满50行的情况
      if instance:
        self.file_lines.append(instance)
  print(len(self.file_lines))
  # self.file_lines= []
  # with open("/data/coding/DIF-RS/data/train/alibaba_cluster_trace_50.txt", 'r', encoding='utf-8') as file:
  #     instance = []
  #     lines = file.readlines()
  #     lines_len = len(lines)
  #     # lines_len = int(arg.dataset_size * lines_len)
  #     # lines_len = (lines_len // arg.task_n) * arg.task_n
  #     # lines = lines[0:int(arg.dataset_size * len(lines))]
  #     with concurrent.futures.ThreadPoolExecutor() as executor:
  #       futures = []
  #       for index in range(lines_len):
  #           line = lines[index]
  #           instance.append([float(item) for item in line.strip().split(' ')])  # 去掉行末的换行符
  #           if (index + 1) % arg.task_n == 0:
  #               temp = instance[0][4]  # 减去第一个任务的时间，获得每个任务的相对时间
  #               for i, _ in enumerate(instance):
  #                   instance[i][4] = abs(instance[i][4] - temp)
  #               # 提交任务到线程池
  #               future = executor.submit(get_idx_list, instance, arg.dataset)
  #               futures.append((future, instance))
  #               instance = []

  #       for future, instance in tqdm(futures, desc="Processing lines", total=len(futures)):
  #           idx_list = future.result()
  #           instance.append(idx_list)
  #           self.file_lines.append(instance)
  #       # 处理最后一组不满50行的情况
  #     if instance:
  #         self.file_lines.append(instance)
  