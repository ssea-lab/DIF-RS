"""Lightning module for training the DIF-RS EDGE model."""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from pytorch_lightning.utilities import rank_zero_info
import matplotlib.pyplot as plt
from co_datasets.edge_graph_dataset import EDGEGraphDataset
from pl_meta_model import COMetaModel
from utils.diffusion_schedulers import InferenceSchedule
from utils.edge_utils import EDGEEvaluator, batched_two_opt_torch, merge_tours
from objective import get_objective
import importlib
from algorithms.rlpnet.config import Config
import pandas as pd

class EDGEModel(COMetaModel):
  def __init__(self,
               param_args=None):
    super(EDGEModel, self).__init__(param_args=param_args, node_feature_only=False)
    if param_args.do_train:
      self.train_dataset = EDGEGraphDataset(
          arg = param_args,
          data_file=os.path.join(self.args.storage_path, self.args.training_split),
          sparse_factor=self.args.sparse_factor,
      )
    self.test_dataset =EDGEGraphDataset(
        arg = param_args,
        data_file=os.path.join(self.args.storage_path, self.args.test_split),
        sparse_factor=self.args.sparse_factor,
    )
    self.validation_dataset =EDGEGraphDataset(
        arg = param_args,
        data_file=os.path.join(self.args.storage_path, self.args.validation_split),
        sparse_factor=self.args.sparse_factor,
        tag="val"
    )
    # self.test_dataset = EDGEGraphDataset(
    #     data_file=os.path.join(self.args.storage_path, self.args.test_split),
    #     sparse_factor=self.args.sparse_factor,
    # )

    # self.validation_dataset = EDGEGraphDataset(
    #     data_file=os.path.join(self.args.storage_path, self.args.validation_split),
    #     sparse_factor=self.args.sparse_factor,
    # )

  def forward(self, x, adj, t, edge_index):
    return self.model(x, t, adj, edge_index)
  

  def categorical_training_step(self, batch, batch_idx):
    edge_index = None
    _, points, adj_matrix, _ = batch      
    t = np.random.randint(1, self.diffusion.T + 1, points.shape[0]).astype(int)
    # Sample from diffusion   #adj_matrix_onehot 其中[0 , 1]代表1，[1 , 0]代表0
    adj_matrix_onehot = F.one_hot(adj_matrix.long(), num_classes=2).float()
    if self.sparse:
      adj_matrix_onehot = adj_matrix_onehot.unsqueeze(1)

    xt = self.diffusion.sample(adj_matrix_onehot, t)
    xt = xt * 2 - 1
    xt = xt * (1.0 + 0.05 * torch.rand_like(xt))

    t = torch.from_numpy(t).float().view(adj_matrix.shape[0])

    # Denoise
    x0_pred = self.forward(
        points.float().to(adj_matrix.device),
        xt.float().to(adj_matrix.device),
        t.float().to(adj_matrix.device),
        edge_index,
    )

    class_weights = torch.tensor([1.0, self.args.task_n - 1]).to(adj_matrix.device)
    # Compute loss
    loss_func = nn.CrossEntropyLoss(weight=class_weights)
    loss = loss_func(x0_pred, adj_matrix.long())
    self.log("train/loss", loss ,on_step=True)
    return loss

  def gaussian_training_step(self, batch, batch_idx):
    if self.sparse:
      # TODO: Implement Gaussian diffusion with sparse graphs
      raise ValueError("DIF-RS with sparse graphs are not supported for Gaussian diffusion")
    _, points, adj_matrix, _ = batch

    adj_matrix = adj_matrix * 2 - 1
    adj_matrix = adj_matrix * (1.0 + 0.05 * torch.rand_like(adj_matrix))
    # Sample from diffusion
    t = np.random.randint(1, self.diffusion.T + 1, adj_matrix.shape[0]).astype(int)
    xt, epsilon = self.diffusion.sample(adj_matrix, t)

    t = torch.from_numpy(t).float().view(adj_matrix.shape[0])
    # Denoise
    epsilon_pred = self.forward(
        points.float().to(adj_matrix.device),
        xt.float().to(adj_matrix.device),
        t.float().to(adj_matrix.device),
        None,
    )
    epsilon_pred = epsilon_pred.squeeze(1)
    assert not torch.isnan(epsilon_pred).any(), "epsilon_pred contains NaNs"
    assert not torch.isinf(epsilon_pred).any(), "epsilon_pred contains Infs"
    assert not torch.isnan(epsilon).any(), "epsilon contains NaNs"
    assert not torch.isinf(epsilon).any(), "epsilon contains Infs"
    # Compute loss
    loss = F.mse_loss(epsilon_pred, epsilon.float())
    self.log("train/loss", loss)
    return loss

  def training_step(self, batch, batch_idx):
    if self.diffusion_type == 'gaussian':
      return self.gaussian_training_step(batch, batch_idx)
    elif self.diffusion_type == 'categorical':
      return self.categorical_training_step(batch, batch_idx)

  def categorical_denoise_step(self, points, xt, t, device, edge_index=None, target_t=None):
    with torch.no_grad():
      t = torch.from_numpy(t).view(1)
      x0_pred = self.forward(
          points.float().to(device),
          xt.float().to(device),
          t.float().to(device),
          edge_index.long().to(device) if edge_index is not None else None,
      )

      if not self.sparse:
        x0_pred_prob = x0_pred.permute((0, 2, 3, 1)).contiguous().softmax(dim=-1)
      else:
        x0_pred_prob = x0_pred.reshape((1, points.shape[0], -1, 2)).softmax(dim=-1)
      xt = self.categorical_posterior(target_t, t, x0_pred_prob, xt)
      return xt,self.categorical_posterior(np.array([0]), t, x0_pred_prob, xt)

  def gaussian_denoise_step(self, points, xt, t, device, edge_index=None, target_t=None):
    with torch.no_grad():
      t = torch.from_numpy(t).view(1)
      pred = self.forward(
          points.float().to(device),
          xt.float().to(device),
          t.float().to(device),
          edge_index.long().to(device) if edge_index is not None else None,
      )
      pred = pred.squeeze(1)
      xt = self.gaussian_posterior(target_t, t, pred, xt)
      return xt,self.gaussian_posterior(np.array([0]), t, pred, xt)
  def getTours(self,np_points,adj_mats):
    solved_tours = []
    for adj_mat in adj_mats:
      if self.diffusion_type == 'gaussian':
          adj_mat = adj_mat.cpu().detach().numpy() * 0.5 + 0.5
      else:
          adj_mat = adj_mat.float().cpu().detach().numpy() + 1e-6
      tours1 = [] 
      for i in range(adj_mat.shape[0]):
          # 获取第 i 行
          # 找到最接近 0 的元素的列索引
          # 将列索引 j 赋值给 tours[i]
          row = adj_mat[i]
      
          # 找到最接近 0 的元素的列索引
          # sorted_indices = np.argsort(np.abs(row))
          # 找到最接近 1 的元素的列索引
          sorted_indices = np.argsort(np.abs(row - 1))
      
          # 找到第一个不在 tours 中的 j
          for j in sorted_indices:
              if j not in tours1:
                tours1.append(j)
                break
        
      tours2 = [-1] * adj_mat.shape[0]
      elements_with_coords = []

        # 遍历矩阵中的每个元素，并将元素及其坐标添加到列表中
      for i in range(adj_mat.shape[0]):
          for j in range(adj_mat.shape[0]):
              elements_with_coords.append((adj_mat[i][j], i, j))

        # 按元素的绝对值从小到大排序
      elements_with_coords.sort(key=lambda x: abs(x[0]-1))

        # 提取排序后的坐标
      sorted_coords = [(i, j) for _, i, j in elements_with_coords]
      for s in sorted_coords:
          if tours2[s[0]] == -1 and s[1] not in tours2:
            tours2[s[0]] = s[1]
     
      if get_objective(np_points,tours1).reward < get_objective(np_points,tours2).reward:
          solved_tours.append(tours1)
      else:
          solved_tours.append(tours2)
    all_solved_objective = [get_objective(np_points,solved_tours[i]) for i in range(len(solved_tours))]
    all_solved_costs = [i.reward for i in all_solved_objective]
    best_solved_objective_index = np.argmin(all_solved_costs)
    return solved_tours[best_solved_objective_index]
  def check_modules_in_file(self,file_path):
    algorithm_list = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('#'):
                continue
            module_name = '.' + line.strip() + '.main'
            algorithm = importlib.import_module(module_name, package='algorithms')
            algorithm_list.append(algorithm)
    return algorithm_list


  def start_exp(self,task_list, algorithm, dataset):
      start_time = int(time.time() * 1000)
      idx_list = algorithm.get_idx_list(task_list, dataset)
      # print(idx_list)
      end_time = int(time.time() * 1000)
      objective = get_objective(task_list, idx_list)
      objective.set_efficiency(end_time - start_time)
      return objective
  
  def robustness_test_step(self, batch, batch_idx):
    for _ in range(self.args.robust_del_num):
      tensor = batch[1]
      # 生成一个随机的索引位置
      random_index = torch.randint(0, tensor.size(1), (1,)).item()
      # 使用索引掩码删除随机位置的数据点
      # torch.cat 用于连接两个切片, 在维度 1 上进行拼接
      batch[1] = torch.cat((tensor[:, :random_index, :], tensor[:, random_index+1:, :]), dim=1)
    
    edge_index = None
    np_edge_index = None
    device = batch[-1].device
    real_batch_idx, points, adj_matrix, gt_tour = batch
    newSize = adj_matrix.shape[1] - self.args.robust_del_num
    adj_matrix = adj_matrix[:,:newSize,:newSize]
    np_points = points.cpu().numpy()[0]
    stacked_tours = []
    if self.args.parallel_sampling > 1:
      if not self.sparse:
        points = points.repeat(self.args.parallel_sampling, 1, 1)
      else:
        points = points.repeat(self.args.parallel_sampling, 1)
        edge_index = self.duplicate_edge_index(edge_index, np_points.shape[0], device)
        
    for _ in range(self.args.sequential_sampling):
      xt = torch.randn_like(adj_matrix.float())
      if self.args.parallel_sampling > 1:
        if not self.sparse:
          xt = xt.repeat(self.args.parallel_sampling, 1, 1)
        else:
          xt = xt.repeat(self.args.parallel_sampling, 1)
        xt = torch.randn_like(xt)

      if self.diffusion_type == 'gaussian':
        xt.requires_grad = True
      else:
        xt = (xt > 0).long()

      if self.sparse:
        xt = xt.reshape(-1)

      steps = self.args.inference_diffusion_steps
      time_schedule = InferenceSchedule(inference_schedule=self.args.inference_schedule,
                                        T=self.diffusion.T, inference_T=steps)
      
      # Diffusion iterations
      for i in range(steps):
        t1, t2 = time_schedule(i)
        t1 = np.array([t1]).astype(int)
        t2 = np.array([t2]).astype(int)

        if self.diffusion_type == 'gaussian':
          xt,adj_mat_mid = self.gaussian_denoise_step(
              points, xt, t1, device, edge_index, target_t=t2)
        else:
          xt,adj_mat_mid = self.categorical_denoise_step(
              points, xt, t1, device, edge_index, target_t=t2)  
        stacked_tours.append(self.getTours(np_points=np_points,adj_mats=adj_mat_mid))

    all_solved_objective = [get_objective(np_points,stacked_tours[i]) for i in range(len(stacked_tours))]
    all_solved_costs = [i.reward for i in all_solved_objective]
    best_solved_objective_index = np.argmin(all_solved_costs)
    best_solved_objective = all_solved_objective[best_solved_objective_index]
    
    metrics = {
        f"our/solved_cost": best_solved_objective.reward,
        f"our/resource_utilization": best_solved_objective.resource_utilization,
        f"our/running_time": best_solved_objective.running_time,
        f"our/waiting_time": best_solved_objective.waiting_time,
      }
    for k, v in metrics.items():
      self.log(k, v, on_step=True, sync_dist=True )
    
    task_list = points[0].cpu().numpy()
    try:
        algorithm_list = self.check_modules_in_file(self.args.algorithms_file_path)
        # print('load algorithms:')
        # for algorithm in algorithm_list:
        #     print('\t' + algorithm.__name__)
    except ImportError as e:
        raise e
      
    for index,algorithm in enumerate(algorithm_list):
      algorithm_name = algorithm.__name__.split('.')[1]
      objective = self.start_exp(task_list, algorithm, self.args.dataset) 
      # print('\t{}: resource_utilization: {}, running_time: {}, waiting_time: {}, cost: {}\n'.format(
      #           algorithm_name, objective.resource_utilization, objective.running_time, objective.waiting_time,objective.reward))
      metrics = {
        f"{index}_{algorithm_name}/solved_cost": objective.reward,
        f"{index}_{algorithm_name}/resource_utilization": objective.resource_utilization,
        f"{index}_{algorithm_name}/running_time": objective.running_time,
        f"{index}_{algorithm_name}/waiting_time": objective.waiting_time,
      }
      for k, v in metrics.items():
        self.log(k, v, on_step=True, sync_dist=True )
      
    return metrics
  def test_step(self, batch, batch_idx, split='test'):
    # print("self.args.robustness_test"+str(self.args.robustness_test))
    if self.args.robustness_test: 
      return self.robustness_test_step(batch, batch_idx)
    begin = time.time()
    edge_index = None
    np_edge_index = None
    device = batch[-1].device
    if not self.sparse:
      real_batch_idx, points, adj_matrix, gt_tour = batch
      np_points = points.cpu().numpy()[0]
      np_gt_tour = gt_tour.cpu().numpy()[0]
    else:
      real_batch_idx, graph_data, point_indicator, edge_indicator, gt_tour = batch
      route_edge_flags = graph_data.edge_attr
      points = graph_data.x
      edge_index = graph_data.edge_index
      num_edges = edge_index.shape[1]
      batch_size = point_indicator.shape[0]
      adj_matrix = route_edge_flags.reshape((batch_size, num_edges // batch_size))
      points = points.reshape((-1, 2))
      edge_index = edge_index.reshape((2, -1))
      np_points = points.cpu().numpy()
      np_gt_tour = gt_tour.cpu().numpy().reshape(-1)
      np_edge_index = edge_index.cpu().numpy()

    stacked_tours = []
    ns, merge_iterations = 0, 0

    if self.args.parallel_sampling > 1:
      if not self.sparse:
        points = points.repeat(self.args.parallel_sampling, 1, 1)
      else:
        points = points.repeat(self.args.parallel_sampling, 1)
        edge_index = self.duplicate_edge_index(edge_index, np_points.shape[0], device)
    

    for _ in range(self.args.sequential_sampling):
      xt = torch.randn_like(adj_matrix.float())
      if self.args.parallel_sampling > 1:
        if not self.sparse:
          xt = xt.repeat(self.args.parallel_sampling, 1, 1)
        else:
          xt = xt.repeat(self.args.parallel_sampling, 1)
        xt = torch.randn_like(xt)

      if self.diffusion_type == 'gaussian':
        xt.requires_grad = True
      else:
        xt = (xt > 0).long()

      if self.sparse:
        xt = xt.reshape(-1)

      steps = self.args.inference_diffusion_steps
      time_schedule = InferenceSchedule(inference_schedule=self.args.inference_schedule,
                                        T=self.diffusion.T, inference_T=steps)
      
      # start_time = time.time()

      # Diffusion iterations
      for i in range(steps):
        t1, t2 = time_schedule(i)
        t1 = np.array([t1]).astype(int)
        t2 = np.array([t2]).astype(int)

        if self.diffusion_type == 'gaussian':
          xt,adj_mat_mid = self.gaussian_denoise_step(
              points, xt, t1, device, edge_index, target_t=t2)
        else:
          xt,adj_mat_mid = self.categorical_denoise_step(
              points, xt, t1, device, edge_index, target_t=t2)  
        # adj_mat_mid_list.append(adj_mat_mid.cpu().numpy()[0])
        stacked_tours.append(self.getTours(np_points=np_points,adj_mats=adj_mat_mid))
      # stacked_tours.append(self.getTours(np_points=np_points,adj_mats=xt))
      # if self.args.save_numpy_heatmap:
      #   self.run_save_numpy_heatmap(adj_mat, np_points, real_batch_idx, split)
    # solved_tours = np.concatenate(stacked_tours, axis=0)

    objective = get_objective(np_points,np_gt_tour)
    gt_cost = objective.reward

    total_sampling = self.args.parallel_sampling * self.args.sequential_sampling
    all_solved_objective = [get_objective(np_points,stacked_tours[i]) for i in range(len(stacked_tours))]
    all_solved_costs = [i.reward for i in all_solved_objective]
    best_solved_objective_index = np.argmin(all_solved_costs)
    best_solved_objective = all_solved_objective[best_solved_objective_index]
    end = time.time()
    # print(f'Best cost : {best_solved_objective.reward}')
    metrics = {
        f"{split}/gt_cost": gt_cost,
        f"{split}/gt_resource_utilization": objective.resource_utilization,
        f"{split}/gt_running_time": objective.running_time,
        f"{split}/gt_waiting_time": objective.waiting_time,
        f"{split}/solved_cost": best_solved_objective.reward,
        f"{split}/resource_utilization": best_solved_objective.resource_utilization,
        f"{split}/running_time": best_solved_objective.running_time,
        f"{split}/waiting_time": best_solved_objective.waiting_time,
        f"{split}/algorithm efficiency": end - begin,
    }
    for k, v in metrics.items():
      self.log(k, v, on_step=True, sync_dist=True )
    self.log(f"{split}/gap", (best_solved_objective.reward - gt_cost) / gt_cost * 100, prog_bar=True, on_step=True, sync_dist=True)
    # self.log(f"{split}/solved_cost", best_solved_objective.reward,on_epoch=True)
    return metrics

  def run_save_numpy_heatmap(self, adj_mat, np_points, real_batch_idx, split):
    if self.args.parallel_sampling > 1 or self.args.sequential_sampling > 1:
      raise NotImplementedError("Save numpy heatmap only support single sampling")
    exp_save_dir = os.path.join(self.logger.save_dir, self.logger.name, self.logger.version)
    heatmap_path = os.path.join(exp_save_dir, 'numpy_heatmap')
    rank_zero_info(f"Saving heatmap to {heatmap_path}")
    os.makedirs(heatmap_path, exist_ok=True)
    real_batch_idx = real_batch_idx.cpu().numpy().reshape(-1)[0]
    np.save(os.path.join(heatmap_path, f"{split}-heatmap-{real_batch_idx}.npy"), adj_mat)
    np.save(os.path.join(heatmap_path, f"{split}-points-{real_batch_idx}.npy"), np_points)

  def validation_step(self, batch, batch_idx):
    # start_time = time.time()
    # re = self.test_step(batch, batch_idx, split='val')
    # end_time = time.time()
    # execution_time = end_time - start_time
    # print(f"Execution time: {execution_time} seconds")
    # return re
    return self.test_step(batch, batch_idx, split='val')
