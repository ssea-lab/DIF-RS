import random
from tqdm import tqdm
from co_datasets.psnet.config import Config
from co_datasets.psnet.sampling import get_idx_list
import sys
from co_datasets.psnet.config import  load_pkl, pkl_parser
from SA import simulated_annealing
# 定义输入文件和输出文件
input_file = './val/google_cluster_trace.txt'
output_file = './val/SA/google_cluster_trace_100.txt'
print(sys.path)
# 读取数据
with open(input_file, 'r') as f:
    data = f.readlines()
print(len(data))
data1 = data
# print(len(data1))
# print(data1[0])
task_n = 100
task_list = []
with open(output_file, "a", encoding='utf-8') as file:
    for index,item in tqdm(enumerate(data1), total=len(data1), desc="Processing items"):
        # print(index)
        task_list.append([float(i) for i in item.split(' ')])
        if (index +1) % task_n == 0:
            temp = task_list[0][4] # 减去第一个任务的时间，获得每个任务的相对时间
            for i,_ in enumerate(task_list):
                task_list[i][4] = abs(task_list[i][4] - temp)
                # print(task_list[i])
                file.write(' '.join([str(j) for j in task_list[i]]) + "\n")
            idx_list = simulated_annealing(task_list)[0]
            idx_list = [str(id) for id in idx_list]
            # print(' '.join(idx_list))
            # print(' '.join(idx_list))
            file.write(' '.join(idx_list)+"\n")
            # print(task_list)
            task_list =[]
        # print(item.split(' '))
        pass