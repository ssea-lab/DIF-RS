import random
from tqdm import tqdm
from co_datasets.psnet.config import Config
from co_datasets.psnet.sampling import get_idx_list
import sys
from co_datasets.psnet.config import  load_pkl, pkl_parser
# 定义输入文件和输出文件
input_file = './train/alibaba_cluster_trace.txt'
output_file = './train/alibaba_cluster_trace_100.txt'
print(sys.path)
# 读取数据
with open(input_file, 'r') as f:
    data = f.readlines()

# 确保数据行数为277800
# assert len(data) == 277800, "数据行数不为277800"

# 计算需要采样的组数
num_groups = 200000
group_size = 100

# 确保可以采样100万组
# assert len(data) >= num_groups * group_size, "数据行数不足以采样100万组"
numbers = [0, 1, 2, 3]
weights = [0.2, 0.52, 0.25, 0.03] 
# 随机采样并写入输出文件
with open(output_file, 'w') as f:
    for _ in tqdm(range(num_groups), desc="Processing", ncols=100):
        # 随机选择50个不重复的索引
        indices = random.sample(range(len(data)), group_size)
        # 对索引进行排序以保持相对顺序
        # indices.sort()
        # 获取一组数据
        time = 0
        group = [data[i] for i in indices]
        task_list = []
        for i,g in enumerate(group):
            g = g.split(' ')
            task_list.append([])
            # g_new = []
            for j,_ in enumerate(g):
                if j == 4:
                    g[j] = str(time)
                    time += random.choices(numbers, weights=weights)[0]
                task_list[i].append(float(g[j]))
            g = ' '.join(g)
            group[i] = g
            f.writelines(g)
        # print(type(group[0][0]))
        # 写入一组数据
        
        # 写入分隔符
        idx_list = get_idx_list(task_list,"alibaba_cluster_trace")
        idx_list = [str(j) for j in idx_list]
        f.write(' '.join(idx_list)+"\n")

print("采样完成，结果已写入到", output_file)