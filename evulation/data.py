from tqdm import tqdm
def getData(data_file,task_n):
    file_lines = []
    with open(data_file, 'r', encoding='utf-8') as file:
      instance = []
      lines = file.readlines()
      lines_len = len(lines)
      # lines = lines[0:int(arg.dataset_size * len(lines))]
      for index in tqdm(range(lines_len), desc="Processing lines", total=lines_len):
        line = lines[index]
        instance.append([float(item) for item in line.strip().split(' ')])  # 去掉行末的换行符
        if (index + 1) % (task_n) == 0:
            # print(instance[50])
            # instance[arg.task_n] = [int(i) for i in instance[arg.task_n]]
            file_lines.append(instance)
            instance = []
            
      # 处理最后一组不满50行的情况
      # if instance:
      #   self.file_lines.append(instance)
    print(f'Loaded "{data_file}" with {len(file_lines)} tasks each has {task_n} task')
    return file_lines