from docplex.mp.model import Model
from objective import get_objective
# 定义任务列表和初始状态
task_list = [
    [0.23000000417232513, 0.03703340142965317, 0.017140299081802368, 0.0005587430205196142, 0.0, 24.0],
    [0.1, 0.05, 0.02, 0.001, 0.0, 12.0],
    [0.3, 0.07, 0.03, 0.002, 0.0, 36.0]
]
initial_state = [1, 1, 1, 1, 1]  # 初始状态，每个指标为1

# 定义目标函数
def getObjective(task_list, idx_list):
    total_cost = 0
    state = initial_state[:]
    for idx in idx_list:
        task = task_list[idx]
        cpu, io, bandwidth, ram, timestamp, duration = task
        state[0] -= cpu
        state[1] -= io
        state[2] -= bandwidth
        state[3] -= ram
        state[4] -= timestamp
        total_cost += sum(state)
    return total_cost

# 创建CPLEX模型
model = Model(name='task_scheduling')

# 定义决策变量
num_tasks = len(task_list)
x = model.binary_var_matrix(num_tasks, num_tasks, name='x')

# 定义约束条件
for i in range(num_tasks):
    model.add_constraint(model.sum(x[i, j] for j in range(num_tasks)) == 1, ctname=f'task_{i}_once')
for j in range(num_tasks):
    model.add_constraint(model.sum(x[i, j] for i in range(num_tasks)) == 1, ctname=f'slot_{j}_once')

# 定义目标函数
objective = model.sum(get_objective(task_list, [j]).reward * x[i, j] 
                      for i in range(num_tasks) for j in range(num_tasks))
model.set_objective('min', objective)

# 求解模型
solution = model.solve()

# 输出结果
if solution:
    print(f'Optimal cost: {solution.get_objective_value()}')
    print('Task schedule:')
    schedule = []
    for j in range(num_tasks):
        for i in range(num_tasks):
            if solution.get_value(x[i, j]) > 0.5:
                schedule.append(i)
                break
    print(f'Schedule: {schedule}')
    print(f'real cost:{get_objective(task_list,schedule).reward}')
else:
    print('No solution found')