import cplex
from objective import get_objective


def solve_scheduling_problem(task_list):
    # 创建CPLEX模型
    model = cplex.Cplex()

    # 定义决策变量
    num_tasks = len(task_list)
    num_resources = 6  # CPU、IO、带宽、RAM、时间戳、持续时间
    x = model.variables.add(names=["x" + str(i) for i in range(num_tasks)],
                            types=[model.variables.type.continuous] * num_tasks,
                            lb=[0.0] * num_tasks,
                            ub=[1.0] * num_tasks)

    # 定义目标函数
    model.objective.set_sense(model.objective.sense.minimize)
    model.objective.set_(get_objective(task_list, x).reward)

    # 定义约束
    for i in range(num_resources):
        model.linear_constraints.add(names=["c" + str(i)],
                                      lin_expr=[cplex.SparsePair(x, [task_list[j][i] for j in range(num_tasks)])],
                                      senses=["L"],
                                      rhs=[1.0])

    # 求解模型
    model.solve()

    # 获取解决方案
    solution = model.solution.get_values(x)

    return solution

# 测试函数
task_list = [[0.23000000417232513, 0.03703340142965317, 0.017140299081802368, 0.0005587430205196142, 0.0, 24.0],
             [0.23000000417232513, 0.03703340142965317, 0.017140299081802368, 0.0005587430205196142, 0.0, 24.0],
             [0.23000000417232513, 0.03703340142965317, 0.017140299081802368, 0.0005587430205196142, 0.0, 24.0]]

solution = solve_scheduling_problem(task_list)
print(solution)