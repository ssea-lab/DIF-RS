from cplex import Cplex
from cplex.exceptions import CplexError
import numpy as np

# Define indices for tasks
cpu_idx = 0
io_idx = 1
bandwidth_idx = 2
ram_idx = 3
timestamp_idx = 4
duration_idx = 5

def solve_scheduling_problem(task_list):
    task_n = len(task_list)
    cpx = Cplex()

    # Define variable names
    x_vars = [f'x_{i}_{j}' for i in range(task_n) for j in range(task_n) if i != j]
    s_vars = [f's_{i}' for i in range(task_n)]
    f_vars = [f'f_{i}' for i in range(task_n)]

    # Add binary variables for task precedence
    cpx.variables.add(names=x_vars, types=['B'] * len(x_vars))

    # Add continuous variables for start and finish times
    cpx.variables.add(names=s_vars + f_vars, types=['C'] * 2 * task_n)

    # Constraints for task precedence
    # Ensure that task i is before task j if x_ij is 1
    for i in range(task_n):
        for j in range(task_n):
            if i != j:
                cpx.linear_constraints.add(
                    lin_expr=[[f's_{i}', f'f_{j}'], [1, -1]],
                    senses=['L'],
                    rhs=[0]
                )

    # Constraints for task durations and start-finish times
    for i in range(task_n):
        duration_i = task_list[i][duration_idx]
        cpx.linear_constraints.add(
            lin_expr=[[f'f_{i}', f's_{i}'], [1, -1]],
            senses=['L'],
            rhs=[-duration_i]
        )

    # Constraints for resource usage
    resource_capacity = np.array([1, 1, 1, 1], dtype=float)  # [CPU, IO, BANDWIDTH, RAM]
    M = 1e6  # Big-M constant

    # Create a resource constraint for each resource type
    for r in range(4):  # Iterate over each resource type
        resource_usage = [task[r] for task in task_list]
        for t in range(task_n):
            task_start = f's_{t}'
            task_end = f'f_{t}'
            resource_var = [f'x_{i}_{t}' for i in range(task_n) if i != t]

            # Adding resource constraints using Big-M method
            for i in range(task_n):
                if i != t:
                    cpx.linear_constraints.add(
                        lin_expr=[[task_start, task_end] + resource_var,
                                  [1] * 2 + [-M] * (task_n - 1)],
                        senses=['L'],
                        rhs=[resource_capacity[r] - resource_usage[t]]
                    )

    # Set the objective function
    # Example coefficients for reward calculation
    alpha = 1
    beta = 1 / task_n
    gamma = 1 / task_n

    # Set up a placeholder objective function (this should be replaced with actual calculation)
    objective_coeffs = [0] * len(s_vars) + [0] * len(f_vars)
    cpx.objective.set_sense(cpx.objective.sense.minimize)
    cpx.objective.set_linear(list(zip(s_vars + f_vars, objective_coeffs)))

    # Solve the problem
    try:
        cpx.solve()
    except CplexError as exc:
        print(exc)
        return None

    # Retrieve solution
    solution = cpx.solution
    print("Optimal value:", solution.get_objective_value())
    print("Task order:", [solution.get_values(f'x_{i}_{j}') for i in range(task_n) for j in range(task_n) if i != j])
    print("Start times:", [solution.get_values(f's_{i}') for i in range(task_n)])
    print("Finish times:", [solution.get_values(f'f_{i}') for i in range(task_n)])

# Example usage
task_list = [
    [0.230, 0.037, 0.017, 0.0006, 0.0, 24.0],  # Example task
    [0.300, 0.045, 0.025, 0.0010, 1.0, 30.0],
    [0.150, 0.050, 0.015, 0.0008, 2.0, 20.0]
]
solve_scheduling_problem(task_list)
