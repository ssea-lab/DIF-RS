import numpy as np
from objective import get_objective
import time
# 假设有多个概率矩阵


def combined_cost_function(sequence,task_list):
    return get_objective(task_list,sequence).reward

# 遗传算法参数
population_size = 10
num_generations = 10
mutation_rate = 0.5

def initialize_population(size, num_tasks):
    return [np.random.permutation(num_tasks) for _ in range(size)]

def select_parents(population, task_list,matrices):
    scores = [combined_cost_function(ind, task_list) for ind in population]
    sorted_indices = np.argsort(scores)
    sorted_population = [population[i] for i in sorted_indices]
    
    # 基于概率矩阵选择父代
    probabilities = np.zeros(len(sorted_population))
    for i, ind in enumerate(sorted_population):
        prob = 1.0
        for matrix in matrices:
            for j in range(len(ind) - 1):
                prob *= matrix[ind[j], ind[j+1]] *100
        probabilities[i] = prob
    probabilities /= probabilities.sum()
    
    parent_indices = np.random.choice(len(sorted_population), size=2, p=probabilities)
    return [sorted_population[parent_indices[0]], sorted_population[parent_indices[1]]]

def crossover(parent1, parent2):
    size = len(parent1)
    
    if size <= 2:
        # 如果任务数小于等于2，直接返回父代，或执行其他处理
        return parent1.copy(), parent2.copy()
    
    crossover_point = np.random.randint(1, size - 1)
    
    child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
    child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
    
    child1 = repair_sequence(child1)
    child2 = repair_sequence(child2)
    
    return child1, child2


def repair_sequence(sequence):
    size = len(sequence)
    seen = set()
    missing = set(range(size))
    
    for i in range(size):
        if sequence[i] in seen:
            sequence[i] = -1  # 标记重复的元素
        else:
            seen.add(sequence[i])
            missing.discard(sequence[i])
    
    for i in range(size):
        if sequence[i] == -1:
            sequence[i] = missing.pop()  # 随机填补缺失的元素
    
    return sequence

def mutate(sequence,matrices):
    sequence = sequence.copy()
    if len(sequence) > 1:
        idx1, idx2 = np.random.choice(len(sequence), 2, replace=False)
        
        # 基于概率矩阵决定是否进行交换
        prob = np.mean([matrix[sequence[idx1], sequence[idx2]] for matrix in matrices])
        if np.random.rand() < prob:
            sequence[idx1], sequence[idx2] = sequence[idx2], sequence[idx1]
    
    return sequence

# 主遗传算法流程
def genetic_algorithm(task_list,matrices):
    num_tasks = matrices[0].shape[0]
    population = initialize_population(population_size, num_tasks)
    
    for generation in range(num_generations):
        parents = select_parents(population, task_list,matrices)
        new_population = []
        
        while len(new_population) < population_size:
            child1, child2 = crossover(parents[0], parents[1])
            if np.random.rand() < mutation_rate:
                child1 = mutate(child1,matrices)
            if np.random.rand() < mutation_rate:
                child2 = mutate(child2,matrices)
            new_population.append(child1)
            new_population.append(child2)
        
        population = new_population[:population_size]
    
    best_sequence = min(population, key=lambda seq: combined_cost_function(seq, task_list))
    print(f'GA Best cost : {combined_cost_function(best_sequence, task_list)}')
    return best_sequence, combined_cost_function(best_sequence, task_list)

# best_sequence, best_cost = genetic_algorithm(task_list)
# print(f"Best sequence: {best_sequence}")
# print(f"Cost: {best_cost}")
if __name__ == '__main__':
    num_matrices = 3
    matrices = [np.random.rand(50, 50) for _ in range(num_matrices)]  # 示例概率矩阵列表
    weights = [0.4, 0.3, 0.3]  # 每个矩阵的权重
    task_list = [[0.23000000417232513, 0.03703340142965317, 0.017140299081802368, 0.0005587430205196142, 0.0, 24.0],
    [0.12999999523162842, 0.05258779972791672, 0.004782139789313078, 0.0008669769740663469 ,2.0, 19.0],
    [0.20000000298023224, 0.05638879910111427 ,0.010149899870157242, 0.0006586550152860582 ,2.0, 17.0],
    [0.11999999731779099, 0.028730500489473343 ,0.017572900280356407,0.0006923460168763995 ,3.0, 31.0],
    [0.09000000357627869, 0.06978379935026169, 0.0009249579743482172,0.0004857640014961362 ,4.0, 50.0],
    [0.09000000357627869, 0.07190360128879547, 0.008396049961447716, 0.00035178198595531285, 5.0 ,41.0],
    [0.10000000149011612, 0.054716501384973526, 0.0015454000094905496,0.0003016190021298826, 6.0 ,30.0],
    [0.23999999463558197, 0.057795699685811996, 0.0013688199687749147,0.0004527500132098794, 6.0 ,26.0],
    [0.23000000417232513, 0.029378699138760567, 0.002207909943535924, 0.00035889100399799645, 7.0 ,40.0],
    [0.09000000357627869, 0.06880410015583038, 0.006933100055903196, 0.00043620599899441004 ,8.0 ,22.0],
    [0.23999999463558197, 0.06506039947271347, 0.009211779572069645, 0.00010435500007588416 ,8.0 ,38.0],
    [0.17000000178813934, 0.023439999669790268 ,0.006429609842598438, 0.00021315700723789632 ,10.0, 26.0],
    [0.2199999988079071, 0.017315899953246117, 0.004512690007686615, 0.0007527199923060834, 10.0 ,14.0],
    [0.11999999731779099, 0.01618349924683571, 0.0032746100332587957 ,0.00012412900105118752 ,10.0, 40.0],
    [0.05000000074505806, 0.028893299400806427, 0.00283497991040349, 0.00036248500691726804 ,11.0, 14.0],
    [0.20999999344348907, 0.0004662119899876416 ,0.0043510799296200275 ,0.00044003999209962785 ,12.0, 14.0],
    [0.10999999940395355, 0.010948499664664268, 0.013250400312244892, 0.00011274300049990416 ,13.0, 15.0],
    [0.10999999940395355, 0.053267400711774826, 0.015538999810814857, 0.00024359500093851238 ,13.0, 32.0],
    [0.1599999964237213, 0.042667701840400696, 0.00028362899320200086, 0.0008797459886409342 ,16.0, 13.0],
    [0.17000000178813934, 0.053460199385881424 ,0.011291200295090675, 0.0006679469952359796 ,17.0 ,21.0],
    [0.10000000149011612, 0.04384779930114746, 0.017945000901818275 ,0.0007004970102570951, 17.0, 14.0],
    [0.18000000715255737, 0.058782000094652176 ,0.018191300332546234, 0.0004986419808119535 ,19.0, 26.0],
    [0.20000000298023224, 0.06691659986972809, 0.01829170063138008 ,0.00039171898970380425 ,20.0 ,28.0],
    [0.2199999988079071, 0.06278710067272186, 0.018054500222206116, 0.00046267101424746215 ,21.0 ,13.0],
    [0.25 ,0.01773579977452755, 0.016567399725317955, 0.0001381960028083995 ,21.0, 28.0],
    [0.17000000178813934, 0.09098870307207108, 0.009843680076301098, 0.00030296799377538264 ,22.0 ,24.0],
    [0.05999999865889549, 0.020165100693702698, 0.01834609918296337, 0.00010025200026575476 ,23.0 ,16.0],
    [0.18000000715255737, 0.018530400469899178, 0.00570939015597105, 0.0005923539865761995 ,24.0 ,29.0],
    [0.05999999865889549, 0.03208030015230179, 0.011168699711561203, 0.0006610149866901338 ,26.0 ,30.0],
    [0.05000000074505806, 0.009360520169138908 ,0.0010553599568083882, 0.0005280119949020445, 26.0 ,10.0],
    [0.07000000029802322, 0.04585900157690048, 0.009430699981749058, 0.0006570140249095857, 28.0 ,15.0],
    [0.07999999821186066, 0.010389800183475018 ,0.00632739020511508, 0.0007010350236669183, 28.0 ,48.0],
    [0.07000000029802322, 0.00966928992420435, 0.003344879951328039, 0.0005341339856386185, 30.0 ,45.0],
    [0.05999999865889549, 0.017100799828767776, 0.016572199761867523 ,0.0005675629945471883, 32.0 ,28.0],
    [0.07000000029802322, 0.05641990154981613, 0.016336500644683838 ,0.00023541500559076667, 34.0 ,45.0],
    [0.14000000059604645, 0.03997319936752319 ,0.013135800138115883 ,0.0004743839963339269 ,36.0 ,14.0],
    [0.11999999731779099, 0.030530299991369247, 0.01666950061917305 ,0.0006656760233454406 ,36.0 ,35.0],
    [0.1899999976158142, 0.0326554998755455, 0.006279930006712675, 0.0009052269742824137 ,37.0, 31.0],
    [0.12999999523162842, 0.07161030173301697, 0.00022127199918031693 ,0.0005291060078889132 ,38.0, 10.0],
    [0.10999999940395355, 0.0692610964179039, 0.0010965700494125485 ,0.0009298489894717932 ,39.0 ,48.0],
    [0.12999999523162842, 0.036024898290634155, 0.004819040186703205 ,6.19276033830829e-05 ,41.0 ,47.0],
    [0.2199999988079071, 0.06763859838247299, 0.0019754699897021055, 0.0005200909799896181 ,41.0 ,12.0],
    [0.05999999865889549,0.03929460048675537, 0.014250200241804123, 0.0004146739956922829 ,42.0 ,17.0],
    [0.10000000149011612,0.08885470032691956, 0.008834750391542912, 0.0005130969802848995 ,42.0 ,48.0],
    [0.10000000149011612,0.040192101150751114 ,0.0014230400556698442, 0.00032146400189958513 ,43.0 ,38.0],
    [0.10999999940395355,0.02969370037317276, 0.01827099919319153 ,6.802629650337622e-05 ,43.0 ,46.0],
    [0.14000000059604645,0.02781590074300766, 0.011765800416469574, 0.00037574098678305745, 44.0, 13.0],
    [0.05999999865889549,0.04992229864001274, 0.004015900194644928, 0.0006746239960193634 ,44.0 ,30.0],
    [0.11999999731779099,0.0707136020064354, 0.003612190019339323, 0.00024590000975877047 ,45.0 ,11.0],
    [0.05000000074505806,0.012977399863302708 ,0.006013260222971439, 0.00020562700228765607 ,45.0, 13.0],]
    start = time.time()
    best_sequence, best_cost = genetic_algorithm(task_list,matrices)
    end = time.time()
    print(f"Best sequence: {best_sequence} , execute:{end-start}")
    print(f"Cost: {best_cost}")

