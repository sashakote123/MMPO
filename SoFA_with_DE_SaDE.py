import random
import math
import numpy as np
from statistics import mean
from matplotlib import pyplot as plt
def Fitness22(point):
    sum = 0.0
    for i in range(len(point)):
        sum = sum + (10 * math.cos(2 * math.pi * point[i]) - point[i]**2)
    denominator = 1 + np.exp(-1/(2*len(point)) * (-10 * len(point) + sum))
    J = 1/denominator
    return J

def Fitness2(point):
    x_mean = np.mean(point)
    return x_mean**2-(10*np.cos(2*np.pi*x_mean)+10)

def generatePopulation(sizeN, sizeM):
    matrix = []
    for i in range(sizeN):
        matrix.append([])
        for j in range(sizeM):
            matrix[i].append(random.randrange(5))
    return matrix


def GenerateProbability(matrix, n):
    list = []
    sum = 0
    for i in range(n):
        sum = sum + matrix[i]
    for i in range(n):
        list.append(matrix[i] / sum)
    return list


def adjust_dimension(matrix, t):
    for i in range(len(matrix)):
        matrix[i].append(random.randrange(5)/(2**t))
    return matrix




def gauss_func(xi, mat_wait, deviation):
    prob_i_comp_in_j_point = 1 / (np.sqrt(2 * np.pi) * deviation) * np.exp(
        -((xi - mat_wait) ** 2) / (2 * deviation ** 2))
    return prob_i_comp_in_j_point


def gauss(i, matrix):
    list_i = []
    for j in range(len(matrix)):
        list_i.append(matrix[j][i])
    mat_wait = 0
    for j in range(len(matrix)):
        mat_wait += matrix[j][i] * (1 / len(matrix))
    mat_wait_sqr = 0
    for j in range(len(matrix)):
        mat_wait_sqr += ((matrix[j][i]) ** 2) * (1 / len(matrix))
    dispers = mat_wait_sqr - mat_wait ** 2
    middle_sqr_deviation = np.sqrt(dispers)
    prob_list_2 = []
    for j in range(len(matrix)):
        prob_list_2.append(gauss_func(list_i[j], mat_wait, middle_sqr_deviation))
    choice = random.choices(list_i, weights=prob_list_2, k=1)
    choice = choice[0]
    return choice


def mutation(point, matrix):
    tmp = []
    for i in range(len(point)):
        tmp.append(gauss(i, matrix))
    return tmp


## max  min
def SoFA(population):
    fitness_population = []
    for i in range(len(population)):
        fitness_population.append(Fitness2(population[i]))
    probability_population = GenerateProbability(fitness_population, len(population))

    choice = random.choices(population, weights=probability_population, k=1)
    choice = choice[0]

    mutant = mutation(choice, population)
    population.append(mutant)
    fitness_population.clear()
    for i in range(len(population)):
        fitness_population.append(Fitness2(population[i]))
    fit = max(fitness_population)
    return fit


def mutantVector_rand_1(F, a, b, c):
    v = []
    for i in range(M):
        temp = a[i] + F * (b[i] + (-1 * c[i]))
        if temp >= top:
            v.append(top)
        elif temp <= bottom:
            v.append(bottom)
        else:
            v.append(temp)
    return v


def mutantVector_rand_2(F1, F2, a, b, c, d, e):
    v = []
    for i in range(M):
        temp = a[i] + F1 * (b[i] + (-1 * c[i])) + F2 * (d[i] + (-1 * e[i]))
        if temp >= top:
            v.append(top)
        elif temp <= bottom:
            v.append(bottom)
        else:
            v.append(temp)
    return v


def mutantVector_curtorand_1(F, cur, b, c):
    v = []
    for i in range(M):
        temp = cur[i] + F * (b[i] + (-1 * c[i]))
        if temp >= top:
            v.append(top)
        elif temp <= bottom:
            v.append(bottom)
        else:
            v.append(temp)
    return v


def mutantVector_randtobest_2(F1, F2, a, best, c, d, e):
    v = []
    for i in range(M):
        temp = a[i] + F1 * (best[i] + (-1 * c[i])) + F2 * (d[i] + (-1 * e[i]))
        if temp >= top:
            v.append(top)
        elif temp <= bottom:
            v.append(bottom)
        else:
            v.append(temp)
    return v


def crossover(CR, x, v):
    u = []
    for i in range(M):
        if random.random() <= CR:
            u.append(v[i])
        else:
            u.append(x[i])
    return u


def DE_rand_1(target, population, F, CR):
    Population = population
    v = mutantVector_rand_1(F, Population[random.randrange(NP)], Population[random.randrange(NP)], Population[random.randrange(NP)])
    u = crossover(CR, target, v)
    winner = selection(target, u)
    if winner == u:
        CR_list_record.append(CR)
    return winner


def DE_rand_2(target, population, F1, F2, CR):
    Population = population
    v = mutantVector_rand_2(F1, F2, Population[random.randrange(NP)], Population[random.randrange(NP)], Population[random.randrange(NP)],
                         Population[random.randrange(NP)], Population[random.randrange(NP)])
    u = crossover(CR, target, v)
    winner = selection(target, u)
    if winner == u:
        CR_list_record.append(CR)
    return winner


def DE_current_to_rand_1(target, population, F, CR):
    Population = population
    v = mutantVector_curtorand_1(F, target, Population[random.randrange(NP)], Population[random.randrange(NP)])
    u = crossover(CR, target, v)
    winner = selection(target, u)
    if winner == u:
        CR_list_record.append(CR)
    return winner


def DE_rand_to_best_2(target, population, F1, F2, CR):
    Population = population
    best = find_best(Population)
    v = mutantVector_randtobest_2(F1, F2, Population[random.randrange(NP)], best, Population[random.randrange(NP)], Population[random.randrange(NP)], Population[random.randrange(NP)])
    u = crossover(CR, target, v)
    winner = selection(target, u)
    if winner == u:
        CR_list_record.append(CR)
    return winner


##### <=  >=
def selection(x, u):
    if Fitness2(u) <= Fitness2(x):
        y = u
    else:
        y = x
    return y

##### <=  >=
def find_best(list_species):
    best = list_species[0]
    J_best = Fitness2(best)
    for i in range(len(list_species)):
        J_tmp = Fitness2(list_species[i])
        if J_tmp <= J_best:
            J_best = J_tmp
            best = list_species[i]
    return best


##### min max
def MAIN_func(n, population, strategy_success, list_f, cr_list):
    list_fit = []
    tmp_population = population
    strategy = ["DE_rand_1", "DE_rand_2", "DE_current_to_rand_1", "DE_rand_to_best_2"]
    strategy_rate = [0, 0, 0, 0]

    index_population = []
    for i in range(n):
        index_population.append(i)
    fitness_population = []
    for i in range(len(tmp_population)):
        fitness_population.append(Fitness2(tmp_population[i]))
    probability_population = GenerateProbability(fitness_population, len(tmp_population))
    choice = random.choices(index_population, weights=probability_population, k=1)
    index = choice[0]
    target_point = tmp_population[index]

    strategy_choice = random.choices(strategy, weights=strategy_success, k=1)
    strategy_choice = strategy_choice[0]
    if strategy_choice == 'DE_rand_1':
        tmp_population[index] = DE_rand_1(target_point, population, random.choice(list_f), random.choice(cr_list))
        strategy_rate[0] += 1
    elif strategy_choice == 'DE_rand_2':
        tmp_population[index] = DE_rand_2(target_point, population, random.choice(list_f), random.choice(list_f), random.choice(cr_list))
        strategy_rate[1] += 1
    elif strategy_choice == 'DE_current_to_rand_1':
        tmp_population[index] = DE_current_to_rand_1(target_point, population, random.choice(list_f), random.choice(cr_list))
        strategy_rate[2] += 1
    else:                  # DE_rand_to_best_2
        tmp_population[index] = DE_rand_to_best_2(target_point, population, random.choice(list_f), random.choice(list_f), random.choice(cr_list))
        strategy_rate[3] += 1

    for i in range(n):
        list_fit.append(Fitness2(tmp_population[i]))
    J = min(list_fit)
    for i in range(4):
        strategy_rate[i] = strategy_rate[i] / 4
    mylist = [J, tmp_population, strategy_rate]
    return mylist

NP = 5
M = 5
top = 10
bottom = 0
CR_list_record = [0.5]
def GetLists(itr):


    # Число итераций
    dim_up = 1000000   # Контроль частоты увеличения размерности пространства параметров

    # Control Parameter Adaptation:
    CRm = 0

    t = 0
    population = []
    list_F = []
    CR_list = []

    strategy_success = []

    x_list = []
    Fitness_list = []

    for i in range(5):
        population = generatePopulation(NP, M)
        List_Fit_in_one_go = []
        strategy_success = [1, 1, 1, 1]
        for j in range(itr):
            if j % 20 == 0:
                if len(CR_list_record) == 0:
                    CR_list_record.append(0)
                CRm = sum(CR_list_record)/len(CR_list_record)
                CR_list.clear()
                for k in range(5):
                    CR_list.append(random.gauss(CRm, 0.1))
                CR_list_record.clear()

            list_F.clear()
            for k in range(4):
                list_F.append(random.gauss(0.5, 0.3))

            my_list = MAIN_func(NP, population, strategy_success, list_F, CR_list)

            ## Рост размерности:
            if j % dim_up == 0:
                population = adjust_dimension(population, t)
                t += 1
            ##

            List_Fit_in_one_go.append(my_list[0])
            for k in range(NP):
                population[k] = my_list[1][k]
            for k in range(4):
                strategy_success[k] = my_list[2][k]

        if i != 0:
            for k in range(itr):
                Fitness_list[k] = (Fitness_list[k] + List_Fit_in_one_go[k])/2
        else:
            for k in range(itr):
                Fitness_list.append(List_Fit_in_one_go[k])


    for i in range(itr):
        x_list.append(i)

    delta = 0
    delta_tmp = 0
    delta_last = 0
    x = 0
    for i in range(itr - 2):
        delta_tmp = Fitness_list[i] - Fitness_list[i + 1]
        delta = Fitness_list[i + 1] - Fitness_list[i + 2]
        if delta_tmp != 0 and delta == 0:
            delta_last = Fitness_list[i+2]
            x = i+2


    return x_list, Fitness_list, delta_last, x

data = GetLists(500)
print(data[2], "на", data[3], 'итерации')

plt.title("DE_func")
plt.plot(data[0], data[1])
plt.show()
