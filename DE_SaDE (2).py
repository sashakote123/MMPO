import random
import math
import numpy as np
from statistics import mean
from matplotlib import pyplot as plt
from graph import *


T = 5
N = 5
G = generateGraph(N, 3)
def Fitness2(point):
    return run2(G,T,point)[0]



def Fitness22(point):
    sum = 0.0
    for i in range(len(point)):
        sum = sum + (10 * math.cos(2 * math.pi * point[i]) - point[i]**2)
    denominator = 1 + np.exp(-1/(2*len(point)) * (-10 * len(point) + sum))
    J = 1/denominator
    return J

def Fitness22(point):
    x_mean = np.mean(point)
    return x_mean**2-(10*np.cos(2*np.pi*x_mean)+10)



def generatePopulation(sizeN, sizeM):
    matrix = []
    for i in range(sizeN):
        matrix.append([])
        for j in range(sizeM):
            matrix[i].append(random.randrange(5))
    return matrix


t = 0
def adjust_dimension(matrix, t):
    population = matrix
    for i in range(len(population)):
        population[i].append(random.randrange(5)/(2**t))
    return population





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
def DE_func(n, population, strategy_success, list_f, cr_list):
    list_fit = []
    tmp_population = population
    strategy = ["DE_rand_1", "DE_rand_2", "DE_current_to_rand_1", "DE_rand_to_best_2"]
    strategy_rate = [0, 0, 0, 0]
    for i in range(n):
        strategy_choice = random.choices(strategy, weights=strategy_success, k=1)
        strategy_choice = strategy_choice[0]
        if strategy_choice == 'DE_rand_1':
            tmp_population[i] = DE_rand_1(population[i], population, random.choice(list_f), random.choice(cr_list))
            strategy_rate[0] += 1
        elif strategy_choice == 'DE_rand_2':
            tmp_population[i] = DE_rand_2(population[i], population, random.choice(list_f), random.choice(list_f), random.choice(cr_list))
            strategy_rate[1] += 1
        elif strategy_choice == 'DE_current_to_rand_1':
            tmp_population[i] = DE_current_to_rand_1(population[i], population, random.choice(list_f), random.choice(cr_list))
            strategy_rate[2] += 1
        else:                  # DE_rand_to_best_2
            tmp_population[i] = DE_rand_to_best_2(population[i], population, random.choice(list_f), random.choice(list_f), random.choice(cr_list))
            strategy_rate[3] += 1

    for i in range(n):
        list_fit.append(Fitness2(tmp_population[i]))
    J = min(list_fit)
    for i in range(4):
        strategy_rate[i] = strategy_rate[i] / 4
    mylist = [J, tmp_population, strategy_rate]
    return mylist


top = 10
bottom = 0

NP = N
M = N

itr = 30   # Число итераций
dim_up = 100000   # Контроль частоты увеличения размерности пространства параметров

# SaDE

# Control Parameter Adaptation:
CRm = 0

population = []
list_F = []
CR_list = []
CR_list_record = [0.5]
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
            CR_list = []
            for k in range(5):
                CR_list.append(random.gauss(CRm, 0.1))
            CR_list_record = []

        list_F = []
        for k in range(4):
            list_F.append(random.gauss(0.5, 0.3))

        my_list = DE_func(NP, population, strategy_success, list_F, CR_list)

        if j % dim_up == 0:
            population = adjust_dimension(population, t)
            t += 1

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


plt.title("DE_SaDE")
plt.plot(x_list, Fitness_list)
plt.show()
