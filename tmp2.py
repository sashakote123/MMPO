import random
import math
import numpy as np
from statistics import mean
from matplotlib import pyplot as plt


def generateZeroMatrix(sizeN, sizeM):
    matrix = [[0 for j in range(sizeM)] for i in range(sizeN)]
    return matrix


def Fitness22(point):
    J = 0
    sum = 0.0
    for i in range(len(point)):
        sum = sum + (10 * math.cos(2 * math.pi * point[i]) - point[i] ** 2)
    denominator = 1 + np.exp(-1 / (2 * len(point)) * (-10 * len(point) + sum))
    J = 1 / denominator
    return J


def Fitness22(point):
    x_mean = np.mean(point)
    return x_mean ** 2 - (10 * np.cos(2 * np.pi * x_mean) + 10)

def Fitness22(point):
    x_mean = np.mean(point)
    return (x_mean-2**(1/2))**2-1

def Fitness22(point):
    J = 0
    sum = 0.0
    for i in range(len(point)):
        sum = sum + point[i]**2

    return sum

def Fitness22(point):
    sum = 0.0
    for i in range(len(point)):
        sum = sum + point[i] ** 2 - np.cos(2 * np.pi * point[i])

    return 10 + sum

def Fitness2(pointx, pointy):
    x = mean(pointx)
    y = mean(pointy)
    return (x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2



def generatePopulation(sizeN, sizeM):
    matrix = []
    for i in range(sizeN):
        matrix.append([])
        for j in range(sizeM):
            matrix[i].append(random.randrange(5))
    return matrix


# Дифференциальная эволюция. Базовая версия

# У нас 10 точек со случайными параметрами
# На каждой итерации алгортима последовательно выбирается каждая точка в качестве опорной (в популяции проходим по ВСЕМ точкам)
# Для опорной точки подбираются три случайные точки из популяции
# Из этих трех точек создается мутантный вектор
# Мутантный вектор скрещивается с опорной точкой, получаем пробную точку
# Если фитнес пробной точки меньше фитнеса опорной, то берем пробную точку на место опорной в следующую популяцию
# Повторяем алгоритм несколько раз, находим точку с минимальным фитнесом (нахождения точки в коде нет)


def mutantVector(F, a, b, c):
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


def crossover(CR, x, v):
    u = []
    for i in range(M):
        if random.random() <= CR:
            u.append(v[i])
        else:
            u.append(x[i])
    return u


def selection(x,y, u,v):
    if Fitness2(u ,v) <= Fitness2(x, y):
        z1 = u
    else:
        z1 = x

    if Fitness2(u, v) <= Fitness2(x, y):
        z2 = v
    else:
        z2 = y

    return z1, z2


def DE_func(N, Population, Population2):
    F = 0.5
    CR = 0.8
    list_fit = []
    for i in range(N):
        v = mutantVector(F, Population[i], Population[random.randrange(N)], Population[random.randrange(N)])
        u = crossover(CR, Population[i], v)
        Population[i] = selection(Population[i],Population2[i], u, v)
    for i in range(N):
        list_fit.append(Fitness2(Population[i], Population2[i]))
    J = min(list_fit)
    mylist = [J, Population]
    return mylist


def DE_func_with_SoFa(N, firstPopulation, firstPopulation2):
    m = len(firstPopulation)
    n = len(firstPopulation[0])
    population = generateZeroMatrix(m, n)
    population2 = generateZeroMatrix(m, n)


    F = 0.5
    CR = 0.8
    for i in range(N):
        for j in range(m):
            fitness_list = []
            fitness_list = Fitness2(firstPopulation, firstPopulation2)
            prob_list = GenerateProbability(fitness_list, len(firstPopulation))

            choice = random.choices(firstPopulation, weights=prob_list)
            choice = choice[0]

            choice2 = random.choices(firstPopulation2, weights=prob_list)
            choice2 = choice2[0]

            v = mutantVector(F, choice, firstPopulation[random.randrange(m)],
                             firstPopulation[random.randrange(m)])
            u = crossover(CR, firstPopulation[j], v)

            v2 = mutantVector(F, choice2, firstPopulation2[random.randrange(m)],
                             firstPopulation2[random.randrange(m)])
            u2 = crossover(CR, firstPopulation2[j], v2)

            population[j] = selection(firstPopulation[j], u, firstPopulation2[j], u2)[0]
            population2[j] = selection(firstPopulation[j], u, firstPopulation2[j], u2)[1]
        # print('Current population:', population)
        firstPopulation = population
        population = generateZeroMatrix(m, n)

        firstPopulation2 = population2
        population2 = generateZeroMatrix(m, n)

    FitnessMatrix = []
    for i in range(m):
        FitnessMatrix.append(Fitness(firstPopulation[i], firstPopulation2[i]))

    # print('Fitness of all points:', FitnessMatrix)
    y = min(FitnessMatrix)
    y2 = max(FitnessMatrix)
    # print('Min Fitness:', y)
    # print('Max Fitness:', y2)
    return y2, y


top = 10
bottom = 0

N = 20
M = 10


def GetLists(itr):
    firstPopulation = generatePopulation(N, M)
    firstPopulation2 = generatePopulation(N, M)

    x_list = []
    fitness_list = []
    population = []


    for i in range(2):
        population = firstPopulation
        population2 = firstPopulation2
        List = []
        for j in range(itr):
            my_list = DE_func(N, population, population2)
            for k in range(N):
                population[k] = my_list[1][k]
            List.append(my_list[0])
        if i != 0:
            for k in range(itr):
                fitness_list[k] = (fitness_list[k] + List[k]) / 2
        else:
            for k in range(itr):
                fitness_list.append(List[k])

    for i in range(itr):
        x_list.append(i)

    delta = 0
    delta_tmp = 0
    delta_last = 0
    x = 0
    for i in range(itr - 2):
        delta_tmp = fitness_list[i] - fitness_list[i + 1]
        delta = fitness_list[i + 1] - fitness_list[i + 2]
        if delta_tmp != 0 and delta == 0:
            delta_last = fitness_list[i+2]
            x = i+2

    return x_list, fitness_list, delta_last, x

data = GetLists(500)
print(data[2], "на", data[3], 'итерации')

plt.title("DE_func")
plt.plot(data[0], data[1])
plt.show()
