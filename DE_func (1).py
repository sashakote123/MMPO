import random
import math
import numpy as np
from statistics import mean
from matplotlib import pyplot as plt


def Func1(i, j, y1, y2):
    return (math.sin(i * math.pi * (y1 - 0.5)) * math.sin(j * math.pi * (y2 - 0.5)))


def Func2(i, j, y1, y2):
    return (math.cos(i * math.pi * (y1 - 0.5)) * math.cos(j * math.pi * (y2 - 0.5)))


def generateMatrix(size):
    matrix = [[random.uniform(-1, 1) for j in range(size)] for i in range(size)]
    return matrix


def generateZeroMatrix(sizeN, sizeM):
    matrix = [[0 for j in range(sizeM)] for i in range(sizeN)]
    return matrix


# def CreatePoint():
#    a = random.randrange(10)
#    b = random.randrange(10)
#    return a, b


def Fitness(point):
    A = generateMatrix(7)
    B = generateMatrix(7)
    C = generateMatrix(7)
    D = generateMatrix(7)
    for i in range(7):
        for j in range(7):
            a = A[i][j] * float(Func1(i, j, point[0], point[1])) + B[i][j] * float(Func2(i, j, point[0], point[1]))

            b = C[i][j] * float(Func1(i, j, point[0], point[1])) + D[i][j] * float(Func2(i, j, point[0], point[1]))

    return (a ** 2 + b ** 2) ** (1 / 2)


def Fitness2(point):
    J = 0
    sum = 0.0
    for i in range(len(point)):
        sum = sum + (10 * math.cos(2 * math.pi * point[i]) - point[i]**2)
    denominator = 1 + np.exp(-1/(2*len(point)) * (-10 * len(point) + sum))
    J = 1/denominator
    return J


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


def gauss_func(xi, mat_wait, deviation):
    prob_i_comp_in_j_point = 1 / (np.sqrt(2 * np.pi) * deviation) * np.exp(
        -((xi - mat_wait) ** 2) / (2 * deviation ** 2))
    return prob_i_comp_in_j_point


def gauss(i, matrix, prob_list):
    list_i = []
    for j in range(len(prob_list)):
        list_i.append(matrix[j][i])
    mat_wait = 0
    for j in range(len(prob_list)):
        mat_wait += matrix[j][i] * (1 / len(prob_list))
    mat_wait_sqr = 0
    for j in range(len(prob_list)):
        mat_wait += (matrix[j][i]) ** 2 * (1 / len(prob_list))
    dispers = abs(mat_wait_sqr - mat_wait ** 2)
    middle_sqr_deviation = np.sqrt(dispers)
    prob_list_2 = []
    for j in range(len(prob_list)):
        prob_list_2.append(gauss_func(list_i[j], mat_wait, middle_sqr_deviation))
    choice = random.choices(list_i, weights=prob_list_2, k=1)
    return choice


def mutation(point, matrix, prob_list):
    tmp = []
    prob_list_2 = []
    for i in range(len(point)):
        tmp.append(point[i])
    for i in range(len(point)):
        prob_list_2.append(gauss(i, matrix, prob_list))
    for i in range(len(point)):
        tmp[i] = gauss(i, matrix, prob_list)
    return tmp


def SoFa(N, matrix):
    i = 0
    while i < N:
        fitness_list = []
        fitness_list = Fitness2(matrix)
        prob_list = GenerateProbability(fitness_list, len(matrix))
        choice = random.choices(matrix, weights=prob_list)
        choice2 = choice[0]
        # print('Выбор на шаге ', i+1, choice2)
        mutant = mutation(choice2, matrix, prob_list)
        mutant2 = []
        for j in range(len(mutant)):
            mutant2.append(mutant[j][0])
        matrix.append(mutant2)
        # print('матрица на шаге ',i+1, matrix )
        i += 1
    fitness_list = Fitness2(matrix)

    index = fitness_list.index(min(fitness_list))
    return matrix, matrix[index], min(fitness_list), index


'''''
testPopulation = generatePopulation(10, 4)
matrix = generateZeroMatrix(10, 3)
probmatrix = generateZeroMatrix(10, 3)

List_J = []
List_J_avg = []
List_x = []
for j in range(1, 50):

    for i in range(2):
        Population = generatePopulation(3,2)
        List_J.append(SoFa(40*j, Population)[2])

    print(mean(List_J))
    List_J_avg.append(mean(List_J))
    List_x.append(40*j)
plt.title('base SoFa')
plt.plot(List_x,List_J_avg)

plt.show()



for i in range(10):
    print(testPopulation[i])
print(' ')
second_population, maximum, ind = SoFa(10, testPopulation)

for i in range(20):
    print(second_population[i])
print(' ')
print(maximum, ind)

'''''


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


def selection(x, u):
    if Fitness2(u) >= Fitness2(x):
        y = u
    else:
        y = x
    return y


def DE_func(N, Population):
    F = 0.5
    CR = 0.8
    list_fit = []
    for i in range(N):
        v = mutantVector(F, Population[i], Population[random.randrange(N)], Population[random.randrange(N)])
        u = crossover(CR, Population[i], v)
        Population[i] = selection(Population[i], u)
    for i in range(N):
        list_fit.append(Fitness2(Population[i]))
    J = min(list_fit)
    mylist = [J, Population]
    return mylist


def DE_func_with_SoFa(N, firstPopulation):
    m = len(firstPopulation)
    n = len(firstPopulation[0])
    population = generateZeroMatrix(m, n)

    F = 0.5
    CR = 0.8
    for i in range(N):
        for j in range(m):
            fitness_list = []
            fitness_list = Fitness2(firstPopulation)
            prob_list = GenerateProbability(fitness_list, len(firstPopulation))
            choice = random.choices(firstPopulation, weights=prob_list)
            choice = choice[0]
            v = mutantVector(F, choice, firstPopulation[random.randrange(m)],
                             firstPopulation[random.randrange(m)])
            u = crossover(CR, firstPopulation[j], v)
            population[j] = selection(firstPopulation[j], u)
        # print('Current population:', population)
        firstPopulation = population
        population = generateZeroMatrix(m, n)

    FitnessMatrix = []
    for i in range(m):
        FitnessMatrix.append(Fitness(firstPopulation[i]))

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

firstPopulation = generatePopulation(N, M)

x_list = []
fitness_list = []
population = []

for i in range(2):
    population = firstPopulation
    List = []
    for j in range(20000):
        my_list = DE_func(N, population)
        for k in range(N):
            population[k] = my_list[1][k]
        List.append(my_list[0])
    if i != 0:
        for k in range(20000):
            fitness_list[k] = (fitness_list[k] + List[k])/2
    else:
        for k in range(20000):
            fitness_list.append(List[k])

for i in range(20000):
    x_list.append(i)

plt.title("DE_func")
plt.plot(x_list, fitness_list)
plt.show()
