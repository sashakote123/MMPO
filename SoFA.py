
import timeit

code_to_test = """
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


t = 0
def adjust_dimension(matrix, t):
    population = matrix
    for i in range(len(population)):
        population[i].append(random.randrange(5)/(2**t))
    return population


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


def Fitness22(point):
    sum = 0.0
    for i in range(len(point)):
        sum = sum + (10 * math.cos(2 * math.pi * point[i]) - point[i]**2)
    denominator = 1 + np.exp(-1/(2*len(point)) * (-10 * len(point) + sum))
    J = 1/denominator
    return J


def Fitness2(point):
    x_mean = np.mean(point)
    denominator = 1 + np.exp(5 - ((10*np.cos(2*np.pi*x_mean) - x_mean**2)**2)/2)
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


def gauss1(i, matrix):
    list_i = []
    for j in range(len(matrix)):
        list_i.append(matrix[j][i])
    list_i_prob = random.choices(list_i, weights=list_i, k=len(matrix))
    mat_wait = 0
    for j in range(len(matrix)):
        mat_wait += matrix[j][i] * list_i_prob[j]
    mat_wait_sqr = 0
    for j in range(len(matrix)):
        mat_wait_sqr += ((matrix[j][i]) ** 2) * list_i_prob[j]
    dispers = mat_wait_sqr - mat_wait ** 2
    middle_sqr_deviation = np.sqrt(dispers)
    prob_list_2 = []
    for j in range(len(matrix)):
        prob_list_2.append(gauss_func(list_i[j], mat_wait, middle_sqr_deviation))
    choice = random.choices(list_i, weights=prob_list_2, k=1)
    choice = choice[0]
    return choice


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
    mut = []
    for i in range(len(point)):
        tmp = gauss(i, matrix)
        if tmp > 10:
            mut.append(10)
        elif tmp > 0:
            mut.append(gauss(i, matrix))
        else:
            mut.append(0)
    return mut


## max
def SoFA(population):
    fitness_population = []
    for i in range(len(population)):
        fitness_population.append(Fitness2(population[i]))
    probability_population = GenerateProbability(fitness_population, len(population))

    choice = random.choices(population, weights=probability_population, k=1)
    choice = choice[0]

    mutant = mutation(choice, population)
    #print(choice, mutant)
    population.append(mutant)
    fitness_population.clear()
    for i in range(len(population)):
        fitness_population.append(Fitness2(population[i]))
    fit = max(fitness_population)
    return fit, population


# SoFA_base

top = 10
bottom = 0

itr = 1000    # Число итераций
dim_up = 1000000   # Контроль частоты увеличения размерности пространства параметров

NP = 5
M = 5

population = []
x_list = []
Fitness_list = []

for i in range(1):
    population.clear()
    population = generatePopulation(NP, M)
    List_Fit_in_one_go = []
    fitness_population = []

    for j in range(len(population)):
        fitness_population.append(Fitness2(population[i]))

    if i == 0:
        Fitness_list.append(max(fitness_population))
    else:
        List_Fit_in_one_go.append(max(fitness_population))

    for j in range(1, itr):
        temp_J, population = SoFA(population)
        List_Fit_in_one_go.append(temp_J)
     ## Рост размерности:
        if j % dim_up == 0:
            population = adjust_dimension(population, t)
            t += 1
     ##
    if i != 0:
        for k in range(itr):
            Fitness_list[k] = (Fitness_list[k] + List_Fit_in_one_go[k])/2
    else:
        for k in range(1, itr):
            Fitness_list.append(List_Fit_in_one_go[k-1])


for i in range(itr):
    x_list.append(i)


#plt.title("SoFA_base")
#plt.plot(x_list, Fitness_list)
#plt.show()

"""

elapsed_time = timeit.timeit(code_to_test, number=3)/3
print('Скорость SoFA_base', elapsed_time, 'секунд')