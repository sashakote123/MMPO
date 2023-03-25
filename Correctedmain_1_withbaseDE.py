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



def Fitness2(vector):
    list = []

    for j in range(len(vector)):
        sum = 10 * math.cos(2 * math.pi * vector[j][0]) - vector[j][0] ** 2
        for i in range(1, len(vector[j])):
            sum = sum + 10 * math.cos(2 * math.pi * vector[j][i]) - vector[j][i] ** 2
        e = math.exp(-(1 / (2 * len(vector[j]))) * (-10 * len(vector[j]) + sum))
        list.append(1 / (1 + e))
    return list

def Fitness3(vector):
    #list = []

    sum = 10 * math.cos(2 * math.pi * vector[0]) - vector[0] ** 2
    for i in range(1, len(vector)):
        sum = sum + 10 * math.cos(2 * math.pi * vector[i]) - vector[i] ** 2
    e = math.exp(-(1 / (2 * len(vector))) * (-10 * len(vector) + sum))
    #list.append(1 / (1 + e))

    return (1/(1+e))

def generatePopulation(sizeN, sizeM):
    matrix = [[random.random() for j in range(sizeM)] for i in range(sizeN)]
    return matrix


def addDimension(matrix):
    for i in range(len(matrix)):
        matrix[i].append(random.random())
    print('dimension is ', len(matrix[0]))
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




def SoFa1(N, matrix):
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
    #fitness_list = Fitness2(matrix)

    #index = fitness_list.index(min(fitness_list))
    return min(Fitness2(matrix))


testPopulation = generatePopulation(1, 1)
matrix = generateZeroMatrix(10, 3)
probmatrix = generateZeroMatrix(10, 3)

# print(testPopulation)
# addDimension(testPopulation)
# print(testPopulation)





List_J_avg = []
List_x = []
for j in range(1, 60):
    List_J = []
    for i in range(2):
        Population = generatePopulation(2,3)
        List_J.append(SoFa(40*j, Population))

    print(mean(List_J))
    List_J_avg.append(mean(List_J))
    List_x.append(40*j)
plt.title('base SoFa')
plt.plot(List_x,List_J_avg)

plt.show()




