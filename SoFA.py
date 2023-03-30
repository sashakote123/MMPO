
import timeit


import random
import math
import numpy as np
from statistics import mean
from matplotlib import pyplot as plt





def adjust_dimension(matrix, t):
    population = matrix
    for i in range(len(population)):
        population[i].append(random.randrange(5)/(2**t))
    return population






def Fitness22(point):
    x_mean = np.mean(point)
    denominator = 1 + np.exp(5 - ((10*np.cos(2*np.pi*x_mean) - x_mean**2)**2)/2)
    J = 1/denominator
    return J

def Fitness2(point):
    x_mean = np.mean(point)
    return x_mean**2-(10*np.cos(2*np.pi*x_mean)+10)

def Fitness22(point):
    x_mean = np.mean(point)
    return (x_mean-2**(1/2))**2-1


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
    population.append(mutant)
    fitness_population.clear()
    for i in range(len(population)):
        fitness_population.append(Fitness2(population[i]))
    fit = min(fitness_population)
    return fit, population


# SoFA_base


def GetLists(itr):
 # Число итераций
    dim_up = 20000 # Контроль частоты увеличения размерности пространства параметров

    NP = 10
    M = 10


    t = 0
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
            Fitness_list.append(min(fitness_population))
        else:
            List_Fit_in_one_go.append(min(fitness_population))

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

plt.title("SoFA")
plt.plot(data[0], data[1])
plt.show()
