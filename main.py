import random
from random import randint
import math


# import numpy as np

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




#def CreatePoint():
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



def generatePopulation(sizeN, sizeM):
    matrix = [[random.randrange(10) for j in range(sizeM)] for i in range(sizeN)]
    return matrix

firstPopulation = generatePopulation(10, 3)
for i in range(10):
    print(firstPopulation[i])


matrix = generateZeroMatrix(10, 3)
matrix2 = generateZeroMatrix(10, 3)

for i in range(10):
    matrix[i] = Fitness(firstPopulation[i])
print(matrix)

sum = 0
for i in range(10):
    sum = sum + matrix[i]

for i in range(10):
     matrix2[i] = matrix[i]/sum

print(matrix2)

vector = [[0.1,0.22,0.33,0.51,0.12,0.53,0.1,0.22,0.33,0.51,0.12,0.53], [4,5,6,4,5,6]]

a = math.sin(math.pi*(vector[0][0]-0.5))
for i in range(1, len(vector[0])):
    print(math.sin(math.pi*(vector[0][i]-0.5)), '  ', vector[0][i])
    a = a * math.sin(math.pi*(vector[0][i]-0.5))
    print(a)
