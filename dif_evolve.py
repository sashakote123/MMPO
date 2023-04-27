import numpy as np
from graph import *
from matplotlib import pyplot as plt


def differential_evolution(f, bounds, pop_size, mut, crossp, maxiter, tol=1e-50):
    # f - функция, которую нужно минимизировать
    # bounds - границы для каждой переменной (например, [(0, 1), (2, 3), (-1, 1)])
    # pop_size - размер популяции (количество векторов)
    # mut - коэффициент мутации
    # crossp - вероятность кроссинговера
    # maxiter - максимальное количество итераций
    # tol - допустимая погрешность

    # Инициализация популяции
    dimensions = len(bounds)
    pop = np.random.rand(pop_size, dimensions)
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop_denorm = min_b + pop * diff

    pop_list = []
    itr = 0
    # Цикл оптимизации
    for i in range(maxiter):
        # Вычисление оценок для всех векторов
        cost = np.asarray([f(ind) for ind in pop_denorm])

        # Проверка на достижение заданной погрешности
        #if np.min(cost) < tol:
            #break

        # Создание новой популяции
        new_pop = np.empty([pop_size, dimensions])

        for j in range(pop_size):
            # Генерация случайных индексов в популяции
            idxs = list(range(pop_size))
            idxs.remove(j)
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]

            # Мутация
            mutant = np.clip(a + mut * (b - c), 0, 1)

            # Кроссинговер
            cross_points = np.random.rand(dimensions) < crossp
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True
            trial = np.where(cross_points, mutant, pop[j])

            # Денормализация
            trial_denorm = min_b + trial * diff

            # Сравнение и выбор лучшего вектора
            if f(trial_denorm) < cost[j]:
                new_pop[j] = trial
                cost[j] = f(trial_denorm)
            else:
                new_pop[j] = pop[j]
        best_idx = np.argmin(cost)
        pop_list.append(cost[best_idx])
        pop = new_pop
        pop_denorm = min_b + pop * diff
        itr += 1

    # Возвращение лучшего вектора и его оценки
    best_idx = np.argmin(cost)
    x_list = np.linspace(0, itr, itr)
    return min_b + pop[best_idx] * diff, cost[best_idx], pop_list, x_list


T = 5
N = 50
G = generateGraph(N, 3)


def Fitness2(point):
    return run2(G, T, point)[0]


def sphere(x):
    return np.sum(x ** 2)


bounds = [(0, 1)] * N
result, score, cost, itr = differential_evolution(Fitness2, bounds, N, 0.7, 0.3, 250)
print("Minimum:", result)
print("Score:", score)


plt.plot(itr, cost)
plt.show()