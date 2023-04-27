import networkx as nx
import matplotlib.pyplot as plt
import random

import numpy as np


def generateGraph(n, m):
    # создаем начальный граф из m вершин
    G = nx.complete_graph(m)

    # добавляем новые вершины
    for i in range(m, n):
        # выбираем m вершин из существующих
        targets = []
        while len(targets) < m:
            # выбираем случайную вершину
            node = random.choice(list(G.nodes()))
            # проверяем, не добавляли ли мы уже эту вершину
            if node not in targets:
                targets.append(node)
        # добавляем новую вершину и соединяем ее с выбранными вершинами
        G.add_node(i)
        for target in targets:
            G.add_edge(i, target)

        for i in range(len(G.nodes)):
            G.nodes[i]['p'] = random.random()
            G.nodes[i]['betta'] = random.random()
            G.nodes[i]['gamma'] = random.random()
            G.nodes[i]['isSick'] = False
            for j in G[i]:
                G[i][j]['w'] = random.random()

    return G


# def generateNewGraph()

def newDay(G):
    for i in range(len(G.nodes)):
        for j in G[i]:
            G[i][j]['w'] = random.random()
    return G


def external_probability(p_list, omega, beta_list, nodes_list):
    N = len(p_list)
    p_vnehn_list = []
    for j in range(N):
        p_tmp = 0
        for i in range(nodes_list[j]):
            p_tmp += omega * beta_list[j] * p_list[j]
        p_vnehn_list.append(p_tmp)
    return p_vnehn_list


def GetNewProbability(G):
    p_list = []
    a = 0
    for i in range(len(G.nodes)):
        for j in G[i]:
            a += G[i][j]['w'] * G.nodes[j]['betta'] * G.nodes[j]['p']
        p_list.append((1 - G.nodes[i]['p']) * a - G.nodes[i]['gamma'] * G.nodes[i]['p'] + G.nodes[i]['p'])
        if p_list[i] < 0:
            p_list[i] = 0

        if p_list[i] > 1:
            p_list[i] = 1
        a = 0

    for i in range(len(G.nodes)):
        G.nodes[i]['p'] = p_list[i]
        G.nodes[i]['isSick'] = random.choices([True, False], weights=[G.nodes[i]['p'], 1 - G.nodes[i]['p']])[0]
    return G


def func(p):
    return np.sqrt(p)


def run(G, t):
    sum = 0
    w_list = np.zeros((t, len(G.nodes), len(G.nodes)))

    for i in range(len(G.nodes)):
        sum += func(G.nodes[i]['p'])

    T = np.arange(1, t, 1)
    for j in T:
        G = GetNewProbability(G)

        for i in range(len(G.nodes)):
            sum += func(G.nodes[i]['p'])
            #print(f"Нода номер: {i}, p =  {G.nodes[i]['p']},w={G[i]},\n Статус: {G.nodes[i]['isSick']} ")
            #print('\n')
            for k in G[i]:
                if i < k:
                    w_list[j][i][k] = G[i][k]['w']
        G = newDay(G)
    print(w_list)
    return sum, w_list

def run2(G, t, x_list):
    sum = 0
    count = 0
    w_list = np.zeros((t, len(G.nodes), len(G.nodes)))

    for i in range(len(G.nodes)):
        sum += func(G.nodes[i]['p'])

    T = np.arange(1, t, 1)
    for j in T:
        G = GetNewProbability(G)
        if G.nodes[j]['isSick'] == True:
            count += 1
        for i in range(len(G.nodes)):
            sum += func(G.nodes[i]['p'])
            #print(f"Нода номер: {i}, p =  {G.nodes[i]['p']},w={G[i]},\n Статус: {G.nodes[i]['isSick']} ")
            #print('\n')
            for k in G[i]:
                if i < k:
                    w_list[j][i][k] = G[i][k]['w']
        G = newDay2(G, x_list)
    print(w_list)
    print(f'Заболевших:{count}')
    return sum, w_list, count

def newDay2(G, x_list):
    for i in range(len(G.nodes)):
        for j in G[i]:
            G[i][j]['w'] = x_list[i]
    return G


T = 5
N = 5
G = generateGraph(N, 3)


w_list = run(G, T)[1]


np.savetxt('file.txt',)
#subax1 = plt.plot()
#nx.draw(G, with_labels=True, node_color='green')
# plt.show()
