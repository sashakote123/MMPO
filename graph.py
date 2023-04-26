import networkx as nx
import matplotlib.pyplot as plt
import random

n = 5  # число вершин
m = 2  # средняя степень вершин

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

subax1 = plt.plot()
nx.draw(G, with_labels=True, node_color='green')
print(nx.edges(G))

for i in range(n):
    print(nx.edges(G, i))
    print(len(nx.edges(G, i)))

T = 2
N = n
omega = 0.5
p_list = []
beta_list = []
gamma_list = []
nodes_list = []

for i in range(N):
    p_list.append(0.2)
    beta_list.append(random.random())
    gamma_list.append(random.random())
    nodes_list.append(len(nx.edges(G, i)))


def external_probability(p_list, omega, beta_list, nodes_list):
    N = len(p_list)
    p_vnehn_list = []
    for j in range(N):
        p_tmp = 0
        for i in range(nodes_list[j]):
            p_tmp += omega * beta_list[j] * p_list[j]
        p_vnehn_list.append(p_tmp)
    return p_vnehn_list


p1_list = []
for i in range(N):
    p1_list.append((1 - p_list[i]) * p_vnehn_list[i] - gamma_list[i] * p_list[i] + p_list[i])

print(p_vnehn_list)
print(p1_list)

for i in range(N):
    p_list[i] = p1_list[i]

p_vnehn_list = []
for j in range(N):
    p_tmp = 0
    for i in range(nodes_list[j]):
        p_tmp += omega * beta_list[j] * p_list[j]
    p_vnehn_list.append(p_tmp)

p1_list = []
for i in range(N):
    p1_list.append((1 - p_list[i]) * p_vnehn_list[i] - gamma_list[i] * p_list[i] + p_list[i])

print(p_vnehn_list)
print(p1_list)

for i in range(N):
    p_list[i] = p1_list[i]

p_vnehn_list = []
for j in range(N):
    p_tmp = 0
    for i in range(nodes_list[j]):
        p_tmp += omega * beta_list[j] * p_list[j]
    p_vnehn_list.append(p_tmp)

p1_list = []
for i in range(N):
    p1_list.append((1 - p_list[i]) * p_vnehn_list[i] - gamma_list[i] * p_list[i] + p_list[i])

print(p_vnehn_list)
print(p1_list)

plt.show()
