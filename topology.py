import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from collections import Counter, deque, defaultdict


def generate_matrix(n):
    # 生成一个n*n的随机矩阵，元素值以1/2的概率为0或1
    matrix = np.random.choice([0, 1], size=(n, n), p=[0.5, 0.5])

    # 确保每一行至少有一个1
    for i in range(n):
        if not np.any(matrix[i, :]):
            col_index = np.random.choice(n)
            matrix[i, col_index] = 1

    # 确保每一列至少有一个1
    for j in range(n):
        if not np.any(matrix[:, j]):
            row_index = np.random.choice(n)
            matrix[row_index, j] = 1

    for c in range(n):
        matrix[c, c] = 1

    return matrix


def save_and_plt(matrix):
    n = len(matrix)
    df = pd.DataFrame(matrix)
    df.to_csv('adjacency_matrix.csv', index=False, header=False)

    # 绘制矩阵
    fig, ax = plt.subplots()
    cax = ax.matshow(matrix, cmap='Reds')
    plt.title('Adjacency Matrix of Topology')
    plt.xlabel('Receive')
    plt.ylabel('Send')

    # 设置坐标轴的刻度位置
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))

    # 设置坐标轴的刻度标签
    ax.set_xticklabels(np.arange(1, n + 1))
    ax.set_yticklabels(np.arange(1, n + 1))

    # 在每个格子中添加数值标签
    for (i, j), val in np.ndenumerate(matrix):
        if val == 0:
            ax.text(j, i, val, ha='center', va='center', color='black')
        else:
            ax.text(j, i, val, ha='center', va='center', color='white')

    # 添加网格线
    # ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    # ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    # ax.grid(which='minor', color='black', linestyle='-', linewidth=1)

    # plt.colorbar(cax, label='Value (0 or 1)')
    plt.show()


def matrix_to_adj_list(matrix):
    # 将矩阵转换为邻接表
    adj_list = {}
    for i in range(len(matrix)):
        adj_list[i + 1] = []
        for j in range(len(matrix)):
            if matrix[i, j] == 1 and i != j:
                adj_list[i + 1].append(j + 1)
    return adj_list


def dfs(graph, start, end, path=[]):
    # 使用dfs算法遍历所有的路径
    path = path + [start]
    if start == end:
        return [path]
    if start not in graph:
        return []
    paths = []
    for node in graph[start]:
        if node not in path:
            new_paths = dfs(graph, node, end, path)
            for new_path in new_paths:
                paths.append(new_path)
    return paths


def bfs_all_shortest_paths(matrix, start):
    n = len(matrix)  # 节点数量
    visited = [False] * (n + 1)  # 访问标记数组，多一个元素，以便从1开始索引
    distances = [-1] * (n + 1)  # 存储从start到每个节点的最短路径长度，从1开始索引
    paths = defaultdict(list)  # 存储所有最短路径的字典，键是目标节点，值是路径列表
    queue = deque([[start]])  # 队列中存储的是路径而不是单个节点

    visited[start] = True
    distances[start] = 0
    paths[start].append([start])  # 初始节点的路径是它自己

    while queue:
        path = queue.popleft()  # 取出路径
        current = path[-1]  # 当前节点是路径的最后一个元素

        for i in range(1, n + 1):
            if matrix[current - 1][i - 1] == 1:  # 调整索引以适应从1开始的设定
                if visited[i] and len(path) > distances[i]:
                    continue  # 如果已访问且当前路径不是最短的，则跳过
                elif not visited[i] or len(path) == distances[i]:
                    if not visited[i]:
                        visited[i] = True
                        distances[i] = len(path)
                        queue.append(path + [i])
                    paths[i].append(path + [i])

    paths = dict(paths)

    dict_paths = {}
    for node in paths.keys():
        if node != start:
            dict_paths[(start, node)] = paths[node]

    return dict_paths


def all_send_path(matrix, verbose=False):
    adj_list = matrix_to_adj_list(matrix)
    all_paths = {}
    for i in range(len(matrix)):
        # 选择聚合节点
        send_node = i+1
        for j in range(len(matrix)):
            if i != j:
                receive_node = j + 1
                paths = dfs(adj_list, send_node, receive_node)
                all_paths[(i+1, j+1)] = paths

    if verbose:
        for key, value in all_paths.items():
            print(key, value)
    return all_paths


def generate_sequences(counts):
    # 生成各个位置的数字范围
    ranges = [range(1, x + 1) for x in counts]
    # 生成所有可能的组合
    product = itertools.product(*ranges)
    # 将组合转换为特定格式的字符串
    result = [''.join(str(x) for x in combo) for combo in product]
    return result


def paths_to_trees(paths, limit, verbose=False):
    node_trees = []
    for path in paths:
        tree_length = max(len(p) for p in path)-1
        node_tree = {key+1:[] for key in range(tree_length)}
        for p in path:
            for i in range(len(p)-1):
                if (p[i], p[i+1]) not in node_tree[i+1]:
                    node_tree[i+1].append((p[i], p[i+1]))
        length = np.sum([len(couple) for couple in node_tree.values()])
        if length <=limit:
            if verbose:
                print(node_tree)
            node_trees.append(node_tree)

    return node_trees


def pull_time_calculating(node_tree):
    t = 0
    for value in node_tree.values():
        node = []
        for i, j in value:
            node.append(i)
        t = t + max(node.count(item) for item in set(node))

    return t

# def push_time_calculating(node_tree):


def f_using_dfs(matrix, agg=1):
    # 获得当前聚合节点下发模型的渠道
    all_paths = all_send_path(matrix)
    agg_send_paths = {}
    for node in range(1, len(matrix)+1):
        if agg != node:
            agg_send_paths[(agg, node)] = all_paths[(agg, node)]

    # 对渠道进行组合
    couple_list = []
    for key, value in agg_send_paths.items():
        couple_list.append(len(value))

    # 生成索引
    index_list = generate_sequences(couple_list)
    print(index_list)

    # 生成路径并进行合并
    paths = []
    for index in index_list:
        path = []
        for i, value in enumerate(agg_send_paths.values()):
            path.append(value[int(index[i])-1])
        print(path)
        paths.append(path)

    node_trees = paths_to_trees(paths)
    t_min = None
    for node_tree in node_trees:
        t = pull_time_calculating(node_tree)
        if t_min is None or t <= t_min:
            t_min = t
            best_tree = node_tree

    return t_min, best_tree


def f_using_bfs(matrix, agg=1, verbose=False):
    all_paths = bfs_all_shortest_paths(matrix, agg)

    couple_list = []
    for key, value in all_paths.items():
        couple_list.append(len(value))

    index_list = generate_sequences(couple_list)

    paths = []
    for index in index_list:
        path = []
        for i, value in enumerate(all_paths.values()):
            path.append(value[int(index[i])-1])
        # print(path)
        paths.append(path)

    node_trees = paths_to_trees(paths, len(matrix)-1, verbose)
    t_min = None
    for node_tree in node_trees:
        t = pull_time_calculating(node_tree)
        if t_min is None or t <= t_min:
            t_min = t
            best_tree = node_tree

    return t_min, best_tree


if __name__ == '__main__':
    # 生成矩阵并进行绘制
    np.random.seed(12345)

    n = 8
    result_matrix = generate_matrix(n)
    save_and_plt(result_matrix)

    for i in range(1, n+1):
        t_min, best_tree = f_using_bfs(result_matrix, agg=i)
        print(t_min)
