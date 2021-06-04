import itertools

import graph_tool
import graph_tool.topology
import numpy as np

from labelled_graph_primitives.cuts import get_cut_vertices
from prediction_strategies.labelpropgation import label_propagation

'''
this is a naive implementation of Dasarathy et al.'s S^2 '15
'''


def local_global_strategy(Y, W, alpha=0.5, iterations=200, eps=0.000001):
    np.fill_diagonal(W, 0)
    D = np.sum(W, axis=0)
    if np.any(D == 0):
        D += D[D > 0].min() / 2
    Dhalfinverse = 1 / np.sqrt(D)
    Dhalfinverse = np.diag(Dhalfinverse)
    S = np.dot(np.dot(Dhalfinverse, W), Dhalfinverse)

    F = np.zeros((Y.shape[0], Y.shape[1]))
    oldF = np.ones((Y.shape[0], Y.shape[1]))
    oldF[:Y.shape[1], :Y.shape[1]] = np.eye(Y.shape[1])
    i = 0
    while (np.abs(oldF - F) > eps).any() or i >= iterations:
        oldF = F
        F = np.dot(alpha * S, F) + (1 - alpha) * Y

    result = np.zeros(Y.shape[0])
    # uniform argmax
    for i in range(Y.shape[0]):
        result[i] = np.random.choice(np.flatnonzero(F[i] == F[i].max()))

    return result

    # return np.argmax(F, axis=1)


def label_propagation2(W, known_labels, labels):
    W = np.exp(-W * W / 2)  # similarity
    Y = np.zeros((W.shape[0], labels.size))

    for i, label in enumerate(labels):
        Y[known_labels == label, i] = 1

    return local_global_strategy(Y, W)


def mssp(g: graph_tool.Graph, weight_prop: graph_tool.EdgePropertyMap, L, known_labels):
    n = g.num_vertices()
    dist_map = np.ones((n, n)) * np.inf

    for i, j in itertools.combinations(L, 2):
        if known_labels[i] != known_labels[j]:
            dist_map[i, j] = graph_tool.topology.shortest_distance(g, i, j, weight_prop)

    i, j = np.unravel_index(dist_map.argmin(), dist_map.shape)

    if weight_prop is None:
        total_weight = g.num_edges() + 1
    else:
        total_weight = np.sum(weight_prop.a) + 1

    if dist_map[i, j] < total_weight:

        path, _ = graph_tool.topology.shortest_path(g, i, j, weight_prop)
        mid_point = path[len(path) // 2]
        return mid_point
    else:
        return None


def s2(g: graph_tool.Graph, weight_prop: graph_tool.EdgePropertyMap, labels, budget=20, use_adjacency=False, starting_vertex = None):
    L = set()

    n = g.num_vertices()

    known_labels = -np.ones(n) * np.inf

    W = graph_tool.topology.shortest_distance(g, weights=weight_prop).get_2d_array(range(n))  # original distance map

    if starting_vertex is None:
        x = np.random.choice(list(set(range(n)).difference(L)))
    else:
        x = starting_vertex

    true_cut = get_cut_vertices(g, labels)

    cut_vertices = set()
    total_budget = budget

    queries = []
    removed_edges = []
    accs = []
    while budget > 0:
        known_labels[x] = labels[x]
        L.add(x)
        if len(L) == n:
            break
        budget -= 1
        to_remove = []
        for e in g.get_out_edges(x):
            if known_labels[e[1]] > -np.inf and known_labels[e[1]] != known_labels[x]:
                to_remove.append(e)
                cut_vertices.add(e[0])
                cut_vertices.add(e[1])

        for e in to_remove:
            g.remove_edge(g.edge(e[0], e[1]))
            removed_edges.append(e)

        mid_point = mssp(g, weight_prop, L, known_labels)

        if mid_point is not None:
            x = int(mid_point)
        else:
            x = np.random.choice(list(set(range(n)).difference(L)))

        queries.append(list(L))
        prediction = label_propagation(W, known_labels, labels, use_adjacency=use_adjacency)
        np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
        larger_class = max(np.where(labels == labels[0])[0].size,
                           labels.size - np.where(labels == labels[0])[0].size) / labels.size

        acc = np.sum(prediction == labels) / labels.size
        accs.append(acc)
        print("labels:  %2d/%2d  (%0.2f),  cut_vertices: %2d/%2d (%0.2f), accuracy: %0.2f, larger_class: %0.2f" % (
            total_budget - budget, total_budget, (total_budget - budget) / total_budget, len(cut_vertices),
            len(true_cut),
            len(cut_vertices) / len(true_cut), acc, larger_class))
        # print("accuracy", np.sum(prediction == labels) / labels.size)

        if len(cut_vertices) == len(true_cut):
            break
    g.add_edge_list(removed_edges)
    return queries, accs

def random_not_s2(g: graph_tool.Graph, weight_prop: graph_tool.EdgePropertyMap, labels, budget=20, use_adjacency=False, starting_vertex=None):
    L = set()

    n = g.num_vertices()

    known_labels = -np.ones(n) * np.inf

    W = graph_tool.topology.shortest_distance(g, weights=weight_prop).get_2d_array(range(n))  # original distance map

    if starting_vertex is None:
        x = np.random.choice(list(set(range(n)).difference(L)))
    else:
        x = starting_vertex

    true_cut = get_cut_vertices(g, labels)

    cut_vertices = set()
    total_budget = budget

    queries = []
    removed_edges = []
    accs = []
    while budget > 0:
        known_labels[x] = labels[x]
        L.add(x)
        if len(L) == n:
            break
        budget -= 1

        mid_point = None#mssp(g, weight_prop, L, known_labels)

        if mid_point is not None:
            x = int(mid_point)
        else:
            x = np.random.choice(list(set(range(n)).difference(L)))

        queries.append(list(L))
        prediction = label_propagation(W, known_labels, labels, use_adjacency=use_adjacency)
        np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
        larger_class = max(np.where(labels == labels[0])[0].size,
                           labels.size - np.where(labels == labels[0])[0].size) / labels.size

        acc = np.sum(prediction == labels) / labels.size
        accs.append(acc)
        print("labels:  %2d/%2d  (%0.2f),  cut_vertices: %2d/%2d (%0.2f), accuracy: %0.2f, larger_class: %0.2f" % (
            total_budget - budget, total_budget, (total_budget - budget) / total_budget, len(cut_vertices),
            len(true_cut),
            len(cut_vertices) / len(true_cut), acc, larger_class))
        # print("accuracy", np.sum(prediction == labels) / labels.size)

        if len(cut_vertices) == len(true_cut):
            break
    g.add_edge_list(removed_edges)
    return queries, accs
