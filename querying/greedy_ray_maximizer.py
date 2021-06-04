import graph_tool as gt
import numpy as np
from graph_tool.topology import shortest_distance

from convexity_operations.convex_hulls import numba_based_interval, numba_based_extension


def binary_greedy_ray_maximizer(g: gt.Graph, labels, weights=None, dists=None, budget=-1, starting_vertex=None):
    n = g.num_vertices()
    accs = []
    pos = np.zeros(n, dtype=np.bool)
    neg = np.zeros(n, dtype=np.bool)

    neg_label, pos_label, = np.unique(labels)

    if budget < 0:
        budget = n

    # what happens with the hulls if we assume negative or positive candidates
    # could also instead of saving all only remembering the current best one. space n^2 --> n

    # gain the following two if candidate is positive:
    pos_hull_with_pos_candidate = np.zeros((n, n), dtype=np.bool)  # neg/I(pos+x) "/" is extension/ray
    neg_hull_with_pos_candidate = np.zeros((n, n), dtype=np.bool)  # I(pos+x)/neg
    # gain the following two if candidate is negative:
    pos_hull_with_neg_candidate = np.zeros((n, n), dtype=np.bool)  # I(neg+x)/pos
    neg_hull_with_neg_candidate = np.zeros((n, n), dtype=np.bool)  # pos/I(neg+x)

    queries = []

    for b in range(budget):
        print(b, np.sum(pos) + np.sum(neg))
        for candidate_vertex in range(n):
            if pos[candidate_vertex] or neg[candidate_vertex]:
                continue

            # try candidate pos
            candidate_pos_class = np.append(np.where(pos)[0], candidate_vertex)
            pos_hull_with_pos_candidate[candidate_vertex] = numba_based_interval(candidate_pos_class, dists)

            candidate_pos_class = np.where(pos_hull_with_pos_candidate[candidate_vertex])[0]
            candidate_neg_class = np.where(neg)[0]
            if candidate_neg_class.size > 0:
                pos_hull_with_pos_candidate[candidate_vertex] = numba_based_extension(candidate_neg_class,
                                                                                      candidate_pos_class, dists)
                neg_hull_with_pos_candidate[candidate_vertex] = numba_based_extension(candidate_pos_class,
                                                                                      candidate_neg_class, dists)

            # try candidate neg
            candidate_neg_class = np.append(np.where(neg)[0], candidate_vertex)
            neg_hull_with_neg_candidate[candidate_vertex] = numba_based_interval(candidate_neg_class, dists)

            candidate_neg_class = np.where(neg_hull_with_neg_candidate[candidate_vertex])[0]
            candidate_pos_class = np.where(pos)[0]
            if candidate_pos_class.size > 0:
                neg_hull_with_neg_candidate[candidate_vertex] = numba_based_extension(candidate_pos_class,
                                                                                      candidate_neg_class, dists)
                pos_hull_with_neg_candidate[candidate_vertex] = numba_based_extension(candidate_neg_class,
                                                                                      candidate_pos_class, dists)

        # select the one vertex greedily that maximizes the number of gained labels . On a single path this is binary search
        # not really gains, rather total known labels
        # is this related to binary search on graphs paper thingy?!?!? i.e. find one cut edge with log queries???
        pos_gains = np.sum(pos_hull_with_pos_candidate ^ neg_hull_with_pos_candidate, axis=1)
        neg_gains = np.sum(neg_hull_with_neg_candidate ^ pos_hull_with_neg_candidate, axis=1)
        pos_gains[pos] = neg_gains[pos] = -1
        pos_gains[neg] = neg_gains[neg] = -1

        min_gains = np.min(np.column_stack((pos_gains, neg_gains)), axis=1)
        maximizers = np.where(min_gains == np.max(min_gains))[0]

        if b > 0 or starting_vertex is None:
            candidate_vertex = np.random.choice(maximizers, size=1)[0]
        else:
            candidate_vertex = starting_vertex

        # "query" it
        if labels[candidate_vertex] == pos_label:
            pos = pos_hull_with_pos_candidate[candidate_vertex]
            neg = neg_hull_with_pos_candidate[candidate_vertex]
        else:
            neg = neg_hull_with_neg_candidate[candidate_vertex]
            pos = pos_hull_with_neg_candidate[candidate_vertex]

        majority = (np.sum(pos) >= np.sum(neg))
        prediction = np.ones(g.num_vertices(), dtype=np.bool)*majority

        prediction[pos] = True
        prediction[neg] = False


        acc = np.sum(prediction == labels) / labels.size
        accs.append(acc)
        print("accuracy: %0.2f" % (acc))

        known_vertices = []
        known_vertices.extend(list(np.where(pos)[0]))
        known_vertices.extend(list(np.where(neg)[0]))
        queries.append(known_vertices)
        # print(labels)
        # print(pos ^ neg)
        if np.sum(pos) + np.sum(neg) == n:
            break

    return queries, accs

def binary_cal_style(g: gt.Graph, labels, weights=None, dists=None, budget=-1, starting_vertex = None):
    '''
    selective sampling
    :param g:
    :param labels:
    :param weights:
    :param dists:
    :param budget:
    :return:
    '''
    n = g.num_vertices()
    accs = []
    pos = np.zeros(n, dtype=np.bool)
    neg = np.zeros(n, dtype=np.bool)

    neg_label, pos_label, = np.unique(labels)

    if budget < 0:
        budget = n

    queries = []

    for b in range(budget):
        print(b)

        pos_class = np.where(pos)[0]
        pos = numba_based_interval(pos_class, dists)
        pos_class = np.where(pos)[0]

        neg_class = np.where(neg)[0]
        neg = numba_based_interval(neg_class, dists)
        neg_class = np.where(neg)[0]

        pos = numba_based_extension(neg_class, pos_class, dists)
        neg = numba_based_extension(pos_class, neg_class, dists)

        # deduced all labels
        if np.sum(pos) + np.sum(neg) == n:
            known_vertices = []
            known_vertices.extend(list(np.where(pos)[0]))
            known_vertices.extend(list(np.where(neg)[0]))
            queries.append(known_vertices)

            majority = (np.sum(pos) >= np.sum(neg))
            prediction = np.ones(g.num_vertices(), dtype=np.bool) * majority

            prediction[pos] = True
            prediction[neg] = False

            acc = np.sum(prediction == labels) / labels.size
            accs.append(acc)

            print("accuracy: %0.2f" % (acc))
            break

        if np.where(~(pos | neg))[0].size == 0:
            return queries, accs

        if b > 0 or starting_vertex is None:
            candidate_vertex = np.random.choice(np.where(~(pos | neg))[0], size=1)[0]
        else:
            candidate_vertex = starting_vertex


        # "query" it
        if labels[candidate_vertex] == pos_label:
            pos[candidate_vertex] = True
        else:
            neg[candidate_vertex] = True
        majority = (np.sum(pos) >= np.sum(neg))
        prediction = np.ones(g.num_vertices(), dtype=np.bool) * majority

        prediction[pos] = True
        prediction[neg] = False

        acc = np.sum(prediction == labels) / labels.size
        accs.append(acc)
        print("accuracy: %0.2f" % (acc))

        known_vertices = []
        known_vertices.extend(list(np.where(pos)[0]))
        known_vertices.extend(list(np.where(neg)[0]))
        queries.append(known_vertices)
    return queries, accs

def binary_random_sampling(g: gt.Graph, labels, weights=None, dists=None, budget=-1):
    n = g.num_vertices()
    pos = np.zeros(n, dtype=np.bool)
    neg = np.zeros(n, dtype=np.bool)

    neg_label, pos_label, = np.unique(labels)

    if budget < 0:
        budget = n

    queries = []
    for_evaluation = []
    for b in range(budget):
        print(b)

        # deduced all labels
        if np.sum(pos) + np.sum(neg) == n:
            break

        candidate_vertex = np.random.choice(np.where(~(pos | neg))[0], size=1)[0]

        # "query" it
        queries.append(candidate_vertex)
        for_evaluation.append(queries.copy())
        if labels[candidate_vertex] == pos_label:
            pos[candidate_vertex] = True
        else:
            neg[candidate_vertex] = True

    return for_evaluation

if __name__ == '__main__':
    g = gt.Graph(directed=False)

    n = 500

    edges = np.column_stack((np.arange(n - 1), np.arange(1, n)))

    g.add_edge_list(edges)

    labels = np.ones(n, dtype=np.bool)
    labels[:n // 2] = False

    distances = shortest_distance(g)
    distance_matrix = distances.get_2d_array(range(g.num_vertices()))

    #binary_greedy_ray_maximizer(g, labels, dists=distance_matrix)

    binary_cal_style(g, labels, dists=distance_matrix)
