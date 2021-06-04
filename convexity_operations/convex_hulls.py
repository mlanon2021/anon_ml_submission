import random

import graph_tool as gt
import numpy as np
from graph_tool.search import dijkstra_search, bfs_search, bfs_iterator
from numba import njit, prange

from read_graphs.read import build_graph


@njit(parallel=True)
def numba_based_interval(F, dists, weights_upper_bound=2000000):
    '''
    makes sense for large F, a lot (>= 3*) faster than single loop python version
    currently only undirected!
    :param F:
    :param dists:
    :return:
    '''
    n = dists.shape[0]
    # naive double loop implementation:
    interval = np.zeros(n, dtype=np.bool_)

    # allow f==g for the case if |F|=1

    if F.size == 1:
        interval[F] = True
        return interval

    for f in prange(F.size - 1):
        for g in range(f, F.size):
            f_g_dist = dists[F[f], F[g]]

            if f_g_dist > weights_upper_bound:  # disconnected
                continue

            f_x_g_dist = dists[F[f], :] + dists[F[g],
                                          :]  # currently only undirected. dists[:, F[g]] seems to be an issue.

            # dirty np.isclose rebuild
            rtol = 1e-05
            atol = 1e-08
            x_on_f_g_interval = np.abs(f_g_dist - f_x_g_dist) <= (atol + rtol * f_g_dist)
            x_on_f_g_interval[f_x_g_dist > weights_upper_bound] = False
            interval[x_on_f_g_interval] = True

    return interval


#@njit(parallel=True)
def numba_based_interval_reduced_dists(F, F_dists, weights_upper_bound=200000, early_break=False):
    '''
    makes sense for large F, a lot (>= 3*) faster than single loop python version
    currently only undirected!
    :param F:
    :param dists: an |F|*|V| array s.t. F_dists[f,v] = dists[f,v]
    :return:
    '''
    n = F_dists.shape[1]
    # naive double loop implementation:
    interval = np.zeros(n, dtype=np.bool_)

    # allow f==g for the case if |F|=1

    interval[F] = True
    if F.shape[0] == 1:
        return interval

    for f in range(F.shape[0] - 1):
        for g in range(f + 1, F.shape[0]):
            f_g_dist = F_dists[f, F[g]]

            if f_g_dist > weights_upper_bound:  # disconnected
                continue

            f_x_g_dist = F_dists[f, :] + F_dists[g,
                                         :]  # currently only undirected. dists[:, F[g]] seems to be an issue.

            # dirty np.isclose rebuild
            rtol = 1e-05
            atol = 1e-08
            x_on_f_g_interval = np.abs(f_g_dist - f_x_g_dist) <= (atol + rtol * f_g_dist)
            x_on_f_g_interval[f_x_g_dist > weights_upper_bound] = False
            x_on_f_g_interval[F_dists[f, :] > weights_upper_bound] = False
            x_on_f_g_interval[F_dists[g, :] > weights_upper_bound] = False
            interval[x_on_f_g_interval] = True
            if early_break and np.where(interval)[0].size > F.size:
                return  interval

    return interval

#@njit(parallel=True)
def numba_based_interval_reduced_dists_directed(F, F_dists_from_to, F_dists_to_from, weights_upper_bound=200000, early_break=False):
    '''
    makes sense for large F, a lot (>= 3*) faster than single loop python version
    :param F:
    :param dists: an |F|*|V| array s.t. F_dists_from_to[f,v] = dists[f,v]
    :param F_dists_to_from F_dists_to_from(f,v) = dists[v,f]
    :return:
    '''
    n = F_dists_from_to.shape[1]
    # naive double loop implementation:
    interval = np.zeros(n, dtype=np.bool_)

    # allow f==g for the case if |F|=1

    interval[F] = True
    if F.shape[0] == 1:
        return interval

    for f in range(F.shape[0]):
        for g in range(F.shape[0]):
            if f==g:
                continue

            f_g_dist = F_dists_from_to[f, F[g]]

            if f_g_dist > weights_upper_bound:  # disconnected
                continue

            f_x_g_dist = F_dists_from_to[f, :] + F_dists_to_from[g, :]

            # dirty np.isclose rebuild
            rtol = 1e-05
            atol = 1e-08
            x_on_f_g_interval = np.abs(f_g_dist - f_x_g_dist) <= (atol + rtol * f_g_dist)
            x_on_f_g_interval[f_x_g_dist > weights_upper_bound] = False
            x_on_f_g_interval[F_dists_from_to[f, :] > weights_upper_bound] = False
            x_on_f_g_interval[F_dists_to_from[g, :] > weights_upper_bound] = False
            interval[x_on_f_g_interval] = True
            if early_break and np.where(interval)[0].size > F.size:
                return  interval

    return interval


@njit(parallel=True)
def numba_based_extension(A, B, dists, weights_upper_bound=200000):
    ''' computes the ray/extension A/B
    currently only undirected!
    :param A:
    :param B:
    :param dists:
    :return:
    '''
    n = dists.shape[0]

    extension = np.zeros(n, dtype=np.bool_)

    extension[B] = True

    # not_A_B_idx = np.ones(n, dtype=np.bool)
    # not_A_B_idx[A] = False
    # not_A_B_idx[B] = False

    for a in prange(A.size):
        for b in range(B.size):
            if dists[A[a], B[b]] > weights_upper_bound:  # disconnected
                continue

            # dirty np.isclose rebuild
            rtol = 1e-05
            atol = 1e-08
            x_behind_b = np.abs(dists[A[a], B[b]] + dists[B[b], :] - dists[A[a], :]) <= (atol + rtol * dists[A[a], :])
            x_behind_b[dists[A[a], :] > weights_upper_bound] = False
            x_behind_b[dists[B[b], :] > weights_upper_bound] = False
            extension[x_behind_b] = True

    return extension


def dist_matrix_based_interval(F, dists, loop=True):
    # careful with two large Fs! Constructs an |V||F|^2 matrix

    # hopefully better single loop implementation:

    if loop:
        n = dists.shape[0]
        starting_vertices = np.zeros(n, dtype=np.bool_)
        starting_vertices[F] = True
        interval = np.zeros(n, dtype=np.bool_)
        for i, f in enumerate(F):
            interval[np.any(
                np.isclose(dists[f, :] + dists[:, starting_vertices].T, dists[f, starting_vertices][:, np.newaxis]),
                axis=0)] = True
            print(i)
        return interval

    # crazy no loop version (possibly needs a lot of memory: |V||F|^2
    pairs = dists[F, :, np.newaxis] + dists[:, F]
    pairs = np.transpose(pairs, axes=(0, 2, 1))  # 3d-array pairs[f,f',x] = dist(f,x) + dist(x,f')
    f_to_f_dists = dists[F][:, F]
    interval = np.any(np.isclose(pairs, f_to_f_dists[:, :, np.newaxis]),
                      axis=(0, 1))  # checks if any f,f' exists s.t. dist(f,x) + dist(x,f') = d(f,f')

    return interval


def dijkstra_based_interval(g: gt.Graph, F, weight_prop=None):
    raise Exception("this does not work, yet!")
    interval = np.zeros(g.num_vertices(), dtype=np.bool)

    n = g.num_vertices()
    '''edges = g.get_edges()
    super_start = g.add_vertex()
    shortest_search_reverse_dag = g.new_edge_property("bool", val=False)

    #weight_upperbound = np.sum(weight_prop.a)* 2 + 1

    for f in F:  # maybe only reached vertices??
        #need super_start --> f. will reverse it later.
        e = g.add_edge(f, super_start)
        shortest_search_reverse_dag[e] = True
        weight_prop[e] = np.inf

    original_vertices = g.new_vertex_property("bool", val=True)
    original_vertices[super_start] = False
    #g.reindex_edges()'''

    weights_in_needed_order = np.zeros(g.num_edges())

    for i, e in enumerate(g.edges()):
        weights_in_needed_order[i] = weight_prop[e]

    for f in F:
        if weight_prop is not None:
            dists_from_f, pred_map = dijkstra_search(g, source=f, weight=weight_prop)
        else:
            dists_from_f, pred_map = bfs_search(g, f)
        '''g.reindex_edges()
        '''

        # numpy magic
        edges_on_f_F_shortest_path = np.isclose(dists_from_f.a[g.get_edges()[:, 0]],
                                                dists_from_f.a[g.get_edges()[:, 1]] + weights_in_needed_order)
        # shortest_search_reverse_dag.a[edges_on_f_F_shortest_path] = True
        edges_on_f_F_shortest_path[np.isclose(dists_from_f.a[g.get_edges()[:, 1]],
                                              dists_from_f.a[g.get_edges()[:, 0]] + weights_in_needed_order)] = True
        # i = 0
        # for z in range(g.num_edges()):
        #    print(g.get_edges()[z], list(g.edges())[z], weight_prop.a[z], weight_prop[g.edge(g.get_edges()[z][0], g.get_edges()[z][1])])

        shortest_search_reverse_dag = g.new_edge_property("bool", vals=edges_on_f_F_shortest_path)

        g.set_edge_filter(shortest_search_reverse_dag)
        orientation = g.is_directed()
        g.set_directed(True)
        g.set_reversed(True)
        for f2 in F:
            if f == f2:
                continue
            interval[bfs_iterator(g, f2, array=True)[:, 1]] = True
        g.set_directed(orientation)
        g.set_reversed(False)
        g.clear_filters()

        # slow double loop
        '''for v in range(n): #maybe only reached vertices??
            for w in g.get_out_neighbors(v):
                if w == super_start:
                    continue
                if np.isclose(dists_from_f[w], dists_from_f[v] + weight_prop[g.edge(v, w)]):
                    shortest_search_reverse_dag[g.edge(v, w)] = True'''

        # do bfs from all vertices in F\f in the reversed f-shortest-path-dag

        '''g.set_edge_filter(shortest_search_reverse_dag)
        orientation = g.is_directed()
        g.set_directed(True)
        g.set_reversed(True)

        interval[bfs_iterator(g, super_start, array=True)[:, 1]] = True

        g.set_directed(orientation)
        g.set_reversed(False)
        g.clear_filters()'''

        print(f, np.sum(interval), n)

    for f in F:
        g.remove_edge(g.edge(f, super_start))

    g.remove_vertex(super_start)
    g.reindex_edges()
    return interval


if __name__ == '__main__':
    np.random.seed(33)
    random.seed(33)
    supported_datasets = ["ENGB", "DE", "email-Eu-core", "ES", "FR", "PTBR",
                          "RU"]  # , "lastfm_asia","git", "facebook", "pubmed", "citeseer", "cora", "cornell", "texas", "washington", "wisconsin"]
    community_datasets = ["dblp", "youtube"]

    for name in supported_datasets:
        if name in community_datasets:
            orientations = [False]
            weights = [False]
        else:
            orientations = [False]
            weights = [True]
        for directed in orientations:  # Be careful with directed on undirected datasets. This enforces some orentiation that wasn't originally there.
            for weighted in weights:
                print(name, "directed:", directed, "weighted:", weighted)
                # log_file_name = "../temp/" + name + "_directed_" + str(directed) + "_weighted_" + str(weighted) + ".txt"
                # with open(log_file_name, "w+") as log:
                #    log.write("Tests start")

                if name in supported_datasets:
                    g, labels, weight_prop, pairwise_dists = build_graph(name, directed=directed, weighted=weighted)
                    distances = gt.topology.shortest_distance(g, weights=weight_prop)
                    distance_matrix = distances.get_2d_array(range(g.num_vertices()))
                    for label in np.unique(labels):
                        current_class = np.where(labels == label)[0]
                        interval = numba_based_interval(current_class[:4], distance_matrix)

                        other_classes = np.where(labels != label)[0][:4]

                        extension = numba_based_extension(current_class[:4], other_classes, distance_matrix)
                        correct_labels = np.where(labels[extension] == label)[0].size
                        print(correct_labels, "/", np.sum(extension), current_class.size)

                        # dijkstra_based stuff ignore for now
                        # interval = dijkstra_based_interval(g, current_class[:4], weight_prop=weight_prop)
                        # correct_labels = np.where(labels[interval] == label)[0].size
                        # print(correct_labels, "/", np.sum(interval), current_class.size)
                elif name in community_datasets:
                    g, communities = build_graph(name)
