import random

import graph_tool as gt
import networkx as nx
import numpy as np
from graph_tool.search import bfs_iterator
from graph_tool.topology import random_shortest_path

from convexity_operations.convex_hulls import numba_based_interval_reduced_dists, \
    numba_based_interval_reduced_dists_directed
from read_graphs.read import build_graph

from joblib import Parallel, delayed

def sample_path_check(g: gt.Graph, labels, weight_prob: gt.EdgePropertyMap = None, number_paths=10, multi_label=False,
                      log_file_name=None):
    """
    This function performs some simple convexity checks, based on sampling shortest path with endpoints in one class.
    In particular, from each class number_paths many shortest paths are randomly sampled by first selecting uniformly at random two connected vertices,
    and then selecting a uniformly at random shortest path between them.

    The checks are primarily based on checking whether the path leaves the class it started from. Then we say (somewhat ambiguously) that it is "convex".

    :param g: directed/undirected graph-tool Graph
    :param labels:
    :param weight_prob: possible edge weights
    :param number_paths: number of shortest paths to sample from each class
    :param multi_label: if true labels is not a vector, but a list of lists, each of which represents one (possibly overlapping) community
    :return: 5 arrays containing a number for each class.
        - approximate_number_convex_paths: the relative number of "convex" paths
        - approximate_number_convex_paths_not_edges: relative number of "convex" paths that are not edges (edges are trivially convex if the weights fulfill the triangle ineq.)
        - number_edges: absolute number of sampled edges. To get the absolute number of convex paths that are not edges
        - approximate_number_correct_inner_vertices: relative number of inner vertices that have the endpoints' labels. Corresponds to the prediction accuracy on the inner vertices. Single vertices might be counted multiple times, as we do not check if a vertex was "predicted" before.
        - number_inner_vertices: Absolute number of inner vertices. To get the number of correctly predicted inner vertices
    """
    if not multi_label:
        unique_labels = np.unique(labels)
        num_classes = unique_labels.size
    else:
        num_classes = len(labels)

    approximate_number_convex_paths = np.zeros(num_classes)
    approximate_number_convex_paths_not_edges = np.zeros(num_classes)

    approximate_number_correct_inner_vertices = np.zeros(num_classes)

    number_edges = np.zeros(num_classes, dtype=np.int)
    number_inner_vertices = np.zeros(num_classes, dtype=np.int)

    # labels_path_length_matrix = np.zeros((num_classes, 5)) # have to set 30 to the diameter of the graph or use a dict instead

    if multi_label:
        # in the case of community setting traverse each of the communities instead of the unique class labels
        unique_labels = labels

    # precompute connected components (ignore orientation)

    connected_components = gt.topology.label_components(g, directed=False)[0].a

    for label_idx, label in enumerate(unique_labels):
        if not multi_label:
            labelled_class = np.where(labels == label)[0]

        else:
            labelled_class = label
            labels = np.zeros(g.num_vertices(), dtype=np.int)
            labels[labelled_class] = 1
            label = 1

            own_component = np.all(connected_components[labelled_class] == connected_components[labelled_class[0]]) and \
                            connected_components[connected_components == connected_components[
                                labelled_class[0]]].size == labelled_class.size

            with open(log_file_name, "a") as log:
                log.write("\n")
                log.write(
                    "Community: " + str(label_idx) + " size: " + str(labelled_class.size) + " own component: " + str(
                        own_component))
            print("Community:", label_idx, "size:", labelled_class.size, "own component:", own_component)

        number_convex_paths = 0
        number_edges[label_idx] = 0

        number_inner_vertices[label_idx] = 0
        number_correct_label_inner_vertices = 0

        if labelled_class.size <= 1:

            with open(log_file_name, "a") as log:
                log.write("\n")
                if not multi_label:
                    log.write("--> Only one vertex with class label=" + str(label) + ", no checks")
                    print("--> Only one vertex with class label=" + str(label) + ", no checks")
                else:
                    log.write("--> Only one vertex in community " + str(label_idx) + ", no checks")
                    print("--> Only one vertex in community " + str(label_idx) + ", no checks")
            continue
        broke = False
        for _ in range(number_paths):
            x = labelled_class[random.randrange(labelled_class.size)]

            if g.is_directed():
                reachable_from_x = np.unique(bfs_iterator(g, x, array=True))
                reachable_from_x = reachable_from_x[labels[reachable_from_x] == labels[x]]
            else:
                # reachable and same class/community
                reachable_from_x = labelled_class[
                    np.where(connected_components[labelled_class] == connected_components[x])[0]]

            # at least one vertex additional with same label has to be reachable (x itself is also reachable)

            candidate_x = set(labelled_class)

            while reachable_from_x.size < 2 and candidate_x:
                x = random.sample(candidate_x, 1)[0]
                candidate_x.remove(x)
                if g.is_directed():
                    reachable_from_x = np.unique(bfs_iterator(g, x, array=True))
                    reachable_from_x = reachable_from_x[labels[reachable_from_x] == labels[x]]
                else:
                    # reachable and same class/community
                    reachable_from_x = labelled_class[
                        np.where(connected_components[labelled_class] == connected_components[x])[0]]

            if np.sum(labels[reachable_from_x] == label) < 2 and not candidate_x:
                with open(log_file_name, "a") as log:
                    log.write("\n")
                    log.write("--> All vertices with class label=" + str(label) + " are disconnected, no checks")
                print("--> All vertices with class label=" + str(label) + " are disconnected, no checks")
                broke = True
                break

            reachable_from_x = np.delete(reachable_from_x, np.where(reachable_from_x == x))

            y = reachable_from_x[random.randrange(reachable_from_x.size)]

            sampled_shortest_path = random_shortest_path(g, x, y, weights=weight_prob)
            sampled_shortest_path = np.array(sampled_shortest_path, dtype=np.int)

            paths_labels = labels[sampled_shortest_path]

            # print(sampled_shortest_path.size)

            if np.all(paths_labels == label):
                number_convex_paths += 1
            if paths_labels.size == 2:
                number_edges[label_idx] += 1



            number_inner_vertices[label_idx] += paths_labels.size - 2

            if paths_labels.size - 2 > 0:
                #relative number of correct inner vertices on this path.
                number_correct_label_inner_vertices += (np.sum(paths_labels == label) - 2)/(paths_labels.size - 2)

            if broke:
                continue

        approximate_number_convex_paths[label_idx] = number_convex_paths / number_paths
        #print(approximate_number_convex_paths[label_idx])
        if number_paths > number_edges[label_idx]:
            approximate_number_convex_paths_not_edges[label_idx] = (number_convex_paths - number_edges[label_idx]) / (
                    number_paths - number_edges[label_idx])
        if number_inner_vertices[label_idx] > 0:  # corresponds to number_edges[label_idx] > 0
            approximate_number_correct_inner_vertices[label_idx] = number_correct_label_inner_vertices / \
                                                                   (number_paths - number_edges[label_idx])
        else:
            approximate_number_correct_inner_vertices[label_idx] = -np.inf
    np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
    # np.savetxt()
    print(approximate_number_convex_paths)
    print(approximate_number_convex_paths_not_edges)
    print(number_edges)
    print(approximate_number_correct_inner_vertices)
    print(number_inner_vertices)

    if log_file_name is not None:
        with open(log_file_name, "a") as log:
            log.write("\n")
            log.write("approximate_number_convex_paths")
            log.write("\n")
            np.savetxt(log, approximate_number_convex_paths, fmt="%.2f")
            log.write("\n")
            log.write("approximate_number_convex_paths_not_edges")
            log.write("\n")
            np.savetxt(log, approximate_number_convex_paths_not_edges, fmt="%.2f")
            log.write("\n")
            log.write("number_edges")
            log.write("\n")
            np.savetxt(log, number_edges, "%.2f")
            log.write("\n")
            log.write("approximate_number_correct_inner_vertices")
            log.write("\n")
            np.savetxt(log, approximate_number_correct_inner_vertices, fmt="%.2f")
            log.write("\n")
            log.write("number_inner_vertices")
            log.write("\n")
            np.savetxt(log, number_inner_vertices, fmt="%.2f")

    return approximate_number_convex_paths, approximate_number_convex_paths_not_edges, number_edges, approximate_number_correct_inner_vertices, number_inner_vertices


def compute_diameter(g: gt.Graph, distance_matrix=None, weights=None):
    if g.num_vertices() > 33000:
        raise RuntimeError("Be careful! Such large graphs might make the system freeze.")

    if distance_matrix is None:
        distances = gt.topology.shortest_distance(g, weights=weights)
        distance_matrix = distances.get_2d_array(range(g.num_vertices()))

    if weights is not None:
        diameter = np.max(distance_matrix[distance_matrix < np.sum(
            weights.a) + 1]) 
    else:
        diameter = np.max(distance_matrix[distance_matrix < g.num_vertices()])

    pseudo_diameter = gt.topology.pseudo_diameter(g, weights=weights)[0]
    print("diam:", diameter, "pseudo-diam:", pseudo_diameter)


def compute_cut_and_border(g: gt.Graph, labels, weights=None):
    unique_labels = np.unique(labels)
    num_classes = unique_labels.size

    edges_between_classes = np.empty((num_classes, num_classes))
    border_vertices = []

    for v in range(g.num_vertices()):
        for w in g.get_out_neighbors(v):
            v_label_idx = np.where(unique_labels == labels[v])[0][0]
            w_label_idx = np.where(unique_labels == labels[w])[0][0]

            if weights is None:
                edges_between_classes[v_label_idx, w_label_idx] += 1
            else:
                edges_between_classes[v_label_idx, w_label_idx] += weights[g.edge(v, w)]
        if np.any(labels[g.get_out_neighbors(v)] != labels[v]):
            border_vertices.append(v)
    print(edges_between_classes[0, 1], len(border_vertices), g.num_vertices())
    return edges_between_classes, border_vertices


def compute_balancednes(labels):
    sizes = np.unique(labels, return_counts=True)[1]
    return sizes, np.min(sizes) / labels.size


def check_percolation(g: gt.Graph, labels):
    networkx_graph = nx.from_edgelist(np.array(g.edges))

    perc_states = dict()
    for v in networkx_graph:
        if labels[v] == labels[0]:  # simple check with the class of the first vertex
            perc_states[v] = 1
        else:
            perc_states[v] = 0

    return nx.percolation_centrality(networkx_graph, states=perc_states)


def check_convexity(g: gt.Graph, labels, weight_prop=None, check_only_one_label=False, log_file_name=None):
    unique_labels = np.unique(labels)

    if check_only_one_label:
        unique_labels = [True]

    for label in unique_labels:

        # compute all dists to the class
        labelled_class = np.where(labels == label)[0]
        print("n =", g.num_vertices(), "class size: ", labelled_class.size)
        #dists = []
        #for v in labelled_class:
        #    v_dists = gt.topology.shortest_distance(g, v).a
        #    dists.append(v_dists)

        if not g.is_directed():
            dists = Parallel(n_jobs=-1, backend="threading")(delayed(gt.topology.shortest_distance)(g=g, source = v) for v in labelled_class)
            dists = np.array([v_dist.a for v_dist in dists])
            interval = numba_based_interval_reduced_dists(labelled_class, dists, early_break=check_only_one_label)
        else:
            F_dists_from_to = Parallel(n_jobs=-1, backend="threading")(delayed(gt.topology.shortest_distance)(g=g, source = v) for v in labelled_class)
            F_dists_from_to = np.array([v_dist.a for v_dist in F_dists_from_to])

            g.set_reversed(True)
            F_dists_to_from = Parallel(n_jobs=-1, backend="threading")(delayed(gt.topology.shortest_distance)(g=g, source=v) for v in labelled_class)
            F_dists_to_from = np.array([v_dist.a for v_dist in F_dists_to_from])
            g.set_reversed(False)

            interval = numba_based_interval_reduced_dists_directed(labelled_class, F_dists_from_to, F_dists_to_from, early_break=check_only_one_label)

        if log_file_name is not None:
            with open(log_file_name, "a") as log:
                if np.all((labels == label) == interval):
                    if check_only_one_label:
                        label = np.where(labels)[0]
                    log.write("label", label)
                    log.write("is convex! Size:", labelled_class.size)
                else:
                    if check_only_one_label:
                        label = np.where(labels)[0]
                    log.write("label", label)
                    log.write(" is not convex! Size:", labelled_class.size, "Difference:",
                              np.sum(interval) - np.sum(labels == label))

        if np.all((labels == label) == interval):
            if check_only_one_label:
                label = np.where(labels)[0]
            print("label", label)
            print("is convex! Size:", labelled_class.size)
        else:
            if check_only_one_label:
                label = np.where(labels)[0]
            print("label", label)
            print(" is not convex! Size:", labelled_class.size, "Difference:",
                  np.sum(interval) - np.sum(labels == label))


if __name__ == '__main__':
    np.random.seed(33)
    random.seed(33)
    community_datasets = ["email-Eu-core"] #,"dblp"]
    separable_datasets = [ "moons","iris"]

    for name in ["dblp"]:#community_datasets:
    	orientations = [True]
	weights = [False]
        for directed in orientations:  # Be careful with directed on undirected datasets. This enforces some orentiation that wasn't originally there.
            for weighted in weights:
                print(name, "directed:", directed, "weighted:", weighted)
                log_file_name = "../temp/" + name + "_directed_" + str(directed) + "_weighted_" + str(weighted) + ".txt"
                with open(log_file_name, "w+") as log:
                    log.write("Tests start")

                if name in ["iris", "moons"]:
                    g, labels, weight_prop, pairwise_dists = build_graph(name, directed=directed, weighted=weighted)
                    check_convexity(g, labels, weight_prop)
                    #sample_path_check(g, labels, weight_prop, log_file_name=log_file_name)
                    #print(compute_balancednes(labels), 1 / np.unique(labels).size)
                    #compute_cut_and_border(g, labels)
                    #compute_diameter(g, weights=weight_prop)
                    check_convexity(g, labels)
                elif name in community_datasets:
                    g, communities = build_graph(name)

                    for community in communities:
                        community_idx = np.zeros(g.num_vertices(), dtype=np.bool)
                        community_idx[community] = True

                        check_convexity(g, community_idx, check_only_one_label=True)
                    #sample_path_check(g, communities, None, log_file_name=log_file_name, multi_label=True)
