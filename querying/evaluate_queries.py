import numpy as np

from labelled_graph_primitives.cuts import get_cut_vertices


def cut_finding_performance(g, true_labels, queried_vertices):
    true_cut = get_cut_vertices(g, true_labels)

    cut_vertices = np.zeros(len(queried_vertices), dtype=np.int)

    for i, q in enumerate(queried_vertices):
        for v in q:
            if v in true_cut:
                cut_vertices[i] += 1

    return cut_vertices