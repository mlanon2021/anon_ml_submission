import graph_tool as gt


def get_cut_vertices(g: gt.Graph, labels):
    # assume binary for now:

    cut_vertices = set()
    for v in g.vertices():
        for e in g.get_out_edges(v):
            if labels[e[0]] != labels[e[1]]:
                cut_vertices.add(v)

    return cut_vertices
