import numpy as np

import graph_tool.topology
from graph_tool.draw import graph_draw
from sklearn import datasets


def risk(Y):
    '''
    ony for binary labels
    :param Y:
    :return:
    '''
    f = Y[:,0]
    return np.sum(np.min(np.column_stack((f, 1-f)), axis=1))

def active_label_prop(g, labels, dists):
    first_query = np.random.randint(0, g.num_vertices())
    known_labels = -np.ones(g.num_vertices())*np.inf
    known_labels[first_query] = labels[first_query]
    return find_best_query(dists, known_labels, labels)

def find_best_query(W, known_labels, labels, iterations=10, delta=0.00001, use_adjacency=False, budget=20):
    if not use_adjacency:
        W = np.exp(-W * W / 2)  # similarity
    true_labels = labels
    labels = np.unique(labels)
    Y = np.zeros((W.shape[0], labels.size))

    given_idx = np.where(known_labels > -np.inf)[0]

    # build one-hot label matrix
    for i, label in enumerate(labels):
        Y[known_labels == label, i] = 1

    D = np.sum(W, axis=1)

    eps = min(np.min(D), 0.000001)
    predictions = []
    D_inverse_dot_W = 1 / (D)[:, np.newaxis] * W
    for _ in range(budget):
        risks = np.ones(W.shape[0])*np.inf
        prev_Y = Y.copy()
        for i in range(W.shape[0]):
            if known_labels[i] > -np.inf:
                continue

            Y[i,0] = 1
            Y[i,1] = 0
            given_idx = np.append(given_idx, i)
            risk_0 = risk(label_propagation_one_step(Y.copy(), delta, iterations, D_inverse_dot_W, given_idx, labels))
            Y = prev_Y
            Y[i, 0] = 0
            Y[i, 1] = 1
            risk_1 = risk(label_propagation_one_step(Y.copy(), delta, iterations, D_inverse_dot_W, given_idx, labels))
            #clean up
            Y[i, 1] = 0
            given_idx = given_idx[:-1]
            #compute expected risk
            risks[i] = prev_Y[i,0]*risk_1 + (1 - prev_Y[i,0])*risk_0


        k = np.argmin(risks)
        print(risks[k])
        known_labels[k] =true_labels[k]

        #given_idx[k] = True
        #update/query
        given_idx = np.where(known_labels > -np.inf)[0]
        for i, label in enumerate(labels):
            Y[known_labels == label, i] = 1
        Y = label_propagation_one_step(Y.copy(), delta, iterations, D_inverse_dot_W, given_idx, labels)
        predictions.append(np.argmax(Y, axis=1))
    return predictions

def label_propagation_one_step(Y, delta, iterations, D_inverse_dot_W, given_idx, labels):
    oldY = np.ones((Y.shape[0], Y.shape[1]))
    i = 0
    while (np.abs(oldY - Y) > delta).any() and i <= iterations:
        oldY = Y
        Y = np.dot(D_inverse_dot_W, Y)
        Y[given_idx] = oldY[given_idx]
        i += 1

    # uniform argmax
    # for i in range(Y.shape[0]):
    #    result[i] = np.random.choice(np.flatnonzero(Y[i] == Y[i].max()))
    # maybe for the future
    # Y[not_given_idx] = np.dot(np.dot(np.linalg.inv(np.eye(len(not_given_idx)) - D_inverse_dot_W[not_given_idx][:,not_given_idx]), D_inverse_dot_W[not_given_idx][:,given_idx]), Y[given_idx])

    return Y

def active_label_prop_with_inverse(g, weight_prop, labels,  starting_vertex, budget=20):
    W = graph_tool.topology.shortest_distance(g, weights=weight_prop).get_2d_array(range(g.num_vertices()))  # original distance map

    #labels = (labels == labels[0])

    known_indices_mask = np.zeros(g.num_vertices(), dtype=bool)
    known_indices_mask[starting_vertex] = True
    #known_indices = [starting_vertex]
    #known_labels = [labels[starting_vertex]]

    current_predictions, inv_laplacian = label_propagation_with_inverse(W, labels, known_indices_mask)#, known_indices)

    accs = [np.average((current_predictions > .5)== labels)]
    queries = [[starting_vertex]]

    for z in range(min([budget - 1, g.num_vertices()-1])):
        risks = np.ones(g.num_vertices()) * np.inf
        if np.unique(labels[known_indices_mask]).size <= 1:
            k = np.random.choice(np.arange(g.num_vertices())[~known_indices_mask], 1)
        else:
            for k, v in enumerate(np.where(~known_indices_mask)[0]):
                #known_indices_mask[v] = True
                #known_indices.append(v)
                #assume positive

                #known_labels.append(True)
                y_v = 1
                prediction_pos = current_predictions[~known_indices_mask] + (y_v - current_predictions[v])*inv_laplacian[:,k]/inv_laplacian[k,k]
                #prediction_pos = label_propagation_with_inverse(W, labels, known_indices_mask, known_indices)


                #assume negative
                #known_labels[-1] = False
                #prediction_neg = label_propagation_with_inverse(W, labels, known_indices_mask, known_indices)

                y_v = 1
                prediction_neg = current_predictions[~known_indices_mask] + (y_v - current_predictions[v]) * inv_laplacian[:, k] / inv_laplacian[k,k]


                risks[v] = current_predictions[v]*np.sum(np.min(np.column_stack((prediction_pos, 1 - prediction_pos)), axis=1)) + \
                           (1-current_predictions[v])*np.sum(np.min(np.column_stack((prediction_neg, 1 - prediction_neg)), axis=1))


                #reset
                #known_indices_mask[v] = False
                #known_indices = known_indices[:-1]
                #known_labels = known_labels[:-1]

            risks[risks < 0.000001] = 0

            k = np.argmin(risks)

        known_indices_mask[k] = True
        #known_indices.append(k)


        #if labels[k] > 0:
            #known_labels.append(True)
        #    current_predictions, inv_laplacian = label_propagation_with_inverse(W, labels, known_indices_mask)#, known_indices)
        #else:
            #known_labels.append(False)
        current_predictions, inv_laplacian = label_propagation_with_inverse(W, labels, known_indices_mask)#, known_indices)

        # iris = datasets.load_iris()
        #X, _ = datasets.make_moons(noise=0.1, random_state=0)
        #X[:,1] *= -1
        #positions_prop = g.new_vertex_property("vector<float>", vals=X[:, :2])
        # vertex_colors = np.zero(g.num_vertices(), 4)
        # vertex_colors[[34, 92, 68, 40, 26, 82]] = []
        # colored_edges =[[34,92],[92,68],[68,26],[26,40],[40,82]]
        # colors[vertex_labels] = 2
        vertex_colors = np.zeros((g.num_vertices(), 4))
        vertex_colors[current_predictions>.5] = [1,0,0,.6]
        vertex_colors[current_predictions<=.5] = [0,0,1,.6]
        vertex_colors[known_indices_mask] += [0,1,0,.2]

        #vertex_colors[known_indices_mask ^(current_predictions>.5)] = [.8,.6,.6,.5]
        vertex_colors_prop = g.new_vertex_property("vector<float>", vals=vertex_colors)
        # for e in colored_edges:
        #    edge_colors_prop[g.edge(e[0],e[1])] = "red"

        # later this is not important now.
        #graph_draw(g, pos=positions_prop)
        #graph_draw(g, pos=positions_prop,vertex_fill_color=vertex_colors_prop)#,output="../temp/graph.png")



        queries.append(np.where(known_indices_mask)[0])
        accs.append(np.average((current_predictions > .5) == labels))
        print(accs)

    return queries,accs


def label_propagation_with_inverse(W, labels, known_indices_mask):
    '''

    :param W:
    :param labels:
    :param l: number of known labelled points, the first l points
    :return:
    '''
    sigma = 1
    W = np.exp(-W * W / (2*sigma*sigma))  # similarity
    #W = 1- np.sign(adjacency_matrix) # simple graph similarity

    d = np.sum(W, axis=1)
    D = np.diag(d)
    #P = np.dot(np.diag(1/d),W) # P = D^-1*W

    predictions = np.empty(W.shape[0])
    inv_laplacian = np.linalg.inv(D[~known_indices_mask][:, ~known_indices_mask] - W[~known_indices_mask][:, ~known_indices_mask])
    predictions[~known_indices_mask] = np.dot(np.dot(inv_laplacian, W[~known_indices_mask][:, known_indices_mask]), labels[known_indices_mask])
    predictions[known_indices_mask] = labels[known_indices_mask]

    return predictions, inv_laplacian

def label_propagation(W, known_labels, labels, iterations=10, delta=0.00001, use_adjacency=False):
    if not use_adjacency:
        W = np.exp(-W * W / 2)  # similarity
    labels = np.unique(labels)
    Y = np.zeros((W.shape[0], labels.size))

    given_idx = np.where(known_labels > -np.inf)[0]

    # build one-hot label matrix
    for i, label in enumerate(labels):
        Y[known_labels == label, i] = 1

    D = np.sum(W, axis=1)

    eps = min(np.min(D), 0.000001)

    D_inverse_dot_W = 1 / (D)[:, np.newaxis] * W

    oldY = np.ones((Y.shape[0], Y.shape[1]))
    i = 0
    while (np.abs(oldY - Y) > delta).any() and i <= iterations:
        oldY = Y
        Y = np.dot(D_inverse_dot_W, Y)
        Y[given_idx] = oldY[given_idx]
        i += 1

    # uniform argmax
    # for i in range(Y.shape[0]):
    #    result[i] = np.random.choice(np.flatnonzero(Y[i] == Y[i].max()))
    # maybe for the future
    # Y[not_given_idx] = np.dot(np.dot(np.linalg.inv(np.eye(len(not_given_idx)) - D_inverse_dot_W[not_given_idx][:,not_given_idx]), D_inverse_dot_W[not_given_idx][:,given_idx]), Y[given_idx])

    result = labels[np.argmax(Y, axis=1)]

    return result
