import json
import os
import pickle

import graph_tool as gt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from graph_tool.draw import graph_draw
from sklearn import datasets
from sklearn.metrics import pairwise_distances


def read_simple_txt(folder='email-Eu-core', name='email-Eu-core', path='../dataset', edges_file_ending='.txt',
                    labels_file_ending='-department-labels.txt'):
    labels = np.genfromtxt(path + '/' + folder + '/' + name + labels_file_ending)[:, -1]

    edges = np.genfromtxt(path + '/' + folder + '/' + name + edges_file_ending)

    features = None

    return edges, labels, features


def read_communities(folder='communities', name='dblp', path='../dataset'):
    """
    Each vertex can be part of multiple communities. Corresponds to multilabel setting
    No weights simply the edges.
    :param folder:
    :param name:
    :param path:
    :return:
    """
    edges = np.genfromtxt(path + '/' + folder + '/' + name + "/" + "com-" + name + ".ungraph.txt", comments="#",
                          dtype=np.int)

    communities = []
    with open(path + '/' + folder + '/' + name + "/" + "com-" + name + ".top5000.cmty.txt") as communities_file:
        for line in communities_file:
            communities.append(np.array(list(map(int, line.split("\t")))))

    return edges, communities

def read_kNN_iris(k=10):
    iris = datasets.load_iris()
    X = iris.data
    labels = np.zeros(X.shape[0], dtype=np.bool)
    labels[iris.target == 0] = True  # first class against the two others is linearly separable in Euclidean space.
    # check two verify linear separability
    # plt.scatter(X[labels, 0], X[labels, 1])
    # plt.scatter(X[~labels, 0], X[~labels, 1])
    # plt.show()
    all_pair_dists = pairwise_distances(X)

    edges = []
    for v in range(X.shape[0]):
        for w in np.argpartition(all_pair_dists[v], k + 1)[:k + 1]:  # k neighbours and itself
            if v < w:
                edges.append([v, w])

    return edges, labels, X


def read_eps_iris(eps=0.3):
    iris = datasets.load_iris()
    X = iris.data#[:,:2]
    labels = np.zeros(X.shape[0], dtype=np.bool)
    labels[iris.target == 0] = True  # first class against the two others is linearly separable in Euclidean space.
    # check two verify linear separability
    import matplotlib.pyplot as plt

    #plt.scatter(X[labels, 0], X[labels, 1])
    #plt.scatter(X[~labels, 0], X[~labels, 1])
    #plt.axis('off')
    #plt.tight_layout()
    #plt.savefig("iris.png", dpi=400, transparent=True, bbox_inches='tight')
    all_pair_dists = pairwise_distances(X)

    max_dist = eps * np.max(all_pair_dists)

    edges = []
    for v in range(X.shape[0]):
        v_dists = all_pair_dists[v]
        for w in np.where(v_dists <= max_dist)[0]:
            if v < w:
                edges.append([v, w])

    return edges, labels, X


def read_two_moons(eps=0.14):
    X, y = datasets.make_moons(noise=0.1, random_state=0)
    labels = np.zeros(X.shape[0], dtype=np.bool)
    labels[y == 0] = True  # first class against the two others is linearly separable in Euclidean space.
    # check two verify linear separability
    import matplotlib.pyplot as plt
    #plt.scatter(X[labels, 0], X[labels, 1])
    #plt.scatter(X[~labels, 0], X[~labels, 1])
    #plt.show()
    all_pair_dists = pairwise_distances(X)

    max_dist = eps * np.max(all_pair_dists)

    edges = []
    for v in range(X.shape[0]):
        v_dists = all_pair_dists[v]
        for w in np.where(v_dists <= max_dist)[0]:
            if v < w:
                edges.append([v, w])

    return edges, labels, X


def build_graph(name='citeseer', weighted=False, directed=False, largest_connected_component=False, ego_circles=False):
   if name in ["email-Eu-core"]:
        edges, vertex_labels, vertex_features = read_simple_txt()
    elif name in ["dblp", "friendster", "lj", "orkut", "youtube", "amazon"]:
        edges, communities = read_communities(name=name)
        directed = False
        vertex_labels = None
        vertex_features = None
        weighted = False
    elif name == "iris":
        edges, vertex_labels, vertex_features = read_eps_iris()
    elif name == "moons":
        edges, vertex_labels, vertex_features = read_two_moons()
  
    g = gt.Graph(directed=directed)

    pairwise_dists = None
    feature_dist_matrix = None
    if weighted and vertex_features is not None:

        if np.array_equal(vertex_features, vertex_features.astype(bool)):
            # do not need the squares here as {0,1}-valued
            pairwise_dists = np.sqrt(
                np.sum(np.abs(vertex_features[edges[:, 0]] - vertex_features[edges[:, 1]]), axis=1))
            # feature_dist_matrix = pairwise_distances(vertex_features, metric='cosine', n_jobs=-1)
        else:
            pairwise_dists = np.sqrt(
                np.sum((vertex_features[edges[:, 0]] - vertex_features[edges[:, 1]]) ** 2, axis=1))
            # zero weighted edges can make problems. add small epsilon.
            pairwise_dists[pairwise_dists == 0] = np.min(pairwise_dists[pairwise_dists > 0]) / 16
            # feature_dist_matrix = pairwise_distances(vertex_features, metric='cosine', n_jobs=-1)

        dists_prop = g.new_ep("double")
        g.add_edge_list(np.c_[edges, pairwise_dists], eprops=[dists_prop])
    else:
        g.add_edge_list(edges)

    if vertex_labels is not None:

        return g, vertex_labels, dists_prop, feature_dist_matrix
    else:
        return g, communities

