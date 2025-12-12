# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import random
import math
import scanpy as sc
import sklearn
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# Mapping Accuracy
def pairwise_exp(adata1, adata2, pi):
    """
    Calculates mapping accuracy between two slices 
    Args:
        adata1: AnnData,
        adata2: AnnData,
        pi: a probabilistic mapping between spots in the two slices
    Returns:
        - mapping accuracy
    """
    spotsA, spotsB = np.nonzero(pi)
    s = 0
    for i in range(len(spotsA)):
        # get the clusters corresponding to each spot
        a = adata1.obs['annotation'][spotsA[i]]
        b = adata2.obs['annotation'][spotsB[i]]
        if a == b:
            s += pi[spotsA[i]][spotsB[i]]
    return s


# LTARI
def compute_alignment_ari(sliceA, sliceB, pi):
    """
    Calculates Label Transfer Adjusted Rand Index between two slices 
    Args:
        adata1: AnnData,
        adata2: AnnData,
        pi: a probabilistic mapping between spots in the two slices
    Returns:
        - Label Transfer Adjusted Rand Index
    """
    mapped_clusters = []
    for j in range(pi.shape[1]):
        mapping = pi[:, j]
        if np.sum(mapping) > 0:
            i = np.argmax(mapping)
            mapped_clusters.append(sliceA.obs['annotation'][i])
        else:
            mapped_clusters.append("NULL")
    assert len(sliceB.obs['annotation']) == len(mapped_clusters)
    true_clusters_mapped_region = []
    mapped_clusters_mapped_region = []
    for i in range(len(sliceB.obs['annotation'])):
        if mapped_clusters[i] != "NULL":
            true_clusters_mapped_region.append(sliceB.obs['annotation'][i])
            mapped_clusters_mapped_region.append(mapped_clusters[i])
    ari = sklearn.metrics.adjusted_rand_score(true_clusters_mapped_region, mapped_clusters_mapped_region)
    return ari


# Spatial coherence score
def create_graph(adata, degree = 4):
        """
        Converts spatial coordinates into graph using networkx library.
        
        param: adata - ST Slice 
        param: degree - number of edges per vertex

        return: 1) G - networkx graph
                2) node_dict - dictionary mapping nodes to spots
        """
        D = cdist(adata.obsm['spatial'], adata.obsm['spatial'])
        # Get column indexes of the degree+1 lowest values per row
        idx = np.argsort(D, 1)[:, 0:degree+1]
        # Remove first column since it results in self loops
        idx = idx[:, 1:]

        G = nx.Graph()
        for r in range(len(idx)):
            for c in idx[r]:
                G.add_edge(r, c)

        node_dict = dict(zip(range(adata.shape[0]), adata.obs.index))
        return G, node_dict

def generate_graph_from_labels(adata, labels_dict):
    """
    Creates and returns the graph and dictionary {node: cluster_label} for specified layer
    """
    
    g, node_to_spot = create_graph(adata)
    spot_to_cluster = labels_dict

    # remove any nodes that are not mapped to a cluster
    removed_nodes = []
    for node in node_to_spot.keys():
        if (node_to_spot[node] not in spot_to_cluster.keys()):
            removed_nodes.append(node)

    for node in removed_nodes:
        del node_to_spot[node]
        g.remove_node(node)
        
    labels = dict(zip(g.nodes(), [spot_to_cluster[node_to_spot[node]] for node in g.nodes()]))
    return g, labels

def spatial_coherence_score(graph, labels):
    g, l = graph, labels
    true_entropy = spatial_entropy(g, l)
    entropies = []
    for i in range(1000):
        new_l = list(l.values())
        random.shuffle(new_l)
        labels = dict(zip(l.keys(), new_l))
        entropies.append(spatial_entropy(g, labels))
        
    return (true_entropy - np.mean(entropies))/np.std(entropies)

def spatial_entropy(g, labels):
    """
    Calculates spatial entropy of graph  
    """
    # construct contiguity matrix C which counts pairs of cluster edges
    cluster_names = np.unique(list(labels.values()))
    C = pd.DataFrame(0,index=cluster_names, columns=cluster_names)

    for e in g.edges():
        C[labels[e[0]]][labels[e[1]]] += 1

    # calculate entropy from C
    C_sum = C.values.sum()
    H = 0
    for i in range(len(cluster_names)):
        for j in range(i, len(cluster_names)):
            if (i == j):
                z = C[cluster_names[i]][cluster_names[j]]
            else:
                z = C[cluster_names[i]][cluster_names[j]] + C[cluster_names[j]][cluster_names[i]]
            if z != 0:
                H += -(z/C_sum)*math.log(z/C_sum)
    return H