# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")

import STAligner

import os
R_dirs="/home/zhaoshangrui/anaconda2/envs/R4.3.1/lib/R"
os.environ['R_HOME']=R_dirs
import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np
import scipy.linalg
from scipy.sparse import csr_matrix

import torch
used_device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

path_to_output_dir = "/home/zhaoshangrui/xuxinyu/datasets/SCC/cached-results/"       
path_to_h5ads = path_to_output_dir + 'H5ADs/'

patient_2 = []
patient_5 = []
patient_9 = []
patient_10 = []

patients = {
    "patient_2" : patient_2,
    "patient_5" : patient_5,
    "patient_9" : patient_9,
    "patient_10" : patient_10,
}

for k in patients.keys():
    for i in range(3):
        patients[k].append(sc.read_h5ad(path_to_h5ads + k + '_slice_' + str(i) + '.h5ad'))
 
all_slices = [patients["patient_2"], patients["patient_5"], patients["patient_9"], patients["patient_10"]]
patient_ids = [2, 5, 9, 10] 

#patient_2
Batch_list = []
adj_list = []
for i in range(3):
    data = patients["patient_2"][i]  
    data.X=csr_matrix(data.X)
    data.obs_names = [f"{x}_{patient_2}_{i}" for x in data.obs_names]
    # Constructing the spatial network
    STAligner.Cal_Spatial_Net(data, rad_cutoff=2.5) # the spatial network are saved in adata.uns[‘adj’]
    # STAligner.Stats_Spatial_Net(adata) # plot the number of spatial neighbors
    adj_list.append(data.uns['adj'])
    Batch_list.append(data)
        
adata_concat = ad.concat(Batch_list, label="batch")
adata_concat.obs["batch_name"], unique_batches = pd.factorize(adata_concat.obs["batch"])
adata_concat.obs["batch_name"] = adata_concat.obs["batch_name"].astype(str)
print('adata_concat.shape: ', adata_concat.shape)

adj_concat = np.asarray(adj_list[0].todense())
for batch_id in range(1,3):
    adj_concat = scipy.linalg.block_diag(adj_concat, np.asarray(adj_list[batch_id].todense()))
adata_concat.uns['edgeList'] = np.nonzero(adj_concat)

adata_concat = STAligner.train_STAligner(adata_concat, verbose=True, knn_neigh = 100, device=used_device) #epochs = 1500,
adata_concat.uns['edgeList'] = list(adata_concat.uns['edgeList'])
adata_concat.write("/home/zhaoshangrui/xuxinyu/STAligner/SCC/SCC_patient_2.h5ad")

staligner_df = pd.DataFrame(adata_concat.obsm['STAligner'])
staligner_df.index = adata_concat.obs_names
staligner_df.to_csv('/home/zhaoshangrui/xuxinyu/STAligner/SCC/SCC_STAligner_embedding_patient_2.csv', index=True)

# Clustering
adata_concat = sc.read_h5ad("/home/zhaoshangrui/xuxinyu/STAligner/SCC/SCC_patient_2.h5ad")

def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='STAGATE', random_seed=666):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")
    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']
    res = rmclust(adata.obsm[used_obsm], num_cluster, modelNames)
    mclust_res = np.array(res[-2])
    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata

mclust_R(adata_concat, num_cluster=12, used_obsm='STAligner')
adata_concat.write("/home/zhaoshangrui/xuxinyu/STAligner/SCC/SCC_cluster_patient_2.h5ad")

adata_concat = sc.read_h5ad("/home/zhaoshangrui/xuxinyu/STAligner/SCC/SCC_cluster_patient_2.h5ad")
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
print('mclust, ARI = %01.3f' % adjusted_rand_score(adata_concat.obs['original_clusters'], adata_concat.obs['mclust']))
# # mclust, ARI = 0.278
print('mclust, NMI = %01.3f' % normalized_mutual_info_score(adata_concat.obs['original_clusters'], adata_concat.obs['mclust']))
#mclust, NMI = 0.417

sc.pp.neighbors(adata_concat, use_rep='STAligner', random_state=666)
sc.tl.umap(adata_concat, random_state=666)

import networkx as nx
def match_cluster_labels(true_labels,est_labels):
    true_labels_arr = np.array(list(true_labels))
    est_labels_arr = np.array(list(est_labels))
    org_cat = list(np.sort(list(pd.unique(true_labels))))
    est_cat = list(np.sort(list(pd.unique(est_labels))))
    B = nx.Graph()
    B.add_nodes_from([i+1 for i in range(len(org_cat))], bipartite=0)
    B.add_nodes_from([-j-1 for j in range(len(est_cat))], bipartite=1)
    for i in range(len(org_cat)):
        for j in range(len(est_cat)):
            weight = np.sum((true_labels_arr==org_cat[i])* (est_labels_arr==est_cat[j]))
            B.add_edge(i+1,-j-1, weight=-weight)
    match = nx.algorithms.bipartite.matching.minimum_weight_full_matching(B)
#     match = minimum_weight_full_matching(B)
    if len(org_cat)>=len(est_cat):
        return np.array([match[-est_cat.index(c)-1]-1 for c in est_labels_arr])
    else:
        unmatched = [c for c in est_cat if not (-est_cat.index(c)-1) in match.keys()]
        l = []
        for c in est_labels_arr:
            if (-est_cat.index(c)-1) in match: 
                l.append(match[-est_cat.index(c)-1]-1)
            else:
                l.append(len(org_cat)+unmatched.index(c))
        return np.array(l) 

adata_concat.obs['mclust'] = pd.Series(match_cluster_labels(adata_concat.obs['original_clusters'], adata_concat.obs['mclust'].values),
                                          index=adata_concat.obs.index, dtype='category')

adata_concat.write("/home/zhaoshangrui/xuxinyu/STAligner/SCC/SCC_cluster_patient_2.h5ad")

adata_concat = sc.read_h5ad("/home/zhaoshangrui/xuxinyu/STAligner/SCC/SCC_cluster_patient_2.h5ad")

import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(20, 3))
sc.pl.umap(adata_concat, color='batch_name', ax=axes[0], show=False, legend_fontsize=10, palette='coolwarm')
axes[0].set_title('Slice')
axes[0].set_xlabel('')
axes[0].set_ylabel('')

sc.pl.umap(adata_concat, color='original_clusters', ax=axes[1], show=False, legend_fontsize=10, palette='cividis')
axes[1].set_title('Layer annotation')
axes[1].set_xlabel('')
axes[1].set_ylabel('')

new_categories = [f'Cluster{i}' for i in range(1, 8)]
adata_concat.obs['mclust'] = adata_concat.obs['mclust'].cat.rename_categories(new_categories)
mclust_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
sc.pl.umap(adata_concat, color='mclust', ax=axes[2], show=False, legend_fontsize=10, palette=mclust_colors)
axes[2].set_title('Cluster')
axes[2].set_xlabel('')
axes[2].set_ylabel('')

plt.subplots_adjust(wspace=1)

plt.show()

