# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")
import STAligner

import os
import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np
import scipy.linalg

import torch
used_device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

###Load Data
Batch_list = []
adj_list = []
section_ids = ['151507','151508','151509','151510']
for section_id in section_ids:
    print(section_id)
    input_dir = os.path.join("/home/zhaoshangrui/xuxinyu/datasets/DLPFC_data/10xGenomicsVisium/spatialLIBD/", section_id)
    adata = sc.read_visium(path=input_dir, count_file='filtered_feature_bc_matrix.h5', load_images=True)
    adata.var_names_make_unique(join="++")    
    # read the annotation
    ground_truth_dir = os.path.join("/home/zhaoshangrui/xuxinyu/datasets/DLPFC_data/10xGenomicsVisium/spatialLIBD/", section_id)
    Ann_df = pd.read_csv(os.path.join(ground_truth_dir, section_id + '_truth.txt'), sep='\t', header=None, index_col=0)
    Ann_df.columns = ['Ground Truth']
    Ann_df[Ann_df.isna()] = "unknown"
    adata.obs['Ground Truth'] = Ann_df.loc[adata.obs_names, 'Ground Truth'].astype('category')
    # make spot name unique
    adata.obs_names = [x + '_' + section_id for x in adata.obs_names]
    STAligner.Cal_Spatial_Net(adata, rad_cutoff=150)
    ## Normalization
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=10000) #ensure enough common HVGs in the combined matrix
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata = adata[:, adata.var['highly_variable']]
    adj_list.append(adata.uns['adj'])
    Batch_list.append(adata)
  
###Concat the scanpy objects for multiple slices    
adata_concat = ad.concat(Batch_list, label="slice_name", keys=section_ids)
adata_concat.obs['Ground Truth'] = adata_concat.obs['Ground Truth'].astype('category')
adata_concat.obs["batch_name"] = adata_concat.obs["slice_name"].astype('category')
print('adata_concat.shape: ', adata_concat.shape)

###Concat the spatial network for multiple slices
adj_concat = np.asarray(adj_list[0].todense())
for batch_id in range(1,len(section_ids)):
    adj_concat = scipy.linalg.block_diag(adj_concat, np.asarray(adj_list[batch_id].todense()))
adata_concat.uns['edgeList'] = np.nonzero(adj_concat)

###Running STAligner
adata_concat = STAligner.train_STAligner(adata_concat, iter_comb = None, verbose=True, device=used_device, margin=1.0)
adata_concat.uns['edgeList'] = list(adata_concat.uns['edgeList'])
adata_concat.write("/home/zhaoshangrui/xuxinyu/STAligner/DLPFC/STAligner_sample1.h5ad")

adata_concat.obs.index = adata_concat.obs.index.map(lambda x: f"{x.split('_')[1]}-{x.split('_')[0]}")
st_aligner_df = pd.DataFrame(adata_concat.obsm['STAligner'], index=adata_concat.obs.index)
st_aligner_df.to_csv('/home/zhaoshangrui/xuxinyu/STAligner/DLPFC/STAligner_embed_sample1.csv')

# Alignment_ICP
iter_comb = [(1, 0), (2, 1), (3, 2)]
for comb in iter_comb:
    print(comb)
    i, j = comb[0], comb[1]
    adata_target = Batch_list[i]
    adata_ref = Batch_list[j]
    slice_target = section_ids[i]
    slice_ref = section_ids[j]
    aligned_coor = STAligner.ICP_align(adata, adata_target, adata_ref, slice_target, slice_ref)
    adata_target.obsm["spatial"] = aligned_coor
    
adata_concat = ad.concat(Batch_list, label="slice_name", keys=section_ids)
adata.obsm['spatial'] = adata_concat.obsm["spatial"]
adata.write("/home/zhaoshangrui/xuxinyu/STAligner/DLPFC/STAligner_sample1_alignment.h5ad")

# Clustering
import os
os.environ['R_HOME'] = '/home/zhaoshangrui/anaconda2/envs/R4.3.1/lib/R'
os.environ['R_LIBS'] = '/home/zhaoshangrui/anaconda2/envs/R4.3.1/lib/R/library'

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

adata_concat=sc.read_h5ad("/home/zhaoshangrui/xuxinyu/STAligner/DLPFC/STAligner_sample1.h5ad")
mclust_R(adata_concat, num_cluster=7, used_obsm='STAligner')
adata_concat.write("/home/zhaoshangrui/xuxinyu/STAligner/DLPFC/STAligner_mclust_sample1.h5ad")

#ARI & NMI
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
print('mclust, ARI = %01.3f' % adjusted_rand_score(adata_concat.obs['Ground Truth'], adata_concat.obs['mclust']))#mclust, 
print('mclust, NMI = %01.3f' % normalized_mutual_info_score(adata_concat.obs['Ground Truth'], adata_concat.obs['mclust']))#mclust, 

# Visualization
sc.pp.neighbors(adata_concat, use_rep='STAligner', random_state=666)
sc.tl.umap(adata_concat, random_state=666)

section_color = ['#f8766d', '#7cae00', '#00bfc4', '#c77cff']
section_color_dict = dict(zip(section_ids, section_color))
adata_concat.uns['batch_name_colors'] = [section_color_dict[x] for x in adata_concat.obs.batch_name.cat.categories]

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
    
adata_concat.obs['mclust'] = pd.Series(match_cluster_labels(adata_concat.obs['Ground Truth'], adata_concat.obs['mclust'].values),
                                         index=adata_concat.obs.index, dtype='category')

# UMAP PLOT
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(20, 3))
sc.pl.umap(adata_concat, color='batch_name', ax=axes[0], show=False, legend_fontsize=10, palette='coolwarm')
axes[0].set_title('Slice')
axes[0].set_xlabel('')
axes[0].set_ylabel('')
sc.pl.umap(adata_concat, color='Ground Truth', ax=axes[1], show=False, legend_fontsize=10, palette='cividis')
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
plt.savefig("/home/zhaoshangrui/xuxinyu/STAligner/DLPFC/STAligner_UMAP_sample1.pdf")
