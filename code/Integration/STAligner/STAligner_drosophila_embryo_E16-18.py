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

adata_st_raw = sc.read_h5ad("/home/zhaoshangrui/xuxinyu/datasets/Drosophila_embryo_dataset_profiled_by_Stereo-seq/E16-18h_a_count_normal_stereoseq.h5ad")
adata_st_raw.X = adata_st_raw.layers['raw_counts']
adata_st_raw.obsm['spatial'] = adata_st_raw.obsm['spatial'][:, :2]

slice_all = sorted(list(set(adata_st_raw.obs['slice_ID'].values)))[:-1]

slices_ID = ['E16-18h_a_S01','E16-18h_a_S02','E16-18h_a_S03','E16-18h_a_S04','E16-18h_a_S05','E16-18h_a_S06','E16-18h_a_S07','E16-18h_a_S08',
  'E16-18h_a_S09','E16-18h_a_S10','E16-18h_a_S11','E16-18h_a_S12','E16-18h_a_S13']

Batch_list = []
adj_list = []
for slice_id in slice_all:
    adata_st_i = adata_st_raw[adata_st_raw.obs['slice_ID'].values == slice_id]
    adata_st_i.obsm['spatial'] = np.concatenate((adata_st_i.obs['raw_x'].values.reshape(-1, 1),
                                                  adata_st_i.obs['raw_y'].values.reshape(-1, 1)), axis=1) / 20
    adata_st_i.obsm['loc_use'] = np.concatenate((adata_st_i.obs['raw_x'].values.reshape(-1, 1),
                                                  adata_st_i.obs['raw_y'].values.reshape(-1, 1)), axis=1) / 20
    adata_st_i.obsm['coor_3d'] = np.concatenate((adata_st_i.obs['new_x'].values.reshape(-1, 1),
                                                  adata_st_i.obs['new_y'].values.reshape(-1, 1),
                                                  adata_st_i.obs['new_z'].values.reshape(-1, 1)), axis=1)
    adata_st_i.obs['array_row'] = adata_st_i.obs['raw_y'].values
    adata_st_i.obs['array_col'] = adata_st_i.obs['raw_x'].values
    adata_st_i.obs_names = [x+'_'+str(slice_id) for x in adata_st_i.obs_names]
    adata_st_i.X = csr_matrix(adata_st_i.X)
    # Constructing the spatial network
    STAligner.Cal_Spatial_Net(adata_st_i, rad_cutoff=1.5) # the spatial network are saved in adata.uns[‘adj’]
    # Normalization
    sc.pp.normalize_total(adata_st_i, target_sum=1e4)
    sc.pp.log1p(adata_st_i)
    sc.pp.highly_variable_genes(adata_st_i, flavor="seurat_v3", n_top_genes=5000)
    adj_list.append(adata_st_i.uns['adj'])
    Batch_list.append(adata_st_i.copy())
    
# Concat the scanpy objects for multiple slices    
adata_concat = ad.concat(Batch_list)
adata_concat.obs["batch_name"], unique_batches = pd.factorize(adata_concat.obs["slice_ID"])
# adata_concat.obs["batch_name"] = adata_concat.obs["batch_name"].astype('category')
adata_concat.obs["batch_name"] = adata_concat.obs["batch_name"].astype(str)
print('adata_concat.shape: ', adata_concat.shape)

# Concat the spatial network for multiple slices
adj_concat = np.asarray(adj_list[0].todense())
for batch_id in range(1,len(slices_ID)):
    adj_concat = scipy.linalg.block_diag(adj_concat, np.asarray(adj_list[batch_id].todense()))
adata_concat.uns['edgeList'] = np.nonzero(adj_concat)

# Running STAligner
adata_concat = STAligner.train_STAligner(adata_concat, verbose=True, knn_neigh = 100, device=used_device) #epochs = 1500,
adata_concat.uns['edgeList'] = list(adata_concat.uns['edgeList'])
adata_concat.write("/home/zhaoshangrui/xuxinyu/STAligner/Drosophila_embryo16-18/Drosophila_embryo16-18.h5ad")

staligner_df = pd.DataFrame(adata_concat.obsm['STAligner'])
staligner_df.index = adata_concat.obs_names
staligner_df.to_csv('/home/zhaoshangrui/xuxinyu/STAligner/Drosophila_embryo16-18/Drosophila_embryo16-18_STAligner_embed.csv', index=True)
