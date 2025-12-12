# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")

import STAligner

import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np
import scipy.linalg

import torch
used_device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


dirs = "/home/zhaoshangrui/xuxinyu/datasets/mouse_brain_data/Sagittal_mouse_brain_data/"

sample_name=["posterior1","posterior2"]
Batch_list = []
adj_list = []
for i in range(len(sample_name)):
    adata = sc.read_visium(path=dirs+str(sample_name[i])+"/",
                        count_file="filtered_feature_bc_matrix.h5", load_images=True)
    adata.var_names_make_unique(join="++")
    adata.obs_names = [sample_name[i]+'-'+x for x in adata.obs_names]
    # Constructing the spatial network
    STAligner.Cal_Spatial_Net(adata, rad_cutoff=150) 
    # Normalization
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=5000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata = adata[:, adata.var['highly_variable']]
    adj_list.append(adata.uns['adj'])
    Batch_list.append(adata)
    
adata_concat = ad.concat(Batch_list, label="slice_name", keys=sample_name)
adata_concat.obs["batch_name"] = adata_concat.obs["slice_name"].astype('category')
print('adata_concat.shape: ', adata_concat.shape)

adj_concat = np.asarray(adj_list[0].todense())
for batch_id in range(1,len(sample_name)):
    adj_concat = scipy.linalg.block_diag(adj_concat, np.asarray(adj_list[batch_id].todense()))
adata_concat.uns['edgeList'] = np.nonzero(adj_concat)

adata_concat = STAligner.train_STAligner(adata_concat, verbose=True, knn_neigh = 100, device=used_device)
adata_concat.uns['edgeList'] = list(adata_concat.uns['edgeList'])
adata_concat.write("/home/zhaoshangrui/xuxinyu/STAligner/Sagittal_mouse_brain_data/Sagittal_mouse_brain_posterior.h5ad")

