# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")

import STAligner

import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np
#import scipy.sparse as sp
import scipy.linalg
from scipy.sparse import csr_matrix

import torch
used_device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

data_dir = "/home/zhaoshangrui/xuxinyu/datasets/breast_tumour_data/breast_tumour_ST_data/"

def load_slices(data_dir, slice_names):
    slices = []  
    for slice_name in slice_names:
        slice_i = sc.read_csv(data_dir + slice_name + ".csv")
        slice_i_coor = np.genfromtxt(data_dir + slice_name + "_coor.csv", delimiter = ',')
        slice_i.obsm['spatial'] = slice_i_coor
        slice_i.obs['batch'] = slice_name
        # Preprocess slices
        sc.pp.filter_genes(slice_i, min_counts = 15)
        sc.pp.filter_cells(slice_i, min_counts = 100)
        sc.pp.normalize_total(slice_i, inplace=True)
        sc.pp.log1p(slice_i)
        sc.pp.highly_variable_genes(
            slice_i, flavor="seurat", n_top_genes=3000, subset=True
        )
        slice_i.X = csr_matrix(slice_i.X)
        slices.append(slice_i)
    return slices
slice_names =["stahl_bc_slice1", "stahl_bc_slice2", "stahl_bc_slice3", "stahl_bc_slice4"]
slices = load_slices(data_dir, slice_names)
slice1, slice2, slice3, slice4 = slices
    
Batch_list = []
adj_list = []
for i in range(4):
    print(i)
    data = slices[i]
    data.obs_names = [x+'_'+str(i) for x in data.obs_names]
    # Constructing the spatial network
    STAligner.Cal_Spatial_Net(data, rad_cutoff=1.5) # the spatial network are saved in adata.uns[‘adj’]
    # STAligner.Stats_Spatial_Net(adata) # plot the number of spatial neighbors
    adj_list.append(data.uns['adj'])
    Batch_list.append(data)
    
adata_concat = ad.concat(Batch_list)
adata_concat.obs["batch_name"] = adata_concat.obs["batch"].astype(str)
print('adata_concat.shape: ', adata_concat.shape)

adj_concat = np.asarray(adj_list[0].todense())
for batch_id in range(1,4):
    adj_concat = scipy.linalg.block_diag(adj_concat, np.asarray(adj_list[batch_id].todense()))
adata_concat.uns['edgeList'] = np.nonzero(adj_concat)

adata_concat = STAligner.train_STAligner(adata_concat, verbose=True, knn_neigh = 100, device=used_device) #epochs = 1500,
adata_concat.uns['edgeList'] = list(adata_concat.uns['edgeList'])
adata_concat.write("/home/zhaoshangrui/xuxinyu/STAligner/breast_tumour/breast_tumour.h5ad")

staligner_df = pd.DataFrame(adata_concat.obsm['STAligner'])
staligner_df.index = adata_concat.obs_names
staligner_df.to_csv('/home/zhaoshangrui/xuxinyu/STAligner/breast_tumour/breast_tumour_STAligner_embedding.csv', index=True)

#Clustering
sc.pp.neighbors(adata_concat, use_rep='STAligner', random_state=666)
sc.tl.louvain(adata_concat, random_state=666, key_added="louvain", resolution=1)
adata_concat.write("/home/zhaoshangrui/xuxinyu/STAligner/breast_tumour/breast_tumour_cluster.h5ad")



