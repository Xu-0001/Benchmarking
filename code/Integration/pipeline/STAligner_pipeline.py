# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")

import STAligner

import anndata as ad
import scanpy as sc
import numpy as np

import scipy.linalg
from scipy.sparse import csr_matrix

import torch
used_device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

Batch_list = []
adj_list = []
section_ids = ['data_slice1','data_slice2']
print(section_ids)

data_slice1 = sc.read_h5ad(".../data_slice1.h5ad")
data_slice1.X = csr_matrix(data_slice1.X)
data_slice1.obs_names = [x+'_'+'Puck_200115_08' for x in data_slice1.obs_names]
STAligner.Cal_Spatial_Net(data_slice1, rad_cutoff=0.2)
adj_list.append(data_slice1.uns['adj'])  
Batch_list.append(data_slice1)                     
                          
data_slice2 = sc.read_h5ad(".../data_slice2.h5ad")
data_slice2.X = csr_matrix(data_slice2.X)
data_slice2.obs_names = [x+'_'+'Puck_191204_01' for x in data_slice2.obs_names]
STAligner.Cal_Spatial_Net(data_slice2, rad_cutoff=0.2)
adj_list.append(data_slice2.uns['adj'])  
Batch_list.append(data_slice2) 

adata_concat = ad.concat(Batch_list, label="slice_name", keys=section_ids)
adata_concat.obs["batch_name"] = adata_concat.obs["slice_name"].astype('category')
print('adata_concat.shape: ', adata_concat.shape)

adj_concat = np.asarray(adj_list[0].todense())
for batch_id in range(1,len(section_ids)):
    adj_concat = scipy.linalg.block_diag(adj_concat, np.asarray(adj_list[batch_id].todense()))
adata_concat.uns['edgeList'] = np.nonzero(adj_concat)

adata_concat = STAligner.train_STAligner(adata_concat, verbose=True, knn_neigh = 100, device=used_device) #epochs = 1500,
adata_concat.uns['edgeList'] = list(adata_concat.uns['edgeList'])
adata_concat.write(".../adata_concat.h5ad")
