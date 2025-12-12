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

adata = sc.read_h5ad("/home/zhaoshangrui/xuxinyu/datasets/Human_hepatocellular_carcinoma/HCC4_seulist.h5ad")
adata.obsm['spatial'] = adata.obsm['X_position']
num = adata.obs['batch'].nunique()    
Batch_list = []
adj_list = []
for i in range(1,adata.obs['batch'].nunique()+1):
    print(i)
    data = adata[adata.obs['batch']==i]
    data.obs_names = [x+'_'+str(i) for x in data.obs_names]
    # Constructing the spatial network
    STAligner.Cal_Spatial_Net(data, rad_cutoff=2.1) # the spatial network are saved in adata.uns[‘adj’]
    # STAligner.Stats_Spatial_Net(adata) # plot the number of spatial neighbors
    adj_list.append(data.uns['adj'])
    Batch_list.append(data)
    
adata_concat = ad.concat(Batch_list)
adata_concat.obs["batch_name"], unique_batches = pd.factorize(adata_concat.obs["batch"])
adata_concat.obs["batch_name"] = adata_concat.obs["batch_name"].astype(str)
print('adata_concat.shape: ', adata_concat.shape)

adj_concat = np.asarray(adj_list[0].todense())
for batch_id in range(1,adata.obs['batch'].nunique()):
    adj_concat = scipy.linalg.block_diag(adj_concat, np.asarray(adj_list[batch_id].todense()))
adata_concat.uns['edgeList'] = np.nonzero(adj_concat)

adata_concat = STAligner.train_STAligner(adata_concat, verbose=True, knn_neigh = 100, device=used_device) #epochs = 1500,
adata_concat.uns['edgeList'] = list(adata_concat.uns['edgeList'])
adata_concat.write("/home/zhaoshangrui/xuxinyu/STAligner/HCC/HCC.h5ad")

staligner_df = pd.DataFrame(adata_concat.obsm['STAligner'])
staligner_df.index = adata_concat.obs_names
staligner_df.to_csv('/home/zhaoshangrui/xuxinyu/STAligner/HCC/HCC_STAligner_embed.csv', index=True)

