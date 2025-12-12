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
used_device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

data = sc.read_h5ad("/home/zhaoshangrui/xuxinyu/STitch3D/adult_mouse_brain/adata_st.h5ad")
data.X = csr_matrix(data.X)
data.obs['cluster_name'] = data.obs['cluster_name'].cat.add_categories(['unknown'])
data.obs['cluster_name'].fillna('unknown', inplace=True)

Batch_list = []
adj_list = []
slices_ID = ['01A', '02A', '03A', '04B', '05A', '06B', '07A', '08B', '09A', '10B',
 '11A', '12A', '13B', '14A', '15A', '16A', '17A', '18A', '19A', '20B',
 '21A', '22A', '23A', '24A', '25A', '26A', '27A', '28A', '29A', '30A',
 '31A', '32A', '33A', '34A', '35A',]
for i in range(len(slices_ID)):
    print(slices_ID[i])
    adata = data[data.obs['slice']==i]
    # make spot name unique
    adata.obs_names = [x + '_' + slices_ID[i] for x in adata.obs_names]
    STAligner.Cal_Spatial_Net(adata, rad_cutoff=500)
    adata.var_names_make_unique(join="++") 
    adj_list.append(adata.uns['adj'])
    Batch_list.append(adata)
    
adata_concat = ad.concat(Batch_list, label="slice_name", keys=slices_ID)
adata_concat.obs["batch_name"] = adata_concat.obs["slice_name"].astype('category')
print('adata_concat.shape: ', adata_concat.shape)

adj_concat = np.asarray(adj_list[0].todense())
for batch_id in range(1,len(slices_ID)):
    adj_concat = scipy.linalg.block_diag(adj_concat, np.asarray(adj_list[batch_id].todense()))
adata_concat.uns['edgeList'] = np.nonzero(adj_concat) 

# Important parameter:
adata_concat = STAligner.train_STAligner(adata_concat, verbose=True, knn_neigh = 100, device=used_device)
adata_concat.uns['edgeList'] = list(adata_concat.uns['edgeList'])
adata_concat.write("/home/zhaoshangrui/xuxinyu/STAligner/adult_mouse_brain/adult_mouse_brain.h5ad")

staligner_df = pd.DataFrame(adata_concat.obsm['STAligner'])
staligner_df.index = adata_concat.obs_names
staligner_df.to_csv("/home/zhaoshangrui/xuxinyu/STAligner/adult_mouse_brain/adult_mouse_brain_STAligner_embedding.csv", index=True)