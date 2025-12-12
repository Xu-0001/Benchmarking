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

data = sc.read_h5ad("/home/zhaoshangrui/xuxinyu/STitch3D/human_heart/adata_st.h5ad")
data.X = csr_matrix(data.X)

Batch_list = []
adj_list = []
section_ids = ['FH6_1000L2_CN73_C2', 'FH6_1000L2_CN73_D2', 'FH6_1000L2_CN73_E1', 'FH6_1000L2_CN73_E2', 'FH6_1000L2_CN74_C1', 'FH6_1000L2_CN74_D1', 'FH6_1000L2_CN74_D2', 'FH6_1000L2_CN74_E1', 'FH6_1000L2_CN74_E2']
for section_id in section_ids:
    print(section_id)
    adata = data[data.obs['Sample']==section_id]
    # make spot name unique
    adata.obs_names = [x + '_' + section_id for x in adata.obs_names]
    STAligner.Cal_Spatial_Net(adata, rad_cutoff=300)
    adj_list.append(adata.uns['adj'])
    Batch_list.append(adata)
    
adata_concat = ad.concat(Batch_list, label="slice_name", keys=section_ids)
adata_concat.obs["batch_name"] = adata_concat.obs["slice_name"].astype('category')
print('adata_concat.shape: ', adata_concat.shape)

adj_concat = np.asarray(adj_list[0].todense())
for batch_id in range(1,len(section_ids)):
    adj_concat = scipy.linalg.block_diag(adj_concat, np.asarray(adj_list[batch_id].todense()))
adata_concat.uns['edgeList'] = np.nonzero(adj_concat) 

adata_concat = STAligner.train_STAligner(adata_concat, verbose=True, knn_neigh = 100, device=used_device)
adata_concat.uns['edgeList'] = list(adata_concat.uns['edgeList'])
adata_concat.write("/home/zhaoshangrui/xuxinyu/STAligner/human_heart/human_heart.h5ad")

staligner_df = pd.DataFrame(adata_concat.obsm['STAligner'])
staligner_df.index = adata_concat.obs_names
staligner_df.to_csv("/home/zhaoshangrui/xuxinyu/STAligner/human_heart/human_heart_STAligner_embedding.csv", index=True)
