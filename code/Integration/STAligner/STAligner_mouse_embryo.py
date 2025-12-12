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

import torch
used_device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

Batch_list = []
adj_list = []
section_ids = ['E9.5_E1S1', 'E10.5_E2S1', 'E11.5_E1S1', 'E12.5_E1S1']
for section_id in section_ids:
    print(section_id)
    adata = sc.read_h5ad(os.path.join("/home/zhaoshangrui/xuxinyu/datasets/mouse_embryo_Stereo-seq/" + section_id + ".MOSTA.h5ad"))
    adata.X = adata.layers['count']
    # make spot name unique
    adata.obs_names = [x + '_' + section_id for x in adata.obs_names]
    STAligner.Cal_Spatial_Net(adata, rad_cutoff=1.3)
    # Normalization
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=5000) #ensure enough common HVGs in the combined matrix
    adata = adata[:, adata.var['highly_variable']]
    adj_list.append(adata.uns['adj'])
    Batch_list.append(adata)

adata_concat = ad.concat(Batch_list, label="slice_name", keys=section_ids)
adata_concat.obs["batch_name"] = adata_concat.obs["slice_name"].astype('category')
print('adata_concat.shape: ', adata_concat.shape)

adj_concat = np.asarray(adj_list[0].todense())
for batch_id in range(1,len(section_ids)):
    adj_concat = scipy.linalg.block_diag(adj_concat, np.asarray(adj_list[batch_id].todense()))
adata_concat.uns['edgeList'] = np.nonzero(adj_concat) 

# Important parameter:
# "iter_comb" is used to specify the order of integration
# "margin" is used to control the intensity/weight of batch correction
iter_comb = [(0, 3), (1, 3), (2, 3)] ## Fix slice 3 as reference to align

adata_concat = STAligner.train_STAligner(adata_concat, verbose=True, knn_neigh = 100, iter_comb = iter_comb,
                                                        margin=2.5,  device=used_device)
adata_concat.uns['edgeList'] = list(adata_concat.uns['edgeList'])
adata_concat.write("/home/zhaoshangrui/xuxinyu/STAligner/mouse_embryo/mouse_embryo.h5ad")

staligner_df = pd.DataFrame(adata_concat.obsm['STAligner'])
staligner_df.index = adata_concat.obs_names
staligner_df.to_csv("/home/zhaoshangrui/xuxinyu/STAligner/mouse_embryo/mouse_embryo_STAligner_embedding.csv", index=True)
