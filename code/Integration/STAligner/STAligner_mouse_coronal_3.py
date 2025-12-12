# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")

import STAligner

import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np
import scipy.linalg
from scipy.sparse import csr_matrix

import torch
used_device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

result_dirs="/home/zhaoshangrui/xuxinyu/SPIRAL/Coronal_mouse_brain/result/"

sample_name=["10X_Normal","10X_DAPI","10X_FFPE"]
Batch_list = []
adj_list = []
rads=[150,300,300]

for k in range(len(sample_name)):
    adata=sc.read_csv(result_dirs+"gtt_input/"+str(sample_name[k])+"_mat.csv")
    adata.X = csr_matrix(adata.X)
    coord=pd.read_csv(result_dirs+"gtt_input/"+str(sample_name[k])+"_coord1.csv",header=0,index_col=0,sep=',')
    coord.columns=['x','y']
    adata.obsm['spatial'] = coord.values
    meta=pd.read_csv(result_dirs+"gtt_input/"+str(sample_name[k])+"_meta.csv",header=0,index_col=0,sep=',')
    adata.obs['celltype']=meta.loc[adata.obs_names,'celltype'].values
    adata.var_names_make_unique(join="++")
    adata.obs_names = [sample_name[k]+'-'+x for x in adata.obs_names]
    # Constructing the spatial network
    STAligner.Cal_Spatial_Net(adata, k_cutoff=6,model='Radius',rad_cutoff=rads[k]) # the spatial network are saved in adata.uns[‘adj’]
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
adata_concat.write("/home/zhaoshangrui/xuxinyu/STAligner/Coronal_mouse_brain/Coronal_mouse_brain_new.h5ad")

adata_concat = sc.read_h5ad("/home/zhaoshangrui/xuxinyu/STAligner/Coronal_mouse_brain/Coronal_mouse_brain_new.h5ad")

st_aligner_df = pd.DataFrame(adata_concat.obsm['STAligner'], index=adata_concat.obs.index)
st_aligner_df.to_csv('/home/zhaoshangrui/xuxinyu/STAligner/Coronal_mouse_brain/STAligner_embedding_new.csv')

