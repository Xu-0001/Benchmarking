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

def process_data(adata, n_top_genes=2000):
    adata.var_names_make_unique(join="++")
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
    sc.pp.filter_cells(adata, min_counts=5000)
    sc.pp.filter_cells(adata, max_counts=35000)
    # adata = adata[adata.obs["pct_counts_mt"] < 20]
    sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata, flavor="seurat", n_top_genes=n_top_genes, subset=True
    )
    return adata

data_slice1 = sc.read_visium("/home/zhaoshangrui/xuxinyu/datasets/mouse_brain_data/Sagittal_mouse_brain_data/GPSA_mouse_brain_serial_section_1/")
data_slice1 = process_data(data_slice1, n_top_genes=6000)
data_slice1.obs['batch'] = 'section_1'
data_slice1.obs_names = [x+'_'+'section_1' for x in data_slice1.obs_names]
STAligner.Cal_Spatial_Net(data_slice1, rad_cutoff=150)
data_slice1 = data_slice1[:, data_slice1.var['highly_variable']]

data_slice2 = sc.read_visium("/home/zhaoshangrui/xuxinyu/datasets/mouse_brain_data/Sagittal_mouse_brain_data/GPSA_mouse_brain_serial_section_2/")
data_slice2 = process_data(data_slice2, n_top_genes=6000)
data_slice2.obs['batch'] = 'section_2'
data_slice2.obs_names = [x+'_'+'section_2' for x in data_slice2.obs_names]
STAligner.Cal_Spatial_Net(data_slice2, rad_cutoff=150)
data_slice2 = data_slice2[:, data_slice2.var['highly_variable']]

adj_list = [data_slice1.uns['adj'], data_slice2.uns['adj']]
Batch_list = [data_slice1, data_slice2]

section_ids = ['section_1','section_2']    
adata_concat = ad.concat(Batch_list, label="slice_name", keys=section_ids)
adata_concat.obs["batch_name"] = adata_concat.obs["slice_name"].astype('category')
print('adata_concat.shape: ', adata_concat.shape)

adj_concat = np.asarray(adj_list[0].todense())
for batch_id in range(1,len(section_ids)):
    adj_concat = scipy.linalg.block_diag(adj_concat, np.asarray(adj_list[batch_id].todense()))
adata_concat.uns['edgeList'] = np.nonzero(adj_concat)

adata_concat = STAligner.train_STAligner(adata_concat, verbose=True, knn_neigh = 100, device=used_device) #epochs = 1500,
adata_concat.uns['edgeList'] = list(adata_concat.uns['edgeList'])
adata_concat.write("/home/zhaoshangrui/xuxinyu/STAligner/mouse_brain/mouse_brain.h5ad")

staligner_df = pd.DataFrame(adata_concat.obsm['STAligner'])
staligner_df.index = adata_concat.obs_names
staligner_df.to_csv('/home/zhaoshangrui/xuxinyu/STAligner/mouse_brain/mouse_brain_STAligner_embedding.csv', index=True)

