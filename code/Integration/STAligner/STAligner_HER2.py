# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")

import STAligner

import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np
import scipy.sparse as sp
import scipy.linalg

import torch
used_device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')


section_ids = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6',
                'B1', 'B2', 'B3', 'B4', 'B5', 'B6',
                'C1', 'C2', 'C3', 'C4', 'C5', 'C6',
                'D1', 'D2', 'D3', 'D4', 'D5', 'D6',
                'E1', 'E2', 'E3',
                'F1', 'F2', 'F3',
                'G1', 'G2', 'G3',
                'H1', 'H2', 'H3',]
data_path = "/home/zhaoshangrui/xuxinyu/datasets/breast_tumour_data/HER2-positive_breast_tumour_dataset_profiled_by_ST_platform/raw_data/"
samples_per_patient = {'A':6,'B':6,'C':6,'D':6,'E':3,'F':3,'G':3,'H':3}
patients = {p:[0 for i in range(samples_per_patient[p])] for p in samples_per_patient}
for p in samples_per_patient:
    print('Patient ',p)
    for i in range(samples_per_patient[p]):
        print("\t Slice ",i+1)
        st_count = pd.read_csv(data_path + '{0}{1}.tsv.gz'.format(p,i+1), index_col=0, sep='\t')
        st_meta = pd.read_csv(data_path + '{0}{1}_selection.tsv'.format(p,i+1), sep='\t')
        st_meta.index = [str(st_meta['x'][i]) + 'x' + str(st_meta['y'][i]) for i in range(st_meta.shape[0])]
        st_meta = st_meta.loc[st_count.index]
        adata_st_i = ad.AnnData(X=st_count.values)
        adata_st_i.obs.index = st_count.index
        adata_st_i.obsm['spatial'] = np.array(list(map(lambda x: x.split('x'),adata_st_i.obs.index)),dtype=float)
        adata_st_i.obs = st_meta
        adata_st_i.var.index = st_count.columns
        sc.pp.filter_genes(adata_st_i, min_counts = 15)
        sc.pp.filter_cells(adata_st_i, min_counts = 100)
        print(adata_st_i.shape)
        patho_anno_file = '{0}2_labeled_coordinates.tsv' if p == 'G' else '{0}1_labeled_coordinates.tsv'
        patho_anno = pd.read_csv(data_path + patho_anno_file.format(p), delimiter='\t')
        if i == (1 if p == 'G' else 0):
            adata_st_i.obs['annotation'] = None
            for idx in adata_st_i.obs.index:
                adata_st_i.obs['annotation'].loc[idx] = patho_anno[
                    (patho_anno["x"] == adata_st_i[idx].obs["new_x"].values[0]) & 
                    (patho_anno["y"] == adata_st_i[idx].obs["new_y"].values[0])
                ]['label'].values[0]
        else:
            adata_st_i.obs['annotation'] = "Unknown" 
        adata_st_i.X = sp.csc_matrix(adata_st_i.X)     
        patients[p][i] = adata_st_i 

Batch_list = []
adj_list = []
for i in range(6):
    adata = patients["A"][i]
    adata.obs_names = [f"{x}_A_{i}" for x in adata.obs_names]
    STAligner.Cal_Spatial_Net(adata, rad_cutoff=1.5)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=5000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata = adata[:, adata.var['highly_variable']]
    adj_list.append(adata.uns['adj'])
    Batch_list.append(adata)

adata_concat = ad.concat(Batch_list, label="batch")
adata_concat.obs["batch_name"], unique_batches = pd.factorize(adata_concat.obs["batch"])
adata_concat.obs["batch_name"] = adata_concat.obs["batch_name"].astype(str)
print('adata_concat.shape: ', adata_concat.shape)

adj_concat = np.asarray(adj_list[0].todense())
for batch_id in range(1,6):
    adj_concat = scipy.linalg.block_diag(adj_concat, np.asarray(adj_list[batch_id].todense()))
adata_concat.uns['edgeList'] = np.nonzero(adj_concat)

adata_concat = STAligner.train_STAligner(adata_concat, verbose=True, knn_neigh = 100, device=used_device) #epochs = 1500,
adata_concat.uns['edgeList'] = list(adata_concat.uns['edgeList'])
adata_concat.write("/home/zhaoshangrui/xuxinyu/STAligner/HER2/HER2_A.h5ad")

staligner_df = pd.DataFrame(adata_concat.obsm['STAligner'])
staligner_df.index = adata_concat.obs_names
staligner_df.to_csv('/home/zhaoshangrui/xuxinyu/STAligner/HER2/HER2_STAligner_embedding_A.csv', index=True)
    