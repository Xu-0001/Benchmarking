# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
from scipy.io import mmread
import os

import STitch3D

import warnings
warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import matplotlib.pyplot as plt


###Preprocessing
#Load single-cell reference dataset:
ref_count = mmread("/home/zhaoshangrui/xuxinyu/STitch3D/Drosophila_embryo/GSE190147_scirnaseq_gene_matrix.mtx")
ref_row = pd.read_csv("/home/zhaoshangrui/xuxinyu/STitch3D/Drosophila_embryo/GSE190147_scirnaseq_gene_matrix.rows.txt",
                      header=None, index_col=0)
ref_col = pd.read_csv("/home/zhaoshangrui/xuxinyu/STitch3D/Drosophila_embryo/GSE190147_scirnaseq_gene_matrix.columns.txt",
                      index_col=0, sep='\t')

adata_ref_raw = ad.AnnData(X=ref_count.tocsr().T)
adata_ref_raw.obs = ref_col
adata_ref_raw.var.index = [str(i) for i in list(ref_row.index)]

adata_ref = ad.AnnData(X=adata_ref_raw.X)
adata_ref.obs = adata_ref_raw.obs
adata_ref.var = adata_ref_raw.var

adata_ref = adata_ref[adata_ref.obs['time'] == 'hrs_16_20']
adata_ref.obs.rename(columns = {'exp': 'exp_idx'}, inplace = True)

clust = pd.read_csv("/home/zhaoshangrui/xuxinyu/STitch3D/Drosophila_embryo/seurat_clust_1620.txt", sep=" ")
clust.index = clust.barcode.values.astype(str)
clust = clust.loc[adata_ref.obs.index, :]
adata_ref.obs["seurat_clust"] = clust.clust.values.astype(str)

#transfer clusters to cell type annotations
anno_table = pd.read_csv("/home/zhaoshangrui/xuxinyu/STitch3D/Drosophila_embryo/cluster_anno_table.csv")

adata_ref.obs['celltype'] = adata_ref.obs['seurat_clust'].values.astype(str)
for i in range(anno_table.shape[0]):
    adata_ref.obs['celltype'] = adata_ref.obs['celltype'].replace(anno_table["cluster"][i].astype(str),anno_table["annotation"][i])

adata_ref = adata_ref[(adata_ref.obs['celltype'] != "lowq") & (adata_ref.obs['celltype'] != "unk"), :]
clust = clust.loc[adata_ref.obs.index, :]

#plot umap
adata_ref_umap = adata_ref.copy()
hvg_num = 2000
sc.pp.highly_variable_genes(adata_ref_umap, flavor='seurat_v3', n_top_genes=hvg_num)
sc.pp.normalize_total(adata_ref_umap, target_sum=1e4)
sc.pp.log1p(adata_ref_umap)
sc.pp.scale(adata_ref_umap, max_value=10)
sc.tl.pca(adata_ref_umap, n_comps=30, svd_solver='arpack')
sc.pp.neighbors(adata_ref_umap, n_pcs=30)
sc.tl.umap(adata_ref_umap)
sc.pl.umap(adata_ref_umap, color=['celltype'])
plt.savefig("/home/zhaoshangrui/xuxinyu/STitch3D/Drosophila_embryo16_18/adata_ref_umap")

#Load spatial transcriptomics datasets
#spatial data
adata_st_raw = sc.read_h5ad("/home/zhaoshangrui/xuxinyu/datasets/Drosophila_embryo_dataset_profiled_by_Stereo-seq/E16-18h_a_count_normal_stereoseq.h5ad")
adata_st_raw.X = adata_st_raw.layers['raw_counts']

slice_all = sorted(list(set(adata_st_raw.obs['slice_ID'].values)))[:-1]

adata_st_list_raw = []
for slice_id in slice_all:
    adata_st_i = adata_st_raw[adata_st_raw.obs['slice_ID'].values == slice_id]
    adata_st_i.obsm['spatial'] = np.concatenate((adata_st_i.obs['raw_x'].values.reshape(-1, 1),
                                                 adata_st_i.obs['raw_y'].values.reshape(-1, 1)), axis=1) / 20
    adata_st_i.obsm['loc_use'] = np.concatenate((adata_st_i.obs['raw_x'].values.reshape(-1, 1),
                                                 adata_st_i.obs['raw_y'].values.reshape(-1, 1)), axis=1) / 20
    adata_st_i.obsm['coor_3d'] = np.concatenate((adata_st_i.obs['new_x'].values.reshape(-1, 1),
                                                 adata_st_i.obs['new_y'].values.reshape(-1, 1),
                                                 adata_st_i.obs['new_z'].values.reshape(-1, 1)), axis=1)
    adata_st_i.obs['array_row'] = adata_st_i.obs['raw_y'].values
    adata_st_i.obs['array_col'] = adata_st_i.obs['raw_x'].values
    adata_st_list_raw.append(adata_st_i.copy())
    
adata_st_list = STitch3D.utils.align_spots(adata_st_list_raw, data_type="ST", coor_key="loc_use",method='paste', plot=True, paste_alpha=0.2)

adata_st, adata_basis = STitch3D.utils.preprocess(adata_st_list,adata_ref,celltype_ref_col="celltype",sample_col="exp_idx",n_hvg_group=500,slice_dist_micron=[1.]*(len(adata_st_list)-1),c2c_dist=1.)


# import time
# start_time = time.time()  
model = STitch3D.model.Model(adata_st, adata_basis)
model.train()
# end_time = time.time()
# elapsed_time = end_time - start_time
# print(f"实验运行时间: {elapsed_time:.2f} 秒")

save_path = "/home/zhaoshangrui/xuxinyu/STitch3D/Drosophila_embryo/result/results_Drosophila_embryo"
result = model.eval(adata_st_list_raw, save=True, output_path=save_path)