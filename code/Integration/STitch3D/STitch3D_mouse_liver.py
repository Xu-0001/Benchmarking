# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import scipy.io
import scipy.sparse
import scanpy as sc
import matplotlib.pyplot as plt

import STitch3D

import warnings
warnings.filterwarnings("ignore")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

meta = pd.read_csv("/home/zhaoshangrui/xuxinyu/datasets/mouseLiver8/Adult-Liver/Adult-Liver_barcodes_anno.csv", index_col=0)
ref_count = pd.read_csv("/home/zhaoshangrui/xuxinyu/datasets/mouseLiver8/Adult-Liver/Adult-Liver_dge.csv", index_col=0)
ref_row = pd.read_csv("/home/zhaoshangrui/xuxinyu/datasets/mouseLiver8/Adult-Liver/Adult-Liver_gene.csv", index_col=0)

sc_count = ref_count
sc_count.index = ref_row['x']

cell_select = meta[meta['Idents.pbmc.'] != 'unclassified'].index
meta_filt = meta.loc[cell_select]
sc_count_filt = sc_count.loc[:, cell_select]

duplicate_gene_names = sc_count_filt.index.duplicated()
sc_count_filt.index = [f"{gene}_dup" if is_dup else gene for gene, is_dup in zip(sc_count_filt.index, duplicate_gene_names)]

adata_ref = sc.AnnData(X=sc_count_filt.T, obs=meta_filt)
adata_ref.obs['celltype'] = adata_ref.obs['Idents.pbmc.']

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
plt.savefig("/home/zhaoshangrui/xuxinyu/STitch3D/HCC/adata_ref_umap")

#Load spatial transcriptomics datasets
#spatial data
adata_st_raw = sc.read_h5ad("/home/zhaoshangrui/xuxinyu/datasets/mouseLiver8/seulist_mouseLiverST8.h5ad")
adata_st_raw.obsm['spatial'] = adata_st_raw.obsm['X_position']

adata_st_list_raw = []
for i in range(1,adata_st_raw.obs['batch'].nunique()+1):
    adata_st= adata_st_raw[adata_st_raw.obs['batch']==i]
    adata_st_list_raw.append(adata_st.copy())
    
adata_st_list = STitch3D.utils.align_spots(adata_st_list_raw, data_type="ST", coor_key="spatial", method='paste', plot=True, paste_alpha=0.1)

adata_st, adata_basis = STitch3D.utils.preprocess(adata_st_list,adata_ref,celltype_ref_col="celltype",n_hvg_group=500)

model = STitch3D.model.Model(adata_st, adata_basis)

model.train()

save_path = "/home/zhaoshangrui/xuxinyu/STitch3D/mouseLiver8/result"

result = model.eval(adata_st_list_raw, save=True, output_path=save_path)
