# -*- coding: utf-8 -*-
import pandas as pd
import scanpy as sc
import anndata as ad
import scipy.io
import matplotlib
matplotlib.use('agg')
import os

import STitch3D

import warnings
warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

###Load single-cell reference dataset:
mat = scipy.io.mmread(".../GeneBarcodeMatrix_Annotated.mtx")
meta = pd.read_csv(".../CellNames.csv", index_col=0)
meta.index = meta.x.values
group = [i.split('.')[1].split('_')[0] for i in list(meta.x.values)]
condition = [i.split('.')[1].split('_')[1] for i in list(meta.x.values)]
celltype = [i.split('.')[0] for i in list(meta.x.values)]
meta["group"] = group
meta["condition"] = condition
meta["celltype"] = celltype
genename = pd.read_csv(".../GeneNames.csv", index_col=0)
genename.index = genename.x.values
adata_ref = ad.AnnData(X=mat.tocsr().T)
adata_ref.obs = meta
adata_ref.var = genename
adata_ref = adata_ref[adata_ref.obs.condition.values.astype(str)=="Control", :]

###Load spatial transcriptomics datasets:
#spatial data
anno_df = pd.read_csv('.../barcode_level_layer_map.tsv', sep='\t', header=None)

slice_idx = [0,1,2,3]

adata_st1 = sc.read_visium(path=".../%d" % slice_idx[0], count_file="filtered_feature_bc_matrix.h5")
anno_df1 = anno_df.iloc[anno_df[1].values.astype(str) == str(slice_idx[0])]
anno_df1.columns = ["barcode", "slice_id", "layer"]
anno_df1.index = anno_df1['barcode']
adata_st1.obs = adata_st1.obs.join(anno_df1, how="left")
# adata_st1 = adata_st1[adata_st1.obs['layer'].notna()]
adata_st1.obs['layer'].fillna('unknown', inplace=True)

adata_st2 = sc.read_visium(path=".../%d" % slice_idx[1], count_file="filtered_feature_bc_matrix.h5")
anno_df2 = anno_df.iloc[anno_df[1].values.astype(str) == str(slice_idx[1])]
anno_df2.columns = ["barcode", "slice_id", "layer"]
anno_df2.index = anno_df2['barcode']
adata_st2.obs = adata_st2.obs.join(anno_df2, how="left")
# adata_st2 = adata_st2[adata_st2.obs['layer'].notna()]
adata_st2.obs['layer'].fillna('unknown', inplace=True)

adata_st3 = sc.read_visium(path=".../%d" % slice_idx[2], count_file="filtered_feature_bc_matrix.h5")
anno_df3 = anno_df.iloc[anno_df[1].values.astype(str) == str(slice_idx[2])]
anno_df3.columns = ["barcode", "slice_id", "layer"]
anno_df3.index = anno_df3['barcode']
adata_st3.obs = adata_st3.obs.join(anno_df3, how="left")
# adata_st3 = adata_st3[adata_st3.obs['layer'].notna()]
adata_st3.obs['layer'].fillna('unknown', inplace=True)

adata_st4 = sc.read_visium(path=".../%d" % slice_idx[3], count_file="filtered_feature_bc_matrix.h5")
anno_df4 = anno_df.iloc[anno_df[1].values.astype(str) == str(slice_idx[3])]
anno_df4.columns = ["barcode", "slice_id", "layer"]
anno_df4.index = anno_df4['barcode']
adata_st4.obs = adata_st4.obs.join(anno_df4, how="left")
# adata_st4 = adata_st4[adata_st4.obs['layer'].notna()]
adata_st4.obs['layer'].fillna('unknown', inplace=True)

###Alignment of ST tissue slices
adata_st_list_raw = [adata_st1, adata_st2, adata_st3, adata_st4]
adata_st_list = STitch3D.utils.align_spots(adata_st_list_raw, plot=True)

celltype_list_use = [...]

adata_st, adata_basis = STitch3D.utils.preprocess(adata_st_list,
                                                  adata_ref,
                                                  celltype_ref=celltype_list_use,
                                                  sample_col="group",
                                                  slice_dist_micron=[10., 300., 10.],
                                                  n_hvg_group=500)

model = STitch3D.model.Model(adata_st, adata_basis)

model.train()


save_path = "..."
result = model.eval(adata_st_list_raw, save=True, output_path=save_path)