# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
import scipy.io
import matplotlib#
matplotlib.use('agg')#
import matplotlib.pyplot as plt
import os
import sys

import STitch3D

import warnings
warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

###Load single-cell reference dataset:
mat = scipy.io.mmread("/home/zhaoshangrui/xuxinyu/datasets/DLPFC_data/10xGenomicsChromiumplatform/GSE144136/GSE144136_GeneBarcodeMatrix_Annotated.mtx")
meta = pd.read_csv("/home/zhaoshangrui/xuxinyu/datasets/DLPFC_data/10xGenomicsChromiumplatform/GSE144136/GSE144136_CellNames.csv", index_col=0)
meta.index = meta.x.values
group = [i.split('.')[1].split('_')[0] for i in list(meta.x.values)]
condition = [i.split('.')[1].split('_')[1] for i in list(meta.x.values)]
celltype = [i.split('.')[0] for i in list(meta.x.values)]
meta["group"] = group
meta["condition"] = condition
meta["celltype"] = celltype
genename = pd.read_csv("/home/zhaoshangrui/xuxinyu/datasets/DLPFC_data/10xGenomicsChromiumplatform/GSE144136/GSE144136_GeneNames.csv", index_col=0)
genename.index = genename.x.values
adata_ref = ad.AnnData(X=mat.tocsr().T)
adata_ref.obs = meta
adata_ref.var = genename
adata_ref = adata_ref[adata_ref.obs.condition.values.astype(str)=="Control", :]

###Load spatial transcriptomics datasets:
#spatial data
anno_df = pd.read_csv('/home/zhaoshangrui/xuxinyu/datasets/DLPFC_data/10xGenomicsVisium/STitch3D/barcode_level_layer_map.tsv', sep='\t', header=None)

slice_idx = [151507,151508,151509,151510]

adata_st1 = sc.read_visium(path="/home/zhaoshangrui/xuxinyu/datasets/DLPFC_data/10xGenomicsVisium/spatialLIBD/%d" % slice_idx[0], count_file="filtered_feature_bc_matrix.h5")
anno_df1 = anno_df.iloc[anno_df[1].values.astype(str) == str(slice_idx[0])]
anno_df1.columns = ["barcode", "slice_id", "layer"]
anno_df1.index = anno_df1['barcode']
adata_st1.obs = adata_st1.obs.join(anno_df1, how="left")
# adata_st1 = adata_st1[adata_st1.obs['layer'].notna()]
adata_st1.obs['layer'].fillna('unknown', inplace=True)

adata_st2 = sc.read_visium(path="/home/zhaoshangrui/xuxinyu/datasets/DLPFC_data/10xGenomicsVisium/spatialLIBD/%d" % slice_idx[1], count_file="filtered_feature_bc_matrix.h5")
anno_df2 = anno_df.iloc[anno_df[1].values.astype(str) == str(slice_idx[1])]
anno_df2.columns = ["barcode", "slice_id", "layer"]
anno_df2.index = anno_df2['barcode']
adata_st2.obs = adata_st2.obs.join(anno_df2, how="left")
# adata_st2 = adata_st2[adata_st2.obs['layer'].notna()]
adata_st2.obs['layer'].fillna('unknown', inplace=True)

adata_st3 = sc.read_visium(path="/home/zhaoshangrui/xuxinyu/datasets/DLPFC_data/10xGenomicsVisium/spatialLIBD/%d" % slice_idx[2], count_file="filtered_feature_bc_matrix.h5")
anno_df3 = anno_df.iloc[anno_df[1].values.astype(str) == str(slice_idx[2])]
anno_df3.columns = ["barcode", "slice_id", "layer"]
anno_df3.index = anno_df3['barcode']
adata_st3.obs = adata_st3.obs.join(anno_df3, how="left")
# adata_st3 = adata_st3[adata_st3.obs['layer'].notna()]
adata_st3.obs['layer'].fillna('unknown', inplace=True)

adata_st4 = sc.read_visium(path="/home/zhaoshangrui/xuxinyu/datasets/DLPFC_data/10xGenomicsVisium/spatialLIBD/%d" % slice_idx[3], count_file="filtered_feature_bc_matrix.h5")
anno_df4 = anno_df.iloc[anno_df[1].values.astype(str) == str(slice_idx[3])]
anno_df4.columns = ["barcode", "slice_id", "layer"]
anno_df4.index = anno_df4['barcode']
adata_st4.obs = adata_st4.obs.join(anno_df4, how="left")
# adata_st4 = adata_st4[adata_st4.obs['layer'].notna()]
adata_st4.obs['layer'].fillna('unknown', inplace=True)

###Alignment of ST tissue slices
adata_st_list_raw = [adata_st1, adata_st2, adata_st3, adata_st4]
adata_st_list = STitch3D.utils.align_spots(adata_st_list_raw, plot=True)

celltype_list_use = ['Astros_1', 'Astros_2', 'Astros_3', 'Endo', 'Micro/Macro',
                     'Oligos_1', 'Oligos_2', 'Oligos_3',
                     'Ex_1_L5_6', 'Ex_2_L5', 'Ex_3_L4_5', 'Ex_4_L_6', 'Ex_5_L5',
                     'Ex_6_L4_6', 'Ex_7_L4_6', 'Ex_8_L5_6', 'Ex_9_L5_6', 'Ex_10_L2_4']

adata_st, adata_basis = STitch3D.utils.preprocess(adata_st_list,
                                                  adata_ref,
                                                  celltype_ref=celltype_list_use,
                                                  sample_col="group",
                                                  slice_dist_micron=[10., 300., 10.],
                                                  n_hvg_group=500)

model = STitch3D.model.Model(adata_st, adata_basis)

model.train()


save_path = "/home/zhaoshangrui/xuxinyu/datasets/DLPFC_data/10xGenomicsVisium/STitch3D/sample1_new"
result = model.eval(adata_st_list_raw, save=True, output_path=save_path)

###Visualizing results in 2D
from sklearn.mixture import GaussianMixture

np.random.seed(1234)
gm = GaussianMixture(n_components=7, covariance_type='tied', init_params='kmeans')
y = gm.fit_predict(model.adata_st.obsm['latent'], y=None)
model.adata_st.obs["GM"] = y
model.adata_st.obs["GM"].to_csv(os.path.join(save_path, "clustering_result.csv"))

#UMAP
import umap

reducer = umap.UMAP(n_neighbors=30,
                    n_components=2,
                    metric="correlation",
                    n_epochs=None,
                    learning_rate=1.0,
                    min_dist=0.3,
                    spread=1.0,
                    set_op_mix_ratio=1.0,
                    local_connectivity=1,
                    repulsion_strength=1,
                    negative_sample_rate=5,
                    a=None,
                    b=None,
                    random_state=1234,
                    metric_kwds=None,
                    angular_rp_forest=False,
                    verbose=True)

embedding = reducer.fit_transform(model.adata_st.obsm['latent'])

n_spots = embedding.shape[0]
size = 120000 / n_spots

model.adata_st.obsm["X_umap"] = embedding
sc.pp.neighbors(model.adata_st, use_rep='latent')
#sc.tl.paga(model.adata_st, groups='layer')

from sklearn import preprocessing
from matplotlib.colors import ListedColormap

le_slice = preprocessing.LabelEncoder()
label_slice = le_slice.fit_transform(model.adata_st.obs['slice_id'])

le_layer = preprocessing.LabelEncoder()
label_layer = le_layer.fit_transform(model.adata_st.obs['layer'])

np.random.seed(1234)
order = np.arange(n_spots)
np.random.shuffle(order)

f = plt.figure(figsize=(45,10))

ax1 = f.add_subplot(1,3,1)
scatter1 = ax1.scatter(embedding[order, 0], embedding[order, 1],
                       s=size, c=label_slice[order], cmap='coolwarm')
ax1.set_title("Slice", fontsize=40)
ax1.tick_params(axis='both',bottom=False, top=False, left=False, right=False, labelleft=False, labelbottom=False, grid_alpha=0)

l1 = ax1.legend(handles=scatter1.legend_elements()[0],
                labels=["Slice %d" % i for i in slice_idx],
                loc="upper left", bbox_to_anchor=(0., 0.),
                markerscale=3., title_fontsize=45, fontsize=30, frameon=False, ncol=1)
l1._legend_box.align = "left"


ax2 = f.add_subplot(1,3,2)
scatter2 = ax2.scatter(embedding[order, 0], embedding[order, 1],
                       s=size, c=model.adata_st.obs['Cluster'][order], cmap='cividis')
ax2.set_title("Cluster", fontsize=40)
ax2.tick_params(axis='both',bottom=False, top=False, left=False, right=False, labelleft=False, labelbottom=False, grid_alpha=0)

l2 = ax2.legend(handles=scatter2.legend_elements()[0],
                labels=["Cluster %d" % i for i in range(1, 8)],
                loc="upper left", bbox_to_anchor=(0., 0.),
                markerscale=3., title_fontsize=45, fontsize=30, frameon=False, ncol=2)

l2._legend_box.align = "left"

ax3 = f.add_subplot(1,3,3)
scatter3 = ax3.scatter(embedding[order, 0], embedding[order, 1],
                       s=size, c=label_layer[order], cmap=ListedColormap(["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]))
ax3.set_title("Layer annotation", fontsize=40)
ax3.tick_params(axis='both',bottom=False, top=False, left=False, right=False, labelleft=False, labelbottom=False, grid_alpha=0)

l3 = ax3.legend(handles=scatter3.legend_elements()[0],
                labels=sorted(set(adata_st.obs['layer'].values)),
                loc="upper left", bbox_to_anchor=(0., 0.),
                markerscale=3., title_fontsize=45, fontsize=30, frameon=False, ncol=2)

l3._legend_box.align = "left"

f.subplots_adjust(hspace=.1, wspace=.1)

plt.savefig("/home/zhaoshangrui/xuxinyu/STitch3D/DLPFC/metrice/umap_STitch3D_sample1.pdf")