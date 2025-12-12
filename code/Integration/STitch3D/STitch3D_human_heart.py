# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
from matplotlib.image import imread
import os

import STitch3D

import warnings
warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

##Load spatial transcriptomics datasets:
count = pd.read_csv("/home/zhaoshangrui/xuxinyu/datasets/human_embryonic_heart_dataset/10x_Genomics_Chromium_platform/Filtered/filtered_ST_matrix_and_meta_data/filtered_matrix.tsv",
                    sep="\t", index_col=0).T

meta = pd.read_csv("/home/zhaoshangrui/xuxinyu/datasets/human_embryonic_heart_dataset/10x_Genomics_Chromium_platform/Filtered/filtered_ST_matrix_and_meta_data/meta_data.tsv",
                   sep="\t", index_col=0)

genes_new = pd.read_csv("/home/zhaoshangrui/xuxinyu/datasets/human_embryonic_heart_dataset/genes_new.txt")
count.columns = list(genes_new.x.values)

count = count.loc[:, count.columns.notna()]

adata_st_list_raw = []

for i in range(1, 10):
    count_i = count[[loc.split("x")[0]==str(i+4) for loc in count.index]]
    count_i.index = [(loc.split("x")[1]+"x"+loc.split("x")[2]) for loc in count_i.index]
    meta_i = meta[[loc.split("x")[0]==str(i+4) for loc in meta.index]]
    meta_i.index = [(loc.split("x")[1]+"x"+loc.split("x")[2]) for loc in meta_i.index]

    loc_i = pd.read_csv("/home/zhaoshangrui/xuxinyu/datasets/human_embryonic_heart_dataset/ST platform/ST_Samples_6.5PCW/ST_Sample_6.5PCW_%d/spot_data-all-ST_Sample_6.5PCW_%d.tsv" % (i, i),
                       sep="\t")

    loc_i.index = [(str(loc_i.x.values[k]) + 'x' + str(loc_i.y.values[k])) for k in range(loc_i.shape[0])]
    loc_i = loc_i.loc[meta_i.index]

    img_i = imread('/home/zhaoshangrui/xuxinyu/datasets/human_embryonic_heart_dataset/ST platform/ST_Samples_6.5PCW/ST_Sample_6.5PCW_%d/ST_Sample_6.5PCW_%d_HE_small.jpg' % (i, i))

    adata_st_i = ad.AnnData(X = count_i.values)

    adata_st_i.obs = meta_i
    adata_st_i.obs['selected'] = loc_i['selected'].values
    adata_st_i.var.index = count_i.columns

    library_id = '0'
    adata_st_i.uns["spatial"] = dict()
    adata_st_i.uns["spatial"]['0'] = dict()
    adata_st_i.uns["spatial"]['0']['images'] = dict()
    adata_st_i.uns["spatial"]['0']['images']['hires'] = img_i
    adata_st_i.uns["spatial"]['0']['scalefactors'] = {'spot_diameter_fullres': 100,
                                                      'tissue_hires_scalef': 1.0,
                                                      'fiducial_diameter_fullres': 100,
                                                      'tissue_lowres_scalef': 1.0}

    adata_st_i.obsm['spatial'] = np.concatenate((loc_i['pixel_x'].values.reshape(-1, 1),
                                                 loc_i['pixel_y'].values.reshape(-1, 1)), axis=1)

    adata_st_i.obsm['loc_use'] = np.concatenate((loc_i['x'].values.reshape(-1, 1),
                                                 loc_i['y'].values.reshape(-1, 1)), axis=1)

    adata_st_i.obs['array_row'] = loc_i['y'].values
    adata_st_i.obs['array_col'] = loc_i['x'].values

    adata_st_i = adata_st_i[adata_st_i.obs['selected'].values != 0]
    adata_st_list_raw.append(adata_st_i.copy())
    
##Load single-cell reference dataset:
count_ref = pd.read_csv("/home/zhaoshangrui/xuxinyu/datasets/human_embryonic_heart_dataset/10x_Genomics_Chromium_platform/Filtered/share_files/all_cells_count_matrix_filtered.tsv",
                        sep='\t', index_col=0).T

meta_ref = pd.read_csv("/home/zhaoshangrui/xuxinyu/datasets/human_embryonic_heart_dataset/10x_Genomics_Chromium_platform/Filtered/share_files/all_cells_meta_data_filtered.tsv",
                        sep='\t', index_col=0)

adata_ref = ad.AnnData(X = count_ref.values)
adata_ref.obs.index = count_ref.index
adata_ref.var.index = count_ref.columns
for col in meta_ref.columns[:-1]:
    adata_ref.obs[col] = meta_ref.loc[count_ref.index][col].values
    
adata_st_list = STitch3D.utils.align_spots(adata_st_list_raw, data_type = "ST", coor_key="loc_use", test_all_angles=True, plot=True)

adata_st, adata_basis = STitch3D.utils.preprocess(adata_st_list,
                                                  adata_ref,
                                                  sample_col="experiment",
                                                  rad_cutoff=1.1, c2c_dist=200., coor_key="loc_use",
                                                  slice_dist_micron=[5., 115., 85., 160.,
                                                                     5., 160., 5., 155.,],
                                                  n_hvg_group=500)
adata_st.write("/home/zhaoshangrui/xuxinyu/STitch3D/human_heart/adata_st.h5ad")

model = STitch3D.model.Model(adata_st, adata_basis)

model.train()
    
save_path = "/home/zhaoshangrui/xuxinyu/STitch3D/human_heart/result"

result = model.eval(adata_st_list_raw, save=True, output_path=save_path)

for i, adata_st_i in enumerate(result):
    print("Slice %d" % (i+1))
    sc.pl.spatial(adata_st_i, img_key="hires", color=model.celltypes, size=1.)
    
sc.pp.neighbors(model.adata_st, use_rep='latent', n_neighbors=30)
sc.tl.louvain(model.adata_st, resolution=0.5)

model.adata_st.obs["louvain"].to_csv(os.path.join(save_path, "clustering_result.csv"))

for i, adata_st_i in enumerate(result):
    adata_st_i.obs["louvain"] = model.adata_st.obs.loc[adata_st_i.obs_names, ]["louvain"]
    
color = ['#7570b3', '#1b9e77', '#d95f02', '#66a61e', '#e7298a']

plt.rcParams["figure.figsize"] = (4, 4)
for i, adata_st_i in enumerate(result):
    sc.pl.spatial(result[i], img_key="hires", color=["louvain"], palette=color, size=1.5)
    
