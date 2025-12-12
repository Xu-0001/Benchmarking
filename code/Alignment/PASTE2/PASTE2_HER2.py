# -*- coding: utf-8 -*-
import time
import anndata as ad
import pandas as pd
import scanpy as sc
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import sys
sys.path.append("/home/zhaoshangrui/xuxinyu/PASTE_2")
from paste2.PASTE2 import partial_pairwise_align
from paste2.projection import partial_stack_slices_pairwise
from paste2.model_selection import select_overlap_fraction

#raw data
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
        sc.pp.highly_variable_genes(adata_st_i, flavor="seurat_v3", n_top_genes=5000)
        sc.pp.normalize_total(adata_st_i, target_sum=1e4)
        sc.pp.log1p(adata_st_i)
        adata_st_i = adata_st_i[:, adata_st_i.var['highly_variable']]
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
        patients[p][i] = adata_st_i 

# Compute partial pairwise alignment using PASTE2
alpha = 0.1
pis = {p:[0 for i in range(len(patients[p])-1)] for p in patients}
for p in patients:
    print('Patient ',p)
    start = time.time()
    for i in range(len(patients[p])-1):
        print("Slices ",i,i+1)
        s_pred = select_overlap_fraction(patients[p][i], patients[p][i+1], alpha=0.1)
        pis[p][i] = partial_pairwise_align(patients[p][i], patients[p][i+1], s=s_pred)
        np.savetxt('/home/zhaoshangrui/xuxinyu/PASTE_2/HER2/result/init_{0}_{1}_{2}.gz'.format(p,i,'ot'), pis[p][i], delimiter=',')
    print('Runtime: ' + str(time.time() - start))

patients_new_slices = {p:[0 for i in range(samples_per_patient[p])] for p in samples_per_patient}
for p in patients:
    patients_new_slices[p] = partial_stack_slices_pairwise(patients[p] , pis[p])    


paste2_layer_groups = partial_stack_slices_pairwise(patients['A'], pis['A']) 

sample_list = ['A1','A2','A3','A4','A5','A6'] 

slice_colors = sns.color_palette()

plt.figure(figsize=(5,5))
for i in range(len(sample_list)):
    colors = slice_colors[i]
    plt.scatter(paste2_layer_groups[i].obsm['spatial'][:,0],paste2_layer_groups[i].obsm['spatial'][:,1],linewidth=0,s=100, marker=".",color=colors)
plt.gca().invert_yaxis()
plt.axis('off') 
plt.savefig("/home/zhaoshangrui/xuxinyu/fig/HER2/PASTE2_alignment.pdf")
plt.close() 

