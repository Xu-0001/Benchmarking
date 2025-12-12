# -*- coding: utf-8 -*-
import time
import anndata as ad
import pandas as pd
import scanpy as sc
import numpy as np

import paste as pst
import seaborn as sns
import matplotlib.pyplot as plt

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

# Pairwise Alignment
        
initial_pis = {p:[pst.match_spots_using_spatial_heuristic(patients[p][i].obsm['spatial'],patients[p][i+1].obsm['spatial']) for i in range(len(patients[p])-1)] for p in patients}

alpha = 0.1
pis = {p:[0 for i in range(len(patients[p])-1)] for p in patients}
for p in patients:
    print('Patient ',p)
    start = time.time()
    for i in range(len(patients[p])-1):
        print("Slices ",i,i+1)
        pis[p][i] = pst.pairwise_align(patients[p][i], patients[p][i+1],alpha=alpha, G_init=initial_pis[p][i])
        np.savetxt('/home/zhaoshangrui/xuxinyu/PASTE_1/HER2/result/init_{0}_{1}_{2}.gz'.format(p,i,'ot'), pis[p][i], delimiter=',')
    print('Runtime: ' + str(time.time() - start))
# path = '/home/zhaoshangrui/xuxinyu/PASTE_1/HER2/result'
# for p in patients:
#     for i in range(len(patients[p]) - 1):
#         file_path = '{}/init_{}_{}_{}.gz'.format(path, p, i, 'ot')
#         pis[p][i] = np.loadtxt(file_path, delimiter=',')
    
paste_layer_groups = pst.stack_slices_pairwise(patients['A'], pis['A'])

sample_list = ['A1','A2','A3','A4','A5','A6'] 

slice_colors = sns.color_palette()

plt.figure(figsize=(5,5))
for i in range(len(sample_list)):
    colors = slice_colors[i]
    plt.scatter(paste_layer_groups[i].obsm['spatial'][:,0],paste_layer_groups[i].obsm['spatial'][:,1],linewidth=0,s=100, marker=".",color=colors)
plt.gca().invert_yaxis()
plt.axis('off') 
plt.savefig("/home/zhaoshangrui/xuxinyu/fig/HER2/PASTE_alignment.pdf")
plt.close()

# Center intergartion
initial_slice = {p:0 for p in patients}
initial_slice['G']=1
center_slices = {p:0 for p in patients}
center_pis = {p:0 for p in patients}
initial_pis_center = {p:[pst.match_spots_using_spatial_heuristic(patients[p][initial_slice[p]].obsm['spatial'],patients[p][i].obsm['spatial'],use_ot=True) for i in range(len(patients[p]))] for p in patients}
alpha=0.1
for p in patients:
    print('Patient',p)
    start = time.time()
    center_slices[p], center_pis[p] = pst.center_align(patients[p][initial_slice[p]], patients[p], alpha=alpha ,random_seed=0, pis_init = initial_pis_center[p], threshold = 1e-5)
    end = time.time()
    print('time',end-start)
    center_slices[p].write("/home/zhaoshangrui/xuxinyu/PASTE_1/HER2/result/center_patient_{0}.h5ad".format(p))