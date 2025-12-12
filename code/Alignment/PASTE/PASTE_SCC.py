# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from matplotlib import style
import matplotlib as mpl
import paste as pst
import scanpy as sc
import anndata

import os
style.use('seaborn-dark')
mpl.rc('xtick', labelsize=14) 
mpl.rc('ytick', labelsize=14) 

def intersect(lst1, lst2):
    """
    Gets and returns intersection of two lists.

    Args:
        lst1: List
        lst2: List

    Returns:
        lst3: List of common elements.
    """
    temp = set(lst2)
    lst3 = [value for value in lst1 if value in temp]
    return lst3

def load_layer(patient, sample, metadata):
    """
    Return Layer object of Patient, Sample
    """
    layer_path = f"D:/A/datasets/SCC/scc_data/scc_p{patient}_layer{sample}"
    layer = layer_path + ".tsv"
    coor_path = layer_path + "_coordinates.tsv"
    adata = anndata.read_csv(layer, delimiter="\t")
    # Data pre-processing
    coor = pd.read_csv(coor_path, sep="\t").iloc[:,:2]
    coor_index = []
    for pair in coor.values:
        coor_index.append('x'.join(str(e) for e in pair))
    coor.index = coor_index
    # The metadata, coordinates, and gene expression might have missing cells between them
    idx = intersect(coor_index, adata.obs.index)
    df = metadata[metadata['patient'] == patient]
    df = df[df['sample'] == sample]
    meta_idx = []
    for i in df.index:
        meta_idx.append(i.split('_')[1])
    idx = intersect(idx, meta_idx)
    adata = adata[idx, :]
    adata.obsm['spatial'] = np.array(coor.loc[idx, :])
    metadata_idx = ['P' + str(patient) + '_' + i + '_' + str(sample) for i in idx]
    adata.obs['original_clusters'] = [str(x) for x in list(metadata.loc[metadata_idx, 'SCT_snn_res.0.8'])]
    return adata

metadata_path =  "D:/A/datasets/SCC/scc_data/ST_all_metadata.txt"
metadata = pd.read_csv(metadata_path, sep="\t", index_col=0)

adata_2_1 = load_layer(2, 1, metadata)
adata_2_2 = load_layer(2, 2, metadata)
adata_2_3 = load_layer(2, 3, metadata)
patient_2 = [adata_2_1, adata_2_2, adata_2_3]

adata_5_1 = load_layer(5, 1, metadata)
adata_5_2 = load_layer(5, 2, metadata)
adata_5_3 = load_layer(5, 3, metadata)
patient_5 = [adata_5_1, adata_5_2, adata_5_3]

adata_9_1 = load_layer(9, 1, metadata)
adata_9_2 = load_layer(9, 2, metadata)
adata_9_3 = load_layer(9, 3, metadata)
patient_9 = [adata_9_1, adata_9_2, adata_9_3]

adata_10_1 = load_layer(10, 1, metadata)
adata_10_2 = load_layer(10, 2, metadata)
adata_10_3 = load_layer(10, 3, metadata)
patient_10 = [adata_10_1, adata_10_2, adata_10_3]

patients = {
    "patient_2" : patient_2,
    "patient_5" : patient_5,
    "patient_9" : patient_9,
    "patient_10" : patient_10,
}

for p in patients.values():
    for adata in p:
        sc.pp.filter_genes(adata, min_cells = 15, inplace = True)
        sc.pp.filter_cells(adata, min_genes = 100, inplace = True)

path_to_output_dir = "D:/A/datasets/SCC/scc_data/cached-results/"
H5ADs_dir = path_to_output_dir + 'H5ADs/'

if not os.path.exists(H5ADs_dir):
    os.makedirs(H5ADs_dir)
    
for k, p in patients.items():
    for i in range(len(p)):
        p[i].write(H5ADs_dir + k + '_slice_' + str(i) + '.h5ad')
        
# Pairwise Alignment Experiment 
path_to_output_dir = "D:/A/datasets/SCC/scc_data/cached-results/"       
path_to_h5ads = path_to_output_dir + 'H5ADs/'

patient_2 = []
patient_5 = []
patient_9 = []
patient_10 = []

patients = {
    "patient_2" : patient_2,
    "patient_5" : patient_5,
    "patient_9" : patient_9,
    "patient_10" : patient_10,
}

for k in patients.keys():
    for i in range(3):
        patients[k].append(sc.read_h5ad(path_to_h5ads + k + '_slice_' + str(i) + '.h5ad'))
    
import time        
start_time = time.time()
pis = {p:[0 for i in range(len(patients[k])-1)] for p in patients}        
for k in patients.keys():
    for i in range(2):
        pis[k][i] = pst.pairwise_align(patients[k][i], patients[k][i+1], alpha = 0.1)    
        np.savetxt('/home/zhaoshangrui/xuxinyu/datasets/SCC/cached-results/results/init_{0}_{1}_{2}.gz'.format(k,i,'ot'), pis[k][i], delimiter=',')
end_time = time.time()
elapsed_time = end_time - start_time
print(f"实验运行时间: {elapsed_time:.2f} 秒") 

# Center Alignment Experiment
patient_2 = []
patient_5 = []
patient_9 = []
patient_10 = []

patients = {
    "patient_2" : patient_2,
    "patient_5" : patient_5,
    "patient_9" : patient_9,
    "patient_10" : patient_10,
}

for k in patients.keys():
    for i in range(3):
        patients[k].append(sc.read_h5ad(path_to_h5ads + k + '_slice_' + str(i) + '.h5ad'))
        
        
path_to_center_dir = path_to_output_dir + 'center_nmfs/'
if not os.path.exists(path_to_center_dir):
    os.makedirs(path_to_center_dir)

start_time = time.time()
saved = []
for k, patient_n in patients.items():
    initial_slice = patient_n[0].copy()
    lmbda = len(patient_n)*[1/len(patient_n)]
    center_slice, pis = pst.center_align(initial_slice, patient_n, lmbda, random_seed = 5)
    center_slice.write(path_to_center_dir + k + '_center.h5ad')
    saved.append(center_slice)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"实验运行时间: {elapsed_time:.2f} 秒")

