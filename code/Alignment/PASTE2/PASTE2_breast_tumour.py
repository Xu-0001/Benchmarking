# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")

import scanpy as sc
import numpy as np

import matplotlib.pyplot as plt

import sys
sys.path.append("/home/zhaoshangrui/xuxinyu/PASTE_2")
from paste2.PASTE2 import partial_pairwise_align
from paste2.projection import partial_stack_slices_pairwise
from paste2.model_selection import select_overlap_fraction

data_dir = "/home/zhaoshangrui/xuxinyu/datasets/breast_tumour_data/breast_tumour_ST_data/"

def load_slices(data_dir, slice_names):
    slices = []  
    for slice_name in slice_names:
        slice_i = sc.read_csv(data_dir + slice_name + ".csv")
        slice_i_coor = np.genfromtxt(data_dir + slice_name + "_coor.csv", delimiter = ',')
        slice_i.obsm['spatial'] = slice_i_coor
        # Preprocess slices
        sc.pp.filter_genes(slice_i, min_counts = 15)
        sc.pp.filter_cells(slice_i, min_counts = 100)
        sc.pp.normalize_total(slice_i, inplace=True)
        sc.pp.log1p(slice_i)
        sc.pp.highly_variable_genes(
            slice_i, flavor="seurat", n_top_genes=3000, subset=True
        )
        slices.append(slice_i)
    return slices
slice_names =["stahl_bc_slice1", "stahl_bc_slice2", "stahl_bc_slice3", "stahl_bc_slice4"]
slices = load_slices(data_dir, slice_names)
slice1, slice2, slice3, slice4 = slices

import time
start = time.time()
alpha=0.1
s_pred = select_overlap_fraction(slice1, slice2, alpha=0.1)
pi12 = partial_pairwise_align(slice1, slice2, alpha=alpha, s=s_pred)
s_pred = select_overlap_fraction(slice2, slice3, alpha=0.1)
pi23 = partial_pairwise_align(slice2, slice3, alpha=alpha, s=s_pred)
s_pred = select_overlap_fraction(slice3, slice4, alpha=0.1)
pi34 = partial_pairwise_align(slice3, slice4, alpha=alpha, s=s_pred)
print('Runtime: ' + str(time.time() - start))

pis = [pi12, pi23, pi34]
slices = [slice1, slice2, slice3, slice4]

paste2_layer_groups = partial_stack_slices_pairwise(slices, pis)  

slice_colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3']

plt.figure(figsize=(5,5))
for i in range(len(slice_names)):
    colors = slice_colors[i]
    plt.scatter(paste2_layer_groups[i].obsm['spatial'][:,0],paste2_layer_groups[i].obsm['spatial'][:,1],linewidth=0,s=100, marker=".",color=colors)
plt.gca().invert_yaxis()
plt.axis('off') 
plt.savefig("/home/zhaoshangrui/xuxinyu/fig/breast_tumour/PASTE2_alignment_new.pdf")
plt.close()  

