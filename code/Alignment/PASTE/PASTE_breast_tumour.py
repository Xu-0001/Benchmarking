# -*- coding: utf-8 -*-
import time
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt

import paste as pst

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

slice_colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3']

fig, axs = plt.subplots(2, 2,figsize=(7,7))
pst.plot_slice(slice1,slice_colors[0],ax=axs[0,0])
pst.plot_slice(slice2,slice_colors[1],ax=axs[0,1])
pst.plot_slice(slice3,slice_colors[2],ax=axs[1,0])
pst.plot_slice(slice4,slice_colors[3],ax=axs[1,1])
plt.savefig("/home/zhaoshangrui/xuxinyu/fig/breast_tumour/original.pdf")
plt.close()

## Pairwise Alignment
start = time.time()

pi12 = pst.pairwise_align(slice1, slice2)
pi23 = pst.pairwise_align(slice2, slice3)
pi34 = pst.pairwise_align(slice3, slice4)

print('Runtime: ' + str(time.time() - start))

pis = [pi12, pi23, pi34]
slices = [slice1, slice2, slice3, slice4]

paste_layer_groups = pst.stack_slices_pairwise(slices, pis)

plt.figure(figsize=(5,5))
for i in range(len(slice_names)):
    colors = slice_colors[i]
    plt.scatter(paste_layer_groups[i].obsm['spatial'][:,0],paste_layer_groups[i].obsm['spatial'][:,1],linewidth=0,s=100, marker=".",color=colors)
plt.gca().invert_yaxis()
plt.axis('off') 
plt.savefig("/home/zhaoshangrui/xuxinyu/fig/breast_tumour/PASTE_alignment.pdf")
plt.close()

## Center Alignment
slices = load_slices(data_dir, slice_names)
slice1, slice2, slice3, slice4 = slices
slices = [slice1, slice2, slice3, slice4]
initial_slice = slice1.copy()
lmbda = len(slices)*[1/len(slices)]
pst.filter_for_common_genes(slices)

b = []
for i in range(len(slices)):
    b.append(pst.match_spots_using_spatial_heuristic(slices[0].X, slices[i].X))
    
start = time.time()

center_slice, pis = pst.center_align(initial_slice, slices, lmbda, random_seed = 5, pis_init = b)
print('Runtime: ' + str(time.time() - start))



