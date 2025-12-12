# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")

import scanpy as sc
import numpy as np

import torch
used_device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

def process_data(adata, n_top_genes=2000):
    adata.var_names_make_unique()
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
    sc.pp.filter_cells(adata, min_counts=5000)
    sc.pp.filter_cells(adata, max_counts=35000)
    # adata = adata[adata.obs["pct_counts_mt"] < 20]
    sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata, flavor="seurat", n_top_genes=n_top_genes, subset=True
    )
    return adata

data_slice1 = sc.read_visium("/home/zhaoshangrui/xuxinyu/datasets/mouse_brain_data/Sagittal_mouse_brain_data/GPSA_mouse_brain_serial_section_1/")
data_slice2 = sc.read_visium("/home/zhaoshangrui/xuxinyu/datasets/mouse_brain_data/Sagittal_mouse_brain_data/GPSA_mouse_brain_serial_section_2/")

data_slice1 = process_data(data_slice1, n_top_genes=6000)
data_slice2 = process_data(data_slice2, n_top_genes=6000)

Batch_list = [data_slice1, data_slice2]

### Pairwise Alignment
import paste as pst
import time
start_time = time.time()
pis = []
alpha = 0.1
pi = pst.pairwise_align(data_slice1, data_slice2, alpha=alpha)
np.savetxt('/home/zhaoshangrui/xuxinyu/PASTE_1/mouse_brain/result/init.gz', pi, delimiter=',')
pis.append(pi)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"实验运行时间: {elapsed_time:.2f} 秒")

start_time = time.time()
new_slices = pst.stack_slices_pairwise(Batch_list , pis)   
end_time = time.time()
elapsed_time = end_time - start_time
print(f"实验运行时间: {elapsed_time:.2f} 秒")

### Center Alignment
initial_slice = data_slice1.copy()
lmbda = len(Batch_list)*[1/len(Batch_list)]
pst.filter_for_common_genes(Batch_list)

b = []
for i in range(len(Batch_list)):
    b.append(pst.match_spots_using_spatial_heuristic(Batch_list[0].X.toarray(), Batch_list[i].X.toarray()))
    
start = time.time()
center_slice, pis = pst.center_align(initial_slice, Batch_list, lmbda, random_seed = 5, pis_init = b)
print('Runtime: ' + str(time.time() - start))
