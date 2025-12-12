# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")

import scanpy as sc
import numpy as np

import torch
used_device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

import sys
sys.path.append("/home/zhaoshangrui/xuxinyu/PASTE_2")
from paste2.PASTE2 import partial_pairwise_align
from paste2.projection import partial_stack_slices_pairwise
from paste2.model_selection import select_overlap_fraction

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
data_slice1 = process_data(data_slice1, n_top_genes=6000)

data_slice2 = sc.read_visium("/home/zhaoshangrui/xuxinyu/datasets/mouse_brain_data/Sagittal_mouse_brain_data/GPSA_mouse_brain_serial_section_2/")
data_slice2 = process_data(data_slice2, n_top_genes=6000)

Batch_list = [data_slice1, data_slice2]

### Pairwise Alignment
import time
start_time = time.time()
pis = []
alpha = 0.1
s_pred = select_overlap_fraction(data_slice1, data_slice2, alpha=0.1)
pi = partial_pairwise_align(data_slice1, data_slice2, alpha=alpha, s=s_pred)
np.savetxt('/home/zhaoshangrui/xuxinyu/PASTE_2/mouse_brain/result/init.gz', pi, delimiter=',')
pis.append(pi)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"实验运行时间: {elapsed_time:.2f} 秒")

start_time = time.time()
new_slices = partial_stack_slices_pairwise(Batch_list , pis)   
end_time = time.time()
elapsed_time = end_time - start_time
print(f"实验运行时间: {elapsed_time:.2f} 秒")

