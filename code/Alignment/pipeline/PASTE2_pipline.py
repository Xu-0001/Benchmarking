# -*- coding: utf-8 -*-
import torch
import numpy as np
import scanpy as sc

import sys
sys.path.append(".../PASTE_2")
from paste2.PASTE2 import partial_pairwise_align
from paste2.projection import partial_stack_slices_pairwise
from paste2.model_selection import select_overlap_fraction

device = "cuda" if torch.cuda.is_available() else "cpu"

section_ids = ['data_slice1','data_slice2']
data_slice1 = sc.read_h5ad(".../data_slice1.h5ad")
data_slice2 = sc.read_h5ad(".../data_slice2.h5ad")
Batch_list = [data_slice1, data_slice2]

# Compute partial pairwise alignment using PASTE2
import time
alpha = 0.1
start_time = time.time()
s_pred = select_overlap_fraction(Batch_list[0], Batch_list[1], alpha=0.1)
pi = partial_pairwise_align(Batch_list[0], Batch_list[1], alpha=alpha, s=s_pred)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"实验运行时间: {elapsed_time:.2f} 秒")
np.savetxt('.../init_ot.gz', pi, delimiter=',')

pis = [pi]
start_time = time.time()
new_slices = partial_stack_slices_pairwise(Batch_list , pis)    
end_time = time.time()
elapsed_time = end_time - start_time
print(f"实验运行时间: {elapsed_time:.2f} 秒")
