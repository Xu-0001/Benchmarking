# -*- coding: utf-8 -*-
import torch
import numpy as np
import scanpy as sc
device = "cuda" if torch.cuda.is_available() else "cpu"

section_ids = ['Puck_200115_08','Puck_191204_01']
data_slice1 = sc.read_h5ad("/home/zhaoshangrui/xuxinyu/GPSA/mouse_hippocampus/result/data_slice1.h5ad")
data_slice2 = sc.read_h5ad("/home/zhaoshangrui/xuxinyu/GPSA/mouse_hippocampus/result/data_slice2.h5ad")
Batch_list = [data_slice1, data_slice2]

# Pairwise Alignment
import paste as pst
import time
start_time = time.time()
alpha = 0.1
pi = pst.pairwise_align(Batch_list[0], Batch_list[1], alpha=alpha)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"实验运行时间: {elapsed_time:.2f} 秒")
np.savetxt('/home/zhaoshangrui/xuxinyu/PASTE_1/mouse_hippocampus/result/init_ot.gz', pi, delimiter=',')

pis = [pi]
start_time = time.time()
new_slices = pst.stack_slices_pairwise(Batch_list , pis)   
end_time = time.time()
elapsed_time = end_time - start_time
print(f"实验运行时间: {elapsed_time:.2f} 秒") 

initial_pis_center = [pst.match_spots_using_spatial_heuristic(Batch_list[0].obsm['spatial'],Batch_list[i].obsm['spatial'],use_ot=True) for i in range(len(Batch_list))]

# Center intergartion
alpha=0.1
start_time = time.time()
lmbda = len(Batch_list)*[1/len(Batch_list)]
center_slices, center_pis = pst.center_align(Batch_list[0], Batch_list, lmbda, alpha=alpha, pis_init = initial_pis_center)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"实验运行时间: {elapsed_time:.2f} 秒") 
center_slices.write("/home/zhaoshangrui/xuxinyu/PASTE_1/mouse_hippocampus/result/center.h5ad")
