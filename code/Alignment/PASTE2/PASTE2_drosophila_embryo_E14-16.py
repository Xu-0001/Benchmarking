# -*- coding: utf-8 -*-
import numpy as np
import scanpy as sc

import sys
sys.path.append("/home/zhaoshangrui/xuxinyu/PASTE_2")
from paste2.PASTE2 import partial_pairwise_align
from paste2.projection import partial_stack_slices_pairwise
from paste2.model_selection import select_overlap_fraction

#spatial data
adata_st_raw = sc.read_h5ad("/home/zhaoshangrui/xuxinyu/datasets/Drosophila_embryo_dataset_profiled_by_Stereo-seq/E14-16h_a_count_normal_stereoseq.h5ad")
adata_st_raw.X = adata_st_raw.layers['raw_counts']
adata_st_raw.obsm['spatial'] = adata_st_raw.obsm['spatial'][:, :2]

slice_all = sorted(list(set(adata_st_raw.obs['slice_ID'].values)))

slices_ID = ['E14-16h_a_S01','E14-16h_a_S02','E14-16h_a_S03','E14-16h_a_S04','E14-16h_a_S05','E14-16h_a_S06','E14-16h_a_S07','E14-16h_a_S08',
 'E14-16h_a_S09','E14-16h_a_S10','E14-16h_a_S11','E14-16h_a_S12','E14-16h_a_S13','E14-16h_a_S14','E14-16h_a_S15','E14-16h_a_S16']  

adata_st_list_raw = []
for slice_id in slice_all:
    adata_st_i = adata_st_raw[adata_st_raw.obs['slice_ID'].values == slice_id]
    adata_st_i.obsm['spatial'] = np.concatenate((adata_st_i.obs['raw_x'].values.reshape(-1, 1),
                                                 adata_st_i.obs['raw_y'].values.reshape(-1, 1)), axis=1) / 20
    adata_st_i.obsm['loc_use'] = np.concatenate((adata_st_i.obs['raw_x'].values.reshape(-1, 1),
                                                 adata_st_i.obs['raw_y'].values.reshape(-1, 1)), axis=1) / 20
    adata_st_i.obsm['coor_3d'] = np.concatenate((adata_st_i.obs['new_x'].values.reshape(-1, 1),
                                                 adata_st_i.obs['new_y'].values.reshape(-1, 1),
                                                 adata_st_i.obs['new_z'].values.reshape(-1, 1)), axis=1)
    adata_st_i.obs['array_row'] = adata_st_i.obs['raw_y'].values
    adata_st_i.obs['array_col'] = adata_st_i.obs['raw_x'].values
    adata_st_list_raw.append(adata_st_i.copy())

# Compute partial pairwise alignment using PASTE2
import time        
start_time = time.time()  
pis = []
for i in range(len(slices_ID)-1):
    s_pred = select_overlap_fraction(adata_st_list_raw[i], adata_st_list_raw[i+1], alpha=0.1)
    pi = partial_pairwise_align(adata_st_list_raw[i], adata_st_list_raw[i+1], s=s_pred)
    pis.append(pi)
    np.savetxt('/home/zhaoshangrui/xuxinyu/PASTE_2/Drosophila_embryo_E14-16/result/init_{0}_{1}.gz'.format(i,'ot'), pis[i], delimiter=',')
end_time = time.time()
elapsed_time = end_time - start_time
print(f"实验运行时间: {elapsed_time:.2f} 秒")
# pis = []
# for j in range(len(slices_ID)-1):
#     pi=np.loadtxt('/home/zhaoshangrui/xuxinyu/PASTE_2/Drosophila_embryo_E14-16/result/init_{0}_{1}.gz'.format(j,'ot'), delimiter=',')
#     pis.append(pi)

start_time = time.time()  
paste_layer_groups = partial_stack_slices_pairwise(adata_st_list_raw, pis) 
end_time = time.time()
elapsed_time = end_time - start_time
print(f"实验运行时间: {elapsed_time:.2f} 秒")      
