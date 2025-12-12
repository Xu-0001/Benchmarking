# -*- coding: utf-8 -*-
import numpy as np
import scanpy as sc
import paste as pst

#spatial data
adata_st_raw = sc.read_h5ad("/home/zhaoshangrui/xuxinyu/datasets/Drosophila_embryo_dataset_profiled_by_Stereo-seq/E16-18h_a_count_normal_stereoseq.h5ad")
adata_st_raw.X = adata_st_raw.layers['raw_counts']
adata_st_raw.obsm['spatial'] = adata_st_raw.obsm['spatial'][:, :2]

slice_all = sorted(list(set(adata_st_raw.obs['slice_ID'].values)))[:-1]

slices_ID = ['E16-18h_a_S01','E16-18h_a_S02','E16-18h_a_S03','E16-18h_a_S04','E16-18h_a_S05','E16-18h_a_S06','E16-18h_a_S07','E16-18h_a_S08',
 'E16-18h_a_S09','E16-18h_a_S10','E16-18h_a_S11','E16-18h_a_S12','E16-18h_a_S13']

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

# Pairwise Alignment  
import time        
start_time = time.time()  
pis = []
for i in range(len(slices_ID)-1):
    pi = pst.pairwise_align(adata_st_list_raw[i], adata_st_list_raw[i+1])
    pis.append(pi)
    np.savetxt('/home/zhaoshangrui/xuxinyu/PASTE_1/Drosophila_embryo_E16-18/result/init_{0}_{1}.gz'.format(i,'ot'), pis[i], delimiter=',')
end_time = time.time()
elapsed_time = end_time - start_time
print(f"实验运行时间: {elapsed_time:.2f} 秒")
# pis = []
# for j in range(len(slices_ID)-1):
#     pi=np.loadtxt('/home/zhaoshangrui/xuxinyu/PASTE_1/Drosophila_embryo_E16-18/result/init_{0}_{1}.gz'.format(j,'ot'), delimiter=',')
#     pis.append(pi)

start_time = time.time()  
paste_layer_groups = pst.stack_slices_pairwise(adata_st_list_raw, pis) 
end_time = time.time()
elapsed_time = end_time - start_time
print(f"实验运行时间: {elapsed_time:.2f} 秒")      

# Center Alignment
start_time = time.time()
initial_slice = adata_st_list_raw[0].copy()
initial_pis_center = [pst.match_spots_using_spatial_heuristic(adata_st_list_raw[0].obsm['spatial'],adata_st_list_raw[i].obsm['spatial'],use_ot=True) for i in range(len(adata_st_list_raw))]
lmbda = len(slice_all)*[1/len(slice_all)]
center_slice, pis = pst.center_align(initial_slice, adata_st_list_raw, lmbda, pis_init = initial_pis_center, random_seed = 5)#, pis_init = b)
center_slice.write('/home/zhaoshangrui/xuxinyu/PASTE_1/Drosophila_embryo_E16-18/result/E16-18h_center.h5ad')
end_time = time.time()
elapsed_time = end_time - start_time
print(f"实验运行时间: {elapsed_time:.2f} 秒") 

