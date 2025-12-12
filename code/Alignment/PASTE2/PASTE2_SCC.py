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

path_to_output_dir = "/home/zhaoshangrui/xuxinyu/datasets/SCC/cached-results/"       
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
        s_pred = select_overlap_fraction(patients[k][i], patients[k][i+1], alpha=0.1)
        pis[k][i] = partial_pairwise_align(patients[k][i], patients[k][i+1], alpha = 0.1,s=s_pred)    
        np.savetxt('/home/zhaoshangrui/xuxinyu/PASTE_2/SCC/result/init_{0}_{1}_{2}.gz'.format(k,i,'ot'), pis[k][i], delimiter=',')
end_time = time.time()
elapsed_time = end_time - start_time
print(f"实验运行时间: {elapsed_time:.2f} 秒")      

# path = "/home/zhaoshangrui/xuxinyu/PASTE_2/SCC/result/"
# pis = {p:[0 for i in range(len(patients[k])-1)] for p in patients}   
# for k in patients.keys():
#     for i in range(2):
#         file_path = '{}/init_{}_{}_{}.gz'.format(path, k, i, 'ot')
#         pis[k][i] = np.loadtxt(file_path, delimiter=',') 
