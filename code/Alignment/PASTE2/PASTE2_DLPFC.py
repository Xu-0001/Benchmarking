# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import scanpy as sc

import sys
sys.path.append("/home/zhaoshangrui/xuxinyu/PASTE_2")
from paste2.PASTE2 import partial_pairwise_align
from paste2.model_selection import select_overlap_fraction


dirs="/home/zhaoshangrui/xuxinyu/SPIRAL/DLPFC/results/"
samples=np.array(['151507','151508','151509','151510','151669','151670','151671','151672','151673','151674','151675','151676'])

flags1=samples[0]
for i in range(1,len(samples)):
    flags1=flags1+'-'+samples[i] 
feat1=sc.read_csv(dirs+"gtt_input_scanpy/"+flags1+'_'+str(samples[0])+"_features-1.txt")
meta1=pd.read_csv(dirs+"gtt_input_scanpy/"+flags1+'_'+str(samples[0])+"_label-1.txt",header=0,index_col=0,sep=',')
coord1=pd.read_csv(dirs+"gtt_input_scanpy/"+flags1+'_'+str(samples[0])+"_positions-1.txt",header=0,index_col=0,sep=',')
coord1=coord1.loc[:,['x','y']]
feat1.obsm['spatial'] = coord1.values
feat1.obs['celltype']=meta1.iloc[:,0].values
feat1.obs.loc[feat1.obs['celltype'].isna(), 'celltype'] = "unknown"

feat2=sc.read_csv(dirs+"gtt_input_scanpy/"+flags1+'_'+str(samples[1])+"_features-1.txt")
meta2=pd.read_csv(dirs+"gtt_input_scanpy/"+flags1+'_'+str(samples[1])+"_label-1.txt",header=0,index_col=0,sep=',')
coord2=pd.read_csv(dirs+"gtt_input_scanpy/"+flags1+'_'+str(samples[1])+"_positions-1.txt",header=0,index_col=0,sep=',')
coord2=coord2.loc[:,['x','y']]
feat2.obsm['spatial'] = coord2.values
feat2.obs['celltype']=meta2.iloc[:,0].values
feat2.obs.loc[feat2.obs['celltype'].isna(), 'celltype'] = "unknown"

feat3=sc.read_csv(dirs+"gtt_input_scanpy/"+flags1+'_'+str(samples[2])+"_features-1.txt")
meta3=pd.read_csv(dirs+"gtt_input_scanpy/"+flags1+'_'+str(samples[2])+"_label-1.txt",header=0,index_col=0,sep=',')
coord3=pd.read_csv(dirs+"gtt_input_scanpy/"+flags1+'_'+str(samples[2])+"_positions-1.txt",header=0,index_col=0,sep=',')
coord3=coord3.loc[:,['x','y']]
feat3.obsm['spatial'] = coord3.values
feat3.obs['celltype']=meta3.iloc[:,0].values
feat3.obs.loc[feat3.obs['celltype'].isna(), 'celltype'] = "unknown"

feat4=sc.read_csv(dirs+"gtt_input_scanpy/"+flags1+'_'+str(samples[3])+"_features-1.txt")
meta4=pd.read_csv(dirs+"gtt_input_scanpy/"+flags1+'_'+str(samples[3])+"_label-1.txt",header=0,index_col=0,sep=',')
coord4=pd.read_csv(dirs+"gtt_input_scanpy/"+flags1+'_'+str(samples[3])+"_positions-1.txt",header=0,index_col=0,sep=',')
coord4=coord4.loc[:,['x','y']]
feat4.obsm['spatial'] = coord4.values
feat4.obs['celltype']=meta4.iloc[:,0].values
feat4.obs.loc[feat4.obs['celltype'].isna(), 'celltype'] = "unknown"

feat5=sc.read_csv(dirs+"gtt_input_scanpy/"+flags1+'_'+str(samples[4])+"_features-1.txt")
meta5=pd.read_csv(dirs+"gtt_input_scanpy/"+flags1+'_'+str(samples[4])+"_label-1.txt",header=0,index_col=0,sep=',')
coord5=pd.read_csv(dirs+"gtt_input_scanpy/"+flags1+'_'+str(samples[4])+"_positions-1.txt",header=0,index_col=0,sep=',')
coord5=coord5.loc[:,['x','y']]
feat5.obsm['spatial'] = coord5.values
feat5.obs['celltype']=meta5.iloc[:,0].values
feat5.obs.loc[feat5.obs['celltype'].isna(), 'celltype'] = "unknown"

feat6=sc.read_csv(dirs+"gtt_input_scanpy/"+flags1+'_'+str(samples[5])+"_features-1.txt")
meta6=pd.read_csv(dirs+"gtt_input_scanpy/"+flags1+'_'+str(samples[5])+"_label-1.txt",header=0,index_col=0,sep=',')
coord6=pd.read_csv(dirs+"gtt_input_scanpy/"+flags1+'_'+str(samples[5])+"_positions-1.txt",header=0,index_col=0,sep=',')
coord6=coord6.loc[:,['x','y']]
feat6.obsm['spatial'] = coord6.values
feat6.obs['celltype']=meta6.iloc[:,0].values
feat6.obs.loc[feat6.obs['celltype'].isna(), 'celltype'] = "unknown"

feat7=sc.read_csv(dirs+"gtt_input_scanpy/"+flags1+'_'+str(samples[6])+"_features-1.txt")
meta7=pd.read_csv(dirs+"gtt_input_scanpy/"+flags1+'_'+str(samples[6])+"_label-1.txt",header=0,index_col=0,sep=',')
coord7=pd.read_csv(dirs+"gtt_input_scanpy/"+flags1+'_'+str(samples[6])+"_positions-1.txt",header=0,index_col=0,sep=',')
coord7=coord7.loc[:,['x','y']]
feat7.obsm['spatial'] = coord7.values
feat7.obs['celltype']=meta7.iloc[:,0].values
feat7.obs.loc[feat7.obs['celltype'].isna(), 'celltype'] = "unknown"

feat8=sc.read_csv(dirs+"gtt_input_scanpy/"+flags1+'_'+str(samples[7])+"_features-1.txt")
meta8=pd.read_csv(dirs+"gtt_input_scanpy/"+flags1+'_'+str(samples[7])+"_label-1.txt",header=0,index_col=0,sep=',')
coord8=pd.read_csv(dirs+"gtt_input_scanpy/"+flags1+'_'+str(samples[7])+"_positions-1.txt",header=0,index_col=0,sep=',')
coord8=coord8.loc[:,['x','y']]
feat8.obsm['spatial'] = coord8.values
feat8.obs['celltype']=meta8.iloc[:,0].values
feat8.obs.loc[feat8.obs['celltype'].isna(), 'celltype'] = "unknown"

feat9=sc.read_csv(dirs+"gtt_input_scanpy/"+flags1+'_'+str(samples[8])+"_features-1.txt")
meta9=pd.read_csv(dirs+"gtt_input_scanpy/"+flags1+'_'+str(samples[8])+"_label-1.txt",header=0,index_col=0,sep=',')
coord9=pd.read_csv(dirs+"gtt_input_scanpy/"+flags1+'_'+str(samples[8])+"_positions-1.txt",header=0,index_col=0,sep=',')
coord9=coord9.loc[:,['x','y']]
feat9.obsm['spatial'] = coord9.values
feat9.obs['celltype']=meta9.iloc[:,0].values
feat9.obs.loc[feat9.obs['celltype'].isna(), 'celltype'] = "unknown"

feat10=sc.read_csv(dirs+"gtt_input_scanpy/"+flags1+'_'+str(samples[9])+"_features-1.txt")
meta10=pd.read_csv(dirs+"gtt_input_scanpy/"+flags1+'_'+str(samples[9])+"_label-1.txt",header=0,index_col=0,sep=',')
coord10=pd.read_csv(dirs+"gtt_input_scanpy/"+flags1+'_'+str(samples[9])+"_positions-1.txt",header=0,index_col=0,sep=',')
coord10=coord10.loc[:,['x','y']]
feat10.obsm['spatial'] = coord10.values
feat10.obs['celltype']=meta10.iloc[:,0].values
feat10.obs.loc[feat10.obs['celltype'].isna(), 'celltype'] = "unknown"

feat11=sc.read_csv(dirs+"gtt_input_scanpy/"+flags1+'_'+str(samples[10])+"_features-1.txt")
meta11=pd.read_csv(dirs+"gtt_input_scanpy/"+flags1+'_'+str(samples[10])+"_label-1.txt",header=0,index_col=0,sep=',')
coord11=pd.read_csv(dirs+"gtt_input_scanpy/"+flags1+'_'+str(samples[10])+"_positions-1.txt",header=0,index_col=0,sep=',')
coord11=coord11.loc[:,['x','y']]
feat11.obsm['spatial'] = coord11.values
feat11.obs['celltype']=meta11.iloc[:,0].values
feat11.obs.loc[feat11.obs['celltype'].isna(), 'celltype'] = "unknown"

feat12=sc.read_csv(dirs+"gtt_input_scanpy/"+flags1+'_'+str(samples[11])+"_features-1.txt")
meta12=pd.read_csv(dirs+"gtt_input_scanpy/"+flags1+'_'+str(samples[11])+"_label-1.txt",header=0,index_col=0,sep=',')
coord12=pd.read_csv(dirs+"gtt_input_scanpy/"+flags1+'_'+str(samples[11])+"_positions-1.txt",header=0,index_col=0,sep=',')
coord12=coord12.loc[:,['x','y']]
feat12.obsm['spatial'] = coord12.values
feat12.obs['celltype']=meta12.iloc[:,0].values
feat12.obs.loc[feat12.obs['celltype'].isna(), 'celltype'] = "unknown"

layer_groups = [[feat1, feat2, feat3, feat4],
                [feat5, feat6, feat7, feat8],
                [feat9, feat10, feat11, feat12]]

# Compute partial pairwise alignment using PASTE2
import time
pis = [[None for i in range(len(layer_groups[j])-1)] for j in range(len(layer_groups))]
for j in range(len(layer_groups)):
    for i in range(len(layer_groups[j])-1):
        start = time.time()
        s_pred = select_overlap_fraction(layer_groups[j][i], layer_groups[j][i+1], alpha=0.1)
        pis[j][i] = partial_pairwise_align(layer_groups[j][i], layer_groups[j][i+1], s=s_pred)
        tt = time.time()-start
        print('time',tt)
        np.savetxt('/home/zhaoshangrui/xuxinyu/PASTE_2/DLPFC/results_new/init_{0}_{1}_{2}.gz'.format(j,i,'ot'), pis[j][i], delimiter=',')

# pis = [[None for i in range(len(layer_groups[j])-1)] for j in range(len(layer_groups))]
# for j in range(len(layer_groups)):
#     for i in range(len(layer_groups[j])-1):
#         pis[j][i]=np.loadtxt('/home/zhaoshangrui/xuxinyu/PASTE_2/DLPFC/results_new/init_{0}_{1}_{2}.gz'.format(j,i,'ot'), delimiter=',')        

# paste2_layer_groups = [partial_stack_slices_pairwise(layer_groups[j], pis[j]) for j in range(len(layer_groups)) ]
