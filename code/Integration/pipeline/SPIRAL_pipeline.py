# -*- coding: utf-8 -*-
import os
import anndata
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import torch

import sys
sys.path.append("/home/zhaoshangrui/xuxinyu/PASTE_1/files")
from spiral.main import SPIRAL_integration
from spiral.layers import *
from spiral.utils import *
from spiral.CoordAlignment import CoordAlignment

R_dirs="/home/zhaoshangrui/anaconda2/envs/R4.3.1/lib/R"
os.environ['R_HOME']=R_dirs
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import rpy2.robjects as robjects
robjects.r.library("mclust")


###Step1:SPIRAL integration 
result_dirs=".../result/"
section_ids=np.array(['data_slice1','data_slice2'])
samples=section_ids[0:2]

SEP=','
net_cate='_KNN_'
rad=150
knn=8

N_WALKS=knn
WALK_LEN=1
N_WALK_LEN=knn
NUM_NEG=knn

feat_file=[]
edge_file=[]
meta_file=[]
coord_file=[]
for i in range(len(samples)):
    feat_file.append(result_dirs+"get_input_scanpy/"+str(samples[i])+"_features-1.txt")
    edge_file.append(result_dirs+"get_input_scanpy/"+str(samples[i])+"_edge_KNN_"+str(knn)+".csv")
    meta_file.append(result_dirs+"get_input_scanpy/"+str(samples[i])+"_label-1.txt")
    coord_file.append(result_dirs+"get_input_scanpy/"+str(samples[i])+"_positions-1.txt")
N=pd.read_csv(feat_file[0],header=0,index_col=0).shape[1]
if (len(samples)==2):
    M=1
else:
    M=len(samples)

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=0, help='The seed of initialization.')
parser.add_argument('--AEdims', type=list, default=[N,[512],32], help='Dim of encoder.')
parser.add_argument('--AEdimsR', type=list, default=[32,[512],N], help='Dim of decoder.')
parser.add_argument('--GSdims', type=list, default=[512,32], help='Dim of GraphSAGE.')
parser.add_argument('--zdim', type=int, default=32, help='Dim of embedding.')
parser.add_argument('--znoise_dim', type=int, default=4, help='Dim of noise embedding.')
parser.add_argument('--CLdims', type=list, default=[4,[],M], help='Dim of classifier.')
parser.add_argument('--DIdims', type=list, default=[28,[32,16],M], help='Dim of discriminator.')
parser.add_argument('--beta', type=float, default=1.0, help='weight of GraphSAGE.')
parser.add_argument('--agg_class', type=str, default=MeanAggregator, help='Function of aggregator.')
parser.add_argument('--num_samples', type=str, default=knn, help='number of neighbors to sample.')#

parser.add_argument('--N_WALKS', type=int, default=N_WALKS, help='number of walks of random work for postive pairs.')
parser.add_argument('--WALK_LEN', type=int, default=WALK_LEN, help='walk length of random work for postive pairs.')
parser.add_argument('--N_WALK_LEN', type=int, default=N_WALK_LEN, help='number of walks of random work for negative pairs.')
parser.add_argument('--NUM_NEG', type=int, default=NUM_NEG, help='number of negative pairs.')


parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=1024, help='Size of batches to train.') ###512 for withon donor;1024 for across donor###
parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay.')
parser.add_argument('--alpha1', type=float, default=N, help='Weight of decoder loss.')
parser.add_argument('--alpha2', type=float, default=1, help='Weight of GraphSAGE loss.')
parser.add_argument('--alpha3', type=float, default=1, help='Weight of classifier loss.')
parser.add_argument('--alpha4', type=float, default=1, help='Weight of discriminator loss.')
parser.add_argument('--lamda', type=float, default=1, help='Weight of GRL.')
parser.add_argument('--Q', type=float, default=10, help='Weight negative loss for sage losss.')

params,unknown=parser.parse_known_args()

import time
start_time = time.time()
SPII=SPIRAL_integration(params,feat_file,edge_file,meta_file)
SPII.train()
end_time = time.time()
elapsed_time = end_time - start_time
print(f"实验运行时间: {elapsed_time:.2f} 秒")

SPII.model.eval()
all_idx=np.arange(SPII.feat.shape[0])
all_layer,all_mapping=layer_map(all_idx.tolist(),SPII.adj,len(SPII.params.GSdims))
all_rows=SPII.adj.tolil().rows[all_layer[0]]
all_feature=torch.Tensor(SPII.feat.iloc[all_layer[0],:].values).float().cuda()
all_embed,ae_out,clas_out,disc_out=SPII.model(all_feature,all_layer,all_mapping,all_rows,SPII.params.lamda,SPII.de_act,SPII.cl_act)
[ae_embed,gs_embed,embed]=all_embed
[x_bar,x]=ae_out
embed=embed.cpu().detach()
names=['GTT_'+str(i) for i in range(embed.shape[1])]
embed1=pd.DataFrame(np.array(embed),index=SPII.feat.index,columns=names)
if not os.path.exists(result_dirs+"get_output/"):
    os.makedirs(result_dirs+"get_output/") 
embed_file=result_dirs+"get_output/SPIRAL"+"_embed_"+str(SPII.params.batch_size)+".csv"
embed1.to_csv(embed_file)
meta=SPII.meta.values

#embed_new=torch.cat((torch.zeros((embed.shape[0],SPII.params.znoise_dim)),embed.iloc[:,SPII.params.znoise_dim:]),dim=1)
embed_new = torch.cat((torch.zeros((embed.shape[0], SPII.params.znoise_dim)), embed[:, SPII.params.znoise_dim:]), dim=1)
xbar_new=np.array(SPII.model.agc.ae.de(embed_new.cuda(),nn.Sigmoid())[1].cpu().detach())
xbar_new1=pd.DataFrame(xbar_new,index=SPII.feat.index,columns=SPII.feat.columns)
xbar_new1.to_csv(result_dirs+"get_output/SPIRAL"+"_correct_"+str(SPII.params.batch_size)+".csv")

meta=SPII.meta.values

###step2: clustering
ann=anndata.AnnData(SPII.feat)
ann.obsm['spiral']=embed1.iloc[:,SPII.params.znoise_dim:].values
sc.pp.neighbors(ann, use_rep='spiral', random_state=666)
sc.tl.louvain(ann, random_state=666, key_added="louvain")
ann.obs['batch']=SPII.meta.loc[:,'batch'].values
ub=np.unique(ann.obs['batch'])
sc.tl.umap(ann)
coord=pd.read_csv(coord_file[0],header=0,index_col=0)
for i in np.arange(1,len(samples)):
    coord=pd.concat((coord,pd.read_csv(coord_file[i],header=0,index_col=0)))

coord.columns=['y','x']
ann.obsm['spatial']=coord.loc[ann.obs_names,:].values
cluster_file=result_dirs+"get_output/SPIRAL_louvain.csv"
pd.DataFrame(ann.obs['louvain']).to_csv(cluster_file)

###step4: SPIRAL alignment
clust_cate='louvain'
input_file=[meta_file,coord_file,embed_file,cluster_file]
output_dirs=result_dirs+"get_output/SPIRAL_alignment/"
if not os.path.exists(output_dirs):
    os.makedirs(output_dirs)
ub=['data_slice1','data_slice2']

alpha=0.5
types="weighted_mean"
R_dirs="/home/zhaoshangrui/anaconda2/envs/R4.3.1/lib/R"
CA=CoordAlignment(input_file=input_file,output_dirs=output_dirs,ub=ub,flags='_',clust_cate=clust_cate,R_dirs=R_dirs,alpha=alpha,types=types)
New_Coord=CA.New_Coord
New_Coord.to_csv(output_dirs+"new_coord_modify.csv")
New_Coord = pd.read_csv(output_dirs+"new_coord_modify.csv", header=0,index_col=0,sep=',')
ann.obsm['aligned_spatial']=New_Coord.loc[ann.obs_names,:].values
ann.obsm['spatial'] = ann.obsm['aligned_spatial'][:,:2] 
ann.obsm['spatial'] = ann.obsm['spatial'].astype(str)
ann.obsm['aligned_spatial'] = ann.obsm['aligned_spatial'].astype(str)
ann.write('.../metrice/ann_aligned.h5ad')
