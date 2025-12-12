# -*- coding: utf-8 -*-
import os
import numpy as np
import scanpy as sc
import sklearn
import argparse
from scipy.sparse import csr_matrix

import pandas as pd
import time

import torch
import sys
sys.path.append("/home/zhaoshangrui/xuxinyu/PASTE_1/files")
from spiral.main import SPIRAL_integration
from spiral.layers import *
from spiral.utils import *
from spiral.CoordAlignment import CoordAlignment
R_dirs="/home/zhaoshangrui/anaconda2/envs/R4.3.1/lib/R"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def Cal_Spatial_Net(adata, rad_cutoff=None, k_cutoff=None, model='Radius', verbose=True):
    assert(model in ['Radius', 'KNN'])
    if verbose:
        print('------Calculating spatial graph...')
    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    if model == 'Radius':
        nbrs = sklearn.neighbors.NearestNeighbors(radius=rad_cutoff).fit(coor)
        distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it]*indices[it].shape[0], indices[it], distances[it])))    
    if model == 'KNN':
        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=k_cutoff+1).fit(coor)
        distances, indices = nbrs.kneighbors(coor)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it]*indices.shape[1],indices[it,:], distances[it,:])))
    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']
    Spatial_Net = KNN_df.copy()
    Spatial_Net = Spatial_Net.loc[Spatial_Net['Distance']>0,]
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)
    if verbose:
        print('The graph contains %d edges, %d cells.' %(Spatial_Net.shape[0], adata.n_obs))
        print('%.4f neighbors per cell on average.' %(Spatial_Net.shape[0]/adata.n_obs))
    adata.uns['Spatial_Net'] = Spatial_Net

sample_name=["10X_Normal","10X_DAPI","10X_FFPE"]
result_dirs="/home/zhaoshangrui/xuxinyu/SPIRAL/Coronal_mouse_brain/result/"

rad=150
knn=6
for k in range(len(sample_name)):
    feat=pd.read_csv(result_dirs+"gtt_input/"+str(sample_name[k])+"_mat.csv",header=0,index_col=0,sep=',')
    coord=pd.read_csv(result_dirs+"gtt_input/"+str(sample_name[k])+"_coord1.csv",header=0,index_col=0,sep=',')
    coord.columns=['x','y']
    adata = sc.AnnData(feat)
    adata.var_names_make_unique()
    adata.X=csr_matrix(adata.X)
    adata.obsm["spatial"] = coord.loc[:,['x','y']].to_numpy()
    Cal_Spatial_Net(adata, rad_cutoff=rad, k_cutoff=knn, model='KNN', verbose=True)
    if 'highly_variable' in adata.var.columns:
        adata_Vars =  adata[:, adata.var['highly_variable']]
    else:
        adata_Vars = adata
    features = pd.DataFrame(adata_Vars.X.toarray()[:, ], index=adata_Vars.obs.index, columns=adata_Vars.var.index)
    cells = np.array(features.index)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    if 'Spatial_Net' not in adata.uns.keys():
        raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!")
    Spatial_Net = adata.uns['Spatial_Net']
    G_df = Spatial_Net.copy()
    np.savetxt(result_dirs+"gtt_input/"+str(sample_name[k])+"_edge_KNN_"+str(knn)+".csv",G_df.values[:,:2],fmt='%s')


result_dirs="/home/zhaoshangrui/xuxinyu/SPIRAL/Coronal_mouse_brain/result/"
samples=["10X_Normal","10X_DAPI","10X_FFPE"]
SEP=','
net_cate='_KNN_'
knn=6

N_WALKS=knn
WALK_LEN=1
N_WALK_LEN=knn
NUM_NEG=knn

feat_file=[]
edge_file=[]
meta_file=[]
coord_file=[]
flags=''
for i in range(len(samples)):
    feat_file.append(result_dirs+"gtt_input/"+str(samples[i])+"_mat.csv")
    edge_file.append(result_dirs+"gtt_input/"+str(samples[i])+"_edge"+net_cate+str(knn)+".csv")
    meta_file.append(result_dirs+"gtt_input/"+str(samples[i])+"_meta.csv")
    coord_file.append(result_dirs+"gtt_input/"+str(samples[i])+"_coord1.csv")
    flags=flags+'_'+samples[i]
N=pd.read_csv(feat_file[0],header=0,index_col=0).shape[1]
if (len(samples)==2):
    M=1
else:
    M=len(samples)
parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=0, help='seed')
parser.add_argument('--AEdims', type=list, default=[N,[512],32], help='Dim of encoder.')
parser.add_argument('--AEdimsR', type=list, default=[32,[512],N], help='Dim of decoder.')
parser.add_argument('--GSdims', type=list, default=[512,32], help='Dim of GraphSAGE.')
parser.add_argument('--zdim', type=int, default=32, help='Dim of embedding.')
parser.add_argument('--znoise_dim', type=int, default=4, help='Dim of noise embedding.')
parser.add_argument('--CLdims', type=list, default=[4,[],M], help='Dim of classifier.')
parser.add_argument('--DIdims', type=list, default=[28,[32,16],M], help='Dim of discriminator.')
parser.add_argument('--beta', type=float, default=1.0, help='weight of GraphSAGE.')
parser.add_argument('--agg_class', type=str, default=MeanAggregator, help='Function of aggregator.')
parser.add_argument('--num_samples', type=str, default=knn, help='number of neighbors to sample.')

parser.add_argument('--N_WALKS', type=int, default=N_WALKS, help='number of walks of random work for postive pairs.')
parser.add_argument('--WALK_LEN', type=int, default=WALK_LEN, help='walk length of random work for postive pairs.')
parser.add_argument('--N_WALK_LEN', type=int, default=N_WALK_LEN, help='number of walks of random work for negative pairs.')
parser.add_argument('--NUM_NEG', type=int, default=NUM_NEG, help='number of negative pairs.')


parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=256, help='Size of batches to train.')
parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay.')
parser.add_argument('--alpha1', type=float, default=N, help='Weight of decoder loss.')
parser.add_argument('--alpha2', type=float, default=1, help='Weight of GraphSAGE loss.')
parser.add_argument('--alpha3', type=float, default=1, help='Weight of classifier loss.')
parser.add_argument('--alpha4', type=float, default=1, help='Weight of discriminator loss.')
parser.add_argument('--lamda', type=float, default=1, help='Weight of GRL.')
parser.add_argument('--Q', type=float, default=10, help='Weight negative loss for sage losss.')

params,unknown=parser.parse_known_args()

start = time.time()
SPII=SPIRAL_integration(params,feat_file,edge_file,meta_file)
SPII.train()
tt = time.time()-start
print('time',tt)

if not os.path.exists(result_dirs+"model/"):
    os.makedirs(result_dirs+"model/")
torch.save(SPII.model.state_dict(),result_dirs+"model/SPIRAL"+flags+"_model_"+str(SPII.params.batch_size)+".pt")

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
if not os.path.exists(result_dirs+"gtt_output/"):
    os.makedirs(result_dirs+"gtt_output/")
    
embed_file=result_dirs+"gtt_output/SPIRAL"+flags+"_embed_"+str(SPII.params.batch_size)+".csv"
embed1.to_csv(embed_file)

## Step2: clustering using seurat method in R in seurat_clustering.ipynb

import anndata
adata=anndata.AnnData(SPII.feat)
adata.obsm['spiral']=embed1.iloc[:,SPII.params.znoise_dim:].values
adata.obs['batch']=SPII.meta.loc[:,'batch'].values
adata.obs['celltype']=SPII.meta.loc[:,'celltype']
adata.write("/home/zhaoshangrui/xuxinyu/SPIRAL/Coronal_mouse_brain/result/ann.h5ad")

coord=pd.read_csv(coord_file[0],header=0,index_col=0)
for i in np.arange(1,len(samples)):
    coord=pd.concat((coord,pd.read_csv(coord_file[i],header=0,index_col=0)))
    
adata.obsm['spatial']=coord.loc[adata.obs_names,:].values
cluster_file="/home/zhaoshangrui/xuxinyu/SPIRAL/Coronal_mouse_brain/result/metrics/spiral_seuratmethod_clust_modify.csv"
clusters=pd.read_csv(cluster_file,header=0,index_col=0)
clusters.index = clusters.index.str.replace("^10X_", "", regex=True)
clusters.columns = ['clusters']
clust=[str(x) for x in clusters.loc[adata.obs_names,:].values[:,0]]
adata.obs['SPIRAL']=clust

## Step3:SPIRAL alignment

clust_cate="seuratmethod"
input_file=[meta_file,coord_file,embed_file,cluster_file]
output_dirs=result_dirs+"gtt_output/SPIRAL_alignment/"
if not os.path.exists(output_dirs):
    os.makedirs(output_dirs)
ub=['10X_Normal','10X_DAPI','10X_FFPE']

alpha=0.5
types="weighted_mean"
clust_cate='mclust'
R_dirs="/home/zhaoshangrui/anaconda2/envs/R4.3.1/lib/R"
CA=CoordAlignment(input_file=input_file,output_dirs=output_dirs,ub=ub,flags=flags,clust_cate=clust_cate,R_dirs=R_dirs,alpha=alpha,types=types)
New_Coord=CA.New_Coord
New_Coord.to_csv(output_dirs+"new_coord"+flags+"_modify_new.csv")

adata.obsm['aligned_spatial']=New_Coord.loc[adata.obs_names,:].values
adata.obsm['aligned_spatial'] = adata.obsm['aligned_spatial'][:,:2] 
adata.obsm['aligned_spatial'] = np.array(adata.obsm['aligned_spatial'], dtype=np.float64)
adata.write("/home/zhaoshangrui/xuxinyu/SPIRAL/Coronal_mouse_brain/result/ann_cluster.h5ad")
