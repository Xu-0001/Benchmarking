# -*- coding: utf-8 -*-
import os
import anndata
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import sklearn
from scipy.sparse import csr_matrix

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

# data process
def Cal_Spatial_Net(adata, rad_cutoff=None, k_cutoff=None, model='Radius', verbose=True):
    assert(model in ['Radius', 'KNN'])
    if verbose:
        print('------Calculating spatial graph...')
    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
#     coor.columns = ['imagerow', 'imagecol']
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

adata = sc.read_h5ad("/home/zhaoshangrui/xuxinyu/datasets/Human_hepatocellular_carcinoma/HCC4_seulist.h5ad")
adata.obsm['spatial'] = adata.obsm['X_position']
adata.obs["batch"] = adata.obs["batch"].astype('category')

Batch_list = []
for i in range(1,adata.obs['batch'].nunique()+1):
    data = adata[adata.obs['batch']==i]
    Batch_list.append(data)
    
result_dirs="/home/zhaoshangrui/xuxinyu/SPIRAL/HCC/result/"
if not os.path.exists(result_dirs+"get_input_scanpy/"):
    os.makedirs(result_dirs+"get_input_scanpy/") 
IDX=np.arange(0,4)
VF=[]
MAT=[]
for k in range(len(IDX)):
    data = Batch_list[k]
    cells=[str(IDX[k])+'-'+i for i in data.obs_names]
    mat1=pd.DataFrame(data.X.toarray(),columns=data.var_names,index=cells)
    coord1=pd.DataFrame(data.obsm['spatial'],columns=['x','y'],index=cells)
    meta1=data.obs['batch']
    meta1.columns=['batch']
    meta1.index=cells
    meta1.to_csv(result_dirs+"get_input_scanpy/"+str(IDX[k])+"_label-1.txt")
    coord1.to_csv(result_dirs+"get_input_scanpy/"+str(IDX[k])+"_positions-1.txt")
    MAT.append(mat1)
    VF=np.union1d(VF,data.var_names)
 
for i in np.arange(len(IDX)):
    mat=MAT[i]
    mat=mat.loc[:,VF]
    mat.to_csv(result_dirs + "get_input_scanpy/" +str(IDX[k])+ "_features-1.txt")    

rad=150
KNN=6
for k in np.arange(len(IDX)):
    features=pd.read_csv(result_dirs+"get_input_scanpy/"+str(IDX[k])+"_features-1.txt",header=0,index_col=0,sep=',')
    meta=pd.read_csv(result_dirs+"get_input_scanpy/"+str(IDX[k])+"_label-1.txt",header=0,index_col=0,sep=',')
    coord=pd.read_csv(result_dirs+"get_input_scanpy/"+str(IDX[k])+"_positions-1.txt",header=0,index_col=0,sep=',')
    # meta=meta.iloc[:meta.shape[0]-1,:]
    adata = sc.AnnData(features)
    adata.var_names_make_unique()
    adata.X=csr_matrix(adata.X)
    adata.obsm["spatial"] = coord.loc[:,['x','y']].to_numpy()
    Cal_Spatial_Net(adata, rad_cutoff=rad, k_cutoff=6, model='KNN', verbose=True)
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
    np.savetxt(result_dirs+"get_input_scanpy/"+str(IDX[k])+"_edge_KNN_"+str(KNN)+".csv",G_df.values[:,:2],fmt='%s')


###Step1:SPIRAL integration 
result_dirs="/home/zhaoshangrui/xuxinyu/SPIRAL/HCC/result/"
samples=np.arange(0,4)

SEP=','
net_cate='_KNN_'
rad=150
knn=6

N_WALKS=knn
WALK_LEN=1
N_WALK_LEN=knn
NUM_NEG=knn

feat_file=[]
edge_file=[]
meta_file=[]
coord_file=[]
for k in range(len(samples)):
    feat_file.append(result_dirs+"get_input_scanpy/"+str(samples[k])+"_features-1.txt")
    edge_file.append(result_dirs+"get_input_scanpy/"+str(samples[k])+"_edge_KNN_"+str(knn)+".csv")
    meta_file.append(result_dirs+"get_input_scanpy/"+str(samples[k])+"_label-1.txt")
    coord_file.append(result_dirs+"get_input_scanpy/"+str(samples[k])+"_positions-1.txt")
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
parser.add_argument('--beta', type=float, default=1.0, help='weight of GraphSAGE.')#
parser.add_argument('--agg_class', type=str, default=MeanAggregator, help='Function of aggregator.')
parser.add_argument('--num_samples', type=str, default=knn, help='number of neighbors to sample.')#

parser.add_argument('--N_WALKS', type=int, default=N_WALKS, help='number of walks of random work for postive pairs.')
parser.add_argument('--WALK_LEN', type=int, default=WALK_LEN, help='walk length of random work for postive pairs.')
parser.add_argument('--N_WALK_LEN', type=int, default=N_WALK_LEN, help='number of walks of random work for negative pairs.')
parser.add_argument('--NUM_NEG', type=int, default=NUM_NEG, help='number of negative pairs.')


parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')#
parser.add_argument('--batch_size', type=int, default=512, help='Size of batches to train.') ###512 for withon donor;1024 for across donor###
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
embed_file=result_dirs+"get_output/SPIRAL"+"_embed_"+str(SPII.params.batch_size)+"_1.csv"
embed1.to_csv(embed_file)

###step2: clustering
ann=anndata.AnnData(SPII.feat)
ann.obsm['spiral']=embed1.iloc[:,SPII.params.znoise_dim:].values
sc.pp.neighbors(ann,use_rep='spiral')
sc.tl.leiden(ann,resolution=1.0)
sc.tl.louvain(ann,resolution=1)
ann.obs['batch']=SPII.meta.loc[:,'batch'].values
ub=np.unique(ann.obs['batch'])
sc.tl.umap(ann)
coord=pd.read_csv(coord_file[0],header=0,index_col=0)
for i in np.arange(1,len(samples)):
    coord=pd.concat((coord,pd.read_csv(coord_file[i],header=0,index_col=0)))

coord.columns=['y','x']
ann.obsm['spatial']=coord.loc[ann.obs_names,:].values
ann.write("/home/zhaoshangrui/xuxinyu/SPIRAL/HCC/result/ann_1.h5ad")
cluster_file=result_dirs+"get_output/SPIRAL_louvain_1.csv"
pd.DataFrame(ann.obs['louvain']).to_csv(cluster_file)

# sc.tl.louvain(ann,resolution=1.2)
# ann1=ann[ann.obs['batch']==ub[0],:]
# sc.pl.spatial(ann1,color="louvain", spot_size=100)
# ann1=ann[ann.obs['batch']==ub[1],:]
# sc.pl.spatial(ann1,color="louvain", spot_size=100)
# ann1=ann[ann.obs['batch']==ub[2],:]
# sc.pl.spatial(ann1,color="louvain", spot_size=100)
# ann1=ann[ann.obs['batch']==ub[3],:]
# sc.pl.spatial(ann1,color="louvain", spot_size=100)


# Step3:SPIRAL alignment
clust_cate='louvain'
input_file=[meta_file,coord_file,embed_file,cluster_file]
output_dirs=result_dirs+"get_output/SPIRAL_alignment/"
if not os.path.exists(output_dirs):
    os.makedirs(output_dirs)
ub=samples

alpha=0.5
types="weighted_mean"
R_dirs="/home/zhaoshangrui/anaconda2/envs/R4.3.1/lib/R"
CA=CoordAlignment(input_file=input_file,output_dirs=output_dirs,ub=ub,flags='_',clust_cate=clust_cate,R_dirs=R_dirs,alpha=alpha,types=types)
New_Coord=CA.New_Coord
New_Coord.to_csv(output_dirs+"new_coord_modify_1.csv")

ann.obsm['aligned_spatial']=New_Coord.loc[ann.obs_names,:].values
ann.obsm['spatial'] = ann.obsm['aligned_spatial'][:,:2] 
ann.write("/home/zhaoshangrui/xuxinyu/SPIRAL/HCC/result/ann_1.h5ad")