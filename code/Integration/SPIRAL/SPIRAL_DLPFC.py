# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import argparse

import time
import torch
import torch.nn as nn
import sklearn
import scanpy as sc
import scipy.sparse as sp

import sys
sys.path.append("/home/zhaoshangrui/xuxinyu/PASTE_1/files")
from spiral.main import SPIRAL_integration
from spiral.layers import *
from spiral.utils import *
from spiral.CoordAlignment import CoordAlignment

R_dirs="/home/zhaoshangrui/anaconda2/envs/R4.3.1/lib/R"
os.environ['R_HOME']=R_dirs
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Data preprocess
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
    
input_dirs="/home/zhaoshangrui/xuxinyu/datasets/DLPFC_data/10xGenomicsVisium/spatialLIBD/"
result_dirs="/home/zhaoshangrui/xuxinyu/SPIRAL/DLPFC/sample1/"
if not os.path.exists(result_dirs+"gtt_input_scanpy/"):
    os.makedirs(result_dirs+"gtt_input_scanpy/") 
sample_name=[151507,151508,151509,151510]
IDX=np.arange(0,4)
VF=[]
MAT=[]
flags=str(sample_name[IDX[0]])
for i in np.arange(1,len(IDX)):
    flags=flags+'-'+str(sample_name[IDX[i]])
flags=flags+"_"
for k in np.arange(len(IDX)):
    adata = sc.read_visium(path=input_dirs+str(sample_name[IDX[k]])+"/",
                        count_file="filtered_feature_bc_matrix.h5")
    adata.var_names_make_unique()
    Ann_df=pd.read_csv(input_dirs+str(sample_name[IDX[k]])+'/'+str(sample_name[IDX[k]])+"_truth.txt", sep='\t',header=None, index_col=0, names = ['Ground Truth'])
    # Ann_df.columns=['Ground Truth','Sample']
    # Ann_df.index=[i.split('-')[1]+'-'+i.split('-')[2] for i in Ann_df.index]
    adata=adata[Ann_df.index,:]
    adata.obs['Ground Truth']=Ann_df.loc[:,'Ground Truth']
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=1000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.obs['batch']=str(sample_name[IDX[k]])
    cells=[str(sample_name[IDX[k]])+'-'+i for i in adata.obs_names]
    mat1=pd.DataFrame(adata.X.toarray(),columns=adata.var_names,index=cells)
    coord1=pd.DataFrame(adata.obsm['spatial'],columns=['x','y'],index=cells)
    meta1=adata.obs[['Ground Truth','batch']]
    meta1.columns=['celltype','batch']
    meta1.index=cells
    meta1.to_csv(result_dirs+"gtt_input_scanpy/"+flags+str(sample_name[IDX[k]])+"_label-1.txt")
    coord1.to_csv(result_dirs+"gtt_input_scanpy/"+flags+str(sample_name[IDX[k]])+"_positions-1.txt")
    MAT.append(mat1)
    VF=np.union1d(VF,adata.var_names[adata.var['highly_variable']])
    
for i in np.arange(len(IDX)):
    mat=MAT[i]
    mat=mat.loc[:,VF]
    mat.to_csv(result_dirs+"gtt_input_scanpy/"+flags+str(sample_name[IDX[i]])+"_features-1.txt")

rad=150
KNN=6
for i in np.arange(len(IDX)):
    features=pd.read_csv(result_dirs+"gtt_input_scanpy/"+flags+str(sample_name[IDX[i]])+"_features-1.txt",header=0,index_col=0,sep=',')
    meta=pd.read_csv(result_dirs+"gtt_input_scanpy/"+flags+str(sample_name[IDX[i]])+"_label-1.txt",header=0,index_col=0,sep=',')
    coord=pd.read_csv(result_dirs+"gtt_input_scanpy/"+flags+str(sample_name[IDX[i]])+"_positions-1.txt",header=0,index_col=0,sep=',')
    adata = sc.AnnData(features)
    adata.var_names_make_unique()
    adata.X=sp.csr_matrix(adata.X)
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
    np.savetxt(result_dirs+"gtt_input_scanpy/"+flags+str(sample_name[IDX[i]])+"_edge_KNN_"+str(KNN)+".csv",G_df.values[:,:2],fmt='%s')

#Step1:SPIRAL integration
result_dirs="/home/zhaoshangrui/xuxinyu/SPIRAL/DLPFC/results/"
sample_name=[151507,151508,151509,151510,151669,151670,151671,151672,151673,151674,151675,151676]
samples=np.array(sample_name)
IDX=np.arange(0,12)
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
flags=str(sample_name[0])
for i in np.arange(1,len(sample_name)):
    flags=flags+'-'+str(sample_name[IDX[i]])
flags=flags+"_"
IDX=np.arange(0,4)
for i in range(len(IDX)):
    feat_file.append(result_dirs+"gtt_input_scanpy/"+flags+str(sample_name[IDX[i]])+"_features-1.txt")
    edge_file.append(result_dirs+"gtt_input_scanpy/"+flags+str(sample_name[IDX[i]])+"_edge_KNN_"+str(knn)+".csv")
    meta_file.append(result_dirs+"gtt_input_scanpy/"+flags+str(sample_name[IDX[i]])+"_label-1.txt")
    coord_file.append(result_dirs+"gtt_input_scanpy/"+flags+str(sample_name[IDX[i]])+"_positions-1.txt")
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
parser.add_argument('--num_samples', type=str, default=knn, help='number of neighbors to sample.')

parser.add_argument('--N_WALKS', type=int, default=N_WALKS, help='number of walks of random work for postive pairs.')
parser.add_argument('--WALK_LEN', type=int, default=WALK_LEN, help='walk length of random work for postive pairs.')
parser.add_argument('--N_WALK_LEN', type=int, default=N_WALK_LEN, help='number of walks of random work for negative pairs.')
parser.add_argument('--NUM_NEG', type=int, default=NUM_NEG, help='number of negative pairs.')


parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=512, help='Size of batches to train.') ####512 for withon donor;1024 for across donor###
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
if not os.path.exists(result_dirs+"gtt_output_sample1/"):
    os.makedirs(result_dirs+"gtt_output_sample1/") 
embed_file=result_dirs+"gtt_output_sample1/SPIRAL"+flags+"_embed_"+str(SPII.params.batch_size)+"_sample1.csv"
embed1.to_csv(embed_file)
meta=SPII.meta.values

#embed_new=torch.cat((torch.zeros((embed.shape[0],SPII.params.znoise_dim)),embed.iloc[:,SPII.params.znoise_dim:]),dim=1)
embed_new = torch.cat((torch.zeros((embed.shape[0], SPII.params.znoise_dim)), embed[:, SPII.params.znoise_dim:]), dim=1)
xbar_new=np.array(SPII.model.agc.ae.de(embed_new.cuda(),nn.Sigmoid())[1].cpu().detach())
xbar_new1=pd.DataFrame(xbar_new,index=SPII.feat.index,columns=SPII.feat.columns)
xbar_new1.to_csv(result_dirs+"gtt_output_sample1/SPIRAL"+flags+"_correct_"+str(SPII.params.batch_size)+"_sample1.csv")


#step2: clustering
import anndata
ann=anndata.AnnData(SPII.feat)
ann.obsm['spiral']=embed1.iloc[:,SPII.params.znoise_dim:].values
sc.pp.neighbors(ann,use_rep='spiral')

n_clust=7
ann = mclust_R(ann, used_obsm='spiral', num_cluster=n_clust)

ann.obs['batch']=SPII.meta.loc[:,'batch'].values
ub=np.unique(ann.obs['batch'])
sc.tl.umap(ann)
coord=pd.read_csv(coord_file[0],header=0,index_col=0)
for i in np.arange(1,len(IDX)):
    coord=pd.concat((coord,pd.read_csv(coord_file[i],header=0,index_col=0)))

coord.columns=['y','x']
ann.obsm['spatial']=coord.loc[ann.obs_names,:].values
cluster_file=result_dirs+"gtt_output/SPIRAL"+flags+"_mclust_sample1.csv"
pd.DataFrame(ann.obs['mclust']).to_csv(cluster_file)

ann.write("/home/zhaoshangrui/xuxinyu/SPIRAL/DLPFC/results/ann_sample1.h5ad")


#step3: smooth clusters
def refine(sample_id, pred, dis, num_nbs=8):
    refined_pred=[]
    pred=pd.DataFrame({"pred": pred}, index=sample_id)
    dis_df=pd.DataFrame(dis, index=sample_id, columns=sample_id)
    for i in range(len(sample_id)):
        index=sample_id[i]
        dis_tmp=dis_df.loc[index, :].sort_values()
        nbs=dis_tmp[0:num_nbs+1]
        nbs_pred=pred.loc[nbs.index, "pred"]
        self_pred=pred.loc[index, "pred"]
        v_c=nbs_pred.value_counts()
        if (v_c.loc[self_pred]<(num_nbs+1)/2) and (np.max(v_c)>(num_nbs+1)/2):
            refined_pred.append(v_c.idxmax())
        else:           
            refined_pred.append(self_pred)
    return refined_pred

from sklearn.metrics.pairwise import euclidean_distances
ann.obs['SPIRAL']=ann.obs['mclust']
ann.obs['SPIRAL_refine']=ann.obs['SPIRAL']
ub=np.unique(ann.obs['batch'])
for i in range(len(ub)):
    idx=np.where(ann.obs['batch']==ub[i])[0]
    ann1=ann[idx,:]
    sample_id=ann1.obs_names
    pred=ann1.obs['SPIRAL']
    dis=euclidean_distances(ann1.obsm['spatial'],ann1.obsm['spatial'])
    refined_pred=refine(sample_id, pred, dis, num_nbs=knn)
    ann.obs['SPIRAL_refine'][idx]=refined_pred
    
cluster_file_save=result_dirs+"metrics/spiral"+flags+"_mclust_modify_sample1.csv"
pd.DataFrame(ann.obs['SPIRAL_refine']).to_csv(cluster_file_save)

ann.write("/home/zhaoshangrui/xuxinyu/SPIRAL/DLPFC/results/ann_sample1.h5ad")


#step4: SPIRAL alignment
clust_cate='louvain'
input_file=[meta_file,coord_file,embed_file,cluster_file]
output_dirs=result_dirs+"gtt_output_sample1/SPIRAL_alignment/"
if not os.path.exists(output_dirs):
    os.makedirs(output_dirs)
ub=samples

alpha=0.5
types="weighted_mean"
CA=CoordAlignment(input_file=input_file,output_dirs=output_dirs,ub=ub,flags=flags,clust_cate=clust_cate,R_dirs=R_dirs,alpha=alpha,types=types)
New_Coord=CA.New_Coord
New_Coord.to_csv(output_dirs+"new_coord"+flags+"_modify_sample1.csv")
ann.obsm['aligned_spatial']=New_Coord.loc[ann.obs_names,:].values

ann.obsm['spatial'] = ann.obsm['aligned_spatial'][:,:2] 
ann.write("/home/zhaoshangrui/xuxinyu/SPIRAL/DLPFC/results/ann_sample1.h5ad")