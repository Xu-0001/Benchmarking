# -*- coding: utf-8 -*-
import os
import anndata as ad
import pandas as pd
import numpy as np
import scanpy as sc
from scipy.sparse import csc_matrix
import argparse

import sklearn
import time
import torch

import sys
sys.path.append("/home/zhaoshangrui/xuxinyu/PASTE_1/files")
from spiral.main import SPIRAL_integration
from spiral.layers import *
from spiral.utils import *
from spiral.CoordAlignment import CoordAlignment

R_dirs="/home/zhaoshangrui/anaconda2/envs/R4.3.1/lib/R"
os.environ['R_HOME']=R_dirs
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import rpy2.robjects as robjects
robjects.r.library("mclust")

#data process
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

Batch_list = []
section_ids = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6',
                'B1', 'B2', 'B3', 'B4', 'B5', 'B6',
                'C1', 'C2', 'C3', 'C4', 'C5', 'C6',
                'D1', 'D2', 'D3', 'D4', 'D5', 'D6',
                'E1', 'E2', 'E3',
                'F1', 'F2', 'F3',
                'G1', 'G2', 'G3',
                'H1', 'H2', 'H3',]
data_path = "/home/zhaoshangrui/xuxinyu/datasets/breast_tumour_data/HER2-positive_breast_tumour_dataset_profiled_by_ST_platform/raw_data/"
samples_per_patient = {'A':6,'B':6,'C':6,'D':6,'E':3,'F':3,'G':3,'H':3}
patients = {p:[0 for i in range(samples_per_patient[p])] for p in samples_per_patient}
for p in samples_per_patient:
    print('Patient ',p)
    for i in range(samples_per_patient[p]):
        print("\t Slice ",i+1)
        st_count = pd.read_csv(data_path + '{0}{1}.tsv.gz'.format(p,i+1), index_col=0, sep='\t')
        st_meta = pd.read_csv(data_path + '{0}{1}_selection.tsv'.format(p,i+1), sep='\t')
        st_meta.index = [str(st_meta['x'][i]) + 'x' + str(st_meta['y'][i]) for i in range(st_meta.shape[0])]
        st_meta = st_meta.loc[st_count.index]
        adata_st_i = ad.AnnData(X=st_count.values)
        adata_st_i.obs.index = st_count.index
        adata_st_i.obsm['spatial'] = np.array(list(map(lambda x: x.split('x'),adata_st_i.obs.index)),dtype=float)
        adata_st_i.obs = st_meta
        adata_st_i.var.index = st_count.columns
        sc.pp.filter_genes(adata_st_i, min_counts = 15)
        sc.pp.filter_cells(adata_st_i, min_counts = 100)
        print(adata_st_i.shape)
        patho_anno_file = '{0}2_labeled_coordinates.tsv' if p == 'G' else '{0}1_labeled_coordinates.tsv'
        patho_anno = pd.read_csv(data_path + patho_anno_file.format(p), delimiter='\t')
        if i == (1 if p == 'G' else 0):
            adata_st_i.obs['annotation'] = None
            for idx in adata_st_i.obs.index:
                adata_st_i.obs['annotation'].loc[idx] = patho_anno[
                    (patho_anno["x"] == adata_st_i[idx].obs["new_x"].values[0]) & 
                    (patho_anno["y"] == adata_st_i[idx].obs["new_y"].values[0])
                ]['label'].values[0]
        else:
            adata_st_i.obs['annotation'] = "Unknown" 
        adata_st_i.X = csc_matrix(adata_st_i.X)     
        patients[p][i] = adata_st_i 
        Batch_list.append(adata_st_i)

result_dirs="/home/zhaoshangrui/xuxinyu/SPIRAL/HER2/result/"
IDX=np.arange(0,6)
VF=[]
MAT=[]
for k in range(len(IDX)):
    adata = Batch_list[IDX[k]]
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=1000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.obs['batch']=str(section_ids[IDX[k]])
    cells=[str(section_ids[IDX[k]])+'-'+i for i in adata.obs_names]
    mat1=pd.DataFrame(adata.X.toarray(),columns=adata.var_names,index=cells)
    coord1=pd.DataFrame(adata.obsm['spatial'],columns=['x','y'],index=cells)
    meta1=adata.obs[['annotation', 'batch']]
    meta1.columns=['celltype','batch']
    meta1.index=cells
    meta1.to_csv(result_dirs+"get_input_scanpy/"+str(section_ids[IDX[k]])+"_label-1.txt")
    coord1.to_csv(result_dirs+"get_input_scanpy/"+str(section_ids[IDX[k]])+"_positions-1.txt")
    MAT.append(mat1)
    VF=np.union1d(VF,adata.var_names[adata.var['highly_variable']])
 
VF_list = list(VF)
for i in range(len(IDX)):
    mat = MAT[i]
    missing_genes = list(set(VF_list) - set(mat.columns))
    mat = pd.concat([mat, pd.DataFrame(0, index=mat.index, columns=missing_genes)], axis=1)
    mat = mat.loc[:, VF_list]
    mat.to_csv(result_dirs + "get_input_scanpy/" + str(section_ids[IDX[i]]) + "_features-1.txt")    

rad=150
KNN=6
for i in np.arange(len(IDX)):
    features=pd.read_csv(result_dirs+"get_input_scanpy/"+str(section_ids[IDX[i]])+"_features-1.txt",header=0,index_col=0,sep=',')
    meta=pd.read_csv(result_dirs+"get_input_scanpy/"+str(section_ids[IDX[i]])+"_label-1.txt",header=0,index_col=0,sep=',')
    coord=pd.read_csv(result_dirs+"get_input_scanpy/"+str(section_ids[IDX[i]])+"_positions-1.txt",header=0,index_col=0,sep=',')
    # meta=meta.iloc[:meta.shape[0]-1,:]
    adata = sc.AnnData(features)
    adata.var_names_make_unique()
    adata.X=csc_matrix(adata.X)
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
    np.savetxt(result_dirs+"get_input_scanpy/"+str(section_ids[IDX[i]])+"_edge_KNN_"+str(KNN)+".csv",G_df.values[:,:2],fmt='%s')


###Step1:SPIRAL integration 
result_dirs="/home/zhaoshangrui/xuxinyu/SPIRAL/HER2/result/"
section_ids=np.array(['A1', 'A2', 'A3', 'A4', 'A5', 'A6'])
samples=section_ids[0:6]

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
if not os.path.exists(result_dirs+"get_output_A/"):
    os.makedirs(result_dirs+"get_output_A/") 
embed_file=result_dirs+"get_output_A/SPIRAL"+"_embed_"+str(SPII.params.batch_size)+"_new.csv"
embed1.to_csv(embed_file)


###step2: clustering
import anndata
# import scanpy as sc
ann=anndata.AnnData(SPII.feat)
ann.obsm['spiral']=embed1.iloc[:,SPII.params.znoise_dim:].values
sc.pp.neighbors(ann,use_rep='spiral')

n_clust=6
ann = mclust_R(ann, used_obsm='spiral', num_cluster=n_clust)

ann.obs['batch']=SPII.meta.loc[:,'batch'].values
ub=np.unique(ann.obs['batch'])
sc.tl.umap(ann)
coord=pd.read_csv(coord_file[0],header=0,index_col=0)
for i in np.arange(1,len(samples)):
    coord=pd.concat((coord,pd.read_csv(coord_file[i],header=0,index_col=0)))

coord.columns=['y','x']
ann.obsm['spatial']=coord.loc[ann.obs_names,:].values
cluster_file=result_dirs+"get_output_A/SPIRAL_mclust_new.csv"
pd.DataFrame(ann.obs['mclust']).to_csv(cluster_file)
ann.write("/home/zhaoshangrui/xuxinyu/SPIRAL/HER2/result/ann_A_new.h5ad")

###step3: SPIRAL alignment
clust_cate='louvain'
input_file=[meta_file,coord_file,embed_file,cluster_file]
output_dirs=result_dirs+"get_output_A/SPIRAL_alignment/"
if not os.path.exists(output_dirs):
    os.makedirs(output_dirs)
ub=samples

alpha=0.5
types="weighted_mean"
R_dirs="/home/zhaoshangrui/anaconda2/envs/R4.3.1/lib/R"
CA=CoordAlignment(input_file=input_file,output_dirs=output_dirs,ub=ub,flags='_',clust_cate=clust_cate,R_dirs=R_dirs,alpha=alpha,types=types)
New_Coord=CA.New_Coord
New_Coord.to_csv(output_dirs+"new_coord_modify.csv")
ann.obsm['aligned_spatial']=New_Coord.loc[ann.obs_names,:].values
ann.obsm['aligned_spatial'] = ann.obsm['aligned_spatial'][:,:2] 
ann.obsm['spatial'] = ann.obsm['aligned_spatial'].astype(str)
ann.write("/home/zhaoshangrui/xuxinyu/SPIRAL/HER2/result/ann_A_new.h5ad")
