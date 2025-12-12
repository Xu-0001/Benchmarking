# -*- coding: utf-8 -*-
import torch
import numpy as np
import pandas as pd

import scanpy as sc
import anndata

import sys
sys.path.append("/home/zhaoshangrui/xuxinyu/PASTE_1/files")
from gpsa import VariationalGPSA, rbf_kernel
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import r2_score

def scale_spatial_coords(X, max_val=10.0):
    X = X - X.min(0)
    X = X / X.max(0)
    return X * max_val

N_GENES = 10
N_SAMPLES = None
N_LAYERS = 4
fixed_view_idx = 1

n_spatial_dims = 2
n_views = 4
m_G = 200
m_X_per_view = 200

N_LATENT_GPS = {"expression": None}

N_EPOCHS = 5000
PRINT_EVERY = 25

input_dirs="/home/zhaoshangrui/xuxinyu/datasets/DLPFC_data/10xGenomicsVisium/spatialLIBD/"
sample_name=[151507,151508,151509,151510,151669,151670,151671,151672,151673,151674,151675,151676]
IDX=np.arange(8,12)
Batch_list = []
for k in np.arange(len(IDX)):
    adata = sc.read_visium(path=input_dirs+str(sample_name[IDX[k]])+"/",
                        count_file="filtered_feature_bc_matrix.h5")
    adata.var_names_make_unique()
    Ann_df=pd.read_csv(input_dirs+str(sample_name[IDX[k]])+'/'+str(sample_name[IDX[k]])+"_truth.txt", sep='\t',header=None, index_col=0, names = ['Ground Truth'])
    adata=adata[Ann_df.index,:]
    adata.obs['celltype']=Ann_df.loc[:,'Ground Truth']
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
    sc.pp.normalize_total(adata, target_sum=1e4, inplace=True)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=10000, subset=True)
    adata.obs['batch']=str(sample_name[IDX[k]])
    Batch_list.append(adata)

import anndata as ad
data = ad.concat(Batch_list)
data_slice1 = Batch_list[0]
shared_gene_names = data.var_names
data_knn = data_slice1[:, shared_gene_names]
X_knn = data_knn.obsm["spatial"]
Y_knn = np.array(data_knn.X.todense()) 
nbrs = NearestNeighbors(n_neighbors=2).fit(X_knn)
distances, indices = nbrs.kneighbors(X_knn)

preds = Y_knn[indices[:, 1]]
r2_vals = r2_score(Y_knn, preds, multioutput="raw_values")

gene_idx_to_keep = np.where(r2_vals > 0.1)[0]
N_GENES = min(N_GENES, len(gene_idx_to_keep))
gene_names_to_keep = data_knn.var_names.values[gene_idx_to_keep]
gene_names_to_keep = gene_names_to_keep[np.argsort(-r2_vals[gene_idx_to_keep])]
r2_vals_sorted = -1 * np.sort(-r2_vals[gene_idx_to_keep])
if N_GENES < len(gene_names_to_keep):
    gene_names_to_keep = gene_names_to_keep[:N_GENES]
data = data[:, gene_names_to_keep]

n_samples_list = [
    Batch_list[0].shape[0],
    Batch_list[1].shape[0],
    Batch_list[2].shape[0],
    Batch_list[3].shape[0],
]
cumulative_sum = np.cumsum(n_samples_list)
cumulative_sum = np.insert(cumulative_sum, 0, 0)
view_idx = [
    np.arange(cumulative_sum[ii], cumulative_sum[ii + 1]) for ii in range(n_views)
]

sample_name=[151673,151674,151675,151676]

X_list = []
Y_list = []
for vv in range(n_views):
    curr_X = np.array(data[data.obs.batch == str(sample_name[vv])].obsm["spatial"])
    curr_Y = data[data.obs.batch == str(sample_name[vv])].X
    curr_Y = curr_Y.toarray()
    curr_X = scale_spatial_coords(curr_X)
    curr_Y = (curr_Y - curr_Y.mean(0)) / curr_Y.std(0)
    X_list.append(curr_X)
    Y_list.append(curr_Y)


X = np.concatenate(X_list)
Y = np.concatenate(Y_list)

device = "cuda" if torch.cuda.is_available() else "cpu"

n_outputs = Y.shape[1]

x = torch.from_numpy(X).float().clone()
y = torch.from_numpy(Y).float().clone()


data_dict = {
    "expression": {
        "spatial_coords": x,
        "outputs": y,
        "n_samples_list": n_samples_list,
    }
}

model = VariationalGPSA(
    data_dict,
    n_spatial_dims=n_spatial_dims,
    m_X_per_view=m_X_per_view,
    m_G=m_G,
    data_init=True,
    minmax_init=False,
    grid_init=False,
    n_latent_gps=N_LATENT_GPS,
    mean_function="identity_fixed",
    kernel_func_warp=rbf_kernel,
    kernel_func_data=rbf_kernel,
    # fixed_warp_kernel_variances=np.ones(n_views) * 1.,
    # fixed_warp_kernel_lengthscales=np.ones(n_views) * 10,
    fixed_view_idx=0,
).to(device)

view_idx, Ns, _, _ = model.create_view_idx_dict(data_dict)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)


def train(model, loss_fn, optimizer):
    model.train()
    # Forward pass
    G_means, G_samples, F_latent_samples, F_samples = model.forward(
        X_spatial={"expression": x}, view_idx=view_idx, Ns=Ns, S=5
    )
    # Compute loss
    loss = loss_fn(data_dict, F_samples)
    # Compute gradients and take optimizer step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item(), G_means


gene_idx = 0

pd.DataFrame(view_idx["expression"]).to_csv("/home/zhaoshangrui/xuxinyu/GPSA/DLPFC/results_12/view_idx_visium_sample3.csv")
pd.DataFrame(X).to_csv("/home/zhaoshangrui/xuxinyu/GPSA/DLPFC/results_12/X_visium_sample3.csv")
pd.DataFrame(Y).to_csv("/home/zhaoshangrui/xuxinyu/GPSA/DLPFC/results_12/Y_visium_sample3.csv")
data.write("/home/zhaoshangrui/xuxinyu/GPSA/DLPFC/results_12/data_visium_sample3.h5")

import time
start = time.time()
for t in range(N_EPOCHS):
    loss, G_means = train(model, model.loss_fn, optimizer)
    if t % PRINT_EVERY == 0:
        print("Iter: {0:<10} LL {1:1.3e}".format(t, -loss), flush=True)
        curr_aligned_coords = G_means["expression"].detach().numpy()
        pd.DataFrame(curr_aligned_coords).to_csv("/home/zhaoshangrui/xuxinyu/GPSA/DLPFC/results_12/aligned_coords_visium_sample3.csv")
end = time.time()
print('time',end-start)
