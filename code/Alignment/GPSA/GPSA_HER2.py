# -*- coding: utf-8 -*-
import torch
import numpy as np
import pandas as pd
import sys
import scanpy as sc
import anndata

sys.path.append("/home/zhaoshangrui/xuxinyu/PASTE_1/files")
from gpsa import VariationalGPSA, matern12_kernel, rbf_kernel
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
from sklearn.metrics import r2_score

import time


def scale_spatial_coords(X, max_val=10.0):
    X = X - X.min(0)
    X = X / X.max(0)
    return X * max_val

N_GENES = 20
N_SAMPLES = None
fixed_view_idx = 1

n_spatial_dims = 2
n_views = 6
m_G = 200
m_X_per_view = 200

N_LATENT_GPS = {"expression": None}

N_EPOCHS = 5000
PRINT_EVERY = 25

#spatial data
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
        adata_st_i = anndata.AnnData(X=st_count.values)
        adata_st_i.obs.index = st_count.index
        adata_st_i.obsm['spatial'] = np.array(list(map(lambda x: x.split('x'),adata_st_i.obs.index)),dtype=float)
        adata_st_i.obs = st_meta
        adata_st_i.var.index = st_count.columns
        sc.pp.filter_genes(adata_st_i, min_counts = 15)
        sc.pp.filter_cells(adata_st_i, min_counts = 100)
        sc.pp.highly_variable_genes(adata_st_i, flavor="seurat_v3", n_top_genes=5000)
        sc.pp.normalize_total(adata_st_i, target_sum=1e4)
        sc.pp.log1p(adata_st_i)
        adata_st_i = adata_st_i[:, adata_st_i.var['highly_variable']]
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
        patients[p][i] = adata_st_i       
        
slice_all = [patients['A'][0], patients['A'][1], patients['A'][2], patients['A'][3], patients['A'][4], patients['A'][5]]

slices_ID = ['A1','A2','A3','A4','A5','A6']  

data = anndata.AnnData.concatenate(patients['A'][0], patients['A'][1], patients['A'][2], patients['A'][3], patients['A'][4], patients['A'][5])
shared_gene_names = data.var_names
data_knn = slice_all[0][:, shared_gene_names]
X_knn = data_knn.obsm["spatial"]
Y_knn = data_knn.X
Y_knn = (Y_knn - Y_knn.mean(0)) / Y_knn.std(0)
knn = KNeighborsRegressor(n_neighbors=10, weights="uniform").fit(X_knn, Y_knn)
preds = knn.predict(X_knn)

r2_vals = r2_score(Y_knn, preds, multioutput="raw_values")

gene_idx_to_keep = np.where(r2_vals > 0.3)[0]
N_GENES = min(N_GENES, len(gene_idx_to_keep))
gene_names_to_keep = data_knn.var_names.values[gene_idx_to_keep]
gene_names_to_keep = gene_names_to_keep[np.argsort(-r2_vals[gene_idx_to_keep])]
r2_vals_sorted = -1 * np.sort(-r2_vals[gene_idx_to_keep])
if N_GENES < len(gene_names_to_keep):
    gene_names_to_keep = gene_names_to_keep[:N_GENES]
data = data[:, gene_names_to_keep]

n_samples_list = [
    slice_all[0].shape[0],
    slice_all[1].shape[0],
    slice_all[2].shape[0],
    slice_all[3].shape[0],
    slice_all[4].shape[0],
    slice_all[5].shape[0],
]
cumulative_sum = np.cumsum(n_samples_list)
cumulative_sum = np.insert(cumulative_sum, 0, 0)
view_idx = [
    np.arange(cumulative_sum[ii], cumulative_sum[ii + 1]) for ii in range(n_views)
]

X_list = []
Y_list = []
data.obs['batch'] = data.obs['batch'].cat.codes
data.obs['batch'] = data.obs['batch'].astype('category')

for vv in range(n_views):
    curr_X = np.array(data[data.obs.batch == vv].obsm["spatial"])
    curr_Y = data[data.obs.batch == vv].X
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
    fixed_view_idx=fixed_view_idx,
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
start_time = time.time()
for t in range(N_EPOCHS):
    loss, G_means = train(model, model.loss_fn, optimizer)
    if t % PRINT_EVERY == 0:
        print("Iter: {0:<10} LL {1:1.3e}".format(t, -loss), flush=True)
        curr_aligned_coords = G_means["expression"].detach().numpy()
        pd.DataFrame(curr_aligned_coords).to_csv("/home/zhaoshangrui/xuxinyu/GPSA/HER2/metrice_new/aligned_coords_st.csv")
        pd.DataFrame(view_idx["expression"]).to_csv("/home/zhaoshangrui/xuxinyu/GPSA/HER2/metrice_new/view_idx_st.csv")
        pd.DataFrame(X).to_csv("/home/zhaoshangrui/xuxinyu/GPSA/HER2/metrice_new/X_st.csv")
        pd.DataFrame(Y).to_csv("/home/zhaoshangrui/xuxinyu/GPSA/HER2/metrice_new/Y_st.csv")
        data.write("/home/zhaoshangrui/xuxinyu/GPSA/HER2/metrice_new/data_st.h5")
        if model.n_latent_gps["expression"] is not None:
            curr_W = model.W_dict["expression"].detach().numpy()
            pd.DataFrame(curr_W).to_csv("/home/zhaoshangrui/xuxinyu/GPSA/HER2/metrice_new/W_st.csv")
end_time = time.time()
elapsed_time = end_time - start_time
print(f"实验运行时间: {elapsed_time:.2f} 秒") 

