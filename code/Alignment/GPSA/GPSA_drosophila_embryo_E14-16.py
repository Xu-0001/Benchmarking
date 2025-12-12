# -*- coding: utf-8 -*-
import torch
import numpy as np
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append("/home/zhaoshangrui/xuxinyu/PASTE_1/files")
from gpsa import VariationalGPSA, rbf_kernel

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score


def scale_spatial_coords(X, max_val=10.0):
    X = X - X.min(0)
    X = X / X.max(0)
    return X * max_val

def process_data(adata, n_top_genes=2000):
    adata.var_names_make_unique()
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata, flavor="seurat", n_top_genes=n_top_genes, subset=True
    )
    return adata


N_GENES = 20
N_SAMPLES = None
N_LAYERS = 2
fixed_view_idx = 0

n_spatial_dims = 2
n_views = 2
m_G = 200
m_X_per_view = 200

N_LATENT_GPS = {"expression": None}

N_EPOCHS = 5000
PRINT_EVERY = 25

#spatial data
adata_st_raw = sc.read_h5ad("/home/zhaoshangrui/xuxinyu/datasets/Drosophila_embryo_dataset_profiled_by_Stereo-seq/E14-16h_a_count_normal_stereoseq.h5ad")
adata_st_raw.X = adata_st_raw.layers['raw_counts']
adata_st_raw.obsm['spatial'] = adata_st_raw.obsm['spatial'][:, :2]

slice_all = sorted(list(set(adata_st_raw.obs['slice_ID'].values)))

slices_ID = ['E14-16h_a_S01','E14-16h_a_S02','E14-16h_a_S03','E14-16h_a_S04','E14-16h_a_S05','E14-16h_a_S06','E14-16h_a_S07','E14-16h_a_S08',
 'E14-16h_a_S09','E14-16h_a_S10','E14-16h_a_S11','E14-16h_a_S12','E14-16h_a_S13','E14-16h_a_S14','E14-16h_a_S15','E14-16h_a_S16']  

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
    process_data(adata_st_i, n_top_genes=5000)
    adata_st_list_raw.append(adata_st_i.copy())
        
data = adata_st_list_raw[7].concatenate(adata_st_list_raw[8])
shared_gene_names = data.var_names
data_knn = data[data.obs['slice_ID']==slices_ID[7]][:, shared_gene_names]
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
if N_GENES < len(gene_names_to_keep):
    gene_names_to_keep = gene_names_to_keep[:N_GENES]
data = data[:, gene_names_to_keep]

n_samples_list = [
    data[data.obs['slice_ID']==slices_ID[7]].shape[0],
    data[data.obs['slice_ID']==slices_ID[8]].shape[0],
]
cumulative_sum = np.cumsum(n_samples_list)
cumulative_sum = np.insert(cumulative_sum, 0, 0)
view_idx = [
    np.arange(cumulative_sum[ii], cumulative_sum[ii + 1]) for ii in range(n_views)
]


data.obs['batch'] = data.obs['slice_ID']
X1 = np.array(data[data.obs.batch == slices_ID[7]].obsm["spatial"])
X2 = np.array(data[data.obs.batch == slices_ID[8]].obsm["spatial"])
Y1 = data[data.obs.batch == slices_ID[7]].X
Y2 = data[data.obs.batch == slices_ID[8]].X
X1 = scale_spatial_coords(X1)
X2 = scale_spatial_coords(X2)
Y1 = (Y1 - Y1.mean(0)) / Y1.std(0)
Y2 = (Y2 - Y2.mean(0)) / Y2.std(0)

X_list = [X1,X2]
Y_list = [Y1,Y2]

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
import time
start_time = time.time() 
for t in range(N_EPOCHS):
    loss, G_means = train(model, model.loss_fn, optimizer)
    if t % PRINT_EVERY == 0:
        print("Iter: {0:<10} LL {1:1.3e}".format(t, -loss), flush=True)
        curr_aligned_coords = G_means["expression"].detach().numpy()
        pd.DataFrame(curr_aligned_coords).to_csv("/home/zhaoshangrui/xuxinyu/GPSA/Drosophila_embryoE14-16/result/aligned_coords_st_new89.csv")
        pd.DataFrame(view_idx["expression"]).to_csv("/home/zhaoshangrui/xuxinyu/GPSA/Drosophila_embryoE14-16/result/view_idx_st_new89.csv")
        pd.DataFrame(X).to_csv("/home/zhaoshangrui/xuxinyu/GPSA/Drosophila_embryoE14-16/result/X_st_new89.csv")
        pd.DataFrame(Y).to_csv("/home/zhaoshangrui/xuxinyu/GPSA/Drosophila_embryoE14-16/result/Y_st_new89.csv")
        data.write("/home/zhaoshangrui/xuxinyu/GPSA/Drosophila_embryoE14-16/result/data_st_new89.h5")
        if model.n_latent_gps["expression"] is not None:
            curr_W = model.W_dict["expression"].detach().numpy()
            pd.DataFrame(curr_W).to_csv("/home/zhaoshangrui/xuxinyu/GPSA/Drosophila_embryoE14-16/result/W_st_new89.csv")
plt.close()
end_time = time.time()
elapsed_time = end_time - start_time
print(f"实验运行时间: {elapsed_time:.2f} 秒")
