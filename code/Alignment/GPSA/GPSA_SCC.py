# -*- coding: utf-8 -*-
import sys
import torch
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

sys.path.append("/home/zhaoshangrui/xuxinyu/PASTE_1/files")
from gpsa import VariationalGPSA, matern12_kernel, rbf_kernel
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
from sklearn.metrics import r2_score


def scale_spatial_coords(X, max_val=10.0):
    X = X - X.min(0)
    X = X / X.max(0)
    return X * max_val


N_GENES = 20
N_SAMPLES = None
fixed_view_idx = 1

n_spatial_dims = 2
n_views = 3
m_G = 200
m_X_per_view = 200

N_LATENT_GPS = {"expression": None}

N_EPOCHS = 5000
PRINT_EVERY = 25

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

  
# patient_2 
Batch_list = patients["patient_2"]        
adata = ad.concat(patients["patient_2"],label="batch")
shared_gene_names = adata.var_names
data_knn = Batch_list[0][:, shared_gene_names]
X_knn = data_knn.obsm["spatial"]
Y_knn = data_knn.X
Y_knn = (Y_knn - Y_knn.mean(0)) / Y_knn.std(0)
knn = KNeighborsRegressor(n_neighbors=10, weights="uniform").fit(X_knn, Y_knn)
preds = knn.predict(X_knn)

r2_vals = r2_score(Y_knn, preds, multioutput="raw_values")

gene_idx_to_keep = np.where(r2_vals > 0.65)[0]
N_GENES = min(N_GENES, len(gene_idx_to_keep))
gene_names_to_keep = data_knn.var_names.values[gene_idx_to_keep]
gene_names_to_keep = gene_names_to_keep[np.argsort(-r2_vals[gene_idx_to_keep])]
r2_vals_sorted = -1 * np.sort(-r2_vals[gene_idx_to_keep])
if N_GENES < len(gene_names_to_keep):
    gene_names_to_keep = gene_names_to_keep[:N_GENES]
adata = adata[:, gene_names_to_keep]

n_samples_list = [
    Batch_list[0].shape[0],
    Batch_list[1].shape[0],
    Batch_list[2].shape[0],
]
cumulative_sum = np.cumsum(n_samples_list)
cumulative_sum = np.insert(cumulative_sum, 0, 0)
view_idx = [
    np.arange(cumulative_sum[ii], cumulative_sum[ii + 1]) for ii in range(n_views)
]

X_list = []
Y_list = []
for vv in range(n_views):
    curr_X = np.array(adata[adata.obs.batch == str(vv)].obsm["spatial"])
    curr_Y = adata[adata.obs.batch == str(vv)].X
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
import time       
start_time = time.time() 
for t in range(N_EPOCHS):
    loss, G_means = train(model, model.loss_fn, optimizer)
    if t % PRINT_EVERY == 0:
        print("Iter: {0:<10} LL {1:1.3e}".format(t, -loss), flush=True)
        curr_aligned_coords = G_means["expression"].detach().numpy()
        pd.DataFrame(curr_aligned_coords).to_csv("/home/zhaoshangrui/xuxinyu/GPSA/SCC/result/aligned_coords_st_patient2.csv")
        pd.DataFrame(view_idx["expression"]).to_csv("/home/zhaoshangrui/xuxinyu/GPSA/SCC/result/view_idx_st_patient2.csv")
        pd.DataFrame(X).to_csv("/home/zhaoshangrui/xuxinyu/GPSA/SCC/result/X_st_patient2.csv")
        pd.DataFrame(Y).to_csv("/home/zhaoshangrui/xuxinyu/GPSA/SCC/result/Y_st_patient2.csv")
        adata.write("/home/zhaoshangrui/xuxinyu/GPSA/SCC/result/data_st_patient2.h5")
        if model.n_latent_gps["expression"] is not None:
            curr_W = model.W_dict["expression"].detach().numpy()
            pd.DataFrame(curr_W).to_csv("/home/zhaoshangrui/xuxinyu/GPSA/SCC/result/W_st_patient2.csv")
end_time = time.time()
elapsed_time = end_time - start_time
print(f"实验运行时间: {elapsed_time:.2f} 秒") 
