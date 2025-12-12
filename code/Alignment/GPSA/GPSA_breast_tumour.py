# -*- coding: utf-8 -*-
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import scanpy as sc
import anndata

sys.path.append("/home/zhaoshangrui/xuxinyu/PASTE_1/files")
from gpsa import VariationalGPSA, rbf_kernel
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score


def scale_spatial_coords(X, max_val=10.0):
    X = X - X.min(0)
    X = X / X.max(0)
    return X * max_val


N_GENES = 20
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


def process_data(adata, n_top_genes=2000):
    adata.var_names_make_unique()
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
    # Preprocess slices
    sc.pp.filter_genes(adata, min_counts = 15)
    sc.pp.filter_cells(adata, min_counts = 100)   
    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata, flavor="seurat", n_top_genes=n_top_genes, subset=True
    )
    return adata

def load_slices(data_dir, slice_names):
    slices = []  
    for slice_name in slice_names:
        slice_i = sc.read_csv(data_dir + slice_name + ".csv")
        slice_i_coor = np.genfromtxt(data_dir + slice_name + "_coor.csv", delimiter = ',')
        slice_i.obsm['spatial'] = slice_i_coor
        slices.append(slice_i)
    return slices

slice_names =["stahl_bc_slice1", "stahl_bc_slice2", "stahl_bc_slice3", "stahl_bc_slice4"]
data_dir = "/home/zhaoshangrui/xuxinyu/datasets/breast_tumour_data/breast_tumour_ST_data/"
data_slice1, data_slice2, data_slice3, data_slice4 = load_slices(data_dir, slice_names)

process_data(data_slice1,n_top_genes=3000)
process_data(data_slice2,n_top_genes=3000)
process_data(data_slice3,n_top_genes=3000)
process_data(data_slice4,n_top_genes=3000)

# # Set rotation angle
# angle = np.deg2rad(135)   # Convert degrees to radians

# slice3_coords = data_slice3.obsm["spatial"].copy()

# # Construct rotation matrix (135 degrees counter-clockwise)
# R = np.array([
#     [np.cos(angle), -np.sin(angle)],
#     [np.sin(angle),  np.cos(angle)]
# ])
# slice3_coords = slice3_coords @ R 

# # Update coordinates
# data_slice3.obsm["spatial"] = slice3_coords

# angle = np.deg2rad(180)
# slice4_coords = data_slice4.obsm["spatial"].copy()
# R = np.array([
#     [np.cos(angle), -np.sin(angle)],
#     [np.sin(angle),  np.cos(angle)]
# ])
# slice4_coords = slice4_coords @ R
# data_slice4.obsm["spatial"] = slice4_coords

data = anndata.AnnData.concatenate(data_slice1, data_slice2, data_slice3, data_slice4)

shared_gene_names = data.var.index.values
data_knn = data_slice1[:, shared_gene_names]
X_knn = data_knn.obsm["spatial"]
Y_knn = data_knn.X
Y_knn = (Y_knn - Y_knn.mean(0)) / Y_knn.std(0)
knn = KNeighborsRegressor(n_neighbors=10, weights="uniform").fit(X_knn, Y_knn)
preds = knn.predict(X_knn)

r2_vals = r2_score(Y_knn, preds, multioutput="raw_values")

gene_idx_to_keep = np.where(r2_vals > 0.3)[0]
N_GENES = min(N_GENES, len(gene_idx_to_keep))
gene_names_to_keep = data_knn.var.index.values[gene_idx_to_keep]
gene_names_to_keep = gene_names_to_keep[np.argsort(-r2_vals[gene_idx_to_keep])]
r2_vals_sorted = -1 * np.sort(-r2_vals[gene_idx_to_keep])
if N_GENES < len(gene_names_to_keep):
    gene_names_to_keep = gene_names_to_keep[:N_GENES]
data = data[:, gene_names_to_keep]

n_samples_list = [
    data_slice1.shape[0],
    data_slice2.shape[0],
    data_slice3.shape[0],
    data_slice4.shape[0],
]
cumulative_sum = np.cumsum(n_samples_list)
cumulative_sum = np.insert(cumulative_sum, 0, 0)
view_idx = [
    np.arange(cumulative_sum[ii], cumulative_sum[ii + 1]) for ii in range(n_views)
]

X_list = []
Y_list = []
for vv in range(n_views):
    curr_X = np.array(data[data.obs.batch == str(vv)].obsm["spatial"])
    curr_Y = data[data.obs.batch == str(vv)].X
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

# Set up figure.
fig = plt.figure(figsize=(10, 5), facecolor="white", constrained_layout=True)
ax1 = fig.add_subplot(121, frameon=False)
ax2 = fig.add_subplot(122, frameon=False)
plt.show(block=False)


gene_idx = 0

for t in range(N_EPOCHS):
    loss, G_means = train(model, model.loss_fn, optimizer)

    if t % PRINT_EVERY == 0:
        print("Iter: {0:<10} LL {1:1.3e}".format(t, -loss), flush=True)
        ax1.cla()
        ax2.cla()

        curr_aligned_coords = G_means["expression"].detach().numpy()
        curr_aligned_coords_slice1 = curr_aligned_coords[view_idx["expression"][0]]
        curr_aligned_coords_slice2 = curr_aligned_coords[view_idx["expression"][1]]
        curr_aligned_coords_slice3 = curr_aligned_coords[view_idx["expression"][2]]

        for vv, curr_X in enumerate(X_list):
            ax1.scatter(curr_X[:, 0], curr_X[:, 1], alpha=0.5)

            if vv == fixed_view_idx:
                ax1.scatter(curr_X[:, 0], curr_X[:, 1], alpha=0.5, color="black")
                ax2.scatter(
                    X_list[vv][:, 0], X_list[vv][:, 1], alpha=0.5, color="black"
                )

            ax2.scatter(
                curr_aligned_coords[view_idx["expression"][vv]][:, 0],
                curr_aligned_coords[view_idx["expression"][vv]][:, 1],
                alpha=0.5,
            )

        plt.draw()
        plt.savefig("/home/zhaoshangrui/xuxinyu/GPSA/breast_tumour/result_new_1/st_alignment.png")
        plt.pause(1 / 60.0)

        pd.DataFrame(curr_aligned_coords).to_csv("/home/zhaoshangrui/xuxinyu/GPSA/breast_tumour/result_new_1/aligned_coords_st.csv")
        pd.DataFrame(view_idx["expression"]).to_csv("/home/zhaoshangrui/xuxinyu/GPSA/breast_tumour/result_new_1/view_idx_st.csv")
        pd.DataFrame(X).to_csv("/home/zhaoshangrui/xuxinyu/GPSA/breast_tumour/result_new_1/X_st.csv")
        pd.DataFrame(Y).to_csv("/home/zhaoshangrui/xuxinyu/GPSA/breast_tumour/result_new_1/Y_st.csv")
        data.write("/home/zhaoshangrui/xuxinyu/GPSA/breast_tumour/result_new_1/data_st.h5")

        if model.n_latent_gps["expression"] is not None:
            curr_W = model.W_dict["expression"].detach().numpy()
            pd.DataFrame(curr_W).to_csv("/home/zhaoshangrui/xuxinyu/GPSA/breast_tumour/result_new_1/W_st.csv")


plt.close()
