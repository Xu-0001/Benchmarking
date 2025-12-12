# -*- coding: utf-8 -*-
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import scanpy as sc
import anndata

from gpsa import VariationalGPSA, rbf_kernel

device = "cuda" if torch.cuda.is_available() else "cpu"


def scale_spatial_coords(X, max_val=10.0):
    X = X - X.min(0)
    X = X / X.max(0)
    return X * max_val


N_SAMPLES = 4000

n_spatial_dims = 2
n_views = 2
m_G = 100  # 200
m_X_per_view = 100  # 200

N_LATENT_GPS = {"expression": None}

N_EPOCHS = 6_000
PRINT_EVERY = 100

data_slice1 = sc.read_h5ad(".../data_slice1.h5ad")
data_slice2 = sc.read_h5ad(".../data_slice2.h5ad")

all_slices = anndata.concat([data_slice1, data_slice2])
data = data_slice1.concatenate(data_slice2)

n_samples_list = [data[data.obs.batch == str(ii)].shape[0] for ii in range(n_views)]

X1 = np.array(data[data.obs.batch == "0"].obsm["spatial"])
X2 = np.array(data[data.obs.batch == "1"].obsm["spatial"])
Y1 = np.array(data[data.obs.batch == "0"].X)
Y2 = np.array(data[data.obs.batch == "1"].X)

Y1 = (Y1 - Y1.mean(0)) / Y1.std(0)
Y2 = (Y2 - Y2.mean(0)) / Y2.std(0)

X = np.concatenate([X1, X2])
Y = np.concatenate([Y1, Y2])

view_idx = [
    np.arange(X1.shape[0]),
    np.arange(X1.shape[0], X1.shape[0] + X2.shape[0]),
]

x = torch.from_numpy(X).float().clone().to(device)
y = torch.from_numpy(Y).float().clone().to(device)


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
    fixed_warp_kernel_variances=np.ones(n_views) * 1e-3,
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


# Set up figure.
fig = plt.figure(figsize=(10, 5), facecolor="white", constrained_layout=True)
ax1 = fig.add_subplot(121, frameon=False)
ax2 = fig.add_subplot(122, frameon=False)
ax1.invert_yaxis()
ax2.invert_yaxis()
plt.show(block=False)

import time
start_time = time.time()
for t in range(N_EPOCHS):
    loss, G_means = train(model, model.loss_fn, optimizer)
    if t % PRINT_EVERY == 0 or t == N_EPOCHS - 1:
        print("Iter: {0:<10} LL {1:1.3e}".format(t, -loss))
        ax1.cla()
        ax2.cla()
        curr_aligned_coords = G_means["expression"].detach().numpy()
        curr_aligned_coords_slice1 = curr_aligned_coords[view_idx["expression"][0]]
        curr_aligned_coords_slice2 = curr_aligned_coords[view_idx["expression"][1]]
        ax1.scatter(X1[:, 0], X1[:, 1], alpha=0.3)
        ax1.scatter(X2[:, 0], X2[:, 1], alpha=0.3)
        ax2.scatter(X1[:, 0], X1[:, 1], alpha=0.3)
        ax2.scatter(
            curr_aligned_coords_slice2[:, 0],
            curr_aligned_coords_slice2[:, 1],
            alpha=0.3,
        )
        ax1.invert_yaxis()
        ax2.invert_yaxis()
        plt.draw()
        plt.savefig(".../alignment.png")
        plt.pause(1 / 60.0)
        pd.DataFrame(curr_aligned_coords).to_csv(".../aligned_coords.csv")
        pd.DataFrame(view_idx["expression"]).to_csv(".../view_idx.csv")
        pd.DataFrame(X).to_csv(".../X.csv")
        pd.DataFrame(Y).to_csv(".../Y.csv")
        data.write(".../data.h5")
        if model.n_latent_gps["expression"] is not None:
            curr_W = model.W_dict["expression"].detach().numpy()
            pd.DataFrame(curr_W).to_csv(".../W.csv")
plt.close()
end_time = time.time()
elapsed_time = end_time - start_time
print(f"实验运行时间: {elapsed_time:.2f} 秒")