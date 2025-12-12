# -*- coding: utf-8 -*-
import paste as pst
import scanpy as sc
from typing import Optional, List, Union
import numpy as np

import sys
sys.path.append("/home/zhaoshangrui/xuxinyu/PASTE_2")
from paste2.PASTE2 import partial_pairwise_align
from paste2.projection import partial_stack_slices_pairwise
from paste2.model_selection import select_overlap_fraction

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import anndata as ad
import scanpy as sc
from typing import Dict, List, Tuple, Optional

# 假设这些函数和类已经在其他地方定义
from kernel_functions import rbf_kernel  # 假设这是一个自定义的RBF核函数
from variational_gpsa import VariationalGPSA  # 假设这是变分GPSA模型


def scale_spatial_coords(X: np.ndarray, max_val: float = 10.0) -> np.ndarray:
    X = X - X.min(axis=0)
    X = X / X.max(axis=0)
    return X * max_val    
    
def load_and_preprocess_data(
    slice_files: List[str],
    n_views: int = 2,
    max_spatial_val: float = 10.0,
    device: str = "cpu"
) -> Tuple[Dict, torch.Tensor, torch.Tensor, List[np.ndarray], Dict]:
    """
    加载并预处理空间数据切片
    
    Args:
        slice_files: 切片数据文件路径列表
        n_views: 视图数（切片数）
        max_spatial_val: 空间坐标缩放的最大值
        device: 计算设备 ('cpu' 或 'cuda')
        
    Returns:
        元组包含:
            - 数据字典
            - 空间坐标张量
            - 表达矩阵张量
            - 视图索引列表
            - 合并后的anndata对象
    """
    # 读取数据切片
    data = sc.read_h5ad(self.spatial_file)
    
    # 提取空间坐标和表达矩阵
    X_list = []
    Y_list = []
    n_samples_list = []
    
    for i in range(n_views):
        batch_data = data[data.obs.batch == str(i)]
        n_samples = batch_data.shape[0]
        n_samples_list.append(n_samples)
        
        # 提取并缩放空间坐标
        X = np.array(batch_data.obsm["spatial"])
        X = scale_spatial_coords(X, max_spatial_val)
        X_list.append(X)
        
        # 提取并标准化表达矩阵
        Y = np.array(batch_data.X)
        Y = (Y - Y.mean(axis=0)) / Y.std(axis=0)
        Y_list.append(Y)
    
    # 合并所有视图的数据
    X = np.concatenate(X_list)
    Y = np.concatenate(Y_list)
    
    # 创建视图索引
    view_idx = [
        np.arange(X_list[0].shape[0]),
        np.arange(X_list[0].shape[0], X_list[0].shape[0] + X_list[1].shape[0]),
    ]
    
    # 转换为PyTorch张量
    x = torch.from_numpy(X).float().clone().to(device)
    y = torch.from_numpy(Y).float().clone().to(device)
    
    # 创建数据字典
    data_dict = {
        "expression": {
            "spatial_coords": x,
            "outputs": y,
            "n_samples_list": n_samples_list,
        }
    }
    
    return data_dict, x, y, view_idx, data

def initialize_model(
    data_dict: Dict,
    n_spatial_dims: int = 2,
    m_G: int = 100,
    m_X_per_view: int = 100,
    n_latent_gps: Dict[str, Optional[int]] = {"expression": None},
    device: str = "cpu"
) -> Tuple[VariationalGPSA, Dict, List]:
    """
    初始化VariationalGPSA模型
    
    Args:
        data_dict: 数据字典
        n_spatial_dims: 空间维度数
        m_G: 全局潜在点数量
        m_X_per_view: 每个视图的局部潜在点数量
        n_latent_gps: 每个视图的潜在GPS数量
        device: 计算设备
        
    Returns:
        元组包含:
            - 初始化的模型
            - 视图索引字典
            - 样本数量列表
    """
    model = VariationalGPSA(
        data_dict,
        n_spatial_dims=n_spatial_dims,
        m_X_per_view=m_X_per_view,
        m_G=m_G,
        data_init=True,
        minmax_init=False,
        grid_init=False,
        n_latent_gps=n_latent_gps,
        mean_function="identity_fixed",
        kernel_func_warp=rbf_kernel,
        kernel_func_data=rbf_kernel,
        fixed_warp_kernel_variances=np.ones(len(data_dict["expression"]["n_samples_list"])) * 1e-3,
        fixed_view_idx=0,
    ).to(device)
    
    # 创建视图索引字典
    view_idx, Ns, _, _ = model.create_view_idx_dict(data_dict)
    
    return model, view_idx, Ns

def train_model(
    model: VariationalGPSA,
    data_dict: Dict,
    x: torch.Tensor,
    y: torch.Tensor,
    view_idx: Dict,
    Ns: List,
    n_epochs: int = 6000,
    print_every: int = 100,
    save_path: str = "./results/",
    data: Optional[ad.AnnData] = None
) -> Dict[str, np.ndarray]:
    """
    训练VariationalGPSA模型
    
    Args:
        model: 初始化的模型
        data_dict: 数据字典
        x: 空间坐标张量
        y: 表达矩阵张量
        view_idx: 视图索引字典
        Ns: 样本数量列表
        n_epochs: 训练轮数
        print_every: 打印频率
        save_path: 结果保存路径
        data: 合并后的anndata对象（用于保存）
        
    Returns:
        训练过程中的对齐坐标
    """
    # 创建保存目录
    import os
    os.makedirs(save_path, exist_ok=True)
    
    # 设置优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    
    # 训练函数
    def train_step(model, loss_fn, optimizer):
        model.train()
        # 前向传播
        G_means, G_samples, F_latent_samples, F_samples = model.forward(
            X_spatial={"expression": x}, view_idx=view_idx, Ns=Ns, S=5
        )
        # 计算损失
        loss = loss_fn(data_dict, F_samples)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item(), G_means
    
    # 训练循环
    for t in range(n_epochs):
        loss, G_means = train_step(model, model.loss_fn, optimizer)
        
        if t % print_every == 0 or t == n_epochs - 1:
            print(f"迭代: {t:<10} 负对数似然: {abs(loss):1.3e}")
            curr_aligned_coords = G_means["expression"].detach().cpu().numpy()
            # 保存当前结果
            pd.DataFrame(curr_aligned_coords).to_csv(f"{save_path}/aligned_coords_epoch_{t}.csv")
            pd.DataFrame(view_idx["expression"]).to_csv(f"{save_path}/view_idx.csv")
            pd.DataFrame(x.cpu().numpy()).to_csv(f"{save_path}/X.csv")
            pd.DataFrame(y.cpu().numpy()).to_csv(f"{save_path}/Y.csv")
            
            if data is not None:
                data.write(f"{save_path}/data.h5ad")
            
            if model.n_latent_gps["expression"] is not None:
                curr_W = model.W_dict["expression"].detach().cpu().numpy()
                pd.DataFrame(curr_W).to_csv(f"{save_path}/W_epoch_{t}.csv")
    return {"aligned_coords": curr_aligned_coords_history}

def get_model_results(
    model: VariationalGPSA,
    x: torch.Tensor,
    view_idx: Dict,
    Ns: List
) -> Dict[str, np.ndarray]:
    """
    获取模型训练结果
    
    Args:
        model: 训练好的模型
        x: 空间坐标张量
        view_idx: 视图索引字典
        Ns: 样本数量列表
        
    Returns:
        包含各种结果的字典
    """
    with torch.no_grad():
        G_means, _, _, _ = model.forward(
            X_spatial={"expression": x}, view_idx=view_idx, Ns=Ns, S=1
        )
        
    return {
        "aligned_coords": G_means["expression"].detach().cpu().numpy(),
        "original_coords": x.cpu().numpy(),
        "view_idx": view_idx["expression"]
    }

from pathlib import Path

    

        
