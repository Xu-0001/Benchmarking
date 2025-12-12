# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import sys
import pickle
import os
import time as tm
from functools import partial
import scipy.stats as st
from scipy.stats import wasserstein_distance
import scipy.stats
import copy
from sklearn.model_selection import KFold
import pandas as pd
import multiprocessing
import matplotlib as mpl
import matplotlib.pyplot as plt
import scanpy as sc
import warnings
warnings.filterwarnings('ignore')

from scipy.spatial import distance_matrix
from sklearn.metrics import matthews_corrcoef
from scipy import stats

from os.path import join

from scipy.spatial.distance import cdist
import h5py
from scipy.stats import spearmanr

import sys
from stPlus import *

from typing import Dict, List, Optional, Union

import paste as pst

import sys
sys.path.append("/home/zhaoshangrui/xuxinyu/PASTE_2")
from paste2.PASTE2 import partial_pairwise_align
from paste2.projection import partial_stack_slices_pairwise
from paste2.model_selection import select_overlap_fraction

import sys
sys.path.append("/home/zhaoshangrui/xuxinyu/GPSA")
from gpsa.preprocess import load_and_preprocess_data, initialize_model, train_model

import anndata as ad

import STAligner

import torch
used_device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

import STitch3D

from pathlib import Path


class GenePrediction:
    def __init__(self, RNA_path, Spatial_path, location_path, device = None, train_list = None, test_list = None, outdir = None, modes = 'cells', annotate = None, CellTypeAnnotate = None):
        """
            @author: wen zhang
            This function integrates spatial and scRNA-seq data to predictes the expression of the spatially unmeasured genes from the scRNA-seq data.
            
            A minimal example usage:
            Assume we have (1) scRNA-seq data count file named RNA_path
            (2) spatial transcriptomics count data file named Spatial_path
            (3) spatial spot coordinate file named location_path
            (4) gene list for integrations names train_list
            (5) gene list for prediction names test_list
            
            >>> import Benchmarking.SpatialGenes as SpatialGenes
            >>> test = SpatialGenes.GenePrediction(RNA_path, Spatial_path, location_path, train_list = train_list, test_list = test_list, outdir = outdir)
            >>> Methods = ['SpaGE','novoSpaRc','SpaOTsc','gimVI','Tangram_image','Seurat','LIGER']
            >>> Result = test.Imputing(Methods)
            
            Parameters
            -------
            RNA_path : str
            scRNA-seq data count file with Tab-delimited (genes X cells, each row is a gene and each col is a cell).
            
            Spatial_path : str
            spatial transcriptomics count data file with Tab-delimited (spots X genes, each col is a gene. Please note that the file has no index).
            
            location_path : str
            spatial spot coordinate file name with Tab-delimited (each col is a spot coordinate. Please note that the file has no index).
            default: None. It is necessary when you use SpaOTsc or novoSpaRc to integrate datasets.
            
            
            device : str
            Option,  [None,'GPU'], defaults to None
            
            train_list : list
            genes for integrations, Please note it must be a list.
            
            test_list : list
            genes for prediction, Please note it must be a list.
            
            outdir : str
            Outfile directory
            
            modes : str
            Only for Tangram. The default mapping mode is mode='cells',Alternatively, one can specify mode='clusters' which averages the single cells beloning to the same cluster (pass annotations via cluster_label). This is faster, and is our chioce when scRNAseq and spatial data come from different species 
           
            annoatet : str
            annotate for scRNA-seq data, if not None, you must be set CellTypeAnnotate labels for tangram.
            
            CellTypeAnnotate : dataframe
            CellType for scRNA-seq data, you can set this parameter for tangram.
            
            """
        
        self.RNA_file = RNA_path
        self.Spatial_file = Spatial_path
        self.locations = np.loadtxt(location_path, skiprows=1)
        self.train_list = train_list
        self.test_list = test_list
        self.RNA_data_adata = sc.read(RNA_path, sep = "\t",first_column_names=True).T
        self.Spatial_data_adata = sc.read(Spatial_path, sep = "\t")
        self.device = device
        self.outdir = outdir
        self.annotate = annotate
        self.CellTypeAnnotate = CellTypeAnnotate
        self.modes = modes
    
    
    def paste_impute(
            self,
            alpha: float = 0.1,
            batch_key: str = "batch",
        ) -> List[sc.AnnData]:
        """Impute spatial data using PASTE method with flexible batch processing.
        
        Args:
            alpha: Alignment parameter for PASTE (default: 0.1)
            batch_key: Column name in obs containing batch info (default: "batch")          
        Returns:
            List of pairwise aligned AnnData objects
        """
        try:
            # Load data
            spatial_data = sc.read_h5ad(self.spatial_file)
            
            # Check if batch column exists
            if batch_key not in spatial_data.obs.columns:
                raise ValueError(f"Data does not contain '{batch_key}' column")
                
            # Get unique batches
            batches = np.sort(spatial_data.obs[batch_key].unique())
            
            # Apply batch limit if specified   
            if len(batches) < 2:
                raise ValueError("Need at least 2 batches for alignment")
                
            # Split data by batches
            batch_list = [spatial_data[spatial_data.obs[batch_key] == b].copy() for b in batches]
            
            # Pairwise alignment
            pis = []
            for i in range(len(batch_list) - 1):
                pi = pst.pairwise_align(batch_list[i], batch_list[i+1], alpha=alpha)
                pis.append(pi)   
                
            # Stack slices using computed alignment matrices
            new_slices = pst.stack_slices_pairwise(batch_list, pis) 
            return new_slices
        
        except FileNotFoundError:
            raise ValueError(f"Could not find file: {self.spatial_file}")
        except Exception as e:
            # Log the error or handle it appropriately
            raise RuntimeError(f"An error occurred during PASTE imputation: {str(e)}")
        
    
    def paste2_impute(
            self,
            alpha: float = 0.1,
            batch_key: str = "batch",
        ) -> List[sc.AnnData]:
        """Impute spatial data using PASTE2 method with flexible batch processing.
        
        Args:
            alpha: Alignment parameter for PASTE2 (default: 0.1)
            batch_key: Column name in obs containing batch info (default: "batch")          
        Returns:
            List of pairwise aligned AnnData objects
        """
        try:
            # Load data
            spatial_data = sc.read_h5ad(self.spatial_file)
            
            # Check if batch column exists
            if batch_key not in spatial_data.obs.columns:
                raise ValueError(f"Data does not contain '{batch_key}' column")
                
            # Get unique batches
            batches = np.sort(spatial_data.obs[batch_key].unique())
            
            # Apply batch limit if specified   
            if len(batches) < 2:
                raise ValueError("Need at least 2 batches for alignment")
                
            # Split data by batches
            batch_list = [spatial_data[spatial_data.obs[batch_key] == b].copy() for b in batches]
            
            # Pairwise alignment
            pis = []
            for i in range(len(batch_list) - 1):
                s_pred = select_overlap_fraction(batch_list[i], batch_list[i+1], alpha=alpha)
                pi = partial_pairwise_align(batch_list[i], batch_list[i+1], s=s_pred)
                pis.append(pi)   
                
            # Stack slices using computed alignment matrices
            new_slices = partial_stack_slices_pairwise(batch_list, pis) 
            return new_slices
        
        except FileNotFoundError:
            raise ValueError(f"Could not find file: {self.spatial_file}")
        except Exception as e:
            # Log the error or handle it appropriately
            raise RuntimeError(f"An error occurred during PASTE2 imputation: {str(e)}")
     
            
    def gpsa_impute(
        slice_files: List[Union[str, Path]],
        n_spatial_dims: int = 2,
        n_views: int = 2,
        m_G: int = 100,
        m_X_per_view: int = 100,
        n_latent_gps: Optional[Dict[str, Optional[int]]] = None,
        max_spatial_val: float = 10.0,
        n_epochs: int = 6000,
        print_every: int = 100,
        save_path: Union[str, Path] = "./results/",
        device: str = "cpu",
        save_intermediate: bool = True
    ) -> sc.AnnData:
        """
        The complete pipeline for spatial data alignment.
        
        Args:
            slice_files: List of paths to slice data files (supports str or Path objects).
            n_spatial_dims: Number of spatial dimensions (default 2D).
            n_views: Number of views/slices (default 2).
            m_G: Number of global latent points (default 100).
            m_X_per_view: Number of local latent points per view (default 100).
            n_latent_gps: Configuration of the number of latent GPS for each modality (default {"expression": None}).
            max_spatial_val: Maximum value for spatial coordinates (used for normalization, default 10.0).
            n_epochs: Number of training epochs (default 6000).
            print_every: Print the training information every n epochs (default 100).
            save_path: Path to save the results (default "./results/").
            device: Computing device ('cpu' or 'cuda', default 'cpu').
            save_intermediate: Whether to save intermediate results (default True).
            
        Returns:
            aligned_slice: AnnData object containing aligned coordinates and expression data.
        """
        # Set default parameters
        if n_latent_gps is None:
            n_latent_gps = {"expression": None}
        
        # Ensure the save path exists
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Load and preprocess data
        data_dict, x, y, view_idx_list, raw_data = load_and_preprocess_data(
            slice_files=slice_files,
            n_views=n_views,
            max_spatial_val=max_spatial_val,
            device=device
        )    
        
        view_idx = {"expression": np.concatenate(view_idx_list)}
         
        # Initialize the model
        model, _, Ns = initialize_model(
            data_dict=data_dict,
            n_spatial_dims=n_spatial_dims,
            m_G=m_G,
            m_X_per_view=m_X_per_view,
            n_latent_gps=n_latent_gps,
            device=device
        )
        
        # Train the model
        train_model(
            model=model,
            data_dict=data_dict,
            x=x,
            y=y,
            view_idx=view_idx,
            Ns=Ns,
            n_epochs=n_epochs,
            print_every=print_every,
            save_path=save_path if save_intermediate else None,
            data=raw_data
        )
        
        # Create the aligned AnnData object
        aligned_slice = raw_data.copy()
        # Forward pass to get the aligned coordinates
        G_means, _, _, _ = model.forward(X_spatial={"expression": x}, view_idx=view_idx, Ns=Ns, S=1)
        aligned_coords = G_means["expression"].detach().cpu().numpy()
        aligned_slice.obsm["spatial"] = aligned_coords
        aligned_slice.X = y.cpu().numpy()
        return aligned_slice  
    
    
    def staligner_impute(
            self,
            batch_key: str = "batch",
        ) -> sc.AnnData:
        """Impute spatial data using STAligner method with flexible batch processing.
        Args:
            batch_key: Column name in obs containing batch info (default: "batch")          
        Returns:
            List of pairwise aligned AnnData objects
        """
        # Load data
        spatial_data = sc.read_h5ad(self.spatial_file)
        
        # Check if batch column exists
        if batch_key not in spatial_data.obs.columns:
            raise ValueError(f"Data does not contain '{batch_key}' column")
            
        # Get unique batches
        batches = np.sort(spatial_data.obs[batch_key].unique())
        
        # Apply batch limit if specified   
        if len(batches) < 2:
            raise ValueError("Need at least 2 batches for alignment")
            
        # Split data by batches
        batch_list = [spatial_data[spatial_data.obs[batch_key] == b].copy() for b in batches]
        
        # STAligner integration
        adj_list = []
        for i in range(len(batch_list) - 1):
            STAligner.Cal_Spatial_Net(batch_list[i], rad_cutoff=0.2)
            adj_list.append(batch_list[i].uns['adj'])  
        
        adata_concat = ad.concat(batch_list, label="batch", keys=batches)
        
        adj_concat = np.asarray(adj_list[0].todense())
        for batch_id in range(1,len(batches)):
            adj_concat = scipy.linalg.block_diag(adj_concat, np.asarray(adj_list[batch_id].todense()))
        adata_concat.uns['edgeList'] = np.nonzero(adj_concat)
        adata_concat = STAligner.train_STAligner(adata_concat, verbose=True, knn_neigh = 100, device=used_device) #epochs = 1500,
        adata_concat.uns['edgeList'] = list(adata_concat.uns['edgeList'])
        return adata_concat
            
    
    def stitch3d_pipeline(
            self,
            batch_key: str = "batch",
            data_dir: str = "./data",
            output_dir: str = "./results",
            slice_dist_micron: List[float] = None
            ) -> sc.AnnData:
        """
        Impute spatial data using STAligner method with flexible batch processing.
        
        Args:
            batch_key: Column name in obs containing batch info
            data_dir: Directory containing input data files
            output_dir: Directory to save results
            slice_dist_micron: Distance parameters for slice alignment [x, y, z]
            return_concat: If True, return concatenated AnnData; else return list
            
        Returns:
            List of aligned AnnData objects (single element if concatenated)
        """
        # Load reference datasets
        ref_expr_path = os.path.join(data_dir, "expr_raw_counts_table.tsv")
        ref_meta_path = os.path.join(data_dir, "meta_table.tsv")
        
        count_ref = pd.read_csv(ref_expr_path, sep="\t", index_col=0)
        meta = pd.read_csv(ref_meta_path, sep="\t", index_col=0)
        
        adata_ref = ad.AnnData(X=count_ref.values)
        adata_ref.obs.index = count_ref.index
        adata_ref.var.index = count_ref.columns
        
        # Add metadata columns
        for col in meta.columns[:-1]:
            adata_ref.obs[col] = meta.loc[count_ref.index, col].values
            
        # Load spatial transcriptomics data
        spatial_data = sc.read_h5ad(self.spatial_file)
            
        # Get unique batches
        batches = np.sort(spatial_data.obs[batch_key].unique())
            
        # Split data by batches
        adata_st_list_raw = [spatial_data[spatial_data.obs[batch_key] == b].copy() for b in batches]
        
        # STitch3D alignment
        adata_st_list = STitch3D.utils.align_spots(adata_st_list_raw, plot=True)
            
        # Preprocess data
        adata_st, adata_basis = STitch3D.utils.preprocess(
            adata_st_list,
            adata_ref,
            sample_col="group",
            slice_dist_micron=slice_dist_micron,
            n_hvg_group=500
        )

        # Train model
        model = STitch3D.model.Model(adata_st, adata_basis)
        model.train()

        # Evaluate and save results
        os.makedirs(output_dir, exist_ok=True)
        result = model.eval(adata_st_list_raw, save=True, output_path=output_dir)
        
        return model.adata_st

    def Imputing(self, need_tools):
        if "SpaGE" in need_tools:
            result_SpaGE = self.SpaGE_impute()
            if not os.path.exists(self.outdir):
                os.mkdir(self.outdir)
            result_SpaGE.to_csv(self.outdir + "/SpaGE_impute.csv",header=1, index=1)
                
        if "gimVI" in need_tools:
            result_GimVI = self.gimVI_impute()
            if not os.path.exists(self.outdir):
                os.mkdir(self.outdir)
            result_GimVI.to_csv(self.outdir + "gimVI_impute.csv",header=1, index=1)
                
        if "novoSpaRc" in need_tools:
            result_Novosparc = self.novoSpaRc_impute()
            if not os.path.exists(self.outdir):
                os.mkdir(self.outdir)
            result_Novosparc.to_csv(self.outdir + "/novoSpaRc_impute.csv",header=1, index=1)
                
        if "SpaOTsc" in need_tools:
            result_Spaotsc = self.SpaOTsc_impute()
            if not os.path.exists(self.outdir):
                os.mkdir(self.outdir)
            result_Spaotsc.to_csv(self.outdir + "/SpaOTsc_impute.csv",header=1, index=1)

        if "Tangram" in need_tools:
            result_Tangram = self.Tangram_impute()
            if not os.path.exists(self.outdir):
                os.mkdir(self.outdir)
            result_Tangram.to_csv(self.outdir + "/Tangram_impute.csv",header=1, index=1)
                
        if "stPlus" in need_tools:
            result_stPlus = self.stPlus_impute()
            if not os.path.exists(self.outdir):
                os.mkdir(self.outdir)
            result_stPlus.to_csv(self.outdir + "stPlus_impute.csv",header=1, index=1)

        if 'LIGER' in need_tools:
            train = ','.join(self.train_list)
            test = ','.join(self.test_list)
            os.system('Rscript Codes/Impute/LIGER.r ' + self.RNA_file + ' ' + self.Spatial_file + ' ' + train + ' ' + test + ' ' + self.outdir + '/LIGER_impute.txt')

        if 'Seurat' in need_tools:
            train = ','.join(self.train_list)
            test = ','.join(self.test_list)
            os.system ('Rscript Codes/Impute/Seurat.r ' + self.RNA_file + ' ' + self.Spatial_file + ' ' + train + ' ' + test + ' ' + self.outdir + '/Seurat_impute.txt')
