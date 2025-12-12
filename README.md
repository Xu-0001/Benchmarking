# Benchmarking for algorithms of spatially resolved transcriptomics data alignment and integration
We collected 149 real spatial transcriptome data to benchmark 7 methods for aligning slices and integrating data.
7 methods are included:
   * PASTE: Alignment and Integration of Spatial Transcriptomics Data
   * PASTE2: PASTE2: Partial Alignment of Multi-slice Spatially Resolved Transcriptomics Data
   * GPSA: Alignment of spatial genomics data using deep Gaussian processes
   * PRECAST: Probabilistic embedding, clustering, and alignment for integrating spatial transcriptomics data with PRECAST
   * SPIRAL: Integrating and aligning spatially resolved transcriptomics data across different experiments, conditions, and technologies
   * STAligner: Integrating spatial transcriptomics data across different conditions, technologies and developmental stages
   * STitch3D: Construction of a 3D whole organism spatial atlas by joint modelling of multiple slices with deep neural networks    

## Overview
The main work is as follows.
### Benchmark on different spatial transcriptome data
7 computational methods were benchmarked on 17 experiment form different spatial transcriptome technologies (10x Visium, ST, Slide-Seq, Stero-Seq). The benchmark encompassed 8 different metrics to assess the methods performance in terms of alignment accuracy, spatial coherence, cluster accuracy, and integration performance. The metrics included mapping accuracy, Label Transfer ARI (LTARI), spatial coherence score (SCS), ARI, NMI, F1 scores, integration LISI(iLISI) and cell-type LISI (cLISI).
 
 |Alignment accuracy  | Spatial coherence  | Clustering accuracy  | Integration performance|
 | :--: | :--: | :--: | :--: |  
 |mapping accuracy， LTARI | SCS | ARI, NMI| F1 scores, iLISI, cLISI |

### Dependencies and requirements for slices alignment
Before you run the pipeline, please make sure that you have installed and python3, R(4.3.1) and all the five packages(paste, paste2, gpsa, spiral, staligner): 

1.Before the installation of these packages, please install Miniconda to manage all needed software and dependencies. You can download Miniconda from https://conda.io/miniconda.html.

2.Download SpatialBenchmarking.zip from https://github.com/ . . . /SRTBenchmarking. Unzipping this package and you will see Benchmarkingenvironment.yml and Config.env.sh located in its folder.

3.Build isolated environment for SpatialBenchmarking: conda env create -f Benchmarkingenvironment.yml

4.Activate Benchmarking environment: conda activate Benchmarking

5.sh Config.env.sh

6.Enter R and install required packages by command : install.packages("xxx")

### Dependencies and requirements for data integration
Before you run the pipeline, please make sure that you have installed and python3, R(4.3.1) and all the five packages(paste, precast, spiral, staligner, stitch3d): 
The package has been tested on Linux system and should work in any valid python environment.
## Tutorial
If you want to analysis your own data, the doc/Tutorial.ipynb is an example showing how to use them to predict new slices alignment and data integration.
## Datasets
All datasets used are publicly available data, for convenience datasets can be downloaded from:
