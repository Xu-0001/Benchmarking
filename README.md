# Benchmarking for algorithms of spatially resolved transcriptomics data alignment and integration

<img width="904" height="408" alt="image" src="https://github.com/user-attachments/assets/d7e09717-43e5-4dc3-8f2a-c7fb6031b82e" />

### Description
We benchmarked nine existing methods for spatial transcriptomics slice integration using 20 real-world and simulated datasets by evaluating their alignment accuracy, integration effectiveness, and robustness. For alignment accuracy, we assessed slice alignment performance using mapping accuracy. For integration effectiveness, we employed multiple metrics: Spatial coherence score to evaluate spatial consistency preservation; Adjusted Rand Index (ARI) and normalized mutual information (NMI) to measure clustering precision; F1 scores, integration LISI (iLISI), and cell-type LISI (cLISI) in the embedding space to quantify batch effect removal. For robustness, we designed experiments under the following conditions: (1) Simulating Dropout Events: For gene expression data, we randomly selected proportions (20%, 40%, and 60%) of non-zero values and set them to zero, mimicking inherent and widespread technical dropout events. This tested the methods' performance on datasets with sparse gene expression due to random information loss. (2) Simulating Sequencing Depth Variations: We performed subsampling at 20%, 40%, and 60% levels (i.e., retaining 80%, 60%, and 40% of spots) on both the gene expression matrix and spatial coordinates to simulate varying sequencing depths, respectively. This evaluated the methods' ability to handle datasets with significant differences in sequencing depth.

### Dependencies and requirements for slices alignment
Before you run the pipeline, please make sure that you have installed and python3, R(4.3.1) and all the nine packages (paste, paste2, st-gears, gpsa, precast, spiral, staligner, stitch3d, staig): 

1.Before the installation of these packages, please install Miniconda to manage all needed software and dependencies. You can download Miniconda from https://conda.io/miniconda.html.

2.Download Benchmarking.zip from https://github.com/Xu-0001/Benchmarking/. Unzipping this package and you will see Benchmarkingenvironment.yml and Config.env.sh located in its folder.

3.Build isolated environment for SpatialBenchmarking: conda env create -f Benchmarkingenvironment.yml

4.Activate Benchmarking environment: conda activate Benchmarking

5.sh Config.env.sh

6.Enter R and install required packages by command

### Dependencies and requirements for data integration
Before you run the pipeline, please make sure that you have installed and python3, R(4.3.1) and all the nine packages(paste, paste2, st-gears, gpsa, precast, spiral, staligner, stitch3d, staig): 
The package has been tested on Linux system and should work in any valid python environment.

### Tutorial
If you want to analysis your own data, the doc/Tutorial.ipynb is an example showing how to use them for aligning and integrating new slices.

You also can run the jupyter notebook of BLAST_GenePrediction.ipynb and BLAST_CelltypeDeconvolution.ipynb to reproduce the results of figure2&4 in our paper.

For more details, please see the Alignment.py & Integration.py in Benchmarking directory.

