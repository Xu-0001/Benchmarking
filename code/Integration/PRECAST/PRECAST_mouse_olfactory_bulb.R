library(PRECAST)
library(Seurat)
library(dplyr)
library(pbapply)
segment_square <- function(pos, sq_nspots=70, by_order=T, verbose=T){
  tmp <- pos[,1]
  if(by_order){
    x_cut <- sort(tmp)[seq(1, length(tmp), length.out=sq_nspots+1)]
  }else{
    x_cut <- seq(min(tmp), max(tmp), length.out=sq_nspots+1)
  }
  tmp <- pos[,2]
  if(by_order){
    y_cut <- sort(tmp)[seq(1, length(tmp), length.out=sq_nspots+1)]
  }else{
    y_cut <- seq(min(tmp), max(tmp), length.out=sq_nspots+1)
  }
  i <- 1
  pos_new <- matrix(NA, sq_nspots^2, 2)
  areaList <- list()
  for(i1 in 1:sq_nspots){
    if(verbose)
      message('i1 = ', i1)
    for(i2 in 1:sq_nspots){
      if(i1 < sq_nspots && i2 < sq_nspots){
        tmp <- which(x_cut[i1] <=pos[,1] & pos[,1]<x_cut[i1+1] & y_cut[i2] <= pos[,2]& pos[,2] < y_cut[i2+1])
      }else if(i1 < sq_nspots && i2 == sq_nspots){
        tmp <- which(x_cut[i1] <=pos[,1] & pos[,1]< x_cut[i1+1] & y_cut[i2] <= pos[,2]& pos[,2] <= y_cut[i2+1])
      }else{
        tmp <- which(x_cut[i1] <=pos[,1] & pos[,1]<=x_cut[i1+1] & y_cut[i2] <= pos[,2]& pos[,2] < y_cut[i2+1])
      }
      areaList[[i]] <- tmp
      pos_new[i, ] <- c((x_cut[i1]+ x_cut[i1+1])/2, (y_cut[i2]+y_cut[i2+1])/2 )
      i <- i + 1
    }
  }
  idx <- which(sapply(areaList, function(x) length(x)>0))
  return(list(spotID_list=areaList[idx], pos_new = pos_new[idx, ]))
}

get_merged_seu <- function(seu, areaList, pos_new){
  require(Seurat)
  n_area <- length(areaList)
  count_new <- matrix(NA, nrow(seu), n_area)
  colnames(count_new) <- paste0("merge_spot", 1:n_area)
  row.names(count_new) <- row.names(seu)
  DefaultAssay(seu) <- "RNA"
  for(i in 1:n_area){ #
    message('i = ', i, '/', n_area)
    if(length(areaList[[i]])>1){
      count_new[, i] <- rowSums(seu[["RNA"]]@counts[,areaList[[i]]])
    }else{
      count_new[, i] <- seu[["RNA"]]@counts[,areaList[[i]]]
    }
  }
  rm(seu)
  meta_data <- data.frame(row=pos_new[,1], col=pos_new[,2])
  row.names(meta_data) <- colnames(count_new)
  CreateSeuratObject(counts= as.sparse(count_new), meta.data = meta_data)
}

rds_files <- sprintf("/home/zhaoshangrui/xuxinyu/datasets/Mouse_olfactory_bulb_data/Slide-seqV2_datasets_PRECAST/GSM51739%02d_OB2_Slide%d.rds", 45:60, 1:16)
csv_files <- sprintf("/home/zhaoshangrui/xuxinyu/datasets/Mouse_olfactory_bulb_data/Slide-seqV2_datasets_PRECAST/GSM51739%02d_OB2_%02d_BeadLocationsForR.csv", 45:60, 1:16)

read_and_merge <- function(rds_file, csv_file) {
  seurat_obj <- readRDS(rds_file)
  bead_locations <- read.csv(csv_file)
  if (nrow(bead_locations) != ncol(seurat_obj)) {
    stop(paste("CSV 文件", csv_file, "的行数与 Seurat 对象的细胞数不匹配"))
  }
  seurat_obj <- AddMetaData(seurat_obj, metadata = bead_locations)
  if ("xcoord" %in% colnames(seurat_obj@meta.data)) {
    colnames(seurat_obj@meta.data)[colnames(seurat_obj@meta.data) == "xcoord"] <- "row"
  }
  if ("ycoord" %in% colnames(seurat_obj@meta.data)) {
    colnames(seurat_obj@meta.data)[colnames(seurat_obj@meta.data) == "ycoord"] <- "col"
  }
  return(seurat_obj)
}

seuList_raw <- pblapply(1:length(rds_files), function(i) {
  read_and_merge(rds_files[i], csv_files[i])
})

barcodeList <- pbapply::pblapply(seuList_raw, function(x) colnames(x))
save(barcodeList, file='/home/zhaoshangrui/xuxinyu/PRECAST/mouseOB_20/result/barcodeList16_before_merge70_Bulb16rep2.rds')
posList_before_merge70 <- pbapply::pblapply(seuList_raw, function(x) cbind(x$row, x$col))
save(posList_before_merge70, file='/home/zhaoshangrui/xuxinyu/PRECAST/mouseOB_20/result/posList16_before_merge70_Bulb16rep2.rds')

#Segement to 70*70 #############
IDmap70List <- list()
seuList_square70 <- list()
for(r in 1:16){
  # r <- 1
  message("r = ", r)
  res_pos_seg1 <- segment_square(posList_before_merge70[[r]])
  IDmap70List[[r]] <- res_pos_seg1$spotID_list
  seu_new1 <- get_merged_seu(seuList_raw[[r]], res_pos_seg1$spotID_list, res_pos_seg1$pos_new)
  seuList_square70[[r]] <- seu_new1
}

saveRDS(seuList_square70, file='/home/zhaoshangrui/xuxinyu/PRECAST/mouseOB_20/result/mergedSquare70_seuList_Bulb16_rep2.RDS')
save(IDmap70List, file='/home/zhaoshangrui/xuxinyu/PRECAST/mouseOB_20/result/IDmap70List16_Bulb16_rep2.rds')

seuList16_merge70 <- readRDS(file='/home/zhaoshangrui/xuxinyu/PRECAST/mouseOB_20/result/mergedSquare70_seuList_Bulb16_rep2.RDS')

## select top 2000 SVGs
for (i in seq_along(seuList16_merge70)) {
  if ("Seurat" %in% class(seuList16_merge70[[i]]) && !is.null(seuList16_merge70[[i]]@assays$RNA)) {
    counts_data <- as(seuList16_merge70[[i]]@assays$RNA@layers$counts, "dgCMatrix")
    rownames(counts_data) <- rownames(seuList16_merge70[[i]])
    colnames(counts_data) <- colnames(seuList16_merge70[[i]])
    new_assay <- CreateAssayObject(counts = counts_data)
    seuList16_merge70[[i]]@assays$RNA <- new_assay
  } else {
    message(paste("Object at index", i, "is not a Seurat object or lacks assays$RNA"))
  }
}
seuList <- lapply(seuList16_merge70, DR.SC::FindSVGs, nfeatures=2000,num_core=1, verbose=TRUE)
spaFeatureList <- lapply(seuList, DR.SC::topSVGs, ntop=2000)

selectIntFeatures <- PRECAST:::selectIntFeatures
genelist <- selectIntFeatures(seuList,spaFeatureList)
filter_spot <- function(seu, min_feature=0){ # each spots at least include 10 non-zero features
  subset(seu, subset = nFeature_RNA > min_feature)
}

seulist <- pbapply::pblapply(seuList, function(x) x[genelist, ])
index_zeroList <- pbapply::pblapply(seuList, function(x) which(x$nFeature_RNA <= 15)) # empty set
seulist <- pbapply::pblapply(seulist, filter_spot)

saveRDS(seulist, file='/home/zhaoshangrui/xuxinyu/PRECAST/mouseOB_20/result/slideV2_MOB16_rep2_merge70_seulist.RDS')
save(genelist, file='/home/zhaoshangrui/xuxinyu/PRECAST/mouseOB_20/result/genelist_MOB16_rep2_merge70.rds')

# Integration analysis using PRECAST --------------------------------------

posList <- lapply(seulist, function(x) cbind(x$row, x$col))
genelist <- row.names(seulist[[1]])
getXList <- PRECAST:::getXList
datList <- getXList(seulist, genelist)


AdjList <- pbapply::pblapply(datList$posList, DR.SC::getAdj_auto,
                             lower.med=4, upper.med=6, radius.upper= 90)


hq <- 15
K_set <- 2:14
num_core <- 10
tic <- proc.time() # 837058 spots
set.seed(1)
resList_merge70 <- ICM.EM(datList$XList,posList=NULL, AdjList=AdjList,
                          q=hq, K=K_set, int.model=NULL,  beta_grid= seq(1,6, by=0.2), maxIter=30,
                          Sigma_equal =F, verbose=T, coreNum = num_core, coreNum_int = num_core)
toc <- proc.time()
(time_used <- toc[3] - tic[3])
lapply(resList_merge70[[1]]$cluster, table)
save(resList_merge70, time_used, file='/home/zhaoshangrui/xuxinyu/PRECAST/mouseOB_20/result/resList_iDRSC_merge70_Bulb16_rep2.rds')


