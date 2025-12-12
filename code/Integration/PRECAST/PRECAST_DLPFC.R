library(Matrix)
library(dplyr)

library(data.table)

library(rhdf5)

library(Seurat)
library(SingleCellExperiment)
library(PRECAST)
library(sceasy)
library(scPOP)

library(mclust)
library(cluster)

library(reticulate)
library(rjson)

source("D:/A/Experiment Progress new/visiualization/r_criteria.R")

generate_spatialObj <- function(image, scale.factors, tissue.positions, filter.matrix = TRUE)
{
  if (filter.matrix) {
    tissue.positions <- tissue.positions[which(tissue.positions$tissue == 1), , drop = FALSE]
  }
  unnormalized.radius <- scale.factors$fiducial_diameter_fullres * scale.factors$tissue_lowres_scalef
  spot.radius <- unnormalized.radius / max(dim(image))
  return(new(Class = 'VisiumV1',
             image = image,
             scale.factors = scalefactors(spot = scale.factors$tissue_hires_scalef,
                                          fiducial = scale.factors$fiducial_diameter_fullres,
                                          hires = scale.factors$tissue_hires_scalef,
                                          lowres = scale.factors$tissue_lowres_scalef),
             coordinates = tissue.positions,
             spot.radius = spot.radius))
}

slice_ids <- c("151673","151674","151675","151676")
flags <- "151507-151508-151509-151510-151669-151670-151671-151672-151673-151674-151675-151676_"
seulist_split <- list()
for (i in 1:length(slice_ids)) {
  file_path_features <- paste0("D:/A/Experiment Progress/DLPFC/SPIRAL/data/gtt_input_scanpy/", flags, slice_ids[i], "_features-1.txt")
  file_path_meta <- paste0("D:/A/Experiment Progress/DLPFC/SPIRAL/data/gtt_input_scanpy/", flags, slice_ids[i], "_label-1.txt")
  file_path_positions <- paste0("D:/A/Experiment Progress/DLPFC/SPIRAL/data/gtt_input_scanpy/",flags, slice_ids[i], "_positions-1.txt")
  mat <- read.table(file_path_features, sep=",", header=T, row.names=1)
  mat <- as.matrix(mat)
  mat <- t(mat)
  mat <- Matrix(mat, sparse = TRUE) # 转成稀疏矩阵
  meta <- read.table(file_path_meta, sep=",", header=T, row.names=1) # 读取metadata信息
  pos <- read.table(file_path_positions, sep=",", header=T, row.names=1) # 读取坐标信息
  obj <- CreateSeuratObject(mat,assay='Spatial',meta.data=meta)
  tissue_lowres_image <- matrix(1, max(pos$y), max(pos$x))
  tissue_positions_list <- data.frame(row.names = colnames(obj),
                                      tissue = 1,
                                      row = pos$y, col = pos$x,
                                      imagerow = pos$y, imagecol = pos$x)
  scalefactors_json <- toJSON(list(fiducial_diameter_fullres = 1,
                                   tissue_hires_scalef = 1,
                                   tissue_lowres_scalef = 1))
  spatialObj <- generate_spatialObj(image = tissue_lowres_image,
                                    scale.factors = fromJSON(scalefactors_json),
                                    tissue.positions = tissue_positions_list)
  spatialObj <- spatialObj[Cells(obj)]
  DefaultAssay(spatialObj) <- 'Spatial'
  obj[['slice1']] <- spatialObj
  obj@meta.data$row <- obj@images$slice1@coordinates$row
  obj@meta.data$col <- obj@images$slice1@coordinates$col
  seulist_split[[i]] <- obj
}  

seulist <- seulist_split

###Create a PRECASTObject object
countList <- lapply(seulist, function(x) {
  assay <- DefaultAssay(x)
  GetAssayData(x, assay = assay, slot = "counts")
})

M <- length(countList)
metadataList <- lapply(seulist, function(x) x@meta.data)

for (r in 1:M) {
  meta_data <- metadataList[[r]]
  all(c("row", "col") %in% colnames(meta_data))  ## the names are correct!
  head(meta_data[, c("row", "col")])
}

# ensure the row.names of metadata in metaList are the same as that of colnames count matrix in countList
for (r in 1:M) {
  row.names(metadataList[[r]]) <- colnames(countList[[r]])
}

## Create the Seurat list object
seuList <- list()
for (r in 1:M) {
  seuList[[r]] <- CreateSeuratObject(counts = countList[[r]], meta.data = metadataList[[r]], project = "DLPFC")
}
saveRDS(seuList, file='/home/zhaoshangrui/xuxinyu/PRECAST/DLPFC/results_sample3/DLPFC_seuList.RDS')

PRECASTObj <- CreatePRECASTObject(seuList, project = "DLPFC", gene.number = 2000, premin.spots = 0, premin.features = 0, postmin.spots = 0, postmin.features = 0,)
PRECASTObj <- AddAdjList(PRECASTObj, platform = "Visuim")
PRECASTObj <- AddParSetting(PRECASTObj, maxIter=30, Sigma_equal = FALSE, verbose = TRUE, int.model = "EEE", seed = 1)

#Fit PRECAST
PRECASTObj <- PRECAST(PRECASTObj, K = 7)

resList <- PRECASTObj@resList
PRECASTObj <- SelectModel(PRECASTObj)

ari_precast <- sapply(1:length(seuList), function(r) mclust::adjustedRandIndex(PRECASTObj@resList$cluster[[r]],
                                                                               PRECASTObj@seulist[[r]]$celltype))
write.csv(ari_precast, "D:/A/Experiment Progress new/1.DLPFC(Visium)/output/ARI_precast.csv", row.names = FALSE)
nmi_precast <- sapply(1:length(seuList), function(r) aricode::NMI(as.vector(PRECASTObj@resList$cluster[[r]]),
                                                                  PRECASTObj@seulist[[r]]$celltype))
write.csv(nmi_precast, "D:/A/Experiment Progress new/1.DLPFC(Visium)/output/NMI_precast.csv", row.names = FALSE)

PRECASTObj@seuList#NULL
seuInt <- IntegrateSpaData(PRECASTObj, species = "Human")
saveRDS(seuInt, file="D:/A/Experiment Progress new/1.DLPFC(Visium)/output/DLPFC_seuInt.RDS")

seuInt <- readRDS("D:/A/Experiment Progress new/1.DLPFC(Visium)/output/DLPFC_seuInt.RDS")
cols_cluster <- c('#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02', '#a6761d')
p12 <- SpaPlot(seuInt, item = "cluster", batch = NULL, point_size = 0.1, cols = cols_cluster, combine = TRUE,
               nrow.legend = 7)
p12
ggsave("D:/A/Experiment Progress new/1.DLPFC(Visium)/output/PRECAST.pdf", plot = p12, width = 10, height = 8, units = "in")

cell_embeddings <- seuInt@reductions$PRECAST@cell.embeddings
write.csv(cell_embeddings, file = "D:/A/Experiment Progress new/1.DLPFC(Visium)/output/PRECAST_embed.csv", row.names = TRUE)
