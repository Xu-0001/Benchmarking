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

source("/home/zhaoshangrui/xuxinyu/PASTE_1/files/r_criteria.R")

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

slice_ids <- c("anterior1","anterior2")
seulist_split <- list()
for (i in 1:length(slice_ids)) {
  file_path_features <- paste0("/home/zhaoshangrui/xuxinyu/SPIRAL/Sagittal_mouse_brain/gtt_input/", slice_ids[i], "_mat.csv")
  file_path_meta <- paste0("/home/zhaoshangrui/xuxinyu/SPIRAL/Sagittal_mouse_brain/gtt_input/", slice_ids[i], "_meta.csv")
  file_path_positions <- paste0("/home/zhaoshangrui/xuxinyu/SPIRAL/Sagittal_mouse_brain/gtt_input/", slice_ids[i], "_coord.csv")
  mat <- read.table(file_path_features, sep=",", header=T, row.names=1)
  mat <- as.matrix(mat)
  mat <- t(mat)
  mat <- Matrix(mat, sparse = TRUE) 
  meta <- read.table(file_path_meta, sep=",", header=T, row.names=1) 
  pos <- read.table(file_path_positions, sep=",", header=T, row.names=1)
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
  seuList[[r]] <- CreateSeuratObject(counts = countList[[r]], meta.data = metadataList[[r]], project = "Sagittal_mouse_brain")
}
saveRDS(seuList, file='/home/zhaoshangrui/xuxinyu/PRECAST/Sagittal_mouse_brain/result/Sagittal_mouse_brain_anterior_seuList.RDS')

###Prepare the PRECASTObject with preprocessing step
set.seed(2024)
PRECASTObj <-  CreatePRECASTObject(seuList, project = "PRECAST", gene.number = 5000, selectGenesMethod = "SPARK-X", premin.spots = 0, 
                                   premin.features = 0, postmin.spots = 0, postmin.features = 0)
#Add the model setting
PRECASTObj@seulist
PRECASTObj@seuList#NULL

## Add adjacency matrix list for a PRECASTObj object to prepare for PRECAST model fitting.
PRECASTObj <- AddAdjList(PRECASTObj, platform = "Visuim")#more platforms to be chosen, including "Visuim", "ST" and "Other_SRT"
## Add a model setting in advance for a PRECASTObj object.
PRECASTObj <- AddParSetting(PRECASTObj, maxIter = 30)

#Fit PRECAST
PRECASTObj <- PRECAST(PRECASTObj, K = NULL)

resList <- PRECASTObj@resList
PRECASTObj <- SelectModel(PRECASTObj)

ari_precast <- sapply(1:length(seuList), function(r) mclust::adjustedRandIndex(PRECASTObj@resList$cluster[[r]],
                                                                               PRECASTObj@seulist[[r]]$celltype))
write.csv(ari_precast, "/home/zhaoshangrui/xuxinyu/PRECAST/Sagittal_mouse_brain/metrice/ARI_anterior.csv", row.names = FALSE)
nmi_precast <- sapply(1:length(seuList), function(r) aricode::NMI(as.vector(PRECASTObj@resList$cluster[[r]]),
                                                                  PRECASTObj@seulist[[r]]$celltype))
write.csv(nmi_precast, "/home/zhaoshangrui/xuxinyu/PRECAST/Sagittal_mouse_brain/metrice/NMI_anterior.csv", row.names = FALSE)

PRECASTObj@seuList#NULL
seuInt <- IntegrateSpaData(PRECASTObj, species = "Mouse")
saveRDS(seuInt, file='/home/zhaoshangrui/xuxinyu/PRECAST/Sagittal_mouse_brain/result/Sagittal_mouse_brain_anterior_seuInt.RDS')

cell_embeddings <- seuInt@reductions$PRECAST@cell.embeddings
write.csv(cell_embeddings, file = "/home/zhaoshangrui/xuxinyu/PRECAST/Sagittal_mouse_brain/result/PRECAST_embed_anterior.csv", row.names = TRUE)
