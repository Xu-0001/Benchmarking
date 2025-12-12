library(rhdf5)
library(Matrix)
library(Seurat)
library(dplyr)
library(data.table)
library(rjson)
library(reticulate)
library(SingleCellExperiment)

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

mydata <- h5read("/home/zhaoshangrui/xuxinyu/PASTE_1/adult_mouse_brain/result/data_mat.h5","mat")
mat <- mydata$block0_values
rownames(mat) <- as.character(mydata$axis0)
colnames(mat) <- as.character(mydata$axis1)
mat <- Matrix(mat, sparse = TRUE)
meta <- read.table("/home/zhaoshangrui/xuxinyu/PASTE_1/adult_mouse_brain/result/data_metadata.tsv",sep="\t",header=T,row.names=1)
pos <- read.table("/home/zhaoshangrui/xuxinyu/PASTE_1/adult_mouse_brain/result/data_position_spatial.tsv",sep="\t",header=T,row.names=1)

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
saveRDS(obj, file='/home/zhaoshangrui/xuxinyu/PASTE_1/adult_mouse_brain/result/original_slice.RDS')

library(Seurat)
library(PRECAST)
library(mclust) 

file_path <- '/home/zhaoshangrui/xuxinyu/PASTE_1/adult_mouse_brain/result/original_slice.RDS'
seulist <- readRDS(file_path)
seulist@meta.data$row <- seulist@images$slice1@coordinates$row
seulist@meta.data$col <- seulist@images$slice1@coordinates$col
seulist_split <- list()
slice_ids <- c(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
               11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 
               21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 
               31, 32, 33, 34)
for (i in 1:length(slice_ids)) {
  slice_id <- slice_ids[i]
  slice_seurat <- subset(seulist, subset = slice == slice_id)
  seulist_split[[i]] <- slice_seurat
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
  all(c("row", "col") %in% colnames(meta_data)) 
  head(meta_data[, c("row", "col")])
}

# ensure the row.names of metadata in metaList are the same as that of colnames count matrix in countList
for (r in 1:M) {
  row.names(metadataList[[r]]) <- colnames(countList[[r]])
}

## Create the Seurat list object
seuList <- list()
for (r in 1:M) {
  seuList[[r]] <- CreateSeuratObject(counts = countList[[r]], meta.data = metadataList[[r]], project = "adult_mouse_brain")
}
saveRDS(seuList, file='/home/zhaoshangrui/xuxinyu/PRECAST/adult_mouse_brain/result/adult_mouse_brain_seuList.RDS')


###Prepare the PRECASTObject with preprocessing step
set.seed(2024)
PRECASTObj <-  CreatePRECASTObject(seuList, project = "PRECAST", selectGenesMethod = "SPARK-X", premin.spots = 0, 
                                   premin.features = 0, postmin.spots = 0, postmin.features = 0)
#Add the model setting
PRECASTObj@seulist
PRECASTObj@seuList#NULL

## Add adjacency matrix list for a PRECASTObj object to prepare for PRECAST model fitting.
PRECASTObj <- AddAdjList(PRECASTObj, platform = "ST")#more platforms to be chosen, including "Visuim", "ST" and "Other_SRT"
## Add a model setting in advance for a PRECASTObj object.
PRECASTObj <- AddParSetting(PRECASTObj, maxIter = 30)

#Fit PRECAST
tic <- proc.time()
PRECASTObj <- PRECAST(PRECASTObj, K = NULL)
toc <- proc.time()
time_used_chooseK <- toc[3] - tic[3]

resList <- PRECASTObj@resList
PRECASTObj <- SelectModel(PRECASTObj)

ari_precast <- sapply(1:length(seuList), function(r) mclust::adjustedRandIndex(PRECASTObj@resList$cluster[[r]],
                                                                               PRECASTObj@seulist[[r]]$cluster_name))
write.csv(ari_precast, "/home/zhaoshangrui/xuxinyu/PRECAST/adult_mouse_brain/metrice/ARI.csv", row.names = FALSE)
nmi_precast <- sapply(1:length(seuList), function(r) aricode::NMI(as.vector(PRECASTObj@resList$cluster[[r]]),
                                                                  PRECASTObj@seulist[[r]]$cluster_name))
write.csv(nmi_precast, "/home/zhaoshangrui/xuxinyu/PRECAST/adult_mouse_brain/metrice/NMI.csv", row.names = FALSE)

PRECASTObj@seuList#NULL
seuInt <- IntegrateSpaData(PRECASTObj, species = "Mouse")
saveRDS(seuInt, file='/home/zhaoshangrui/xuxinyu/PRECAST/adult_mouse_brain/result/adult_mouse_brain_seuInt.RDS')

cell_embeddings <- seuInt@reductions$PRECAST@cell.embeddings
write.csv(cell_embeddings, file = "/home/zhaoshangrui/xuxinyu/PRECAST/adult_mouse_brain/result/PRECAST_embed.csv", row.names = TRUE)