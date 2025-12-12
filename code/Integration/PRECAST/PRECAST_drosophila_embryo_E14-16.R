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

mydata <- h5read("/home/zhaoshangrui/xuxinyu/PASTE_1/Drosophila_embryo_E14-16/result/data_mat.h5","mat")
mat <- mydata$block0_values
rownames(mat) <- as.character(mydata$axis0)
colnames(mat) <- as.character(mydata$axis1)
mat <- Matrix(mat, sparse = TRUE)
meta <- read.table("/home/zhaoshangrui/xuxinyu/PASTE_1/Drosophila_embryo_E14-16/result/data_metadata.tsv",sep="\t",header=T,row.names=1)
pos <- read.table("/home/zhaoshangrui/xuxinyu/PASTE_1/Drosophila_embryo_E14-16/result/data_position_spatial.tsv",sep="\t",header=T,row.names=1)

obj <- CreateSeuratObject(mat,assay='Spatial',meta.data=meta)
tissue_lowres_image <- matrix(1, max(pos$y), max(pos$x))
tissue_positions_list <- data.frame(row.names = colnames(obj),
                                    tissue = 1,
                                    row = pos$y, col = pos$x,
                                    imagerow = pos$y, imagecol = pos$x)
scalefactors_json <- toJSON(list(fiducial_diameter_fullres = 1,
                                 tissue_hires_scalef = 1,
                                 tissue_lowres_scalef = 1))
mat <- obj@assays$Spatial$counts

seurat_spatialObj <- CreateSeuratObject(mat, project = 'Spatial', assay = 'Spatial', meta.data=meta)


spatialObj <- generate_spatialObj(image = tissue_lowres_image,
                                  scale.factors = fromJSON(scalefactors_json),
                                  tissue.positions = tissue_positions_list)

spatialObj <- spatialObj[Cells(seurat_spatialObj)]
DefaultAssay(spatialObj) <- 'Spatial'
seurat_spatialObj[['slice1']] <- spatialObj
saveRDS(seurat_spatialObj, file='/home/zhaoshangrui/xuxinyu/PASTE_1/Drosophila_embryo_E14-16/result/original_slice.RDS')

file_path <- "/home/zhaoshangrui/xuxinyu/PASTE_1/Drosophila_embryo_E14-16/result/original_slice.RDS"
seulist <- readRDS(file_path)
seulist@meta.data$row <- seulist@images$slice1@coordinates$row
seulist@meta.data$col <- seulist@images$slice1@coordinates$col
seulist_split <- list()
slice_ids <- c('E14-16h_a_S01','E14-16h_a_S02','E14-16h_a_S03','E14-16h_a_S04','E14-16h_a_S05','E14-16h_a_S06','E14-16h_a_S07','E14-16h_a_S08',
               'E14-16h_a_S09','E14-16h_a_S10','E14-16h_a_S11','E14-16h_a_S12','E14-16h_a_S13','E14-16h_a_S14','E14-16h_a_S15','E14-16h_a_S16')
for (i in 1:length(slice_ids)) {
  slice_id <- slice_ids[i]
  slice_seurat <- subset(seulist, subset = batch == slice_id)
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
  seuList[[r]] <- CreateSeuratObject(counts = countList[[r]], meta.data = metadataList[[r]], project = "Drosophila_embryo")
}
saveRDS(seuList, file='/home/zhaoshangrui/xuxinyu/PRECAST/Drosophila_embryo_E14-16/result/Drosophila_embryo_E14-16_seuList.RDS')

###Prepare the PRECASTObject with preprocessing step
set.seed(2024)
PRECASTObj <-  CreatePRECASTObject(seuList, project = "PRECAST", gene.number = 5000, selectGenesMethod = "SPARK-X", premin.spots = 0, 
                                   premin.features = 0, postmin.spots = 0, postmin.features = 0)
#Add the model setting
PRECASTObj@seulist
PRECASTObj@seuList#NULL

## Add adjacency matrix list for a PRECASTObj object to prepare for PRECAST model fitting.
PRECASTObj <- AddAdjList(PRECASTObj, platform = "ST")#more platforms to be chosen, including "Visuim", "ST" and "Other_SRT"
## Add a model setting in advance for a PRECASTObj object.
PRECASTObj <- AddParSetting(PRECASTObj, maxIter = 30)

#Fit PRECAST
PRECASTObj <- PRECAST(PRECASTObj, K = 10)

resList <- PRECASTObj@resList
PRECASTObj <- SelectModel(PRECASTObj)


ari_precast <- sapply(1:length(seuList), function(r) mclust::adjustedRandIndex(PRECASTObj@resList$cluster[[r]],
                                                                               PRECASTObj@seulist[[r]]$annotation))
write.csv(ari_precast, "/home/zhaoshangrui/xuxinyu/PRECAST/Drosophila_embryo_E14-16/metrice/ARI.csv", row.names = FALSE)
nmi_precast <- sapply(1:length(seuList), function(r) aricode::NMI(as.vector(PRECASTObj@resList$cluster[[r]]),
                                                                  PRECASTObj@seulist[[r]]$annotation))
write.csv(nmi_precast, "/home/zhaoshangrui/xuxinyu/PRECAST/Drosophila_embryo_E14-16/metrice/NMI.csv", row.names = FALSE)

seuInt <- IntegrateSpaData(PRECASTObj, species = "Unknown")
saveRDS(seuInt, file='/home/zhaoshangrui/xuxinyu/PRECAST/Drosophila_embryo_E14-16/result/Drosophila_embryo_E14-16_seuInt.RDS')

cell_embeddings <- seuInt@reductions$PRECAST@cell.embeddings
write.csv(cell_embeddings, file = "/home/zhaoshangrui/xuxinyu/PRECAST/Drosophila_embryo_E14-16/result/Drosophila_embryo_E14-16_embed.csv", row.names = TRUE)
