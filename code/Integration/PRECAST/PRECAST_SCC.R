library(rhdf5)
library(Matrix)
library(Seurat)
library(dplyr)
library(data.table)
library(Matrix)
library(rjson)

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

mydata <- h5read("/home/zhaoshangrui/xuxinyu/PASTE_1/SCC/result/data_mat_patient_2.h5","mat")
mat <- mydata$block0_values
rownames(mat) <- as.character(mydata$axis0)
colnames(mat) <- as.character(mydata$axis1)
mat <- Matrix(mat, sparse = TRUE)
meta <- read.table("/home/zhaoshangrui/xuxinyu/PASTE_1/SCC/result/data_metadata_patient_2.tsv",sep="\t",header=T,row.names=1)
pos <- read.table("/home/zhaoshangrui/xuxinyu/PASTE_1/SCC/result/data_position_spatial_patient_2.tsv",sep="\t",header=T,row.names=1)

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
saveRDS(seurat_spatialObj, file='/home/zhaoshangrui/xuxinyu/PASTE_1/SCC/result/original_slice_patient_2.RDS')


library(Seurat)
library(PRECAST)
library(mclust) 

file_path <- '/home/zhaoshangrui/xuxinyu/PASTE_1/SCC/result/original_slice_patient_2.RDS'
seulist <- readRDS(file_path)
seulist@meta.data$row <- seulist@images$slice1@coordinates$row
seulist@meta.data$col <- seulist@images$slice1@coordinates$col
slice_ids <- c('0','1','2')
seulist_split <- list()
for (i in 1:3) {
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
  seuList[[r]] <- CreateSeuratObject(counts = countList[[r]], meta.data = metadataList[[r]], project = "SCC")
}
saveRDS(seuList, file='/home/zhaoshangrui/xuxinyu/PRECAST/SCC/SCC_seuList_patient_2.RDS')

###Prepare the PRECASTObject with preprocessing step
set.seed(2024)
PRECASTObj <-  CreatePRECASTObject(seuList, project = "PRECAST",, premin.spots = 0, premin.features = 0, postmin.spots = 0, postmin.features = 0)
#Add the model setting
PRECASTObj@seulist
PRECASTObj@seuList#NULL

## Add adjacency matrix list for a PRECASTObj object to prepare for PRECAST model fitting.
PRECASTObj <- AddAdjList(PRECASTObj, platform = "ST")#more platforms to be chosen, including "Visuim", "ST" and "Other_SRT"
## Add a model setting in advance for a PRECASTObj object.
PRECASTObj <- AddParSetting(PRECASTObj, Sigma_equal = FALSE, verbose = TRUE, maxIter = 30)

#Fit PRECAST
PRECASTObj <- PRECAST(PRECASTObj, K = 12)

resList <- PRECASTObj@resList
PRECASTObj <- SelectModel(PRECASTObj)

ari_precast <- sapply(1:length(seuList), function(r) mclust::adjustedRandIndex(PRECASTObj@resList$cluster[[r]],
                                                                               PRECASTObj@seulist[[r]]$original_clusters))
write.csv(ari_precast, "/home/zhaoshangrui/xuxinyu/PRECAST/SCC/metrice/ARI_patient_2.csv", row.names = FALSE)
nmi_precast <- sapply(1:length(seuList), function(r) aricode::NMI(as.vector(PRECASTObj@resList$cluster[[r]]),
                                                                  PRECASTObj@seulist[[r]]$original_clusters))
write.csv(nmi_precast, "/home/zhaoshangrui/xuxinyu/PRECAST/SCC/metrice/NMI_patient_2.csv", row.names = FALSE)

PRECASTObj@seuList#NULL
seuInt <- IntegrateSpaData(PRECASTObj, species = "Human")
saveRDS(seuInt, file='/home/zhaoshangrui/xuxinyu/PRECAST/SCC/SCC_seuInt_patient_2.RDS')

cell_embeddings <- seuInt@reductions$PRECAST@cell.embeddings
write.csv(cell_embeddings, file = "/home/zhaoshangrui/xuxinyu/PRECAST/SCC/result/PRECAST_embed_patient_2.csv", row.names = TRUE)
