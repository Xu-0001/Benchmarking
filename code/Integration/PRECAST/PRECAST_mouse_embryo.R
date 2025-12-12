library(rhdf5)
library(Matrix)
library(Seurat)
library(dplyr)
library(data.table)
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

mydata <- h5read("/home/zhaoshangrui/xuxinyu/PASTE_1/mouse_embryo/result/data_mat.h5","mat")
mat <- mydata$block0_values
rownames(mat) <- as.character(mydata$axis0)
colnames(mat) <- as.character(mydata$axis1)
mat <- Matrix(mat, sparse = TRUE)
meta <- read.table("/home/zhaoshangrui/xuxinyu/PASTE_1/mouse_embryo/result/data_metadata.tsv",sep="\t",header=T,row.names=1)
pos <- read.table("/home/zhaoshangrui/xuxinyu/PASTE_1/mouse_embryo/result/data_position_spatial.tsv",sep="\t",header=T,row.names=1)

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
saveRDS(obj, file='/home/zhaoshangrui/xuxinyu/PASTE_1/mouse_embryo/result/original_slice.RDS')


library(Seurat)
library(PRECAST)
library(mclust) 

file_path <- '/home/zhaoshangrui/xuxinyu/PASTE_1/mouse_embryo/result/original_slice.RDS'
seulist <- readRDS(file_path)
seulist@meta.data$row <- seulist@images$slice1@coordinates$row
seulist@meta.data$col <- seulist@images$slice1@coordinates$col
seulist_split <- list()
slice_ids <- c('E9.5_E1S1', 'E10.5_E2S1', 'E11.5_E1S1', 'E12.5_E1S1')
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
  seuList[[r]] <- CreateSeuratObject(counts = countList[[r]], meta.data = metadataList[[r]], project = "mouse_embryo")
}
saveRDS(seuList, file='/home/zhaoshangrui/xuxinyu/PRECAST/mouse_embryo/result/mouse_embryo_seuList.RDS')


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
PRECASTObj <- PRECAST(PRECASTObj, K = NULL)

resList <- PRECASTObj@resList
PRECASTObj <- SelectModel(PRECASTObj)

PRECASTObj@seuList#NULL
seuInt <- IntegrateSpaData(PRECASTObj, species = "Mouse")
saveRDS(seuInt, file='/home/zhaoshangrui/xuxinyu/PRECAST/mouse_embryo/result/mouse_embryo_seuInt.RDS')

cell_embeddings <- seuInt@reductions$PRECAST@cell.embeddings
write.csv(cell_embeddings, file = "/home/zhaoshangrui/xuxinyu/PRECAST/mouse_embryo/result/PRECAST_embed.csv", row.names = TRUE)
