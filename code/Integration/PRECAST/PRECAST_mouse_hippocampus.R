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

file_path <- "D:/A/Experiment Progress/original_slice.RDS"
seulist <- readRDS(file_path)
seulist@meta.data$row <- seulist@images$slice1@coordinates$row
seulist@meta.data$col <- seulist@images$slice1@coordinates$col
seulist_split <- list()
slice_ids <- c('Puck_200115_08','Puck_191204_01')
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
  seuList[[r]] <- CreateSeuratObject(counts = countList[[r]], meta.data = metadataList[[r]], project = "mouse hippocampus")
}
saveRDS(seuList, file='D:/A/Experiment Progress/mouse_hippocampus/PRECAST/mouse_hippocampus_seuList.RDS')

file_path <- "D:/A/Experiment Progress/mouse_hippocampus/PRECAST/mouse_hippocampus_seuList.RDS"
seuList <- readRDS(file_path)

# ###Prepare the PRECASTObject with preprocessing step
# set.seed(2024)
# PRECASTObj <-  CreatePRECASTObject(seuList, project = "mouse hippocampus")
# #Add the model setting
# PRECASTObj@seulist
# PRECASTObj@seuList#NULL
# 
# ## Add adjacency matrix list for a PRECASTObj object to prepare for PRECAST model fitting.
# PRECASTObj <- AddAdjList(PRECASTObj, platform = "Other_SRT")#more platforms to be chosen, including "Visuim", "ST" and "Other_SRT"
# ## Add a model setting in advance for a PRECASTObj object.
# PRECASTObj <- AddParSetting(PRECASTObj, Sigma_equal = FALSE, verbose = TRUE, maxIter = 30)

PRECASTObj <- CreatePRECASTObject(seuList, project = "mouse hippocampus", gene.number = 2000, premin.spots = 0, premin.features = 0, postmin.spots = 0, postmin.features = 0,)
PRECASTObj <- AddAdjList(PRECASTObj, platform = "Other_SRT")
PRECASTObj <- AddParSetting(PRECASTObj, maxIter=30, Sigma_equal = FALSE, verbose = TRUE, int.model = "EEE", seed = 1)

#Fit PRECAST
PRECASTObj <- PRECAST(PRECASTObj, K = NULL)

resList <- PRECASTObj@resList
PRECASTObj <- SelectModel(PRECASTObj)

PRECASTObj@seuList#NULL
seuInt <- IntegrateSpaData(PRECASTObj, species = "Mouse")
saveRDS(seuInt, file='D:/A/Experiment Progress/mouse_hippocampus/PRECAST/mouse_hippocampus_seuInt_1.RDS')

cell_embeddings <- seuInt@reductions$PRECAST@cell.embeddings
write.csv(cell_embeddings, file = "D:/A/Experiment Progress/mouse_hippocampus/PRECAST/PRECAST_embed.csv", row.names = TRUE)
