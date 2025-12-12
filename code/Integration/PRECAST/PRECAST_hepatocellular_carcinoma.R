library(DR.SC)
library(Seurat)
library(Matrix)
library(dplyr)
library(PRECAST)

url_hcc <- "/home/zhaoshangrui/xuxinyu/datasets/Human_hepatocellular_carcinoma/"
posList <- list()
seuList <- list()
for(iter in 1: 4){
  # iter <- 1
  message("iter = ", iter)
  hcc <- readRDS(paste0(url_hcc,"HCC", iter, "_seu.RDS"))
  seuList[[iter]] <- hcc
  posList[[iter]] <- cbind(row=hcc$row, col=hcc$col)
}

### Save seuList for HCC4 data##################
saveRDS(seuList, file='/home/zhaoshangrui/xuxinyu/datasets/Human_hepatocellular_carcinoma/HCC4_seuList.RDS')
save(posList, file = '/home/zhaoshangrui/xuxinyu/datasets/Human_hepatocellular_carcinoma/HCC4_posList.rds')

load('/home/zhaoshangrui/xuxinyu/datasets/Human_hepatocellular_carcinoma/HCC4_spatialFeatureList.rds')

selectIntFeatures <- PRECAST:::selectIntFeatures
getXList <- PRECAST:::getXList

genelist <- selectIntFeatures(seuList, spaFeatureList)
length(unique(genelist))
save(genelist, file='/home/zhaoshangrui/xuxinyu/datasets/Human_hepatocellular_carcinoma/HCC4_genelist2000.rds')

seulist <- pbapply::pblapply(seuList, function(x) x[genelist, ])
saveRDS(seulist, file='/home/zhaoshangrui/xuxinyu/datasets/Human_hepatocellular_carcinoma/HCC4_seulist.RDS')

datList <- getXList(seuList, genelist)
saveRDS(datList, file='/home/zhaoshangrui/xuxinyu/datasets/Human_hepatocellular_carcinoma/HCC4_datList.RDS')

metadataList <- lapply(seuList, function(x) x@meta.data)
save(metadataList, file='/home/zhaoshangrui/xuxinyu/datasets/Human_hepatocellular_carcinoma/metadataList_hcc4.rds')


XList <- datList$XList
posList <- datList$posList
indexList <- datList$indxList

## Integration analysis using PRECAST ############################################################
q <- 15; K <- 2:11
tic <- proc.time() # 
set.seed(1)
resList <- ICM.EM(XList, posList=posList, q=q, K=K, 
                  platform = 'Visium',maxIter = 30,
                  Sigma_equal =F, verbose=T, coreNum = 5, coreNum_int = 5)
toc <- proc.time()
time_used_chooseK <- toc[3] - tic[3] #29.773 mins

save(time_used_chooseK, resList, file ="/home/zhaoshangrui/xuxinyu/datasets/Human_hepatocellular_carcinoma/idrsc_HCC4_chooseK.rds")  

reslist <- SelectModel(resList)

preobj <- new(Class = "PRECASTObj", seuList = NULL, seulist = seulist, 
              AdjList = NULL, parameterList = list(), resList = reslist, 
              project = "PRECAST")
PRECASTObj <- preobj
num_core <- 5
PRECASTObj@parameterList <- model_set( maxIter=30, Sigma_equal =F, verbose=T, coreNum = num_core, coreNum_int = num_core)
seuInt <- IntegrateSpaData(PRECASTObj, species = "human")
saveRDS(seuInt, file='/home/zhaoshangrui/xuxinyu/datasets/Human_hepatocellular_carcinoma/HCC4_seuInt.RDS')
