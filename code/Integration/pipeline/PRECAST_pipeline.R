library(PRECAST)
library(Seurat)

dir <- ".../seulist.RDS"
seulist <- readRDS(dir)

genelist <- row.names(seulist[[1]])
getXList <- PRECAST:::getXList
datList <- getXList(seulist, genelist)

AdjList <- pbapply::pblapply(datList$posList, DR.SC::getAdj_auto, radius.upper= 10)

XList <- lapply(datList$XList, scale, scale=F)

# Integration analysis using PRECAST --------------------------------------

hq <- 15
K_set <- 3:11
num_core <- 5
tic <- proc.time()  
set.seed(1)
resList <- ICM.EM(XList,posList=NULL, AdjList=AdjList, q=hq, K=K_set, maxIter=30,
                  Sigma_equal =F, verbose=T, coreNum = num_core, coreNum_int = 1)
toc <- proc.time()
(time_used <- toc[3] - tic[3]) 
save(resList, time_used, file='.../resList_idrsc_chooseK.rds')

reslist <- SelectModel(resList)

preobj <- new(Class = "PRECASTObj", seuList = NULL, seulist = seulist, 
              AdjList = NULL, parameterList = list(), resList = reslist, 
              project = "PRECAST")
PRECASTObj <- preobj
PRECASTObj@parameterList <- model_set( maxIter=30, Sigma_equal =F, verbose=T, coreNum = num_core, coreNum_int = num_core)
seuInt <- IntegrateSpaData(PRECASTObj, species = "Mouse")
saveRDS(seuInt, file='.../seuInt.RDS')