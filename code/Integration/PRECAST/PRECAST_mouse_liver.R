library(PRECAST)
library(Seurat)

dir <- "/home/zhaoshangrui/xuxinyu/datasets/mouseLiver8/seulist_mouseLiverST8.RDS"
seulist <- readRDS(dir)
yList <- lapply(seulist, function(x) x$cluster)
table(unlist(yList)) ## K= 6
save(yList, file='/home/zhaoshangrui/xuxinyu/datasets/mouseLiver8/yList_mouseLiver8.rds')

library(Seurat)
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
save(resList, time_used, file='/home/zhaoshangrui/xuxinyu/datasets/mouseLiver8/resList_idrsc_chooseK_mouseLiverST8.rds')

reslist <- SelectModel(resList)

preobj <- new(Class = "PRECASTObj", seuList = NULL, seulist = seulist, 
              AdjList = NULL, parameterList = list(), resList = reslist, 
              project = "PRECAST")
PRECASTObj <- preobj
PRECASTObj@parameterList <- model_set( maxIter=30, Sigma_equal =F, verbose=T, coreNum = num_core, coreNum_int = num_core)
seuInt <- IntegrateSpaData(PRECASTObj, species = "Mouse")
saveRDS(seuInt, file='/home/zhaoshangrui/xuxinyu/datasets/mouseLiver8/mouseLiver8_seuInt.RDS')
seuInt <- readRDS('/home/zhaoshangrui/xuxinyu/datasets/mouseLiver8/mouseLiver8_seuInt.RDS')

clusterList <- reslist$cluster
save(clusterList, file='/home/zhaoshangrui/xuxinyu/datasets/mouseLiver8/idrsc_cluster7_mouseLiver8.rds')
cluster_metric <- function(hy, y, type='ARI'){
  require(mclust)
  require(aricode)
  switch(type, 
         ARI= adjustedRandIndex(hy, y),
         NMI = NMI(as.vector(hy), y))
}
combine_metric <- c(ARI=cluster_metric(unlist(reslist$cluster), unlist(yList)),
                    NMI=cluster_metric(unlist(reslist$cluster), unlist(yList), type="NMI"))
write.csv(combine_metric, "/home/zhaoshangrui/xuxinyu/datasets/mouseLiver8/combine_metric_ARI_NMI.csv", row.names = FALSE)
sep_metric_ARI <- sapply(1:8, function(j) cluster_metric((reslist$cluster[[j]]), (yList[[j]])))
write.csv(sep_metric_ARI, '/home/zhaoshangrui/xuxinyu/datasets/mouseLiver8/sep_metric_ARI.csv', row.names = FALSE)
sep_metric_NMI <- sapply(1:8, function(j) cluster_metric(reslist$cluster[[j]], yList[[j]], type="NMI"))
write.csv(sep_metric_NMI, '/home/zhaoshangrui/xuxinyu/datasets/mouseLiver8/sep_metric_NMI.csv', row.names = FALSE)

posList <- datList$posList
save(posList, file="/home/zhaoshangrui/xuxinyu/datasets/mouseLiver8/posList_mouseLiver8.rds")