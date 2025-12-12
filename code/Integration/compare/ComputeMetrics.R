library(SingleCellExperiment)
library(Seurat)
library(Matrix)
library(dplyr)
library(cluster)
library(scPOP)

library(PRECAST)
library(ggplot2)

##F1_score
F1_score_silho <- function(embeddings, celltype, sampleID){
  require(scPOP)
  metadata <- data.frame(celltype=celltype, sampleID = sampleID)
  metadata$celltype <- as.factor(metadata$celltype)
  sh_scores_pro <- silhouette_width(embeddings, meta.data = metadata, c('celltype', "sampleID") )
  sh_ct <- (1+sh_scores_pro[1])/2 # larger is better
  sh_si <- (1+sh_scores_pro[2])/2 # smaller is better
  f1_score <- (2* (1-sh_si)*sh_ct) / ((1-sh_si) + sh_ct)
  return(f1_score)
}


silho_allspots <- function(embeddings,  category){
  
  metadata <- data.frame(celltype=category)
  dd <- dist(embeddings)
  sh_scores_pro <- sapply(names(metadata), function(x) {
    cluster::silhouette(as.numeric(as.factor(metadata[[x]])),
                        dd)[,3]
  })
  sh_scores_pro
}
F1_score_silho_1 <- function(embeddings, celltype, sampleID){
  require(scPOP)
  metadata <- data.frame(celltype=celltype, sampleID = sampleID)
  metadata$celltype <- as.factor(metadata$celltype)
  sh_scores_pro_1 <- silho_allspots(embeddings,metadata$celltype)
  sh_scores_pro_2 <- silho_allspots(embeddings,metadata$sampleID)
  sh_ct <- (1+sh_scores_pro_1)/2 # larger is better
  sh_si <- (1+sh_scores_pro_2)/2 # smaller is better
  f1_score <- (2* (1-sh_si)*sh_ct) / ((1-sh_si) + sh_ct)
  return(f1_score)
}


##lisi
ilsi_avg_scores_allspots <- function(embeddings,  category){
  require(scPOP)
  metadata <- data.frame(category=category)
  lisi_scores_pro <- lisi(embeddings, meta_data = metadata, 'category')
  lisi_scores_pro$category # return a vector
}


##tSNE
library(ggthemes)
library(colorspace)
chooseColors <- function (palettes_name = c("Nature 10", "Light 13", "Classic 20",
                                            "Blink 23", "Hue n"), n_colors = 7, alpha = 1, plot_colors = FALSE)
{
  palettes_name <- match.arg(palettes_name)
  colors <- if (palettes_name == "Classic 20") {
    pal1 <- tableau_color_pal(palettes_name)
    pal1(n_colors)
  }
  else if (palettes_name == "Nature 10") {
    cols <- c("#E04D50", "#4374A5", "#F08A21", "#2AB673",
              "#FCDDDE", "#70B5B0", "#DFE0EE", "#DFCDE4", "#FACB12",
              "#f9decf")
    cols[1:n_colors]
  }
  else if (palettes_name == "Blink 23") {
    cols <- c("#c10023", "#008e17", "#fb8500", "#f60000",
              "#FE0092", "#bc9000", "#4ffc00", "#00bcac", "#0099cc",
              "#D35400", "#00eefd", "#cf6bd6", "#99cc00", "#aa00ff",
              "#ff00ff", "#0053c8", "#f2a287", "#ffb3ff", "#800000",
              "#77a7b7", "#00896e", "#00cc99", "#007CC8")
    cols[1:n_colors]
  }
  else if (palettes_name == "Light 13") {
    cols <- c("#FD7446", "#709AE1", "#31A354", "#9EDAE5",
              "#DE9ED6", "#BCBD22", "#CE6DBD", "#DADAEB", "#FF9896",
              "#91D1C2", "#C7E9C0", "#6B6ECF", "#7B4173")
    cols[1:n_colors]
  }
  else if (palettes_name == "Hue n") {
    gg_color_hue(n_colors)
  }
  else {
    stop(paste0("chooseColors: check palettes_name! Unsupported palettes_name: ",
                palettes_name))
  }
  colors_new = adjust_transparency(colors, alpha = alpha)
  if (plot_colors) {
    barplot(rep(1, length(colors_new)), axes = FALSE, space = 0,
            col = colors_new)
  }
  return(colors_new)
}
Add_embed <- function (embed, seu, embed_name = "tSNE", assay = "RNA")
{
  row.names(embed) <- colnames(t(seuInt@assays$Spatial$counts))
  colnames(embed) <- paste0(embed_name, 1:ncol(embed))
  seu@reductions[[embed_name]] <- CreateDimReducObject(embeddings = embed,
                                                       key = paste0(toupper(embed_name), "_"), assay = assay)
  seu
}
mytheme <- function (base_size = 14, base_family = "")
{
  theme_grey(base_size = base_size, base_family = base_family) %+replace%
    theme(axis.title.x = element_text(margin = margin(10,0,0,0)),
          #axis.title.x = element_text(vjust = -1.5),
          #axis.title.y = element_text(margin = margin(0,20,0,0)),
          #axis.title.y = element_text(vjust = -0.1),
          axis.text = element_text(size = rel(0.8)),
          axis.ticks = element_line(colour = "black"),
          legend.key = element_rect(colour = "grey80"),
          panel.background = element_rect(fill = "white", colour = NA),
          panel.border = element_rect(fill = NA, colour = "grey50"),
          panel.grid.major = element_line(colour = "grey90", linewidth = 0.2),
          panel.grid.minor = element_line(colour = "grey98", linewidth = 0.5),
          strip.background = element_rect(fill = "grey80", colour = "grey50", linewidth = 0.2))
}