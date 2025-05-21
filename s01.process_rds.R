#!/usr/bin/env Rscript

## ————————————————
## 单文件 Seurat 数据提取脚本（始终提取 Count 和 Metadata）
##  Usage:
##    Rscript process_seurat.R \
##      /path/to/sample.seurat.rds \
##      /output/base/dir \
##      genes.txt

# /mnt/public3/caogs/anaconda3/envs/R4.3/bin/Rscript s01_extra_minor_matrix_use_h5.R ./data/pbmc_demo.rds ./output data/ranked_hvg.list 

## ————————————————

suppressPackageStartupMessages({
  library(fs)
  library(Seurat)
  library(dplyr)
  library(rhdf5)
})

# 解析命令行参数
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("Usage: Rscript process_seurat.R <rds_file> <out_dir> <ranked_hvg_file>")
}

rds_file       <- args[1]
out_dir        <- args[2]
gene_list_file <- args[3]

# 校验输入
stopifnot(file.exists(rds_file))
stopifnot(file.exists(gene_list_file))

# 准备输出目录
dir_create(out_dir, recurse = TRUE)
matrix_dir <- out_dir
meta_dir   <- out_dir
for (d in list(matrix_dir, meta_dir)) {
  dir_create(d)
  message("Ensured output directory: ", d)
}

# 读取基因列表
gene_list <- readLines(gene_list_file) %>% unique()
message("Loaded ", length(gene_list), " unique genes from ", gene_list_file)

# 工具函数：保存 CSV
save_csv <- function(df, path, label) {
  write.csv(df, file = path, row.names = TRUE)
  message("✔ Saved ", label, " to: ", path)
}

# 工具函数：保存 HDF5 矩阵
save_h5 <- function(mat, cell_names, gene_names, path, label) {
  if (file.exists(path)) file.remove(path)
  h5createFile(path)
  grp <- paste0(label, "/")
  h5createGroup(path, grp)
  mat <- t(mat)
  dims <- dim(mat)
  chunk <- c(min(1000, dims[1]), dims[2])
  h5createDataset(path, paste0(grp, "data"), dims = dims,
                  chunk = chunk, storage.mode = "double")
  h5write(mat, path, paste0(grp, "data"))
  h5write(cell_names, path, paste0(grp, "cell_names"))
  h5write(gene_names, path, paste0(grp, "gene_names"))
  message("✔ Saved HDF5 ", label, " to: ", path)
}

# 加载 Seurat 对象
message("Loading Seurat object from: ", rds_file)
seurat_obj <- readRDS(rds_file)

# 选择 assay：integrated 优先，否则 RNA
if ("integrated" %in% Assays(seurat_obj)) {
  DefaultAssay(seurat_obj) <- "integrated"
  message("Using assay: integrated")
} else {
  DefaultAssay(seurat_obj) <- "RNA"
  message("Using assay: RNA")
}

# 匹配基因
rna_mat <- GetAssayData(seurat_obj, assay = DefaultAssay(seurat_obj), slot = "data")
rna_genes <- rownames(rna_mat)
matched_genes <- intersect(gene_list, rna_genes)
if (length(matched_genes) == 0) stop("❌ No genes from gene_list found in the Seurat object.")
message("Matched ", length(matched_genes), "/", length(gene_list), " genes.")

# 提取并保存 Count 数据
message("Extracting count data...")
count_mat <- rna_mat[matched_genes, , drop = FALSE]
count_mat <- t(as.matrix(count_mat))
h5_file <- path(matrix_dir, paste0(path_ext_remove(basename(rds_file)), ".h5"))
save_h5(count_mat, rownames(count_mat), colnames(count_mat), h5_file, "count")

# 提取并保存 Metadata
message("Extracting metadata...")
meta <- seurat_obj@meta.data
csv_file <- path(meta_dir, paste0(path_ext_remove(basename(rds_file)), ".meta.csv"))
save_csv(meta, csv_file, "metadata")

message("All done!")
