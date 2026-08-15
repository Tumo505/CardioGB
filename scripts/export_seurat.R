args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) {
  stop("Usage: export_seurat.R INPUT.rds OUTPUT_DIR ASSAY")
}

suppressPackageStartupMessages(library(Seurat))
suppressPackageStartupMessages(library(Matrix))

input_path <- normalizePath(args[[1]], mustWork = TRUE)
output_dir <- normalizePath(args[[2]], mustWork = FALSE)
assay_name <- args[[3]]
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

obj <- readRDS(input_path)
if (!inherits(obj, "Seurat")) {
  stop(paste("Expected a Seurat object, found:", paste(class(obj), collapse = ", ")))
}
if (!(assay_name %in% names(obj@assays))) {
  stop(paste("Assay not found:", assay_name, "available:", paste(names(obj@assays), collapse = ", ")))
}

get_counts <- function(object, assay) {
  result <- tryCatch(
    GetAssayData(object, assay = assay, layer = "counts"),
    error = function(e) NULL
  )
  if (is.null(result)) {
    result <- GetAssayData(object, assay = assay, slot = "counts")
  }
  result
}

counts <- get_counts(obj, assay_name)
if (nrow(counts) == 0 || ncol(counts) == 0) {
  stop("The selected assay has no observed count matrix")
}
metadata <- obj@meta.data
if (!identical(colnames(counts), rownames(metadata))) {
  if (!all(colnames(counts) %in% rownames(metadata))) {
    stop("Expression barcodes are not all present in metadata")
  }
  metadata <- metadata[colnames(counts), , drop = FALSE]
}

writeMM(counts, file.path(output_dir, "matrix.mtx"))
write.table(rownames(counts), file.path(output_dir, "genes.tsv"), quote = FALSE,
            row.names = FALSE, col.names = FALSE, sep = "\t")
write.table(colnames(counts), file.path(output_dir, "barcodes.tsv"), quote = FALSE,
            row.names = FALSE, col.names = FALSE, sep = "\t")
write.table(metadata, file.path(output_dir, "metadata.tsv"), quote = FALSE,
            row.names = TRUE, col.names = NA, sep = "\t")

cat("class=", paste(class(obj), collapse = "|"), "\n", sep = "")
cat("assay=", assay_name, "\n", sep = "")
cat("genes=", nrow(counts), "\n", sep = "")
cat("observations=", ncol(counts), "\n", sep = "")
