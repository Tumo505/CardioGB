args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) {
  stop("Usage: inspect_seurat.R INPUT.rds")
}

suppressPackageStartupMessages(library(Seurat))
input_path <- normalizePath(args[[1]], mustWork = TRUE)
obj <- readRDS(input_path)

cat("r_version=", R.version.string, "\n", sep = "")
cat("seurat_version=", as.character(packageVersion("Seurat")), "\n", sep = "")
cat("class=", paste(class(obj), collapse = "|"), "\n", sep = "")
cat("object_size_bytes=", as.numeric(object.size(obj)), "\n", sep = "")
if (!inherits(obj, "Seurat")) quit(status = 2)

cat("assays=", paste(names(obj@assays), collapse = "|"), "\n", sep = "")
for (assay_name in names(obj@assays)) {
  assay <- obj[[assay_name]]
  assay_layers <- tryCatch(Layers(assay), error = function(e) character())
  cat("assay.", assay_name, ".class=", paste(class(assay), collapse = "|"), "\n", sep = "")
  cat("assay.", assay_name, ".features=", nrow(assay), "\n", sep = "")
  cat("assay.", assay_name, ".observations=", ncol(assay), "\n", sep = "")
  cat("assay.", assay_name, ".layers=", paste(assay_layers, collapse = "|"), "\n", sep = "")
}

metadata <- obj@meta.data
cat("metadata_rows=", nrow(metadata), "\n", sep = "")
cat("metadata_columns=", paste(colnames(metadata), collapse = "|"), "\n", sep = "")
for (column in c("orig.ident", "isolate", "cid", "time_points", "annotation")) {
  if (column %in% colnames(metadata)) {
    cat("metadata.", column, ".unique=", length(unique(metadata[[column]])), "\n", sep = "")
  }
}
for (column in c("x", "y")) {
  if (column %in% colnames(metadata)) {
    numeric_values <- suppressWarnings(as.numeric(as.character(metadata[[column]])))
    cat("metadata.", column, ".finite=", sum(is.finite(numeric_values)), "\n", sep = "")
  }
}
