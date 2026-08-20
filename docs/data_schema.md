# Verified data schema

## Zebrafish regeneration

The metadata files were verified before implementation:

- Spatial metadata: 159,293 rows, 105 sections, 35 isolates, 24 `cid`
  biological units, eight stages, and 20 annotations.
- scRNA metadata: 206,719 cells, eight stage-level `orig.ident` values, 27
  libraries, and 25 annotations.
- All spatial `x` and `y` values parse as numeric.

Both TSV files have one more field in each data row than in the header. The
first field is an unlabeled record identifier; all declared columns otherwise
shift by one under a naïve parser. `cardiogb.data.loaders.read_metadata_tsv`
detects and repairs this layout explicitly.

The regeneration stages map to days as follows:

| Label | Days |
|---|---:|
| uninjured | 0 |
| 6 hpa | 0.25 |
| 12 hpa | 0.5 |
| 1 dpa | 1 |
| 3 dpa | 3 |
| 7 dpa | 7 |
| 14 dpa | 14 |
| 28 dpa | 28 |

The observed spatial RDS was verified using project-local R 4.4.3 and Seurat
5.5.1. It is a Seurat object with one `Spatial` assay (legacy `Assay` class),
29,539 features, 159,293 observations, and explicit `counts` and `data` layers.
The primary Python H5AD was exported exclusively from the observed `counts`
layer. Its shape is 159,293 spots × 29,539 genes and it retains all metadata
plus finite `spatial` coordinates. No imputed expression is used.

The model-ready state file contains 159,293 × 6 bounded programme scores, 105
section identifiers, 24 `cid` biological units, eight numeric stages, and the
20 supplied tissue-domain annotations.

## Human MI

All 27 processed Visium objects from Zenodo 6578047 are checksum-verified and
readable. Together they contain 88,704 spots. Every object has a finite,
unique `X_spatial` coordinate pair for each spot and retains raw expression.

## Scientific interpretation

All zebrafish stages are cross-sectional. Record identifiers, spots, sections,
and biological units must never be interpreted as tracked longitudinal
entities across stages.
