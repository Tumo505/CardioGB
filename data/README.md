# Data layout

- `raw/zebrafish/`: immutable primary and optional zebrafish source files.
- `external/mouse/`: neonatal mouse spatial and scRNA validation sources.
- `external/human_mi/zenodo_6578047/`: checksum-verified processed human MI files.
- `external/human_mi/hca_export/`: raw 10x/HCA export and CellxGene derivatives.
- `external/human_mi/loose_fragments/`: redundant partial download fragments,
  retained for provenance but excluded from primary analysis.
- `interim/`: reproducible intermediate conversions.
- `processed/`: model-ready, versioned outputs.

Source data are treated as immutable. Processing code must write to `interim`
or `processed`, never back into source directories.

