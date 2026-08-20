# Neonatal mouse external validation

## Scope

External validation is pathway-level and descriptive. It does not perform
zero-shot transfer of the zebrafish dynamical model. Four neonatal mouse Visium
samples are compared across ordinal repair phases, supported by eight mouse
scRNA-seq MI/sham samples.

## Processed data

- Visium: 6,150 tissue spots, 32,285 genes, stages 3, 7, 14, and 21 days.
- scRNA-seq: P1 and P8 hearts, 1 and 3 days, MI and sham conditions.
- Only filtered Space Ranger matrices and coordinates were extracted; BAMs,
  raw matrices, histology images, and pipeline intermediates remain archived.

## Cross-species pathway translation

The automated Ensembl table is retained as a conservative provenance report,
but it contains biologically implausible low-identity choices for several
zebrafish paralogs. `configs/mouse_pathways.yaml` therefore freezes a reviewed
mouse translation. Examples include `atp2a2a -> Atp2a2`, `mpx -> Mpo`, and
`lyz -> Lyz2`. Duplicated zebrafish paralogs collapse to one mouse gene and
ambiguous mappings are omitted.

All translated genes used for scoring are present in the Visium feature matrix.
Programme scores use the same library-size normalization, per-gene scaling, and
robust [0,1] output transformation as the primary analysis.

## Results

Across the four ordinal phase matches, activation and cardiomyocyte-regeneration
programmes have Spearman rho 0.8, fibroblast/ECM rho 0.6, vascularisation rho
0.4, mature myocardium rho 0.2, and inflammation rho -0.8. None is individually
significant under an exact four-point permutation test; the minimum attainable
resolution is poor and the analysis is evidence of pattern agreement/divergence,
not confirmation of dynamical conservation.

The scRNA-seq contrasts are consistent with an early P1 MI-associated increase
in inflammation, activation, fibroblast/ECM, and cardiomyocyte-regeneration
scores. These contrasts have one sample in each age/day/condition cell and are
therefore descriptive rather than inferential.

## Guardrails

- There is one Visium sample per stage, so spots are not treated as biological
  replicates.
- Four matched phases cannot support strong pathway-specific significance.
- Cross-species score correlation is not causal evidence and does not validate
  individual mechanistic parameters.
- Direct model transfer remains disabled by design.

