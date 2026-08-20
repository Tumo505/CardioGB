# Six-state biological programme curation

## Version and scope

`atlas_curated_v1` is the first evidence-anchored gene-set version for the six
CardioGB state variables. It replaces the engineering placeholders. It is
curated for the observed zebrafish Stereo-seq counts and is not presented as a
universal cross-species signature.

Primary evidence is the open-access Li et al. zebrafish regeneration atlas
(Nature Communications 2025, DOI `10.1038/s41467-025-59070-0`) and the marker
script supplied with its Zenodo release. The article is available at
https://www.nature.com/articles/s41467-025-59070-0 and its indexed record is
https://pubmed.ncbi.nlm.nih.gov/40253397/.

## Programme definitions

| State | Biological interpretation | Atlas anchors | Expected behaviour |
|---|---|---|---|
| I | Injury-associated innate immune/inflammatory activity | `tnfa`, `grnas`, macrophage and neutrophil markers | Early rise around 12 hpa–1 dpa; later immune domains may persist |
| A | Early endocardial, epicardial, and organ-wide injury activation | `aldh1a2`, `cd151`, `ackr3b`, `inhbaa`, `sema3aa`, `tnfrsf11b`, `cd63`, `hif1ab` | Strong 6 hpa response, declining after early injury; epicardial component near 3 dpa |
| F | Pro-regenerative fibroblast/ECM remodelling | `col12a1a/b`, `lum`, collagen/fibronectin and fibroblast markers | Wound-localised rise around 3–7 dpa |
| C | CM dedifferentiation, cell-cycle entry, proliferation, and repair | `nppa/b`, `mustn1b`, `pcna`, `cdk1`, `ccng1/2`, `ccn1`, `tpm4a` | Dedifferentiation from 12 hpa; proliferation peaks near 7 dpa |
| V | Endothelial/coronary vascular scaffold and revascularisation | `kdrl`, `pecam1`, `cdh5`, `plvap*`, `cxcl12b`, `tagln` | Tissue-replacement and late vascular-restoration signal |
| M | Contractile myocardium, oxidative metabolism, and calcium handling | sarcomere genes plus `cox6a2`, `cox5b2`, `cox4i1l`, `atp2a2a`, `slc8a1a`, `ryr2b` | Myocardium-enriched throughout; late maturation features near 14–28 dpa |

The exact versioned lists and machine-readable provenance live in
`configs/pathways.yaml`.

## Dataset checks

- All curated genes match the observed 29,539-gene `Spatial` counts layer.
- Scores are computed from library-size-normalised `log1p` counts.
- Each gene is standardised across spots before within-programme averaging.
- Scores are clipped using the 1st and 99th percentiles and mapped to `[0,1]`
  to reduce sensitivity to extreme spots.
- Pairwise state correlations are moderate; the largest is C–M, consistent
  with both programmes being cardiomyocyte-related but biologically distinct.
- The stage summaries reproduce early A/I elevation, 3–7 dpa F elevation, and
  a 7 dpa C maximum. The 28 dpa I elevation is retained rather than edited
  away because the atlas itself reports a macrophage-enriched outflow-tract
  domain at that stage.
- Domain QC places I in the macrophage domain, A in activated border-zone
  myocardium, F in regenerative fibroblasts, and C in proliferative border-zone
  myocardium. V is biologically plausible but less domain-specific.
- Rank-mean scores agree strongly with the primary mean-scaled scores. The
  module-score sensitivity is less stable for V and M stage ordering, so it is
  retained as a sensitivity analysis rather than the primary definition.
- The Ensembl batch orthology table maps 64/88 source genes (72.7%) to each of
  mouse and human. Unresolved zebrafish symbols and paralogs remain explicit;
  they are not silently replaced by hand-selected mammalian genes.

## Interpretation guardrails

Scores measure transcriptional programme activity in multi-cell Stereo-seq
spots. They are not direct cell counts, causal pathway activities, or tracked
cell trajectories. `atlas_curated_v1` is frozen for the core zebrafish pilot
after domain-level QC, alternative-scoring sensitivity analysis, and orthology
coverage were recorded. It must not be altered in response to benchmark
outcomes. Any later revision is a new version and requires the same validation
sequence.
