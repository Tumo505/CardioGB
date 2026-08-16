# CardioGB

CardioGB is a grey-box graph neural differential-equation framework for
cross-sectional spatial cardiac-regeneration data. It decomposes a learned
vector field into an interpretable mechanistic term and a spatial neural
residual:

```text
observed dynamics = known biology + learned discrepancy
```

The primary dataset is the eight-stage zebrafish heart-regeneration atlas.
Neonatal mouse data provide cross-species validation, and the human myocardial
infarction collection provides optional regeneration-versus-fibrosis analysis.

## Development setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,ode]"
pytest
```

Inspect the configured datasets with:

```powershell
python scripts\inspect_data.py --config configs\data.yaml
python scripts\check_environment.py
```

The project-local R/Seurat environment is reproducible from
`environment-r.yml`. The current workspace uses `.r-env` (R 4.4.3 and Seurat
5.5.1); run R through `conda run` so its native libraries are on `PATH`:

```powershell
conda env create --prefix .\.r-env --file environment-r.yml
conda run --prefix .\.r-env Rscript scripts\inspect_seurat.R data\raw\zebrafish\regeneration\observed\Stereo-seq-regeneration.rds
python scripts\preprocess_zebrafish.py --rds data\raw\zebrafish\regeneration\observed\Stereo-seq-regeneration.rds --bundle data\interim\zebrafish_counts --output data\interim\zebrafish_observed.h5ad --rscript .\.r-env\Scripts\Rscript.exe
python scripts\score_pathways.py --input data\interim\zebrafish_observed.h5ad --output data\processed\zebrafish_states.npz
python scripts\build_graphs.py --input data\interim\zebrafish_observed.h5ad --output data\processed\graphs.npz --k 4 6 8 12 16 --qc-output results\tables\graph_qc.csv
python scripts\pathway_sensitivity.py --input data\interim\zebrafish_observed.h5ad
python scripts\run_benchmark.py --data data\processed\zebrafish_states.npz --output-dir results\real_e1
```

The preprocessed observed-count H5AD and state/graph files already exist in this
workspace; pass `--skip-export` to preprocessing when repeating only Python-side
conversion.

Large source datasets are intentionally excluded from version control.
