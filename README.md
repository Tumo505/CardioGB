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

After installing R plus Seurat, the primary execution sequence is:

```powershell
python scripts\preprocess_zebrafish.py --rds data\raw\zebrafish\regeneration\observed\Stereo-seq-regeneration.rds --bundle data\interim\zebrafish_counts --output data\interim\zebrafish_observed.h5ad
python scripts\score_pathways.py --input data\interim\zebrafish_observed.h5ad --output data\processed\zebrafish_states.npz
python scripts\build_graphs.py --input data\interim\zebrafish_observed.h5ad --output data\processed\graphs.npz --k 4 6 8 12 16 --qc-output results\tables\graph_qc.csv
python scripts\run_benchmark.py --data data\processed\zebrafish_states.npz
```

Large source datasets are intentionally excluded from version control.
