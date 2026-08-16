import numpy as np
from scipy import sparse

from cardiogb.data.pathway_scoring import score_pathways


def test_sparse_pathway_scoring_and_gene_diagnostics() -> None:
    expression = sparse.csr_matrix([[0, 1, 3], [2, 1, 0], [4, 1, 1]])
    result = score_pathways(
        expression,
        ["GeneA", "GeneB", "GeneC"],
        {"state": ["genea", "GENEC", "missing"]},
        min_genes=2,
    )
    assert result.values.shape == (3, 1)
    assert result.values.min() == 0
    assert result.values.max() == 1
    assert result.matched_genes["state"] == ("GeneA", "GeneC")
    assert result.missing_genes["state"] == ("missing",)


def test_rank_and_module_score_variants_are_bounded() -> None:
    expression = sparse.csr_matrix(np.arange(60, dtype=float).reshape(10, 6))
    for method in ("rank_mean", "module_score"):
        result = score_pathways(
            expression,
            [f"g{i}" for i in range(6)],
            {"state": ["g0", "g1"]},
            method=method,
            output_scaling="robust_minmax",
            min_genes=2,
            module_bins=2,
        )
        assert np.isfinite(result.values).all()
        assert result.values.min() >= 0 and result.values.max() <= 1
