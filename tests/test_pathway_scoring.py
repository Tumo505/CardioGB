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

