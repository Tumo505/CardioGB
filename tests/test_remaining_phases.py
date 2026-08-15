import numpy as np
import torch

from cardiogb.ablations import disable_interaction, shuffle_graph
from cardiogb.metrics.uncertainty import distribution_mean_calibration
from cardiogb.models.message_passing import TorchGraph
from cardiogb.statistics import grouped_bootstrap, paired_group_permutation_test
from cardiogb.validation.cross_species import matched_pathway_correlations, orthology_coverage


def test_grouped_statistics_resample_units():
    result = grouped_bootstrap(
        np.array([1.0, 2.0, 3.0, 4.0]), np.array(["a", "a", "b", "b"]), n_resamples=50
    )
    assert result["n_biological_units"] == 2
    assert result["lower"] <= result["estimate"] <= result["upper"]
    assert 0 <= paired_group_permutation_test(
        np.array([1.0, 2.0]), np.array([2.0, 3.0]), n_permutations=50
    )["p_value"] <= 1


def test_ablation_and_graph_shuffle():
    config = {"interactions": [{"source": "I", "target": "F", "parameter": "a"}]}
    assert disable_interaction(config, "I->F")["interactions"] == []
    graph = TorchGraph(torch.tensor([[0, 1], [1, 0]]), torch.ones(2, 3))
    shuffled = shuffle_graph(graph, seed=2)
    assert shuffled.edge_index.shape == graph.edge_index.shape


def test_uncertainty_and_cross_species_summaries():
    predictions = np.stack([np.zeros((4, 2)), np.ones((4, 2))])
    calibration = distribution_mean_calibration(predictions, np.full((7, 2), 0.5))
    assert calibration["coverage"] == 1.0
    correlations = matched_pathway_correlations(
        np.arange(12).reshape(6, 2), np.arange(12).reshape(6, 2), ["I", "A"]
    )
    assert correlations == {"I": 1.0, "A": 1.0}
    assert orthology_coverage({"I": ["a", "b"]}, {"a": "A"})["I"]["coverage"] == 0.5
