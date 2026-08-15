import numpy as np

from cardiogb.data.graphs import build_spatial_knn_graph


def test_graph_never_crosses_sections() -> None:
    coordinates = np.array([[0, 0], [1, 0], [2, 0], [0, 0], [1, 0]], dtype=float)
    sections = np.array(["a", "a", "a", "b", "b"])
    graph = build_spatial_knn_graph(coordinates, sections, k=1, symmetric=True)
    source, target = graph.edge_index
    assert np.all(sections[source] == sections[target])
    assert graph.edge_attr.shape[1] == 3
    assert np.allclose(graph.edge_attr[:, 0], 1.0)


def test_graph_rejects_non_finite_coordinates() -> None:
    coordinates = np.array([[0, 0], [np.nan, 1]])
    try:
        build_spatial_knn_graph(coordinates, ["a", "a"])
    except ValueError as error:
        assert "non-finite" in str(error)
    else:
        raise AssertionError("Expected non-finite coordinates to be rejected")

