import numpy as np

from cardiogb.data.state_dataset import StateDataset


def test_state_dataset_builds_unmatched_adjacent_transitions() -> None:
    dataset = StateDataset(
        states=np.random.default_rng(1).random((12, 6)).astype(np.float32),
        coordinates=np.tile(np.array([[0, 0], [1, 0], [0, 1]]), (4, 1)),
        sections=np.repeat(["s0", "s1", "s2", "s3"], 3),
        times=np.repeat([0.0, 1.0, 3.0, 7.0], 3),
        groups=np.repeat(["g0", "g1", "g2", "g3"], 3),
        state_names=("I", "A", "F", "C", "V", "M"),
    )
    transitions = dataset.transitions(k=1)
    assert len(transitions) == 3
    assert transitions[0].source_states.shape == (3, 6)
    assert transitions[0].target_states.shape == (3, 6)


def test_state_dataset_round_trip_coerces_object_labels(tmp_path) -> None:
    dataset = StateDataset(
        states=np.zeros((2, 6), dtype=np.float32),
        coordinates=np.zeros((2, 2)),
        sections=np.array(["s1", "s2"], dtype=object),
        times=np.array([0.0, 1.0]),
        groups=np.array(["g1", "g2"], dtype=object),
        state_names=("I", "A", "F", "C", "V", "M"),
    )
    path = tmp_path / "object_labels.npz"
    dataset.save(path)
    assert StateDataset.load(path).sections.tolist() == ["s1", "s2"]


def test_state_dataset_builds_bounded_section_patches() -> None:
    rng = np.random.default_rng(4)
    dataset = StateDataset(
        states=rng.random((11, 6)).astype(np.float32),
        coordinates=np.column_stack((np.arange(11), np.zeros(11))),
        sections=np.array(["s0"] * 4 + ["s1"] * 4 + ["s2"] * 3),
        times=np.array([0.0] * 8 + [1.0] * 3),
        groups=np.array(["g0"] * 4 + ["g1"] * 4 + ["g2"] * 3),
        state_names=("I", "A", "F", "C", "V", "M"),
    )
    transitions = dataset.transitions(k=1, max_nodes=3)
    assert len(transitions) == 4
    assert all(len(item.source_states) <= 3 for item in transitions)
    assert {item.evaluation_group for item in transitions} == {"0_to_1"}
    assert sum(len(item.source_states) for item in transitions) == 8
