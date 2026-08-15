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
