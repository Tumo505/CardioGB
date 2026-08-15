import pandas as pd

from cardiogb.data.splits import grouped_split


def test_grouped_split_keeps_biological_units_intact() -> None:
    metadata = pd.DataFrame(
        {
            "heart": [f"h{i}" for i in range(10) for _ in range(3)],
            "stage": ["a", "b", "c"] * 10,
        }
    )
    train, validation, test, definition = grouped_split(
        metadata, group_column="heart", stage_column="stage", seed=7
    )
    assert (train | validation | test).all()
    assert set(definition.train_samples).isdisjoint(definition.test_samples)
    for heart, rows in metadata.groupby("heart").groups.items():
        memberships = {(bool(train[i]), bool(validation[i]), bool(test[i])) for i in rows}
        assert len(memberships) == 1, heart
