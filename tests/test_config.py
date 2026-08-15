from cardiogb.utils.config import deep_merge


def test_deep_merge_preserves_nested_defaults() -> None:
    base = {"graph": {"k": 8, "symmetric": True}, "seed": 1}
    override = {"graph": {"k": 12}}
    assert deep_merge(base, override) == {
        "graph": {"k": 12, "symmetric": True},
        "seed": 1,
    }
    assert base["graph"]["k"] == 8

