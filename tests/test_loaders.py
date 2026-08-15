from pathlib import Path

from cardiogb.data.loaders import read_metadata_tsv


def test_unlabelled_row_id_is_repaired(tmp_path: Path) -> None:
    path = tmp_path / "meta.tsv"
    path.write_text("stage\tx\nspot1\t0\t1.5\n", encoding="utf-8")
    frame = read_metadata_tsv(path)
    assert list(frame.columns) == ["record_id", "stage", "x"]
    assert frame.loc[0, "record_id"] == "spot1"
    assert frame.attrs["repaired_unlabelled_row_id"] is True

