"""Dataset discovery and metadata loading."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


EXPECTED_STAGES = (
    "uninjured",
    "6 hpa",
    "12 hpa",
    "1 dpa",
    "3 dpa",
    "7 dpa",
    "14 dpa",
    "28 dpa",
)


@dataclass(frozen=True)
class MetadataSummary:
    path: str
    rows: int
    columns: tuple[str, ...]
    repaired_unlabelled_row_id: bool
    unique_sections: int | None
    unique_isolates: int | None
    unique_biological_units: int | None
    stage_counts: dict[str, int]
    annotation_count: int | None
    invalid_coordinate_rows: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _field_widths(path: Path, delimiter: str = "\t") -> tuple[list[str], int]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        header = next(reader)
        first_row = next(reader, None)
    if first_row is None:
        raise ValueError(f"Metadata file has a header but no rows: {path}")
    return header, len(first_row)


def read_metadata_tsv(path: str | Path, row_id_name: str = "record_id") -> pd.DataFrame:
    """Read a TSV, repairing the atlas files' unlabeled first field explicitly."""
    metadata_path = Path(path).expanduser().resolve()
    header, row_width = _field_widths(metadata_path)
    if row_width == len(header):
        frame = pd.read_csv(metadata_path, sep="\t", low_memory=False)
    elif row_width == len(header) + 1:
        frame = pd.read_csv(
            metadata_path,
            sep="\t",
            header=0,
            names=[row_id_name, *header],
            low_memory=False,
        )
        frame.attrs["repaired_unlabelled_row_id"] = True
    else:
        raise ValueError(
            f"Unexpected TSV width in {metadata_path}: header={len(header)}, row={row_width}"
        )
    if frame.columns.duplicated().any():
        raise ValueError(f"Duplicate metadata columns in {metadata_path}")
    return frame


def summarize_metadata(
    frame: pd.DataFrame,
    path: str | Path,
    schema: Mapping[str, Any],
) -> MetadataSummary:
    """Validate configured columns and summarize biological units and stages."""
    required = [
        schema["section_column"],
        schema["stage_column"],
        schema["annotation_column"],
    ]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Missing required metadata columns: {missing}")

    coordinates = schema.get("coordinate_columns", [])
    invalid_coordinate_rows: int | None = None
    if coordinates:
        missing_coordinates = [column for column in coordinates if column not in frame]
        if missing_coordinates:
            raise ValueError(f"Missing coordinate columns: {missing_coordinates}")
        numeric = frame[coordinates].apply(pd.to_numeric, errors="coerce")
        invalid_coordinate_rows = int(numeric.isna().any(axis=1).sum())

    stage_column = schema["stage_column"]
    stage_counts = {str(key): int(value) for key, value in frame[stage_column].value_counts().items()}
    return MetadataSummary(
        path=str(Path(path).resolve()),
        rows=len(frame),
        columns=tuple(map(str, frame.columns)),
        repaired_unlabelled_row_id=bool(frame.attrs.get("repaired_unlabelled_row_id", False)),
        unique_sections=int(frame[schema["section_column"]].nunique()),
        unique_isolates=_optional_nunique(frame, schema.get("isolate_column")),
        unique_biological_units=_optional_nunique(frame, schema.get("biological_unit_column")),
        stage_counts=stage_counts,
        annotation_count=int(frame[schema["annotation_column"]].nunique()),
        invalid_coordinate_rows=invalid_coordinate_rows,
    )


def _optional_nunique(frame: pd.DataFrame, column: str | None) -> int | None:
    if not column or column not in frame:
        return None
    return int(frame[column].nunique())


def validate_stage_set(summary: MetadataSummary, expected: tuple[str, ...] = EXPECTED_STAGES) -> None:
    observed = set(summary.stage_counts)
    if observed != set(expected):
        raise ValueError(f"Stage mismatch: missing={set(expected)-observed}, extra={observed-set(expected)}")

