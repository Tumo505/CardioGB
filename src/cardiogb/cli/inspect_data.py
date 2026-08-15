"""Inspect configured CardioGB datasets without loading full expression matrices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cardiogb.data.loaders import read_metadata_tsv, summarize_metadata, validate_stage_set
from cardiogb.data.rds_conversion import inspect_rds
from cardiogb.utils.config import load_yaml, resolve_project_path


def inspect(config_path: str | Path) -> dict[str, object]:
    config_path = Path(config_path).resolve()
    config = load_yaml(config_path)
    root_value = config.get("project_root", "..")
    project_root = (config_path.parent / root_value).resolve()
    zebrafish = config["datasets"]["zebrafish"]
    schema = zebrafish["metadata"]
    output: dict[str, object] = {"project_root": str(project_root), "zebrafish": {}}
    for modality, key in (("spatial", "spatial_metadata"), ("scrna", "scrna_metadata")):
        path = resolve_project_path(zebrafish[key], project_root)
        frame = read_metadata_tsv(path, schema.get("row_id_column", "record_id"))
        modality_schema = dict(schema)
        if modality == "scrna":
            modality_schema.pop("coordinate_columns", None)
            modality_schema.pop("isolate_column", None)
            modality_schema.pop("biological_unit_column", None)
        summary = summarize_metadata(frame, path, modality_schema)
        validate_stage_set(summary)
        output["zebrafish"][modality] = summary.to_dict()  # type: ignore[index]
    for modality, key in (("spatial_rds", "spatial_rds"), ("scrna_rds", "scrna_rds")):
        path = resolve_project_path(zebrafish[key], project_root)
        output["zebrafish"][modality] = inspect_rds(path).to_dict()  # type: ignore[index]
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = inspect(args.config)
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()

