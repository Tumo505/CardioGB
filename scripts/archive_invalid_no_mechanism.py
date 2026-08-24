from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_SEEDS = {20260825, 20260826, 20260827, 20260828, 20260829}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recoverably archive the five legacy, unfair no-mechanism runs."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/final_full_ablations"),
    )
    parser.add_argument("--archive-name", default=None)
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    source = (results_dir / "no_mechanism").resolve()
    if source.parent != results_dir or not source.is_dir():
        raise SystemExit(f"Expected source directory is missing or unsafe: {source}")

    markers = sorted(source.glob("seed_*/done.json"))
    seeds = {int(path.parent.name.removeprefix("seed_")) for path in markers}
    if seeds != EXPECTED_SEEDS:
        raise SystemExit(f"Refusing archive: expected seeds {sorted(EXPECTED_SEEDS)}, found {sorted(seeds)}")

    metadata = [json.loads(path.read_text(encoding="utf-8")) for path in markers]
    if any(item.get("status") != "complete" for item in metadata):
        raise SystemExit("Refusing archive: at least one legacy run is not complete")
    if any(item.get("mechanistic_component") is False for item in metadata):
        raise SystemExit("Refusing archive: corrected no-mechanism output is present")

    archive_name = args.archive_name or (
        "archive_invalid_no_mechanism_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    destination = (results_dir.parent / archive_name).resolve()
    if destination.parent != results_dir.parent or destination.exists():
        raise SystemExit(f"Archive destination is unsafe or already exists: {destination}")

    shutil.move(str(source), str(destination))
    record = {
        "status": "archived",
        "reason": "Legacy no-mechanism runs did not disable the full mechanistic vector field.",
        "source": str(source),
        "destination": str(destination),
        "seeds": sorted(seeds),
        "archived_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (destination / "ARCHIVE_RECORD.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
