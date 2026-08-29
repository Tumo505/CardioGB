"""Finish, promote, and repeatedly resume the identifiable CardioGB pipeline."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
STAGING = RESULTS / "identifiable_revision_multiseed"
CANONICAL = RESULTS / "final_full_multiseed"
SEEDS_FIVE = [str(seed) for seed in range(20260825, 20260830)]


def process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def manifest_status(path: Path) -> str:
    if not path.is_file():
        return "missing"
    try:
        return str(json.loads(path.read_text(encoding="utf-8")).get("status", "partial"))
    except (OSError, json.JSONDecodeError):
        return "invalid"


def run(command: list[str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = "src"
    environment["PYTORCH_ALLOC_CONF"] = "backend:cudaMallocAsync"
    environment["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync"
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[{datetime.now().isoformat()}] RUN {' '.join(command)}\n")
        handle.flush()
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
        handle.write(f"[{datetime.now().isoformat()}] EXIT {completed.returncode}\n")
        return completed.returncode


def wait_for_active_benchmark(pid: int) -> None:
    while process_alive(pid):
        time.sleep(60)


def finish_staging_benchmark(log: Path) -> None:
    if not STAGING.exists() and manifest_status(CANONICAL / "run_manifest.json") == "complete":
        return
    command = [
        sys.executable,
        "scripts/run_parallel_multiseed_benchmark.py",
        "--data",
        "data/processed/zebrafish_states.npz",
        "--output-dir",
        str(STAGING.relative_to(ROOT)),
        "--epochs",
        "200",
        "--seeds",
        *SEEDS_FIVE,
        "--workers", "2",
        "--patch-batch-size", "8",
        "--memory-fraction-per-worker", "0.48",
    ]
    while manifest_status(STAGING / "run_manifest.json") != "complete":
        run(command, log)
        if manifest_status(STAGING / "run_manifest.json") != "complete":
            time.sleep(60)


def inside_workspace(path: Path) -> Path:
    resolved = path.resolve()
    resolved.relative_to(ROOT)
    return resolved


def archive(path: Path, archive_root: Path) -> None:
    path = inside_workspace(path)
    if not path.exists():
        return
    relative = path.relative_to(ROOT)
    destination = archive_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"archive destination already exists: {destination}")
    shutil.move(str(path), str(destination))


def promote_and_archive() -> Path:
    archive_root = RESULTS / "archive" / "pre_identifiability_20260826"
    if CANONICAL.exists() and not (archive_root / CANONICAL.relative_to(ROOT)).exists():
        archive(CANONICAL, archive_root)
    if STAGING.exists():
        CANONICAL.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(STAGING), str(CANONICAL))
    elif not CANONICAL.exists():
        raise FileNotFoundError(STAGING)

    paths = [
        RESULTS / "synthetic_recovery_full",
        RESULTS / "e2_interpolation_revised",
        RESULTS / "e3_extrapolation_revised",
        RESULTS / "e3_extrapolation_horizon_calibrated",
        RESULTS / "e4_group_cv_full",
        RESULTS / "final_full_ablations",
        RESULTS / "final_full_ensemble",
        RESULTS / "external_predictive_validation_revised",
        RESULTS / "e7_full_interpretation",
        RESULTS / "formal_statistics_revised",
        RESULTS / "mouse_species_adapter_revised",
        RESULTS / "verification",
        RESULTS / "final_pipeline_manifest.json",
        ROOT / "figures" / "manuscript",
        ROOT / "manuscript" / "main_tables",
        ROOT / "manuscript" / "supplementary_tables",
        ROOT / "manuscript" / "manuscript_sections.md",
    ]
    for path in paths:
        destination = archive_root / path.resolve().relative_to(ROOT)
        if path.exists() and not destination.exists():
            archive(path, archive_root)
    return archive_root


def finish_full_pipeline(log: Path) -> None:
    command = [sys.executable, "scripts/run_revised_full_pipeline.py"]
    manifest = RESULTS / "final_pipeline_manifest.json"
    while True:
        return_code = run(command, log)
        if return_code == 0 and manifest_status(manifest) == "complete":
            return
        time.sleep(60)




def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--active-benchmark-pid", type=int, default=43680)
    parser.add_argument(
        "--log", type=Path, default=Path("results/completion_supervisor/pipeline.log")
    )
    args = parser.parse_args()
    log = ROOT / args.log
    wait_for_active_benchmark(args.active_benchmark_pid)
    finish_staging_benchmark(log)
    archive_root = promote_and_archive()
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"[{datetime.now().isoformat()}] archived prior outputs to {archive_root}\n")
    finish_full_pipeline(log)
    print(json.dumps({"status": "complete", "archive": str(archive_root), "log": str(log)}))


if __name__ == "__main__":
    main()
