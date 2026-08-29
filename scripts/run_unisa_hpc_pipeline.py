"""Resume and complete the revised CardioGB pipeline on one UNISA GPU node."""

from __future__ import annotations

import json
import os
import subprocess
import sys

from complete_identifiable_pipeline import (
    CANONICAL,
    ROOT,
    STAGING,
    manifest_status,
    promote_and_archive,
)
from run_revised_full_pipeline import stages


SEEDS_FIVE = [str(seed) for seed in range(20260825, 20260830)]
MANIFEST = ROOT / "results" / "final_pipeline_manifest.json"


def environment() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    env["PYTORCH_ALLOC_CONF"] = "backend:cudaMallocAsync"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync"
    env["CARDIOGB_PATCH_BATCH_SIZE"] = "8"
    env["CARDIOGB_CUDA_MEMORY_FRACTION"] = "0.94"
    return env


def run(command: list[str]) -> None:
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=environment(), check=True)


def finish_staging() -> None:
    if manifest_status(STAGING / "run_manifest.json") == "complete":
        return
    if not STAGING.exists() and manifest_status(CANONICAL / "run_manifest.json") == "complete":
        return
    run(
        [
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
            "--workers",
            "1",
            "--patch-batch-size",
            "8",
            "--memory-fraction-per-worker",
            "0.94",
        ]
    )
    if manifest_status(STAGING / "run_manifest.json") != "complete":
        raise RuntimeError("The five-seed staging benchmark is still incomplete")


def set_option(command: list[str], option: str, value: str) -> None:
    command[command.index(option) + 1] = value


def hpc_stages() -> list[tuple[str, list[str]]]:
    matrix = stages()
    e1 = matrix[0][1]
    set_option(e1, "--workers", "1")
    set_option(e1, "--patch-batch-size", "8")
    set_option(e1, "--memory-fraction-per-worker", "0.94")
    return matrix


def run_remaining_stages() -> None:
    matrix = hpc_stages()
    names = [name for name, _ in matrix]
    completed: list[str] = []
    if MANIFEST.is_file():
        completed = json.loads(MANIFEST.read_text(encoding="utf-8")).get("completed", [])
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    for name, command in matrix:
        if name in completed:
            print(f"SKIP completed stage {name}", flush=True)
            continue
        MANIFEST.write_text(
            json.dumps(
                {"status": "running", "completed": completed, "current_stage": name, "stages": names},
                indent=2,
            ),
            encoding="utf-8",
        )
        run(command)
        completed.append(name)
        MANIFEST.write_text(
            json.dumps(
                {"status": "partial", "completed": completed, "current_stage": None, "stages": names},
                indent=2,
            ),
            encoding="utf-8",
        )
    MANIFEST.write_text(
        json.dumps(
            {"status": "complete", "completed": completed, "current_stage": None, "stages": names},
            indent=2,
        ),
        encoding="utf-8",
    )
    run([sys.executable, "scripts/audit_manuscript_completion.py"])


def main() -> None:
    finish_staging()
    if STAGING.exists():
        archive_root = promote_and_archive()
        print(f"Archived superseded outputs under {archive_root}", flush=True)
    elif manifest_status(CANONICAL / "run_manifest.json") != "complete":
        raise RuntimeError("Neither a resumable staging benchmark nor a complete canonical benchmark exists")
    run_remaining_stages()


if __name__ == "__main__":
    main()
