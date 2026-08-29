"""Run independent benchmark seeds concurrently and merge their outputs safely."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from cardiogb.utils.io import atomic_json
from run_multiseed_benchmark import LEARNED_MODELS, aggregate


ROOT = Path(__file__).resolve().parents[1]


def seed_complete(output: Path, seed: int) -> bool:
    metrics = output / f"seed_{seed}" / "metrics"
    return all(
        (metrics / f"{model}_test.csv").is_file()
        for model in ("persistence", *LEARNED_MODELS)
    )


def completed_records(output: Path, seeds: list[int]) -> list[dict[str, object]]:
    records = []
    for seed in seeds:
        metrics = output / f"seed_{seed}" / "metrics"
        for model in ("persistence", *LEARNED_MODELS):
            if (metrics / f"{model}_test.csv").is_file():
                records.append({"seed": seed, "model": model})
    return records


def run_seed(
    *, data: Path, output: Path, seed: int, epochs: int,
    patch_batch_size: int, memory_fraction: float,
) -> tuple[int, int]:
    log = output / "logs" / f"parallel_seed_{seed}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "scripts/run_multiseed_benchmark.py",
        "--data", str(data),
        "--output-dir", str(output),
        "--epochs", str(epochs),
        "--seeds", str(seed),
        "--max-new-models", "0",
        "--manifest-name", f"workers/seed_{seed}.json",
        "--skip-aggregate",
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = "src"
    environment["PYTORCH_ALLOC_CONF"] = "backend:cudaMallocAsync"
    environment["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync"
    environment["CARDIOGB_PATCH_BATCH_SIZE"] = str(patch_batch_size)
    environment["CARDIOGB_CUDA_MEMORY_FRACTION"] = str(memory_fraction)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[{datetime.now().isoformat()}] RUN {' '.join(command)}\n")
        handle.flush()
        result = subprocess.run(
            command, cwd=ROOT, env=environment, stdout=handle,
            stderr=subprocess.STDOUT, check=False,
        )
        handle.write(f"[{datetime.now().isoformat()}] EXIT {result.returncode}\n")
    return seed, result.returncode


def write_combined_manifest(
    output: Path, seeds: list[int], epochs: int, workers: int,
    patch_batch_size: int, memory_fraction: float,
) -> dict[str, object]:
    completed = completed_records(output, seeds)
    expected = len(seeds) * (1 + len(LEARNED_MODELS))
    payload = {
        "status": "complete" if len(completed) == expected else "partial",
        "execution": "isolated_parallel_seed_workers",
        "workers": workers,
        "epochs_requested": epochs,
        "batch_max_nodes": 8000,
        "patch_batch_size": patch_batch_size,
        "cuda_memory_fraction_per_worker": memory_fraction,
        "cooldown_seconds": 0,
        "seeds": seeds,
        "models": ["persistence", *LEARNED_MODELS],
        "split": "grouped biological-unit, stratified by stage",
        "completed": completed,
    }
    atomic_json(payload, output / "run_manifest.json")
    aggregate(output, seeds)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--patch-batch-size", type=int, default=8)
    parser.add_argument("--memory-fraction-per-worker", type=float, default=0.48)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pending = [seed for seed in args.seeds if not seed_complete(args.output_dir, seed)]
    write_combined_manifest(
        args.output_dir, args.seeds, args.epochs, args.workers,
        args.patch_batch_size, args.memory_fraction_per_worker,
    )
    failures: dict[int, int] = {}
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(
                run_seed, data=args.data, output=args.output_dir, seed=seed,
                epochs=args.epochs, patch_batch_size=args.patch_batch_size,
                memory_fraction=args.memory_fraction_per_worker,
            ): seed
            for seed in pending
        }
        for future in as_completed(futures):
            seed, return_code = future.result()
            if return_code != 0 or not seed_complete(args.output_dir, seed):
                failures[seed] = return_code
            write_combined_manifest(
                args.output_dir, args.seeds, args.epochs, args.workers,
                args.patch_batch_size, args.memory_fraction_per_worker,
            )
    # A worker can exceed its GPU partition when two CardioGB fits peak together.
    # Retry only failed seeds serially with the original conservative patch size.
    for seed in list(failures):
        _, return_code = run_seed(
            data=args.data,
            output=args.output_dir,
            seed=seed,
            epochs=args.epochs,
            patch_batch_size=max(4, args.patch_batch_size // 2),
            memory_fraction=0.90,
        )
        if return_code == 0 and seed_complete(args.output_dir, seed):
            failures.pop(seed, None)
    manifest = write_combined_manifest(
        args.output_dir, args.seeds, args.epochs, args.workers,
        args.patch_batch_size, args.memory_fraction_per_worker,
    )
    if failures or manifest["status"] != "complete":
        raise RuntimeError(f"parallel benchmark incomplete: failures={failures}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
