from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from cardiogb.utils.io import atomic_json


def count(pattern: str, text: str) -> int:
    matches = re.findall(pattern, text)
    return int(matches[-1]) if matches else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the final CPU-isolated CardioGB verification suite")
    parser.add_argument("--output-dir", type=Path, default=Path("results/verification"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "-1"
    environment["PYTHONPATH"] = "src"
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "--basetemp",
        str(args.output_dir / "pytest_tmp"),
    ]
    started = time.time()
    result = subprocess.run(command, env=environment, text=True, capture_output=True)
    elapsed = time.time() - started
    output = (result.stdout or "") + (result.stderr or "")
    print(output, end="")
    (args.output_dir / "pytest_output.txt").write_text(output, encoding="utf-8")
    passed = count(r"(\d+) passed", output)
    failed = count(r"(\d+) failed", output)
    errors = count(r"(\d+) error(?:s)?", output)
    manifest = {
        "status": "complete" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "elapsed_seconds": elapsed,
        "gpu_visible_to_tests": False,
        "command": command,
    }
    atomic_json(manifest, args.output_dir / "test_manifest.json")
    print(json.dumps(manifest, indent=2))
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
