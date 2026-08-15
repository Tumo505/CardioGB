from __future__ import annotations

import argparse
import json

import numpy as np

from cardiogb.metrics import distribution_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predicted", required=True, help="NPY matrix [observations, states]")
    parser.add_argument("--observed", required=True, help="NPY matrix [observations, states]")
    args = parser.parse_args()
    print(json.dumps(distribution_metrics(np.load(args.predicted), np.load(args.observed)), indent=2))
