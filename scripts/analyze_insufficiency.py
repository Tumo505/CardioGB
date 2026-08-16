from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cardiogb.data.graphs import build_spatial_knn_graph
from cardiogb.data.state_dataset import StateDataset
from cardiogb.interpretation.mechanistic_insufficiency import mechanistic_insufficiency
from cardiogb.models.factory import build_model
from cardiogb.models.message_passing import TorchGraph
from cardiogb.utils.config import load_yaml
from cardiogb.utils.device import resolve_device
from cardiogb.utils.io import export_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Export observed-state CardioGB MI scores")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/interpretation"))
    parser.add_argument("--k", type=int, default=8)
    args = parser.parse_args()
    dataset = StateDataset.load(args.data)
    if dataset.domains is None:
        raise ValueError("state dataset must retain domain annotations")
    model = build_model(
        "cardiogb", load_yaml("configs/model.yaml"), load_yaml("configs/mechanistic_model.yaml")
    )
    device = resolve_device("auto").selected
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device).eval()
    scores = np.empty(len(dataset.states), dtype=np.float32)
    with torch.no_grad():
        for section in np.unique(dataset.sections):
            index = np.flatnonzero(dataset.sections == section)
            section_times = np.unique(dataset.times[index])
            if len(section_times) != 1:
                raise ValueError(f"section {section} spans multiple stages")
            spatial = build_spatial_knn_graph(
                dataset.coordinates[index], np.repeat(str(section), len(index)), k=args.k
            )
            edge_index, edge_attr = spatial.to_torch(device)
            graph = TorchGraph(edge_index, edge_attr)
            states = torch.as_tensor(dataset.states[index], dtype=torch.float32, device=device)
            terms = model.vector_field(float(section_times[0]), states, graph)
            scores[index] = mechanistic_insufficiency(
                terms["mechanistic"], terms["residual"]
            ).cpu().numpy()
    frame = pd.DataFrame(
        {
            "spot_index": np.arange(len(scores)),
            "mi": scores,
            "stage_days": dataset.times,
            "domain": dataset.domains,
            "section": dataset.sections,
            "biological_unit": dataset.groups,
        }
    )
    export_table(frame, args.output_dir / "mi_spots.csv")
    for column in ("stage_days", "domain", "section", "biological_unit"):
        summary = frame.groupby(column, observed=True)["mi"].agg(
            ["count", "mean", "std", "median"]
        ).reset_index()
        export_table(summary, args.output_dir / f"mi_by_{column}.csv")
    print(frame["mi"].describe().to_string())


if __name__ == "__main__":
    main()
