from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cardiogb.data.graphs import build_spatial_knn_graph
from cardiogb.data.state_dataset import StateDataset
from cardiogb.interpretation.attribution import integrated_gradients
from cardiogb.models.factory import build_model
from cardiogb.models.message_passing import TorchGraph
from cardiogb.utils.config import load_yaml
from cardiogb.utils.device import resolve_device
from cardiogb.utils.io import export_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Residual attribution on representative sections")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--k", type=int, default=8)
    args = parser.parse_args()
    dataset = StateDataset.load(args.data)
    model = build_model(
        "cardiogb", load_yaml("configs/model.yaml"), load_yaml("configs/mechanistic_model.yaml")
    )
    device = resolve_device("auto").selected
    model.load_state_dict(
        torch.load(args.checkpoint, map_location=device, weights_only=False)["model"]
    )
    model = model.to(device).eval()
    rows = []
    for time in sorted(np.unique(dataset.times)):
        candidate_sections = sorted(np.unique(dataset.sections[dataset.times == time]).astype(str))
        section = candidate_sections[0]
        index = np.flatnonzero(dataset.sections.astype(str) == section)
        spatial = build_spatial_knn_graph(
            dataset.coordinates[index], np.repeat(section, len(index)), k=args.k
        )
        edge_index, edge_attr = spatial.to_torch(device)
        graph = TorchGraph(edge_index, edge_attr)
        states = torch.as_tensor(dataset.states[index], dtype=torch.float32, device=device)
        for target, target_name in enumerate(dataset.state_names):
            attribution = integrated_gradients(
                model.residual_model,
                float(time),
                states,
                graph,
                target_state=target,
                steps=args.steps,
            )
            values = attribution.abs().mean(dim=0).cpu().numpy()
            for source, value in zip(dataset.state_names, values):
                rows.append(
                    {
                        "stage_days": float(time),
                        "section": section,
                        "target_state": target_name,
                        "input_state": source,
                        "mean_absolute_integrated_gradient": float(value),
                        "claim_scope": "model-derived association; hypothesis generating",
                    }
                )
    export_table(pd.DataFrame(rows), args.output)
    print(f"exported {len(rows)} attribution summaries")


if __name__ == "__main__":
    main()
