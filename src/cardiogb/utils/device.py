"""Torch device discovery and selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class DeviceReport:
    requested: str
    selected: str
    torch_version: str | None
    cuda_available: bool
    cuda_version: str | None
    device_name: str | None
    total_memory_bytes: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_device(requested: str = "auto") -> DeviceReport:
    """Resolve ``auto``, ``cuda``, or ``cpu`` and return diagnostic details."""
    if requested not in {"auto", "cuda", "cpu"}:
        raise ValueError("requested device must be one of: auto, cuda, cpu")
    try:
        import torch
    except ImportError:
        if requested == "cuda":
            raise RuntimeError("CUDA was requested but PyTorch is not installed") from None
        return DeviceReport(requested, "cpu", None, False, None, None, None)

    available = torch.cuda.is_available()
    if requested == "cuda" and not available:
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    selected = "cuda" if available and requested in {"auto", "cuda"} else "cpu"
    if selected == "cuda":
        props = torch.cuda.get_device_properties(0)
        name = torch.cuda.get_device_name(0)
        memory = int(props.total_memory)
    else:
        name = None
        memory = None
    return DeviceReport(
        requested=requested,
        selected=selected,
        torch_version=torch.__version__,
        cuda_available=available,
        cuda_version=torch.version.cuda,
        device_name=name,
        total_memory_bytes=memory,
    )

