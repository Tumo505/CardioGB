"""Minimal reproducible file/console logging setup."""

from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(path: str | Path | None = None, *, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("cardiogb")
    logger.setLevel(level)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    if path is not None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(target, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger
