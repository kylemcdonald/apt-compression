"""Read and write four-column Atom Probe ``.POS`` files."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_pos(path: str | Path) -> np.ndarray:
    source = Path(path)
    size = source.stat().st_size
    if size == 0 or size % 16:
        raise ValueError(f"{source} is not a non-empty four-column float32 POS file")
    mapped = np.memmap(source, dtype=">f4", mode="r", shape=(size // 16, 4))
    points = np.asarray(mapped, dtype=np.float32)
    if not np.isfinite(points).all():
        raise ValueError(f"{source} contains non-finite values")
    return points


def write_pos(path: str | Path, points: np.ndarray) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(points, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 4:
        raise ValueError("points must have shape (N, 4)")
    array.astype(">f4", copy=False).tofile(destination)
