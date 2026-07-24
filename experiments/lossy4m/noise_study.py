"""Compare sub-cell dither modes on representative fully expanded clouds."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from cpos.io import read_pos

from .codec import decode, encode
from .metrics import make_spatial_reference, spatial_quality

DEFAULT_FILES = (
    Path(
        "/Users/kyle/Documents/GitHub/uap/rangefinder/controls/"
        "control_Ck10_steel_felfer_R56_01769.pos"
    ),
    Path(
        "/Users/kyle/Documents/GitHub/uap/apt-analysis/data/"
        "Sample 3 -POS file.POS"
    ),
)


def _save(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", action="append", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grid-size", type=int, default=64)
    args = parser.parse_args()
    paths = tuple(args.file) if args.file else DEFAULT_FILES
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "grid_size": args.grid_size,
        "files": [],
    }
    _save(args.output, result)
    for file_index, path in enumerate(paths, start=1):
        print(f"[{file_index}/{len(paths)}] {path.name}", flush=True)
        points = read_pos(path)
        point_count = len(points)
        reference = make_spatial_reference(
            points,
            grid_size=args.grid_size,
        )
        payload = encode(points)
        del points
        file_record = {
            "path": str(path),
            "point_count": point_count,
            "file_bytes": len(payload),
            "modes": [],
        }
        for mode in ("none", "uniform", "gaussian"):
            started = time.perf_counter()
            expanded = decode(payload, noise=mode)
            quality = spatial_quality(reference, expanded.points)
            seconds = time.perf_counter() - started
            if len(expanded.points) != point_count:
                raise AssertionError("noise study expansion count mismatch")
            file_record["modes"].append({
                "noise": mode,
                "decode_and_measure_seconds": round(seconds, 6),
                "quality": quality,
            })
            print(
                f"  {mode:8s}: JS={quality['spatial_js_bits']:.8f}, "
                f"occupancy={quality['occupied_voxel_recall']:.6f}, "
                f"axis EMD={quality['mean_axis_emd']:.8f}",
                flush=True,
            )
        result["files"].append(file_record)
        _save(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
