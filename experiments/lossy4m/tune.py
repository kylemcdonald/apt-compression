"""Sweep mass-bin width and allocation exponent on representative files."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from cpos.io import read_pos

from .codec import decode_retained, encode
from .metrics import (
    expansion_weights,
    make_spatial_reference,
    mass_counts,
    mass_quality,
    retention_metrics,
    spatial_quality,
)

DEFAULT_FILES = (
    Path(
        "/Users/kyle/Documents/GitHub/uap/rangefinder/controls/"
        "control_Ck10_steel_felfer_R56_01769.pos"
    ),
    Path(
        "/Users/kyle/Documents/GitHub/uap/apt-analysis/data/"
        "2166bb75-7ff6-4c85-bf2d-564431f0b089.POS"
    ),
    Path(
        "/Users/kyle/Documents/GitHub/uap/apt-analysis/data/"
        "Sample 3 -POS file.POS"
    ),
)


def save(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", action="append", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-points", type=int, default=4_000_000)
    args = parser.parse_args()
    paths = tuple(args.file) if args.file else DEFAULT_FILES
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_points": args.target_points,
        "files": [],
    }
    save(args.output, result)
    for file_index, path in enumerate(paths, start=1):
        print(f"[{file_index}/{len(paths)}] {path.name}", flush=True)
        points = read_pos(path)
        reference = make_spatial_reference(points)
        file_record = {
            "path": str(path),
            "point_count": len(points),
            "candidates": [],
        }
        for bin_width in (0.01, 0.05, 0.1):
            source_mass = mass_counts(points, bin_width)
            for exponent in (0.25, 0.5, 0.75, 1.0):
                started = time.perf_counter()
                payload = encode(
                    points,
                    target_points=args.target_points,
                    bin_width_da=bin_width,
                    allocation_exponent=exponent,
                )
                retained = decode_retained(payload)
                weights = expansion_weights(
                    retained.bins,
                    retained.true_counts,
                    retained.stored_counts,
                )
                quality = {
                    **spatial_quality(
                        reference,
                        retained.points,
                        weights=weights,
                    ),
                    **mass_quality(
                        source_mass,
                        retained.points,
                        bin_width_da=bin_width,
                        weights=weights,
                    ),
                    **retention_metrics(
                        retained.true_counts,
                        retained.stored_counts,
                    ),
                }
                candidate = {
                    "bin_width_da": bin_width,
                    "allocation_exponent": exponent,
                    "file_bytes": len(payload),
                    "seconds": round(time.perf_counter() - started, 6),
                    "quality": quality,
                }
                file_record["candidates"].append(candidate)
                save(args.output, {
                    **result,
                    "files": [*result["files"], file_record],
                })
                print(
                    f"  {bin_width:.2f} Da α={exponent:.2f}: "
                    f"{len(payload) / 2**20:.2f} MiB, "
                    f"JS={quality['spatial_js_bits']:.6f}, "
                    f"rare={quality['rare_point_retention']:.3f}, "
                    f"major={quality['major_point_retention']:.3f}",
                    flush=True,
                )
        result["files"].append(file_record)
        save(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
