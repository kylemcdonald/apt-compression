"""Benchmark experimental CP4M against the released CPOS 1.0 codec."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from cpos import decode as decode_cpos
from cpos import encode as encode_cpos
from cpos.codec import spectrum_counts as cpos_spectrum_counts
from cpos.io import read_pos

from .codec import (
    DEFAULT_ALLOCATION_EXPONENT,
    DEFAULT_BIN_WIDTH_DA,
    DEFAULT_TARGET_POINTS,
    decode,
    encode,
)
from .metrics import (
    expansion_weights,
    make_spatial_reference,
    mass_counts,
    mass_quality,
    retention_metrics,
    spatial_quality,
)

DEFAULT_ROOTS = (
    Path("/Users/kyle/Documents/GitHub/uap/rangefinder/controls"),
    Path("/Users/kyle/Documents/GitHub/uap/apt-analysis/data"),
)


def discover(roots: tuple[Path, ...]) -> list[tuple[str, Path]]:
    output = []
    for root in roots:
        label = root.parent.name + "/" + root.name
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() == ".pos":
                output.append((f"{label}/{path.relative_to(root)}", path))
    return output


def _save(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(path)


def _timed(function, *args, **kwargs):
    started = time.perf_counter()
    value = function(*args, **kwargs)
    return value, time.perf_counter() - started


def _cpos_bin_ids(stored_counts: np.ndarray) -> np.ndarray:
    return np.repeat(
        np.arange(len(stored_counts), dtype=np.int32),
        stored_counts.astype(np.int64),
    )


def benchmark_file(
    label: str,
    path: Path,
    *,
    target_points: int,
    bin_width_da: float,
    allocation_exponent: float,
) -> dict:
    points, read_seconds = _timed(read_pos, path)
    reference = make_spatial_reference(points)
    source_mass_counts = mass_counts(points, bin_width_da)

    cpos_payload, cpos_encode_seconds = _timed(
        encode_cpos,
        points,
        499_000,
    )
    cpos_points, cpos_decode_seconds = _timed(decode_cpos, cpos_payload)
    cpos_true, cpos_stored = cpos_spectrum_counts(cpos_payload)
    cpos_bins = _cpos_bin_ids(cpos_stored)
    cpos_weights = expansion_weights(cpos_bins, cpos_true, cpos_stored)
    cpos_quality = {
        **spatial_quality(reference, cpos_points, weights=cpos_weights),
        **mass_quality(
            source_mass_counts,
            cpos_points,
            bin_width_da=bin_width_da,
            weights=cpos_weights,
        ),
        **retention_metrics(cpos_true, cpos_stored),
    }

    payload, encode_seconds = _timed(
        encode,
        points,
        target_points=target_points,
        bin_width_da=bin_width_da,
        allocation_exponent=allocation_exponent,
    )
    point_count = len(points)
    del points
    expanded, decode_seconds = _timed(decode, payload)
    if len(expanded.points) != point_count:
        raise AssertionError("CP4M did not restore the original point count")
    if int(expanded.exact.sum()) != expanded.header.stored_point_count:
        raise AssertionError("CP4M exact provenance count is invalid")
    if not np.array_equal(
        np.bincount(
            expanded.bins,
            minlength=expanded.header.spectrum_bin_count,
        ),
        expanded.true_counts,
    ):
        raise AssertionError("CP4M expansion did not restore the source histogram")
    if not np.isfinite(expanded.points).all():
        raise AssertionError("CP4M expansion produced non-finite points")
    quality = {
        **spatial_quality(reference, expanded.points),
        **mass_quality(
            source_mass_counts,
            expanded.points,
            bin_width_da=bin_width_da,
        ),
        **retention_metrics(
            expanded.true_counts,
            expanded.stored_counts,
        ),
    }
    raw_bytes = path.stat().st_size
    return {
        "label": label,
        "path": str(path),
        "point_count": point_count,
        "raw_bytes": raw_bytes,
        "read_seconds": round(read_seconds, 6),
        "cpos_1_0": {
            "file_bytes": len(cpos_payload),
            "stored_points": len(cpos_points),
            "expanded_points": len(cpos_points),
            "exact_fraction": round(len(cpos_points) / point_count, 9),
            "ratio_vs_pos": round(raw_bytes / len(cpos_payload), 6),
            "encode_seconds": round(cpos_encode_seconds, 6),
            "decode_seconds": round(cpos_decode_seconds, 6),
            "quality": cpos_quality,
        },
        "cp4m": {
            "file_bytes": len(payload),
            "stored_points": expanded.header.stored_point_count,
            "expanded_points": len(expanded.points),
            "exact_fraction": round(
                expanded.header.stored_point_count
                / expanded.header.original_point_count,
                9,
            ),
            "ratio_vs_pos": round(raw_bytes / len(payload), 6),
            "encode_seconds": round(encode_seconds, 6),
            "decode_expand_seconds": round(decode_seconds, 6),
            "quality": quality,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", action="append", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-points", type=int, default=DEFAULT_TARGET_POINTS)
    parser.add_argument("--bin-width-da", type=float, default=DEFAULT_BIN_WIDTH_DA)
    parser.add_argument(
        "--allocation-exponent",
        type=float,
        default=DEFAULT_ALLOCATION_EXPONENT,
    )
    parser.add_argument("--limit-files", type=int)
    args = parser.parse_args()

    roots = tuple(args.root) if args.root else DEFAULT_ROOTS
    corpus = discover(roots)
    if args.limit_files:
        corpus = corpus[:args.limit_files]
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "roots": [str(root) for root in roots],
        "target_points": args.target_points,
        "bin_width_da": args.bin_width_da,
        "allocation_exponent": args.allocation_exponent,
        "files": [],
    }
    _save(args.output, result)
    for index, (label, path) in enumerate(corpus, start=1):
        print(f"[{index}/{len(corpus)}] {label}", flush=True)
        record = benchmark_file(
            label,
            path,
            target_points=args.target_points,
            bin_width_da=args.bin_width_da,
            allocation_exponent=args.allocation_exponent,
        )
        result["files"].append(record)
        _save(args.output, result)
        print(
            f"  CPOS {record['cpos_1_0']['file_bytes'] / 2**20:.2f} MiB; "
            f"CP4M {record['cp4m']['file_bytes'] / 2**20:.2f} MiB; "
            f"spatial JS {record['cpos_1_0']['quality']['spatial_js_bits']:.5f}"
            f" → {record['cp4m']['quality']['spatial_js_bits']:.5f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
