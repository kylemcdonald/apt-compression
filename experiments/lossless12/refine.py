"""Refine full-corpus winners without rerunning every rejected candidate."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from benchmark import benchmark_file, save

AXES = ("xzy", "yxz", "yzx", "zxy", "zyx")
PREFIX_PATTERN = re.compile(r"^mass_prefix_(\d+)_hilbert(?:_[a-z]+)?$")


def best_method(record: dict, codec: str, *, ordered: bool = False) -> str:
    return min(
        (
            details["compressed"][codec]["bytes"],
            method,
        )
        for method, details in record["methods"].items()
        if details["ordered"] is ordered and codec in details["compressed"]
    )[1]


def axis_candidates(method: str) -> set[str]:
    match = PREFIX_PATTERN.fullmatch(method)
    if match is None:
        raise ValueError(f"unexpected sweep winner: {method}")
    bits = int(match.group(1))
    base = f"mass_prefix_{bits}_hilbert"
    return {base, *(f"{base}_{axes}" for axes in AXES)}


def run_axes(source: dict, source_path: Path, output_path: Path) -> dict:
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stage": "axes",
        "source": str(source_path),
        "files": [],
    }
    save(output_path, result)
    for index, source_file in enumerate(source["files"], start=1):
        winner = best_method(source_file, "zstd-19")
        methods = axis_candidates(winner)
        print(
            f"[{index}/{len(source['files'])}] {source_file['label']} "
            f"({winner})",
            flush=True,
        )
        record = benchmark_file(
            source_file["label"],
            Path(source_file["path"]),
            zstd_levels=(19,),
            use_xz=False,
            use_brotli=False,
            use_brotli_large=False,
            methods_filter=methods,
        )
        result["files"].append(record)
        save(output_path, result)
        refined = best_method(record, "zstd-19")
        size = record["methods"][refined]["compressed"]["zstd-19"]["bytes"]
        print(
            f"  {refined}: {size / record['point_count']:.3f} B/point",
            flush=True,
        )
    return result


def run_backends(source: dict, source_path: Path, output_path: Path) -> dict:
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stage": "backends",
        "source": str(source_path),
        "files": [],
    }
    save(output_path, result)
    for index, source_file in enumerate(source["files"], start=1):
        winner = best_method(source_file, "zstd-19")
        print(
            f"[{index}/{len(source['files'])}] {source_file['label']} "
            f"({winner})",
            flush=True,
        )
        record = benchmark_file(
            source_file["label"],
            Path(source_file["path"]),
            zstd_levels=(19,),
            use_xz=True,
            use_brotli=True,
            use_brotli_large=False,
            methods_filter={winner},
        )
        result["files"].append(record)
        save(output_path, result)
        variants = record["methods"][winner]["compressed"]
        codec, details = min(
            variants.items(),
            key=lambda item: item[1]["bytes"],
        )
        print(
            f"  {codec}: {details['bytes_per_point']:.3f} B/point",
            flush=True,
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("axes", "backends"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.input.read_text())
    if args.stage == "axes":
        run_axes(source, args.input, args.output)
    else:
        run_backends(source, args.input, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
