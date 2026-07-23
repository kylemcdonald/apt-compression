"""Benchmark candidate lossless encodings for four 12-bit APT fields."""

from __future__ import annotations

import argparse
import gc
import json
import struct
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from lossless12 import (
    BITS,
    QUANTIZER_METADATA_BYTES,
    brotli_compress,
    brotli_compress_large,
    encode_mass_delta_varint,
    encode_mass_elias_fano,
    encode_mass_rice,
    encode_mass_rice_blocks,
    encode_mass_rice_split,
    encode_mass_prefix_spatial,
    encode_spatial_mass,
    grouped_curve_sorted,
    mass_spatial_sorted,
    mass_prefix_spatial_sorted,
    morton4,
    pack_all_bitplanes,
    pack_columns12,
    pack_low_six,
    quantize12,
    quantizer_metadata,
    read_pos,
    spatial_mass_sorted,
    timed,
    tuple_keys,
    varint_encode,
    xz_compress,
    zstd_compress,
)

DEFAULT_ROOTS = (
    Path("/Users/kyle/Documents/GitHub/uap/rangefinder/controls"),
    Path("/Users/kyle/Documents/GitHub/uap/apt-analysis/data"),
)
PILOT_NAMES = {
    "extra/steelHD_5534859_70a59eff-003c-4337-832a-604c260dc623.POS",
    "synthetic/synthetic_al_mg_si.POS",
    "control_Ck10_steel_felfer_R56_01769.pos",
    "Sample 3 -POS file.POS",
}


def discover(roots: tuple[Path, ...]) -> list[tuple[str, Path]]:
    output = []
    for root in roots:
        label = root.parent.name + "/" + root.name
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() == ".pos":
                output.append((f"{label}/{path.relative_to(root)}", path))
    return output


def entropy_bits(values: np.ndarray) -> float:
    counts = np.bincount(values.astype(np.int64), minlength=4096)
    probabilities = counts[counts > 0].astype(np.float64) / len(values)
    return float(-(probabilities * np.log2(probabilities)).sum())


def common_header(count: int, metadata: bytes) -> bytes:
    assert len(metadata) == QUANTIZER_METADATA_BYTES
    return struct.pack("<4sQ", b"L12Q", count) + metadata


def summarize_size(size: int, point_count: int, raw_bytes: int) -> dict:
    baseline = point_count * 6 + 44
    return {
        "bytes": int(size),
        "bytes_per_point": round(size / point_count, 6),
        "ratio_vs_float32_pos": round(raw_bytes / size, 6),
        "fraction_of_12bit_baseline": round(size / baseline, 6),
    }


def add_representation(
    methods: dict,
    name: str,
    data: bytes,
    *,
    ordered: bool,
    point_count: int,
    raw_bytes: int,
    transform_seconds: float,
    zstd_levels: tuple[int, ...],
    use_xz: bool,
    use_brotli: bool,
    use_brotli_large: bool,
) -> None:
    record = {
        "ordered": ordered,
        "transform_seconds": round(transform_seconds, 6),
        "encoding": summarize_size(len(data), point_count, raw_bytes),
        "compressed": {},
    }
    for level in zstd_levels:
        compressed, seconds = timed(zstd_compress, data, level)
        record["compressed"][f"zstd-{level}"] = {
            **summarize_size(len(compressed), point_count, raw_bytes),
            "seconds": round(seconds, 6),
        }
        del compressed
    if use_xz:
        compressed, seconds = timed(xz_compress, data)
        record["compressed"]["xz-9e"] = {
            **summarize_size(len(compressed), point_count, raw_bytes),
            "seconds": round(seconds, 6),
        }
        del compressed
    if use_brotli:
        compressed, seconds = timed(brotli_compress, data)
        record["compressed"]["brotli-11"] = {
            **summarize_size(len(compressed), point_count, raw_bytes),
            "seconds": round(seconds, 6),
        }
        del compressed
    if use_brotli_large:
        compressed, seconds = timed(brotli_compress_large, data)
        record["compressed"]["brotli-11-large"] = {
            **summarize_size(len(compressed), point_count, raw_bytes),
            "seconds": round(seconds, 6),
        }
        del compressed
    methods[name] = record


def benchmark_file(
    label: str,
    path: Path,
    *,
    zstd_levels: tuple[int, ...],
    use_xz: bool,
    use_brotli: bool,
    use_brotli_large: bool,
    methods_filter: set[str] | None,
) -> dict:
    raw_bytes = path.stat().st_size
    started = time.perf_counter()
    quantized, quantize_seconds = timed(quantize12, read_pos(path))
    values = quantized.values
    count = len(values)
    header = common_header(count, quantizer_metadata(quantized))
    del quantized
    gc.collect()

    output = {
        "label": label,
        "path": str(path),
        "raw_bytes": raw_bytes,
        "point_count": count,
        "quantized_baseline_bytes": count * 6 + len(header),
        "quantize_seconds": round(quantize_seconds, 6),
        "marginal_entropy_bits_per_point": round(
            sum(entropy_bits(values[:, column]) for column in range(4)),
            6,
        ),
        "methods": {},
    }
    methods = output["methods"]

    def enabled(name: str) -> bool:
        return methods_filter is None or name in methods_filter

    tuple_key = None
    if any(enabled(name) for name in ("row12", "byte_shuffle", "columns12")):
        tuple_key = tuple_keys(values)

    if enabled("row12"):
        packed, seconds = timed(pack_low_six, tuple_key)
        add_representation(
            methods, "row12", header + packed, ordered=True,
            point_count=count, raw_bytes=raw_bytes, transform_seconds=seconds,
            zstd_levels=zstd_levels, use_xz=use_xz,
            use_brotli=use_brotli,
            use_brotli_large=use_brotli_large,
        )
        del packed

    if enabled("byte_shuffle"):
        started_transform = time.perf_counter()
        packed = np.frombuffer(pack_low_six(tuple_key), dtype=np.uint8)
        shuffled = packed.reshape(-1, 6).T.copy().tobytes()
        seconds = time.perf_counter() - started_transform
        add_representation(
            methods, "byte_shuffle", header + shuffled, ordered=True,
            point_count=count, raw_bytes=raw_bytes, transform_seconds=seconds,
            zstd_levels=zstd_levels, use_xz=use_xz,
            use_brotli=use_brotli,
            use_brotli_large=use_brotli_large,
        )
        del packed, shuffled

    if enabled("columns12"):
        packed, seconds = timed(pack_columns12, values)
        add_representation(
            methods, "columns12", header + packed, ordered=True,
            point_count=count, raw_bytes=raw_bytes, transform_seconds=seconds,
            zstd_levels=zstd_levels, use_xz=use_xz,
            use_brotli=use_brotli,
            use_brotli_large=use_brotli_large,
        )
        del packed

    if enabled("bitplanes"):
        packed, seconds = timed(pack_all_bitplanes, values)
        add_representation(
            methods, "bitplanes", header + packed, ordered=True,
            point_count=count, raw_bytes=raw_bytes, transform_seconds=seconds,
            zstd_levels=zstd_levels, use_xz=use_xz,
            use_brotli=use_brotli,
            use_brotli_large=use_brotli_large,
        )
        del packed

    del tuple_key
    gc.collect()

    if enabled("morton4_raw") or enabled("morton4_delta_varint"):
        started_transform = time.perf_counter()
        ordered4 = np.sort(morton4(values))
        sort_seconds = time.perf_counter() - started_transform
        if enabled("morton4_raw"):
            packed, pack_seconds = timed(pack_low_six, ordered4)
            add_representation(
                methods, "morton4_raw", header + packed, ordered=False,
                point_count=count, raw_bytes=raw_bytes,
                transform_seconds=sort_seconds + pack_seconds,
                zstd_levels=zstd_levels, use_xz=use_xz,
                use_brotli=use_brotli,
                use_brotli_large=use_brotli_large,
            )
            del packed
        if enabled("morton4_delta_varint"):
            started_transform = time.perf_counter()
            deltas = np.empty_like(ordered4)
            deltas[0] = ordered4[0]
            deltas[1:] = ordered4[1:] - ordered4[:-1]
            packed = varint_encode(deltas)
            seconds = time.perf_counter() - started_transform
            add_representation(
                methods, "morton4_delta_varint", header + packed, ordered=False,
                point_count=count, raw_bytes=raw_bytes,
                transform_seconds=sort_seconds + seconds,
                zstd_levels=zstd_levels, use_xz=use_xz,
                use_brotli=use_brotli,
                use_brotli_large=use_brotli_large,
            )
            del deltas, packed
        del ordered4
        gc.collect()

    mass_names = {
        "mass_spatial_raw",
        "mass_spatial_delta_varint",
        "mass_spatial_elias_fano",
        "mass_spatial_rice",
        "mass_spatial_rice_b256",
        "mass_spatial_rice_b1024",
        "mass_spatial_rice_b4096",
    }
    if any(enabled(name) for name in mass_names):
        (ordered_mass, mass_counts), sort_seconds = timed(
            mass_spatial_sorted,
            values,
        )
        if enabled("mass_spatial_raw"):
            packed, seconds = timed(pack_low_six, ordered_mass)
            add_representation(
                methods, "mass_spatial_raw", header + packed, ordered=False,
                point_count=count, raw_bytes=raw_bytes,
                transform_seconds=sort_seconds + seconds,
                zstd_levels=zstd_levels, use_xz=use_xz,
                use_brotli=use_brotli,
                use_brotli_large=use_brotli_large,
            )
            del packed
        for name, function in (
            ("mass_spatial_delta_varint", encode_mass_delta_varint),
            ("mass_spatial_elias_fano", encode_mass_elias_fano),
            ("mass_spatial_rice", encode_mass_rice),
            (
                "mass_spatial_rice_b256",
                lambda ordered, counts: encode_mass_rice_blocks(
                    ordered,
                    counts,
                    block_size=256,
                ),
            ),
            (
                "mass_spatial_rice_b1024",
                lambda ordered, counts: encode_mass_rice_blocks(
                    ordered,
                    counts,
                    block_size=1024,
                ),
            ),
            (
                "mass_spatial_rice_b4096",
                lambda ordered, counts: encode_mass_rice_blocks(
                    ordered,
                    counts,
                    block_size=4096,
                ),
            ),
        ):
            if not enabled(name):
                continue
            packed, seconds = timed(function, ordered_mass, mass_counts)
            add_representation(
                methods, name, header + packed, ordered=False,
                point_count=count, raw_bytes=raw_bytes,
                transform_seconds=sort_seconds + seconds,
                zstd_levels=zstd_levels, use_xz=use_xz,
                use_brotli=use_brotli,
                use_brotli_large=use_brotli_large,
            )
            del packed
            gc.collect()
        del ordered_mass, mass_counts

    grouped_methods = (
        ("x_morton_rice", 0, "morton"),
        ("y_morton_rice", 1, "morton"),
        ("z_morton_rice", 2, "morton"),
        ("x_hilbert_rice", 0, "hilbert"),
        ("y_hilbert_rice", 1, "hilbert"),
        ("z_hilbert_rice", 2, "hilbert"),
        ("mass_hilbert_rice", 3, "hilbert"),
        ("mass_hilbert_rice_split", 3, "hilbert"),
    )
    for name, primary, curve in grouped_methods:
        if not enabled(name):
            continue
        (ordered_group, group_counts), sort_seconds = timed(
            grouped_curve_sorted,
            values,
            primary,
            curve,
        )
        encoder = (
            encode_mass_rice_split
            if name.endswith("_split")
            else encode_mass_rice
        )
        packed, seconds = timed(
            encoder,
            ordered_group,
            group_counts,
        )
        add_representation(
            methods, name, header + packed, ordered=False,
            point_count=count, raw_bytes=raw_bytes,
            transform_seconds=sort_seconds + seconds,
            zstd_levels=zstd_levels, use_xz=use_xz,
            use_brotli=use_brotli,
            use_brotli_large=use_brotli_large,
        )
        del ordered_group, group_counts, packed
        gc.collect()

    for name, curve in (
        ("spatial_morton_plus_mass", "morton"),
        ("spatial_hilbert_plus_mass", "hilbert"),
    ):
        if not enabled(name):
            continue
        ordered_spatial, sort_seconds = timed(
            spatial_mass_sorted,
            values,
            curve,
        )
        packed, seconds = timed(encode_spatial_mass, ordered_spatial)
        add_representation(
            methods, name, header + packed, ordered=False,
            point_count=count, raw_bytes=raw_bytes,
            transform_seconds=sort_seconds + seconds,
            zstd_levels=zstd_levels, use_xz=use_xz,
            use_brotli=use_brotli,
            use_brotli_large=use_brotli_large,
        )
        del ordered_spatial, packed
        gc.collect()

    for curve in ("morton", "hilbert"):
        for prefix_bits in range(BITS + 1):
            name = f"mass_prefix_{prefix_bits}_{curve}"
            if not enabled(name):
                continue
            ordered_prefix, sort_seconds = timed(
                mass_prefix_spatial_sorted,
                values,
                curve,
                prefix_bits,
            )
            packed, seconds = timed(
                encode_mass_prefix_spatial,
                ordered_prefix,
                prefix_bits,
            )
            add_representation(
                methods, name, header + packed, ordered=False,
                point_count=count, raw_bytes=raw_bytes,
                transform_seconds=sort_seconds + seconds,
                zstd_levels=zstd_levels, use_xz=use_xz,
                use_brotli=use_brotli,
                use_brotli_large=use_brotli_large,
            )
            del ordered_prefix, packed
            gc.collect()

    for axis_name, axis_order in (
        ("xzy", (0, 2, 1)),
        ("yxz", (1, 0, 2)),
        ("yzx", (1, 2, 0)),
        ("zxy", (2, 0, 1)),
        ("zyx", (2, 1, 0)),
    ):
        for prefix_bits in range(BITS + 1):
            name = f"mass_prefix_{prefix_bits}_hilbert_{axis_name}"
            if not enabled(name):
                continue
            ordered_prefix, sort_seconds = timed(
                mass_prefix_spatial_sorted,
                values,
                "hilbert",
                prefix_bits,
                axis_order,
            )
            packed, seconds = timed(
                encode_mass_prefix_spatial,
                ordered_prefix,
                prefix_bits,
            )
            add_representation(
                methods, name, header + packed, ordered=False,
                point_count=count, raw_bytes=raw_bytes,
                transform_seconds=sort_seconds + seconds,
                zstd_levels=zstd_levels, use_xz=use_xz,
                use_brotli=use_brotli,
                use_brotli_large=use_brotli_large,
            )
            del ordered_prefix, packed
            gc.collect()

    del values
    gc.collect()
    output["total_seconds"] = round(time.perf_counter() - started, 6)
    return output


def save(output_path: Path, result: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".part")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(output_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--root", action="append", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", action="append")
    parser.add_argument("--zstd-level", action="append", type=int)
    parser.add_argument("--xz", action="store_true")
    parser.add_argument("--brotli", action="store_true")
    parser.add_argument("--brotli-large", action="store_true")
    parser.add_argument("--limit-files", type=int)
    args = parser.parse_args()

    roots = tuple(args.root) if args.root else DEFAULT_ROOTS
    corpus = discover(roots)
    if args.preset == "pilot":
        corpus = [
            item for item in corpus
            if item[1].name in PILOT_NAMES or str(item[1].relative_to(
                next(root for root in roots if item[1].is_relative_to(root))
            )) in PILOT_NAMES
        ]
    if args.limit_files:
        corpus = corpus[:args.limit_files]
    zstd_levels = tuple(args.zstd_level or (
        (3, 19) if args.preset == "pilot" else (19,)
    ))
    methods_filter = set(args.method) if args.method else None

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "preset": args.preset,
        "roots": [str(root) for root in roots],
        "zstd_levels": list(zstd_levels),
        "xz": args.xz,
        "brotli": args.brotli,
        "brotli_large": args.brotli_large,
        "files": [],
    }
    save(args.output, result)
    for index, (label, path) in enumerate(corpus, start=1):
        print(f"[{index}/{len(corpus)}] {label}", flush=True)
        file_result = benchmark_file(
            label,
            path,
            zstd_levels=zstd_levels,
            use_xz=args.xz,
            use_brotli=args.brotli,
            use_brotli_large=args.brotli_large,
            methods_filter=methods_filter,
        )
        result["files"].append(file_result)
        save(args.output, result)
        best = min(
            (
                (variant["bytes"], f"{method}/{codec}")
                for method, details in file_result["methods"].items()
                for codec, variant in details["compressed"].items()
            ),
            default=(0, "none"),
        )
        print(
            f"  {file_result['point_count']:,} points; "
            f"best {best[1]} = {best[0] / file_result['point_count']:.3f} B/point; "
            f"{file_result['total_seconds']:.1f}s",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
