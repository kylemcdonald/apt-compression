from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import os
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np

from hypernetwork_common import load_reference, train_hypernetwork
import preprocess as pp


MAGIC = b"CDFHYP1\0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CDF-range mass-conditioned density-grid hypernetwork.")
    parser.add_argument("--dataset-filter", default="499e563f-0c0c-4c6f-bc08-b8e76f59c31b")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--ranges", type=int, default=32)
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--grid-shape", default="64,64,256", help="Density grid shape as x,y,z. Use 'cubic' for --grid^3.")
    parser.add_argument("--hist-bins", type=int, default=16384)
    parser.add_argument("--hyper-rank", type=int, default=0, help="Low-rank basis size. 0 chooses a rank near --target-artifact-mb.")
    parser.add_argument("--hyper-epochs", type=int, default=8000)
    parser.add_argument("--basis-dtype", choices=["int8", "float16"], default="int8")
    parser.add_argument("--target-artifact-mb", type=float, default=10.0)
    parser.add_argument("--teacher-workers", type=int, default=min(4, max(1, os.cpu_count() or 1)))
    parser.add_argument("--metric-points", type=int, default=250000)
    parser.add_argument("--spectrum-bins", type=int, default=1024)
    parser.add_argument("--profile-bins", type=int, default=64)
    parser.add_argument("--spatial-bins", type=int, default=6)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--chunk-atoms", type=int, default=2_000_000)
    return parser.parse_args()


def parse_grid_shape(value: str, base_grid: int) -> tuple[int, int, int]:
    text = (value or "").strip().lower()
    if text in ("", "cubic"):
        return int(base_grid), int(base_grid), int(base_grid)
    parts = [part.strip() for part in text.replace("x", ",").split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError("--grid-shape must be 'cubic' or three integers like 64,64,256")
    gx, gy, gz = (int(part) for part in parts)
    if gx <= 0 or gy <= 0 or gz <= 0:
        raise ValueError("--grid-shape dimensions must be positive")
    return gx, gy, gz


def grid_label(grid_shape: tuple[int, int, int]) -> str:
    gx, gy, gz = grid_shape
    return f"{gx}x{gy}x{gz}"


def estimate_artifact_bytes(param_count: int, range_count: int, rank: int, basis_dtype: str) -> int:
    mean_bytes = param_count * 2
    basis_bytes = param_count * rank * (1 if basis_dtype == "int8" else 2)
    support_bytes = range_count * int(math.ceil(param_count / 8))
    basis_scale_bytes = rank * 4 if basis_dtype == "int8" else 0
    coeff_stats_bytes = rank * 8
    coeff_net_values = (13 * 96 + 96) + (96 * 96 + 96) + (rank * 96 + rank)
    coeff_net_bytes = coeff_net_values * 4
    header_slack = 24_000
    return mean_bytes + basis_bytes + support_bytes + basis_scale_bytes + coeff_stats_bytes + coeff_net_bytes + header_slack


def choose_hyper_rank(args: argparse.Namespace, param_count: int, range_count: int) -> int:
    max_rank = max(1, min(range_count - 1, range_count))
    if args.hyper_rank > 0:
        return min(args.hyper_rank, max_rank)
    target = int(float(args.target_artifact_mb) * 1024 * 1024)
    candidates = [
        (rank, estimate_artifact_bytes(param_count, range_count, rank, args.basis_dtype))
        for rank in range(1, max_rank + 1)
    ]
    # Prefer the most expressive rank within 10% of the target; otherwise use the closest size.
    in_band = [item for item in candidates if item[1] <= target * 1.10]
    if in_band:
        return max(in_band, key=lambda item: item[0])[0]
    return min(candidates, key=lambda item: abs(item[1] - target))[0]


def load_mass_values(path: Path, chunk_atoms: int) -> np.ndarray:
    mm, count, _ = pp.open_pos(path)
    parts: list[np.ndarray] = []
    for _, _, chunk in pp.iter_chunks(mm, chunk_atoms):
        finite = np.isfinite(chunk).all(axis=1)
        if finite.any():
            parts.append(chunk[finite, 3].astype(np.float32, copy=True))
    if not parts:
        return np.empty(0, dtype=np.float32)
    return np.concatenate(parts)


def cdf_ranges(path: Path, ref: pp.ReferenceStats, args: argparse.Namespace) -> list[dict[str, Any]]:
    masses = load_mass_values(path, args.chunk_atoms)
    if len(masses) == 0:
        raise ValueError(f"No finite mass values found in {path}")
    quantiles = np.linspace(0.0, 1.0, args.ranges + 1, dtype=np.float64)
    edges = np.quantile(masses, quantiles).astype(np.float64)
    edges[0] = float(ref.mass_range[0])
    edges[-1] = float(ref.mass_range[1])
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = np.nextafter(edges[i - 1], np.inf)

    ranges: list[dict[str, Any]] = []
    for i in range(args.ranges):
        lo = float(edges[i])
        hi = float(edges[i + 1])
        if i == args.ranges - 1:
            count = int(((masses >= lo) & (masses <= hi)).sum())
        else:
            count = int(((masses >= lo) & (masses < hi)).sum())
        ranges.append(
            {
                "id": f"range_{i + 1:02d}",
                "mass": float((lo + hi) * 0.5),
                "mass_min": lo,
                "mass_max": hi,
                "atom_count": count,
                "quantile_min": float(quantiles[i]),
                "quantile_max": float(quantiles[i + 1]),
            }
        )
    return ranges


def density_grid_teacher(
    path: Path,
    ref: pp.ReferenceStats,
    mass_min: float,
    mass_max: float,
    grid_shape: tuple[int, int, int],
    chunk_atoms: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    counts = pp.build_mass_range_density_grid(path, ref, grid_shape, mass_min, mass_max, chunk_atoms)
    target_scale = max(float(np.log1p(counts.max())), 1.0)
    teacher = (np.log1p(counts) / target_scale).astype(np.float32)
    support = counts > 0
    return teacher.reshape(-1), support.reshape(-1), target_scale


def generated_grid_from_weights(
    vector: np.ndarray,
    target_scale: float,
    support: np.ndarray,
    grid_shape: tuple[int, int, int],
) -> np.ndarray:
    density = np.expm1(np.clip(vector.astype(np.float32), 0.0, 8.0) * float(target_scale)).astype(np.float32)
    density[~support] = 0.0
    gx, gy, gz = grid_shape
    return density.reshape((gz, gy, gx))


def build_teacher_worker(job: tuple[int, str, pp.ReferenceStats, dict[str, Any], tuple[int, int, int], int, argparse.Namespace]):
    index, path_text, ref, item, grid_shape, chunk_atoms, args = job
    path = Path(path_text)
    vector, support, scale = density_grid_teacher(path, ref, item["mass_min"], item["mass_max"], grid_shape, chunk_atoms)
    range_ref = pp.build_mass_range_reference(path, ref, args, item["mass_min"], item["mass_max"])
    return index, vector, support, scale, range_ref


def pack_bits(mask: np.ndarray) -> np.ndarray:
    return np.packbits(mask.astype(np.uint8), bitorder="big")


def half_bytes(array: np.ndarray) -> bytes:
    return np.asarray(array, dtype=np.float16).astype("<f2", copy=False).tobytes()


def float32_bytes(array: np.ndarray) -> bytes:
    return np.asarray(array, dtype="<f4").tobytes()


def uint8_bytes(array: np.ndarray) -> bytes:
    return np.asarray(array, dtype=np.uint8).tobytes()


def int8_bytes(array: np.ndarray) -> bytes:
    return np.asarray(array, dtype=np.int8).tobytes()


def quantize_basis_int8(basis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scale = np.maximum(np.max(np.abs(basis), axis=1) / 127.0, 1e-12).astype(np.float32)
    quantized = np.rint(basis / scale[:, None]).clip(-127, 127).astype(np.int8)
    return quantized, scale


def add_array(arrays: list[dict[str, Any]], chunks: list[bytes], name: str, dtype: str, shape: list[int], body: bytes) -> None:
    offset = sum(len(chunk) for chunk in chunks)
    arrays.append({"name": name, "dtype": dtype, "shape": shape, "offset": offset, "nbytes": len(body)})
    chunks.append(body)


def write_hypernetwork_artifact(
    path: Path,
    *,
    ref: pp.ReferenceStats,
    ranges: list[dict[str, Any]],
    settings: dict[str, Any],
    hyper_metrics: dict[str, Any],
    coeff_model: Any,
    mean: np.ndarray,
    basis: np.ndarray,
    coeff_mean: np.ndarray,
    coeff_std: np.ndarray,
    supports: list[np.ndarray],
    basis_dtype: str,
) -> int:
    arrays: list[dict[str, Any]] = []
    chunks: list[bytes] = []

    add_array(arrays, chunks, "mean", "float16", [int(mean.size)], half_bytes(mean))
    if basis_dtype == "int8":
        basis_q, basis_scale = quantize_basis_int8(basis)
        add_array(arrays, chunks, "basis", "int8", [int(basis_q.shape[0]), int(basis_q.shape[1])], int8_bytes(basis_q))
        add_array(arrays, chunks, "basis_scale", "float32", [int(basis_scale.size)], float32_bytes(basis_scale))
    else:
        add_array(arrays, chunks, "basis", "float16", [int(basis.shape[0]), int(basis.shape[1])], half_bytes(basis))
    add_array(arrays, chunks, "coeff_mean", "float32", [int(coeff_mean.size)], float32_bytes(coeff_mean))
    add_array(arrays, chunks, "coeff_std", "float32", [int(coeff_std.size)], float32_bytes(coeff_std))
    add_array(
        arrays,
        chunks,
        "support_masks",
        "uint8",
        [len(supports), int(math.ceil(mean.size / 8))],
        uint8_bytes(np.stack([pack_bits(support) for support in supports])),
    )

    state = {k: v.detach().cpu().numpy() for k, v in coeff_model.state_dict().items()}
    state_order = ["net.0.weight", "net.0.bias", "net.2.weight", "net.2.bias", "net.4.weight", "net.4.bias"]
    for key in state_order:
        add_array(arrays, chunks, f"coeff.{key}", "float32", list(state[key].shape), float32_bytes(state[key]))

    header = {
        "version": 1,
        "kind": "cdf_density_grid_hypernetwork",
        "dataset_bounds": pp.json_ready(ref.bounds),
        "dataset_mass_range": pp.json_ready(ref.mass_range),
        "ranges": [{k: pp.json_ready(v) for k, v in item.items()} for item in ranges],
        "settings": {k: pp.json_ready(v) for k, v in {**settings, **hyper_metrics}.items()},
        "arrays": arrays,
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", len(header_bytes)))
        f.write(header_bytes)
        for chunk in chunks:
            f.write(chunk)
    return path.stat().st_size


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    manifest_path = args.out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    ds = next(
        d for d in manifest["datasets"]
        if args.dataset_filter.lower() in d["name"].lower() or args.dataset_filter.lower() in d["id"].lower()
    )
    path = Path(ds["raw_path"])
    ds_dir = args.out_dir / "datasets" / ds["id"]
    ref = load_reference(ds, ds_dir)

    import torch
    import torch.nn as nn  # noqa: F401 - needed by imported hypernetwork trainer
    import torch.nn.functional as F

    device = "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
    ranges = cdf_ranges(path, ref, args)
    grid_shape = parse_grid_shape(args.grid_shape, args.grid)
    gx, gy, gz = grid_shape
    param_count = gx * gy * gz
    args.hyper_rank = choose_hyper_rank(args, param_count, len(ranges))
    estimated_size = estimate_artifact_bytes(param_count, len(ranges), args.hyper_rank, args.basis_dtype)
    print(f"Built {len(ranges)} equal-CDF mass ranges:")
    for item in ranges:
        print(
            f"  {item['id']} mass={item['mass']:.3f} window={item['mass_min']:.3f}-{item['mass_max']:.3f} "
            f"atoms={item['atom_count']:,}",
            flush=True,
        )
    print(
        f"Teacher grid shape: {grid_label(grid_shape)} ({param_count:,} cells, {param_count * 4 / (1024 * 1024):.2f} MB float32 each)",
        flush=True,
    )
    print(
        f"Hypernetwork basis: rank={args.hyper_rank}, dtype={args.basis_dtype}, estimated artifact={pp.mb(estimated_size):.2f} MB",
        flush=True,
    )

    teacher_vectors: list[np.ndarray | None] = [None] * len(ranges)
    supports: list[np.ndarray | None] = [None] * len(ranges)
    target_scales: list[float | None] = [None] * len(ranges)
    range_refs: list[pp.ReferenceStats | None] = [None] * len(ranges)
    settings = {
        "grid": args.grid,
        "grid_shape": [gx, gy, gz],
        "teacher_param_count": param_count,
        "density_network": f"trilinear {grid_label(grid_shape)} log-density grid",
        "input": "mass/charge range center",
        "output": "xyz-to-density grid weights",
        "hyper_rank": args.hyper_rank,
        "hyper_epochs": args.hyper_epochs,
        "basis_dtype": args.basis_dtype,
        "target_artifact_mb": args.target_artifact_mb,
    }

    jobs = [
        (i, str(path), ref, item, grid_shape, args.chunk_atoms, args)
        for i, item in enumerate(ranges)
    ]
    worker_count = max(1, min(int(args.teacher_workers), len(jobs)))
    print(f"Building {len(jobs)} teacher grids with {worker_count} worker(s)...", flush=True)
    if worker_count == 1:
        for job in jobs:
            i = job[0]
            item = ranges[i]
            print(f"  teacher {i + 1}/{len(ranges)}: {item['id']} @ {item['mass']:.3f}", flush=True)
            index, vector, support, scale, range_ref = build_teacher_worker(job)
            teacher_vectors[index] = vector
            supports[index] = support
            target_scales[index] = scale
            range_refs[index] = range_ref
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as pool:
            futures = [pool.submit(build_teacher_worker, job) for job in jobs]
            for done_count, future in enumerate(as_completed(futures), start=1):
                index, vector, support, scale, range_ref = future.result()
                item = ranges[index]
                teacher_vectors[index] = vector
                supports[index] = support
                target_scales[index] = scale
                range_refs[index] = range_ref
                print(f"  teacher {done_count}/{len(ranges)} complete: {item['id']} @ {item['mass']:.3f}", flush=True)
    for item, scale in zip(ranges, target_scales):
        item["target_scale"] = float(scale or 1.0)

    teacher_matrix = np.stack([v for v in teacher_vectors if v is not None]).astype(np.float32)
    masses = np.array([item["mass"] for item in ranges], dtype=np.float32)
    print("Training CDF hypernetwork over range teacher grids...", flush=True)
    coeff_model, mean, basis, coeff_mean, coeff_std, generated_weights, hyper_metrics = train_hypernetwork(
        masses, teacher_matrix, args, torch, torch.nn, F, device
    )
    metric_weights = generated_weights
    if args.basis_dtype == "int8":
        basis_q, basis_scale = quantize_basis_int8(basis)
        basis_for_artifact = basis_q.astype(np.float32) * basis_scale[:, None]
        predicted_coeff = (generated_weights - mean[None, :]) @ basis.T
        metric_weights = (mean[None, :] + predicted_coeff @ basis_for_artifact).astype(np.float32)
        hyper_metrics["relative_teacher_weight_error_quantized_basis"] = float(
            np.linalg.norm(metric_weights - teacher_matrix) / max(np.linalg.norm(teacher_matrix - mean[None, :]), 1e-6)
        )

    variants: list[dict[str, Any]] = []
    support_list = [support for support in supports if support is not None]
    range_ref_list = [range_ref for range_ref in range_refs if range_ref is not None]
    for i, (item, weights, support, range_ref) in enumerate(zip(ranges, metric_weights, support_list, range_ref_list)):
        print(f"Computing metric sample for {item['id']}...", flush=True)
        density = generated_grid_from_weights(weights, item["target_scale"], support, grid_shape)
        point_target = min(int(item["atom_count"]), args.metric_points)
        points = pp.sample_from_density_weights(density, grid_shape, ref.bounds, item["mass"], point_target, args.seed + i * 997)
        metrics = pp.range_density_metrics(points, range_ref)
        variants.append(
            {
                "id": item["id"],
                "label": f"CDF range {i + 1:02d} @ {item['mass']:.2f} ({item['mass_min']:.2f}-{item['mass_max']:.2f})",
                "available": True,
                "points": int(point_target),
                "compressed_size_bytes": 0,
                "compression_ratio": 0.0,
                "metrics": metrics,
                "range": {k: pp.json_ready(v) for k, v in item.items() if k not in ("target_scale",)},
                "notes": "Generated in the browser from the single CDF hypernetwork artifact.",
            }
        )

    hyper_dir = ds_dir / "cdf_hypernetwork"
    hyper_path = hyper_dir / "cdf_hypernetwork.bin"
    compressed_size = write_hypernetwork_artifact(
        hyper_path,
        ref=ref,
        ranges=ranges,
        settings=settings,
        hyper_metrics=hyper_metrics,
        coeff_model=coeff_model,
        mean=mean,
        basis=basis,
        coeff_mean=coeff_mean,
        coeff_std=coeff_std,
        supports=support_list,
        basis_dtype=args.basis_dtype,
    )
    for variant in variants:
        variant["compressed_size_bytes"] = compressed_size
        variant["compression_ratio"] = ref.raw_size_bytes / max(compressed_size, 1)

    method = {
        "label": "CDF hypernetwork",
        "method_label": "CDF hypernetwork",
        "available": True,
        "frontend_generated": True,
        "compressed_size_bytes": compressed_size,
        "compression_ratio": ref.raw_size_bytes / max(compressed_size, 1),
        "points": sum(item["atom_count"] for item in ranges),
        "artifact": str(hyper_path),
        "artifact_endpoint": f"/api/hypernetwork/{ds['id']}/cdf_hypernetwork",
        "preprocess_sec": time.perf_counter() - started,
        "metrics": None,
        "settings": {k: pp.json_ready(v) for k, v in {**settings, **hyper_metrics}.items()},
        "ranges": [{k: pp.json_ready(v) for k, v in item.items() if k not in ("target_scale",)} for item in ranges],
        "range_metrics": variants,
        "notes": f"Single browser-side hypernetwork maps CDF range centers to {grid_label(grid_shape)} xyz-to-density grid networks.",
    }

    ds.setdefault("methods", {})["cdf_hypernetwork"] = method
    manifest.setdefault("methods", {})["cdf_hypernetwork"] = "CDF hypernetwork"
    pp.write_json(ds_dir / "dataset_manifest.json", ds)
    pp.write_json(manifest_path, manifest)

    print(f"CDF hypernetwork: {pp.mb(compressed_size):.2f} MB, ratio {method['compression_ratio']:.2f}x")
    print(f"Manifest updated: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
