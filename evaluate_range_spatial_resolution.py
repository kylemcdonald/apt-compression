from __future__ import annotations

import argparse
import json
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np

import preprocess as pp
from hypernetwork_common import load_reference
from train_range_nif import find_dataset, grid_loss_metrics, load_range_points, log_target, parse_shape


GRID_MAGIC = b"RNGGRID1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate 31-32 Da representations on a higher-resolution comparison grid.")
    parser.add_argument("--dataset-filter", default="499e563f-0c0c-4c6f-bc08-b8e76f59c31b")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--mass-min", type=float, default=31.0)
    parser.add_argument("--mass-max", type=float, default=32.0)
    parser.add_argument("--base-grid-shape", default="128,128,512")
    parser.add_argument("--eval-grid-shape", default="256,256,1024")
    parser.add_argument("--chunk-atoms", type=int, default=2_000_000)
    parser.add_argument("--chunk-cells", type=int, default=262144)
    parser.add_argument("--target-methods", default="range31_grid_q4,range31_sparse_q4")
    parser.add_argument("--spectrum-bins", type=int, default=1024)
    parser.add_argument("--profile-bins", type=int, default=64)
    parser.add_argument("--spatial-bins", type=int, default=6)
    return parser.parse_args()


def read_grid_artifact(path: Path) -> tuple[dict[str, Any], bytes]:
    with path.open("rb") as f:
        magic = f.read(8)
        if magic != GRID_MAGIC:
            raise ValueError(f"{path} is not a range grid artifact")
        header_len = struct.unpack("<I", f.read(4))[0]
        header = json.loads(f.read(header_len))
        payload = f.read()
    return header, payload


def unpack_nibbles(payload: bytes, count: int) -> np.ndarray:
    packed = np.frombuffer(payload, dtype=np.uint8)
    out = np.empty(packed.size * 2, dtype=np.uint8)
    out[0::2] = packed >> 4
    out[1::2] = packed & 15
    return out[:count]


def upsample_q4_to_eval(q4_base: np.ndarray, base_shape: tuple[int, int, int], eval_shape: tuple[int, int, int]) -> np.ndarray:
    bx, by, bz = base_shape
    ex, ey, ez = eval_shape
    if ex % bx or ey % by or ez % bz:
        raise ValueError("Eval shape must be an integer multiple of base shape for q4 upsampling.")
    rx, ry, rz = ex // bx, ey // by, ez // bz
    base = q4_base.reshape((bz, by, bx)).astype(np.float32) / 15.0
    return np.repeat(np.repeat(np.repeat(base, rz, axis=0), ry, axis=1), rx, axis=2)


def gradient_energy(log_grid: np.ndarray) -> float:
    total = 0.0
    count = 0
    for axis in range(3):
        diff = np.diff(log_grid, axis=axis)
        total += float(np.mean(diff * diff))
        count += 1
    return total / max(count, 1)


def occupied_edge_error(pred_log: np.ndarray, target_log: np.ndarray, target_counts: np.ndarray) -> float:
    occupied = target_counts > 0
    edge = np.zeros_like(occupied, dtype=bool)
    for axis in range(3):
        edge_slice_a = [slice(None)] * 3
        edge_slice_b = [slice(None)] * 3
        edge_slice_a[axis] = slice(1, None)
        edge_slice_b[axis] = slice(None, -1)
        mismatch = occupied[tuple(edge_slice_a)] != occupied[tuple(edge_slice_b)]
        edge[tuple(edge_slice_a)] |= mismatch
        edge[tuple(edge_slice_b)] |= mismatch
    if not edge.any():
        return 0.0
    diff = pred_log[edge] - target_log[edge]
    return float(np.mean(diff * diff))


def high_res_metrics(pred_log: np.ndarray, target_log: np.ndarray, target_counts: np.ndarray) -> dict[str, float]:
    base = grid_loss_metrics(pred_log, target_log, target_counts)
    base["hires_gradient_energy"] = gradient_energy(pred_log)
    base["hires_target_gradient_energy"] = gradient_energy(target_log)
    base["hires_gradient_energy_ratio"] = base["hires_gradient_energy"] / max(base["hires_target_gradient_energy"], 1e-12)
    base["hires_edge_log_mse"] = occupied_edge_error(pred_log, target_log, target_counts)
    return {f"hires_{key}" if not key.startswith("hires_") else key: value for key, value in base.items()}


def merge_manifest_metrics(manifest_path: Path, dataset_id: str, updates: dict[str, dict[str, float]]) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for dataset in manifest.get("datasets", []):
        if dataset.get("id") != dataset_id:
            continue
        for method, metrics in updates.items():
            if method in dataset.get("methods", {}):
                dataset["methods"][method].setdefault("metrics", {}).update(metrics)
    pp.write_json(manifest_path, manifest)
    ds_manifest = manifest_path.parent / "datasets" / dataset_id / "dataset_manifest.json"
    if ds_manifest.exists():
        ds = next(dataset for dataset in manifest["datasets"] if dataset["id"] == dataset_id)
        pp.write_json(ds_manifest, ds)


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    base_shape = parse_shape(args.base_grid_shape)
    eval_shape = parse_shape(args.eval_grid_shape)
    manifest_path = args.out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset = find_dataset(manifest, args.dataset_filter)
    raw_path = Path(dataset["raw_path"])
    ds_dir = args.out_dir / "datasets" / dataset["id"]
    ref = load_reference(dataset, ds_dir)
    methods = dataset.get("methods", {})

    print(f"Building high-res target {eval_shape} for {args.mass_min:g}-{args.mass_max:g} Da...", flush=True)
    points = load_range_points(raw_path, args.mass_min, args.mass_max, args.chunk_atoms)
    target_counts = pp.build_mass_range_density_grid(raw_path, ref, eval_shape, args.mass_min, args.mass_max, args.chunk_atoms)
    target_log, target_scale = log_target(target_counts)
    print(f"  atoms={len(points):,}, cells={np.prod(eval_shape):,}, target_scale={target_scale:.6f}", flush=True)

    updates: dict[str, dict[str, float]] = {}
    for method in [item.strip() for item in args.target_methods.split(",") if item.strip()]:
        info = methods[method]
        print(f"Evaluating {method}...", flush=True)
        if method in {"range31_grid_q4", "range31_sparse_q4"}:
            q4_artifact = Path(methods["range31_grid_q4"]["artifact"])
            header, payload = read_grid_artifact(q4_artifact)
            q4 = unpack_nibbles(payload, int(np.prod(base_shape)))
            pred_log = upsample_q4_to_eval(q4, base_shape, eval_shape)
        else:
            raise ValueError(f"No high-res evaluator for {method}")
        metrics = high_res_metrics(pred_log, target_log, target_counts)
        metrics["hires_grid_cells"] = int(np.prod(eval_shape))
        metrics["hires_grid_x"] = eval_shape[0]
        metrics["hires_grid_y"] = eval_shape[1]
        metrics["hires_grid_z"] = eval_shape[2]
        updates[method] = metrics
        print(
            f"  log_mse={metrics['hires_log_mse']:.8f} "
            f"pos_mse={metrics['hires_positive_log_mse']:.8f} "
            f"rel_l1={metrics['hires_relative_count_l1']:.6f} "
            f"grad_ratio={metrics['hires_gradient_energy_ratio']:.4f} "
            f"edge_mse={metrics['hires_edge_log_mse']:.8f}",
            flush=True,
        )

    out_path = ds_dir / f"range_neural_{args.mass_min:g}_{args.mass_max:g}".replace(".", "p") / "hires_metrics.json"
    pp.write_json(out_path, {"eval_shape": list(eval_shape), "updates": updates, "elapsed_sec": time.perf_counter() - started})
    merge_manifest_metrics(manifest_path, dataset["id"], updates)
    print(f"High-res metrics written to {out_path}")
    print(f"Manifest updated: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
