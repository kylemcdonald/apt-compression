from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

import preprocess as pp
from hypernetwork_common import load_reference
from train_range_nif import (
    counts_from_points,
    find_dataset,
    grid_loss_metrics,
    load_range_points,
    log_target,
    make_method,
    mb,
    pack_nibbles,
    parse_shape,
    sample_points_from_density,
    shape_label,
    upsample_coarse_counts_to_fine,
    write_grid_artifact,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train newer 31-32 Da neural density representations.")
    parser.add_argument("--dataset-filter", default="499e563f-0c0c-4c6f-bc08-b8e76f59c31b")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--mass-min", type=float, default=31.0)
    parser.add_argument("--mass-max", type=float, default=32.0)
    parser.add_argument("--coarse-grid-shape", default="64,64,256")
    parser.add_argument("--fine-grid-shape", default="128,128,512")
    parser.add_argument("--display-points", type=int, default=None)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch", type=int, default=65536)
    parser.add_argument("--eval-every", type=int, default=150)
    parser.add_argument("--val-samples", type=int, default=250000)
    parser.add_argument("--eval-batch", type=int, default=262144)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=456)
    parser.add_argument("--chunk-atoms", type=int, default=2_000_000)
    parser.add_argument("--spectrum-bins", type=int, default=1024)
    parser.add_argument("--profile-bins", type=int, default=64)
    parser.add_argument("--spatial-bins", type=int, default=6)
    return parser.parse_args()


def import_torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    return torch, nn, F


def split_indices(idx, gx: int, gy: int):
    xy = gx * gy
    z = idx // xy
    rem = idx - z * xy
    y = rem // gx
    x = rem - y * gx
    return x, y, z


def normalized_coords(idx, shape: tuple[int, int, int], torch: Any):
    gx, gy, _ = shape
    x, y, z = split_indices(idx, gx, gy)
    gz = shape[2]
    return torch.stack(
        [
            ((x.to(torch.float32) + 0.5) / shape[0]) * 2.0 - 1.0,
            ((y.to(torch.float32) + 0.5) / shape[1]) * 2.0 - 1.0,
            ((z.to(torch.float32) + 0.5) / shape[2]) * 2.0 - 1.0,
        ],
        dim=1,
    )


def model_param_count(model: Any) -> int:
    return int(sum(param.numel() for param in model.parameters()))


def support_mask_bytes(pos_idx: np.ndarray, cells: int) -> bytes:
    mask = np.zeros(cells, dtype=np.uint8)
    mask[pos_idx] = 1
    return np.packbits(mask, bitorder="big").tobytes()


def write_representation_npz(
    path: Path,
    model: Any | None,
    metadata: dict[str, Any],
    extra_arrays: dict[str, np.ndarray] | None = None,
    model_dtype: str = "float16",
) -> int:
    arrays: dict[str, Any] = {}
    if model is not None:
        dtype = np.float16 if model_dtype == "float16" else np.float32
        arrays.update({name: param.detach().cpu().numpy().astype(dtype) for name, param in model.state_dict().items()})
    for name, value in (extra_arrays or {}).items():
        arrays[name] = value
    arrays["metadata_json"] = np.array(json.dumps(metadata, separators=(",", ":")), dtype=np.str_)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return path.stat().st_size


def hash_levels(shape: tuple[int, int, int]) -> list[tuple[int, int, int]]:
    gx, gy, gz = shape
    return [
        (max(4, gx // 8), max(4, gy // 8), max(8, gz // 8)),
        (max(4, gx // 4), max(4, gy // 4), max(8, gz // 4)),
        (max(4, gx // 2), max(4, gy // 2), max(8, gz // 2)),
        (gx, gy, gz),
    ]


def make_mlp(nn: Any, in_dim: int, out_dim: int, width: int = 512, hidden_layers: int = 4):
    layers: list[Any] = []
    last = in_dim
    for _ in range(hidden_layers):
        layers.extend([nn.Linear(last, width), nn.SiLU()])
        last = width
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


class HashFieldMixin:
    def hash_encode(self, idx):
        torch = self._torch
        gx, gy, gz = self.shape
        x, y, z = split_indices(idx, gx, gy)
        features = [normalized_coords(idx, self.shape, torch)]
        for table, (rx, ry, rz), level in zip(self.tables, self.levels, range(len(self.levels))):
            xi = torch.clamp((x * rx) // gx, 0, rx - 1)
            yi = torch.clamp((y * ry) // gy, 0, ry - 1)
            zi = torch.clamp((z * rz) // gz, 0, rz - 1)
            h = (xi * 73856093 + yi * 19349663 + zi * 83492791 + level * 2654435761) % self.hash_size
            features.append(table(h))
        return torch.cat(features, dim=1)


def build_hash_regressor(shape: tuple[int, int, int], torch: Any, nn: Any, out_dim: int = 1):
    class HashRegressor(nn.Module, HashFieldMixin):
        def __init__(self):
            super().__init__()
            self._torch = torch
            self.shape = shape
            self.hash_size = 32768
            self.levels = hash_levels(shape)
            self.tables = nn.ModuleList([nn.Embedding(self.hash_size, 2) for _ in self.levels])
            self.net = make_mlp(nn, 3 + len(self.levels) * 2, out_dim, width=512, hidden_layers=4)

        def forward(self, idx):
            out = self.net(self.hash_encode(idx))
            return out.squeeze(1) if out.shape[1] == 1 else out

    return HashRegressor()


def build_block_field(shape: tuple[int, int, int], torch: Any, nn: Any):
    class BlockLatentField(nn.Module):
        def __init__(self):
            super().__init__()
            self.shape = shape
            self._torch = torch
            self.blocks = 16
            self.latent_dim = 128
            self.latents = nn.Embedding(self.blocks, self.latent_dim)
            self.net = make_mlp(nn, self.latent_dim + 6, 1, width=512, hidden_layers=4)

        def forward(self, idx):
            gx, gy, gz = self.shape
            x, y, z = split_indices(idx, gx, gy)
            block = torch.clamp((z * self.blocks) // gz, 0, self.blocks - 1)
            block_start = block.to(torch.float32) / self.blocks
            block_end = (block.to(torch.float32) + 1.0) / self.blocks
            zn = (z.to(torch.float32) + 0.5) / gz
            local_z = ((zn - block_start) / torch.clamp(block_end - block_start, min=1e-6)) * 2.0 - 1.0
            coords = normalized_coords(idx, self.shape, torch)
            local = torch.stack([coords[:, 0], coords[:, 1], local_z], dim=1)
            return self.net(torch.cat([coords, local, self.latents(block)], dim=1)).squeeze(1)

    return BlockLatentField()


def sample_indices(source: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    return rng.choice(source, size=count, replace=len(source) < count).astype(np.int64)


def make_validation_indices(pos_idx: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    return sample_indices(pos_idx, min(count, len(pos_idx)), rng)


def evaluate_regression(model: Any, indices: np.ndarray, target: np.ndarray, torch: Any, device: str, batch: int) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for start in range(0, len(indices), batch):
            idx_np = indices[start : start + batch]
            idx = torch.from_numpy(idx_np).to(device)
            y = torch.from_numpy(target[idx_np].astype(np.float32, copy=False)).to(device)
            losses.append(((model(idx) - y) ** 2).mean().detach().cpu().item())
    return float(np.mean(losses)) if losses else 0.0


def evaluate_classifier(model: Any, indices: np.ndarray, target: np.ndarray, torch: Any, device: str, batch: int) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for start in range(0, len(indices), batch):
            idx_np = indices[start : start + batch]
            idx = torch.from_numpy(idx_np).to(device)
            y = torch.from_numpy(target[idx_np].astype(np.int64, copy=False)).to(device)
            losses.append(torch.nn.functional.cross_entropy(model(idx), y).detach().cpu().item())
    return float(np.mean(losses)) if losses else 0.0


def train_regressor(
    label: str,
    model: Any,
    target_flat: np.ndarray,
    train_idx: np.ndarray,
    args: argparse.Namespace,
    torch: Any,
    F: Any,
    device: str,
) -> tuple[Any, dict[str, Any]]:
    seed = args.seed + sum((i + 1) * ord(ch) for i, ch in enumerate(label))
    rng = np.random.default_rng(seed)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    val_idx = make_validation_indices(train_idx, args.val_samples, rng)
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    started = time.perf_counter()

    for step in range(1, args.steps + 1):
        batch_idx = sample_indices(train_idx, args.batch, rng)
        idx_t = torch.from_numpy(batch_idx).to(device)
        y_t = torch.from_numpy(target_flat[batch_idx].astype(np.float32, copy=False)).to(device)
        pred = model(idx_t)
        loss = F.mse_loss(pred, y_t)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            val = evaluate_regression(model, val_idx, target_flat, torch, device, args.eval_batch)
            if val < best_val:
                best_val = val
                best_state = copy.deepcopy(model.state_dict())
            print(f"    {label:24s} step {step:5d}/{args.steps} train={loss.detach().cpu().item():.6f} val={val:.6f}", flush=True)

    model.load_state_dict(best_state)
    return model, {
        "param_count": model_param_count(model),
        "best_positive_val_mse": best_val,
        "train_sec": time.perf_counter() - started,
    }


def train_classifier(
    label: str,
    model: Any,
    target_class: np.ndarray,
    train_idx: np.ndarray,
    args: argparse.Namespace,
    torch: Any,
    F: Any,
    device: str,
) -> tuple[Any, dict[str, Any]]:
    seed = args.seed + sum((i + 1) * ord(ch) for i, ch in enumerate(label))
    rng = np.random.default_rng(seed)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    val_idx = make_validation_indices(train_idx, args.val_samples, rng)
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    started = time.perf_counter()

    for step in range(1, args.steps + 1):
        batch_idx = sample_indices(train_idx, args.batch, rng)
        idx_t = torch.from_numpy(batch_idx).to(device)
        y_t = torch.from_numpy(target_class[batch_idx].astype(np.int64, copy=False)).to(device)
        logits = model(idx_t)
        loss = F.cross_entropy(logits, y_t)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            val = evaluate_classifier(model, val_idx, target_class, torch, device, args.eval_batch)
            if val < best_val:
                best_val = val
                best_state = copy.deepcopy(model.state_dict())
            print(f"    {label:24s} step {step:5d}/{args.steps} train={loss.detach().cpu().item():.6f} val={val:.6f}", flush=True)

    model.load_state_dict(best_state)
    return model, {
        "param_count": model_param_count(model),
        "best_positive_val_ce": best_val,
        "train_sec": time.perf_counter() - started,
    }


def predict_on_support(model: Any, pos_idx: np.ndarray, cells: int, torch: Any, device: str, batch: int) -> np.ndarray:
    out = np.zeros(cells, dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(pos_idx), batch):
            idx_np = pos_idx[start : start + batch]
            idx = torch.from_numpy(idx_np).to(device)
            out[idx_np] = model(idx).detach().cpu().numpy().astype(np.float32, copy=False)
    return out


def predict_classes_on_support(model: Any, pos_idx: np.ndarray, cells: int, torch: Any, device: str, batch: int) -> np.ndarray:
    out = np.zeros(cells, dtype=np.uint8)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(pos_idx), batch):
            idx_np = pos_idx[start : start + batch]
            idx = torch.from_numpy(idx_np).to(device)
            out[idx_np] = model(idx).argmax(dim=1).detach().cpu().numpy().astype(np.uint8, copy=False)
    return out


def entropy_bits(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    counts = np.bincount(values.astype(np.int64), minlength=16).astype(np.float64)
    probs = counts[counts > 0] / counts.sum()
    return float(-(probs * np.log2(probs)).sum() * len(values))


def add_method_from_log_prediction(
    methods: dict[str, Any],
    *,
    key: str,
    label: str,
    pred_log_flat: np.ndarray,
    artifact_size: int,
    artifact_path: Path,
    settings: dict[str, Any],
    notes: str,
    fine_shape: tuple[int, int, int],
    fine_log: np.ndarray,
    fine_counts: np.ndarray,
    fine_scale: float,
    ref: pp.ReferenceStats,
    range_ref: pp.ReferenceStats,
    mass_value: float,
    mass_range: tuple[float, float],
    raw_range_bytes: int,
    display_points: int,
    out_dir: Path,
    seed: int,
) -> None:
    pred_log = np.clip(pred_log_flat.reshape(fine_counts.shape), 0.0, 1.0).astype(np.float32)
    pred_density = np.expm1(pred_log * fine_scale).astype(np.float32)
    points = sample_points_from_density(pred_density, fine_shape, ref.bounds, mass_value, display_points, seed)
    display_path = out_dir / f"{key}_points.npz"
    pp.write_point_pack(display_path, points, ref.bounds, np.array(mass_range, dtype=np.float32))
    metrics = {
        **pp.range_density_metrics(points, range_ref),
        **grid_loss_metrics(pred_log, fine_log, fine_counts),
        **settings,
    }
    methods[key] = make_method(
        label=label,
        raw_size_bytes=raw_range_bytes,
        compressed_size_bytes=artifact_size,
        points=len(points),
        display_artifact=str(display_path),
        artifact=str(artifact_path),
        metrics=metrics,
        notes=notes,
        settings=settings,
    )


def remove_old_range31_neural_methods(dataset: dict[str, Any], manifest: dict[str, Any], ds_dir: Path) -> None:
    old_keys = [key for key in dataset.get("methods", {}) if key.startswith("range31_nif_") or key.startswith("range31_new_")]
    for key in old_keys:
        dataset["methods"].pop(key, None)
        manifest.get("methods", {}).pop(key, None)
    old_dir = ds_dir / "range_neural_31_32"
    if old_dir.exists():
        shutil.rmtree(old_dir)


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    mass_range = (float(args.mass_min), float(args.mass_max))
    mass_value = float((args.mass_min + args.mass_max) * 0.5)
    coarse_shape = parse_shape(args.coarse_grid_shape)
    fine_shape = parse_shape(args.fine_grid_shape)
    manifest_path = args.out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset = find_dataset(manifest, args.dataset_filter)
    raw_path = Path(dataset["raw_path"])
    ds_dir = args.out_dir / "datasets" / dataset["id"]
    ref = load_reference(dataset, ds_dir)
    out_dir = ds_dir / f"range_neural_{args.mass_min:g}_{args.mass_max:g}".replace(".", "p")
    remove_old_range31_neural_methods(dataset, manifest, ds_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading exact {args.mass_min:g}-{args.mass_max:g} Da atoms from {raw_path.name}...", flush=True)
    points_exact = load_range_points(raw_path, args.mass_min, args.mass_max, args.chunk_atoms)
    atom_count = int(len(points_exact))
    display_points = atom_count if args.display_points is None else min(int(args.display_points), atom_count)
    raw_range_bytes = atom_count * 16
    print(f"  found {atom_count:,} atoms ({mb(raw_range_bytes):.2f} MiB raw range)", flush=True)

    range_ref = pp.build_mass_range_reference(raw_path, ref, args, args.mass_min, args.mass_max)
    print(f"Building grids: coarse {shape_label(coarse_shape)}, fine {shape_label(fine_shape)}...", flush=True)
    coarse_counts = counts_from_points(points_exact, ref.bounds, coarse_shape)
    fine_counts = counts_from_points(points_exact, ref.bounds, fine_shape)
    coarse_log, coarse_scale = log_target(coarse_counts)
    fine_log, fine_scale = log_target(fine_counts)
    fine_flat = fine_log.reshape(-1).astype(np.float32, copy=False)
    cells = int(fine_flat.size)
    counts_flat = fine_counts.reshape(-1)
    pos_idx = np.flatnonzero(counts_flat > 0).astype(np.int64)
    print(f"  fine cells={cells:,}, occupied={len(pos_idx):,}, empty={cells - len(pos_idx):,}", flush=True)

    methods: dict[str, Any] = {}
    exact_metrics = pp.range_density_metrics(points_exact, range_ref)
    methods["range31_full"] = make_method(
        label="31-32 Da full data",
        raw_size_bytes=raw_range_bytes,
        compressed_size_bytes=raw_range_bytes,
        points=atom_count,
        display_artifact=None,
        artifact=None,
        raw_range=mass_range,
        metrics={**exact_metrics, "log_mse": 0.0, "relative_count_l1": 0.0},
        notes=f"Exact raw POS atoms filtered to {args.mass_min:g}-{args.mass_max:g} Da on the local server.",
        settings={"mass_range": list(mass_range)},
    )

    grid_f32_path = out_dir / "grid_64x64x256_f32.bin"
    grid_f32_size = write_grid_artifact(
        grid_f32_path,
        kind="range_log_grid_f32",
        shape=coarse_shape,
        mass_range=mass_range,
        atom_count=atom_count,
        target_scale=coarse_scale,
        payload=coarse_log.astype("<f4", copy=False).tobytes(),
        payload_dtype="float32",
    )
    coarse_points = sample_points_from_density(coarse_counts, coarse_shape, ref.bounds, mass_value, display_points, args.seed + 10)
    coarse_display = out_dir / "grid_64x64x256_f32_points.npz"
    pp.write_point_pack(coarse_display, coarse_points, ref.bounds, np.array(mass_range, dtype=np.float32))
    coarse_up_counts = upsample_coarse_counts_to_fine(coarse_counts, coarse_shape, fine_shape)
    coarse_up_log = (np.log1p(coarse_up_counts) / fine_scale).astype(np.float32)
    methods["range31_grid_f32"] = make_method(
        label="31-32 Da grid f32 64x64x256",
        raw_size_bytes=raw_range_bytes,
        compressed_size_bytes=grid_f32_size,
        points=len(coarse_points),
        display_artifact=str(coarse_display),
        artifact=str(grid_f32_path),
        metrics={**pp.range_density_metrics(coarse_points, range_ref), **grid_loss_metrics(coarse_up_log, fine_log, fine_counts)},
        notes="Dense 32-bit normalized log-count grid; sampled uniformly within occupied density cells for display.",
        settings={"grid_shape": list(coarse_shape), "target_scale": coarse_scale},
    )

    q4 = np.rint(np.clip(fine_log, 0.0, 1.0) * 15.0).astype(np.uint8)
    q4_flat = q4.reshape(-1)
    grid_q4_path = out_dir / "grid_128x128x512_q4.bin"
    grid_q4_size = write_grid_artifact(
        grid_q4_path,
        kind="range_log_grid_q4",
        shape=fine_shape,
        mass_range=mass_range,
        atom_count=atom_count,
        target_scale=fine_scale,
        payload=pack_nibbles(q4),
        payload_dtype="uint4",
    )
    q4_log = (q4.astype(np.float32) / 15.0).reshape(fine_counts.shape)
    q4_density = np.expm1(q4_log * fine_scale).astype(np.float32)
    q4_points = sample_points_from_density(q4_density, fine_shape, ref.bounds, mass_value, display_points, args.seed + 20)
    q4_display = out_dir / "grid_128x128x512_q4_points.npz"
    pp.write_point_pack(q4_display, q4_points, ref.bounds, np.array(mass_range, dtype=np.float32))
    q4_metrics = {**pp.range_density_metrics(q4_points, range_ref), **grid_loss_metrics(q4_log, fine_log, fine_counts)}
    methods["range31_grid_q4"] = make_method(
        label="31-32 Da grid 4-bit 128x128x512",
        raw_size_bytes=raw_range_bytes,
        compressed_size_bytes=grid_q4_size,
        points=len(q4_points),
        display_artifact=str(q4_display),
        artifact=str(grid_q4_path),
        metrics=q4_metrics,
        notes="Four-bit normalized log-count grid at twice the linear resolution of the f32 baseline.",
        settings={"grid_shape": list(fine_shape), "target_scale": fine_scale},
    )

    sparse_payload = support_mask_bytes(pos_idx, cells) + pack_nibbles(q4_flat[pos_idx])
    sparse_q4_path = out_dir / "sparse_q4_support_values.bin"
    sparse_q4_size = write_grid_artifact(
        sparse_q4_path,
        kind="range_sparse_q4_support_values",
        shape=fine_shape,
        mass_range=mass_range,
        atom_count=atom_count,
        target_scale=fine_scale,
        payload=sparse_payload,
        payload_dtype="support_bitmask_plus_uint4_values",
    )
    methods["range31_sparse_q4"] = make_method(
        label="31-32 Da sparse support + q4 values",
        raw_size_bytes=raw_range_bytes,
        compressed_size_bytes=sparse_q4_size,
        points=len(q4_points),
        display_artifact=str(q4_display),
        artifact=str(sparse_q4_path),
        metrics={**q4_metrics, "support_bytes": len(support_mask_bytes(pos_idx, cells))},
        notes="Non-neural sparse baseline: exact occupied support plus 4-bit values only for occupied cells.",
        settings={"grid_shape": list(fine_shape), "target_scale": fine_scale, "occupied_cells": int(len(pos_idx))},
    )

    torch, nn, F = import_torch()
    device = "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
    print(f"Training newer neural representations on {device}...", flush=True)
    support_bytes_array = np.frombuffer(support_mask_bytes(pos_idx, cells), dtype=np.uint8)

    def train_and_add_regression(key: str, label: str, model: Any, target: np.ndarray, base_log: np.ndarray | None, seed: int, notes: str, extra_arrays: dict[str, np.ndarray] | None = None):
        trained, train_metrics = train_regressor(key, model, target, pos_idx, args, torch, F, device)
        pred = predict_on_support(trained, pos_idx, cells, torch, device, args.eval_batch)
        pred_log = pred if base_log is None else base_log.reshape(-1) + pred
        pred_log[counts_flat <= 0] = 0.0
        artifact_path = out_dir / f"{key}.npz"
        extras = {"support_mask": support_bytes_array}
        if extra_arrays:
            extras.update(extra_arrays)
        artifact_size = write_representation_npz(
            artifact_path,
            trained,
            {"method": key, "grid_shape": list(fine_shape), "mass_range": list(mass_range), **train_metrics},
            extras,
            model_dtype="float16",
        )
        settings = {
            "param_count": train_metrics["param_count"],
            "train_sec": train_metrics["train_sec"],
            "best_positive_val_mse": train_metrics["best_positive_val_mse"],
        }
        add_method_from_log_prediction(
            methods,
            key=key,
            label=label,
            pred_log_flat=pred_log,
            artifact_size=artifact_size,
            artifact_path=artifact_path,
            settings=settings,
            notes=notes,
            fine_shape=fine_shape,
            fine_log=fine_log,
            fine_counts=fine_counts,
            fine_scale=fine_scale,
            ref=ref,
            range_ref=range_ref,
            mass_value=mass_value,
            mass_range=mass_range,
            raw_range_bytes=raw_range_bytes,
            display_points=display_points,
            out_dir=out_dir,
            seed=seed,
        )
        return trained

    train_and_add_regression(
        "range31_new_support_hash",
        "31-32 Da support hash NIF",
        build_hash_regressor(fine_shape, torch, nn, out_dim=1),
        fine_flat,
        None,
        args.seed + 101,
        "Exact support bitmask plus hash-grid neural field for occupied-cell normalized log counts; outside support is exactly zero.",
    )

    coarse_q4 = np.rint(np.clip(coarse_log, 0.0, 1.0) * 15.0).astype(np.uint8)
    coarse_q4_log = (coarse_q4.astype(np.float32) / 15.0)
    coarse_q4_up = np.repeat(np.repeat(np.repeat(coarse_q4_log, 2, axis=0), 2, axis=1), 2, axis=2).astype(np.float32)
    residual_target = (fine_log.reshape(-1) - coarse_q4_up.reshape(-1)).astype(np.float32)
    train_and_add_regression(
        "range31_new_coarse_residual_hash",
        "31-32 Da coarse q4 + hash residual",
        build_hash_regressor(fine_shape, torch, nn, out_dim=1),
        residual_target,
        coarse_q4_up,
        args.seed + 102,
        "64x64x256 4-bit coarse grid, exact support bitmask, and hash-grid residual field on occupied fine cells.",
        {"coarse_q4": np.frombuffer(pack_nibbles(coarse_q4), dtype=np.uint8)},
    )

    train_and_add_regression(
        "range31_new_block_latent",
        "31-32 Da support block latent",
        build_block_field(fine_shape, torch, nn),
        fine_flat,
        None,
        args.seed + 104,
        "Exact support bitmask plus z-block latent coordinate field for occupied-cell normalized log counts.",
    )

    classifier = build_hash_regressor(fine_shape, torch, nn, out_dim=16)
    classifier, class_metrics = train_classifier("range31_new_support_classifier", classifier, q4_flat, pos_idx, args, torch, F, device)
    pred_class = predict_classes_on_support(classifier, pos_idx, cells, torch, device, args.eval_batch)
    pred_class_log = (pred_class.astype(np.float32) / 15.0)
    class_artifact = out_dir / "range31_new_support_classifier.npz"
    class_artifact_size = write_representation_npz(
        class_artifact,
        classifier,
        {"method": "range31_new_support_classifier", "grid_shape": list(fine_shape), **class_metrics},
        {"support_mask": support_bytes_array},
        model_dtype="float16",
    )
    add_method_from_log_prediction(
        methods,
        key="range31_new_support_classifier",
        label="31-32 Da support q4 classifier",
        pred_log_flat=pred_class_log,
        artifact_size=class_artifact_size,
        artifact_path=class_artifact,
        settings={
            "param_count": class_metrics["param_count"],
            "train_sec": class_metrics["train_sec"],
            "best_positive_val_ce": class_metrics["best_positive_val_ce"],
        },
        notes="Exact support bitmask plus hash-grid classifier over 16 quantized log-count classes for occupied cells.",
        fine_shape=fine_shape,
        fine_log=fine_log,
        fine_counts=fine_counts,
        fine_scale=fine_scale,
        ref=ref,
        range_ref=range_ref,
        mass_value=mass_value,
        mass_range=mass_range,
        raw_range_bytes=raw_range_bytes,
        display_points=display_points,
        out_dir=out_dir,
        seed=args.seed + 105,
    )

    residual = ((q4_flat[pos_idx].astype(np.int16) - pred_class[pos_idx].astype(np.int16)) & 15).astype(np.uint8)
    residual_bits = entropy_bits(residual)
    entropy_payload_bytes = int(math.ceil(residual_bits / 8.0))
    entropy_size = class_artifact_size + entropy_payload_bytes
    entropy_path = out_dir / "range31_new_entropy_q4_predictor.json"
    pp.write_json(
        entropy_path,
        {
            "kind": "estimated_entropy_coded_q4_residual",
            "classifier_artifact": str(class_artifact),
            "residual_entropy_bits": residual_bits,
            "estimated_residual_bytes": entropy_payload_bytes,
            "estimated_total_bytes": entropy_size,
        },
    )
    methods["range31_new_entropy_q4_predictor"] = make_method(
        label="31-32 Da neural q4 entropy residual",
        raw_size_bytes=raw_range_bytes,
        compressed_size_bytes=entropy_size,
        points=len(q4_points),
        display_artifact=str(q4_display),
        artifact=str(entropy_path),
        metrics={
            **q4_metrics,
            "param_count": class_metrics["param_count"],
            "residual_entropy_bits_per_occupied_cell": float(residual_bits / max(len(pos_idx), 1)),
        },
        notes="Estimated lossless q4 codec: support classifier predicts q4 values, entropy-coded residual reconstructs exact q4 grid.",
        settings={"grid_shape": list(fine_shape), "estimated": True},
    )

    dataset.setdefault("methods", {}).update(methods)
    labels = {key: method["label"] for key, method in methods.items()}
    manifest.setdefault("methods", {}).update(labels)
    pp.write_json(ds_dir / "dataset_manifest.json", dataset)
    pp.write_json(manifest_path, manifest)
    pp.write_json(
        out_dir / "summary.json",
        {
            "dataset": dataset["name"],
            "mass_range": list(mass_range),
            "atom_count": atom_count,
            "preprocess_sec": time.perf_counter() - started,
            "methods": {key: pp.json_ready(value) for key, value in methods.items()},
        },
    )

    print("\n31-32 Da newer representation summary:")
    print("method | size MiB | ratio | log MSE | positive log MSE | relative count L1")
    print("-------+----------+-------+---------+------------------+------------------")
    for key, method in methods.items():
        metrics = method["metrics"]
        print(
            f"{key} | {mb(method['compressed_size_bytes']):.2f} | {method['compression_ratio']:.2f} | "
            f"{float(metrics.get('log_mse', 0.0)):.6f} | {float(metrics.get('positive_log_mse', 0.0)):.6f} | "
            f"{float(metrics.get('relative_count_l1', 0.0)):.4f}",
            flush=True,
        )
    print(f"Manifest updated: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
