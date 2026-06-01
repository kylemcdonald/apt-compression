from __future__ import annotations

import argparse
import copy
import json
import math
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np

import preprocess as pp
from hypernetwork_common import load_reference


GRID_MAGIC = b"RNGGRID1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train xyz-only NIF teachers for one mass range.")
    parser.add_argument("--dataset-filter", default="499e563f-0c0c-4c6f-bc08-b8e76f59c31b")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--mass-min", type=float, default=31.0)
    parser.add_argument("--mass-max", type=float, default=32.0)
    parser.add_argument("--coarse-grid-shape", default="64,64,256")
    parser.add_argument("--fine-grid-shape", default="128,128,512")
    parser.add_argument("--display-points", type=int, default=None)
    parser.add_argument("--variants", default="continuous,fourier,onehot,axis_embedding")
    parser.add_argument("--steps", type=int, default=1800)
    parser.add_argument("--batch", type=int, default=65536)
    parser.add_argument("--eval-every", type=int, default=150)
    parser.add_argument("--val-samples", type=int, default=250000)
    parser.add_argument("--eval-batch", type=int, default=262144)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--chunk-atoms", type=int, default=2_000_000)
    parser.add_argument("--spectrum-bins", type=int, default=1024)
    parser.add_argument("--profile-bins", type=int, default=64)
    parser.add_argument("--spatial-bins", type=int, default=6)
    return parser.parse_args()


def parse_shape(text: str) -> tuple[int, int, int]:
    parts = [part.strip() for part in text.lower().replace("x", ",").split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError(f"Expected x,y,z grid shape, got {text!r}")
    gx, gy, gz = (int(part) for part in parts)
    if gx <= 0 or gy <= 0 or gz <= 0:
        raise ValueError("Grid dimensions must be positive.")
    return gx, gy, gz


def shape_label(shape: tuple[int, int, int]) -> str:
    return f"{shape[0]}x{shape[1]}x{shape[2]}"


def mb(value: int | float) -> float:
    return float(value) / (1024.0 * 1024.0)


def find_dataset(manifest: dict[str, Any], dataset_filter: str) -> dict[str, Any]:
    needle = dataset_filter.lower()
    return next(
        dataset
        for dataset in manifest["datasets"]
        if needle in dataset["name"].lower() or needle in dataset["id"].lower()
    )


def load_range_points(path: Path, mass_min: float, mass_max: float, chunk_atoms: int) -> np.ndarray:
    mm, _, _ = pp.open_pos(path)
    parts: list[np.ndarray] = []
    for _, _, chunk in pp.iter_chunks(mm, chunk_atoms):
        finite = np.isfinite(chunk).all(axis=1)
        if not finite.any():
            continue
        c = chunk[finite]
        keep = (c[:, 3] >= mass_min) & (c[:, 3] <= mass_max)
        if keep.any():
            parts.append(c[keep].astype(np.float32, copy=True))
    if not parts:
        return np.empty((0, 4), dtype=np.float32)
    return np.concatenate(parts, axis=0)


def counts_from_points(points: np.ndarray, bounds: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    gx, gy, gz = shape
    counts = np.zeros(gx * gy * gz, dtype=np.float32)
    if len(points) == 0:
        return counts.reshape((gz, gy, gx))
    extent = np.maximum(bounds[1] - bounds[0], 1e-6)
    norm = np.clip((points[:, :3] - bounds[0]) / extent, 0.0, 0.999999)
    xi = np.floor(norm[:, 0] * gx).astype(np.int32)
    yi = np.floor(norm[:, 1] * gy).astype(np.int32)
    zi = np.floor(norm[:, 2] * gz).astype(np.int32)
    flat = ((zi * gy + yi) * gx + xi).astype(np.int64)
    counts += np.bincount(flat, minlength=counts.size).astype(np.float32)
    return counts.reshape((gz, gy, gx))


def log_target(counts: np.ndarray) -> tuple[np.ndarray, float]:
    scale = max(float(np.log1p(counts.max())), 1.0)
    return (np.log1p(counts) / scale).astype(np.float32), scale


def write_grid_artifact(
    path: Path,
    *,
    kind: str,
    shape: tuple[int, int, int],
    mass_range: tuple[float, float],
    atom_count: int,
    target_scale: float,
    payload: bytes,
    payload_dtype: str,
) -> int:
    header = {
        "version": 1,
        "kind": kind,
        "grid_shape": list(shape),
        "mass_range": list(mass_range),
        "atom_count": int(atom_count),
        "target_scale": float(target_scale),
        "payload_dtype": payload_dtype,
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(GRID_MAGIC)
        f.write(struct.pack("<I", len(header_bytes)))
        f.write(header_bytes)
        f.write(payload)
    return path.stat().st_size


def pack_nibbles(values: np.ndarray) -> bytes:
    flat = np.asarray(values, dtype=np.uint8).reshape(-1)
    if flat.size % 2:
        flat = np.pad(flat, (0, 1), constant_values=0)
    packed = (flat[0::2] << 4) | flat[1::2]
    return packed.tobytes()


def sample_points_from_density(
    density: np.ndarray,
    shape: tuple[int, int, int],
    bounds: np.ndarray,
    mass_value: float,
    target: int,
    seed: int,
) -> np.ndarray:
    return pp.sample_from_density_weights(density.reshape((shape[2], shape[1], shape[0])), shape, bounds, mass_value, target, seed)


def upsample_coarse_counts_to_fine(
    coarse_counts: np.ndarray,
    coarse_shape: tuple[int, int, int],
    fine_shape: tuple[int, int, int],
) -> np.ndarray:
    cx, cy, cz = coarse_shape
    fx, fy, fz = fine_shape
    if fx % cx or fy % cy or fz % cz:
        raise ValueError("Fine grid shape must be an integer multiple of coarse grid shape.")
    rx, ry, rz = fx // cx, fy // cy, fz // cz
    up = np.repeat(np.repeat(np.repeat(coarse_counts, rz, axis=0), ry, axis=1), rx, axis=2)
    return (up / float(rx * ry * rz)).astype(np.float32)


def grid_loss_metrics(pred_log: np.ndarray, target_log: np.ndarray, target_counts: np.ndarray) -> dict[str, float]:
    pred = np.clip(pred_log.astype(np.float32), 0.0, 1.0)
    target = target_log.astype(np.float32)
    diff = pred - target
    pos = target_counts.reshape(-1) > 0
    flat_diff = diff.reshape(-1)
    pred_density = np.expm1(pred * max(float(np.log1p(target_counts.max())), 1.0))
    denom = max(float(target_counts.sum()), 1e-6)
    return {
        "log_mse": float(np.mean(flat_diff * flat_diff)),
        "log_mae": float(np.mean(np.abs(flat_diff))),
        "positive_log_mse": float(np.mean(flat_diff[pos] ** 2)) if pos.any() else 0.0,
        "zero_log_mse": float(np.mean(flat_diff[~pos] ** 2)) if (~pos).any() else 0.0,
        "relative_count_l1": float(np.abs(pred_density - target_counts).sum() / denom),
        "predicted_count_ratio": float(pred_density.sum() / denom),
    }


def update_metric_aliases(metrics: dict[str, Any]) -> dict[str, Any]:
    metrics = dict(metrics)
    metrics.setdefault("spatial_spectral_error", metrics.get("log_mse"))
    return metrics


def make_method(
    *,
    label: str,
    raw_size_bytes: int,
    compressed_size_bytes: int,
    points: int,
    display_artifact: str | None,
    artifact: str | None,
    metrics: dict[str, Any],
    notes: str,
    raw_range: tuple[float, float] | None = None,
    settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    method = {
        "label": label,
        "method_label": label,
        "available": True,
        "raw_size_bytes": int(raw_size_bytes),
        "compressed_size_bytes": int(compressed_size_bytes),
        "compression_ratio": float(raw_size_bytes / max(compressed_size_bytes, 1)),
        "points": int(points),
        "metrics": update_metric_aliases(metrics),
        "notes": notes,
    }
    if display_artifact is not None:
        method["display_artifact"] = display_artifact
    if artifact is not None:
        method["artifact"] = artifact
    if raw_range is not None:
        method["raw_range"] = list(raw_range)
    if settings is not None:
        method["settings"] = settings
    return method


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


def model_param_count(model: Any) -> int:
    return int(sum(param.numel() for param in model.parameters()))


def build_model(variant: str, shape: tuple[int, int, int], torch: Any, nn: Any):
    gx, gy, gz = shape
    width = 512

    def mlp(in_dim: int, hidden_layers: int):
        layers: list[Any] = []
        last = in_dim
        for _ in range(hidden_layers):
            layers.extend([nn.Linear(last, width), nn.SiLU()])
            last = width
        layers.append(nn.Linear(last, 1))
        return nn.Sequential(*layers)

    class ContinuousField(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = mlp(3, 5)

        def encode(self, idx):
            x, y, z = split_indices(idx, gx, gy)
            coords = torch.stack(
                [
                    ((x.to(torch.float32) + 0.5) / gx) * 2.0 - 1.0,
                    ((y.to(torch.float32) + 0.5) / gy) * 2.0 - 1.0,
                    ((z.to(torch.float32) + 0.5) / gz) * 2.0 - 1.0,
                ],
                dim=1,
            )
            return coords

        def forward(self, idx):
            return self.net(self.encode(idx)).squeeze(1)

    class FourierField(ContinuousField):
        def __init__(self):
            super().__init__()
            self.register_buffer("freq", torch.tensor([1.0, 2.0, 4.0, 8.0, 16.0, 32.0], dtype=torch.float32), persistent=False)
            self.net = mlp(39, 5)

        def encode(self, idx):
            coords = super().encode(idx)
            xb = coords[..., None] * self.freq.to(coords.device)
            return torch.cat([coords, torch.sin(math.pi * xb).flatten(1), torch.cos(math.pi * xb).flatten(1)], dim=1)

    class OneHotAxisField(nn.Module):
        def __init__(self):
            super().__init__()
            self.x = nn.Embedding(gx, width)
            self.y = nn.Embedding(gy, width)
            self.z = nn.Embedding(gz, width)
            self.bias = nn.Parameter(torch.zeros(width))
            self.body = mlp(width, 2)

        def forward(self, idx):
            x, y, z = split_indices(idx, gx, gy)
            h = self.x(x) + self.y(y) + self.z(z) + self.bias
            return self.body(h).squeeze(1)

    class AxisEmbeddingField(nn.Module):
        def __init__(self):
            super().__init__()
            dim = 96
            self.x = nn.Embedding(gx, dim)
            self.y = nn.Embedding(gy, dim)
            self.z = nn.Embedding(gz, dim)
            self.net = mlp(dim * 3, 4)

        def forward(self, idx):
            x, y, z = split_indices(idx, gx, gy)
            return self.net(torch.cat([self.x(x), self.y(y), self.z(z)], dim=1)).squeeze(1)

    if variant == "continuous":
        return ContinuousField()
    if variant == "fourier":
        return FourierField()
    if variant == "onehot":
        return OneHotAxisField()
    if variant == "axis_embedding":
        return AxisEmbeddingField()
    raise ValueError(f"Unknown NIF variant {variant!r}")


def make_validation_indices(
    pos_idx: np.ndarray,
    neg_idx: np.ndarray,
    sample_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    half = max(1, sample_count // 2)
    pos = rng.choice(pos_idx, size=min(half, len(pos_idx)), replace=len(pos_idx) < half)
    neg_count = max(1, sample_count - len(pos))
    neg = rng.choice(neg_idx, size=min(neg_count, len(neg_idx)), replace=len(neg_idx) < neg_count)
    out = np.concatenate([pos, neg]).astype(np.int64)
    rng.shuffle(out)
    return out


def evaluate_indices(model: Any, indices: np.ndarray, target_flat: np.ndarray, torch: Any, device: str, batch: int) -> float:
    errors = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), batch):
            idx_np = indices[start : start + batch]
            idx = torch.from_numpy(idx_np).to(device)
            target = torch.from_numpy(target_flat[idx_np].astype(np.float32, copy=False)).to(device)
            pred = model(idx)
            errors.append(((pred - target) ** 2).mean().detach().cpu().item())
    return float(np.mean(errors)) if errors else 0.0


def predict_all(model: Any, cells: int, torch: Any, device: str, batch: int) -> np.ndarray:
    out = np.empty(cells, dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, cells, batch):
            stop = min(cells, start + batch)
            idx = torch.arange(start, stop, dtype=torch.long, device=device)
            out[start:stop] = model(idx).detach().cpu().numpy().astype(np.float32, copy=False)
    return out


def write_model_npz(path: Path, model: Any, metadata: dict[str, Any]) -> int:
    arrays = {name: param.detach().cpu().numpy().astype(np.float32) for name, param in model.state_dict().items()}
    arrays["metadata_json"] = np.array(json.dumps(metadata, separators=(",", ":")), dtype=np.str_)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return path.stat().st_size


def train_variant(
    variant: str,
    target_flat: np.ndarray,
    target_counts: np.ndarray,
    pos_idx: np.ndarray,
    neg_idx: np.ndarray,
    shape: tuple[int, int, int],
    args: argparse.Namespace,
    device: str,
    torch: Any,
    nn: Any,
    F: Any,
) -> tuple[Any, dict[str, Any]]:
    variant_seed = sum((i + 1) * ord(ch) for i, ch in enumerate(variant))
    rng = np.random.default_rng(args.seed + variant_seed)
    model = build_model(variant, shape, torch, nn).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    val_idx = make_validation_indices(pos_idx, neg_idx, args.val_samples, rng)
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    started = time.perf_counter()

    for step in range(1, args.steps + 1):
        pos_n = args.batch // 2
        neg_n = args.batch - pos_n
        pos = rng.choice(pos_idx, size=pos_n, replace=len(pos_idx) < pos_n)
        neg = rng.choice(neg_idx, size=neg_n, replace=len(neg_idx) < neg_n)
        batch_idx = np.concatenate([pos, neg]).astype(np.int64)
        rng.shuffle(batch_idx)

        idx_t = torch.from_numpy(batch_idx).to(device)
        target_t = torch.from_numpy(target_flat[batch_idx].astype(np.float32, copy=False)).to(device)
        pred = model(idx_t)
        err = (pred - target_t) ** 2
        pos_mask = target_t > 0
        pos_loss = err[pos_mask].mean() if pos_mask.any() else err.mean()
        neg_loss = err[~pos_mask].mean() if (~pos_mask).any() else err.mean()
        loss = 0.65 * pos_loss + 0.35 * neg_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            val_loss = evaluate_indices(model, val_idx, target_flat, torch, device, args.eval_batch)
            if val_loss < best_val:
                best_val = val_loss
                best_state = copy.deepcopy(model.state_dict())
            print(
                f"    {variant:14s} step {step:5d}/{args.steps} train={loss.detach().cpu().item():.6f} val={val_loss:.6f}",
                flush=True,
            )

    model.load_state_dict(best_state)
    return model, {
        "variant": variant,
        "param_count": model_param_count(model),
        "best_balanced_val_mse": best_val,
        "train_sec": time.perf_counter() - started,
        "steps": args.steps,
        "batch": args.batch,
    }


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    mass_range = (float(args.mass_min), float(args.mass_max))
    mass_value = float((args.mass_min + args.mass_max) * 0.5)
    coarse_shape = parse_shape(args.coarse_grid_shape)
    fine_shape = parse_shape(args.fine_grid_shape)
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]

    manifest_path = args.out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset = find_dataset(manifest, args.dataset_filter)
    raw_path = Path(dataset["raw_path"])
    ds_dir = args.out_dir / "datasets" / dataset["id"]
    ref = load_reference(dataset, ds_dir)
    range_ref = pp.build_mass_range_reference(raw_path, ref, args, args.mass_min, args.mass_max)
    out_dir = ds_dir / f"range_nif_{args.mass_min:g}_{args.mass_max:g}".replace(".", "p")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading exact {args.mass_min:g}-{args.mass_max:g} Da atoms from {raw_path.name}...", flush=True)
    points_exact = load_range_points(raw_path, args.mass_min, args.mass_max, args.chunk_atoms)
    atom_count = int(len(points_exact))
    display_points = atom_count if args.display_points is None else min(int(args.display_points), atom_count)
    raw_range_bytes = atom_count * 16
    print(f"  found {atom_count:,} atoms ({mb(raw_range_bytes):.2f} MiB raw range)", flush=True)

    print(f"Building coarse grid {shape_label(coarse_shape)} and fine grid {shape_label(fine_shape)}...", flush=True)
    coarse_counts = counts_from_points(points_exact, ref.bounds, coarse_shape)
    fine_counts = counts_from_points(points_exact, ref.bounds, fine_shape)
    coarse_log, coarse_scale = log_target(coarse_counts)
    fine_log, fine_scale = log_target(fine_counts)
    fine_flat = fine_log.reshape(-1).astype(np.float32, copy=False)
    cells = int(fine_flat.size)
    pos_idx = np.flatnonzero(fine_counts.reshape(-1) > 0).astype(np.int64)
    neg_idx = np.flatnonzero(fine_counts.reshape(-1) <= 0).astype(np.int64)
    print(f"  fine cells={cells:,}, occupied={len(pos_idx):,}, empty={len(neg_idx):,}", flush=True)

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
    coarse_metrics = {
        **pp.range_density_metrics(coarse_points, range_ref),
        **grid_loss_metrics(coarse_up_log, fine_log, fine_counts),
    }
    methods["range31_grid_f32"] = make_method(
        label="31-32 Da grid f32 64x64x256",
        raw_size_bytes=raw_range_bytes,
        compressed_size_bytes=grid_f32_size,
        points=len(coarse_points),
        display_artifact=str(coarse_display),
        artifact=str(grid_f32_path),
        metrics=coarse_metrics,
        notes="Dense 32-bit normalized log-count grid; sampled uniformly within occupied density cells for display.",
        settings={"grid_shape": list(coarse_shape), "target_scale": coarse_scale},
    )

    q4 = np.rint(np.clip(fine_log, 0.0, 1.0) * 15.0).astype(np.uint8)
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
    q4_metrics = {
        **pp.range_density_metrics(q4_points, range_ref),
        **grid_loss_metrics(q4_log, fine_log, fine_counts),
    }
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

    torch, nn, F = import_torch()
    device = "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
    print(f"Training NIF variants on {device}...", flush=True)
    nif_summaries: list[dict[str, Any]] = []
    for i, variant in enumerate(variants):
        print(f"  NIF {i + 1}/{len(variants)}: {variant}", flush=True)
        model, train_metrics = train_variant(
            variant,
            fine_flat,
            fine_counts,
            pos_idx,
            neg_idx,
            fine_shape,
            args,
            device,
            torch,
            nn,
            F,
        )
        pred = predict_all(model, cells, torch, device, args.eval_batch).reshape(fine_counts.shape)
        pred_log = np.clip(pred, 0.0, 1.0).astype(np.float32)
        pred_density = np.expm1(pred_log * fine_scale).astype(np.float32)
        nif_points = sample_points_from_density(pred_density, fine_shape, ref.bounds, mass_value, display_points, args.seed + 100 + i)
        display_path = out_dir / f"nif_{variant}_points.npz"
        pp.write_point_pack(display_path, nif_points, ref.bounds, np.array(mass_range, dtype=np.float32))
        model_path = out_dir / f"nif_{variant}.npz"
        metric_values = {
            **train_metrics,
            **pp.range_density_metrics(nif_points, range_ref),
            **grid_loss_metrics(pred_log, fine_log, fine_counts),
        }
        model_size = write_model_npz(
            model_path,
            model,
            {
                "variant": variant,
                "grid_shape": list(fine_shape),
                "mass_range": list(mass_range),
                "target_scale": fine_scale,
                "metrics": pp.json_ready(metric_values),
            },
        )
        metric_values["model_size_bytes"] = model_size
        methods[f"range31_nif_{variant}"] = make_method(
            label=f"31-32 Da NIF {variant.replace('_', ' ')}",
            raw_size_bytes=raw_range_bytes,
            compressed_size_bytes=model_size,
            points=len(nif_points),
            display_artifact=str(display_path),
            artifact=str(model_path),
            metrics=metric_values,
            notes=f"~1M parameter xyz-only neural implicit field trained to predict normalized log count on {shape_label(fine_shape)} cells.",
            settings={
                "variant": variant,
                "grid_shape": list(fine_shape),
                "target_scale": fine_scale,
                "param_count": train_metrics["param_count"],
                "device": device,
            },
        )
        nif_summaries.append({"method": f"range31_nif_{variant}", **metric_values})
        print(
            f"    {variant} params={train_metrics['param_count']:,} "
            f"log_mse={metric_values['log_mse']:.6f} pos_mse={metric_values['positive_log_mse']:.6f} "
            f"count_l1={metric_values['relative_count_l1']:.4f} size={mb(model_size):.2f} MiB",
            flush=True,
        )

    dataset.setdefault("methods", {}).update(methods)
    labels = {
        "range31_full": "31-32 Da full data",
        "range31_grid_f32": "31-32 Da grid f32 64x64x256",
        "range31_grid_q4": "31-32 Da grid 4-bit 128x128x512",
    }
    labels.update({f"range31_nif_{variant}": f"31-32 Da NIF {variant.replace('_', ' ')}" for variant in variants})
    manifest.setdefault("methods", {}).update(labels)
    pp.write_json(ds_dir / "dataset_manifest.json", dataset)
    pp.write_json(manifest_path, manifest)

    summary = {
        "dataset": dataset["name"],
        "mass_range": list(mass_range),
        "atom_count": atom_count,
        "coarse_grid_shape": list(coarse_shape),
        "fine_grid_shape": list(fine_shape),
        "preprocess_sec": time.perf_counter() - started,
        "methods": {key: pp.json_ready(value) for key, value in methods.items()},
    }
    pp.write_json(out_dir / "summary.json", summary)

    print("\n31-32 Da representation summary:")
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
