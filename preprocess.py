from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


METHOD_LABELS = {
    "full": "Full file",
    "jitter": "Random 10% + jitter 10x",
}


@dataclass
class ReferenceStats:
    atom_count: int
    raw_size_bytes: int
    bounds: np.ndarray
    mass_range: np.ndarray
    spectrum_edges: np.ndarray
    spectrum_counts: np.ndarray
    z_counts: np.ndarray
    radial_counts: np.ndarray
    local_counts: np.ndarray
    spatial_bins: int
    metric_mass_bins: int


def now() -> float:
    return time.perf_counter()


def mb(n: int | float) -> float:
    return float(n) / (1024.0 * 1024.0)


def mass_range_label(mass_min: float, mass_max: float) -> str:
    return f"{mass_min:g}-{mass_max:g}"


def mass_range_slug(mass_min: float, mass_max: float) -> str:
    return mass_range_label(mass_min, mass_max).replace("-", "_").replace(".", "p")


def finite_range(lo: float, hi: float, pad: float = 1e-6) -> tuple[float, float]:
    if not np.isfinite(lo) or not np.isfinite(hi):
        return 0.0, 1.0
    if hi <= lo:
        return lo - pad, hi + pad
    return lo, hi


def json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(data), indent=2), encoding="utf-8")


def slug_for_path(path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem).strip("_") or "dataset"
    stem = stem[:80]
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:8]
    return f"{stem}_{digest}"


def find_pos_files(data_dir: Path) -> list[Path]:
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
    files = [p for p in data_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pos"]
    return sorted(files, key=lambda p: p.name.lower())


def open_pos(path: Path) -> tuple[np.memmap, int, int]:
    size = path.stat().st_size
    if size % 16 != 0:
        raise ValueError(f"{path} is {size} bytes, not divisible by 16")
    atom_count = size // 16
    mm = np.memmap(path, dtype=">f4", mode="r", shape=(atom_count, 4))
    return mm, atom_count, size


def iter_chunks(mm: np.memmap, chunk_atoms: int):
    total = len(mm)
    for start in range(0, total, chunk_atoms):
        stop = min(start + chunk_atoms, total)
        yield start, stop, np.asarray(mm[start:stop], dtype=np.float32)


def generate_synthetic_pos(data_dir: Path, atoms: int, seed: int) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / f"synthetic_cone_{atoms}.pos"
    if path.exists() and path.stat().st_size == atoms * 16:
        return path

    rng = np.random.default_rng(seed)
    z_values: list[np.ndarray] = []
    needed = atoms
    while needed > 0:
        candidates = rng.random(max(needed * 2, 10000)).astype(np.float32)
        radius = 3.0 + 26.0 * candidates
        keep = rng.random(len(candidates)) < (radius / 29.0) ** 2
        accepted = candidates[keep]
        z_values.append(accepted[:needed])
        needed -= min(needed, len(accepted))
    zn = np.concatenate(z_values)[:atoms]
    radius = 3.0 + 26.0 * zn
    theta = rng.random(atoms).astype(np.float32) * (2.0 * np.pi)
    disk_r = np.sqrt(rng.random(atoms).astype(np.float32)) * radius
    x = np.cos(theta) * disk_r
    y = np.sin(theta) * disk_r
    z = zn * 180.0

    xn = x / np.maximum(radius, 1e-6)
    yn = y / np.maximum(radius, 1e-6)
    inclusion = np.exp(-(((xn - 0.35) ** 2 + (yn + 0.2) ** 2) / 0.035 + ((zn - 0.62) ** 2) / 0.008))
    gradient = np.clip(0.08 + 0.72 * zn + 0.18 * xn, 0.02, 0.9)

    base = np.array([12.0, 16.0, 27.0, 28.0, 56.0], dtype=np.float32)
    masses = np.empty(atoms, dtype=np.float32)
    r = rng.random(atoms)
    p_inclusion = np.clip(0.02 + 0.42 * inclusion, 0.0, 0.55)
    p_gradient = np.clip(0.04 + 0.34 * gradient, 0.0, 0.5)
    p_rare = 0.012 + 0.025 * (zn > 0.78)

    rare_mask = r < p_rare
    inclusion_mask = (~rare_mask) & (r < p_rare + p_inclusion)
    gradient_mask = (~rare_mask) & (~inclusion_mask) & (r < p_rare + p_inclusion + p_gradient)
    base_mask = ~(rare_mask | inclusion_mask | gradient_mask)

    masses[rare_mask] = 92.0 + rng.normal(0, 0.06, rare_mask.sum())
    masses[inclusion_mask] = 72.0 + rng.normal(0, 0.08, inclusion_mask.sum())
    masses[gradient_mask] = 63.0 + rng.normal(0, 0.09, gradient_mask.sum())
    masses[base_mask] = rng.choice(base, size=base_mask.sum(), p=[0.08, 0.13, 0.34, 0.2, 0.25])
    masses[base_mask] += rng.normal(0, 0.05, base_mask.sum())

    arr = np.column_stack([x, y, z, masses]).astype(">f4")
    arr.tofile(path)
    return path


def compute_reference(path: Path, out_dir: Path, args: argparse.Namespace) -> ReferenceStats:
    mm, atom_count, raw_size = open_pos(path)
    if atom_count == 0:
        raise ValueError(f"{path} contains no records")

    xyz_min = np.full(3, np.inf, dtype=np.float64)
    xyz_max = np.full(3, -np.inf, dtype=np.float64)
    mass_min = np.inf
    mass_max = -np.inf

    for _, _, chunk in iter_chunks(mm, args.chunk_atoms):
        finite = np.isfinite(chunk).all(axis=1)
        if not finite.any():
            continue
        c = chunk[finite]
        xyz_min = np.minimum(xyz_min, c[:, :3].min(axis=0))
        xyz_max = np.maximum(xyz_max, c[:, :3].max(axis=0))
        mass_min = min(mass_min, float(c[:, 3].min()))
        mass_max = max(mass_max, float(c[:, 3].max()))

    for i in range(3):
        xyz_min[i], xyz_max[i] = finite_range(float(xyz_min[i]), float(xyz_max[i]))
    mass_min, mass_max = finite_range(float(mass_min), float(mass_max))
    bounds = np.stack([xyz_min, xyz_max]).astype(np.float32)
    mass_range = np.array([mass_min, mass_max], dtype=np.float32)

    spectrum_edges = np.linspace(mass_min, mass_max, args.spectrum_bins + 1, dtype=np.float32)
    spectrum_counts = np.zeros(args.spectrum_bins, dtype=np.float64)
    z_counts = np.zeros(args.profile_bins, dtype=np.float64)
    radial_counts = np.zeros(args.profile_bins, dtype=np.float64)
    spatial_bins = args.spatial_bins
    metric_mass_bins = args.metric_mass_bins
    local_counts = np.zeros((spatial_bins ** 3, metric_mass_bins), dtype=np.float64)

    center_xy = (bounds[0, :2] + bounds[1, :2]) * 0.5
    max_radius = float(np.linalg.norm(np.maximum(np.abs(bounds[:, :2] - center_xy), 1e-6), axis=1).max())
    extent = np.maximum(bounds[1] - bounds[0], 1e-6)
    mass_extent = max(float(mass_range[1] - mass_range[0]), 1e-6)

    for _, _, chunk in iter_chunks(mm, args.chunk_atoms):
        finite = np.isfinite(chunk).all(axis=1)
        if not finite.any():
            continue
        c = chunk[finite]
        xyz = c[:, :3]
        mass = c[:, 3]
        spectrum_counts += np.histogram(mass, bins=spectrum_edges)[0]

        zn = np.clip((xyz[:, 2] - bounds[0, 2]) / extent[2], 0.0, 0.999999)
        zi = np.floor(zn * args.profile_bins).astype(np.int32)
        z_counts += np.bincount(zi, minlength=args.profile_bins)

        rn = np.linalg.norm(xyz[:, :2] - center_xy, axis=1) / max_radius
        ri = np.floor(np.clip(rn, 0.0, 0.999999) * args.profile_bins).astype(np.int32)
        radial_counts += np.bincount(ri, minlength=args.profile_bins)

        norm = np.clip((xyz - bounds[0]) / extent, 0.0, 0.999999)
        xb = np.floor(norm[:, 0] * spatial_bins).astype(np.int32)
        yb = np.floor(norm[:, 1] * spatial_bins).astype(np.int32)
        zb = np.floor(norm[:, 2] * spatial_bins).astype(np.int32)
        mb = np.floor(np.clip((mass - mass_range[0]) / mass_extent, 0.0, 0.999999) * metric_mass_bins).astype(np.int32)
        spatial_key = (zb * spatial_bins + yb) * spatial_bins + xb
        joint_key = spatial_key * metric_mass_bins + mb
        local_counts += np.bincount(joint_key, minlength=local_counts.size).reshape(local_counts.shape)

    np.savez_compressed(
        out_dir / "reference_stats.npz",
        bounds=bounds,
        mass_range=mass_range,
        spectrum_edges=spectrum_edges,
        spectrum_counts=spectrum_counts,
        z_counts=z_counts,
        radial_counts=radial_counts,
        local_counts=local_counts,
        spatial_bins=np.array([spatial_bins], dtype=np.int32),
        metric_mass_bins=np.array([metric_mass_bins], dtype=np.int32),
    )

    return ReferenceStats(
        atom_count=atom_count,
        raw_size_bytes=raw_size,
        bounds=bounds,
        mass_range=mass_range,
        spectrum_edges=spectrum_edges,
        spectrum_counts=spectrum_counts,
        z_counts=z_counts,
        radial_counts=radial_counts,
        local_counts=local_counts,
        spatial_bins=spatial_bins,
        metric_mass_bins=metric_mass_bins,
    )


def profile_error(a: np.ndarray, b: np.ndarray) -> float:
    sa = float(a.sum())
    sb = float(b.sum())
    if sa <= 0 and sb <= 0:
        return 0.0
    if sa <= 0 or sb <= 0:
        return 1.0
    pa = a.astype(np.float64) / sa
    pb = b.astype(np.float64) / sb
    return float(0.5 * np.abs(pa - pb).sum())


def point_metrics(points: np.ndarray, ref: ReferenceStats) -> dict[str, Any]:
    if len(points) == 0:
        return {
            "mass_spectrum_error": 1.0,
            "z_profile_error": 1.0,
            "radial_profile_error": 1.0,
            "spatial_spectral_error": 1.0,
            "reconstructed_spectrum": [0.0 for _ in range(len(ref.spectrum_counts))],
        }

    bounds = ref.bounds
    mass_range = ref.mass_range
    extent = np.maximum(bounds[1] - bounds[0], 1e-6)
    mass_extent = max(float(mass_range[1] - mass_range[0]), 1e-6)
    xyz = points[:, :3]
    mass = points[:, 3]

    mass_counts = np.histogram(mass, bins=ref.spectrum_edges)[0].astype(np.float64)
    zn = np.clip((xyz[:, 2] - bounds[0, 2]) / extent[2], 0.0, 0.999999)
    zi = np.floor(zn * len(ref.z_counts)).astype(np.int32)
    z_counts = np.bincount(zi, minlength=len(ref.z_counts)).astype(np.float64)

    center_xy = (bounds[0, :2] + bounds[1, :2]) * 0.5
    max_radius = float(np.linalg.norm(np.maximum(np.abs(bounds[:, :2] - center_xy), 1e-6), axis=1).max())
    rn = np.linalg.norm(xyz[:, :2] - center_xy, axis=1) / max_radius
    ri = np.floor(np.clip(rn, 0.0, 0.999999) * len(ref.radial_counts)).astype(np.int32)
    radial_counts = np.bincount(ri, minlength=len(ref.radial_counts)).astype(np.float64)

    s = ref.spatial_bins
    m = ref.metric_mass_bins
    norm = np.clip((xyz - bounds[0]) / extent, 0.0, 0.999999)
    xb = np.floor(norm[:, 0] * s).astype(np.int32)
    yb = np.floor(norm[:, 1] * s).astype(np.int32)
    zb = np.floor(norm[:, 2] * s).astype(np.int32)
    mbin = np.floor(np.clip((mass - mass_range[0]) / mass_extent, 0.0, 0.999999) * m).astype(np.int32)
    spatial_key = (zb * s + yb) * s + xb
    joint_key = spatial_key * m + mbin
    local_counts = np.bincount(joint_key, minlength=ref.local_counts.size).reshape(ref.local_counts.shape).astype(np.float64)

    raw_cell = ref.local_counts.sum(axis=1)
    rec_cell = local_counts.sum(axis=1)
    active = raw_cell > 0
    if active.any():
        raw_dist = np.divide(ref.local_counts[active], raw_cell[active, None], out=np.zeros_like(ref.local_counts[active]), where=raw_cell[active, None] > 0)
        rec_dist = np.divide(local_counts[active], rec_cell[active, None], out=np.zeros_like(local_counts[active]), where=rec_cell[active, None] > 0)
        weights = raw_cell[active] / max(raw_cell[active].sum(), 1.0)
        local_l1 = 0.5 * np.abs(raw_dist - rec_dist).sum(axis=1)
        spatial_spectral_error = float((weights * local_l1).sum())
    else:
        spatial_spectral_error = 1.0

    scale = ref.atom_count / max(len(points), 1)
    return {
        "mass_spectrum_error": profile_error(ref.spectrum_counts, mass_counts),
        "z_profile_error": profile_error(ref.z_counts, z_counts),
        "radial_profile_error": profile_error(ref.radial_counts, radial_counts),
        "spatial_spectral_error": spatial_spectral_error,
        "reconstructed_spectrum": (mass_counts * scale).tolist(),
    }


def quantize_points(points: np.ndarray, bounds: np.ndarray, mass_range: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    extent = np.maximum(bounds[1] - bounds[0], 1e-6)
    mass_extent = max(float(mass_range[1] - mass_range[0]), 1e-6)
    xyz_norm = np.clip((points[:, :3] - bounds[0]) / extent, 0.0, 1.0)
    mass_norm = np.clip((points[:, 3] - mass_range[0]) / mass_extent, 0.0, 1.0)
    xyz_q = np.rint(xyz_norm * 65535.0).astype(np.uint16)
    mass_q = np.rint(mass_norm * 65535.0).astype(np.uint16)
    return xyz_q, mass_q


def write_point_pack(path: Path, points: np.ndarray, bounds: np.ndarray, mass_range: np.ndarray) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    xyz_q, mass_q = quantize_points(points, bounds, mass_range)
    np.savez_compressed(
        path,
        xyz=xyz_q,
        mass=mass_q,
        bounds=bounds.astype(np.float32),
        mass_range=mass_range.astype(np.float32),
        count=np.array([len(points)], dtype=np.int64),
    )
    return path.stat().st_size


def stratum_keys(points: np.ndarray, bounds: np.ndarray, mass_range: np.ndarray, xy_bins: int, z_bins: int, mass_bins: int) -> np.ndarray:
    extent = np.maximum(bounds[1] - bounds[0], 1e-6)
    mass_extent = max(float(mass_range[1] - mass_range[0]), 1e-6)
    norm = np.clip((points[:, :3] - bounds[0]) / extent, 0.0, 0.999999)
    xb = np.floor(norm[:, 0] * xy_bins).astype(np.int32)
    yb = np.floor(norm[:, 1] * xy_bins).astype(np.int32)
    zb = np.floor(norm[:, 2] * z_bins).astype(np.int32)
    mb = np.floor(np.clip((points[:, 3] - mass_range[0]) / mass_extent, 0.0, 0.999999) * mass_bins).astype(np.int32)
    return (((zb * xy_bins + yb) * xy_bins + xb) * mass_bins + mb).astype(np.int32)


def allocate_strata(counts: np.ndarray, target: int, alpha: float = 0.65) -> np.ndarray:
    total = int(counts.sum())
    target = min(target, total)
    alloc = np.zeros_like(counts, dtype=np.int64)
    if target <= 0 or total <= 0:
        return alloc
    if target >= total:
        return counts.astype(np.int64)

    nonzero = counts > 0
    weights = np.zeros_like(counts, dtype=np.float64)
    weights[nonzero] = np.power(counts[nonzero].astype(np.float64), alpha)
    quota = weights / weights.sum() * target
    alloc = np.floor(quota).astype(np.int64)
    alloc = np.minimum(alloc, counts)

    nz_count = int(nonzero.sum())
    if target >= nz_count:
        missing = nonzero & (alloc == 0)
        alloc[missing] = 1
        while int(alloc.sum()) > target:
            reducible = np.where(alloc > 1)[0]
            if len(reducible) == 0:
                break
            i = reducible[np.argmax(alloc[reducible])]
            alloc[i] -= 1

    remaining = target - int(alloc.sum())
    if remaining > 0:
        capacity = counts - alloc
        candidates = np.where(capacity > 0)[0]
        if len(candidates):
            frac = quota[candidates] - np.floor(quota[candidates])
            order = candidates[np.argsort(-frac)]
            for i in order:
                if remaining <= 0:
                    break
                add = min(int(capacity[i]), remaining)
                alloc[i] += add
                remaining -= add
    return alloc


def stratified_sample(path: Path, ref: ReferenceStats, target: int, args: argparse.Namespace, seed: int) -> np.ndarray:
    mm, atom_count, _ = open_pos(path)
    target = min(int(target), atom_count)
    xy_bins = args.lod_xy_bins
    z_bins = args.lod_z_bins
    mass_bins = args.lod_mass_bins
    stratum_count = xy_bins * xy_bins * z_bins * mass_bins

    counts = np.zeros(stratum_count, dtype=np.int64)
    for _, _, chunk in iter_chunks(mm, args.chunk_atoms):
        finite = np.isfinite(chunk).all(axis=1)
        if not finite.any():
            continue
        keys = stratum_keys(chunk[finite], ref.bounds, ref.mass_range, xy_bins, z_bins, mass_bins)
        counts += np.bincount(keys, minlength=stratum_count)

    alloc = allocate_strata(counts, target)
    probs = np.divide(alloc, counts, out=np.zeros_like(alloc, dtype=np.float64), where=counts > 0)
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []

    for _, _, chunk in iter_chunks(mm, args.chunk_atoms):
        finite = np.isfinite(chunk).all(axis=1)
        if not finite.any():
            continue
        c = chunk[finite]
        keys = stratum_keys(c, ref.bounds, ref.mass_range, xy_bins, z_bins, mass_bins)
        keep = rng.random(len(c)) < probs[keys]
        if keep.any():
            selected.append(c[keep])

    if selected:
        points = np.concatenate(selected, axis=0).astype(np.float32, copy=False)
    else:
        points = np.empty((0, 4), dtype=np.float32)

    if len(points) > target:
        idx = rng.choice(len(points), size=target, replace=False)
        points = points[idx]
    return points


def random_sample_exact(path: Path, target: int, chunk_atoms: int, seed: int) -> np.ndarray:
    mm, atom_count, _ = open_pos(path)
    target = min(max(int(target), 0), atom_count)
    if target == 0:
        return np.empty((0, 4), dtype=np.float32)

    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(atom_count, size=target, replace=False))
    selected: list[np.ndarray] = []
    for start in range(0, atom_count, chunk_atoms):
        stop = min(start + chunk_atoms, atom_count)
        lo = np.searchsorted(indices, start, side="left")
        hi = np.searchsorted(indices, stop, side="left")
        if hi <= lo:
            continue
        local = indices[lo:hi] - start
        selected.append(np.asarray(mm[start:stop][local], dtype=np.float32))
        if hi >= len(indices):
            break
    if not selected:
        return np.empty((0, 4), dtype=np.float32)
    return np.concatenate(selected, axis=0).astype(np.float32, copy=False)


def estimate_atom_spacing(path: Path, ref: ReferenceStats, chunk_atoms: int, occupancy_grid: int) -> float:
    grid = max(4, int(occupancy_grid))
    mm, _, _ = open_pos(path)
    bounds = ref.bounds
    extent = np.maximum(bounds[1] - bounds[0], 1e-6)
    occupied = np.zeros(grid * grid * grid, dtype=bool)
    for _, _, chunk in iter_chunks(mm, chunk_atoms):
        finite = np.isfinite(chunk).all(axis=1)
        if not finite.any():
            continue
        xyz = chunk[finite, :3]
        norm = np.clip((xyz - bounds[0]) / extent, 0.0, 0.999999)
        xi = np.floor(norm[:, 0] * grid).astype(np.int32)
        yi = np.floor(norm[:, 1] * grid).astype(np.int32)
        zi = np.floor(norm[:, 2] * grid).astype(np.int32)
        occupied[(zi * grid + yi) * grid + xi] = True
    occupied_cells = max(int(occupied.sum()), 1)
    cell_volume = float(np.prod(extent / float(grid)))
    occupied_volume = occupied_cells * cell_volume
    return float(max((occupied_volume / max(ref.atom_count, 1)) ** (1.0 / 3.0), 1e-6))


def jitter_duplicate_points(
    sample: np.ndarray,
    copies: int,
    target: int,
    radius: float,
    bounds: np.ndarray,
    seed: int,
) -> np.ndarray:
    if len(sample) == 0 or copies <= 0 or target <= 0:
        return np.empty((0, 4), dtype=np.float32)

    total = len(sample) * int(copies)
    target = min(int(target), total)
    rng = np.random.default_rng(seed)
    if target == total:
        base_idx = np.repeat(np.arange(len(sample), dtype=np.int64), int(copies))
    else:
        clone_idx = np.sort(rng.choice(total, size=target, replace=False))
        base_idx = clone_idx // int(copies)

    points = np.empty((target, 4), dtype=np.float32)
    block = 1_000_000
    radius = float(max(radius, 0.0))
    for start in range(0, target, block):
        stop = min(start + block, target)
        base = sample[base_idx[start:stop]]
        offsets = rng.normal(size=(stop - start, 3)).astype(np.float32)
        norm = np.linalg.norm(offsets, axis=1, keepdims=True)
        offsets = offsets / np.maximum(norm, 1e-12)
        shell = np.cbrt(rng.random(stop - start, dtype=np.float32))[:, None]
        xyz = base[:, :3] + offsets * (shell * radius)
        points[start:stop, :3] = np.clip(xyz, bounds[0], bounds[1])
        points[start:stop, 3] = base[:, 3]
    return points


def build_jitter(path: Path, ds_dir: Path, ref: ReferenceStats, args: argparse.Namespace) -> dict[str, Any]:
    started = now()
    reduction = max(1, int(args.jitter_reduction))
    copies = max(1, int(args.jitter_copies))
    sample_target = max(1, int(math.ceil(ref.atom_count / float(reduction))))
    sample = random_sample_exact(path, sample_target, args.chunk_atoms, args.seed + 707)

    spacing = estimate_atom_spacing(path, ref, args.chunk_atoms, args.jitter_occupancy_grid)
    radius = float(args.jitter_radius) if args.jitter_radius is not None and args.jitter_radius > 0 else spacing * float(args.jitter_radius_scale)
    display_target = len(sample) * copies
    if args.jitter_points is not None:
        display_target = min(display_target, int(args.jitter_points))
    points = jitter_duplicate_points(sample, copies, display_target, radius, ref.bounds, args.seed + 808)

    jitter_dir = ds_dir / "jitter"
    jitter_dir.mkdir(parents=True, exist_ok=True)
    sample_path = jitter_dir / "sample_10pct.pos"
    sample.astype(">f4", copy=False).tofile(sample_path)
    meta_path = jitter_dir / "settings.json"
    metadata = {
        "sample_fraction": len(sample) / max(ref.atom_count, 1),
        "reduction": reduction,
        "copies": copies,
        "jitter_radius": radius,
        "estimated_atom_spacing": spacing,
        "display_points": int(len(points)),
    }
    write_json(meta_path, metadata)

    display_path = jitter_dir / "jittered_points.npz"
    display_size = write_point_pack(display_path, points, ref.bounds, ref.mass_range)
    compressed_size = sample_path.stat().st_size + meta_path.stat().st_size
    metrics = point_metrics(points, ref)

    return {
        "label": METHOD_LABELS["jitter"],
        "available": True,
        "compressed_size_bytes": compressed_size,
        "display_cache_size_bytes": display_size,
        "compression_ratio": ref.raw_size_bytes / max(compressed_size, 1),
        "points": int(len(points)),
        "artifact": str(sample_path),
        "display_artifact": str(display_path),
        "preprocess_sec": now() - started,
        "reconstruction_sec": 0.0,
        "metrics": metrics,
        "settings": metadata,
        "notes": f"Uniform random {100.0 / reduction:.1f}% atom sample; each kept atom is jittered into {copies} display copies within radius {radius:.4g}.",
    }


def build_mass_range_reference(
    path: Path,
    ref: ReferenceStats,
    args: argparse.Namespace,
    mass_min: float,
    mass_max: float,
) -> ReferenceStats:
    mm, _, _ = open_pos(path)
    spectrum_edges = np.linspace(mass_min, mass_max, args.spectrum_bins + 1, dtype=np.float32)
    spectrum_counts = np.zeros(args.spectrum_bins, dtype=np.float64)
    z_counts = np.zeros(args.profile_bins, dtype=np.float64)
    radial_counts = np.zeros(args.profile_bins, dtype=np.float64)
    spatial_bins = args.spatial_bins
    local_counts = np.zeros((spatial_bins ** 3, 1), dtype=np.float64)

    bounds = ref.bounds
    extent = np.maximum(bounds[1] - bounds[0], 1e-6)
    center_xy = (bounds[0, :2] + bounds[1, :2]) * 0.5
    max_radius = float(np.linalg.norm(np.maximum(np.abs(bounds[:, :2] - center_xy), 1e-6), axis=1).max())

    atom_count = 0
    for _, _, chunk in iter_chunks(mm, args.chunk_atoms):
        finite = np.isfinite(chunk).all(axis=1)
        if not finite.any():
            continue
        c = chunk[finite]
        keep = (c[:, 3] >= mass_min) & (c[:, 3] <= mass_max)
        if not keep.any():
            continue
        c = c[keep]
        atom_count += len(c)
        xyz = c[:, :3]
        mass = c[:, 3]
        spectrum_counts += np.histogram(mass, bins=spectrum_edges)[0]

        zn = np.clip((xyz[:, 2] - bounds[0, 2]) / extent[2], 0.0, 0.999999)
        zi = np.floor(zn * args.profile_bins).astype(np.int32)
        z_counts += np.bincount(zi, minlength=args.profile_bins)

        rn = np.linalg.norm(xyz[:, :2] - center_xy, axis=1) / max_radius
        ri = np.floor(np.clip(rn, 0.0, 0.999999) * args.profile_bins).astype(np.int32)
        radial_counts += np.bincount(ri, minlength=args.profile_bins)

        norm = np.clip((xyz - bounds[0]) / extent, 0.0, 0.999999)
        xb = np.floor(norm[:, 0] * spatial_bins).astype(np.int32)
        yb = np.floor(norm[:, 1] * spatial_bins).astype(np.int32)
        zb = np.floor(norm[:, 2] * spatial_bins).astype(np.int32)
        spatial_key = (zb * spatial_bins + yb) * spatial_bins + xb
        local_counts[:, 0] += np.bincount(spatial_key, minlength=local_counts.shape[0])

    return ReferenceStats(
        atom_count=atom_count,
        raw_size_bytes=atom_count * 16,
        bounds=ref.bounds,
        mass_range=np.array([mass_min, mass_max], dtype=np.float32),
        spectrum_edges=spectrum_edges,
        spectrum_counts=spectrum_counts,
        z_counts=z_counts,
        radial_counts=radial_counts,
        local_counts=local_counts,
        spatial_bins=spatial_bins,
        metric_mass_bins=1,
    )


def grid_shape3(grid_size: int | tuple[int, int, int] | list[int]) -> tuple[int, int, int]:
    if isinstance(grid_size, (tuple, list)):
        if len(grid_size) != 3:
            raise ValueError("grid shape must have three entries: x,y,z")
        gx, gy, gz = (int(v) for v in grid_size)
    else:
        gx = gy = gz = int(grid_size)
    if gx <= 0 or gy <= 0 or gz <= 0:
        raise ValueError("grid dimensions must be positive")
    return gx, gy, gz


def build_mass_range_density_grid(
    path: Path,
    ref: ReferenceStats,
    grid_size: int | tuple[int, int, int] | list[int],
    mass_min: float,
    mass_max: float,
    chunk_atoms: int,
) -> np.ndarray:
    mm, _, _ = open_pos(path)
    gx, gy, gz = grid_shape3(grid_size)
    counts = np.zeros(gx * gy * gz, dtype=np.float32)
    bounds = ref.bounds
    extent = np.maximum(bounds[1] - bounds[0], 1e-6)

    for _, _, chunk in iter_chunks(mm, chunk_atoms):
        finite = np.isfinite(chunk).all(axis=1)
        if not finite.any():
            continue
        c = chunk[finite]
        keep = (c[:, 3] >= mass_min) & (c[:, 3] <= mass_max)
        if not keep.any():
            continue
        xyz = c[keep, :3]
        norm = np.clip((xyz - bounds[0]) / extent, 0.0, 0.999999)
        xi = np.floor(norm[:, 0] * gx).astype(np.int32)
        yi = np.floor(norm[:, 1] * gy).astype(np.int32)
        zi = np.floor(norm[:, 2] * gz).astype(np.int32)
        flat = ((zi * gy + yi) * gx + xi).astype(np.int64)
        counts += np.bincount(flat, minlength=counts.size).astype(np.float32)
    return counts.reshape((gz, gy, gx))


def sample_from_density_weights(
    weights: np.ndarray,
    grid: int | tuple[int, int, int] | list[int],
    bounds: np.ndarray,
    mass_value: float,
    target: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    flat = np.clip(weights.reshape(-1).astype(np.float64), 0.0, None)
    total = float(flat.sum())
    if total <= 0 or target <= 0:
        return np.empty((0, 4), dtype=np.float32)
    if weights.ndim == 3:
        gz, gy, gx = (int(v) for v in weights.shape)
    else:
        gx, gy, gz = grid_shape3(grid)
    probs = flat / total
    idx = rng.choice(len(flat), size=int(target), replace=True, p=probs)
    zi = idx // (gx * gy)
    rem = idx % (gx * gy)
    yi = rem // gx
    xi = rem % gx
    jitter = rng.random((int(target), 3), dtype=np.float32)
    norm = np.column_stack([xi, yi, zi]).astype(np.float32)
    norm = (norm + jitter) / np.array([gx, gy, gz], dtype=np.float32)
    xyz = bounds[0] + norm * (bounds[1] - bounds[0])
    mass = np.full(int(target), float(mass_value), dtype=np.float32)
    return np.column_stack([xyz, mass]).astype(np.float32)


def support_mask_for_eval_grid(target_grid: np.ndarray, eval_grid: int) -> np.ndarray:
    train_grid = target_grid.shape[0]
    support = target_grid.reshape(-1) > 0
    eval_idx = np.arange(eval_grid * eval_grid * eval_grid, dtype=np.int64)
    ezi = eval_idx // (eval_grid * eval_grid)
    rem = eval_idx % (eval_grid * eval_grid)
    eyi = rem // eval_grid
    exi = rem % eval_grid
    txi = np.floor(((exi + 0.5) / float(eval_grid)) * train_grid).astype(np.int32).clip(0, train_grid - 1)
    tyi = np.floor(((eyi + 0.5) / float(eval_grid)) * train_grid).astype(np.int32).clip(0, train_grid - 1)
    tzi = np.floor(((ezi + 0.5) / float(eval_grid)) * train_grid).astype(np.int32).clip(0, train_grid - 1)
    return support[(tzi * train_grid + tyi) * train_grid + txi]


def range_density_metrics(points: np.ndarray, ref: ReferenceStats) -> dict[str, Any]:
    if len(points) == 0 or ref.atom_count == 0:
        return {
            "mass_spectrum_error": None,
            "z_profile_error": 1.0,
            "radial_profile_error": 1.0,
            "spatial_spectral_error": 1.0,
            "target_atom_count": ref.atom_count,
        }

    bounds = ref.bounds
    extent = np.maximum(bounds[1] - bounds[0], 1e-6)
    xyz = points[:, :3]
    zn = np.clip((xyz[:, 2] - bounds[0, 2]) / extent[2], 0.0, 0.999999)
    zi = np.floor(zn * len(ref.z_counts)).astype(np.int32)
    z_counts = np.bincount(zi, minlength=len(ref.z_counts)).astype(np.float64)

    center_xy = (bounds[0, :2] + bounds[1, :2]) * 0.5
    max_radius = float(np.linalg.norm(np.maximum(np.abs(bounds[:, :2] - center_xy), 1e-6), axis=1).max())
    rn = np.linalg.norm(xyz[:, :2] - center_xy, axis=1) / max_radius
    ri = np.floor(np.clip(rn, 0.0, 0.999999) * len(ref.radial_counts)).astype(np.int32)
    radial_counts = np.bincount(ri, minlength=len(ref.radial_counts)).astype(np.float64)

    s = ref.spatial_bins
    norm = np.clip((xyz - bounds[0]) / extent, 0.0, 0.999999)
    xb = np.floor(norm[:, 0] * s).astype(np.int32)
    yb = np.floor(norm[:, 1] * s).astype(np.int32)
    zb = np.floor(norm[:, 2] * s).astype(np.int32)
    spatial_key = (zb * s + yb) * s + xb
    spatial_counts = np.bincount(spatial_key, minlength=s ** 3).astype(np.float64)

    return {
        "mass_spectrum_error": None,
        "z_profile_error": profile_error(ref.z_counts, z_counts),
        "radial_profile_error": profile_error(ref.radial_counts, radial_counts),
        "spatial_spectral_error": profile_error(ref.local_counts[:, 0], spatial_counts),
        "target_atom_count": ref.atom_count,
    }


def build_range_density_variant(
    path: Path,
    ds_dir: Path,
    ref: ReferenceStats,
    range_ref: ReferenceStats,
    target_grid: np.ndarray,
    support_grid: np.ndarray,
    args: argparse.Namespace,
    spec: dict[str, Any],
) -> dict[str, Any]:
    started = now()
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except Exception as exc:
        return {
            "id": spec["id"],
            "label": spec["label"],
            "available": False,
            "compressed_size_bytes": 0,
            "compression_ratio": 0.0,
            "points": 0,
            "preprocess_sec": now() - started,
            "metrics": None,
            "notes": f"PyTorch is not available: {exc}",
        }

    try:
        device = "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
        train_grid = target_grid.shape[0]
        counts = target_grid.reshape(-1).astype(np.float32)
        target_scale = max(float(np.log1p(counts.max())), 1.0)
        positive_idx = np.flatnonzero(counts > 0)
        if len(positive_idx) == 0:
            raise ValueError(f"No atoms found in mass range {args.range_density_min:g}-{args.range_density_max:g}.")
        positive_prob = np.log1p(counts[positive_idx]).astype(np.float64)
        positive_prob /= positive_prob.sum()

        cell_indices = np.arange(counts.size, dtype=np.int64)
        zi = cell_indices // (train_grid * train_grid)
        rem = cell_indices % (train_grid * train_grid)
        yi = rem // train_grid
        xi = rem % train_grid
        xyz_centers = np.column_stack([xi, yi, zi]).astype(np.float32)
        xyz_centers = ((xyz_centers + 0.5) / float(train_grid)) * 2.0 - 1.0
        xyz_centers_t = torch.from_numpy(xyz_centers).to(device)

        class XYZDensityField(nn.Module):
            def __init__(self, levels: list[int], channels: int, widths: list[int]):
                super().__init__()
                self.levels = levels
                self.grids = nn.ParameterList(
                    [nn.Parameter(torch.randn(1, channels, res, res, res) * 0.01) for res in levels]
                )
                self.register_buffer("freq", torch.tensor([1.0, 2.0, 4.0, 8.0], dtype=torch.float32), persistent=False)
                in_dim = len(levels) * channels + 3 + 3 * 2 * len(self.freq)
                layers = []
                last = in_dim
                for width in widths:
                    layers.append(nn.Linear(last, width))
                    layers.append(nn.SiLU())
                    last = width
                layers.append(nn.Linear(last, 1))
                self.net = nn.Sequential(*layers)

            def sample_grid(self, grid_param: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
                res = grid_param.shape[-1]
                channels = grid_param.shape[1]
                pos = (xyz + 1.0) * (0.5 * float(res - 1))
                x = pos[:, 0].clamp(0.0, float(res - 1))
                y = pos[:, 1].clamp(0.0, float(res - 1))
                z = pos[:, 2].clamp(0.0, float(res - 1))
                x0 = torch.floor(x).to(torch.long).clamp(0, res - 1)
                y0 = torch.floor(y).to(torch.long).clamp(0, res - 1)
                z0 = torch.floor(z).to(torch.long).clamp(0, res - 1)
                x1 = (x0 + 1).clamp(0, res - 1)
                y1 = (y0 + 1).clamp(0, res - 1)
                z1 = (z0 + 1).clamp(0, res - 1)
                wx = (x - x0.to(x.dtype))[:, None]
                wy = (y - y0.to(y.dtype))[:, None]
                wz = (z - z0.to(z.dtype))[:, None]
                flat = grid_param[0].permute(1, 2, 3, 0).reshape(-1, channels)

                def gather(ix, iy, iz):
                    return flat[(iz * res + iy) * res + ix]

                c000 = gather(x0, y0, z0)
                c100 = gather(x1, y0, z0)
                c010 = gather(x0, y1, z0)
                c110 = gather(x1, y1, z0)
                c001 = gather(x0, y0, z1)
                c101 = gather(x1, y0, z1)
                c011 = gather(x0, y1, z1)
                c111 = gather(x1, y1, z1)
                c00 = c000 * (1.0 - wx) + c100 * wx
                c10 = c010 * (1.0 - wx) + c110 * wx
                c01 = c001 * (1.0 - wx) + c101 * wx
                c11 = c011 * (1.0 - wx) + c111 * wx
                c0 = c00 * (1.0 - wy) + c10 * wy
                c1 = c01 * (1.0 - wy) + c11 * wy
                return c0 * (1.0 - wz) + c1 * wz

            def encode_xyz(self, xyz: torch.Tensor) -> torch.Tensor:
                freq = self.freq.to(xyz.device)
                xb = xyz[..., None] * freq
                return torch.cat([xyz, torch.sin(math.pi * xb).flatten(1), torch.cos(math.pi * xb).flatten(1)], dim=1)

            def forward(self, xyz: torch.Tensor) -> torch.Tensor:
                features = [self.sample_grid(grid_param, xyz) for grid_param in self.grids]
                features.append(self.encode_xyz(xyz))
                return self.net(torch.cat(features, dim=1)).squeeze(1)

        levels = [int(x) for x in spec["levels"]]
        channels = int(spec["channels"])
        widths = [int(x) for x in spec["widths"]]
        model = XYZDensityField(levels, channels, widths).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=float(spec["lr"]), weight_decay=1e-6)
        rng = np.random.default_rng(args.seed + int(spec["seed_offset"]))
        batch = int(spec["batch"])
        pos_fraction = float(spec["positive_fraction"])

        for _ in range(int(spec["epochs"])):
            pos_count = min(len(positive_idx), max(1, int(batch * pos_fraction)))
            neg_count = max(0, batch - pos_count)
            pos = rng.choice(positive_idx, size=pos_count, replace=len(positive_idx) < pos_count, p=positive_prob)
            neg_parts = []
            while sum(len(x) for x in neg_parts) < neg_count:
                candidate = rng.integers(0, len(counts), size=max(neg_count * 2, 1024), dtype=np.int64)
                candidate = candidate[counts[candidate] == 0]
                if len(candidate):
                    neg_parts.append(candidate[: max(0, neg_count - sum(len(x) for x in neg_parts))])
            idx_np = np.concatenate([pos, *neg_parts]) if neg_parts else pos
            rng.shuffle(idx_np)
            idx_t = torch.from_numpy(idx_np).to(device)
            target_np = (np.log1p(counts[idx_np]) / target_scale).astype(np.float32)
            target_t = torch.from_numpy(target_np).to(device)
            pred = F.softplus(model(xyz_centers_t[idx_t]))
            weight = torch.where(target_t > 0, torch.tensor(3.0, device=device), torch.tensor(0.25, device=device))
            loss = (weight * (pred - target_t).pow(2)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

        range_label = spec.get("range_label", mass_range_label(args.range_density_min, args.range_density_max))
        range_slug = spec.get("range_slug", mass_range_slug(args.range_density_min, args.range_density_max))
        subdir = ds_dir / f"range_density_{range_slug}" / spec["id"]
        subdir.mkdir(parents=True, exist_ok=True)
        model_path = subdir / "model.pt"
        support_path = subdir / "support_mask.npz"
        mass_value = (float(args.range_density_min) + float(args.range_density_max)) * 0.5
        support_train = support_grid.reshape(-1) > 0
        np.savez_compressed(
            support_path,
            bits=np.packbits(support_train.astype(np.uint8)),
            shape=np.array(support_grid.shape, dtype=np.int32),
        )
        torch.save(
            {
                "state_dict": model.state_dict(),
                "bounds": ref.bounds,
                "mass_range": np.array([args.range_density_min, args.range_density_max], dtype=np.float32),
                "display_mass": np.array([mass_value], dtype=np.float32),
                "settings": {
                    "architecture": "xyz_to_density_multires_grid",
                    "train_grid": train_grid,
                    "eval_grid": args.range_density_eval_grid,
                    "support_grid": support_grid.shape[0],
                    "levels": levels,
                    "channels": channels,
                    "widths": widths,
                    "epochs": spec["epochs"],
                    "device": device,
                    "target_scale": target_scale,
                    "target_atom_count": range_ref.atom_count,
                    "support_mask": str(support_path),
                },
            },
            model_path,
        )
        compressed_size = model_path.stat().st_size + support_path.stat().st_size

        eval_grid = int(args.range_density_eval_grid)
        eval_cells = eval_grid * eval_grid * eval_grid
        eval_idx = np.arange(eval_cells, dtype=np.int64)
        ezi = eval_idx // (eval_grid * eval_grid)
        erem = eval_idx % (eval_grid * eval_grid)
        eyi = erem // eval_grid
        exi = erem % eval_grid
        eval_xyz = np.column_stack([exi, eyi, ezi]).astype(np.float32)
        eval_xyz = ((eval_xyz + 0.5) / float(eval_grid)) * 2.0 - 1.0
        weights = np.empty(eval_cells, dtype=np.float32)
        eval_batch = int(args.range_density_eval_batch)
        with torch.no_grad():
            for start in range(0, eval_cells, eval_batch):
                xyz_t = torch.from_numpy(eval_xyz[start : start + eval_batch]).to(device)
                pred = F.softplus(model(xyz_t)).cpu().numpy()
                weights[start : start + len(pred)] = np.expm1(np.clip(pred, 0.0, None) * target_scale).astype(np.float32)

        eval_support = support_mask_for_eval_grid(support_grid, eval_grid)
        weights[~eval_support] = 0.0

        point_target = range_ref.atom_count if args.range_density_points is None else min(int(args.range_density_points), range_ref.atom_count)
        points = sample_from_density_weights(
            weights.reshape((eval_grid, eval_grid, eval_grid)),
            eval_grid,
            ref.bounds,
            mass_value,
            point_target,
            args.seed + int(spec["seed_offset"]) + 1000,
        )
        display_path = subdir / "reconstruction_points.npz"
        display_size = write_point_pack(display_path, points, ref.bounds, ref.mass_range)
        metrics = range_density_metrics(points, range_ref)
        label = f"{spec['label']} ({mb(compressed_size):.2f} MB)"
        return {
            "id": spec["id"],
            "label": label,
            "available": True,
            "compressed_size_bytes": compressed_size,
            "display_cache_size_bytes": display_size,
            "compression_ratio": ref.raw_size_bytes / max(compressed_size, 1),
            "points": int(len(points)),
            "artifact": str(model_path),
            "support_artifact": str(support_path),
            "display_artifact": str(display_path),
            "preprocess_sec": now() - started,
            "reconstruction_sec": 0.0,
            "metrics": metrics,
            "settings": {
                "architecture": "xyz_to_density_multires_grid",
                "mass_range": [args.range_density_min, args.range_density_max],
                "display_mass": mass_value,
                "train_grid": train_grid,
                "eval_grid": eval_grid,
                "support_grid": support_grid.shape[0],
                "levels": levels,
                "channels": channels,
                "widths": widths,
                "epochs": spec["epochs"],
                "device": device,
                "target_atom_count": range_ref.atom_count,
                "support_mask": "raw filtered occupied cells",
            },
            "notes": f"Single density field for mass {range_label}; xyz-only density is sampled only inside the raw filtered occupied support.",
        }
    except Exception as exc:
        range_slug = spec.get("range_slug", mass_range_slug(args.range_density_min, args.range_density_max))
        write_json(
            ds_dir / f"range_density_{range_slug}" / spec["id"] / "unavailable.json",
            {"error": str(exc), "traceback": traceback.format_exc()},
        )
        return {
            "id": spec["id"],
            "label": spec["label"],
            "available": False,
            "compressed_size_bytes": 0,
            "compression_ratio": 0.0,
            "points": 0,
            "preprocess_sec": now() - started,
            "metrics": None,
            "notes": f"Range density preprocessing failed: {exc}",
        }


def build_range_density(path: Path, ds_dir: Path, ref: ReferenceStats, args: argparse.Namespace) -> dict[str, Any]:
    started = now()
    mass_min = float(args.range_density_min)
    mass_max = float(args.range_density_max)
    range_label = mass_range_label(mass_min, mass_max)
    range_slug = mass_range_slug(mass_min, mass_max)
    range_ref = build_mass_range_reference(path, ref, args, mass_min, mass_max)
    target_grids: dict[int, np.ndarray] = {}
    support_grids: dict[int, np.ndarray] = {}

    def get_target_grid(size: int) -> np.ndarray:
        if size not in target_grids:
            target_grids[size] = build_mass_range_density_grid(path, ref, size, mass_min, mass_max, args.chunk_atoms)
        return target_grids[size]

    def get_support_grid(size: int) -> np.ndarray:
        if size not in support_grids:
            support_grids[size] = build_mass_range_density_grid(path, ref, size, mass_min, mass_max, args.chunk_atoms)
        return support_grids[size]
    specs = [
        {
            "id": "target_500kb",
            "label": f"{range_label} density 500 KB",
            "range_label": range_label,
            "range_slug": range_slug,
            "levels": [16, 32],
            "channels": 3,
            "widths": [64, 64],
            "epochs": 1000,
            "batch": 65536,
            "lr": 2e-3,
            "positive_fraction": 0.7,
            "seed_offset": 1700,
        },
        {
            "id": "target_1mb",
            "label": f"{range_label} density 1 MB",
            "range_label": range_label,
            "range_slug": range_slug,
            "levels": [16, 32],
            "channels": 6,
            "widths": [96, 96],
            "epochs": 1200,
            "batch": 65536,
            "lr": 2e-3,
            "positive_fraction": 0.7,
            "seed_offset": 1800,
        },
        {
            "id": "target_10mb",
            "label": f"{range_label} density 10 MB",
            "range_label": range_label,
            "range_slug": range_slug,
            "train_grid": 128,
            "levels": [16, 32, 64],
            "channels": 8,
            "widths": [128, 128],
            "epochs": 2400,
            "batch": 65536,
            "lr": 1.5e-3,
            "positive_fraction": 0.7,
            "seed_offset": 1917,
            "support_grid": 128,
        },
    ]
    variants = []
    for spec in specs:
        print(f"  range density {spec['label']}...", flush=True)
        target_grid = get_target_grid(int(spec.get("train_grid", args.range_density_grid)))
        support_grid = get_support_grid(int(spec.get("support_grid", args.range_density_support_grid)))
        variant = build_range_density_variant(path, ds_dir, ref, range_ref, target_grid, support_grid, args, spec)
        variants.append(variant)
        if variant.get("available"):
            print(
                f"    complete: {mb(variant.get('compressed_size_bytes', 0)):.2f} MB, "
                f"{variant.get('compression_ratio', 0.0):.2f}x",
                flush=True,
            )
        else:
            print(f"    unavailable: {variant.get('notes', '')}", flush=True)
    available = [variant for variant in variants if variant.get("available")]
    default = available[1] if len(available) > 1 else (available[0] if available else variants[0])
    method = {
        **default,
        "label": f"{range_label} density field",
        "method_label": f"{range_label} density field",
        "variants": variants,
        "preprocess_sec": now() - started,
        "notes": f"Range-specific xyz-to-density fields for mass {range_label}; display points are tagged at the range midpoint for filtering.",
    }
    return method


def build_lod(path: Path, ds_dir: Path, ref: ReferenceStats, args: argparse.Namespace) -> dict[str, Any]:
    started = now()
    levels: list[dict[str, Any]] = []
    total_size = 0
    level_targets = [min(int(x), ref.atom_count) for x in args.lod_levels if int(x) > 0]
    level_targets = sorted(set(level_targets))
    if not level_targets:
        level_targets = [min(ref.atom_count, 250000)]

    for i, target in enumerate(level_targets):
        level_started = now()
        points = stratified_sample(path, ref, target, args, seed=args.seed + i * 17)
        level_id = f"lod_{len(points)}"
        pack_path = ds_dir / "lod" / f"{level_id}.npz"
        size = write_point_pack(pack_path, points, ref.bounds, ref.mass_range)
        total_size += size
        metrics = point_metrics(points, ref)
        levels.append(
            {
                "id": level_id,
                "label": f"{len(points):,} atoms",
                "target_points": target,
                "points": int(len(points)),
                "artifact": str(pack_path),
                "compressed_size_bytes": size,
                "compression_ratio": ref.raw_size_bytes / max(size, 1),
                "preprocess_sec": now() - level_started,
                "metrics": metrics,
            }
        )

    return {
        "label": METHOD_LABELS["lod"],
        "available": True,
        "compressed_size_bytes": total_size,
        "compression_ratio": ref.raw_size_bytes / max(total_size, 1),
        "preprocess_sec": now() - started,
        "levels": levels,
        "notes": "Stratified representative samples over x/y/z and mass bins.",
    }


def build_spectral_grid(path: Path, ref: ReferenceStats, grid_size: int, mass_bins: int, chunk_atoms: int) -> tuple[np.ndarray, np.ndarray]:
    mm, _, _ = open_pos(path)
    total_cells = mass_bins * grid_size * grid_size * grid_size
    counts = np.zeros(total_cells, dtype=np.float32)
    bounds = ref.bounds
    mass_range = ref.mass_range
    extent = np.maximum(bounds[1] - bounds[0], 1e-6)
    mass_extent = max(float(mass_range[1] - mass_range[0]), 1e-6)
    mass_edges = np.linspace(float(mass_range[0]), float(mass_range[1]), mass_bins + 1, dtype=np.float32)

    for _, _, chunk in iter_chunks(mm, chunk_atoms):
        finite = np.isfinite(chunk).all(axis=1)
        if not finite.any():
            continue
        c = chunk[finite]
        xyz = c[:, :3]
        mass = c[:, 3]
        norm = np.clip((xyz - bounds[0]) / extent, 0.0, 0.999999)
        xi = np.floor(norm[:, 0] * grid_size).astype(np.int32)
        yi = np.floor(norm[:, 1] * grid_size).astype(np.int32)
        zi = np.floor(norm[:, 2] * grid_size).astype(np.int32)
        mi = np.floor(np.clip((mass - mass_range[0]) / mass_extent, 0.0, 0.999999) * mass_bins).astype(np.int32)
        flat = (((mi * grid_size + zi) * grid_size + yi) * grid_size + xi).astype(np.int64)
        counts += np.bincount(flat, minlength=total_cells).astype(np.float32)

    return counts.reshape((mass_bins, grid_size, grid_size, grid_size)), mass_edges


def require_power_of_two(value: int, name: str) -> None:
    if value < 2 or value & (value - 1):
        raise ValueError(f"{name} must be a power of two, got {value}")


def haar_forward_3d(fields: np.ndarray) -> np.ndarray:
    out = fields.astype(np.float32, copy=True)
    n = out.shape[1]
    while n > 1:
        for axis in (1, 2, 3):
            sl = [slice(None)] * out.ndim
            sl[axis] = slice(0, n)
            block = out[tuple(sl)]
            moved = np.moveaxis(block, axis, -1)
            even = moved[..., 0:n:2].copy()
            odd = moved[..., 1:n:2].copy()
            moved[..., : n // 2] = (even + odd) * 0.5
            moved[..., n // 2 : n] = (even - odd) * 0.5
        n //= 2
    return out


def haar_inverse_3d(coeffs: np.ndarray) -> np.ndarray:
    out = coeffs.astype(np.float32, copy=True)
    grid = out.shape[1]
    n = 1
    while n < grid:
        n *= 2
        half = n // 2
        for axis in (3, 2, 1):
            sl = [slice(None)] * out.ndim
            sl[axis] = slice(0, n)
            block = out[tuple(sl)]
            moved = np.moveaxis(block, axis, -1)
            avg = moved[..., :half].copy()
            diff = moved[..., half:n].copy()
            moved[..., 0:n:2] = avg + diff
            moved[..., 1:n:2] = avg - diff
    return out


def select_wavelet_coefficients(coeffs: np.ndarray, budget: int, reserve_per_mass_bin: int) -> np.ndarray:
    """Keep a global top-k while reserving a few coefficients for each active mass bin."""
    mass_bins = coeffs.shape[0]
    cells_per_mass = int(np.prod(coeffs.shape[1:]))
    flat_by_mass = coeffs.reshape(mass_bins, cells_per_mass)
    active = np.any(flat_by_mass != 0.0, axis=1)
    active_count = int(active.sum())
    selected_parts: list[np.ndarray] = []

    if reserve_per_mass_bin > 0 and active_count > 0:
        reserve = min(reserve_per_mass_bin, cells_per_mass, max(1, budget // active_count))
        for mass_idx in np.flatnonzero(active):
            values = np.abs(flat_by_mass[mass_idx])
            if reserve >= cells_per_mass:
                local = np.arange(cells_per_mass, dtype=np.uint32)
            else:
                local = np.argpartition(values, -reserve)[-reserve:].astype(np.uint32)
            selected_parts.append(local + np.uint32(mass_idx * cells_per_mass))

    if selected_parts:
        selected = np.unique(np.concatenate(selected_parts).astype(np.uint32))
    else:
        selected = np.empty(0, dtype=np.uint32)

    remaining = max(0, budget - len(selected))
    if remaining > 0:
        scores = np.abs(coeffs.reshape(-1))
        if len(selected):
            scores = scores.copy()
            scores[selected] = -1.0
        remaining = min(remaining, len(scores) - len(selected))
        if remaining > 0:
            global_idx = np.argpartition(scores, -remaining)[-remaining:].astype(np.uint32)
            selected = np.unique(np.concatenate([selected, global_idx]))

    if len(selected) > budget:
        scores = np.abs(coeffs.reshape(-1)[selected])
        selected = selected[np.argpartition(scores, -budget)[-budget:]]
    return selected.astype(np.uint32, copy=False)


def sample_from_spectral_field(fields: np.ndarray, bounds: np.ndarray, mass_edges: np.ndarray, target: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    weights = np.clip(fields.reshape(-1).astype(np.float64), 0.0, None)
    total = float(weights.sum())
    if total <= 0 or target <= 0:
        return np.empty((0, 4), dtype=np.float32)
    probs = weights / total
    idx = rng.choice(len(weights), size=target, replace=True, p=probs)
    grid = fields.shape[1]
    cells_per_mass = grid * grid * grid
    mi = idx // cells_per_mass
    rem = idx % cells_per_mass
    zi = rem // (grid * grid)
    rem = rem % (grid * grid)
    yi = rem // grid
    xi = rem % grid
    jitter = rng.random((target, 3), dtype=np.float32)
    norm = np.column_stack([xi, yi, zi]).astype(np.float32)
    norm = (norm + jitter) / float(grid)
    xyz = bounds[0] + norm * (bounds[1] - bounds[0])
    mass_lo = mass_edges[mi]
    mass_hi = mass_edges[mi + 1]
    mass = mass_lo + rng.random(target, dtype=np.float32) * (mass_hi - mass_lo)
    return np.column_stack([xyz, mass]).astype(np.float32)


def build_wavelet(
    path: Path,
    ds_dir: Path,
    ref: ReferenceStats,
    args: argparse.Namespace,
    variant_id: str = "default",
    variant_label: str | None = None,
    subdir: str = "wavelet",
) -> dict[str, Any]:
    started = now()
    require_power_of_two(args.wavelet_grid, "wavelet grid")
    fields, mass_edges = build_spectral_grid(path, ref, args.wavelet_grid, args.wavelet_mass_bins, args.chunk_atoms)
    coeffs = haar_forward_3d(fields)
    flat = coeffs.reshape(-1)
    budget = min(int(args.wavelet_coefficients), len(flat))
    if budget <= 0:
        raise ValueError("wavelet coefficient budget must be positive")
    top_idx = select_wavelet_coefficients(coeffs, budget, args.wavelet_reserve_per_mass_bin)
    values = flat[top_idx]
    max_abs = float(np.max(np.abs(values))) if len(values) else 0.0
    scale = max(max_abs / 32767.0, 1e-12)
    q = np.rint(values / scale).clip(-32767, 32767).astype(np.int16)

    coeff_path = ds_dir / subdir / "coefficients.npz"
    coeff_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        coeff_path,
        idx=top_idx,
        q=q,
        scale=np.array([scale], dtype=np.float32),
        shape=np.array(coeffs.shape, dtype=np.int32),
        bounds=ref.bounds.astype(np.float32),
        mass_edges=mass_edges.astype(np.float32),
        total_atoms=np.array([ref.atom_count], dtype=np.int64),
    )
    compressed_size = coeff_path.stat().st_size

    rec_flat = np.zeros_like(flat, dtype=np.float32)
    rec_flat[top_idx] = q.astype(np.float32) * scale
    rec_fields = haar_inverse_3d(rec_flat.reshape(coeffs.shape))
    rec_fields = np.clip(rec_fields, 0.0, None)
    point_budget = min(int(args.wavelet_points), ref.atom_count)
    points = sample_from_spectral_field(rec_fields, ref.bounds, mass_edges, point_budget, seed=args.seed + 101)
    display_path = ds_dir / subdir / "reconstruction_points.npz"
    display_size = write_point_pack(display_path, points, ref.bounds, ref.mass_range)
    metrics = point_metrics(points, ref)

    return {
        "id": variant_id,
        "label": variant_label or METHOD_LABELS["wavelet"],
        "available": True,
        "compressed_size_bytes": compressed_size,
        "display_cache_size_bytes": display_size,
        "compression_ratio": ref.raw_size_bytes / max(compressed_size, 1),
        "points": int(len(points)),
        "artifact": str(coeff_path),
        "display_artifact": str(display_path),
        "preprocess_sec": now() - started,
        "reconstruction_sec": 0.0,
        "metrics": metrics,
        "settings": {
            "grid": args.wavelet_grid,
            "mass_bins": args.wavelet_mass_bins,
            "coefficients": budget,
            "reserve_per_mass_bin": args.wavelet_reserve_per_mass_bin,
            "display_points": point_budget,
        },
        "notes": "3D Haar coefficients per high-resolution spectral mass bin, quantized with a per-bin coefficient reserve.",
    }


def parse_widths(value: Any) -> list[int]:
    if value is None:
        return [128, 128, 64]
    if isinstance(value, str):
        return [int(x.strip()) for x in value.split(",") if x.strip()]
    return [int(x) for x in value]


def build_neural(
    path: Path,
    ds_dir: Path,
    ref: ReferenceStats,
    args: argparse.Namespace,
    variant_id: str = "default",
    variant_label: str | None = None,
    subdir: str = "neural",
) -> dict[str, Any]:
    started = now()
    if args.skip_neural:
        return {
            "label": METHOD_LABELS["neural"],
            "available": False,
            "compressed_size_bytes": 0,
            "compression_ratio": 0.0,
            "points": 0,
            "preprocess_sec": now() - started,
            "metrics": None,
            "notes": "Skipped by --skip-neural.",
        }

    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except Exception as exc:
        return {
            "label": METHOD_LABELS["neural"],
            "available": False,
            "compressed_size_bytes": 0,
            "compression_ratio": 0.0,
            "points": 0,
            "preprocess_sec": now() - started,
            "metrics": None,
            "notes": f"PyTorch is not available: {exc}",
        }

    try:
        grid = args.neural_grid
        mass_bins = args.neural_mass_bins
        widths = parse_widths(getattr(args, "neural_widths", None))
        fields, mass_edges = build_spectral_grid(path, ref, grid, mass_bins, args.chunk_atoms)
        density = fields.sum(axis=0).astype(np.float32)
        mass_prob = np.divide(fields, np.maximum(density[None, :, :, :], 1.0), out=np.zeros_like(fields), where=density[None, :, :, :] > 0)

        coords = np.stack(np.meshgrid(
            np.linspace(-1.0, 1.0, grid, dtype=np.float32),
            np.linspace(-1.0, 1.0, grid, dtype=np.float32),
            np.linspace(-1.0, 1.0, grid, dtype=np.float32),
            indexing="ij",
        ), axis=-1).reshape(-1, 3)
        y_density = np.log1p(density.reshape(-1))
        if y_density.max() > 0:
            y_density = y_density / y_density.max()
        y_mass = np.moveaxis(mass_prob, 0, -1).reshape(-1, mass_bins)

        device = "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
        x_t = torch.from_numpy(coords).to(device)
        d_t = torch.from_numpy(y_density[:, None].astype(np.float32)).to(device)
        m_t = torch.from_numpy(y_mass.astype(np.float32)).to(device)
        occupied_weight = torch.from_numpy((density.reshape(-1) > 0).astype(np.float32)[:, None]).to(device)

        class Model(nn.Module):
            def __init__(self, bins: int, hidden_widths: list[int]):
                super().__init__()
                self.freq = torch.tensor([1.0, 2.0, 4.0, 8.0], dtype=torch.float32)
                encoded_dim = 3 + 3 * 2 * 4
                layers = []
                last_dim = encoded_dim
                for width in hidden_widths:
                    layers.append(nn.Linear(last_dim, width))
                    layers.append(nn.SiLU())
                    last_dim = width
                layers.append(nn.Linear(last_dim, 1 + bins))
                self.net = nn.Sequential(*layers)

            def encode(self, x):
                freq = self.freq.to(x.device)
                xb = x[..., None] * freq
                enc = [x, torch.sin(math.pi * xb).flatten(1), torch.cos(math.pi * xb).flatten(1)]
                return torch.cat(enc, dim=1)

            def forward(self, x):
                return self.net(self.encode(x))

        model = Model(mass_bins, widths).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.neural_lr, weight_decay=1e-5)
        rng = np.random.default_rng(args.seed + 202)
        total = len(coords)
        batch = min(args.neural_batch, total)
        for _ in range(args.neural_epochs):
            idx = rng.choice(total, size=batch, replace=False)
            idx_t = torch.from_numpy(idx).to(device)
            out = model(x_t[idx_t])
            d_pred = torch.sigmoid(out[:, :1])
            logits = out[:, 1:]
            loss_density = F.mse_loss(d_pred, d_t[idx_t])
            mass_target = m_t[idx_t]
            mass_loss = -(mass_target * F.log_softmax(logits, dim=1)).sum(dim=1, keepdim=True)
            loss_mass = (mass_loss * occupied_weight[idx_t]).sum() / occupied_weight[idx_t].sum().clamp_min(1.0)
            loss = loss_density + 0.25 * loss_mass
            opt.zero_grad()
            loss.backward()
            opt.step()

        model_path = ds_dir / subdir / "model.pt"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "mass_edges": mass_edges,
                "bounds": ref.bounds,
                "settings": {
                    "grid": grid,
                    "mass_bins": mass_bins,
                    "epochs": args.neural_epochs,
                    "device": device,
                    "widths": widths,
                },
            },
            model_path,
        )

        with torch.no_grad():
            out_chunks = []
            eval_batch = 32768
            for start in range(0, total, eval_batch):
                pred = model(x_t[start : start + eval_batch])
                d = torch.sigmoid(pred[:, :1]).cpu().numpy().reshape(-1)
                p = torch.softmax(pred[:, 1:], dim=1).cpu().numpy()
                out_chunks.append((d, p))
        pred_density = np.concatenate([x[0] for x in out_chunks]).reshape(grid, grid, grid)
        pred_probs = np.concatenate([x[1] for x in out_chunks], axis=0).reshape(grid, grid, grid, mass_bins)
        pred_fields = np.moveaxis(pred_probs * pred_density[..., None], -1, 0)
        points = sample_from_spectral_field(pred_fields, ref.bounds, mass_edges, min(args.neural_points, ref.atom_count), seed=args.seed + 303)
        display_path = ds_dir / subdir / "reconstruction_points.npz"
        display_size = write_point_pack(display_path, points, ref.bounds, ref.mass_range)
        compressed_size = model_path.stat().st_size
        metrics = point_metrics(points, ref)
        return {
            "id": variant_id,
            "label": variant_label or METHOD_LABELS["neural"],
            "available": True,
            "compressed_size_bytes": compressed_size,
            "display_cache_size_bytes": display_size,
            "compression_ratio": ref.raw_size_bytes / max(compressed_size, 1),
            "points": int(len(points)),
            "artifact": str(model_path),
            "display_artifact": str(display_path),
            "preprocess_sec": now() - started,
            "metrics": metrics,
            "settings": {
                "grid": grid,
                "mass_bins": mass_bins,
                "epochs": args.neural_epochs,
                "device": device,
                "widths": widths,
            },
            "notes": f"Fourier-feature MLP trained on {device}.",
        }
    except Exception as exc:
        write_json(
            ds_dir / "neural" / "unavailable.json",
            {"error": str(exc), "traceback": traceback.format_exc()},
        )
        return {
            "label": METHOD_LABELS["neural"],
            "available": False,
            "compressed_size_bytes": 0,
            "compression_ratio": 0.0,
            "points": 0,
            "preprocess_sec": now() - started,
            "metrics": None,
            "notes": f"Neural preprocessing failed: {exc}",
        }


def parse_ints(value: Any, default: list[int]) -> list[int]:
    if value is None:
        return default
    if isinstance(value, str):
        return [int(x.strip()) for x in value.split(",") if x.strip()]
    return [int(x) for x in value]


def sample_from_flat_weights(
    flat_weights: np.ndarray,
    grid: int,
    mass_edges: np.ndarray,
    bounds: np.ndarray,
    target: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    weights = np.clip(flat_weights.astype(np.float64, copy=False), 0.0, None)
    total = float(weights.sum())
    if total <= 0 or target <= 0:
        return np.empty((0, 4), dtype=np.float32)
    probs = weights / total
    idx = rng.choice(len(weights), size=target, replace=True, p=probs)
    cells = grid * grid * grid
    cell = idx // (len(mass_edges) - 1)
    mi = idx % (len(mass_edges) - 1)
    zi = cell // (grid * grid)
    rem = cell % (grid * grid)
    yi = rem // grid
    xi = rem % grid
    jitter = rng.random((target, 3), dtype=np.float32)
    norm = np.column_stack([xi, yi, zi]).astype(np.float32)
    norm = (norm + jitter) / float(grid)
    xyz = bounds[0] + norm * (bounds[1] - bounds[0])
    mass_lo = mass_edges[mi]
    mass_hi = mass_edges[mi + 1]
    mass = mass_lo + rng.random(target, dtype=np.float32) * (mass_hi - mass_lo)
    return np.column_stack([xyz, mass]).astype(np.float32)


def build_neural_spatial_grid(
    path: Path,
    ds_dir: Path,
    ref: ReferenceStats,
    args: argparse.Namespace,
    variant_id: str = "spatial_grid_4d",
    variant_label: str | None = None,
    subdir: str = "neural_spatial_grid_4d",
) -> dict[str, Any]:
    started = now()
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except Exception as exc:
        return {
            "id": variant_id,
            "label": variant_label or "Neural spatial grid 4D",
            "available": False,
            "compressed_size_bytes": 0,
            "compression_ratio": 0.0,
            "points": 0,
            "preprocess_sec": now() - started,
            "metrics": None,
            "notes": f"PyTorch is not available: {exc}",
        }

    try:
        grid = args.neural_grid4d_grid
        mass_bins = args.neural_grid4d_mass_bins
        grid_levels = parse_ints(args.neural_grid4d_levels, [16, 32, 64])
        channels = int(args.neural_grid4d_channels)
        widths = parse_widths(args.neural_grid4d_widths)
        fields, mass_edges = build_spectral_grid(path, ref, grid, mass_bins, args.chunk_atoms)

        counts = np.moveaxis(fields, 0, -1).reshape(-1).astype(np.float32)
        target_scale = max(float(np.log1p(counts.max())), 1.0)
        positive_idx = np.flatnonzero(counts > 0)
        if len(positive_idx) == 0:
            raise ValueError("No occupied spectral bins found for neural spatial-grid training.")
        positive_prob = np.log1p(counts[positive_idx]).astype(np.float64)
        positive_prob /= positive_prob.sum()

        cells = grid * grid * grid
        total_bins = cells * mass_bins
        cell_indices = np.arange(cells, dtype=np.int64)
        zi = cell_indices // (grid * grid)
        rem = cell_indices % (grid * grid)
        yi = rem // grid
        xi = rem % grid
        xyz_centers = np.column_stack([xi, yi, zi]).astype(np.float32)
        xyz_centers = ((xyz_centers + 0.5) / float(grid)) * 2.0 - 1.0
        mass_centers = np.linspace(-1.0 + 1.0 / mass_bins, 1.0 - 1.0 / mass_bins, mass_bins, dtype=np.float32)

        device = "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"

        class SpatialGridField(nn.Module):
            def __init__(self, levels: list[int], grid_channels: int, hidden_widths: list[int]):
                super().__init__()
                self.levels = levels
                self.grids = nn.ParameterList(
                    [nn.Parameter(torch.randn(1, grid_channels, res, res, res) * 0.01) for res in levels]
                )
                self.register_buffer("mass_freq", torch.tensor([1.0, 2.0, 4.0, 8.0, 16.0], dtype=torch.float32), persistent=False)
                in_dim = len(levels) * grid_channels + 1 + 2 * len(self.mass_freq)
                layers = []
                last = in_dim
                for width in hidden_widths:
                    layers.append(nn.Linear(last, width))
                    layers.append(nn.SiLU())
                    last = width
                layers.append(nn.Linear(last, 1))
                self.net = nn.Sequential(*layers)

            def sample_grid(self, grid_param: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
                res = grid_param.shape[-1]
                channels = grid_param.shape[1]
                pos = (xyz + 1.0) * (0.5 * float(res - 1))
                x = pos[:, 0].clamp(0.0, float(res - 1))
                y = pos[:, 1].clamp(0.0, float(res - 1))
                z = pos[:, 2].clamp(0.0, float(res - 1))
                x0 = torch.floor(x).to(torch.long).clamp(0, res - 1)
                y0 = torch.floor(y).to(torch.long).clamp(0, res - 1)
                z0 = torch.floor(z).to(torch.long).clamp(0, res - 1)
                x1 = (x0 + 1).clamp(0, res - 1)
                y1 = (y0 + 1).clamp(0, res - 1)
                z1 = (z0 + 1).clamp(0, res - 1)
                wx = (x - x0.to(x.dtype))[:, None]
                wy = (y - y0.to(y.dtype))[:, None]
                wz = (z - z0.to(z.dtype))[:, None]

                flat = grid_param[0].permute(1, 2, 3, 0).reshape(-1, channels)

                def gather(ix, iy, iz):
                    return flat[(iz * res + iy) * res + ix]

                c000 = gather(x0, y0, z0)
                c100 = gather(x1, y0, z0)
                c010 = gather(x0, y1, z0)
                c110 = gather(x1, y1, z0)
                c001 = gather(x0, y0, z1)
                c101 = gather(x1, y0, z1)
                c011 = gather(x0, y1, z1)
                c111 = gather(x1, y1, z1)
                c00 = c000 * (1.0 - wx) + c100 * wx
                c10 = c010 * (1.0 - wx) + c110 * wx
                c01 = c001 * (1.0 - wx) + c101 * wx
                c11 = c011 * (1.0 - wx) + c111 * wx
                c0 = c00 * (1.0 - wy) + c10 * wy
                c1 = c01 * (1.0 - wy) + c11 * wy
                return c0 * (1.0 - wz) + c1 * wz

            def forward(self, xyz: torch.Tensor, mass: torch.Tensor) -> torch.Tensor:
                features = []
                for grid_param in self.grids:
                    features.append(self.sample_grid(grid_param, xyz))
                m = mass[:, None]
                mf = m * self.mass_freq.to(m.device)[None, :]
                mass_features = torch.cat([m, torch.sin(math.pi * mf), torch.cos(math.pi * mf)], dim=1)
                return self.net(torch.cat(features + [mass_features], dim=1)).squeeze(1)

        model = SpatialGridField(grid_levels, channels, widths).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.neural_grid4d_lr, weight_decay=1e-6)
        xyz_centers_t = torch.from_numpy(xyz_centers).to(device)
        mass_centers_t = torch.from_numpy(mass_centers).to(device)
        rng = np.random.default_rng(args.seed + 404)
        batch = int(args.neural_grid4d_batch)
        pos_fraction = float(args.neural_grid4d_positive_fraction)

        for _ in range(int(args.neural_grid4d_epochs)):
            pos_count = min(len(positive_idx), max(1, int(batch * pos_fraction)))
            neg_count = max(0, batch - pos_count)
            pos = rng.choice(positive_idx, size=pos_count, replace=len(positive_idx) < pos_count, p=positive_prob)
            neg_parts = []
            while sum(len(x) for x in neg_parts) < neg_count:
                candidate = rng.integers(0, total_bins, size=max(neg_count * 2, 1024), dtype=np.int64)
                candidate = candidate[counts[candidate] == 0]
                if len(candidate):
                    neg_parts.append(candidate[: max(0, neg_count - sum(len(x) for x in neg_parts))])
            if neg_parts:
                idx_np = np.concatenate([pos, *neg_parts])
            else:
                idx_np = pos
            rng.shuffle(idx_np)
            cell_np = idx_np // mass_bins
            mass_np = idx_np % mass_bins
            xyz_t = xyz_centers_t[torch.from_numpy(cell_np).to(device)]
            mass_t = mass_centers_t[torch.from_numpy(mass_np).to(device)]
            target_np = (np.log1p(counts[idx_np]) / target_scale).astype(np.float32)
            target_t = torch.from_numpy(target_np).to(device)
            pred = F.softplus(model(xyz_t, mass_t))
            weight = torch.where(target_t > 0, torch.tensor(3.0, device=device), torch.tensor(0.35, device=device))
            loss = (weight * (pred - target_t).pow(2)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

        model_path = ds_dir / subdir / "model.pt"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "bounds": ref.bounds,
                "mass_edges": mass_edges,
                "settings": {
                    "architecture": "spatial_grid_4d_scalar_density",
                    "grid": grid,
                    "mass_bins": mass_bins,
                    "grid_levels": grid_levels,
                    "channels": channels,
                    "widths": widths,
                    "epochs": args.neural_grid4d_epochs,
                    "device": device,
                    "target_scale": target_scale,
                },
            },
            model_path,
        )

        weights = np.empty(total_bins, dtype=np.float32)
        eval_mass_chunk = int(args.neural_grid4d_eval_mass_chunk)
        eval_cell_chunk = int(args.neural_grid4d_eval_cell_chunk)
        with torch.no_grad():
            for cell_start in range(0, cells, eval_cell_chunk):
                cell_stop = min(cell_start + eval_cell_chunk, cells)
                xyz_t = xyz_centers_t[cell_start:cell_stop]
                for mass_start in range(0, mass_bins, eval_mass_chunk):
                    mass_stop = min(mass_start + eval_mass_chunk, mass_bins)
                    xyz_rep = xyz_t.repeat_interleave(mass_stop - mass_start, dim=0)
                    mass_rep = mass_centers_t[mass_start:mass_stop].repeat(cell_stop - cell_start)
                    pred = F.softplus(model(xyz_rep, mass_rep)).cpu().numpy()
                    pred_counts = np.expm1(np.clip(pred, 0.0, None) * target_scale).astype(np.float32)
                    block = pred_counts.reshape(cell_stop - cell_start, mass_stop - mass_start)
                    for offset, mass_idx in enumerate(range(mass_start, mass_stop)):
                        weights[(cell_start * mass_bins + mass_idx) : (cell_stop * mass_bins + mass_idx) : mass_bins] = block[:, offset]

        weight_matrix = weights.reshape(cells, mass_bins)
        target_matrix = counts.reshape(cells, mass_bins)
        pred_mass = weight_matrix.sum(axis=0)
        target_mass = target_matrix.sum(axis=0)
        mass_scale = np.divide(target_mass, pred_mass, out=np.zeros_like(target_mass), where=pred_mass > 0)
        weight_matrix *= mass_scale[None, :]
        torch.save(
            {
                "state_dict": model.state_dict(),
                "bounds": ref.bounds,
                "mass_edges": mass_edges,
                "mass_scale": mass_scale.astype(np.float32),
                "settings": {
                    "architecture": "spatial_grid_4d_scalar_density",
                    "grid": grid,
                    "mass_bins": mass_bins,
                    "grid_levels": grid_levels,
                    "channels": channels,
                    "widths": widths,
                    "epochs": args.neural_grid4d_epochs,
                    "device": device,
                    "target_scale": target_scale,
                    "mass_marginal_calibration": True,
                },
            },
            model_path,
        )
        compressed_size = model_path.stat().st_size

        points = sample_from_flat_weights(
            weights,
            grid,
            mass_edges,
            ref.bounds,
            min(args.neural_grid4d_points, ref.atom_count),
            seed=args.seed + 505,
        )
        display_path = ds_dir / subdir / "reconstruction_points.npz"
        display_size = write_point_pack(display_path, points, ref.bounds, ref.mass_range)
        metrics = point_metrics(points, ref)
        label = variant_label or "Neural spatial grid 4D"
        if variant_label and "(" not in variant_label:
            label = f"{variant_label} ({mb(compressed_size):.2f} MB)"
        return {
            "id": variant_id,
            "label": label,
            "available": True,
            "compressed_size_bytes": compressed_size,
            "display_cache_size_bytes": display_size,
            "compression_ratio": ref.raw_size_bytes / max(compressed_size, 1),
            "points": int(len(points)),
            "artifact": str(model_path),
            "display_artifact": str(display_path),
            "preprocess_sec": now() - started,
            "metrics": metrics,
            "settings": {
                "architecture": "spatial_grid_4d_scalar_density",
                "grid": grid,
                "mass_bins": mass_bins,
                "grid_levels": grid_levels,
                "channels": channels,
                "widths": widths,
                "epochs": args.neural_grid4d_epochs,
                "device": device,
                "mass_marginal_calibration": True,
            },
            "notes": f"Mass-conditioned scalar density with multiresolution spatial feature grids trained on {device}.",
        }
    except Exception as exc:
        write_json(
            ds_dir / subdir / "unavailable.json",
            {"error": str(exc), "traceback": traceback.format_exc()},
        )
        return {
            "id": variant_id,
            "label": variant_label or "Neural spatial grid 4D",
            "available": False,
            "compressed_size_bytes": 0,
            "compression_ratio": 0.0,
            "points": 0,
            "preprocess_sec": now() - started,
            "metrics": None,
            "notes": f"Neural spatial-grid preprocessing failed: {exc}",
        }


def full_method(ref: ReferenceStats) -> dict[str, Any]:
    return {
        "label": METHOD_LABELS["full"],
        "available": True,
        "compressed_size_bytes": ref.raw_size_bytes,
        "compression_ratio": 1.0,
        "points": ref.atom_count,
        "preprocess_sec": 0.0,
        "metrics": {
            "mass_spectrum_error": 0.0,
            "z_profile_error": 0.0,
            "radial_profile_error": 0.0,
            "spatial_spectral_error": 0.0,
            "reconstructed_spectrum": ref.spectrum_counts.tolist(),
        },
        "notes": "Streams and samples directly from the original .pos file.",
    }


def process_dataset(path: Path, args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ds_id = slug_for_path(path)
    ds_dir = args.out_dir / "datasets" / ds_id
    ds_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing {path.name}")
    started = now()
    ref = compute_reference(path, ds_dir, args)
    print(f"  atoms: {ref.atom_count:,}, raw: {mb(ref.raw_size_bytes):.1f} MB")

    methods: dict[str, Any] = {"full": full_method(ref)}
    summary_rows = [
        {
            "dataset": path.name,
            "method": "Full file",
            "raw_mb": mb(ref.raw_size_bytes),
            "compressed_mb": mb(ref.raw_size_bytes),
            "ratio": 1.0,
            "points": ref.atom_count,
            "preprocess_sec": 0.0,
            "notes": methods["full"]["notes"],
        }
    ]

    jitter = build_jitter(path, ds_dir, ref, args)
    methods["jitter"] = jitter
    summary_rows.append(
        {
            "dataset": path.name,
            "method": jitter["label"],
            "raw_mb": mb(ref.raw_size_bytes),
            "compressed_mb": mb(jitter["compressed_size_bytes"]),
            "ratio": jitter["compression_ratio"],
            "points": jitter["points"],
            "preprocess_sec": jitter["preprocess_sec"],
            "notes": jitter["notes"],
        }
    )
    print("  jitter complete")

    dataset = {
        "id": ds_id,
        "name": path.name,
        "raw_path": str(path.resolve()),
        "source_mtime": path.stat().st_mtime,
        "atom_count": ref.atom_count,
        "raw_size_bytes": ref.raw_size_bytes,
        "bounds": ref.bounds,
        "mass_range": ref.mass_range,
        "spectrum": {
            "edges": ref.spectrum_edges,
            "counts": ref.spectrum_counts,
        },
        "methods": methods,
        "preprocess_sec": now() - started,
    }
    write_json(ds_dir / "dataset_manifest.json", dataset)
    return dataset, summary_rows


def print_summary(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    headers = ["dataset", "method", "raw MB", "compressed MB", "ratio", "points shown", "preprocess sec", "notes"]
    normalized = []
    for row in rows:
        normalized.append(
            [
                row["dataset"],
                row["method"],
                f"{row['raw_mb']:.2f}",
                f"{row['compressed_mb']:.2f}",
                f"{row['ratio']:.2f}",
                f"{int(row['points']):,}",
                f"{row['preprocess_sec']:.2f}",
                row["notes"],
            ]
        )
    widths = [len(h) for h in headers]
    for row in normalized:
        widths = [max(w, len(str(v))) for w, v in zip(widths, row)]
    print("\n" + " | ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("-+-".join("-" * w for w in widths))
    for row in normalized:
        print(" | ".join(str(v).ljust(w) for v, w in zip(row, widths)))


def parse_levels(value: str) -> list[int]:
    levels = []
    for item in value.split(","):
        item = item.strip().replace("_", "")
        if item:
            levels.append(int(item))
    return levels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess APT .pos files for compression comparison.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--quick", action="store_true", help="Fast settings and first dataset only unless --all is set.")
    parser.add_argument("--all", action="store_true", help="In quick mode, still process every dataset.")
    parser.add_argument("--max-datasets", type=int, default=None)
    parser.add_argument("--dataset-filter", default="", help="Comma-separated filename/id substrings limiting datasets to preprocess.")
    parser.add_argument("--synthetic", action="store_true", help="Generate and process a synthetic dataset.")
    parser.add_argument("--synthetic-atoms", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--chunk-atoms", type=int, default=2_000_000)
    parser.add_argument("--spectrum-bins", type=int, default=1024)
    parser.add_argument("--profile-bins", type=int, default=64)
    parser.add_argument("--spatial-bins", type=int, default=6)
    parser.add_argument("--metric-mass-bins", type=int, default=32)
    parser.add_argument("--jitter-reduction", type=int, default=10)
    parser.add_argument("--jitter-copies", type=int, default=10)
    parser.add_argument("--jitter-radius", type=float, default=None, help="Position jitter radius; defaults to an occupancy-estimated atom spacing.")
    parser.add_argument("--jitter-radius-scale", type=float, default=1.25)
    parser.add_argument("--jitter-occupancy-grid", type=int, default=48)
    parser.add_argument("--jitter-points", type=int, default=None, help="Optional cap for the expanded jittered display cache.")
    args = parser.parse_args()

    if args.quick:
        args.synthetic_atoms = args.synthetic_atoms or 180_000
        args.jitter_points = args.jitter_points or 1_000_000
    else:
        args.synthetic_atoms = args.synthetic_atoms or 750_000
        args.jitter_points = args.jitter_points or None

    return args


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        synthetic = generate_synthetic_pos(args.data_dir, args.synthetic_atoms, args.seed)
        print(f"Synthetic POS ready: {synthetic}")

    files = find_pos_files(args.data_dir)
    if not files:
        synthetic = generate_synthetic_pos(args.data_dir, args.synthetic_atoms, args.seed)
        print(f"No .pos files found; generated {synthetic}")
        files = [synthetic]

    dataset_tokens = [x.strip().lower() for x in args.dataset_filter.split(",") if x.strip()]
    if dataset_tokens:
        files = [
            path for path in files
            if any(token in path.name.lower() or token in slug_for_path(path).lower() for token in dataset_tokens)
        ]
        if not files:
            raise SystemExit(f"No .pos files matched --dataset-filter={args.dataset_filter!r}")

    if args.quick and not args.all:
        files = files[:1]
    if args.max_datasets is not None:
        files = files[: args.max_datasets]

    manifest = {
        "version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "data_dir": str(args.data_dir.resolve()),
        "settings": {
            "quick": args.quick,
            "jitter_reduction": args.jitter_reduction,
            "jitter_copies": args.jitter_copies,
            "jitter_radius": args.jitter_radius,
            "jitter_radius_scale": args.jitter_radius_scale,
            "jitter_points": args.jitter_points,
        },
        "methods": METHOD_LABELS,
        "datasets": [],
    }

    all_rows: list[dict[str, Any]] = []
    for path in files:
        dataset, rows = process_dataset(path, args)
        manifest["datasets"].append(dataset)
        all_rows.extend(rows)

    write_json(args.out_dir / "manifest.json", manifest)
    print_summary(all_rows)
    print(f"\nManifest written to {args.out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
