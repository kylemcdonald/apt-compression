from __future__ import annotations

import argparse
import heapq
import json
import math
import struct
import time
import zlib
from pathlib import Path
from typing import Any

import numpy as np

import preprocess as pp
from hypernetwork_common import load_reference


MAGIC = b"CDFV2\0\0\0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train improved CDF density-grid hypernetworks with compressed supports.")
    parser.add_argument("--dataset-filter", default="499e563f-0c0c-4c6f-bc08-b8e76f59c31b")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--ranges", type=int, default=64)
    parser.add_argument("--targets-mb", default="10")
    parser.add_argument(
        "--density-mode",
        choices=["linear"],
        default="linear",
        help="Linear count-density targets. The argument is retained for command compatibility.",
    )
    parser.add_argument(
        "--min-mass-bin-width",
        type=float,
        default=0.1,
        help="Minimum output mass-bin width in Da. Also used as the fine histogram width for spectrum-CDF binning.",
    )
    parser.add_argument(
        "--min-range-atom-fraction",
        type=float,
        default=0.0001,
        help="Minimum fraction of dataset atoms in each output mass range. Default 0.0001 means 0.01%%.",
    )
    parser.add_argument(
        "--mass-bin-cdf",
        choices=["spatial-change", "log-derivative", "derivative", "log-count", "atom-count"],
        default="spatial-change",
        help="Allocate mass ranges from a fine spectrum CDF/change metric: adjacent spatial-density change, abs derivative of log1p(counts), abs derivative of counts, log1p(counts), or raw atom-count quantiles.",
    )
    parser.add_argument("--spatial-cdf-grid-shape", default="8,8,32", help="Coarse xyz grid used by --mass-bin-cdf spatial-change.")
    parser.add_argument("--spatial-cdf-min-atoms", type=float, default=2000.0, help="Reliability scale for spatial-change adjacent-bin comparisons.")
    parser.add_argument("--spatial-cdf-spatial-weight", type=float, default=1.0, help="Weight for spatial-distribution change in --mass-bin-cdf spatial-change.")
    parser.add_argument("--spatial-cdf-count-weight", type=float, default=1.5, help="Weight for log count-rate jumps in --mass-bin-cdf spatial-change.")
    parser.add_argument("--base-grid-shape", default="64,64,256")
    parser.add_argument("--residual-grid-shape", default="128,128,512")
    parser.add_argument("--auto-grid", action="store_true", help="Scale the spatial grid from a reference dataset/grid to keep approximately constant voxel size.")
    parser.add_argument("--reference-dataset-filter", default="499e563f-0c0c-4c6f-bc08-b8e76f59c31b")
    parser.add_argument("--reference-grid-shape", default="64,64,256")
    parser.add_argument("--support-mode", choices=["per-bin", "union", "hybrid", "adaptive"], default="adaptive")
    parser.add_argument(
        "--hybrid-support-max-cells",
        type=int,
        default=50_000,
        help="For --support-mode hybrid, store exact per-bin supports only for ranges at or below this occupied-cell count.",
    )
    parser.add_argument(
        "--adaptive-support-budget-mb",
        type=float,
        default=0.0,
        help="Compressed support-mask budget for --support-mode adaptive. If zero, derive it from --adaptive-support-budget-fraction.",
    )
    parser.add_argument(
        "--adaptive-support-budget-fraction",
        type=float,
        default=0.35,
        help="Fraction of the largest target artifact size reserved for support masks when --adaptive-support-budget-mb is zero.",
    )
    parser.add_argument("--residual-cells-per-range", type=int, default=8192)
    parser.add_argument(
        "--residual-mode",
        choices=["highres-excess", "base-excess"],
        default="base-excess",
        help="Sparse residual source: high-resolution excess over the base grid, or base-grid atoms above a one-atom occupancy floor.",
    )
    parser.add_argument(
        "--residual-base-floor",
        type=float,
        default=1.0,
        help="Base-grid count floor subtracted by --residual-mode base-excess.",
    )
    parser.add_argument("--max-rank", type=int, default=32)
    parser.add_argument("--compress-arrays", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hyper-epochs", type=int, default=9000)
    parser.add_argument("--hyper-width", type=int, default=96)
    parser.add_argument("--basis-dtype", choices=["int8"], default="int8")
    parser.add_argument("--metric-samples", type=int, default=300000)
    parser.add_argument("--seed", type=int, default=55)
    parser.add_argument("--chunk-atoms", type=int, default=2_000_000)
    parser.add_argument("--spectrum-bins", type=int, default=1024)
    parser.add_argument("--profile-bins", type=int, default=64)
    parser.add_argument("--spatial-bins", type=int, default=6)
    return parser.parse_args()


def find_dataset(manifest: dict[str, Any], dataset_filter: str) -> dict[str, Any]:
    needle = dataset_filter.lower()
    return next(
        dataset
        for dataset in manifest["datasets"]
        if needle in dataset["name"].lower() or needle in dataset["id"].lower()
    )


def parse_shape(value: str) -> tuple[int, int, int]:
    parts = [part.strip() for part in value.lower().replace("x", ",").split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError("shape must be three integers like 64,64,256")
    shape = tuple(int(part) for part in parts)
    if any(v <= 0 for v in shape):
        raise ValueError("shape dimensions must be positive")
    return shape


def shape_label(shape: tuple[int, int, int]) -> str:
    return f"{shape[0]}x{shape[1]}x{shape[2]}"


def auto_grid_shape(dataset: dict[str, Any], reference: dict[str, Any], reference_shape: tuple[int, int, int]) -> tuple[int, int, int]:
    bounds = np.asarray(dataset["bounds"], dtype=np.float64)
    ref_bounds = np.asarray(reference["bounds"], dtype=np.float64)
    extent = np.maximum(bounds[1] - bounds[0], 1e-6)
    ref_extent = np.maximum(ref_bounds[1] - ref_bounds[0], 1e-6)
    voxel = ref_extent / np.asarray(reference_shape, dtype=np.float64)
    shape = np.ceil(extent / np.maximum(voxel, 1e-6)).astype(np.int64)
    return tuple(int(max(8, value)) for value in shape)


def target_values(text: str) -> list[float]:
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("--targets-mb must contain at least one value")
    return values


def load_mass_values(path: Path, chunk_atoms: int) -> np.ndarray:
    mm, _, _ = pp.open_pos(path)
    parts: list[np.ndarray] = []
    for _, _, chunk in pp.iter_chunks(mm, chunk_atoms):
        finite = np.isfinite(chunk).all(axis=1)
        if finite.any():
            parts.append(chunk[finite, 3].astype(np.float32, copy=True))
    if not parts:
        return np.empty(0, dtype=np.float32)
    return np.concatenate(parts)


def isotonic_non_decreasing(values: np.ndarray) -> np.ndarray:
    levels: list[float] = []
    weights: list[int] = []
    lengths: list[int] = []
    for value in values:
        levels.append(float(value))
        weights.append(1)
        lengths.append(1)
        while len(levels) >= 2 and levels[-2] > levels[-1]:
            weight = weights[-2] + weights[-1]
            level = (levels[-2] * weights[-2] + levels[-1] * weights[-1]) / weight
            length = lengths[-2] + lengths[-1]
            levels[-2:] = [level]
            weights[-2:] = [weight]
            lengths[-2:] = [length]
    return np.repeat(np.asarray(levels, dtype=np.float64), lengths)


def enforce_minimum_edge_spacing(edges: np.ndarray, min_width: float) -> np.ndarray:
    edges = np.asarray(edges, dtype=np.float64).copy()
    if len(edges) <= 2:
        return edges
    if min_width <= 0:
        for i in range(1, len(edges)):
            if edges[i] <= edges[i - 1]:
                edges[i] = np.nextafter(edges[i - 1], np.inf)
        return edges

    bin_count = len(edges) - 1
    total_width = float(edges[-1] - edges[0])
    required_width = float(bin_count * min_width)
    if total_width + 1e-9 < required_width:
        raise ValueError(
            f"Cannot fit {bin_count} bins with minimum width {min_width:g} Da "
            f"inside mass range {total_width:g} Da"
        )

    # Transform x_i into y_i = x_i - i*w. Then x_i - x_(i-1) >= w
    # becomes y_i >= y_(i-1). Project the quantile edges onto that
    # monotone constraint while preserving the observed min/max range.
    index_offsets = np.arange(len(edges), dtype=np.float64) * float(min_width)
    y = edges - index_offsets
    y_min = float(edges[0])
    y_max = float(edges[-1] - required_width)
    if len(edges) > 2:
        inner = np.clip(y[1:-1], y_min, y_max)
        inner = np.clip(isotonic_non_decreasing(inner), y_min, y_max)
        y = np.concatenate([[y_min], inner, [y_max]])
    else:
        y = np.asarray([y_min, y_max], dtype=np.float64)
    adjusted = y + index_offsets
    adjusted[0] = edges[0]
    adjusted[-1] = edges[-1]

    # Final bounded passes absorb small floating-point violations.
    for i in range(1, len(adjusted)):
        min_edge = adjusted[i - 1] + min_width
        if adjusted[i] < min_edge:
            adjusted[i] = min_edge
    adjusted[-1] = edges[-1]
    for i in range(len(adjusted) - 2, -1, -1):
        max_edge = adjusted[i + 1] - min_width
        if adjusted[i] > max_edge:
            adjusted[i] = max_edge
    adjusted[0] = edges[0]
    return adjusted


def fine_mass_edges(mass_min: float, mass_max: float, width: float) -> np.ndarray:
    if width <= 0:
        raise ValueError("spectrum-CDF binning requires --min-mass-bin-width > 0")
    if mass_max <= mass_min:
        return np.asarray([mass_min, mass_max], dtype=np.float64)
    extent = float(mass_max - mass_min)
    regular_bins = int(math.floor((extent + width * 1e-9) / width))
    if regular_bins < 1:
        return np.asarray([mass_min, mass_max], dtype=np.float64)
    edges = mass_min + np.arange(regular_bins + 1, dtype=np.float64) * width
    if mass_max - edges[-1] > width * 1e-9:
        # Merge the partial tail into the final fine bin so no fine bin is
        # narrower than the requested width.
        edges[-1] = mass_max
    else:
        edges[-1] = mass_max
    return edges


def fine_mass_histogram(masses: np.ndarray, ref: pp.ReferenceStats, fine_width: float) -> tuple[np.ndarray, np.ndarray]:
    mass_min = float(ref.mass_range[0])
    mass_max = float(ref.mass_range[1])
    fine_edges = fine_mass_edges(mass_min, mass_max, fine_width)
    fine_counts = np.histogram(masses, bins=fine_edges)[0].astype(np.int64)
    return fine_edges, fine_counts


def weighted_cdf_edge_indices(weights: np.ndarray, range_count: int) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    fine_count = int(weights.size)
    if fine_count < range_count:
        raise ValueError(f"Need at least {range_count} fine mass bins, found {fine_count}")
    edge_indices = np.empty(range_count + 1, dtype=np.int64)
    edge_indices[0] = 0
    edge_indices[-1] = fine_count
    total = float(weights.sum())
    if total <= 0:
        for i in range(1, range_count):
            edge_indices[i] = int(round(i * fine_count / range_count))
        return np.maximum.accumulate(edge_indices)

    cdf = np.cumsum(weights)
    for i in range(1, range_count):
        target = total * i / range_count
        candidate = int(np.searchsorted(cdf, target, side="left") + 1)
        lower = int(edge_indices[i - 1] + 1)
        upper = fine_count - (range_count - i)
        edge_indices[i] = min(max(candidate, lower), upper)
    return edge_indices


def minimum_range_atom_count(fine_counts: np.ndarray, args: argparse.Namespace) -> int:
    fraction = max(float(args.min_range_atom_fraction), 0.0)
    if fraction <= 0.0:
        return 0
    total_atoms = int(np.asarray(fine_counts, dtype=np.int64).sum())
    return int(math.ceil(total_atoms * fraction))


def boundary_strength_from_bin_weights(weights: np.ndarray | None, fine_count: int) -> np.ndarray:
    if fine_count <= 1:
        return np.zeros(0, dtype=np.float64)
    if weights is None:
        return np.zeros(fine_count - 1, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.size != fine_count:
        return np.zeros(fine_count - 1, dtype=np.float64)
    return (weights[:-1] + weights[1:]) * 0.5


def enforce_minimum_range_atoms(
    edge_indices: np.ndarray,
    fine_counts: np.ndarray,
    args: argparse.Namespace,
    boundary_strength: np.ndarray | None = None,
) -> np.ndarray:
    fine_counts = np.asarray(fine_counts, dtype=np.int64)
    fine_count = int(fine_counts.size)
    target_count = int(args.ranges)
    min_atoms = minimum_range_atom_count(fine_counts, args)
    if min_atoms <= 0 or fine_count <= 1:
        return np.asarray(edge_indices, dtype=np.int64)

    prefix = np.concatenate([[0], np.cumsum(fine_counts, dtype=np.int64)])
    boundary_strength = np.asarray(
        boundary_strength if boundary_strength is not None else np.zeros(max(fine_count - 1, 0), dtype=np.float64),
        dtype=np.float64,
    )
    if boundary_strength.size != max(fine_count - 1, 0):
        boundary_strength = np.zeros(max(fine_count - 1, 0), dtype=np.float64)

    edges = [int(value) for value in np.asarray(edge_indices, dtype=np.int64)]
    edges[0] = 0
    edges[-1] = fine_count

    def segment_count(index: int) -> int:
        return int(prefix[edges[index + 1]] - prefix[edges[index]])

    def edge_strength(edge_value: int) -> float:
        if edge_value <= 0 or edge_value >= fine_count:
            return float("inf")
        return float(boundary_strength[edge_value - 1])

    while len(edges) > 2:
        counts = [segment_count(i) for i in range(len(edges) - 1)]
        small = [i for i, count in enumerate(counts) if count < min_atoms]
        if not small:
            break
        index = min(small, key=lambda i: (counts[i], edges[i + 1] - edges[i]))
        if index == 0:
            remove_edge_at = 1
        elif index == len(counts) - 1:
            remove_edge_at = len(edges) - 2
        else:
            left_boundary = edge_strength(edges[index])
            right_boundary = edge_strength(edges[index + 1])
            remove_edge_at = index if left_boundary <= right_boundary else index + 1
        del edges[remove_edge_at]

    while len(edges) - 1 < target_count:
        best: tuple[float, int, int, int] | None = None
        for index in range(len(edges) - 1):
            start = edges[index]
            stop = edges[index + 1]
            if stop - start < 2:
                continue
            total = int(prefix[stop] - prefix[start])
            if total < min_atoms * 2:
                continue
            candidates = np.arange(start + 1, stop, dtype=np.int64)
            left_counts = prefix[candidates] - prefix[start]
            right_counts = prefix[stop] - prefix[candidates]
            valid = (left_counts >= min_atoms) & (right_counts >= min_atoms)
            if not valid.any():
                continue
            valid_edges = candidates[valid]
            strengths = boundary_strength[valid_edges - 1] if boundary_strength.size else np.zeros(len(valid_edges))
            balance = np.abs(left_counts[valid] - right_counts[valid]) / max(total, 1)
            scores = strengths - balance * 0.01
            local = int(np.argmax(scores))
            split = int(valid_edges[local])
            score = float(scores[local])
            candidate = (score, total, index, split)
            if best is None or candidate > best:
                best = candidate
        if best is None:
            break
        _, _, index, split = best
        edges.insert(index + 1, split)

    return np.asarray(edges, dtype=np.int64)


def fine_ranges_from_edge_indices(
    fine_edges: np.ndarray,
    fine_counts: np.ndarray,
    edge_indices: np.ndarray,
    args: argparse.Namespace,
    mode: str,
    weights: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    fine_width = float(args.min_mass_bin_width)
    atom_prefix = np.concatenate([[0], np.cumsum(fine_counts, dtype=np.int64)])
    total_atoms = max(int(atom_prefix[-1]), 1)
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        weight_prefix = np.concatenate([[0.0], np.cumsum(weights, dtype=np.float64)])
        total_weight = max(float(weight_prefix[-1]), 1e-12)
    else:
        weight_prefix = None
        total_weight = 1.0

    range_count = int(len(edge_indices) - 1)
    min_atoms = minimum_range_atom_count(fine_counts, args)
    ranges: list[dict[str, Any]] = []
    for i in range(range_count):
        start = int(edge_indices[i])
        stop = int(edge_indices[i + 1])
        lo = float(fine_edges[start])
        hi = float(fine_edges[stop])
        atom_count = int(atom_prefix[stop] - atom_prefix[start])
        weight = float(weight_prefix[stop] - weight_prefix[start]) if weight_prefix is not None else 0.0
        ranges.append(
            {
                "id": f"range_{i + 1:02d}",
                "mass": float((lo + hi) * 0.5),
                "mass_min": lo,
                "mass_max": hi,
                "mass_width": float(hi - lo),
                "atom_count": atom_count,
                "quantile_min": float(i / args.ranges),
                "quantile_max": float((i + 1) / args.ranges),
                "atom_count_cdf_min": float(atom_prefix[start] / total_atoms),
                "atom_count_cdf_max": float(atom_prefix[stop] / total_atoms),
                "mass_weight_cdf_min": float(weight_prefix[start] / total_weight),
                "mass_weight_cdf_max": float(weight_prefix[stop] / total_weight),
                "mass_weight": weight,
                "fine_bin_start": start,
                "fine_bin_stop": stop,
                "fine_mass_bin_width": fine_width,
                "mass_bin_cdf": mode,
                "min_mass_bin_width": fine_width,
                "min_range_atom_fraction": float(ranges[0].get("min_range_atom_fraction", 0.0)) if ranges else 0.0,
                "min_range_atom_count": int(min_atoms),
            }
        )
    return ranges


def weighted_fine_cdf_ranges(
    fine_edges: np.ndarray,
    fine_counts: np.ndarray,
    weights: np.ndarray,
    args: argparse.Namespace,
    mode: str,
) -> list[dict[str, Any]]:
    weights = np.asarray(weights, dtype=np.float64)
    edge_indices = weighted_cdf_edge_indices(weights, args.ranges)
    edge_indices = enforce_minimum_range_atoms(
        edge_indices,
        fine_counts,
        args,
        boundary_strength_from_bin_weights(weights, len(fine_counts)),
    )
    return fine_ranges_from_edge_indices(fine_edges, fine_counts, edge_indices, args, mode, weights)


def log_count_cdf_ranges(masses: np.ndarray, ref: pp.ReferenceStats, args: argparse.Namespace) -> list[dict[str, Any]]:
    fine_edges, fine_counts = fine_mass_histogram(masses, ref, float(args.min_mass_bin_width))
    weights = np.log1p(fine_counts.astype(np.float64))
    ranges = weighted_fine_cdf_ranges(fine_edges, fine_counts, weights, args, "log-count")
    for item in ranges:
        item["log_count_cdf_min"] = item["mass_weight_cdf_min"]
        item["log_count_cdf_max"] = item["mass_weight_cdf_max"]
        item["log_count_weight"] = item["mass_weight"]
    return ranges


def derivative_cdf_ranges(
    masses: np.ndarray,
    ref: pp.ReferenceStats,
    args: argparse.Namespace,
    *,
    log_counts: bool,
) -> list[dict[str, Any]]:
    fine_edges, fine_counts = fine_mass_histogram(masses, ref, float(args.min_mass_bin_width))
    values = fine_counts.astype(np.float64)
    source = "log1p-counts" if log_counts else "counts"
    mode = "log-derivative" if log_counts else "derivative"
    if log_counts:
        values = np.log1p(values)
    if len(values) <= 1:
        weights = np.ones_like(values, dtype=np.float64)
    else:
        derivative = np.abs(np.diff(values))
        weights = np.zeros_like(values, dtype=np.float64)
        # A derivative sample lies between two fine mass bins. Split its
        # weight across both bins so the resulting ranges bracket sharp
        # rising and falling spectral edges instead of shifting to one side.
        weights[:-1] += derivative * 0.5
        weights[1:] += derivative * 0.5
    ranges = weighted_fine_cdf_ranges(fine_edges, fine_counts, weights, args, mode)
    for item in ranges:
        item["derivative_cdf_min"] = item["mass_weight_cdf_min"]
        item["derivative_cdf_max"] = item["mass_weight_cdf_max"]
        item["derivative_weight"] = item["mass_weight"]
        item["spectrum_derivative_source"] = source
        if log_counts:
            item["log_derivative_cdf_min"] = item["mass_weight_cdf_min"]
            item["log_derivative_cdf_max"] = item["mass_weight_cdf_max"]
            item["log_derivative_weight"] = item["mass_weight"]
    return ranges


def build_fine_spatial_density(
    path: Path,
    ref: pp.ReferenceStats,
    fine_edges: np.ndarray,
    grid_shape: tuple[int, int, int],
    chunk_atoms: int,
) -> np.ndarray:
    mm, _, _ = pp.open_pos(path)
    gx, gy, gz = grid_shape
    fine_count = len(fine_edges) - 1
    cell_count = gx * gy * gz
    density = np.zeros((fine_count, cell_count), dtype=np.float32)
    flat_density = density.reshape(-1)
    bounds = ref.bounds
    extent = np.maximum(bounds[1] - bounds[0], 1e-6)

    for _, _, chunk in pp.iter_chunks(mm, chunk_atoms):
        finite = np.isfinite(chunk).all(axis=1)
        if not finite.any():
            continue
        c = chunk[finite]
        mass = c[:, 3]
        mass_bin = np.searchsorted(fine_edges, mass, side="right") - 1
        valid = (mass_bin >= 0) & (mass_bin < fine_count)
        if not valid.any():
            continue
        mass_bin = mass_bin[valid].astype(np.int64, copy=False)
        xyz = c[valid, :3]
        norm = np.clip((xyz - bounds[0]) / extent, 0.0, 0.999999)
        xi = np.floor(norm[:, 0] * gx).astype(np.int32)
        yi = np.floor(norm[:, 1] * gy).astype(np.int32)
        zi = np.floor(norm[:, 2] * gz).astype(np.int32)
        cell = ((zi * gy + yi) * gx + xi).astype(np.int64)
        keys = mass_bin * cell_count + cell
        unique, counts = np.unique(keys, return_counts=True)
        flat_density[unique] += counts.astype(np.float32)
    return density


def hellinger_distance_for_counts(a: np.ndarray, a_total: float, b: np.ndarray, b_total: float) -> float:
    if a_total <= 0.0 or b_total <= 0.0:
        return 0.0
    affinity = float(np.sqrt(a * b, dtype=np.float64).sum() / math.sqrt(a_total * b_total))
    return math.sqrt(max(0.0, 1.0 - min(1.0, affinity)))


def spatial_merge_cost(
    density: np.ndarray,
    totals: np.ndarray,
    starts: np.ndarray,
    stops: np.ndarray,
    left: int,
    right: int,
    min_atoms: float,
    fine_width: float,
    spatial_weight: float,
    count_weight: float,
    log_rate_scale: float,
) -> float:
    left_total = float(totals[left])
    right_total = float(totals[right])
    if left_total <= 0.0 or right_total <= 0.0:
        return 0.0
    spatial_distance = hellinger_distance_for_counts(density[left], left_total, density[right], right_total)
    left_width = max(float(stops[left] - starts[left]) * fine_width, fine_width)
    right_width = max(float(stops[right] - starts[right]) * fine_width, fine_width)
    left_rate = left_total / left_width
    right_rate = right_total / right_width
    count_distance = abs(math.log1p(left_rate) - math.log1p(right_rate)) / max(log_rate_scale, 1e-6)
    count_distance = min(1.0, count_distance)
    effective_atoms = (2.0 * left_total * right_total) / max(left_total + right_total, 1e-6)
    spatial_reliability = effective_atoms / (effective_atoms + max(float(min_atoms), 1e-6))
    count_support = max(left_total, right_total)
    count_reliability = count_support / (count_support + max(float(min_atoms), 1e-6))
    return (
        max(0.0, spatial_weight) * spatial_distance * spatial_reliability
        + max(0.0, count_weight) * count_distance * count_reliability
    )


def boundary_indices_from_spatial_agglomeration(
    density: np.ndarray,
    range_count: int,
    min_atoms: float,
    fine_width: float,
    spatial_weight: float,
    count_weight: float,
) -> np.ndarray:
    fine_count = int(density.shape[0])
    if fine_count < range_count:
        raise ValueError(f"Need at least {range_count} fine mass bins, found {fine_count}")
    if range_count <= 1:
        return np.asarray([0, fine_count], dtype=np.int64)

    totals = density.sum(axis=1).astype(np.float64)
    fine_rates = totals / max(float(fine_width), 1e-6)
    positive_rates = fine_rates[fine_rates > 0.0]
    if len(positive_rates):
        log_rate_scale = max(float(np.log1p(positive_rates).max() - np.log1p(positive_rates).min()), 1.0)
    else:
        log_rate_scale = 1.0
    starts = np.arange(fine_count, dtype=np.int64)
    stops = starts + 1
    prev = np.arange(-1, fine_count - 1, dtype=np.int64)
    next_ = np.arange(1, fine_count + 1, dtype=np.int64)
    next_[-1] = -1
    alive = np.ones(fine_count, dtype=bool)
    version = np.zeros(fine_count, dtype=np.int64)
    heap: list[tuple[float, int, int, int, int]] = []

    def push(left: int) -> None:
        right = int(next_[left])
        if right < 0 or not alive[left] or not alive[right]:
            return
        cost = spatial_merge_cost(
            density,
            totals,
            starts,
            stops,
            left,
            right,
            min_atoms,
            fine_width,
            spatial_weight,
            count_weight,
            log_rate_scale,
        )
        heapq.heappush(heap, (cost, int(stops[left]), left, int(version[left]), int(version[right])))

    for i in range(fine_count - 1):
        push(i)

    segment_count = fine_count
    while segment_count > range_count and heap:
        _, _, left, left_version, right_version = heapq.heappop(heap)
        right = int(next_[left])
        if (
            right < 0
            or not alive[left]
            or not alive[right]
            or version[left] != left_version
            or version[right] != right_version
        ):
            continue

        density[left] += density[right]
        totals[left] += totals[right]
        stops[left] = stops[right]
        alive[right] = False
        version[left] += 1
        after = int(next_[right])
        next_[left] = after
        if after >= 0:
            prev[after] = left
        segment_count -= 1

        before = int(prev[left])
        if before >= 0:
            push(before)
        push(left)

    edges = [0]
    current = int(np.flatnonzero(alive)[0])
    while current >= 0:
        edges.append(int(stops[current]))
        current = int(next_[current])
    return np.asarray(edges, dtype=np.int64)


def spatial_change_cdf_ranges(path: Path, masses: np.ndarray, ref: pp.ReferenceStats, args: argparse.Namespace) -> list[dict[str, Any]]:
    fine_edges, fine_counts = fine_mass_histogram(masses, ref, float(args.min_mass_bin_width))
    grid_shape = parse_shape(args.spatial_cdf_grid_shape)
    density = build_fine_spatial_density(path, ref, fine_edges, grid_shape, args.chunk_atoms)
    totals = density.sum(axis=1).astype(np.float64)
    sqrt_density = np.zeros_like(density, dtype=np.float32)
    nonzero = totals > 0
    if nonzero.any():
        sqrt_density[nonzero] = np.sqrt(density[nonzero] / totals[nonzero, None]).astype(np.float32)

    if len(fine_counts) <= 1:
        boundary_scores = np.zeros(0, dtype=np.float64)
        boundary_spatial = np.zeros(0, dtype=np.float64)
        boundary_count = np.zeros(0, dtype=np.float64)
    else:
        similarity = np.einsum("ij,ij->i", sqrt_density[:-1], sqrt_density[1:], dtype=np.float64)
        boundary_spatial = np.sqrt(np.clip(1.0 - similarity, 0.0, 1.0))
        fine_rates = totals / max(float(args.min_mass_bin_width), 1e-6)
        positive_rates = fine_rates[fine_rates > 0.0]
        if len(positive_rates):
            log_rate_scale = max(float(np.log1p(positive_rates).max() - np.log1p(positive_rates).min()), 1.0)
        else:
            log_rate_scale = 1.0
        boundary_count = np.abs(np.diff(np.log1p(fine_rates))) / max(log_rate_scale, 1e-6)
        boundary_count = np.clip(boundary_count, 0.0, 1.0)
        left_totals = totals[:-1]
        right_totals = totals[1:]
        effective_atoms = (2.0 * left_totals * right_totals) / np.maximum(left_totals + right_totals, 1e-6)
        spatial_reliability = effective_atoms / (effective_atoms + max(float(args.spatial_cdf_min_atoms), 1e-6))
        count_support = np.maximum(left_totals, right_totals)
        count_reliability = count_support / (count_support + max(float(args.spatial_cdf_min_atoms), 1e-6))
        boundary_scores = (
            max(float(args.spatial_cdf_spatial_weight), 0.0) * boundary_spatial * spatial_reliability
            + max(float(args.spatial_cdf_count_weight), 0.0) * boundary_count * count_reliability
        )

    fine_weights = np.zeros(len(fine_counts), dtype=np.float64)
    if len(boundary_scores):
        fine_weights[:-1] += boundary_scores * 0.5
        fine_weights[1:] += boundary_scores * 0.5
    edge_indices = boundary_indices_from_spatial_agglomeration(
        density.copy(),
        args.ranges,
        args.spatial_cdf_min_atoms,
        args.min_mass_bin_width,
        args.spatial_cdf_spatial_weight,
        args.spatial_cdf_count_weight,
    )
    edge_indices = enforce_minimum_range_atoms(edge_indices, fine_counts, args, boundary_scores)
    ranges = fine_ranges_from_edge_indices(fine_edges, fine_counts, edge_indices, args, "spatial-change", fine_weights)

    for item in ranges:
        start = int(item["fine_bin_start"])
        stop = int(item["fine_bin_stop"])
        left_score = float(boundary_scores[start - 1]) if start > 0 and len(boundary_scores) else 0.0
        right_score = float(boundary_scores[stop - 1]) if stop < len(fine_counts) and len(boundary_scores) else 0.0
        left_spatial = float(boundary_spatial[start - 1]) if start > 0 and len(boundary_spatial) else 0.0
        right_spatial = float(boundary_spatial[stop - 1]) if stop < len(fine_counts) and len(boundary_spatial) else 0.0
        left_count = float(boundary_count[start - 1]) if start > 0 and len(boundary_count) else 0.0
        right_count = float(boundary_count[stop - 1]) if stop < len(fine_counts) and len(boundary_count) else 0.0
        item["spatial_change_cdf_min"] = item["mass_weight_cdf_min"]
        item["spatial_change_cdf_max"] = item["mass_weight_cdf_max"]
        item["spatial_change_weight"] = item["mass_weight"]
        item["spatial_change_left_score"] = left_score
        item["spatial_change_right_score"] = right_score
        item["spatial_change_boundary_score"] = max(left_score, right_score)
        item["spatial_change_left_spatial_score"] = left_spatial
        item["spatial_change_right_spatial_score"] = right_spatial
        item["spatial_change_left_count_score"] = left_count
        item["spatial_change_right_count_score"] = right_count
        item["spatial_change_metric"] = "hellinger+log-count-rate"
        item["spatial_change_grid_shape"] = list(grid_shape)
        item["spatial_change_min_atoms"] = float(args.spatial_cdf_min_atoms)
        item["spatial_change_spatial_weight"] = float(args.spatial_cdf_spatial_weight)
        item["spatial_change_count_weight"] = float(args.spatial_cdf_count_weight)
        item["spatial_change_selection"] = "adjacent-agglomerative-hellinger+count-rate"
    return ranges


def atom_count_cdf_ranges(masses: np.ndarray, ref: pp.ReferenceStats, args: argparse.Namespace) -> list[dict[str, Any]]:
    quantiles = np.linspace(0.0, 1.0, args.ranges + 1, dtype=np.float64)
    raw_edges = np.quantile(masses, quantiles).astype(np.float64)
    raw_edges[0] = float(ref.mass_range[0])
    raw_edges[-1] = float(ref.mass_range[1])
    edges = enforce_minimum_edge_spacing(raw_edges, float(args.min_mass_bin_width))

    ranges: list[dict[str, Any]] = []
    for i in range(args.ranges):
        lo = float(edges[i])
        hi = float(edges[i + 1])
        if i == args.ranges - 1:
            selected = (masses >= lo) & (masses <= hi)
        else:
            selected = (masses >= lo) & (masses < hi)
        ranges.append(
            {
                "id": f"range_{i + 1:02d}",
                "mass": float((lo + hi) * 0.5),
                "mass_min": lo,
                "mass_max": hi,
                "mass_width": float(hi - lo),
                "atom_count": int(selected.sum()),
                "quantile_min": float(quantiles[i]),
                "quantile_max": float(quantiles[i + 1]),
                "mass_bin_cdf": "atom-count",
                "min_mass_bin_width": float(args.min_mass_bin_width),
            }
        )
    return ranges


def cdf_ranges(path: Path, ref: pp.ReferenceStats, args: argparse.Namespace) -> list[dict[str, Any]]:
    masses = load_mass_values(path, args.chunk_atoms)
    if len(masses) == 0:
        raise ValueError(f"No finite mass values found in {path}")
    if args.mass_bin_cdf == "atom-count":
        return atom_count_cdf_ranges(masses, ref, args)
    if args.mass_bin_cdf == "log-count":
        return log_count_cdf_ranges(masses, ref, args)
    if args.mass_bin_cdf == "spatial-change":
        return spatial_change_cdf_ranges(path, masses, ref, args)
    return derivative_cdf_ranges(masses, ref, args, log_counts=args.mass_bin_cdf == "log-derivative")


def import_torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    return torch, nn, F


def half_bytes(array: np.ndarray) -> bytes:
    return np.asarray(array, dtype=np.float16).astype("<f2", copy=False).tobytes()


def float32_bytes(array: np.ndarray) -> bytes:
    return np.asarray(array, dtype="<f4").tobytes()


def uint32_bytes(array: np.ndarray) -> bytes:
    return np.asarray(array, dtype="<u4").tobytes()


def uint8_bytes(array: np.ndarray) -> bytes:
    return np.asarray(array, dtype=np.uint8).tobytes()


def int8_bytes(array: np.ndarray) -> bytes:
    return np.asarray(array, dtype=np.int8).tobytes()


def maybe_compress(body: bytes, enabled: bool, level: int = 6) -> tuple[bytes, str | None, int]:
    if not enabled or len(body) < 1024:
        return body, None, len(body)
    compressed = zlib.compress(body, level)
    if len(compressed) >= len(body):
        return body, None, len(body)
    return compressed, "zlib", len(body)


def stored_size(body: bytes, enabled: bool) -> int:
    return len(maybe_compress(body, enabled)[0])


def add_array(
    arrays: list[dict[str, Any]],
    chunks: list[bytes],
    name: str,
    dtype: str,
    shape: list[int],
    body: bytes,
    *,
    compress: bool = False,
) -> None:
    body, compression, raw_nbytes = maybe_compress(body, compress)
    offset = sum(len(chunk) for chunk in chunks)
    spec = {"name": name, "dtype": dtype, "shape": shape, "offset": offset, "nbytes": len(body)}
    if compression:
        spec["compression"] = compression
        spec["raw_nbytes"] = raw_nbytes
    arrays.append(spec)
    chunks.append(body)


def compressed_mask(mask: np.ndarray, level: int = 6) -> bytes:
    packed = np.packbits(mask.reshape(-1).astype(np.uint8), bitorder="big")
    return zlib.compress(packed.tobytes(), level)


def teacher_from_counts(counts: np.ndarray) -> tuple[np.ndarray, float]:
    flat = counts.reshape(-1).astype(np.float32, copy=False)
    scale = max(float(flat.max()), 1.0)
    return (flat / scale).astype(np.float32), scale


def normalized_to_counts(values: np.ndarray, scale: float) -> np.ndarray:
    return np.clip(values, 0.0, 1.0) * scale


def build_teachers(
    raw_path: Path,
    ref: pp.ReferenceStats,
    ranges: list[dict[str, Any]],
    base_shape: tuple[int, int, int],
    residual_shape: tuple[int, int, int],
    args: argparse.Namespace,
) -> tuple[np.ndarray, bytes, list[dict[str, Any]], np.ndarray, list[np.ndarray]]:
    param_count = math.prod(base_shape)
    teachers = np.empty((len(ranges), param_count), dtype=np.float32)
    mask_payloads: list[bytes] = []
    mask_chunks: list[dict[str, Any]] = []
    per_bin_masks: list[bytes] = []
    per_bin_chunks: list[dict[str, Any]] = []
    union_mask = np.zeros(param_count, dtype=bool)
    target_scales = np.ones(len(ranges), dtype=np.float32)
    residual_sources: list[np.ndarray] = []
    payload_offset = 0
    residual_enabled = args.residual_cells_per_range > 0

    for i, item in enumerate(ranges):
        base_counts = pp.build_mass_range_density_grid(
            raw_path, ref, base_shape, item["mass_min"], item["mass_max"], args.chunk_atoms
        )
        teacher, scale = teacher_from_counts(base_counts)
        mask = base_counts.reshape(-1) > 0
        union_mask |= mask
        if args.support_mode in {"per-bin", "hybrid", "adaptive"}:
            body = compressed_mask(mask)
            chunk = {
                "offset": 0,
                "nbytes": int(len(body)),
                "raw_nbytes": int(math.ceil(param_count / 8)),
                "occupied": int(mask.sum()),
            }
            item["support_exact_bytes"] = int(len(body))
            if args.support_mode == "per-bin":
                chunk["offset"] = int(payload_offset)
                mask_payloads.append(body)
                mask_chunks.append(chunk)
                payload_offset += len(body)
            else:
                per_bin_masks.append(body)
                per_bin_chunks.append(chunk)
        teachers[i] = teacher
        target_scales[i] = scale
        item["target_scale"] = float(scale)
        item["support_density_floor"] = float(min(1.0, 1.0 / max(scale, 1e-6)))
        item["density_mode"] = "linear"
        item["base_occupied_cells"] = int(mask.sum())

        if residual_enabled:
            if args.residual_mode == "base-excess":
                residual = np.maximum(base_counts.reshape(-1) - float(args.residual_base_floor), 0.0).astype(np.float32)
            else:
                high_counts = pp.build_mass_range_density_grid(
                    raw_path, ref, residual_shape, item["mass_min"], item["mass_max"], args.chunk_atoms
                ).reshape(-1)
                up = np.repeat(np.repeat(np.repeat(base_counts / 8.0, 2, axis=0), 2, axis=1), 2, axis=2).reshape(-1)
                residual = np.maximum(high_counts - up, 0.0).astype(np.float32)
            residual_sum = float(residual.sum())
        else:
            residual = np.zeros(0, dtype=np.float32)
            residual_sum = 0.0
        item["residual_source_atoms"] = residual_sum
        item["residual_source_fraction"] = float(residual_sum / max(float(item.get("atom_count", 0.0)), 1.0))
        residual_sources.append(residual)
        print(
            f"  teacher {i + 1:02d}/{len(ranges)} {item['mass_min']:.4f}-{item['mass_max']:.4f} "
            f"atoms={item['atom_count']:,} base_occ={int(mask.sum()):,} residual_sum={residual_sum:.1f}",
            flush=True,
        )

    if args.support_mode == "union":
        body = compressed_mask(union_mask)
        mask_payloads = [body]
        mask_chunks = [
            {
                "offset": 0,
                "nbytes": int(len(body)),
                "raw_nbytes": int(math.ceil(param_count / 8)),
                "occupied": int(union_mask.sum()),
            }
        ]
        for item in ranges:
            item["support_chunk_index"] = 0
    elif args.support_mode == "per-bin":
        for i, item in enumerate(ranges):
            item["support_chunk_index"] = i
            item["support_mode"] = "per-bin"
    elif args.support_mode == "hybrid":
        union_body = compressed_mask(union_mask)
        mask_payloads = [union_body]
        mask_chunks = [
            {
                "offset": 0,
                "nbytes": int(len(union_body)),
                "raw_nbytes": int(math.ceil(param_count / 8)),
                "occupied": int(union_mask.sum()),
                "shared": True,
            }
        ]
        payload_offset = len(union_body)
        exact_count = 0
        for i, item in enumerate(ranges):
            if int(item["base_occupied_cells"]) <= int(args.hybrid_support_max_cells):
                body = per_bin_masks[i]
                chunk = dict(per_bin_chunks[i])
                chunk["offset"] = int(payload_offset)
                chunk["range_index"] = i
                mask_payloads.append(body)
                mask_chunks.append(chunk)
                payload_offset += len(body)
                item["support_chunk_index"] = len(mask_chunks) - 1
                item["support_mode"] = "per-bin"
                exact_count += 1
            else:
                item["support_chunk_index"] = 0
                item["support_mode"] = "union"
        for item in ranges:
            item["hybrid_support_max_cells"] = int(args.hybrid_support_max_cells)
        print(
            f"Hybrid support: exact per-bin masks for {exact_count}/{len(ranges)} ranges "
            f"<= {args.hybrid_support_max_cells:,} occupied cells",
            flush=True,
        )
    elif args.support_mode == "adaptive":
        union_body = compressed_mask(union_mask)
        union_occupied = int(union_mask.sum())
        exact_total = int(sum(len(body) for body in per_bin_masks))
        support_target_mb = float(getattr(args, "_support_target_mb", 10.0))
        if float(args.adaptive_support_budget_mb) > 0:
            budget_bytes = int(float(args.adaptive_support_budget_mb) * 1024 * 1024)
        else:
            budget_bytes = int(
                max(0.0, support_target_mb * float(args.adaptive_support_budget_fraction)) * 1024 * 1024
            )
        budget_bytes = max(0, budget_bytes)

        if exact_total <= budget_bytes:
            mask_payloads = []
            mask_chunks = []
            payload_offset = 0
            for i, item in enumerate(ranges):
                body = per_bin_masks[i]
                chunk = dict(per_bin_chunks[i])
                chunk["offset"] = int(payload_offset)
                chunk["range_index"] = i
                mask_payloads.append(body)
                mask_chunks.append(chunk)
                payload_offset += len(body)
                item["support_chunk_index"] = i
                item["support_mode"] = "per-bin"
                item["adaptive_support_score"] = None
            print(
                f"Adaptive support: all {len(ranges)}/{len(ranges)} bins use exact per-bin masks "
                f"({exact_total / (1024 * 1024):.2f} MB <= {budget_bytes / (1024 * 1024):.2f} MB budget)",
                flush=True,
            )
        else:
            selected: set[int] = set()
            remaining = max(0, budget_bytes - len(union_body))
            candidates: list[tuple[float, int, int]] = []
            for i, item in enumerate(ranges):
                occ = max(int(item["base_occupied_cells"]), 1)
                cost = max(len(per_bin_masks[i]), 1)
                atom_count = max(float(item.get("atom_count", 0.0)), 1.0)
                false_ratio = max(float(union_occupied) / float(occ) - 1.0, 0.0)
                residual_fraction = max(float(item.get("residual_source_fraction", 0.0)), 0.0)
                density_scale = max(float(item.get("target_scale", 1.0)), 1.0)
                hotspot_boost = 1.0 + min(3.0, 6.0 * residual_fraction) + min(2.0, math.log1p(density_scale) / 4.0)
                score = atom_count * math.log1p(false_ratio) * hotspot_boost / cost
                item["adaptive_support_score"] = float(score)
                item["adaptive_support_false_ratio"] = float(false_ratio)
                candidates.append((score, cost, i))
            for _, cost, i in sorted(candidates, reverse=True):
                if cost <= remaining:
                    selected.add(i)
                    remaining -= cost

            mask_payloads = [union_body]
            mask_chunks = [
                {
                    "offset": 0,
                    "nbytes": int(len(union_body)),
                    "raw_nbytes": int(math.ceil(param_count / 8)),
                    "occupied": union_occupied,
                    "shared": True,
                }
            ]
            payload_offset = len(union_body)
            for i, item in enumerate(ranges):
                if i in selected:
                    body = per_bin_masks[i]
                    chunk = dict(per_bin_chunks[i])
                    chunk["offset"] = int(payload_offset)
                    chunk["range_index"] = i
                    mask_payloads.append(body)
                    mask_chunks.append(chunk)
                    payload_offset += len(body)
                    item["support_chunk_index"] = len(mask_chunks) - 1
                    item["support_mode"] = "per-bin"
                else:
                    item["support_chunk_index"] = 0
                    item["support_mode"] = "union"
            print(
                f"Adaptive support: exact per-bin masks for {len(selected)}/{len(ranges)} bins, "
                f"support={payload_offset / (1024 * 1024):.2f} MB, "
                f"budget={budget_bytes / (1024 * 1024):.2f} MB, union={len(union_body) / (1024 * 1024):.2f} MB",
                flush=True,
            )
        for item in ranges:
            item["support_policy"] = "adaptive"
            item["adaptive_support_budget_mb"] = float(budget_bytes / (1024 * 1024))
            item["union_support_occupied_cells"] = union_occupied
    return teachers, b"".join(mask_payloads), mask_chunks, target_scales, residual_sources


def feature_values(ranges: list[dict[str, Any]], mode: str) -> tuple[np.ndarray, dict[str, float]]:
    centers = np.array([item["mass"] for item in ranges], dtype=np.float32)
    widths = np.array([item["mass_max"] - item["mass_min"] for item in ranges], dtype=np.float32)
    stats = {
        "center_min": float(centers.min()),
        "center_max": float(centers.max()),
        "width_min": float(widths.min()),
        "width_max": float(widths.max()),
    }
    center_norm = ((centers - stats["center_min"]) / max(stats["center_max"] - stats["center_min"], 1e-6) * 2.0 - 1.0).astype(np.float32)
    if mode == "center":
        return center_norm[:, None], stats
    width_norm = ((widths - stats["width_min"]) / max(stats["width_max"] - stats["width_min"], 1e-6) * 2.0 - 1.0).astype(np.float32)
    return np.stack([center_norm, width_norm], axis=1).astype(np.float32), stats


def encoded_features(base: np.ndarray, freqs: list[float]) -> np.ndarray:
    parts = [base.astype(np.float32)]
    for freq in freqs:
        parts.append(np.sin(np.pi * base * freq).astype(np.float32))
    for freq in freqs:
        parts.append(np.cos(np.pi * base * freq).astype(np.float32))
    return np.concatenate(parts, axis=1).astype(np.float32)


def train_coeff_net(
    features: np.ndarray,
    coeff_norm: np.ndarray,
    rank: int,
    width: int,
    epochs: int,
    torch: Any,
    nn: Any,
    F: Any,
    device: str,
):
    class CoeffNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(features.shape[1], width),
                nn.SiLU(),
                nn.Linear(width, width),
                nn.SiLU(),
                nn.Linear(width, rank),
            )

        def forward(self, x):
            return self.net(x)

    model = CoeffNet().to(device)
    x_t = torch.from_numpy(features).to(device)
    y_t = torch.from_numpy(coeff_norm.astype(np.float32)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-6)
    for _ in range(epochs):
        pred = model(x_t)
        loss = F.mse_loss(pred, y_t)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred_norm = model(x_t).detach().cpu().numpy().astype(np.float32)
    return model, pred_norm, float(loss.detach().cpu().item())


def quantize_basis_int8(basis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scale = np.maximum(np.max(np.abs(basis), axis=1) / 127.0, 1e-12).astype(np.float32)
    quantized = np.rint(basis / scale[:, None]).clip(-127, 127).astype(np.int8)
    return quantized, scale


def evaluate_reconstruction(
    predicted: np.ndarray,
    teachers: np.ndarray,
    ranges: list[dict[str, Any]],
    samples: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    log_mse: list[float] = []
    pos_mse: list[float] = []
    rel_l1: list[float] = []
    param_count = teachers.shape[1]
    for i in range(len(ranges)):
        target = teachers[i]
        pred = np.clip(predicted[i], 0.0, 1.0)
        pos = np.flatnonzero(target > 0)
        random_count = max(1, samples // 2)
        pos_count = max(1, samples - random_count)
        random_idx = rng.integers(0, param_count, size=random_count, dtype=np.int64)
        pos_idx = rng.choice(pos, size=pos_count, replace=len(pos) < pos_count).astype(np.int64) if len(pos) else np.empty(0, dtype=np.int64)
        idx = np.concatenate([random_idx, pos_idx])
        p = pred[idx]
        t = target[idx]
        log_mse.append(float(np.mean((p - t) ** 2)))
        if len(pos_idx):
            pos_mse.append(float(np.mean((p[random_count:] - t[random_count:]) ** 2)))
        scale = float(ranges[i]["target_scale"])
        pc = normalized_to_counts(p, scale)
        tc = normalized_to_counts(t, scale)
        rel_l1.append(float(np.abs(pc - tc).sum() / max(tc.sum(), 1e-6)))
    mse = float(np.mean(log_mse))
    positive_mse = float(np.mean(pos_mse)) if pos_mse else None
    return {
        "log_mse": mse,
        "positive_log_mse": positive_mse,
        "density_mse": mse,
        "positive_density_mse": positive_mse,
        "relative_count_l1": float(np.mean(rel_l1)),
    }


def choose_rank(target_mb: float, param_count: int, support_bytes: int, residual_bytes: int, width: int, input_dim: int) -> int:
    target = int(target_mb * 1024 * 1024)
    mean_bytes = param_count
    coeff_net_bytes = ((width * input_dim + width) + (width * width + width)) * 4
    fixed = support_bytes + residual_bytes + mean_bytes + coeff_net_bytes + 64_000
    available = max(0, target - fixed)
    return max(1, int(available // param_count))


def residual_array_bytes(range_count: int, per_range: int) -> int:
    return range_count * per_range * 5 + range_count * 8


def residual_budget_for_target(target_mb: float, args: argparse.Namespace) -> int:
    return max(0, int(args.residual_cells_per_range))


def build_residual_arrays(
    ranges: list[dict[str, Any]],
    residual_sources: list[np.ndarray],
    target_mb: float,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    per_range = residual_budget_for_target(target_mb, args)
    range_count = len(residual_sources)
    for item in ranges:
        item["residual_mode"] = args.residual_mode if per_range > 0 else "none"
        item["residual_base_floor"] = float(args.residual_base_floor)
        item["residual_cells"] = 0
        item["residual_selected_atoms"] = 0.0
        item["residual_atom_fraction"] = 0.0
    if per_range <= 0:
        return (
            np.zeros((range_count, 0), dtype=np.uint32),
            np.zeros((range_count, 0), dtype=np.uint8),
            np.zeros(range_count, dtype=np.uint32),
            np.ones(range_count, dtype=np.float32),
        )
    indices = np.zeros((range_count, per_range), dtype=np.uint32)
    values = np.zeros((range_count, per_range), dtype=np.uint8)
    counts = np.zeros(range_count, dtype=np.uint32)
    scales = np.ones(range_count, dtype=np.float32)
    for i, residual in enumerate(residual_sources):
        nz = np.flatnonzero(residual > 0)
        if len(nz) == 0:
            continue
        take = min(per_range, len(nz))
        if take < len(nz):
            order = np.argpartition(residual[nz], -take)[-take:]
            chosen = nz[order]
            chosen = chosen[np.argsort(chosen)]
        else:
            chosen = nz
        vals = residual[chosen].astype(np.float32)
        scale = max(float(vals.max()), 1e-6)
        indices[i, :take] = chosen.astype(np.uint32)
        values[i, :take] = np.rint(vals / scale * 255.0).clip(1, 255).astype(np.uint8)
        counts[i] = take
        scales[i] = scale
        selected_atoms = float(vals.sum())
        ranges[i]["residual_cells"] = int(take)
        ranges[i]["residual_selected_atoms"] = selected_atoms
        ranges[i]["residual_atom_fraction"] = float(selected_atoms / max(float(ranges[i].get("atom_count", 0.0)), 1.0))
        ranges[i]["residual_value_scale"] = float(scale)
    return indices, values, counts, scales


def low_rank_basis(centered: np.ndarray, rank: int) -> np.ndarray:
    if rank <= 0:
        return np.empty((0, centered.shape[1]), dtype=np.float32)
    gram = (centered @ centered.T).astype(np.float64)
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1]
    basis_rows: list[np.ndarray] = []
    for idx in order:
        if len(basis_rows) >= rank:
            break
        singular = math.sqrt(max(float(values[idx]), 0.0))
        if singular <= 1e-8:
            continue
        row = (vectors[:, idx].astype(np.float32) @ centered) / singular
        norm = np.linalg.norm(row)
        if norm > 1e-8:
            row = row / norm
        basis_rows.append(row.astype(np.float32))
    if not basis_rows:
        return np.zeros((1, centered.shape[1]), dtype=np.float32)
    return np.stack(basis_rows).astype(np.float32)


def estimate_artifact_size(
    *,
    target_mb: float,
    mean_q: np.ndarray,
    basis: np.ndarray,
    support_payload: bytes,
    residual_indices: np.ndarray,
    residual_values: np.ndarray,
    residual_counts: np.ndarray,
    residual_scales: np.ndarray,
    width: int,
    input_dim: int,
    rank: int,
    compress_arrays: bool,
) -> int:
    basis_q, basis_scale = quantize_basis_int8(basis[:rank])
    coeff_net_bytes = (
        (width * input_dim + width)
        + (width * width + width)
        + (rank * width + rank)
        + rank * 2
    ) * 4
    size = 12 + 180_000
    size += stored_size(uint8_bytes(mean_q), compress_arrays)
    size += stored_size(int8_bytes(basis_q), compress_arrays)
    size += stored_size(float32_bytes(basis_scale), compress_arrays)
    size += coeff_net_bytes
    size += len(support_payload)
    size += stored_size(uint32_bytes(residual_indices), compress_arrays)
    size += stored_size(uint8_bytes(residual_values), compress_arrays)
    size += stored_size(uint32_bytes(residual_counts), compress_arrays)
    size += stored_size(float32_bytes(residual_scales), compress_arrays)
    return size


def choose_rank_by_estimated_size(
    *,
    target_mb: float,
    mean_q: np.ndarray,
    basis: np.ndarray,
    support_payload: bytes,
    residual_indices: np.ndarray,
    residual_values: np.ndarray,
    residual_counts: np.ndarray,
    residual_scales: np.ndarray,
    width: int,
    input_dim: int,
    compress_arrays: bool,
) -> int:
    target = int(target_mb * 1024 * 1024)
    best_rank = 1
    best_size = None
    for rank in range(1, basis.shape[0] + 1):
        size = estimate_artifact_size(
            target_mb=target_mb,
            mean_q=mean_q,
            basis=basis,
            support_payload=support_payload,
            residual_indices=residual_indices,
            residual_values=residual_values,
            residual_counts=residual_counts,
            residual_scales=residual_scales,
            width=width,
            input_dim=input_dim,
            rank=rank,
            compress_arrays=compress_arrays,
        )
        if size <= target * 1.02:
            best_rank = rank
            best_size = size
        elif best_size is not None:
            break
    return best_rank


def write_artifact(
    path: Path,
    *,
    ref: pp.ReferenceStats,
    ranges: list[dict[str, Any]],
    base_shape: tuple[int, int, int],
    residual_shape: tuple[int, int, int],
    target_mb: float,
    support_payload: bytes,
    support_chunks: list[dict[str, Any]],
    mean_q: np.ndarray,
    basis: np.ndarray,
    coeff_mean: np.ndarray,
    coeff_std: np.ndarray,
    coeff_model: Any,
    input_mode: str,
    feature_stats: dict[str, float],
    feature_freqs: list[float],
    metrics: dict[str, Any],
    mode_trials: dict[str, Any],
    residual_indices: np.ndarray,
    residual_values: np.ndarray,
    residual_counts: np.ndarray,
    residual_scales: np.ndarray,
    density_mode: str,
    support_mode: str,
    compress_arrays: bool,
) -> int:
    arrays: list[dict[str, Any]] = []
    chunks: list[bytes] = []
    add_array(arrays, chunks, "mean", "uint8", [int(mean_q.size)], uint8_bytes(mean_q), compress=compress_arrays)
    basis_q, basis_scale = quantize_basis_int8(basis)
    add_array(arrays, chunks, "basis", "int8", [int(basis_q.shape[0]), int(basis_q.shape[1])], int8_bytes(basis_q), compress=compress_arrays)
    add_array(arrays, chunks, "basis_scale", "float32", [int(basis_scale.size)], float32_bytes(basis_scale), compress=compress_arrays)
    add_array(arrays, chunks, "coeff_mean", "float32", [int(coeff_mean.size)], float32_bytes(coeff_mean), compress=compress_arrays)
    add_array(arrays, chunks, "coeff_std", "float32", [int(coeff_std.size)], float32_bytes(coeff_std), compress=compress_arrays)
    add_array(arrays, chunks, "support_payload", "uint8", [len(support_payload)], support_payload)
    add_array(arrays, chunks, "residual_indices", "uint32", list(residual_indices.shape), uint32_bytes(residual_indices), compress=compress_arrays)
    add_array(arrays, chunks, "residual_values", "uint8", list(residual_values.shape), uint8_bytes(residual_values), compress=compress_arrays)
    add_array(arrays, chunks, "residual_counts", "uint32", [int(residual_counts.size)], uint32_bytes(residual_counts), compress=compress_arrays)
    add_array(arrays, chunks, "residual_scales", "float32", [int(residual_scales.size)], float32_bytes(residual_scales), compress=compress_arrays)

    state = {key: value.detach().cpu().numpy() for key, value in coeff_model.state_dict().items()}
    for key in ("net.0.weight", "net.0.bias", "net.2.weight", "net.2.bias", "net.4.weight", "net.4.bias"):
        add_array(arrays, chunks, f"coeff.{key}", "float32", list(state[key].shape), float32_bytes(state[key]), compress=compress_arrays)

    header = {
        "version": 1,
        "kind": "cdf_grid_v2",
        "dataset_bounds": pp.json_ready(ref.bounds),
        "dataset_mass_range": pp.json_ready(ref.mass_range),
        "ranges": [{k: pp.json_ready(v) for k, v in item.items()} for item in ranges],
        "support_compression": "zlib",
        "support_mode": support_mode,
        "support_chunks": support_chunks,
        "settings": pp.json_ready(
            {
                "base_grid_shape": list(base_shape),
                "residual_grid_shape": list(residual_shape),
                "teacher_param_count": int(mean_q.size),
                "density_mode": density_mode,
                "mean_dtype": "uint8_normalized_density",
                "basis_dtype": "int8",
                "hyper_rank": int(basis.shape[0]),
                "hyper_width": int(state["net.0.bias"].shape[0]),
                "input_mode": input_mode,
                "feature_stats": feature_stats,
                "feature_freqs": feature_freqs,
                "target_artifact_mb": target_mb,
                "adaptive_support_budget_mb": float(ranges[0].get("adaptive_support_budget_mb", 0.0)) if ranges else 0.0,
                "adaptive_support_exact_bins": int(sum(1 for item in ranges if item.get("support_mode") == "per-bin")),
                "mass_bin_cdf": ranges[0].get("mass_bin_cdf") if ranges else None,
                "fine_mass_bin_width": ranges[0].get("fine_mass_bin_width") if ranges else None,
                "spatial_change_metric": ranges[0].get("spatial_change_metric") if ranges else None,
                "spatial_change_selection": ranges[0].get("spatial_change_selection") if ranges else None,
                "spatial_change_grid_shape": ranges[0].get("spatial_change_grid_shape") if ranges else None,
                "spatial_change_min_atoms": ranges[0].get("spatial_change_min_atoms") if ranges else None,
                "spatial_change_spatial_weight": ranges[0].get("spatial_change_spatial_weight") if ranges else None,
                "spatial_change_count_weight": ranges[0].get("spatial_change_count_weight") if ranges else None,
                "min_mass_bin_width": float(min(item["mass_width"] for item in ranges)) if ranges else None,
                "requested_min_mass_bin_width": float(ranges[0].get("min_mass_bin_width", 0.0)) if ranges else None,
                "min_range_atom_fraction": float(ranges[0].get("min_range_atom_fraction", 0.0)) if ranges else 0.0,
                "min_range_atom_count": int(ranges[0].get("min_range_atom_count", 0)) if ranges else 0,
                "residual_cells_per_range": int(residual_indices.shape[1]),
                "residual_mode": ranges[0].get("residual_mode", "none") if ranges else "none",
                "residual_base_floor": float(ranges[0].get("residual_base_floor", 1.0)) if ranges else 1.0,
                "metrics": metrics,
                "mode_trials": mode_trials,
            }
        ),
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


def method_key(target_mb: float) -> str:
    suffix = f"{target_mb:g}mb".replace(".", "p")
    return f"cdf_v2_linear_64_{suffix}"


def method_label(target_mb: float) -> str:
    return f"CDF grid v2 linear 64-bin {target_mb:g}MB"


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    target_mbs = target_values(args.targets_mb)
    args._support_target_mb = max(target_mbs) if target_mbs else 10.0
    manifest_path = args.out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset = find_dataset(manifest, args.dataset_filter)
    if args.auto_grid:
        reference = find_dataset(manifest, args.reference_dataset_filter)
        base_shape = auto_grid_shape(dataset, reference, parse_shape(args.reference_grid_shape))
        residual_shape = tuple(int(value * 2) for value in base_shape)
    else:
        base_shape = parse_shape(args.base_grid_shape)
        residual_shape = parse_shape(args.residual_grid_shape)
    artifact_residual_shape = base_shape if args.residual_mode == "base-excess" else residual_shape
    raw_path = Path(dataset["raw_path"])
    ds_dir = args.out_dir / "datasets" / dataset["id"]
    ref = load_reference(dataset, ds_dir)
    torch, nn, F = import_torch()
    device = "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"

    ranges = cdf_ranges(raw_path, ref, args)
    mass_widths = [item["mass_width"] for item in ranges]
    print(
        f"Training CDF grid v2 for {raw_path.name}: {args.ranges} CDF ranges, "
        f"density={args.density_mode}, mass_cdf={args.mass_bin_cdf}",
        flush=True,
    )
    print(
        f"Mass bins: min_width={min(mass_widths):.4f} Da requested_min={args.min_mass_bin_width:.4f} Da",
        flush=True,
    )
    print(
        f"Base grid={shape_label(base_shape)}, residual grid={shape_label(artifact_residual_shape)} "
        f"mode={args.residual_mode}, targets={target_mbs}",
        flush=True,
    )
    teachers, support_payload, support_chunks, scales, residual_sources = build_teachers(
        raw_path, ref, ranges, base_shape, artifact_residual_shape, args
    )
    print(f"Compressed supports: {len(support_payload) / (1024 * 1024):.2f} MB ({args.support_mode})", flush=True)

    mean = teachers.mean(axis=0).astype(np.float32)
    mean_q = np.rint(np.clip(mean, 0.0, 1.0) * 255.0).astype(np.uint8)
    centered = teachers - (mean_q.astype(np.float32) / 255.0)[None, :]
    freqs = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    feature_options = {}
    for mode in ("center", "range"):
        base, stats = feature_values(ranges, mode)
        feature_options[mode] = (encoded_features(base, freqs), stats)
    max_input_dim = max(features.shape[1] for features, _ in feature_options.values())
    max_needed_rank = min(len(ranges) - 1, max(1, args.max_rank))
    print(f"Computing rank-{max_needed_rank} teacher basis...", flush=True)
    vt = low_rank_basis(centered, max_needed_rank)

    out_dir = ds_dir / "cdf_v2"
    for target_mb in target_mbs:
        residual_indices, residual_values, residual_counts, residual_scales = build_residual_arrays(
            ranges, residual_sources, target_mb, args
        )
        rank = choose_rank_by_estimated_size(
            target_mb=target_mb,
            mean_q=mean_q,
            basis=vt,
            support_payload=support_payload,
            residual_indices=residual_indices,
            residual_values=residual_values,
            residual_counts=residual_counts,
            residual_scales=residual_scales,
            width=args.hyper_width,
            input_dim=max_input_dim,
            compress_arrays=args.compress_arrays,
        )
        basis = vt[:rank].astype(np.float32)
        coeff = centered @ basis.T
        coeff_mean = coeff.mean(axis=0).astype(np.float32)
        coeff_std = np.maximum(coeff.std(axis=0).astype(np.float32), 1e-6)
        coeff_norm = ((coeff - coeff_mean[None, :]) / coeff_std[None, :]).astype(np.float32)
        mode_trials: dict[str, Any] = {}
        best: dict[str, Any] | None = None
        print(f"Training {target_mb:g} MB v2 hypernetwork: rank={rank}, residual/range={residual_indices.shape[1]}", flush=True)
        for mode, (features, stats) in feature_options.items():
            model, pred_norm, coeff_loss = train_coeff_net(
                features, coeff_norm, rank, args.hyper_width, args.hyper_epochs, torch, nn, F, device
            )
            pred_coeff = pred_norm * coeff_std[None, :] + coeff_mean[None, :]
            basis_q, basis_scale = quantize_basis_int8(basis)
            basis_deq = basis_q.astype(np.float32) * basis_scale[:, None]
            predicted = (mean_q.astype(np.float32) / 255.0)[None, :] + pred_coeff @ basis_deq
            metrics = evaluate_reconstruction(
                predicted,
                teachers,
                ranges,
                args.metric_samples,
                args.seed + int(target_mb * 100) + (0 if mode == "center" else 17),
            )
            rel_weight = float(np.linalg.norm(predicted - teachers) / max(np.linalg.norm(teachers - mean[None, :]), 1e-6))
            trial = {
                "input_mode": mode,
                "coeff_loss": coeff_loss,
                "relative_teacher_weight_error": rel_weight,
                **metrics,
            }
            mode_trials[mode] = trial
            print(
                f"  {mode:6s}: rel_weight={rel_weight:.5f} density_mse={metrics['density_mse']:.6f} "
                f"pos_mse={metrics['positive_density_mse']:.6f} count_l1={metrics['relative_count_l1']:.4f}",
                flush=True,
            )
            if best is None or metrics["log_mse"] < best["metrics"]["log_mse"]:
                best = {"mode": mode, "stats": stats, "model": model, "metrics": metrics, "rel_weight": rel_weight}
        assert best is not None
        key = method_key(target_mb)
        artifact_path = out_dir / f"{key}.bin"
        final_metrics = {
            **best["metrics"],
            "relative_teacher_weight_error": best["rel_weight"],
            "hyper_rank": rank,
            "density_mode": "linear",
        }
        label = method_label(target_mb)
        if args.support_mode == "per-bin":
            label += " per-bin support"
        elif args.support_mode == "hybrid":
            label += " hybrid support"
        elif args.support_mode == "adaptive":
            label += " adaptive support"
        compressed_size = write_artifact(
            artifact_path,
            ref=ref,
            ranges=ranges,
            base_shape=base_shape,
            residual_shape=artifact_residual_shape,
            target_mb=target_mb,
            support_payload=support_payload,
            support_chunks=support_chunks,
            mean_q=mean_q,
            basis=basis,
            coeff_mean=coeff_mean,
            coeff_std=coeff_std,
            coeff_model=best["model"],
            input_mode=best["mode"],
            feature_stats=best["stats"],
            feature_freqs=freqs,
            metrics=final_metrics,
            mode_trials=mode_trials,
            residual_indices=residual_indices,
            residual_values=residual_values,
            residual_counts=residual_counts,
            residual_scales=residual_scales,
            density_mode="linear",
            support_mode=args.support_mode,
            compress_arrays=args.compress_arrays,
        )
        method = {
            "label": label,
            "method_label": label,
            "available": True,
            "frontend_generated": True,
            "cdf_v2": True,
            "compressed_size_bytes": int(compressed_size),
            "compression_ratio": float(ref.raw_size_bytes / max(compressed_size, 1)),
            "points": int(sum(item["atom_count"] for item in ranges)),
            "artifact": str(artifact_path),
            "artifact_endpoint": f"/api/artifact/{dataset['id']}/{key}",
            "preprocess_sec": float(time.perf_counter() - started),
            "metrics": pp.json_ready(final_metrics),
            "settings": pp.json_ready(
                {
                    "base_grid_shape": list(base_shape),
                    "residual_grid_shape": list(artifact_residual_shape),
                    "residual_mode": args.residual_mode if int(residual_indices.shape[1]) > 0 else "none",
                    "residual_base_floor": float(args.residual_base_floor),
                    "support_compressed_mb": len(support_payload) / (1024 * 1024),
                    "support_mode": args.support_mode,
                    "hybrid_support_max_cells": int(args.hybrid_support_max_cells) if args.support_mode == "hybrid" else None,
                    "adaptive_support_budget_mb": float(ranges[0].get("adaptive_support_budget_mb", 0.0)) if ranges else 0.0,
                    "adaptive_support_exact_bins": int(sum(1 for item in ranges if item.get("support_mode") == "per-bin")),
                    "array_compression": "zlib" if args.compress_arrays else "none",
                    "residual_cells_per_range": int(residual_indices.shape[1]),
                    "density_mode": "linear",
                    "input_mode": best["mode"],
                    "hyper_rank": rank,
                    "target_artifact_mb": target_mb,
                    "mass_bin_cdf": ranges[0].get("mass_bin_cdf") if ranges else None,
                    "fine_mass_bin_width": ranges[0].get("fine_mass_bin_width") if ranges else None,
                    "spatial_change_metric": ranges[0].get("spatial_change_metric") if ranges else None,
                    "spatial_change_selection": ranges[0].get("spatial_change_selection") if ranges else None,
                    "spatial_change_grid_shape": ranges[0].get("spatial_change_grid_shape") if ranges else None,
                    "spatial_change_min_atoms": ranges[0].get("spatial_change_min_atoms") if ranges else None,
                    "spatial_change_spatial_weight": ranges[0].get("spatial_change_spatial_weight") if ranges else None,
                    "spatial_change_count_weight": ranges[0].get("spatial_change_count_weight") if ranges else None,
                    "min_mass_bin_width": float(min(item["mass_width"] for item in ranges)) if ranges else None,
                    "requested_min_mass_bin_width": float(args.min_mass_bin_width),
                    "min_range_atom_fraction": float(args.min_range_atom_fraction),
                    "min_range_atom_count": int(ranges[0].get("min_range_atom_count", 0)) if ranges else 0,
                    "mode_trials": mode_trials,
                }
            ),
            "ranges": [{k: pp.json_ready(v) for k, v in item.items()} for item in ranges],
            "notes": f"CDF grid v2 using linear count-density targets, {args.support_mode} support mask, uint8 mean grid, int8 SVD basis, and sparse base-excess residual cells.",
        }
        dataset.setdefault("methods", {})[key] = method
        manifest.setdefault("methods", {})[key] = method["label"]
        print(
            f"{key}: {pp.mb(compressed_size):.2f} MB ratio={method['compression_ratio']:.2f}x "
            f"mode={best['mode']} rank={rank} density_mse={final_metrics['density_mse']:.6f} "
            f"pos_mse={final_metrics['positive_density_mse']:.6f} count_l1={final_metrics['relative_count_l1']:.4f}",
            flush=True,
        )

    pp.write_json(ds_dir / "dataset_manifest.json", dataset)
    pp.write_json(manifest_path, manifest)
    print(f"Manifest updated: {manifest_path}")
    print(f"Total time: {time.perf_counter() - started:.1f} sec")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
