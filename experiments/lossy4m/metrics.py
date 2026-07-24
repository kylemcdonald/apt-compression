"""Distribution metrics for comparing retained point-cloud previews."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .codec import SPECTRUM_MAX_DA, SPECTRUM_MIN_DA


@dataclass(frozen=True)
class SpatialReference:
    minimum: np.ndarray
    maximum: np.ndarray
    grid_size: int
    voxel_distribution: np.ndarray
    occupied: np.ndarray
    axis_distributions: tuple[np.ndarray, np.ndarray, np.ndarray]


def _normalized_indices(
    values: np.ndarray,
    minimum: np.ndarray,
    maximum: np.ndarray,
    bins: int,
) -> list[np.ndarray]:
    output = []
    for axis in range(values.shape[1]):
        extent = float(maximum[axis] - minimum[axis])
        safe_extent = extent if extent > 0 else 1.0
        indices = np.floor(
            (values[:, axis].astype(np.float64) - minimum[axis])
            / safe_extent * bins,
        ).astype(np.int64)
        output.append(np.clip(indices, 0, bins - 1))
    return output


def _distribution(
    indices: np.ndarray,
    size: int,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    counts = np.bincount(indices, weights=weights, minlength=size).astype(
        np.float64,
    )
    total = counts.sum()
    return counts / total if total else counts


def make_spatial_reference(
    points: np.ndarray,
    *,
    grid_size: int = 32,
    axis_bins: int = 256,
) -> SpatialReference:
    positions = np.asarray(points, dtype=np.float32)[:, :3]
    minimum = positions.min(axis=0).astype(np.float64)
    maximum = positions.max(axis=0).astype(np.float64)
    indices = _normalized_indices(
        positions,
        minimum,
        maximum,
        grid_size,
    )
    voxels = (
        (indices[0] * grid_size + indices[1]) * grid_size + indices[2]
    )
    voxel_distribution = _distribution(voxels, grid_size ** 3)
    axis_indices = _normalized_indices(
        positions,
        minimum,
        maximum,
        axis_bins,
    )
    axis_distributions = tuple(
        _distribution(axis, axis_bins) for axis in axis_indices
    )
    return SpatialReference(
        minimum=minimum,
        maximum=maximum,
        grid_size=grid_size,
        voxel_distribution=voxel_distribution,
        occupied=voxel_distribution > 0,
        axis_distributions=axis_distributions,
    )


def _js_divergence(first: np.ndarray, second: np.ndarray) -> float:
    midpoint = (first + second) * 0.5
    first_mask = first > 0
    second_mask = second > 0
    divergence = 0.5 * np.sum(
        first[first_mask] * np.log2(first[first_mask] / midpoint[first_mask]),
    )
    divergence += 0.5 * np.sum(
        second[second_mask]
        * np.log2(second[second_mask] / midpoint[second_mask]),
    )
    return float(divergence)


def spatial_quality(
    reference: SpatialReference,
    points: np.ndarray,
    *,
    weights: np.ndarray | None = None,
) -> dict:
    positions = np.asarray(points, dtype=np.float32)[:, :3]
    indices = _normalized_indices(
        positions,
        reference.minimum,
        reference.maximum,
        reference.grid_size,
    )
    voxels = (
        (indices[0] * reference.grid_size + indices[1])
        * reference.grid_size
        + indices[2]
    )
    candidate = _distribution(
        voxels,
        reference.grid_size ** 3,
        weights,
    )
    occupied_reference = int(reference.occupied.sum())
    occupied_candidate = candidate > 0
    recall = (
        np.count_nonzero(reference.occupied & occupied_candidate)
        / occupied_reference
        if occupied_reference
        else 1.0
    )
    axis_bins = len(reference.axis_distributions[0])
    axis_indices = _normalized_indices(
        positions,
        reference.minimum,
        reference.maximum,
        axis_bins,
    )
    axis_emd = []
    for source, index in zip(
        reference.axis_distributions,
        axis_indices,
        strict=True,
    ):
        current = _distribution(index, axis_bins, weights)
        axis_emd.append(float(np.mean(np.abs(
            np.cumsum(source) - np.cumsum(current),
        ))))
    return {
        "spatial_js_bits": round(
            _js_divergence(reference.voxel_distribution, candidate),
            9,
        ),
        "spatial_total_variation": round(
            float(0.5 * np.abs(reference.voxel_distribution - candidate).sum()),
            9,
        ),
        "occupied_voxel_recall": round(float(recall), 9),
        "mean_axis_emd": round(float(np.mean(axis_emd)), 9),
    }


def mass_counts(points: np.ndarray, bin_width_da: float) -> np.ndarray:
    bin_count = int(round(
        (SPECTRUM_MAX_DA - SPECTRUM_MIN_DA) / bin_width_da,
    ))
    bins = np.floor(
        (
            np.asarray(points, dtype=np.float32)[:, 3].astype(np.float64)
            - SPECTRUM_MIN_DA
        ) / bin_width_da,
    ).astype(np.int64)
    bins = np.clip(bins, 0, bin_count - 1)
    return np.bincount(bins, minlength=bin_count)


def mass_quality(
    source_counts: np.ndarray,
    points: np.ndarray,
    *,
    bin_width_da: float,
    weights: np.ndarray | None = None,
) -> dict:
    candidate_counts = mass_counts(points, bin_width_da).astype(np.float64)
    if weights is not None:
        bin_count = len(source_counts)
        bins = np.floor(
            (
                np.asarray(points, dtype=np.float32)[:, 3].astype(np.float64)
                - SPECTRUM_MIN_DA
            ) / bin_width_da,
        ).astype(np.int64)
        bins = np.clip(bins, 0, bin_count - 1)
        candidate_counts = np.bincount(
            bins,
            weights=weights,
            minlength=bin_count,
        )
    source = source_counts.astype(np.float64)
    source /= source.sum()
    candidate = candidate_counts.astype(np.float64)
    candidate /= candidate.sum()
    return {
        "mass_js_bits": round(_js_divergence(source, candidate), 9),
        "mass_total_variation": round(
            float(0.5 * np.abs(source - candidate).sum()),
            9,
        ),
    }


def expansion_weights(
    bins: np.ndarray,
    true_counts: np.ndarray,
    stored_counts: np.ndarray,
) -> np.ndarray:
    ratios = np.zeros(len(true_counts), dtype=np.float64)
    active = stored_counts > 0
    ratios[active] = (
        true_counts[active].astype(np.float64)
        / stored_counts[active].astype(np.float64)
    )
    return ratios[np.asarray(bins, dtype=np.int64)]


def retention_metrics(
    true_counts: np.ndarray,
    stored_counts: np.ndarray,
) -> dict:
    true = np.asarray(true_counts, dtype=np.int64)
    stored = np.asarray(stored_counts, dtype=np.int64)
    active = true > 0
    rates = stored[active] / true[active]
    exact_bins = active & (true == stored)
    active_counts = true[active]
    median_count = np.median(active_counts)
    rare = active & (true <= median_count)
    major_threshold = np.quantile(active_counts, 0.9)
    major = active & (true >= major_threshold)

    def point_rate(mask: np.ndarray) -> float:
        total = int(true[mask].sum())
        return float(stored[mask].sum() / total) if total else 1.0

    return {
        "active_bins": int(active.sum()),
        "exact_bins": int(exact_bins.sum()),
        "exact_bin_fraction": round(
            float(exact_bins.sum() / active.sum()) if active.any() else 1.0,
            9,
        ),
        "median_bin_retention": round(float(np.median(rates)), 9),
        "rare_point_retention": round(point_rate(rare), 9),
        "major_point_retention": round(point_rate(major), 9),
    }
