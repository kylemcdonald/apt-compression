from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from cpos.io import write_pos

from .codec import (
    CODEC_VERSION,
    CONTAINER_VERSION,
    allocate_sublinear,
    decode,
    decode_retained,
    encode,
    inspect,
)


def fixture(seed: int = 81, count: int = 80_003) -> np.ndarray:
    rng = np.random.default_rng(seed)
    points = np.empty((count, 4), dtype=np.float32)
    points[:, 0] = rng.normal(0, 18, count)
    points[:, 1] = rng.normal(0, 16, count)
    points[:, 2] = rng.uniform(0, 220, count)
    peaks = rng.choice(
        np.array([14.0, 27.98, 55.94, 91.2], dtype=np.float32),
        count,
        p=[0.03, 0.72, 0.23, 0.02],
    )
    points[:, 3] = peaks + rng.normal(0, 0.025, count)
    return points


def test_sublinear_allocation_retains_rare_bins_at_higher_rates():
    counts = np.array([1, 4, 100, 10_000], dtype=np.int64)
    allocation = allocate_sublinear(counts, 1_000, 0.5)
    rates = allocation / counts
    assert allocation.sum() == 1_000
    assert np.all(allocation <= counts)
    assert np.all(rates[:-1] >= rates[1:])
    assert allocation[0] == counts[0]


@pytest.mark.parametrize("bin_width", [0.01, 0.1])
@pytest.mark.parametrize("exponent", [0.25, 0.5, 0.75, 1.0])
def test_retained_tuples_and_expanded_histogram_are_exact(
    bin_width: float,
    exponent: float,
):
    points = fixture(count=20_003)
    payload = encode(
        points,
        target_points=4_003,
        bin_width_da=bin_width,
        allocation_exponent=exponent,
    )
    header = inspect(payload)
    assert header.container_version == CONTAINER_VERSION
    assert header.codec_version == CODEC_VERSION
    assert header.original_point_count == len(points)
    assert header.stored_point_count == 4_003

    retained = decode_retained(payload)
    expanded = decode(payload)
    assert len(retained.quantized) == 4_003
    assert len(expanded.points) == len(points)
    assert int(expanded.exact.sum()) == 4_003
    assert np.isfinite(expanded.points).all()
    assert np.array_equal(
        np.bincount(
            expanded.bins,
            minlength=header.spectrum_bin_count,
        ),
        expanded.true_counts,
    )
    decoded_mass_bins = np.floor(
        expanded.points[:, 3].astype(np.float64) / bin_width,
    ).astype(np.int64)
    decoded_mass_bins = np.clip(
        decoded_mass_bins,
        0,
        header.spectrum_bin_count - 1,
    )
    assert np.array_equal(
        np.bincount(
            decoded_mass_bins,
            minlength=header.spectrum_bin_count,
        ),
        expanded.true_counts,
    )
    assert np.all(retained.stored_counts <= retained.true_counts)


def test_no_decimation_keeps_every_quantized_point_exact():
    points = fixture(count=10_003)
    payload = encode(points, target_points=20_000)
    retained = decode_retained(payload)
    expanded = decode(payload)
    assert retained.header.stored_point_count == len(points)
    assert expanded.exact.all()
    assert len(expanded.points) == len(points)


def test_dither_changes_only_synthesized_records():
    payload = encode(fixture(count=12_003), target_points=4_003)
    without_noise = decode(payload, noise="none")
    with_uniform = decode(payload, noise="uniform")
    with_gaussian = decode(payload, noise="gaussian")
    assert np.array_equal(without_noise.exact, with_uniform.exact)
    assert np.array_equal(without_noise.exact, with_gaussian.exact)
    assert np.array_equal(
        without_noise.points[without_noise.exact],
        with_uniform.points[with_uniform.exact],
    )
    assert np.array_equal(
        without_noise.points[without_noise.exact],
        with_gaussian.points[with_gaussian.exact],
    )
    assert np.any(
        without_noise.points[~without_noise.exact, :3]
        != with_uniform.points[~with_uniform.exact, :3],
    )
    assert np.any(
        without_noise.points[~without_noise.exact, :3]
        != with_gaussian.points[~with_gaussian.exact, :3],
    )
    assert np.array_equal(without_noise.bins, with_uniform.bins)
    assert np.array_equal(without_noise.bins, with_gaussian.bins)


def test_checksum_and_other_versions_are_rejected():
    payload = bytearray(encode(fixture(count=5_003), target_points=2_003))
    payload[-1] ^= 0x80
    with pytest.raises(ValueError, match="checksum"):
        inspect(payload)

    payload = bytearray(encode(fixture(count=5_003), target_points=2_003))
    payload[8] = 2
    with pytest.raises(ValueError, match="codec"):
        inspect(payload)

    payload = bytearray(encode(fixture(count=5_003), target_points=2_003))
    payload[168] = 1
    with pytest.raises(ValueError, match="reserved"):
        inspect(payload)

    payload = bytearray(encode(fixture(count=5_003), target_points=2_003))
    payload[64] = 9
    with pytest.raises(ValueError, match="noise"):
        inspect(payload)


def test_javascript_decoder_matches_python(tmp_path: Path):
    payload = encode(fixture(count=50_003), target_points=12_003)
    path = tmp_path / "fixture.cp4m"
    path.write_bytes(payload)
    retained = decode_retained(payload)
    expected_hash = hashlib.sha256(retained.quantized.tobytes()).hexdigest()
    root = Path(__file__).parents[2]
    completed = subprocess.run(
        (
            "node",
            str(root / "experiments/lossy4m/javascript/inspect.mjs"),
            str(path),
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    assert result["quantizedHash"] == expected_hash
    assert result["trueTotal"] == 50_003
    assert result["storedTotal"] == 12_003
    assert result["displayPoints"] == 50_003
    assert result["displayExact"] == 12_003


@pytest.mark.parametrize(
    ("bin_width", "exponent"),
    [(0.1, 0.75), (0.01, 0.5)],
)
def test_javascript_encoder_matches_python_byte_for_byte(
    tmp_path: Path,
    bin_width: float,
    exponent: float,
):
    points = fixture(count=30_003)
    expected = encode(
        points,
        target_points=12_003,
        bin_width_da=bin_width,
        allocation_exponent=exponent,
    )
    source_path = tmp_path / "fixture.pos"
    output_path = tmp_path / "javascript.cp4m"
    write_pos(source_path, points)
    root = Path(__file__).parents[2]
    subprocess.run(
        (
            "node",
            str(root / "experiments/lossy4m/javascript/encode.mjs"),
            str(source_path),
            str(output_path),
            "12003",
            str(bin_width),
            str(exponent),
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    assert output_path.read_bytes() == expected
