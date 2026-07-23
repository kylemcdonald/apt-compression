from __future__ import annotations

import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

from cpos import (
    ALGORITHM_VERSION,
    CONTAINER_VERSION,
    CposVersionError,
    decode,
    encode,
    inspect,
)
from cpos.codec import HEADER_SIZE, spectrum_counts
from cpos.io import write_pos

ROOT = Path(__file__).resolve().parents[1]


def synthetic_points(seed: int = 7, count: int = 30_000) -> np.ndarray:
    rng = np.random.default_rng(seed)
    xyz = rng.normal(size=(count, 3)).astype(np.float32)
    xyz[:, 0] = xyz[:, 0] * 12.0 + 4.0
    xyz[:, 1] *= 8.0
    xyz[:, 2] = xyz[:, 2] * 35.0 + 80.0
    categories = rng.choice(5, size=count, p=[0.54, 0.23, 0.12, 0.08, 0.03])
    centers = np.array([6.0, 27.98, 55.94, 119.0, 421.5], dtype=np.float32)
    mass = centers[categories] + rng.normal(0, 0.035, size=count)
    return np.column_stack([xyz, mass]).astype(np.float32)


def test_versioned_header_and_lossy_decode():
    points = synthetic_points()
    payload = encode(points, max_points=4_999)
    header = inspect(payload)
    decoded = decode(payload)
    original_counts, retained_counts = spectrum_counts(payload)

    assert payload[:4] == b"CPOS"
    assert header.container_version == CONTAINER_VERSION
    assert header.algorithm_version == ALGORITHM_VERSION
    assert header.header_size == HEADER_SIZE
    assert header.original_point_count == len(points)
    assert header.stored_point_count == 4_999
    assert header.file_size == len(payload)
    assert decoded.shape == (4_999, 4)
    assert decoded.dtype == np.float32
    assert np.isfinite(decoded).all()
    assert int(original_counts.sum()) == len(points)
    assert int(retained_counts.sum()) == len(decoded)
    assert np.all(retained_counts <= original_counts)
    original_distribution = original_counts / original_counts.sum()
    retained_distribution = retained_counts / retained_counts.sum()
    spectrum_tv = 0.5 * np.abs(
        original_distribution - retained_distribution
    ).sum()
    assert spectrum_tv < 0.01
    source_min = points[:, :3].min(0)
    source_max = points[:, :3].max(0)
    assert np.allclose(np.asarray(header.bounds[0]), source_min)
    assert np.allclose(np.asarray(header.bounds[1]), source_max)
    assert np.all(decoded[:, :3].min(0) >= source_min - 0.01)
    assert np.all(decoded[:, :3].max(0) <= source_max + 0.01)
    assert decoded[:, 3].min() >= 0
    assert decoded[:, 3].max() <= 300


def test_checksum_and_every_other_version_are_rejected():
    payload = bytearray(encode(synthetic_points(count=2_000), max_points=499))

    corrupted = payload.copy()
    corrupted[-1] ^= 0x80
    with pytest.raises(ValueError, match="checksum mismatch"):
        inspect(corrupted)

    for offset, value, pattern in (
        (4, 2, "unsupported CPOS container"),
        (6, 1, "unsupported CPOS container"),
        (8, 2, "unsupported CPOS codec"),
        (10, 1, "unsupported CPOS codec"),
        (12, 1, "unsupported CPOS codec"),
    ):
        other_version = payload.copy()
        struct.pack_into("<H", other_version, offset, value)
        with pytest.raises(CposVersionError, match=pattern):
            inspect(other_version)


@pytest.mark.skipif(shutil.which("node") is None, reason="Node is not installed")
def test_javascript_library():
    subprocess.run(
        ["node", "--test", str(ROOT / "javascript" / "test.mjs")],
        cwd=ROOT,
        check=True,
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="Node is not installed")
def test_javascript_matches_python_byte_for_byte(tmp_path):
    points = synthetic_points(count=18_000)
    pos_path = tmp_path / "fixture.pos"
    python_path = tmp_path / "python.cpos"
    javascript_path = tmp_path / "javascript.cpos"
    javascript_pos_path = tmp_path / "javascript.pos"
    write_pos(pos_path, points)
    python_path.write_bytes(encode(points, max_points=4_999))

    subprocess.run(
        [
            "node",
            str(ROOT / "javascript" / "cli.mjs"),
            "encode",
            str(pos_path),
            str(javascript_path),
            "--max-points",
            "4999",
        ],
        cwd=ROOT,
        check=True,
    )
    assert javascript_path.read_bytes() == python_path.read_bytes()

    subprocess.run(
        [
            "node",
            str(ROOT / "javascript" / "cli.mjs"),
            "decode",
            str(javascript_path),
            str(javascript_pos_path),
        ],
        cwd=ROOT,
        check=True,
    )
    expected_pos_path = tmp_path / "python.pos"
    write_pos(expected_pos_path, decode(python_path.read_bytes()))
    assert javascript_pos_path.read_bytes() == expected_pos_path.read_bytes()
