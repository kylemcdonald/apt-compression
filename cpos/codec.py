"""Reference implementation of the versioned CPOS preview codec."""

from __future__ import annotations

import struct
import zlib
from dataclasses import asdict, dataclass

import numpy as np

MAGIC = b"CPOS"
CONTAINER_VERSION = (1, 0)
ALGORITHM_VERSION = (1, 0, 0)
HEADER_SIZE = 128
ENDIAN_MARKER = 0x01020304
FLAGS = 1
RECORD_SIZE = 8
SPECTRUM_MIN_DA = 0.0
SPECTRUM_MAX_DA = 300.0
SPECTRUM_BIN_DA = 0.05
SPECTRUM_BINS = int(round(
    (SPECTRUM_MAX_DA - SPECTRUM_MIN_DA) / SPECTRUM_BIN_DA
))
DEFAULT_MAX_POINTS = 499_000
UINT32_MAX = (1 << 32) - 1

_HEADER = struct.Struct("<4s6H5I3f6f6I")
assert _HEADER.size == 96


class CposVersionError(ValueError):
    """Raised when a CPOS file requires an unsupported format version."""


@dataclass(frozen=True)
class CposHeader:
    container_version: tuple[int, int]
    algorithm_version: tuple[int, int, int]
    header_size: int
    flags: int
    original_point_count: int
    stored_point_count: int
    spectrum_bin_count: int
    spectrum_min_da: float
    spectrum_bin_da: float
    spectrum_max_da: float
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]]
    true_counts_offset: int
    stored_counts_offset: int
    records_offset: int
    file_size: int
    payload_crc32: int
    max_points: int

    def to_dict(self) -> dict:
        output = asdict(self)
        output["container_version"] = ".".join(map(str, self.container_version))
        output["algorithm_version"] = ".".join(map(str, self.algorithm_version))
        output["payload_crc32"] = f"{self.payload_crc32:08x}"
        return output


def _point_array(points: np.ndarray) -> np.ndarray:
    array = np.asarray(points, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 4:
        raise ValueError("points must have shape (N, 4)")
    if len(array) == 0:
        raise ValueError("CPOS cannot encode an empty point cloud")
    if len(array) > UINT32_MAX:
        raise ValueError("CPOS v1 supports at most 2^32 - 1 input points")
    if not np.isfinite(array).all():
        raise ValueError("points contain non-finite values")
    return array


def _mass_bins(mass: np.ndarray) -> np.ndarray:
    scaled = (
        (mass.astype(np.float64) - SPECTRUM_MIN_DA) / SPECTRUM_BIN_DA
    )
    return np.clip(np.floor(scaled), 0, SPECTRUM_BINS - 1).astype(np.int32)


def _allocation(counts: np.ndarray, limit: int) -> np.ndarray:
    """Allocate retained points with proportional largest remainders."""
    total = int(counts.sum())
    if total <= limit:
        return counts.astype(np.int64)

    counts64 = counts.astype(np.int64)
    allocation = np.zeros_like(counts64)
    capacities = counts64
    slots = limit
    capacity_total = int(capacities.sum())
    quotients = np.zeros_like(counts64)
    remainders = np.zeros_like(counts64)
    for index in np.flatnonzero(capacities):
        quotient, remainder = divmod(
            int(capacities[index]) * slots,
            capacity_total,
        )
        quotients[index] = quotient
        remainders[index] = remainder
    allocation += quotients

    leftover = limit - int(allocation.sum())
    if leftover:
        candidates = np.flatnonzero(capacities)
        order = candidates[np.argsort(
            -remainders[candidates],
            kind="stable",
        )]
        allocation[order[:leftover]] += 1
    return allocation


def _select_indices(bins: np.ndarray, counts: np.ndarray,
                    allocation: np.ndarray) -> np.ndarray:
    """Select deterministic midpoint samples, grouped by ascending mass bin."""
    order = np.argsort(bins, kind="stable")
    starts = np.empty(SPECTRUM_BINS + 1, dtype=np.int64)
    starts[0] = 0
    np.cumsum(counts, dtype=np.int64, out=starts[1:])
    selected = np.empty(int(allocation.sum()), dtype=np.int64)

    cursor = 0
    for bin_index in np.flatnonzero(allocation):
        take = int(allocation[bin_index])
        count = int(counts[bin_index])
        positions = np.floor(
            (np.arange(take, dtype=np.float64) + 0.5) * count / take
        ).astype(np.int64)
        selected[cursor:cursor + take] = (
            order[starts[bin_index] + positions]
        )
        cursor += take
    return selected


def _quantize(points: np.ndarray, selected: np.ndarray,
              bounds: np.ndarray) -> np.ndarray:
    subset = points[selected].astype(np.float64)
    extent = bounds[1] - bounds[0]
    safe_extent = np.where(extent > 0, extent, 1.0)
    records = np.empty((len(subset), 4), dtype="<u2")
    normalized = (subset[:, :3] - bounds[0]) / safe_extent
    records[:, :3] = np.clip(
        np.floor(normalized * 65535.0 + 0.5), 0, 65535
    ).astype(np.uint16)
    mass_normalized = (
        (subset[:, 3] - SPECTRUM_MIN_DA)
        / (SPECTRUM_MAX_DA - SPECTRUM_MIN_DA)
    )
    records[:, 3] = np.clip(
        np.floor(mass_normalized * 65535.0 + 0.5), 0, 65535
    ).astype(np.uint16)
    return records


def encode(points: np.ndarray, max_points: int = DEFAULT_MAX_POINTS) -> bytes:
    """Encode an ``N x 4`` point array as one CPOS v1 byte string."""
    array = _point_array(points)
    if not isinstance(max_points, (int, np.integer)) or max_points <= 0:
        raise ValueError("max_points must be a positive integer")
    if max_points > UINT32_MAX:
        raise ValueError("max_points exceeds the CPOS v1 uint32 limit")

    bounds = np.stack([
        array[:, :3].min(axis=0),
        array[:, :3].max(axis=0),
    ]).astype(np.float64)
    bins = _mass_bins(array[:, 3])
    true_counts = np.bincount(
        bins, minlength=SPECTRUM_BINS
    ).astype("<u4")
    allocation = _allocation(true_counts.astype(np.int64), int(max_points))
    selected = _select_indices(bins, true_counts.astype(np.int64), allocation)
    records = _quantize(array, selected, bounds)
    stored_counts = allocation.astype("<u4")

    true_counts_offset = HEADER_SIZE
    stored_counts_offset = true_counts_offset + SPECTRUM_BINS * 4
    records_offset = stored_counts_offset + SPECTRUM_BINS * 4
    file_size = records_offset + len(records) * RECORD_SIZE
    if file_size > UINT32_MAX:
        raise ValueError("encoded file exceeds the CPOS v1 uint32 size limit")

    payload = (
        true_counts.tobytes()
        + stored_counts.tobytes()
        + records.tobytes()
    )
    payload_crc32 = zlib.crc32(payload) & UINT32_MAX
    header = bytearray(HEADER_SIZE)
    _HEADER.pack_into(
        header,
        0,
        MAGIC,
        *CONTAINER_VERSION,
        *ALGORITHM_VERSION,
        HEADER_SIZE,
        ENDIAN_MARKER,
        FLAGS,
        len(array),
        len(records),
        SPECTRUM_BINS,
        SPECTRUM_MIN_DA,
        SPECTRUM_BIN_DA,
        SPECTRUM_MAX_DA,
        *bounds[0],
        *bounds[1],
        true_counts_offset,
        stored_counts_offset,
        records_offset,
        file_size,
        payload_crc32,
        int(max_points),
    )
    return bytes(header) + payload


def inspect(data: bytes | bytearray | memoryview,
            verify_checksum: bool = True) -> CposHeader:
    """Parse and validate a CPOS header without decoding its point records."""
    view = memoryview(data).cast("B")
    if len(view) < HEADER_SIZE:
        raise ValueError("truncated CPOS header")
    values = _HEADER.unpack_from(view, 0)
    (
        magic,
        container_major,
        container_minor,
        algorithm_major,
        algorithm_minor,
        algorithm_patch,
        header_size,
        endian_marker,
        flags,
        original_count,
        stored_count,
        spectrum_bins,
        spectrum_min,
        spectrum_bin,
        spectrum_max,
        xmin,
        ymin,
        zmin,
        xmax,
        ymax,
        zmax,
        true_counts_offset,
        stored_counts_offset,
        records_offset,
        file_size,
        payload_crc32,
        max_points,
    ) = values

    if magic != MAGIC:
        raise ValueError("not a CPOS file")
    if (container_major, container_minor) != CONTAINER_VERSION:
        raise CposVersionError(
            f"unsupported CPOS container {container_major}.{container_minor}"
        )
    if (
        algorithm_major,
        algorithm_minor,
        algorithm_patch,
    ) != ALGORITHM_VERSION:
        raise CposVersionError(
            f"unsupported CPOS codec {algorithm_major}."
            f"{algorithm_minor}.{algorithm_patch}"
        )
    if header_size != HEADER_SIZE:
        raise ValueError(f"unsupported CPOS header size {header_size}")
    if endian_marker != ENDIAN_MARKER:
        raise ValueError("invalid CPOS endian marker")
    if flags != FLAGS:
        raise ValueError(f"unsupported CPOS flags 0x{flags:08x}")
    if original_count == 0 or stored_count == 0:
        raise ValueError("CPOS point counts must be non-zero")
    if stored_count > original_count or stored_count > max_points:
        raise ValueError("invalid CPOS retained point count")
    if spectrum_bins != SPECTRUM_BINS:
        raise ValueError(f"unsupported CPOS spectrum size {spectrum_bins}")

    expected_stored_offset = HEADER_SIZE + spectrum_bins * 4
    expected_records_offset = expected_stored_offset + spectrum_bins * 4
    expected_size = expected_records_offset + stored_count * RECORD_SIZE
    if (
        true_counts_offset != HEADER_SIZE
        or stored_counts_offset != expected_stored_offset
        or records_offset != expected_records_offset
        or file_size != expected_size
        or file_size != len(view)
    ):
        raise ValueError("invalid CPOS section offsets or file size")
    if verify_checksum:
        actual_crc32 = zlib.crc32(view[header_size:]) & UINT32_MAX
        if actual_crc32 != payload_crc32:
            raise ValueError(
                f"CPOS checksum mismatch: expected {payload_crc32:08x}, "
                f"got {actual_crc32:08x}"
            )

    return CposHeader(
        container_version=(container_major, container_minor),
        algorithm_version=(algorithm_major, algorithm_minor, algorithm_patch),
        header_size=header_size,
        flags=flags,
        original_point_count=original_count,
        stored_point_count=stored_count,
        spectrum_bin_count=spectrum_bins,
        spectrum_min_da=spectrum_min,
        spectrum_bin_da=spectrum_bin,
        spectrum_max_da=spectrum_max,
        bounds=((xmin, ymin, zmin), (xmax, ymax, zmax)),
        true_counts_offset=true_counts_offset,
        stored_counts_offset=stored_counts_offset,
        records_offset=records_offset,
        file_size=file_size,
        payload_crc32=payload_crc32,
        max_points=max_points,
    )


def _sections(data: bytes | bytearray | memoryview,
              header: CposHeader) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    view = memoryview(data).cast("B")
    true_counts = np.frombuffer(
        view,
        dtype="<u4",
        count=header.spectrum_bin_count,
        offset=header.true_counts_offset,
    )
    stored_counts = np.frombuffer(
        view,
        dtype="<u4",
        count=header.spectrum_bin_count,
        offset=header.stored_counts_offset,
    )
    records = np.frombuffer(
        view,
        dtype="<u2",
        count=header.stored_point_count * 4,
        offset=header.records_offset,
    ).reshape(-1, 4)
    if int(true_counts.sum(dtype=np.uint64)) != header.original_point_count:
        raise ValueError("CPOS original spectrum counts do not match the header")
    if int(stored_counts.sum(dtype=np.uint64)) != header.stored_point_count:
        raise ValueError("CPOS retained spectrum counts do not match the header")
    if np.any(stored_counts > true_counts):
        raise ValueError("CPOS retained spectrum exceeds the original spectrum")
    return true_counts, stored_counts, records


def decode(data: bytes | bytearray | memoryview) -> np.ndarray:
    """Decode retained CPOS preview points into an ``N x 4 float32`` array."""
    header = inspect(data)
    _, _, records = _sections(data, header)
    quantized = records.astype(np.float64)
    bounds = np.asarray(header.bounds, dtype=np.float64)
    points = np.empty((header.stored_point_count, 4), dtype=np.float32)
    points[:, :3] = (
        bounds[0]
        + quantized[:, :3] / 65535.0 * (bounds[1] - bounds[0])
    ).astype(np.float32)
    points[:, 3] = (
        header.spectrum_min_da
        + quantized[:, 3] / 65535.0
        * (header.spectrum_max_da - header.spectrum_min_da)
    ).astype(np.float32)
    return points


def spectrum_counts(
    data: bytes | bytearray | memoryview,
) -> tuple[np.ndarray, np.ndarray]:
    """Return copies of the original and retained v1 spectrum counts."""
    header = inspect(data)
    true_counts, stored_counts, _ = _sections(data, header)
    return true_counts.copy(), stored_counts.copy()
