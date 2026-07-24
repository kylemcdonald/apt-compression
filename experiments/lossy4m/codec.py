"""Experimental mass-aware lossy codec with exact 12-bit retained seeds."""

from __future__ import annotations

import struct
import zlib
from dataclasses import asdict, dataclass

import numpy as np

from experiments.lossless12.lossless12 import (
    BITS,
    MASK12,
    SPATIAL_BITS,
    _best_rice_parameter,
    pack_bitplanes,
    quantize12,
    spatial_morton,
    unpack_bitplanes,
)

MAGIC = b"CP4M"
CONTAINER_VERSION = (1, 0)
CODEC_VERSION = (1, 0, 0)
HEADER_SIZE = 192
FLAGS = 1
DEFAULT_TARGET_POINTS = 4_000_000
DEFAULT_BIN_WIDTH_DA = 0.1
DEFAULT_ALLOCATION_EXPONENT = 0.75
SPECTRUM_MIN_DA = 0.0
SPECTRUM_MAX_DA = 300.0
CORE_GROUPED_MORTON_RICE = 1
AXIS_ORDER = (0, 1, 2)
NOISE_NONE = 0
NOISE_UNIFORM = 1
NOISE_GAUSSIAN = 2
DEFAULT_NOISE = NOISE_UNIFORM
DEFAULT_SEED = 0xC0454D
UINT32_MAX = (1 << 32) - 1

# magic; versions and header size; flags; point counts; histogram fields;
# default noise and seed; quantizer bounds; prefix/axis order; section layout;
# payload CRC and reserved.
_HEADER = struct.Struct("<4s6HI3QI4fIQ8f4B6Q2I")
assert _HEADER.size == 168


@dataclass(frozen=True)
class Lossy4mHeader:
    container_version: tuple[int, int]
    codec_version: tuple[int, int, int]
    original_point_count: int
    stored_point_count: int
    target_point_count: int
    spectrum_bin_count: int
    spectrum_min_da: float
    spectrum_max_da: float
    spectrum_bin_da: float
    allocation_exponent: float
    default_noise: int
    seed: int
    minimum: tuple[float, float, float, float]
    maximum: tuple[float, float, float, float]
    core_method: int
    axis_order: tuple[int, int, int]
    true_counts_offset: int
    stored_counts_offset: int
    core_offset: int
    core_compressed_size: int
    core_uncompressed_size: int
    file_size: int
    payload_crc32: int

    def to_dict(self) -> dict:
        output = asdict(self)
        output["container_version"] = ".".join(map(str, self.container_version))
        output["codec_version"] = ".".join(map(str, self.codec_version))
        output["payload_crc32"] = f"{self.payload_crc32:08x}"
        return output


@dataclass(frozen=True)
class RetainedCloud:
    header: Lossy4mHeader
    quantized: np.ndarray
    points: np.ndarray
    bins: np.ndarray
    true_counts: np.ndarray
    stored_counts: np.ndarray


@dataclass(frozen=True)
class DecodedCloud:
    header: Lossy4mHeader
    points: np.ndarray
    exact: np.ndarray
    bins: np.ndarray
    true_counts: np.ndarray
    stored_counts: np.ndarray


def _point_array(points: np.ndarray) -> np.ndarray:
    array = np.asarray(points, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 4 or len(array) == 0:
        raise ValueError("points must have shape (N, 4) with N > 0")
    if len(array) > UINT32_MAX:
        raise ValueError("codec supports at most 2^32 - 1 source points")
    if not np.isfinite(array).all():
        raise ValueError("points contain non-finite values")
    return array


def _bin_count(bin_width: float) -> int:
    if not np.isfinite(bin_width) or bin_width <= 0:
        raise ValueError("bin_width_da must be positive and finite")
    count = int(round((SPECTRUM_MAX_DA - SPECTRUM_MIN_DA) / bin_width))
    if count <= 0 or count > UINT32_MAX:
        raise ValueError("unsupported histogram bin width")
    if not np.isclose(
        count * bin_width,
        SPECTRUM_MAX_DA - SPECTRUM_MIN_DA,
        rtol=0,
        atol=1e-4,
    ):
        raise ValueError("bin_width_da must divide the 0–300 Da range")
    return count


def _dequantize_values(
    quantized: np.ndarray,
    minimum: np.ndarray,
    maximum: np.ndarray,
) -> np.ndarray:
    values = np.asarray(quantized, dtype=np.float64)
    extent = maximum.astype(np.float64) - minimum.astype(np.float64)
    return minimum.astype(np.float64) + values / MASK12 * extent


def _mass_bins(
    quantized_mass: np.ndarray,
    minimum_mass: float,
    maximum_mass: float,
    bin_width: float,
    bin_count: int,
) -> np.ndarray:
    extent = float(maximum_mass) - float(minimum_mass)
    mass = float(minimum_mass) + (
        np.asarray(quantized_mass, dtype=np.float64) / MASK12 * extent
    )
    bins = np.floor(
        (mass - SPECTRUM_MIN_DA) / bin_width,
    ).astype(np.int64)
    return np.clip(bins, 0, bin_count - 1).astype(np.int32)


def _source_mass_bins(
    mass: np.ndarray,
    bin_width: float,
    bin_count: int,
) -> np.ndarray:
    bins = np.floor(
        (
            np.asarray(mass, dtype=np.float32).astype(np.float64)
            - SPECTRUM_MIN_DA
        ) / bin_width,
    ).astype(np.int64)
    return np.clip(bins, 0, bin_count - 1).astype(np.int32)


def _force_mass_bins(
    mass: np.ndarray,
    bins: np.ndarray,
    bin_width: float,
    bin_count: int,
) -> np.ndarray:
    output = np.asarray(mass, dtype=np.float32).copy()
    actual = _source_mass_bins(output, bin_width, bin_count)
    mismatch = actual != bins
    if mismatch.any():
        output[mismatch] = (
            SPECTRUM_MIN_DA
            + (bins[mismatch].astype(np.float64) + 0.5) * bin_width
        ).astype(np.float32)
    if not np.array_equal(
        _source_mass_bins(output, bin_width, bin_count),
        bins,
    ):
        raise AssertionError("unable to place decoded masses in source bins")
    return output


def allocate_sublinear(
    counts: np.ndarray,
    limit: int,
    exponent: float,
) -> np.ndarray:
    """Allocate a capped c**exponent quota with deterministic remainders."""
    source = np.asarray(counts, dtype=np.int64)
    if source.ndim != 1 or np.any(source < 0):
        raise ValueError("counts must be a one-dimensional nonnegative array")
    total = int(source.sum(dtype=np.int64))
    if limit <= 0:
        raise ValueError("limit must be positive")
    if not np.isfinite(exponent) or not 0 <= exponent <= 1:
        raise ValueError("allocation exponent must be in [0, 1]")
    if total <= limit:
        return source.copy()

    active = source > 0
    active_count = int(active.sum())
    if limit < active_count:
        raise ValueError(
            "limit is smaller than the number of nonempty mass bins",
        )
    output = active.astype(np.int64)
    remaining = limit - active_count
    if remaining == 0:
        return output
    capacity = source - output
    weights = np.zeros(len(source), dtype=np.float64)
    weights[active] = np.power(source[active].astype(np.float64), exponent)
    can_grow = capacity > 0
    upper = float(np.max(capacity[can_grow] / weights[can_grow]))
    lower = 0.0
    for _ in range(80):
        midpoint = (lower + upper) * 0.5
        allocated = np.minimum(
            capacity.astype(np.float64),
            midpoint * weights,
        ).sum()
        if allocated < remaining:
            lower = midpoint
        else:
            upper = midpoint

    ideal = np.minimum(capacity.astype(np.float64), upper * weights)
    output += np.floor(ideal).astype(np.int64)
    leftover = limit - int(output.sum(dtype=np.int64))
    if leftover:
        candidates = np.flatnonzero(output < source)
        fractional = (
            ideal[candidates]
            - np.floor(ideal[candidates])
        )
        order = np.lexsort((
            candidates,
            source[candidates],
            -fractional,
        ))
        output[candidates[order[:leftover]]] += 1
    if int(output.sum(dtype=np.int64)) != limit or np.any(output > source):
        raise AssertionError("invalid sublinear allocation")
    if np.any(output[active] == 0):
        raise AssertionError("active mass bin received no retained seed")
    return output


def _select_quantized(
    values: np.ndarray,
    bins: np.ndarray,
    counts: np.ndarray,
    allocation: np.ndarray,
) -> np.ndarray:
    spatial = spatial_morton(values[:, :3])
    combined = (
        bins.astype(np.uint64) << np.uint64(SPATIAL_BITS)
    ) | spatial
    order = np.argsort(combined, kind="stable")
    starts = np.empty(len(counts) + 1, dtype=np.int64)
    starts[0] = 0
    np.cumsum(counts, dtype=np.int64, out=starts[1:])
    selected = np.empty(int(allocation.sum(dtype=np.int64)), dtype=np.int64)
    cursor = 0
    for bin_index in np.flatnonzero(allocation):
        take = int(allocation[bin_index])
        count = int(counts[bin_index])
        positions = np.floor(
            (np.arange(take, dtype=np.float64) + 0.5) * count / take,
        ).astype(np.int64)
        selected[cursor:cursor + take] = (
            order[starts[bin_index] + positions]
        )
        cursor += take
    return values[selected]


def _spatial_morton_inverse(keys: np.ndarray) -> np.ndarray:
    source = np.asarray(keys, dtype=np.uint64)
    output = np.zeros((len(source), 3), dtype=np.uint16)
    for bit in range(BITS):
        for dimension in range(3):
            output[:, dimension] |= (
                (
                    source >> np.uint64(bit * 3 + dimension)
                ) & np.uint64(1)
            ).astype(np.uint16) << np.uint16(bit)
    return output


def _group_slices(counts: np.ndarray):
    cursor = 0
    for bin_index, count in enumerate(counts):
        size = int(count)
        if size:
            yield bin_index, cursor, cursor + size
            cursor += size


def _encode_grouped_core(
    values: np.ndarray,
    stored_counts: np.ndarray,
) -> bytes:
    active = np.flatnonzero(stored_counts)
    widths = np.empty(len(active), dtype=np.uint8)
    unary_lengths = np.empty(len(active), dtype="<u8")
    remainder_streams: list[bytes] = []
    unary_streams: list[bytes] = []
    spatial_all = spatial_morton(values[:, :3])
    for active_index, (_, start, end) in enumerate(_group_slices(stored_counts)):
        spatial = spatial_all[start:end]
        if np.any(spatial[1:] < spatial[:-1]):
            raise AssertionError("grouped spatial keys are not sorted")
        gaps = np.empty_like(spatial)
        gaps[0] = spatial[0]
        gaps[1:] = spatial[1:] - spatial[:-1]
        width = _best_rice_parameter(gaps)
        widths[active_index] = width
        remainder = gaps & np.uint64((1 << width) - 1 if width else 0)
        remainder_streams.append(pack_bitplanes(remainder, width))
        quotient = gaps >> np.uint64(width)
        ends = np.cumsum(quotient + np.uint64(1), dtype=np.uint64) - 1
        unary_length = int(ends[-1]) + 1
        unary_lengths[active_index] = unary_length
        unary = np.zeros(unary_length, dtype=np.uint8)
        unary[ends.astype(np.int64)] = 1
        unary_streams.append(np.packbits(unary, bitorder="little").tobytes())
    mass_stream = pack_bitplanes(values[:, 3], BITS)
    return (
        b"G12R"
        + struct.pack("<QI", len(values), len(active))
        + widths.tobytes()
        + unary_lengths.tobytes()
        + b"".join(remainder_streams)
        + b"".join(unary_streams)
        + mass_stream
    )


def _decode_grouped_core(
    data: bytes,
    stored_counts: np.ndarray,
) -> np.ndarray:
    if data[:4] != b"G12R":
        raise ValueError("invalid CP4M grouped core")
    count, active_count = struct.unpack_from("<QI", data, 4)
    active = np.flatnonzero(stored_counts)
    if len(active) != active_count:
        raise ValueError("CP4M grouped core active-bin mismatch")
    offset = 16
    widths = np.frombuffer(
        data,
        dtype=np.uint8,
        count=active_count,
        offset=offset,
    )
    offset += active_count
    unary_lengths = np.frombuffer(
        data,
        dtype="<u8",
        count=active_count,
        offset=offset,
    )
    offset += active_count * 8
    remainder_bytes = sum(
        ((int(stored_counts[bin_index]) + 7) // 8) * int(width)
        for bin_index, width in zip(active, widths, strict=True)
    )
    unary_bytes = sum((int(length) + 7) // 8 for length in unary_lengths)
    remainder_cursor = offset
    remainder_end = offset + remainder_bytes
    unary_cursor = remainder_end
    unary_end = unary_cursor + unary_bytes
    mass_size = ((int(count) + 7) // 8) * BITS
    if unary_end + mass_size != len(data):
        raise ValueError("invalid CP4M grouped core length")

    output = np.empty((count, 4), dtype=np.uint16)
    cursor = 0
    for active_index, (_, _, end) in enumerate(_group_slices(stored_counts)):
        size = end - cursor
        width = int(widths[active_index])
        remainder_size = ((size + 7) // 8) * width
        remainder = unpack_bitplanes(
            data[remainder_cursor:remainder_cursor + remainder_size],
            size,
            width,
        )
        remainder_cursor += remainder_size
        unary_length = int(unary_lengths[active_index])
        unary_size = (unary_length + 7) // 8
        unary = np.unpackbits(
            np.frombuffer(
                data,
                dtype=np.uint8,
                count=unary_size,
                offset=unary_cursor,
            ),
            bitorder="little",
        )[:unary_length]
        unary_cursor += unary_size
        ends = np.flatnonzero(unary).astype(np.int64)
        if len(ends) != size:
            raise ValueError("invalid CP4M Rice unary stream")
        previous = np.concatenate((
            np.array([-1], dtype=np.int64),
            ends[:-1],
        ))
        quotient = (ends - previous - 1).astype(np.uint64)
        gaps = (quotient << np.uint64(width)) | remainder
        spatial = np.cumsum(gaps, dtype=np.uint64)
        output[cursor:end, :3] = _spatial_morton_inverse(spatial)
        cursor = end
    output[:, 3] = unpack_bitplanes(
        data[unary_end:],
        int(count),
        BITS,
    ).astype(np.uint16)
    if cursor != count:
        raise ValueError("CP4M grouped core point count mismatch")
    return output


def encode(
    points: np.ndarray,
    *,
    target_points: int = DEFAULT_TARGET_POINTS,
    bin_width_da: float = DEFAULT_BIN_WIDTH_DA,
    allocation_exponent: float = DEFAULT_ALLOCATION_EXPONENT,
    seed: int = DEFAULT_SEED,
) -> bytes:
    """Encode points as retained 12-bit seeds plus exact histogram metadata."""
    array = _point_array(points)
    if not isinstance(target_points, (int, np.integer)) or target_points <= 0:
        raise ValueError("target_points must be a positive integer")
    if not isinstance(seed, (int, np.integer)) or not 0 <= seed < (1 << 64):
        raise ValueError("seed must be an unsigned 64-bit integer")
    bin_count = _bin_count(bin_width_da)
    quantized = quantize12(array)
    values = quantized.values
    bins = _source_mass_bins(
        array[:, 3],
        bin_width_da,
        bin_count,
    )
    true_counts = np.bincount(
        bins,
        minlength=bin_count,
    ).astype(np.int64)
    stored_count = min(len(values), int(target_points))
    allocation = allocate_sublinear(
        true_counts,
        stored_count,
        allocation_exponent,
    )
    selected = _select_quantized(
        values,
        bins,
        true_counts,
        allocation,
    )
    stored_counts = allocation
    core = _encode_grouped_core(selected, stored_counts)
    compressed_core = zlib.compress(core, level=9)

    true_counts_offset = HEADER_SIZE
    stored_counts_offset = true_counts_offset + bin_count * 4
    core_offset = stored_counts_offset + bin_count * 4
    file_size = core_offset + len(compressed_core)
    true_bytes = true_counts.astype("<u4").tobytes()
    stored_bytes = stored_counts.astype("<u4").tobytes()
    payload = true_bytes + stored_bytes + compressed_core
    payload_crc32 = zlib.crc32(payload) & UINT32_MAX
    header = bytearray(HEADER_SIZE)
    _HEADER.pack_into(
        header,
        0,
        MAGIC,
        *CONTAINER_VERSION,
        *CODEC_VERSION,
        HEADER_SIZE,
        FLAGS,
        len(array),
        len(selected),
        int(target_points),
        bin_count,
        SPECTRUM_MIN_DA,
        SPECTRUM_MAX_DA,
        float(bin_width_da),
        float(allocation_exponent),
        DEFAULT_NOISE,
        int(seed),
        *quantized.minimum.astype(np.float32),
        *quantized.maximum.astype(np.float32),
        CORE_GROUPED_MORTON_RICE,
        *AXIS_ORDER,
        true_counts_offset,
        stored_counts_offset,
        core_offset,
        len(compressed_core),
        len(core),
        file_size,
        payload_crc32,
        0,
    )
    return bytes(header) + payload


def inspect(
    data: bytes | bytearray | memoryview,
    *,
    verify_checksum: bool = True,
) -> Lossy4mHeader:
    view = memoryview(data).cast("B")
    if len(view) < HEADER_SIZE:
        raise ValueError("truncated CP4M header")
    values = _HEADER.unpack_from(view, 0)
    (
        magic,
        container_major,
        container_minor,
        codec_major,
        codec_minor,
        codec_patch,
        header_size,
        flags,
        original_count,
        stored_count,
        target_count,
        spectrum_bins,
        spectrum_min,
        spectrum_max,
        spectrum_bin,
        allocation_exponent,
        default_noise,
        seed,
        *tail,
    ) = values
    minimum = tuple(tail[:4])
    maximum = tuple(tail[4:8])
    core_method, axis_x, axis_y, axis_z = tail[8:12]
    (
        true_counts_offset,
        stored_counts_offset,
        core_offset,
        core_compressed_size,
        core_uncompressed_size,
        file_size,
        payload_crc32,
        reserved,
    ) = tail[12:]

    if magic != MAGIC:
        raise ValueError("not a CP4M file")
    if (container_major, container_minor) != CONTAINER_VERSION:
        raise ValueError(
            f"unsupported CP4M container {container_major}.{container_minor}",
        )
    if (codec_major, codec_minor, codec_patch) != CODEC_VERSION:
        raise ValueError(
            f"unsupported CP4M codec {codec_major}.{codec_minor}.{codec_patch}",
        )
    if header_size != HEADER_SIZE or flags != FLAGS or reserved != 0:
        raise ValueError("unsupported CP4M header")
    if any(view[_HEADER.size:HEADER_SIZE]):
        raise ValueError("unsupported CP4M reserved header data")
    if original_count == 0 or stored_count == 0 or stored_count > original_count:
        raise ValueError("invalid CP4M point counts")
    if stored_count > target_count:
        raise ValueError("stored point count exceeds target")
    if (
        not np.isclose(spectrum_min, SPECTRUM_MIN_DA, rtol=0, atol=1e-6)
        or not np.isclose(spectrum_max, SPECTRUM_MAX_DA, rtol=0, atol=1e-6)
        or spectrum_bins != _bin_count(spectrum_bin)
    ):
        raise ValueError("invalid CP4M histogram configuration")
    canonical_bin_width = (
        float(spectrum_max) - float(spectrum_min)
    ) / spectrum_bins
    if (
        not np.isfinite(allocation_exponent)
        or not 0 <= allocation_exponent <= 1
    ):
        raise ValueError("invalid CP4M allocation exponent")
    if default_noise not in (NOISE_NONE, NOISE_UNIFORM, NOISE_GAUSSIAN):
        raise ValueError("invalid CP4M default noise")
    if (
        not np.isfinite(minimum).all()
        or not np.isfinite(maximum).all()
        or np.any(np.asarray(maximum) < np.asarray(minimum))
    ):
        raise ValueError("invalid CP4M quantizer bounds")
    if (
        core_method != CORE_GROUPED_MORTON_RICE
        or (axis_x, axis_y, axis_z) != AXIS_ORDER
    ):
        raise ValueError("unsupported CP4M core ordering")
    expected_stored = HEADER_SIZE + spectrum_bins * 4
    expected_core = expected_stored + spectrum_bins * 4
    expected_size = expected_core + core_compressed_size
    if (
        true_counts_offset != HEADER_SIZE
        or stored_counts_offset != expected_stored
        or core_offset != expected_core
        or file_size != expected_size
        or file_size != len(view)
    ):
        raise ValueError("invalid CP4M section layout")
    if verify_checksum:
        actual = zlib.crc32(view[HEADER_SIZE:]) & UINT32_MAX
        if actual != payload_crc32:
            raise ValueError(
                f"CP4M checksum mismatch: expected {payload_crc32:08x}, "
                f"got {actual:08x}",
            )
    return Lossy4mHeader(
        container_version=(container_major, container_minor),
        codec_version=(codec_major, codec_minor, codec_patch),
        original_point_count=original_count,
        stored_point_count=stored_count,
        target_point_count=target_count,
        spectrum_bin_count=spectrum_bins,
        spectrum_min_da=spectrum_min,
        spectrum_max_da=spectrum_max,
        spectrum_bin_da=canonical_bin_width,
        allocation_exponent=allocation_exponent,
        default_noise=default_noise,
        seed=seed,
        minimum=minimum,
        maximum=maximum,
        core_method=core_method,
        axis_order=(axis_x, axis_y, axis_z),
        true_counts_offset=true_counts_offset,
        stored_counts_offset=stored_counts_offset,
        core_offset=core_offset,
        core_compressed_size=core_compressed_size,
        core_uncompressed_size=core_uncompressed_size,
        file_size=file_size,
        payload_crc32=payload_crc32,
    )


def _sections(
    data: bytes | bytearray | memoryview,
    header: Lossy4mHeader,
) -> tuple[np.ndarray, np.ndarray, bytes]:
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
    core = bytes(view[
        header.core_offset:header.core_offset + header.core_compressed_size
    ])
    if int(true_counts.sum(dtype=np.uint64)) != header.original_point_count:
        raise ValueError("CP4M true histogram does not match point count")
    if int(stored_counts.sum(dtype=np.uint64)) != header.stored_point_count:
        raise ValueError("CP4M stored histogram does not match point count")
    if np.any(stored_counts > true_counts):
        raise ValueError("CP4M stored histogram exceeds source histogram")
    return true_counts, stored_counts, core


def decode_retained(
    data: bytes | bytearray | memoryview,
) -> RetainedCloud:
    """Decode the exact retained 12-bit tuples without synthesizing points."""
    header = inspect(data)
    true_counts, stored_counts, compressed_core = _sections(data, header)
    core = zlib.decompress(compressed_core)
    if len(core) != header.core_uncompressed_size:
        raise ValueError("CP4M core size mismatch")
    quantized = _decode_grouped_core(core, stored_counts)
    if len(quantized) != header.stored_point_count:
        raise ValueError("CP4M retained point count mismatch")
    minimum = np.asarray(header.minimum, dtype=np.float64)
    maximum = np.asarray(header.maximum, dtype=np.float64)
    bins = np.repeat(
        np.arange(header.spectrum_bin_count, dtype=np.int32),
        stored_counts.astype(np.int64),
    )
    points = _dequantize_values(
        quantized,
        minimum,
        maximum,
    ).astype(np.float32)
    points[:, 3] = _force_mass_bins(
        points[:, 3],
        bins,
        header.spectrum_bin_da,
        header.spectrum_bin_count,
    )
    return RetainedCloud(
        header=header,
        quantized=quantized,
        points=points,
        bins=bins,
        true_counts=true_counts.copy(),
        stored_counts=stored_counts.copy(),
    )


def _hash_uniform(
    count: int,
    seed: int,
    bin_index: int,
    dimension: int,
) -> np.ndarray:
    index = np.arange(count, dtype=np.uint64)
    value = (
        index * np.uint64(0x9E3779B1)
        + np.uint64(seed & UINT32_MAX)
        + np.uint64(bin_index * 0x85EBCA6B)
        + np.uint64(dimension * 0xC2B2AE35)
    ) & np.uint64(UINT32_MAX)
    value ^= value >> np.uint64(16)
    value = (value * np.uint64(0x7FEB352D)) & np.uint64(UINT32_MAX)
    value ^= value >> np.uint64(15)
    value = (value * np.uint64(0x846CA68B)) & np.uint64(UINT32_MAX)
    value ^= value >> np.uint64(16)
    return value.astype(np.float64) / (UINT32_MAX + 1.0)


def _noise_values(
    count: int,
    seed: int,
    bin_index: int,
    mode: int,
) -> np.ndarray:
    if mode == NOISE_NONE:
        return np.zeros((count, 4), dtype=np.float64)
    uniforms = np.column_stack([
        _hash_uniform(count, seed, bin_index, dimension)
        for dimension in range(4)
    ])
    if mode == NOISE_UNIFORM:
        return uniforms - 0.5
    if mode == NOISE_GAUSSIAN:
        companion = np.column_stack([
            _hash_uniform(count, seed ^ 0xA511E9B3, bin_index, dimension)
            for dimension in range(4)
        ])
        gaussian = np.sqrt(-2.0 * np.log(np.maximum(uniforms, 1e-12))) * np.cos(
            2.0 * np.pi * companion,
        )
        return np.clip(gaussian * 0.22, -0.5, 0.5)
    raise ValueError("noise must be 'none', 'uniform', or 'gaussian'")


def _noise_mode(noise: str | int | None, default: int) -> int:
    if noise is None:
        return default
    if isinstance(noise, str):
        mapping = {
            "none": NOISE_NONE,
            "uniform": NOISE_UNIFORM,
            "gaussian": NOISE_GAUSSIAN,
        }
        if noise not in mapping:
            raise ValueError("noise must be 'none', 'uniform', or 'gaussian'")
        return mapping[noise]
    if noise in (NOISE_NONE, NOISE_UNIFORM, NOISE_GAUSSIAN):
        return int(noise)
    raise ValueError("invalid noise mode")


def decode(
    data: bytes | bytearray | memoryview,
    *,
    noise: str | int | None = None,
) -> DecodedCloud:
    """Decode and deterministically expand back to the original point count."""
    retained = decode_retained(data)
    header = retained.header
    mode = _noise_mode(noise, header.default_noise)
    minimum = np.asarray(header.minimum, dtype=np.float64)
    maximum = np.asarray(header.maximum, dtype=np.float64)
    extent = maximum - minimum

    order = np.argsort(retained.bins, kind="stable")
    grouped = retained.quantized[order]
    starts = np.empty(header.spectrum_bin_count + 1, dtype=np.int64)
    starts[0] = 0
    np.cumsum(retained.stored_counts, dtype=np.int64, out=starts[1:])
    output = np.empty((header.original_point_count, 4), dtype=np.float32)
    exact = np.empty(header.original_point_count, dtype=np.bool_)
    output_bins = np.empty(header.original_point_count, dtype=np.uint32)
    cursor = 0
    for bin_index in np.flatnonzero(retained.true_counts):
        total = int(retained.true_counts[bin_index])
        stored = int(retained.stored_counts[bin_index])
        seeds = grouped[starts[bin_index]:starts[bin_index + 1]]
        if len(seeds) != stored or stored == 0:
            raise ValueError("CP4M cannot expand a bin without retained seeds")

        exact_points = _dequantize_values(seeds, minimum, maximum)
        exact_points[:, 3] = _force_mass_bins(
            exact_points[:, 3],
            np.full(stored, bin_index, dtype=np.int32),
            header.spectrum_bin_da,
            header.spectrum_bin_count,
        )
        output[cursor:cursor + stored] = exact_points.astype(np.float32)
        exact[cursor:cursor + stored] = True
        output_bins[cursor:cursor + stored] = bin_index
        cursor += stored

        synthesized = total - stored
        if synthesized:
            parents = np.floor(
                (np.arange(synthesized, dtype=np.float64) + 0.5)
                * stored / synthesized,
            ).astype(np.int64)
            values = seeds[parents].astype(np.float64)
            values += _noise_values(
                synthesized,
                header.seed,
                int(bin_index),
                mode,
            )
            values = np.clip(values, 0, MASK12)
            points = minimum + values / MASK12 * extent
            lower = header.spectrum_min_da + (
                bin_index * header.spectrum_bin_da
            )
            upper = min(
                header.spectrum_max_da,
                lower + header.spectrum_bin_da,
            )
            points[:, 3] = np.clip(
                points[:, 3],
                np.nextafter(lower, upper),
                np.nextafter(upper, lower),
            )
            points[:, 3] = _force_mass_bins(
                points[:, 3],
                np.full(synthesized, bin_index, dtype=np.int32),
                header.spectrum_bin_da,
                header.spectrum_bin_count,
            )
            output[cursor:cursor + synthesized] = points.astype(np.float32)
            exact[cursor:cursor + synthesized] = False
            output_bins[cursor:cursor + synthesized] = bin_index
            cursor += synthesized
    if cursor != header.original_point_count:
        raise AssertionError("expanded point count mismatch")
    return DecodedCloud(
        header=header,
        points=output,
        exact=exact,
        bins=output_bins,
        true_counts=retained.true_counts,
        stored_counts=retained.stored_counts,
    )
