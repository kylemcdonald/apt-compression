"""Experimental exact codecs for four unsigned 12-bit APT fields."""

from __future__ import annotations

import io
import struct
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import lzma
import numpy as np
import zstandard

BITS = 12
LEVELS = 1 << BITS
MASK12 = LEVELS - 1
SPATIAL_BITS = 36
SPATIAL_MASK = (1 << SPATIAL_BITS) - 1
MASS_GROUPS = LEVELS
QUANTIZER_METADATA_BYTES = 32


@dataclass(frozen=True)
class QuantizedPoints:
    values: np.ndarray
    minimum: np.ndarray
    maximum: np.ndarray

    @property
    def count(self) -> int:
        return len(self.values)


def read_pos(path: str | Path) -> np.ndarray:
    source = Path(path)
    size = source.stat().st_size
    if size == 0 or size % 16:
        raise ValueError(f"{source} is not a non-empty four-column POS file")
    mapped = np.memmap(source, dtype=">f4", mode="r", shape=(size // 16, 4))
    points = np.asarray(mapped, dtype=np.float32)
    if not np.isfinite(points).all():
        raise ValueError(f"{source} contains non-finite values")
    return points


def quantize12(points: np.ndarray) -> QuantizedPoints:
    array = np.asarray(points, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 4 or len(array) == 0:
        raise ValueError("points must have shape (N, 4) with N > 0")
    minimum = array.min(axis=0).astype(np.float32)
    maximum = array.max(axis=0).astype(np.float32)
    extent = maximum.astype(np.float64) - minimum.astype(np.float64)
    safe_extent = np.where(extent > 0, extent, 1.0)
    normalized = (
        array.astype(np.float64) - minimum.astype(np.float64)
    ) / safe_extent
    values = np.clip(
        np.floor(normalized * MASK12 + 0.5),
        0,
        MASK12,
    ).astype(np.uint16)
    return QuantizedPoints(values=values, minimum=minimum, maximum=maximum)


def quantizer_metadata(quantized: QuantizedPoints) -> bytes:
    return (
        quantized.minimum.astype("<f4", copy=False).tobytes()
        + quantized.maximum.astype("<f4", copy=False).tobytes()
    )


def _spread_table(dimensions: int) -> np.ndarray:
    table = np.zeros(LEVELS, dtype=np.uint64)
    for value in range(LEVELS):
        spread = 0
        for bit in range(BITS):
            spread |= ((value >> bit) & 1) << (bit * dimensions)
        table[value] = spread
    return table


MORTON3 = _spread_table(3)
MORTON4 = _spread_table(4)


def spatial_morton(values: np.ndarray) -> np.ndarray:
    return (
        MORTON3[values[:, 0]]
        | (MORTON3[values[:, 1]] << np.uint64(1))
        | (MORTON3[values[:, 2]] << np.uint64(2))
    )


def spatial_hilbert(values: np.ndarray) -> np.ndarray:
    """Return 36-bit Hilbert distances for three 12-bit coordinates."""
    source = np.asarray(values, dtype=np.uint16)
    if source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("values must have shape (N, 3)")
    axes = source.astype(np.uint64).T.copy()

    # Skilling's axes-to-transpose transform, vectorized over points.
    q = 1 << (BITS - 1)
    while q > 1:
        p = q - 1
        for dimension in range(3):
            exchange = (axes[0] ^ axes[dimension]) & np.uint64(p)
            inverted = (axes[dimension] & np.uint64(q)) != 0
            axes[0] ^= np.where(inverted, np.uint64(p), exchange)
            axes[dimension] ^= np.where(
                inverted,
                np.uint64(0),
                exchange,
            )
        q >>= 1
    axes[1] ^= axes[0]
    axes[2] ^= axes[1]
    correction = np.zeros(source.shape[0], dtype=np.uint64)
    q = 1 << (BITS - 1)
    while q > 1:
        correction ^= np.where(
            (axes[2] & np.uint64(q)) != 0,
            np.uint64(q - 1),
            np.uint64(0),
        )
        q >>= 1
    axes ^= correction

    distance = np.zeros(source.shape[0], dtype=np.uint64)
    for bit in range(BITS):
        for dimension in range(3):
            distance |= (
                (axes[dimension] >> np.uint64(bit)) & np.uint64(1)
            ) << np.uint64(bit * 3 + (2 - dimension))
    return distance


def spatial_hilbert_inverse(distance: np.ndarray) -> np.ndarray:
    """Invert :func:`spatial_hilbert` exactly."""
    source = np.asarray(distance, dtype=np.uint64)
    axes = np.zeros((3, len(source)), dtype=np.uint64)
    for bit in range(BITS):
        for dimension in range(3):
            axes[dimension] |= (
                (
                    source >> np.uint64(bit * 3 + (2 - dimension))
                ) & np.uint64(1)
            ) << np.uint64(bit)

    correction = axes[2] >> np.uint64(1)
    axes[2] ^= axes[1]
    axes[1] ^= axes[0]
    axes[0] ^= correction
    q = 2
    while q != LEVELS:
        p = q - 1
        for dimension in range(2, -1, -1):
            exchange = (axes[0] ^ axes[dimension]) & np.uint64(p)
            inverted = (axes[dimension] & np.uint64(q)) != 0
            axes[0] ^= np.where(inverted, np.uint64(p), exchange)
            axes[dimension] ^= np.where(
                inverted,
                np.uint64(0),
                exchange,
            )
        q <<= 1
    return axes.T.astype(np.uint16)


def morton4(values: np.ndarray) -> np.ndarray:
    return (
        MORTON4[values[:, 0]]
        | (MORTON4[values[:, 1]] << np.uint64(1))
        | (MORTON4[values[:, 2]] << np.uint64(2))
        | (MORTON4[values[:, 3]] << np.uint64(3))
    )


def tuple_keys(values: np.ndarray) -> np.ndarray:
    q = values.astype(np.uint64, copy=False)
    return (
        q[:, 0]
        | (q[:, 1] << np.uint64(12))
        | (q[:, 2] << np.uint64(24))
        | (q[:, 3] << np.uint64(36))
    )


def pack_low_six(keys: np.ndarray) -> bytes:
    little = np.asarray(keys, dtype="<u8")
    octets = little.view(np.uint8).reshape(-1, 8)
    return octets[:, :6].copy().tobytes()


def unpack_low_six(data: bytes) -> np.ndarray:
    if len(data) % 6:
        raise ValueError("six-byte key stream is truncated")
    count = len(data) // 6
    octets = np.zeros((count, 8), dtype=np.uint8)
    octets[:, :6] = np.frombuffer(data, dtype=np.uint8).reshape(-1, 6)
    return octets.view("<u8").reshape(-1)


def pack_columns12(values: np.ndarray) -> bytes:
    output = io.BytesIO()
    for column in range(4):
        source = values[:, column].astype(np.uint16, copy=False)
        if len(source) % 2:
            source = np.append(source, np.uint16(0))
        pairs = source.reshape(-1, 2)
        packed = np.empty((len(pairs), 3), dtype=np.uint8)
        packed[:, 0] = pairs[:, 0] & 0xFF
        packed[:, 1] = (pairs[:, 0] >> 8) | ((pairs[:, 1] & 0xF) << 4)
        packed[:, 2] = pairs[:, 1] >> 4
        output.write(packed.tobytes())
    return output.getvalue()


def unpack_columns12(data: bytes, count: int) -> np.ndarray:
    bytes_per_column = ((count + 1) // 2) * 3
    if len(data) != bytes_per_column * 4:
        raise ValueError("column stream has an unexpected length")
    output = np.empty((count, 4), dtype=np.uint16)
    for column in range(4):
        start = column * bytes_per_column
        packed = np.frombuffer(
            data,
            dtype=np.uint8,
            count=bytes_per_column,
            offset=start,
        ).reshape(-1, 3)
        pairs = np.empty((len(packed), 2), dtype=np.uint16)
        pairs[:, 0] = packed[:, 0].astype(np.uint16) | (
            (packed[:, 1] & 0xF).astype(np.uint16) << 8
        )
        pairs[:, 1] = (packed[:, 1].astype(np.uint16) >> 4) | (
            packed[:, 2].astype(np.uint16) << 4
        )
        output[:, column] = pairs.reshape(-1)[:count]
    return output


def pack_bitplanes(values: np.ndarray, bits: int) -> bytes:
    if bits == 0 or len(values) == 0:
        return b""
    source = np.asarray(values, dtype=np.uint64)
    output = io.BytesIO()
    for bit in range(bits):
        plane = ((source >> np.uint64(bit)) & np.uint64(1)).astype(np.uint8)
        output.write(np.packbits(plane, bitorder="little").tobytes())
    return output.getvalue()


def unpack_bitplanes(data: bytes, count: int, bits: int) -> np.ndarray:
    if bits == 0:
        return np.zeros(count, dtype=np.uint64)
    stride = (count + 7) // 8
    if len(data) != stride * bits:
        raise ValueError("bitplane stream has an unexpected length")
    output = np.zeros(count, dtype=np.uint64)
    for bit in range(bits):
        plane = np.unpackbits(
            np.frombuffer(data, dtype=np.uint8, count=stride, offset=bit * stride),
            bitorder="little",
        )[:count]
        output |= plane.astype(np.uint64) << np.uint64(bit)
    return output


def pack_all_bitplanes(values: np.ndarray) -> bytes:
    output = io.BytesIO()
    for column in range(4):
        output.write(pack_bitplanes(values[:, column], BITS))
    return output.getvalue()


def varint_encode(values: np.ndarray) -> bytes:
    source = np.asarray(values, dtype=np.uint64)
    if len(source) == 0:
        return b""
    lengths = np.ones(len(source), dtype=np.uint8)
    for shift in range(7, 49, 7):
        lengths += source >= np.uint64(1 << shift)
    offsets = np.empty(len(source), dtype=np.int64)
    offsets[0] = 0
    if len(source) > 1:
        np.cumsum(lengths[:-1], dtype=np.int64, out=offsets[1:])
    output = np.empty(int(lengths.sum(dtype=np.uint64)), dtype=np.uint8)
    for byte_index in range(7):
        mask = lengths > byte_index
        if not mask.any():
            break
        chunk = (
            (source[mask] >> np.uint64(byte_index * 7)) & np.uint64(0x7F)
        ).astype(np.uint8)
        chunk[lengths[mask] > byte_index + 1] |= 0x80
        output[offsets[mask] + byte_index] = chunk
    return output.tobytes()


def varint_decode(data: bytes, count: int) -> np.ndarray:
    source = np.frombuffer(data, dtype=np.uint8)
    output = np.empty(count, dtype=np.uint64)
    value = 0
    shift = 0
    index = 0
    for octet in source:
        value |= int(octet & 0x7F) << shift
        if octet & 0x80:
            shift += 7
            continue
        if index >= count:
            raise ValueError("varint stream contains too many values")
        output[index] = value
        index += 1
        value = 0
        shift = 0
    if index != count or shift:
        raise ValueError("varint stream is truncated")
    return output


def mass_spatial_sorted(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return grouped_curve_sorted(values, primary=3, curve="morton")


def grouped_curve_sorted(
    values: np.ndarray,
    primary: int,
    curve: str,
) -> tuple[np.ndarray, np.ndarray]:
    if primary not in range(4):
        raise ValueError("primary must be in [0, 3]")
    remaining = [dimension for dimension in range(4) if dimension != primary]
    if curve == "morton":
        spatial = spatial_morton(values[:, remaining])
    elif curve == "hilbert":
        spatial = spatial_hilbert(values[:, remaining])
    else:
        raise ValueError("curve must be 'morton' or 'hilbert'")
    combined = (
        values[:, primary].astype(np.uint64) << np.uint64(36)
    ) | spatial
    ordered = np.sort(combined, kind="stable")
    counts = np.bincount(
        (ordered >> np.uint64(36)).astype(np.int64),
        minlength=MASS_GROUPS,
    ).astype(np.uint32)
    return ordered, counts


def spatial_mass_sorted(
    values: np.ndarray,
    curve: str,
) -> np.ndarray:
    if curve == "morton":
        spatial = spatial_morton(values[:, :3])
    elif curve == "hilbert":
        spatial = spatial_hilbert(values[:, :3])
    else:
        raise ValueError("curve must be 'morton' or 'hilbert'")
    combined = (
        spatial << np.uint64(12)
    ) | values[:, 3].astype(np.uint64)
    return np.sort(combined, kind="stable")


def mass_prefix_spatial_sorted(
    values: np.ndarray,
    curve: str,
    prefix_bits: int,
    axis_order: tuple[int, int, int] = (0, 1, 2),
) -> np.ndarray:
    if prefix_bits not in range(BITS + 1):
        raise ValueError("prefix_bits must be in [0, 12]")
    if sorted(axis_order) != [0, 1, 2]:
        raise ValueError("axis_order must be a permutation of (0, 1, 2)")
    spatial_values = values[:, list(axis_order)]
    if curve == "morton":
        spatial = spatial_morton(spatial_values)
    elif curve == "hilbert":
        spatial = spatial_hilbert(spatial_values)
    else:
        raise ValueError("curve must be 'morton' or 'hilbert'")
    low_bits = BITS - prefix_bits
    mass = values[:, 3].astype(np.uint64)
    prefix = mass >> np.uint64(low_bits)
    suffix_mask = (1 << low_bits) - 1 if low_bits else 0
    suffix = mass & np.uint64(suffix_mask)
    combined = (
        prefix << np.uint64(SPATIAL_BITS + low_bits)
    ) | (spatial << np.uint64(low_bits)) | suffix
    return np.sort(combined, kind="stable")


def grouped_spatial_deltas(
    ordered: np.ndarray,
    counts: np.ndarray,
) -> np.ndarray:
    spatial = ordered & np.uint64(SPATIAL_MASK)
    deltas = np.empty_like(spatial)
    if len(spatial) == 0:
        return deltas
    deltas[0] = spatial[0]
    deltas[1:] = spatial[1:] - spatial[:-1]
    starts = np.cumsum(counts, dtype=np.uint64)[:-1]
    starts = starts[counts[1:] > 0]
    deltas[starts.astype(np.int64)] = spatial[starts.astype(np.int64)]
    return deltas


def encode_mass_delta_varint(
    ordered: np.ndarray,
    counts: np.ndarray,
) -> bytes:
    return (
        b"L12V"
        + struct.pack("<Q", len(ordered))
        + counts.astype("<u4", copy=False).tobytes()
        + varint_encode(grouped_spatial_deltas(ordered, counts))
    )


def decode_mass_delta_varint(data: bytes) -> np.ndarray:
    if data[:4] != b"L12V":
        raise ValueError("invalid mass-delta stream")
    count = struct.unpack_from("<Q", data, 4)[0]
    offset = 12
    counts = np.frombuffer(data, dtype="<u4", count=MASS_GROUPS, offset=offset)
    offset += MASS_GROUPS * 4
    deltas = varint_decode(data[offset:], count)
    output = np.empty(count, dtype=np.uint64)
    cursor = 0
    for mass, group_count in enumerate(counts):
        n = int(group_count)
        if not n:
            continue
        spatial = np.cumsum(deltas[cursor:cursor + n], dtype=np.uint64)
        output[cursor:cursor + n] = (
            np.uint64(mass) << np.uint64(36)
        ) | spatial
        cursor += n
    if cursor != count:
        raise ValueError("mass histogram does not match point count")
    return output


def _group_slices(counts: np.ndarray):
    cursor = 0
    for mass, count in enumerate(counts):
        n = int(count)
        if n:
            yield mass, cursor, cursor + n
            cursor += n


def encode_mass_elias_fano(
    ordered: np.ndarray,
    counts: np.ndarray,
) -> bytes:
    low_bits = np.zeros(MASS_GROUPS, dtype=np.uint8)
    high_lengths = np.zeros(MASS_GROUPS, dtype="<u8")
    streams: list[bytes] = []
    for _, start, end in _group_slices(counts):
        spatial = ordered[start:end] & np.uint64(SPATIAL_MASK)
        n = len(spatial)
        ratio = (1 << SPATIAL_BITS) // n
        width = max(0, min(SPATIAL_BITS, ratio.bit_length() - 1))
        low_bits[int(ordered[start] >> np.uint64(36))] = width
        low = spatial & np.uint64((1 << width) - 1 if width else 0)
        low_stream = pack_bitplanes(low, width)
        high = spatial >> np.uint64(width)
        positions = high + np.arange(n, dtype=np.uint64)
        high_length = int(positions[-1]) + 1
        high_lengths[int(ordered[start] >> np.uint64(36))] = high_length
        unary = np.zeros(high_length, dtype=np.uint8)
        unary[positions.astype(np.int64)] = 1
        high_stream = np.packbits(unary, bitorder="little").tobytes()
        streams.extend((low_stream, high_stream))
    return (
        b"L12E"
        + struct.pack("<Q", len(ordered))
        + counts.astype("<u4", copy=False).tobytes()
        + low_bits.tobytes()
        + high_lengths.tobytes()
        + b"".join(streams)
    )


def decode_mass_elias_fano(data: bytes) -> np.ndarray:
    if data[:4] != b"L12E":
        raise ValueError("invalid Elias-Fano stream")
    count = struct.unpack_from("<Q", data, 4)[0]
    offset = 12
    counts = np.frombuffer(data, dtype="<u4", count=MASS_GROUPS, offset=offset)
    offset += MASS_GROUPS * 4
    widths = np.frombuffer(data, dtype=np.uint8, count=MASS_GROUPS, offset=offset)
    offset += MASS_GROUPS
    high_lengths = np.frombuffer(
        data,
        dtype="<u8",
        count=MASS_GROUPS,
        offset=offset,
    )
    offset += MASS_GROUPS * 8
    output = np.empty(count, dtype=np.uint64)
    cursor = 0
    for mass, group_count in enumerate(counts):
        n = int(group_count)
        if not n:
            continue
        width = int(widths[mass])
        low_size = ((n + 7) // 8) * width
        low = unpack_bitplanes(data[offset:offset + low_size], n, width)
        offset += low_size
        high_length = int(high_lengths[mass])
        high_size = (high_length + 7) // 8
        unary = np.unpackbits(
            np.frombuffer(data, dtype=np.uint8, count=high_size, offset=offset),
            bitorder="little",
        )[:high_length]
        offset += high_size
        positions = np.flatnonzero(unary).astype(np.uint64)
        if len(positions) != n:
            raise ValueError("invalid Elias-Fano high-bit stream")
        high = positions - np.arange(n, dtype=np.uint64)
        spatial = (high << np.uint64(width)) | low
        output[cursor:cursor + n] = (
            np.uint64(mass) << np.uint64(36)
        ) | spatial
        cursor += n
    if cursor != count or offset != len(data):
        raise ValueError("invalid Elias-Fano stream length")
    return output


def _best_rice_parameter(gaps: np.ndarray) -> int:
    n = len(gaps)
    best_width = 0
    best_bits = None
    for width in range(SPATIAL_BITS + 1):
        quotient_sum = int(
            (gaps >> np.uint64(width)).sum(dtype=np.uint64)
        )
        bits = n * (width + 1) + quotient_sum
        if best_bits is None or bits < best_bits:
            best_bits = bits
            best_width = width
    return best_width


def encode_mass_rice(
    ordered: np.ndarray,
    counts: np.ndarray,
) -> bytes:
    widths = np.zeros(MASS_GROUPS, dtype=np.uint8)
    unary_lengths = np.zeros(MASS_GROUPS, dtype="<u8")
    streams: list[bytes] = []
    for mass, start, end in _group_slices(counts):
        spatial = ordered[start:end] & np.uint64(SPATIAL_MASK)
        gaps = np.empty_like(spatial)
        gaps[0] = spatial[0]
        gaps[1:] = spatial[1:] - spatial[:-1]
        width = _best_rice_parameter(gaps)
        widths[mass] = width
        remainder = gaps & np.uint64((1 << width) - 1 if width else 0)
        remainder_stream = pack_bitplanes(remainder, width)
        quotient = gaps >> np.uint64(width)
        ends = np.cumsum(quotient + np.uint64(1), dtype=np.uint64) - 1
        unary_length = int(ends[-1]) + 1
        unary_lengths[mass] = unary_length
        unary = np.zeros(unary_length, dtype=np.uint8)
        unary[ends.astype(np.int64)] = 1
        unary_stream = np.packbits(unary, bitorder="little").tobytes()
        streams.extend((remainder_stream, unary_stream))
    return (
        b"L12R"
        + struct.pack("<Q", len(ordered))
        + counts.astype("<u4", copy=False).tobytes()
        + widths.tobytes()
        + unary_lengths.tobytes()
        + b"".join(streams)
    )


def decode_mass_rice(data: bytes) -> np.ndarray:
    if data[:4] != b"L12R":
        raise ValueError("invalid Rice stream")
    count = struct.unpack_from("<Q", data, 4)[0]
    offset = 12
    counts = np.frombuffer(data, dtype="<u4", count=MASS_GROUPS, offset=offset)
    offset += MASS_GROUPS * 4
    widths = np.frombuffer(data, dtype=np.uint8, count=MASS_GROUPS, offset=offset)
    offset += MASS_GROUPS
    unary_lengths = np.frombuffer(
        data,
        dtype="<u8",
        count=MASS_GROUPS,
        offset=offset,
    )
    offset += MASS_GROUPS * 8
    output = np.empty(count, dtype=np.uint64)
    cursor = 0
    for mass, group_count in enumerate(counts):
        n = int(group_count)
        if not n:
            continue
        width = int(widths[mass])
        remainder_size = ((n + 7) // 8) * width
        remainder = unpack_bitplanes(
            data[offset:offset + remainder_size],
            n,
            width,
        )
        offset += remainder_size
        unary_length = int(unary_lengths[mass])
        unary_size = (unary_length + 7) // 8
        unary = np.unpackbits(
            np.frombuffer(data, dtype=np.uint8, count=unary_size, offset=offset),
            bitorder="little",
        )[:unary_length]
        offset += unary_size
        ends = np.flatnonzero(unary).astype(np.int64)
        if len(ends) != n:
            raise ValueError("invalid Rice unary stream")
        previous = np.concatenate((np.array([-1], dtype=np.int64), ends[:-1]))
        quotient = (ends - previous - 1).astype(np.uint64)
        gaps = (quotient << np.uint64(width)) | remainder
        spatial = np.cumsum(gaps, dtype=np.uint64)
        output[cursor:cursor + n] = (
            np.uint64(mass) << np.uint64(36)
        ) | spatial
        cursor += n
    if cursor != count or offset != len(data):
        raise ValueError("invalid Rice stream length")
    return output


def encode_mass_rice_split(
    ordered: np.ndarray,
    counts: np.ndarray,
) -> bytes:
    """Rice coding with remainder and unary regions kept contiguous."""
    widths = np.zeros(MASS_GROUPS, dtype=np.uint8)
    unary_lengths = np.zeros(MASS_GROUPS, dtype="<u8")
    remainder_streams: list[bytes] = []
    unary_streams: list[bytes] = []
    for mass, start, end in _group_slices(counts):
        spatial = ordered[start:end] & np.uint64(SPATIAL_MASK)
        gaps = np.empty_like(spatial)
        gaps[0] = spatial[0]
        gaps[1:] = spatial[1:] - spatial[:-1]
        width = _best_rice_parameter(gaps)
        widths[mass] = width
        remainder = gaps & np.uint64((1 << width) - 1 if width else 0)
        remainder_streams.append(pack_bitplanes(remainder, width))
        quotient = gaps >> np.uint64(width)
        ends = np.cumsum(quotient + np.uint64(1), dtype=np.uint64) - 1
        unary_length = int(ends[-1]) + 1
        unary_lengths[mass] = unary_length
        unary = np.zeros(unary_length, dtype=np.uint8)
        unary[ends.astype(np.int64)] = 1
        unary_streams.append(np.packbits(unary, bitorder="little").tobytes())
    return (
        b"L12S"
        + struct.pack("<Q", len(ordered))
        + counts.astype("<u4", copy=False).tobytes()
        + widths.tobytes()
        + unary_lengths.tobytes()
        + b"".join(remainder_streams)
        + b"".join(unary_streams)
    )


def decode_mass_rice_split(data: bytes) -> np.ndarray:
    if data[:4] != b"L12S":
        raise ValueError("invalid split Rice stream")
    count = struct.unpack_from("<Q", data, 4)[0]
    offset = 12
    counts = np.frombuffer(data, dtype="<u4", count=MASS_GROUPS, offset=offset)
    offset += MASS_GROUPS * 4
    widths = np.frombuffer(data, dtype=np.uint8, count=MASS_GROUPS, offset=offset)
    offset += MASS_GROUPS
    unary_lengths = np.frombuffer(
        data,
        dtype="<u8",
        count=MASS_GROUPS,
        offset=offset,
    )
    offset += MASS_GROUPS * 8
    remainder_bytes = sum(
        ((int(n) + 7) // 8) * int(width)
        for n, width in zip(counts, widths, strict=True)
    )
    remainder_cursor = offset
    remainder_end = offset + remainder_bytes
    unary_cursor = remainder_end

    output = np.empty(count, dtype=np.uint64)
    cursor = 0
    for mass, group_count in enumerate(counts):
        n = int(group_count)
        if not n:
            continue
        width = int(widths[mass])
        remainder_size = ((n + 7) // 8) * width
        remainder = unpack_bitplanes(
            data[remainder_cursor:remainder_cursor + remainder_size],
            n,
            width,
        )
        remainder_cursor += remainder_size
        unary_length = int(unary_lengths[mass])
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
        if len(ends) != n:
            raise ValueError("invalid split Rice unary stream")
        previous = np.concatenate((
            np.array([-1], dtype=np.int64),
            ends[:-1],
        ))
        quotient = (ends - previous - 1).astype(np.uint64)
        spatial = np.cumsum(
            (quotient << np.uint64(width)) | remainder,
            dtype=np.uint64,
        )
        output[cursor:cursor + n] = (
            np.uint64(mass) << np.uint64(36)
        ) | spatial
        cursor += n
    if (
        cursor != count
        or remainder_cursor != remainder_end
        or unary_cursor != len(data)
    ):
        raise ValueError("invalid split Rice stream length")
    return output


def encode_mass_rice_compact(
    ordered: np.ndarray,
    counts: np.ndarray,
) -> bytes:
    """Split Rice coding with sparse, varint-coded group metadata."""
    active = np.flatnonzero(counts).astype(np.int64)
    widths = np.empty(len(active), dtype=np.uint8)
    unary_lengths = np.empty(len(active), dtype=np.uint64)
    remainder_streams: list[bytes] = []
    unary_streams: list[bytes] = []
    for active_index, (_, start, end) in enumerate(_group_slices(counts)):
        spatial = ordered[start:end] & np.uint64(SPATIAL_MASK)
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

    group_mask = np.packbits(counts > 0, bitorder="little").tobytes()
    count_stream = varint_encode(counts[active].astype(np.uint64))
    unary_metadata = varint_encode(unary_lengths)
    return (
        b"L12C"
        + struct.pack(
            "<QHII",
            len(ordered),
            len(active),
            len(count_stream),
            len(unary_metadata),
        )
        + group_mask
        + count_stream
        + widths.tobytes()
        + unary_metadata
        + b"".join(remainder_streams)
        + b"".join(unary_streams)
    )


def decode_mass_rice_compact(data: bytes) -> np.ndarray:
    if data[:4] != b"L12C":
        raise ValueError("invalid compact Rice stream")
    count, active_count, counts_size, unary_metadata_size = (
        struct.unpack_from("<QHII", data, 4)
    )
    offset = 22
    mask_size = MASS_GROUPS // 8
    group_mask = np.unpackbits(
        np.frombuffer(data, dtype=np.uint8, count=mask_size, offset=offset),
        bitorder="little",
    )
    offset += mask_size
    active = np.flatnonzero(group_mask).astype(np.int64)
    if len(active) != active_count:
        raise ValueError("compact Rice active-group count mismatch")
    active_counts = varint_decode(
        data[offset:offset + counts_size],
        active_count,
    )
    offset += counts_size
    active_widths = np.frombuffer(
        data,
        dtype=np.uint8,
        count=active_count,
        offset=offset,
    )
    offset += active_count
    active_unary_lengths = varint_decode(
        data[offset:offset + unary_metadata_size],
        active_count,
    )
    offset += unary_metadata_size
    counts = np.zeros(MASS_GROUPS, dtype=np.uint64)
    counts[active] = active_counts
    widths = np.zeros(MASS_GROUPS, dtype=np.uint8)
    widths[active] = active_widths
    unary_lengths = np.zeros(MASS_GROUPS, dtype=np.uint64)
    unary_lengths[active] = active_unary_lengths

    remainder_bytes = sum(
        ((int(n) + 7) // 8) * int(width)
        for n, width in zip(counts, widths, strict=True)
    )
    remainder_cursor = offset
    remainder_end = offset + remainder_bytes
    unary_cursor = remainder_end
    output = np.empty(count, dtype=np.uint64)
    cursor = 0
    for mass, group_count in enumerate(counts):
        n = int(group_count)
        if not n:
            continue
        width = int(widths[mass])
        remainder_size = ((n + 7) // 8) * width
        remainder = unpack_bitplanes(
            data[remainder_cursor:remainder_cursor + remainder_size],
            n,
            width,
        )
        remainder_cursor += remainder_size
        unary_length = int(unary_lengths[mass])
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
        if len(ends) != n:
            raise ValueError("invalid compact Rice unary stream")
        previous = np.concatenate((
            np.array([-1], dtype=np.int64),
            ends[:-1],
        ))
        quotient = (ends - previous - 1).astype(np.uint64)
        spatial = np.cumsum(
            (quotient << np.uint64(width)) | remainder,
            dtype=np.uint64,
        )
        output[cursor:cursor + n] = (
            np.uint64(mass) << np.uint64(36)
        ) | spatial
        cursor += n
    if (
        cursor != count
        or remainder_cursor != remainder_end
        or unary_cursor != len(data)
    ):
        raise ValueError("invalid compact Rice stream length")
    return output


def encode_spatial_mass(ordered: np.ndarray) -> bytes:
    """Encode one sorted spatial stream plus its aligned 12-bit mass labels."""
    spatial = ordered >> np.uint64(12)
    mass = (ordered & np.uint64(MASK12)).astype(np.uint16)
    counts = np.zeros(MASS_GROUPS, dtype=np.uint32)
    counts[0] = len(ordered)
    spatial_stream = encode_mass_rice_split(spatial, counts)
    mass_stream = pack_bitplanes(mass, BITS)
    return (
        b"L12P"
        + struct.pack("<QQ", len(ordered), len(spatial_stream))
        + spatial_stream
        + mass_stream
    )


def decode_spatial_mass(data: bytes) -> np.ndarray:
    if data[:4] != b"L12P":
        raise ValueError("invalid spatial-plus-mass stream")
    count, spatial_size = struct.unpack_from("<QQ", data, 4)
    offset = 20
    spatial = decode_mass_rice_split(data[offset:offset + spatial_size])
    offset += spatial_size
    mass = unpack_bitplanes(data[offset:], count, BITS)
    if len(spatial) != count:
        raise ValueError("spatial-plus-mass point count mismatch")
    return (spatial << np.uint64(12)) | mass


def encode_mass_prefix_spatial(
    ordered: np.ndarray,
    prefix_bits: int,
) -> bytes:
    """Encode mass-prefix groups, spatial gaps, and aligned mass suffixes."""
    if prefix_bits not in range(BITS + 1):
        raise ValueError("prefix_bits must be in [0, 12]")
    low_bits = BITS - prefix_bits
    suffix_mask = (1 << low_bits) - 1 if low_bits else 0
    suffix = (ordered & np.uint64(suffix_mask)).astype(np.uint16)
    spatial = (
        ordered >> np.uint64(low_bits)
    ) & np.uint64(SPATIAL_MASK)
    prefix = ordered >> np.uint64(SPATIAL_BITS + low_bits)
    group_spatial = (prefix << np.uint64(SPATIAL_BITS)) | spatial
    counts = np.bincount(
        prefix.astype(np.int64),
        minlength=MASS_GROUPS,
    ).astype(np.uint32)
    spatial_stream = encode_mass_rice_split(group_spatial, counts)
    suffix_stream = pack_bitplanes(suffix, low_bits)
    return (
        b"L12H"
        + struct.pack(
            "<BQQ",
            prefix_bits,
            len(ordered),
            len(spatial_stream),
        )
        + spatial_stream
        + suffix_stream
    )


def decode_mass_prefix_spatial(data: bytes) -> np.ndarray:
    if data[:4] != b"L12H":
        raise ValueError("invalid mass-prefix stream")
    prefix_bits, count, spatial_size = struct.unpack_from("<BQQ", data, 4)
    if prefix_bits > BITS:
        raise ValueError("invalid mass-prefix width")
    low_bits = BITS - prefix_bits
    offset = 21
    group_spatial = decode_mass_rice_split(
        data[offset:offset + spatial_size],
    )
    offset += spatial_size
    suffix = unpack_bitplanes(data[offset:], count, low_bits)
    if len(group_spatial) != count:
        raise ValueError("mass-prefix point count mismatch")
    prefix = group_spatial >> np.uint64(SPATIAL_BITS)
    spatial = group_spatial & np.uint64(SPATIAL_MASK)
    return (
        prefix << np.uint64(SPATIAL_BITS + low_bits)
    ) | (spatial << np.uint64(low_bits)) | suffix


def encode_mass_rice_blocks(
    ordered: np.ndarray,
    counts: np.ndarray,
    block_size: int = 1024,
) -> bytes:
    """Mass-grouped spatial Morton gaps with a Rice parameter per block."""
    if block_size <= 0 or block_size > 65535:
        raise ValueError("block_size must be in [1, 65535]")
    widths: list[int] = []
    unary_lengths: list[int] = []
    streams: list[bytes] = []
    for _, start, end in _group_slices(counts):
        previous = np.uint64(0)
        for block_start in range(start, end, block_size):
            block_end = min(block_start + block_size, end)
            spatial = (
                ordered[block_start:block_end] & np.uint64(SPATIAL_MASK)
            )
            gaps = np.empty_like(spatial)
            gaps[0] = spatial[0] - previous
            gaps[1:] = spatial[1:] - spatial[:-1]
            previous = spatial[-1]
            width = _best_rice_parameter(gaps)
            widths.append(width)
            remainder = gaps & np.uint64((1 << width) - 1 if width else 0)
            remainder_stream = pack_bitplanes(remainder, width)
            quotient = gaps >> np.uint64(width)
            ends = np.cumsum(quotient + np.uint64(1), dtype=np.uint64) - 1
            unary_length = int(ends[-1]) + 1
            unary_lengths.append(unary_length)
            unary = np.zeros(unary_length, dtype=np.uint8)
            unary[ends.astype(np.int64)] = 1
            unary_stream = np.packbits(unary, bitorder="little").tobytes()
            streams.extend((remainder_stream, unary_stream))

    widths_array = np.asarray(widths, dtype=np.uint8)
    unary_array = np.asarray(unary_lengths, dtype="<u8")
    return (
        b"L12B"
        + struct.pack("<QHI", len(ordered), block_size, len(widths))
        + counts.astype("<u4", copy=False).tobytes()
        + widths_array.tobytes()
        + unary_array.tobytes()
        + b"".join(streams)
    )


def decode_mass_rice_blocks(data: bytes) -> np.ndarray:
    if data[:4] != b"L12B":
        raise ValueError("invalid block-Rice stream")
    count, block_size, block_count = struct.unpack_from("<QHI", data, 4)
    offset = 18
    counts = np.frombuffer(data, dtype="<u4", count=MASS_GROUPS, offset=offset)
    offset += MASS_GROUPS * 4
    widths = np.frombuffer(data, dtype=np.uint8, count=block_count, offset=offset)
    offset += block_count
    unary_lengths = np.frombuffer(
        data,
        dtype="<u8",
        count=block_count,
        offset=offset,
    )
    offset += block_count * 8

    output = np.empty(count, dtype=np.uint64)
    cursor = 0
    block_index = 0
    for mass, group_count in enumerate(counts):
        n = int(group_count)
        if not n:
            continue
        previous = np.uint64(0)
        group_end = cursor + n
        while cursor < group_end:
            take = min(block_size, group_end - cursor)
            width = int(widths[block_index])
            remainder_size = ((take + 7) // 8) * width
            remainder = unpack_bitplanes(
                data[offset:offset + remainder_size],
                take,
                width,
            )
            offset += remainder_size
            unary_length = int(unary_lengths[block_index])
            unary_size = (unary_length + 7) // 8
            unary = np.unpackbits(
                np.frombuffer(
                    data,
                    dtype=np.uint8,
                    count=unary_size,
                    offset=offset,
                ),
                bitorder="little",
            )[:unary_length]
            offset += unary_size
            ends = np.flatnonzero(unary).astype(np.int64)
            if len(ends) != take:
                raise ValueError("invalid block-Rice unary stream")
            before = np.concatenate((
                np.array([-1], dtype=np.int64),
                ends[:-1],
            ))
            quotient = (ends - before - 1).astype(np.uint64)
            gaps = (quotient << np.uint64(width)) | remainder
            spatial = previous + np.cumsum(gaps, dtype=np.uint64)
            output[cursor:cursor + take] = (
                np.uint64(mass) << np.uint64(36)
            ) | spatial
            previous = spatial[-1]
            cursor += take
            block_index += 1
    if (
        cursor != count
        or block_index != block_count
        or offset != len(data)
    ):
        raise ValueError("invalid block-Rice stream length")
    return output


def zstd_compress(data: bytes, level: int = 19, threads: int = -1) -> bytes:
    return zstandard.ZstdCompressor(
        level=level,
        threads=threads,
        write_checksum=True,
    ).compress(data)


def xz_compress(data: bytes, preset: int = 9) -> bytes:
    return lzma.compress(data, preset=preset | lzma.PRESET_EXTREME)


def brotli_compress(data: bytes, quality: int = 11) -> bytes:
    completed = subprocess.run(
        ("brotli", f"--quality={quality}", "--stdout"),
        input=data,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return completed.stdout


def brotli_compress_large(data: bytes, quality: int = 11) -> bytes:
    completed = subprocess.run(
        (
            "brotli",
            f"--quality={quality}",
            "--large_window=30",
            "--stdout",
        ),
        input=data,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return completed.stdout


def timed(function: Callable, *args, **kwargs):
    started = time.perf_counter()
    result = function(*args, **kwargs)
    return result, time.perf_counter() - started
