"""Bit-packing and integer coding helpers (numpy-vectorized)."""

from __future__ import annotations

import numpy as np


def pack_bits(values: np.ndarray, width: int) -> bytes:
    """Pack nonnegative ints into a dense little-endian bitstream of `width` bits."""
    v = values.astype(np.uint64)
    n = len(v)
    total_bits = n * width
    out = np.zeros((total_bits + 7) // 8, dtype=np.uint8)
    bitpos = np.arange(n, dtype=np.uint64) * width
    for b in range(width):
        bit = ((v >> b) & 1).astype(np.uint8)
        pos = bitpos + b
        np.bitwise_or.at(out, (pos // 8).astype(np.int64), bit << (pos % 8).astype(np.uint8))
    return out.tobytes()


def unpack_bits(data: bytes, width: int, count: int) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    bits = np.unpackbits(arr, bitorder="little")
    bits = bits[: count * width].reshape(count, width).astype(np.uint64)
    weights = (1 << np.arange(width, dtype=np.uint64))
    return (bits * weights).sum(axis=1)


def varint_encode(values: np.ndarray) -> bytes:
    """LEB128-style varint encoding for a uint64 array."""
    v = values.astype(np.uint64)
    nbytes = np.maximum((64 - np.clip(_bit_length(v), 1, 64)), 0)
    lengths = np.maximum((_bit_length(v) + 6) // 7, 1)
    total = int(lengths.sum())
    out = np.zeros(total, dtype=np.uint8)
    offsets = np.zeros(len(v), dtype=np.int64)
    np.cumsum(lengths[:-1], out=offsets[1:])
    rem = v.copy()
    max_len = int(lengths.max()) if len(lengths) else 0
    for i in range(max_len):
        active = lengths > i
        byte = (rem[active] & 0x7F).astype(np.uint8)
        more = (lengths[active] > i + 1).astype(np.uint8) << 7
        out[offsets[active] + i] = byte | more
        rem[active] >>= np.uint64(7)
    return out.tobytes()


def varint_decode(data: bytes, count: int) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    ends = np.nonzero((arr & 0x80) == 0)[0][:count]
    starts = np.concatenate([[0], ends[:-1] + 1])
    out = np.zeros(count, dtype=np.uint64)
    max_len = int((ends - starts).max()) + 1 if count else 0
    for i in range(max_len):
        active = starts + i <= ends
        out[active] |= (arr[starts[active] + i].astype(np.uint64) & 0x7F) << np.uint64(7 * i)
    return out


def _bit_length(v: np.ndarray) -> np.ndarray:
    out = np.zeros(len(v), dtype=np.int64)
    x = v.copy()
    while True:
        nz = x > 0
        if not nz.any():
            break
        out[nz] += 1
        x = x >> np.uint64(1)
    return out


def morton3(x: np.ndarray, y: np.ndarray, z: np.ndarray, bits: int) -> np.ndarray:
    """Interleave three `bits`-wide ints into a Morton code (uint64)."""
    return morton3_aniso(x, y, z, (bits, bits, bits))


def demorton3(code: np.ndarray, bits: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return demorton3_aniso(code, (bits, bits, bits))


def _aniso_positions(bits_xyz) -> list[list[int]]:
    """Code-bit position of each axis bit, LSB-aligned round-robin.

    LSB alignment groups same-physical-scale bits together when per-axis bit
    widths are chosen to equalize quantization step, so Z-order locality is
    preserved for anisotropic boxes. Equal widths reproduce classic Morton.
    """
    positions: list[list[int]] = [[], [], []]
    pos = 0
    for b in range(max(bits_xyz)):
        for a in range(3):
            if b < bits_xyz[a]:
                positions[a].append(pos)
                pos += 1
    return positions


def morton3_aniso(x: np.ndarray, y: np.ndarray, z: np.ndarray, bits_xyz) -> np.ndarray:
    """Interleave ints of per-axis widths (bx, by, bz) into one code."""
    code = np.zeros(len(x), dtype=np.uint64)
    axes = (x.astype(np.uint64), y.astype(np.uint64), z.astype(np.uint64))
    positions = _aniso_positions(tuple(bits_xyz))
    for a in range(3):
        for b, pos in enumerate(positions[a]):
            code |= ((axes[a] >> np.uint64(b)) & np.uint64(1)) << np.uint64(pos)
    return code


def demorton3_aniso(code: np.ndarray, bits_xyz) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    out = [np.zeros(len(code), dtype=np.uint64) for _ in range(3)]
    positions = _aniso_positions(tuple(bits_xyz))
    for a in range(3):
        for b, pos in enumerate(positions[a]):
            out[a] |= ((code >> np.uint64(pos)) & np.uint64(1)) << np.uint64(b)
    return out[0], out[1], out[2]
