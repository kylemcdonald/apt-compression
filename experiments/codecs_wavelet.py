"""Wavelet backbone: 3D biorthogonal wavelet on sqrt(counts), keep the top-T
coefficients, sparse-index them with delta varints, quantize to int16."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pywt

import bitpack
import common

WAVELET = "bior4.4"
LEVEL = 3


def encode_grid(grid: np.ndarray, path: Path, budget_coeffs=400_000) -> dict:
    vol = np.sqrt(grid.astype(np.float32))
    coeffs = pywt.wavedecn(vol, WAVELET, level=LEVEL, mode="periodization")
    arr, _ = pywt.coeffs_to_array(coeffs)
    flat = arr.ravel()
    t = min(budget_coeffs, flat.size)
    if t < flat.size:
        thresh = np.partition(np.abs(flat), -t)[-t]
        keep = np.abs(flat) >= thresh
    else:
        keep = np.ones_like(flat, dtype=bool)
    idx = np.nonzero(keep)[0]
    vals = flat[idx]
    scale = float(np.abs(vals).max()) / 32767.0 if len(vals) else 1.0
    q = np.round(vals / max(scale, 1e-12)).astype(np.int16)
    delta = np.diff(idx.astype(np.uint64), prepend=np.uint64(0))
    idx_blob = np.frombuffer(bitpack.varint_encode(delta), dtype=np.uint8)
    common.zsave(
        path,
        idx_blob=idx_blob,
        vals=q,
        scale=np.array([scale], dtype=np.float64),
        arr_shape=np.array(arr.shape, dtype=np.int32),
        grid_shape=np.array(grid.shape, dtype=np.int32),
        n=np.array([len(idx)], dtype=np.int64),
    )
    return {"kept_coeffs": int(len(idx))}


def decode_grid(path: Path) -> np.ndarray:
    d = common.zload(path)
    grid_shape = tuple(d["grid_shape"])
    n = int(d["n"][0])
    delta = bitpack.varint_decode(d["idx_blob"].tobytes(), n)
    idx = np.cumsum(delta.astype(np.uint64)).astype(np.int64)
    arr = np.zeros(tuple(d["arr_shape"]), dtype=np.float32)
    arr.ravel()[idx] = d["vals"].astype(np.float32) * float(d["scale"][0])
    dummy = pywt.wavedecn(np.zeros(grid_shape, dtype=np.float32), WAVELET,
                          level=LEVEL, mode="periodization")
    _, slices = pywt.coeffs_to_array(dummy)
    coeffs = pywt.array_to_coeffs(arr, slices, output_format="wavedecn")
    vol = pywt.waverecn(coeffs, WAVELET, mode="periodization")
    vol = vol[: grid_shape[0], : grid_shape[1], : grid_shape[2]]
    return np.clip(vol, 0.0, None) ** 2
