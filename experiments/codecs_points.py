"""Point-cloud codecs: random subsample baseline, and full-count quantized
points with Morton-order delta coding + varint bit packing."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import bitpack
import common
import spectrum_store


# --- subsample baseline ------------------------------------------------------


def encode_subsample(pts: np.ndarray, ranging, outdir: Path, fraction=0.1, seed=0):
    rng = np.random.default_rng(seed)
    keep = rng.random(len(pts)) < fraction
    sub = pts[keep]
    bounds = np.stack([pts[:, :3].min(axis=0), pts[:, :3].max(axis=0)])
    mass_max = float(pts[:, 3].max())
    xyz_q = np.round(
        (sub[:, :3] - bounds[0]) / (bounds[1] - bounds[0]) * 65535.0
    ).astype(np.uint16)
    mass_q = np.round(sub[:, 3] / mass_max * 65535.0).astype(np.uint16)
    common.zsave(outdir / "points.zst", xyz=xyz_q, mass=mass_q,
                 bounds=bounds.astype(np.float32),
                 mass_max=np.array([mass_max], dtype=np.float32))
    (outdir / "manifest.json").write_text(json.dumps({"fraction": fraction}))


def decode_subsample(outdir: Path, rng: np.random.Generator) -> np.ndarray:
    d = common.zload(outdir / "points.zst")
    bounds = d["bounds"].astype(np.float32)
    xyz = bounds[0] + d["xyz"].astype(np.float32) / 65535.0 * (bounds[1] - bounds[0])
    mass = d["mass"].astype(np.float32) / 65535.0 * float(d["mass_max"][0])
    return np.column_stack([xyz, mass]).astype(np.float32)


# --- exact rare species + sampled abundant species --------------------------


def encode_hybrid_sample(
    pts: np.ndarray,
    ranging,
    outdir: Path,
    fraction=0.02,
    sample_cap=None,
    exact_threshold=100_000,
    bits=12,
    kernel_sigma_nm=0.20,
    preserve_spectrum=True,
    seed=0,
):
    """Store rare species in full and a spatial sample of abundant species.

    Unlike the grid hybrids, this keeps point-supported high-frequency
    structure for abundant species. Decode expands each sampled point into a
    small Gaussian kernel so the original species counts and fine spectrum are
    restored without introducing grid-cell boundaries.
    """
    if sample_cap is None and not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    if sample_cap is not None and sample_cap <= 0:
        raise ValueError("sample_cap must be positive")
    bounds = np.stack([pts[:, :3].min(axis=0), pts[:, :3].max(axis=0)])
    species = ranging.assign(pts[:, 3])
    counts = np.bincount(species, minlength=len(ranging.labels))
    spectrum_store.encode(pts[:, 3], outdir / "spectrum.zst")
    exact_ids = [
        s for s in range(1, len(ranging.labels))
        if 0 < counts[s] < exact_threshold
    ]
    sampled_ids = [
        s for s in range(len(ranging.labels))
        if counts[s] > 0 and s not in exact_ids
    ]

    rng = np.random.default_rng(seed)
    blobs = {}
    stored_counts = {}
    for s in range(len(ranging.labels)):
        n = int(counts[s])
        if n == 0:
            stored_counts[str(s)] = 0
            continue
        sub = pts[species == s]
        if s in exact_ids:
            stored_n = n
        elif sample_cap is not None:
            stored_n = min(n, int(sample_cap))
        else:
            stored_n = max(1, int(round(n * fraction)))
        if stored_n < n:
            chosen = rng.choice(n, stored_n, replace=False)
            sub = sub[chosen]
        stored_counts[str(s)] = stored_n
        pos_blob, order = pack_species_points_with_order(sub[:, :3], bounds, bits)
        blobs[f"pos_{s}"] = pos_blob
        mass_bin = np.floor(sub[order, 3] / common.SPECTRUM_BIN_DA)
        blobs[f"mass_{s}"] = mass_bin.clip(0, 65535).astype(np.uint16).tobytes()

    payload = b""
    index = []
    for key, blob in blobs.items():
        index.append([key, len(blob)])
        payload += blob
    (outdir / "points.zst").write_bytes(common.zbytes(payload))
    manifest = {
        "codec": "hybrid_sample",
        "fraction": fraction,
        "sample_cap": sample_cap,
        "exact_threshold": exact_threshold,
        "bits": bits,
        "kernel_sigma_nm": kernel_sigma_nm,
        "preserve_spectrum": preserve_spectrum,
        "mass_bin_da": common.SPECTRUM_BIN_DA,
        "bounds": bounds.tolist(),
        "exact_ids": exact_ids,
        "sampled_ids": sampled_ids,
        "species_counts": {
            str(s): int(counts[s]) for s in range(len(ranging.labels))
        },
        "stored_counts": stored_counts,
        "index": index,
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest))


def decode_hybrid_sample(outdir: Path, rng: np.random.Generator) -> np.ndarray:
    meta = json.loads((outdir / "manifest.json").read_text())
    ranging = common.load_ranging_for_artifact(outdir)
    spectrum = spectrum_store.SpectrumStore(outdir / "spectrum.zst", ranging)
    bounds = np.asarray(meta["bounds"], dtype=np.float32)
    payload = common._dctx.decompress((outdir / "points.zst").read_bytes())
    parts = {}
    off = 0
    for key, length in meta["index"]:
        parts[key] = payload[off:off + length]
        off += length

    exact_ids = set(meta["exact_ids"])
    out = []
    for s_str, n_value in meta["species_counts"].items():
        s, n = int(s_str), int(n_value)
        stored_n = int(meta["stored_counts"].get(s_str, 0))
        if n == 0 or stored_n == 0:
            continue
        centers = unpack_species_points(
            parts[f"pos_{s}"], stored_n, bounds, meta["bits"], rng)
        center_mass = np.frombuffer(
            parts[f"mass_{s}"], dtype=np.uint16).astype(np.float32)
        center_mass = (center_mass + 0.5) * float(meta["mass_bin_da"])
        if s in exact_ids or stored_n == n:
            xyz, mass = centers, center_mass
        else:
            xyz, mass = expand_sample_centers_and_values(
                centers, center_mass, n, float(meta["kernel_sigma_nm"]),
                bounds, rng)
            if meta.get("preserve_spectrum", False):
                # Match the stored fine spectrum without severing the
                # mass/position relationship: quantile-map target masses onto
                # the sampled masses.  A plain shuffle would recreate the old
                # hybrid bug by making color independent of location.
                target = spectrum.sample_for_species(s, n, rng)
                source_order = np.argsort(mass, kind="stable")
                target.sort()
                mapped = np.empty_like(target)
                mapped[source_order] = target
                mass = mapped
        out.append(np.column_stack([xyz, mass.astype(np.float32)]))
    return np.concatenate(out).astype(np.float32)


# --- Morton-packed full point cloud -----------------------------------------


MASS_TAIL_SCALE = 82.0  # uint16 over 0..799 Da -> 0.0122 Da steps


def _norm_bits(bits) -> tuple[int, int, int]:
    """Accept a scalar bit width or per-axis (bx, by, bz)."""
    if isinstance(bits, (list, tuple, np.ndarray)):
        return (int(bits[0]), int(bits[1]), int(bits[2]))
    return (int(bits), int(bits), int(bits))


def axis_bits(bounds: np.ndarray, bits: int, min_bits: int = 6) -> list[int]:
    """Per-axis bit widths that equalize the physical quantization step.

    The longest axis gets `bits`; shorter axes drop bits so every axis has
    roughly the same nm step instead of wasting precision on narrow axes.
    """
    extent = np.maximum(np.asarray(bounds[1], dtype=np.float64)
                        - np.asarray(bounds[0], dtype=np.float64), 1e-9)
    step = extent.max() / (2 ** bits)
    out = []
    for e in extent:
        b = int(np.ceil(np.log2(max(e / step, 1.0))))
        out.append(int(np.clip(b, min_bits, bits)))
    return out


def _quantize_xyz(xyz: np.ndarray, bounds: np.ndarray, bits) -> np.ndarray:
    bx = np.array(_norm_bits(bits), dtype=np.float64)
    scale = (2.0 ** bx - 1.0) / (bounds[1] - bounds[0])
    return np.round((xyz - bounds[0]) * scale).astype(np.uint64)


def _species_windows(ranging, s: int) -> list[tuple[float, float]]:
    return [(float(ranging.lo[w]), float(ranging.hi[w]))
            for w in range(len(ranging.lo)) if ranging.species_of_window[w] == s]


def _stack_mass(mass: np.ndarray, wins) -> np.ndarray:
    """Map masses in disjoint windows onto one contiguous [0, L) domain."""
    out = np.zeros_like(mass)
    offset = 0.0
    assigned = np.zeros(len(mass), dtype=bool)
    for lo, hi in wins:
        sel = (~assigned) & (mass >= lo) & (mass < hi)
        out[sel] = mass[sel] - lo + offset
        assigned |= sel
        offset += hi - lo
    return out


def _unstack_mass(stacked: np.ndarray, wins) -> np.ndarray:
    out = np.zeros_like(stacked)
    offset = 0.0
    done = np.zeros(len(stacked), dtype=bool)
    for lo, hi in wins:
        width = hi - lo
        sel = (~done) & (stacked < offset + width)
        out[sel] = stacked[sel] - offset + lo
        done |= sel
        offset += width
    out[~done] = wins[-1][1] - 1e-4
    return out


def display_mass_max_for(ranging) -> float:
    """Upper edge of the viewer's six-segment mass palette for this ranging."""
    return min(
        common.SPECTRUM_MAX_DA,
        float(np.ceil(ranging.hi.max() + 12.0)) if len(ranging.hi)
        else common.SPECTRUM_MAX_DA,
    )


def encode_qpoint(pts: np.ndarray, ranging, outdir: Path, bits=12,
                  store_mass="u8", aniso=False, mass_bands=6, seed=0):
    """Group atoms by species; Morton-sort each group; delta+varint encode.

    store_mass: "u8" stores a per-atom within-window mass residual byte
    (unranged atoms get a uint16 full-range mass); "synth" stores no per-atom
    mass and synthesizes from the global spectrum on decode; "band" stores a
    per-atom viewer palette band index for unranged atoms only (ranged atoms
    synthesize within their narrow species windows), which keeps mass/color
    attached to position at band granularity for ~1/6 the cost of "u8".

    aniso=True drops bits on short axes so every axis has about the same
    physical quantization step (bits applies to the longest axis).
    """
    bounds = np.stack([pts[:, :3].min(axis=0), pts[:, :3].max(axis=0)])
    species = ranging.assign(pts[:, 3])
    spectrum_store.encode(pts[:, 3], outdir / "spectrum.zst")
    bits_xyz = axis_bits(bounds, bits) if aniso else [bits, bits, bits]
    display_max = display_mass_max_for(ranging)

    blobs: dict[str, bytes] = {}
    meta = {"bits": bits, "bits_xyz": bits_xyz, "store_mass": store_mass,
            "mass_bands": mass_bands, "display_mass_max": display_max,
            "bounds": bounds.tolist(), "counts": {}}
    for s in range(len(ranging.labels)):
        sel = species == s
        n = int(sel.sum())
        meta["counts"][str(s)] = n
        if n == 0:
            continue
        sub = pts[sel]
        q = _quantize_xyz(sub[:, :3], bounds, bits_xyz)
        code = bitpack.morton3_aniso(q[:, 0], q[:, 1], q[:, 2], bits_xyz)
        order = np.argsort(code, kind="stable")
        code = code[order]
        sub = sub[order]
        delta = np.diff(code, prepend=code[:1])
        delta[0] = code[0]
        blobs[f"pos_{s}"] = bitpack.varint_encode(delta)
        if store_mass == "u8":
            if s == 0:
                mq = np.round(sub[:, 3] * MASS_TAIL_SCALE).clip(0, 65535).astype(np.uint16)
                blobs[f"mass_{s}"] = mq.tobytes()
            else:
                wins = _species_windows(ranging, s)
                span = sum(hi - lo for lo, hi in wins)
                stacked = _stack_mass(sub[:, 3].astype(np.float64), wins)
                mq = np.round(stacked / max(span, 1e-9) * 255.0).clip(0, 255)
                blobs[f"mass_{s}"] = mq.astype(np.uint8).tobytes()
        elif store_mass == "band" and s == 0:
            band = np.minimum(
                np.floor(np.maximum(sub[:, 3], 0.0) / display_max
                         * mass_bands).astype(np.uint8),
                mass_bands - 1,
            )
            blobs[f"band_{s}"] = band.tobytes()

    payload = b""
    index = []
    for k, b in blobs.items():
        index.append([k, len(b)])
        payload += b
    (outdir / "blob.zst").write_bytes(common.zbytes(payload))
    meta["index"] = index
    (outdir / "manifest.json").write_text(json.dumps(meta))


def decode_qpoint(outdir: Path, rng: np.random.Generator) -> np.ndarray:
    meta = json.loads((outdir / "manifest.json").read_text())
    ranging = common.load_ranging_for_artifact(outdir)
    store = spectrum_store.SpectrumStore(outdir / "spectrum.zst", ranging)
    bounds = np.array(meta["bounds"], dtype=np.float32)
    bits_xyz = _norm_bits(meta.get("bits_xyz", meta["bits"]))
    payload = common._dctx.decompress((outdir / "blob.zst").read_bytes())
    parts = {}
    off = 0
    for k, ln in meta["index"]:
        parts[k] = payload[off : off + ln]
        off += ln

    out = []
    for s_str, n in meta["counts"].items():
        s = int(s_str)
        if n == 0:
            continue
        delta = bitpack.varint_decode(parts[f"pos_{s}"], n)
        code = np.cumsum(delta.astype(np.uint64))
        qx, qy, qz = bitpack.demorton3_aniso(code, bits_xyz)
        scale = (bounds[1] - bounds[0]) / (
            2.0 ** np.array(bits_xyz, dtype=np.float32) - 1.0)
        jitter = rng.random((n, 3), dtype=np.float32) - 0.5
        xyz = bounds[0] + (np.stack([qx, qy, qz], axis=1).astype(np.float32) + jitter) * scale
        if meta["store_mass"] == "u8":
            if s == 0:
                mq = np.frombuffer(parts[f"mass_{s}"], dtype=np.uint16)
                mass = mq.astype(np.float32) / MASS_TAIL_SCALE
            else:
                wins = _species_windows(ranging, s)
                span = sum(hi - lo for lo, hi in wins)
                mq = np.frombuffer(parts[f"mass_{s}"], dtype=np.uint8)
                stacked = np.clip(
                    (mq.astype(np.float64) + rng.random(n) - 0.5) / 255.0 * span,
                    0.0, span - 1e-6,
                )
                mass = _unstack_mass(stacked, wins).astype(np.float32)
        elif meta["store_mass"] == "band" and s == 0 and f"band_{s}" in parts:
            band = np.frombuffer(parts[f"band_{s}"], dtype=np.uint8)
            n_bands = int(meta["mass_bands"])
            display_max = float(meta["display_mass_max"])
            mass = np.empty(n, dtype=np.float32)
            for b in range(n_bands):
                idx = np.flatnonzero(band == b)
                if len(idx):
                    mass[idx] = store.sample_for_species_band(
                        s, b, n_bands, display_max, len(idx), rng)
        else:
            mass = store.sample_for_species(s, n, rng)
        out.append(np.column_stack([xyz, mass.astype(np.float32)]))
    return np.concatenate(out).astype(np.float32)


# --- helpers reused by hybrid codecs ----------------------------------------


def pack_species_points(xyz: np.ndarray, bounds: np.ndarray, bits: int) -> bytes:
    blob, _ = pack_species_points_with_order(xyz, bounds, bits)
    return blob


def pack_species_points_with_order(
    xyz: np.ndarray, bounds: np.ndarray, bits
) -> tuple[bytes, np.ndarray]:
    """Pack positions and return the Morton order for aligned attributes."""
    bits_xyz = _norm_bits(bits)
    q = _quantize_xyz(xyz, bounds, bits_xyz)
    code = bitpack.morton3_aniso(q[:, 0], q[:, 1], q[:, 2], bits_xyz)
    order = np.argsort(code, kind="stable")
    code = code[order]
    delta = np.diff(code, prepend=code[:1])
    delta[0] = code[0]
    return bitpack.varint_encode(delta), order


def unpack_species_points(
    blob: bytes, n: int, bounds: np.ndarray, bits, rng: np.random.Generator
) -> np.ndarray:
    bits_xyz = _norm_bits(bits)
    delta = bitpack.varint_decode(blob, n)
    code = np.cumsum(delta.astype(np.uint64))
    qx, qy, qz = bitpack.demorton3_aniso(code, bits_xyz)
    scale = (bounds[1] - bounds[0]) / (
        2.0 ** np.array(bits_xyz, dtype=np.float32) - 1.0)
    jitter = rng.random((n, 3), dtype=np.float32) - 0.5
    return bounds[0] + (np.stack([qx, qy, qz], axis=1).astype(np.float32) + jitter) * scale


def expand_sample_centers(
    centers: np.ndarray,
    n: int,
    kernel_sigma_nm: float,
    bounds: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Expand equally weighted sample centers without adding count noise."""
    stored_n = len(centers)
    if stored_n == 0 or n <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    if stored_n >= n:
        chosen = rng.choice(stored_n, n, replace=False)
        return centers[chosen].copy()
    repeats, remainder = divmod(n, stored_n)
    xyz = np.repeat(centers, repeats, axis=0)
    if remainder:
        chosen = rng.choice(stored_n, remainder, replace=False)
        xyz = np.concatenate([xyz, centers[chosen]], axis=0)
    if kernel_sigma_nm > 0:
        noise = rng.standard_normal(xyz.shape, dtype=np.float32)
        noise *= kernel_sigma_nm
        xyz += noise
    np.clip(xyz, bounds[0], bounds[1], out=xyz)
    return xyz


def expand_sample_centers_and_values(
    centers: np.ndarray,
    values: np.ndarray,
    n: int,
    kernel_sigma_nm: float,
    bounds: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Expand point centers and keep a per-center attribute aligned."""
    stored_n = len(centers)
    if stored_n == 0 or n <= 0:
        return (np.zeros((0, 3), dtype=np.float32),
                np.zeros(0, dtype=values.dtype))
    if stored_n >= n:
        chosen = rng.choice(stored_n, n, replace=False)
        xyz, expanded = centers[chosen].copy(), values[chosen].copy()
    else:
        repeats, remainder = divmod(n, stored_n)
        xyz = np.repeat(centers, repeats, axis=0)
        expanded = np.repeat(values, repeats, axis=0)
        if remainder:
            chosen = rng.choice(stored_n, remainder, replace=False)
            xyz = np.concatenate([xyz, centers[chosen]], axis=0)
            expanded = np.concatenate([expanded, values[chosen]], axis=0)
    if kernel_sigma_nm > 0:
        noise = rng.standard_normal(xyz.shape, dtype=np.float32)
        noise *= kernel_sigma_nm
        xyz += noise
    np.clip(xyz, bounds[0], bounds[1], out=xyz)
    return xyz, expanded
