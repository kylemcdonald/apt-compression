"""Grid-family codecs: a fine "density backbone" for where atoms are, plus a
compact "composition model" for how density splits into species, plus the
shared global spectrum store for mass synthesis.

backbone:  "u8" quantized counts | "wavelet" | "inr" | "hyper" (neural)
comp:      "global" | "kmeans" | "kmeans_resid" | "nmf" | "pca" | "direct"
exact:     optionally store rare species as exact bit-packed points and only
           model the abundant remainder (Al+ and unranged background here).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import common
import codecs_points
import spectrum_store
from common import GridSpec

EPS = 1e-9


# --- composition models ------------------------------------------------------


def _cell_species_counts(pts, species, model_ids, spec: GridSpec):
    id_of = {s: i for i, s in enumerate(model_ids)}
    sel = np.isin(species, model_ids)
    cells = spec.flat(pts[sel, :3])
    sidx = np.array([id_of[s] for s in species[sel]], dtype=np.int64)
    joint = cells * len(model_ids) + sidx
    C = np.bincount(joint, minlength=spec.ncells * len(model_ids))
    return C.reshape(spec.ncells, len(model_ids)).astype(np.float64)


def fit_composition(pts, species, model_ids, spec: GridSpec, model, k, seed):
    """Return (stored_arrays, meta) for the chosen composition model."""
    C = _cell_species_counts(pts, species, model_ids, spec)
    total = C.sum(axis=1)
    active = total > 0
    F = np.zeros_like(C)
    F[active] = C[active] / total[active, None]
    global_frac = C.sum(axis=0) / max(C.sum(), 1.0)
    meta = {"model": model, "k": k, "cells": spec.ncells,
            "shape": [int(v) for v in spec.shape],
            "global_frac": global_frac.tolist()}

    if model == "global":
        return {}, meta

    if model in ("kmeans", "kmeans_resid"):
        from sklearn.cluster import MiniBatchKMeans

        train = F[active]
        std = train.std(axis=0)
        std[std < EPS] = 1.0
        km = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=4096,
                             n_init=5, max_iter=200)
        labels_active = km.fit_predict(train / std)
        labels = np.full(spec.ncells, 255, dtype=np.uint8)
        labels[active] = labels_active.astype(np.uint8)
        table = np.zeros((k, len(model_ids)), dtype=np.float64)
        for j in range(k):
            member = C[active][labels_active == j]
            tot = member.sum()
            table[j] = member.sum(axis=0) / tot if tot > 0 else global_frac
        arrays = {"labels": labels, "table": table.astype(np.float32)}
        if model == "kmeans_resid":
            # coarser per-species deviation from the cluster baseline
            rspec = GridSpec(spec.origin, spec.voxel * 2.0,
                             tuple((np.array(spec.shape) + 1) // 2))
            Cr = _cell_species_counts(pts, species, model_ids, rspec)
            tot_r = Cr.sum(axis=1)
            base = _upmap_cells(labels, spec, rspec)
            base_frac = np.where(base[:, None] < 255, table[np.clip(base, 0, k - 1)],
                                 global_frac[None, :])
            Fr = np.zeros_like(Cr)
            act = tot_r > 0
            Fr[act] = Cr[act] / tot_r[act, None]
            ratio = np.ones_like(Cr)
            ratio[act] = (Fr[act] + 1e-4) / (base_frac[act] + 1e-4)
            logr = np.clip(np.log2(ratio), -3.0, 3.0)
            q = np.round((logr + 3.0) / 6.0 * 255.0).astype(np.uint8)
            arrays["resid"] = q
            meta["resid_shape"] = [int(v) for v in rspec.shape]
        return arrays, meta

    if model == "nmf":
        from sklearn.decomposition import NMF

        nmf = NMF(n_components=k, init="nndsvda", random_state=seed, max_iter=400)
        W = nmf.fit_transform(F[active])
        H = nmf.components_
        Wfull = np.zeros((spec.ncells, k), dtype=np.float64)
        Wfull[active] = W
        wmax = Wfull.max(axis=0)
        wmax[wmax < EPS] = 1.0
        Wq = np.round(Wfull / wmax * 255.0).astype(np.uint8)
        return ({"W": Wq, "wmax": wmax.astype(np.float32), "H": H.astype(np.float32)},
                meta)

    if model in ("pca", "ica"):
        if model == "pca":
            from sklearn.decomposition import PCA

            dec = PCA(n_components=k, random_state=seed)
            coef = dec.fit_transform(F[active])
            mean, comps = dec.mean_, dec.components_
        else:
            from sklearn.decomposition import FastICA

            dec = FastICA(n_components=k, random_state=seed, max_iter=500)
            coef = dec.fit_transform(F[active])
            mean, comps = dec.mean_, dec.mixing_.T
        lo, hi = coef.min(axis=0), coef.max(axis=0)
        span = np.maximum(hi - lo, EPS)
        coef_full = np.zeros((spec.ncells, k), dtype=np.float64)
        coef_full[active] = coef
        q = np.round((coef_full - lo) / span * 255.0).clip(0, 255).astype(np.uint8)
        return ({"coef": q, "lo": lo.astype(np.float32), "hi": hi.astype(np.float32),
                 "mean": mean.astype(np.float32),
                 "components": comps.astype(np.float32),
                 "active": active.astype(np.uint8)}, meta)

    if model == "direct":
        q = np.round(np.sqrt(F) * 255.0).astype(np.uint8)
        return {"F": q}, meta

    raise ValueError(model)


def eval_composition(arrays, meta) -> np.ndarray:
    """Return (ncells, S_model) fraction matrix from stored arrays."""
    ncells = meta["cells"]
    S = len(meta["global_frac"])
    gf = np.array(meta["global_frac"])
    model = meta["model"]
    if model == "global":
        return np.tile(gf, (ncells, 1))
    if model in ("kmeans", "kmeans_resid"):
        labels = arrays["labels"]
        table = arrays["table"].astype(np.float64)
        F = np.where(labels[:, None] < 255,
                     table[np.clip(labels.astype(int), 0, len(table) - 1)],
                     gf[None, :])
        return F
    if model == "nmf":
        W = arrays["W"].astype(np.float64) / 255.0 * arrays["wmax"][None, :]
        F = W @ arrays["H"].astype(np.float64)
        tot = F.sum(axis=1, keepdims=True)
        F = np.divide(F, tot, out=np.tile(gf, (ncells, 1)), where=tot > EPS)
        return F
    if model in ("pca", "ica"):
        lo, hi = arrays["lo"].astype(np.float64), arrays["hi"].astype(np.float64)
        coef = arrays["coef"].astype(np.float64) / 255.0 * (hi - lo) + lo
        F = arrays["mean"].astype(np.float64) + coef @ arrays["components"].astype(np.float64)
        F = np.clip(F, 0.0, None)
        inactive = arrays["active"] == 0
        F[inactive] = gf
        tot = F.sum(axis=1, keepdims=True)
        F = np.divide(F, tot, out=np.tile(gf, (ncells, 1)), where=tot > EPS)
        return F
    if model == "direct":
        F = (arrays["F"].astype(np.float64) / 255.0) ** 2
        tot = F.sum(axis=1, keepdims=True)
        F = np.divide(F, tot, out=np.tile(gf, (ncells, 1)), where=tot > EPS)
        return F
    raise ValueError(model)


def _upmap_cells(labels: np.ndarray, fine: GridSpec, coarse: GridSpec) -> np.ndarray:
    """Majority-free quick map: coarse cell -> label of its central fine cell."""
    mx, my, mz = coarse.shape
    nx, ny, nz = fine.shape
    cz, cy, cx = np.meshgrid(np.arange(mz), np.arange(my), np.arange(mx), indexing="ij")
    fx = np.clip(((cx + 0.5) * coarse.voxel / fine.voxel).astype(int), 0, nx - 1)
    fy = np.clip(((cy + 0.5) * coarse.voxel / fine.voxel).astype(int), 0, ny - 1)
    fz = np.clip(((cz + 0.5) * coarse.voxel / fine.voxel).astype(int), 0, nz - 1)
    flat = (fz.ravel() * ny + fy.ravel()) * nx + fx.ravel()
    return labels[flat]


def upsample_trilinear(vol: np.ndarray, bspec: GridSpec, cspec: GridSpec) -> np.ndarray:
    """Trilinearly interpolate a coarse (mz,my,mx) cell volume onto the fine
    backbone grid (nz,ny,nx), sampling at fine-voxel centers in coarse
    cell-center coordinates. Processed in z-slabs to bound memory."""
    nx, ny, nz = bspec.shape
    mx, my, mz = cspec.shape
    r = bspec.voxel / cspec.voxel

    def axis_idx(nf, mf):
        pos = (np.arange(nf) + 0.5) * r - 0.5
        lo = np.clip(np.floor(pos).astype(np.int64), 0, mf - 1)
        hi = np.clip(lo + 1, 0, mf - 1)
        f = np.clip(pos - lo, 0.0, 1.0).astype(np.float32)
        return lo, hi, f

    ix0, ix1, fx = axis_idx(nx, mx)
    iy0, iy1, fy = axis_idx(ny, my)
    iz0, iz1, fz = axis_idx(nz, mz)
    out = np.empty((nz, ny, nx), dtype=np.float32)
    vol = vol.astype(np.float32)
    slab = max(1, int(4e6 // (ny * nx)))
    for z0 in range(0, nz, slab):
        z1 = min(z0 + slab, nz)
        a0, a1, az = iz0[z0:z1], iz1[z0:z1], fz[z0:z1][:, None, None]
        acc = np.zeros((z1 - z0, ny, nx), dtype=np.float32)
        for zi, wz in ((a0, 1 - az), (a1, az)):
            for yi, wy in ((iy0, (1 - fy)[None, :, None]), (iy1, fy[None, :, None])):
                for xi, wx in ((ix0, (1 - fx)[None, None, :]), (ix1, fx[None, None, :])):
                    acc += vol[np.ix_(zi, yi, xi)] * (wz * wy * wx)
        out[z0:z1] = acc
    return out


def _fine_to_cell_map(bspec: GridSpec, cspec: GridSpec) -> np.ndarray:
    """Flat index of the composition cell for every backbone voxel."""
    nx, ny, nz = bspec.shape
    mx, my, mz = cspec.shape
    ix = np.clip(((np.arange(nx) + 0.5) * bspec.voxel / cspec.voxel).astype(int), 0, mx - 1)
    iy = np.clip(((np.arange(ny) + 0.5) * bspec.voxel / cspec.voxel).astype(int), 0, my - 1)
    iz = np.clip(((np.arange(nz) + 0.5) * bspec.voxel / cspec.voxel).astype(int), 0, mz - 1)
    zz, yy, xx = np.meshgrid(iz, iy, ix, indexing="ij")
    return ((zz.astype(np.int64) * my + yy) * mx + xx).ravel()


# --- backbone models ---------------------------------------------------------


def encode_backbone(grid: np.ndarray, outdir: Path, mode: str, **kw) -> dict:
    if mode == "u8":
        m = grid.max()
        if m <= 255:
            q = grid.astype(np.uint8)
        else:
            q = grid.astype(np.uint16)
        common.zsave(outdir / "backbone.zst", counts=q)
        return {}
    if mode == "wavelet":
        import codecs_wavelet

        return codecs_wavelet.encode_grid(grid, outdir / "backbone.zst",
                                          budget_coeffs=kw.get("wavelet_coeffs", 400_000))
    if mode in ("inr", "hyper"):
        import codecs_neural

        fn = codecs_neural.encode_inr if mode == "inr" else codecs_neural.encode_hyper
        return fn(grid, outdir / "backbone.zst", **kw.get("neural_kw", {}))
    raise ValueError(mode)


def decode_backbone(outdir: Path, mode: str, shape) -> np.ndarray:
    if mode == "u8":
        return common.zload(outdir / "backbone.zst")["counts"].astype(np.float32)
    if mode == "wavelet":
        import codecs_wavelet

        return codecs_wavelet.decode_grid(outdir / "backbone.zst")
    if mode in ("inr", "hyper"):
        import codecs_neural

        fn = codecs_neural.decode_inr if mode == "inr" else codecs_neural.decode_hyper
        return fn(outdir / "backbone.zst", tuple(shape))
    raise ValueError(mode)


def _nearest_guide_mass_bins(
    xyz: np.ndarray,
    centers: np.ndarray,
    center_mass_bins: np.ndarray,
    rng: np.random.Generator,
    neighbors: int,
) -> np.ndarray:
    """Transfer local guide masses to decoded grid points.

    A small stochastic nearest-neighbor blend avoids hard Voronoi boundaries;
    the subsequent quantile map restores the global fine spectrum while
    preserving this local mass ordering.
    """
    from scipy.spatial import cKDTree

    if len(centers) == 0:
        return np.zeros(len(xyz), dtype=np.uint16)
    k = max(1, min(int(neighbors), len(centers)))
    tree = cKDTree(centers)
    out = np.empty(len(xyz), dtype=np.uint16)
    chunk = 500_000
    for start in range(0, len(xyz), chunk):
        stop = min(start + chunk, len(xyz))
        dist, idx = tree.query(xyz[start:stop], k=k, workers=-1)
        if k == 1:
            out[start:stop] = center_mass_bins[idx]
            continue
        # Nearby samples vote with inverse-distance weights. Random selection
        # keeps mixtures as point-supported distributions rather than averaging
        # two masses into a physically meaningless intermediate mass.
        weights = 1.0 / np.maximum(dist.astype(np.float32), 0.05) ** 2
        cdf = np.cumsum(weights, axis=1)
        draw = rng.random(stop - start, dtype=np.float32) * cdf[:, -1]
        pick = (cdf < draw[:, None]).sum(axis=1)
        out[start:stop] = center_mass_bins[idx[np.arange(stop - start), pick]]
    return out


def _quantile_map_guide_mass(
    source_bins: np.ndarray,
    store: spectrum_store.SpectrumStore,
    species_id: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Match the fine spectrum while retaining guide-derived mass rank."""
    if len(source_bins) == 0:
        return np.zeros(0, dtype=np.float32)
    source_hist = np.bincount(source_bins.astype(np.int64)).astype(np.float64)
    source_lo = np.cumsum(source_hist) - source_hist

    target = store.species_hist(species_id).astype(np.float64)
    if species_id == 0 and len(store.tail):
        tail_bins = np.floor(store.tail / spectrum_store.BIN).astype(np.int64)
        n_target = max(len(target), int(tail_bins.max()) + 1)
        expanded = np.zeros(n_target, dtype=np.float64)
        expanded[:len(target)] = target
        expanded += np.bincount(tail_bins, minlength=n_target)
        target = expanded
    target_cdf = np.cumsum(target)
    target_total = float(target_cdf[-1]) if len(target_cdf) else 0.0
    if target_total <= 0:
        return np.full(len(source_bins), -1.0, dtype=np.float32)

    out = np.empty(len(source_bins), dtype=np.float32)
    source_total = float(len(source_bins))
    chunk = 1_000_000
    for start in range(0, len(source_bins), chunk):
        stop = min(start + chunk, len(source_bins))
        b = source_bins[start:stop].astype(np.int64)
        rank = source_lo[b] + rng.random(stop - start) * source_hist[b]
        target_rank = rank / source_total * target_total
        mapped = np.searchsorted(target_cdf, target_rank, side="right")
        out[start:stop] = (
            mapped + rng.random(stop - start)) * spectrum_store.BIN
    return out


def _sample_mass_for_band(
    store: spectrum_store.SpectrumStore,
    species_id: int,
    band: int,
    band_count: int,
    display_mass_max: float,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample the stored spectrum restricted to one viewer-palette band."""
    return store.sample_for_species_band(
        species_id, band, band_count, display_mass_max, n, rng)


# --- main grid codec ---------------------------------------------------------


def encode_grid(
    pts: np.ndarray,
    ranging,
    outdir: Path,
    backbone="u8",
    backbone_voxel=1.0,
    comp_model="kmeans",
    comp_cell=3.0,
    k=8,
    exact_species="none",
    rare_threshold=100_000,
    guide_fraction=0.0,
    guide_cap=None,
    guide_mix=0.0,
    guide_scale=0.0,
    guide_sigma_nm=0.10,
    guide_condition_mass=False,
    guide_mass_neighbors=4,
    guide_min_per_mass_band=0,
    unranged_mass_bands=0,
    bits=12,
    seed=0,
    **backbone_kw,
):
    bounds = np.stack([pts[:, :3].min(axis=0), pts[:, :3].max(axis=0)])
    species = ranging.assign(pts[:, 3])
    spectrum_store.encode(pts[:, 3], outdir / "spectrum.zst")
    counts = np.bincount(species, minlength=len(ranging.labels))

    exact_ids = []
    if exact_species == "rare":
        exact_ids = [s for s in range(1, len(ranging.labels))
                     if 0 < counts[s] < rare_threshold]
    model_ids = [s for s in range(len(ranging.labels))
                 if counts[s] > 0 and s not in exact_ids]
    category_display_mass_max = None
    mass_band = None
    exact_unranged_bands = []
    if unranged_mass_bands:
        category_display_mass_max = min(
            spectrum_store.MAX,
            float(np.ceil(ranging.hi.max() + 12.0)) if len(ranging.hi)
            else spectrum_store.MAX,
        )
        mass_band = np.minimum(
            np.floor(np.maximum(pts[:, 3], 0.0)
                     / category_display_mass_max * unranged_mass_bands
                     ).astype(np.int16),
            unranged_mass_bands - 1,
        )
        unranged_band_counts = np.bincount(
            mass_band[species == 0], minlength=unranged_mass_bands)
        exact_unranged_bands = [
            int(band) for band, n in enumerate(unranged_band_counts)
            if 0 < n < rare_threshold
        ]
    if guide_cap is None and not 0.0 <= guide_fraction <= 1.0:
        raise ValueError("guide_fraction must be in [0, 1]")
    if guide_cap is not None and guide_cap <= 0:
        raise ValueError("guide_cap must be positive")
    if not 0.0 <= guide_mix <= 1.0:
        raise ValueError("guide_mix must be in [0, 1]")
    if guide_scale < 0:
        raise ValueError("guide_scale must be nonnegative")
    has_guide = (guide_cap is not None or guide_fraction > 0
                 or guide_min_per_mass_band > 0)
    if (guide_mix > 0 or guide_scale > 0) and not has_guide:
        raise ValueError("guide mixing requires guide_fraction or guide_cap")

    # exact rare species: Morton-packed quantized points
    exact_blobs = {}
    for s in exact_ids:
        xyz = pts[species == s, :3]
        exact_blobs[f"exact_{s}"] = codecs_points.pack_species_points(xyz, bounds, bits)
    if exact_blobs:
        payload = b"".join(exact_blobs.values())
        (outdir / "exact.zst").write_bytes(common.zbytes(payload))
        exact_index = [[k_, len(v)] for k_, v in exact_blobs.items()]
    else:
        exact_index = []

    # Rare mass/color bands hidden inside the otherwise huge unranged bucket
    # are exact mass-position pairs. A species-only threshold misses these.
    exact_band_index = []
    if exact_unranged_bands:
        band_blobs = []
        for band in exact_unranged_bands:
            sub = pts[(species == 0) & (mass_band == band)]
            pos_blob, order = codecs_points.pack_species_points_with_order(
                sub[:, :3], bounds, bits)
            mass_bin = np.floor(sub[order, 3] / spectrum_store.BIN)
            mass_blob = mass_bin.clip(0, 65535).astype(np.uint16).tobytes()
            band_blobs.extend([pos_blob, mass_blob])
            exact_band_index.append({
                "band": int(band),
                "count": int(len(sub)),
                "pos_length": len(pos_blob),
                "mass_length": len(mass_blob),
            })
        (outdir / "exact_bands.zst").write_bytes(
            common.zbytes(b"".join(band_blobs)))

    # Optional point guide for modeled species. The grid still carries their
    # smooth bulk density; this compact sample restores sub-grid inclusions and
    # interfaces that would otherwise be averaged out by composition cells.
    # When guide_condition_mass is enabled, original mass bins are kept in
    # Morton order too. Decode transfers their local mass ordering onto the
    # smooth grid points instead of randomizing mass independently of space.
    guide_index = []
    guide_stored_counts = {}
    if has_guide:
        guide_rng = np.random.default_rng(seed)
        guide_blobs = {}
        display_mass_max = min(
            spectrum_store.MAX,
            float(np.ceil(ranging.hi.max() + 12.0)) if len(ranging.hi)
            else spectrum_store.MAX,
        )
        for s in model_ids:
            sub = pts[species == s]
            if guide_min_per_mass_band > 0:
                mass_band = np.minimum(
                    np.floor(np.maximum(sub[:, 3], 0.0)
                             / display_mass_max * 6).astype(np.int8), 5)
                chosen_parts = []
                for band in range(6):
                    candidates = np.flatnonzero(mass_band == band)
                    if len(candidates) == 0:
                        continue
                    take = min(len(candidates), max(
                        int(guide_min_per_mass_band),
                        int(round(len(candidates) * guide_fraction)),
                    ))
                    if take < len(candidates):
                        candidates = guide_rng.choice(
                            candidates, take, replace=False)
                    chosen_parts.append(candidates)
                chosen = np.concatenate(chosen_parts)
                sub = sub[chosen]
                stored_n = len(sub)
            else:
                stored_n = (min(len(sub), int(guide_cap)) if guide_cap is not None
                            else max(1, int(round(len(sub) * guide_fraction))))
                if stored_n < len(sub):
                    chosen = guide_rng.choice(len(sub), stored_n, replace=False)
                    sub = sub[chosen]
            guide_stored_counts[str(s)] = stored_n
            if guide_condition_mass:
                pos_blob, order = codecs_points.pack_species_points_with_order(
                    sub[:, :3], bounds, bits)
                guide_blobs[f"guide_{s}"] = pos_blob
                mass_bin = np.floor(sub[order, 3] / spectrum_store.BIN)
                guide_blobs[f"guide_mass_{s}"] = (
                    mass_bin.clip(0, 65535).astype(np.uint16).tobytes())
            else:
                guide_blobs[f"guide_{s}"] = codecs_points.pack_species_points(
                    sub[:, :3], bounds, bits)
        payload = b"".join(guide_blobs.values())
        (outdir / "guide.zst").write_bytes(common.zbytes(payload))
        guide_index = [[key, len(blob)] for key, blob in guide_blobs.items()]

    # Optionally split the broad unranged population into the same mass/color
    # bands used by the viewer.  Modeling species 0 as one bucket and then
    # synthesizing its masses independently is what erased localized red
    # structures in the original hybrids.
    comp_species = species
    comp_ids = model_ids
    model_categories = []
    category_counts = []
    if unranged_mass_bands:
        comp_species = np.full(len(species), -1, dtype=np.int16)
        for s in model_ids:
            bands = range(unranged_mass_bands) if s == 0 else (None,)
            for band in bands:
                if s == 0 and band in exact_unranged_bands:
                    continue
                sel = species == s
                if band is not None:
                    sel &= mass_band == band
                n_category = int(sel.sum())
                if n_category == 0:
                    continue
                category_id = len(model_categories)
                comp_species[sel] = category_id
                model_categories.append({
                    "species_id": int(s),
                    "mass_band": band,
                })
                category_counts.append(n_category)
        comp_ids = list(range(len(model_categories)))

    # backbone over modeled atoms
    modeled = np.isin(species, model_ids)
    if exact_unranged_bands:
        modeled &= ~((species == 0) & np.isin(mass_band, exact_unranged_bands))
    bspec = GridSpec.for_bounds(bounds, backbone_voxel)
    grid = common.density_grid(pts[modeled, :3], bspec)
    encode_backbone(grid, outdir, backbone, **backbone_kw)

    # composition over modeled atoms
    cspec = GridSpec.for_bounds(bounds, comp_cell)
    arrays, comp_meta = fit_composition(
        pts, comp_species, comp_ids, cspec, comp_model, k, seed)
    if arrays:
        common.zsave(outdir / "comp.zst", **arrays)

    manifest = {
        "backbone": backbone, "backbone_voxel": backbone_voxel,
        "backbone_shape": [int(v) for v in bspec.shape],
        "comp_model": comp_model, "comp_cell": comp_cell, "k": k,
        "exact_species": exact_species, "rare_threshold": rare_threshold,
        "guide_fraction": guide_fraction, "guide_cap": guide_cap,
        "guide_mix": guide_mix, "guide_scale": guide_scale,
        "guide_sigma_nm": guide_sigma_nm,
        "guide_condition_mass": guide_condition_mass,
        "guide_mass_neighbors": guide_mass_neighbors,
        "guide_min_per_mass_band": guide_min_per_mass_band,
        "unranged_mass_bands": unranged_mass_bands,
        "category_display_mass_max": category_display_mass_max,
        "model_categories": model_categories,
        "category_counts": category_counts,
        "guide_index": guide_index,
        "guide_stored_counts": guide_stored_counts,
        "bits": bits, "bounds": bounds.tolist(),
        "model_ids": model_ids, "exact_ids": exact_ids,
        "exact_index": exact_index,
        "exact_unranged_bands": exact_unranged_bands,
        "exact_band_index": exact_band_index,
        "species_counts": {str(s): int(counts[s]) for s in range(len(counts))},
        "comp_meta": comp_meta,
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest))


def decode_grid(outdir: Path, rng: np.random.Generator) -> np.ndarray:
    meta = json.loads((outdir / "manifest.json").read_text())
    ranging = common.load_ranging_for_artifact(outdir)
    store = spectrum_store.SpectrumStore(outdir / "spectrum.zst", ranging)
    bounds = np.array(meta["bounds"], dtype=np.float32)
    bspec = GridSpec(bounds[0].copy(), meta["backbone_voxel"],
                     tuple(meta["backbone_shape"]))
    grid = decode_backbone(outdir, meta["backbone"], meta["backbone_shape"])
    grid = np.clip(grid, 0.0, None)

    cspec = GridSpec(bounds[0].copy(), meta["comp_cell"],
                     tuple(meta["comp_meta"]["shape"]))
    arrays = common.zload(outdir / "comp.zst") if (outdir / "comp.zst").exists() else {}
    F = eval_composition(arrays, meta["comp_meta"])  # (cells, S_model)
    mx, my, mz = cspec.shape

    resid = None
    if meta["comp_meta"]["model"] == "kmeans_resid":
        rspec = GridSpec(bounds[0].copy(), meta["comp_cell"] * 2.0,
                         tuple(meta["comp_meta"]["resid_shape"]))
        resid_q = arrays["resid"].astype(np.float64)
        resid = 2.0 ** (resid_q / 255.0 * 6.0 - 3.0)  # (rcells, S_model)

    guide_parts = {}
    if meta.get("guide_index"):
        guide_payload = common._dctx.decompress((outdir / "guide.zst").read_bytes())
        guide_off = 0
        for key, length in meta["guide_index"]:
            guide_parts[key] = guide_payload[guide_off:guide_off + length]
            guide_off += length

    out = []
    grid3 = grid.reshape(bspec.shape[2], bspec.shape[1], bspec.shape[0])
    model_ids = meta["model_ids"]
    categories = meta.get("model_categories", [])
    units = categories if categories else [
        {"species_id": int(s), "mass_band": None} for s in model_ids]
    for j, category in enumerate(units):
        s = int(category["species_id"])
        band = category.get("mass_band")
        n = (int(meta["category_counts"][j]) if categories
             else int(meta["species_counts"][str(s)]))
        if n == 0:
            continue
        guide_n = 0
        stored_n = int(meta.get("guide_stored_counts", {}).get(str(s), 0))
        if f"guide_{s}" in guide_parts:
            guide_fraction = stored_n / max(n, 1)
            mix = float(meta.get("guide_mix", 0.0))
            if float(meta.get("guide_scale", 0.0)) > 0:
                mix = min(1.0, float(meta["guide_scale"]) * guide_fraction)
            guide_n = int(round(n * mix))
        grid_n = n - guide_n
        # composition fields interpolated trilinearly onto the backbone grid
        f_fine = upsample_trilinear(F[:, j].reshape(mz, my, mx), bspec, cspec)
        w = grid3 * f_fine
        del f_fine
        if resid is not None:
            rx, ry, rz = rspec.shape
            w = w * upsample_trilinear(resid[:, j].reshape(rz, ry, rx), bspec, rspec)
        xyz_parts = []
        if grid_n:
            xyz_parts.append(common.sample_points_from_grid(w, bspec, grid_n, rng))
        del w
        if guide_n:
            centers = codecs_points.unpack_species_points(
                guide_parts[f"guide_{s}"], stored_n, bounds, meta["bits"], rng)
            xyz_parts.append(codecs_points.expand_sample_centers(
                centers, guide_n, float(meta.get("guide_sigma_nm", 0.10)),
                bounds, rng))
        xyz = xyz_parts[0] if len(xyz_parts) == 1 else np.concatenate(xyz_parts)
        if (meta.get("guide_condition_mass", False)
                and f"guide_mass_{s}" in guide_parts):
            centers = codecs_points.unpack_species_points(
                guide_parts[f"guide_{s}"], stored_n, bounds, meta["bits"], rng)
            center_mass_bins = np.frombuffer(
                guide_parts[f"guide_mass_{s}"], dtype=np.uint16)
            source_bins = _nearest_guide_mass_bins(
                xyz, centers, center_mass_bins, rng,
                int(meta.get("guide_mass_neighbors", 4)))
            mass = _quantile_map_guide_mass(source_bins, store, s, rng)
        elif band is not None:
            mass = _sample_mass_for_band(
                store, s, int(band), int(meta["unranged_mass_bands"]),
                float(meta["category_display_mass_max"]), n, rng)
        else:
            mass = store.sample_for_species(s, n, rng)
        out.append(np.column_stack([xyz, mass]))

    if meta["exact_index"]:
        payload = common._dctx.decompress((outdir / "exact.zst").read_bytes())
        off = 0
        for key, ln in meta["exact_index"]:
            s = int(key.split("_")[1])
            blob = payload[off : off + ln]
            off += ln
            n = meta["species_counts"][str(s)]
            xyz = codecs_points.unpack_species_points(blob, n, bounds, meta["bits"], rng)
            mass = store.sample_for_species(s, n, rng)
            out.append(np.column_stack([xyz, mass]))
    if meta.get("exact_band_index"):
        payload = common._dctx.decompress(
            (outdir / "exact_bands.zst").read_bytes())
        off = 0
        for info in meta["exact_band_index"]:
            pos_length = int(info["pos_length"])
            mass_length = int(info["mass_length"])
            pos_blob = payload[off:off + pos_length]
            off += pos_length
            mass_blob = payload[off:off + mass_length]
            off += mass_length
            n = int(info["count"])
            xyz = codecs_points.unpack_species_points(
                pos_blob, n, bounds, meta["bits"], rng)
            mass = (np.frombuffer(mass_blob, dtype=np.uint16).astype(np.float32)
                    + 0.5) * spectrum_store.BIN
            out.append(np.column_stack([xyz, mass]))
    return np.concatenate(out).astype(np.float32)


# --- mass-range grid codec (stand-in for current cdf_v2 approach) -----------


def encode_massrange(pts: np.ndarray, ranging, outdir: Path, n_ranges=64,
                     budget_mb=8.0, seed=0):
    """64 adaptive (equal-count) mass ranges, each with its own density grid
    sized by an atom-count power law — the spirit of the current production
    codec, with spectrum-faithful mass synthesis."""
    bounds = np.stack([pts[:, :3].min(axis=0), pts[:, :3].max(axis=0)])
    mass = pts[:, 3]
    spectrum_store.encode(mass, outdir / "spectrum.zst")
    qs = np.linspace(0, 1, n_ranges + 1)
    edges = np.quantile(mass, qs)
    edges[0], edges[-1] = 0.0, float(mass.max()) + 1e-3
    edges = np.unique(edges)
    volume = float(np.prod(bounds[1] - bounds[0]))

    counts, grids = [], {}
    weights = []
    for r in range(len(edges) - 1):
        sel = (mass >= edges[r]) & (mass < edges[r + 1])
        counts.append(int(sel.sum()))
        weights.append(max(counts[-1], 1) ** 0.65)
    weights = np.array(weights) / sum(weights)
    budget_cells = budget_mb * 1e6  # ~1 byte per cell pre-zstd

    voxels = []
    for r in range(len(edges) - 1):
        cells = max(weights[r] * budget_cells, 512)
        voxel = max((volume / cells) ** (1 / 3), 0.4)
        voxels.append(voxel)
        sel = (mass >= edges[r]) & (mass < edges[r + 1])
        spec = GridSpec.for_bounds(bounds, voxel)
        g = common.density_grid(pts[sel, :3], spec)
        m = g.max()
        grids[f"g{r}"] = g.astype(np.uint16 if m > 255 else np.uint8)
    common.zsave(outdir / "grids.zst", **grids)
    (outdir / "manifest.json").write_text(json.dumps({
        "edges": [float(e) for e in edges], "counts": counts,
        "voxels": voxels, "bounds": bounds.tolist(),
    }))


def decode_massrange(outdir: Path, rng: np.random.Generator) -> np.ndarray:
    meta = json.loads((outdir / "manifest.json").read_text())
    ranging = common.load_ranging_for_artifact(outdir)
    store = spectrum_store.SpectrumStore(outdir / "spectrum.zst", ranging)
    bounds = np.array(meta["bounds"], dtype=np.float32)
    grids = common.zload(outdir / "grids.zst")
    edges = meta["edges"]
    out = []
    bin_edges_lo = 0.0
    for r in range(len(edges) - 1):
        n = meta["counts"][r]
        if n == 0:
            continue
        spec = GridSpec.for_bounds(bounds, meta["voxels"][r])
        g = grids[f"g{r}"].astype(np.float32)
        xyz = common.sample_points_from_grid(g, spec, n, rng)
        # masses: global spectrum restricted to this range
        lo_b = int(edges[r] / spectrum_store.BIN)
        hi_b = int(min(edges[r + 1], spectrum_store.MAX) / spectrum_store.BIN)
        h = np.zeros_like(store.hist)
        h[lo_b:hi_b] = store.hist[lo_b:hi_b]
        if h.sum() > 0:
            mass = common.sample_mass_from_hist(h, 0.0, spectrum_store.BIN, n, rng)
        else:
            mass = np.full(n, (edges[r] + edges[r + 1]) / 2, dtype=np.float32)
        if edges[r + 1] > spectrum_store.MAX and len(store.tail):
            tail_sel = (store.tail >= edges[r]) & (store.tail < edges[r + 1])
            tails = store.tail[tail_sel]
            if len(tails):
                n_tail = min(int(round(n * len(tails) / (len(tails) + h.sum() + EPS))), n)
                if n_tail > 0:
                    mass[:n_tail] = tails[rng.integers(0, len(tails), n_tail)]
                    rng.shuffle(mass)
        out.append(np.column_stack([xyz, mass.astype(np.float32)]))
    return np.concatenate(out).astype(np.float32)
