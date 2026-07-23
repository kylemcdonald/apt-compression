"""Adaptive-resolution hybrid codec ("hybrid_adaptive").

Motivated by two observations from the fixed-grid hybrids:
  * hybrid_massbands spends most bytes on 1nm composition fractions while its
    density backbone stays coarse -> mass/color structure survives but spatial
    resolution is visibly low.
  * hybrid_ultra spends most bytes on a uniform 0.25nm backbone, including
    homogeneous interiors where sub-voxel variation is pure Poisson noise ->
    sharp edges, but bytes are wasted where there is no structure and there is
    no mass-band conditioning.

This codec gives every abundant category (ranged species, or unranged
viewer-palette mass band) its own count grid at spatially ADAPTIVE resolution:
a coarse grid sized by category abundance, plus 2x / 4x refined blocks stored
only where a Poisson chi-square test says the sub-voxel distribution carries
real structure (sample edges, striations, poles, inclusions). Rare species and
rare unranged bands are exact Morton-packed points with per-axis bit widths.
Masses synthesize from the shared global spectrum restricted per category, so
the spectrum stays near-lossless and color stays attached to position.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import bitpack
import codecs_points
import common
import spectrum_store
from common import GridSpec

EPS = 1e-9


# --- helpers -----------------------------------------------------------------


def _block_grid_shape(shape, block):
    nx, ny, nz = shape
    return (-(-nx // block), -(-ny // block), -(-nz // block))


def _tent(rng, n):
    return (rng.random((n, 3), dtype=np.float32)
            + rng.random((n, 3), dtype=np.float32) - 0.5)


def _chunked_block_hists(lcell_sorted, starts, counts, ncells, chunk_blocks=512):
    """Yield (block_slice, hist) for blocks given per-block atom slices.

    lcell_sorted: per-atom local cell id, grouped by block (sorted-by-bid
    order). starts/counts: the slice of each block in that array. Yields
    histograms of shape (m, ncells) int32 per chunk of blocks.
    """
    nblocks = len(starts)
    for lo in range(0, nblocks, chunk_blocks):
        hi = min(lo + chunk_blocks, nblocks)
        m = hi - lo
        lengths = counts[lo:hi]
        idx = np.concatenate([
            np.arange(starts[i], starts[i] + counts[i]) for i in range(lo, hi)
        ]) if m else np.zeros(0, dtype=np.int64)
        slots = np.repeat(np.arange(m, dtype=np.int64), lengths)
        key = slots * ncells + lcell_sorted[idx]
        h = np.bincount(key, minlength=m * ncells).astype(np.int32)
        yield slice(lo, hi), h.reshape(m, ncells)


def _score_category(xyz, cspec, block, refine_min_atoms):
    """Chi-square structure scores for every candidate refinement block.

    Returns dict with per-candidate arrays (bid, atoms, score1, score2,
    occ8, occ16) plus the sorted per-atom (bid, lcell16) arrays needed to
    materialize selected blocks later.
    """
    fpb = block * 4
    ci = cspec.indices(xyz)  # (n,3) x,y,z
    nbx, nby, nbz = _block_grid_shape(cspec.shape, block)
    bc = ci // block
    bid = (bc[:, 2].astype(np.int64) * nby + bc[:, 1]) * nbx + bc[:, 0]
    off = (xyz - cspec.origin) / cspec.voxel - ci
    sub4 = np.clip((off * 4).astype(np.int32), 0, 3)
    l4 = (ci % block) * 4 + sub4
    lcell = (l4[:, 2].astype(np.int64) * fpb + l4[:, 1]) * fpb + l4[:, 0]

    order = np.argsort(bid, kind="stable")
    bid_s = bid[order]
    lcell_s = lcell[order]
    ub, starts, counts = np.unique(bid_s, return_index=True, return_counts=True)
    cand = counts >= refine_min_atoms
    cb, cs, cc = ub[cand], starts[cand], counts[cand]

    score1 = np.zeros(len(cb))
    score2 = np.zeros(len(cb))
    sigma1 = np.ones(len(cb))
    sigma2 = np.ones(len(cb))
    occ8 = np.zeros(len(cb), dtype=np.int64)
    occ16 = np.zeros(len(cb), dtype=np.int64)
    edge = np.zeros(len(cb), dtype=bool)
    for sl, h16 in _chunked_block_hists(lcell_s, cs, cc, fpb ** 3):
        m = h16.shape[0]
        atoms = cc[sl].astype(np.float64)
        g16 = h16.reshape(m, fpb, fpb, fpb).astype(np.float64)
        n8 = fpb // 2
        f8 = g16.reshape(m, n8, 2, n8, 2, n8, 2).sum(axis=(2, 4, 6))
        cl = f8.reshape(m, block, 2, block, 2, block, 2).sum(axis=(2, 4, 6))
        e8 = np.repeat(np.repeat(np.repeat(cl, 2, 1), 2, 2), 2, 3) / 8.0
        m8 = e8 > 0
        chi1 = np.where(m8, (f8 - e8) ** 2 / np.where(m8, e8, 1.0), 0.0)
        npos8 = m8.sum(axis=(1, 2, 3))
        dof1 = np.maximum(npos8 - (cl > 0).sum(axis=(1, 2, 3)), 1)
        score1[sl] = chi1.sum(axis=(1, 2, 3)) / dof1
        # std of chi2/dof for Poisson cells with mean e: sqrt((2 + 1/e)/dof)
        ebar8 = atoms / np.maximum(npos8, 1)
        sigma1[sl] = np.sqrt((2.0 + 1.0 / np.maximum(ebar8, 1e-6)) / dof1)
        e16 = np.repeat(np.repeat(np.repeat(f8, 2, 1), 2, 2), 2, 3) / 8.0
        m16 = e16 > 0
        chi2 = np.where(m16, (g16 - e16) ** 2 / np.where(m16, e16, 1.0), 0.0)
        npos16 = m16.sum(axis=(1, 2, 3))
        dof2 = np.maximum(npos16 - (f8 > 0).sum(axis=(1, 2, 3)), 1)
        score2[sl] = chi2.sum(axis=(1, 2, 3)) / dof2
        ebar16 = atoms / np.maximum(npos16, 1)
        sigma2[sl] = np.sqrt((2.0 + 1.0 / np.maximum(ebar16, 1e-6)) / dof2)
        occ8[sl] = (f8 > 0).sum(axis=(1, 2, 3))
        occ16[sl] = (g16 > 0).sum(axis=(1, 2, 3))
        # blocks mixing empty and occupied coarse voxels straddle the support
        # boundary of this category's density: the sample surface or an
        # interface. The chi-square of a half-filled voxel is bounded (~1.2 at
        # typical counts) so edges are flagged explicitly and floored later.
        empty_frac = (cl <= 0).mean(axis=(1, 2, 3))
        edge[sl] = (empty_frac > 0.03) & (empty_frac < 0.97)

    return {
        "cand_bid": cb, "cand_start": cs, "cand_count": cc,
        "score1": score1, "score2": score2,
        "sigma1": sigma1, "sigma2": sigma2,
        "occ8": occ8, "occ16": occ16,
        "edge": edge,
        "bid_sorted": bid_s, "lcell_sorted": lcell_s,
        "block_shape": (nbx, nby, nbz),
    }


def _materialize_blocks(sc, sel_idx, block, r):
    """Exact fine-count arrays for the selected candidate blocks at ratio r."""
    fpb4 = block * 4
    fpb = block * r
    ncells = fpb ** 3
    shrink = 4 // r
    n = len(sel_idx)
    out = np.zeros((n, ncells), dtype=np.int32)
    starts = sc["cand_start"][sel_idx]
    counts = sc["cand_count"][sel_idx]
    lcell4 = sc["lcell_sorted"]
    for i in range(n):
        cells4 = lcell4[starts[i]:starts[i] + counts[i]]
        if r == 4:
            cells = cells4
        else:
            lz = cells4 // (fpb4 * fpb4)
            ly = (cells4 // fpb4) % fpb4
            lx = cells4 % fpb4
            cells = ((lz // shrink) * fpb + (ly // shrink)) * fpb + (lx // shrink)
        out[i] = np.bincount(cells, minlength=ncells)
    return out


# --- encode ------------------------------------------------------------------


def encode_adaptive(
    pts: np.ndarray,
    ranging,
    outdir: Path,
    target_mb=8.0,
    coarse_target_count=8.0,
    coarse_min_voxel=1.0,
    coarse_max_voxel=4.0,
    block=4,
    refine_z=1.0,
    refine_min_atoms=48,
    est_byte_per_cell=0.75,
    rare_threshold=100_000,
    band_exact_threshold=150_000,
    unranged_mass_bands=6,
    shared=False,
    comp_cell=2.0,
    auto_range=False,
    bits=12,
    seed=0,
):
    """shared=False: every category gets its own adaptive density field
    (best when categories occupy distinct regions or carry distinct texture).
    shared=True: ONE adaptive field over all modeled atoms plus per-cell
    category fractions at comp_cell — texture that is common to all
    categories (e.g. detector striations in a homogeneously mixed sample)
    is stored once instead of once per category.

    auto_range=True ignores the supplied chemistry-derived ranging and
    detects peak windows directly from the spectrum (common.auto_ranging);
    the windows are stored in the manifest so decode needs no external
    ranging table. Species labels never influence the codec either way —
    only the mass intervals do."""
    if auto_range:
        ranging = common.auto_ranging(pts[:, 3])
    bounds = np.stack([pts[:, :3].min(axis=0), pts[:, :3].max(axis=0)])
    volume = float(np.prod(np.maximum(bounds[1] - bounds[0], 1e-6)))
    species = ranging.assign(pts[:, 3])
    counts = np.bincount(species, minlength=len(ranging.labels))
    spectrum_store.encode(pts[:, 3], outdir / "spectrum.zst")
    bits_xyz = codecs_points.axis_bits(bounds, bits)
    display_max = codecs_points.display_mass_max_for(ranging)

    mass_band = np.minimum(
        np.floor(np.maximum(pts[:, 3], 0.0)
                 / display_max * unranged_mass_bands).astype(np.int16),
        unranged_mass_bands - 1,
    )

    # --- exact parts ---------------------------------------------------------
    exact_ids = [s for s in range(1, len(ranging.labels))
                 if 0 < counts[s] < rare_threshold]
    exact_blobs = {}
    for s in exact_ids:
        xyz = pts[species == s, :3]
        exact_blobs[f"exact_{s}"] = codecs_points.pack_species_points(
            xyz, bounds, bits_xyz)
    exact_index = []
    if exact_blobs:
        payload = b"".join(exact_blobs.values())
        (outdir / "exact.zst").write_bytes(common.zbytes(payload))
        exact_index = [[k, len(v)] for k, v in exact_blobs.items()]

    unranged_band_counts = np.bincount(
        mass_band[species == 0], minlength=unranged_mass_bands)
    exact_unranged_bands = [
        int(b) for b, nb in enumerate(unranged_band_counts)
        if 0 < nb < band_exact_threshold
    ]
    exact_band_index = []
    if exact_unranged_bands:
        band_blobs = []
        for band in exact_unranged_bands:
            sub = pts[(species == 0) & (mass_band == band)]
            pos_blob, order = codecs_points.pack_species_points_with_order(
                sub[:, :3], bounds, bits_xyz)
            mass_bin = np.floor(sub[order, 3] / spectrum_store.BIN)
            mass_blob = mass_bin.clip(0, 65535).astype(np.uint16).tobytes()
            band_blobs.extend([pos_blob, mass_blob])
            exact_band_index.append({
                "band": int(band), "count": int(len(sub)),
                "pos_length": len(pos_blob), "mass_length": len(mass_blob),
            })
        (outdir / "exact_bands.zst").write_bytes(
            common.zbytes(b"".join(band_blobs)))

    # --- modeled categories --------------------------------------------------
    categories = []
    cat_masks = []
    for s in range(len(ranging.labels)):
        if counts[s] == 0 or s in exact_ids:
            continue
        if s == 0 and unranged_mass_bands:
            for band in range(unranged_mass_bands):
                if band in exact_unranged_bands:
                    continue
                sel = (species == 0) & (mass_band == band)
                n_cat = int(sel.sum())
                if n_cat == 0:
                    continue
                categories.append({"species_id": 0, "mass_band": int(band),
                                   "count": n_cat})
                cat_masks.append(sel)
        else:
            categories.append({"species_id": int(s), "mass_band": None,
                               "count": int(counts[s])})
            cat_masks.append(species == s)

    # shared mode: one field over all modeled atoms + category fractions
    comp_meta = {}
    if shared:
        cat_id = np.full(len(pts), -1, dtype=np.int16)
        for i, m in enumerate(cat_masks):
            cat_id[m] = i
        modeled_mask = cat_id >= 0
        ncat = len(categories)
        ccspec = GridSpec.for_bounds(bounds, comp_cell)
        cells = ccspec.flat(pts[modeled_mask, :3])
        joint = cells * ncat + cat_id[modeled_mask].astype(np.int64)
        C = np.bincount(joint, minlength=ccspec.ncells * ncat)
        C = C.reshape(ccspec.ncells, ncat).astype(np.float64)
        tot = C.sum(axis=1)
        active = tot > 0
        F = np.zeros_like(C)
        F[active] = C[active] / tot[active, None]
        common.zsave(outdir / "comp.zst",
                     F=np.round(np.sqrt(F) * 255.0).astype(np.uint8))
        comp_meta = {
            "comp_cell": comp_cell,
            "comp_shape": [int(v) for v in ccspec.shape],
            "global_frac": (C.sum(axis=0) / max(C.sum(), 1.0)).tolist(),
        }
        field_units = [{"count": int(modeled_mask.sum())}]
        field_masks = [modeled_mask]
    else:
        field_units = categories
        field_masks = cat_masks

    encode_fields(
        pts, bounds, field_units, field_masks, outdir, target_mb,
        coarse_target_count=coarse_target_count,
        coarse_min_voxel=coarse_min_voxel,
        coarse_max_voxel=coarse_max_voxel,
        block=block, refine_z=refine_z,
        refine_min_atoms=refine_min_atoms,
        est_byte_per_cell=est_byte_per_cell,
    )
    manifest = {
        "codec": "hybrid_adaptive",
        "target_mb": target_mb,
        "coarse_target_count": coarse_target_count,
        "block": block, "refine_z": refine_z,
        "refine_min_atoms": refine_min_atoms,
        "bits": bits, "bits_xyz": bits_xyz,
        "bounds": bounds.tolist(),
        "exact_species": "rare", "rare_threshold": rare_threshold,
        "band_exact_threshold": band_exact_threshold,
        "unranged_mass_bands": unranged_mass_bands,
        "category_display_mass_max": display_max,
        "exact_ids": exact_ids,
        "model_ids": sorted({c["species_id"] for c in categories}),
        "exact_index": exact_index,
        "exact_unranged_bands": exact_unranged_bands,
        "exact_band_index": exact_band_index,
        "model_categories": [
            {"species_id": c["species_id"], "mass_band": c["mass_band"]}
            for c in categories
        ],
        "category_counts": [c["count"] for c in categories],
        "species_counts": {str(s): int(counts[s]) for s in range(len(counts))},
        "categories": categories,
        "shared": bool(shared),
        **comp_meta,
    }
    if auto_range:
        manifest["auto_windows"] = [
            [float(ranging.lo[w]), float(ranging.hi[w])]
            for w in range(len(ranging.lo))
        ]
    if shared:
        manifest["shared_field"] = field_units[0]
    (outdir / "manifest.json").write_text(json.dumps(manifest))


def encode_fields(
    pts: np.ndarray,
    bounds: np.ndarray,
    field_units: list,
    field_masks: list,
    outdir: Path,
    target_mb: float,
    coarse_target_count=8.0,
    coarse_min_voxel=1.0,
    coarse_max_voxel=4.0,
    block=4,
    refine_z=1.0,
    refine_min_atoms=48,
    est_byte_per_cell=0.75,
):
    """Encode adaptive-resolution count fields (coarse.zst + refine.zst).

    Mutates each field_units entry with voxel/shape/blocks1/blocks2. The
    refinement byte budget is target_mb minus whatever is already in outdir.
    """
    volume = float(np.prod(np.maximum(bounds[1] - bounds[0], 1e-6)))
    coarse_arrays = {}
    scores = []
    all_cands = []  # (priority, field_idx, cand_idx, level, est_cost)
    for ci_idx, (cat, mask) in enumerate(zip(field_units, field_masks)):
        n_cat = cat["count"]
        voxel = float(np.clip((volume * coarse_target_count / n_cat) ** (1 / 3),
                              coarse_min_voxel, coarse_max_voxel))
        cspec = GridSpec.for_bounds(bounds, voxel)
        xyz = pts[mask, :3]
        flat = cspec.flat(xyz)
        cgrid = np.bincount(flat, minlength=cspec.ncells)
        nx, ny, nz = cspec.shape
        cgrid = cgrid.reshape(nz, ny, nx)
        cat["voxel"] = voxel
        cat["shape"] = [int(v) for v in cspec.shape]
        coarse_arrays[f"c{ci_idx}"] = cgrid
        sc = _score_category(xyz, cspec, block, refine_min_atoms)
        scores.append(sc)
        atoms = sc["cand_count"].astype(np.float64)
        cost1 = 64.0 + est_byte_per_cell * sc["occ8"]
        cost2 = 96.0 + est_byte_per_cell * sc["occ16"]
        # A block is a refinement candidate when its chi-square exceeds its own
        # Poisson noise floor (1 + z*sigma, sigma known from atoms/dof), so
        # dense categories admit subtle structure while sparse ones need more.
        # Support-boundary blocks are unconditional candidates with a floored
        # effective score: a half-filled voxel's chi-square is bounded (~1.2)
        # yet edges are exactly what must stay sharp.
        s1_eff = np.where(sc["edge"], np.maximum(sc["score1"], 1.5), sc["score1"])
        s2_eff = np.where(sc["edge"], np.maximum(sc["score2"], 1.25), sc["score2"])
        take1 = sc["edge"] | (sc["score1"] > 1.0 + refine_z * sc["sigma1"])
        take2 = ((sc["score2"] > 1.0 + refine_z * sc["sigma2"])
                 | (sc["edge"] & (sc["score2"] > 1.0)))
        # Volume-wide weak structure (e.g. faint crystallographic layering
        # everywhere) hides below every individual block's noise gate while
        # being obvious in aggregate. When the mean block score is confidently
        # above 1, every block becomes a candidate whose gain is floored at the
        # category mean (the best estimate of its true structure), and the byte
        # budget decides coverage depth, strongest evidence first.
        floor1 = floor2 = 0.0
        ncand = len(sc["cand_bid"])
        if ncand:
            sem1 = float(np.mean(sc["sigma1"])) / np.sqrt(ncand)
            sem2 = float(np.mean(sc["sigma2"])) / np.sqrt(ncand)
            mean1 = float(np.mean(sc["score1"]))
            mean2 = float(np.mean(sc["score2"]))
            if mean1 > 1.0 + 4.0 * sem1:
                floor1 = mean1 - 1.0
                take1 |= True
            if mean2 > 1.0 + 4.0 * sem2:
                floor2 = mean2 - 1.0
                take2 |= True
        gain1 = np.maximum(s1_eff - 1.0, floor1) * atoms
        gain2 = np.maximum(s2_eff - 1.0, floor2) * atoms
        for j in range(len(sc["cand_bid"])):
            if take2[j]:
                all_cands.append(((gain1[j] + gain2[j]) / cost2[j],
                                  ci_idx, j, 2, cost2[j]))
            elif take1[j]:
                all_cands.append((gain1[j] / cost1[j], ci_idx, j, 1, cost1[j]))

    # --- budget selection with compressed-size calibration -------------------
    fixed_bytes = common.dir_size(outdir)
    coarse_est = 0
    for k, g in coarse_arrays.items():
        m = g.max() if g.size else 0
        coarse_est += g.size * (2 if m > 255 else 1)
    coarse_est = int(coarse_est * 0.35)  # zstd-19 typical on count grids
    refine_budget = max(target_mb * 1e6 - fixed_bytes - coarse_est, 0.0)
    all_cands.sort(key=lambda t: -t[0])

    def select(budget_est: float) -> dict:
        chosen = {}
        spent = 0.0
        for prio, c_idx, j, level, cost in all_cands:
            if spent + cost > budget_est:
                continue
            chosen[(c_idx, j)] = level
            spent += cost
        return chosen

    def materialize(chosen: dict) -> dict:
        arrays = {}
        for c_idx, cat in enumerate(field_units):
            sc = scores[c_idx]
            for level, r, tag in ((1, 2, "1"), (2, 4, "2")):
                sel = np.array(sorted(j for (c, j), lv in chosen.items()
                                      if c == c_idx and lv == level),
                               dtype=np.int64)
                cat[f"blocks{tag}"] = int(len(sel))
                if not len(sel):
                    continue
                arr = _materialize_blocks(sc, sel, block, r)
                m = arr.max() if arr.size else 0
                arrays[f"b{tag}_{c_idx}"] = sc["cand_bid"][sel].astype(np.uint32)
                arrays[f"a{tag}_{c_idx}"] = arr.astype(
                    np.uint16 if m > 255 else np.uint8)
        return arrays

    # Estimated cost vs zstd-compressed reality differs by 2-4x, so measure and
    # re-select: converge the actual refine.zst size onto the byte budget.
    refine_arrays = {}
    selected = {}
    if all_cands and refine_budget > 0:
        budget_est = refine_budget
        for _attempt in range(4):
            selected = select(budget_est)
            refine_arrays = materialize(selected)
            if not refine_arrays:
                break
            actual = common.zsave(outdir / "refine.zst", **refine_arrays)
            est_spent = sum(cost for prio, c, j, lv, cost in all_cands
                            if (c, j) in selected)
            in_band = 0.90 * refine_budget <= actual <= 1.08 * refine_budget
            exhausted = (len(selected) == len(all_cands)
                         and actual <= 1.08 * refine_budget)
            if in_band or exhausted:
                break
            scale = actual / max(est_spent, 1.0)
            budget_est = refine_budget / max(scale, 1e-3)

    # zero the coarse counts now carried by refined blocks
    for c_idx, cat in enumerate(field_units):
        sc = scores[c_idx]
        nbx, nby, nbz = sc["block_shape"]
        cgrid = coarse_arrays[f"c{c_idx}"]
        bids = np.concatenate([
            refine_arrays[f"b{tag}_{c_idx}"].astype(np.int64)
            for tag in ("1", "2") if f"b{tag}_{c_idx}" in refine_arrays
        ]) if any(f"b{tag}_{c_idx}" in refine_arrays for tag in ("1", "2")) \
            else np.zeros(0, dtype=np.int64)
        bz = bids // (nbx * nby)
        rem = bids % (nbx * nby)
        by = rem // nbx
        bx = rem % nbx
        for k in range(len(bids)):
            cgrid[bz[k] * block:(bz[k] + 1) * block,
                  by[k] * block:(by[k] + 1) * block,
                  bx[k] * block:(bx[k] + 1) * block] = 0

    for k in list(coarse_arrays):
        g = coarse_arrays[k]
        m = g.max() if g.size else 0
        coarse_arrays[k] = g.astype(np.uint16 if m > 255 else np.uint8)
    common.zsave(outdir / "coarse.zst", **coarse_arrays)


# --- decode ------------------------------------------------------------------


def _place_counts(idx3, counts, origin, voxel, rng, grid=None):
    """Place exactly `counts[i]` atoms in each cell idx3[i] with a tent kernel.

    When `grid` (nz, ny, nx counts) is given, the tent jitter is clamped on any
    axis side whose neighboring cell is empty, so unrefined boundary voxels do
    not bleed atoms past the sample surface or across internal interfaces.
    """
    reps = counts.astype(np.int64)
    base = np.repeat(idx3, reps, axis=0).astype(np.float32)
    n = len(base)
    tent = _tent(rng, n)
    if grid is not None and len(idx3):
        nz, ny, nx = grid.shape
        occ = np.asarray(grid) > 0
        lo_lim = np.full((len(idx3), 3), -0.5, dtype=np.float32)
        hi_lim = np.full((len(idx3), 3), 1.5, dtype=np.float32)
        ix, iy, iz = idx3[:, 0], idx3[:, 1], idx3[:, 2]
        for a, (ii, size) in enumerate(((ix, nx), (iy, ny), (iz, nz))):
            minus = ii > 0
            neigh = [iz.copy(), iy.copy(), ix.copy()]
            neigh[2 - a] = np.maximum(ii - 1, 0)
            empty_minus = ~occ[neigh[0], neigh[1], neigh[2]] | ~minus
            lo_lim[empty_minus, a] = 0.0
            plus = ii < size - 1
            neigh = [iz.copy(), iy.copy(), ix.copy()]
            neigh[2 - a] = np.minimum(ii + 1, size - 1)
            empty_plus = ~occ[neigh[0], neigh[1], neigh[2]] | ~plus
            hi_lim[empty_plus, a] = 1.0 - 1e-4
        tent = np.clip(tent, np.repeat(lo_lim, reps, axis=0),
                       np.repeat(hi_lim, reps, axis=0))
    return origin.astype(np.float32) + (base + tent) * np.float32(voxel)


def _place_field(fu: dict, idx: int, coarse: dict, refine: dict, block: int,
                 bounds: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Reconstruct one adaptive field's atom positions (coarse + refined)."""
    voxel = float(fu["voxel"])
    nx, ny, nz = fu["shape"]
    nbx, nby, nbz = _block_grid_shape((nx, ny, nz), block)
    parts = []

    cgrid = coarse[f"c{idx}"]
    # jitter-clamp occupancy: refined-block regions count as occupied even
    # though their coarse counts were zeroed at encode time
    occ = cgrid.astype(np.int64).copy()
    for tag in ("1", "2"):
        bkey = f"b{tag}_{idx}"
        if bkey not in refine:
            continue
        for bb in refine[bkey].astype(np.int64):
            bz0 = int(bb // (nbx * nby))
            remb0 = int(bb % (nbx * nby))
            occ[bz0 * block:(bz0 + 1) * block,
                (remb0 // nbx) * block:(remb0 // nbx + 1) * block,
                (remb0 % nbx) * block:(remb0 % nbx + 1) * block] += 1
    flat = np.nonzero(cgrid.ravel())[0]
    if len(flat):
        cnt = cgrid.ravel()[flat].astype(np.int64)
        iz, rem = np.divmod(flat, nx * ny)
        iy, ix = np.divmod(rem, nx)
        idx3 = np.stack([ix, iy, iz], axis=1)
        parts.append(_place_counts(idx3, cnt, bounds[0], voxel, rng, grid=occ))

    for tag, r in (("1", 2), ("2", 4)):
        bkey, akey = f"b{tag}_{idx}", f"a{tag}_{idx}"
        if bkey not in refine:
            continue
        bids = refine[bkey].astype(np.int64)
        fpb = block * r
        arr = refine[akey].reshape(len(bids), fpb ** 3)
        bi, cell = np.nonzero(arr)
        cnt = arr[bi, cell].astype(np.int64)
        bb = bids[bi]
        bz = bb // (nbx * nby)
        remb = bb % (nbx * nby)
        by = remb // nbx
        bx = remb % nbx
        lz = cell // (fpb * fpb)
        ly = (cell // fpb) % fpb
        lx = cell % fpb
        idx3 = np.stack([bx * fpb + lx, by * fpb + ly, bz * fpb + lz], axis=1)
        parts.append(_place_counts(idx3, cnt, bounds[0], voxel / r, rng))

    xyz = parts[0] if len(parts) == 1 else np.concatenate(parts)
    hi_clip = bounds[1] - 1e-4
    np.clip(xyz, bounds[0], hi_clip, out=xyz)
    return xyz


def _category_mass(store, cat: dict, n: int, n_bands: int, display_max: float,
                   rng: np.random.Generator) -> np.ndarray:
    s = int(cat["species_id"])
    band = cat["mass_band"]
    if band is None:
        return store.sample_for_species(s, n, rng)
    return store.sample_for_species_band(
        s, int(band), n_bands, display_max, n, rng)


def decode_adaptive(outdir: Path, rng: np.random.Generator) -> np.ndarray:
    meta = json.loads((outdir / "manifest.json").read_text())
    if meta.get("auto_windows") is not None:
        ranging = common.ranging_from_windows(meta["auto_windows"])
    else:
        ranging = common.load_ranging_for_artifact(outdir)
    store = spectrum_store.SpectrumStore(outdir / "spectrum.zst", ranging)
    bounds = np.array(meta["bounds"], dtype=np.float32)
    block = int(meta["block"])
    n_bands = int(meta["unranged_mass_bands"])
    display_max = float(meta["category_display_mass_max"])
    coarse = common.zload(outdir / "coarse.zst")
    refine = (common.zload(outdir / "refine.zst")
              if (outdir / "refine.zst").exists() else {})

    out = []
    if meta.get("shared"):
        xyz = _place_field(meta["shared_field"], 0, coarse, refine, block,
                           bounds, rng)
        n = len(xyz)
        ncat = len(meta["categories"])
        Fq = common.zload(outdir / "comp.zst")["F"].astype(np.float64)
        F = (Fq / 255.0) ** 2
        tot = F.sum(axis=1)
        gf = np.asarray(meta["global_frac"], dtype=np.float64)
        F = np.where(tot[:, None] > EPS,
                     F / np.maximum(tot, EPS)[:, None], gf[None, :])
        ccspec = GridSpec(bounds[0].copy(), float(meta["comp_cell"]),
                          tuple(meta["comp_shape"]))
        cum = np.cumsum(F, axis=1)
        cells = ccspec.flat(xyz)
        cat_of = np.empty(n, dtype=np.int16)
        chunk = 2_000_000
        for a in range(0, n, chunk):
            b = min(a + chunk, n)
            c = cum[cells[a:b]]
            r = rng.random(b - a) * c[:, -1]
            cat_of[a:b] = (c < r[:, None]).sum(axis=1)
        np.clip(cat_of, 0, ncat - 1, out=cat_of)
        for j, cat in enumerate(meta["categories"]):
            idx = np.flatnonzero(cat_of == j)
            if not len(idx):
                continue
            mass = _category_mass(store, cat, len(idx), n_bands, display_max, rng)
            out.append(np.column_stack([xyz[idx], mass.astype(np.float32)]))
    else:
        for ci_idx, cat in enumerate(meta["categories"]):
            xyz = _place_field(cat, ci_idx, coarse, refine, block, bounds, rng)
            mass = _category_mass(store, cat, len(xyz), n_bands, display_max, rng)
            out.append(np.column_stack([xyz, mass.astype(np.float32)]))

    bits_xyz = meta.get("bits_xyz", meta["bits"])
    if meta["exact_index"]:
        payload = common._dctx.decompress((outdir / "exact.zst").read_bytes())
        off = 0
        for key, ln in meta["exact_index"]:
            s = int(key.split("_")[1])
            blob = payload[off:off + ln]
            off += ln
            n = meta["species_counts"][str(s)]
            xyz = codecs_points.unpack_species_points(
                blob, n, bounds, bits_xyz, rng)
            mass = store.sample_for_species(s, n, rng)
            out.append(np.column_stack([xyz, mass]))
    if meta.get("exact_band_index"):
        payload = common._dctx.decompress(
            (outdir / "exact_bands.zst").read_bytes())
        off = 0
        for info in meta["exact_band_index"]:
            pos_blob = payload[off:off + int(info["pos_length"])]
            off += int(info["pos_length"])
            mass_blob = payload[off:off + int(info["mass_length"])]
            off += int(info["mass_length"])
            n = int(info["count"])
            xyz = codecs_points.unpack_species_points(
                pos_blob, n, bounds, bits_xyz, rng)
            mass = (np.frombuffer(mass_blob, dtype=np.uint16).astype(np.float32)
                    + 0.5) * spectrum_store.BIN
            out.append(np.column_stack([xyz, mass]))
    return np.concatenate(out).astype(np.float32)
