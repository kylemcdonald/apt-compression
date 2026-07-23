"""Bin-sorted hybrid codec ("hybrid_binsort") — the unified final codec.

Kyle's idea, taken literally: slice the mass spectrum into fine bins on a
single u16 code axis (0.01 Da below 120 Da, 0.025 Da above), sort bins by
occupancy, store the atoms of LOW-count bins exactly (Morton positions +
per-atom code, mass implicit), and model the HIGH-count bins with one shared
adaptive-resolution density field plus per-cell category fractions, where a
category is a contiguous run (<= ~0.3 Da) of modeled bins.

No ranging, no chemistry: the exact/modeled split is one occupancy threshold
(plus valley-split light-run promotion), reconstructible at decode from the
stored spectrum. Three structural refinements over the first version:

* The FRACTION FIELD is adaptive-resolution too: a coarse base grid of
  sqrt-quantized fractions, plus 2x/4x finer cells storing raw category
  counts only where a multinomial chi-square test says the local mix deviates
  from the parent prediction. Decode uses the finest stored ancestor.
* A budget-closing OUTER LOOP: when the density/composition evidence pools
  exhaust below the byte target (homogeneous samples), the remaining budget
  raises the exactness threshold instead — more of the spectrum becomes
  exact atoms. With a large enough budget the codec approaches lossless.
* The exact mass-code stream is stored as separate low/high byte planes,
  which zstd compresses better than interleaved u16.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import codecs_adaptive
import codecs_points
import common
import spectrum_store
from common import GridSpec

NBINS = spectrum_store.NBINS
BIN = spectrum_store.BIN
MAX = spectrum_store.MAX
TAIL_STEP = 1.0 / spectrum_store.TAIL_SCALE  # Da per tail code above MAX
TAIL_BASE = NBINS  # u16 codes >= TAIL_BASE encode the >MAX tail
CODE_N = 65536
EPS = 1e-9


# --- mass code axis ----------------------------------------------------------


def _exact_bins_for(hist: np.ndarray, threshold: int) -> np.ndarray:
    """Bins stored exactly: occupied and at-or-below the count threshold."""
    return (hist > 0) & (hist <= threshold)


def _mass_codes(mass: np.ndarray) -> np.ndarray:
    """u16 code per atom: fine bin below MAX, coarse tail step above."""
    body = np.minimum((np.maximum(mass, 0.0) / BIN).astype(np.int64),
                      NBINS - 1)
    tail = TAIL_BASE + np.round((mass - MAX) / TAIL_STEP).astype(np.int64)
    return np.where(mass >= MAX, np.clip(tail, TAIL_BASE, CODE_N - 1),
                    body).astype(np.uint16)


def _codes_to_mass(code_vals: np.ndarray) -> np.ndarray:
    """Continuous code values (code + within-bin uniform) -> mass in Da."""
    body = code_vals * BIN
    tail = MAX + (code_vals - TAIL_BASE) * TAIL_STEP
    return np.where(code_vals >= TAIL_BASE, tail, body).astype(np.float32)


def _decode_code_hist(spectrum: dict) -> np.ndarray:
    """Exact code-axis histogram reconstructed from the stored spectrum."""
    code_hist = np.zeros(CODE_N, dtype=np.float64)
    code_hist[:NBINS] = spectrum["hist"].astype(np.float64)
    tail_codes = TAIL_BASE + np.clip(
        spectrum["tail"].astype(np.int64), 0, CODE_N - 1 - TAIL_BASE)
    if len(tail_codes):
        code_hist += np.bincount(tail_codes, minlength=CODE_N)
    return code_hist


def _threshold_for_budget(hist: np.ndarray, budget_atoms: float) -> int:
    """Largest count threshold whose exact atoms fit the atom budget."""
    counts = np.sort(hist[hist > 0])
    cum = np.cumsum(counts)
    k = int(np.searchsorted(cum, budget_atoms, side="right"))
    if k == 0:
        return 0
    return int(counts[k - 1])


def _model_runs(hist: np.ndarray, exact_bins: np.ndarray,
                gap_merge_da=0.05, max_run_da=0.3) -> list:
    """Contiguous runs of modeled bins -> categories [[lo_bin, hi_bin), ...]."""
    modeled = (hist > 0) & ~exact_bins
    edges = np.diff(modeled.astype(np.int8), prepend=0, append=0)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    gap = max(int(round(gap_merge_da / BIN)), 1)
    runs = []
    for a, b in zip(starts, ends):
        if runs and a - runs[-1][1] <= gap:
            runs[-1][1] = int(b)
        else:
            runs.append([int(a), int(b)])
    max_bins = max(int(round(max_run_da / BIN)), 1)
    out = []
    for a, b in runs:
        pieces = max(1, -(-(b - a) // max_bins))
        cuts = np.linspace(a, b, pieces + 1).round().astype(int)
        for i in range(pieces):
            if cuts[i + 1] > cuts[i]:
                out.append([int(cuts[i]), int(cuts[i + 1])])
    return out


def _category_hist(store_hist: np.ndarray, exact_bins: np.ndarray,
                   run) -> np.ndarray:
    h = np.zeros_like(store_hist)
    lo, hi = int(run[0]), int(run[1])
    h[lo:hi] = np.where(exact_bins[lo:hi], 0.0, store_hist[lo:hi])
    return h


# --- exact atom stream -------------------------------------------------------


def _pack_exact(pts, codes, exact_mask, bounds, bits_xyz) -> bytes:
    sub = pts[exact_mask]
    pos_blob, order = codecs_points.pack_species_points_with_order(
        sub[:, :3], bounds, bits_xyz)
    c = codes[exact_mask][order]
    lo = (c & 0xFF).astype(np.uint8)
    hi = (c >> 8).astype(np.uint8)
    return common.zbytes(pos_blob + lo.tobytes() + hi.tobytes())


def _unpack_exact(payload, n, bounds, bits_xyz, rng):
    raw = common._dctx.decompress(payload)
    pos_blob = raw[:-2 * n]
    lo = np.frombuffer(raw[-2 * n:-n], dtype=np.uint8)
    hi = np.frombuffer(raw[-n:], dtype=np.uint8)
    codes = lo.astype(np.uint16) | (hi.astype(np.uint16) << 8)
    xyz = codecs_points.unpack_species_points(pos_blob, n, bounds, bits_xyz, rng)
    mass = _codes_to_mass(codes.astype(np.float64) + rng.random(n))
    return xyz, mass


# --- adaptive fraction field -------------------------------------------------


def _parent_flat(flat, spec_child, spec_parent):
    nxc, nyc, _ = spec_child.shape
    nxp, nyp, _ = spec_parent.shape
    iz = flat // (nxc * nyc)
    iy = (flat // nxc) % nyc
    ix = flat % nxc
    return ((np.minimum(iz // 2, spec_parent.shape[2] - 1) * nyp
             + np.minimum(iy // 2, nyp - 1)) * nxp
            + np.minimum(ix // 2, nxp - 1))


def _encode_comp(xyz, cat, bounds, ncat, cc0, budget, refine_z=1.0,
                 min_cell_atoms=(24, 12), chunk_cells=2048):
    """Adaptive-resolution category fractions.

    Base grid at cc0 stores sqrt-u8 fractions everywhere. Cells at cc0/2 and
    cc0/4 whose observed category mix deviates from the parent prediction
    (multinomial chi-square above the Poisson noise floor) store raw counts,
    greedily within `budget`. Returns (arrays, meta).
    """
    spec0 = GridSpec.for_bounds(bounds, cc0)
    spec1 = GridSpec.for_bounds(bounds, cc0 / 2.0)
    spec2 = GridSpec.for_bounds(bounds, cc0 / 4.0)
    cell0 = spec0.flat(xyz)
    cell1 = spec1.flat(xyz)
    cell2 = spec2.flat(xyz)

    C0 = np.bincount(cell0 * ncat + cat.astype(np.int64),
                     minlength=spec0.ncells * ncat)
    C0 = C0.reshape(spec0.ncells, ncat).astype(np.float64)
    tot0 = C0.sum(axis=1)
    global_frac = C0.sum(axis=0) / max(C0.sum(), 1.0)
    F0 = np.where(tot0[:, None] > 0, C0 / np.maximum(tot0, 1.0)[:, None],
                  global_frac[None, :])

    # child index of each atom's spec2 cell within its spec1 cell
    i1 = spec1.indices(xyz)
    i2 = spec2.indices(xyz)
    child = np.clip(i2 - 2 * i1, 0, 1)
    childflat = ((child[:, 2].astype(np.int64) * 2 + child[:, 1]) * 2
                 + child[:, 0])

    order = np.argsort(cell1, kind="stable")
    c1s = cell1[order]
    cats = cat[order].astype(np.int64)
    childs = childflat[order]
    uc, ustart, ucount = np.unique(c1s, return_index=True, return_counts=True)

    cands = []  # (prio, level, cell_id, n, nnz, row)
    for lo_i in range(0, len(uc), chunk_cells):
        hi_i = min(lo_i + chunk_cells, len(uc))
        m = hi_i - lo_i
        a = ustart[lo_i]
        b = ustart[hi_i - 1] + ucount[hi_i - 1]
        slots = np.repeat(np.arange(m, dtype=np.int64), ucount[lo_i:hi_i])
        h1 = np.bincount(slots * ncat + cats[a:b],
                         minlength=m * ncat).reshape(m, ncat).astype(np.float64)
        n1 = h1.sum(axis=1)
        p0 = F0[_parent_flat(uc[lo_i:hi_i], spec1, spec0)]
        e1 = n1[:, None] * p0
        m1 = e1 > 0
        chi1 = np.where(m1, (h1 - e1) ** 2 / np.where(m1, e1, 1.0), 0.0)
        kpos1 = np.maximum(m1.sum(axis=1), 2)
        dof1 = kpos1 - 1.0
        score1 = chi1.sum(axis=1) / dof1
        ebar1 = n1 / kpos1
        sigma1 = np.sqrt((2.0 + 1.0 / np.maximum(ebar1, 1e-6)) / dof1)
        nnz1 = (h1 > 0).sum(axis=1)
        take1 = (n1 >= min_cell_atoms[0]) & (score1 > 1.0 + refine_z * sigma1)
        for j in np.flatnonzero(take1):
            prio = (score1[j] - 1.0) * n1[j] / (6.0 + 1.2 * nnz1[j])
            cands.append((prio, 1, int(uc[lo_i + j]), h1[j]))

        h2 = np.bincount((slots * 8 + childs[a:b]) * ncat + cats[a:b],
                         minlength=m * 8 * ncat)
        h2 = h2.reshape(m * 8, ncat).astype(np.float64)
        n2 = h2.sum(axis=1)
        e2 = np.repeat(h1, 8, axis=0) / 8.0
        m2 = e2 > 0
        chi2 = np.where(m2, (h2 - e2) ** 2 / np.where(m2, e2, 1.0), 0.0)
        kpos2 = np.maximum(m2.sum(axis=1), 2)
        dof2 = kpos2 - 1.0
        score2 = chi2.sum(axis=1) / dof2
        ebar2 = n2 / kpos2
        sigma2 = np.sqrt((2.0 + 1.0 / np.maximum(ebar2, 1e-6)) / dof2)
        nnz2 = (h2 > 0).sum(axis=1)
        take2 = (n2 >= min_cell_atoms[1]) & (score2 > 1.0 + refine_z * sigma2)
        if take2.any():
            # spec2 flat ids for the chunk's (cell1, child) rows
            nx1, ny1, _ = spec1.shape
            nx2, ny2, nz2s = spec2.shape
            ucells = uc[lo_i:hi_i]
            iz1 = ucells // (nx1 * ny1)
            iy1 = (ucells // nx1) % ny1
            ix1 = ucells % nx1
            for j in np.flatnonzero(take2):
                ci, ch = divmod(j, 8)
                cz, rem = divmod(ch, 4)
                cy, cx = divmod(rem, 2)
                ix = min(2 * ix1[ci] + cx, nx2 - 1)
                iy = min(2 * iy1[ci] + cy, ny2 - 1)
                iz = min(2 * iz1[ci] + cz, nz2s - 1)
                flat2 = (iz * ny2 + iy) * nx2 + ix
                prio = (score2[j] - 1.0) * n2[j] / (6.0 + 1.2 * nnz2[j])
                cands.append((prio, 2, int(flat2), h2[j]))

    cands.sort(key=lambda t: -t[0])
    sel = {1: [], 2: []}
    spent = 0.0
    for prio, level, cid, row in cands:
        cost = 6.0 + 1.2 * float((row > 0).sum())
        if spent + cost > budget:
            continue
        sel[level].append((cid, row))
        spent += cost

    arrays = {
        "F0": np.round(np.sqrt(F0) * 255.0).astype(np.uint8),
    }
    meta = {
        "comp_cell": float(cc0),
        "comp_shape0": [int(v) for v in spec0.shape],
        "comp_shape1": [int(v) for v in spec1.shape],
        "comp_shape2": [int(v) for v in spec2.shape],
        "global_frac": global_frac.tolist(),
        "comp_refined": [len(sel[1]), len(sel[2])],
    }
    for level in (1, 2):
        if not sel[level]:
            continue
        sel[level].sort(key=lambda t: t[0])
        ids = np.array([t[0] for t in sel[level]], dtype=np.uint32)
        rows = np.stack([t[1] for t in sel[level]])
        mmax = rows.max() if rows.size else 0
        arrays[f"cid{level}"] = ids
        arrays[f"crow{level}"] = rows.astype(
            np.uint16 if mmax > 255 else np.uint8)
    return arrays, meta


def _decode_comp_assign(meta, arrays, bounds, xyz, rng, chunk=100_000):
    """Per-atom category via the finest stored composition ancestor."""
    ncat = len(meta["global_frac"])
    gf = np.asarray(meta["global_frac"], dtype=np.float64)
    cc0 = float(meta["comp_cell"])
    spec0 = GridSpec(bounds[0].copy(), cc0, tuple(meta["comp_shape0"]))
    spec1 = GridSpec(bounds[0].copy(), cc0 / 2.0, tuple(meta["comp_shape1"]))
    spec2 = GridSpec(bounds[0].copy(), cc0 / 4.0, tuple(meta["comp_shape2"]))

    F0 = (arrays["F0"].astype(np.float64) / 255.0) ** 2
    t0 = F0.sum(axis=1)
    F0 = np.where(t0[:, None] > EPS, F0 / np.maximum(t0, EPS)[:, None],
                  gf[None, :])
    cum0 = np.cumsum(F0, axis=1).astype(np.float32)
    cums = {0: cum0}
    ids = {}
    for level in (1, 2):
        if f"cid{level}" in arrays:
            ids[level] = arrays[f"cid{level}"].astype(np.int64)
            rows = arrays[f"crow{level}"].astype(np.float64)
            tot = rows.sum(axis=1)
            fr = np.where(tot[:, None] > 0, rows / np.maximum(tot, 1.0)[:, None],
                          gf[None, :])
            cums[level] = np.cumsum(fr, axis=1).astype(np.float32)

    n = len(xyz)
    cat_of = np.empty(n, dtype=np.int16)
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        sub = xyz[a:b]
        rows = np.empty((b - a, ncat), dtype=np.float32)
        remaining = np.ones(b - a, dtype=bool)
        for level, spec in ((2, spec2), (1, spec1)):
            if level not in ids:
                continue
            c = spec.flat(sub)
            j = np.searchsorted(ids[level], c)
            j = np.minimum(j, len(ids[level]) - 1)
            hit = remaining & (ids[level][j] == c)
            rows[hit] = cums[level][j[hit]]
            remaining &= ~hit
        if remaining.any():
            c0 = spec0.flat(sub[remaining])
            rows[remaining] = cum0[c0]
        r = rng.random(b - a, dtype=np.float32) * rows[:, -1]
        cat_of[a:b] = (rows < r[:, None]).sum(axis=1)
    return np.clip(cat_of, 0, ncat - 1)


# --- main codec --------------------------------------------------------------


def encode_binsort(
    pts: np.ndarray,
    ranging,  # unused: the codec is chemistry- and ranging-agnostic
    outdir: Path,
    target_mb=10.0,
    exact_share_init=0.4,
    threshold_share=0.6,
    run_exact_threshold=150_000,
    comp_cell=4.0,
    comp_share=0.3,
    gap_merge_da=0.05,
    max_run_da=0.3,
    max_categories=384,
    coarse_target_count=8.0,
    coarse_min_voxel=1.0,
    coarse_max_voxel=4.0,
    block=4,
    refine_z=1.0,
    refine_min_atoms=48,
    est_byte_per_cell=0.75,
    bits=12,
    seed=0,
):
    bounds = np.stack([pts[:, :3].min(axis=0), pts[:, :3].max(axis=0)])
    mass = pts[:, 3]
    spectrum_store.encode(mass, outdir / "spectrum.zst")
    bits_xyz = codecs_points.axis_bits(bounds, bits)
    codes = _mass_codes(mass)
    code_hist = np.bincount(codes, minlength=CODE_N).astype(np.float64)
    target = target_mb * 1e6

    # sample-shaped expectation for run-level spatial structure scoring
    spec0p = GridSpec.for_bounds(bounds, comp_cell)
    cell0_all = spec0p.flat(pts[:, :3])
    bulk = np.bincount(cell0_all, minlength=spec0p.ncells).astype(np.float64)
    bulk_p = bulk / max(bulk.sum(), 1.0)

    exact_budget = exact_share_init * target
    bytes_per_atom = 3.0
    result = None
    for _outer in range(3):
        for name in ("exact.zst", "comp.zst", "coarse.zst", "refine.zst"):
            p = outdir / name
            if p.exists():
                p.unlink()

        # --- exact side: occupancy threshold + capped light-run promotion ----
        threshold = _threshold_for_budget(
            code_hist, threshold_share * exact_budget / bytes_per_atom)
        exact_bins = _exact_bins_for(code_hist, threshold)
        promoted_runs = []
        exact_atoms = float(code_hist[exact_bins].sum())
        atom_cap = exact_budget / max(bytes_per_atom, 1e-6)

        # Promotion of fine chunks (~0.3 Da, independent of the category cap)
        # to exact storage: a chunk whose atoms concentrate differently from
        # the bulk sample (layers, dots, wires) is exactly where distribution
        # modeling fails, whatever its bin heights. Chunk granularity keeps a
        # structured peak's signal undiluted by the diffuse tail bins around
        # it. Structured chunks smallest-first — sparse structured chunks are
        # both the cheapest to keep and the ones the modeled fields cannot
        # carry (trace components vanish into fraction quantization) — then
        # diffuse light chunks, while the exact-atom budget lasts.
        chunks = _model_runs(code_hist, exact_bins, gap_merge_da, max_run_da)
        if chunks:
            runmap = np.full(CODE_N, -1, dtype=np.int32)
            for ridx, (lo, hi) in enumerate(chunks):
                seg = slice(lo, hi)
                runmap[seg] = np.where(exact_bins[seg], -1, ridx)
            rid = runmap[codes]
            rsel = rid >= 0
            ncell = spec0p.ncells
            rid_sel = rid[rsel].astype(np.int64)
            cell_sel = cell0_all[rsel]
            nchunk = len(chunks)
            nrun = np.zeros(nchunk)
            score = np.zeros(nchunk)
            sigma = np.ones(nchunk)
            group = 64
            for g0 in range(0, nchunk, group):
                g1 = min(g0 + group, nchunk)
                gsel = (rid_sel >= g0) & (rid_sel < g1)
                joint = (rid_sel[gsel] - g0) * ncell + cell_sel[gsel]
                H = np.bincount(joint, minlength=(g1 - g0) * ncell)
                H = H.reshape(g1 - g0, ncell).astype(np.float64)
                n_g = H.sum(axis=1)
                e = n_g[:, None] * bulk_p[None, :]
                mk = e > 0
                chi = np.where(mk, (H - e) ** 2 / np.where(mk, e, 1.0), 0.0)
                dof = np.maximum(mk.sum(axis=1) - 1, 1)
                nrun[g0:g1] = n_g
                score[g0:g1] = chi.sum(axis=1) / dof
                ebar = n_g / np.maximum(mk.sum(axis=1), 1)
                sigma[g0:g1] = np.sqrt(
                    (2.0 + 1.0 / np.maximum(ebar, 1e-6)) / dof)
            structured = score > 1.0 + 3.0 * sigma
            order = sorted(
                range(nchunk),
                key=lambda i: (0 if structured[i] else 1, nrun[i]))
            for i in order:
                lo, hi = chunks[i]
                if nrun[i] <= 0:
                    continue
                if not structured[i] and nrun[i] >= run_exact_threshold:
                    continue
                if exact_atoms + nrun[i] > atom_cap:
                    continue
                exact_bins[lo:hi] |= code_hist[lo:hi] > 0
                promoted_runs.append([int(lo), int(hi)])
                exact_atoms += nrun[i]

        exact_mask = exact_bins[codes]
        n_exact = int(exact_mask.sum())
        payload = (_pack_exact(pts, codes, exact_mask, bounds, bits_xyz)
                   if n_exact else common.zbytes(b""))
        (outdir / "exact.zst").write_bytes(payload)
        if n_exact:
            bytes_per_atom = len(payload) / n_exact

        # --- modeled categories (rebuilt after promotion) --------------------
        run_da = max_run_da
        while True:
            runs = _model_runs(code_hist, exact_bins, gap_merge_da, run_da)
            if len(runs) <= max_categories or run_da >= 4.0:
                break
            run_da *= 2.0
        ncat = len(runs)
        modeled_mask = ~exact_mask
        n_modeled = int(modeled_mask.sum())
        cat_of_bin = np.full(CODE_N, -1, dtype=np.int16)
        for j, (lo, hi) in enumerate(runs):
            seg = slice(lo, hi)
            keep = ~exact_bins[seg]
            cat_of_bin[seg] = np.where(keep, j, cat_of_bin[seg])
        cat_id = cat_of_bin[codes]
        cat_id[exact_mask] = -1

        comp_meta = {}
        if n_modeled and ncat:
            remaining = max(target - common.dir_size(outdir), 0.0)
            # coarser base as category count grows keeps the dense fraction
            # matrix affordable; adaptive refinement restores local detail
            cc0_eff = comp_cell * max(1.0, (ncat / 96.0) ** (1.0 / 3.0))
            comp_arrays, comp_meta = _encode_comp(
                pts[modeled_mask, :3], cat_id[modeled_mask], bounds, ncat,
                cc0_eff, comp_share * remaining, refine_z=refine_z)
            common.zsave(outdir / "comp.zst", **comp_arrays)

        field = {"count": n_modeled}
        if n_modeled:
            codecs_adaptive.encode_fields(
                pts, bounds, [field], [modeled_mask], outdir, target_mb,
                coarse_target_count=coarse_target_count,
                coarse_min_voxel=coarse_min_voxel,
                coarse_max_voxel=coarse_max_voxel,
                block=block, refine_z=refine_z,
                refine_min_atoms=refine_min_atoms,
                est_byte_per_cell=est_byte_per_cell,
            )

        result = {
            "threshold": int(threshold),
            "promoted_runs": promoted_runs,
            "n_exact": n_exact, "n_modeled": n_modeled,
            "runs": runs, "field": field, "comp_meta": comp_meta,
        }
        total = common.dir_size(outdir)
        if total > 1.06 * target:
            exact_budget = max(exact_budget - (total - target) * 1.1, 0.1e6)
            continue
        if total < 0.85 * target and n_exact < len(pts):
            # pools exhausted below target: spend the slack on exactness
            exact_budget += (target - total) * 0.9
            continue
        break

    manifest = {
        "codec": "hybrid_binsort",
        "target_mb": target_mb,
        "threshold": result["threshold"],
        "promoted_runs": result["promoted_runs"],
        "bits": bits, "bits_xyz": bits_xyz,
        "block": block,
        "bounds": bounds.tolist(),
        "exact_count": result["n_exact"],
        "modeled_count": result["n_modeled"],
        "runs": result["runs"],
        "shared_field": result["field"],
        **result["comp_meta"],
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest))


def decode_binsort(outdir: Path, rng: np.random.Generator) -> np.ndarray:
    meta = json.loads((outdir / "manifest.json").read_text())
    bounds = np.array(meta["bounds"], dtype=np.float32)
    bits_xyz = meta["bits_xyz"]
    block = int(meta["block"])
    code_hist = _decode_code_hist(common.zload(outdir / "spectrum.zst"))
    exact_bins = _exact_bins_for(code_hist, int(meta["threshold"]))
    for lo, hi in meta.get("promoted_runs", []):
        exact_bins[int(lo):int(hi)] |= code_hist[int(lo):int(hi)] > 0
    out = []

    n_exact = int(meta["exact_count"])
    if n_exact:
        xyz, mass = _unpack_exact(
            (outdir / "exact.zst").read_bytes(), n_exact, bounds, bits_xyz, rng)
        out.append(np.column_stack([xyz, mass]))

    if int(meta["modeled_count"]):
        coarse = common.zload(outdir / "coarse.zst")
        refine = (common.zload(outdir / "refine.zst")
                  if (outdir / "refine.zst").exists() else {})
        xyz = codecs_adaptive._place_field(
            meta["shared_field"], 0, coarse, refine, block, bounds, rng)
        comp_arrays = common.zload(outdir / "comp.zst")
        cat_of = _decode_comp_assign(meta, comp_arrays, bounds, xyz, rng)
        runs = meta["runs"]
        for j, run in enumerate(runs):
            idx = np.flatnonzero(cat_of == j)
            if not len(idx):
                continue
            h = _category_hist(code_hist, exact_bins, run)
            code_vals = common.sample_mass_from_hist(h, 0.0, 1.0, len(idx), rng)
            mass = _codes_to_mass(code_vals.astype(np.float64))
            out.append(np.column_stack([xyz[idx], mass]))

    return np.concatenate(out).astype(np.float32)
