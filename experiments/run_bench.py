"""Benchmark runner: encode + decode + evaluate every codec on one POS file.

Usage:
  .venv/bin/python experiments/run_bench.py --methods all
  .venv/bin/python experiments/run_bench.py --methods hybrid_exact,inr_kmeans --limit 500000
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

import common
import codecs_adaptive
import codecs_grid
import codecs_points

METHODS: dict[str, dict] = {
    "subsample10": dict(
        kind="points", fn="subsample", params=dict(fraction=0.1),
        label="Random 10% subsample (current fallback baseline)"),
    "hybrid_sample5": dict(
        kind="points", fn="hybrid_sample",
        params=dict(fraction=0.05, kernel_sigma_nm=0.05),
        label="Rare species exact; abundant species represented by a 5% point sample"),
    "hybrid_sample10": dict(
        kind="points", fn="hybrid_sample",
        params=dict(fraction=0.10, kernel_sigma_nm=0.05),
        label="Rare species exact; abundant species represented by a 10% point sample"),
    "massrange64": dict(
        kind="massrange", params=dict(n_ranges=64, budget_mb=8.0),
        label="64 adaptive mass ranges x density grids (current-codec stand-in)"),
    "qpoint_mass": dict(
        kind="points", fn="qpoint", params=dict(bits=12, store_mass="u8"),
        label="All atoms, Morton delta + varint bit packing, per-atom mass byte"),
    "qpoint_synth": dict(
        kind="points", fn="qpoint", params=dict(bits=12, store_mass="synth"),
        label="All atoms, Morton delta + varint, masses synthesized from spectrum"),
    "qpoint_band": dict(
        kind="points", fn="qpoint",
        params=dict(bits=12, store_mass="band", aniso=True),
        label="All atoms, per-axis-bit Morton packing; per-atom color band kept "
              "for unranged atoms, masses synthesized within band"),
    "grid_global": dict(
        kind="grid", params=dict(backbone="u8", backbone_voxel=1.0, comp_model="global"),
        label="1nm density grid + single global composition (control)"),
    "grid_kmeans": dict(
        kind="grid", params=dict(backbone="u8", backbone_voxel=1.0,
                                 comp_model="kmeans", comp_cell=3.0, k=8),
        label="1nm density grid + k-means compositional clusters (K=8, 3nm cells)"),
    "grid_kmeans_resid": dict(
        kind="grid", params=dict(backbone="u8", backbone_voxel=1.0,
                                 comp_model="kmeans_resid", comp_cell=3.0, k=8),
        label="k-means clusters + coarse per-species deviation grids"),
    "grid_nmf": dict(
        kind="grid", params=dict(backbone="u8", backbone_voxel=1.0,
                                 comp_model="nmf", comp_cell=3.0, k=8),
        label="1nm density grid + NMF soft composition (k=8)"),
    "grid_pca": dict(
        kind="grid", params=dict(backbone="u8", backbone_voxel=1.0,
                                 comp_model="pca", comp_cell=3.0, k=8),
        label="1nm density grid + PCA composition (k=8, clipped)"),
    "grid_ica": dict(
        kind="grid", params=dict(backbone="u8", backbone_voxel=1.0,
                                 comp_model="ica", comp_cell=3.0, k=8),
        label="1nm density grid + ICA composition (k=8, clipped)"),
    "grid_direct": dict(
        kind="grid", params=dict(backbone="u8", backbone_voxel=1.0,
                                 comp_model="direct", comp_cell=3.0),
        label="1nm density grid + direct per-species fraction grids (3nm)"),
    "hybrid_exact": dict(
        kind="grid", params=dict(backbone="u8", backbone_voxel=1.0,
                                 comp_model="direct", comp_cell=3.0,
                                 exact_species="rare"),
        label="Rare species stored exactly; majority/background as 1nm grid"),
    "hybrid_hifi": dict(
        kind="grid", params=dict(backbone="u8", backbone_voxel=0.5,
                                 comp_model="direct", comp_cell=2.0,
                                 exact_species="rare"),
        label="Rare species exact; majority/background at 0.5nm grid"),
    "hybrid_ultra": dict(
        kind="grid", params=dict(backbone="u8", backbone_voxel=0.25,
                                 comp_model="direct", comp_cell=2.0,
                                 exact_species="rare"),
        label="Rare species exact; majority/background at 0.25nm grid"),
    "hybrid_massbands": dict(
        kind="grid", params=dict(backbone="u8", backbone_voxel=1.0,
                                 comp_model="direct", comp_cell=1.0,
                                 exact_species="rare",
                                 unranged_mass_bands=6),
        label="Rare species exact; abundant density conditioned on six mass/color bands"),
    "hybrid_massbands_hifi": dict(
        kind="grid", params=dict(backbone="u8", backbone_voxel=0.5,
                                 comp_model="direct", comp_cell=1.0,
                                 exact_species="rare",
                                 unranged_mass_bands=6),
        label="Rare species exact; 0.5nm density conditioned on six mass/color bands"),
    "hybrid_adaptive": dict(
        kind="adaptive", params=dict(target_mb=10.0),
        label="Per-species/color-band density fields at adaptive resolution "
              "(refined to 0.25nm where structured); rare species and rare "
              "color bands exact (~10 MB target)"),
    "hybrid_adaptive4": dict(
        kind="adaptive", params=dict(target_mb=4.0),
        label="Adaptive-resolution per-category density fields, ~4 MB target"),
    "hybrid_adaptive_shared": dict(
        kind="adaptive", params=dict(target_mb=10.0, shared=True),
        label="One shared adaptive-resolution density field + per-cell "
              "category fractions; texture common to all species/bands is "
              "stored once (~10 MB target)"),
    "hybrid_adaptive_auto": dict(
        kind="adaptive", params=dict(target_mb=10.0, shared=True,
                                     auto_range=True),
        label="Chemistry-agnostic hybrid_adaptive_shared: peak windows "
              "auto-detected from the spectrum, no external ranging table"),
    "hybrid_binsort": dict(
        kind="binsort", params=dict(target_mb=10.0),
        label="Spectrum bins sorted by occupancy: light/structured spectral "
              "chunks stored as exact atoms (mass implicit), heavy bins as one "
              "shared adaptive field with adaptive-resolution composition; "
              "no ranging at all (~10 MB target)"),
    "wavelet_kmeans": dict(
        kind="grid", params=dict(backbone="wavelet", backbone_voxel=1.0,
                                 comp_model="kmeans", comp_cell=3.0, k=8,
                                 wavelet_coeffs=400_000),
        label="Wavelet-coded 1nm density + k-means composition"),
    "inr_kmeans": dict(
        kind="grid", params=dict(backbone="inr", backbone_voxel=1.0,
                                 comp_model="kmeans", comp_cell=3.0, k=8,
                                 neural_kw=dict(log2T=15, steps=3000)),
        label="Hash-grid neural field density + k-means composition"),
    "hyper_kmeans": dict(
        kind="grid", params=dict(backbone="hyper", backbone_voxel=1.0,
                                 comp_model="kmeans", comp_cell=3.0, k=8,
                                 neural_kw=dict(steps=4000)),
        label="Per-slice hypernetwork density + k-means composition"),
}


def run_method(name: str, pts: np.ndarray, ranging, ref: common.Reference,
               outroot: Path, pos_path: Path) -> dict:
    cfg = METHODS[name]
    outdir = outroot / name
    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True)
    common.write_artifact_dataset(outdir, pos_path)
    rng = np.random.default_rng(1234)

    t0 = time.time()
    if cfg["kind"] == "points":
        if cfg["fn"] == "subsample":
            codecs_points.encode_subsample(pts, ranging, outdir, **cfg["params"])
            decode = codecs_points.decode_subsample
        elif cfg["fn"] == "hybrid_sample":
            codecs_points.encode_hybrid_sample(pts, ranging, outdir, **cfg["params"])
            decode = codecs_points.decode_hybrid_sample
        else:
            codecs_points.encode_qpoint(pts, ranging, outdir, **cfg["params"])
            decode = codecs_points.decode_qpoint
    elif cfg["kind"] == "massrange":
        codecs_grid.encode_massrange(pts, ranging, outdir, **cfg["params"])
        decode = codecs_grid.decode_massrange
    elif cfg["kind"] == "adaptive":
        codecs_adaptive.encode_adaptive(pts, ranging, outdir, **cfg["params"])
        decode = codecs_adaptive.decode_adaptive
    elif cfg["kind"] == "binsort":
        import codecs_binsort

        codecs_binsort.encode_binsort(pts, ranging, outdir, **cfg["params"])
        decode = codecs_binsort.decode_binsort
    else:
        codecs_grid.encode_grid(pts, ranging, outdir, **cfg["params"])
        decode = codecs_grid.decode_grid
    enc_s = time.time() - t0

    size = common.dir_size(outdir)
    t0 = time.time()
    decoded = decode(outdir, rng)
    dec_s = time.time() - t0
    metrics = ref.evaluate(decoded)
    row = common.result_row(name, size, enc_s, dec_s, metrics)
    row["ratio_vs_raw"] = round(pos_path.stat().st_size / max(size, 1), 1)
    row["label"] = cfg["label"]
    print(json.dumps(row, indent=None)[:400])
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="all")
    ap.add_argument("--pos", default=str(common.POS_PATH))
    ap.add_argument("--limit", type=int, default=0, help="use only first N atoms")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    names = list(METHODS) if args.methods == "all" else args.methods.split(",")
    for n in names:
        assert n in METHODS, f"unknown method {n}"

    pos_path = Path(args.pos)
    slug = common.dataset_slug(pos_path)
    pts = common.load_pos(pos_path)
    if args.limit:
        pts = pts[: args.limit]
    ranging = common.load_ranging(pos_path=pos_path)
    print(f"[{slug}] building reference on {len(pts):,} atoms ...")
    t0 = time.time()
    ref = common.Reference.build(pts, ranging)
    print(f"reference built in {time.time()-t0:.1f}s; elements {ref.elements}; "
          f"rare {ref.rare_elements}")

    suffix = f"_{args.limit}" if args.limit else ""
    outroot = common.ART_DIR / f"bench_{slug}{suffix}"
    results_path = Path(args.out) if args.out else outroot / "results.json"
    results = {}
    if results_path.exists():
        results = {r["method"]: r for r in json.loads(results_path.read_text())}

    for name in names:
        print(f"\n=== {slug} :: {name} ===")
        try:
            results[name] = run_method(name, pts, ranging, ref, outroot, pos_path)
        except Exception as e:
            import traceback

            traceback.print_exc()
            results[name] = {"method": name, "error": str(e)}
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.write_text(json.dumps(list(results.values()), indent=2))
    print(f"\nwrote {results_path}")


if __name__ == "__main__":
    main()
