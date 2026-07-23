"""Re-evaluate existing codec artifacts after benchmark metrics change.

This avoids paying the encode cost again when a new diagnostic is added.

Usage:
  .venv/bin/python experiments/rescore.py \
    --methods hybrid_exact,hybrid_hifi,hybrid_ultra,qpoint_synth,qpoint_mass
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import common
import codecs_adaptive
import codecs_grid
import codecs_points
from run_bench import METHODS


def decoder_for(name: str):
    cfg = METHODS[name]
    if cfg["kind"] == "points":
        if cfg["fn"] == "subsample":
            return codecs_points.decode_subsample
        if cfg["fn"] == "hybrid_sample":
            return codecs_points.decode_hybrid_sample
        return codecs_points.decode_qpoint
    if cfg["kind"] == "massrange":
        return codecs_grid.decode_massrange
    if cfg["kind"] == "adaptive":
        return codecs_adaptive.decode_adaptive
    if cfg["kind"] == "binsort":
        import codecs_binsort

        return codecs_binsort.decode_binsort
    return codecs_grid.decode_grid


def pos_files() -> list[Path]:
    return sorted(
        p for p in common.DATA_DIR.iterdir()
        if p.is_file() and p.suffix.lower() == ".pos"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", required=True)
    ap.add_argument("--pos", action="append", default=[])
    ap.add_argument(
        "--only-missing", action="store_true",
        help="skip rows that already contain mass_band_min_proj_corr",
    )
    args = ap.parse_args()
    names = args.methods.split(",")
    for name in names:
        if name not in METHODS:
            raise ValueError(f"unknown method {name}")

    paths = [Path(p) for p in args.pos] if args.pos else pos_files()
    for pos_path in paths:
        slug = common.dataset_slug(pos_path)
        bench = common.ART_DIR / f"bench_{slug}"
        results_path = bench / "results.json"
        if not results_path.exists():
            print(f"[{slug}] skip: no results")
            continue
        rows = json.loads(results_path.read_text())
        by_name = {row.get("method"): row for row in rows}
        pending = [
            name for name in names
            if name in by_name
            and not by_name[name].get("error")
            and (bench / name).exists()
            and not (
                args.only_missing
                and "mass_band_min_proj_corr" in by_name[name]
            )
        ]
        if not pending:
            print(f"[{slug}] nothing to rescore")
            continue

        pts = common.load_pos(pos_path)
        ranging = common.load_ranging(pos_path=pos_path)
        print(f"[{slug}] building reference on {len(pts):,} atoms")
        ref = common.Reference.build(pts, ranging)
        for name in pending:
            print(f"[{slug}] rescoring {name}")
            decoded = decoder_for(name)(
                bench / name, np.random.default_rng(1234))
            metrics = ref.evaluate(decoded)
            for key, value in metrics.items():
                by_name[name][key] = (
                    round(value, 5) if isinstance(value, float) else value
                )
            del decoded
            results_path.write_text(json.dumps(rows, indent=2))
        print(f"[{slug}] wrote {results_path}")


if __name__ == "__main__":
    main()
