"""Export decoded payloads + metrics for the interactive explainer (v2).

Per dataset writes docs/compression-explainer/data/<slug>/:
  <method>.bin   mass-bin-sorted point records for range selection
  meta.json      metrics for every benchmarked method, ranging, species table
and a top-level data/index.json listing datasets.

Binary layout (little-endian):
  u32 magic 'APT2' (0x32545041), u32 nbins, f32 bin_w, u32 n_records
  u32 true_counts[nbins]     decoded spectrum at bin_w (uncapped)
  u32 stored_counts[nbins]   records actually stored per bin (capped)
  records sorted by bin: x,y,z u16 (bounds-normalized), massQ u16 (mass/MAX)

Per-bin cap keeps every point of sparse (rare-species) bins while bounding
dense majority peaks, so "select a peak" can show all its atoms up to the
render budget.
"""

from __future__ import annotations

import argparse
import json
import statistics
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import common
import codecs_adaptive
import codecs_grid
import codecs_points

SPEC_BIN = 0.05
SPEC_MAX = common.SPECTRUM_MAX_DA
NBINS = int(round(SPEC_MAX / SPEC_BIN))
BIN_CAP = 300_000
MAGIC = 0x32545041

DOCS = common.REPO / "docs" / "compression-explainer"
DATA = DOCS / "data"

# methods that get a browsable payload on secondary datasets (all methods get
# one on the primary dataset); metrics are exported for every method regardless
PAYLOAD_SUBSET = [
    "subsample10", "massrange64", "grid_global", "grid_nmf", "grid_direct",
    "hybrid_exact", "hybrid_ultra", "hybrid_massbands", "hybrid_massbands_hifi",
    "hybrid_adaptive", "hybrid_adaptive_shared", "hybrid_adaptive_auto",
    "hybrid_adaptive4", "hybrid_binsort", "qpoint_synth", "qpoint_band",
]
PRIMARY_SLUG = "86a2fa56-8593-4856-bd42-b73716197abf"

QUALITY_GATE = {
    # Projection correlation catches wires, layers, and precipitates that a
    # globally correct composition can still wash out spatially.
    "worst_rare_min_proj_corr": 0.99,
    "median_density_corr_2nm": 0.995,
    "worst_spatial_species_error": 0.015,
    # Mass/color must remain attached to spatial features, including within
    # the otherwise undifferentiated "unranged" population.
    "worst_mass_band_min_proj_corr": 0.99,
    # Sub-2nm spatial fidelity (edges, striations, poles): the 2nm metric
    # saturates near 0.996 for every grid codec while codecs differ visibly.
    "median_density_corr_1nm": 0.995,
    "worst_density_corr_1nm": 0.99,
}


def pos_files() -> list[Path]:
    """All POS inputs in the article data directory, case-insensitively."""
    return sorted(
        (p for p in common.DATA_DIR.iterdir()
         if p.is_file() and p.suffix.lower() == ".pos"),
        key=lambda p: (common.dataset_slug(p) != PRIMARY_SLUG, p.name.lower()),
    )


def _numbers(rows: list[dict], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)) and np.isfinite(value):
            values.append(float(value))
    return values


def build_cross_dataset_summary(paths: list[Path]) -> dict:
    """Aggregate the common codec runs and choose the smallest quality-gated one."""
    from run_bench import METHODS

    by_method: dict[str, list[dict]] = {}
    labels: dict[str, str] = {}
    dataset_slugs = []
    for pos_path in paths:
        slug = common.dataset_slug(pos_path)
        results_path = common.ART_DIR / f"bench_{slug}" / "results.json"
        if not results_path.exists():
            continue
        dataset_slugs.append(slug)
        for row in json.loads(results_path.read_text()):
            if (row.get("error") or not row.get("method")
                    or row.get("method") not in METHODS):
                continue
            by_method.setdefault(row["method"], []).append(row)
            labels[row["method"]] = row.get("label", row["method"])

    def median(rows: list[dict], key: str):
        values = _numbers(rows, key)
        return statistics.median(values) if values else None

    def worst_high(rows: list[dict], key: str):
        values = _numbers(rows, key)
        return min(values) if values else None

    def worst_low(rows: list[dict], key: str):
        values = _numbers(rows, key)
        return max(values) if values else None

    methods = []
    for method, rows in by_method.items():
        rare_proj = _numbers(rows, "rare_min_proj_corr")
        mass_band_proj = _numbers(rows, "mass_band_min_proj_corr")
        entry = {
            "id": method,
            "label": labels[method],
            "dataset_count": len(rows),
            "rare_dataset_count": len(rare_proj),
            "mass_band_dataset_count": len(mass_band_proj),
            "median_size_bytes": median(rows, "size_bytes"),
            "max_size_bytes": worst_low(rows, "size_bytes"),
            "median_ratio_vs_raw": median(rows, "ratio_vs_raw"),
            "median_spectrum_tv": median(rows, "spectrum_tv"),
            "worst_spectrum_tv": worst_low(rows, "spectrum_tv"),
            "median_density_corr_2nm": median(rows, "density_corr_2nm"),
            "worst_density_corr_2nm": worst_high(rows, "density_corr_2nm"),
            "median_density_corr_1nm": median(rows, "density_corr_1nm"),
            "worst_density_corr_1nm": worst_high(rows, "density_corr_1nm"),
            "median_density_tv_2nm": median(rows, "density_tv_2nm"),
            "worst_density_tv_2nm": worst_low(rows, "density_tv_2nm"),
            "median_spatial_species_error": median(rows, "spatial_species_error"),
            "worst_spatial_species_error": worst_low(rows, "spatial_species_error"),
            "median_rare_min_density_corr": median(rows, "rare_min_density_corr"),
            "worst_rare_min_density_corr": worst_high(rows, "rare_min_density_corr"),
            "median_rare_min_proj_corr": median(rows, "rare_min_proj_corr"),
            "worst_rare_min_proj_corr": worst_high(rows, "rare_min_proj_corr"),
            "median_mass_band_min_proj_corr": median(
                rows, "mass_band_min_proj_corr"),
            "worst_mass_band_min_proj_corr": worst_high(
                rows, "mass_band_min_proj_corr"),
            "median_encode_seconds": median(rows, "encode_seconds"),
            "median_decode_seconds": median(rows, "decode_seconds"),
        }
        entry["meets_quality_gate"] = bool(
            entry["dataset_count"] == len(dataset_slugs)
            and entry["rare_dataset_count"] > 0
            and entry["mass_band_dataset_count"] == len(dataset_slugs)
            and entry["worst_rare_min_proj_corr"] >= QUALITY_GATE["worst_rare_min_proj_corr"]
            and entry["median_density_corr_2nm"] >= QUALITY_GATE["median_density_corr_2nm"]
            and entry["worst_spatial_species_error"] <= QUALITY_GATE["worst_spatial_species_error"]
            and entry["worst_mass_band_min_proj_corr"] >= QUALITY_GATE["worst_mass_band_min_proj_corr"]
            and (entry["median_density_corr_1nm"] or 0.0) >= QUALITY_GATE["median_density_corr_1nm"]
            and (entry["worst_density_corr_1nm"] or 0.0) >= QUALITY_GATE["worst_density_corr_1nm"]
        )
        # Avoid noisy float tails in the browser payload.
        for key, value in list(entry.items()):
            if isinstance(value, float):
                entry[key] = round(value, 6)
        methods.append(entry)

    methods.sort(key=lambda m: (m["median_size_bytes"] or float("inf"), m["id"]))
    candidates = [m for m in methods if m["meets_quality_gate"]]
    winner = min(candidates, key=lambda m: m["median_size_bytes"]) if candidates else None
    return {
        "dataset_count": len(dataset_slugs),
        "dataset_slugs": dataset_slugs,
        "method_count": len(methods),
        "benchmark_run_count": sum(m["dataset_count"] for m in methods),
        "quality_gate": QUALITY_GATE,
        "winner_id": winner["id"] if winner else None,
        "methods": methods,
    }


def pack_v2(pts: np.ndarray, bounds: np.ndarray, rng: np.random.Generator) -> bytes:
    mass = np.clip(pts[:, 3], 0.0, SPEC_MAX - 1e-4)
    bins = np.minimum((mass / SPEC_BIN).astype(np.int64), NBINS - 1)
    true_counts = np.bincount(bins, minlength=NBINS).astype(np.uint32)

    keep_idx = []
    order = np.argsort(bins, kind="stable")
    sorted_bins = bins[order]
    starts = np.searchsorted(sorted_bins, np.arange(NBINS))
    ends = np.searchsorted(sorted_bins, np.arange(NBINS), side="right")
    for b in range(NBINS):
        s, e = starts[b], ends[b]
        n = e - s
        if n == 0:
            continue
        if n <= BIN_CAP:
            keep_idx.append(order[s:e])
        else:
            sel = rng.choice(n, BIN_CAP, replace=False)
            sel.sort()
            keep_idx.append(order[s:e][sel])
    idx = np.concatenate(keep_idx) if keep_idx else np.zeros(0, dtype=np.int64)
    stored_counts = np.bincount(bins[idx], minlength=NBINS).astype(np.uint32)

    sub = pts[idx]
    ext = np.maximum(bounds[1] - bounds[0], 1e-9)
    rec = np.zeros((len(sub), 4), dtype=np.uint16)
    rec[:, :3] = np.round(
        (sub[:, :3] - bounds[0]) / ext * 65535.0).clip(0, 65535).astype(np.uint16)
    rec[:, 3] = np.round(
        np.clip(sub[:, 3], 0, SPEC_MAX) / SPEC_MAX * 65535.0).astype(np.uint16)

    head = struct.pack("<IIfI", MAGIC, NBINS, SPEC_BIN, len(sub))
    return head + true_counts.tobytes() + stored_counts.tobytes() + rec.tobytes()


def decoder_for(name: str):
    from run_bench import METHODS

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


def hybrid_representation(bench: Path, name: str, config: dict) -> dict | None:
    """Describe the per-species exact/distribution split for the viewer."""
    if not name.startswith("hybrid_"):
        return None
    manifest_path = bench / name / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text())
    params = config.get("params", {})
    exact_ids = [int(s) for s in manifest.get("exact_ids", [])]
    model_ids = [int(s) for s in manifest.get("model_ids", [])]
    if not model_ids:
        model_ids = [int(s) for s in manifest.get("sampled_ids", [])]
    counts = manifest.get("species_counts", {})
    exact_band_index = manifest.get("exact_band_index", [])
    exact_band_count = sum(int(item.get("count", 0)) for item in exact_band_index)
    rare_threshold = int(manifest.get(
        "rare_threshold", params.get("rare_threshold", 100_000)))
    representation = {
        "kind": "hybrid",
        "decision_unit": "species_label",
        "selection_mode": manifest.get(
            "exact_species", params.get("exact_species", "rare")),
        "rare_threshold": rare_threshold,
        "exact_species_ids": exact_ids,
        "modeled_species_ids": model_ids,
        "exact_ion_count": (
            sum(int(counts.get(str(s), 0)) for s in exact_ids)
            + exact_band_count),
        "modeled_ion_count": (
            sum(int(counts.get(str(s), 0)) for s in model_ids)
            - exact_band_count),
    }
    mass_band_count = int(manifest.get("unranged_mass_bands", 0))
    if mass_band_count:
        display_max = float(manifest["category_display_mass_max"])
        representation.update({
            "decision_unit": "species_label_and_unranged_mass_band",
            "selection_mode": "rare_species_and_rare_unranged_mass_bands",
            "modeled_mode": "mass_band_distribution",
            "unranged_mass_band_count": mass_band_count,
            "mass_band_display_max_da": display_max,
            "exact_unranged_mass_bands": [
                {
                    "band": int(item["band"]),
                    "lo": display_max * int(item["band"]) / mass_band_count,
                    "hi": (SPEC_MAX if int(item["band"]) == mass_band_count - 1
                           else display_max * (int(item["band"]) + 1)
                           / mass_band_count),
                    "count": int(item["count"]),
                }
                for item in exact_band_index
            ],
            "modeled_categories": manifest.get("model_categories", []),
        })
    guide_counts = manifest.get("guide_stored_counts", {})
    if guide_counts:
        guide_scale = float(manifest.get("guide_scale", 0.0))
        fixed_mix = float(manifest.get("guide_mix", 0.0))
        guide_species = {}
        for s in model_ids:
            total = int(counts.get(str(s), 0))
            stored = int(guide_counts.get(str(s), 0))
            fraction = stored / max(total, 1)
            mix = min(1.0, guide_scale * fraction) if guide_scale > 0 else fixed_mix
            guide_species[str(s)] = {
                "stored_points": stored,
                "sample_fraction": round(fraction, 6),
                "decode_mix": round(mix, 6),
            }
        representation.update({
            "modeled_mode": "grid_guided",
            "guide_cap": manifest.get("guide_cap"),
            "guide_fraction": manifest.get("guide_fraction", 0.0),
            "guide_scale": guide_scale,
            "guide_species": guide_species,
        })
    elif manifest.get("codec") == "hybrid_sample":
        representation.update({
            "modeled_mode": "sampled_points",
            "sample_cap": manifest.get("sample_cap"),
            "sample_fraction": manifest.get("fraction"),
            "stored_counts": manifest.get("stored_counts", {}),
        })
    elif "modeled_mode" not in representation:
        representation["modeled_mode"] = "distribution"
    return representation


def export_dataset(pos_path: Path) -> dict | None:
    from run_bench import METHODS

    slug = common.dataset_slug(pos_path)
    bench = common.ART_DIR / f"bench_{slug}"
    results_path = bench / "results.json"
    if not results_path.exists():
        print(f"skip {slug}: no results")
        return None
    results = json.loads(results_path.read_text())
    ranging = common.load_ranging(pos_path=pos_path)
    pts = common.load_pos(pos_path)
    species = ranging.assign(pts[:, 3])
    bounds = np.stack([pts[:, :3].min(axis=0), pts[:, :3].max(axis=0)])
    outdir = DATA / slug
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)

    payload_methods = (list(METHODS) if slug == PRIMARY_SLUG else PAYLOAD_SUBSET)

    (outdir / "original.bin").write_bytes(pack_v2(pts, bounds, rng))
    print(f"[{slug}] original done")

    methods_meta = [{
        "id": "original", "label": "Original (full raw file)",
        "size_bytes": pos_path.stat().st_size, "has_payload": True, "metrics": {},
    }]
    # rare set for reporting: share < 2% with at least 3000 atoms (broader than
    # the bench's 1% summary; derived from per-element metrics, no re-run needed)
    n_atoms = len(pts)
    el_counts = {}
    for s, el in enumerate(ranging.elements):
        if el:
            el_counts[el] = el_counts.get(el, 0) + int(
                (species == s).sum())
    rare_els = [el for el, c in el_counts.items()
                if c / n_atoms < 0.02 and c >= 3000]

    for row in results:
        name = row.get("method")
        if name not in METHODS:
            continue
        metrics = {k: v for k, v in row.items()
                   if k not in ("method", "label") and isinstance(v, (int, float))}
        dens = [metrics.get(f"{el}_density_corr") for el in rare_els]
        dens = [v for v in dens if v is not None]
        if dens:
            metrics["rare_min_density_corr"] = min(dens)
        projs = [metrics.get(f"{el}_proj_corr_{ax}")
                 for el in rare_els for ax in ("x", "y", "z")]
        projs = [v for v in projs if v is not None]
        if projs:
            metrics["rare_min_proj_corr"] = min(projs)
        entry = {
            "id": name, "label": METHODS[name]["label"],
            "size_bytes": row.get("size_bytes", 0),
            "has_payload": False,
            "metrics": metrics,
        }
        representation = hybrid_representation(bench, name, METHODS[name])
        if representation:
            entry["representation"] = representation
        if "error" in row:
            entry["error"] = row["error"]
        elif name in payload_methods:
            decoded = decoder_for(name)(bench / name, np.random.default_rng(1234))
            (outdir / f"{name}.bin").write_bytes(pack_v2(decoded, bounds, rng))
            entry["has_payload"] = True
            del decoded
            print(f"[{slug}] {name} done")
        methods_meta.append(entry)

    counts = np.bincount(species, minlength=len(ranging.labels))
    species_table = []
    for s in range(len(ranging.labels)):
        wins = [[float(ranging.lo[w]), float(ranging.hi[w])]
                for w in range(len(ranging.lo))
                if ranging.species_of_window[w] == s]
        species_table.append({
            "label": ranging.labels[s], "element": ranging.elements[s],
            "count": int(counts[s]), "windows": wins,
        })
    display_max = min(SPEC_MAX, float(np.ceil(ranging.hi.max() + 12)) if len(ranging.hi) else SPEC_MAX)

    meta = {
        "slug": slug,
        "pos_file": pos_path.name,
        "raw_size_bytes": pos_path.stat().st_size,
        "atom_count": int(len(pts)),
        "bounds": bounds.tolist(),
        "spectrum_bin_da": SPEC_BIN,
        "spectrum_max_da": SPEC_MAX,
        "display_max_da": display_max,
        "rare_elements": rare_els,
        "ranging_windows": [
            {
                "lo": float(ranging.lo[w]),
                "hi": float(ranging.hi[w]),
                "species_id": int(ranging.species_of_window[w]),
            }
            for w in range(len(ranging.lo))
        ],
        "species": species_table,
        "methods": methods_meta,
    }
    (outdir / "meta.json").write_text(json.dumps(meta))
    ref_row = next((m for m in methods_meta if m["id"] == "hybrid_exact"), None)
    return {
        "slug": slug, "name": pos_path.name, "atom_count": int(len(pts)),
        "raw_mb": round(pos_path.stat().st_size / 1e6, 1),
        "hybrid_exact_mb": round(ref_row["size_bytes"] / 1e6, 2) if ref_row else None,
        "rare_min_proj_corr": (ref_row["metrics"].get("rare_min_proj_corr")
                               if ref_row else None),
    }


def refresh_existing_metadata(paths: list[Path]) -> int:
    """Sync metrics/representations without rebuilding large point payloads."""
    from run_bench import METHODS

    updated = 0
    for pos_path in paths:
        slug = common.dataset_slug(pos_path)
        meta_path = DATA / slug / "meta.json"
        if not meta_path.exists():
            print(f"skip {slug}: no exported meta.json")
            continue
        meta = json.loads(meta_path.read_text())
        ranging = common.load_ranging(pos_path=pos_path)
        meta["ranging_windows"] = [
            {
                "lo": float(ranging.lo[w]),
                "hi": float(ranging.hi[w]),
                "species_id": int(ranging.species_of_window[w]),
            }
            for w in range(len(ranging.lo))
        ]
        bench = common.ART_DIR / f"bench_{slug}"
        results_path = bench / "results.json"
        if not results_path.exists():
            print(f"skip {slug}: no benchmark results")
            continue
        results = {
            row.get("method"): row
            for row in json.loads(results_path.read_text())
            if row.get("method") and not row.get("error")
        }
        original = next(
            (entry for entry in meta.get("methods", [])
             if entry.get("id") == "original"),
            {
                "id": "original", "label": "Original (full raw file)",
                "size_bytes": pos_path.stat().st_size,
                "has_payload": (DATA / slug / "original.bin").exists(),
                "metrics": {},
            },
        )
        synced = [original]
        for name, config in METHODS.items():
            row = results.get(name)
            if not row:
                continue
            metrics = {
                key: value for key, value in row.items()
                if key not in ("method", "label")
                and isinstance(value, (int, float))
            }
            method = {
                "id": name,
                "label": config["label"],
                "size_bytes": row.get("size_bytes", 0),
                "has_payload": (DATA / slug / f"{name}.bin").exists(),
                "metrics": metrics,
            }
            representation = hybrid_representation(
                bench, name, config)
            if representation:
                method["representation"] = representation
            synced.append(method)
        meta["methods"] = synced
        meta_path.write_text(json.dumps(meta))
        updated += 1
        print(f"[{slug}] metadata refreshed")
    return updated


def export_selected_payloads(paths: list[Path], method_names: list[str]) -> int:
    """Export only named decoded payloads and merge their metadata in place."""
    from run_bench import METHODS

    unknown = [name for name in method_names if name not in METHODS]
    if unknown:
        raise ValueError(f"unknown methods: {', '.join(unknown)}")
    written = 0
    for pos_path in paths:
        slug = common.dataset_slug(pos_path)
        outdir = DATA / slug
        meta_path = outdir / "meta.json"
        bench = common.ART_DIR / f"bench_{slug}"
        results_path = bench / "results.json"
        if not meta_path.exists() or not results_path.exists():
            print(f"skip {slug}: existing metadata/results required")
            continue
        meta = json.loads(meta_path.read_text())
        results = {
            row.get("method"): row for row in json.loads(results_path.read_text())
            if row.get("method")
        }
        entries = {entry["id"]: entry for entry in meta.get("methods", [])}
        bounds = np.asarray(meta["bounds"], dtype=np.float32)
        rng = np.random.default_rng(7)
        for name in method_names:
            row = results.get(name)
            if not row or row.get("error"):
                print(f"skip {slug}/{name}: no successful benchmark")
                continue
            decoded = decoder_for(name)(bench / name, np.random.default_rng(1234))
            (outdir / f"{name}.bin").write_bytes(pack_v2(decoded, bounds, rng))
            del decoded
            metrics = {
                key: value for key, value in row.items()
                if key not in ("method", "label") and isinstance(value, (int, float))
            }
            entry = {
                "id": name,
                "label": METHODS[name]["label"],
                "size_bytes": row.get("size_bytes", 0),
                "has_payload": True,
                "metrics": metrics,
            }
            representation = hybrid_representation(bench, name, METHODS[name])
            if representation:
                entry["representation"] = representation
            entries[name] = entry
            written += 1
            print(f"[{slug}] {name} payload done")
        # Preserve the established method order, then append newly introduced
        # methods in the order requested on the command line.
        ordered_ids = [entry["id"] for entry in meta.get("methods", [])]
        ordered_ids.extend(name for name in method_names if name not in ordered_ids)
        meta["methods"] = [entries[name] for name in ordered_ids if name in entries]
        meta_path.write_text(json.dumps(meta))
    return written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("slugs", nargs="*", help="optional dataset slugs")
    parser.add_argument(
        "--metadata-only", action="store_true",
        help="refresh hybrid/ranging metadata without rebuilding point payloads",
    )
    parser.add_argument(
        "--payload-only", metavar="METHODS",
        help="comma-separated methods to export into existing dataset metadata",
    )
    args = parser.parse_args()
    only = set(args.slugs) if args.slugs else None
    DATA.mkdir(parents=True, exist_ok=True)
    paths = pos_files()
    selected = [p for p in paths if not only or common.dataset_slug(p) in only]
    if args.metadata_only:
        updated = refresh_existing_metadata(selected)
        summary = build_cross_dataset_summary(paths)
        (DATA / "summary.json").write_text(json.dumps(summary))
        print(f"refreshed {updated} dataset metadata files")
        print(f"cross-dataset winner: {summary.get('winner_id') or 'none'}")
        return
    if args.payload_only:
        names = [name for name in args.payload_only.split(",") if name]
        written = export_selected_payloads(selected, names)
        print(f"exported {written} selected method payloads")
        return

    index = []
    for pos in selected:
        entry = export_dataset(pos)
        if entry:
            index.append(entry)
    (DATA / "index.json").write_text(json.dumps(index))
    summary = build_cross_dataset_summary(paths)
    (DATA / "summary.json").write_text(json.dumps(summary))
    total = sum(f.stat().st_size for f in DATA.rglob("*") if f.is_file())
    winner = summary.get("winner_id") or "none"
    print(f"wrote {DATA} ({total/1e9:.2f} GB); "
          f"cross-dataset winner: {winner}")


if __name__ == "__main__":
    main()
