# APT compression explainer

Interactive side-by-side comparison of 21 compression techniques across every
`.POS` file in the article data directory. The current corpus contains seven
files and 147 completed codec runs. The page includes a cross-dataset verdict,
aggregate table, per-dataset metrics, and synchronized original/compressed
point-cloud panels.

## Run

```bash
cd docs/compression-explainer
python3 -m http.server 8791
# open http://localhost:8791
```

## Regenerate data

```bash
experiments/run_all_datasets.sh                   # benchmark all codecs/files
.venv/bin/python experiments/export_explainer.py # rebuild data/ payloads
```

`data/<dataset>/*.bin` are decoded render samples (all stored atoms in sparse
mass bins, capped dense bins). Per-dataset `meta.json` files carry metrics,
spectra, and species tables; `data/summary.json` carries the cross-dataset
quality gate and aggregate winner. Benchmarks and artifacts live under
`experiments/artifacts/bench_<dataset>/`.

When either comparison panel uses a `hybrid_` codec, a mass-aligned storage
band appears directly below the spectrum. Green intervals are individually
stored exact points, purple intervals are distribution-modeled, and amber
intervals are original-mass point samples. The final mass-band hybrids can
make the exact/model decision inside the broad unranged population as well as
per ranged species. Hovering a 0.05 Da spectrum bin reports its assignment,
including a percentage split when a boundary crosses the bin. Refresh metrics
and hybrid/ranging metadata without rebuilding the large point payloads with:

```bash
.venv/bin/python experiments/export_explainer.py --metadata-only
```

The current winner is `hybrid_massbands`: exact bit-packed points for ranged
species below 100k atoms and for rare mass/color bands hidden inside
`unranged`; only frequent categories use smoothly interpolated 1 nm density
fields. Its median artifact is 2.08 MB (58.8:1); worst-case rare-element and
mass/color projection correlations are 0.9979 and 0.9980, respectively.

The 3D panels use the same camera conventions as `uap-archive`: the APT depth
axis starts left-to-right, left-drag orbits with a fixed z-up axis, right-drag
pans in screen space, middle-drag or the wheel dollies, and double-click resets
the fitted camera. Both comparison panels share one synchronized camera.
