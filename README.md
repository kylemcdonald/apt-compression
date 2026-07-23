# APT POS Compression Prototype

Local research prototype for viewing and comparing compressed Atom Probe Tomography `.pos` files.

This repository contains source code only. Raw `.pos` files, generated artifacts, built frontend assets, local virtualenvs, and generated WASM binaries are intentionally ignored.

## Setup

```bash
python3 -m pip install -r requirements.txt
npm install
npm run build:wasm
```

Put or symlink `.pos` files into `data/`. If `data/` is empty, preprocessing can generate a synthetic dataset.

## Run

Quick local smoke test:

```bash
python3 run.py --preprocess --quick --synthetic
```

Full preprocessing and viewer:

```bash
python3 preprocess.py
python3 run.py
```

The runner starts the Python API server, starts Vite, and prints the viewer URL.

## Current Primary Codec

The current primary lossy representation is:

- 64 adaptive mass/charge ranges
- linear count-density CDF grid
- adaptive exact per-bin support masks under a support-byte budget
- sparse base-excess residual cells for localized high-density spots
- browser-side decoding with WebGPU and WASM/JS fallbacks

The full/raw method reads the original `.pos` file through the local API. It does not create a canonical converted copy for full mode.

## Compression experiments (July 2026)

`experiments/` contains a self-contained benchmark of 21 codec variants
(point packing, mass-range grids, k-means/NMF/PCA/ICA composition models,
rare-species-exact and exact/sample hybrids, mass-conditioned hybrids,
wavelet, hash-grid INR, hypernetwork) with spectrum, density, composition,
rare-element, and mass/color spatial metrics. All 21 methods have
been evaluated on all seven `.POS` files currently present in the article data
directory (147 successful runs). Per-dataset results live under
`experiments/artifacts/bench_<dataset>/results.json`; the interactive
side-by-side and cross-dataset viewer is in `docs/compression-explainer/`.

The best balanced method is `hybrid_massbands`: store every ranged species
below 100k atoms as exact bit-packed points, split the broad `unranged` bucket
into the viewer's six mass/color bands, and also store any such band below
100k exactly. Only frequent species/bands are density-modeled on a 1 nm grid.
It is the smallest method that passes the cross-dataset fidelity gate, with a
2.08 MB median artifact (58.8:1 median compression), 0.9979 worst-case rare
structure correlation, and 0.9980 worst-case mass/color structure correlation.
Grid sampling uses a tent kernel, equivalent to piecewise-linear interpolation
between voxel centers, so decoded clouds do not expose grid-cell seams.
