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
