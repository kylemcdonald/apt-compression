# CPOS 4M research codec

This branch tests a lossy successor to CPOS 1.0 for dense previews:

1. quantize `x`, `y`, `z`, and mass-to-charge ratio to 12 bits;
2. keep at most 4,000,000 exact quantized seeds;
3. encode those seeds with the sorted, lossless 12-bit transform;
4. decode the seeds exactly and deterministically synthesize records until the
   original point count is restored.

The experimental container uses the `.cp4m` extension so it cannot be confused
with released `.cpos` files. It has its own versioned header and checksum. CPOS
1.0 and the production GitHub Pages site remain unchanged on this branch.

## Mass-aware allocation

The source spectrum is divided into 0.1 Da bins over 0–300 Da. If bin `i`
contains `c_i` points, its uncapped quota is proportional to:

```text
c_i ^ 0.75
```

The allocator caps every quota at the source count, assigns at least one seed
to every active bin, and distributes exactly four million seeds when the input
is larger than that. This is sublinear: large peaks still receive more seeds
in absolute terms, while small peaks receive a much higher retention rate.

Within each mass bin, points are sorted by their 12-bit 3D Morton key and
sampled at evenly spaced ranks. The retained tuples are then encoded without
further loss using per-bin Rice-coded spatial gaps, a 12-bit mass bitplane
stream, and an outer Deflate stream.

The file also stores exact source and retained histograms. Expansion therefore
restores the original point count and the exact 0.1 Da histogram; it does not
guess the original acquisition order or recover discarded spatial detail.

## Dither

Synthesized records use deterministic uniform noise over ±0.5 of one 12-bit
quantization cell by default. This is dequantization dither: under the usual
round-to-nearest model, every value in that interval is equally plausible.
Exact retained seeds are never dithered and remain explicitly marked as exact.

`none` and a clipped Gaussian option are available for comparison. Uniform is
the default because it improves voxel occupancy and axis-distribution error on
the representative benchmark without implying a Gaussian sub-cell
distribution that the file cannot support.

## Python

```python
from cpos.io import read_pos
from experiments.lossy4m import decode, decode_retained, encode

points = read_pos("input.pos")
payload = encode(points)

retained = decode_retained(payload)
assert retained.header.stored_point_count <= 4_000_000

expanded = decode(payload, noise="uniform")
assert len(expanded.points) == len(points)
assert expanded.exact.sum() == retained.header.stored_point_count
```

`expanded.exact` identifies retained versus synthesized records and
`expanded.bins` identifies the structural mass bin of every record.

The research CLI can encode, inspect, and decode files:

```bash
python3 -m experiments.lossy4m.cli encode input.pos preview.cp4m
python3 -m experiments.lossy4m.cli inspect preview.cp4m
python3 -m experiments.lossy4m.cli decode preview.cp4m expanded.pos
```

## JavaScript and visualizer

[`javascript/cp4m.js`](javascript/cp4m.js) is a dependency-free ES module for
inspecting, decoding, and expanding `.cp4m` in browsers and Node. Rendering is
kept separately in [`demo/renderer.js`](demo/renderer.js).

Build the bundled public Ck10 example and a standalone static site:

```bash
python3 -m experiments.lossy4m.build_demo \
  --site experiments/lossy4m/site
python3 -m http.server 8000 -d experiments/lossy4m/site
```

The visualizer shows CPOS 1.0 retained points, CP4M exact seeds, and the CP4M
expanded cloud side by side. Exact and synthesized records use separate
colors. The spectrum draws original, exact, and synthesized counts, with a
bin-level exact/synthesized strip and per-bin hover details.

## Validation and benchmark

```bash
python3 -m pytest -q experiments/lossy4m/test_codec.py

python3 -m experiments.lossy4m.benchmark \
  --output experiments/lossy4m/results/full.json

python3 -m experiments.lossy4m.report \
  experiments/lossy4m/results/full.json \
  --output experiments/lossy4m/RESULTS.md
```

The tests verify Python round trips, exact retained 12-bit tuples, exact
expanded point counts and spectra, checksum/version rejection, allocation
invariants, and JavaScript decoder parity. The benchmark fully expands every
file, validates provenance and histogram counts, and compares size and
distribution metrics with CPOS 1.0.

The source `.POS` files are read from the two local public/reference corpora
and are never committed:

- `/Users/kyle/Documents/GitHub/uap/rangefinder/controls`
- `/Users/kyle/Documents/GitHub/uap/apt-analysis/data`

See [FORMAT.md](FORMAT.md) for the binary layout,
[TUNING.md](TUNING.md) for the allocation/bin-width/dither experiments, and
[RESULTS.md](RESULTS.md) for the completed 18-file comparison.
