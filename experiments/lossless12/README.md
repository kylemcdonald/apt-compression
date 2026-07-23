# Lossless 12-bit CPOS research

This branch investigates lossless compression of an assumed source
representation containing four unsigned 12-bit values per ion: `x`, `y`, `z`,
and mass-to-charge ratio.

"Lossless" here means exact recovery of those four quantized integers. It does
not mean exact recovery of the original four float32 values. Per-file float32
minimum and maximum values define the reference quantizer and require 32 bytes
of metadata.

The completed 18-file benchmark reaches 2.1247 bytes per point, or 7.53×
compression relative to float32 `.POS`. See [RESULTS.md](RESULTS.md) for the
aggregate, per-file results, limitations, and interpretation.

The benchmark distinguishes:

- **ordered** transforms, which preserve event order;
- **canonical-set** transforms, which sort points and preserve the exact
  multiset of 12-bit tuples but not acquisition order.

Run the unit tests:

```bash
python3 -m pytest experiments/lossless12/test_lossless12.py
```

Run a pilot benchmark:

```bash
python3 experiments/lossless12/benchmark.py \
  --preset pilot \
  --output experiments/lossless12/results/pilot.json
```

Run the selected full-corpus benchmark:

```bash
python3 experiments/lossless12/benchmark.py --help
python3 experiments/lossless12/refine.py --help
```

The committed `results/full_sweep.json`, `results/full_axes.json`, and
`results/full_backends.json` preserve every measured candidate used in the
report. The benchmark writes each file record immediately, so a long run can
be inspected even if it is interrupted.

Reference roots default to:

- `/Users/kyle/Documents/GitHub/uap/rangefinder/controls`
- `/Users/kyle/Documents/GitHub/uap/apt-analysis/data`
