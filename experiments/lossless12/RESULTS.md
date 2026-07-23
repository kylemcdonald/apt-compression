# Lossless 12-bit compression experiment

## Bottom line

Across 18 reference `.POS` files containing 117,917,065 points, the best
canonical-set encoding occupies **250,539,848 bytes (238.93 MiB)**:

- **2.1247 bytes / 16.998 bits per point**
- **7.53× smaller** than the 1,886,673,040-byte float32 `.POS` corpus
- **2.824× smaller** than fixed packing of four 12-bit values
- **86.72% less space** than float32 `.POS`
- **64.59% less space** than the 12-bit baseline

This result is lossless with respect to four unsigned 12-bit integers per
point. It is not lossless with respect to the source float32 values.

The winning transform sorts the points. It recovers the exact multiset,
including duplicates, but does not recover acquisition order. The best
order-preserving zstd-19 baseline is 551,399,626 bytes: 4.6762 bytes per point
and only 22.06% below fixed 12-bit packing.

| Representation | Total bytes | Bytes/point | Ratio vs `.POS` | Reduction vs 12-bit |
| --- | ---: | ---: | ---: | ---: |
| Source float32 `.POS` | 1,886,673,040 | 16.0000 | 1.00× | — |
| Fixed four-field 12-bit packing | 707,503,182 | 6.0000 | 2.67× | 0.00% |
| Best order-preserving + zstd-19 | 551,399,626 | 4.6762 | 3.42× | 22.06% |
| Best sorted canonical-set result | 250,539,848 | 2.1247 | 7.53× | 64.59% |

## Winning transform

The best result for each file uses the same adaptive family:

1. Quantize `x`, `y`, `z`, and mass-to-charge ratio to unsigned 12-bit values.
2. Select zero to twelve leading mass bits as group identifiers.
3. Map the three spatial values to a 36-bit Hilbert distance.
4. Sort each mass-prefix group by that distance.
5. Rice-code the spatial gaps with a parameter selected per group.
6. Store the remaining mass suffix bits as an aligned bitplane stream.
7. Compress the complete reversible stream with Brotli-11, xz-9e, or
   zstd-19, selecting the smallest result.

The benchmark exhaustively tested all 13 mass-prefix widths. It then tested
all six spatial-axis permutations for each winner. Brotli won 13 files and xz
won five.

## Per-file results

`Prefix` is the number of leading mass bits used to partition the points.
`Axes` is the Hilbert coordinate order.

| File | Points | Prefix | Axes | Backend | Bytes | B/point | Ratio vs `.POS` |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: |
| `control_Ck10_steel_felfer_R56_01769.pos` | 5,525,361 | 11 | xzy | Brotli-11 | 10,973,434 | 1.986 | 8.06× |
| `control_MoHf_leitner_R21_08680.pos` | 4,583,568 | 11 | yxz | xz-9e | 11,327,044 | 2.471 | 6.47× |
| `control_ODSsteel_wang_R31_06365.pos` | 4,868,202 | 11 | xyz | Brotli-11 | 10,762,051 | 2.211 | 7.24× |
| `control_Si_apav_usa_denton_smith.pos` | 945,211 | 10 | yxz | Brotli-11 | 2,188,086 | 2.315 | 6.91× |
| `steelHD_5534859_70a59eff-003c-4337-832a-604c260dc623.POS` | 122,023 | 0 | zyx | Brotli-11 | 418,665 | 3.431 | 4.66× |
| `steelHD_5534859_8fe936a9-c8bf-417a-8006-bff810375bc7.POS` | 228,176 | 2 | yxz | Brotli-11 | 726,725 | 3.185 | 5.02× |
| `steelHD_5534859_aa137df8-ce0e-481e-93e6-a22cb5a34882.POS` | 142,122 | 9 | zyx | Brotli-11 | 442,065 | 3.110 | 5.14× |
| `synthetic_al_mg_si.POS` | 1,500,000 | 9 | zyx | Brotli-11 | 3,616,526 | 2.411 | 6.64× |
| `synthetic_si_o.POS` | 1,500,000 | 11 | xyz | Brotli-11 | 3,928,029 | 2.619 | 6.11× |
| `synthetic_zn_cu_al.POS` | 1,500,000 | 11 | yxz | Brotli-11 | 4,168,048 | 2.779 | 5.76× |
| `synthetic_zn_layer_on_al.POS` | 1,500,000 | 11 | yzx | Brotli-11 | 3,827,099 | 2.551 | 6.27× |
| `12d8ba7d-b728-4362-a879-558857c7b4d2.POS` | 20,949,148 | 11 | yzx | xz-9e | 50,103,904 | 2.392 | 6.69× |
| `2166bb75-7ff6-4c85-bf2d-564431f0b089.POS` | 29,810,068 | 12 | xzy | Brotli-11 | 46,903,591 | 1.573 | 10.17× |
| `499e563f-0c0c-4c6f-bc08-b8e76f59c31b.POS` | 5,721,296 | 11 | xyz | xz-9e | 14,235,256 | 2.488 | 6.43× |
| `86a2fa56-8593-4856-bd42-b73716197abf.POS` | 8,657,555 | 11 | xyz | Brotli-11 | 16,360,391 | 1.890 | 8.47× |
| `Sample 1- POS file.POS` | 4,013,368 | 11 | yzx | xz-9e | 11,261,260 | 2.806 | 5.70× |
| `Sample 3 -POS file.POS` | 17,949,114 | 10 | yzx | Brotli-11 | 44,263,650 | 2.466 | 6.49× |
| `e14f067b-4c6f-4c34-8e25-ac1e4a217c57.POS` | 8,401,853 | 11 | zyx | xz-9e | 15,034,024 | 1.789 | 8.94× |

Per-file results range from 1.573 to 3.431 bytes per point, or 4.66× to
10.17× smaller than float32 `.POS`.

## Alternatives tested

The experiment also measured:

- fixed 48-bit rows, column packing, byte shuffling, and bitplanes;
- order-preserving general-purpose compression;
- four-dimensional Morton sorting and delta varints;
- exact-mass grouping with three-dimensional Morton or Hilbert ordering;
- grouped varints, Elias–Fano coding, and Rice coding;
- adaptive Rice blocks of 256, 1,024, and 4,096 points;
- a single spatial stream with mass stored as a side channel;
- every mass-prefix width from zero through twelve bits;
- all six spatial-axis permutations;
- zstd levels 3, 19, and 22, xz-9e, Brotli-11, and large-window Brotli.

Global per-group Rice parameters beat adaptive blocks on the representative
large files. Hilbert ordering consistently but slightly beat Morton. Compact
varint metadata and large-window Brotli did not improve the aggregate.
Spatial-axis selection saved only 76,667 bytes corpus-wide, and choosing the
best final compressor saved another 797,396 bytes. The major gain comes from
canonical sorting and mass-prefix partitioning.

## Validation

The unit suite verifies exact round trips for fixed-width packing, varints,
Morton/Hilbert grouping, Elias–Fano, global Rice, split Rice, compact Rice,
block Rice, and every tested mass-prefix width.

An additional end-to-end check used the 5,525,361-point Ck10 control. Its
prefix-11/xzy stream survived Brotli compression and decompression and decoded
to an array exactly equal to the sorted quantized input. The pre-backend
stream SHA-256 was
`2dcf65b83d51d8d55761310e3c1aa9914b6846d2f93a4af7624017364964f768`.

## Interpretation

This looks promising as an archival or transfer representation if all four
values are already treated as 12-bit and acquisition order is disposable.
It is not a replacement for lossy CPOS previews: it retains every point, so
files remain materially larger than a capped preview.

For the public Ck10 example, the experimental lossless result is 10.97 MB for
all 5,525,361 quantized points. CPOS 1.0.0 is 4.04 MB because it retains only
499,000 preview points. That is a useful trade: about 2.7× the bytes retains
about 11.1× the points, under the 12-bit assumption.

If acquisition order must also be retained, the 4.676-byte/point result is
much less compelling. Encoding a permutation that restores the original order
would consume a significant part of the sorting gain and was intentionally
not counted as lossless canonical-set compression.

## Reproducibility

The source `.POS` files are not committed. The benchmark reads them from:

- `/Users/kyle/Documents/GitHub/uap/rangefinder/controls`
- `/Users/kyle/Documents/GitHub/uap/apt-analysis/data`

Raw measurements are in:

- `results/full_sweep.json` — all prefix widths plus ordered baselines;
- `results/full_axes.json` — all axis permutations for each prefix winner;
- `results/full_backends.json` — zstd, xz, and Brotli for each final transform.

The three recorded benchmark stages took 27.6 minutes on the test machine.
