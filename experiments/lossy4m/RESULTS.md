# CPOS 4M experiment: 18-file results

## Bottom line

Across 18 files containing 117,917,065 points, CP4M stores 47,437,532 exact 12-bit seeds and expands back to all 117,917,065 records.

CP4M occupies **126,658,113 bytes (120.79 MiB)** versus **64,684,872 bytes (61.69 MiB)** for CPOS 1.0.

That is 1.958× the bytes for 5.947× as many exact retained points. CP4M's point-weighted spatial JS divergence is 88.2× lower.

| Aggregate | CPOS 1.0 | CP4M |
| --- | ---: | ---: |
| File bytes | 64,684,872 | 126,658,113 |
| MiB | 61.69 | 120.79 |
| Exact retained seeds | 7,977,321 | 47,437,532 |
| Exact fraction of corpus | 6.77% | 40.23% |
| Decoded/expanded points | 7,977,321 | 117,917,065 |
| Ratio vs float32 `.POS` | 29.17× | 14.90× |
| Encode time (sum) | 9.85 s | 40.92 s |
| Decode time (sum) | 0.17 s | 17.40 s |
| Spatial JS divergence | 0.004729763 | 0.000053607 |
| Spatial total variation | 0.061102029 | 0.004344036 |
| Occupied-voxel recall | 97.107184% | 99.391514% |
| Mean axis EMD | 0.000362127 | 0.000027690 |
| Mass JS divergence (0.1 Da) | 0.000516204 | 0.000000000 |
| Mass total variation (0.1 Da) | 0.012056316 | 0.000000000 |

## Per-file comparison

| File | Points | `.POS` MiB | CPOS MiB | CP4M MiB | CP4M exact | CPOS spatial JS | CP4M spatial JS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `control_Ck10_steel_felfer_R56_01769.pos` | 5,525,361 | 84.31 | 3.85 | 8.54 | 72.4% | 0.0053855 | 0.0000154 |
| `control_MoHf_leitner_R21_08680.pos` | 4,583,568 | 69.94 | 3.85 | 10.15 | 87.3% | 0.0058856 | 0.0000270 |
| `control_ODSsteel_wang_R31_06365.pos` | 4,868,202 | 74.28 | 3.85 | 9.28 | 82.2% | 0.0071296 | 0.0000209 |
| `control_Si_apav_usa_denton_smith.pos` | 945,211 | 14.42 | 3.85 | 2.19 | 100.0% | 0.0024606 | 0.0000350 |
| `steelHD_5534859_70a59eff-003c-4337-832a-604c260dc623.POS` | 122,023 | 1.86 | 0.98 | 0.50 | 100.0% | 0.0000174 | 0.0001313 |
| `steelHD_5534859_8fe936a9-c8bf-417a-8006-bff810375bc7.POS` | 228,176 | 3.48 | 1.79 | 0.83 | 100.0% | 0.0000059 | 0.0000961 |
| `steelHD_5534859_aa137df8-ce0e-481e-93e6-a22cb5a34882.POS` | 142,122 | 2.17 | 1.13 | 0.47 | 100.0% | 0.0000093 | 0.0001399 |
| `synthetic_al_mg_si.POS` | 1,500,000 | 22.89 | 3.85 | 3.64 | 100.0% | 0.0068861 | 0.0000371 |
| `synthetic_si_o.POS` | 1,500,000 | 22.89 | 3.85 | 3.99 | 100.0% | 0.0068806 | 0.0000383 |
| `synthetic_zn_cu_al.POS` | 1,500,000 | 22.89 | 3.85 | 4.28 | 100.0% | 0.0068366 | 0.0000380 |
| `synthetic_zn_layer_on_al.POS` | 1,500,000 | 22.89 | 3.85 | 3.93 | 100.0% | 0.0069341 | 0.0000374 |
| `12d8ba7d-b728-4362-a879-558857c7b4d2.POS` | 20,949,148 | 319.66 | 3.85 | 12.15 | 19.1% | 0.0045728 | 0.0001159 |
| `2166bb75-7ff6-4c85-bf2d-564431f0b089.POS` | 29,810,068 | 454.87 | 3.85 | 8.61 | 13.4% | 0.0044693 | 0.0000153 |
| `499e563f-0c0c-4c6f-bc08-b8e76f59c31b.POS` | 5,721,296 | 87.30 | 3.85 | 11.16 | 69.9% | 0.0045218 | 0.0000311 |
| `86a2fa56-8593-4856-bd42-b73716197abf.POS` | 8,657,555 | 132.10 | 3.85 | 9.13 | 46.2% | 0.0043634 | 0.0000126 |
| `Sample 1- POS file.POS` | 4,013,368 | 61.24 | 3.85 | 11.53 | 99.7% | 0.0041090 | 0.0000097 |
| `Sample 3 -POS file.POS` | 17,949,114 | 273.88 | 3.85 | 12.29 | 22.3% | 0.0043944 | 0.0001323 |
| `e14f067b-4c6f-4c34-8e25-ac1e4a217c57.POS` | 8,401,853 | 128.20 | 3.85 | 8.13 | 47.6% | 0.0041183 | 0.0000130 |

## What the metrics mean

Spatial metrics use a 32³ normalized voxel grid plus 256-bin axis marginals. CPOS points are weighted by the source/retained count ratio of their native mass bin. CP4M metrics use the actual fully expanded, uniformly dithered output.

Mass metrics use 0.1 Da bins. CP4M stores the complete structural histogram and therefore restores it exactly. This does not mean that discarded within-bin mass positions were recovered.

An exact CP4M seed means its stored four-field 12-bit tuple was recovered losslessly. Synthesized records are deterministic children of those seeds and are explicitly marked as synthesized.

## Reproduction

```bash
python3 -m experiments.lossy4m.benchmark \
  --output experiments/lossy4m/results/full.json
python3 -m experiments.lossy4m.report \
  experiments/lossy4m/results/full.json \
  --output experiments/lossy4m/RESULTS.md
```

Parameters: target `4,000,000`, histogram `0.1` Da, allocation exponent `0.75`.
