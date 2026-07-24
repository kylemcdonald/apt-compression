# CP4M parameter and dither study

The defaults were selected on three deliberately different files: the public
Ck10 control, the largest 29.8-million-point file, and the difficult
17.9-million-point Sample 3 file. The full sweep is in
[`results/tuning.json`](results/tuning.json).

## Allocation exponent

The selected `count ^ 0.75` allocator was compared with proportional
`count ^ 1.0` allocation at 0.1 Da. `Rare retention` and `major retention` are
point-weighted retention rates for bins below the median and above the 90th
count percentile, respectively.

| File | Exponent | MiB | Spatial JS | Rare retention | Major retention |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ck10 | **0.75** | 8.54 | **0.0000133** | **100.0%** | 71.5% |
| Ck10 | 1.00 | **8.20** | 0.0000309 | 74.6% | **72.3%** |
| 29.8M points | **0.75** | 8.61 | **0.0000160** | **100.0%** | 12.8% |
| 29.8M points | 1.00 | **7.64** | 0.0000266 | 15.3% | **13.4%** |
| Sample 3 | **0.75** | 12.29 | **0.0001331** | **56.9%** | 15.7% |
| Sample 3 | 1.00 | **11.44** | 0.0001700 | 22.4% | **22.3%** |

The sublinear allocator costs 0.34–0.97 MiB on these files. In exchange, it
substantially protects rare bins and lowers spatial divergence. Exponents
`0.25` and `0.50` overprotected small bins on Sample 3 and starved large peaks;
at 0.1 Da their spatial JS divergences were `0.002255` and `0.000377`.

## Histogram width

At exponent 0.75, moving from 0.1 to 0.01 Da increased file size without
improving the measured spatial distribution:

| File | 0.1 Da MiB | 0.01 Da MiB | 0.1 Da spatial JS | 0.01 Da spatial JS |
| --- | ---: | ---: | ---: | ---: |
| Ck10 | **8.54** | 10.07 | **0.0000133** | 0.0000219 |
| 29.8M points | **8.61** | 9.94 | **0.0000160** | 0.0000268 |
| Sample 3 | **12.29** | 14.29 | **0.0001331** | 0.0003088 |

The format and encoder still support 0.01 Da when the finer structural
histogram is worth the additional bytes. The default is 0.1 Da.

## Sub-cell dither

The dither study fully expanded Ck10 and Sample 3 and measured a finer 64³
voxel grid. Raw measurements are in
[`results/noise.json`](results/noise.json).

| File | Dither | Spatial JS | Occupied-voxel recall | Mean axis EMD |
| --- | --- | ---: | ---: | ---: |
| Ck10 | none | **0.00020164** | 99.7234% | 0.00004651 |
| Ck10 | **uniform** | 0.00020790 | **99.7384%** | **0.00003407** |
| Ck10 | Gaussian | 0.00020409 | 99.7326% | 0.00003940 |
| Sample 3 | none | 0.00219945 | 98.9186% | 0.00005335 |
| Sample 3 | **uniform** | **0.00217108** | **99.0374%** | **0.00002952** |
| Sample 3 | Gaussian | 0.00217593 | 99.0172% | 0.00003671 |

Uniform ±0.5-cell noise is the default. It consistently improves occupancy and
axis marginals, and it improves Sample 3's spatial JS. Ck10's spatial JS is
3.1% worse, so this is not a universal metric win. The stronger reason for
uniform is statistical: it is the natural dequantization distribution after
round-to-nearest 12-bit quantization. Gaussian noise asserts an unsupported
sub-cell shape and did not outperform uniform overall.

Dither is applied only to synthesized records. Exact retained seeds never
move, and provenance remains available to renderers and downstream code.
