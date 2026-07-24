# Experimental CP4M 1.0 format

CP4M is a research container for a lossy, mass-aware four-million-seed point
cloud. Multibyte integers and floats are little-endian. Readers accept exactly
container `1.0` and codec `1.0.0`.

## File layout

```text
192-byte header
source histogram       spectrum_bin_count × uint32
retained histogram     spectrum_bin_count × uint32
Deflate-compressed grouped 12-bit core
```

The CRC32 covers everything after the 192-byte header.

## Header

| Offset | Size | Type | Meaning |
| ---: | ---: | --- | --- |
| 0 | 4 | bytes | magic `CP4M` |
| 4 | 4 | 2 × uint16 | container major, minor (`1`, `0`) |
| 8 | 6 | 3 × uint16 | codec major, minor, patch (`1`, `0`, `0`) |
| 14 | 2 | uint16 | header size (`192`) |
| 16 | 4 | uint32 | flags (`1`) |
| 20 | 8 | uint64 | original point count |
| 28 | 8 | uint64 | retained point count |
| 36 | 8 | uint64 | requested target point count |
| 44 | 4 | uint32 | spectrum bin count |
| 48 | 12 | 3 × float32 | spectrum minimum, maximum, stored bin width |
| 60 | 4 | float32 | allocation exponent |
| 64 | 4 | uint32 | default dither mode |
| 68 | 8 | uint64 | deterministic synthesis seed |
| 76 | 32 | 8 × float32 | four minima followed by four maxima |
| 108 | 1 | uint8 | core method (`1`, grouped Morton/Rice) |
| 109 | 3 | 3 × uint8 | spatial axis order (`0`, `1`, `2`) |
| 112 | 48 | 6 × uint64 | source counts offset, retained counts offset, core offset, compressed core size, uncompressed core size, file size |
| 160 | 4 | uint32 | payload CRC32 |
| 164 | 4 | uint32 | reserved, zero |
| 168 | 24 | bytes | reserved, zero |

The stored float32 bin width is validated, but decoders derive the canonical
width as `(spectrum_max - spectrum_min) / spectrum_bin_count`. This avoids
float32 representations such as `0.10000000149` shifting values across a bin
boundary.

## Histograms and allocation

The default histogram has 3,000 bins spanning `[0, 300)` Da at 0.1 Da. Values
outside the range are clamped into its first or last bin.

The first histogram stores exact source counts. The second stores exact
retained counts. Both totals must match their corresponding header counts and
no retained count may exceed its source count.

For inputs larger than the target, the encoder allocates a capped quota
proportional to `count ^ allocation_exponent`, with one seed reserved for every
nonempty bin. The default exponent is `0.75`. The deterministic remainder
rule produces exactly the requested number of retained points.

## Quantization and retained selection

Each of the four fields is quantized independently to an unsigned 12-bit value
using per-file float32 minima and maxima. This is the only numerical loss
applied to a retained seed.

Within each structural mass bin, the encoder sorts the 12-bit `(x, y, z)`
values by a 36-bit Morton key. It selects evenly spaced ranks from that
canonical order according to the bin quota. Acquisition order is discarded.

## Grouped core

After Deflate decompression, the core is:

```text
"G12R"                           4 bytes
retained point count             uint64
active mass-bin count            uint32
Rice remainder widths            active_count × uint8
Rice unary bit lengths           active_count × uint64
spatial remainder bitplanes      variable
spatial unary streams            variable
mass-value bitplanes             ceil(point_count / 8) × 12
```

For each active mass bin, sorted Morton gaps are Rice coded using the width
that minimizes that bin's stream. Remainders are stored as bitplanes and
quotients as unary streams. The 12-bit mass values are one aligned bitplane
stream in retained-record order.

The outer core uses zlib/Deflate level 9 so the same file can be decoded with
the browser-native `DecompressionStream("deflate")`.

## Expansion

Decoding the retained core exactly reconstructs the stored multiset of 12-bit
tuples. Full expansion then processes every active mass bin:

1. emit every retained seed and mark it exact;
2. distribute the required synthesized count evenly over those seeds;
3. add deterministic sub-cell dither to synthesized records only;
4. clamp each synthesized record to the 12-bit bounds and its structural mass
   bin.

The default dither is independent uniform noise in `[-0.5, 0.5)` quantization
units. `none` and clipped Gaussian modes are decoder options and do not change
the file.

The expanded result exactly matches the stored original point count and
structural mass histogram. It does not reconstruct discarded coordinates,
local density below the retained sampling scale, acquisition order, or the
original float32 values.
