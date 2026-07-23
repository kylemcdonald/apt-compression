# CPOS binary format

CPOS is a little-endian, single-file preview format for four-column APT
`.POS` data. Version 1 uses mass-stratified sampling and unsigned 16-bit
quantization.

## Versioning policy

- The container version describes the byte layout.
- The codec semantic version describes the sampling and quantization rules.
- The reference reader accepts exactly the current container and codec
  versions. It has no legacy decoding paths.

The reference encoders and readers use container `1.0` and codec `1.0.0`.

## Header

The header is 128 bytes. All unused bytes must be zero.

| Offset | Type | Field |
|---:|---|---|
| 0 | `char[4]` | magic: `CPOS` |
| 4 | `uint16` | container major |
| 6 | `uint16` | container minor |
| 8 | `uint16` | codec major |
| 10 | `uint16` | codec minor |
| 12 | `uint16` | codec patch |
| 14 | `uint16` | header size: 128 |
| 16 | `uint32` | endian marker: `0x01020304` |
| 20 | `uint32` | flags |
| 24 | `uint32` | original point count |
| 28 | `uint32` | retained point count |
| 32 | `uint32` | spectrum bin count |
| 36 | `float32` | spectrum minimum mass |
| 40 | `float32` | spectrum bin width |
| 44 | `float32` | spectrum maximum mass |
| 48 | `float32[3]` | minimum `x`, `y`, `z` |
| 60 | `float32[3]` | maximum `x`, `y`, `z` |
| 72 | `uint32` | original spectrum-count offset |
| 76 | `uint32` | retained spectrum-count offset |
| 80 | `uint32` | point-record offset |
| 84 | `uint32` | total file size |
| 88 | `uint32` | CRC32 of bytes after the header |
| 92 | `uint32` | requested maximum point count |
| 96 | 32 bytes | reserved, zero |

Flag bit 0 identifies the v1 stratified unsigned-16 representation.

The two spectrum tables contain one `uint32` per bin. They retain the original
and sampled bin occupancies. Point records are grouped by ascending spectrum
bin and contain four little-endian `uint16` values: quantized `x`, `y`, `z`,
and mass.

Codec 1.0 allocates retained points proportionally across the original mass
spectrum using largest-remainder apportionment. Within each bin, deterministic
midpoint samples are retained.

Version 1 uses 0–300 Da in 0.05 Da bins. Values outside this range are clipped
for preview storage. Spatial values are linearly quantized between the six
bounds in the header.
