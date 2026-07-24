# cpos

`cpos` is a lossy APT `.POS` codec for quick web-based previews. It keeps a
proportional mass-stratified sample of an atom cloud, quantizes each retained
`x`, `y`, `z`, and mass value to 16 bits, and stores the result in one
versioned `.cpos` file.

[Open the browser encoder and visualization](https://kylemcdonald.github.io/apt-cpos/)

CPOS is deliberately a preview format. Decoding returns the retained preview
points, not a synthetic cloud with the original ion count.

## Python

```bash
python3 -m pip install -e .
cpos encode input.pos preview.cpos --max-points 499000
cpos inspect preview.cpos
cpos decode preview.cpos reconstructed.pos
```

The Python API accepts an `N × 4` `float32` array containing `x`, `y`, `z`
(nanometers), and mass-to-charge ratio (Da):

```python
from cpos import decode, encode

payload = encode(points, max_points=499_000)
preview_points = decode(payload)
```

## JavaScript

The dependency-free ES module is in [`javascript/cpos.js`](javascript/cpos.js).
It works in browsers and Node:

```js
import { decodeCpos, encodePos } from "./javascript/cpos.js";

const payload = encodePos(posArrayBuffer, { maxPoints: 499_000 });
const { header, points } = decodeCpos(payload);
```

Python tests invoke Node and require byte-for-byte parity between both
encoders, then compare their decoded `.POS` output:

```bash
python3 -m pytest
```

## Versioned format

Every file begins with a fixed 128-byte `CPOS` header containing separate
container and codec versions. CPOS currently writes container `1.0` and codec
`1.0.0`. Readers accept exactly this version and verify the payload
CRC32 before decoding. See [FORMAT.md](FORMAT.md) for the binary layout and
compatibility policy.

## Public example

The Pages example is generated with 499,000 retained points from Peter
Felfer's public Ck10 steel control:

- Zenodo record [7979668](https://zenodo.org/records/7979668), CC-BY-4.0
- source archive `ger_erlangen_felfer_ck10.zip`
- source member `R56_01769-v01.pos`

The raw 88.4 MB `.pos` file is ignored and never committed. Download and build
the compact derived example with:

```bash
python3 scripts/download_example.py
python3 scripts/build_demo.py
python3 scripts/build_site.py
```

You can also pass an existing copy directly:

```bash
python3 scripts/build_demo.py --pos /path/to/R56_01769-v01.pos
```

Earlier compression experiments remain isolated on the
[`archive/all-codecs`](https://github.com/kylemcdonald/apt-cpos/tree/archive/all-codecs)
branch.

The [`research/lossy-4m`](experiments/lossy4m/README.md) branch contains an
experimental mass-aware codec that retains up to four million exact 12-bit
seeds and expands them back to the source point count with explicit
exact/synthesized provenance. Its JavaScript encoder is byte-equivalent to
Python, and its local visualizer can encode a dropped `.pos` file and compare
the original, CPOS, and CP4M results side by side.
