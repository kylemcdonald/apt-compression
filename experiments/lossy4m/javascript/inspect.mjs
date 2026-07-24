import { readFile } from "node:fs/promises";
import { createHash } from "node:crypto";

import {
  decodeRetainedCp4m,
  expandCp4m,
} from "./cp4m.js";

const [path] = process.argv.slice(2);
if (!path) {
  console.error("usage: node inspect.mjs FILE.cp4m");
  process.exit(2);
}

const payload = await readFile(path);
const decoded = await decodeRetainedCp4m(payload);
const expanded = expandCp4m(decoded, {
  maxPoints: Math.min(decoded.header.originalPointCount, 100_000),
});
const quantizedHash = createHash("sha256")
  .update(Buffer.from(
    decoded.quantized.buffer,
    decoded.quantized.byteOffset,
    decoded.quantized.byteLength,
  ))
  .digest("hex");

console.log(JSON.stringify({
  header: {
    containerVersion: decoded.header.containerVersion,
    codecVersion: decoded.header.codecVersion,
    originalPointCount: decoded.header.originalPointCount,
    storedPointCount: decoded.header.storedPointCount,
    spectrumBinCount: decoded.header.spectrumBinCount,
    spectrumBinDa: decoded.header.spectrumBinDa,
    allocationExponent: decoded.header.allocationExponent,
  },
  trueTotal: decoded.trueCounts.reduce((sum, value) => sum + value, 0),
  storedTotal: decoded.storedCounts.reduce((sum, value) => sum + value, 0),
  quantizedHash,
  displayPoints: expanded.points.length / 4,
  displayExact: expanded.exact.reduce((sum, value) => sum + value, 0),
}));
