import assert from "node:assert/strict";
import test from "node:test";

import {
  ALGORITHM_VERSION,
  CposVersionError,
  decodeCpos,
  encodePos,
  samplePosBySpectrum,
} from "./cpos.js";

function fixturePos(count = 20_000) {
  const buffer = new ArrayBuffer(count * 16);
  const view = new DataView(buffer);
  for (let point = 0; point < count; point += 1) {
    const offset = point * 16;
    view.setFloat32(offset, Math.sin(point * 0.013) * 20, false);
    view.setFloat32(offset + 4, Math.cos(point * 0.007) * 15, false);
    view.setFloat32(offset + 8, point * 0.002, false);
    const selector = point % 10;
    const mass = selector < 7 ? 27.98 : selector < 9 ? 55.94 : 119.0;
    view.setFloat32(offset + 12, mass + (point % 7 - 3) * 0.004, false);
  }
  return buffer;
}

test("proportional spectrum and matched source preview", () => {
  const source = fixturePos();
  const payload = encodePos(source, { maxPoints: 4_999 });
  const decoded = decodeCpos(payload);
  const original = samplePosBySpectrum(
    source,
    decoded.trueCounts,
    decoded.storedCounts,
  );

  assert.deepEqual(decoded.header.algorithmVersion, ALGORITHM_VERSION);
  assert.equal(decoded.header.storedPointCount, 4_999);
  assert.equal(original.length, decoded.points.length);

  let variation = 0;
  for (let bin = 0; bin < decoded.trueCounts.length; bin += 1) {
    variation += Math.abs(
      decoded.trueCounts[bin] / decoded.header.originalPointCount
      - decoded.storedCounts[bin] / decoded.header.storedPointCount
    );
  }
  assert.ok(variation * 0.5 < 0.01);

  let maximumSpatialError = 0;
  for (let index = 0; index < original.length; index += 4) {
    for (let axis = 0; axis < 3; axis += 1) {
      maximumSpatialError = Math.max(
        maximumSpatialError,
        Math.abs(original[index + axis] - decoded.points[index + axis]),
      );
    }
  }
  assert.ok(maximumSpatialError < 0.001);

  const otherVersion = payload.slice(0);
  new DataView(otherVersion).setUint16(10, 1, true);
  assert.throws(
    () => decodeCpos(otherVersion),
    CposVersionError,
  );
});
