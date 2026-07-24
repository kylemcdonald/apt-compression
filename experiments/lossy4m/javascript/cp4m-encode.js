/**
 * Browser/Node encoder for the experimental CP4M 1.0.0 container.
 *
 * Input POS records are four big-endian float32 values: x, y, z, and m/z.
 * The emitted bytes are identical to experiments/lossy4m/codec.py when the
 * same options are used. Pako 2.1.0 supplies the zlib 1.2.x-compatible
 * level-9 Deflate stream used by the Python encoder.
 */

import { deflate } from "./vendor/pako.esm.mjs";
import {
  CODEC_VERSION,
  CONTAINER_VERSION,
  CORE_GROUPED_MORTON_RICE,
  HEADER_SIZE,
  NOISE_UNIFORM,
  crc32,
} from "./cp4m.js";

export const DEFAULT_TARGET_POINTS = 4_000_000;
export const DEFAULT_BIN_WIDTH_DA = 0.1;
export const DEFAULT_ALLOCATION_EXPONENT = 0.75;
export const DEFAULT_SEED = 0xc0454dn;

const BITS = 12;
const MASK12 = 4095;
const SPATIAL_BITS = 36;
const SPECTRUM_MIN_DA = 0;
const SPECTRUM_MAX_DA = 300;
const FLAGS = 1;
const UINT32_MAX = 0xffffffff;

function bytesOf(input) {
  if (input instanceof Uint8Array) return input;
  if (input instanceof ArrayBuffer) return new Uint8Array(input);
  if (ArrayBuffer.isView(input)) {
    return new Uint8Array(input.buffer, input.byteOffset, input.byteLength);
  }
  throw new TypeError("expected an ArrayBuffer or typed-array view");
}

function report(onProgress, stage, fraction) {
  if (onProgress) onProgress({ stage, fraction });
}

function spectrumBinCount(binWidthDa) {
  if (!Number.isFinite(binWidthDa) || binWidthDa <= 0) {
    throw new RangeError("binWidthDa must be positive and finite");
  }
  const count = Math.round(
    (SPECTRUM_MAX_DA - SPECTRUM_MIN_DA) / binWidthDa,
  );
  if (
    count <= 0
    || count > UINT32_MAX
    || Math.abs(
      count * binWidthDa - (SPECTRUM_MAX_DA - SPECTRUM_MIN_DA),
    ) > 1e-4
  ) {
    throw new RangeError("binWidthDa must divide the 0–300 Da range");
  }
  return count;
}

function sourceMassBin(mass, binWidthDa, binCount) {
  return Math.max(
    0,
    Math.min(
      binCount - 1,
      Math.floor((mass - SPECTRUM_MIN_DA) / binWidthDa),
    ),
  );
}

export function allocateSublinear(counts, limit, exponent) {
  if (!(counts instanceof Uint32Array)) {
    throw new TypeError("counts must be a Uint32Array");
  }
  if (!Number.isInteger(limit) || limit <= 0) {
    throw new RangeError("limit must be a positive integer");
  }
  if (!Number.isFinite(exponent) || exponent < 0 || exponent > 1) {
    throw new RangeError("exponent must be in [0, 1]");
  }

  let total = 0;
  let activeCount = 0;
  for (const count of counts) {
    total += count;
    if (count) activeCount += 1;
  }
  if (total <= limit) return Uint32Array.from(counts);
  if (limit < activeCount) {
    throw new RangeError("limit is smaller than the number of nonempty bins");
  }

  const output = new Uint32Array(counts.length);
  const capacity = new Float64Array(counts.length);
  const weights = new Float64Array(counts.length);
  let remaining = limit - activeCount;
  let upper = 0;
  for (let bin = 0; bin < counts.length; bin += 1) {
    if (!counts[bin]) continue;
    output[bin] = 1;
    capacity[bin] = counts[bin] - 1;
    weights[bin] = Math.pow(counts[bin], exponent);
    if (capacity[bin]) {
      upper = Math.max(upper, capacity[bin] / weights[bin]);
    }
  }
  if (!remaining) return output;

  let lower = 0;
  for (let iteration = 0; iteration < 80; iteration += 1) {
    const midpoint = (lower + upper) * 0.5;
    let allocated = 0;
    for (let bin = 0; bin < counts.length; bin += 1) {
      allocated += Math.min(capacity[bin], midpoint * weights[bin]);
    }
    if (allocated < remaining) lower = midpoint;
    else upper = midpoint;
  }

  const candidates = [];
  let used = activeCount;
  for (let bin = 0; bin < counts.length; bin += 1) {
    if (!counts[bin]) continue;
    const ideal = Math.min(capacity[bin], upper * weights[bin]);
    const whole = Math.floor(ideal);
    output[bin] += whole;
    used += whole;
    if (output[bin] < counts[bin]) {
      candidates.push({
        bin,
        source: counts[bin],
        fractional: ideal - whole,
      });
    }
  }
  const leftover = limit - used;
  candidates.sort((first, second) => (
    second.fractional - first.fractional
    || first.source - second.source
    || first.bin - second.bin
  ));
  for (let index = 0; index < leftover; index += 1) {
    output[candidates[index].bin] += 1;
  }

  let check = 0;
  for (let bin = 0; bin < counts.length; bin += 1) {
    check += output[bin];
    if (output[bin] > counts[bin]) {
      throw new Error("invalid sublinear allocation");
    }
  }
  if (check !== limit) throw new Error("invalid sublinear allocation total");
  return output;
}

const MORTON3 = (() => {
  const table = new Float64Array(4096);
  for (let value = 0; value < table.length; value += 1) {
    let spread = 0;
    for (let bit = 0; bit < BITS; bit += 1) {
      spread += ((value >>> bit) & 1) * (2 ** (bit * 3));
    }
    table[value] = spread;
  }
  return table;
})();

function spatialMorton(values, record) {
  return (
    MORTON3[values[record]]
    + MORTON3[values[record + 1]] * 2
    + MORTON3[values[record + 2]] * 4
  );
}

function combinedDigit(key, bin, pass) {
  if (pass < 4) return Math.floor(key / (2 ** (pass * 8))) & 0xff;
  if (pass === 4) {
    return (Math.floor(key / (2 ** 32)) & 0x0f) | ((bin & 0x0f) << 4);
  }
  if (pass === 5) return (bin >>> 4) & 0xff;
  return (bin >>> 12) & 0xff;
}

function radixOrder(values, bins, onProgress) {
  const count = bins.length;
  let source = new Uint32Array(count);
  let target = new Uint32Array(count);
  const keys = new Float64Array(count);
  for (let point = 0; point < count; point += 1) {
    source[point] = point;
    keys[point] = spatialMorton(values, point * 4);
  }
  const counts = new Uint32Array(256);
  const offsets = new Uint32Array(256);
  for (let pass = 0; pass < 7; pass += 1) {
    counts.fill(0);
    for (let index = 0; index < count; index += 1) {
      const point = source[index];
      counts[combinedDigit(keys[point], bins[point], pass)] += 1;
    }
    let cursor = 0;
    for (let digit = 0; digit < counts.length; digit += 1) {
      offsets[digit] = cursor;
      cursor += counts[digit];
    }
    for (let index = 0; index < count; index += 1) {
      const point = source[index];
      const digit = combinedDigit(keys[point], bins[point], pass);
      target[offsets[digit]] = point;
      offsets[digit] += 1;
    }
    [source, target] = [target, source];
    report(onProgress, "sorting spatial keys", 0.34 + pass * 0.04);
  }
  return source;
}

function selectQuantized(
  values,
  order,
  trueCounts,
  storedCounts,
  storedPointCount,
) {
  const starts = new Float64Array(trueCounts.length + 1);
  for (let bin = 0; bin < trueCounts.length; bin += 1) {
    starts[bin + 1] = starts[bin] + trueCounts[bin];
  }
  const selected = new Uint16Array(storedPointCount * 4);
  let cursor = 0;
  for (let bin = 0; bin < trueCounts.length; bin += 1) {
    const take = storedCounts[bin];
    if (!take) continue;
    const count = trueCounts[bin];
    for (let local = 0; local < take; local += 1) {
      const position = Math.floor((local + 0.5) * count / take);
      const sourcePoint = order[starts[bin] + position];
      const sourceRecord = sourcePoint * 4;
      const outputRecord = cursor * 4;
      selected[outputRecord] = values[sourceRecord];
      selected[outputRecord + 1] = values[sourceRecord + 1];
      selected[outputRecord + 2] = values[sourceRecord + 2];
      selected[outputRecord + 3] = values[sourceRecord + 3];
      cursor += 1;
    }
  }
  if (cursor !== storedPointCount) {
    throw new Error("CP4M selected point count mismatch");
  }
  return selected;
}

function setAscii(output, offset, text) {
  for (let index = 0; index < text.length; index += 1) {
    output[offset + index] = text.charCodeAt(index);
  }
}

function groupMetadata(selected, storedCounts) {
  const active = [];
  let maximumSize = 0;
  for (let bin = 0; bin < storedCounts.length; bin += 1) {
    if (!storedCounts[bin]) continue;
    active.push(bin);
    maximumSize = Math.max(maximumSize, storedCounts[bin]);
  }
  const widths = new Uint8Array(active.length);
  const unaryLengths = new Float64Array(active.length);
  const remainderSizes = new Float64Array(active.length);
  const unarySizes = new Float64Array(active.length);
  const gaps = new Float64Array(maximumSize);
  const quotientSums = new Float64Array(SPATIAL_BITS + 1);
  let selectedCursor = 0;
  let remainderBytes = 0;
  let unaryBytes = 0;

  for (let activeIndex = 0; activeIndex < active.length; activeIndex += 1) {
    const size = storedCounts[active[activeIndex]];
    let previous = 0;
    quotientSums.fill(0);
    for (let local = 0; local < size; local += 1) {
      const spatial = spatialMorton(selected, (selectedCursor + local) * 4);
      const gap = local ? spatial - previous : spatial;
      gaps[local] = gap;
      previous = spatial;
      let quotient = gap;
      for (let width = 0; width <= SPATIAL_BITS; width += 1) {
        quotientSums[width] += quotient;
        quotient = Math.floor(quotient / 2);
      }
    }
    let bestWidth = 0;
    let bestBits = Infinity;
    for (let width = 0; width <= SPATIAL_BITS; width += 1) {
      const bits = size * (width + 1) + quotientSums[width];
      if (bits < bestBits) {
        bestBits = bits;
        bestWidth = width;
      }
    }
    const unaryLength = quotientSums[bestWidth] + size;
    const remainderSize = Math.ceil(size / 8) * bestWidth;
    const unarySize = Math.ceil(unaryLength / 8);
    widths[activeIndex] = bestWidth;
    unaryLengths[activeIndex] = unaryLength;
    remainderSizes[activeIndex] = remainderSize;
    unarySizes[activeIndex] = unarySize;
    remainderBytes += remainderSize;
    unaryBytes += unarySize;
    selectedCursor += size;
  }
  return {
    active,
    widths,
    unaryLengths,
    remainderSizes,
    unarySizes,
    remainderBytes,
    unaryBytes,
    gaps,
  };
}

function encodeGroupedCore(selected, storedCounts, onProgress) {
  const metadata = groupMetadata(selected, storedCounts);
  const count = selected.length / 4;
  const massSize = Math.ceil(count / 8) * BITS;
  const metadataSize = (
    16
    + metadata.active.length
    + metadata.active.length * 8
  );
  const core = new Uint8Array(
    metadataSize
    + metadata.remainderBytes
    + metadata.unaryBytes
    + massSize,
  );
  const view = new DataView(core.buffer);
  setAscii(core, 0, "G12R");
  view.setBigUint64(4, BigInt(count), true);
  view.setUint32(12, metadata.active.length, true);
  core.set(metadata.widths, 16);
  let offset = 16 + metadata.active.length;
  for (const length of metadata.unaryLengths) {
    view.setBigUint64(offset, BigInt(length), true);
    offset += 8;
  }

  let remainderCursor = metadataSize;
  let unaryCursor = metadataSize + metadata.remainderBytes;
  let selectedCursor = 0;
  for (
    let activeIndex = 0;
    activeIndex < metadata.active.length;
    activeIndex += 1
  ) {
    const size = storedCounts[metadata.active[activeIndex]];
    const width = metadata.widths[activeIndex];
    const stride = Math.ceil(size / 8);
    let previous = 0;
    for (let local = 0; local < size; local += 1) {
      const spatial = spatialMorton(selected, (selectedCursor + local) * 4);
      metadata.gaps[local] = local ? spatial - previous : spatial;
      previous = spatial;
    }
    for (let bit = 0; bit < width; bit += 1) {
      const plane = remainderCursor + bit * stride;
      const divisor = 2 ** bit;
      for (let local = 0; local < size; local += 1) {
        if (Math.floor(metadata.gaps[local] / divisor) & 1) {
          core[plane + (local >>> 3)] |= 1 << (local & 7);
        }
      }
    }
    let unaryPosition = 0;
    const divisor = 2 ** width;
    for (let local = 0; local < size; local += 1) {
      unaryPosition += Math.floor(metadata.gaps[local] / divisor) + 1;
      const end = unaryPosition - 1;
      core[unaryCursor + Math.floor(end / 8)] |= 1 << (end & 7);
    }
    remainderCursor += metadata.remainderSizes[activeIndex];
    unaryCursor += metadata.unarySizes[activeIndex];
    selectedCursor += size;
    if ((activeIndex & 127) === 0) {
      report(
        onProgress,
        "packing exact 12-bit seeds",
        0.63 + 0.17 * activeIndex / metadata.active.length,
      );
    }
  }

  const massOffset = metadataSize + metadata.remainderBytes + metadata.unaryBytes;
  const massStride = Math.ceil(count / 8);
  for (let bit = 0; bit < BITS; bit += 1) {
    const plane = massOffset + bit * massStride;
    for (let point = 0; point < count; point += 1) {
      if ((selected[point * 4 + 3] >>> bit) & 1) {
        core[plane + (point >>> 3)] |= 1 << (point & 7);
      }
    }
  }
  return core;
}

function writeHeader(
  output,
  {
    originalPointCount,
    storedPointCount,
    targetPointCount,
    spectrumBinCount: binCount,
    binWidthDa,
    allocationExponent,
    seed,
    minimum,
    maximum,
    coreOffset,
    coreCompressedSize,
    coreUncompressedSize,
    payloadCrc32,
  },
) {
  const view = new DataView(output.buffer, output.byteOffset, output.byteLength);
  setAscii(output, 0, "CP4M");
  view.setUint16(4, CONTAINER_VERSION[0], true);
  view.setUint16(6, CONTAINER_VERSION[1], true);
  view.setUint16(8, CODEC_VERSION[0], true);
  view.setUint16(10, CODEC_VERSION[1], true);
  view.setUint16(12, CODEC_VERSION[2], true);
  view.setUint16(14, HEADER_SIZE, true);
  view.setUint32(16, FLAGS, true);
  view.setBigUint64(20, BigInt(originalPointCount), true);
  view.setBigUint64(28, BigInt(storedPointCount), true);
  view.setBigUint64(36, BigInt(targetPointCount), true);
  view.setUint32(44, binCount, true);
  view.setFloat32(48, SPECTRUM_MIN_DA, true);
  view.setFloat32(52, SPECTRUM_MAX_DA, true);
  view.setFloat32(56, binWidthDa, true);
  view.setFloat32(60, allocationExponent, true);
  view.setUint32(64, NOISE_UNIFORM, true);
  view.setBigUint64(68, seed, true);
  for (let axis = 0; axis < 4; axis += 1) {
    view.setFloat32(76 + axis * 4, minimum[axis], true);
    view.setFloat32(92 + axis * 4, maximum[axis], true);
  }
  view.setUint8(108, CORE_GROUPED_MORTON_RICE);
  view.setUint8(109, 0);
  view.setUint8(110, 1);
  view.setUint8(111, 2);
  view.setBigUint64(112, BigInt(HEADER_SIZE), true);
  view.setBigUint64(120, BigInt(HEADER_SIZE + binCount * 4), true);
  view.setBigUint64(128, BigInt(coreOffset), true);
  view.setBigUint64(136, BigInt(coreCompressedSize), true);
  view.setBigUint64(144, BigInt(coreUncompressedSize), true);
  view.setBigUint64(152, BigInt(output.byteLength), true);
  view.setUint32(160, payloadCrc32, true);
}

export function encodePosCp4m(
  input,
  {
    targetPoints = DEFAULT_TARGET_POINTS,
    binWidthDa = DEFAULT_BIN_WIDTH_DA,
    allocationExponent = DEFAULT_ALLOCATION_EXPONENT,
    seed = DEFAULT_SEED,
    onProgress = null,
  } = {},
) {
  const sourceBytes = bytesOf(input);
  if (!sourceBytes.byteLength || sourceBytes.byteLength % 16 !== 0) {
    throw new Error("POS input must contain non-empty 16-byte records");
  }
  const originalPointCount = sourceBytes.byteLength / 16;
  if (originalPointCount > UINT32_MAX) {
    throw new RangeError("CP4M supports at most 2^32 - 1 source points");
  }
  if (!Number.isInteger(targetPoints) || targetPoints <= 0) {
    throw new RangeError("targetPoints must be a positive integer");
  }
  if (
    !Number.isFinite(allocationExponent)
    || allocationExponent < 0
    || allocationExponent > 1
  ) {
    throw new RangeError("allocationExponent must be in [0, 1]");
  }
  if (typeof seed === "number") seed = BigInt(seed);
  if (typeof seed !== "bigint" || seed < 0n || seed >= (1n << 64n)) {
    throw new RangeError("seed must be an unsigned 64-bit integer");
  }

  const binCount = spectrumBinCount(binWidthDa);
  const trueCounts = new Uint32Array(binCount);
  const minimum = new Float64Array([Infinity, Infinity, Infinity, Infinity]);
  const maximum = new Float64Array([-Infinity, -Infinity, -Infinity, -Infinity]);
  const source = new DataView(
    sourceBytes.buffer,
    sourceBytes.byteOffset,
    sourceBytes.byteLength,
  );
  report(onProgress, "reading source points", 0);
  for (let point = 0; point < originalPointCount; point += 1) {
    const record = point * 16;
    for (let dimension = 0; dimension < 4; dimension += 1) {
      const value = source.getFloat32(record + dimension * 4, false);
      if (!Number.isFinite(value)) {
        throw new Error(`POS contains a non-finite value at point ${point}`);
      }
      minimum[dimension] = Math.min(minimum[dimension], value);
      maximum[dimension] = Math.max(maximum[dimension], value);
    }
    const mass = source.getFloat32(record + 12, false);
    trueCounts[sourceMassBin(mass, binWidthDa, binCount)] += 1;
  }

  report(onProgress, "quantizing to 12 bits", 0.12);
  const values = new Uint16Array(originalPointCount * 4);
  const bins = new Uint16Array(originalPointCount);
  for (let point = 0; point < originalPointCount; point += 1) {
    const sourceRecord = point * 16;
    const outputRecord = point * 4;
    for (let dimension = 0; dimension < 4; dimension += 1) {
      const value = source.getFloat32(sourceRecord + dimension * 4, false);
      const extent = maximum[dimension] - minimum[dimension];
      const normalized = (
        value - minimum[dimension]
      ) / (extent > 0 ? extent : 1);
      values[outputRecord + dimension] = Math.max(
        0,
        Math.min(MASK12, Math.floor(normalized * MASK12 + 0.5)),
      );
    }
    bins[point] = sourceMassBin(
      source.getFloat32(sourceRecord + 12, false),
      binWidthDa,
      binCount,
    );
  }

  const storedPointCount = Math.min(originalPointCount, targetPoints);
  const storedCounts = allocateSublinear(
    trueCounts,
    storedPointCount,
    allocationExponent,
  );
  report(onProgress, "sorting spatial keys", 0.3);
  const order = radixOrder(values, bins, onProgress);
  const selected = selectQuantized(
    values,
    order,
    trueCounts,
    storedCounts,
    storedPointCount,
  );
  report(onProgress, "packing exact 12-bit seeds", 0.62);
  const core = encodeGroupedCore(selected, storedCounts, onProgress);
  report(onProgress, "compressing CP4M", 0.82);
  const compressedCore = deflate(core, { level: 9 });

  const trueCountsOffset = HEADER_SIZE;
  const storedCountsOffset = trueCountsOffset + binCount * 4;
  const coreOffset = storedCountsOffset + binCount * 4;
  const output = new Uint8Array(coreOffset + compressedCore.byteLength);
  const outputView = new DataView(output.buffer);
  for (let bin = 0; bin < binCount; bin += 1) {
    outputView.setUint32(trueCountsOffset + bin * 4, trueCounts[bin], true);
    outputView.setUint32(storedCountsOffset + bin * 4, storedCounts[bin], true);
  }
  output.set(compressedCore, coreOffset);
  const payloadCrc32 = crc32(output.subarray(HEADER_SIZE));
  writeHeader(output, {
    originalPointCount,
    storedPointCount,
    targetPointCount: targetPoints,
    spectrumBinCount: binCount,
    binWidthDa,
    allocationExponent,
    seed,
    minimum,
    maximum,
    coreOffset,
    coreCompressedSize: compressedCore.byteLength,
    coreUncompressedSize: core.byteLength,
    payloadCrc32,
  });
  report(onProgress, "complete", 1);
  return output;
}
