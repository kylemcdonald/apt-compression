/**
 * Dependency-free CPOS v1 encoder and decoder for browsers and Node.
 *
 * Input POS records are four big-endian float32 values: x, y, z, and m/z.
 * CPOS records are grouped by mass bin and quantized to four little-endian
 * uint16 values. See ../FORMAT.md for the complete layout.
 */

export const CONTAINER_VERSION = Object.freeze([1, 0]);
export const ALGORITHM_VERSION = Object.freeze([1, 0, 0]);
export const HEADER_SIZE = 128;
export const DEFAULT_MAX_POINTS = 499_000;
export const SPECTRUM_MIN_DA = 0;
export const SPECTRUM_MAX_DA = 300;
export const SPECTRUM_BIN_DA = 0.05;
export const SPECTRUM_BINS = 6000;

const ENDIAN_MARKER = 0x01020304;
const FLAGS = 1;
const RECORD_SIZE = 8;
const UINT32_MAX = 0xffffffff;

export class CposVersionError extends Error {
  constructor(message) {
    super(message);
    this.name = "CposVersionError";
  }
}

function bytesOf(input) {
  if (input instanceof Uint8Array) return input;
  if (input instanceof ArrayBuffer) return new Uint8Array(input);
  if (ArrayBuffer.isView(input)) {
    return new Uint8Array(input.buffer, input.byteOffset, input.byteLength);
  }
  throw new TypeError("expected an ArrayBuffer or typed-array view");
}

function dataViewOf(input) {
  const bytes = bytesOf(input);
  return new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
}

function massBin(mass) {
  const scaled = Math.floor((mass - SPECTRUM_MIN_DA) / SPECTRUM_BIN_DA);
  return Math.max(0, Math.min(SPECTRUM_BINS - 1, scaled));
}

function allocate(counts, limit) {
  let total = 0;
  for (const count of counts) {
    total += count;
  }
  if (total <= limit) return Uint32Array.from(counts);

  const output = new Uint32Array(counts.length);
  const capacities = new Uint32Array(counts.length);
  let capacityTotal = 0;
  let used = 0;
  for (let index = 0; index < counts.length; index += 1) {
    capacities[index] = counts[index];
    capacityTotal += capacities[index];
  }

  const slots = limit - used;
  const slotsBig = BigInt(slots);
  const capacityTotalBig = BigInt(capacityTotal);
  const remainders = [];
  for (let index = 0; index < counts.length; index += 1) {
    if (capacities[index] === 0) continue;
    const product = BigInt(capacities[index]) * slotsBig;
    const quotient = Number(product / capacityTotalBig);
    output[index] += quotient;
    used += quotient;
    remainders.push({ index, remainder: product % capacityTotalBig });
  }
  remainders.sort((first, second) => {
    if (first.remainder === second.remainder) return first.index - second.index;
    return first.remainder > second.remainder ? -1 : 1;
  });
  const leftover = limit - used;
  for (let index = 0; index < leftover; index += 1) {
    output[remainders[index].index] += 1;
  }
  return output;
}

function quantize(value, minimum, maximum) {
  if (maximum <= minimum) return 0;
  const normalized = (value - minimum) / (maximum - minimum);
  return Math.max(0, Math.min(65535, Math.floor(normalized * 65535 + 0.5)));
}

let crcTable;

function makeCrcTable() {
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n += 1) {
    let c = n;
    for (let k = 0; k < 8; k += 1) {
      c = (c & 1) ? (0xedb88320 ^ (c >>> 1)) : (c >>> 1);
    }
    table[n] = c >>> 0;
  }
  return table;
}

export function crc32(input) {
  const bytes = bytesOf(input);
  if (!crcTable) crcTable = makeCrcTable();
  let crc = 0xffffffff;
  for (let index = 0; index < bytes.length; index += 1) {
    crc = crcTable[(crc ^ bytes[index]) & 0xff] ^ (crc >>> 8);
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function writeHeader(view, fields) {
  view.setUint8(0, 0x43);
  view.setUint8(1, 0x50);
  view.setUint8(2, 0x4f);
  view.setUint8(3, 0x53);
  view.setUint16(4, CONTAINER_VERSION[0], true);
  view.setUint16(6, CONTAINER_VERSION[1], true);
  view.setUint16(8, ALGORITHM_VERSION[0], true);
  view.setUint16(10, ALGORITHM_VERSION[1], true);
  view.setUint16(12, ALGORITHM_VERSION[2], true);
  view.setUint16(14, HEADER_SIZE, true);
  view.setUint32(16, ENDIAN_MARKER, true);
  view.setUint32(20, FLAGS, true);
  view.setUint32(24, fields.originalPointCount, true);
  view.setUint32(28, fields.storedPointCount, true);
  view.setUint32(32, SPECTRUM_BINS, true);
  view.setFloat32(36, SPECTRUM_MIN_DA, true);
  view.setFloat32(40, SPECTRUM_BIN_DA, true);
  view.setFloat32(44, SPECTRUM_MAX_DA, true);
  for (let axis = 0; axis < 3; axis += 1) {
    view.setFloat32(48 + axis * 4, fields.minimum[axis], true);
    view.setFloat32(60 + axis * 4, fields.maximum[axis], true);
  }
  view.setUint32(72, fields.trueCountsOffset, true);
  view.setUint32(76, fields.storedCountsOffset, true);
  view.setUint32(80, fields.recordsOffset, true);
  view.setUint32(84, fields.fileSize, true);
  view.setUint32(88, fields.payloadCrc32, true);
  view.setUint32(92, fields.maxPoints, true);
}

/**
 * Encode a complete POS ArrayBuffer as CPOS v1.
 *
 * Encoding is deterministic and uses two linear passes through the POS data.
 */
export function encodePos(input, { maxPoints = DEFAULT_MAX_POINTS } = {}) {
  const sourceBytes = bytesOf(input);
  if (sourceBytes.byteLength === 0 || sourceBytes.byteLength % 16 !== 0) {
    throw new Error("POS input must contain non-empty 16-byte records");
  }
  if (!Number.isInteger(maxPoints) || maxPoints <= 0 || maxPoints > UINT32_MAX) {
    throw new RangeError("maxPoints must be a positive uint32 integer");
  }
  const originalPointCount = sourceBytes.byteLength / 16;
  if (originalPointCount > UINT32_MAX) {
    throw new RangeError("CPOS v1 supports at most 2^32 - 1 input points");
  }

  const source = dataViewOf(sourceBytes);
  const trueCounts = new Uint32Array(SPECTRUM_BINS);
  const minimum = [Infinity, Infinity, Infinity];
  const maximum = [-Infinity, -Infinity, -Infinity];

  for (let point = 0; point < originalPointCount; point += 1) {
    const offset = point * 16;
    const x = source.getFloat32(offset, false);
    const y = source.getFloat32(offset + 4, false);
    const z = source.getFloat32(offset + 8, false);
    const mass = source.getFloat32(offset + 12, false);
    if (![x, y, z, mass].every(Number.isFinite)) {
      throw new Error(`POS input contains a non-finite value at point ${point}`);
    }
    minimum[0] = Math.min(minimum[0], x);
    minimum[1] = Math.min(minimum[1], y);
    minimum[2] = Math.min(minimum[2], z);
    maximum[0] = Math.max(maximum[0], x);
    maximum[1] = Math.max(maximum[1], y);
    maximum[2] = Math.max(maximum[2], z);
    trueCounts[massBin(mass)] += 1;
  }

  const storedCounts = allocate(trueCounts, Math.min(maxPoints, originalPointCount));
  const offsets = new Uint32Array(SPECTRUM_BINS + 1);
  for (let bin = 0; bin < SPECTRUM_BINS; bin += 1) {
    offsets[bin + 1] = offsets[bin] + storedCounts[bin];
  }
  const storedPointCount = offsets[SPECTRUM_BINS];
  const trueCountsOffset = HEADER_SIZE;
  const storedCountsOffset = trueCountsOffset + SPECTRUM_BINS * 4;
  const recordsOffset = storedCountsOffset + SPECTRUM_BINS * 4;
  const fileSize = recordsOffset + storedPointCount * RECORD_SIZE;
  if (fileSize > UINT32_MAX) {
    throw new RangeError("encoded file exceeds the CPOS v1 uint32 size limit");
  }

  const output = new ArrayBuffer(fileSize);
  const view = new DataView(output);
  for (let bin = 0; bin < SPECTRUM_BINS; bin += 1) {
    view.setUint32(trueCountsOffset + bin * 4, trueCounts[bin], true);
    view.setUint32(storedCountsOffset + bin * 4, storedCounts[bin], true);
  }

  const seen = new Uint32Array(SPECTRUM_BINS);
  const taken = new Uint32Array(SPECTRUM_BINS);
  for (let point = 0; point < originalPointCount; point += 1) {
    const inputOffset = point * 16;
    const mass = source.getFloat32(inputOffset + 12, false);
    const bin = massBin(mass);
    const take = storedCounts[bin];
    const selectedIndex = taken[bin];
    if (selectedIndex < take) {
      const target = Math.floor(
        (selectedIndex + 0.5) * trueCounts[bin] / take
      );
      if (seen[bin] === target) {
        const record = offsets[bin] + selectedIndex;
        const outputOffset = recordsOffset + record * RECORD_SIZE;
        view.setUint16(
          outputOffset,
          quantize(source.getFloat32(inputOffset, false), minimum[0], maximum[0]),
          true,
        );
        view.setUint16(
          outputOffset + 2,
          quantize(
            source.getFloat32(inputOffset + 4, false),
            minimum[1],
            maximum[1],
          ),
          true,
        );
        view.setUint16(
          outputOffset + 4,
          quantize(
            source.getFloat32(inputOffset + 8, false),
            minimum[2],
            maximum[2],
          ),
          true,
        );
        view.setUint16(
          outputOffset + 6,
          quantize(mass, SPECTRUM_MIN_DA, SPECTRUM_MAX_DA),
          true,
        );
        taken[bin] += 1;
      }
    }
    seen[bin] += 1;
  }

  const payloadCrc32 = crc32(new Uint8Array(output, HEADER_SIZE));
  writeHeader(view, {
    originalPointCount,
    storedPointCount,
    minimum,
    maximum,
    trueCountsOffset,
    storedCountsOffset,
    recordsOffset,
    fileSize,
    payloadCrc32,
    maxPoints,
  });
  return output;
}

function versionString(values) {
  return values.join(".");
}

/** Parse and validate a CPOS header. */
export function inspectCpos(input, { verifyChecksum = true } = {}) {
  const bytes = bytesOf(input);
  if (bytes.byteLength < HEADER_SIZE) throw new Error("truncated CPOS header");
  const view = dataViewOf(bytes);
  if (
    view.getUint8(0) !== 0x43
    || view.getUint8(1) !== 0x50
    || view.getUint8(2) !== 0x4f
    || view.getUint8(3) !== 0x53
  ) {
    throw new Error("not a CPOS file");
  }

  const containerVersion = [view.getUint16(4, true), view.getUint16(6, true)];
  const algorithmVersion = [
    view.getUint16(8, true),
    view.getUint16(10, true),
    view.getUint16(12, true),
  ];
  if (
    containerVersion[0] !== CONTAINER_VERSION[0]
    || containerVersion[1] !== CONTAINER_VERSION[1]
  ) {
    throw new CposVersionError(
      `unsupported CPOS container ${versionString(containerVersion)}`,
    );
  }
  if (
    algorithmVersion[0] !== ALGORITHM_VERSION[0]
    || algorithmVersion[1] !== ALGORITHM_VERSION[1]
    || algorithmVersion[2] !== ALGORITHM_VERSION[2]
  ) {
    throw new CposVersionError(
      `unsupported CPOS codec ${versionString(algorithmVersion)}`,
    );
  }

  const headerSize = view.getUint16(14, true);
  const endianMarker = view.getUint32(16, true);
  const flags = view.getUint32(20, true);
  const originalPointCount = view.getUint32(24, true);
  const storedPointCount = view.getUint32(28, true);
  const spectrumBinCount = view.getUint32(32, true);
  const spectrumMinDa = view.getFloat32(36, true);
  const spectrumBinDa = view.getFloat32(40, true);
  const spectrumMaxDa = view.getFloat32(44, true);
  const minimum = [0, 1, 2].map((axis) => view.getFloat32(48 + axis * 4, true));
  const maximum = [0, 1, 2].map((axis) => view.getFloat32(60 + axis * 4, true));
  const trueCountsOffset = view.getUint32(72, true);
  const storedCountsOffset = view.getUint32(76, true);
  const recordsOffset = view.getUint32(80, true);
  const fileSize = view.getUint32(84, true);
  const payloadCrc32 = view.getUint32(88, true);
  const maxPoints = view.getUint32(92, true);

  if (headerSize !== HEADER_SIZE) {
    throw new Error(`unsupported CPOS header size ${headerSize}`);
  }
  if (endianMarker !== ENDIAN_MARKER) throw new Error("invalid CPOS endian marker");
  if (flags !== FLAGS) throw new Error(`unsupported CPOS flags 0x${flags.toString(16)}`);
  if (originalPointCount === 0 || storedPointCount === 0) {
    throw new Error("CPOS point counts must be non-zero");
  }
  if (storedPointCount > originalPointCount || storedPointCount > maxPoints) {
    throw new Error("invalid CPOS retained point count");
  }
  if (spectrumBinCount !== SPECTRUM_BINS) {
    throw new Error(`unsupported CPOS spectrum size ${spectrumBinCount}`);
  }

  const expectedStoredOffset = HEADER_SIZE + spectrumBinCount * 4;
  const expectedRecordsOffset = expectedStoredOffset + spectrumBinCount * 4;
  const expectedSize = expectedRecordsOffset + storedPointCount * RECORD_SIZE;
  if (
    trueCountsOffset !== HEADER_SIZE
    || storedCountsOffset !== expectedStoredOffset
    || recordsOffset !== expectedRecordsOffset
    || fileSize !== expectedSize
    || fileSize !== bytes.byteLength
  ) {
    throw new Error("invalid CPOS section offsets or file size");
  }
  if (verifyChecksum) {
    const actual = crc32(bytes.subarray(headerSize));
    if (actual !== payloadCrc32) {
      throw new Error(
        `CPOS checksum mismatch: expected ${payloadCrc32.toString(16).padStart(8, "0")}, `
        + `got ${actual.toString(16).padStart(8, "0")}`,
      );
    }
  }

  return Object.freeze({
    containerVersion: Object.freeze(containerVersion),
    algorithmVersion: Object.freeze(algorithmVersion),
    headerSize,
    flags,
    originalPointCount,
    storedPointCount,
    spectrumBinCount,
    spectrumMinDa,
    spectrumBinDa,
    spectrumMaxDa,
    bounds: Object.freeze([
      Object.freeze(minimum),
      Object.freeze(maximum),
    ]),
    trueCountsOffset,
    storedCountsOffset,
    recordsOffset,
    fileSize,
    payloadCrc32,
    maxPoints,
  });
}

/**
 * Decode CPOS into retained float32 points and copies of both spectrum tables.
 */
export function decodeCpos(input, { verifyChecksum = true } = {}) {
  const bytes = bytesOf(input);
  const header = inspectCpos(bytes, { verifyChecksum });
  const view = dataViewOf(bytes);
  const trueCounts = new Uint32Array(header.spectrumBinCount);
  const storedCounts = new Uint32Array(header.spectrumBinCount);
  let originalTotal = 0;
  let storedTotal = 0;
  for (let bin = 0; bin < header.spectrumBinCount; bin += 1) {
    const original = view.getUint32(header.trueCountsOffset + bin * 4, true);
    const retained = view.getUint32(header.storedCountsOffset + bin * 4, true);
    if (retained > original) {
      throw new Error("CPOS retained spectrum exceeds the original spectrum");
    }
    trueCounts[bin] = original;
    storedCounts[bin] = retained;
    originalTotal += original;
    storedTotal += retained;
  }
  if (originalTotal !== header.originalPointCount) {
    throw new Error("CPOS original spectrum counts do not match the header");
  }
  if (storedTotal !== header.storedPointCount) {
    throw new Error("CPOS retained spectrum counts do not match the header");
  }

  const points = new Float32Array(header.storedPointCount * 4);
  const minimum = header.bounds[0];
  const maximum = header.bounds[1];
  for (let point = 0; point < header.storedPointCount; point += 1) {
    const inputOffset = header.recordsOffset + point * RECORD_SIZE;
    const outputOffset = point * 4;
    for (let axis = 0; axis < 3; axis += 1) {
      const q = view.getUint16(inputOffset + axis * 2, true);
      points[outputOffset + axis] = (
        minimum[axis] + q / 65535 * (maximum[axis] - minimum[axis])
      );
    }
    points[outputOffset + 3] = (
      header.spectrumMinDa
      + view.getUint16(inputOffset + 6, true) / 65535
      * (header.spectrumMaxDa - header.spectrumMinDa)
    );
  }
  return { header, points, trueCounts, storedCounts };
}

/** Return a uniformly spaced float32 display sample from a POS buffer. */
export function samplePos(input, maxPoints = DEFAULT_MAX_POINTS) {
  const bytes = bytesOf(input);
  if (bytes.byteLength === 0 || bytes.byteLength % 16 !== 0) {
    throw new Error("POS input must contain non-empty 16-byte records");
  }
  if (!Number.isInteger(maxPoints) || maxPoints <= 0) {
    throw new RangeError("maxPoints must be a positive integer");
  }
  const view = dataViewOf(bytes);
  const count = bytes.byteLength / 16;
  const take = Math.min(count, maxPoints);
  const points = new Float32Array(take * 4);
  for (let index = 0; index < take; index += 1) {
    const sourcePoint = Math.floor((index + 0.5) * count / take);
    for (let field = 0; field < 4; field += 1) {
      points[index * 4 + field] = view.getFloat32(
        sourcePoint * 16 + field * 4,
        false,
      );
    }
  }
  return points;
}

/**
 * Return the exact unquantized POS points selected by a CPOS spectrum.
 *
 * This lets before/after views contain the same ions, with only CPOS
 * quantization separating them.
 */
export function samplePosBySpectrum(input, trueCounts, storedCounts) {
  const bytes = bytesOf(input);
  if (bytes.byteLength === 0 || bytes.byteLength % 16 !== 0) {
    throw new Error("POS input must contain non-empty 16-byte records");
  }
  if (
    !(trueCounts instanceof Uint32Array)
    || !(storedCounts instanceof Uint32Array)
    || trueCounts.length !== SPECTRUM_BINS
    || storedCounts.length !== SPECTRUM_BINS
  ) {
    throw new TypeError("spectrum counts must be CPOS v1 Uint32Array tables");
  }

  const pointCount = bytes.byteLength / 16;
  let trueTotal = 0;
  let storedTotal = 0;
  const offsets = new Uint32Array(SPECTRUM_BINS + 1);
  for (let bin = 0; bin < SPECTRUM_BINS; bin += 1) {
    if (storedCounts[bin] > trueCounts[bin]) {
      throw new Error("retained spectrum exceeds the original spectrum");
    }
    trueTotal += trueCounts[bin];
    storedTotal += storedCounts[bin];
    offsets[bin + 1] = storedTotal;
  }
  if (trueTotal !== pointCount) {
    throw new Error("spectrum counts do not match the POS point count");
  }

  const source = dataViewOf(bytes);
  const output = new Float32Array(storedTotal * 4);
  const seen = new Uint32Array(SPECTRUM_BINS);
  const taken = new Uint32Array(SPECTRUM_BINS);
  for (let point = 0; point < pointCount; point += 1) {
    const inputOffset = point * 16;
    const mass = source.getFloat32(inputOffset + 12, false);
    const bin = massBin(mass);
    const take = storedCounts[bin];
    const selectedIndex = taken[bin];
    if (selectedIndex < take) {
      const target = Math.floor(
        (selectedIndex + 0.5) * trueCounts[bin] / take
      );
      if (seen[bin] === target) {
        const outputOffset = (offsets[bin] + selectedIndex) * 4;
        for (let field = 0; field < 4; field += 1) {
          output[outputOffset + field] = source.getFloat32(
            inputOffset + field * 4,
            false,
          );
        }
        taken[bin] += 1;
      }
    }
    seen[bin] += 1;
  }
  return output;
}

/** Serialize an interleaved float array as four-column big-endian POS. */
export function pointsToPos(points) {
  if (!(points instanceof Float32Array) || points.length % 4 !== 0) {
    throw new TypeError("points must be an interleaved Float32Array");
  }
  const output = new ArrayBuffer(points.length * 4);
  const view = new DataView(output);
  for (let index = 0; index < points.length; index += 1) {
    view.setFloat32(index * 4, points[index], false);
  }
  return output;
}
