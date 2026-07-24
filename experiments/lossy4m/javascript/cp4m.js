/**
 * Browser/Node decoder for the experimental CP4M 1.0.0 container.
 *
 * Rendering is intentionally kept outside this module.
 */

export const CONTAINER_VERSION = Object.freeze([1, 0]);
export const CODEC_VERSION = Object.freeze([1, 0, 0]);
export const HEADER_SIZE = 192;
export const CORE_GROUPED_MORTON_RICE = 1;
export const NOISE_NONE = 0;
export const NOISE_UNIFORM = 1;
export const NOISE_GAUSSIAN = 2;

const UINT32_SCALE = 0x1_0000_0000;
const MASK12 = 4095;
const BITS = 12;

function bytesOf(input) {
  if (input instanceof Uint8Array) return input;
  if (input instanceof ArrayBuffer) return new Uint8Array(input);
  if (ArrayBuffer.isView(input)) {
    return new Uint8Array(input.buffer, input.byteOffset, input.byteLength);
  }
  throw new TypeError("expected an ArrayBuffer or typed-array view");
}

function viewOf(input) {
  const bytes = bytesOf(input);
  return new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
}

let crcTable;

function makeCrcTable() {
  const table = new Uint32Array(256);
  for (let value = 0; value < 256; value += 1) {
    let crc = value;
    for (let bit = 0; bit < 8; bit += 1) {
      crc = (crc & 1) ? (0xedb88320 ^ (crc >>> 1)) : (crc >>> 1);
    }
    table[value] = crc >>> 0;
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

function versionString(version) {
  return version.join(".");
}

function number64(view, offset) {
  const value = view.getBigUint64(offset, true);
  if (value > BigInt(Number.MAX_SAFE_INTEGER)) {
    throw new RangeError("CP4M uint64 exceeds JavaScript's exact integer range");
  }
  return Number(value);
}

function magicAt(bytes, offset, expected) {
  for (let index = 0; index < expected.length; index += 1) {
    if (bytes[offset + index] !== expected.charCodeAt(index)) return false;
  }
  return true;
}

export function inspectCp4m(input, { verifyChecksum = true } = {}) {
  const bytes = bytesOf(input);
  if (bytes.byteLength < HEADER_SIZE) throw new Error("truncated CP4M header");
  if (!magicAt(bytes, 0, "CP4M")) throw new Error("not a CP4M file");
  const view = viewOf(bytes);
  const containerVersion = [
    view.getUint16(4, true),
    view.getUint16(6, true),
  ];
  const codecVersion = [
    view.getUint16(8, true),
    view.getUint16(10, true),
    view.getUint16(12, true),
  ];
  if (
    containerVersion[0] !== CONTAINER_VERSION[0]
    || containerVersion[1] !== CONTAINER_VERSION[1]
  ) {
    throw new Error(`unsupported CP4M container ${versionString(containerVersion)}`);
  }
  if (
    codecVersion[0] !== CODEC_VERSION[0]
    || codecVersion[1] !== CODEC_VERSION[1]
    || codecVersion[2] !== CODEC_VERSION[2]
  ) {
    throw new Error(`unsupported CP4M codec ${versionString(codecVersion)}`);
  }

  const headerSize = view.getUint16(14, true);
  const flags = view.getUint32(16, true);
  const originalPointCount = number64(view, 20);
  const storedPointCount = number64(view, 28);
  const targetPointCount = number64(view, 36);
  const spectrumBinCount = view.getUint32(44, true);
  const spectrumMinDa = view.getFloat32(48, true);
  const spectrumMaxDa = view.getFloat32(52, true);
  const storedSpectrumBinDa = view.getFloat32(56, true);
  const allocationExponent = view.getFloat32(60, true);
  const defaultNoise = view.getUint32(64, true);
  const seed = view.getBigUint64(68, true);
  const minimum = Array.from(
    { length: 4 },
    (_, axis) => view.getFloat32(76 + axis * 4, true),
  );
  const maximum = Array.from(
    { length: 4 },
    (_, axis) => view.getFloat32(92 + axis * 4, true),
  );
  const coreMethod = view.getUint8(108);
  const axisOrder = [view.getUint8(109), view.getUint8(110), view.getUint8(111)];
  const trueCountsOffset = number64(view, 112);
  const storedCountsOffset = number64(view, 120);
  const coreOffset = number64(view, 128);
  const coreCompressedSize = number64(view, 136);
  const coreUncompressedSize = number64(view, 144);
  const fileSize = number64(view, 152);
  const payloadCrc32 = view.getUint32(160, true);
  const reserved = view.getUint32(164, true);

  if (headerSize !== HEADER_SIZE || flags !== 1 || reserved !== 0) {
    throw new Error("unsupported CP4M header");
  }
  for (let offset = 168; offset < HEADER_SIZE; offset += 1) {
    if (bytes[offset] !== 0) {
      throw new Error("unsupported CP4M reserved header data");
    }
  }
  if (
    originalPointCount <= 0
    || storedPointCount <= 0
    || storedPointCount > originalPointCount
    || storedPointCount > targetPointCount
  ) {
    throw new Error("invalid CP4M point counts");
  }
  const expectedBins = Math.round(
    (spectrumMaxDa - spectrumMinDa) / storedSpectrumBinDa,
  );
  if (
    Math.abs(spectrumMinDa) > 1e-6
    || Math.abs(spectrumMaxDa - 300) > 1e-6
    || !Number.isFinite(storedSpectrumBinDa)
    || storedSpectrumBinDa <= 0
    || spectrumBinCount !== expectedBins
  ) {
    throw new Error("invalid CP4M histogram configuration");
  }
  const spectrumBinDa = (
    spectrumMaxDa - spectrumMinDa
  ) / spectrumBinCount;
  if (
    !Number.isFinite(allocationExponent)
    || allocationExponent < 0
    || allocationExponent > 1
  ) {
    throw new Error("invalid CP4M allocation exponent");
  }
  if (![NOISE_NONE, NOISE_UNIFORM, NOISE_GAUSSIAN].includes(defaultNoise)) {
    throw new Error("invalid CP4M default noise");
  }
  if (
    minimum.some((value) => !Number.isFinite(value))
    || maximum.some((value) => !Number.isFinite(value))
    || maximum.some((value, axis) => value < minimum[axis])
  ) {
    throw new Error("invalid CP4M quantizer bounds");
  }
  if (
    coreMethod !== CORE_GROUPED_MORTON_RICE
    || axisOrder[0] !== 0
    || axisOrder[1] !== 1
    || axisOrder[2] !== 2
  ) {
    throw new Error("unsupported CP4M core ordering");
  }
  const expectedStored = HEADER_SIZE + spectrumBinCount * 4;
  const expectedCore = expectedStored + spectrumBinCount * 4;
  const expectedSize = expectedCore + coreCompressedSize;
  if (
    trueCountsOffset !== HEADER_SIZE
    || storedCountsOffset !== expectedStored
    || coreOffset !== expectedCore
    || fileSize !== expectedSize
    || fileSize !== bytes.byteLength
  ) {
    throw new Error("invalid CP4M section layout");
  }
  if (verifyChecksum) {
    const actual = crc32(bytes.subarray(HEADER_SIZE));
    if (actual !== payloadCrc32) {
      throw new Error(
        `CP4M checksum mismatch: expected ${payloadCrc32.toString(16).padStart(8, "0")}, `
        + `got ${actual.toString(16).padStart(8, "0")}`,
      );
    }
  }
  return {
    containerVersion,
    codecVersion,
    originalPointCount,
    storedPointCount,
    targetPointCount,
    spectrumBinCount,
    spectrumMinDa,
    spectrumMaxDa,
    spectrumBinDa,
    allocationExponent,
    defaultNoise,
    seed,
    minimum,
    maximum,
    coreMethod,
    axisOrder,
    trueCountsOffset,
    storedCountsOffset,
    coreOffset,
    coreCompressedSize,
    coreUncompressedSize,
    fileSize,
    payloadCrc32,
  };
}

function readCounts(bytes, offset, count) {
  const view = viewOf(bytes);
  const output = new Uint32Array(count);
  for (let index = 0; index < count; index += 1) {
    output[index] = view.getUint32(offset + index * 4, true);
  }
  return output;
}

async function inflate(bytes) {
  if (typeof DecompressionStream !== "function") {
    throw new Error("this browser does not provide deflate decompression");
  }
  const stream = new Blob([bytes])
    .stream()
    .pipeThrough(new DecompressionStream("deflate"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

const mortonTables = (() => {
  const tables = [
    new Uint8Array(4096),
    new Uint8Array(4096),
    new Uint8Array(4096),
  ];
  for (let value = 0; value < 4096; value += 1) {
    for (let bit = 0; bit < 4; bit += 1) {
      for (let dimension = 0; dimension < 3; dimension += 1) {
        tables[dimension][value] |= (
          ((value >>> (bit * 3 + dimension)) & 1) << bit
        );
      }
    }
  }
  return tables;
})();

function mortonInverse(key) {
  let x = 0;
  let y = 0;
  let z = 0;
  for (let chunk = 0; chunk < 3; chunk += 1) {
    const value = Math.floor(key / (2 ** (chunk * 12))) % 4096;
    const shift = chunk * 4;
    x |= mortonTables[0][value] << shift;
    y |= mortonTables[1][value] << shift;
    z |= mortonTables[2][value] << shift;
  }
  return [x, y, z];
}

function bitplaneValues(bytes, offset, count, width) {
  const output = new Float64Array(count);
  if (width === 0 || count === 0) return output;
  const stride = Math.ceil(count / 8);
  for (let bit = 0; bit < width; bit += 1) {
    const scale = 2 ** bit;
    const planeOffset = offset + bit * stride;
    for (let index = 0; index < count; index += 1) {
      if ((bytes[planeOffset + (index >>> 3)] >>> (index & 7)) & 1) {
        output[index] += scale;
      }
    }
  }
  return output;
}

function decodeGroupedCore(core, storedCounts, expectedCount) {
  if (!magicAt(core, 0, "G12R")) throw new Error("invalid CP4M grouped core");
  const view = viewOf(core);
  const count = number64(view, 4);
  const activeCount = view.getUint32(12, true);
  if (count !== expectedCount) throw new Error("CP4M grouped core count mismatch");
  const active = [];
  for (let bin = 0; bin < storedCounts.length; bin += 1) {
    if (storedCounts[bin]) active.push(bin);
  }
  if (active.length !== activeCount) {
    throw new Error("CP4M grouped core active-bin mismatch");
  }
  let offset = 16;
  const widths = core.subarray(offset, offset + activeCount);
  offset += activeCount;
  const unaryLengths = new Float64Array(activeCount);
  for (let index = 0; index < activeCount; index += 1) {
    unaryLengths[index] = number64(view, offset + index * 8);
  }
  offset += activeCount * 8;
  let remainderBytes = 0;
  let unaryBytes = 0;
  for (let index = 0; index < activeCount; index += 1) {
    const size = storedCounts[active[index]];
    remainderBytes += Math.ceil(size / 8) * widths[index];
    unaryBytes += Math.ceil(unaryLengths[index] / 8);
  }
  let remainderCursor = offset;
  let unaryCursor = offset + remainderBytes;
  const unaryEnd = unaryCursor + unaryBytes;
  const massSize = Math.ceil(count / 8) * BITS;
  if (unaryEnd + massSize !== core.length) {
    throw new Error("invalid CP4M grouped core length");
  }

  const quantized = new Uint16Array(count * 4);
  const bins = new Uint32Array(count);
  let cursor = 0;
  for (let activeIndex = 0; activeIndex < activeCount; activeIndex += 1) {
    const bin = active[activeIndex];
    const size = storedCounts[bin];
    const width = widths[activeIndex];
    const remainderSize = Math.ceil(size / 8) * width;
    const remainders = bitplaneValues(
      core,
      remainderCursor,
      size,
      width,
    );
    remainderCursor += remainderSize;
    const unaryLength = unaryLengths[activeIndex];
    const unarySize = Math.ceil(unaryLength / 8);
    let previous = -1;
    let local = 0;
    let spatial = 0;
    for (let position = 0; position < unaryLength; position += 1) {
      if ((core[unaryCursor + (position >>> 3)] >>> (position & 7)) & 1) {
        const quotient = position - previous - 1;
        spatial += quotient * (2 ** width) + remainders[local];
        const coordinates = mortonInverse(spatial);
        const record = (cursor + local) * 4;
        quantized[record] = coordinates[0];
        quantized[record + 1] = coordinates[1];
        quantized[record + 2] = coordinates[2];
        bins[cursor + local] = bin;
        previous = position;
        local += 1;
      }
    }
    if (local !== size) throw new Error("invalid CP4M Rice unary stream");
    unaryCursor += unarySize;
    cursor += size;
  }
  const masses = bitplaneValues(core, unaryEnd, count, BITS);
  for (let index = 0; index < count; index += 1) {
    quantized[index * 4 + 3] = masses[index];
  }
  if (cursor !== count) throw new Error("CP4M grouped core point count mismatch");
  return { quantized, bins };
}

function sourceMassBin(mass, header) {
  return Math.max(
    0,
    Math.min(
      header.spectrumBinCount - 1,
      Math.floor((mass - header.spectrumMinDa) / header.spectrumBinDa),
    ),
  );
}

function forceMassBin(mass, bin, header) {
  if (sourceMassBin(mass, header) === bin) return mass;
  return header.spectrumMinDa + (bin + 0.5) * header.spectrumBinDa;
}

function dequantizeRetained(quantized, bins, header) {
  const points = new Float32Array(quantized.length);
  for (let point = 0; point < bins.length; point += 1) {
    const record = point * 4;
    for (let dimension = 0; dimension < 4; dimension += 1) {
      points[record + dimension] = (
        header.minimum[dimension]
        + quantized[record + dimension] / MASK12
        * (header.maximum[dimension] - header.minimum[dimension])
      );
    }
    points[record + 3] = forceMassBin(
      points[record + 3],
      bins[point],
      header,
    );
  }
  return points;
}

export async function decodeRetainedCp4m(input) {
  const bytes = bytesOf(input);
  const header = inspectCp4m(bytes);
  const trueCounts = readCounts(
    bytes,
    header.trueCountsOffset,
    header.spectrumBinCount,
  );
  const storedCounts = readCounts(
    bytes,
    header.storedCountsOffset,
    header.spectrumBinCount,
  );
  let trueTotal = 0;
  let storedTotal = 0;
  for (let index = 0; index < trueCounts.length; index += 1) {
    trueTotal += trueCounts[index];
    storedTotal += storedCounts[index];
    if (storedCounts[index] > trueCounts[index]) {
      throw new Error("CP4M stored histogram exceeds source histogram");
    }
  }
  if (
    trueTotal !== header.originalPointCount
    || storedTotal !== header.storedPointCount
  ) {
    throw new Error("CP4M histogram totals do not match the header");
  }
  const compressed = bytes.subarray(
    header.coreOffset,
    header.coreOffset + header.coreCompressedSize,
  );
  const core = await inflate(compressed);
  if (core.length !== header.coreUncompressedSize) {
    throw new Error("CP4M core size mismatch");
  }
  const { quantized, bins } = decodeGroupedCore(
    core,
    storedCounts,
    header.storedPointCount,
  );
  return {
    header,
    trueCounts,
    storedCounts,
    quantized,
    bins,
    points: dequantizeRetained(quantized, bins, header),
  };
}

function proportionalAllocation(counts, limit) {
  let total = 0;
  for (const count of counts) total += count;
  if (total <= limit) return Uint32Array.from(counts);
  const output = new Uint32Array(counts.length);
  const remainders = [];
  let used = 0;
  const totalBig = BigInt(total);
  const limitBig = BigInt(limit);
  for (let index = 0; index < counts.length; index += 1) {
    if (!counts[index]) continue;
    const product = BigInt(counts[index]) * limitBig;
    const value = Number(product / totalBig);
    output[index] = value;
    used += value;
    remainders.push({ index, remainder: product % totalBig });
  }
  remainders.sort((first, second) => {
    if (first.remainder === second.remainder) return first.index - second.index;
    return first.remainder > second.remainder ? -1 : 1;
  });
  for (let index = 0; index < limit - used; index += 1) {
    output[remainders[index].index] += 1;
  }
  return output;
}

function hashUniform(index, seed, bin, dimension) {
  let value = (
    Math.imul(index, 0x9e3779b1)
    + seed
    + Math.imul(bin, 0x85ebca6b)
    + Math.imul(dimension, 0xc2b2ae35)
  ) >>> 0;
  value = (value ^ (value >>> 16)) >>> 0;
  value = Math.imul(value, 0x7feb352d) >>> 0;
  value = (value ^ (value >>> 15)) >>> 0;
  value = Math.imul(value, 0x846ca68b) >>> 0;
  value = (value ^ (value >>> 16)) >>> 0;
  return value / UINT32_SCALE;
}

function noiseValue(index, seed, bin, dimension, mode) {
  if (mode === NOISE_NONE) return 0;
  const first = hashUniform(index, seed, bin, dimension);
  if (mode === NOISE_UNIFORM) return first - 0.5;
  const second = hashUniform(index, seed ^ 0xa511e9b3, bin, dimension);
  const gaussian = Math.sqrt(-2 * Math.log(Math.max(first, 1e-12)))
    * Math.cos(2 * Math.PI * second) * 0.22;
  return Math.max(-0.5, Math.min(0.5, gaussian));
}

function noiseMode(value, fallback) {
  if (value == null) return fallback;
  if (value === "none") return NOISE_NONE;
  if (value === "uniform") return NOISE_UNIFORM;
  if (value === "gaussian") return NOISE_GAUSSIAN;
  if ([NOISE_NONE, NOISE_UNIFORM, NOISE_GAUSSIAN].includes(value)) return value;
  throw new Error("noise must be none, uniform, or gaussian");
}

export function expandCp4m(
  decoded,
  {
    maxPoints = decoded.header.originalPointCount,
    noise = null,
  } = {},
) {
  if (!Number.isInteger(maxPoints) || maxPoints <= 0) {
    throw new RangeError("maxPoints must be a positive integer");
  }
  const outputCount = Math.min(maxPoints, decoded.header.originalPointCount);
  const outputCounts = proportionalAllocation(decoded.trueCounts, outputCount);
  const points = new Float32Array(outputCount * 4);
  const exact = new Uint8Array(outputCount);
  const bins = new Uint32Array(outputCount);
  const mode = noiseMode(noise, decoded.header.defaultNoise);
  const seed = Number(decoded.header.seed & 0xffffffffn);
  const starts = new Uint32Array(decoded.storedCounts.length + 1);
  for (let bin = 0; bin < decoded.storedCounts.length; bin += 1) {
    starts[bin + 1] = starts[bin] + decoded.storedCounts[bin];
  }

  let cursor = 0;
  for (let bin = 0; bin < outputCounts.length; bin += 1) {
    const display = outputCounts[bin];
    if (!display) continue;
    const source = decoded.trueCounts[bin];
    const stored = decoded.storedCounts[bin];
    const exactCount = outputCount === decoded.header.originalPointCount
      ? stored
      : Math.min(stored, Math.round(display * stored / source));
    const synthesized = display - exactCount;
    for (let local = 0; local < exactCount; local += 1) {
      const selected = Math.min(
        stored - 1,
        Math.floor((local + 0.5) * stored / exactCount),
      );
      const sourceRecord = (starts[bin] + selected) * 4;
      const outputRecord = (cursor + local) * 4;
      points.set(decoded.points.subarray(sourceRecord, sourceRecord + 4), outputRecord);
      exact[cursor + local] = 1;
      bins[cursor + local] = bin;
    }
    cursor += exactCount;
    for (let local = 0; local < synthesized; local += 1) {
      const parent = Math.min(
        stored - 1,
        Math.floor((local + 0.5) * stored / synthesized),
      );
      const sourceRecord = (starts[bin] + parent) * 4;
      const outputRecord = (cursor + local) * 4;
      for (let dimension = 0; dimension < 4; dimension += 1) {
        const quantized = Math.max(
          0,
          Math.min(
            MASK12,
            decoded.quantized[sourceRecord + dimension]
            + noiseValue(local, seed, bin, dimension, mode),
          ),
        );
        points[outputRecord + dimension] = (
          decoded.header.minimum[dimension]
          + quantized / MASK12
          * (
            decoded.header.maximum[dimension]
            - decoded.header.minimum[dimension]
          )
        );
      }
      points[outputRecord + 3] = forceMassBin(
        points[outputRecord + 3],
        bin,
        decoded.header,
      );
      bins[cursor + local] = bin;
    }
    cursor += synthesized;
  }
  if (cursor !== outputCount) throw new Error("CP4M display expansion mismatch");
  return { points, exact, bins, outputCounts, noiseMode: mode };
}
