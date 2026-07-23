import { generateCdfV2WebGpu } from './cdfV2Gpu.js';
import { generateCdfV2Wasm } from './cdfV2Wasm.js';

const MAGIC = 'CDFV2\0\0\0';
const cache = new Map();
const cloudCache = new Map();

function typedArrayFromBytes(bytes, spec) {
  const slice = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  if (spec.dtype === 'float32') return new Float32Array(slice);
  if (spec.dtype === 'int8') return new Int8Array(slice);
  if (spec.dtype === 'uint8') return new Uint8Array(slice);
  if (spec.dtype === 'uint32') return new Uint32Array(slice);
  throw new Error(`Unsupported CDF v2 array dtype: ${spec.dtype}`);
}

async function decodeArray(buffer, dataStart, spec) {
  const start = dataStart + spec.offset;
  const end = start + spec.nbytes;
  let bytes = new Uint8Array(buffer, start, spec.nbytes);
  if (spec.compression === 'zlib') {
    bytes = await inflateZlib(bytes, spec.raw_nbytes);
  }
  return typedArrayFromBytes(bytes, spec);
}

async function parseArtifact(buffer) {
  const bytes = new Uint8Array(buffer, 0, 8);
  const magic = String.fromCharCode(...bytes);
  if (magic !== MAGIC) throw new Error('Invalid CDF v2 artifact.');
  const view = new DataView(buffer);
  const headerLength = view.getUint32(8, true);
  const headerText = new TextDecoder().decode(new Uint8Array(buffer, 12, headerLength));
  const header = JSON.parse(headerText);
  const dataStart = 12 + headerLength;
  const arrays = {};
  for (const spec of header.arrays) arrays[spec.name] = await decodeArray(buffer, dataStart, spec);
  return { header, arrays, supportMasks: new Map() };
}

export async function loadCdfV2(methodInfo, apiBase, progress = () => {}) {
  const endpoint = methodInfo?.artifact_endpoint;
  if (!endpoint) throw new Error('CDF v2 artifact endpoint is missing.');
  const url = new URL(endpoint, apiBase || window.location.origin);
  if (!cache.has(url.href)) {
    cache.set(url.href, (async () => {
      const started = performance.now();
      progress('Downloading CDF v2 artifact...');
      const response = await fetch(url);
      if (!response.ok) throw new Error(`CDF v2 request failed: ${response.status}`);
      const buffer = await response.arrayBuffer();
      const downloaded = performance.now();
      progress('Parsing CDF v2 artifact...');
      const artifact = await parseArtifact(buffer);
      const parsed = performance.now();
      artifact.timings = {
        downloadMs: downloaded - started,
        parseMs: parsed - downloaded,
        artifactMs: parsed - started
      };
      return artifact;
    })());
  }
  return cache.get(url.href);
}

async function inflateZlib(bytes, expectedLength) {
  if (typeof DecompressionStream === 'undefined') {
    throw new Error('This browser does not support DecompressionStream for CDF v2 masks.');
  }
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('deflate'));
  const buffer = await new Response(stream).arrayBuffer();
  const out = new Uint8Array(buffer);
  if (expectedLength && out.length !== expectedLength) {
    throw new Error(`CDF v2 mask length mismatch: expected ${expectedLength}, got ${out.length}`);
  }
  return out;
}

function supportChunkIndexForRange(artifact, rangeIndex) {
  const rangeChunk = artifact.header.ranges?.[rangeIndex]?.support_chunk_index;
  if (Number.isInteger(rangeChunk)) return rangeChunk;
  return artifact.header.support_chunks[rangeIndex] ? rangeIndex : 0;
}

async function supportMaskForRange(artifact, rangeIndex) {
  const chunkIndex = supportChunkIndexForRange(artifact, rangeIndex);
  if (!artifact.supportMasks.has(chunkIndex)) {
    artifact.supportMasks.set(chunkIndex, (async () => {
      const chunk = artifact.header.support_chunks[chunkIndex];
      const payload = artifact.arrays.support_payload.subarray(chunk.offset, chunk.offset + chunk.nbytes);
      return inflateZlib(payload, chunk.raw_nbytes);
    })());
  }
  return artifact.supportMasks.get(chunkIndex);
}

function supportIndicesFromMask(mask, paramCount, expectedCount = null) {
  let count = Number(expectedCount);
  if (!Number.isFinite(count) || count < 0) {
    count = 0;
    for (let byteIndex = 0; byteIndex < mask.length; byteIndex++) {
      let byte = mask[byteIndex];
      while (byte) {
        count += byte & 1;
        byte >>= 1;
      }
    }
  }
  const indices = new Uint32Array(count);
  let cursor = 0;
  for (let byteIndex = 0; byteIndex < mask.length; byteIndex++) {
    const byte = mask[byteIndex];
    if (!byte) continue;
    const base = byteIndex * 8;
    for (let bit = 7; bit >= 0; bit--) {
      if ((byte & (1 << bit)) === 0) continue;
      const index = base + (7 - bit);
      if (index >= paramCount) break;
      if (cursor < indices.length) indices[cursor] = index;
      cursor += 1;
    }
  }
  return cursor === indices.length ? indices : indices.slice(0, cursor);
}

async function supportIndicesForRange(artifact, rangeIndex) {
  if (!artifact.supportIndices) artifact.supportIndices = new Map();
  const chunkIndex = supportChunkIndexForRange(artifact, rangeIndex);
  if (!artifact.supportIndices.has(chunkIndex)) {
    artifact.supportIndices.set(chunkIndex, (async () => {
      const mask = await supportMaskForRange(artifact, rangeIndex);
      const chunk = artifact.header.support_chunks[chunkIndex];
      return supportIndicesFromMask(mask, artifact.header.settings.teacher_param_count, chunk?.occupied);
    })());
  }
  return artifact.supportIndices.get(chunkIndex);
}

function silu(x) {
  return x / (1 + Math.exp(-x));
}

function linear(input, weight, bias, outDim, inDim, activate = false) {
  const output = new Float32Array(outDim);
  for (let row = 0; row < outDim; row++) {
    let sum = bias[row];
    const offset = row * inDim;
    for (let col = 0; col < inDim; col++) sum += weight[offset + col] * input[col];
    output[row] = activate ? silu(sum) : sum;
  }
  return output;
}

function normalizedRangeValues(range, settings) {
  const stats = settings.feature_stats || {};
  const center = Number(range.mass ?? ((range.mass_min + range.mass_max) * 0.5));
  const width = Number(range.mass_max) - Number(range.mass_min);
  const centerNorm = ((center - stats.center_min) / Math.max(stats.center_max - stats.center_min, 1e-6)) * 2 - 1;
  if (settings.input_mode !== 'range') return [centerNorm];
  const widthNorm = ((width - stats.width_min) / Math.max(stats.width_max - stats.width_min, 1e-6)) * 2 - 1;
  return [centerNorm, widthNorm];
}

function coeffInput(range, settings) {
  const base = normalizedRangeValues(range, settings);
  const freqs = settings.feature_freqs || [1, 2, 4, 8, 16, 32];
  const out = new Float32Array(base.length * (1 + freqs.length * 2));
  let cursor = 0;
  for (const value of base) out[cursor++] = value;
  for (const freq of freqs) {
    for (const value of base) out[cursor++] = Math.sin(Math.PI * value * freq);
  }
  for (const freq of freqs) {
    for (const value of base) out[cursor++] = Math.cos(Math.PI * value * freq);
  }
  return out;
}

function predictCoeff(artifact, range) {
  const { header, arrays } = artifact;
  const settings = header.settings;
  const input = coeffInput(range, settings);
  const width = settings.hyper_width || 96;
  const h0 = linear(input, arrays['coeff.net.0.weight'], arrays['coeff.net.0.bias'], width, input.length, true);
  const h1 = linear(h0, arrays['coeff.net.2.weight'], arrays['coeff.net.2.bias'], width, width, true);
  const predNorm = linear(h1, arrays['coeff.net.4.weight'], arrays['coeff.net.4.bias'], settings.hyper_rank, width, false);
  const coeff = new Float32Array(settings.hyper_rank);
  for (let i = 0; i < coeff.length; i++) coeff[i] = predNorm[i] * arrays.coeff_std[i] + arrays.coeff_mean[i];
  return coeff;
}

function generateGrid(artifact, range) {
  const { header, arrays } = artifact;
  const settings = header.settings;
  const paramCount = settings.teacher_param_count;
  const grid = new Float32Array(paramCount);
  for (let i = 0; i < paramCount; i++) grid[i] = arrays.mean[i] / 255.0;
  const coeff = predictCoeff(artifact, range);
  for (let r = 0; r < settings.hyper_rank; r++) {
    const c = coeff[r] * arrays.basis_scale[r];
    const offset = r * paramCount;
    for (let i = 0; i < paramCount; i++) grid[i] += arrays.basis[offset + i] * c;
  }
  return grid;
}

function supportDensityFloor(range) {
  if (range?.support_mode !== 'per-bin') return 0;
  const explicit = Number(range.support_density_floor);
  if (Number.isFinite(explicit) && explicit > 0) return Math.min(1, explicit);
  const scale = Number(range?.target_scale || 1);
  return scale > 0 ? Math.min(1, 1 / scale) : 0;
}

function backendPreference() {
  const params = new URLSearchParams(window.location.search);
  return (params.get('cdfBackend') || params.get('backend') || 'webgpu').toLowerCase();
}

function maskHas(mask, index) {
  const byte = mask[index >> 3];
  const bit = 7 - (index & 7);
  return ((byte >> bit) & 1) === 1;
}

function lowerBound(cdf, value) {
  let lo = 0;
  let hi = cdf.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (cdf[mid] < value) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

function mulberry32(seed) {
  let t = seed >>> 0;
  return function rand() {
    t += 0x6d2b79f5;
    let x = Math.imul(t ^ (t >>> 15), 1 | t);
    x ^= x + Math.imul(x ^ (x >>> 7), 61 | x);
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
  };
}

function makeBaseCdf(artifact, rangeIndex, gridWeights, mask, range) {
  const paramCount = artifact.header.settings.teacher_param_count;
  const cdf = new Float64Array(paramCount);
  let total = 0;
  const scale = Number(range.target_scale || 1);
  const floor = supportDensityFloor(range);
  for (let i = 0; i < paramCount; i++) {
    if (maskHas(mask, i)) {
      const v = Math.max(floor, Math.max(0, Math.min(1, gridWeights[i])));
      total += v * scale;
    }
    cdf[i] = total;
  }
  return { cdf, total };
}

function makeResidualCdf(artifact, rangeIndex) {
  const { arrays } = artifact;
  const maxCount = artifact.header.settings.residual_cells_per_range || 0;
  const count = maxCount ? Number(arrays.residual_counts[rangeIndex] || 0) : 0;
  const cdf = new Float64Array(count);
  let total = 0;
  const offset = rangeIndex * maxCount;
  const scale = Number(arrays.residual_scales[rangeIndex] || 1);
  for (let i = 0; i < count; i++) {
    total += (arrays.residual_values[offset + i] / 255.0) * scale;
    cdf[i] = total;
  }
  return {
    indices: arrays.residual_indices.subarray(offset, offset + count),
    cdf,
    total
  };
}

function residualPointFraction(range, baseTotal, residualTotal) {
  if (!(residualTotal > 0)) return 0;
  const explicit = Number(range?.residual_atom_fraction);
  if (Number.isFinite(explicit) && explicit > 0) return Math.min(0.35, explicit);
  return Math.min(0.35, residualTotal / Math.max(baseTotal + residualTotal, 1e-6));
}

function interpolatedCellCoordinate(index, cellCount, rand) {
  // A tent kernel is a box convolved with a second box. Mixing one tent per
  // voxel by that voxel's weight produces the piecewise-linear interpolation
  // of the grid instead of a piecewise-constant block with visible seams.
  let coordinate = index + rand() + rand() - 0.5;
  if (coordinate < 0) coordinate = -coordinate;
  if (coordinate > cellCount) coordinate = 2 * cellCount - coordinate;
  return Math.max(0, Math.min(cellCount, coordinate));
}

function sampleFromCdf(cdf, total, indices, target, shape, bounds, range, seed) {
  const points = new Float32Array(target * 4);
  if (!(total > 0) || !target) return points;
  const [gx, gy, gz] = shape;
  const min = bounds[0];
  const max = bounds[1];
  const extent = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
  const massMin = Number(range.mass_min ?? range.mass ?? 0);
  const massMax = Number(range.mass_max ?? range.mass ?? massMin);
  const massWidth = Math.max(0, massMax - massMin);
  const rand = mulberry32(seed);
  const xy = gx * gy;
  for (let i = 0; i < target; i++) {
    const cdfIndex = lowerBound(cdf, rand() * total);
    const idx = indices ? indices[cdfIndex] : cdfIndex;
    const zi = Math.floor(idx / xy);
    const rem = idx - zi * xy;
    const yi = Math.floor(rem / gx);
    const xi = rem - yi * gx;
    points[i * 4 + 0] = min[0] + (interpolatedCellCoordinate(xi, gx, rand) / gx) * extent[0];
    points[i * 4 + 1] = min[1] + (interpolatedCellCoordinate(yi, gy, rand) / gy) * extent[1];
    points[i * 4 + 2] = min[2] + (interpolatedCellCoordinate(zi, gz, rand) / gz) * extent[2];
    points[i * 4 + 3] = massMin + rand() * massWidth;
  }
  return points;
}

function allocateRangeTargets(ranges, budget) {
  const totalAtoms = ranges.reduce((sum, range) => sum + (Number(range.atom_count) || 0), 0);
  const targetTotal = Math.max(0, Math.min(Math.floor(Number(budget) || 0), totalAtoms));
  if (!targetTotal || !totalAtoms) return ranges.map(() => 0);
  if (targetTotal >= totalAtoms) return ranges.map((range) => Number(range.atom_count) || 0);
  const raw = ranges.map((range) => ((Number(range.atom_count) || 0) / totalAtoms) * targetTotal);
  const targets = raw.map((value, index) => Math.min(Math.floor(value), Number(ranges[index].atom_count) || 0));
  let remainder = targetTotal - targets.reduce((sum, value) => sum + value, 0);
  const order = raw
    .map((value, index) => ({ index, frac: value - Math.floor(value) }))
    .sort((a, b) => b.frac - a.frac);
  let cursor = 0;
  while (remainder > 0 && order.length) {
    const index = order[cursor % order.length].index;
    const max = Number(ranges[index].atom_count) || 0;
    if (targets[index] < max) {
      targets[index] += 1;
      remainder -= 1;
    }
    cursor += 1;
  }
  return targets;
}

async function buildGenerationPlan(artifact, budget) {
  const ranges = artifact.header.ranges || [];
  const targets = allocateRangeTargets(ranges, budget);
  const totalPoints = targets.reduce((sum, value) => sum + value, 0);
  const offsets = [];
  let pointOffset = 0;
  for (let i = 0; i < ranges.length; i++) {
    offsets.push({ id: ranges[i].id, start: pointOffset, count: targets[i] });
    pointOffset += targets[i];
  }
  const coeffs = ranges.map((range) => predictCoeff(artifact, range));
  const masks = await Promise.all(ranges.map((_, index) => supportMaskForRange(artifact, index)));
  const supportIndices = await Promise.all(ranges.map((_, index) => supportIndicesForRange(artifact, index)));
  return { ranges, targets, totalPoints, offsets, coeffs, masks, supportIndices };
}

async function generateRangePoints(artifact, rangeIndex, target, progress = () => {}) {
  const range = artifact.header.ranges[rangeIndex];
  if (!target) return new Float32Array(0);
  progress(`Generating CDF v2 grid for ${range.id}...`);
  const gridWeights = generateGrid(artifact, range);
  const mask = await supportMaskForRange(artifact, rangeIndex);
  const base = makeBaseCdf(artifact, rangeIndex, gridWeights, mask, range);
  const residual = makeResidualCdf(artifact, rangeIndex);
  const residualFraction = residualPointFraction(range, base.total, residual.total);
  const residualTarget = Math.min(target, Math.round(target * residualFraction));
  const baseTarget = target - residualTarget;
  const bounds = artifact.header.dataset_bounds;
  const basePoints = sampleFromCdf(
    base.cdf,
    base.total,
    null,
    baseTarget,
    artifact.header.settings.base_grid_shape,
    bounds,
    range,
    0xa21f35 ^ rangeIndex ^ target
  );
  const residualPoints = sampleFromCdf(
    residual.cdf,
    residual.total,
    residual.indices,
    residualTarget,
    artifact.header.settings.residual_grid_shape,
    bounds,
    range,
    0x5f3759df ^ rangeIndex ^ target
  );
  const points = new Float32Array(target * 4);
  points.set(basePoints, 0);
  points.set(residualPoints, basePoints.length);
  return points;
}

function cacheKey(methodInfo, budget, apiBase) {
  const endpoint = methodInfo?.artifact_endpoint || '';
  return `${new URL(endpoint, apiBase || window.location.origin).href}|${Math.floor(Number(budget) || 0)}`;
}

export async function generateAllCdfV2Points(methodInfo, budget, apiBase, progress = () => {}) {
  const requestStarted = performance.now();
  const requestedBackend = backendPreference();
  const key = `${cacheKey(methodInfo, budget, apiBase)}|${requestedBackend}`;
  const hadCachedCloud = cloudCache.has(key);
  if (!hadCachedCloud) {
    cloudCache.set(key, (async () => {
      const artifact = await loadCdfV2(methodInfo, apiBase, progress);
      const generationStarted = performance.now();
      const plan = await buildGenerationPlan(artifact, budget);
      let backendResult = null;
      let backendError = null;
      if (requestedBackend !== 'wasm' && requestedBackend !== 'js') {
        try {
          backendResult = await generateCdfV2WebGpu(artifact, plan, progress);
        } catch (error) {
          backendError = error;
          console.warn('WebGPU CDF v2 generation failed; falling back to WASM.', error);
          progress(`WebGPU unavailable, using WASM fallback (${error.message})...`);
        }
      }
      if (!backendResult && requestedBackend !== 'js') {
        try {
          backendResult = await generateCdfV2Wasm(artifact, plan, progress);
        } catch (error) {
          backendError = error;
          console.warn('WASM CDF v2 generation failed; falling back to JS.', error);
          progress(`WASM unavailable, using JS fallback (${error.message})...`);
        }
      }
      if (!backendResult) {
        const points = new Float32Array(plan.totalPoints * 4);
        for (let i = 0; i < plan.ranges.length; i++) {
          const rangePoints = await generateRangePoints(artifact, i, plan.targets[i], progress);
          points.set(rangePoints, plan.offsets[i].start * 4);
        }
        backendResult = {
          points,
          timings: {
            backend: 'js',
            backendMs: performance.now() - generationStarted,
            fallbackReason: backendError?.message || null
          }
        };
      }
      const bins = new Float32Array(plan.totalPoints);
      for (let i = 0; i < plan.offsets.length; i++) {
        bins.fill(i, plan.offsets[i].start, plan.offsets[i].start + plan.offsets[i].count);
      }
      const generated = performance.now();
      return {
        points: backendResult.points,
        bins,
        offsets: plan.offsets,
        ranges: plan.ranges,
        totalPoints: plan.totalPoints,
        timings: {
          downloadMs: artifact.timings?.downloadMs ?? 0,
          parseMs: artifact.timings?.parseMs ?? 0,
          artifactMs: artifact.timings?.artifactMs ?? 0,
          pointGenerationMs: generated - generationStarted,
          totalMs: generated - requestStarted,
          backend: backendResult.timings?.backend || 'unknown',
          backendMs: backendResult.timings?.backendMs ?? null,
          attempts: backendResult.timings?.attempts ?? null,
          fallbackReason: backendResult.timings?.fallbackReason ?? null,
          cached: false
        }
      };
    })());
  } else {
    progress('Using cached CDF v2 point clouds...');
  }
  const result = await cloudCache.get(key);
  if (hadCachedCloud && result.timings) {
    return {
      ...result,
      timings: { ...result.timings, cached: true, totalMs: performance.now() - requestStarted }
    };
  }
  return result;
}
