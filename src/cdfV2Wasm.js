const WASM_URL = new URL(
  'cdf_v2_sampler.wasm',
  new URL(import.meta.env.BASE_URL || '/', window.location.origin)
).href;
let wasmPromise = null;

function bytesOf(array) {
  return new Uint8Array(array.buffer, array.byteOffset, array.byteLength);
}

async function loadWasm() {
  if (!wasmPromise) {
    wasmPromise = (async () => {
      const response = await fetch(WASM_URL);
      if (!response.ok) throw new Error(`WASM sampler request failed: ${response.status}`);
      const bytes = await response.arrayBuffer();
      const result = await WebAssembly.instantiate(bytes, {
        env: {
          abort() {
            throw new Error('CDF v2 WASM sampler aborted.');
          }
        }
      });
      return result.instance;
    })();
  }
  return wasmPromise;
}

function malloc(instance, nbytes) {
  const ptr = instance.exports.wasm_malloc(nbytes);
  if (!ptr) throw new Error(`WASM allocation failed for ${nbytes} bytes.`);
  return ptr >>> 0;
}

function copyToWasm(instance, ptr, array) {
  new Uint8Array(instance.exports.memory.buffer, ptr, array.byteLength).set(bytesOf(array));
}

function ensureFreshMemory(instance) {
  return new Uint8Array(instance.exports.memory.buffer);
}

function supportDensityFloor(range) {
  if (range?.support_mode !== 'per-bin') return 0;
  const explicit = Number(range.support_density_floor);
  if (Number.isFinite(explicit) && explicit > 0) return Math.min(1, explicit);
  const scale = Number(range?.target_scale || 1);
  return scale > 0 ? Math.min(1, 1 / scale) : 0;
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

function residualPointFraction(range, baseTarget, residualTotal) {
  if (!(residualTotal > 0)) return 0;
  const explicit = Number(range?.residual_atom_fraction);
  if (Number.isFinite(explicit) && explicit > 0) return Math.min(0.35, explicit);
  return Math.min(0.35, residualTotal / Math.max(baseTarget + residualTotal, 1e-6));
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
    points[i * 4 + 0] = min[0] + ((xi + rand()) / gx) * extent[0];
    points[i * 4 + 1] = min[1] + ((yi + rand()) / gy) * extent[1];
    points[i * 4 + 2] = min[2] + ((zi + rand()) / gz) * extent[2];
    points[i * 4 + 3] = massMin + rand() * massWidth;
  }
  return points;
}

export async function generateCdfV2Wasm(artifact, plan, progress = () => {}) {
  const started = performance.now();
  const instance = await loadWasm();
  instance.exports.wasm_reset();

  const { arrays } = artifact;
  const rank = artifact.header.settings.hyper_rank;
  const meanPtr = malloc(instance, arrays.mean.byteLength);
  const basisPtr = malloc(instance, arrays.basis.byteLength);
  const basisScalePtr = malloc(instance, arrays.basis_scale.byteLength);
  const maxSupportBytes = Math.max(4, ...plan.supportIndices.map((indices) => indices.byteLength));
  const supportPtr = malloc(instance, maxSupportBytes);
  const coeffPtr = malloc(instance, rank * 4);
  const maxRangeTarget = Math.max(1, ...plan.targets);
  const outPtr = malloc(instance, maxRangeTarget * 16);

  ensureFreshMemory(instance);
  copyToWasm(instance, meanPtr, arrays.mean);
  copyToWasm(instance, basisPtr, arrays.basis);
  copyToWasm(instance, basisScalePtr, arrays.basis_scale);

  const points = new Float32Array(plan.totalPoints * 4);
  const [gx, gy, gz] = artifact.header.settings.base_grid_shape;
  const bounds = artifact.header.dataset_bounds;
  const min = bounds[0];
  const max = bounds[1];
  const attempts = Number(new URLSearchParams(window.location.search).get('wasmAttempts')) || 64;

  for (let rangeIndex = 0; rangeIndex < plan.ranges.length; rangeIndex++) {
    const target = plan.targets[rangeIndex];
    if (!target) continue;
    const range = plan.ranges[rangeIndex];
    const supportIndices = plan.supportIndices[rangeIndex];
    const residual = makeResidualCdf(artifact, rangeIndex);
    const residualTarget = Math.min(target, Math.round(target * residualPointFraction(range, target, residual.total)));
    const baseTarget = target - residualTarget;
    progress(`WASM sampling ${range.id} (${rangeIndex + 1}/${plan.ranges.length})...`);
    if (baseTarget && supportIndices.length) {
      copyToWasm(instance, coeffPtr, plan.coeffs[rangeIndex]);
      copyToWasm(instance, supportPtr, supportIndices);
      instance.exports.sample_range(
        meanPtr,
        basisPtr,
        basisScalePtr,
        coeffPtr,
        supportPtr,
        outPtr,
        artifact.header.settings.teacher_param_count,
        supportIndices.length,
        gx,
        gy,
        gz,
        rank,
        baseTarget,
        0,
        0x7f4a7c15 ^ rangeIndex ^ target,
        attempts,
        supportDensityFloor(range),
        Number(range.mass_min ?? range.mass ?? 0),
        Math.max(0, Number(range.mass_max ?? range.mass ?? range.mass_min ?? 0) - Number(range.mass_min ?? range.mass ?? 0)),
        Number(min[0]),
        Number(min[1]),
        Number(min[2]),
        Number(max[0]) - Number(min[0]),
        Number(max[1]) - Number(min[1]),
        Number(max[2]) - Number(min[2])
      );
      const view = new Float32Array(instance.exports.memory.buffer, outPtr, baseTarget * 4);
      points.set(view, plan.offsets[rangeIndex].start * 4);
    }
    if (residualTarget) {
      const residualPoints = sampleFromCdf(
        residual.cdf,
        residual.total,
        residual.indices,
        residualTarget,
        artifact.header.settings.residual_grid_shape,
        artifact.header.dataset_bounds,
        range,
        0x5f3759df ^ rangeIndex ^ target
      );
      points.set(residualPoints, (plan.offsets[rangeIndex].start + baseTarget) * 4);
    }
  }

  return {
    points,
    timings: {
      backend: 'wasm',
      backendMs: performance.now() - started,
      attempts
    }
  };
}
