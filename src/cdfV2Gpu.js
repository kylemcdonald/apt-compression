let devicePromise = null;
let pipelinePromise = null;

const SHADER = `
struct Params {
  totalCells: u32,
  gx: u32,
  gy: u32,
  gz: u32,
  rank: u32,
  targetCount: u32,
  pointOffset: u32,
  seed: u32,
  attempts: u32,
  supportCount: u32,
  supportFloor: f32,
  _pad1: u32,
  massMin: f32,
  massWidth: f32,
  minX: f32,
  minY: f32,
  minZ: f32,
  extentX: f32,
  extentY: f32,
  extentZ: f32,
};

@group(0) @binding(0) var<storage, read> meanPacked: array<u32>;
@group(0) @binding(1) var<storage, read> basisPacked: array<u32>;
@group(0) @binding(2) var<storage, read> basisScale: array<f32>;
@group(0) @binding(3) var<storage, read> coeff: array<f32>;
@group(0) @binding(4) var<storage, read> supportIndex: array<u32>;
@group(0) @binding(5) var<storage, read_write> points: array<vec4<f32>>;
@group(0) @binding(6) var<uniform> params: Params;

fn byte_at(data: ptr<storage, array<u32>, read>, index: u32) -> u32 {
  let word = (*data)[index >> 2u];
  return (word >> ((index & 3u) * 8u)) & 255u;
}

fn signed_byte_at(data: ptr<storage, array<u32>, read>, index: u32) -> f32 {
  let value = byte_at(data, index);
  if (value >= 128u) {
    return f32(i32(value) - 256);
  }
  return f32(value);
}

fn hash_u32(x0: u32) -> u32 {
  var x = x0;
  x = x ^ (x >> 16u);
  x = x * 0x7feb352du;
  x = x ^ (x >> 15u);
  x = x * 0x846ca68bu;
  x = x ^ (x >> 16u);
  return x;
}

fn rand01(seed: u32) -> f32 {
  return f32(hash_u32(seed) & 0x00ffffffu) / 16777216.0;
}

fn interpolated_cell_coordinate(cell: u32, cellCount: u32, seedA: u32, seedB: u32) -> f32 {
  var coordinate = f32(cell) + rand01(seedA) + rand01(seedB) - 0.5;
  let limit = f32(cellCount);
  if (coordinate < 0.0) {
    coordinate = -coordinate;
  }
  if (coordinate > limit) {
    coordinate = 2.0 * limit - coordinate;
  }
  return clamp(coordinate, 0.0, limit);
}

fn cell_weight(cell: u32) -> f32 {
  var value = f32(byte_at(&meanPacked, cell)) / 255.0;
  for (var r = 0u; r < params.rank; r = r + 1u) {
    value = value + signed_byte_at(&basisPacked, r * params.totalCells + cell) * basisScale[r] * coeff[r];
  }
  value = clamp(value, 0.0, 1.0);
  return max(value, params.supportFloor);
}

@compute @workgroup_size(128)
fn sample_points(@builtin(global_invocation_id) gid: vec3<u32>) {
  let outIndex = gid.x;
  if (outIndex >= params.targetCount) {
    return;
  }
  if (params.supportCount == 0u) {
    return;
  }

  let pointIndex = params.pointOffset + outIndex;
  let seedBase = params.seed ^ (pointIndex * 747796405u);
  var bestCell = supportIndex[hash_u32(seedBase) % params.supportCount];
  var bestWeight = -1.0;
  var accepted = false;
  for (var attempt = 0u; attempt < params.attempts; attempt = attempt + 1u) {
    let s = seedBase + attempt * 2891336453u;
    let cell = supportIndex[hash_u32(s) % params.supportCount];
    let w = cell_weight(cell);
    if (w > bestWeight) {
      bestWeight = w;
      bestCell = cell;
    }
    if (w > 0.0 && rand01(s ^ 0xa511e9b3u) <= w) {
      bestCell = cell;
      accepted = true;
      break;
    }
  }

  let xy = params.gx * params.gy;
  let zi = bestCell / xy;
  let rem = bestCell - zi * xy;
  let yi = rem / params.gx;
  let xi = rem - yi * params.gx;
  let jitterSeed = seedBase ^ select(0x1f123bb5u, 0x9e3779b9u, accepted);
  let xCell = interpolated_cell_coordinate(xi, params.gx, jitterSeed ^ 0x11u, jitterSeed ^ 0x19u);
  let yCell = interpolated_cell_coordinate(yi, params.gy, jitterSeed ^ 0x21u, jitterSeed ^ 0x29u);
  let zCell = interpolated_cell_coordinate(zi, params.gz, jitterSeed ^ 0x41u, jitterSeed ^ 0x49u);
  let x = params.minX + xCell / f32(params.gx) * params.extentX;
  let y = params.minY + yCell / f32(params.gy) * params.extentY;
  let z = params.minZ + zCell / f32(params.gz) * params.extentZ;
  let mass = params.massMin + rand01(jitterSeed ^ 0x81u) * params.massWidth;
  points[outIndex] = vec4<f32>(x, y, z, mass);
}
`;

function packBytesToU32(bytes) {
  const out = new Uint8Array(Math.ceil(bytes.byteLength / 4) * 4);
  out.set(new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength));
  return new Uint32Array(out.buffer);
}

function makeBuffer(device, data, usage) {
  const size = Math.max(4, Math.ceil(data.byteLength / 4) * 4);
  const buffer = device.createBuffer({ size, usage, mappedAtCreation: true });
  const mapped = new Uint8Array(buffer.getMappedRange());
  mapped.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
  buffer.unmap();
  return buffer;
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

function interpolatedCellCoordinate(index, cellCount, rand) {
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

async function getDevice() {
  if (!devicePromise) {
    devicePromise = (async () => {
      if (!navigator.gpu) throw new Error('WebGPU is not available in this browser.');
      const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
      if (!adapter) throw new Error('No WebGPU adapter is available.');
      const device = await adapter.requestDevice();
      device.lost.then((info) => {
        console.warn(`WebGPU device lost: ${info.message}`);
        devicePromise = null;
        pipelinePromise = null;
      });
      return device;
    })();
  }
  return devicePromise;
}

async function getPipeline(device) {
  if (!pipelinePromise) {
    pipelinePromise = (async () => {
      const module = device.createShaderModule({ code: SHADER });
      return device.createComputePipelineAsync({
        layout: 'auto',
        compute: { module, entryPoint: 'sample_points' }
      });
    })();
  }
  return pipelinePromise;
}

function paramsArray({ artifact, range, targetCount, pointOffset, seed, attempts, supportCount }) {
  const settings = artifact.header.settings;
  const [gx, gy, gz] = settings.base_grid_shape;
  const bounds = artifact.header.dataset_bounds;
  const min = bounds[0];
  const max = bounds[1];
  const data = new ArrayBuffer(96);
  const u32 = new Uint32Array(data);
  const f32 = new Float32Array(data);
  u32[0] = settings.teacher_param_count >>> 0;
  u32[1] = gx >>> 0;
  u32[2] = gy >>> 0;
  u32[3] = gz >>> 0;
  u32[4] = settings.hyper_rank >>> 0;
  u32[5] = targetCount >>> 0;
  u32[6] = pointOffset >>> 0;
  u32[7] = seed >>> 0;
  u32[8] = attempts >>> 0;
  u32[9] = supportCount >>> 0;
  f32[10] = supportDensityFloor(range);
  f32[12] = Number(range.mass_min ?? range.mass ?? 0);
  f32[13] = Math.max(0, Number(range.mass_max ?? range.mass ?? range.mass_min ?? 0) - f32[12]);
  f32[14] = Number(min[0]);
  f32[15] = Number(min[1]);
  f32[16] = Number(min[2]);
  f32[17] = Number(max[0]) - Number(min[0]);
  f32[18] = Number(max[1]) - Number(min[1]);
  f32[19] = Number(max[2]) - Number(min[2]);
  return data;
}

export async function generateCdfV2WebGpu(artifact, plan, progress = () => {}) {
  const started = performance.now();
  const device = await getDevice();
  const pipeline = await getPipeline(device);
  const { arrays } = artifact;
  const meanPacked = packBytesToU32(arrays.mean);
  const basisPacked = packBytesToU32(arrays.basis);
  const meanBuffer = makeBuffer(device, meanPacked, GPUBufferUsage.STORAGE);
  const basisBuffer = makeBuffer(device, basisPacked, GPUBufferUsage.STORAGE);
  const basisScaleBuffer = makeBuffer(device, arrays.basis_scale, GPUBufferUsage.STORAGE);
  const maxSupportBytes = Math.max(...plan.supportIndices.map((indices) => Math.ceil(indices.byteLength / 4) * 4));
  const supportIndexBuffer = device.createBuffer({
    size: Math.max(4, maxSupportBytes),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
  });
  const coeffBuffer = device.createBuffer({
    size: Math.max(4, Math.ceil(artifact.header.settings.hyper_rank * 4 / 4) * 4),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
  });
  const paramBuffer = device.createBuffer({
    size: 96,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
  });

  const points = new Float32Array(plan.totalPoints * 4);
  const maxChunkPoints = Math.min(524288, Math.max(1, Math.floor((device.limits.maxBufferSize || 268435456) / 32)));
  const chunkBytes = maxChunkPoints * 16;
  const outputBuffer = device.createBuffer({
    size: chunkBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  });
  const readBuffer = device.createBuffer({
    size: chunkBytes,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
  });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: meanBuffer } },
      { binding: 1, resource: { buffer: basisBuffer } },
      { binding: 2, resource: { buffer: basisScaleBuffer } },
      { binding: 3, resource: { buffer: coeffBuffer } },
      { binding: 4, resource: { buffer: supportIndexBuffer } },
      { binding: 5, resource: { buffer: outputBuffer } },
      { binding: 6, resource: { buffer: paramBuffer } }
    ]
  });

  const attempts = Number(new URLSearchParams(window.location.search).get('gpuAttempts')) || 64;
  for (let rangeIndex = 0; rangeIndex < plan.ranges.length; rangeIndex++) {
    const target = plan.targets[rangeIndex];
    if (!target) continue;
    const range = plan.ranges[rangeIndex];
    const supportIndices = plan.supportIndices[rangeIndex];
    const residual = makeResidualCdf(artifact, rangeIndex);
    const residualTarget = Math.min(target, Math.round(target * residualPointFraction(range, target, residual.total)));
    const baseTarget = target - residualTarget;
    progress(`WebGPU sampling ${range.id} (${rangeIndex + 1}/${plan.ranges.length})...`);
    if (baseTarget && supportIndices.length) {
      device.queue.writeBuffer(coeffBuffer, 0, plan.coeffs[rangeIndex]);
      device.queue.writeBuffer(supportIndexBuffer, 0, supportIndices);
      let done = 0;
      while (done < baseTarget) {
        const chunkCount = Math.min(maxChunkPoints, baseTarget - done);
        const params = paramsArray({
          artifact,
          range,
          targetCount: chunkCount,
          pointOffset: done,
          seed: 0x51f15eED ^ rangeIndex ^ target,
          attempts,
          supportCount: supportIndices.length
        });
        device.queue.writeBuffer(paramBuffer, 0, params);
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(Math.ceil(chunkCount / 128));
        pass.end();
        encoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, chunkCount * 16);
        device.queue.submit([encoder.finish()]);
        await readBuffer.mapAsync(GPUMapMode.READ, 0, chunkCount * 16);
        const chunk = new Float32Array(readBuffer.getMappedRange(0, chunkCount * 16).slice(0));
        readBuffer.unmap();
        points.set(chunk, (plan.offsets[rangeIndex].start + done) * 4);
        done += chunkCount;
      }
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
      backend: 'webgpu',
      backendMs: performance.now() - started,
      attempts
    }
  };
}
