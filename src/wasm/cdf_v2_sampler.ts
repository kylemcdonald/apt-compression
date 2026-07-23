let heapPtr: u32 = 65536;

export function wasm_reset(): void {
  heapPtr = 65536;
}

export function wasm_malloc(size: u32): u32 {
  const ptr = (heapPtr + 15) & ~15;
  const end = ptr + size;
  const neededPages = (end + 65535) >>> 16;
  const currentPages = <u32>memory.size();
  if (neededPages > currentPages) {
    memory.grow(neededPages - currentPages);
  }
  heapPtr = end;
  return ptr;
}

function hashU32(x0: u32): u32 {
  let x = x0;
  x ^= x >>> 16;
  x = x * 0x7feb352d;
  x ^= x >>> 15;
  x = x * 0x846ca68b;
  x ^= x >>> 16;
  return x;
}

function rand01(seed: u32): f32 {
  return <f32>(hashU32(seed) & 0x00ffffff) / 16777216.0;
}

function interpolatedCellCoordinate(cell: u32, cellCount: u32, a: f32, b: f32): f32 {
  let coordinate = <f32>cell + a + b - 0.5;
  const limit = <f32>cellCount;
  if (coordinate < 0.0) coordinate = -coordinate;
  if (coordinate > limit) coordinate = 2.0 * limit - coordinate;
  if (coordinate < 0.0) return 0.0;
  if (coordinate > limit) return limit;
  return coordinate;
}

function cellWeight(
  mean: u32,
  basis: u32,
  basisScale: u32,
  coeff: u32,
  totalCells: u32,
  rank: u32,
  cell: u32,
  supportFloor: f32
): f32 {
  let value = <f32>load<u8>(mean + cell) / 255.0;
  for (let r: u32 = 0; r < rank; r++) {
    const basisValue = <f32>load<i8>(basis + r * totalCells + cell);
    value += basisValue * load<f32>(basisScale + r * 4) * load<f32>(coeff + r * 4);
  }
  if (value < supportFloor) return supportFloor;
  if (value > 1.0) return 1.0;
  return value;
}

export function sample_range(
  mean: u32,
  basis: u32,
  basisScale: u32,
  coeff: u32,
  supportIndex: u32,
  out: u32,
  totalCells: u32,
  supportCount: u32,
  gx: u32,
  gy: u32,
  gz: u32,
  rank: u32,
  targetCount: u32,
  pointOffset: u32,
  seed: u32,
  attempts: u32,
  supportFloor: f32,
  massMin: f32,
  massWidth: f32,
  minX: f32,
  minY: f32,
  minZ: f32,
  extentX: f32,
  extentY: f32,
  extentZ: f32
): void {
  if (supportCount == 0) return;
  const xy = gx * gy;
  for (let i: u32 = 0; i < targetCount; i++) {
    const pointIndex = pointOffset + i;
    const seedBase = seed ^ (pointIndex * 747796405);
    let bestCell = load<u32>(supportIndex + (hashU32(seedBase) % supportCount) * 4);
    let bestWeight: f32 = -1.0;
    let accepted = false;
    for (let attempt: u32 = 0; attempt < attempts; attempt++) {
      const s = seedBase + attempt * 2891336453;
      const cell = load<u32>(supportIndex + (hashU32(s) % supportCount) * 4);
      const weight = cellWeight(mean, basis, basisScale, coeff, totalCells, rank, cell, supportFloor);
      if (weight > bestWeight) {
        bestWeight = weight;
        bestCell = cell;
      }
      if (weight > 0.0 && rand01(s ^ 0xa511e9b3) <= weight) {
        bestCell = cell;
        accepted = true;
        break;
      }
    }

    const zi = bestCell / xy;
    const rem = bestCell - zi * xy;
    const yi = rem / gx;
    const xi = rem - yi * gx;
    const jitterSeed = seedBase ^ (accepted ? 0x9e3779b9 : 0x1f123bb5);
    const base = out + i * 16;
    const xCell = interpolatedCellCoordinate(
      xi, gx, rand01(jitterSeed ^ 0x11), rand01(jitterSeed ^ 0x19));
    const yCell = interpolatedCellCoordinate(
      yi, gy, rand01(jitterSeed ^ 0x21), rand01(jitterSeed ^ 0x29));
    const zCell = interpolatedCellCoordinate(
      zi, gz, rand01(jitterSeed ^ 0x41), rand01(jitterSeed ^ 0x49));
    store<f32>(base, minX + xCell / <f32>gx * extentX);
    store<f32>(base + 4, minY + yCell / <f32>gy * extentY);
    store<f32>(base + 8, minZ + zCell / <f32>gz * extentZ);
    store<f32>(base + 12, massMin + rand01(jitterSeed ^ 0x81) * massWidth);
  }
}
