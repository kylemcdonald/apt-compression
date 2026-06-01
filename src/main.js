import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { generateAllCdfV2Points } from './cdfV2.js';
import './style.css';

const API_BASE = import.meta.env.VITE_API_BASE || window.location.origin;
const MAX_RENDER_BUDGET = 16_000_000;
const DEFAULT_RENDER_BUDGET = 1_000_000;
const DEFAULT_DATASET_PREFIX = '499';
const DEFAULT_METHOD = new URLSearchParams(window.location.search).get('method') || 'cdf_v2_linear_64_10mb';
const DEFAULT_CAMERA_DIRECTION = new THREE.Vector3(0.0, -1.0, 0.42).normalize();
const ORBIT_UP = new THREE.Vector3(0.0, 0.0, 1.0);
const POINT_IMAGE_PLANE_ROTATION = Math.PI / 2;
const BOUNDS_TICK_SPACING_NM = 10;
const BOUNDS_TICK_LENGTH_SCALE = 1 / 3;

const METHOD_OPTIONS = [
  { id: 'full', label: 'Full file' },
  { id: 'cdf_v2_linear_64_10mb', label: 'CDF grid v2 linear 64-bin 10MB' }
];

const dom = {
  dataset: document.getElementById('datasetSelect'),
  method: document.getElementById('methodSelect'),
  pointSize: document.getElementById('pointSizeInput'),
  pointSizeValue: document.getElementById('pointSizeValue'),
  opacity: document.getElementById('opacityInput'),
  budget: document.getElementById('budgetInput'),
  budgetRange: document.getElementById('budgetRange'),
  colorMode: document.getElementById('colorModeSelect'),
  spectrumScale: document.getElementById('spectrumScaleSelect'),
  massMin: document.getElementById('massMinInput'),
  massMax: document.getElementById('massMaxInput'),
  binWrap: document.getElementById('binSliderWrap'),
  binSlider: document.getElementById('binSliderInput'),
  binLabel: document.getElementById('binSliderLabel'),
  status: document.getElementById('statusLine'),
  metrics: document.getElementById('metricsPanel'),
  host: document.getElementById('canvasHost'),
  notice: document.getElementById('budgetNotice'),
  spectrum: document.getElementById('spectrumCanvas')
};

const vertexShader = `
  attribute float mass;
  attribute float binIndex;
  uniform float uPointSize;
  uniform float uMassMin;
  uniform float uMassMax;
  uniform float uMassLow;
  uniform float uMassHigh;
  uniform float uColorMode;
  uniform float uBinCount;
  uniform float uOpacity;
  varying vec3 vColor;
  varying float vVisible;

  vec3 palette(float t) {
    t = clamp(t, 0.0, 1.0);
    vec3 c0 = vec3(0.50, 0.00, 1.00);
    vec3 c1 = vec3(0.00, 0.25, 1.00);
    vec3 c2 = vec3(0.00, 0.85, 1.00);
    vec3 c3 = vec3(0.20, 1.00, 0.20);
    vec3 c4 = vec3(1.00, 0.92, 0.00);
    vec3 c5 = vec3(1.00, 0.42, 0.00);
    vec3 c6 = vec3(1.00, 0.00, 0.00);
    if (t < 0.1666667) return mix(c0, c1, smoothstep(0.0, 0.1666667, t));
    if (t < 0.3333333) return mix(c1, c2, smoothstep(0.1666667, 0.3333333, t));
    if (t < 0.5000000) return mix(c2, c3, smoothstep(0.3333333, 0.5000000, t));
    if (t < 0.6666667) return mix(c3, c4, smoothstep(0.5000000, 0.6666667, t));
    if (t < 0.8333333) return mix(c4, c5, smoothstep(0.6666667, 0.8333333, t));
    return mix(c5, c6, smoothstep(0.8333333, 1.0, t));
  }

  void main() {
    vVisible = (mass >= uMassMin && mass <= uMassMax) ? 1.0 : 0.0;
    float massT = (mass - uMassLow) / max(uMassHigh - uMassLow, 0.00001);
    float binT = binIndex / max(uBinCount - 1.0, 1.0);
    vColor = palette(uColorMode < 0.5 ? massT : binT);
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = max(uPointSize, 0.25);
    gl_Position = projectionMatrix * mvPosition;
  }
`;

const fragmentShader = `
  precision highp float;
  uniform float uOpacity;
  varying vec3 vColor;
  varying float vVisible;

  void main() {
    if (vVisible < 0.5) discard;
    vec2 p = gl_PointCoord - vec2(0.5);
    if (dot(p, p) > 0.25) discard;
    gl_FragColor = vec4(vColor, uOpacity);
  }
`;

let manifest = null;
let scene;
let camera;
let renderer;
let controls;
let cloudGroup = null;
let pointCloud = null;
let pointMaterial = null;
let pointFloats = null;
let pointBinIndices = null;
let generatedSpectrumCounts = null;
let loadedPointCount = 0;
let displayedPointCount = 0;
let frontendGenerationMs = null;
let frontendGenerationBackend = null;
let massMin = 0;
let massMax = 1;
let loadToken = 0;
let loadTimer = null;
let activePointAbort = null;
let hasFitCamera = false;
let renderLoopActive = false;
let renderOnceQueued = false;
let cameraMoving = false;
let settleFrames = 0;
let boundsLine = null;
let cloudBounds = null;

function formatNumber(value) {
  return Number(value || 0).toLocaleString();
}

function formatBytes(bytes) {
  const mb = (bytes || 0) / (1024 * 1024);
  if (mb >= 1024) return `${(mb / 1024).toFixed(2)} GB`;
  return `${mb.toFixed(2)} MB`;
}

function formatRatio(value) {
  if (!Number.isFinite(value) || value <= 0) return 'n/a';
  return `${value.toFixed(2)}x`;
}

function formatError(value) {
  if (value === undefined || value === null || Number.isNaN(value)) return 'n/a';
  return value.toFixed(4);
}

function currentDataset() {
  return manifest?.datasets?.find((dataset) => dataset.id === dom.dataset.value) || null;
}

function selectedMethod() {
  const [base, variantId = ''] = dom.method.value.split(':');
  return { base, variantId };
}

function currentMethodInfo() {
  const dataset = currentDataset();
  return dataset?.methods?.[selectedMethod().base] || null;
}

function currentLevelInfo() {
  return currentMethodInfo();
}

function currentMetrics() {
  const info = currentLevelInfo();
  return info?.metrics || null;
}

function currentColorModeValue() {
  return dom.colorMode?.value === 'mass' ? 0 : 1;
}

function currentPointSize() {
  const value = Number(dom.pointSize.value);
  return Math.max(0.5, Math.min(5, Number.isFinite(value) ? value : 1.25));
}

function syncPointSizeLabel() {
  dom.pointSizeValue.textContent = currentPointSize().toFixed(2);
}

function methodRanges(dataset = currentDataset(), methodId = selectedMethod().base) {
  const method = dataset?.methods?.[methodId];
  if (!method) return [];
  if (method.ranges?.length) return method.ranges;
  return (method.variants || []).map((variant) => variant.range).filter(Boolean);
}

function datasetCdfRanges(dataset = currentDataset(), methodId = selectedMethod().base) {
  return methodRanges(dataset, methodId).length
    ? methodRanges(dataset, methodId)
    : methodRanges(dataset, 'cdf_v2_linear_64_10mb');
}

function currentCdfRanges() {
  return datasetCdfRanges(currentDataset(), selectedMethod().base);
}

function isCdfMethod(methodId = selectedMethod().base, methodInfo = currentMethodInfo()) {
  return methodId.startsWith('cdf_v2_') || Boolean(methodInfo?.cdf_v2);
}

function nearestCdfBinIndex() {
  const ranges = currentCdfRanges();
  if (!ranges.length) return 0;
  const center = (massMin + massMax) * 0.5;
  let best = 0;
  let bestDistance = Infinity;
  for (let i = 0; i < ranges.length; i++) {
    const range = ranges[i];
    const rangeCenter = Number(range.mass ?? ((range.mass_min + range.mass_max) * 0.5));
    const distance = Math.abs(center - rangeCenter);
    if (distance < bestDistance) {
      best = i;
      bestDistance = distance;
    }
  }
  return best;
}

function syncBinSlider() {
  const ranges = currentCdfRanges();
  if (!ranges.length) {
    dom.binWrap.classList.remove('visible');
    return;
  }
  const index = nearestCdfBinIndex();
  const range = ranges[index];
  dom.binWrap.classList.add('visible');
  dom.binSlider.min = '1';
  dom.binSlider.max = String(ranges.length);
  dom.binSlider.value = String(index + 1);
  dom.binLabel.textContent = `Bin ${String(index + 1).padStart(2, '0')} / ${ranges.length} (${Number(range.mass_min).toFixed(2)}-${Number(range.mass_max).toFixed(2)})`;
}

function setMassRangeFromCdfBin(index) {
  const ranges = currentCdfRanges();
  if (!ranges.length) return;
  const clamped = Math.max(0, Math.min(ranges.length - 1, index));
  const range = ranges[clamped];
  massMin = Number(range.mass_min);
  massMax = Number(range.mass_max);
  syncMassInputs();
  updateMassFilter({ syncBin: false });
  syncBinSlider();
}

function setStatus(message) {
  dom.status.textContent = message;
}

function currentBudgetLimit() {
  const dataset = currentDataset();
  const method = currentMethodInfo();
  if (method?.points) {
    return Math.max(1000, Number(method.points));
  }
  if (selectedMethod().base === 'full' && dataset) {
    return Math.max(1000, dataset.atom_count);
  }
  return MAX_RENDER_BUDGET;
}

function clampBudget(value) {
  const n = Math.floor(Number(value) || 0);
  return Math.max(1000, Math.min(currentBudgetLimit(), n));
}

function syncBudgetControls() {
  const dataset = currentDataset();
  const method = currentMethodInfo();
  const isFixedPointSet = selectedMethod().base === 'full' || Boolean(method?.raw_range);
  const limit = currentBudgetLimit();
  dom.budget.max = String(limit);
  dom.budgetRange.max = String(Math.max(10000, limit));
  dom.budget.disabled = isFixedPointSet;
  dom.budgetRange.disabled = isFixedPointSet;

  const budget = isFixedPointSet && method?.points
    ? Number(method.points)
    : selectedMethod().base === 'full' && dataset
      ? dataset.atom_count
      : clampBudget(dom.budget.value || DEFAULT_RENDER_BUDGET);
  dom.budget.value = String(budget);
  dom.budgetRange.value = String(Math.min(budget, Number(dom.budgetRange.max)));
}

function setDefaultBudgetForSelectedMethod() {
  if (!isCdfMethod()) return;
  const budget = currentBudgetLimit();
  dom.budget.value = String(budget);
  dom.budgetRange.value = String(Math.min(budget, Number(dom.budgetRange.max) || budget));
}

function initThree() {
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x070807);

  camera = new THREE.PerspectiveCamera(55, 1, 0.001, 100000);
  camera.position.set(0, -120, 80);
  camera.up.copy(ORBIT_UP);

  renderer = new THREE.WebGLRenderer({ antialias: false, powerPreference: 'high-performance' });
  renderer.sortObjects = false;
  dom.host.appendChild(renderer.domElement);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.screenSpacePanning = true;
  controls.addEventListener('start', () => {
    cameraMoving = true;
    startRenderLoop();
  });
  controls.addEventListener('end', () => {
    cameraMoving = false;
    settleFrames = 24;
    startRenderLoop();
  });
  controls.addEventListener('change', requestRender);

  scene.add(new THREE.AmbientLight(0xffffff, 0.35));

  resizeRenderer();
  window.addEventListener('resize', () => {
    resizeRenderer();
    drawSpectrum();
    requestRender();
  });

  const observer = new ResizeObserver(() => {
    resizeRenderer();
    drawSpectrum();
    requestRender();
  });
  observer.observe(dom.host);

  requestRender();
}

function resizeRenderer() {
  const rect = dom.host.getBoundingClientRect();
  const width = Math.max(1, Math.floor(rect.width));
  const height = Math.max(1, Math.floor(rect.height));
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  renderer.setPixelRatio(dpr);
  renderer.setSize(width, height, true);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
}

function requestRender() {
  if (renderLoopActive || renderOnceQueued) return;
  renderOnceQueued = true;
  requestAnimationFrame(() => {
    renderOnceQueued = false;
    updateBoundsGuide();
    renderer.render(scene, camera);
  });
}

function startRenderLoop() {
  if (renderLoopActive) return;
  renderLoopActive = true;
  requestAnimationFrame(renderWhileMoving);
}

function renderWhileMoving() {
  controls.update();
  updateBoundsGuide();
  renderer.render(scene, camera);
  if (cameraMoving || settleFrames > 0) {
    settleFrames = Math.max(0, settleFrames - 1);
    requestAnimationFrame(renderWhileMoving);
  } else {
    renderLoopActive = false;
  }
}

async function loadManifest() {
  const response = await fetch(`${API_BASE}/api/manifest`);
  if (!response.ok) {
    throw new Error(`Manifest request failed: ${response.status}`);
  }
  manifest = await response.json();
}

function populateDatasets() {
  dom.dataset.innerHTML = '';
  for (const dataset of manifest.datasets || []) {
    const option = document.createElement('option');
    option.value = dataset.id;
    option.textContent = `${dataset.name} (${formatNumber(dataset.atom_count)} atoms)`;
    dom.dataset.appendChild(option);
  }
  const preferred = (manifest.datasets || []).find((dataset) =>
    dataset.name?.startsWith(DEFAULT_DATASET_PREFIX) || dataset.id?.startsWith(DEFAULT_DATASET_PREFIX)
  );
  if (preferred) dom.dataset.value = preferred.id;
}

function populateMethods() {
  const dataset = currentDataset();
  const prior = dom.method.value;
  dom.method.innerHTML = '';
  for (const method of METHOD_OPTIONS) {
    const info = dataset?.methods?.[method.id];
    const option = document.createElement('option');
    option.value = method.id;
    option.textContent = info?.method_label || info?.label || method.label;
    option.disabled = !info || info.available === false;
    if (info?.available === false) option.textContent += ' (unavailable)';
    dom.method.appendChild(option);
  }
  const canKeep = Array.from(dom.method.options).some((option) => option.value === prior && !option.disabled);
  if (canKeep) {
    dom.method.value = prior;
  } else {
    const options = Array.from(dom.method.options);
    const preferred = options.find((option) => option.value === DEFAULT_METHOD && !option.disabled);
    const firstAvailable = options.find((option) => !option.disabled);
    if (preferred || firstAvailable) dom.method.value = (preferred || firstAvailable).value;
  }
}

function populateLevels() {
  syncBinSlider();
}

function setMassInputsFromDataset() {
  const dataset = currentDataset();
  if (!dataset) return;
  massMin = dataset.mass_range[0];
  massMax = dataset.mass_range[1];
  for (const input of [dom.massMin, dom.massMax]) {
    input.min = dataset.mass_range[0];
    input.max = dataset.mass_range[1];
  }
  syncMassInputs();
}

function syncMassInputs() {
  dom.massMin.value = massMin.toFixed(3);
  dom.massMax.value = massMax.toFixed(3);
}

function applyMassFromInputs() {
  const dataset = currentDataset();
  if (!dataset) return;
  const lo = dataset.mass_range[0];
  const hi = dataset.mass_range[1];
  let a = Number(dom.massMin.value);
  let b = Number(dom.massMax.value);
  if (!Number.isFinite(a)) a = lo;
  if (!Number.isFinite(b)) b = hi;
  a = Math.max(lo, Math.min(hi, a));
  b = Math.max(lo, Math.min(hi, b));
  if (a > b) [a, b] = [b, a];
  massMin = a;
  massMax = b;
  syncMassInputs();
  updateMassFilter();
}

function updateMassFilter(options = {}) {
  if (pointMaterial) {
    pointMaterial.uniforms.uMassMin.value = massMin;
    pointMaterial.uniforms.uMassMax.value = massMax;
  }
  displayedPointCount = countDisplayedPoints();
  if (options.syncBin !== false) syncBinSlider();
  updateMetricsPanel();
  drawSpectrum();
  requestRender();
}

function updateColorMode() {
  if (pointMaterial) {
    pointMaterial.uniforms.uColorMode.value = currentColorModeValue();
    requestRender();
  }
}

function updateOpacity() {
  if (pointMaterial) {
    pointMaterial.uniforms.uOpacity.value = Number(dom.opacity.value) || 0.94;
    requestRender();
  }
}

function countDisplayedPoints() {
  if (!pointFloats) return 0;
  let count = 0;
  for (let i = 3; i < pointFloats.length; i += 4) {
    const mass = pointFloats[i];
    if (mass >= massMin && mass <= massMax) count += 1;
  }
  return count;
}

function massToCdfBinIndex(mass, ranges) {
  if (!ranges.length) return 0;
  let lo = 0;
  let hi = ranges.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (mass > Number(ranges[mid].mass_max)) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

function computeCdfBinIndices(floats) {
  const ranges = datasetCdfRanges(currentDataset(), selectedMethod().base);
  const count = Math.floor(floats.length / 4);
  const bins = new Float32Array(count);
  if (!ranges.length) return bins;
  for (let i = 0; i < count; i++) {
    bins[i] = massToCdfBinIndex(floats[i * 4 + 3], ranges);
  }
  return bins;
}

function computeGeneratedSpectrum(floats, dataset) {
  const edges = dataset?.spectrum?.edges;
  const rawCounts = dataset?.spectrum?.counts;
  const count = Math.floor(floats.length / 4);
  if (!edges?.length || !rawCounts?.length || !count || !isCdfMethod()) return null;
  const counts = new Float64Array(rawCounts.length);
  const minMass = Number(edges[0]);
  const maxMass = Number(edges[edges.length - 1]);
  const extent = Math.max(maxMass - minMass, 1e-6);
  for (let i = 3; i < floats.length; i += 4) {
    const mass = floats[i];
    if (!Number.isFinite(mass) || mass < minMass || mass > maxMass) continue;
    const index = Math.min(counts.length - 1, Math.max(0, Math.floor(((mass - minMass) / extent) * counts.length)));
    counts[index] += 1;
  }
  const targetTotal = Number(dataset.atom_count) || count;
  const scale = targetTotal / Math.max(count, 1);
  if (scale !== 1) {
    for (let i = 0; i < counts.length; i++) counts[i] *= scale;
  }
  return counts;
}

function clearPointCloud() {
  if (cloudGroup) {
    scene.remove(cloudGroup);
    cloudGroup.traverse((child) => {
      if (child.geometry) child.geometry.dispose();
      if (child.material) child.material.dispose();
    });
  }
  cloudGroup = null;
  pointCloud = null;
  pointMaterial = null;
  pointFloats = null;
  pointBinIndices = null;
  generatedSpectrumCounts = null;
  boundsLine = null;
  cloudBounds = null;
  loadedPointCount = 0;
  displayedPointCount = 0;
  frontendGenerationMs = null;
  frontendGenerationBackend = null;
}

function createCloudGroup() {
  cloudGroup = new THREE.Group();
  cloudGroup.quaternion.setFromAxisAngle(DEFAULT_CAMERA_DIRECTION, POINT_IMAGE_PLANE_ROTATION);
  scene.add(cloudGroup);
}

function createPointCloud(floats, options = {}) {
  clearPointCloud();
  createCloudGroup();
  pointFloats = floats;
  pointBinIndices = options.binIndices || null;
  frontendGenerationMs = options.frontendGenerationMs ?? null;
  loadedPointCount = Math.floor(floats.length / 4);
  generatedSpectrumCounts = computeGeneratedSpectrum(floats, currentDataset());

  const interleaved = new THREE.InterleavedBuffer(floats, 4);
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.InterleavedBufferAttribute(interleaved, 3, 0, false));
  geometry.setAttribute('mass', new THREE.InterleavedBufferAttribute(interleaved, 1, 3, false));
  const binIndices = pointBinIndices && pointBinIndices.length === loadedPointCount
    ? pointBinIndices
    : computeCdfBinIndices(floats);
  geometry.setAttribute('binIndex', new THREE.BufferAttribute(binIndices, 1));

  const dataset = currentDataset();
  pointMaterial = new THREE.ShaderMaterial({
    uniforms: {
      uPointSize: { value: currentPointSize() },
      uMassMin: { value: massMin },
      uMassMax: { value: massMax },
      uMassLow: { value: dataset?.mass_range?.[0] ?? massMin },
      uMassHigh: { value: dataset?.mass_range?.[1] ?? massMax },
      uColorMode: { value: currentColorModeValue() },
      uBinCount: { value: options.binCount || datasetCdfRanges(dataset).length || 1 },
      uOpacity: { value: Number(dom.opacity.value) || 0.94 }
    },
    vertexShader,
    fragmentShader,
    depthTest: false,
    depthWrite: false,
    transparent: true
  });

  pointCloud = new THREE.Points(geometry, pointMaterial);
  geometry.computeBoundingBox();
  if (geometry.boundingBox) {
    const center = geometry.boundingBox.getCenter(new THREE.Vector3());
    pointCloud.position.set(-center.x, -center.y, -center.z);
    cloudBounds = new THREE.Box3(
      geometry.boundingBox.min.clone().sub(center),
      geometry.boundingBox.max.clone().sub(center)
    );
  }
  cloudGroup.add(pointCloud);
  createBoundsGuide();
  fitCameraToCloud(geometry);
  updateMassFilter();
}

function createBoundsGuide() {
  if (!cloudBounds) return;
  boundsLine = new THREE.LineSegments(
    new THREE.BufferGeometry(),
    new THREE.LineBasicMaterial({
      color: 0xd8dfda,
      transparent: true,
      opacity: 0.72,
      depthTest: false
    })
  );
  boundsLine.frustumCulled = false;
  cloudGroup.add(boundsLine);
  updateBoundsGuide();
}

function pushSegment(vertices, a, b) {
  vertices.push(a.x, a.y, a.z, b.x, b.y, b.z);
}

function updateBoundsGuide() {
  if (!boundsLine || !cloudBounds || !camera) return;
  const min = cloudBounds.min;
  const max = cloudBounds.max;
  cloudGroup.updateMatrixWorld(true);
  const cameraPos = cloudGroup.worldToLocal(camera.position.clone());
  const corners = [];
  for (const x of [min.x, max.x]) {
    for (const y of [min.y, max.y]) {
      for (const z of [min.z, max.z]) {
        corners.push(new THREE.Vector3(x, y, z));
      }
    }
  }
  let front = corners[0];
  let bestDistance = cameraPos.distanceToSquared(front);
  for (const corner of corners.slice(1)) {
    const distance = cameraPos.distanceToSquared(corner);
    if (distance < bestDistance) {
      front = corner;
      bestDistance = distance;
    }
  }

  const vertices = [];
  const ranges = [
    { axis: 'x', min: min.x, max: max.x },
    { axis: 'y', min: min.y, max: max.y },
    { axis: 'z', min: min.z, max: max.z }
  ];
  const tickSize = Math.max(4, cloudBounds.getSize(new THREE.Vector3()).length() * 0.012) * BOUNDS_TICK_LENGTH_SCALE;
  for (const range of ranges) {
    const start = front.clone();
    const end = front.clone();
    end[range.axis] = Math.abs(front[range.axis] - range.min) < Math.abs(front[range.axis] - range.max) ? range.max : range.min;
    pushSegment(vertices, start, end);

    const length = Math.abs(end[range.axis] - start[range.axis]);
    const steps = Math.floor(length / BOUNDS_TICK_SPACING_NM);
    const direction = Math.sign(end[range.axis] - start[range.axis]) || 1;
    const tickAxis = range.axis === 'x' ? 'y' : 'x';
    const tickDirection = Math.abs(front[tickAxis] - min[tickAxis]) < Math.abs(front[tickAxis] - max[tickAxis]) ? 1 : -1;
    for (let i = 1; i <= steps; i++) {
      const tickStart = start.clone();
      tickStart[range.axis] += direction * i * BOUNDS_TICK_SPACING_NM;
      const tickEnd = tickStart.clone();
      tickEnd[tickAxis] += tickDirection * tickSize;
      pushSegment(vertices, tickStart, tickEnd);
    }
  }
  boundsLine.geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
  boundsLine.geometry.computeBoundingSphere();
}

function fitCameraToCloud(geometry) {
  if (!geometry || hasFitCamera) return;
  geometry.computeBoundingSphere();
  const sphere = geometry.boundingSphere;
  if (!sphere) return;

  const radius = Math.max(sphere.radius, 1);
  const verticalFov = THREE.MathUtils.degToRad(camera.fov);
  const horizontalFov = 2 * Math.atan(Math.tan(verticalFov / 2) * camera.aspect);
  const fitFov = Math.max(0.001, Math.min(verticalFov, horizontalFov));
  const distance = (radius * 1.18) / Math.sin(fitFov / 2);

  camera.position.copy(DEFAULT_CAMERA_DIRECTION.clone().multiplyScalar(distance));
  camera.up.copy(ORBIT_UP);
  camera.near = Math.max(distance - radius * 3.0, radius / 1000.0, 0.001);
  camera.far = distance + radius * 3.0;
  camera.lookAt(0, 0, 0);
  camera.updateProjectionMatrix();
  controls.target.set(0, 0, 0);
  controls.update();
  hasFitCamera = true;
}

function selectedPointBudget() {
  const dataset = currentDataset();
  const method = currentMethodInfo();
  const fixedPoints = method?.raw_range && method?.points;
  const budget = fixedPoints
    ? Number(method.points)
    : selectedMethod().base === 'full' && dataset
      ? dataset.atom_count
      : clampBudget(dom.budget.value);
  syncBudgetControls();
  return budget;
}

function scheduleLoad(delay = 250) {
  clearTimeout(loadTimer);
  loadTimer = setTimeout(() => {
    loadPoints().catch((error) => {
      if (error.name === 'AbortError') return;
      console.error(error);
      setStatus(error.message);
    });
  }, delay);
}

async function loadPoints() {
  const dataset = currentDataset();
  const selected = selectedMethod();
  const method = selected.base;
  const methodInfo = currentMethodInfo();
  if (!dataset || !methodInfo || methodInfo.available === false) {
    clearPointCloud();
    setStatus(methodInfo?.notes || 'Selected method is unavailable.');
    updateMetricsPanel();
    drawSpectrum();
    return;
  }

  const token = ++loadToken;
  const budget = selectedPointBudget();
  if (activePointAbort) activePointAbort.abort();
  const controller = new AbortController();
  activePointAbort = controller;
  if (methodInfo.cdf_v2 && methodInfo.frontend_generated) {
    const started = performance.now();
    setStatus('Unpacking CDF v2 grids locally...');
    const generated = await generateAllCdfV2Points(methodInfo, budget, API_BASE, (message) => {
      if (token === loadToken) setStatus(message);
    });
    if (activePointAbort === controller) activePointAbort = null;
    if (token !== loadToken) return;
    createPointCloud(generated.points, {
      binIndices: generated.bins,
      binCount: generated.offsets.length
    });
    const elapsedMs = performance.now() - started;
    frontendGenerationMs = elapsedMs;
    frontendGenerationBackend = generated.timings?.backend || null;
    updateMetricsPanel();
    const backendLabel = frontendGenerationBackend ? ` via ${frontendGenerationBackend}` : '';
    setStatus(`Generated ${generated.offsets.length} CDF v2 clouds locally${backendLabel} (${formatNumber(loadedPointCount)} visible-source atoms) in ${elapsedMs.toFixed(0)} ms.`);
    updateBudgetNotice();
    return;
  }

  const url = new URL(`/api/points/${encodeURIComponent(dataset.id)}/${encodeURIComponent(method)}`, API_BASE);
  url.searchParams.set('budget', String(budget));
  if (selected.variantId) {
    url.searchParams.set('variant', selected.variantId);
  }

  const selectedLabel = dom.method.selectedOptions[0]?.textContent || METHOD_OPTIONS.find((x) => x.id === method)?.label || method;
  const fullLabel = method === 'full' || methodInfo.raw_range ? ` all ${formatNumber(budget)}` : '';
  setStatus(`Loading${fullLabel} ${selectedLabel} points...`);
  const started = performance.now();
  const response = await fetch(url, { signal: controller.signal });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Point request failed: ${response.status}`);
  }
  const buffer = await response.arrayBuffer();
  if (activePointAbort === controller) activePointAbort = null;
  if (token !== loadToken) return;
  createPointCloud(new Float32Array(buffer));
  const elapsed = (performance.now() - started) / 1000;
  setStatus(`Loaded ${formatNumber(loadedPointCount)} atoms in ${elapsed.toFixed(2)} s.`);
  updateBudgetNotice();
}

function updateBudgetNotice() {
  const dataset = currentDataset();
  if (!dataset || !loadedPointCount) {
    dom.notice.classList.remove('visible');
    dom.notice.textContent = '';
    return;
  }
  if (loadedPointCount < dataset.atom_count) {
    const method = selectedMethod().base === 'full' || currentMethodInfo()?.raw_range ? 'raw POS sample' : 'representative/reconstructed set';
    dom.notice.textContent = `Showing ${formatNumber(loadedPointCount)} ${method} atoms from ${formatNumber(dataset.atom_count)} source atoms. Increase render budget for more points.`;
    dom.notice.classList.add('visible');
  } else {
    dom.notice.classList.remove('visible');
    dom.notice.textContent = '';
  }
}

function updateMetricsPanel() {
  const dataset = currentDataset();
  const method = currentMethodInfo();
  const info = currentLevelInfo();
  const metrics = info?.metrics || null;
  dom.metrics.innerHTML = '';
  if (!dataset || !info) return;

  const rows = [
    ['Raw size', formatBytes(info.raw_size_bytes || dataset.raw_size_bytes)],
    ['Compressed size', formatBytes(info.compressed_size_bytes || method?.compressed_size_bytes || 0)],
    ['Compression ratio', formatRatio(info.compression_ratio || method?.compression_ratio)],
    ['Loaded atoms', formatNumber(loadedPointCount)],
    ['Displayed atoms', formatNumber(displayedPointCount)],
    ['Frontend generation', frontendGenerationMs === null ? 'n/a' : `${frontendGenerationMs.toFixed(0)} ms`],
    ['Generation backend', frontendGenerationBackend || 'n/a'],
    ['Preprocess time', `${(info.preprocess_sec || method?.preprocess_sec || 0).toFixed(2)} s`],
    ['Reconstruction time', `${(info.reconstruction_sec || method?.reconstruction_sec || 0).toFixed(2)} s`],
    ['Mass spectrum error', formatError(metrics?.mass_spectrum_error)],
    ['Z-profile error', formatError(metrics?.z_profile_error)],
    ['Radial profile error', formatError(metrics?.radial_profile_error)],
    ['Spatial/spectral error', formatError(metrics?.spatial_spectral_error)]
  ];
  const extraMetrics = [
    ['Log MSE', metrics?.log_mse],
    ['Positive log MSE', metrics?.positive_log_mse],
    ['Zero log MSE', metrics?.zero_log_mse],
    ['Relative count L1', metrics?.relative_count_l1],
    ['Predicted count ratio', metrics?.predicted_count_ratio],
    ['Hi-res log MSE', metrics?.hires_log_mse],
    ['Hi-res positive MSE', metrics?.hires_positive_log_mse],
    ['Hi-res count L1', metrics?.hires_relative_count_l1],
    ['Hi-res edge MSE', metrics?.hires_edge_log_mse],
    ['Hi-res gradient ratio', metrics?.hires_gradient_energy_ratio],
    ['Balanced val MSE', metrics?.best_balanced_val_mse],
    ['Parameters', metrics?.param_count]
  ];
  for (const [key, value] of extraMetrics) {
    if (value === undefined || value === null || Number.isNaN(value)) continue;
    rows.push([key, key === 'Parameters' ? formatNumber(value) : formatError(Number(value))]);
  }
  if (method?.available === false || info?.available === false) {
    rows.push(['Notes', method?.notes || info?.notes || 'Unavailable']);
  }
  for (const [key, value] of rows) {
    const dt = document.createElement('dt');
    const dd = document.createElement('dd');
    dt.textContent = key;
    dd.textContent = value;
    dom.metrics.append(dt, dd);
  }
}

function resizeSpectrumCanvas() {
  const rect = dom.spectrum.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.floor(rect.width * dpr));
  const height = Math.max(1, Math.floor(rect.height * dpr));
  if (dom.spectrum.width !== width || dom.spectrum.height !== height) {
    dom.spectrum.width = width;
    dom.spectrum.height = height;
  }
  return { width, height, dpr };
}

function massToX(mass, plot, minMass, maxMass) {
  return plot.left + ((mass - minMass) / Math.max(maxMass - minMass, 1e-6)) * plot.width;
}

function xToMass(x, plot, minMass, maxMass) {
  const t = (x - plot.left) / Math.max(plot.width, 1);
  return minMass + Math.max(0, Math.min(1, t)) * (maxMass - minMass);
}

function drawSpectrum() {
  const dataset = currentDataset();
  const { width, height, dpr } = resizeSpectrumCanvas();
  const ctx = dom.spectrum.getContext('2d');
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = '#111312';
  ctx.fillRect(0, 0, width, height);
  if (!dataset) return;

  const edges = dataset.spectrum.edges;
  const counts = dataset.spectrum.counts;
  const overlay = isCdfMethod() ? generatedSpectrumCounts : null;
  const useLog = dom.spectrumScale?.value !== 'linear';
  const plot = {
    left: 42 * dpr,
    top: 10 * dpr,
    width: width - 54 * dpr,
    height: height - 34 * dpr
  };
  const minMass = edges[0];
  const maxMass = edges[edges.length - 1];
  const maxCount = Math.max(1, ...counts, ...(overlay || []));
  const logMax = Math.log10(maxCount + 1);
  const scaleCount = (value) => {
    if (useLog) return Math.log10(value + 1) / logMax;
    return value / maxCount;
  };

  ctx.strokeStyle = '#303532';
  ctx.lineWidth = 1 * dpr;
  ctx.strokeRect(plot.left, plot.top, plot.width, plot.height);

  const binWidth = plot.width / counts.length;
  ctx.fillStyle = 'rgba(225, 230, 226, 0.58)';
  for (let i = 0; i < counts.length; i++) {
    const h = scaleCount(counts[i]) * plot.height;
    const x = plot.left + i * binWidth;
    ctx.fillRect(x, plot.top + plot.height - h, Math.max(1, binWidth), h);
  }

  if (overlay && overlay.length === counts.length) {
    ctx.beginPath();
    ctx.strokeStyle = '#e5b45a';
    ctx.lineWidth = 1.4 * dpr;
    for (let i = 0; i < overlay.length; i++) {
      const x = plot.left + (i + 0.5) * binWidth;
      const y = plot.top + plot.height - scaleCount(overlay[i]) * plot.height;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  if (overlay && overlay.length === counts.length) {
    ctx.font = `${11 * dpr}px ui-sans-serif, system-ui`;
    ctx.textAlign = 'left';
    ctx.fillStyle = 'rgba(225, 230, 226, 0.72)';
    ctx.fillText('raw', plot.left + 8 * dpr, plot.top + 14 * dpr);
    ctx.fillStyle = '#e5b45a';
    ctx.fillText('generated', plot.left + 42 * dpr, plot.top + 14 * dpr);
  }

  const x1 = massToX(massMin, plot, minMass, maxMass);
  const x2 = massToX(massMax, plot, minMass, maxMass);
  ctx.fillStyle = 'rgba(89, 186, 169, 0.16)';
  ctx.fillRect(x1, plot.top, x2 - x1, plot.height);
  ctx.strokeStyle = '#59baa9';
  ctx.lineWidth = 2 * dpr;
  ctx.beginPath();
  ctx.moveTo(x1, plot.top);
  ctx.lineTo(x1, plot.top + plot.height);
  ctx.moveTo(x2, plot.top);
  ctx.lineTo(x2, plot.top + plot.height);
  ctx.stroke();

  ctx.fillStyle = '#9ea8a1';
  ctx.font = `${11 * dpr}px ui-sans-serif, system-ui`;
  ctx.textAlign = 'left';
  ctx.fillText(minMass.toFixed(1), plot.left, height - 8 * dpr);
  ctx.textAlign = 'right';
  ctx.fillText(maxMass.toFixed(1), plot.left + plot.width, height - 8 * dpr);
  ctx.textAlign = 'left';
  ctx.fillText(useLog ? 'log count' : 'linear count', 8 * dpr, plot.top + 10 * dpr);
}

function setupSpectrumDrag() {
  let drag = null;
  const handleRadius = 9;

  dom.spectrum.addEventListener('pointerdown', (event) => {
    const dataset = currentDataset();
    if (!dataset) return;
    const { dpr } = resizeSpectrumCanvas();
    const rect = dom.spectrum.getBoundingClientRect();
    const x = (event.clientX - rect.left) * dpr;
    const edges = dataset.spectrum.edges;
    const plot = {
      left: 42 * dpr,
      top: 10 * dpr,
      width: dom.spectrum.width - 54 * dpr,
      height: dom.spectrum.height - 34 * dpr
    };
    const minMass = edges[0];
    const maxMass = edges[edges.length - 1];
    const minX = massToX(massMin, plot, minMass, maxMass);
    const maxX = massToX(massMax, plot, minMass, maxMass);
    const mass = xToMass(x, plot, minMass, maxMass);
    let mode = 'new';
    if (Math.abs(x - minX) < handleRadius * dpr) mode = 'min';
    else if (Math.abs(x - maxX) < handleRadius * dpr) mode = 'max';
    else if (x > minX && x < maxX) mode = 'range';
    drag = {
      mode,
      startMass: mass,
      startMin: massMin,
      startMax: massMax,
      plot,
      minMass,
      maxMass
    };
    dom.spectrum.setPointerCapture(event.pointerId);
  });

  dom.spectrum.addEventListener('pointermove', (event) => {
    if (!drag) return;
    const rect = dom.spectrum.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const x = (event.clientX - rect.left) * dpr;
    const mass = xToMass(x, drag.plot, drag.minMass, drag.maxMass);
    if (drag.mode === 'min') {
      massMin = Math.min(mass, massMax);
    } else if (drag.mode === 'max') {
      massMax = Math.max(mass, massMin);
    } else if (drag.mode === 'range') {
      const width = drag.startMax - drag.startMin;
      const delta = mass - drag.startMass;
      let nextMin = drag.startMin + delta;
      let nextMax = drag.startMax + delta;
      if (nextMin < drag.minMass) {
        nextMin = drag.minMass;
        nextMax = nextMin + width;
      }
      if (nextMax > drag.maxMass) {
        nextMax = drag.maxMass;
        nextMin = nextMax - width;
      }
      massMin = nextMin;
      massMax = nextMax;
    } else {
      massMin = Math.min(drag.startMass, mass);
      massMax = Math.max(drag.startMass, mass);
    }
    syncMassInputs();
    updateMassFilter();
  });

  const end = (event) => {
    if (!drag) return;
    drag = null;
    try {
      dom.spectrum.releasePointerCapture(event.pointerId);
    } catch {
      // Pointer capture may already be released by the browser.
    }
  };
  dom.spectrum.addEventListener('pointerup', end);
  dom.spectrum.addEventListener('pointercancel', end);
}

function setupEvents() {
  dom.dataset.addEventListener('change', () => {
    hasFitCamera = false;
    populateMethods();
    populateLevels();
    setMassInputsFromDataset();
    syncBinSlider();
    syncBudgetControls();
    setDefaultBudgetForSelectedMethod();
    updateMetricsPanel();
    drawSpectrum();
    scheduleLoad(0);
  });

  dom.method.addEventListener('change', () => {
    populateLevels();
    syncBudgetControls();
    setDefaultBudgetForSelectedMethod();
    updateMetricsPanel();
    drawSpectrum();
    scheduleLoad(0);
  });

  dom.pointSize.addEventListener('input', () => {
    syncPointSizeLabel();
    if (pointMaterial) {
      pointMaterial.uniforms.uPointSize.value = currentPointSize();
      requestRender();
    }
  });

  dom.opacity.addEventListener('input', updateOpacity);

  dom.budget.addEventListener('change', () => {
    const budget = selectedPointBudget();
    dom.budgetRange.value = String(budget);
    scheduleLoad();
  });

  dom.budgetRange.addEventListener('input', () => {
    dom.budget.value = dom.budgetRange.value;
  });
  dom.budgetRange.addEventListener('change', () => scheduleLoad());

  dom.binSlider.addEventListener('input', () => {
    const selected = selectedMethod();
    if (!currentCdfRanges().length) return;
    if (selected.variantId) {
      dom.method.value = selected.base;
      populateLevels();
      syncBudgetControls();
      scheduleLoad(0);
    }
    setMassRangeFromCdfBin(Math.floor(Number(dom.binSlider.value) || 1) - 1);
  });

  dom.colorMode.addEventListener('change', updateColorMode);
  dom.spectrumScale.addEventListener('change', drawSpectrum);
  dom.massMin.addEventListener('change', applyMassFromInputs);
  dom.massMax.addEventListener('change', applyMassFromInputs);
  setupSpectrumDrag();
}

async function main() {
  initThree();
  setupEvents();
  try {
    await loadManifest();
    if (!manifest.datasets?.length) {
      setStatus('No datasets in manifest. Run preprocess.py.');
      return;
    }
    populateDatasets();
    populateMethods();
    populateLevels();
    syncPointSizeLabel();
    setMassInputsFromDataset();
    syncBinSlider();
    syncBudgetControls();
    setDefaultBudgetForSelectedMethod();
    updateMetricsPanel();
    drawSpectrum();
    await loadPoints();
  } catch (error) {
    console.error(error);
    setStatus(error.message);
  }
}

main();
