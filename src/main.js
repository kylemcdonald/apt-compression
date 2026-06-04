import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { generateAllCdfV2Points } from './cdfV2.js';
import './style.css';

const STATIC_DEMO = import.meta.env.VITE_STATIC_DEMO === 'true';
const APP_BASE = new URL(import.meta.env.BASE_URL || '/', window.location.origin).href;
const API_BASE = STATIC_DEMO ? APP_BASE : (import.meta.env.VITE_API_BASE || window.location.origin);
const MAX_RENDER_BUDGET = 16_000_000;
const DEFAULT_RENDER_BUDGET = 1_000_000;
const DEFAULT_DATASET_PREFIX = '499';
const COMPRESSED_METHOD = new URLSearchParams(window.location.search).get('method') || 'cdf_v2_linear_64_10mb';
const DEFAULT_CAMERA_DIRECTION = new THREE.Vector3(0.0, -1.0, 0.42).normalize();
const ORBIT_UP = new THREE.Vector3(0.0, 0.0, 1.0);
const POINT_IMAGE_PLANE_ROTATION = Math.PI / 2;
const BOUNDS_TICK_SPACING_NM = 10;
const BOUNDS_TICK_LENGTH_SCALE = 1 / 3;
const DEFAULT_OPACITY = 0.10;
const AUTO_POINT_SIZE_BASE = 0.5;
const AUTO_POINT_SIZE_EXPONENT = 1.5;
const AUTO_POINT_SIZE_MAX = 10;

const dom = {
  app: document.getElementById('app'),
  panelContent: document.getElementById('panelContent'),
  paneToggle: document.getElementById('paneToggle'),
  dataset: document.getElementById('datasetSelect'),
  compressed: document.getElementById('compressedToggle'),
  pointSize: document.getElementById('pointSizeInput'),
  pointSizeValue: document.getElementById('pointSizeValue'),
  opacity: document.getElementById('opacityInput'),
  opacityValue: document.getElementById('opacityValue'),
  budget: document.getElementById('budgetInput'),
  budgetRange: document.getElementById('budgetRange'),
  spectrumScale: document.getElementById('spectrumScaleSelect'),
  massMin: document.getElementById('massMinInput'),
  massMax: document.getElementById('massMaxInput'),
  binWrap: document.getElementById('binSliderWrap'),
  binSlider: document.getElementById('binSliderInput'),
  binLabel: document.getElementById('binSliderLabel'),
  status: document.getElementById('statusLine'),
  metricsToggle: document.getElementById('metricsToggle'),
  metrics: document.getElementById('metricsPanel'),
  host: document.getElementById('canvasHost'),
  notice: document.getElementById('budgetNotice'),
  spectrum: document.getElementById('spectrumCanvas'),
  spectrumHint: document.getElementById('spectrumHint')
};

const vertexShader = `
  attribute float mass;
  attribute float binIndex;
  uniform float uPointSize;
  uniform float uMassMin;
  uniform float uMassMax;
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
    float binT = binIndex / max(uBinCount - 1.0, 1.0);
    vColor = palette(binT);
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
let massRangeActive = false;
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
let axisLabelElements = [];
let fitCameraDistance = null;
let manualPointSize = null;
let manualOpacity = null;
let opacityRenderedPointCount = null;
let pointSizeDragging = false;
let metricsExpanded = false;
let paneCollapsed = true;

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
  if (STATIC_DEMO) return { base: COMPRESSED_METHOD, variantId: '' };
  return { base: dom.compressed.checked ? COMPRESSED_METHOD : 'full', variantId: '' };
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

function autoPointSize() {
  const scale = Math.max(1, currentViewScale());
  const size = AUTO_POINT_SIZE_BASE * Math.pow(scale, AUTO_POINT_SIZE_EXPONENT);
  return Math.max(AUTO_POINT_SIZE_BASE, Math.min(AUTO_POINT_SIZE_MAX, size));
}

function currentPointSize() {
  return manualPointSize ?? autoPointSize();
}

function clampPointSize(value) {
  const size = Number(value);
  return Math.max(AUTO_POINT_SIZE_BASE, Math.min(AUTO_POINT_SIZE_MAX, Number.isFinite(size) ? size : autoPointSize()));
}

function renderedPointCountForOpacity() {
  if (!pointFloats && !loadedPointCount) return null;
  return Math.max(0, displayedPointCount);
}

function autoOpacity() {
  const total = Number(currentDataset()?.atom_count || loadedPointCount || displayedPointCount || 0);
  const rendered = renderedPointCountForOpacity();
  if (!total || rendered === null) return DEFAULT_OPACITY;
  if (rendered <= 0) return 1;
  const fraction = Math.min(1, Math.max(rendered / total, 0));
  return Math.min(1, Math.max(DEFAULT_OPACITY, DEFAULT_OPACITY / Math.max(fraction, 1e-6)));
}

function currentOpacity() {
  return manualOpacity ?? autoOpacity();
}

function clampOpacity(value) {
  const opacity = Number(value);
  return Math.max(0.02, Math.min(1, Number.isFinite(opacity) ? opacity : autoOpacity()));
}

function syncPointSizeLabel() {
  const size = currentPointSize();
  dom.pointSize.value = size.toFixed(2);
  dom.pointSizeValue.textContent = size.toFixed(2);
}

function syncOpacityLabel() {
  const opacity = currentOpacity();
  dom.opacity.value = opacity.toFixed(2);
  dom.opacityValue.textContent = `${Math.round(opacity * 100)}%`;
}

function currentViewScale() {
  if (!camera || !controls || !fitCameraDistance) return 1;
  const distance = camera.position.distanceTo(controls.target);
  if (!Number.isFinite(distance) || distance <= 0) return 1;
  return fitCameraDistance / distance;
}

function syncViewScaleLabel() {
  // View scale is internal; point size follows it unless temporarily overridden.
}

function syncPointSizeForViewScale({ clearManual = false } = {}) {
  if (clearManual) manualPointSize = null;
  const size = currentPointSize();
  syncPointSizeLabel();
  if (pointMaterial) {
    pointMaterial.uniforms.uPointSize.value = size;
  }
}

function syncOpacityForRenderedPoints({ forceAuto = false } = {}) {
  const rendered = renderedPointCountForOpacity();
  if (forceAuto || rendered !== opacityRenderedPointCount) {
    manualOpacity = null;
    opacityRenderedPointCount = rendered;
  }
  syncOpacityLabel();
  if (pointMaterial) {
    pointMaterial.uniforms.uOpacity.value = currentOpacity();
  }
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
  if (!ranges.length || !massRangeActive) {
    dom.binWrap.classList.remove('visible');
    return;
  }
  const index = nearestCdfBinIndex();
  const range = ranges[index];
  dom.binWrap.classList.add('visible');
  dom.binSlider.min = '1';
  dom.binSlider.max = String(ranges.length);
  dom.binSlider.value = String(index + 1);
  dom.binLabel.textContent = `${String(index + 1).padStart(2, '0')} / ${ranges.length} (${Number(range.mass_min).toFixed(2)}-${Number(range.mass_max).toFixed(2)})`;
}

function setMassRangeFromCdfBin(index) {
  const ranges = currentCdfRanges();
  if (!ranges.length) return;
  const clamped = Math.max(0, Math.min(ranges.length - 1, index));
  const range = ranges[clamped];
  massMin = Number(range.mass_min);
  massMax = Number(range.mass_max);
  massRangeActive = true;
  syncMassInputs();
  updateMassFilter({ syncBin: false });
  syncBinSlider();
}

function setStatus(message, visible = false) {
  dom.status.textContent = message;
  dom.status.classList.toggle('visible', Boolean(message && visible));
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
  controls.addEventListener('change', () => {
    syncViewScaleLabel();
    syncPointSizeForViewScale();
    requestRender();
  });
  renderer.domElement.addEventListener('dblclick', (event) => {
    event.preventDefault();
    resetCamera();
  });

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
  syncViewScaleLabel();
  syncPointSizeForViewScale();
  updateAxisLabelScreenPositions();
}

function requestRender() {
  if (renderLoopActive || renderOnceQueued) return;
  renderOnceQueued = true;
  requestAnimationFrame(() => {
    renderOnceQueued = false;
    updateBoundsGuide();
    updateAxisLabelScreenPositions();
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
  updateAxisLabelScreenPositions();
  renderer.render(scene, camera);
  if (cameraMoving || settleFrames > 0) {
    settleFrames = Math.max(0, settleFrames - 1);
    requestAnimationFrame(renderWhileMoving);
  } else {
    renderLoopActive = false;
  }
}

async function loadManifest() {
  const url = STATIC_DEMO
    ? new URL('artifacts/manifest.json', API_BASE)
    : new URL('/api/manifest', API_BASE);
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Manifest request failed: ${response.status}`);
  }
  manifest = await response.json();
}

function populateDatasets() {
  dom.dataset.innerHTML = '';
  const datasets = STATIC_DEMO
    ? (manifest.datasets || []).filter((dataset) => dataset.name?.toLowerCase().startsWith(DEFAULT_DATASET_PREFIX))
    : (manifest.datasets || []);
  for (const dataset of datasets) {
    const option = document.createElement('option');
    option.value = dataset.id;
    option.textContent = `${dataset.name} (${formatNumber(dataset.atom_count)} atoms)`;
    dom.dataset.appendChild(option);
  }
  const preferred = datasets.find((dataset) =>
    dataset.name?.toLowerCase().startsWith(DEFAULT_DATASET_PREFIX)
    && dataset.name?.toLowerCase().endsWith('.pos')
  ) || datasets.find((dataset) =>
    dataset.name?.toLowerCase().startsWith(DEFAULT_DATASET_PREFIX)
    || dataset.id?.toLowerCase().startsWith(DEFAULT_DATASET_PREFIX)
  );
  if (preferred) dom.dataset.value = preferred.id;
}

function setDefaultControlValues() {
  dom.opacity.value = DEFAULT_OPACITY.toFixed(2);
  dom.pointSize.disabled = false;
  syncOpacityForRenderedPoints({ forceAuto: true });
  syncPointSizeForViewScale({ clearManual: true });
}

function syncCompressedToggle() {
  if (STATIC_DEMO) {
    dom.compressed.checked = true;
    dom.compressed.disabled = true;
    return;
  }
  const dataset = currentDataset();
  const compressed = dataset?.methods?.[COMPRESSED_METHOD];
  const canLoadCompressed = Boolean(compressed && compressed.available !== false);
  dom.compressed.disabled = !canLoadCompressed;
  if (!canLoadCompressed) dom.compressed.checked = false;
}

function populateLevels() {
  syncBinSlider();
}

function setMassInputsFromDataset() {
  const dataset = currentDataset();
  if (!dataset) return;
  massMin = dataset.mass_range[0];
  massMax = dataset.mass_range[1];
  massRangeActive = false;
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
  massRangeActive = !(a <= lo && b >= hi);
  syncMassInputs();
  updateMassFilter();
}

function updateMassFilter(options = {}) {
  if (pointMaterial) {
    pointMaterial.uniforms.uMassMin.value = massMin;
    pointMaterial.uniforms.uMassMax.value = massMax;
  }
  displayedPointCount = countDisplayedPoints();
  syncOpacityForRenderedPoints();
  if (options.syncBin !== false) syncBinSlider();
  updateMetricsPanel();
  drawSpectrum();
  requestRender();
}

function updateOpacity() {
  manualOpacity = clampOpacity(dom.opacity.value);
  opacityRenderedPointCount = renderedPointCountForOpacity();
  syncOpacityLabel();
  if (pointMaterial) {
    pointMaterial.uniforms.uOpacity.value = currentOpacity();
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
      if (child.material?.map) child.material.map.dispose();
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
  for (const item of axisLabelElements) item.element.remove();
  axisLabelElements = [];
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
      uBinCount: { value: options.binCount || datasetCdfRanges(dataset).length || 1 },
      uOpacity: { value: currentOpacity() }
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
  createAxisLabelElements();
  updateBoundsGuide();
}

function pushSegment(vertices, a, b) {
  vertices.push(a.x, a.y, a.z, b.x, b.y, b.z);
}

function createAxisLabelElements() {
  axisLabelElements = ['x', 'y', 'z'].map((axis) => {
    const element = document.createElement('div');
    element.className = 'axis-label';
    element.hidden = true;
    dom.host.appendChild(element);
    return {
      axis,
      element,
      localPosition: new THREE.Vector3(),
      visible: false
    };
  });
}

function updateAxisLabelElement(label, text, position) {
  label.element.textContent = text;
  label.localPosition.copy(position);
  label.visible = true;
  label.element.hidden = false;
}

function updateAxisLabelScreenPositions() {
  if (!camera || !cloudGroup || !axisLabelElements.length) return;
  const rect = dom.host.getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  cloudGroup.updateMatrixWorld(true);
  for (const label of axisLabelElements) {
    if (!label.visible) {
      label.element.hidden = true;
      continue;
    }
    const position = cloudGroup.localToWorld(label.localPosition.clone()).project(camera);
    const onscreen = position.z >= -1 && position.z <= 1;
    label.element.hidden = !onscreen;
    if (!onscreen) continue;
    const x = (position.x * 0.5 + 0.5) * rect.width;
    const y = (-position.y * 0.5 + 0.5) * rect.height;
    label.element.style.transform = `translate3d(${x}px, ${y}px, 0) translate(-50%, -50%)`;
  }
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
  const labels = [];
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
      if (i === steps) {
        const labelPosition = tickEnd.clone();
        labelPosition[tickAxis] += tickDirection * tickSize * 2.1;
        labels.push({
          text: `${i * BOUNDS_TICK_SPACING_NM} nm`,
          position: labelPosition
        });
      }
    }
  }
  boundsLine.geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
  boundsLine.geometry.computeBoundingSphere();
  for (let i = 0; i < axisLabelElements.length; i++) {
    const label = labels[i];
    if (label) updateAxisLabelElement(axisLabelElements[i], label.text, label.position);
    else {
      axisLabelElements[i].visible = false;
      axisLabelElements[i].element.hidden = true;
    }
  }
}

function fitCameraToCloud(geometry, force = false) {
  if (!geometry || (!force && hasFitCamera)) return;
  geometry.computeBoundingSphere();
  const sphere = geometry.boundingSphere;
  if (!sphere) return;

  const radius = Math.max(sphere.radius, 1);
  const verticalFov = THREE.MathUtils.degToRad(camera.fov);
  const horizontalFov = 2 * Math.atan(Math.tan(verticalFov / 2) * camera.aspect);
  const fitFov = Math.max(0.001, Math.min(verticalFov, horizontalFov));
  const distance = (radius * 1.18) / Math.sin(fitFov / 2);
  fitCameraDistance = distance;

  camera.position.copy(DEFAULT_CAMERA_DIRECTION.clone().multiplyScalar(distance));
  camera.up.copy(ORBIT_UP);
  camera.near = Math.max(distance - radius * 3.0, radius / 1000.0, 0.001);
  camera.far = distance + radius * 3.0;
  camera.lookAt(0, 0, 0);
  camera.updateProjectionMatrix();
  controls.target.set(0, 0, 0);
  controls.update();
  syncViewScaleLabel();
  syncPointSizeForViewScale({ clearManual: true });
  hasFitCamera = true;
}

function resetCamera() {
  if (!pointCloud?.geometry) return;
  fitCameraToCloud(pointCloud.geometry, true);
  settleFrames = 12;
  requestRender();
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
      setStatus(error.message, true);
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
    setStatus(methodInfo?.notes || 'Selected method is unavailable.', true);
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
    setStatus('');
    const generated = await generateAllCdfV2Points(methodInfo, budget, API_BASE, () => {});
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
    setStatus('');
    updateBudgetNotice();
    return;
  }

  const url = new URL(`/api/points/${encodeURIComponent(dataset.id)}/${encodeURIComponent(method)}`, API_BASE);
  url.searchParams.set('budget', String(budget));
  if (selected.variantId) {
    url.searchParams.set('variant', selected.variantId);
  }

  setStatus('');
  const response = await fetch(url, { signal: controller.signal });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Point request failed: ${response.status}`);
  }
  const buffer = await response.arrayBuffer();
  if (activePointAbort === controller) activePointAbort = null;
  if (token !== loadToken) return;
  createPointCloud(new Float32Array(buffer));
  setStatus('');
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
    dom.notice.textContent = `Only showing ${formatNumber(loadedPointCount)} representative atoms of ${formatNumber(dataset.atom_count)} total.`;
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

function syncMetricsVisibility() {
  dom.metrics.hidden = !metricsExpanded;
  dom.metricsToggle.setAttribute('aria-expanded', String(metricsExpanded));
  dom.metricsToggle.classList.toggle('expanded', metricsExpanded);
}

function syncPaneVisibility() {
  dom.app.classList.toggle('pane-collapsed', paneCollapsed);
  dom.paneToggle.textContent = paneCollapsed ? '>' : '<';
  dom.paneToggle.title = paneCollapsed ? 'Show controls' : 'Hide controls';
  dom.paneToggle.setAttribute('aria-label', paneCollapsed ? 'Show controls' : 'Hide controls');
  dom.paneToggle.setAttribute('aria-expanded', String(!paneCollapsed));
  dom.panelContent.inert = paneCollapsed;
  dom.panelContent.setAttribute('aria-hidden', String(paneCollapsed));
  requestAnimationFrame(() => {
    resizeRenderer();
    drawSpectrum();
    requestRender();
  });
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

function fillSpectrumArea(ctx, values, plot, scaleCount, color) {
  if (!values?.length) return;
  const baseline = plot.top + plot.height;
  const binWidth = plot.width / values.length;
  ctx.beginPath();
  ctx.moveTo(plot.left, baseline);
  for (let i = 0; i < values.length; i++) {
    const x = plot.left + (i + 0.5) * binWidth;
    const y = baseline - scaleCount(values[i]) * plot.height;
    ctx.lineTo(x, y);
  }
  ctx.lineTo(plot.left + plot.width, baseline);
  ctx.closePath();
  ctx.fillStyle = color;
  ctx.fill();
}

function formatSpectrumTick(value) {
  const abs = Math.abs(value);
  if (abs >= 1_000_000) return `${(value / 1_000_000).toFixed(abs >= 10_000_000 ? 0 : 1)}M`;
  if (abs >= 1_000) return `${(value / 1_000).toFixed(abs >= 10_000 ? 0 : 1)}K`;
  if (abs >= 10) return value.toFixed(0);
  return value.toFixed(1);
}

function drawSpectrumTicks(ctx, plot, dpr, minMass, maxMass, maxCount, logMax, useLog) {
  ctx.save();
  ctx.font = `${10 * dpr}px ui-sans-serif, system-ui`;
  ctx.fillStyle = '#8f9892';
  ctx.strokeStyle = 'rgba(143, 152, 146, 0.32)';
  ctx.lineWidth = 1 * dpr;

  const xTicks = 5;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  for (let i = 0; i < xTicks; i++) {
    const t = i / (xTicks - 1);
    const x = plot.left + t * plot.width;
    const mass = minMass + t * (maxMass - minMass);
    ctx.beginPath();
    ctx.moveTo(x, plot.top + plot.height);
    ctx.lineTo(x, plot.top + plot.height + 4 * dpr);
    ctx.stroke();
    ctx.fillText(formatSpectrumTick(mass), x, plot.top + plot.height + 7 * dpr);
  }

  const yTicks = [0.25, 0.5, 0.75, 1.0];
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (const t of yTicks) {
    const y = plot.top + plot.height - t * plot.height;
    const count = useLog ? Math.pow(10, logMax * t) - 1 : maxCount * t;
    ctx.beginPath();
    ctx.moveTo(plot.left - 4 * dpr, y);
    ctx.lineTo(plot.left, y);
    ctx.stroke();
    ctx.fillText(formatSpectrumTick(count), plot.left - 7 * dpr, y);
  }
  ctx.restore();
}

function drawSpectrumLegend(ctx, plot, dpr, hasGenerated) {
  if (!hasGenerated) return;
  ctx.save();
  ctx.font = `${11 * dpr}px ui-sans-serif, system-ui`;
  ctx.textAlign = 'left';
  ctx.textBaseline = 'middle';
  const items = [
    { label: 'raw', color: 'rgba(225, 230, 226, 0.72)' },
    { label: 'generated', color: '#e5b45a' }
  ];
  const square = 8 * dpr;
  const rowHeight = 15 * dpr;
  const labelWidth = Math.max(...items.map((item) => ctx.measureText(item.label).width));
  const x = plot.left + plot.width - labelWidth - square - 18 * dpr;
  let y = plot.top + 12 * dpr;
  for (const item of items) {
    ctx.fillStyle = item.color;
    ctx.fillRect(x, y - square / 2, square, square);
    ctx.fillStyle = '#dce1dd';
    ctx.fillText(item.label, x + square + 6 * dpr, y);
    y += rowHeight;
  }
  ctx.restore();
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
    left: 52 * dpr,
    top: 10 * dpr,
    width: width - 64 * dpr,
    height: height - 38 * dpr
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

  fillSpectrumArea(ctx, counts, plot, scaleCount, 'rgba(225, 230, 226, 0.54)');

  if (overlay && overlay.length === counts.length) {
    fillSpectrumArea(ctx, overlay, plot, scaleCount, 'rgba(229, 180, 90, 0.42)');
  }

  drawSpectrumTicks(ctx, plot, dpr, minMass, maxMass, maxCount, logMax, useLog);
  drawSpectrumLegend(ctx, plot, dpr, overlay && overlay.length === counts.length);

  if (massRangeActive) {
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
  }

  dom.spectrumHint.textContent = massRangeActive ? `${massMin.toFixed(2)}-${massMax.toFixed(2)} Da` : '';

  ctx.fillStyle = '#9ea8a1';
  ctx.font = `${11 * dpr}px ui-sans-serif, system-ui`;
  ctx.textAlign = 'left';
  ctx.fillText(useLog ? 'log count' : 'linear count', 8 * dpr, plot.top + 10 * dpr);
}

function setupSpectrumDrag() {
  let drag = null;
  const handleRadius = 9;

  const contextForEvent = (event) => {
    const dataset = currentDataset();
    if (!dataset) return null;
    const { dpr } = resizeSpectrumCanvas();
    const rect = dom.spectrum.getBoundingClientRect();
    const x = (event.clientX - rect.left) * dpr;
    const edges = dataset.spectrum.edges;
    const plot = {
      left: 52 * dpr,
      top: 10 * dpr,
      width: dom.spectrum.width - 64 * dpr,
      height: dom.spectrum.height - 38 * dpr
    };
    const minMass = edges[0];
    const maxMass = edges[edges.length - 1];
    const minX = massToX(massMin, plot, minMass, maxMass);
    const maxX = massToX(massMax, plot, minMass, maxMass);
    const mass = xToMass(x, plot, minMass, maxMass);
    let mode = 'new';
    if (massRangeActive) {
      const borderTolerance = 2 * dpr;
      const leftEdge = Math.abs(x - minX) <= borderTolerance || (x < minX && minX - x <= handleRadius * dpr);
      const rightEdge = Math.abs(x - maxX) <= borderTolerance || (x > maxX && x - maxX <= handleRadius * dpr);
      if (leftEdge) mode = 'min';
      else if (rightEdge) mode = 'max';
      else if (x > minX && x < maxX) mode = 'range';
    }
    return { dpr, x, mass, mode, plot, minMass, maxMass };
  };

  const syncSpectrumCursor = (event) => {
    const context = contextForEvent(event);
    if (!context || !massRangeActive) {
      dom.spectrum.style.cursor = 'crosshair';
      return;
    }
    if (context.mode === 'min' || context.mode === 'max') {
      dom.spectrum.style.cursor = 'col-resize';
    } else if (context.mode === 'range') {
      dom.spectrum.style.cursor = 'move';
    } else {
      dom.spectrum.style.cursor = 'crosshair';
    }
  };

  dom.spectrum.addEventListener('pointerdown', (event) => {
    const context = contextForEvent(event);
    if (!context) return;
    if (context.mode === 'new') {
      massRangeActive = true;
      massMin = context.mass;
      massMax = context.mass;
      syncMassInputs();
    }
    drag = {
      mode: context.mode,
      startMass: context.mass,
      startMin: massMin,
      startMax: massMax,
      plot: context.plot,
      minMass: context.minMass,
      maxMass: context.maxMass,
      moved: false
    };
    syncSpectrumCursor(event);
    dom.spectrum.setPointerCapture(event.pointerId);
  });

  dom.spectrum.addEventListener('pointermove', (event) => {
    if (!drag) {
      syncSpectrumCursor(event);
      return;
    }
    const rect = dom.spectrum.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const x = (event.clientX - rect.left) * dpr;
    const mass = xToMass(x, drag.plot, drag.minMass, drag.maxMass);
    drag.moved = true;
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
    syncSpectrumCursor(event);
  });

  const end = (event) => {
    if (!drag) return;
    const endedDrag = drag;
    drag = null;
    const minWidth = Math.max((endedDrag.maxMass - endedDrag.minMass) * 0.0005, 0.001);
    if (endedDrag.mode === 'new' && (!endedDrag.moved || Math.abs(massMax - massMin) < minWidth)) {
      massRangeActive = false;
      massMin = endedDrag.minMass;
      massMax = endedDrag.maxMass;
      syncMassInputs();
      updateMassFilter();
    }
    try {
      dom.spectrum.releasePointerCapture(event.pointerId);
    } catch {
      // Pointer capture may already be released by the browser.
    }
    syncSpectrumCursor(event);
  };
  dom.spectrum.addEventListener('pointerup', end);
  dom.spectrum.addEventListener('pointercancel', end);
  dom.spectrum.addEventListener('pointerleave', () => {
    if (!drag) dom.spectrum.style.cursor = 'crosshair';
  });
}

function setupEvents() {
  dom.paneToggle.addEventListener('click', () => {
    paneCollapsed = !paneCollapsed;
    syncPaneVisibility();
  });

  dom.dataset.addEventListener('change', () => {
    hasFitCamera = false;
    fitCameraDistance = null;
    syncViewScaleLabel();
    syncPointSizeForViewScale({ clearManual: true });
    syncCompressedToggle();
    populateLevels();
    setMassInputsFromDataset();
    syncBinSlider();
    syncBudgetControls();
    setDefaultBudgetForSelectedMethod();
    updateMetricsPanel();
    drawSpectrum();
    scheduleLoad(0);
  });

  dom.compressed.addEventListener('change', () => {
    populateLevels();
    syncBudgetControls();
    setDefaultBudgetForSelectedMethod();
    updateMetricsPanel();
    drawSpectrum();
    scheduleLoad(0);
  });

  dom.pointSize.addEventListener('input', () => {
    manualPointSize = clampPointSize(dom.pointSize.value);
    syncPointSizeLabel();
    if (pointMaterial) {
      pointMaterial.uniforms.uPointSize.value = manualPointSize;
    }
    requestRender();
  });
  dom.pointSize.addEventListener('pointerdown', () => {
    pointSizeDragging = true;
    manualPointSize = clampPointSize(dom.pointSize.value);
  });
  const finishPointSizeDrag = () => {
    pointSizeDragging = false;
  };
  dom.pointSize.addEventListener('pointerup', finishPointSizeDrag);
  dom.pointSize.addEventListener('pointercancel', finishPointSizeDrag);

  dom.opacity.addEventListener('input', updateOpacity);
  dom.metricsToggle.addEventListener('click', () => {
    metricsExpanded = !metricsExpanded;
    syncMetricsVisibility();
  });

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
      populateLevels();
      syncBudgetControls();
      scheduleLoad(0);
    }
    setMassRangeFromCdfBin(Math.floor(Number(dom.binSlider.value) || 1) - 1);
  });

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
      setStatus('No datasets in manifest. Run preprocess.py.', true);
      return;
    }
    populateDatasets();
    dom.compressed.checked = true;
    syncCompressedToggle();
    populateLevels();
    setDefaultControlValues();
    syncViewScaleLabel();
    syncMetricsVisibility();
    syncPaneVisibility();
    setMassInputsFromDataset();
    syncBinSlider();
    syncBudgetControls();
    setDefaultBudgetForSelectedMethod();
    updateMetricsPanel();
    drawSpectrum();
    await loadPoints();
  } catch (error) {
    console.error(error);
    setStatus(error.message, true);
  }
}

main();
