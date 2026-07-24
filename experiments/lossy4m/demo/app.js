import {
  PointCloudRenderer,
  createSharedCamera,
} from "./renderer.js";
import { SpectrumView } from "./spectrum.js";

const sourceMode = new URL(import.meta.url).pathname.includes(
  "/experiments/lossy4m/demo/",
);
const cp4mSpecifier = sourceMode ? "../javascript/cp4m.js" : "./cp4m.js";
const cposSpecifier = sourceMode ? "../../../javascript/cpos.js" : "./cpos.js";
const [{ inspectCp4m }, { inspectCpos }] = await Promise.all([
  import(cp4mSpecifier),
  import(cposSpecifier),
]);

const elements = {
  body: document.body,
  dropZone: document.getElementById("drop-zone"),
  fileInput: document.getElementById("file-input"),
  chooseFile: document.getElementById("choose-file"),
  displayPoints: document.getElementById("display-points"),
  noise: document.getElementById("noise"),
  colorMode: document.getElementById("color-mode"),
  opacity: document.getElementById("opacity"),
  opacityValue: document.getElementById("opacity-value"),
  pointSize: document.getElementById("point-size"),
  pointSizeValue: document.getElementById("point-size-value"),
  additive: document.getElementById("additive"),
  perspective: document.getElementById("perspective"),
  resetCamera: document.getElementById("reset-camera"),
  status: document.getElementById("status"),
  progress: document.getElementById("progress"),
  summary: document.getElementById("summary"),
  originalStats: document.getElementById("original-stats"),
  cposStats: document.getElementById("cpos-stats"),
  cp4mStats: document.getElementById("cp4m-stats"),
  originalEmpty: document.getElementById("original-empty"),
  downloadCpos: document.getElementById("download-cpos"),
  downloadCp4m: document.getElementById("download-cp4m"),
  spectrum: document.getElementById("spectrum"),
  tooltip: document.getElementById("bin-tooltip"),
  clearSpectrum: document.getElementById("clear-spectrum"),
  metricOriginalSize: document.getElementById("metric-original-size"),
  metricOriginalDetail: document.getElementById("metric-original-detail"),
  metricCposSize: document.getElementById("metric-cpos-size"),
  metricCposDetail: document.getElementById("metric-cpos-detail"),
  metricCp4mSize: document.getElementById("metric-cp4m-size"),
  metricCp4mDetail: document.getElementById("metric-cp4m-detail"),
};

const camera = createSharedCamera();
const originalRenderer = new PointCloudRenderer(
  document.getElementById("original-cloud"),
  camera,
  { colorMode: "mass" },
);
const cposRenderer = new PointCloudRenderer(
  document.getElementById("cpos-cloud"),
  camera,
  { colorMode: "mass" },
);
const cp4mRenderer = new PointCloudRenderer(
  document.getElementById("cp4m-cloud"),
  camera,
  { colorMode: "provenance" },
);
const renderers = [originalRenderer, cposRenderer, cp4mRenderer];

const spectrum = new SpectrumView(elements.spectrum, elements.tooltip, {
  onSelection: (selection) => {
    for (const renderer of renderers) renderer.setMassWindow(selection);
    elements.clearSpectrum.disabled = !selection;
  },
});

let worker = null;
let activeJob = 0;
let busy = false;
let expansionRequest = 0;
let currentName = "Ck10 steel";
let originalBytes = 0;
let originalPoints = null;
let cposPoints = null;
let cposPayload = null;
let cp4mPayload = null;
let cposHeader = null;
let cp4mHeader = null;
let cp4mExactShown = 0;
let cp4mShown = 0;

function formatInteger(value) {
  return new Intl.NumberFormat("en-US").format(value);
}

function formatBytes(value) {
  if (!Number.isFinite(value) || value <= 0) return "—";
  if (value < 1024) return `${value} B`;
  const units = ["KiB", "MiB", "GiB"];
  let size = value;
  let unit = -1;
  do {
    size /= 1024;
    unit += 1;
  } while (size >= 1024 && unit < units.length - 1);
  return `${size.toFixed(size >= 10 ? 1 : 2)} ${units[unit]}`;
}

function formatOpacity(value) {
  const percent = value * 100;
  if (percent >= 1) return `${percent.toFixed(percent >= 10 ? 0 : 1)}%`;
  return `${percent.toFixed(3)}%`;
}

function samplePoints(points, maximum) {
  if (!points) return null;
  const sourceCount = points.length / 4;
  if (sourceCount <= maximum) return points;
  const output = new Float32Array(maximum * 4);
  for (let index = 0; index < maximum; index += 1) {
    const source = Math.min(
      sourceCount - 1,
      Math.floor((index + 0.5) * sourceCount / maximum),
    );
    output.set(points.subarray(source * 4, source * 4 + 4), index * 4);
  }
  return output;
}

function sum(values) {
  let result = 0;
  for (const value of values) result += value;
  return result;
}

function rebinCounts(values, sourceWidth, targetWidth, targetLength) {
  const output = new Uint32Array(targetLength);
  for (let bin = 0; bin < values.length; bin += 1) {
    const target = Math.max(
      0,
      Math.min(
        targetLength - 1,
        Math.floor((bin + 0.5) * sourceWidth / targetWidth),
      ),
    );
    output[target] += values[bin];
  }
  return output;
}

function baseName(name) {
  return name.replace(/\.(pos|cpos|cp4m)$/i, "") || "preview";
}

function download(payload, extension) {
  if (!payload) return;
  const url = URL.createObjectURL(new Blob([payload], {
    type: "application/octet-stream",
  }));
  const link = document.createElement("a");
  link.href = url;
  link.download = `${baseName(currentName)}.${extension}`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1_000);
}

function commonBounds() {
  if (!cp4mHeader) return null;
  return [
    cp4mHeader.minimum.slice(0, 3),
    cp4mHeader.maximum.slice(0, 3),
  ];
}

function applyBounds(renderer = null) {
  const bounds = commonBounds();
  if (!bounds) return;
  const targets = renderer ? [renderer] : renderers;
  for (const target of targets) target.setBounds(bounds[0], bounds[1]);
}

function refreshLocalClouds() {
  const maximum = Number(elements.displayPoints.value);
  if (originalPoints) {
    const displayed = samplePoints(originalPoints, maximum);
    originalRenderer.setPoints(displayed);
    elements.originalEmpty.hidden = true;
  } else {
    originalRenderer.clear();
    elements.originalEmpty.hidden = false;
  }
  if (cposPoints) cposRenderer.setPoints(samplePoints(cposPoints, maximum));
  applyBounds();
  updateStats();
}

function setExpanded(points, exact) {
  cp4mShown = points.length / 4;
  cp4mExactShown = sum(exact);
  cp4mRenderer.setPoints(points, exact);
  cp4mRenderer.setColorMode(elements.colorMode.value);
  applyBounds(cp4mRenderer);
  updateStats();
}

function updateStats() {
  if (!cposHeader || !cp4mHeader) return;
  const originalCount = cp4mHeader.originalPointCount;
  const localOriginalShown = originalPoints
    ? Math.min(originalPoints.length / 4, Number(elements.displayPoints.value))
    : 0;
  const localCposShown = cposPoints
    ? Math.min(cposPoints.length / 4, Number(elements.displayPoints.value))
    : 0;
  elements.originalStats.textContent = originalPoints
    ? (
      `${formatInteger(originalCount)} points · ${formatBytes(originalBytes)}`
      + ` · ${formatInteger(localOriginalShown)} shown`
    )
    : `${formatInteger(originalCount)} points · ${formatBytes(originalBytes)} · not bundled`;
  elements.cposStats.textContent = (
    `${formatInteger(cposHeader.storedPointCount)} retained`
    + ` · ${formatBytes(cposPayload?.byteLength || 0)}`
    + ` · ${formatInteger(localCposShown)} shown`
  );
  elements.cp4mStats.textContent = (
    `${formatInteger(cp4mHeader.storedPointCount)} exact seeds`
    + ` · ${formatBytes(cp4mPayload?.byteLength || 0)}`
    + ` · ${formatInteger(cp4mExactShown)} exact / `
    + `${formatInteger(cp4mShown - cp4mExactShown)} synth shown`
  );
  elements.metricOriginalSize.textContent = formatBytes(originalBytes);
  elements.metricOriginalDetail.textContent = (
    `${formatInteger(originalCount)} × 16-byte float32 records`
  );
  elements.metricCposSize.textContent = formatBytes(cposPayload?.byteLength || 0);
  elements.metricCposDetail.textContent = (
    `${(originalBytes / cposPayload.byteLength).toFixed(1)}× smaller`
    + ` · ${(cposHeader.storedPointCount / originalCount * 100).toFixed(1)}% retained`
  );
  elements.metricCp4mSize.textContent = formatBytes(cp4mPayload?.byteLength || 0);
  elements.metricCp4mDetail.textContent = (
    `${(originalBytes / cp4mPayload.byteLength).toFixed(1)}× smaller`
    + ` · ${(cp4mHeader.storedPointCount / originalCount * 100).toFixed(1)}% exact seeds`
  );
  elements.summary.textContent = (
    `${currentName} · ${formatInteger(originalCount)} source points`
  );
}

function updateAppearance() {
  const opacity = Number(elements.opacity.value);
  const pointSize = Number(elements.pointSize.value);
  elements.opacityValue.textContent = formatOpacity(opacity);
  elements.pointSizeValue.textContent = pointSize.toFixed(2);
  for (const renderer of renderers) {
    renderer.setAppearance({
      opacity,
      pointSize,
      additive: elements.additive.checked,
      perspective: elements.perspective.checked,
    });
  }
}

function setBusy(value) {
  busy = value;
  elements.body.classList.toggle("busy", value);
  elements.progress.hidden = !value;
  if (!value) elements.progress.value = 0;
}

function showError(message) {
  console.error(message);
  setBusy(false);
  elements.status.textContent = message;
}

function installResult(data) {
  currentName = data.name;
  originalBytes = data.originalBytes;
  originalPoints = data.originalPoints;
  cposPoints = data.cposPoints;
  cposPayload = data.cposPayload;
  cp4mPayload = data.cp4mPayload;
  cposHeader = inspectCpos(cposPayload);
  cp4mHeader = inspectCp4m(cp4mPayload);
  refreshLocalClouds();
  setExpanded(data.expandedPoints, data.expandedExact);

  const cposRebinned = rebinCounts(
    data.cposStoredCounts,
    cposHeader.spectrumBinDa,
    cp4mHeader.spectrumBinDa,
    data.cp4mTrueCounts.length,
  );
  spectrum.setData({
    binWidth: cp4mHeader.spectrumBinDa,
    trueCounts: data.cp4mTrueCounts,
    cposStoredCounts: cposRebinned,
    cp4mStoredCounts: data.cp4mStoredCounts,
  });
  elements.clearSpectrum.disabled = true;
  elements.downloadCpos.disabled = false;
  elements.downloadCp4m.disabled = false;
  elements.status.textContent = data.mode === "encoded"
    ? `${data.name} encoded entirely in this browser`
    : `${data.name} example`;
  setBusy(false);
}

function handleWorkerMessage({ data }) {
  if (data.type === "progress") {
    if (data.jobId !== activeJob) return;
    elements.progress.value = data.fraction;
    elements.status.textContent = `${data.stage} · ${Math.round(data.fraction * 100)}%`;
    return;
  }
  if (data.type === "result") {
    if (data.jobId !== activeJob) return;
    try {
      installResult(data);
    } catch (error) {
      showError(error.message || String(error));
    }
    return;
  }
  if (data.type === "expanded") {
    if (data.requestId !== expansionRequest) return;
    setExpanded(data.points, data.exact);
    elements.status.textContent = `${currentName} · CP4M preview updated`;
    return;
  }
  if (data.type === "error") {
    if (
      data.jobId != null
      && data.jobId !== activeJob
    ) return;
    if (
      data.requestId != null
      && data.requestId !== expansionRequest
    ) return;
    showError(data.message || "codec worker failed");
  }
}

function createWorker() {
  const next = new Worker(new URL("./encoder-worker.js", import.meta.url), {
    type: "module",
  });
  next.addEventListener("message", handleWorkerMessage);
  next.addEventListener("error", (event) => {
    showError(event.message || "unable to start codec worker");
  });
  return next;
}

function replaceWorker() {
  if (worker) worker.terminate();
  worker = createWorker();
  expansionRequest += 1;
}

async function encodeFile(file) {
  if (!/\.pos$/i.test(file.name)) {
    showError("Drop a .pos file containing 16-byte big-endian float32 records.");
    return;
  }
  if (!file.size || file.size % 16 !== 0) {
    showError("POS file size must be a non-zero multiple of 16 bytes.");
    return;
  }
  if (busy) replaceWorker();
  activeJob += 1;
  const jobId = activeJob;
  setBusy(true);
  elements.status.textContent = `reading ${file.name}…`;
  try {
    const posPayload = await file.arrayBuffer();
    if (jobId !== activeJob) return;
    worker.postMessage({
      type: "encode",
      jobId,
      name: file.name,
      posPayload,
      displayPoints: Number(elements.displayPoints.value),
      noise: elements.noise.value,
    }, [posPayload]);
  } catch (error) {
    showError(error.message || String(error));
  }
}

function requestExpansion() {
  refreshLocalClouds();
  if (!cp4mHeader || !worker) return;
  expansionRequest += 1;
  elements.status.textContent = "expanding CP4M preview…";
  worker.postMessage({
    type: "expand",
    requestId: expansionRequest,
    displayPoints: Number(elements.displayPoints.value),
    noise: elements.noise.value,
  });
}

async function loadExample() {
  activeJob += 1;
  const jobId = activeJob;
  setBusy(true);
  try {
    const cposUrl = sourceMode
      ? "../../../demo/data/example.cpos"
      : "./data/example.cpos";
    const [cposResponse, cp4mResponse, metadataResponse] = await Promise.all([
      fetch(cposUrl),
      fetch("./data/example.cp4m"),
      fetch("./data/example.json"),
    ]);
    if (!cposResponse.ok || !cp4mResponse.ok || !metadataResponse.ok) {
      throw new Error("public comparison data is unavailable");
    }
    const [exampleCpos, exampleCp4m, metadata] = await Promise.all([
      cposResponse.arrayBuffer(),
      cp4mResponse.arrayBuffer(),
      metadataResponse.json(),
    ]);
    if (jobId !== activeJob) return;
    worker.postMessage({
      type: "decode-demo",
      jobId,
      name: metadata.title || "Ck10 steel",
      originalBytes: metadata.source_points * 16,
      cposPayload: exampleCpos,
      cp4mPayload: exampleCp4m,
      displayPoints: Number(elements.displayPoints.value),
      noise: elements.noise.value,
    }, [exampleCpos, exampleCp4m]);
  } catch (error) {
    showError(error.message || String(error));
  }
}

elements.chooseFile.addEventListener("click", () => elements.fileInput.click());
elements.fileInput.addEventListener("change", () => {
  const [file] = elements.fileInput.files;
  if (file) encodeFile(file);
  elements.fileInput.value = "";
});

for (const eventName of ["dragenter", "dragover"]) {
  document.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.dropZone.classList.add("dragging");
  });
}
document.addEventListener("dragleave", (event) => {
  if (!event.relatedTarget) elements.dropZone.classList.remove("dragging");
});
document.addEventListener("drop", (event) => {
  event.preventDefault();
  elements.dropZone.classList.remove("dragging");
  const [file] = event.dataTransfer.files;
  if (file) encodeFile(file);
});

elements.displayPoints.addEventListener("change", requestExpansion);
elements.noise.addEventListener("change", requestExpansion);
elements.colorMode.addEventListener("change", () => {
  cp4mRenderer.setColorMode(elements.colorMode.value);
});
for (const input of [
  elements.opacity,
  elements.pointSize,
  elements.additive,
  elements.perspective,
]) {
  input.addEventListener("input", updateAppearance);
  input.addEventListener("change", updateAppearance);
}
elements.resetCamera.addEventListener("click", () => camera.reset());
elements.clearSpectrum.addEventListener("click", () => {
  spectrum.setSelection(null, { notify: true });
});
elements.downloadCpos.addEventListener("click", () => download(cposPayload, "cpos"));
elements.downloadCp4m.addEventListener("click", () => download(cp4mPayload, "cp4m"));

worker = createWorker();
updateAppearance();
loadExample();
