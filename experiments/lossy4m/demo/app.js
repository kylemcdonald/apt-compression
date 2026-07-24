import {
  ProvenanceCloudRenderer,
  createSharedCamera,
  drawAllocationSpectrum,
} from "./renderer.js";

const sourceMode = new URL(import.meta.url).pathname.includes(
  "/experiments/lossy4m/demo/",
);
const cp4mSpecifier = sourceMode ? "../javascript/cp4m.js" : "./cp4m.js";
const cposSpecifier = sourceMode ? "../../../javascript/cpos.js" : "./cpos.js";
const [{ decodeRetainedCp4m, expandCp4m }, { decodeCpos }] = await Promise.all([
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
  status: document.getElementById("status"),
  summary: document.getElementById("summary"),
  cposStats: document.getElementById("cpos-stats"),
  exactStats: document.getElementById("exact-stats"),
  expandedStats: document.getElementById("expanded-stats"),
  spectrum: document.getElementById("spectrum"),
  tooltip: document.getElementById("bin-tooltip"),
  metricSource: document.getElementById("metric-source"),
  metricCposSize: document.getElementById("metric-cpos-size"),
  metricCp4mSize: document.getElementById("metric-cp4m-size"),
  metricExact: document.getElementById("metric-exact"),
  metricSynth: document.getElementById("metric-synth"),
  metricBins: document.getElementById("metric-bins"),
  benchmarkSummary: document.getElementById("benchmark-summary"),
  benchmarkBody: document.getElementById("benchmark-body"),
};

const camera = createSharedCamera();
const cposRenderer = new ProvenanceCloudRenderer(
  document.getElementById("cpos-cloud"),
  camera,
  { colorMode: "mass" },
);
const exactRenderer = new ProvenanceCloudRenderer(
  document.getElementById("exact-cloud"),
  camera,
  { colorMode: "mass" },
);
const expandedRenderer = new ProvenanceCloudRenderer(
  document.getElementById("expanded-cloud"),
  camera,
  { colorMode: "provenance" },
);

let decodedCp4m = null;
let cp4mBytes = 0;
let cposBytes = 0;
let spectrumGeometry = null;

function formatInteger(value) {
  return new Intl.NumberFormat("en-US").format(value);
}

function formatBytes(value) {
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

function samplePoints(points, maximum) {
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

function redrawSpectrum() {
  if (!decodedCp4m) return;
  spectrumGeometry = drawAllocationSpectrum(
    elements.spectrum,
    decodedCp4m.trueCounts,
    decodedCp4m.storedCounts,
    { binWidth: decodedCp4m.header.spectrumBinDa },
  );
}

function updateMetrics() {
  if (!decodedCp4m) return;
  const { header, trueCounts, storedCounts } = decodedCp4m;
  let exactBins = 0;
  let activeBins = 0;
  for (let bin = 0; bin < trueCounts.length; bin += 1) {
    if (!trueCounts[bin]) continue;
    activeBins += 1;
    if (trueCounts[bin] === storedCounts[bin]) exactBins += 1;
  }
  elements.metricSource.textContent = formatInteger(header.originalPointCount);
  elements.metricCposSize.textContent = cposBytes ? formatBytes(cposBytes) : "—";
  elements.metricCp4mSize.textContent = formatBytes(cp4mBytes);
  elements.metricExact.textContent = (
    `${formatInteger(header.storedPointCount)} · `
    + `${(header.storedPointCount / header.originalPointCount * 100).toFixed(1)}%`
  );
  elements.metricSynth.textContent = formatInteger(
    header.originalPointCount - header.storedPointCount,
  );
  elements.metricBins.textContent = (
    `${formatInteger(exactBins)} / ${formatInteger(activeBins)}`
  );
}

function updateExpansion() {
  if (!decodedCp4m) return;
  const display = Number(elements.displayPoints.value);
  const expanded = expandCp4m(decodedCp4m, {
    maxPoints: display,
    noise: elements.noise.value,
  });
  expandedRenderer.setPoints(expanded.points, expanded.exact);
  expandedRenderer.setColorMode(elements.colorMode.value);
  let exactDisplay = 0;
  for (const value of expanded.exact) exactDisplay += value;
  elements.expandedStats.textContent = (
    `${formatInteger(expanded.points.length / 4)} shown · `
    + `${formatInteger(exactDisplay)} exact · `
    + `${formatInteger(expanded.points.length / 4 - exactDisplay)} synthesized`
  );
}

async function showCp4m(payload, name) {
  elements.body.classList.add("busy");
  elements.status.textContent = `Decoding ${name}…`;
  try {
    decodedCp4m = await decodeRetainedCp4m(payload);
    cp4mBytes = payload.byteLength;
    const display = Number(elements.displayPoints.value);
    exactRenderer.setPoints(samplePoints(decodedCp4m.points, display));
    elements.exactStats.textContent = (
      `${formatInteger(decodedCp4m.header.storedPointCount)} exact · `
      + `${formatBytes(payload.byteLength)}`
    );
    updateExpansion();
    redrawSpectrum();
    updateMetrics();
    elements.status.textContent = name;
    elements.summary.textContent = (
      `${formatInteger(decodedCp4m.header.originalPointCount)} source → `
      + `${formatInteger(decodedCp4m.header.storedPointCount)} exact → `
      + `${formatInteger(decodedCp4m.header.originalPointCount)} expanded`
    );
  } catch (error) {
    console.error(error);
    elements.status.textContent = error.message || "Unable to decode CP4M.";
  } finally {
    elements.body.classList.remove("busy");
  }
}

async function loadCurrentCpos() {
  const url = sourceMode
    ? "../../../demo/data/example.cpos"
    : "https://kylemcdonald.github.io/apt-cpos/data/example.cpos";
  const response = await fetch(url);
  if (!response.ok) throw new Error("current CPOS comparison is unavailable");
  const payload = await response.arrayBuffer();
  const decoded = decodeCpos(payload);
  cposBytes = payload.byteLength;
  cposRenderer.setPoints(decoded.points);
  elements.cposStats.textContent = (
    `${formatInteger(decoded.header.storedPointCount)} retained · `
    + `${formatBytes(payload.byteLength)}`
  );
}

async function loadExample() {
  try {
    const [_, response] = await Promise.all([
      loadCurrentCpos(),
      fetch("data/example.cp4m"),
    ]);
    if (!response.ok) throw new Error("CP4M example is unavailable");
    await showCp4m(await response.arrayBuffer(), "Ck10 steel · CP4M 4M example");
  } catch (error) {
    console.error(error);
    elements.status.textContent = error.message;
  }
}

async function loadBenchmark() {
  try {
    const response = await fetch("data/benchmark.json");
    if (!response.ok) throw new Error("benchmark unavailable");
    const result = await response.json();
    let cposBytesTotal = 0;
    let cp4mBytesTotal = 0;
    let cposJsWeighted = 0;
    let cp4mJsWeighted = 0;
    for (const record of result.files) {
      cposBytesTotal += record.cpos_bytes;
      cp4mBytesTotal += record.cp4m_bytes;
      cposJsWeighted += record.points * record.cpos_spatial_js;
      cp4mJsWeighted += record.points * record.cp4m_spatial_js;
      const row = document.createElement("tr");
      const values = [
        record.name,
        formatInteger(record.points),
        formatBytes(record.raw_bytes),
        formatBytes(record.cpos_bytes),
        formatBytes(record.cp4m_bytes),
        `${(record.cp4m_exact_fraction * 100).toFixed(1)}%`,
        record.cpos_spatial_js.toFixed(7),
        record.cp4m_spatial_js.toFixed(7),
      ];
      for (const [index, value] of values.entries()) {
        const cell = document.createElement(index === 0 ? "th" : "td");
        cell.textContent = value;
        if (index === 0) cell.scope = "row";
        row.append(cell);
      }
      elements.benchmarkBody.append(row);
    }
    const improvement = cposJsWeighted / cp4mJsWeighted;
    elements.benchmarkSummary.textContent = (
      `${formatBytes(cposBytesTotal)} → ${formatBytes(cp4mBytesTotal)} · `
      + `${improvement.toFixed(1)}× lower spatial JS`
    );
  } catch (error) {
    elements.benchmarkSummary.textContent = error.message;
  }
}

elements.chooseFile.addEventListener("click", () => elements.fileInput.click());
elements.fileInput.addEventListener("change", async () => {
  const [file] = elements.fileInput.files;
  if (file) await showCp4m(await file.arrayBuffer(), file.name);
  elements.fileInput.value = "";
});
for (const eventName of ["dragenter", "dragover"]) {
  elements.dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.dropZone.classList.add("dragging");
  });
}
for (const eventName of ["dragleave", "drop"]) {
  elements.dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.dropZone.classList.remove("dragging");
  });
}
elements.dropZone.addEventListener("drop", async (event) => {
  const [file] = event.dataTransfer.files;
  if (file) await showCp4m(await file.arrayBuffer(), file.name);
});
elements.displayPoints.addEventListener("change", () => {
  if (!decodedCp4m) return;
  exactRenderer.setPoints(samplePoints(
    decodedCp4m.points,
    Number(elements.displayPoints.value),
  ));
  updateExpansion();
});
elements.noise.addEventListener("change", updateExpansion);
elements.colorMode.addEventListener("change", () => {
  expandedRenderer.setColorMode(elements.colorMode.value);
});
window.addEventListener("resize", redrawSpectrum);

elements.spectrum.addEventListener("pointermove", (event) => {
  if (!decodedCp4m || !spectrumGeometry) return;
  const rect = elements.spectrum.getBoundingClientRect();
  const ratioX = elements.spectrum.width / rect.width;
  const x = (event.clientX - rect.left) * ratioX;
  const local = x - spectrumGeometry.padding.left;
  if (local < 0 || local >= spectrumGeometry.plotWidth) {
    elements.tooltip.hidden = true;
    return;
  }
  const bin = Math.min(
    spectrumGeometry.visibleBins - 1,
    Math.floor(local / spectrumGeometry.plotWidth * spectrumGeometry.visibleBins),
  );
  const exact = decodedCp4m.storedCounts[bin];
  const total = decodedCp4m.trueCounts[bin];
  const synthesized = total - exact;
  const low = bin * decodedCp4m.header.spectrumBinDa;
  const high = low + decodedCp4m.header.spectrumBinDa;
  elements.tooltip.textContent = (
    `${low.toFixed(2)}–${high.toFixed(2)} Da\n`
    + `exact  ${formatInteger(exact)}\n`
    + `synth  ${formatInteger(synthesized)}\n`
    + `total  ${formatInteger(total)}`
  );
  elements.tooltip.hidden = false;
  elements.tooltip.style.left = `${Math.min(
    rect.width - 184,
    Math.max(4, event.clientX - rect.left + 12),
  )}px`;
  elements.tooltip.style.top = `${Math.max(4, event.clientY - rect.top - 72)}px`;
});
elements.spectrum.addEventListener("pointerleave", () => {
  elements.tooltip.hidden = true;
});

Promise.all([loadExample(), loadBenchmark()]);
