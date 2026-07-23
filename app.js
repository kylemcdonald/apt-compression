import {
  CloudRenderer,
  createSharedCamera,
  drawSpectrum,
} from "./renderer.js";

const librarySpecifier = new URL(import.meta.url).pathname.includes("/demo/")
  ? "../javascript/cpos.js"
  : "./javascript/cpos.js";
const {
  decodeCpos,
  encodePos,
  samplePosBySpectrum,
} = await import(librarySpecifier);

const elements = {
  body: document.body,
  dropZone: document.getElementById("drop-zone"),
  fileInput: document.getElementById("file-input"),
  chooseFile: document.getElementById("choose-file"),
  maxPoints: document.getElementById("max-points"),
  download: document.getElementById("download"),
  status: document.getElementById("status"),
  summary: document.getElementById("summary"),
  sourceStats: document.getElementById("source-stats"),
  cposStats: document.getElementById("cpos-stats"),
  sourceEmpty: document.getElementById("source-empty"),
  cposEmpty: document.getElementById("cpos-empty"),
  spectrum: document.getElementById("spectrum"),
  credit: document.getElementById("example-credit"),
};

const camera = createSharedCamera();
const sourceRenderer = new CloudRenderer(
  document.getElementById("source-cloud"),
  camera,
);
const cposRenderer = new CloudRenderer(
  document.getElementById("cpos-cloud"),
  camera,
);

let currentPayload = null;
let downloadName = "preview.cpos";
let lastSpectrum = null;

function formatInteger(value) {
  return new Intl.NumberFormat("en-US").format(value);
}

function formatBytes(value) {
  if (value < 1024) return `${value} B`;
  const units = ["KB", "MB", "GB"];
  let size = value;
  let unit = -1;
  do {
    size /= 1024;
    unit += 1;
  } while (size >= 1024 && unit < units.length - 1);
  return `${size.toFixed(size >= 10 ? 1 : 2)} ${units[unit]}`;
}

function setBusy(busy) {
  elements.body.classList.toggle("busy", busy);
  elements.chooseFile.disabled = busy;
  elements.maxPoints.disabled = busy;
  elements.download.disabled = busy || !currentPayload;
}

function nextPaint() {
  return new Promise((resolve) => requestAnimationFrame(() => resolve()));
}

function showDecoded(decoded) {
  cposRenderer.setPoints(decoded.points);
  elements.cposEmpty.hidden = true;
  elements.cposStats.textContent = (
    `${formatInteger(decoded.header.storedPointCount)} points · `
    + `v${decoded.header.algorithmVersion.join(".")}`
  );
  lastSpectrum = decoded;
  drawSpectrum(elements.spectrum, decoded.trueCounts, decoded.storedCounts, {
    binWidth: decoded.header.spectrumBinDa,
  });
}

async function loadExample() {
  try {
    const [metadataResponse, payloadResponse] = await Promise.all([
      fetch("data/example.json"),
      fetch("data/example.cpos"),
    ]);
    if (!metadataResponse.ok || !payloadResponse.ok) {
      throw new Error("public example is unavailable");
    }
    const metadata = await metadataResponse.json();
    const payload = await payloadResponse.arrayBuffer();
    const decoded = decodeCpos(payload);
    showDecoded(decoded);
    elements.status.textContent = `${metadata.title} public example`;
    elements.summary.textContent = (
      `${formatBytes(metadata.original_size_bytes)} → `
      + `${formatBytes(metadata.cpos_size_bytes)} · `
      + `${metadata.compression_ratio.toFixed(1)}× smaller`
    );
    elements.credit.textContent = (
      `Public example: ${metadata.title} · Zenodo 7979668 · ${metadata.license}`
    );
  } catch (error) {
    elements.status.textContent = error.message;
    elements.cposEmpty.textContent = "example unavailable";
  }
}

async function processPos(file) {
  if (!file.name.toLowerCase().endsWith(".pos")) {
    elements.status.textContent = "Choose a four-column .pos file.";
    return;
  }
  setBusy(true);
  elements.status.textContent = `Reading ${file.name}…`;
  elements.summary.textContent = "";
  await nextPaint();

  try {
    const sourceBuffer = await file.arrayBuffer();
    const maxPoints = Number(elements.maxPoints.value);
    elements.status.textContent = (
      `Encoding ${formatInteger(sourceBuffer.byteLength / 16)} points in this browser…`
    );
    await nextPaint();
    const started = performance.now();
    const payload = encodePos(sourceBuffer, { maxPoints });
    const decoded = decodeCpos(payload);
    const sourcePoints = samplePosBySpectrum(
      sourceBuffer,
      decoded.trueCounts,
      decoded.storedCounts,
    );
    const elapsed = (performance.now() - started) / 1000;

    sourceRenderer.setPoints(sourcePoints);
    elements.sourceEmpty.hidden = true;
    elements.sourceStats.textContent = (
      `${formatInteger(sourceBuffer.byteLength / 16)} points · `
      + `${formatBytes(sourceBuffer.byteLength)}`
    );
    showDecoded(decoded);
    currentPayload = payload;
    downloadName = file.name.replace(/\.pos$/i, "") + ".cpos";
    elements.status.textContent = `${file.name} encoded in ${elapsed.toFixed(2)} s`;
    elements.summary.textContent = (
      `${formatBytes(sourceBuffer.byteLength)} → ${formatBytes(payload.byteLength)} · `
      + `${(sourceBuffer.byteLength / payload.byteLength).toFixed(1)}× smaller`
    );
  } catch (error) {
    console.error(error);
    elements.status.textContent = error.message || "Encoding failed.";
  } finally {
    setBusy(false);
  }
}

elements.chooseFile.addEventListener("click", () => elements.fileInput.click());
elements.fileInput.addEventListener("change", () => {
  const [file] = elements.fileInput.files;
  if (file) processPos(file);
  elements.fileInput.value = "";
});
elements.download.addEventListener("click", () => {
  if (!currentPayload) return;
  const url = URL.createObjectURL(
    new Blob([currentPayload], { type: "application/octet-stream" }),
  );
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = downloadName;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
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
elements.dropZone.addEventListener("drop", (event) => {
  const [file] = event.dataTransfer.files;
  if (file) processPos(file);
});

window.addEventListener("resize", () => {
  if (lastSpectrum) {
    drawSpectrum(
      elements.spectrum,
      lastSpectrum.trueCounts,
      lastSpectrum.storedCounts,
      { binWidth: lastSpectrum.header.spectrumBinDa },
    );
  }
});

loadExample();
