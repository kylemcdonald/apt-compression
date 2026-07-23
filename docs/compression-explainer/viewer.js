/* APT compression explainer v2.
   Rendering follows the uap-archive APT viewer: constant screen-space point
   size auto-scaled by zoom (0.5 * scale^1.5, clamped [0.5, 10]), alpha-blended
   points with auto opacity (0.10 / rendered-fraction), mass-rainbow palette,
   depth test off, and its damped z-up orbit/pan/dolly camera. Single bottom
   spectrum with range selection; selections show every stored point in range
   up to a 2M point budget per panel. */
"use strict";

const BUDGET = 2_000_000;
const DEFAULT_OPACITY = 0.10;
const AUTO_POINT_SIZE_BASE = 0.5;
const AUTO_POINT_SIZE_EXPONENT = 1.5;
const AUTO_POINT_SIZE_MAX = 10;
const CAMERA_FOV = 55 * Math.PI / 180;
const CAMERA_DAMPING = 0.08;
const DEFAULT_CAMERA_DIRECTION = normalize3([0, -1, 0.42]);
const DEFAULT_CAMERA_AZIMUTH = Math.atan2(
  DEFAULT_CAMERA_DIRECTION[1], DEFAULT_CAMERA_DIRECTION[0]);
const DEFAULT_CAMERA_ELEVATION = Math.asin(DEFAULT_CAMERA_DIRECTION[2]);
const POINT_IMAGE_PLANE_ROTATION = axisAngle3(
  DEFAULT_CAMERA_DIRECTION, Math.PI / 2);

const state = {
  index: [],
  summary: null,
  slug: null,
  meta: null,
  payloads: new Map(),   // `${slug}/${method}` -> parsed payload
  ranges: [],            // [[loDa, hiDa], ...]
  activeChip: null,
  sizeAuto: true,
  opacityAuto: true,
  manualSize: 0.5,
  manualOpacity: DEFAULT_OPACITY,
  cam: null,
  specView: null,        // {x0, x1}
  panels: [],
};

const tooltip = document.createElement("div");
tooltip.className = "viz-tooltip";
document.body.appendChild(tooltip);

function normalizedCloudRadius() {
  if (!state.meta?.bounds) return 0.6;
  const b = state.meta.bounds;
  const ext = [b[1][0] - b[0][0], b[1][1] - b[0][1], b[1][2] - b[0][2]];
  const maxExt = Math.max(...ext, 1e-9);
  return 0.5 * Math.hypot(...ext.map((value) => value / maxExt));
}

function primaryPanelAspect() {
  const canvas = state.panels[0]?.canvas;
  if (!canvas?.clientWidth || !canvas?.clientHeight) return 16 / 9;
  return canvas.clientWidth / canvas.clientHeight;
}

function fittedCameraDistance(aspect = primaryPanelAspect()) {
  const horizontalFov = 2 * Math.atan(Math.tan(CAMERA_FOV / 2) * Math.max(aspect, 1e-6));
  const fitFov = Math.max(0.001, Math.min(CAMERA_FOV, horizontalFov));
  return normalizedCloudRadius() * 1.18 / Math.sin(fitFov / 2);
}

function resetCam() {
  const fitDist = fittedCameraDistance();
  return {
    az: DEFAULT_CAMERA_AZIMUTH,
    el: DEFAULT_CAMERA_ELEVATION,
    dist: fitDist,
    fitDist,
    target: [0, 0, 0],
  };
}

// ---------- data ----------

async function fetchJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.json();
}

function parsePayload(buf) {
  const dv = new DataView(buf);
  const magic = dv.getUint32(0, true);
  if (magic !== 0x32545041) throw new Error("bad payload magic");
  const nbins = dv.getUint32(4, true);
  const binW = dv.getFloat32(8, true);
  const nRec = dv.getUint32(12, true);
  let off = 16;
  const trueCounts = new Uint32Array(buf, off, nbins); off += nbins * 4;
  const storedCounts = new Uint32Array(buf, off, nbins); off += nbins * 4;
  const records = new Uint16Array(buf, off, nRec * 4);
  const offsets = new Uint32Array(nbins + 1);
  for (let b = 0; b < nbins; b++) offsets[b + 1] = offsets[b] + storedCounts[b];
  return { nbins, binW, nRec, trueCounts, storedCounts, records, offsets };
}

async function loadPayload(method) {
  const key = `${state.slug}/${method}`;
  if (state.payloads.has(key)) return state.payloads.get(key);
  const r = await fetch(`data/${state.slug}/${method}.bin`);
  if (!r.ok) throw new Error(`payload ${method}: ${r.status}`);
  const payload = parsePayload(await r.arrayBuffer());
  if (state.payloads.size > 8) {
    const first = state.payloads.keys().next().value;
    state.payloads.delete(first);
  }
  state.payloads.set(key, payload);
  return payload;
}

function methodMeta(id) {
  return state.meta.methods.find((m) => m.id === id);
}

// ---------- WebGL panels (uap-archive shaders) ----------

const VSH = `
attribute vec3 aPos; attribute float aMassQ;
uniform mat4 uMVP; uniform float uPointSize;
uniform float uMassScale; uniform float uDisplayMax;
varying vec3 vColor;
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
  float mass = aMassQ * uMassScale;
  vColor = palette(mass / uDisplayMax);
  gl_Position = uMVP * vec4(aPos, 1.0);
  gl_PointSize = max(uPointSize, 0.25);
}`;
const FSH = `
precision mediump float;
uniform float uOpacity;
varying vec3 vColor;
void main() {
  vec2 p = gl_PointCoord - vec2(0.5);
  if (dot(p, p) > 0.25) discard;
  gl_FragColor = vec4(vColor, uOpacity);
}`;

function makePanel(root, tag) {
  const canvas = root.querySelector("canvas.cloud");
  const gl = canvas.getContext("webgl", {
    antialias: false, alpha: false, preserveDrawingBuffer: true,
  });
  const prog = gl.createProgram();
  for (const [type, src] of [[gl.VERTEX_SHADER, VSH], [gl.FRAGMENT_SHADER, FSH]]) {
    const sh = gl.createShader(type);
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS))
      console.error(gl.getShaderInfoLog(sh));
    gl.attachShader(prog, sh);
  }
  gl.linkProgram(prog);
  const veil = document.createElement("div");
  veil.className = "loading-veil";
  veil.style.display = "none";
  root.style.position = "relative";
  root.appendChild(veil);
  return {
    tag, root, canvas, gl, prog, veil,
    select: root.querySelector(".method-select"),
    chips: root.querySelector(".chips"),
    ptCount: root.querySelector(".pt-count"),
    methodId: null, buf: gl.createBuffer(), count: 0,
    shownTrue: 0, shownStored: 0,
    loc: {
      aPos: gl.getAttribLocation(prog, "aPos"),
      aMassQ: gl.getAttribLocation(prog, "aMassQ"),
      uMVP: gl.getUniformLocation(prog, "uMVP"),
      uPointSize: gl.getUniformLocation(prog, "uPointSize"),
      uMassScale: gl.getUniformLocation(prog, "uMassScale"),
      uDisplayMax: gl.getUniformLocation(prog, "uDisplayMax"),
      uOpacity: gl.getUniformLocation(prog, "uOpacity"),
    },
  };
}

// ---------- selection gathering ----------

function selectedBinSpans(payload) {
  const { nbins, binW } = payload;
  if (!state.ranges.length) return [[0, nbins]];
  const spans = [];
  for (const [lo, hi] of state.ranges) {
    const b0 = Math.max(0, Math.floor(lo / binW));
    const b1 = Math.min(nbins, Math.ceil(hi / binW));
    if (b1 > b0) spans.push([b0, b1]);
  }
  spans.sort((a, b) => a[0] - b[0]);
  const merged = [];
  for (const s of spans) {
    const last = merged[merged.length - 1];
    if (last && s[0] <= last[1]) last[1] = Math.max(last[1], s[1]);
    else merged.push([...s]);
  }
  return merged;
}

function gather(payload) {
  const spans = selectedBinSpans(payload);
  let stored = 0, trueN = 0;
  for (const [b0, b1] of spans) {
    stored += payload.offsets[b1] - payload.offsets[b0];
    for (let b = b0; b < b1; b++) trueN += payload.trueCounts[b];
  }
  const n = Math.min(stored, BUDGET);
  const out = new Uint16Array(n * 4);
  if (stored <= BUDGET) {
    let o = 0;
    for (const [b0, b1] of spans) {
      const r0 = payload.offsets[b0] * 4, r1 = payload.offsets[b1] * 4;
      out.set(payload.records.subarray(r0, r1), o);
      o += r1 - r0;
    }
  } else {
    const stride = stored / n;
    let o = 0, acc = 0;
    for (const [b0, b1] of spans) {
      for (let r = payload.offsets[b0]; r < payload.offsets[b1]; r++) {
        acc += 1;
        if (acc >= stride) {
          acc -= stride;
          if (o < n * 4) {
            out[o] = payload.records[r * 4];
            out[o + 1] = payload.records[r * 4 + 1];
            out[o + 2] = payload.records[r * 4 + 2];
            out[o + 3] = payload.records[r * 4 + 3];
            o += 4;
          }
        }
      }
    }
  }
  shuffleRecords(out, n);
  return { data: out, count: n, stored, trueN };
}

// records arrive sorted by mass bin; without depth testing that would paint
// high-mass points over everything, so randomize draw order
function shuffleRecords(arr, n) {
  let seed = 0x9e3779b9;
  const rand = () => {
    seed ^= seed << 13; seed ^= seed >>> 17; seed ^= seed << 5;
    return (seed >>> 0) / 4294967296;
  };
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    for (let k2 = 0; k2 < 4; k2++) {
      const t = arr[i * 4 + k2];
      arr[i * 4 + k2] = arr[j * 4 + k2];
      arr[j * 4 + k2] = t;
    }
  }
}

async function refreshPanel(panel) {
  const id = panel.methodId;
  if (!id) return;
  panel.veil.textContent = `loading ${id}…`;
  panel.veil.style.display = "flex";
  try {
    const payload = await loadPayload(id);
    const { data, count, stored, trueN } = gather(payload);
    const gl = panel.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, panel.buf);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
    panel.count = count;
    panel.shownStored = stored;
    panel.shownTrue = trueN;
    const capped = count < trueN;
    panel.ptCount.textContent =
      `${count.toLocaleString()} shown` +
      (capped ? ` of ${trueN.toLocaleString()} in range` : "");
    renderChips(panel);
  } catch (e) {
    panel.ptCount.textContent = String(e.message || e);
  }
  panel.veil.style.display = "none";
  drawAll();
  drawSpectrum();
  markTableSelection();
}

function renderChips(panel) {
  const m = methodMeta(panel.methodId);
  if (!m) { panel.chips.innerHTML = ""; return; }
  const met = m.metrics || {};
  const chips = [`<span class="chip"><b>${(m.size_bytes / 1e6).toFixed(2)} MB</b></span>`];
  if (met.ratio_vs_raw) chips.push(`<span class="chip">${met.ratio_vs_raw.toFixed(0)}:1</span>`);
  if (met.spectrum_tv != null) chips.push(`<span class="chip">spectrum err <b>${met.spectrum_tv.toFixed(4)}</b></span>`);
  if (met.rare_min_density_corr != null)
    chips.push(`<span class="chip">rare corr <b>${met.rare_min_density_corr.toFixed(3)}</b></span>`);
  panel.chips.innerHTML = chips.join("");
}

// ---------- camera / drawing ----------

function dot3(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function cross3(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function normalize3(v) {
  const length = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / length, v[1] / length, v[2] / length];
}

function axisAngle3(axis, angle) {
  const [x, y, z] = normalize3(axis);
  const c = Math.cos(angle), s = Math.sin(angle), t = 1 - c;
  return [
    t * x * x + c, t * x * y - s * z, t * x * z + s * y,
    t * x * y + s * z, t * y * y + c, t * y * z - s * x,
    t * x * z - s * y, t * y * z + s * x, t * z * z + c,
  ];
}

function mul4(a, b) {
  const out = new Array(16).fill(0);
  for (let row = 0; row < 4; row++)
    for (let col = 0; col < 4; col++)
      for (let k = 0; k < 4; k++)
        out[row * 4 + col] += a[row * 4 + k] * b[k * 4 + col];
  return out;
}

function cameraBasis(cam = state.cam) {
  const cosEl = Math.cos(cam.el);
  const outward = [
    cosEl * Math.cos(cam.az),
    cosEl * Math.sin(cam.az),
    Math.sin(cam.el),
  ];
  const forward = [-outward[0], -outward[1], -outward[2]];
  const right = normalize3(cross3(forward, [0, 0, 1]));
  const up = normalize3(cross3(right, forward));
  return { outward, forward, right, up };
}

function mvpMatrix(aspect) {
  const b = state.meta.bounds;
  const ext = [b[1][0] - b[0][0], b[1][1] - b[0][1], b[1][2] - b[0][2]];
  const maxExt = Math.max(...ext);
  const c = state.cam;
  const S = ext.map((e) => e / 65535.0 / maxExt);
  const T = [-0.5 * ext[0] / maxExt, -0.5 * ext[1] / maxExt, -0.5 * ext[2] / maxExt];
  const R = POINT_IMAGE_PLANE_ROTATION;
  const model = [
    R[0] * S[0], R[1] * S[1], R[2] * S[2], dot3(R.slice(0, 3), T),
    R[3] * S[0], R[4] * S[1], R[5] * S[2], dot3(R.slice(3, 6), T),
    R[6] * S[0], R[7] * S[1], R[8] * S[2], dot3(R.slice(6, 9), T),
    0, 0, 0, 1,
  ];

  const { outward, forward, right, up } = cameraBasis(c);
  const position = c.target.map((value, i) => value + outward[i] * c.dist);
  const view = [
    right[0], right[1], right[2], -dot3(right, position),
    up[0], up[1], up[2], -dot3(up, position),
    -forward[0], -forward[1], -forward[2], dot3(forward, position),
    0, 0, 0, 1,
  ];

  const radius = normalizedCloudRadius();
  const near = Math.max(c.dist - radius * 3, radius / 1000, 0.001);
  const far = Math.max(near + 0.001, c.dist + radius * 3);
  const f = 1 / Math.tan(CAMERA_FOV / 2);
  const projection = [
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far + near) / (near - far), (2 * far * near) / (near - far),
    0, 0, -1, 0,
  ];
  const combined = mul4(projection, mul4(view, model));
  const out = new Float32Array(16);
  for (let row = 0; row < 4; row++)
    for (let col = 0; col < 4; col++)
      out[col * 4 + row] = combined[row * 4 + col];
  return out;
}

function autoPointSize() {
  const scale = Math.max(1, state.cam.fitDist / state.cam.dist);
  const size = AUTO_POINT_SIZE_BASE * Math.pow(scale, AUTO_POINT_SIZE_EXPONENT);
  return Math.max(AUTO_POINT_SIZE_BASE, Math.min(AUTO_POINT_SIZE_MAX, size));
}

function autoOpacity(panel) {
  const total = state.meta.atom_count || panel.count;
  if (!total || !panel.count) return DEFAULT_OPACITY;
  const fraction = Math.min(1, panel.count / total);
  return Math.min(1, Math.max(DEFAULT_OPACITY, DEFAULT_OPACITY / Math.max(fraction, 1e-6)));
}

function syncSliders() {
  const size = state.sizeAuto ? autoPointSize() : state.manualSize;
  document.getElementById("point-size").value = size;
  if (state.opacityAuto && state.panels[0]) {
    document.getElementById("opacity").value = autoOpacity(state.panels[0]).toFixed(2);
  }
}

function drawPanel(panel) {
  const gl = panel.gl;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = panel.canvas.clientWidth * dpr, h = panel.canvas.clientHeight * dpr;
  if (panel.canvas.width !== w || panel.canvas.height !== h) {
    panel.canvas.width = w;
    panel.canvas.height = h;
  }
  gl.viewport(0, 0, w, h);
  gl.clearColor(0.027, 0.031, 0.027, 1);
  gl.clear(gl.COLOR_BUFFER_BIT);
  if (!panel.count) return;
  gl.useProgram(panel.prog);
  gl.disable(gl.DEPTH_TEST);
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

  gl.bindBuffer(gl.ARRAY_BUFFER, panel.buf);
  gl.enableVertexAttribArray(panel.loc.aPos);
  gl.vertexAttribPointer(panel.loc.aPos, 3, gl.UNSIGNED_SHORT, false, 8, 0);
  gl.enableVertexAttribArray(panel.loc.aMassQ);
  gl.vertexAttribPointer(panel.loc.aMassQ, 1, gl.UNSIGNED_SHORT, false, 8, 6);

  gl.uniformMatrix4fv(panel.loc.uMVP, false, mvpMatrix(w / h));
  const size = state.sizeAuto ? autoPointSize() : state.manualSize;
  // gl_PointSize is already in physical pixels. uap-archive passes the auto
  // value through unchanged even when the drawing buffer uses a high DPR.
  gl.uniform1f(panel.loc.uPointSize, size);
  const opacity = state.opacityAuto ? autoOpacity(panel) : state.manualOpacity;
  gl.uniform1f(panel.loc.uOpacity, opacity);
  gl.uniform1f(panel.loc.uMassScale, state.meta.spectrum_max_da / 65535.0);
  gl.uniform1f(panel.loc.uDisplayMax, state.meta.display_max_da);
  gl.drawArrays(gl.POINTS, 0, panel.count);
}

function drawAll() {
  syncSliders();
  for (const p of state.panels) drawPanel(p);
}

const cameraMotion = {
  az: 0,
  el: 0,
  pan: [0, 0, 0],
  frame: null,
};

function clearCameraMotion() {
  cameraMotion.az = 0;
  cameraMotion.el = 0;
  cameraMotion.pan.fill(0);
  if (cameraMotion.frame != null) cancelAnimationFrame(cameraMotion.frame);
  cameraMotion.frame = null;
}

function stepCameraMotion() {
  cameraMotion.frame = null;
  if (!state.cam) return;
  state.cam.az += cameraMotion.az * CAMERA_DAMPING;
  state.cam.el += cameraMotion.el * CAMERA_DAMPING;
  state.cam.el = Math.max(-Math.PI / 2 + 1e-4,
    Math.min(Math.PI / 2 - 1e-4, state.cam.el));
  for (let i = 0; i < 3; i++)
    state.cam.target[i] += cameraMotion.pan[i] * CAMERA_DAMPING;

  const decay = 1 - CAMERA_DAMPING;
  cameraMotion.az *= decay;
  cameraMotion.el *= decay;
  for (let i = 0; i < 3; i++) cameraMotion.pan[i] *= decay;
  drawAll();

  const remaining = Math.abs(cameraMotion.az) + Math.abs(cameraMotion.el) +
    Math.hypot(...cameraMotion.pan);
  if (remaining > 1e-5)
    cameraMotion.frame = requestAnimationFrame(stepCameraMotion);
}

function startCameraMotion() {
  if (cameraMotion.frame == null)
    cameraMotion.frame = requestAnimationFrame(stepCameraMotion);
}

function dollyCamera(deltaY) {
  if (!deltaY || !state.cam) return;
  const scale = Math.pow(0.95, Math.abs(deltaY * 0.01));
  state.cam.dist *= deltaY > 0 ? 1 / scale : scale;
  state.cam.dist = Math.max(normalizedCloudRadius() * 0.02,
    Math.min(state.cam.fitDist * 100, state.cam.dist));
  drawAll();
}

function setupCloudInteraction(panel) {
  const cv = panel.canvas;
  let drag = null;
  cv.addEventListener("mousedown", (e) => {
    let mode = null;
    if (e.button === 0)
      mode = (e.ctrlKey || e.metaKey || e.shiftKey) ? "pan" : "rotate";
    else if (e.button === 1) mode = "dolly";
    else if (e.button === 2) mode = "pan";
    if (!mode) return;
    e.preventDefault();
    drag = { x: e.clientX, y: e.clientY, mode };
  });
  window.addEventListener("mousemove", (e) => {
    if (!drag) return;
    e.preventDefault();
    const dx = e.clientX - drag.x;
    const dy = e.clientY - drag.y;
    drag.x = e.clientX; drag.y = e.clientY;
    const height = Math.max(1, cv.clientHeight);
    if (drag.mode === "rotate") {
      cameraMotion.az -= 2 * Math.PI * dx / height;
      cameraMotion.el += 2 * Math.PI * dy / height;
      startCameraMotion();
    } else if (drag.mode === "pan") {
      const { right, up } = cameraBasis();
      const targetDistance = state.cam.dist * Math.tan(CAMERA_FOV / 2);
      const panLeft = 2 * dx * targetDistance / height;
      const panUp = 2 * dy * targetDistance / height;
      for (let i = 0; i < 3; i++)
        cameraMotion.pan[i] += -right[i] * panLeft + up[i] * panUp;
      startCameraMotion();
    } else {
      dollyCamera(dy);
    }
  });
  window.addEventListener("mouseup", () => (drag = null));
  cv.addEventListener("contextmenu", (e) => e.preventDefault());
  cv.addEventListener("wheel", (e) => {
    e.preventDefault();
    dollyCamera(e.deltaY);
  }, { passive: false });
  cv.addEventListener("dblclick", (e) => {
    e.preventDefault();
    clearCameraMotion();
    state.cam = resetCam();
    drawAll();
  });
}

// ---------- spectrum ----------

const spec = {
  canvas: null, pad: { l: 46, r: 14, t: 8, b: 20 },
  dragSel: null, dragPan: null, hover: null,
};
const representationMap = {
  canvas: null, rows: [], layout: [], hover: null,
};
const REPRESENTATION_COLORS = {
  exact: "#51b878",
  modeled: "#75619e",
  sampled: "#c7903c",
  panel: ["#3987e5", "#d95926"],
};

function specXRange() {
  return state.specView || { x0: 0, x1: state.meta.display_max_da };
}

function daAtCanvasX(px, canvas) {
  const { x0, x1 } = specXRange();
  const r = canvas.getBoundingClientRect();
  const frac = (px - r.left - spec.pad.l) / (r.width - spec.pad.l - spec.pad.r);
  return x0 + Math.min(Math.max(frac, 0), 1) * (x1 - x0);
}

function daAtX(px) {
  return daAtCanvasX(px, spec.canvas);
}

function hybridRepresentationRows() {
  if (!state.meta) return [];
  const rows = [];
  state.panels.forEach((panel, panelIndex) => {
    const method = methodMeta(panel.methodId);
    const representation = method?.representation;
    if (representation?.kind !== "hybrid") return;
    rows.push({
      panel, panelIndex, method, representation,
      exactIds: new Set(representation.exact_species_ids || []),
      exactMassBands: representation.exact_unranged_mass_bands || [],
    });
  });
  return rows;
}

// Ranging assignment is ordered: later overlapping windows replace earlier
// ones, matching Ranging.assign() in experiments/common.py.
function speciesIdAtMass(mass) {
  let speciesId = 0;
  for (const window of state.meta?.ranging_windows || []) {
    if (mass >= window.lo && mass < window.hi) speciesId = window.species_id;
  }
  return speciesId;
}

function representationSegments(row, lo, hi) {
  if (!(hi > lo)) return [];
  const cuts = [lo, hi];
  for (const window of state.meta?.ranging_windows || []) {
    if (window.lo > lo && window.lo < hi) cuts.push(window.lo);
    if (window.hi > lo && window.hi < hi) cuts.push(window.hi);
  }
  for (const band of row.exactMassBands) {
    if (band.lo > lo && band.lo < hi) cuts.push(band.lo);
    if (band.hi > lo && band.hi < hi) cuts.push(band.hi);
  }
  cuts.sort((a, b) => a - b);
  const uniqueCuts = cuts.filter((value, index) =>
    index === 0 || Math.abs(value - cuts[index - 1]) > 1e-9);
  const segments = [];
  for (let i = 0; i + 1 < uniqueCuts.length; i++) {
    const a = uniqueCuts[i], b = uniqueCuts[i + 1];
    if (b - a <= 1e-9) continue;
    const speciesId = speciesIdAtMass((a + b) / 2);
    const midpoint = (a + b) / 2;
    const exactMassBand = speciesId === 0 && row.exactMassBands.some(
      (band) => midpoint >= band.lo && midpoint < band.hi);
    const exact = row.exactIds.has(speciesId) || exactMassBand;
    const mode = exact ? "exact"
      : row.representation.modeled_mode === "sampled_points" ? "sampled"
      : "modeled";
    segments.push({
      lo: a, hi: b, speciesId,
      exact, mode,
    });
  }
  return segments;
}

function representationForBin(row, bin) {
  const binW = state.meta.spectrum_bin_da;
  const lo = bin * binW;
  const hi = Math.min((bin + 1) * binW, state.meta.spectrum_max_da);
  const segments = representationSegments(row, lo, hi);
  const exactWidth = segments.reduce(
    (sum, segment) => sum + (segment.exact ? segment.hi - segment.lo : 0), 0);
  const exactFraction = exactWidth / Math.max(hi - lo, 1e-12);
  const sampledWidth = segments.reduce(
    (sum, segment) => sum + (segment.mode === "sampled" ? segment.hi - segment.lo : 0), 0);
  const sampledFraction = sampledWidth / Math.max(hi - lo, 1e-12);
  const epsilon = 1e-7;
  const status = exactFraction >= 1 - epsilon ? "exact"
    : sampledFraction >= 1 - epsilon ? "sampled"
    : exactFraction <= epsilon ? "modeled" : "mixed";
  const speciesLabels = [...new Set(segments.map((segment) =>
    state.meta.species[segment.speciesId]?.label || `species ${segment.speciesId}`))];
  return { lo, hi, exactFraction, sampledFraction, status, speciesLabels };
}

function representationStatusText(summary, compact = false) {
  if (summary.status === "exact") return compact ? "exact" : "exact points";
  if (summary.status === "sampled")
    return compact ? "sampled" : "sampled points with original mass-position pairing";
  if (summary.status === "modeled")
    return compact ? "distribution" : "modeled distribution";
  const exactPercent = Math.min(99, Math.max(1,
    Math.round(summary.exactFraction * 100)));
  const remainder = summary.sampledFraction > 0 ? "sampled" : "distribution";
  return compact ? `mixed (${exactPercent}% exact)`
    : `mixed · ${exactPercent}% exact by mass width, ${100 - exactPercent}% ${remainder}`;
}

function drawRepresentationMap() {
  const cv = representationMap.canvas;
  const block = document.getElementById("representation-block");
  if (!cv || !block || !state.meta) return;
  const rows = hybridRepresentationRows();
  representationMap.rows = rows;
  block.hidden = rows.length === 0;
  if (!rows.length) {
    representationMap.layout = [];
    return;
  }

  const rowHeight = 15, rowGap = 5, top = 4, bottom = 4;
  const cssHeight = top + bottom + rows.length * rowHeight +
    Math.max(0, rows.length - 1) * rowGap;
  cv.style.height = `${cssHeight}px`;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = cv.clientWidth * dpr, h = cssHeight * dpr;
  if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
  const ctx = cv.getContext("2d");
  ctx.clearRect(0, 0, w, h);
  const padLeft = spec.pad.l * dpr, padRight = spec.pad.r * dpr;
  const plotWidth = w - padLeft - padRight;
  const { x0, x1 } = specXRange();
  const xOfDa = (da) => padLeft + ((da - x0) / (x1 - x0)) * plotWidth;
  const binPixels = state.meta.spectrum_bin_da / (x1 - x0) * (plotWidth / dpr);
  representationMap.layout = [];

  rows.forEach((row, index) => {
    const yCss = top + index * (rowHeight + rowGap);
    const y = yCss * dpr, rh = rowHeight * dpr;
    representationMap.layout.push({ row, y0: yCss, y1: yCss + rowHeight });
    ctx.fillStyle = REPRESENTATION_COLORS.panel[row.panelIndex];
    ctx.font = `600 ${10 * dpr}px system-ui`;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.fillText(row.panel.tag, padLeft - 8 * dpr, y + rh / 2);

    for (const segment of representationSegments(row, x0, x1)) {
      const xa = Math.max(padLeft, xOfDa(segment.lo));
      const xb = Math.min(w - padRight, xOfDa(segment.hi));
      if (xb <= xa) continue;
      ctx.fillStyle = REPRESENTATION_COLORS[segment.mode];
      ctx.fillRect(xa, y, xb - xa, rh);
    }

    if (binPixels >= 4) {
      const binW = state.meta.spectrum_bin_da;
      ctx.strokeStyle = "rgba(17,17,16,0.38)";
      ctx.lineWidth = dpr;
      const firstBin = Math.ceil(x0 / binW);
      for (let bin = firstBin; bin * binW < x1; bin++) {
        const x = xOfDa(bin * binW);
        ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x, y + rh); ctx.stroke();
      }
    }
    ctx.strokeStyle = "rgba(255,255,255,0.18)";
    ctx.lineWidth = dpr;
    ctx.strokeRect(padLeft + 0.5 * dpr, y + 0.5 * dpr,
      plotWidth - dpr, rh - dpr);
  });

  const hover = representationMap.hover ?? spec.hover;
  if (hover != null) {
    const x = xOfDa(hover);
    ctx.strokeStyle = "rgba(255,255,255,0.72)";
    ctx.lineWidth = dpr;
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
  }

  const title = document.getElementById("representation-title");
  title.textContent = `hybrid storage by species / rare mass band · ${rows.map((row) =>
    `${row.panel.tag}: ${row.method.id}`).join(" · ")}`;
  title.title = "The decision is made per ranged species; mass-band hybrids also split the broad unranged population by color band.";
  cv.setAttribute("aria-label", rows.map((row) =>
    `${row.panel.tag} ${row.method.id}: green intervals are exact, purple intervals are modeled distributions, and amber intervals are sampled points`).join(". "));
}

function drawSpectrum() {
  const cv = spec.canvas;
  if (!cv || !state.meta) return;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = cv.clientWidth * dpr, h = cv.clientHeight * dpr;
  if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
  const ctx = cv.getContext("2d");
  ctx.clearRect(0, 0, w, h);
  const pad = { l: spec.pad.l * dpr, r: spec.pad.r * dpr, t: spec.pad.t * dpr, b: spec.pad.b * dpr };
  const pw = w - pad.l - pad.r, ph = h - pad.t - pad.b;
  const { x0, x1 } = specXRange();
  const binW = state.meta.spectrum_bin_da;

  const orig = state.payloads.get(`${state.slug}/original`);
  if (!orig) { drawRepresentationMap(); return; }
  const maxCount = Math.max(1, ...orig.trueCounts);
  const yOf = (c) => pad.t + ph * (1 - Math.log10(1 + c) / Math.log10(1 + maxCount));
  const xOfDa = (da) => pad.l + ((da - x0) / (x1 - x0)) * pw;

  // selection bands under everything
  for (const [lo, hi] of state.ranges) {
    const xa = Math.max(pad.l, xOfDa(lo)), xb = Math.min(w - pad.r, xOfDa(hi));
    if (xb > xa) {
      ctx.fillStyle = "rgba(255,255,255,0.07)";
      ctx.fillRect(xa, pad.t, xb - xa, ph);
      ctx.strokeStyle = "rgba(255,255,255,0.25)";
      ctx.strokeRect(xa + 0.5, pad.t + 0.5, xb - xa - 1, ph - 1);
    }
  }
  if (spec.dragSel) {
    const xa = xOfDa(Math.min(spec.dragSel.a, spec.dragSel.b));
    const xb = xOfDa(Math.max(spec.dragSel.a, spec.dragSel.b));
    ctx.fillStyle = "rgba(57,135,229,0.18)";
    ctx.fillRect(xa, pad.t, xb - xa, ph);
  }

  ctx.strokeStyle = "#2c2c2a";
  ctx.lineWidth = 1;
  ctx.fillStyle = "#898781";
  ctx.font = `${10 * dpr}px system-ui`;
  ctx.textAlign = "right";
  for (let d = 0; Math.pow(10, d) <= maxCount; d++) {
    const y = yOf(Math.pow(10, d));
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
    ctx.fillText(d === 0 ? "1" : `1e${d}`, pad.l - 4 * dpr, y + 3 * dpr);
  }
  ctx.textAlign = "center";
  const daStep = niceStep((x1 - x0) / 10);
  for (let da = Math.ceil(x0 / daStep) * daStep; da <= x1 + 1e-9; da += daStep) {
    ctx.fillText(da.toFixed(daStep < 1 ? 1 : 0), xOfDa(da), h - 5 * dpr);
  }

  const plot = (counts, color, lw) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = lw * dpr;
    ctx.beginPath();
    let started = false;
    const b0 = Math.max(0, Math.floor(x0 / binW));
    const b1 = Math.min(counts.length, Math.ceil(x1 / binW));
    const binsPerPx = Math.max(1, Math.floor((b1 - b0) / (pw / dpr)));
    for (let b = b0; b < b1; b += binsPerPx) {
      let v = 0;
      for (let k2 = b; k2 < Math.min(b + binsPerPx, b1); k2++) v = Math.max(v, counts[k2]);
      const x = xOfDa((b + 0.5) * binW);
      const y = yOf(v);
      if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
    }
    ctx.stroke();
  };
  plot(orig.trueCounts, "#898781", 1);
  const seen = new Set(["original"]);
  const colors = { 0: "#3987e5", 1: "#d95926" };
  state.panels.forEach((p, i) => {
    if (!p.methodId || seen.has(p.methodId)) return;
    const pl = state.payloads.get(`${state.slug}/${p.methodId}`);
    if (pl) { plot(pl.trueCounts, colors[i], 1.5); seen.add(p.methodId); }
  });

  if (spec.hover != null) {
    const x = xOfDa(spec.hover);
    ctx.strokeStyle = "#52514e";
    ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, h - pad.b); ctx.stroke();
  }
  drawRepresentationMap();
}

function niceStep(raw) {
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  for (const m of [1, 2, 5, 10]) if (raw <= m * mag) return m * mag;
  return 10 * mag;
}

function commitRange(a, b) {
  const lo = Math.min(a, b), hi = Math.max(a, b);
  if (hi - lo < 0.05) return;
  state.ranges.push([lo, hi]);
  state.activeChip = null;
  onSelectionChanged();
}

function onSelectionChanged() {
  renderSelChips();
  updateSpeciesChipStates();
  for (const p of state.panels) refreshPanel(p);
}

function renderSelChips() {
  const holder = document.getElementById("sel-chips");
  holder.innerHTML = "";
  state.ranges.forEach(([lo, hi], i) => {
    const chip = document.createElement("button");
    chip.className = "sel-chip";
    chip.textContent = `${lo.toFixed(2)}–${hi.toFixed(2)} Da ✕`;
    chip.title = "remove this range";
    chip.addEventListener("click", () => {
      state.ranges.splice(i, 1);
      state.activeChip = null;
      onSelectionChanged();
    });
    holder.appendChild(chip);
  });
  document.getElementById("clear-sel").style.display =
    state.ranges.length ? "" : "none";
}

function zoomSpectrumAt(da, deltaY) {
  const f = Math.exp(deltaY * 0.0012);
  let { x0, x1 } = specXRange();
  x0 = da - (da - x0) * f;
  x1 = da + (x1 - da) * f;
  x0 = Math.max(0, x0);
  x1 = Math.min(state.meta.spectrum_max_da, x1);
  if (x1 - x0 > 0.2) state.specView = { x0, x1 };
  requestAnimationFrame(drawSpectrum);
}

function setupSpectrumInteraction() {
  const cv = spec.canvas;
  cv.addEventListener("mousedown", (e) => {
    const da = daAtX(e.clientX);
    if (e.shiftKey) spec.dragPan = { x: e.clientX };
    else spec.dragSel = { a: da, b: da };
  });
  window.addEventListener("mousemove", (e) => {
    if (spec.dragSel) {
      spec.dragSel.b = daAtX(e.clientX);
      requestAnimationFrame(drawSpectrum);
    } else if (spec.dragPan) {
      const { x0, x1 } = specXRange();
      const r = cv.getBoundingClientRect();
      const dDa = ((e.clientX - spec.dragPan.x) / (r.width - spec.pad.l - spec.pad.r)) * (x1 - x0);
      spec.dragPan.x = e.clientX;
      let n0 = x0 - dDa, n1 = x1 - dDa;
      if (n0 >= 0 && n1 <= state.meta.spectrum_max_da) state.specView = { x0: n0, x1: n1 };
      requestAnimationFrame(drawSpectrum);
    }
  });
  window.addEventListener("mouseup", () => {
    if (spec.dragSel) {
      commitRange(spec.dragSel.a, spec.dragSel.b);
      spec.dragSel = null;
      drawSpectrum();
    }
    spec.dragPan = null;
  });
  cv.addEventListener("wheel", (e) => {
    e.preventDefault();
    zoomSpectrumAt(daAtX(e.clientX), e.deltaY);
  }, { passive: false });
  cv.addEventListener("dblclick", () => {
    state.specView = null;
    drawSpectrum();
  });
  cv.addEventListener("mousemove", (e) => {
    spec.hover = daAtX(e.clientX);
    representationMap.hover = spec.hover;
    const orig = state.payloads.get(`${state.slug}/original`);
    if (orig) {
      const bin = Math.floor(spec.hover / state.meta.spectrum_bin_da);
      if (bin >= 0 && bin < orig.trueCounts.length) {
        tooltip.style.display = "block";
        tooltip.style.left = e.clientX + 14 + "px";
        tooltip.style.top = e.clientY - 28 + "px";
        let txt = `${spec.hover.toFixed(2)} Da · original ${orig.trueCounts[bin].toLocaleString()}`;
        for (const p of state.panels) {
          if (p.methodId && p.methodId !== "original") {
            const pl = state.payloads.get(`${state.slug}/${p.methodId}`);
            if (pl) txt += ` · ${p.tag}:${pl.trueCounts[bin].toLocaleString()}`;
          }
        }
        for (const row of hybridRepresentationRows()) {
          const summary = representationForBin(row, bin);
          txt += ` · ${row.panel.tag} storage: ${representationStatusText(summary, true)}`;
        }
        tooltip.textContent = txt;
      }
    }
    requestAnimationFrame(drawSpectrum);
  });
  cv.addEventListener("mouseleave", () => {
    spec.hover = null;
    representationMap.hover = null;
    tooltip.style.display = "none";
    drawSpectrum();
  });
}

function setupRepresentationInteraction() {
  const cv = representationMap.canvas;
  cv.addEventListener("mousemove", (e) => {
    const da = daAtCanvasX(e.clientX, cv);
    const bin = Math.min(
      Math.floor(state.meta.spectrum_max_da / state.meta.spectrum_bin_da) - 1,
      Math.max(0, Math.floor(da / state.meta.spectrum_bin_da)));
    const rect = cv.getBoundingClientRect();
    const y = e.clientY - rect.top;
    let hit = representationMap.layout.find((item) => y >= item.y0 && y <= item.y1);
    if (!hit && representationMap.layout.length) {
      hit = representationMap.layout.reduce((best, item) => {
        const distance = Math.abs(y - (item.y0 + item.y1) / 2);
        return !best || distance < best.distance ? { ...item, distance } : best;
      }, null);
    }
    spec.hover = da;
    representationMap.hover = da;
    if (hit) {
      const summary = representationForBin(hit.row, bin);
      const species = summary.speciesLabels.join(" + ");
      tooltip.style.display = "block";
      tooltip.style.left = e.clientX + 14 + "px";
      tooltip.style.top = e.clientY - 28 + "px";
      tooltip.textContent = `${hit.row.panel.tag} ${hit.row.method.id} · ` +
        `${summary.lo.toFixed(2)}–${summary.hi.toFixed(2)} Da · ${species} · ` +
        representationStatusText(summary);
    }
    requestAnimationFrame(drawSpectrum);
  });
  cv.addEventListener("mouseleave", () => {
    spec.hover = null;
    representationMap.hover = null;
    tooltip.style.display = "none";
    drawSpectrum();
  });
  cv.addEventListener("wheel", (e) => {
    e.preventDefault();
    zoomSpectrumAt(daAtCanvasX(e.clientX, cv), e.deltaY);
  }, { passive: false });
  cv.addEventListener("dblclick", () => {
    state.specView = null;
    drawSpectrum();
  });
}

// ---------- species quick-select chips ----------

function paletteJS(t) {
  const stops = [
    [0.5, 0, 1], [0, 0.25, 1], [0, 0.85, 1], [0.2, 1, 0.2],
    [1, 0.92, 0], [1, 0.42, 0], [1, 0, 0]];
  t = Math.min(Math.max(t, 0), 1);
  const seg = Math.min(Math.floor(t * 6), 5);
  const f = t * 6 - seg;
  const s = (x) => x * x * (3 - 2 * x);
  const m = s(f);
  return stops[seg].map((a, i) => Math.round((a + (stops[seg + 1][i] - a) * m) * 255));
}

function renderSpeciesChips() {
  const holder = document.getElementById("species-chips");
  holder.innerHTML = "";
  const top = state.meta.species
    .filter((s) => s.label !== "unranged" && s.count > 0)
    .sort((a, b) => b.count - a.count)
    .slice(0, 12);
  for (const s of top) {
    const chip = document.createElement("button");
    chip.className = "sp-chip";
    chip.dataset.label = s.label;
    const anchor = s.windows.length ? (s.windows[0][0] + s.windows[0][1]) / 2 : 0;
    const c = paletteJS(anchor / state.meta.display_max_da);
    chip.innerHTML = `<span class="dot" style="background:rgb(${c.join(",")})"></span>` +
      `${s.label} <span style="color:var(--muted)">${fmtCount(s.count)}</span>`;
    chip.title = s.windows.map(([a, b]) => `${a.toFixed(2)}–${b.toFixed(2)} Da`).join(", ") +
      " — click to isolate, shift-click to add";
    chip.addEventListener("click", (e) => {
      if (state.activeChip === s.label && !e.shiftKey) {
        state.ranges = [];
        state.activeChip = null;
      } else {
        if (e.shiftKey) state.ranges.push(...s.windows.map((w) => [...w]));
        else state.ranges = s.windows.map((w) => [...w]);
        state.activeChip = e.shiftKey ? null : s.label;
      }
      onSelectionChanged();
    });
    holder.appendChild(chip);
  }
}

function updateSpeciesChipStates() {
  for (const chip of document.querySelectorAll(".sp-chip")) {
    chip.classList.toggle("active", chip.dataset.label === state.activeChip);
  }
}

function fmtCount(n) {
  if (n >= 1e6) return (n / 1e6).toFixed(1) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(0) + "k";
  return String(n);
}

// ---------- table / scatter / notes ----------

const TABLE_COLS = [
  ["label", "method", (m) => m.label, "str"],
  ["size", "size MB", (m) => m.size_bytes / 1e6, "mb"],
  ["ratio", "ratio", (m) => m.metrics.ratio_vs_raw ?? null, "x"],
  ["spectrum_tv", "spectrum err", (m) => m.metrics.spectrum_tv, "err"],
  ["density", "density corr", (m) => m.metrics.density_corr_2nm, "corr"],
  ["rare", "rare-min corr", (m) => m.metrics.rare_min_density_corr, "corr"],
  ["rarep", "rare structure", (m) => m.metrics.rare_min_proj_corr, "corr"],
  ["massp", "mass/color structure", (m) => m.metrics.mass_band_min_proj_corr, "corr"],
  ["mix", "species-mix err", (m) => m.metrics.spatial_species_error, "err"],
  ["dec", "decode s", (m) => m.metrics.decode_seconds, "s"],
];

let sortCol = 1, sortAsc = true;

function fmtMetric(value, digits = 4) {
  return value == null ? "–" : Number(value).toFixed(digits);
}

function renderAggregate() {
  const summary = state.summary;
  const hero = document.getElementById("aggregate-verdict");
  const table = document.getElementById("aggregate-table");
  if (!summary?.methods?.length) {
    hero.hidden = true;
    table.innerHTML = "";
    document.getElementById("aggregate-note").textContent = "No complete cross-dataset results found.";
    return;
  }
  const winner = summary.methods.find((m) => m.id === summary.winner_id);
  if (winner) {
    const medianMb = winner.median_size_bytes / 1e6;
    hero.hidden = false;
    document.getElementById("verdict-eyebrow").textContent =
      `${summary.benchmark_run_count} completed runs · ${summary.method_count} techniques · ${summary.dataset_count} POS files`;
    document.getElementById("verdict-method").textContent = winner.id;
    document.getElementById("verdict-copy").textContent =
      `It is the smallest method that preserves rare-element projection structure above ` +
      `${summary.quality_gate.worst_rare_min_proj_corr.toFixed(2)} and mass/color structure above ` +
      `${summary.quality_gate.worst_mass_band_min_proj_corr.toFixed(2)} on every dataset, while meeting the density and ` +
      `spatial-composition gates. Rare species—and rare mass bands hidden inside “unranged”—stay exact; frequent bands ` +
      `are sampled from smoothly interpolated density fields.`;
    document.getElementById("verdict-stats").innerHTML = [
      [medianMb.toFixed(2) + " MB", "median artifact"],
      [winner.median_ratio_vs_raw.toFixed(0) + ":1", "median compression"],
      [winner.worst_mass_band_min_proj_corr.toFixed(4), "worst mass/color structure"],
      [winner.worst_rare_min_proj_corr.toFixed(4), "worst rare structure"],
    ].map(([value, label]) =>
      `<div class="verdict-stat"><b>${value}</b><span>${label}</span></div>`).join("");
  }

  const rareCount = Math.max(...summary.methods.map((m) => m.rare_dataset_count || 0));
  document.getElementById("aggregate-note").textContent =
    `Medians and worst cases cover all ${summary.dataset_count} files; rare-structure columns cover the ` +
    `${rareCount} files with qualifying rare elements. “gate” means rare projection ≥ ` +
    `${summary.quality_gate.worst_rare_min_proj_corr.toFixed(2)}, median density correlation ≥ ` +
    `${summary.quality_gate.median_density_corr_2nm.toFixed(3)}, and worst species-mix error ≤ ` +
    `${summary.quality_gate.worst_spatial_species_error.toFixed(3)}, with mass/color structure ≥ ` +
    `${summary.quality_gate.worst_mass_band_min_proj_corr.toFixed(2)}.`;

  const rows = [...summary.methods].sort((a, b) =>
    a.median_size_bytes - b.median_size_bytes);
  let html = `<thead><tr><th>method</th><th>median MB</th><th>max MB</th>` +
    `<th>median ratio</th><th>median density</th><th>worst density</th>` +
    `<th>worst rare structure</th><th>worst mass/color structure</th>` +
    `<th>worst species-mix err</th><th>median decode s</th></tr></thead><tbody>`;
  for (const m of rows) {
    const isWinner = m.id === summary.winner_id;
    const badges = (isWinner ? `<span class="winner-badge">winner</span>` : "") +
      (m.meets_quality_gate && !isWinner ? `<span class="gate-badge">gate</span>` : "");
    html += `<tr class="${isWinner ? "winner" : ""}">` +
      `<td class="name" title="${m.label}">${m.id}${badges}</td>` +
      `<td>${(m.median_size_bytes / 1e6).toFixed(2)}</td>` +
      `<td>${(m.max_size_bytes / 1e6).toFixed(2)}</td>` +
      `<td>${m.median_ratio_vs_raw.toFixed(0)}:1</td>` +
      `<td>${fmtMetric(m.median_density_corr_2nm)}</td>` +
      `<td>${fmtMetric(m.worst_density_corr_2nm)}</td>` +
      `<td>${fmtMetric(m.worst_rare_min_proj_corr)}</td>` +
      `<td>${fmtMetric(m.worst_mass_band_min_proj_corr)}</td>` +
      `<td>${fmtMetric(m.worst_spatial_species_error)}</td>` +
      `<td>${fmtMetric(m.median_decode_seconds, 2)}</td></tr>`;
  }
  table.innerHTML = html + "</tbody>";
}

function renderTable() {
  const tbl = document.getElementById("metrics-table");
  const methods = state.meta.methods.filter((m) => m.id !== "original");
  const rows = [...methods].sort((a, b) => {
    const va = TABLE_COLS[sortCol][2](a), vb = TABLE_COLS[sortCol][2](b);
    if (va == null) return 1;
    if (vb == null) return -1;
    return (va < vb ? -1 : va > vb ? 1 : 0) * (sortAsc ? 1 : -1);
  });
  const best = {};
  for (let ci = 1; ci < TABLE_COLS.length; ci++) {
    const [, , get, fmt] = TABLE_COLS[ci];
    const vals = methods.map(get).filter((v) => v != null);
    if (!vals.length) continue;
    best[ci] = (fmt === "corr" || fmt === "x") ? Math.max(...vals) : Math.min(...vals);
  }
  let html = "<thead><tr><th></th>";
  TABLE_COLS.forEach(([, name], i) => {
    const arrow = i === sortCol ? (sortAsc ? " ↑" : " ↓") : "";
    html += `<th data-ci="${i}">${name}${arrow}</th>`;
  });
  html += "</tr></thead><tbody>";
  for (const m of rows) {
    const dis = m.has_payload ? "" : "disabled title=\"metrics only — no stored payload for this dataset\"";
    html += `<tr data-id="${m.id}"><td>` +
      `<button class="ab-btn" data-p="0" ${dis}>A</button> ` +
      `<button class="ab-btn" data-p="1" ${dis}>B</button></td>`;
    TABLE_COLS.forEach(([, , get, fmt], ci) => {
      const v = get(m);
      let txt = "–";
      if (v != null) {
        if (fmt === "str") txt = m.id;
        else if (fmt === "mb") txt = v >= 10 ? v.toFixed(1) : v.toFixed(2);
        else if (fmt === "x") txt = v.toFixed(0) + ":1";
        else if (fmt === "corr" || fmt === "err") txt = v.toFixed(4);
        else txt = v.toFixed(2);
      }
      const cls = [ci === 0 ? "name" : "", v != null && best[ci] === v ? "best" : ""].join(" ");
      html += `<td class="${cls}" title="${fmt === "str" ? m.label : ""}">${txt}</td>`;
    });
    html += "</tr>";
  }
  html += "</tbody>";
  tbl.innerHTML = html;
  tbl.querySelectorAll("th[data-ci]").forEach((th) => {
    th.addEventListener("click", () => {
      const ci = +th.dataset.ci;
      if (ci === sortCol) sortAsc = !sortAsc;
      else { sortCol = ci; sortAsc = TABLE_COLS[ci][3] !== "corr"; }
      renderTable();
      markTableSelection();
    });
  });
  tbl.querySelectorAll(".ab-btn:not([disabled])").forEach((btn) => {
    btn.addEventListener("click", () => {
      const id = btn.closest("tr").dataset.id;
      setPanelMethod(state.panels[+btn.dataset.p], id);
    });
  });
}

function markTableSelection() {
  document.querySelectorAll("#metrics-table tbody tr").forEach((tr) => {
    tr.classList.toggle("hl-A", tr.dataset.id === state.panels[0]?.methodId);
    tr.classList.toggle("hl-B", tr.dataset.id === state.panels[1]?.methodId);
  });
}

function renderScatter() {
  const holder = document.getElementById("scatter");
  const metric = (m) => {
    const values = [m.metrics.rare_min_proj_corr, m.metrics.mass_band_min_proj_corr]
      .filter((v) => v != null);
    return values.length ? Math.min(...values) : null;
  };
  const methods = state.meta.methods.filter((m) => m.id !== "original" && metric(m) != null);
  const rareEls = (state.meta.rare_elements || []).join(", ");
  document.getElementById("scatter-note").textContent = methods.length
    ? `y-axis: worst spatial correlation over mass/color bands and qualifying rare elements (${rareEls || "none"}) vs the original. Click a point to load it into panel B.`
    : "No spatial-structure metrics are available for this dataset.";
  if (!methods.length) { holder.innerHTML = ""; return; }
  const W = holder.clientWidth || 1200, H = 360;
  const pad = { l: 52, r: 24, t: 16, b: 40 };
  const sizes = methods.map((m) => m.size_bytes / 1e6);
  const xmin = Math.min(...sizes) * 0.7, xmax = Math.max(...sizes) * 1.6;
  const xOf = (mb) => pad.l + (Math.log10(mb / xmin) / Math.log10(xmax / xmin)) * (W - pad.l - pad.r);
  const ymin = Math.min(0.3, ...methods.map(metric)) - 0.02, ymax = 1.005;
  const yOf = (c) => pad.t + (1 - (Math.max(c, ymin) - ymin) / (ymax - ymin)) * (H - pad.t - pad.b);
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="size vs rare-element fidelity">`;
  for (const g of [0.4, 0.6, 0.8, 0.9, 0.95, 1.0]) {
    if (g < ymin) continue;
    s += `<line x1="${pad.l}" y1="${yOf(g)}" x2="${W - pad.r}" y2="${yOf(g)}" stroke="#2c2c2a"/>` +
      `<text x="${pad.l - 8}" y="${yOf(g) + 4}" fill="#898781" font-size="11" text-anchor="end">${g.toFixed(2)}</text>`;
  }
  for (const mb of [0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 50]) {
    if (mb < xmin || mb > xmax) continue;
    s += `<line x1="${xOf(mb)}" y1="${pad.t}" x2="${xOf(mb)}" y2="${H - pad.b}" stroke="#2c2c2a"/>` +
      `<text x="${xOf(mb)}" y="${H - pad.b + 16}" fill="#898781" font-size="11" text-anchor="middle">${mb}</text>`;
  }
  s += `<text x="${W / 2}" y="${H - 6}" fill="#c3c2b7" font-size="12" text-anchor="middle">compressed size, MB (log)</text>`;
  for (const m of methods) {
    const x = xOf(m.size_bytes / 1e6), y = yOf(metric(m));
    s += `<circle data-id="${m.id}" cx="${x}" cy="${y}" r="6" fill="#3987e5" ` +
      `stroke="#111110" stroke-width="2" style="cursor:pointer"></circle>` +
      `<text x="${x + 9}" y="${y + 4}" fill="#c3c2b7" font-size="11">${m.id}</text>`;
  }
  s += "</svg>";
  holder.innerHTML = s;
  holder.querySelectorAll("circle").forEach((c) => {
    c.addEventListener("click", () => {
      const m = methodMeta(c.dataset.id);
      if (m?.has_payload) setPanelMethod(state.panels[1], c.dataset.id);
    });
    c.addEventListener("mousemove", (e) => {
      const m = methodMeta(c.dataset.id);
      tooltip.style.display = "block";
      tooltip.style.left = e.clientX + 14 + "px";
      tooltip.style.top = e.clientY - 10 + "px";
      tooltip.textContent = `${m.id} — ${(m.size_bytes / 1e6).toFixed(2)} MB, worst structure ` +
        metric(m).toFixed(4);
    });
    c.addEventListener("mouseleave", () => (tooltip.style.display = "none"));
  });
}

const NOTES = {
  subsample10: {
    what: "Keep a random 10% of atoms, quantize to 16 bits. This matches the repository's existing fallback method.",
    pros: "Trivially simple; per-atom masses are real, so any analysis still works; unbiased.",
    cons: "Rare species keep only 10% of their atoms — wires and precipitates get noisy; fine spectrum becomes Poisson-noisy; still one of the larger artifacts.",
  },
  massrange64: {
    what: "Stand-in for the current production codec: 64 adaptive equal-count mass ranges, each stored as its own density grid.",
    pros: "Simple decode; good total-density and majority-species fidelity.",
    cons: "Equal-count ranges give ~90% of the budget to the majority peak; rare species share ranges with background and smear. This is exactly the observed 'rare element washes out' failure.",
  },
  qpoint_mass: {
    what: "Every atom kept: positions quantized to 12 bits/axis, Morton-sorted, delta+varint bit-packed, plus a per-atom mass byte inside its species window.",
    pros: "Effectively lossless for viewing (±0.03nm); real per-atom masses; best possible joint detail.",
    cons: "Largest artifact; decode is heavier; per-atom mass residual is mostly redundant with the global spectrum.",
  },
  qpoint_synth: {
    what: "Same bit-packed positions, but per-atom masses are re-synthesized from the stored fine spectrum per species.",
    pros: "Everything qpoint_mass gives at ~30% less size; spectrum is exact by construction.",
    cons: "Individual atom masses are plausible rather than true (fine for viewing, not for correlative isotope studies).",
  },
  grid_global: {
    what: "Control experiment: 1nm total-density grid + one global composition vector (no spatial composition modeling).",
    pros: "Tiny; total density and spectrum look perfect.",
    cons: "Composition is uniform everywhere — rare-element structure completely disappears. Shows why composition modeling matters.",
  },
  grid_kmeans: {
    what: "1nm density grid + k-means compositional clusters on 3nm cells (K=8): each cluster stores its own species mix.",
    pros: "Captures distinct phases with crisp boundaries; very small.",
    cons: "Hard assignment quantizes gradients: within-cluster composition is uniform, so smooth enrichment gradients flatten.",
  },
  grid_kmeans_resid: {
    what: "k-means clusters plus a coarser per-species grid of log-ratio deviations from the cluster baseline.",
    pros: "Recovers most within-cluster variation; still tiny.",
    cons: "More moving parts than direct fractions for slightly worse quality; residual quantization adds noise for ultra-rare species.",
  },
  grid_nmf: {
    what: "Non-negative matrix factorization of the cell x species matrix (k=8): spatially varying non-negative mixtures of baseline compositions.",
    pros: "Physically meaningful non-negative parts; soft boundaries; excellent rare-element fidelity at tiny sizes.",
    cons: "Factor maps cost slightly more than hard labels; components are data-driven and need interpretation.",
  },
  grid_pca: {
    what: "Same architecture with PCA components (signed), clipped to non-negative on decode.",
    pros: "Comparable quality to NMF here.",
    cons: "Signed components can produce negative fractions that must be clipped — a modeling mismatch NMF avoids.",
  },
  grid_ica: {
    what: "Same with FastICA components, clipped on decode.",
    pros: "Comparable quality to NMF here.",
    cons: "Same negativity issue as PCA; ICA convergence is less stable on this data.",
  },
  grid_direct: {
    what: "1nm density grid + per-species fraction grids stored directly on 3nm cells (sqrt-quantized bytes).",
    pros: "Simplest composition model, no fitting; near-perfect on all species at ~0.3 MB.",
    cons: "Storage grows linearly with species count; fractions still smoothed to 3nm cells.",
  },
  hybrid_exact: {
    what: "Key trick: every species below 100k atoms is stored EXACTLY as bit-packed points; only the majority species and background are density-modeled.",
    pros: "Rare-element structure is atom-for-atom perfect — rare species are cheap because there are few of them; ~1 MB artifacts.",
    cons: "Mass is synthesized independently inside modeled species. A rare high-mass structure hidden in the large unranged bucket can therefore disappear.",
  },
  hybrid_hifi: {
    what: "Same, with the majority/background density grid refined to 0.5nm.",
    pros: "Perfect rare species plus visibly sharper majority density.",
    cons: "Uses the same <100k species boundary as hybrid_exact; the finer grid cannot restore mass-position correlation that was never stored.",
  },
  hybrid_ultra: {
    what: "Same, with the majority grid pushed to 0.25nm — approaching interatomic spacing.",
    pros: "Near point-level fidelity for everything at a fraction of qpoint's size.",
    cons: "Still uses the identical <100k species boundary and independent mass synthesis, so Sample 3's red structures remain missing despite the larger grid.",
  },
  hybrid_sample5: {
    what: "Store species below 100k exactly; keep original mass-position pairs for a 5% sample of every abundant species and expand those samples smoothly on decode.",
    pros: "Restores Sample 3's red structures and preserves rare species and the fine spectrum.",
    cons: "Five percent is too noisy for total density and for rare mass bands nested inside a large unranged species on some datasets.",
  },
  hybrid_sample10: {
    what: "The same exact-rare / sampled-frequent codec at a 10% abundant-species sample rate.",
    pros: "Sharper than 5%; on Sample 3 it preserves the red inclusions at 8.41 MB, below subsample10 and hybrid_ultra.",
    cons: "Uniform species-level sampling still undersamples rare within-species mass bands and misses the all-dataset density gate.",
  },
  hybrid_massbands: {
    what: "Store rare ranged species exactly, split the broad unranged bucket into the six viewer mass/color bands, store any band below 100k exactly, and model only frequent bands on interpolated 1nm fields.",
    pros: "Smallest codec to pass every rare-species, mass/color, density, and composition gate across all seven available POS files.",
    cons: "The frequent fields are resolved at 1nm; use the hifi variant when majority/background edge sharpness matters more than size.",
  },
  hybrid_massbands_hifi: {
    what: "The mass-band-aware hybrid with a 0.5nm density backbone; exact/model decisions are otherwise identical to hybrid_massbands.",
    pros: "Raises worst-case total-density fidelity substantially while retaining exact rare species and rare unranged mass bands.",
    cons: "About one-third larger at the median than hybrid_massbands, with nearly identical mass/color and rare-species scores.",
  },
  wavelet_kmeans: {
    what: "The 1nm density backbone compressed with a 3D bior4.4 wavelet transform (top coefficients kept), k-means composition.",
    pros: "Smooth density representation; graceful quality dial via coefficient budget.",
    cons: "At these grid sizes zstd on raw voxel bytes is already near-optimal, so the transform does not pay for itself; ringing near sharp edges.",
  },
  inr_kmeans: {
    what: "Neural implicit representation: a multiresolution hash-grid MLP (instant-ngp style) fit to log-density, weights stored as fp16.",
    pros: "Resolution-free field, smooth interpolation; competitive density quality.",
    cons: "GPU encode; ~1 MB of weights beats neither the plain grid's size nor its accuracy here; decode needs a NN runtime in the browser.",
  },
  hyper_kmeans: {
    what: "Hypernetwork: per-z-slice latent codes feed a small network that generates the weights of a per-slice 2D MLP density field.",
    pros: "Smallest neural variant; a fun architecture demonstration.",
    cons: "Clearly worse density fidelity; slice artifacts; same NN-decode burden. Neural approaches lose to zstd'd grids at this data scale.",
  },
};

function renderNotes() {
  const holder = document.getElementById("notes-body");
  let html = "";
  for (const m of state.meta.methods) {
    if (m.id === "original") continue;
    const n = NOTES[m.id] || { what: m.label, pros: "", cons: "" };
    html += `<div class="note-card"><h3>${m.id}<span class="size">${(m.size_bytes / 1e6).toFixed(2)} MB</span></h3>` +
      `<p>${n.what}</p>` +
      (n.pros ? `<div class="pros">+ ${n.pros}</div>` : "") +
      (n.cons ? `<div class="cons">− ${n.cons}</div>` : "") + `</div>`;
  }
  holder.innerHTML = html;
}

// ---------- panels / dataset switching ----------

async function setPanelMethod(panel, id) {
  panel.methodId = id;
  panel.select.value = id;
  const legend = document.getElementById(panel.tag === "A" ? "legend-a" : "legend-b");
  legend.textContent = id;
  legend.parentElement.style.display = id === "original" ? "none" : "";
  await refreshPanel(panel);
}

async function loadDataset(slug) {
  state.slug = slug;
  state.meta = await fetchJSON(`data/${slug}/meta.json`);
  state.ranges = [];
  state.activeChip = null;
  state.specView = null;
  clearCameraMotion();
  state.cam = resetCam();
  const m = state.meta;
  document.getElementById("subtitle").textContent =
    `${m.pos_file} — ${(m.atom_count / 1e6).toFixed(2)}M atoms, ` +
    `${(m.raw_size_bytes / 1e6).toFixed(1)} MB raw — colors follow mass (uap-archive palette); ` +
    `range selections show every stored atom up to ${(BUDGET / 1e6).toFixed(0)}M points per panel`;

  for (const p of state.panels) {
    p.select.innerHTML = "";
    for (const mm of m.methods) {
      if (!mm.has_payload) continue;
      const opt = document.createElement("option");
      opt.value = mm.id;
      opt.textContent = `${mm.id} — ${(mm.size_bytes / 1e6).toFixed(2)} MB`;
      p.select.appendChild(opt);
    }
  }
  renderSpeciesChips();
  renderSelChips();
  renderTable();
  renderScatter();
  renderNotes();
  await loadPayload("original");   // spectrum baseline
  await setPanelMethod(state.panels[0], "original");
  const bDefault = [state.summary?.winner_id, "hybrid_exact", "massrange64"].find(
    (id) => methodMeta(id)?.has_payload) || "original";
  await setPanelMethod(state.panels[1], bDefault);
  drawSpectrum();
}

// ---------- boot ----------

async function main() {
  [state.index, state.summary] = await Promise.all([
    fetchJSON("data/index.json"),
    fetchJSON("data/summary.json").catch(() => null),
  ]);
  renderAggregate();
  const dsSel = document.getElementById("dataset-select");
  for (const d of state.index) {
    const opt = document.createElement("option");
    opt.value = d.slug;
    opt.textContent = `${d.name} (${(d.atom_count / 1e6).toFixed(1)}M atoms)`;
    dsSel.appendChild(opt);
  }
  dsSel.addEventListener("change", () => loadDataset(dsSel.value));

  const panelA = makePanel(document.getElementById("panel-A"), "A");
  const panelB = makePanel(document.getElementById("panel-B"), "B");
  state.panels = [panelA, panelB];
  for (const p of state.panels) {
    p.select.addEventListener("change", () => setPanelMethod(p, p.select.value));
    setupCloudInteraction(p);
  }
  spec.canvas = document.getElementById("spectrum");
  representationMap.canvas = document.getElementById("representation-map");
  setupSpectrumInteraction();
  setupRepresentationInteraction();

  const sizeSlider = document.getElementById("point-size");
  const sizeAuto = document.getElementById("size-auto");
  sizeSlider.addEventListener("input", () => {
    state.sizeAuto = false;
    sizeAuto.checked = false;
    state.manualSize = +sizeSlider.value;
    drawAll();
  });
  sizeAuto.addEventListener("change", () => {
    state.sizeAuto = sizeAuto.checked;
    drawAll();
  });
  const opSlider = document.getElementById("opacity");
  const opAuto = document.getElementById("opacity-auto");
  opSlider.addEventListener("input", () => {
    state.opacityAuto = false;
    opAuto.checked = false;
    state.manualOpacity = +opSlider.value;
    drawAll();
  });
  opAuto.addEventListener("change", () => {
    state.opacityAuto = opAuto.checked;
    drawAll();
  });
  document.getElementById("reset-view").addEventListener("click", () => {
    clearCameraMotion();
    state.cam = resetCam();
    drawAll();
  });
  document.getElementById("clear-sel").addEventListener("click", () => {
    state.ranges = [];
    state.activeChip = null;
    onSelectionChanged();
  });
  window.addEventListener("resize", () => { drawAll(); drawSpectrum(); });

  await loadDataset(state.index[0].slug);
}

main();
