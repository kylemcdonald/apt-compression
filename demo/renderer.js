const VERTEX_SHADER = `
attribute vec3 aPosition;
attribute float aMass;
uniform vec3 uCenter;
uniform float uScale;
uniform float uYaw;
uniform float uPitch;
uniform float uZoom;
uniform float uAspect;
uniform float uPointSize;
varying vec3 vColor;

vec3 palette(float t) {
  t = clamp(t, 0.0, 1.0);
  vec3 a = vec3(0.37, 0.34, 0.95);
  vec3 b = vec3(0.00, 0.82, 0.91);
  vec3 c = vec3(0.35, 0.91, 0.35);
  vec3 d = vec3(1.00, 0.72, 0.08);
  vec3 e = vec3(0.94, 0.18, 0.10);
  if (t < 0.25) return mix(a, b, t * 4.0);
  if (t < 0.50) return mix(b, c, (t - 0.25) * 4.0);
  if (t < 0.75) return mix(c, d, (t - 0.50) * 4.0);
  return mix(d, e, (t - 0.75) * 4.0);
}

void main() {
  vec3 p = (aPosition - uCenter) * uScale;
  float cy = cos(uYaw);
  float sy = sin(uYaw);
  float cp = cos(uPitch);
  float sp = sin(uPitch);
  float xr = cy * p.x - sy * p.y;
  float yr = sy * p.x + cy * p.y;
  float ys = cp * p.z - sp * yr;
  gl_Position = vec4(xr * uZoom / uAspect, ys * uZoom, 0.0, 1.0);
  gl_PointSize = uPointSize;
  vColor = palette(aMass / 120.0);
}`;

const FRAGMENT_SHADER = `
precision mediump float;
uniform float uOpacity;
varying vec3 vColor;

void main() {
  vec2 p = gl_PointCoord - vec2(0.5);
  if (dot(p, p) > 0.25) discard;
  gl_FragColor = vec4(vColor, uOpacity);
}`;

function shader(gl, type, source) {
  const value = gl.createShader(type);
  gl.shaderSource(value, source);
  gl.compileShader(value);
  if (!gl.getShaderParameter(value, gl.COMPILE_STATUS)) {
    throw new Error(gl.getShaderInfoLog(value) || "WebGL shader compilation failed");
  }
  return value;
}

function program(gl) {
  const value = gl.createProgram();
  gl.attachShader(value, shader(gl, gl.VERTEX_SHADER, VERTEX_SHADER));
  gl.attachShader(value, shader(gl, gl.FRAGMENT_SHADER, FRAGMENT_SHADER));
  gl.linkProgram(value);
  if (!gl.getProgramParameter(value, gl.LINK_STATUS)) {
    throw new Error(gl.getProgramInfoLog(value) || "WebGL program linking failed");
  }
  return value;
}

export function createSharedCamera() {
  return {
    yaw: -0.72,
    pitch: 0.34,
    zoom: 0.72,
    views: new Set(),
  };
}

function redraw(camera) {
  for (const view of camera.views) view.draw();
}

export class CloudRenderer {
  constructor(canvas, camera) {
    this.canvas = canvas;
    this.camera = camera;
    this.gl = canvas.getContext("webgl", {
      antialias: false,
      alpha: false,
      preserveDrawingBuffer: false,
    });
    if (!this.gl) throw new Error("WebGL is unavailable");
    this.program = program(this.gl);
    this.buffer = this.gl.createBuffer();
    this.count = 0;
    this.center = [0, 0, 0];
    this.scale = 1;
    this.locations = {
      position: this.gl.getAttribLocation(this.program, "aPosition"),
      mass: this.gl.getAttribLocation(this.program, "aMass"),
      center: this.gl.getUniformLocation(this.program, "uCenter"),
      scale: this.gl.getUniformLocation(this.program, "uScale"),
      yaw: this.gl.getUniformLocation(this.program, "uYaw"),
      pitch: this.gl.getUniformLocation(this.program, "uPitch"),
      zoom: this.gl.getUniformLocation(this.program, "uZoom"),
      aspect: this.gl.getUniformLocation(this.program, "uAspect"),
      pointSize: this.gl.getUniformLocation(this.program, "uPointSize"),
      opacity: this.gl.getUniformLocation(this.program, "uOpacity"),
    };
    camera.views.add(this);
    this.bindInteraction();
    this.resizeObserver = new ResizeObserver(() => this.draw());
    this.resizeObserver.observe(canvas);
  }

  bindInteraction() {
    let dragging = false;
    let previousX = 0;
    let previousY = 0;
    this.canvas.addEventListener("pointerdown", (event) => {
      dragging = true;
      previousX = event.clientX;
      previousY = event.clientY;
      this.canvas.setPointerCapture(event.pointerId);
    });
    this.canvas.addEventListener("pointermove", (event) => {
      if (!dragging) return;
      this.camera.yaw += (event.clientX - previousX) * 0.008;
      this.camera.pitch = Math.max(
        -1.45,
        Math.min(1.45, this.camera.pitch + (event.clientY - previousY) * 0.008),
      );
      previousX = event.clientX;
      previousY = event.clientY;
      redraw(this.camera);
    });
    const end = () => {
      dragging = false;
    };
    this.canvas.addEventListener("pointerup", end);
    this.canvas.addEventListener("pointercancel", end);
    this.canvas.addEventListener("wheel", (event) => {
      event.preventDefault();
      this.camera.zoom = Math.max(
        0.15,
        Math.min(8, this.camera.zoom * Math.exp(-event.deltaY * 0.001)),
      );
      redraw(this.camera);
    }, { passive: false });
    this.canvas.addEventListener("dblclick", () => {
      this.camera.yaw = -0.72;
      this.camera.pitch = 0.34;
      this.camera.zoom = 0.72;
      redraw(this.camera);
    });
  }

  setPoints(points) {
    if (!(points instanceof Float32Array) || points.length % 4 !== 0) {
      throw new TypeError("renderer points must be an interleaved Float32Array");
    }
    this.count = points.length / 4;
    if (this.count === 0) {
      this.draw();
      return;
    }
    const minimum = [Infinity, Infinity, Infinity];
    const maximum = [-Infinity, -Infinity, -Infinity];
    for (let index = 0; index < points.length; index += 4) {
      for (let axis = 0; axis < 3; axis += 1) {
        minimum[axis] = Math.min(minimum[axis], points[index + axis]);
        maximum[axis] = Math.max(maximum[axis], points[index + axis]);
      }
    }
    this.center = minimum.map((value, axis) => (value + maximum[axis]) * 0.5);
    const extent = maximum.map((value, axis) => value - minimum[axis]);
    this.scale = 1 / Math.max(...extent, 1e-9);
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
    gl.bufferData(gl.ARRAY_BUFFER, points, gl.STATIC_DRAW);
    this.draw();
  }

  clear() {
    this.count = 0;
    this.draw();
  }

  draw() {
    const gl = this.gl;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(1, Math.round(this.canvas.clientWidth * ratio));
    const height = Math.max(1, Math.round(this.canvas.clientHeight * ratio));
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
    }
    gl.viewport(0, 0, width, height);
    gl.clearColor(0.027, 0.031, 0.027, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    if (!this.count) return;

    gl.useProgram(this.program);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
    gl.enableVertexAttribArray(this.locations.position);
    gl.vertexAttribPointer(this.locations.position, 3, gl.FLOAT, false, 16, 0);
    gl.enableVertexAttribArray(this.locations.mass);
    gl.vertexAttribPointer(this.locations.mass, 1, gl.FLOAT, false, 16, 12);
    gl.uniform3fv(this.locations.center, this.center);
    gl.uniform1f(this.locations.scale, this.scale);
    gl.uniform1f(this.locations.yaw, this.camera.yaw);
    gl.uniform1f(this.locations.pitch, this.camera.pitch);
    gl.uniform1f(this.locations.zoom, this.camera.zoom);
    gl.uniform1f(this.locations.aspect, width / height);
    gl.uniform1f(this.locations.pointSize, Math.max(1.2, ratio * 0.9));
    gl.uniform1f(
      this.locations.opacity,
      Math.max(0.035, Math.min(0.18, 18000 / this.count)),
    );
    gl.disable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.drawArrays(gl.POINTS, 0, this.count);
  }
}

function resizeCanvas(canvas) {
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.round(canvas.clientWidth * ratio));
  const height = Math.max(1, Math.round(canvas.clientHeight * ratio));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  return { width, height, ratio };
}

function pixelMaximum(counts, binsPerPixel, pixel) {
  const start = Math.floor(pixel * binsPerPixel);
  const end = Math.min(counts.length, Math.ceil((pixel + 1) * binsPerPixel));
  let maximum = 0;
  for (let bin = start; bin < end; bin += 1) {
    maximum = Math.max(maximum, counts[bin]);
  }
  return maximum;
}

export function drawSpectrum(canvas, trueCounts, storedCounts, {
  binWidth = 0.05,
  maxMass = 120,
} = {}) {
  const { width, height, ratio } = resizeCanvas(canvas);
  const context = canvas.getContext("2d");
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#111110";
  context.fillRect(0, 0, width, height);
  if (!trueCounts || !storedCounts) return;

  const padding = {
    left: 42 * ratio,
    right: 10 * ratio,
    top: 9 * ratio,
    bottom: 22 * ratio,
  };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  const visibleBins = Math.min(trueCounts.length, Math.ceil(maxMass / binWidth));
  const trueView = trueCounts.subarray(0, visibleBins);
  const storedView = storedCounts.subarray(0, visibleBins);
  let trueTotal = 0;
  let storedTotal = 0;
  for (const count of trueView) trueTotal += count;
  for (const count of storedView) storedTotal += count;
  const retainedScale = storedTotal > 0 ? trueTotal / storedTotal : 1;
  let peak = 1;
  for (const count of trueView) peak = Math.max(peak, count);
  for (const count of storedView) {
    peak = Math.max(peak, count * retainedScale);
  }
  const logPeak = Math.log1p(peak);
  const binsPerPixel = visibleBins / plotWidth;

  context.strokeStyle = "rgba(255,255,255,0.10)";
  context.lineWidth = ratio;
  context.beginPath();
  context.moveTo(padding.left, padding.top + plotHeight);
  context.lineTo(padding.left + plotWidth, padding.top + plotHeight);
  context.stroke();

  function trace(counts, color, scale = 1) {
    context.strokeStyle = color;
    context.lineWidth = ratio;
    context.beginPath();
    for (let x = 0; x < plotWidth; x += 1) {
      const count = pixelMaximum(counts, binsPerPixel, x) * scale;
      const y = padding.top + plotHeight * (1 - Math.log1p(count) / logPeak);
      if (x === 0) context.moveTo(padding.left + x, y);
      else context.lineTo(padding.left + x, y);
    }
    context.stroke();
  }

  trace(trueView, "#898781");
  trace(storedView, "#3987e5", retainedScale);
  context.fillStyle = "#898781";
  context.font = `${10 * ratio}px system-ui`;
  context.textAlign = "center";
  context.textBaseline = "top";
  for (const mass of [0, 20, 40, 60, 80, 100, 120]) {
    const x = padding.left + mass / maxMass * plotWidth;
    context.fillText(`${mass}`, x, padding.top + plotHeight + 5 * ratio);
  }
  context.save();
  context.translate(11 * ratio, padding.top + plotHeight / 2);
  context.rotate(-Math.PI / 2);
  context.textBaseline = "top";
  context.fillText("log count", 0, 0);
  context.restore();
  context.textAlign = "right";
  context.fillText("m/z (Da)", padding.left + plotWidth, height - 12 * ratio);
}
