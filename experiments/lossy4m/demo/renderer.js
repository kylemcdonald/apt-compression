const VERTEX_SHADER = `
attribute vec3 aPosition;
attribute float aMass;
attribute float aExact;
uniform vec3 uCenter;
uniform float uScale;
uniform float uYaw;
uniform float uPitch;
uniform float uZoom;
uniform vec2 uPan;
uniform float uAspect;
uniform float uPointSize;
uniform float uColorMode;
uniform float uPerspective;
uniform float uMassMin;
uniform float uMassMax;
varying vec3 vColor;
varying float vVisible;

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
  vVisible = (aMass >= uMassMin && aMass <= uMassMax) ? 1.0 : 0.0;

  // APT reconstructions are conventionally tall along z. Put z on the
  // horizontal image axis, matching Rangefinder's point-image orientation.
  vec3 p = vec3(
    aPosition.z - uCenter.z,
    aPosition.x - uCenter.x,
    aPosition.y - uCenter.y
  ) * uScale;

  float cy = cos(uYaw);
  float sy = sin(uYaw);
  float cp = cos(uPitch);
  float sp = sin(uPitch);
  float xr = cy * p.x + sy * p.z;
  float zr = -sy * p.x + cy * p.z;
  float yr = cp * p.y - sp * zr;
  float zd = sp * p.y + cp * zr;

  vec2 orthographic = vec2(xr / uAspect, yr) * uZoom;
  float cameraDistance = 2.4;
  float perspectiveScale = 1.6 / max(0.25, cameraDistance - zd);
  vec2 perspective = orthographic * perspectiveScale;
  vec2 projected = mix(orthographic, perspective, uPerspective) + uPan;
  gl_Position = vec4(projected, 0.0, 1.0);
  gl_PointSize = max(0.25, uPointSize);

  vec3 exactColor = vec3(0.20, 0.66, 0.96);
  vec3 synthesizedColor = vec3(0.98, 0.25, 0.08);
  vec3 provenance = mix(synthesizedColor, exactColor, aExact);
  vColor = mix(palette(aMass / 120.0), provenance, uColorMode);
}`;

const FRAGMENT_SHADER = `
precision mediump float;
uniform float uOpacity;
varying vec3 vColor;
varying float vVisible;

void main() {
  if (vVisible < 0.5 || uOpacity <= 0.0) discard;
  vec2 p = gl_PointCoord - vec2(0.5);
  if (dot(p, p) > 0.25) discard;
  gl_FragColor = vec4(vColor, uOpacity);
}`;

function compileShader(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    throw new Error(
      gl.getShaderInfoLog(shader) || "WebGL shader compilation failed",
    );
  }
  return shader;
}

function createProgram(gl) {
  const program = gl.createProgram();
  gl.attachShader(program, compileShader(gl, gl.VERTEX_SHADER, VERTEX_SHADER));
  gl.attachShader(
    program,
    compileShader(gl, gl.FRAGMENT_SHADER, FRAGMENT_SHADER),
  );
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(
      gl.getProgramInfoLog(program) || "WebGL program linking failed",
    );
  }
  return program;
}

function redraw(camera) {
  for (const view of camera.views) view.draw();
}

export function createSharedCamera() {
  return {
    yaw: 0.08,
    pitch: -0.1,
    zoom: 1,
    panX: 0,
    panY: 0,
    perspective: true,
    views: new Set(),
    reset() {
      this.yaw = 0.08;
      this.pitch = -0.1;
      this.zoom = 1;
      this.panX = 0;
      this.panY = 0;
      redraw(this);
    },
  };
}

export class PointCloudRenderer {
  constructor(canvas, camera, {
    colorMode = "mass",
    pointSize = 0.7,
    opacity = 0.05,
    additive = true,
  } = {}) {
    this.canvas = canvas;
    this.camera = camera;
    this.colorMode = colorMode;
    this.pointSize = pointSize;
    this.opacity = opacity;
    this.additive = additive;
    this.massMin = -1e9;
    this.massMax = 1e9;
    this.gl = canvas.getContext("webgl", {
      antialias: false,
      alpha: false,
      preserveDrawingBuffer: false,
      powerPreference: "high-performance",
    });
    if (!this.gl) throw new Error("WebGL is unavailable");
    this.program = createProgram(this.gl);
    this.pointBuffer = this.gl.createBuffer();
    this.exactBuffer = this.gl.createBuffer();
    this.count = 0;
    this.center = [0, 0, 0];
    this.extent = [1, 1, 1];
    this.locations = {
      position: this.gl.getAttribLocation(this.program, "aPosition"),
      mass: this.gl.getAttribLocation(this.program, "aMass"),
      exact: this.gl.getAttribLocation(this.program, "aExact"),
      center: this.gl.getUniformLocation(this.program, "uCenter"),
      scale: this.gl.getUniformLocation(this.program, "uScale"),
      yaw: this.gl.getUniformLocation(this.program, "uYaw"),
      pitch: this.gl.getUniformLocation(this.program, "uPitch"),
      zoom: this.gl.getUniformLocation(this.program, "uZoom"),
      pan: this.gl.getUniformLocation(this.program, "uPan"),
      aspect: this.gl.getUniformLocation(this.program, "uAspect"),
      pointSize: this.gl.getUniformLocation(this.program, "uPointSize"),
      opacity: this.gl.getUniformLocation(this.program, "uOpacity"),
      colorMode: this.gl.getUniformLocation(this.program, "uColorMode"),
      perspective: this.gl.getUniformLocation(this.program, "uPerspective"),
      massMin: this.gl.getUniformLocation(this.program, "uMassMin"),
      massMax: this.gl.getUniformLocation(this.program, "uMassMax"),
    };
    camera.views.add(this);
    this.bindInteraction();
    this.resizeObserver = new ResizeObserver(() => this.draw());
    this.resizeObserver.observe(canvas);
  }

  bindInteraction() {
    let drag = null;
    this.canvas.addEventListener("contextmenu", (event) => {
      event.preventDefault();
    });
    this.canvas.addEventListener("pointerdown", (event) => {
      drag = {
        x: event.clientX,
        y: event.clientY,
        pan: event.button !== 0 || event.shiftKey,
      };
      this.canvas.setPointerCapture(event.pointerId);
    });
    this.canvas.addEventListener("pointermove", (event) => {
      if (!drag) return;
      const dx = event.clientX - drag.x;
      const dy = event.clientY - drag.y;
      if (drag.pan) {
        this.camera.panX += dx / Math.max(1, this.canvas.clientWidth) * 2;
        this.camera.panY -= dy / Math.max(1, this.canvas.clientHeight) * 2;
      } else {
        this.camera.yaw += dx * 0.007;
        this.camera.pitch = Math.max(
          -1.48,
          Math.min(1.48, this.camera.pitch + dy * 0.007),
        );
      }
      drag.x = event.clientX;
      drag.y = event.clientY;
      redraw(this.camera);
    });
    const end = () => {
      drag = null;
    };
    this.canvas.addEventListener("pointerup", end);
    this.canvas.addEventListener("pointercancel", end);
    this.canvas.addEventListener("wheel", (event) => {
      event.preventDefault();
      this.camera.zoom = Math.max(
        0.08,
        Math.min(40, this.camera.zoom * Math.exp(-event.deltaY * 0.001)),
      );
      redraw(this.camera);
    }, { passive: false });
    this.canvas.addEventListener("dblclick", () => this.camera.reset());
  }

  clear() {
    this.count = 0;
    this.draw();
  }

  setColorMode(mode) {
    if (mode !== "mass" && mode !== "provenance") {
      throw new Error("color mode must be mass or provenance");
    }
    this.colorMode = mode;
    this.draw();
  }

  setAppearance({
    pointSize = this.pointSize,
    opacity = this.opacity,
    additive = this.additive,
    perspective = this.camera.perspective,
  } = {}) {
    this.pointSize = Math.max(0.25, Math.min(10, Number(pointSize)));
    this.opacity = Math.max(1 / 65536, Math.min(1, Number(opacity)));
    this.additive = Boolean(additive);
    this.camera.perspective = Boolean(perspective);
    redraw(this.camera);
  }

  setMassWindow(window) {
    this.massMin = window ? window.min : -1e9;
    this.massMax = window ? window.max : 1e9;
    this.draw();
  }

  setBounds(minimum, maximum) {
    if (
      !Array.isArray(minimum)
      || !Array.isArray(maximum)
      || minimum.length < 3
      || maximum.length < 3
    ) {
      throw new TypeError("bounds must contain three spatial dimensions");
    }
    this.center = [0, 1, 2].map(
      (axis) => (Number(minimum[axis]) + Number(maximum[axis])) * 0.5,
    );
    this.extent = [0, 1, 2].map(
      (axis) => Number(maximum[axis]) - Number(minimum[axis]),
    );
    if (
      this.center.some((value) => !Number.isFinite(value))
      || this.extent.some((value) => !Number.isFinite(value) || value < 0)
    ) {
      throw new Error("bounds must be finite and ordered");
    }
    this.draw();
  }

  setPoints(points, exact = null) {
    if (!(points instanceof Float32Array) || points.length % 4 !== 0) {
      throw new TypeError("points must be an interleaved Float32Array");
    }
    this.count = points.length / 4;
    const exactValues = exact || new Uint8Array(this.count).fill(1);
    if (exactValues.length !== this.count) {
      throw new Error("provenance length does not match points");
    }
    if (!this.count) {
      this.draw();
      return;
    }
    const minimum = [Infinity, Infinity, Infinity];
    const maximum = [-Infinity, -Infinity, -Infinity];
    for (let record = 0; record < points.length; record += 4) {
      for (let axis = 0; axis < 3; axis += 1) {
        minimum[axis] = Math.min(minimum[axis], points[record + axis]);
        maximum[axis] = Math.max(maximum[axis], points[record + axis]);
      }
    }
    this.center = minimum.map((value, axis) => (
      (value + maximum[axis]) * 0.5
    ));
    this.extent = minimum.map((value, axis) => maximum[axis] - value);
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.pointBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, points, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.exactBuffer);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      Float32Array.from(exactValues),
      gl.STATIC_DRAW,
    );
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

    const aspect = width / height;
    const horizontalExtent = this.extent[2];
    const verticalExtent = this.extent[0];
    const depthExtent = this.extent[1];
    const fitExtent = Math.max(
      horizontalExtent / Math.max(aspect, 1e-6),
      verticalExtent,
      depthExtent * 0.45,
      1e-9,
    );
    const fitScale = 1.7 / fitExtent;

    gl.useProgram(this.program);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.pointBuffer);
    gl.enableVertexAttribArray(this.locations.position);
    gl.vertexAttribPointer(this.locations.position, 3, gl.FLOAT, false, 16, 0);
    gl.enableVertexAttribArray(this.locations.mass);
    gl.vertexAttribPointer(this.locations.mass, 1, gl.FLOAT, false, 16, 12);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.exactBuffer);
    gl.enableVertexAttribArray(this.locations.exact);
    gl.vertexAttribPointer(this.locations.exact, 1, gl.FLOAT, false, 4, 0);
    gl.uniform3fv(this.locations.center, this.center);
    gl.uniform1f(this.locations.scale, fitScale);
    gl.uniform1f(this.locations.yaw, this.camera.yaw);
    gl.uniform1f(this.locations.pitch, this.camera.pitch);
    gl.uniform1f(this.locations.zoom, this.camera.zoom);
    gl.uniform2f(
      this.locations.pan,
      this.camera.panX,
      this.camera.panY,
    );
    gl.uniform1f(this.locations.aspect, aspect);
    gl.uniform1f(this.locations.pointSize, this.pointSize * ratio);
    gl.uniform1f(this.locations.opacity, this.opacity);
    gl.uniform1f(
      this.locations.colorMode,
      this.colorMode === "provenance" ? 1 : 0,
    );
    gl.uniform1f(
      this.locations.perspective,
      this.camera.perspective ? 1 : 0,
    );
    gl.uniform1f(this.locations.massMin, this.massMin);
    gl.uniform1f(this.locations.massMax, this.massMax);
    gl.disable(gl.DEPTH_TEST);
    gl.depthMask(false);
    gl.enable(gl.BLEND);
    gl.blendFunc(
      gl.SRC_ALPHA,
      this.additive ? gl.ONE : gl.ONE_MINUS_SRC_ALPHA,
    );
    gl.drawArrays(gl.POINTS, 0, this.count);
  }
}
