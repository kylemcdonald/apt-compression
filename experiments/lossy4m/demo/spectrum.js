const COLORS = Object.freeze({
  background: "#10110f",
  grid: "rgba(255,255,255,0.08)",
  text: "#898781",
  source: "#aaa9a2",
  cpos: "#4ba3f2",
  cp4m: "#75d06f",
  exact: "#36a6f4",
  synthesized: "#ef6534",
  selection: "rgba(255, 207, 74, 0.14)",
  selectionEdge: "#ffcf4a",
});

function clamp(value, minimum, maximum) {
  return Math.max(minimum, Math.min(maximum, value));
}

function total(values) {
  let result = 0;
  if (values) {
    for (const value of values) result += value;
  }
  return result;
}

function formatInteger(value) {
  return new Intl.NumberFormat("en-US", {
    maximumFractionDigits: 0,
  }).format(value);
}

function niceStep(span, target = 6) {
  const rough = span / target;
  const magnitude = 10 ** Math.floor(Math.log10(Math.max(rough, 1e-12)));
  const normalized = rough / magnitude;
  const multiplier = normalized <= 1 ? 1 : normalized <= 2 ? 2 : (
    normalized <= 5 ? 5 : 10
  );
  return multiplier * magnitude;
}

export class SpectrumView {
  constructor(canvas, tooltip, { onSelection = null } = {}) {
    this.canvas = canvas;
    this.tooltip = tooltip;
    this.onSelection = onSelection;
    this.data = null;
    this.viewMin = 0;
    this.viewMax = 120;
    this.fullMax = 120;
    this.selection = null;
    this.hoverMass = null;
    this.yBlend = 1;
    this.drag = null;
    this.padding = { left: 52, right: 14, top: 12, bottom: 28 };
    this.bindInteraction();
    this.resizeObserver = new ResizeObserver(() => this.draw());
    this.resizeObserver.observe(canvas);
  }

  setData({
    binWidth,
    trueCounts,
    cposStoredCounts,
    cp4mStoredCounts,
  }) {
    if (
      !(trueCounts instanceof Uint32Array)
      || !(cposStoredCounts instanceof Uint32Array)
      || !(cp4mStoredCounts instanceof Uint32Array)
      || trueCounts.length !== cposStoredCounts.length
      || trueCounts.length !== cp4mStoredCounts.length
    ) {
      throw new TypeError("spectrum tables must be equally sized Uint32Arrays");
    }
    const sourceTotal = total(trueCounts);
    this.data = {
      binWidth,
      trueCounts,
      cposStoredCounts,
      cp4mStoredCounts,
      sourceTotal,
      cposScale: sourceTotal / Math.max(1, total(cposStoredCounts)),
      cp4mScale: sourceTotal / Math.max(1, total(cp4mStoredCounts)),
    };
    let last = trueCounts.length - 1;
    while (last > 0 && trueCounts[last] === 0) last -= 1;
    this.fullMax = clamp(
      Math.ceil((last + 1) * binWidth / 5) * 5,
      Math.min(10, trueCounts.length * binWidth),
      trueCounts.length * binWidth,
    );
    this.viewMin = 0;
    this.viewMax = this.fullMax;
    this.selection = null;
    this.hoverMass = null;
    this.hideTooltip();
    this.draw();
  }

  setSelection(selection, { notify = false } = {}) {
    if (!selection) {
      this.selection = null;
    } else {
      const minimum = clamp(
        Math.min(selection.min, selection.max),
        0,
        this.data ? this.data.trueCounts.length * this.data.binWidth : 300,
      );
      const maximum = clamp(
        Math.max(selection.min, selection.max),
        minimum,
        this.data ? this.data.trueCounts.length * this.data.binWidth : 300,
      );
      this.selection = { min: minimum, max: maximum };
    }
    this.draw();
    if (notify && this.onSelection) this.onSelection(this.selection);
  }

  reset() {
    if (!this.data) return;
    this.viewMin = 0;
    this.viewMax = this.fullMax;
    this.setSelection(null, { notify: true });
  }

  geometry() {
    const width = Math.max(1, this.canvas.clientWidth);
    const height = Math.max(1, this.canvas.clientHeight);
    return {
      width,
      height,
      plotLeft: this.padding.left,
      plotTop: this.padding.top,
      plotWidth: Math.max(1, width - this.padding.left - this.padding.right),
      plotHeight: Math.max(1, height - this.padding.top - this.padding.bottom),
    };
  }

  massAt(clientX) {
    const rect = this.canvas.getBoundingClientRect();
    const geometry = this.geometry();
    const fraction = clamp(
      (clientX - rect.left - geometry.plotLeft) / geometry.plotWidth,
      0,
      1,
    );
    return this.viewMin + fraction * (this.viewMax - this.viewMin);
  }

  xAt(mass, geometry) {
    return (
      geometry.plotLeft
      + (mass - this.viewMin) / (this.viewMax - this.viewMin)
      * geometry.plotWidth
    );
  }

  inAxisGutter(clientX) {
    const rect = this.canvas.getBoundingClientRect();
    return clientX - rect.left < this.padding.left;
  }

  bindInteraction() {
    this.canvas.addEventListener("pointerdown", (event) => {
      event.preventDefault();
      const axis = this.inAxisGutter(event.clientX);
      this.drag = axis ? {
        mode: "axis",
        startY: event.clientY,
        startBlend: this.yBlend,
      } : {
        mode: "brush",
        startMass: this.massAt(event.clientX),
        currentMass: this.massAt(event.clientX),
      };
      this.canvas.setPointerCapture(event.pointerId);
      this.canvas.style.cursor = axis ? "ns-resize" : "crosshair";
    });
    this.canvas.addEventListener("pointermove", (event) => {
      if (!this.data) return;
      this.hoverMass = this.massAt(event.clientX);
      if (this.drag?.mode === "axis") {
        this.yBlend = clamp(
          this.drag.startBlend
          + (this.drag.startY - event.clientY) / 120,
          0,
          1,
        );
        this.hideTooltip();
      } else if (this.drag?.mode === "brush") {
        this.drag.currentMass = this.hoverMass;
        const selection = {
          min: Math.min(this.drag.startMass, this.drag.currentMass),
          max: Math.max(this.drag.startMass, this.drag.currentMass),
        };
        this.setSelection(selection, { notify: true });
      } else if (this.inAxisGutter(event.clientX)) {
        this.canvas.style.cursor = "ns-resize";
        this.hideTooltip();
      } else {
        this.canvas.style.cursor = "crosshair";
        this.showTooltip(event);
      }
      this.draw();
    });
    const endDrag = () => {
      if (this.drag?.mode === "brush") {
        const width = Math.abs(
          this.drag.currentMass - this.drag.startMass,
        );
        if (width < this.data.binWidth * 0.5) {
          this.setSelection(null, { notify: true });
        }
      }
      this.drag = null;
    };
    this.canvas.addEventListener("pointerup", endDrag);
    this.canvas.addEventListener("pointercancel", endDrag);
    this.canvas.addEventListener("pointerleave", () => {
      if (!this.drag) {
        this.hoverMass = null;
        this.hideTooltip();
        this.draw();
      }
    });
    this.canvas.addEventListener("wheel", (event) => {
      if (!this.data) return;
      event.preventDefault();
      const domainMax = this.data.trueCounts.length * this.data.binWidth;
      const span = this.viewMax - this.viewMin;
      if (
        event.shiftKey
        || Math.abs(event.deltaX) > Math.abs(event.deltaY)
      ) {
        const movement = (
          event.deltaX + (event.shiftKey ? event.deltaY : 0)
        ) / Math.max(1, this.canvas.clientWidth) * span;
        const nextMin = clamp(this.viewMin + movement, 0, domainMax - span);
        this.viewMin = nextMin;
        this.viewMax = nextMin + span;
      } else {
        const anchor = this.massAt(event.clientX);
        const nextSpan = clamp(
          span * Math.exp(event.deltaY * 0.0015),
          this.data.binWidth * 10,
          domainMax,
        );
        const anchorFraction = (anchor - this.viewMin) / span;
        let nextMin = anchor - anchorFraction * nextSpan;
        nextMin = clamp(nextMin, 0, domainMax - nextSpan);
        this.viewMin = nextMin;
        this.viewMax = nextMin + nextSpan;
      }
      this.hideTooltip();
      this.draw();
    }, { passive: false });
    this.canvas.addEventListener("dblclick", () => this.reset());
  }

  showTooltip(event) {
    if (!this.data || !this.tooltip) return;
    const bin = clamp(
      Math.floor(this.hoverMass / this.data.binWidth),
      0,
      this.data.trueCounts.length - 1,
    );
    const low = bin * this.data.binWidth;
    const high = low + this.data.binWidth;
    const source = this.data.trueCounts[bin];
    const cpos = this.data.cposStoredCounts[bin];
    const exact = this.data.cp4mStoredCounts[bin];
    const synthesized = source - exact;
    const exactRate = source ? exact / source * 100 : 0;
    this.tooltip.textContent = (
      `${low.toFixed(2)}–${high.toFixed(2)} Da\n`
      + `source       ${formatInteger(source)}\n`
      + `CPOS kept    ${formatInteger(cpos)}\n`
      + `CP4M exact   ${formatInteger(exact)} (${exactRate.toFixed(1)}%)\n`
      + `synthesized  ${formatInteger(synthesized)}`
    );
    const rect = this.canvas.getBoundingClientRect();
    this.tooltip.hidden = false;
    this.tooltip.style.left = `${clamp(
      event.clientX - rect.left + 12,
      4,
      rect.width - 210,
    )}px`;
    this.tooltip.style.top = `${clamp(
      event.clientY - rect.top - 92,
      4,
      rect.height - 100,
    )}px`;
  }

  hideTooltip() {
    if (this.tooltip) this.tooltip.hidden = true;
  }

  drawTrace(context, values, scale, maximum, geometry, color, width) {
    const { binWidth } = this.data;
    const first = clamp(
      Math.floor(this.viewMin / binWidth),
      0,
      values.length - 1,
    );
    const last = clamp(
      Math.ceil(this.viewMax / binWidth),
      first + 1,
      values.length,
    );
    context.beginPath();
    let started = false;
    for (let bin = first; bin < last; bin += 1) {
      const x = this.xAt((bin + 0.5) * binWidth, geometry);
      const value = values[bin] * scale;
      const linear = value / Math.max(maximum, 1);
      const logarithmic = Math.log1p(value) / Math.log1p(Math.max(maximum, 1));
      const normalized = (
        linear * (1 - this.yBlend) + logarithmic * this.yBlend
      );
      const y = (
        geometry.plotTop
        + geometry.plotHeight
        - normalized * (geometry.plotHeight - 6)
      );
      if (!started) {
        context.moveTo(x, y);
        started = true;
      } else {
        context.lineTo(x, y);
      }
    }
    context.strokeStyle = color;
    context.lineWidth = width;
    context.stroke();
  }

  draw() {
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const geometry = this.geometry();
    const pixelWidth = Math.max(1, Math.round(geometry.width * ratio));
    const pixelHeight = Math.max(1, Math.round(geometry.height * ratio));
    if (
      this.canvas.width !== pixelWidth
      || this.canvas.height !== pixelHeight
    ) {
      this.canvas.width = pixelWidth;
      this.canvas.height = pixelHeight;
    }
    const context = this.canvas.getContext("2d");
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, geometry.width, geometry.height);
    context.fillStyle = COLORS.background;
    context.fillRect(0, 0, geometry.width, geometry.height);
    if (!this.data) return;

    const { binWidth, trueCounts, cposStoredCounts, cp4mStoredCounts } = this.data;
    const first = clamp(
      Math.floor(this.viewMin / binWidth),
      0,
      trueCounts.length - 1,
    );
    const last = clamp(
      Math.ceil(this.viewMax / binWidth),
      first + 1,
      trueCounts.length,
    );
    let maximum = 1;
    for (let bin = first; bin < last; bin += 1) {
      maximum = Math.max(
        maximum,
        trueCounts[bin],
        cposStoredCounts[bin] * this.data.cposScale,
        cp4mStoredCounts[bin] * this.data.cp4mScale,
      );
    }

    context.strokeStyle = COLORS.grid;
    context.lineWidth = 1;
    for (let line = 0; line <= 4; line += 1) {
      const y = (
        geometry.plotTop + geometry.plotHeight * line / 4
      );
      context.beginPath();
      context.moveTo(geometry.plotLeft, y);
      context.lineTo(geometry.plotLeft + geometry.plotWidth, y);
      context.stroke();
    }

    const step = niceStep(this.viewMax - this.viewMin);
    const firstTick = Math.ceil(this.viewMin / step) * step;
    context.fillStyle = COLORS.text;
    context.font = "10px system-ui, sans-serif";
    context.textAlign = "center";
    context.textBaseline = "top";
    for (let mass = firstTick; mass <= this.viewMax + 1e-8; mass += step) {
      const x = this.xAt(mass, geometry);
      context.fillText(`${Number(mass.toFixed(4))}`, x, (
        geometry.plotTop + geometry.plotHeight + 7
      ));
    }
    context.textAlign = "right";
    context.textBaseline = "middle";
    context.fillText(
      this.yBlend > 0.98 ? "log" : this.yBlend < 0.02 ? "linear" : "mixed",
      geometry.plotLeft - 7,
      geometry.plotTop + 5,
    );
    context.fillText("0", geometry.plotLeft - 7, (
      geometry.plotTop + geometry.plotHeight
    ));

    if (this.selection) {
      const left = this.xAt(this.selection.min, geometry);
      const right = this.xAt(this.selection.max, geometry);
      context.fillStyle = COLORS.selection;
      context.fillRect(
        left,
        geometry.plotTop,
        right - left,
        geometry.plotHeight,
      );
      context.strokeStyle = COLORS.selectionEdge;
      context.beginPath();
      context.moveTo(left, geometry.plotTop);
      context.lineTo(left, geometry.plotTop + geometry.plotHeight);
      context.moveTo(right, geometry.plotTop);
      context.lineTo(right, geometry.plotTop + geometry.plotHeight);
      context.stroke();
    }

    this.drawTrace(
      context,
      trueCounts,
      1,
      maximum,
      geometry,
      COLORS.source,
      1,
    );
    this.drawTrace(
      context,
      cposStoredCounts,
      this.data.cposScale,
      maximum,
      geometry,
      COLORS.cpos,
      1.2,
    );
    this.drawTrace(
      context,
      cp4mStoredCounts,
      this.data.cp4mScale,
      maximum,
      geometry,
      COLORS.cp4m,
      1.2,
    );

    const stripY = geometry.plotTop + geometry.plotHeight - 4;
    for (let bin = first; bin < last; bin += 1) {
      const source = trueCounts[bin];
      if (!source) continue;
      const fraction = cp4mStoredCounts[bin] / source;
      const red = Math.round(239 + (54 - 239) * fraction);
      const green = Math.round(101 + (166 - 101) * fraction);
      const blue = Math.round(52 + (244 - 52) * fraction);
      const x0 = this.xAt(bin * binWidth, geometry);
      const x1 = this.xAt((bin + 1) * binWidth, geometry);
      context.fillStyle = `rgb(${red},${green},${blue})`;
      context.fillRect(x0, stripY, Math.max(1, x1 - x0), 4);
    }

    if (this.hoverMass != null && !this.drag) {
      const x = this.xAt(this.hoverMass, geometry);
      context.strokeStyle = "rgba(255,255,255,0.35)";
      context.beginPath();
      context.moveTo(x, geometry.plotTop);
      context.lineTo(x, geometry.plotTop + geometry.plotHeight);
      context.stroke();
    }
  }
}
