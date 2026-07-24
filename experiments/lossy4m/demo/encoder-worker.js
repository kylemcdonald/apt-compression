const sourceMode = new URL(import.meta.url).pathname.includes(
  "/experiments/lossy4m/demo/",
);
const cposSpecifier = sourceMode ? "../../../javascript/cpos.js" : "./cpos.js";
const cp4mSpecifier = sourceMode ? "../javascript/cp4m.js" : "./cp4m.js";
const encoderSpecifier = sourceMode
  ? "../javascript/cp4m-encode.js"
  : "./cp4m-encode.js";

const [
  {
    DEFAULT_MAX_POINTS,
    decodeCpos,
    encodePos,
    samplePosBySpectrum,
  },
  {
    decodeRetainedCp4m,
    expandCp4m,
  },
  {
    encodePosCp4m,
  },
] = await Promise.all([
  import(cposSpecifier),
  import(cp4mSpecifier),
  import(encoderSpecifier),
]);

let currentCp4m = null;

function errorMessage(error) {
  if (error instanceof Error) {
    return {
      message: error.message,
      stack: error.stack || "",
    };
  }
  return { message: String(error), stack: "" };
}

function transferableBuffer(value) {
  if (value instanceof ArrayBuffer) return value;
  if (
    ArrayBuffer.isView(value)
    && value.byteOffset === 0
    && value.byteLength === value.buffer.byteLength
  ) {
    return value.buffer;
  }
  if (ArrayBuffer.isView(value)) {
    return value.buffer.slice(value.byteOffset, value.byteOffset + value.byteLength);
  }
  throw new TypeError("expected a transferable byte buffer");
}

function postProgress(jobId, stage, fraction) {
  postMessage({
    type: "progress",
    jobId,
    stage,
    fraction: Math.max(0, Math.min(1, fraction)),
  });
}

function postDecodedResult({
  jobId,
  mode,
  name,
  originalBytes,
  originalPoints,
  cposPayload,
  cpos,
  cp4mPayload,
  cp4m,
  expanded,
}) {
  const cp4mTrueCounts = cp4m.trueCounts.slice();
  const cp4mStoredCounts = cp4m.storedCounts.slice();
  const cposBuffer = transferableBuffer(cposPayload);
  const cp4mBuffer = transferableBuffer(cp4mPayload);
  const message = {
    type: "result",
    jobId,
    mode,
    name,
    originalBytes,
    originalPoints,
    cposPayload: cposBuffer,
    cposPoints: cpos.points,
    cposTrueCounts: cpos.trueCounts,
    cposStoredCounts: cpos.storedCounts,
    cp4mPayload: cp4mBuffer,
    cp4mTrueCounts,
    cp4mStoredCounts,
    expandedPoints: expanded.points,
    expandedExact: expanded.exact,
  };
  const transfers = [
    cposBuffer,
    cp4mBuffer,
    cpos.points.buffer,
    cpos.trueCounts.buffer,
    cpos.storedCounts.buffer,
    cp4mTrueCounts.buffer,
    cp4mStoredCounts.buffer,
    expanded.points.buffer,
    expanded.exact.buffer,
  ];
  if (originalPoints) transfers.push(originalPoints.buffer);
  postMessage(message, transfers);
}

async function decodePair({
  jobId,
  mode,
  name,
  originalBytes,
  originalPoints = null,
  cposPayload,
  cp4mPayload,
  displayPoints,
  noise,
}) {
  postProgress(jobId, "decoding CPOS", 0.9);
  const cpos = decodeCpos(cposPayload);
  postProgress(jobId, "decoding CP4M", 0.93);
  const cp4m = await decodeRetainedCp4m(cp4mPayload);
  currentCp4m = cp4m;
  postProgress(jobId, "expanding CP4M preview", 0.97);
  const expanded = expandCp4m(cp4m, {
    maxPoints: displayPoints,
    noise,
  });
  postDecodedResult({
    jobId,
    mode,
    name,
    originalBytes,
    originalPoints,
    cposPayload,
    cpos,
    cp4mPayload,
    cp4m,
    expanded,
  });
}

async function encodePosFile(message) {
  const {
    jobId,
    name,
    posPayload,
    displayPoints,
    noise,
  } = message;
  postProgress(jobId, "encoding CPOS", 0.01);
  const cposPayload = encodePos(posPayload, {
    maxPoints: DEFAULT_MAX_POINTS,
  });
  const cpos = decodeCpos(cposPayload);
  postProgress(jobId, "sampling original POS", 0.08);
  const originalPoints = samplePosBySpectrum(
    posPayload,
    cpos.trueCounts,
    cpos.storedCounts,
  );
  postProgress(jobId, "encoding CP4M", 0.1);
  const cp4mPayload = encodePosCp4m(posPayload, {
    onProgress: ({ stage, fraction }) => {
      postProgress(jobId, stage, 0.1 + fraction * 0.78);
    },
  });
  await decodePair({
    jobId,
    mode: "encoded",
    name,
    originalBytes: posPayload.byteLength,
    originalPoints,
    cposPayload,
    cp4mPayload,
    displayPoints,
    noise,
  });
}

onmessage = async ({ data }) => {
  try {
    if (data.type === "encode") {
      await encodePosFile(data);
      return;
    }
    if (data.type === "decode-demo") {
      await decodePair({
        ...data,
        mode: "demo",
        originalBytes: data.originalBytes,
        originalPoints: null,
      });
      return;
    }
    if (data.type === "expand") {
      if (!currentCp4m) throw new Error("no CP4M file is loaded");
      const expanded = expandCp4m(currentCp4m, {
        maxPoints: data.displayPoints,
        noise: data.noise,
      });
      postMessage({
        type: "expanded",
        requestId: data.requestId,
        points: expanded.points,
        exact: expanded.exact,
      }, [expanded.points.buffer, expanded.exact.buffer]);
      return;
    }
    throw new Error(`unknown worker request ${data.type}`);
  } catch (error) {
    postMessage({
      type: "error",
      jobId: data.jobId,
      requestId: data.requestId,
      ...errorMessage(error),
    });
  }
};
