"use strict";
// Hand tracking runs here, in the patient's browser. Only 21 normalised points per
// analysed frame leave this page; the video itself is never sent anywhere.
const preview = document.getElementById("preview");
const toggle = document.getElementById("toggle");
const device = document.getElementById("device");
const status = document.getElementById("status");
const fpsLabel = document.getElementById("fps");

const FLUSH_MS = 400;
const MAX_PENDING = 110;
const MIN_FRAME_MS = 1000 / 30;

let labels = {};
let stream = null;
let detector = null;
let detectorPromise = null;
let running = false;
let busy = false;
let pending = [];
let ack = -1;
let lastVideoTime = -1;
let lastTimestamp = 0;
let lastAnalysis = 0;
let flushTimer = null;
let fpsCount = 0;
let fpsSince = 0;
let devicesRequested = false;

function send(type, extra = {}) {
  parent.postMessage({ isStreamlitMessage: true, type, ...extra }, "*");
}
function value(data) {
  send("streamlit:setComponentValue", { value: data, dataType: "json" });
}
function resize() {
  send("streamlit:setFrameHeight", { height: document.body.scrollHeight + 2 });
}
function message(text, error = false) {
  status.textContent = text || "";
  status.className = error ? "error" : "";
  resize();
}
function asset(path) {
  return new URL(path, window.location.href).href;
}

async function createDetector() {
  const fileset = {
    wasmLoaderPath: asset("vendor/wasm/vision_wasm_internal.js"),
    wasmBinaryPath: asset("vendor/wasm/vision_wasm_internal.wasm"),
  };
  const options = delegate => ({
    baseOptions: { modelAssetPath: asset("hand_landmarker.task"), delegate },
    runningMode: "VIDEO",
    numHands: 1,
    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
  try {
    return await Vision.HandLandmarker.createFromOptions(fileset, options("GPU"));
  } catch (gpuError) {
    return await Vision.HandLandmarker.createFromOptions(fileset, options("CPU"));
  }
}
function loadDetector() {
  if (!detectorPromise) {
    detectorPromise = createDetector().then(instance => (detector = instance));
    detectorPromise.catch(() => { detectorPromise = null; });
  }
  return detectorPromise;
}

function round(number) {
  return Math.round(number * 10000) / 10000;
}
function record(result, timestamp) {
  const hands = result.landmarks || [];
  if (!hands.length) {
    pending.push([timestamp, null, 0, null]);
  } else {
    const category = ((result.handedness || [])[0] || [])[0] || {};
    const coordinates = [];
    hands[0].forEach(point => coordinates.push(round(point.x), round(point.y), round(point.z)));
    const score = Math.min(1, Math.max(0, category.score || 0));
    pending.push([timestamp, category.categoryName || "Right", round(score), coordinates]);
  }
  if (pending.length > MAX_PENDING) pending.splice(0, pending.length - MAX_PENDING);
}
function flush(playing = true) {
  pending = pending.filter(frame => frame[0] > ack);
  if (!pending.length && playing) return;
  value({ playing, frames: pending.slice() });
}
function analyse(now) {
  if (!running) return;
  if (detector && preview.readyState >= 2 && preview.currentTime !== lastVideoTime &&
      now - lastAnalysis >= MIN_FRAME_MS) {
    lastVideoTime = preview.currentTime;
    lastAnalysis = now;
    const timestamp = Math.max(lastTimestamp + 1, Math.round(performance.now()));
    lastTimestamp = timestamp;
    try {
      record(detector.detectForVideo(preview, timestamp), timestamp);
      fpsCount += 1;
      if (now - fpsSince >= 1000) {
        fpsLabel.textContent = (labels.fps || "{fps}").replace("{fps}", String(fpsCount));
        fpsCount = 0;
        fpsSince = now;
      }
    } catch (error) {
      fail(error);
      return;
    }
  }
  schedule();
}
function schedule() {
  if (!running) return;
  if (preview.requestVideoFrameCallback) {
    preview.requestVideoFrameCallback(now => analyse(now));
  } else {
    requestAnimationFrame(analyse);
  }
}

function release() {
  running = false;
  clearInterval(flushTimer);
  flushTimer = null;
  if (stream) {
    stream.getTracks().forEach(track => { track.onended = null; track.stop(); });
    stream = null;
  }
  preview.srcObject = null;
  fpsLabel.textContent = "";
  busy = false;
  toggle.disabled = false;
  toggle.textContent = labels.start || "";
  device.disabled = false;
}
function fail(error) {
  const name = (error && error.name) || "CameraError";
  const wasRunning = running;
  release();
  message(name === "NotAllowedError" ? labels.denied :
    name === "NotFoundError" ? labels.missing : labels.error, true);
  // Only the error category is sent: no device identifiers and no images.
  value({ playing: false, error: name, started: wasRunning, frames: pending.slice() });
}
async function devices() {
  const previous = device.value;
  const cameras = (await navigator.mediaDevices.enumerateDevices())
    .filter(item => item.kind === "videoinput");
  device.replaceChildren();
  const automatic = document.createElement("option");
  automatic.value = "";
  automatic.textContent = labels.device || "Selecionar câmera";
  device.append(automatic);
  cameras.forEach((item, index) => {
    const option = document.createElement("option");
    option.value = item.deviceId;
    option.textContent = item.label || `Câmera ${index + 1}`;
    device.append(option);
  });
  device.value = previous;
  resize();
}
async function start() {
  if (!navigator.mediaDevices || !window.isSecureContext) {
    message(labels.unsupported, true);
    return;
  }
  busy = true;
  toggle.disabled = true;
  device.disabled = true;
  message(labels.loading);
  try {
    const video = {
      width: { ideal: 960, max: 1280 },
      height: { ideal: 540, max: 720 },
      frameRate: { ideal: 30, max: 30 },
    };
    if (device.value) video.deviceId = { exact: device.value };
    const [media] = await Promise.all([
      navigator.mediaDevices.getUserMedia({ video, audio: false }),
      loadDetector(),
    ]);
    stream = media;
    preview.srcObject = stream;
    await preview.play();
    stream.getVideoTracks()[0].onended = stop;
    await devices();
    running = true;
    busy = false;
    lastVideoTime = -1;
    fpsSince = performance.now();
    toggle.disabled = false;
    toggle.textContent = labels.stop;
    message(labels.connected);
    flushTimer = setInterval(() => flush(true), FLUSH_MS);
    schedule();
  } catch (error) {
    fail(error);
  }
}
function stop() {
  release();
  message("");
  flush(false);
}

toggle.onclick = () => { if (running) stop(); else if (!busy) start(); };
window.addEventListener("message", event => {
  if (event.source !== parent || event.data.type !== "streamlit:render") return;
  const args = event.data.args || {};
  labels = args.labels || labels;
  if (typeof args.ack === "number" && args.ack > ack) {
    ack = args.ack;
    pending = pending.filter(frame => frame[0] > ack);
  }
  if (!devicesRequested) {
    devicesRequested = true;
    if (navigator.mediaDevices) devices().catch(() => {});
    // Start fetching the model early, so pressing the button feels instant.
    loadDetector().catch(() => {});
  }
  if (!running && !busy) toggle.textContent = labels.start;
  resize();
});
window.addEventListener("pagehide", () => { if (running) stop(); });
new ResizeObserver(resize).observe(document.body);
send("streamlit:componentReady", { apiVersion: 1 });
resize();
