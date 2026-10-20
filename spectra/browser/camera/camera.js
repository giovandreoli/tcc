"use strict";
const preview = document.getElementById("preview");
const toggle = document.getElementById("toggle");
const device = document.getElementById("device");
const status = document.getElementById("status");
let labels = {};
let stream = null;
let peer = null;
let offerId = null;
let busy = false;
let timeout = null;
let answered = false;
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
  status.textContent = text;
  status.className = error ? "error" : "";
  resize();
}
function reset() {
  clearTimeout(timeout);
  timeout = null;
  if (peer) {
    peer.onconnectionstatechange = null;
    peer.close();
    peer = null;
  }
  if (stream) {
    stream.getTracks().forEach(track => { track.onended = null; track.stop(); });
    stream = null;
  }
  preview.srcObject = null;
  busy = false;
  answered = false;
  toggle.disabled = false;
  toggle.textContent = labels.start || "";
  device.disabled = false;
}
function fail(error) {
  const name = error.name || "CameraError";
  reset();
  message(name === "NotAllowedError" ? labels.denied :
    name === "NotFoundError" ? labels.missing : labels.error, true);
  // Only the error category is sent; no device identifiers or images are logged.
  value({ playing: false, error: name });
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
function gathered(pc) {
  if (pc.iceGatheringState === "complete") return Promise.resolve();
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      pc.removeEventListener("icegatheringstatechange", change);
      reject(new Error("ICETimeout"));
    }, 10000);
    function change() {
      if (pc.iceGatheringState === "complete") {
        clearTimeout(timer);
        pc.removeEventListener("icegatheringstatechange", change);
        resolve();
      }
    }
    pc.addEventListener("icegatheringstatechange", change);
  });
}
async function start() {
  busy = true;
  toggle.disabled = true;
  device.disabled = true;
  offerId = crypto.randomUUID();
  message(labels.connecting);
  try {
    const video = {
      width: { ideal: 960, max: 960 },
      height: { ideal: 540, max: 540 },
      frameRate: { ideal: 30, max: 30 }
    };
    if (device.value) video.deviceId = { exact: device.value };
    stream = await navigator.mediaDevices.getUserMedia({ video, audio: false });
    preview.srcObject = stream;
    await preview.play();
    stream.getVideoTracks()[0].onended = stop;
    await devices();
    peer = new RTCPeerConnection({ iceServers: [] });
    const pc = peer;
    stream.getVideoTracks().forEach(track => {
      pc.addTransceiver(track, { direction: "sendonly", streams: [stream] });
    });
    pc.onconnectionstatechange = () => {
      if (pc !== peer) return;
      if (pc.connectionState === "connected") {
        clearTimeout(timeout);
        busy = false;
        toggle.disabled = false;
        toggle.textContent = labels.stop;
        message(labels.connected);
      } else if (["failed", "disconnected"].includes(pc.connectionState)) {
        fail(new Error("CameraDisconnected"));
      }
    };
    await pc.setLocalDescription(await pc.createOffer());
    await gathered(pc);
    value({ playing: true, id: offerId, offer: {
      type: pc.localDescription.type, sdp: pc.localDescription.sdp
    } });
    timeout = setTimeout(() => fail(new Error("ConnectionTimeout")), 20000);
  } catch (error) {
    fail(error);
  }
}
function stop() {
  reset();
  message("");
  value({ playing: false, id: offerId });
}
toggle.onclick = () => { if (stream) stop(); else if (!busy) start(); };
window.addEventListener("message", async event => {
  if (event.source !== parent || event.data.type !== "streamlit:render") return;
  labels = event.data.args.labels;
  if (!devicesRequested) {
    devicesRequested = true;
    if (navigator.mediaDevices) devices().catch(fail);
  }
  if (!stream && !busy) toggle.textContent = labels.start;
  const answer = event.data.args.answer;
  if (answer && answer.id === offerId && peer && !answered) {
    answered = true;
    try {
      await peer.setRemoteDescription({ type: answer.type, sdp: answer.sdp });
    } catch (error) { fail(error); }
  }
  resize();
});
window.addEventListener("pagehide", reset);
new ResizeObserver(resize).observe(preview);
send("streamlit:componentReady", { apiVersion: 1 });
resize();
