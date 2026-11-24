// wakeword.js
// Connects the wake word engine to your phone: downloads the models once,
// listens to the microphone, and calls onWake() when someone says the wake word.
//
// Needs these packages (installed by the setup commands):
//   onnxruntime-react-native, react-native-live-audio-stream, expo-file-system

import { PermissionsAndroid, Platform } from "react-native";
import { WakeWordEngine, BlockSplitter, WakeDetector, base64ToInt16 } from "./wakeword-engine";

// Renux uses its own wake word model, trained for "Hey Renux" and hosted in your GitHub.
// (Ready-made models on the openWakeWord page: hey_jarvis_v0.1.onnx, alexa_v0.1.onnx,
// hey_mycroft_v0.1.onnx, hey_marvin_v0.1.onnx, timer_v0.1.onnx, weather_v0.1.onnx)
export const WAKE_MODEL = "hey_renux.onnx";
export const WAKE_PHRASE = "Hey Renux";

const BASE = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/";
// The model you trained. The GitHub repo must be PUBLIC for the app to download it.
const WAKE_URL = "https://raw.githubusercontent.com/RemixMc647/renux-wakeword/main/hey_renux.onnx";
const FILES = ["melspectrogram.onnx", "embedding_model.onnx", WAKE_MODEL];

// Optional native modules: if one is missing, the wake word says so instead of crashing the app.
let ort = null;
let ortError = "";
try { ort = require("onnxruntime-react-native"); } catch (e) { ortError = (e && e.message) || "unknown"; }

let LiveAudioStream = null;
try {
  const m = require("react-native-live-audio-stream");
  LiveAudioStream = m.default || m;
} catch (e) {}

let FS = null;
const usable = (m) => (m && typeof m.downloadAsync === "function" ? m : m && m.default && typeof m.default.downloadAsync === "function" ? m.default : null);
try { FS = usable(require("expo-file-system/legacy")); } catch (e) {}
if (!FS) { try { FS = usable(require("expo-file-system")); } catch (e) {} }

// Download the three model files the first time (about 3.6 MB), then reuse them.
async function ensureModels(onStatus) {
  const dir = FS.documentDirectory + "wakeword/";
  const info = await FS.getInfoAsync(dir);
  if (!info.exists) await FS.makeDirectoryAsync(dir, { intermediates: true });

  const paths = {};
  for (const f of FILES) {
    const p = dir + f;
    const i = await FS.getInfoAsync(p);
    if (!i.exists || (i.size || 0) < (f === WAKE_MODEL ? 20000 : 500000)) {
      onStatus && onStatus("Downloading wake word model: " + f);
      const r = await FS.downloadAsync(f === WAKE_MODEL ? WAKE_URL : BASE + f, p);
      if (r.status !== 200) {
        await FS.deleteAsync(p, { idempotent: true });
        throw new Error("Download failed (" + r.status + ") for " + f);
      }
    }
    paths[f] = p.replace(/^file:\/\//, "");
  }
  return paths;
}

export function createWakeWord({ onWake, onStatus, threshold = 0.5 }) {
  let engine = null;
  let running = false;
  let starting = null;
  let subscribed = false;
  let queue = Promise.resolve();
  let pending = 0;
  let stage = "init";                         // which step we are on, shown if something fails
  const describe = (e) => {
    let msg = "";
    try {
      msg = (e && (e.message || e.code || e.name)) || String(e);
      if (e && e.nativeStackAndroid) msg += " | " + String(e.nativeStackAndroid[0] && e.nativeStackAndroid[0].methodName);
    } catch (x) {}
    return "[" + stage + "] " + (msg || "no details");
  };
  const splitter = new BlockSplitter();
  const detector = new WakeDetector(threshold, 2, 2500);

  async function load() {
    if (engine) return;
    if (!ort) throw new Error("onnxruntime-react-native failed to load: " + (ortError || "not in this build"));
    if (!LiveAudioStream) throw new Error("react-native-live-audio-stream is not in this build");
    if (!FS) throw new Error("expo-file-system is not available");

    stage = "download models";
    const p = await ensureModels(onStatus);
    onStatus && onStatus("Loading wake word...");
    stage = "load mel model";
    const mel = await ort.InferenceSession.create(p["melspectrogram.onnx"]);
    stage = "load embedding model";
    const emb = await ort.InferenceSession.create(p["embedding_model.onnx"]);
    stage = "load hey_renux model";
    const wake = await ort.InferenceSession.create(p[WAKE_MODEL]);
    stage = "create engine";
    engine = new WakeWordEngine(ort, { mel, emb, wake });
  }

  function handleBlock(block) {
    if (pending > 15) return;                 // phone can't keep up: skip instead of lagging behind
    pending++;
    queue = queue
      .then(async () => {
        if (!running) return;
        const score = await engine.processBlock(block);
        if (running && detector.update(score)) onWake(score);
      })
      .catch(() => {})
      .then(() => { pending--; });
  }

  function onData(b64) {
    if (!running) return;
    const blocks = splitter.push(base64ToInt16(b64));
    for (const b of blocks) handleBlock(b);
  }

  async function doStart() {
    try {
      await doStartInner();
    } catch (e) {
      throw new Error(describe(e));
    }
  }

  async function doStartInner() {
    stage = "check native modules";
    await load();

    stage = "mic permission";
    if (Platform.OS === "android") {
      const res = await PermissionsAndroid.request(PermissionsAndroid.PERMISSIONS.RECORD_AUDIO);
      if (res !== PermissionsAndroid.RESULTS.GRANTED) throw new Error("Microphone permission denied");
    }

    stage = "warm up";
    engine.reset();
    await engine.warmUp();                    // about half a second
    splitter.reset();
    detector.reset();

    stage = "mic stream";
    LiveAudioStream.init({ sampleRate: 16000, channels: 1, bitsPerSample: 16, audioSource: 6, bufferSize: 4096 });
    if (!subscribed) {
      LiveAudioStream.on("data", onData);
      subscribed = true;
    }
    running = true;
    LiveAudioStream.start();
    onStatus && onStatus('Listening for "' + WAKE_PHRASE + '"...');
  }

  return {
    // Starts listening. Safe to call many times.
    start() {
      if (running) return Promise.resolve();
      if (!starting) starting = doStart().finally(() => { starting = null; });
      return starting;
    },
    // Stops the microphone (needed before the normal speech recogniser can use it).
    stop() {
      if (!running) return;
      running = false;
      try { LiveAudioStream.stop(); } catch (e) {}
    },
  };
}
