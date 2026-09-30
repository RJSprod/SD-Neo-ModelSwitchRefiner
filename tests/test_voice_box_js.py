"""The Voice Box page, executed rather than read.

``javascript/voice_box.js`` owns everything under one root Python paints, so
what can go wrong in it is only visible with the file running: a stage drawn
out of order, a request without the page token or without a deadline, a
second press that sends a second request, a poll that keeps going at one
second after the render is done, a lane that keeps playing while Voice Chat
speaks. None of that needs a browser; it needs a DOM whose numbers a test can
set and a clock a test can advance, which is what the harness below is,
following ``test_voice_chat_js.py`` and ``test_literals_js.py``.

The fakes with opinions:

    fetch          records every request -- url, method, headers, body and
                   whether an AbortSignal came with it -- and answers from a
                   table, or holds a route open until the scenario lets go;
    the timers     queued rather than run, so "polled every 15 s" is a number
                   a test reads rather than a wait;
    audio elements remember whether they were played and paused, which is how
                   "one lane at a time" and "paused for Voice Chat" are asserted;
    the microphone hands back tracks that record having been stopped.

The second harness loads ``javascript/voice_chat.js`` alone, with the two
hidden holders it reads its tokens from, to prove its half of the audio-focus
event. These run under node, which is not a Forge dependency, so they skip
without it.
"""

from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "javascript" / "voice_box.js"
VOICE_CHAT = ROOT / "javascript" / "voice_chat.js"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")

TOKEN = "PAGE-TOKEN"
PREFIX = "/model-chain/voice-box"

# --------------------------------------------------------------------------- #
# The DOM both harnesses share
# --------------------------------------------------------------------------- #

DOM_JS = r"""
const realSetTimeout = setTimeout;
const realConsole = console;
const linkClicks = [];

function camel(name) {
    return name.replace(/-([a-z])/g, (_, letter) => letter.toUpperCase());
}

class El {
    constructor(tag) {
        this.tagName = String(tag).toUpperCase();
        this.children = [];
        this.parentNode = null;
        this.attributes = {};
        this.dataset = {};
        this.style = {};
        this._classes = new Set();
        this._listeners = {};
        this._text = "";
        this.value = "";
        this.checked = false;
        this.disabled = false;
        this.hidden = false;
        this.id = "";
        this.type = "";
        this.files = [];
        this.clientWidth = 240;
        this.clientHeight = 48;
        if (this.tagName === "AUDIO" || this.tagName === "VIDEO") {
            this.paused = true;
            this.currentTime = 0;
            this.duration = NaN;
            this.loop = false;
            this.plays = 0;
            this.pauses = 0;
            this.src = "";
        }
        if (this.tagName === "CANVAS") {
            this.width = 300;
            this.height = 150;
            this.draws = 0;
        }
    }
    get className() { return Array.from(this._classes).join(" "); }
    set className(value) {
        this._classes = new Set(String(value).split(/\s+/).filter(Boolean));
    }
    get classList() {
        const owner = this;
        return {
            add(...names) { names.forEach((n) => owner._classes.add(n)); },
            remove(...names) { names.forEach((n) => owner._classes.delete(n)); },
            contains(name) { return owner._classes.has(name); },
            toggle(name, force) {
                const on = force === undefined ? !owner._classes.has(name) : !!force;
                if (on) owner._classes.add(name); else owner._classes.delete(name);
                return on;
            },
        };
    }
    get textContent() {
        return this._text + this.children.map((child) => child.textContent).join("");
    }
    set textContent(value) {
        this.children.forEach((child) => { child.parentNode = null; });
        this.children = [];
        this._text = String(value);
    }
    get innerHTML() { return this.textContent; }
    set innerHTML(value) { this.textContent = String(value); }
    get firstChild() { return this.children[0] || null; }
    get lastChild() { return this.children[this.children.length - 1] || null; }
    get parentElement() { return this.parentNode; }
    get nextSibling() {
        if (!this.parentNode) return null;
        const at = this.parentNode.children.indexOf(this);
        return this.parentNode.children[at + 1] || null;
    }
    get options() { return this.children.filter((child) => child.tagName === "OPTION"); }
    appendChild(child) {
        if (child.parentNode) child.parentNode.removeChild(child);
        child.parentNode = this;
        this.children.push(child);
        return child;
    }
    removeChild(child) {
        this.children = this.children.filter((entry) => entry !== child);
        child.parentNode = null;
        return child;
    }
    insertBefore(child, before) {
        if (child.parentNode) child.parentNode.removeChild(child);
        child.parentNode = this;
        const at = before ? this.children.indexOf(before) : -1;
        if (at < 0) this.children.push(child);
        else this.children.splice(at, 0, child);
        return child;
    }
    remove() { if (this.parentNode) this.parentNode.removeChild(this); }
    setAttribute(name, value) {
        this.attributes[name] = String(value);
        if (name === "id") this.id = String(value);
        if (name === "class") this.className = value;
        if (name.indexOf("data-") === 0) this.dataset[camel(name.slice(5))] = String(value);
        if (name === "hidden") this.hidden = true;
        if (name === "type") this.type = String(value);
    }
    getAttribute(name) {
        return Object.prototype.hasOwnProperty.call(this.attributes, name)
            ? this.attributes[name] : null;
    }
    removeAttribute(name) {
        delete this.attributes[name];
        if (name === "hidden") this.hidden = false;
    }
    hasAttribute(name) { return Object.prototype.hasOwnProperty.call(this.attributes, name); }
    addEventListener(type, fn) {
        (this._listeners[type] = this._listeners[type] || []).push(fn);
    }
    removeEventListener(type, fn) {
        this._listeners[type] = (this._listeners[type] || []).filter((f) => f !== fn);
    }
    dispatchEvent(event) {
        try { if (!event.target) event.target = this; } catch (error) { /* a native event */ }
        (this._listeners[event.type] || []).slice().forEach((fn) => fn.call(this, event));
        if (event.bubbles && this.parentNode) this.parentNode.dispatchEvent(event);
        return true;
    }
    click() {
        if (this.disabled) return;
        if (this.tagName === "A") linkClicks.push({href: this.href || "", download: this.download || ""});
        this.dispatchEvent({type: "click", preventDefault() {}});
    }
    focus() { globalThis.document.activeElement = this; }
    blur() { if (globalThis.document.activeElement === this) globalThis.document.activeElement = null; }
    select() {}
    scrollIntoView() {}
    setPointerCapture() {}
    getBoundingClientRect() {
        return {left: 0, top: 0, width: this.clientWidth, height: this.clientHeight};
    }
    getContext() {
        if (this.tagName !== "CANVAS") return null;
        const canvas = this;
        return new Proxy({}, {
            get(target, name) {
                if (name === "canvas") return canvas;
                return function () { canvas.draws += 1; };
            },
            set() { return true; },
        });
    }
    play() {
        this.paused = false;
        this.plays += 1;
        this.dispatchEvent({type: "play"});
        return Promise.resolve();
    }
    pause() {
        if (this.paused) return;
        this.paused = true;
        this.pauses += 1;
        this.dispatchEvent({type: "pause"});
    }
    load() {}
    walk() {
        return this.children.reduce((found, child) => found.concat([child], child.walk()), []);
    }
    closest(selector) {
        let at = this;
        while (at) {
            if (at instanceof El && matches(at, selector)) return at;
            at = at.parentNode;
        }
        return null;
    }
    contains(other) {
        let at = other;
        while (at) {
            if (at === this) return true;
            at = at.parentNode;
        }
        return false;
    }
    querySelector(selector) { return this.walk().filter((el) => matches(el, selector))[0] || null; }
    querySelectorAll(selector) { return this.walk().filter((el) => matches(el, selector)); }
}

function simple(element, token) {
    let rest = token;
    const tag = /^[a-zA-Z][\w-]*/.exec(rest);
    if (tag) {
        if (element.tagName !== tag[0].toUpperCase()) return false;
        rest = rest.slice(tag[0].length);
    }
    const parts = /([#.])([\w-]+)|\[([\w-]+)(?:([$^*]?=)"?([^"\]]*)"?)?\]/g;
    let found;
    while ((found = parts.exec(rest))) {
        if (found[1] === "#") {
            if (element.id !== found[2]) return false;
        } else if (found[1] === ".") {
            if (!element._classes.has(found[2])) return false;
        } else {
            const value = element.getAttribute(found[3]);
            if (found[4] === undefined) { if (value === null) return false; }
            else if (found[4] === "=") { if (value !== found[5]) return false; }
            else if (found[4] === "$=") { if (value === null || !value.endsWith(found[5])) return false; }
            else if (found[4] === "^=") { if (value === null || value.indexOf(found[5]) !== 0) return false; }
            else if (found[4] === "*=") { if (value === null || value.indexOf(found[5]) === -1) return false; }
        }
    }
    return true;
}

function matches(element, selector) {
    return String(selector).split(",").map((part) => part.trim()).filter(Boolean).some((part) => {
        const tokens = part.split(/\s+/);
        if (!simple(element, tokens[tokens.length - 1])) return false;
        let at = element.parentNode;
        for (let index = tokens.length - 2; index >= 0; index -= 1) {
            while (at && !(at instanceof El && simple(at, tokens[index]))) at = at.parentNode;
            if (!at) return false;
            at = at.parentNode;
        }
        return true;
    });
}

const document = new El("#document");
document.body = document.appendChild(new El("body"));
document.documentElement = document;
document.readyState = "complete";
document.visibilityState = "visible";
document.hidden = false;
document.activeElement = null;
document.createElement = (tag) => new El(tag);
document.createTextNode = (text) => { const node = new El("#text"); node._text = String(text); return node; };
document.getElementById = (id) => document.querySelector("#" + id);
globalThis.document = document;
globalThis.window = globalThis;
globalThis.gradioApp = () => document;
globalThis.gradio_config = {root: ""};
globalThis.location = {pathname: "/", origin: "https://forge.example"};
globalThis.getComputedStyle = () => ({color: "rgb(10, 20, 30)", getPropertyValue: () => ""});
globalThis.devicePixelRatio = 1;
globalThis.isSecureContext = true;
const rafs = [];
globalThis.requestAnimationFrame = (fn) => { rafs.push(fn); return rafs.length; };
globalThis.cancelAnimationFrame = () => {};
globalThis.MutationObserver = class { observe() {} disconnect() {} };

// The clock: held still, timers queued rather than run.
let NOW = 1000;
const timers = [];
globalThis.setTimeout = (fn, ms) => {
    timers.push({fn, ms: ms || 0, at: NOW + (ms || 0), done: false, cancelled: false});
    return timers.length;
};
globalThis.clearTimeout = (handle) => { if (timers[handle - 1]) timers[handle - 1].cancelled = true; };
globalThis.setInterval = () => 0;
globalThis.clearInterval = () => {};
globalThis.performance = {now() { return NOW; }};

function advance(ms) {
    NOW += ms;
    const due = timers.filter((t) => !t.done && !t.cancelled && t.at <= NOW)
        .sort((a, b) => a.at - b.at);
    due.forEach((t) => { t.done = true; t.fn(); });
    return due.length;
}
function pendingDelays() {
    return timers.filter((t) => !t.done && !t.cancelled).map((t) => t.ms);
}
async function flush(times) {
    for (let index = 0; index < (times || 6); index += 1) {
        await new Promise((resolve) => realSetTimeout(resolve, 0));
    }
}

const warnings = [];
globalThis.console = Object.assign({}, realConsole, {
    warn(...args) { warnings.push(args.map(String).join(" ")); },
    error(...args) { warnings.push("ERROR " + args.map(String).join(" ")); },
});

// Every audio-focus event on the document, whoever sent it.
const focusEvents = [];
document.addEventListener("mc:audio-focus", (event) => {
    focusEvents.push(Object.assign({}, event.detail));
});

// The microphone: tracks that record having been stopped.
const tracks = [];
const microphoneRequests = [];
function track() {
    const item = {stopped: false, stop() { item.stopped = true; }, label: "Mic",
                  getSettings: () => ({sampleRate: 48000, channelCount: 1, deviceId: "d1"})};
    tracks.push(item);
    return item;
}
Object.defineProperty(globalThis, "navigator", {
    configurable: true,
    writable: true,
    value: MICROPHONE ? {mediaDevices: {getUserMedia(constraints) {
        microphoneRequests.push(constraints);
        const only = [track()];
        return Promise.resolve({getTracks: () => only, getAudioTracks: () => only});
    }}} : {},
});
"""

# --------------------------------------------------------------------------- #
# The Voice Box page's harness
# --------------------------------------------------------------------------- #

HARNESS = DOM_JS + r"""
const MICROPHONE_FLAG = MICROPHONE;

// Recording: a MediaRecorder that hands its one chunk over on stop.
const recorders = [];
globalThis.MediaRecorder = class {
    constructor(stream) {
        this.stream = stream;
        this.state = "inactive";
        this.mimeType = "audio/webm";
        this.ondataavailable = null;
        this.onstop = null;
        recorders.push(this);
    }
    start() { this.state = "recording"; }
    stop() {
        this.state = "inactive";
        if (this.ondataavailable) this.ondataavailable({data: new Blob([new Uint8Array(100)])});
        if (this.onstop) this.onstop();
    }
};

// Web Audio: a decoder that answers with DECODE_SECONDS of a 48 kHz tone, a
// media-element source, a script processor the scenario can feed, and no
// AudioWorklet (so the capture path takes the processor).
const contexts = [];
globalThis.lastProcessor = null;
globalThis.AudioContext = class {
    constructor() {
        this.state = "running";
        this.sampleRate = 48000;
        this.destination = {};
        this.currentTime = 0;
        contexts.push(this);
    }
    resume() { return Promise.resolve(); }
    decodeAudioData(buffer) {
        if (!DECODE_WORKS) return Promise.reject(new Error("EncodingError"));
        const rate = 48000;
        const length = Math.round(DECODE_SECONDS * rate);
        const data = new Float32Array(length);
        for (let index = 0; index < length; index += 1) data[index] = 0.5 * Math.sin(index / 7);
        return Promise.resolve({numberOfChannels: 1, length, sampleRate: rate,
                                duration: length / rate, getChannelData: () => data});
    }
    createMediaElementSource() { return {connect() {}, disconnect() {}}; }
    createScriptProcessor() {
        const node = {connect() {}, disconnect() {}, onaudioprocess: null};
        globalThis.lastProcessor = node;
        return node;
    }
    createGain() { return {gain: {value: 1}, connect() {}, disconnect() {}}; }
    createBufferSource() { return {buffer: null, connect() {}, start() {}, stop() {}}; }
    createBuffer(channels, length, rate) {
        const data = new Float32Array(length);
        return {numberOfChannels: channels, length, sampleRate: rate,
                duration: length / rate, getChannelData: () => data};
    }
};
if (OFFLINE) {
    // A resampler whose output is unmistakable: every sample a quarter.
    globalThis.OfflineAudioContext = class {
        constructor(channels, length, rate) { this.length = length; this.sampleRate = rate; this.destination = {}; }
        createBuffer(channels, length, rate) {
            const data = new Float32Array(length);
            return {numberOfChannels: channels, length, sampleRate: rate, getChannelData: () => data};
        }
        createBufferSource() { return {buffer: null, connect() {}, start() {}}; }
        startRendering() {
            const data = new Float32Array(this.length).fill(0.25);
            return Promise.resolve({getChannelData: () => data});
        }
    };
}

// Object URLs are counted, never dereferenced.
const objectUrls = [];
URL.createObjectURL = (blob) => { objectUrls.push(blob && blob.size); return "blob:test-" + objectUrls.length; };
URL.revokeObjectURL = () => {};

// localStorage: a store, or one that refuses.
const storage = {};
const storageLog = [];
Object.defineProperty(globalThis, "localStorage", {
    configurable: true,
    writable: true,
    value: {
        getItem: (key) => (Object.prototype.hasOwnProperty.call(storage, key) ? storage[key] : null),
        setItem: (key, value) => {
            storageLog.push([key, String(value)]);
            if (STORAGE_THROWS) throw new Error("QuotaExceededError");
            storage[key] = String(value);
        },
        removeItem: (key) => { delete storage[key]; },
    },
});

// fetch: every request recorded; answers from the table; held routes wait.
const requests = [];
const answers = ANSWERS_TABLE;
globalThis.answers = answers;
const held = new Set(HELD_ROUTES);
const holds = [];
function routeOf(address) {
    const path = String(address).split("?")[0];
    const at = path.indexOf(PREFIX_TEXT);
    return at === -1 ? path : path.slice(at + PREFIX_TEXT.length);
}
function response(spec) {
    const status = spec.status || 200;
    const json = spec.json === undefined ? {ok: true} : spec.json;
    const body = JSON.stringify(json);
    const bytes = new Uint8Array(spec.bytes || 44);
    return {
        ok: status >= 200 && status < 300,
        status,
        headers: {get: () => null},
        json: () => Promise.resolve(JSON.parse(body)),
        text: () => Promise.resolve(spec.text !== undefined ? spec.text : body),
        blob: () => Promise.resolve(new Blob([bytes], {type: "audio/wav"})),
        arrayBuffer: () => Promise.resolve(bytes.buffer),
    };
}
function nextSpec(route) {
    let spec = answers[route] || {json: {ok: true}};
    if (spec.sequence) {
        spec = spec.sequence.length > 1 ? spec.sequence.shift() : spec.sequence[0];
    }
    return spec;
}
globalThis.fetch = function (address, init) {
    const options = init || {};
    const headers = {};
    Object.keys(options.headers || {}).forEach((name) => { headers[name.toLowerCase()] = options.headers[name]; });
    let body = null;
    if (typeof options.body === "string") {
        try { body = JSON.parse(options.body); } catch (error) { body = options.body; }
    } else if (options.body) {
        body = {bytes: options.body.byteLength || options.body.size || 0, raw: options.body};
    }
    const route = routeOf(address);
    const record = {url: String(address), route, method: options.method || "GET", headers, body,
                    signal: !!(options.signal && typeof options.signal.addEventListener === "function"),
                    aborted: false};
    requests.push(record);
    const spec = nextSpec(route);
    return new Promise((resolve, reject) => {
        const fail = () => {
            record.aborted = true;
            const error = new Error("The operation was aborted.");
            error.name = "AbortError";
            reject(error);
        };
        if (options.signal && options.signal.addEventListener) {
            options.signal.addEventListener("abort", fail);
        }
        if (held.has(route)) { holds.push({route, resolve: () => resolve(response(spec))}); return; }
        if (spec.never) return;
        resolve(response(spec));
    });
};
function release(route) {
    const due = holds.filter((hold) => !route || hold.route === route);
    due.forEach((hold) => hold.resolve());
    holds.splice(0, holds.length, ...holds.filter((hold) => due.indexOf(hold) === -1));
}
function requestsTo(route) {
    return requests.filter((request) => request.route === route);
}
function plain(request) {
    return {url: request.url, route: request.route, method: request.method, headers: request.headers,
            body: request.body && request.body.raw ? {bytes: request.body.bytes} : request.body,
            signal: request.signal, aborted: request.aborted};
}
function wavInfo(raw) {
    const buffer = raw.buffer ? raw.buffer : raw;
    const view = new DataView(buffer);
    const ascii = (at, count) => String.fromCharCode(...new Uint8Array(buffer, at, count));
    const samples = [];
    for (let index = 0; index < 8 && 44 + index * 2 + 1 < view.byteLength; index += 1) {
        samples.push(view.getInt16(44 + index * 2, true));
    }
    return {riff: ascii(0, 4), riffSize: view.getUint32(4, true), wave: ascii(8, 4), fmt: ascii(12, 4),
            fmtSize: view.getUint32(16, true), format: view.getUint16(20, true),
            channels: view.getUint16(22, true), rate: view.getUint32(24, true),
            byteRate: view.getUint32(28, true), blockAlign: view.getUint16(32, true),
            bits: view.getUint16(34, true), data: ascii(36, 4), dataLength: view.getUint32(40, true),
            total: view.byteLength, samples};
}

// The root Python paints, unless the scenario is about a page without one.
if (ROOT_PRESENT) {
    const root = new El("div");
    root.setAttribute("id", "mc-voice-box");
    root.className = "mc-voice-box";
    root.setAttribute("data-mc-voice-key", "PAGE-TOKEN");
    root.setAttribute("data-mc-voice-box-prefix", PREFIX_TEXT);
    root.setAttribute("data-mc-voice-box-boot", "1");
    document.body.appendChild(root);
}

const loaded = [];
const afterUpdate = [];
globalThis.onUiLoaded = (fn) => loaded.push(fn);
globalThis.onAfterUiUpdate = (fn) => afterUpdate.push(fn);

@@SOURCE@@

loaded.forEach((fn) => fn());

function find(selector) { return document.querySelector(selector); }
function all(selector) { return document.querySelectorAll(selector); }
function texts(selector) { return all(selector).map((node) => node.textContent); }
function buttonNamed(label, within) {
    return (within || document).querySelectorAll("button").filter((b) => b.textContent === label)[0] || null;
}
function press(label, within) {
    const found = buttonNamed(label, within);
    if (!found) throw new Error("no button " + label);
    found.click();
    return found;
}
function type(node, value) {
    node.value = value;
    node.dispatchEvent({type: "input"});
}
function choose(selector, value) {
    const node = find(selector);
    if (!node) throw new Error("no select " + selector);
    node.value = value;
    node.dispatchEvent({type: "change"});
    return node;
}
function shown(selector) {
    const node = find(selector);
    return !!node && !node.hidden;
}
function focusFrom(owner, kind) {
    document.dispatchEvent(new CustomEvent("mc:audio-focus", {detail: {owner, kind}}));
}
function report(extra) {
    realConsole.log(JSON.stringify(Object.assign({
        requests: requests.map(plain),
        focusEvents,
        warnings,
        timers: pendingDelays(),
        linkClicks,
        storageLog,
        state: globalThis.mcVoiceBox ? globalThis.mcVoiceBox.state() : null,
        status: find(".mc-voice-box-status") ? find(".mc-voice-box-status").textContent : null,
    }, extra || {})));
}

await (async function () {
@@SCENARIO@@
})();
"""

# --------------------------------------------------------------------------- #
# The answers a fresh page gets
# --------------------------------------------------------------------------- #

SAMPLE = {"id": "s1", "title": "Ada", "seconds": 8.0, "rate": 24000, "peaks": [0.2] * 240,
          "source": "file", "created": 1}
CONFIGURATION = {"id": "c1", "name": "Default", "model_id": "vibevoice-7b", "card_uuid": "GPU-a",
                 "steps": 10, "cfg_scale": 1.3, "seed": None, "max_new_tokens": None,
                 "speakers": {"1": "s1"}, "created": 1, "updated": 1}
PIPELINE = {"id": "p1", "name": "Pipeline 1", "prompt": "Speaker 1: Hello there.",
            "configuration_id": "c1", "outputs": ["o1"], "created": 1, "updated": 1}
OUTPUT = {"id": "o1", "name": "Take 1", "pipeline_id": "p1", "seconds": 4.5, "rate": 24000,
          "peaks": [0.3] * 240, "loop": False, "created": 2,
          "render": {"model_id": "vibevoice-7b", "card": "GPU-a", "seed": 42, "steps": 10,
                     "cfg_scale": 1.3, "speakers": [{"n": 1, "sample_id": "s1", "title": "Ada"}],
                     "prompt": "Speaker 1: Hello there.", "render_seconds": 3.0,
                     "peak_bytes": 1, "sections": 1}}
JOB = {"id": "j1", "name": "Pipeline 1 2", "pipeline_id": "p1", "phase": "queued", "reason": "",
       "warning": "", "progress": {}, "output_id": "", "created": 3, "started": None,
       "ended": None, "card": "GPU-a", "live": True}
STATUS = {
    "ok": True,
    "engine": {"installed": True, "ready": True, "supported": True, "message": "Installed",
               "label": "VibeVoice", "model_id": "vibevoice-7b", "model_label": "VibeVoice 7B",
               "models": [{"id": "vibevoice-7b", "label": "VibeVoice 7B"}],
               "download_bytes": 0, "parts": []},
    "progress": {"running": False, "text": "", "fraction": 0.0, "failed": False, "model": ""},
    "engine_settings": {"card_uuid": "GPU-a", "model_id": "vibevoice-7b", "steps": 10,
                        "cfg_scale": 1.3, "seed": None, "max_new_tokens": None, "keep_warm": True},
    "settings": {"save_folder": "", "card_uuid": "GPU-a", "model_id": "vibevoice-7b",
                 "keep_warm": True},
    "cards": [{"uuid": "GPU-a", "key": "a", "index": 0, "name": "RTX 3090", "image_card": True,
               "wangp_card": False},
              {"uuid": "GPU-b", "key": "b", "index": 1, "name": "RTX 5090", "image_card": False,
               "wangp_card": True}],
    "runtime": {"cards": {}, "last_error": ""},
    "turns": [],
    "jobs": [],
}
ANSWERS = {
    "/status": {"json": STATUS},
    "/samples": {"json": {"ok": True, "samples": [SAMPLE]}},
    "/prompts": {"json": {"ok": True,
                          "history": [{"id": "h1", "text": "Speaker 1: Hello there.",
                                       "favourite": False, "used": 2}],
                          "favourites": []}},
    "/configurations": {"json": {"ok": True, "configurations": [CONFIGURATION]}},
    "/pipelines": {"json": {"ok": True, "pipelines": [PIPELINE]}},
    "/outputs": {"json": {"ok": True, "outputs": [OUTPUT]}},
    "/render": {"json": {"ok": True, "job": JOB}},
    "/outputs/audio": {"bytes": 1044},
    "/samples/audio": {"bytes": 1044},
    "/samples/upload": {"json": {"ok": True, "sample": dict(SAMPLE, id="s2", title="song")}},
    "/outputs/save": {"json": {"ok": True, "path": "D:/renders/Take 1.wav"}},
    "/settings/folder": {"json": {"ok": True, "save_folder": "D:/renders", "chosen": True}},
    "/settings": {"json": {"ok": True, "settings": STATUS["settings"],
                           "engine_settings": STATUS["engine_settings"]}},
}

BOOT_ROUTES = {"/status", "/samples", "/prompts", "/configurations", "/pipelines", "/outputs"}


def status_with(**changes) -> dict:
    found = json.loads(json.dumps(STATUS))
    found.update(changes)
    return found


# The engine as it describes its models (mc_voice_vibevoice.models_info) and its
# LoRA library (loras()). The presets arrive in the manifest's order, German
# before English, which is what makes the grouping's order worth asserting.
REALTIME = "vibevoice-realtime-0.5b"
PRESETS = [
    {"id": "de-Spk0_man", "name": "Spk0", "language": "de", "language_label": "German",
     "gender": "man", "experimental": True, "installed": True},
    {"id": "en-Carter_man", "name": "Carter", "language": "en", "language_label": "English",
     "gender": "man", "experimental": False, "installed": True},
    {"id": "en-Emma_woman", "name": "Emma", "language": "en", "language_label": "English",
     "gender": "woman", "experimental": False, "installed": True},
    {"id": "jp-Spk1_woman", "name": "Spk1", "language": "jp", "language_label": "Japanese",
     "gender": "woman", "experimental": True, "installed": False},
]
MODEL_7B = {"id": "vibevoice-7b", "label": "VibeVoice 7B", "kind": "longform", "installed": True,
            "runtime_installed": True, "precisions": ["bf16", "int8", "nf4"], "lora": True,
            "max_speakers": 4, "voices": "samples", "defaults": {"steps": 10, "cfg_scale": 1.3},
            "need_vram_bytes": {"bf16": 20_000_000_000, "int8": 13_000_000_000,
                                "nf4": 9_000_000_000},
            "need_ram_bytes": 3_000_000_000, "presets": [], "message": "", "download_bytes": 0}
MODEL_REALTIME = {"id": REALTIME, "label": "VibeVoice Realtime 0.5B", "kind": "realtime",
                  "installed": True, "runtime_installed": True, "precisions": ["bf16"],
                  "lora": False, "max_speakers": 1, "voices": "presets",
                  "defaults": {"steps": 5, "cfg_scale": 1.5},
                  "need_vram_bytes": {"bf16": 3_000_000_000}, "need_ram_bytes": 2_000_000_000,
                  "presets": PRESETS, "message": "", "download_bytes": 0}
LORA = {"id": "0123456789abcdef", "name": "Narrator", "base": "vibevoice-7b", "bytes": 83_886_080,
        "parts": ["llm", "diffusion_head"], "created": 1.0}
WARM = {"id": "fedcba9876543210", "name": "Warm", "base": "vibevoice-7b", "bytes": 2_500_000_000,
        "parts": ["llm"], "created": 2.0}


def described(*models, loras=None, settings=None, **engine) -> dict:
    """The answers of a page whose engine describes its models and its LoRAs."""
    found = status_with(loras=[LORA] if loras is None else loras)
    found["engine"].update({
        "models": list(models or (MODEL_7B, MODEL_REALTIME)), "runtime_installed": True,
        "parts": [{"id": "runtime", "installed": True, "bytes": 4_300_000_000, "message": ""},
                  {"id": "model", "installed": True, "bytes": 0, "message": ""}]})
    found["engine"].update(engine)
    if settings:
        found["settings"].update(settings)
        found["engine_settings"].update(settings)
    return {"/status": {"json": found}}


def _node(harness: str) -> dict:
    with tempfile.TemporaryDirectory() as room:
        entry = pathlib.Path(room) / "scenario.mjs"
        entry.write_text(harness, encoding="utf-8")
        result = subprocess.run(["node", str(entry)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    lines = [line for line in result.stdout.splitlines() if line.startswith("{")]
    assert lines, result.stdout or "the harness reported nothing"
    return json.loads(lines[-1])


def run(scenario: str, *, answers: dict | None = None, held=(), root: bool = True,
        decode_seconds: float = 10, decode_works: bool = True, offline: bool = False,
        storage_throws: bool = False, microphone: bool = True) -> dict:
    """One scenario, in a real JavaScript engine, against the real page script.

    Written to a file rather than passed as an argument: the page script alone
    is past the size at which a single argument fails on Linux.
    """
    table = json.loads(json.dumps(ANSWERS))
    table.update(answers or {})
    harness = (HARNESS
               .replace("@@SOURCE@@", SCRIPT.read_text(encoding="utf-8"))
               .replace("@@SCENARIO@@", scenario)
               .replace("ANSWERS_TABLE", json.dumps(table))
               .replace("HELD_ROUTES", json.dumps(list(held)))
               .replace("PREFIX_TEXT", json.dumps(PREFIX))
               .replace("ROOT_PRESENT", "true" if root else "false")
               .replace("DECODE_SECONDS", json.dumps(decode_seconds))
               .replace("DECODE_WORKS", "true" if decode_works else "false")
               .replace("OFFLINE", "true" if offline else "false")
               .replace("STORAGE_THROWS", "true" if storage_throws else "false")
               .replace("MICROPHONE", "true" if microphone else "false"))
    return _node(harness)


def test_the_files_under_test_avoid_the_placeholders():
    """The harness substitutes bare words; a script that contained one would
    be rewritten mid-token into a syntax error twenty lines from anything."""
    for path in (SCRIPT, VOICE_CHAT):
        source = path.read_text(encoding="utf-8")
        for word in ("@@SOURCE@@", "@@SCENARIO@@", "ANSWERS_TABLE", "HELD_ROUTES", "PREFIX_TEXT",
                     "ROOT_PRESENT", "DECODE_SECONDS", "DECODE_WORKS", "STORAGE_THROWS",
                     "MICROPHONE_FLAG"):
            assert word not in source, (path.name, word)


# --------------------------------------------------------------------------- #
# Boot
# --------------------------------------------------------------------------- #


class TestBoot:
    def test_it_draws_the_pipeline_bar_the_four_stages_in_order_and_the_footer(self):
        found = run("""
            await flush();
            const stages = find(".mc-voice-box-stages");
            report({
                title: find(".mc-voice-box-title").textContent,
                bar: texts(".mc-voice-box-pipeline-bar button"),
                pipelines: find(".mc-voice-box-pipeline-select").options.map((o) => o.textContent),
                stages: texts(".mc-voice-box-stage-title"),
                row: stages.children.map((child) => child.className.split(" ")[0]),
                render: !!find(".mc-voice-box-footer .mc-voice-box-render"),
                cards: find(".mc-voice-box-cards").textContent,
                help: find(".mc-voice-box-prompt-help").textContent,
            });
        """)

        assert found["stages"] == ["INPUT", "PROMPT", "CONFIGURATION", "OUTPUTS"]
        assert found["row"] == ["mc-voice-box-stage", "mc-voice-box-connector",
                                "mc-voice-box-stage", "mc-voice-box-connector",
                                "mc-voice-box-stage", "mc-voice-box-connector",
                                "mc-voice-box-stage"]
        assert found["bar"] == ["New", "Rename", "Delete"]
        assert found["pipelines"] == ["Pipeline 1"]
        assert found["title"] == "Voice Box · Pipeline 1"
        assert found["render"] is True
        assert found["cards"] == "No card is busy."
        assert "[pause:1500]" in found["help"]
        assert found["warnings"] == []

    def test_it_fetches_everything_once_on_boot(self):
        found = run("await flush(); report();")

        routes = [request["route"] for request in found["requests"]]
        assert set(routes) == BOOT_ROUTES
        assert len(routes) == len(BOOT_ROUTES), routes
        outputs = [r for r in found["requests"] if r["route"] == "/outputs"][0]
        assert outputs["body"] == {"pipeline_id": "p1"}
        assert found["state"]["pipelineId"] == "p1"
        assert found["state"]["configurationId"] == "c1"
        assert found["state"]["prompt"] == "Speaker 1: Hello there."

    def test_without_a_root_the_script_does_nothing(self):
        """Forge loads every extension script on every page."""
        found = run("await flush(); report({children: document.body.children.length});",
                    root=False)

        assert found["requests"] == []
        assert found["state"]["booted"] is False
        assert found["children"] == 0
        assert found["warnings"] == []

    def test_the_page_shows_what_it_was_told(self):
        found = run("""
            await flush();
            const lane = find(".mc-voice-box-lane");
            report({
                samples: texts(".mc-voice-box-sample-title"),
                sampleSpeakers: all(".mc-voice-box-sample-speaker").map((b) => b.getAttribute("aria-pressed")),
                history: texts(".mc-voice-box-history-list .mc-voice-box-prompt-text"),
                lanes: texts(".mc-voice-box-lane-name"),
                meta: find(".mc-voice-box-lane-meta").textContent,
                speakerOne: find('.mc-voice-box-speaker[data-speaker="1"] .mc-voice-box-speaker-title').textContent,
                speakerTwo: find('.mc-voice-box-speaker[data-speaker="2"] .mc-voice-box-speaker-title').textContent,
                cardOptions: find('[data-field="card_uuid"]').options.map((o) => o.textContent),
                configurations: find(".mc-voice-box-configuration-select").options.map((o) => o.textContent),
                keepWarm: find(".mc-voice-box-keep-warm input").checked,
                summary: find(".mc-voice-box-prompt-summary").textContent,
                renderDisabled: find(".mc-voice-box-render").disabled,
            });
        """)

        assert found["samples"] == ["Ada"]
        assert found["sampleSpeakers"] == ["true", "false", "false", "false"]
        assert found["history"] == ["Speaker 1: Hello there."]
        assert found["lanes"] == ["Take 1"]
        assert found["meta"] == "vibevoice-7b · seed 42 · 10 steps · CFG 1.3 · S1 Ada · 4.5 s · GPU-a"
        assert found["speakerOne"] == "Ada"
        assert found["speakerTwo"] == "no sample"
        assert found["cardOptions"] == ["RTX 3090 — image model's card", "RTX 5090 — WanGP's card"]
        assert found["configurations"] == ["(unsaved)", "Default"]
        assert found["keepWarm"] is True
        assert found["summary"] == "1 speaker, 0 pauses, ~2 words"
        assert found["renderDisabled"] is False

    def test_the_last_open_pipeline_is_remembered_and_storage_may_refuse(self):
        remembered = run("await flush(); report();")
        refused = run("await flush(); report();", storage_throws=True)

        assert ["mc-voice-box:pipeline", "p1"] in remembered["storageLog"]
        assert refused["state"]["pipelineId"] == "p1"
        assert refused["warnings"] == []
        assert refused["state"]["booted"] is True

    def test_a_remembered_pipeline_is_opened_over_the_first(self):
        """The remembered id is read once the lists have arrived, so storage
        written before they do is what the page opens -- and a second boot,
        which Forge's after-update hook will deliver, changes nothing."""
        found = run("""
            storage["mc-voice-box:pipeline"] = "p2";
            await flush();
            const booted = globalThis.mcVoiceBox.boot();
            await flush();
            report({booted, title: find(".mc-voice-box-title").textContent,
                    prompt: find(".mc-voice-box-prompt").value});
        """, answers={"/pipelines": {"json": {"ok": True, "pipelines": [
            PIPELINE, dict(PIPELINE, id="p2", name="Second", prompt="Speaker 1: Two.")]}}})

        assert found["state"]["pipelineId"] == "p2"
        assert found["title"] == "Voice Box · Second"
        assert found["prompt"] == "Speaker 1: Two."
        assert found["booted"] is True
        assert len([r for r in found["requests"] if r["route"] == "/pipelines"]) == 1
        assert len([r for r in found["requests"] if r["route"] == "/outputs"]) == 1


# --------------------------------------------------------------------------- #
# The prompt
# --------------------------------------------------------------------------- #


class TestThePromptSummary:
    def test_it_counts_speakers_pauses_and_words_from_the_box(self):
        found = run("""
            await flush();
            type(find(".mc-voice-box-prompt"),
                 "Speaker 1: Hello there friend.\\n[pause]\\nSpeaker 2: Hi.\\n[pause:1500]\\nSpeaker 1: Bye now.");
            report({summary: find(".mc-voice-box-prompt-summary").textContent});
        """)

        assert found["summary"] == "2 speakers, 2 pauses, ~6 words"

    def test_it_counts_the_way_the_server_counts(self):
        """mc_voice_box.parse_script's grammar: a tag at either end adds no
        pause, an unnamed line continues the speaker before it, and a speaker
        outside 1..4 is an error the summary shows."""
        found = run("""
            const s = globalThis.mcVoiceBox.summarize;
            report({
                empty: s(""),
                edges: s("[pause]\\nHello\\n[pause]"),
                continued: s("Speaker 2: a\\nb c"),
                bracket: s("[3]: one two\\n[pause=900]\\n[4]: three"),
                tooMany: s("Speaker 7: no"),
                onlyTags: s("[pause] [pause:200]"),
            });
        """)

        assert found["empty"] == "Nothing to render yet."
        assert found["edges"] == "1 speaker, 0 pauses, ~1 words"
        assert found["continued"] == "1 speaker, 0 pauses, ~3 words"
        assert found["bracket"] == "2 speakers, 1 pause, ~3 words"
        assert found["tooMany"] == "Speakers are numbered 1 to 4; the prompt names Speaker 7."
        assert found["onlyTags"] == "Nothing to render yet."

    def test_a_change_saves_to_the_pipeline_after_a_second(self):
        found = run("""
            await flush();
            type(find(".mc-voice-box-prompt"), "Speaker 1: New words.");
            const before = requestsTo("/pipelines/save").length;
            advance(999);
            await flush();
            const early = requestsTo("/pipelines/save").length;
            advance(1);
            await flush();
            report({before, early, saves: requestsTo("/pipelines/save").map(plain)});
        """)

        assert found["before"] == 0
        assert found["early"] == 0
        assert [save["body"] for save in found["saves"]] == [{"id": "p1", "prompt": "Speaker 1: New words."}]
        assert found["state"]["dirty"]["prompt"] is False

    def test_a_history_entry_loads_into_the_box_and_a_star_is_a_request(self):
        found = run("""
            await flush();
            type(find(".mc-voice-box-prompt"), "");
            find(".mc-voice-box-history-list .mc-voice-box-prompt-text").click();
            const star = find(".mc-voice-box-history-list .mc-voice-box-prompt-star");
            star.click();
            await flush();
            report({prompt: find(".mc-voice-box-prompt").value,
                    favourite: requestsTo("/prompts/favourite").map(plain)});
        """)

        assert found["prompt"] == "Speaker 1: Hello there."
        assert found["favourite"][0]["body"] == {"id": "h1", "on": True, "favourite": True}


# --------------------------------------------------------------------------- #
# The WAV encoder and the trimmer
# --------------------------------------------------------------------------- #


class TestTheWavEncoder:
    def test_a_synthetic_buffer_becomes_mono_16_bit_24k_wav(self):
        found = run("""
            const bytes = globalThis.mcVoiceBox.encodeWav(new Float32Array([0, 0.5, -0.5, 1, -1, 2]), 24000);
            report({wav: wavInfo(bytes), length: bytes.byteLength});
        """)

        wav = found["wav"]
        assert found["length"] == 44 + 6 * 2
        assert wav["riff"] == "RIFF" and wav["wave"] == "WAVE" and wav["fmt"] == "fmt " and wav["data"] == "data"
        assert wav["riffSize"] == 36 + 12
        assert wav["fmtSize"] == 16
        assert wav["format"] == 1
        assert wav["channels"] == 1
        assert wav["rate"] == 24000
        assert wav["byteRate"] == 48000
        assert wav["blockAlign"] == 2
        assert wav["bits"] == 16
        assert wav["dataLength"] == 12
        assert wav["samples"][:6] == [0, 16384, -16384, 32767, -32768, 32767]

    def test_the_linear_resampler_halves_a_48k_buffer(self):
        found = run("""
            const out = globalThis.mcVoiceBox.resampleLinear(new Float32Array([0, 1, 0, 1, 0, 1, 0, 1]), 48000, 24000);
            report({length: out.length, values: Array.from(out)});
        """)

        assert found["length"] == 4
        assert found["values"] == [0, 0, 0, 0]


class TestTheTrimmer:
    CHOOSE = """
        await flush();
        const input = find(".mc-voice-box-file-input");
        input.files = [new File([new Uint8Array(1000)], "song.mp3", {type: "audio/mpeg"})];
        input.dispatchEvent({type: "change"});
        await flush();
    """

    def test_a_chosen_file_is_decoded_and_the_whole_selection_saved_as_a_sample(self):
        found = run(self.CHOOSE + """
            const before = JSON.parse(JSON.stringify(globalThis.mcVoiceBox.state().trimming));
            find(".mc-voice-box-trimmer-name").value = "Ada intro";
            press("Save as sample");
            await flush(10);
            const upload = requestsTo("/samples/upload")[0];
            report({before, trimmerShown: !find(".mc-voice-box-trimmer").hidden,
                    readout: find(".mc-voice-box-trimmer-readout").textContent,
                    upload: upload ? plain(upload) : null,
                    wav: upload ? wavInfo(upload.body.raw) : null,
                    samplesRefetched: requestsTo("/samples").length});
        """)

        assert found["before"] == {"name": "song.mp3", "from": "file", "mode": "buffer", "duration": 10}
        assert found["trimmerShown"] is True
        assert found["readout"] == "0.0 s to 10.0 s (10.0 s of 10.0 s)"
        upload = found["upload"]
        assert upload["method"] == "POST"
        assert upload["headers"]["x-model-chain-voice"] == TOKEN
        assert upload["headers"]["x-mc-title"] == "Ada%20intro"
        assert upload["headers"]["x-mc-source"] == "file"
        assert upload["headers"]["content-type"] == "audio/wav"
        assert upload["signal"] is True
        wav = found["wav"]
        assert (wav["channels"], wav["rate"], wav["bits"]) == (1, 24000, 16)
        assert wav["dataLength"] == 10 * 24000 * 2
        assert wav["total"] == 44 + 10 * 24000 * 2
        assert found["samplesRefetched"] == 2

    def test_only_the_selection_is_encoded(self):
        found = run(self.CHOOSE + """
            const wave = find(".mc-voice-box-trimmer-wave");
            wave.dispatchEvent({type: "pointerdown", clientX: 120, pointerId: 1, preventDefault() {}});
            wave.dispatchEvent({type: "pointerup", pointerId: 1});
            const readout = find(".mc-voice-box-trimmer-readout").textContent;
            press("Save as sample");
            await flush(10);
            const upload = requestsTo("/samples/upload")[0];
            report({readout, wav: upload ? wavInfo(upload.body.raw) : null});
        """)

        assert found["readout"] == "5.0 s to 10.0 s (5.0 s of 10.0 s)"
        assert found["wav"]["dataLength"] == 5 * 24000 * 2

    def test_an_offline_audio_context_does_the_resampling_when_there_is_one(self):
        found = run(self.CHOOSE + """
            press("Save as sample");
            await flush(10);
            const upload = requestsTo("/samples/upload")[0];
            report({wav: upload ? wavInfo(upload.body.raw) : null});
        """, offline=True)

        assert found["wav"]["samples"][:4] == [8192, 8192, 8192, 8192]
        assert found["wav"]["dataLength"] == 10 * 24000 * 2

    def test_a_file_the_browser_will_not_decode_is_captured_while_it_plays(self):
        found = run("""
            await flush();
            find(".mc-voice-box-trimmer-audio").duration = 30;
            const input = find(".mc-voice-box-file-input");
            input.files = [new File([new Uint8Array(1000)], "long.mkv", {type: "video/x-matroska"})];
            input.dispatchEvent({type: "change"});
            await flush();
            const mode = globalThis.mcVoiceBox.state().trimming;
            press("Save as sample");
            await flush();
            const audio = find(".mc-voice-box-trimmer-audio");
            const capturing = globalThis.mcVoiceBox.state().capturing;
            const progressShown = !find(".mc-voice-box-progress").hidden;
            const playsDuringCapture = audio.plays;
            for (let index = 0; index < 3; index += 1) {
                lastProcessor.onaudioprocess({inputBuffer: {length: 4096, numberOfChannels: 1,
                                              getChannelData: () => new Float32Array(4096).fill(0.5)}});
            }
            audio.currentTime = 15;
            audio.dispatchEvent({type: "timeupdate"});
            const halfway = find(".mc-voice-box-progress-bar").style.width;
            audio.currentTime = 30;
            audio.dispatchEvent({type: "timeupdate"});
            await flush(10);
            const upload = requestsTo("/samples/upload")[0];
            report({mode, capturing, progressShown, playsDuringCapture, halfway,
                    paused: audio.paused, wav: upload ? wavInfo(upload.body.raw) : null,
                    focus: focusEvents});
        """, decode_works=False)

        assert found["mode"] == {"name": "long.mkv", "from": "file", "mode": "capture", "duration": 30}
        assert found["capturing"] is True
        assert found["progressShown"] is True
        assert found["playsDuringCapture"] == 1
        assert found["halfway"] == "50%"
        assert found["paused"] is True
        assert found["wav"]["dataLength"] == 3 * 4096 // 2 * 2
        assert found["wav"]["rate"] == 24000
        assert found["state"]["capturing"] is False
        assert {"owner": "voice-box", "kind": "playback"} in found["focus"]

    def test_the_trimmer_plays_the_selection_and_pauses_at_its_end(self):
        found = run(self.CHOOSE + """
            const audio = find(".mc-voice-box-trimmer-audio");
            press("Play selection");
            await flush();
            const plays = audio.plays;
            audio.currentTime = 9.99;
            audio.dispatchEvent({type: "timeupdate"});
            report({plays, paused: audio.paused, focus: focusEvents});
        """)

        assert found["plays"] == 1
        assert found["paused"] is True
        assert found["focus"] == [{"owner": "voice-box", "kind": "playback"}]


class TestTheSelectionMaths:
    def test_the_in_handle_leaves_the_minimum_and_the_out_handle_the_maximum(self):
        found = run("""
            const c = globalThis.mcVoiceBox.clampSelection;
            report({
                inTooClose: c({in: 0, out: 10}, "in", 9.5, 10),
                inNegative: c({in: 2, out: 10}, "in", -5, 10),
                outTooFar: c({in: 0, out: 10}, "out", 200, 100),
                outPastEnd: c({in: 0, out: 10}, "out", 15, 12),
                outTooClose: c({in: 4, out: 10}, "out", 4.5, 10),
                shortFile: c({in: 0, out: 2}, "in", 1.5, 2),
                inPastMaximum: c({in: 0, out: 100}, "in", 10, 100),
            });
        """)

        assert found["inTooClose"] == {"in": 7, "out": 10}
        assert found["inNegative"] == {"in": 0, "out": 10}
        assert found["outTooFar"] == {"in": 0, "out": 60}
        assert found["outPastEnd"] == {"in": 0, "out": 12}
        assert found["outTooClose"] == {"in": 4, "out": 7}
        assert found["shortFile"] == {"in": 0, "out": 2}
        assert found["inPastMaximum"] == {"in": 40, "out": 100}


# --------------------------------------------------------------------------- #
# Audio focus
# --------------------------------------------------------------------------- #


class TestAudioFocus:
    PLAY_LANE = """
        await flush();
        const lane = find(".mc-voice-box-lane");
        const audio = lane.querySelector("audio");
        press("Play", lane);
        await flush();
    """

    def test_playing_a_lane_says_so_before_the_audio_starts(self):
        found = run(self.PLAY_LANE + """
            const fetched = requestsTo("/outputs/audio")[0];
            report({plays: audio.plays, paused: audio.paused, src: audio.src,
                    label: lane.querySelector(".mc-voice-box-lane-play").textContent,
                    fetched: fetched ? plain(fetched) : null});
        """)

        assert found["focusEvents"] == [{"owner": "voice-box", "kind": "playback"}]
        assert found["plays"] == 1 and found["paused"] is False
        assert found["src"].startswith("blob:")
        assert found["label"] == "Pause"
        assert found["fetched"]["method"] == "GET"
        assert found["fetched"]["headers"]["x-model-chain-voice"] == TOKEN
        assert found["fetched"]["signal"] is True
        assert found["fetched"]["body"] is None
        assert found["state"]["focus"]["sent"] == [{"owner": "voice-box", "kind": "playback"}]

    def test_voice_chat_speaking_pauses_a_playing_lane(self):
        found = run(self.PLAY_LANE + """
            focusFrom("voice-chat", "speech");
            report({paused: audio.paused, pauses: audio.pauses,
                    label: lane.querySelector(".mc-voice-box-lane-play").textContent,
                    last: globalThis.mcVoiceBox.focusEvent().last});
        """)

        assert found["paused"] is True
        assert found["pauses"] == 1
        assert found["label"] == "Play"
        assert found["last"] == {"owner": "voice-chat", "kind": "speech"}

    def test_its_own_announcement_pauses_nothing(self):
        found = run(self.PLAY_LANE + """
            focusFrom("voice-box", "playback");
            report({paused: audio.paused, pauses: audio.pauses});
        """)

        assert found["paused"] is False
        assert found["pauses"] == 0

    def test_only_one_lane_plays_at_a_time(self):
        found = run("""
            await flush();
            const lanes = all(".mc-voice-box-lane");
            press("Play", lanes[0]);
            await flush();
            press("Play", lanes[1]);
            await flush();
            report({first: lanes[0].querySelector("audio").paused,
                    second: lanes[1].querySelector("audio").paused,
                    names: texts(".mc-voice-box-lane-name")});
        """, answers={"/outputs": {"json": {"ok": True, "outputs": [
            OUTPUT, dict(OUTPUT, id="o2", name="Take 2", created=3)]}}})

        assert found["names"] == ["Take 2", "Take 1"]
        assert found["first"] is True
        assert found["second"] is False

    def test_recording_announces_capture_and_voice_chat_stops_it(self):
        found = run("""
            await flush();
            press("Record");
            await flush();
            const during = {state: recorders[0].state, label: find(".mc-voice-box-record").textContent,
                            pressed: find(".mc-voice-box-record").getAttribute("aria-pressed"),
                            focus: focusEvents.slice()};
            focusFrom("voice-chat", "capture");
            await flush();
            report({during, after: recorders[0].state, tracks: tracks.map((t) => t.stopped),
                    label: find(".mc-voice-box-record").textContent,
                    trimming: globalThis.mcVoiceBox.state().trimming});
        """)

        assert found["during"] == {"state": "recording", "label": "Stop", "pressed": "true",
                                   "focus": [{"owner": "voice-box", "kind": "capture"}]}
        assert found["after"] == "inactive"
        assert found["tracks"] == [True]
        assert found["label"] == "Record"
        assert found["trimming"] == {"name": "Recording", "from": "microphone", "mode": "buffer",
                                     "duration": 10}

    def test_a_page_without_a_microphone_says_so_instead_of_failing(self):
        found = run("""
            await flush();
            press("Record");
            await flush();
            report({recorders: recorders.length});
        """, microphone=False)

        assert found["recorders"] == 0
        assert found["status"] == "This browser cannot record here."
        assert found["focusEvents"] == []


# --------------------------------------------------------------------------- #
# Requests
# --------------------------------------------------------------------------- #


class TestEveryRequest:
    def test_every_request_carries_the_token_and_an_abort_signal(self):
        found = run("""
            await flush();
            press("Render");
            await flush();
            press("Loop");
            await flush();
            press("Play", find(".mc-voice-box-lane"));
            await flush();
            press("Delete", find(".mc-voice-box-sample"));
            await flush();
            report();
        """)

        routes = {request["route"] for request in found["requests"]}
        assert {"/render", "/outputs/loop", "/outputs/audio", "/samples/delete"} <= routes
        for request in found["requests"]:
            assert request["headers"]["x-model-chain-voice"] == TOKEN, request
            assert request["signal"] is True, request
            if request["method"] == "POST" and request["route"] != "/samples/upload":
                assert request["headers"]["content-type"] == "application/json", request
        assert found["warnings"] == []

    def test_a_json_request_is_abandoned_after_twenty_seconds_and_says_so(self):
        found = run("""
            await flush();
            const delays = pendingDelays().slice();
            advance(20000);
            await flush();
            report({delays, aborted: requestsTo("/status")[0].aborted});
        """, answers={"/status": {"never": True}})

        assert 20000 in found["delays"]
        assert found["aborted"] is True
        assert found["status"] == "The WebUI did not answer within 20 s."
        assert found["state"]["messageKind"] == "error"

    def test_audio_gets_two_minutes(self):
        found = run("""
            await flush();
            press("Play", find(".mc-voice-box-lane"));
            await flush();
            report({delays: pendingDelays()});
        """, answers={"/outputs/audio": {"never": True}})

        assert 120000 in found["delays"]

    def test_a_refusal_lands_in_the_status_line(self):
        found = run("""
            await flush();
            press("Delete", find(".mc-voice-box-sample"));
            await flush();
            report();
        """, answers={"/samples/delete": {"status": 404, "json": {"ok": False,
                                                                   "error": "That sample is no longer in the library."}}})

        assert found["status"] == "That sample is no longer in the library."
        assert found["state"]["messageKind"] == "error"


class TestASecondPress:
    def test_a_second_press_waits_for_the_first(self):
        found = run("""
            await flush();
            press("Render");
            press("Render");
            await flush();
            const during = requestsTo("/render").length;
            release("/render");
            await flush();
            const afterFirst = {count: requestsTo("/render").length,
                                own: Object.keys(globalThis.mcVoiceBox.state().ownJobs)};
            press("Render");
            await flush();
            report({during, afterFirst, later: requestsTo("/render").length});
        """, held=["/render"])

        assert found["during"] == 1
        assert found["afterFirst"] == {"count": 1, "own": ["j1"]}
        assert found["later"] == 2


    def test_a_write_with_a_different_payload_queues_behind_the_first(self):
        """Loop toggled twice while the first toggle is in flight: folding the
        second into the first would leave the server on and the button off."""
        found = run("""
            await flush();
            const lane = find(".mc-voice-box-lane");
            press("Loop", lane);
            press("Loop", lane);
            await flush();
            const during = requestsTo("/outputs/loop").length;
            release("/outputs/loop");
            await flush();
            const second = requestsTo("/outputs/loop").length;
            release("/outputs/loop");
            await flush();
            report({during, second, bodies: requestsTo("/outputs/loop").map((r) => r.body.loop),
                    pressed: lane.querySelector(".mc-voice-box-lane-loop").getAttribute("aria-pressed")});
        """, held=["/outputs/loop"])

        assert found["during"] == 1
        assert found["second"] == 2
        assert found["bodies"] == [True, False]
        assert found["pressed"] == "false"

    def test_a_rename_is_not_lost_behind_a_prompt_save(self):
        found = run("""
            await flush();
            type(find(".mc-voice-box-prompt"), "Speaker 1: Typed.");
            advance(1000);
            await flush();
            globalThis.prompt = () => "Renamed";
            press("Rename");
            await flush();
            const during = requestsTo("/pipelines/save").map((r) => r.body);
            release("/pipelines/save");
            await flush();
            release("/pipelines/save");
            await flush();
            report({during, after: requestsTo("/pipelines/save").map((r) => r.body)});
        """, held=["/pipelines/save"])

        assert found["during"] == [{"id": "p1", "prompt": "Speaker 1: Typed."}]
        assert found["after"] == [{"id": "p1", "prompt": "Speaker 1: Typed."}, {"id": "p1", "name": "Renamed"}]


# --------------------------------------------------------------------------- #
# Polling
# --------------------------------------------------------------------------- #


class TestPolling:
    def test_the_cadence_follows_the_live_jobs(self):
        found = run("""
            await flush();
            const idle = {delay: globalThis.mcVoiceBox.state().poll.delay, timers: pendingDelays()};
            answers["/status"] = {json: Object.assign({}, answers["/status"].json, {jobs: [LIVE_JOB]})};
            press("Render");
            await flush();
            const live = {delay: globalThis.mcVoiceBox.state().poll.delay, timers: pendingDelays()};
            const outputsBefore = requestsTo("/outputs").length;
            answers["/status"] = {json: Object.assign({}, answers["/status"].json,
                                  {jobs: [Object.assign({}, LIVE_JOB, {phase: "done", live: false, output_id: "o9"})]})};
            advance(1000);
            await flush(10);
            const done = {delay: globalThis.mcVoiceBox.state().poll.delay, timers: pendingDelays(),
                          outputs: requestsTo("/outputs").length - outputsBefore,
                          prompts: requestsTo("/prompts").length};
            report({idle, live, done, jobs: texts(".mc-voice-box-job-text")});
        """.replace("LIVE_JOB", json.dumps(JOB)))

        assert found["idle"]["delay"] == 15000 and 15000 in found["idle"]["timers"]
        assert found["live"]["delay"] == 1000 and 1000 in found["live"]["timers"]
        assert 15000 not in found["live"]["timers"]
        assert found["done"]["delay"] == 15000
        assert found["done"]["outputs"] == 1
        assert found["done"]["prompts"] == 2
        assert found["jobs"] == ["Pipeline 1 2 — done"]

    def test_somebody_elses_job_does_not_speed_the_poll_up(self):
        found = run("await flush(); report({jobs: texts('.mc-voice-box-job-text')});",
                    answers={"/status": {"json": status_with(jobs=[JOB])}})

        assert found["state"]["poll"]["delay"] == 15000
        assert found["jobs"] == ["Pipeline 1 2 — queued"]

    def test_a_hidden_tab_is_not_polled_and_a_shown_one_is(self):
        found = run("""
            await flush();
            document.visibilityState = "hidden";
            document.dispatchEvent(new Event("visibilitychange"));
            const hidden = {delay: globalThis.mcVoiceBox.state().poll.delay, timers: pendingDelays()};
            const statuses = requestsTo("/status").length;
            document.visibilityState = "visible";
            document.dispatchEvent(new Event("visibilitychange"));
            await flush();
            report({hidden, statuses, refetched: requestsTo("/status").length - statuses});
        """)

        assert found["hidden"]["delay"] == 0
        assert 15000 not in found["hidden"]["timers"]
        assert found["refetched"] == 1
        assert found["state"]["poll"]["delay"] == 15000

    def test_a_live_job_can_be_cancelled(self):
        found = run("""
            await flush();
            press("Render");
            await flush();
            press("Cancel");
            await flush();
            report({cancel: requestsTo("/jobs/cancel").map(plain)});
        """, answers={"/status": {"json": status_with(jobs=[JOB])}})

        assert [c["body"] for c in found["cancel"]] == [{"id": "j1"}]


# --------------------------------------------------------------------------- #
# Outputs
# --------------------------------------------------------------------------- #


class TestOutputs:
    def test_save_opens_the_folder_dialog_once_when_there_is_no_folder(self):
        found = run("""
            await flush();
            press("Save", find(".mc-voice-box-lane"));
            await flush(10);
            report({order: requests.map((r) => r.route).filter((r) => r === "/outputs/save" || r === "/settings/folder")});
        """, answers={"/outputs/save": {"sequence": [
            {"status": 409, "json": {"ok": False, "error": "Choose a folder to save renders into first."}},
            {"json": {"ok": True, "path": "D:/renders/Take 1.wav"}}]}})

        assert found["order"] == ["/outputs/save", "/settings/folder", "/outputs/save"]
        assert found["status"] == "Saved to D:/renders/Take 1.wav."

    def test_a_save_with_a_folder_is_one_request(self):
        found = run("""
            await flush();
            press("Save", find(".mc-voice-box-lane"));
            await flush();
            report({order: requests.map((r) => r.route).filter((r) => r === "/outputs/save" || r === "/settings/folder")});
        """)

        assert found["order"] == ["/outputs/save"]

    def test_download_fetches_with_the_token_and_clicks_a_download_link(self):
        found = run("""
            await flush();
            press("Download", find(".mc-voice-box-lane"));
            await flush();
            report({audio: requestsTo("/outputs/audio").map(plain)});
        """)

        request = found["audio"][0]
        assert request["url"].endswith("/outputs/audio?id=o1&download=1")
        assert request["method"] == "GET"
        assert request["headers"]["x-model-chain-voice"] == TOKEN
        assert found["linkClicks"] == [{"href": "blob:test-1", "download": "Take 1.wav"}]

    def test_loop_is_persisted_and_shown(self):
        found = run("""
            await flush();
            const lane = find(".mc-voice-box-lane");
            press("Loop", lane);
            await flush();
            report({loop: requestsTo("/outputs/loop").map(plain),
                    pressed: lane.querySelector(".mc-voice-box-lane-loop").getAttribute("aria-pressed"),
                    audioLoop: lane.querySelector("audio").loop});
        """)

        assert [r["body"] for r in found["loop"]] == [{"id": "o1", "on": True, "loop": True}]
        assert found["pressed"] == "true"
        assert found["audioLoop"] is True

    def test_trim_to_sample_opens_the_trimmer_on_the_render(self):
        found = run("""
            await flush();
            press("Trim to sample", find(".mc-voice-box-lane"));
            await flush(10);
            press("Save as sample");
            await flush(10);
            const upload = requestsTo("/samples/upload")[0];
            report({trimming: globalThis.mcVoiceBox.state().trimming,
                    source: upload ? upload.headers["x-mc-source"] : null,
                    title: upload ? upload.headers["x-mc-title"] : null});
        """)

        assert found["trimming"] == {"name": "Take 1", "from": "output:o1", "mode": "buffer", "duration": 10}
        assert found["source"] == "output:o1"
        assert found["title"] == "Take%201"

    def test_a_press_on_the_waveform_seeks(self):
        found = run("""
            await flush();
            const lane = find(".mc-voice-box-lane");
            const audio = lane.querySelector("audio");
            audio.duration = 4.5;
            lane.querySelector(".mc-voice-box-lane-wave").dispatchEvent(
                {type: "pointerdown", clientX: 120, pointerId: 1, preventDefault() {}});
            report({at: audio.currentTime});
        """)

        assert found["at"] == 2.25

    def test_delete_and_rename(self):
        found = run("""
            await flush();
            const lane = find(".mc-voice-box-lane");
            lane.querySelector(".mc-voice-box-lane-name").dispatchEvent({type: "dblclick"});
            const box = find(".mc-voice-box-rename");
            box.value = "Intro";
            box.dispatchEvent({type: "keydown", key: "Enter"});
            await flush();
            press("Delete", lane);
            await flush();
            report({rename: requestsTo("/outputs/rename").map(plain),
                    remove: requestsTo("/outputs/delete").map(plain),
                    outputs: requestsTo("/outputs").length});
        """)

        assert [r["body"] for r in found["rename"]] == [{"id": "o1", "name": "Intro"}]
        assert [r["body"] for r in found["remove"]] == [{"id": "o1"}]
        assert found["outputs"] == 3


# --------------------------------------------------------------------------- #
# Render, its gate and the configuration
# --------------------------------------------------------------------------- #


class TestTheRenderGate:
    def test_the_button_says_why_it_is_disabled(self):
        found = run("""
            await flush();
            const reasons = {};
            const reason = () => ({disabled: find(".mc-voice-box-render").disabled,
                                   why: find(".mc-voice-box-render-reason").textContent});
            reasons.ready = reason();
            type(find(".mc-voice-box-prompt"), "");
            reasons.empty = reason();
            type(find(".mc-voice-box-prompt"), "Speaker 2: hi");
            reasons.noSample = reason();
            type(find(".mc-voice-box-prompt"), "Speaker 7: x");
            reasons.tooMany = reason();
            report({reasons});
        """)

        assert found["reasons"]["ready"] == {"disabled": False, "why": ""}
        assert found["reasons"]["empty"] == {"disabled": True, "why": "Write a script first."}
        assert found["reasons"]["noSample"] == {"disabled": True, "why": "Speaker 2 has no sample."}
        assert found["reasons"]["tooMany"]["why"] == "Speakers are numbered 1 to 4; the prompt names Speaker 7."

    def test_an_engine_that_is_not_installed_offers_install(self):
        found = run("""
            await flush();
            const install = find(".mc-voice-box-install");
            const before = {hidden: install.hidden, label: install.textContent,
                            reason: find(".mc-voice-box-render-reason").textContent,
                            disabled: find(".mc-voice-box-render").disabled};
            install.click();
            await flush();
            report({before, install: requestsTo("/install").map(plain),
                    progress: find(".mc-voice-box-install-progress").textContent});
        """, answers={"/status": {"json": status_with(
            engine={"ready": False, "supported": True, "message": "VibeVoice is not installed.",
                    "model_id": "vibevoice-7b", "download_bytes": 19_000_000_000},
            progress={"running": True, "text": "Downloading shard 3 of 8", "fraction": 0.375,
                      "failed": False, "model": "vibevoice-7b"})}})

        assert found["before"] == {"hidden": False, "label": "Install VibeVoice (19.0 GB)",
                                   "reason": "VibeVoice is not installed.", "disabled": True}
        assert [r["body"] for r in found["install"]] == [{"part": "", "folder": ""}]
        assert found["progress"] == "Downloading shard 3 of 8 (38%)"

    def test_an_installed_engine_hides_the_install_button(self):
        found = run("await flush(); report({hidden: find('.mc-voice-box-install').hidden});")

        assert found["hidden"] is True

    def test_render_sends_the_saved_configuration_by_id(self):
        found = run("""
            await flush();
            press("Render");
            await flush();
            report({render: requestsTo("/render").map(plain)});
        """)

        assert [r["body"] for r in found["render"]] == [
            {"pipeline_id": "p1", "prompt": "Speaker 1: Hello there.", "configuration_id": "c1", "name": ""}]
        assert found["status"] == "Render queued."

    def test_render_sends_an_unsaved_configuration_inline(self):
        found = run("""
            await flush();
            press("2", find(".mc-voice-box-sample"));
            const dirty = {save: find(".mc-voice-box-configuration-save").textContent,
                           slot: find('.mc-voice-box-speaker[data-speaker="2"] .mc-voice-box-speaker-title').textContent};
            press("Render");
            await flush();
            report({dirty, render: requestsTo("/render").map(plain),
                    saves: requestsTo("/configurations/save").length});
        """)

        assert found["dirty"] == {"save": "Save •", "slot": "Ada"}
        body = found["render"][0]["body"]
        assert body["configuration_id"] == "c1"
        assert body["configuration"]["speakers"] == {"1": "s1", "2": "s1"}
        assert body["configuration"]["steps"] == 10
        assert found["saves"] == 0

    def test_save_writes_the_configuration_and_keep_warm_is_a_setting(self):
        found = run("""
            await flush();
            const steps = find('[data-field="steps"]');
            steps.value = "20";
            steps.dispatchEvent({type: "change"});
            find(".mc-voice-box-configuration-save").click();
            await flush();
            const warm = find(".mc-voice-box-keep-warm input");
            warm.checked = false;
            warm.dispatchEvent({type: "change"});
            await flush();
            report({save: requestsTo("/configurations/save").map(plain),
                    settings: requestsTo("/settings").map(plain)});
        """)

        save = found["save"][0]["body"]
        assert save["id"] == "c1" and save["name"] == "Default" and save["steps"] == 20
        assert "keep_warm" not in save
        assert [r["body"] for r in found["settings"]] == [{"keep_warm": False}]

    def test_choosing_a_card_becomes_the_default_too(self):
        found = run("""
            await flush();
            const card = find('[data-field="card_uuid"]');
            card.value = "GPU-b";
            card.dispatchEvent({type: "change"});
            await flush();
            report({settings: requestsTo("/settings").map(plain)});
        """)

        assert [r["body"] for r in found["settings"]] == [{"card_uuid": "GPU-b"}]
        assert found["state"]["working"]["card_uuid"] == "GPU-b"
        assert found["state"]["dirty"]["configuration"] is True

    def test_the_cards_line_and_unload(self):
        found = run("""
            await flush();
            const line = find(".mc-voice-box-cards").textContent;
            press("Unload VibeVoice from RTX 3090");
            await flush();
            report({line, runtime: requestsTo("/runtime").map(plain)});
        """, answers={"/status": {"json": status_with(
            turns=[{"card": "RTX 3090", "uuid": "GPU-a", "active": "VibeVoice 7B rendering",
                    "queued": [{"label": "image"}], "warm": "", "parked": True, "lease": False}],
            runtime={"cards": {"abc": {"running": True, "loaded": True, "rendering": True, "job": "j1",
                                       "resident_bytes": 19_300_000_000, "peak_bytes": 0,
                                       "device_name": "RTX 3090", "uuid": "GPU-a"}},
                     "last_error": ""})}})

        assert found["line"] == ("RTX 3090: VibeVoice 7B rendering, 1 waiting, image model parked · "
                                 "VibeVoice rendering on RTX 3090 (19.3 GB)")
        assert [r["body"] for r in found["runtime"]] == [{"action": "unload", "card_uuid": "GPU-a"}]


class TestPipelines:
    def test_new_creates_and_opens_a_pipeline(self):
        found = run("""
            await flush();
            press("New");
            await flush();
            report({made: requestsTo("/pipelines/new").map(plain),
                    title: find(".mc-voice-box-title").textContent});
        """, answers={"/pipelines/new": {"json": {"ok": True, "pipeline": dict(PIPELINE, id="p2", name="Pipeline 2", prompt="")}},
                      "/pipelines": {"sequence": [
                          {"json": {"ok": True, "pipelines": [PIPELINE]}},
                          {"json": {"ok": True, "pipelines": [dict(PIPELINE, id="p2", name="Pipeline 2", prompt=""), PIPELINE]}}]}})

        assert [r["body"] for r in found["made"]] == [{"name": "Pipeline 2"}]
        assert found["title"] == "Voice Box · Pipeline 2"
        assert found["state"]["pipelineId"] == "p2"
        assert found["storageLog"][-1] == ["mc-voice-box:pipeline", "p2"]

    def test_a_page_with_no_pipeline_makes_the_first(self):
        found = run("await flush(); report({made: requestsTo('/pipelines/new').map(plain)});",
                    answers={"/pipelines": {"sequence": [
                        {"json": {"ok": True, "pipelines": []}},
                        {"json": {"ok": True, "pipelines": [PIPELINE]}}]},
                        "/pipelines/new": {"json": {"ok": True, "pipeline": PIPELINE}}})

        assert [r["body"] for r in found["made"]] == [{"name": "Pipeline 1"}]
        assert found["state"]["pipelineId"] == "p1"


# --------------------------------------------------------------------------- #
# The models: what each supports, its defaults, its voices
# --------------------------------------------------------------------------- #

# What the configuration shows, read off the page.
SHOWN = """
    const shownFields = () => ({
        precision: shown('[data-field-wrap="precision"]'),
        lora: shown('[data-field-wrap="lora_id"]'),
        strength: shown('[data-field-wrap="lora_scale"]'),
        slots: all(".mc-voice-box-speaker[data-speaker]").filter((s) => !s.hidden)
            .map((s) => s.getAttribute("data-speaker")),
        steps: find('[data-field="steps"]').value,
        cfg: find('[data-field="cfg_scale"]').value,
    });
"""


class TestTheModels:
    def test_the_model_select_lists_the_engines_models_and_says_which_are_not_installed(self):
        realtime = dict(MODEL_REALTIME, installed=False,
                        message="VibeVoice Realtime 0.5B is not installed.")
        found = run("""
            await flush();
            const select = find('[data-field="model_id"]');
            report({models: select.options.map((o) => [o.value, o.textContent]),
                    chosen: select.value});
        """, answers=described(MODEL_7B, realtime))

        assert found["models"] == [["vibevoice-7b", "VibeVoice 7B"],
                                   [REALTIME, "VibeVoice Realtime 0.5B — not installed"]]
        assert found["chosen"] == "vibevoice-7b"

    def test_choosing_a_model_applies_its_defaults_and_shows_only_what_it_supports(self):
        found = run(SHOWN + """
            await flush();
            const before = shownFields();
            const precisions = find('[data-field="precision"]').options.map((o) => o.textContent);
            const loras = find('[data-field="lora_id"]').options.map((o) => o.textContent);
            const strength = find('[data-field="lora_scale"]');
            const scale = {disabled: strength.disabled, min: strength.getAttribute("min"),
                           max: strength.getAttribute("max"), step: strength.getAttribute("step")};
            choose('[data-field="model_id"]', "REALTIME_ID");
            await flush();
            const working = globalThis.mcVoiceBox.state().working;
            report({before, precisions, loras, scale, after: shownFields(),
                    steps: working.steps, cfg: working.cfg_scale,
                    settings: requestsTo("/settings").map((r) => r.body)});
        """.replace("REALTIME_ID", REALTIME), answers=described())

        assert found["before"] == {"precision": True, "lora": True, "strength": True,
                                   "slots": ["1", "2", "3", "4"], "steps": "10", "cfg": "1.3"}
        assert found["precisions"] == ["Full (bf16) · 20.0 GB", "8-bit · 13.0 GB",
                                       "4-bit (NF4) · 9.0 GB"]
        assert found["loras"] == ["(no LoRA)", "Narrator"]
        assert found["scale"] == {"disabled": True, "min": "0", "max": "2", "step": "0.05"}
        assert found["after"] == {"precision": False, "lora": False, "strength": False,
                                  "slots": ["preset"], "steps": "5", "cfg": "1.5"}
        assert (found["steps"], found["cfg"]) == (5, 1.5)
        assert found["settings"] == [{"model_id": REALTIME}]
        assert found["state"]["dirty"]["configuration"] is True

    def test_an_engine_that_describes_no_model_shows_what_it_always_did(self):
        """mc_voice_box._FALLBACK_MODEL: the 7B's shape at full precision."""
        found = run(SHOWN + """
            await flush();
            report({fields: shownFields(), library: shown(".mc-voice-box-lora-library"),
                    preset: shown(".mc-voice-box-speaker-preset"),
                    installs: shown(".mc-voice-box-installs")});
        """)

        assert found["fields"] == {"precision": False, "lora": False, "strength": False,
                                   "slots": ["1", "2", "3", "4"], "steps": "10", "cfg": "1.3"}
        assert found["library"] is False
        assert found["preset"] is False
        assert found["installs"] is False

    def test_a_model_with_fewer_speakers_shows_fewer_slots_and_says_so_in_the_plural(self):
        duo = dict(MODEL_7B, id="duo", label="Duo", max_speakers=2)
        found = run(SHOWN + """
            await flush();
            choose('[data-field="model_id"]', "duo");
            await flush();
            type(find(".mc-voice-box-prompt"), "Speaker 1: a\\nSpeaker 3: b");
            report({fields: shownFields(),
                    assign: all(".mc-voice-box-sample-speaker").map((b) => b.disabled),
                    summary: find(".mc-voice-box-prompt-summary").textContent,
                    why: find(".mc-voice-box-render-reason").textContent});
        """, answers=described(MODEL_7B, duo))

        assert found["fields"]["slots"] == ["1", "2"]
        assert found["assign"] == [False, False, True, True]
        assert found["why"] == "Duo speaks with up to 2 voices, and the script names Speaker 3."
        assert found["summary"].endswith("— " + found["why"])

    def test_the_preset_select_is_grouped_by_language_with_experimental_marked(self):
        found = run("""
            await flush();
            choose('[data-field="model_id"]', "REALTIME_ID");
            await flush();
            const select = find(".mc-voice-box-preset-select");
            const groups = select.querySelectorAll("optgroup").map((g) => [
                g.getAttribute("label"),
                g.querySelectorAll("option").map((o) => [o.value, o.textContent, o.disabled])]);
            const lead = [select.children[0].value, select.children[0].textContent];
            const assign = all(".mc-voice-box-sample-speaker").map((b) => b.disabled);
            choose(".mc-voice-box-preset-select", "preset:en-Carter_man");
            await flush();
            report({groups, lead, assign, speakers: globalThis.mcVoiceBox.state().working.speakers});
        """.replace("REALTIME_ID", REALTIME), answers=described())

        assert found["groups"] == [
            ["English", [["preset:en-Carter_man", "Carter (man)", False],
                         ["preset:en-Emma_woman", "Emma (woman)", False]]],
            ["German", [["preset:de-Spk0_man", "Spk0 (man) · experimental", False]]],
            ["Japanese", [["preset:jp-Spk1_woman", "Spk1 (woman) · experimental · not installed",
                           True]]]]
        assert found["lead"] == ["", "(choose a voice)"]
        assert found["assign"] == [True, True, True, True]
        assert found["speakers"] == {"1": "preset:en-Carter_man"}
        assert found["status"] == "Speaker 1 is Carter. Save the configuration to keep it."

    def test_a_new_configuration_starts_from_its_models_defaults(self):
        found = run("await flush(); report();", answers=dict(
            described(settings={"model_id": REALTIME}),
            **{"/configurations": {"json": {"ok": True, "configurations": []}}}))

        working = found["state"]["working"]
        assert working["model_id"] == REALTIME
        assert (working["steps"], working["cfg_scale"]) == (5, 1.5)
        assert (working["precision"], working["lora_id"], working["lora_scale"]) == ("bf16", "", 1)

    def test_a_poll_brings_the_engines_news_a_voice_installed_and_a_lora_added(self):
        installed = [dict(preset, installed=True) for preset in PRESETS]
        later = described(MODEL_7B, dict(MODEL_REALTIME, presets=installed),
                          loras=[LORA, WARM])["/status"]["json"]
        found = run("""
            await flush();
            choose('[data-field="model_id"]', "REALTIME_ID");
            await flush();
            const japanese = () => find(".mc-voice-box-preset-select")
                .querySelectorAll("option").filter((o) => o.value === "preset:jp-Spk1_woman")[0].disabled;
            const before = {japanese: japanese(), loras: texts(".mc-voice-box-lora-name")};
            answers["/status"] = {json: LATER};
            advance(15000);
            await flush();
            report({before, after: {japanese: japanese(), loras: texts(".mc-voice-box-lora-name")}});
        """.replace("REALTIME_ID", REALTIME).replace("LATER", json.dumps(later)),
            answers=described())

        assert found["before"] == {"japanese": True, "loras": ["Narrator"]}
        assert found["after"] == {"japanese": False, "loras": ["Narrator", "Warm"]}

    def test_a_poll_does_not_rebuild_a_select_whose_options_are_unchanged(self):
        """A select rebuilt under an open dropdown closes it; the poll repaints
        the configuration every fifteen seconds."""
        found = run("""
            await flush();
            const first = find('[data-field="precision"]').children[0];
            const statuses = requestsTo("/status").length;
            advance(15000);
            await flush();
            report({polled: requestsTo("/status").length - statuses,
                    same: find('[data-field="precision"]').children[0] === first});
        """, answers=described())

        assert found["polled"] == 1
        assert found["same"] is True


class TestTheSavedConfiguration:
    def test_the_save_carries_the_precision_the_lora_and_its_strength(self):
        found = run("""
            await flush();
            choose('[data-field="precision"]', "int8");
            choose('[data-field="lora_id"]', "0123456789abcdef");
            const strength = find('[data-field="lora_scale"]');
            const enabled = !strength.disabled;
            type(strength, "0.75");
            find(".mc-voice-box-configuration-save").click();
            await flush();
            report({enabled, save: requestsTo("/configurations/save").map((r) => r.body)});
        """, answers=described())

        assert found["enabled"] is True
        save = found["save"][0]
        assert save["id"] == "c1" and save["name"] == "Default"
        assert save["model_id"] == "vibevoice-7b"
        assert (save["precision"], save["lora_id"], save["lora_scale"]) == \
            ("int8", "0123456789abcdef", 0.75)
        assert save["speakers"] == {"1": "s1"}

    def test_a_preset_model_is_saved_at_its_own_precision_with_no_lora_and_a_preset(self):
        """What the working copy holds for the 7B -- a quantised precision, a
        LoRA, a sample on speaker 1 -- is not what the Realtime model takes."""
        found = run("""
            await flush();
            choose('[data-field="precision"]', "nf4");
            choose('[data-field="lora_id"]', "0123456789abcdef");
            choose('[data-field="model_id"]', "REALTIME_ID");
            choose(".mc-voice-box-preset-select", "preset:en-Carter_man");
            find(".mc-voice-box-configuration-save").click();
            await flush();
            report({save: requestsTo("/configurations/save").map((r) => r.body)});
        """.replace("REALTIME_ID", REALTIME), answers=described())

        save = found["save"][0]
        assert save["model_id"] == REALTIME
        assert (save["precision"], save["lora_id"], save["lora_scale"]) == ("bf16", "", 1)
        assert save["speakers"] == {"1": "preset:en-Carter_man"}
        assert (save["steps"], save["cfg_scale"]) == (5, 1.5)

    def test_an_unsaved_preset_configuration_renders_inline_in_the_same_form(self):
        found = run("""
            await flush();
            choose('[data-field="precision"]', "int8");
            choose('[data-field="lora_id"]', "0123456789abcdef");
            choose('[data-field="model_id"]', "REALTIME_ID");
            choose(".mc-voice-box-preset-select", "preset:en-Carter_man");
            press("Render");
            await flush();
            report({render: requestsTo("/render").map((r) => r.body)});
        """.replace("REALTIME_ID", REALTIME), answers=described())

        body = found["render"][0]
        assert body["configuration_id"] == "c1"
        inline = body["configuration"]
        assert inline["model_id"] == REALTIME
        assert inline["speakers"] == {"1": "preset:en-Carter_man"}
        assert (inline["precision"], inline["lora_id"], inline["lora_scale"]) == ("bf16", "", 1)

    def test_the_speakers_of_the_other_kind_of_model_wait_on_the_shelf(self):
        found = run("""
            await flush();
            const speakers = () => JSON.parse(JSON.stringify(globalThis.mcVoiceBox.state().working.speakers));
            const seen = [speakers()];
            choose('[data-field="model_id"]', "REALTIME_ID");
            seen.push(speakers());
            choose(".mc-voice-box-preset-select", "preset:en-Carter_man");
            choose('[data-field="model_id"]', "vibevoice-7b");
            seen.push(speakers());
            const slot = find('.mc-voice-box-speaker[data-speaker="1"] .mc-voice-box-speaker-title').textContent;
            choose('[data-field="model_id"]', "REALTIME_ID");
            seen.push(speakers());
            report({seen, slot, preset: find(".mc-voice-box-preset-select").value});
        """.replace("REALTIME_ID", REALTIME), answers=described())

        assert found["seen"] == [{"1": "s1"}, {}, {"1": "s1"}, {"1": "preset:en-Carter_man"}]
        assert found["slot"] == "Ada"
        assert found["preset"] == "preset:en-Carter_man"

    def test_the_save_leaves_out_slots_past_the_models_last_speaker(self):
        duo = dict(MODEL_7B, id="duo", label="Duo", max_speakers=2)
        found = run("""
            await flush();
            find(".mc-voice-box-configuration-save").click();
            await flush();
            report({save: requestsTo("/configurations/save").map((r) => r.body)});
        """, answers=dict(described(MODEL_7B, duo), **{"/configurations": {"json": {
            "ok": True, "configurations": [dict(CONFIGURATION, model_id="duo",
                                                speakers={"1": "s1", "4": "s1"})]}}}))

        assert found["save"][0]["speakers"] == {"1": "s1"}

    def test_a_lora_chosen_while_its_strength_is_empty_starts_at_full_strength(self):
        found = run("""
            await flush();
            choose('[data-field="lora_id"]', "0123456789abcdef");
            type(find('[data-field="lora_scale"]'), "");
            const emptied = globalThis.mcVoiceBox.state().working.lora_scale;
            choose('[data-field="lora_id"]', "");
            choose('[data-field="lora_id"]', "0123456789abcdef");
            report({emptied, box: find('[data-field="lora_scale"]').value,
                    scale: globalThis.mcVoiceBox.state().working.lora_scale});
        """, answers=described())

        assert found["emptied"] is None
        assert found["scale"] == 1
        assert found["box"] == "1"

    def test_a_deleted_sample_leaves_the_shelf_too(self):
        found = run("""
            await flush();
            choose('[data-field="model_id"]', "REALTIME_ID");
            press("Delete", find(".mc-voice-box-sample"));
            await flush();
            choose('[data-field="model_id"]', "vibevoice-7b");
            report({speakers: globalThis.mcVoiceBox.state().working.speakers});
        """.replace("REALTIME_ID", REALTIME), answers=described())

        assert found["speakers"] == {}


class TestTheSpeakerLimit:
    def test_the_summary_warns_and_render_says_why_when_the_script_names_too_many(self):
        found = run("""
            await flush();
            const look = () => ({text: find(".mc-voice-box-prompt-summary").textContent,
                                 kind: find(".mc-voice-box-prompt-summary").getAttribute("data-kind"),
                                 disabled: find(".mc-voice-box-render").disabled,
                                 why: find(".mc-voice-box-render-reason").textContent});
            choose('[data-field="model_id"]', "REALTIME_ID");
            choose(".mc-voice-box-preset-select", "preset:en-Carter_man");
            type(find(".mc-voice-box-prompt"), "Speaker 1: Hello.\\nSpeaker 2: Hi there.");
            const warned = look();
            press("Render");
            await flush();
            type(find(".mc-voice-box-prompt"), "Speaker 1: Hello there.");
            const fine = look();
            type(find(".mc-voice-box-prompt"), "Speaker 1: Hello.\\nSpeaker 2: Hi there.");
            choose('[data-field="model_id"]', "vibevoice-7b");
            const seven = look();
            report({warned, fine, seven, renders: requestsTo("/render").length});
        """.replace("REALTIME_ID", REALTIME), answers=described())

        sentence = "VibeVoice Realtime 0.5B speaks with one voice, and the script names Speaker 2."
        assert found["warned"] == {"text": "2 speakers, 0 pauses, ~3 words — " + sentence,
                                   "kind": "warn", "disabled": True, "why": sentence}
        assert found["renders"] == 0
        assert found["fine"] == {"text": "1 speaker, 0 pauses, ~2 words", "kind": "info",
                                 "disabled": False, "why": ""}
        assert found["seven"] == {"text": "2 speakers, 0 pauses, ~3 words", "kind": "info",
                                  "disabled": True, "why": "Speaker 2 has no sample."}

    def test_a_preset_model_asks_for_a_voice_before_it_renders(self):
        found = run("""
            await flush();
            choose('[data-field="model_id"]', "REALTIME_ID");
            report({why: find(".mc-voice-box-render-reason").textContent,
                    disabled: find(".mc-voice-box-render").disabled});
        """.replace("REALTIME_ID", REALTIME), answers=described())

        assert found["why"] == "Choose a voice for Speaker 1 in the configuration."
        assert found["disabled"] is True


# --------------------------------------------------------------------------- #
# The LoRA library
# --------------------------------------------------------------------------- #


class TestTheLoRALibrary:
    def test_the_library_lists_each_lora_with_its_size_and_parts(self):
        found = run("""
            await flush();
            report({shown: shown(".mc-voice-box-lora-library"),
                    summary: find(".mc-voice-box-lora-summary").textContent,
                    names: texts(".mc-voice-box-lora-name"), meta: texts(".mc-voice-box-lora-meta"),
                    heading: find(".mc-voice-box-lora-add .mc-voice-box-subtitle").textContent});
        """, answers=described(loras=[LORA, WARM]))

        assert found["shown"] is True
        assert found["summary"] == "LoRA library · 2 LoRAs"
        assert found["names"] == ["Narrator", "Warm"]
        assert found["meta"] == ["84 MB · language model, diffusion head", "2.5 GB · language model"]
        assert found["heading"] == "Add a LoRA from a folder on this PC"

    def test_add_copies_a_folder_with_the_long_deadline_and_a_second_press_waits(self):
        found = run("""
            await flush();
            find(".mc-voice-box-lora-folder").value = "D:/loras/warm";
            find(".mc-voice-box-lora-add-name").value = "Warm";
            find(".mc-voice-box-lora-add-button").click();
            find(".mc-voice-box-lora-add-button").click();
            await flush();
            const during = {count: requestsTo("/loras/add").length, delays: pendingDelays(),
                            status: find(".mc-voice-box-status").textContent};
            release("/loras/add");
            await flush();
            report({during, add: requestsTo("/loras/add").map(plain),
                    names: texts(".mc-voice-box-lora-name"),
                    boxes: [find(".mc-voice-box-lora-folder").value,
                            find(".mc-voice-box-lora-add-name").value],
                    offered: find('[data-field="lora_id"]').options.map((o) => o.textContent)});
        """, answers=dict(described(), **{"/loras/add": {"json": {
            "ok": True, "lora": WARM, "loras": [LORA, WARM]}}}), held=["/loras/add"])

        assert found["during"]["count"] == 1
        assert 120000 in found["during"]["delays"]
        assert found["during"]["status"] == "Copying the LoRA from D:/loras/warm…"
        add = found["add"][0]
        assert add["body"] == {"folder": "D:/loras/warm", "name": "Warm"}
        assert add["headers"]["x-model-chain-voice"] == TOKEN and add["signal"] is True
        assert found["names"] == ["Narrator", "Warm"]
        assert found["boxes"] == ["", ""]
        assert found["offered"] == ["(no LoRA)", "Narrator", "Warm"]
        assert found["status"] == "Added the LoRA Warm."

    def test_a_folder_the_server_refuses_is_said_in_the_status_line(self):
        found = run("""
            await flush();
            find(".mc-voice-box-lora-folder").value = "D:/music";
            find(".mc-voice-box-lora-add-button").click();
            await flush();
            report({names: texts(".mc-voice-box-lora-name"),
                    folder: find(".mc-voice-box-lora-folder").value});
        """, answers=dict(described(), **{"/loras/add": {"status": 400, "json": {
            "ok": False, "error": "That folder holds no LoRA adapter for the language model."}}}))

        assert found["status"] == "That folder holds no LoRA adapter for the language model."
        assert found["state"]["messageKind"] == "error"
        assert found["names"] == ["Narrator"]
        assert found["folder"] == "D:/music"

    def test_an_empty_folder_box_is_not_sent(self):
        found = run("""
            await flush();
            find(".mc-voice-box-lora-add-button").click();
            await flush();
            report({adds: requestsTo("/loras/add").length});
        """, answers=described())

        assert found["adds"] == 0
        assert found["status"] == "Give the folder that holds the LoRA."

    def test_an_add_with_no_answer_says_the_copy_may_still_be_running(self):
        found = run("""
            await flush();
            find(".mc-voice-box-lora-folder").value = "D:/loras/huge";
            find(".mc-voice-box-lora-add-button").click();
            await flush();
            advance(120000);
            await flush();
            report({aborted: requestsTo("/loras/add")[0].aborted});
        """, answers=dict(described(), **{"/loras/add": {"never": True}}))

        assert found["aborted"] is True
        assert found["status"] == ("No answer within 120 s. The copy may still be running; "
                                   "the library shows the LoRA when it is done.")

    def test_rename_and_delete_and_the_chosen_lora_lets_go(self):
        found = run("""
            await flush();
            press("Rename", find(".mc-voice-box-lora"));
            const box = find(".mc-voice-box-rename");
            box.value = "Storyteller";
            box.dispatchEvent({type: "keydown", key: "Enter"});
            await flush();
            const renamed = texts(".mc-voice-box-lora-name");
            choose('[data-field="lora_id"]', "0123456789abcdef");
            const chosen = globalThis.mcVoiceBox.state().working.lora_id;
            press("Delete", find(".mc-voice-box-lora"));
            await flush();
            report({renamed, chosen, rename: requestsTo("/loras/rename").map(plain),
                    remove: requestsTo("/loras/delete").map(plain),
                    after: globalThis.mcVoiceBox.state().working.lora_id,
                    empty: texts(".mc-voice-box-loras .mc-voice-box-empty"),
                    offered: find('[data-field="lora_id"]').options.map((o) => o.textContent),
                    lists: requestsTo("/loras").length});
        """, answers=dict(described(), **{
            "/loras/rename": {"json": {"ok": True, "lora": dict(LORA, name="Storyteller"),
                                       "loras": [dict(LORA, name="Storyteller")]}},
            "/loras/delete": {"json": {"ok": True, "loras": []}}}))

        assert [r["body"] for r in found["rename"]] == [{"id": "0123456789abcdef",
                                                         "name": "Storyteller"}]
        assert found["renamed"] == ["Storyteller"]
        assert [r["body"] for r in found["remove"]] == [{"id": "0123456789abcdef"}]
        for request in found["rename"] + found["remove"]:
            assert request["headers"]["x-model-chain-voice"] == TOKEN and request["signal"] is True
        assert found["chosen"] == "0123456789abcdef"
        assert found["after"] == ""
        assert found["state"]["dirty"]["configuration"] is True
        assert found["empty"] == ["No LoRAs yet. Add one from a folder on this PC."]
        assert found["offered"] == ["(no LoRA)"]
        assert found["lists"] == 0

    def test_a_saved_configuration_the_server_cleared_is_not_left_unsaved(self):
        """The server takes a deleted LoRA out of the configurations that used it
        (mc_voice_box.forget_lora); the page reads them again, and a working copy
        that agrees with its saved configuration is not marked unsaved."""
        found = run("""
            await flush();
            const before = globalThis.mcVoiceBox.state().working.lora_id;
            const reads = requestsTo("/configurations").length;
            press("Delete", find(".mc-voice-box-lora"));
            await flush();
            const working = globalThis.mcVoiceBox.state().working;
            report({before, after: working.lora_id, scale: working.lora_scale,
                    reads: requestsTo("/configurations").length - reads,
                    save: find(".mc-voice-box-configuration-save").textContent});
        """, answers=dict(described(), **{
            "/configurations": {"json": {"ok": True, "configurations": [dict(
                CONFIGURATION, lora_id="0123456789abcdef", lora_scale=0.5)]}},
            "/loras/delete": {"json": {"ok": True, "loras": [], "configurations": ["c1"]}}}))

        assert found["before"] == "0123456789abcdef"
        assert (found["after"], found["scale"]) == ("", 1)
        assert found["state"]["dirty"]["configuration"] is False
        assert found["save"] == "Save"
        assert found["reads"] == 1

    def test_an_answer_without_the_library_is_followed_by_a_read_of_it(self):
        found = run("""
            await flush();
            press("Delete", find(".mc-voice-box-lora"));
            await flush();
            report({lists: requestsTo("/loras").map(plain), names: texts(".mc-voice-box-lora-name")});
        """, answers=dict(described(), **{"/loras/delete": {"json": {"ok": True}},
                                          "/loras": {"json": {"ok": True, "loras": [WARM]}}}))

        assert [r["body"] for r in found["lists"]] == [{}]
        assert found["names"] == ["Warm"]

    def test_a_poll_leaves_a_rename_in_progress_alone(self):
        found = run("""
            await flush();
            press("Rename", find(".mc-voice-box-lora"));
            const statuses = requestsTo("/status").length;
            advance(15000);
            await flush();
            const box = find(".mc-voice-box-rename");
            const survived = !!box;
            if (box) {
                box.value = "Storyteller";
                box.dispatchEvent({type: "keydown", key: "Enter"});
                await flush();
            }
            report({polled: requestsTo("/status").length - statuses, survived,
                    rename: requestsTo("/loras/rename").map((r) => r.body)});
        """, answers=described())

        assert found["polled"] == 1
        assert found["survived"] is True
        assert found["rename"] == [{"id": "0123456789abcdef", "name": "Storyteller"}]

    def test_a_lora_made_for_another_model_is_not_offered(self):
        found = run("""
            await flush();
            report({offered: find('[data-field="lora_id"]').options.map((o) => o.textContent)});
        """, answers=described(loras=[LORA, dict(WARM, base="another-model")]))

        assert found["offered"] == ["(no LoRA)", "Narrator"]


# --------------------------------------------------------------------------- #
# Installing, model by model
# --------------------------------------------------------------------------- #

NOT_INSTALLED = dict(MODEL_REALTIME, installed=False, runtime_installed=False,
                     download_bytes=2_100_000_000,
                     message="VibeVoice Realtime 0.5B is not installed — install it below.")


class TestInstallingModelByModel:
    def test_the_footer_offers_the_runtime_and_each_missing_model(self):
        runtime_missing = dict(MODEL_7B, runtime_installed=False,
                               message="VibeVoice's runtime still to install.")
        found = run("""
            await flush();
            const offers = all(".mc-voice-box-install-part").map((b) => [
                b.textContent, b.getAttribute("data-part"), b.getAttribute("data-model"), b.disabled]);
            const legacy = shown(".mc-voice-box-install");
            all(".mc-voice-box-install-part").forEach((b) => b.click());
            await flush();
            report({offers, legacy, installs: requestsTo("/install").map(plain),
                    why: find(".mc-voice-box-render-reason").textContent});
        """, answers=described(runtime_missing, NOT_INSTALLED, runtime_installed=False, ready=False,
                               parts=[{"id": "runtime", "installed": False,
                                       "bytes": 4_300_000_000, "message": ""}]))

        assert found["offers"] == [
            ["Install the VibeVoice runtime (4.3 GB)", "runtime", None, False],
            ["Install VibeVoice Realtime 0.5B (2.1 GB)", "model", REALTIME, False]]
        assert found["legacy"] is False
        assert [r["body"] for r in found["installs"]] == [
            {"part": "runtime", "folder": "", "model_id": ""},
            {"part": "model", "folder": "", "model_id": REALTIME}]
        for request in found["installs"]:
            assert request["headers"]["x-model-chain-voice"] == TOKEN and request["signal"] is True
        assert found["why"] == "VibeVoice's runtime still to install."
        assert found["status"] == "Installing VibeVoice Realtime 0.5B… progress shows below."

    def test_while_an_install_runs_the_other_offers_wait(self):
        found = run("""
            await flush();
            const offers = all(".mc-voice-box-install-part").map((b) => b.disabled);
            all(".mc-voice-box-install-part").forEach((b) => b.click());
            await flush();
            const first = find(".mc-voice-box-install-part");
            advance(15000);
            await flush();
            report({offers, installs: requestsTo("/install").length,
                    kept: find(".mc-voice-box-install-part") === first,
                    progress: find(".mc-voice-box-install-progress").textContent});
        """, answers=dict(described(MODEL_7B, NOT_INSTALLED), **{"/status": {"json": dict(
            described(MODEL_7B, NOT_INSTALLED)["/status"]["json"],
            progress={"running": True, "text": "Downloading the 7B", "fraction": 0.5,
                      "failed": False, "model": "vibevoice-7b"})}}))

        assert found["offers"] == [True]
        assert found["installs"] == 0
        assert found["kept"] is True, "a poll rebuilt an offer that had not changed"
        assert found["progress"] == "Downloading the 7B (50%)"

    def test_an_install_already_running_on_the_server_is_said(self):
        found = run("""
            await flush();
            find(".mc-voice-box-install-part").click();
            await flush();
            report();
        """, answers=dict(described(MODEL_7B, NOT_INSTALLED),
                          **{"/install": {"json": {"ok": True, "already": True}}}))

        assert found["status"] == "Another install is running; press again when it has finished."

    def test_render_waits_on_the_chosen_models_own_installation(self):
        """The engine's `ready` speaks for the settings model. A render with an
        installed model is not held back by another model that is not."""
        found = run("""
            await flush();
            const seven = {disabled: find(".mc-voice-box-render").disabled,
                           why: find(".mc-voice-box-render-reason").textContent};
            choose('[data-field="model_id"]', "REALTIME_ID");
            report({seven, realtime: find(".mc-voice-box-render-reason").textContent});
        """.replace("REALTIME_ID", REALTIME), answers=described(
            MODEL_7B, NOT_INSTALLED, ready=False, message="Setup required."))

        assert found["seven"] == {"disabled": False, "why": ""}
        assert found["realtime"] == "VibeVoice Realtime 0.5B is not installed — install it below."


# --------------------------------------------------------------------------- #
# Samples made in Voice Chat, and what a render was made with
# --------------------------------------------------------------------------- #


class TestWhereThingsCameFrom:
    def test_a_sample_made_in_voice_chat_says_so(self):
        found = run("""
            await flush();
            report({rows: all(".mc-voice-box-sample").map((row) => [
                row.querySelector(".mc-voice-box-sample-title").textContent,
                row.querySelector(".mc-voice-box-sample-origin")
                    ? row.querySelector(".mc-voice-box-sample-origin").textContent : null])});
        """, answers={"/samples": {"json": {"ok": True, "samples": [
            SAMPLE, dict(SAMPLE, id="s2", title="Warm", source="voice-chat", created=2)]}}})

        assert found["rows"] == [["Ada", None], ["Warm", "made in Voice Chat"]]

    def test_a_lane_names_the_model_its_precision_its_lora_and_a_preset_voice(self):
        quantised = dict(OUTPUT, render=dict(OUTPUT["render"], model="VibeVoice 7B",
                                             precision="nf4",
                                             lora={"id": LORA["id"], "name": "Narrator",
                                                   "scale": 0.75}))
        realtime = dict(OUTPUT, id="o2", name="Take 2", created=3, render=dict(
            OUTPUT["render"], model="VibeVoice Realtime 0.5B", precision="bf16", lora=None,
            speakers=[{"n": 1, "sample_id": "", "title": "", "preset": "en-Carter_man"}]))
        found = run("await flush(); report({meta: texts('.mc-voice-box-lane-meta')});",
                    answers={"/outputs": {"json": {"ok": True, "outputs": [quantised, realtime]}}})

        assert found["meta"] == [
            "VibeVoice Realtime 0.5B · seed 42 · 10 steps · CFG 1.3 · S1 en-Carter_man · 4.5 s · GPU-a",
            "VibeVoice 7B · 4-bit (NF4) · LoRA Narrator ×0.75 · seed 42 · 10 steps · CFG 1.3 · "
            "S1 Ada · 4.5 s · GPU-a"]


# --------------------------------------------------------------------------- #
# Voice Chat's half of the audio focus
# --------------------------------------------------------------------------- #

VOICE_CHAT_HARNESS = DOM_JS + r"""
// The two hidden holders Voice Chat reads its tokens from.
function holder(id, value) {
    const box = new El("div");
    box.setAttribute("id", id);
    const field = new El("textarea");
    field.value = value;
    box.appendChild(field);
    document.body.appendChild(box);
    return box;
}
holder("mc-llm-chat-voice-key", "PAGE-TOKEN");
holder("mc-llm-chat-voice-token", "T1");

const played = [];
const stopped = [];
globalThis.AudioContext = function () {
    return {
        state: "running",
        sampleRate: 48000,
        currentTime: 0,
        destination: {},
        resume() { return Promise.resolve(); },
        createMediaStreamSource() { return {connect() {}, disconnect() {}}; },
        createScriptProcessor() { return {connect() {}, disconnect() {}, onaudioprocess: null}; },
        createGain() { return {gain: {}, connect() {}, disconnect() {}}; },
        createBuffer(channels, length, rate) {
            const data = new Float32Array(length);
            return {numberOfChannels: channels, length, sampleRate: rate, duration: length / rate,
                    getChannelData: () => data};
        },
        createBufferSource() {
            const source = {buffer: null, onended: null, connect() {}, disconnect() {},
                            start() { played.push(source); }, stop() { stopped.push(source); }};
            return source;
        },
        decodeAudioData(buffer, ok) {
            const data = new Float32Array(2400);
            ok({numberOfChannels: 1, length: 2400, sampleRate: 24000, duration: 0.1,
                getChannelData: () => data});
        },
    };
};

const requests = [];
globalThis.fetch = function (address, init) {
    requests.push({url: String(address), body: init && typeof init.body === "string" ? JSON.parse(init.body) : null});
    const json = {ok: true};
    return Promise.resolve({
        ok: true, status: 200, headers: {get: () => null},
        json: () => Promise.resolve(json), text: () => Promise.resolve(JSON.stringify(json)),
        arrayBuffer: () => Promise.resolve(new ArrayBuffer(64)),
    });
};

const loaded = [];
globalThis.onUiLoaded = (fn) => loaded.push(fn);
globalThis.onAfterUiUpdate = () => {};

@@SOURCE@@

function focusFrom(owner, kind) {
    document.dispatchEvent(new CustomEvent("mc:audio-focus", {detail: {owner, kind}}));
}
function report(extra) {
    realConsole.log(JSON.stringify(Object.assign({
        focusEvents, warnings, played: played.length, stopped: stopped.length,
        tracks: tracks.map((t) => t.stopped),
        requests: requests.map((r) => r.url.replace(/^.*model-chain\//, "")),
    }, extra || {})));
}

await (async function () {
@@SCENARIO@@
})();
"""


def run_voice_chat(scenario: str) -> dict:
    harness = (VOICE_CHAT_HARNESS
               .replace("@@SOURCE@@", VOICE_CHAT.read_text(encoding="utf-8"))
               .replace("@@SCENARIO@@", scenario)
               .replace("MICROPHONE", "true"))
    return _node(harness)


class TestVoiceChatsHalf:
    def test_speaking_announces_itself_and_voice_box_stops_it(self):
        found = run_voice_chat("""
            const speech = window.forgeAssistant.speech;
            speech.listen();
            await flush();
            const announced = focusEvents.slice();
            const playedBefore = played.length;
            focusFrom("somebody-else", "playback");
            await flush();
            const stoppedByStranger = stopped.length;
            focusFrom("voice-box", "playback");
            await flush();
            report({announced, playedBefore, stoppedByStranger});
        """)

        assert found["announced"] == [{"owner": "voice-chat", "kind": "speech"}]
        assert found["playedBefore"] == 1
        assert found["stoppedByStranger"] == 0
        assert found["stopped"] == 1
        assert found["warnings"] == []

    def test_a_microphone_announces_capture_and_voice_box_closes_it(self):
        found = run_voice_chat("""
            const speech = window.forgeAssistant.speech;
            const started = await speech.startDictation({});
            await flush();
            const during = {capturing: speech.getSpeechSnapshot().capturing,
                            tracks: tracks.map((t) => t.stopped),
                            focus: focusEvents.slice()};
            focusFrom("voice-box", "capture");
            await flush();
            report({started, during, capturing: speech.getSpeechSnapshot().capturing});
        """)

        assert found["started"] is True
        assert found["during"] == {"capturing": True, "tracks": [False],
                                   "focus": [{"owner": "voice-chat", "kind": "capture"}]}
        assert found["capturing"] is False
        assert found["tracks"] == [True]
        assert found["warnings"] == []

    def test_voice_chats_own_announcement_changes_nothing(self):
        found = run_voice_chat("""
            const speech = window.forgeAssistant.speech;
            await speech.startDictation({});
            await flush();
            focusFrom("voice-chat", "speech");
            await flush();
            report({capturing: speech.getSpeechSnapshot().capturing});
        """)

        assert found["capturing"] is True
        assert found["tracks"] == [False]

    def test_a_page_without_the_event_is_the_page_as_before(self):
        """The existing Voice Chat harness's document has no dispatchEvent and a
        no-op addEventListener; both hooks have to be inert there."""
        found = run_voice_chat("""
            const speech = window.forgeAssistant.speech;
            document.dispatchEvent = undefined;
            speech.listen();
            await flush();
            report({});
        """)

        assert found["played"] == 1
        assert found["focusEvents"] == []
        assert found["warnings"] == []
