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
    frames         queued too, and run by the scenario (``runFrames``), so "one
                   fit per frame, however many resizes" is a count;
    the geometry   an element's box is numbers a scenario sets (its width, its
                   height, where its top is); the window's height and the page's
                   scroll height likewise. There is no layout: what the
                   stylesheet makes of the numbers is test_voice_box_layout.py's,
                   in a real browser;
    styles         log every write, however it was made, which is how "the
                   height is written only when it changed" is counted;
    audio elements remember whether they were played and paused, which is how
                   "one lane at a time" and "paused for Voice Chat" are asserted;
    the microphone hands back tracks that record having been stopped.

``press`` finds only a button that is showing -- neither it nor any box it is
in hidden -- because a compact output lane keeps its buttons in the page, out
of sight, and a press on one of those is not something anybody can do.

The second harness loads ``javascript/voice_chat.js`` alone, with the two
hidden holders it reads its tokens from, to prove its half of the audio-focus
event. These run under node, which is not a Forge dependency, so they skip
without it.
"""

from __future__ import annotations

import json
import os
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

// An element's style: plain properties, the CSSOM's three methods, and a log
// of every write however it was made (`style.height = ...` or setProperty),
// which is how "written only when it changed" is counted.
function makeStyle() {
    const values = {};
    const writes = [];
    const methods = {
        setProperty(name, value) { proxy[name] = value; },
        getPropertyValue(name) { return values[name] === undefined ? "" : String(values[name]); },
        removeProperty(name) {
            const old = values[name];
            delete values[name];
            return old === undefined ? "" : String(old);
        },
    };
    const proxy = new Proxy(values, {
        get(target, key) {
            if (key === "writes") return writes;
            if (Object.prototype.hasOwnProperty.call(methods, key)) return methods[key];
            return target[key];
        },
        set(target, key, value) {
            target[key] = String(value);
            writes.push([String(key), String(value)]);
            return true;
        },
    });
    return proxy;
}

class El {
    constructor(tag) {
        this.tagName = String(tag).toUpperCase();
        this.children = [];
        this.parentNode = null;
        this.attributes = {};
        this.dataset = {};
        this.style = makeStyle();
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
    get nextElementSibling() {
        let at = this.nextSibling;
        while (at && at.tagName === "#TEXT") at = at.nextSibling;
        return at;
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
        const top = this.boxTop || 0;
        return {left: 0, top, width: this.clientWidth, height: this.clientHeight,
                right: this.clientWidth, bottom: top + this.clientHeight};
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

// The window's own events, the visual viewport's, the fonts and the two
// observers: everything the page's fit listens to, fired by the scenario.
const windowListeners = {};
globalThis.addEventListener = (type, fn) => {
    (windowListeners[type] = windowListeners[type] || []).push(fn);
};
globalThis.removeEventListener = (type, fn) => {
    windowListeners[type] = (windowListeners[type] || []).filter((f) => f !== fn);
};
function fireWindow(type) {
    (windowListeners[type] || []).slice().forEach((fn) => fn({type}));
}
const viewportListeners = {};
globalThis.visualViewport = {
    addEventListener(type, fn) { (viewportListeners[type] = viewportListeners[type] || []).push(fn); },
    removeEventListener() {},
};
function fireViewport(type) {
    (viewportListeners[type] || []).slice().forEach((fn) => fn({type}));
}
let fontsLoaded = null;
document.fonts = {ready: new Promise((resolve) => { fontsLoaded = resolve; })};
const intersections = [];
globalThis.IntersectionObserver = class {
    constructor(callback) { this.callback = callback; this.targets = []; intersections.push(this); }
    observe(target) { this.targets.push(target); }
    unobserve() {}
    disconnect() {}
};
function intersect(target, on) {
    intersections.forEach((observer) => {
        if (observer.targets.indexOf(target) !== -1) {
            observer.callback([{target, isIntersecting: on}], observer);
        }
    });
}
const resizeObserved = [];
globalThis.ResizeObserver = class {
    constructor(callback) { this.callback = callback; }
    observe(target) { resizeObserved.push(target); }
    unobserve() {}
    disconnect() {}
};
globalThis.innerHeight = 900;
globalThis.scrollY = 0;
function runFrames() {
    let ran = 0;
    for (let round = 0; round < 20 && rafs.length; round += 1) {
        const due = rafs.splice(0, rafs.length);
        due.forEach((fn) => fn(NOW));
        ran += due.length;
    }
    return ran;
}
function heights(node) {
    return node.style.writes.filter((write) => write[0] === "height").map((write) => write[1]);
}

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
        blob: () => Promise.resolve(new Blob([bytes], {type: spec.type || "audio/wav"})),
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
    // 0 is a tab that is not on screen: Gradio hides it with display:none.
    root.clientWidth = ROOT_WIDTH;
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
function shown(node) {
    for (let at = node; at && at instanceof El; at = at.parentNode) {
        if (at.hidden || at.hasAttribute("hidden")) return false;
    }
    return true;
}
function buttonNamed(label, within) {
    return (within || document).querySelectorAll("button")
        .filter((b) => b.textContent === label && shown(b))[0] || null;
}
function press(label, within) {
    const found = buttonNamed(label, within);
    if (!found) throw new Error("no button " + label + " showing");
    found.click();
    return found;
}
function labels(within) {
    return within.querySelectorAll("button").filter(shown).map((b) => b.textContent);
}
// A press on something inside a lane, travelling up to the lane the way a
// real one does (the harness's own click() does not bubble).
function tap(node, extra) {
    node.dispatchEvent(Object.assign({type: "click", bubbles: true, clientX: 0, clientY: 0,
                                      preventDefault() {}}, extra || {}));
}
// A poll now: the page refreshes its status whenever the tab is shown again.
async function poll() {
    document.visibilityState = "visible";
    document.dispatchEvent(new Event("visibilitychange"));
    await flush();
}
function line() {
    const status = find(".mc-voice-box-status");
    return {text: status.textContent, state: status.getAttribute("data-state"),
            kind: status.getAttribute("data-kind"),
            buttons: labels(find(".mc-voice-box-status-line"))};
}
function type(node, value) {
    node.value = value;
    node.dispatchEvent({type: "input"});
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
INFOTEXT = ("Speaker 1: Hello there.\nSteps: 10, CFG scale: 1.3, Seed: 42, Model: vibevoice-7b, "
            "Speaker 1: Ada, Length: 4.5 s, Render time: 3.0 s")
OUTPUT = {"id": "o1", "name": "Take 1", "pipeline_id": "p1", "seconds": 4.5, "rate": 24000,
          "peaks": [0.3] * 240, "loop": False, "created": 1600695200, "format": "mp3",
          "infotext": INFOTEXT,
          "render": {"model_id": "vibevoice-7b", "card": "GPU-a", "seed": 42, "seed_drawn": True,
                     "steps": 10, "cfg_scale": 1.3, "max_new_tokens": None,
                     "speakers": [{"n": 1, "sample_id": "s1", "title": "Ada"}],
                     "prompt": "Speaker 1: Hello there.", "render_seconds": 3.0,
                     "peak_bytes": 1, "sections": 1,
                     "configuration": {"id": "c1", "name": "Default", "model_id": "vibevoice-7b",
                                       "card_uuid": "GPU-a", "steps": 10, "cfg_scale": 1.3,
                                       "seed": 42, "max_new_tokens": None,
                                       "speakers": {"1": "s1"}}}}
"""An output as the server hands one out now: an MP3, its infotext, the seed it
used and the configuration it was made with. 1600695200 is 21 Sep 2020 13:33 UTC."""
OLD_OUTPUT = {"id": "o0", "name": "Old take", "pipeline_id": "p1", "seconds": 3.0, "rate": 24000,
              "peaks": [0.1] * 240, "loop": False, "created": 1600000000, "format": "wav",
              "infotext": "Speaker 1: Before.\nSteps: 12, CFG scale: 2, Model: vibevoice-7b",
              "render": {"model_id": "vibevoice-7b", "card": "GPU-b", "seed": None, "steps": 12,
                         "cfg_scale": 2.0, "max_new_tokens": 900,
                         "speakers": [{"n": 1, "sample_id": "s1", "title": "Ada"}],
                         "prompt": "Speaker 1: Before.", "render_seconds": 2.0,
                         "peak_bytes": 1, "sections": 1}}
"""One made before seeds and configurations were recorded, with a blank seed."""
JOB = {"id": "j1", "name": "Pipeline 1 2", "pipeline_id": "p1", "phase": "queued", "reason": "",
       "warning": "", "progress": {}, "output_id": "", "created": 3, "started": None,
       "ended": None, "card": "GPU-a", "live": True, "elapsed": None, "seed": 42,
       "seed_drawn": True}


def fnv_hue(text: str) -> int:
    """The page's hue for a sample id, worked out independently: FNV-1a over the
    UTF-16 code units, 32 bits, modulo 360."""
    value = 0x811C9DC5
    units = text.encode("utf-16-le")
    for index in range(0, len(units), 2):
        value ^= int.from_bytes(units[index:index + 2], "little")
        value = (value * 0x01000193) & 0xFFFFFFFF
    return value % 360
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


def _node(harness: str) -> dict:
    # Dates on the page are local time; the scenarios read them in UTC.
    environment = dict(os.environ, TZ="UTC")
    with tempfile.TemporaryDirectory() as room:
        entry = pathlib.Path(room) / "scenario.mjs"
        entry.write_text(harness, encoding="utf-8")
        result = subprocess.run(["node", str(entry)], capture_output=True, text=True, timeout=60,
                                env=environment)
    assert result.returncode == 0, result.stderr
    lines = [line for line in result.stdout.splitlines() if line.startswith("{")]
    assert lines, result.stdout or "the harness reported nothing"
    return json.loads(lines[-1])


def run(scenario: str, *, answers: dict | None = None, held=(), root: bool = True,
        decode_seconds: float = 10, decode_works: bool = True, offline: bool = False,
        storage_throws: bool = False, microphone: bool = True, root_width: int = 240) -> dict:
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
               .replace("ROOT_WIDTH", json.dumps(root_width))
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
                     "ROOT_PRESENT", "ROOT_WIDTH", "DECODE_SECONDS", "DECODE_WORKS",
                     "STORAGE_THROWS", "MICROPHONE_FLAG"):
            assert word not in source, (path.name, word)


# --------------------------------------------------------------------------- #
# Boot
# --------------------------------------------------------------------------- #


class TestBoot:
    def test_it_draws_the_pipeline_bar_the_four_stages_in_order_and_the_outputs_header(self):
        """Render and the status line moved from a footer into the Outputs
        stage's header, and the footer went."""
        found = run("""
            await flush();
            const stages = find(".mc-voice-box-stages");
            const head = find(".mc-voice-box-stage-outputs .mc-voice-box-stage-head");
            report({
                title: find(".mc-voice-box-title").textContent,
                bar: texts(".mc-voice-box-pipeline-bar button"),
                pipelines: find(".mc-voice-box-pipeline-select").options.map((o) => o.textContent),
                stages: texts(".mc-voice-box-stage-title"),
                row: stages.children.map((child) => child.className.split(" ")[0]),
                render: !!head.querySelector(".mc-voice-box-render"),
                line: !!head.querySelector(".mc-voice-box-status-line .mc-voice-box-status"),
                footer: !!find(".mc-voice-box-footer"),
                rootChildren: find("#mc-voice-box").children.map((child) => child.className),
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
        assert found["line"] is True
        assert found["footer"] is False
        assert found["rootChildren"] == ["mc-voice-box-head", "mc-voice-box-stages"]
        assert found["status"] == "Ready"
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
            lane.click();
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
        lane.click();
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
            lanes[0].click();
            press("Play", lanes[0]);
            await flush();
            lanes[1].click();
            press("Play", lanes[1]);
            await flush();
            report({first: lanes[0].querySelector("audio").paused,
                    second: lanes[1].querySelector("audio").paused,
                    names: texts(".mc-voice-box-lane-name")});
        """, answers={"/outputs": {"json": {"ok": True, "outputs": [
            OUTPUT, dict(OUTPUT, id="o2", name="Take 2", created=OUTPUT["created"] + 1)]}}})

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
            find(".mc-voice-box-lane").click();
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
            find(".mc-voice-box-lane").click();
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
            lane.click();
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
            const during = line().text;
            advance(1000);
            await flush(10);
            const done = {delay: globalThis.mcVoiceBox.state().poll.delay, timers: pendingDelays(),
                          outputs: requestsTo("/outputs").length - outputsBefore,
                          prompts: requestsTo("/prompts").length};
            report({idle, live, done, during, after: line().text});
        """.replace("LIVE_JOB", json.dumps(JOB)))

        assert found["idle"]["delay"] == 15000 and 15000 in found["idle"]["timers"]
        assert found["live"]["delay"] == 1000 and 1000 in found["live"]["timers"]
        assert 15000 not in found["live"]["timers"]
        assert found["done"]["delay"] == 15000
        assert found["done"]["outputs"] == 1
        assert found["done"]["prompts"] == 2
        assert found["during"] == "1 queued"
        assert found["after"] == "Ready"

    def test_somebody_elses_job_does_not_speed_the_poll_up(self):
        found = run("await flush(); report({line: line()});",
                    answers={"/status": {"json": status_with(jobs=[JOB])}})

        assert found["state"]["poll"]["delay"] == 15000
        assert found["line"]["text"] == "1 queued"
        assert found["line"]["buttons"] == ["Clear queue"]

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
        """The running job's Cancel is the status line's now."""
        found = run("""
            await flush();
            press("Render");
            await flush();
            press("Cancel", find(".mc-voice-box-status-line"));
            await flush();
            report({cancel: requestsTo("/jobs/cancel").map(plain)});
        """, answers={"/status": {"json": status_with(jobs=[dict(JOB, phase="rendering")])}})

        assert [c["body"] for c in found["cancel"]] == [{"id": "j1"}]


# --------------------------------------------------------------------------- #
# Outputs
# --------------------------------------------------------------------------- #


class TestOutputs:
    def test_save_opens_the_folder_dialog_once_when_there_is_no_folder(self):
        found = run("""
            await flush();
            find(".mc-voice-box-lane").click();
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
            find(".mc-voice-box-lane").click();
            press("Save", find(".mc-voice-box-lane"));
            await flush();
            report({order: requests.map((r) => r.route).filter((r) => r === "/outputs/save" || r === "/settings/folder")});
        """)

        assert found["order"] == ["/outputs/save"]

    def test_download_fetches_with_the_token_and_clicks_a_download_link(self):
        found = run("""
            await flush();
            find(".mc-voice-box-lane").click();
            press("Download", find(".mc-voice-box-lane"));
            await flush();
            report({audio: requestsTo("/outputs/audio").map(plain)});
        """)

        request = found["audio"][0]
        assert request["url"].endswith("/outputs/audio?id=o1&download=1")
        assert request["method"] == "GET"
        assert request["headers"]["x-model-chain-voice"] == TOKEN
        assert found["linkClicks"] == [{"href": "blob:test-1", "download": "Take 1.mp3"}]

    def test_loop_is_persisted_and_shown(self):
        found = run("""
            await flush();
            const lane = find(".mc-voice-box-lane");
            lane.click();
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
            find(".mc-voice-box-lane").click();
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
        """On the selected lane; on a compact one the press only selects it
        (TestTheLanes)."""
        found = run("""
            await flush();
            const lane = find(".mc-voice-box-lane");
            const audio = lane.querySelector("audio");
            audio.duration = 4.5;
            lane.click();
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
            lane.click();
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
        """The reason is the Render button's title, and the status line's when
        nothing ahead of it applies."""
        found = run("""
            await flush();
            const reasons = {};
            const reason = () => ({disabled: find(".mc-voice-box-render").disabled,
                                   title: find(".mc-voice-box-render").getAttribute("title"),
                                   why: line().text});
            reasons.ready = reason();
            type(find(".mc-voice-box-prompt"), "");
            reasons.empty = reason();
            type(find(".mc-voice-box-prompt"), "Speaker 2: hi");
            reasons.noSample = reason();
            type(find(".mc-voice-box-prompt"), "Speaker 7: x");
            reasons.tooMany = reason();
            report({reasons});
        """)

        assert found["reasons"]["ready"] == {"disabled": False, "title": "Render this pipeline",
                                             "why": "Ready"}
        assert found["reasons"]["empty"] == {"disabled": True, "title": "Write a script first.",
                                             "why": "Write a script first."}
        assert found["reasons"]["noSample"] == {"disabled": True, "title": "Speaker 2 has no sample.",
                                                "why": "Speaker 2 has no sample."}
        assert found["reasons"]["tooMany"]["why"] == "Speakers are numbered 1 to 4; the prompt names Speaker 7."

    def test_an_engine_that_is_not_installed_offers_install(self):
        """The install's progress is the status line's first state."""
        found = run("""
            await flush();
            const install = find(".mc-voice-box-install");
            const before = {hidden: install.hidden, label: install.textContent,
                            reason: find(".mc-voice-box-render").getAttribute("title"),
                            disabled: find(".mc-voice-box-render").disabled};
            install.click();
            await flush();
            report({before, install: requestsTo("/install").map(plain), line: line()});
        """, answers={"/status": {"json": status_with(
            engine={"ready": False, "supported": True, "message": "VibeVoice is not installed.",
                    "model_id": "vibevoice-7b", "download_bytes": 19_000_000_000},
            progress={"running": True, "text": "Downloading shard 3 of 8", "fraction": 0.375,
                      "failed": False, "model": "vibevoice-7b"})}})

        assert found["before"] == {"hidden": False, "label": "Install VibeVoice (19.0 GB)",
                                   "reason": "VibeVoice is not installed.", "disabled": True}
        assert [r["body"] for r in found["install"]] == [{"part": "", "folder": ""}]
        assert found["line"]["text"] == "Downloading shard 3 of 8 — 38 %"
        assert found["line"]["state"] == "install"

    def test_an_installed_engine_hides_the_install_button(self):
        found = run("await flush(); report({hidden: find('.mc-voice-box-install').hidden});")

        assert found["hidden"] is True

    def test_render_sends_the_saved_configuration_by_id(self):
        """The status line answers the press with the job it made."""
        found = run("""
            await flush();
            press("Render");
            await flush();
            report({render: requestsTo("/render").map(plain)});
        """, answers={"/status": {"sequence": [{"json": STATUS},
                                               {"json": status_with(jobs=[JOB])}]}})

        assert [r["body"] for r in found["render"]] == [
            {"pipeline_id": "p1", "prompt": "Speaker 1: Hello there.", "configuration_id": "c1", "name": ""}]
        assert found["status"] == "1 queued"

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
        """A warm VibeVoice is the status line's, with the Unload the footer had
        (through the same route); what the cards are doing in full is the
        line's tooltip."""
        found = run("""
            await flush();
            const shown = line();
            const title = find(".mc-voice-box-status").getAttribute("title");
            press("Unload", find(".mc-voice-box-status-line"));
            await flush();
            report({shown, title, runtime: requestsTo("/runtime").map(plain)});
        """, answers={"/status": {"json": status_with(
            turns=[{"card": "RTX 3090", "uuid": "GPU-a", "active": "",
                    "queued": [{"label": "image"}], "warm": "vibevoice", "parked": True,
                    "lease": False}],
            runtime={"cards": {"abc": {"running": True, "loaded": True, "rendering": False, "job": "",
                                       "resident_bytes": 19_300_000_000, "peak_bytes": 0,
                                       "device_name": "RTX 3090", "uuid": "GPU-a"}},
                     "last_error": ""})}})

        assert found["shown"]["text"] == "VibeVoice warm on RTX 3090"
        assert found["shown"]["buttons"] == ["Unload"]
        assert found["title"] == ("VibeVoice warm on RTX 3090\n"
                                  "RTX 3090: idle, 1 waiting, vibevoice warm, image model parked · "
                                  "VibeVoice loaded on RTX 3090 (19.3 GB)")
        assert [r["body"] for r in found["runtime"]] == [{"action": "unload", "card_uuid": "GPU-a"}]
        assert found["status"] == "VibeVoice was unloaded from RTX 3090."


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
# Fitting the window
# --------------------------------------------------------------------------- #


class TestFittingTheWindow:
    """The numbers the fit works from and what it writes. Whether the
    stylesheet then fits the stages into that height is
    test_voice_box_layout.py's question, in a real browser."""

    def test_the_layout_is_columns_from_900_px_and_a_stack_below(self):
        found = run("""
            await flush();
            const root = find("#mc-voice-box");
            const seen = [];
            for (const width of [1440, 900, 899, 390, 1200]) {
                root.clientWidth = width;
                fireWindow("resize");
                runFrames();
                seen.push([width, root.getAttribute("data-layout")]);
            }
            report({seen});
        """)

        assert found["seen"] == [[1440, "columns"], [900, "columns"], [899, "stack"],
                                 [390, "stack"], [1200, "columns"]]
        assert found["state"]["layout"]["mode"] == "columns"

    def test_the_root_takes_the_windows_height_less_what_is_above_and_below_it(self):
        """A 160 px header above the root and a 60 px footer after it: the root
        is the window's height less both, in pixels on the root itself, never
        less than 420 px -- and a scrolled page changes nothing, because the
        top that counts is the root's on the document."""
        found = run("""
            await flush();
            const root = find("#mc-voice-box");
            const footer = document.createElement("div");
            footer.clientHeight = 60;
            document.body.appendChild(footer);
            root.boxTop = 160;
            root.clientWidth = 1440;
            const seen = [];
            for (const height of [900, 700, 500]) {
                globalThis.innerHeight = height;
                fireWindow("resize");
                runFrames();
                seen.push(root.style.height);
            }
            globalThis.innerHeight = 900;
            globalThis.scrollY = 300;
            root.boxTop = -140;
            fireWindow("resize");
            runFrames();
            seen.push(root.style.height);
            report({seen});
        """)

        assert found["seen"] == ["680px", "480px", "420px", "680px"]

    def test_the_height_is_written_only_when_it_changes(self):
        found = run("""
            await flush();
            const root = find("#mc-voice-box");
            root.clientWidth = 1440;
            globalThis.innerHeight = 800;
            fireWindow("resize");
            runFrames();
            const first = heights(root).slice();
            fireWindow("resize");
            fireViewport("resize");
            fireWindow("orientationchange");
            runFrames();
            fireWindow("resize");
            runFrames();
            const same = heights(root).slice();
            globalThis.innerHeight = 810;
            fireViewport("resize");
            runFrames();
            report({first, same, after: heights(root)});
        """)

        # The boot's own fit wrote 900 px for the harness's window; then one
        # write for each height that changed, and none for the events that
        # changed nothing.
        assert found["first"] == ["900px", "800px"]
        assert found["same"] == ["900px", "800px"]
        assert found["after"] == ["900px", "800px", "810px"]
        assert found["state"]["layout"]["writes"] == 3

    def test_each_event_that_changes_the_room_fits_it_on_its_own(self):
        """The window's resize, the visual viewport's (a phone's address bar)
        and an orientation change: each alone is enough."""
        found = run("""
            await flush();
            runFrames();
            const root = find("#mc-voice-box");
            const seen = [];
            for (const [height, fire] of [[700, () => fireWindow("resize")],
                                          [650, () => fireViewport("resize")],
                                          [600, () => fireWindow("orientationchange")]]) {
                globalThis.innerHeight = height;
                fire();
                runFrames();
                seen.push(root.style.height);
            }
            report({seen});
        """)

        assert found["seen"] == ["700px", "650px", "600px"]

    def test_many_events_in_one_frame_are_one_fit(self):
        found = run("""
            await flush();
            runFrames();
            const root = find("#mc-voice-box");
            root.clientWidth = 1440;
            globalThis.innerHeight = 640;
            for (let index = 0; index < 5; index += 1) fireWindow("resize");
            fireViewport("resize");
            fireWindow("orientationchange");
            const queued = rafs.length;
            const ran = runFrames();
            report({queued, ran, heights: heights(root)});
        """)

        assert found["queued"] == 1
        assert found["ran"] == 1
        assert found["heights"] == ["900px", "640px"]

    def test_a_tab_that_is_not_on_screen_is_measured_when_it_comes_into_view(self):
        """Gradio hides a tab with display:none, so its root has no width and
        nothing is measured. The IntersectionObserver on the root hears it come
        into view; no ResizeObserver watches the root, the box the page resizes."""
        found = run("""
            await flush();
            runFrames();
            const root = find("#mc-voice-box");
            const hidden = {layout: root.getAttribute("data-layout"), heights: heights(root).slice()};
            root.clientWidth = 1280;
            globalThis.innerHeight = 720;
            intersect(root, false);
            const unseen = rafs.length;
            intersect(root, true);
            runFrames();
            report({hidden, unseen, layout: root.getAttribute("data-layout"), heights: heights(root),
                    observed: intersections.map((o) => o.targets.map((t) => t.id)),
                    resized: resizeObserved.length});
        """, root_width=0)

        assert found["hidden"] == {"layout": None, "heights": []}
        assert found["unseen"] == 0
        assert found["layout"] == "columns"
        assert found["heights"] == ["720px"]
        assert found["observed"] == [["mc-voice-box"]]
        assert found["resized"] == 0

    def test_the_fonts_arriving_fit_it_again(self):
        found = run("""
            await flush();
            runFrames();
            const root = find("#mc-voice-box");
            globalThis.innerHeight = 760;
            fontsLoaded();
            await flush();
            const queued = rafs.length;
            runFrames();
            report({queued, heights: heights(root)});
        """)

        assert found["queued"] == 1
        assert found["heights"] == ["900px", "760px"]

    def test_what_the_sum_missed_is_taken_off_and_kept_while_the_window_keeps_its_shape(self):
        """The page scrolling is the fact the sum is checked against: whatever
        it missed comes off at once, and the height does not swing back on the
        next event; a new window shape starts from the sum again."""
        found = run("""
            await flush();
            runFrames();
            const root = find("#mc-voice-box");
            root.boxTop = 160;
            root.clientWidth = 1440;
            // A page 25 px longer than the sum knows about.
            Object.defineProperty(document, "scrollHeight", {configurable: true,
                get: () => 160 + parseInt(root.style.height, 10) + 25});
            Object.defineProperty(document, "clientHeight", {configurable: true,
                get: () => globalThis.innerHeight});
            globalThis.innerHeight = 900;
            fireWindow("resize");
            runFrames();
            const first = heights(root).slice();
            fireWindow("resize");
            runFrames();
            fireWindow("resize");
            runFrames();
            const steady = heights(root).slice();
            globalThis.innerHeight = 1000;
            fireWindow("resize");
            runFrames();
            report({first, steady, after: heights(root)});
        """)

        assert found["first"] == ["900px", "740px", "715px"]
        assert found["steady"] == found["first"]
        assert found["after"] == ["900px", "740px", "715px", "840px", "815px"]

    def test_the_waveforms_are_drawn_again_when_the_width_moves_and_only_then(self):
        found = run("""
            await flush();
            runFrames();
            const wave = find(".mc-voice-box-lane-wave");
            const root = find("#mc-voice-box");
            const before = wave.draws;
            globalThis.innerHeight = 700;
            fireWindow("resize");
            runFrames();
            const heightOnly = wave.draws;
            root.clientWidth = 1440;
            fireWindow("resize");
            runFrames();
            report({before, heightOnly, widthMoved: wave.draws});
        """)

        assert found["heightOnly"] == found["before"]
        assert found["widthMoved"] > found["heightOnly"]


# --------------------------------------------------------------------------- #
# Output lanes: compact until selected
# --------------------------------------------------------------------------- #

TWO_OUTPUTS = {"/outputs": {"json": {"ok": True, "outputs": [
    OUTPUT, dict(OUTPUT, id="o2", name="Take 2", created=OUTPUT["created"] + 60)]}}}


class TestTheLanes:
    def test_a_lane_is_its_waveform_and_one_line_until_it_is_selected(self):
        found = run("""
            await flush();
            const lane = find(".mc-voice-box-lane");
            const date = lane.querySelector(".mc-voice-box-lane-date");
            const compact = {
                expanded: lane.getAttribute("aria-expanded"),
                tabindex: lane.getAttribute("tabindex"),
                line: lane.querySelector(".mc-voice-box-lane-head").children.map((n) => n.textContent),
                datetime: date.getAttribute("datetime"),
                wave: shown(lane.querySelector(".mc-voice-box-lane-wave")),
                buttons: labels(lane),
                meta: shown(lane.querySelector(".mc-voice-box-lane-meta")),
                infotext: shown(lane.querySelector(".mc-voice-box-infotext")),
            };
            lane.click();
            const open = {
                expanded: lane.getAttribute("aria-expanded"),
                buttons: labels(lane),
                meta: shown(lane.querySelector(".mc-voice-box-lane-meta")),
                infotext: shown(lane.querySelector(".mc-voice-box-infotext"))
                    ? lane.querySelector(".mc-voice-box-infotext").textContent : null,
            };
            report({compact, open});
        """)

        assert found["compact"] == {"expanded": "false", "tabindex": "0",
                                    "line": ["21 Sep 2020 13:33", "Take 1"],
                                    "datetime": "2020-09-21T13:33:20.000Z", "wave": True,
                                    "buttons": [], "meta": False, "infotext": False}
        assert found["open"] == {"expanded": "true",
                                 "buttons": ["Play", "Loop", "Trim to sample", "Save", "Download",
                                             "Delete", "Copy", "Use seed", "Reuse settings"],
                                 "meta": True, "infotext": INFOTEXT}
        assert found["state"]["selectedOutput"] == "o1"

    def test_an_empty_pipeline_says_so_under_the_header(self):
        """An empty list takes no room, so its sentence is not pushed to the
        foot of the stage; the first output brings the list back."""
        found = run("""
            await flush();
            const empty = {list: shown(find(".mc-voice-box-lanes")),
                           sentence: shown(find(".mc-voice-box-stage-outputs .mc-voice-box-empty")),
                           text: find(".mc-voice-box-stage-outputs .mc-voice-box-empty").textContent};
            answers["/outputs"] = {json: {ok: true, outputs: [FIRST]}};
            press("New");
            await flush();
            report({empty, list: shown(find(".mc-voice-box-lanes")),
                    sentence: shown(find(".mc-voice-box-stage-outputs .mc-voice-box-empty"))});
        """.replace("FIRST", json.dumps(dict(OUTPUT, pipeline_id="p2"))),
            answers={"/outputs": {"json": {"ok": True, "outputs": []}},
                     "/pipelines/new": {"json": {"ok": True, "pipeline": dict(PIPELINE, id="p2", name="Two")}},
                     "/pipelines": {"sequence": [
                         {"json": {"ok": True, "pipelines": [PIPELINE]}},
                         {"json": {"ok": True, "pipelines": [PIPELINE, dict(PIPELINE, id="p2", name="Two")]}}]}})

        assert found["empty"] == {"list": False, "sentence": True,
                                  "text": "No renders in this pipeline yet."}
        assert found["list"] is True
        assert found["sentence"] is False

    def test_a_date_of_this_year_leaves_the_year_out(self):
        import re
        import time

        found = run("""
            await flush();
            report({date: find(".mc-voice-box-lane-date").textContent});
        """, answers={"/outputs": {"json": {"ok": True, "outputs": [
            dict(OUTPUT, created=time.time())]}}})

        assert re.fullmatch(r"\d{1,2} [A-Z][a-z]{2} \d{2}:\d{2}", found["date"]), found["date"]

    def test_selecting_another_lane_collapses_the_one_before(self):
        found = run("""
            await flush();
            const lanes = all(".mc-voice-box-lane");
            const expanded = () => lanes.map((lane) => lane.getAttribute("aria-expanded"));
            lanes[0].click();
            const first = expanded();
            lanes[1].click();
            const second = expanded();
            lanes[1].click();
            report({first, second, again: expanded(), names: texts(".mc-voice-box-lane-name"),
                    details: all(".mc-voice-box-lane-details").map(shown)});
        """, answers=TWO_OUTPUTS)

        assert found["names"] == ["Take 2", "Take 1"]
        assert found["first"] == ["true", "false"]
        assert found["second"] == ["false", "true"]
        assert found["again"] == ["false", "true"]
        assert found["details"] == [False, True]
        assert found["state"]["selectedOutput"] == "o1"

    def test_a_drag_is_not_a_press(self):
        found = run("""
            await flush();
            const lane = find(".mc-voice-box-lane");
            lane.dispatchEvent({type: "pointerdown", clientX: 10, clientY: 10});
            tap(lane, {clientX: 60, clientY: 12});
            const dragged = lane.getAttribute("aria-expanded");
            lane.dispatchEvent({type: "pointerdown", clientX: 10, clientY: 10});
            tap(lane, {clientX: 14, clientY: 13});
            report({dragged, pressed: lane.getAttribute("aria-expanded")});
        """)

        assert found["dragged"] == "false"
        assert found["pressed"] == "true"

    def test_enter_and_space_on_a_lane_select_it(self):
        found = run("""
            await flush();
            const lanes = all(".mc-voice-box-lane");
            const expanded = () => lanes.map((lane) => lane.getAttribute("aria-expanded"));
            const key = (lane, name, target) => {
                const event = {type: "keydown", key: name, target: target || lane, prevented: false,
                               preventDefault() { this.prevented = true; }};
                lane.dispatchEvent(event);
                return event.prevented;
            };
            const other = {prevented: key(lanes[0], "a"), expanded: expanded()};
            const enter = {prevented: key(lanes[0], "Enter"), expanded: expanded()};
            const inner = {prevented: key(lanes[1], " ", lanes[1].querySelector(".mc-voice-box-lane-name")),
                           expanded: expanded()};
            const space = {prevented: key(lanes[1], " "), expanded: expanded()};
            report({other, enter, inner, space});
        """, answers=TWO_OUTPUTS)

        assert found["other"] == {"prevented": False, "expanded": ["false", "false"]}
        assert found["enter"] == {"prevented": True, "expanded": ["true", "false"]}
        assert found["inner"] == {"prevented": False, "expanded": ["true", "false"]}
        assert found["space"] == {"prevented": True, "expanded": ["false", "true"]}

    def test_a_rename_box_takes_its_own_presses(self):
        """A double click renames a compact lane without opening it, and a
        press in the box is the box's."""
        found = run("""
            await flush();
            const lane = find(".mc-voice-box-lane");
            lane.querySelector(".mc-voice-box-lane-name").dispatchEvent({type: "dblclick"});
            const box = find(".mc-voice-box-rename");
            tap(box);
            const expanded = lane.getAttribute("aria-expanded");
            box.value = "Intro";
            box.dispatchEvent({type: "keydown", key: "Enter"});
            await flush();
            report({expanded, rename: requestsTo("/outputs/rename").map((r) => r.body)});
        """)

        assert found["expanded"] == "false"
        assert found["rename"] == [{"id": "o1", "name": "Intro"}]

    def test_a_press_on_a_compact_lanes_waveform_selects_it_without_seeking(self):
        found = run("""
            await flush();
            const lane = find(".mc-voice-box-lane");
            const audio = lane.querySelector("audio");
            audio.duration = 4.5;
            const wave = lane.querySelector(".mc-voice-box-lane-wave");
            wave.dispatchEvent({type: "pointerdown", clientX: 120, pointerId: 1, preventDefault() {}});
            const at = audio.currentTime;
            tap(wave, {clientX: 120, clientY: 0});
            report({at, expanded: lane.getAttribute("aria-expanded")});
        """)

        assert found["at"] == 0
        assert found["expanded"] == "true"

    def test_the_selection_survives_a_refresh_of_the_list(self):
        newer = dict(OUTPUT, id="o3", name="Take 3", created=OUTPUT["created"] + 99)
        found = run("""
            await flush();
            const lane = find(".mc-voice-box-lane");
            lane.click();
            answers["/outputs"] = {json: {ok: true, outputs: [NEWER, Object.assign({}, FIRST, {name: "Intro"})]}};
            lane.querySelector(".mc-voice-box-lane-name").dispatchEvent({type: "dblclick"});
            const box = find(".mc-voice-box-rename");
            box.value = "Intro";
            box.dispatchEvent({type: "keydown", key: "Enter"});
            await flush();
            const rows = all(".mc-voice-box-lane");
            report({names: texts(".mc-voice-box-lane-name"),
                    expanded: rows.map((row) => row.getAttribute("aria-expanded")),
                    same: rows[1] === lane, details: shown(lane.querySelector(".mc-voice-box-lane-details"))});
        """.replace("NEWER", json.dumps(newer)).replace("FIRST", json.dumps(OUTPUT)))

        assert found["names"] == ["Take 3", "Intro"]
        assert found["expanded"] == ["false", "true"]
        assert found["same"] is True
        assert found["details"] is True
        assert found["state"]["selectedOutput"] == "o1"

    def test_a_render_this_page_started_is_selected_when_it_finishes(self):
        """Its output opens; somebody else's finishing opens nothing."""
        made = dict(OUTPUT, id="o2", name="Pipeline 1 2", created=OUTPUT["created"] + 60)
        found = run("""
            await flush();
            const base = answers["/status"].json;
            const set = (jobs) => { answers["/status"] = {json: Object.assign({}, base, {jobs})}; };
            // Somebody else's render, running and then done, selects nothing.
            set([Object.assign({}, OTHER, {phase: "rendering"})]);
            await poll();
            set([Object.assign({}, OTHER, {phase: "done", live: false, output_id: "o1"})]);
            await poll();
            const others = globalThis.mcVoiceBox.state().selectedOutput;
            // This page's own.
            set([LIVE]);
            press("Render");
            await flush();
            set([Object.assign({}, LIVE, {phase: "done", live: false, output_id: "o2"})]);
            answers["/outputs"] = {json: {ok: true, outputs: [MADE, FIRST]}};
            advance(1000);
            await flush(10);
            report({others, rows: all(".mc-voice-box-lane").map((row) =>
                [row.getAttribute("data-id"), row.getAttribute("aria-expanded")])});
        """.replace("OTHER", json.dumps(dict(JOB, id="j9", name="Theirs")))
           .replace("LIVE", json.dumps(JOB)).replace("MADE", json.dumps(made))
           .replace("FIRST", json.dumps(OUTPUT)))

        assert found["others"] == ""
        assert found["rows"] == [["o2", "true"], ["o1", "false"]]
        assert found["state"]["selectedOutput"] == "o2"


# --------------------------------------------------------------------------- #
# A render's infotext, its seed and its settings
# --------------------------------------------------------------------------- #


class TestTheInfotext:
    def test_copy_puts_the_infotext_on_the_clipboard(self):
        found = run("""
            await flush();
            const copied = [];
            navigator.clipboard = {writeText(text) { copied.push(text); return Promise.resolve(); }};
            const lane = find(".mc-voice-box-lane");
            lane.click();
            press("Copy", lane);
            await flush();
            report({copied});
        """)

        assert found["copied"] == [INFOTEXT]
        assert found["status"] == "Copied the infotext of “Take 1”."

    def test_a_clipboard_that_refuses_leaves_the_text_selected_to_copy_by_hand(self):
        found = run("""
            await flush();
            const selected = [];
            globalThis.getSelection = () => ({selectAllChildren(node) { selected.push(node.className); }});
            navigator.clipboard = {writeText() { return Promise.reject(new Error("NotAllowedError")); }};
            const lane = find(".mc-voice-box-lane");
            lane.click();
            press("Copy", lane);
            await flush();
            const refused = {selected: selected.slice(), line: line()};
            delete navigator.clipboard;
            press("Copy", lane);
            await flush();
            report({refused, absent: selected});
        """)

        assert found["refused"]["selected"] == ["mc-voice-box-infotext"]
        assert found["refused"]["line"]["kind"] == "warn"
        assert found["refused"]["line"]["text"] == (
            "The browser would not copy it, so the infotext is selected: copy it from there.")
        assert found["absent"] == ["mc-voice-box-infotext", "mc-voice-box-infotext"]
        assert found["warnings"] == []

    def test_a_render_without_an_infotext_offers_no_copy(self):
        found = run("""
            await flush();
            const lane = find(".mc-voice-box-lane");
            lane.click();
            report({buttons: labels(lane), shown: shown(lane.querySelector(".mc-voice-box-infotext"))});
        """, answers={"/outputs": {"json": {"ok": True, "outputs": [dict(OUTPUT, infotext="")]}}})

        assert found["buttons"] == ["Play", "Loop", "Trim to sample", "Save", "Download", "Delete",
                                    "Use seed", "Reuse settings"]
        assert found["shown"] is False


class TestUseSeed:
    def test_use_seed_puts_the_seed_the_render_used_into_the_configuration(self):
        found = run("""
            await flush();
            const lane = find(".mc-voice-box-lane");
            lane.click();
            const before = {save: find(".mc-voice-box-configuration-save").textContent,
                            seed: find('[data-field="seed"]').value};
            press("Use seed", lane);
            report({before, save: find(".mc-voice-box-configuration-save").textContent,
                    seed: find('[data-field="seed"]').value,
                    title: lane.querySelector(".mc-voice-box-lane-seed").getAttribute("title")});
        """)

        assert found["before"] == {"save": "Save", "seed": ""}
        assert found["save"] == "Save •"
        assert found["seed"] == "42"
        assert found["state"]["working"]["seed"] == 42
        assert found["state"]["dirty"]["configuration"] is True
        assert found["title"] == "Put seed 42 in the configuration"
        assert found["status"] == "Seed 42 is in the configuration; save it to keep it."

    def test_a_render_made_before_seeds_were_recorded_has_none_to_give(self):
        found = run("""
            await flush();
            const lane = find(".mc-voice-box-lane");
            lane.click();
            const seed = lane.querySelector(".mc-voice-box-lane-seed");
            seed.click();
            report({disabled: seed.disabled, title: seed.getAttribute("title"),
                    working: globalThis.mcVoiceBox.state().working.seed,
                    save: find(".mc-voice-box-configuration-save").textContent});
        """, answers={"/outputs": {"json": {"ok": True, "outputs": [OLD_OUTPUT]}}})

        assert found["disabled"] is True
        assert found["title"] == "This render was made before seeds were recorded."
        assert found["working"] is None
        assert found["save"] == "Save"


class TestReuseSettings:
    REUSE = """
        await flush();
        const lane = find(".mc-voice-box-lane");
        lane.click();
        press("Reuse settings", lane);
        await flush();
    """

    def test_it_selects_the_configuration_it_came_from_and_applies_the_render(self):
        """The prompt goes in as if typed (and is saved to the pipeline the
        usual way, a second later); the configuration it was made with is
        chosen, and the render's values are unsaved edits over it."""
        render = dict(OUTPUT["render"], prompt="Speaker 1: Hi.\nSpeaker 2: Yo.",
                      speakers=[{"n": 1, "sample_id": "s1", "title": "Ada"},
                                {"n": 2, "sample_id": "s2", "title": "Bo"}],
                      configuration=dict(OUTPUT["render"]["configuration"], steps=20, cfg_scale=1.5,
                                         seed=42, max_new_tokens=900,
                                         speakers={"1": "s1", "2": "s2"}))
        other = dict(CONFIGURATION, id="c2", name="Other", steps=30, cfg_scale=2.5,
                     speakers={"1": "s2"})
        found = run(self.REUSE + """
            const now = {prompt: find(".mc-voice-box-prompt").value,
                         chosen: find(".mc-voice-box-configuration-select").value,
                         save: find(".mc-voice-box-configuration-save").textContent,
                         steps: find('[data-field="steps"]').value,
                         slot2: find('.mc-voice-box-speaker[data-speaker="2"] .mc-voice-box-speaker-title').textContent};
            const early = requestsTo("/pipelines/save").map((r) => r.body);
            advance(1000);
            await flush();
            report({now, early, saves: requestsTo("/pipelines/save").map((r) => r.body)});
        """, answers={"/samples": {"json": {"ok": True, "samples": [SAMPLE, dict(SAMPLE, id="s2", title="Bo")]}},
                      "/configurations": {"json": {"ok": True, "configurations": [CONFIGURATION, other]}},
                      "/pipelines": {"json": {"ok": True, "pipelines": [dict(PIPELINE, configuration_id="c2")]}},
                      "/outputs": {"json": {"ok": True, "outputs": [dict(OUTPUT, render=render)]}}})

        assert found["now"] == {"prompt": "Speaker 1: Hi.\nSpeaker 2: Yo.", "chosen": "c1",
                                "save": "Save •", "steps": "20", "slot2": "Bo"}
        working = found["state"]["working"]
        assert (working["model_id"], working["card_uuid"], working["steps"], working["cfg_scale"],
                working["seed"], working["max_new_tokens"]) == ("vibevoice-7b", "GPU-a", 20, 1.5, 42, 900)
        assert working["speakers"] == {"1": "s1", "2": "s2"}
        assert found["state"]["configurationId"] == "c1"
        assert found["early"] == [{"id": "p1", "configuration_id": "c1"}]
        assert found["saves"] == [{"id": "p1", "configuration_id": "c1"},
                                  {"id": "p1", "prompt": "Speaker 1: Hi.\nSpeaker 2: Yo."}]
        assert found["status"] == ("Loaded the prompt and settings of “Take 1”; "
                                   "save the configuration to keep them.")

    def test_the_configuration_as_it_stands_leaves_nothing_to_save(self):
        saved = dict(CONFIGURATION, seed=7)
        render = dict(OUTPUT["render"], configuration=dict(OUTPUT["render"]["configuration"], seed=7))
        found = run(self.REUSE + """
            report({save: find(".mc-voice-box-configuration-save").textContent});
        """, answers={"/configurations": {"json": {"ok": True, "configurations": [saved]}},
                      "/outputs": {"json": {"ok": True, "outputs": [dict(OUTPUT, render=render)]}}})

        assert found["save"] == "Save"
        assert found["state"]["dirty"]["configuration"] is False
        assert found["status"] == "Loaded the prompt and settings of “Take 1”."

    def test_a_deleted_configuration_starts_a_new_unsaved_one_named_after_the_render(self):
        render = dict(OUTPUT["render"], configuration=dict(OUTPUT["render"]["configuration"],
                                                           id="c9", steps=12))
        found = run(self.REUSE + """
            const select = find(".mc-voice-box-configuration-select");
            const now = {chosen: select.value, label: select.options[0].textContent,
                         save: find(".mc-voice-box-configuration-save").textContent};
            find(".mc-voice-box-configuration-save").click();
            await flush();
            report({now, saved: requestsTo("/configurations/save").map((r) => r.body)});
        """, answers={"/outputs": {"json": {"ok": True, "outputs": [dict(OUTPUT, render=render)]}}})

        assert found["now"] == {"chosen": "", "label": "(unsaved) Take 1", "save": "Save •"}
        saved = found["saved"][0]
        assert saved["name"] == "Take 1"
        assert "id" not in saved
        assert (saved["steps"], saved["seed"], saved["speakers"]) == (12, 42, {"1": "s1"})

    def test_a_speaker_whose_sample_has_left_the_library_is_left_empty_and_said(self):
        render = dict(OUTPUT["render"],
                      speakers=[{"n": 1, "sample_id": "s1", "title": "Ada"},
                                {"n": 2, "sample_id": "5a5a5a5a5a5a5a5a", "title": "Bo"}],
                      configuration=dict(OUTPUT["render"]["configuration"],
                                         speakers={"1": "s1", "2": "5a5a5a5a5a5a5a5a"}))
        found = run(self.REUSE + """
            report({line: line(),
                    slot2: find('.mc-voice-box-speaker[data-speaker="2"] .mc-voice-box-speaker-title').textContent});
        """, answers={"/outputs": {"json": {"ok": True, "outputs": [dict(OUTPUT, render=render)]}}})

        assert found["state"]["working"]["speakers"] == {"1": "s1"}
        assert found["slot2"] == "no sample"
        assert found["line"]["text"] == "Speaker 2's sample “Bo” is no longer in the library."
        assert found["line"]["kind"] == "warn"

    def test_a_default_model_or_card_is_left_as_the_editor_has_it(self):
        """A configuration records "" for the model and the card it left to the
        defaults; putting it back leaves the editor's own."""
        render = dict(OUTPUT["render"], configuration=dict(OUTPUT["render"]["configuration"],
                                                           id="c9", model_id="", card_uuid=""))
        found = run(self.REUSE + "report();",
                    answers={"/outputs": {"json": {"ok": True, "outputs": [dict(OUTPUT, render=render)]}}})

        working = found["state"]["working"]
        assert (working["model_id"], working["card_uuid"]) == ("vibevoice-7b", "GPU-a")

    def test_an_older_render_is_rebuilt_from_the_fields_it_recorded(self):
        found = run(self.REUSE + """
            report({prompt: find(".mc-voice-box-prompt").value,
                    label: find(".mc-voice-box-configuration-select").options[0].textContent});
        """, answers={"/outputs": {"json": {"ok": True, "outputs": [OLD_OUTPUT]}}})

        working = found["state"]["working"]
        assert found["state"]["configurationId"] == ""
        assert found["prompt"] == "Speaker 1: Before."
        assert found["label"] == "(unsaved) Old take"
        assert (working["name"], working["model_id"], working["card_uuid"], working["steps"],
                working["cfg_scale"], working["seed"], working["max_new_tokens"]) == (
            "Old take", "vibevoice-7b", "GPU-b", 12, 2.0, None, 900)
        assert working["speakers"] == {"1": "s1"}
        assert found["state"]["dirty"]["configuration"] is True


class TestTheDownloadName:
    DOWNLOAD = """
        await flush();
        const lane = find(".mc-voice-box-lane");
        lane.click();
        press("Download", lane);
        await flush();
        report();
    """

    def test_an_mp3_render_downloads_as_an_mp3(self):
        found = run(self.DOWNLOAD, answers={"/outputs/audio": {"bytes": 1044, "type": "audio/mpeg"}})

        assert found["linkClicks"] == [{"href": "blob:test-1", "download": "Take 1.mp3"}]

    def test_a_wav_render_downloads_as_a_wav(self):
        found = run(self.DOWNLOAD, answers={"/outputs": {"json": {"ok": True, "outputs": [
            dict(OUTPUT, format="wav")]}}})

        assert found["linkClicks"] == [{"href": "blob:test-1", "download": "Take 1.wav"}]

    def test_a_record_without_a_format_is_named_by_its_bytes(self):
        bare = {key: value for key, value in OUTPUT.items() if key != "format"}
        mp3 = run(self.DOWNLOAD, answers={"/outputs": {"json": {"ok": True, "outputs": [bare]}},
                                          "/outputs/audio": {"bytes": 1044, "type": "audio/mpeg"}})
        wav = run(self.DOWNLOAD, answers={"/outputs": {"json": {"ok": True, "outputs": [bare]}}})

        assert mp3["linkClicks"] == [{"href": "blob:test-1", "download": "Take 1.mp3"}]
        assert wav["linkClicks"] == [{"href": "blob:test-1", "download": "Take 1.wav"}]


class TestTheTint:
    IDS = ["s1", "s2", "0123456789abcdef", "ffffffffffffffff", "Ádá"]

    def test_a_samples_hue_is_worked_out_from_its_id_and_never_changes(self):
        scenario = "report({hues: IDS.map((id) => globalThis.mcVoiceBox.hueOf(id))});".replace(
            "IDS", json.dumps(self.IDS))
        once = run(scenario, root=False)
        twice = run(scenario, root=False)

        assert once["hues"] == twice["hues"] == [fnv_hue(identifier) for identifier in self.IDS]
        assert all(0 <= hue < 360 for hue in once["hues"])
        assert len(set(once["hues"])) == len(self.IDS)

    def test_a_sample_row_has_an_edge_in_its_tint(self):
        found = run("""
            await flush();
            const edge = find('.mc-voice-box-sample[data-id="s1"]').querySelector(".mc-voice-box-edge");
            report({hidden: edge.getAttribute("aria-hidden"),
                    segments: edge.children.map((s) => [s.getAttribute("data-sample"),
                                                        s.style.getPropertyValue("--mc-voice-box-hue")])});
        """)

        assert found["hidden"] == "true"
        assert found["segments"] == [["s1", str(fnv_hue("s1"))]]

    def test_a_lane_repeats_its_speakers_tints_one_segment_per_sample_in_speaker_order(self):
        render = dict(OUTPUT["render"], speakers=[
            {"n": 3, "sample_id": "s1", "title": "Ada"}, {"n": 1, "sample_id": "s2", "title": "Bo"},
            {"n": 2, "sample_id": "s1", "title": "Ada"}])
        found = run("""
            await flush();
            const edge = find(".mc-voice-box-lane").querySelector(".mc-voice-box-edge");
            report({hidden: edge.getAttribute("aria-hidden"), shown: shown(edge),
                    segments: edge.children.map((s) => [s.getAttribute("data-sample"),
                                                        s.style.getPropertyValue("--mc-voice-box-hue")])});
        """, answers={"/outputs": {"json": {"ok": True, "outputs": [dict(OUTPUT, render=render)]}}})

        assert found["hidden"] == "true"
        assert found["shown"] is True
        assert found["segments"] == [["s2", str(fnv_hue("s2"))], ["s1", str(fnv_hue("s1"))]]

    def test_a_lane_without_speakers_has_no_edge(self):
        render = dict(OUTPUT["render"], speakers=[])
        found = run("""
            await flush();
            report({edge: !!find(".mc-voice-box-lane").querySelector(".mc-voice-box-edge"),
                    segments: all(".mc-voice-box-lane .mc-voice-box-edge-segment").length});
        """, answers={"/outputs": {"json": {"ok": True, "outputs": [dict(OUTPUT, render=render)]}}})

        assert found["edge"] is False
        assert found["segments"] == 0


# --------------------------------------------------------------------------- #
# The Outputs header's one status line
# --------------------------------------------------------------------------- #

WARM = {"cards": {"abc": {"running": True, "loaded": True, "rendering": False, "job": "",
                          "resident_bytes": 19_300_000_000, "peak_bytes": 0,
                          "device_name": "NVIDIA GeForce RTX 3090", "uuid": "GPU-a"}},
        "last_error": ""}
FAILED = dict(JOB, phase="failed", live=False, warning="The card could not be made ready.",
              started=10, ended=12, elapsed=2.0)
QUEUED = dict(JOB, id="j2", name="Other 1", pipeline_id="p9", phase="queued")
RUNNING = dict(JOB, id="j3", name="Trailer 3", pipeline_id="p9", phase="rendering", started=5,
               elapsed=42.0, progress={"section": 1, "sections": 2, "seconds": 3.1})
INSTALLING = {"running": True, "text": "Installing the runtime", "fraction": 0.45,
              "failed": False, "model": ""}


def with_records(scenario: str) -> str:
    for name, value in (("WARM", WARM), ("FAILED", FAILED), ("QUEUED", QUEUED),
                        ("RUNNING", RUNNING), ("INSTALLING", INSTALLING), ("LIVE", JOB)):
        scenario = scenario.replace(name, json.dumps(value))
    return scenario


class TestTheStatusLine:
    def test_it_shows_the_first_of_its_seven_states_that_applies(self):
        """Ready, then each state that outranks the one before arriving in turn:
        Render's reason, VibeVoice warm, the last render having failed, the
        queue, a job with its time, an install -- and a message over them all
        until it is put away."""
        found = run(with_records("""
            await flush();
            const seen = [];
            const note = () => seen.push(line());
            const base = answers["/status"].json;
            const set = (changes) => { answers["/status"] = {json: Object.assign({}, base, changes)}; };
            note();
            type(find(".mc-voice-box-prompt"), "");
            note();
            set({runtime: WARM});
            await poll();
            note();
            type(find(".mc-voice-box-prompt"), "Speaker 1: Hello there.");
            set({runtime: WARM, jobs: [FAILED]});
            press("Render");
            await flush();
            note();
            set({runtime: WARM, jobs: [FAILED, QUEUED]});
            await poll();
            note();
            set({runtime: WARM, jobs: [FAILED, QUEUED, RUNNING]});
            await poll();
            note();
            set({runtime: WARM, jobs: [FAILED, QUEUED, RUNNING], progress: INSTALLING});
            await poll();
            note();
            find(".mc-voice-box-lane").click();
            press("Use seed");
            note();
            press("×", find(".mc-voice-box-status-line"));
            note();
            report({seen});
        """))

        assert [(entry["state"], entry["text"], entry["buttons"]) for entry in found["seen"]] == [
            ("ready", "Ready", []),
            ("blocked", "Write a script first.", []),
            ("warm", "VibeVoice warm on NVIDIA GeForce RTX 3090", ["Unload"]),
            ("failed", "Last render failed: The card could not be made ready.", ["×"]),
            ("queued", "1 queued", ["Clear queue"]),
            ("job", "Rendering “Trailer 3” · section 1 of 2 · 0:42 · 1 queued", ["Cancel", "Clear queue"]),
            ("install", "Installing the runtime — 45 %", []),
            ("message", "Seed 42 is in the configuration; save it to keep it.", ["×"]),
            ("install", "Installing the runtime — 45 %", []),
        ]
        assert found["seen"][3]["kind"] == "error"

    def test_each_state_gives_way_to_the_next_when_it_ends(self):
        found = run(with_records("""
            await flush();
            const seen = [];
            const note = () => seen.push(line());
            const base = answers["/status"].json;
            const set = (changes) => { answers["/status"] = {json: Object.assign({}, base, changes)}; };
            set({runtime: WARM, jobs: [FAILED]});
            press("Render");
            await flush();
            set({runtime: WARM, jobs: [FAILED, QUEUED, RUNNING], progress: INSTALLING});
            await poll();
            note();
            set({runtime: WARM, jobs: [FAILED, QUEUED, RUNNING]});
            await poll();
            note();
            press("Clear queue", find(".mc-voice-box-status-line"));
            await flush();
            note();
            set({runtime: WARM, jobs: [FAILED]});
            await poll();
            note();
            press("×", find(".mc-voice-box-status-line"));
            note();
            set({});
            await poll();
            note();
            report({seen, cleared: requestsTo("/jobs/clear").map(plain)});
        """), answers={"/jobs/clear": {"json": {"ok": True, "cleared": 1, "jobs": [
            FAILED, dict(QUEUED, phase="cancelled", live=False), RUNNING]}}})

        assert [(entry["state"], entry["text"], entry["buttons"]) for entry in found["seen"]] == [
            ("install", "Installing the runtime — 45 %", []),
            ("job", "Rendering “Trailer 3” · section 1 of 2 · 0:42 · 1 queued", ["Cancel", "Clear queue"]),
            ("job", "Rendering “Trailer 3” · section 1 of 2 · 0:42", ["Cancel"]),
            ("failed", "Last render failed: The card could not be made ready.", ["×"]),
            ("warm", "VibeVoice warm on NVIDIA GeForce RTX 3090", ["Unload"]),
            ("ready", "Ready", []),
        ]
        cleared = found["cleared"]
        assert [(c["method"], c["body"]) for c in cleared] == [("POST", {})]
        assert cleared[0]["headers"]["x-model-chain-voice"] == TOKEN
        assert cleared[0]["signal"] is True

    def test_the_clock_counts_on_from_the_servers_elapsed(self):
        """From the job's `elapsed` when the server answered plus the time since
        -- not from its `started`, which is the server's clock, not this one."""
        running = dict(RUNNING, started=1e12, progress={"section": 1, "sections": 1})
        found = run("""
            await flush();
            const first = line().text;
            const ticking = pendingDelays().filter((ms) => ms === 1000).length;
            const clock = find(".mc-voice-box-status-clock");
            const live = {line: find(".mc-voice-box-status").getAttribute("aria-live"),
                          clock: clock ? clock.getAttribute("aria-live") : null};
            advance(1000);
            await flush();
            const second = line().text;
            advance(2000);
            await flush();
            const third = line().text;
            answers["/status"] = {json: Object.assign({}, answers["/status"].json,
                                  {jobs: [Object.assign({}, RUNNING, {elapsed: 70.4})]})};
            await poll();
            const answered = line().text;
            answers["/status"] = {json: Object.assign({}, answers["/status"].json,
                                  {jobs: [Object.assign({}, RUNNING, {elapsed: null})]})};
            await poll();
            const unknown = {text: line().text, ticking: pendingDelays().filter((ms) => ms === 1000).length};
            answers["/status"] = {json: Object.assign({}, answers["/status"].json, {jobs: []})};
            await poll();
            report({first, ticking, live, second, third, answered, unknown,
                    idle: pendingDelays().filter((ms) => ms === 1000).length});
        """.replace("RUNNING", json.dumps(running)),
            answers={"/status": {"json": status_with(jobs=[running])}})

        assert found["first"] == "Rendering “Trailer 3” · 0:42"
        assert found["ticking"] == 1
        assert found["live"] == {"line": "polite", "clock": "off"}
        assert found["second"] == "Rendering “Trailer 3” · 0:43"
        assert found["third"] == "Rendering “Trailer 3” · 0:45"
        assert found["answered"] == "Rendering “Trailer 3” · 1:10"
        assert found["unknown"] == {"text": "Rendering “Trailer 3”", "ticking": 0}
        assert found["idle"] == 0

    def test_a_hidden_tab_does_not_tick(self):
        """Not when it is hidden, and not when the line is drawn again while it
        is (a speaker cleared, say)."""
        found = run("""
            await flush();
            const before = pendingDelays().filter((ms) => ms === 1000).length;
            document.visibilityState = "hidden";
            document.dispatchEvent(new Event("visibilitychange"));
            const hidden = pendingDelays().filter((ms) => ms === 1000).length;
            press("×", find('.mc-voice-box-speaker[data-speaker="1"]'));
            report({before, hidden, redrawn: pendingDelays().filter((ms) => ms === 1000).length,
                    line: line().state});
        """, answers={"/status": {"json": status_with(jobs=[RUNNING])}})

        assert found["before"] == 1
        assert found["hidden"] == 0
        assert found["redrawn"] == 0
        assert found["line"] == "job"

    def test_a_card_that_is_rendering_is_not_warm(self):
        rendering = {"cards": {"abc": dict(WARM["cards"]["abc"], rendering=True, job="j3")},
                     "last_error": ""}
        found = run("await flush(); report({line: line()});",
                    answers={"/status": {"json": status_with(runtime=rendering)}})

        assert (found["line"]["state"], found["line"]["text"]) == ("ready", "Ready")

    def test_the_phase_words_and_this_pages_own_job_first(self):
        """Waiting says what for; a job of this page's is the one shown when
        two cards are busy, whichever started first."""
        waiting = dict(JOB, id="j4", name="Mine", phase="waiting", started=50, elapsed=3.0,
                       reason="the image model is generating")
        loading = dict(waiting, phase="loading", reason="loading the model")
        found = run("""
            await flush();
            const base = answers["/status"].json;
            const set = (jobs) => { answers["/status"] = {json: Object.assign({}, base, {jobs})}; };
            set([RUNNING]);
            await poll();
            const theirs = line().text;
            answers["/render"] = {json: {ok: true, job: WAITING}};
            set([RUNNING, WAITING]);
            press("Render");
            await flush();
            const waiting = line().text;
            set([RUNNING, LOADING]);
            await poll();
            report({theirs, waiting, loading: line().text});
        """.replace("RUNNING", json.dumps(RUNNING)).replace("WAITING", json.dumps(waiting))
           .replace("LOADING", json.dumps(loading)))

        assert found["theirs"] == "Rendering “Trailer 3” · section 1 of 2 · 0:42"
        assert found["waiting"] == "Waiting for the card: the image model is generating · “Mine” · 0:03"
        assert found["loading"] == "Loading VibeVoice · “Mine” · 0:03"

    def test_cancel_stops_the_running_job_even_one_from_another_pipeline(self):
        found = run("""
            await flush();
            press("Cancel", find(".mc-voice-box-status-line"));
            await flush();
            report({cancel: requestsTo("/jobs/cancel").map((r) => r.body)});
        """, answers={"/status": {"json": status_with(jobs=[QUEUED, RUNNING])}})

        assert found["cancel"] == [{"id": "j3"}]

    def test_a_failure_stays_until_its_cross_or_the_next_render_press(self):
        found = run(with_records("""
            await flush();
            const base = answers["/status"].json;
            const set = (jobs) => { answers["/status"] = {json: Object.assign({}, base, {jobs})}; };
            set([FAILED]);
            press("Render");
            await flush();
            const failed = line().text;
            press("×", find(".mc-voice-box-status-line"));
            const dismissed = line().text;
            answers["/render"] = {json: {ok: true, job: Object.assign({}, LIVE, {id: "j5"})}};
            set([FAILED, Object.assign({}, FAILED, {id: "j5", warning: "Out of memory."})]);
            press("Render");
            await flush();
            const again = line().text;
            answers["/render"] = {json: {ok: true, job: Object.assign({}, LIVE, {id: "j6"})}};
            set([Object.assign({}, LIVE, {id: "j6", phase: "done", live: false, output_id: "o1"})]);
            press("Render");
            await flush(10);
            report({failed, dismissed, again, pressed: line().text});
        """))

        assert found["failed"] == "Last render failed: The card could not be made ready."
        assert found["dismissed"] == "Ready"
        assert found["again"] == "Last render failed: Out of memory."
        assert found["pressed"] == "Ready"

    def test_a_failed_install_is_why_render_is_disabled(self):
        found = run("""
            await flush();
            report({line: line(), title: find(".mc-voice-box-render").getAttribute("title"),
                    install: shown(find(".mc-voice-box-install"))});
        """, answers={"/status": {"json": status_with(
            engine={"ready": False, "supported": True, "message": "VibeVoice is not installed.",
                    "model_id": "vibevoice-7b", "download_bytes": 0},
            progress={"running": False, "text": "No space left on the drive.", "fraction": 0.0,
                      "failed": True, "model": ""})}})

        assert found["line"]["text"] == "Install failed: No space left on the drive."
        assert found["line"]["state"] == "blocked"
        assert found["title"] == "Install failed: No space left on the drive."
        assert found["install"] is True

    def test_a_message_holds_the_line_for_a_few_seconds_then_gives_it_back(self):
        info = run("""
            await flush();
            find(".mc-voice-box-lane").click();
            press("Use seed");
            advance(4999);
            const held = line().state;
            advance(1);
            report({held, after: line()});
        """)
        warn = run("""
            await flush();
            press("Record");
            await flush();
            advance(11999);
            const held = line();
            advance(1);
            report({held, after: line().text});
        """, microphone=False)

        assert info["held"] == "message"
        assert (info["after"]["state"], info["after"]["text"]) == ("ready", "Ready")
        assert warn["held"]["text"] == "This browser cannot record here."
        assert warn["held"]["kind"] == "warn"
        assert warn["after"] == "Ready"


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
