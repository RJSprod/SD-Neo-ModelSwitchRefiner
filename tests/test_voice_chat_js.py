"""The Voice Chat browser script, executed rather than read.

Everything in ``javascript/voice_chat.js`` that could go wrong quietly is here:
a press-and-hold that starts two recordings, a microphone left open after the
finger came up, a transcript that replaced somebody's half-typed message, an
auto-send that fired when the setting was off, a reply spoken twice because a
token was seen twice, and a second set of pointer listeners installed by the
next Gradio update.

None of those can be found by reading the file, and none of them needs a
browser. They need a DOM whose numbers a test can set and a clock a test can
advance -- which is what the harness below is, following the pattern
``test_llm_studio_js.py`` established for the same reason.

The interesting fakes are the ones with opinions:

    the clock          held still, so "how long was the button held" is a value
                       the test chooses rather than a race with the machine;
    the timers         queued rather than run, so the sixty-second cap is a
                       timer a test can decide not to fire;
    getUserMedia       hands back tracks that record having been stopped, which
                       is how "the microphone was closed" is asserted rather
                       than assumed;
    fetch              records every request, so what left the page -- and what
                       did not -- is readable.

These run under node, which is not a Forge dependency, so they skip without it.
"""

from __future__ import annotations

import json
import pathlib
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "javascript" / "voice_chat.js"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


HARNESS = r"""
// --- a DOM with numbers a test can set ---------------------------------- //

let NOW = 1000;
const timers = [];
const intervals = [];
const requests = [];
const consoleErrors = [];
const consoleInfo = [];

function element(id, tag) {
    const listeners = {};
    const node = {
        id,
        tagName: tag || "DIV",
        dataset: {},
        disabled: false,
        value: "",
        // The slide moves the handle with a transform and reads the track's
        // width to know how far "all the way to the right" is, so both are
        // part of the DOM this harness has to have.
        style: {},
        offsetWidth: 0,
        clientWidth: 0,
        offsetParent: {},
        innerHTML: "",
        textContent: "",
        children: [],
        classList: {
            names: new Set(),
            // Counted, because a browser counts them too: `classList.remove` of
            // a class that is not there still rewrites the element's `class`
            // attribute and wakes every MutationObserver watching it. "The
            // composer state is asserted and not rewritten" is a claim about
            // this number.
            writes: 0,
            add(...names) {
                node.classList.writes += 1;
                names.forEach((n) => node.classList.names.add(n));
            },
            remove(...names) {
                node.classList.writes += 1;
                names.forEach((n) => node.classList.names.delete(n));
            },
            contains(name) { return node.classList.names.has(name); },
        },
        listeners,
        addEventListener(name, handler) {
            (listeners[name] = listeners[name] || []).push(handler);
        },
        removeEventListener() {},
        setAttribute(name, value) { node[name] = value; },
        getAttribute(name) { return node[name] === undefined ? null : node[name]; },
        removeAttribute(name) { delete node[name]; },
        appendChild(child) { node.children.push(child); node.textContent = child.textContent; },
        querySelector(selector) {
            if (selector === "button") return node.tagName === "BUTTON" ? node : node.inner;
            if (selector.indexOf("textarea") === 0 || selector === "textarea, input") {
                return node.inner || null;
            }
            if (selector === ".mc-llm-notice") return node.notice || null;
            return null;
        },
        querySelectorAll() { return []; },
        setPointerCapture(id) { node.captured = id; },
        releasePointerCapture() { node.captured = null; },
        click() { node.clicks = (node.clicks || 0) + 1; },
        dispatchEvent(event) {
            (listeners[event.type] || []).forEach((handler) => handler(event));
            return true;
        },
        fire(name, detail) {
            const event = Object.assign({type: name, preventDefault() {}, button: 0},
                                        detail || {});
            (listeners[name] || []).forEach((handler) => handler(event));
        },
        get handlers() { return listeners; },
    };
    return node;
}

function field(id) {
    const holder = element(id);
    const inner = element(id + "-inner", "TEXTAREA");
    holder.inner = inner;
    return holder;
}

const elements = {};
["mc-llm-chat-voice-mic", "mc-llm-chat-send", "mc-llm-chat-stop",
 "mc-llm-chat-to-voice"].forEach(function (id) {
    elements[id] = element(id, "BUTTON");
});
["mc-llm-chat-voice-token", "mc-llm-chat-voice-turn", "mc-llm-chat-voice-run-state",
 "mc-llm-chat-voice-key", "mc-llm-chat-message"].forEach(function (id) {
    elements[id] = field(id);
});
elements["mc-llm-chat-voice-run-state"].inner.value = "idle";
elements["mc-llm-chat-status"] = element("mc-llm-chat-status");
elements["mc-llm-chat-status"].notice = element("notice");

// The Voice flyout's live engine block, which Python renders and this script
// repaints. Its parts are found by data attribute, exactly as the real one's
// are, so a selector typo in the script is a test failure rather than a panel
// that quietly never updates.
const enginePanel = element("mc-llm-chat-voice-engine");
// A closed flyout, which is where a page starts. Section 33: nothing is polled
// while the overlay is not on screen.
enginePanel.offsetParent = null;
enginePanel.button = element("engine-button", "BUTTON");
enginePanel.button.setAttribute("data-mc-voice-runtime", "load");
enginePanel.button.closest = (selector) =>
    selector.indexOf("mc-voice-runtime") !== -1 ? enginePanel.button : null;
enginePanel.line = element("engine-line");
enginePanel.voice = element("engine-voice");
enginePanel.querySelector = (selector) => {
    if (selector.indexOf("engine-line") !== -1) return enginePanel.line;
    if (selector.indexOf("mc-voice-runtime") !== -1) return enginePanel.button;
    if (selector.indexOf("voice-default") !== -1) return enginePanel.voice;
    return null;
};
elements["mc-llm-chat-voice-engine"] = enginePanel;

["mc-llm-chat-voice-auto-send", "mc-llm-chat-voice-auto-speak"].forEach(function (id) {
    elements[id] = element(id, "INPUT");
    elements[id].type = "checkbox";
    elements[id].checked = true;
});

// The composer's slide track: two handles across, the handle 44px of it. Real
// numbers, because the gesture's engage threshold is a fraction of the travel
// and a track with no width would engage on the press.
const micTrack = element("mc-llm-chat-voice-track");
micTrack.clientWidth = 88;
elements["mc-llm-chat-voice-track"] = micTrack;
elements["mc-llm-chat-voice-mic"].offsetWidth = 44;
elements["mc-llm-chat-voice-mic"].closest = (selector) =>
    selector.indexOf("mc-llm-voice-track") !== -1 ? micTrack : null;

elements["mc-llm-chat-voice-key"].inner.value = "PAGE-TOKEN";
// Hidden: Send is showing, Stop is not, which is the idle composer.
elements["mc-llm-chat-stop"].offsetParent = null;

// The Settings page row, which is drawn by Python and made live by this script.
// Its own little DOM because it is not inside the Conversation panel and is
// found by class rather than by id.
const settingsParts = {
    runtime: element("runtime"),
    sttLine: element("stt-line"),
    ttsLine: element("tts-line"),
    sttButton: element("stt-button", "BUTTON"),
    ttsButton: element("tts-button", "BUTTON"),
};
settingsParts.sttButton.textContent = "Download default STT";
settingsParts.ttsButton.textContent = "Download default TTS";
settingsParts.sttButton["data-mc-voice-install"] = "stt";
settingsParts.ttsButton["data-mc-voice-install"] = "tts";
settingsParts.sttLocal = element("stt-local", "BUTTON");
settingsParts.ttsLocal = element("tts-local", "BUTTON");
settingsParts.sttLocal["data-mc-voice-local"] = "stt";
settingsParts.ttsLocal["data-mc-voice-local"] = "tts";
settingsParts.sttFolder = element("stt-folder", "INPUT");
settingsParts.ttsFolder = element("tts-folder", "INPUT");
settingsParts.runtimeLine = element("runtime-line");
settingsParts.runtimeButton = element("runtime-button", "BUTTON");
settingsParts.runtimeButton["data-mc-voice-install"] = "runtime";

// Speech to text is three qualities rather than one bundle, so its row is three
// cards with a Download and a Use each. Modelled here in the same shape the
// real markup has, so a selector typo in the script is a failing test rather
// than three cards that never move.
function tierCard(identifier) {
    const card = element("tier-" + identifier);
    card.installButton = element("tier-install-" + identifier, "BUTTON");
    card.installButton.setAttribute("data-mc-voice-tier-install", identifier);
    card.useButton = element("tier-use-" + identifier, "BUTTON");
    card.useButton.setAttribute("data-mc-voice-tier-use", identifier);
    card.state = element("tier-state-" + identifier);
    card.mark = element("tier-mark-" + identifier);
    card.local = element("tier-local-" + identifier, "BUTTON");
    card.local.setAttribute("data-mc-voice-local", "stt");
    card.local.setAttribute("data-mc-voice-model", identifier);
    card.local.setAttribute("data-mc-voice-scope", identifier);
    card.folder = element("tier-folder-" + identifier, "INPUT");
    card.querySelector = function (selector) {
        if (selector.indexOf("tier-install") !== -1) return card.installButton;
        if (selector.indexOf("tier-use") !== -1) return card.useButton;
        if (selector.indexOf("tier-state") !== -1) return card.state;
        if (selector.indexOf("tier-mark") !== -1) return card.mark;
        if (selector.indexOf("mc-voice-local") !== -1) return card.local;
        return null;
    };
    const answer = (selector) => (selector.indexOf("tier-install") !== -1
                                  || selector.indexOf("tier-use") !== -1
                                  || selector.indexOf("mc-voice-local") !== -1);
    card.installButton.closest = (selector) =>
        selector.indexOf("tier-install") !== -1 ? card.installButton : null;
    card.useButton.closest = (selector) =>
        selector.indexOf("tier-use") !== -1 ? card.useButton : null;
    card.local.closest = (selector) => answer(selector) ? card.local : null;
    return card;
}

settingsParts.tiers = {
    "whisper-base-int8": tierCard("whisper-base-int8"),
    "whisper-small-int8": tierCard("whisper-small-int8"),
    "whisper-medium-int8": tierCard("whisper-medium-int8"),
};
settingsParts.tierList = element("tier-list");
settingsParts.tierList.querySelector = function (selector) {
    const found = Object.keys(settingsParts.tiers).filter(
        (id) => selector.indexOf(id) !== -1)[0];
    return found ? settingsParts.tiers[found] : null;
};
settingsParts.chosenLabel = element("stt-chosen");

// The text-to-speech engine selector. Two cards; the selected one is disabled,
// which is what stops somebody re-selecting the engine they already have.
settingsParts.enginePanel = element("engines");
settingsParts.enginePanel["data-mc-voice-engines"] = "";
settingsParts.engineCards = {
    kokoro: element("engine-kokoro", "BUTTON"),
    sopro: element("engine-sopro", "BUTTON"),
};
settingsParts.engineCards.kokoro["data-mc-voice-engine-pick"] = "kokoro";
settingsParts.engineCards.sopro["data-mc-voice-engine-pick"] = "sopro";
settingsParts.engineCards.kokoro.disabled = true;
settingsParts.enginePanel.querySelectorAll = function (selector) {
    if (selector === "[data-mc-voice-engine-pick]") {
        return [settingsParts.engineCards.kokoro, settingsParts.engineCards.sopro];
    }
    return [];
};
settingsParts.enginePanel.querySelector = function () { return null; };
settingsParts.engineCards.kokoro.closest = function (selector) {
    return selector === "[data-mc-voice-engine-pick]" ? settingsParts.engineCards.kokoro : null;
};
settingsParts.engineCards.sopro.closest = function (selector) {
    return selector === "[data-mc-voice-engine-pick]" ? settingsParts.engineCards.sopro : null;
};

const settingsRow = element("settings");
settingsRow["data-mc-voice-key"] = "PAGE-TOKEN";
settingsRow.querySelector = function (selector) {
    if (selector === ".mc-voice-runtime") return settingsParts.runtime;
    if (selector === '[data-mc-voice-status="runtime"]') return settingsParts.runtimeLine;
    if (selector === '[data-mc-voice-install="runtime"]') return settingsParts.runtimeButton;
    if (selector === '[data-mc-voice-local="stt"]') return settingsParts.sttLocal;
    if (selector === '[data-mc-voice-local="tts"]') return settingsParts.ttsLocal;
    if (selector === '[data-mc-voice-status="stt"]') return settingsParts.sttLine;
    if (selector === '[data-mc-voice-status="tts"]') return settingsParts.ttsLine;
    if (selector === '[data-mc-voice-install="stt"]') return settingsParts.sttButton;
    if (selector === '[data-mc-voice-install="tts"]') return settingsParts.ttsButton;
    if (selector === '[data-mc-voice-folder="stt"]') return settingsParts.sttFolder;
    if (selector === '[data-mc-voice-folder="tts"]') return settingsParts.ttsFolder;
    if (selector === "[data-mc-voice-engines]") return settingsParts.enginePanel;
    if (selector === '[data-mc-voice-kind="sopro"]') return null;
    if (selector === '[data-mc-voice-tiers="stt"]') return settingsParts.tierList;
    if (selector === '[data-mc-voice-chosen="stt"]') return settingsParts.chosenLabel;
    if (selector.indexOf("data-mc-voice-tier=") !== -1) {
        const wanted = Object.keys(settingsParts.tiers).filter(
            (id) => selector.indexOf(id) !== -1)[0];
        return wanted ? settingsParts.tiers[wanted] : null;
    }
    if (selector.indexOf("data-mc-voice-folder=") !== -1) {
        const wanted = Object.keys(settingsParts.tiers).filter(
            (id) => selector.indexOf(id) !== -1)[0];
        return wanted ? settingsParts.tiers[wanted].folder : null;
    }
    return null;
};
settingsRow.querySelectorAll = function (selector) {
    if (selector === "[data-mc-voice-install]") {
        return [settingsParts.runtimeButton, settingsParts.sttButton, settingsParts.ttsButton];
    }
    if (selector === "[data-mc-voice-local]") {
        return [settingsParts.sttLocal, settingsParts.ttsLocal].concat(
            Object.keys(settingsParts.tiers).map((id) => settingsParts.tiers[id].local));
    }
    return [];
};

// The Voices row, which is the other block Python draws on the Settings page.
// Off screen to begin with, because that is where a settings row starts on
// every page of this WebUI that is not the Settings page -- and nothing is
// fetched for a row nobody can see.
const voicesRow = element("voices");
voicesRow["data-mc-voice-key"] = "PAGE-TOKEN";
voicesRow.offsetParent = VOICES_VISIBLE ? {} : null;
voicesRow.getBoundingClientRect = () => ({width: 0, height: 0});

// The four delivery sliders, which live in the Voices row and are the same four
// controls the character screen draws as Gradio sliders. Modelled here because
// they are the half of that pair this script owns.
const voicesParts = {sliders: {}, outputs: {}};
voicesParts.deliveryPanel = element("delivery");
voicesParts.deliverySummary = element("delivery-summary");
voicesParts.resetButton = element("delivery-reset", "BUTTON");
voicesParts.testButton = element("delivery-test", "BUTTON");
["speed", "pitch", "gain", "pause"].forEach(function (name) {
    const slider = element("slider-" + name, "INPUT");
    slider.type = "range";
    slider.value = name === "speed" ? "1" : "0";
    slider.setAttribute("data-mc-voice-slider-input", name);
    slider.closest = (selector) =>
        selector.indexOf("slider-input") !== -1 ? slider : null;
    voicesParts.sliders[name] = slider;
    voicesParts.outputs[name] = element("output-" + name);
});
voicesParts.deliveryPanel.querySelector = function (selector) {
    const name = ["speed", "pitch", "gain", "pause"].filter(
        (key) => selector.indexOf('"' + key + '"') !== -1
                 || selector.indexOf("=" + key) !== -1)[0];
    if (selector.indexOf("slider-input") !== -1) {
        return name ? voicesParts.sliders[name] : null;
    }
    if (selector.indexOf("slider-value") !== -1) {
        return name ? voicesParts.outputs[name] : null;
    }
    if (selector.indexOf("delivery-summary") !== -1) return voicesParts.deliverySummary;
    return null;
};
voicesParts.deliveryPanel.querySelectorAll = function (selector) {
    if (selector.indexOf("slider-input") !== -1) {
        return ["speed", "pitch", "gain", "pause"].map((k) => voicesParts.sliders[k]);
    }
    return [];
};
voicesRow.querySelector = function (selector) {
    if (selector.indexOf("mc-voice-delivery]") !== -1) return voicesParts.deliveryPanel;
    return null;
};

globalThis.document = {
    documentElement: element("html"),
    querySelector(selector) {
        if (selector.charAt(0) === "#") return elements[selector.slice(1)] || null;
        if (selector === ".mc-voice-settings") return SETTINGS_PRESENT ? settingsRow : null;
        if (selector === ".mc-voice-voices") return VOICES_PRESENT ? voicesRow : null;
        if (selector === "[data-mc-voice-key]") return SETTINGS_PRESENT ? settingsRow : null;
        return null;
    },
    createElement(tag) { return element("created", tag.toUpperCase()); },
    addEventListener() {},
    readyState: "complete",
    hidden: HIDDEN,
};
globalThis.gradioApp = () => globalThis.document;
globalThis.window = globalThis;
const reloads = [];
globalThis.location = {pathname: "/", origin: "https://forge.example",
                       reload: () => { reloads.push(NOW); }};
globalThis.isSecureContext = SECURE;
globalThis.gradio_config = GRADIO_CONFIG;

globalThis.setTimeout = (fn, ms) => {
    timers.push({fn, at: NOW + (ms || 0)});
    return timers.length;
};
globalThis.clearTimeout = (handle) => {
    if (timers[handle - 1]) timers[handle - 1].cancelled = true;
};
globalThis.setInterval = (fn) => { intervals.push(fn); return intervals.length; };
globalThis.clearInterval = () => {};
// The wall clock is NOW plus an offset a test can move on its own. That is how
// "the machine synchronised its time in the middle of a recording" is modelled:
// `Date.now()` jumps and `performance.now()` does not, which is the whole reason
// durations are measured with the second one.
let WALL = 0;
globalThis.Date = class extends Date { static now() { return NOW + WALL; } };
// The monotonic clock, moving with the same NOW the tests advance. It is a
// separate global because the script measures every *duration* with it and only
// asks the wall clock for absolute moments -- so a test that moves NOW is
// moving both, and a test could move one without the other to model a machine
// whose wall clock jumped mid-recording.
globalThis.performance = {now() { return NOW; }};
const windowHandlers = {};
globalThis.windowHandlers = windowHandlers;
globalThis.addEventListener = (name, handler) => {
    (windowHandlers[name] = windowHandlers[name] || []).push(handler);
};
globalThis.console = Object.assign({}, console, {
    error(...args) { consoleErrors.push(args.map(String).join(" ")); },
    // Captured rather than printed. The playback and capture reports are the
    // page's own content-free diagnostics, and "what is in them, and what is
    // deliberately not" is a claim worth asserting.
    info(...args) { consoleInfo.push(args.map(String).join(" ")); },
});

function runTimers() {
    // Everything due at the current instant, once. Cancelled entries stay
    // cancelled, which is how "the sixty second cap did not fire" is asserted.
    const due = timers.filter((t) => !t.cancelled && !t.done && t.at <= NOW);
    due.forEach((t) => { t.done = true; t.fn(); });
    return due.length;
}

function pending() {
    return timers.filter((t) => !t.cancelled && !t.done).length;
}

// --- audio ---------------------------------------------------------------- //

const tracks = [];
const played = [];
const stopped = [];
let pendingMicrophone = null;
let pendingStatus = null;

function track() {
    // `label` and `getSettings` are how the script finds out what the browser
    // actually gave it -- which device, at what rate. On Android that is the
    // difference between the handset's own microphone and a Bluetooth headset's
    // narrowband one, and it is the only evidence a user has that it changed.
    const item = {
        stopped: false,
        stop() { item.stopped = true; },
        label: TRACK_LABEL,
        getSettings: () => ({sampleRate: TRACK_RATE, channelCount: 1, deviceId: "d1"}),
    };
    tracks.push(item);
    return item;
}

const scheduled = [];

// How many times the capture processor's module was loaded. `registerProcessor`
// puts a name in the context's worklet scope, so a second load for the same
// context is a duplicate registration -- which a browser may refuse outright and
// which is in any case a Blob, an object URL and a module load per utterance.
let moduleLoads = 0;
const context = {
    state: CONTEXT_STATE,
    sampleRate: SAMPLE_RATE,
    currentTime: 0,
    audioWorklet: WORKLET_AVAILABLE ? {
        addModule() {
            moduleLoads += 1;
            return WORKLET_LOADS
                ? Promise.resolve()
                : Promise.reject(new Error("NotSupportedError"));
        },
    } : null,
    resume() { context.state = RESUME_WORKS ? "running" : "suspended"; return Promise.resolve(); },
    createMediaStreamSource() { return {connect() {}, disconnect() {}}; },
    createScriptProcessor() {
        const node = {connect() {}, disconnect() {}, onaudioprocess: null};
        globalThis.processorNode = node;
        return node;
    },
    createGain() { return {gain: {}, connect() {}, disconnect() {}}; },
    createBuffer(channels, length, rate) {
        const data = new Float32Array(length);
        return {
            numberOfChannels: channels,
            length,
            sampleRate: rate,
            duration: length / rate,
            getChannelData: () => data,
        };
    },
    createBufferSource() {
        const source = {
            buffer: null, onended: null,
            connect() {}, disconnect() {},
            start(at) { played.push(source); scheduled.push({source, at: at || 0}); },
            stop() { stopped.push(source); },
        };
        return source;
    },
    decodeAudioData(buffer, ok, fail) {
        if (DECODE_WORKS) ok({duration: 1}); else fail(new Error("nope"));
    },
    destination: {},
};

globalThis.AudioContext = function () { return context; };
globalThis.AudioWorkletNode = function () {
    const node = {port: {onmessage: null}, connect() {}, disconnect() {}};
    globalThis.workletNode = node;
    return node;
};
globalThis.Blob = function () {};
globalThis.URL = {createObjectURL: () => "blob:worklet", revokeObjectURL() {}};

// defineProperty rather than assignment: node 22 publishes navigator as a
// getter-only global, and `globalThis.navigator = ...` throws.
Object.defineProperty(globalThis, "navigator", {
    configurable: true,
    writable: true,
    value: (function () {
        let opened = 0;
        const open = function (constraints) {
            requests.push({kind: "getUserMedia", constraints});
            opened += 1;
            // A browser that rejects the constraint *set* rather than the
            // request. The script is expected to ask again for any microphone
            // at all rather than report that capture is unavailable.
            if (FIRST_OPEN_FAILS && opened === 1) {
                const refusal = new Error("constraints");
                refusal.name = DENIAL;
                return Promise.reject(refusal);
            }
            if (!PERMISSION) {
                const error = new Error("refused");
                error.name = DENIAL;
                return Promise.reject(error);
            }
            const only = [track()];
            const stream = {getTracks: () => only, getAudioTracks: () => only};
            if (SLOW_OPEN) {
                // A permission prompt nobody has answered yet, or a device the
                // platform is still waking. Released by `openMicrophoneNow()`.
                return new Promise(function (resolve) { pendingMicrophone = resolve; })
                    .then(() => stream);
            }
            return Promise.resolve(stream);
        };
        // The prefixed callback form, which is what some Android WebViews have
        // instead of `mediaDevices`. The script has to reach it, or a phone
        // that could have recorded is told it cannot.
        if (LEGACY) {
            return {
                webkitGetUserMedia(constraints, resolve, reject) {
                    open(constraints).then(resolve, reject);
                },
            };
        }
        return MICROPHONE ? {mediaDevices: {getUserMedia: open}} : {};
    })(),
});

globalThis.Event = function (type, options) {
    return {type, bubbles: !!(options && options.bubbles), preventDefault() {}};
};

// --- fetch ---------------------------------------------------------------- //

// Chunk boundaries are the interesting part of a stream answer: `chunks` is a
// list of byte arrays that a reader hands over one at a time, exactly as
// arbitrarily-sized network reads would.
const readers = [];

function streamBody(chunks, stall) {
    let index = 0;
    // A plan rather than a literal: the high-water tests need tens of thousands
    // of samples, and `node -e` has an argument-length limit that a literal
    // array of them comfortably exceeds.
    if (chunks && chunks.plan) {
        const block = new Array(chunks.plan.bytes).fill(chunks.plan.fill || 0);
        chunks = new Array(chunks.plan.count).fill(block);
    }
    const reader = {
        cancelled: false,
        read() {
            if (reader.cancelled) return Promise.resolve({done: true, value: undefined});
            if (index >= chunks.length) {
                // A stream the server has not finished. Held open rather than
                // ended, so "a reply is still being read aloud" is a state a
                // test can stand in the middle of.
                if (stall) return new Promise(() => {});
                return Promise.resolve({done: true, value: undefined});
            }
            const value = Uint8Array.from(chunks[index]);
            index += 1;
            return Promise.resolve({done: false, value});
        },
        cancel() { reader.cancelled = true; return Promise.resolve(); },
    };
    readers.push(reader);
    return {getReader: () => reader};
}

globalThis.AbortController = function () {
    this.signal = {aborted: false};
    const signal = this.signal;
    this.abort = function () { signal.aborted = true; aborts.push(1); };
};
const aborts = [];

globalThis.MutationObserver = function (fn) {
    this.observe = function () { observers.push(fn); };
    this.disconnect = function () {};
};
const observers = [];

globalThis.fetch = function (url, options) {
    requests.push({url, options: options || {},
                   body: options && options.body,
                   headers: (options && options.headers) || {}});
    const answer = ANSWERS[Object.keys(ANSWERS).find((k) => url.indexOf(k) !== -1)];
    if (!answer) return Promise.reject(new Error("no route " + url));
    if (answer.reject) return Promise.reject(new Error("network"));
    const headerBag = answer.headers || {};
    const response = {
        ok: answer.status === undefined || answer.status < 400,
        status: answer.status || 200,
        headers: {get: (name) => headerBag[name] === undefined ? null : headerBag[name]},
        body: answer.chunks ? streamBody(answer.chunks, answer.stall) : null,
        json: () => Promise.resolve(answer.json),
        arrayBuffer: () => Promise.resolve(answer.audio || new ArrayBuffer(8)),
    };
    if (answer.defer) {
        // The WebUI has been asked and has not answered. Released by
        // `answerStatus()`, which is how "samples arrived before the status
        // check came back" becomes a thing a test can stand in the middle of.
        return new Promise(function (resolve) {
            const waiting = pendingStatus || [];
            waiting.push(() => resolve(response));
            pendingStatus = waiting;
        });
    }
    return Promise.resolve(response);
};

const loaded = [];
const updated = [];
globalThis.onUiLoaded = (fn) => loaded.push(fn);
globalThis.onAfterUiUpdate = (fn) => updated.push(fn);

SOURCE

loaded.forEach((fn) => fn());

const settle = () => new Promise((resolve) => process.nextTick(resolve));

async function tick(times) {
    for (let i = 0; i < (times || 8); i += 1) {
        await settle();
        runTimers();
    }
}

const mic = elements["mc-llm-chat-voice-mic"];
const send = elements["mc-llm-chat-send"];
const stop = elements["mc-llm-chat-stop"];
const message = elements["mc-llm-chat-message"].inner;
const token = elements["mc-llm-chat-voice-token"].inner;
const turnField = elements["mc-llm-chat-voice-turn"].inner;
const runState = elements["mc-llm-chat-voice-run-state"].inner;

// Move the audio clock forward, which is how "the queue drained" and "playback
// has caught up" are things a test can make happen.
function advanceAudio(seconds) {
    context.currentTime += seconds;
}

// Everything the page does on its own clock: the poll that notices a new turn
// and keeps the composer honest, plus whatever timers are due.
async function pump(times) {
    for (let i = 0; i < (times || 8); i += 1) {
        intervals.forEach((fn) => fn());
        await settle();
        runTimers();
        await settle();
    }
}

// Give the browser a turn to speak, then let the reader run to completion.
async function speak(id, times) {
    turnField.value = id;
    await pump(times || 30);
}

// Gradio writing a hidden field: the value changes and an `input` event is
// dispatched on the control itself. This is the fast path the script prefers,
// and it deliberately does *not* run the poll.
async function announceTurn(id, times) {
    turnField.value = id;
    turnField.fire("input", {});
    await tick(times || 30);
}

// Let a held-open getUserMedia resolve, and a held-open status route answer.
async function openMicrophoneNow() {
    const resolve = pendingMicrophone;
    pendingMicrophone = null;
    if (resolve) resolve();
    await tick(4);
}

async function answerStatus() {
    const waiting = pendingStatus || [];
    pendingStatus = null;
    waiting.forEach((fn) => fn());
    await tick(4);
}

// Gradio re-rendering the panel: the holder keeps its id and the control inside
// it is a different element with no listeners on it. Returns the new control.
function rerenderField(id) {
    const holder = elements[id];
    const replacement = element(id + "-inner", "TEXTAREA");
    holder.inner = replacement;
    observers.forEach((fn) => fn());
    return replacement;
}
const status = elements["mc-llm-chat-status"];

function feed(samples) {
    // Whatever capture path was installed, hand it some audio.
    if (globalThis.workletNode && globalThis.workletNode.port.onmessage) {
        globalThis.workletNode.port.onmessage({data: new Float32Array(samples)});
    } else if (globalThis.processorNode && globalThis.processorNode.onaudioprocess) {
        globalThis.processorNode.onaudioprocess({
            inputBuffer: {getChannelData: () => new Float32Array(samples)},
        });
    }
}

// The row schedules its next poll with setTimeout rather than a fixed
// interval, so "poll again" is "move the clock and run what is due".
async function repaint() {
    NOW += 60000;
    await tick();
}

// Pointer moves and releases are listened for on the window, not the handle:
// pointer capture normally keeps them on the button, and where it is missing or
// lost these are what still end a recording. Firing them the same way the
// browser would is the only way to test that.
function fireWindow(name, detail) {
    const event = Object.assign({type: name, preventDefault() {}, cancelable: true,
                                 button: 0}, detail || {});
    (globalThis.windowHandlers[name] || []).forEach((fn) => fn(event));
}

// Slide-to-talk, as one call. Press on the handle at the left of the track,
// carry it past the engage threshold, and recording starts.
//
// A tick between the press and the movement, because a slide is not
// instantaneous and the script relies on that: pressing starts the readiness
// check, and the engagement at the end of the slide reads its answer instead of
// waiting for one.
async function engageMic(pointerId) {
    const id = pointerId === undefined ? 7 : pointerId;
    mic.fire("pointerdown", {pointerId: id, clientX: 0});
    await tick();
    fireWindow("pointermove", {pointerId: id, clientX: 400});
}

function releaseMic(pointerId, cancelled) {
    fireWindow(cancelled ? "pointercancel" : "pointerup",
               {pointerId: pointerId === undefined ? 7 : pointerId});
}

async function hold(ms, sampleCount, level) {
    await engageMic(7);
    await tick();
    feed(new Array(sampleCount === undefined ? 8000 : sampleCount)
         .fill(level === undefined ? 0.2 : level));
    NOW += ms;
    releaseMic(7);
    await tick();
}

function report(extra) {
    return Object.assign({
        reloads: reloads.length,
        requests: requests.map((r) => ({url: r.url, kind: r.kind,
                                        headers: r.headers,
                                        constraints: r.constraints,
                                        bodyLength: r.body && r.body.byteLength,
                                        bodyText: typeof r.body === "string" ? r.body : null})),
        micClasses: Array.from(mic.classList.names),
        micTransform: mic.style.transform || "",
        trackClasses: Array.from(micTrack.classList.names),
        sendWrites: send.classList.writes,
        stopWrites: stop.classList.writes,
        micLabel: mic.getAttribute("aria-label"),
        composer: message.value,
        sends: send.clicks || 0,
        tracks: tracks.map((t) => t.stopped),
        played: played.length,
        stopped: stopped.length,
        status: status.notice.textContent,
        scheduled: scheduled.map(function (item) {
            const buffer = item.source.buffer;
            const streamed = buffer && typeof buffer.getChannelData === "function";
            return {at: item.at,
                    length: streamed ? buffer.length : 0,
                    rate: streamed ? buffer.sampleRate : 0,
                    samples: streamed
                        ? Array.from(buffer.getChannelData(0).slice(0, 4)) : []};
        }),
        aborts: aborts.length,
        readerCancelled: readers.map((r) => r.cancelled),
        sendHidden: (elements["mc-llm-chat-send"].classList
                     .contains("mc-llm-voice-hidden")),
        stopHidden: (elements["mc-llm-chat-stop"].classList
                     .contains("mc-llm-voice-hidden")),
        // Visible and *actionable* are two claims. A Stop that Gradio has
        // rendered disabled is on screen during exactly the phase it cannot
        // stop -- and a disabled button dispatches no click at all, so the
        // browser's own capture-phase listener never runs either.
        stopDisabled: !!elements["mc-llm-chat-stop"].disabled
            || elements["mc-llm-chat-stop"].getAttribute("aria-disabled") === "true",
        pendingTimers: pending(),
        consoleErrors,
        consoleInfo,
        moduleLoads,
        listeners: Object.keys(mic.handlers).map((k) => [k, mic.handlers[k].length]),
        settings: {
            runtime: settingsParts.runtime.textContent,
            sttLine: settingsParts.sttLine.textContent,
            ttsLine: settingsParts.ttsLine.textContent,
            sttButton: settingsParts.sttButton.textContent,
            ttsButton: settingsParts.ttsButton.textContent,
            sttDisabled: settingsParts.sttButton.disabled,
            ttsDisabled: settingsParts.ttsButton.disabled,
            sttFailed: settingsParts.sttLine.classList.contains("mc-voice-failed"),
            sttLocalDisabled: settingsParts.sttLocal.disabled,
            sttLocalLabel: settingsParts.sttLocal.textContent,
            ttsLocalDisabled: settingsParts.ttsLocal.disabled,
            ttsLocalLabel: settingsParts.ttsLocal.textContent,
            runtimeLine: settingsParts.runtimeLine.textContent,
            runtimeButton: settingsParts.runtimeButton.textContent,
            runtimeDisabled: settingsParts.runtimeButton.disabled,
            chosenLabel: settingsParts.chosenLabel.textContent,
            tiers: Object.keys(settingsParts.tiers).reduce((found, id) => {
                const card = settingsParts.tiers[id];
                found[id] = {
                    install: card.installButton.textContent,
                    installDisabled: card.installButton.disabled,
                    use: card.useButton.textContent,
                    useDisabled: card.useButton.disabled,
                    state: card.state.textContent,
                    mark: card.mark.textContent,
                    chosen: card.classList.contains("mc-voice-tier-chosen"),
                    failed: card.state.classList.contains("mc-voice-failed"),
                    localLabel: card.local.textContent,
                };
                return found;
            }, {}),
        },
    }, extra || {});
}

await (async function () {
SCENARIO
})();
"""


DEFAULTS = {
    "SETTINGS_PRESENT": "false",
    "HIDDEN": "false",
    "SECURE": "true",
    "MICROPHONE": "true",
    "LEGACY": "false",
    "VOICES_PRESENT": "false",
    "VOICES_VISIBLE": "false",
    "PERMISSION": "true",
    "DENIAL": '"NotAllowedError"',
    "SAMPLE_RATE": "48000",
    "TRACK_LABEL": '"Built-in microphone"',
    "TRACK_RATE": "48000",
    "FIRST_OPEN_FAILS": "false",
    "CONTEXT_STATE": '"running"',
    "RESUME_WORKS": "true",
    "DECODE_WORKS": "true",
    "WORKLET_AVAILABLE": "true",
    # Whether `addModule` succeeds. False models a browser that refuses the
    # module -- a duplicate registration, a Content-Security-Policy that will not
    # take a blob: URL -- and the capture path must fall back rather than lose
    # the recording.
    "WORKLET_LOADS": "true",
    # Whether getUserMedia settles by itself. False holds it open until the
    # scenario calls `openMicrophoneNow()`, which is how a permission prompt
    # nobody has answered yet is modelled.
    "SLOW_OPEN": "false",
    "GRADIO_CONFIG": '{"root": ""}',
    "ANSWERS": json.dumps({
        "voice/tts-stream": {
            "headers": {"X-Model-Chain-Voice-Rate": "24000",
                        "X-Model-Chain-Voice-Turn": "T1"},
            # Three network reads whose boundaries fall wherever they like --
            # including one that ends on the first byte of a sample.
            "chunks": [[1, 0, 2, 0, 3], [0, 4, 0], [5, 0, 6, 0]],
        },
        "voice/cancel": {"json": {"ok": True, "cancelled": True}},
        "voice/runtime": {"json": {"ok": True, "engine": {"loaded": True, "state": "idle"}}},
        "voice/voices": {"json": {"ok": True, "voices": [], "default": "official:af_heart",
                                  "test_text": "This is a test of voice cloning.",
                                  "capacity": {"used": 0, "total": 32, "free": 32},
                                  "warnings": []}},
        "voice/cloning/status": {"json": {"ok": True, "state": "not_installed",
                                          "message": "Voice cloning is not installed.",
                                          "checks": [], "sources": {},
                                          "capacity": {"free": 32},
                                          "job": {"status": "idle", "active": False}}},
        "voice/status": {"json": {"ok": True, "ready": True, "stt_ready": True,
                                  "tts_ready": True, "runtime_ready": True,
                                  "auto_send": False, "auto_speak": False,
                                  "progress": {}, "runtime_message": "Installed",
                                  "summary": "Voice Chat is ready.",
                                  "stt_message": "Installed", "tts_message": "Installed",
                                  "not_ready_message": "Voice Chat is not set up."}},
        "voice/stt": {"json": {"ok": True, "text": "the quick brown fox",
                               "auto_send": False}},
        "voice/tts": {"audio": None},
        "voice/install": {"json": {"ok": True, "already": False}},
        "voice/engine/select": {"json": {"ok": True, "active": "sopro",
                                         "label": "Sopro V2", "engines": []}},
        "voice/models": {"json": {
            "ok": True, "kind": "stt", "chosen": "whisper-small-int8", "progress": {},
            "models": [
                {"id": "whisper-base-int8", "tier": "low", "tier_label": "Low",
                 "label": "Whisper Base", "summary": "Fastest.", "notes": "",
                 "about_label": "90 MB", "ram_label": "1.0 GB", "installed": False,
                 "message": "Not installed", "chosen": False, "sources": []},
                {"id": "whisper-small-int8", "tier": "medium", "tier_label": "Medium",
                 "label": "Whisper Small", "summary": "Balanced.", "notes": "",
                 "about_label": "250 MB", "ram_label": "1.5 GB", "installed": True,
                 "message": "Installed", "chosen": True, "sources": []},
                {"id": "whisper-medium-int8", "tier": "high", "tier_label": "High",
                 "label": "Whisper Medium", "summary": "Most accurate.", "notes": "",
                 "about_label": "800 MB", "ram_label": "3.0 GB", "installed": False,
                 "message": "Not installed", "chosen": False, "sources": []},
            ]}},
        "voice/profile": {"json": {
            "ok": True,
            "profile": {"speed": 1.0, "pitch": 0.0, "gain": 0.0, "pause": 0.0},
            "fields": ["speed", "pitch", "gain", "pause"],
            "summary": "Kokoro's own delivery",
            "controls": {
                "speed": {"label": "Speed", "unit": "x", "minimum": 0.5, "maximum": 2.0,
                          "step": 0.05, "default": 1.0, "decimals": 2, "help": ""},
                "pitch": {"label": "Pitch", "unit": " semitones", "minimum": -12.0,
                          "maximum": 12.0, "step": 0.5, "default": 0.0, "decimals": 1,
                          "help": ""},
                "gain": {"label": "Volume", "unit": " dB", "minimum": -12.0,
                         "maximum": 12.0, "step": 0.5, "default": 0.0, "decimals": 1,
                         "help": ""},
                "pause": {"label": "Pause between sentences", "unit": " ms",
                          "minimum": 0.0, "maximum": 1200.0, "step": 25.0,
                          "default": 0.0, "decimals": 0, "help": ""},
            }}},
    }),
}


def status_answer(**changes):
    """The status payload, with a few fields changed."""
    answers = json.loads(DEFAULTS["ANSWERS"])
    answers["voice/status"]["json"].update(changes)
    return answers


def run(scenario: str, **overrides) -> dict:
    """One scenario, in a real JavaScript engine, against the real script.

    Written to a file rather than passed with ``node -e``. The script under test
    is the whole of ``javascript/voice_chat.js``, and an argument vector is not
    an unbounded thing: on Linux a single argument is capped at 128 KiB, and the
    harness plus the script crossed that as the script grew. The failure that
    produces is ``OSError: Argument list too long`` from every test at once,
    which reads like the suite is broken rather than like a limit was reached --
    so it is not a limit this suite goes near any more. ``.mjs`` because that is
    how a file asks for the module semantics ``--input-type=module`` used to.
    """
    settings = dict(DEFAULTS)
    settings.update({key: value for key, value in overrides.items()})
    harness = HARNESS.replace("SOURCE", SCRIPT.read_text()).replace("SCENARIO", scenario)
    for key, value in settings.items():
        harness = harness.replace(key, value if isinstance(value, str) else json.dumps(value))
    with tempfile.TemporaryDirectory() as room:
        entry = pathlib.Path(room) / "scenario.mjs"
        entry.write_text(harness, encoding="utf-8")
        result = subprocess.run(["node", str(entry)],
                                capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def urls(found: dict) -> list[str]:
    return [request["url"] for request in found["requests"] if request.get("url")]


# --------------------------------------------------------------------------- #
# Where the requests go
# --------------------------------------------------------------------------- #


class TestTheDeploymentRoot:
    def test_requests_go_to_the_root_gradio_published(self):
        """R2-4. Not ``location.origin`` -- a WebUI mounted under ``--subpath``
        is a WebUI whose routes are not at the origin root, and a request built
        against the origin reaches the proxy in front of it."""
        found = run("console.log(JSON.stringify(report()));",
                    GRADIO_CONFIG='{"root": "/forge"}')
        for url in urls(found):
            assert url.startswith("/forge/model-chain/voice/"), url

    def test_a_root_deployment_still_works(self):
        found = run("await hold(900); console.log(JSON.stringify(report()));")
        assert any(url == "/model-chain/voice/stt" for url in urls(found))

    def test_a_trailing_slash_does_not_double_up(self):
        found = run("await hold(900); console.log(JSON.stringify(report()));",
                    GRADIO_CONFIG='{"root": "/forge/"}')
        assert all("//" not in url for url in urls(found))

    def test_without_a_published_root_it_falls_back_to_the_page_directory(self):
        """Still not the origin: the last resort is where this page is being
        served from, which under a subpath is the subpath."""
        found = run("await hold(900); console.log(JSON.stringify(report()));",
                    GRADIO_CONFIG="null")
        assert urls(found), "no request was made at all"
        assert all(url.startswith("/model-chain/voice/") for url in urls(found)), urls(found)

    def test_every_request_carries_the_page_token(self):
        found = run("await hold(900); console.log(JSON.stringify(report()));")
        for request in found["requests"]:
            if request.get("url"):
                assert request["headers"]["X-Model-Chain-Voice"] == "PAGE-TOKEN"


# --------------------------------------------------------------------------- #
# The gesture
# --------------------------------------------------------------------------- #


class TestSlideToTalk:
    """The composer gesture. Slide the handle to the right-hand end of its
    track and hold; release to transcribe.

    It replaced press-and-hold because a long press on Android is the operating
    system's gesture before it is a web page's -- it raises the context menu
    over the composer -- and because a recording is worth being deliberate
    about. Everything past the moment of engagement is unchanged, which is why
    most of what follows is the same assertion it always was.
    """

    def test_a_hold_records_and_posts_exactly_one_recording(self):
        found = run("await hold(900); console.log(JSON.stringify(report()));")
        posts = [r for r in found["requests"] if r.get("url", "").endswith("/stt")]
        assert len(posts) == 1
        assert posts[0]["bodyLength"] > 44, "the WAV had no samples in it"

    def test_a_second_slide_during_one_does_not_open_a_second_microphone(self):
        found = run("""
            await engageMic(1);
            await tick();
            await engageMic(2);
            await tick();
            console.log(JSON.stringify(report()));
        """)
        opens = [r for r in found["requests"] if r.get("kind") == "getUserMedia"]
        assert len(opens) == 1, "a second slide opened a second microphone"

    def test_the_microphone_is_closed_when_the_finger_comes_up(self):
        """The most alarming thing a page can do is leave the recording
        indicator on."""
        found = run("await hold(900); console.log(JSON.stringify(report()));")
        assert found["tracks"], "no microphone was ever opened"
        assert all(found["tracks"]), "a MediaStream track was left running"

    def test_pointercancel_discards_the_recording_and_closes_the_microphone(self):
        found = run("""
            await engageMic(3);
            await tick();
            feed(new Array(8000).fill(0.2));
            NOW += 900;
            releaseMic(3, true);
            await tick();
            console.log(JSON.stringify(report()));
        """)
        assert not any(r.get("url", "").endswith("/stt") for r in found["requests"])
        assert all(found["tracks"])

    def test_a_tap_does_not_reach_the_server(self):
        """Under 250 ms is somebody finding out what the button does."""
        found = run("await hold(120); console.log(JSON.stringify(report()));")
        assert not any(r.get("url", "").endswith("/stt") for r in found["requests"])
        assert "Hold at the right-hand end" in found["status"]

    def test_the_finger_is_captured_so_sliding_off_still_ends_the_recording(self):
        found = run("""
            await engageMic(11);
            await tick();
            console.log(JSON.stringify(report({captured: mic.captured})));
        """)
        assert found["captured"] == 11

    def test_a_sixty_second_cap_is_armed_and_does_not_fire_early(self):
        found = run("""
            await engageMic(4);
            await tick();
            console.log(JSON.stringify(report({armed: pending()})));
        """)
        assert found["armed"] >= 1

    def test_the_cap_stops_the_recording_by_itself(self):
        found = run("""
            await engageMic(5);
            await tick();
            feed(new Array(8000).fill(0.2));
            NOW += 60000;
            await tick(12);
            console.log(JSON.stringify(report()));
        """)
        assert any(r.get("url", "").endswith("/stt") for r in found["requests"])

    def test_engagement_opens_the_microphone_without_a_round_trip(self):
        """The gesture is not held up by the WebUI. What it enters is OPENING:
        the microphone has been asked for and nothing has been heard yet."""
        found = run("""
            await engageMic(6);
            console.log(JSON.stringify(report()));
        """)
        assert any(r.get("kind") == "getUserMedia" for r in found["requests"])
        assert "mc-llm-voice-opening" in found["micClasses"]
        assert "mc-llm-voice-recording" not in found["micClasses"]
        assert found["micLabel"] == "Opening microphone"

    def test_red_begins_at_the_first_sample_and_not_before(self):
        """The whole of section 7.3. Permission granted is not recording, a
        connected node is not recording; audio arriving is recording, and on a
        phone those can be a second apart."""
        found = run("""
            await engageMic(6);
            await tick();
            const opening = {mic: Array.from(mic.classList.names),
                             track: Array.from(micTrack.classList.names),
                             label: mic.getAttribute("aria-label")};
            feed(new Array(800).fill(0.2));
            await tick();
            console.log(JSON.stringify(report({opening})));
        """)
        assert "mc-llm-voice-recording" not in found["opening"]["mic"]
        assert "mc-llm-voice-armed" not in found["opening"]["track"]
        assert found["opening"]["label"] == "Opening microphone"
        assert "mc-llm-voice-recording" in found["micClasses"]
        assert "mc-llm-voice-armed" in found["trackClasses"]
        assert found["micLabel"] == "Recording — release to transcribe"

    def test_the_control_returns_to_idle_afterwards(self):
        found = run("await hold(900); console.log(JSON.stringify(report()));")
        assert "mc-llm-voice-recording" not in found["micClasses"]
        assert found["micLabel"] == "Dictate — slide right and hold, or hold Space"


class TestTheSlideItself:
    """What the gesture is, as opposed to what happens once it has engaged."""

    def test_a_press_that_never_slides_uploads_nothing(self):
        """The whole point of the shape, restated for pre-roll.

        A finger that lands on the handle -- a scroll that started there, a
        mis-tap on a phone -- may now open the microphone, because the slide is
        the window everything before the first sample has to happen in. What it
        must never do is turn any of that into an utterance: nothing is
        uploaded, no transcription is asked for, the tracks stop, and the
        samples are dropped.
        """
        found = run("""
            mic.fire("pointerdown", {pointerId: 21, clientX: 0});
            await tick(4);
            feed(new Array(8000).fill(0.2));
            await tick();
            releaseMic(21);
            await tick(4);
            console.log(JSON.stringify(report()));
        """)
        assert not [r for r in found["requests"] if (r.get("url") or "").endswith("/stt")]
        assert found["tracks"] and all(found["tracks"]), "a microphone was left open"
        assert "mc-llm-voice-recording" not in found["micClasses"]
        assert "Slide the microphone all the way to the right" in found["status"]

    def test_a_slide_short_of_the_end_does_not_engage(self):
        """Nine tenths of the travel, and this is eight. The threshold is not
        the whole width because a fingertip and the pointer position it reports
        disagree by a few pixels; it is not half of it either, because a gesture
        that engages halfway is a gesture nobody trusts."""
        found = run("""
            mic.fire("pointerdown", {pointerId: 22, clientX: 0});
            await tick(4);
            fireWindow("pointermove", {pointerId: 22, clientX: 30});
            feed(new Array(8000).fill(0.2));
            await tick();
            console.log(JSON.stringify(report()));
        """)
        # The microphone is open -- that is what the press is for now -- and
        # samples are being kept. What has not happened is the commitment.
        assert "mc-llm-voice-recording" not in found["micClasses"]
        assert "mc-llm-voice-buffering" in found["micClasses"]
        assert not [r for r in found["requests"] if (r.get("url") or "").endswith("/stt")]

    def test_the_handle_follows_the_finger_and_is_put_back_on_release(self):
        found = run("""
            mic.fire("pointerdown", {pointerId: 23, clientX: 0});
            await tick();
            fireWindow("pointermove", {pointerId: 23, clientX: 20});
            const midway = mic.style.transform;
            releaseMic(23);
            await tick();
            console.log(JSON.stringify(report({midway})));
        """)
        assert found["midway"] == "translateX(20px)"
        assert found["micTransform"] == "", "the handle did not return to the left"

    def test_the_track_is_marked_while_it_is_recording_and_after(self):
        """Two classes now, because they are two statements. Reaching the end of
        the track is `engaged` and is not red; `armed` is red and waits for a
        sample."""
        found = run("""
            await engageMic(24);
            await tick();
            const pinned = Array.from(micTrack.classList.names);
            feed(new Array(8000).fill(0.2));
            await tick();
            const armed = Array.from(micTrack.classList.names);
            NOW += 900;
            releaseMic(24);
            await tick();
            console.log(JSON.stringify(report({pinned, armed})));
        """)
        assert "mc-llm-voice-engaged" in found["pinned"]
        assert "mc-llm-voice-armed" not in found["pinned"]
        assert "mc-llm-voice-armed" in found["armed"]
        assert "mc-llm-voice-armed" not in found["trackClasses"]
        assert "mc-llm-voice-engaged" not in found["trackClasses"]

    def test_the_handle_is_pinned_at_the_end_once_it_has_engaged(self):
        """A finger drifting back down the track is a finger drifting, not a
        decision. A microphone that stopped and started under one continuous
        hold would be worse than the control this replaced."""
        found = run("""
            await engageMic(25);
            await tick();
            fireWindow("pointermove", {pointerId: 25, clientX: 5});
            await tick();
            console.log(JSON.stringify(report()));
        """)
        assert found["micTransform"] != ""
        assert "mc-llm-voice-opening" in found["micClasses"]

    def test_holding_space_records_and_releasing_it_sends(self):
        """A control that can only be dragged is a control somebody using a
        keyboard cannot use at all."""
        found = run("""
            mic.fire("keydown", {key: " "});
            await tick();
            feed(new Array(8000).fill(0.2));
            NOW += 900;
            mic.fire("keyup", {key: " "});
            await tick();
            console.log(JSON.stringify(report()));
        """)
        posts = [r for r in found["requests"] if r.get("url", "").endswith("/stt")]
        assert len(posts) == 1

    def test_a_pointer_that_leaves_the_window_ends_the_recording(self):
        """An incoming call, a task switch. The failure this must not have is a
        microphone left open."""
        found = run("""
            await engageMic(26);
            await tick();
            feed(new Array(8000).fill(0.2));
            NOW += 900;
            fireWindow("pointercancel", {pointerId: 26});
            await tick();
            console.log(JSON.stringify(report()));
        """)
        assert found["tracks"], "no microphone was ever opened"
        assert all(found["tracks"]), "a MediaStream track was left running"
        assert not any(r.get("url", "").endswith("/stt") for r in found["requests"])


class TestWhenItCannotRecord:
    def test_t_js_13_an_insecure_page_is_not_refused_by_this_extension(self):
        """T-JS-13. The V1 script asked ``isSecureContext`` and refused first.

        Section 39 removes exactly that: Voice Chat may not reject a page for
        being HTTP. Where the browser has nevertheless exposed getUserMedia --
        localhost, a tunnel, a browser flag, a future policy -- the microphone
        is opened and the browser makes the decision that is genuinely its own.
        """
        found = run("await hold(900); console.log(JSON.stringify(report()));",
                    SECURE="false")
        assert "HTTPS" not in found["status"], found["status"]
        opens = [r for r in found["requests"] if r.get("kind") == "getUserMedia"]
        assert len(opens) == 1, "an insecure page was refused by the extension itself"

    def test_t_js_14_a_browser_without_getusermedia_reports_capability(self):
        """T-JS-14. Not "HTTPS is required" -- that is a claim about a policy
        this extension no longer has, and it is not reliably the reason
        either."""
        found = run("await hold(900); console.log(JSON.stringify(report()));",
                    MICROPHONE="false")
        assert "did not make microphone capture available" in found["status"]
        assert "HTTPS" not in found["status"]
        assert found["consoleErrors"] == []

    def test_an_insecure_page_with_no_capture_is_told_the_actual_reason(self):
        """The report that came back from a phone: "my browser did not request
        microphone". It had not, and this is why -- Chrome does not expose
        getUserMedia at all on an origin it does not trust, so there was nothing
        to prompt with. "Capture is not available" is true and useless; the
        origin being insecure is true and fixable, so that is what is said."""
        found = run("await hold(900); console.log(JSON.stringify(report()));",
                    MICROPHONE="false", SECURE="false")
        assert "only opens a microphone on a secure page" in found["status"]
        assert "HTTPS" in found["status"]
        assert "unsafely-treat-insecure-origin-as-secure" in found["status"]

    def test_the_legacy_entry_point_is_used_when_mediadevices_is_missing(self):
        """Some Android WebViews and in-app browsers have the prefixed callback
        form and no `mediaDevices`. Refusing there was refusing a browser that
        could have recorded."""
        found = run("await hold(900); console.log(JSON.stringify(report()));",
                    MICROPHONE="false", LEGACY="true")
        opens = [r for r in found["requests"] if r.get("kind") == "getUserMedia"]
        assert len(opens) == 1, "the prefixed getUserMedia was never called"
        posts = [r for r in found["requests"] if r.get("url", "").endswith("/stt")]
        assert len(posts) == 1

    def test_the_microphone_is_opened_before_the_status_answer_when_nothing_is_known(self):
        """Why the order matters. A browser raises a permission prompt while it
        still considers a user gesture in progress; a promise chain that has
        already awaited a round trip to the WebUI is past that on a phone. So
        with no cached answer the microphone is asked for first and closed again
        if the answer turns out to be no."""
        found = run("""
            // No tick between the press and the slide: nothing has come back
            // from the status route yet, which is the first gesture on a page.
            mic.fire("pointerdown", {pointerId: 31, clientX: 0});
            fireWindow("pointermove", {pointerId: 31, clientX: 400});
            console.log(JSON.stringify(report({
                askedBeforeAnyAnswer: requests.filter(
                    (r) => r.kind === "getUserMedia").length,
            })));
        """)
        assert found["askedBeforeAnyAnswer"] == 1

    def test_a_denied_permission_says_so_and_does_not_retry(self):
        found = run("await hold(900); console.log(JSON.stringify(report()));",
                    PERMISSION="false")
        assert "not allowed by the browser or device" in found["status"]
        opens = [r for r in found["requests"] if r.get("kind") == "getUserMedia"]
        assert len(opens) == 1, "a denied permission was asked for again"

    def test_each_browser_failure_maps_to_its_own_sentence(self):
        """Section 39's map. "There is no microphone" and "permission was
        refused" have nothing in common except that no audio arrived, and
        collapsing them costs the user the one thing they could act on."""
        for name, wanted in (("NotAllowedError", "not allowed by the browser"),
                             ("NotFoundError", "No microphone is available"),
                             ("NotReadableError", "could not be opened")):
            found = run("await hold(900); console.log(JSON.stringify(report()));",
                        PERMISSION="false", DENIAL='"%s"' % name)
            assert wanted in found["status"], (name, found["status"])

    def test_a_missing_setup_is_reported_and_records_nothing(self):
        """The first gesture on a page cannot know: the answer arrives while the
        pre-roll is being captured, and when it says no the capture is dropped
        and the tracks are stopped. The *second* gesture knows, and never opens
        anything at all."""
        answers = json.loads(DEFAULTS["ANSWERS"])
        answers["voice/status"]["json"].update({"ready": False, "stt_ready": False})
        found = run("""
            await hold(900);
            const first = requests.filter((r) => r.kind === "getUserMedia").length;
            await hold(900);
            console.log(JSON.stringify(report({first})));
        """, ANSWERS=json.dumps(answers))
        assert "not set up" in found["status"]
        assert "mc-llm-voice-error" in found["micClasses"]
        assert not [r for r in found["requests"] if (r.get("url") or "").endswith("/stt")]
        assert found["tracks"] and all(found["tracks"]), "a microphone was left open"
        opened = len([r for r in found["requests"] if r.get("kind") == "getUserMedia"])
        assert opened == found["first"], "the second gesture opened a microphone anyway"

    def test_it_will_not_record_while_a_reply_is_generating(self):
        """Busy is Python's run-state value now rather than the last CSS the
        panel happened to apply -- section 25. The button's visibility is still
        the fallback for a page built before that field existed."""
        found = run("""
            runState.value = "llm";
            await hold(900);
            console.log(JSON.stringify(report()));
        """)
        assert "Wait for the reply" in found["status"]

    def test_the_button_is_still_the_fallback_where_there_is_no_run_state(self):
        found = run("""
            delete elements["mc-llm-chat-voice-run-state"];
            elements["mc-llm-chat-stop"].offsetParent = {};
            await hold(900);
            console.log(JSON.stringify(report()));
        """)
        assert "Wait for the reply" in found["status"]
        assert not any(r.get("kind") == "getUserMedia" for r in found["requests"])


# --------------------------------------------------------------------------- #
# What arrives on the wire
# --------------------------------------------------------------------------- #


def deferred_status(**changes):
    """The default answers with the status route held open until asked."""
    answers = json.loads(DEFAULTS["ANSWERS"])
    answers["voice/status"]["json"].update(changes)
    answers["voice/status"]["defer"] = True
    return json.dumps(answers)


class TestOpeningAndRecording:
    """The nine promise orderings a real device produces, and what each must do.

    All of them are the same question asked from different angles: what does the
    control say, and is what it says true? The old answer was that the far right
    of the track meant red, which was a statement about a gesture dressed up as
    a statement about a microphone.
    """

    def test_m1_samples_are_accepted_before_the_status_check_answers(self):
        """Sample acceptance used to wait for an HTTP round trip to the WebUI,
        and every callback that arrived first was dropped -- audio the microphone
        really had captured, thrown away because a status check was in flight."""
        found = run("""
            await engageMic(30);
            await tick();
            feed(new Array(8000).fill(0.2));
            await tick();
            const early = {mic: Array.from(mic.classList.names)};
            await answerStatus();
            NOW += 900;
            releaseMic(30);
            await tick(6);
            console.log(JSON.stringify(report({early})));
        """, ANSWERS=deferred_status())
        assert "mc-llm-voice-recording" in found["early"]["mic"], (
            "red waited for the WebUI rather than for audio")
        posts = [r for r in found["requests"] if r.get("url", "").endswith("/stt")]
        assert len(posts) == 1
        assert posts[0]["bodyLength"] > 44, "the samples captured first were dropped"

    def test_m2_a_release_while_opening_never_becomes_red(self):
        """The slow-phone case. getUserMedia outliving the gesture must not
        produce a recording, a red control, or a transcription."""
        found = run("""
            await engageMic(31);
            await tick();
            releaseMic(31);
            await tick();
            const afterRelease = Array.from(mic.classList.names);
            await openMicrophoneNow();
            feed(new Array(8000).fill(0.2));
            await tick(4);
            console.log(JSON.stringify(report({afterRelease})));
        """, SLOW_OPEN="true")
        assert "mc-llm-voice-recording" not in found["afterRelease"]
        assert "mc-llm-voice-recording" not in found["micClasses"]
        assert found["tracks"] and all(found["tracks"]), (
            "a MediaStream that arrived after the release was left running")
        assert not [r for r in found["requests"] if r.get("url", "").endswith("/stt")]

    def test_m3_a_late_not_ready_answer_stops_and_discards(self):
        """Accepting samples early is only safe because the answer, when it
        comes, is still obeyed."""
        found = run("""
            await engageMic(32);
            await tick();
            feed(new Array(8000).fill(0.2));
            await tick();
            await answerStatus();
            NOW += 900;
            releaseMic(32);
            await tick(6);
            console.log(JSON.stringify(report()));
        """, ANSWERS=deferred_status(ready=False,
                                     not_ready_message="Install both models."))
        assert all(found["tracks"]), "the microphone was left open"
        assert "mc-llm-voice-recording" not in found["micClasses"]
        assert not [r for r in found["requests"] if r.get("url", "").endswith("/stt")]
        assert "Install both models." in found["status"]

    def test_m4_two_utterances_load_the_module_once(self):
        """`registerProcessor` registers a name in the context's worklet scope.
        Loading the module again for the same context is a duplicate
        registration a browser may refuse -- and is in any case a Blob, an
        object URL and a module load per press of the microphone."""
        found = run("""
            await hold(900);
            await tick(4);
            await hold(900);
            await tick(4);
            console.log(JSON.stringify(report()));
        """)
        assert found["moduleLoads"] == 1
        posts = [r for r in found["requests"] if r.get("url", "").endswith("/stt")]
        assert len(posts) == 2, "the second recording was lost"

    def test_m5_a_worklet_that_will_not_load_still_records(self):
        """And red still means the first sample, whichever path produced it."""
        found = run("""
            await engageMic(33);
            await tick(4);
            feed(new Array(8000).fill(0.2));
            await tick();
            const armed = Array.from(mic.classList.names);
            NOW += 900;
            releaseMic(33);
            await tick(6);
            console.log(JSON.stringify(report({armed})));
        """, WORKLET_LOADS="false")
        assert "mc-llm-voice-recording" in found["armed"]
        posts = [r for r in found["requests"] if r.get("url", "").endswith("/stt")]
        assert len(posts) == 1 and posts[0]["bodyLength"] > 44

    def test_m6_the_minimum_hold_is_measured_from_the_first_sample(self):
        """Eight hundred milliseconds opening a Bluetooth microphone and three
        hundred of speech is a recording, not a tap. Charging the device's wait
        to the user's allowance is how a real utterance got refused."""
        found = run("""
            await engageMic(34);
            await tick();
            NOW += 800;
            feed(new Array(8000).fill(0.2));
            await tick();
            NOW += 300;
            releaseMic(34);
            await tick(6);
            console.log(JSON.stringify(report()));
        """)
        posts = [r for r in found["requests"] if r.get("url", "").endswith("/stt")]
        assert len(posts) == 1, found["status"]
        assert "Hold at the right-hand end" not in found["status"]

    def test_a_genuinely_short_hold_is_still_refused(self):
        """The guard is kept; only its clock moved -- and this is the direction
        the old clock got wrong. Eight hundred milliseconds of opening and eighty
        of speech measured from the gesture is 880 ms, comfortably past the
        minimum, and what it transcribes is a tap."""
        found = run("""
            await engageMic(35);
            await tick();
            NOW += 800;
            feed(new Array(8000).fill(0.2));
            await tick();
            NOW += 80;
            releaseMic(35);
            await tick(6);
            console.log(JSON.stringify(report()));
        """)
        assert not [r for r in found["requests"] if r.get("url", "").endswith("/stt")]
        assert "Hold at the right-hand end" in found["status"]

    def test_a_release_before_any_sample_says_nothing_about_being_too_short(self):
        """Section 7.5. There was no recording, so "that was too short" would be
        a sentence about the user when the wait was the device's."""
        found = run("""
            await engageMic(36);
            await tick();
            releaseMic(36);
            await tick(4);
            console.log(JSON.stringify(report()));
        """, SLOW_OPEN="true")
        assert "Hold at the right-hand end" not in found["status"]

    def test_m7_a_known_not_ready_state_never_opens_the_microphone(self):
        """A permission prompt raised for a feature that cannot run is a prompt
        asked for nothing. "Known" is the operative word: this is the cheap
        synchronous check the press makes, and it can only refuse on an answer
        that is already in hand."""
        found = run("""
            // The first gesture fills the readiness cache and is refused when
            // the answer arrives.
            mic.fire("pointerdown", {pointerId: 37, clientX: 0});
            await tick(6);
            releaseMic(37);
            await tick(4);
            const before = requests.filter((r) => r.kind === "getUserMedia").length;
            // The second one knows before it touches anything.
            mic.fire("pointerdown", {pointerId: 38, clientX: 0});
            await tick(4);
            fireWindow("pointermove", {pointerId: 38, clientX: 400});
            await tick(4);
            console.log(JSON.stringify(report({before})));
        """, ANSWERS=status_answer(ready=False, not_ready_message="Install both models."))
        opened = len([r for r in found["requests"] if r.get("kind") == "getUserMedia"])
        assert opened == found["before"], "a known-unavailable Voice Chat opened a microphone"
        assert "Install both models." in found["status"]

    def test_m8_engagement_does_not_wait_for_the_worklet_to_finish_loading(self):
        """A browser raises a permission prompt only while it still considers a
        user gesture in progress. A chain that has awaited a module load is past
        that on a phone, which is how a page that could have asked reported that
        capture was unavailable instead."""
        found = run("""
            mic.fire("pointerdown", {pointerId: 38, clientX: 0});
            fireWindow("pointermove", {pointerId: 38, clientX: 400});
            // No await at all between the gesture and this line: whatever
            // getUserMedia calls exist were made in the same task.
            const asked = requests.filter((r) => r.kind === "getUserMedia").length;
            await tick(6);
            console.log(JSON.stringify(report({asked})));
        """)
        assert found["asked"] == 1

    def test_m9_a_wall_clock_that_jumps_does_not_change_the_hold(self):
        """Durations come from performance.now(). A machine that synchronised
        its time mid-utterance used to be able to make a held button look like
        it had been held for a negative length of time."""
        found = run("""
            await engageMic(39);
            await tick();
            feed(new Array(8000).fill(0.2));
            await tick();
            NOW += 900;
            WALL -= 60000;
            releaseMic(39);
            await tick(6);
            console.log(JSON.stringify(report()));
        """)
        posts = [r for r in found["requests"] if r.get("url", "").endswith("/stt")]
        assert len(posts) == 1, found["status"]

    def test_the_press_opens_the_microphone_and_prepares_the_worklet_together(self):
        """Revision 4's boundary, and it has moved. The gesture is what asks for
        the microphone; the module load runs alongside it and never in front of
        it. What the press does not do is commit anything."""
        found = run("""
            mic.fire("pointerdown", {pointerId: 40, clientX: 0});
            const asked = requests.filter((r) => r.kind === "getUserMedia").length;
            await tick(4);
            console.log(JSON.stringify(report({asked})));
        """)
        assert found["asked"] == 1, "the microphone was not asked for in the gesture's task"
        assert found["moduleLoads"] == 1
        assert "mc-llm-voice-recording" not in found["micClasses"]

    def test_the_slide_and_the_engagement_share_one_status_request(self):
        """Two callers a few hundred milliseconds apart asking the same
        question. On a phone over a mesh VPN the second one is a round trip
        standing between a gesture and a permission prompt."""
        found = run("""
            mic.fire("pointerdown", {pointerId: 41, clientX: 0});
            fireWindow("pointermove", {pointerId: 41, clientX: 400});
            await tick(6);
            console.log(JSON.stringify(report()));
        """, ANSWERS=deferred_status())
        asked = [r for r in found["requests"] if r.get("url", "").endswith("voice/status")]
        assert len(asked) == 1, asked

    def test_a_microphone_that_never_opens_gives_the_control_back(self):
        """The other half of splitting the two timeouts. A getUserMedia that
        never settles -- an unanswered prompt, a device another application has
        exclusive -- must not leave the handle pinned at the right for ever."""
        found = run("""
            await engageMic(42);
            await tick();
            NOW += 20000;
            await tick(6);
            console.log(JSON.stringify(report()));
        """, SLOW_OPEN="true")
        assert found["micTransform"] == "", "the handle stayed at the recording end"
        assert "mc-llm-voice-recording" not in found["micClasses"]
        assert "mc-llm-voice-armed" not in found["trackClasses"]
        assert "The microphone could not be opened." in found["status"]

    def test_the_recording_cap_is_armed_at_the_first_sample(self):
        """Section 7.7. A second spent opening a Bluetooth microphone is not a
        second of the user's sixty."""
        found = run("""
            await engageMic(43);
            await tick();
            // Fifteen seconds of a device waking up, then the first sample.
            NOW += 15000;
            await tick(2);
            feed(new Array(8000).fill(0.2));
            await tick();
            // Sixty-five seconds since the gesture, fifty since the first
            // sample. A cap counted from the gesture has fired by now.
            NOW += 50000;
            await tick(6);
            const stillRecording = mic.classList.names.has("mc-llm-voice-recording");
            const sent = requests.filter((r) => (r.url || "").endsWith("/stt")).length;
            // Seventy-seven seconds since the gesture, sixty-two of recording.
            NOW += 12000;
            await tick(12);
            console.log(JSON.stringify(report({stillRecording, sent})));
        """)
        assert found["stillRecording"], "the wait was charged to the recording allowance"
        assert found["sent"] == 0
        posts = [r for r in found["requests"] if r.get("url", "").endswith("/stt")]
        assert len(posts) == 1, "the cap never stopped the recording"
        assert found["micTransform"] == "", "the handle stayed at the recording end"

    def test_the_capture_report_carries_durations_and_no_device(self):
        """§8.6. Relative durations, which capture path was used, and what
        became of the recording. No device label, no level, no samples: the
        console line above already says what was *heard*, and this says only how
        long it took to start hearing it."""
        found = run("""
            await hold(900);
            console.log(JSON.stringify(report()));
        """)
        rows = _captures(found)
        assert len(rows) == 1, rows
        row = rows.pop()
        assert row["result"] == "sent"
        assert row["graph"] in ("worklet", "script")
        assert row["recorded_ms"] == 900
        assert "Built-in microphone" not in json.dumps(row)
        assert "turn" not in row

    def test_an_abandoned_gesture_reports_that_it_was_discarded(self):
        found = run("""
            await engageMic(44);
            await tick();
            NOW += 40;
            releaseMic(44);
            await tick(4);
            console.log(JSON.stringify(report()));
        """)
        rows = _captures(found)
        assert rows and rows[0]["result"] == "discarded", rows
        assert not [r for r in found["requests"] if (r.get("url") or "").endswith("/stt")]


class TestPreRollCapture:
    """The lead-in, and the seven ways it must not misbehave.

    Everything before the first sample -- permission, opening the device, the
    stream, assembling the graph -- used to sit between the slide finishing and
    the recording starting, so a user who began talking as they slid lost the
    first word of it. The slide is now the window those things happen in.

    What that changes is when the microphone is touched. What it deliberately
    does not change is when anything leaves this page: the gesture still decides
    that, and a gesture that is abandoned uploads nothing.
    """

    def test_m1_samples_captured_before_engagement_are_part_of_the_utterance(self):
        """Section 5.6: the committed utterance begins at the first retained
        sample, because that pre-roll is deliberately part of what was said."""
        found = run("""
            mic.fire("pointerdown", {pointerId: 50, clientX: 0});
            await tick(4);
            // Talking while sliding.
            feed(new Array(8000).fill(0.2));
            await tick();
            NOW += 300;
            fireWindow("pointermove", {pointerId: 50, clientX: 400});
            await tick();
            feed(new Array(8000).fill(0.2));
            NOW += 600;
            releaseMic(50);
            await tick(4);
            console.log(JSON.stringify(report()));
        """)
        posts = [r for r in found["requests"] if (r.get("url") or "").endswith("/stt")]
        assert len(posts) == 1, found["status"]
        # Both blocks are in the upload: 16000 samples of 48 kHz resampled to
        # 16 kHz is about 5333 frames, so ten and a half thousand bytes.
        assert posts[0]["bodyLength"] > 9000, posts[0]["bodyLength"]
        row = _captures(found).pop()
        assert row["preroll_ms"] == 300, row
        assert row["result"] == "sent"

    def test_m2_an_abandoned_gesture_uploads_nothing(self):
        found = run("""
            mic.fire("pointerdown", {pointerId: 51, clientX: 0});
            await tick(4);
            feed(new Array(8000).fill(0.2));
            await tick();
            NOW += 900;
            releaseMic(51);
            await tick(4);
            console.log(JSON.stringify(report()));
        """)
        assert not [r for r in found["requests"] if (r.get("url") or "").endswith("/stt")]
        assert found["tracks"] and all(found["tracks"]), "a microphone was left open"
        row = _captures(found).pop()
        assert row["result"] == "discarded"
        assert row["preroll_ms"] == 0, "an abandoned gesture reported an engagement"

    def test_m3_a_pending_worklet_does_not_delay_an_open_microphone(self):
        """Section 5.4. The preferred graph is preferred, not waited for: a
        module load that has not finished when the stream arrives gets the
        ScriptProcessor for this utterance and finishes for the next one."""
        found = run("""
            await hold(900);
            console.log(JSON.stringify(report()));
        """, WORKLET_LOADS="false")
        posts = [r for r in found["requests"] if (r.get("url") or "").endswith("/stt")]
        assert len(posts) == 1 and posts[0]["bodyLength"] > 44
        assert _captures(found).pop()["graph"] == "script"

    def test_m4_a_stream_that_arrives_after_the_gesture_is_stopped_at_once(self):
        """Section 5.5. Release, then the permission prompt is answered. What
        comes back is attached to nothing and stopped immediately."""
        found = run("""
            mic.fire("pointerdown", {pointerId: 52, clientX: 0});
            await tick();
            releaseMic(52);
            await tick();
            const beforeStream = report().tracks.filter(Boolean).length;
            await openMicrophoneNow();
            feed(new Array(8000).fill(0.2));
            await tick(4);
            console.log(JSON.stringify(report({beforeStream})));
        """, SLOW_OPEN="true")
        assert found["tracks"], "no microphone was asked for at all"
        assert all(found["tracks"]), "a late stream was left running"
        assert not [r for r in found["requests"] if (r.get("url") or "").endswith("/stt")]
        assert "mc-llm-voice-recording" not in found["micClasses"]

    def test_m5_tts_is_silenced_before_any_pre_roll_is_accepted(self):
        """The assistant's own loudspeaker is the thing most likely to end up in
        a microphone opened beside it, so it is stopped at the press now rather
        than at the far end of the track."""
        found = run("""
            await speak('T1', 4);
            const playing = report().played;
            mic.fire("pointerdown", {pointerId: 53, clientX: 0});
            const stoppedAtPress = report().stopped;
            await tick(4);
            console.log(JSON.stringify(report({playing, stoppedAtPress})));
        """, ANSWERS=stream_answers(seconds=1.0, count=6, stall=True))
        assert found["playing"] >= 1, "nothing was playing to interrupt"
        assert found["stoppedAtPress"] >= 1, "the speaker was still going during pre-roll"
        assert any(r["url"].endswith("voice/cancel") for r in found["requests"])

    def test_m6_no_track_survives_any_exit(self):
        """Release, cancel, and a readiness answer that says no."""
        for scenario in ("releaseMic(54); await tick(4);",
                         "fireWindow('pointercancel', {pointerId: 54}); await tick(4);",
                         "await answerStatus();"):
            found = run("""
                mic.fire("pointerdown", {pointerId: 54, clientX: 0});
                await tick(4);
                feed(new Array(8000).fill(0.2));
                await tick();
                SCENARIO_STEP
                console.log(JSON.stringify(report()));
            """.replace("SCENARIO_STEP", scenario),
                        ANSWERS=deferred_status(ready=False,
                                                not_ready_message="Install both models."))
            assert found["tracks"], "no microphone was opened at all: " + scenario
            assert all(found["tracks"]), "a track was left running: " + scenario

    def test_a_gesture_that_only_ever_buffers_is_still_bounded(self):
        """The two bounds swap at the first sample. Before it, the risk is a
        microphone that never opens; after it, a microphone that stays open --
        and a finger resting on the handle for a minute is the second one."""
        found = run("""
            mic.fire("pointerdown", {pointerId: 56, clientX: 0});
            await tick(4);
            feed(new Array(8000).fill(0.2));
            await tick();
            const buffering = Array.from(mic.classList.names);
            NOW += 61000;
            await tick(8);
            console.log(JSON.stringify(report({buffering})));
        """)
        assert "mc-llm-voice-buffering" in found["buffering"]
        assert found["tracks"] and all(found["tracks"]), "a microphone was held open"
        assert not [r for r in found["requests"] if (r.get("url") or "").endswith("/stt")]
        assert _captures(found).pop()["result"] == "discarded"

    def test_the_opening_timeout_stops_covering_a_microphone_that_opened(self):
        """It bounds a device that never produced anything, and hands over the
        moment one does. A gesture buffering happily at twenty-one seconds must
        not be torn down by the timeout that was waiting for it."""
        found = run("""
            mic.fire("pointerdown", {pointerId: 57, clientX: 0});
            await tick(4);
            feed(new Array(8000).fill(0.2));
            await tick();
            NOW += 21000;
            await tick(8);
            const stillOpen = Array.from(mic.classList.names);
            fireWindow("pointermove", {pointerId: 57, clientX: 400});
            await tick();
            NOW += 900;
            releaseMic(57);
            await tick(4);
            console.log(JSON.stringify(report({stillOpen})));
        """)
        assert "mc-llm-voice-buffering" in found["stillOpen"], found["stillOpen"]
        posts = [r for r in found["requests"] if (r.get("url") or "").endswith("/stt")]
        assert len(posts) == 1, found["status"]

    def test_m7_buffering_never_claims_to_be_recording(self):
        """Sample capture and utterance commitment are different states, and the
        control says which one it is in -- in the colour and, for anybody who
        cannot read a colour, in the label."""
        found = run("""
            mic.fire("pointerdown", {pointerId: 55, clientX: 0});
            await tick(4);
            feed(new Array(8000).fill(0.2));
            await tick();
            console.log(JSON.stringify(report({
                label: mic.getAttribute("aria-label"),
                track: Array.from(micTrack.classList.names),
            })));
        """)
        assert "mc-llm-voice-buffering" in found["micClasses"]
        assert "mc-llm-voice-recording" not in found["micClasses"]
        assert "mc-llm-voice-armed" not in found["track"]
        assert found["label"] == "Microphone open — slide to record"

    def test_the_keyboard_route_opens_and_commits_in_one_gesture(self):
        """There is no slide to hold Space through, so the press and the
        engagement happen in one task and the pre-roll is whatever the device
        manages in it. What must not change is that it records and sends."""
        found = run("""
            mic.fire("keydown", {key: " "});
            await tick(4);
            feed(new Array(8000).fill(0.2));
            NOW += 900;
            mic.fire("keyup", {key: " "});
            await tick(4);
            console.log(JSON.stringify(report()));
        """)
        posts = [r for r in found["requests"] if (r.get("url") or "").endswith("/stt")]
        assert len(posts) == 1, found["status"]


class TestTheRecording:
    def test_it_is_a_sixteen_kilohertz_mono_pcm_wav(self):
        found = run("""
            await hold(900, 48000);
            const post = requests.filter((r) => (r.url || "").endsWith("/stt"))[0];
            const view = new DataView(post.body);
            const text = (at) => String.fromCharCode(view.getUint8(at), view.getUint8(at + 1),
                                                     view.getUint8(at + 2), view.getUint8(at + 3));
            console.log(JSON.stringify(report({
                riff: text(0), wave: text(8), fmt: text(12),
                format: view.getUint16(20, true),
                channels: view.getUint16(22, true),
                rate: view.getUint32(24, true),
                bits: view.getUint16(34, true),
                dataBytes: view.getUint32(40, true),
            })));
        """)
        assert found["riff"] == "RIFF"
        assert found["wave"] == "WAVE"
        assert found["format"] == 1
        assert found["channels"] == 1
        assert found["rate"] == 16000
        assert found["bits"] == 16
        assert found["dataBytes"] > 0

    def test_forty_eight_kilohertz_capture_is_resampled_down(self):
        """Android commonly runs its audio graph at 48 kHz, and Whisper is
        expecting 16. One second in has to be one second out."""
        found = run("""
            await hold(900, 48000);
            const post = requests.filter((r) => (r.url || "").endsWith("/stt"))[0];
            const view = new DataView(post.body);
            console.log(JSON.stringify(report({frames: view.getUint32(40, true) / 2})));
        """, SAMPLE_RATE="48000")
        assert 15000 <= found["frames"] <= 17000, found["frames"]

    def test_a_capture_already_at_sixteen_kilohertz_is_passed_through(self):
        found = run("""
            await hold(900, 16000);
            const post = requests.filter((r) => (r.url || "").endsWith("/stt"))[0];
            const view = new DataView(post.body);
            console.log(JSON.stringify(report({frames: view.getUint32(40, true) / 2})));
        """, SAMPLE_RATE="16000")
        assert found["frames"] == 16000

    def test_a_browser_without_audioworklet_still_captures_pcm(self):
        """The fallback is a deprecated node rather than MediaRecorder, because
        what MediaRecorder would give us is WebM/Opus -- a codec in the speech
        worker, which is the dependency this design is avoiding."""
        found = run("await hold(900); console.log(JSON.stringify(report()));",
                    WORKLET_AVAILABLE="false")
        posts = [r for r in found["requests"] if r.get("url", "").endswith("/stt")]
        assert len(posts) == 1
        assert posts[0]["bodyLength"] > 44


class TestABluetoothMicrophone:
    """The reported bug: dictation was good on an Android handset's own
    microphone and produced "(music)" and "(static)" through a Bluetooth
    headset.

    None of that is this page's to fix and all of it is this page's to survive.
    A headset has no microphone over A2DP, so capturing from one opens an HFP
    SCO link -- narrowband, telephony-codec, and far quieter than a handset
    microphone the platform gain-stages itself. What Whisper does with a quiet
    band-limited stream is emit the annotation tokens it was trained to use for
    non-speech passages. See mc_voice_hearing.py for the same account from the
    model's side.
    """

    def test_a_quiet_capture_is_lifted_before_it_is_sent(self):
        """The single biggest practical win: a Bluetooth capture is routinely
        quiet enough to be misheard and loud enough to transcribe perfectly once
        it has been normalised."""
        found = run("""
            await hold(900, 16000, 0.05);
            const post = requests.filter((r) => (r.url || "").endsWith("/stt"))[0];
            const view = new DataView(post.body);
            let peak = 0;
            for (let at = 44; at + 1 < post.body.byteLength; at += 2) {
                const value = Math.abs(view.getInt16(at, true));
                if (value > peak) peak = value;
            }
            console.log(JSON.stringify(report({peak: peak / 32768})));
        """, SAMPLE_RATE="16000")
        assert 0.5 < found["peak"] <= 0.65, found["peak"]

    def test_an_already_loud_capture_is_left_alone(self):
        found = run("""
            await hold(900, 16000, 0.8);
            const post = requests.filter((r) => (r.url || "").endsWith("/stt"))[0];
            const view = new DataView(post.body);
            let peak = 0;
            for (let at = 44; at + 1 < post.body.byteLength; at += 2) {
                const value = Math.abs(view.getInt16(at, true));
                if (value > peak) peak = value;
            }
            console.log(JSON.stringify(report({peak: peak / 32768})));
        """, SAMPLE_RATE="16000")
        assert found["peak"] > 0.75, found["peak"]

    def test_a_silent_capture_is_never_sent_at_all(self):
        """Running a large model over silence to be told it was silence is a
        slower way to the same place, and the answer that comes back is
        `(music)` rather than nothing."""
        found = run("""
            await hold(900, 16000, 0.0002);
            console.log(JSON.stringify(report()));
        """, SAMPLE_RATE="16000")
        assert not [r for r in found["requests"] if (r.get("url") or "").endswith("/stt")]

    def test_the_refusal_names_the_headset_when_that_is_what_it_is(self):
        found = run("""
            await hold(900, 16000, 0.0002);
            console.log(JSON.stringify(report()));
        """, SAMPLE_RATE="16000", TRACK_RATE="8000",
             TRACK_LABEL='"Jabra Elite (Bluetooth)"')
        assert "Bluetooth" in found["status"]

    def test_a_good_transcript_still_mentions_the_headset_once(self):
        """Once. A note repeated after every successful transcription is a note
        nobody reads, and the point of it is that somebody who is being misheard
        can tell why."""
        found = run("""
            await hold(900, 16000, 0.3);
            const first = status.notice.textContent;
            await hold(900, 16000, 0.3);
            console.log(JSON.stringify(report({first})));
        """, SAMPLE_RATE="16000", TRACK_RATE="8000",
             TRACK_LABEL='"Jabra Elite (Bluetooth)"')
        assert "Jabra" in found["first"]
        assert "Jabra" not in found["status"]
        assert found["status"].startswith("Transcribed.")

    def test_the_handset_microphone_is_never_complained_about(self):
        found = run("""
            await hold(900, 16000, 0.3);
            console.log(JSON.stringify(report()));
        """, SAMPLE_RATE="48000")
        assert found["status"] == "Transcribed."

    def test_the_request_asks_for_what_helps_and_insists_on_nothing(self):
        """Every processor is advisory. They are tuned for a wideband
        microphone, and a browser that cannot apply them to an HFP stream should
        hand over the stream rather than fail the request."""
        found = run("await hold(900); console.log(JSON.stringify(report()));")
        opens = [r for r in found["requests"] if r.get("kind") == "getUserMedia"]
        assert opens, "the microphone was never opened"
        audio = opens[0]["constraints"]["audio"]
        for name in ("echoCancellation", "noiseSuppression", "autoGainControl"):
            assert audio[name] == {"ideal": True}, name

    def test_it_does_not_ask_the_device_for_a_rate_it_will_not_be_read_at(self):
        """The samples come out of the AudioContext, not off the track, and are
        resampled from `ctx.sampleRate` whatever the device was persuaded to do
        -- so asking was one more thing to negotiate while opening the device,
        for no effect on the audio."""
        found = run("await hold(900); console.log(JSON.stringify(report()));")
        opens = [r for r in found["requests"] if r.get("kind") == "getUserMedia"]
        assert "sampleRate" not in opens[0]["constraints"]["audio"]

    def test_the_recording_is_still_sixteen_kilohertz(self):
        """Which is the point: dropping the hint changed the request, not the
        result."""
        found = run("""
            await hold(900, 48000);
            const post = requests.filter((r) => (r.url || "").endsWith("/stt"))[0];
            const view = new DataView(post.body);
            console.log(JSON.stringify(report({rate: view.getUint32(24, true)})));
        """, SAMPLE_RATE="48000")
        assert found["rate"] == 16000

    def test_a_refused_constraint_set_falls_back_to_any_microphone(self):
        """The devices most likely to reject a constraint set are the Android
        WebViews this feature already bends over backwards for, and a second
        attempt is the difference between a worse recording and no recording."""
        found = run("await hold(900); console.log(JSON.stringify(report()));",
                    DENIAL='"OverconstrainedError"', FIRST_OPEN_FAILS="true")
        opens = [r for r in found["requests"] if r.get("kind") == "getUserMedia"]
        assert len(opens) == 2, opens
        assert opens[1]["constraints"] == {"audio": True}
        assert [r for r in found["requests"] if (r.get("url") or "").endswith("/stt")]


# --------------------------------------------------------------------------- #
# The composer
# --------------------------------------------------------------------------- #


class TestTheComposer:
    def test_a_transcript_lands_in_the_existing_message_box(self):
        found = run("await hold(900); console.log(JSON.stringify(report()));")
        assert found["composer"] == "the quick brown fox"

    def test_existing_text_is_kept_and_the_transcript_appended(self):
        """There is no undo in a Gradio textbox, and somebody who typed half a
        message and dictated the rest has not asked for the half they typed to
        be thrown away."""
        found = run("""
            message.value = "I was already writing";
            await hold(900);
            console.log(JSON.stringify(report()));
        """)
        assert found["composer"] == "I was already writing the quick brown fox"

    def test_gradio_is_told_the_value_changed(self):
        found = run("""
            const seen = [];
            message.addEventListener("input", () => seen.push("input"));
            message.addEventListener("change", () => seen.push("change"));
            await hold(900);
            console.log(JSON.stringify(report({seen})));
        """)
        assert found["seen"] == ["input", "change"]

    def test_auto_send_off_never_presses_send(self):
        found = run("await hold(900); console.log(JSON.stringify(report()));")
        assert found["sends"] == 0

    def test_auto_send_on_presses_the_existing_send_button(self):
        """The existing one. Attachments, thread selection, sampling values and
        cancellation all already live behind that button."""
        answers = json.loads(DEFAULTS["ANSWERS"])
        answers["voice/stt"]["json"]["auto_send"] = True
        found = run("await hold(900); await tick(); console.log(JSON.stringify(report()));",
                    ANSWERS=json.dumps(answers))
        assert found["sends"] == 1
        assert found["composer"] == "the quick brown fox"

    def test_auto_send_does_not_fire_into_a_run_that_is_still_going(self):
        answers = json.loads(DEFAULTS["ANSWERS"])
        answers["voice/stt"]["json"]["auto_send"] = True
        found = run("""
            await engageMic(7);
            await tick();
            feed(new Array(8000).fill(0.2));
            NOW += 900;
            releaseMic(7);
            // The reply starts while the transcription is still in flight,
            // which is the race auto-send has to lose.
            runState.value = "llm";
            await tick();
            console.log(JSON.stringify(report()));
        """, ANSWERS=json.dumps(answers))
        assert found["composer"] == "the quick brown fox"
        assert found["sends"] == 0

    def test_a_failed_transcription_says_so_and_writes_nothing(self):
        answers = json.loads(DEFAULTS["ANSWERS"])
        answers["voice/stt"] = {"json": {"ok": False, "error": "No speech was detected."}}
        found = run("await hold(900); console.log(JSON.stringify(report()));",
                    ANSWERS=json.dumps(answers))
        assert found["composer"] == ""
        assert "No speech was detected." in found["status"]

    def test_a_dropped_connection_does_not_lose_the_composer(self):
        answers = json.loads(DEFAULTS["ANSWERS"])
        answers["voice/stt"] = {"reject": True}
        found = run("""
            message.value = "half a message";
            await hold(900);
            console.log(JSON.stringify(report()));
        """, ANSWERS=json.dumps(answers))
        assert found["composer"] == "half a message"
        assert found["status"]


# --------------------------------------------------------------------------- #
# Speaking a reply
# --------------------------------------------------------------------------- #


class TestSpeaking:
    def test_a_new_token_is_exchanged_for_audio_and_played(self):
        found = run("""
            token.value = "TOKEN-1";
            intervals.forEach((fn) => fn());
            await tick();
            console.log(JSON.stringify(report()));
        """)
        posts = [r for r in found["requests"] if r.get("url", "").endswith("/tts")]
        assert len(posts) == 1
        assert json.loads(posts[0]["bodyText"]) == {"token": "TOKEN-1"}
        assert found["played"] == 1

    def test_the_same_token_is_never_spoken_twice(self):
        """A hidden box is re-read on a timer and on every UI update; a duplicate
        would speak the reply again."""
        found = run("""
            token.value = "TOKEN-1";
            intervals.forEach((fn) => fn());
            await tick();
            intervals.forEach((fn) => fn());
            await tick();
            console.log(JSON.stringify(report()));
        """)
        assert len([r for r in found["requests"] if r.get("url", "").endswith("/tts")]) == 1
        assert found["played"] == 1

    def test_an_empty_token_asks_for_nothing(self):
        """Which is what a run that failed, was Stopped, or ran with the switch
        off leaves behind."""
        found = run("""
            intervals.forEach((fn) => fn());
            await tick();
            console.log(JSON.stringify(report()));
        """)
        assert not any(r.get("url", "").endswith("/tts") for r in found["requests"])
        assert found["played"] == 0

    def test_no_reply_text_is_ever_sent_by_the_browser(self):
        """The browser is given a handle, never the words. It could not send the
        reply text if it wanted to, because it never had it."""
        found = run("""
            token.value = "TOKEN-1";
            intervals.forEach((fn) => fn());
            await tick();
            console.log(JSON.stringify(report()));
        """)
        for request in found["requests"]:
            body = request.get("bodyText")
            if body:
                assert set(json.loads(body)) <= {"token", "kind"}

    def test_playback_blocked_by_autoplay_is_actionable_and_not_retried(self):
        """A persisted "speak replies" is intent; a fresh page may not be allowed
        to make a sound until somebody has touched it."""
        found = run("""
            token.value = "TOKEN-1";
            intervals.forEach((fn) => fn());
            await tick();
            console.log(JSON.stringify(report()));
        """, CONTEXT_STATE='"suspended"', RESUME_WORKS="false")
        assert "tap Voice or the microphone" in found["status"]
        assert not any(r.get("url", "").endswith("/tts") for r in found["requests"])

    def test_a_failed_synthesis_says_so_without_touching_the_transcript(self):
        answers = json.loads(DEFAULTS["ANSWERS"])
        answers["voice/tts"] = {"status": 503, "json": {"ok": False}}
        found = run("""
            token.value = "TOKEN-1";
            intervals.forEach((fn) => fn());
            await tick();
            console.log(JSON.stringify(report()));
        """, ANSWERS=json.dumps(answers))
        assert "could not read it aloud" in found["status"]
        assert found["played"] == 0

    def test_holding_the_microphone_stops_the_speaker_first(self):
        """The V1 interrupt gesture, and the thing that stops a phone recording
        its own loudspeaker."""
        found = run("""
            token.value = "TOKEN-1";
            intervals.forEach((fn) => fn());
            await tick();
            const before = played.length;
            await engageMic(9);
            await tick();
            console.log(JSON.stringify(report({before, stoppedNow: stopped.length})));
        """)
        assert found["before"] == 1
        assert found["stoppedNow"] == 1

    def test_a_second_reply_stops_the_first_before_playing(self):
        found = run("""
            token.value = "TOKEN-1";
            intervals.forEach((fn) => fn());
            await tick();
            token.value = "TOKEN-2";
            intervals.forEach((fn) => fn());
            await tick();
            console.log(JSON.stringify(report()));
        """)
        assert found["played"] == 2
        assert found["stopped"] == 1


# --------------------------------------------------------------------------- #
# Re-wiring
# --------------------------------------------------------------------------- #


class TestIdempotentWiring:
    def test_a_gradio_update_does_not_add_a_second_set_of_listeners(self):
        """Section 59. Two pointerdown listeners is two recordings, two uploads
        and two transcripts appended to the composer."""
        found = run("""
            updated.forEach((fn) => fn());
            updated.forEach((fn) => fn());
            updated.forEach((fn) => fn());
            console.log(JSON.stringify(report()));
        """)
        for name, count in found["listeners"]:
            assert count == 1, f"{name} was wired {count} times"

    def test_a_hold_after_several_updates_still_posts_one_recording(self):
        found = run("""
            updated.forEach((fn) => fn());
            updated.forEach((fn) => fn());
            await hold(900);
            console.log(JSON.stringify(report()));
        """)
        assert len([r for r in found["requests"] if r.get("url", "").endswith("/stt")]) == 1
        assert found["composer"] == "the quick brown fox"

    def test_a_missing_microphone_element_is_skipped_rather_than_fatal(self):
        """The tab is fully usable with this file absent, which is the test of
        whether it is really polish -- and one missing id must not take the rest
        of the wiring down with it."""
        found = run("""
            delete elements["mc-llm-chat-voice-mic"];
            updated.forEach((fn) => fn());
            console.log(JSON.stringify(report()));
        """)
        assert found["consoleErrors"] == []


# --------------------------------------------------------------------------- #
# The Settings page row
# --------------------------------------------------------------------------- #


class TestTheSettingsRow:
    """The row that shipped able to freeze on "Starting…" with nothing to say.

    Every test here is one of the ways it could stop moving. The rule they all
    check is the same: whatever happens, the row says something and the button
    comes back — a control that disabled itself on the way into a request it
    never got an answer to is a control somebody sits and watches.
    """

    def test_it_draws_what_is_installed_on_the_first_pass(self):
        found = run("await tick(); console.log(JSON.stringify(report()));",
                    SETTINGS_PRESENT="true")
        assert found["settings"]["runtime"] == "Voice Chat is ready."
        assert found["settings"]["runtimeLine"] == "Installed"
        assert found["settings"]["ttsButton"] == "Installed"
        assert found["settings"]["ttsDisabled"] is True

    def test_a_pressed_button_that_is_refused_says_why_and_comes_back(self):
        """The reported bug, as a test. The build cannot install anything, the
        route says so in its reply, and the row has to show that rather than
        sitting on "Starting…"."""
        answers = status_answer(stt_ready=False, tts_ready=False, ready=False,
                                stt_message="Not installed")
        answers["voice/install"] = {"status": 409, "json": {
            "ok": False,
            "error": "Whisper Small cannot be installed by this build: its artifacts are "
                     "not pinned.",
        }}
        found = run("""
            await tick();
            settingsParts.sttButton.fire("click");
            await tick();
            console.log(JSON.stringify(report()));
        """, SETTINGS_PRESENT="true", ANSWERS=json.dumps(answers))

        assert "not pinned" in found["settings"]["sttLine"]
        assert found["settings"]["sttFailed"] is True
        assert found["settings"]["sttButton"] == "Download default STT"
        assert found["settings"]["sttDisabled"] is False
        assert any("could not start" in line for line in found["consoleErrors"])

    def test_a_dropped_connection_on_the_press_says_so_and_comes_back(self):
        answers = status_answer(stt_ready=False, ready=False)
        answers["voice/install"] = {"reject": True}
        found = run("""
            await tick();
            settingsParts.sttButton.fire("click");
            await tick();
            console.log(JSON.stringify(report()));
        """, SETTINGS_PRESENT="true", ANSWERS=json.dumps(answers))

        assert "Could not reach this WebUI" in found["settings"]["sttLine"]
        assert found["settings"]["sttDisabled"] is False

    def test_a_status_route_that_refuses_is_reported_rather_than_swallowed(self):
        """This is what actually froze the row: `refreshStatus` resolved with
        null on any failure and every caller quietly did nothing."""
        answers = status_answer()
        answers["voice/status"] = {"status": 403, "json": {
            "ok": False, "error": "This page is out of date with the WebUI. Reload it."}}
        found = run("await tick(); console.log(JSON.stringify(report()));",
                    SETTINGS_PRESENT="true", ANSWERS=json.dumps(answers))

        assert "out of date" in found["settings"]["runtime"]
        assert "out of date" in found["settings"]["sttLine"]
        assert found["settings"]["sttFailed"] is True
        assert found["settings"]["sttDisabled"] is False
        assert any("status refused" in line for line in found["consoleErrors"])

    def test_an_unreachable_status_route_is_reported_too(self):
        answers = status_answer()
        answers["voice/status"] = {"reject": True}
        found = run("await tick(); console.log(JSON.stringify(report()));",
                    SETTINGS_PRESENT="true", ANSWERS=json.dumps(answers))

        assert "Could not reach this WebUI" in found["settings"]["runtime"]
        assert found["settings"]["sttDisabled"] is False

    def test_progress_is_drawn_while_a_download_runs(self):
        """The other half of the same defect: the installer wrote a sentence for
        every step and the route was never given anywhere to put it, so the row
        had nothing but its own initial text for the length of a 375 MB
        download."""
        answers = status_answer(stt_ready=False, ready=False, progress={
            "stt": {"running": True, "fraction": 0.41,
                    "text": "Downloading 2 of 3 — decoder.onnx (262 MB)"}})
        found = run("await tick(); console.log(JSON.stringify(report()));",
                    SETTINGS_PRESENT="true", ANSWERS=json.dumps(answers))

        assert "decoder.onnx" in found["settings"]["sttLine"]
        assert "41%" in found["settings"]["sttLine"]
        assert found["settings"]["sttButton"] == "Installing…"
        assert found["settings"]["sttDisabled"] is True

    def test_a_failed_install_keeps_its_reason_on_screen(self):
        answers = status_answer(stt_ready=False, ready=False, progress={
            "stt": {"running": False, "failed": True, "fraction": 0.0,
                    "text": "decoder.onnx failed its hash check."}})
        found = run("await tick(); console.log(JSON.stringify(report()));",
                    SETTINGS_PRESENT="true", ANSWERS=json.dumps(answers))

        assert found["settings"]["sttLine"] == "decoder.onnx failed its hash check."
        assert found["settings"]["sttFailed"] is True
        assert found["settings"]["sttDisabled"] is False, (
            "a failed download left its button disabled, so it cannot be retried")

    def test_a_finished_install_clears_the_failure_styling(self):
        answers = status_answer(progress={"stt": {"running": False, "failed": False,
                                                  "fraction": 1.0, "text": "Installed."}})
        found = run("""
            await tick();
            settingsParts.sttLine.classList.add("mc-voice-failed");
            await repaint();
            console.log(JSON.stringify(report()));
        """, SETTINGS_PRESENT="true", ANSWERS=json.dumps(answers))
        assert found["settings"]["sttFailed"] is False

    def test_the_press_carries_the_kind_and_nothing_else(self):
        found = run("""
            await tick();
            settingsParts.sttButton.fire("click");
            await tick();
            console.log(JSON.stringify(report()));
        """, SETTINGS_PRESENT="true")
        posts = [r for r in found["requests"] if r.get("url", "").endswith("/install")]
        assert len(posts) == 1
        assert json.loads(posts[0]["bodyText"]) == {"kind": "stt"}

    def test_the_row_is_wired_once_however_many_updates_arrive(self):
        found = run("""
            await tick();
            updated.forEach((fn) => fn());
            updated.forEach((fn) => fn());
            settingsParts.sttButton.fire("click");
            await tick();
            console.log(JSON.stringify(report()));
        """, SETTINGS_PRESENT="true")
        posts = [r for r in found["requests"] if r.get("url", "").endswith("/install")]
        assert len(posts) == 1, "one press sent the install request more than once"

    def test_a_page_without_the_row_is_not_an_error(self):
        found = run("await tick(); console.log(JSON.stringify(report()));")
        assert found["consoleErrors"] == []

    def test_the_row_reads_its_own_token_and_not_the_conversation_panel_s(self):
        """The Settings page is not the Conversation panel, and a row that got
        its token from a hidden component in another tab is a row that sends an
        empty token -- and is refused 403 -- whenever that component is not
        there or has not been hydrated yet."""
        found = run("""
            await tick();
            // The Conversation panel is gone and the row's own token is the
            // only one left. Requests from before this point are not the
            // subject, so they are cleared.
            delete elements["mc-llm-chat-voice-key"];
            settingsRow["data-mc-voice-key"] = "SETTINGS-TOKEN";
            requests.length = 0;
            settingsParts.sttButton.fire("click");
            await repaint();
            console.log(JSON.stringify(report()));
        """, SETTINGS_PRESENT="true")
        posts = [r for r in found["requests"] if r.get("url")]
        assert posts, "the row made no request at all"
        for request in posts:
            assert request["headers"]["X-Model-Chain-Voice"] == "SETTINGS-TOKEN"


class TestInstallingFromAFolder:
    """The second way in, for a build that cannot download the models itself."""

    def test_the_folder_is_sent_with_the_kind(self):
        found = run("""
            await tick();
            settingsParts.sttFolder.value = "C:\\\\Users\\\\me\\\\Downloads\\\\whisper";
            requests.length = 0;
            settingsParts.sttLocal.fire("click");
            await tick();
            console.log(JSON.stringify(report()));
        """, SETTINGS_PRESENT="true")
        posts = [r for r in found["requests"] if r.get("url", "").endswith("/install")]
        assert len(posts) == 1
        assert json.loads(posts[0]["bodyText"]) == {
            "kind": "stt", "folder": "C:\\Users\\me\\Downloads\\whisper"}

    def test_an_empty_folder_box_asks_for_one_rather_than_posting(self):
        found = run("""
            await tick();
            requests.length = 0;
            settingsParts.sttLocal.fire("click");
            await tick();
            console.log(JSON.stringify(report()));
        """, SETTINGS_PRESENT="true")
        assert not any(r.get("url", "").endswith("/install") for r in found["requests"])
        assert "Type the folder" in found["settings"]["sttLine"]

    def test_surrounding_space_is_trimmed(self):
        found = run("""
            await tick();
            settingsParts.sttFolder.value = "   /home/me/voice   ";
            requests.length = 0;
            settingsParts.sttLocal.fire("click");
            await tick();
            console.log(JSON.stringify(report()));
        """, SETTINGS_PRESENT="true")
        posts = [r for r in found["requests"] if r.get("url", "").endswith("/install")]
        assert json.loads(posts[0]["bodyText"])["folder"] == "/home/me/voice"

    def test_a_refused_folder_install_says_why_and_frees_the_button(self):
        answers = status_answer(stt_ready=False, ready=False)
        answers["voice/install"] = {"status": 409, "json": {
            "ok": False, "error": "C:\\\\nope does not have everything Whisper Small needs "
                                  "— tokens.txt is missing."}}
        found = run("""
            await tick();
            settingsParts.sttFolder.value = "C:\\\\nope";
            settingsParts.sttLocal.fire("click");
            await tick();
            console.log(JSON.stringify(report()));
        """, SETTINGS_PRESENT="true", ANSWERS=json.dumps(answers))
        assert "tokens.txt is missing" in found["settings"]["sttLine"]
        assert found["settings"]["sttFailed"] is True
        assert found["settings"]["sttLocalDisabled"] is False


class TestThePollingStops:
    """A route a page polls forever is a log that scrolls forever. It did."""

    def test_an_idle_row_polls_slowly(self):
        found = run("""
            await tick();
            requests.length = 0;
            NOW += 5000;
            await tick();
            const soon = requests.length;
            NOW += 25000;
            await tick();
            console.log(JSON.stringify(report({soon, later: requests.length})));
        """, SETTINGS_PRESENT="true")
        assert found["soon"] == 0, "an idle row polled again within five seconds"
        assert found["later"] >= 1

    def test_a_running_install_polls_quickly(self):
        answers = status_answer(stt_ready=False, ready=False, progress={
            "stt": {"running": True, "fraction": 0.2, "text": "Downloading…"}})
        found = run("""
            await tick();
            requests.length = 0;
            NOW += 2000;
            await tick();
            console.log(JSON.stringify(report({polls: requests.length})));
        """, SETTINGS_PRESENT="true", ANSWERS=json.dumps(answers))
        assert found["polls"] >= 1, "a running download was not being watched"

    def test_repeated_failures_back_off_rather_than_hammering(self):
        answers = status_answer()
        answers["voice/status"] = {"status": 401, "json": {
            "ok": False, "error": "Sign in to the WebUI to use Voice Chat."}}
        found = run("""
            await tick();
            const gaps = [];
            for (let i = 0; i < 6; i += 1) {
                const before = requests.length;
                let waited = 0;
                while (requests.length === before && waited < 300000) {
                    NOW += 1000;
                    await tick(2);
                    waited += 1000;
                }
                gaps.push(waited);
            }
            console.log(JSON.stringify(report({gaps})));
        """, SETTINGS_PRESENT="true", ANSWERS=json.dumps(answers))

        gaps = found["gaps"]
        assert gaps[0] < gaps[-1], f"the failing poll did not back off: {gaps}"
        assert max(gaps) <= 61000, f"the back-off ran past its cap: {gaps}"

    def test_a_background_tab_is_not_polled(self):
        found = run("""
            await tick();
            requests.length = 0;
            for (let i = 0; i < 10; i += 1) { NOW += 30000; await tick(); }
            console.log(JSON.stringify(report({polls: requests.length})));
        """, SETTINGS_PRESENT="true", HIDDEN="true")
        assert found["polls"] == 0, "a hidden tab kept polling the WebUI"


class TestTheThreeQualities:
    """The speech-to-text row is three cards, and the two buttons on each do
    different things. Download fetches that tier; Use points Voice Chat at it.

    Separate on purpose: keeping all three on disk and switching between them
    should not be a download, and choosing the high tier and *then* starting its
    download is the order people actually do it in.
    """

    def test_each_tier_is_drawn_with_its_own_state(self):
        found = run("await tick(); console.log(JSON.stringify(report()));",
                    SETTINGS_PRESENT="true")
        tiers = found["settings"]["tiers"]
        assert tiers["whisper-small-int8"]["mark"] == "In use"
        assert tiers["whisper-small-int8"]["chosen"] is True
        assert tiers["whisper-small-int8"]["install"] == "Download again"
        assert tiers["whisper-small-int8"]["useDisabled"] is True
        assert tiers["whisper-base-int8"]["mark"] == ""
        assert tiers["whisper-base-int8"]["install"] == "Download"
        assert tiers["whisper-base-int8"]["use"] == "Use this"
        assert tiers["whisper-base-int8"]["useDisabled"] is False
        assert found["settings"]["chosenLabel"] == "Whisper Small"

    def test_pressing_use_names_the_tier_and_nothing_else(self):
        found = run("""
            await tick();
            settingsParts.tierList.fire("click", {
                target: settingsParts.tiers["whisper-medium-int8"].useButton});
            await tick();
            console.log(JSON.stringify(report()));
        """, SETTINGS_PRESENT="true")
        chose = [r for r in found["requests"]
                 if "voice/models" in r["url"] and "select" in (r["bodyText"] or "")]
        assert chose, "pressing Use never asked the WebUI to change the model"
        body = json.loads(chose[-1]["bodyText"])
        assert body == {"kind": "stt", "select": "whisper-medium-int8"}

    def test_pressing_download_carries_the_kind_and_the_model(self):
        found = run("""
            await tick();
            settingsParts.tierList.fire("click", {
                target: settingsParts.tiers["whisper-medium-int8"].installButton});
            await tick();
            console.log(JSON.stringify(report()));
        """, SETTINGS_PRESENT="true")
        installs = [r for r in found["requests"] if "voice/install" in r["url"]]
        assert installs, "pressing Download never reached the install route"
        assert json.loads(installs[-1]["bodyText"]) == {
            "kind": "stt", "model": "whisper-medium-int8"}

    def test_a_download_draws_its_progress_on_its_own_card_only(self):
        """A kind is three tiers now. A row that knew a download was running but
        not which of its three buttons started it would put the bar on all of
        them."""
        answers = json.loads(DEFAULTS["ANSWERS"])
        answers["voice/models"]["json"]["progress"] = {
            "running": True, "fraction": 0.41, "model": "whisper-medium-int8",
            "text": "Downloading 2 of 3 — decoder.onnx (490 MB)"}
        found = run("await tick(); console.log(JSON.stringify(report()));",
                    SETTINGS_PRESENT="true", ANSWERS=json.dumps(answers))
        tiers = found["settings"]["tiers"]
        assert "decoder.onnx" in tiers["whisper-medium-int8"]["state"]
        assert "41%" in tiers["whisper-medium-int8"]["state"]
        assert tiers["whisper-medium-int8"]["installDisabled"] is True
        assert "decoder.onnx" not in tiers["whisper-base-int8"]["state"]
        assert tiers["whisper-base-int8"]["installDisabled"] is False

    def test_a_failed_download_keeps_its_reason_on_its_own_card(self):
        answers = json.loads(DEFAULTS["ANSWERS"])
        answers["voice/models"]["json"]["progress"] = {
            "running": False, "failed": True, "fraction": 0.0,
            "model": "whisper-base-int8", "text": "decoder.onnx failed its hash check."}
        found = run("await tick(); console.log(JSON.stringify(report()));",
                    SETTINGS_PRESENT="true", ANSWERS=json.dumps(answers))
        tiers = found["settings"]["tiers"]
        assert tiers["whisper-base-int8"]["state"] == "decoder.onnx failed its hash check."
        assert tiers["whisper-base-int8"]["failed"] is True
        assert tiers["whisper-base-int8"]["installDisabled"] is False, (
            "a failed download left its button disabled, so it cannot be retried")
        assert tiers["whisper-medium-int8"]["failed"] is False

    def test_a_folder_install_reads_the_box_under_the_tier_it_belongs_to(self):
        """Three folder boxes on one page. Pressing Install under the high tier
        while something is typed under the low one must not install the low
        one's folder."""
        found = run("""
            await tick();
            settingsParts.tiers["whisper-base-int8"].folder.value = "/downloads/base";
            settingsParts.tiers["whisper-medium-int8"].folder.value = "/downloads/medium";
            settingsParts.tiers["whisper-medium-int8"].local.fire("click");
            await tick();
            console.log(JSON.stringify(report()));
        """, SETTINGS_PRESENT="true")
        installs = [r for r in found["requests"] if "voice/install" in r["url"]]
        assert installs, "the folder install never reached the route"
        assert json.loads(installs[-1]["bodyText"]) == {
            "kind": "stt", "folder": "/downloads/medium", "model": "whisper-medium-int8"}


class TestTheDeliverySliders:
    """Four sliders for the default voice, written on release rather than on
    every pixel of drag: each write is a settings-file save."""

    def test_they_are_drawn_from_the_route(self):
        found = run("""
            await tick();
            console.log(JSON.stringify({
                pitch: voicesParts.sliders.pitch.value,
                shown: voicesParts.outputs.pitch.textContent,
                summary: voicesParts.deliverySummary.textContent,
            }));
        """, VOICES_PRESENT="true", VOICES_VISIBLE="true")
        assert found["shown"] == "0 semitones"
        assert found["summary"] == "Kokoro's own delivery"

    def test_dragging_moves_the_number_without_saving(self):
        found = run("""
            await tick();
            voicesParts.sliders.pitch.value = "-3";
            voicesParts.deliveryPanel.fire("input", {target: voicesParts.sliders.pitch});
            await tick();
            console.log(JSON.stringify(report({
                shown: voicesParts.outputs.pitch.textContent,
            })));
        """, VOICES_PRESENT="true", VOICES_VISIBLE="true")
        assert found["shown"] == "-3 semitones"
        assert not [r for r in found["requests"]
                    if "voice/profile" in r["url"] and "profile" in (r["bodyText"] or "")]

    def test_releasing_saves_all_four(self):
        found = run("""
            await tick();
            voicesParts.sliders.speed.value = "1.25";
            voicesParts.deliveryPanel.fire("change", {target: voicesParts.sliders.speed});
            await tick();
            console.log(JSON.stringify(report()));
        """, VOICES_PRESENT="true", VOICES_VISIBLE="true")
        writes = [r for r in found["requests"]
                  if "voice/profile" in r["url"] and "profile" in (r["bodyText"] or "")]
        assert writes, "releasing a slider never saved anything"
        assert json.loads(writes[-1]["bodyText"])["profile"]["speed"] == 1.25

    def test_reset_puts_every_control_back_and_saves_once(self):
        found = run("""
            await tick();
            voicesParts.sliders.speed.value = "1.9";
            voicesParts.sliders.pitch.value = "5";
            voicesParts.deliveryPanel.fire("click", {
                target: {closest: (s) => s.indexOf("delivery-reset") !== -1
                                         ? voicesParts.resetButton : null}});
            await tick();
            console.log(JSON.stringify(report({
                speed: voicesParts.sliders.speed.value,
                pitch: voicesParts.sliders.pitch.value,
            })));
        """, VOICES_PRESENT="true", VOICES_VISIBLE="true")
        assert float(found["speed"]) == 1.0
        assert float(found["pitch"]) == 0.0
        writes = [r for r in found["requests"]
                  if "voice/profile" in r["url"] and "profile" in (r["bodyText"] or "")]
        assert len(writes) == 1, "Reset saved once per control instead of once"


class TestTheEngineRow:
    """The engine has a button of its own, because the reasoning that it did not
    need one had a hole and somebody fell in.

    Install both models from files you already had — which downloads nothing —
    and the engine is still missing, both model buttons read "Installed", and
    there is nothing left to press."""

    def test_it_is_drawn_and_can_be_pressed(self):
        answers = status_answer(runtime_ready=False, ready=False,
                                runtime_message="Not installed",
                                stt_ready=True, tts_ready=True,
                                summary="Voice Chat is not ready yet — still to install: "
                                        "the voice engine.")
        found = run("""
            await tick();
            requests.length = 0;
            settingsParts.runtimeButton.fire("click");
            await tick();
            console.log(JSON.stringify(report()));
        """, SETTINGS_PRESENT="true", ANSWERS=json.dumps(answers))

        posts = [r for r in found["requests"] if r.get("url", "").endswith("/install")]
        assert json.loads(posts[0]["bodyText"]) == {"kind": "runtime"}

    def test_the_summary_names_what_is_still_missing(self):
        """Three separate "Not installed" lines and a feature that does not work
        is a puzzle. One sentence solves it."""
        answers = status_answer(runtime_ready=False, ready=False,
                                stt_ready=True, tts_ready=True,
                                runtime_message="Not installed",
                                summary="Voice Chat is not ready yet — still to install: "
                                        "the voice engine.")
        found = run("await tick(); console.log(JSON.stringify(report()));",
                    SETTINGS_PRESENT="true", ANSWERS=json.dumps(answers))
        assert "the voice engine" in found["settings"]["runtime"]
        assert found["settings"]["runtimeButton"] == "Install voice engine"
        assert found["settings"]["runtimeDisabled"] is False

    def test_an_installed_engine_disables_its_button(self):
        found = run("await tick(); console.log(JSON.stringify(report()));",
                    SETTINGS_PRESENT="true")
        assert found["settings"]["runtimeButton"] == "Installed"
        assert found["settings"]["runtimeDisabled"] is True


class TestTheEngineSelector:
    """Choosing an engine is a runtime boundary, not a preference.

    The server cancels speech, stops whichever worker was running and persists
    the choice before it answers, and the page then *reloads* rather than
    repaints. That is deliberate: the inactive engine's controls have to be
    absent from the document rather than hidden in it, and the cheapest way to
    be certain of that is to ask the server for a document that never contained
    them.
    """

    def test_choosing_an_engine_posts_it_and_reloads(self):
        found = run("""
            await tick();
            requests.length = 0;
            settingsParts.engineCards.sopro.fire("click");
            await tick();
            await hold(200);
            console.log(JSON.stringify(report()));
        """, SETTINGS_PRESENT="true")

        posts = [r for r in found["requests"]
                 if r.get("url", "").endswith("/engine/select")]
        assert len(posts) == 1
        assert json.loads(posts[0]["bodyText"]) == {"engine": "sopro"}
        assert found["reloads"] == 1

    def test_the_selected_engine_cannot_be_re_selected(self):
        """Its card is disabled, so a second press is not a second switch --
        which would cancel speech and stop a worker for nothing."""
        found = run("""
            await tick();
            requests.length = 0;
            settingsParts.engineCards.kokoro.fire("click");
            await tick();
            console.log(JSON.stringify(report()));
        """, SETTINGS_PRESENT="true")

        assert not [r for r in found["requests"]
                    if r.get("url", "").endswith("/engine/select")]
        assert found["reloads"] == 0

    def test_a_refused_switch_re_enables_the_cards_and_does_not_reload(self):
        """A page that disabled both cards and then reloaded on a failure would
        be a page you cannot get back to the engine you were on."""
        answers = status_answer()
        answers["voice/engine/select"] = {
            "status": 400, "json": {"ok": False, "error": "no", "active": "kokoro"}}
        found = run("""
            await tick();
            settingsParts.engineCards.sopro.fire("click");
            await tick();
            await hold(200);
            console.log(JSON.stringify(report({
                soproDisabled: settingsParts.engineCards.sopro.disabled,
            })));
        """, SETTINGS_PRESENT="true", ANSWERS=json.dumps(answers))

        assert found["reloads"] == 0
        assert found["soproDisabled"] is False


class TestTheEngineChangedUnderThisPage:
    """Somebody switched the engine in another tab.

    Every control in this document belongs to an engine that is no longer
    selected, so repainting is not a fix. The server says so with a flag rather
    than only a status code, because 409 already means several other things on
    these routes -- a Lab session that expired, an install already running, a
    turn that was over before anything listened to it -- and a page that
    reloaded for all of them would reload when a button is pressed twice.
    """

    def test_a_flagged_refusal_reloads_the_page(self):
        answers = status_answer()
        answers["voice/voices"] = {
            "status": 409,
            "json": {"ok": False, "error": "Kokoro is no longer selected.",
                     "engine_mismatch": True}}
        found = run("""
            await tick();
            await hold(200);
            console.log(JSON.stringify(report()));
        """, VOICES_PRESENT="true", VOICES_VISIBLE="true", ANSWERS=json.dumps(answers))
        assert found["reloads"] == 1

    def test_an_ordinary_refusal_does_not(self):
        answers = status_answer()
        answers["voice/voices"] = {
            "status": 409, "json": {"ok": False, "error": "That is already running."}}
        found = run("""
            await tick();
            await hold(200);
            console.log(JSON.stringify(report()));
        """, VOICES_PRESENT="true", VOICES_VISIBLE="true", ANSWERS=json.dumps(answers))
        assert found["reloads"] == 0

    def test_it_reloads_once_however_many_requests_were_in_flight(self):
        """Several polls can be in the air at the moment of a switch, and a
        reload storm is worse than a stale panel."""
        answers = status_answer()
        answers["voice/voices"] = {
            "status": 409,
            "json": {"ok": False, "error": "gone", "engine_mismatch": True}}
        answers["voice/profile"] = {
            "status": 409,
            "json": {"ok": False, "error": "gone", "engine_mismatch": True}}
        found = run("""
            await tick();
            await hold(400);
            console.log(JSON.stringify(report()));
        """, VOICES_PRESENT="true", VOICES_VISIBLE="true", ANSWERS=json.dumps(answers))
        assert found["reloads"] == 1


class TestTheFolderButtonComesBack:
    """The second stuck button. `paintSettings` repainted the primary install
    button and never the "Install from this folder" one beside it, so a folder
    install left that button saying "Starting…" for good.

    Against the text-to-speech row, which is the one that still has the shape
    this defect was found in: one bundle, one Download, one folder box. Speech
    to text became three tier cards with a folder box each, and
    ``TestTheThreeQualities`` below is where that shape is checked.
    """

    def test_it_is_restored_after_a_folder_install_finishes(self):
        answers = status_answer()
        found = run("""
            await tick();
            settingsParts.ttsFolder.value = "C:\\\\Roots\\\\downloads";
            settingsParts.ttsLocal.fire("click");
            await tick();
            await repaint();
            console.log(JSON.stringify(report()));
        """, SETTINGS_PRESENT="true", ANSWERS=json.dumps(answers))

        assert found["settings"]["ttsLocalDisabled"] is False
        assert found["settings"]["ttsLocalLabel"] != "Starting…"

    def test_an_installed_bundle_offers_to_reinstall_rather_than_going_dead(self):
        found = run("await tick(); console.log(JSON.stringify(report()));",
                    SETTINGS_PRESENT="true")
        assert found["settings"]["ttsLocalLabel"] == "Reinstall from a folder"
        assert found["settings"]["ttsLocalDisabled"] is False

    def test_it_shows_progress_while_one_runs(self):
        answers = status_answer(tts_ready=False, ready=False, progress={
            "tts": {"running": True, "fraction": 0.5, "text": "Copying model.onnx…"}})
        found = run("await tick(); console.log(JSON.stringify(report()));",
                    SETTINGS_PRESENT="true", ANSWERS=json.dumps(answers))
        assert found["settings"]["ttsLocalLabel"] == "Installing…"
        assert found["settings"]["ttsLocalDisabled"] is True

    def test_a_failure_frees_it_too(self):
        answers = status_answer(stt_ready=False, ready=False, progress={
            "stt": {"running": False, "failed": True, "fraction": 0.0,
                    "text": "tokens.txt is missing."}})
        found = run("await tick(); console.log(JSON.stringify(report()));",
                    SETTINGS_PRESENT="true", ANSWERS=json.dumps(answers))
        assert found["settings"]["sttLocalDisabled"] is False


# --------------------------------------------------------------------------- #
# Streaming speech in the browser
# --------------------------------------------------------------------------- #


def stream_answers(chunks=None, *, seconds=None, count=1, **extra):
    """The default answers with a chosen set of network chunk boundaries.

    ``seconds`` asks for ``count`` blocks of that many seconds of 24 kHz PCM16
    each, built inside the harness -- a literal array of a hundred thousand
    zeroes is longer than ``node -e`` will accept as an argument.
    """
    answers = json.loads(DEFAULTS["ANSWERS"])
    if seconds is not None:
        chunks = {"plan": {"bytes": int(seconds * 24000) * 2, "count": count, "fill": 0}}
    answers["voice/tts-stream"] = dict(answers["voice/tts-stream"], chunks=chunks, **extra)
    return json.dumps(answers)


class TestTheStreamReader:
    def test_t_js_1_arbitrary_chunk_sizes_all_arrive(self):
        """Network reads are not sample-aligned and are not the server's writes.
        Six samples split three ways, one of the splits landing mid-sample."""
        found = run("await speak('T1'); console.log(JSON.stringify(report()));")
        total = sum(item["length"] for item in found["scheduled"])
        assert total == 6, found["scheduled"]

    def test_t_js_2_an_odd_trailing_byte_is_carried_and_not_played(self):
        """The first half of a sample whose second half is in the next chunk.
        Treating it as a whole sample turns the rest of the reply into noise."""
        found = run("await speak('T1'); console.log(JSON.stringify(report()));",
                    ANSWERS=stream_answers([[1, 0, 2], [0, 3, 0]]))
        blocks = [item for item in found["scheduled"] if item["length"]]
        assert sum(item["length"] for item in blocks) == 3
        # 1, 2 and 3 as little-endian int16 over 0x7fff.
        played = [value for item in blocks for value in item["samples"]]
        assert played[:3] == pytest.approx([1 / 0x7fff, 2 / 0x7fff, 3 / 0x7fff], rel=1e-3)

    def test_t_js_3_pcm16_becomes_float32_at_the_right_sign_and_scale(self):
        """0x8000 is -1.0 and 0x7fff is +1.0; getting the sign wrong is a reply
        that sounds inverted and the scale wrong is one that clips."""
        found = run("await speak('T1'); console.log(JSON.stringify(report()));",
                    ANSWERS=stream_answers([[0x00, 0x80, 0xff, 0x7f, 0x00, 0x00]]))
        samples = found["scheduled"][0]["samples"]
        assert samples[0] == pytest.approx(-1.0, abs=1e-4)
        assert samples[1] == pytest.approx(1.0, abs=1e-4)
        assert samples[2] == pytest.approx(0.0, abs=1e-6)

    def test_the_buffer_is_built_at_the_rate_the_header_declares(self):
        found = run("await speak('T1'); console.log(JSON.stringify(report()));",
                    ANSWERS=stream_answers([[1, 0, 2, 0]],
                                           headers={"X-Model-Chain-Voice-Rate": "22050",
                                                    "X-Model-Chain-Voice-Turn": "T1"}))
        assert found["scheduled"][0]["rate"] == 22050

    def test_blocks_are_scheduled_back_to_back_and_not_all_at_once(self):
        """Web Audio has no "append": back-to-back playback is arithmetic on the
        context's own clock, and getting it wrong plays every block over the
        first one."""
        found = run("await speak('T1'); console.log(JSON.stringify(report()));",
                    ANSWERS=stream_answers(seconds=1.0, count=2))
        times = [item["at"] for item in found["scheduled"] if item["length"]]
        assert len(times) == 2
        assert times[1] > times[0], times

    def test_a_stream_that_answers_for_a_different_turn_is_not_played(self):
        """Section 24 at the last possible moment: a response stamped with
        somebody else's turn is a response for a reply that is over."""
        found = run("await speak('T1'); console.log(JSON.stringify(report()));",
                    ANSWERS=stream_answers([[1, 0]],
                                           headers={"X-Model-Chain-Voice-Rate": "24000",
                                                    "X-Model-Chain-Voice-Turn": "OTHER"}))
        assert not [item for item in found["scheduled"] if item["length"]]

    def test_the_request_carries_the_turn_id_and_no_text(self):
        found = run("await speak('T1'); console.log(JSON.stringify(report()));")
        sent = [r for r in found["requests"] if r["url"].endswith("tts-stream")]
        assert sent and json.loads(sent[0]["bodyText"]) == {"turn": "T1"}
        assert sent[0]["headers"]["X-Model-Chain-Voice"] == "PAGE-TOKEN"

    def test_t_js_5_a_deep_queue_stops_the_reader_until_it_drains(self):
        """Section 22's high-water mark. Not reading is the whole mechanism:
        the socket stops being drained and the backpressure reaches the worker."""
        found = run("""
            turnField.value = "T1";
            await tick(6);
            const before = report().scheduled.length;
            // The audio clock does not move, so everything scheduled is still
            // queued and the reader should have stopped asking for more.
            await tick(20);
            console.log(JSON.stringify(report({before})));
        """, ANSWERS=stream_answers(seconds=2.0, count=8))
        queued = sum(item["length"] for item in found["scheduled"]) / 24000.0
        assert queued < 20, "the reader kept reading past the high-water mark"


# A stream answer with no turn stamp on it, for scenarios that speak more than
# one reply: the harness serves one canned response to every request, and a
# stamp naming the first turn is -- correctly -- refused for the second.
ANY_TURN = {"X-Model-Chain-Voice-Rate": "24000"}


def _playbacks(found: dict) -> list[dict]:
    """Every playback report the page made, as field dictionaries."""
    rows = []
    for request in found["requests"]:
        if not (request.get("url") or "").endswith("voice/telemetry"):
            continue
        payload = json.loads(request["bodyText"])
        if payload.get("kind") == "playback":
            rows.append(payload)
    return rows


def _captures(found: dict) -> list[dict]:
    rows = []
    for request in found["requests"]:
        if not (request.get("url") or "").endswith("voice/telemetry"):
            continue
        payload = json.loads(request["bodyText"])
        if payload.get("kind") == "capture":
            rows.append(payload)
    return rows


def _startup_targets(found: dict) -> list[int]:
    """The startup buffer each spoken turn actually used, in milliseconds."""
    return [row["startup_buffer_ms"] for row in _playbacks(found)]


def _underruns(found: dict) -> list[int]:
    return [row["underrun_count"] for row in _playbacks(found)]


class TestNoticingANewTurn:
    """Section 5.5. A 400 ms poll is 0-400 ms of latency on every reply, paid
    for a change the page could have been told about."""

    def test_the_input_event_starts_the_stream_without_the_poll(self):
        """`announceTurn` fires the event and runs the timers; it deliberately
        never runs the interval, so a request here can only have come from the
        event path."""
        found = run("""
            await announceTurn('T1');
            console.log(JSON.stringify(report()));
        """, ANSWERS=stream_answers(seconds=0.2, count=2))
        assert any(r["url"].endswith("tts-stream") for r in found["requests"])
        assert [item for item in found["scheduled"] if item["length"]]

    def test_a_value_written_with_no_event_is_still_recovered_by_the_poll(self):
        """`value` is an IDL property: assigning it fires nothing and changes no
        attribute an observer is watching. A theme or a future component that
        writes it directly must still be heard."""
        found = run("""
            turnField.value = "T1";
            await tick(8);
            const beforePoll = requests.filter((r) => r.url.indexOf("tts-stream") !== -1).length;
            await pump(20);
            console.log(JSON.stringify(report({beforePoll})));
        """, ANSWERS=stream_answers(seconds=0.2, count=2))
        assert found["beforePoll"] == 0, "the event path claimed a change nobody announced"
        assert any(r["url"].endswith("tts-stream") for r in found["requests"])

    def test_a_re_rendered_control_is_bound_again(self):
        """Gradio replaces the control, and a listener on a node that has left
        the document will never fire again."""
        found = run("""
            const replaced = rerenderField("mc-llm-chat-voice-turn");
            await tick(2);
            replaced.value = "T1";
            replaced.fire("input", {});
            await tick(20);
            console.log(JSON.stringify(report({
                bound: Object.keys(replaced.handlers).sort(),
            })));
        """, ANSWERS=stream_answers(seconds=0.2, count=2))
        assert found["bound"] == ["change", "input"]
        assert any(r["url"].endswith("tts-stream") for r in found["requests"])

    def test_re_binding_does_not_stack_listeners_on_the_same_control(self):
        """`wire` runs again after every Gradio update, and a second listener on
        a surviving control would speak one reply twice."""
        found = run("""
            updated.forEach((fn) => fn());
            updated.forEach((fn) => fn());
            observers.forEach((fn) => fn());
            await tick(2);
            console.log(JSON.stringify(report({
                inputs: (turnField.handlers.input || []).length,
                changes: (turnField.handlers.change || []).length,
            })));
        """)
        assert found["inputs"] == 1
        assert found["changes"] == 1

    def test_one_turn_still_opens_one_stream_however_it_is_noticed(self):
        """Both paths are live at once. The turn id is the guard, and it has to
        be, because the poll will see the same value the event announced."""
        found = run("""
            await announceTurn('T1', 10);
            await pump(20);
            console.log(JSON.stringify(report()));
        """, ANSWERS=stream_answers(seconds=0.2, count=2))
        opened = [r for r in found["requests"] if r["url"].endswith("tts-stream")]
        assert len(opened) == 1, opened


class TestTheStartupBuffer:
    """Section 5.6. Lower, and still adaptive: the point is earlier *continuous*
    audio, not earlier audio."""

    def test_playback_begins_well_before_the_old_one_and_a_half_seconds(self):
        found = run("""
            await speak('T1');
            console.log(JSON.stringify(report()));
        """, ANSWERS=stream_answers(seconds=0.25, count=8))
        played = [item for item in found["scheduled"] if item["length"]]
        assert played, "nothing was scheduled at all"
        # Everything released together is the prebuffer emptying, so the size of
        # the first release is the startup target the page actually used.
        first = played[0]["at"]
        held = sum(item["length"] for item in played if item["at"] == first) / 24000.0
        assert held <= 1.0, f"{held} s was held back before playback began"

    def test_a_short_reply_is_not_swallowed_by_the_prebuffer(self):
        """However little there is, the end of the stream releases it."""
        found = run("await speak('T1'); console.log(JSON.stringify(report()));",
                    ANSWERS=stream_answers([[1, 0, 2, 0]]))
        assert sum(item["length"] for item in found["scheduled"]) == 2

    def test_an_underrun_raises_the_target_for_the_next_turn(self):
        """The queue ran dry mid-sentence. The next reply starts with a deeper
        buffer because of it, which is the whole reason the number moves."""
        found = run("""
            // Read until the reader parks itself on the high-water mark, then
            // run the audio clock past everything scheduled and let it resume:
            // the next block arrives to find the speaker already silent, which
            // is exactly what an underrun is.
            turnField.value = "T1";
            await pump(6);
            advanceAudio(60);
            NOW += 1000;
            await pump(30);
            turnField.value = "";
            await pump(2);
            turnField.value = "T2";
            turnField.fire("input", {});
            for (let i = 0; i < 10; i += 1) { NOW += 500; advanceAudio(20); await pump(6); }
            console.log(JSON.stringify(report()));
        """, ANSWERS=stream_answers(seconds=1.0, count=20, headers=ANY_TURN))
        targets = _startup_targets(found)
        assert len(targets) == 2, found["consoleInfo"]
        assert _underruns(found)[0] >= 1, found["consoleInfo"]
        assert targets[1] > targets[0], found["consoleInfo"]

    def test_clean_turns_relax_the_target_again(self):
        """And back down, in small steps, so one bad moment does not cost every
        later reply its latency for the life of the tab."""
        found = run("""
            for (let i = 0; i < 7; i += 1) {
                turnField.value = "";
                await pump(2);
                await announceTurn('T' + i, 20);
            }
            console.log(JSON.stringify(report()));
        """, ANSWERS=stream_answers(seconds=0.2, count=4, headers=ANY_TURN))
        targets = _startup_targets(found)
        assert len(targets) == 7, found["consoleInfo"]
        assert _underruns(found) == [0] * 7, found["consoleInfo"]
        assert targets[-1] < targets[0], targets
        assert targets[-1] >= 400, "the buffer relaxed past its own floor"

    def test_the_target_a_turn_started_with_is_the_one_it_uses(self):
        """Raising the buffer in the middle of a reply would deepen the queue of
        the very sentence that is already playing."""
        found = run("""
            await speak('T1');
            console.log(JSON.stringify(report()));
        """, ANSWERS=stream_answers(seconds=0.25, count=8))
        targets = _startup_targets(found)
        assert targets == [700], found["consoleInfo"]

    def test_the_playback_report_carries_durations_and_nothing_else(self):
        """Content-free, and in one clock domain: every number is a difference
        between two `performance.now()` readings taken in this page. The turn id
        goes with it so that two records of one response can be recognised as
        one response, and nothing else does."""
        found = run("""
            await speak('T1');
            console.log(JSON.stringify(report()));
        """, ANSWERS=stream_answers(seconds=0.25, count=4))
        rows = _playbacks(found)
        assert len(rows) == 1, rows
        row = rows.pop()
        assert row["turn"] == "T1"
        assert row["playback_end_reason"] == "finished"
        for wanted in ("turn_seen_to_headers_ms", "headers_to_first_pcm_ms",
                       "first_pcm_to_playback_ms", "startup_buffer_ms", "underrun_count",
                       "max_underrun_gap_ms", "total_underrun_gap_ms"):
            assert wanted in row, wanted
        for name, value in row.items():
            if name in ("kind", "turn", "playback_end_reason"):
                continue
            assert value is None or isinstance(value, int), (name, value)


class TestPlaybackTelemetry:
    """The browser is the only party that can say whether the speaker ran dry.

    Server real-time factors cannot: a synthesis that kept up on average still
    produces a four-second hole if it fell behind once. The Web Audio scheduler
    knows the moment it discovers its next start time is already in the past,
    and how far in the past it is, which is the length of the silence somebody
    heard.
    """

    def test_the_gap_is_measured_and_not_only_counted(self):
        """A count alone cannot tell four imperceptible hiccups from one
        four-second hole, and those are different bugs."""
        found = run("""
            turnField.value = "T1";
            await pump(6);
            // The audio clock runs a long way past everything scheduled, so the
            // next block arrives to find the speaker silent -- and the distance
            // it ran past is the gap that was heard.
            advanceAudio(60);
            NOW += 1000;
            await pump(30);
            console.log(JSON.stringify(report()));
        """, ANSWERS=stream_answers(seconds=1.0, count=20, headers=ANY_TURN))
        rows = _playbacks(found)
        assert rows, "no playback report was sent"
        row = rows[0]
        assert row["underrun_count"] >= 1
        assert row["max_underrun_gap_ms"] > 0, row
        assert row["total_underrun_gap_ms"] >= row["max_underrun_gap_ms"]
        assert row["first_underrun_after_play_ms"] is not None

    def test_a_clean_turn_reports_no_starvation(self):
        found = run("await speak('T1'); console.log(JSON.stringify(report()));",
                    ANSWERS=stream_answers(seconds=0.25, count=4))
        row = _playbacks(found)[0]
        assert row["underrun_count"] == 0
        assert row["max_underrun_gap_ms"] == 0
        assert row["total_underrun_gap_ms"] == 0

    def test_a_cancelled_turn_says_so(self):
        found = run("""
            runState.value = "llm";
            await speak('T1', 4);
            elements["mc-llm-chat-stop"].fire("click", {});
            await pump(2);
            console.log(JSON.stringify(report()));
        """, ANSWERS=stream_answers(seconds=1.0, count=6, stall=True))
        rows = _playbacks(found)
        assert rows and rows[0]["playback_end_reason"] == "cancelled", rows

    def test_a_superseded_turn_says_it_was_replaced(self):
        """A reply cut off by the next reply is not a reply somebody stopped."""
        found = run("""
            await speak('T1', 6);
            turnField.value = "";
            await pump(2);
            await announceTurn('T2', 20);
            console.log(JSON.stringify(report()));
        """, ANSWERS=stream_answers(seconds=1.0, count=6, stall=True, headers=ANY_TURN))
        reasons = [row["playback_end_reason"] for row in _playbacks(found)]
        assert "replaced" in reasons, reasons

    def test_one_report_per_turn_however_many_stops_it_gets(self):
        """Two paths deliberately press Stop, and a page that reported twice
        would put one response in the log as two."""
        found = run("""
            runState.value = "llm";
            await speak('T1', 4);
            elements["mc-llm-chat-stop"].fire("click", {});
            elements["mc-llm-chat-stop"].fire("click", {});
            await pump(4);
            console.log(JSON.stringify(report()));
        """, ANSWERS=stream_answers(seconds=1.0, count=6, stall=True))
        assert len(_playbacks(found)) == 1, _playbacks(found)

    def test_a_telemetry_route_that_fails_changes_nothing(self):
        """The whole contract of this route. A page that cannot report its own
        timings has nothing wrong with its audio."""
        answers = json.loads(stream_answers(seconds=0.25, count=4))
        answers["voice/telemetry"] = {"reject": True}
        found = run("await speak('T1'); console.log(JSON.stringify(report()));",
                    ANSWERS=json.dumps(answers))
        assert [item for item in found["scheduled"] if item["length"]], "playback stopped"
        assert found["consoleErrors"] == []


class TestStoppingInTheBrowser:
    def test_t_js_7_stop_aborts_the_fetch_and_stops_every_source(self):
        """The audible half of one unified Stop, and the target is under 100 ms
        -- which is why it is a local listener rather than a round trip."""
        found = run("""
            await speak('T1', 4);
            elements["mc-llm-chat-stop"].fire("click", {});
            await pump(2);
            console.log(JSON.stringify(report()));
        """, ANSWERS=stream_answers(seconds=1.0, count=6, stall=True))
        assert found["aborts"] >= 1
        assert found["stopped"] >= 1
        assert any(r["url"].endswith("voice/cancel") for r in found["requests"])

    def test_the_cancel_names_the_turn_it_is_stopping(self):
        found = run("""
            await speak('T1', 4);
            elements["mc-llm-chat-stop"].fire("click", {});
            await pump(2);
            console.log(JSON.stringify(report()));
        """, ANSWERS=stream_answers(seconds=1.0, count=6, stall=True))
        cancel = [r for r in found["requests"]
                  if (r.get("url") or "").endswith("voice/cancel")][0]
        body = json.loads(cancel["bodyText"])
        assert body["turn"] == "T1"
        # A fixed word, never text and never a message: it is going into a log.
        assert body["reason"] == "user"

    def test_t_js_8_a_new_turn_stops_the_previous_one_before_it_plays(self):
        found = run("""
            await speak('T1', 4);
            await speak('T2', 20);
            console.log(JSON.stringify(report()));
        """, ANSWERS=stream_answers(seconds=1.0, count=6))
        assert found["stopped"] >= 1, "the first reply was not stopped"
        assert found["aborts"] >= 1

    def test_t_js_10_turning_auto_speak_off_stops_the_current_reply(self):
        """Section 28. A switch that says "do not read replies aloud" while a
        reply is being read aloud is a switch nobody trusts again."""
        found = run("""
            await speak('T1', 4);
            const box = elements["mc-llm-chat-voice-auto-speak"];
            box.type = "checkbox";
            box.checked = false;
            box.fire("change", {target: box});
            await pump(2);
            console.log(JSON.stringify(report()));
        """, ANSWERS=stream_answers(seconds=1.0, count=6, stall=True))
        assert found["aborts"] >= 1
        assert any(r["url"].endswith("voice/cancel") for r in found["requests"])

    def test_t_js_11_the_microphone_cancels_the_reply_before_it_records(self):
        """Section 14. Not merely silencing the speaker -- an obsolete Kokoro
        run would go on taking the CPU the transcription is about to need."""
        found = run("""
            await speak('T1', 4);
            await hold(900);
            console.log(JSON.stringify(report()));
        """, ANSWERS=stream_answers(seconds=1.0, count=6, stall=True))
        order = [r.get("url") or "" for r in found["requests"]]
        cancels = [index for index, url in enumerate(order) if url.endswith("voice/cancel")]
        transcriptions = [index for index, url in enumerate(order)
                          if url.endswith("voice/stt")]
        assert cancels, order
        assert transcriptions, order
        assert cancels[0] < transcriptions[0], order

    def test_t_js_12_leaving_the_page_stops_the_speaker_and_the_stream(self):
        found = run("""
            await speak('T1', 4);
            (globalThis.windowHandlers["pagehide"] || []).forEach((fn) => fn());
            await pump(2);
            console.log(JSON.stringify(report()));
        """, ANSWERS=stream_answers(seconds=1.0, count=6, stall=True))
        assert found["aborts"] >= 1
        assert found["stopped"] >= 1


class TestWhatThePageCostsToLoad:
    """Section 33, and a bug report. This script is in every page this WebUI
    serves, including the ones with nothing to do with speech, and it used to
    make three requests and rewrite two class attributes twice a second whether
    or not anybody was looking at anything it draws. On a theme that observes
    the DOM -- which is most of them -- that is work with no end and no reader,
    and the tab that pays for it is whichever one is trying to paint."""

    def test_a_settings_row_nobody_can_see_is_asked_nothing(self):
        found = run("""
            settingsRow.offsetParent = null;
            settingsRow.getBoundingClientRect = () => ({width: 0, height: 0});
            await repaint();
            await repaint();
            console.log(JSON.stringify(report()));
        """, SETTINGS_PRESENT="true")
        assert not any("voice/status" in (r.get("url") or "") for r in found["requests"])

    def test_the_voices_row_is_not_fetched_until_it_is_on_screen(self):
        found = run("""
            await tick(2);
            const before = requests.length;
            voicesRow.offsetParent = {};
            NOW += 2000;
            await tick(4);
            console.log(JSON.stringify(report({before})));
        """, VOICES_PRESENT="true")
        early = found["requests"][:found["before"]]
        assert not any("voice/voices" in (r.get("url") or "") for r in early), \
            "the voice list was fetched before anybody could see it"
        assert any("voice/voices" in (r.get("url") or "") for r in found["requests"]), \
            "the voice list was never fetched once it was visible"

    def test_an_idle_tick_rewrites_no_class_attributes(self):
        """A `classList.remove` of a class that is not there still rewrites the
        element's `class` attribute, and every MutationObserver on the page
        wakes up for it. Twice a second, forever, is what a heavy theme could
        not settle under."""
        found = run("""
            await tick(2);
            const before = send.classList.writes + stop.classList.writes;
            intervals.forEach((fn) => fn());
            intervals.forEach((fn) => fn());
            await tick(2);
            console.log(JSON.stringify(report({
                before, after: send.classList.writes + stop.classList.writes,
            })));
        """)
        assert found["after"] == found["before"], \
            "an idle composer tick still wrote to the DOM"

    def test_a_stale_page_token_stops_the_polling_after_one_refusal(self):
        """The observed failure: a tab left open across a WebUI restart carries
        last run's page token, and there is no answer it can get but 403. It
        used to ask again every 1.5 seconds for as long as it stayed open --
        against a WebUI that had just started, out of the same small pool of
        connections the tab actually being loaded needs."""
        answers = json.loads(DEFAULTS["ANSWERS"])
        answers["voice/status"] = {"status": 403,
                                   "json": {"ok": False, "error": "reload it"}}
        found = run("""
            await repaint();
            const after = requests.length;
            await repaint();
            await repaint();
            console.log(JSON.stringify(report({after})));
        """, SETTINGS_PRESENT="true", ANSWERS=json.dumps(answers))
        assert found["after"] >= 1, "the first request was never made"
        assert len(found["requests"]) == found["after"], \
            "a refused page went on asking"


class TestTheComposerBusyState:
    def test_t_js_9_stop_stays_while_voice_is_still_speaking(self):
        """Release blocker four. The reply finished, Gradio put Send back, and
        the speaker was still talking with nothing on screen to stop it."""
        found = run("""
            runState.value = "llm";
            await speak('T1', 4);
            runState.value = "idle";
            await pump(2);
            console.log(JSON.stringify(report()));
        """, ANSWERS=stream_answers(seconds=2.0, count=6))
        assert found["sendHidden"] is True
        assert found["stopHidden"] is False

    def test_stop_is_revealed_and_not_merely_left_alone(self):
        """The half that is easy to miss. Python's own IDLE update hides Stop
        when the model finishes -- correctly, as far as Python knows -- so a
        composer where Voice is still speaking needs Gradio's marker taken
        *off* Stop rather than a class of ours added to Send."""
        found = run("""
            runState.value = "llm";
            await speak('T1', 4);
            runState.value = "idle";
            // Gradio hides Stop the way it hides anything.
            elements["mc-llm-chat-stop"].classList.add("hidden");
            await pump(2);
            console.log(JSON.stringify(report({
                stopStillHidden: elements["mc-llm-chat-stop"].classList.contains("hidden"),
            })));
        """, ANSWERS=stream_answers(seconds=2.0, count=6, stall=True))
        assert found["stopStillHidden"] is False
        assert found["stopHidden"] is False
        assert found["sendHidden"] is True

    def test_stop_is_actionable_and_not_merely_visible_during_voice(self):
        """S-2. The bug this closes: Python built Stop `interactive=False`, the
        browser revealed it when the model went idle and Voice kept speaking,
        and what appeared was a Stop that could not be pressed. A disabled
        button dispatches no click, so even the capture-phase listener that
        silences the speaker locally never ran."""
        found = run("""
            runState.value = "llm";
            await speak('T1', 4);
            runState.value = "idle";
            // Gradio re-rendering the component and reasserting its
            // server-side attributes, which it may do at any point.
            elements["mc-llm-chat-stop"].disabled = true;
            elements["mc-llm-chat-stop"].setAttribute("aria-disabled", "true");
            await pump(2);
            console.log(JSON.stringify(report()));
        """, ANSWERS=stream_answers(seconds=2.0, count=6, stall=True))
        assert found["stopHidden"] is False
        assert found["stopDisabled"] is False

    def test_stop_stays_actionable_while_the_model_is_running(self):
        """S-1, and the same assertion from the other side."""
        found = run("""
            runState.value = "llm";
            elements["mc-llm-chat-stop"].disabled = true;
            await pump(2);
            console.log(JSON.stringify(report()));
        """)
        assert found["stopHidden"] is False
        assert found["stopDisabled"] is False

    def test_one_click_silences_playback_and_cancels_the_server(self):
        """S-3. The browser's half is immediate and local; the POST is what
        stops Kokoro computing speech nobody is going to hear."""
        found = run("""
            runState.value = "llm";
            await speak('T1', 4);
            runState.value = "idle";
            await pump(2);
            elements["mc-llm-chat-stop"].fire("click", {});
            await pump(2);
            console.log(JSON.stringify(report()));
        """, ANSWERS=stream_answers(seconds=2.0, count=6, stall=True))
        assert found["stopped"] >= 1, "scheduled audio was left playing"
        assert found["aborts"] >= 1, "the stream request was not aborted"
        assert any(r["url"].endswith("voice/cancel") for r in found["requests"])

    def test_a_second_click_is_harmless(self):
        """S-5. Two paths deliberately press this button, and a cancellation
        race is an ordinary thing rather than an error."""
        found = run("""
            runState.value = "llm";
            await speak('T1', 4);
            runState.value = "idle";
            await pump(2);
            elements["mc-llm-chat-stop"].fire("click", {});
            elements["mc-llm-chat-stop"].fire("click", {});
            await pump(4);
            console.log(JSON.stringify(report()));
        """, ANSWERS=stream_answers(seconds=2.0, count=6, stall=True))
        assert found["consoleErrors"] == []
        assert found["sendHidden"] is False, "the composer was left showing Stop"

    def test_send_comes_back_only_when_both_are_idle(self):
        found = run("""
            runState.value = "llm";
            await pump(2);
            const busy = report();
            runState.value = "idle";
            await pump(2);
            console.log(JSON.stringify(report({wasBusy: busy.stopHidden})));
        """)
        assert found["wasBusy"] is False, "Stop was hidden while the model was running"
        assert found["sendHidden"] is False
        assert found["stopHidden"] is True


class TestTheEnginePanel:
    def test_load_and_unload_go_to_the_runtime_route(self):
        found = run("""
            const panel = elements["mc-llm-chat-voice-engine"];
            panel.offsetParent = {};
            panel.fire("click", {target: panel.button});
            await pump(2);
            console.log(JSON.stringify(report()));
        """)
        sent = [r for r in found["requests"] if r["url"].endswith("voice/runtime")]
        assert sent, [r["url"] for r in found["requests"]]
        assert json.loads(sent[0]["bodyText"])["action"] in ("load", "unload")

    def test_unloading_stops_a_reply_that_is_being_read_aloud(self):
        found = run("""
            await speak('T1', 4);
            const panel = elements["mc-llm-chat-voice-engine"];
            panel.offsetParent = {};
            panel.button.setAttribute("data-mc-voice-runtime", "unload");
            panel.fire("click", {target: panel.button});
            await pump(2);
            console.log(JSON.stringify(report()));
        """, ANSWERS=stream_answers(seconds=1.0, count=6, stall=True))
        assert found["aborts"] >= 1
