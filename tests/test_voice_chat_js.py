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
import shutil
import subprocess
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

function element(id, tag) {
    const listeners = {};
    const node = {
        id,
        tagName: tag || "DIV",
        dataset: {},
        disabled: false,
        value: "",
        offsetParent: {},
        innerHTML: "",
        textContent: "",
        children: [],
        classList: {
            names: new Set(),
            add(...names) { names.forEach((n) => node.classList.names.add(n)); },
            remove(...names) { names.forEach((n) => node.classList.names.delete(n)); },
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
    return null;
};
settingsRow.querySelectorAll = function (selector) {
    if (selector === "[data-mc-voice-install]") {
        return [settingsParts.runtimeButton, settingsParts.sttButton, settingsParts.ttsButton];
    }
    if (selector === "[data-mc-voice-local]") {
        return [settingsParts.sttLocal, settingsParts.ttsLocal];
    }
    return [];
};

globalThis.document = {
    documentElement: element("html"),
    querySelector(selector) {
        if (selector.charAt(0) === "#") return elements[selector.slice(1)] || null;
        if (selector === ".mc-voice-settings") return SETTINGS_PRESENT ? settingsRow : null;
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
globalThis.location = {pathname: "/", origin: "https://forge.example"};
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
globalThis.Date = class extends Date { static now() { return NOW; } };
const windowHandlers = {};
globalThis.windowHandlers = windowHandlers;
globalThis.addEventListener = (name, handler) => {
    (windowHandlers[name] = windowHandlers[name] || []).push(handler);
};
globalThis.console = Object.assign({}, console, {
    error(...args) { consoleErrors.push(args.map(String).join(" ")); },
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

function track() {
    const item = {stopped: false, stop() { item.stopped = true; }};
    tracks.push(item);
    return item;
}

const scheduled = [];

const context = {
    state: CONTEXT_STATE,
    sampleRate: SAMPLE_RATE,
    currentTime: 0,
    audioWorklet: WORKLET_AVAILABLE ? {addModule: () => Promise.resolve()} : null,
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
    value: MICROPHONE ? {
        mediaDevices: {
            getUserMedia(constraints) {
                requests.push({kind: "getUserMedia", constraints});
                if (!PERMISSION) {
                    const error = new Error("refused");
                    error.name = DENIAL;
                    return Promise.reject(error);
                }
                return Promise.resolve({getTracks: () => [track()]});
            },
        },
    } : {},
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
    return Promise.resolve({
        ok: answer.status === undefined || answer.status < 400,
        status: answer.status || 200,
        headers: {get: (name) => headerBag[name] === undefined ? null : headerBag[name]},
        body: answer.chunks ? streamBody(answer.chunks, answer.stall) : null,
        json: () => Promise.resolve(answer.json),
        arrayBuffer: () => Promise.resolve(answer.audio || new ArrayBuffer(8)),
    });
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

async function hold(ms, sampleCount) {
    mic.fire("pointerdown", {pointerId: 7});
    await tick();
    feed(new Array(sampleCount === undefined ? 8000 : sampleCount).fill(0.2));
    NOW += ms;
    mic.fire("pointerup", {pointerId: 7});
    await tick();
}

function report(extra) {
    return Object.assign({
        requests: requests.map((r) => ({url: r.url, kind: r.kind,
                                        headers: r.headers,
                                        bodyLength: r.body && r.body.byteLength,
                                        bodyText: typeof r.body === "string" ? r.body : null})),
        micClasses: Array.from(mic.classList.names),
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
        pendingTimers: pending(),
        consoleErrors,
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
            runtimeLine: settingsParts.runtimeLine.textContent,
            runtimeButton: settingsParts.runtimeButton.textContent,
            runtimeDisabled: settingsParts.runtimeButton.disabled,
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
    "PERMISSION": "true",
    "DENIAL": '"NotAllowedError"',
    "SAMPLE_RATE": "48000",
    "CONTEXT_STATE": '"running"',
    "RESUME_WORKS": "true",
    "DECODE_WORKS": "true",
    "WORKLET_AVAILABLE": "true",
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
    }),
}


def status_answer(**changes):
    """The status payload, with a few fields changed."""
    answers = json.loads(DEFAULTS["ANSWERS"])
    answers["voice/status"]["json"].update(changes)
    return answers


def run(scenario: str, **overrides) -> dict:
    settings = dict(DEFAULTS)
    settings.update({key: value for key, value in overrides.items()})
    harness = HARNESS.replace("SOURCE", SCRIPT.read_text()).replace("SCENARIO", scenario)
    for key, value in settings.items():
        harness = harness.replace(key, value if isinstance(value, str) else json.dumps(value))
    result = subprocess.run(["node", "--input-type=module", "-e", harness],
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
        assert all(url.endswith("/model-chain/voice/status")
                   or url.endswith("/model-chain/voice/stt") for url in urls(found))

    def test_every_request_carries_the_page_token(self):
        found = run("await hold(900); console.log(JSON.stringify(report()));")
        for request in found["requests"]:
            if request.get("url"):
                assert request["headers"]["X-Model-Chain-Voice"] == "PAGE-TOKEN"


# --------------------------------------------------------------------------- #
# The gesture
# --------------------------------------------------------------------------- #


class TestPressAndHold:
    def test_a_hold_records_and_posts_exactly_one_recording(self):
        found = run("await hold(900); console.log(JSON.stringify(report()));")
        posts = [r for r in found["requests"] if r.get("url", "").endswith("/stt")]
        assert len(posts) == 1
        assert posts[0]["bodyLength"] > 44, "the WAV had no samples in it"

    def test_pointerdown_starts_exactly_one_capture(self):
        found = run("""
            mic.fire("pointerdown", {pointerId: 1});
            await tick();
            mic.fire("pointerdown", {pointerId: 2});
            await tick();
            console.log(JSON.stringify(report()));
        """)
        opens = [r for r in found["requests"] if r.get("kind") == "getUserMedia"]
        assert len(opens) == 1, "a second press opened a second microphone"

    def test_the_microphone_is_closed_when_the_finger_comes_up(self):
        """The most alarming thing a page can do is leave the recording
        indicator on."""
        found = run("await hold(900); console.log(JSON.stringify(report()));")
        assert found["tracks"], "no microphone was ever opened"
        assert all(found["tracks"]), "a MediaStream track was left running"

    def test_pointercancel_discards_the_recording_and_closes_the_microphone(self):
        found = run("""
            mic.fire("pointerdown", {pointerId: 3});
            await tick();
            feed(new Array(8000).fill(0.2));
            NOW += 900;
            mic.fire("pointercancel", {pointerId: 3});
            await tick();
            console.log(JSON.stringify(report()));
        """)
        assert not any(r.get("url", "").endswith("/stt") for r in found["requests"])
        assert all(found["tracks"])

    def test_a_tap_does_not_reach_the_server(self):
        """Under 250 ms is somebody finding out what the button does."""
        found = run("await hold(120); console.log(JSON.stringify(report()));")
        assert not any(r.get("url", "").endswith("/stt") for r in found["requests"])
        assert "Hold to speak" in found["status"]

    def test_the_finger_is_captured_so_sliding_off_still_ends_the_recording(self):
        found = run("""
            mic.fire("pointerdown", {pointerId: 11});
            await tick();
            console.log(JSON.stringify(report({captured: mic.captured})));
        """)
        assert found["captured"] == 11

    def test_a_sixty_second_cap_is_armed_and_does_not_fire_early(self):
        found = run("""
            mic.fire("pointerdown", {pointerId: 4});
            await tick();
            console.log(JSON.stringify(report({armed: pending()})));
        """)
        assert found["armed"] >= 1

    def test_the_cap_stops_the_recording_by_itself(self):
        found = run("""
            mic.fire("pointerdown", {pointerId: 5});
            await tick();
            feed(new Array(8000).fill(0.2));
            NOW += 60000;
            await tick(12);
            console.log(JSON.stringify(report()));
        """)
        assert any(r.get("url", "").endswith("/stt") for r in found["requests"])

    def test_recording_state_is_shown_without_a_round_trip(self):
        found = run("""
            mic.fire("pointerdown", {pointerId: 6});
            console.log(JSON.stringify(report()));
        """)
        assert "mc-llm-voice-recording" in found["micClasses"]
        assert found["micLabel"] == "Recording — release to transcribe"

    def test_the_control_returns_to_idle_afterwards(self):
        found = run("await hold(900); console.log(JSON.stringify(report()));")
        assert "mc-llm-voice-recording" not in found["micClasses"]
        assert found["micLabel"] == "Hold to dictate"


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

    def test_a_missing_setup_flashes_an_error_and_never_opens_the_microphone(self):
        answers = json.loads(DEFAULTS["ANSWERS"])
        answers["voice/status"]["json"].update({"ready": False, "stt_ready": False})
        found = run("await hold(900); console.log(JSON.stringify(report()));",
                    ANSWERS=json.dumps(answers))
        assert "not set up" in found["status"]
        assert "mc-llm-voice-error" in found["micClasses"]
        assert not any(r.get("kind") == "getUserMedia" for r in found["requests"])

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
            mic.fire("pointerdown", {pointerId: 7});
            await tick();
            feed(new Array(8000).fill(0.2));
            NOW += 900;
            mic.fire("pointerup", {pointerId: 7});
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
            mic.fire("pointerdown", {pointerId: 9});
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
        assert found["settings"]["sttButton"] == "Installed"
        assert found["settings"]["sttDisabled"] is True

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


class TestTheFolderButtonComesBack:
    """The second stuck button. `paintSettings` repainted the primary install
    button and never the "Install from this folder" one beside it, so a folder
    install left that button saying "Starting…" for good."""

    def test_it_is_restored_after_a_folder_install_finishes(self):
        answers = status_answer()
        found = run("""
            await tick();
            settingsParts.sttFolder.value = "C:\\\\Roots\\\\downloads";
            settingsParts.sttLocal.fire("click");
            await tick();
            await repaint();
            console.log(JSON.stringify(report()));
        """, SETTINGS_PRESENT="true", ANSWERS=json.dumps(answers))

        assert found["settings"]["sttLocalDisabled"] is False
        assert found["settings"]["sttLocalLabel"] != "Starting…"

    def test_an_installed_bundle_offers_to_reinstall_rather_than_going_dead(self):
        found = run("await tick(); console.log(JSON.stringify(report()));",
                    SETTINGS_PRESENT="true")
        assert found["settings"]["sttLocalLabel"] == "Reinstall from a folder"
        assert found["settings"]["sttLocalDisabled"] is False

    def test_it_shows_progress_while_one_runs(self):
        answers = status_answer(stt_ready=False, ready=False, progress={
            "stt": {"running": True, "fraction": 0.5, "text": "Copying encoder.onnx…"}})
        found = run("await tick(); console.log(JSON.stringify(report()));",
                    SETTINGS_PRESENT="true", ANSWERS=json.dumps(answers))
        assert found["settings"]["sttLocalLabel"] == "Installing…"
        assert found["settings"]["sttLocalDisabled"] is True

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
        cancel = [r for r in found["requests"] if r["url"].endswith("voice/cancel")][0]
        assert json.loads(cancel["bodyText"]) == {"turn": "T1"}

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
