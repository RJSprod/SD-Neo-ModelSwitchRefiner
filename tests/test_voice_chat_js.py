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
["mc-llm-chat-voice-token", "mc-llm-chat-voice-key",
 "mc-llm-chat-message"].forEach(function (id) {
    elements[id] = field(id);
});
elements["mc-llm-chat-status"] = element("mc-llm-chat-status");
elements["mc-llm-chat-status"].notice = element("notice");

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

const settingsRow = element("settings");
settingsRow["data-mc-voice-key"] = "PAGE-TOKEN";
settingsRow.querySelector = function (selector) {
    if (selector === ".mc-voice-runtime") return settingsParts.runtime;
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
        return [settingsParts.sttButton, settingsParts.ttsButton];
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
globalThis.addEventListener = () => {};
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

const context = {
    state: CONTEXT_STATE,
    sampleRate: SAMPLE_RATE,
    audioWorklet: WORKLET_AVAILABLE ? {addModule: () => Promise.resolve()} : null,
    resume() { context.state = RESUME_WORKS ? "running" : "suspended"; return Promise.resolve(); },
    createMediaStreamSource() { return {connect() {}, disconnect() {}}; },
    createScriptProcessor() {
        const node = {connect() {}, disconnect() {}, onaudioprocess: null};
        globalThis.processorNode = node;
        return node;
    },
    createGain() { return {gain: {}, connect() {}, disconnect() {}}; },
    createBufferSource() {
        const source = {
            buffer: null, onended: null,
            connect() {}, disconnect() {},
            start() { played.push(source); },
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
                    const error = new Error("denied");
                    error.name = "NotAllowedError";
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

globalThis.fetch = function (url, options) {
    requests.push({url, options: options || {},
                   body: options && options.body,
                   headers: (options && options.headers) || {}});
    const answer = ANSWERS[Object.keys(ANSWERS).find((k) => url.indexOf(k) !== -1)];
    if (!answer) return Promise.reject(new Error("no route " + url));
    if (answer.reject) return Promise.reject(new Error("network"));
    return Promise.resolve({
        ok: answer.status === undefined || answer.status < 400,
        status: answer.status || 200,
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
    "SAMPLE_RATE": "48000",
    "CONTEXT_STATE": '"running"',
    "RESUME_WORKS": "true",
    "DECODE_WORKS": "true",
    "WORKLET_AVAILABLE": "true",
    "GRADIO_CONFIG": '{"root": ""}',
    "ANSWERS": json.dumps({
        "voice/status": {"json": {"ok": True, "ready": True, "stt_ready": True,
                                  "tts_ready": True, "runtime_ready": True,
                                  "auto_send": False, "auto_speak": False,
                                  "progress": {}, "runtime_message": "Installed",
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
    def test_an_insecure_context_is_explained_and_nothing_is_opened(self):
        """From an Android phone, ``http://192.168.x.x`` is not a secure context
        and ``getUserMedia`` does not exist. Saying so is the whole feature
        here."""
        found = run("await hold(900); console.log(JSON.stringify(report()));",
                    SECURE="false")
        assert "HTTPS" in found["status"]
        assert not any(r.get("kind") == "getUserMedia" for r in found["requests"])

    def test_a_browser_without_getusermedia_fails_cleanly(self):
        found = run("await hold(900); console.log(JSON.stringify(report()));",
                    MICROPHONE="false")
        assert found["status"]
        assert found["consoleErrors"] == []

    def test_a_denied_permission_says_so_and_does_not_retry(self):
        found = run("await hold(900); console.log(JSON.stringify(report()));",
                    PERMISSION="false")
        assert "permission was denied" in found["status"]
        opens = [r for r in found["requests"] if r.get("kind") == "getUserMedia"]
        assert len(opens) == 1, "a denied permission was asked for again"

    def test_a_missing_setup_flashes_an_error_and_never_opens_the_microphone(self):
        answers = json.loads(DEFAULTS["ANSWERS"])
        answers["voice/status"]["json"].update({"ready": False, "stt_ready": False})
        found = run("await hold(900); console.log(JSON.stringify(report()));",
                    ANSWERS=json.dumps(answers))
        assert "not set up" in found["status"]
        assert "mc-llm-voice-error" in found["micClasses"]
        assert not any(r.get("kind") == "getUserMedia" for r in found["requests"])

    def test_it_will_not_record_while_a_reply_is_generating(self):
        found = run("""
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
            elements["mc-llm-chat-stop"].offsetParent = {};
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
        assert found["settings"]["runtime"] == "Installed"
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
