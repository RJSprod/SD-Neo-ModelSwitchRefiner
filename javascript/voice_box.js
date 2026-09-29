// The Voice Box page: INPUT -> PROMPT -> CONFIGURATION -> OUTPUTS.
//
// Python paints one element, `#mc-voice-box` (mc_voice_box_ui.py), and this
// file draws everything inside it. That shape is deliberate and comes from
// Mini Paint NEO's Gradio 4.40 traps, adopted by docs/23-voice-box.md section 8:
//
//   * state is fetched on boot and after every change, never pushed through
//     an event graph, so there is no outputs list for another tab's component
//     to poison and no "equal string is no change" repaint to get wrong;
//   * every request has a deadline (AbortController: 20 s for JSON, 120 s for
//     audio and uploads) and a second press while one is in flight waits for
//     it rather than adding another;
//   * render progress is read by polling /status -- every second while a job
//     this page started is live, every fifteen while the tab is visible and
//     idle, not at all while it is hidden -- never through an open stream;
//   * audio bytes always arrive through fetch with the page token in a header,
//     into blob URLs. A bare `src=` to a token-checked route would be a 403 the
//     user could not see.
//
// Audio focus is one event on `document`, "mc:audio-focus", `{owner, kind}`:
// this page announces itself before it plays or records, and gives way -- every
// player paused, the recorder stopped -- to any owner that is not itself. Voice
// Chat's side is in javascript/voice_chat.js.
//
// The file is loaded by Forge on every page, like every extension script, so
// nothing here touches the DOM until the root exists.

(function () {
    "use strict";

    const ROOT_ID = "mc-voice-box";
    const DEFAULT_PREFIX = "/model-chain/voice-box";
    const TOKEN_HEADER = "x-model-chain-voice";
    const FOCUS_EVENT = "mc:audio-focus";
    const OWNER = "voice-box";
    const STORAGE_KEY = "mc-voice-box:pipeline";
    const DEADLINE = {json: 20000, audio: 120000};
    const POLL = {live: 1000, idle: 15000};
    const SAVE_DEBOUNCE_MS = 1000;
    const SAMPLE_RATE = 24000;
    // The server's envelope for a sample (mc_voice_reference.normalize): a
    // selection shorter than three seconds or longer than sixty would be
    // refused after the upload, so the trimmer keeps it inside from the start.
    const SELECTION = {min: 3, max: 60};
    // Above either bound a file is captured in real time instead of decoded
    // whole: decodeAudioData holds the entire decoded PCM in memory, and twenty
    // minutes of 48 kHz stereo is close to half a gigabyte of it.
    const DECODE_LIMIT = {bytes: 200 * 1024 * 1024, seconds: 20 * 60};
    const PEAK_BUCKETS = 240;
    const LIVE_PHASES = {queued: true, waiting: true, loading: true, rendering: true};
    const STAGES = [
        {key: "input", title: "INPUT"},
        {key: "prompt", title: "PROMPT"},
        {key: "configuration", title: "CONFIGURATION"},
        {key: "outputs", title: "OUTPUTS"},
    ];
    const PROMPT_HELP = "Speaker 1: to Speaker 4: start a line ([2]: works too); a line "
        + "without one is Speaker 1. [pause] is 700 ms and [pause:1500] is 1500 ms of "
        + "silence between separately rendered sections.";

    const ROUTES = {
        status: "/status",
        install: "/install",
        settings: "/settings",
        folder: "/settings/folder",
        samples: "/samples",
        sampleUpload: "/samples/upload",
        sampleRename: "/samples/rename",
        sampleDelete: "/samples/delete",
        sampleAudio: "/samples/audio",
        prompts: "/prompts",
        promptAdd: "/prompts/add",
        promptFavourite: "/prompts/favourite",
        promptDelete: "/prompts/delete",
        configurations: "/configurations",
        configurationSave: "/configurations/save",
        configurationDelete: "/configurations/delete",
        pipelines: "/pipelines",
        pipelineNew: "/pipelines/new",
        pipelineSave: "/pipelines/save",
        pipelineDelete: "/pipelines/delete",
        render: "/render",
        jobCancel: "/jobs/cancel",
        outputs: "/outputs",
        outputRename: "/outputs/rename",
        outputLoop: "/outputs/loop",
        outputDelete: "/outputs/delete",
        outputSave: "/outputs/save",
        outputAudio: "/outputs/audio",
        runtime: "/runtime",
    };

    // -- state ----------------------------------------------------------------- //

    // Plain data only: `window.mcVoiceBox.state()` hands this out, and a test
    // reads it as JSON. Elements live in `nodes`; blobs and object URLs in
    // `cache`; the trimmer's decoded audio in `trimmer`.
    const state = {
        booted: false,
        token: "",
        prefix: DEFAULT_PREFIX,
        status: null,
        samples: [],
        prompts: {history: [], favourites: []},
        configurations: [],
        pipelines: [],
        outputs: [],
        jobs: [],
        ownJobs: {},
        pipelineId: "",
        configurationId: "",
        prompt: "",
        working: null,
        dirty: {prompt: false, configuration: false},
        poll: {delay: 0, live: false},
        recording: false,
        capturing: false,
        message: "",
        messageKind: "",
        focus: {last: null, sent: []},
    };

    const nodes = {lanes: {}};
    const cache = {urls: {}, blobs: {}};
    const players = [];
    const inflight = {};
    const timers = {prompt: 0, poll: 0};

    // -- small helpers --------------------------------------------------------- //

    function pick(value, fallback) {
        return value === undefined || value === null ? fallback : value;
    }

    function el(tag, className, text) {
        const node = document.createElement(tag);
        if (className) node.className = className;
        if (text !== undefined && text !== null) node.textContent = String(text);
        return node;
    }

    function button(label, ariaLabel, onClick, className) {
        const node = el("button", "mc-voice-box-button" + (className ? " " + className : ""),
                        label);
        node.setAttribute("type", "button");
        if (ariaLabel) node.setAttribute("aria-label", ariaLabel);
        if (onClick) {
            node.addEventListener("click", function (event) {
                if (event && typeof event.preventDefault === "function") event.preventDefault();
                try {
                    const result = onClick(event);
                    if (result && typeof result.catch === "function") result.catch(report);
                } catch (error) {
                    report(error);
                }
            });
        }
        return node;
    }

    function clear(node) {
        if (!node) return;
        while (node.firstChild) node.removeChild(node.firstChild);
    }

    function show(node, on) {
        if (!node) return;
        node.hidden = !on;
        if (on) node.removeAttribute("hidden"); else node.setAttribute("hidden", "");
    }

    function pressed(node, on) {
        if (node) node.setAttribute("aria-pressed", on ? "true" : "false");
    }

    function seconds(value) {
        const number = Number(value) || 0;
        return (Math.round(number * 10) / 10).toFixed(1) + " s";
    }

    function gigabytes(bytes) {
        return (Math.round((Number(bytes) || 0) / 1e8) / 10).toFixed(1) + " GB";
    }

    function safeName(name) {
        return String(name || "voice-box").replace(/[\\/:*?"<>|]+/g, "-").trim() || "voice-box";
    }

    function focused(node) {
        try {
            return !!node && document.activeElement === node;
        } catch (error) {
            return false;
        }
    }

    function ask(question, fallback) {
        // The host's own prompt dialog for a name, where there is one. A
        // browser without it (or a test) gets the fallback, and the name can be
        // changed afterwards through Rename or a double click.
        if (typeof window.prompt !== "function") return fallback;
        let answer = null;
        try {
            answer = window.prompt(question, fallback);
        } catch (error) {
            answer = fallback;
        }
        if (answer === null) return null;
        answer = String(answer).trim();
        return answer || fallback;
    }

    // -- the status line ------------------------------------------------------- //

    function say(text, kind) {
        state.message = text || "";
        state.messageKind = kind || "info";
        if (!nodes.status) return;
        nodes.status.textContent = state.message;
        nodes.status.setAttribute("data-kind", state.messageKind);
    }

    class RequestError extends Error {
        constructor(message, status, timedOut) {
            super(message);
            this.status = status || 0;
            this.timedOut = !!timedOut;
        }
    }

    function describe(error) {
        if (!error) return "Something went wrong.";
        if (typeof error === "string") return error;
        return error.message || String(error);
    }

    // Every failure ends here: in the status line, never thrown away. It is
    // also a valid promise rejection handler, so a chain can end in `.catch(report)`.
    function report(error) {
        say(describe(error), "error");
        try {
            console.warn("Model Chain: Voice Box: " + describe(error));
        } catch (ignored) { /* a console that will not take it is not a failure */ }
        return null;
    }

    // -- requests -------------------------------------------------------------- //

    function appRoot() {
        let base = "";
        try {
            const config = window.gradio_config;
            if (config && typeof config.root === "string") base = config.root;
        } catch (error) {
            base = "";
        }
        if (!base) {
            try {
                const path = (window.location && window.location.pathname) || "/";
                base = path.replace(/[^/]*$/, "");
            } catch (error) {
                base = "/";
            }
        }
        return base.replace(/\/+$/, "");
    }

    function url(route) {
        return appRoot() + state.prefix + route;
    }

    // One request per `kind` at a time. A second call for the same kind while
    // the first is in flight gets the first's promise back: a button pressed
    // twice waits for its answer instead of sending twice (Mini Paint's
    // 2026-09-23 lesson, eleven presses making eleven requests that never
    // reached the server). A write whose payload can differ from the one in
    // flight -- a loop toggled twice, a rename racing a prompt save -- asks
    // for `queue` instead, and goes after it rather than being folded into
    // it, so nothing the user did is silently dropped. Every request carries
    // the page token and a deadline.
    function request(kind, route, options) {
        const settings = options || {};
        if (inflight[kind]) {
            if (!settings.queue) return inflight[kind];
            return inflight[kind].then(function () { return null; }, function () { return null; })
                .then(function () { return request(kind, route, options); });
        }
        const deadline = settings.deadline || DEADLINE.json;
        const headers = {};
        headers[TOKEN_HEADER] = state.token;
        let body = settings.body;
        if (body !== undefined && body !== null && !settings.raw) {
            headers["Content-Type"] = "application/json";
            body = JSON.stringify(body);
        }
        Object.keys(settings.headers || {}).forEach(function (name) {
            headers[name] = settings.headers[name];
        });
        const init = {method: settings.method || "POST", credentials: "same-origin",
                      headers: headers};
        if (body !== undefined && body !== null) init.body = body;
        const controller = typeof AbortController === "function" ? new AbortController() : null;
        if (controller) init.signal = controller.signal;
        const clock = {timer: 0, timedOut: false};
        if (controller) {
            clock.timer = window.setTimeout(function () {
                clock.timedOut = true;
                try { controller.abort(); } catch (error) { /* already settled */ }
            }, deadline);
        }
        const finish = function () {
            if (clock.timer) window.clearTimeout(clock.timer);
            if (inflight[kind] === settled) delete inflight[kind];
        };
        const settled = Promise.resolve().then(function () {
            return fetch(url(route), init);
        }).then(function (response) {
            if (!response || !response.ok) return refusal(response);
            if (settings.expect === "blob") return response.blob();
            if (settings.expect === "text") return response.text();
            return response.json();
        }).catch(function (error) {
            if (clock.timedOut) {
                throw new RequestError("The WebUI did not answer within "
                                       + Math.round(deadline / 1000) + " s.", 0, true);
            }
            throw error;
        }).then(function (found) {
            finish();
            return found;
        }, function (error) {
            finish();
            throw error;
        });
        inflight[kind] = settled;
        return settled;
    }

    function refusal(response) {
        const status = response ? response.status : 0;
        const fallback = function () {
            throw new RequestError("The WebUI answered " + status + ".", status, false);
        };
        if (!response || typeof response.text !== "function") return fallback();
        return response.text().then(function (text) {
            let reason = "";
            try {
                reason = JSON.parse(text).error || "";
            } catch (error) {
                reason = "";
            }
            if (status === 403 && !reason) {
                reason = "This page was loaded before the WebUI restarted. Reload it.";
            }
            throw new RequestError(reason || ("The WebUI answered " + status + "."), status, false);
        }, fallback);
    }

    function listOf(reply, key) {
        if (Array.isArray(reply)) return reply;
        if (reply && Array.isArray(reply[key])) return reply[key];
        return [];
    }

    function recordOf(reply, key) {
        if (reply && reply[key] && typeof reply[key] === "object") return reply[key];
        return reply || {};
    }

    // -- audio focus ----------------------------------------------------------- //

    function claimFocus(kind) {
        state.focus.sent.push({owner: OWNER, kind: kind});
        if (state.focus.sent.length > 20) state.focus.sent.shift();
        if (typeof CustomEvent !== "function" || typeof document.dispatchEvent !== "function") {
            return false;
        }
        try {
            document.dispatchEvent(new CustomEvent(FOCUS_EVENT,
                                                   {detail: {owner: OWNER, kind: kind}}));
            return true;
        } catch (error) {
            return false;
        }
    }

    function onFocus(event) {
        const detail = event && event.detail;
        state.focus.last = detail ? {owner: detail.owner, kind: detail.kind} : null;
        if (!detail || detail.owner === OWNER) return;
        yieldFocus();
    }

    // Somebody else's speaker or microphone: every player paused, the recorder
    // stopped, a real-time capture ended. Nothing is deleted -- a paused lane
    // resumes with Play, a stopped recording lands in the trimmer as usual.
    function yieldFocus() {
        pauseAll(null);
        stopRecording();
        stopCapture("Voice Chat took the microphone; the capture was stopped.");
    }

    let listening = false;

    function listen() {
        if (listening || typeof document.addEventListener !== "function") return;
        listening = true;
        document.addEventListener(FOCUS_EVENT, onFocus);
        document.addEventListener("visibilitychange", function () {
            if (document.visibilityState === "hidden") {
                schedulePoll();
            } else {
                refreshStatus().catch(report).then(schedulePoll);
            }
        });
        if (typeof window.addEventListener === "function") {
            window.addEventListener("resize", function () { redrawAll(); });
        }
    }

    // -- players --------------------------------------------------------------- //

    function player(className) {
        const audio = document.createElement("audio");
        audio.className = className || "";
        audio.setAttribute("preload", "none");
        players.push(audio);
        return audio;
    }

    function pauseAll(except) {
        players.forEach(function (audio) {
            if (audio === except) return;
            try {
                if (!audio.paused && typeof audio.pause === "function") audio.pause();
            } catch (error) { /* an element with no source */ }
        });
    }

    function playing(audio) {
        return !!audio && audio.paused === false;
    }

    // The bytes of a sample or an output, once: fetched with the token header
    // and kept as a blob URL for the life of the page (or until the row goes).
    function audioUrl(key, route) {
        if (cache.urls[key]) return Promise.resolve(cache.urls[key]);
        return request("audio:" + key, route, {method: "GET", expect: "blob",
                                                deadline: DEADLINE.audio})
            .then(function (blob) {
                cache.blobs[key] = blob;
                cache.urls[key] = URL.createObjectURL(blob);
                return cache.urls[key];
            });
    }

    function forget(key) {
        if (cache.urls[key]) {
            try { URL.revokeObjectURL(cache.urls[key]); } catch (error) { /* ignore */ }
        }
        delete cache.urls[key];
        delete cache.blobs[key];
    }

    function startPlayback(audio, address) {
        pauseAll(audio);
        if (audio.mcVoiceBoxSource !== address) {
            audio.src = address;
            audio.mcVoiceBoxSource = address;
        }
        claimFocus("playback");
        let started;
        try {
            started = audio.play();
        } catch (error) {
            return Promise.reject(error);
        }
        return Promise.resolve(started).catch(function (error) {
            throw new Error("Playback was refused by the browser: " + describe(error));
        });
    }

    // -- waveforms ------------------------------------------------------------- //

    function peaksOf(samples, buckets) {
        const count = buckets || PEAK_BUCKETS;
        const found = new Array(count).fill(0);
        if (!samples || !samples.length) return found;
        const per = samples.length / count;
        for (let bucket = 0; bucket < count; bucket += 1) {
            const from = Math.floor(bucket * per);
            const to = Math.max(from + 1, Math.floor((bucket + 1) * per));
            let peak = 0;
            for (let index = from; index < to && index < samples.length; index += 1) {
                const value = Math.abs(samples[index]);
                if (value > peak) peak = value;
            }
            found[bucket] = Math.min(1, peak);
        }
        return found;
    }

    function inkOf(node) {
        try {
            const style = window.getComputedStyle(node);
            return (style && style.color) || "gray";
        } catch (error) {
            return "gray";
        }
    }

    function canvasOf(className) {
        const canvas = el("canvas", "mc-voice-box-wave " + className);
        canvas.setAttribute("aria-hidden", "true");
        return canvas;
    }

    // Bars from `peaks` (0..1 per bucket), a shaded selection, two handles and a
    // playhead, all in the element's own `color` so the stylesheet names none.
    function draw(canvas, peaks, options) {
        if (!canvas || typeof canvas.getContext !== "function") return;
        const ctx = canvas.getContext("2d");
        if (!ctx) return;
        const settings = options || {};
        let width = 0;
        let height = 0;
        try {
            const box = canvas.getBoundingClientRect();
            width = box.width;
            height = box.height;
        } catch (error) { /* no layout */ }
        width = Math.max(1, Math.round(width || canvas.clientWidth || 240));
        height = Math.max(1, Math.round(height || canvas.clientHeight || 48));
        const ratio = window.devicePixelRatio || 1;
        canvas.width = Math.round(width * ratio);
        canvas.height = Math.round(height * ratio);
        canvas.mcVoiceBoxWidth = width;
        if (typeof ctx.setTransform === "function") ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
        else if (typeof ctx.scale === "function") ctx.scale(ratio, ratio);
        const ink = inkOf(canvas);
        ctx.clearRect(0, 0, width, height);
        ctx.fillStyle = ink;
        ctx.strokeStyle = ink;
        if (settings.selection) {
            ctx.globalAlpha = 0.14;
            const from = Math.round(settings.selection.from * width);
            const to = Math.round(settings.selection.to * width);
            ctx.fillRect(from, 0, Math.max(1, to - from), height);
        }
        const bars = peaks && peaks.length ? peaks : new Array(PEAK_BUCKETS).fill(0.02);
        ctx.globalAlpha = settings.selection ? 0.55 : 0.8;
        const middle = height / 2;
        const step = width / bars.length;
        for (let index = 0; index < bars.length; index += 1) {
            const half = Math.max(0.5, (Math.min(1, Number(bars[index]) || 0) * (height - 4)) / 2);
            ctx.fillRect(index * step, middle - half, Math.max(0.5, step - 1), half * 2);
        }
        ctx.globalAlpha = 1;
        if (settings.selection) {
            [settings.selection.from, settings.selection.to].forEach(function (at) {
                const x = Math.round(at * width);
                ctx.fillRect(Math.max(0, x - 1), 0, 3, height);
                ctx.fillRect(Math.max(0, x - 5), 0, 11, 6);
                ctx.fillRect(Math.max(0, x - 5), height - 6, 11, 6);
            });
        }
        if (typeof settings.head === "number" && settings.head >= 0) {
            const x = Math.round(settings.head * width);
            ctx.fillRect(Math.max(0, x - 1), 0, 2, height);
        }
    }

    function redrawAll() {
        Object.keys(nodes.lanes).forEach(function (id) {
            const lane = nodes.lanes[id].mcVoiceBoxLane;
            if (lane) drawLane(lane);
        });
        if (nodes.samples) {
            const rows = nodes.samples.children || [];
            for (let index = 0; index < rows.length; index += 1) {
                const row = rows[index];
                if (row.mcVoiceBoxSample) draw(row.mcVoiceBoxSample.wave, row.mcVoiceBoxSample.peaks);
            }
        }
        drawSpeakers();
        drawTrimmer();
    }

    // -- the prompt, parsed the way the server parses it ----------------------- //

    // mc_voice_box.parse_script's grammar, so the summary counts what the
    // render will do: pause tags split the script into sections; inside one,
    // `Speaker N:` or `[N]:` starts a line and a line without a name continues
    // the line before it (Speaker 1's when nobody has spoken yet); the pauses
    // are the gaps between sections, so a tag at either end adds none.
    const PAUSE_TAG = /\[\s*pause(?:\s*[:=]\s*(\d{1,6})\s*(?:ms)?)?\s*\]/gi;
    const SPEAKER_PREFIX = /^\s*(?:speaker\s*(\d+)|\[\s*(\d+)\s*\])\s*[:：]\s*(.*)$/i;
    const MAX_SPEAKERS = 4;

    function squeeze(words) {
        return String(words || "").split(/\s+/).filter(Boolean).join(" ");
    }

    function linesOf(chunk) {
        const lines = [];
        let current = null;
        String(chunk || "").split(/\r?\n/).forEach(function (raw) {
            const match = SPEAKER_PREFIX.exec(raw);
            if (match) {
                const number = Number(match[1] || match[2]);
                if (!(number >= 1 && number <= MAX_SPEAKERS)) {
                    throw new Error("Speakers are numbered 1 to " + MAX_SPEAKERS
                                    + "; the prompt names Speaker " + number + ".");
                }
                current = number;
                lines.push({speaker: number, text: squeeze(match[3])});
                return;
            }
            const body = squeeze(raw);
            if (!body) return;
            if (current === null) {
                current = 1;
                lines.push({speaker: 1, text: body});
            } else {
                const last = lines[lines.length - 1];
                last.text = (last.text + " " + body).trim();
            }
        });
        return lines.filter(function (line) { return line.text; });
    }

    function parseScript(text) {
        const found = {sections: [], pauses: 0, speakers: {}, words: 0, error: ""};
        const words = String(text || "");
        const chunks = [];
        let cursor = 0;
        PAUSE_TAG.lastIndex = 0;
        let match = PAUSE_TAG.exec(words);
        while (match) {
            chunks.push(words.slice(cursor, match.index));
            cursor = match.index + match[0].length;
            match = PAUSE_TAG.exec(words);
        }
        chunks.push(words.slice(cursor));
        try {
            chunks.forEach(function (chunk) {
                const lines = linesOf(chunk);
                if (!lines.length) return;
                found.sections.push({lines: lines});
                lines.forEach(function (line) {
                    found.speakers[line.speaker] = true;
                    found.words += line.text.split(/\s+/).filter(Boolean).length;
                });
            });
        } catch (error) {
            found.error = describe(error);
            found.sections = [];
            found.words = 0;
            found.speakers = {};
        }
        found.pauses = Math.max(0, found.sections.length - 1);
        return found;
    }

    function count(number, singular, plural) {
        return number + " " + (number === 1 ? singular : plural);
    }

    function summarize(text) {
        const parsed = parseScript(text);
        if (parsed.error) return parsed.error;
        if (!parsed.words) return "Nothing to render yet.";
        const speakers = Object.keys(parsed.speakers).length;
        return count(speakers, "speaker", "speakers") + ", "
            + count(parsed.pauses, "pause", "pauses") + ", ~"
            + parsed.words + " words";
    }

    // -- the selection's arithmetic -------------------------------------------- //

    // `which` is the handle being moved. The other one stays, and the moved one
    // is kept inside the file, at least the minimum apart and at most the
    // maximum; a file shorter than the minimum is selected whole, because the
    // clamps to its ends leave nothing else.
    function clampSelection(selection, which, value, duration) {
        const total = Math.max(0, Number(duration) || 0);
        const shortest = SELECTION.min;
        const longest = SELECTION.max;
        let from = Math.max(0, Math.min(total, Number(selection.in) || 0));
        let to = Math.max(0, Math.min(total, Number(selection.out) || 0));
        const wanted = Math.max(0, Math.min(total, Number(value) || 0));
        if (which === "in") {
            from = Math.min(wanted, to - shortest);
            from = Math.max(from, to - longest, 0);
        } else {
            to = Math.max(wanted, from + shortest);
            to = Math.min(to, from + longest, total);
        }
        if (to - from < shortest) {
            if (which === "in") from = Math.max(0, to - shortest);
            else to = Math.min(total, from + shortest);
        }
        return {in: from, out: to};
    }

    // -- encoding: mono, 24 kHz, 16-bit WAV, built by hand ----------------------- //

    function encodeWav(samples, rate) {
        const count = samples ? samples.length : 0;
        const buffer = new ArrayBuffer(44 + count * 2);
        const view = new DataView(buffer);
        const ascii = function (at, text) {
            for (let index = 0; index < text.length; index += 1) {
                view.setUint8(at + index, text.charCodeAt(index));
            }
        };
        ascii(0, "RIFF");
        view.setUint32(4, 36 + count * 2, true);
        ascii(8, "WAVE");
        ascii(12, "fmt ");
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true);
        view.setUint16(22, 1, true);
        view.setUint32(24, rate, true);
        view.setUint32(28, rate * 2, true);
        view.setUint16(32, 2, true);
        view.setUint16(34, 16, true);
        ascii(36, "data");
        view.setUint32(40, count * 2, true);
        let at = 44;
        for (let index = 0; index < count; index += 1) {
            const value = Math.max(-1, Math.min(1, samples[index] || 0));
            view.setInt16(at, value < 0 ? Math.round(value * 32768) : Math.round(value * 32767),
                          true);
            at += 2;
        }
        return buffer;
    }

    function resampleLinear(samples, from, to) {
        if (!samples || !samples.length || !from || from === to) return samples || new Float32Array(0);
        const length = Math.max(1, Math.round(samples.length * to / from));
        const out = new Float32Array(length);
        const ratio = from / to;
        for (let index = 0; index < length; index += 1) {
            const at = index * ratio;
            const left = Math.floor(at);
            const right = Math.min(samples.length - 1, left + 1);
            const mix = at - left;
            out[index] = samples[left] * (1 - mix) + samples[right] * mix;
        }
        return out;
    }

    // An OfflineAudioContext resamples with the browser's own filter; the
    // linear fallback is for a browser without one, so the button never dies.
    function resampleTo(samples, from, to) {
        if (!samples || !samples.length || from === to) return Promise.resolve(samples);
        if (typeof OfflineAudioContext !== "function") {
            return Promise.resolve(resampleLinear(samples, from, to));
        }
        return new Promise(function (resolve) {
            try {
                const length = Math.max(1, Math.ceil(samples.length * to / from));
                const offline = new OfflineAudioContext(1, length, to);
                const buffer = offline.createBuffer(1, samples.length, from);
                buffer.getChannelData(0).set(samples);
                const source = offline.createBufferSource();
                source.buffer = buffer;
                source.connect(offline.destination);
                source.start(0);
                Promise.resolve(offline.startRendering()).then(function (rendered) {
                    resolve(rendered.getChannelData(0));
                }, function () {
                    resolve(resampleLinear(samples, from, to));
                });
            } catch (error) {
                resolve(resampleLinear(samples, from, to));
            }
        });
    }

    function mixdown(buffer, from, to) {
        const rate = buffer.sampleRate;
        const start = Math.max(0, Math.floor(from * rate));
        const end = Math.min(buffer.length, Math.floor(to * rate));
        const channels = Math.max(1, buffer.numberOfChannels || 1);
        const out = new Float32Array(Math.max(0, end - start));
        for (let channel = 0; channel < channels; channel += 1) {
            const data = buffer.getChannelData(channel);
            for (let index = 0; index < out.length; index += 1) {
                out[index] += data[start + index] / channels;
            }
        }
        return out;
    }

    // -- boot ------------------------------------------------------------------ //

    function findRoot() {
        const app = typeof gradioApp === "function" ? gradioApp() : document;
        let found = null;
        try {
            found = (app || document).querySelector("#" + ROOT_ID);
        } catch (error) {
            found = null;
        }
        if (!found && app !== document) {
            try {
                found = document.querySelector("#" + ROOT_ID);
            } catch (error) {
                found = null;
            }
        }
        return found;
    }

    function boot() {
        const root = findRoot();
        if (!root) return false;
        if (root.getAttribute("data-mc-voice-box-booted") === "1") return true;
        root.setAttribute("data-mc-voice-box-booted", "1");
        state.booted = true;
        state.token = root.getAttribute("data-mc-voice-key") || "";
        state.prefix = root.getAttribute("data-mc-voice-box-prefix") || DEFAULT_PREFIX;
        nodes.root = root;
        build(root);
        listen();
        loadAll().catch(report).then(schedulePoll);
        return true;
    }

    function loadAll() {
        say("Loading…", "info");
        return Promise.all([refreshStatus(), refreshSamples(), refreshPrompts(),
                            refreshConfigurations(), refreshPipelines()])
            .then(function () {
                let id = remembered();
                if (!pipelineById(id)) id = state.pipelines.length ? state.pipelines[0].id : "";
                if (!id) return newPipeline("Pipeline 1");
                return choosePipeline(id);
            })
            .then(function () {
                if (state.message === "Loading…") say("", "info");
            });
    }

    function remembered() {
        try {
            return window.localStorage.getItem(STORAGE_KEY) || "";
        } catch (error) {
            return "";
        }
    }

    function remember(id) {
        try {
            window.localStorage.setItem(STORAGE_KEY, id || "");
        } catch (error) { /* private mode, or storage denied: the page still works */ }
    }

    // -- building the page ----------------------------------------------------- //

    function build(root) {
        clear(root);

        const head = el("div", "mc-voice-box-head");
        nodes.title = el("h2", "mc-voice-box-title", "Voice Box");
        head.appendChild(nodes.title);
        const bar = el("div", "mc-voice-box-pipeline-bar");
        const label = el("label", "mc-voice-box-pipeline-label", "Pipeline ");
        nodes.pipelineSelect = el("select", "mc-voice-box-pipeline-select");
        nodes.pipelineSelect.setAttribute("aria-label", "Pipeline");
        nodes.pipelineSelect.addEventListener("change", function () {
            choosePipeline(nodes.pipelineSelect.value).catch(report);
        });
        label.appendChild(nodes.pipelineSelect);
        bar.appendChild(label);
        nodes.pipelineNew = button("New", "New pipeline", function () { return newPipeline(); },
                                   "mc-voice-box-pipeline-new");
        nodes.pipelineRename = button("Rename", "Rename this pipeline", renamePipeline,
                                      "mc-voice-box-pipeline-rename");
        nodes.pipelineDelete = button("Delete", "Delete this pipeline", deletePipeline,
                                      "mc-voice-box-pipeline-delete");
        bar.appendChild(nodes.pipelineNew);
        bar.appendChild(nodes.pipelineRename);
        bar.appendChild(nodes.pipelineDelete);
        head.appendChild(bar);
        root.appendChild(head);

        const stages = el("div", "mc-voice-box-stages");
        STAGES.forEach(function (stage, index) {
            if (index) {
                const connector = el("div", "mc-voice-box-connector");
                connector.setAttribute("aria-hidden", "true");
                stages.appendChild(connector);
            }
            const card = el("section", "mc-voice-box-stage mc-voice-box-stage-" + stage.key);
            card.setAttribute("data-stage", stage.key);
            card.appendChild(el("h3", "mc-voice-box-stage-title", stage.title));
            const body = el("div", "mc-voice-box-stage-body");
            card.appendChild(body);
            nodes["stage_" + stage.key] = body;
            stages.appendChild(card);
        });
        root.appendChild(stages);

        buildInput(nodes.stage_input);
        buildPrompt(nodes.stage_prompt);
        buildConfiguration(nodes.stage_configuration);
        buildOutputs(nodes.stage_outputs);

        const footer = el("div", "mc-voice-box-footer");
        const row = el("div", "mc-voice-box-footer-row");
        nodes.render = button("Render", "Render this pipeline", renderNow, "mc-voice-box-render");
        nodes.renderReason = el("span", "mc-voice-box-render-reason", "");
        nodes.install = button("Install VibeVoice", "Install the VibeVoice runtime and model",
                               installEngine, "mc-voice-box-install");
        show(nodes.install, false);
        nodes.installProgress = el("span", "mc-voice-box-install-progress", "");
        row.appendChild(nodes.render);
        row.appendChild(nodes.renderReason);
        row.appendChild(nodes.install);
        row.appendChild(nodes.installProgress);
        footer.appendChild(row);
        nodes.jobs = el("ul", "mc-voice-box-list mc-voice-box-jobs");
        footer.appendChild(nodes.jobs);
        nodes.cards = el("div", "mc-voice-box-cards", "");
        footer.appendChild(nodes.cards);
        nodes.runtimeActions = el("div", "mc-voice-box-row mc-voice-box-runtime-actions");
        footer.appendChild(nodes.runtimeActions);
        nodes.status = el("div", "mc-voice-box-status", "");
        nodes.status.setAttribute("role", "status");
        nodes.status.setAttribute("aria-live", "polite");
        footer.appendChild(nodes.status);
        root.appendChild(footer);
    }

    // -- INPUT: a file or a recording, the trimmer, the sample library ----------- //

    const trimmer = {
        source: null,
        mode: "",
        buffer: null,
        peaks: null,
        duration: 0,
        selection: {in: 0, out: 0},
        loop: false,
        dragging: null,
        mediaUrl: "",
        mediaSource: null,
        capture: null,
        context: null,
    };

    function buildInput(body) {
        const controls = el("div", "mc-voice-box-row mc-voice-box-input-controls");
        const pickLabel = el("label", "mc-voice-box-button mc-voice-box-file", "Choose a file");
        nodes.file = el("input", "mc-voice-box-file-input");
        nodes.file.setAttribute("type", "file");
        nodes.file.type = "file";
        nodes.file.setAttribute("accept", "audio/*,video/*");
        nodes.file.setAttribute("aria-label", "Choose an audio or video file");
        nodes.file.addEventListener("change", function () {
            const file = nodes.file.files && nodes.file.files[0];
            if (!file) return;
            openTrimmer({name: file.name || "file", blob: file, from: "file"}).catch(report);
            try { nodes.file.value = ""; } catch (error) { /* some browsers refuse */ }
        });
        pickLabel.appendChild(nodes.file);
        controls.appendChild(pickLabel);
        nodes.record = button("Record", "Record from the microphone", toggleRecording,
                              "mc-voice-box-record");
        pressed(nodes.record, false);
        controls.appendChild(nodes.record);
        body.appendChild(controls);

        const box = el("div", "mc-voice-box-trimmer");
        nodes.trimmer = box;
        nodes.trimmerSource = el("div", "mc-voice-box-trimmer-source", "");
        box.appendChild(nodes.trimmerSource);
        nodes.trimmerWave = canvasOf("mc-voice-box-trimmer-wave");
        wireHandles(nodes.trimmerWave);
        box.appendChild(nodes.trimmerWave);
        nodes.trimmerReadout = el("div", "mc-voice-box-trimmer-readout", "");
        box.appendChild(nodes.trimmerReadout);
        const row = el("div", "mc-voice-box-row mc-voice-box-trimmer-row");
        nodes.trimmerPlay = button("Play selection", "Play the selection", playSelection,
                                   "mc-voice-box-trimmer-play");
        nodes.trimmerLoop = button("Loop selection", "Loop the selection while it plays",
                                   toggleTrimmerLoop, "mc-voice-box-trimmer-loop");
        pressed(nodes.trimmerLoop, false);
        nodes.trimmerName = el("input", "mc-voice-box-trimmer-name");
        nodes.trimmerName.setAttribute("type", "text");
        nodes.trimmerName.setAttribute("aria-label", "Sample name");
        nodes.trimmerName.setAttribute("placeholder", "Sample name");
        nodes.trimmerName.setAttribute("maxlength", "80");
        nodes.trimmerSave = button("Save as sample", "Save the selection as a sample",
                                   saveSelection, "mc-voice-box-trimmer-save");
        nodes.trimmerDiscard = button("Discard", "Discard this file", closeTrimmer,
                                      "mc-voice-box-trimmer-discard");
        row.appendChild(nodes.trimmerPlay);
        row.appendChild(nodes.trimmerLoop);
        row.appendChild(nodes.trimmerName);
        row.appendChild(nodes.trimmerSave);
        row.appendChild(nodes.trimmerDiscard);
        box.appendChild(row);
        nodes.trimmerProgress = el("div", "mc-voice-box-progress");
        nodes.trimmerProgress.setAttribute("role", "progressbar");
        nodes.trimmerProgressBar = el("span", "mc-voice-box-progress-bar", "");
        nodes.trimmerProgress.appendChild(nodes.trimmerProgressBar);
        show(nodes.trimmerProgress, false);
        box.appendChild(nodes.trimmerProgress);
        nodes.trimmerAudio = player("mc-voice-box-trimmer-audio");
        nodes.trimmerAudio.addEventListener("timeupdate", followSelection);
        nodes.trimmerAudio.addEventListener("pause", function () { drawTrimmer(); });
        nodes.trimmerAudio.addEventListener("ended", function () { drawTrimmer(); });
        box.appendChild(nodes.trimmerAudio);
        show(box, false);
        body.appendChild(box);

        const library = el("div", "mc-voice-box-library");
        library.appendChild(el("h4", "mc-voice-box-subtitle", "Sample library"));
        nodes.samples = el("ul", "mc-voice-box-list mc-voice-box-samples");
        library.appendChild(nodes.samples);
        nodes.sampleAudio = player("mc-voice-box-sample-audio");
        nodes.sampleAudio.addEventListener("pause", function () { markSamplePlaying(""); });
        nodes.sampleAudio.addEventListener("ended", function () { markSamplePlaying(""); });
        library.appendChild(nodes.sampleAudio);
        body.appendChild(library);
    }

    function renderSamples() {
        if (!nodes.samples) return;
        clear(nodes.samples);
        if (!state.samples.length) {
            nodes.samples.appendChild(el("li", "mc-voice-box-empty",
                                         "No samples yet. Choose a file or record one, trim "
                                         + "it, and save it as a sample."));
            return;
        }
        const speakers = (state.working && state.working.speakers) || {};
        state.samples.forEach(function (sample) {
            const row = el("li", "mc-voice-box-sample");
            row.setAttribute("data-id", sample.id);
            const wave = canvasOf("mc-voice-box-sample-wave");
            row.appendChild(wave);
            const line = el("div", "mc-voice-box-row");
            const title = el("span", "mc-voice-box-sample-title", sample.title || "Untitled");
            title.setAttribute("title", "Double-click to rename");
            title.addEventListener("dblclick", function () {
                renameInline(title, sample.title || "", function (name) {
                    return request("samples/rename", ROUTES.sampleRename,
                                   {body: {id: sample.id, title: name}, queue: true})
                        .then(refreshSamples);
                });
            });
            line.appendChild(title);
            const meta = el("span", "mc-voice-box-sample-meta",
                            sample.seconds ? seconds(sample.seconds) : "");
            line.appendChild(meta);
            const play = button("Play", "Play " + (sample.title || "this sample"), function () {
                return playSample(sample);
            }, "mc-voice-box-sample-play");
            line.appendChild(play);
            const assign = el("span", "mc-voice-box-sample-speakers");
            [1, 2, 3, 4].forEach(function (number) {
                const slot = button(String(number), "Assign to speaker " + number, function () {
                    assignSpeaker(number, sample.id);
                }, "mc-voice-box-sample-speaker");
                pressed(slot, speakers[String(number)] === sample.id);
                assign.appendChild(slot);
            });
            line.appendChild(assign);
            line.appendChild(button("Delete", "Delete " + (sample.title || "this sample"),
                                    function () { return deleteSample(sample); },
                                    "mc-voice-box-sample-delete"));
            row.appendChild(line);
            row.mcVoiceBoxSample = {wave: wave, peaks: sample.peaks || [], play: play};
            nodes.samples.appendChild(row);
            draw(wave, sample.peaks || []);
        });
        markSamplePlaying(state.playingSample || "");
    }

    function markSamplePlaying(id) {
        state.playingSample = id || "";
        if (!nodes.samples) return;
        const rows = nodes.samples.children || [];
        for (let index = 0; index < rows.length; index += 1) {
            const row = rows[index];
            if (!row.mcVoiceBoxSample) continue;
            const on = !!id && row.getAttribute("data-id") === id && playing(nodes.sampleAudio);
            row.mcVoiceBoxSample.play.textContent = on ? "Pause" : "Play";
        }
    }

    function playSample(sample) {
        const audio = nodes.sampleAudio;
        if (state.playingSample === sample.id && playing(audio)) {
            audio.pause();
            return Promise.resolve();
        }
        return audioUrl("sample:" + sample.id,
                        ROUTES.sampleAudio + "?id=" + encodeURIComponent(sample.id))
            .then(function (address) {
                return startPlayback(audio, address);
            })
            .then(function () {
                markSamplePlaying(sample.id);
            });
    }

    function deleteSample(sample) {
        forget("sample:" + sample.id);
        return request("samples/delete:" + sample.id, ROUTES.sampleDelete, {body: {id: sample.id}})
            .then(function () {
                // A configuration that referred to it has had the reference dropped
                // on the server; the working copy follows.
                if (state.working) {
                    Object.keys(state.working.speakers).forEach(function (number) {
                        if (state.working.speakers[number] === sample.id) {
                            delete state.working.speakers[number];
                        }
                    });
                }
                return Promise.all([refreshSamples(), refreshConfigurations()]);
            });
    }

    function assignSpeaker(number, sampleId) {
        const working = ensureWorking();
        working.speakers[String(number)] = sampleId;
        markConfigurationDirty();
        renderSamples();
        drawSpeakers();
        renderFooter();
        say("Speaker " + number + " is " + titleOf(sampleId) + ". Save the configuration to keep it.",
            "info");
    }

    function clearSpeaker(number) {
        const working = ensureWorking();
        delete working.speakers[String(number)];
        markConfigurationDirty();
        renderSamples();
        drawSpeakers();
        renderFooter();
    }

    function titleOf(sampleId) {
        const found = state.samples.filter(function (sample) { return sample.id === sampleId; })[0];
        return found ? (found.title || "Untitled") : "a missing sample";
    }

    function sampleById(id) {
        return state.samples.filter(function (sample) { return sample.id === id; })[0] || null;
    }

    // -- the trimmer ----------------------------------------------------------- //

    function audioContext() {
        if (trimmer.context) return trimmer.context;
        const Ctor = window.AudioContext || window.webkitAudioContext;
        if (!Ctor) return null;
        try {
            trimmer.context = new Ctor();
        } catch (error) {
            trimmer.context = null;
        }
        return trimmer.context;
    }

    function decode(bytes) {
        const ctx = audioContext();
        if (!ctx || typeof ctx.decodeAudioData !== "function") {
            return Promise.reject(new Error("no decoder"));
        }
        return new Promise(function (resolve, reject) {
            let settled = false;
            const done = function (error, buffer) {
                if (settled) return;
                settled = true;
                if (error || !buffer) reject(error || new Error("decode failed"));
                else resolve(buffer);
            };
            try {
                const result = ctx.decodeAudioData(bytes, function (buffer) { done(null, buffer); },
                                                   function (error) { done(error || new Error("decode failed")); });
                if (result && typeof result.then === "function") {
                    result.then(function (buffer) { done(null, buffer); },
                                function (error) { done(error || new Error("decode failed")); });
                }
            } catch (error) {
                done(error);
            }
        });
    }

    // Opens a file, a recording or an output in the trimmer. Short files are
    // decoded whole and drawn from their samples; long ones (or ones the
    // browser will not decode) are played through the media element and, when
    // saved, captured in real time -- so a twenty-second sample takes twenty
    // seconds, and a two-hour recording never has to be decoded whole.
    function openTrimmer(source) {
        closeTrimmer();
        trimmer.source = source;
        state.trimming = {name: source.name, from: source.from};
        nodes.trimmerSource.textContent = "Decoding " + source.name + "…";
        nodes.trimmerName.value = defaultSampleName(source);
        show(nodes.trimmer, true);
        say("Decoding " + source.name + "…", "info");
        const blob = source.blob;
        let address = "";
        try {
            address = URL.createObjectURL(blob);
        } catch (error) {
            address = "";
        }
        trimmer.mediaUrl = address;
        if (address) {
            nodes.trimmerAudio.src = address;
            nodes.trimmerAudio.mcVoiceBoxSource = address;
        }
        const tooBig = blob && typeof blob.size === "number" && blob.size > DECODE_LIMIT.bytes;
        const whole = tooBig
            ? Promise.reject(new Error("too large to decode whole"))
            : Promise.resolve(blob.arrayBuffer()).then(decode);
        return whole.then(function (buffer) {
            if (buffer.duration > DECODE_LIMIT.seconds) throw new Error("too long to decode whole");
            trimmer.mode = "buffer";
            trimmer.buffer = buffer;
            trimmer.duration = buffer.duration;
            trimmer.peaks = peaksOf(mixdown(buffer, 0, buffer.duration));
            return buffer.duration;
        }, function () {
            trimmer.mode = "capture";
            trimmer.buffer = null;
            trimmer.peaks = null;
            return mediaDuration(nodes.trimmerAudio);
        }).then(function (duration) {
            if (trimmer.source !== source) return;
            trimmer.duration = Math.max(0, Number(duration) || 0);
            trimmer.selection = clampSelection({in: 0, out: trimmer.duration}, "out",
                                               trimmer.duration, trimmer.duration);
            state.trimming = {name: source.name, from: source.from, mode: trimmer.mode,
                              duration: trimmer.duration};
            nodes.trimmerSource.textContent = source.name + " · " + seconds(trimmer.duration)
                + (trimmer.mode === "capture"
                    ? " · long or undecodable: the selection is captured while it plays"
                    : "");
            say(trimmer.mode === "capture"
                ? "Drag the handles, then Save as sample: the selection is captured in real time."
                : "Drag the handles to choose the selection.", "info");
            drawTrimmer();
        });
    }

    function defaultSampleName(source) {
        const base = String(source.name || "sample").replace(/\.[a-z0-9]{1,5}$/i, "");
        return base || "sample";
    }

    function mediaDuration(audio) {
        const known = function () {
            const value = Number(audio.duration);
            return isFinite(value) && value > 0 ? value : 0;
        };
        if (known()) return Promise.resolve(known());
        return new Promise(function (resolve) {
            let settled = false;
            const finish = function () {
                if (settled) return;
                settled = true;
                resolve(known());
            };
            audio.addEventListener("loadedmetadata", finish);
            audio.addEventListener("durationchange", function () { if (known()) finish(); });
            audio.addEventListener("error", finish);
            window.setTimeout(finish, DEADLINE.json);
            try {
                if (typeof audio.load === "function") audio.load();
            } catch (error) { /* an element that refuses is reported as 0 s */ }
        });
    }

    function closeTrimmer() {
        stopCapture("");
        if (nodes.trimmerAudio) {
            try {
                if (!nodes.trimmerAudio.paused) nodes.trimmerAudio.pause();
            } catch (error) { /* no source */ }
        }
        if (trimmer.mediaUrl) {
            try { URL.revokeObjectURL(trimmer.mediaUrl); } catch (error) { /* ignore */ }
        }
        trimmer.source = null;
        trimmer.mode = "";
        trimmer.buffer = null;
        trimmer.peaks = null;
        trimmer.duration = 0;
        trimmer.selection = {in: 0, out: 0};
        trimmer.mediaUrl = "";
        trimmer.dragging = null;
        state.trimming = null;
        if (nodes.trimmer) show(nodes.trimmer, false);
    }

    function selectionFractions() {
        if (!trimmer.duration) return null;
        return {from: trimmer.selection.in / trimmer.duration,
                to: trimmer.selection.out / trimmer.duration};
    }

    function drawTrimmer() {
        if (!nodes.trimmerWave || !trimmer.source) return;
        const audio = nodes.trimmerAudio;
        const head = trimmer.duration && playing(audio)
            ? (Number(audio.currentTime) || 0) / trimmer.duration : -1;
        draw(nodes.trimmerWave, trimmer.peaks, {selection: selectionFractions(), head: head});
        if (nodes.trimmerReadout) {
            nodes.trimmerReadout.textContent = seconds(trimmer.selection.in) + " to "
                + seconds(trimmer.selection.out) + " (" + seconds(trimmer.selection.out - trimmer.selection.in)
                + " of " + seconds(trimmer.duration) + ")";
        }
        if (nodes.trimmerPlay) nodes.trimmerPlay.textContent = playing(audio) ? "Pause" : "Play selection";
    }

    // Where along a waveform a pointer is, 0..1.
    function fractionAt(canvas, event) {
        let left = 0;
        let width = canvas.mcVoiceBoxWidth || canvas.clientWidth || 240;
        try {
            const box = canvas.getBoundingClientRect();
            left = box.left || 0;
            width = box.width || width;
        } catch (error) { /* no layout */ }
        const x = (Number(event.clientX) || 0) - left;
        return Math.max(0, Math.min(1, width ? x / width : 0));
    }

    function timeAt(canvas, event) {
        return fractionAt(canvas, event) * trimmer.duration;
    }

    function wireHandles(canvas) {
        const down = function (event) {
            if (!trimmer.source || !trimmer.duration) return;
            const at = timeAt(canvas, event);
            const toIn = Math.abs(at - trimmer.selection.in);
            const toOut = Math.abs(at - trimmer.selection.out);
            trimmer.dragging = toIn <= toOut ? "in" : "out";
            trimmer.selection = clampSelection(trimmer.selection, trimmer.dragging, at, trimmer.duration);
            try {
                if (typeof canvas.setPointerCapture === "function" && event.pointerId !== undefined) {
                    canvas.setPointerCapture(event.pointerId);
                }
            } catch (error) { /* not every pointer can be captured */ }
            if (typeof event.preventDefault === "function") event.preventDefault();
            drawTrimmer();
        };
        const move = function (event) {
            if (!trimmer.dragging || !trimmer.duration) return;
            trimmer.selection = clampSelection(trimmer.selection, trimmer.dragging,
                                               timeAt(canvas, event), trimmer.duration);
            drawTrimmer();
        };
        const up = function () {
            trimmer.dragging = null;
        };
        canvas.addEventListener("pointerdown", down);
        canvas.addEventListener("pointermove", move);
        canvas.addEventListener("pointerup", up);
        canvas.addEventListener("pointercancel", up);
    }

    function playSelection() {
        const audio = nodes.trimmerAudio;
        if (!trimmer.source || !trimmer.mediaUrl) return Promise.resolve();
        if (playing(audio)) {
            audio.pause();
            drawTrimmer();
            return Promise.resolve();
        }
        try { audio.currentTime = trimmer.selection.in; } catch (error) { /* not seekable yet */ }
        return startPlayback(audio, trimmer.mediaUrl).then(drawTrimmer);
    }

    function toggleTrimmerLoop() {
        trimmer.loop = !trimmer.loop;
        pressed(nodes.trimmerLoop, trimmer.loop);
    }

    // Keeps playback inside the selection: back to the in point when looping,
    // paused at the out point otherwise. `timeupdate` fires a few times a
    // second, which is close enough for auditioning a trim.
    function followSelection() {
        const audio = nodes.trimmerAudio;
        if (!trimmer.source) return;
        const at = Number(audio.currentTime) || 0;
        if (at >= trimmer.selection.out - 0.02) {
            if (trimmer.capture) {
                finishCapture();
            } else if (trimmer.loop) {
                try { audio.currentTime = trimmer.selection.in; } catch (error) { /* ignore */ }
            } else {
                audio.pause();
            }
        }
        if (trimmer.capture) {
            const span = Math.max(0.001, trimmer.selection.out - trimmer.selection.in);
            const fraction = Math.max(0, Math.min(1, (at - trimmer.selection.in) / span));
            nodes.trimmerProgressBar.style.width = Math.round(fraction * 100) + "%";
            nodes.trimmerProgress.setAttribute("aria-valuenow", String(Math.round(fraction * 100)));
        }
        drawTrimmer();
    }

    function saveSelection() {
        if (!trimmer.source) return Promise.resolve();
        const title = (nodes.trimmerName.value || "").trim() || defaultSampleName(trimmer.source);
        const source = trimmer.source.from || "file";
        if (trimmer.mode === "buffer" && trimmer.buffer) {
            say("Encoding the selection…", "info");
            const samples = mixdown(trimmer.buffer, trimmer.selection.in, trimmer.selection.out);
            return resampleTo(samples, trimmer.buffer.sampleRate, SAMPLE_RATE).then(function (mono) {
                return uploadSample(encodeWav(mono, SAMPLE_RATE), title, source);
            });
        }
        return captureSelection().then(function (found) {
            if (!found) return null;
            return resampleTo(found.samples, found.rate, SAMPLE_RATE).then(function (mono) {
                return uploadSample(encodeWav(mono, SAMPLE_RATE), title, source);
            });
        });
    }

    function uploadSample(bytes, title, source) {
        say("Uploading " + title + "…", "info");
        const headers = {"Content-Type": "audio/wav",
                         "x-mc-title": encodeURIComponent(title),
                         "x-mc-source": source};
        return request("samples/upload", ROUTES.sampleUpload,
                       {raw: true, body: bytes, headers: headers, deadline: DEADLINE.audio,
                        queue: true})
            .then(function () {
                say("Saved sample " + title + ".", "info");
                return refreshSamples();
            });
    }

    // -- the real-time capture for files decodeAudioData will not take whole ----- //

    const TAP_WORKLET = "class McVoiceBoxTap extends AudioWorkletProcessor {"
        + " process(inputs) { const input = inputs[0]; if (input && input[0]) {"
        + " const mono = new Float32Array(input[0].length);"
        + " for (let c = 0; c < input.length; c += 1) { const data = input[c];"
        + " for (let i = 0; i < data.length; i += 1) mono[i] += data[i] / input.length; }"
        + " this.port.postMessage(mono, [mono.buffer]); } return true; } }"
        + " registerProcessor('mc-voice-box-tap', McVoiceBoxTap);";

    function tapNode(ctx) {
        const processor = function () {
            const node = ctx.createScriptProcessor(4096, 2, 1);
            return {node: node, kind: "script"};
        };
        if (!ctx.audioWorklet || typeof ctx.audioWorklet.addModule !== "function"
            || typeof Blob === "undefined" || !window.URL || !window.URL.createObjectURL) {
            return Promise.resolve(processor());
        }
        let address = "";
        try {
            address = window.URL.createObjectURL(new Blob([TAP_WORKLET], {type: "text/javascript"}));
        } catch (error) {
            return Promise.resolve(processor());
        }
        return Promise.resolve(ctx.audioWorklet.addModule(address)).then(function () {
            try { window.URL.revokeObjectURL(address); } catch (error) { /* ignore */ }
            return {node: new window.AudioWorkletNode(ctx, "mc-voice-box-tap"), kind: "worklet"};
        }, function () {
            return processor();
        });
    }

    function captureSelection() {
        const audio = nodes.trimmerAudio;
        const ctx = audioContext();
        if (!ctx || typeof ctx.createMediaElementSource !== "function") {
            return Promise.reject(new Error("This browser cannot capture the selection."));
        }
        if (trimmer.capture) return trimmer.capture.promise;
        // A media element can be attached to a context once, ever; after that
        // its sound reaches the speakers only through the graph.
        if (!trimmer.mediaSource) {
            try {
                trimmer.mediaSource = ctx.createMediaElementSource(audio);
                trimmer.mediaSource.connect(ctx.destination);
            } catch (error) {
                return Promise.reject(new Error("The selection could not be captured: " + describe(error)));
            }
        }
        const source = trimmer.source;
        return tapNode(ctx).then(function (tap) {
            return new Promise(function (resolve, reject) {
                const record = {chunks: [], rate: ctx.sampleRate || 48000, tap: tap.node,
                                kind: tap.kind, resolve: resolve, reject: reject, promise: null,
                                silence: null};
                if (tap.kind === "worklet") {
                    tap.node.port.onmessage = function (event) { record.chunks.push(event.data); };
                } else {
                    tap.node.onaudioprocess = function (event) {
                        const input = event.inputBuffer;
                        const mono = new Float32Array(input.length);
                        const channels = Math.max(1, input.numberOfChannels || 1);
                        for (let channel = 0; channel < channels; channel += 1) {
                            const data = input.getChannelData(channel);
                            for (let index = 0; index < mono.length; index += 1) {
                                mono[index] += data[index] / channels;
                            }
                        }
                        record.chunks.push(mono);
                    };
                }
                record.silence = ctx.createGain();
                record.silence.gain.value = 0;
                trimmer.mediaSource.connect(tap.node);
                tap.node.connect(record.silence);
                record.silence.connect(ctx.destination);
                trimmer.capture = record;
                state.capturing = true;
                show(nodes.trimmerProgress, true);
                nodes.trimmerProgressBar.style.width = "0%";
                say("Capturing the selection in real time…", "info");
                try { audio.currentTime = trimmer.selection.in; } catch (error) { /* ignore */ }
                const wasLooping = trimmer.loop;
                record.wasLooping = wasLooping;
                trimmer.loop = false;
                startPlayback(audio, trimmer.mediaUrl).catch(function (error) {
                    if (trimmer.capture === record && trimmer.source === source) {
                        stopCapture("");
                        reject(error);
                    }
                });
            });
        }).then(function (found) {
            return found;
        });
    }

    function finishCapture() {
        const record = trimmer.capture;
        if (!record) return;
        const audio = nodes.trimmerAudio;
        try { audio.pause(); } catch (error) { /* ignore */ }
        detachCapture(record);
        let length = 0;
        record.chunks.forEach(function (chunk) { length += chunk.length; });
        const wanted = Math.round((trimmer.selection.out - trimmer.selection.in) * record.rate);
        const samples = new Float32Array(Math.min(length, Math.max(0, wanted || length)));
        let at = 0;
        record.chunks.forEach(function (chunk) {
            if (at >= samples.length) return;
            const take = Math.min(chunk.length, samples.length - at);
            samples.set(take === chunk.length ? chunk : chunk.subarray(0, take), at);
            at += take;
        });
        record.resolve(samples.length ? {samples: samples, rate: record.rate} : null);
    }

    function detachCapture(record) {
        [record.tap, record.silence].forEach(function (node) {
            if (!node) return;
            try { node.disconnect(); } catch (error) { /* ignore */ }
            if (node.port) node.port.onmessage = null;
            if (node.onaudioprocess !== undefined) node.onaudioprocess = null;
        });
        try {
            if (trimmer.mediaSource && record.tap) trimmer.mediaSource.disconnect(record.tap);
        } catch (error) { /* ignore */ }
        trimmer.loop = !!record.wasLooping;
        trimmer.capture = null;
        state.capturing = false;
        show(nodes.trimmerProgress, false);
    }

    function stopCapture(why) {
        const record = trimmer.capture;
        if (!record) return;
        try { nodes.trimmerAudio.pause(); } catch (error) { /* ignore */ }
        detachCapture(record);
        record.resolve(null);
        if (why) say(why, "warn");
    }

    // -- recording from the microphone ----------------------------------------- //

    let recorder = null;

    function toggleRecording() {
        if (recorder) {
            stopRecording();
            return Promise.resolve();
        }
        const media = navigator && navigator.mediaDevices;
        if (!media || typeof media.getUserMedia !== "function" || typeof MediaRecorder !== "function") {
            say("This browser cannot record here.", "warn");
            return Promise.resolve();
        }
        return media.getUserMedia({audio: true}).then(function (stream) {
            claimFocus("capture");
            let taker;
            try {
                taker = new MediaRecorder(stream);
            } catch (error) {
                stopTracks(stream);
                throw new Error("Recording could not start: " + describe(error));
            }
            const chunks = [];
            const record = {taker: taker, stream: stream, chunks: chunks};
            taker.ondataavailable = function (event) {
                if (event && event.data) chunks.push(event.data);
            };
            taker.onstop = function () {
                stopTracks(stream);
                if (recorder === record) recorder = null;
                state.recording = false;
                nodes.record.textContent = "Record";
                pressed(nodes.record, false);
                const blob = new Blob(chunks, {type: taker.mimeType || "audio/webm"});
                if (!blob.size) {
                    say("Nothing was recorded.", "warn");
                    return;
                }
                openTrimmer({name: "Recording", blob: blob, from: "microphone"}).catch(report);
            };
            taker.start();
            recorder = record;
            state.recording = true;
            nodes.record.textContent = "Stop";
            pressed(nodes.record, true);
            say("Recording… press Stop when you are done.", "info");
        }).catch(function (error) {
            say("The microphone could not be opened: " + describe(error), "warn");
        });
    }

    function stopRecording() {
        const record = recorder;
        if (!record) return;
        recorder = null;
        state.recording = false;
        nodes.record.textContent = "Record";
        pressed(nodes.record, false);
        try {
            if (record.taker.state !== "inactive") record.taker.stop();
            else stopTracks(record.stream);
        } catch (error) {
            stopTracks(record.stream);
        }
    }

    function stopTracks(stream) {
        try {
            (stream.getTracks() || []).forEach(function (track) { track.stop(); });
        } catch (error) { /* ignore */ }
    }

    // -- PROMPT ---------------------------------------------------------------- //

    function buildPrompt(body) {
        nodes.prompt = el("textarea", "mc-voice-box-prompt");
        nodes.prompt.setAttribute("aria-label", "Script");
        nodes.prompt.setAttribute("placeholder", "Speaker 1: Welcome back to the show.\nSpeaker 2: Thanks for having me.\n[pause]\nSpeaker 1: Today…");
        nodes.prompt.setAttribute("rows", "10");
        nodes.prompt.setAttribute("maxlength", "20000");
        nodes.prompt.addEventListener("input", promptChanged);
        nodes.prompt.addEventListener("change", promptChanged);
        body.appendChild(nodes.prompt);
        body.appendChild(el("div", "mc-voice-box-prompt-help", PROMPT_HELP));
        nodes.summary = el("div", "mc-voice-box-prompt-summary", summarize(""));
        body.appendChild(nodes.summary);
        const lists = el("div", "mc-voice-box-prompt-lists");
        const history = el("div", "mc-voice-box-history");
        history.appendChild(el("h4", "mc-voice-box-subtitle", "History"));
        nodes.history = el("ul", "mc-voice-box-list mc-voice-box-history-list");
        history.appendChild(nodes.history);
        const favourites = el("div", "mc-voice-box-favourites");
        favourites.appendChild(el("h4", "mc-voice-box-subtitle", "Favourites"));
        nodes.favourites = el("ul", "mc-voice-box-list mc-voice-box-favourites-list");
        favourites.appendChild(nodes.favourites);
        lists.appendChild(history);
        lists.appendChild(favourites);
        body.appendChild(lists);
    }

    function promptChanged() {
        state.prompt = nodes.prompt.value || "";
        nodes.summary.textContent = summarize(state.prompt);
        state.dirty.prompt = true;
        if (timers.prompt) window.clearTimeout(timers.prompt);
        timers.prompt = window.setTimeout(function () {
            timers.prompt = 0;
            savePrompt().catch(report);
        }, SAVE_DEBOUNCE_MS);
        renderFooter();
    }

    function savePrompt() {
        if (!state.dirty.prompt || !state.pipelineId) return Promise.resolve();
        const text = state.prompt;
        return request("pipelines/save", ROUTES.pipelineSave,
                       {body: {id: state.pipelineId, prompt: text}, queue: true})
            .then(function () {
                if (state.prompt === text) state.dirty.prompt = false;
                const pipeline = pipelineById(state.pipelineId);
                if (pipeline) pipeline.prompt = text;
            });
    }

    function renderPrompt() {
        if (!nodes.prompt) return;
        if (nodes.prompt.value !== state.prompt && !focused(nodes.prompt)) {
            nodes.prompt.value = state.prompt;
        }
        nodes.summary.textContent = summarize(state.prompt);
    }

    function promptEntry(entry, favourite) {
        const row = el("li", "mc-voice-box-prompt-entry");
        const text = String(entry.text || "");
        const use = button(text.length > 90 ? text.slice(0, 87) + "…" : text,
                           "Use this prompt", function () {
                               state.prompt = text;
                               nodes.prompt.value = text;
                               promptChanged();
                           }, "mc-voice-box-prompt-text");
        use.setAttribute("title", text);
        row.appendChild(use);
        const on = favourite || !!entry.favourite;
        const star = button(on ? "★" : "☆",
                            on ? "Remove from favourites" : "Add to favourites", function () {
                                return request("prompts/favourite", ROUTES.promptFavourite,
                                               {body: {id: entry.id, on: !on, favourite: !on},
                                                queue: true})
                                    .then(refreshPrompts);
                            }, "mc-voice-box-prompt-star");
        pressed(star, on);
        row.appendChild(star);
        row.appendChild(button("×", "Forget this prompt", function () {
            return request("prompts/delete:" + entry.id, ROUTES.promptDelete, {body: {id: entry.id}})
                .then(refreshPrompts);
        }, "mc-voice-box-prompt-delete"));
        return row;
    }

    function renderPromptLists() {
        if (!nodes.history) return;
        clear(nodes.history);
        clear(nodes.favourites);
        const history = state.prompts.history || [];
        const favourites = state.prompts.favourites || [];
        if (!history.length) nodes.history.appendChild(el("li", "mc-voice-box-empty", "No prompts yet."));
        history.forEach(function (entry) { nodes.history.appendChild(promptEntry(entry, false)); });
        if (!favourites.length) nodes.favourites.appendChild(el("li", "mc-voice-box-empty", "Star a prompt to keep it here."));
        favourites.forEach(function (entry) { nodes.favourites.appendChild(promptEntry(entry, true)); });
    }

    // -- CONFIGURATION --------------------------------------------------------- //

    const FIELDS = [
        {key: "model_id", label: "Model", kind: "select"},
        {key: "card_uuid", label: "Card", kind: "select"},
        {key: "steps", label: "Diffusion steps", kind: "number", min: 1, max: 50, step: 1},
        {key: "cfg_scale", label: "CFG", kind: "number", min: 1, max: 3, step: 0.1},
        {key: "seed", label: "Seed", kind: "text", placeholder: "random"},
        {key: "max_new_tokens", label: "Max new tokens", kind: "text", placeholder: "automatic"},
    ];

    function buildConfiguration(body) {
        const bar = el("div", "mc-voice-box-row mc-voice-box-configuration-bar");
        nodes.configurationSelect = el("select", "mc-voice-box-configuration-select");
        nodes.configurationSelect.setAttribute("aria-label", "Configuration");
        nodes.configurationSelect.addEventListener("change", function () {
            chooseConfiguration(nodes.configurationSelect.value).catch(report);
        });
        bar.appendChild(nodes.configurationSelect);
        nodes.configurationSave = button("Save", "Save this configuration", function () {
            return saveConfiguration(false);
        }, "mc-voice-box-configuration-save");
        nodes.configurationSaveAs = button("Save as", "Save as a new configuration", function () {
            return saveConfiguration(true);
        }, "mc-voice-box-configuration-save-as");
        nodes.configurationDelete = button("Delete", "Delete this configuration",
                                           deleteConfiguration, "mc-voice-box-configuration-delete");
        bar.appendChild(nodes.configurationSave);
        bar.appendChild(nodes.configurationSaveAs);
        bar.appendChild(nodes.configurationDelete);
        body.appendChild(bar);

        const fields = el("div", "mc-voice-box-fields");
        nodes.fields = {};
        FIELDS.forEach(function (field) {
            const wrap = el("label", "mc-voice-box-field" + (field.kind === "checkbox" ? " mc-voice-box-field-check" : ""));
            const caption = el("span", "mc-voice-box-field-label", field.label);
            let input;
            if (field.kind === "select") {
                input = el("select", "mc-voice-box-field-input");
            } else {
                input = el("input", "mc-voice-box-field-input");
                input.setAttribute("type", field.kind);
                input.type = field.kind;
                if (field.min !== undefined) input.setAttribute("min", String(field.min));
                if (field.max !== undefined) input.setAttribute("max", String(field.max));
                if (field.step !== undefined) input.setAttribute("step", String(field.step));
                if (field.placeholder) input.setAttribute("placeholder", field.placeholder);
            }
            input.setAttribute("aria-label", field.label);
            input.setAttribute("data-field", field.key);
            const changed = function () { fieldChanged(field, input); };
            input.addEventListener("change", changed);
            if (field.kind === "text" || field.kind === "number") input.addEventListener("input", changed);
            if (field.kind === "checkbox") {
                wrap.appendChild(input);
                wrap.appendChild(caption);
            } else {
                wrap.appendChild(caption);
                wrap.appendChild(input);
            }
            nodes.fields[field.key] = input;
            fields.appendChild(wrap);
        });
        body.appendChild(fields);

        // Keep warm is Voice Box's setting rather than a configuration's field
        // (the server keeps it beside the save folder), so it saves at once.
        const warm = el("label", "mc-voice-box-field mc-voice-box-field-check mc-voice-box-keep-warm");
        nodes.keepWarm = el("input", "mc-voice-box-field-input");
        nodes.keepWarm.setAttribute("type", "checkbox");
        nodes.keepWarm.type = "checkbox";
        nodes.keepWarm.setAttribute("aria-label", "Keep VibeVoice warm between renders");
        nodes.keepWarm.addEventListener("change", function () {
            saveSettings({keep_warm: !!nodes.keepWarm.checked}).catch(report);
        });
        warm.appendChild(nodes.keepWarm);
        warm.appendChild(el("span", "mc-voice-box-field-label", "Keep VibeVoice warm between renders"));
        body.appendChild(warm);

        const speakers = el("div", "mc-voice-box-speakers");
        nodes.speakers = {};
        [1, 2, 3, 4].forEach(function (number) {
            const slot = el("div", "mc-voice-box-speaker");
            slot.setAttribute("data-speaker", String(number));
            slot.appendChild(el("span", "mc-voice-box-speaker-label", "Speaker " + number));
            const wave = canvasOf("mc-voice-box-speaker-wave");
            slot.appendChild(wave);
            const title = el("span", "mc-voice-box-speaker-title", "no sample");
            slot.appendChild(title);
            const clearButton = button("×", "Clear speaker " + number, function () {
                clearSpeaker(number);
            }, "mc-voice-box-speaker-clear");
            slot.appendChild(clearButton);
            nodes.speakers[number] = {wave: wave, title: title, clear: clearButton};
            speakers.appendChild(slot);
        });
        body.appendChild(speakers);
    }

    function fieldChanged(field, input) {
        const working = ensureWorking();
        let value;
        if (field.kind === "checkbox") {
            value = !!input.checked;
        } else if (field.kind === "number") {
            value = input.value === "" ? null : Number(input.value);
            if (value !== null && !isFinite(value)) value = null;
        } else if (field.key === "seed" || field.key === "max_new_tokens") {
            const text = String(input.value || "").trim();
            value = text === "" ? null : Number(text);
            if (value !== null && !isFinite(value)) value = null;
        } else {
            value = input.value;
        }
        if (working[field.key] === value) return;
        working[field.key] = value;
        markConfigurationDirty();
        // The card and the model chosen last become Voice Box's defaults as
        // well, so the next configuration -- and a render with none -- starts
        // from them (design intent section 9: the chosen card is stored).
        if (field.key === "card_uuid" || field.key === "model_id") {
            const change = {};
            change[field.key] = value || "";
            saveSettings(change).catch(report);
        }
    }

    function saveSettings(values) {
        return request("settings", ROUTES.settings, {body: values, queue: true}).then(function (reply) {
            if (!state.status) state.status = {};
            if (reply && reply.settings) state.status.settings = reply.settings;
            if (reply && reply.engine_settings) state.status.engine_settings = reply.engine_settings;
            renderConfigurationFields();
        });
    }

    function engineModels() {
        const engine = (state.status && state.status.engine) || {};
        const found = [];
        (engine.models || []).forEach(function (model) {
            if (model && model.id) found.push({id: model.id, label: model.label || model.id});
        });
        const chosen = engine.model_id || "vibevoice-7b";
        if (!found.some(function (model) { return model.id === chosen; })) {
            found.unshift({id: chosen, label: engine.model_label || engine.label || chosen});
        }
        return found;
    }

    function cardLabel(card) {
        const roles = [];
        if (card.image_card) roles.push("image model's card");
        if (card.wangp_card) roles.push("WanGP's card");
        return (card.name || card.uuid || "card") + (roles.length ? " — " + roles.join(", ") : "");
    }

    function fillSelect(select, options, value) {
        clear(select);
        options.forEach(function (option) {
            const node = el("option", "", option.label);
            node.value = option.value;
            node.setAttribute("value", option.value);
            select.appendChild(node);
        });
        select.value = value;
    }

    // A new configuration starts from Voice Box's settings (the card and model
    // chosen last) and the engine's (its steps, CFG, seed and token cap).
    function defaultConfiguration() {
        const status = state.status || {};
        const settings = status.settings || {};
        const engineSettings = status.engine_settings || {};
        const cards = status.cards || [];
        return {
            id: "",
            name: "",
            model_id: settings.model_id || engineSettings.model_id || engineModels()[0].id,
            card_uuid: settings.card_uuid || engineSettings.card_uuid
                || (cards[0] ? cards[0].uuid : ""),
            steps: pick(engineSettings.steps, 10),
            cfg_scale: pick(engineSettings.cfg_scale, 1.3),
            seed: pick(engineSettings.seed, null),
            max_new_tokens: pick(engineSettings.max_new_tokens, null),
            speakers: {},
        };
    }

    function copyConfiguration(found) {
        const base = defaultConfiguration();
        if (!found) return base;
        Object.keys(base).forEach(function (key) {
            if (found[key] !== undefined) base[key] = found[key];
        });
        base.speakers = Object.assign({}, found.speakers || {});
        return base;
    }

    function ensureWorking() {
        if (!state.working) state.working = copyConfiguration(configurationById(state.configurationId));
        return state.working;
    }

    function configurationById(id) {
        return state.configurations.filter(function (found) { return found.id === id; })[0] || null;
    }

    function loadWorking() {
        state.working = copyConfiguration(configurationById(state.configurationId));
        state.dirty.configuration = false;
    }

    function markConfigurationDirty() {
        state.dirty.configuration = true;
        renderConfigurationBar();
        renderFooter();
    }

    function chooseConfiguration(id) {
        state.configurationId = configurationById(id) ? id : "";
        loadWorking();
        renderConfiguration();
        renderSamples();
        renderFooter();
        const pipeline = pipelineById(state.pipelineId);
        if (!pipeline || pipeline.configuration_id === state.configurationId) return Promise.resolve();
        return request("pipelines/save", ROUTES.pipelineSave,
                       {body: {id: state.pipelineId, configuration_id: state.configurationId},
                        queue: true})
            .then(function () { pipeline.configuration_id = state.configurationId; });
    }

    // Save writes the working copy over the selected configuration; Save as (or
    // Save with nothing selected) makes a new one. Render does not save: an
    // unsaved copy is sent inline with the render, so the screen and the render
    // agree without a configuration appearing that nobody named.
    function saveConfiguration(asNew) {
        const working = ensureWorking();
        const existing = asNew ? null : configurationById(state.configurationId);
        let name = existing ? existing.name : "";
        if (!existing) {
            name = ask("Name this configuration",
                       "Configuration " + (state.configurations.length + 1));
            if (name === null) return Promise.resolve(null);
        }
        const body = Object.assign({}, working, {name: name});
        if (existing) body.id = existing.id; else delete body.id;
        return request("configurations/save", ROUTES.configurationSave, {body: body, queue: true})
            .then(function (reply) {
                const saved = recordOf(reply, "configuration");
                if (saved.id) state.configurationId = saved.id;
                state.dirty.configuration = false;
                if (reply && Array.isArray(reply.configurations)) {
                    state.configurations = reply.configurations;
                    return null;
                }
                return refreshConfigurations();
            })
            .then(function () {
                loadWorking();
                renderConfiguration();
                renderSamples();
                renderFooter();
                const pipeline = pipelineById(state.pipelineId);
                if (pipeline && pipeline.configuration_id !== state.configurationId) {
                    return request("pipelines/save", ROUTES.pipelineSave,
                                   {body: {id: state.pipelineId, configuration_id: state.configurationId},
                                    queue: true})
                        .then(function () { pipeline.configuration_id = state.configurationId; });
                }
                return null;
            })
            .then(function () {
                say("Configuration saved.", "info");
            });
    }

    function deleteConfiguration() {
        const found = configurationById(state.configurationId);
        if (!found) {
            state.working = null;
            loadWorking();
            renderConfiguration();
            return Promise.resolve();
        }
        return request("configurations/delete", ROUTES.configurationDelete, {body: {id: found.id}})
            .then(function () {
                state.configurationId = "";
                return refreshConfigurations();
            })
            .then(function () {
                state.configurationId = state.configurations[0] ? state.configurations[0].id : "";
                loadWorking();
                renderConfiguration();
                renderSamples();
                renderFooter();
            });
    }

    function renderConfigurationBar() {
        if (!nodes.configurationSelect) return;
        const options = state.configurations.map(function (found) {
            return {value: found.id, label: found.name || "Untitled"};
        });
        options.unshift({value: "", label: state.configurations.length ? "(unsaved)" : "(no configuration yet)"});
        fillSelect(nodes.configurationSelect, options, state.configurationId || "");
        nodes.configurationSave.textContent = state.dirty.configuration ? "Save •" : "Save";
        nodes.configurationSave.setAttribute("aria-label", state.dirty.configuration
            ? "Save this configuration (unsaved changes)" : "Save this configuration");
        nodes.configurationDelete.disabled = !state.configurationId;
    }

    function renderConfigurationFields() {
        if (!nodes.fields) return;
        const working = ensureWorking();
        const cards = (state.status && state.status.cards) || [];
        FIELDS.forEach(function (field) {
            const input = nodes.fields[field.key];
            if (field.key === "model_id") {
                fillSelect(input, engineModels().map(function (model) {
                    return {value: model.id, label: model.label};
                }), working.model_id || "");
                return;
            }
            if (field.key === "card_uuid") {
                const options = cards.map(function (card) {
                    return {value: card.uuid || "", label: cardLabel(card)};
                });
                if (!options.some(function (option) { return option.value === (working.card_uuid || ""); })) {
                    options.unshift({value: working.card_uuid || "",
                                     label: working.card_uuid ? working.card_uuid : "(no card reported yet)"});
                }
                fillSelect(input, options, working.card_uuid || "");
                return;
            }
            if (focused(input)) return;
            const value = working[field.key];
            const text = value === null || value === undefined ? "" : String(value);
            if (input.value !== text) input.value = text;
        });
        if (nodes.keepWarm && !focused(nodes.keepWarm)) {
            const settings = (state.status && state.status.settings) || {};
            nodes.keepWarm.checked = settings.keep_warm !== false;
        }
    }

    function drawSpeakers() {
        if (!nodes.speakers) return;
        const working = ensureWorking();
        [1, 2, 3, 4].forEach(function (number) {
            const slot = nodes.speakers[number];
            const sample = sampleById(working.speakers[String(number)]);
            slot.title.textContent = sample ? (sample.title || "Untitled") : "no sample";
            slot.clear.disabled = !sample;
            draw(slot.wave, sample ? (sample.peaks || []) : []);
        });
    }

    function renderConfiguration() {
        renderConfigurationBar();
        renderConfigurationFields();
        drawSpeakers();
    }

    // -- OUTPUTS --------------------------------------------------------------- //

    function buildOutputs(body) {
        nodes.lanesList = el("ul", "mc-voice-box-list mc-voice-box-lanes");
        body.appendChild(nodes.lanesList);
        nodes.lanesEmpty = el("div", "mc-voice-box-empty", "No renders in this pipeline yet.");
        body.appendChild(nodes.lanesEmpty);
    }

    function outputById(id) {
        return state.outputs.filter(function (output) { return output.id === id; })[0] || null;
    }

    function metadata(output) {
        const render = output.render || {};
        const parts = [];
        if (render.model_id) parts.push(render.model_id);
        if (render.seed !== undefined && render.seed !== null) parts.push("seed " + render.seed);
        if (render.steps) parts.push(render.steps + " steps");
        if (render.cfg_scale) parts.push("CFG " + render.cfg_scale);
        if (Array.isArray(render.speakers) && render.speakers.length) {
            parts.push(render.speakers.map(function (speaker) {
                return "S" + speaker.n + " " + (speaker.title || speaker.sample_id || "?");
            }).join(", "));
        }
        if (output.seconds) parts.push(seconds(output.seconds));
        if (render.card) parts.push(render.card);
        return parts.join(" · ");
    }

    function drawLane(lane) {
        const audio = lane.audio;
        const total = Number(audio.duration) || Number(lane.output.seconds) || 0;
        const head = total && (playing(audio) || Number(audio.currentTime) > 0)
            ? Math.min(1, (Number(audio.currentTime) || 0) / total) : -1;
        draw(lane.wave, lane.output.peaks || [], {head: head});
        lane.play.textContent = playing(audio) ? "Pause" : "Play";
        lane.play.setAttribute("aria-label", (playing(audio) ? "Pause " : "Play ") + (lane.output.name || "this render"));
    }

    function laneNode(output) {
        const row = el("li", "mc-voice-box-lane");
        row.setAttribute("data-id", output.id);
        const head = el("div", "mc-voice-box-row mc-voice-box-lane-head");
        const name = el("span", "mc-voice-box-lane-name", output.name || "Untitled");
        name.setAttribute("title", "Double-click to rename");
        name.addEventListener("dblclick", function () {
            renameInline(name, output.name || "", function (text) {
                return request("outputs/rename", ROUTES.outputRename,
                               {body: {id: output.id, name: text}, queue: true})
                    .then(refreshOutputs);
            });
        });
        head.appendChild(name);
        const meta = el("span", "mc-voice-box-lane-meta", metadata(output));
        head.appendChild(meta);
        row.appendChild(head);
        const wave = canvasOf("mc-voice-box-lane-wave");
        row.appendChild(wave);
        const audio = player("mc-voice-box-lane-audio");
        audio.loop = !!output.loop;
        row.appendChild(audio);
        // A press on the waveform seeks: the lane's length is the element's once
        // it has loaded, the record's before that.
        wave.addEventListener("pointerdown", function (event) {
            const total = Number(audio.duration) || Number(output.seconds) || 0;
            if (!total) return;
            try { audio.currentTime = fractionAt(wave, event) * total; } catch (error) { /* not seekable */ }
            drawLane(lane);
        });
        const actions = el("div", "mc-voice-box-row mc-voice-box-lane-actions");
        const play = button("Play", "Play " + (output.name || "this render"), function () {
            return toggleLane(lane);
        }, "mc-voice-box-lane-play");
        const loop = button("Loop", "Loop " + (output.name || "this render"), function () {
            return toggleLoop(lane);
        }, "mc-voice-box-lane-loop");
        pressed(loop, !!output.loop);
        const trim = button("Trim to sample", "Open " + (output.name || "this render") + " in the trimmer",
                            function () { return trimOutput(lane); }, "mc-voice-box-lane-trim");
        const save = button("Save", "Save " + (output.name || "this render") + " to the folder",
                            function () { return saveOutput(lane); }, "mc-voice-box-lane-save");
        const download = button("Download", "Download " + (output.name || "this render"),
                                function () { return downloadOutput(lane); }, "mc-voice-box-lane-download");
        const remove = button("Delete", "Delete " + (output.name || "this render"),
                              function () { return deleteOutput(lane); }, "mc-voice-box-lane-delete");
        [play, loop, trim, save, download, remove].forEach(function (node) { actions.appendChild(node); });
        row.appendChild(actions);
        const lane = {output: output, row: row, name: name, meta: meta, wave: wave, audio: audio,
                      play: play, loop: loop};
        row.mcVoiceBoxLane = lane;
        audio.addEventListener("timeupdate", function () { drawLane(lane); });
        audio.addEventListener("play", function () { drawLane(lane); });
        audio.addEventListener("pause", function () { drawLane(lane); });
        audio.addEventListener("ended", function () { drawLane(lane); });
        drawLane(lane);
        return row;
    }

    // Lanes are kept, not rebuilt: a playing <audio> would stop if its element
    // were replaced under it. New outputs go on top (newest first); rows whose
    // output has gone are removed; the rest are updated in place.
    function renderOutputs() {
        if (!nodes.lanesList) return;
        const wanted = {};
        const ordered = state.outputs.slice().sort(function (a, b) {
            return (Number(b.created) || 0) - (Number(a.created) || 0);
        });
        ordered.forEach(function (output) { wanted[output.id] = output; });
        Object.keys(nodes.lanes).forEach(function (id) {
            if (wanted[id]) return;
            const row = nodes.lanes[id];
            try { row.mcVoiceBoxLane.audio.pause(); } catch (error) { /* ignore */ }
            forget("output:" + id);
            if (row.parentNode) row.parentNode.removeChild(row);
            delete nodes.lanes[id];
        });
        for (let index = ordered.length - 1; index >= 0; index -= 1) {
            const output = ordered[index];
            const row = nodes.lanes[output.id];
            if (row) {
                const lane = row.mcVoiceBoxLane;
                lane.output = output;
                lane.name.textContent = output.name || "Untitled";
                lane.meta.textContent = metadata(output);
                lane.audio.loop = !!output.loop;
                pressed(lane.loop, !!output.loop);
                drawLane(lane);
            } else {
                const made = laneNode(output);
                nodes.lanes[output.id] = made;
                if (nodes.lanesList.firstChild) nodes.lanesList.insertBefore(made, nodes.lanesList.firstChild);
                else nodes.lanesList.appendChild(made);
            }
        }
        show(nodes.lanesEmpty, !ordered.length);
    }

    function clearLanes() {
        Object.keys(nodes.lanes).forEach(function (id) {
            const row = nodes.lanes[id];
            try { row.mcVoiceBoxLane.audio.pause(); } catch (error) { /* ignore */ }
            forget("output:" + id);
            if (row.parentNode) row.parentNode.removeChild(row);
        });
        nodes.lanes = {};
    }

    function toggleLane(lane) {
        if (playing(lane.audio)) {
            lane.audio.pause();
            return Promise.resolve();
        }
        return audioUrl("output:" + lane.output.id,
                        ROUTES.outputAudio + "?id=" + encodeURIComponent(lane.output.id))
            .then(function (address) { return startPlayback(lane.audio, address); })
            .then(function () { drawLane(lane); });
    }

    function toggleLoop(lane) {
        const on = !lane.output.loop;
        lane.output.loop = on;
        lane.audio.loop = on;
        pressed(lane.loop, on);
        return request("outputs/loop:" + lane.output.id, ROUTES.outputLoop,
                       {body: {id: lane.output.id, on: on, loop: on}, queue: true})
            .catch(function (error) {
                lane.output.loop = !on;
                lane.audio.loop = !on;
                pressed(lane.loop, !on);
                throw error;
            });
    }

    function trimOutput(lane) {
        return audioUrl("output:" + lane.output.id,
                        ROUTES.outputAudio + "?id=" + encodeURIComponent(lane.output.id))
            .then(function () {
                return openTrimmer({name: lane.output.name || "render",
                                    blob: cache.blobs["output:" + lane.output.id],
                                    from: "output:" + lane.output.id});
            })
            .then(function () {
                if (nodes.trimmer && typeof nodes.trimmer.scrollIntoView === "function") {
                    try { nodes.trimmer.scrollIntoView({block: "nearest"}); } catch (error) { /* ignore */ }
                }
            });
    }

    // Save goes to the folder the server remembers. The first press on an
    // installation with no folder answers 409 with a sentence about choosing
    // one; the native picker is opened then, and the save tried once more.
    function saveOutput(lane) {
        const save = function () {
            return request("outputs/save:" + lane.output.id, ROUTES.outputSave,
                           {body: {id: lane.output.id}});
        };
        return save().catch(function (error) {
            if (!(error instanceof RequestError) || error.status !== 409
                || !/folder/i.test(error.message)) {
                throw error;
            }
            say("Choose where Voice Box saves renders…", "info");
            return request("settings/folder", ROUTES.folder, {body: {}}).then(function (found) {
                if (state.status && found && found.save_folder) {
                    state.status.settings = Object.assign({}, state.status.settings || {},
                                                          {save_folder: found.save_folder});
                }
                return save();
            });
        }).then(function (found) {
            say("Saved" + (found && found.path ? " to " + found.path : "") + ".", "info");
        });
    }

    function downloadOutput(lane) {
        const route = ROUTES.outputAudio + "?id=" + encodeURIComponent(lane.output.id) + "&download=1";
        return request("download:" + lane.output.id, route,
                       {method: "GET", expect: "blob", deadline: DEADLINE.audio})
            .then(function (blob) {
                const address = URL.createObjectURL(blob);
                const link = document.createElement("a");
                link.href = address;
                link.setAttribute("href", address);
                const filename = safeName(lane.output.name) + ".wav";
                link.download = filename;
                link.setAttribute("download", filename);
                link.hidden = true;
                document.body.appendChild(link);
                try { link.click(); } finally {
                    document.body.removeChild(link);
                }
                window.setTimeout(function () {
                    try { URL.revokeObjectURL(address); } catch (error) { /* ignore */ }
                }, 60000);
            });
    }

    function deleteOutput(lane) {
        return request("outputs/delete:" + lane.output.id, ROUTES.outputDelete,
                       {body: {id: lane.output.id}})
            .then(refreshOutputs);
    }

    // -- pipelines ------------------------------------------------------------- //

    function pipelineById(id) {
        return state.pipelines.filter(function (found) { return found.id === id; })[0] || null;
    }

    function currentPipeline() {
        return pipelineById(state.pipelineId);
    }

    function renderPipelineBar() {
        if (!nodes.pipelineSelect) return;
        fillSelect(nodes.pipelineSelect, state.pipelines.map(function (found) {
            return {value: found.id, label: found.name || "Untitled"};
        }), state.pipelineId);
        const pipeline = currentPipeline();
        nodes.title.textContent = "Voice Box" + (pipeline ? " · " + (pipeline.name || "Untitled") : "");
        nodes.pipelineRename.disabled = !pipeline;
        nodes.pipelineDelete.disabled = !pipeline;
    }

    function choosePipeline(id) {
        const found = pipelineById(id) || state.pipelines[0] || null;
        const changed = !found || found.id !== state.pipelineId;
        state.pipelineId = found ? found.id : "";
        remember(state.pipelineId);
        if (found) {
            state.prompt = found.prompt || "";
            state.dirty.prompt = false;
            state.configurationId = configurationById(found.configuration_id)
                ? found.configuration_id
                : (state.configurations[0] ? state.configurations[0].id : "");
            loadWorking();
        }
        if (changed) clearLanes();
        renderAll();
        return refreshOutputs();
    }

    function newPipeline(givenName) {
        const fallback = "Pipeline " + (state.pipelines.length + 1);
        const name = givenName || ask("Name the new pipeline", fallback);
        if (name === null) return Promise.resolve(null);
        return request("pipelines/new", ROUTES.pipelineNew, {body: {name: name}})
            .then(function (reply) {
                const made = recordOf(reply, "pipeline");
                return refreshPipelines().then(function () {
                    return choosePipeline(made.id || (state.pipelines.length
                        ? state.pipelines[state.pipelines.length - 1].id : ""));
                });
            });
    }

    function renamePipeline() {
        const pipeline = currentPipeline();
        if (!pipeline) return Promise.resolve();
        const name = ask("Rename this pipeline", pipeline.name || "");
        if (name === null || name === pipeline.name) return Promise.resolve();
        return request("pipelines/save", ROUTES.pipelineSave,
                       {body: {id: pipeline.id, name: name}, queue: true})
            .then(refreshPipelines)
            .then(renderPipelineBar);
    }

    function deletePipeline() {
        const pipeline = currentPipeline();
        if (!pipeline) return Promise.resolve();
        return request("pipelines/delete", ROUTES.pipelineDelete, {body: {id: pipeline.id}})
            .then(function () {
                state.pipelineId = "";
                return refreshPipelines();
            })
            .then(function () {
                if (!state.pipelines.length) return newPipeline("Pipeline 1");
                return choosePipeline(state.pipelines[0].id);
            });
    }

    // -- fetching -------------------------------------------------------------- //

    function isLive(job) {
        if (!job) return false;
        if (job.live !== undefined) return !!job.live;
        return !!LIVE_PHASES[job.phase];
    }

    function liveJobIds() {
        return state.jobs.filter(function (job) {
            return state.ownJobs[job.id] && isLive(job);
        }).map(function (job) { return job.id; });
    }

    function jobById(id) {
        return state.jobs.filter(function (job) { return job.id === id; })[0] || null;
    }

    function refreshStatus() {
        const before = liveJobIds();
        return request("status", ROUTES.status, {body: {}}).then(function (found) {
            state.status = found || {};
            state.jobs = Array.isArray(state.status.jobs) ? state.status.jobs : [];
            renderFooter();
            renderConfigurationFields();
            const finished = before.filter(function (id) {
                return !isLive(jobById(id));
            });
            if (finished.length) {
                return Promise.all([refreshOutputs(), refreshPrompts()]).then(function () {
                    return found;
                });
            }
            return found;
        });
    }

    function refreshSamples() {
        return request("samples", ROUTES.samples, {body: {}}).then(function (found) {
            state.samples = listOf(found, "samples");
            renderSamples();
            drawSpeakers();
            renderFooter();
        });
    }

    function refreshPrompts() {
        return request("prompts", ROUTES.prompts, {body: {}}).then(function (found) {
            const record = recordOf(found, "prompts");
            state.prompts = {history: listOf(record.history, "history"),
                             favourites: listOf(record.favourites, "favourites")};
            renderPromptLists();
        });
    }

    function refreshConfigurations() {
        return request("configurations", ROUTES.configurations, {body: {}}).then(function (found) {
            state.configurations = listOf(found, "configurations");
            if (state.configurationId && !configurationById(state.configurationId)) {
                state.configurationId = "";
            }
            renderConfigurationBar();
        });
    }

    function refreshPipelines() {
        return request("pipelines", ROUTES.pipelines, {body: {}}).then(function (found) {
            state.pipelines = listOf(found, "pipelines");
            renderPipelineBar();
        });
    }

    function refreshOutputs() {
        if (!state.pipelineId) {
            state.outputs = [];
            renderOutputs();
            return Promise.resolve();
        }
        return request("outputs", ROUTES.outputs, {body: {pipeline_id: state.pipelineId}})
            .then(function (found) {
                state.outputs = listOf(found, "outputs").filter(function (output) {
                    return !output.pipeline_id || output.pipeline_id === state.pipelineId;
                });
                renderOutputs();
            });
    }

    function renderAll() {
        renderPipelineBar();
        renderPrompt();
        renderPromptLists();
        renderConfiguration();
        renderSamples();
        renderOutputs();
        renderFooter();
    }

    // -- the footer: Render, the jobs, the cards -------------------------------- //

    function renderBlocker() {
        if (!state.status) return "Loading…";
        const engine = state.status.engine || {};
        if (engine.ready === false) {
            return engine.message || "VibeVoice is not installed.";
        }
        if (!state.pipelineId) return "No pipeline.";
        const parsed = parseScript(state.prompt);
        if (parsed.error) return parsed.error;
        if (!parsed.words) return "Write a script first.";
        const working = ensureWorking();
        const settings = state.status.settings || {};
        if (!working.card_uuid && !settings.card_uuid) return "Choose the card VibeVoice renders on.";
        const numbers = Object.keys(parsed.speakers).map(Number).sort();
        for (let index = 0; index < numbers.length; index += 1) {
            const number = numbers[index];
            if (!sampleById(working.speakers[String(number)])) {
                return "Speaker " + number + " has no sample.";
            }
        }
        return "";
    }

    function flushSaves() {
        if (timers.prompt) {
            window.clearTimeout(timers.prompt);
            timers.prompt = 0;
        }
        return state.dirty.prompt ? savePrompt() : Promise.resolve();
    }

    function renderNow() {
        const why = renderBlocker();
        if (why) {
            say(why, "warn");
            return Promise.resolve(null);
        }
        const pipelineId = state.pipelineId;
        const working = ensureWorking();
        // The saved configuration when the screen shows it unchanged, otherwise
        // what the screen shows, inline, validated by the server the same way.
        const unsaved = state.dirty.configuration || !configurationById(state.configurationId);
        return flushSaves().then(function () {
            const body = {pipeline_id: pipelineId, prompt: state.prompt,
                          configuration_id: state.configurationId, name: ""};
            if (unsaved) body.configuration = Object.assign({}, working);
            return request("render", ROUTES.render, {body: body});
        }).then(function (reply) {
            const job = recordOf(reply, "job");
            if (job.id) {
                state.ownJobs[job.id] = true;
                if (!jobById(job.id)) state.jobs = [job].concat(state.jobs);
            }
            say("Render queued.", "info");
            renderFooter();
            schedulePoll();
            return refreshStatus();
        });
    }

    function installEngine() {
        return request("install", ROUTES.install, {body: {part: "", folder: ""}}).then(function () {
            say("Installing VibeVoice… progress shows below.", "info");
            return refreshStatus();
        });
    }

    function cancelJob(job) {
        return request("jobs/cancel:" + job.id, ROUTES.jobCancel, {body: {id: job.id}})
            .then(refreshStatus);
    }

    // One flat record (mc_voice_vibevoice.progress): running, text, fraction, failed.
    function progressLine(progress) {
        const found = progress || {};
        if (found.failed) return "Install failed" + (found.text ? ": " + found.text : ".");
        if (!found.running && !found.text) return "";
        let text = found.text || (found.running ? "Installing…" : "");
        const fraction = Math.max(0, Math.min(1, Number(found.fraction) || 0));
        if (found.running && fraction > 0) text += " (" + Math.round(fraction * 100) + "%)";
        return text;
    }

    function cardsLine(status) {
        const parts = [];
        (status.turns || []).forEach(function (turn) {
            let text = (turn.card || turn.uuid || "a card") + ": " + (turn.active || "idle");
            if (turn.queued && turn.queued.length) text += ", " + turn.queued.length + " waiting";
            if (turn.warm) text += ", " + turn.warm + " warm";
            if (turn.parked) text += ", image model parked";
            if (turn.lease) text += ", on lease from WanGP";
            parts.push(text);
        });
        const runtime = (status.runtime && status.runtime.cards) || {};
        Object.keys(runtime).forEach(function (key) {
            const card = runtime[key] || {};
            if (!card.running) return;
            parts.push("VibeVoice " + (card.rendering ? "rendering" : (card.loaded ? "loaded" : "starting"))
                       + " on " + (card.device_name || key) + " (" + gigabytes(card.resident_bytes) + ")");
        });
        if (status.runtime && status.runtime.last_error) parts.push("last error: " + status.runtime.last_error);
        return parts.length ? parts.join(" · ") : "No card is busy.";
    }

    function renderFooter() {
        if (!nodes.render) return;
        const status = state.status || {};
        const engine = status.engine || {};
        const why = renderBlocker();
        nodes.render.disabled = !!why;
        nodes.render.setAttribute("aria-disabled", why ? "true" : "false");
        nodes.renderReason.textContent = why;
        const installable = state.status && engine.ready === false && engine.supported !== false;
        show(nodes.install, !!installable);
        if (installable) {
            nodes.install.textContent = "Install VibeVoice"
                + (engine.download_bytes ? " (" + gigabytes(engine.download_bytes) + ")" : "");
        }
        nodes.installProgress.textContent = progressLine(status.progress);
        clear(nodes.jobs);
        const jobs = state.jobs.filter(function (job) {
            return !job.pipeline_id || job.pipeline_id === state.pipelineId;
        });
        jobs.forEach(function (job) {
            const row = el("li", "mc-voice-box-job");
            row.setAttribute("data-phase", job.phase || "");
            let text = (job.name || ("Job " + String(job.id || "").slice(0, 6))) + " — " + (job.phase || "?");
            if (job.reason) text += ": " + job.reason;
            if (job.warning) text += " — " + job.warning;
            const progress = job.progress || {};
            if (progress.sections) {
                text += " (section " + (progress.section || 0) + "/" + progress.sections
                    + (progress.seconds ? ", " + seconds(progress.seconds) : "") + ")";
            }
            row.appendChild(el("span", "mc-voice-box-job-text", text));
            if (isLive(job)) {
                row.appendChild(button("Cancel", "Cancel this render", function () {
                    return cancelJob(job);
                }, "mc-voice-box-job-cancel"));
            }
            nodes.jobs.appendChild(row);
        });
        nodes.cards.textContent = state.status ? cardsLine(status) : "";
        clear(nodes.runtimeActions);
        const runtime = (status.runtime && status.runtime.cards) || {};
        Object.keys(runtime).forEach(function (key) {
            const card = runtime[key] || {};
            if (!card.running) return;
            const uuid = card.uuid || key;
            nodes.runtimeActions.appendChild(button(
                "Unload VibeVoice from " + (card.device_name || key),
                "Unload VibeVoice from " + (card.device_name || key), function () {
                    return request("runtime:" + uuid, ROUTES.runtime,
                                   {body: {action: "unload", card_uuid: uuid}})
                        .then(function () {
                            say("VibeVoice was unloaded.", "info");
                            return refreshStatus();
                        });
                }, "mc-voice-box-unload"));
        });
    }

    // -- polling --------------------------------------------------------------- //

    function schedulePoll() {
        if (timers.poll) {
            window.clearTimeout(timers.poll);
            timers.poll = 0;
        }
        if (!state.booted) return;
        if (document.visibilityState === "hidden") {
            state.poll.delay = 0;
            state.poll.live = false;
            return;
        }
        const live = liveJobIds().length > 0;
        state.poll.live = live;
        state.poll.delay = live ? POLL.live : POLL.idle;
        timers.poll = window.setTimeout(function () {
            timers.poll = 0;
            refreshStatus().catch(report).then(schedulePoll);
        }, state.poll.delay);
    }

    // -- inline renaming ------------------------------------------------------- //

    function renameInline(target, current, save) {
        if (target.getAttribute("data-editing") === "1") return;
        target.setAttribute("data-editing", "1");
        const input = el("input", "mc-voice-box-rename");
        input.setAttribute("type", "text");
        input.setAttribute("aria-label", "New name");
        input.value = current;
        let done = false;
        const finish = function (commit) {
            if (done) return;
            done = true;
            if (input.parentNode) input.parentNode.removeChild(input);
            target.removeAttribute("data-editing");
            show(target, true);
            const name = String(input.value || "").trim();
            if (commit && name && name !== current) {
                Promise.resolve(save(name)).catch(report);
            }
        };
        input.addEventListener("keydown", function (event) {
            if (event.key === "Enter") finish(true);
            else if (event.key === "Escape") finish(false);
        });
        input.addEventListener("blur", function () { finish(true); });
        show(target, false);
        if (target.parentNode) target.parentNode.insertBefore(input, target.nextSibling);
        try {
            if (typeof input.focus === "function") input.focus();
            if (typeof input.select === "function") input.select();
        } catch (error) { /* ignore */ }
    }

    // -- the public face ------------------------------------------------------- //

    window.mcVoiceBox = {
        state: function () { return state; },
        boot: boot,
        focusEvent: function () {
            return {name: FOCUS_EVENT, owner: OWNER, last: state.focus.last,
                    sent: state.focus.sent.slice()};
        },
        encodeWav: encodeWav,
        clampSelection: clampSelection,
        summarize: summarize,
        parseScript: parseScript,
        resampleLinear: resampleLinear,
        peaksOf: peaksOf,
    };

    if (typeof onUiLoaded === "function") {
        onUiLoaded(boot);
    } else if (document.readyState !== "loading") {
        boot();
    } else {
        document.addEventListener("DOMContentLoaded", boot);
    }
    // The root is painted by Gradio once, but a UI rebuilt after settings were
    // applied paints it again; `boot` is idempotent through the attribute it
    // sets, so re-running it costs one query.
    if (typeof onAfterUiUpdate === "function") {
        onAfterUiUpdate(boot);
    }
})();
