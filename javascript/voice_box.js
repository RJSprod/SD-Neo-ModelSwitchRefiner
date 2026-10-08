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
// The page fits the window rather than lengthening it. The root's height is
// measured and written in pixels on the root itself (a percentage height
// inside Gradio's containers resolves to auto -- Mini Paint NEO's lesson), each
// long list scrolls inside its own stage, and below a root width of 900 px the
// stages are cards side by side, one screen wide each, swiped left and right
// with horizontal scroll-snap and a stage bar over them that marks the one in
// view; there each stage's body is its one vertical scroller and nothing in
// it scrolls on its own. `data-layout` ("columns" or "stack") on the root says
// which, for the stylesheet. Nothing observes the root's size, because the
// root is the box this file resizes: the height is worked out again on the
// window's and the visual viewport's resize, an orientation change, the tab
// coming into view (an IntersectionObserver, which watches visibility, not
// size), the fonts arriving and the first data paint -- once per animation
// frame, and written only when it changed.
//
// Render, the Install button and one status line live in the Configuration
// stage's header. The line shows the first of: a message a press just caused
// (for a few seconds), an install running, the job on a card -- how many
// takes, for a batch -- with its time ticking from the server's own count, the
// queue, the last render this page started having failed, VibeVoice warm on a
// card, why Render is disabled, Ready.
//
// Every player -- a sample row, the trimmer, the selected output lane -- has
// the same transport: Play/Pause, From the start and Stop, as icons. One
// player at a time is the active one (the one last started and not stopped),
// and only its waveform seeks and draws a playhead.
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
    // A job holding its card, as opposed to one queued behind it.
    const ACTIVE_PHASES = {waiting: true, loading: true, rendering: true};
    // Below this width of the root the stages are cards, one screen wide each.
    const STACK_BELOW = 900;
    // The least height the root is given, whatever the window leaves it.
    const MIN_HEIGHT = 420;
    // How long a message holds the status line before the line goes back to
    // what the page is doing. Its Dismiss puts it away sooner.
    const MESSAGE_MS = {info: 5000, other: 12000};
    const TICK_MS = 1000;
    // A press that travels further than this before it lets go is a drag or
    // a scroll, and selects nothing.
    const DRAG_PX = 8;
    const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct",
                    "Nov", "Dec"];
    // `bar` is the stage's word in the phone's stage bar, short enough for
    // four of them across 360 px; `name` is the whole of it, for its label.
    const STAGES = [
        {key: "input", title: "INPUT", bar: "Input", name: "Input"},
        {key: "prompt", title: "PROMPT", bar: "Prompt", name: "Prompt"},
        {key: "configuration", title: "CONFIGURATION", bar: "Config", name: "Configuration"},
        {key: "outputs", title: "OUTPUTS", bar: "Outputs", name: "Outputs"},
    ];
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
        jobClear: "/jobs/clear",
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
        // The output lane showing its controls, by output id; it survives
        // polls and pipeline switches, and a render this page started takes
        // it when it finishes.
        selectedOutput: "",
        revealOutput: "",
        // The last render this page started, when it failed: {id, name, warning}.
        lastFailure: null,
        // performance.now() when the last list of jobs arrived. A job's time is
        // its server-computed `elapsed` plus the time since then -- never this
        // browser's clock against the server's timestamps.
        answeredAt: 0,
        statusLine: "",
        layout: {mode: "", width: 0, height: 0, raw: 0, correction: 0, writes: 0},
        // The one player last started with Play and not stopped since:
        // {kind: "sample" | "trimmer" | "output", id}, or null. Only its
        // waveform seeks, draws a playhead and takes sideways drags.
        activePlayer: null,
        // The stage the phone's stage bar marks: the one in view.
        stage: "input",
    };

    const nodes = {lanes: {}};
    const cache = {urls: {}, blobs: {}};
    const players = [];
    const inflight = {};
    const timers = {prompt: 0, poll: 0, tick: 0, message: 0};
    const frames = {fit: 0, track: 0};

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

    // A message is what a press just did, or why it could not: it holds the
    // Configuration header's one status line for a few seconds (MESSAGE_MS), and the
    // line then goes back to what the page is doing (renderStatus).
    function say(text, kind) {
        state.message = text || "";
        state.messageKind = kind || "info";
        if (timers.message) {
            window.clearTimeout(timers.message);
            timers.message = 0;
        }
        if (state.message) {
            const shown = state.message;
            timers.message = window.setTimeout(function () {
                timers.message = 0;
                if (state.message !== shown) return;
                state.message = "";
                renderStatus();
            }, state.messageKind === "info" ? MESSAGE_MS.info : MESSAGE_MS.other);
        }
        renderStatus();
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
                scheduleTick(null);
            } else {
                refreshStatus().catch(report).then(schedulePoll);
                scheduleFit();
            }
        });
        // Everything that changes the room the window leaves the root. The
        // canvases are redrawn by the fit when the root's width has moved.
        if (typeof window.addEventListener === "function") {
            window.addEventListener("resize", scheduleFit);
            window.addEventListener("orientationchange", scheduleFit);
        }
        const viewport = window.visualViewport;
        if (viewport && typeof viewport.addEventListener === "function") {
            viewport.addEventListener("resize", scheduleFit);
        }
        try {
            const fonts = document.fonts;
            if (fonts && fonts.ready && typeof fonts.ready.then === "function") {
                fonts.ready.then(scheduleFit, function () { /* nothing to wait for */ });
            }
        } catch (error) { /* a document without the font loading API */ }
        // A tap anywhere outside the selected output lane, and Escape, put it
        // away (onPageClick, onPageKey).
        document.addEventListener("pointerdown", notePress, true);
        document.addEventListener("click", onPageClick);
        document.addEventListener("keydown", onPageKey);
    }

    // Gradio shows a tab by switching its panel from display:none, which moves
    // nothing a window event would report. An IntersectionObserver on the root
    // hears the tab come into view; it watches visibility, not size, so the
    // height this file writes on the root cannot call it back.
    function watchRoot(root) {
        if (typeof IntersectionObserver !== "function" || !root) return;
        try {
            const observer = new IntersectionObserver(function (entries) {
                const seen = (entries || []).some(function (entry) { return entry && entry.isIntersecting; });
                if (seen) scheduleFit();
            });
            observer.observe(root);
            nodes.rootObserver = observer;
        } catch (error) { /* the window's events still fit it */ }
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

    // Plays `audio` from `address`: from where it is, or from `at` seconds
    // when that is given (From the start). Every other player is paused first.
    function startPlayback(audio, address, at) {
        pauseAll(audio);
        if (audio.mcVoiceBoxSource !== address) {
            audio.src = address;
            audio.mcVoiceBoxSource = address;
        }
        if (typeof at === "number") {
            // Before the file's length is known this is where playback will
            // begin, which is all From the start needs.
            try { audio.currentTime = at; } catch (error) { /* not seekable yet */ }
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

    // -- the active player ----------------------------------------------------- //
    //
    // One player at a time is active: the one last started with Play (or From
    // the start) and not stopped since -- playing, or paused by its own Pause.
    // Starting another makes that one active, the first having been paused by
    // startPlayback; Stop ends it. Only the active player's waveform seeks when
    // pressed or dragged, only it draws a playhead, and only it takes sideways
    // drags on a touch screen (touch-action: pan-y), so a swipe over any other
    // waveform is the phone's stage pager's.

    function isActivePlayer(kind, id) {
        const active = state.activePlayer;
        return !!active && active.kind === kind && (kind === "trimmer" || active.id === id);
    }

    function setActivePlayer(kind, id) {
        state.activePlayer = kind ? {kind: kind, id: id || ""} : null;
        refreshPlayers();
    }

    // Every player's waveform and transport, after the active one changed.
    function refreshPlayers() {
        Object.keys(nodes.lanes).forEach(function (key) {
            const lane = nodes.lanes[key] && nodes.lanes[key].mcVoiceBoxLane;
            if (lane) drawLane(lane);
        });
        drawSampleRows();
        drawTrimmer();
    }

    function setTouch(canvas, on) {
        if (!canvas || !canvas.style) return;
        const wanted = on ? "pan-y" : "";
        if ((canvas.style.touchAction || "") !== wanted) canvas.style.touchAction = wanted;
    }

    // A finger that lands on a waveform may be starting to scroll the stage
    // (the waveform leaves up and down to the page): its press counts once
    // it moves along the waveform or lifts, and a scroll -- which the browser
    // ends with pointercancel -- changes nothing. A mouse's press counts at once.
    function byFinger(event) {
        return !!event && event.pointerType === "touch";
    }

    // A press or a drag along a waveform seeks -- on the active player's only;
    // anywhere else the press is left to whatever else it does (on a lane, to
    // select it).
    function wireScrub(canvas, active, seekTo) {
        let held = false;
        let waiting = false;
        canvas.addEventListener("pointerdown", function (event) {
            if (!active()) return;
            held = true;
            try {
                if (typeof canvas.setPointerCapture === "function" && event.pointerId !== undefined) {
                    canvas.setPointerCapture(event.pointerId);
                }
            } catch (error) { /* not every pointer can be captured */ }
            waiting = byFinger(event);
            if (!waiting) seekTo(fractionAt(canvas, event));
        });
        canvas.addEventListener("pointermove", function (event) {
            if (!held || !active()) return;
            waiting = false;
            seekTo(fractionAt(canvas, event));
        });
        canvas.addEventListener("pointerup", function (event) {
            if (held && waiting && active()) seekTo(fractionAt(canvas, event));
            held = false;
            waiting = false;
        });
        canvas.addEventListener("pointercancel", function () {
            held = false;
            waiting = false;
        });
    }

    function seekAudio(audio, fraction, fallbackSeconds) {
        const total = Number(audio.duration) || Number(fallbackSeconds) || 0;
        if (!total) return;
        try { audio.currentTime = fraction * total; } catch (error) { /* not seekable yet */ }
    }

    // Stop's half that is the element's: paused, and back at `at` seconds.
    function stopAudio(audio, at) {
        try {
            if (!audio.paused && typeof audio.pause === "function") audio.pause();
        } catch (error) { /* an element with no source */ }
        try { audio.currentTime = at || 0; } catch (error) { /* not seekable yet */ }
    }

    // -- transports: play/pause, from the start, stop, loop --------------------- //

    const SVG = "http://www.w3.org/2000/svg";
    // 16 x 16, in currentColor: nothing fetched, no icon font, no emoji.
    const ICONS = {
        play: [["path", {d: "M4.5 2.5v11l9-5.5z"}]],
        pause: [["rect", {x: "3.5", y: "2.5", width: "3", height: "11"}],
                ["rect", {x: "9.5", y: "2.5", width: "3", height: "11"}]],
        start: [["rect", {x: "2.5", y: "2.5", width: "2", height: "11"}],
                ["path", {d: "M13.5 2.5v11L5.5 8z"}]],
        stop: [["rect", {x: "3", y: "3", width: "10", height: "10"}]],
        loop: [["path", {d: "M12.6 8.6A4.7 4.7 0 1 1 11 4.3", fill: "none", stroke: "currentColor",
                         "stroke-width": "1.8", "stroke-linecap": "round"}],
               ["path", {d: "M13.8 1.8v4.6H9.2z"}]],
        close: [["path", {d: "M4 4l8 8M12 4l-8 8", fill: "none", stroke: "currentColor",
                          "stroke-width": "1.8", "stroke-linecap": "round"}]],
        // A disk: its body with a cut corner, the shutter at the top and the
        // label at the bottom.
        save: [["path", {d: "M2.5 2.5h8.2l2.8 2.8v8.2h-11z", fill: "none", stroke: "currentColor",
                         "stroke-width": "1.4", "stroke-linejoin": "round"}],
               ["rect", {x: "5", y: "2.5", width: "4.5", height: "3.2"}],
               ["rect", {x: "4.6", y: "8.6", width: "6.8", height: "4.9", rx: "0.6"}]],
        // The disk, smaller, with a plus beside it: a new configuration.
        saveAs: [["path", {d: "M1.5 1.5h6.4l2.1 2.1v6.4h-8.5z", fill: "none", stroke: "currentColor",
                           "stroke-width": "1.3", "stroke-linejoin": "round"}],
                 ["rect", {x: "3.5", y: "1.5", width: "3.4", height: "2.4"}],
                 ["rect", {x: "3.2", y: "6.2", width: "5", height: "3.8", rx: "0.5"}],
                 ["path", {d: "M12.5 9.5v5M10 12h5", fill: "none", stroke: "currentColor",
                           "stroke-width": "1.6", "stroke-linecap": "round"}]],
        // A bin: the lid and its handle, the can and two ribs.
        delete: [["path", {d: "M2.5 4.2h11M6.2 4.2V2.6h3.6v1.6", fill: "none", stroke: "currentColor",
                           "stroke-width": "1.4", "stroke-linecap": "round", "stroke-linejoin": "round"}],
                 ["path", {d: "M3.8 4.2l.8 9.3h6.8l.8-9.3", fill: "none", stroke: "currentColor",
                           "stroke-width": "1.4", "stroke-linejoin": "round"}],
                 ["path", {d: "M6.6 6.6v4.6M9.4 6.6v4.6", fill: "none", stroke: "currentColor",
                           "stroke-width": "1.2", "stroke-linecap": "round"}]],
    };

    function svgNode(tag, attributes) {
        const node = typeof document.createElementNS === "function"
            ? document.createElementNS(SVG, tag) : document.createElement(tag);
        Object.keys(attributes || {}).forEach(function (name) {
            node.setAttribute(name, attributes[name]);
        });
        return node;
    }

    function icon(name) {
        const svg = svgNode("svg", {viewBox: "0 0 16 16", width: "16", height: "16",
                                    fill: "currentColor", "aria-hidden": "true", focusable: "false"});
        (ICONS[name] || []).forEach(function (part) { svg.appendChild(svgNode(part[0], part[1])); });
        return svg;
    }

    // Its picture and its name; the name is both the label read out and the
    // tooltip. A player redraws several times a second while it plays, so
    // each is written only when it changes.
    function setIcon(node, name, label) {
        if (node.getAttribute("data-icon") !== name) {
            clear(node);
            node.appendChild(icon(name));
            node.setAttribute("data-icon", name);
        }
        if (node.getAttribute("aria-label") !== label) node.setAttribute("aria-label", label);
        if (node.getAttribute("title") !== label) node.setAttribute("title", label);
    }

    // A square button the size of the section's others, with an icon for a face.
    function iconButton(name, label, onClick, className) {
        const node = button("", label, onClick, "mc-voice-box-icon " + className);
        setIcon(node, name, label);
        return node;
    }

    // Play/Pause, From the start and Stop, for one player; `prefix` names its
    // buttons' classes (mc-voice-box-lane-play and so on).
    function transport(prefix, handlers) {
        const box = el("span", "mc-voice-box-transport");
        const play = iconButton("play", "Play", handlers.play, prefix + "-play");
        const start = iconButton("start", "Play from the start", handlers.start, prefix + "-start");
        const stop = iconButton("stop", "Stop", handlers.stop, prefix + "-stop");
        [play, start, stop].forEach(function (node) { box.appendChild(node); });
        return {box: box, play: play, start: start, stop: stop};
    }

    // The Play/Pause button's face and name follow its player.
    function showPlaying(controls, on) {
        if (controls && controls.play) setIcon(controls.play, on ? "pause" : "play", on ? "Pause" : "Play");
    }

    // -- switches: an on/off setting as one button -------------------------------- //
    //
    // A drawn track and knob beside the setting's name, all of it one button:
    // no native checkbox, whose look is the host theme's to squeeze (a theme
    // took Sampling's down to a sliver nobody could see was a control). Off,
    // the knob sits left in an outlined track; on, it sits right in a filled
    // one -- the side says it without the colour. role="switch" with
    // aria-checked says it to assistive technology, and being a button, Space
    // and Enter work it.

    const SWITCH = {off: "7", on: "17"};

    function switchPicture() {
        const svg = svgNode("svg", {viewBox: "0 0 24 14", width: "34", height: "20", "aria-hidden": "true",
                                    focusable: "false", "class": "mc-voice-box-switch"});
        svg.appendChild(svgNode("rect", {"class": "mc-voice-box-switch-track", x: "0.75", y: "0.75",
                                         width: "22.5", height: "12.5", rx: "6.25"}));
        svg.appendChild(svgNode("circle", {"class": "mc-voice-box-switch-thumb", cx: SWITCH.off, cy: "7",
                                           r: "4.5"}));
        return svg;
    }

    function switchOn(node) {
        return !!node && node.getAttribute("aria-checked") === "true";
    }

    function setSwitch(node, on) {
        if (!node) return;
        const wanted = on ? "true" : "false";
        if (node.getAttribute("aria-checked") !== wanted) node.setAttribute("aria-checked", wanted);
        const thumb = node.mcVoiceBoxThumb;
        const x = on ? SWITCH.on : SWITCH.off;
        if (thumb && thumb.getAttribute("cx") !== x) thumb.setAttribute("cx", x);
    }

    // `text` is what the button shows; `label`, when given, is its whole name
    // (and its tooltip), for a setting whose words are shortened on the face.
    // A press flips it and hands the new state to `changed`.
    function switchButton(text, label, changed, className) {
        const node = button("", label || "", null, "mc-voice-box-switch-button " + (className || ""));
        node.setAttribute("role", "switch");
        if (label) node.setAttribute("title", label);
        else node.removeAttribute("aria-label");
        const picture = switchPicture();
        node.appendChild(picture);
        node.mcVoiceBoxThumb = picture.lastChild;
        node.appendChild(el("span", "mc-voice-box-switch-label", text));
        setSwitch(node, false);
        node.addEventListener("click", function (event) {
            if (event && typeof event.preventDefault === "function") event.preventDefault();
            const on = !switchOn(node);
            setSwitch(node, on);
            try {
                const result = changed(on);
                if (result && typeof result.catch === "function") result.catch(report);
            } catch (error) {
                report(error);
            }
        });
        return node;
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
        // Where the playhead was drawn, -1 for none: the picture itself cannot
        // be read back, and whether a waveform shows a head is a rule.
        canvas.mcVoiceBoxHead = typeof settings.head === "number" && settings.head >= 0 ? settings.head : -1;
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
        drawSampleRows();
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
        state.layout = {mode: "", width: 0, height: 0, raw: 0, correction: 0, writes: 0};
        build(root);
        listen();
        watchRoot(root);
        fit();
        loadAll().catch(report).then(function () {
            scheduleFit();
            schedulePoll();
        });
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

        // The phone's stage bar (the stylesheet shows it only when the stages
        // are cards swiped left and right): a press goes to a stage, and the
        // stage in view is the one marked.
        const stagebar = el("nav", "mc-voice-box-stagebar");
        stagebar.setAttribute("aria-label", "Stages");
        nodes.stagebar = {};
        STAGES.forEach(function (stage) {
            const node = button(stage.bar, stage.name, function () { goToStage(stage.key); },
                                "mc-voice-box-stagebar-button");
            node.setAttribute("data-stage", stage.key);
            node.setAttribute("title", stage.name);
            nodes.stagebar[stage.key] = node;
            stagebar.appendChild(node);
        });
        root.appendChild(stagebar);

        // Each stage is a head that stays and a body under it, the stage's one
        // vertical scroller. Side by side, the long list in a body takes what
        // room is left and scrolls inside it; as a phone's cards, the body
        // scrolls whole and nothing inside it scrolls on its own.
        const stages = el("div", "mc-voice-box-stages");
        STAGES.forEach(function (stage, index) {
            if (index) {
                const connector = el("div", "mc-voice-box-connector");
                connector.setAttribute("aria-hidden", "true");
                stages.appendChild(connector);
            }
            const card = el("section", "mc-voice-box-stage mc-voice-box-stage-" + stage.key);
            card.setAttribute("data-stage", stage.key);
            const top = el("div", "mc-voice-box-stage-head");
            top.appendChild(el("h3", "mc-voice-box-stage-title", stage.title));
            card.appendChild(top);
            const body = el("div", "mc-voice-box-stage-body");
            card.appendChild(body);
            nodes["stage_" + stage.key] = body;
            nodes["stageHead_" + stage.key] = top;
            nodes["stageCard_" + stage.key] = card;
            stages.appendChild(card);
        });
        stages.addEventListener("scroll", scheduleTrack);
        nodes.stages = stages;
        root.appendChild(stages);

        buildInput(nodes.stage_input);
        buildPrompt(nodes.stage_prompt);
        buildConfiguration(nodes.stage_configuration, nodes.stageHead_configuration);
        buildOutputs(nodes.stage_outputs);
        markStage(state.stage);
    }

    // -- the phone's stage bar ----------------------------------------------------- //

    // The stage whose left edge is nearest the stages' own: the one in view.
    function stageInView() {
        const container = nodes.stages;
        if (!container) return "";
        let origin = 0;
        try {
            origin = Number(container.getBoundingClientRect().left) || 0;
        } catch (error) {
            return "";
        }
        let found = "";
        let nearest = Infinity;
        STAGES.forEach(function (stage) {
            const card = nodes["stageCard_" + stage.key];
            let left = 0;
            try {
                left = (Number(card.getBoundingClientRect().left) || 0) - origin;
            } catch (error) {
                return;
            }
            if (Math.abs(left) < nearest) {
                nearest = Math.abs(left);
                found = stage.key;
            }
        });
        return found;
    }

    function markStage(key) {
        if (key) state.stage = key;
        STAGES.forEach(function (stage) {
            const node = nodes.stagebar && nodes.stagebar[stage.key];
            if (!node) return;
            if (stage.key === state.stage) node.setAttribute("aria-current", "true");
            else node.removeAttribute("aria-current");
        });
    }

    // The stages scroll: the bar follows once per frame, however many scroll
    // events the frame had.
    function scheduleTrack() {
        if (frames.track) return;
        if (typeof window.requestAnimationFrame !== "function") {
            markStage(stageInView());
            return;
        }
        frames.track = window.requestAnimationFrame(function () {
            frames.track = 0;
            markStage(stageInView());
        });
    }

    function reducedMotion() {
        try {
            return typeof window.matchMedia === "function"
                && !!window.matchMedia("(prefers-reduced-motion: reduce)").matches;
        } catch (error) {
            return false;
        }
    }

    function goToStage(key) {
        const container = nodes.stages;
        const card = nodes["stageCard_" + key];
        if (!container || !card) return;
        let left = 0;
        try {
            left = (Number(card.getBoundingClientRect().left) || 0)
                - (Number(container.getBoundingClientRect().left) || 0)
                + (Number(container.scrollLeft) || 0);
        } catch (error) {
            return;
        }
        const behavior = reducedMotion() ? "auto" : "smooth";
        if (typeof container.scrollTo === "function") container.scrollTo({left: left, behavior: behavior});
        else container.scrollLeft = left;
        markStage(key);
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
        // Its handles take sideways drags on a touch screen, always: up and
        // down still scroll the stage.
        setTouch(nodes.trimmerWave, true);
        box.appendChild(nodes.trimmerWave);
        nodes.trimmerReadout = el("div", "mc-voice-box-trimmer-readout", "");
        box.appendChild(nodes.trimmerReadout);
        const row = el("div", "mc-voice-box-row mc-voice-box-trimmer-row");
        nodes.trimmerControls = transport("mc-voice-box-trimmer", {
            play: playTrimmer, start: playTrimmerFromStart, stop: stopTrimmer});
        nodes.trimmerLoop = iconButton("loop", "Loop", toggleTrimmerLoop, "mc-voice-box-trimmer-loop");
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
        row.appendChild(nodes.trimmerControls.box);
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
        ["play", "pause", "ended"].forEach(function (type) {
            nodes.trimmerAudio.addEventListener(type, function () { drawTrimmer(); });
        });
        box.appendChild(nodes.trimmerAudio);
        show(box, false);
        body.appendChild(box);

        const library = el("div", "mc-voice-box-library");
        library.appendChild(el("h4", "mc-voice-box-subtitle", "Sample library"));
        nodes.samples = el("ul", "mc-voice-box-list mc-voice-box-samples");
        library.appendChild(nodes.samples);
        // One element plays every sample; the row whose sample it holds is
        // the active player while it is one.
        nodes.sampleAudio = player("mc-voice-box-sample-audio");
        ["timeupdate", "play", "pause", "ended"].forEach(function (type) {
            nodes.sampleAudio.addEventListener(type, drawActiveSample);
        });
        library.appendChild(nodes.sampleAudio);
        body.appendChild(library);
    }

    // What a scroller has scrolled, put back after the list in it is rebuilt:
    // drawing the new rows' waveforms lays the page out while the list is
    // still short, and the browser clamps the scroll to that.
    function keepScroll(scrollers) {
        const kept = scrollers.filter(Boolean).map(function (node) {
            return {node: node, at: Number(node.scrollTop) || 0};
        });
        return function () {
            kept.forEach(function (entry) {
                if (entry.at && entry.node.scrollTop !== entry.at) entry.node.scrollTop = entry.at;
            });
        };
    }

    // The list is rebuilt whole, and it scrolls inside its stage (or, on a
    // phone, the stage's body does): the scroll is put back afterwards, or
    // pressing a speaker button on a sample far down the list would throw the
    // list back to its top.
    function renderSamples() {
        if (!nodes.samples) return;
        const restore = keepScroll([nodes.samples, nodes.stage_input]);
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
            const edge = edgeOf([sample.id]);
            if (edge) row.appendChild(edge);
            const wave = canvasOf("mc-voice-box-sample-wave");
            wireScrub(wave, function () { return isActivePlayer("sample", sample.id); }, function (fraction) {
                seekAudio(nodes.sampleAudio, fraction, sample.seconds);
                drawActiveSample();
            });
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
            const controls = transport("mc-voice-box-sample", {
                play: function () { return playSample(sample); },
                start: function () { return playSampleFromStart(sample); },
                stop: function () { stopSample(sample); },
            });
            line.appendChild(controls.box);
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
            row.mcVoiceBoxSample = {id: sample.id, seconds: sample.seconds, wave: wave,
                                    peaks: sample.peaks || [], controls: controls};
            nodes.samples.appendChild(row);
            drawSampleRow(row);
        });
        restore();
    }

    // -- the tint a sample carries onto its outputs ---------------------------- //

    // A quiet hue per sample, from its id: FNV-1a over the id's characters, so
    // the same sample is the same hue on every page and every visit. The
    // stylesheet owns the saturation and the lightness (one for the light
    // theme, one for the dark); the page only says which hue. Decorative --
    // the titles say the same thing in words -- so it is hidden from
    // assistive technology.
    function hueOf(id) {
        let hash = 0x811c9dc5;
        const text = String(id || "");
        for (let index = 0; index < text.length; index += 1) {
            hash ^= text.charCodeAt(index);
            hash = Math.imul(hash, 0x01000193) >>> 0;
        }
        return hash % 360;
    }

    function setHue(node, hue) {
        const style = node.style;
        if (!style) return;
        if (typeof style.setProperty === "function") style.setProperty("--mc-voice-box-hue", String(hue));
        else style["--mc-voice-box-hue"] = String(hue);
    }

    // A 4 px edge of one equal segment per distinct sample, in the order
    // given; nothing at all for no samples.
    function edgeOf(ids) {
        const seen = {};
        const distinct = (ids || []).filter(function (id) {
            if (!id || seen[id]) return false;
            seen[id] = true;
            return true;
        });
        if (!distinct.length) return null;
        const edge = el("span", "mc-voice-box-edge");
        edge.setAttribute("aria-hidden", "true");
        distinct.forEach(function (id) {
            const segment = el("span", "mc-voice-box-edge-segment");
            segment.setAttribute("data-sample", id);
            setHue(segment, hueOf(id));
            edge.appendChild(segment);
        });
        return edge;
    }

    // An output's samples in speaker order: `render.speakers`, by number.
    function speakerSamples(output) {
        const render = (output && output.render) || {};
        const speakers = Array.isArray(render.speakers) ? render.speakers.slice() : [];
        speakers.sort(function (a, b) { return (Number(a && a.n) || 0) - (Number(b && b.n) || 0); });
        return speakers.map(function (speaker) { return speaker && speaker.sample_id; });
    }

    // -- a sample row as a player ---------------------------------------------- //

    function sampleRows() {
        const found = [];
        const rows = (nodes.samples && nodes.samples.children) || [];
        for (let index = 0; index < rows.length; index += 1) {
            if (rows[index].mcVoiceBoxSample) found.push(rows[index]);
        }
        return found;
    }

    // Its waveform with the playhead while it is the active player (and the
    // sideways drags that go with it), and its Play/Pause face.
    function drawSampleRow(row) {
        const entry = row.mcVoiceBoxSample;
        const audio = nodes.sampleAudio;
        const active = isActivePlayer("sample", entry.id);
        const total = Number(audio && audio.duration) || Number(entry.seconds) || 0;
        const head = active && total ? Math.min(1, (Number(audio.currentTime) || 0) / total) : -1;
        draw(entry.wave, entry.peaks, {head: head});
        setTouch(entry.wave, active);
        showPlaying(entry.controls, active && playing(audio));
    }

    function drawSampleRows() {
        sampleRows().forEach(drawSampleRow);
    }

    // The shared element's events concern the active sample's row alone.
    function drawActiveSample() {
        const active = state.activePlayer;
        if (!active || active.kind !== "sample") return;
        sampleRows().forEach(function (row) {
            if (row.mcVoiceBoxSample.id === active.id) drawSampleRow(row);
        });
    }

    function sampleAddress(sample) {
        return audioUrl("sample:" + sample.id,
                        ROUTES.sampleAudio + "?id=" + encodeURIComponent(sample.id));
    }

    // Play pauses the sample while it plays; otherwise it plays on from where
    // it was paused -- the start, for a sample the element does not hold.
    function playSample(sample) {
        const audio = nodes.sampleAudio;
        if (isActivePlayer("sample", sample.id) && playing(audio)) {
            audio.pause();
            return Promise.resolve();
        }
        return sampleAddress(sample).then(function (address) {
            setActivePlayer("sample", sample.id);
            return startPlayback(audio, address);
        });
    }

    function playSampleFromStart(sample) {
        return sampleAddress(sample).then(function (address) {
            setActivePlayer("sample", sample.id);
            return startPlayback(nodes.sampleAudio, address, 0);
        });
    }

    // The element is stopped only when it holds this sample: it may be
    // another's.
    function stopSample(sample) {
        const audio = nodes.sampleAudio;
        const address = cache.urls["sample:" + sample.id];
        if (address && audio.mcVoiceBoxSource === address) stopAudio(audio, 0);
        if (isActivePlayer("sample", sample.id)) setActivePlayer(null);
    }

    function deleteSample(sample) {
        if (isActivePlayer("sample", sample.id)) {
            stopAudio(nodes.sampleAudio, 0);
            state.activePlayer = null;
        }
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
        renderStatus();
        say("Speaker " + number + " is " + titleOf(sampleId) + ". Save the configuration to keep it.",
            "info");
    }

    function clearSpeaker(number) {
        const working = ensureWorking();
        delete working.speakers[String(number)];
        markConfigurationDirty();
        renderSamples();
        drawSpeakers();
        renderStatus();
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
        if (isActivePlayer("trimmer")) setActivePlayer(null);
        if (nodes.trimmer) show(nodes.trimmer, false);
    }

    function selectionFractions() {
        if (!trimmer.duration) return null;
        return {from: trimmer.selection.in / trimmer.duration,
                to: trimmer.selection.out / trimmer.duration};
    }

    // The trimmer's waveform is its handles' and never seeks; its playhead
    // shows while it is the active player.
    function drawTrimmer() {
        if (!nodes.trimmerWave || !trimmer.source) return;
        const audio = nodes.trimmerAudio;
        const head = trimmer.duration && isActivePlayer("trimmer")
            ? Math.min(1, (Number(audio.currentTime) || 0) / trimmer.duration) : -1;
        draw(nodes.trimmerWave, trimmer.peaks, {selection: selectionFractions(), head: head});
        if (nodes.trimmerReadout) {
            nodes.trimmerReadout.textContent = seconds(trimmer.selection.in) + " to "
                + seconds(trimmer.selection.out) + " (" + seconds(trimmer.selection.out - trimmer.selection.in)
                + " of " + seconds(trimmer.duration) + ")";
        }
        showPlaying(nodes.trimmerControls, playing(audio));
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

    // The handle nearest a press moves to it and follows the drag. A finger's
    // press waits until it moves along the waveform or lifts (byFinger): the
    // waveform leaves up and down to the page, so a finger scrolling the
    // stage moves no handle.
    function wireHandles(canvas) {
        let waiting = null;
        const grab = function (event) {
            const at = timeAt(canvas, event);
            const toIn = Math.abs(at - trimmer.selection.in);
            const toOut = Math.abs(at - trimmer.selection.out);
            trimmer.dragging = toIn <= toOut ? "in" : "out";
            trimmer.selection = clampSelection(trimmer.selection, trimmer.dragging, at, trimmer.duration);
            drawTrimmer();
        };
        const down = function (event) {
            if (!trimmer.source || !trimmer.duration) return;
            try {
                if (typeof canvas.setPointerCapture === "function" && event.pointerId !== undefined) {
                    canvas.setPointerCapture(event.pointerId);
                }
            } catch (error) { /* not every pointer can be captured */ }
            if (byFinger(event)) {
                waiting = event;
                return;
            }
            if (typeof event.preventDefault === "function") event.preventDefault();
            grab(event);
        };
        const move = function (event) {
            if (waiting) {
                const first = waiting;
                waiting = null;
                if (trimmer.source && trimmer.duration) grab(first);
            }
            if (!trimmer.dragging || !trimmer.duration) return;
            trimmer.selection = clampSelection(trimmer.selection, trimmer.dragging,
                                               timeAt(canvas, event), trimmer.duration);
            drawTrimmer();
        };
        const up = function (event) {
            if (waiting && trimmer.source && trimmer.duration) grab(event);
            waiting = null;
            trimmer.dragging = null;
        };
        const cancel = function () {
            waiting = null;
            trimmer.dragging = null;
        };
        canvas.addEventListener("pointerdown", down);
        canvas.addEventListener("pointermove", move);
        canvas.addEventListener("pointerup", up);
        canvas.addEventListener("pointercancel", cancel);
    }

    // Play pauses the selection while it plays; otherwise it plays on from
    // where it was paused inside the selection, or from the selection's start
    // (never started, stopped, or run to the out point).
    function playTrimmer() {
        const audio = nodes.trimmerAudio;
        if (!trimmer.source || !trimmer.mediaUrl) return Promise.resolve();
        if (playing(audio)) {
            audio.pause();
            drawTrimmer();
            return Promise.resolve();
        }
        const at = Number(audio.currentTime) || 0;
        const inside = at >= trimmer.selection.in && at < trimmer.selection.out - 0.05;
        return beginTrimmer(inside ? undefined : trimmer.selection.in);
    }

    // The beginning, for the trimmer, is the selection's start. A capture in
    // progress (Save as sample on a long file) is a playthrough of the
    // selection, so From the start and Stop end it, and nothing is saved.
    function playTrimmerFromStart() {
        if (!trimmer.source || !trimmer.mediaUrl) return Promise.resolve();
        stopCapture("The capture was stopped; nothing was saved.");
        return beginTrimmer(trimmer.selection.in);
    }

    function beginTrimmer(at) {
        setActivePlayer("trimmer", "");
        return startPlayback(nodes.trimmerAudio, trimmer.mediaUrl, at).then(drawTrimmer);
    }

    function stopTrimmer() {
        stopCapture("The capture was stopped; nothing was saved.");
        if (!trimmer.source) return;
        stopAudio(nodes.trimmerAudio, trimmer.selection.in);
        if (isActivePlayer("trimmer")) setActivePlayer(null);
        else drawTrimmer();
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
                // The capture plays the selection through: the trimmer is the
                // player playing, and its Play/Pause holds the capture.
                setActivePlayer("trimmer", "");
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
        // No description under the box (the user asked for none): the empty
        // box's placeholder is an example of the script's syntax.
        nodes.prompt.setAttribute("placeholder", "Speaker 1: Welcome back to the show.\nSpeaker 2: Thanks for having me.\n[pause]\nSpeaker 1: Today…");
        nodes.prompt.setAttribute("rows", "10");
        nodes.prompt.setAttribute("maxlength", "20000");
        nodes.prompt.addEventListener("input", promptChanged);
        nodes.prompt.addEventListener("change", promptChanged);
        body.appendChild(nodes.prompt);
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

    // On a phone the script box grows with its words, so the Prompt stage's
    // body stays its only scroller; side by side the box keeps the
    // stylesheet's height and scrolls inside itself, and only the height this
    // function wrote is taken back there -- one the user set with the box's
    // own handle stays. It is measured at no height (the stylesheet's floor
    // holds it), which shortens the stage for a moment: the stage's own
    // scroll is put back after.
    function fitPrompt() {
        const box = nodes.prompt;
        if (!box || !box.style) return;
        if (state.layout.mode !== "stack") {
            if (box.mcVoiceBoxGrown) {
                box.style.height = "";
                box.mcVoiceBoxGrown = false;
            }
            return;
        }
        const restore = keepScroll([nodes.stage_prompt]);
        box.style.height = "0px";
        const style = styleOf(box);
        const wanted = Math.ceil((Number(box.scrollHeight) || 0)
                                 + pixels(style.borderTopWidth) + pixels(style.borderBottomWidth));
        box.style.height = wanted + "px";
        box.mcVoiceBoxGrown = true;
        restore();
    }

    function promptChanged() {
        state.prompt = nodes.prompt.value || "";
        nodes.summary.textContent = summarize(state.prompt);
        fitPrompt();
        state.dirty.prompt = true;
        if (timers.prompt) window.clearTimeout(timers.prompt);
        timers.prompt = window.setTimeout(function () {
            timers.prompt = 0;
            savePrompt().catch(report);
        }, SAVE_DEBOUNCE_MS);
        renderStatus();
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
        fitPrompt();
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
        // Both lists scroll inside the stage (on a phone, the stage's body
        // does); a star pressed far down one must not throw it back to its top.
        const restore = keepScroll([nodes.history, nodes.favourites, nodes.stage_prompt]);
        clear(nodes.history);
        clear(nodes.favourites);
        const history = state.prompts.history || [];
        const favourites = state.prompts.favourites || [];
        if (!history.length) nodes.history.appendChild(el("li", "mc-voice-box-empty", "No prompts yet."));
        history.forEach(function (entry) { nodes.history.appendChild(promptEntry(entry, false)); });
        if (!favourites.length) nodes.favourites.appendChild(el("li", "mc-voice-box-empty", "Star a prompt to keep it here."));
        favourites.forEach(function (entry) { nodes.favourites.appendChild(promptEntry(entry, true)); });
        restore();
    }

    // -- CONFIGURATION --------------------------------------------------------- //

    // `needs` names the switch a field means nothing without: while it is
    // off the field is disabled, and keeps its value for when it is on again.
    // `list` names the server's list a select offers (engineOptions).
    const FIELDS = [
        {key: "model_id", label: "Model", kind: "select"},
        {key: "card_uuid", label: "Card", kind: "select"},
        {key: "steps", label: "Diffusion steps", kind: "number", min: 1, max: 50, step: 1},
        {key: "cfg_scale", label: "CFG", kind: "number", min: 1, max: 3, step: 0.1},
        {key: "solver", label: "Solver", kind: "select", list: "solvers"},
        {key: "attention", label: "Attention", kind: "select", list: "attention"},
        {key: "seed", label: "Seed", kind: "text", placeholder: "random"},
        {key: "batch", label: "Batch", kind: "select"},
        {key: "max_new_tokens", label: "Max new tokens", kind: "text", placeholder: "automatic"},
        {key: "sampling", label: "Sampling", kind: "switch", wide: true},
        {key: "temperature", label: "Temperature", kind: "number", min: 0.1, max: 2, step: 0.05,
         needs: "sampling"},
        {key: "top_p", label: "Top-p", kind: "number", min: 0.05, max: 1, step: 0.01, needs: "sampling"},
    ];
    // The server's defaults for them (mc_voice_box.CONFIGURATION_DEFAULTS):
    // off is the model's own greedy choice.
    const SAMPLING = {sampling: false, temperature: 0.95, top_p: 0.95};
    // What a render may ask for when the status does not say (an older
    // server, or a part of the status that failed): the lists this page was
    // written against (mc_voice_vibevoice.options), Flash attention 2
    // unavailable since nothing said it was installed. The defaults are a new
    // configuration's; they are also what every render made before Solver,
    // Attention and Batch were choices used -- the model's own solver, SDPA,
    // one take.
    const OPTIONS_FALLBACK = {
        solvers: [{id: "dpmpp_2m", name: "DPM++ 2M", label: "DPM++ 2M (upstream)"},
                  {id: "dpmpp_2m_sde", name: "DPM++ 2M SDE", label: "DPM++ 2M SDE (upstream demo)"},
                  {id: "dpmpp_3m", name: "DPM++ 3M", label: "DPM++ 3M"},
                  {id: "dpmpp_1m", name: "DPM++ 1M", label: "DPM++ 1M"},
                  {id: "dpmpp_1m_sde", name: "DPM++ 1M SDE", label: "DPM++ 1M SDE"}],
        attention: [{id: "sdpa", name: "SDPA", label: "SDPA", available: true, reason: ""},
                    {id: "eager", name: "Eager", label: "Eager", available: true, reason: ""},
                    {id: "flash_attention_2", name: "Flash attention 2",
                     label: "Flash attention 2 (upstream)", available: false, reason: "not installed"}],
        batch_max: 4,
        defaults: {solver: "dpmpp_2m", attention: "sdpa", batch: 1, steps: 12},
    };
    // The fields whose values are picked from those lists.
    const LISTED = ["solver", "attention", "batch"];

    // Render, Install and the one status line with its buttons sit in the
    // stage's header, above the configuration, and stay there while the
    // configuration scrolls under them.
    function buildRenderBlock(head) {
        const actions = el("span", "mc-voice-box-stage-actions");
        nodes.install = button("Install VibeVoice", "Install the VibeVoice runtime and model",
                               installEngine, "mc-voice-box-install");
        show(nodes.install, false);
        nodes.render = button("Render", "Render this pipeline", renderNow, "mc-voice-box-render");
        actions.appendChild(nodes.install);
        actions.appendChild(nodes.render);
        head.appendChild(actions);

        const line = el("div", "mc-voice-box-status-line");
        nodes.status = el("span", "mc-voice-box-status", "");
        nodes.status.setAttribute("role", "status");
        nodes.status.setAttribute("aria-live", "polite");
        line.appendChild(nodes.status);
        // No "×" is ever written as text here. LobeTheme's icon option
        // (replaceIcon) goes through every <span> on the page and, where one's
        // text contains "×", replaces all of its content with a 36 px X: this
        // box's dismiss once made it do that to the whole box -- Cancel, Clear
        // queue and Unload went with it, and a large X sat beside "Ready". So
        // the dismiss draws its cross, and this box is a div.
        const buttons = el("div", "mc-voice-box-status-actions");
        nodes.statusCancel = button("Cancel", "Cancel the running render", cancelRunning,
                                    "mc-voice-box-status-cancel");
        nodes.statusClear = button("Clear queue", "Withdraw every queued render", clearQueue,
                                   "mc-voice-box-status-clear");
        nodes.statusDismiss = iconButton("close", "Dismiss", dismissStatus, "mc-voice-box-status-dismiss");
        nodes.statusUnloads = el("span", "mc-voice-box-status-unloads");
        [nodes.statusCancel, nodes.statusClear, nodes.statusDismiss, nodes.statusUnloads]
            .forEach(function (node) {
                show(node, false);
                buttons.appendChild(node);
            });
        line.appendChild(buttons);
        nodes.statusLine = line;
        head.appendChild(line);
    }

    function buildConfiguration(body, head) {
        buildRenderBlock(head);
        // One row at every width: the select takes the room left, and Save,
        // Save as and Delete are square icons. Unsaved changes are a dot on
        // Save (renderConfigurationBar).
        const bar = el("div", "mc-voice-box-configuration-bar");
        nodes.configurationSelect = el("select", "mc-voice-box-configuration-select");
        nodes.configurationSelect.setAttribute("aria-label", "Configuration");
        nodes.configurationSelect.addEventListener("change", function () {
            chooseConfiguration(nodes.configurationSelect.value).catch(report);
        });
        bar.appendChild(nodes.configurationSelect);
        nodes.configurationSave = iconButton("save", "Save", function () {
            return saveConfiguration(false);
        }, "mc-voice-box-configuration-save");
        nodes.configurationSaveAs = iconButton("saveAs", "Save as a new configuration", function () {
            return saveConfiguration(true);
        }, "mc-voice-box-configuration-save-as");
        nodes.configurationDelete = iconButton("delete", "Delete this configuration", deleteConfiguration,
                                               "mc-voice-box-configuration-delete");
        bar.appendChild(nodes.configurationSave);
        bar.appendChild(nodes.configurationSaveAs);
        bar.appendChild(nodes.configurationDelete);
        body.appendChild(bar);

        // Everything under the header row scrolls inside the stage.
        const form = el("div", "mc-voice-box-configuration-form");
        nodes.configurationForm = form;
        body.appendChild(form);

        const fields = el("div", "mc-voice-box-fields");
        nodes.fields = {};
        nodes.fieldBoxes = {};
        FIELDS.forEach(function (field) {
            if (field.kind === "switch") {
                const toggle = switchButton(field.label, "", function () { fieldChanged(field, toggle); },
                                            "mc-voice-box-field-switch");
                toggle.setAttribute("data-field", field.key);
                nodes.fields[field.key] = toggle;
                const cell = el("div", "mc-voice-box-field mc-voice-box-field-toggle"
                                + (field.wide ? " mc-voice-box-field-wide" : ""));
                cell.appendChild(toggle);
                nodes.fieldBoxes[field.key] = cell;
                fields.appendChild(cell);
                return;
            }
            const wrap = el("label", "mc-voice-box-field" + (field.wide ? " mc-voice-box-field-wide" : ""));
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
            wrap.appendChild(caption);
            wrap.appendChild(input);
            nodes.fields[field.key] = input;
            nodes.fieldBoxes[field.key] = wrap;
            fields.appendChild(wrap);
        });
        form.appendChild(fields);

        // Keep warm is Voice Box's setting rather than a configuration's field
        // (the server keeps it beside the save folder), so it saves at once.
        // Its face says "Keep warm"; its name is the whole sentence. While a
        // save is on its way -- two, after two quick presses -- a status that
        // arrives with the old value does not flip it back
        // (renderConfigurationFields); the last answer decides.
        const warm = el("div", "mc-voice-box-field mc-voice-box-field-toggle");
        nodes.keepWarm = switchButton("Keep warm", "Keep VibeVoice warm between renders", function (on) {
            const settle = function () {
                nodes.keepWarm.mcVoiceBoxSaving -= 1;
                renderConfigurationFields();
            };
            nodes.keepWarm.mcVoiceBoxSaving = (nodes.keepWarm.mcVoiceBoxSaving || 0) + 1;
            return saveSettings({keep_warm: on}).then(settle, function (error) {
                settle();
                throw error;
            });
        }, "mc-voice-box-keep-warm");
        warm.appendChild(nodes.keepWarm);
        form.appendChild(warm);

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
        form.appendChild(speakers);
    }

    function fieldChanged(field, input) {
        const working = ensureWorking();
        let value;
        if (field.kind === "switch") {
            value = switchOn(input);
        } else if (field.kind === "number") {
            value = input.value === "" ? null : Number(input.value);
            if (value !== null && !isFinite(value)) value = null;
        } else if (field.key === "seed" || field.key === "max_new_tokens") {
            const text = String(input.value || "").trim();
            value = text === "" ? null : Number(text);
            if (value !== null && !isFinite(value)) value = null;
        } else if (field.key === "batch") {
            // A count of takes, held as the number it is.
            value = Number(input.value) || 1;
        } else {
            value = input.value;
        }
        if (working[field.key] === value) return;
        working[field.key] = value;
        markConfigurationDirty();
        if (field.kind === "switch") syncNeeds();
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

    // What a render may ask for -- its solvers, attentions, most takes and
    // their defaults -- as the server lists them (status.engine.options), a
    // part it did not send taken from OPTIONS_FALLBACK. Whatever the lists
    // hold is what the selects offer.
    function engineOptions() {
        const engine = (state.status && state.status.engine) || {};
        const sent = engine.options && typeof engine.options === "object" ? engine.options : {};
        const listed = function (key) {
            const found = (Array.isArray(sent[key]) ? sent[key] : []).filter(function (option) {
                return !!option && typeof option.id === "string" && !!option.id;
            });
            return found.length ? found : OPTIONS_FALLBACK[key];
        };
        const given = sent.defaults && typeof sent.defaults === "object" ? sent.defaults : {};
        const defaults = {};
        Object.keys(OPTIONS_FALLBACK.defaults).forEach(function (key) {
            defaults[key] = pick(given[key], OPTIONS_FALLBACK.defaults[key]);
        });
        const most = Math.floor(Number(sent.batch_max));
        return {solvers: listed("solvers"), attention: listed("attention"),
                batch_max: most >= 1 ? most : OPTIONS_FALLBACK.batch_max, defaults: defaults};
    }

    // An option as its select shows it: its label; one the server lists but
    // cannot run is disabled, with the reason after its label.
    function choiceOf(option) {
        const label = String(option.label || option.name || option.id);
        if (option.available !== false) return {value: option.id, label: label};
        return {value: option.id, label: label + (option.reason ? " — " + option.reason : ""),
                disabled: true};
    }

    function cardLabel(card) {
        const roles = [];
        if (card.image_card) roles.push("image model's card");
        if (card.wangp_card) roles.push("WanGP's card");
        return (card.name || card.uuid || "card") + (roles.length ? " — " + roles.join(", ") : "");
    }

    // The options are replaced only when they change: the fields are drawn
    // again on every poll, each second while a render runs, and a list
    // rebuilt under an open select can shut it on the user. A disabled option
    // is still selected when it is the value: that is how a configuration
    // holding what this machine cannot run shows it.
    function fillSelect(select, options, value) {
        const key = JSON.stringify(options.map(function (option) {
            return [option.value, option.label, !!option.disabled];
        }));
        if (select.mcVoiceBoxOptions !== key) {
            clear(select);
            options.forEach(function (option) {
                const node = el("option", "", option.label);
                node.value = option.value;
                node.setAttribute("value", option.value);
                if (option.disabled) {
                    node.disabled = true;
                    node.setAttribute("disabled", "");
                }
                select.appendChild(node);
            });
            select.mcVoiceBoxOptions = key;
        }
        if (select.value !== value) select.value = value;
    }

    // A new configuration starts from Voice Box's settings (the card and model
    // chosen last), the engine's (its steps, CFG, seed and token cap) and the
    // server's defaults for the rest (a solver, an attention, one take, and
    // twelve steps where the engine names none).
    function defaultConfiguration() {
        const status = state.status || {};
        const settings = status.settings || {};
        const engineSettings = status.engine_settings || {};
        const cards = status.cards || [];
        const defaults = engineOptions().defaults;
        return {
            id: "",
            name: "",
            model_id: settings.model_id || engineSettings.model_id || engineModels()[0].id,
            card_uuid: settings.card_uuid || engineSettings.card_uuid
                || (cards[0] ? cards[0].uuid : ""),
            steps: pick(engineSettings.steps, defaults.steps),
            cfg_scale: pick(engineSettings.cfg_scale, 1.3),
            solver: defaults.solver,
            attention: defaults.attention,
            seed: pick(engineSettings.seed, null),
            batch: defaults.batch,
            max_new_tokens: pick(engineSettings.max_new_tokens, null),
            sampling: SAMPLING.sampling,
            temperature: SAMPLING.temperature,
            top_p: SAMPLING.top_p,
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
        renderStatus();
    }

    function chooseConfiguration(id) {
        state.configurationId = configurationById(id) ? id : "";
        loadWorking();
        renderConfiguration();
        renderSamples();
        renderStatus();
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
            // An unsaved configuration a render's settings started is named
            // after that render until somebody names it otherwise.
            const suggested = !state.configurationId && working.name
                ? working.name : "Configuration " + (state.configurations.length + 1);
            name = ask("Name this configuration", suggested);
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
                renderStatus();
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
                renderStatus();
            });
    }

    function renderConfigurationBar() {
        if (!nodes.configurationSelect) return;
        const options = state.configurations.map(function (found) {
            return {value: found.id, label: found.name || "Untitled"};
        });
        let unsaved = state.configurations.length ? "(unsaved)" : "(no configuration yet)";
        if (!state.configurationId && state.working && state.working.name) {
            unsaved = "(unsaved) " + state.working.name;
        }
        options.unshift({value: "", label: unsaved});
        fillSelect(nodes.configurationSelect, options, state.configurationId || "");
        // Unsaved changes: a dot of the accent on Save (the stylesheet draws
        // it from data-dirty), and its name says so.
        const dirty = !!state.dirty.configuration;
        setIcon(nodes.configurationSave, "save", dirty ? "Save — unsaved changes" : "Save");
        nodes.configurationSave.setAttribute("data-dirty", dirty ? "true" : "false");
        nodes.configurationDelete.disabled = !state.configurationId;
    }

    function renderConfigurationFields() {
        if (!nodes.fields) return;
        const working = ensureWorking();
        const cards = (state.status && state.status.cards) || [];
        const options = engineOptions();
        FIELDS.forEach(function (field) {
            const input = nodes.fields[field.key];
            if (field.list) {
                // What the configuration holds stays selected even where it
                // cannot run: Render is then refused, with the server's reason.
                fillSelect(input, options[field.list].map(choiceOf), String(working[field.key] || ""));
                return;
            }
            if (field.key === "batch") {
                const counts = [];
                for (let count = 1; count <= options.batch_max; count += 1) {
                    counts.push({value: String(count), label: String(count)});
                }
                fillSelect(input, counts, String(working.batch || ""));
                return;
            }
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
            if (field.kind === "switch") {
                setSwitch(input, working[field.key] === true);
                return;
            }
            if (focused(input)) return;
            const value = working[field.key];
            const text = value === null || value === undefined ? "" : String(value);
            if (input.value !== text) input.value = text;
        });
        syncNeeds();
        if (nodes.keepWarm && !nodes.keepWarm.mcVoiceBoxSaving) {
            const settings = (state.status && state.status.settings) || {};
            setSwitch(nodes.keepWarm, settings.keep_warm !== false);
        }
    }

    // Temperature and Top-p while Sampling is off: greyed and not editable,
    // their values kept in the fields and in the configuration.
    function syncNeeds() {
        if (!nodes.fields) return;
        const working = ensureWorking();
        FIELDS.forEach(function (field) {
            if (!field.needs) return;
            const off = working[field.needs] !== true;
            nodes.fields[field.key].disabled = off;
            nodes.fieldBoxes[field.key].setAttribute("data-disabled", off ? "true" : "false");
        });
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

    // The stage is its title, the lanes (scrolling inside the stage) and the
    // sentence an empty pipeline shows instead of them.
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
        // "sampling · temperature 0.95 · top-p 0.95", for a render that
        // sampled; a greedy one (or one made before the choice) says nothing.
        if (render.sampling === true) {
            parts.push("sampling");
            if (typeof render.temperature === "number") parts.push("temperature " + render.temperature);
            if (typeof render.top_p === "number") parts.push("top-p " + render.top_p);
        }
        if (Array.isArray(render.speakers) && render.speakers.length) {
            parts.push(render.speakers.map(function (speaker) {
                return "S" + speaker.n + " " + (speaker.title || speaker.sample_id || "?");
            }).join(", "));
        }
        if (output.seconds) parts.push(seconds(output.seconds));
        if (render.card) parts.push(render.card);
        return parts.join(" · ");
    }

    // When a render was made, the short way: "30 Sep 14:05", with the year
    // when it is not this one. `created` is the server's seconds since 1970.
    function when(created) {
        const stamp = Number(created);
        if (!isFinite(stamp) || stamp <= 0) return "";
        const date = new Date(stamp * 1000);
        if (isNaN(date.getTime())) return "";
        const two = function (number) { return (number < 10 ? "0" : "") + number; };
        let day = date.getDate() + " " + MONTHS[date.getMonth()];
        if (date.getFullYear() !== new Date().getFullYear()) day += " " + date.getFullYear();
        return day + " " + two(date.getHours()) + ":" + two(date.getMinutes());
    }

    function isoOf(created) {
        const stamp = Number(created);
        if (!isFinite(stamp) || stamp <= 0) return "";
        try {
            return new Date(stamp * 1000).toISOString();
        } catch (error) {
            return "";
        }
    }

    // The seed a render used, when it was recorded: every render records it
    // now (drawn when the configuration left it blank); one made before that
    // with a blank seed has none.
    function seedOf(output) {
        const render = (output && output.render) || {};
        const seed = render.seed;
        if (seed === null || seed === undefined || seed === "") return null;
        const number = Number(seed);
        return isFinite(number) ? number : null;
    }

    function selected(lane) {
        return !!lane && !!lane.output && state.selectedOutput === lane.output.id;
    }

    // The playhead is the active player's, selected or compact: it shows
    // where that lane plays or was paused, and on no other lane.
    function drawLane(lane) {
        const audio = lane.audio;
        const active = isActivePlayer("output", lane.output.id);
        const total = Number(audio.duration) || Number(lane.output.seconds) || 0;
        const head = active && total ? Math.min(1, (Number(audio.currentTime) || 0) / total) : -1;
        draw(lane.wave, lane.output.peaks || [], {head: head});
        setTouch(lane.wave, active);
        showPlaying(lane.controls, playing(audio));
    }

    // A lane that is not the selected one shows the tint of its speakers'
    // samples, its waveform at full size and one line: when it was made and
    // its name (a double click renames it). Everything else -- the transport
    // and the other buttons, the metadata, the infotext with Copy, Use seed
    // and Reuse settings -- is in `details`, shown while the lane is
    // selected. Lanes are kept rather than rebuilt (renderOutputs), so a
    // playing one plays on.
    function laneNode(output) {
        const row = el("li", "mc-voice-box-lane");
        row.setAttribute("data-id", output.id);
        row.setAttribute("tabindex", "0");
        row.setAttribute("aria-expanded", "false");
        const edge = edgeOf(speakerSamples(output));
        if (edge) row.appendChild(edge);
        const head = el("div", "mc-voice-box-lane-head");
        const date = el("time", "mc-voice-box-lane-date", when(output.created));
        const iso = isoOf(output.created);
        if (iso) date.setAttribute("datetime", iso);
        head.appendChild(date);
        const name = el("span", "mc-voice-box-lane-name", output.name || "Untitled");
        name.setAttribute("title", "Double-click to rename");
        name.addEventListener("dblclick", function () {
            renameInline(name, lane.output.name || "", function (text) {
                return request("outputs/rename", ROUTES.outputRename,
                               {body: {id: lane.output.id, name: text}, queue: true})
                    .then(refreshOutputs);
            });
        });
        head.appendChild(name);
        row.appendChild(head);
        const wave = canvasOf("mc-voice-box-lane-wave");
        row.appendChild(wave);
        const audio = player("mc-voice-box-lane-audio");
        audio.loop = !!output.loop;
        row.appendChild(audio);

        const details = el("div", "mc-voice-box-lane-details");
        const meta = el("div", "mc-voice-box-lane-meta", metadata(output));
        details.appendChild(meta);
        const actions = el("div", "mc-voice-box-row mc-voice-box-lane-actions");
        const controls = transport("mc-voice-box-lane", {
            play: function () { return toggleLane(lane); },
            start: function () { return playLaneFromStart(lane); },
            stop: function () { stopLane(lane); },
        });
        const loop = iconButton("loop", "Loop", function () { return toggleLoop(lane); },
                                "mc-voice-box-lane-loop");
        pressed(loop, !!output.loop);
        const trim = button("Trim to sample", "Open " + (output.name || "this render") + " in the trimmer",
                            function () { return trimOutput(lane); }, "mc-voice-box-lane-trim");
        const save = button("Save", "Save " + (output.name || "this render") + " to the folder",
                            function () { return saveOutput(lane); }, "mc-voice-box-lane-save");
        const download = button("Download", "Download " + (output.name || "this render"),
                                function () { return downloadOutput(lane); }, "mc-voice-box-lane-download");
        const remove = button("Delete", "Delete " + (output.name || "this render"),
                              function () { return deleteOutput(lane); }, "mc-voice-box-lane-delete");
        [controls.box, loop, trim, save, download, remove].forEach(function (node) { actions.appendChild(node); });
        details.appendChild(actions);
        const info = el("div", "mc-voice-box-lane-info");
        const infotext = el("div", "mc-voice-box-infotext", "");
        info.appendChild(infotext);
        const infoActions = el("div", "mc-voice-box-row mc-voice-box-lane-info-actions");
        const copy = button("Copy", "Copy the infotext", function () {
            return copyInfotext(lane);
        }, "mc-voice-box-lane-copy");
        const seed = button("Use seed", "Put this render's seed in the configuration", function () {
            return useSeed(lane.output);
        }, "mc-voice-box-lane-seed");
        const reuse = button("Reuse settings", "Load this render's prompt and settings", function () {
            return reuseSettings(lane.output);
        }, "mc-voice-box-lane-reuse");
        [copy, seed, reuse].forEach(function (node) { infoActions.appendChild(node); });
        info.appendChild(infoActions);
        details.appendChild(info);
        row.appendChild(details);

        const lane = {output: output, row: row, name: name, date: date, meta: meta, wave: wave,
                      audio: audio, controls: controls, loop: loop, details: details,
                      infotext: infotext, copy: copy, seed: seed, reuse: reuse};
        row.mcVoiceBoxLane = lane;
        wireLane(lane);
        audio.addEventListener("timeupdate", function () { drawLane(lane); });
        audio.addEventListener("play", function () { drawLane(lane); });
        audio.addEventListener("pause", function () { drawLane(lane); });
        audio.addEventListener("ended", function () { drawLane(lane); });
        fillLane(lane);
        return row;
    }

    // The parts of a lane that follow its record. The infotext is written only
    // when it changed, so text somebody is selecting in it survives a poll.
    function fillLane(lane) {
        const text = String(lane.output.infotext || "");
        if (lane.infotext.textContent !== text) lane.infotext.textContent = text;
        show(lane.infotext, !!text);
        show(lane.copy, !!text);
        const seed = seedOf(lane.output);
        lane.seed.disabled = seed === null;
        lane.seed.setAttribute("aria-disabled", seed === null ? "true" : "false");
        lane.seed.setAttribute("title", seed === null
            ? "This render was made before seeds were recorded."
            : "Put seed " + seed + " in the configuration");
    }

    // A press on a lane that is neither a drag nor on one of its controls
    // selects it, and so do Enter and Space on the lane itself. On the active
    // player's waveform a press or a drag seeks as well; on any other lane's
    // it is only the press that selects.
    function wireLane(lane) {
        const row = lane.row;
        const press = {down: false, x: 0, y: 0};
        row.addEventListener("pointerdown", function (event) {
            press.down = true;
            press.x = Number(event && event.clientX) || 0;
            press.y = Number(event && event.clientY) || 0;
        });
        row.addEventListener("pointercancel", function () { press.down = false; });
        row.addEventListener("click", function (event) {
            const moved = press.down && Math.max(
                Math.abs((Number(event && event.clientX) || 0) - press.x),
                Math.abs((Number(event && event.clientY) || 0) - press.y)) > DRAG_PX;
            press.down = false;
            if (moved || fromControl(event && event.target, row)) return;
            selectOutput(lane.output.id);
        });
        row.addEventListener("keydown", function (event) {
            if (!event || event.target !== row) return;
            if (event.key !== "Enter" && event.key !== " " && event.key !== "Spacebar") return;
            if (typeof event.preventDefault === "function") event.preventDefault();
            selectOutput(lane.output.id);
        });
        wireScrub(lane.wave, function () { return isActivePlayer("output", lane.output.id); },
                  function (fraction) {
                      seekAudio(lane.audio, fraction, lane.output.seconds);
                      drawLane(lane);
                  });
    }

    // A button, a field -- the rename box above all -- or a link inside the
    // lane: their presses are theirs, not the lane's.
    function fromControl(target, row) {
        for (let at = target; at && at !== row; at = at.parentNode) {
            const tag = String(at.tagName || "").toUpperCase();
            if (tag === "BUTTON" || tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA"
                || tag === "A" || tag === "LABEL") {
                return true;
            }
        }
        return false;
    }

    function selectOutput(id) {
        if (!id || state.selectedOutput === id) return;
        state.selectedOutput = id;
        syncExpanded();
        const row = nodes.lanes[id];
        if (row && row.mcVoiceBoxLane) revealLane(row.mcVoiceBoxLane);
    }

    // Every lane shown open or compact as the selection says.
    function syncExpanded() {
        Object.keys(nodes.lanes).forEach(function (key) {
            const lane = nodes.lanes[key] && nodes.lanes[key].mcVoiceBoxLane;
            if (!lane) return;
            if ((lane.row.getAttribute("aria-expanded") === "true") !== selected(lane)) expandLane(lane);
        });
    }

    // No lane selected: the open one goes back to its compact line. Nothing
    // else changes -- a lane that is playing plays on, still the active
    // player, its playhead on its compact waveform -- and a poll keeps it
    // closed, because the selection is what a poll redraws from.
    function deselectOutput() {
        if (!state.selectedOutput) return;
        state.selectedOutput = "";
        syncExpanded();
    }

    // -- putting the selected lane away: a tap outside it, or Escape --------- //

    const page = {press: null};

    // Where the last press went down, to tell a tap from a drag: a click that
    // ends a drag -- a scrub, a text selection -- is not a tap.
    function notePress(event) {
        page.press = event ? {x: Number(event.clientX) || 0, y: Number(event.clientY) || 0} : null;
    }

    // Only while the tab is on screen: a click in another of Forge's tabs is
    // somewhere the lane could not be.
    function onScreen() {
        try {
            return !!nodes.root && Number(nodes.root.getBoundingClientRect().width) > 0;
        } catch (error) {
            return false;
        }
    }

    // A click outside the selected lane's box -- its waveform, buttons,
    // metadata and infotext are inside -- puts it away. A click is what a tap
    // makes and a scroll or a card swipe does not, which is why this listens
    // for it rather than for pointerdown. A click on another lane is that
    // lane's to select, which it has done by the time this hears it. The
    // page's own synthetic clicks (a download's link) are not taps.
    function onPageClick(event) {
        if (!event || !state.selectedOutput) return;
        const press = page.press;
        page.press = null;
        if (event.isTrusted === false || !onScreen()) return;
        if (press && Number(event.detail) !== 0 && Math.max(
            Math.abs((Number(event.clientX) || 0) - press.x),
            Math.abs((Number(event.clientY) || 0) - press.y)) > DRAG_PX) {
            return;
        }
        const row = nodes.lanes[state.selectedOutput];
        if (!row) return;
        // The path the click took, as it was when it happened: a handler on
        // the way may have rebuilt what was pressed.
        const path = typeof event.composedPath === "function" ? event.composedPath() : null;
        const inside = path && path.length ? path.indexOf(row) !== -1
            : !!event.target && typeof row.contains === "function" && row.contains(event.target);
        if (!inside) deselectOutput();
    }

    // Escape puts the selected lane away too, unless it is being said to a
    // field (a rename being given up) or to something else on the page that
    // has the focus.
    function onPageKey(event) {
        if (!event || event.key !== "Escape" || !state.selectedOutput || !onScreen()) return;
        if (event.defaultPrevented) return;
        const target = event.target;
        const tag = String((target && target.tagName) || "").toUpperCase();
        if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
        // Focus somewhere else on the page -- still in it -- keeps its Escape.
        const active = document.activeElement;
        const elsewhere = !!active && active !== document.body && typeof document.contains === "function"
            && document.contains(active) && !!nodes.root && !nodes.root.contains(active);
        if (elsewhere) return;
        deselectOutput();
    }

    function expandLane(lane) {
        const on = selected(lane);
        lane.row.setAttribute("aria-expanded", on ? "true" : "false");
        show(lane.details, on);
        drawLane(lane);
    }

    // Scrolls what holds the lanes -- the list side by side, the Outputs
    // stage's body on a phone -- and nothing around it, so a phone is never
    // carried to another stage, until a lane is in view: all of it where it
    // fits, its top where it does not.
    function revealLane(lane) {
        const scroller = state.layout.mode === "stack" ? nodes.stage_outputs : nodes.lanesList;
        if (!scroller || !lane || !lane.row) return;
        const view = Number(scroller.clientHeight) || 0;
        if (!view) return;
        const at = Number(scroller.scrollTop) || 0;
        let top = 0;
        let height = 0;
        try {
            const box = scroller.getBoundingClientRect();
            const row = lane.row.getBoundingClientRect();
            top = (Number(row.top) || 0) - (Number(box.top) || 0) - (Number(scroller.clientTop) || 0) + at;
            height = Number(row.height) || 0;
        } catch (error) {
            return;
        }
        if (top < at) scroller.scrollTop = top;
        else if (top + height > at + view) scroller.scrollTop = Math.min(top, top + height - view);
    }

    // Lanes are kept, not rebuilt: a playing <audio> would stop if its element
    // were replaced under it. New outputs go on top (newest first); rows whose
    // output has gone are removed; the rest are updated in place, the selected
    // one included, which is how a selection survives a poll.
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
            if (isActivePlayer("output", id)) state.activePlayer = null;
            forget("output:" + id);
            if (row.parentNode) row.parentNode.removeChild(row);
            delete nodes.lanes[id];
        });
        for (let index = ordered.length - 1; index >= 0; index -= 1) {
            const output = ordered[index];
            const row = nodes.lanes[output.id];
            let lane = null;
            if (row) {
                lane = row.mcVoiceBoxLane;
                lane.output = output;
                lane.name.textContent = output.name || "Untitled";
                lane.date.textContent = when(output.created);
                lane.meta.textContent = metadata(output);
                lane.audio.loop = !!output.loop;
                pressed(lane.loop, !!output.loop);
                fillLane(lane);
            } else {
                const made = laneNode(output);
                nodes.lanes[output.id] = made;
                if (nodes.lanesList.firstChild) nodes.lanesList.insertBefore(made, nodes.lanesList.firstChild);
                else nodes.lanesList.appendChild(made);
                lane = made.mcVoiceBoxLane;
            }
            // Shown compact or open (its details' visibility is decided here,
            // for a new lane as for an old one), and drawn once the row is in
            // the page, at the width it has there.
            expandLane(lane);
        }
        // An empty list takes no room, so the sentence saying so sits under
        // the header rather than at the foot of the stage.
        show(nodes.lanesList, ordered.length > 0);
        show(nodes.lanesEmpty, !ordered.length);
        const reveal = state.revealOutput ? nodes.lanes[state.revealOutput] : null;
        if (reveal && reveal.mcVoiceBoxLane) {
            state.revealOutput = "";
            revealLane(reveal.mcVoiceBoxLane);
        }
    }

    function clearLanes() {
        Object.keys(nodes.lanes).forEach(function (id) {
            const row = nodes.lanes[id];
            try { row.mcVoiceBoxLane.audio.pause(); } catch (error) { /* ignore */ }
            if (isActivePlayer("output", id)) state.activePlayer = null;
            forget("output:" + id);
            if (row.parentNode) row.parentNode.removeChild(row);
        });
        nodes.lanes = {};
    }

    function laneAddress(lane) {
        return audioUrl("output:" + lane.output.id,
                        ROUTES.outputAudio + "?id=" + encodeURIComponent(lane.output.id));
    }

    // Play pauses the lane while it plays; otherwise it plays on from where
    // it was paused (from the start once it has run to its end).
    function toggleLane(lane) {
        if (playing(lane.audio)) {
            lane.audio.pause();
            return Promise.resolve();
        }
        return laneAddress(lane)
            .then(function (address) {
                setActivePlayer("output", lane.output.id);
                return startPlayback(lane.audio, address);
            })
            .then(function () { drawLane(lane); });
    }

    function playLaneFromStart(lane) {
        return laneAddress(lane)
            .then(function (address) {
                setActivePlayer("output", lane.output.id);
                return startPlayback(lane.audio, address, 0);
            })
            .then(function () { drawLane(lane); });
    }

    function stopLane(lane) {
        stopAudio(lane.audio, 0);
        if (isActivePlayer("output", lane.output.id)) setActivePlayer(null);
        else drawLane(lane);
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

    // The browser decodes an MP3 exactly as it decodes a WAV, so a render in
    // either format opens in the trimmer the same way.
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

    // The file's own format names the download: MP3 now; WAV for a render made
    // before MP3, or on a machine whose Forge cannot encode it. A record
    // without the field is told by the bytes' type.
    function extensionOf(output, blob) {
        const format = String((output && output.format) || "").toLowerCase().replace(/[^a-z0-9]/g, "");
        if (format) return format;
        const type = String((blob && blob.type) || "").toLowerCase();
        return /mpeg|mp3/.test(type) ? "mp3" : "wav";
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
                const filename = safeName(lane.output.name) + "." + extensionOf(lane.output, blob);
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

    // -- a render's fingerprint: its infotext, its seed, its settings ---------- //

    // The clipboard can refuse (a page that is not a secure context, a denied
    // permission); the text is then selected, for the keyboard's copy.
    function copyInfotext(lane) {
        const text = String(lane.output.infotext || "");
        if (!text) return Promise.resolve();
        const fallback = function () {
            selectText(lane.infotext);
            say("The browser would not copy it, so the infotext is selected: copy it from there.",
                "warn");
        };
        const clipboard = typeof navigator !== "undefined" && navigator ? navigator.clipboard : null;
        if (!clipboard || typeof clipboard.writeText !== "function") {
            fallback();
            return Promise.resolve();
        }
        return Promise.resolve().then(function () {
            return clipboard.writeText(text);
        }).then(function () {
            say("Copied the infotext of “" + (lane.output.name || "this render") + "”.", "info");
        }, fallback);
    }

    function selectText(node) {
        try {
            const selection = typeof window.getSelection === "function" ? window.getSelection() : null;
            if (!selection || !node) return false;
            if (typeof selection.selectAllChildren === "function") {
                selection.selectAllChildren(node);
                return true;
            }
            if (typeof document.createRange === "function" && typeof selection.addRange === "function") {
                const range = document.createRange();
                range.selectNodeContents(node);
                if (typeof selection.removeAllRanges === "function") selection.removeAllRanges();
                selection.addRange(range);
                return true;
            }
        } catch (error) { /* nothing to select with */ }
        return false;
    }

    function useSeed(output) {
        const seed = seedOf(output);
        if (seed === null) return;
        const working = ensureWorking();
        if (working.seed !== seed) {
            working.seed = seed;
            markConfigurationDirty();
        }
        renderConfigurationFields();
        say("Seed " + seed + " is in the configuration"
            + (state.dirty.configuration ? "; save it to keep it." : "."), "info");
    }

    const REUSED = ["model_id", "card_uuid", "steps", "cfg_scale", "solver", "attention", "seed", "batch",
                    "max_new_tokens"];
    const SAMPLED = ["sampling", "temperature", "top_p"];

    // The configuration a render used: the one it recorded, or -- for a render
    // made before configurations were recorded -- rebuilt from the fields it
    // did record. Sampling, Temperature, Top-p, Solver, Attention and Batch
    // are there only when the render recorded them.
    function renderedConfiguration(output) {
        const render = (output && output.render) || {};
        const recorded = render.configuration && typeof render.configuration === "object"
            ? render.configuration : null;
        if (recorded) {
            const found = {id: String(recorded.id || ""),
                           speakers: Object.assign({}, recorded.speakers || {})};
            REUSED.concat(SAMPLED).forEach(function (key) { found[key] = recorded[key]; });
            return found;
        }
        const speakers = {};
        (Array.isArray(render.speakers) ? render.speakers : []).forEach(function (speaker) {
            if (speaker && speaker.n !== undefined && speaker.n !== null && speaker.sample_id) {
                speakers[String(speaker.n)] = speaker.sample_id;
            }
        });
        return {id: "", model_id: render.model_id, card_uuid: render.card, steps: render.steps,
                cfg_scale: render.cfg_scale, seed: render.seed, max_new_tokens: render.max_new_tokens,
                sampling: render.sampling, temperature: render.temperature, top_p: render.top_p,
                speakers: speakers};
    }

    function sameConfiguration(working, saved) {
        if (!working || !saved) return false;
        const same = REUSED.concat(SAMPLED).every(function (key) {
            return pick(working[key], null) === pick(saved[key], null);
        });
        const a = working.speakers || {};
        const b = saved.speakers || {};
        const keys = Object.keys(a).concat(Object.keys(b));
        return same && keys.every(function (key) { return (a[key] || "") === (b[key] || ""); });
    }

    // The prompt goes into the box as if typed (so the pipeline keeps it the
    // usual way), and the configuration into the editor as unsaved edits: over
    // the configuration it came from while that still exists, otherwise over a
    // new, unsaved one named after the render. A speaker whose sample has left
    // the library is left empty, and the status line says so.
    function reuseSettings(output) {
        const render = (output && output.render) || {};
        const values = renderedConfiguration(output);
        if (typeof render.prompt === "string" && nodes.prompt) {
            state.prompt = render.prompt;
            nodes.prompt.value = render.prompt;
            promptChanged();
        }
        const existing = values.id ? configurationById(values.id) : null;
        const chosen = chooseConfiguration(existing ? existing.id : "");
        const working = ensureWorking();
        if (!existing) working.name = (output && output.name) || "";
        REUSED.forEach(function (key) {
            const value = values[key];
            if (value === undefined) return;
            // A recorded model or card of "" is the configuration's way of
            // saying "the default": what the editor holds already stays.
            if (value === "" && (key === "model_id" || key === "card_uuid")) return;
            working[key] = value === "" ? null : value;
        });
        // A render made before Solver, Attention and Batch were choices
        // recorded none of them, and was made with the model's own solver,
        // SDPA and one take: the fallback's defaults, which it puts back.
        LISTED.forEach(function (key) {
            if (values[key] === undefined || values[key] === null || values[key] === "") {
                working[key] = OPTIONS_FALLBACK.defaults[key];
            }
        });
        // Sampling is on only when the render says it was: one made before
        // the choice existed was greedy. Temperature and Top-p are taken when
        // recorded, and otherwise the editor keeps its own.
        working.sampling = values.sampling === true;
        ["temperature", "top_p"].forEach(function (key) {
            const value = values[key];
            if (typeof value === "number" && isFinite(value)) working[key] = value;
        });
        working.speakers = {};
        const missing = [];
        Object.keys(values.speakers || {}).sort(function (a, b) { return Number(a) - Number(b); })
            .forEach(function (number) {
                const id = values.speakers[number];
                if (!id) return;
                if (sampleById(id)) {
                    working.speakers[String(number)] = id;
                    return;
                }
                const spoken = (Array.isArray(render.speakers) ? render.speakers : []).filter(function (speaker) {
                    return speaker && String(speaker.n) === String(number);
                })[0];
                missing.push("Speaker " + number + "'s sample “" + ((spoken && spoken.title) || id)
                             + "” is no longer in the library.");
            });
        // Against the saved configuration as the editor would hold it, so a
        // field it was saved before is read at its default.
        state.dirty.configuration = !existing || !sameConfiguration(working, copyConfiguration(existing));
        renderConfiguration();
        renderSamples();
        renderStatus();
        const name = "“" + ((output && output.name) || "this render") + "”";
        if (missing.length) say(missing.join(" "), "warn");
        else say("Loaded the prompt and settings of " + name
                 + (state.dirty.configuration ? "; save the configuration to keep them." : "."), "info");
        return chosen;
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
        if (changed) {
            // The pipeline before's renders go with their lanes; drawn again
            // from its list, they would show under this one until its own
            // list arrives.
            clearLanes();
            state.outputs = [];
        }
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
            state.answeredAt = now();
            const finished = before.filter(function (id) {
                return !isLive(jobById(id));
            });
            finished.forEach(noteFinished);
            renderStatus();
            renderConfigurationFields();
            if (finished.length) {
                return Promise.all([refreshOutputs(), refreshPrompts()]).then(function () {
                    return found;
                });
            }
            return found;
        });
    }

    // A render this page started has ended: its output becomes the selected
    // lane, or its failure the status line's until dismissed or the next
    // Render press.
    function noteFinished(id) {
        const job = jobById(id);
        if (!job) return;
        if (job.phase === "done" && job.output_id) {
            state.selectedOutput = job.output_id;
            state.revealOutput = job.output_id;
        } else if (job.phase === "failed") {
            state.lastFailure = {id: job.id, name: job.name || "", warning: job.warning || ""};
        }
    }

    function refreshSamples() {
        return request("samples", ROUTES.samples, {body: {}}).then(function (found) {
            state.samples = listOf(found, "samples");
            renderSamples();
            drawSpeakers();
            renderStatus();
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
        renderStatus();
    }

    // -- the Configuration header: Render, Install and the one status line ----- //

    function renderBlocker() {
        return renderBlockerFor(state.prompt, "Write a script first.");
    }

    // Why `text` cannot be rendered with the page's current pipeline and
    // configuration, or "" when it can. The Render button's reason, and the
    // reason another tab gets for a script of its own (`mcVoiceBox.canRender`).
    function renderBlockerFor(text, emptyMessage) {
        if (!state.booted) return "Voice Box has not loaded.";
        if (!state.status) return "Loading…";
        const engine = state.status.engine || {};
        if (engine.ready === false) {
            const progress = state.status.progress || {};
            if (progress.failed && !progress.running && progress.text) {
                return "Install failed: " + progress.text;
            }
            return engine.message || "VibeVoice is not installed.";
        }
        if (!state.pipelineId) return "No pipeline.";
        const parsed = parseScript(text);
        if (parsed.error) return parsed.error;
        if (!parsed.words) return emptyMessage || "Nothing to speak.";
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

    // The render request for `prompt` as the page would send its own: the
    // pipeline on screen, the saved configuration by id when the screen shows it
    // unchanged, otherwise what the screen shows inline, validated by the
    // server the same way. `name` and `origin` are another tab's to give.
    function renderBody(prompt, name, origin) {
        const working = ensureWorking();
        const unsaved = state.dirty.configuration || !configurationById(state.configurationId);
        const body = {pipeline_id: state.pipelineId, prompt: String(prompt || ""),
                      configuration_id: state.configurationId, name: String(name || "")};
        if (unsaved) body.configuration = Object.assign({}, working);
        if (origin && typeof origin === "object") body.origin = origin;
        return body;
    }

    // A job the server has just taken: shown on the status line at once --
    // queued, or on its card -- and polled at the live rate until it ends.
    function adoptJob(reply) {
        const job = recordOf(reply, "job");
        if (job.id) {
            state.ownJobs[job.id] = true;
            if (!jobById(job.id)) state.jobs = [job].concat(state.jobs);
        }
        renderStatus();
        schedulePoll();
        return job;
    }

    function renderNow() {
        // A press is the answer to the last failure: the line stops saying it.
        state.lastFailure = null;
        const why = renderBlocker();
        if (why) {
            say(why, "warn");
            return Promise.resolve(null);
        }
        return flushSaves().then(function () {
            return request("render", ROUTES.render, {body: renderBody(state.prompt, "", null)});
        }).then(function (reply) {
            // The status line shows the new job at once -- queued, or on its
            // card -- which is the answer to the press. A message in its place
            // would hide the job's Cancel for as long as it held the line.
            adoptJob(reply);
            return refreshStatus();
        });
    }

    // -- another tab's render ----------------------------------------------- //
    //
    // LLM Studio's Send to VibeVoice: a message's words rendered with whatever
    // this page is set up with -- its pipeline, its configuration and the
    // samples in it -- and nothing of the page's own prompt touched. The job is
    // this page's as much as a Render press's: it shows on the status line,
    // its output lands in the pipeline's list, and the poll follows it.
    //
    // Queued behind a render in flight rather than folded into it: two
    // messages sent in a row are two renders, and the second must never be
    // handed the first's job (the in-flight rule's `queue`).
    function renderText(text, options) {
        const settings = options || {};
        const why = renderBlockerFor(text, "Nothing to speak.");
        if (why) return Promise.reject(new Error(why));
        return request("render", ROUTES.render,
                       {body: renderBody(text, settings.name, settings.origin), queue: true})
            .then(function (reply) {
                const job = adoptJob(reply);
                refreshStatus().catch(report);
                return job;
            });
    }

    // Every output whose origin key starts with `prefix`, newest first: how
    // another tab finds its renders again after a reload. Its own request
    // kind, so it is never folded into the page's own outputs read, which
    // asks for one pipeline.
    function outputsFor(prefix) {
        const wanted = String(prefix || "");
        return request("outputs:origin", ROUTES.outputs, {body: {pipeline_id: ""}})
            .then(function (found) {
                return listOf(found, "outputs").filter(function (output) {
                    const origin = output && output.render && output.render.origin;
                    const key = origin && typeof origin.key === "string" ? origin.key : "";
                    return !!key && key.indexOf(wanted) === 0;
                });
            });
    }

    // An output's sound as a blob URL, fetched with the page token the way the
    // lanes fetch theirs, and cached with them.
    function outputAudioUrl(id) {
        const key = "output:" + String(id || "");
        return audioUrl(key, ROUTES.outputAudio + "?id=" + encodeURIComponent(String(id || "")));
    }

    // The install's progress is the status line's first state.
    function installEngine() {
        return request("install", ROUTES.install, {body: {part: "", folder: ""}}).then(function () {
            return refreshStatus();
        });
    }

    function cancelJob(job) {
        return request("jobs/cancel:" + job.id, ROUTES.jobCancel, {body: {id: job.id}})
            .then(refreshStatus);
    }

    function cancelRunning() {
        const job = activeJob();
        return job ? cancelJob(job) : Promise.resolve();
    }

    // Every queued render, on every card, whichever pipeline it came from; the
    // running one is left to Cancel. The answer carries the jobs as they now
    // are, so the line follows it without another request -- and says so by
    // no longer counting a queue, rather than by a message over the running
    // job's Cancel.
    function clearQueue() {
        return request("jobs/clear", ROUTES.jobClear, {body: {}}).then(function (reply) {
            if (reply && Array.isArray(reply.jobs)) {
                state.jobs = reply.jobs;
                state.answeredAt = now();
            }
            renderStatus();
            schedulePoll();
        });
    }

    function unloadCard(card) {
        return request("runtime:" + card.uuid, ROUTES.runtime,
                       {body: {action: "unload", card_uuid: card.uuid}})
            .then(function () {
                say("VibeVoice was unloaded from " + card.name + ".", "info");
                return refreshStatus();
            });
    }

    function dismissStatus() {
        if (state.statusLine === "message") {
            say("", "info");
        } else if (state.statusLine === "failed") {
            state.lastFailure = null;
            renderStatus();
        }
    }

    function now() {
        try {
            if (window.performance && typeof window.performance.now === "function") {
                return window.performance.now();
            }
        } catch (error) { /* no monotonic clock: the wall clock will do for a difference */ }
        return Date.now();
    }

    // "0:42", "12:05", "1:02:09".
    function clockOf(value) {
        const total = Math.max(0, Math.floor(Number(value) || 0));
        const two = function (number) { return (number < 10 ? "0" : "") + number; };
        const hours = Math.floor(total / 3600);
        const minutes = Math.floor((total % 3600) / 60);
        const rest = total % 60;
        return hours ? hours + ":" + two(minutes) + ":" + two(rest) : minutes + ":" + two(rest);
    }

    // The server's count of a job's seconds when it answered, plus the time
    // that has passed here since -- a difference of this page's own clock,
    // never this browser's clock against the server's timestamps.
    function elapsedOf(job) {
        if (!job || job.elapsed === null || job.elapsed === undefined) return null;
        const base = Number(job.elapsed);
        if (!isFinite(base)) return null;
        return Math.max(0, base + Math.max(0, now() - (state.answeredAt || now())) / 1000);
    }

    function startOf(job) {
        const started = job.started === null || job.started === undefined ? NaN : Number(job.started);
        return isFinite(started) ? started : (Number(job.created) || 0);
    }

    // The job holding a card, whichever pipeline it came from: this page's own
    // before another's, then the one that started first.
    function activeJob() {
        let found = null;
        state.jobs.forEach(function (job) {
            if (!job || !ACTIVE_PHASES[job.phase]) return;
            if (!found) {
                found = job;
                return;
            }
            const mine = !!state.ownJobs[job.id];
            if (mine !== !!state.ownJobs[found.id]) {
                if (mine) found = job;
                return;
            }
            if (startOf(job) < startOf(found)) found = job;
        });
        return found;
    }

    // Renders wait per card, whatever pipeline they came from, so the count is
    // of all of them.
    function queuedCount() {
        return state.jobs.filter(function (job) { return job && job.phase === "queued"; }).length;
    }

    // " · 4 takes" for a job that renders more than one, and nothing for one.
    // Words, never a "×": the line is a span, and Lobe replaces any span
    // holding one.
    function takesOf(job) {
        const count = Math.floor(Number(job.batch) || 0);
        return count > 1 ? " · " + count + " takes" : "";
    }

    function jobLine(job, queued) {
        const name = "“" + (job.name || "a render") + "”" + takesOf(job);
        let before;
        if (job.phase === "rendering") {
            before = "Rendering " + name;
            const progress = job.progress || {};
            const sections = Number(progress.sections) || 0;
            if (sections > 1) before += " · section " + (Number(progress.section) || 1) + " of " + sections;
        } else if (job.phase === "loading") {
            before = "Loading VibeVoice · " + name;
        } else {
            before = "Waiting for the card" + (job.reason ? ": " + job.reason : "") + " · " + name;
        }
        const elapsed = elapsedOf(job);
        return {before: elapsed === null ? before : before + " · ",
                clock: elapsed === null ? undefined : clockOf(elapsed),
                after: queued ? " · " + queued + " queued" : ""};
    }

    // One flat record (mc_voice_vibevoice.progress): running, text, fraction,
    // failed. "Installing the runtime — 45 %"; the words alone before a
    // fraction is known.
    function installLine(progress) {
        const words = String(progress.text || "") || "Installing VibeVoice…";
        const fraction = Math.max(0, Math.min(1, Number(progress.fraction) || 0));
        if (!(fraction > 0)) return words;
        return (words.replace(/[\s.…]+$/, "") || "Installing VibeVoice") + " — "
            + Math.round(fraction * 100) + " %";
    }

    function cardKey(uuid) {
        return String(uuid || "").toLowerCase().replace(/[^0-9a-f]/g, "");
    }

    function cardName(uuid) {
        const key = cardKey(uuid);
        const cards = (state.status && state.status.cards) || [];
        const found = cards.filter(function (card) { return key && cardKey(card.uuid) === key; })[0];
        return found ? (found.name || "") : "";
    }

    // Where VibeVoice is up and not rendering: its worker's own report first,
    // then the turn system's warm stays.
    function warmCards() {
        const status = state.status || {};
        const found = [];
        const seen = {};
        const runtime = (status.runtime && status.runtime.cards) || {};
        Object.keys(runtime).forEach(function (key) {
            const card = runtime[key] || {};
            if (!card.running || card.rendering) return;
            const uuid = card.uuid || key;
            seen[cardKey(uuid)] = true;
            found.push({uuid: uuid, name: card.device_name || cardName(uuid) || uuid});
        });
        (status.turns || []).forEach(function (turn) {
            if (!turn || !turn.warm) return;
            const uuid = turn.uuid || "";
            if (!uuid || seen[cardKey(uuid)]) return;
            seen[cardKey(uuid)] = true;
            found.push({uuid: uuid, name: turn.card || cardName(uuid) || uuid});
        });
        return found;
    }

    // What the cards are doing, in full, for the status line's tooltip.
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
        return parts.join(" · ");
    }

    // The first that applies: a fresh message, an install running, a job on a
    // card (with its time and the queue), the queue alone, this page's last
    // render having failed, VibeVoice warm on a card, why Render is disabled,
    // and otherwise Ready.
    function statusLine() {
        if (state.message) {
            return {state: "message", kind: state.messageKind || "info", before: state.message};
        }
        const status = state.status || {};
        const progress = status.progress || {};
        if (progress.running) return {state: "install", kind: "info", before: installLine(progress)};
        const queued = queuedCount();
        const job = activeJob();
        if (job) {
            return Object.assign({state: "job", kind: "info", job: job, queued: queued},
                                 jobLine(job, queued));
        }
        if (queued) return {state: "queued", kind: "info", queued: queued, before: queued + " queued"};
        if (state.lastFailure) {
            return {state: "failed", kind: "error",
                    before: "Last render failed" + (state.lastFailure.warning
                        ? ": " + state.lastFailure.warning : ".")};
        }
        const warm = warmCards();
        if (warm.length) {
            return {state: "warm", kind: "info", warm: warm,
                    before: "VibeVoice warm on " + warm.map(function (card) { return card.name; }).join(" and ")};
        }
        const why = renderBlocker();
        if (why) return {state: "blocked", kind: "info", before: why};
        return {state: "ready", kind: "info", before: "Ready"};
    }

    // The line's words, with the job's clock in a span of its own: the words
    // are rewritten only when they change, so a screen reader hears a new
    // phase or section but not every second the clock ticks.
    function writeStatus(line) {
        const hasClock = line.clock !== undefined;
        const shape = [line.state, line.kind, line.before, hasClock ? "clock" : "", line.after || ""].join("\u0001");
        if (nodes.statusShape !== shape) {
            clear(nodes.status);
            nodes.status.appendChild(document.createTextNode(line.before));
            nodes.statusClock = null;
            if (hasClock) {
                nodes.statusClock = el("span", "mc-voice-box-status-clock", "");
                nodes.statusClock.setAttribute("aria-live", "off");
                nodes.status.appendChild(nodes.statusClock);
            }
            if (line.after) nodes.status.appendChild(document.createTextNode(line.after));
            nodes.statusShape = shape;
        }
        if (nodes.statusClock) nodes.statusClock.textContent = line.clock;
        nodes.status.setAttribute("data-kind", line.kind || "info");
        nodes.status.setAttribute("data-state", line.state);
        const details = state.status ? cardsLine(state.status) : "";
        nodes.status.setAttribute("title", line.before + (hasClock ? line.clock : "") + (line.after || "")
                                  + (details ? "\n" + details : ""));
    }

    // One Unload per card VibeVoice is warm on, rebuilt only when that set
    // changes -- a button replaced under a finger loses the press.
    function syncUnloads(warm) {
        const key = warm.map(function (card) { return card.uuid + "=" + card.name; }).join("|");
        if (nodes.unloadKey === key) return;
        nodes.unloadKey = key;
        clear(nodes.statusUnloads);
        warm.forEach(function (card) {
            nodes.statusUnloads.appendChild(button(warm.length > 1 ? "Unload from " + card.name : "Unload",
                                                   "Unload VibeVoice from " + card.name, function () {
                                                       return unloadCard(card);
                                                   }, "mc-voice-box-unload"));
        });
        show(nodes.statusUnloads, warm.length > 0);
    }

    function renderStatus() {
        if (!nodes.render) return;
        const status = state.status || {};
        const engine = status.engine || {};
        const why = renderBlocker();
        nodes.render.disabled = !!why;
        nodes.render.setAttribute("aria-disabled", why ? "true" : "false");
        nodes.render.setAttribute("title", why || "Render this pipeline");
        const installable = !!state.status && engine.ready === false && engine.supported !== false;
        show(nodes.install, installable);
        if (installable) {
            nodes.install.textContent = "Install VibeVoice"
                + (engine.download_bytes ? " (" + gigabytes(engine.download_bytes) + ")" : "");
        }
        const line = statusLine();
        state.statusLine = line.state;
        writeStatus(line);
        show(nodes.statusCancel, line.state === "job");
        show(nodes.statusClear, (line.state === "job" && line.queued > 0) || line.state === "queued");
        show(nodes.statusDismiss, line.state === "message" || line.state === "failed");
        syncUnloads(line.state === "warm" ? line.warm : []);
        scheduleTick(line);
    }

    // The job's clock moves every second between polls, from what the last
    // answer said; nothing ticks while the tab is hidden or nothing runs.
    function scheduleTick(line) {
        if (timers.tick) {
            window.clearTimeout(timers.tick);
            timers.tick = 0;
        }
        if (!state.booted || document.visibilityState === "hidden") return;
        if (!line || line.state !== "job" || line.clock === undefined) return;
        timers.tick = window.setTimeout(function () {
            timers.tick = 0;
            renderStatus();
        }, TICK_MS);
    }

    // -- fitting the window ---------------------------------------------------- //

    // One fit per animation frame, however many events asked for it.
    function scheduleFit() {
        if (frames.fit) return;
        if (typeof window.requestAnimationFrame !== "function") {
            fit();
            return;
        }
        frames.fit = window.requestAnimationFrame(function () {
            frames.fit = 0;
            fit();
        });
    }

    function pixels(value) {
        const number = parseFloat(value);
        return isFinite(number) ? number : 0;
    }

    function styleOf(node) {
        try {
            return window.getComputedStyle(node) || {};
        } catch (error) {
            return {};
        }
    }

    function nextElement(node) {
        if (node.nextElementSibling !== undefined) return node.nextElementSibling;
        let at = node.nextSibling;
        while (at && at.nodeType !== undefined && at.nodeType !== 1) at = at.nextSibling;
        return at || null;
    }

    // The height of what the document lays out below the root -- Forge's
    // footer, the containers' own bottom padding -- summed level by level up to
    // the body. Heights are summed rather than positions read, so a container
    // stretched to the window, whose emptiness is room the root may take, does
    // not count; a sibling in a row sits beside the root, not below it.
    function belowRoot(root) {
        let total = 0;
        let node = root;
        while (node && node !== document.documentElement) {
            total += pixels(styleOf(node).marginBottom);
            const parent = node.parentElement || node.parentNode;
            if (!parent || parent === document) break;
            const outer = styleOf(parent);
            const display = String(outer.display || "");
            const row = /flex/.test(display) && !/column/.test(String(outer.flexDirection || ""));
            if (!row) {
                const gap = /flex|grid/.test(display) ? pixels(outer.rowGap) : 0;
                for (let sibling = nextElement(node); sibling; sibling = nextElement(sibling)) {
                    const style = styleOf(sibling);
                    if (style.display === "none" || style.position === "absolute"
                        || style.position === "fixed") {
                        continue;
                    }
                    let height = 0;
                    try {
                        height = Number(sibling.getBoundingClientRect().height) || 0;
                    } catch (error) {
                        height = 0;
                    }
                    total += height + pixels(style.marginTop) + pixels(style.marginBottom) + gap;
                }
            }
            total += pixels(outer.paddingBottom) + pixels(outer.borderBottomWidth);
            node = parent;
        }
        return total;
    }

    function viewportHeight() {
        const inner = Number(window.innerHeight);
        if (isFinite(inner) && inner > 0) return inner;
        const page = document.documentElement;
        return Number(page && page.clientHeight) || 0;
    }

    function pageScroll() {
        const y = Number(window.scrollY !== undefined ? window.scrollY : window.pageYOffset);
        return isFinite(y) ? y : 0;
    }

    // How far the page scrolls: what the fit exists to keep at nothing.
    function pageOverflow() {
        const page = document.documentElement;
        if (!page) return 0;
        const over = (Number(page.scrollHeight) || 0) - (Number(page.clientHeight) || 0);
        return isFinite(over) && over > 0 ? Math.ceil(over) : 0;
    }

    function writeHeight(root, wanted) {
        const height = Math.max(MIN_HEIGHT, Math.floor(wanted));
        if (height === state.layout.height) return;
        state.layout.height = height;
        state.layout.writes += 1;
        if (root.style && typeof root.style.setProperty === "function") root.style.setProperty("height", height + "px");
        else if (root.style) root.style.height = height + "px";
    }

    // The root takes the height the window leaves it -- the window's height,
    // less where the root starts on the document and what the document lays
    // out below it -- never less than MIN_HEIGHT, written in pixels on the
    // root itself and only when it changed. A root with no width is a tab
    // that is not on screen: nothing is measured.
    function fit() {
        const root = nodes.root;
        if (!root) return;
        let box = null;
        try {
            box = root.getBoundingClientRect();
        } catch (error) {
            box = null;
        }
        const width = box ? Math.round(Number(box.width) || 0) : 0;
        if (!width) return;
        const mode = width < STACK_BELOW ? "stack" : "columns";
        if (root.getAttribute("data-layout") !== mode) {
            root.setAttribute("data-layout", mode);
            // The stage bar marks the card in view once the cards are laid out.
            if (mode === "stack") scheduleTrack();
        }
        state.layout.mode = mode;
        const top = (Number(box.top) || 0) + pageScroll();
        const raw = Math.floor(viewportHeight() - top - belowRoot(root));
        if (raw !== state.layout.raw) {
            state.layout.raw = raw;
            state.layout.correction = 0;
        }
        writeHeight(root, raw - state.layout.correction);
        // The sum is a model of the page; the page scrolling is the fact.
        // What the model missed is taken off, and remembered while the window
        // keeps this shape, so the height does not swing back and forth.
        const over = pageOverflow();
        if (over > 0 && state.layout.height > MIN_HEIGHT) {
            state.layout.correction += over;
            writeHeight(root, raw - state.layout.correction);
        }
        if (width !== state.layout.width) {
            state.layout.width = width;
            redrawAll();
            fitPrompt();
        }
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
        // For another tab on this page. LLM Studio's Send to VibeVoice reads
        // these five and nothing else: why a text cannot be rendered right
        // now, a render of it with the page's current setup, a job by id (the
        // page's poll keeps `state.jobs` current while one is live), the
        // outputs that carry an origin key, and an output's sound.
        canRender: function (text) { return renderBlockerFor(text, "Nothing to speak."); },
        renderText: renderText,
        jobById: jobById,
        outputsFor: outputsFor,
        outputAudioUrl: outputAudioUrl,
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
        hueOf: hueOf,
        clockOf: clockOf,
        fit: fit,
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
