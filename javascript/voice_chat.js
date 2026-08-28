// Model Chain -- Voice Chat.
//
// The browser half of a feature whose other half is a CPU speech worker on the
// Forge machine. The division is the same one llm_studio.js states and for the
// same reason: Python owns models, persistence, inference and conversation
// state; this file owns the things only a browser can do.
//
//   * the press-and-hold gesture, with pointer capture, so a finger that slides
//     off the button still ends the recording it started;
//   * opening the microphone, and closing it again the moment the utterance is
//     over;
//   * turning what the microphone gave us into one deterministic format --
//     16 kHz mono PCM16 in a RIFF/WAVE container, built in memory;
//   * posting it to this WebUI, and putting the transcript in the composer that
//     already exists;
//   * pressing the Send button that already exists, when the setting says so;
//   * unlocking Web Audio on a real user gesture, and playing a reply back.
//
// Four things about it are deliberate and are easy to get wrong.
//
// The deployment root. Every URL below is built from the root Gradio publishes,
// never from location.origin. Forge can be mounted under --subpath or behind a
// reverse proxy, and "/model-chain/voice/stt" against the origin reaches the
// proxy's root rather than the WebUI. See appRoot().
//
// AudioWorklet, not MediaRecorder. MediaRecorder on Chromium emits WebM/Opus,
// which would push container parsing and codec decoding into the speech worker
// and make this feature depend on whatever media libraries Forge happens to
// ship. A worklet hands over raw float samples, and what leaves this file is
// always the same bytes. Where a worklet cannot be installed we fall back to a
// ScriptProcessorNode, which is deprecated and still gives us PCM -- a worse
// API for the same guarantee, which is a better trade than a codec.
//
// No audio is ever a file. Not a Blob URL, not a download, not a Gradio audio
// component. Samples live in JS arrays until the request resolves, the response
// lives in an ArrayBuffer until it has been decoded, and both are dropped.
//
// The reply this speaks is chosen by Python. This file never reads a transcript
// bubble and never sends reply text anywhere: it is given an opaque one-shot
// token, and it exchanges that token for audio. What the token stands for was
// decided at the instant the reply completed, so editing the thread between then
// and now cannot change what is spoken.
//
// Everything is found by this extension's own element ids, and every feature is
// skipped rather than fatal when an id is missing -- so a page without Voice
// Chat installed is a page where typed Conversation works exactly as it did.

(function () {
    "use strict";

    const IDS = {
        mic: "mc-llm-chat-voice-mic",
        token: "mc-llm-chat-voice-token",
        key: "mc-llm-chat-voice-key",
        message: "mc-llm-chat-message",
        send: "mc-llm-chat-send",
        stop: "mc-llm-chat-stop",
        status: "mc-llm-chat-status",
        chip: "mc-llm-chat-to-voice",
    };

    const ROUTES = {
        status: "model-chain/voice/status",
        stt: "model-chain/voice/stt",
        tts: "model-chain/voice/tts",
        install: "model-chain/voice/install",
    };

    // 250 ms. Shorter than this is a tap, and a tap is somebody finding out what
    // the button does -- transcribing a tenth of a second of room tone and
    // dropping the result in their composer would be a worse answer than a
    // sentence saying how the control works.
    const MIN_HOLD_MS = 250;
    // Sixty seconds, then stop and transcribe what there is. This bounds memory
    // and it bounds surprise: a button held down in a pocket is not a five
    // minute upload.
    const MAX_HOLD_MS = 60000;
    const TARGET_RATE = 16000;
    const STATUS_HOLD_MS = 2600;
    const TOKEN_POLL_MS = 400;
    const SETTINGS_POLL_MS = 1500;

    const MESSAGES = {
        notReady: "Voice Chat is not set up. Install both models in Settings → Voice Chat.",
        insecure: "Microphone access requires HTTPS when Forge is opened from another device.",
        denied: "Microphone permission was denied.",
        tooShort: "Hold to speak.",
        failed: "Voice transcription failed. Your message was not sent.",
        empty: "No speech was detected.",
        speakFailed: "The reply was generated, but Voice could not read it aloud.",
        blocked: "Voice is enabled; tap Voice or the microphone once to allow audio playback.",
        busy: "Wait for the reply or press Stop.",
        unsupported: "This browser cannot open the microphone.",
    };

    function root() {
        const app = typeof gradioApp === "function" ? gradioApp() : document;
        return app || document;
    }

    function byId(id) {
        try {
            return root().querySelector("#" + id);
        } catch (error) {
            return null;
        }
    }

    function clickable(id) {
        const holder = byId(id);
        if (!holder) return null;
        return holder.tagName === "BUTTON" ? holder : holder.querySelector("button");
    }

    function fieldValue(id) {
        const holder = byId(id);
        if (!holder) return "";
        const field = holder.tagName === "TEXTAREA" || holder.tagName === "INPUT"
            ? holder
            : holder.querySelector("textarea, input");
        return field ? (field.value || "") : "";
    }

    // -- where this WebUI actually lives ----------------------------------- //

    // R2-4. Gradio publishes the application root it was mounted at; Forge
    // passes --subpath through to it. Reading it is the difference between a
    // request that reaches the WebUI and one that reaches whatever is at the
    // root of the proxy in front of it.
    function appRoot() {
        let base = "";
        try {
            const config = typeof window !== "undefined" ? window.gradio_config : null;
            if (config && typeof config.root === "string") base = config.root;
        } catch (error) {
            base = "";
        }
        if (!base) {
            try {
                // A page served from /subpath/ has that as its directory. The
                // last resort, and still better than assuming the origin root.
                const path = window.location && window.location.pathname || "/";
                base = path.replace(/[^/]*$/, "");
            } catch (error) {
                base = "/";
            }
        }
        // An absolute root is honoured as-is; a path is joined to the origin by
        // the browser. Either way exactly one slash goes in the middle.
        return base.replace(/\/+$/, "");
    }

    function url(route) {
        return appRoot() + "/" + route;
    }

    function pageKey() {
        const fromPanel = fieldValue(IDS.key);
        if (fromPanel) return fromPanel;
        const holder = document.querySelector("[data-mc-voice-key]");
        return holder ? (holder.getAttribute("data-mc-voice-key") || "") : "";
    }

    function headers(extra) {
        const found = {"X-Model-Chain-Voice": pageKey()};
        if (extra) Object.keys(extra).forEach(function (name) { found[name] = extra[name]; });
        return found;
    }

    // -- the status line ---------------------------------------------------- //

    let statusTimer = 0;

    function say(text, kind) {
        const holder = byId(IDS.status);
        if (!holder) return;
        const target = holder.querySelector(".mc-llm-notice") || holder;
        if (!holder.dataset.mcVoiceKept) {
            holder.dataset.mcVoiceKept = target.innerHTML || "";
        }
        target.innerHTML = "";
        const line = document.createElement("div");
        line.className = "mc-llm-notice mc-llm-notice-" + (kind || "info");
        line.textContent = text;
        target.appendChild(line);
        if (statusTimer) window.clearTimeout(statusTimer);
        // Put back what Python had said. Voice messages are transient and the
        // conversation's own status is not: leaving "Hold to speak." where
        // "Reply complete." belongs would be this file overwriting the panel.
        statusTimer = window.setTimeout(function () {
            const current = byId(IDS.status);
            if (!current) return;
            const kept = current.dataset.mcVoiceKept;
            if (typeof kept === "string") {
                const inner = current.querySelector(".mc-llm-notice") || current;
                inner.innerHTML = kept;
            }
            delete current.dataset.mcVoiceKept;
        }, STATUS_HOLD_MS);
    }

    // -- what is installed --------------------------------------------------- //

    let readiness = null;
    let readinessAt = 0;

    function refreshStatus(force) {
        const now = Date.now();
        if (!force && readiness && now - readinessAt < 5000) {
            return Promise.resolve(readiness);
        }
        return fetch(url(ROUTES.status), {
            method: "POST", credentials: "same-origin", headers: headers(),
        }).then(function (response) {
            return response.json();
        }).then(function (payload) {
            readiness = payload;
            readinessAt = Date.now();
            return payload;
        }).catch(function () {
            readiness = null;
            return null;
        });
    }

    // -- audio out ----------------------------------------------------------- //

    let context = null;
    let playing = null;

    function audioContext() {
        if (context) return context;
        const Ctor = window.AudioContext || window.webkitAudioContext;
        if (!Ctor) return null;
        try {
            context = new Ctor();
        } catch (error) {
            context = null;
        }
        return context;
    }

    // Mobile browsers suspend Web Audio until the page has been interacted with,
    // and a persisted "speak replies automatically" is intent rather than
    // permission. Every explicit voice gesture resumes it -- pressing the mic,
    // opening the Voice flyout, changing either switch -- so by the time a reply
    // arrives the common case is already unlocked.
    function unlock() {
        const ctx = audioContext();
        if (!ctx) return Promise.resolve(false);
        if (ctx.state !== "suspended") return Promise.resolve(true);
        try {
            return Promise.resolve(ctx.resume()).then(function () {
                return ctx.state !== "suspended";
            }).catch(function () { return false; });
        } catch (error) {
            return Promise.resolve(false);
        }
    }

    function stopPlayback() {
        if (!playing) return;
        try {
            playing.onended = null;
            playing.stop();
        } catch (error) {
            // A source that has already finished throws, which is not a failure.
        }
        playing = null;
    }

    function play(buffer) {
        const ctx = audioContext();
        if (!ctx) return Promise.reject(new Error("no audio"));
        return new Promise(function (resolve, reject) {
            let settled = false;
            const done = function (error) {
                if (settled) return;
                settled = true;
                if (error) reject(error); else resolve();
            };
            try {
                ctx.decodeAudioData(buffer, function (decoded) {
                    stopPlayback();
                    const source = ctx.createBufferSource();
                    source.buffer = decoded;
                    source.connect(ctx.destination);
                    source.onended = function () {
                        if (playing === source) playing = null;
                    };
                    playing = source;
                    source.start(0);
                    done(null);
                }, function (error) { done(error || new Error("decode failed")); });
            } catch (error) {
                done(error);
            }
        });
    }

    // -- audio in ------------------------------------------------------------ //

    // A worklet that does one thing: copy the mono channel it is given into a
    // message. No processing, no buffering policy, no state -- everything that
    // decides anything happens on the main thread where it can be tested.
    const WORKLET = [
        "class McVoiceTap extends AudioWorkletProcessor {",
        "  process(inputs) {",
        "    const input = inputs[0];",
        "    if (input && input[0] && input[0].length) {",
        "      this.port.postMessage(new Float32Array(input[0]));",
        "    }",
        "    return true;",
        "  }",
        "}",
        "registerProcessor('mc-voice-tap', McVoiceTap);",
    ].join("\n");

    function resample(chunks, from, to) {
        let length = 0;
        for (let i = 0; i < chunks.length; i += 1) length += chunks[i].length;
        const joined = new Float32Array(length);
        let at = 0;
        for (let i = 0; i < chunks.length; i += 1) {
            joined.set(chunks[i], at);
            at += chunks[i].length;
        }
        if (!from || from === to || !length) return joined;
        // Linear interpolation. This is dictation, not mastering: the artefacts
        // a windowed filter would remove are an order of magnitude below what
        // Whisper is robust to, and a resampling dependency for one function is
        // a dependency this feature does not need.
        const ratio = from / to;
        const count = Math.max(1, Math.floor(length / ratio));
        const out = new Float32Array(count);
        for (let i = 0; i < count; i += 1) {
            const position = i * ratio;
            const low = Math.floor(position);
            const high = Math.min(low + 1, length - 1);
            const fraction = position - low;
            out[i] = joined[low] * (1 - fraction) + joined[high] * fraction;
        }
        return out;
    }

    function encodeWav(samples, rate) {
        const bytes = samples.length * 2;
        const buffer = new ArrayBuffer(44 + bytes);
        const view = new DataView(buffer);
        const text = function (offset, value) {
            for (let i = 0; i < value.length; i += 1) view.setUint8(offset + i, value.charCodeAt(i));
        };
        text(0, "RIFF");
        view.setUint32(4, 36 + bytes, true);
        text(8, "WAVE");
        text(12, "fmt ");
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true);
        view.setUint16(22, 1, true);
        view.setUint32(24, rate, true);
        view.setUint32(28, rate * 2, true);
        view.setUint16(32, 2, true);
        view.setUint16(34, 16, true);
        text(36, "data");
        view.setUint32(40, bytes, true);
        let offset = 44;
        for (let i = 0; i < samples.length; i += 1) {
            let value = samples[i];
            if (value > 1) value = 1;
            else if (value < -1) value = -1;
            view.setInt16(offset, value < 0 ? value * 0x8000 : value * 0x7fff, true);
            offset += 2;
        }
        return buffer;
    }

    function secureContext() {
        try {
            if (typeof window.isSecureContext === "boolean") return window.isSecureContext;
        } catch (error) {
            return false;
        }
        return true;
    }

    function microphoneAvailable() {
        return !!(navigator && navigator.mediaDevices
                  && typeof navigator.mediaDevices.getUserMedia === "function");
    }

    // One capture at a time, and every resource it holds named here so that
    // stopping it can never leave a track open. An open microphone the user did
    // not ask for is the failure this feature must not have.
    let capture = null;

    function startCapture() {
        const ctx = audioContext();
        if (!ctx) return Promise.reject(new Error("no audio"));
        return navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true,
            },
        }).then(function (stream) {
            const state = {stream: stream, chunks: [], rate: ctx.sampleRate, nodes: []};
            const source = ctx.createMediaStreamSource(stream);
            state.nodes.push(source);

            const useProcessor = function () {
                const processor = ctx.createScriptProcessor(4096, 1, 1);
                processor.onaudioprocess = function (event) {
                    if (!capture || capture !== state) return;
                    state.chunks.push(new Float32Array(event.inputBuffer.getChannelData(0)));
                };
                source.connect(processor);
                // Connected to the destination with no gain of its own: a
                // ScriptProcessorNode that is not in the graph is not called at
                // all in several browsers. It contributes silence.
                const silence = ctx.createGain();
                silence.gain.value = 0;
                processor.connect(silence);
                silence.connect(ctx.destination);
                state.nodes.push(processor, silence);
                return state;
            };

            if (!ctx.audioWorklet || typeof ctx.audioWorklet.addModule !== "function"
                || typeof Blob === "undefined" || !window.URL || !window.URL.createObjectURL) {
                return useProcessor();
            }
            let address = "";
            try {
                address = window.URL.createObjectURL(new Blob([WORKLET],
                                                             {type: "text/javascript"}));
            } catch (error) {
                return useProcessor();
            }
            return ctx.audioWorklet.addModule(address).then(function () {
                try { window.URL.revokeObjectURL(address); } catch (error) { /* ignore */ }
                const node = new window.AudioWorkletNode(ctx, "mc-voice-tap");
                node.port.onmessage = function (event) {
                    if (!capture || capture !== state) return;
                    state.chunks.push(event.data);
                };
                source.connect(node);
                state.nodes.push(node);
                return state;
            }).catch(function () {
                try { window.URL.revokeObjectURL(address); } catch (error) { /* ignore */ }
                return useProcessor();
            });
        });
    }

    function releaseCapture(state) {
        if (!state) return;
        (state.nodes || []).forEach(function (node) {
            try { node.disconnect(); } catch (error) { /* ignore */ }
            if (node.port) { try { node.port.onmessage = null; } catch (error) { /* ignore */ } }
            if (typeof node.onaudioprocess !== "undefined") node.onaudioprocess = null;
        });
        // Every track, every time. A MediaStream left alive is a microphone
        // indicator left on, which is the most alarming thing a page can do.
        try {
            (state.stream.getTracks() || []).forEach(function (track) { track.stop(); });
        } catch (error) { /* ignore */ }
        state.nodes = [];
    }

    // -- the composer -------------------------------------------------------- //

    function composer() {
        const holder = byId(IDS.message);
        if (!holder) return null;
        return holder.tagName === "TEXTAREA" ? holder : holder.querySelector("textarea");
    }

    function insert(text) {
        const field = composer();
        if (!field) return false;
        const existing = field.value || "";
        // Appended, never replaced. Somebody who typed half a message and then
        // dictated the rest has not asked for the half they typed to be thrown
        // away, and there is no undo in a Gradio textbox.
        const joined = existing.trim()
            ? existing.replace(/\s*$/, "") + " " + text
            : text;
        field.value = joined;
        try {
            field.selectionStart = field.selectionEnd = joined.length;
        } catch (error) { /* a browser that will not move the caret is not a failure */ }
        // The pattern this repository already uses when the browser writes into
        // a Gradio control: assign, then tell it, on both events it listens to.
        field.dispatchEvent(new Event("input", {bubbles: true}));
        field.dispatchEvent(new Event("change", {bubbles: true}));
        return true;
    }

    function generating() {
        const stop = clickable(IDS.stop);
        return !!(stop && stop.offsetParent !== null && !stop.disabled);
    }

    function pressSend() {
        const button = clickable(IDS.send);
        if (!button || button.disabled) return false;
        button.click();
        return true;
    }

    // -- the gesture --------------------------------------------------------- //

    let holding = null;

    function markMic(state) {
        const button = clickable(IDS.mic);
        if (!button) return;
        button.classList.remove("mc-llm-voice-recording", "mc-llm-voice-working",
                                "mc-llm-voice-error");
        if (state) button.classList.add("mc-llm-voice-" + state);
        const labels = {
            recording: "Recording — release to transcribe",
            working: "Transcribing",
            error: "Voice Chat is not set up",
        };
        button.setAttribute("aria-label", labels[state] || "Hold to dictate");
    }

    function refuse(text) {
        markMic("error");
        say(text, "warn");
        window.setTimeout(function () { markMic(""); }, STATUS_HOLD_MS);
    }

    function beginHold(event) {
        if (holding || capture) return;
        const button = clickable(IDS.mic);
        if (!button) return;
        // Whatever else happens, this press is a user gesture and Web Audio may
        // be unlocked by it. Done first and unconditionally so that a press that
        // is then refused still leaves playback able to work.
        unlock();
        // The V1 form of "interrupt the assistant": the speaker stops before the
        // microphone opens, which also stops a phone feeding its own output back
        // into the recording.
        stopPlayback();

        if (generating()) {
            refuse(MESSAGES.busy);
            return;
        }
        if (!secureContext()) {
            refuse(MESSAGES.insecure);
            return;
        }
        if (!microphoneAvailable()) {
            refuse(MESSAGES.unsupported);
            return;
        }

        try {
            if (event && event.pointerId !== undefined && button.setPointerCapture) {
                button.setPointerCapture(event.pointerId);
            }
        } catch (error) { /* a browser without pointer capture still works */ }

        const session = {at: Date.now(), pointerId: event && event.pointerId, cancelled: false};
        holding = session;
        markMic("recording");

        refreshStatus(false).then(function (found) {
            if (holding !== session) return null;
            if (!found || !found.ok) {
                holding = null;
                refuse(MESSAGES.failed);
                return null;
            }
            if (!found.ready) {
                holding = null;
                refuse(found.not_ready_message || MESSAGES.notReady);
                return null;
            }
            return startCapture().then(function (state) {
                if (holding !== session || session.cancelled) {
                    releaseCapture(state);
                    return;
                }
                capture = state;
                session.timer = window.setTimeout(function () {
                    if (holding === session) endHold(false);
                }, MAX_HOLD_MS);
            });
        }).catch(function (error) {
            if (holding === session) holding = null;
            markMic("");
            const name = error && error.name;
            refuse(name === "NotAllowedError" || name === "SecurityError"
                ? MESSAGES.denied
                : MESSAGES.unsupported);
        });
    }

    function endHold(cancelled) {
        const session = holding;
        holding = null;
        if (!session) return;
        if (session.timer) window.clearTimeout(session.timer);
        session.cancelled = cancelled;

        const state = capture;
        capture = null;
        const held = Date.now() - session.at;
        releaseCapture(state);
        markMic("");

        if (cancelled || !state) return;
        if (held < MIN_HOLD_MS) {
            refuse(MESSAGES.tooShort);
            return;
        }
        const samples = resample(state.chunks, state.rate, TARGET_RATE);
        state.chunks = [];
        if (!samples.length) {
            refuse(MESSAGES.empty);
            return;
        }
        send(encodeWav(samples, TARGET_RATE));
    }

    function send(wav) {
        markMic("working");
        say("Transcribing…");
        fetch(url(ROUTES.stt), {
            method: "POST",
            credentials: "same-origin",
            headers: headers({"Content-Type": "audio/wav"}),
            body: wav,
        }).then(function (response) {
            return response.json().catch(function () { return {ok: false}; });
        }).then(function (payload) {
            markMic("");
            if (!payload || !payload.ok) {
                say((payload && payload.error) || MESSAGES.failed, "warn");
                return;
            }
            if (!insert(payload.text)) {
                say(MESSAGES.failed, "error");
                return;
            }
            say("Transcribed.");
            if (!payload.auto_send) return;
            // One task, so Gradio has taken the value off the textarea before
            // the Send handler reads it. Send is pressed rather than reproduced:
            // attachments, thread selection, sampling values and cancellation
            // all already live behind that button.
            window.setTimeout(function () {
                if (!generating()) pressSend();
            }, 0);
        }).catch(function () {
            markMic("");
            say(MESSAGES.failed, "error");
        });
    }

    function wireMicrophone() {
        const button = clickable(IDS.mic);
        if (!button || button.dataset.mcVoiceWired === "1") return;
        button.dataset.mcVoiceWired = "1";
        markMic("");
        button.addEventListener("pointerdown", function (event) {
            if (event.button !== undefined && event.button !== 0) return;
            if (event.preventDefault) event.preventDefault();
            beginHold(event);
        });
        button.addEventListener("pointerup", function () { endHold(false); });
        button.addEventListener("pointercancel", function () { endHold(true); });
        // A press that leaves the window entirely -- an incoming call, a task
        // switch -- ends the recording rather than leaving the microphone open.
        button.addEventListener("lostpointercapture", function () {
            if (holding) endHold(false);
        });
        button.addEventListener("contextmenu", function (event) {
            if (event.preventDefault) event.preventDefault();
        });
        // A click handler that does nothing, so a browser which synthesises one
        // after the pointer sequence cannot make the button do anything twice.
        button.addEventListener("click", function (event) {
            if (event.preventDefault) event.preventDefault();
        });
    }

    // -- speaking a reply ----------------------------------------------------- //

    let lastToken = "";
    let speaking = false;

    function checkToken() {
        const token = fieldValue(IDS.token);
        if (!token || token === lastToken) return;
        lastToken = token;
        if (speaking) return;
        speaking = true;
        unlock().then(function (unlocked) {
            if (!unlocked) {
                speaking = false;
                say(MESSAGES.blocked, "warn");
                return null;
            }
            return fetch(url(ROUTES.tts), {
                method: "POST",
                credentials: "same-origin",
                headers: headers({"Content-Type": "application/json"}),
                body: JSON.stringify({token: token}),
            }).then(function (response) {
                if (!response.ok) throw new Error("tts");
                return response.arrayBuffer();
            }).then(function (buffer) {
                return play(buffer);
            }).then(function () {
                speaking = false;
            });
        }).catch(function () {
            speaking = false;
            say(MESSAGES.speakFailed, "warn");
        });
    }

    // -- the Settings page row ------------------------------------------------ //

    function wireSettings() {
        const holder = document.querySelector(".mc-voice-settings");
        if (!holder || holder.dataset.mcVoiceWired === "1") return;
        holder.dataset.mcVoiceWired = "1";
        const buttons = holder.querySelectorAll("[data-mc-voice-install]");
        Array.prototype.forEach.call(buttons, function (button) {
            button.addEventListener("click", function (event) {
                if (event.preventDefault) event.preventDefault();
                const kind = button.getAttribute("data-mc-voice-install");
                button.disabled = true;
                button.textContent = "Starting…";
                fetch(url(ROUTES.install), {
                    method: "POST",
                    credentials: "same-origin",
                    headers: headers({"Content-Type": "application/json"}),
                    body: JSON.stringify({kind: kind}),
                }).catch(function () { /* the row below reports it */ });
            });
        });
        window.setInterval(function () { paintSettings(holder); }, SETTINGS_POLL_MS);
        paintSettings(holder);
    }

    function paintSettings(holder) {
        refreshStatus(true).then(function (payload) {
            if (!payload || !payload.ok) return;
            const runtime = holder.querySelector(".mc-voice-runtime");
            if (runtime) runtime.textContent = payload.runtime_message || "";
            ["stt", "tts"].forEach(function (kind) {
                const line = holder.querySelector('[data-mc-voice-status="' + kind + '"]');
                const button = holder.querySelector('[data-mc-voice-install="' + kind + '"]');
                const progress = (payload.progress || {})[kind] || {};
                const ready = kind === "stt" ? payload.stt_ready : payload.tts_ready;
                if (line) {
                    line.textContent = progress.running
                        ? (progress.text || "Downloading…") + " "
                          + Math.round((progress.fraction || 0) * 100) + "%"
                        : (kind === "stt" ? payload.stt_message : payload.tts_message);
                }
                if (button) {
                    button.disabled = !!progress.running || !!ready;
                    button.textContent = ready
                        ? "Installed"
                        : "Download default " + kind.toUpperCase();
                }
            });
        });
    }

    // -- wiring --------------------------------------------------------------- //

    function wireGestures() {
        // Every explicit voice gesture is an opportunity to unlock playback, and
        // the flyout and the two switches are gestures a user makes precisely
        // because they want to hear something.
        const chip = clickable(IDS.chip);
        if (chip && chip.dataset.mcVoiceUnlock !== "1") {
            chip.dataset.mcVoiceUnlock = "1";
            chip.addEventListener("click", function () { unlock(); refreshStatus(true); });
        }
        ["mc-llm-chat-voice-auto-send", "mc-llm-chat-voice-auto-speak"].forEach(function (id) {
            const holder = byId(id);
            if (!holder || holder.dataset.mcVoiceUnlock === "1") return;
            holder.dataset.mcVoiceUnlock = "1";
            holder.addEventListener("change", function () { unlock(); });
        });
    }

    function attempt(what, run) {
        try {
            run();
        } catch (error) {
            console.error("Model Chain: Voice Chat could not " + what, error);
        }
    }

    function wire() {
        attempt("wire the microphone", wireMicrophone);
        attempt("wire the voice gestures", wireGestures);
        attempt("wire the Voice Chat settings", wireSettings);
    }

    let watching = false;

    function watch() {
        if (watching) return;
        watching = true;
        window.setInterval(function () {
            attempt("check for a reply to speak", checkToken);
        }, TOKEN_POLL_MS);
        // Leaving the page with the speaker running is not something a user asks
        // for, and a suspended tab that resumes and starts talking is worse.
        window.addEventListener("pagehide", stopPlayback);
        window.addEventListener("beforeunload", stopPlayback);
    }

    function start() {
        wire();
        watch();
    }

    if (typeof onUiLoaded === "function") {
        onUiLoaded(start);
    } else if (document.readyState !== "loading") {
        start();
    } else {
        document.addEventListener("DOMContentLoaded", start);
    }

    // The panel is rebuilt on some UI updates, so the wiring is re-applied. Each
    // step is idempotent through the dataset flags above, so re-running it costs
    // a query and nothing else -- and, critically, never a second pointer
    // listener on the same button.
    if (typeof onAfterUiUpdate === "function") {
        onAfterUiUpdate(wire);
    }
})();
