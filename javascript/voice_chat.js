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
        turn: "mc-llm-chat-voice-turn",
        runState: "mc-llm-chat-voice-run-state",
        key: "mc-llm-chat-voice-key",
        message: "mc-llm-chat-message",
        send: "mc-llm-chat-send",
        stop: "mc-llm-chat-stop",
        status: "mc-llm-chat-status",
        chip: "mc-llm-chat-to-voice",
        engine: "mc-llm-chat-voice-engine",
        autoSpeak: "mc-llm-chat-voice-auto-speak",
    };

    const ROUTES = {
        status: "model-chain/voice/status",
        stt: "model-chain/voice/stt",
        tts: "model-chain/voice/tts",
        stream: "model-chain/voice/tts-stream",
        cancel: "model-chain/voice/cancel",
        runtime: "model-chain/voice/runtime",
        install: "model-chain/voice/install",
        voices: "model-chain/voice/voices",
        voiceDefault: "model-chain/voice/voice/default",
        voiceTest: "model-chain/voice/voice/test",
        voiceRename: "model-chain/voice/voice/rename",
        voiceDelete: "model-chain/voice/voice/delete",
        cloningStatus: "model-chain/voice/cloning/status",
        cloningInstall: "model-chain/voice/cloning/install",
        cloningStart: "model-chain/voice/cloning/start",
        cloningAbort: "model-chain/voice/cloning/abort",
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
    // Nothing is happening most of the time, and a status route polled every
    // second and a half for the life of a browser tab is a cost with no reader.
    const SETTINGS_IDLE_MS = 20000;
    const SETTINGS_POLL_MAX_MS = 60000;

    // How long a stream may produce nothing before it is called a stream that is
    // not streaming. Section 40: an intermediary that buffers the whole response
    // is the one deployment failure that otherwise looks exactly like a slow
    // machine, and saying so is better than claiming streaming that is not
    // happening. Generous, because a cold Kokoro on a busy CPU really can take
    // this long to produce a first sentence.
    const STREAM_SILENCE_MS = 30000;

    const MESSAGES = {
        notReady: "Voice Chat is not set up. Install both models in Settings → Voice Chat.",
        denied: "Microphone access was not allowed by the browser or device.",
        notFound: "No microphone is available.",
        unreadable: "The microphone could not be opened.",
        tooShort: "Hold to speak.",
        failed: "Voice transcription failed. Your message was not sent.",
        empty: "No speech was detected.",
        speakFailed: "The reply was generated, but Voice could not read it aloud.",
        notStreaming: "The reply is being read aloud, but no audio has arrived — something "
            + "between this page and the WebUI may be buffering the whole response.",
        blocked: "Voice is enabled; tap Voice or the microphone once to allow audio playback.",
        busy: "Wait for the reply or press Stop.",
        // Section 39. Not "HTTPS is required" -- that is a claim about this
        // extension's policy that stopped being true, and it is not even
        // reliably the reason: what a browser actually does on an origin it
        // does not trust is decline to expose mediaDevices at all, and the
        // honest report of that is what it is.
        unsupported: "This browser did not make microphone capture available for this page.",
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

    // Python puts this process's page token in two places, because the two
    // surfaces that need it are in different parts of the document: a hidden
    // component inside Conversation, and an attribute on the Settings row.
    // `scope` names the one to read first, so the Settings row never depends on
    // the Conversation panel having been built and hydrated — it reads the
    // attribute sitting on its own container. Without a token every request is
    // refused with a 403, which is a failure worth not depending on the wrong
    // element for.
    function pageKey(scope) {
        if (scope && scope.getAttribute) {
            const own = scope.getAttribute("data-mc-voice-key");
            if (own) return own;
        }
        const fromPanel = fieldValue(IDS.key);
        if (fromPanel) return fromPanel;
        const holder = document.querySelector("[data-mc-voice-key]")
            || root().querySelector("[data-mc-voice-key]");
        return holder ? (holder.getAttribute("data-mc-voice-key") || "") : "";
    }

    function headers(extra, scope) {
        const found = {"X-Model-Chain-Voice": pageKey(scope)};
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

    // Always resolves, and always with something that has an `ok` and, when it
    // is false, an `error` somebody can read. The first version resolved with
    // null on every failure, which meant each caller quietly did nothing --
    // and the visible result of that was a Settings row frozen on "Starting…"
    // with nothing anywhere to say why. A status check that cannot report its
    // own failure is worse than no status check.
    function refreshStatus(force, scope) {
        const now = Date.now();
        if (!force && readiness && now - readinessAt < 5000) {
            return Promise.resolve(readiness);
        }
        return fetch(url(ROUTES.status), {
            method: "POST", credentials: "same-origin", headers: headers(null, scope),
        }).then(function (response) {
            return response.json().catch(function () {
                return {ok: false, error: "The WebUI answered HTTP " + response.status
                                          + " with something this page cannot read."};
            }).then(function (payload) {
                if (!payload || typeof payload !== "object") {
                    payload = {ok: false, error: "The WebUI sent an empty answer."};
                }
                if (!response.ok && payload.ok === undefined) payload.ok = false;
                if (payload.ok) {
                    readiness = payload;
                    readinessAt = Date.now();
                } else {
                    readiness = null;
                    console.error("Model Chain: Voice Chat status refused (HTTP "
                                  + response.status + ") — " + (payload.error || ""));
                }
                return payload;
            });
        }).catch(function (error) {
            readiness = null;
            console.error("Model Chain: Voice Chat could not reach its status route", error);
            return {ok: false,
                    error: "Could not reach this WebUI to read the Voice Chat status ("
                           + ((error && error.message) || "network error") + ")."};
        });
    }

    // -- audio out ----------------------------------------------------------- //

    // Everything below plays a reply that is still being written. The V1 path
    // waited for a completed reply, fetched a whole WAV, and handed the
    // ArrayBuffer to decodeAudioData -- which is correct and is also the reason
    // nothing was heard until every sentence had been synthesised. What replaces
    // it is a reader over a body that has no end yet:
    //
    //   fetch -> response.body.getReader()
    //         -> carry an odd trailing byte between reads
    //         -> Int16 little-endian -> Float32
    //         -> AudioBuffer blocks scheduled back to back
    //         -> a queue depth the reader throttles itself against
    //
    // Two things about it are easy to get wrong and are handled deliberately.
    //
    // Network chunks are not sample-aligned. A read can end on the first byte of
    // a 16-bit sample, and treating that byte as a whole sample turns the rest of
    // the reply into noise. `carry` is the byte held back.
    //
    // Audio from a cancelled turn must never be heard. Every scheduled source
    // belongs to a turn id, and everything -- the reader, the decoder, the
    // scheduler and the stop path -- checks that id before it acts. Section 24.

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

    // -- the adaptive playback buffer --------------------------------------- //

    // Section 22. The browser is authoritative about playback timing because it
    // is the only party that knows when the speaker actually consumed a sample.
    // START is how much has to exist before playback begins; going lower risks a
    // gap in the first sentence, and every millisecond of it is latency the user
    // hears as lag. So it moves: an underrun raises it for the next turn, and
    // several clean turns lower it back towards the floor.
    const BUFFER = {
        start: 1.6,
        min: 1.2,
        max: 3.2,
        high: 12.0,
        clean: 0,
    };

    function raiseStartBuffer() {
        BUFFER.start = Math.min(BUFFER.max, Math.round((BUFFER.start + 0.4) * 10) / 10);
        BUFFER.clean = 0;
    }

    function relaxStartBuffer() {
        BUFFER.clean += 1;
        if (BUFFER.clean >= 3) {
            BUFFER.start = Math.max(BUFFER.min, Math.round((BUFFER.start - 0.2) * 10) / 10);
            BUFFER.clean = 0;
        }
    }

    // -- one spoken turn ------------------------------------------------------ //

    // The state of the turn being spoken right now, and the only thing any of the
    // stop paths has to look at. `id` is the server's opaque turn token; a null
    // `id` means nothing is speaking.
    let speech = null;

    function newSpeech(turnId) {
        return {
            id: turnId,
            controller: (typeof AbortController === "function") ? new AbortController() : null,
            sources: new Set(),
            nextStart: 0,
            queued: 0,
            started: false,
            underruns: 0,
            ended: false,
            pending: [],
            pendingSeconds: 0,
            draining: false,
        };
    }

    function queuedSeconds(state) {
        const ctx = audioContext();
        if (!ctx || !state.started) return state.queued;
        return Math.max(0, state.nextStart - ctx.currentTime);
    }

    // Stops the speaker now. Everything scheduled for this turn is stopped and
    // dropped, which is what makes Stop audible inside one animation frame
    // rather than at the end of whatever block was already playing.
    function stopPlayback() {
        if (playing) {
            try {
                playing.onended = null;
                playing.stop();
            } catch (error) {
                // A source that has already finished throws, which is not a failure.
            }
            playing = null;
        }
        if (!speech) return;
        speech.ended = true;
        speech.sources.forEach(function (source) {
            try {
                source.onended = null;
                source.stop();
            } catch (error) { /* already finished */ }
        });
        speech.sources.clear();
        speech.pending = [];
        speech.pendingSeconds = 0;
        speech.nextStart = 0;
        speech.started = false;
        if (speech.controller) {
            try { speech.controller.abort(); } catch (error) { /* already aborted */ }
        }
    }

    // Tell the server the turn is over as well. The browser's own stop is
    // immediate and local; this is what stops Kokoro computing speech nobody is
    // going to hear. Best effort by design -- a failed cancel must not keep the
    // composer showing Stop.
    function tellServerToStop(turnId, reason) {
        if (!turnId) return Promise.resolve();
        return fetch(url(ROUTES.cancel), {
            method: "POST",
            credentials: "same-origin",
            headers: headers({"Content-Type": "application/json"}),
            // A fixed word from a fixed list, never text and never a message.
            // It exists so that "the browser could not play this reply" is a
            // line in model_chain.log rather than a status the panel overwrites
            // a moment later.
            body: JSON.stringify({turn: turnId, reason: reason || "user"}),
            keepalive: true,
        }).catch(function () { /* the turn ends on its own when the stream closes */ });
    }

    // The one entry point every stop path uses: the Stop button, the microphone,
    // "Speak replies automatically" being switched off, unloading the engine, and
    // leaving the page. Idempotent -- section 26 requires that, because two of
    // those paths deliberately fire together.
    function stopSpeaking(alsoTellServer, reason) {
        const turnId = speech ? speech.id : "";
        stopPlayback();
        speech = null;
        setVoiceBusy(false);
        if (alsoTellServer !== false) tellServerToStop(turnId, reason);
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

    // -- PCM16 -> Float32 ----------------------------------------------------- //

    // Little-endian, explicitly, matching what the worker writes. Written out
    // rather than handed to a typed-array view because a Int16Array over an
    // arbitrary byte offset requires 2-byte alignment and a network chunk does
    // not promise one -- so this reads through a DataView, which does not care.
    function toFloat32(bytes) {
        const samples = new Float32Array(bytes.length >> 1);
        const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.length);
        for (let index = 0; index < samples.length; index += 1) {
            const value = view.getInt16(index * 2, true);
            samples[index] = value < 0 ? value / 0x8000 : value / 0x7fff;
        }
        return samples;
    }

    // Schedule one block at the end of what is already scheduled. `nextStart` is
    // the clock this keeps: Web Audio gives no "append", so back-to-back playback
    // is arithmetic on the context's own timeline. Resampling to the device rate
    // is the browser's job and is why the buffer is created at the source rate.
    // Section 22. Blocks are held back until START_BUFFER seconds of them
    // exist, and then released together -- which is what stops the first
    // sentence stuttering while the machine is still producing the second. The
    // held blocks are released the moment the stream ends too, however little
    // there is, so a two-word reply is not silently swallowed by the buffer.
    function schedule(state, samples, rate) {
        if (!samples.length || state.ended) return;
        state.pending.push({samples: samples, rate: rate});
        state.pendingSeconds += samples.length / (rate || 24000);
        if (!state.started && state.pendingSeconds < BUFFER.start && !state.draining) return;
        release(state);
    }

    function release(state) {
        const waiting = state.pending;
        state.pending = [];
        state.pendingSeconds = 0;
        waiting.forEach(function (block) { play_(state, block.samples, block.rate); });
    }

    function play_(state, samples, rate) {
        const ctx = audioContext();
        if (!ctx || !samples.length || state.ended) return;
        let buffer;
        try {
            buffer = ctx.createBuffer(1, samples.length, rate);
        } catch (error) {
            return;
        }
        buffer.getChannelData(0).set(samples);
        const source = ctx.createBufferSource();
        source.buffer = buffer;
        source.connect(ctx.destination);
        const now = ctx.currentTime;
        if (!state.started || state.nextStart < now) {
            if (state.started) {
                // The queue ran dry while the stream was still producing: the
                // speaker fell silent mid-sentence. Counted, and the next turn
                // starts with a deeper buffer because of it.
                state.underruns += 1;
            }
            state.nextStart = now + 0.02;
        }
        source.onended = function () { state.sources.delete(source); };
        state.sources.add(source);
        try {
            source.start(state.nextStart);
        } catch (error) {
            state.sources.delete(source);
            return;
        }
        state.nextStart += buffer.duration;
        state.started = true;
    }

    // -- the stream reader ---------------------------------------------------- //

    function speakTurn(turnId) {
        if (!turnId || (speech && speech.id === turnId)) return;
        // A new reply supersedes the last one. Locally first -- the server has
        // already cancelled its side by creating the new turn.
        if (speech) stopSpeaking(true);
        const state = newSpeech(turnId);
        speech = state;
        setVoiceBusy(true);

        unlock().then(function (unlocked) {
            if (!unlocked) {
                say(MESSAGES.blocked, "warn");
                // Not fatal and not silent: the stream is still cancelled so the
                // server stops synthesising for a speaker that will not play,
                // and the reason is reported so it appears in the log rather
                // than only in a status line the panel is about to overwrite.
                if (speech === state) stopSpeaking(true, "audio is locked until a gesture");
                return null;
            }
            return fetch(url(ROUTES.stream), {
                method: "POST",
                credentials: "same-origin",
                headers: headers({"Content-Type": "application/json"}),
                body: JSON.stringify({turn: turnId}),
                signal: state.controller ? state.controller.signal : undefined,
            }).then(function (response) {
                if (!response.ok || !response.body || !response.body.getReader) {
                    throw new Error("stream");
                }
                const rate = parseInt(response.headers.get("X-Model-Chain-Voice-Rate")
                                      || "24000", 10) || 24000;
                const stamped = response.headers.get("X-Model-Chain-Voice-Turn");
                if (stamped && stamped !== turnId) throw new Error("turn");
                return pump(state, response.body.getReader(), rate);
            });
        }).catch(function (error) {
            if (speech !== state) return;
            if (!error || error.name === "AbortError") {
                stopSpeaking(false);
                return;
            }
            say(MESSAGES.speakFailed, "warn");
            console.warn("Model Chain: Voice could not play a reply", error);
            // Reported rather than only shown. The status line this writes to
            // is the panel's own, and the panel overwrites it with "Reply
            // complete." a moment later -- so a user watching for an
            // explanation sees nothing at all.
            stopSpeaking(true, "the stream could not be played: "
                         + ((error && error.message) || "unknown"));
        });
    }

    function pump(state, reader, rate) {
        let carry = null;
        const opened = Date.now();
        let complained = false;

        const step = function () {
            if (state.ended || speech !== state) {
                try { reader.cancel(); } catch (error) { /* already done */ }
                return Promise.resolve();
            }
            // Section 22's high-water mark. Not reading is the whole mechanism:
            // the socket stops being drained, the server's send blocks, its queue
            // fills, and the sherpa callback stops being called. Backpressure all
            // the way to the producer without one sample being dropped.
            if (queuedSeconds(state) > BUFFER.high) {
                return wait(200).then(step);
            }
            if (!complained && !state.started && Date.now() - opened > STREAM_SILENCE_MS) {
                complained = true;
                say(MESSAGES.notStreaming, "warn");
            }
            return reader.read().then(function (result) {
                if (result.done) {
                    if (carry) carry = null;
                    // Whatever is still held back goes now, however little it
                    // is: the stream is over, so there is nothing left to wait
                    // for and a short reply must not die in the prebuffer.
                    state.draining = true;
                    release(state);
                    finishSpeech(state);
                    return null;
                }
                let bytes = new Uint8Array(result.value);
                if (carry) {
                    const joined = new Uint8Array(carry.length + bytes.length);
                    joined.set(carry, 0);
                    joined.set(bytes, carry.length);
                    bytes = joined;
                    carry = null;
                }
                if (bytes.length % 2) {
                    // The odd trailing byte is the first half of a sample whose
                    // second half is in the next chunk. Kept, never played.
                    carry = bytes.slice(bytes.length - 1);
                    bytes = bytes.subarray(0, bytes.length - 1);
                }
                if (bytes.length) schedule(state, toFloat32(bytes), rate);
                if (!complained && !state.started
                        && Date.now() - opened > STREAM_SILENCE_MS) {
                    complained = true;
                    say(MESSAGES.notStreaming, "warn");
                }
                return step();
            });
        };

        return step();
    }

    function finishSpeech(state) {
        if (speech !== state) return;
        if (state.underruns) raiseStartBuffer(); else relaxStartBuffer();
        // The reader is done but the speaker is not: what is scheduled still has
        // to play out before the composer stops showing Stop.
        const remaining = Math.max(0, queuedSeconds(state) * 1000) + 120;
        window.setTimeout(function () {
            if (speech !== state) return;
            speech = null;
            setVoiceBusy(false);
        }, remaining);
    }

    function wait(ms) {
        return new Promise(function (resolve) { window.setTimeout(resolve, ms); });
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

    // Capability, never policy. The question this asks is "did this browser give
    // this page a way to open a microphone", and the answer is the browser's
    // own -- which on an origin it does not trust is usually to leave
    // mediaDevices undefined. Voice Chat's requirement (section 39) is to stop
    // refusing pages of its own accord, not to pretend it can overrule that.
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

    // -- who is busy, and who says so ----------------------------------------- //

    // Section 25. Send/Stop visibility used to be read off the last CSS state
    // Gradio happened to apply, which is fine while the LLM is the only thing
    // that can be busy and wrong the moment Voice can outlive it: the reply
    // finishes, Gradio swaps Stop for Send, and the speaker is still talking
    // with no way to stop it. So the two facts are held separately and combined
    // in one place --
    //
    //     busy = (python run state === "llm") || voiceBusy
    //
    // -- and JS never fights Gradio for the same element: Python owns the run
    // state and writes it into a hidden field, this file owns whether Voice is
    // speaking, and only `applyComposerState` touches visibility.

    let voiceBusy = false;

    function runState() {
        return (fieldValue(IDS.runState) || "").trim();
    }

    function llmBusy() {
        if (byId(IDS.runState)) return runState() === "llm";
        // No hidden field on this page: fall back to what the button says. Worse,
        // and still better than nothing on a panel built before this existed.
        const stop = clickable(IDS.stop);
        return !!(stop && stop.offsetParent !== null && !stop.disabled);
    }

    function generating() {
        return llmBusy();
    }

    function busy() {
        return llmBusy() || voiceBusy;
    }

    function setVoiceBusy(value) {
        const wanted = !!value;
        if (voiceBusy === wanted) return;
        voiceBusy = wanted;
        applyComposerState();
    }

    // Shows Stop while either half is working, and Send when neither is. Only
    // called when something actually changed, so it is not a loop fighting
    // Gradio's own re-render.
    function applyComposerState() {
        const send = clickable(IDS.send);
        const stop = clickable(IDS.stop);
        if (!send || !stop) return;
        const working = busy();
        show(holderOf(send), !working);
        show(holderOf(stop), working);
    }

    // Hiding is ours; *revealing* has to undo Gradio's, and that is the half
    // that is easy to miss. When the model finishes, Python's own IDLE update
    // hides Stop -- correctly, as far as Python knows -- and if Voice is still
    // speaking, adding a class of our own to Send would leave a composer with
    // neither button in it. So this takes Gradio's marker off the control that
    // must be visible, and puts ours on the one that must not be. Reasserted on
    // every tick and on every run-state write, which is what makes it settle
    // rather than flicker.
    function show(node, wanted) {
        if (!node || !node.classList) return;
        if (wanted) {
            node.classList.remove("mc-llm-voice-hidden");
            node.classList.remove("hidden");
            if (node.style && node.style.display === "none") node.style.display = "";
        } else {
            node.classList.add("mc-llm-voice-hidden");
        }
    }

    // The wrapper Gradio put the button in, when there is one. Written to work
    // on a bare node too: this runs on every poll, and a composer being
    // re-rendered underneath it is an ordinary moment rather than an error.
    function holderOf(button) {
        if (!button) return button;
        if (typeof button.closest === "function") {
            try {
                return button.closest(".mc-llm-btn, .gradio-button, .form, div") || button;
            } catch (error) { /* a selector this browser dislikes is not fatal */ }
        }
        return button.parentElement || button;
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

    // Section 39's error map. Each of these is a different thing for the user to
    // do about it, which is the entire reason not to collapse them into one
    // sentence about HTTPS -- "permission was denied" and "there is no
    // microphone" have nothing in common except that no audio arrived.
    function captureFailure(error) {
        const name = (error && error.name) || "";
        if (name === "NotAllowedError" || name === "SecurityError") return MESSAGES.denied;
        if (name === "NotFoundError" || name === "OverconstrainedError") return MESSAGES.notFound;
        if (name === "NotReadableError" || name === "AbortError") return MESSAGES.unreadable;
        return MESSAGES.unreadable;
    }

    function beginHold(event) {
        if (holding || capture) return;
        const button = clickable(IDS.mic);
        if (!button) return;
        // Whatever else happens, this press is a user gesture and Web Audio may
        // be unlocked by it. Done first and unconditionally so that a press that
        // is then refused still leaves playback able to work.
        unlock();
        // Section 14: pressing the microphone while a reply is being read aloud
        // means the reply is over. The speaker stops here -- which also stops a
        // phone feeding its own output back into the recording -- and the server
        // is told, so an obsolete Kokoro run stops burning the CPU the
        // transcription is about to need rather than merely being inaudible.
        stopSpeaking(true);

        if (llmBusy()) {
            refuse(MESSAGES.busy);
            return;
        }
        // No secure-context gate. Capability detection, then let the browser
        // make the decision that is genuinely the browser's -- section 39.
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
                // The route's own sentence, not a generic one: "This page is out
                // of date with the WebUI. Reload it." is actionable and
                // "Voice transcription failed" is not.
                refuse((found && found.error) || MESSAGES.failed);
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
            // Section 39's map, in one place. The browser's own error name is
            // the only thing that knows which of these happened, and each of
            // them is a different thing for the user to do next.
            refuse(captureFailure(error));
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

    // Two hidden fields, two mechanisms, and never both for one reply.
    //
    //   `turn`   an opaque id for a reply that is being generated *now*. This is
    //            the streaming path and it is what Auto Speak normally uses.
    //   `token`  the V1 completed-reply target. Python writes one only when the
    //            run produced no turn -- so a deployment that cannot stream, or a
    //            reply that finished before Voice was ready, still gets spoken,
    //            and section 20's "two mechanisms must not both fire for one
    //            reply" holds because the server decides which one exists.

    let lastTurn = "";
    let lastToken = "";
    let speaking = false;

    function checkSpeech() {
        const turnId = fieldValue(IDS.turn);
        if (turnId && turnId !== lastTurn) {
            lastTurn = turnId;
            speakTurn(turnId);
            return;
        }
        const token = fieldValue(IDS.token);
        if (!token || token === lastToken) return;
        lastToken = token;
        speakCompleted(token);
    }

    // The non-streaming fallback, kept whole from V1: one completed reply, one
    // WAV, decoded and played. It is the honest answer for a deployment where
    // streaming cannot work -- section 40 says to fall back visibly rather than
    // to claim streaming while an intermediary buffers the whole response.
    function speakCompleted(token) {
        if (speaking) return;
        speaking = true;
        setVoiceBusy(true);
        unlock().then(function (unlocked) {
            if (!unlocked) {
                speaking = false;
                setVoiceBusy(false);
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
                setVoiceBusy(false);
            });
        }).catch(function () {
            speaking = false;
            setVoiceBusy(false);
            say(MESSAGES.speakFailed, "warn");
        });
    }

    // -- the Voice engine panel ------------------------------------------------ //

    // Section 32. The flyout distinguishes "installed" from "in RAM right now",
    // which the V1 panel could not: Load and Unload are buttons in an HTML block
    // that Python renders, wired here, exactly as the Settings install rows are.

    let enginePoll = 0;

    function engineHolder() {
        return byId(IDS.engine);
    }

    function paintEngine(payload) {
        const holder = engineHolder();
        if (!holder || !payload) return;
        const engine = payload.engine || {};
        const line = holder.querySelector("[data-mc-voice-engine-line]");
        const button = holder.querySelector("[data-mc-voice-runtime]");
        const states = {
            unloaded: ["\u25cb", "Unloaded — loads automatically on next voice use"],
            loading: ["\u25cc", "Loading speech models…"],
            idle: ["\u25cf", "Loaded — CPU, idle"],
            stt: ["\u25cf", "Loaded — Listening"],
            tts: ["\u25cf", "Loaded — Speaking"],
            stopping: ["\u25cc", "Unloading…"],
            error: ["\u25cf", engine.error || "The speech engine could not start"],
        };
        const shown = states[engine.state] || states.unloaded;
        if (line) line.textContent = shown[0] + " " + shown[1];
        if (button) {
            const busyState = engine.state === "loading" || engine.state === "stopping";
            button.textContent = engine.loaded ? "Unload" : "Load";
            button.setAttribute("data-mc-voice-runtime", engine.loaded ? "unload" : "load");
            button.disabled = busyState || !payload.ready;
        }
        const voice = holder.querySelector("[data-mc-voice-default]");
        if (voice && payload.voice) {
            voice.textContent = payload.voice.label || payload.voice.name || "";
        }
    }

    function pollEngine() {
        const holder = engineHolder();
        window.clearTimeout(enginePoll);
        // Section 33: no permanent high-frequency poll. Nothing is asked for at
        // all while the flyout is closed or the tab is hidden.
        if (!holder || holder.offsetParent === null || document.hidden) return;
        refreshStatus(true).then(function (payload) {
            paintEngine(payload);
            const state = (payload && payload.engine && payload.engine.state) || "unloaded";
            const quick = state === "loading" || state === "stopping" || state === "tts"
                || state === "stt";
            enginePoll = window.setTimeout(pollEngine, quick ? 1000 : 4000);
        }).catch(function () {
            enginePoll = window.setTimeout(pollEngine, 5000);
        });
    }

    function wireEngine() {
        const holder = engineHolder();
        if (!holder || holder.dataset.mcVoiceWired) return;
        holder.dataset.mcVoiceWired = "1";
        holder.addEventListener("click", function (event) {
            const button = event.target.closest("[data-mc-voice-runtime]");
            if (!button) return;
            event.preventDefault();
            const action = button.getAttribute("data-mc-voice-runtime");
            // Unloading while a reply is being read aloud cancels it. The panel
            // says so by simply going quiet -- section 31.
            if (action === "unload") stopSpeaking(true);
            button.disabled = true;
            unlock();
            fetch(url(ROUTES.runtime), {
                method: "POST",
                credentials: "same-origin",
                headers: headers({"Content-Type": "application/json"}),
                body: JSON.stringify({action: action}),
            }).then(function (response) {
                return response.json().catch(function () { return {ok: false}; });
            }).then(function (payload) {
                if (payload && payload.error) say(payload.error, "warn");
                pollEngine();
            }).catch(function () {
                say("The speech engine could not be changed.", "warn");
                pollEngine();
            });
        });
        pollEngine();
    }

    // -- the Settings page row ------------------------------------------------ //

    // The engine is a row too, and is painted by the same loop. It has no
    // "install from a folder" half -- it comes from PyPI or not at all.
    const KINDS = ["runtime", "stt", "tts"];

    function installLabel(kind) {
        return kind === "runtime" ? "Install voice engine"
            : "Download default " + kind.toUpperCase();
    }

    function settingsLine(holder, kind) {
        return holder.querySelector('[data-mc-voice-status="' + kind + '"]');
    }

    function installButton(holder, kind) {
        return holder.querySelector('[data-mc-voice-install="' + kind + '"]');
    }

    function localButton(holder, kind) {
        return holder.querySelector('[data-mc-voice-local="' + kind + '"]');
    }

    function sayInRow(holder, kind, text, bad) {
        const line = settingsLine(holder, kind);
        if (!line) return;
        line.textContent = text;
        if (line.classList) {
            if (bad) line.classList.add("mc-voice-failed");
            else line.classList.remove("mc-voice-failed");
        }
    }

    // Whatever went wrong, *both* buttons come back. A control that disabled
    // itself on the way into a request it never got an answer to is a control
    // somebody sits and watches -- which is exactly what happened, for minutes,
    // twice: the second time because this only ever repainted the primary
    // button and left "Install from this folder" saying "Starting…" forever.
    function releaseButton(holder, kind, ready) {
        const button = installButton(holder, kind);
        if (button) {
            button.disabled = !!ready;
            button.textContent = ready ? "Installed" : installLabel(kind);
        }
        const local = localButton(holder, kind);
        if (local) {
            local.disabled = false;
            local.textContent = ready ? "Reinstall from a folder" : "Install from this folder";
        }
    }

    function wireSettings() {
        const holder = document.querySelector(".mc-voice-settings")
            || root().querySelector(".mc-voice-settings");
        if (!holder || holder.dataset.mcVoiceWired === "1") return;
        holder.dataset.mcVoiceWired = "1";

        Array.prototype.forEach.call(
            holder.querySelectorAll("[data-mc-voice-install]"), function (button) {
                button.addEventListener("click", function (event) {
                    if (event.preventDefault) event.preventDefault();
                    startInstall(holder, button.getAttribute("data-mc-voice-install"),
                                 button, "");
                });
            });

        // The other way in: files somebody already has. Same route, same row,
        // one extra field.
        Array.prototype.forEach.call(
            holder.querySelectorAll("[data-mc-voice-local]"), function (button) {
                button.addEventListener("click", function (event) {
                    if (event.preventDefault) event.preventDefault();
                    const kind = button.getAttribute("data-mc-voice-local");
                    const box = holder.querySelector(
                        '[data-mc-voice-folder="' + kind + '"]');
                    const folder = box ? (box.value || "").trim() : "";
                    if (!folder) {
                        sayInRow(holder, kind, "Type the folder the downloaded files are "
                                 + "in, then press this again.", true);
                        return;
                    }
                    startInstall(holder, kind, button, folder);
                });
            });

        schedulePaint(holder, 0);
    }

    // The poll used to be a fixed 1.5 seconds, forever, whatever happened. On a
    // WebUI that was refusing the request that meant a warning a second in the
    // console for as long as the page stayed open -- a hundred and thirty-six
    // of them in three minutes, burying the one line that explained it. So the
    // cadence follows what is actually happening: fast while a download is
    // running, slow when nothing is, backing off when it is failing, and not at
    // all while the tab is in the background.
    let paintTimer = 0;
    let paintFailures = 0;

    function nextPaintDelay(payload) {
        if (!payload || !payload.ok) {
            paintFailures += 1;
            return Math.min(SETTINGS_POLL_MS * Math.pow(2, paintFailures),
                            SETTINGS_POLL_MAX_MS);
        }
        paintFailures = 0;
        const progress = payload.progress || {};
        const busy = KINDS.some(function (kind) {
            return (progress[kind] || {}).running;
        });
        return busy ? SETTINGS_POLL_MS : SETTINGS_IDLE_MS;
    }

    function schedulePaint(holder, delay) {
        if (paintTimer) window.clearTimeout(paintTimer);
        paintTimer = window.setTimeout(function () {
            if (document.hidden) {
                schedulePaint(holder, SETTINGS_IDLE_MS);
                return;
            }
            paintSettings(holder).then(function (payload) {
                schedulePaint(holder, nextPaintDelay(payload));
            });
        }, delay);
    }

    function startInstall(holder, kind, button, folder) {
        button.disabled = true;
        button.textContent = "Starting…";
        sayInRow(holder, kind, "Starting…", false);

        const failed = function (text) {
            sayInRow(holder, kind, text, true);
            releaseButton(holder, kind, false);
            button.disabled = false;
            console.error("Model Chain: Voice Chat could not start the " + kind
                          + " install — " + text);
        };

        fetch(url(ROUTES.install), {
            method: "POST",
            credentials: "same-origin",
            headers: headers({"Content-Type": "application/json"}, holder),
            body: JSON.stringify(folder ? {kind: kind, folder: folder} : {kind: kind}),
        }).then(function (response) {
            return response.json().catch(function () {
                return {ok: false, error: "The WebUI answered HTTP " + response.status
                                          + " with something this page cannot read."};
            });
        }).then(function (payload) {
            // The answer is read, not discarded. A build that cannot install
            // anything says so in this reply, and saying so is the difference
            // between a row that explains itself and a row that never moves.
            if (!payload || !payload.ok) {
                failed((payload && payload.error)
                       || "Voice Chat could not start that install.");
                return;
            }
            paintFailures = 0;
            schedulePaint(holder, 0);
        }).catch(function (error) {
            failed("Could not reach this WebUI to start the install ("
                   + ((error && error.message) || "network error") + ").");
        });
    }

    function paintSettings(holder) {
        return refreshStatus(true, holder).then(function (payload) {
            const runtime = holder.querySelector(".mc-voice-runtime");
            if (!payload || !payload.ok) {
                // The failure is drawn rather than swallowed. This is where the
                // row used to give up silently and leave both buttons disabled.
                const text = (payload && payload.error)
                    || "Voice Chat could not read its own status.";
                if (runtime) runtime.textContent = text;
                KINDS.forEach(function (kind) {
                    sayInRow(holder, kind, text, true);
                    releaseButton(holder, kind, false);
                });
                return payload;
            }
            if (runtime) runtime.textContent = payload.summary || "";
            KINDS.forEach(function (kind) {
                const progress = (payload.progress || {})[kind] || {};
                const ready = kind === "runtime" ? payload.runtime_ready
                    : kind === "stt" ? payload.stt_ready : payload.tts_ready;
                const message = kind === "runtime" ? payload.runtime_message
                    : kind === "stt" ? payload.stt_message : payload.tts_message;
                const button = installButton(holder, kind);

                if (progress.running) {
                    sayInRow(holder, kind,
                             (progress.text || "Working…") + "  "
                             + Math.round((progress.fraction || 0) * 100) + "%", false);
                    if (button) {
                        button.disabled = true;
                        button.textContent = "Installing…";
                    }
                    const busy = localButton(holder, kind);
                    if (busy) {
                        busy.disabled = true;
                        busy.textContent = "Installing…";
                    }
                    return;
                }
                sayInRow(holder, kind, progress.failed ? progress.text : message,
                         !!progress.failed);
                releaseButton(holder, kind, ready);
            });
            return payload;
        });
    }

    // -- wiring --------------------------------------------------------------- //

    function wireGestures() {
        const chip = byId(IDS.chip);
        if (chip && !chip.dataset.mcVoiceUnlock) {
            chip.dataset.mcVoiceUnlock = "1";
            chip.addEventListener("click", function () {
                unlock();
                refreshStatus(true);
                // The flyout is about to be visible, so its live engine state
                // starts being worth asking for. Deferred one tick because
                // Gradio has not made it visible yet at click time.
                window.setTimeout(function () { attempt("poll the voice engine", pollEngine); },
                                  120);
            });
        }
        ["mc-llm-chat-voice-auto-send", IDS.autoSpeak].forEach(function (id) {
            const holder = byId(id);
            if (!holder || holder.dataset.mcVoiceUnlock) return;
            holder.dataset.mcVoiceUnlock = "1";
            holder.addEventListener("change", function (event) {
                unlock();
                // Section 28. Turning Auto Speak off while a reply is being read
                // aloud stops it, now -- leaving the current reply talking after
                // the switch says it should not would make the switch a lie for
                // however long the reply lasts. Turning it *on* deliberately does
                // nothing to the reply in flight: it applies from the next turn.
                if (id !== IDS.autoSpeak) return;
                // The event's own target where there is one, and the component
                // otherwise: Gradio dispatches change events at the wrapper on
                // some paths, and a switch that only worked on one of them is a
                // switch that works until the next Gradio update.
                const box = event.target || clickable(id) || byId(id);
                const checked = box && (box.checked !== undefined
                    ? box.checked : box.getAttribute("aria-checked") === "true");
                if (checked === false) stopSpeaking(true);
            });
        });
        wireStop();
        wireEngine();
    }

    // -- the one Stop ---------------------------------------------------------- //

    // Section 26. Both halves of Stop are deliberate and neither replaces the
    // other: this listener gives immediate silence in the browser, and the
    // Gradio click handler that is already on the same button cancels the LLM
    // run and the server-side turn. Attached in the capture phase so the speaker
    // stops on the press rather than after a round trip -- the target is under
    // 100 ms and a round trip to a phone over a VPN is not.
    function wireStop() {
        const button = clickable(IDS.stop);
        if (!button || button.dataset.mcVoiceStop) return;
        button.dataset.mcVoiceStop = "1";
        button.addEventListener("click", function () {
            attempt("stop the voice", function () { stopSpeaking(true); });
        }, true);
    }

    // Gradio rewrites the run-state field when a reply starts and ends, and the
    // composer has to follow it. An observer rather than a poll because the
    // interesting moments are exactly the writes.
    let stateWatcher = null;

    function watchRunState() {
        const holder = byId(IDS.runState);
        if (!holder || stateWatcher || typeof MutationObserver !== "function") return;
        stateWatcher = new MutationObserver(function () {
            attempt("update the composer", applyComposerState);
        });
        try {
            stateWatcher.observe(holder, {subtree: true, childList: true,
                                          attributes: true, characterData: true});
        } catch (error) {
            stateWatcher = null;
        }
    }

    // -- the Settings voice list ---------------------------------------------- //

    // Section 71. The list, Test, Set as Default, Rename and Delete, and the
    // optional cloning panel under it. Painted from `/voice/voices` rather than
    // rendered by Python, so a clone that finishes while this page is open shows
    // up on the next paint instead of needing a reload.
    //
    // The asterisk on a custom voice is added *here*, at the moment of drawing.
    // It is not in the stored name, not in the id and not in any filename --
    // section 41 -- so renaming a clone to "Alice" beside an official "Alice"
    // is a display question and nothing more.

    let voicesTimer = 0;
    let auditioning = null;

    function voicesHolder() {
        return document.querySelector(".mc-voice-voices");
    }

    function post(route, body, holder) {
        return fetch(url(route), {
            method: "POST",
            credentials: "same-origin",
            headers: headers({"Content-Type": "application/json"}, holder),
            body: JSON.stringify(body || {}),
        }).then(function (response) {
            return response.json().catch(function () {
                return {ok: false, error: "The WebUI did not answer."};
            });
        });
    }

    function voiceRow(entry, current) {
        const row = document.createElement("div");
        row.className = "mc-voice-entry" + (entry.id === current ? " mc-voice-entry-default" : "");
        row.setAttribute("data-mc-voice-id", entry.id);
        const name = document.createElement("span");
        name.className = "mc-voice-entry-name";
        name.textContent = entry.label || entry.display_name;
        const kind = document.createElement("span");
        kind.className = "mc-voice-entry-kind";
        kind.textContent = entry.official ? "Official" : "Custom";
        row.appendChild(name);
        row.appendChild(kind);
        const actions = [["test", "Test"], ["default", "Set as Default"]];
        if (entry.editable) actions.push(["rename", "Rename"]);
        if (entry.deletable) actions.push(["delete", "Delete"]);
        actions.forEach(function (pair) {
            const button = document.createElement("button");
            button.type = "button";
            button.className = "mc-voice-entry-action";
            button.setAttribute("data-mc-voice-action", pair[0]);
            button.textContent = pair[1];
            if (pair[0] === "default" && entry.id === current) button.disabled = true;
            row.appendChild(button);
        });
        return row;
    }

    function paintVoices(holder, payload) {
        if (!holder || !payload) return;
        const list = holder.querySelector("[data-mc-voice-list]");
        const current = payload.default || "";
        if (list) {
            list.textContent = "";
            const groups = [
                ["Official — American English",
                 payload.voices.filter(function (v) { return v.official && v.language === "en-US"; })],
                ["Official — British English",
                 payload.voices.filter(function (v) { return v.official && v.language === "en-GB"; })],
                ["Custom", payload.voices.filter(function (v) { return !v.official; })],
            ];
            groups.forEach(function (group) {
                if (!group[1].length) return;
                const heading = document.createElement("div");
                heading.className = "mc-voice-group";
                heading.textContent = group[0];
                list.appendChild(heading);
                group[1].forEach(function (entry) {
                    list.appendChild(voiceRow(entry, current));
                });
            });
        }
        const chosen = payload.voices.filter(function (v) { return v.id === current; })[0];
        const label = holder.querySelector("[data-mc-voice-current]");
        if (label) {
            label.textContent = chosen
                ? "Default voice: " + (chosen.label || chosen.display_name)
                : "No voice is installed yet.";
        }
        const text = holder.querySelector("[data-mc-voice-test-text]");
        if (text && document.activeElement !== text && payload.test_text !== undefined) {
            text.value = payload.test_text;
        }
        const warnings = holder.querySelector("[data-mc-voice-warnings]");
        if (warnings) {
            warnings.textContent = (payload.warnings || []).join(" ");
            warnings.hidden = !(payload.warnings || []).length;
        }
    }

    function audition(holder, voiceId) {
        const text = holder.querySelector("[data-mc-voice-test-text]");
        if (auditioning) {
            try { auditioning.stop(); } catch (error) { /* already finished */ }
            auditioning = null;
        }
        unlock();
        return fetch(url(ROUTES.voiceTest), {
            method: "POST",
            credentials: "same-origin",
            headers: headers({"Content-Type": "application/json"}, holder),
            body: JSON.stringify({voice: voiceId, text: text ? text.value : ""}),
        }).then(function (response) {
            if (!response.ok) throw new Error("test");
            return response.arrayBuffer();
        }).then(function (buffer) {
            // The same in-memory path a spoken reply takes. No Blob URL, no
            // audio element with a src, no file -- section 45 asks for a Test
            // that does not accumulate anything, and this one cannot.
            return play(buffer);
        });
    }

    function wireVoices() {
        const holder = voicesHolder();
        if (!holder || holder.dataset.mcVoiceWired) return;
        holder.dataset.mcVoiceWired = "1";

        holder.addEventListener("click", function (event) {
            const action = event.target.closest("[data-mc-voice-action]");
            if (action) {
                event.preventDefault();
                const row = action.closest("[data-mc-voice-id]");
                const voiceId = row ? row.getAttribute("data-mc-voice-id") : "";
                const kind = action.getAttribute("data-mc-voice-action");
                if (kind === "test") {
                    action.disabled = true;
                    audition(holder, voiceId).catch(function () {
                        sayInHolder(holder, "That voice could not be played.");
                    }).then(function () { action.disabled = false; });
                    return;
                }
                if (kind === "default") {
                    post(ROUTES.voiceDefault, {voice: voiceId}, holder)
                        .then(function (payload) { applyVoices(holder, payload); });
                    return;
                }
                if (kind === "rename") {
                    const row2 = action.closest("[data-mc-voice-id]");
                    const shown = row2 ? row2.querySelector(".mc-voice-entry-name") : null;
                    const was = shown ? shown.textContent.replace(/^\*\s*/, "") : "";
                    const wanted = window.prompt("New name for this voice", was);
                    if (!wanted) return;
                    post(ROUTES.voiceRename, {voice: voiceId, display_name: wanted}, holder)
                        .then(function (payload) { applyVoices(holder, payload); });
                    return;
                }
                if (kind === "delete") {
                    if (!window.confirm("Delete this voice? This cannot be undone.")) return;
                    action.disabled = true;
                    post(ROUTES.voiceDelete, {voice: voiceId}, holder)
                        .then(function (payload) { applyVoices(holder, payload); });
                }
                return;
            }
            const adopt = event.target.closest("[data-mc-voice-cloning-adopt]");
            if (adopt) {
                event.preventDefault();
                const folder = holder.querySelector("[data-mc-voice-cloning-folder]");
                adopt.disabled = true;
                post(ROUTES.cloningInstall, {folder: folder ? folder.value : ""}, holder)
                    .then(function (payload) {
                        adopt.disabled = false;
                        applyCloning(holder, payload);
                    });
                return;
            }
            if (event.target.closest("[data-mc-voice-clone-start]")) {
                event.preventDefault();
                startClone(holder);
                return;
            }
            if (event.target.closest("[data-mc-voice-clone-abort]")) {
                event.preventDefault();
                post(ROUTES.cloningAbort, {}, holder).then(function (payload) {
                    applyCloning(holder, payload);
                });
            }
        });

        const text = holder.querySelector("[data-mc-voice-test-text]");
        if (text) {
            // Saved when the field is left rather than on every keystroke: this
            // is a setting, and one write per character would be one config file
            // save per character.
            text.addEventListener("change", function () {
                post(ROUTES.voices, {test_text: text.value}, holder);
            });
        }
        refreshVoices(holder);
    }

    function sayInHolder(holder, text) {
        const warnings = holder.querySelector("[data-mc-voice-warnings]");
        if (!warnings) return;
        warnings.textContent = text;
        warnings.hidden = false;
    }

    function applyVoices(holder, payload) {
        if (payload && payload.error) sayInHolder(holder, payload.error);
        if (payload && payload.voices) paintVoices(holder, payload);
    }

    function refreshVoices(holder) {
        post(ROUTES.voices, {}, holder).then(function (payload) {
            applyVoices(holder, payload);
        }).catch(function () { /* the next paint tries again */ });
        refreshCloning(holder);
    }

    // -- cloning --------------------------------------------------------------- //

    function refreshCloning(holder) {
        window.clearTimeout(voicesTimer);
        post(ROUTES.cloningStatus, {}, holder).then(function (payload) {
            applyCloning(holder, payload);
            const active = payload && payload.job && payload.job.active;
            if (active) {
                // While a clone runs. Not otherwise -- section 33's rule about
                // permanent polls applies to this page too.
                voicesTimer = window.setTimeout(function () { refreshCloning(holder); }, 2000);
            }
        }).catch(function () { /* a Settings page that cannot reach the WebUI is not fatal */ });
    }

    function applyCloning(holder, payload) {
        if (!holder || !payload) return;
        if (payload.error) sayInHolder(holder, payload.error);
        const status = holder.querySelector("[data-mc-voice-cloning-status]");
        if (status) status.textContent = payload.message || "";
        const checks = holder.querySelector("[data-mc-voice-cloning-checks]");
        if (checks) {
            checks.textContent = "";
            (payload.checks || []).forEach(function (item) {
                const line = document.createElement("div");
                line.className = "mc-voice-check" + (item.ok ? " mc-voice-check-ok" : "");
                line.textContent = (item.ok ? "\u2713 " : "\u2717 ") + item.item
                    + (item.detail ? " — " + item.detail : "");
                checks.appendChild(line);
            });
        }
        const links = holder.querySelector("[data-mc-voice-cloning-links]");
        if (links && !links.children.length && payload.sources) {
            [["Storytime", payload.sources.upstream], ["Kokoro", payload.sources.kokoro]]
                .forEach(function (pair) {
                    if (!pair[1]) return;
                    const item = document.createElement("li");
                    const link = document.createElement("a");
                    link.href = pair[1];
                    link.target = "_blank";
                    link.rel = "noreferrer";
                    link.textContent = pair[0];
                    item.appendChild(link);
                    links.appendChild(item);
                });
        }
        const job = (payload.job || {});
        const form = holder.querySelector("[data-mc-voice-clone-form]");
        const panel = holder.querySelector("[data-mc-voice-clone-job]");
        const full = payload.capacity && payload.capacity.free === 0;
        if (form) form.hidden = payload.state !== "installed" || job.active || full;
        if (panel) panel.hidden = !job.status || job.status === "idle";
        if (full && status) {
            status.textContent = "Custom voice capacity is full. Delete a custom voice "
                + "before creating another.";
        }
        if (!panel) return;
        setText(panel, "[data-mc-voice-job-name]", job.label || "");
        setText(panel, "[data-mc-voice-job-state]", cloneState(job));
        const bar = panel.querySelector("[data-mc-voice-job-bar]");
        if (bar) bar.style.width = Math.max(0, Math.min(100, job.percent || 0)) + "%";
        setText(panel, "[data-mc-voice-job-step]",
                job.step ? "Step " + job.step + " / " + job.total_steps : "");
        const abort = panel.querySelector("[data-mc-voice-clone-abort]");
        if (abort) abort.hidden = !job.active;
        if (job.status === "complete") refreshVoices(holder);
    }

    function cloneState(job) {
        const words = {
            preparing: "Preparing", cloning: "Cloning",
            validating_source: "Checking the new voice",
            building_bank: "Installing the new voice",
            validating_runtime: "Testing the new voice",
            complete: "Done", failed: "Failed", aborting: "Stopping",
            aborted: "Stopped", interrupted: "Interrupted when the WebUI closed",
        };
        const word = words[job.status] || "";
        return job.error ? word + " — " + job.error : word;
    }

    function setText(root, selector, value) {
        const node = root.querySelector(selector);
        if (node) node.textContent = value;
    }

    function startClone(holder) {
        const name = holder.querySelector("[data-mc-voice-clone-name]");
        const language = holder.querySelector("[data-mc-voice-clone-language]");
        const file = holder.querySelector("[data-mc-voice-clone-file]");
        const chosen = file && file.files && file.files[0];
        if (!chosen) {
            sayInHolder(holder, "Choose a WAV recording first.");
            return;
        }
        const form = new FormData();
        form.append("name", name ? name.value : "");
        form.append("language", language ? language.value : "en-US");
        form.append("reference", chosen);
        fetch(url(ROUTES.cloningStart), {
            method: "POST",
            credentials: "same-origin",
            // No Content-Type: the browser has to set the multipart boundary.
            headers: headers(null, holder),
            body: form,
        }).then(function (response) {
            // 415 means this host cannot parse a multipart body at all. The
            // recording is the same recording; only its encoding changes.
            if (response.status === 415) return sendCloneAsJson(holder, chosen, name, language);
            return response.json().catch(function () { return {ok: false}; });
        }).then(function (payload) {
            applyCloning(holder, payload);
            refreshCloning(holder);
        }).catch(function () {
            sayInHolder(holder, "The clone could not be started.");
        });
    }

    function sendCloneAsJson(holder, chosen, name, language) {
        return chosen.arrayBuffer().then(function (buffer) {
            const bytes = new Uint8Array(buffer);
            let binary = "";
            // In blocks, because `apply` on a megabyte-long array overflows the
            // argument list on several browsers.
            for (let index = 0; index < bytes.length; index += 0x8000) {
                binary += String.fromCharCode.apply(
                    null, bytes.subarray(index, index + 0x8000));
            }
            return fetch(url(ROUTES.cloningStart), {
                method: "POST",
                credentials: "same-origin",
                headers: headers({"Content-Type": "application/json"}, holder),
                body: JSON.stringify({
                    name: name ? name.value : "",
                    language: language ? language.value : "en-US",
                    reference: window.btoa(binary),
                }),
            }).then(function (response) {
                return response.json().catch(function () { return {ok: false}; });
            });
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
        attempt("wire the voice list", wireVoices);
        attempt("watch the run state", watchRunState);
        attempt("update the composer", applyComposerState);
    }

    let watching = false;

    function watch() {
        if (watching) return;
        watching = true;
        window.setInterval(function () {
            attempt("check for a reply to speak", checkSpeech);
            attempt("update the composer", applyComposerState);
        }, TOKEN_POLL_MS);
        // Leaving the page with the speaker running is not something a user asks
        // for, and a suspended tab that resumes and starts talking is worse. The
        // server is told too, with a keepalive request, so a closed tab does not
        // leave Kokoro synthesising for nobody.
        const leave = function () { attempt("stop the voice", function () { stopSpeaking(true); }); };
        window.addEventListener("pagehide", leave);
        window.addEventListener("beforeunload", leave);
        document.addEventListener("visibilitychange", function () {
            if (document.hidden) {
                window.clearTimeout(enginePoll);
            } else {
                attempt("poll the voice engine", pollEngine);
            }
        });
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
