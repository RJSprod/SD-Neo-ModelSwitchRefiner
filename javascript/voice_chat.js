// Model Chain -- Voice Chat.
//
// The browser half of a feature whose other half is a CPU speech worker on the
// Forge machine. The division is the same one llm_studio.js states and for the
// same reason: Python owns models, persistence, inference and conversation
// state; this file owns the things only a browser can do.
//
//   * the slide-to-talk gesture, with pointer capture, so a finger that leaves
//     the track still ends the recording it started;
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
        track: "mc-llm-chat-voice-track",
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
        characterVoice: "mc-llm-chat-character-voice",
        characterVoiceList: "mc-llm-chat-character-voice-list",
        characterVoiceCustom: "mc-llm-chat-character-voice-custom",
    };

    const ROUTES = {
        status: "model-chain/voice/status",
        stt: "model-chain/voice/stt",
        tts: "model-chain/voice/tts",
        stream: "model-chain/voice/tts-stream",
        cancel: "model-chain/voice/cancel",
        telemetry: "model-chain/voice/telemetry",
        runtime: "model-chain/voice/runtime",
        install: "model-chain/voice/install",
        models: "model-chain/voice/models",
        profile: "model-chain/voice/profile",
        voices: "model-chain/voice/voices",
        voiceDefault: "model-chain/voice/voice/default",
        voiceTest: "model-chain/voice/voice/test",
        voiceRename: "model-chain/voice/voice/rename",
        voiceDelete: "model-chain/voice/voice/delete",
        cloningStatus: "model-chain/voice/cloning/status",
        cloningInstall: "model-chain/voice/cloning/install",
        cloningStart: "model-chain/voice/cloning/start",
        cloningAbort: "model-chain/voice/cloning/abort",
        engines: "model-chain/voice/engines",
        engineSelect: "model-chain/voice/engine/select",
        surface: "model-chain/voice/surface",
        sopro: "model-chain/voice/sopro",
        soproInstall: "model-chain/voice/sopro/install",
        soproSettings: "model-chain/voice/sopro/settings",
        soproClone: "model-chain/voice/sopro/clone",
        soproSave: "model-chain/voice/sopro/save",
        soproDiscard: "model-chain/voice/sopro/discard",
        soproRebuild: "model-chain/voice/sopro/rebuild",
        soproStarter: "model-chain/voice/sopro/starter",
        soproValidate: "model-chain/voice/sopro/validate",
        cleanup: "model-chain/voice/cleanup",
        cleanupInstall: "model-chain/voice/cleanup/install",
        cleanupRun: "model-chain/voice/cleanup/run",
        lab: "model-chain/voice/lab",
        labUpdate: "model-chain/voice/lab/update",
        labReset: "model-chain/voice/lab/reset",
        labPlay: "model-chain/voice/lab/play",
    };

    // 250 ms. Shorter than this is a tap, and a tap is somebody finding out what
    // the button does -- transcribing a tenth of a second of room tone and
    // dropping the result in their composer would be a worse answer than a
    // sentence saying how the control works.
    const MIN_HOLD_MS = 250;
    // Sixty seconds, then stop and transcribe what there is. This bounds memory
    // and it bounds surprise: a button held down in a pocket is not a five
    // minute upload. It is counted from the first sample, not from the
    // engagement: a second spent opening a Bluetooth microphone is the device's
    // and not the user's.
    const MAX_HOLD_MS = 60000;
    // And the other half of that split. A getUserMedia that never settles --
    // a permission prompt nobody answers, a device another application has
    // exclusive -- must not leave the control pinned at the right for ever.
    // Generous, because a permission prompt is a human deciding.
    const OPEN_TIMEOUT_MS = 20000;
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
        tooShort: "Hold at the right-hand end to talk.",
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
        // The same fact, when the browser has told us *why*. Reported
        // separately because "not available" is not something anybody can act
        // on and this is: a phone reaching a WebUI at http://192.168.x.x is
        // the ordinary way to arrive here, and it is the one case where the
        // browser is withholding the microphone for a reason with a remedy.
        insecure: "Your browser only opens a microphone on a secure page, and this address "
            + "is not one. Reach this WebUI over HTTPS, or allow this exact address in "
            + "chrome://flags/#unsafely-treat-insecure-origin-as-secure and restart the "
            + "browser.",
        slide: "Slide the microphone all the way to the right, and hold, to talk.",
        // The Bluetooth answer, and it is deliberately about the microphone
        // rather than about Voice Chat. See the block above `startCapture`.
        tooQuiet: "That recording was almost silent, so nothing was sent. A Bluetooth "
            + "headset microphone is narrowband and much quieter than the phone's own — "
            + "switching back to the phone's microphone usually fixes it.",
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

    // The control inside one of Python's hidden holders, or null. Split out of
    // `fieldValue` because two things now want it: reading the value, and
    // binding a listener to the element that emits `input` when Gradio writes.
    function fieldOf(id) {
        const holder = byId(id);
        if (!holder) return null;
        return holder.tagName === "TEXTAREA" || holder.tagName === "INPUT"
            ? holder
            : holder.querySelector("textarea, input");
    }

    function fieldValue(id) {
        const field = fieldOf(id);
        return field ? (field.value || "") : "";
    }

    // One monotonic clock for every duration this file measures.
    //
    // `Date.now()` is a wall clock: it moves when the machine syncs time, when
    // a laptop wakes, and when somebody changes the timezone -- so a recording
    // held for 300 ms could be measured as -400 ms, and a phone that adjusted
    // its clock mid-utterance would refuse the recording as too short. Every
    // *duration* below therefore comes from `performance.now()`, and the only
    // remaining use of the wall clock is the readiness cache, where an absolute
    // moment is genuinely what is wanted.
    //
    // These numbers never leave the browser and are never compared with the
    // server's own monotonic clock: two monotonic clocks in two processes share
    // no origin, and subtracting one from the other produces a number that
    // looks like a latency and is not one.
    function nowMs() {
        try {
            if (typeof performance === "object" && performance
                    && typeof performance.now === "function") {
                return performance.now();
            }
        } catch (error) { /* a browser without it falls through */ }
        return Date.now();
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

    // -- a page that is no longer this WebUI's -------------------------------- //

    // Every voice route answers 403 for exactly two reasons, and both of them
    // mean the same thing about this page: it cannot talk to this WebUI any
    // more. The page token is minted per WebUI process, so a tab that was open
    // when Forge restarted carries last run's token; and a request that did not
    // come from this WebUI is refused on origin.
    //
    // What that used to cost: the tab kept polling. Every 1.5 seconds, for as
    // long as it stayed open, against a WebUI that had just started -- which is
    // a warning in somebody's console during startup, a socket out of the
    // browser's small per-origin pool while the *new* tab is trying to load,
    // and no possibility of the answer ever changing. So the first 403 stops
    // all of it, once, with a sentence that says what to do.
    let stale = false;

    const RELOAD = "This page was loaded before the WebUI restarted. Reload it.";

    function markStale() {
        if (stale) return;
        stale = true;
        window.clearTimeout(paintTimer);
        window.clearTimeout(enginePoll);
        window.clearTimeout(voicesTimer);
        console.warn("Model Chain: Voice Chat stopped asking this WebUI anything — "
                     + RELOAD);
    }

    // The text-to-speech engine changed under this page -- somebody switched it
    // in another tab, or in this one before a request already in flight came
    // back. The document was built for an engine that is no longer selected and
    // every control in it belongs to that engine, so repainting it is not a fix:
    // the whole surface has to be replaced with the other engine's.
    //
    // Driven by the server's own `engine_mismatch` flag rather than by the
    // status code: 409 already means several other things on these routes.
    function engineChanged(payload) {
        if (!payload || !payload.engine_mismatch) return false;
        swapSurface();
        return true;
    }

    // Replace the settings surface with markup built for the engine that is
    // selected now.
    //
    // Reloading used to stand here, on the reasoning that the cheapest way to
    // be certain the inactive engine's controls are absent is to ask the server
    // for a document that never contained them. That reasoning was wrong about
    // this host: Forge builds a settings row's HTML once, when the extension is
    // imported, and serves that same string for the life of the process. A
    // reload therefore came back to the *same* stale markup, which was still
    // for the wrong engine, which reloaded -- a loop that only stopped when the
    // tab was closed.
    //
    // So the nodes are replaced instead. That is the same guarantee by a
    // different route: what is in the document afterwards was built for the
    // engine selected now, and the other engine's controls are gone from it
    // because they were removed, not hidden.
    //
    // One swap at a time, because several polls can be in flight at the moment
    // of a switch and each of them will notice the same mismatch -- and no more
    // than one every few seconds, whatever happens after that.
    //
    // The cooldown is the bound the reload never had. Replacing the surface
    // resolves the mismatch that asked for it, so in a browser one swap is the
    // end of it; the cooldown is there for the case where it somehow is not,
    // and it turns "ask again on the next poll, forever" into a request every
    // three seconds rather than a page that reloads until the tab is closed.
    // An explicit press bypasses it: somebody who just chose an engine is owed
    // the surface for it now.
    let swapping = false;
    let swappedAt = -Infinity;

    const SWAP_COOLDOWN_MS = 3000;

    function swapSurface(pressed) {
        if (stale || swapping) return Promise.resolve(false);
        if (!pressed && nowMs() - swappedAt < SWAP_COOLDOWN_MS) {
            return Promise.resolve(false);
        }
        swappedAt = nowMs();
        swapping = true;
        // The three timers that paint nodes about to be thrown away. Not
        // `enginePoll`: the Voice flyout is not part of this surface, it is not
        // replaced, and stopping its residency line would be a second bug.
        window.clearTimeout(paintTimer);
        window.clearTimeout(voicesTimer);
        window.clearTimeout(soproTimer);
        return post(ROUTES.surface, {}).then(function (payload) {
            if (!payload || !payload.ok || !payload.settings) return false;
            [[".mc-voice-settings", payload.settings],
             [".mc-voice-voices", payload.voices]].forEach(function (pair) {
                const node = document.querySelector(pair[0]);
                if (node && pair[1]) node.outerHTML = pair[1];
            });
            const name = document.querySelector("[data-mc-voice-engine-name]");
            if (name && payload.engine_label) name.textContent = payload.engine_label;
            attempt("wire the Voice Chat settings", wireSettings);
            attempt("wire the voice list", wireVoices);
            return true;
        }).catch(function () {
            return false;
        }).then(function (done) {
            swapping = false;
            return done;
        });
    }

    function refused(response) {
        if (response && response.status === 403) markStale();
        return response;
    }

    // Whether a row is on screen at all. Nothing is fetched for one that is
    // not: a settings row that fetched on page load put three requests in front
    // of the first paint of every tab in this WebUI, including the ones with
    // nothing to do with speech.
    function onScreen(node) {
        if (!node) return false;
        if (node.offsetParent !== null) return true;
        try {
            const box = node.getBoundingClientRect();
            return !!(box && (box.width || box.height));
        } catch (error) {
            return false;
        }
    }

    // Runs `job` the first time `node` is actually visible, and looks again
    // once a second until then. A DOM read costs nothing; the request it is
    // standing in front of does not.
    function whenOnScreen(node, job) {
        const look = function () {
            if (!node || stale) return;
            if (node.isConnected === false) return;
            if (!document.hidden && onScreen(node)) {
                job();
                return;
            }
            window.setTimeout(look, 1000);
        };
        look();
    }

    // -- repainting without moving the page under somebody ------------------- //

    // Emptying a list and refilling it costs the scroll position of whatever is
    // scrolling it, and that is not a subtle effect: the browser clamps
    // `scrollTop` to the content that is there *at that moment*, so a list wiped
    // to nothing clamps to zero, and refilling it afterwards does not put it
    // back. Every poll that repainted a voice list therefore threw somebody back
    // to the top of the flyout, and expanding a section -- which is followed by
    // a repaint -- did it reliably enough to look like the expanding caused it.
    //
    // So every rebuild happens inside this, which notes where the page and the
    // nearest scrolling ancestor were and puts them back afterwards.
    function scrollParent(node) {
        let here = node && node.parentElement;
        while (here) {
            let overflow = "";
            try {
                overflow = window.getComputedStyle(here).overflowY || "";
            } catch (error) { /* a detached node has no style */ }
            if ((overflow === "auto" || overflow === "scroll")
                    && here.scrollHeight > here.clientHeight) {
                return here;
            }
            here = here.parentElement;
        }
        return null;
    }

    function keepingPlace(node, job) {
        const scroller = scrollParent(node);
        const was = scroller ? scroller.scrollTop : 0;
        const pageWas = (typeof window.scrollY === "number") ? window.scrollY : 0;
        try {
            job();
        } finally {
            // After the rebuild, when the content is tall enough to hold the
            // position again.
            if (scroller && scroller.scrollTop !== was) scroller.scrollTop = was;
            if (typeof window.scrollTo === "function" && window.scrollY !== pageWas) {
                try {
                    window.scrollTo(window.scrollX || 0, pageWas);
                } catch (error) { /* a harness without a real window */ }
            }
        }
    }

    // -- what is installed --------------------------------------------------- //

    let readiness = null;
    let readinessAt = 0;
    // The one unforced status request that is currently in flight, so that two
    // callers a few hundred milliseconds apart share it. Cleared when it
    // settles; see `refreshStatus`.
    let readinessPending = null;

    // What the last status answer said, if it is recent enough to act on.
    // Separate from `refreshStatus` because the gesture needs an answer it can
    // use *now*: the slide starts the check and the engagement at the end of
    // the slide reads it, which is a few hundred milliseconds later and costs
    // no round trip at all.
    function knownReadiness() {
        if (stale) return {ok: false, error: RELOAD};
        if (readiness && Date.now() - readinessAt < 5000) return readiness;
        return null;
    }

    // Always resolves, and always with something that has an `ok` and, when it
    // is false, an `error` somebody can read. The first version resolved with
    // null on every failure, which meant each caller quietly did nothing --
    // and the visible result of that was a Settings row frozen on "Starting…"
    // with nothing anywhere to say why. A status check that cannot report its
    // own failure is worse than no status check.
    function refreshStatus(force, scope) {
        const now = Date.now();
        if (stale) return Promise.resolve({ok: false, error: RELOAD});
        if (!force && readiness && now - readinessAt < 5000) {
            return Promise.resolve(readiness);
        }
        // One request, however many callers. The slide starts a check and the
        // engagement at the end of it asks again a few hundred milliseconds
        // later; without this the second call opens a second connection to ask
        // the same question, and on a phone over a mesh VPN that is a round
        // trip standing between a gesture and a permission prompt. A forced
        // poll keeps its own semantics -- it is asked precisely because the
        // caller wants a *new* answer.
        if (!force && readinessPending) return readinessPending;
        const request = statusRequest(scope);
        if (force) return request;
        // `statusRequest` always resolves -- with an `ok: false` and a sentence
        // where anything went wrong -- so there is no rejection path to clear.
        const wrapped = request.then(function (payload) {
            if (readinessPending === wrapped) readinessPending = null;
            return payload;
        });
        readinessPending = wrapped;
        return wrapped;
    }

    function statusRequest(scope) {
        return fetch(url(ROUTES.status), {
            method: "POST", credentials: "same-origin", headers: headers(null, scope),
        }).then(refused).then(function (response) {
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
    // The numbers moved down once the producer stopped running dry. They were
    // chosen for a pipeline where the second segment could be several seconds
    // behind the first, and 1.6 seconds of prebuffer is 1.6 seconds of latency
    // paid on every single reply to hide a gap that now mostly is not there.
    // What is deliberately *not* done is dropping to zero: the envelope still
    // moves, an underrun still raises it, and a slow machine still recovers to
    // something that plays without stuttering. Earlier continuous audio, not
    // merely earlier audio.
    const BUFFER = {
        start: 0.7,
        min: 0.4,
        max: 2.0,
        high: 12.0,
        clean: 0,
    };

    // Section 22. What a head start cannot buy.
    //
    // START is a fixed lead, and a fixed lead only ever hides a *bounded*
    // shortfall. When the producer is slower than real time the shortfall is
    // not bounded: it grows for as long as the reply lasts, so a buffer deep
    // enough for a ten-second reply still runs dry partway through a
    // forty-second one. Sizing START for the longest reply would mean paying
    // that latency on every short one, which is the wrong trade in both
    // directions.
    //
    // What *is* fixable is the shape of the failure. Resuming on the first
    // block to arrive after the queue has emptied guarantees the block after
    // it is late as well, so one shortfall becomes a rattle that lasts the
    // rest of the turn -- fifty-odd gaps of a twentieth of a second each,
    // which is heard as a broken speaker rather than as a slow one. So a dry
    // queue is treated as a rebuffer: blocks are held until there are REBUFFER
    // seconds of them, and the hold doubles every time it turns out to have
    // been too short. The listener gets a few pauses between sentences instead,
    // and on a machine only slightly behind, the first pause is the only one.
    //
    // This is the one adjustment that is deliberately *not* deferred to the
    // next turn. START is, because a head start that grows mid-sentence is a
    // gap the listener did not need to hear. A rebuffer is the opposite: the
    // speaker has already fallen silent, and the only question left is whether
    // it resumes into another gap.
    const REBUFFER = {
        first: 1.0,
        max: 6.0,
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
            // The mid-turn rebuffer. `starving` means the queue reached zero
            // while the stream was still producing, so blocks are being held
            // rather than played; `rebufferTarget` is how many seconds of them
            // it takes to resume, and it doubles each time the hold proves too
            // short. Both are per turn: a machine that fell behind on one
            // reply is not condemned to a deep hold on the next one.
            starving: false,
            rebufferTarget: 0,
            rebuffers: 0,
            // Marks, all on this browser's own monotonic clock and all
            // durations from `seenAt`. They are what tells "the server was slow"
            // apart from "this page took a third of a second to notice", which
            // a single end-to-end number cannot.
            seenAt: nowMs(),
            headersAt: 0,
            firstPcmAt: 0,
            releasedAt: 0,
            startTarget: BUFFER.start,
            underrunTotal: 0,
            underrunMax: 0,
            firstUnderrunAt: 0,
            endReason: "",
            reported: false,
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
        // Before the state is torn down, because everything the report is made
        // of is about to be reset. "Cancelled" unless a caller has already said
        // what happened -- a turn replaced by the next reply and a stream that
        // could not be played are different endings and should read that way.
        reportPlayback(speech, speech.endReason || "cancelled");
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
        // The stream is over. Whatever is held goes now, however little it is
        // and whatever the hold was waiting for -- there is nothing left to
        // wait for, and a rebuffer that swallowed the last sentence would be
        // a worse bug than the stutter it exists to prevent.
        if (state.draining) {
            release(state);
            return;
        }
        // The turn's own target, captured when it started. Reading BUFFER.start
        // here instead would let an underrun raise the target of the very turn
        // that is already playing, which is a buffer that grows in the middle
        // of a sentence rather than before the next one.
        if (!state.started) {
            if (state.pendingSeconds < state.startTarget) return;
            release(state);
            return;
        }
        // Playing, and this block arrived to find nothing scheduled ahead of
        // it: the speaker is already silent. Releasing now would put one block
        // of speech in front of the next gap, so instead the hold starts here
        // and this block waits with the ones behind it.
        if (!state.starving && queuedSeconds(state) <= 0) {
            state.starving = true;
            state.rebuffers += 1;
            state.rebufferTarget = state.rebufferTarget
                ? Math.min(REBUFFER.max, state.rebufferTarget * 2)
                : REBUFFER.first;
        }
        if (state.starving && state.pendingSeconds < state.rebufferTarget) return;
        state.starving = false;
        release(state);
    }

    function release(state) {
        const waiting = state.pending;
        state.pending = [];
        state.pendingSeconds = 0;
        if (!state.releasedAt && waiting.length) state.releasedAt = nowMs();
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
                //
                // How *long* it was silent is the number that was missing, and
                // it has to be taken here, before `nextStart` is reset: the
                // distance the audio clock has run past the end of what was
                // scheduled is exactly the gap the listener heard. A count
                // alone cannot tell four imperceptible hiccups from one
                // four-second hole, and those are different bugs.
                const gap = Math.max(0, now - state.nextStart);
                state.underruns += 1;
                state.underrunTotal += gap;
                if (gap > state.underrunMax) state.underrunMax = gap;
                if (!state.firstUnderrunAt) state.firstUnderrunAt = nowMs();
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
        if (speech) {
            speech.endReason = "replaced";
            stopSpeaking(true);
        }
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
                state.headersAt = nowMs();
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
            state.endReason = "error";
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
                if (bytes.length) {
                    if (!state.firstPcmAt) state.firstPcmAt = nowMs();
                    schedule(state, toFloat32(bytes), rate);
                }
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

    // What this page knows and the WebUI cannot: whether the speaker actually
    // ran dry, and for how long.
    //
    // Every number is a difference between two `performance.now()` readings
    // taken here, so they are all in one clock domain and none of them is
    // comparable with a timestamp from the server. Between them they name the
    // stage that was slow: noticing the turn, opening the stream, the first
    // sample arriving, this page holding audio back in its own prebuffer, or
    // the queue emptying underneath the speaker.
    //
    // Never the audio and never the text. The turn id goes with it because two
    // records of one response have to be recognisable as one response; it is
    // opaque, it belongs to a single reply, and it does not outlive the run.
    function playbackReport(state, reason) {
        const gap = function (mark) {
            return mark ? Math.round(mark - state.seenAt) : null;
        };
        return {
            kind: "playback",
            turn: state.id,
            turn_seen_to_headers_ms: gap(state.headersAt),
            headers_to_first_pcm_ms: (state.headersAt && state.firstPcmAt)
                ? Math.round(state.firstPcmAt - state.headersAt) : null,
            first_pcm_to_playback_ms: (state.firstPcmAt && state.releasedAt)
                ? Math.round(state.releasedAt - state.firstPcmAt) : null,
            startup_buffer_ms: Math.round(state.startTarget * 1000),
            underrun_count: state.underruns,
            first_underrun_after_play_ms: (state.firstUnderrunAt && state.releasedAt)
                ? Math.round(state.firstUnderrunAt - state.releasedAt) : null,
            max_underrun_gap_ms: Math.round(state.underrunMax * 1000),
            total_underrun_gap_ms: Math.round(state.underrunTotal * 1000),
            // Two counts rather than one, because they answer different
            // questions. `underrun_count` is how often the speaker ran dry;
            // `rebuffer_count` is how often this page decided to stay quiet
            // and refill rather than resume into the next gap. A turn with
            // many underruns and few rebuffers is a producer that is a little
            // behind; one where the two track each other is a producer that
            // cannot keep up at all, and no amount of buffering will fix it.
            rebuffer_count: state.rebuffers,
            rebuffer_target_ms: Math.round(state.rebufferTarget * 1000),
            playback_end_reason: reason || state.endReason || "finished",
        };
    }

    // Best effort in the strongest sense: nothing waits for it, nothing reads
    // its answer, and a failure is swallowed. A page that cannot report its own
    // timings has nothing wrong with its audio, and this must never be a reason
    // anything else stops.
    function sendReport(payload) {
        try {
            console.info("Model Chain: Voice " + payload.kind + " timing — "
                         + Object.keys(payload).filter(function (name) {
                             return name !== "kind" && name !== "turn"
                                 && payload[name] !== null;
                         }).map(function (name) {
                             return name + "=" + payload[name];
                         }).join(", "));
        } catch (error) { /* a console that will not take it is not a failure */ }
        try {
            fetch(url(ROUTES.telemetry), {
                method: "POST",
                credentials: "same-origin",
                headers: headers({"Content-Type": "application/json"}),
                body: JSON.stringify(payload),
                keepalive: true,
            }).catch(function () { /* a timing nobody recorded is not a failure */ });
        } catch (error) { /* nor is a fetch this browser would not make */ }
    }

    // Every ending, not only the tidy one. The envelope used to be adjusted in
    // `finishSpeech`, which is reached only when the stream runs to its end --
    // so a turn the listener stopped taught it nothing. That is exactly
    // backwards: the reply somebody gives up on is the reply that stuttered,
    // and the log from a machine that could not keep up showed a 700 ms head
    // start on turn after turn because every one of those turns was cancelled
    // before it could report.
    //
    // Raising and relaxing are not symmetrical, though. Underruns are evidence
    // however the turn ended -- the speaker did fall silent, and the listener
    // did hear it. A clean run is only evidence if it was allowed to finish: a
    // turn stopped two seconds in has not shown that anything works, so it may
    // raise the target but never lower it.
    function learnFrom(state, finished) {
        if (state.underruns) raiseStartBuffer();
        else if (finished) relaxStartBuffer();
    }

    function reportPlayback(state, reason) {
        if (!state || state.reported) return;
        state.reported = true;
        learnFrom(state, reason === "finished");
        sendReport(playbackReport(state, reason));
    }

    function finishSpeech(state) {
        if (speech !== state) return;
        reportPlayback(state, "finished");
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
    // Section 39, rewritten after a phone.
    //
    // This used to be a capability check that refused before anything was
    // asked: no `navigator.mediaDevices.getUserMedia`, no recording, and the
    // browser was never given the chance to raise its own permission prompt.
    // On Android that check is the whole bug -- the extension answered a
    // question that was the browser's to answer, and the user got a sentence
    // about capture not being available instead of a permission dialog.
    //
    // Nothing decides for the browser here any more. Every entry point a
    // browser has ever exposed for this is tried, in order, and the request is
    // actually made. Only when there is genuinely nothing to call does this
    // fail -- and it fails with a name of its own so the error map can say why
    // rather than guessing.
    function getUserMedia(constraints) {
        const media = navigator && navigator.mediaDevices;
        if (media && typeof media.getUserMedia === "function") {
            try {
                return media.getUserMedia(constraints);
            } catch (error) {
                return Promise.reject(error);
            }
        }
        // The prefixed callback form. It predates the promise API, and it is
        // still the only one present in some Android WebViews and older
        // in-app browsers -- exactly the places `mediaDevices` is missing.
        const legacy = navigator && (navigator.getUserMedia
                                     || navigator.webkitGetUserMedia
                                     || navigator.mozGetUserMedia
                                     || navigator.msGetUserMedia);
        if (typeof legacy === "function") {
            return new Promise(function (resolve, reject) {
                try {
                    legacy.call(navigator, constraints, resolve, reject);
                } catch (error) {
                    reject(error);
                }
            });
        }
        const missing = new Error("this browser exposes no way to open a microphone");
        missing.name = "McVoiceNoCapture";
        return Promise.reject(missing);
    }

    // A browser withholds `getUserMedia` entirely for one common reason, and
    // it is one somebody can do something about. Asked only once there is
    // nothing left to call, so a browser that simply refused permission is
    // never told it has an HTTPS problem it does not have.
    function noCaptureReason() {
        let secure = true;
        try {
            if (typeof window.isSecureContext === "boolean") secure = window.isSecureContext;
        } catch (error) { /* a browser that will not answer is not a verdict */ }
        return secure ? MESSAGES.unsupported : MESSAGES.insecure;
    }

    // -- what a Bluetooth microphone actually is ---------------------------- //
    //
    // The report this exists for: dictation from an Android phone was good on
    // the handset's own microphone and produced "(music)" and "(static)" the
    // moment a Bluetooth headset was connected.
    //
    // Nothing here can fix that, and everything here is about surviving it.
    // A headset carries audio *out* over A2DP, which has no microphone in it at
    // all, so capturing from one needs the hands-free profile and an SCO link —
    // the link a phone call uses. Outside a call Android decides whether to open
    // it, and what it opens is narrowband: 8 kHz for plain HFP, 16 kHz with
    // mSBC. What reaches this page is band-limited to about 4 kHz, already
    // through a codec built for telephony, and usually far quieter than the
    // phone's own microphone, which the platform gain-stages and beam-forms.
    // `mc_voice_hearing.py` says the same thing from Whisper's side.
    //
    // Three things follow, and they are the three changes in this block.
    //
    // The constraints ask for what actually helps and stop asking for what
    // does not.
    //
    // `sampleRate` used to be here, asking for the 16 kHz Whisper wants rather
    // than the 48 kHz that gets thrown away. It never did anything: the samples
    // this file receives come out of the AudioContext, not off the track, and
    // `resample` converts from `ctx.sampleRate` to 16 kHz whatever the device
    // was persuaded to do. What it could do is give the browser one more thing
    // to negotiate while it opens the device -- and opening the device is the
    // largest remaining cost in the capture path, 210 to 386 ms in the measured
    // session. So it is gone, and `stream_ready_ms` will say whether that
    // mattered. The track's real rate is still reported by `describeTrack`,
    // which is the piece of evidence a user has that their microphone changed
    // under them.
    //
    // The three processors are *advisory* rather than required, because they
    // are tuned for a wideband microphone and a browser that cannot apply them
    // to an HFP stream should give us the stream rather than fail.
    //
    // The capture path is reported, not guessed. `track.getSettings()` and the
    // track's own label say which device this is and at what rate, which is the
    // one piece of evidence a user has that their microphone changed under them.
    //
    // The level is measured and normalised. A quiet capture is made louder by a
    // bounded amount before it is encoded, and one that is under the floor even
    // after that is not sent at all — a round trip and a large model are a slow
    // way to be told a recording was silent.

    const CAPTURE_CONSTRAINTS = {
        channelCount: 1,
        echoCancellation: {ideal: true},
        noiseSuppression: {ideal: true},
        autoGainControl: {ideal: true},
    };

    // Under this peak there was nothing to transcribe. The same floor
    // `mc_voice_hearing.SILENCE_PEAK` uses, deliberately: refusing here saves
    // the round trip and refusing there covers a page that predates this.
    const QUIET_PEAK = 0.012;
    // How much a quiet capture may be lifted. Twelve times is about 21 dB,
    // which covers the gap between a Bluetooth headset and a handset
    // microphone; past that the noise comes up with the speech and nothing is
    // gained.
    const MAX_MAKEUP = 12.0;
    // What a normalised recording should peak at. Not 1.0: Whisper is not
    // helped by a signal against the ceiling, and leaving headroom means the
    // limiter below never has anything to do on ordinary speech.
    const TARGET_PEAK = 0.6;

    // What the microphone turned out to be, for the status line and the console.
    // Read once per capture, after the track exists — before that it is a guess.
    function describeTrack(stream) {
        let track = null;
        try {
            // `getAudioTracks` where the browser has it, and `getTracks` where
            // it does not -- the older of the two is still the only one present
            // in some Android WebViews, which are exactly the browsers this
            // whole path exists for.
            const found = typeof stream.getAudioTracks === "function"
                ? stream.getAudioTracks() : stream.getTracks();
            track = (found || [])[0] || null;
        } catch (error) { /* a stream that will not answer is not a verdict */ }
        if (!track) return {};
        let settings = {};
        try {
            settings = (typeof track.getSettings === "function" ? track.getSettings() : {})
                || {};
        } catch (error) { /* older browsers */ }
        return {
            label: track.label || "",
            rate: settings.sampleRate || 0,
            channels: settings.channelCount || 0,
            device: settings.deviceId || "",
        };
    }

    // A device whose name or capture rate says "telephony". Only ever used to
    // choose which sentence to show — never to refuse a capture, and never to
    // change what is recorded.
    function looksNarrowband(found) {
        if (!found) return false;
        if (found.rate && found.rate <= 16000) return true;
        const label = (found.label || "").toLowerCase();
        return label.indexOf("bluetooth") >= 0 || label.indexOf("headset") >= 0
            || label.indexOf("hands-free") >= 0 || label.indexOf("hfp") >= 0
            || label.indexOf("sco") >= 0;
    }

    // Peak and RMS of what was captured, in one pass over the samples we are
    // about to encode anyway.
    function levelOf(samples) {
        let peak = 0;
        let total = 0;
        for (let i = 0; i < samples.length; i += 1) {
            const value = samples[i];
            const magnitude = value < 0 ? -value : value;
            if (magnitude > peak) peak = magnitude;
            total += value * value;
        }
        return {
            peak: peak,
            rms: samples.length ? Math.sqrt(total / samples.length) : 0,
        };
    }

    // Lift a quiet recording towards TARGET_PEAK, by no more than MAX_MAKEUP,
    // and never at all when it is already loud enough. In place, because the
    // array is this function's caller's and is about to be encoded and dropped.
    //
    // No compression and no limiter: the gain is chosen so the loudest sample
    // lands on TARGET_PEAK, which cannot clip by construction. A recording that
    // is quiet *because it is mostly silence with one loud word in it* is
    // therefore left alone, which is correct — that word was already audible.
    function normalise(samples, level) {
        if (!samples.length || level.peak <= 0) return 1;
        if (level.peak >= TARGET_PEAK) return 1;
        const wanted = Math.min(TARGET_PEAK / level.peak, MAX_MAKEUP);
        if (wanted <= 1.01) return 1;
        for (let i = 0; i < samples.length; i += 1) samples[i] *= wanted;
        return wanted;
    }

    // One capture at a time, and every resource it holds named here so that
    // stopping it can never leave a track open. An open microphone the user did
    // not ask for is the failure this feature must not have.
    let capture = null;

    // The constrained request first, then a bare one. Every constraint above is
    // advisory, but a browser is entitled to reject a constraint *set* it does
    // not understand, and the devices most likely to do that are the Android
    // WebViews this feature already bends over backwards for. A second attempt
    // asking for nothing but audio is the difference between a worse recording
    // and no recording.
    function openMicrophone() {
        return getUserMedia({audio: CAPTURE_CONSTRAINTS}).catch(function (error) {
            const name = (error && error.name) || "";
            if (name !== "OverconstrainedError" && name !== "ConstraintNotSatisfiedError"
                && name !== "TypeError" && name !== "NotSupportedError") {
                return Promise.reject(error);
            }
            console.warn("Model Chain: Voice Chat's microphone constraints were refused ("
                         + name + "); asking for any microphone instead.");
            return getUserMedia({audio: true});
        });
    }

    // -- the capture processor, registered once per AudioContext -------------- //

    // `registerProcessor` puts a name in the AudioWorkletGlobalScope, and that
    // scope belongs to the AudioContext rather than to one recording. So
    // `addModule` was being called again for every utterance to register a name
    // that was already there -- which is a duplicate registration the
    // specification allows a browser to refuse with NotSupportedError, and
    // which even where it is tolerated is a Blob, an object URL and a module
    // load per press of the microphone.
    //
    // One preparation per context, remembered, and a fresh AudioWorkletNode per
    // utterance -- which is what a node is for. If the context is ever replaced,
    // the record is replaced with it: a module registered in a scope that no
    // longer exists is not a module this page has.
    let captureWorklet = {context: null, promise: null, ready: false};

    function ensureCaptureWorklet(ctx) {
        if (!ctx) return Promise.resolve(false);
        if (captureWorklet.context === ctx && captureWorklet.promise) {
            return captureWorklet.promise;
        }
        const settle = function (promise) {
            captureWorklet = {context: ctx, promise: promise, ready: false};
            promise.then(function (ok) {
                if (captureWorklet.promise === promise) captureWorklet.ready = !!ok;
            });
            return promise;
        };
        if (!ctx.audioWorklet || typeof ctx.audioWorklet.addModule !== "function"
            || typeof Blob === "undefined" || !window.URL || !window.URL.createObjectURL) {
            return settle(Promise.resolve(false));
        }
        let address = "";
        try {
            address = window.URL.createObjectURL(new Blob([WORKLET],
                                                         {type: "text/javascript"}));
        } catch (error) {
            return settle(Promise.resolve(false));
        }
        const done = function (ok) {
            try { window.URL.revokeObjectURL(address); } catch (error) { /* ignore */ }
            return ok;
        };
        let loading;
        try {
            loading = Promise.resolve(ctx.audioWorklet.addModule(address));
        } catch (error) {
            return settle(Promise.resolve(done(false)));
        }
        return settle(loading.then(function () { return done(true); },
                                   function () { return done(false); }));
    }

    // Every sample that is kept passes through here, from either capture path,
    // so that "recording" has exactly one definition: the first PCM frame this
    // page accepted for the session the user is holding. Not permission
    // granted, not a node connected -- audio arriving.
    function acceptChunk(state, samples) {
        // The dictation path keeps exactly one live capture in `capture`, and
        // this guard is what stops a stream that has been abandoned from still
        // filling memory. A standalone capture -- the Sopro clone recorder --
        // is not that stream and is not in that variable: it owns itself, it is
        // released by the thing that started it, and it is deliberately allowed
        // to run without becoming the dictation capture.
        if (!state.standalone && (!capture || capture !== state)) return;
        if (!samples || !samples.length) return;
        if (!state.firstPcmAt) {
            state.firstPcmAt = nowMs();
            const session = sliding;
            if (session && session.capture === state && !session.cancelled) {
                armCaptureLimits(session);
                if (session.engaged) {
                    // The gesture finished before the device did. These samples
                    // are the beginning of the utterance and the control says so
                    // now.
                    session.recordingAt = state.firstPcmAt;
                    beginRecording(session);
                } else {
                    // Pre-roll. Audio is arriving and it belongs to nobody yet:
                    // it is kept in this page's memory and it becomes an
                    // utterance only if the slide completes. The control says
                    // the microphone is open, which is true, and does not say
                    // it is recording, which is not.
                    markMic("buffering");
                }
            }
        }
        state.chunks.push(samples);
    }

    function startCapture(session, standalone) {
        const ctx = audioContext();
        if (!ctx) return Promise.reject(new Error("no audio"));
        // Started here and deliberately not awaited here. A browser raises a
        // permission prompt only while it still considers a user gesture in
        // progress, and a promise chain that has already awaited a module load
        // is past that on a phone -- so the microphone is asked for in this
        // same task and the worklet is picked up below if it is ready by then.
        const worklet = ensureCaptureWorklet(ctx);
        if (session) session.workletAt = 0;
        worklet.then(function () {
            if (session && !session.workletAt) session.workletAt = nowMs();
        });
        return openMicrophone().then(function (stream) {
            if (session) session.mediaAt = nowMs();
            const state = {stream: stream, chunks: [], rate: ctx.sampleRate, nodes: [],
                           track: describeTrack(stream), firstPcmAt: 0, graph: "none",
                           standalone: !!standalone};
            const source = ctx.createMediaStreamSource(stream);
            state.nodes.push(source);

            const useProcessor = function () {
                state.graph = "script";
                const processor = ctx.createScriptProcessor(4096, 1, 1);
                processor.onaudioprocess = function (event) {
                    acceptChunk(state, new Float32Array(event.inputBuffer.getChannelData(0)));
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

            return worklet.then(function (ready) {
                if (!ready) return useProcessor();
                try {
                    const node = new window.AudioWorkletNode(ctx, "mc-voice-tap");
                    state.graph = "worklet";
                    node.port.onmessage = function (event) {
                        acceptChunk(state, event.data);
                    };
                    source.connect(node);
                    state.nodes.push(node);
                    return state;
                } catch (error) {
                    // The module is registered and this context still could not
                    // make a node of it. One utterance on the fallback is better
                    // than no utterance at all.
                    return useProcessor();
                }
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
        // Revealing a disabled button is worse than leaving it hidden: it is a
        // Stop on screen during exactly the phase it cannot stop. Python builds
        // Stop interactive now, but a Gradio re-render can reassert the
        // server-side attributes of a component at any point, so this is
        // asserted here as well -- conditionally, like every other write in
        // this file, because rewriting an attribute that is already right wakes
        // every observer on the page.
        if (working) enable(stop);
        const sendHolder = holderOf(send);
        const stopHolder = holderOf(stop);
        // On a build where Send and Stop share a wrapper, `holderOf` answers
        // with that wrapper for both -- and hiding one would hide the other,
        // leaving a composer with no button in it at all. The buttons
        // themselves are always distinct, so that is what is used instead.
        if (sendHolder === stopHolder) {
            show(send, !working);
            show(stop, working);
            return;
        }
        show(sendHolder, !working);
        show(stopHolder, working);
    }

    // Hiding is ours; *revealing* has to undo Gradio's, and that is the half
    // that is easy to miss. When the model finishes, Python's own IDLE update
    // hides Stop -- correctly, as far as Python knows -- and if Voice is still
    // speaking, adding a class of our own to Send would leave a composer with
    // neither button in it. So this takes Gradio's marker off the control that
    // must be visible, and puts ours on the one that must not be. Reasserted on
    // every tick and on every run-state write, which is what makes it settle
    // rather than flicker.
    //
    // Every write here is conditional, and that is not a micro-optimisation.
    // `classList.remove` of a class that is not there still rewrites the
    // element's `class` attribute, and rewriting an attribute wakes every
    // MutationObserver watching that subtree. This function runs on a 400 ms
    // tick and again after every Gradio update; a theme that observes
    // attributes -- LobeTheme is one -- then has work to do twice a second
    // forever, and a page whose observers never settle is a page whose first
    // paint never finishes. Asserting the state is the requirement. Rewriting
    // it when it is already right is what broke somebody's txt2img tab.
    // Whatever "this control is not available" is currently spelled as. All
    // three are removed together because a browser, a theme and Gradio each use
    // a different one, and a control that answers a pointer but reads as
    // disabled to a screen reader is only half a control.
    function enable(node) {
        if (!node) return;
        if (node.disabled) node.disabled = false;
        try {
            if (node.getAttribute && node.getAttribute("disabled") !== null) {
                node.removeAttribute("disabled");
            }
            if (node.getAttribute && node.getAttribute("aria-disabled") === "true") {
                node.setAttribute("aria-disabled", "false");
            }
        } catch (error) { /* a node that will not take it is not a failure */ }
        if (node.classList && node.classList.contains("disabled")) {
            node.classList.remove("disabled");
        }
    }

    function show(node, wanted) {
        if (!node || !node.classList) return;
        if (wanted) {
            if (node.classList.contains("mc-llm-voice-hidden")) {
                node.classList.remove("mc-llm-voice-hidden");
            }
            if (node.classList.contains("hidden")) node.classList.remove("hidden");
            if (node.style && node.style.display === "none") node.style.display = "";
        } else if (!node.classList.contains("mc-llm-voice-hidden")) {
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

    // Six states, and the three before red are the point of this function.
    //
    //   opening     the microphone has been asked for and nothing has come back
    //   buffering   samples are arriving and belong to nobody yet
    //   recording   the slide reached the end, so those samples are an utterance
    //
    // Those are three different facts about the world and they can be a second
    // apart on a phone. Red is the claim that the user is being recorded, and
    // it is earned by the gesture completing over audio that exists -- not by
    // permission, not by a connected node, and not by a finger arriving.
    //
    // The text is authoritative, not the colour: a screen reader is told
    // "Microphone open — slide to record" and then "Recording", which is the
    // same distinction in the channel that has no colour to read.
    function markMic(state) {
        const button = clickable(IDS.mic);
        if (!button) return;
        button.classList.remove("mc-llm-voice-opening", "mc-llm-voice-buffering",
                                "mc-llm-voice-recording", "mc-llm-voice-working",
                                "mc-llm-voice-error");
        if (state) button.classList.add("mc-llm-voice-" + state);
        const labels = {
            opening: "Opening microphone",
            buffering: "Microphone open — slide to record",
            recording: "Recording — release to transcribe",
            working: "Transcribing",
            error: "Voice Chat is not set up",
        };
        button.setAttribute("aria-label", labels[state]
            || "Dictate — slide right and hold, or hold Space");
    }

    // The transition every part of the interface waits for. Called once per
    // session, from `acceptChunk`, at the moment the first sample is kept.
    // Armed at the first sample, whether or not the gesture has finished.
    //
    // Two bounds swap over here, and the swap is the point. Up to this moment
    // the risk is a microphone that never opens, and the opening timeout covers
    // it; from this moment the risk is a microphone that stays open, and the
    // recording cap covers that. A device that took a second to wake has spent
    // its own second and not the user's -- section 7.7 -- and a gesture still
    // buffering after a minute is a microphone held open for a minute, which is
    // the same thing the cap exists to stop.
    function armCaptureLimits(session) {
        if (session.openTimer) {
            window.clearTimeout(session.openTimer);
            session.openTimer = 0;
        }
        if (session.timer) return;
        session.timer = window.setTimeout(function () {
            if (sliding !== session && holding !== session) return;
            // The slider goes back with it: a control still sitting at the
            // recording end of its track after the recording has stopped is a
            // control that is lying about what it is doing.
            resetSlider();
            if (session.engaged) {
                endHold(false);
                return;
            }
            // A minute of pre-roll nobody committed. It is not an utterance and
            // it never was, so it goes the way every abandoned gesture goes.
            abandon(session, "discarded");
        }, MAX_HOLD_MS);
    }

    function beginRecording(session) {
        armCaptureLimits(session);
        markMic("recording");
        armTrack(clickable(IDS.mic), true);
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
        if (name === "McVoiceNoCapture") return noCaptureReason();
        if (name === "NotAllowedError" || name === "SecurityError") return MESSAGES.denied;
        if (name === "NotFoundError" || name === "OverconstrainedError") return MESSAGES.notFound;
        if (name === "NotReadableError" || name === "AbortError") return MESSAGES.unreadable;
        return MESSAGES.unreadable;
    }

    // Whether this gesture may open a microphone at all, decided synchronously
    // from what is already known.
    //
    // Synchronously is the whole point. The press is the moment the microphone
    // has to be asked for, and a check that waits for a round trip to the WebUI
    // is a check that has spent the pre-roll it was meant to protect. So the
    // two cheap answers are taken here -- the model is generating, or the last
    // status answer already said no -- and the expensive one runs alongside the
    // capture and is reconciled when it arrives.
    //
    // A refusal is remembered rather than announced. A press that never becomes
    // a slide is not a request for anything, and a control that complains about
    // being touched is a control people stop touching.
    function eligible(session) {
        if (llmBusy()) {
            session.refused = MESSAGES.busy;
            return false;
        }
        // A Sopro clone recording is open on the Settings page. Two microphone
        // streams at once is two indicators and two devices, and dictating into
        // somebody's voice reference is not a thing either of them asked for.
        if (soproRecorder) {
            session.refused = "A voice recording is in progress in Settings.";
            return false;
        }
        const known = knownReadiness();
        if (known && !known.ok) {
            session.refused = known.error || MESSAGES.failed;
            return false;
        }
        if (known && !known.ready) {
            session.refused = known.not_ready_message || MESSAGES.notReady;
            return false;
        }
        return true;
    }

    // The microphone, asked for at the beginning of the gesture rather than at
    // the end of it.
    //
    // This is the change revision 4 is mostly about. Permission, device open,
    // stream and graph assembly all used to sit between the slide finishing and
    // the first sample being kept, so a user who began talking as they slid
    // lost the first word or two of it. Now the slide *is* the window those
    // things happen in, and what they produce is kept locally until the gesture
    // says whether it is an utterance.
    //
    // Nothing is uploaded and nothing is claimed. The microphone is open, which
    // is a state the control shows and a state the browser's own indicator
    // shows; whether any of it is a recording is decided at the far end of the
    // track.
    function openForGesture(session) {
        if (!eligible(session)) return;
        // Section 14, and now one gesture earlier: the assistant's own speaker
        // output is the thing most likely to end up in a microphone opened
        // beside it, so it is silenced before any pre-roll is accepted rather
        // than after.
        stopSpeaking(true);
        markMic("opening");

        const opening = startCapture(session);
        session.opening = opening;
        // Handled here as well so that a rejection which the reconciliation
        // below decides not to look at is not an unhandled one.
        opening.catch(function () { /* reported by the chain below */ });

        // A microphone that never opens must not leave the control waiting for
        // ever. Separate from the recording cap, because they bound entirely
        // different things -- see section 7.7.
        session.openTimer = window.setTimeout(function () {
            // Only while nothing has been captured at all. Once samples are
            // arriving the recording cap owns the gesture -- see
            // `armCaptureLimits`, which cancels this on its way past.
            if (sliding !== session) return;
            session.refused = MESSAGES.unreadable;
            abandon(session, "error");
            if (session.engaged) {
                holding = null;
                resetSlider();
                refuse(MESSAGES.unreadable);
            }
        }, OPEN_TIMEOUT_MS);

        opening.then(function (state) {
            if (sliding !== session || session.cancelled) {
                // The gesture is over and this stream arrived anyway. Section
                // 5.5: stop every track, attach nothing, accept nothing.
                releaseCapture(state);
                return;
            }
            capture = state;
            session.capture = state;
            session.graphAt = nowMs();
            if (!session.engaged) markMic("buffering");
        }).catch(function (error) {
            if (sliding !== session) return;
            session.refused = captureFailure(error);
            abandon(session, "error");
            if (session.engaged) {
                holding = null;
                resetSlider();
                // Section 39's map, in one place. The browser's own error name
                // is the only thing that knows which of these happened, and
                // each of them is a different thing for the user to do next.
                refuse(session.refused);
            }
        });

        // The expensive answer, running alongside rather than in front. What it
        // still decides is whether the recording is *kept*.
        refreshStatus(false).then(function (found) {
            if (sliding !== session || session.cancelled) return;
            session.readinessAt = nowMs();
            if (found && found.ok && found.ready) return;
            const why = (!found || !found.ok)
                ? ((found && found.error) || MESSAGES.failed)
                : (found.not_ready_message || MESSAGES.notReady);
            session.refused = why;
            abandon(session, "error");
            holding = null;
            resetSlider();
            refuse(why);
        });
    }

    // Everything a gesture has to give back, on every path that ends one
    // without sending: release, cancel, timeout, readiness failure, capture
    // failure. Section 5.7, in one place, because five callers each doing four
    // of the five things is how a microphone gets left open.
    function abandon(session, result) {
        if (session.done) return;
        session.done = true;
        session.cancelled = true;
        if (session.openTimer) window.clearTimeout(session.openTimer);
        if (session.timer) window.clearTimeout(session.timer);
        if (!session.releasedAt) session.releasedAt = nowMs();
        const state = session.capture;
        if (capture === state) capture = null;
        releaseCapture(state);
        if (state) state.chunks = [];
        if (session.opening) {
            // A stream still on its way. It is stopped the moment it arrives --
            // the guard above already refuses to attach it, and this makes sure
            // its tracks do not stay live because nobody was left to look.
            session.opening.then(releaseCapture, function () { /* nothing opened */ });
        }
        markMic("");
        reportCaptureTiming(session, state, result || "discarded");
    }

    // Called the instant the slide reaches the end of its track, and from the
    // keyboard equivalent. What it does now is *commit*: the microphone is
    // already open, samples may already be arriving, and this is the moment
    // they stop being the page's and become the user's utterance.
    function engage() {
        const session = sliding;
        if (!session || session.engaged) return;
        session.engaged = true;
        session.engagedAt = nowMs();

        // Whatever this gesture was refused for at the press, said now -- at
        // the point the user actually asked for something.
        if (session.refused) {
            // The interrupt gesture still applies. Sliding the microphone while
            // a reply is being read aloud means the reply is over, whether or
            // not this page is willing to record.
            stopSpeaking(true);
            abandon(session, "error");
            resetSlider();
            refuse(session.refused);
            return;
        }

        holding = session;
        if (session.capture && session.capture.firstPcmAt) {
            // The device was ready before the finger arrived. Section 5.6: the
            // committed utterance begins at the first retained sample, because
            // that pre-roll is deliberately part of what was said.
            session.recordingAt = session.capture.firstPcmAt;
            beginRecording(session);
        } else {
            markMic("opening");
        }
    }

    function endHold(cancelled) {
        const session = holding;
        holding = null;
        if (!session || session.done) return;
        session.done = true;
        if (session.timer) window.clearTimeout(session.timer);
        if (session.openTimer) window.clearTimeout(session.openTimer);
        session.cancelled = true;
        session.releasedAt = nowMs();

        const state = capture || session.capture;
        capture = null;
        // From the first sample, not from the gesture. A phone that takes a
        // second to open a Bluetooth microphone was charging that second to the
        // user: hold for four hundred milliseconds of speech and the recording
        // was "long enough" because the wait counted, or -- with the clock the
        // other way round -- three hundred milliseconds of real speech was
        // refused as a tap. `performance.now()` rather than `Date.now()`,
        // because a wall clock that syncs mid-utterance can make a held button
        // look like it was held for a negative length of time.
        const held = session.recordingAt ? nowMs() - session.recordingAt : 0;
        releaseCapture(state);
        markMic("");
        // Reported once, at the end, because the interesting field is what
        // *happened* to the recording -- and none of the exits below knows that
        // until it gets there. Every one of them goes through this.
        const done = function (result) {
            reportCaptureTiming(session, state, result);
        };

        if (cancelled || !state) {
            done("discarded");
            return;
        }
        if (!session.recordingAt) {
            // Released while the microphone was still opening. Nothing was
            // captured, so there is nothing to transcribe and nothing to refuse
            // -- section 7.5. Saying "that was too short" here would be a
            // sentence about the user when the wait was the device's.
            done("discarded");
            return;
        }
        if (held < MIN_HOLD_MS) {
            refuse(MESSAGES.tooShort);
            done("discarded");
            return;
        }
        const samples = resample(state.chunks, state.rate, TARGET_RATE);
        state.chunks = [];
        if (!samples.length) {
            refuse(MESSAGES.empty);
            done("discarded");
            return;
        }

        // Measured, then lifted, then measured against the floor. In that
        // order: a Bluetooth capture is routinely quiet enough to need the
        // gain and still loud enough to transcribe perfectly once it has it,
        // so refusing on the raw level would throw away recordings that are
        // fine. What is refused is what is still silent after the lift.
        const before = levelOf(samples);
        const applied = normalise(samples, before);
        const after = applied === 1 ? before : levelOf(samples);
        reportCapture(state, before, applied);
        if (after.peak < QUIET_PEAK) {
            refuse(lastCapture.narrowband ? MESSAGES.tooQuiet : MESSAGES.empty);
            done("discarded");
            return;
        }
        send(encodeWav(samples, TARGET_RATE));
        done("sent");
    }

    // One console line per recording, and a status line only when there is
    // something worth saying. The console line is what makes "my microphone
    // changed under me" a thing somebody can actually check: it names the
    // device, the rate the browser gave us, and the level that came out.
    //
    // Never the audio, and never a transcript — this file's half of I-6.
    function reportCapture(state, level, applied) {
        const track = state.track || {};
        try {
            console.info("Model Chain: Voice Chat captured "
                         + (track.label || "an unnamed microphone")
                         + " at " + (track.rate || state.rate || 0) + " Hz — peak "
                         + Math.round(level.peak * 100) + "%, rms "
                         + Math.round(level.rms * 1000) / 10 + "%"
                         + (applied > 1 ? ", lifted " + Math.round(applied * 10) / 10 + "x"
                                        : ""));
        } catch (error) { /* a console that will not take it is not a failure */ }
        // Remembered rather than said here. The status line is about to read
        // "Transcribing…" and would overwrite anything written now; what this
        // is for is the moment the answer comes back, which is where a note
        // about the microphone is actually useful.
        lastCapture = {narrowband: looksNarrowband(track),
                       label: track.label || "a Bluetooth microphone"};
    }

    let lastCapture = {narrowband: false, label: ""};

    // Where a slow microphone actually went, in one line of durations.
    //
    // Every number is milliseconds from the engagement, on this browser's own
    // monotonic clock, and together they separate the four things that used to
    // be one unexplained second: the browser's permission and hardware
    // (`microphone`), the module load (`worklet`), assembling the graph
    // (`graph`), the WebUI's status answer (`ready`) and the first callback
    // (`audio`). The one that matters most is the gap between `graph` and
    // `audio`, because that is the device warming up and nothing this page can
    // shorten -- and it is exactly the gap the control used to spend claiming
    // to be recording.
    //
    // No device name, no level, no samples: `reportCapture` above says what was
    // heard, and this says only how long it took to start hearing it.
    function reportCaptureTiming(session, state, result) {
        if (session.timed) return;
        session.timed = true;
        const since = function (mark) {
            return mark ? Math.round(mark - session.startedAt) : null;
        };
        const firstPcm = state && state.firstPcmAt;
        sendReport({
            kind: "capture",
            // From the press, because the press is where the microphone is now
            // asked for. Every one of these is a duration from that moment, so
            // together they separate the browser's permission and hardware from
            // the module load from the graph from the device simply taking a
            // while to produce its first sample.
            mic_request_ms: 0,
            stream_ready_ms: since(session.mediaAt),
            first_pcm_ms: since(firstPcm),
            engaged_ms: since(session.engagedAt),
            // How much of the utterance existed before the slide finished --
            // which is the whole of what revision 4 added, and the number that
            // says whether it was worth adding. On a discarded gesture it is
            // how much was thrown away.
            preroll_ms: (firstPcm && session.engagedAt && session.engagedAt > firstPcm)
                ? Math.round(session.engagedAt - firstPcm) : 0,
            recorded_ms: (session.recordingAt && session.releasedAt)
                ? Math.round(session.releasedAt - session.recordingAt) : null,
            graph: (state && state.graph) || "none",
            result: result || "discarded",
        });
    }
    let narrowbandSaid = false;

    // What to add to a status line when the microphone is the likely reason.
    // Once per session for a success -- a note repeated after every good
    // transcription is a note nobody reads -- and every time for a failure,
    // where it is the answer rather than a remark.
    function microphoneNote(failed) {
        if (!lastCapture.narrowband) return "";
        if (!failed && narrowbandSaid) return "";
        narrowbandSaid = true;
        return " Recording from " + lastCapture.label
            + " — Bluetooth microphones are narrowband and often misheard; the phone's own "
            + "microphone is usually much more accurate.";
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
                say(((payload && payload.error) || MESSAGES.failed) + microphoneNote(true),
                    "warn");
                return;
            }
            if (!insert(payload.text)) {
                say(MESSAGES.failed, "error");
                return;
            }
            say("Transcribed." + microphoneNote(false));
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

    // -- the slide ------------------------------------------------------------ //

    // Press-and-hold was the wrong gesture for a phone, and it was the wrong
    // gesture for the reason no amount of `touch-action` fixes: on Android a
    // long press belongs to the operating system before it belongs to a web
    // page. It raises the context menu, or the selection callout, over the
    // composer. So the microphone is not held any more, it is *moved*: it rests
    // at the left of a track two of its own widths across, recording starts
    // when it has been slid to the far right and is being held there, and
    // releasing -- anywhere -- ends the recording.
    //
    // Two things fall out of that and both are worth having. There is no
    // gesture left for the platform to claim. And opening a microphone has
    // stopped being something anybody does by brushing against a button, which
    // for a control that starts a recording is the right amount of deliberate.

    // Nine tenths of the travel, not all of it: the last few pixels of a slide
    // are where a fingertip's contact patch and the pointer's reported position
    // disagree, and a gesture that had to be perfect would be a gesture that
    // failed half the time on the surface it was designed for.
    const SLIDE_ENGAGE = 0.9;
    // What to assume when the track cannot be measured -- a panel that has not
    // been laid out yet, a component mid-render. Roughly one finger width, so
    // the gesture is still a slide rather than a tap.
    const SLIDE_FLOOR = 34;

    let sliding = null;
    let slideWired = false;

    function trackOf(button) {
        return byId(IDS.track)
            || (button && typeof button.closest === "function"
                ? button.closest(".mc-llm-voice-track") : null)
            || (button ? button.parentElement : null);
    }

    function slideTravel(button) {
        const track = trackOf(button);
        if (!track || !button) return SLIDE_FLOOR;
        const room = (track.clientWidth || 0) - (button.offsetWidth || 0);
        return room > 8 ? room : SLIDE_FLOOR;
    }

    // `transform` and not `left`: it moves the handle without moving anything
    // else, and the composer is the one surface in this panel that must not
    // reflow while somebody is using it.
    function slideTo(button, offset) {
        if (!button || !button.style) return;
        const wanted = offset ? "translateX(" + Math.round(offset) + "px)" : "";
        if (button.style.transform !== wanted) button.style.transform = wanted;
    }

    // Conditional for the reason `show` is: this is on the composer, which is
    // the surface every theme in this WebUI observes, and rewriting a class
    // attribute that is already right is work somebody else has to do.
    function trackClass(button, name, wanted) {
        const track = trackOf(button);
        if (!track || !track.classList) return;
        const has = track.classList.contains(name);
        if (wanted && !has) track.classList.add(name);
        else if (!wanted && has) track.classList.remove(name);
    }

    // Red, and only red. "Armed" used to be added the instant the slide reached
    // the end, which made the track say "recording" while the microphone was
    // still opening; it now means what its colour means.
    function armTrack(button, armed) {
        trackClass(button, "mc-llm-voice-armed", armed);
    }

    // Pinned at the right and waiting, in a colour that is not the recording
    // one. The feedback the gesture needs -- the handle has arrived and the
    // control has taken the press -- without the claim that audio is arriving.
    function engageTrack(button, engaged) {
        trackClass(button, "mc-llm-voice-engaged", engaged);
    }

    // Everything the control has to be put back to, from wherever it got to.
    // One function because there are five paths to it -- release, cancel, the
    // opening timeout, a readiness refusal and the recording cap -- and four of
    // them used to do a subset.
    function resetSlider() {
        const button = clickable(IDS.mic);
        sliding = null;
        markSliding(button, false);
        slideTo(button, 0);
        engageTrack(button, false);
        armTrack(button, false);
    }

    // While a finger is on the handle it moves with the finger and does not
    // ease; when it is let go it glides back. That is one class, and it is what
    // makes the drag feel attached and the release feel like a spring.
    function markSliding(button, active) {
        trackClass(button, "mc-llm-voice-sliding", active);
    }

    let gestures = 0;

    function beginSlide(event) {
        const button = clickable(IDS.mic);
        if (!button || sliding) return;
        // Unlocked on the press whatever the slide turns out to mean: a gesture
        // that is abandoned halfway should still have left playback able to
        // work, and this is the user gesture that permits that.
        unlock();
        gestures += 1;
        sliding = {
            x: (event && event.clientX) || 0,
            pointerId: event && event.pointerId,
            travel: slideTravel(button),
            engaged: false,
            // The identity everything asynchronous is checked against. A
            // getUserMedia, a status answer or a module load that resolves
            // after this gesture is over belongs to nothing, and section 5.5
            // says what to do about it: stop the tracks, attach nothing.
            token: gestures,
            startedAt: nowMs(),
            cancelled: false,
            done: false,
            capture: null,
            opening: null,
            refused: "",
            mediaAt: 0,
            workletAt: 0,
            graphAt: 0,
            readinessAt: 0,
            engagedAt: 0,
            recordingAt: 0,
            releasedAt: 0,
            openTimer: 0,
            timer: 0,
        };
        markSliding(button, true);
        // The microphone, here, in the same task as the press. Everything the
        // browser has to do before a sample exists -- permission, opening the
        // device, the stream, the graph -- used to sit *after* the slide, so a
        // user who began talking as they slid lost the first word of it. The
        // slide is that window, and what it produces is kept locally and
        // uploaded only if the gesture completes.
        openForGesture(sliding);
        try {
            if (event && event.pointerId !== undefined && button.setPointerCapture) {
                button.setPointerCapture(event.pointerId);
            }
        } catch (error) { /* a browser without pointer capture still works */ }
    }

    function moveSlide(event) {
        if (!sliding || sliding.engaged) return;
        const button = clickable(IDS.mic);
        if (!button) return;
        const moved = Math.max(0, Math.min(sliding.travel,
                                           ((event && event.clientX) || 0) - sliding.x));
        if (moved >= sliding.travel * SLIDE_ENGAGE) {
            // `engage` marks the session engaged itself, synchronously, and the
            // guard at the top of this function is what stops a second move
            // from arriving in the meantime. Setting it here as well was how
            // the commit stopped happening: `engage` saw a session that was
            // already engaged and returned without committing anything.
            //
            // Pinned at the end for as long as the hold lasts. Once a recording
            // has started, a finger drifting back down the track is a finger
            // drifting -- not a decision -- and a microphone that stopped and
            // started again under one continuous hold would be a worse control
            // than the one this replaced.
            slideTo(button, sliding.travel);
            engageTrack(button, true);
            engage();
            return;
        }
        slideTo(button, moved);
    }

    function endSlide(cancelled) {
        const session = sliding;
        resetSlider();
        if (!session) return;
        if (session.engaged) {
            endHold(!!cancelled);
            return;
        }
        // Let go short of the end. The microphone may well have been open and
        // samples may well have been arriving; none of it becomes an utterance,
        // none of it is uploaded, and the tracks stop now. Section 5.1's
        // abandoned path, and the reason opening the microphone early is not a
        // privacy change: what the gesture decides is not whether the device is
        // touched but whether anything leaves this page.
        abandon(session, "discarded");
        // And the control says what it wanted, because a gesture that does
        // nothing and explains nothing is the one that gets reported as broken.
        if (!cancelled && !session.refused) say(MESSAGES.slide);
    }

    function wireMicrophone() {
        const button = clickable(IDS.mic);
        if (!button || button.dataset.mcVoiceWired === "1") return;
        button.dataset.mcVoiceWired = "1";
        markMic("");
        markSliding(button, false);
        slideTo(button, 0);
        engageTrack(button, false);
        armTrack(button, false);

        button.addEventListener("pointerdown", function (event) {
            if (event.button !== undefined && event.button !== 0) return;
            if (event.preventDefault) event.preventDefault();
            beginSlide(event);
        });
        button.addEventListener("contextmenu", function (event) {
            if (event.preventDefault) event.preventDefault();
        });
        // A click handler that does nothing, so a browser which synthesises one
        // after the pointer sequence cannot make the button do anything twice.
        button.addEventListener("click", function (event) {
            if (event.preventDefault) event.preventDefault();
        });

        // The keyboard has no slide, and a control that can only be dragged is
        // a control somebody using a keyboard cannot use at all. Holding Space
        // or Enter is the same contract by the other route: hold to talk,
        // release to send.
        button.addEventListener("keydown", function (event) {
            const key = event && event.key;
            if (key !== " " && key !== "Spacebar" && key !== "Enter") return;
            if (event.repeat || sliding) return;
            if (event.preventDefault) event.preventDefault();
            // The same gesture by the other route, and it goes through the
            // same two steps: the press opens the microphone, and the
            // engagement commits what it produced. There is no slide here, so
            // the two happen in one task and the pre-roll is whatever the
            // device manages in it.
            beginSlide({clientX: 0});
            if (!sliding) return;
            sliding.keyboard = true;
            engageTrack(button, true);
            engage();
        });
        button.addEventListener("keyup", function (event) {
            const key = event && event.key;
            if (key !== " " && key !== "Spacebar" && key !== "Enter") return;
            if (!sliding || !sliding.keyboard) return;
            if (event.preventDefault) event.preventDefault();
            endSlide(false);
        });
        // Tabbing away mid-hold is not a message somebody meant to dictate.
        button.addEventListener("blur", function () {
            if (sliding && sliding.keyboard) endSlide(true);
        });

        wireSlideWindow();
    }

    // On the window and not on the button, and only ever once. Pointer capture
    // normally keeps the whole sequence on the handle, but where it is missing
    // or has been lost -- a finger that leaves the track, a task switch, an
    // incoming call -- these are what still end the recording rather than
    // leaving a microphone open. Both paths reaching them is fine: `moveSlide`
    // only computes, and the second `endSlide` finds nothing to end.
    function wireSlideWindow() {
        if (slideWired) return;
        slideWired = true;
        window.addEventListener("pointermove", function (event) {
            if (!sliding) return;
            if (event.cancelable && event.preventDefault) event.preventDefault();
            moveSlide(event);
        }, {passive: false});
        window.addEventListener("pointerup", function () {
            if (sliding) endSlide(false);
        });
        window.addEventListener("pointercancel", function () {
            if (sliding) endSlide(true);
        });
        window.addEventListener("blur", function () {
            if (sliding) endSlide(true);
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

    // -- noticing a new turn, without waiting for a poll ---------------------- //

    // Section 5.5's hierarchy, and the reason it is a hierarchy rather than one
    // mechanism.
    //
    //   1. Gradio writes these hidden fields and dispatches `input` on them, so
    //      a listener on the control itself is the fastest and cheapest signal
    //      there is -- zero to four hundred milliseconds of poll delay becomes
    //      one task.
    //   2. Gradio also *replaces* those controls when it re-renders the panel,
    //      and a listener on a node that is no longer in the document is a
    //      listener that will never fire again. An observer on the holder
    //      re-binds when that happens, and re-binding is idempotent -- the flag
    //      is on the field, so a control that survived a re-render is not bound
    //      a second time.
    //   3. The poll below stays. `input` is dispatched by Gradio and not by the
    //      DOM: a value assigned straight to `field.value` by a theme, a custom
    //      component or a future version emits nothing at all, and observing an
    //      input's *attributes* would not see it either -- `value` is an IDL
    //      property, and assigning it changes no attribute a MutationObserver
    //      is watching. So the poll is the recovery path, not the mechanism.

    const speechWatchers = {};

    function bindSpeechField(id) {
        const field = fieldOf(id);
        if (!field || !field.dataset || field.dataset.mcVoiceSpeech === "1") return;
        field.dataset.mcVoiceSpeech = "1";
        const look = function () { attempt("check for a reply to speak", checkSpeech); };
        field.addEventListener("input", look);
        field.addEventListener("change", look);
        // A control that arrived already carrying a value -- which is what a
        // re-render mid-reply looks like -- has no event left to emit.
        look();
    }

    function watchSpeechFields() {
        [IDS.turn, IDS.token].forEach(function (id) {
            bindSpeechField(id);
            const holder = byId(id);
            if (!holder || typeof MutationObserver !== "function") return;
            const seen = speechWatchers[id];
            if (seen && seen.holder === holder) return;
            if (seen && seen.observer) {
                try { seen.observer.disconnect(); } catch (error) { /* already gone */ }
            }
            const observer = new MutationObserver(function () {
                attempt("re-bind the reply watcher", function () { bindSpeechField(id); });
            });
            try {
                observer.observe(holder, {childList: true, subtree: true});
                speechWatchers[id] = {holder: holder, observer: observer};
            } catch (error) {
                speechWatchers[id] = {holder: holder, observer: null};
            }
        });
    }

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
        // `engine_state`, not `engine`: `engine` is the selected engine's id in
        // every payload this feature sends, and one key meaning two things is
        // how the residency line came to be replaced by the string "sopro".
        const engine = payload.engine_state || {};
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
            const state = (payload && payload.engine_state
                           && payload.engine_state.state) || "unloaded";
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
        // The tier cards have one of these each, and they answer to a scope of
        // their own -- `paintTiers` owns those. This is for the rows that do
        // not: the engine and Kokoro. Filtered rather than expressed as a
        // `:not()` selector because the distinction is "has a model attribute",
        // and a selector that says so is one more thing every caller has to
        // spell identically.
        const found = Array.prototype.filter.call(
            holder.querySelectorAll("[data-mc-voice-local]"), function (button) {
                return button.getAttribute("data-mc-voice-local") === kind
                    && !button.getAttribute("data-mc-voice-model");
            });
        return found[0] || null;
    }

    function tierCard(holder, model) {
        return holder ? holder.querySelector('[data-mc-voice-tier="' + cssEscape(model)
                                             + '"]') : null;
    }

    function sayInCard(card, text, bad) {
        const line = card ? card.querySelector("[data-mc-voice-tier-state]") : null;
        if (!line) return;
        line.textContent = text;
        setClass(line, "mc-voice-failed", !!bad);
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

    // -- the three speech-to-text qualities ---------------------------------- //
    //
    // Two buttons per card and they do different things. Download fetches that
    // tier; Use points Voice Chat at it. Separate because keeping all three on
    // disk and switching between them should not be a download, and because
    // choosing the high tier and *then* starting its download is the order
    // people actually do it in.
    //
    // Everything about the cards is painted from `/voice/models`, which answers
    // with what is on disk beside what is offered — joined on the server, so a
    // row can never end up saying "Installed" against the wrong tier.

    function tiersHolder(holder) {
        return holder ? holder.querySelector('[data-mc-voice-tiers="stt"]') : null;
    }

    function paintTiers(holder, payload) {
        const list = tiersHolder(holder);
        if (!list || !payload || !payload.ok) return;
        const chosen = payload.chosen || "";
        const progress = payload.progress || {};
        (payload.models || []).forEach(function (entry) {
            const card = list.querySelector('[data-mc-voice-tier="' + cssEscape(entry.id)
                                            + '"]');
            if (!card) return;
            const running = progress.running && progress.model === entry.id;
            setClass(card, "mc-voice-tier-chosen", entry.id === chosen);
            const mark = card.querySelector("[data-mc-voice-tier-mark]");
            if (mark) mark.textContent = entry.id === chosen ? "In use" : "";
            const state = card.querySelector("[data-mc-voice-tier-state]");
            if (state) {
                state.textContent = running
                    ? (progress.text || "Working…") + "  "
                      + Math.round((progress.fraction || 0) * 100) + "%"
                    : (progress.failed && progress.model === entry.id
                       ? progress.text : entry.message || "");
                setClass(state, "mc-voice-failed",
                         !!(progress.failed && progress.model === entry.id && !running));
            }
            const install = card.querySelector("[data-mc-voice-tier-install]");
            if (install) {
                install.disabled = !!running;
                install.textContent = running ? "Downloading…"
                    : entry.installed ? "Download again" : "Download";
            }
            const use = card.querySelector("[data-mc-voice-tier-use]");
            if (use) {
                use.disabled = entry.id === chosen;
                use.textContent = entry.id === chosen ? "In use" : "Use this";
            }
            const local = card.querySelector("[data-mc-voice-local]");
            if (local && !running) {
                local.disabled = false;
                local.textContent = entry.installed ? "Reinstall from a folder"
                                                    : "Install from this folder";
            } else if (local) {
                local.disabled = true;
                local.textContent = "Installing…";
            }
        });
        const label = holder.querySelector('[data-mc-voice-chosen="stt"]');
        const current = (payload.models || []).filter(function (entry) {
            return entry.id === chosen;
        })[0];
        if (label && current) label.textContent = current.label || "";
    }

    function refreshTiers(holder, select) {
        // Nothing asked for a row that has no cards in it -- an unsupported
        // platform draws the heading and the reason and no tiers, and a poll
        // that fetched a catalogue for it would be a request with no reader.
        if (!tiersHolder(holder)) return Promise.resolve(null);
        return post(ROUTES.models, select ? {kind: "stt", select: select} : {kind: "stt"},
                    holder).then(function (payload) {
            paintTiers(holder, payload);
            return payload;
        });
    }

    function wireTiers(holder) {
        const list = tiersHolder(holder);
        if (!list) return;
        list.addEventListener("click", function (event) {
            const use = event.target.closest("[data-mc-voice-tier-use]");
            if (use) {
                event.preventDefault();
                use.disabled = true;
                refreshTiers(holder, use.getAttribute("data-mc-voice-tier-use"))
                    .then(function () { schedulePaint(holder, 0); });
                return;
            }
            const install = event.target.closest("[data-mc-voice-tier-install]");
            if (install) {
                event.preventDefault();
                startInstall(holder, "stt", install, "",
                             install.getAttribute("data-mc-voice-tier-install"));
            }
        });
        whenOnScreen(holder, function () { refreshTiers(holder, ""); });
    }

    // `CSS.escape` where the browser has it, and a conservative fallback where
    // it does not. The ids come from this extension's own manifest and are
    // ordinary words and hyphens, so the fallback is never actually exercised —
    // it is here so that a manifest edited later cannot produce a selector that
    // silently matches nothing.
    function cssEscape(value) {
        const text = String(value || "");
        if (window.CSS && typeof window.CSS.escape === "function") {
            return window.CSS.escape(text);
        }
        return text.replace(/[^\w-]/g, "\\$&");
    }

    function setClass(node, name, wanted) {
        if (!node || !node.classList) return;
        // Conditional for the reason `show` is: rewriting a class attribute that
        // is already right wakes every MutationObserver on this page, and a
        // theme is one of them.
        if (wanted && !node.classList.contains(name)) node.classList.add(name);
        else if (!wanted && node.classList.contains(name)) node.classList.remove(name);
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
        //
        // Sopro's two folder buttons are deliberately excluded: they speak a
        // different route with a different body, and `wireSopro` handles them.
        // Attaching this handler to them as well would post every folder twice,
        // once to a route that would refuse it.
        Array.prototype.forEach.call(
            holder.querySelectorAll("[data-mc-voice-local]"), function (button) {
                if ((button.getAttribute("data-mc-voice-local") || "")
                        .indexOf("sopro-") === 0) {
                    return;
                }
                button.addEventListener("click", function (event) {
                    if (event.preventDefault) event.preventDefault();
                    const kind = button.getAttribute("data-mc-voice-local");
                    const model = button.getAttribute("data-mc-voice-model") || "";
                    const scope = button.getAttribute("data-mc-voice-scope") || kind;
                    const box = holder.querySelector(
                        '[data-mc-voice-folder="' + cssEscape(scope) + '"]');
                    const folder = box ? (box.value || "").trim() : "";
                    if (!folder) {
                        const complaint = "Type the folder the downloaded files are in, "
                            + "then press this again.";
                        if (model) sayInCard(tierCard(holder, model), complaint, true);
                        else sayInRow(holder, kind, complaint, true);
                        return;
                    }
                    startInstall(holder, kind, button, folder, model);
                });
            });

        wireTiers(holder);
        wireEngineSelector(holder);
        wireSopro(holder);
        wireCleanup(holder);
        wireValidate(holder);
        schedulePaint(holder, 0);
    }

    // -- the engine selector --------------------------------------------------- //
    //
    // One engine speaks for the whole WebUI at a time, and choosing one is a
    // runtime boundary rather than a preference: the server cancels any speech,
    // stops whichever TTS worker was running and persists the choice before it
    // answers.
    //
    // The surface is then *replaced* rather than repainted. The design rule is
    // that the inactive engine's operational controls are absent from the
    // document rather than hidden in it, and `swapSurface` is what delivers
    // that here: the server builds the markup for the engine selected now and
    // the old nodes are thrown away, so no stale node can be found by a
    // selector afterwards.

    function wireEngineSelector(holder) {
        const panel = holder.querySelector("[data-mc-voice-engines]");
        if (!panel || panel.dataset.mcVoiceWired === "1") return;
        panel.dataset.mcVoiceWired = "1";

        // A listener per card rather than one on the panel. Delegation would be
        // the tidier JavaScript, and the buttons beside these -- Download, Use
        // this, Install from this folder -- are all wired the same direct way,
        // so this matches what the rest of this row does.
        const cards = panel.querySelectorAll("[data-mc-voice-engine-pick]");
        Array.prototype.forEach.call(cards, function (pick) {
            pick.addEventListener("click", function (event) {
                if (event.preventDefault) event.preventDefault();
                // The selected engine's card is disabled, so a second press is
                // not a second switch -- which would cancel speech and stop a
                // worker for nothing.
                if (pick.disabled) return;
                const wanted = pick.getAttribute("data-mc-voice-engine-pick");
                Array.prototype.forEach.call(cards, function (button) {
                    button.disabled = true;
                });
                post(ROUTES.engineSelect, {engine: wanted}, holder).then(function (payload) {
                    if (payload && payload.ok) {
                        swapSurface(true);
                        return;
                    }
                    // Put the cards back. A page that disabled both and then
                    // did not reload would be a page you cannot get back to the
                    // engine you were on.
                    Array.prototype.forEach.call(cards, function (button) {
                        button.disabled = button.getAttribute("data-mc-voice-engine-pick")
                            === (payload && payload.active);
                    });
                    sayInRow(holder, "engines",
                             (payload && payload.error)
                                 || "The engine could not be changed.", true);
                });
            });
        });
    }

    // -- Sopro ----------------------------------------------------------------- //

    let soproTimer = 0;

    // The recording-cleanup row: install, and a poll while it is installing.
    //
    // Its own wiring rather than the generic `[data-mc-voice-install]` handler,
    // because that one posts a *kind* to Kokoro's install route and this engine
    // has a route of its own. Which is exactly how it came to be unwired: the
    // markup, the route, the module, the worker and the runtime were all there,
    // and pressing the button did nothing at all.
    let cleanupTimer = 0;

    // -- Sopro validation ----------------------------------------------------- //

    // Section 20's measurement, as a button. It changes nothing: it starts
    // processes, times them, and writes a table to model_chain.log. So the only
    // thing this has to get right is not lying about what is happening -- a
    // sweep takes minutes, and a button that looked idle for four of them would
    // be pressed again.
    let validateTimer = 0;

    function wireValidate(holder) {
        const row = holder.querySelector("[data-mc-voice-sopro-validate-row]");
        if (!row || row.dataset.mcVoiceWired === "1") return;
        row.dataset.mcVoiceWired = "1";

        const button = row.querySelector("[data-mc-voice-sopro-validate]");
        if (button) {
            button.addEventListener("click", function (event) {
                if (event.preventDefault) event.preventDefault();
                if (button.disabled) return;
                button.disabled = true;
                setText(row, "[data-mc-voice-sopro-validate-status]", "Starting…");
                post(ROUTES.soproValidate, {start: true}, holder).then(function (payload) {
                    if (!payload || !payload.ok) {
                        button.disabled = false;
                        setText(row, "[data-mc-voice-sopro-validate-status]",
                                (payload && payload.error)
                                || "That could not be started.");
                        return;
                    }
                    paintValidate(row, payload);
                    pollValidate(holder, row, 900);
                }).catch(function () {
                    button.disabled = false;
                    setText(row, "[data-mc-voice-sopro-validate-status]",
                            "That could not be started.");
                });
            });
        }
        // Picks up a sweep started before this panel was drawn -- a reopened
        // flyout, or a second tab -- rather than showing an idle button beside
        // a run that is minutes from finishing.
        whenOnScreen(row, function () { pollValidate(holder, row, 0); });
    }

    function pollValidate(holder, row, delay) {
        window.clearTimeout(validateTimer);
        validateTimer = window.setTimeout(function () {
            post(ROUTES.soproValidate, {}, holder).then(function (payload) {
                paintValidate(row, payload);
                if (payload && payload.running) pollValidate(holder, row, 1500);
            }).catch(function () { /* the next paint tries again */ });
        }, Math.max(0, delay || 0));
    }

    // The table is drawn here as well as logged. The log is what gets sent on,
    // but somebody who just waited four minutes should not have to go and find
    // a file to see the answer.
    function validateTable(rows) {
        const head = "precision  threads    rate   RTF@7s  break-even";
        const body = rows.map(function (row) {
            const name = String(row.precision || "?");
            const threads = String(row.threads || row.asked_threads || "?");
            if (row.rtf === null || row.rtf === undefined) {
                return pad(name, 9) + "  " + pad(threads, 7) + "  "
                    + (row.error || "no result");
            }
            return pad(name, 9) + "  " + pad(threads, 7) + "  "
                + pad(row.rate.toFixed(3), 6) + "  " + pad(row.rtf.toFixed(3), 7) + "  "
                + pad((row.break_even_speed || 0).toFixed(2) + "x", 10);
        });
        return [head].concat(body).join("\n");
    }

    function pad(text, width) {
        let found = String(text);
        while (found.length < width) found = " " + found;
        return found;
    }

    function paintValidate(row, payload) {
        if (!row || !payload || !payload.ok) return;
        const running = !!payload.running;
        const button = row.querySelector("[data-mc-voice-sopro-validate]");
        if (button) {
            button.disabled = running;
            button.textContent = running ? "Measuring…" : "Run validation";
        }
        let status = payload.message || "";
        if (payload.error) status = payload.error;
        else if (running && payload.total) {
            status = "Step " + payload.step + " of " + payload.total
                + (payload.message ? " — " + payload.message : "");
        } else if (!running && payload.done && payload.best) {
            status = "Fastest here: " + payload.best.precision + " at "
                + payload.best.threads + " threads, RTF "
                + Number(payload.best.rtf).toFixed(3) + " — clean up to Speed "
                + Number(payload.best.break_even_speed).toFixed(2)
                + "x. The full table is in model_chain.log.";
        }
        setText(row, "[data-mc-voice-sopro-validate-status]", status);
        const table = row.querySelector("[data-mc-voice-sopro-validate-table]");
        if (table) {
            const rows = payload.rows || [];
            table.hidden = !rows.length;
            if (rows.length) table.textContent = validateTable(rows);
        }
    }

    function wireCleanup(holder) {
        const row = holder.querySelector('[data-mc-voice-kind="cleanup"]');
        if (!row || row.dataset.mcVoiceWired === "1") return;
        row.dataset.mcVoiceWired = "1";

        const install = row.querySelector("[data-mc-voice-cleanup-install]");
        if (install) {
            install.addEventListener("click", function (event) {
                if (event.preventDefault) event.preventDefault();
                if (install.disabled) return;
                install.disabled = true;
                setText(row, "[data-mc-voice-status]", "Starting…");
                post(ROUTES.cleanupInstall, {}, holder).then(function (payload) {
                    if (!payload || !payload.ok) {
                        install.disabled = false;
                        setText(row, "[data-mc-voice-status]",
                                (payload && payload.error) || "That could not be started.");
                        return;
                    }
                    pollCleanup(holder, row, 800);
                }).catch(function () {
                    install.disabled = false;
                    setText(row, "[data-mc-voice-status]", "That could not be started.");
                });
            });
        }
        whenOnScreen(row, function () { pollCleanup(holder, row, 0); });
    }

    function pollCleanup(holder, row, delay) {
        window.clearTimeout(cleanupTimer);
        cleanupTimer = window.setTimeout(function () {
            post(ROUTES.cleanup, {}, holder).then(function (payload) {
                paintCleanup(row, payload);
                const progress = payload && payload.progress && payload.progress.cleanup;
                // Only while something is running. A quarter of a gigabyte takes
                // minutes and says so; an idle row asks for nothing.
                if (progress && progress.running) pollCleanup(holder, row, 1200);
            }).catch(function () { /* the next paint tries again */ });
        }, Math.max(0, delay || 0));
    }

    function paintCleanup(row, payload) {
        if (!row || !payload || !payload.ok) return;
        setText(row, "[data-mc-voice-cleanup-runtime]", payload.runtime_message);
        setText(row, "[data-mc-voice-cleanup-model]", payload.model_message);
        const progress = (payload.progress && payload.progress.cleanup) || {};
        const running = !!progress.running;
        setText(row, "[data-mc-voice-status]",
                running ? (progress.text || "Installing…") : payload.message);
        const bar = row.querySelector("[data-mc-voice-progress-bar]");
        const track = row.querySelector("[data-mc-voice-progress]");
        if (track) track.hidden = !running;
        if (bar) bar.style.width = Math.round((progress.fraction || 0) * 100) + "%";
        const install = row.querySelector("[data-mc-voice-cleanup-install]");
        if (install) {
            install.disabled = running || payload.installed || !payload.platform_supported;
            install.textContent = payload.installed ? "Installed" : "Install cleanup";
        }
    }

    function wireSopro(holder) {
        const row = holder.querySelector('[data-mc-voice-kind="sopro"]');
        if (!row || row.dataset.mcVoiceWired === "1") return;
        row.dataset.mcVoiceWired = "1";

        const install = row.querySelector("[data-mc-voice-sopro-install]");
        if (install) {
            install.addEventListener("click", function (event) {
                if (event.preventDefault) event.preventDefault();
                install.disabled = true;
                post(ROUTES.soproInstall, {}, holder).then(function (payload) {
                    if (payload && payload.error) {
                        install.disabled = false;
                        setText(row, "[data-mc-voice-status]", payload.error);
                        return;
                    }
                    pollSopro(holder, row, 1200);
                });
            });
        }

        // The manual half. `startInstall` speaks the Kokoro install route, and
        // Sopro's is a different route with a different body, so the two Sopro
        // folder buttons are handled here rather than being made to look like
        // Kokoro rows that they are not.
        Array.prototype.forEach.call(
            holder.querySelectorAll('[data-mc-voice-local^="sopro-"]'), function (button) {
                button.addEventListener("click", function (event) {
                    if (event.preventDefault) event.preventDefault();
                    const kind = button.getAttribute("data-mc-voice-local");
                    const scope = button.getAttribute("data-mc-voice-scope") || kind;
                    const box = holder.querySelector(
                        '[data-mc-voice-folder="' + cssEscape(scope) + '"]');
                    const folder = box ? (box.value || "").trim() : "";
                    if (!folder) {
                        setText(row, "[data-mc-voice-status]",
                                "Type the folder the downloaded files are in, then press "
                                + "this again.");
                        return;
                    }
                    button.disabled = true;
                    post(ROUTES.soproInstall,
                         {part: kind === "sopro-model" ? "model" : "runtime", folder: folder},
                         holder).then(function (payload) {
                        button.disabled = false;
                        if (payload && payload.error) {
                            setText(row, "[data-mc-voice-status]", payload.error);
                            return;
                        }
                        pollSopro(holder, row, 1200);
                    });
                });
            });

        const settings = row.querySelector("[data-mc-voice-sopro-settings]");
        if (settings) {
            settings.addEventListener("change", function (event) {
                const select = event.target.closest("[data-mc-voice-sopro-setting]");
                if (!select) return;
                const body = {};
                body[select.getAttribute("data-mc-voice-sopro-setting")] = select.value;
                post(ROUTES.soproSettings, body, holder).then(function (payload) {
                    paintSopro(row, payload);
                });
            });
        }
        whenOnScreen(row, function () { pollSopro(holder, row, 0); });
    }

    function pollSopro(holder, row, delay) {
        window.clearTimeout(soproTimer);
        soproTimer = window.setTimeout(function () {
            post(ROUTES.sopro, {}, holder).then(function (payload) {
                paintSopro(row, payload);
                const progress = payload && payload.progress && payload.progress.sopro;
                // Only while something is running. A permanent poll on a
                // settings page is a request a second for as long as the tab
                // stays open, which is what the Kokoro rows already refuse to do.
                if (progress && progress.running) pollSopro(holder, row, 1200);
            }).catch(function () { /* the next paint tries again */ });
        }, Math.max(0, delay || 0));
    }

    function setText(root_, selector, text) {
        const node = root_ ? root_.querySelector(selector) : null;
        if (node) node.textContent = text || "";
    }

    function paintSopro(row, payload) {
        if (!row || !payload || !payload.ok) return;
        const progress = (payload.progress && payload.progress.sopro) || {};
        setText(row, '[data-mc-voice-status="sopro"]',
                progress.running ? progress.text
                    : (progress.failed && progress.text) || soproMessage(payload));
        setText(row, "[data-mc-voice-sopro-runtime]", payload.runtime_message);
        setText(row, "[data-mc-voice-sopro-model]", payload.model_message);
        const install = row.querySelector("[data-mc-voice-sopro-install]");
        if (install) {
            install.disabled = !!progress.running || !payload.platform_supported;
            install.textContent = payload.installed ? "Installed" : "Install Sopro";
        }
    }

    function soproMessage(payload) {
        if (!payload.platform_supported) return payload.runtime_message || "";
        if (payload.installed) return "Installed.";
        return payload.runtime_message || payload.model_message || "";
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
        if (stale) return;
        paintTimer = window.setTimeout(function () {
            if (document.hidden) {
                schedulePaint(holder, SETTINGS_IDLE_MS);
                return;
            }
            // Forge builds every settings row whether or not anybody has
            // opened the Settings tab, so "wired" is not "being looked at".
            // A row nobody can see is asked nothing at all, and is looked at
            // again a second later, which costs one DOM read.
            if (!onScreen(holder)) {
                schedulePaint(holder, 1000);
                return;
            }
            paintSettings(holder).then(function (payload) {
                schedulePaint(holder, nextPaintDelay(payload));
            });
        }, delay);
    }

    function startInstall(holder, kind, button, folder, model) {
        button.disabled = true;
        button.textContent = "Starting…";
        // A tier card has a state line of its own; the runtime and text-to-speech
        // rows have the shared one. Writing the shared one for a tier would put
        // "Starting…" above three cards and against none of them.
        const card = model ? tierCard(holder, model) : null;
        const say = function (text, bad) {
            if (card) sayInCard(card, text, bad);
            else sayInRow(holder, kind, text, bad);
        };
        say("Starting…", false);

        const failed = function (text) {
            say(text, true);
            if (!card) releaseButton(holder, kind, false);
            button.disabled = false;
            console.error("Model Chain: Voice Chat could not start the " + kind
                          + " install — " + text);
        };

        const body = {kind: kind};
        if (folder) body.folder = folder;
        if (model) body.model = model;
        fetch(url(ROUTES.install), {
            method: "POST",
            credentials: "same-origin",
            headers: headers({"Content-Type": "application/json"}, holder),
            body: JSON.stringify(body),
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
            // The three tiers redraw from their own route, which is the only
            // one that knows which of them is installed and which is in use.
            // Asked for here rather than on a timer of its own, so a download
            // that is running moves both this row's bar and the card's.
            refreshTiers(holder, "");
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
                // Speech to text has no button of its own any more -- its three
                // tier cards have one each, and `paintTiers` owns them.
                if (kind !== "stt") releaseButton(holder, kind, ready);
            });
            return payload;
        });
    }

    // -- the character's own voice ------------------------------------------- //
    //
    // The same list Settings draws, in a third of the room, inside the character
    // editor. It is painted from the same `/voice/voices` route for the same
    // reason: the list changes when a clone finishes or a voice is renamed, and
    // a copy built into the page would go stale.
    //
    // The selection is not this file's to keep. It goes into a hidden Gradio
    // textbox, and the ordinary Save character button reads it with everything
    // else — so there is no second save, no second store, and nothing to get
    // out of step with the character file.
    //
    // The first row is "the default voice" rather than a voice. A character that
    // names no voice follows Settings, and that has to be something somebody can
    // choose *back*, not only a state they start in.

    function pickerHolder() {
        return byId(IDS.characterVoiceList) || null;
    }

    function pickerValue() {
        return (fieldValue(IDS.characterVoice) || "").trim();
    }

    function setPickerValue(wanted) {
        const holder = byId(IDS.characterVoice);
        if (!holder) return false;
        const field = holder.tagName === "TEXTAREA" || holder.tagName === "INPUT"
            ? holder : holder.querySelector("textarea, input");
        if (!field) return false;
        field.value = wanted;
        // Assign, then tell it, on both events Gradio listens to. The pattern
        // this repository already uses everywhere the browser writes into a
        // Gradio control -- see `insert`.
        field.dispatchEvent(new Event("input", {bubbles: true}));
        field.dispatchEvent(new Event("change", {bubbles: true}));
        return true;
    }

    function pickerRow(entry, chosen) {
        const row = document.createElement("div");
        row.className = "mc-voice-pick";
        row.setAttribute("data-mc-voice-pick", entry.id);
        if (entry.id === chosen) row.classList.add("mc-voice-pick-chosen");

        const name = document.createElement("button");
        name.type = "button";
        name.className = "mc-voice-pick-name";
        name.setAttribute("data-mc-voice-pick-choose", entry.id);
        // The asterisk is presentation and only presentation -- section 41. It
        // is not in the display name, not in the id, and not in any filename.
        name.textContent = (entry.official ? "" : "* ") + (entry.display_name || entry.label
                                                           || entry.id);
        row.appendChild(name);

        const play = document.createElement("button");
        play.type = "button";
        play.className = "mc-voice-pick-play";
        play.setAttribute("data-mc-voice-pick-test", entry.id);
        play.title = "Hear this voice";
        play.setAttribute("aria-label", "Hear this voice");
        play.textContent = "\u25b6";
        row.appendChild(play);
        return row;
    }

    function defaultRow(chosen, payload) {
        const entry = (payload.voices || []).filter(function (voice) {
            return voice.id === payload.default;
        })[0];
        return pickerRow({
            id: "",
            official: true,
            display_name: "Default voice"
                + (entry ? " (" + (entry.display_name || entry.label) + ")" : ""),
        }, chosen);
    }

    function paintPicker(holder, payload) {
        if (!holder || !payload || !payload.ok) return;
        const list = holder.querySelector("[data-mc-voice-picker-list]");
        const chosen = pickerValue();
        if (list) keepingPlace(list, function () {
            list.textContent = "";
            list.appendChild(defaultRow(chosen, payload));
            const groups = [
                ["American", (payload.voices || []).filter(function (v) {
                    return v.official && v.language === "en-US";
                })],
                ["British", (payload.voices || []).filter(function (v) {
                    return v.official && v.language === "en-GB";
                })],
                ["Custom", (payload.voices || []).filter(function (v) {
                    return !v.official;
                })],
            ];
            groups.forEach(function (group) {
                if (!group[1].length) return;
                const heading = document.createElement("div");
                heading.className = "mc-voice-pick-group";
                heading.textContent = group[0];
                list.appendChild(heading);
                group[1].forEach(function (entry) {
                    list.appendChild(pickerRow(entry, chosen));
                });
            });
        });
        holder.dataset.mcVoiceDefault = payload.default || "";
        markPicker(holder);
    }

    // Which row is highlighted, read off the hidden field rather than
    // remembered here. Python writes that field when a character is loaded, and
    // Gradio does not tell this file when it does -- so the highlight follows
    // the value on a tick rather than only on a click.
    function markPicker(holder) {
        if (!holder) return;
        const chosen = pickerValue();
        let seen = false;
        Array.prototype.forEach.call(
            holder.querySelectorAll("[data-mc-voice-pick]"), function (row) {
                const mine = row.getAttribute("data-mc-voice-pick") === chosen;
                if (mine) seen = true;
                setClass(row, "mc-voice-pick-chosen", mine);
            });
        const line = holder.querySelector("[data-mc-voice-picker-current]");
        if (!line) return;
        const row = holder.querySelector('[data-mc-voice-pick="' + cssEscape(chosen) + '"]'
                                         + " .mc-voice-pick-name");
        // A character naming a voice that is no longer installed says so rather
        // than showing nothing selected: it is still going to be spoken, in the
        // default voice, and that is worth knowing before the next reply.
        line.textContent = seen && row
            ? "Speaking as: " + row.textContent
            : (chosen ? "Speaking as: " + chosen + " — not installed, so the default voice "
                        + "will be used"
                      : "Speaking as: the default voice");
    }

    function refreshPicker(holder) {
        return post(ROUTES.voices, {}, holder).then(function (payload) {
            paintPicker(holder, payload);
            return payload;
        });
    }

    // The four sliders in the character editor are Gradio's, so their values are
    // read out of the DOM. Only sent when the checkbox beside them is on --
    // otherwise this character has no delivery of its own and an audition has to
    // demonstrate the default one.
    function characterProfile() {
        const custom = byId(IDS.characterVoiceCustom);
        const box = custom ? custom.querySelector('input[type="checkbox"]') : null;
        if (!box || !box.checked) return null;
        const found = {};
        ["speed", "pitch", "gain", "pause"].forEach(function (name) {
            const holder = byId(IDS.characterVoice + "-" + name);
            if (!holder) return;
            const input = holder.querySelector('input[type="number"], input[type="range"]');
            if (input && input.value !== "") found[name] = Number(input.value);
        });
        return Object.keys(found).length ? found : null;
    }

    function wirePicker() {
        const holder = pickerHolder();
        if (!holder || holder.dataset.mcVoiceWired === "1") return;
        holder.dataset.mcVoiceWired = "1";

        holder.addEventListener("click", function (event) {
            const choose = event.target.closest("[data-mc-voice-pick-choose]");
            if (choose) {
                event.preventDefault();
                setPickerValue(choose.getAttribute("data-mc-voice-pick-choose"));
                markPicker(holder);
                return;
            }
            const test = event.target.closest("[data-mc-voice-pick-test]");
            if (test) {
                event.preventDefault();
                test.disabled = true;
                const wanted = test.getAttribute("data-mc-voice-pick-test");
                unlock();
                const body = {voice: wanted || holder.dataset.mcVoiceDefault || "",
                              text: ""};
                const profile = characterProfile();
                if (profile) body.profile = profile;
                fetch(url(ROUTES.voiceTest), {
                    method: "POST",
                    credentials: "same-origin",
                    headers: headers({"Content-Type": "application/json"}, holder),
                    body: JSON.stringify(body),
                }).then(function (response) {
                    if (!response.ok) throw new Error("test");
                    return response.arrayBuffer();
                }).then(play).catch(function () {
                    const line = holder.querySelector("[data-mc-voice-picker-current]");
                    if (line) line.textContent = "That voice could not be played.";
                }).then(function () { test.disabled = false; });
            }
        });

        whenOnScreen(holder, function () {
            refreshPicker(holder);
            // A DOM read a second, which is what stands in for an event Gradio
            // does not give us: the hidden field changes when a character is
            // loaded, and the highlight has to follow it. Cheap, conditional,
            // and stopped when the screen is not on screen.
            window.setInterval(function () {
                if (stale || !onScreen(holder)) return;
                markPicker(holder);
            }, 1000);
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
        if (stale) return Promise.resolve({ok: false, error: RELOAD});
        return fetch(url(route), {
            method: "POST",
            credentials: "same-origin",
            headers: headers({"Content-Type": "application/json"}, holder),
            body: JSON.stringify(body || {}),
        }).then(refused).then(function (response) {
            return response.json().catch(function () {
                return {ok: false, error: "The WebUI did not answer."};
            });
        }).then(function (payload) {
            engineChanged(payload);
            return payload;
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
        // Offered only for a voice whose preparation no longer matches the
        // installed build and whose recording is still here. A stale voice
        // without one has to be created again, and a button that could not
        // succeed would be a button somebody presses three times.
        if (entry.compatible === false && entry.has_source) {
            actions.push(["rebuild", "Rebuild"]);
        }
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
        if (list) keepingPlace(list, function () {
            list.textContent = "";
            // Kokoro's list is grouped by accent because its bank is; Sopro's
            // is one list of voices somebody made, because that is what it is.
            // Driven by the payload rather than by a flag on the page, so a
            // list painted before a switch cannot end up with the other
            // engine's headings over this engine's voices.
            const groups = payload.engine === "sopro"
                ? [["Your voices", payload.voices]]
                : [
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
        });
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
                if (kind === "rebuild") {
                    action.disabled = true;
                    post(ROUTES.soproRebuild, {voice: voiceId}, holder)
                        .then(function (payload) {
                            action.disabled = false;
                            applyVoices(holder, payload);
                            if (payload && payload.audio) {
                                unlock();
                                play(base64ToBuffer(payload.audio))
                                    .catch(function () { /* silent */ });
                            }
                        });
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
        // Not on page load. This row is built into every page this WebUI
        // serves, and two requests fired from `onUiLoaded` are two requests
        // ahead of the first paint of a tab that has nothing to do with
        // speech. The list is fetched the first time somebody can see it.
        whenOnScreen(holder, function () {
            refreshVoices(holder);
            wireDelivery(holder);
            wireSoproStarter(holder);
            // Only ever present when Sopro is the selected engine, because the
            // markup they wire is only rendered then. Both are no-ops on a
            // Kokoro page rather than branches on the engine -- absence is the
            // scoping (section 5).
            wireSoproClone(holder);
            wireLab(holder);
        });
    }

    // -- how the default voice is delivered ---------------------------------- //
    //
    // Four sliders, written through on release rather than on every pixel of
    // drag: each write is a settings-file save, and a save per pixel would be
    // hundreds of them for one gesture. `input` moves the number beside the
    // slider so it still feels live; `change` is what persists.

    let deliveryControls = {};

    function deliveryHolder(holder) {
        return holder ? holder.querySelector("[data-mc-voice-delivery]") : null;
    }

    function readDelivery(holder) {
        const found = {};
        const panel = deliveryHolder(holder);
        if (!panel) return found;
        Array.prototype.forEach.call(
            panel.querySelectorAll("[data-mc-voice-slider-input]"), function (input) {
                found[input.getAttribute("data-mc-voice-slider-input")] = Number(input.value);
            });
        return found;
    }

    function paintDelivery(holder, payload) {
        const panel = deliveryHolder(holder);
        if (!panel || !payload || !payload.ok) return;
        deliveryControls = payload.controls || deliveryControls;
        const profile = payload.profile || {};
        Object.keys(profile).forEach(function (name) {
            const input = panel.querySelector('[data-mc-voice-slider-input="'
                                              + cssEscape(name) + '"]');
            // Never over a slider somebody has hold of. A repaint that moved the
            // control under a finger would be a control that fights its user.
            if (input && document.activeElement !== input) input.value = profile[name];
            showDeliveryValue(panel, name, profile[name]);
        });
        const summary = panel.querySelector("[data-mc-voice-delivery-summary]");
        if (summary) summary.textContent = payload.summary || "";
    }

    // The same label Python writes, built here because the slider has to move
    // its number without a round trip. `mc_voice_profile.value_label` is the
    // definition; this follows it, and the next paint from the server is what
    // corrects it if it ever drifts.
    function deliveryLabel(name, value) {
        const spec = deliveryControls[name];
        if (!spec) return String(value);
        const decimals = spec.decimals || 0;
        let text = Number(value).toFixed(decimals);
        if (decimals) text = text.replace(/0+$/, "").replace(/\.$/, "") || "0";
        const sign = (name === "pitch" || name === "gain") && Number(value) > 0 ? "+" : "";
        return sign + text + (spec.unit || "");
    }

    function showDeliveryValue(panel, name, value) {
        const output = panel.querySelector('[data-mc-voice-slider-value="'
                                           + cssEscape(name) + '"]');
        if (output) output.textContent = deliveryLabel(name, value);
    }

    function saveDelivery(holder) {
        return post(ROUTES.profile, {profile: readDelivery(holder)}, holder)
            .then(function (payload) {
                paintDelivery(holder, payload);
                return payload;
            });
    }

    function wireDelivery(holder) {
        const panel = deliveryHolder(holder);
        // Guarded, because `whenOnScreen` is allowed to answer more than once
        // and a second set of listeners would write the settings file twice for
        // every slider release.
        if (!panel || panel.dataset.mcVoiceWired === "1") return;
        panel.dataset.mcVoiceWired = "1";
        panel.addEventListener("input", function (event) {
            const input = event.target.closest("[data-mc-voice-slider-input]");
            if (!input) return;
            showDeliveryValue(panel, input.getAttribute("data-mc-voice-slider-input"),
                              input.value);
        });
        panel.addEventListener("change", function (event) {
            const input = event.target.closest("[data-mc-voice-slider-input]");
            if (!input) return;
            saveDelivery(holder);
        });
        panel.addEventListener("click", function (event) {
            if (event.target.closest("[data-mc-voice-delivery-reset]")) {
                event.preventDefault();
                Array.prototype.forEach.call(
                    panel.querySelectorAll("[data-mc-voice-slider-input]"),
                    function (input) {
                        const spec = deliveryControls[
                            input.getAttribute("data-mc-voice-slider-input")];
                        if (spec) input.value = spec.default;
                    });
                saveDelivery(holder);
                return;
            }
            const test = event.target.closest("[data-mc-voice-delivery-test]");
            if (test) {
                event.preventDefault();
                test.disabled = true;
                // Saved first, so what is auditioned is what is stored -- an
                // audition of unsaved slider positions would be a preview of a
                // setting nobody has.
                saveDelivery(holder).then(function () {
                    return audition(holder, "");
                }).catch(function () {
                    sayInHolder(holder, "That voice could not be played.");
                }).then(function () { test.disabled = false; });
            }
        });
        post(ROUTES.profile, {}, holder).then(function (payload) {
            paintDelivery(holder, payload);
        });
    }

    function sayInHolder(holder, text) {
        const warnings = holder.querySelector("[data-mc-voice-warnings]");
        if (!warnings) return;
        warnings.textContent = text;
        warnings.hidden = false;
    }

    function applyVoices(holder, payload) {
        if (payload && payload.error) sayInHolder(holder, payload.error);
        if (!payload) return;
        // The document was built for one engine and the server is answering for
        // another, which means somebody switched in a different tab. Reloading
        // is the only correct response: painting this answer into markup built
        // for the other engine is exactly the mixed surface the design forbids.
        const mine = holder.getAttribute("data-mc-voice-engine-id");
        if (payload.engine && mine && payload.engine !== mine) {
            swapSurface();
            return;
        }
        if (payload.voices) paintVoices(holder, payload);
    }

    function refreshVoices(holder) {
        post(ROUTES.voices, {}, holder).then(function (payload) {
            applyVoices(holder, payload);
        }).catch(function () { /* the next paint tries again */ });
        refreshCloning(holder);
    }

    // -- Sopro: making a voice from a recording -------------------------------- //
    //
    // Sopro's clone is not Storytime's. There is no separate bundle to install,
    // no background job to poll and nothing to abort: preparing a reference is
    // part of the model's ordinary capability, it runs in the same worker that
    // will later speak the voice, and it takes seconds. So this is one request
    // that returns the finished voice and an audition of it -- and the audition
    // is played immediately, because the point of validating through the
    // production path is that the user hears what Conversation will produce.

    let soproRecorder = null;

    // The chosen recording, decoded, with the part of it the user has picked.
    //
    // One object for both ways in. A file the browser can play and a take
    // recorded here arrive at exactly the same place, are trimmed with the same
    // controls, and leave as the same mono 16-bit PCM WAV -- so the server keeps
    // one narrow thing to validate and the person keeps every format they own.
    let soproClip = null;

    const CLIP_MIN_SECONDS = 5;
    const CLIP_MAX_SECONDS = 20;
    const CLIP_SUGGESTED = 15;

    function clipDuration() {
        if (!soproClip) return 0;
        return Math.max(0, soproClip.end - soproClip.start);
    }

    // Decode whatever was handed over. `decodeAudioData` is the browser's own
    // decoder, so this is every format it can play -- MP3, M4A, Opus, FLAC, OGG,
    // WebM and WAV -- without this extension shipping a codec or the WebUI
    // growing a dependency.
    function loadSoproClip(form, blob, label) {
        const ctx = audioContext();
        const trim = form.querySelector("[data-mc-voice-trim]");
        if (!ctx) {
            sayTrim(form, "This browser has no Web Audio, so a recording cannot be "
                    + "trimmed here.");
            return Promise.resolve(false);
        }
        sayTrim(form, "Reading " + (label || "the recording") + "…");
        return blob.arrayBuffer().then(function (bytes) {
            return new Promise(function (resolve, reject) {
                // The callback form, not the promise form: Safari still ships
                // only the callback one, and a voice reference is exactly the
                // file somebody brings from a phone.
                ctx.decodeAudioData(bytes, resolve, function (error) {
                    reject(error || new Error("decode failed"));
                });
            });
        }).then(function (decoded) {
            soproClip = {buffer: decoded, start: 0, end: decoded.duration,
                         label: label || "recording", how: "page",
                         engineCleaned: null};
            // A file is usually far longer than Sopro's window, so the opening
            // selection is a usable one rather than the whole thing: somebody
            // who presses Create straight away gets a voice, not a refusal.
            if (decoded.duration > CLIP_MAX_SECONDS) {
                soproClip.end = CLIP_SUGGESTED;
            }
            if (trim) trim.hidden = false;
            paintTrim(form);
            return true;
        }).catch(function () {
            soproClip = null;
            if (trim) trim.hidden = true;
            sayTrim(form, "That file could not be read as audio. Try a WAV, MP3, M4A, "
                    + "FLAC or OGG.");
            return false;
        });
    }

    function sayTrim(form, text) {
        const state = form.querySelector("[data-mc-voice-trim-state]");
        if (state) state.textContent = text || "";
    }

    // Min and max per pixel column, which is what a waveform is: an average
    // would draw a quiet grey band for speech and hide exactly the silences
    // somebody is trying to trim off the ends.
    function drawWave(form) {
        const canvas = form.querySelector("[data-mc-voice-wave]");
        if (!canvas || !soproClip) return;
        const width = Math.max(1, Math.floor(canvas.clientWidth || canvas.width || 600));
        const height = canvas.height || 96;
        if (canvas.width !== width) canvas.width = width;
        const ctx2d = canvas.getContext ? canvas.getContext("2d") : null;
        if (!ctx2d) return;
        const data = soproClip.buffer.getChannelData(0);
        const step = data.length / width;
        const ink = window.getComputedStyle(canvas).color || "#888";
        ctx2d.clearRect(0, 0, width, height);

        // The selection first, underneath, so the wave stays readable on top.
        const total = soproClip.buffer.duration || 1;
        const left = Math.round((soproClip.start / total) * width);
        const right = Math.round((soproClip.end / total) * width);
        ctx2d.globalAlpha = 0.18;
        ctx2d.fillStyle = ink;
        ctx2d.fillRect(left, 0, Math.max(1, right - left), height);
        ctx2d.globalAlpha = 1;

        ctx2d.fillStyle = ink;
        for (let column = 0; column < width; column += 1) {
            let low = 1;
            let high = -1;
            const from = Math.floor(column * step);
            const to = Math.min(data.length, Math.floor((column + 1) * step));
            for (let index = from; index < to; index += 1) {
                const value = data[index];
                if (value < low) low = value;
                if (value > high) high = value;
            }
            if (low > high) { low = 0; high = 0; }
            const top = (1 - high) * height / 2;
            const bottom = (1 - low) * height / 2;
            ctx2d.globalAlpha = (column >= left && column < right) ? 1 : 0.35;
            ctx2d.fillRect(column, top, 1, Math.max(1, bottom - top));
        }
        ctx2d.globalAlpha = 1;
        // The two edges, drawn last so they are never buried by a loud sample.
        ctx2d.fillRect(left, 0, 2, height);
        ctx2d.fillRect(Math.max(left + 2, right - 2), 0, 2, height);
    }

    function paintTrim(form) {
        if (!soproClip) return;
        const total = soproClip.buffer.duration;
        soproClip.start = Math.max(0, Math.min(soproClip.start, total));
        soproClip.end = Math.max(soproClip.start, Math.min(soproClip.end, total));
        const start = form.querySelector("[data-mc-voice-trim-start]");
        const end = form.querySelector("[data-mc-voice-trim-end]");
        if (start) {
            start.max = String(total.toFixed(1));
            if (document.activeElement !== start) start.value = soproClip.start.toFixed(1);
        }
        if (end) {
            end.max = String(total.toFixed(1));
            if (document.activeElement !== end) end.value = soproClip.end.toFixed(1);
        }
        drawWave(form);
        const chosen = clipDuration();
        const shown = chosen.toFixed(1) + " s of " + total.toFixed(1) + " s";
        if (chosen < CLIP_MIN_SECONDS) {
            sayTrim(form, shown + " selected — Sopro needs at least "
                    + CLIP_MIN_SECONDS + " s. Drag the edges wider.");
        } else if (chosen > CLIP_MAX_SECONDS) {
            sayTrim(form, shown + " selected — Sopro takes at most "
                    + CLIP_MAX_SECONDS + " s. Drag the edges in.");
        } else {
            sayTrim(form, shown + " selected — ready to create."
                    + (soproClip.clean
                       ? (soproClip.how === "deepfilternet" && soproClip.engineCleaned
                          ? " Cleaned with DeepFilterNet."
                          : " Cleaning is on.")
                       : ""));
        }
    }

    // The selection as one mono 16-bit PCM WAV at the source's own rate. Mono
    // because Sopro conditions on one speaker and the server would downmix it
    // anyway; the source rate because resampling twice is worse than once and
    // the server does the one that matters.
    // -- cleaning up a reference recording ------------------------------------ //
    //
    // Sopro clones what it is given, hiss included, and a phone recording made
    // in a room with a fan in it clones the fan. The obvious answer is
    // DeepFilterNet, which is not here yet rather than impossible: its Rust
    // extension ships wheels for CPython 3.10 and 3.11, so it needs an
    // interpreter of its own and a second copy of Torch beside the one Sopro
    // already has -- about 150 MB, and nothing that could be verified from the
    // workspace this was written in. See docs/17-voice-chat-sopro.md.
    //
    // So this is spectral subtraction, in the tab, on the selection that is
    // about to be uploaded. It is not a learned denoiser and does not pretend
    // to be: it takes out steady broadband noise -- hiss, hum, fan, room tone --
    // and leaves everything that is not steady alone. That is most of what a
    // bad reference recording suffers from, and it costs no dependency at all.

    const FFT_SIZE = 1024;
    const FFT_HOP = FFT_SIZE / 4;
    // Section 25. Numbers chosen for a cleaning pass that must not damage a
    // *cloning reference*, which is a stricter job than making a recording
    // pleasant to listen to: whatever this removes from the timbre, the model
    // conditions on the hole and reproduces it under every sentence forever.
    //
    // 2.5 was too much. Subtracting two and a half times the estimated noise
    // floor from every bin eats the low-energy parts of real speech -- consonant
    // bursts, the tails of vowels, breath -- and the -30 dB floor under it meant
    // the bins it emptied went almost silent, which is where musical noise comes
    // from. Both are pulled back: less removed, and what is removed is removed
    // less deeply.
    const NOISE_OVERSUBTRACT = 1.5;
    const NOISE_FLOOR_GAIN = 0.10;
    const GAIN_SMOOTHING = 0.5;
    // What fraction of frames, taken quietest-first by broadband energy, is
    // treated as containing no speech.
    const NOISE_FRAMES = 0.2;

    // In-place iterative radix-2, which is all a power-of-two window needs.
    function fft(re, im, inverse) {
        const n = re.length;
        for (let i = 1, j = 0; i < n; i += 1) {
            let bit = n >> 1;
            for (; j & bit; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j) {
                let swap = re[i]; re[i] = re[j]; re[j] = swap;
                swap = im[i]; im[i] = im[j]; im[j] = swap;
            }
        }
        for (let len = 2; len <= n; len <<= 1) {
            const angle = (inverse ? 2 : -2) * Math.PI / len;
            const wr = Math.cos(angle);
            const wi = Math.sin(angle);
            for (let start = 0; start < n; start += len) {
                let cr = 1;
                let ci = 0;
                for (let k = 0; k < len / 2; k += 1) {
                    const ar = re[start + k];
                    const ai = im[start + k];
                    const br = re[start + k + len / 2] * cr - im[start + k + len / 2] * ci;
                    const bi = re[start + k + len / 2] * ci + im[start + k + len / 2] * cr;
                    re[start + k] = ar + br;
                    im[start + k] = ai + bi;
                    re[start + k + len / 2] = ar - br;
                    im[start + k + len / 2] = ai - bi;
                    const nr = cr * wr - ci * wi;
                    ci = cr * wi + ci * wr;
                    cr = nr;
                }
            }
        }
        if (inverse) {
            for (let i = 0; i < n; i += 1) { re[i] /= n; im[i] /= n; }
        }
    }

    // Three things spectral subtraction cannot touch, because none of them is
    // noise: a click is one broken sample, clipping is a peak that was never
    // recorded, and a quiet recording is quiet everywhere including under the
    // speech. Each is ordinary in a phone recording and each survives a
    // denoiser perfectly intact.

    const CLICK_SPREAD = 4.0;
    const CLICK_FLOOR = 0.02;
    const CLIP_LEVEL = 0.985;
    const CLIP_MIN_RUN = 3;
    const TARGET_RMS = 0.1;
    const PEAK_CEILING = 0.95;

    // A second-difference test, which is what tells a click from loud audio.
    //
    // Band-limited sound moves smoothly between samples: the step it takes from
    // one to the next can be large, but the *change* in that step is small. A
    // click is the opposite -- one sample somewhere the waveform was never
    // going -- so it shows up as a second difference far larger than the first
    // differences on either side of it. Comparing against a plain amplitude
    // threshold instead would take the top off every plosive.
    function repairClicks(samples) {
        const found = Float32Array.from(samples);
        for (let index = 2; index < samples.length - 2; index += 1) {
            const predicted = (samples[index - 1] + samples[index + 1]) / 2;
            const local = Math.max(Math.abs(samples[index - 1] - samples[index - 2]),
                                   Math.abs(samples[index + 2] - samples[index + 1]));
            if (Math.abs(samples[index] - predicted)
                    > CLICK_SPREAD * local + CLICK_FLOOR) {
                found[index] = predicted;
            }
        }
        return found;
    }

    // A run of samples pinned at full scale is a peak the recorder could not
    // write down. Leaving the flat top in place leaves a square wave in the
    // reference -- broadband, harsh, and exactly the sort of thing a voice
    // model will faithfully learn -- so an arc goes back over it. The bulge is
    // bounded and grows with the length of the run, because a longer flat top
    // means more of the peak went missing.
    function repairClipping(samples) {
        const found = Float32Array.from(samples);
        let index = 0;
        while (index < found.length) {
            if (Math.abs(found[index]) < CLIP_LEVEL) { index += 1; continue; }
            let end = index;
            while (end < found.length && Math.abs(found[end]) >= CLIP_LEVEL) end += 1;
            const width = end - index;
            if (width >= CLIP_MIN_RUN && index > 0 && end < found.length) {
                const before = found[index - 1];
                const after = found[end];
                const sign = found[index] >= 0 ? 1 : -1;
                const bulge = Math.min(0.25, 0.02 * width);
                for (let step = 0; step < width; step += 1) {
                    const phase = (step + 1) / (width + 1);
                    found[index + step] = before + (after - before) * phase
                        + sign * bulge * Math.sin(Math.PI * phase);
                }
            }
            index = end;
        }
        return found;
    }

    // An RMS target with a peak ceiling, rather than peak normalisation alone.
    // Peak normalisation on a recording with one door slam in it turns the
    // speech down to make room for the slam; what a reference wants is a
    // consistent *speech* level, with the ceiling only there to stop it
    // clipping on the way out.
    function levelOut(samples) {
        let sum = 0;
        let peak = 0;
        for (let index = 0; index < samples.length; index += 1) {
            const value = samples[index];
            sum += value * value;
            const size = Math.abs(value);
            if (size > peak) peak = size;
        }
        const rms = Math.sqrt(sum / Math.max(1, samples.length));
        if (!(rms > 1e-6) || !(peak > 1e-6)) return samples;
        let gain = TARGET_RMS / rms;
        if (peak * gain > PEAK_CEILING) gain = PEAK_CEILING / peak;
        for (let index = 0; index < samples.length; index += 1) samples[index] *= gain;
        return samples;
    }

    // A one-pole high-pass at about 80 Hz. Rumble, handling and desk thumps all
    // live below where a voice does, and none of it survives being cloned into
    // something the model then reproduces under every sentence.
    function highPass(samples, rate) {
        const cut = 80;
        const rc = 1 / (2 * Math.PI * cut);
        const dt = 1 / rate;
        const alpha = rc / (rc + dt);
        const found = new Float32Array(samples.length);
        let last = 0;
        let lastIn = 0;
        for (let index = 0; index < samples.length; index += 1) {
            last = alpha * (last + samples[index] - lastIn);
            lastIn = samples[index];
            found[index] = last;
        }
        return found;
    }

    function denoise(input, rate) {
        // Repairs first: a click or a flat top is a broken sample rather than
        // noise, and feeding either into the noise estimate below teaches it
        // that broadband energy is normal here.
        const samples = highPass(repairClipping(repairClicks(input)), rate);
        const frames = Math.max(1, Math.floor((samples.length - FFT_SIZE) / FFT_HOP) + 1);
        if (samples.length < FFT_SIZE * 2) return samples;
        const bins = FFT_SIZE / 2 + 1;
        const window = new Float32Array(FFT_SIZE);
        for (let i = 0; i < FFT_SIZE; i += 1) {
            window[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / FFT_SIZE);
        }
        const magnitudes = [];
        const phasesRe = [];
        const phasesIm = [];
        for (let frame = 0; frame < frames; frame += 1) {
            const re = new Float64Array(FFT_SIZE);
            const im = new Float64Array(FFT_SIZE);
            const from = frame * FFT_HOP;
            for (let i = 0; i < FFT_SIZE; i += 1) re[i] = samples[from + i] * window[i];
            fft(re, im, false);
            const mag = new Float64Array(bins);
            for (let bin = 0; bin < bins; bin += 1) {
                mag[bin] = Math.sqrt(re[bin] * re[bin] + im[bin] * im[bin]);
            }
            magnitudes.push(mag);
            phasesRe.push(re);
            phasesIm.push(im);
        }

        // The noise floor, from the frames that hold no speech.
        //
        // This used to be a per-bin percentile over the whole clip: for each
        // bin independently, the 10th percentile of its magnitude over time.
        // That is not a noise floor, it is the tenth-quietest moment of that
        // frequency -- and in fifteen seconds of near-continuous speech the
        // tenth percentile of a vowel bin is quiet *speech*. So the estimate
        // was inflated by the voice, and then multiplied by the
        // over-subtraction factor and taken back out of the voice.
        //
        // Frames are ranked by broadband energy instead and the quietest fifth
        // -- the pauses, the breaths between phrases -- are averaged per bin.
        // That is an estimate of the room rather than of the speaker, which is
        // the thing being subtracted.
        const energies = [];
        for (let frame = 0; frame < frames; frame += 1) {
            let total = 0;
            for (let bin = 0; bin < bins; bin += 1) total += magnitudes[frame][bin];
            energies.push({frame: frame, total: total});
        }
        energies.sort(function (a, b) { return a.total - b.total; });
        const quiet = Math.max(1, Math.floor(frames * NOISE_FRAMES));
        const noise = new Float64Array(bins);
        for (let index = 0; index < quiet; index += 1) {
            const mag = magnitudes[energies[index].frame];
            for (let bin = 0; bin < bins; bin += 1) noise[bin] += mag[bin];
        }
        for (let bin = 0; bin < bins; bin += 1) noise[bin] /= quiet;

        const out = new Float32Array(samples.length);
        const weight = new Float32Array(samples.length);
        const gains = new Float64Array(bins);
        const smoothed = new Float64Array(bins);
        const previous = new Float64Array(bins);
        previous.fill(1);
        for (let frame = 0; frame < frames; frame += 1) {
            const mag = magnitudes[frame];
            const re = phasesRe[frame];
            const im = phasesIm[frame];
            for (let bin = 0; bin < bins; bin += 1) {
                const level = mag[bin];
                let gain = level > 0
                    ? (level - NOISE_OVERSUBTRACT * noise[bin]) / level
                    : 0;
                if (!(gain > NOISE_FLOOR_GAIN)) gain = NOISE_FLOOR_GAIN;
                if (gain > 1) gain = 1;
                gains[bin] = gain;
            }
            // Smoothed across frequency and then across time, which is what
            // keeps this from sounding worse than the noise it removed. An
            // unsmoothed gain flips between "keep" and "floor" bin by bin and
            // frame by frame, and the result is musical noise: a shimmer of
            // tones where there used to be honest hiss. Smoothing is what buys
            // the moderate over-subtraction above; without it the only way to
            // remove this much noise is to be aggressive enough to hear.
            for (let bin = 0; bin < bins; bin += 1) {
                const low = gains[Math.max(0, bin - 1)];
                const high = gains[Math.min(bins - 1, bin + 1)];
                smoothed[bin] = (low + gains[bin] + high) / 3;
            }
            for (let bin = 0; bin < bins; bin += 1) {
                const gain = GAIN_SMOOTHING * previous[bin]
                    + (1 - GAIN_SMOOTHING) * smoothed[bin];
                previous[bin] = gain;
                re[bin] *= gain;
                im[bin] *= gain;
                if (bin > 0 && bin < FFT_SIZE / 2) {
                    // The mirrored half, so the inverse transform comes back real.
                    re[FFT_SIZE - bin] = re[bin];
                    im[FFT_SIZE - bin] = -im[bin];
                }
            }
            fft(re, im, true);
            const from = frame * FFT_HOP;
            for (let i = 0; i < FFT_SIZE; i += 1) {
                out[from + i] += re[i] * window[i];
                weight[from + i] += window[i] * window[i];
            }
        }
        // The first and last window's worth of samples have only partial
        // overlap, so their weight is a fraction of the steady-state one and
        // dividing by it amplifies them into a spike at each end. That spike is
        // then the loudest thing in the clip, and the levelling below turns the
        // whole recording down to make room for it. So the divisor is floored
        // at half the steady-state weight, and the ends are faded.
        let full = 0;
        for (let index = 0; index < weight.length; index += 1) {
            if (weight[index] > full) full = weight[index];
        }
        const floor = Math.max(full * 0.5, 1e-6);
        for (let index = 0; index < out.length; index += 1) {
            out[index] /= Math.max(weight[index], floor);
        }
        const fade = Math.min(Math.floor(rate * 0.005), Math.floor(out.length / 4));
        for (let step = 0; step < fade; step += 1) {
            const ramp = step / fade;
            out[step] *= ramp;
            out[out.length - 1 - step] *= ramp;
        }
        return levelOut(out);
    }

    // The selection as mono samples, cleaned if that is what is showing. One
    // function, so what is played and what is uploaded cannot be different
    // things -- which is the only way "you can hear what you are sending" is a
    // true sentence rather than a hopeful one.
    // The selection, downmixed, and nothing else done to it. Split out of
    // `clipSamples` because asking for the untouched audio used to mean setting
    // `how` to "page" and calling the cleaning path -- which returns the
    // page's *denoised* audio, not the raw selection. `cleanWithEngine` did
    // exactly that, so choosing DeepFilterNet ran spectral subtraction first
    // and then handed the result to a learned denoiser as though it were a
    // recording. Two denoisers in series, the second one treating the first
    // one's musical noise as signal. That is the whole of "the cleanup makes
    // it worse" when the better engine was selected.
    function rawClip() {
        if (!soproClip) return null;
        const buffer = soproClip.buffer;
        const rate = buffer.sampleRate;
        const from = Math.max(0, Math.floor(soproClip.start * rate));
        const to = Math.min(buffer.length, Math.ceil(soproClip.end * rate));
        const length = Math.max(0, to - from);
        if (!length) return null;
        const mixed = new Float32Array(length);
        for (let channel = 0; channel < buffer.numberOfChannels; channel += 1) {
            const data = buffer.getChannelData(channel);
            for (let index = 0; index < length; index += 1) {
                mixed[index] += data[from + index];
            }
        }
        if (buffer.numberOfChannels > 1) {
            for (let index = 0; index < length; index += 1) {
                mixed[index] /= buffer.numberOfChannels;
            }
        }
        return mixed;
    }

    function clipSamples() {
        const mixed = rawClip();
        if (!mixed) return null;
        const rate = soproClip.buffer.sampleRate;
        if (!soproClip.clean) return mixed;
        // Cached against the exact selection *and* the method, so switching
        // between the two recomputes rather than serving the other one's work.
        const key = soproClip.start.toFixed(3) + ":" + soproClip.end.toFixed(3)
            + ":" + (soproClip.how || "page");
        if (soproClip.cleanedKey === key) return soproClip.cleaned;
        if ((soproClip.how || "page") !== "page") {
            // DeepFilterNet's answer, when it has been fetched. Asking for it
            // is asynchronous and this is not, so a selection whose answer has
            // not arrived yet plays and uploads the page's own pass rather than
            // silently doing nothing -- and `cleanWithEngine` repaints when it
            // lands.
            // Raw, not the page's pass, when the answer has not landed. The
            // substitute for "the engine you chose has not answered yet" must
            // never be "the engine you did not choose", or Create quietly
            // uploads something other than what the label says and what you
            // last heard.
            return soproClip.engineCleaned || mixed;
        }
        soproClip.cleaned = denoise(mixed, rate);
        soproClip.cleanedKey = key;
        return soproClip.cleaned;
    }

    function clipToWav() {
        const samples = clipSamples();
        if (!samples || !samples.length) return null;
        return encodeWav(samples, soproClip.buffer.sampleRate);
    }

    // Ask the cleanup engine for this selection, and remember what came back.
    //
    // A round trip rather than more arithmetic in the page: DeepFilterNet is a
    // learned model in a process of its own, on an interpreter of its own, and
    // the whole reason it exists here is that the page cannot do what it does.
    function cleanWithEngine(form, holder) {
        if (!soproClip) return Promise.resolve(false);
        const key = soproClip.start.toFixed(3) + ":" + soproClip.end.toFixed(3)
            + ":deepfilternet";
        if (soproClip.cleanedKey === key) return Promise.resolve(true);
        // The untouched selection. DeepFilterNet is a learned denoiser and it
        // wants a recording, not something another denoiser has already been
        // over.
        const plain = rawClip();
        if (!plain || !plain.length) return Promise.resolve(false);
        const rate = soproClip.buffer.sampleRate;
        sayTrim(form, "Cleaning with DeepFilterNet\u2026");
        const body = new FormData();
        body.append("name", "cleanup");
        body.append("language", "");
        body.append("reference",
                    new Blob([encodeWav(plain, rate)], {type: "audio/wav"}), "clip.wav");
        return fetch(url(ROUTES.cleanupRun), {
            method: "POST",
            credentials: "same-origin",
            headers: headers({}, holder),
            body: body,
        }).then(refused).then(function (response) {
            if (!response.ok) throw new Error("cleanup");
            return response.arrayBuffer();
        }).then(function (found) {
            const samples = wavToFloat32(found);
            if (!samples || !samples.length) throw new Error("empty");
            soproClip.engineCleaned = samples;
            soproClip.cleaned = samples;
            soproClip.cleanedKey = key;
            paintTrim(form);
            return true;
        }).catch(function () {
            soproClip.engineCleaned = null;
            sayTrim(form, "DeepFilterNet could not clean that, so the page's own cleanup "
                    + "is being used instead.");
            return false;
        });
    }

    // The mono PCM16 the cleanup route answers with, as samples.
    function wavToFloat32(buffer) {
        const view = new DataView(buffer);
        if (view.byteLength < 44) return null;
        let offset = 12;
        let start = 0;
        let length = 0;
        while (offset + 8 <= view.byteLength) {
            const id = String.fromCharCode(view.getUint8(offset), view.getUint8(offset + 1),
                                           view.getUint8(offset + 2), view.getUint8(offset + 3));
            const size = view.getUint32(offset + 4, true);
            if (id === "data") { start = offset + 8; length = size; break; }
            offset += 8 + size + (size % 2);
        }
        if (!length || start + length > view.byteLength) return null;
        const count = Math.floor(length / 2);
        const found = new Float32Array(count);
        for (let index = 0; index < count; index += 1) {
            const value = view.getInt16(start + index * 2, true);
            found[index] = value < 0 ? value / 0x8000 : value / 0x7fff;
        }
        return found;
    }

    // `which` is "raw" or "clean", and it is the whole point of there being two
    // buttons: A and B under two fingers, neither depending on the state of a
    // checkbox. "clean" is exactly what Create would upload, so the comparison
    // is the real one rather than an approximation of it.
    function playSelection(form, button, which) {
        if (!soproClip) return;
        const ctx = audioContext();
        if (!ctx) return;
        const label = button ? (button.dataset.mcVoiceLabel
                                || button.textContent) : "";
        if (button) button.dataset.mcVoiceLabel = label;
        // A second press stops, rather than starting a second copy over the
        // first. Pressing the other button while this one plays also stops it,
        // because both go through `stopPlayback`.
        const wasPlaying = playing;
        unlock();
        stopPlayback();
        resetPlayLabels(form);
        if (wasPlaying && button && button.dataset.mcVoicePlaying === "1") {
            button.dataset.mcVoicePlaying = "";
            return;
        }
        const samples = (which === "clean") ? clipSamples() : rawClip();
        if (!samples || !samples.length) return;
        const rate = soproClip.buffer.sampleRate;
        const held = ctx.createBuffer(1, samples.length, rate);
        held.getChannelData(0).set(samples);
        const source = ctx.createBufferSource();
        source.buffer = held;
        source.connect(ctx.destination);
        source.onended = function () {
            if (playing === source) playing = null;
            if (button) {
                button.textContent = label;
                button.dataset.mcVoicePlaying = "";
            }
        };
        playing = source;
        if (button) {
            button.textContent = "Stop";
            button.dataset.mcVoicePlaying = "1";
        }
        source.start(0);
    }

    function resetPlayLabels(form) {
        ["[data-mc-voice-trim-play]", "[data-mc-voice-trim-play-clean]"].forEach(
            function (selector) {
                const found = form.querySelector(selector);
                if (!found || !found.dataset.mcVoiceLabel) return;
                found.textContent = found.dataset.mcVoiceLabel;
                found.dataset.mcVoicePlaying = "";
            });
    }

    function wireTrim(form, holder) {
        const canvas = form.querySelector("[data-mc-voice-wave]");
        const play = form.querySelector("[data-mc-voice-trim-play]");
        const playClean = form.querySelector("[data-mc-voice-trim-play-clean]");
        const best = form.querySelector("[data-mc-voice-trim-best]");
        const start = form.querySelector("[data-mc-voice-trim-start]");
        const end = form.querySelector("[data-mc-voice-trim-end]");

        if (canvas) {
            let dragging = false;
            let anchor = 0;
            const at = function (event) {
                const box = canvas.getBoundingClientRect();
                const ratio = box.width ? (event.clientX - box.left) / box.width : 0;
                const total = soproClip ? soproClip.buffer.duration : 0;
                return Math.max(0, Math.min(total, ratio * total));
            };
            canvas.addEventListener("pointerdown", function (event) {
                if (!soproClip) return;
                dragging = true;
                anchor = at(event);
                soproClip.start = anchor;
                soproClip.end = anchor;
                if (canvas.setPointerCapture) canvas.setPointerCapture(event.pointerId);
                paintTrim(form);
            });
            canvas.addEventListener("pointermove", function (event) {
                if (!dragging || !soproClip) return;
                const here = at(event);
                soproClip.start = Math.min(anchor, here);
                soproClip.end = Math.max(anchor, here);
                paintTrim(form);
            });
            const release = function () { dragging = false; };
            canvas.addEventListener("pointerup", release);
            canvas.addEventListener("pointercancel", release);
        }
        if (play) {
            play.addEventListener("click", function (event) {
                if (event.preventDefault) event.preventDefault();
                if (playing) {
                    stopPlayback();
                    play.textContent = "Play selection";
                    return;
                }
                playSelection(form, play, "raw");
            });
        }
        if (playClean) {
            playClean.addEventListener("click", function (event) {
                if (event.preventDefault) event.preventDefault();
                if (!soproClip) return;
                // Cleaning is what this button means, so it turns it on rather
                // than refusing to play when the box is unticked. Anything else
                // makes the comparison a two-step.
                if (!soproClip.clean) {
                    soproClip.clean = true;
                    const box = form.querySelector("[data-mc-voice-clean]");
                    if (box) box.checked = true;
                    paintTrim(form);
                }
                if ((soproClip.how || "page") !== "page" && !soproClip.engineCleaned) {
                    // The engine's answer is what this button promises, so it
                    // waits for it rather than playing the other cleaner.
                    cleanWithEngine(form, holder).then(function () {
                        playSelection(form, playClean, "clean");
                    });
                    return;
                }
                playSelection(form, playClean, "clean");
            });
        }
        if (best) {
            best.addEventListener("click", function (event) {
                if (event.preventDefault) event.preventDefault();
                if (!soproClip) return;
                // The loudest fifteen seconds, which is the closest thing to
                // "the part with speech in it" that costs one pass over the
                // samples. Not clever, and much better than the first fifteen
                // when a clip opens with silence or a count-in.
                soproClip.start = loudestWindow(soproClip.buffer, CLIP_SUGGESTED);
                soproClip.end = Math.min(soproClip.buffer.duration,
                                         soproClip.start + CLIP_SUGGESTED);
                paintTrim(form);
            });
        }
        const clean = form.querySelector("[data-mc-voice-clean]");
        const how = form.querySelector("[data-mc-voice-clean-how]");
        if (playClean) playClean.hidden = false;
        const chose = function () {
            if (!soproClip) return;
            soproClip.clean = !!(clean && clean.checked);
            soproClip.how = how ? how.value : "page";
            paintTrim(form);
            if (soproClip.clean && soproClip.how !== "page") {
                cleanWithEngine(form, holder);
            }
        };
        if (clean) clean.addEventListener("change", chose);
        if (how) how.addEventListener("change", chose);
        // The choice is only a choice when there is something to choose. Asked
        // once, when the form is wired, rather than polled: installing the
        // engine is a deliberate act on another row and the page is repainted
        // when it finishes.
        if (how) {
            // Default to the better one when it is there. The page's own pass
            // is the fallback for an installation without the engine, not a
            // recommendation -- and leaving it selected meant somebody who had
            // deliberately installed DeepFilterNet still got spectral
            // subtraction unless they noticed a second dropdown.
            how.value = "page";
            post(ROUTES.cleanup, {}, holder).then(function (payload) {
                if (payload && payload.ok && payload.installed) {
                    how.hidden = false;
                    how.value = "deepfilternet";
                    if (soproClip) soproClip.how = "deepfilternet";
                }
            }).catch(function () { /* not installed is the ordinary answer */ });
        }
        [[start, "start"], [end, "end"]].forEach(function (pair) {
            if (!pair[0]) return;
            pair[0].addEventListener("input", function () {
                if (!soproClip) return;
                const value = parseFloat(pair[0].value);
                if (!isFinite(value)) return;
                soproClip[pair[1]] = value;
                if (soproClip.end < soproClip.start) {
                    soproClip.end = soproClip.start;
                }
                paintTrim(form);
            });
        });
        window.addEventListener("resize", function () {
            if (soproClip) drawWave(form);
        });
    }

    // One pass, on second-wide blocks, keeping the window with the most energy.
    function loudestWindow(buffer, seconds) {
        const data = buffer.getChannelData(0);
        const rate = buffer.sampleRate;
        const blocks = Math.max(1, Math.floor(buffer.duration));
        if (buffer.duration <= seconds) return 0;
        const energy = new Float64Array(blocks);
        for (let block = 0; block < blocks; block += 1) {
            let sum = 0;
            const from = block * rate;
            const to = Math.min(data.length, from + rate);
            for (let index = from; index < to; index += 1) sum += data[index] * data[index];
            energy[block] = sum;
        }
        const width = Math.max(1, Math.round(seconds));
        let best = 0;
        let bestSum = -1;
        for (let block = 0; block + width <= blocks; block += 1) {
            let sum = 0;
            for (let inner = 0; inner < width; inner += 1) sum += energy[block + inner];
            if (sum > bestSum) { bestSum = sum; best = block; }
        }
        return Math.min(best, Math.max(0, buffer.duration - seconds));
    }

    function soproForm(holder) {
        return holder ? holder.querySelector("[data-mc-voice-sopro-form]") : null;
    }

    // Starter voices, one request each.
    //
    // A loop rather than one long call: each voice is a full Sopro preparation,
    // four of them in a single request is a minute of silence and a browser that
    // may give up in the middle of it, and doing them one at a time lets the row
    // say which voice it is on.
    function wireSoproStarter(holder) {
        const row = holder.querySelector("[data-mc-voice-sopro-starter]");
        if (!row || row.dataset.mcVoiceWired === "1") return;
        row.dataset.mcVoiceWired = "1";
        const button = row.querySelector("[data-mc-voice-starter-make]");
        if (!button) return;
        const say = function (text) {
            const status = row.querySelector("[data-mc-voice-starter-status]");
            if (status) status.textContent = text || "";
        };
        button.addEventListener("click", function (event) {
            if (event.preventDefault) event.preventDefault();
            button.disabled = true;
            say("Making the first starter voice\u2026");
            const step = function (made) {
                return post(ROUTES.soproStarter, {}, holder).then(function (payload) {
                    if (!payload || !payload.ok) {
                        say((payload && payload.error) || "That could not be done.");
                        return false;
                    }
                    applyVoices(holder, payload);
                    const count = made + (payload.created ? 1 : 0);
                    if (payload.remaining > 0) {
                        say("Made " + (payload.created || "a voice") + ". "
                            + payload.remaining + " to go\u2026");
                        return step(count);
                    }
                    say(count
                        ? "Done \u2014 " + count + " starter voice"
                          + (count === 1 ? "" : "s") + " added."
                        : "The starter voices are already here.");
                    return true;
                });
            };
            step(0).catch(function () {
                say("That could not be done.");
            }).then(function () {
                button.disabled = false;
            });
        });
    }

    function wireSoproClone(holder) {
        const form = soproForm(holder);
        if (!form || form.dataset.mcVoiceWired === "1") return;
        form.dataset.mcVoiceWired = "1";

        const record = form.querySelector("[data-mc-voice-sopro-record]");
        if (record) {
            record.addEventListener("click", function (event) {
                if (event.preventDefault) event.preventDefault();
                if (soproRecorder) {
                    stopSoproRecording(form, record);
                    return;
                }
                startSoproRecording(form, record);
            });
        }
        const file = form.querySelector("[data-mc-voice-sopro-file]");
        if (file) {
            file.addEventListener("change", function () {
                const chosen = file.files && file.files[0];
                if (!chosen) return;
                loadSoproClip(form, chosen, chosen.name || "that file");
            });
        }
        const create = form.querySelector("[data-mc-voice-sopro-create]");
        if (create) {
            create.addEventListener("click", function (event) {
                if (event.preventDefault) event.preventDefault();
                createSoproVoice(holder, form, create);
            });
        }
        wireTrim(form, holder);
        wirePreview(holder, form);
    }

    // Reuses the capture primitives dictation uses -- `startCapture`,
    // `releaseCapture`, `resample` and `encodeWav` -- and deliberately not the
    // dictation *flow*: this recording is never sent to Whisper, never reaches
    // the composer, and is not bounded by the dictation gesture's timings.
    //
    // It is also not downsampled to 16 kHz on the way out. Dictation resamples
    // because Whisper wants 16 kHz and everything above 8 kHz is wasted upload;
    // a voice reference is the one recording where that band is the point, so
    // this sends the microphone's own rate and lets the server produce Sopro's
    // canonical 24 kHz from it.

    const SOPRO_MAX_SECONDS = 20;

    function startSoproRecording(form, button) {
        const note = form.querySelector("[data-mc-voice-sopro-recording]");
        if (capture) {
            if (note) note.textContent = "The microphone is already in use for dictation.";
            return;
        }
        unlock();
        startCapture(null, true).then(function (state) {
            soproRecorder = state;
            soproClip = null;
            const trim = form.querySelector("[data-mc-voice-trim]");
            if (trim) trim.hidden = true;
            button.textContent = "Stop recording";
            if (note) note.textContent = "Recording…";
            // Stopped for them at the top of the supported window. A microphone
            // left open by somebody who walked away is the failure this bound
            // exists for, and twenty seconds is the longest reference Sopro
            // uses anyway.
            window.setTimeout(function () {
                if (soproRecorder === state) stopSoproRecording(form, button);
            }, (SOPRO_MAX_SECONDS + 1) * 1000);
        }).catch(function (error) {
            if (note) note.textContent = captureFailure(error);
        });
    }

    function stopSoproRecording(form, button) {
        const note = form.querySelector("[data-mc-voice-sopro-recording]");
        const state = soproRecorder;
        soproRecorder = null;
        button.textContent = "Record here";
        if (!state) return;
        let samples;
        try {
            samples = resample(state.chunks || [], 0, 0);
        } catch (error) {
            samples = null;
        }
        releaseCapture(state);
        if (!samples || !samples.length) {
            if (note) note.textContent = "Nothing was recorded.";
            return;
        }
        const seconds = samples.length / (state.rate || 1);
        if (note) note.textContent = "Recorded " + seconds.toFixed(1) + " s";
        // Into the trimmer, not into a variable of its own. A take with three
        // seconds of throat-clearing at the front is the ordinary case, and
        // before this the only way to fix one was to record it again.
        loadSoproClip(form, new Blob([encodeWav(samples, state.rate)], {type: "audio/wav"}),
                      "your recording");
    }

    function createSoproVoice(holder, form, button) {
        const name = form.querySelector("[data-mc-voice-sopro-name]");
        const language = form.querySelector("[data-mc-voice-sopro-language]");
        const say = function (text) {
            const status = holder.querySelector("[data-mc-voice-sopro-clone-status]");
            if (status) status.textContent = text || "";
        };
        if (!name || !(name.value || "").trim()) {
            say("Give the voice a name.");
            return;
        }
        if (!soproClip) {
            say("Choose an audio file or record something first.");
            return;
        }
        const chosen = clipDuration();
        if (chosen < CLIP_MIN_SECONDS || chosen > CLIP_MAX_SECONDS) {
            // Said here as well as under the waveform, because this is the
            // button somebody pressed and the answer belongs next to it.
            say("Sopro clones from " + CLIP_MIN_SECONDS + " to " + CLIP_MAX_SECONDS
                + " seconds. " + chosen.toFixed(1) + " s is selected.");
            return;
        }
        const wav = clipToWav();
        if (!wav) {
            say("That selection is empty.");
            return;
        }
        button.disabled = true;
        say("Preparing the voice…");
        const body = new FormData();
        body.append("name", name.value);
        body.append("language", language ? language.value : "");
        body.append("reference", new Blob([wav], {type: "audio/wav"}), "reference.wav");
        fetch(url(ROUTES.soproClone), {
            method: "POST",
            credentials: "same-origin",
            headers: headers({}, holder),
            body: body,
        }).then(refused).then(function (response) {
            return response.json();
        }).then(function (payload) {
            button.disabled = false;
            if (!payload || !payload.ok) {
                say((payload && payload.error) || "That voice could not be created.");
                return;
            }
            // Nothing is saved and nothing is cleared. The name, the file and
            // the selection all stay exactly where they are, because the next
            // thing that happens may well be "no, try a different fifteen
            // seconds" -- and a form that emptied itself on preview would make
            // trying again mean starting again.
            soproPreview = {token: payload.token || "",
                            name: payload.name || name.value,
                            audio: payload.audio || ""};
            say("Ready. Listen, then Save voice or Discard.");
            showPreview(holder, form);
            playPreview();
        }).catch(function () {
            button.disabled = false;
            say("That voice could not be created.");
        });
    }

    // -- the voice that exists but is not saved ------------------------------ //

    // Section 23's seam, which was always there and was not being used:
    // everything up to the registry write leaves nothing anybody can see, so a
    // prepared voice with no registry entry is not half-saved -- it is a voice
    // that does not exist yet and costs one directory to stop existing.
    //
    // Create used to write the registry entry, make the voice the default if it
    // was the first, and *then* play the audition. Hearing it was a receipt.
    let soproPreview = null;

    function previewRow(holder) {
        return holder.querySelector("[data-mc-voice-sopro-preview]");
    }

    function showPreview(holder, form) {
        const row = previewRow(holder);
        if (!row) return;
        row.hidden = !soproPreview;
        const note = row.querySelector("[data-mc-voice-sopro-preview-note]");
        if (note) {
            note.textContent = soproPreview
                ? ("\u201c" + soproPreview.name + "\u201d is ready to listen to. "
                   + "It is not saved yet.")
                : "";
        }
        if (!soproPreview && form) {
            const trim = form.querySelector("[data-mc-voice-trim]");
            if (trim) trim.hidden = true;
        }
    }

    function playPreview() {
        if (!soproPreview || !soproPreview.audio) return;
        unlock();
        play(base64ToBuffer(soproPreview.audio)).catch(function () { /* silent */ });
    }

    function wirePreview(holder, form) {
        const row = previewRow(holder);
        if (!row || row.dataset.mcVoiceWired === "1") return;
        row.dataset.mcVoiceWired = "1";

        const say = function (text) {
            const status = holder.querySelector("[data-mc-voice-sopro-clone-status]");
            if (status) status.textContent = text || "";
        };
        const again = row.querySelector("[data-mc-voice-sopro-preview-play]");
        const save = row.querySelector("[data-mc-voice-sopro-preview-save]");
        const discard = row.querySelector("[data-mc-voice-sopro-preview-discard]");

        if (again) {
            again.addEventListener("click", function (event) {
                if (event.preventDefault) event.preventDefault();
                playPreview();
            });
        }
        if (save) {
            save.addEventListener("click", function (event) {
                if (event.preventDefault) event.preventDefault();
                if (!soproPreview || save.disabled) return;
                save.disabled = true;
                post(ROUTES.soproSave, {token: soproPreview.token}, holder)
                    .then(function (payload) {
                        save.disabled = false;
                        if (!payload || !payload.ok) {
                            say((payload && payload.error)
                                || "That voice could not be saved.");
                            return;
                        }
                        say("Saved " + soproPreview.name + ".");
                        soproPreview = null;
                        showPreview(holder, form);
                        // Only now: the list has only now changed.
                        applyVoices(holder, payload);
                        const name = form
                            && form.querySelector("[data-mc-voice-sopro-name]");
                        if (name) name.value = "";
                        soproClip = null;
                    }).catch(function () {
                        save.disabled = false;
                        say("That voice could not be saved.");
                    });
            });
        }
        if (discard) {
            discard.addEventListener("click", function (event) {
                if (event.preventDefault) event.preventDefault();
                if (!soproPreview) return;
                const token = soproPreview.token;
                const gone = soproPreview.name;
                // Locally first. The server call removes the directory, and a
                // failed one must not leave a Save button for a voice the user
                // has said no to.
                soproPreview = null;
                showPreview(holder, form);
                say("Discarded " + gone + ".");
                post(ROUTES.soproDiscard, {token: token}, holder)
                    .catch(function () { /* the shutdown path clears it too */ });
            });
        }
    }

    function base64ToBuffer(text) {
        const binary = window.atob(String(text || ""));
        const bytes = new Uint8Array(binary.length);
        for (let index = 0; index < binary.length; index += 1) {
            bytes[index] = binary.charCodeAt(index);
        }
        return bytes.buffer;
    }

    // -- the Voice Lab --------------------------------------------------------- //
    //
    // Experimental, and built so it cannot be anything else. There is no save,
    // no apply and no promote here, and there is no route it could call if
    // there were: the session lives in the server's memory, it is dropped when
    // the engine changes or the page is reloaded, and every audition returns a
    // WAV rather than changing anything.

    let labToken = "";
    let labPending = 0;

    function labPanel(holder) {
        return holder ? holder.querySelector("[data-mc-voice-lab]") : null;
    }

    function wireLab(holder) {
        const panel = labPanel(holder);
        if (!panel || panel.dataset.mcVoiceWired === "1") return;
        panel.dataset.mcVoiceWired = "1";

        // Opened on first expand rather than on page load: the Lab needs the
        // worker to read a voice's saved style controls, and starting Sopro
        // because somebody scrolled past a closed `<details>` would be a
        // settings page that quietly allocates a Torch runtime.
        panel.addEventListener("toggle", function () {
            if (panel.open && !labToken) openLab(holder, panel, "");
        });
        panel.addEventListener("input", function (event) {
            const slider = event.target.closest("[data-mc-voice-lab-input]");
            if (slider) {
                showLabValue(panel, slider.getAttribute("data-mc-voice-lab-input"),
                             slider.value);
                return;
            }
            const weight = event.target.closest("[data-mc-voice-lab-blend-weight]");
            if (weight) {
                const output = panel.querySelector("[data-mc-voice-lab-blend-value]");
                if (output) output.textContent = Number(weight.value).toFixed(2);
            }
        });
        panel.addEventListener("change", function (event) {
            if (event.target.closest("[data-mc-voice-lab-voice]")) {
                openLab(holder, panel, panel.querySelector(
                    "[data-mc-voice-lab-voice]").value);
                return;
            }
            sendLab(holder, panel);
        });
        panel.addEventListener("click", function (event) {
            const play_ = event.target.closest("[data-mc-voice-lab-play]");
            if (play_) {
                if (event.preventDefault) event.preventDefault();
                playLab(holder, panel, play_.getAttribute("data-mc-voice-lab-play"), play_);
                return;
            }
            if (event.target.closest("[data-mc-voice-lab-reset]")) {
                if (event.preventDefault) event.preventDefault();
                post(ROUTES.labReset, {token: labToken}, holder).then(function (payload) {
                    paintLab(panel, payload && payload.session);
                });
            }
        });
    }

    function openLab(holder, panel, voiceId) {
        post(ROUTES.lab, {token: "", voice: voiceId || ""}, holder).then(function (payload) {
            if (!payload || !payload.ok) {
                setText(panel, "[data-mc-voice-lab-metrics]",
                        (payload && payload.error) || "The Voice Lab could not be opened.");
                return;
            }
            labToken = payload.session.token;
            fillLabVoices(panel, payload.panel, payload.session);
            paintLab(panel, payload.session);
        });
    }

    function fillLabVoices(panel, info, session) {
        [["[data-mc-voice-lab-voice]", false], ["[data-mc-voice-lab-blend-voice]", true]]
            .forEach(function (pair) {
                const select = panel.querySelector(pair[0]);
                if (!select) return;
                const was = select.value;
                select.textContent = "";
                if (pair[1]) {
                    const none = document.createElement("option");
                    none.value = "";
                    none.textContent = "No blend";
                    select.appendChild(none);
                }
                (info.voices || []).forEach(function (entry) {
                    const option = document.createElement("option");
                    option.value = entry.id;
                    option.textContent = entry.label;
                    select.appendChild(option);
                });
                select.value = pair[1] ? was : (session.voice_id || was);
            });
    }

    function readLab(panel) {
        const deltas = [];
        Array.prototype.forEach.call(
            panel.querySelectorAll("[data-mc-voice-lab-input]"), function (input) {
                deltas[Number(input.getAttribute("data-mc-voice-lab-input"))] =
                    Number(input.value);
            });
        const blendVoice = panel.querySelector("[data-mc-voice-lab-blend-voice]");
        const weight = panel.querySelector("[data-mc-voice-lab-blend-weight]");
        const blend = {voice_id: blendVoice ? blendVoice.value : "",
                       weight: weight ? Number(weight.value) : 0};
        Array.prototype.forEach.call(
            panel.querySelectorAll("[data-mc-voice-lab-blend-field]"), function (box) {
                blend[box.getAttribute("data-mc-voice-lab-blend-field")] = box.checked;
            });
        const fixed = panel.querySelector("[data-mc-voice-lab-fixed-seed]");
        const seed = panel.querySelector("[data-mc-voice-lab-seed]");
        const text = panel.querySelector("[data-mc-voice-lab-text]");
        const voice = panel.querySelector("[data-mc-voice-lab-voice]");
        return {
            voice_id: voice ? voice.value : "",
            deltas: deltas,
            blend: blend,
            seed: fixed && fixed.checked && seed ? Number(seed.value) : null,
            text: text ? text.value : "",
        };
    }

    function sendLab(holder, panel) {
        if (!labToken) return Promise.resolve(null);
        labPending += 1;
        return post(ROUTES.labUpdate, {token: labToken, values: readLab(panel)}, holder)
            .then(function (payload) {
                labPending -= 1;
                if (payload && payload.session) paintLab(panel, payload.session, true);
                return payload;
            }).catch(function () { labPending -= 1; return null; });
    }

    function paintLab(panel, session, keepInputs) {
        if (!panel || !session) return;
        if (!keepInputs) {
            (session.deltas || []).forEach(function (value, index) {
                const input = panel.querySelector(
                    '[data-mc-voice-lab-input="' + index + '"]');
                if (input && document.activeElement !== input) input.value = value;
                showLabValue(panel, index, value);
            });
            const text = panel.querySelector("[data-mc-voice-lab-text]");
            if (text && document.activeElement !== text) text.value = session.text || "";
        }
        const metrics = panel.querySelector("[data-mc-voice-lab-metrics]");
        const last = session.last || {};
        if (metrics) {
            metrics.textContent = last.side
                ? (last.side === "a" ? "A" : "B") + " — first audio "
                    + last.first_audio_ms + " ms, total " + last.elapsed_ms + " ms, "
                    + (last.audio_ms / 1000).toFixed(1) + " s of speech, RTF "
                    + last.rtf + ", " + last.chunks + " chunks"
                : "";
        }
    }

    function showLabValue(panel, index, value) {
        const output = panel.querySelector('[data-mc-voice-lab-value="' + index + '"]');
        if (output) output.textContent = Number(value).toFixed(2);
    }

    function playLab(holder, panel, side, button) {
        if (!labToken) return;
        button.disabled = true;
        sendLab(holder, panel).then(function () {
            return post(ROUTES.labPlay, {token: labToken, side: side}, holder);
        }).then(function (payload) {
            button.disabled = false;
            if (!payload || !payload.ok) {
                setText(panel, "[data-mc-voice-lab-metrics]",
                        (payload && payload.error) || "That audition could not be played.");
                return;
            }
            paintLab(panel, payload.session, true);
            if (payload.audio) {
                unlock();
                play(base64ToBuffer(payload.audio)).catch(function () { /* silent */ });
            }
        }).catch(function () {
            button.disabled = false;
            setText(panel, "[data-mc-voice-lab-metrics]",
                    "That audition could not be played.");
        });
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
        if (checks) keepingPlace(checks, function () {
            checks.textContent = "";
            (payload.checks || []).forEach(function (item) {
                const line = document.createElement("div");
                line.className = "mc-voice-check" + (item.ok ? " mc-voice-check-ok" : "");
                line.textContent = (item.ok ? "\u2713 " : "\u2717 ") + item.item
                    + (item.detail ? " — " + item.detail : "");
                checks.appendChild(line);
            });
        });
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
        attempt("watch for a reply to speak", watchSpeechFields);
        attempt("wire the voice gestures", wireGestures);
        attempt("wire the Voice Chat settings", wireSettings);
        attempt("wire the voice list", wireVoices);
        attempt("wire the character's voice", wirePicker);
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
