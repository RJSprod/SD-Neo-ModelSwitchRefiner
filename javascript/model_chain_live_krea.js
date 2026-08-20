// Model Chain -- Krea Live, browser side.
//
// This file schedules and it intercepts. It does not decide anything about
// prompts, models, caching or memory, and it never talks to a model: every
// question of the form "does this state need a new expansion?" is answered in
// Python by mc_live_krea, which is the only place that can answer it correctly
// and the only place a test can hold to it. What is genuinely a browser's job,
// and cannot be done anywhere else, is the four things below.
//
//   * Idle debounce. "Five seconds after the last keystroke" is a question
//     about a textarea that only the page holding the textarea can answer. The
//     server never sees the keystrokes and must not: a round trip per
//     character is the design this feature exists to avoid.
//
//   * Interception. Live changes what Generate means, and the change has to
//     happen before the native click reaches Forge -- an LLM run waits for the
//     host to stop generating, so an expansion asked for after the image job
//     had started would be waiting on the job that is waiting on it. The gate
//     is a capture-phase listener and a one-shot bypass flag, which is the
//     smallest thing that can let exactly one native click through afterwards.
//
//   * Repetition. Reroll draws a new image seed from the already-expanded
//     prompt. It is a timer that clicks a button once the last generation has
//     finished, which is a browser's job by definition.
//
//   * The armed indicator. The one visible sign that the positive prompt has
//     changed meaning, painted on the button whose meaning changed.
//
// Everything is found by this extension's own element ids and by the host's
// documented ones (#txt2img_prompt, #txt2img_generate). No selector below
// depends on a class Gradio generated, so a theme that replaces Gradio's DOM
// changes how the strip looks and cannot stop it working. If an id is missing,
// the feature it drives is skipped and the rest carry on.
//
// The failure posture is the same as the rest of this extension's JavaScript:
// if anything here throws, txt2img must still generate images. Every entry
// point is wrapped, and every wrapper's fallback is "behave as though Krea Live
// were switched off".

(function () {
    "use strict";

    const IDS = {
        toggle: "mc-krea-live-toggle",
        creativity: "mc-krea-live-creativity",
        delay: "mc-krea-live-delay",
        reroll: "mc-krea-live-reroll",
        run: "mc-krea-live-run",
        revise: "mc-krea-live-revise",
        halt: "mc-krea-live-halt",
        token: "mc-krea-live-token",
        prompt: "txt2img_prompt",
        generate: "txt2img_generate",
        results: "txt2img_results",
        status: "mc-krea-live-status",
    };

    // How often the token box is read while an expansion is in flight. It is a
    // property read on one element, so the cost is nil; the number is about how
    // quickly the image starts after the prompt is ready, and a tenth of a
    // second is under what anybody notices.
    const TOKEN_POLL_MS = 100;

    // How long to keep reading it before giving up. Generous on purpose: a cold
    // llama-server reads its weights off disk before it answers the first
    // request, and twenty seconds of that is normal rather than a fault.
    const TOKEN_TIMEOUT_MS = 15 * 60 * 1000;

    // How often the page is asked whether a generation is still running, while
    // waiting to reroll.
    const IDLE_POLL_MS = 400;

    // A reroll waits this long after the gallery settles before clicking again,
    // so a run of rerolls can be stopped by a person rather than only by the
    // Stop button winning a race.
    const REROLL_GAP_MS = 350;

    const ARMED_CLASS = "mc-krea-live-armed";

    const state = {
        revision: 0,
        timer: null,
        // "idle" | "expanding" | "generating". One cycle at a time, always:
        // there is one active cycle and one newest pending source revision, and
        // deliberately nothing in between -- a queue here would be a queue of
        // every half-typed sentence, each one an LLM call.
        phase: "idle",
        // Set when the native click is ours rather than the user's. Consumed by
        // the gate on the very next click, which is what stops the programmatic
        // click we make after an expansion from being intercepted again.
        bypass: false,
        // A debounce that fired while an image was being generated. The newest
        // one survives; there is deliberately no queue.
        pending: false,
        rerolling: false,
        wired: false,
        menuWired: false,
        // Whether the server has already been told that the run in flight is
        // about text that has moved on. One telling per cycle is enough, and
        // one per keystroke would be the round trip per character this whole
        // design exists to avoid.
        revised: false,
    };

    // -- finding things ---------------------------------------------------- //

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

    function field(id, selector) {
        const holder = byId(id);
        return holder ? holder.querySelector(selector) : null;
    }

    function promptBox() {
        return field(IDS.prompt, "textarea");
    }

    function toggleBox() {
        return field(IDS.toggle, "input[type=checkbox]");
    }

    function rerollBox() {
        return field(IDS.reroll, "input[type=checkbox]");
    }

    function tokenBox() {
        return field(IDS.token, "textarea, input");
    }

    function isLive() {
        const box = toggleBox();
        return !!(box && box.checked);
    }

    function wantsReroll() {
        const box = rerollBox();
        return !!(box && box.checked);
    }

    // The idle delay, in milliseconds, read from the strip rather than cached:
    // the control is a number box the user may change between two keystrokes,
    // and the value that matters is the one showing when the timer is armed.
    function delayMs() {
        const box = field(IDS.delay, "input");
        const seconds = box ? parseFloat(box.value) : NaN;
        if (!isFinite(seconds)) return 5000;
        return Math.max(500, Math.min(30000, seconds * 1000));
    }

    function sourceText() {
        const box = promptBox();
        return box ? String(box.value || "").trim() : "";
    }

    // -- the armed indicator ------------------------------------------------ //

    function paintIndicator() {
        const button = clickable(IDS.generate);
        if (!button) return;
        button.classList.toggle(ARMED_CLASS, isLive());
    }

    // -- talking to the server ---------------------------------------------- //

    function press(id) {
        const button = clickable(id);
        if (button) button.click();
        return !!button;
    }

    // Wait for the hidden token box to say something new. Resolves with true
    // when an expansion is ready and false for every other outcome, including
    // Live being switched off mid-flight -- and "false" always means "do not
    // start a native generation", which is the only decision the caller makes
    // from it.
    function awaitToken(previous) {
        return new Promise(function (resolve) {
            const started = Date.now();
            const poll = window.setInterval(function () {
                let value = "";
                try {
                    const box = tokenBox();
                    value = box ? String(box.value || "") : "";
                } catch (error) {
                    value = "";
                }

                if (value && value !== previous) {
                    window.clearInterval(poll);
                    resolve(value.indexOf("ready:") === 0);
                    return;
                }
                if (!isLive() || Date.now() - started > TOKEN_TIMEOUT_MS) {
                    window.clearInterval(poll);
                    resolve(false);
                }
            }, TOKEN_POLL_MS);
        });
    }

    // -- is the host generating? -------------------------------------------- //

    // The host's own progress element, which javascript/progressbar.js creates
    // when a job starts and removes when it ends. Read rather than hooked: this
    // file must not wrap the host's progress plumbing, which another file in
    // this extension already decorates, and one reader of a DOM node cannot
    // interfere with anybody.
    function hostGenerating() {
        // Scoped to the txt2img results column where that exists, and to the
        // whole app where it does not. The narrow scope is right on an ordinary
        // page; the fallback is what stops a renamed or restructured results
        // column turning "is it still generating?" into a permanent no, which
        // would have reroll clicking Generate on top of a running job.
        const scope = byId(IDS.results) || root();
        try {
            return !!(scope && scope.querySelector && scope.querySelector(".progressDiv"));
        } catch (error) {
            return false;
        }
    }

    // Resolve once the current generation has finished. The grace period is
    // what makes this correct at the start: the progress element does not exist
    // for the first moment after a click, so a check made immediately would
    // report "finished" for a job that had not started yet.
    function whenIdle() {
        return new Promise(function (resolve) {
            let sawBusy = false;
            let waited = 0;
            const poll = window.setInterval(function () {
                waited += IDLE_POLL_MS;
                if (hostGenerating()) {
                    sawBusy = true;
                    return;
                }
                if (sawBusy || waited >= 3000) {
                    window.clearInterval(poll);
                    resolve(sawBusy);
                }
            }, IDLE_POLL_MS);
        });
    }

    // -- the cycle ----------------------------------------------------------- //

    // One expansion, then one native generation. The only path to a Live image.
    //
    // "One at a time" is enforced here rather than trusted: a second call while
    // a cycle is running marks the newest source as pending and returns, so the
    // latest input wins and nothing in between it and the running cycle is ever
    // generated.
    function runCycle(options) {
        const settings = options || {};
        if (state.phase !== "idle") {
            state.pending = true;
            return Promise.resolve(false);
        }
        if (!sourceText()) return Promise.resolve(false);
        if (settings.repeat) state.rerolling = true;

        const previous = (function () {
            const box = tokenBox();
            return box ? String(box.value || "") : "";
        })();

        state.phase = "expanding";
        state.revised = false;
        const startedAt = state.revision;

        if (!press(IDS.run)) {
            state.phase = "idle";
            return Promise.resolve(false);
        }

        return awaitToken(previous).then(function (ready) {
            state.phase = "idle";
            if (!ready) return false;
            // The text moved on while the model was writing. The server has
            // already refused to cache or arm that answer; this is the browser
            // declining to draw an image from it.
            if (state.revision !== startedAt) {
                state.pending = true;
                return false;
            }
            state.phase = "generating";
            state.bypass = true;
            const button = clickable(IDS.generate);
            if (button) {
                button.click();
            } else {
                state.bypass = false;
                state.phase = "idle";
                return false;
            }
            return whenIdle().then(function () {
                state.phase = "idle";
                afterGeneration(settings);
                return true;
            });
        }).catch(function (error) {
            console.error("Model Chain: the Krea Live cycle failed", error);
            state.phase = "idle";
            return false;
        });
    }

    // What happens once an image has been delivered: the newest pending source
    // first, then a reroll, and nothing at all if neither applies.
    //
    // Pending beats reroll deliberately. A user who typed while the picture was
    // being drawn has said what they want next more recently than the reroll
    // checkbox did.
    function afterGeneration(settings) {
        if (state.pending) {
            state.pending = false;
            if (isLive()) schedule();
            return;
        }
        if (!settings.repeat || !state.rerolling || !isLive() || !wantsReroll()) {
            state.rerolling = false;
            return;
        }
        window.setTimeout(function () {
            if (state.rerolling && isLive() && wantsReroll()) {
                runCycle({repeat: true});
            }
        }, REROLL_GAP_MS);
    }

    // -- the debounce -------------------------------------------------------- //

    function schedule() {
        window.clearTimeout(state.timer);
        if (!isLive()) return;
        const armedAt = state.revision;
        state.timer = window.setTimeout(function () {
            // The revision is re-checked rather than assumed: a timer that has
            // already begun running cannot be cleared, so a keystroke landing
            // in that gap has to be caught here.
            if (state.revision !== armedAt) return;
            if (!isLive() || !sourceText()) return;
            if (state.phase !== "idle" || hostGenerating()) {
                state.pending = true;
                return;
            }
            runCycle({repeat: wantsReroll()});
        }, delayMs());
    }

    function onSourceEdit() {
        state.revision += 1;
        window.clearTimeout(state.timer);
        if (state.phase === "expanding" && !state.revised) {
            // Only now, and only once per cycle: the server drops the cached
            // expansion and cancels the writer cooperatively. There is nothing
            // a second telling could add -- the run is already cancelled and its
            // answer already condemned.
            state.revised = true;
            press(IDS.revise);
        }
        if (!isLive()) return;
        schedule();
    }

    // -- the Generate gate ---------------------------------------------------- //

    // Capture phase, on the document, so this runs before any handler the host
    // or another extension attached to the button itself.
    function onGenerateClick(event) {
        try {
            const button = clickable(IDS.generate);
            if (!button || !event.target) return;
            if (event.target !== button && !button.contains(event.target)) return;
            if (!isLive()) return;

            if (state.bypass) {
                // Ours, and exactly one of them. The flag is cleared before the
                // click proceeds, so a second programmatic click cannot ride
                // through on the same permission.
                state.bypass = false;
                return;
            }

            // A click nobody made. Forge's own "Generate forever" drives the
            // button from a timer, and two repeat schedulers racing to press
            // the same button is a failure worth refusing rather than trying to
            // referee. Krea Live's own Reroll knows how to reuse the cached
            // expansion; native repetition does not.
            if (event.isTrusted === false) {
                stopNativeForever();
                event.preventDefault();
                event.stopImmediatePropagation();
                return;
            }

            event.preventDefault();
            event.stopImmediatePropagation();
            state.pending = false;
            runCycle({repeat: wantsReroll()});
        } catch (error) {
            // A gate that throws must not be a Generate button that does
            // nothing: the click is left alone and txt2img behaves natively.
            console.error("Model Chain: the Krea Live gate failed, generating natively", error);
        }
    }

    // Stop the host's own repeat loop, and say why.
    //
    // Forge's contextMenus.js keeps the loop's timer on window as
    // generateOnRepeatInterval, and clearing it is exactly what its own
    // "Cancel generate forever" entry does. Reading that one documented global
    // is a smaller dependency than scraping its menu -- which is not in the DOM
    // while the loop is running anyway, because the menu closes when the entry
    // is chosen.
    function stopNativeForever() {
        try {
            if (window.generateOnRepeatInterval) {
                window.clearInterval(window.generateOnRepeatInterval);
            }
        } catch (error) {
            console.error("Model Chain: could not stop Generate forever", error);
        }
        const refusal = "Generate forever is refused while Krea Live is on. Use Reroll in "
            + "the Krea Live strip: it reuses the expanded prompt and only varies the "
            + "image seed.";
        sayLocally(refusal);
        console.warn("Model Chain: " + refusal);
    }

    // Put a line in the Live status element without a round trip.
    //
    // The server owns that element and will overwrite this on its next update,
    // which is correct: this is for the two things the server never hears about
    // -- a refused native repeat, and a click the browser declined to pass on.
    function sayLocally(text) {
        try {
            const holder = byId(IDS.status);
            if (!holder) return;
            const line = holder.querySelector("div") || holder;
            line.textContent = text;
        } catch (error) {
            // A status line is not worth an exception.
        }
    }

    // -- the toggle ------------------------------------------------------------ //

    function setLive(on) {
        const box = toggleBox();
        if (!box || box.checked === !!on) {
            paintIndicator();
            return;
        }
        box.checked = !!on;
        // Both events: Gradio's checkbox listens for "change", and the host's
        // own updateInput helper dispatches "input". Sending both means this
        // works whichever the running version wants, and a duplicate is a
        // no-op.
        box.dispatchEvent(new Event("input", {bubbles: true}));
        box.dispatchEvent(new Event("change", {bubbles: true}));
        paintIndicator();
    }

    function startLive() {
        setLive(true);
        state.pending = false;
        schedule();
    }

    function stopLive() {
        state.rerolling = false;
        state.pending = false;
        window.clearTimeout(state.timer);
        state.revision += 1;
        press(IDS.halt);
        setLive(false);
    }

    // -- the context menu -------------------------------------------------------- //

    // Forge Neo already puts a right-click menu on the Generate button and
    // exposes appendContextMenuOption for extensions to add to it. Extending
    // that menu rather than drawing a floating one of our own is what makes
    // these entries feel native, and is why "Generate forever" and "Cancel
    // generate forever" are still there afterwards: nothing here replaces the
    // menu, it only appends.
    function wireContextMenu() {
        if (typeof window.appendContextMenuOption !== "function") return;
        if (state.menuWired) return;
        state.menuWired = true;

        const target = "#" + IDS.generate;
        try {
            window.appendContextMenuOption(target, "Generate with Krea Live once", function () {
                // Works with persistent Live off, which is the point of it:
                // one expansion, one image, and no idle watcher armed
                // afterwards. If the cached expansion still matches the current
                // source, Creativity and model, no model request is made at all.
                state.rerolling = false;
                state.pending = false;
                const wasLive = isLive();
                if (!wasLive) setLive(true);
                runCycle({repeat: false}).then(function () {
                    if (!wasLive) stopLive();
                });
            });

            window.appendContextMenuOption(target, "Start Krea Live", function () {
                startLive();
            });

            window.appendContextMenuOption(target, "Stop Krea Live", function () {
                stopLive();
            });
        } catch (error) {
            console.error("Model Chain: could not add the Krea Live context-menu entries", error);
        }
    }

    // -- wiring -------------------------------------------------------------------- //

    function wire() {
        try {
            wireContextMenu();
            paintIndicator();

            const prompt = promptBox();
            if (prompt && !prompt.dataset.mcKreaLive) {
                prompt.dataset.mcKreaLive = "1";
                prompt.addEventListener("input", onSourceEdit);
            }

            const toggle = toggleBox();
            if (toggle && !toggle.dataset.mcKreaLive) {
                toggle.dataset.mcKreaLive = "1";
                toggle.addEventListener("change", function () {
                    paintIndicator();
                    if (isLive()) {
                        schedule();
                    } else {
                        state.rerolling = false;
                        state.pending = false;
                        window.clearTimeout(state.timer);
                    }
                });
            }

            const reroll = rerollBox();
            if (reroll && !reroll.dataset.mcKreaLive) {
                reroll.dataset.mcKreaLive = "1";
                reroll.addEventListener("change", function () {
                    state.rerolling = wantsReroll();
                    if (state.rerolling && isLive() && state.phase === "idle"
                        && !hostGenerating()) {
                        runCycle({repeat: true});
                    }
                });
            }

            if (!state.wired) {
                state.wired = true;
                document.addEventListener("click", onGenerateClick, true);
            }
        } catch (error) {
            console.error("Model Chain: Krea Live wiring failed", error);
        }
    }

    // Re-applied rather than installed once: Gradio rebuilds parts of the tab
    // on some updates. Every wiring above is idempotent through the dataset
    // flags, so re-running it costs a handful of queries and nothing else.
    if (typeof onUiLoaded === "function") {
        onUiLoaded(wire);
    } else if (document.readyState !== "loading") {
        wire();
    } else {
        document.addEventListener("DOMContentLoaded", wire);
    }

    if (typeof onAfterUiUpdate === "function") {
        onAfterUiUpdate(wire);
    }

    // Exposed for the tests, which drive this file under node against a fake
    // page. Nothing in the extension reads it.
    window.modelChainKreaLive = {
        state: state,
        schedule: schedule,
        onSourceEdit: onSourceEdit,
        onGenerateClick: onGenerateClick,
        runCycle: runCycle,
        startLive: startLive,
        stopLive: stopLive,
        wire: wire,
    };
})();
