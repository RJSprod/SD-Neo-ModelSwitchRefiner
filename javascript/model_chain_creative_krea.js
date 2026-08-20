// Model Chain -- Krea Creative Mode, browser side.
//
// One job: when Creative Mode is on, a press of Generate has to run the
// creative roll first and the native image job second. Everything else about
// the feature is Python.
//
// Why it needs a browser at all
// -----------------------------
// The roll ends in a Krea writer call, and an LLM run waits for the host to
// stop generating -- so a roll asked for from inside a running image job would
// be waiting on the job that is waiting on it. The order has to be enforced
// before the native submission, and the only place that can happen is in front
// of the click.
//
// So: intercept the click, press a hidden button that does the roll, wait for
// the server to say it is ready, then click Generate again with a one-shot flag
// that lets exactly that click through.
//
// What is deliberately absent
// ---------------------------
// No idle timer. No observer on the prompt box. No repeat loop. No status
// machine. This file has one entry point that is not wiring, and it fires when
// somebody presses a button. That is the whole difference between this and the
// Live controller it replaces, and it is most of why this file is a third of
// the length.
//
// Failure posture, as everywhere else in this extension's JavaScript: if
// anything here throws, txt2img must still generate images. Every entry point
// is wrapped and every fallback is "behave as though Creative Mode were off".

(function () {
    "use strict";

    const IDS = {
        toggle: "mc-krea-creative-toggle",
        run: "mc-krea-creative-run",
        token: "mc-krea-creative-token",
        status: "mc-krea-creative-status",
        prompt: "txt2img_prompt",
        generate: "txt2img_generate",
        // Where the host draws its progress bar for txt2img. The second is what
        // older layouts call it; whichever exists is where the roll's bar goes,
        // so it appears exactly where image progress appears rather than in a
        // place of this extension's choosing.
        gallery: "txt2img_gallery_container",
        results: "txt2img_results",
        images: "txt2img_gallery",
    };

    // How often the token box is read while a roll is in flight. It is a
    // property read on one element, so the cost is nil; the number is about how
    // quickly the image starts once the prompt is ready, and a tenth of a
    // second is under what anybody notices.
    const TOKEN_POLL_MS = 100;

    // How long to keep reading it before giving up. Generous on purpose: a cold
    // llama-server reads its weights off disk before it answers the first
    // request, and twenty seconds of that is normal rather than a fault.
    const TOKEN_TIMEOUT_MS = 15 * 60 * 1000;

    const ARMED_CLASS = "mc-krea-creative-armed";

    const state = {
        // Set when the native click is ours rather than the user's, and spent
        // by the very next click. One boolean, wrong in one direction if
        // nothing ever generates and in the other if every click generates
        // twice, which is why the tests drive it through the real listener.
        bypass: false,
        rolling: false,
        wired: false,
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

    function field(id, selector) {
        const holder = byId(id);
        return holder ? holder.querySelector(selector) : null;
    }

    function toggleBox() {
        return field(IDS.toggle, "input[type=checkbox]");
    }

    function tokenBox() {
        return field(IDS.token, "textarea, input");
    }

    function isCreative() {
        const box = toggleBox();
        return !!(box && box.checked);
    }

    function sourceText() {
        const box = field(IDS.prompt, "textarea");
        return box ? String(box.value || "").trim() : "";
    }

    // The one visible sign that Generate has changed meaning, painted on the
    // button whose meaning changed. A state indicator on the other side of the
    // panel is a state indicator nobody reads in time.
    function paintIndicator() {
        const button = clickable(IDS.generate);
        if (button) button.classList.toggle(ARMED_CLASS, isCreative());
    }

    // Put a line in the status element without a round trip. The server owns
    // that element and overwrites this on its next update, which is correct:
    // this is only for the thing the server never hears about, which is a click
    // the browser declined to pass on.
    function sayLocally(text) {
        try {
            const holder = byId(IDS.status);
            if (!holder) return;
            (holder.querySelector("div") || holder).textContent = text;
        } catch (error) {
            // A status line is not worth an exception.
        }
    }

    // Wait for the hidden token box to say something new. Resolves true when a
    // roll is ready and false for every other outcome, including Creative Mode
    // being switched off mid-flight -- and false always means "do not start a
    // native generation", which is the only decision the caller makes from it.
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
                if (!isCreative() || Date.now() - started > TOKEN_TIMEOUT_MS) {
                    window.clearInterval(poll);
                    resolve(false);
                }
            }, TOKEN_POLL_MS);
        });
    }

    // -- the host's own progress bar ----------------------------------------- //

    // Mint a task id, ask Forge to draw and poll its progress bar for it, and
    // hand the id to the server as argument zero.
    //
    // This is the host's own submit() idiom, deliberately: ui.js does exactly
    // these three things around every Generate. Doing it the same way means the
    // roll gets the real bar -- with its real ETA and its real Interrupt button
    // -- instead of something this extension drew that looks nearly like one.
    //
    // A js= hook rather than a hidden textbox because it is race-free. The id is
    // written into the request as Gradio builds it, rather than into a component
    // whose value may not have reached Gradio's state by the time the click is
    // processed.
    function submitRoll() {
        const args = Array.prototype.slice.call(arguments);
        let id;
        try {
            id = (typeof randomId === "function")
                ? randomId()
                : "task(mckrea" + Math.random().toString(36).slice(2, 9) + ")";
            args[0] = id;
        } catch (error) {
            console.error("Model Chain: could not mint a Creative Mode task id", error);
            return args;
        }

        try {
            if (typeof requestProgress === "function") {
                const container = byId(IDS.gallery) || byId(IDS.results);
                if (container) {
                    // No gallery is passed: a creative roll produces no image and
                    // has no live preview, and handing over the gallery would
                    // invite the progress plumbing to redraw one.
                    requestProgress(id, container, null, function () {});
                }
            }
        } catch (error) {
            // A bar that will not draw must never be a roll that will not run.
            console.error("Model Chain: could not start the Creative Mode progress bar",
                error);
        }
        return args;
    }

    // One roll, then one native generation. The only path to a Creative image.
    function runRoll() {
        if (state.rolling) return Promise.resolve(false);
        if (!sourceText()) return Promise.resolve(false);

        const button = clickable(IDS.run);
        if (!button) return Promise.resolve(false);

        const box = tokenBox();
        const previous = box ? String(box.value || "") : "";

        state.rolling = true;
        button.click();

        return awaitToken(previous).then(function (ready) {
            state.rolling = false;
            if (!ready) return false;
            const generate = clickable(IDS.generate);
            if (!generate) return false;
            state.bypass = true;
            generate.click();
            return true;
        }).catch(function (error) {
            console.error("Model Chain: the Creative Mode roll failed", error);
            state.rolling = false;
            return false;
        });
    }

    // Capture phase, on the document, so this runs before any handler the host
    // or another extension attached to the button itself.
    function onGenerateClick(event) {
        try {
            const button = clickable(IDS.generate);
            if (!button || !event.target) return;
            if (event.target !== button && !button.contains(event.target)) return;
            if (!isCreative()) return;

            if (state.bypass) {
                // Ours, and exactly one of them. Cleared before the click
                // proceeds, so a second programmatic click cannot ride through
                // on the same permission.
                state.bypass = false;
                return;
            }

            // A roll is already in flight and somebody pressed Generate again.
            // Swallowed rather than queued: the roll in flight will click
            // Generate itself when it is ready, and a queued second roll would
            // be a second model call nobody asked for.
            if (state.rolling) {
                event.preventDefault();
                event.stopImmediatePropagation();
                sayLocally("Still writing the prompt for the last press.");
                return;
            }

            event.preventDefault();
            event.stopImmediatePropagation();
            runRoll();
        } catch (error) {
            // A gate that throws must not be a Generate button that does
            // nothing: the click is left alone and txt2img behaves natively.
            console.error("Model Chain: the Creative Mode gate failed, generating natively",
                error);
        }
    }

    function wire() {
        try {
            paintIndicator();

            const toggle = toggleBox();
            if (toggle && !toggle.dataset.mcKreaCreative) {
                toggle.dataset.mcKreaCreative = "1";
                toggle.addEventListener("change", paintIndicator);
            }

            if (!state.wired) {
                state.wired = true;
                document.addEventListener("click", onGenerateClick, true);
            }
        } catch (error) {
            console.error("Model Chain: Creative Mode wiring failed", error);
        }
    }

    // Re-applied rather than installed once: Gradio rebuilds parts of the tab
    // on some updates. Both wirings are idempotent -- one through a dataset
    // flag, one through a flag on state -- so re-running costs a query.
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

    // Named on window because Gradio resolves a js= hook by name at call time.
    // This is the one thing in this file the page itself calls.
    window.mcKreaCreativeSubmit = submitRoll;

    // Exposed for the tests, which drive this file under node against a fake
    // page. Nothing in the extension reads it.
    window.modelChainKreaCreative = {
        state: state,
        onGenerateClick: onGenerateClick,
        runRoll: runRoll,
        submitRoll: submitRoll,
        wire: wire,
    };
})();
