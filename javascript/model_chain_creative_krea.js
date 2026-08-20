// Model Chain -- Krea Creative Mode, browser side.
//
// One job, and it is cosmetic: while Creative Mode is on, say so on the button
// it changes the behaviour of. Everything else about the feature is Python.
//
// Why there is so little here
// ---------------------------
// There used to be a gate. A press of Generate was intercepted, a hidden button
// was pressed to run the creative roll, a hidden textbox was polled until the
// server said the prompt was ready, and only then was Generate clicked again
// with a one-shot flag that let that click through. The order mattered -- the
// roll ends in a language-model request, and one of those used to refuse to
// start while the host was generating -- and in front of the click was the only
// place the browser could enforce it.
//
// The price was that a generation could not finish without this file running.
// The press started a roll and nothing else; what actually started the image
// was a setInterval callback, and browsers throttle those to one tick a second
// in a hidden tab, one a minute in a frozen one, and none at all in a tab that
// has been closed. Change windows and the image was late. Close the tab and the
// roll's result sat armed on the server with nothing left to spend it.
//
// So the ordering is enforced in Python now, where the thing being ordered
// lives: the roll runs inside the generation, from before_process, and
// mc_broker.host_job() tells the language model that the image job is blocked
// waiting for it rather than competing with it. Generate is an ordinary
// Generate again. Press it and close the browser; Forge finishes the job and
// writes the files, exactly as it does for every other generation.
//
// What is deliberately absent
// ---------------------------
// No click listener. No idle timer. No observer on the prompt box. No repeat
// loop. No status machine. Nothing here can stop, delay or start a generation,
// which is the property worth keeping: the worst this file can do if every line
// of it fails is leave a button unpainted.

(function () {
    "use strict";

    const IDS = {
        toggle: "mc-krea-creative-toggle",
        generate: "txt2img_generate",
    };

    const ARMED_CLASS = "mc-krea-creative-armed";

    const state = {
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

    function toggleBox() {
        const holder = byId(IDS.toggle);
        return holder ? holder.querySelector("input[type=checkbox]") : null;
    }

    function isCreative() {
        const box = toggleBox();
        return !!(box && box.checked);
    }

    // The one visible sign that Generate will write a prompt before it makes a
    // picture, painted on the button that will do it. A state indicator on the
    // other side of the panel is a state indicator nobody reads in time.
    function paintIndicator() {
        const button = clickable(IDS.generate);
        if (button) button.classList.toggle(ARMED_CLASS, isCreative());
    }

    function wire() {
        try {
            paintIndicator();

            const toggle = toggleBox();
            if (toggle && !toggle.dataset.mcKreaCreative) {
                toggle.dataset.mcKreaCreative = "1";
                toggle.addEventListener("change", paintIndicator);
            }
            state.wired = true;
        } catch (error) {
            console.error("Model Chain: Creative Mode wiring failed", error);
        }
    }

    // Re-applied rather than installed once: Gradio rebuilds parts of the tab
    // on some updates. The wiring is idempotent -- one dataset flag -- so
    // re-running costs a query.
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
    window.modelChainKreaCreative = {
        state: state,
        paintIndicator: paintIndicator,
        wire: wire,
    };
})();
