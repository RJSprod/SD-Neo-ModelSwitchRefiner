// Model Chain -- optional progress-bar appearance.
//
// This file is the whole styling layer, and it is deliberately separate from
// the extension's Python: it changes how the host's progress bar looks and
// never what it reports. Turning it off does not affect Model Chain's
// whole-job progress calculation, and turning it on does not alter the numbers
// an ordinary Forge generation produces.
//
// Everything is driven by CSS classes and one custom property on the document
// element. The host creates and destroys .progressDiv on every generation
// (javascript/progressbar.js), so there is nothing stable to decorate directly
// -- class-and-variable styling survives that lifecycle, one-time DOM edits do
// not.

"use strict";

(function () {
    const ROOT_CLASS = "mc-progress-styled";
    const GRADIENT_CLASS = "mc-progress-gradient";
    const GLOW_CLASS = "mc-progress-glow";
    const FLASH_CLASS = "mc-progress-flash";

    // A job that ends below this is treated as interrupted rather than
    // finished, and gets no completion effect. The host cannot tell us which it
    // was -- an interrupted task is "completed" as far as /internal/progress is
    // concerned -- so the last progress value seen is the only signal there is.
    const COMPLETE_THRESHOLD = 0.95;

    const FLASH_MS = 700;

    let warnedAboutColor = false;

    function root() {
        return document.documentElement;
    }

    function setting(name, fallback) {
        if (typeof opts === "undefined" || opts === null) return fallback;
        const value = opts[name];
        return value === undefined || value === null ? fallback : value;
    }

    function resolveColor() {
        const raw = String(setting("model_chain_style_color", "") || "").trim();
        if (!raw) return "";

        // Anything CSS understands is allowed through -- named colours, hex,
        // rgb(), rgba(), hsl(), var(). Anything it does not is dropped rather
        // than written to the variable, where it would silently blank the fill.
        const supported = typeof CSS !== "undefined" && CSS.supports && CSS.supports("color", raw);
        if (!supported) {
            if (!warnedAboutColor) {
                console.warn(`Model Chain: "${raw}" is not a colour CSS understands; using the theme accent instead.`);
                warnedAboutColor = true;
            }
            return "";
        }
        return raw;
    }

    function apply() {
        const element = root();
        if (!element) return;

        const on = Boolean(setting("model_chain_style_enable", false));
        element.classList.toggle(ROOT_CLASS, on);
        element.classList.toggle(GRADIENT_CLASS, on && Boolean(setting("model_chain_style_gradient", false)));
        element.classList.toggle(GLOW_CLASS, on && Boolean(setting("model_chain_style_glow", false)));

        const color = on ? resolveColor() : "";
        if (color) {
            element.style.setProperty("--mc-progress-fill", color);
        } else {
            // Removing rather than resetting hands the colour back to the
            // stylesheet default, which is the active theme's own accent.
            element.style.removeProperty("--mc-progress-fill");
        }
    }

    function flashEnabled() {
        return Boolean(setting("model_chain_style_enable", false)) && Boolean(setting("model_chain_style_complete", false));
    }

    function clearFlash(parent) {
        if (!parent) return;
        parent.querySelectorAll(`.${FLASH_CLASS}`).forEach((node) => node.remove());
    }

    function showFlash(container) {
        const parent = container && container.parentNode;
        if (!parent) return;

        clearFlash(parent);

        // The host's own classes come along so the flash inherits whatever
        // geometry the active theme gives the real bar. Themes move this
        // element -- Lobe takes it out of the host's absolute overlay and puts
        // it in normal flow -- and borrowing the classes is what keeps the
        // flash in the same place as the bar it replaces under either.
        const flash = document.createElement("div");
        flash.className = `progressDiv ${FLASH_CLASS}`;
        const fill = document.createElement("div");
        fill.className = "progress";
        flash.appendChild(fill);

        const remove = () => flash.remove();
        fill.addEventListener("animationend", remove, { once: true });
        // Backstop: if the animation never runs (reduced motion, a theme that
        // suppresses it) the element must still go, or it stays on screen as a
        // full bar over an idle generate button.
        setTimeout(remove, FLASH_MS);

        parent.insertBefore(flash, container);
    }

    // -- completion effect -------------------------------------------------

    // requestProgress is a plain global function declaration in the host's
    // javascript/progressbar.js, and its callers look it up by name at click
    // time, so replacing the global is enough. Extension scripts are injected
    // after the host's own, so it is already defined by the time this runs.
    function wrapRequestProgress() {
        if (typeof window.requestProgress !== "function") return false;
        if (window.requestProgress.mcWrapped) return true;

        const original = window.requestProgress;

        const wrapped = function (id_task, progressbarContainer, gallery, atEnd, onProgress, inactivityTimeout) {
            let peak = 0;

            clearFlash(progressbarContainer && progressbarContainer.parentNode);

            const trackProgress = function (res) {
                if (res && typeof res.progress === "number" && res.progress > peak) {
                    peak = res.progress;
                }
                if (onProgress) onProgress(res);
            };

            const wrappedAtEnd = function () {
                try {
                    if (flashEnabled() && peak >= COMPLETE_THRESHOLD) {
                        showFlash(progressbarContainer);
                    }
                } catch (error) {
                    console.error(error);
                }
                if (atEnd) atEnd();
            };

            return original(id_task, progressbarContainer, gallery, wrappedAtEnd, trackProgress, inactivityTimeout);
        };

        wrapped.mcWrapped = true;
        window.requestProgress = wrapped;
        return true;
    }

    // -- wiring ------------------------------------------------------------

    wrapRequestProgress();

    if (typeof onOptionsAvailable === "function") onOptionsAvailable(apply);
    if (typeof onOptionsChanged === "function") onOptionsChanged(apply);
    if (typeof onUiLoaded === "function") {
        onUiLoaded(function () {
            wrapRequestProgress();
            apply();
        });
    }
})();
