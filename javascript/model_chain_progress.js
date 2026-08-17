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
//
// ---------------------------------------------------------------------------
// A note on why the requestProgress wrapper below is so defensive.
//
// The host's submit() runs in this order:
//
//     const id = randomId();
//     requestProgress(id, ...);            // <- the wrapper runs here
//     const res = create_submit_args(arguments);
//     res[0] = id;                         // <- the backend learns the id here
//     return res;
//
// submit() is a Gradio `_js` handler, and `res[0]` is what reaches
// `wrap_gradio_gpu_call`, where `id_task = args[0]` becomes the task the
// progress endpoint reports on. So an exception escaping requestProgress does
// not merely lose the styling -- it aborts submit() before the task id is
// attached, the browser polls for an id the backend never registered, and
// /internal/progress answers `active: false` for the whole run. The bar then
// sits on "Waiting..." until the generation finishes and it disappears, with
// the images arriving normally and nothing in the log to connect the two.
//
// A cosmetic feature must not be able to do that. Hence: the wrapper is only
// installed when something actually needs it, every added line runs inside its
// own guard, and the host's call is made unconditionally even if all of them
// failed.
// ---------------------------------------------------------------------------

"use strict";

(function () {
    const ROOT_CLASS = "mc-progress-styled";
    const SMOOTH_CLASS = "mc-progress-smooth";
    const FLASH_CLASS = "mc-progress-flash";

    // What the host polls at when it has no opinion, matching the fallback in
    // its own javascript/progressbar.js.
    const DEFAULT_TICK_MS = 500;

    // The four effects style.css implements. A named theme is a choice of
    // these; the Custom theme is the same choice made by the user's toggles.
    // Keeping the themes here rather than as CSS of their own means there is
    // one implementation of each effect to get right.
    const EFFECTS = ["gradient", "sheen", "glow", "pulse", "intense", "ooze"];

    const THEMES = {
        // Plain fill in the chosen colour. What the host does, recoloured.
        Flat: {},
        // A single lightening towards the leading edge, so the bar reads as
        // moving between percentage updates.
        Gradient: { gradient: true },
        // The above plus a highlight band travelling the bar.
        Sheen: { gradient: true, sheen: true },
        // Flat fill with a breathing halo. Legible from across the room
        // without anything moving inside the bar.
        Pulse: { glow: true, pulse: true },
        // Everything, turned up. The loud one.
        Neon: { gradient: true, sheen: true, glow: true, intense: true },
        // Toxic sludge. The only theme that paints outside the bar: bubbles
        // rise through the fill and escape above it. See the ooze section of
        // style.css for the two exceptions it makes and why.
        Ooze: { ooze: true },
    };

    const DEFAULT_THEME = "Flat";
    const CUSTOM_THEME = "Custom";

    // Themes are exported so the Settings dropdown and this file cannot drift
    // apart: Python builds the choice list from the same names.
    window.modelChainProgressThemes = Object.keys(THEMES).concat([CUSTOM_THEME]);

    // A job that ends below this is treated as interrupted rather than
    // finished, and gets no completion effect. The host cannot tell us which it
    // was -- an interrupted task is "completed" as far as /internal/progress is
    // concerned -- so the last progress value seen is the only signal there is.
    const COMPLETE_THRESHOLD = 0.95;

    const FLASH_MS = 700;

    let warnedAboutColor = false;
    let overlayChecked = false;

    function root() {
        return document.documentElement;
    }

    function setting(name, fallback) {
        try {
            if (typeof opts === "undefined" || opts === null) return fallback;
            const value = opts[name];
            return value === undefined || value === null ? fallback : value;
        } catch (error) {
            return fallback;
        }
    }

    function styleEnabled() {
        return Boolean(setting("model_chain_style_enable", false));
    }

    function flashEnabled() {
        return styleEnabled() && Boolean(setting("model_chain_style_complete", false));
    }

    // -- smooth advance -----------------------------------------------------

    function smoothEnabled() {
        return Boolean(setting("model_chain_smooth_progress", true));
    }

    // The gap the transition has to cover is the host's own poll interval, and
    // the host publishes it. Reading it rather than assuming means a user who
    // has turned the refresh rate down still gets a bar that glides instead of
    // one that races ahead and then waits.
    //
    // Stretched past that interval on purpose. The host schedules its next poll
    // from inside the response handler, so the true gap between two width
    // writes is the configured period *plus* a round trip to the server. A
    // transition of exactly the period therefore always lands early and leaves
    // the bar sitting still until the next one arrives -- measurably: glide,
    // hold, glide. Overshooting costs a little lag on a value that only ever
    // increases, which is invisible, where the stall is the entire complaint.
    const TICK_SLACK = 1.4;

    function tickMilliseconds() {
        const raw = Number(setting("live_preview_refresh_period", DEFAULT_TICK_MS));
        const period = Number.isFinite(raw) && raw > 0 ? raw : DEFAULT_TICK_MS;
        return Math.round(period * TICK_SLACK);
    }

    // -- appearance ---------------------------------------------------------

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

    function resolveEffects() {
        const name = String(setting("model_chain_style_theme", DEFAULT_THEME));

        if (name === CUSTOM_THEME) {
            return {
                gradient: Boolean(setting("model_chain_style_gradient", false)),
                sheen: Boolean(setting("model_chain_style_sheen", false)),
                glow: Boolean(setting("model_chain_style_glow", false)),
                // The glow toggle carries its own animation: a static halo on a
                // bar that is otherwise moving reads as a rendering fault.
                pulse: Boolean(setting("model_chain_style_glow", false)),
            };
        }

        // A theme saved before this version, or one typed by hand, falls back
        // rather than leaving the bar unstyled with the toggle switched on.
        return THEMES[name] || THEMES[DEFAULT_THEME];
    }

    function apply() {
        try {
            const element = root();
            if (!element) return;

            // Independent of everything below it. Smoothing the host's own bar
            // is not part of a theme and is not gated on one being chosen.
            element.classList.toggle(SMOOTH_CLASS, smoothEnabled());
            element.style.setProperty("--mc-progress-tick", tickMilliseconds() + "ms");

            const on = styleEnabled();
            element.classList.toggle(ROOT_CLASS, on);

            const effects = on ? resolveEffects() : {};
            for (const effect of EFFECTS) {
                element.classList.toggle(`mc-fx-${effect}`, Boolean(effects[effect]));
            }

            const color = on ? resolveColor() : "";
            if (color) {
                element.style.setProperty("--mc-progress-fill", color);
            } else {
                // Removing rather than resetting hands the colour back to the
                // stylesheet default, which is the active theme's own accent.
                element.style.removeProperty("--mc-progress-fill");
            }

            // A theme change can re-enable an effect the overlay check removed.
            overlayChecked = false;

            // Turning the flash on mid-session is what installs the wrapper.
            if (flashEnabled()) wrapRequestProgress();
        } catch (error) {
            console.error("Model Chain: could not apply the progress-bar appearance", error);
        }
    }

    // -- co-existing with a theme that decorates the bar --------------------

    // Some WebUI themes animate their own overlay on the bar through
    // .progress::before -- Lobe draws travelling diagonal stripes there. Our
    // sheen would then be the second thing moving on one small element, which
    // looks like a fault rather than a style. Rather than name the themes this
    // could apply to, the actual conflict is measured: if a pseudo-element with
    // content exists on the live bar, the sheen stands down and the colour and
    // glow carry the look.
    function checkForThemeOverlay(bar) {
        if (overlayChecked || !bar) return;
        overlayChecked = true;

        if (!root().classList.contains("mc-fx-sheen")) return;

        try {
            const before = getComputedStyle(bar, "::before");
            const decorated =
                before &&
                before.content &&
                before.content !== "none" &&
                before.content !== "normal";
            if (decorated) {
                root().classList.remove("mc-fx-sheen");
            }
        } catch (error) {
            // A browser that will not report pseudo-element styles simply
            // keeps the sheen; the worst case is a busy-looking bar.
        }
    }

    // -- ooze ---------------------------------------------------------------

    const OOZE_OVERLAY = "mc-ooze-overlay";

    // Dense enough to read as a rolling boil rather than as separate circles.
    // They animate transform and opacity only, so the compositor carries them
    // and the count is far cheaper than the number suggests.
    const BUBBLE_COUNT = 120;

    // Bubbles still under the surface. Fewer, because they are confined to the
    // fill's own height rather than to the band above it -- but placed and
    // timed by exactly the same code, which is the whole reason they exist.
    // They replaced two tiled background layers, and a tiled background is a
    // grid: those bubbles sat in exact rows and a whole row broke the surface
    // at once, so the sludge bunched while the air above it did not.
    const INNER_COUNT = 80;

    // Inset from both ends. A bubble centred on the very first pixel is half
    // clipped by the overlay's own left edge; at the right edge that is wanted,
    // because it is the ooze front arriving, but on the left it is just a half
    // circle sitting there for the whole job.
    const FIELD_START = 3;
    const FIELD_WIDTH = 94;

    function oozeEnabled() {
        return styleEnabled() && Boolean(resolveEffects().ooze);
    }

    // One generator for both populations. The submerged bubbles and the ones in
    // the air differ only in how far they travel and how they are drawn; their
    // placement and timing come from the same lines, so the two can never bunch
    // differently from one another.
    function seedBubbles(overlay, count, className, sizeAt, liftAt) {
        // Stratified rather than uniform: the width is divided into one cell
        // per bubble and each is placed at random *within its own cell*.
        //
        // Plain Math.random() across the whole width is what produced the
        // clumps-and-voids the field was reported for. Uniform random points
        // are not evenly spaced -- they cluster, by definition -- so at any
        // density there are stretches with four bubbles overlapping and
        // stretches with none. One per cell keeps the spacing even while the
        // jitter inside the cell keeps it from looking like a comb.
        const cell = FIELD_WIDTH / count;

        for (let i = 0; i < count; i++) {
            const bubble = document.createElement("span");
            bubble.className = className;

            // Bigger bubbles rise further and slower, as they would in
            // something viscous. Tying the three together stops the field
            // looking like noise.
            //
            // Biased towards the small end: a uniform spread at this density
            // reads as a clutter of blobs, where mostly-small with a few large
            // ones reads as fizz.
            const scale = Math.pow(Math.random(), 1.35);
            const duration = 0.35 + scale * 0.42 + Math.random() * 0.2;

            bubble.style.setProperty("--x", (FIELD_START + (i + Math.random()) * cell).toFixed(2) + "%");
            bubble.style.setProperty("--size", sizeAt(scale).toFixed(1) + "px");
            bubble.style.setProperty("--dur", duration.toFixed(2) + "s");
            // Negative, so every bubble is already mid-flight on the first
            // frame instead of the whole field launching at once -- and scaled
            // to this bubble's *own* duration, which is what makes the phases
            // uniform. A delay drawn from one fixed window instead left every
            // bubble's position in its cycle correlated with its speed, so they
            // rose in visible ranks with gaps between them.
            bubble.style.setProperty("--delay", (-Math.random() * duration).toFixed(3) + "s");
            bubble.style.setProperty("--drift", (Math.random() * 16 - 8).toFixed(1) + "px");
            bubble.style.setProperty("--lift", liftAt(scale).toFixed(0) + "px");
            overlay.appendChild(bubble);
        }
    }

    function buildBubbles(overlay, surface) {
        // Above the surface: a tight band rather than a fountain. The bar sits
        // directly under the Generate button in the host's layout, and bubbles
        // carrying on up into it stop reading as a surface fizzing and start
        // reading as something escaping.
        seedBubbles(
            overlay,
            BUBBLE_COUNT,
            "mc-ooze-bubble",
            (scale) => 3.5 + scale * 8,
            (scale) => 10 + scale * 16 + Math.random() * 10,
        );

        // Below it: bounded by the fill's own height, so they fade just under
        // the skin rather than crossing it. A bar too short to show any travel
        // gets none of them rather than a row of dots that never move.
        const headroom = Math.max(surface - 6, 0);
        if (headroom < 5) return;

        seedBubbles(
            overlay,
            INNER_COUNT,
            "mc-ooze-inner",
            (scale) => 2 + scale * 4,
            (scale) => headroom * (0.55 + scale * 0.35 + Math.random() * 0.1),
        );
    }

    // The overlay is clipped at the ooze front, so a bubble only appears once
    // the sludge has reached its position. The front is the fill's own width,
    // which the host writes as an inline style on every poll -- so it is read
    // back from there rather than recomputed.
    function trackLevel(overlay, bar) {
        const update = () => {
            try {
                overlay.style.setProperty("--mc-ooze-level", bar.style.width || "0%");
                // The height of the fill is where the surface is, and it is the
                // active WebUI theme's decision rather than ours -- Forge uses
                // 20px, another theme need not. Measuring it is what lets the
                // bubbles break the surface instead of the floor.
                const height = bar.offsetHeight;
                if (height > 0) {
                    overlay.style.setProperty("--mc-ooze-surface", height + "px");
                }
            } catch (error) {
                /* the bar has gone; the overlay goes with it */
            }
        };
        update();

        const observer = new MutationObserver(update);
        observer.observe(bar, { attributes: true, attributeFilter: ["style"] });
        return observer;
    }

    function decorate(progressDiv) {
        if (!progressDiv || progressDiv.querySelector(`.${OOZE_OVERLAY}`)) return;
        if (!oozeEnabled()) return;

        const bar = progressDiv.querySelector(".progress");
        if (!bar) return;

        const overlay = document.createElement("div");
        overlay.className = OOZE_OVERLAY;
        // The submerged bubbles have to fit inside the fill, so how tall it is
        // decides how far they can travel -- and that is the active WebUI
        // theme's decision, not ours.
        buildBubbles(overlay, bar.offsetHeight || 20);
        // Appended to .progressDiv, not to .progress: the host rewrites
        // .progress's textContent twice a second, which would wipe these
        // within half a second of adding them.
        progressDiv.appendChild(overlay);

        trackLevel(overlay, bar);
    }

    // Watching the document rather than hooking the host's submit path. The bar
    // is created and destroyed per generation, so something has to observe that
    // lifecycle -- but an observer runs after the fact and cannot abort
    // anything, which is exactly the property the note at the top of this file
    // says the alternative lacks.
    function watchForBars() {
        if (typeof MutationObserver !== "function" || !document.body) return;

        new MutationObserver((records) => {
            for (const record of records) {
                for (const node of record.addedNodes) {
                    if (!node || node.nodeType !== 1) continue;
                    try {
                        if (node.classList && node.classList.contains("progressDiv")) {
                            decorate(node);
                        } else if (node.querySelector) {
                            node.querySelectorAll(".progressDiv").forEach(decorate);
                        }
                    } catch (error) {
                        console.error("Model Chain: could not decorate the progress bar", error);
                    }
                }
            }
        }).observe(document.body, { childList: true, subtree: true });
    }

    // -- completion effect -------------------------------------------------

    function clearFlash(parent) {
        if (!parent || !parent.querySelectorAll) return;
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

    // requestProgress is a plain global function declaration in the host's
    // javascript/progressbar.js, and its callers look it up by name at click
    // time, so replacing the global is enough.
    //
    // Installed lazily rather than at load: the completion flash is the only
    // thing that needs it and is off by default, so an install that never turns
    // it on never has this hook at all. See the note at the top of the file for
    // what is at stake if this ever misbehaves.
    function wrapRequestProgress() {
        if (typeof window.requestProgress !== "function") return false;
        if (window.requestProgress.mcWrapped) return true;

        const original = window.requestProgress;

        const wrapped = function (id_task, progressbarContainer, gallery, atEnd, onProgress, inactivityTimeout) {
            let hooks = null;

            try {
                let peak = 0;
                const parent = progressbarContainer && progressbarContainer.parentNode;
                clearFlash(parent);

                hooks = {
                    parent: parent,
                    onProgress: function (res) {
                        try {
                            if (res && typeof res.progress === "number" && res.progress > peak) {
                                peak = res.progress;
                            }
                        } catch (error) {
                            console.error(error);
                        }
                        if (onProgress) onProgress(res);
                    },
                    atEnd: function () {
                        try {
                            if (flashEnabled() && peak >= COMPLETE_THRESHOLD) {
                                showFlash(progressbarContainer);
                            }
                        } catch (error) {
                            console.error(error);
                        }
                        if (atEnd) atEnd();
                    },
                };
            } catch (error) {
                console.error("Model Chain: progress-bar decoration disabled for this run", error);
            }

            // The host's call happens either way. If anything above failed, its
            // own callbacks are passed straight through, and the only thing
            // lost is the flash.
            const result = hooks
                ? original(id_task, progressbarContainer, gallery, hooks.atEnd, hooks.onProgress, inactivityTimeout)
                : original(id_task, progressbarContainer, gallery, atEnd, onProgress, inactivityTimeout);

            try {
                if (hooks && hooks.parent) {
                    // The host builds the bar synchronously above; one frame
                    // later it has been laid out and its styles can be read.
                    requestAnimationFrame(() =>
                        checkForThemeOverlay(hooks.parent.querySelector(".progressDiv > .progress")),
                    );
                }
            } catch (error) {
                console.error(error);
            }

            return result;
        };

        wrapped.mcWrapped = true;
        window.requestProgress = wrapped;
        return true;
    }

    // -- wiring ------------------------------------------------------------

    if (typeof onOptionsAvailable === "function") onOptionsAvailable(apply);
    if (typeof onOptionsChanged === "function") onOptionsChanged(apply);
    if (typeof onUiLoaded === "function") {
        onUiLoaded(function () {
            apply();
            try {
                watchForBars();
            } catch (error) {
                console.error("Model Chain: could not watch for the progress bar", error);
            }
        });
    }
})();
