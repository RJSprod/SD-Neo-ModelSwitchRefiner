// Model Chain -- a quieter night mode.
//
// In the dark theme the host paints a primary button -- Generate above all --
// as a solid block of the theme's accent, and a theme is entitled to a light
// accent: LobeTheme's night mode has a white one out of the box, which made
// Generate the brightest thing on a black page, with Send, Apply settings,
// Add to Queue and every other primary button the same white.
//
// The Settings toggle this reads turns those blocks into a dim, accent-tinted
// surface. style.css does the drawing, keyed on the one attribute this file
// puts on the root; nothing is written on any button, so a theme or a user
// stylesheet can still override the rule the ordinary way, and turning the
// setting off takes the attribute off at the next update -- the solid buttons
// come straight back, no reload.
//
// The same shape as LLM Studio's footer switch (javascript/llm_studio.js):
// read the option off the host's `opts`, write the attribute only when its
// value moves, and never throw -- a script this small must not be able to take
// a page's other wiring down with it.
(function () {
    "use strict";

    // The option Python registers (scripts/model_chain.py, OPT_QUIET_NIGHT).
    const OPTION = "model_chain_quiet_night";
    // The attribute the stylesheet's rules are keyed on, and its one value.
    const ATTRIBUTE = "data-mc-night";
    const QUIET = "quiet";

    function setting(name, fallback) {
        // The host publishes every registered option on a global. Read
        // defensively: it does not exist before the settings have loaded.
        try {
            if (typeof opts === "undefined" || opts === null) return fallback;
            const value = opts[name];
            return value === undefined || value === null ? fallback : value;
        } catch (error) {
            return fallback;
        }
    }

    function apply() {
        try {
            const html = document.documentElement;
            if (!html) return;
            // The fallback matches the option's default, so the buttons do
            // not flash bright for the moment before the host publishes
            // `opts`.
            const wanted = setting(OPTION, true) ? QUIET : "";
            const current = html.getAttribute(ATTRIBUTE);
            if (!wanted) {
                if (current !== null) html.removeAttribute(ATTRIBUTE);
                return;
            }
            // Written only when it changes: this runs after every update the
            // host makes, and setting an attribute invalidates style whether
            // or not the value moved.
            if (current !== wanted) html.setAttribute(ATTRIBUTE, wanted);
        } catch (error) {
            console.error("Model Chain: could not apply the night mode setting", error);
        }
    }

    // At once, with the default, so the first paint is already the quiet one;
    // again when the page is up, whenever the settings change, and after
    // every update as the safety net for a host that publishes `opts` without
    // saying so.
    apply();
    if (typeof onUiLoaded === "function") onUiLoaded(apply);
    if (typeof onOptionsChanged === "function") onOptionsChanged(apply);
    if (typeof onAfterUiUpdate === "function") onAfterUiUpdate(apply);
})();
