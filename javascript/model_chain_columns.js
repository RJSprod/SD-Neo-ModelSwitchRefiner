// Model Chain -- Txt2Img's and Img2Img's two columns, swapped by a double tap
// on the divider between them.
//
// Asked (2026-11-23): "I want an option to be able to switch the two main
// columns on text to image and image to image tab. I'm thinking, double tab the
// drag area that i can drag to resize the columns should switch them left and
// right. Double tap seems like a natural gesture."
//
// The drag area is Forge's own divider. Forge Neo's javascript/resizeHandle.js
// makes the row holding `#<tab>_settings` and `#<tab>_results` a grid of three
// tracks -- the settings, a 16 px handle, the results -- and sizes the first
// track by a drag of the handle; a double click on the handle, or two touches
// on it within 200 ms, put its first template back. On these two rows that
// double tap is this file's now: it puts each column on the other side, and a
// second one puts them back. Kept in each browser, per tab.
//
// How the columns change sides
// ----------------------------
// Nothing is moved. Gradio, Forge's scripts, the Forge Assistant's dock and the
// Model Chain theme all hold these columns by id, and none of them needs the
// results column to come second in the row. The stylesheet orders the grid's
// items -- the results, the handle, the settings -- under an attribute on the
// root naming the swapped tabs (`data-mc-columns-swapped`), written from this
// browser's choice when the script loads, so a page is drawn swapped from the
// moment Forge builds its grid. The order holds only while Forge's divider
// shows: below 704 px Forge stacks the row (a flex row, the handle hidden) and
// the stacking stays Forge's, the settings on top.
//
// Each column keeps its width across a swap. Forge's template names the first
// track's width: its first one, equal halves, needs nothing; any other (a drag
// writes `Npx 16px 1fr`) is written again with the width of the column now
// first -- the one that was last.
//
// What would have gone wrong
// --------------------------
// - Forge's drag sizes the first track from the pointer's travel and from the
//   width of the row's first child in the document (`leftCol`, kept in a
//   closure) -- the settings, wherever they are drawn. Swapped, the first
//   track holds the results, which a drag would have jumped to the settings'
//   width. So while a tab is swapped this file takes the handle's press in the
//   window's capture phase, before it reaches Forge's listener on the handle,
//   and drags the first track itself, writing Forge's own `Npx 16px 1fr`, so
//   Forge's scaling of it on a window resize goes on working.
// - Forge's mobile script (extensions-builtin/mobile) takes a results column at
//   `offsetLeft === 0` for a phone's stacked layout and moves Generate's box
//   into it at the next window resize. Swapped, the results are the row's left
//   edge: the stylesheet gives them a pixel of margin, as the Forge Assistant's
//   dock does (`forge-assistant-docked-beside`).
// - The Model Chain theme's one scroller and the Forge Assistant's fill took
//   "the settings' right edge past the results' left" for a stacked row; they
//   ask now whether the two columns share any of the row's width.
//
// The Forge Assistant's panel: while it is open, a press outside it only puts
// it away (the user's rule: "the first tap outside of an open flyout menu to do
// nothing except close the flyout menu ... I have to tap twice to do something
// outside"). A double tap whose first press landed while it was open is that
// closing tap and one plain tap on the divider: neither a swap nor Forge's
// reset. Decided from the assistant's own state, read at the press -- its
// panel open with Tap Outside to Close on, for a touch; for a double click,
// whose first press the browser never hands the page once the assistant has
// cancelled it, the tap it records as it closes the panel (`lastTapOutside`).
//
// Touch: the window's touch listener is passive, as every listener there should
// be, and stopping Forge's listener needs no more than `stopPropagation`; the
// page does not pan under a finger on the divider because the stylesheet says
// `touch-action: none` about it. Nothing cancels the double tap's touches, so
// the browser makes mouse events of them, which land where the finger was --
// on the other column, now: those are taken for half a second (`swallow`).
(function () {
    "use strict";

    const VERSION = 1;
    const TABS = ["txt2img", "img2img"];
    // The Forge setting (Settings, Model Chain: Page), on by default.
    const OPTION = "model_chain_swap_columns";
    // Per browser and per tab: "1" while swapped.
    const STORE = "mc-columns-swapped:";
    // The setting as last read, for the first paint: Forge's options arrive
    // after the page is built.
    const ENABLED_STORE = "mc-columns-enabled";
    // On the root: the tabs drawn swapped, by name.
    const ATTRIBUTE = "data-mc-columns-swapped";
    // Forge's own numbers (resizeHandle.js): the handle's track, and the
    // narrowest a column may be dragged when the row does not say.
    const GUTTER = 16;
    const MIN_COLUMN = 320;
    // Two touches on the divider this close together are a double tap -- the
    // double-tap time Android uses everywhere; Forge allowed 200 ms.
    const DOUBLE_TAP_MS = 300;
    // A first touch that travelled further than this was a drag, and the
    // touch after it is not the second of a double tap.
    const TAP_SLOP = 10;
    // One gesture, one swap: a second swap asked for this soon after the
    // first is the same gesture's -- a browser reporting it twice.
    const SWAP_GUARD_MS = 500;
    // A double click whose first click put the Forge Assistant's panel away:
    // the closing tap this long before it at most -- the longest double-click
    // time a system lets its user choose is under a second -- and this close.
    const CLOSING_TAP_MS = 1000;
    const CLOSING_TAP_PX = 16;

    const state = {lastSwap: -Infinity, tap: null, drag: null, tapWatch: null, swallowUntil: -Infinity};

    function app() {
        try {
            if (typeof gradioApp === "function") return gradioApp() || document;
        } catch (error) { /* fall through */ }
        return document;
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

    function readStore(key) {
        try { return window.localStorage.getItem(key); } catch (error) { return null; }
    }

    function writeStore(key, value) {
        try {
            if (value === null) window.localStorage.removeItem(key);
            else window.localStorage.setItem(key, value);
        } catch (error) { /* a private window keeps the choice for this page only */ }
    }

    /** Whether the swap is on: Forge's setting, or the last reading of it
     *  until Forge's options have arrived. */
    function enabled() {
        const value = setting(OPTION, undefined);
        if (value === undefined) return readStore(ENABLED_STORE) !== "0";
        return value !== false;
    }

    function rememberEnabled() {
        const value = setting(OPTION, undefined);
        if (value === undefined) return;
        const text = value === false ? "0" : "1";
        if (readStore(ENABLED_STORE) !== text) writeStore(ENABLED_STORE, text);
    }

    function kept(tab) { return readStore(STORE + tab) === "1"; }

    function keep(tab, swapped) { writeStore(STORE + tab, swapped ? "1" : null); }

    /** The tabs to draw swapped: those kept so, while the setting is on. */
    function wanted() { return enabled() ? TABS.filter(kept) : []; }

    /** Whether `tab` is drawn swapped now: the root's attribute says. */
    function shown(tab) {
        const value = document.documentElement.getAttribute(ATTRIBUTE) || "";
        return value.split(/\s+/).indexOf(tab) >= 0;
    }

    function writeAttribute(tabs) {
        const root = document.documentElement;
        const value = tabs.join(" ");
        if (value) {
            if (root.getAttribute(ATTRIBUTE) !== value) root.setAttribute(ATTRIBUTE, value);
        } else if (root.hasAttribute(ATTRIBUTE)) {
            root.removeAttribute(ATTRIBUTE);
        }
    }

    // -- the rows ----------------------------------------------------------------- //

    /** Forge's grid row holding `tab`'s results, or null. */
    function rowOf(tab) {
        const root = app();
        const results = root.querySelector ? root.querySelector("#" + tab + "_results") : null;
        const row = results && results.parentElement;
        return row && row.classList && row.classList.contains("resize-handle-row") ? row : null;
    }

    function handleOf(row) {
        return Array.prototype.find.call(row.children, (child) => child.classList.contains("resize-handle")) || null;
    }

    /** The tab whose divider `node` is (or is inside), or "" for any other
     *  node -- Forge's other dividers (Extras, the extra networks' tree)
     *  are left to Forge. */
    function tabOfHandle(node) {
        const handle = node && node.closest ? node.closest(".resize-handle") : null;
        const row = handle && handle.parentElement;
        if (!row || !row.classList || !row.classList.contains("resize-handle-row")) return "";
        for (const tab of TABS) {
            const results = app().querySelector ? app().querySelector("#" + tab + "_results") : null;
            if (results && results.parentElement === row) return tab;
        }
        return "";
    }

    /** Whether the row is Forge's grid with its divider showing: the only
     *  layout the swap is drawn in. */
    function divided(row) {
        const handle = handleOf(row);
        if (!handle) return false;
        return window.getComputedStyle(row).display === "grid"
            && window.getComputedStyle(handle).display !== "none";
    }

    /** The row's three tracks in pixels, as laid out, or null. */
    function tracks(row) {
        const text = window.getComputedStyle(row).gridTemplateColumns || "";
        const parts = text.trim().split(/\s+/).map(parseFloat);
        return parts.length === 3 && parts.every(Number.isFinite) ? parts : null;
    }

    /** The template that keeps each column's width when the two change
     *  sides, or null to leave the row's as it is: Forge's first template
     *  when its halves are equal; otherwise Forge's own `Npx 16px 1fr`,
     *  the first track as wide as the last one is now -- the column that is
     *  coming first. `laidOut` is the three tracks in pixels. Pure. */
    function swappedTemplate(current, original, laidOut) {
        const text = String(current || "").trim();
        const parts = text.split(/\s+/);
        if (text && text === String(original || "").trim() && parts.length === 3 && parts[0] === parts[2]) return null;
        if (!laidOut || laidOut.length !== 3 || !(laidOut[2] > 0)) return null;
        return Math.round(laidOut[2]) + "px " + GUTTER + "px 1fr";
    }

    /** The first track's width for a drag that started at `start` and has
     *  travelled `travel` pixels to the right, in a row `rowWidth` wide:
     *  Forge's own clamp, with the minimum of the column drawn first and of
     *  the one drawn last. Pure. */
    function dragWidth(start, travel, rowWidth, minFirst, minLast) {
        return Math.round(Math.max(Math.min(start + travel, rowWidth - minLast - GUTTER), minFirst));
    }

    /** Whether a touch at `now` on `tab`'s divider is the second of a double
     *  tap after `previous` (the last touch on a divider, or null). Pure. */
    function isDoubleTap(previous, tab, now) {
        return !!previous && previous.tab === tab && !previous.moved
            && now - previous.time >= 0 && now - previous.time <= DOUBLE_TAP_MS;
    }

    /** A column minimum Forge keeps on the row, or Forge's own default. */
    function minimum(value) {
        return typeof value === "number" && Number.isFinite(value) && value >= 0 ? value : MIN_COLUMN;
    }

    /** The root's attribute as the kept choices and the setting say, and every
     *  row whose columns change sides given the template that keeps their
     *  widths -- read before the attribute changes, written after, with
     *  nothing drawn in between. */
    function render() {
        const tabs = wanted();
        const plans = [];
        TABS.forEach((tab) => {
            if (shown(tab) === (tabs.indexOf(tab) >= 0)) return;
            const row = rowOf(tab);
            if (!row || !divided(row)) return;
            const template = swappedTemplate(row.style.gridTemplateColumns, row.style.originalGridTemplateColumns, tracks(row));
            if (template) plans.push([row, template]);
        });
        writeAttribute(tabs);
        plans.forEach((plan) => { plan[0].style.gridTemplateColumns = plan[1]; });
    }

    /** Put `tab`'s columns on the other sides, and keep it so. */
    function swap(tab) {
        if (TABS.indexOf(tab) < 0 || !enabled()) return false;
        const now = Date.now();
        if (now - state.lastSwap < SWAP_GUARD_MS) return false;
        state.lastSwap = now;
        keep(tab, !kept(tab));
        render();
        return true;
    }

    // -- the drag of a swapped row ---------------------------------------------- //

    function startDrag(tab, row, x, touch) {
        const laid = tracks(row);
        if (!laid) return false;
        endDrag();
        // Forge keeps its minimums on the row: the first is the minimum of the
        // row's first child in the document, the settings. Drawn swapped, the
        // results are first.
        const drag = {tab, row, x, touch, start: laid[0], width: Math.round(laid[0]),
                      minFirst: minimum(row.minRightColWidth), minLast: minimum(row.minLeftColWidth)};
        const move = (event) => {
            if (touch) {
                if (!event.touches || event.touches.length !== 1) return;
                moveDrag(event.touches[0].clientX);
            } else {
                moveDrag(event.clientX);
            }
        };
        const end = () => endDrag();
        drag.listeners = touch
            ? [["touchmove", move, {passive: true}], ["touchend", end, false], ["touchcancel", end, false]]
            : [["mousemove", move, false], ["mouseup", end, false]];
        drag.listeners.push(["blur", end, false]);
        drag.listeners.forEach((entry) => window.addEventListener(entry[0], entry[1], entry[2]));
        state.drag = drag;
        // Forge's own class for its drag: nothing on the page takes the
        // pointer meanwhile, and the Forge Assistant's fill waits for the end.
        if (document.body) document.body.classList.add("resizing");
        return true;
    }

    function moveDrag(x) {
        const drag = state.drag;
        if (!drag) return;
        if (state.tap && Math.abs(x - drag.x) > TAP_SLOP) state.tap.moved = true;
        const width = dragWidth(drag.start, x - drag.x, drag.row.offsetWidth, drag.minFirst, drag.minLast);
        if (width === drag.width) return;
        drag.width = width;
        drag.row.style.gridTemplateColumns = width + "px " + GUTTER + "px 1fr";
    }

    function endDrag() {
        const drag = state.drag;
        if (!drag) return;
        state.drag = null;
        drag.listeners.forEach((entry) => window.removeEventListener(entry[0], entry[1], entry[2]));
        if (document.body) document.body.classList.remove("resizing");
    }

    // -- the presses --------------------------------------------------------------- //

    /** The Forge Assistant's shell, when its panel is on the page. */
    function assistant() {
        const found = window.forgeAssistant;
        return found && found.shell && typeof found.shell === "object" ? found.shell : null;
    }

    /** Whether a press made now only puts the Forge Assistant's panel away:
     *  its panel open, with Tap Outside to Close on. Read at the press. */
    function closingPress() {
        const shell = assistant();
        const state = shell && shell.state;
        return !!(state && state.panelOpen && state.tapToClose);
    }

    /** Whether a double click's first click was the tap that put the Forge
     *  Assistant's panel away: the tap it recorded, at the pair's place and
     *  within a double-click's time. */
    function afterClosingTap(event) {
        const shell = assistant();
        const tap = shell && shell.lastTapOutside;
        if (!tap || tap.frame || !(typeof tap.at === "number")) return false;
        const since = Date.now() - tap.at;
        return since >= 0 && since <= CLOSING_TAP_MS
            && Math.abs(Number(event.clientX) - tap.x) <= CLOSING_TAP_PX
            && Math.abs(Number(event.clientY) - tap.y) <= CLOSING_TAP_PX;
    }

    /** After a double tap's swap the columns have moved under the finger, and
     *  the mouse events the browser makes of that touch -- a mousedown, a
     *  click, and Chromium's dblclick for the pair -- would land on whatever
     *  is there now: a picture in the gallery, a button among the settings.
     *  They are the tap's, and are taken. */
    function swallow(event) {
        if (Date.now() >= state.swallowUntil) return false;
        event.stopPropagation();
        event.preventDefault();
        return true;
    }

    /** A row this file draws: the divider's tab, the row laid out as Forge's
     *  grid with the divider showing. Null for anything else. */
    function pressed(event) {
        if (!enabled()) return null;
        const tab = tabOfHandle(event.target);
        if (!tab) return null;
        const row = rowOf(tab);
        return row && divided(row) ? {tab, row} : null;
    }

    function onMouseDown(event) {
        if (swallow(event) || event.button !== 0) return;
        const found = pressed(event);
        if (!found) return;
        // A press that only puts the assistant's panel away does nothing
        // else, whichever side the columns are on: no drag of this file's,
        // and Forge's listener on the divider does not hear it either. A
        // browser hands the page no mousedown for a press whose pointerdown
        // the assistant cancelled; this is for one that does.
        if (closingPress()) {
            event.stopPropagation();
            return;
        }
        if (!shown(found.tab)) return;
        // Swapped: the drag is this file's, and Forge's listener on the
        // handle never hears the press.
        event.stopPropagation();
        event.preventDefault();
        startDrag(found.tab, found.row, event.clientX, false);
    }

    function onDoubleClick(event) {
        if (swallow(event)) return;
        const found = pressed(event);
        if (!found) return;
        // Forge's double click put its first widths back; here it swaps --
        // unless its first click only put the assistant's panel away, when it
        // does neither.
        event.stopPropagation();
        event.preventDefault();
        if (afterClosingTap(event)) return;
        swap(found.tab);
    }

    /** While a first touch on a divider is down: whether it travels. */
    function watchTap() {
        if (state.tapWatch) return;
        const move = (event) => {
            const tap = state.tap;
            const touch = event.touches && event.touches[0];
            if (!tap || !touch) return;
            if (Math.abs(touch.clientX - tap.x) > TAP_SLOP || Math.abs(touch.clientY - tap.y) > TAP_SLOP) tap.moved = true;
        };
        const end = () => {
            window.removeEventListener("touchmove", move, {passive: true});
            window.removeEventListener("touchend", end, false);
            window.removeEventListener("touchcancel", cancel, false);
            state.tapWatch = null;
        };
        const cancel = () => {
            if (state.tap) state.tap.moved = true;
            end();
        };
        window.addEventListener("touchmove", move, {passive: true});
        window.addEventListener("touchend", end, false);
        window.addEventListener("touchcancel", cancel, false);
        state.tapWatch = end;
    }

    function onTouchStart(event) {
        if (!event.touches || event.touches.length !== 1) { state.tap = null; return; }
        const found = pressed(event);
        if (!found) { state.tap = null; return; }
        if (closingPress()) {
            // The touch that puts the assistant's panel away does nothing
            // else: no first tap of a double tap, no drag of this file's,
            // and Forge's listener on the divider never hears it -- no drag
            // of Forge's left running under the assistant's touchend, no
            // double tap of Forge's counted from it.
            state.tap = null;
            event.stopPropagation();
            return;
        }
        const touch = event.touches[0];
        const now = Date.now();
        if (isDoubleTap(state.tap, found.tab, now)) {
            state.tap = null;
            // The second touch of a double tap is this file's whichever side
            // the columns are on: Forge's own double tap would have put its
            // first widths back.
            event.stopPropagation();
            if (swap(found.tab)) state.swallowUntil = Date.now() + SWAP_GUARD_MS;
            return;
        }
        state.tap = {tab: found.tab, time: now, x: touch.clientX, y: touch.clientY, moved: false};
        watchTap();
        if (!shown(found.tab)) return;
        event.stopPropagation();
        startDrag(found.tab, found.row, touch.clientX, true);
    }

    function wire() {
        rememberEnabled();
        render();
    }

    function describe() {
        return TABS.map((tab) => {
            const row = rowOf(tab);
            return {tab, kept: kept(tab), shown: shown(tab), divided: !!(row && divided(row)),
                    template: row ? row.style.gridTemplateColumns : ""};
        });
    }

    window.mcColumns = {
        version: VERSION,
        OPTION, STORE, ENABLED_STORE, ATTRIBUTE, DOUBLE_TAP_MS, SWAP_GUARD_MS,
        swap, render, wire, describe, shown, kept, enabled,
        swappedTemplate, dragWidth, isDoubleTap,
    };

    // The first paint: the last choice, before Gradio has built anything.
    try { render(); } catch (error) { console.error("Model Chain: the columns could not be ordered", error); }
    window.addEventListener("mousedown", onMouseDown, true);
    window.addEventListener("dblclick", onDoubleClick, true);
    window.addEventListener("mouseup", swallow, true);
    window.addEventListener("click", swallow, true);
    window.addEventListener("touchstart", onTouchStart, {capture: true, passive: true});
    if (typeof onUiLoaded === "function") onUiLoaded(wire);
    if (typeof onOptionsChanged === "function") onOptionsChanged(wire);
})();
