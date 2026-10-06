// Forge Assistant -- the generation tabs' layout: the results column fills the
// window, and Txt2Img's settings column can be docked into the panel, where a
// field takes a drag only after a tap has engaged it.
//
// Two things, one file, because both are about the same two columns of the
// host's page and each has to know what the other did to them.
//
// THE FILL. Asked for: "the right column (generate, gallery, gallery buttons)
// does not scale all the way to fill the space. There is a gap at the bottom";
// and then, once it did, exactly how: "The gallery need to fill its area, only
// bounded by the screen border, generate / interrupt / cancel, progress bar
// (when in progress), and gallery buttons. Everything else should be off page.
// The image should scale up to fill the maximum allowed space. The padding
// above generate and below gallery buttons should be the same, giving view of
// gallery in perfect center ... All other content below should still be
// rendered, but off page and below the view, thus requiring a scroll to see
// it."
//
// So what is kept in view is the column from its first control -- Generate,
// which Lobe's split previewer moves into the gallery's container, with
// Interrupt and Skip in the same box, and the progress bar above it while a
// generation runs -- down to the gallery's buttons. The gap from the top of the
// view to the first of those is measured, and the same gap is left under the
// buttons; the gallery is everything in between, in pixels. The generation's
// infotext, its log and anything else after the buttons is laid out as ever,
// below the view: the page scrolls to it. The first build fitted the whole
// column into the window instead, and every infotext took its height out of
// the gallery -- half a window of picture under a table of settings.
//
// The top of the view is the scroller's own top edge -- focus mode's root -- or,
// when the page itself scrolls, the bottom of whatever holds the top of the
// window over the column: Lobe's header is sticky. A position kept as a share
// of the window ("85vh" in the person's user.css) cannot be right both in focus
// and out of it, which is why this is measured at all; "heights that have to
// be exact are measured and written as pixels" was learned the hard way next
// door (Mini Paint NEO's WanGP frame).
//
// The pixels travel as one custom property on the results column, and the
// stylesheet applies them with `!important` on two ids: a user.css rule such as
// `#txt2img_gallery_container { height: 85vh !important }` is one id, so it
// loses to them whatever order the sheets load in, and the person's file needs
// no change. Nothing else in the column is touched.
//
// THE DOCK. The flyout's third state, beside the conversation and the tab bar:
// Txt2Img's left column -- prompts and every setting -- taken out of the page
// and shown, scrolling, in the panel, so the gallery has the whole width.
//
// The column is never moved in the DOM. Forge finds the prompt boxes through
// `gradioApp()`, the <gradio-app> element, and so do this extension's own
// scripts and every other extension's; the panel lives under <body>, outside
// it. So the column stays exactly where it is and is *drawn* over the panel's
// empty body: `position: fixed`, at the pixels of a placeholder in the panel,
// with the grid row it came from told to give its track to the gallery. The
// panel is see-through over the placeholder and lets presses through it. The
// column is drawn under the panel rather than above it because focus mode's
// root is a stacking context of its own (layer 1100) below the panel's (1200):
// nothing inside it can be drawn above the panel, so the panel gets out of the
// way instead, and the same arrangement then works with focus on and off.
//
// Undocking puts back exactly the inline values it replaced, and takes its two
// classes off; Forge's own resize handle then lays the row out as it was.

(function () {
    "use strict";

    const NS = (window.forgeAssistant = window.forgeAssistant || {});

    // -- the fill ------------------------------------------------------------ //

    const FILL_TABS = ["txt2img", "img2img"];
    const FILL_MARK = "data-forge-assistant-fill";
    const FILL_VAR = "--forge-assistant-fill-gallery";
    // The room left under the gallery's buttons, so what follows them starts
    // at the bottom of the view: the gap under the buttons is the gap above
    // Generate, and nothing shows in it.
    const FILL_AFTER = "--forge-assistant-fill-after";
    // The assistant's own root: never what holds the top of the window.
    const ASSISTANT_ROOT = "forge-assistant-root";
    // Below this the gallery is not a gallery, and a window that short is better
    // served by the page scrolling than by a strip.
    const FILL_FLOOR = 240;

    function app() {
        return (typeof gradioApp === "function" ? gradioApp() : null) || document;
    }

    function byId(id) {
        const root = app();
        let found = null;
        try {
            found = root.querySelector ? root.querySelector("#" + id) : null;
        } catch (error) {
            found = null;
        }
        return found || document.getElementById(id);
    }

    function styleOf(node) {
        try {
            return window.getComputedStyle ? window.getComputedStyle(node) : null;
        } catch (error) {
            return null;
        }
    }

    function px(value) {
        const found = parseFloat(value);
        return Number.isFinite(found) ? found : 0;
    }

    function laidOut(node) {
        if (!node || !node.getBoundingClientRect) return false;
        const box = node.getBoundingClientRect();
        return box.width > 0 && box.height > 0;
    }

    /** The nearest ancestor that scrolls up and down, or null for the page. */
    function scrollerOf(node) {
        let walk = node && node.parentElement;
        while (walk && walk !== document.body && walk !== document.documentElement) {
            const style = styleOf(walk);
            const overflow = style ? (style.overflowY || "visible") : "visible";
            if (overflow === "auto" || overflow === "scroll" || overflow === "overlay") return walk;
            walk = walk.parentElement;
        }
        return null;
    }

    /** The child of `row` that holds `node`, or null if `row` does not. */
    function childOf(row, node) {
        let walk = node;
        while (walk && walk.parentElement !== row) walk = walk.parentElement;
        return walk || null;
    }

    function Fill() {
        this.frame = 0;
        this.observer = null;
        this.watched = new Set();
        this.started = false;
        this.disposers = [];
    }

    /** Where the view starts, over a column the page itself scrolls: the top
     *  of the window, or the bottom of what is held over it there -- a sticky or
     *  fixed header (Lobe's), found by asking the page what is drawn at the top
     *  of the window above the column. This extension's own panel is never the
     *  answer, and neither is anything that holds the column itself. */
    function coveredTop(results, top) {
        if (typeof document.elementsFromPoint !== "function") return top;
        const box = results.getBoundingClientRect();
        const x = Math.max(0, Math.min((window.innerWidth || 0) - 1, box.left + box.width / 2));
        let covered = top;
        // A second bar under the first is looked for too; three is plenty.
        for (let round = 0; round < 3; round += 1) {
            let found = covered;
            let hits = [];
            try {
                hits = document.elementsFromPoint(x, covered + 1);
            } catch (error) {
                hits = [];
            }
            hits.forEach((hit) => {
                if (!hit || (hit.closest && hit.closest("#" + ASSISTANT_ROOT))) return;
                let walk = hit;
                while (walk && walk !== document.body && walk !== document.documentElement) {
                    const style = styleOf(walk);
                    const position = style ? style.position : "";
                    if (position === "fixed" || position === "sticky") {
                        if (!walk.contains(results)) {
                            found = Math.max(found, walk.getBoundingClientRect().bottom);
                        }
                        break;
                    }
                    walk = walk.parentElement;
                }
            });
            if (!(found > covered + 0.5)) break;
            covered = found;
        }
        return covered;
    }

    /** Laid out and in the column's flow: an overlay (a theme's absolute
     *  progress bar, Forge's live preview) moves nothing and is not counted. */
    function inFlow(node) {
        if (!laidOut(node)) return false;
        const style = styleOf(node);
        const position = style ? style.position : "";
        return position !== "absolute" && position !== "fixed";
    }

    /** The part of the column kept in view, as it is drawn now: from the first
     *  of Generate's box (with Interrupt and Skip), the progress bar while a
     *  generation runs, the gallery's container and the gallery, down to the
     *  bottom of the gallery's buttons. */
    function keptBox(tab, results, gallery) {
        const tops = [];
        const take = (node) => {
            if (node && node !== results && results.contains(node) && inFlow(node)) {
                tops.push(node.getBoundingClientRect().top);
            }
        };
        take(results.querySelector("[id$='_generate_box']"));
        Array.prototype.forEach.call(results.querySelectorAll(".progressDiv"), take);
        take(byId(tab + "_gallery_container"));
        const galleryBox = gallery.getBoundingClientRect();
        tops.push(galleryBox.top);
        let bottom = galleryBox.bottom;
        let last = gallery;
        const buttons = byId("image_buttons_" + tab);
        if (buttons && results.contains(buttons) && inFlow(buttons)) {
            bottom = Math.max(bottom, buttons.getBoundingClientRect().bottom);
            last = buttons;
        }
        return {top: Math.min.apply(null, tops), bottom, last};
    }

    /** The top of the first thing laid out after `node` in the column, or
     *  null: its following siblings, then its parent's, up to the column. */
    function nextTop(node, column) {
        let walk = node;
        while (walk && walk !== column) {
            for (let sibling = walk.nextElementSibling; sibling;
                sibling = sibling.nextElementSibling) {
                if (inFlow(sibling)) return sibling.getBoundingClientRect().top;
            }
            walk = walk.parentElement;
        }
        return null;
    }

    /** What the gallery's height should be, in pixels, or null to leave the
     *  column alone -- with the reason, for `explain`. Reads the layout and
     *  writes nothing.
     *
     *  The column is measured where it rests: its place in the row with the
     *  scroller at its start, or, when it is sticky, where it is held if that
     *  is lower. From there, the gap between the top of the view and the first
     *  thing kept in view is left again under the last (`keptBox`), and the
     *  gallery takes everything between them that the rest of the kept part
     *  does not -- so a progress bar arriving above Generate shortens the
     *  gallery rather than pushing the buttons off the screen, and the
     *  infotext arriving after the buttons changes nothing. */
    Fill.prototype.measure = function (tab) {
        const results = byId(tab + "_results");
        const gallery = byId(tab + "_gallery");
        if (!results || !gallery) return {height: null, reason: "no results column"};
        if (!laidOut(results) || !laidOut(gallery)) {
            return {height: null, reason: "not on screen", keep: true};
        }
        const row = results.parentElement;
        const resultsBox = results.getBoundingClientRect();
        // Beside the settings, or alone in the row (docked, or a tab without
        // them). Stacked under them -- a phone, or Forge's own narrow layout --
        // the column is one more thing in a page that scrolls, and a window's
        // worth of gallery there would be a gallery to scroll past.
        const left = row && row.firstElementChild !== results ? row.firstElementChild : null;
        if (left && laidOut(left) && !docked(left)) {
            const leftBox = left.getBoundingClientRect();
            if (leftBox.right > resultsBox.left + 1) {
                return {height: null, reason: "stacked under the settings"};
            }
        }
        // The view: what scrolls the column, from its top edge to its bottom
        // edge -- its padding is part of the gap, on both sides alike.
        const scroller = scrollerOf(results);
        let viewTop = 0;
        let viewBottom = 0;
        let scrolled = 0;
        if (scroller) {
            const box = scroller.getBoundingClientRect();
            viewTop = box.top + (scroller.clientTop || 0);
            viewBottom = viewTop + scroller.clientHeight;
            scrolled = scroller.scrollTop || 0;
        } else {
            const view = NS.viewport ? NS.viewport()
                : {top: 0, height: window.innerHeight};
            const top = view.top || 0;
            viewBottom = top + (view.height || window.innerHeight);
            viewTop = coveredTop(results, top);
            scrolled = window.scrollY || window.pageYOffset || 0;
        }
        // Where the column would be in the row, at this scroll and with the
        // scroller at its start.
        const rowBox = row ? row.getBoundingClientRect() : resultsBox;
        const rowStyle = row ? styleOf(row) : null;
        const style = styleOf(results);
        const natural = rowBox.top + px(rowStyle && rowStyle.paddingTop)
            + px(rowStyle && rowStyle.borderTopWidth) + px(style && style.marginTop);
        let top = natural + scrolled;
        // A sticky column drawn lower than that is being held where it sticks,
        // and where it is drawn is that place -- read from the layout rather
        // than worked out from `top`, because browsers do not agree on whether
        // a scroller's padding counts (Chromium adds it). Drawn where it would
        // be, it is not held, and it sticks no lower than its place at rest.
        if (style && style.position === "sticky" && resultsBox.top > natural + 0.5) {
            top = Math.max(top, resultsBox.top);
        }
        const kept = keptBox(tab, results, gallery);
        const galleryBox = gallery.getBoundingClientRect();
        const above = galleryBox.top - kept.top;
        const below = kept.bottom - galleryBox.bottom;
        // The first thing kept in view, where the column rests.
        const first = kept.top - resultsBox.top + top;
        const gap = Math.max(0, first - viewTop);
        const fits = Math.floor(viewBottom - gap - first - above - below);
        if (!(fits > 0)) return {height: null, reason: "no room"};
        const drawn = Math.max(FILL_FLOOR, fits);
        // A content-box gallery is drawn its height plus its own padding and
        // border; the height worked out above is the whole box.
        let height = drawn;
        const own = styleOf(gallery);
        if (own && own.boxSizing === "content-box") {
            height -= px(own.paddingTop) + px(own.paddingBottom)
                + px(own.borderTopWidth) + px(own.borderBottomWidth);
        }
        // What follows the buttons starts at the bottom of the view: the room
        // between them is made up from where the buttons will end -- the
        // drawn height, rounded as it is drawn -- counting the room there is
        // already and leaving out what this wrote last time.
        let after = 0;
        const next = nextTop(kept.last, results);
        if (next !== null) {
            const written = results.hasAttribute(FILL_MARK)
                ? px(results.style.getPropertyValue(FILL_AFTER)) : 0;
            const between = next - kept.bottom - written;
            const end = first + above + drawn + below;
            after = Math.max(0, Math.ceil(viewBottom - end - between));
        }
        return {height, after, reason: "", gap: Math.round(gap)};
    };

    /** Measure one tab and write what it found, or take the fill off. */
    Fill.prototype.apply = function (tab) {
        const results = byId(tab + "_results");
        if (!results) return null;
        this.watch(results);
        const found = this.measure(tab);
        if (found.height === null) {
            // A tab that is not on screen keeps what it had: it is measured
            // again the moment it is shown, because showing it resizes it.
            if (!found.keep && results.hasAttribute(FILL_MARK)) {
                results.removeAttribute(FILL_MARK);
                results.style.removeProperty(FILL_VAR);
                results.style.removeProperty(FILL_AFTER);
            }
            return found;
        }
        // Written only when they changed: the column is watched, these writes
        // resize it, and an answer to its own write must find nothing to do.
        const value = found.height + "px";
        if (results.style.getPropertyValue(FILL_VAR) !== value) {
            results.style.setProperty(FILL_VAR, value);
        }
        const after = found.after + "px";
        if (results.style.getPropertyValue(FILL_AFTER) !== after) {
            results.style.setProperty(FILL_AFTER, after);
        }
        if (!results.hasAttribute(FILL_MARK)) results.setAttribute(FILL_MARK, "");
        return found;
    };

    Fill.prototype.watch = function (node) {
        if (!node || this.watched.has(node) || !this.observer) return;
        this.watched.add(node);
        this.observer.observe(node);
    };

    /** Measure again on the next frame -- once, however many asked. */
    Fill.prototype.refresh = function () {
        if (!this.started || this.frame) return;
        this.frame = window.requestAnimationFrame(() => {
            this.frame = 0;
            this.now();
        });
    };

    Fill.prototype.now = function () {
        FILL_TABS.forEach((tab) => {
            try {
                this.apply(tab);
            } catch (error) {
                console.error("Forge Assistant: the results column could not be fitted", error);
            }
        });
    };

    /** What `measure` sees, for the console: `forgeAssistant.fill.explain()`. */
    Fill.prototype.explain = function () {
        return FILL_TABS.map((tab) => Object.assign({tab}, this.measure(tab)));
    };

    Fill.prototype.start = function () {
        if (this.started) return;
        this.started = true;
        // The column changes size when it is shown (a tab switch is display
        // none to block), when its content changes (an infotext, the preview
        // opening) and when its row's width changes (the dock, the resize
        // handle); each of those is a reason to measure.
        if (typeof ResizeObserver === "function") {
            this.observer = new ResizeObserver(() => this.refresh());
        }
        const again = () => this.refresh();
        window.addEventListener("resize", again);
        this.disposers.push(() => window.removeEventListener("resize", again));
        if (window.visualViewport) {
            window.visualViewport.addEventListener("resize", again);
            this.disposers.push(() => window.visualViewport.removeEventListener("resize", again));
        }
        this.now();
    };

    Fill.prototype.stop = function () {
        if (this.frame) window.cancelAnimationFrame(this.frame);
        this.frame = 0;
        if (this.observer) this.observer.disconnect();
        this.watched.clear();
        this.disposers.splice(0).forEach((dispose) => dispose());
        FILL_TABS.forEach((tab) => {
            const results = byId(tab + "_results");
            if (!results) return;
            results.removeAttribute(FILL_MARK);
            results.style.removeProperty(FILL_VAR);
            results.style.removeProperty(FILL_AFTER);
        });
        this.started = false;
    };

    // -- the guard ----------------------------------------------------------- //
    //
    // Asked for once the column was in the panel: "when i try to scroll i might
    // scroll a text box, or move a slider. I would like scroll input fields or
    // sliding sliders etc, it should require a selection tap. So by default, the
    // entire content column is scrollable in the flyout menu, but once i want to
    // say type, i have to tap the input field first to select it, then i am in
    // it, and i can type. Same with slider ... I would like the component to
    // throw some sort of highlight state outline so i know i'm in".
    //
    // So while the column is docked, every control that takes a drag or a wheel
    // for itself -- a text box (it scrolls its own text), a number box, a
    // slider, the compact spatial canvas -- is given no presses at all by the
    // stylesheet (`pointer-events: none`). A drag that starts on one is then a
    // drag on the block around it, and the column scrolls. A tap (a `click`,
    // which a browser does not fire after a scroll) on the block *engages* it:
    // the block gets `forge-assistant-engaged`, an outline, and its controls get
    // their presses back; a text box is focused, with the caret at its end, so
    // the keyboard comes up. A tap anywhere outside it, or Escape, lets it go.
    // Dropdowns, checkboxes, radios and buttons are left alone: a drag across
    // them scrolls already, and they act only on a tap.
    //
    // The selector is the stylesheet's too (`.forge-assistant-guarded` in
    // style.css); the two lists must name the same controls.

    const GUARD_CLASS = "forge-assistant-guarded";
    const ENGAGED_CLASS = "forge-assistant-engaged";
    const GUARDED = [
        "textarea",
        "input[type=text]",
        "input:not([type])",
        "input[type=search]",
        "input[type=number]",
        "input[type=range]",
        "[contenteditable=true]",
        ".mc-krea-spatial-compact-frame",
    ].map((selector) => selector + ":not(.gradio-dropdown *)").join(", ");
    // Controls a tap on which should bring a keyboard up.
    const TYPED = "textarea, input[type=text], input:not([type]), input[type=search], " +
        "input[type=number], [contenteditable=true]";
    const COMPACT = ".mc-krea-spatial-compact-frame";

    function shows(node) {
        if (!node || !node.getBoundingClientRect) return false;
        const rect = node.getBoundingClientRect();
        return rect.width > 0 && rect.height > 0;
    }

    function within(rect, x, y) {
        return x >= rect.left && x <= rect.right && y >= rect.top && y <= rect.bottom;
    }

    function Guard() {
        this.column = null;
        this.engaged = null;
        this.onClick = this.click.bind(this);
        this.onKey = this.key.bind(this);
        this.onFocus = this.focus.bind(this);
    }

    /** The block a guarded control answers for: the compact canvas is its own,
     *  every other control its Gradio block (a slider's label, track and
     *  number box are one block, and a tap on any of them engages all three). */
    Guard.prototype.unitOf = function (control) {
        if (!control || !control.closest) return null;
        const compact = control.closest(COMPACT);
        if (compact) return compact;
        const block = control.closest(".block");
        return block && this.column && this.column.contains(block) ? block : control;
    };

    /** The guarded controls of `unit`, as drawn. */
    Guard.prototype.controls = function (unit) {
        const found = unit.matches && unit.matches(GUARDED) ? [unit] : [];
        unit.querySelectorAll(GUARDED).forEach((node) => found.push(node));
        return found.filter(shows);
    };

    /** The unit a tap at (x, y) on `target` engages, or null. Guarded controls
     *  take no presses, so the target is whatever is around them: the block a
     *  control answers for, or something inside it (a label), or a block that
     *  holds several -- an accordion -- where only the one under the point is
     *  meant. */
    Guard.prototype.unitAt = function (target, x, y) {
        if (!target || !target.closest || !this.column.contains(target)) return null;
        const compact = target.closest(COMPACT);
        if (compact) return compact;
        const block = target.closest(".block");
        const scope = block && this.column.contains(block) ? block : this.column;
        const controls = this.controls(scope);
        const hit = controls.find((node) => within(node.getBoundingClientRect(), x, y));
        if (hit) return this.unitOf(hit);
        // A tap beside the controls -- a label, a slider's ends -- engages the
        // block when every control in it is that block's own.
        if (block && controls.length && controls.every((node) => this.unitOf(node) === block)) {
            return block;
        }
        return null;
    };

    Guard.prototype.attach = function (column) {
        if (this.column === column) return;
        this.detach();
        if (!column) return;
        this.column = column;
        column.classList.add(GUARD_CLASS);
        // Capture, on the document: a tap outside the column lets go as well,
        // and this has to decide before the control's own handlers run.
        document.addEventListener("click", this.onClick, true);
        document.addEventListener("keydown", this.onKey, true);
        column.addEventListener("focusin", this.onFocus);
    };

    Guard.prototype.detach = function () {
        const column = this.column;
        if (!column) return;
        this.release();
        column.classList.remove(GUARD_CLASS);
        document.removeEventListener("click", this.onClick, true);
        document.removeEventListener("keydown", this.onKey, true);
        column.removeEventListener("focusin", this.onFocus);
        this.column = null;
    };

    /** Engage `unit`: its outline, its presses, and -- for a text box tapped,
     *  or a block with nothing to slide -- the keyboard. */
    Guard.prototype.engage = function (unit, x, y) {
        if (this.engaged === unit) return;
        this.release();
        this.engaged = unit;
        unit.classList.add(ENGAGED_CLASS);
        if (x === undefined) return;
        const controls = this.controls(unit);
        const hit = controls.find((node) => within(node.getBoundingClientRect(), x, y));
        const slides = controls.some((node) => node.matches("input[type=range]"));
        let chosen = hit && hit.matches(TYPED) ? hit : null;
        if (!chosen && !hit && !slides) chosen = controls.find((node) => node.matches(TYPED));
        if (!chosen && hit && hit.matches("input[type=range]")) chosen = hit;
        if (!chosen) return;
        try {
            chosen.focus({preventScroll: true});
        } catch (error) {
            chosen.focus();
        }
        if (chosen.matches("textarea, input[type=text], input:not([type]), input[type=search]")) {
            try {
                const end = String(chosen.value || "").length;
                chosen.setSelectionRange(end, end);
            } catch (error) {
                // Not every input takes a selection; the focus is what matters.
            }
        }
    };

    /** Let the engaged block go, and its keyboard with it. */
    Guard.prototype.release = function () {
        const unit = this.engaged;
        if (!unit) return false;
        this.engaged = null;
        unit.classList.remove(ENGAGED_CLASS);
        const active = document.activeElement;
        if (active && active !== document.body && unit.contains(active) && active.blur) {
            active.blur();
        }
        return true;
    };

    Guard.prototype.click = function (event) {
        const column = this.column;
        if (!column) return;
        const target = event.target;
        if (this.engaged && !this.engaged.isConnected) this.engaged = null;
        if (this.engaged && target && this.engaged.contains(target)) return;
        const unit = this.unitAt(target, event.clientX, event.clientY);
        if (!unit) {
            this.release();
            return;
        }
        // The tap is the engaging one: a label's own focus, or anything else
        // the press would have done, is this function's to decide.
        event.preventDefault();
        this.engage(unit, event.clientX, event.clientY);
    };

    Guard.prototype.key = function (event) {
        if (event.key === "Escape" && this.release()) event.stopPropagation();
    };

    /** A control reached from the keyboard (Tab) is engaged, outline and all:
     *  it is as much "in it" as a tapped one. */
    Guard.prototype.focus = function (event) {
        const target = event.target;
        if (!target || !target.matches || !target.matches(GUARDED)) return;
        const unit = this.unitOf(target);
        if (unit && unit !== this.engaged) this.engage(unit);
    };

    // -- the dock ------------------------------------------------------------ //

    const DOCK_WORKSPACE = "tab_txt2img";
    const DOCK_SETTINGS = "txt2img_settings";
    const DOCK_COLUMN = "forge-assistant-docked-column";
    const DOCK_ROW = "forge-assistant-docked-row";
    // The inline properties the dock writes, each put back as it was found.
    const DOCK_PROPS = ["left", "top", "width", "height", "visibility"];

    function docked(node) {
        return !!(node && node.classList && node.classList.contains(DOCK_COLUMN));
    }

    function Dock() {
        this.active = null;
        this.guard = new Guard();
    }

    /** The column and the row it sits in: the row's child that holds
     *  `#txt2img_settings` -- the column itself, or the accordion Forge wraps
     *  it in when "Open for Settings" is on. Null when the page has none. */
    Dock.prototype.find = function () {
        const settings = byId(DOCK_SETTINGS);
        if (!settings) return null;
        const row = settings.closest ? settings.closest(".resize-handle-row") : null;
        const column = row ? childOf(row, settings) : settings;
        return column ? {row, column} : null;
    };

    /** Can the dock be offered on this workspace? */
    Dock.prototype.available = function (workspaceId) {
        return workspaceId === DOCK_WORKSPACE && !!this.find();
    };

    /** Draw the column in `rect` (the panel's placeholder, in the viewport's
     *  pixels). A rect with no area -- the placeholder hidden under a menu --
     *  keeps the column docked and out of sight. */
    Dock.prototype.place = function (rect) {
        if (!this.active) {
            const found = this.find();
            if (!found) return false;
            const saved = DOCK_PROPS.map((name) => ({
                name,
                value: found.column.style.getPropertyValue(name),
                priority: found.column.style.getPropertyPriority(name),
            }));
            this.active = Object.assign({saved}, found);
            found.column.classList.add(DOCK_COLUMN);
            if (found.row) found.row.classList.add(DOCK_ROW);
            this.guard.attach(found.column);
            if (NS.fill) NS.fill.refresh();
        }
        const style = this.active.column.style;
        if (!rect || !(rect.width > 0) || !(rect.height > 0)) {
            this.guard.release();
            style.setProperty("visibility", "hidden", "important");
            return true;
        }
        const write = (left, top) => {
            style.setProperty("left", Math.round(left) + "px", "important");
            style.setProperty("top", Math.round(top) + "px", "important");
        };
        style.setProperty("width", Math.round(rect.width) + "px", "important");
        style.setProperty("height", Math.round(rect.height) + "px", "important");
        style.setProperty("visibility", "visible", "important");
        write(rect.left, rect.top);
        // `position: fixed` resolves against an ancestor with a transform, a
        // filter or containment instead of the window (focus mode checks for
        // the same trap). Whatever such an ancestor adds, the box is now off
        // by exactly that; it is taken back once, from where the box landed.
        const landed = this.active.column.getBoundingClientRect();
        const dx = landed.left - rect.left;
        const dy = landed.top - rect.top;
        if (Math.abs(dx) > 0.5 || Math.abs(dy) > 0.5) write(rect.left - dx, rect.top - dy);
        return true;
    };

    /** Back in the page, exactly as it was. */
    Dock.prototype.release = function () {
        const active = this.active;
        if (!active) return false;
        this.active = null;
        this.guard.detach();
        const style = active.column.style;
        active.saved.forEach((entry) => {
            if (entry.value) style.setProperty(entry.name, entry.value, entry.priority || "");
            else style.removeProperty(entry.name);
        });
        active.column.classList.remove(DOCK_COLUMN);
        if (active.row) active.row.classList.remove(DOCK_ROW);
        if (NS.fill) NS.fill.refresh();
        return true;
    };

    Dock.prototype.isDocked = function () {
        return !!this.active;
    };

    NS.Fill = Fill;
    NS.Dock = Dock;
    NS.Guard = Guard;
    NS.GUARDED = GUARDED;
    NS.fill = new Fill();
    NS.dock = new Dock();
    NS.DOCK_WORKSPACE = DOCK_WORKSPACE;

    const start = () => {
        try {
            NS.fill.start();
        } catch (error) {
            console.error("Forge Assistant: the results column fill could not start", error);
        }
    };
    if (typeof onUiLoaded === "function") onUiLoaded(start);
    else if (document.readyState !== "loading") start();
    else document.addEventListener("DOMContentLoaded", start);
})();
