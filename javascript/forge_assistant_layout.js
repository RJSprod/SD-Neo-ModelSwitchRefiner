// Forge Assistant -- the generation tabs' layout: the results column fills the
// window, and Txt2Img's settings column can be docked into the panel.
//
// Two things, one file, because both are about the same two columns of the
// host's page and each has to know what the other did to them.
//
// THE FILL. Asked for: "the right column (generate, gallery, gallery buttons)
// does not scale all the way to fill the space. There is a gap at the bottom
// ... The right column should not scroll, it should just fill the space." The
// person's own user.css had tried, with `85vh` on the gallery's container: a
// height taken from the window, not from what is left of it under the column's
// top. Outside focus that happened to land near the bottom; in focus the header
// is gone, the column starts higher, and 85vh stops a long way short. No single
// viewport fraction is right in both, so the gap is measured: the space from
// where the column rests to the bottom of whatever scrolls it, less everything
// in the column that is not the gallery, is the gallery's height, written in
// pixels. "Heights that have to be exact are measured and written as pixels"
// was learned the hard way next door (Mini Paint NEO's WanGP frame); a
// percentage inside Gradio's wrappers resolves to auto.
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
    // Under a column nothing pads -- the page itself scrolls -- this much is left
    // below it, the same as focus mode's root pads its bottom edge.
    const FILL_MARGIN = 8;
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

    /** What the gallery's height should be, in pixels, or null to leave the
     *  column alone -- with the reason, for `explain`. Reads the layout and
     *  writes nothing.
     *
     *  The column's top is where it rests: its place in the row with the
     *  scroller at its start, or, when it is sticky, where it sticks if that is
     *  lower. Its
     *  bottom is the scroller's inner bottom edge -- its padding's top, the
     *  same margin focus mode gives the top -- or the window's less
     *  FILL_MARGIN. Everything between the column's top and the gallery, and
     *  between the gallery and the last thing in the column, is measured as it
     *  is, so a generation's infotext arriving under the buttons shrinks the
     *  gallery rather than pushing the buttons off the screen. */
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
        const scroller = scrollerOf(results);
        let bottom = 0;
        let scrolled = 0;
        if (scroller) {
            const box = scroller.getBoundingClientRect();
            const style = styleOf(scroller);
            bottom = box.top + (scroller.clientTop || 0) + scroller.clientHeight
                - px(style && style.paddingBottom);
            scrolled = scroller.scrollTop || 0;
        } else {
            const view = NS.viewport ? NS.viewport() : {height: window.innerHeight};
            bottom = (view.height || window.innerHeight) - FILL_MARGIN;
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
        const galleryBox = gallery.getBoundingClientRect();
        const above = galleryBox.top - resultsBox.top;
        // The last thing in the column, not the column's own bottom: a column
        // stretched by its row would otherwise count its stretch as content,
        // and every pass would take that much more off the gallery.
        let last = galleryBox.bottom;
        Array.prototype.forEach.call(results.children || [], (child) => {
            if (!laidOut(child)) return;
            const box = child.getBoundingClientRect();
            const childStyle = styleOf(child);
            last = Math.max(last, box.bottom + px(childStyle && childStyle.marginBottom));
        });
        const below = last - galleryBox.bottom + px(style && style.paddingBottom)
            + px(style && style.borderBottomWidth);
        let height = Math.floor(bottom - top - above - below);
        // A content-box gallery is drawn its height plus its own padding and
        // border; the height above is the whole box.
        const own = styleOf(gallery);
        if (own && own.boxSizing === "content-box") {
            height -= px(own.paddingTop) + px(own.paddingBottom)
                + px(own.borderTopWidth) + px(own.borderBottomWidth);
        }
        if (!(height > 0)) return {height: null, reason: "no room"};
        return {height: Math.max(FILL_FLOOR, height), reason: ""};
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
            }
            return found;
        }
        const value = found.height + "px";
        // Written only when it changed: the column is watched, this write
        // resizes it, and an answer to its own write must find nothing to do.
        if (results.style.getPropertyValue(FILL_VAR) !== value) {
            results.style.setProperty(FILL_VAR, value);
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
        });
        this.started = false;
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
            if (NS.fill) NS.fill.refresh();
        }
        const style = this.active.column.style;
        if (!rect || !(rect.width > 0) || !(rect.height > 0)) {
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
