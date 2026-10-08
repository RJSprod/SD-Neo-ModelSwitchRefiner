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
    // The grid view (the gallery with no picture open): its scroller's height,
    // the height of one row of thumbnails, and the scroller's top padding, so a
    // row snaps to where the first one starts.
    const FILL_GRID = "--forge-assistant-fill-grid";
    const FILL_ROW = "--forge-assistant-fill-row";
    const FILL_GRID_PAD = "--forge-assistant-fill-grid-pad";
    const ROWS_MARK = "data-forge-assistant-rows";
    const FILL_PROPS = [FILL_VAR, FILL_AFTER, FILL_GRID, FILL_ROW, FILL_GRID_PAD];
    // The space between the kept things when the gallery's buttons are not
    // there to measure it from.
    const SPACE_FALLBACK = 8;
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

    /** Forge's divider drag: `body.resizing`, under which its stylesheet
     *  gives every element `pointer-events: none`, so the page answers
     *  `elementsFromPoint` with nothing -- no header, no ceiling. */
    function dragging() {
        return !!(document.body && document.body.classList
                  && document.body.classList.contains("resizing"));
    }

    function Fill() {
        this.frame = 0;
        this.observer = null;
        this.watched = new Set();
        this.started = false;
        this.disposers = [];
        // A measure asked for during Forge's divider drag, taken when it ends.
        this.deferred = false;
        // Per tab, the margins written to space the kept things evenly: each
        // node with the inline margin-top it had before, put back when the
        // column is measured (so it is read as the theme draws it) and when the
        // fill comes off.
        this.spaced = {};
    }

    function putMargin(node, saved) {
        if (saved.value) node.style.setProperty("margin-top", saved.value, saved.priority);
        else node.style.removeProperty("margin-top");
    }

    function ownMargin(node) {
        return {value: node.style.getPropertyValue("margin-top"),
                priority: node.style.getPropertyPriority("margin-top")};
    }

    /** The grid view (no picture open): its scroller fitted to the bottom of
     *  the gallery as it will be drawn, and the height of a row of thumbnails
     *  that puts a whole number of rows in it -- as many as come nearest the
     *  pictures' own shape at the grid's column width, so a picture is cut by
     *  neither the gallery's edge nor its row. Null when there is no grid;
     *  no row until a thumbnail's picture has loaded. */
    function gridOf(gallery, galleryBox, drawn) {
        const wrap = gallery.querySelector(".grid-wrap");
        if (!wrap || !laidOut(wrap)) return null;
        const own = styleOf(gallery);
        const box = wrap.getBoundingClientRect();
        const outer = Math.floor(drawn - (box.top - galleryBox.top)
            - px(own && own.paddingBottom) - px(own && own.borderBottomWidth));
        if (!(outer > 0)) return null;
        const style = styleOf(wrap);
        const frame = px(style && style.paddingTop) + px(style && style.paddingBottom)
            + px(style && style.borderTopWidth) + px(style && style.borderBottomWidth);
        const height = style && style.boxSizing === "content-box" ? outer - frame : outer;
        const found = {height, row: null, rows: 0, pad: px(style && style.paddingTop)};
        const container = wrap.querySelector(".grid-container");
        const thumb = container && container.querySelector(".thumbnail-item");
        const picture = thumb && thumb.querySelector("img");
        if (!thumb || !laidOut(thumb) || !picture || !(picture.naturalWidth > 0)) return found;
        const room = outer - frame;
        const gap = px(styleOf(container) && styleOf(container).rowGap);
        const natural = thumb.getBoundingClientRect().width
            * picture.naturalHeight / picture.naturalWidth;
        if (!(room > 0) || !(natural > 0)) return found;
        const rows = Math.max(1, Math.round((room + gap) / (natural + gap)));
        found.rows = rows;
        found.row = Math.floor((room - (rows - 1) * gap) / rows * 100) / 100;
        return found;
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
                        const box = walk.getBoundingClientRect();
                        // A header is a bar. Something fixed over most of the
                        // window -- a lightbox, a dialog -- is not a header,
                        // and a column under it is not shorter for it.
                        if (!walk.contains(results)
                            && box.height <= (window.innerHeight || 0) / 2) {
                            found = Math.max(found, box.bottom);
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
     *  bottom of the gallery's buttons. `anchor` is the one whose top is that
     *  first top (the outermost, on a tie), and `leaves` are the things spaced
     *  evenly -- Generate, the progress bar, the gallery, the buttons -- in the
     *  order they are drawn. */
    function keptBox(tab, results, gallery) {
        const candidates = [];
        const take = (node) => {
            if (node && node !== results && results.contains(node) && inFlow(node)) {
                candidates.push(node);
                return node;
            }
            return null;
        };
        const leaves = [];
        const generate = take(results.querySelector("[id$='_generate_box']"));
        if (generate) leaves.push(generate);
        Array.prototype.forEach.call(results.querySelectorAll(".progressDiv"), (bar) => {
            if (take(bar)) leaves.push(bar);
        });
        take(byId(tab + "_gallery_container"));
        candidates.push(gallery);
        leaves.push(gallery);
        const galleryBox = gallery.getBoundingClientRect();
        let bottom = galleryBox.bottom;
        let last = gallery;
        let buttons = byId("image_buttons_" + tab);
        if (buttons && results.contains(buttons) && inFlow(buttons)) {
            bottom = Math.max(bottom, buttons.getBoundingClientRect().bottom);
            last = buttons;
            leaves.push(buttons);
        } else {
            buttons = null;
        }
        const tops = candidates.map((node) => node.getBoundingClientRect().top);
        const top = Math.min.apply(null, tops);
        const level = candidates.filter((node, index) => tops[index] <= top + 0.5);
        const anchor = level.find((node) => level.every((other) => node.contains(other)))
            || level[0];
        leaves.sort((a, b) => a.getBoundingClientRect().top - b.getBoundingClientRect().top);
        return {top, bottom, last, anchor, leaves, buttons};
    }

    /** The outermost box inside `stop` that `node` is the first thing in: up
     *  from `node` while it is the first in-flow child of its parent and the
     *  parent is not `stop` itself. A theme's panel around the whole right
     *  side -- Lobe's gray box, Forge's own `variant="panel"` -- is one, with
     *  its padding above Generate; so is the box Gradio wraps a group's
     *  components in, which clips. The space goes above the frame and the
     *  frame is what moves, so what is in it stays in it: moving the first
     *  kept thing alone pulled Generate up out of the panel, and moving the
     *  gallery inside its clipping wrapper pulled its top edge out through
     *  the clip -- a frame with no top, and the picture's top cut. */
    function frameOf(node, stop) {
        let frame = node;
        while (frame.parentElement && frame.parentElement !== stop
            && stop.contains(frame.parentElement)) {
            const parent = frame.parentElement;
            const first = Array.prototype.find.call(parent.children, inFlow);
            if (first !== frame) break;
            frame = parent;
        }
        return frame;
    }

    /** The nearest ancestor of `node` that holds `other` too. */
    function commonAncestor(node, other) {
        let walk = node.parentElement;
        while (walk && !walk.contains(other)) walk = walk.parentElement;
        return walk || document.documentElement;
    }

    /** The bottom of the lowest thing laid out above `node` and across it, up
     *  to `stop` (the scroller, or the page), or null: what the first thing kept
     *  in view may come up to and no further -- the tab buttons over a column
     *  the page scrolls. Sticky and fixed things are `coveredTop`'s. */
    function aboveBottom(node, stop) {
        const box = node.getBoundingClientRect();
        let found = null;
        let walk = node;
        while (walk && walk.parentElement && walk !== stop
            && walk !== document.body && walk !== document.documentElement) {
            for (let sibling = walk.previousElementSibling; sibling;
                sibling = sibling.previousElementSibling) {
                if (!inFlow(sibling)) continue;
                const style = styleOf(sibling);
                if (style && style.position === "sticky") continue;
                const other = sibling.getBoundingClientRect();
                if (other.bottom > box.top + 0.5) continue;
                if (other.right <= box.left || other.left >= box.right) continue;
                found = found === null ? other.bottom : Math.max(found, other.bottom);
            }
            walk = walk.parentElement;
        }
        return found;
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
     *  column alone -- with the reason, for `explain` -- and the margins and
     *  grid that go with it. Reads the layout as the theme draws it (this
     *  extension's spacing margins off) and writes nothing.
     *
     *  The column is measured where it rests: its place in the row with the
     *  scroller at its start, or, when it is sticky, where it is held if that
     *  is lower. One space -- the gap the theme draws between the gallery and
     *  its buttons -- is put above the frame around the first thing kept in
     *  view (a theme's panel, whose own padding then frames Generate inside
     *  it; `frameOf`) under the top of the view, or under what is laid out
     *  above the column; between each kept thing and the next; and under the
     *  buttons (`keptBox`). The gallery takes everything between them that
     *  the rest of the kept part does not
     *  -- so a progress bar arriving above Generate shortens the gallery
     *  rather than pushing the buttons off the screen, and the infotext
     *  arriving after the buttons changes nothing. */
    Fill.prototype.measure = function (tab) {
        return this.bare(tab, () => this.read(tab));
    };

    /** Run `read` with the spacing margins this wrote taken off, and put them
     *  back after: nothing is laid out in between, and nothing changes. */
    Fill.prototype.bare = function (tab, read) {
        const spaced = this.spaced[tab];
        if (!spaced || !spaced.size) return read();
        const ours = [];
        spaced.forEach((saved, node) => {
            ours.push([node, ownMargin(node)]);
            putMargin(node, saved);
        });
        try {
            return read();
        } finally {
            ours.forEach(([node, value]) => putMargin(node, value));
        }
    };

    /** Write the spacing margins `margins` ([node, px] pairs) for a tab, and
     *  put back every other one written before. */
    Fill.prototype.space = function (tab, margins) {
        const spaced = this.spaced[tab] || (this.spaced[tab] = new Map());
        const keep = new Set(margins.map(([node]) => node));
        spaced.forEach((saved, node) => {
            if (keep.has(node)) return;
            putMargin(node, saved);
            spaced.delete(node);
        });
        margins.forEach(([node, value]) => {
            if (!spaced.has(node)) spaced.set(node, ownMargin(node));
            const text = value + "px";
            if (node.style.getPropertyValue("margin-top") !== text
                || node.style.getPropertyPriority("margin-top") !== "important") {
                node.style.setProperty("margin-top", text, "important");
            }
        });
    };

    Fill.prototype.read = function (tab) {
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
        // One space between everything kept in view, and between it and the
        // view's edges: the gap the theme draws between the gallery and its
        // buttons.
        const space = kept.buttons
            ? Math.max(0, kept.buttons.getBoundingClientRect().top - galleryBox.bottom)
            : SPACE_FALLBACK;
        // From where things are drawn now to where the column rests.
        const rest = top - resultsBox.top;
        // How high the first thing kept in view may go: the top of the view, or
        // the bottom of what is laid out above the column, where it rests.
        let ceiling = viewTop;
        const over = aboveBottom(kept.anchor, scroller);
        if (over !== null) ceiling = Math.max(ceiling, over + rest);
        // The frame around the first kept thing, and how far inside it that
        // thing sits (the frame's padding): the frame goes one space under
        // the ceiling, and the first kept thing that much further down.
        const frame = frameOf(kept.anchor, results);
        const inset = Math.max(0, kept.top - frame.getBoundingClientRect().top);
        const first = kept.top + rest;
        const lift = ceiling + space - (first - inset);
        const margins = [];
        const nudged = new Set();
        const nudge = (node, by) => {
            if (Math.abs(by) < 0.25) return;
            const own = styleOf(node);
            nudged.add(node);
            margins.push([node, Math.round((px(own && own.marginTop) + by) * 100) / 100]);
        };
        nudge(frame, lift);
        // Each kept thing one space under the one before it (the buttons are
        // already: that gap is the space), moved by its own frame -- the
        // outermost box, short of what it shares with the thing before it,
        // that it is the first thing in -- and how far that moves the gallery.
        let moved = 0;
        for (let index = 1; index < kept.leaves.length; index += 1) {
            const node = kept.leaves[index];
            if (node === kept.buttons) continue;
            const before = kept.leaves[index - 1];
            const gap = node.getBoundingClientRect().top - before.getBoundingClientRect().bottom;
            const by = space - gap;
            const own = frameOf(node, commonAncestor(node, before));
            nudge(nudged.has(own) ? node : own, by);
            if (Math.abs(by) >= 0.25 && node.getBoundingClientRect().top <= galleryBox.top + 0.5) {
                moved += by;
            }
        }
        const start = ceiling + space + inset;
        const above = galleryBox.top - kept.top + moved;
        const below = kept.bottom - galleryBox.bottom;
        const fits = Math.floor(viewBottom - space - start - above - below);
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
            const end = start + above + drawn + below;
            after = Math.max(0, Math.ceil(viewBottom - end - between));
        }
        const grid = gridOf(gallery, galleryBox, drawn);
        return {height, after, margins, grid, reason: "", gap: Math.round(space)};
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
            if (!found.keep) this.clear(tab, results);
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
        const grid = found.grid;
        write(results, FILL_GRID, grid ? grid.height + "px" : null);
        const row = grid && grid.row ? grid.row : null;
        write(results, FILL_ROW, row ? row + "px" : null);
        write(results, FILL_GRID_PAD, row ? grid.pad + "px" : null);
        if (row && !results.hasAttribute(ROWS_MARK)) results.setAttribute(ROWS_MARK, "");
        if (!row && results.hasAttribute(ROWS_MARK)) results.removeAttribute(ROWS_MARK);
        this.space(tab, found.margins);
        if (!results.hasAttribute(FILL_MARK)) results.setAttribute(FILL_MARK, "");
        return found;
    };

    /** Take the fill off a tab's column: its marks, its properties, its margins. */
    Fill.prototype.clear = function (tab, results) {
        this.space(tab, []);
        results.removeAttribute(FILL_MARK);
        results.removeAttribute(ROWS_MARK);
        FILL_PROPS.forEach((name) => results.style.removeProperty(name));
    };

    function write(node, name, value) {
        if (value === null) {
            if (node.style.getPropertyValue(name)) node.style.removeProperty(name);
        } else if (node.style.getPropertyValue(name) !== value) {
            node.style.setProperty(name, value);
        }
    }

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
        // Under the drag nothing can be seen at the top of the window, and a
        // measure then put the column under the header, where it stayed once
        // the drag ended without another resize. What was written stays until
        // the drag ends; the body's class coming off is the measure's cue.
        if (dragging()) {
            this.deferred = true;
            return;
        }
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
        // The end of Forge's divider drag: its class leaving the body.
        if (typeof MutationObserver === "function" && document.body) {
            const ended = new MutationObserver(() => {
                if (this.deferred && !dragging()) {
                    this.deferred = false;
                    this.refresh();
                }
            });
            ended.observe(document.body, {attributes: true, attributeFilter: ["class"]});
            this.disposers.push(() => ended.disconnect());
        }
        // A picture loading in the grid view: the first one's shape decides
        // how many rows fit, and a load does not resize the column.
        const loaded = (event) => {
            const target = event.target;
            if (target && target.tagName === "IMG" && target.closest
                && target.closest(".grid-container")) {
                this.refresh();
            }
        };
        document.addEventListener("load", loaded, true);
        this.disposers.push(() => document.removeEventListener("load", loaded, true));
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
            if (results) this.clear(tab, results);
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

    // Asked for next: "input fields should not scroll. They should always
    // present full content ... I dont want the view to scroll up to the top
    // when i select it to begin typing, i want what is in view to stay in view,
    // so i can put the cursor exactly where i need it". So while docked every
    // text box is as tall as its text (`fit`): the column is the one thing that
    // scrolls. Its height is written inline with priority, and again whenever
    // anything rewrites it (Gradio sizes a box to its line limit on every
    // input) or the column's content changes (an accordion opening shows a box
    // that had no width to measure); undocking puts back what was there. And
    // the engaging tap puts the caret where the finger was, focused without
    // scrolling, so what is on screen stays on screen.

    function Guard() {
        this.column = null;
        this.engaged = null;
        this.saved = new Map();
        this.observer = null;
        this.pending = false;
        this.onClick = this.click.bind(this);
        this.onKey = this.key.bind(this);
        this.onFocus = this.focus.bind(this);
        this.onInput = (event) => this.fit(event.target);
    }

    /** Make `area` as tall as its text, keeping the column where it is. */
    Guard.prototype.fit = function (area) {
        if (!area || area.tagName !== "TEXTAREA" || !this.column || !shows(area)) return;
        const column = this.column;
        const keep = column.scrollTop;
        const style = area.style;
        if (!this.saved.has(area)) {
            this.saved.set(area, {value: style.getPropertyValue("height"),
                                  priority: style.getPropertyPriority("height")});
        }
        const before = style.getPropertyValue("height");
        // Measured at its natural height (its rows), so it shrinks as well as
        // grows; nothing is painted in between.
        style.setProperty("height", "auto", "important");
        const computed = window.getComputedStyle(area);
        const px = (name) => parseFloat(computed.getPropertyValue(name)) || 0;
        const wanted = computed.boxSizing === "border-box"
            ? area.scrollHeight + px("border-top-width") + px("border-bottom-width")
            : area.scrollHeight - px("padding-top") - px("padding-bottom");
        const value = Math.ceil(wanted) + "px";
        style.setProperty("height", value, "important");
        if (column.scrollTop !== keep) column.scrollTop = keep;
        if (before !== value) this.forget();
    };

    Guard.prototype.fitAll = function () {
        if (!this.column) return;
        this.column.querySelectorAll("textarea").forEach((area) => this.fit(area));
        this.forget();
    };

    /** Our own writes are not news to the observer. */
    Guard.prototype.forget = function () {
        if (this.observer && typeof this.observer.takeRecords === "function") {
            this.observer.takeRecords();
        }
    };

    Guard.prototype.schedule = function () {
        if (this.pending) return;
        this.pending = true;
        window.requestAnimationFrame(() => {
            this.pending = false;
            this.fitAll();
        });
    };

    /** Put back every height `fit` wrote. */
    Guard.prototype.unfit = function () {
        this.saved.forEach((entry, area) => {
            if (entry.value) area.style.setProperty("height", entry.value, entry.priority || "");
            else area.style.removeProperty("height");
        });
        this.saved.clear();
    };

    // What a mirror of a text box copies so its text wraps line for line as
    // the box's does.
    const MIRRORED = ["box-sizing", "padding-top", "padding-right", "padding-bottom",
        "padding-left", "border-top-width", "border-right-width", "border-bottom-width",
        "border-left-width", "font-family", "font-size", "font-style", "font-weight",
        "font-variant", "font-stretch", "line-height", "letter-spacing", "word-spacing",
        "text-indent", "text-transform", "text-align", "tab-size", "direction",
        "word-break", "overflow-wrap", "white-space"];

    /** The character offset in `area` under (x, y), or null when it cannot be
     *  said. Read off a copy of the box's text laid out exactly over it,
     *  because no browser answers this for a text box itself line for line:
     *  Chromium's `caretPositionFromPoint` over a textarea always answers the
     *  first line. The copy is see-through and gone before anything paints. */
    function caretAt(area, x, y) {
        let mirror = null;
        try {
            const rect = area.getBoundingClientRect();
            const computed = window.getComputedStyle(area);
            mirror = document.createElement("div");
            const style = mirror.style;
            MIRRORED.forEach((name) => style.setProperty(name, computed.getPropertyValue(name)));
            if (area.tagName === "TEXTAREA") style.setProperty("white-space", "pre-wrap");
            else style.setProperty("white-space", "pre");
            style.setProperty("border-style", "solid");
            style.setProperty("border-color", "transparent");
            style.setProperty("position", "fixed");
            style.setProperty("left", rect.left + "px");
            style.setProperty("top", rect.top - (area.scrollTop || 0) + "px");
            style.setProperty("width", rect.width + "px");
            style.setProperty("margin", "0");
            style.setProperty("overflow", "hidden");
            style.setProperty("opacity", "0");
            style.setProperty("pointer-events", "auto");
            style.setProperty("z-index", "2147483647");
            const text = document.createTextNode(String(area.value || ""));
            mirror.appendChild(text);
            // A trailing line break only lays out with something after it.
            mirror.appendChild(document.createTextNode("\u200b"));
            document.body.appendChild(mirror);
            let node = null;
            let offset = null;
            if (typeof document.caretPositionFromPoint === "function") {
                const found = document.caretPositionFromPoint(x, y);
                if (found) { node = found.offsetNode; offset = found.offset; }
            } else if (typeof document.caretRangeFromPoint === "function") {
                const found = document.caretRangeFromPoint(x, y);
                if (found) { node = found.startContainer; offset = found.startOffset; }
            }
            if (node === text) return Math.min(offset, text.length);
            if (node && node.parentNode === mirror) return text.length;
            return null;
        } catch (error) {
            return null;
        } finally {
            if (mirror && mirror.parentNode) mirror.parentNode.removeChild(mirror);
        }
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
        // Bubbling, so it runs after the box's own handlers -- Gradio's
        // resize among them -- in the same task, before anything is painted.
        column.addEventListener("input", this.onInput);
        if (typeof MutationObserver === "function") {
            this.observer = new MutationObserver(() => this.schedule());
            this.observer.observe(column, {childList: true, subtree: true, attributes: true,
                                           attributeFilter: ["style", "class", "hidden"]});
        }
        this.fitAll();
    };

    Guard.prototype.detach = function () {
        const column = this.column;
        if (!column) return;
        this.release();
        if (this.observer) this.observer.disconnect();
        this.observer = null;
        this.unfit();
        column.classList.remove(GUARD_CLASS);
        document.removeEventListener("click", this.onClick, true);
        document.removeEventListener("keydown", this.onKey, true);
        column.removeEventListener("focusin", this.onFocus);
        column.removeEventListener("input", this.onInput);
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
        // Wherever the focus or the caret goes, the column stays where it was:
        // what the reader tapped is what they are looking at.
        const column = this.column;
        const keep = column ? column.scrollTop : 0;
        const offset = chosen === hit && chosen.matches("textarea, input[type=text], " +
            "input:not([type]), input[type=search]") ? caretAt(chosen, x, y) : null;
        try {
            chosen.focus({preventScroll: true});
        } catch (error) {
            chosen.focus();
        }
        if (offset !== null) {
            try {
                chosen.setSelectionRange(offset, offset);
            } catch (error) {
                // Not every input takes a selection; the focus is what matters.
            }
        }
        if (column && column.scrollTop !== keep) column.scrollTop = keep;
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
        // Only a press of the reader's. Scripts on this page -- Forge's, this
        // extension's, Mini Paint NEO's -- press hidden buttons with
        // `.click()`, and each of those used to read as a tap outside the
        // engaged field: it let go and took the keyboard away mid-word. An
        // engaged field is let go by the reader (a tap elsewhere, Escape) and
        // by nothing else.
        if (event && event.isTrusted === false) return;
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
    // On the results column while it is docked beside the settings rather
    // than under them: a pixel of margin, so it never sits on the row's left
    // edge. Forge's own mobile script (extensions-builtin/mobile, on every
    // window resize) reads a results column at `offsetLeft === 0` as a
    // phone's stacked layout and moves Generate's box into that column, and
    // back into the toprow's actions column once the edge is clear -- which,
    // with Lobe's split previewer keeping the box in the gallery's container,
    // left Generate in the left column after focus had gone on and off around
    // the dock (the browser's full screen resizes the window both ways). The
    // dock gives the row's whole width to the results, so without this the
    // column was the edge. A column that was at the edge already -- a phone --
    // gets no margin and stays what the script took it for.
    const DOCK_BESIDE = "forge-assistant-docked-beside";
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

    /** The results column of `row` while it is drawn beside the settings --
     *  off the row's left edge -- or null: under them (a phone), or not
     *  there. Read before the row changes, because it is the row before the
     *  dock that says which. See `DOCK_BESIDE`. */
    function besideOf(row) {
        if (!row || !row.children) return null;
        const results = Array.prototype.find.call(row.children,
            (child) => child.id && /_results$/.test(child.id));
        if (!results || !laidOut(results)) return null;
        return results.offsetLeft > 0 ? results : null;
    }

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
            const beside = besideOf(found.row);
            this.active = Object.assign({saved, beside}, found);
            found.column.classList.add(DOCK_COLUMN);
            if (found.row) found.row.classList.add(DOCK_ROW);
            if (beside) beside.classList.add(DOCK_BESIDE);
            this.guard.attach(found.column);
            if (NS.fill) NS.fill.refresh();
        }
        const style = this.active.column.style;
        // A wider or narrower panel rewraps every box.
        if (rect && Math.round(rect.width) !== this.active.width) {
            this.active.width = Math.round(rect.width || 0);
            this.guard.schedule();
        }
        if (!rect || !(rect.width > 0) || !(rect.height > 0)) {
            // An engaged field stays engaged: only the reader lets it go. The
            // slot has no height for a moment more often than it looks -- a
            // phone's keyboard shrinking the visible window caps the panel,
            // and the slot gives way first -- and letting go here closed the
            // keyboard under somebody typing.
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
        if (active.beside) active.beside.classList.remove(DOCK_BESIDE);
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
