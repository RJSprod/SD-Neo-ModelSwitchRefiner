// Model Chain -- Krea Creative Mode Spatial Layout, browser side.
//
// This file owns one thing: the layout editor. Somebody presses Edit Layout, a
// workspace opens in the page, they place boxes and type into them, they press
// Save & Return, and the serialized document goes into a hidden textbox that
// travels with the next Generate like any other control. That is the whole
// contract.
//
// What this file must never become
// --------------------------------
// Creative Mode used to have a browser gate: it swallowed the Generate click,
// pressed a hidden button to run the roll, polled a hidden textbox on a
// setInterval until the server wrote a token into it, and only then clicked
// Generate again. It worked in a focused tab and nowhere else -- browsers
// throttle those timers to one tick a second in a hidden tab and one a minute
// in a frozen one, and run them not at all in a closed one -- so a generation
// was late if you changed windows and never happened if you closed the tab.
//
// The state box here looks like that box and is the opposite of it. It is an
// *input*. Nothing polls it, nothing waits for it, and no generation is held up
// by it: whatever is in it when the request leaves is what the compositor
// composes, in Python, on the thread the host is already running the job on.
// Press Generate and close the browser and the picture still arrives.
//
// So: no click listener on Generate, no timer of any kind, no polling, and
// nothing here that a generation's completion depends on. If every line of this
// file fails, the Edit Layout button does nothing and the last saved layout is
// still the one that gets composed.
//
// Why the canvas is a div
// -----------------------
// Regions are elements. That makes each one focusable, styleable by a theme,
// findable by id, and readable by a test that never opens a browser -- and it
// makes hit-testing, dragging and z-order the browser's job rather than a
// redraw loop's. A <canvas> would put all of that behind a bitmap in order to
// draw rectangles with square corners.
//
// Why one pointer path and not two
// --------------------------------
// The editor used to listen for mousedown/mousemove/mouseup on the document.
// On a tablet that is either nothing at all or a synthesised echo arriving
// after a 300ms delay and a scroll. Pointer Events are one path for mouse,
// finger and pen, and setPointerCapture keeps the drag attached to the finger
// that started it even when the finger leaves the frame -- which is what makes
// the document-level listeners unnecessary rather than merely unfashionable.
//
// Why the selected region has a proxy
// -----------------------------------
// Hit-testing follows paint order, so a region behind another one cannot be
// grabbed where they overlap however firmly the region list says it is
// selected. Raising the selection would fix the grab by changing the
// composition, which is not a trade the user asked for. So the selection is
// drawn a second time -- outline, label, move surface, eight handles -- in a
// proxy element above every region body, reading and writing the selected
// region's bbox and never touching its z. Selecting changes what you can drag
// and nothing about what the compositor is told.
//
// Why it is still one file
// ------------------------
// Forge loads javascript/*.js as plain scripts in directory order, with no
// module system and no guaranteed order, so splitting the editor across files
// buys separate files and a load-order bug. It is split into the controllers
// the design intent names -- state, history, frame, selection, pointer, list,
// inspector, serialization -- as sections with no shared scope beyond `state`.

(function () {
    "use strict";

    const P = "mc-krea-spatial";
    const SCALE = 1000;
    const MIN_SIZE = 8;          // normalized units; smaller than this is a tap
    const MAX_REGIONS = 24;      // matches prompt_master/krea/spatial.MAX_REGIONS
    const HISTORY_DEPTH = 50;    // §10.3 asks for 25-50; a snapshot is a short string
    const NUDGE = 5;             // normalized units per arrow press
    const NUDGE_BIG = 25;
    const ADD_SIZE = 340;        // a new region is ~34% of the frame, per §7.1
    const ZOOMS = [0.5, 0.75, 1, 1.25, 1.5, 2, 3, 4];

    const IDS = {
        // Gradio's, not the workspace's.
        open: P + "-open",
        state: P + "-state",
        // The workspace.
        workspace: P + "-workspace",
        canvas: P + "-canvas",
        scroll: P + "-scroll",
        guides: P + "-guides",
        regions: P + "-regions",
        proxy: P + "-proxy",
        proxyLabel: P + "-proxy-label",
        list: P + "-list",
        empty: P + "-empty",
        count: P + "-count",
        selectedName: P + "-selected-name",
        name: P + "-name",
        type: P + "-type",
        text: P + "-text",
        textField: P + "-text-field",
        prompt: P + "-prompt",
        promptLabel: P + "-prompt-label",
        literalPrefix: P + "-literal-prefix",
        literalSuffix: P + "-literal-suffix",
        framing: P + "-framing",
        framingField: P + "-framing-field",
        angle: P + "-angle",
        angleField: P + "-angle-field",
        autoHint: P + "-auto-hint",
        grid: P + "-grid",
        size: P + "-size",
        warning: P + "-warning",
        message: P + "-message",
        confirm: P + "-confirm",
        keep: P + "-keep",
        discard: P + "-discard",
        add: P + "-add",
        draw: P + "-draw",
        clear: P + "-clear",
        undo: P + "-undo",
        redo: P + "-redo",
        duplicate: P + "-duplicate",
        remove: P + "-delete",
        front: P + "-front",
        forward: P + "-raise",
        back: P + "-lower",
        bottom: P + "-bottom",
        zoomFit: P + "-zoom-fit",
        zoomIn: P + "-zoom-in",
        zoomOut: P + "-zoom-out",
        zoomLevel: P + "-zoom-level",
        full: P + "-full",
        save: P + "-save",
        cancel: P + "-cancel",
    };

    // ------------------------------------------------------------------ //
    // SpatialEditorState -- everything the editor is doing right now
    // ------------------------------------------------------------------ //
    //
    // One object so a test can read it, and so "what state is this in" has
    // exactly one answer. The split that matters is the one §15 draws: only
    // `working` is model-facing. Selection, zoom, history, the drag in flight
    // and the confirm bar are browser state and are never serialized.

    const state = {
        wired: false,
        listening: false,   // the one document-level listener, installed once
        dragging: "",       // the region id a list drag is carrying, if any
        pendingSave: false, // an edit finished and not yet committed
        typing: false,      // a keystroke is in flight; see settle()
        committed: "",      // what was last written to the state box
        workspace: null,    // the element that is live, once claimed
        open: false,
        drawing: false,
        working: null,      // the layout being edited; null while closed
        savedCanvas: {width: 0, height: 0},  // the frame the layout was drawn on
        selected: "",       // region id
        drag: null,         // {kind, id, from, origin, before}
        pointer: null,      // the pointerId that owns the drag
        counter: 0,
        zoom: 1,
        baseline: "",       // serialize() as it was on entry; Back compares to it
        confirming: false,
        editing: "",        // history coalescing token; see mark()
        past: [],
        future: [],
    };

    function root() {
        const app = typeof gradioApp === "function" ? gradioApp() : document;
        return app || document;
    }

    function byId(id) {
        try {
            return root().querySelector("#" + id) || document.getElementById(id);
        } catch (error) {
            return null;
        }
    }

    function stateBox() {
        const holder = byId(IDS.state);
        return holder ? holder.querySelector("textarea, input") : null;
    }

    function clickable(id) {
        const holder = byId(id);
        if (!holder) return null;
        return holder.tagName === "BUTTON" ? holder : holder.querySelector("button");
    }

    // Gradio binds to the input event, so setting .value alone updates the page
    // and tells the server nothing. The host ships updateInput for exactly this;
    // the fallback is what it does.
    function publish(element, value) {
        element.value = value;
        if (typeof updateInput === "function") {
            updateInput(element);
            return;
        }
        element.dispatchEvent(new Event("input", {bubbles: true}));
        element.dispatchEvent(new Event("change", {bubbles: true}));
    }

    function prevent(event) {
        if (event && typeof event.preventDefault === "function") event.preventDefault();
    }

    // style.setProperty is how a custom property is set, and a stub DOM in a
    // test has a plain object for a style. Both are served.
    function setVar(element, name, value) {
        if (!element || !element.style) return;
        if (typeof element.style.setProperty === "function") {
            element.style.setProperty(name, value);
            return;
        }
        element.style[name] = value;
    }

    function clamp(value, low, high) {
        return Math.max(low, Math.min(high, value));
    }

    function enable(id, on) {
        const element = byId(id);
        if (element) element.disabled = !on;
    }

    // ------------------------------------------------------------------ //
    // SerializationAdapter -- the document, unchanged in shape
    // ------------------------------------------------------------------ //
    //
    // §15: version 1 in, version 1 out, the same keys in the same order. Every
    // addition this refactor makes -- selection, zoom, undo, the proxy -- is
    // browser state and appears nowhere below.

    function blank() {
        return {
            version: 1,
            canvas: {width: 0, height: 0, grid: "none"},
            compose_mode: "smart",
            auto_position_hint: true,
            regions: [],
        };
    }

    function readLayout() {
        const box = stateBox();
        const text = box ? String(box.value || "").trim() : "";
        if (!text) return blank();
        let parsed;
        try {
            parsed = JSON.parse(text);
        } catch (error) {
            // Refused, not repaired. A document this file cannot read is a
            // document it must not overwrite: Python says the same thing about
            // the same string, out loud, on the panel.
            console.warn("Model Chain: the saved spatial layout could not be read", error);
            return null;
        }
        if (!parsed || typeof parsed !== "object") return null;
        if (Number(parsed.version) !== 1) return null;
        const made = blank();
        made.canvas = Object.assign(made.canvas, parsed.canvas || {});
        made.compose_mode = parsed.compose_mode === "direct" ? "direct" : "smart";
        made.auto_position_hint = parsed.auto_position_hint !== false;
        made.regions = (Array.isArray(parsed.regions) ? parsed.regions : [])
            .map(normalise).filter(Boolean);
        return made;
    }

    function normalise(entry) {
        if (!entry || typeof entry !== "object") return null;
        const box = orient(entry.bbox);
        if (!box) return null;
        const kind = entry.type === "text" ? "text" : "obj";
        return {
            id: String(entry.id || nextId()),
            name: String(entry.name || ""),
            type: kind,
            bbox: box,
            prompt: String(entry.prompt || ""),
            // This region's two Literal Prompt fields, carried through the
            // editor untouched. They are never parsed here and never merged
            // into the prompt: Python does both, once, at generation time.
            literalPrefix: String(entry.literal_prefix || ""),
            literalSuffix: String(entry.literal_suffix || ""),
            text: String(entry.text || ""),
            framing: String(entry.framing || ""),
            angle: String(entry.angle || ""),
            z: Number.isFinite(Number(entry.z)) ? Number(entry.z) : 0,
        };
    }

    // The canonical box, defined here the same way prompt_master/krea/spatial.py
    // defines it: clamped, ordered, and rejected outright when it has no area.
    // A drag from bottom-right to top-left is not an error and a tap is not a
    // region.
    function orient(values) {
        if (!Array.isArray(values) || values.length < 4) return null;
        const numbers = values.slice(0, 4).map(function (value) {
            const number = Math.round(Number(value));
            if (!Number.isFinite(number)) return 0;
            return clamp(number, 0, SCALE);
        });
        let [x0, y0, x1, y1] = numbers;
        if (x1 < x0) { const swap = x0; x0 = x1; x1 = swap; }
        if (y1 < y0) { const swap = y0; y0 = y1; y1 = swap; }
        if (x1 <= x0 || y1 <= y0) return null;
        return [x0, y0, x1, y1];
    }

    function nextId() {
        state.counter += 1;
        return "r" + state.counter;
    }

    // Paint order, and therefore hit-test order: the last one drawn is the one
    // on top and the one a pointer lands on. Taken over a document rather than
    // over `state.working`, because the compact canvas has a working document
    // of its own and must order it by the same rule -- two orderings of one
    // layout is how a box you can see behind another becomes the one that moves.
    function orderedIn(document_) {
        return document_.regions
            .map(function (region, index) { return {region: region, index: index}; })
            .sort(function (left, right) {
                return (left.region.z - right.region.z) || (left.index - right.index);
            })
            .map(function (entry) { return entry.region; });
    }

    function ordered() {
        return orderedIn(state.working);
    }

    function find(id) {
        if (!state.working) return null;
        return state.working.regions.filter(function (region) {
            return region.id === id;
        })[0] || null;
    }

    function serializeIn(source) {
        const document_ = {
            version: 1,
            canvas: {
                width: Number(source.canvas.width) || 0,
                height: Number(source.canvas.height) || 0,
                grid: source.canvas.grid || "none",
            },
            compose_mode: source.compose_mode,
            auto_position_hint: !!source.auto_position_hint,
            regions: orderedIn(source).map(function (region) {
                const entry = {
                    id: region.id,
                    name: region.name || region.id,
                    type: region.type,
                    bbox: region.bbox.slice(),
                    prompt: region.prompt,
                };
                if (region.type === "text") entry.text = region.text;
                // Written only when they carry something, so a layout with no
                // region literals in it serializes to the bytes it always did.
                if (region.literalPrefix) entry.literal_prefix = region.literalPrefix;
                if (region.literalSuffix) entry.literal_suffix = region.literalSuffix;
                entry.framing = region.framing;
                entry.angle = region.angle;
                entry.z = region.z;
                return entry;
            }),
        };
        return JSON.stringify(document_);
    }

    function serialize() {
        return serializeIn(state.working);
    }

    // ------------------------------------------------------------------ //
    // HistoryController -- one entry per completed action
    // ------------------------------------------------------------------ //
    //
    // A snapshot is the editable half of the document as a string: regions,
    // the auto-hint flag and the grid. The frame size is not in it, because the
    // frame is a fact about txt2img rather than an edit, and undoing back onto
    // a stale one would be a lie about what is going to be generated.
    //
    // `mark(token)` is what keeps a drag from becoming four hundred history
    // entries and a typed word from becoming one per keystroke: the first call
    // with a given token records, every later call with the same token is a
    // no-op, and any other action clears the token.

    function snapshot() {
        return JSON.stringify({
            regions: state.working.regions,
            auto: !!state.working.auto_position_hint,
            grid: state.working.canvas.grid || "none",
        });
    }

    function restore(text) {
        let read;
        try {
            read = JSON.parse(text);
        } catch (error) {
            return;
        }
        state.working.regions = (read.regions || []).map(normalise).filter(Boolean);
        state.working.auto_position_hint = read.auto !== false;
        state.working.canvas.grid = read.grid || "none";
    }

    function keep(before) {
        state.past.push(before);
        if (state.past.length > HISTORY_DEPTH) state.past.shift();
        state.future.length = 0;
    }

    function mark(token) {
        if (!state.working) return;
        state.pendingSave = true;
        if (token && token === state.editing) return;
        state.editing = token || "";
        keep(snapshot());
    }

    // A drag records the state it started from, and only if the drag changed
    // something: §10.3 asks for one history item per completed drag, and a drag
    // that moved nothing is not a completed anything.
    function record(before) {
        if (!before || before === snapshot()) return;
        state.editing = "";
        state.pendingSave = true;
        keep(before);
    }

    function undo() {
        if (!state.working || !state.past.length) return;
        // Undo is an edit like any other as far as committing goes: §6.4 asks
        // for the undone position to be saved too, or the screen and the
        // generation disagree -- which is the one thing Auto Save exists to
        // prevent. The compact canvas has said so since it was written.
        state.pendingSave = true;
        const now = snapshot();
        restore(state.past.pop());
        state.future.push(now);
        if (state.future.length > HISTORY_DEPTH) state.future.shift();
        after();
    }

    function redo() {
        if (!state.working || !state.future.length) return;
        state.pendingSave = true;
        const now = snapshot();
        restore(state.future.pop());
        state.past.push(now);
        if (state.past.length > HISTORY_DEPTH) state.past.shift();
        after();
    }

    // Undoing past the creation of the selected region leaves a selection
    // pointing at nothing, which the inspector and the proxy both have to be
    // told about rather than left to discover.
    function after() {
        state.editing = "";
        if (state.selected && !find(state.selected)) state.selected = "";
        say("");
        paint();
    }

    // ------------------------------------------------------------------ //
    // FrameController -- the frame is the shape of the image
    // ------------------------------------------------------------------ //

    function hostSize() {
        function read(id) {
            const holder = byId(id);
            if (!holder) return 0;
            const field = holder.querySelector("input[type=number], input");
            return field ? Number(field.value) || 0 : 0;
        }
        return {width: read("txt2img_width"), height: read("txt2img_height")};
    }

    function divisor(a, b) {
        let left = Math.abs(Math.round(a));
        let right = Math.abs(Math.round(b));
        while (right) { const rest = left % right; left = right; right = rest; }
        return left || 1;
    }

    function ratio(width, height) {
        const by = divisor(width, height);
        return (width / by) + ":" + (height / by);
    }

    function shape(width, height) {
        if (width === height) return "Square";
        return width > height ? "Landscape" : "Portrait";
    }

    // "1024 × 1536 · 2:3 · Portrait", and the exact reduced ratio when it is
    // not a familiar one -- §4.3 wants the number, not a nearest-neighbour lie.
    function dimensions(width, height) {
        return width + " × " + height + " · " + ratio(width, height)
            + " · " + shape(width, height);
    }

    function brief(width, height) {
        return width + " × " + height + " (" + ratio(width, height) + ")";
    }

    // Normalized coordinates are fractions of the frame, so a resolution change
    // moves nothing and a *ratio* change reframes everything. The boxes are left
    // exactly where they are and the change is reported, which is the design
    // intent's rule in the strongest terms it uses: never silently delete layout
    // state. Reprojecting would be this file deciding which of somebody's boxes
    // deserved to keep its shape.
    function reframe() {
        if (!state.working) return;
        const size = hostSize();
        const canvas = byId(IDS.canvas);
        const warning = byId(IDS.warning);
        const note = byId(IDS.size);
        if (!(size.width > 0 && size.height > 0)) {
            if (note) note.textContent = "";
            return;
        }
        if (canvas) {
            canvas.style.aspectRatio = size.width + " / " + size.height;
            setVar(canvas, "--mc-ar-w", String(size.width));
            setVar(canvas, "--mc-ar-h", String(size.height));
        }
        if (note) note.textContent = dimensions(size.width, size.height);

        // Compared against the frame the layout was *saved* on rather than
        // against whatever the last reframe wrote, so the notice survives a
        // second reframe and still tells the truth if the user changes the
        // size again while the editor is open.
        const was = state.savedCanvas;
        const now = size.width / size.height;
        const changed = state.working.regions.length > 0
            && Number(was.width) > 0 && Number(was.height) > 0
            && Math.abs((was.width / was.height) - now) / now > 0.02;
        if (warning) {
            warning.hidden = !changed;
            warning.textContent = changed
                ? "Frame changed from " + brief(was.width, was.height) + " to "
                  + brief(size.width, size.height) + ". Region coordinates were "
                  + "preserved — the boxes are unchanged, but they now cover "
                  + "different parts of the picture. Clear all starts again."
                : "";
        }
        state.working.canvas.width = size.width;
        state.working.canvas.height = size.height;
    }

    // Zoom is display only and never touches a stored coordinate: it scales the
    // frame's CSS width and lets the scroll container do the rest. §4.5.
    function zoomTo(value) {
        state.zoom = clamp(value, ZOOMS[0], ZOOMS[ZOOMS.length - 1]);
        const workspace = byId(IDS.workspace);
        setVar(workspace, "--mc-zoom", String(state.zoom));
        const level = byId(IDS.zoomLevel);
        if (level) level.textContent = Math.round(state.zoom * 100) + "%";
        if (workspace && workspace.classList) {
            workspace.classList.toggle("zoomed", state.zoom !== 1);
        }
    }

    function zoomStep(direction) {
        const at = ZOOMS.filter(function (value) {
            return direction > 0 ? value > state.zoom + 0.001 : value < state.zoom - 0.001;
        });
        if (!at.length) return;
        zoomTo(direction > 0 ? at[0] : at[at.length - 1]);
    }

    // Full screen: the whole display for the frame, and Back still comes back.
    //
    // Two mechanisms, and the order matters. The Fullscreen API is asked first
    // because it is the only one nothing can clip, overlap or out-stack: the
    // element is promoted out of the page's layout entirely, so no theme's
    // `overflow: hidden`, `transform` or z-index is in the argument. It also
    // takes the browser's own chrome away, which on a tablet is most of the
    // screen this is asking for.
    //
    // A page that refuses it -- an iframe without `allowfullscreen`, a browser
    // that does not offer it, a user who said no -- falls back to a fixed block
    // over the page. That is the thing §3.1 told this editor not to be, and it
    // is fine here for the reason the modal was not: it is somewhere the user
    // asked to go and can leave, rather than the only way to edit at all.
    function fullscreenOn() {
        const workspace = byId(IDS.workspace);
        if (!workspace) return false;
        if (typeof document !== "undefined" && document.fullscreenElement === workspace) {
            return true;
        }
        return !!(workspace.classList && workspace.classList.contains("fullpage"));
    }

    function leaveFullscreen() {
        const workspace = byId(IDS.workspace);
        if (workspace && workspace.classList) workspace.classList.remove("fullpage");
        if (typeof document === "undefined") return;
        if (document.fullscreenElement === workspace
            && typeof document.exitFullscreen === "function") {
            try {
                const left = document.exitFullscreen();
                if (left && typeof left.catch === "function") left.catch(function () {});
            } catch (error) {
                // Already out, or refused. The class is off either way.
            }
        }
    }

    function toggleFullscreen() {
        const workspace = byId(IDS.workspace);
        if (!workspace) return;
        if (fullscreenOn()) {
            leaveFullscreen();
            paintChrome();
            return;
        }
        function fallback() {
            if (workspace.classList) workspace.classList.add("fullpage");
            paintChrome();
        }
        if (typeof workspace.requestFullscreen !== "function") {
            fallback();
            return;
        }
        try {
            const asked = workspace.requestFullscreen();
            if (asked && typeof asked.catch === "function") asked.catch(fallback);
        } catch (error) {
            fallback();
            return;
        }
        paintChrome();
    }

    // ------------------------------------------------------------------ //
    // Painting -- frame, list, inspector, chrome
    // ------------------------------------------------------------------ //

    function paint() {
        if (!state.working) return;
        paintFrame();
        paintList();
        paintInspector();
        paintChrome();
        settle();
    }

    // §6.4, carried through from the compact canvas: with Auto Save on, an edit
    // is committed the moment it is finished, in both editors, because a switch
    // called Auto Save that only one of the two canvases obeyed is a switch that
    // means different things in two places on one panel.
    //
    // Every edit here already brackets itself: `mark()` before it, `paint()`
    // after. So this is one funnel rather than a call at the end of a dozen
    // actions -- reorder, delete, duplicate, nudge, orient, undo, the grid, the
    // position hint -- and the two things it must not do are stated where they
    // are known:
    //
    //   `state.drag`    a gesture in flight paints on every pointermove, and
    //                   committing on each of them would be a round trip per
    //                   pixel. The pointerup repaints, and that one commits.
    //   `state.typing`  a keystroke in a text field is not a finished edit.
    //                   §"save when the cursor leaves the field": the change
    //                   event, which the browser fires on blur, paints and
    //                   commits like everything else.
    function settle() {
        if (!state.pendingSave) return;
        if (state.drag || state.dragging || state.typing) return;
        state.pendingSave = false;
        if (!autoSaveOn()) return;
        const box = stateBox();
        if (!box || !state.working) return;
        const said = serialize();
        // Nothing new to say is nothing to send. Gradio treats every publish as
        // an input event and a round trip, and a repaint that changed only which
        // row is selected must not cost one.
        if (said === state.committed) return;
        state.committed = said;
        publish(box, said);
        // The compact canvas is a view of the same document, so it is stale the
        // moment this writes one.
        compactLoad(true);
    }

    function place(element, bbox) {
        element.style.left = (bbox[0] / SCALE * 100) + "%";
        element.style.top = (bbox[1] / SCALE * 100) + "%";
        element.style.width = ((bbox[2] - bbox[0]) / SCALE * 100) + "%";
        element.style.height = ((bbox[3] - bbox[1]) / SCALE * 100) + "%";
    }

    function body(region) {
        const element = document.createElement("div");
        element.className = P + "-region";
        element.dataset.regionId = region.id;
        const label = document.createElement("span");
        label.className = P + "-label";
        element.appendChild(label);
        return element;
    }

    // Rebuilt only when the set of regions changes; otherwise the same elements
    // are moved. A drag repaints on every pointermove, and throwing away and
    // recreating two dozen elements sixty times a second is how a finger drag
    // on a tablet turns into a slideshow.
    function paintFrame() {
        const holder = byId(IDS.regions);
        if (!holder || !state.working) return;
        const wanted = ordered();
        const signature = wanted.map(function (region) { return region.id; }).join(",");
        if (holder.dataset.signature !== signature) {
            holder.textContent = "";
            wanted.forEach(function (region) { holder.appendChild(body(region)); });
            holder.dataset.signature = signature;
        }
        const found = {};
        (holder.querySelectorAll("." + P + "-region") || []).forEach(function (node) {
            found[node.dataset.regionId] = node;
        });
        wanted.forEach(function (region, depth) {
            const node = found[region.id];
            if (!node) return;
            place(node, region.bbox);
            node.style.zIndex = String(depth + 1);
            node.title = region.name || region.id;
            node.classList.toggle("selected", region.id === state.selected);
            node.classList.toggle("text", region.type === "text");
            const label = node.querySelector("." + P + "-label");
            if (label) label.textContent = region.name || region.id;
        });
        paintProxy();
        const guides = byId(IDS.guides);
        if (guides) guides.className = P + "-guides "
            + (state.working.canvas.grid || "none");
    }

    // SelectionController's visible half: one element, above every region body,
    // carrying the outline, the label, the move surface and the eight handles
    // of whatever is selected. It reads and writes that region's bbox and never
    // touches its z -- which is the entire answer to "the box I selected is
    // buried and I cannot grab it".
    function paintProxy() {
        const proxy = byId(IDS.proxy);
        if (!proxy || !state.working) return;
        const region = find(state.selected);
        const label = byId(IDS.proxyLabel);
        if (!region) {
            proxy.hidden = true;
            proxy.dataset.regionId = "";
            if (label) label.textContent = "";
            return;
        }
        proxy.hidden = false;
        proxy.dataset.regionId = region.id;
        place(proxy, region.bbox);
        proxy.classList.toggle("text", region.type === "text");
        if (label) label.textContent = region.name || region.id;
    }

    // RegionListController. Prompt order, top to bottom: the first row is the
    // first element in the composed prompt and the last row is the one drawn on
    // top of the others. It used to be reversed -- frontmost first, the way a
    // layers panel reads -- which meant the list and the prompt disagreed about
    // what "first" was, and the list is the thing somebody drags.
    //
    // One number does both jobs. `z` decides the order the compositor writes
    // the elements array in *and* which box is on top where two overlap, so a
    // drag that changes one changes the other, and the "i" on the panel header
    // says so.
    function paintList() {
        const list = byId(IDS.list);
        if (!list || !state.working) return;
        const rows = ordered();
        list.textContent = "";
        rows.forEach(function (region) {
            const row = document.createElement("div");
            row.className = P + "-row" + (region.id === state.selected ? " selected" : "");
            row.dataset.regionId = region.id;
            // Reordering by pointer. The rows are rebuilt on every paint, so
            // the listeners for this are delegated to the container -- see
            // wire() -- and all a row carries is the flag that makes it a drag
            // source.
            row.draggable = true;
            row.setAttribute && row.setAttribute("role", "option");
            row.setAttribute && row.setAttribute("aria-selected",
                                                 region.id === state.selected ? "true" : "false");
            row.tabIndex = 0;

            const grip = document.createElement("span");
            grip.className = P + "-row-grip";
            grip.textContent = "⠿";
            grip.setAttribute && grip.setAttribute("aria-hidden", "true");
            row.appendChild(grip);

            const icon = document.createElement("span");
            icon.className = P + "-row-icon";
            icon.textContent = region.type === "text" ? "T" : "□";
            row.appendChild(icon);

            const name = document.createElement("span");
            name.className = P + "-row-name";
            name.textContent = region.name || region.id;
            row.appendChild(name);

            const trash = document.createElement("button");
            trash.className = P + "-row-trash";
            trash.type = "button";
            trash.dataset.regionId = region.id;
            trash.title = "Delete this region";
            trash.setAttribute && trash.setAttribute("aria-label",
                                                     "Delete " + (region.name || region.id));
            trash.textContent = "✕";
            row.appendChild(trash);

            list.appendChild(row);
        });
        const empty = byId(IDS.empty);
        if (empty) empty.hidden = rows.length > 0;
        const count = byId(IDS.count);
        if (count) {
            count.textContent = rows.length
                ? rows.length + " of " + MAX_REGIONS
                : "";
        }
    }

    // InspectorController. Every field the editor has ever had, plus the four
    // numbers §8.3 asks for, and the selected region's name at the top so the
    // panel says what it is about before it says anything else.
    function paintInspector() {
        if (!state.working) return;
        const region = find(state.selected);
        const fields = [IDS.name, IDS.type, IDS.text, IDS.prompt, IDS.framing,
                        IDS.angle, IDS.literalPrefix, IDS.literalSuffix];
        fields.forEach(function (id) { enable(id, !!region); });
        enable(IDS.duplicate, !!region);
        enable(IDS.remove, !!region);
        [IDS.front, IDS.forward, IDS.back, IDS.bottom].forEach(function (id) {
            enable(id, !!region);
        });

        const auto = byId(IDS.autoHint);
        const grid = byId(IDS.grid);
        const title = byId(IDS.selectedName);
        if (auto) auto.checked = !!state.working.auto_position_hint;
        if (grid) grid.value = state.working.canvas.grid || "none";
        // Emptied rather than left showing the last region's answers: an
        // inspector still saying "Text" and "How the text should look" after
        // the text region it described was deleted is a panel lying about what
        // is selected.
        if (!region) {
            if (title) title.textContent = "nothing selected";
            [IDS.name, IDS.text, IDS.prompt, IDS.literalPrefix, IDS.literalSuffix]
                .forEach(function (id) { set(id, ""); });
            set(IDS.type, "obj");
            set(IDS.framing, "");
            set(IDS.angle, "");
            const blank = byId(IDS.promptLabel);
            if (blank) blank.textContent = "Region prompt";
            show(IDS.textField, false);
            show(IDS.framingField, true);
            show(IDS.angleField, true);
            return;
        }
        if (title) title.textContent = region.name || region.id;
        set(IDS.name, region.name);
        set(IDS.type, region.type);
        set(IDS.text, region.text);
        set(IDS.prompt, region.prompt);
        set(IDS.literalPrefix, region.literalPrefix);
        set(IDS.literalSuffix, region.literalSuffix);
        set(IDS.framing, region.framing);
        set(IDS.angle, region.angle);
        show(IDS.textField, region.type === "text");
        show(IDS.framingField, region.type !== "text");
        show(IDS.angleField, region.type !== "text");
        const label = byId(IDS.promptLabel);
        if (label) {
            label.textContent = region.type === "text"
                ? "How the text should look (not rendered as words)"
                : "Region prompt";
        }
    }

    function paintChrome() {
        if (!state.working) return;
        enable(IDS.undo, state.past.length > 0);
        enable(IDS.redo, state.future.length > 0);
        enable(IDS.clear, state.working.regions.length > 0);
        enable(IDS.add, state.working.regions.length < MAX_REGIONS);
        const draw = byId(IDS.draw);
        if (draw) {
            draw.classList.toggle("active", state.drawing);
            draw.textContent = state.drawing ? "Drawing — cancel" : "Draw region";
            draw.setAttribute && draw.setAttribute("aria-pressed",
                                                   state.drawing ? "true" : "false");
        }
        const canvas = byId(IDS.canvas);
        if (canvas && canvas.classList) canvas.classList.toggle("drawing", state.drawing);
        const confirm = byId(IDS.confirm);
        if (confirm) confirm.hidden = !state.confirming;
        const full = byId(IDS.full);
        if (full) {
            const on = fullscreenOn();
            full.textContent = on ? "Exit full screen" : "Full screen";
            full.classList.toggle("active", on);
            full.setAttribute && full.setAttribute("aria-pressed", on ? "true" : "false");
        }
    }

    function set(id, value) {
        const field = byId(id);
        if (field && field.value !== value) field.value = value;
    }

    function show(id, visible) {
        const field = byId(id);
        if (field) field.hidden = !visible;
    }

    // The one transient line: a region cap reached, a Clear All that can be
    // undone. Non-blocking by construction -- there is nothing to dismiss, so
    // there is nothing that can eat somebody's layout while they dismiss it.
    function say(text) {
        const element = byId(IDS.message);
        if (!element) return;
        element.textContent = text || "";
        element.hidden = !text;
    }

    // ------------------------------------------------------------------ //
    // SelectionController -- one selection, three surfaces
    // ------------------------------------------------------------------ //

    function select(id) {
        state.selected = id || "";
        state.editing = "";
        paint();
    }

    // §6.2: deleting the selection picks the next row, or the previous one, or
    // nothing -- never "whatever ends up first", which on a reordered list is a
    // different box every time.
    function neighbour(id) {
        const rows = ordered().slice().reverse();
        const at = rows.findIndex(function (region) { return region.id === id; });
        if (at < 0) return "";
        const next = rows[at + 1] || rows[at - 1];
        return next ? next.id : "";
    }

    // ------------------------------------------------------------------ //
    // Editing
    // ------------------------------------------------------------------ //

    function create(bbox, history) {
        if (state.working.regions.length >= MAX_REGIONS) {
            say("That is as many regions as one layout carries (" + MAX_REGIONS + ").");
            return null;
        }
        if (history !== false) mark("");
        const highest = state.working.regions.reduce(function (top, region) {
            return Math.max(top, region.z);
        }, -1);
        const region = {
            id: nextId(),
            name: "Region " + (state.working.regions.length + 1),
            type: "obj",
            bbox: bbox,
            prompt: "",
            literalPrefix: "",
            literalSuffix: "",
            text: "",
            framing: "",
            angle: "",
            z: highest + 1,
        };
        state.working.regions.push(region);
        say("");
        select(region.id);
        const field = byId(IDS.prompt);
        if (field && typeof field.focus === "function") field.focus();
        return region;
    }

    // §7.1: a touch user does not have to perform a precision drag merely to
    // get a box. Centred, a third of the frame, and cascaded a little so that
    // pressing it twice does not hide the first one exactly underneath.
    function add() {
        const drift = (state.working.regions.length % 5) * 28 - 56;
        const low = clamp((SCALE - ADD_SIZE) / 2 + drift, 0, SCALE - ADD_SIZE);
        return create(orient([low, low, low + ADD_SIZE, low + ADD_SIZE]));
    }

    function remove(id) {
        const target = id || state.selected;
        if (!target || !find(target)) return;
        mark("");
        const next = target === state.selected ? neighbour(target) : state.selected;
        state.working.regions = state.working.regions.filter(function (region) {
            return region.id !== target;
        });
        select(next);
    }

    // §10.2. One action, and Undo brings all of them back -- which is what
    // makes it safe to offer without a confirmation dialog in the way.
    function clear() {
        if (!state.working.regions.length) return;
        const many = state.working.regions.length;
        mark("");
        state.working.regions = [];
        select("");
        say("Cleared " + many + (many === 1 ? " region" : " regions") + ". Undo brings "
            + (many === 1 ? "it" : "them") + " back.");
    }

    function duplicate() {
        const region = find(state.selected);
        if (!region) return;
        if (state.working.regions.length >= MAX_REGIONS) {
            say("That is as many regions as one layout carries (" + MAX_REGIONS + ").");
            return;
        }
        mark("");
        const shift = 20;
        const moved = orient([
            Math.min(SCALE - MIN_SIZE, region.bbox[0] + shift),
            Math.min(SCALE - MIN_SIZE, region.bbox[1] + shift),
            Math.min(SCALE, region.bbox[2] + shift),
            Math.min(SCALE, region.bbox[3] + shift),
        ]) || region.bbox.slice();
        const copy = Object.assign({}, region, {
            id: nextId(),
            name: (region.name || region.id) + " copy",
            bbox: moved,
            z: region.z + 1,
        });
        state.working.regions.push(copy);
        select(copy.id);
    }

    // z is renumbered from the visible order after every move, so "forward"
    // means one place forward however the numbers arrived -- including from a
    // hand-edited document where three regions all claim z 0.
    //
    // Reordering the list changes z. Selecting in it does not. §9.
    function restack(direction) {
        const list = ordered();
        const at = list.findIndex(function (region) { return region.id === state.selected; });
        if (at < 0) return;
        let to = at + direction;
        if (direction === Infinity) to = list.length - 1;
        if (direction === -Infinity) to = 0;
        if (to < 0 || to >= list.length || to === at) return;
        mark("");
        const moved = list.slice();
        moved.splice(to, 0, moved.splice(at, 1)[0]);
        moved.forEach(function (region, index) { region.z = index; });
        paint();
    }

    // Drop `moved` next to `target`, and renumber z from the order that leaves.
    //
    // Renumbered rather than adjusted, for the same reason restack() renumbers:
    // a layout hand-edited elsewhere can arrive with three regions all claiming
    // z 0, and "one place later" has to mean one place later in what somebody
    // is looking at rather than in arithmetic nobody can see.
    function reorder(movedId, targetId, before) {
        if (!state.working) return false;
        const list = ordered();
        const from = list.findIndex(function (region) { return region.id === movedId; });
        if (from < 0) return false;
        const region = list[from];

        let to;
        if (targetId === null || targetId === undefined || targetId === "") {
            to = before ? 0 : list.length - 1;
        } else {
            const at = list.findIndex(function (entry) { return entry.id === targetId; });
            if (at < 0 || targetId === movedId) return false;
            to = before ? at : at + 1;
            if (from < to) to -= 1;
        }
        if (to === from) return false;

        mark("");
        const moved = list.slice();
        moved.splice(from, 1);
        moved.splice(to, 0, region);
        moved.forEach(function (entry, index) { entry.z = index; });
        paint();
        return true;
    }

    // Above the middle of the row it is over, or below it. The only geometry
    // this needs: everything else about the drop is list order.
    function dropBefore(event, row) {
        try {
            const box = row.getBoundingClientRect();
            return (event.clientY - box.top) < (box.height / 2);
        } catch (error) {
            return false;
        }
    }

    function nudge(dx, dy) {
        const region = find(state.selected);
        if (!region) return;
        mark("nudge:" + region.id);
        const width = region.bbox[2] - region.bbox[0];
        const height = region.bbox[3] - region.bbox[1];
        const x0 = clamp(region.bbox[0] + dx, 0, SCALE - width);
        const y0 = clamp(region.bbox[1] + dy, 0, SCALE - height);
        region.bbox = [x0, y0, x0 + width, y0 + height];
        paint();
    }

    // ------------------------------------------------------------------ //
    // PointerController -- one path for mouse, finger and pen
    // ------------------------------------------------------------------ //

    function point(event) {
        const canvas = byId(IDS.canvas);
        if (!canvas) return {x: 0, y: 0};
        const rect = canvas.getBoundingClientRect();
        if (!rect.width || !rect.height) return {x: 0, y: 0};
        return {
            x: clamp(Math.round((event.clientX - rect.left) / rect.width * SCALE), 0, SCALE),
            y: clamp(Math.round((event.clientY - rect.top) / rect.height * SCALE), 0, SCALE),
        };
    }

    // §5.2. The drag follows the contact that started it even when the contact
    // leaves the frame, which is what makes the document-level mousemove and
    // mouseup this file used to install unnecessary rather than merely untidy.
    function grab(event) {
        const canvas = byId(IDS.canvas);
        state.pointer = event && event.pointerId !== undefined ? event.pointerId : null;
        if (!canvas || typeof canvas.setPointerCapture !== "function") return;
        try {
            canvas.setPointerCapture(event.pointerId);
        } catch (error) {
            // A synthetic pointer, or one already released. The drag still
            // works; it just stops following a contact that leaves the frame.
        }
    }

    function release() {
        const canvas = byId(IDS.canvas);
        if (canvas && typeof canvas.releasePointerCapture === "function"
            && state.pointer !== null) {
            try {
                canvas.releasePointerCapture(state.pointer);
            } catch (error) {
                // Already gone; nothing to release.
            }
        }
        state.pointer = null;
    }

    // §5.4: one primary pointer edits. A second finger on the frame is ignored
    // rather than fighting the first for the same region.
    function mine(event) {
        if (!state.drag) return false;
        if (state.pointer === null || event.pointerId === undefined) return true;
        return event.pointerId === state.pointer;
    }

    function onDown(event) {
        if (!state.open || !state.working) return;
        if (event.isPrimary === false) return;
        if (state.drag) return;
        if (typeof event.button === "number" && event.button > 0) return;
        state.editing = "";
        const at = point(event);
        const target = event.target || {};
        const handle = target.closest ? target.closest("." + P + "-handle") : null;
        const onProxy = target.closest ? target.closest("." + P + "-proxy") : null;
        const onRegion = target.closest ? target.closest("." + P + "-region") : null;

        if (state.drawing) {
            prevent(event);
            grab(event);
            state.drag = {kind: "draw", from: at, to: at, before: snapshot()};
            paintGhost();
            return;
        }
        if (handle && onProxy) {
            const chosen = find(state.selected);
            if (!chosen) return;
            prevent(event);
            grab(event);
            state.drag = {kind: "resize", corner: handle.dataset.corner, id: chosen.id,
                          origin: chosen.bbox.slice(), from: at, before: snapshot()};
            return;
        }
        if (onProxy) {
            const chosen = find(state.selected);
            if (!chosen) return;
            prevent(event);
            grab(event);
            state.drag = {kind: "move", id: chosen.id, origin: chosen.bbox.slice(),
                          from: at, before: snapshot()};
            return;
        }
        if (onRegion) {
            prevent(event);
            grab(event);
            select(onRegion.dataset.regionId);
            const chosen = find(state.selected);
            if (!chosen) return;
            state.drag = {kind: "move", id: chosen.id, origin: chosen.bbox.slice(),
                          from: at, before: snapshot()};
            return;
        }
        select("");
    }

    function onMove(event) {
        if (!mine(event)) return;
        const at = point(event);
        if (state.drag.kind === "draw") {
            state.drag.to = at;
            paintGhost();
            return;
        }
        const region = find(state.drag.id);
        if (!region) return;
        const dx = at.x - state.drag.from.x;
        const dy = at.y - state.drag.from.y;
        const origin = state.drag.origin;
        if (state.drag.kind === "move") {
            const width = origin[2] - origin[0];
            const height = origin[3] - origin[1];
            const x0 = clamp(origin[0] + dx, 0, SCALE - width);
            const y0 = clamp(origin[1] + dy, 0, SCALE - height);
            region.bbox = [x0, y0, x0 + width, y0 + height];
        } else {
            region.bbox = resized(origin, state.drag.corner, dx, dy);
        }
        paintFrame();
    }

    // §8.2: an edge dragged past its opposite clamps at the minimum size rather
    // than turning the box inside out. Nothing here can produce an inverted
    // bbox, so nothing downstream has to cope with one.
    function resized(origin, corner, dx, dy) {
        let [x0, y0, x1, y1] = origin;
        if (corner.indexOf("w") >= 0) x0 = clamp(origin[0] + dx, 0, origin[2] - MIN_SIZE);
        if (corner.indexOf("e") >= 0) x1 = clamp(origin[2] + dx, origin[0] + MIN_SIZE, SCALE);
        if (corner.indexOf("n") >= 0) y0 = clamp(origin[1] + dy, 0, origin[3] - MIN_SIZE);
        if (corner.indexOf("s") >= 0) y1 = clamp(origin[3] + dy, origin[1] + MIN_SIZE, SCALE);
        return orient([x0, y0, x1, y1]) || origin.slice();
    }

    function onUp(event) {
        if (state.drag && !mine(event)) return;
        const drag = state.drag;
        state.drag = null;
        release();
        clearGhost();
        if (!drag) return;
        if (drag.kind === "draw") {
            state.drawing = false;
            const made = orient([drag.from.x, drag.from.y, drag.to.x, drag.to.y]);
            // A tap is not a region, and a two-pixel drag is a tap with a shaky
            // hand. Both are dropped rather than turned into a box the
            // compositor would refuse later, where the reason would arrive on
            // the finished image instead of under the finger.
            if (made && (made[2] - made[0]) >= MIN_SIZE && (made[3] - made[1]) >= MIN_SIZE) {
                keep(drag.before);
                state.editing = "";
                create(made, false);
                return;
            }
            paint();
            return;
        }
        record(drag.before);
        paint();
    }

    function onCancel(event) {
        if (state.drag && !mine(event)) return;
        const drag = state.drag;
        state.drag = null;
        release();
        clearGhost();
        if (!drag) return;
        // A cancelled pointer is the system taking the contact away -- a phone
        // call, a gesture the browser decided was a scroll. The edit goes back
        // to where it started rather than stopping halfway.
        if (drag.kind !== "draw") restore(drag.before);
        state.drawing = false;
        paint();
    }

    function paintGhost() {
        const holder = byId(IDS.regions);
        if (!holder || !state.drag || state.drag.kind !== "draw") return;
        let ghost = holder.querySelector("." + P + "-ghost");
        if (!ghost) {
            ghost = document.createElement("div");
            ghost.className = P + "-ghost";
            holder.appendChild(ghost);
        }
        const from = state.drag.from;
        const to = state.drag.to;
        ghost.style.left = (Math.min(from.x, to.x) / SCALE * 100) + "%";
        ghost.style.top = (Math.min(from.y, to.y) / SCALE * 100) + "%";
        ghost.style.width = (Math.abs(to.x - from.x) / SCALE * 100) + "%";
        ghost.style.height = (Math.abs(to.y - from.y) / SCALE * 100) + "%";
    }

    function clearGhost() {
        const holder = byId(IDS.regions);
        const ghost = holder ? holder.querySelector("." + P + "-ghost") : null;
        if (ghost && ghost.parentNode) ghost.parentNode.removeChild(ghost);
    }

    // ------------------------------------------------------------------ //
    // Opening, saving, leaving
    // ------------------------------------------------------------------ //

    function open() {
        const workspace = byId(IDS.workspace);
        if (!workspace) return;
        const read = readLayout();
        if (read === null) {
            // The panel already says why in words. Opening onto an empty canvas
            // would look like the layout had been deleted, so the editor stays
            // shut and the document stays exactly as it is.
            console.warn("Model Chain: the saved spatial layout was not readable; the "
                         + "editor was not opened");
            return;
        }
        state.working = read;
        state.savedCanvas = {width: Number(read.canvas.width) || 0,
                             height: Number(read.canvas.height) || 0};
        state.counter = read.regions.reduce(function (top, region) {
            const number = parseInt(String(region.id).replace(/^r/, ""), 10);
            return Number.isFinite(number) ? Math.max(top, number) : top;
        }, 0);
        state.selected = read.regions.length ? ordered()[ordered().length - 1].id : "";
        state.drawing = false;
        state.confirming = false;
        state.editing = "";
        state.pendingSave = false;
        state.typing = false;
        // What the document already holds. Opening the editor commits nothing:
        // an Auto Save that fired on open would put a "changed" mark against a
        // layout somebody only looked at.
        state.committed = serialize();
        state.past.length = 0;
        state.future.length = 0;
        state.open = true;
        workspace.hidden = false;
        say("");
        zoomTo(1);
        reframe();
        // Taken after reframe, so that opening an editor and leaving it alone
        // is not "changed" merely because the frame is a different size than
        // the day the layout was drawn.
        state.baseline = serialize();
        paint();
        const canvas = byId(IDS.canvas);
        if (canvas && typeof canvas.focus === "function") canvas.focus();
    }

    function close() {
        leaveFullscreen();
        const workspace = byId(IDS.workspace);
        if (workspace) workspace.hidden = true;
        state.open = false;
        state.drawing = false;
        state.confirming = false;
        state.drag = null;
        state.working = null;
        state.past.length = 0;
        state.future.length = 0;
        release();
    }

    function save() {
        const box = stateBox();
        if (box && state.working) {
            state.committed = serialize();
            publish(box, state.committed);
        }
        close();
        // The compact canvas is a view of the same document, so it is stale the
        // moment the editor writes one. Forced, because "somebody just saved a
        // layout" is exactly the case the ordinary guard refuses.
        compactLoad(true);
    }

    // §14. Unchanged leaves at once; changed asks, in the page, with two
    // buttons. A browser confirm() would be a modal dialog reintroduced into an
    // editor whose whole point was to stop being one.
    function leave() {
        if (!state.working) { close(); return; }
        if (serialize() === state.baseline) { close(); return; }
        state.confirming = true;
        paintChrome();
    }

    // ------------------------------------------------------------------ //
    // Wiring
    // ------------------------------------------------------------------ //

    // Elements only. A dataset flag is what makes re-wiring idempotent, and
    // `document` has no dataset -- so passing it here would add a fresh listener
    // on every onAfterUiUpdate, which fires often. The one document-level
    // listener is installed once, under its own flag, in wire().
    function once(element, event, handler, tag) {
        if (!element || !element.dataset) return;
        // `tag` separates two controllers that legitimately listen to the same
        // event on the same element. The compact canvas and the editor both
        // follow the txt2img size fields, and without it whichever wired first
        // would claim the flag and silently take the other's listener away.
        const flag = "mcKreaSpatial" + (tag || "") + event;
        if (element.dataset[flag]) return;
        element.dataset[flag] = "1";
        element.addEventListener(event, handler);
    }

    // The workspace, kept singular.
    //
    // It is no longer moved anywhere: it lives where Gradio put it, in ordinary
    // document flow, which is most of what this refactor is. What survives from
    // the overlay days is the duplicate guard. If Gradio rebuilds the HTML
    // component, the rebuilt copy arrives carrying every id the old one has, so
    // whichever `byId` found first would be a coin toss between a wired
    // workspace and an unwired one. The newest copy wins -- it is the one
    // Gradio is now managing -- and the rest are removed.
    function claim() {
        let found;
        try {
            found = Array.prototype.slice.call(
                (root().querySelectorAll ? root() : document)
                    .querySelectorAll("#" + IDS.workspace));
        } catch (error) {
            found = [];
        }
        if (!found.length) return null;
        const live = found[found.length - 1];
        found.forEach(function (element) {
            if (element !== live && element.parentNode) {
                element.parentNode.removeChild(element);
            }
        });
        state.workspace = live;
        return live;
    }

    // A text field commits to history once per burst of typing, not once per
    // keystroke: `mark` is given a token naming the field and the region, and
    // repeats it until something else happens.
    function field(id, apply, repaint) {
        const element = byId(id);
        once(element, "input", function () {
            const region = find(state.selected);
            if (!region) return;
            // Typing is not a finished edit. The commit happens on `change`,
            // which the browser fires when the cursor leaves the field -- one
            // round trip per field rather than one per keystroke, and the same
            // moment somebody would expect a text box to have taken.
            state.typing = true;
            try {
                mark(id + ":" + region.id);
                apply(region, element.value);
                if (repaint) paint();
                else paintChrome();
            } finally {
                state.typing = false;
            }
        });
        once(element, "change", function () {
            const region = find(state.selected);
            if (!region) return;
            mark(id + ":" + region.id);
            apply(region, element.value);
            state.editing = "";
            paint();
        });
    }

    function press(id, handler) {
        once(clickable(id) || byId(id), "click", function (event) {
            prevent(event);
            handler();
        });
    }

    function wire() {
        try {
            // Before the editor's own claim, and outside its `state.wired`
            // guard: the compact canvas lives in the pipeline rather than in
            // the workspace, so it exists on pages where the workspace has not
            // been claimed, and it has to be redrawn after every Gradio update
            // that could have replaced the layout under it.
            try {
                wireCompact();
            } catch (error) {
                console.error("Model Chain: the compact spatial canvas failed", error);
            }

            if (!claim()) return;

            press(IDS.open, open);
            press(IDS.add, function () { add(); });
            press(IDS.draw, function () {
                if (!state.open) return;
                state.drawing = !state.drawing;
                if (!state.drawing) { state.drag = null; clearGhost(); }
                paintChrome();
            });
            press(IDS.clear, clear);
            press(IDS.undo, undo);
            press(IDS.redo, redo);
            press(IDS.duplicate, duplicate);
            press(IDS.remove, function () { remove(); });
            press(IDS.front, function () { restack(Infinity); });
            press(IDS.forward, function () { restack(1); });
            press(IDS.back, function () { restack(-1); });
            press(IDS.bottom, function () { restack(-Infinity); });
            press(IDS.zoomFit, function () { zoomTo(1); });
            press(IDS.zoomIn, function () { zoomStep(1); });
            press(IDS.zoomOut, function () { zoomStep(-1); });
            press(IDS.full, toggleFullscreen);
            press(IDS.save, save);
            press(IDS.cancel, leave);
            press(IDS.discard, close);
            press(IDS.keep, function () {
                state.confirming = false;
                paintChrome();
            });

            // The region list is rebuilt on every paint, so its rows cannot
            // carry their own listeners without leaking one per repaint. One
            // delegated listener on the container outlives every row.
            once(byId(IDS.list), "click", function (event) {
                const target = event.target || {};
                const trash = target.closest ? target.closest("." + P + "-row-trash") : null;
                if (trash) {
                    prevent(event);
                    remove(trash.dataset.regionId);
                    return;
                }
                const row = target.closest ? target.closest("." + P + "-row") : null;
                if (row) select(row.dataset.regionId);
            });
            once(byId(IDS.list), "keydown", function (event) {
                const target = event.target || {};
                const row = target.closest ? target.closest("." + P + "-row") : null;
                if (!row) return;
                if (event.key === "Enter" || event.key === " ") {
                    prevent(event);
                    select(row.dataset.regionId);
                    return;
                }
                // The same move, without a pointer. Dragging is not something
                // every hand or every input device can do, and a reorder that
                // only a mouse can reach is a reorder half the people using
                // this cannot make.
                if (!event.altKey) return;
                const step = event.key === "ArrowUp" ? -1
                    : event.key === "ArrowDown" ? 1 : 0;
                if (!step) return;
                prevent(event);
                const list = ordered();
                const at = list.findIndex(function (entry) {
                    return entry.id === row.dataset.regionId;
                });
                const next = list[at + step];
                if (!next) return;
                reorder(row.dataset.regionId, next.id, step < 0);
                const moved = byId(IDS.list) &&
                    byId(IDS.list).querySelector('[data-region-id="'
                                                + row.dataset.regionId + '"]');
                if (moved && moved.focus) moved.focus();
            });

            // Reordering by pointer. Four delegated listeners on the container,
            // because the rows are rebuilt on every paint and a listener per row
            // would be a listener per row per repaint.
            //
            // `dataTransfer` is set because Firefox refuses to start a drag
            // without it; nothing reads it back. The id being moved is held here
            // instead, where it cannot be replaced by a drag that started
            // somewhere else on the page.
            once(byId(IDS.list), "dragstart", function (event) {
                const row = event.target && event.target.closest
                    ? event.target.closest("." + P + "-row") : null;
                if (!row) return;
                state.dragging = row.dataset.regionId;
                row.classList.add("dragging");
                if (event.dataTransfer) {
                    event.dataTransfer.effectAllowed = "move";
                    try {
                        event.dataTransfer.setData("text/plain", row.dataset.regionId);
                    } catch (error) {
                        // Some browsers refuse this outside a user gesture. The
                        // drag still works; only the payload nobody reads is
                        // missing.
                    }
                }
            });

            once(byId(IDS.list), "dragover", function (event) {
                if (!state.dragging) return;
                prevent(event);
                if (event.dataTransfer) event.dataTransfer.dropEffect = "move";
                const row = event.target && event.target.closest
                    ? event.target.closest("." + P + "-row") : null;
                const list = byId(IDS.list);
                if (!list) return;
                list.querySelectorAll("." + P + "-row").forEach(function (entry) {
                    entry.classList.remove("drop-before", "drop-after");
                });
                if (!row || row.dataset.regionId === state.dragging) return;
                row.classList.add(dropBefore(event, row) ? "drop-before" : "drop-after");
            });

            once(byId(IDS.list), "drop", function (event) {
                if (!state.dragging) return;
                prevent(event);
                const row = event.target && event.target.closest
                    ? event.target.closest("." + P + "-row") : null;
                const moving = state.dragging;
                state.dragging = "";
                if (!row) {
                    // Dropped on the empty space under the last row: the end of
                    // the list is the answer somebody means by that.
                    reorder(moving, "", false);
                    return;
                }
                reorder(moving, row.dataset.regionId, dropBefore(event, row));
            });

            once(byId(IDS.list), "dragend", function () {
                state.dragging = "";
                const list = byId(IDS.list);
                if (!list) return;
                list.querySelectorAll("." + P + "-row").forEach(function (entry) {
                    entry.classList.remove("dragging", "drop-before", "drop-after");
                });
            });

            once(byId(IDS.autoHint), "change", function (event) {
                if (!state.working) return;
                mark("");
                state.working.auto_position_hint = !!event.target.checked;
                paintChrome();
            });
            once(byId(IDS.grid), "change", function (event) {
                if (!state.working) return;
                mark("");
                state.working.canvas.grid = String(event.target.value || "none");
                paint();
            });

            field(IDS.name, function (region, value) { region.name = value; }, true);
            field(IDS.type, function (region, value) {
                region.type = value === "text" ? "text" : "obj";
            }, true);
            field(IDS.text, function (region, value) { region.text = value; });
            field(IDS.prompt, function (region, value) { region.prompt = value; });
            field(IDS.literalPrefix, function (region, value) {
                region.literalPrefix = value;
            });
            field(IDS.literalSuffix, function (region, value) {
                region.literalSuffix = value;
            });
            field(IDS.framing, function (region, value) { region.framing = value; });
            field(IDS.angle, function (region, value) { region.angle = value; });

            // §5.1/§5.2: one pointer path, captured on the frame, so a drag
            // that leaves the frame is still this drag and a finger is not a
            // second implementation.
            const canvas = byId(IDS.canvas);
            once(canvas, "pointerdown", onDown);
            once(canvas, "pointermove", onMove);
            once(canvas, "pointerup", onUp);
            once(canvas, "pointercancel", onCancel);
            // Only where Pointer Events are missing entirely. Modern WebKit,
            // Blink and Gecko all have them; this is for the one embedded
            // browser that does not, and it is deliberately the old path and
            // not a second maintained one.
            if (typeof window !== "undefined" && !("PointerEvent" in window)) {
                once(canvas, "mousedown", onDown);
                once(canvas, "mousemove", onMove);
                once(canvas, "mouseup", onUp);
            }

            // The frame follows the generation size while the editor is open,
            // rather than only at the moment it was opened. §4.2 asks for the
            // component values; an extension cannot read a Gradio component
            // without a server round-trip, and a round-trip is the one thing
            // this file is not allowed to make a generation wait for -- so the
            // host's own number fields are read, and listened to.
            ["txt2img_width", "txt2img_height"].forEach(function (id) {
                const holder = byId(id);
                const input = holder ? holder.querySelector("input") : null;
                once(input, "change", function () { if (state.open) { reframe(); paint(); } });
                once(input, "input", function () { if (state.open) { reframe(); paint(); } });
            });

            // Once, ever. A key pressed with focus anywhere in the page still
            // has to be heard, so this one lives on the document -- which makes
            // it the one that a dataset flag cannot guard.
            if (!state.listening) {
                state.listening = true;
                document.addEventListener("keydown", onKey);
                // The browser owns Escape while the Fullscreen API is in
                // charge, so the only way to know it was pressed is to be told.
                document.addEventListener("fullscreenchange", function () {
                    if (state.open) paintChrome();
                });
            }

            // Gradio rebuilding the tab under an open editor would otherwise
            // leave it open in `state` and hidden on screen.
            if (state.open && state.working) {
                const live = byId(IDS.workspace);
                if (live) live.hidden = false;
                zoomTo(state.zoom);
                reframe();
                paint();
            }

            state.wired = true;
        } catch (error) {
            console.error("Model Chain: the Spatial Layout wiring failed", error);
        }
    }

    // ------------------------------------------------------------------ //
    // CompactCanvasController -- position correction, in the pipeline
    // ------------------------------------------------------------------ //
    //
    // Section 6.2. One verb: move a box. No create, no delete, no resize, no
    // rename, no type change, no stacking, no framing. Every one of those still
    // exists in the full editor, one button away, and none of them is reachable
    // from here -- which is what keeps this a shortcut into the same layout
    // state rather than a second layout system competing for the same document.
    //
    // Hit testing is the browser's. Regions are elements appended in paint
    // order, so `event.target.closest(...)` returns the topmost region under
    // the pointer by construction: no z-order arithmetic, no proxy, and no way
    // for the answer to disagree with what somebody can see.
    //
    // The full editor's `state` is untouched here. This has a working document
    // of its own, because the two canvases are open at different times and a
    // shared one would mean the compact canvas writing into a document the
    // editor was in the middle of editing.

    const COMPACT_UNDO = 25;   // §6.4 asks for a bounded history; a move is small

    const compact = {
        working: null,      // the layout on the compact canvas, or null
        past: [],           // snapshots, newest last
        dirty: false,       // moved since the last commit, with Auto Save off
        drag: null,         // {id, from, origin}
        pointer: null,
    };

    const COMPACT_IDS = {
        host: P + "-compact",
        frame: P + "-compact-frame",
        regions: P + "-compact-regions",
        empty: P + "-compact-empty",
        note: P + "-compact-note",
        undo: P + "-compact-undo",
        commit: P + "-compact-commit",
        autosave: P + "-autosave",
    };

    function autoSaveOn() {
        const holder = byId(COMPACT_IDS.autosave);
        // The attribute selector first because a Gradio Checkbox has a label
        // and could grow another input beside it; a plain `input` after,
        // because that is what every other lookup in this file uses and it is
        // what a stripped-down DOM answers.
        const box = holder
            ? (holder.querySelector("input[type=checkbox]")
               || holder.querySelector("input"))
            : null;
        // Default on, which is what the server-side default says too. A missing
        // control must not silently turn committing off -- that is the setting
        // whose absence loses work.
        return box ? !!box.checked : true;
    }

    function compactLoad(force) {
        // Never over an unsaved move or a drag in flight. `onAfterUiUpdate`
        // fires often and for reasons that have nothing to do with this canvas,
        // and reloading under somebody's finger would undo the move they are
        // making.
        if (!force && (compact.drag || compact.dirty)) return;
        const read = readLayout();
        if (read === null) return;     // unreadable; Python says so in words
        compact.working = read;
        compact.past.length = 0;
        compact.dirty = false;
        compactPaint();
    }

    // One history entry, and the one place the bound is applied. Takes the
    // regions to record rather than reading the live ones, because what a drag
    // has to remember is the layout as it was *before* the drag -- which is no
    // longer anywhere by the time the pointer comes up.
    function compactKeep(regions) {
        compact.past.push(JSON.stringify(regions));
        if (compact.past.length > COMPACT_UNDO) compact.past.shift();
    }

    function compactFind(id) {
        if (!compact.working) return null;
        return compact.working.regions.filter(function (region) {
            return region.id === id;
        })[0] || null;
    }

    function compactPaint() {
        const holder = byId(COMPACT_IDS.regions);
        const frame = byId(COMPACT_IDS.frame);
        const empty = byId(COMPACT_IDS.empty);
        if (!holder || !compact.working) return;

        const size = hostSize();
        if (frame && size.width > 0 && size.height > 0) {
            frame.style.aspectRatio = size.width + " / " + size.height;
            setVar(frame, "--mc-ar-w", String(size.width));
            setVar(frame, "--mc-ar-h", String(size.height));
        }

        holder.textContent = "";
        const regions = orderedIn(compact.working);
        regions.forEach(function (region) {
            const element = document.createElement("div");
            element.className = P + "-compact-region";
            if (region.type === "text") element.className += " " + P + "-compact-text";
            element.dataset.regionId = region.id;
            element.title = region.name || region.id;
            const [x0, y0, x1, y1] = region.bbox;
            element.style.left = (x0 / SCALE * 100) + "%";
            element.style.top = (y0 / SCALE * 100) + "%";
            element.style.width = ((x1 - x0) / SCALE * 100) + "%";
            element.style.height = ((y1 - y0) / SCALE * 100) + "%";
            const label = document.createElement("span");
            label.className = P + "-compact-label";
            label.textContent = region.name || region.id;
            element.appendChild(label);
            holder.appendChild(element);
        });

        if (empty) empty.hidden = regions.length > 0;
        compactNote();
    }

    function compactNote() {
        const note = byId(COMPACT_IDS.note);
        if (!note) return;
        if (!compact.working || !compact.working.regions.length) {
            note.textContent = "";
            note.classList.remove(P + "-unsaved");
            return;
        }
        if (compact.dirty) {
            // §6.3. Said plainly and in the page, because the state it
            // describes is one where what is on screen is *not* yet what the
            // next Generate composes -- the one place in this refactor where
            // that is true, and therefore the one place it has to be said.
            note.textContent = "Unsaved working layout — press Save working "
                + "layout, or switch Auto Save on.";
            note.classList.add(P + "-unsaved");
            return;
        }
        note.classList.remove(P + "-unsaved");
        note.textContent = "Drag a box to move it. Edit Layout… for everything else.";
    }

    function compactCommit() {
        const box = stateBox();
        if (!box || !compact.working) return;
        publish(box, serializeIn(compact.working));
        compact.dirty = false;
        compactNote();
    }

    function compactPoint(event) {
        const frame = byId(COMPACT_IDS.frame);
        if (!frame || typeof frame.getBoundingClientRect !== "function") return null;
        const box = frame.getBoundingClientRect();
        if (!(box.width > 0 && box.height > 0)) return null;
        return {
            x: (event.clientX - box.left) / box.width * SCALE,
            y: (event.clientY - box.top) / box.height * SCALE,
        };
    }

    function compactDown(event) {
        if (!compact.working) return;
        if (event.button !== undefined && event.button !== 0) return;
        const target = event.target || {};
        const element = target.closest
            ? target.closest("." + P + "-compact-region") : null;
        if (!element) return;                 // the frame itself does nothing

        const region = compactFind(element.dataset.regionId);
        const at = compactPoint(event);
        if (!region || !at) return;

        prevent(event);
        compact.drag = {id: region.id, from: at, origin: region.bbox.slice()};
        compact.pointer = event.pointerId;
        // Keeps the drag attached to the finger that started it even when the
        // finger leaves the frame, which is what makes document-level listeners
        // unnecessary rather than merely unfashionable.
        if (typeof event.target.setPointerCapture === "function"
            && event.pointerId !== undefined) {
            try { event.target.setPointerCapture(event.pointerId); } catch (error) { /* not fatal */ }
        }
    }

    function compactMove(event) {
        if (!compact.drag) return;
        const at = compactPoint(event);
        if (!at) return;
        const region = compactFind(compact.drag.id);
        if (!region) return;

        prevent(event);
        const [x0, y0, x1, y1] = compact.drag.origin;
        const width = x1 - x0;
        const height = y1 - y0;
        // Clamped to the frame rather than allowed off it, and clamped by
        // *offset* so the box keeps its size: a drag that resized what it was
        // moving would be changing something nobody asked to change.
        const left = clamp(Math.round(x0 + (at.x - compact.drag.from.x)), 0, SCALE - width);
        const top = clamp(Math.round(y0 + (at.y - compact.drag.from.y)), 0, SCALE - height);
        region.bbox = [left, top, left + width, top + height];
        compactPaint();
    }

    function compactUp(event) {
        if (!compact.drag) return;
        const region = compactFind(compact.drag.id);
        const moved = region
            && JSON.stringify(region.bbox) !== JSON.stringify(compact.drag.origin);
        const origin = compact.drag.origin;
        const id = compact.drag.id;

        compact.drag = null;
        compact.pointer = null;
        if (event && event.target
            && typeof event.target.releasePointerCapture === "function"
            && event.pointerId !== undefined) {
            try { event.target.releasePointerCapture(event.pointerId); } catch (error) { /* not fatal */ }
        }

        // One history entry per completed drag, and a drag that moved nothing
        // is not a completed anything.
        if (!moved) return;
        compactKeep(compact.working.regions.map(function (entry) {
            return entry.id === id
                ? Object.assign({}, entry, {bbox: origin})
                : entry;
        }));

        if (autoSaveOn()) {
            compactCommit();
        } else {
            compact.dirty = true;
            compactNote();
        }
    }

    function compactCancel() {
        if (!compact.drag) return;
        const region = compactFind(compact.drag.id);
        if (region) region.bbox = compact.drag.origin.slice();
        compact.drag = null;
        compact.pointer = null;
        compactPaint();
    }

    function compactUndo() {
        if (!compact.working || !compact.past.length) return;
        let read;
        try {
            read = JSON.parse(compact.past.pop());
        } catch (error) {
            return;
        }
        compact.working.regions = (read || []).map(normalise).filter(Boolean);
        compactPaint();
        // §6.4: with Auto Save on, the undone position is committed too --
        // otherwise Undo would leave the screen and the generation disagreeing,
        // which is the one thing Auto Save exists to prevent.
        if (autoSaveOn()) {
            compactCommit();
        } else {
            compact.dirty = true;
            compactNote();
        }
    }

    function wireCompact() {
        const frame = byId(COMPACT_IDS.frame);
        if (!frame) return;

        once(frame, "pointerdown", compactDown, "Compact");
        once(frame, "pointermove", compactMove, "Compact");
        once(frame, "pointerup", compactUp, "Compact");
        once(frame, "pointercancel", compactCancel, "Compact");
        // Only where Pointer Events are missing entirely; deliberately the old
        // path and not a second maintained one.
        if (typeof window !== "undefined" && !("PointerEvent" in window)) {
            once(frame, "mousedown", compactDown, "Compact");
            once(frame, "mousemove", compactMove, "Compact");
            once(frame, "mouseup", compactUp, "Compact");
        }

        once(clickable(COMPACT_IDS.undo), "click", function (event) {
            prevent(event);
            compactUndo();
        }, "Compact");
        once(clickable(COMPACT_IDS.commit), "click", function (event) {
            prevent(event);
            compactCommit();
        }, "Compact");

        // The frame follows the generation size the way the editor's does.
        ["txt2img_width", "txt2img_height"].forEach(function (id) {
            const holder = byId(id);
            const input = holder ? holder.querySelector("input") : null;
            once(input, "change", compactPaint, "Compact");
            once(input, "input", compactPaint, "Compact");
        });

        compactLoad(compact.working === null);
    }

    function onKey(event) {
        if (!state.open) return;
        const target = event.target || {};
        const typing = target.tagName === "INPUT" || target.tagName === "TEXTAREA"
            || target.isContentEditable;

        if (event.key === "Escape") {
            prevent(event);
            if (state.drag || state.drawing) {
                state.drag = null;
                state.drawing = false;
                release();
                clearGhost();
                paint();
                return;
            }
            if (state.confirming) {
                state.confirming = false;
                paintChrome();
                return;
            }
            leave();
            return;
        }

        const accel = event.ctrlKey || event.metaKey;
        if (accel && (event.key === "z" || event.key === "Z")) {
            // Inside a text field the browser's own undo is the right one --
            // taking it away is how a mistyped prompt costs somebody a region.
            if (typing) return;
            prevent(event);
            if (event.shiftKey) redo(); else undo();
            return;
        }
        if (accel && (event.key === "y" || event.key === "Y")) {
            if (typing) return;
            prevent(event);
            redo();
            return;
        }

        if (typing) return;

        if (event.key === "Delete" || event.key === "Backspace") {
            prevent(event);
            remove();
            return;
        }
        const step = event.shiftKey ? NUDGE_BIG : NUDGE;
        if (event.key === "ArrowLeft") { prevent(event); nudge(-step, 0); return; }
        if (event.key === "ArrowRight") { prevent(event); nudge(step, 0); return; }
        if (event.key === "ArrowUp") { prevent(event); nudge(0, -step); return; }
        if (event.key === "ArrowDown") { prevent(event); nudge(0, step); return; }
        state.editing = "";
    }

    if (typeof onUiLoaded === "function") {
        onUiLoaded(wire);
    } else if (document.readyState !== "loading") {
        wire();
    } else {
        document.addEventListener("DOMContentLoaded", wire);
    }

    // Re-applied rather than installed once: Gradio rebuilds parts of the tab on
    // some updates. Every listener is guarded by a dataset flag, so re-running
    // costs a handful of queries and can never double-bind.
    if (typeof onAfterUiUpdate === "function") {
        onAfterUiUpdate(wire);
    }

    // Exposed for the tests, which drive this file under node against a fake
    // page. Nothing in the extension reads it.
    window.modelChainKreaSpatial = {
        state: state,
        wire: wire,
        open: open,
        close: close,
        leave: leave,
        save: save,
        serialize: function () { return state.working ? serialize() : ""; },
        orient: orient,
        create: create,
        add: add,
        select: select,
        remove: remove,
        clear: clear,
        duplicate: duplicate,
        restack: restack,
        reorder: reorder,
        settle: settle,
        undo: undo,
        redo: redo,
        zoomTo: zoomTo,
        toggleFullscreen: toggleFullscreen,
        fullscreenOn: fullscreenOn,
        dimensions: dimensions,
        ordered: function () { return state.working ? ordered() : []; },
        compact: compact,
        wireCompact: wireCompact,
        compactLoad: compactLoad,
        compactPaint: compactPaint,
        compactCommit: compactCommit,
        compactUndo: compactUndo,
        compactRegions: function () {
            return compact.working ? orderedIn(compact.working) : [];
        },
    };
})();
