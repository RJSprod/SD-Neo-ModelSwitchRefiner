// Model Chain -- Krea Creative Mode Spatial Layout, browser side.
//
// This file owns one thing: the layout editor. Somebody presses Full Screen, a
// composition workspace takes over the txt2img work area, they place boxes and
// silhouettes and type into them, they press Save, and the serialized document
// goes into a hidden textbox that travels with the next Generate like any other
// control. That is the whole contract.
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
// So: no listener on the Generate button, no timer of any kind, no polling, and
// nothing here that a generation's completion depends on. The Gallery widget
// added by §16 does not weaken that rule, it is the reason the rule is written
// the way it is: its Generate *presses the host's own button*, once, inside the
// click the user made, and the results and the progress bar are read through
// MutationObservers -- which is hearing about a change rather than asking sixty
// times a minute whether one happened. If every line of this file fails, the
// Full Screen button does nothing and the last saved layout is still the one
// that gets composed.
//
// Why the workspace does not go anywhere
// --------------------------------------
// It used to be a fixed overlay moved to document.body, and then a block in
// flow with a Fullscreen API button on it. §3.1 asks for neither: "Full Screen"
// means the Spatial workspace has the tab, so opening it marks the ancestors
// between the workspace and #tab_txt2img and lets the stylesheet hide
// everything else on the tab. Nothing is moved, no id exists twice, the browser
// keeps its own chrome and its own Back button, and Close removes the marks.
//
// Why the canvas is a div
// -----------------------
// Regions are elements. That makes each one focusable, styleable by a theme,
// findable by id, and readable by a test that never opens a browser -- and it
// makes hit-testing, dragging and z-order the browser's job rather than a
// redraw loop's. A <canvas> would put all of that behind a bitmap in order to
// draw rectangles with square corners. The anatomy silhouettes §9 asks for are
// the same answer one step on: a clip-path on the region's shape layer, so a
// head is a head on screen and an axis-aligned rectangle everywhere else.
//
// Why one pointer path and not two
// --------------------------------
// The editor used to listen for mousedown/mousemove/mouseup on the document.
// On a tablet that is either nothing at all or a synthesised echo arriving
// after a 300ms delay and a scroll. Pointer Events are one path for mouse,
// finger and pen, and setPointerCapture keeps the drag attached to the finger
// that started it even when the finger leaves the frame -- which is what makes
// the document-level listeners unnecessary rather than merely unfashionable.
// The same is true of the palettes: tapping a shape and dragging one are the
// same gesture told apart by whether the contact moved.
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
// Why the editor metadata is optional in both directions
// ------------------------------------------------------
// ui_shape and ui_rotation are the only fields §21.2 adds to the document, and
// they are written only when they say something. A rectangle layout serializes
// to the bytes it always did, a layout written before they existed loads as
// rectangles at zero degrees, and a shape from a later build draws as a
// rectangle rather than as an error. Nothing downstream of this file reads
// either one: whatever is on the canvas, the compositor is handed the same
// axis-aligned box.
//
// Why it is still one file
// ------------------------
// Forge loads javascript/*.js as plain scripts in directory order, with no
// module system and no guaranteed order, so splitting the editor across files
// buys separate files and a load-order bug. It is split into the controllers
// the design intent names -- state, history, frame, takeover, host bridge,
// painting, selection, pointer, popups, panels, palette, prompts, gallery,
// serialization -- as sections with no shared scope beyond `state`.
(function () {
    "use strict";

    const P = "mc-krea-spatial";
    const SCALE = 1000;
    const MIN_SIZE = 8;          // normalized units; smaller than this is a tap
    const MAX_REGIONS = 24;      // matches prompt_master/krea/spatial.MAX_REGIONS
    const HISTORY_DEPTH = 50;    // §20 asks for 50; a snapshot is a short string
    const NUDGE = 5;             // normalized units per arrow press
    const NUDGE_BIG = 25;
    const ADD_SIZE = 340;        // Quick Add's opening reference size, per §10.1
    const TAP = 24;              // pixels a palette press may wander and stay a tap
    const ZOOMS = [0.5, 0.75, 1, 1.25, 1.5, 2, 3, 4];

    // §6.3 and §9.3. Editor metadata, all of it: what a shape is called, the
    // proportions it starts at relative to the Quick Add size, and the prompt it
    // arrives carrying. `rect` is the row with no silhouette and no prompt --
    // the box this editor has always drawn, named here so that everything else
    // can be another row rather than a special case.
    //
    // Nothing downstream of the canvas reads this. §2.5: whatever is drawn, the
    // compositor is handed the same axis-aligned rectangle, so a shape added
    // here changes no branch outside this file and no byte of the prompt.
    const SHAPES = {
        rect: {label: "Box", w: 1, h: 1, prompt: ""},
        head: {label: "Head", w: 0.62, h: 0.74, prompt: "head"},
        chest: {label: "Chest", w: 1, h: 0.86, prompt: "chest"},
        waist: {label: "Waist", w: 0.9, h: 0.6, prompt: "waist"},
        left_arm: {label: "Left arm", w: 0.36, h: 1.02, prompt: "left arm"},
        right_arm: {label: "Right arm", w: 0.36, h: 1.02, prompt: "right arm"},
        left_hand: {label: "Left hand", w: 0.34, h: 0.34, prompt: "left hand"},
        right_hand: {label: "Right hand", w: 0.34, h: 0.34, prompt: "right hand"},
        left_leg: {label: "Left leg", w: 0.4, h: 1.16, prompt: "left leg"},
        right_leg: {label: "Right leg", w: 0.4, h: 1.16, prompt: "right leg"},
        left_foot: {label: "Left foot", w: 0.42, h: 0.28, prompt: "left foot"},
        right_foot: {label: "Right foot", w: 0.42, h: 0.28, prompt: "right foot"},
    };

    // §26: an unknown ui_shape is a rectangle rather than an error. A layout
    // written by a later build arrives with boxes in the right places and one
    // silhouette this build cannot draw, which is the failure worth having.
    function shapeOf(name) {
        return SHAPES[name] || SHAPES.rect;
    }

    function knownShape(name) {
        return Object.prototype.hasOwnProperty.call(SHAPES, name) ? name : "rect";
    }

    // §12.2, in the order the rail builds them. One list, three readers: the
    // rail, the Panels popup and the collapse/hide state below.
    const PANELS = ["prompts", "person", "layers", "inspector", "gallery", "session"];

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
        rail: P + "-rail",
        list: P + "-list",
        count: P + "-count",
        selectedName: P + "-selected-name",
        name: P + "-name",
        shape: P + "-shape",
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
        rotation: P + "-rotation",
        rotationField: P + "-rotation-field",
        rotationRead: P + "-rotation-read",
        bboxX: P + "-bbox-x",
        bboxY: P + "-bbox-y",
        bboxW: P + "-bbox-w",
        bboxH: P + "-bbox-h",
        autoHint: P + "-auto-hint",
        grid: P + "-grid",
        aspect: P + "-aspect",
        message: P + "-message",
        confirm: P + "-confirm",
        keep: P + "-keep",
        discard: P + "-discard",
        // The action bar.
        power: P + "-power",
        modeDirect: P + "-mode-direct",
        modeSmart: P + "-mode-smart",
        quick: P + "-quick",
        quickPopup: P + "-quick-popup",
        quickSize: P + "-quick-size",
        draw: P + "-draw",
        clear: P + "-clear",
        panels: P + "-panels",
        panelsPopup: P + "-panels-popup",
        collapseAll: P + "-collapse-all",
        expandVisible: P + "-expand-visible",
        undo: P + "-undo",
        redo: P + "-redo",
        save: P + "-save",
        cancel: P + "-cancel",
        // Layers.
        duplicate: P + "-duplicate",
        remove: P + "-delete",
        front: P + "-front",
        back: P + "-lower",
        // The frame toolbar.
        zoomFit: P + "-zoom-fit",
        zoomIn: P + "-zoom-in",
        zoomOut: P + "-zoom-out",
        zoomLevel: P + "-zoom-level",
        // Prompts.
        scene: P + "-scene",
        literalPlus: P + "-literal-plus",
        literalMinus: P + "-literal-minus",
        creative: P + "-creative",
        // Person.
        person: P + "-person",
        // Gallery.
        shot: P + "-shot",
        shotAt: P + "-shot-at",
        shotPrevious: P + "-shot-previous",
        shotNext: P + "-shot-next",
        generate: P + "-generate",
        progress: P + "-progress",
        progressBar: P + "-progress-bar",
        progressRead: P + "-progress-read",
        // Session.
        factFrame: P + "-fact-frame",
        factRatio: P + "-fact-ratio",
        factRegions: P + "-fact-regions",
        factPipeline: P + "-fact-pipeline",
        factState: P + "-fact-state",
        sizeWidth: P + "-size-width",
        sizeHeight: P + "-size-height",
    };

    // The host's own controls. Every one of these is a Gradio component or a
    // Forge element that already exists: §13.2 and §16.2 both say the same
    // thing in different words -- the workspace shows the canonical value, it
    // does not keep a second copy of it.
    const HOST = {
        tab: "tab_txt2img",
        prompt: "txt2img_prompt",
        width: "txt2img_width",
        height: "txt2img_height",
        gallery: "txt2img_gallery",
        results: "txt2img_results",
        generate: "txt2img_generate",
        toggle: "mc-krea-creative-spatial-toggle",
        compose: "mc-krea-creative-spatial-compose",
        creative: "mc-krea-creative-toggle",
        literalPlus: "mc-krea-creative-literal-positive",
        literalMinus: "mc-krea-creative-literal-negative",
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
        reframed: false,    // the frame's aspect changed under an open layout
        selected: "",       // region id
        drag: null,         // {kind, id, from, origin, before}
        pointer: null,      // the pointerId that owns the drag
        counter: 0,
        zoom: 1,
        baseline: "",       // serialize() as it was on entry; Close compares to it
        confirming: false,
        editing: "",        // history coalescing token; see mark()
        past: [],
        future: [],
        // §20: panel visibility, what is collapsed, which popup is open and the
        // Quick Add size are session UI preferences. None of them is layout
        // history and none of them is serialized.
        popup: "",          // the id of the open popup, or ""
        hidden: {},         // panel key -> true when switched off in Panels
        collapsed: {},      // panel key -> true when collapsed
        quickSize: ADD_SIZE,
        placing: null,      // a palette press in flight: {shape, from, moved, ghost}
        syncing: false,     // a prompt mirror is writing; do not echo it back
        shots: [],          // the txt2img gallery's images, newest last
        shotAt: -1,         // which one the Gallery widget is showing
        watching: false,    // the element the gallery observer is watching
        watcher: null,      // that observer, so a rebuilt gallery can replace it
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

    function write(id, value) {
        const element = byId(id);
        if (element) element.textContent = value === undefined ? "" : String(value);
    }

    // A Gradio component is a wrapper with the real control somewhere inside it,
    // and which control depends on the component. Both lookups are here so that
    // "the host's Prompt box" is one call rather than a query repeated eight
    // times with eight chances to differ.
    function hostField(id) {
        const holder = byId(id);
        if (!holder) return null;
        if (holder.tagName === "TEXTAREA" || holder.tagName === "INPUT") return holder;
        return holder.querySelector("textarea, input");
    }

    function hostCheck(id) {
        const holder = byId(id);
        if (!holder) return null;
        if (holder.tagName === "INPUT") return holder;
        return holder.querySelector("input[type=checkbox]") || holder.querySelector("input");
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
            // §2.6 and §26. Editor metadata, optional in both directions: a
            // layout drawn before these existed arrives with no ui_shape and no
            // ui_rotation and becomes a rectangle at zero degrees, which is
            // exactly what it was. An unknown shape from a later build becomes
            // a rectangle rather than an error.
            shape: knownShape(String(entry.ui_shape || "rect")),
            rotation: rotate(entry.ui_rotation),
        };
    }

    // -180..180, rounded, and 0 for anything that is not a number. §9.6: the
    // one thing rotation may never do is reach the bbox, so it is clamped here
    // and read nowhere near orient().
    function rotate(value) {
        const number = Math.round(Number(value));
        if (!Number.isFinite(number)) return 0;
        return clamp(number, -180, 180);
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
                // §21.2. Written only when they say something, for the reason
                // the region literals are: a rectangle layout serializes to the
                // bytes it always did, and a build that predates the editor
                // metadata still reads every document this one writes.
                if (region.shape && region.shape !== "rect") entry.ui_shape = region.shape;
                if (region.rotation) entry.ui_rotation = region.rotation;
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

    // "1024 × 1536 · 2:3 · Portrait", for the Session widget. §18.2 keeps the
    // long form out of the action bar; the bar gets the ratio on its own.
    function dimensions(width, height) {
        return width + " × " + height + " · " + ratio(width, height)
            + " · " + shape(width, height);
    }

    function brief(width, height) {
        return width + " × " + height + " (" + ratio(width, height) + ")";
    }

    // Normalized coordinates are fractions of the frame, so a resolution change
    // moves nothing and a *ratio* change reframes everything. §18.3: the boxes
    // are left exactly where they are, the frame is redrawn around them, and
    // the Session widget says the frame changed. Reprojecting would be this
    // file deciding which of somebody's boxes deserved to keep its shape.
    function reframe() {
        if (!state.working) return;
        const size = hostSize();
        const canvas = byId(IDS.canvas);
        if (!(size.width > 0 && size.height > 0)) {
            state.reframed = false;
            return;
        }
        if (canvas) {
            canvas.style.aspectRatio = size.width + " / " + size.height;
            setVar(canvas, "--mc-ar-w", String(size.width));
            setVar(canvas, "--mc-ar-h", String(size.height));
        }

        // Compared against the frame the layout was *saved* on rather than
        // against whatever the last reframe wrote, so the fact survives a
        // second reframe and still tells the truth if the size changes again
        // while the workspace is open.
        const was = state.savedCanvas;
        const now = size.width / size.height;
        state.reframed = state.working.regions.length > 0
            && Number(was.width) > 0 && Number(was.height) > 0
            && Math.abs((was.width / was.height) - now) / now > 0.02;
        state.working.canvas.width = size.width;
        state.working.canvas.height = size.height;
    }

    // Zoom is display only and never touches a stored coordinate: it scales the
    // frame's CSS width and lets the scroll container do the rest. §5.5.
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

    // ------------------------------------------------------------------ //
    // TakeoverController -- the workspace gets the txt2img work area
    // ------------------------------------------------------------------ //
    //
    // §3.1. "Full Screen" means the Spatial workspace has the tab, not that the
    // browser has the screen: the Fullscreen API is a non-goal (§29), and what
    // it actually bought -- an element nothing can clip -- is bought here more
    // cheaply and without taking the browser's own chrome away.
    //
    // Nothing is moved. The workspace stays exactly where Gradio put it, and
    // every element between it and the txt2img tab is marked as being on the
    // path to it. One class on the tab, one attribute per ancestor, and the
    // stylesheet hides every child of a path element that is not itself on the
    // path. Close removes the marks and the tab is as it was.
    //
    // That is the whole of §22.4: no DOM teleport, no second copy of any id, no
    // document.body overlay, and nothing about the page's scrolling changed for
    // anybody who never opens the editor.

    const PATH = "data-mc-spatial-path";

    function tab() {
        return byId(HOST.tab);
    }

    function takeover() {
        const workspace = byId(IDS.workspace);
        const host = tab();
        if (!workspace || !host) return false;
        let at = workspace;
        // The workspace itself is on the path, and so is everything up to and
        // including the tab. `contains` is not used: a detached ancestor chain
        // walks to null and marks nothing, which is the safe answer.
        let guard = 0;
        while (at && guard < 64) {
            if (at.setAttribute) at.setAttribute(PATH, "1");
            if (at === host) break;
            at = at.parentNode;
            guard += 1;
        }
        if (host.classList) host.classList.add(P + "-taken");
        return true;
    }

    function surrender() {
        const host = tab();
        if (host && host.classList) host.classList.remove(P + "-taken");
        let found = [];
        try {
            found = Array.prototype.slice.call(
                root().querySelectorAll("[" + PATH + "]"));
        } catch (error) {
            found = [];
        }
        found.forEach(function (element) {
            if (element.removeAttribute) element.removeAttribute(PATH);
        });
    }


    // ------------------------------------------------------------------ //
    // HostBridge -- the workspace shows the host's values, never a copy
    // ------------------------------------------------------------------ //
    //
    // §13.2 and §16.2 say the same thing about different controls: the Spatial
    // switch, the composition mode, the prompt, the two Literal boxes, the
    // frame size and the result gallery all already exist on the txt2img tab,
    // and the workspace shows *those*. Nothing below stores a value of its own,
    // so there is no second copy to drift, no save step between the two, and
    // nothing to reconcile when the workspace closes.
    //
    // Reading is a query. Writing is the event the host is already listening
    // for -- `updateInput` where Forge offers it, `input` and `change` where it
    // does not. §13.2's other half: no polling, no interval, nothing here that
    // runs when nobody has touched anything.

    function fireOn(element, type) {
        if (!element || typeof element.dispatchEvent !== "function") return;
        element.dispatchEvent(new Event(type, {bubbles: true}));
    }

    // A checkbox is not a textbox: Gradio binds it to `change` and its `value`
    // means nothing. `publish` is for the boxes that carry text.
    function toggleHost(box, on) {
        if (!box) return;
        if (!!box.checked === !!on) return;
        box.checked = !!on;
        fireOn(box, "input");
        fireOn(box, "change");
    }

    function spatialOn() {
        const box = hostCheck(HOST.toggle);
        return box ? !!box.checked : true;
    }

    function creativeOn() {
        const box = hostCheck(HOST.creative);
        return box ? !!box.checked : false;
    }

    // Gradio renders a Radio as a group of inputs, and which of `value` and the
    // label text carries the choice has moved between versions. Both are read,
    // in that order, and the answer is normalized to the two words the
    // compositor knows -- so a version that labels them differently still gets
    // "smart" or "direct" out of this rather than a string nothing matches.
    function radioInputs() {
        const holder = byId(HOST.compose);
        if (!holder || !holder.querySelectorAll) return [];
        return Array.prototype.slice.call(holder.querySelectorAll("input"));
    }

    function radioMode(input) {
        const said = String((input && input.value) || "").toLowerCase();
        if (said.indexOf("direct") >= 0) return "direct";
        if (said.indexOf("smart") >= 0) return "smart";
        const label = input && input.closest ? input.closest("label") : null;
        const text = label ? String(label.textContent || "").toLowerCase() : "";
        if (text.indexOf("direct") >= 0) return "direct";
        if (text.indexOf("smart") >= 0) return "smart";
        return "";
    }

    // The host's radio when there is one, the working document when there is
    // not. Both are true answers: the layout carries the mode it was saved with
    // and the panel carries the one that is set now, and they agree except in
    // the moment between the two.
    function composeMode() {
        const found = radioInputs().filter(function (input) { return input.checked; })
            .map(radioMode).filter(Boolean);
        if (found.length) return found[0];
        if (state.working) return state.working.compose_mode === "direct" ? "direct" : "smart";
        return "smart";
    }

    function setComposeMode(mode) {
        const wanted = mode === "direct" ? "direct" : "smart";
        if (state.working) state.working.compose_mode = wanted;
        const inputs = radioInputs();
        const input = inputs.filter(function (entry) {
            return radioMode(entry) === wanted;
        })[0];
        if (!input || input.checked) return;
        // Written rather than left to the browser: a radio group only clears
        // its siblings when they share a `name`, and what Gradio puts there has
        // moved between versions. Two of them checked would make composeMode()
        // answer with whichever came first in the markup.
        inputs.forEach(function (entry) {
            if (entry !== input) entry.checked = false;
        });
        input.checked = true;
        fireOn(input, "input");
        fireOn(input, "change");
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
        // Before the facts, not after: settle() is what makes "Saved" true, and
        // a Session widget painted first would say "Unsaved" about an edit that
        // was committed a line later and stay wrong until the next repaint.
        settle();
        paintFacts();
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

    // A region is three elements: the body that carries the bbox, the shape
    // layer that carries the silhouette and the rotation, and the label. §6.2
    // centres the label inside both rectangles and silhouettes, which is a CSS
    // job -- what matters here is that there is exactly one element to centre
    // it in whichever the region turns out to be.
    function body(region) {
        const element = document.createElement("div");
        element.className = P + "-region";
        element.dataset.regionId = region.id;
        const art = document.createElement("div");
        art.className = P + "-shape";
        element.appendChild(art);
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
            node.dataset.shape = region.shape;
            node.classList.toggle("selected", region.id === state.selected);
            node.classList.toggle("text", region.type === "text");
            node.classList.toggle("figure", region.shape !== "rect");
            const art = node.querySelector("." + P + "-shape");
            if (art) {
                art.className = P + "-shape " + P + "-shape-" + region.shape;
                // §9.6: the silhouette turns and the bbox does not. This is the
                // only place rotation is applied and it is a CSS transform, so
                // there is no arithmetic anywhere that could leak it into a
                // coordinate.
                art.style.transform = region.rotation
                    ? "rotate(" + region.rotation + "deg)" : "";
            }
            const label = node.querySelector("." + P + "-label");
            if (label) label.textContent = region.name || region.id;
        });
        paintProxy();
        const guides = byId(IDS.guides);
        if (guides) guides.className = P + "-guides "
            + (state.working.canvas.grid || "none");
        const aspect = byId(IDS.aspect);
        if (aspect) {
            const size = state.working.canvas;
            aspect.textContent = size.width > 0 && size.height > 0
                ? ratio(size.width, size.height) : "";
        }
    }

    // SelectionController's visible half: one element, above every region body,
    // carrying the outline, the label, the move surface and the eight handles
    // of whatever is selected. It reads and writes that region's bbox and never
    // touches its z -- §7, which is the entire answer to "the box I selected is
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

    // LayersController. Prompt order, top to bottom: the first row is the first
    // element in the composed prompt and the last row is the one drawn on top
    // of the others. §14.1 recommends frontmost first; this list is the thing
    // somebody drags to change the *prompt* order, and the two orders are one
    // number, so it reads in prompt order and says so by being draggable.
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

            // §14.1's type marker: the silhouette itself for an object region,
            // drawn by the same clip-path the canvas uses, and a letter for a
            // text region because text has no shape.
            const icon = document.createElement("span");
            icon.className = P + "-row-icon " + P + "-shape-" + region.shape;
            icon.dataset.shape = region.shape;
            if (region.type === "text") {
                icon.className = P + "-row-icon " + P + "-row-icon-text";
                icon.textContent = "T";
            }
            row.appendChild(icon);

            const name = document.createElement("span");
            name.className = P + "-row-name";
            name.textContent = region.name || region.id;
            row.appendChild(name);

            const size = document.createElement("span");
            size.className = P + "-row-size";
            size.textContent = (region.bbox[2] - region.bbox[0]) + "×"
                + (region.bbox[3] - region.bbox[1]);
            row.appendChild(size);

            const trash = document.createElement("button");
            trash.className = P + "-row-trash";
            trash.type = "button";
            trash.dataset.regionId = region.id;
            trash.setAttribute && trash.setAttribute("aria-label",
                                                     "Delete " + (region.name || region.id));
            trash.textContent = "✕";
            row.appendChild(trash);

            list.appendChild(row);
        });
        const count = byId(IDS.count);
        if (count) count.textContent = rows.length ? String(rows.length) : "";
    }

    // InspectorController. §15.1's fields, and the four numbers §8.3 asks for.
    function paintInspector() {
        if (!state.working) return;
        const region = find(state.selected);
        const fields = [IDS.name, IDS.type, IDS.text, IDS.prompt, IDS.framing,
                        IDS.angle, IDS.literalPrefix, IDS.literalSuffix,
                        IDS.rotation, IDS.bboxX, IDS.bboxY, IDS.bboxW, IDS.bboxH];
        fields.forEach(function (id) { enable(id, !!region); });
        enable(IDS.duplicate, !!region);
        enable(IDS.remove, !!region);
        [IDS.front, IDS.back].forEach(function (id) { enable(id, !!region); });

        const auto = byId(IDS.autoHint);
        const grid = byId(IDS.grid);
        const title = byId(IDS.selectedName);
        if (auto) auto.checked = !!state.working.auto_position_hint;
        if (grid) grid.value = state.working.canvas.grid || "none";
        // Emptied rather than left showing the last region's answers: an
        // inspector still saying "Text" after the text region it described was
        // deleted is a panel lying about what is selected.
        if (!region) {
            if (title) title.textContent = "";
            [IDS.name, IDS.text, IDS.prompt, IDS.literalPrefix, IDS.literalSuffix,
             IDS.shape, IDS.bboxX, IDS.bboxY, IDS.bboxW, IDS.bboxH]
                .forEach(function (id) { set(id, ""); });
            set(IDS.type, "obj");
            set(IDS.framing, "");
            set(IDS.angle, "");
            set(IDS.rotation, "0");
            write(IDS.rotationRead, "0°");
            const blank = byId(IDS.promptLabel);
            if (blank) blank.textContent = "Prompt";
            show(IDS.textField, false);
            show(IDS.framingField, true);
            show(IDS.angleField, true);
            show(IDS.rotationField, false);
            return;
        }
        if (title) title.textContent = region.name || region.id;
        set(IDS.name, region.name);
        set(IDS.shape, shapeOf(region.shape).label);
        set(IDS.type, region.type);
        set(IDS.text, region.text);
        set(IDS.prompt, region.prompt);
        set(IDS.literalPrefix, region.literalPrefix);
        set(IDS.literalSuffix, region.literalSuffix);
        set(IDS.framing, region.framing);
        set(IDS.angle, region.angle);
        set(IDS.rotation, String(region.rotation));
        write(IDS.rotationRead, region.rotation + "°");
        set(IDS.bboxX, String(region.bbox[0]));
        set(IDS.bboxY, String(region.bbox[1]));
        set(IDS.bboxW, String(region.bbox[2] - region.bbox[0]));
        set(IDS.bboxH, String(region.bbox[3] - region.bbox[1]));
        show(IDS.textField, region.type === "text");
        show(IDS.framingField, region.type !== "text");
        show(IDS.angleField, region.type !== "text");
        // §9.6: rotation belongs to the silhouettes. A rectangle has nothing to
        // turn, and offering the slider anyway would imply the bbox rotates.
        show(IDS.rotationField, region.shape !== "rect");
        const label = byId(IDS.promptLabel);
        if (label) label.textContent = region.type === "text" ? "Appearance" : "Prompt";
    }

    // §12.2. Hidden is a panel switched off in the Panels popup; collapsed is a
    // panel whose header was pressed. Both are session preferences (§20) and
    // neither is history, so this is the only place either one is read.
    function paintPanels() {
        PANELS.forEach(function (key) {
            const section = byId(P + "-panel-" + key);
            const shown = !state.hidden[key];
            const open = !state.collapsed[key];
            if (section) {
                section.hidden = !shown;
                section.classList && section.classList.toggle("collapsed", !open);
            }
            const inner = byId(P + "-panel-" + key + "-body");
            if (inner) inner.hidden = !open;
            const head = section
                ? section.querySelector("." + P + "-widget-head") : null;
            if (head && head.setAttribute) {
                head.setAttribute("aria-expanded", open ? "true" : "false");
            }
            const check = byId(P + "-show-" + key);
            if (check) check.checked = shown;
        });
        const rail = byId(IDS.rail);
        if (rail && rail.classList) {
            rail.classList.toggle("empty", PANELS.every(function (key) {
                return state.hidden[key];
            }));
        }
    }

    function paintChrome() {
        if (!state.working) return;
        enable(IDS.undo, state.past.length > 0);
        enable(IDS.redo, state.future.length > 0);
        enable(IDS.clear, state.working.regions.length > 0);
        const room = state.working.regions.length < MAX_REGIONS;
        enable(IDS.quick, room);
        enable(IDS.draw, room);
        const draw = byId(IDS.draw);
        if (draw) {
            draw.classList.toggle("active", state.drawing);
            draw.setAttribute && draw.setAttribute("aria-pressed",
                                                   state.drawing ? "true" : "false");
        }
        const canvas = byId(IDS.canvas);
        if (canvas && canvas.classList) canvas.classList.toggle("drawing", state.drawing);
        const confirm = byId(IDS.confirm);
        if (confirm) confirm.hidden = !state.confirming;
        [[IDS.quick, IDS.quickPopup], [IDS.panels, IDS.panelsPopup]]
            .forEach(function (pair) {
                const open = state.popup === pair[1];
                const popup = byId(pair[1]);
                if (popup) popup.hidden = !open;
                const button = byId(pair[0]);
                if (button) {
                    button.classList && button.classList.toggle("active", open);
                    button.setAttribute && button.setAttribute("aria-expanded",
                                                               open ? "true" : "false");
                }
            });
        const power = byId(IDS.power);
        if (power) power.checked = spatialOn();
        const mode = composeMode();
        [[IDS.modeDirect, "direct"], [IDS.modeSmart, "smart"]].forEach(function (pair) {
            const button = byId(pair[0]);
            if (!button) return;
            const on = mode === pair[1];
            button.classList && button.classList.toggle("active", on);
            button.setAttribute && button.setAttribute("aria-pressed", on ? "true" : "false");
        });
        paintPanels();
    }

    // §17. Everything that used to be a permanent strip across the top, in the
    // one widget somebody can collapse or switch off.
    function paintFacts() {
        if (!state.working) return;
        const size = state.working.canvas;
        const known = Number(size.width) > 0 && Number(size.height) > 0;
        write(IDS.factFrame, known ? size.width + " × " + size.height : "—");
        write(IDS.factRatio, known
            ? ratio(size.width, size.height) + " · " + shape(size.width, size.height)
            : "—");
        write(IDS.factRegions, state.working.regions.length + " of " + MAX_REGIONS);
        write(IDS.factPipeline, (composeMode() === "direct" ? "Direct BBOX" : "Smart Spatial")
            + (spatialOn() ? "" : " · off") + (creativeOn() ? " · Creative first" : ""));
        const dirty = serialize() !== state.committed;
        write(IDS.factState, state.reframed
            ? "Reframed from " + brief(state.savedCanvas.width, state.savedCanvas.height)
            : dirty ? "Unsaved" : "Saved");
        const field = byId(IDS.factState);
        if (field && field.classList) field.classList.toggle("warn", dirty || state.reframed);
        if (known) {
            set(IDS.sizeWidth, String(size.width));
            set(IDS.sizeHeight, String(size.height));
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

    // The one transient line, and the only sentence the workspace has left:
    // §26's region cap, which is a fact about what just failed to happen rather
    // than a description of a control. Non-blocking by construction -- there is
    // nothing to dismiss, so there is nothing that can eat somebody's layout
    // while they dismiss it.
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

    // §6.1: the default name is the number and nothing else. "Region 4" is the
    // word "region" repeated on every row of a panel called Layers, and the
    // number is the half somebody actually reads.
    function nextName() {
        const taken = {};
        state.working.regions.forEach(function (region) { taken[region.name] = true; });
        let at = state.working.regions.length + 1;
        while (taken[String(at)]) at += 1;
        return String(at);
    }

    // §15.2. A field that is switched off in Panels or collapsed cannot be
    // typed into, so the panel it lives in is opened before the focus is asked
    // for -- otherwise a new rectangle's prompt goes to a box nobody can see.
    function focusPrompt() {
        state.hidden.inspector = false;
        state.collapsed.inspector = false;
        paintPanels();
        const field = byId(IDS.prompt);
        if (field && typeof field.focus === "function") field.focus();
    }

    function create(bbox, options) {
        const chosen = options || {};
        if (state.working.regions.length >= MAX_REGIONS) {
            say(MAX_REGIONS + " regions is the limit");
            return null;
        }
        // A drawn region records its own history entry before the gesture
        // started, so it asks for no second one -- but it is still a finished
        // edit, and Auto Save exists so that a finished edit is what the next
        // Generate composes. Marking it here rather than only in mark() is the
        // difference between drawing a box and having drawn one.
        if (chosen.history !== false) mark("");
        else state.pendingSave = true;
        const shape = knownShape(chosen.shape || "rect");
        const highest = state.working.regions.reduce(function (top, region) {
            return Math.max(top, region.z);
        }, -1);
        const region = {
            id: nextId(),
            name: nextName(),
            type: "obj",
            bbox: bbox,
            // §9.3: a person part arrives with the words for itself already in
            // it. A rectangle arrives blank, because only the user knows what
            // they drew it around.
            prompt: shapeOf(shape).prompt,
            literalPrefix: "",
            literalSuffix: "",
            text: "",
            framing: "",
            angle: "",
            z: highest + 1,
            shape: shape,
            rotation: 0,
        };
        state.working.regions.push(region);
        say("");
        select(region.id);
        // §10.2 and §11.3. A blank rectangle needs its prompt typed and the
        // cursor is put there; a silhouette does not, and taking the focus away
        // from the canvas after every dropped limb would make placing a figure
        // a fight with a text box.
        if (shape === "rect") focusPrompt();
        return region;
    }

    // §10.1: the Quick Add size is a reference length, and each shape is a
    // proportion of it. A head is not square and a leg is not a head.
    function sized(shape, size) {
        const spec = shapeOf(shape);
        return {
            width: clamp(Math.round(size * spec.w), MIN_SIZE, SCALE),
            height: clamp(Math.round(size * spec.h), MIN_SIZE, SCALE),
        };
    }

    // Centred on `at`, or on the frame when there is no `at` -- §10.2's tap and
    // §10.3's drop, which differ in one coordinate and nothing else. A tap
    // cascades a little so that pressing Head twice does not hide the first one
    // exactly underneath the second.
    function placeShape(name, at) {
        if (!state.working) return null;
        const shape = knownShape(name);
        const box = sized(shape, state.quickSize);
        const drift = at ? 0 : (state.working.regions.length % 5) * 24 - 48;
        const centre = at || {x: SCALE / 2, y: SCALE / 2};
        const x0 = clamp(Math.round(centre.x - box.width / 2) + drift, 0, SCALE - box.width);
        const y0 = clamp(Math.round(centre.y - box.height / 2) + drift, 0, SCALE - box.height);
        const made = orient([x0, y0, x0 + box.width, y0 + box.height]);
        if (!made) return null;
        return create(made, {shape: shape});
    }

    function add() {
        return placeShape("rect", null);
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

    // §4.5. One action, one history item, and Undo brings all of them back --
    // which is what makes it safe to offer without a confirmation dialog in the
    // way. It touches the regions and nothing else: not the prompt, not the
    // gallery, not the Spatial switch.
    function clear() {
        if (!state.working.regions.length) return;
        mark("");
        state.working.regions = [];
        select("");
    }

    function duplicate() {
        const region = find(state.selected);
        if (!region) return;
        if (state.working.regions.length >= MAX_REGIONS) {
            say(MAX_REGIONS + " regions is the limit");
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
            name: nextName(),
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
                create(made, {history: false});
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
    // PopupController -- two menus, one at a time
    // ------------------------------------------------------------------ //
    //
    // §4.3 and §4.6. Both popups are anchored elements inside the action bar,
    // not overlays: they are in the bar's own stacking context, they close on
    // Escape, on a press outside them and on each other, and neither of them
    // can be left open behind a workspace that has been closed.

    function popupOpen(id) {
        // §11.4: opening a menu is switching to another high-level action, and
        // Draw does not stay armed across one.
        if (state.drawing) { state.drawing = false; clearGhost(); }
        state.popup = state.popup === id ? "" : id;
        paintChrome();
    }

    function popupsShut() {
        if (!state.popup) return;
        state.popup = "";
        paintChrome();
    }

    // ------------------------------------------------------------------ //
    // PanelController -- what is in the rail, and what is open
    // ------------------------------------------------------------------ //
    //
    // §12.2 and §20. Two flags per widget, both session preferences: hidden is
    // the Panels switch, collapsed is the header. Neither is layout history and
    // neither is serialized, so hiding a panel cannot mark a layout as changed
    // and Undo cannot bring a panel back.

    function panelShow(key, on) {
        state.hidden[key] = !on;
        paintPanels();
    }

    function panelCollapse(key, on) {
        state.collapsed[key] = !!on;
        paintPanels();
    }

    function collapseAll() {
        PANELS.forEach(function (key) { state.collapsed[key] = true; });
        paintPanels();
    }

    function expandVisible() {
        PANELS.forEach(function (key) {
            if (!state.hidden[key]) state.collapsed[key] = false;
        });
        paintPanels();
    }

    // ------------------------------------------------------------------ //
    // PaletteController -- tap to place, drag to place where dropped
    // ------------------------------------------------------------------ //
    //
    // §10.2, §10.3 and §9.4. Quick Add and the Person outline are the same
    // gesture on two sets of buttons, so they are one implementation: press,
    // and if the contact moves it carries a ghost and drops a region where it
    // is let go; if it does not move it is a tap and the region lands centred.
    //
    // The capture is taken on the workspace rather than on the button, because
    // the Quick Add popup may close while a contact is still down and a capture
    // held by a hidden element is a drag that stops reporting. Pointer Events
    // throughout: §2.3, one path for mouse, finger and pen.

    function pointIn(clientX, clientY) {
        const canvas = byId(IDS.canvas);
        if (!canvas || typeof canvas.getBoundingClientRect !== "function") return null;
        const rect = canvas.getBoundingClientRect();
        if (!(rect.width > 0 && rect.height > 0)) return null;
        const x = (clientX - rect.left) / rect.width * SCALE;
        const y = (clientY - rect.top) / rect.height * SCALE;
        // §10.3: an invalid drop creates nothing. Outside the frame is outside.
        if (x < 0 || y < 0 || x > SCALE || y > SCALE) return null;
        return {x: Math.round(x), y: Math.round(y)};
    }

    function dropGhostAt(shape, clientX, clientY) {
        const workspace = byId(IDS.workspace);
        if (!workspace) return;
        let ghost = workspace.querySelector("." + P + "-drop-ghost");
        if (!ghost) {
            ghost = document.createElement("div");
            ghost.className = P + "-drop-ghost " + P + "-shape-" + shape;
            workspace.appendChild(ghost);
        }
        ghost.style.left = clientX + "px";
        ghost.style.top = clientY + "px";
    }

    function dropGhost() {
        const workspace = byId(IDS.workspace);
        const ghost = workspace && workspace.querySelector
            ? workspace.querySelector("." + P + "-drop-ghost") : null;
        if (ghost && ghost.parentNode) ghost.parentNode.removeChild(ghost);
    }

    function paletteDown(event, button) {
        if (!state.open || !state.working) return;
        if (typeof event.button === "number" && event.button > 0) return;
        prevent(event);
        state.placing = {
            shape: knownShape(button.dataset.shape),
            from: {x: event.clientX, y: event.clientY},
            moved: false,
        };
        const workspace = byId(IDS.workspace);
        if (workspace && typeof workspace.setPointerCapture === "function"
            && event.pointerId !== undefined) {
            try {
                workspace.setPointerCapture(event.pointerId);
                state.placing.pointer = event.pointerId;
            } catch (error) {
                // A synthetic pointer, or one already released. The gesture
                // still works; it just stops following a contact that leaves.
            }
        }
    }

    function paletteMove(event) {
        if (!state.placing) return;
        const from = state.placing.from;
        const far = Math.abs(event.clientX - from.x) + Math.abs(event.clientY - from.y);
        if (!state.placing.moved && far < TAP) return;
        state.placing.moved = true;
        prevent(event);
        dropGhostAt(state.placing.shape, event.clientX, event.clientY);
    }

    function paletteUp(event) {
        const placing = state.placing;
        state.placing = null;
        dropGhost();
        if (!placing) return;
        const workspace = byId(IDS.workspace);
        if (workspace && typeof workspace.releasePointerCapture === "function"
            && placing.pointer !== undefined) {
            try { workspace.releasePointerCapture(placing.pointer); } catch (error) { /* gone */ }
        }
        if (!state.open || !state.working) return;
        if (!placing.moved) {
            placeShape(placing.shape, null);
        } else {
            const at = pointIn(event.clientX, event.clientY);
            if (!at) { paint(); return; }
            placeShape(placing.shape, at);
        }
        // §4.3: the popup may close after a successful add, and does. The
        // Person panel is a palette rather than a menu and stays where it is.
        popupsShut();
        paint();
    }

    function paletteCancel() {
        if (!state.placing) return;
        state.placing = null;
        dropGhost();
    }

    // ------------------------------------------------------------------ //
    // PromptMirror -- one prompt, shown twice
    // ------------------------------------------------------------------ //
    //
    // §13.1 and §13.2. These three boxes are not saved copies: each one is a
    // second view of a component that already exists on the tab, and editing
    // either view writes the other. The write is the event the host is already
    // bound to, so there is no polling, no interval and no reconciliation step.

    const MIRRORS = [
        {here: IDS.scene, there: HOST.prompt},
        {here: IDS.literalPlus, there: HOST.literalPlus},
        {here: IDS.literalMinus, there: HOST.literalMinus},
    ];

    function syncFromHost() {
        state.syncing = true;
        try {
            MIRRORS.forEach(function (pair) {
                const mine = byId(pair.here);
                const theirs = hostField(pair.there);
                if (!mine || !theirs) return;
                if (mine.value !== theirs.value) mine.value = theirs.value;
            });
            const creative = byId(IDS.creative);
            if (creative) creative.checked = creativeOn();
        } finally {
            state.syncing = false;
        }
    }

    function syncToHost(pair) {
        if (state.syncing) return;
        const mine = byId(pair.here);
        const theirs = hostField(pair.there);
        if (!mine || !theirs || theirs.value === mine.value) return;
        publish(theirs, mine.value);
    }

    // ------------------------------------------------------------------ //
    // GalleryController -- the tab's own results, beside the canvas
    // ------------------------------------------------------------------ //
    //
    // §16.2: the same result gallery as txt2img, read rather than duplicated.
    // §16.4: Generate is the host's button, pressed on the click the user made
    // -- no second endpoint, no queued click, no timer, and nothing here that a
    // generation's completion waits for. §16.5: the progress the host is
    // already drawing, mirrored.
    //
    // The observers are MutationObservers and not intervals, which is the
    // difference between hearing about a change and asking whether one happened
    // sixty times a minute in a tab nobody is looking at.

    function shotImages() {
        const gallery = byId(HOST.gallery);
        if (!gallery || !gallery.querySelectorAll) return [];
        const seen = {};
        const found = [];
        Array.prototype.slice.call(gallery.querySelectorAll("img")).forEach(function (image) {
            const src = String(image.src
                || (image.getAttribute ? image.getAttribute("src") : "") || "");
            if (!src || seen[src]) return;
            seen[src] = true;
            found.push(src);
        });
        return found;
    }

    function readShots() {
        const found = shotImages();
        const same = found.length === state.shots.length
            && found.every(function (src, at) { return src === state.shots[at]; });
        state.shots = found;
        // A new set of results shows the newest of them; anything else leaves
        // the carousel where the user put it.
        if (!same) state.shotAt = found.length ? found.length - 1 : -1;
        if (state.shotAt >= found.length) state.shotAt = found.length - 1;
        paintShot();
    }

    // §16.3: first ← previous → last, last → next → first.
    function stepShot(direction) {
        const many = state.shots.length;
        if (!many) return;
        const at = state.shotAt < 0 ? 0 : state.shotAt;
        state.shotAt = ((at + direction) % many + many) % many;
        paintShot();
    }

    function paintShot() {
        const holder = byId(IDS.shot);
        if (!holder) return;
        const src = state.shotAt >= 0 ? state.shots[state.shotAt] : "";
        let image = holder.querySelector("img");
        if (!src) {
            holder.textContent = "";
        } else {
            if (!image) {
                image = document.createElement("img");
                image.className = P + "-shot-image";
                image.alt = "";
                holder.appendChild(image);
            }
            if (image.src !== src) image.src = src;
        }
        write(IDS.shotAt, state.shots.length
            ? (state.shotAt + 1) + " / " + state.shots.length : "");
        enable(IDS.shotPrevious, state.shots.length > 1);
        enable(IDS.shotNext, state.shots.length > 1);
    }

    function paintProgress() {
        const holder = byId(IDS.progress);
        if (!holder) return;
        const results = byId(HOST.results) || byId(HOST.gallery);
        // Two lookups rather than one descendant selector, because the host
        // rebuilds .progressDiv on every generation and the bar inside it is
        // the only thing worth finding when it is there.
        const holder_ = results && results.querySelector
            ? results.querySelector(".progressDiv") : null;
        const bar = holder_ && holder_.querySelector
            ? holder_.querySelector(".progress") : null;
        if (!bar) {
            holder.hidden = true;
            write(IDS.progressRead, "");
            return;
        }
        holder.hidden = false;
        const mine = byId(IDS.progressBar);
        if (mine && mine.style) mine.style.width = (bar.style && bar.style.width) || "0%";
        write(IDS.progressRead, String(bar.textContent || "").trim());
    }

    // §16.4's last line, and the one this whole file exists to keep true: the
    // host's Generate button, clicked, in the user's own gesture. Nothing is
    // swallowed, nothing is deferred and nothing is polled -- press it and
    // close the browser and the picture still arrives.
    function generate() {
        const button = clickable(HOST.generate);
        if (button && typeof button.click === "function") button.click();
    }

    function watchHost() {
        if (typeof MutationObserver !== "function") return;
        const gallery = byId(HOST.gallery);
        const results = byId(HOST.results) || (gallery && gallery.parentNode);
        if (!gallery && !results) return;
        // Gradio rebuilds the results column, and an observer still watching
        // the element it replaced is an observer that has stopped observing.
        // The element is remembered rather than a boolean, so a rebuild is
        // noticed by the same `wire()` pass that re-binds everything else.
        if (state.watching === (gallery || results)) return;
        if (state.watcher && typeof state.watcher.disconnect === "function") {
            try { state.watcher.disconnect(); } catch (error) { /* already gone */ }
        }
        const seen = new MutationObserver(function () {
            if (!state.open) return;
            readShots();
            paintProgress();
        });
        try {
            if (gallery) seen.observe(gallery, {childList: true, subtree: true,
                                                attributes: true,
                                                attributeFilter: ["src"]});
            if (results && results !== gallery) {
                seen.observe(results, {childList: true, subtree: true,
                                       attributes: true,
                                       attributeFilter: ["style", "class"]});
            }
            state.watching = gallery || results;
            state.watcher = seen;
        } catch (error) {
            state.watching = false;
            state.watcher = null;
        }
    }

    // ------------------------------------------------------------------ //
    // Numeric geometry -- the inspector's four boxes
    // ------------------------------------------------------------------ //
    //
    // §8.3: the canvas writes these and these write the canvas, in the one
    // coordinate system the whole document is in. §26: an edit that cannot make
    // a valid box is refused rather than applied halfway, and the field is
    // repainted from the region so the screen never shows a number the layout
    // does not hold.

    function applyBbox(settled) {
        const region = find(state.selected);
        if (!region) return;
        function read(id, fallback) {
            const field = byId(id);
            const said = field ? String(field.value).trim() : "";
            // A field somebody has just emptied on the way to typing a new
            // number is not a request for a box of zero width. Number("") is 0
            // and finite, which is exactly the wrong answer here.
            if (!said) return fallback;
            const number = Math.round(Number(said));
            return Number.isFinite(number) ? number : fallback;
        }
        const width = region.bbox[2] - region.bbox[0];
        const height = region.bbox[3] - region.bbox[1];
        const w = clamp(read(IDS.bboxW, width), MIN_SIZE, SCALE);
        const h = clamp(read(IDS.bboxH, height), MIN_SIZE, SCALE);
        const x = clamp(read(IDS.bboxX, region.bbox[0]), 0, SCALE - w);
        const y = clamp(read(IDS.bboxY, region.bbox[1]), 0, SCALE - h);
        const made = orient([x, y, x + w, y + h]);
        if (!made) { paintInspector(); return; }
        mark("bbox:" + region.id);
        region.bbox = made;
        // Repainting the inspector while somebody is mid-number would rewrite
        // the field under the cursor, so a live edit moves the frame and leaves
        // the four boxes alone until the cursor leaves them.
        if (settled) { paint(); return; }
        paintFrame();
        paintChrome();
        paintFacts();
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
        state.popup = "";
        state.placing = null;
        state.past.length = 0;
        state.future.length = 0;
        state.open = true;
        workspace.hidden = false;
        // §3.1: the workspace takes the txt2img work area rather than escaping
        // to document.body or asking the browser for the screen.
        takeover();
        say("");
        zoomTo(1);
        reframe();
        // Both taken after reframe, so that opening a workspace and leaving it
        // alone is not "changed" merely because the frame is a different size
        // than the day the layout was drawn. Opening commits nothing: an Auto
        // Save that fired on open would write a round trip for a layout
        // somebody only looked at, and the Session widget would call a
        // untouched layout unsaved.
        state.committed = serialize();
        state.baseline = state.committed;
        watchHost();
        syncFromHost();
        readShots();
        paint();
        const canvas = byId(IDS.canvas);
        if (canvas && typeof canvas.focus === "function") canvas.focus();
    }

    function close() {
        surrender();
        const workspace = byId(IDS.workspace);
        if (workspace) workspace.hidden = true;
        state.open = false;
        state.drawing = false;
        state.confirming = false;
        state.popup = "";
        state.placing = null;
        state.drag = null;
        state.working = null;
        state.past.length = 0;
        state.future.length = 0;
        clearGhost();
        dropGhost();
        release();
    }

    // §3.2. Save commits and stays: the whole point of §3.3 is that somebody
    // can save a layout, generate from it, change it and save again without
    // the workspace closing under them once.
    function save() {
        const box = stateBox();
        if (box && state.working) {
            state.committed = serialize();
            state.baseline = state.committed;
            state.pendingSave = false;
            publish(box, state.committed);
        }
        // The compact canvas is a view of the same document, so it is stale the
        // moment this writes one. Forced, because "somebody just saved a
        // layout" is exactly the case the ordinary guard refuses.
        compactLoad(true);
        if (state.working) paint();
    }

    // §3.2. Unchanged leaves at once; changed asks, in the page, with two
    // buttons and no silent save. A browser confirm() would be a modal dialog
    // reintroduced into a workspace whose whole point was to stop being one.
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
            press(IDS.draw, function () {
                if (!state.open) return;
                // §11.4: pressing Draw again cancels. One shot, and the mode
                // exits itself the moment a region is made -- see onUp().
                popupsShut();
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
            press(IDS.back, function () { restack(-Infinity); });
            press(IDS.zoomFit, function () { zoomTo(1); });
            press(IDS.zoomIn, function () { zoomStep(1); });
            press(IDS.zoomOut, function () { zoomStep(-1); });
            press(IDS.save, save);
            press(IDS.cancel, leave);
            press(IDS.discard, close);
            press(IDS.keep, function () {
                state.confirming = false;
                paintChrome();
            });

            // §4.1 and §4.2. Both of these are the panel's own controls shown
            // in the bar, so pressing one presses the component Gradio is bound
            // to and the setting is remembered exactly as it would have been.
            once(byId(IDS.power), "change", function (event) {
                toggleHost(hostCheck(HOST.toggle), !!event.target.checked);
                paintChrome();
                paintFacts();
            });
            [[IDS.modeDirect, "direct"], [IDS.modeSmart, "smart"]].forEach(function (pair) {
                press(pair[0], function () {
                    setComposeMode(pair[1]);
                    if (state.working) { mark(""); paint(); }
                    else paintChrome();
                });
            });

            // §4.3 and §4.6.
            press(IDS.quick, function () { popupOpen(IDS.quickPopup); });
            press(IDS.panels, function () { popupOpen(IDS.panelsPopup); });
            press(IDS.collapseAll, collapseAll);
            press(IDS.expandVisible, expandVisible);
            once(byId(IDS.quickSize), "input", function (event) {
                const size = Math.round(Number(event.target.value));
                if (Number.isFinite(size)) state.quickSize = clamp(size, MIN_SIZE, SCALE);
            });
            PANELS.forEach(function (key) {
                once(byId(P + "-show-" + key), "change", function (event) {
                    panelShow(key, !!event.target.checked);
                });
            });

            // §12.2's collapse, delegated: the rail's widget headers are static
            // markup, but one listener on the rail is one listener however many
            // widgets the rail grows.
            once(byId(IDS.rail), "click", function (event) {
                const target = event.target || {};
                const head = target.closest
                    ? target.closest("." + P + "-widget-head") : null;
                if (!head) return;
                prevent(event);
                const key = head.dataset.panel;
                panelCollapse(key, !state.collapsed[key]);
            });

            // §10 and §9: one gesture, both palettes. Delegated from the
            // workspace because the Quick Add buttons live in a popup that
            // opens and closes and the Person buttons do not.
            const workspace = byId(IDS.workspace);
            once(workspace, "pointerdown", function (event) {
                const target = event.target || {};
                const button = target.closest
                    ? target.closest("." + P + "-shape-button") : null;
                if (!button) return;
                paletteDown(event, button);
            }, "Palette");
            once(workspace, "pointermove", paletteMove, "Palette");
            once(workspace, "pointerup", paletteUp, "Palette");
            once(workspace, "pointercancel", paletteCancel, "Palette");
            // A press outside either popup closes it, and a press inside one
            // does not. The same listener, because "outside" is what closest()
            // answers.
            once(workspace, "click", function (event) {
                if (!state.popup) return;
                const target = event.target || {};
                const inside = target.closest
                    ? target.closest("." + P + "-menu") : null;
                if (inside) return;
                popupsShut();
            }, "Popups");

            // §13.2, both directions, and neither of them a copy: each box
            // writes the component the other reads.
            MIRRORS.forEach(function (pair) {
                once(byId(pair.here), "input", function () { syncToHost(pair); });
                once(byId(pair.here), "change", function () { syncToHost(pair); });
                const theirs = hostField(pair.there);
                once(theirs, "input", function () {
                    if (state.syncing) return;
                    syncFromHost();
                }, "Mirror");
                once(theirs, "change", function () {
                    if (state.syncing) return;
                    syncFromHost();
                }, "Mirror");
            });
            // §13.3: the same Creative Mode state as txt2img, never a second
            // local value.
            once(byId(IDS.creative), "change", function (event) {
                toggleHost(hostCheck(HOST.creative), !!event.target.checked);
                paintFacts();
            });
            once(hostCheck(HOST.creative), "change", function () {
                if (!state.open) return;
                syncFromHost();
                paintFacts();
            }, "Mirror");
            once(hostCheck(HOST.toggle), "change", function () {
                if (!state.open) return;
                paintChrome();
                paintFacts();
            }, "Mirror");

            // §16.
            press(IDS.shotPrevious, function () { stepShot(-1); });
            press(IDS.shotNext, function () { stepShot(1); });
            press(IDS.generate, generate);

            // §17's optional half: the frame's dimensions, edited from the
            // workspace, written into the host's own number fields so that the
            // change is the same change the sliders make.
            [[IDS.sizeWidth, HOST.width], [IDS.sizeHeight, HOST.height]]
                .forEach(function (pair) {
                    once(byId(pair[0]), "change", function (event) {
                        const number = Math.round(Number(event.target.value));
                        if (!Number.isFinite(number) || number <= 0) return;
                        const field = hostField(pair[1]);
                        if (!field || Number(field.value) === number) return;
                        publish(field, String(number));
                    });
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

            // §9.6. Rotation is editor state and only that: it changes the
            // silhouette and the restoration document, and it is applied here
            // rather than anywhere near orient() so that there is no arithmetic
            // in this file by which it could reach a coordinate.
            field(IDS.rotation, function (region, value) {
                region.rotation = rotate(value);
            }, true);

            // §8.3. Live while a number is being typed -- the frame follows the
            // field -- and normalized when the cursor leaves, which is the one
            // moment the field may be rewritten under somebody's hands.
            [IDS.bboxX, IDS.bboxY, IDS.bboxW, IDS.bboxH].forEach(function (id) {
                once(byId(id), "input", function () {
                    state.typing = true;
                    try { applyBbox(false); } finally { state.typing = false; }
                });
                once(byId(id), "change", function () {
                    applyBbox(true);
                    state.editing = "";
                    paint();
                });
            });

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
            }

            // §16.2 and §16.5, hooked up whether or not the workspace is open:
            // observers cost nothing while nothing changes, and installing them
            // on open would miss a generation started before it.
            watchHost();

            // Gradio rebuilding the tab under an open workspace would otherwise
            // leave it open in `state` and hidden on screen -- and, since §3.1's
            // takeover is marks on the ancestors Gradio just replaced, leave the
            // rest of txt2img showing through underneath it.
            if (state.open && state.working) {
                const live = byId(IDS.workspace);
                if (live) live.hidden = false;
                takeover();
                zoomTo(state.zoom);
                reframe();
                syncFromHost();
                readShots();
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
        note.textContent = "Drag a box to move it. Full Screen for everything else.";
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
            // §11.4 and §4.3, innermost first: a menu, then a gesture, then the
            // confirmation, then the workspace itself. Escape never skips a
            // level, so it is never the key that closed something the user was
            // still looking at.
            if (state.popup) { popupsShut(); return; }
            if (state.placing) { paletteCancel(); return; }
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
        dimensions: dimensions,
        placeShape: placeShape,
        popupOpen: popupOpen,
        popupsShut: popupsShut,
        panelShow: panelShow,
        panelCollapse: panelCollapse,
        collapseAll: collapseAll,
        expandVisible: expandVisible,
        composeMode: composeMode,
        setComposeMode: setComposeMode,
        spatialOn: spatialOn,
        syncFromHost: syncFromHost,
        readShots: readShots,
        stepShot: stepShot,
        paintProgress: paintProgress,
        generate: generate,
        takeover: takeover,
        surrender: surrender,
        shapes: SHAPES,
        panels: PANELS,
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
