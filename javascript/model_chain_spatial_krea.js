// Model Chain -- Krea Creative Mode Spatial Layout, browser side.
//
// This file owns one thing: the layout editor. Somebody presses Edit Layout, an
// overlay opens, they draw boxes and type into them, they press Save & Close,
// and the serialized document goes into a hidden textbox that travels with the
// next Generate like any other control. That is the whole contract.
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

(function () {
    "use strict";

    const P = "mc-krea-spatial";
    const SCALE = 1000;
    const MIN_SIZE = 8;          // normalized units; smaller than this is a click
    const MAX_REGIONS = 24;      // matches prompt_master/krea/spatial.MAX_REGIONS

    const IDS = {
        open: P + "-open",
        state: P + "-state",
        overlay: P + "-overlay",
        canvas: P + "-canvas",
        regions: P + "-regions",
        list: P + "-list",
        name: P + "-name",
        type: P + "-type",
        text: P + "-text",
        textField: P + "-text-field",
        prompt: P + "-prompt",
        promptLabel: P + "-prompt-label",
        framing: P + "-framing",
        framingField: P + "-framing-field",
        angle: P + "-angle",
        angleField: P + "-angle-field",
        autoHint: P + "-auto-hint",
        bbox: P + "-bbox",
        size: P + "-size",
        warning: P + "-warning",
        draw: P + "-draw",
        duplicate: P + "-duplicate",
        remove: P + "-delete",
        forward: P + "-raise",
        back: P + "-lower",
        save: P + "-save",
        cancel: P + "-cancel",
    };

    // Everything the editor is doing right now. One object so a test can read
    // it, and so "what state is this in" has exactly one answer.
    const state = {
        wired: false,
        listening: false,   // the document-level listeners, installed once
        overlay: null,      // the overlay element that is live, once adopted
        open: false,
        drawing: false,
        working: null,      // the layout being edited; null while closed
        selected: "",       // region id
        drag: null,         // {kind, id, startX, startY, origin}
        counter: 0,
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

    // ------------------------------------------------------------------ //
    // The document
    // ------------------------------------------------------------------ //

    function blank() {
        return {
            version: 1,
            canvas: {width: 0, height: 0, grid: "thirds"},
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
            text: String(entry.text || ""),
            framing: String(entry.framing || ""),
            angle: String(entry.angle || ""),
            z: Number.isFinite(Number(entry.z)) ? Number(entry.z) : 0,
        };
    }

    // The canonical box, defined here the same way prompt_master/krea/spatial.py
    // defines it: clamped, ordered, and rejected outright when it has no area.
    // A drag from bottom-right to top-left is not an error and a click is not a
    // region.
    function orient(values) {
        if (!Array.isArray(values) || values.length < 4) return null;
        const numbers = values.slice(0, 4).map(function (value) {
            const number = Math.round(Number(value));
            if (!Number.isFinite(number)) return 0;
            return Math.max(0, Math.min(SCALE, number));
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

    function ordered() {
        return state.working.regions
            .map(function (region, index) { return {region: region, index: index}; })
            .sort(function (left, right) {
                return (left.region.z - right.region.z) || (left.index - right.index);
            })
            .map(function (entry) { return entry.region; });
    }

    function find(id) {
        return state.working.regions.filter(function (region) {
            return region.id === id;
        })[0] || null;
    }

    function serialize() {
        const document_ = {
            version: 1,
            canvas: {
                width: Number(state.working.canvas.width) || 0,
                height: Number(state.working.canvas.height) || 0,
                grid: state.working.canvas.grid || "thirds",
            },
            compose_mode: state.working.compose_mode,
            auto_position_hint: !!state.working.auto_position_hint,
            regions: ordered().map(function (region) {
                const entry = {
                    id: region.id,
                    name: region.name || region.id,
                    type: region.type,
                    bbox: region.bbox.slice(),
                    prompt: region.prompt,
                };
                if (region.type === "text") entry.text = region.text;
                entry.framing = region.framing;
                entry.angle = region.angle;
                entry.z = region.z;
                return entry;
            }),
        };
        return JSON.stringify(document_);
    }

    // ------------------------------------------------------------------ //
    // The frame
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

    // Normalized coordinates are fractions of the frame, so a resolution change
    // moves nothing and a *ratio* change reframes everything. The boxes are left
    // exactly where they are and the change is reported, which is the design
    // intent's rule in the strongest terms it uses: never silently delete layout
    // state. Reprojecting would be this file deciding which of somebody's boxes
    // deserved to keep its shape.
    function reframe() {
        const size = hostSize();
        const canvas = byId(IDS.canvas);
        const warning = byId(IDS.warning);
        const note = byId(IDS.size);
        const before = state.working.canvas;
        if (size.width > 0 && size.height > 0) {
            if (canvas) canvas.style.aspectRatio = size.width + " / " + size.height;
            if (note) note.textContent = size.width + " × " + size.height;
            const was = (Number(before.width) || 0) / (Number(before.height) || 1);
            const now = size.width / size.height;
            const changed = Number(before.width) > 0 && Number(before.height) > 0
                && Math.abs(was - now) / now > 0.02;
            if (warning) {
                warning.hidden = !changed;
                warning.textContent = changed
                    ? "The frame is a different shape than when this layout was drawn "
                      + "— the boxes are unchanged, but they now cover different parts "
                      + "of the picture."
                    : "";
            }
            state.working.canvas.width = size.width;
            state.working.canvas.height = size.height;
        } else if (note) {
            note.textContent = "";
        }
    }

    // ------------------------------------------------------------------ //
    // Painting
    // ------------------------------------------------------------------ //

    function paint() {
        const holder = byId(IDS.regions);
        if (!holder || !state.working) return;
        holder.textContent = "";
        ordered().forEach(function (region, depth) {
            holder.appendChild(box(region, depth));
        });
        paintList();
        paintInspector();
    }

    function box(region, depth) {
        const element = document.createElement("div");
        element.className = P + "-region" + (region.id === state.selected ? " selected" : "")
            + (region.type === "text" ? " text" : "");
        element.dataset.region = region.id;
        element.style.left = (region.bbox[0] / SCALE * 100) + "%";
        element.style.top = (region.bbox[1] / SCALE * 100) + "%";
        element.style.width = ((region.bbox[2] - region.bbox[0]) / SCALE * 100) + "%";
        element.style.height = ((region.bbox[3] - region.bbox[1]) / SCALE * 100) + "%";
        element.style.zIndex = String(depth + 1);
        element.title = region.name || region.id;

        const label = document.createElement("span");
        label.className = P + "-label";
        label.textContent = region.name || region.id;
        element.appendChild(label);

        ["nw", "ne", "se", "sw"].forEach(function (corner) {
            const handle = document.createElement("span");
            handle.className = P + "-handle " + P + "-handle-" + corner;
            handle.dataset.corner = corner;
            element.appendChild(handle);
        });
        return element;
    }

    function paintList() {
        const list = byId(IDS.list);
        if (!list) return;
        list.textContent = "";
        ordered().forEach(function (region) {
            const option = document.createElement("option");
            option.value = region.id;
            option.textContent = (region.type === "text" ? "T  " : "□  ")
                + (region.name || region.id);
            option.selected = region.id === state.selected;
            list.appendChild(option);
        });
    }

    function paintInspector() {
        const region = find(state.selected);
        const fields = [IDS.name, IDS.type, IDS.text, IDS.prompt, IDS.framing, IDS.angle];
        fields.forEach(function (id) {
            const field = byId(id);
            if (field) field.disabled = !region;
        });
        const readout = byId(IDS.bbox);
        const auto = byId(IDS.autoHint);
        if (auto) auto.checked = !!state.working.auto_position_hint;
        if (!region) {
            if (readout) readout.textContent = "—";
            show(IDS.textField, false);
            return;
        }
        set(IDS.name, region.name);
        set(IDS.type, region.type);
        set(IDS.text, region.text);
        set(IDS.prompt, region.prompt);
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
        if (readout) readout.textContent = region.bbox.join(", ");
    }

    function set(id, value) {
        const field = byId(id);
        if (field && field.value !== value) field.value = value;
    }

    function show(id, visible) {
        const field = byId(id);
        if (field) field.hidden = !visible;
    }

    // ------------------------------------------------------------------ //
    // Editing
    // ------------------------------------------------------------------ //

    function select(id) {
        state.selected = id || "";
        paint();
    }

    function create(bbox) {
        if (state.working.regions.length >= MAX_REGIONS) {
            const warning = byId(IDS.warning);
            if (warning) {
                warning.hidden = false;
                warning.textContent = "That is as many regions as one layout carries ("
                    + MAX_REGIONS + ").";
            }
            return null;
        }
        const highest = state.working.regions.reduce(function (top, region) {
            return Math.max(top, region.z);
        }, -1);
        const region = {
            id: nextId(),
            name: "Region " + (state.working.regions.length + 1),
            type: "obj",
            bbox: bbox,
            prompt: "",
            text: "",
            framing: "",
            angle: "",
            z: highest + 1,
        };
        state.working.regions.push(region);
        select(region.id);
        const field = byId(IDS.prompt);
        if (field && typeof field.focus === "function") field.focus();
        return region;
    }

    function remove() {
        if (!state.selected) return;
        state.working.regions = state.working.regions.filter(function (region) {
            return region.id !== state.selected;
        });
        select("");
    }

    function duplicate() {
        const region = find(state.selected);
        if (!region) return;
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
    function restack(direction) {
        const list = ordered();
        const at = list.findIndex(function (region) { return region.id === state.selected; });
        if (at < 0) return;
        const to = at + direction;
        if (to < 0 || to >= list.length) return;
        const moved = list.slice();
        moved.splice(to, 0, moved.splice(at, 1)[0]);
        moved.forEach(function (region, index) { region.z = index; });
        paint();
    }

    // ------------------------------------------------------------------ //
    // The canvas
    // ------------------------------------------------------------------ //

    function point(event) {
        const canvas = byId(IDS.canvas);
        if (!canvas) return {x: 0, y: 0};
        const rect = canvas.getBoundingClientRect();
        if (!rect.width || !rect.height) return {x: 0, y: 0};
        return {
            x: Math.max(0, Math.min(SCALE, Math.round((event.clientX - rect.left) / rect.width * SCALE))),
            y: Math.max(0, Math.min(SCALE, Math.round((event.clientY - rect.top) / rect.height * SCALE))),
        };
    }

    function onDown(event) {
        if (!state.open || !state.working) return;
        const at = point(event);
        const handle = event.target.closest ? event.target.closest("." + P + "-handle") : null;
        const region = event.target.closest ? event.target.closest("." + P + "-region") : null;

        if (state.drawing) {
            event.preventDefault();
            state.drag = {kind: "draw", from: at, to: at};
            paintGhost();
            return;
        }
        if (handle && region) {
            event.preventDefault();
            select(region.dataset.region);
            const chosen = find(state.selected);
            if (!chosen) return;
            state.drag = {kind: "resize", corner: handle.dataset.corner,
                          origin: chosen.bbox.slice(), from: at};
            return;
        }
        if (region) {
            event.preventDefault();
            select(region.dataset.region);
            const chosen = find(state.selected);
            if (!chosen) return;
            state.drag = {kind: "move", origin: chosen.bbox.slice(), from: at};
            return;
        }
        select("");
    }

    function onMove(event) {
        if (!state.drag) return;
        const at = point(event);
        if (state.drag.kind === "draw") {
            state.drag.to = at;
            paintGhost();
            return;
        }
        const region = find(state.selected);
        if (!region) return;
        const dx = at.x - state.drag.from.x;
        const dy = at.y - state.drag.from.y;
        const origin = state.drag.origin;
        if (state.drag.kind === "move") {
            const width = origin[2] - origin[0];
            const height = origin[3] - origin[1];
            const x0 = Math.max(0, Math.min(SCALE - width, origin[0] + dx));
            const y0 = Math.max(0, Math.min(SCALE - height, origin[1] + dy));
            region.bbox = [x0, y0, x0 + width, y0 + height];
        } else {
            const corner = state.drag.corner;
            const box_ = origin.slice();
            if (corner.indexOf("w") >= 0) box_[0] = origin[0] + dx;
            if (corner.indexOf("e") >= 0) box_[2] = origin[2] + dx;
            if (corner.indexOf("n") >= 0) box_[1] = origin[1] + dy;
            if (corner.indexOf("s") >= 0) box_[3] = origin[3] + dy;
            const settled = orient(box_);
            if (settled) region.bbox = settled;
        }
        paint();
    }

    function onUp() {
        const drag = state.drag;
        state.drag = null;
        clearGhost();
        if (!drag) return;
        if (drag.kind === "draw") {
            const made = orient([drag.from.x, drag.from.y, drag.to.x, drag.to.y]);
            state.drawing = false;
            paintDrawButton();
            // A click is not a region, and a two-pixel drag is a click with a
            // shaky hand. Both are dropped rather than turned into a box the
            // compositor would refuse later, where the reason would arrive on
            // the finished image instead of under the cursor.
            if (made && (made[2] - made[0]) >= MIN_SIZE && (made[3] - made[1]) >= MIN_SIZE) {
                create(made);
                return;
            }
        }
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

    function paintDrawButton() {
        const button = byId(IDS.draw);
        if (button) button.classList.toggle("active", state.drawing);
    }

    // ------------------------------------------------------------------ //
    // Opening, saving, closing
    // ------------------------------------------------------------------ //

    function open() {
        const overlay = byId(IDS.overlay);
        if (!overlay) return;
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
        state.counter = read.regions.reduce(function (top, region) {
            const number = parseInt(String(region.id).replace(/^r/, ""), 10);
            return Number.isFinite(number) ? Math.max(top, number) : top;
        }, 0);
        state.selected = read.regions.length ? ordered()[0].id : "";
        state.drawing = false;
        state.open = true;
        overlay.hidden = false;
        reframe();
        paintDrawButton();
        paint();
        const canvas = byId(IDS.canvas);
        if (canvas && typeof canvas.focus === "function") canvas.focus();
    }

    function close() {
        const overlay = byId(IDS.overlay);
        if (overlay) overlay.hidden = true;
        state.open = false;
        state.drawing = false;
        state.drag = null;
        state.working = null;
    }

    function save() {
        const box = stateBox();
        if (box && state.working) publish(box, serialize());
        close();
    }

    // ------------------------------------------------------------------ //
    // Wiring
    // ------------------------------------------------------------------ //

    // Elements only. A dataset flag is what makes re-wiring idempotent, and
    // `document` has no dataset -- so passing it here would add a fresh listener
    // on every onAfterUiUpdate, which fires often. The document-level listeners
    // are installed once, under their own flag, in wire().
    function once(element, event, handler) {
        if (!element || !element.dataset) return;
        const flag = "mcKreaSpatial" + event;
        if (element.dataset[flag]) return;
        element.dataset[flag] = "1";
        element.addEventListener(event, handler);
    }

    // The overlay, moved to the body and kept singular.
    //
    // Two things make this more than one line. A `position: fixed` modal inside
    // a Gradio accordion is one `overflow: hidden` or one `transform` away from
    // being a modal nobody can see, and neither of those is this extension's to
    // promise about a theme -- so it is moved. And if Gradio ever rebuilds the
    // HTML component, the rebuilt copy arrives carrying every id the moved one
    // already has, so whichever `byId` found first would be a coin toss between
    // a wired overlay and an unwired one. The already-adopted copy wins and the
    // duplicate is removed: the markup is static, so the two are the same page,
    // and only one of them has the listeners.
    function adopt() {
        let found;
        try {
            found = Array.prototype.slice.call(
                (root().querySelectorAll ? root() : document)
                    .querySelectorAll("#" + IDS.overlay));
        } catch (error) {
            found = [];
        }
        if (!found.length) return null;
        const keep = (state.overlay && found.indexOf(state.overlay) >= 0)
            ? state.overlay : found[0];
        found.forEach(function (element) {
            if (element !== keep && element.parentNode) {
                element.parentNode.removeChild(element);
            }
        });
        if (keep.parentNode !== document.body) document.body.appendChild(keep);
        state.overlay = keep;
        return keep;
    }

    function field(id, apply) {
        const element = byId(id);
        once(element, "input", function () {
            const region = find(state.selected);
            if (!region) return;
            apply(region, element.value);
            if (id === IDS.name || id === IDS.type) paint();
        });
        once(element, "change", function () {
            const region = find(state.selected);
            if (!region) return;
            apply(region, element.value);
            paint();
        });
    }

    function wire() {
        try {
            if (!adopt()) return;

            once(clickable(IDS.open) || byId(IDS.open), "click", function (event) {
                event.preventDefault();
                open();
            });
            once(byId(IDS.draw), "click", function () {
                state.drawing = !state.drawing;
                paintDrawButton();
            });
            once(byId(IDS.duplicate), "click", function () { duplicate(); });
            once(byId(IDS.remove), "click", function () { remove(); });
            once(byId(IDS.forward), "click", function () { restack(1); });
            once(byId(IDS.back), "click", function () { restack(-1); });
            once(byId(IDS.save), "click", save);
            once(byId(IDS.cancel), "click", close);
            once(byId(IDS.list), "change", function (event) {
                select(event.target.value);
            });
            once(byId(IDS.autoHint), "change", function (event) {
                if (state.working) state.working.auto_position_hint = !!event.target.checked;
            });

            field(IDS.name, function (region, value) { region.name = value; });
            field(IDS.type, function (region, value) {
                region.type = value === "text" ? "text" : "obj";
            });
            field(IDS.text, function (region, value) { region.text = value; });
            field(IDS.prompt, function (region, value) { region.prompt = value; });
            field(IDS.framing, function (region, value) { region.framing = value; });
            field(IDS.angle, function (region, value) { region.angle = value; });

            once(byId(IDS.canvas), "mousedown", onDown);

            // Once, ever. A drag that leaves the canvas still has to be
            // followed, and a key pressed with focus anywhere in the modal still
            // has to be heard, so these three live on the document -- which
            // means they are the three that a dataset flag cannot guard.
            if (!state.listening) {
                state.listening = true;
                document.addEventListener("mousemove", onMove);
                document.addEventListener("mouseup", onUp);
                document.addEventListener("keydown", onKey);
            }

            state.wired = true;
        } catch (error) {
            console.error("Model Chain: the Spatial Layout wiring failed", error);
        }
    }

    function onKey(event) {
        if (!state.open) return;
        const target = event.target || {};
        const typing = target.tagName === "INPUT" || target.tagName === "TEXTAREA"
            || target.isContentEditable;
        if (event.key === "Escape") {
            event.preventDefault();
            if (state.drag || state.drawing) {
                state.drag = null;
                state.drawing = false;
                clearGhost();
                paintDrawButton();
                paint();
                return;
            }
            close();
            return;
        }
        if (typing) return;
        if (event.key === "Delete" || event.key === "Backspace") {
            event.preventDefault();
            remove();
        }
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
        save: save,
        serialize: function () { return state.working ? serialize() : ""; },
        orient: orient,
        create: create,
        select: select,
        remove: remove,
        duplicate: duplicate,
        restack: restack,
        ordered: function () { return state.working ? ordered() : []; },
    };
})();
