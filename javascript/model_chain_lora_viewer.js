// Model Chain -- the LoRA viewer.
//
// A button at the end of the prompt tool row, after Forge's Apply Styles, on
// Txt2Img and Img2Img. Pressed, it covers the page under the tab bar with every
// LoRA in the prompt -- and on Txt2Img in the two Literal boxes -- one thin row
// each: its picture, its name, the box it is in, its trigger words (a tap copies
// one), its weight in a box that takes any number, and a slider from -5 to 5 in
// steps of 0.05, with - and + at its ends for the last step a finger cannot make.
// Cancel changes nothing. Apply writes each changed weight into its tag where the
// tag is, takes out the tags marked for removal, and leaves every other
// character of the prompt as it was.
//
// What it reads and writes
// ------------------------
// The tags are read here, from the boxes, the moment the viewer opens, so the
// names and weights are on screen at once. The pictures and trigger words come
// from the extension's route (mc_lora_viewer.py) in one request with a deadline:
// a request without one is a page that can stall on a connection that has
// stopped answering, and a stalled viewer would be a modal nobody could use.
// Written back, a box gets the input event Forge's own `updateInput` sends, so
// Gradio holds the value a press of Generate sends; a Literal box also gets the
// blur its own handler saves it on. A box whose text changed while the viewer
// was open is not written at all, and the page says so.
//
// What it must not do
// -------------------
// * Move a slider when a finger only meant to scroll the list. A sideways drag,
//   a tap on the track, a drag of the knob and the keys move one; a vertical
//   swipe is the browser's (`touch-action: pan-y` on the track).
// * Let Escape reach anything else. Forge's own Escape interrupts a running
//   generation, and the Forge Assistant's leaves focus mode; the viewer takes
//   Escape on the window's capture phase while it is open, before either.
// * Close on a stray tap. Nothing outside it closes it but Escape and a change
//   of tab, which is Cancel.
// * Be covered. It draws on the dialog layer Mini Paint NEO and the assistant
//   share (`--minipaint-dialog-layer`, 2000) and announces itself with the
//   `minipaint:overlay` event, which puts the assistant's panel away.
// * Write a `×` as text, or size a button by its own text: a theme replaces
//   every span holding one with an icon and says `min-width: fit-content
//   !important` about every button. Icons are drawn; a pill's text is a span
//   with its own width.

(function () {
    "use strict";

    if (window.mcLoraViewer) return;

    const VERSION = 1;
    const ROUTE = "/model-chain/lora-viewer/details";
    const DEADLINE_MS = 8000;
    const MIN = -5;
    const MAX = 5;
    const STEP = 0.05;
    const PAD = 11;          // the track's inset: the knob's radius, so its centre reaches both ends
    const TAP_SLOP = 6;      // pixels a finger may move and still be a tap on the track
    const SCROLL_SLOP = 8;   // pixels of vertical travel that make a touch a scroll
    const REPEAT_AFTER = 380;
    const REPEAT_EVERY = 70;
    const TOAST_MS = 4000;
    const OVERLAY = "minipaint:overlay";

    // The tags the viewer edits: Forge loads <lyco:...> as a LoRA too.
    const TAG = /<(lora|lyco):([^:<>]+)((?::[^<>]*)?)>/gi;
    // What is not prompt words, for "is this trigger already in the prompt".
    const ANY_TAG = /<(?:lora|lyco|hypernet):[^<>]*>/gi;

    const LITERAL_POSITIVE = "mc-krea-creative-literal-positive";
    const LITERAL_NEGATIVE = "mc-krea-creative-literal-negative";
    const TABS = {
        txt2img: {
            tools: "txt2img_tools", after: "txt2img_style_apply", panel: "tab_txt2img",
            boxes: [
                {id: "txt2img_prompt", label: "Prompt"},
                {id: LITERAL_POSITIVE, label: "Literal +", literal: true},
                {id: LITERAL_NEGATIVE, label: "Literal −", literal: true},
            ],
        },
        img2img: {
            tools: "img2img_tools", after: "img2img_style_apply", panel: "tab_img2img",
            boxes: [{id: "img2img_prompt", label: "Prompt"}],
        },
    };

    // Lucide (ISC), drawn as SVG: no glyph a theme could swap.
    const ICONS = {
        "sliders-vertical": '<path d="M10 8h4"/><path d="M12 21v-9"/><path d="M12 8V3"/><path d="M17 16h4"/><path d="M19 12V3"/><path d="M19 21v-5"/><path d="M3 14h4"/><path d="M5 10V3"/><path d="M5 21v-7"/>',
        "trash": '<path d="M10 11v6"/><path d="M14 11v6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6"/><path d="M3 6h18"/><path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>',
        "undo-2": '<path d="M9 14 4 9l5-5"/><path d="M4 9h10.5a5.5 5.5 0 0 1 5.5 5.5a5.5 5.5 0 0 1-5.5 5.5H11"/>',
        "rotate-ccw": '<path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/>',
        "check": '<path d="M20 6 9 17l-5-5"/>',
        "copy": '<rect width="14" height="14" x="8" y="8" rx="2" ry="2"/><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/>',
        "minus": '<path d="M5 12h14"/>',
        "plus": '<path d="M5 12h14"/><path d="M12 5v14"/>',
        "triangle-alert": '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
    };

    function icon(name) {
        return '<svg class="mc-lv-icon" viewBox="0 0 24 24" width="24" height="24" fill="none" '
            + 'stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" '
            + 'aria-hidden="true" focusable="false">' + (ICONS[name] || "") + "</svg>";
    }

    // -- small helpers ----------------------------------------------------------

    function app() {
        try {
            return (typeof gradioApp === "function" && gradioApp()) || document;
        } catch (error) {
            return document;
        }
    }

    function byId(id) {
        try {
            return app().querySelector("#" + id) || document.getElementById(id);
        } catch (error) {
            return document.getElementById(id);
        }
    }

    function fieldOf(id) {
        const holder = byId(id);
        if (!holder) return null;
        if (holder.matches && holder.matches("textarea, input")) return holder;
        return holder.querySelector("textarea, input");
    }

    function make(tag, className, text) {
        const node = document.createElement(tag);
        if (className) node.className = className;
        if (text !== undefined) node.textContent = text;
        return node;
    }

    function fmt(value) {
        const rounded = Math.round(value * 10000) / 10000;
        return Object.is(rounded, -0) ? "0" : String(rounded);
    }

    function snap(value) {
        return Number((Math.round(value / STEP) * STEP).toFixed(2));
    }

    function same(a, b) {
        return Math.abs(a - b) < 1e-9;
    }

    function place(value) {
        return (Math.min(MAX, Math.max(MIN, value)) - MIN) / (MAX - MIN);
    }

    function parseWeight(text) {
        const s = String(text).trim().replace(",", ".").replace("−", "-");
        if (!/^[-+]?(\d+\.?\d*|\.\d+)$/.test(s)) return null;
        const value = Number(s);
        return Number.isFinite(value) ? value : null;
    }

    function initials(name) {
        const parts = String(name).split(/[_\-\s.]+/).filter(Boolean);
        const text = parts.length > 1 ? parts[0][0] + parts[1][0] : String(name).slice(0, 2);
        return text.toUpperCase();
    }

    function escapeRegExp(text) {
        return String(text).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    }

    // -- reading the boxes ------------------------------------------------------

    function boxesOf(tab) {
        const found = [];
        (TABS[tab] ? TABS[tab].boxes : []).forEach(function (box) {
            const field = fieldOf(box.id);
            if (field) found.push({id: box.id, label: box.label, literal: !!box.literal, field: field,
                                  text: String(field.value || "")});
        });
        return found;
    }

    // One row per tag, in the order of the boxes on screen and of the tags in each.
    // `tail` is everything after the first weight, kept exactly as typed: a second
    // (UNet) weight, or Forge's named options. A tag whose first option is not a
    // plain number (`te=0.5`, Forge's `@` form) has its weights set by name, which
    // a slider of the first weight could not change: it is shown fixed.
    function readRows(boxes) {
        const rows = [];
        boxes.forEach(function (box) {
            TAG.lastIndex = 0;
            let match;
            while ((match = TAG.exec(box.text))) {
                const parts = match[3] ? match[3].slice(1).split(":") : [];
                let weight = 1;
                let tail = "";
                let fixed = false;
                if (parts.length && parseWeight(parts[0]) !== null) {
                    weight = parseWeight(parts[0]);
                    tail = parts.slice(1).map(function (part) { return ":" + part; }).join("");
                } else if (parts.length) {
                    fixed = true;
                }
                rows.push({
                    box: box, start: match.index, end: match.index + match[0].length,
                    kind: match[1], rawName: match[2], name: match[2].trim(),
                    weight: weight, orig: weight, tail: tail, fixed: fixed,
                    removed: false, editing: false, el: null,
                });
            }
        });
        return rows;
    }

    // The prompt's words, without its tags: what a trigger is looked for in.
    function wordsOf(boxes) {
        return boxes.map(function (box) { return box.text.replace(ANY_TAG, " "); }).join("\n");
    }

    function inPrompt(words, trigger) {
        const word = String(trigger || "").trim();
        if (!word) return false;
        try {
            return new RegExp("(^|[^\\p{L}\\p{N}_])" + escapeRegExp(word) + "($|[^\\p{L}\\p{N}_])", "iu")
                .test(words);
        } catch (error) {
            return words.toLowerCase().indexOf(word.toLowerCase()) !== -1;
        }
    }

    // -- the dialog -------------------------------------------------------------

    const state = {
        tab: "", boxes: [], rows: [], open: false, button: null, request: null,
        timer: 0, observer: null, toastTimer: 0, nodes: null, listening: false,
    };

    function build() {
        if (state.nodes) return state.nodes;
        const scrim = make("div", "mc-lv-scrim");
        scrim.hidden = true;
        scrim.setAttribute("aria-hidden", "true");
        const dialog = make("div", "mc-lv");
        dialog.id = "mc-lora-viewer";
        dialog.hidden = true;
        dialog.tabIndex = -1;
        dialog.setAttribute("role", "dialog");
        dialog.setAttribute("aria-modal", "true");
        dialog.setAttribute("aria-labelledby", "mc-lora-viewer-title");
        dialog.innerHTML =
            '<div class="mc-lv-head">'
            + '<div class="mc-lv-title"><b id="mc-lora-viewer-title">LoRAs</b>'
            + '<span class="mc-lv-count"></span></div>'
            + '<button type="button" class="mc-lv-text mc-lv-reset-all" hidden>'
            + icon("rotate-ccw") + "<span>Reset all</span></button>"
            + "</div>"
            + '<div class="mc-lv-status" hidden><span></span>'
            + '<button type="button" class="mc-lv-text mc-lv-retry"><span>Try again</span></button></div>'
            + '<div class="mc-lv-list" role="list"></div>'
            + '<div class="mc-lv-foot">'
            + '<button type="button" class="mc-lv-button-cancel">Cancel</button>'
            + '<button type="button" class="mc-lv-button-apply">Apply</button>'
            + "</div>";
        const toast = make("div", "mc-lv-toast");
        toast.hidden = true;
        toast.setAttribute("role", "status");
        document.body.appendChild(scrim);
        document.body.appendChild(dialog);
        document.body.appendChild(toast);

        const nodes = {
            scrim: scrim, dialog: dialog, toast: toast,
            count: dialog.querySelector(".mc-lv-count"),
            resetAll: dialog.querySelector(".mc-lv-reset-all"),
            status: dialog.querySelector(".mc-lv-status"),
            statusText: dialog.querySelector(".mc-lv-status span"),
            retry: dialog.querySelector(".mc-lv-retry"),
            list: dialog.querySelector(".mc-lv-list"),
            cancel: dialog.querySelector(".mc-lv-button-cancel"),
            apply: dialog.querySelector(".mc-lv-button-apply"),
        };
        nodes.cancel.addEventListener("click", function () { close(false); });
        nodes.apply.addEventListener("click", function () { close(true); });
        nodes.resetAll.addEventListener("click", resetAll);
        nodes.retry.addEventListener("click", function () { loadDetails(); });
        dialog.addEventListener("keydown", trapTab);
        // The page under the scrim does not scroll while the viewer is open.
        scrim.addEventListener("wheel", function (event) { event.preventDefault(); }, {passive: false});
        scrim.addEventListener("touchmove", function (event) { event.preventDefault(); }, {passive: false});
        state.nodes = nodes;
        return nodes;
    }

    function renderRows() {
        const nodes = state.nodes;
        nodes.list.textContent = "";
        if (!state.rows.length) {
            const empty = make("div", "mc-lv-empty");
            empty.appendChild(make("b", "", state.tab === "img2img"
                ? "No LoRAs in the prompt."
                : "No LoRAs in the prompt or the Literal boxes."));
            empty.appendChild(make("span", "", "Add one from the LoRA browser, then open this again."));
            nodes.list.appendChild(empty);
        } else {
            const fragment = document.createDocumentFragment();
            state.rows.forEach(function (row, index) { fragment.appendChild(rowNode(row, index)); });
            nodes.list.appendChild(fragment);
        }
        const count = state.rows.length;
        nodes.count.textContent = count === 1 ? "1 in this prompt" : count + " in this prompt";
        tally();
    }

    function rowNode(row, index) {
        const node = make("div", "mc-lv-row");
        node.setAttribute("role", "listitem");
        let ticks = "";
        for (let k = 0; k <= 10; k += 1) {
            ticks += '<span class="mc-lv-tick' + (k === 5 ? " mc-lv-zero" : "") + '" style="--p:' + (k / 10) + '"></span>';
        }
        node.innerHTML =
            '<div class="mc-lv-thumb"><span class="mc-lv-skeleton"></span></div>'
            + '<div class="mc-lv-main"><div class="mc-lv-line"><span class="mc-lv-name"></span>'
            + '<span class="mc-lv-box"></span></div>'
            + '<div class="mc-lv-pills"><span class="mc-lv-skeleton"></span>'
            + '<span class="mc-lv-skeleton mc-lv-wide"></span></div></div>'
            + '<div class="mc-lv-controls">'
            + '<button type="button" class="mc-lv-icon-button mc-lv-reset" hidden>' + icon("rotate-ccw") + "</button>"
            + '<input class="mc-lv-weight" id="mc-lv-weight-' + index + '" inputmode="decimal" '
            + 'autocomplete="off" spellcheck="false" enterkeyhint="done">'
            + '<button type="button" class="mc-lv-icon-button mc-lv-remove" aria-pressed="false">' + icon("trash") + "</button>"
            + "</div>"
            + '<div class="mc-lv-slide">'
            + '<button type="button" class="mc-lv-step" data-step="-1" aria-label="Lower by 0.05">' + icon("minus") + "</button>"
            + '<div class="mc-lv-track" role="slider" tabindex="0" aria-valuemin="' + MIN + '" aria-valuemax="' + MAX + '">'
            + '<span class="mc-lv-rail"></span>' + ticks + '<span class="mc-lv-fill"></span>'
            + '<span class="mc-lv-knob"><span class="mc-lv-bubble"></span></span></div>'
            + '<button type="button" class="mc-lv-step" data-step="1" aria-label="Raise by 0.05">' + icon("plus") + "</button>"
            + "</div>";
        row.el = {
            row: node,
            thumb: node.querySelector(".mc-lv-thumb"),
            name: node.querySelector(".mc-lv-name"),
            box: node.querySelector(".mc-lv-box"),
            pills: node.querySelector(".mc-lv-pills"),
            reset: node.querySelector(".mc-lv-reset"),
            weight: node.querySelector(".mc-lv-weight"),
            remove: node.querySelector(".mc-lv-remove"),
            track: node.querySelector(".mc-lv-track"),
            fill: node.querySelector(".mc-lv-fill"),
            knob: node.querySelector(".mc-lv-knob"),
            bubble: node.querySelector(".mc-lv-bubble"),
            steps: Array.prototype.slice.call(node.querySelectorAll(".mc-lv-step")),
        };
        row.el.name.textContent = row.name;
        row.el.name.title = row.name;
        row.el.box.textContent = row.box.label;
        row.el.weight.setAttribute("aria-label", "Weight of " + row.name);
        row.el.track.setAttribute("aria-label", "Weight of " + row.name);
        row.el.reset.setAttribute("aria-label", "Back to " + fmt(row.orig));
        row.el.reset.title = "Back to " + fmt(row.orig);
        if (row.fixed) {
            node.setAttribute("data-fixed", "");
            row.el.weight.disabled = true;
            row.el.track.setAttribute("aria-disabled", "true");
            row.el.track.tabIndex = -1;
            row.el.steps.forEach(function (step) { step.disabled = true; });
            row.el.weight.title = "This tag sets its weights by name; change them in the prompt.";
        }
        labelRemove(row);
        bind(row);
        paint(row);
        return node;
    }

    function paint(row) {
        const el = row.el;
        if (!el) return;
        const t = place(row.weight);
        const zero = place(0);
        el.track.style.setProperty("--t", String(t));
        el.track.style.setProperty("--a", String(Math.min(t, zero)));
        el.track.style.setProperty("--w", String(Math.abs(t - zero)));
        const negative = row.weight < 0;
        const beyond = row.weight < MIN || row.weight > MAX;
        el.fill.classList.toggle("mc-lv-negative", negative);
        el.knob.classList.toggle("mc-lv-negative", negative);
        el.knob.classList.toggle("mc-lv-beyond", beyond);
        el.bubble.textContent = fmt(row.weight);
        el.track.setAttribute("aria-valuenow", String(Math.min(MAX, Math.max(MIN, row.weight))));
        el.track.setAttribute("aria-valuetext", fmt(row.weight) + (beyond ? ", beyond the slider" : ""));
        el.knob.title = beyond ? fmt(row.weight) + " is beyond the slider; the box holds it" : "";
        if (!row.editing) el.weight.value = row.fixed ? "named" : fmt(row.weight);
        const changed = !same(row.weight, row.orig);
        el.row.toggleAttribute("data-changed", changed);
        el.reset.hidden = !changed || row.removed;
    }

    function changes() {
        return state.rows.filter(function (row) { return row.removed || !same(row.weight, row.orig); });
    }

    function tally() {
        const count = changes().length;
        state.nodes.apply.textContent = count
            ? "Apply " + count + (count === 1 ? " change" : " changes")
            : "Apply";
        state.nodes.resetAll.hidden = !count;
    }

    function setWeight(row, value) {
        if (row.fixed || !Number.isFinite(value)) return;
        row.weight = Number(value.toFixed(4));
        paint(row);
        tally();
    }

    function stepBy(row, steps) {
        const weight = row.weight;
        const next = steps > 0
            ? Math.floor(weight / STEP + 1e-9) * STEP + STEP * steps
            : Math.ceil(weight / STEP - 1e-9) * STEP + STEP * steps;
        setWeight(row, Number(next.toFixed(2)));
    }

    function labelRemove(row) {
        const button = row.el.remove;
        button.innerHTML = icon(row.removed ? "undo-2" : "trash");
        button.setAttribute("aria-pressed", row.removed ? "true" : "false");
        button.setAttribute("aria-label", row.removed
            ? "Keep " + row.name
            : "Remove " + row.name + " from " + row.box.label);
        button.title = row.removed ? "Keep it" : "Remove from " + row.box.label + " on Apply";
    }

    function resetAll() {
        state.rows.forEach(function (row) {
            row.weight = row.orig;
            row.removed = false;
            if (row.el) {
                row.el.row.removeAttribute("data-removed");
                labelRemove(row);
                paint(row);
            }
        });
        tally();
    }

    // -- one row's controls -----------------------------------------------------

    function bind(row) {
        const el = row.el;
        let gesture = null;

        function fromX(x) {
            const rect = el.track.getBoundingClientRect();
            const t = Math.min(1, Math.max(0, (x - rect.left - PAD) / Math.max(1, rect.width - 2 * PAD)));
            setWeight(row, snap(MIN + t * (MAX - MIN)));
        }
        function begin(event) {
            gesture.live = true;
            try { el.track.setPointerCapture(gesture.id); } catch (error) { /* the pointer is gone */ }
            el.track.classList.add("mc-lv-dragging");
            fromX(event.clientX - gesture.offset);
        }
        function finish() {
            gesture = null;
            el.track.classList.remove("mc-lv-dragging");
        }

        el.track.addEventListener("pointerdown", function (event) {
            if (row.removed || row.fixed) return;
            if (event.pointerType === "mouse" && event.button !== 0) return;
            const onKnob = !!(event.target && event.target.closest && event.target.closest(".mc-lv-knob"));
            gesture = {id: event.pointerId, x: event.clientX, y: event.clientY, live: false, offset: 0};
            if (onKnob) {
                const knob = el.knob.getBoundingClientRect();
                gesture.offset = event.clientX - (knob.left + knob.width / 2);
            }
            // A mouse and a grabbed knob drag at once; a finger on the track
            // waits to see which way it goes.
            if (event.pointerType === "mouse" || onKnob) {
                event.preventDefault();
                begin(event);
            }
        });
        el.track.addEventListener("pointermove", function (event) {
            if (!gesture || event.pointerId !== gesture.id) return;
            if (!gesture.live) {
                const dx = event.clientX - gesture.x;
                const dy = event.clientY - gesture.y;
                if (Math.abs(dy) > SCROLL_SLOP && Math.abs(dy) > Math.abs(dx)) {
                    gesture = null;
                    return;
                }
                if (Math.abs(dx) <= TAP_SLOP) return;
                begin(event);
                return;
            }
            fromX(event.clientX - gesture.offset);
        });
        el.track.addEventListener("pointerup", function (event) {
            if (!gesture || event.pointerId !== gesture.id) return;
            if (!gesture.live) fromX(event.clientX);   // a tap on the track: the knob goes there
            finish();
        });
        el.track.addEventListener("pointercancel", function (event) {
            if (gesture && event.pointerId === gesture.id) finish();   // the browser took it for a scroll
        });
        el.track.addEventListener("keydown", function (event) {
            if (row.removed || row.fixed) return;
            const steps = {ArrowLeft: -1, ArrowDown: -1, ArrowRight: 1, ArrowUp: 1, PageDown: -10, PageUp: 10};
            if (Object.prototype.hasOwnProperty.call(steps, event.key)) {
                event.preventDefault();
                stepBy(row, steps[event.key]);
            } else if (event.key === "Home") {
                event.preventDefault();
                setWeight(row, MIN);
            } else if (event.key === "End") {
                event.preventDefault();
                setWeight(row, MAX);
            }
        });

        el.steps.forEach(function (button) {
            const steps = Number(button.getAttribute("data-step"));
            let wait = 0;
            let again = 0;
            function stop() {
                clearTimeout(wait);
                clearInterval(again);
                wait = again = 0;
            }
            button.addEventListener("pointerdown", function (event) {
                if (row.removed || row.fixed) return;
                if (event.pointerType === "mouse" && event.button !== 0) return;
                stepBy(row, steps);
                wait = setTimeout(function () {
                    again = setInterval(function () { stepBy(row, steps); }, REPEAT_EVERY);
                }, REPEAT_AFTER);
            });
            ["pointerup", "pointerleave", "pointercancel"].forEach(function (type) {
                button.addEventListener(type, stop);
            });
            // A held finger repeats; it does not open the browser's menu.
            button.addEventListener("contextmenu", function (event) { event.preventDefault(); });
            // The keyboard's press (a click with no pointer behind it) is one step.
            button.addEventListener("click", function (event) {
                if (event.detail === 0 && !row.removed && !row.fixed) stepBy(row, steps);
            });
        });

        el.weight.addEventListener("focus", function () {
            row.editing = true;
            try { el.weight.select(); } catch (error) { /* nothing to select */ }
        });
        el.weight.addEventListener("input", function () {
            const raw = el.weight.value.trim();
            const value = parseWeight(raw);
            el.weight.toggleAttribute("data-bad",
                value === null && ["", "-", ".", "-.", "+"].indexOf(raw) === -1);
            if (value !== null) setWeight(row, value);
        });
        el.weight.addEventListener("blur", function () {
            row.editing = false;
            el.weight.removeAttribute("data-bad");
            el.weight.value = fmt(row.weight);
        });
        el.weight.addEventListener("keydown", function (event) {
            if (event.key === "Enter") {
                event.preventDefault();
                el.weight.blur();
            }
        });

        el.reset.addEventListener("click", function () {
            setWeight(row, row.orig);
            el.track.focus({preventScroll: true});
        });
        el.remove.addEventListener("click", function () {
            row.removed = !row.removed;
            el.row.toggleAttribute("data-removed", row.removed);
            labelRemove(row);
            paint(row);
            tally();
        });
    }

    // -- pictures and trigger words ------------------------------------------------

    function loadDetails() {
        const names = [];
        state.rows.forEach(function (row) {
            if (names.indexOf(row.name) === -1) names.push(row.name);
        });
        if (state.request) state.request.abort();
        state.nodes.status.hidden = true;
        if (!names.length) return;
        state.rows.forEach(function (row) {
            if (row.el && row.el.row.hasAttribute("data-unloaded")) {
                row.el.row.removeAttribute("data-unloaded");
                row.el.pills.innerHTML = '<span class="mc-lv-skeleton"></span><span class="mc-lv-skeleton mc-lv-wide"></span>';
            }
        });
        const controller = typeof AbortController === "function" ? new AbortController() : null;
        const request = {abort: function () { if (controller) controller.abort(); }, done: false};
        state.request = request;
        const deadline = setTimeout(function () { request.timedOut = true; request.abort(); }, DEADLINE_MS);
        const query = names.map(function (name) { return "name=" + encodeURIComponent(name); }).join("&");
        fetch(ROUTE + "?" + query, {
            method: "GET", credentials: "same-origin", cache: "no-store",
            headers: {"Accept": "application/json"}, signal: controller ? controller.signal : undefined,
        }).then(function (response) {
            if (!response.ok) throw new Error("HTTP " + response.status);
            return response.json();
        }).then(function (body) {
            clearTimeout(deadline);
            if (state.request !== request || !state.open) return;
            state.request = null;
            fill(body && Array.isArray(body.loras) ? body.loras : []);
        }).catch(function (error) {
            clearTimeout(deadline);
            if (state.request !== request || !state.open) return;
            state.request = null;
            unloaded(request.timedOut
                ? "Pictures and trigger words did not arrive in time."
                : "Pictures and trigger words could not be loaded.");
            if (!request.timedOut && error && error.name !== "AbortError") {
                console.warn("Model Chain: the LoRA viewer's details failed", error);
            }
        });
    }

    function unloaded(message) {
        state.nodes.statusText.textContent = message;
        state.nodes.status.hidden = false;
        state.rows.forEach(function (row) {
            if (!row.el) return;
            row.el.row.setAttribute("data-unloaded", "");
            row.el.thumb.textContent = "";
            row.el.thumb.appendChild(make("span", "mc-lv-initials", initials(row.name)));
            row.el.pills.textContent = "";
            row.el.pills.appendChild(make("span", "mc-lv-note", "No details"));
        });
    }

    function fill(loras) {
        const byName = {};
        loras.forEach(function (lora) { if (lora && lora.name) byName[lora.name] = lora; });
        const words = wordsOf(state.boxes);
        state.rows.forEach(function (row) {
            const el = row.el;
            if (!el) return;
            const lora = byName[row.name] || {found: true, picture: "", triggers: [], title: ""};
            el.thumb.textContent = "";
            el.pills.textContent = "";
            el.pills.removeAttribute("data-open");
            if (lora.title && lora.title !== row.name) el.name.title = lora.title + " (" + row.name + ")";
            if (!lora.found) {
                el.row.setAttribute("data-missing", "");
                const warning = make("span", "mc-lv-warning");
                warning.innerHTML = icon("triangle-alert");
                warning.title = "No LoRA with this name in your LoRA folders";
                el.thumb.appendChild(warning);
                el.pills.appendChild(make("span", "mc-lv-note", "Not in your LoRA folders"));
                return;
            }
            drawPicture(el.thumb, lora.picture, row.name);
            const triggers = Array.isArray(lora.triggers) ? lora.triggers : [];
            if (!triggers.length) {
                el.pills.appendChild(make("span", "mc-lv-note", "No trigger words"));
                return;
            }
            triggers.forEach(function (word) {
                el.pills.appendChild(pill(String(word), inPrompt(words, word)));
            });
        });
        state.rows.forEach(fold);
    }

    function drawPicture(holder, picture, name) {
        if (!picture) {
            holder.appendChild(make("span", "mc-lv-initials", initials(name)));
            return;
        }
        const image = document.createElement("img");
        image.alt = "";
        image.decoding = "async";
        image.draggable = false;
        image.addEventListener("error", function () {
            holder.textContent = "";
            holder.appendChild(make("span", "mc-lv-initials", initials(name)));
        });
        image.src = picture;
        holder.appendChild(image);
    }

    function pill(word, already) {
        const button = make("button", "mc-lv-pill");
        button.type = "button";
        if (already) button.setAttribute("data-in", "");
        button.title = (already ? "Already in the prompt. " : "") + "Copy “" + word + "”";
        button.setAttribute("aria-label", "Copy " + word + (already ? ", already in the prompt" : ""));
        face(button, word, already);
        button.addEventListener("click", function () { copy(button, word, already); });
        return button;
    }

    function face(button, word, already) {
        button.innerHTML = (already ? icon("check") : "") + '<span class="mc-lv-pill-text"></span>';
        button.lastChild.textContent = word;
    }

    function copy(button, word, already) {
        function done(ok) {
            button.style.minWidth = button.offsetWidth + "px";
            button.setAttribute("data-copied", "");
            button.innerHTML = icon(ok ? "check" : "copy") + '<span class="mc-lv-pill-text"></span>';
            button.lastChild.textContent = ok ? "Copied" : "Copy failed";
            clearTimeout(button._mcLvTimer);
            button._mcLvTimer = setTimeout(function () {
                button.removeAttribute("data-copied");
                button.style.minWidth = "";
                face(button, word, already);
            }, 1200);
        }
        function fallback() {
            let ok = false;
            try {
                const area = document.createElement("textarea");
                area.value = word;
                area.setAttribute("readonly", "");
                area.style.cssText = "position:fixed;top:0;left:0;width:1px;height:1px;opacity:0";
                document.body.appendChild(area);
                area.select();
                ok = document.execCommand("copy");
                area.remove();
                button.focus({preventScroll: true});
            } catch (error) {
                ok = false;
            }
            done(ok);
        }
        try {
            if (navigator.clipboard && typeof navigator.clipboard.writeText === "function") {
                navigator.clipboard.writeText(word).then(function () { done(true); }, fallback);
                return;
            }
        } catch (error) {
            /* fall back */
        }
        fallback();
    }

    // Pills that do not fit the row's one line fold into "+N", which opens the row.
    function fold(row) {
        const strip = row.el && row.el.pills;
        if (!strip) return;
        const pills = Array.prototype.slice.call(strip.querySelectorAll(".mc-lv-pill:not(.mc-lv-more)"));
        const old = strip.querySelector(".mc-lv-more");
        if (old) old.remove();
        pills.forEach(function (one) { one.hidden = false; });
        if (strip.hasAttribute("data-open") || pills.length < 2) return;
        if (getComputedStyle(strip).flexWrap !== "nowrap") return;
        const edge = strip.getBoundingClientRect().right;
        const rights = pills.map(function (one) { return one.getBoundingClientRect().right; });
        let cut = -1;
        for (let i = 0; i < rights.length; i += 1) {
            if (rights[i] > edge + 0.5) { cut = i; break; }
        }
        if (cut === -1) return;
        while (cut > 1 && rights[cut - 1] > edge - 48) cut -= 1;
        cut = Math.max(1, cut);
        for (let j = cut; j < pills.length; j += 1) pills[j].hidden = true;
        const hidden = pills.length - cut;
        const more = make("button", "mc-lv-pill mc-lv-more", "+" + hidden);
        more.type = "button";
        more.setAttribute("aria-label", "Show " + hidden + " more trigger " + (hidden === 1 ? "word" : "words"));
        more.addEventListener("click", function () {
            strip.setAttribute("data-open", "");
            more.remove();
            pills.forEach(function (one) { one.hidden = false; });
        });
        strip.appendChild(more);
    }

    // -- writing back ---------------------------------------------------------------

    // The tag taken out, and the gap it leaves closed: one space between words,
    // a comma not doubled, nothing left at either end of a line.
    function cut(text, start, end) {
        let before = text.slice(0, start).replace(/[ \t]+$/, "");
        let after = text.slice(end).replace(/^[ \t]+/, "");
        if (/,$/.test(before) && /^,/.test(after)) after = after.slice(1).replace(/^[ \t]+/, "");
        if (!before || /\n$/.test(before)) return before + after.replace(/^,[ \t]*/, "");
        if (!after || /^\n/.test(after)) return before.replace(/,$/, "") + after;
        return before + (/^,/.test(after) ? "" : " ") + after;
    }

    function send(field, literal) {
        if (typeof updateInput === "function") {
            updateInput(field);
        } else {
            field.dispatchEvent(new Event("input", {bubbles: true}));
        }
        // A Literal box keeps its text on blur (scripts/model_chain_krea_creative.py).
        if (literal) field.dispatchEvent(new Event("blur"));
    }

    function apply() {
        const byBox = {};
        changes().forEach(function (row) {
            (byBox[row.box.id] = byBox[row.box.id] || []).push(row);
        });
        const moved = [];
        Object.keys(byBox).forEach(function (id) {
            const box = byBox[id][0].box;
            const field = box.field;
            if (!field || String(field.value || "") !== box.text) {
                moved.push(box.label);
                return;
            }
            let text = box.text;
            byBox[id].sort(function (a, b) { return b.start - a.start; }).forEach(function (row) {
                if (row.removed) {
                    text = cut(text, row.start, row.end);
                } else {
                    text = text.slice(0, row.start) + "<" + row.kind + ":" + row.rawName + ":"
                        + fmt(row.weight) + row.tail + ">" + text.slice(row.end);
                }
            });
            field.value = text;
            send(field, box.literal);
        });
        if (moved.length) {
            toast((moved.length === 1 ? "The " + moved[0] + " box" : "The " + moved.join(" and ") + " boxes")
                + " changed while the viewer was open, so its weights were not written. Open it again.");
        }
    }

    function toast(message) {
        const node = state.nodes.toast;
        node.textContent = message;
        node.hidden = false;
        clearTimeout(state.toastTimer);
        state.toastTimer = setTimeout(function () { node.hidden = true; }, TOAST_MS);
    }

    // -- placement ----------------------------------------------------------------

    // The viewer starts under whatever tab bar is on screen -- the theme's
    // sticky header, the bar focus mode keeps, or Forge's own when the page is
    // at its top -- and beside it when the bar is a rail down one side of the
    // window (the assistant's Transpose View): a box taller than it is wide.
    function chrome() {
        const out = {top: 0, left: 0, right: 0};
        const bar = document.querySelector("#tabs > .tab-nav") || app().querySelector(".tab-nav");
        if (!bar) return out;
        const rect = bar.getBoundingClientRect();
        if (rect.width <= 0 || rect.height <= 0) return out;
        if (rect.height > rect.width) {
            const width = document.documentElement.clientWidth || window.innerWidth;
            if (rect.left + rect.width / 2 < width / 2) out.left = Math.max(0, Math.round(rect.right));
            else out.right = Math.max(0, Math.round(width - rect.left));
            return out;
        }
        if (rect.bottom <= 0 || rect.top >= window.innerHeight / 2) return out;
        out.top = Math.round(rect.bottom);
        return out;
    }

    function measure() {
        const nodes = state.nodes;
        if (!nodes || !state.open) return;
        const bar = chrome();
        const view = window.visualViewport;
        let hidden = 0;
        if (view) hidden = Math.max(0, window.innerHeight - (view.offsetTop + view.height));
        const root = document.documentElement;
        root.style.setProperty("--mc-lv-top", bar.top + "px");
        root.style.setProperty("--mc-lv-left", bar.left + "px");
        root.style.setProperty("--mc-lv-right", bar.right + "px");
        root.style.setProperty("--mc-lv-keyboard", Math.round(hidden) + "px");
        const active = document.activeElement;
        if (hidden && active && nodes.dialog.contains(active) && active.scrollIntoView) {
            active.scrollIntoView({block: "nearest"});
        }
    }

    function refold() {
        if (state.open) state.rows.forEach(fold);
    }

    // -- open and close -------------------------------------------------------------

    function onKey(event) {
        if (!state.open || event.key !== "Escape") return;
        // Before Forge's Escape (it interrupts a generation) and the assistant's
        // (it leaves focus mode): this one is the viewer's.
        event.preventDefault();
        event.stopPropagation();
        if (event.stopImmediatePropagation) event.stopImmediatePropagation();
        close(false);
    }

    function trapTab(event) {
        if (event.key !== "Tab") return;
        const nodes = state.nodes;
        const stops = Array.prototype.slice.call(nodes.dialog.querySelectorAll(
            "button:not([disabled]), input:not([disabled]), [tabindex='0']"))
            .filter(function (node) { return !node.hidden && node.offsetParent !== null; });
        if (!stops.length) return;
        const first = stops[0];
        const last = stops[stops.length - 1];
        if (event.shiftKey && (document.activeElement === first || document.activeElement === nodes.dialog)) {
            event.preventDefault();
            last.focus();
        } else if (!event.shiftKey && document.activeElement === last) {
            event.preventDefault();
            first.focus();
        }
    }

    function onResize() {
        measure();
        refold();
    }

    function listen(on) {
        if (on === state.listening) return;
        state.listening = on;
        const method = on ? "addEventListener" : "removeEventListener";
        window[method]("keydown", onKey, true);
        window[method]("resize", onResize);
        if (window.visualViewport) {
            window.visualViewport[method]("resize", measure);
            window.visualViewport[method]("scroll", measure);
        }
    }

    // A change of tab is Cancel: the viewer belongs to the tab whose prompt it read.
    function watchTab(tab) {
        if (state.observer) state.observer.disconnect();
        state.observer = null;
        const panel = byId(TABS[tab].panel);
        if (!panel || typeof MutationObserver !== "function") return;
        state.observer = new MutationObserver(function () {
            if (state.open && panelHidden(panel)) close(false);
        });
        state.observer.observe(panel, {attributes: true, attributeFilter: ["style", "class", "hidden"]});
    }

    function panelHidden(panel) {
        if (panel.hidden || panel.style.display === "none") return true;
        try {
            return getComputedStyle(panel).display === "none";
        } catch (error) {
            return false;
        }
    }

    function announce(open) {
        try {
            document.dispatchEvent(new CustomEvent(OVERLAY, {
                detail: {name: "lora-viewer", open: open, modal: true},
            }));
        } catch (error) {
            /* nobody listening is fine */
        }
    }

    function open(tab) {
        if (!TABS[tab]) return false;
        if (state.open) close(false);
        const nodes = build();
        state.tab = tab;
        state.boxes = boxesOf(tab);
        state.rows = readRows(state.boxes);
        state.button = byId("mc-lora-viewer-" + tab);
        state.open = true;
        renderRows();
        nodes.scrim.hidden = false;
        nodes.dialog.hidden = false;
        nodes.dialog.setAttribute("data-tab", tab);
        if (state.button) state.button.setAttribute("aria-expanded", "true");
        listen(true);
        watchTab(tab);
        measure();
        announce(true);
        try {
            nodes.dialog.focus({preventScroll: true});
        } catch (error) {
            /* focus is a courtesy */
        }
        loadDetails();
        return true;
    }

    function close(applying) {
        if (!state.open) return false;
        if (applying) apply();
        if (state.request) {
            const request = state.request;
            state.request = null;
            request.abort();
        }
        state.open = false;
        listen(false);
        if (state.observer) state.observer.disconnect();
        state.observer = null;
        state.nodes.scrim.hidden = true;
        state.nodes.dialog.hidden = true;
        state.nodes.list.textContent = "";
        const button = state.button;
        if (button) button.setAttribute("aria-expanded", "false");
        announce(false);
        state.rows = [];
        state.boxes = [];
        if (button && document.body.contains(button)) {
            try {
                button.focus({preventScroll: true});
            } catch (error) {
                /* focus is a courtesy */
            }
        }
        return true;
    }

    // -- the button -------------------------------------------------------------------

    // One button per tab, put after Forge's Apply Styles with its neighbours'
    // classes, so a theme draws it as it draws them. Checked on every update and
    // added only when missing: an after-update callback that changed the page
    // every time would raise the next update itself.
    function ensureButton(tab) {
        const conf = TABS[tab];
        const tools = byId(conf.tools);
        if (!tools) return false;
        if (byId("mc-lora-viewer-" + tab)) return true;
        const anchor = byId(conf.after);
        const template = anchor || tools.querySelector("button");
        const button = document.createElement("button");
        button.type = "button";
        button.id = "mc-lora-viewer-" + tab;
        button.className = ((template && template.className) || "lg secondary gradio-button tool")
            + " mc-lv-open";
        button.title = "LoRAs in this prompt";
        button.setAttribute("aria-label", "LoRAs in this prompt");
        button.setAttribute("aria-haspopup", "dialog");
        button.setAttribute("aria-expanded", "false");
        button.innerHTML = icon("sliders-vertical");
        button.addEventListener("click", function () { open(tab); });
        if (anchor && anchor.parentNode) {
            anchor.parentNode.insertBefore(button, anchor.nextSibling);
        } else {
            tools.appendChild(button);
        }
        return true;
    }

    function wire() {
        Object.keys(TABS).forEach(ensureButton);
    }

    window.mcLoraViewer = {
        version: VERSION,
        ROUTE: ROUTE,
        isOpen: function () { return state.open; },
        open: open,
        close: close,
        wire: wire,
    };

    if (typeof onUiLoaded === "function") onUiLoaded(wire);
    if (typeof onAfterUiUpdate === "function") onAfterUiUpdate(wire);
})();
