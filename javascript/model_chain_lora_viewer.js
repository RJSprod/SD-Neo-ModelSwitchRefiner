// Model Chain -- the LoRA viewer.
//
// A button named Lora Viewer, as wide as the prompt boxes, directly under the
// Negative Prompt on Txt2Img and Img2Img -- under the Prompt while CFG is 1 and
// the Negative Prompt is put away, and above the Literal row: "a button that is
// named "Lora Viewer" ... below the negative prompt box ... as wide as the prompt
// input fields and just below negative (or positive when CFG = 1), and above the
// literals" (2026-11-06; it was a tool button after Apply Styles before that, out
// of a thumb's reach on a phone). Pressed, it covers the page under the tab bar with every
// LoRA in the prompt -- and on Txt2Img in the two Literal boxes -- one thin row
// each: its picture, its name, the box it is in, its trigger words (a tap copies
// one), its weight in a box that takes any number, and a slider from -5 to 5 in
// steps of 0.05, with - and + at its ends for the last step a finger cannot make.
// Cancel changes nothing. Apply writes each changed weight into its tag where the
// tag is, takes out the tags marked for removal, and leaves every other
// character of the prompt as it was.
//
// Adding one
// ----------
// A box under the title adds a LoRA, one at a time: "the ability to add lora
// from this view, but only for the model type I have in the preset. This should
// work the same way as the extra networks panel. What ever lora folder I have
// assigned should be the source for adding ... see a suggested auto complete
// when I am typing in a lora name". The names come from the extension's second
// route (mc_lora_viewer.py, `listing`), kept to the folders assigned to the
// active UI Preset as the LoRA browser keeps its cards, listed once a viewer
// when the box is first used, with a deadline. What is typed is matched as
// Forge's own browser search matches -- every word somewhere in the name --
// the names that start with it first. A choice is a new row, at the card's
// own weight (its Preferred weight, else Forge's default multiplier), written
// at the end of the Prompt box on Apply after Forge's separator, as a press on
// the card writes it; a LoRA already in the prompt is shown, not added twice.
// Each suggestion shows the LoRA's picture: "I need to see the image next to
// the lora name in that list". The names come without pictures, so a folder
// of thousands lists at once; the pictures of the names shown are asked of
// the details route a moment after the typing stops, one request with a
// deadline at a time, and kept for the viewer's life, so a name seen once is
// drawn at once after that and a LoRA added from the list needs no request
// of its own.
//
// Its description
// ---------------
// A row's picture is a button that opens the card's description -- Forge's
// own editor for it, the one the LoRA browser's edit button opens (its
// `extraNetworksEditUserMetadata`), with LoRA Details' library rows in it:
// "the ability to get to the extra network content description by our new
// lora menu". Forge draws it on its global popup, under the viewer's layer, so
// the viewer steps aside while it is open, keeping every change made in it,
// and comes back when Forge puts the popup away (Save, Cancel, its backdrop,
// Escape), the row's details asked again since its activation text or
// Preferred weight may have changed.
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
// * Hand a tap outside it to the page under it. A tap outside closes it, as
//   Escape and a change of tab do -- Cancel, nothing written: "Tapping outside
//   of that lora pop up should close it" (2026-11-06; until then a tap outside
//   did nothing, so that a stray one could not throw changes away, and when
//   one does now the page says so). Decided on the tap's click, its last
//   event, so nothing under the scrim gets what follows the press once the
//   scrim has gone; a press that moved is a swipe, and one that began inside
//   the viewer (a knob dragged past its edge) is the viewer's. The tab bar
//   still answers: a tap on a tab closes the viewer and changes the tab.
// * Make the add box wait for a tap. It has the focus from the moment the
//   viewer opens, inside the press that opened it, so a phone's keyboard comes
//   up with it: "the text field should be auto selected, cursor in the field,
//   ready for me to type a character" (2026-11-06).
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
    const NAMES_ROUTE = "/model-chain/lora-viewer/names";
    const SUGGESTIONS = 8;   // the most names the add box shows at once
    const PICTURES_WAIT_MS = 150;   // after the last key, before the shown names' pictures are asked
    const EDITOR = "_lora_edit_user_metadata";   // Forge's description editor: "<tab>" + EDITOR
    const DEADLINE_MS = 8000;
    const MIN = -5;
    const MAX = 5;
    const STEP = 0.05;
    const PAD = 11;          // the track's inset: the knob's radius, so its centre reaches both ends
    const TAP_SLOP = 6;      // pixels a finger may move and still be a tap on the track
    const OUTSIDE_SLOP = 16; // pixels a press outside may move and still be a tap that closes
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
            negative: "txt2img_neg_prompt", panel: "tab_txt2img",
            boxes: [
                {id: "txt2img_prompt", label: "Prompt"},
                {id: LITERAL_POSITIVE, label: "Literal +", literal: true},
                {id: LITERAL_NEGATIVE, label: "Literal −", literal: true},
            ],
        },
        img2img: {
            negative: "img2img_neg_prompt", panel: "tab_img2img",
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
        "search": '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>',
        "info": '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>',
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
        // The add box: the names it suggests from (listed once a viewer), the
        // request listing them, the suggestions shown and the one chosen by
        // the arrow keys, and the requests for the details of added rows.
        names: null, namesRequest: null, shown: [], active: -1, extra: [],
        // What the details route answered for a name, kept for the viewer's
        // life: the suggestions' pictures, and an added row's details.
        known: {}, picturing: null, picturesTimer: 0,
        // Stepped aside for Forge's description editor, and its watch.
        aside: false, popupWatch: null,
        // A press made outside the viewer, until its click says it was a tap.
        outside: null,
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
            + '<div class="mc-lv-add">'
            + '<label class="mc-lv-add-field">' + icon("search")
            + '<input class="mc-lv-add-input" type="text" role="combobox" aria-autocomplete="list" '
            + 'aria-expanded="false" aria-controls="mc-lora-viewer-suggestions" '
            + 'aria-label="Add a LoRA from this preset\'s folders" placeholder="Add a LoRA\u2026" '
            + 'autocomplete="off" autocapitalize="off" spellcheck="false" enterkeyhint="done"></label>'
            + '<div class="mc-lv-suggest" id="mc-lora-viewer-suggestions" role="listbox" '
            + 'aria-label="LoRAs" hidden></div>'
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
            add: dialog.querySelector(".mc-lv-add-input"),
            suggest: dialog.querySelector(".mc-lv-suggest"),
            cancel: dialog.querySelector(".mc-lv-button-cancel"),
            apply: dialog.querySelector(".mc-lv-button-apply"),
        };
        nodes.cancel.addEventListener("click", function () { close(false); });
        nodes.apply.addEventListener("click", function () { close(true); });
        nodes.resetAll.addEventListener("click", resetAll);
        nodes.retry.addEventListener("click", function () { loadDetails(); });
        bindAdd(nodes);
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
            empty.appendChild(make("span", "", "Add one with the box above."));
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
            '<button type="button" class="mc-lv-thumb" disabled><span class="mc-lv-skeleton"></span></button>'
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
        row.el.box.textContent = row.added ? "New in " + row.box.label : row.box.label;
        if (row.added) node.setAttribute("data-added", "");
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

    // An added row is a change until it is taken back; any other row once its
    // weight moved or it is marked for removal.
    function changes() {
        return state.rows.filter(function (row) {
            return row.added ? !row.removed : (row.removed || !same(row.weight, row.orig));
        });
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
        row.touched = true;
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
        if (row.added) {
            button.setAttribute("aria-label", row.removed ? "Add " + row.name + " after all"
                : "Do not add " + row.name);
            button.title = row.removed ? "Add it after all" : "Do not add it";
            return;
        }
        button.setAttribute("aria-label", row.removed
            ? "Keep " + row.name
            : "Remove " + row.name + " from " + row.box.label);
        button.title = row.removed ? "Keep it" : "Remove from " + row.box.label + " on Apply";
    }

    // Every change undone: the weights back, nothing marked, nothing added.
    function resetAll() {
        if (state.rows.some(function (row) { return row.added; })) {
            state.rows = state.rows.filter(function (row) { return !row.added; });
            state.rows.forEach(function (row) {
                row.weight = row.orig;
                row.removed = false;
            });
            renderRows();
            fill(state.filled || [], state.rows);
            return;
        }
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

        el.thumb.addEventListener("click", function () { openDescription(row); });

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

    // The pictures, trigger words and preferred weights of every row -- or,
    // for rows added in the viewer, of those (`only`), on a request of their
    // own that the next full load does not cut short.
    function loadDetails(only) {
        const rows = only || state.rows;
        const names = [];
        rows.forEach(function (row) {
            if (names.indexOf(row.name) === -1) names.push(row.name);
        });
        if (!only) {
            if (state.request) state.request.abort();
            state.nodes.status.hidden = true;
        }
        if (!names.length) return;
        rows.forEach(function (row) {
            if (row.el && row.el.row.hasAttribute("data-unloaded")) {
                row.el.row.removeAttribute("data-unloaded");
                row.el.pills.innerHTML = '<span class="mc-lv-skeleton"></span><span class="mc-lv-skeleton mc-lv-wide"></span>';
            }
        });
        const controller = typeof AbortController === "function" ? new AbortController() : null;
        const request = {abort: function () { if (controller) controller.abort(); }, done: false};
        if (only) state.extra.push(request);
        else state.request = request;
        const current = function () {
            return state.open && (only ? state.extra.indexOf(request) !== -1 : state.request === request);
        };
        const settle = function () {
            if (only) state.extra = state.extra.filter(function (one) { return one !== request; });
            else state.request = null;
        };
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
            if (!current()) return;
            settle();
            const loras = body && Array.isArray(body.loras) ? body.loras : [];
            state.filled = (state.filled || []).concat(loras);
            fill(loras, rows);
        }).catch(function (error) {
            clearTimeout(deadline);
            if (!current()) return;
            settle();
            unloaded(request.timedOut
                ? "Pictures and trigger words did not arrive in time."
                : "Pictures and trigger words could not be loaded.", rows);
            if (!request.timedOut && error && error.name !== "AbortError") {
                console.warn("Model Chain: the LoRA viewer's details failed", error);
            }
        });
    }

    function unloaded(message, rows) {
        state.nodes.statusText.textContent = message;
        state.nodes.status.hidden = false;
        (rows || state.rows).forEach(function (row) {
            if (!row.el) return;
            row.el.row.setAttribute("data-unloaded", "");
            row.el.thumb.textContent = "";
            row.el.thumb.appendChild(make("span", "mc-lv-initials", initials(row.name)));
            row.el.pills.textContent = "";
            row.el.pills.appendChild(make("span", "mc-lv-note", "No details"));
        });
    }

    function fill(loras, rows) {
        const byName = {};
        loras.forEach(function (lora) { if (lora && lora.name) byName[lora.name] = lora; });
        const words = wordsOf(state.boxes);
        (rows || state.rows).forEach(function (row) {
            const el = row.el;
            if (!el) return;
            const lora = byName[row.name] || {found: true, picture: "", triggers: [], title: ""};
            // An added row takes the card's Preferred weight, as a press on the
            // card would write it, unless it has been moved since.
            if (row.added && !row.touched && typeof lora.weight === "number" && Number.isFinite(lora.weight)
                && !same(lora.weight, row.weight)) {
                row.weight = row.orig = Number(lora.weight.toFixed(4));
                paint(row);
                tally();
            }
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
            row.card = String(lora.card || "");
            describable(row);
            const triggers = Array.isArray(lora.triggers) ? lora.triggers : [];
            if (!triggers.length) {
                el.pills.appendChild(make("span", "mc-lv-note", "No trigger words"));
                return;
            }
            triggers.forEach(function (word) {
                el.pills.appendChild(pill(String(word), inPrompt(words, word)));
            });
        });
        (rows || state.rows).forEach(fold);
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
            if (image.parentNode) image.parentNode.replaceChild(make("span", "mc-lv-initials", initials(name)), image);
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
            byBox[id].filter(function (row) { return !row.added; })
                .sort(function (a, b) { return b.start - a.start; }).forEach(function (row) {
                    if (row.removed) {
                        text = cut(text, row.start, row.end);
                    } else {
                        text = text.slice(0, row.start) + "<" + row.kind + ":" + row.rawName + ":"
                            + fmt(row.weight) + row.tail + ">" + text.slice(row.end);
                    }
                });
            // Added ones at the end, in the order they were added, after
            // Forge's separator, as a press on a card adds one.
            byBox[id].filter(function (row) { return row.added; }).forEach(function (row) {
                text = appended(text, "<" + row.kind + ":" + row.rawName + ":" + fmt(row.weight) + ">");
            });
            field.value = text;
            send(field, box.literal);
        });
        if (moved.length) {
            toast((moved.length === 1 ? "The " + moved[0] + " box" : "The " + moved.join(" and ") + " boxes")
                + " changed while the viewer was open, so its weights were not written. Open it again.");
        }
    }

    // Forge's separator between the prompt and a tag a card adds
    // (`extra_networks_add_text_separator`, a space unless changed); none at
    // the start of an empty box or after a space or a line already there.
    function appended(text, tag) {
        if (!text) return tag;
        let separator = " ";
        try {
            if (window.opts && typeof window.opts.extra_networks_add_text_separator === "string") {
                separator = window.opts.extra_networks_add_text_separator;
            }
        } catch (error) {
            separator = " ";
        }
        return /\s$/.test(text) ? text + tag : text + separator + tag;
    }

    function toast(message) {
        const node = state.nodes.toast;
        node.textContent = message;
        node.hidden = false;
        clearTimeout(state.toastTimer);
        state.toastTimer = setTimeout(function () { node.hidden = true; }, TOAST_MS);
    }

    // -- adding one ----------------------------------------------------------------

    function bindAdd(nodes) {
        const input = nodes.add;
        input.addEventListener("focus", function () { listNames(); });
        input.addEventListener("input", function () {
            state.active = -1;
            listNames();
            showSuggestions();
        });
        input.addEventListener("keydown", function (event) {
            const count = state.shown.length;
            if (event.key === "ArrowDown" || event.key === "ArrowUp") {
                if (nodes.suggest.hidden) showSuggestions();
                if (!state.shown.length) return;
                event.preventDefault();
                const step = event.key === "ArrowDown" ? 1 : -1;
                state.active = state.active < 0
                    ? (step > 0 ? 0 : state.shown.length - 1)
                    : (state.active + step + state.shown.length) % state.shown.length;
                markActive();
            } else if (event.key === "Enter") {
                event.preventDefault();
                if (!count) return;
                addLora(state.shown[state.active >= 0 ? state.active : 0]);
            }
        });
        input.addEventListener("blur", function () {
            // After a press on a suggestion has been heard (it keeps the focus
            // here while it is pressed; this is the box being left).
            setTimeout(function () {
                if (document.activeElement !== input) hideSuggestions();
            }, 0);
        });
        // A press on a suggestion keeps the keyboard where it is, and adds it.
        nodes.suggest.addEventListener("pointerdown", function (event) { event.preventDefault(); });
        nodes.suggest.addEventListener("mousedown", function (event) { event.preventDefault(); });
        nodes.suggest.addEventListener("click", function (event) {
            const option = event.target && event.target.closest ? event.target.closest(".mc-lv-option") : null;
            if (!option) return;
            const lora = state.shown[Number(option.getAttribute("data-index"))];
            if (lora) addLora(lora);
        });
    }

    // The names, once a viewer: listed when the box is first used, with a
    // deadline, and shown as soon as they arrive.
    function listNames() {
        if (state.names || state.namesRequest) return;
        const controller = typeof AbortController === "function" ? new AbortController() : null;
        const request = {abort: function () { if (controller) controller.abort(); }};
        state.namesRequest = request;
        const deadline = setTimeout(function () { request.timedOut = true; request.abort(); }, DEADLINE_MS);
        fetch(NAMES_ROUTE, {
            method: "GET", credentials: "same-origin", cache: "no-store",
            headers: {"Accept": "application/json"}, signal: controller ? controller.signal : undefined,
        }).then(function (response) {
            if (!response.ok) throw new Error("HTTP " + response.status);
            return response.json();
        }).then(function (body) {
            clearTimeout(deadline);
            if (state.namesRequest !== request || !state.open) return;
            state.namesRequest = null;
            state.names = {
                loras: (body && Array.isArray(body.loras) ? body.loras : []).filter(function (one) {
                    return one && typeof one.name === "string" && one.name;
                }),
                preset: String((body && body.preset) || ""),
                filtered: !!(body && body.filtered),
            };
            showSuggestions();
        }).catch(function (error) {
            clearTimeout(deadline);
            if (state.namesRequest !== request || !state.open) return;
            state.namesRequest = null;
            state.names = {failed: request.timedOut
                ? "Your LoRAs did not arrive in time. Type again to try again."
                : "Your LoRAs could not be listed. Type again to try again.", loras: []};
            if (!request.timedOut && error && error.name !== "AbortError") {
                console.warn("Model Chain: the LoRA viewer's list of LoRAs failed", error);
            }
            showSuggestions();
        });
    }

    // Forge's browser search: every word typed somewhere in the name (or the
    // tag's name). Those whose name starts with what was typed first, then
    // those with a word that does, then the rest, each alphabetically.
    function matches(loras, query) {
        const typed = String(query || "").trim().toLowerCase();
        const words = typed.split(/\s+/).filter(Boolean);
        if (!words.length) return [];
        const found = [];
        loras.forEach(function (lora) {
            const name = lora.name.toLowerCase();
            const hay = name + " " + String(lora.tag || "").toLowerCase();
            for (let i = 0; i < words.length; i += 1) {
                if (hay.indexOf(words[i]) === -1) return;
            }
            const rank = name.indexOf(typed) === 0 ? 0
                : (name.split(/[_\-\s.()\[\]]+/).some(function (part) { return part.indexOf(words[0]) === 0; }) ? 1 : 2);
            found.push({lora: lora, rank: rank});
        });
        found.sort(function (a, b) {
            return a.rank - b.rank || (a.lora.name.toLowerCase() < b.lora.name.toLowerCase() ? -1
                : (a.lora.name.toLowerCase() > b.lora.name.toLowerCase() ? 1 : 0));
        });
        return found.slice(0, SUGGESTIONS).map(function (one) { return one.lora; });
    }

    // The prompt box's row of a LoRA, if it has one that stays.
    function rowOf(lora) {
        const keys = [String(lora.tag || "").toLowerCase(), String(lora.name || "").toLowerCase()];
        const prompt = promptBox();
        for (let i = 0; i < state.rows.length; i += 1) {
            const row = state.rows[i];
            if (row.removed || (prompt && row.box !== prompt)) continue;
            if (keys.indexOf(row.name.toLowerCase()) !== -1) return row;
        }
        return null;
    }

    function promptBox() {
        const id = TABS[state.tab] ? TABS[state.tab].boxes[0].id : "";
        for (let i = 0; i < state.boxes.length; i += 1) {
            if (state.boxes[i].id === id) return state.boxes[i];
        }
        return null;
    }

    function note(text) {
        state.nodes.suggest.textContent = "";
        state.nodes.suggest.appendChild(make("div", "mc-lv-suggest-note", text));
        state.nodes.suggest.hidden = false;
        state.nodes.add.setAttribute("aria-expanded", "false");
        state.nodes.add.removeAttribute("aria-activedescendant");
    }

    function showSuggestions() {
        const nodes = state.nodes;
        if (!nodes || !state.open) return;
        const typed = nodes.add.value.trim();
        state.shown = [];
        if (!typed || document.activeElement !== nodes.add) {
            hideSuggestions();
            return;
        }
        if (!state.names) {
            note("Listing your LoRAs\u2026");
            return;
        }
        if (state.names.failed) {
            const failed = state.names.failed;
            state.names = null;   // typed again, it lists them again
            note(failed);
            return;
        }
        state.shown = matches(state.names.loras, typed);
        if (!state.shown.length) {
            note(state.names.filtered && state.names.preset
                ? "No LoRA matches \u201c" + typed + "\u201d in the " + state.names.preset + " preset's folders."
                : "No LoRA matches \u201c" + typed + "\u201d.");
            return;
        }
        nodes.suggest.textContent = "";
        state.shown.forEach(function (lora, index) {
            const option = make("div", "mc-lv-option");
            option.id = "mc-lv-option-" + index;
            option.setAttribute("role", "option");
            option.setAttribute("data-index", String(index));
            const here = rowOf(lora);
            if (here) option.setAttribute("data-in", "");
            const thumb = make("span", "mc-lv-option-thumb");
            thumb.setAttribute("data-for", lora.tag || lora.name);
            paintThumb(thumb, lora.name);
            option.appendChild(thumb);
            const name = make("span", "mc-lv-option-name", lora.name);
            option.appendChild(name);
            if (lora.tag && lora.tag !== lora.name) option.appendChild(make("span", "mc-lv-option-tag", lora.tag));
            if (here) {
                const mark = make("span", "mc-lv-option-in");
                mark.innerHTML = icon("check");
                mark.appendChild(make("span", "", "In the prompt"));
                option.appendChild(mark);
            }
            option.title = here ? lora.name + " is in the prompt already" : "Add " + lora.name;
            nodes.suggest.appendChild(option);
        });
        nodes.suggest.hidden = false;
        nodes.add.setAttribute("aria-expanded", "true");
        if (state.active >= state.shown.length) state.active = -1;
        markActive();
        askPictures();
    }

    function hideSuggestions() {
        state.shown = [];
        state.active = -1;
        const nodes = state.nodes;
        if (!nodes) return;
        nodes.suggest.hidden = true;
        nodes.suggest.textContent = "";
        nodes.add.setAttribute("aria-expanded", "false");
        nodes.add.removeAttribute("aria-activedescendant");
    }

    function markActive() {
        const nodes = state.nodes;
        const options = nodes.suggest.querySelectorAll(".mc-lv-option");
        Array.prototype.forEach.call(options, function (option, index) {
            option.setAttribute("aria-selected", index === state.active ? "true" : "false");
        });
        const active = state.active >= 0 ? options[state.active] : null;
        if (active) {
            nodes.add.setAttribute("aria-activedescendant", active.id);
            if (active.scrollIntoView) active.scrollIntoView({block: "nearest"});
        } else {
            nodes.add.removeAttribute("aria-activedescendant");
        }
    }

    // Forge's default weight for a tag a card adds without a preferred one.
    function defaultWeight() {
        try {
            const value = Number(window.opts && window.opts.extra_networks_default_multiplier);
            if (Number.isFinite(value) && value !== 0) return value;
        } catch (error) {
            /* Forge's options are not on this page */
        }
        return 1;
    }

    // One LoRA chosen: a new row for the Prompt box, or the row it has already
    // shown. The box is emptied for the next one.
    function addLora(lora) {
        const nodes = state.nodes;
        if (!lora || !nodes) return;
        nodes.add.value = "";
        hideSuggestions();
        const here = rowOf(lora);
        if (here) {
            flash(here);
            return;
        }
        const box = promptBox();
        if (!box) {
            toast("There is no Prompt box on this tab to add it to.");
            return;
        }
        const tag = String(lora.tag || lora.name);
        const weight = defaultWeight();
        const row = {
            box: box, start: -1, end: -1, kind: "lora", rawName: tag, name: tag,
            weight: weight, orig: weight, tail: "", fixed: false,
            removed: false, editing: false, el: null, added: true, touched: false,
        };
        const first = !state.rows.length;
        state.rows.push(row);
        if (first) {
            renderRows();
        } else {
            nodes.list.appendChild(rowNode(row, state.rows.length - 1));
            const count = state.rows.length;
            nodes.count.textContent = count + " in this prompt";
            tally();
        }
        if (row.el && row.el.row.scrollIntoView) row.el.row.scrollIntoView({block: "nearest"});
        // Its details came with its picture in the list, most likely.
        const known = state.known[tag];
        if (known && !known.failed) fill([known], [row]);
        else loadDetails([row]);
    }

    // -- the suggestions' pictures --------------------------------------------------

    /** A suggestion's picture: the LoRA's when it is known, its initials when
     *  it has none, a shimmer while it is asked for. */
    function paintThumb(thumb, name) {
        const known = state.known[thumb.getAttribute("data-for")];
        const want = !known ? "waiting" : (known.picture ? "picture:" + known.picture : "initials");
        if (thumb.getAttribute("data-drawn") === want) return;
        thumb.setAttribute("data-drawn", want);
        thumb.textContent = "";
        if (!known) {
            thumb.appendChild(make("span", "mc-lv-skeleton"));
            return;
        }
        drawPicture(thumb, known.picture, name);
    }

    function repaintThumbs() {
        const nodes = state.nodes;
        if (!nodes) return;
        Array.prototype.forEach.call(nodes.suggest.querySelectorAll(".mc-lv-option-thumb"), function (thumb) {
            const lora = state.shown[Number(thumb.parentNode.getAttribute("data-index"))];
            paintThumb(thumb, lora ? lora.name : thumb.getAttribute("data-for"));
        });
    }

    /** The shown names' pictures, once the typing has paused. */
    function askPictures() {
        clearTimeout(state.picturesTimer);
        state.picturesTimer = setTimeout(fetchPictures, PICTURES_WAIT_MS);
    }

    /** One request at a time, with a deadline, for the shown names not known
     *  yet; when it answers, the names shown by then are asked for in turn. */
    function fetchPictures() {
        if (!state.open || state.picturing) return;
        const wanted = [];
        state.shown.forEach(function (lora) {
            const key = lora.tag || lora.name;
            if (!Object.prototype.hasOwnProperty.call(state.known, key) && wanted.indexOf(key) === -1) wanted.push(key);
        });
        if (!wanted.length) return;
        const controller = typeof AbortController === "function" ? new AbortController() : null;
        const request = {abort: function () { if (controller) controller.abort(); }};
        state.picturing = request;
        const deadline = setTimeout(function () { request.timedOut = true; request.abort(); }, DEADLINE_MS);
        const query = wanted.map(function (name) { return "name=" + encodeURIComponent(name); }).join("&");
        const done = function (loras, failed) {
            clearTimeout(deadline);
            if (state.picturing !== request || !state.open) return;
            state.picturing = null;
            loras.forEach(function (lora) { if (lora && lora.name) state.known[lora.name] = lora; });
            // A name the answer left out, or a request that failed, is drawn
            // as its initials and not asked again in this viewer.
            wanted.forEach(function (key) {
                if (!Object.prototype.hasOwnProperty.call(state.known, key)) {
                    state.known[key] = {name: key, found: true, picture: "", triggers: [], failed: !!failed};
                }
            });
            repaintThumbs();
            fetchPictures();
        };
        fetch(ROUTE + "?" + query, {
            method: "GET", credentials: "same-origin", cache: "no-store",
            headers: {"Accept": "application/json"}, signal: controller ? controller.signal : undefined,
        }).then(function (response) {
            if (!response.ok) throw new Error("HTTP " + response.status);
            return response.json();
        }).then(function (body) {
            done(body && Array.isArray(body.loras) ? body.loras : [], false);
        }).catch(function (error) {
            if (!request.timedOut && error && error.name !== "AbortError") {
                console.warn("Model Chain: the LoRA viewer's pictures failed", error);
            }
            done([], true);
        });
    }

    // -- a LoRA's description -------------------------------------------------------

    /** Forge's description editor for a tab's LoRA cards, when the page has it. */
    function editorOn(tab) {
        return typeof window.extraNetworksEditUserMetadata === "function" && !!byId(tab + EDITOR);
    }

    /** A row's picture opens its card's description, once the card is known
     *  and Forge's editor is on the page. */
    function describable(row) {
        const thumb = row.el && row.el.thumb;
        if (!thumb) return;
        const can = !!row.card && editorOn(state.tab);
        thumb.disabled = !can;
        thumb.toggleAttribute("data-describable", can);
        thumb.setAttribute("aria-label", can ? "Description of " + row.name : row.name);
        thumb.title = can ? "Description and details" : "";
        let badge = thumb.querySelector(".mc-lv-thumb-badge");
        if (can && !badge) {
            badge = make("span", "mc-lv-thumb-badge");
            badge.innerHTML = icon("info");
            thumb.appendChild(badge);
        } else if (!can && badge) {
            badge.remove();
        }
    }

    /** What a press on a card's edit button hands Forge: Forge reads the
     *  card's name off the pressed element's grandparent (`data-name`). */
    function cardEvent(card) {
        const named = document.createElement("div");
        named.setAttribute("data-name", card);
        const middle = document.createElement("div");
        const target = document.createElement("span");
        named.appendChild(middle);
        middle.appendChild(target);
        return {target: target, currentTarget: target, stopPropagation: function () {}, preventDefault: function () {}};
    }

    function forgePopup() {
        const found = document.querySelector(".global-popup");
        return found && found.isConnected && found.style.display !== "none" ? found : null;
    }

    function stepAside(on) {
        state.aside = !!on;
        state.nodes.scrim.hidden = !!on;
        state.nodes.dialog.hidden = !!on;
        if (on) hideSuggestions();
    }

    function openDescription(row) {
        if (!state.open || state.aside || !row.card) return;
        if (!editorOn(state.tab)) {
            toast("Forge's description editor for LoRAs is not on this page.");
            return;
        }
        stepAside(true);
        try {
            window.extraNetworksEditUserMetadata(cardEvent(row.card), state.tab, "lora", row.card);
        } catch (error) {
            console.warn("Model Chain: the LoRA viewer could not open a description", error);
        }
        const shown = forgePopup();
        if (!shown) {
            stepAside(false);
            toast("The description could not be opened.");
            return;
        }
        if (state.popupWatch) state.popupWatch.disconnect();
        const watch = new MutationObserver(function () {
            if (forgePopup() === shown) return;
            // A viewer closed meanwhile (a change of tab) disconnected this.
            watch.disconnect();
            if (state.popupWatch === watch) state.popupWatch = null;
            backFromDescription(row);
        });
        state.popupWatch = watch;
        watch.observe(shown, {attributes: true, attributeFilter: ["style"]});
    }

    /** Forge put its editor away: the viewer as it was, and the details of the
     *  LoRA described asked again. */
    function backFromDescription(row) {
        stepAside(false);
        measure();
        const same = state.rows.filter(function (one) { return one.card === row.card || one.name === row.name; });
        same.forEach(function (one) { delete state.known[one.name]; });
        loadDetails(same);
        try {
            if (row.el && document.body.contains(row.el.thumb)) row.el.thumb.focus({preventScroll: true});
        } catch (error) {
            /* focus is a courtesy */
        }
    }


    function flash(row) {
        if (!row.el) return;
        if (row.el.row.scrollIntoView) row.el.row.scrollIntoView({block: "nearest"});
        row.el.row.setAttribute("data-flash", "");
        clearTimeout(row.flashTimer);
        row.flashTimer = setTimeout(function () {
            if (row.el) row.el.row.removeAttribute("data-flash");
        }, 1200);
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
        // Stepped aside, Escape is Forge's: it puts its description editor away.
        if (!state.open || state.aside || event.key !== "Escape") return;
        // Before Forge's Escape (it interrupts a generation) and the assistant's
        // (it leaves focus mode): this one is the viewer's.
        event.preventDefault();
        event.stopPropagation();
        if (event.stopImmediatePropagation) event.stopImmediatePropagation();
        // The add box's suggestions go first, and then what was typed.
        const add = state.nodes && state.nodes.add;
        if (add && document.activeElement === add && (!state.nodes.suggest.hidden || add.value)) {
            if (!state.nodes.suggest.hidden) hideSuggestions();
            else add.value = "";
            return;
        }
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

    // -- a tap outside ----------------------------------------------------------

    function ours(target) {
        const nodes = state.nodes;
        return !!(nodes && target && target.nodeType === 1
                  && (nodes.dialog.contains(target) || nodes.toast.contains(target)));
    }

    // A primary press, made by a person, outside the viewer: remembered until
    // its click. Stepped aside, the page is Forge's description editor's.
    function onOutsideDown(event) {
        state.outside = null;
        if (!state.open || state.aside || !event.isTrusted || event.isPrimary === false
            || event.button !== 0 || ours(event.target)) return;
        state.outside = {x: event.clientX, y: event.clientY};
    }

    // The tap's last event: a press the browser took for a scroll or a drag
    // sends none. On the scrim it is taken (there is nothing under it to give
    // it to); on the tab bar it goes on, and the tab changes too.
    function onOutsideClick(event) {
        const press = state.outside;
        state.outside = null;
        if (!press || !state.open || !event.isTrusted || ours(event.target)) return;
        if (Math.hypot(event.clientX - press.x, event.clientY - press.y) > OUTSIDE_SLOP) return;
        if (event.target === state.nodes.scrim) {
            event.preventDefault();
            event.stopPropagation();
        }
        const dropped = changes().length;
        close(false);
        if (dropped) {
            toast("Closed without applying: " + (dropped === 1 ? "the change was" : dropped + " changes were")
                  + " not written.");
        }
    }

    function listen(on) {
        if (on === state.listening) return;
        state.listening = on;
        const method = on ? "addEventListener" : "removeEventListener";
        window[method]("keydown", onKey, true);
        window[method]("pointerdown", onOutsideDown, true);
        window[method]("click", onOutsideClick, true);
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
        state.filled = [];
        state.names = null;
        state.known = {};
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
            // The add box, ready to type into: inside the press that opened the
            // viewer, so a phone's keyboard comes up with it. Its focus lists
            // the names (listNames), so the first letter typed has them.
            const field = nodes.add && !nodes.add.disabled ? nodes.add : nodes.dialog;
            field.focus({preventScroll: true});
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
        state.extra.splice(0).forEach(function (request) { request.abort(); });
        if (state.namesRequest) {
            const listing = state.namesRequest;
            state.namesRequest = null;
            listing.abort();
        }
        state.names = null;
        state.filled = [];
        state.nodes.add.value = "";
        hideSuggestions();
        clearTimeout(state.picturesTimer);
        if (state.picturing) {
            const pictures = state.picturing;
            state.picturing = null;
            pictures.abort();
        }
        state.known = {};
        if (state.popupWatch) state.popupWatch.disconnect();
        state.popupWatch = null;
        state.aside = false;
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

    // One button per tab, in a holder of its own placed directly after the
    // Negative Prompt's block, inside Forge's own negative prompt row: a Gradio
    // row wraps, and the holder takes the whole line (the stylesheet's
    // .mc-lv-launch), so the button is as wide as the boxes above it. While CFG
    // is 1 the Literal script puts the Negative Prompt away
    // (model_chain_literals.js), and the button is then the line under the
    // Prompt. The Literal row is placed after this holder by that script, so
    // the two never trade places.
    //
    // Checked on every update and moved or added only when it is not already
    // there: an after-update callback that changed the page every time would
    // raise the next update itself. A page without a Negative Prompt (a theme
    // that took it out) gets nothing -- the viewer is still mcLoraViewer.open().
    function launcher(tab) {
        const holder = document.createElement("div");
        holder.id = "mc-lora-viewer-launch-" + tab;
        holder.className = "mc-lv-launch";
        // The mark the Literal script places its row after
        // (model_chain_literals.js, place): it finds this by the mark alone.
        holder.setAttribute("data-mc-under-negative", "");
        const button = document.createElement("button");
        button.type = "button";
        button.id = "mc-lora-viewer-" + tab;
        button.className = "lg secondary gradio-button mc-lv-open";
        button.title = "LoRAs in this prompt";
        button.setAttribute("aria-label", "Lora Viewer: the LoRAs in this prompt");
        button.setAttribute("aria-haspopup", "dialog");
        button.setAttribute("aria-expanded", "false");
        button.innerHTML = icon("sliders-vertical") + '<span class="mc-lv-open-label">Lora Viewer</span>';
        button.addEventListener("click", function () { open(tab); });
        holder.appendChild(button);
        return holder;
    }

    function ensureButton(tab) {
        const conf = TABS[tab];
        const negative = byId(conf.negative);
        if (!negative || !negative.parentNode) return false;
        const holder = byId("mc-lora-viewer-launch-" + tab) || launcher(tab);
        if (negative.nextElementSibling === holder) return true;
        negative.parentNode.insertBefore(holder, negative.nextSibling);
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
