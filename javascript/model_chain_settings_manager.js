// Model Chain -- the generation settings as accordions, and the Settings Manager.
//
// Asked (2026-11-23), over a screenshot of Txt2Img's settings column on a
// phone: "i want everything that is not prompt (positive, negative, literal,
// and lora viewer), to be in an accordion. Hi res already has it, but
// generation config like batch and resolution should be collapsible as well.
// Honestly, i would like a last item added to this list that is a "Settings
// Manager", which would have options for me to rearrange or show/hide the
// other accordions in real time."
//
// What this does, on Txt2Img and Img2Img, once the page is built:
//
//   1. Everything in the settings column under the prompt boxes becomes a list
//      of sections (`.mc-sm-list`), one item each. A block of Forge's that is
//      not an accordion -- the tool row and styles, Sampling, Size and Batch,
//      CFG, Seed, Img2Img's image and its denoising, the Script choice -- is
//      put inside an accordion of this file's, open until it is closed. An
//      accordion already there -- Hires. fix, Refiner, every script's own --
//      is an item as it is.
//   2. The last item is the Settings Manager: a row per section with a grip to
//      drag it, ↑ and ↓, and an eye to show or hide it. Every change is made
//      to the page as it is pressed and kept in this browser, per tab
//      (localStorage); Reset puts Forge's order back, everything shown and
//      open.
//
// How the accordions look like Gradio's
// -------------------------------------
// They are built of Gradio's own accordion's classes, read off an accordion
// already on the page -- its generated `svelte-…` classes included, since the
// rules that draw an accordion are scoped to them -- so under Forge's own
// stylesheet, the Model Chain theme or any other, one of these is drawn as
// that theme draws an accordion. A page without an accordion to read gets
// this file's stylesheet's version instead. The open state is this file's:
// a press on the label opens or closes it, as Gradio's does.
//
// What is moved, and how it goes back
// -----------------------------------
// Gradio's blocks are moved -- one insertBefore each, never rebuilt or copied;
// Gradio addresses them by id and does not care where in the document they
// are -- into the items, out of the containers Forge built them in (the
// column itself, its accordions row, the script container). Every move is
// recorded, and turning the setting off (Settings, Model Chain: Page) puts
// every block back where it was found, in its old order, and takes the list
// away. Nothing Forge or an extension does to a block is changed: an
// accordion's open state is never written here (Forge's InputAccordion opens
// its accordion when it is switched on, and the reverse), and a block Gradio
// hides is hidden inside its item -- an item with nothing to show is not
// drawn (Override Settings until a paste fills it, Img2Img's inpaint controls
// off the Inpaint tab), and is drawn again when it has.
//
// What is not touched: the prompt boxes and what this extension placed among
// them (the Lora Viewer button, the Literal row), the toprow Forge builds for
// the classic prompt-box style (and the theme moves into the column), the
// results column. The list is put after them.
//
// What this file must never become: a second way of setting anything. It
// reads no value and writes none; a block is where Gradio drew it or inside
// one of these items, and works the same in both.
(function () {
    "use strict";

    const VERSION = 1;
    const TABS = ["txt2img", "img2img"];
    // The Forge setting (Settings, Model Chain: Page), on by default.
    const OPTION = "model_chain_settings_manager";
    // Per browser and per tab: {v, order, hidden, open}.
    const STORE = "mc-settings-manager:";
    const MANAGER = "manager";
    const MANAGER_LABEL = "Settings Manager";
    // How far a grip is dragged before it moves anything.
    const DRAG_SLOP = 4;

    // Forge's own blocks that are not accordions, by what they hold, in the
    // order Forge builds them (modules/ui.py, ui_toprow.py and the
    // processing_scripts it places by section). The first that matches names
    // the section; consecutive pieces with the same key are one section (the
    // seed's row and its extras, Img2Img's image tabs and its resize mode).
    const KNOWN = [
        {key: "tools", label: "Tools & Styles",
         ids: (tab) => [tab + "_tools", tab + "_styles_row"], classes: ["toprow-compact-stylerow"]},
        {key: "image", label: "Image", ids: () => ["mode_img2img", "resize_mode"]},
        {key: "inpaint", label: "Inpaint", ids: () => ["inpaint_controls"]},
        {key: "sampler", label: "Sampling", ids: (tab) => ["sampler_selection_" + tab, tab + "_sampling"]},
        {key: "size", label: "Size", ids: (tab) => [tab + "_column_size", tab + "_width"]},
        {key: "batch", label: "Batch", ids: (tab) => [tab + "_column_batch", tab + "_batch_count"]},
        {key: "cfg", label: "CFG", ids: (tab) => [tab + "_cfg_scale", tab + "_distilled_cfg_scale"]},
        {key: "denoise", label: "Denoising", ids: (tab) => [tab + "_denoising_strength"]},
        {key: "seed", label: "Seed", ids: (tab) => [tab + "_seed_row", tab + "_seed_extras", tab + "_seed"]},
        {key: "override", label: "Override Settings",
         ids: (tab) => [tab + "_override_settings_row", tab + "_override_settings"]},
        {key: "script", label: "Script", ids: () => ["script_list"]},
    ];

    // Lucide (ISC), drawn as SVG: no glyph a theme could swap.
    const ICONS = {
        "grip-vertical": '<circle cx="9" cy="12" r="1"/><circle cx="9" cy="5" r="1"/><circle cx="9" cy="19" r="1"/><circle cx="15" cy="12" r="1"/><circle cx="15" cy="5" r="1"/><circle cx="15" cy="19" r="1"/>',
        "chevron-up": '<path d="m18 15-6-6-6 6"/>',
        "chevron-down": '<path d="m6 9 6 6 6-6"/>',
        "eye": '<path d="M2.062 12.348a1 1 0 0 1 0-.696 10.75 10.75 0 0 1 19.876 0 1 1 0 0 1 0 .696 10.75 10.75 0 0 1-19.876 0"/><circle cx="12" cy="12" r="3"/>',
        "eye-off": '<path d="M10.733 5.076a10.744 10.744 0 0 1 11.205 6.575 1 1 0 0 1 0 .696 10.747 10.747 0 0 1-1.444 2.49"/><path d="M14.084 14.158a3 3 0 0 1-4.242-4.242"/><path d="M17.479 17.499a10.75 10.75 0 0 1-15.417-5.151 1 1 0 0 1 0-.696 10.75 10.75 0 0 1 4.446-5.143"/><path d="m2 2 20 20"/>',
        "rotate-ccw": '<path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/>',
    };

    function icon(name) {
        return '<svg class="mc-sm-icon" viewBox="0 0 24 24" width="18" height="18" fill="none" '
            + 'stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" '
            + 'aria-hidden="true" focusable="false">' + (ICONS[name] || "") + "</svg>";
    }

    // -- small helpers ---------------------------------------------------------

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

    function make(tag, classes, attributes) {
        const node = document.createElement(tag);
        (classes || []).forEach((name) => { if (name) node.classList.add(name); });
        Object.keys(attributes || {}).forEach((name) => node.setAttribute(name, attributes[name]));
        return node;
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

    function enabled() {
        return setting(OPTION, true) !== false;
    }

    function matchesOne(node, selectors) {
        return !!(node && node.matches && selectors.some((selector) => node.matches(selector)));
    }

    // Whether `node` is, or holds, one of `ids` (or carries one of `classes`).
    function holds(node, ids, classes) {
        if (!node || node.nodeType !== 1) return false;
        if (ids.indexOf(node.id) >= 0) return true;
        if ((classes || []).some((name) => node.classList.contains(name))) return true;
        return ids.some((id) => {
            try {
                return !!node.querySelector("#" + id);
            } catch (error) {
                return false;
            }
        });
    }

    // Whether `node` draws anything: displayed itself, and a block Gradio
    // shows (or a control) somewhere inside. The first visible leaf ends the
    // search, so a big panel costs a few reads.
    function shows(node) {
        if (!node || node.nodeType !== 1 || node.hidden) return false;
        let style;
        try {
            style = window.getComputedStyle(node);
        } catch (error) {
            return true;
        }
        if (style.display === "none") return false;
        if (matchesOne(node, [".block", "button", "input", "select", "textarea", "img", "canvas",
                              "video", "label", "svg"])) {
            return true;
        }
        const children = node.children;
        if (!children || !children.length) return !!(node.textContent || "").trim();
        for (let i = 0; i < children.length; i += 1) {
            if (shows(children[i])) return true;
        }
        return false;
    }

    function meaningfulId(node) {
        const id = node && node.id;
        if (!id || /^component-\d+$/.test(id) || /^input-accordion-\d+$/.test(id)) return "";
        return id;
    }

    function textOf(node) {
        return node ? (node.textContent || "").replace(/\s+/g, " ").trim() : "";
    }

    function slug(text) {
        return String(text).toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 48)
            || "section";
    }

    // -- what a piece is ---------------------------------------------------------

    function isAccordion(node) {
        return !!(node && node.nodeType === 1
                  && (node.classList.contains("gradio-accordion")
                      || (node.classList.contains("block") && !!childWith(node, "label-wrap"))));
    }

    function childWith(node, name) {
        if (!node || !node.children) return null;
        for (let i = 0; i < node.children.length; i += 1) {
            if (node.children[i].classList.contains(name)) return node.children[i];
        }
        return null;
    }

    // The accordion a piece is: the piece itself, or the one thing a script's
    // group draws (Forge wraps every script's UI in a group of its own, with
    // an inner `.styler`, and an InputAccordion's own hidden checkbox can sit
    // beside it).
    function accordionOf(piece) {
        if (isAccordion(piece)) return piece;
        if (!piece || !piece.classList) return null;
        if (!(piece.classList.contains("gr-group") || piece.classList.contains("gradio-group"))) return null;
        const inner = childWith(piece, "styler") || piece;
        const drawn = Array.prototype.filter.call(inner.children, shows);
        if (drawn.length === 1 && isAccordion(drawn[0])) return drawn[0];
        if (!drawn.length) {
            const all = Array.prototype.filter.call(inner.children, isAccordion);
            if (all.length === 1) return all[0];
        }
        return null;
    }

    function accordionLabel(accordion) {
        const wrap = childWith(accordion, "label-wrap");
        if (!wrap) return "";
        const span = Array.prototype.find.call(wrap.children,
            (child) => child.tagName === "SPAN" && !child.classList.contains("icon"));
        return textOf(span) || textOf(wrap).replace(/[▼▶◀]/g, "").trim();
    }

    function knownOf(piece, tab) {
        for (const rule of KNOWN) {
            if (holds(piece, rule.ids(tab), rule.classes)) {
                let label = rule.label;
                if (rule.key === "size" && holds(piece, [tab + "_column_batch", tab + "_batch_count"])) {
                    label = "Size & Batch";
                }
                return {key: rule.key, label};
            }
        }
        return null;
    }

    function guessLabel(piece) {
        let probe = null;
        try {
            probe = piece.querySelector('.label-wrap > span, [data-testid="block-info"], label > span, h1, h2, h3, h4');
        } catch (error) {
            probe = null;
        }
        return textOf(probe).slice(0, 60) || "Settings";
    }

    // -- the prompt area, and the containers that are walked into -----------------

    function isPromptArea(node, tab) {
        if (!node || node.nodeType !== 1) return false;
        if (node.id === tab + "_toprow" || node.id === tab + "_prompt_container") return true;
        try {
            return !!node.querySelector("#" + tab + "_prompt, #" + tab + "_neg_prompt");
        } catch (error) {
            return false;
        }
    }

    // The containers whose children are sections: Forge's accordions row, the
    // script container (its `.styler`), and a form Gradio put round several
    // components in the column.
    function walkInto(node, tab) {
        if (!node || node.nodeType !== 1) return null;
        if (node.id === tab + "_accordions") return node;
        if (node.id === tab + "_script_container") return childWith(node, "styler") || node;
        if (node.classList.contains("form") && node.children.length) return node;
        return null;
    }

    // -- per tab ------------------------------------------------------------------

    const tabs = {};
    let drag = null;

    function load(tab) {
        let data = null;
        try {
            const raw = window.localStorage.getItem(STORE + tab);
            data = raw ? JSON.parse(raw) : null;
        } catch (error) {
            data = null;
        }
        const list = (value) => (Array.isArray(value) ? value.filter((item) => typeof item === "string") : []);
        const open = {};
        if (data && data.open && typeof data.open === "object") {
            Object.keys(data.open).forEach((key) => { open[key] = !!data.open[key]; });
        }
        return {order: list(data && data.order), hidden: list(data && data.hidden), open};
    }

    function save(entry) {
        const data = {v: VERSION, order: entry.order.slice(), hidden: Array.from(entry.hidden),
                      open: Object.assign({}, entry.open)};
        try {
            window.localStorage.setItem(STORE + entry.tab, JSON.stringify(data));
        } catch (error) {
            /* a choice that is not kept is still made on this page */
        }
    }

    function forget(tab) {
        try {
            window.localStorage.removeItem(STORE + tab);
        } catch (error) {
            /* nothing kept */
        }
    }

    // Gradio's accordion's classes, read off one on the page: the generated
    // ones are what its scoped rules match.
    function template(column) {
        let accordion = column.querySelector(".gradio-accordion:not(.mc-sm-acc)");
        if (!accordion) accordion = app().querySelector(".gradio-accordion:not(.mc-sm-acc)");
        if (!accordion) return null;
        const scoped = (node, keep) => (node ? Array.prototype.filter.call(node.classList,
            (name) => name.indexOf("svelte-") === 0 || keep.indexOf(name) >= 0) : []);
        const wrap = childWith(accordion, "label-wrap");
        const span = wrap ? Array.prototype.find.call(wrap.children,
            (child) => child.tagName === "SPAN" && !child.classList.contains("icon")) : null;
        const mark = wrap ? childWith(wrap, "icon") : null;
        const content = wrap ? wrap.nextElementSibling : null;
        const inner = content ? content.firstElementChild : null;
        return {
            block: scoped(accordion, ["block", "gradio-accordion", "padded"]),
            wrap: scoped(wrap, ["label-wrap"]),
            span: scoped(span, []),
            icon: scoped(mark, ["icon"]),
            inner: scoped(inner, ["gap"]),
        };
    }

    function makeAccordion(entry, key, label, open) {
        const tpl = entry.template;
        const block = make("div", (tpl ? tpl.block : ["block", "gradio-accordion", "padded", "mc-sm-own"])
            .concat(["mc-sm-acc"]));
        block.id = "mc-sm-" + entry.tab + "-" + slug(key);
        block.style.borderStyle = "solid";
        block.style.borderWidth = "var(--block-border-width)";
        block.style.overflow = "visible";
        block.style.minWidth = "min(0px, 100%)";
        const button = make("button", tpl ? tpl.wrap : ["label-wrap"], {type: "button"});
        const name = make("span", tpl ? tpl.span : []);
        name.textContent = label;
        const mark = make("span", tpl ? tpl.icon : ["icon"], {"aria-hidden": "true"});
        mark.textContent = "▼";
        button.appendChild(name);
        button.appendChild(document.createTextNode(" "));
        button.appendChild(mark);
        const body = make("div", ["mc-sm-body"]);
        body.id = block.id + "-body";
        const inner = make("div", (tpl ? tpl.inner : []).concat(["mc-sm-inner"]));
        body.appendChild(inner);
        button.setAttribute("aria-controls", body.id);
        block.appendChild(button);
        block.appendChild(body);
        const accordion = {block, button, body, inner, mark};
        drawOpen(accordion, open);
        button.addEventListener("click", function () {
            setOpen(entry.tab, key, !accordion.open);
        });
        return accordion;
    }

    function drawOpen(accordion, open) {
        accordion.open = !!open;
        accordion.button.classList.toggle("open", accordion.open);
        accordion.button.setAttribute("aria-expanded", accordion.open ? "true" : "false");
        accordion.mark.style.transform = accordion.open ? "rotate(0deg)" : "rotate(90deg)";
        accordion.body.style.display = accordion.open ? "block" : "none";
    }

    function moveInto(entry, node, parent, before) {
        if (!node || !parent) return;
        entry.moved.push({node, parent: node.parentNode, next: node.nextSibling});
        parent.insertBefore(node, before || null);
    }

    // The pieces under the prompt area, in the order Forge built them: each
    // child of the column, and each child of a container walked into.
    function collect(entry) {
        const pieces = [];
        const column = entry.column;
        Array.prototype.slice.call(column.children).forEach((child) => {
            if (child === entry.list || child.classList.contains("mc-sm-list")) return;
            if (isPromptArea(child, entry.tab) || entry.known.has(child)) return;
            // A container already walked into stays one, emptied or not, so a
            // block an extension adds to it later is found there.
            const into = entry.containers.has(child) ? (walkInto(child, entry.tab) || child)
                : walkInto(child, entry.tab);
            if (into) {
                entry.containers.add(child);
                Array.prototype.slice.call(into.children).forEach((node) => {
                    if (!entry.known.has(node)) pieces.push({node, from: child});
                });
                return;
            }
            pieces.push({node: child, from: null});
        });
        return pieces;
    }

    function uniqueKey(entry, key) {
        if (!entry.items.has(key) && !entry.claimed.has(key)) return key;
        let n = 2;
        while (entry.items.has(key + "#" + n) || entry.claimed.has(key + "#" + n)) n += 1;
        return key + "#" + n;
    }

    // Pieces into sections. An accordion is a section as it is; a block Forge
    // names is wrapped; anything else that draws is wrapped under the first
    // label it holds; a piece that draws nothing and is not Forge's rides with
    // the section after it (an InputAccordion's hidden checkbox, a script
    // group with nothing visible), or the one before at the end. Everything
    // after the Script choice in the script container is that choice's: the
    // chosen script's own controls.
    function classify(entry, pieces) {
        const sections = [];
        let pending = [];
        let script = null;
        pieces.forEach((piece) => {
            const node = piece.node;
            if (script && piece.from && piece.from === script.from) {
                script.pieces.push(node);
                return;
            }
            const accordion = accordionOf(node);
            let section;
            if (accordion) {
                const label = accordionLabel(accordion) || "Accordion";
                const id = meaningfulId(node) || meaningfulId(accordion);
                section = {key: id ? "id:" + id : "label:" + label, label, kind: "accordion", known: ""};
            } else {
                const known = knownOf(node, entry.tab);
                if (!known && !shows(node)) {
                    pending.push(node);
                    return;
                }
                if (known) {
                    const last = sections[sections.length - 1];
                    if (last && last.known === known.key) {
                        last.pieces = last.pieces.concat(pending, [node]);
                        pending = [];
                        return;
                    }
                    section = {key: known.key, label: known.label, kind: "wrap", known: known.key};
                } else {
                    const label = guessLabel(node);
                    const id = meaningfulId(node);
                    section = {key: id ? "id:" + id : "label:" + label, label, kind: "wrap", known: ""};
                }
            }
            section.key = uniqueKey(entry, section.key);
            entry.claimed.add(section.key);
            section.pieces = pending.concat([node]);
            pending = [];
            sections.push(section);
            if (section.known === "script") {
                script = section;
                script.from = piece.from;
            }
        });
        if (pending.length && sections.length) {
            const last = sections[sections.length - 1];
            last.pieces = last.pieces.concat(pending);
        }
        return sections;
    }

    function makeItem(entry, section) {
        const element = make("div", ["mc-sm-item"], {"data-mc-sm-key": section.key,
                                                     "data-mc-sm-kind": section.kind});
        const item = {key: section.key, label: section.label, kind: section.kind, element,
                      pieces: section.pieces.slice(), accordion: null, empty: false};
        if (section.kind === "wrap") {
            const open = Object.prototype.hasOwnProperty.call(entry.open, section.key)
                ? entry.open[section.key] : true;
            item.accordion = makeAccordion(entry, section.key, section.label, open);
            element.appendChild(item.accordion.block);
            item.pieces.forEach((node) => moveInto(entry, node, item.accordion.inner, null));
        } else {
            item.pieces.forEach((node) => moveInto(entry, node, element, null));
        }
        item.pieces.forEach((node) => entry.known.add(node));
        return item;
    }

    function markEmptied(entry) {
        entry.containers.forEach((container) => {
            const into = walkInto(container, entry.tab) || container;
            const left = Array.prototype.some.call(into.children, (child) => !entry.known.has(child));
            if (!left && !container.classList.contains("mc-sm-emptied")) {
                container.classList.add("mc-sm-emptied");
                entry.emptied.push(container);
            }
        });
    }

    // The order to draw: the kept one, with every section it does not name
    // after the section Forge put before it.
    function arranged(entry) {
        const present = entry.natural.filter((key) => entry.items.has(key));
        const order = entry.order.filter((key) => entry.items.has(key));
        present.forEach((key, index) => {
            if (order.indexOf(key) >= 0) return;
            let at = 0;
            for (let back = index - 1; back >= 0; back -= 1) {
                const found = order.indexOf(present[back]);
                if (found >= 0) {
                    at = found + 1;
                    break;
                }
            }
            order.splice(at, 0, key);
        });
        return order;
    }

    // Draw the order, the hidden and the empty; moves only what is out of place.
    function apply(entry) {
        const order = arranged(entry);
        const current = Array.prototype.filter.call(entry.list.children,
            (child) => child !== entry.manager.element).map((child) => child.getAttribute("data-mc-sm-key"));
        if (current.join("\n") !== order.join("\n")) {
            order.forEach((key) => {
                entry.list.insertBefore(entry.items.get(key).element, entry.manager.element);
            });
        }
        entry.items.forEach((item, key) => {
            const hidden = entry.hidden.has(key);
            if (item.element.hidden !== hidden) item.element.hidden = hidden;
        });
        return order;
    }

    function refreshEmpty(entry) {
        let changed = false;
        entry.items.forEach((item) => {
            const empty = !item.pieces.some(shows);
            if (empty !== item.empty) {
                item.empty = empty;
                changed = true;
            }
            if (empty !== item.element.hasAttribute("data-mc-sm-empty")) {
                if (empty) item.element.setAttribute("data-mc-sm-empty", "");
                else item.element.removeAttribute("data-mc-sm-empty");
            }
        });
        if (changed) drawRows(entry);
    }

    function scheduleEmpty(entry) {
        if (entry.emptyPending) return;
        entry.emptyPending = true;
        const run = function () {
            entry.emptyPending = false;
            if (tabs[entry.tab] === entry) refreshEmpty(entry);
        };
        if (typeof requestAnimationFrame === "function") requestAnimationFrame(run);
        else setTimeout(run, 16);
    }

    // -- the Settings Manager ------------------------------------------------------

    function makeManager(entry) {
        const element = make("div", ["mc-sm-item", "mc-sm-manager-item"],
                             {"data-mc-sm-key": MANAGER, "data-mc-sm-kind": "manager"});
        const open = Object.prototype.hasOwnProperty.call(entry.open, MANAGER) ? entry.open[MANAGER] : false;
        const accordion = makeAccordion(entry, MANAGER, MANAGER_LABEL, open);
        element.appendChild(accordion.block);
        const panel = make("div", ["mc-sm-manager"], {role: "group", "aria-label": MANAGER_LABEL});
        const rows = make("ol", ["mc-sm-rows"]);
        const foot = make("div", ["mc-sm-foot"]);
        const reset = make("button", ["mc-sm-reset"], {type: "button",
                                                       title: "Forge's order, every section shown and open"});
        reset.innerHTML = icon("rotate-ccw") + "<span>Reset</span>";
        reset.addEventListener("click", function () { resetTab(entry.tab); });
        foot.appendChild(reset);
        panel.appendChild(rows);
        panel.appendChild(foot);
        accordion.inner.appendChild(panel);
        rows.addEventListener("pointerdown", (event) => gripDown(entry, event));
        rows.addEventListener("pointermove", (event) => gripMove(entry, event));
        rows.addEventListener("pointerup", (event) => gripUp(entry, event, false));
        rows.addEventListener("pointercancel", (event) => gripUp(entry, event, true));
        rows.addEventListener("lostpointercapture", (event) => gripUp(entry, event, true));
        return {element, accordion, rows};
    }

    function rowButton(name, classes, label, attributes) {
        const button = make("button", ["mc-sm-button"].concat(classes),
                            Object.assign({type: "button", "aria-label": label, title: label}, attributes || {}));
        button.innerHTML = icon(name);
        return button;
    }

    function drawRows(entry) {
        const rows = entry.manager.rows;
        const order = arranged(entry);
        const focused = document.activeElement && rows.contains(document.activeElement)
            ? {key: document.activeElement.closest(".mc-sm-row").getAttribute("data-mc-sm-key"),
               role: Array.prototype.find.call(document.activeElement.classList,
                   (name) => ["mc-sm-grip", "mc-sm-up", "mc-sm-down", "mc-sm-eye"].indexOf(name) >= 0)}
            : null;
        rows.textContent = "";
        order.forEach((key, index) => {
            const item = entry.items.get(key);
            const shown = !entry.hidden.has(key);
            const row = make("li", ["mc-sm-row"], {"data-mc-sm-key": key});
            if (!shown) row.classList.add("mc-sm-off");
            if (item.empty) row.classList.add("mc-sm-idle");
            const grip = rowButton("grip-vertical", ["mc-sm-grip"], "Move " + item.label + " (drag, or the arrow keys)");
            const name = make("span", ["mc-sm-name"]);
            name.textContent = item.label;
            if (item.empty) name.title = "Nothing to show here now";
            const up = rowButton("chevron-up", ["mc-sm-up"], "Move " + item.label + " up");
            const down = rowButton("chevron-down", ["mc-sm-down"], "Move " + item.label + " down");
            up.disabled = index === 0;
            down.disabled = index === order.length - 1;
            const eye = rowButton(shown ? "eye" : "eye-off", ["mc-sm-eye"],
                                  (shown ? "Hide " : "Show ") + item.label,
                                  {"aria-pressed": shown ? "true" : "false"});
            up.addEventListener("click", () => move(entry.tab, key, -1));
            down.addEventListener("click", () => move(entry.tab, key, 1));
            eye.addEventListener("click", () => setShown(entry.tab, key, entry.hidden.has(key)));
            grip.addEventListener("keydown", (event) => {
                if (event.key === "ArrowUp" || event.key === "ArrowDown") {
                    event.preventDefault();
                    move(entry.tab, key, event.key === "ArrowUp" ? -1 : 1);
                }
            });
            row.appendChild(grip);
            row.appendChild(name);
            row.appendChild(up);
            row.appendChild(down);
            row.appendChild(eye);
            rows.appendChild(row);
        });
        if (focused && focused.role) {
            const row = rows.querySelector('.mc-sm-row[data-mc-sm-key="' + cssEscape(focused.key) + '"]');
            const again = row && row.querySelector("." + focused.role);
            if (again && !again.disabled) {
                try {
                    again.focus({preventScroll: true});
                } catch (error) {
                    /* focus is a courtesy */
                }
            }
        }
    }

    function cssEscape(text) {
        if (window.CSS && typeof window.CSS.escape === "function") return window.CSS.escape(text);
        return String(text).replace(/["\\]/g, "\\$&");
    }

    // -- dragging a row ----------------------------------------------------------

    function gripDown(entry, event) {
        const grip = event.target && event.target.closest ? event.target.closest(".mc-sm-grip") : null;
        if (!grip || event.button !== 0 || drag) return;
        const row = grip.closest(".mc-sm-row");
        drag = {entry, row, key: row.getAttribute("data-mc-sm-key"), pointer: event.pointerId,
                startY: event.clientY, active: false, touch: event.pointerType !== "mouse", locked: []};
        try {
            entry.manager.rows.setPointerCapture(event.pointerId);
        } catch (error) {
            /* the moves still arrive while the pointer is over the list */
        }
        event.preventDefault();
    }

    function gripMove(entry, event) {
        if (!drag || drag.entry !== entry || event.pointerId !== drag.pointer) return;
        const rows = entry.manager.rows;
        if (!drag.active) {
            if (Math.abs(event.clientY - drag.startY) < DRAG_SLOP) return;
            drag.active = true;
            drag.before = entry.order.slice();
            drag.row.classList.add("mc-sm-dragging");
            if (drag.touch) lockScroll(rows);
        }
        event.preventDefault();
        const others = Array.prototype.filter.call(rows.children, (row) => row !== drag.row);
        let index = 0;
        others.forEach((row) => {
            const box = row.getBoundingClientRect();
            if (event.clientY > box.top + box.height / 2) index += 1;
        });
        const current = Array.prototype.indexOf.call(rows.children, drag.row);
        if (index === current) return;
        rows.insertBefore(drag.row, others[index] || null);
        place(entry, drag.key, index, false);
    }

    function gripUp(entry, event, cancelled) {
        if (!drag || drag.entry !== entry || (event.pointerId !== undefined && event.pointerId !== drag.pointer)) return;
        const done = drag;
        drag = null;
        unlockScroll(done);
        try {
            if (entry.manager.rows.hasPointerCapture && entry.manager.rows.hasPointerCapture(done.pointer)) {
                entry.manager.rows.releasePointerCapture(done.pointer);
            }
        } catch (error) {
            /* already let go */
        }
        done.row.classList.remove("mc-sm-dragging");
        if (!done.active) return;
        // A drag the browser cancels (it took the pointer for a gesture of its
        // own) puts the order back as the drag found it; one let go keeps it.
        if (cancelled) {
            entry.order = done.before;
            apply(entry);
        } else {
            save(entry);
        }
        drawRows(entry);
    }

    // A finger's drag holds the page still: a Gecko browser pans the page under
    // a touch the page has claimed once it is slow to answer, whatever
    // touch-action says (CLAUDE.md, "A Gecko browser pans the page ...").
    function lockScroll(from) {
        const targets = [document.scrollingElement || document.documentElement];
        let node = from && from.parentElement;
        while (node && node !== document.body && node !== document.documentElement) {
            const style = window.getComputedStyle(node);
            if (/(auto|scroll)/.test(style.overflowY) && node.scrollHeight > node.clientHeight) {
                targets.push(node);
                break;
            }
            node = node.parentElement;
        }
        targets.forEach((target) => {
            drag.locked.push({target, overflow: target.style.getPropertyValue("overflow"),
                              priority: target.style.getPropertyPriority("overflow")});
            target.style.setProperty("overflow", "hidden", "important");
        });
    }

    function unlockScroll(done) {
        (done.locked || []).forEach((record) => {
            if (record.overflow) record.target.style.setProperty("overflow", record.overflow, record.priority);
            else record.target.style.removeProperty("overflow");
        });
        done.locked = [];
    }

    // -- what the manager does -------------------------------------------------------

    function place(entry, key, index, redraw) {
        const order = arranged(entry).filter((other) => other !== key);
        const at = Math.max(0, Math.min(order.length, index));
        order.splice(at, 0, key);
        const absent = entry.order.filter((other) => !entry.items.has(other));
        entry.order = order.concat(absent);
        apply(entry);
        if (redraw !== false) {
            save(entry);
            drawRows(entry);
        }
    }

    function move(tab, key, delta) {
        const entry = tabs[tab];
        if (!entry || !entry.items.has(key)) return false;
        const order = arranged(entry);
        const index = order.indexOf(key);
        const target = index + delta;
        if (target < 0 || target >= order.length) return false;
        place(entry, key, target, true);
        return true;
    }

    function moveTo(tab, key, index) {
        const entry = tabs[tab];
        if (!entry || !entry.items.has(key)) return false;
        place(entry, key, index, true);
        return true;
    }

    function setShown(tab, key, shown) {
        const entry = tabs[tab];
        if (!entry || !entry.items.has(key)) return false;
        if (shown) entry.hidden.delete(key);
        else entry.hidden.add(key);
        apply(entry);
        save(entry);
        drawRows(entry);
        return true;
    }

    function setOpen(tab, key, open) {
        const entry = tabs[tab];
        if (!entry) return false;
        const accordion = key === MANAGER ? entry.manager.accordion
            : (entry.items.has(key) ? entry.items.get(key).accordion : null);
        if (!accordion) return false;
        drawOpen(accordion, open);
        entry.open[key] = !!open;
        save(entry);
        return true;
    }

    function resetTab(tab) {
        const entry = tabs[tab];
        if (!entry) return false;
        entry.order = entry.natural.slice();
        entry.hidden = new Set();
        const keepManager = entry.manager.accordion.open;
        entry.open = {};
        entry.items.forEach((item) => {
            if (item.accordion) drawOpen(item.accordion, true);
        });
        apply(entry);
        forget(tab);
        if (keepManager) {
            entry.open[MANAGER] = true;
            save(entry);
        }
        drawRows(entry);
        return true;
    }

    // -- building and taking down ------------------------------------------------

    function build(tab) {
        if (!enabled() || tabs[tab]) return tabs[tab] || null;
        const column = byId(tab + "_settings");
        if (!column || !column.querySelector) return null;
        const stored = load(tab);
        const entry = {
            tab, column, list: null, manager: null, template: template(column),
            items: new Map(), natural: [], order: stored.order, hidden: new Set(stored.hidden),
            open: stored.open, moved: [], emptied: [], containers: new Set(), known: new Set(),
            claimed: new Set(), observer: null, emptyPending: false,
        };
        const pieces = collect(entry);
        if (!pieces.length) return null;
        const sections = classify(entry, pieces);
        if (!sections.length) return null;
        tabs[tab] = entry;

        const list = make("div", ["mc-sm-list", "mc-theme-plain"]);
        list.id = "mc-sm-list-" + tab;
        entry.list = list;
        // After the prompt area: the last child of the column that is part of
        // it, or the column's top when it holds none.
        const children = Array.prototype.slice.call(column.children);
        let after = null;
        children.forEach((child) => { if (isPromptArea(child, tab)) after = child; });
        column.insertBefore(list, after ? after.nextSibling : column.firstChild);

        entry.manager = makeManager(entry);
        list.appendChild(entry.manager.element);
        sections.forEach((section) => {
            const item = makeItem(entry, section);
            entry.items.set(item.key, item);
            entry.natural.push(item.key);
            list.insertBefore(item.element, entry.manager.element);
        });
        markEmptied(entry);
        apply(entry);
        refreshEmpty(entry);
        drawRows(entry);
        watch(entry);
        return entry;
    }

    // A block that turns up later, at the column's top level or in one of the
    // containers walked into (an extension that draws after the page loads),
    // joins the list in Forge's order after the last section built.
    function adopt(entry) {
        const pieces = collect(entry);
        if (!pieces.length) return false;
        const sections = classify(entry, pieces);
        if (!sections.length) return false;
        sections.forEach((section) => {
            const item = makeItem(entry, section);
            entry.items.set(item.key, item);
            entry.natural.push(item.key);
            entry.list.insertBefore(item.element, entry.manager.element);
        });
        markEmptied(entry);
        apply(entry);
        refreshEmpty(entry);
        drawRows(entry);
        return true;
    }

    function watch(entry) {
        if (typeof MutationObserver !== "function") return;
        const observer = new MutationObserver(() => scheduleEmpty(entry));
        observer.observe(entry.list, {attributes: true, subtree: true,
                                      attributeFilter: ["class", "style", "hidden"]});
        entry.observer = observer;
    }

    function teardown(tab) {
        const entry = tabs[tab];
        if (!entry) return false;
        if (drag && drag.entry === entry) {
            unlockScroll(drag);
            drag = null;
        }
        if (entry.observer) entry.observer.disconnect();
        entry.moved.slice().reverse().forEach((record) => {
            if (!record.parent) return;
            const next = record.next && record.next.parentNode === record.parent ? record.next : null;
            try {
                record.parent.insertBefore(record.node, next);
            } catch (error) {
                /* a parent Gradio has since taken away */
            }
        });
        entry.emptied.forEach((container) => container.classList.remove("mc-sm-emptied"));
        if (entry.list && entry.list.parentNode) entry.list.parentNode.removeChild(entry.list);
        delete tabs[tab];
        return true;
    }

    function wire() {
        TABS.forEach((tab) => {
            try {
                if (!enabled()) {
                    teardown(tab);
                    return;
                }
                const entry = tabs[tab];
                if (!entry) {
                    build(tab);
                    return;
                }
                if (!entry.list.isConnected) {
                    teardown(tab);
                    build(tab);
                    return;
                }
                adopt(entry);
                refreshEmpty(entry);
            } catch (error) {
                console.error("Model Chain: the settings sections could not be laid out", error);
            }
        });
    }

    function describe(tab) {
        const entry = tabs[tab];
        if (!entry) return null;
        return arranged(entry).map((key) => {
            const item = entry.items.get(key);
            return {key, label: item.label, kind: item.kind, hidden: entry.hidden.has(key),
                    empty: item.empty, open: item.accordion ? item.accordion.open : null};
        });
    }

    window.mcSettingsManager = {
        version: VERSION,
        OPTION, STORE, MANAGER,
        wire, build, teardown, describe,
        move, moveTo, setShown, setOpen, reset: resetTab,
        refresh: (tab) => { if (tabs[tab]) refreshEmpty(tabs[tab]); },
        entry: (tab) => tabs[tab] || null,
    };

    if (typeof onUiLoaded === "function") onUiLoaded(wire);
    if (typeof onAfterUiUpdate === "function") onAfterUiUpdate(wire);
    if (typeof onOptionsChanged === "function") onOptionsChanged(wire);
})();
