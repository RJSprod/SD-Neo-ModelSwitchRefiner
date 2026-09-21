// Forge Assistant -- the floating panel, and everywhere it is allowed to sit.
//
// One launcher, six resting places, and a conversation inside it that is the
// same conversation LLM Studio has open. It runs no model, keeps no history of
// its own and starts no session; every word in it came from, and goes back to,
// the one guarded service.
//
// Why it is a DOM shell and not a Gradio component
// ------------------------------------------------
// There is no supported Forge callback that mounts a Gradio component outside
// the tab tree, and a panel that is only available on one tab is not a global
// assistant. So the shell is appended to `document.body` and talks to Python
// over HTTP; the handful of Gradio components it genuinely needs -- the
// capability field, the refresh bridge, the paste bridge -- live inside the LLM
// Studio tab, where components can live. That asymmetry is accepted rather than
// worked around.
//
// The geometry, and why it is arithmetic rather than CSS
// ------------------------------------------------------
// Six anchors, `position: fixed`, and a gap. What makes it fiddly is that the
// *visual* viewport is not the layout viewport once a phone's keyboard is up or
// the page is pinched: `visualViewport` reports the box the user can actually
// see, and a panel positioned against the layout viewport ends up under the
// keyboard. Safe-area insets are added once, then the gap -- adding the gap
// first and the inset afterwards puts a panel on an iPhone's home indicator.
// `devicePixelRatio` is never multiplied by anything here; CSS pixels are
// already the unit `position: fixed` speaks.
//
// What is stored is the anchor's *name*. Never pixels: a window resized between
// sessions would restore a panel to a coordinate that is now off-screen, and
// the one thing a launcher must never be is unreachable.
//
// Markdown, and why it is written out here
// ----------------------------------------
// The transcript is rendered by this file rather than scraped from the tab's
// `gr.Chatbot` -- scraping would tie the flyout to Gradio's internal DOM and
// break on a theme. Rendering means sanitising, and sanitising means an
// allow-list: text nodes, a few inline shapes, code, and links whose scheme is
// http, https or mailto. There is no path here by which a reply can introduce
// an element, an attribute or a URL scheme that was not written in this file.

(function () {
    "use strict";

    const NS = (window.forgeAssistant = window.forgeAssistant || {});

    const ROOT_ID = "forge-assistant-root";
    const ANCHORS = ["top-left", "top-center", "top-right",
                     "bottom-left", "bottom-center", "bottom-right"];
    const DEFAULT_ANCHOR = "bottom-right";

    const WIDE_GAP = 24;
    const NARROW_GAP = 16;
    const NARROW_VIEWPORT = 640;
    const DRAG_THRESHOLD = 6;
    const MIN_WIDTH = 320;
    const MAX_WIDTH = 640;
    const NOMINAL_WIDTH = 360;
    const RESIZE_STEP = 16;
    const BOTTOM_SLACK = 100;

    // -- geometry ---------------------------------------------------------- //

    function viewport() {
        const vv = window.visualViewport;
        if (vv && vv.width && vv.height) {
            return {left: vv.offsetLeft || 0, top: vv.offsetTop || 0,
                    width: vv.width, height: vv.height};
        }
        return {left: 0, top: 0,
                width: (document.documentElement && document.documentElement.clientWidth)
                    || window.innerWidth || 0,
                height: (document.documentElement && document.documentElement.clientHeight)
                    || window.innerHeight || 0};
    }

    function insets() {
        const probe = document.getElementById(ROOT_ID);
        if (!probe || !window.getComputedStyle) return {top: 0, right: 0, bottom: 0, left: 0};
        const style = window.getComputedStyle(probe);
        const read = (name) => parseFloat(style.getPropertyValue(name)) || 0;
        return {top: read("--forge-assistant-inset-top"),
                right: read("--forge-assistant-inset-right"),
                bottom: read("--forge-assistant-inset-bottom"),
                left: read("--forge-assistant-inset-left")};
    }

    function gapFor(view) {
        return view.width >= NARROW_VIEWPORT ? WIDE_GAP : NARROW_GAP;
    }

    // The one piece of arithmetic everything else depends on. Pure, so it can
    // be asserted against numbers a test chooses rather than against a browser.
    function anchorPoint(anchor, box, view, inset) {
        const gap = gapFor(view);
        const pad = inset || {top: 0, right: 0, bottom: 0, left: 0};
        const width = Math.min(box.width, view.width - pad.left - pad.right - gap * 2);
        const height = Math.min(box.height, view.height - pad.top - pad.bottom - gap * 2);
        const name = ANCHORS.indexOf(anchor) >= 0 ? anchor : DEFAULT_ANCHOR;
        const [vertical, horizontal] = name.split("-");
        let x;
        if (horizontal === "left") x = view.left + pad.left + gap;
        else if (horizontal === "right") x = view.left + view.width - pad.right - gap - width;
        else x = view.left + (view.width - width) / 2;
        const y = vertical === "top"
            ? view.top + pad.top + gap
            : view.top + view.height - pad.bottom - gap - height;
        return {left: Math.round(x), top: Math.round(y),
                width: Math.round(width), height: Math.round(height)};
    }

    // Nearest by the distance between the dragged element's centre and each
    // snapped rectangle's centre. On an exact tie the previous candidate wins,
    // so a drag held exactly between two anchors does not flicker between them.
    function nearestAnchor(centre, box, view, inset, previous) {
        const held = previous && ANCHORS.indexOf(previous) >= 0 ? previous : "";
        let best = held || DEFAULT_ANCHOR;
        let shortest = Infinity;
        // Squared distances, so the comparison never takes a square root, and
        // a tolerance because two of these are equal by construction rather
        // than by luck -- the two centre anchors are equidistant from the
        // middle of the viewport, and floating point does not promise that two
        // ways of computing the same number agree to the last bit.
        const TIE = 0.5;
        ANCHORS.forEach((anchor) => {
            const at = anchorPoint(anchor, box, view, inset);
            const dx = (at.left + at.width / 2) - centre.x;
            const dy = (at.top + at.height / 2) - centre.y;
            const distance = dx * dx + dy * dy;
            if (distance < shortest - TIE) {
                shortest = distance;
                best = anchor;
            } else if (Math.abs(distance - shortest) <= TIE && anchor === held) {
                // An exact tie keeps the candidate the drag already had.
                // Without this the earlier anchor in the list always wins, so a
                // drag held exactly between two of them flickers under the
                // finger at 60 frames a second.
                best = anchor;
            }
        });
        return best;
    }

    // -- markdown ---------------------------------------------------------- //

    function escapeHtml(text) {
        return String(text === undefined || text === null ? "" : text)
            .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
    }

    function safeUrl(raw) {
        const url = String(raw || "").trim();
        // An allow-list of three schemes and nothing else. `javascript:`,
        // `data:` and `vbscript:` are the obvious ones; the list is positive
        // rather than negative because the next scheme somebody finds is one a
        // blocklist has never heard of.
        if (/^(https?:|mailto:)/i.test(url)) return url;
        if (/^\//.test(url) && !/^\/\//.test(url)) return url;   // same-origin path
        return "";
    }

    function renderMarkdown(text) {
        // Escaped first, always. Everything below adds markup to text that can
        // no longer contain any.
        const source = escapeHtml(text);
        const blocks = [];
        // Fenced code, lifted out whole before anything inline is looked at, so
        // a backtick block containing asterisks is not italicised.
        let held = source.replace(/```([\s\S]*?)```/g, (match, body) => {
            blocks.push(body.replace(/^\n/, ""));
            return "\u0000CODE" + (blocks.length - 1) + "\u0000";
        });
        held = held
            .replace(/`([^`\n]+)`/g, "<code>$1</code>")
            .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
            .replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>")
            .replace(/\[([^\]\n]+)\]\(([^)\s]+)\)/g, (match, label, href) => {
                const url = safeUrl(href);
                if (!url) return label;
                // A new tab without an opener: a link in a reply must not be
                // able to reach back into this page.
                return '<a href="' + escapeHtml(url) + '" target="_blank" '
                    + 'rel="noopener noreferrer">' + label + "</a>";
            });
        held = held.split(/\n{2,}/).map((paragraph) => {
            if (/^\u0000CODE\d+\u0000$/.test(paragraph.trim())) return paragraph;
            return "<p>" + paragraph.replace(/\n/g, "<br>") + "</p>";
        }).join("");
        return held.replace(/\u0000CODE(\d+)\u0000/g, (match, index) =>
            '<pre class="forge-assistant-code"><code>' + blocks[Number(index)]
            + "</code></pre>");
    }

    // -- the shell --------------------------------------------------------- //

    function element(tag, className, text) {
        const node = document.createElement(tag);
        if (className) node.className = className;
        if (text !== undefined) node.textContent = text;
        return node;
    }

    function Shell(store, host, focus) {
        this.store = store;
        this.host = host;
        this.focus = focus;
        this.state = {
            anchorOverride: null,
            panelWidth: null,
            panelOpen: false,
            conversationExpanded: true,
            focusEnabled: false,
            focusWorkspaceId: null,
            showAnyway: false,
        };
        this.settings = {
            enabled: true,
            appearance: "text",
            label: "Forge Assistant",
            icon: "chat",
            defaultAnchor: DEFAULT_ANCHOR,
            bubbleWidth: 75,
            textSize: 0,
            density: "comfortable",
            avatars: true,
        };
        this.nodes = {};
        this.drag = null;
        this.disposers = [];
        this.frame = 0;
        this.following = true;
        this.lastRendered = "";
        this.shownEpoch = "";
        this.settled = false;
        this.settling = false;
        this._restore();
    }

    Shell.prototype.anchor = function () {
        return this.state.anchorOverride || this.settings.defaultAnchor || DEFAULT_ANCHOR;
    };

    Shell.prototype._storageKey = function () {
        return "ui:v5";
    };

    Shell.prototype._restore = function () {
        let raw = null;
        try {
            raw = window.sessionStorage.getItem(
                "forge-assistant:" + (NS.basePath() || "/") + ":" + this._storageKey());
        } catch (error) {
            raw = null;
        }
        if (!raw) return;
        try {
            const found = JSON.parse(raw);
            if (!found || found.schemaVersion !== 5) return;
            if (ANCHORS.indexOf(found.anchorOverride) >= 0) {
                this.state.anchorOverride = found.anchorOverride;
            }
            if (typeof found.panelWidth === "number") {
                this.state.panelWidth = Math.min(MAX_WIDTH,
                                                 Math.max(MIN_WIDTH, found.panelWidth));
            }
            ["panelOpen", "conversationExpanded"].forEach((name) => {
                if (typeof found[name] === "boolean") this.state[name] = found[name];
            });
            // Focus is deliberately not restored here. A reload starts outside
            // focus and the saved workspace is only honoured once an eligible
            // root has been verified -- see `resumeFocus`.
            this.state.focusWorkspaceId = typeof found.focusWorkspaceId === "string"
                ? found.focusWorkspaceId : null;
        } catch (error) {
            // Malformed session state is ignored, never repaired: a half-read
            // layout is worse than the default one.
        }
    };

    Shell.prototype._save = function () {
        const payload = {
            schemaVersion: 5,
            anchorOverride: this.state.anchorOverride,
            panelWidth: this.state.panelWidth,
            panelOpen: this.state.panelOpen,
            conversationExpanded: this.state.conversationExpanded,
            focusEnabled: this.state.focusEnabled,
            focusWorkspaceId: this.state.focusWorkspaceId,
        };
        try {
            window.sessionStorage.setItem(
                "forge-assistant:" + (NS.basePath() || "/") + ":" + this._storageKey(),
                JSON.stringify(payload));
        } catch (error) { /* memory only; the panel still works */ }
    };

    Shell.prototype.mount = function () {
        if (document.getElementById(ROOT_ID)) return false;
        const root = element("div", "forge-assistant-root");
        root.id = ROOT_ID;
        root.setAttribute("data-anchor", this.anchor());
        document.body.appendChild(root);
        this.nodes.root = root;

        this.buildLauncher();
        this.buildPanel();
        this.wire();
        // One statement decides which of the two is on screen, here and in
        // open() and close(), so there is no third place for them to disagree
        // and end up both drawn at once.
        this.showOpen(this.state.panelOpen);
        this.render(this.store.snapshot());
        return true;
    };

    Shell.prototype.buildLauncher = function () {
        const launcher = element("button", "forge-assistant-launcher");
        launcher.type = "button";
        launcher.id = "forge-assistant-launcher";
        launcher.setAttribute("aria-expanded", "false");
        launcher.setAttribute("aria-controls", "forge-assistant-panel");
        this.nodes.launcher = launcher;
        this.nodes.launcherLabel = element("span", "forge-assistant-launcher-label",
                                           this.settings.label);
        this.nodes.launcherIcon = element("span", "forge-assistant-launcher-icon");
        this.nodes.launcherIcon.setAttribute("aria-hidden", "true");
        this.nodes.launcherIcon.textContent = "●";
        this.nodes.unread = element("span", "forge-assistant-unread");
        this.nodes.unread.hidden = true;
        launcher.appendChild(this.nodes.launcherIcon);
        launcher.appendChild(this.nodes.launcherLabel);
        launcher.appendChild(this.nodes.unread);
        this.nodes.root.appendChild(launcher);
        this.applyAppearance();
    };

    Shell.prototype.applyAppearance = function () {
        const mode = this.settings.appearance;
        this.nodes.launcherIcon.hidden = mode === "text";
        this.nodes.launcherLabel.hidden = mode === "icon";
        // An accessible name even when the label is not drawn -- an icon-only
        // control with no name is a control a screen reader calls "button".
        this.nodes.launcher.setAttribute("aria-label", this.settings.label);
        this.nodes.launcher.title = this.settings.label;
        this.nodes.launcherLabel.textContent = this.settings.label;
    };

    Shell.prototype.buildPanel = function () {
        const panel = element("section", "forge-assistant-panel");
        panel.id = "forge-assistant-panel";
        panel.hidden = true;
        panel.setAttribute("aria-label", this.settings.label);
        this.nodes.panel = panel;

        const header = element("header", "forge-assistant-header");
        const title = element("h2", "forge-assistant-title", this.settings.label);
        // One control in the header, and it is the one people reach for: put
        // the panel away. There was a pin beside it for "keep open while
        // changing workspaces" -- a preference nobody asked for, occupying the
        // corner everybody aims at.
        const minimize = element("button", "forge-assistant-icon-button");
        minimize.type = "button";
        minimize.setAttribute("aria-label", "Minimize the assistant");
        minimize.title = "Minimize";
        minimize.textContent = "✕";
        header.appendChild(title);
        header.appendChild(minimize);
        panel.appendChild(header);
        Object.assign(this.nodes, {header, minimize, title});

        const nav = element("div", "forge-assistant-nav");
        const picker = element("button", "forge-assistant-nav-button", "Workspace");
        picker.type = "button";
        picker.setAttribute("aria-haspopup", "menu");
        picker.setAttribute("aria-expanded", "false");
        const utilities = element("button", "forge-assistant-nav-button", "⋯");
        utilities.type = "button";
        utilities.setAttribute("aria-haspopup", "menu");
        utilities.setAttribute("aria-expanded", "false");
        utilities.setAttribute("aria-label", "More actions");
        const focusToggle = element("button", "forge-assistant-nav-button", "Focus");
        focusToggle.type = "button";
        focusToggle.setAttribute("aria-pressed", "false");
        nav.appendChild(picker);
        nav.appendChild(focusToggle);
        nav.appendChild(utilities);
        panel.appendChild(nav);
        Object.assign(this.nodes, {nav, picker, utilities, focusToggle});

        const menu = element("div", "forge-assistant-menu");
        menu.hidden = true;
        menu.setAttribute("role", "menu");
        panel.appendChild(menu);
        this.nodes.menu = menu;

        // The accordion heading owns the ENTIRE conversation body. Collapsed,
        // all of it leaves the layout and the accessibility tree -- a body that
        // is only visually hidden is a body a screen reader still reads out.
        const heading = element("button", "forge-assistant-accordion");
        heading.type = "button";
        heading.setAttribute("aria-expanded", String(this.state.conversationExpanded));
        heading.setAttribute("aria-controls", "forge-assistant-conversation");
        heading.innerHTML = '<span class="forge-assistant-chevron" aria-hidden="true">'
            + "›</span>";
        heading.appendChild(document.createTextNode("Conversation"));
        panel.appendChild(heading);
        this.nodes.heading = heading;

        const body = element("div", "forge-assistant-conversation");
        body.id = "forge-assistant-conversation";
        panel.appendChild(body);
        this.nodes.body = body;

        // Which conversation this is. It was an empty div: the panel gave no
        // indication of which thread it was showing, which is fine until it is
        // showing a different one from the tab behind it -- and then it is the
        // only thing that would have said so.
        const selector = element("div", "forge-assistant-selector");
        const who = element("button", "forge-assistant-who", "\u2026");
        who.type = "button";
        who.setAttribute("aria-haspopup", "menu");
        who.setAttribute("aria-label", "Choose a conversation");
        selector.appendChild(who);
        body.appendChild(selector);
        Object.assign(this.nodes, {selector, who});

        const suppressed = element("p", "forge-assistant-suppressed");
        suppressed.hidden = true;
        body.appendChild(suppressed);
        this.nodes.suppressed = suppressed;

        const transcript = element("div", "forge-assistant-transcript");
        transcript.setAttribute("role", "log");
        transcript.setAttribute("aria-live", "off");
        body.appendChild(transcript);
        this.nodes.transcript = transcript;

        const jump = element("button", "forge-assistant-jump", "New response");
        jump.type = "button";
        jump.hidden = true;
        body.appendChild(jump);
        this.nodes.jump = jump;

        const status = element("p", "forge-assistant-status");
        status.setAttribute("role", "status");
        status.setAttribute("aria-live", "polite");
        body.appendChild(status);
        this.nodes.status = status;

        const composer = element("div", "forge-assistant-composer");
        const chip = element("div", "forge-assistant-chip");
        chip.hidden = true;
        const input = element("textarea", "forge-assistant-input");
        input.rows = 1;
        input.setAttribute("aria-label", "Message");
        input.placeholder = "Message…";
        const toolbar = element("div", "forge-assistant-toolbar");
        const attach = element("button", "forge-assistant-icon-button", "\u{1F4CE}");
        attach.type = "button";
        attach.setAttribute("aria-label", "Attach an image");
        const dictate = element("button", "forge-assistant-icon-button", "\u{1F3A4}");
        dictate.type = "button";
        dictate.setAttribute("aria-label", "Dictate a message");
        const readAloud = element("button", "forge-assistant-icon-button", "\u{1F50A}");
        readAloud.type = "button";
        readAloud.setAttribute("role", "switch");
        readAloud.setAttribute("aria-checked", "false");
        readAloud.setAttribute("aria-label", "Read replies aloud");
        const send = element("button", "forge-assistant-send", "Send");
        send.type = "button";
        const stop = element("button", "forge-assistant-stop", "Stop");
        stop.type = "button";
        stop.hidden = true;
        const filePicker = element("input");
        filePicker.type = "file";
        filePicker.accept = "image/png,image/jpeg,image/webp";
        filePicker.hidden = true;
        toolbar.appendChild(attach);
        toolbar.appendChild(dictate);
        toolbar.appendChild(readAloud);
        toolbar.appendChild(send);
        toolbar.appendChild(stop);
        composer.appendChild(chip);
        composer.appendChild(input);
        composer.appendChild(toolbar);
        composer.appendChild(filePicker);
        body.appendChild(composer);
        Object.assign(this.nodes, {composer, chip, input, toolbar, attach, dictate,
                                   readAloud, send, stop, filePicker});

        const handle = element("div", "forge-assistant-resize");
        handle.setAttribute("role", "separator");
        handle.setAttribute("tabindex", "0");
        handle.setAttribute("aria-label", "Resize the assistant");
        handle.setAttribute("aria-orientation", "vertical");
        panel.appendChild(handle);
        this.nodes.resize = handle;

        this.nodes.root.appendChild(panel);
        this.menuHandle = {close: () => this.closeMenu()};
        this.disposers.push(this.host.registerMenu(this.menuHandle));
        this.applyAccordion();
    };

    Shell.prototype.applyAccordion = function () {
        const open = this.state.conversationExpanded;
        this.nodes.heading.setAttribute("aria-expanded", String(open));
        // `hidden`, not a class: the body has to leave the accessibility tree,
        // not merely stop being painted.
        this.nodes.body.hidden = !open;
        this.nodes.panel.classList.toggle("forge-assistant-collapsed", !open);
    };

    // -- placing ----------------------------------------------------------- //

    Shell.prototype.place = function () {
        if (this.frame) return;
        this.frame = window.requestAnimationFrame(() => {
            this.frame = 0;
            this.placeNow();
        });
    };

    Shell.prototype.placeNow = function () {
        const root = this.nodes.root;
        if (!root) return;
        const open = this.state.panelOpen;
        const node = open ? this.nodes.panel : this.nodes.launcher;
        const view = viewport();
        const narrow = view.width < NARROW_VIEWPORT;
        root.setAttribute("data-anchor", this.anchor());
        root.classList.toggle("forge-assistant-narrow", narrow);
        if (open && narrow) {
            // The mobile sheet: full width, anchored to the half the anchor
            // names, and modal -- dialog semantics, contained focus, inert
            // background. A non-modal sheet on a phone is a sheet the page
            // scrolls behind.
            this.nodes.panel.classList.add("forge-assistant-sheet");
            this.nodes.panel.setAttribute("role", "dialog");
            this.nodes.panel.setAttribute("aria-modal", "true");
            node.style.left = "";
            node.style.top = "";
            node.style.width = "";
            return;
        }
        this.nodes.panel.classList.remove("forge-assistant-sheet");
        this.nodes.panel.removeAttribute("aria-modal");
        this.nodes.panel.setAttribute("role", "complementary");
        if (open) {
            const width = Math.min(Math.max(this.state.panelWidth || NOMINAL_WIDTH,
                                            MIN_WIDTH),
                                   Math.min(MAX_WIDTH, view.width - 32));
            node.style.width = width + "px";
        }
        const box = {width: node.offsetWidth || NOMINAL_WIDTH,
                     height: node.offsetHeight || 44};
        const at = anchorPoint(this.anchor(), box, view, insets());
        node.style.left = at.left + "px";
        node.style.top = at.top + "px";
    };

    // -- drag -------------------------------------------------------------- //
    //
    // Pointer Events, so mouse, touch and pen are one code path. The state
    // machine is idle -> pressed -> dragging -> committed | cancelled, and the
    // fiddly parts are all at the end of it: the commit is cleared *before* the
    // capture is released, so a late `lostpointercapture` cannot undo it, and
    // the click is suppressed only if the gesture actually exceeded the
    // threshold -- a press that moved two pixels is a click, not a drag.

    Shell.prototype.startDrag = function (event, node) {
        if (this.drag) return;
        if (event.button !== undefined && event.button !== 0) return;
        if (!event.isPrimary) return;
        const target = event.target;
        if (target && target !== node && target.closest
            && target.closest("button, a, input, textarea, select")
            && target.closest("button, a, input, textarea, select") !== node) return;
        const box = node.getBoundingClientRect();
        this.drag = {
            node,
            pointerId: event.pointerId,
            startX: event.clientX,
            startY: event.clientY,
            offsetX: event.clientX - box.left,
            offsetY: event.clientY - box.top,
            width: box.width,
            height: box.height,
            moved: false,
            anchor: this.anchor(),
            committed: false,
        };
        try {
            node.setPointerCapture(event.pointerId);
        } catch (error) { /* capture is an optimisation, not a requirement */ }
    };

    Shell.prototype.moveDrag = function (event) {
        const drag = this.drag;
        if (!drag || event.pointerId !== drag.pointerId) return;
        const dx = event.clientX - drag.startX;
        const dy = event.clientY - drag.startY;
        if (!drag.moved && Math.abs(dx) < DRAG_THRESHOLD && Math.abs(dy) < DRAG_THRESHOLD) {
            return;
        }
        drag.moved = true;
        this.nodes.root.classList.add("forge-assistant-dragging");
        const left = event.clientX - drag.offsetX;
        const top = event.clientY - drag.offsetY;
        drag.node.style.left = Math.round(left) + "px";
        drag.node.style.top = Math.round(top) + "px";
        const centre = {x: left + drag.width / 2, y: top + drag.height / 2};
        drag.anchor = nearestAnchor(centre, {width: drag.width, height: drag.height},
                                    viewport(), insets(), drag.anchor);
        this.preview(drag.anchor);
    };

    Shell.prototype.preview = function (anchor) {
        let ghost = this.nodes.ghost;
        if (!ghost) {
            ghost = this.nodes.ghost = element("div", "forge-assistant-ghost");
            ghost.setAttribute("aria-hidden", "true");
            this.nodes.root.appendChild(ghost);
        }
        const drag = this.drag;
        const at = anchorPoint(anchor, {width: drag.width, height: drag.height},
                               viewport(), insets());
        ghost.hidden = false;
        ghost.style.left = at.left + "px";
        ghost.style.top = at.top + "px";
        ghost.style.width = at.width + "px";
        ghost.style.height = at.height + "px";
    };

    Shell.prototype.endDrag = function (event, cancelled) {
        const drag = this.drag;
        if (!drag || (event && event.pointerId !== drag.pointerId)) return;
        // Cleared first. A `lostpointercapture` arriving after this must find
        // nothing to undo.
        this.drag = null;
        this.nodes.root.classList.remove("forge-assistant-dragging");
        if (this.nodes.ghost) this.nodes.ghost.hidden = true;
        if (!cancelled && drag.moved) {
            this.state.anchorOverride = drag.anchor;
            this.suppressClick = true;
            this._save();
        }
        drag.node.style.left = "";
        drag.node.style.top = "";
        try {
            drag.node.releasePointerCapture(drag.pointerId);
        } catch (error) { /* never held, or already released */ }
        this.placeNow();
    };

    // -- wiring ------------------------------------------------------------- //

    Shell.prototype.on = function (node, type, handler, options) {
        node.addEventListener(type, handler, options);
        this.disposers.push(() => node.removeEventListener(type, handler, options));
    };

    Shell.prototype.wire = function () {
        const nodes = this.nodes;

        [nodes.launcher, nodes.header].forEach((handle) => {
            this.on(handle, "pointerdown", (event) => {
                this.startDrag(event, this.state.panelOpen ? nodes.panel : nodes.launcher);
            });
        });
        this.on(window, "pointermove", (event) => this.moveDrag(event));
        this.on(window, "pointerup", (event) => this.endDrag(event, false));
        this.on(window, "pointercancel", (event) => this.endDrag(event, true));

        this.on(nodes.launcher, "click", (event) => {
            if (this.suppressClick) {
                // Only this gesture's click, and only once. A later keyboard
                // activation must not be swallowed by a drag that has finished.
                this.suppressClick = false;
                event.preventDefault();
                return;
            }
            this.open();
        });
        this.on(nodes.minimize, "click", () => this.close());
        this.on(nodes.heading, "click", () => {
            this.state.conversationExpanded = !this.state.conversationExpanded;
            this.applyAccordion();
            this._save();
            this.place();
        });
        this.on(nodes.who, "click", () => this.toggleMenu("threads"));
        this.on(nodes.picker, "click", () => this.toggleMenu("workspaces"));
        this.on(nodes.utilities, "click", () => this.toggleMenu("utilities"));
        this.on(nodes.focusToggle, "click", () => this.toggleFocus());

        this.on(nodes.input, "input", () => {
            this.store.setDraftText(nodes.input.value);
            this.grow();
        });
        this.on(nodes.input, "keydown", (event) => this.composerKey(event));
        this.on(nodes.input, "paste", (event) => this.paste(event));
        this.on(nodes.send, "click", () => this.send());
        this.on(nodes.stop, "click", () => this.stopReply());
        this.on(nodes.attach, "click", () => nodes.filePicker.click());
        this.on(nodes.filePicker, "change", () => {
            const file = nodes.filePicker.files && nodes.filePicker.files[0];
            if (file) this.stageFile(file);
            nodes.filePicker.value = "";
        });
        this.on(nodes.dictate, "click", () => this.dictate());
        this.on(nodes.readAloud, "click", () => this.toggleReadAloud());
        this.on(nodes.jump, "click", () => {
            this.following = true;
            this.toBottom();
            nodes.jump.hidden = true;
        });
        this.on(nodes.transcript, "scroll", () => {
            // Locked to the latest while the reader is at the end of it, and
            // left exactly where it is the moment they are not. The slack is
            // what makes "at the bottom" survive a font metric and a rounded
            // pixel; without it a transcript can be at the end and not know it.
            const distance = nodes.transcript.scrollHeight - nodes.transcript.scrollTop
                - nodes.transcript.clientHeight;
            this.following = distance <= BOTTOM_SLACK;
            if (this.following) nodes.jump.hidden = true;
            else if (this.settled) nodes.jump.hidden = false;
        });
        // A picture in a bubble arrives after the bubble does and grows it,
        // which moves the bottom out from under a reader who was at it. Caught
        // in the capture phase because `load` does not bubble.
        this.on(nodes.transcript, "load", () => {
            if (this.following) this.toBottom();
        }, true);
        this.on(nodes.resize, "keydown", (event) => this.resizeKey(event));
        this.on(nodes.resize, "pointerdown", (event) => this.startResize(event));

        this.on(window, "resize", () => this.place());
        if (window.visualViewport) {
            this.on(window.visualViewport, "resize", () => this.place());
            this.on(window.visualViewport, "scroll", () => this.place());
        }
        this.on(window, "orientationchange", () => this.place());
        this.on(document, "visibilitychange", () => {
            if (!document.hidden) this.store.reconcile();
        });
        this.on(window, "online", () => this.store.reconcile());
        this.on(window, "pageshow", () => this.store.reconcile());
        this.on(window, "pagehide", () => this.store.flushDrafts());
        this.on(document, "keydown", (event) => this.documentKey(event), true);

        this.disposers.push(this.store.subscribeState((view) => this.render(view)));
        this.disposers.push(this.host.subscribeNavigation((active) => {
            this.activeWorkspace = active;
            this.applySuppression();
            if (this.state.focusEnabled && active
                && active !== this.focus.activeWorkspace()) {
                const moved = this.focus.moveTo(active, this.host);
                if (!moved.ok) {
                    this.state.focusEnabled = false;
                    this.state.focusWorkspaceId = null;
                    this.nodes.focusToggle.setAttribute("aria-pressed", "false");
                    this.say(moved.reason, "warn");
                    this._save();
                } else if (moved.note) {
                    this.say(moved.note, "warn");
                }
            }
        }));
    };

    Shell.prototype.grow = function () {
        const input = this.nodes.input;
        input.style.height = "auto";
        const lineHeight = 20;
        const max = lineHeight * 6 + 12;
        input.style.height = Math.min(input.scrollHeight, max) + "px";
        input.style.overflowY = input.scrollHeight > max ? "auto" : "hidden";
    };

    // -- open, close, focus -------------------------------------------------- //

    // Exactly one of the launcher and the panel is on screen, and this is the
    // only function that says which. It is also the only one that has to know
    // that `hidden` alone is not enough here: the panel's own `display: flex`
    // is an author rule and beats the browser's `[hidden] { display: none }`,
    // which is why the stylesheet carries an explicit `[hidden]` rule scoped to
    // this root. Without it ✕ set an attribute and changed nothing visible.
    Shell.prototype.showOpen = function (open) {
        this.state.panelOpen = !!open;
        this.nodes.panel.hidden = !open;
        this.nodes.launcher.hidden = !!open;
        this.nodes.launcher.setAttribute("aria-expanded", String(!!open));
        if (!open) this.closeMenu();
        this._save();
        this.placeNow();
    };

    Shell.prototype.open = function () {
        this.showOpen(true);
        this.store.clearUnread();
        const input = this.nodes.input;
        if (input && !input.hidden) input.focus();
    };

    Shell.prototype.close = function () {
        this.showOpen(false);
        // Focus goes back to the control that opened it. A panel that closes
        // and leaves focus on the document is a panel a keyboard user has to
        // tab back to from the top of the page.
        this.nodes.launcher.focus();
    };

    Shell.prototype.toggle = function () {
        if (this.state.panelOpen) this.close();
        else this.open();
    };

    Shell.prototype.toggleFocus = function () {
        if (this.focus.isActive()) {
            this.focus.exit();
            this.state.focusEnabled = false;
            this.state.focusWorkspaceId = null;
            this.nodes.focusToggle.setAttribute("aria-pressed", "false");
            this._save();
            this.place();
            return;
        }
        const active = this.host.getActiveWorkspace();
        const found = this.focus.enter(active, this.host);
        if (!found.ok) {
            this.say(found.reason, "warn");
            return;
        }
        if (found.note) this.say(found.note, "warn");
        this.state.focusEnabled = true;
        this.state.focusWorkspaceId = active;
        this.nodes.focusToggle.setAttribute("aria-pressed", "true");
        this._save();
        this.place();
    };

    // -- menus --------------------------------------------------------------- //

    Shell.prototype.toggleMenu = function (which) {
        const menu = this.nodes.menu;
        if (!menu.hidden && menu.dataset.which === which) {
            this.closeMenu();
            return;
        }
        menu.innerHTML = "";
        menu.dataset.which = which;
        const items = which === "workspaces" ? this.workspaceItems()
            : (which === "threads" ? this.threadItems() : this.utilityItems());
        items.forEach((item) => menu.appendChild(item));
        menu.hidden = false;
        this.nodes.picker.setAttribute("aria-expanded",
                                       String(which === "workspaces"));
        this.nodes.utilities.setAttribute("aria-expanded",
                                          String(which === "utilities"));
        // The assistant's own menus are one-at-a-time with each other and with
        // nothing else: LLM Studio's seven sheets are exempt in both directions
        // (specification 15.2). What is registered is a closer for the *menu* --
        // registering the shell itself would make "close the menus" close the
        // panel.
        this.host.openOnly(this.menuHandle);
    };

    Shell.prototype.closeMenu = function () {
        const menu = this.nodes.menu;
        if (!menu) return false;
        const had = !menu.hidden;
        menu.hidden = true;
        menu.dataset.which = "";
        this.nodes.picker.setAttribute("aria-expanded", "false");
        this.nodes.utilities.setAttribute("aria-expanded", "false");
        return had;
    };

    Shell.prototype.workspaceItems = function () {
        const active = this.host.getActiveWorkspace();
        return this.host.listWorkspaces().map((workspace) => {
            const item = element("button", "forge-assistant-menu-item", workspace.label);
            item.type = "button";
            item.setAttribute("role", "menuitem");
            // The highlight follows the host's selection, not this click. A
            // switch made anywhere else -- a header button, another extension's
            // Send-to -- has to move it too.
            item.setAttribute("aria-current", String(workspace.id === active));
            item.disabled = !workspace.available;
            item.addEventListener("click", () => {
                item.classList.add("forge-assistant-pending");
                // The menu closes on the press, not on the confirmation. A
                // menu held open until the host confirms is a menu that stays
                // open for ever the moment the confirmation does not arrive --
                // and it did not, because the watchdog rejects after four
                // seconds on any page whose tab bar this adapter reads
                // differently from the way it expected to.
                //
                // Nothing is lost by closing early: the highlight follows the
                // host's own selection rather than this press, so a switch
                // that fails is visible in the picker next time it is opened,
                // and it is reported in the status line here and now.
                this.closeMenu();
                this.host.activateWorkspace(workspace.id).catch((error) => {
                    this.say(error.message || "That workspace did not open.", "warn");
                });
            });
            return item;
        });
    };

    Shell.prototype.threadItems = function () {
        const view = this.store.snapshot();
        const conversation = view.conversation || {};
        const threads = conversation.threads || [];
        if (!threads.length) {
            const empty = element("p", "forge-assistant-menu-empty",
                                  "No threads yet. Start one in LLM Studio.");
            return [empty];
        }
        return threads.map((thread) => {
            const item = element("button", "forge-assistant-menu-item", thread.title);
            item.type = "button";
            item.setAttribute("role", "menuitem");
            item.setAttribute("aria-current",
                              String(thread.thread_id === view.selection.thread));
            item.addEventListener("click", () => {
                this.closeMenu();
                this.store.select(view.selection.character, thread.thread_id);
            });
            return item;
        });
    };

    Shell.prototype.utilityItems = function () {
        return this.host.listUtilities().map((utility) => {
            const item = element("button", "forge-assistant-menu-item", utility.label);
            item.type = "button";
            item.setAttribute("role", "menuitem");
            if (utility.title) item.title = utility.title;
            item.disabled = !utility.enabled;
            item.addEventListener("click", () => {
                this.closeMenu();
                if (utility.kind === "unload") this.unload(utility.scope, utility.label);
                else if (typeof utility.invoke === "function") utility.invoke();
            });
            return item;
        });
    };

    // Giving a card back is slow enough to be worth saying so about, and it is
    // one of the few things here that half-succeed: the language model stops
    // and the checkpoint refuses because a generation is running. So the answer
    // is what was actually released, in the words the server used, rather than
    // a tick.
    Shell.prototype.unload = function (scope, label) {
        this.say(label + "\u2026", "info");
        this.store.request("/unload", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({scope}),
        }).then((found) => {
            this.say(found.message || "Released.", found.failed && found.failed.length
                ? "warn" : "info");
        }).catch((error) => {
            this.say((error && error.message) || "Nothing could be unloaded.", "warn");
        });
    };

    // -- keyboard ------------------------------------------------------------ //

    Shell.prototype.composerKey = function (event) {
        if (event.isComposing || event.keyCode === 229) return;   // IME
        if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
            event.preventDefault();
            this.send();
            return;
        }
        if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            this.send();
        }
    };

    Shell.prototype.documentKey = function (event) {
        if (event.key !== "Escape") return;
        const context = {
            composing: event.isComposing || event.keyCode === 229,
            nativeDialog: false,
            hostDialogOpen: NS.hostInDialog(document.activeElement),
            insideAssistant: this.nodes.root
                && this.nodes.root.contains(document.activeElement),
            assistantMenuOpen: this.nodes.menu && !this.nodes.menu.hidden,
            assistantEditing: !!this.editing,
            focusActive: this.focus.isActive(),
        };
        const action = NS.escapeOrder(event, context);
        if (!action) return;                  // the host's Escape still stands
        event.preventDefault();
        event.stopPropagation();
        if (action === "close-menu") this.closeMenu();
        else if (action === "cancel-edit") this.cancelEdit();
        else if (action === "exit-focus") this.toggleFocus();
    };

    Shell.prototype.resizeKey = function (event) {
        if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
        event.preventDefault();
        const step = event.key === "ArrowLeft" ? RESIZE_STEP : -RESIZE_STEP;
        const width = (this.state.panelWidth || NOMINAL_WIDTH) + step;
        this.state.panelWidth = Math.min(MAX_WIDTH, Math.max(MIN_WIDTH, width));
        this._save();
        this.placeNow();
    };

    Shell.prototype.startResize = function (event) {
        const start = event.clientX;
        const from = this.state.panelWidth || this.nodes.panel.offsetWidth || NOMINAL_WIDTH;
        const move = (moved) => {
            const width = from + (start - moved.clientX);
            this.state.panelWidth = Math.min(MAX_WIDTH, Math.max(MIN_WIDTH, width));
            this.placeNow();
        };
        const finish = () => {
            window.removeEventListener("pointermove", move);
            window.removeEventListener("pointerup", finish);
            this._save();
        };
        window.addEventListener("pointermove", move);
        window.addEventListener("pointerup", finish);
        event.preventDefault();
    };

    // -- attachments ---------------------------------------------------------- //

    Shell.prototype.paste = function (event) {
        const data = event.clipboardData;
        if (!data || !data.items) return;
        const images = Array.prototype.filter.call(data.items, (item) =>
            item.kind === "file" && /^image\/(png|jpeg|webp)$/.test(item.type));
        if (!images.length) return;          // ordinary text paste, untouched
        // Only the image half is intercepted. Text pasted alongside it still
        // lands in the box, because taking somebody's words away to keep their
        // picture is not a trade anybody asked for.
        event.preventDefault();
        const file = images[0].getAsFile();
        if (file) this.stageFile(file, images.length > 1);
    };

    Shell.prototype.stageFile = function (file, several) {
        const draft = this.store.draft();
        if (draft.attachment && draft.attachment.state !== "failed") {
            if (!window.confirm("Replace the picture already attached to this message?")) {
                return;
            }
        }
        // A local identity per attempt, so a late completion of an upload the
        // reader has since removed or replaced cannot put it back.
        const localId = NS.uuid();
        this.store.setAttachment({localId, state: "uploading", name: file.name || "Image",
                                  preview: URL.createObjectURL(file)});
        if (several) this.say("One image per message — the first one was used.", "info");
        this.store.upload(file).then((found) => {
            const current = this.store.draft().attachment;
            if (!current || current.localId !== localId) return;   // removed or replaced
            this.store.setAttachment({localId, state: "ready", token: found.token,
                                      name: found.name || file.name || "Image",
                                      preview: current.preview});
        }).catch((error) => {
            const current = this.store.draft().attachment;
            if (!current || current.localId !== localId) return;
            this.store.setAttachment({localId, state: "failed", name: file.name || "Image",
                                      preview: current.preview,
                                      reason: (error && error.message)
                                          || "That picture could not be used."});
        });
    };

    // -- sending --------------------------------------------------------------- //

    Shell.prototype.send = function () {
        const view = this.store.snapshot();
        if (!this.canSend(view)) return;
        this.store.setDraftText(this.nodes.input.value);
        this.store.submit().then((outcome) => {
            if (outcome && outcome.lost) {
                this.say("Checking whether your message was sent…", "warn");
                this.store.checkPending().then((found) => {
                    if (found && found.ok) this.store.refresh();
                    else this.say("Your message was not sent. Press Send to try again.",
                                  "warn");
                });
                return;
            }
            if (outcome && !outcome.ok) {
                this.say((outcome.error && outcome.error.message)
                    || "That could not be sent.", "warn");
            }
        });
    };

    Shell.prototype.canSend = function (view) {
        // Deliberately not "only while the event stream is open". A command is
        // an ordinary request and the server validates it whatever the stream
        // is doing; refusing to send because live updates have not started is
        // refusing to do the thing that works because the thing that watches it
        // does not.
        if (!view.ready || view.error) return false;
        if (!view.conversation) return false;
        if (view.operation && !view.operation.terminal) return false;
        const draft = view.draft;
        if (draft.attachment && draft.attachment.state !== "ready") return false;
        return !!(draft.text && draft.text.trim()) || !!draft.attachment;
    };

    Shell.prototype.stopReply = function () {
        const view = this.store.snapshot();
        if (view.operation) this.store.stop(view.operation.id);
        if (NS.speech && typeof NS.speech.stopPlayback === "function") {
            NS.speech.stopPlayback({origin: "any"});
        }
    };

    // -- voice ------------------------------------------------------------------ //

    Shell.prototype.dictate = function () {
        if (!NS.speech || typeof NS.speech.startDictation !== "function") {
            this.say("Dictation is not available on this page.", "warn");
            return;
        }
        const view = this.store.snapshot();
        const key = NS.conversationKey(view.selection.character, view.selection.thread);
        if (this.dictating) {
            NS.speech.stopDictation({commit: true});
            this.dictating = false;
            this.nodes.dictate.setAttribute("aria-pressed", "false");
            return;
        }
        this.dictating = true;
        this.nodes.dictate.setAttribute("aria-pressed", "true");
        // Review mode, always, whatever the host's auto-send preference says.
        // A flyout that sent what it heard would be a flyout that sent a
        // half-heard sentence, and the preference is not this control's to
        // change.
        NS.speech.startDictation({conversationKey: key,
                                  draftVersion: view.draft.draftVersion,
                                  mode: "review"});
    };

    Shell.prototype.toggleReadAloud = function () {
        const on = this.nodes.readAloud.getAttribute("aria-checked") === "true";
        this.nodes.readAloud.setAttribute("aria-checked", String(!on));
        if (NS.speech && typeof NS.speech.setAutomaticReadAloud === "function") {
            NS.speech.setAutomaticReadAloud(!on);
        }
    };

    // -- rendering ---------------------------------------------------------------- //

    Shell.prototype.say = function (text, kind) {
        const status = this.nodes.status;
        if (!status) return;
        status.textContent = text;
        status.dataset.kind = kind || "info";
    };

    Shell.prototype.applySuppression = function () {
        const view = this.store.snapshot();
        const studio = this.activeWorkspace
            && /llm[_-]?studio/i.test(String(this.activeWorkspace));
        const suppress = !!studio && view.mode === "chat" && !this.state.showAnyway;
        this.nodes.suppressed.hidden = !suppress;
        if (suppress && !this.nodes.suppressed.firstChild) {
            this.nodes.suppressed.appendChild(
                document.createTextNode("You're viewing this conversation in LLM Studio. "));
            const anyway = element("button", "forge-assistant-link", "Show anyway");
            anyway.type = "button";
            anyway.addEventListener("click", () => {
                // Visit-scoped: leaving the mode and coming back suppresses it
                // again, because the reason it was suppressed has not changed.
                this.state.showAnyway = true;
                this.applySuppression();
            });
            this.nodes.suppressed.appendChild(anyway);
        }
        [this.nodes.selector, this.nodes.transcript, this.nodes.composer,
         this.nodes.status].forEach((node) => {
            if (node) node.hidden = suppress;
        });
        // Suppression hides; it never unsubscribes, never cancels a reply and
        // never stops playback. What is behind it is still happening.
    };

    Shell.prototype.render = function (view) {
        if (!this.nodes.root) return;
        const nodes = this.nodes;
        nodes.unread.hidden = !view.unreadTotal;
        nodes.unread.textContent = String(view.unreadTotal || "");

        if (nodes.input.value !== view.draft.text && document.activeElement !== nodes.input) {
            nodes.input.value = view.draft.text || "";
            this.grow();
        }
        this.renderSelector(view);
        this.renderChip(view.draft.attachment);
        this.renderTranscript(view);
        this.renderStatus(view);
        nodes.send.disabled = !this.canSend(view);
        const busy = !!(view.operation && !view.operation.terminal);
        nodes.stop.hidden = !(busy || view.speech.playing);
        this.applySuppression();
    };

    Shell.prototype.renderSelector = function (view) {
        const conversation = view.conversation && view.conversation.conversation;
        const character = (conversation && conversation.character)
            || view.selection.character;
        const title = conversation && conversation.title;
        this.nodes.who.textContent = title
            ? (character ? character + " \u00b7 " + title : title)
            : (character || "Choose a conversation");
    };

    Shell.prototype.renderChip = function (attachment) {
        const chip = this.nodes.chip;
        chip.hidden = !attachment;
        if (!attachment) {
            chip.innerHTML = "";
            return;
        }
        if (chip.dataset.localId === attachment.localId
            && chip.dataset.state === attachment.state) return;
        chip.dataset.localId = attachment.localId;
        chip.dataset.state = attachment.state;
        chip.innerHTML = "";
        if (attachment.preview) {
            const image = element("img", "forge-assistant-thumb");
            image.src = attachment.preview;
            image.alt = attachment.name || "Attached image";
            chip.appendChild(image);
        }
        const states = {uploading: "Uploading…", processing: "Preparing…",
                        ready: "Ready", failed: attachment.reason || "Failed"};
        chip.appendChild(element("span", "forge-assistant-chip-state",
                                 states[attachment.state] || attachment.state));
        const remove = element("button", "forge-assistant-icon-button", "✕");
        remove.type = "button";
        remove.setAttribute("aria-label", "Remove the attached image");
        remove.addEventListener("click", () => {
            const current = this.store.draft().attachment;
            if (current && current.token) {
                fetch(NS.route("/attachments/" + encodeURIComponent(current.token)),
                      {method: "DELETE", credentials: "same-origin",
                       headers: this.store.headers()}).catch(() => undefined);
            }
            this.store.setAttachment(null);
        });
        chip.appendChild(remove);
    };

    Shell.prototype.renderTranscript = function (view) {
        const transcript = this.nodes.transcript;
        const conversation = view.conversation;
        // A conversation opens at its latest message, which means the arrival
        // of a *different* conversation resets the reader's place rather than
        // inheriting it. Without this, a thread left scrolled halfway up put
        // the next thread halfway up too -- at an offset measured against a
        // message that is no longer on the page.
        const epoch = view.selection.epoch || "";
        if (epoch !== this.shownEpoch) {
            this.shownEpoch = epoch;
            this.following = true;
            this.lastRendered = "";
            this.settled = false;
        }
        if (!conversation || !conversation.messages) {
            transcript.innerHTML = "";
            this.lastRendered = "";
            this.nodes.jump.hidden = true;
            return;
        }
        // The reader's place, captured before anything moves. Restoring a pixel
        // offset against the message that was on top is the only way a reply
        // arriving above the fold does not shove what somebody is reading.
        const wasFollowing = this.following;
        const first = transcript.querySelector("[data-index]");
        const offset = first ? first.getBoundingClientRect().top : 0;

        const rows = conversation.messages.slice();
        const operation = view.operation;
        if (operation && !operation.terminal && operation.text) {
            const at = operation.targetIndex;
            if (at !== undefined && at !== null && at < rows.length) {
                rows[at] = Object.assign({}, rows[at], {text: operation.text,
                                                        provisional: true});
            } else {
                rows.push({index: rows.length, role: "assistant", text: operation.text,
                           versions: [operation.text], active: 0, provisional: true});
            }
        }

        // What a skipped redraw has to agree on. The revision is in here
        // because a thread that moved is not the thread that was drawn: the
        // text can be identical and the actions under it still stale, which is
        // a redraw skipped for a transcript that needed one.
        //
        // The thread itself is not, because it cannot be: arriving at another
        // conversation clears `lastRendered` above, so there is nothing for a
        // second thread's fingerprint to collide with.
        const revision = conversation.conversation && conversation.conversation.revision;
        const fingerprint = JSON.stringify([revision,
            rows.map((row) => [row.index, row.role, row.text, row.active,
                               (row.versions || []).length, !!row.provisional])]);
        if (fingerprint === this.lastRendered) {
            this.pin(wasFollowing);
            return;
        }
        // Keyed so that only the message that changed is rewritten. A full
        // redraw per token would restart every animation and drop the
        // selection in the bubble somebody is reading.
        //
        // The thread and its revision are part of the key, and that is not
        // caution. `updateBubble` refreshes the text of a reused node and
        // nothing else -- the action buttons under it closed over the row and
        // the revision they were *built* with, and they send those. Reuse a
        // bubble across a thread change and every action under it is aimed at
        // the conversation that is no longer on screen; reuse one across a
        // revision change and every action is refused as stale. A reply
        // arriving token by token does not move the revision, so the redraw
        // this exists to avoid is still avoided.
        const stamp = epoch + ":" + revision + ":";
        const existing = new Map();
        Array.prototype.forEach.call(transcript.children, (node) => {
            existing.set(node.dataset.key, node);
        });
        const wanted = [];
        rows.forEach((row) => {
            const key = stamp + row.index + ":" + (row.active || 0)
                + ":" + (row.provisional ? "p" : "s");
            let node = existing.get(key);
            if (!node) node = this.bubble(row, view);
            else this.updateBubble(node, row);
            node.dataset.key = key;
            wanted.push(node);
        });
        transcript.innerHTML = "";
        wanted.forEach((node) => transcript.appendChild(node));
        this.lastRendered = fingerprint;

        if (wasFollowing) {
            this.toBottom();
        } else {
            const now = transcript.querySelector("[data-index]");
            if (now) transcript.scrollTop += now.getBoundingClientRect().top - offset;
        }
        this.showJump(view);
    };

    // Put the latest message on screen, now and again once the browser has
    // finished with it.
    //
    // Once is not enough, and both of the reasons are ordinary. A panel that
    // has just opened -- or whose conversation section was collapsed -- has a
    // transcript of no height at the moment the messages go into it, so
    // `scrollHeight` is the height of nothing and the assignment does nothing.
    // And a bubble carrying a picture grows when the picture arrives, which is
    // after this returns, taking the bottom with it.
    Shell.prototype.toBottom = function () {
        const transcript = this.nodes.transcript;
        if (!transcript) return;
        transcript.scrollTop = transcript.scrollHeight;
        if (this.settling) return;
        this.settling = true;
        window.requestAnimationFrame(() => {
            this.settling = false;
            if (!this.following || !this.nodes.transcript) return;
            this.nodes.transcript.scrollTop = this.nodes.transcript.scrollHeight;
            this.settled = true;
        });
    };

    // Called on a redraw that changed nothing: the content is the same, but the
    // box it is in may not be -- the panel was resized, the conversation
    // section was expanded, a picture finished loading. Following means the
    // latest message, whatever the box did.
    Shell.prototype.pin = function (following) {
        if (following) this.toBottom();
    };

    Shell.prototype.showJump = function (view) {
        const jump = this.nodes.jump;
        if (!jump) return;
        jump.hidden = !!this.following;
        // Honest about which it is. "New response" on a button that has been
        // sitting there since before the reply started is a button nobody
        // believes the second time.
        jump.textContent = view && view.unread ? "New response" : "Jump to latest";
    };

    Shell.prototype.bubble = function (row, view) {
        const node = element("article", "forge-assistant-bubble forge-assistant-"
            + (row.role === "assistant" ? "reply" : "mine"));
        node.dataset.index = String(row.index);
        node.style.maxWidth = this.settings.bubbleWidth + "%";
        const text = element("div", "forge-assistant-text");
        text.innerHTML = renderMarkdown(row.text);
        node.appendChild(text);
        if (row.provisional) node.classList.add("forge-assistant-provisional");
        if (row.attachment) {
            const figure = element("figure", "forge-assistant-attachment");
            if (row.attachment.url && row.attachment.available) {
                const image = element("img");
                image.src = NS.basePath() + row.attachment.url;
                image.alt = row.attachment.name || "Attached image";
                figure.appendChild(image);
            }
            figure.appendChild(element("figcaption", "",
                                       row.attachment.available
                                           ? (row.attachment.name || "Image")
                                           : "This picture is no longer on disk."));
            node.appendChild(figure);
        }
        node.appendChild(this.actions(row, view));
        return node;
    };

    Shell.prototype.updateBubble = function (node, row) {
        const text = node.querySelector(".forge-assistant-text");
        if (text) text.innerHTML = renderMarkdown(row.text);
        node.classList.toggle("forge-assistant-provisional", !!row.provisional);
    };

    // Every action, reachable without hover, with a visible keyboard-focusable
    // control per message. Each carries the (index, version, revision) captured
    // at the moment it was drawn -- a stale one is refused by the server rather
    // than quietly reinterpreted as the current last message.
    Shell.prototype.actions = function (row, view) {
        const bar = element("div", "forge-assistant-actions");
        const revision = view.conversation && view.conversation.conversation
            && view.conversation.conversation.revision;
        const last = view.conversation
            && row.index === view.conversation.messages.length - 1;
        const reply = row.role === "assistant";
        const add = (label, action, payload) => {
            const button = element("button", "forge-assistant-action", label);
            button.type = "button";
            button.addEventListener("click", () => this.act(action, row, revision, payload));
            bar.appendChild(button);
        };
        add("Edit", "edit_message");
        if (reply) add(last ? "Regenerate" : "Regenerate in a branch", "regenerate");
        if (reply && row.text) add(last ? "Continue" : "Continue in a branch", "continue");
        if (!reply) add("Send again from here", "resend_from_user");
        add("Branch from here", "branch");
        if ((row.versions || []).length > 1) {
            add("◀", "select_version", {version: Math.max(0, (row.active || 0) - 1)});
            bar.appendChild(element("span", "forge-assistant-pager",
                                    (row.active + 1) + "/" + row.versions.length));
            add("▶", "select_version",
                {version: Math.min(row.versions.length - 1, (row.active || 0) + 1)});
            add("Delete this version", "drop_version");
        }
        add("Delete message", "delete_message");
        add("Delete from here", "delete_from",
            {confirm_count: view.conversation.messages.length - row.index});
        if (reply) add("Listen", "listen");
        return bar;
    };

    Shell.prototype.act = function (action, row, revision, payload) {
        if (action === "listen") {
            if (NS.speech && typeof NS.speech.listen === "function") {
                NS.speech.listen({index: row.index, version: row.active});
            }
            return;
        }
        if (action === "edit_message") {
            this.startEdit(row, revision);
            return;
        }
        if (action === "delete_from" && !window.confirm(
            "Delete " + (payload.confirm_count) + " message(s) from here?")) return;
        const envelope = this.store.envelope(action, {
            target: {index: row.index, version: row.active},
            expected_revision: revision === undefined || revision === null ? null
                : (typeof revision === "number" ? {kind: "revision", value: revision}
                    : {kind: "legacy", fingerprint: String(revision)}),
        });
        const wanted = Object.assign({}, payload || {});
        if (action === "select_version") {
            envelope.target.version = wanted.version;
            delete wanted.version;
        }
        envelope.payload = wanted;
        this.store.send(envelope).then((outcome) => {
            if (outcome && !outcome.ok) {
                this.say((outcome.error && outcome.error.message)
                    || "That could not be done.", "warn");
            }
        });
    };

    Shell.prototype.startEdit = function (row, revision) {
        this.editing = {index: row.index, version: row.active, revision,
                        text: row.text};
        const buffer = window.prompt("Edit this message", row.text);
        if (buffer === null) {
            this.editing = null;
            return;
        }
        const envelope = this.store.envelope("edit_message", {
            target: {index: row.index, version: row.active},
            expected_revision: typeof revision === "number"
                ? {kind: "revision", value: revision}
                : (revision ? {kind: "legacy", fingerprint: String(revision)} : null),
        });
        envelope.payload = {text: buffer, image_action: "keep"};
        this.editing = null;
        this.store.send(envelope).then((outcome) => {
            if (outcome && !outcome.ok) {
                this.say((outcome.error && outcome.error.message)
                    || "That edit was refused.", "warn");
            }
        });
    };

    Shell.prototype.cancelEdit = function () {
        this.editing = null;
    };

    Shell.prototype.renderStatus = function (view) {
        if (!view.ready) {
            this.say(view.error || "Connecting…", view.error ? "warn" : "info");
            return;
        }
        if (view.error) {
            this.say(view.error, "warn");
            return;
        }
        if (!view.connected) {
            // "Reconnecting" is only true the second time. Saying it on a feed
            // that has never opened describes a drop that never happened, and
            // it was the first thing on screen when the panel came up with
            // nothing in it -- a sentence pointing at the wrong problem.
            if (!view.everConnected) {
                this.say("Connecting…", "info");
                return;
            }
            this.say(view.polling
                ? "Reconnecting… Following along by polling. Your draft is safe."
                : "Reconnecting… Your draft is safe.", "warn");
            return;
        }
        if (view.operation && !view.operation.terminal) {
            this.say(view.operation.status || "Generating…", "info");
            return;
        }
        // Connected, with nothing chosen to show. Worth saying: an empty
        // transcript under the word "Ready" reads like a conversation that
        // lost its messages rather than one that was never picked.
        if (!view.selection.thread) {
            this.say(view.characters && view.characters.length
                ? "Pick a conversation to begin."
                : "No conversations yet. Start one in LLM Studio.", "info");
            return;
        }
        this.say("Ready.", "info");
    };

    Shell.prototype.dispose = function () {
        this.disposers.forEach((off) => {
            try {
                off();
            } catch (error) { /* already gone */ }
        });
        this.disposers = [];
        if (this.nodes.root && this.nodes.root.parentElement) {
            this.nodes.root.parentElement.removeChild(this.nodes.root);
        }
        this.nodes = {};
    };

    // -- mounting --------------------------------------------------------------- //

    function settingsFrom(document_) {
        // Read off the Settings page's own controls where they exist, so the
        // panel follows them without a second copy of the values. Absent
        // controls leave the defaults, which is what an installation that has
        // never opened Settings has.
        const read = (id, fallback) => {
            const node = document_.getElementById(id);
            if (!node) return fallback;
            const input = node.querySelector ? node.querySelector("input, textarea, select")
                : null;
            const value = (input || node).value;
            return value === undefined || value === "" ? fallback : value;
        };
        return {
            label: read("setting_forge_assistant_label", "Forge Assistant"),
            appearance: read("setting_forge_assistant_appearance", "text"),
            defaultAnchor: read("setting_forge_assistant_default_anchor", DEFAULT_ANCHOR),
            bubbleWidth: Number(read("setting_forge_assistant_bubble_width", 75)) || 75,
        };
    }

    function enabled() {
        const node = document.getElementById("setting_forge_assistant_enabled");
        if (!node) return true;
        const input = node.querySelector("input[type=checkbox]");
        return !input || input.checked;
    }

    let shell = null;

    function mount() {
        if (shell) return;
        if (!enabled()) return;
        // Forge loads javascript/*.js alphabetically, so this file is parsed
        // before the store, the host adapter and the focus registry. Mounting
        // happens on `onUiLoaded`, by which time all four are there -- but the
        // check is here rather than assumed, because a file that failed to
        // parse would otherwise take the panel down with a TypeError instead
        // of simply not drawing it.
        if (typeof NS.store !== "function" || typeof NS.host !== "function"
            || !NS.focus) {
            console.warn("Forge Assistant: its other scripts did not load; "
                         + "the panel is not drawn and nothing else is affected");
            return;
        }
        const store = NS.store();
        const host = NS.host();
        const focus = NS.focus;
        shell = new Shell(store, host, focus);
        Object.assign(shell.settings, settingsFrom(document));
        try {
            if (!shell.mount()) {
                shell = null;
                return;
            }
        } catch (error) {
            console.error("Forge Assistant: the panel could not be built", error);
            // Leave nothing behind: no backdrop, no inert siblings, no root.
            try {
                focus.exit();
            } catch (nested) { /* nothing was entered */ }
            const orphan = document.getElementById(ROOT_ID);
            if (orphan && orphan.parentElement) orphan.parentElement.removeChild(orphan);
            shell = null;
            return;
        }
        NS.shell = shell;
        store.start();
    }

    NS.anchorPoint = anchorPoint;
    NS.nearestAnchor = nearestAnchor;
    NS.renderMarkdown = renderMarkdown;
    NS.safeUrl = safeUrl;
    NS.ANCHORS = ANCHORS;
    NS.Shell = Shell;
    NS.mount = mount;
    NS.viewport = viewport;

    if (typeof onUiLoaded === "function") {
        onUiLoaded(() => {
            try {
                mount();
            } catch (error) {
                console.error("Forge Assistant: could not start", error);
            }
        });
    } else if (document.readyState !== "loading") {
        mount();
    } else {
        document.addEventListener("DOMContentLoaded", mount);
    }
})();
