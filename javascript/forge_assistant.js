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

    // How often `recover` may run. A handler that fails on every pointermove
    // must not turn into a forced layout on every pointermove as well.
    const RECOVER_COOLDOWN = 1000;

    // Two presses on the launcher that land on something else, this close
    // together, are somebody trying to use it and being stopped. One is a
    // press near it.
    const COVERED_WINDOW = 4000;
    // Quick actions on the launcher: two presses go back to the workspace
    // before this one, three turn focus on or off. A press belongs to the
    // same gesture when it lands within this long of the one before it --
    // a double-click's pace, so two unhurried presses stay two presses and a
    // fast pair is never missed. A single press waits this long to be sure it
    // is single before the panel opens.
    const TAP_WINDOW = 300;
    // How long a request for the browser's full screen is waited on before
    // the next press may ask again. Browsers answer within a frame or two;
    // this is for one that never answers at all.
    const SCREEN_WAIT = 5000;
    const MIN_WIDTH = 320;
    const MAX_WIDTH = 640;
    const NOMINAL_WIDTH = 360;
    const RESIZE_STEP = 16;
    const BOTTOM_SLACK = 100;

    // A menu's gap from the row that opened it, its margin from the edge of
    // the window, and the least room worth opening into rather than refusing.
    const MENU_GAP = 4;
    const MENU_EDGE = 8;
    const MENU_FLOOR = 120;

    // How long the answer to a press holds the status line against "Ready.".
    // See `tell`.
    const TOLD_FOR = 6000;

    // What another extension on this page says when a dialog of its own takes
    // over, and gives way again. Mini Paint NEO publishes it for its Send to
    // WanGP popup; see `yieldTo`.
    const FOREIGN_OVERLAY = "minipaint:overlay";

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

    function clamp01(value) {
        if (!isFinite(value)) return 0;
        return value < 0 ? 0 : (value > 1 ? 1 : value);
    }

    // Free float, and why it remembers a fraction rather than a pixel.
    //
    // The six anchors exist so that a window resized between sessions cannot
    // leave the panel off-screen: what is stored is which corner, and a corner
    // is a corner at any size. Free float has to store a position, so it
    // stores how far across the *available travel* the panel was -- 0 against
    // one edge, 1 against the other. That maps onto any later viewport and can
    // strand the panel no more than an anchor can, which is the property the
    // anchors were protecting and the one thing a remembered pixel would lose.
    function fractionOf(box, view) {
        return {x: clamp01((box.left - view.left) / Math.max(1, view.width - box.width)),
                y: clamp01((box.top - view.top) / Math.max(1, view.height - box.height))};
    }

    // The same clamping as `anchorPoint`: inside the safe-area insets, inside
    // the gap, and never larger than the window.
    function floatPoint(at, box, view, inset) {
        const gap = gapFor(view);
        const pad = inset || {top: 0, right: 0, bottom: 0, left: 0};
        const width = Math.min(box.width, view.width - pad.left - pad.right - gap * 2);
        const height = Math.min(box.height, view.height - pad.top - pad.bottom - gap * 2);
        const minX = view.left + pad.left + gap;
        const minY = view.top + pad.top + gap;
        const travelX = Math.max(0, (view.width - pad.right - gap - width)
            - (pad.left + gap));
        const travelY = Math.max(0, (view.height - pad.bottom - gap - height)
            - (pad.top + gap));
        return {left: Math.round(minX + clamp01(at && at.x) * travelX),
                top: Math.round(minY + clamp01(at && at.y) * travelY),
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

    // A covering element, named well enough to find it in the inspector.
    function describe(node) {
        if (!node || !node.tagName) return String(node);
        let text = node.tagName.toLowerCase();
        if (node.id) text += "#" + node.id;
        if (node.className && typeof node.className === "string") {
            text += "." + node.className.trim().split(/\s+/).join(".");
        }
        return text;
    }

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
            freeFloat: false,
            floatAt: null,
            panelWidth: null,
            panelOpen: false,
            conversationExpanded: true,
            focusEnabled: false,
            focusWorkspaceId: null,
            showAnyway: false,
            autoAttach: false,
            sendToGenerate: false,
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
        this._restoreFloat();
        this._restoreAutoAttach();
        this._restoreSendToGenerate();
        // Which picture Auto Attach last sent, per conversation. See
        // `autoAttachable`.
        this.autoSent = new Map();
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

    // Free float is remembered across sessions, and the rest of the layout is
    // not, on purpose.
    //
    // A panel left open, a width dragged wider and a corner chosen are this
    // tab's business, which is what `sessionStorage` is for. Whether the panel
    // snaps to corners *at all* is a preference about how the thing works: it
    // is answered once and it should stay answered, so it lives in
    // `localStorage` under a key of its own. Keeping it out of the session
    // payload also means it survives that payload being rejected by a schema
    // bump.
    Shell.prototype._floatKey = function () {
        return "forge-assistant-float:" + (NS.basePath() || "/");
    };

    Shell.prototype._restoreFloat = function () {
        let raw = null;
        try {
            raw = window.localStorage.getItem(this._floatKey());
        } catch (error) {
            raw = null;
        }
        if (!raw) return;
        try {
            const found = JSON.parse(raw);
            if (!found) return;
            if (typeof found.freeFloat === "boolean") {
                this.state.freeFloat = found.freeFloat;
            }
            const at = found.floatAt;
            if (at && typeof at.x === "number" && typeof at.y === "number") {
                this.state.floatAt = {x: clamp01(at.x), y: clamp01(at.y)};
            }
        } catch (error) {
            // A half-read preference is the default one.
        }
    };

    Shell.prototype._saveFloat = function () {
        try {
            window.localStorage.setItem(this._floatKey(), JSON.stringify(
                {freeFloat: this.state.freeFloat, floatAt: this.state.floatAt}));
        } catch (error) { /* memory only; the panel still works */ }
    };

    // Auto Attach is a preference about how the composer works, answered once
    // -- remembered across sessions like Free Float, under a key of its own.
    Shell.prototype._autoAttachKey = function () {
        return "forge-assistant-auto-attach:" + (NS.basePath() || "/");
    };

    Shell.prototype._restoreAutoAttach = function () {
        try {
            this.state.autoAttach = window.localStorage.getItem(this._autoAttachKey()) === "on";
        } catch (error) {
            this.state.autoAttach = false;
        }
    };

    Shell.prototype._saveAutoAttach = function () {
        try {
            window.localStorage.setItem(this._autoAttachKey(),
                                        this.state.autoAttach ? "on" : "off");
        } catch (error) { /* memory only; the mode still works */ }
    };

    // Send to Generate, the same way: a preference about what one button does,
    // remembered in this browser under a key of its own.
    Shell.prototype._sendToGenerateKey = function () {
        return "forge-assistant-send-to-generate:" + (NS.basePath() || "/");
    };

    Shell.prototype._restoreSendToGenerate = function () {
        try {
            this.state.sendToGenerate =
                window.localStorage.getItem(this._sendToGenerateKey()) === "on";
        } catch (error) {
            this.state.sendToGenerate = false;
        }
    };

    Shell.prototype._saveSendToGenerate = function () {
        try {
            window.localStorage.setItem(this._sendToGenerateKey(),
                                        this.state.sendToGenerate ? "on" : "off");
        } catch (error) { /* memory only; the switch still works */ }
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
        this.applyLook();
    };

    /** The launcher's customized look, over what Settings says about it.
     *
     * `look` is a draft while the customize dialog is open, and the saved one
     * otherwise. Without the look module this does nothing and the launcher
     * is exactly what Settings makes it. A new look can change the launcher's
     * size, so it is placed again.
     */
    Shell.prototype.applyLook = function (look) {
        if (!NS.look || !this.nodes.launcher) return null;
        const found = NS.look.apply(this.nodes.launcher, look || NS.look.load(),
                                    {label: this.settings.label,
                                     appearance: this.settings.appearance},
                                    {label: this.nodes.launcherLabel,
                                     icon: this.nodes.launcherIcon});
        if (found) {
            // The accessible name follows the title that is actually drawn.
            this.nodes.launcher.setAttribute("aria-label", found.label);
            this.nodes.launcher.title = found.label;
        }
        this.place();
        return found;
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

        // One row, and everything is in it.
        //
        // It was two: a title row carrying the panel's name and the ✕, and a
        // nav row under it carrying Workspace, Focus and ⋯. The name was the
        // one thing on the screen nobody needed telling -- it is the only
        // floating panel on the page -- and two rows of chrome above a
        // collapsed conversation was most of the panel. The panel keeps its
        // name where a name is actually used: `aria-label`, set above, which
        // is what a screen reader announces when focus enters it.
        //
        // The space between the menus and the ✕ is the drag handle. It has to
        // be explicit now: `startDrag` ignores a press that lands on a
        // control, so with the row full of controls there would otherwise be
        // nothing left to take hold of.
        const header = element("header", "forge-assistant-header");
        const picker = element("button", "forge-assistant-nav-button", "Workspace");
        picker.type = "button";
        picker.setAttribute("aria-haspopup", "menu");
        picker.setAttribute("aria-expanded", "false");
        const focusToggle = element("button", "forge-assistant-nav-button", "Focus");
        focusToggle.type = "button";
        focusToggle.setAttribute("aria-pressed", "false");
        const utilities = element("button", "forge-assistant-nav-button", "⋯");
        utilities.type = "button";
        utilities.setAttribute("aria-haspopup", "menu");
        utilities.setAttribute("aria-expanded", "false");
        utilities.setAttribute("aria-label", "More actions");
        const grip = element("div", "forge-assistant-grip");
        grip.setAttribute("aria-hidden", "true");
        const minimize = element("button", "forge-assistant-icon-button");
        minimize.type = "button";
        minimize.setAttribute("aria-label", "Minimize the assistant");
        minimize.title = "Minimize";
        minimize.textContent = "✕";
        header.appendChild(picker);
        header.appendChild(focusToggle);
        header.appendChild(utilities);
        header.appendChild(grip);
        header.appendChild(minimize);
        panel.appendChild(header);
        Object.assign(this.nodes,
                      {header, minimize, picker, utilities, focusToggle, grip});

        const menu = element("div", "forge-assistant-menu");
        menu.hidden = true;
        menu.setAttribute("role", "menu");
        panel.appendChild(menu);
        this.nodes.menu = menu;

        // The workspaces, as a row, for when the conversation is collapsed.
        //
        // Collapsed, this panel was a header and a closed accordion: four
        // controls, none of which did anything without opening something else
        // first. The one thing it is well placed to be in that state is a tab
        // bar -- so that is what it is. Every workspace side by side, one
        // press each, and the panel puts itself away afterwards because a
        // switch is the whole of what somebody opened it for.
        //
        // Hidden rather than absent when the conversation is open: the panel
        // has the workspace menu in its header there, and two ways to change
        // workspace on one screen is one too many.
        const workspaces = element("div", "forge-assistant-workspaces");
        workspaces.setAttribute("role", "group");
        workspaces.setAttribute("aria-label", "Workspaces");
        panel.appendChild(workspaces);
        this.nodes.workspaces = workspaces;

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
        // Editing a message happens in this box, not in a dialog of the
        // browser's: the strip says which message and is the way out. See
        // `startEdit`.
        const editBar = element("div", "forge-assistant-edit-bar");
        editBar.hidden = true;
        const editLabel = element("span", "forge-assistant-edit-label",
                                  "Editing your message");
        const editCancel = element("button", "forge-assistant-edit-cancel", "Cancel");
        editCancel.type = "button";
        editCancel.title = "Keep the message as it was";
        editBar.appendChild(editLabel);
        editBar.appendChild(editCancel);
        const chip = element("div", "forge-assistant-chip");
        chip.hidden = true;
        const input = element("textarea", "forge-assistant-input");
        input.rows = 1;
        input.setAttribute("aria-label", "Message");
        input.placeholder = "Message…";
        const toolbar = element("div", "forge-assistant-toolbar");
        const attach = element("button", "forge-assistant-icon-button forge-assistant-attach",
                               "\u{1F4CE}");
        attach.type = "button";
        attach.setAttribute("aria-label", "Attach an image");
        const dictate = element("button", "forge-assistant-icon-button", "\u{1F3A4}");
        dictate.type = "button";
        dictate.setAttribute("aria-label", "Dictate a message");
        // Drawn from the setting by `renderReadAloud`, never assumed: see
        // there for why the two states look nothing alike.
        const readAloud = element("button",
                                  "forge-assistant-icon-button forge-assistant-read-aloud");
        readAloud.type = "button";
        readAloud.setAttribute("role", "switch");
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
        composer.appendChild(editBar);
        composer.appendChild(chip);
        composer.appendChild(input);
        composer.appendChild(toolbar);
        composer.appendChild(filePicker);
        body.appendChild(composer);
        Object.assign(this.nodes, {composer, editBar, editLabel, editCancel, chip, input,
                                   toolbar, attach, dictate, readAloud, send, stop,
                                   filePicker});

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
        this.nodes.workspaces.hidden = open;
        // The header's Workspace menu and the row are the same list. Collapsed
        // the row is right there under it, so the menu is a press that buys
        // nothing; it goes, and like the body it leaves the tab order rather
        // than merely stopping being painted. If it is open at the moment it
        // is hidden, it closes -- a menu whose button is gone cannot be
        // dismissed by pressing that button again.
        this.nodes.picker.hidden = !open;
        if (!open) {
            if (this.nodes.menu.dataset.which === "workspaces") this.closeMenu();
            this.renderWorkspaces();
        }
        // The panel is sized to its content while collapsed, so the row
        // appearing or going changes how wide it is and therefore where its
        // anchor puts it.
        this.place();
    };

    /** The workspace row: every tab this installation has, one press each.
     *
     * Rebuilt when the set of tabs changes, and otherwise only re-marked.
     * While the navigation observer fired on every attribute change in the
     * application, this rebuilt a dozen buttons on every progress tick of a
     * generation -- DOM churn nobody could see, and a keyboard focus on one of
     * those buttons dropped back to the page each time. Re-marking keeps the
     * same buttons, so focus, hover and a drag in progress survive a switch.
     */
    Shell.prototype.renderWorkspaces = function () {
        const row = this.nodes.workspaces;
        if (!row || row.hidden) return;
        const active = this.host.getActiveWorkspace();
        const found = this.host.pairs ? this.host.pairs() : this.host.listWorkspaces();
        const shape = found.map((workspace) => workspace.id + "\u0000" + workspace.label
            + "\u0000" + (workspace.available ? 1 : 0)).join("\u0001");
        if (found.length && shape === row.dataset.shape
            && row.children.length === found.length) {
            found.forEach((workspace, index) => {
                row.children[index].setAttribute("aria-current",
                                                 String(workspace.id === active));
            });
            return;
        }
        row.dataset.shape = shape;
        row.innerHTML = "";
        if (!found.length) {
            row.appendChild(element("p", "forge-assistant-menu-empty",
                                    "No workspaces on this page."));
            return;
        }
        found.forEach((workspace) => {
            const button = element("button", "forge-assistant-workspace", workspace.label);
            button.type = "button";
            // The highlight follows the host's own selection, not this press,
            // so a switch made anywhere else moves it too.
            button.setAttribute("aria-current", String(workspace.id === active));
            button.disabled = !workspace.available;
            button.addEventListener("click", () => {
                // Away first, so the page is not switching underneath a panel
                // that is about to stop being there. A failed switch is
                // reported in the status line of the panel that comes back.
                this.close();
                this.switchWorkspace(workspace.id);
            });
            row.appendChild(button);
        });
        this.markOverflow();
    };

    /** Which edge of the strip has row behind it.
     *
     * The strip is one line, narrower than its contents, and the stylesheet
     * fades whichever side is cut off -- so a row that continues looks like it
     * continues rather than like it ends there. The scrollbar is hidden
     * because it would cost the panel eight pixels of height to say the same
     * thing less clearly.
     */
    Shell.prototype.markOverflow = function () {
        const row = this.nodes.workspaces;
        if (!row) return;
        const slack = (row.scrollWidth || 0) - (row.clientWidth || 0);
        const at = row.scrollLeft || 0;
        // A pixel of slack is rounding, not room.
        const side = slack <= 1 ? "none"
            : (at <= 1 ? "end"
               : (at >= slack - 1 ? "start" : "both"));
        row.setAttribute("data-overflow", side);
    };

    // -- dragging the workspace strip --------------------------------------- //
    //
    // The strip holds more tabs than the panel is wide, and the panel is kept
    // to its own width on purpose -- a row that grew to the screen was a row
    // that covered the screen. So it scrolls, and on a mouse it scrolls by
    // being dragged: a horizontal scrollbar under a tab row is eight pixels
    // spent on a control nobody aims at, and a trackpad gesture is not
    // something every pointer has.
    //
    // Touch and pen are not driven from here. `touch-action: pan-x` hands
    // those to the browser, which pans natively and then cancels this
    // pointer; running both would fight the native gesture with a frame of
    // lag and lose.

    Shell.prototype.startStrip = function (event) {
        const row = this.nodes.workspaces;
        // Cleared on every press: a gesture that moved and then ended over a
        // gap leaves no click to consume the flag, and a stale one would eat
        // somebody's next real press.
        this.stripMoved = false;
        if (this.strip || !row || row.hidden) return;
        if (event.button !== undefined && event.button !== 0) return;
        if (!event.isPrimary) return;
        if (event.pointerType && event.pointerType !== "mouse") return;
        this.strip = {pointerId: event.pointerId, startX: event.clientX,
                      from: row.scrollLeft || 0, moved: false};
    };

    Shell.prototype.moveStrip = function (event) {
        const strip = this.strip;
        if (!strip || event.pointerId !== strip.pointerId) return;
        const dx = event.clientX - strip.startX;
        if (!strip.moved && Math.abs(dx) < DRAG_THRESHOLD) return;
        const row = this.nodes.workspaces;
        if (!strip.moved) {
            strip.moved = true;
            row.classList.add("forge-assistant-strip-dragging");
            try {
                row.setPointerCapture(event.pointerId);
            } catch (error) { /* capture is an optimisation, not a requirement */ }
        }
        // Content follows the hand: drag left and the row behind the right
        // edge comes in.
        row.scrollLeft = strip.from - dx;
        this.markOverflow();
    };

    Shell.prototype.endStrip = function (event, cancelled) {
        const strip = this.strip;
        if (!strip || (event && event.pointerId !== strip.pointerId)) return;
        // Cleared first, for the same reason `endDrag` does it: a late
        // `lostpointercapture` must find nothing to undo.
        this.strip = null;
        const row = this.nodes.workspaces;
        row.classList.remove("forge-assistant-strip-dragging");
        // The click that follows a gesture which actually moved is the tail of
        // that gesture, not a choice of tab.
        this.stripMoved = !cancelled && strip.moved;
        try {
            row.releasePointerCapture(strip.pointerId);
        } catch (error) { /* never held, or already released */ }
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
            this.placeMenu();
            return;
        }
        this.nodes.panel.classList.remove("forge-assistant-sheet");
        this.nodes.panel.removeAttribute("aria-modal");
        this.nodes.panel.setAttribute("role", "complementary");
        if (open) {
            // One width, collapsed or not. The workspace row is kept inside it
            // rather than allowed to set it: a row that grew to its contents
            // reached the far side of the screen on an installation with a
            // dozen tabs, which is a panel covering the page it is a control
            // for. It scrolls instead -- see `startStrip`.
            const width = Math.min(Math.max(this.state.panelWidth || NOMINAL_WIDTH,
                                            MIN_WIDTH),
                                   Math.min(MAX_WIDTH, view.width - 32));
            node.style.width = width + "px";
        }
        const box = {width: node.offsetWidth || NOMINAL_WIDTH,
                     height: node.offsetHeight || 44};
        if (open) this.placedHeight = node.offsetHeight;
        // Free float does not apply to the phone sheet above: a sheet is
        // anchored to a half of the screen and covers it, and there is nothing
        // for a floating position to mean.
        const at = this.state.freeFloat && this.state.floatAt
            ? floatPoint(this.state.floatAt, box, view, insets())
            : anchorPoint(this.anchor(), box, view, insets());
        node.style.left = at.left + "px";
        node.style.top = at.top + "px";
        // An open menu is positioned against the window, so a move, a resize
        // or a keyboard appearing has to take it along.
        this.placeMenu();
    };

    // -- drag -------------------------------------------------------------- //
    //
    // Pointer Events, so mouse, touch and pen are one code path. The state
    // machine is idle -> pressed -> dragging -> committed | cancelled, and the
    // fiddly parts are all at the end of it: the commit is cleared *before* the
    // capture is released, so a late `lostpointercapture` cannot undo it, and
    // the click is suppressed only if the gesture actually exceeded the
    // threshold -- a press that moved two pixels is a click, not a drag.

    /** The panel changed size. Placed again if it changed height.
     *
     * Only height: `placeNow` writes the panel's width itself, and a callback
     * that answered its own write would be an observer loop. A width that
     * reflowed the text into another height is a height change, and placed
     * once; the second pass finds the height it placed and stops.
     */
    Shell.prototype.resized = function () {
        const panel = this.nodes.panel;
        if (!panel || !this.state.panelOpen) return false;
        if (panel.offsetHeight === this.placedHeight) return false;
        this.place();
        return true;
    };

    Shell.prototype.startDrag = function (event, node) {
        if (event.button !== undefined && event.button !== 0) return;
        if (!event.isPrimary) return;
        // `supersede` has already ended any drag whose release never came, so
        // one still here is this same press arriving twice (the launcher and
        // the header are both handles); the first one stands.
        if (this.drag) return;
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
        // The button came up somewhere this page never heard about -- outside
        // the window, in another application, across a remote-desktop session
        // that changed hands. Without this the panel follows the cursor until
        // the next click, which then lands as the end of a drag.
        if (event.pointerType === "mouse" && event.buttons === 0) {
            this.endDrag(null, true);
            return;
        }
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
        // Free float has nothing to snap to, so there is nothing to preview
        // and nothing to choose: where it is put is where it stays.
        if (this.state.freeFloat) {
            drag.at = fractionOf({left, top, width: drag.width, height: drag.height},
                                 viewport());
            return;
        }
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
            if (this.state.freeFloat) {
                if (drag.at) {
                    this.state.floatAt = drag.at;
                    this._saveFloat();
                }
            } else {
                this.state.anchorOverride = drag.anchor;
            }
            // Only the launcher's own click is the tail of this gesture. A
            // panel dragged by its header leaves no click on the launcher, and
            // a flag set for one used to eat the next real press on it.
            if (drag.node === this.nodes.launcher) this.suppressClick = true;
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

    // Every listener this shell adds goes through here, and every one of them
    // is guarded. A handler that throws used to leave whatever it was half way
    // through exactly as it was -- a drag begun and never ended, a menu open
    // with nothing to close it -- and the page kept that state until it was
    // reloaded. Now the exception is logged and the shell is put back into a
    // state it knows: see `recover`.
    Shell.prototype.on = function (node, type, handler, options) {
        const guarded = (event) => {
            try {
                return handler(event);
            } catch (error) {
                this.fault(error);
                return undefined;
            }
        };
        node.addEventListener(type, guarded, options);
        this.disposers.push(() => node.removeEventListener(type, guarded, options));
    };

    Shell.prototype.fault = function (error) {
        // Once per distinct failure: a handler that fails on every pointermove
        // would otherwise bury the console as well as the page. Distinct means
        // the message and the line that threw -- not the whole stack, whose
        // callers differ from one dispatch to the next.
        const stack = String((error && error.stack) || "").split("\n");
        const text = String((error && error.message) || error) + "|" + (stack[1] || "");
        this.faults = this.faults || new Set();
        if (!this.faults.has(text)) {
            this.faults.add(text);
            console.error("Forge Assistant: a handler failed; the panel was reset", error);
        }
        this.recover();
    };

    /** Put the shell back into a state it knows, keeping everything that
     *  matters.
     *
     * Every gesture ends, every flag a gesture owns is cleared, the preview is
     * put away, and exactly one of the launcher and the panel is drawn again
     * where it belongs. The conversation, the draft, the open/closed state and
     * focus mode are left exactly as they were -- this undoes a *stuck*
     * interaction, never a chosen one. Rate-limited, and it cannot throw.
     */
    Shell.prototype.recover = function () {
        const now = Date.now();
        if (this.recoveredAt && now - this.recoveredAt < RECOVER_COOLDOWN) return false;
        this.recoveredAt = now;
        try {
            this.cancelGestures();
            const root = this.nodes.root;
            if (!root) return false;
            // A theme that rebuilds the body takes this root out with it. The
            // shell is still alive and still holds its nodes; they only need
            // putting back.
            if (!root.isConnected && document.body) document.body.appendChild(root);
            if (this.frame) {
                window.cancelAnimationFrame(this.frame);
                this.frame = 0;
            }
            this.showOpen(this.state.panelOpen);
            return true;
        } catch (error) {
            console.error("Forge Assistant: the panel could not be reset", error);
            return false;
        }
    };

    /** End every gesture in progress, as a cancellation.
     *
     * For the moments nobody can still be mid-gesture: the window lost focus,
     * the page was hidden, a new primary pointer went down. A drag returns to
     * where it came from; a resize keeps the width it had reached, because
     * that is what was on screen when the hand came off.
     */
    Shell.prototype.cancelGestures = function () {
        if (this.drag) this.endDrag(null, true);
        if (this.strip) this.endStrip(null, true);
        if (this.resizing) this.endResize(null, true);
        this.suppressClick = false;
        this.stripMoved = false;
        this.resetTaps();
        const root = this.nodes.root;
        if (root) root.classList.remove("forge-assistant-dragging");
        if (this.nodes.ghost) this.nodes.ghost.hidden = true;
    };

    /** A primary pointer went down, anywhere on the page.
     *
     * Which means every gesture before it is over, whether or not its end was
     * ever delivered: a mouse has one pointer and cannot press again while it
     * is still pressed, and a new primary touch only exists once every earlier
     * finger has lifted. The case this is for is a release that never arrives
     * -- a remote-desktop session dropping mid-drag is the classic one -- which
     * used to leave the drag running for the life of the page. For a touch it
     * could never end at all, because every later touch has a different id.
     *
     * Runs in the capture phase on the window, before anything else sees the
     * press, and costs two comparisons when nothing is stuck.
     */
    Shell.prototype.supersede = function (event) {
        if (!event || !event.isPrimary) return;
        if (this.drag || this.strip || this.resizing) this.cancelGestures();
        this.suppressClick = false;
        this.stripMoved = false;
        this.noticeCover(event);
    };

    /** Somebody pressed where the launcher is, and the press went elsewhere.
     *
     * The belt to `supersede`'s braces. Whatever is lying over the launcher --
     * something of ours left behind, or something of somebody else's -- the
     * person pressing it is stuck, and in focus mode the launcher is their
     * only way to another workspace. Two such presses close together reset the
     * shell; if the launcher is *still* covered after that and focus mode is
     * on, focus mode is left, so the host's own tab bar is back. Nothing is
     * done about a dialog: a dialog over the launcher is meant to be there.
     */
    Shell.prototype.noticeCover = function (event) {
        const launcher = this.nodes.launcher;
        const root = this.nodes.root;
        if (!launcher || launcher.hidden || !root || this.state.panelOpen) return;
        const target = event.target;
        if (target && root.contains(target)) {
            this.coveredAt = 0;
            return;
        }
        if (target && target.closest && target.closest(
            "dialog, [role=\"dialog\"], [aria-modal=\"true\"], #lightboxModal")) return;
        const box = launcher.getBoundingClientRect();
        const x = event.clientX;
        const y = event.clientY;
        if (!(x >= box.left && x <= box.right && y >= box.top && y <= box.bottom)) return;
        const now = Date.now();
        if (!this.coveredAt || now - this.coveredAt > COVERED_WINDOW) {
            this.coveredAt = now;
            return;
        }
        this.coveredAt = 0;
        console.warn("Forge Assistant: something is lying over the launcher",
                     describe(target));
        this.recoveredAt = 0;
        this.recover();
        if (this.focus.isActive() && !this.reachable()) {
            this.focus.exit();
            this.focusOff();
            this._save();
            this.say("Focus mode was turned off: something was covering the assistant.",
                     "warn");
        }
    };

    /** Is the visible control the thing a press at its centre would reach? */
    Shell.prototype.reachable = function () {
        const node = this.state.panelOpen ? this.nodes.header : this.nodes.launcher;
        if (!node || typeof document.elementFromPoint !== "function") return true;
        const box = node.getBoundingClientRect();
        if (!box.width || !box.height) return false;
        const hit = document.elementFromPoint(box.left + box.width / 2,
                                              box.top + box.height / 2);
        return !!(hit && this.nodes.root.contains(hit));
    };

    /** The page came back: visible again, restored, resumed, refocused.
     *
     * Gestures are not touched here -- they were cancelled on the way out
     * (`blur`, `visibilitychange`, `pagehide`), and a window that regains
     * focus because somebody pressed the launcher from inside an iframe is in
     * the middle of starting one. What is checked is the shell itself: still
     * on the page, and not waiting on a frame that a hidden page never drew.
     */
    Shell.prototype.heal = function () {
        const root = this.nodes.root;
        if (root && !root.isConnected && document.body) document.body.appendChild(root);
        if (this.frame) {
            window.cancelAnimationFrame(this.frame);
            this.frame = 0;
        }
        this.place();
    };

    Shell.prototype.wire = function () {
        const nodes = this.nodes;

        // Before anything else sees a press. See `supersede`.
        this.on(window, "pointerdown", (event) => this.supersede(event), true);
        // And before a press lands, a message's open actions go if the press
        // is not on that message. See `dismissActions`.
        this.on(window, "pointerdown", (event) => this.dismissActions(event), true);

        [nodes.launcher, nodes.header].forEach((handle) => {
            this.on(handle, "pointerdown", (event) => {
                this.startDrag(event, this.state.panelOpen ? nodes.panel : nodes.launcher);
            });
        });
        this.on(window, "pointermove", (event) => this.moveDrag(event));
        this.on(window, "pointerup", (event) => this.endDrag(event, false));
        this.on(window, "pointercancel", (event) => this.endDrag(event, true));
        // Capture taken away without a release -- the element was hidden, or
        // the browser decided the gesture was its own. `endDrag` clears the
        // drag before it releases capture itself, so its own release lands
        // here and finds nothing to do.
        [nodes.launcher, nodes.panel].forEach((node) => {
            this.on(node, "lostpointercapture", (event) => this.endDrag(event, true));
        });

        // The workspace strip scrolls by being dragged. Its pointer stream is
        // separate from the panel's: the row is not in the header, so the two
        // gestures cannot both start, and neither needs to know about the
        // other.
        this.on(nodes.workspaces, "pointerdown", (event) => this.startStrip(event));
        this.on(window, "pointermove", (event) => this.moveStrip(event));
        this.on(window, "pointerup", (event) => this.endStrip(event, false));
        this.on(window, "pointercancel", (event) => this.endStrip(event, true));
        // Capture, so the press is stopped before it reaches the button it
        // landed on. `detail` is 0 for a keyboard activation, which is never
        // the tail of a drag and must not be swallowed by a stale flag.
        this.on(nodes.workspaces, "click", (event) => {
            if (!this.stripMoved) return;
            this.stripMoved = false;
            if (!event.detail) return;
            event.stopPropagation();
            event.preventDefault();
        }, true);
        // Native panning -- touch, pen, a trackpad, the keyboard inside the
        // row -- moves the strip without going through `moveStrip`, and the
        // fade has to follow it.
        this.on(nodes.workspaces, "scroll", () => this.markOverflow());

        this.on(nodes.launcher, "click", (event) => this.launcherClick(event));
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

        this.on(nodes.input, "input", () => this.typed());
        this.on(nodes.editCancel, "click", () => this.cancelEdit());
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
        // A tap on a message shows its actions. See `tapBubble`.
        this.on(nodes.transcript, "click", (event) => this.tapBubble(event));
        this.on(nodes.transcript, "keydown", (event) => this.bubbleKey(event));
        // A picture in a bubble arrives after the bubble does and grows it,
        // which moves the bottom out from under a reader who was at it. Caught
        // in the capture phase because `load` does not bubble.
        this.on(nodes.transcript, "load", () => {
            if (this.following) this.toBottom();
        }, true);
        this.on(nodes.resize, "keydown", (event) => this.resizeKey(event));
        this.on(nodes.resize, "pointerdown", (event) => this.startResize(event));
        this.on(window, "pointermove", (event) => this.moveResize(event));
        this.on(window, "pointerup", (event) => this.endResize(event, false));
        this.on(window, "pointercancel", (event) => this.endResize(event, true));
        this.on(nodes.resize, "lostpointercapture", (event) => this.endResize(event, true));

        this.on(window, "resize", () => this.place());
        if (window.visualViewport) {
            this.on(window.visualViewport, "resize", () => this.place());
            this.on(window.visualViewport, "scroll", () => this.place());
        }
        this.on(window, "orientationchange", () => this.place());
        // The panel is placed from its own height, and its height changes with
        // nobody placing it: a message arrives, a picture in a bubble loads, a
        // tap opens a message's actions. Placed only on the events above, a
        // panel docked along the bottom grew downwards off the window, taking
        // the composer and Send with it. Observed rather than re-placed on
        // every render: a ResizeObserver reports after layout and only when
        // the box changed, so a reply streaming into a transcript that is
        // already at its full height and scrolling costs nothing.
        if (typeof ResizeObserver === "function") {
            const watched = new ResizeObserver(() => this.resized());
            watched.observe(nodes.panel);
            this.disposers.push(() => watched.disconnect());
        }
        // Nobody is mid-gesture across a hidden page or an unfocused window,
        // and a release that happened while we were not looking is never
        // coming. Coming back, the shell is checked over as well as the
        // conversation.
        this.on(document, "visibilitychange", () => {
            if (document.hidden) {
                this.cancelGestures();
                // Nothing held open while away: see `Store.sleep`.
                this.store.sleep();
                return;
            }
            this.heal();
            this.store.wake();
        });
        this.on(window, "blur", () => this.cancelGestures());
        this.on(window, "focus", () => this.heal());
        this.on(window, "online", () => this.store.reconcile(true));
        this.on(window, "pageshow", (event) => {
            this.heal();
            // Restored from the back-forward cache: every connection the page
            // had was closed when it went in.
            this.store.reconcile(!!(event && event.persisted));
        });
        // Chrome freezes background tabs and thaws them later; a frozen page's
        // stream is as good as gone. `resume` is its word for "thawed".
        this.on(document, "resume", () => {
            this.heal();
            this.store.reconcile(true);
        });
        this.on(window, "pagehide", () => {
            this.cancelGestures();
            this.store.flushDrafts();
        });
        this.on(document, "keydown", (event) => this.documentKey(event), true);
        // The browser's full screen, which focus asks for and the browser can
        // end without asking anybody. Both names: Safari's are prefixed. See
        // `enterScreen`.
        ["fullscreenchange", "webkitfullscreenchange"].forEach((type) => {
            this.on(document, type, () => this.screenChanged());
        });
        ["fullscreenerror", "webkitfullscreenerror"].forEach((type) => {
            this.on(document, type, (event) => this.screenRefused(event));
        });
        // Another extension's dialog has taken the page. See `yieldTo`.
        this.on(document, FOREIGN_OVERLAY, (event) => this.yieldTo(event));

        // Drawn once a frame, with the latest view. A streamed reply announces
        // every token, and each announcement used to redraw the panel on the
        // spot -- markdown, fingerprint, a layout read -- several times per
        // frame the screen could only show once. A hidden tab draws nothing
        // until it is looked at again.
        this.disposers.push(this.store.subscribeState((view) => {
            this.pendingView = view;
            if (this.viewFrame) return;
            this.viewFrame = window.requestAnimationFrame(() => {
                this.viewFrame = 0;
                const latest = this.pendingView;
                this.pendingView = null;
                if (!latest) return;
                try {
                    this.render(latest);
                } catch (error) {
                    this.fault(error);
                }
            });
        }));
        this.disposers.push(this.host.subscribeNavigation((active) => {
            this.noteWorkspace(active);
            this.applySuppression();
            this.renderWorkspaces();
            if (this.state.focusEnabled && active && this.focus.isActive()
                && active !== this.focus.activeWorkspace()) {
                const moved = this.focus.moveTo(active, this.host);
                if (!moved.ok) {
                    this.focusOff();
                    this.say(moved.reason, "warn");
                    this._save();
                } else if (moved.note) {
                    this.say(moved.note, "warn");
                }
            }
        }));
    };

    /** The box was typed into. It is the draft -- unless it holds an edit,
     *  and then it is the message: the draft is kept as it was and comes back
     *  when the edit ends. */
    Shell.prototype.typed = function () {
        const nodes = this.nodes;
        if (this.editing) {
            this.grow();
            nodes.send.disabled = !this.canSaveEdit(this.store.snapshot());
            return;
        }
        this.store.setDraftText(nodes.input.value);
        this.grow();
    };

    Shell.prototype.grow = function () {
        const input = this.nodes.input;
        input.style.height = "auto";
        const lineHeight = 20;
        // Six lines for a message; more for an edit, because the prompt being
        // edited is often long and the transcript above gives the room up
        // (it is the panel's one part that shrinks). Past that, it scrolls.
        const max = this.editing ? editHeight(lineHeight) : lineHeight * 6 + 12;
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
        // One of the moments the panel looks, because nothing is pushed to it
        // while no reply is on its way. See `Store.check`.
        this.store.check();
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

    Shell.prototype.launcherClick = function (event) {
        if (this.suppressClick) {
            // Only this gesture's click, and only once. A later keyboard
            // activation must not be swallowed by a drag that has finished.
            // A drag also ends any press count it interrupted.
            this.suppressClick = false;
            this.resetTaps();
            event.preventDefault();
            return;
        }
        this.tapLauncher(event);
    };

    /** The host's selection moved: remember the workspace it replaced, for
     * the launcher's double press. Only a move from one real workspace to
     * another counts -- a blank between two reads is not somewhere anybody
     * was.
     */
    Shell.prototype.noteWorkspace = function (active) {
        // Changing workspace is another of the moments the panel looks.
        if (active && this.activeWorkspace && active !== this.activeWorkspace
            && this.store && typeof this.store.check === "function") {
            this.store.check();
        }
        if (active) {
            if (this.lastWorkspace && active !== this.lastWorkspace) {
                this.previousWorkspace = this.lastWorkspace;
            }
            this.lastWorkspace = active;
        }
        this.activeWorkspace = active;
    };

    /** A press on the launcher, counted.
     *
     * One press opens the panel, two go back to the previous workspace, three
     * toggle focus. The count is ours rather than the click's `detail`,
     * because `detail` follows the system's double-click setting and does not
     * count taps the same way on every touchscreen.
     *
     * Three acts at once, inside the press, because entering focus asks for
     * the browser's full screen and a browser grants that only to a press
     * that is still being handled. One and two cannot act until the window
     * has passed with no further press, since either may still become more.
     */
    Shell.prototype.tapLauncher = function (event) {
        // A keyboard activation is one press, and nobody pressing Enter
        // expects to wait to find out whether they pressed it again.
        if (!event || !event.detail) {
            this.resetTaps();
            this.open();
            return;
        }
        const taps = (this.taps || 0) + 1;
        this.resetTaps();
        if (taps >= 3) {
            this.toggleFocus();
            return;
        }
        this.taps = taps;
        this.tapTimer = window.setTimeout(() => {
            const counted = this.taps;
            this.resetTaps();
            try {
                if (counted === 2) this.backToPrevious();
                else this.open();
            } catch (error) {
                this.fault(error);
            }
        }, TAP_WINDOW);
    };

    /** Forget a half-counted launcher gesture. */
    Shell.prototype.resetTaps = function () {
        if (this.tapTimer) window.clearTimeout(this.tapTimer);
        this.tapTimer = 0;
        this.taps = 0;
    };

    /** Back to the workspace before this one -- and, pressed again, forward to
     * the one just left, because that is now the one before. Through
     * `switchWorkspace`, so focus mode follows the switch rather than hiding
     * it. With nowhere to go back to yet, nothing happens.
     */
    Shell.prototype.backToPrevious = function () {
        const back = this.previousWorkspace;
        if (!back) return Promise.resolve("");
        return this.switchWorkspace(back);
    };

    Shell.prototype.toggle = function () {
        if (this.state.panelOpen) this.close();
        else this.open();
    };

    // One press, both ways: focus and the browser's full screen go on
    // together and come off together. See `enterScreen`.
    Shell.prototype.toggleFocus = function () {
        if (this.focus.isActive()) {
            this.focus.exit();
            this.focusOff();
            this._save();
            this.place();
            return;
        }
        if (this.refocus(this.host.getActiveWorkspace())) this.enterScreen();
    };

    /** Focus mode is off, whichever way it came to be.
     *
     * The toggle, Escape, a switch that could land nowhere, a covered
     * launcher, the browser leaving full screen: each used to write these
     * three lines itself. They are one place now because there is a fourth
     * -- the full screen that came with focus goes with it -- and a path that
     * forgot it would leave the browser full screen with the tab bar back.
     */
    Shell.prototype.focusOff = function () {
        this.state.focusEnabled = false;
        this.state.focusWorkspaceId = null;
        if (this.nodes.focusToggle) this.nodes.focusToggle.setAttribute("aria-pressed", "false");
        this.leaveScreen();
    };

    // Focus a workspace and do the bookkeeping: the toggle's pressed state,
    // what the next session starts with, and the panel's own position, which
    // is measured against the focused box.
    //
    // Shared with the workspace picker, because entering focus and moving it
    // to another workspace are one operation with the same three outcomes --
    // refused, entered with a caveat, entered cleanly -- and only one of the
    // two used to report all three.
    Shell.prototype.refocus = function (id) {
        const found = this.focus.enter(id, this.host);
        if (!found.ok) {
            this.focusOff();
            this.say(found.reason, "warn");
            this._save();
            this.place();
            return false;
        }
        if (found.note) this.say(found.note, "warn");
        this.state.focusEnabled = true;
        this.state.focusWorkspaceId = id;
        this.nodes.focusToggle.setAttribute("aria-pressed", "true");
        this._save();
        this.place();
        return true;
    };

    // -- the browser's full screen, with focus ------------------------------ //
    //
    // Focus takes the page's own chrome away -- the tab bar, a theme's header
    // and footer. Asked for alongside it: the browser's chrome as well, from
    // the same press. So the Focus toggle asks the browser for full screen
    // when it turns focus on, and ends it when it turns focus off.
    //
    // What goes full screen is the whole document, never the workspace. An
    // element in full screen is the only thing the browser draws -- the
    // assistant, a dialog, a toast, anything outside it would be gone, and
    // with the assistant gone so is the way out of focus. The document in
    // full screen is just the page with the browser's bars taken away, and
    // focus inside it works exactly as it does without.
    //
    // Only a full screen this shell asked for is ever ended by it. F11, a
    // video somebody made full screen, another extension's own: none of them
    // are its to end. And when the browser ends ours by its own means --
    // Escape, Android's back gesture, a switch of browser tab -- focus comes
    // off with it, so the toggle never says one thing while the screen shows
    // another.
    //
    // Where the page cannot have it -- iPhone Safari gives full screen to
    // videos and nothing else; a frame without `allowfullscreen` gives it to
    // nobody -- nothing is said and focus is what it always was.

    /** What is full screen now, under either of the API's names. */
    function screenElement() {
        return document.fullscreenElement || document.webkitFullscreenElement || null;
    }

    function screenExit() {
        const leave = document.exitFullscreen || document.webkitExitFullscreen;
        if (typeof leave !== "function") return;
        try {
            const left = leave.call(document);
            if (left && typeof left.catch === "function") left.catch(() => undefined);
        } catch (error) { /* the browser had already left it */ }
    }

    /** Ask for full screen, from inside the press that turned focus on.
     *
     * It has to be inside the press: a browser grants full screen only to a
     * page somebody has just interacted with, which is also why a reload
     * cannot bring it back. A reload has never brought focus back either, so
     * the two still start together.
     */
    Shell.prototype.enterScreen = function () {
        // Already ours, or already asked for and not yet answered. An answer
        // that never came -- a browser that neither granted nor refused -- is
        // not waited on for ever.
        if (this.screenOwned) return false;
        if (this.screenPending && Date.now() - this.screenPending < SCREEN_WAIT) return false;
        const page = document.documentElement;
        if (!page || !(document.fullscreenEnabled || document.webkitFullscreenEnabled)) {
            return false;
        }
        // Something else is full screen already -- F11 is not counted by the
        // API, so this is a video, or another extension's. Left alone, and
        // not claimed: turning focus off must not end it.
        if (screenElement()) return false;
        const request = page.requestFullscreen || page.webkitRequestFullscreen;
        if (typeof request !== "function") return false;
        let asked;
        try {
            // The unprefixed call takes options; Safari's prefixed one takes a
            // number, so it is given nothing. "hide" asks Android to put its
            // navigation bar away too -- the most of the screen it will give.
            asked = request === page.requestFullscreen
                ? request.call(page, {navigationUI: "hide"})
                : request.call(page);
        } catch (error) {
            return false;
        }
        this.screenPending = Date.now();
        if (asked && typeof asked.then === "function") {
            // The answer is read off the page rather than taken from the
            // promise, the same way the change event's is, so the two cannot
            // disagree about which of them came first. Settled either way,
            // it is no longer pending: a grant that was over before it was
            // read must not leave the next press waiting on it.
            asked.then(() => {
                this.screenChanged();
                this.screenPending = 0;
            }, () => this.screenRefused());
        }
        return true;
    };

    /** The browser said no: a frame without `allowfullscreen`, a press it did
     *  not count, a setting. Focus stands on its own.
     *
     * The error event is fired at whichever element asked, and bubbles: a
     * video's refusal is the video's, and must not cancel a request of ours
     * that is still waiting for its answer.
     */
    Shell.prototype.screenRefused = function (event) {
        const target = event && event.target;
        if (target && target !== document && target !== document.documentElement) return;
        this.screenPending = 0;
    };

    /** End the full screen that came with focus -- if it is still ours. */
    Shell.prototype.leaveScreen = function () {
        // A request still in flight is dealt with when it is answered: see
        // `screenChanged`, which finds focus off and gives the screen back.
        if (!this.screenOwned) return;
        // Cleared first, so the change this causes is not taken for the
        // browser ending it.
        this.screenOwned = false;
        // Something is full screen on top of the page -- a video -- and ending
        // it would end the thing being watched. The page's full screen goes
        // when that one does, by the browser's own means.
        if (screenElement() !== document.documentElement) return;
        screenExit();
    };

    /** The browser's full screen changed: granted, ended, or stacked on. */
    Shell.prototype.screenChanged = function () {
        const now = screenElement();
        const page = document.documentElement;
        if (this.screenPending && now === page) {
            this.screenPending = 0;
            if (this.state.focusEnabled) {
                this.screenOwned = true;
                return;
            }
            // Focus came off while the browser was deciding. The answer is to
            // a question nobody is asking any more.
            screenExit();
            return;
        }
        if (now || !this.screenOwned) return;
        // Ended by the browser -- Escape, a back gesture, a switch of tab.
        // Focus follows it off, so the toggle and the screen agree.
        this.screenOwned = false;
        if (!this.state.focusEnabled) return;
        this.focus.exit();
        this.focusOff();
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
        menu.appendChild(this.cancelItem());
        menu.hidden = false;
        this.placeMenu();
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

    // Every menu ends with a way out that chooses nothing.
    //
    // Escape closes them, and so does opening the other one, and so would a
    // press on the button that opened it. None of that is any use to a thumb:
    // a phone has no Escape key, and the two header buttons are a small target
    // beside a menu that is covering them. So once a menu was open the only
    // obvious way past it was to pick something from it, and a menu you cannot
    // leave without committing is a menu people learn not to open.
    //
    // Added here rather than in each builder, so a menu added later cannot
    // forget it.
    Shell.prototype.cancelItem = function () {
        const item = element("button",
                             "forge-assistant-menu-item forge-assistant-cancel",
                             "Cancel");
        item.type = "button";
        item.setAttribute("role", "menuitem");
        item.addEventListener("click", () => this.closeMenu());
        return item;
    };

    // The menu is positioned against the window, not inside the panel.
    //
    // It used to be an absolutely positioned child with `max-height: 50vh`,
    // which the panel's own `overflow: hidden` then clipped to the panel's
    // box. With the conversation collapsed that box is a header and an
    // accordion tall, so most of the workspace list was simply cut off -- and
    // unreachable, because scrolling a menu whose visible region is shorter
    // than its own scroll viewport cannot bring the bottom of it into view.
    // Cancel is the last item, so the one control added to let people out of a
    // menu was the first thing to be cut off it.
    //
    // Fixed to the window, sized to the room that is actually there, and
    // opened upwards when there is more room above -- which there is whenever
    // the panel is docked along the bottom, where a downward menu has only the
    // few pixels between the header and the bottom of the screen.
    //
    // `top` in both directions and never `bottom`: `bottom` on a fixed element
    // is measured against the layout viewport while everything else here is
    // measured against the visual one, and mixing the two is how a panel ends
    // up behind a phone's keyboard. Upwards costs one measurement of the
    // menu's own height, which is the only way to know where its top goes.
    Shell.prototype.placeMenu = function () {
        const menu = this.nodes.menu;
        const header = this.nodes.header;
        if (!menu || menu.hidden || !header) return;
        const box = header.getBoundingClientRect();
        const view = viewport();
        const below = (view.top + view.height) - box.bottom - MENU_GAP - MENU_EDGE;
        const above = box.top - view.top - MENU_GAP - MENU_EDGE;
        const up = above > below;
        const room = Math.max(MENU_FLOOR, up ? above : below);
        menu.style.left = Math.round(box.left) + "px";
        menu.style.width = Math.round(box.width) + "px";
        menu.style.maxHeight = Math.round(room) + "px";
        if (!up) {
            menu.style.top = Math.round(box.bottom + MENU_GAP) + "px";
            return;
        }
        // Measured with the cap already applied, so this is the height it will
        // actually be drawn at rather than the height it would like.
        const tall = Math.min(menu.offsetHeight || room, room);
        menu.style.top = Math.round(Math.max(view.top + MENU_EDGE,
                                             box.top - MENU_GAP - tall)) + "px";
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
                this.switchWorkspace(workspace.id);
            });
            return item;
        });
    };

    // Switching workspace, with focus mode taken into account.
    //
    // Focus hides every panel but the one it is filling, and it hides them
    // with `display: none !important` -- which beats the inline
    // `display: block` Gradio writes on the panel it has just switched to. So
    // with focus on, pressing a tab button changed the host's selection and
    // changed nothing on the screen: the destination stayed hidden, the old
    // workspace stayed fixed to the viewport, and `getActiveWorkspace()` --
    // which answers with the panel that is *showing* -- went on naming the
    // focused one. The switch could not even be observed, so the confirmation
    // watchdog ran its four seconds out and reported that the workspace had
    // not opened. What that looked like is a picker that refuses to move until
    // focus mode is turned off.
    //
    // So focus comes off for the switch and goes back on at the destination.
    // Off first, because the host cannot show a panel this code is hiding;
    // back on afterwards, because somebody in focus mode who asks for another
    // workspace is asking for that workspace, not for the end of focus mode.
    // And back on at the *old* one if the switch fails, because a switch that
    // did not happen should not cost focus mode either.
    Shell.prototype.switchWorkspace = function (id) {
        const focused = this.focus.isActive() ? this.focus.activeWorkspace() : "";
        if (focused) this.focus.exit();
        return this.host.activateWorkspace(id).then(() => {
            if (focused) this.refocus(id);
            return id;
        }).catch((error) => {
            if (focused) this.refocus(focused);
            this.say(error.message || "That workspace did not open.", "warn");
            return "";
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
        // Free float first, because it is a mode rather than an action, and
        // because the two below it give a graphics card back and are not what
        // anybody wants to hit by accident. It is the shell's own preference,
        // so it is added here rather than in the host's list of utilities,
        // which is about things the *host* can be asked to do.
        const own = [this.floatItem(), this.autoAttachItem(), this.sendToGenerateItem(),
                     this.newThreadItem()];
        if (NS.systemEditor) own.push(this.systemPromptItem());
        if (NS.look) own.push(this.customizeItem());
        return own.concat(this.host.listUtilities().map((utility) => {
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
        }));
    };

    Shell.prototype.floatItem = function () {
        const on = !!this.state.freeFloat;
        const item = element("button",
                             "forge-assistant-menu-item forge-assistant-float",
                             "Free Float");
        item.type = "button";
        // A checkbox, not a command: it reports its state rather than only
        // acting, so a screen reader says whether the mode is on.
        item.setAttribute("role", "menuitemcheckbox");
        item.setAttribute("aria-checked", String(on));
        item.title = on
            ? "Snap the panel back to one of the six resting places"
            : "Put the panel anywhere in the window";
        item.addEventListener("click", () => {
            this.closeMenu();
            this.setFreeFloat(!on);
        });
        return item;
    };

    Shell.prototype.autoAttachItem = function () {
        const on = !!this.state.autoAttach;
        const item = element("button",
                             "forge-assistant-menu-item forge-assistant-auto-attach",
                             "Auto Attach");
        item.type = "button";
        // A mode, like Free Float above it: it reports its state.
        item.setAttribute("role", "menuitemcheckbox");
        item.setAttribute("aria-checked", String(on));
        item.title = on
            ? "Stop attaching the picture showing in txt2img or img2img"
            : "Attach the picture showing in txt2img or img2img to each message";
        item.addEventListener("click", () => {
            this.closeMenu();
            this.setAutoAttach(!on);
        });
        return item;
    };

    Shell.prototype.sendToGenerateItem = function () {
        const on = !!this.state.sendToGenerate;
        const item = element("button",
                             "forge-assistant-menu-item forge-assistant-send-generate",
                             "Send to Generate");
        item.type = "button";
        // A mode, like the two above it: it reports its state.
        item.setAttribute("role", "menuitemcheckbox");
        item.setAttribute("aria-checked", String(on));
        item.title = on
            ? "Make the send button under a reply only write the prompt"
            : "Make the send button under a reply write the prompt and press Generate";
        item.addEventListener("click", () => {
            this.closeMenu();
            this.setSendToGenerate(!on);
        });
        return item;
    };

    Shell.prototype.setSendToGenerate = function (on) {
        this.state.sendToGenerate = !!on;
        this._saveSendToGenerate();
        this.renderSendMode();
        this.tell(on ? "Send to Generate is on: the button under a reply writes the prompt "
            + "and presses Generate." : "Send to Generate is off: the button under a reply "
            + "only writes the prompt.", "info");
        return this.state.sendToGenerate;
    };

    /** "Add the ability to start a new thread directly from the fly out menu.
     *  It should start a fresh thread with the character." */
    Shell.prototype.newThreadItem = function () {
        const character = this.store.snapshot().selection.character;
        const item = element("button",
                             "forge-assistant-menu-item forge-assistant-new-thread",
                             "New thread");
        item.type = "button";
        item.setAttribute("role", "menuitem");
        item.disabled = !character;
        item.title = character ? "Start a fresh thread with " + character
            : "Choose a conversation first";
        item.addEventListener("click", () => {
            this.closeMenu();
            this.startThread(character);
        });
        return item;
    };

    /** "For flyout, put this option to open this view in the '...' menu": the
     *  full-page editor for the character's system prompt, the one LLM
     *  Studio's character screen opens too. The panel stays where it is --
     *  the editor is over everything, and closing it is back to the
     *  conversation. */
    Shell.prototype.systemPromptItem = function () {
        const character = this.store.snapshot().selection.character;
        const item = element("button",
                             "forge-assistant-menu-item forge-assistant-system-prompt",
                             "System prompt\u2026");
        item.type = "button";
        item.setAttribute("role", "menuitem");
        item.setAttribute("aria-haspopup", "dialog");
        item.disabled = !character;
        item.title = character ? "Edit what " + character + " is told before every conversation"
            : "Choose a conversation first";
        item.addEventListener("click", () => {
            this.closeMenu();
            NS.systemEditor.open(character);
        });
        return item;
    };

    Shell.prototype.startThread = function (character) {
        this.tell("Starting a new thread with " + character + "\u2026", "info");
        return this.store.createThread(character).then((outcome) => {
            if (!outcome || !outcome.ok) {
                this.tell((outcome && outcome.error && outcome.error.message)
                    || "A new thread could not be started.", "warn");
                return false;
            }
            // Somebody who asked for a new thread is about to write in it.
            if (!this.state.conversationExpanded) {
                this.state.conversationExpanded = true;
                this.applyAccordion();
                this._save();
            }
            this.tell("New thread with " + character + ".", "info");
            return true;
        });
    };

    Shell.prototype.setAutoAttach = function (on) {
        this.state.autoAttach = !!on;
        this._saveAutoAttach();
        this.renderAutoAttach();
        this.tell(on ? "Auto Attach is on: the picture showing in txt2img or img2img "
            + "goes with your next message." : "Auto Attach is off.", "info");
        return this.state.autoAttach;
    };

    /** The paperclip says when Auto Attach is on, so nobody is surprised by a
     *  picture on a message they did not attach one to. */
    Shell.prototype.renderAutoAttach = function () {
        const attach = this.nodes.attach;
        if (!attach) return;
        const on = !!this.state.autoAttach;
        // Every render calls this, a streamed reply once a frame: written only
        // when it changed.
        if (attach.dataset.auto === (on ? "on" : "off")) return;
        attach.dataset.auto = on ? "on" : "off";
        const title = on
            ? "Attach an image (Auto Attach is on: the picture showing in txt2img or "
              + "img2img goes with each message)"
            : "Attach an image";
        attach.title = title;
        attach.setAttribute("aria-label", title);
    };

    Shell.prototype.customizeItem = function () {
        const item = element("button", "forge-assistant-menu-item", "Customize\u2026");
        item.type = "button";
        item.setAttribute("role", "menuitem");
        item.setAttribute("aria-haspopup", "dialog");
        item.title = "Colours, title, corners and animation for the button";
        item.addEventListener("click", () => {
            this.closeMenu();
            // Back to the launcher first: it is the thing being customized,
            // and it takes each change live behind the dialog.
            if (this.state.panelOpen) this.close();
            NS.look.editor(this).open();
        });
        return item;
    };

    Shell.prototype.setFreeFloat = function (on) {
        this.state.freeFloat = !!on;
        // Turned on where the panel already is, rather than somewhere of its
        // own choosing: a mode that moves the thing you were looking at is a
        // mode people turn off again to find it.
        if (this.state.freeFloat && !this.state.floatAt) {
            const node = this.state.panelOpen ? this.nodes.panel : this.nodes.launcher;
            const box = node && node.getBoundingClientRect
                ? node.getBoundingClientRect() : null;
            if (box) this.state.floatAt = fractionOf(box, viewport());
        }
        this._saveFloat();
        this._save();
        this.placeNow();
        return this.state.freeFloat;
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

    // Get out of the way of somebody else's dialog.
    //
    // Reported against Mini Paint NEO's Send to WanGP popup: the assistant's
    // launcher sat on top of it, and in focus mode the focused workspace
    // covered it outright. Both were stacking order -- that popup drew at
    // `z-index: 60` against this extension's 1100 and 1200 -- and it is fixed
    // on that side, where the number belongs. What is left is the part only
    // this side can do: a panel covering a dialog is still covering it,
    // whatever the numbers say, so the panel puts itself away.
    //
    // Back to the launcher, not gone: the launcher is small, it is what the
    // panel is minimised to by its own ✕, and it leaves the way back exactly
    // where it always is. Nothing is reopened when the dialog closes -- a
    // panel that springs back over the page somebody has returned to is the
    // same complaint from the other end.
    //
    // Focus mode is deliberately untouched. The dialog draws above it now, so
    // there is nothing to leave, and a workspace that emptied itself every
    // time a popup opened would be worse than the bug.
    Shell.prototype.yieldTo = function (event) {
        const detail = (event && event.detail) || {};
        if (!detail.open) return false;
        if (!this.state.panelOpen) return false;
        this.closeMenu();
        this.close();
        return true;
    };

    Shell.prototype.documentKey = function (event) {
        if (event.key !== "Escape") return;
        // The customize dialog's Escape is its own: it cancels the edit. Taken
        // here it would have left focus mode instead, behind the dialog. The
        // system prompt editor's is its own for the same reason: it asks
        // before an unapplied edit is thrown away.
        if (this.lookEditor && this.lookEditor.isOpen()) return;
        if (NS.systemEditor && NS.systemEditor.isOpen()) return;
        const context = {
            composing: event.isComposing || event.keyCode === 229,
            nativeDialog: false,
            hostDialogOpen: NS.hostInDialog(document.activeElement),
            insideAssistant: this.nodes.root
                && this.nodes.root.contains(document.activeElement),
            assistantMenuOpen: this.nodes.menu && !this.nodes.menu.hidden,
            // Only from inside the panel: an Escape pressed in Forge's own
            // prompt box is that box's, and taking it would drop an edit
            // nobody was looking at.
            assistantEditing: !!this.editing && !!(this.nodes.root
                && this.nodes.root.contains(document.activeElement)),
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

    // The edge handle. It used to add a pointermove and a pointerup listener
    // to the window per gesture and take them off only on pointerup -- so a
    // gesture that ended any other way (a touch the browser cancelled, a
    // release outside the window) left them there, and from then on every
    // pointer movement anywhere on the page resized the panel and forced a
    // layout doing it. Now it is the same machine as the other two: one set of
    // listeners for the life of the shell, a pointer id, capture, and an end
    // that every way out reaches.
    Shell.prototype.startResize = function (event) {
        if (event.button !== undefined && event.button !== 0) return;
        if (!event.isPrimary) return;
        this.resizing = {
            pointerId: event.pointerId,
            start: event.clientX,
            from: this.state.panelWidth || this.nodes.panel.offsetWidth || NOMINAL_WIDTH,
        };
        try {
            this.nodes.resize.setPointerCapture(event.pointerId);
        } catch (error) { /* capture is an optimisation, not a requirement */ }
        event.preventDefault();
    };

    Shell.prototype.moveResize = function (event) {
        const resizing = this.resizing;
        if (!resizing || event.pointerId !== resizing.pointerId) return;
        const width = resizing.from + (resizing.start - event.clientX);
        this.state.panelWidth = Math.min(MAX_WIDTH, Math.max(MIN_WIDTH, width));
        // Once a frame, not once an event: a pointer reports far more often
        // than the screen draws, and every placement reads the panel's size.
        this.place();
    };

    Shell.prototype.endResize = function (event, cancelled) {
        const resizing = this.resizing;
        if (!resizing || (event && event.pointerId !== resizing.pointerId)) return;
        // Cleared before capture is released, so the `lostpointercapture` that
        // releasing causes finds nothing to end.
        this.resizing = null;
        try {
            this.nodes.resize.releasePointerCapture(resizing.pointerId);
        } catch (error) { /* never held, or already released */ }
        // A cancelled resize keeps the width it reached: it was on screen the
        // whole time, and snapping back would be the surprise.
        void cancelled;
        this._save();
        this.placeNow();
    };

    // -- attachments ---------------------------------------------------------- //

    Shell.prototype.paste = function (event) {
        const data = event.clipboardData;
        if (!data || !data.items) return;
        const images = Array.prototype.filter.call(data.items, (item) =>
            item.kind === "file" && /^image\/(png|jpeg|webp)$/.test(item.type));
        if (!images.length) return;          // ordinary text paste, untouched
        if (this.editing) {
            // An edit changes the words. The picture on the message stays as
            // it is, and one pasted now would land in the draft unseen.
            event.preventDefault();
            this.tell("A picture cannot be added while you edit a message.", "warn");
            return;
        }
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
        if (this.editing) {
            this.saveEdit();
            return;
        }
        const view = this.store.snapshot();
        if (!this.canSend(view)) return;
        this.store.setDraftText(this.nodes.input.value);
        const picked = this.autoAttachable(this.store.snapshot());
        if (picked.note) this.tell(picked.note, picked.kind);
        if (!picked.picture) {
            this.submitDraft(null);
            return;
        }
        // The picture first, then the message -- through the same upload the
        // paperclip uses, so it arrives exactly as one attached by hand does.
        // Send is disabled while it uploads (an attachment that is not ready
        // cannot be sent), so a second press cannot send the words without it.
        const key = NS.conversationKey(view.selection.character, view.selection.thread);
        const epoch = view.selection.epoch;
        this.attachShowing(picked.picture, key).then((attached) => {
            if (this.store.snapshot().selection.epoch !== epoch) {
                this.tell("The conversation changed while the picture was attaching. "
                          + "It is in that conversation's draft; press Send there.", "warn");
                return;
            }
            if (!attached) {
                this.tell("Auto Attach could not attach the picture from "
                          + picked.picture.label + (this.attachFailure
                          ? " (" + this.attachFailure + ")" : "")
                          + ". Sent without it.", "warn");
            }
            this.submitDraft(attached ? picked.picture : null);
        });
    };

    Shell.prototype.submitDraft = function (picture) {
        const view = this.store.snapshot();
        const key = NS.conversationKey(view.selection.character, view.selection.thread);
        this.store.submit().then((outcome) => {
            if (outcome && outcome.ok && picture) this.autoSent.set(key, picture.src);
            if (outcome && outcome.ok) this.emptyComposer(key);
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

    /** The box, emptied once what was in it has been sent.
     *
     * `render` never rewrites the box while it has focus -- a redraw arriving
     * while somebody types must not take their words away -- and a message
     * sent with Enter is a message sent from a box that still has focus. So
     * the store emptied the draft and the box went on showing it: "if i hit
     * enter ... the prompt stays in the input field". Send by the button moved
     * focus to the button first, which is why that way worked.
     *
     * Emptied only when the store emptied the draft, which it does only if
     * nothing was typed while the send was in flight: anything typed then is
     * newer than the send, and stays.
     */
    Shell.prototype.emptyComposer = function (key) {
        const input = this.nodes.input;
        if (!input || this.store.draft(key).text) return false;
        input.value = "";
        this.grow();
        return true;
    };

    // -- Auto Attach ---------------------------------------------------------- //
    //
    // "When conversation mode is open, I want an option to enable the current
    // visible image in the text to image or image to image tab (which ever is
    // open at the time) to be an automatic input to the next LLM prompt ... If
    // i am not on either of those tabs, or gallery is empty, then nothing
    // should be attached."
    //
    // "Visible" is Forge's own answer to "which picture", the one its Send to
    // img2img buttons use: the picture you clicked in the gallery, and the
    // first one when you have not clicked any. Read off the gallery rather than
    // asked of Python, because which thumbnail is selected is a fact only the
    // page knows.
    //
    // Three reasons a picture is left out even with the mode on, each said:
    //
    // * a picture of your own is already attached -- that one is sent;
    // * the model running cannot see pictures -- the server refuses any
    //   message carrying one, so every message would bounce;
    // * it is the same picture Auto Attach sent last time in this conversation
    //   and the conversation still has a picture in it. The model keeps up to
    //   four pictures of a thread's history in view (`prompt_master/chat/
    //   prompt.py`, MAX_IMAGES), so a second copy of the same one is context
    //   spent on nothing. A new picture in the gallery is attached as usual.

    const GALLERIES = {
        tab_txt2img: {id: "txt2img_gallery", label: "txt2img"},
        tab_img2img: {id: "img2img_gallery", label: "img2img"},
    };

    // In order: Gradio's preview of the selected picture, the selected
    // thumbnail, the first thumbnail, and -- for a theme that renames all of
    // those -- the first picture in the gallery that is not Forge's live
    // preview of a generation still running.
    const SHOWING = [".preview img[data-testid=\"detailed-image\"]",
                     ".preview .media-button img",
                     ".thumbnail-item.selected img",
                     ".thumbnail-item img"];

    function showingIn(gallery) {
        for (let at = 0; at < SHOWING.length; at += 1) {
            const found = gallery.querySelector(SHOWING[at]);
            if (found) return found;
        }
        const all = typeof gallery.querySelectorAll === "function"
            ? gallery.querySelectorAll("img") : [];
        return Array.prototype.find.call(all, (image) =>
            !(typeof image.closest === "function" && image.closest(".livePreview"))) || null;
    }

    function fileNameOf(src) {
        const path = String(src || "").split(/[?#]/)[0];
        const last = path.slice(path.lastIndexOf("/") + 1);
        let name = last;
        try {
            name = decodeURIComponent(last);
        } catch (error) { /* a name that is not valid percent-encoding stays as it is */ }
        name = name.replace(/^file=/, "");
        name = name.slice(name.lastIndexOf("/") + 1);
        return name || "image.png";
    }

    /** The picture showing in this workspace's gallery, or null. */
    function showingPicture(workspace) {
        const where = GALLERIES[workspace];
        if (!where) return null;
        const gallery = hostElement(where.id);
        if (!gallery || typeof gallery.querySelector !== "function") return null;
        const image = showingIn(gallery);
        const src = image && (image.currentSrc || image.src
            || (typeof image.getAttribute === "function" ? image.getAttribute("src") : ""));
        if (!src) return null;
        return {src: String(src), label: where.label, name: fileNameOf(src)};
    }

    /** What Auto Attach would do with this send: `{picture}` to attach one,
     *  and a `note` whenever it leaves one out for a reason worth saying. */
    Shell.prototype.autoAttachable = function (view) {
        if (!this.state.autoAttach) return {};
        if (view.draft && view.draft.attachment) return {};
        const picture = showingPicture(this.host.getActiveWorkspace());
        if (!picture) return {};
        if (view.capabilities && view.capabilities.vision === false) {
            return {note: "Auto Attach left the picture out: the model running cannot "
                          + "see pictures.", kind: "warn"};
        }
        const key = NS.conversationKey(view.selection.character, view.selection.thread);
        const messages = (view.conversation && view.conversation.messages) || [];
        if (this.autoSent.get(key) === picture.src
            && messages.some((message) => message.attachment)) {
            return {note: "Same picture as last time, so it was not attached again.",
                    kind: "info"};
        }
        return {picture};
    };

    // What the server takes as it is, and how large. Anything else -- a format
    // Forge was told to save in that staging does not read, or an upscale past
    // the limit -- is drawn onto a canvas and sent as a JPEG instead: the model
    // sees a few hundred pixels of it whatever it is sent.
    const STAGEABLE = /^image\/(png|jpeg|webp)$/;
    const STAGE_LIMIT = 19 * 1024 * 1024;
    const REENCODE_EDGE = 2048;

    function stageable(blob) {
        if (STAGEABLE.test(blob.type || "") && blob.size <= STAGE_LIMIT) {
            return Promise.resolve(blob);
        }
        if (typeof createImageBitmap !== "function") return Promise.resolve(blob);
        return createImageBitmap(blob).then((bitmap) => {
            const scale = Math.min(1, REENCODE_EDGE / Math.max(bitmap.width, bitmap.height, 1));
            const canvas = document.createElement("canvas");
            canvas.width = Math.max(1, Math.round(bitmap.width * scale));
            canvas.height = Math.max(1, Math.round(bitmap.height * scale));
            canvas.getContext("2d").drawImage(bitmap, 0, 0, canvas.width, canvas.height);
            return new Promise((resolve) => {
                canvas.toBlob((made) => resolve(made || blob), "image/jpeg", 0.92);
            });
        }).catch(() => blob);
    }

    /** Fetch the showing picture and stage it as this draft's attachment.
     *
     * Resolves true when it is ready to send. The chip shows it the whole
     * way, marked with where it came from. A picture removed from the chip
     * while it was uploading stays removed.
     */
    Shell.prototype.attachShowing = function (picture, key) {
        const localId = NS.uuid();
        const mark = {localId, name: picture.name, preview: picture.src, from: picture.label};
        this.attachFailure = "";
        this.store.setAttachment(Object.assign({state: "uploading"}, mark), key);
        const still = () => {
            const current = this.store.draft(key).attachment;
            return !!current && current.localId === localId;
        };
        return fetch(picture.src, {credentials: "same-origin"}).then((response) => {
            if (!response.ok) throw new Error("the gallery answered " + response.status);
            return response.blob();
        }).then(stageable).then((blob) => {
            const name = /jpeg/.test(blob.type || "") && !/\.jpe?g$/i.test(picture.name)
                ? picture.name.replace(/\.[^.]*$/, "") + ".jpg" : picture.name;
            return this.store.upload(new File([blob], name, {type: blob.type || "image/png"}));
        }).then((found) => {
            if (!still()) return false;
            this.store.setAttachment(Object.assign({state: "ready", token: found.token},
                                                   mark, {name: found.name || picture.name}),
                                     key);
            return true;
        }).catch((error) => {
            this.attachFailure = (error && error.message) || "";
            if (still()) this.store.setAttachment(null, key);
            return false;
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

    /** The read-aloud switch, pressed.
     *
     * It used to flip its own attribute and press Voice Chat's checkbox in the
     * LLM Studio tab -- and start out reading "off" whatever that checkbox
     * said. Pressed while speech was on, it turned on something already on;
     * pressed where the checkbox was not on the page, it changed nothing at
     * all. Either way the replies went on being spoken and synthesised.
     *
     * Now the server is told (`Store.setReadAloud`) and the switch shows what
     * the server says. Off also silences this page at once, before the round
     * trip, and tells the server to stop the reply being spoken. The tab's
     * checkbox is brought into line afterwards, so the two views agree.
     */
    Shell.prototype.toggleReadAloud = function () {
        const view = this.store.snapshot();
        const on = view.readAloud !== true;
        if (!on && NS.speech && typeof NS.speech.stopPlayback === "function") {
            NS.speech.stopPlayback({origin: "any"});
        }
        return this.store.setReadAloud(on).then((stored) => {
            if (NS.speech && typeof NS.speech.setAutomaticReadAloud === "function") {
                NS.speech.setAutomaticReadAloud(stored);
            }
            this.tell(stored ? "Replies will be read aloud." : "Replies will not be read aloud.",
                      "info");
            return stored;
        }).catch((error) => {
            this.tell((error && error.message) || "Read aloud could not be changed.", "warn");
            return null;
        });
    };

    /** On and off, so that nobody has to guess which.
     *
     * A different glyph for each -- a speaker, and a speaker struck through --
     * and the stylesheet lights the on state with the accent and takes the
     * colour out of the off one, because the report was a switch that "does
     * not change state when i click it": an attribute nothing drew is not a
     * state anybody can see.
     * The words are in the tooltip and the accessible name as well, for
     * whoever the glyphs say nothing to. Unknown -- the server has not said
     * yet -- is drawn as unknown rather than as off.
     */
    Shell.prototype.renderReadAloud = function (view) {
        const button = this.nodes.readAloud;
        if (!button) return;
        const known = view.readAloud === true || view.readAloud === false;
        const on = view.readAloud === true;
        const state = known ? (on ? "on" : "off") : "unknown";
        if (button.dataset.state === state) return;
        button.dataset.state = state;
        button.setAttribute("aria-checked", String(on));
        button.textContent = on ? "\u{1F50A}" : (known ? "\u{1F507}" : "\u{1F508}");
        button.title = known ? "Read replies aloud: " + (on ? "On" : "Off")
            : "Read replies aloud";
    };

    // -- rendering ---------------------------------------------------------------- //

    Shell.prototype.say = function (text, kind) {
        const status = this.nodes.status;
        if (!status) return;
        status.textContent = text;
        status.dataset.kind = kind || "info";
    };

    /** Say something about a press, and keep it on screen long enough to read.
     *
     * `say` alone is overwritten by the next render, and a press that changes
     * the store *causes* the next render -- a frame later the line is back to
     * "Ready." and the answer to the press was never seen. This holds the
     * line over the idle sentences for TOLD_FOR, and a warning over the
     * progress of a reply as well -- the press that left a picture out is
     * usually the press that started the reply. An error from the server and
     * a lost connection still take the line at once.
     */
    Shell.prototype.tell = function (text, kind) {
        this.told = {text, kind: kind || "info", at: Date.now()};
        this.say(text, kind);
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

        if (this.editing && this.editing.key !== NS.conversationKey(
            view.selection.character, view.selection.thread)) {
            // The message being edited is in a conversation that is no longer
            // on screen, and Save would be aimed at it from this one. The box
            // takes the draft of the conversation that is -- even focused,
            // which the ordinary sync below leaves alone.
            this.finishEdit(view.draft.text || "");
            this.tell("The conversation changed, so the edit was put away.", "warn");
        }
        if (!this.editing && nodes.input.value !== view.draft.text
            && document.activeElement !== nodes.input) {
            nodes.input.value = view.draft.text || "";
            this.grow();
        }
        this.renderSelector(view);
        this.renderChip(view.draft.attachment);
        this.renderAutoAttach();
        this.renderReadAloud(view);
        this.renderTranscript(view);
        this.renderStatus(view);
        nodes.send.disabled = this.editing ? !this.canSaveEdit(view) : !this.canSend(view);
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
        const said = states[attachment.state] || attachment.state;
        chip.appendChild(element("span", "forge-assistant-chip-state",
                                 attachment.from ? "From " + attachment.from + " \u00b7 " + said
                                     : said));
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
        if (this.revealed) this.reveal(this.revealed);
        this.markEditTarget();

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
        if (row.attachment) node.appendChild(this.figure(row.attachment));
        const bar = this.actions(row, view);
        node.appendChild(bar);
        // A bubble with anything to offer is something you tap. Reachable by
        // keyboard too: focusable, and Enter or Space does what a tap does.
        if (bar.children.length) {
            node.dataset.actions = String(bar.children.length);
            node.setAttribute("tabindex", "0");
            node.setAttribute("aria-expanded", "false");
        }
        return node;
    };

    /** A message's picture -- or, where there is none to show, a placeholder.
     *
     * Never the file's name. The picture used to be an <img> pointed straight
     * at its ticket, which the route refuses without the page's key, so it
     * never loaded: the browser drew the alternative text -- the name -- and a
     * caption under it said the name again. The store fetches it with the key
     * (`Store.picture`); until it arrives the frame is empty, and if it cannot
     * arrive, or the file is gone, the frame says so without naming anything.
     */
    Shell.prototype.figure = function (attachment) {
        const figure = element("figure", "forge-assistant-attachment");
        let image = null;
        // Once: a fetch that fails and a picture that will not decode can both
        // report, and the frame says it once.
        const missing = () => {
            if (figure.classList.contains("forge-assistant-attachment-missing")) return;
            if (image) {
                try {
                    figure.removeChild(image);
                } catch (error) { /* not in the frame */ }
                image = null;
            }
            figure.classList.add("forge-assistant-attachment-missing");
            figure.appendChild(element("span", "forge-assistant-attachment-note",
                                       "\u{1F5BC} Picture unavailable"));
        };
        const store = this.store;
        if (!attachment.url || !attachment.available || !store
            || typeof store.picture !== "function") {
            missing();
            return figure;
        }
        image = element("img");
        image.alt = "Attached picture";
        image.addEventListener("error", missing);
        figure.appendChild(image);
        store.picture(attachment.url).then((made) => {
            if (image) image.src = made;
        }, missing);
        return figure;
    };

    Shell.prototype.updateBubble = function (node, row) {
        const text = node.querySelector(".forge-assistant-text");
        if (text) text.innerHTML = renderMarkdown(row.text);
        node.classList.toggle("forge-assistant-provisional", !!row.provisional);
    };

    // The actions the panel offers, and which message each belongs on.
    //
    // There were eight of them and a version pager, on every message. At panel
    // width that wrapped onto three rows under every bubble, so a thread was
    // more chrome than conversation, and the ones aimed at a message in the
    // middle of it were the heavy ones -- branch, truncate, renumber -- which
    // want the room the tab has to explain what they are about to do. So the
    // panel keeps what you want on the thing you just said or just read. The
    // rest of the set is unchanged in the tab: nothing here removes an action
    // from the conversation, only from this view of it.
    //
    // And none of them is drawn until you ask. They sat under the newest
    // message all the time, which was asked to stop: "i want these buttons to
    // only show up when i tap on the reply". A tap on a message shows its row;
    // a tap anywhere else, or on the message again, puts it away. See
    // `tapBubble`.
    //
    // `newest` is the newest message only; `role` is the kind of message.
    //
    // * Edit, Regenerate and Delete are the newest message's, as they were.
    //   Regenerate is a reply's: on an unanswered message of yours there is no
    //   reply to ask for again.
    // * Send to prompt is every reply's. It changes nothing in the conversation
    //   -- it writes the reply into the image prompt -- so there is no reason
    //   to keep it off an older one. See `sendToPrompt`.
    // * Send again is your own newest message's: the state a deleted, stopped
    //   or failed reply leaves, where the thread ends with you and the only
    //   useful thing left is to ask again. The server answers that message in
    //   place (`_plan_resend`); further up the thread it would branch, and the
    //   tab is where branching is explained.
    //
    // `\u21bb` is the same glyph the tab draws on a reply for Regenerate
    // (`javascript/llm_studio.js`), so the two views agree on what it means.
    // `\u27a4` is the send arrow asked for ("just looks like a send icon on a
    // button"); Send again's hooked arrow is a send that goes round again.
    const ACTIONS = [
        {action: "edit_message", glyph: "\u270e", label: "Edit", newest: true},
        {action: "regenerate", glyph: "\u21bb", label: "Regenerate", newest: true,
         role: "assistant"},
        {action: "delete_message", glyph: "\u2715", label: "Delete", newest: true},
        {action: "send_prompt", glyph: "\u27a4", label: "Send to prompt",
         role: "assistant"},
        {action: "resend_from_user", glyph: "\u21aa", label: "Send again", newest: true,
         role: "user"},
    ];

    // Each button carries the (index, version, revision) captured at the moment
    // it was drawn -- a stale one is refused by the server rather than quietly
    // reinterpreted as the current last message. A reply still being written
    // has nothing to offer yet.
    Shell.prototype.actions = function (row, view) {
        const bar = element("div", "forge-assistant-actions");
        bar.hidden = true;
        if (row.provisional) return bar;
        const messages = (view.conversation && view.conversation.messages) || [];
        const newest = messages.length > 0 && row.index === messages.length - 1;
        const revision = view.conversation && view.conversation.conversation
            && view.conversation.conversation.revision;
        const role = row.role === "assistant" ? "assistant" : "user";
        ACTIONS.forEach((spec) => {
            if (spec.newest && !newest) return;
            if (spec.role && spec.role !== role) return;
            const button = element("button", "forge-assistant-action", spec.glyph);
            button.type = "button";
            button.dataset.action = spec.action;
            // An icon with no accessible name is a button only sighted people
            // have. Both, because `title` is the hover and the long press and
            // `aria-label` is what a screen reader reads.
            button.title = spec.label;
            button.setAttribute("aria-label", spec.label);
            if (spec.action === "send_prompt") this.drawSendMode(button);
            button.addEventListener("click", () => {
                // Pressed is chosen: the row goes away, whatever the action
                // does next.
                this.reveal("");
                this.act(spec.action, row, revision);
            });
            bar.appendChild(button);
        });
        return bar;
    };

    // With Send to Generate on, the reply's send button says so -- a play
    // glyph and its own name -- because the same press now starts a
    // generation, and that should never be a surprise. U+FE0E keeps the glyph
    // text rather than a coloured emoji.
    const SEND_PROMPT = {glyph: "\u27a4", label: "Send to prompt"};
    const SEND_GENERATE = {glyph: "\u25b6\ufe0e", label: "Send to Generate"};

    Shell.prototype.drawSendMode = function (button) {
        const mode = this.state.sendToGenerate ? SEND_GENERATE : SEND_PROMPT;
        if (button.textContent !== mode.glyph) button.textContent = mode.glyph;
        button.title = mode.label;
        button.setAttribute("aria-label", mode.label);
        button.dataset.generate = String(!!this.state.sendToGenerate);
    };

    /** Redraw every reply's send button: bubbles are kept across renders, so
     *  the ones already on the page are told when the switch moves. */
    Shell.prototype.renderSendMode = function () {
        const transcript = this.nodes.transcript;
        if (!transcript) return;
        Array.prototype.forEach.call(transcript.children || [], (node) => {
            const bar = actionsOf(node);
            if (!bar) return;
            Array.prototype.forEach.call(bar.children || [], (button) => {
                if (button.dataset && button.dataset.action === "send_prompt") {
                    this.drawSendMode(button);
                }
            });
        });
    };

    function actionsOf(node) {
        return Array.prototype.find.call(node.children || [], (child) =>
            child.classList && child.classList.contains("forge-assistant-actions")) || null;
    }

    /** A tap on the transcript: show a message's actions, or put them away.
     *
     * Delegated from the transcript rather than bound per bubble, because
     * bubbles are rebuilt whenever the thread moves and a listener on each is a
     * listener to lose. A press on a control or a link inside the bubble is
     * that control's, and text being selected is somebody reading, not
     * tapping -- neither toggles anything.
     */
    Shell.prototype.tapBubble = function (event) {
        const target = event && event.target;
        if (!target || typeof target.closest !== "function") return false;
        if (target.closest("button, a, input, textarea, select")) return false;
        const node = target.closest(".forge-assistant-bubble");
        if (!node || !node.dataset || !node.dataset.actions) return false;
        const selection = typeof window.getSelection === "function"
            ? window.getSelection() : null;
        if (selection && !selection.isCollapsed && selection.anchorNode
            && typeof node.contains === "function" && node.contains(selection.anchorNode)) {
            return false;
        }
        const key = node.dataset.key || "";
        this.reveal(this.revealed === key ? "" : key);
        return true;
    };

    /** Enter or Space on a focused message does what a tap does. */
    Shell.prototype.bubbleKey = function (event) {
        if (!event || (event.key !== "Enter" && event.key !== " ")) return false;
        const node = event.target;
        if (!node || !node.dataset || !node.dataset.actions) return false;
        if (!node.classList || !node.classList.contains("forge-assistant-bubble")) return false;
        event.preventDefault();
        const key = node.dataset.key || "";
        this.reveal(this.revealed === key ? "" : key);
        return true;
    };

    /** A press anywhere but on the open message puts its actions away.
     *
     * In the capture phase, so it runs before the press lands: a tap on
     * another message closes this one here and opens that one in `tapBubble`,
     * and a tap on the open message itself is left to `tapBubble` to close.
     */
    Shell.prototype.dismissActions = function (event) {
        if (!this.revealed) return false;
        const transcript = this.nodes.transcript;
        const open = transcript && Array.prototype.find.call(transcript.children || [],
            (node) => node.dataset && node.dataset.key === this.revealed);
        const target = event && event.target;
        if (open && target && typeof open.contains === "function" && open.contains(target)) {
            return false;
        }
        this.reveal("");
        return true;
    };

    /** Show the actions of the message with this key, and nobody else's.
     *
     * Kept as a key rather than as a node, and applied again after every
     * redraw: a reply arriving token by token redraws the transcript once a
     * frame, and a row that closed whenever a word arrived could not be used
     * while a reply was being written. A key that is no longer on the page --
     * the thread moved, the action ran -- shows nothing and is forgotten.
     */
    Shell.prototype.reveal = function (key) {
        this.revealed = key || "";
        const transcript = this.nodes.transcript;
        if (!transcript) return;
        let found = null;
        Array.prototype.forEach.call(transcript.children || [], (node) => {
            if (!node.dataset || !node.dataset.actions) return;
            const open = !!this.revealed && node.dataset.key === this.revealed;
            if (open) found = node;
            const bar = actionsOf(node);
            if (bar && bar.hidden === open) bar.hidden = !open;
            if (node.classList) node.classList.toggle("forge-assistant-revealed", open);
            if (node.getAttribute && node.getAttribute("aria-expanded") !== String(open)) {
                node.setAttribute("aria-expanded", String(open));
            }
        });
        if (!found) {
            this.revealed = "";
            return;
        }
        // A row opened under the last message would open below the fold.
        if (typeof found.getBoundingClientRect === "function"
            && typeof transcript.getBoundingClientRect === "function") {
            const box = found.getBoundingClientRect();
            const edge = transcript.getBoundingClientRect();
            if (box.bottom > edge.bottom) transcript.scrollTop += box.bottom - edge.bottom;
        }
    };

    // -- Send to prompt --------------------------------------------------------- //
    //
    // "The send prompt button ... should take that reply, and automatically
    // replace the current prompt in the positive prompt field. It should
    // replace everything except lora or content inside a literal command ...
    // prepend the LLM prompt following by a new line for separation. Pressing
    // the SEND PROMPT button should just apply the text work, it should not
    // cause an image to be auto generated."
    //
    // What is kept is what this extension already refuses to let a language
    // model rewrite, read the same way:
    //
    // * a literal command -- `[[...]]`, `+[[...]]` or `-[[...]]` -- exactly as
    //   typed. The grammar is `prompt_master/krea/literals.py`'s: the first
    //   `]]` closes, an empty one carries nothing, and one never closed is
    //   ordinary text (so it goes, like any other text);
    // * an extra-network tag -- `<lora:...>`, `<lyco:...>`, `<hypernet:...>`,
    //   any case -- outside a literal command. The closed list is
    //   `prompt_master/krea/extra_networks.py`'s, and for its reason: an open
    //   `<word:...>` shape would keep other people's syntax by accident.
    //
    // Kept in the order they were in, one space apart, on a line after the
    // reply. The reply goes in as written: it is the reply you chose, and
    // guessing which of its sentences are "the prompt" is how a sentence you
    // wanted disappears.

    const LITERAL_OPEN = "[[";
    const LITERAL_CLOSE = "]]";
    const EXTRA_NETWORK = /<(?:lora|lyco|hypernet):[^<>]*>/gi;

    function keptFromPrompt(prompt) {
        const source = String(prompt || "");
        const kept = [];
        const spans = [];
        let position = 0;
        for (;;) {
            const start = source.indexOf(LITERAL_OPEN, position);
            if (start < 0) break;
            const end = source.indexOf(LITERAL_CLOSE, start + LITERAL_OPEN.length);
            if (end < 0) break;
            const signed = start > 0 && (source[start - 1] === "+" || source[start - 1] === "-");
            const from = signed ? start - 1 : start;
            const to = end + LITERAL_CLOSE.length;
            if (source.slice(start + LITERAL_OPEN.length, end).trim()) {
                kept.push({at: from, text: source.slice(from, to)});
            }
            spans.push([from, to]);
            position = to;
        }
        EXTRA_NETWORK.lastIndex = 0;
        let match = EXTRA_NETWORK.exec(source);
        while (match) {
            const at = match.index;
            if (!spans.some((span) => at >= span[0] && at < span[1])) {
                kept.push({at, text: match[0]});
            }
            match = EXTRA_NETWORK.exec(source);
        }
        kept.sort((a, b) => a.at - b.at);
        return kept.map((item) => item.text);
    }

    function promptFrom(reply, current) {
        const body = String(reply || "").trim();
        const kept = keptFromPrompt(current);
        if (!kept.length) return body;
        return body ? body + "\n" + kept.join(" ") : kept.join(" ");
    }

    // Which prompt: the image tab you are on, and txt2img from anywhere else.
    const PROMPT_TARGETS = {
        tab_img2img: {id: "img2img_prompt", label: "img2img", tab: "tab_img2img",
                      name: "img2img"},
    };
    const DEFAULT_PROMPT_TARGET = {id: "txt2img_prompt", label: "txt2img",
                                   tab: "tab_txt2img", name: "txt2img"};

    function editHeight(lineHeight) {
        const view = window.visualViewport;
        const high = (view && view.height) || window.innerHeight || 800;
        return Math.max(lineHeight * 6 + 12,
                        Math.min(lineHeight * 14 + 12, Math.round(high * 0.4)));
    }

    function promptTarget(workspace) {
        return PROMPT_TARGETS[workspace] || DEFAULT_PROMPT_TARGET;
    }

    function hostElement(id) {
        const app = typeof gradioApp === "function" ? gradioApp() : null;
        return (app && typeof app.querySelector === "function"
            && app.querySelector("#" + id)) || document.getElementById(id);
    }

    function promptBox(id) {
        const holder = hostElement(id);
        if (!holder) return null;
        if (holder.tagName === "TEXTAREA") return holder;
        return typeof holder.querySelector === "function"
            ? holder.querySelector("textarea") : null;
    }

    // Gradio binds to the input event, so setting `value` alone changes the
    // page and tells Gradio nothing -- the next Generate would send the old
    // prompt. Forge ships `updateInput` for exactly this; the fallback is what
    // it does. Nothing here presses anything: an input event is typing.
    function publish(box, value) {
        box.value = value;
        if (typeof updateInput === "function") {
            updateInput(box);
            return;
        }
        box.dispatchEvent(new Event("input", {bubbles: true}));
    }

    Shell.prototype.sendToPrompt = function (row) {
        if (this.state.sendToGenerate) return this.sendToGenerate(row);
        const target = promptTarget(this.host.getActiveWorkspace());
        const box = promptBox(target.id);
        if (!box) {
            this.tell("There is no " + target.label + " prompt on this page.", "warn");
            return false;
        }
        if (!String(row.text || "").trim()) {
            this.tell("That reply has no words to send.", "warn");
            return false;
        }
        publish(box, promptFrom(row.text, box.value));
        this.tell("Prompt sent to " + target.label + ".", "info");
        return true;
    };

    // -- Send to Generate -------------------------------------------------------- //
    //
    // "I want a toggle that enhances the "send to prompt" with a "Send to
    // Generate" ... pressing the button under a reply would send the prompt in
    // to replace the current, and invoke a generation as if user pressed the
    // button. All the settings on the page still applied, just generated with
    // the LLM reply as prompt (respecting our lora and literals)."
    //
    // The prompt is written exactly as Send to prompt writes it -- the reply,
    // then what is kept of the old prompt -- and then the tab's own Generate
    // is pressed. Pressed, not imitated: everything that button does with the
    // page, the pipeline, the literal boxes and whatever else is on it happens
    // because it is the button. Four details.
    //
    // * Where. The image tab you are on; from any other tab, txt2img, and the
    //   panel switches there first so the generation is watched rather than
    //   started out of sight. Focus mode comes along, as it does for the
    //   workspace picker. A switch that fails still generates, and says where.
    // * When. Gradio reads the prompt from its own state, which the input
    //   event updates; the press waits for the page to paint twice, so the
    //   generation is of the new prompt and never of the old one. If the box
    //   no longer holds what was written by then, nothing is pressed.
    // * Not twice. Forge covers Generate with Interrupt and Skip while a run
    //   is on, so a person cannot press it then; neither does this. The
    //   prompt is left in place and the status line says to press Generate
    //   when the run ends.
    // * Honest. Every way it stops short says so, and what it did do.

    Shell.prototype.sendToGenerate = function (row) {
        if (!String(row.text || "").trim()) {
            this.tell("That reply has no words to send.", "warn");
            return Promise.resolve(false);
        }
        const active = this.host.getActiveWorkspace();
        const target = promptTarget(active);
        const arrive = active === target.tab ? Promise.resolve(target.tab)
            : this.switchWorkspace(target.tab);
        return arrive.then((reached) => {
            const box = promptBox(target.id);
            if (!box) {
                this.tell("There is no " + target.label + " prompt on this page.", "warn");
                return false;
            }
            const wanted = promptFrom(row.text, box.value);
            publish(box, wanted);
            const generate = hostElement(target.name + "_generate");
            if (!generate) {
                this.tell("Prompt sent to " + target.label + ", but there is no Generate "
                          + "button to press on this page.", "warn");
                return false;
            }
            return afterPaint().then(() => {
                if (box.value !== wanted) {
                    this.tell("The " + target.label + " prompt changed before Generate "
                              + "was pressed, so it was not pressed.", "warn");
                    return false;
                }
                // Asked at the moment of the press, which is the moment that
                // matters: a run can start in the frames just waited.
                if (generating(target.name) || generate.disabled) {
                    this.tell(target.label + " is already generating. The prompt is in "
                              + "place: press Generate when it finishes.", "warn");
                    return false;
                }
                generate.click();
                this.tell(reached === target.tab || active === target.tab
                    ? "Prompt sent to " + target.label + " and generating."
                    : "Prompt sent to " + target.label + " and generating there -- its "
                      + "tab could not be opened.", "info");
                return true;
            });
        });
    };

    // A run is on while Forge shows Interrupt, Skip or Interrupting... over
    // Generate (`setSubmitButtonsVisibility` writes their display), or has
    // hidden Generate itself. Computed, not inline: idle, the stylesheet is
    // what hides them.
    function generating(name) {
        const shown = (id) => {
            const node = hostElement(id);
            if (!node) return null;
            const style = typeof window.getComputedStyle === "function"
                ? window.getComputedStyle(node) : null;
            const display = (style && style.display) || (node.style && node.style.display);
            return display ? display !== "none" : null;
        };
        if (["_interrupt", "_skip", "_interrupting"].some((suffix) =>
            shown(name + suffix) === true)) return true;
        return shown(name + "_generate") === false;
    }

    // Two frames, or a quarter of a second in a tab the browser is not
    // painting: long enough for Gradio to have taken the input event, short
    // enough that the press still feels like the press.
    function afterPaint() {
        return new Promise((resolve) => {
            let done = false;
            const finish = () => {
                if (done) return;
                done = true;
                resolve();
            };
            if (typeof window.requestAnimationFrame === "function") {
                window.requestAnimationFrame(() => window.requestAnimationFrame(finish));
            }
            setTimeout(finish, 250);
        });
    }

    Shell.prototype.act = function (action, row, revision) {
        if (action === "edit_message") {
            this.startEdit(row, revision);
            return;
        }
        if (action === "send_prompt") {
            this.sendToPrompt(row);
            return;
        }
        const envelope = this.store.envelope(action, {
            target: {index: row.index, version: row.active},
            expected_revision: revision === undefined || revision === null ? null
                : (typeof revision === "number" ? {kind: "revision", value: revision}
                    : {kind: "legacy", fingerprint: String(revision)}),
        });
        envelope.payload = {};
        this.store.send(envelope).then((outcome) => {
            if (outcome && !outcome.ok) {
                this.tell((outcome.error && outcome.error.message)
                    || "That could not be done.", "warn");
            }
        });
    };

    // -- Editing a message ------------------------------------------------------ //
    //
    // "This is what happens when i try to edit a prompt in the flyout view ...
    // I dont want the browser doing this, i need UI in our flyout ... make sure
    // it feels clear that I am editing a previous message, not simply
    // submitting a new one."
    //
    // It was `window.prompt`: one line, the browser's own chrome, no way to
    // read a long prompt, and on a phone a dialog over everything. Now the
    // message goes into the panel's own box, and the box says it is an edit:
    // a strip above it naming the message, an outline and a glow on the box,
    // the same outline on the message in the thread, and Send reading Save.
    //
    // The box rather than the bubble, and that was the decision asked for.
    // Bubbles are rebuilt whenever the thread moves, and an editor inside one
    // would have to survive every redraw; the transcript is a third of the
    // window, too little room for a long prompt. The box already has the
    // keyboard, the phone's keyboard and the growing, and it is where somebody
    // expects to type.
    //
    // The draft is not touched: what was half-typed before Edit is kept in the
    // store while the box holds the edit, and put back when the edit ends. And
    // Save replaces the words and does nothing else -- no new reply, as the
    // service's own `_edit` says (a rewritten question keeps the answer under
    // it). A save the server refuses leaves the edit in the box, with why.

    Shell.prototype.startEdit = function (row, revision) {
        const view = this.store.snapshot();
        const character = view.selection.character;
        const thread = view.selection.thread;
        const key = NS.conversationKey(character, thread);
        const current = this.editing;
        if (current && current.key === key && current.index === row.index) {
            this.nodes.input.focus();
            return;
        }
        if (current) this.finishEdit();
        const conversation = view.conversation && view.conversation.conversation;
        this.editing = {key, character, thread, index: row.index, version: row.active,
                        revision, text: String(row.text || ""), role: row.role,
                        who: (conversation && conversation.character) || character,
                        saving: false};
        const input = this.nodes.input;
        input.value = this.editing.text;
        this.applyEditing();
        // Save's own rule from the first moment: Send's was about the draft,
        // and an empty draft left Save greyed out over a message full of words.
        this.nodes.send.disabled = !this.canSaveEdit(view);
        this.grow();
        input.focus();
        try {
            input.setSelectionRange(input.value.length, input.value.length);
        } catch (error) { /* a box that will not take a caret still takes the edit */ }
        this.markEditTarget(true);
    };

    Shell.prototype.canSaveEdit = function (view) {
        const editing = this.editing;
        if (!editing || editing.saving) return false;
        if (!view.ready || view.error || !view.conversation) return false;
        if (view.operation && !view.operation.terminal) return false;
        return !!this.nodes.input.value.trim();
    };

    Shell.prototype.saveEdit = function () {
        const editing = this.editing;
        if (!editing || editing.saving) return Promise.resolve(false);
        const text = this.nodes.input.value;
        if (!text.trim()) {
            this.tell("Type the message's new words, or press Cancel to keep it.", "warn");
            return Promise.resolve(false);
        }
        if (text.trim() === editing.text.trim()) {
            this.finishEdit();
            this.tell("Nothing was changed.", "info");
            return Promise.resolve(false);
        }
        const revision = editing.revision;
        const envelope = this.store.envelope("edit_message", {
            // The conversation the edit began in, whatever is selected by the
            // time the request goes.
            conversation: {character: editing.character, thread_id: editing.thread},
            target: {index: editing.index, version: editing.version},
            expected_revision: typeof revision === "number"
                ? {kind: "revision", value: revision}
                : (revision ? {kind: "legacy", fingerprint: String(revision)} : null),
        });
        envelope.payload = {text, image_action: "keep"};
        editing.saving = true;
        this.nodes.send.disabled = true;
        this.say("Saving your edit\u2026", "info");
        return this.store.send(envelope).then((outcome) => outcome, (error) => ({
            ok: false, error: {message: (error && error.message) || ""},
        })).then((outcome) => {
            editing.saving = false;
            if (this.editing !== editing) return !!(outcome && outcome.ok);
            if (outcome && outcome.ok) {
                this.finishEdit();
                this.tell("Message edited.", "info");
                return true;
            }
            this.tell(((outcome && outcome.error && outcome.error.message)
                       || "That edit was refused.") + " Your edit is still in the box.",
                      "warn");
            this.nodes.send.disabled = !this.canSaveEdit(this.store.snapshot());
            return false;
        });
    };

    /** The edit is over, saved or not: the box is the draft's again --
     *  the draft of the conversation the edit was in, unless told which. */
    Shell.prototype.finishEdit = function (text) {
        const editing = this.editing;
        if (!editing) return false;
        this.editing = null;
        const input = this.nodes.input;
        input.value = typeof text === "string" ? text
            : (this.store.draft(editing.key).text || "");
        this.applyEditing();
        this.grow();
        this.markEditTarget();
        this.nodes.send.disabled = !this.canSend(this.store.snapshot());
        return true;
    };

    Shell.prototype.cancelEdit = function () {
        if (!this.finishEdit()) return false;
        this.tell("Edit cancelled. The message is as it was.", "info");
        return true;
    };

    /** Everything that says the box holds an edit, drawn from `editing`. */
    Shell.prototype.applyEditing = function () {
        const nodes = this.nodes;
        const editing = this.editing;
        nodes.composer.classList.toggle("forge-assistant-editing", !!editing);
        nodes.editBar.hidden = !editing;
        if (editing) {
            nodes.editLabel.textContent = editing.role === "assistant"
                ? "\u270e Editing " + (editing.who || "the") + "\u2019s reply"
                : "\u270e Editing your message";
        }
        nodes.send.textContent = editing ? "Save" : "Send";
        nodes.send.title = editing ? "Replace the message with these words" : "";
        nodes.input.setAttribute("aria-label", editing ? "Edit the message" : "Message");
        nodes.input.placeholder = editing ? "The message\u2019s new words\u2026"
            : "Message\u2026";
        // An edit changes the words, so the tools that add to a message stand
        // aside. The picture already on it is kept.
        nodes.attach.disabled = !!editing;
        nodes.dictate.disabled = !!editing;
    };

    /** Outline the message being edited, and bring it into view when asked. */
    Shell.prototype.markEditTarget = function (show) {
        const transcript = this.nodes.transcript;
        if (!transcript) return;
        const editing = this.editing;
        let found = null;
        Array.prototype.forEach.call(transcript.children || [], (node) => {
            const on = !!editing && !!node.dataset
                && node.dataset.index === String(editing.index);
            if (on) found = node;
            if (node.classList) node.classList.toggle("forge-assistant-edit-target", on);
        });
        if (show && found && typeof found.scrollIntoView === "function") {
            try {
                found.scrollIntoView({block: "nearest"});
            } catch (error) { /* an older engine's scrollIntoView takes no options */ }
        }
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
        // The feed is closed on purpose whenever nothing is coming, which is
        // most of the time. That is not a connection problem and says nothing.
        if (!view.connected && !view.idle) {
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
        // A warning about a press -- a picture left out, a prompt with nowhere
        // to go -- is read before the progress of the reply that press started,
        // or it is on screen for one frame. See `tell`.
        const fresh = this.told && Date.now() - this.told.at < TOLD_FOR;
        if (fresh && this.told.kind === "warn") {
            this.say(this.told.text, "warn");
            return;
        }
        if (view.operation && !view.operation.terminal) {
            this.say(view.operation.status || "Generating…", "info");
            return;
        }
        // The answer to a press, while it is fresh. See `tell`.
        if (fresh) {
            this.say(this.told.text, this.told.kind);
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
        this.resetTaps();
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
    NS.promptFrom = promptFrom;
    NS.showingPicture = showingPicture;
    NS.fileNameOf = fileNameOf;
    NS.stageable = stageable;
    NS.keptFromPrompt = keptFromPrompt;
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
