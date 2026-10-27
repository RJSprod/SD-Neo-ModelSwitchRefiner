// Forge Assistant -- workspace focus, as a transaction that can be undone.
//
// Focus makes one workspace take the whole browser window: a theme's sidebars,
// the footer, the other tabs -- none of it is visible until the person leaves
// focus mode, by the control on the assistant or by pressing Escape. The one
// piece of chrome it can keep is the workspace's own tab bar, along the top
// (see "the workspace's own tab bar" below); the shell decides whether. It
// is not the Fullscreen API, and does not use it: an element in full screen is
// the only thing the browser draws, so a workspace made full screen would take
// the assistant -- the way back out -- off the screen with everything else. The
// browser's full screen that the Focus toggle asks for alongside this is the
// whole document's, and belongs to the shell: see `enterScreen` in
// forge_assistant.js.
//
// It never moves, removes or rewrites anything of the host's. What it does is
// classes and a few recorded inline values: the chrome taken out of the
// layout by a rule that reads the classes, the workspace fixed over the
// window, the kept tab bar fixed along its top. That is what keeps it safe:
// nothing about the host's DOM changes, so nothing about the host can break,
// and leaving is the classes coming off.
//
// The other rule: never change a canvas's bitmap dimensions. Mini Paint and the
// Krea spatial editor both draw into one, and resizing a canvas element clears
// it. An adapter that needs the editor to re-fit calls the component's own fit
// method; there is no fallback that touches the bitmap.

(function () {
    "use strict";

    const NS = (window.forgeAssistant = window.forgeAssistant || {});

    const ROOT_CLASS = "forge-assistant-focus-root";
    const BODY_CLASS = "forge-assistant-focused";
    const PATH_CLASS = "forge-assistant-focus-path";
    const HIDDEN_CLASS = "forge-assistant-focus-hidden";
    const ASSISTANT_ROOT = "forge-assistant-root";
    // The workspace's own tab bar, kept along the top of the window, and what
    // the body says about it while focus is on: "shown" above the workspace,
    // "hidden" above the window's edge, "peek" over the workspace's top for a
    // moment. The focus root starts under a shown bar, at the property.
    const BAR_CLASS = "forge-assistant-focus-bar";
    const BAR_ATTRIBUTE = "data-forge-assistant-focus-bar";
    const BAR_MODES = ["shown", "hidden", "peek"];
    const TOP_PROPERTY = "--forge-assistant-focus-top";
    // Chrome of somebody else's that goes with the kept bar -- the Model Chain
    // theme's quick settings drawer, which the bar's button opens -- left in
    // the layout by the rule that takes the rest out. It draws itself.
    const KEEP_CLASS = "forge-assistant-focus-keep";
    // Super focus (docs/22 §3.39, §3.40): on Mini Paint NEO's WanGP tab, focus
    // with WanGP's page laid out as a studio -- its prompt beside its Generate,
    // its progress and its gallery. This class beside ROOT_CLASS on the focused
    // panel is the whole of it here: Mini Paint's tab watches the panel's
    // classes and tells WanGP's page (its bridge 1.15.0). Shared with that
    // extension -- do not rename (CLAUDE.md, "Shared with Mini Paint NEO").
    const STUDIO_CLASS = "forge-assistant-wangp-studio";

    // Tab bars outside the focused workspace. Marked one at a time by the
    // script rather than selected by a stylesheet, because the decision that
    // makes it safe -- "unless it contains the workspace" -- is one CSS cannot
    // express, and the version that left it to CSS hid the page.
    const TAB_BARS = ".tab-nav, [role=\"tablist\"]";

    // Properties on an ancestor that would make `position: fixed` resolve
    // against that ancestor instead of the viewport. A focus root inside one of
    // these is a focus root that fills a box rather than the screen -- and the
    // failure is silent, which is why it is checked rather than assumed.
    const TRAPS = ["transform", "filter", "perspective", "backdropFilter", "contain",
                   "willChange"];

    function Focus() {
        this.adapters = new Map();
        this.active = null;
        this.observer = null;
        this.barMode = "shown";
        this.barObserver = null;
    }

    Focus.prototype.registerAdapter = function (adapter) {
        if (!adapter || !adapter.workspaceId) return () => undefined;
        this.adapters.set(adapter.workspaceId, adapter);
        return () => this.adapters.delete(adapter.workspaceId);
    };

    Focus.prototype.adapterFor = function (id) {
        return this.adapters.get(id) || this.defaultAdapter;
    };

    // The adapter for an ordinary Gradio tab panel: a root, and nothing to
    // re-fit. Most workspaces are this, which is why it is the default rather
    // than a thing each one has to declare.
    Focus.prototype.defaultAdapter = {
        workspaceId: "",
        canFocus(root) {
            return root ? {ok: true, reason: ""}
                : {ok: false, reason: "This workspace has no panel to fill with."};
        },
        enter() {},
        exit() {},
        onResize() {},
    };

    Focus.prototype.canFocus = function (id, root) {
        const adapter = this.adapterFor(id);
        if (!root) return {ok: false, reason: "This workspace has no panel to fill with."};
        if (typeof adapter.canFocus === "function") {
            const found = adapter.canFocus(root);
            if (found && found.ok === false) return found;
        }
        // A transformed ancestor makes `position: fixed` resolve against that
        // ancestor rather than the viewport, so it used to be a refusal. It is
        // a *note* now: the chrome is taken out of the layout as well as
        // covered, so a workspace that cannot quite escape its ancestor still
        // fills what is left of the window -- which is the thing somebody
        // asked for. Refusing outright meant a press that did nothing at all
        // and a sentence nobody could act on.
        const trap = trapped(root);
        return {ok: true,
                reason: "",
                note: trap
                    ? "This workspace sits inside a " + trap
                        + ", so it fills that area rather than the whole window."
                    : ""};
    };

    function each(selector, visit) {
        let found = [];
        try {
            found = Array.prototype.slice.call(document.querySelectorAll(selector));
        } catch (error) {
            return;
        }
        found.forEach((node) => {
            if (node && node.classList) visit(node);
        });
    }

    // Is this element actually being drawn? Asked of the layout rather than of
    // the classes, because the question is whether somebody can see the
    // workspace and the ways to hide an element are not enumerable from here --
    // a `display: none` anywhere above it is enough, and so is a zero box.
    //
    // `getBoundingClientRect` forces the pending layout, which is what makes
    // this answerable in the same turn as the marking that caused it.
    function painted(root) {
        try {
            const box = root.getBoundingClientRect();
            return box.width > 0 && box.height > 0;
        } catch (error) {
            // A host with no layout to ask -- assume the marking was fine
            // rather than throwing away a mode that may well be working.
            return true;
        }
    }

    // Every mark this module put on the page, taken off. Split out of `exit`
    // because the safety valve needs exactly this and none of the rest of
    // leaving: no scroll restored, no observer stopped, no adapter told.
    function undo(context) {
        (context.path || []).forEach((node) => {
            if (node && node.classList) node.classList.remove(PATH_CLASS);
        });
        (context.hidden || []).forEach((node) => {
            if (node && node.classList) node.classList.remove(HIDDEN_CLASS);
        });
        // Put back exactly what was there: the value and its priority, or no
        // inline `top` at all, which is what most of them had.
        (context.stuck || []).forEach((entry) => {
            const style = entry.node && entry.node.style;
            if (!style) return;
            if (entry.value) style.setProperty("top", entry.value, entry.priority || "");
            else style.removeProperty("top");
        });
        context.path = [];
        context.hidden = [];
        context.stuck = [];
        clearBar(context);
    }

    // -- the workspace's own tab bar ---------------------------------------- //
    //
    // The strip that holds the button showing this workspace -- Forge's
    // `#tabs > .tab-nav`, the Model Chain theme's header -- kept, at the top
    // of the window, instead of taken out with the rest of the chrome: asked
    // for once every tab was a button on it ("Let's make that entire row
    // available in focus mode"). Found by the button, not by a name, for the
    // reason the other tab bars are marked one at a time: the names are on
    // every nested tab group too. Kept only where its parent is on the
    // workspace's path, because a bar under any other parent sits inside
    // chrome the rule below takes out, and is hidden as it always was.
    // Nothing is moved: the class makes it fixed along the top, and a press
    // on one of its tabs is the host's own, which focus follows to the new
    // workspace (the shell's `subscribeNavigation`).

    function ownBar(root) {
        if (!root || !root.id) return null;
        let found = null;
        each(TAB_BARS, (bar) => {
            if (found || bar === root || root.contains(bar) || bar.contains(root)) return;
            const buttons = Array.prototype.slice.call(bar.children || []);
            if (buttons.some((button) => button && button.getAttribute
                             && button.getAttribute("aria-controls") === root.id)) {
                found = bar;
            }
        });
        if (!found || !found.parentElement || !found.parentElement.classList
            || !found.parentElement.classList.contains(PATH_CLASS)) return null;
        return found;
    }

    function clearBar(context) {
        if (context.bar && context.bar.classList) context.bar.classList.remove(BAR_CLASS);
        context.bar = null;
        const body = document.body;
        if (!body) return;
        if (body.hasAttribute(BAR_ATTRIBUTE)) body.removeAttribute(BAR_ATTRIBUTE);
        if (body.style && body.style.getPropertyValue(TOP_PROPERTY)) body.style.removeProperty(TOP_PROPERTY);
    }

    function paddingTop(root) {
        try {
            const style = window.getComputedStyle ? window.getComputedStyle(root) : null;
            const found = style ? parseFloat(style.paddingTop) : NaN;
            return found > 0 ? found : 0;
        } catch (error) {
            return 0;
        }
    }

    // Every element under `root` that is laid out, depth first, within a
    // budget. Hidden subtrees are skipped whole -- an inactive nested tab is
    // a thousand elements nobody can see -- and the budget is what keeps a
    // pathological page from turning a toggle into a pause.
    const WALK_BUDGET = 8000;

    function visitLaidOut(root, visit) {
        let seen = 0;
        (function descend(parent) {
            const children = parent.children || [];
            for (let i = 0; i < children.length && seen < WALK_BUDGET; i += 1) {
                const child = children[i];
                if (!child || !child.style) continue;
                seen += 1;
                const style = window.getComputedStyle ? window.getComputedStyle(child) : null;
                if (style && style.display === "none") continue;
                visit(child);
                descend(child);
            }
        })(root);
    }

    // Is there a scroll container between this element and the root? If so,
    // a sticky offset on the element is measured against that container, not
    // against the workspace, and it is not this module's business.
    function scrollsBetween(node, root) {
        let parent = node.parentElement;
        while (parent && parent !== root) {
            const style = window.getComputedStyle ? window.getComputedStyle(parent) : null;
            const overflow = style ? (style.overflowY || style.overflow || "visible") : "visible";
            if (overflow !== "visible" && overflow !== "clip") return true;
            parent = parent.parentElement;
        }
        return false;
    }

    // The results column fills the window (forge_assistant_layout.js), and what
    // the window is changes on the way in and out.
    function refill() {
        try {
            if (NS.fill && typeof NS.fill.refresh === "function") NS.fill.refresh();
        } catch (error) { /* the fill measures again on its own next resize */ }
    }

    function trapped(root) {
        // Walked to the document, because one ancestor with a transform is
        // enough. Reported by name so the reason a workspace cannot be focused
        // is a sentence rather than a shrug.
        let walk = root && root.parentElement;
        while (walk && walk !== document.documentElement) {
            const style = window.getComputedStyle ? window.getComputedStyle(walk) : null;
            if (style) {
                for (let i = 0; i < TRAPS.length; i += 1) {
                    const value = style[TRAPS[i]];
                    if (value && value !== "none" && value !== "auto" && value !== "normal") {
                        return TRAPS[i] === "backdropFilter" ? "backdrop filter" : TRAPS[i];
                    }
                }
            }
            walk = walk.parentElement;
        }
        return "";
    }

    // -- entering and leaving ------------------------------------------------ //
    //
    // The whole mechanism, and it is deliberately one line of DOM work:
    //
    //     root.classList.add("forge-assistant-focus-root")
    //
    // The stylesheet makes that `position: fixed; inset: 0` with an opaque
    // themed background and a z-index above the tab bar, the sidebars and the
    // footer. Nothing of the host's is hidden, moved, restyled or made
    // unreachable -- the tab bar is still exactly where it was, underneath --
    // and leaving is one class removal. That is what makes this safe: nothing
    // about the host changes, so nothing about the host can break.
    //
    // It was not one line to begin with. The first version also locked the
    // body's scrolling and walked up the tree setting `inert` on every sibling
    // at every level, to make the covered page unreachable. Both were beyond
    // what this needs -- the focus root covers the viewport, so there is
    // nothing to reach -- and between them they produced the failure that was
    // reported: on Txt2Img the tab bar stayed exactly where it was and the only
    // thing that changed was that the page would no longer scroll.
    //
    // What is left of that: `overscroll-behavior: contain` in the stylesheet,
    // which stops a wheel gesture inside the focused workspace from chaining
    // out to the page behind it, and does it without touching the page at all.

    function saveState(root) {
        // The one thing worth recording, because it is the one thing that can
        // be changed by something other than the class: where the workspace was
        // scrolled to. A focus that returned somebody to the top of a long tab
        // would be a focus nobody used twice.
        return {
            root,
            scrollTop: root.scrollTop || 0,
            scrollLeft: root.scrollLeft || 0,
            pageX: window.scrollX || 0,
            pageY: window.scrollY || 0,
            focused: document.activeElement,
        };
    }

    Focus.prototype.enter = function (id, host) {
        // A second press on a workspace that is already focused answers the
        // same thing the first one did, note included, rather than a bare
        // success that would quietly drop the caveat.
        if (this.active && this.active.id === id) {
            return {ok: true, reason: "", note: this.active.note || ""};
        }
        if (this.active) this.exit();
        const root = host && typeof host.resolveWorkspaceRoot === "function"
            ? host.resolveWorkspaceRoot(id)
            : null;
        const allowed = this.canFocus(id, root);
        if (!allowed.ok) return allowed;

        const adapter = this.adapterFor(id);
        const context = Object.assign(saveState(root),
                                      {id, adapter, path: [], hidden: [], stuck: [],
                                       note: allowed.note || ""});

        // Mark the workspace's ancestors, from its parent up to <body>. One
        // stylesheet rule then hides every child of a marked element that is
        // neither on that path nor the workspace nor the assistant -- which is
        // the tab bar, the theme's header and sidebars, the footer, and the
        // other tabs, without this code having to know what any of them are
        // called.
        //
        // Covering them was not enough, and that is the whole of why this is
        // here. The reference implementation lays the workspace over the page
        // with a z-index, which works while the chrome is ordinary content.
        // Under a theme that draws its own header -- Lobe does -- that header
        // is positioned and has a stacking context of its own, so it stayed
        // exactly where it was with the workspace "over" it. Taking it out of
        // the layout is the only thing that reliably removes it, and a class
        // that a stylesheet reads is the most reversible way to do that: no
        // inline styles to restore, no DOM moved, and leaving is the classes
        // coming off.
        let walk = root.parentElement;
        while (walk) {
            walk.classList.add(PATH_CLASS);
            context.path.push(walk);
            if (walk === document.body) break;
            walk = walk.parentElement;
        }

        // The workspace's own tab bar, kept along the top.
        context.bar = ownBar(root);
        if (context.bar) context.bar.classList.add(BAR_CLASS);

        // Tab bars the marking above did not reach, because the theme moved
        // them out of the workspace's ancestry. Never one that contains the
        // workspace, never one inside it -- a workspace with nested tabs
        // keeps its own -- and never the bar kept above.
        each(TAB_BARS, (bar) => {
            if (bar === root || root.contains(bar) || bar.contains(root)) return;
            if (bar === context.bar) return;
            if (bar.id === ASSISTANT_ROOT || bar.querySelector("#" + ASSISTANT_ROOT)) return;
            bar.classList.add(HIDDEN_CLASS);
            context.hidden.push(bar);
        });

        root.classList.add(ROOT_CLASS);
        document.body.classList.add(BODY_CLASS);

        // Sticky offsets inside the workspace, and the gap they leave.
        //
        // A theme that draws a header reserves room under it. Lobe's split
        // previewer makes the results column `position: sticky; top: 80px` --
        // 64px of header and a margin -- so the picture stays on screen while
        // the prompt column scrolls under it. In focus mode the header is gone
        // and the workspace is its own scroll container, so the same 80px is
        // nothing but a gap above the gallery: the column sits at the same
        // place on the screen whether focus is on or off, which is exactly how
        // it was reported.
        //
        // Reasoned about generally rather than by naming the theme's ids: a
        // sticky element whose offset is measured against the workspace's own
        // scroll edge -- no scroller of its own between it and the root -- was
        // clearing something above the workspace, and everything above the
        // workspace is hidden now. Its offset becomes the root's padding, so a
        // stuck column keeps the margin it has at rest. One inside an inner
        // scroller is measured against that scroller and is left alone.
        //
        // An inline style, because the theme's declaration carries !important
        // on an id and nothing in a stylesheet outranks that reliably; the
        // value it replaces is recorded exactly and put back on the way out.
        const inset = paddingTop(root);
        visitLaidOut(root, (node) => {
            const style = window.getComputedStyle ? window.getComputedStyle(node) : null;
            if (!style || style.position !== "sticky") return;
            const top = parseFloat(style.top);
            if (!(top > inset)) return;
            if (scrollsBetween(node, root)) return;
            context.stuck.push({node,
                                value: node.style.getPropertyValue("top"),
                                priority: node.style.getPropertyPriority("top")});
            node.style.setProperty("top", inset + "px", "important");
        });

        // The safety valve, and the reason it is here: the rule that hides the
        // chrome is written against a shape this code cannot see -- somebody
        // else's theme, on somebody else's Forge. When it got that shape wrong
        // the whole page went blank, which is a worse outcome than focus mode
        // not working, because nothing on screen says what happened or how to
        // undo it.
        //
        // So the one thing focus mode exists to show is checked, after the
        // marking and before anybody looks at it. If the workspace is not being
        // painted, the marking comes off and what is left is the plain overlay
        // -- the reference implementation's behaviour, which is imperfect under
        // a theme that draws its own header and is never blank.
        if (!painted(root)) {
            undo(context);
            context.degraded = true;
        }

        try {
            if (typeof adapter.enter === "function") adapter.enter(context);
        } catch (error) {
            console.error("Forge Assistant: a focus adapter failed to enter", error);
        }

        // Anything in the workspace sized against the window -- a fitted
        // canvas, a panel with a max-height taken from the viewport -- has to
        // be recomputed now, because the root's box just changed from "a row in
        // the page" to "the whole window".
        if (typeof ResizeObserver === "function") {
            this.observer = new ResizeObserver(() => {
                try {
                    if (typeof adapter.onResize === "function") adapter.onResize(context);
                } catch (error) {
                    console.error("Forge Assistant: a focus adapter failed to resize",
                                  error);
                }
            });
            this.observer.observe(root);
        }

        this.active = context;
        // The kept bar along the top, and the root under it. Measured again
        // whenever the bar changes size -- a theme's thinner bar, Forge's own
        // wrapping onto a second row -- because the root starts where it ends.
        this.placeBar();
        if (context.bar && typeof ResizeObserver === "function") {
            this.barObserver = new ResizeObserver(() => {
                if (this.placeBar()) refill();
            });
            this.barObserver.observe(context.bar);
        }
        // The results column's height is measured against what scrolls it,
        // and that just changed from the page to this root.
        refill();
        // One line in the console saying what was actually done, because the
        // reports that led here were "it does not work" three times over and
        // each one meant something different. Cheap, and the first thing to
        // ask for next time.
        try {
            console.info("Forge Assistant: focus on #" + id
                + (context.degraded ? " (chrome left in place)" : "")
                + ", " + context.path.length + " ancestors marked, "
                + context.hidden.length + " tab bars hidden, "
                + (context.bar ? "its own tab bar " + this.barMode + ", " : "")
                + context.stuck.length + " sticky offsets closed"
                + (context.note ? ", note: " + context.note : ""));
        } catch (error) { /* a console that cannot be written to */ }
        // The note travels out with the success. A containing-block trap is
        // not a refusal any more, and neither is a page whose chrome would not
        // come out of the layout, so the sentence describing what the caller
        // actually got is the only way it hears about either.
        const note = context.degraded
            ? "This page's layout would not let the chrome go, so the workspace "
                + "is laid over it instead."
            : context.note;
        return {ok: true, reason: "", note};
    };

    Focus.prototype.exit = function () {
        const context = this.active;
        if (!context) return false;
        this.active = null;
        if (this.observer) {
            this.observer.disconnect();
            this.observer = null;
        }
        if (this.barObserver) {
            this.barObserver.disconnect();
            this.barObserver = null;
        }
        try {
            if (typeof context.adapter.exit === "function") context.adapter.exit(context);
        } catch (error) {
            console.error("Forge Assistant: a focus adapter failed to exit", error);
        }
        // The class comes off even if the root was detached while focused -- a
        // workspace rebuilt by a Gradio update is still a workspace that has to
        // stop being fixed to the viewport.
        if (context.root && context.root.classList) {
            context.root.classList.remove(ROOT_CLASS, STUDIO_CLASS);
        }
        undo(context);
        document.body.classList.remove(BODY_CLASS);
        if (context.root) {
            context.root.scrollTop = context.scrollTop;
            context.root.scrollLeft = context.scrollLeft;
        }
        if (typeof window.scrollTo === "function") {
            try {
                window.scrollTo(context.pageX, context.pageY);
            } catch (error) { /* a window that will not be scrolled is not a failure */ }
        }
        refill();
        if (context.focused && typeof context.focused.focus === "function"
            && document.contains(context.focused)) {
            try {
                context.focused.focus();
            } catch (error) { /* an element that will not take focus */ }
        }
        return true;
    };

    // Switching workspaces with focus on: the treatment comes off the old root
    // -- even a detached one -- and goes onto the new one if its adapter says
    // yes. A destination that cannot be focused turns focus off and says why,
    // rather than leaving a page whose header is covered by nothing.
    Focus.prototype.moveTo = function (id, host) {
        if (!this.active) return {ok: false, reason: "Focus is not on."};
        this.exit();
        return this.enter(id, host);
    };

    /** The kept bar's mode, said on the body, and where the root starts: under
     *  a shown bar, at the window's top edge otherwise -- a hidden bar is
     *  above that edge and a peeking one over the workspace's top. Written
     *  only when it changes; true when the root's top moved, for what has to
     *  measure again. */
    Focus.prototype.placeBar = function () {
        const context = this.active;
        const body = document.body;
        if (!context || !context.bar || !body) return false;
        if (body.getAttribute(BAR_ATTRIBUTE) !== this.barMode) body.setAttribute(BAR_ATTRIBUTE, this.barMode);
        let top = 0;
        if (this.barMode === "shown") {
            try {
                const box = context.bar.getBoundingClientRect();
                top = Math.max(0, Math.round((box.top || 0) + (box.height || 0)));
            } catch (error) {
                top = 0;
            }
        }
        const value = top + "px";
        if (body.style.getPropertyValue(TOP_PROPERTY) === value) return false;
        body.style.setProperty(TOP_PROPERTY, value);
        return true;
    };

    /** Shown, hidden or peeking: kept across focus going off and on, because
     *  it is the person's choice and not the workspace's. */
    Focus.prototype.setBar = function (mode) {
        if (BAR_MODES.indexOf(mode) === -1) return this.barMode;
        this.barMode = mode;
        if (this.placeBar()) refill();
        return this.barMode;
    };

    /** Super focus on the focused panel, or off it: the studio's class, and
     *  nothing else. True when the panel says what was asked; false with
     *  focus off, which is where the class cannot go. Off with focus, always
     *  (`exit`). */
    Focus.prototype.setStudio = function (on) {
        const context = this.active;
        if (!context || !context.root || !context.root.classList) return false;
        context.studio = !!on;
        if (on) context.root.classList.add(STUDIO_CLASS);
        else context.root.classList.remove(STUDIO_CLASS);
        return true;
    };

    Focus.prototype.isStudio = function () {
        return !!(this.active && this.active.studio);
    };

    /** The bar kept for the focused workspace, or null. */
    Focus.prototype.keptBar = function () {
        return this.active ? this.active.bar || null : null;
    };

    Focus.prototype.isActive = function () {
        return !!this.active;
    };

    Focus.prototype.activeWorkspace = function () {
        return this.active ? this.active.id : "";
    };

    Focus.prototype.dispose = function () {
        this.exit();
        this.adapters.clear();
    };

    // -- the editors ------------------------------------------------------- //
    //
    // A canvas editor cannot be left to CSS: the element's box changes and the
    // bitmap does not, so the drawing is either stretched or cropped. Each of
    // these calls the component's *own* fit method and nothing else -- there is
    // deliberately no fallback that sets width/height on the canvas, because
    // that clears it.

    function canvasAdapter(workspaceId, refit) {
        return {
            workspaceId,
            canFocus(root) {
                return root ? {ok: true, reason: ""}
                    : {ok: false, reason: "This editor is not on the page."};
            },
            enter(context) {
                refit(context);
            },
            exit(context) {
                refit(context);
            },
            onResize(context) {
                refit(context);
            },
        };
    }

    function refitMiniPaint(context) {
        // The component's own fit, called by whichever name the installed copy
        // exposes. Nothing is resized here if none of them is present: a
        // picture that has to be scrolled is a far smaller problem than a
        // picture that has been erased.
        const api = window.miniPaint || window.MiniPaint || null;
        if (api && typeof api.fit === "function") {
            api.fit();
            return;
        }
        if (api && api.GUI && typeof api.GUI.prepare_canvas === "function") {
            api.GUI.prepare_canvas();
            return;
        }
        const button = context.root
            && context.root.querySelector("[id$='minipaint-fit'], [id$='fit-canvas']");
        if (button) button.click();
    }

    function refitSpatial(context) {
        const api = window.kreaSpatial || null;
        if (api && typeof api.fit === "function") {
            api.fit();
            return;
        }
        const button = context.root
            && context.root.querySelector("[id$='spatial-fit']");
        if (button) button.click();
    }

    NS.Focus = Focus;
    NS.FOCUS_BAR_CLASS = BAR_CLASS;
    NS.FOCUS_KEEP_CLASS = KEEP_CLASS;
    NS.FOCUS_STUDIO_CLASS = STUDIO_CLASS;
    NS.focusTrapped = trapped;
    NS.canvasAdapter = canvasAdapter;

    NS.focus = (function () {
        const focus = new Focus();
        // Registered by id fragment rather than by a hard-coded tab id, because
        // the id a tab is registered under is a host fact and G3 is what settles
        // it. `registerAdapter` is public, so an installation whose ids differ
        // can add its own without this file changing.
        focus.registerAdapter(canvasAdapter("minipaint", refitMiniPaint));
        focus.registerAdapter(canvasAdapter("mini_paint", refitMiniPaint));
        focus.registerAdapter(canvasAdapter("krea_spatial", refitSpatial));
        return focus;
    })();
})();
