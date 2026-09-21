// Forge Assistant -- workspace focus, as a transaction that can be undone.
//
// Focus gives the whole viewport to the workspace that is open: the host's
// header, sidebars and footer go under an opaque background and the workspace
// fills what is left. It is not the Fullscreen API -- that takes the browser's
// own chrome as well, needs a gesture, and cannot be entered for one element
// without the page losing the ability to draw anything outside it.
//
// The hard part is not entering. It is *leaving*, on a page whose DOM another
// extension may have changed in the meantime.
//
// So this is written as a transaction. Before anything is touched, every value
// that will change is recorded -- the exact inline styles, the scroll
// positions, the tabindex and inert and aria state of what will be made
// unreachable, and where the keyboard focus was. Exit restores those values.
// It never sets `inert = false`, `tabindex = 0` or `overflow: auto` as a guess
// at what was there before, because a page where something else had already set
// `inert` would come back subtly broken and nobody would connect it to this.
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
        const trap = trapped(root);
        if (trap) {
            return {ok: false,
                    reason: "This workspace is inside a " + trap
                        + " and cannot be made full-screen safely."};
        }
        if (typeof adapter.canFocus === "function") {
            const found = adapter.canFocus(root);
            if (found && found.ok === false) return found;
        }
        return {ok: true, reason: ""};
    };

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

    // -- the transaction ---------------------------------------------------- //

    function record(node, properties) {
        const kept = {node, style: {}, attributes: {}};
        properties.forEach((name) => {
            // The *inline* value, not the computed one. Restoring a computed
            // value would write a stylesheet's answer into the element and
            // freeze it there.
            kept.style[name] = node.style[name];
        });
        return kept;
    }

    function restore(kept) {
        Object.keys(kept.style).forEach((name) => {
            const was = kept.style[name];
            if (was === undefined || was === "") kept.node.style.removeProperty
                ? kept.node.style.removeProperty(hyphenate(name))
                : (kept.node.style[name] = "");
            else kept.node.style[name] = was;
        });
        Object.keys(kept.attributes).forEach((name) => {
            const was = kept.attributes[name];
            if (was === null) kept.node.removeAttribute(name);
            else kept.node.setAttribute(name, was);
        });
        if (kept.inert !== undefined) {
            if (kept.inert === null) kept.node.removeAttribute("inert");
            else kept.node.inert = kept.inert;
        }
        if (kept.scroll !== undefined) {
            kept.node.scrollTop = kept.scroll.top;
            kept.node.scrollLeft = kept.scroll.left;
        }
    }

    function hyphenate(name) {
        return name.replace(/[A-Z]/g, (letter) => "-" + letter.toLowerCase());
    }

    Focus.prototype.enter = function (id, host) {
        if (this.active && this.active.id === id) return {ok: true, reason: ""};
        if (this.active) this.exit();
        const root = host && typeof host.resolveWorkspaceRoot === "function"
            ? host.resolveWorkspaceRoot(id)
            : null;
        const allowed = this.canFocus(id, root);
        if (!allowed.ok) return allowed;

        const adapter = this.adapterFor(id);
        const context = {
            id,
            root,
            adapter,
            saved: [],
            focused: document.activeElement,
        };

        // Everything that will change, written down before anything changes.
        context.saved.push(Object.assign(record(document.body, ["overflow"]),
                                         {scroll: {top: window.scrollY || 0,
                                                   left: window.scrollX || 0}}));
        context.saved.push(record(root, ["position", "inset", "zIndex", "margin",
                                         "padding", "overflow", "background"]));

        // Only the *siblings* outside the root are made unreachable, and only
        // as far up as the root's own parents. Never an ancestor -- inerting an
        // ancestor inerts the root inside it -- and never `aria-hidden` on the
        // application, which would take the whole page out of the accessibility
        // tree rather than the part that is covered.
        const assistantRoot = document.getElementById("forge-assistant-root");
        let walk = root;
        while (walk && walk.parentElement && walk.parentElement !== document.body
               .parentElement) {
            const parent = walk.parentElement;
            Array.prototype.forEach.call(parent.children, (sibling) => {
                if (sibling === walk || sibling === assistantRoot) return;
                if (assistantRoot && sibling.contains && sibling.contains(assistantRoot)) {
                    return;
                }
                const kept = {node: sibling, style: {}, attributes: {},
                              inert: sibling.hasAttribute("inert")
                                  ? sibling.inert : null};
                sibling.inert = true;
                context.saved.push(kept);
            });
            walk = parent;
        }

        document.body.style.overflow = "hidden";
        root.classList.add(ROOT_CLASS);
        document.body.classList.add(BODY_CLASS);

        try {
            if (typeof adapter.enter === "function") adapter.enter(context);
        } catch (error) {
            console.error("Forge Assistant: a focus adapter failed to enter", error);
        }

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
        return {ok: true, reason: ""};
    };

    Focus.prototype.exit = function () {
        const context = this.active;
        if (!context) return false;
        this.active = null;
        if (this.observer) {
            this.observer.disconnect();
            this.observer = null;
        }
        try {
            if (typeof context.adapter.exit === "function") context.adapter.exit(context);
        } catch (error) {
            console.error("Forge Assistant: a focus adapter failed to exit", error);
        }
        // The class comes off even if the root was detached while focused --
        // a workspace rebuilt by a Gradio update is still a workspace whose
        // siblings have to come back.
        if (context.root && context.root.classList) {
            context.root.classList.remove(ROOT_CLASS);
        }
        document.body.classList.remove(BODY_CLASS);
        context.saved.slice().reverse().forEach((kept) => {
            try {
                restore(kept);
            } catch (error) {
                console.error("Forge Assistant: could not restore a focused element", error);
            }
        });
        if (context.focused && typeof context.focused.focus === "function"
            && document.contains(context.focused)) {
            try {
                context.focused.focus();
            } catch (error) { /* an element that will not take focus */ }
        } else if (context.root && typeof context.root.querySelector === "function") {
            const first = context.root.querySelector(
                "button, [href], input, select, textarea, [tabindex]:not([tabindex='-1'])");
            if (first && typeof first.focus === "function") first.focus();
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
