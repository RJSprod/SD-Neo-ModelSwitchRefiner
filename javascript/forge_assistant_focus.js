// Forge Assistant -- workspace focus, as a transaction that can be undone.
//
// Focus makes one workspace take the whole browser window: the Forge Neo tab
// bar, a theme's sidebars, the footer -- none of it is visible until the person
// leaves focus mode, by the control on the assistant or by pressing Escape. It
// is not the Fullscreen API: that takes the browser's own chrome as well, needs
// a gesture every time, and cannot be entered for one element without the page
// losing the ability to draw anything outside it.
//
// It does not hide, remove or restyle the host's tab bar, and it never touches
// the host's DOM. The tab bar is still there, underneath; this extension's own
// root element is simply laid over everything else. That is what keeps it safe:
// nothing about the host changes, so nothing about the host can break, and
// leaving is one class removal.
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
        if (this.active && this.active.id === id) return {ok: true, reason: ""};
        if (this.active) this.exit();
        const root = host && typeof host.resolveWorkspaceRoot === "function"
            ? host.resolveWorkspaceRoot(id)
            : null;
        const allowed = this.canFocus(id, root);
        if (!allowed.ok) return allowed;

        const adapter = this.adapterFor(id);
        const context = Object.assign(saveState(root), {id, adapter});

        root.classList.add(ROOT_CLASS);
        document.body.classList.add(BODY_CLASS);

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
        // The class comes off even if the root was detached while focused -- a
        // workspace rebuilt by a Gradio update is still a workspace that has to
        // stop being fixed to the viewport.
        if (context.root && context.root.classList) {
            context.root.classList.remove(ROOT_CLASS);
        }
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
