// Model Chain -- TEMPORARY: a watcher on Txt2Img's Generate box.
//
// Reported: with the Lobe theme's split previewer on, a sequence of focus
// mode and the flyout's docked settings column ends with Generate's box
// (`#txt2img_generate_box`, which the theme moves into the gallery's
// container) back in the left column, at the place Gradio first drew it. No
// script on the page is known to move it there, so this watches the box
// itself and writes, into this extension's own log, who moved it and when:
//
//   - every DOM call that inserts or removes the box, with the stack of the
//     caller (the one fact that names the mover);
//   - every change of the box's ancestry seen by a MutationObserver, with the
//     ancestry before and after, so a move made by any other means -- a
//     rebuilt subtree, a replaced element -- is caught too, and a new element
//     under the same id is told apart from the old one moved;
//   - the moments around it: focus on and off, the dock placed and released,
//     the panel opened and closed, the browser's full screen, window resizes,
//     a tab change, and Gradio queue requests.
//
// Each record carries the state at that moment (focus, dock, panel, full
// screen, window size, selected tab, the focused element) and goes to
// `mc_generate_watch`, which writes one log line per record. Nothing here
// reads a prompt or any text: elements are named by id and class only, and
// the server side keeps only printable ASCII and caps every field.
//
// To be removed once the log has caught the mover.

(function () {
    "use strict";

    const WATCHED = ["txt2img_generate_box", "img2img_generate_box"];
    const ROUTE = "/model-chain/generate-box/watch";
    const FLUSH_MS = 300;
    const STACK_LINES = 18;

    const queue = [];
    let flushTimer = 0;
    let sequence = 0;
    const last = {};

    function nameOf(node) {
        if (!node) return "(none)";
        if (node.nodeType === 9) return "#document";
        if (node.nodeType !== 1) return node.nodeName;
        if (node.id) return "#" + node.id;
        const classes = node.className && typeof node.className === "string"
            ? node.className.trim().split(/\s+/).slice(0, 2).join(".") : "";
        return node.nodeName.toLowerCase() + (classes ? "." + classes : "");
    }

    function chainOf(node) {
        const parts = [];
        let walk = node;
        while (walk && parts.length < 8) {
            parts.push(nameOf(walk));
            walk = walk.parentNode;
        }
        return parts.join(" < ");
    }

    function boxOf(id) {
        try {
            return document.getElementById(id);
        } catch (error) {
            return null;
        }
    }

    function snapshot() {
        const found = {};
        try {
            const NS = window.forgeAssistant || {};
            found.focus = !!(NS.focus && NS.focus.isActive && NS.focus.isActive());
            found.docked = !!(NS.dock && NS.dock.isDocked && NS.dock.isDocked());
            found.open = !!(NS.shell && NS.shell.state && NS.shell.state.panelOpen);
            found.full = !!(document.fullscreenElement || document.webkitFullscreenElement);
            found.width = window.innerWidth;
            found.height = window.innerHeight;
            found.hidden = !!document.hidden;
            const tab = document.querySelector("#tabs > .tab-nav > button.selected");
            found.tab = tab ? (tab.textContent || "").trim().slice(0, 24) : "";
            found.active = nameOf(document.activeElement);
            WATCHED.forEach((id) => {
                found[id] = chainOf(boxOf(id));
            });
        } catch (error) {
            found.error = String(error && error.message || error).slice(0, 120);
        }
        return found;
    }

    function stack() {
        try {
            return (new Error().stack || "").split("\n").slice(1, 1 + STACK_LINES)
                .map((line) => line.trim()).filter(Boolean);
        } catch (error) {
            return [];
        }
    }

    function record(kind, detail, withStack, state) {
        sequence += 1;
        const entry = Object.assign({seq: sequence, t: Date.now(), kind}, detail || {});
        entry.state = state || snapshot();
        if (withStack) entry.stack = stack();
        queue.push(entry);
        if (!flushTimer) flushTimer = window.setTimeout(flush, FLUSH_MS);
    }

    function flush() {
        flushTimer = 0;
        if (!queue.length || typeof fetch !== "function") return;
        const records = queue.splice(0, 50);
        if (queue.length && !flushTimer) flushTimer = window.setTimeout(flush, 0);
        try {
            fetch(ROUTE, {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({records}),
                keepalive: true,
            }).catch(() => undefined);
        } catch (error) { /* a log line, not a feature */ }
    }

    function watched(node) {
        return !!(node && node.nodeType === 1 && WATCHED.indexOf(node.id) >= 0);
    }

    // -- the DOM calls that move it ----------------------------------------- //

    function noteCall(method, node, target) {
        if (!watched(node)) return;
        record("dom." + method, {
            box: "#" + node.id,
            from: chainOf(node.parentNode),
            to: chainOf(target),
        }, true);
    }

    try {
        const N = Node.prototype;
        const E = Element.prototype;
        const insertBefore = N.insertBefore;
        N.insertBefore = function (node, reference) {
            noteCall("insertBefore", node, this);
            return insertBefore.call(this, node, reference);
        };
        const appendChild = N.appendChild;
        N.appendChild = function (node) {
            noteCall("appendChild", node, this);
            return appendChild.call(this, node);
        };
        const replaceChild = N.replaceChild;
        N.replaceChild = function (node, old) {
            noteCall("replaceChild", node, this);
            if (watched(old)) noteCall("replaceChild.replaced", old, null);
            return replaceChild.call(this, node, old);
        };
        const removeChild = N.removeChild;
        N.removeChild = function (node) {
            noteCall("removeChild", node, null);
            return removeChild.call(this, node);
        };
        ["prepend", "append", "before", "after", "replaceWith"].forEach((name) => {
            const original = E[name];
            if (typeof original !== "function") return;
            E[name] = function () {
                const target = name === "prepend" || name === "append" ? this : this.parentNode;
                for (let index = 0; index < arguments.length; index += 1) {
                    noteCall(name, arguments[index], target);
                }
                return original.apply(this, arguments);
            };
        });
        const remove = E.remove;
        if (typeof remove === "function") {
            E.remove = function () {
                noteCall("remove", this, null);
                return remove.call(this);
            };
        }
    } catch (error) {
        record("patch.failed", {message: String(error && error.message || error).slice(0, 200)});
    }

    // -- what the DOM says afterwards --------------------------------------- //

    function check(why) {
        WATCHED.forEach((id) => {
            const box = boxOf(id);
            const chain = chainOf(box);
            const before = last[id];
            if (!before) {
                last[id] = {node: box, chain};
                if (box) record("seen", {box: "#" + id, chain, why});
                return;
            }
            if (before.chain === chain && before.node === box) return;
            record(before.node && box && before.node !== box ? "replaced" : "moved",
                   {box: "#" + id, before: before.chain, after: chain, why});
            last[id] = {node: box, chain};
        });
    }

    try {
        const observer = new MutationObserver((mutations) => {
            check("mutation x" + mutations.length);
        });
        observer.observe(document.documentElement, {childList: true, subtree: true});
    } catch (error) {
        record("observer.failed", {message: String(error && error.message || error).slice(0, 200)});
    }

    // -- the moments around it ---------------------------------------------- //

    ["fullscreenchange", "webkitfullscreenchange", "visibilitychange", "pagehide", "pageshow"]
        .forEach((name) => {
            document.addEventListener(name, () => {
                record("event." + name, {});
                if (name === "pagehide") flush();
            });
        });
    let resizeTimer = 0;
    window.addEventListener("resize", () => {
        if (resizeTimer) window.clearTimeout(resizeTimer);
        resizeTimer = window.setTimeout(() => {
            resizeTimer = 0;
            record("event.resize", {});
            check("resize");
        }, 150);
    });

    // Gradio's queue: a server round trip is what a Svelte re-render follows.
    try {
        const originalFetch = window.fetch;
        if (typeof originalFetch === "function") {
            window.fetch = function (input, init) {
                try {
                    const url = typeof input === "string" ? input : (input && input.url) || "";
                    if (/\/(queue\/join|run\/predict|queue\/data)/.test(url)) {
                        let detail = {url: url.replace(/^https?:\/\/[^/]+/, "").slice(0, 80)};
                        if (init && typeof init.body === "string") {
                            try {
                                const body = JSON.parse(init.body);
                                detail.fn_index = body.fn_index;
                                detail.trigger_id = body.trigger_id;
                            } catch (error) { /* not JSON */ }
                        }
                        record("gradio.request", detail);
                    }
                } catch (error) { /* never in the way of the page */ }
                return originalFetch.apply(this, arguments);
            };
        }
    } catch (error) { /* never in the way of the page */ }

    // `changed` says whether a call is worth two lines: the dock is placed
    // on every move of the panel and released on every placement while it is
    // not docked, and only the calls that change whether it is docked matter.
    function wrap(owner, name, label, changed) {
        if (!owner || typeof owner[name] !== "function" || owner[name].mcWatched) return;
        const original = owner[name];
        const wrapped = function () {
            const before = changed ? changed() : null;
            const was = snapshot();
            let result;
            try {
                result = original.apply(this, arguments);
            } finally {
                if (!changed || changed() !== before) {
                    record(label + ":before", {was: was.active}, false, was);
                    record(label + ":after", {});
                }
                check(label);
            }
            return result;
        };
        wrapped.mcWatched = true;
        owner[name] = wrapped;
    }

    function hookAssistant(attempt) {
        const NS = window.forgeAssistant;
        if (!NS || !NS.shell || !NS.focus || !NS.dock) {
            if (attempt < 40) window.setTimeout(() => hookAssistant(attempt + 1), 500);
            return;
        }
        const docked = () => !!(NS.dock.isDocked && NS.dock.isDocked());
        const open = () => !!(NS.shell.state && NS.shell.state.panelOpen);
        wrap(NS.focus, "enter", "focus.enter");
        wrap(NS.focus, "exit", "focus.exit");
        wrap(NS.dock, "place", "dock.place", docked);
        wrap(NS.dock, "release", "dock.release", docked);
        wrap(NS.shell, "toggleFocus", "shell.toggleFocus");
        wrap(NS.shell, "toggleDock", "shell.toggleDock");
        wrap(NS.shell, "open", "shell.open", open);
        wrap(NS.shell, "close", "shell.close", open);
        record("hooked", {});
        check("hooked");
    }

    function start() {
        record("start", {userAgent: String(navigator.userAgent || "").slice(0, 120)});
        check("start");
        if (typeof onUiTabChange === "function") {
            onUiTabChange(() => {
                record("event.tabchange", {});
                check("tabchange");
            });
        }
        hookAssistant(0);
    }

    if (typeof onUiLoaded === "function") onUiLoaded(start);
    else if (document.readyState !== "loading") start();
    else document.addEventListener("DOMContentLoaded", start);
})();
