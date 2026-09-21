// Forge Assistant -- the host, as an interface with seven functions.
//
// Everything the assistant knows about Forge is behind `listWorkspaces`,
// `getActiveWorkspace`, `activateWorkspace`, `subscribeNavigation`,
// `resolveWorkspaceRoot`, `listUtilities` and `dispose`. That is not
// architecture for its own sake: the tab bar is the part of this whole feature
// most likely to be different on the machine it runs on, and an adapter is the
// difference between "the picker lists nothing" and "the assistant does not
// load".
//
// Two rules the implementation below keeps, and both have cost somebody a
// working panel somewhere:
//
// *Never match a tab by its display name.* Labels are translated, themed and
// renamed. Tabs are found by the ids Gradio puts on the button and the panel,
// and a tab whose id cannot be found is simply not offered.
//
// *Activation is the host's own operation, forwarded.* The picker clicks the
// host's own tab button. It does not set a class, hide a panel or dispatch a
// synthetic navigation, because every one of those leaves the host's own state
// disagreeing with the screen. Highlighting then follows the host's selection
// rather than the click -- including a switch made somewhere else entirely,
// which is the case that catches an implementation that highlights optimistically.
//
// The utility menu is the same idea for the header's buttons: each entry calls
// the original handler once, with its own enabled state, and a gesture-dependent
// one (a file picker, a link) keeps direct gesture execution rather than being
// routed through anything.

(function () {
    "use strict";

    const NS = (window.forgeAssistant = window.forgeAssistant || {});

    // How long to wait for the host to confirm a tab switch before saying so.
    // A watchdog that *reports* -- it never activates anything a second time,
    // because a second activation is how a slow switch becomes two switches.
    const CONFIRM_TIMEOUT = 4000;

    function app() {
        return (typeof gradioApp === "function" ? gradioApp() : null) || document;
    }

    function all(selector, root) {
        try {
            return Array.prototype.slice.call((root || app()).querySelectorAll(selector));
        } catch (error) {
            return [];
        }
    }

    function visible(node) {
        if (!node) return false;
        const style = window.getComputedStyle ? window.getComputedStyle(node) : null;
        if (style && (style.display === "none" || style.visibility === "hidden")) return false;
        // No offsetParent means this or an ancestor is not laid out -- unless
        // the element is fixed, which has none by definition. The *computed*
        // position, because the focused workspace is fixed by a class and its
        // inline style says nothing; reading the inline value reported the one
        // workspace on screen as the one that was not.
        if (node.offsetParent !== undefined && node.offsetParent === null
            && !(style && style.position === "fixed")) return false;
        return true;
    }

    function depth(node, top) {
        let count = 0;
        let walk = node;
        while (walk && walk !== top) {
            count += 1;
            walk = walk.parentElement;
        }
        return walk === top ? count : Infinity;
    }

    function Host() {
        this.listeners = new Set();
        this.observer = null;
        this.watchdog = null;
        this.pending = "";
        this.menus = new Set();
        this.openMenu = null;
        this.disposed = false;
    }

    // -- the tab bar ------------------------------------------------------- //
    //
    // Forge's tab bar is a Gradio Tabs component: one button per tab, and one
    // panel per tab carrying the id the extension registered. The buttons carry
    // no id of their own on every theme, so the pairing is done by position
    // within the bar -- which is the host's own ordering and therefore the one
    // the picker must preserve.

    // -- the tab bar ------------------------------------------------------- //
    //
    // Gradio renders `gr.Tabs` as
    //
    //     div#tabs.tabs
    //       div.tab-nav[role=tablist]
    //         button[role=tab][aria-controls=tab_x][id=tab_x-button] ...
    //       div#tab_x.tabitem[role=tabpanel][style="display: block|none"]
    //       ...
    //
    // and it renders every NESTED `gr.Tabs` -- the extra-network tabs inside
    // Txt2Img, the mode tabs inside Img2Img, the pages inside Settings -- with
    // exactly the same classes and the same roles. Every line below is written
    // against that fact, because the version that ignored it collected every
    // tab button on the page as a workspace, threw away every panel with a
    // nested tab group inside it (which is most of them), and paired what was
    // left of the two lists by position. Txt2Img came out as a hidden panel or
    // as nothing at all, and focus mode did exactly what it was told with that:
    // hid the page around a panel that was not showing.
    //
    // So: the top-level bar is the SHALLOWEST bar under `#tabs` -- a theme may
    // wrap it, but a nested one is always inside a panel and therefore deeper.
    // A workspace is a panel paired with its button by `aria-controls`, which
    // Gradio writes on every tab button; position is the fallback for a host
    // that does not, never the rule.

    Host.prototype.bar = function () {
        const tabs = app().querySelector("#tabs");
        if (!tabs) return null;
        const bars = all(".tab-nav, [role='tablist']", tabs);
        if (!bars.length) return null;
        let strip = bars[0];
        let shallowest = depth(strip, tabs);
        bars.forEach((bar) => {
            const found = depth(bar, tabs);
            if (found < shallowest) {
                strip = bar;
                shallowest = found;
            }
        });
        // The buttons that are tabs, and not a control a theme has added to
        // the same strip. Direct children as the fallback for a host that
        // writes no roles; anything at all as the last resort.
        let buttons = all("[role='tab']", strip);
        if (!buttons.length) {
            buttons = all("button", strip).filter((button) => button.parentElement === strip);
        }
        if (!buttons.length) buttons = all("button", strip);
        return buttons.length ? {tabs, strip, buttons} : null;
    };

    Host.prototype.panels = function () {
        const tabs = app().querySelector("#tabs");
        if (!tabs) return [];
        const bar = this.bar();
        // A panel is a direct child of `#tabs` with an id. Gradio names them
        // `tab_<name>`; the looser shapes are for a host that does not.
        let found = all(":scope > [id^='tab_']", tabs);
        if (!found.length) found = all(":scope > [role='tabpanel'][id], :scope > .tabitem[id]", tabs);
        if (!found.length) found = all(":scope > div[id]", tabs);
        // Never the bar, and never something wrapped around the bar: a
        // "panel" with the tab bar inside it, focused, is the tab bar laid
        // over the window. Nested tab groups INSIDE a panel are not that --
        // they are the panel's own content -- and excluding a panel for
        // having one is how Txt2Img stopped being a workspace.
        return found.filter((panel) => panel.id && panel.id !== "tabs"
            && !(bar && (panel === bar.strip || panel.contains(bar.strip))));
    };

    Host.prototype.listWorkspaces = function () {
        const bar = this.bar();
        const panels = this.panels();
        if (!bar) return [];
        const byId = new Map();
        panels.forEach((panel) => byId.set(panel.id, panel));
        const taken = new Set();
        const pick = (button, order) => {
            // `aria-controls` first: it is the pairing Gradio itself writes,
            // and it survives a tab that is rendered but not offered (its
            // panel is in the DOM, its button is not), which shifts every
            // positional pairing after it by one.
            const controls = button.getAttribute("aria-controls");
            if (controls && byId.has(controls)) return byId.get(controls);
            const named = button.id && /-button$/.test(button.id)
                ? button.id.replace(/-button$/, "") : "";
            if (named && byId.has(named)) return byId.get(named);
            const positional = panels.filter((panel) => !taken.has(panel));
            return positional[order - taken.size] || null;
        };
        return bar.buttons.map((button, order) => {
            const panel = pick(button, order);
            if (panel) taken.add(panel);
            const id = (panel && panel.id) || ("tab-" + order);
            const capability = this.focusCapability(id, panel);
            return {
                id,
                label: (button.textContent || "").trim() || id,
                order,
                available: !button.disabled,
                focusCapability: capability.ok,
                reason: capability.reason,
                button,
                panel,
            };
        });
    };

    Host.prototype.focusCapability = function (id, panel) {
        if (!panel) return {ok: false, reason: "This workspace has no panel to fill with."};
        if (NS.focus && typeof NS.focus.canFocus === "function") {
            return NS.focus.canFocus(id, panel);
        }
        return {ok: true, reason: ""};
    };

    Host.prototype.getActiveWorkspace = function () {
        const workspaces = this.listWorkspaces();
        if (!workspaces.length) return "";
        // The authority is which *panel* is showing, not which button looks
        // selected: a theme is free to restyle the button and several do.
        const showing = workspaces.find((item) => item.panel && visible(item.panel));
        if (showing) return showing.id;
        const selected = workspaces.find((item) => item.button.classList.contains("selected")
            || item.button.getAttribute("aria-selected") === "true");
        return selected && selected.panel ? selected.id : "";
    };

    Host.prototype.resolveWorkspaceRoot = function (id) {
        const paired = this.listWorkspaces().find((item) => item.id === id);
        if (paired && paired.panel) return paired.panel;
        return this.panels().find((panel) => panel.id === id) || null;
    };

    Host.prototype.activateWorkspace = function (id) {
        const wanted = this.listWorkspaces().find((item) => item.id === id);
        if (!wanted || !wanted.button) {
            return Promise.reject(new Error("That workspace is not on this page."));
        }
        if (this.getActiveWorkspace() === id) return Promise.resolve(id);
        this.pending = id;
        // The host's own control, pressed once. Nothing here sets a class or
        // hides a panel: a switch this code performed itself is a switch the
        // host does not know about.
        wanted.button.click();
        return new Promise((resolve, reject) => {
            const started = Date.now();
            const check = () => {
                if (this.disposed) return reject(new Error("The assistant went away."));
                if (this.getActiveWorkspace() === id) {
                    this.pending = "";
                    this.notify();
                    return resolve(id);
                }
                if (Date.now() - started > CONFIRM_TIMEOUT) {
                    this.pending = "";
                    this.notify();
                    // Reported, never retried. A watchdog that activates again
                    // is a watchdog that switches twice on a slow machine.
                    return reject(new Error("That workspace did not open."));
                }
                window.requestAnimationFrame(check);
                return undefined;
            };
            check();
        });
    };

    Host.prototype.subscribeNavigation = function (listener) {
        this.listeners.add(listener);
        this.watch();
        return () => this.listeners.delete(listener);
    };

    Host.prototype.notify = function () {
        const active = this.getActiveWorkspace();
        this.listeners.forEach((listener) => {
            try {
                listener(active, this.pending);
            } catch (error) {
                console.error("Forge Assistant: a navigation listener failed", error);
            }
        });
    };

    Host.prototype.watch = function () {
        if (this.observer || typeof MutationObserver !== "function") return;
        const tabs = app().querySelector("#tabs");
        if (!tabs) return;
        // Watching the host's own selection rather than our own clicks, so a
        // switch made anywhere -- a header button, another extension's
        // "Send to img2img" -- moves the highlight too.
        this.observer = new MutationObserver(() => this.notify());
        this.observer.observe(tabs, {attributes: true, subtree: true,
                                     attributeFilter: ["class", "style", "aria-selected"]});
    };

    // -- the header's utilities -------------------------------------------- //

    Host.prototype.listUtilities = function () {
        // Two entries, written down rather than discovered.
        //
        // This used to walk the header and offer whatever buttons it found,
        // which on a real installation is "Apply settings", "Reload UI" and a
        // column of controls whose only visible text is the word JSON. A menu
        // assembled from whatever happened to be in the DOM is a menu nobody
        // can predict the contents of, and half of what it found was already
        // one click away on the page behind it.
        //
        // What is here instead is the thing somebody actually opens this menu
        // for: I need this card back, now. Both go to the server rather than
        // pressing a control in the page, because the Settings page's Actions
        // row is a button that may or may not exist under a given theme and a
        // menu entry that silently does nothing on half of them is worse than
        // not offering it.
        return [
            {
                id: "unload-all",
                label: "Unload All Models",
                title: "Release the checkpoint, this extension's model cache and the "
                    + "language model",
                enabled: true,
                kind: "unload",
                scope: "all",
            },
            {
                id: "unload-llm",
                label: "Unload LLM",
                title: "Stop llama-server and release its VRAM and RAM",
                enabled: true,
                kind: "unload",
                scope: "llm",
            },
        ];
    };

    // -- the assistant's own menus ----------------------------------------- //
    //
    // One at a time with each other, and deliberately not with LLM Studio's
    // sheets: the assistant is exempt from that rule in both directions
    // (specification 15.2). It is a second window onto the conversation, not
    // another sheet over it.

    Host.prototype.registerMenu = function (menu) {
        this.menus.add(menu);
        return () => this.menus.delete(menu);
    };

    Host.prototype.openOnly = function (menu) {
        this.menus.forEach((other) => {
            if (other !== menu && typeof other.close === "function") other.close();
        });
        this.openMenu = menu || null;
    };

    Host.prototype.closeMenus = function () {
        const had = !!this.openMenu;
        this.menus.forEach((menu) => {
            if (typeof menu.close === "function") menu.close();
        });
        this.openMenu = null;
        return had;
    };

    Host.prototype.dispose = function () {
        this.disposed = true;
        if (this.observer) {
            this.observer.disconnect();
            this.observer = null;
        }
        window.clearTimeout(this.watchdog);
        this.listeners.clear();
        this.menus.clear();
        this.openMenu = null;
    };

    // -- keyboard arbitration ---------------------------------------------- //
    //
    // One key event, at most one action, in a fixed order (specification 17.1).
    // The order is the whole of it: Escape means six different things on this
    // page and the wrong precedence is how Escape stops a reply somebody was
    // reading instead of closing the menu in front of them.
    //
    // No new document-wide Send or Stop shortcut is installed. Ctrl/Cmd+Enter
    // submits the composer that has focus and nothing else -- never the image
    // Generate button, which is what a global binding would eventually reach.

    function editing(node) {
        if (!node) return false;
        const tag = (node.tagName || "").toLowerCase();
        return tag === "input" || tag === "textarea" || tag === "select"
            || node.isContentEditable === true;
    }

    function inDialog(node) {
        let walk = node;
        while (walk) {
            if (walk.getAttribute && (walk.getAttribute("role") === "dialog"
                || walk.getAttribute("aria-modal") === "true")) return true;
            if (walk.tagName === "DIALOG") return true;
            walk = walk.parentElement;
        }
        return false;
    }

    function escapeOrder(event, context) {
        // Returns the name of the one thing this Escape does, or "" for
        // "leave it to the host". Pure, so the precedence can be asserted
        // without a browser -- which is the only way it is ever going to be
        // asserted, because six of these cases cannot be produced by hand
        // reliably.
        if (context.composing) return "";                  // IME: never ours
        if (context.nativeDialog) return "";               // browser's own
        if (context.hostDialogOpen && !context.insideAssistant) return "";
        if (context.assistantMenuOpen) return "close-menu";
        if (context.assistantEditing) return "cancel-edit";
        if (context.focusActive) return "exit-focus";
        return "";                                         // the host's Escape
    }

    function submitShortcut(event, context) {
        if (context.composing) return false;
        if (!(event.ctrlKey || event.metaKey)) return false;
        if (event.key !== "Enter") return false;
        // Only a chat composer, only a visible one, only the focused one. The
        // image workspaces' Generate is reachable by a shortcut of the host's
        // and this must never be mistaken for it.
        return !!context.focusedComposer;
    }

    NS.Host = Host;
    NS.escapeOrder = escapeOrder;
    NS.submitShortcut = submitShortcut;
    NS.hostEditing = editing;
    NS.hostInDialog = inDialog;

    NS.host = function () {
        if (!NS._host) NS._host = new Host();
        return NS._host;
    };
})();
