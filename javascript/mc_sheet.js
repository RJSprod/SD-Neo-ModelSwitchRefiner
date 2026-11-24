/**
 * Model Chain -- the pop-over every large editor opens: a sheet over 90% of
 * the window, with the page dimmed around its edge.
 *
 * Asked for on 2026-11-23, of Mini Paint NEO's editors: "a pop up that covers
 * 90% of the screen, leaving a hint of [what is] behind it around the edge.
 * This [should] be a comfortable typing view with focus on text field ...
 * consistent and modularized, these should look the same". Mini Paint drew it
 * first; the next day the user asked "you put the gear and system promot
 * editor in minipaint, but those are related LLM studio IMO ... should it be
 * removed from mini paint and put in modelswitchrefiner?" -- and it was. The
 * instructions the system prompt editor edits are this extension's writers'
 * (mc_llm_instructions.py), the gear sits on Forge's prompt boxes beside the
 * rest of this extension's prompt tooling, and both of this extension's system
 * prompt editors -- a writer's and a character's -- are drawn here so that
 * they look the same. Mini Paint's Send to WanGP popup opens its prompt editor
 * from here too (window.mcSheet), with its own Enhance.
 *
 * So there is ONE sheet, and what is built from it:
 *
 *   open(spec)             the sheet alone, which everything below is
 *   prompt(options)        a prompt in a large box: Cancel and Done, and
 *                          Enhance and System prompt when the opener gives
 *                          them. The gear on Forge's prompt boxes opens it
 *                          with neither; Mini Paint's popup with both.
 *   systemPrompt(options)  the writers' six instruction sets, read and written
 *                          over /model-chain/llm/instructions. LLM Studio's
 *                          MiniMax workspace opens it, and Mini Paint's
 *                          Clipboard tab and prompt editor (stacked).
 *   installGears()         a gear on txt2img's and img2img's Prompt and
 *                          Negative prompt; done at every page load
 *
 * The character's system prompt (forge_assistant_system.js) is drawn on
 * open() as well.
 *
 * SHEETS STACK. A second sheet opens over the first, a little smaller so the
 * first shows around it, and Escape, a tap on the dimmed edge and the cross
 * are the top sheet's alone. The keyboard is the top sheet's too: Tab goes
 * round inside it, and Ctrl+Enter is its own -- Forge's script.js presses
 * Generate on Ctrl+Enter from the document, behind the sheet. Each sheet
 * announces itself with `minipaint:overlay` (`{name: "sheet:<name>", open,
 * modal: true, sheet: true}`), the event both extensions' dialogs share, and
 * sits on the dialog layer, `--minipaint-dialog-layer` (2000), over
 * everything either extension draws. The Forge Assistant leaves its panel
 * where it is under a sheet: the sheet covers and dims it, and a gear pressed
 * in the docked settings column comes back to the panel it was pressed in.
 *
 * The layer is the dialog (role="dialog", aria-modal): focus mode's path rule
 * spares whatever says it is one, and so does the assistant's tap outside.
 *
 * CLOSING NEVER THROWS TYPING AWAY. A prompt editor closed by Escape, the
 * edge or the cross keeps what was typed, as Done does; only Cancel puts the
 * text back. A system prompt with an unsaved change asks first: save,
 * discard or keep editing.
 *
 * SIZED TO WHAT CAN BE SEEN. On a phone the on-screen keyboard takes the
 * bottom of the window without making the window smaller, so a sheet sized to
 * the window would put its buttons under the keyboard. The layer follows the
 * visual viewport instead, and the sheet shrinks above the keyboard.
 *
 * NOTHING IS HELD OPEN. Every request is one bounded POST; an enhancement is
 * read again a second at a time while the editor that started it is open,
 * each read answered at once, and stopped when that editor closes.
 *
 * Nothing here writes a prompt, a system prompt or a file name anywhere.
 */
window.mcSheet = window.mcSheet || (function () {
    "use strict";

    const VERSION = 1;
    const CLASS = "mc-sheet";
    const GEAR_CLASS = "mc-prompt-gear";
    const GEAR_HOST_CLASS = "mc-has-gear";
    const SVG_NS = "http://www.w3.org/2000/svg";
    //: The writers' instructions (mc_llm_instructions_api.py), behind the key
    //: LLM Studio's conversation routes take: the page's own, in a header.
    const INSTRUCTIONS_ROUTE = "/model-chain/llm/instructions";
    const KEY_HEADER = "x-mc-conversation-key";
    const KEY_FIELD = "mc-llm-chat-conversation-key";
    const OVERLAY_EVENT = "minipaint:overlay";
    //: One request's deadline. A request that never answers is a connection
    //: held for ever, and a page has one (HTTP/2) or six.
    const ASK_TIMEOUT_MS = 20000;
    //: How often a running enhancement is read again, and how many reads in a
    //: row may go unanswered before the editor stops asking and says so.
    const POLL_MS = 1200;
    const POLL_MISSES = 5;
    //: The first sheet is one step over the dialog layer (Mini Paint's popup is
    //: on it, and a sheet opens from it); each sheet over another ten higher.
    const LAYER_STEP = 10;
    //: Forge's prompt boxes that get a gear: the box, what it is called, the
    //: tab, and Forge's token counter beside it (laid over the box's top edge,
    //: so the gear keeps clear of it).
    const GEAR_TARGETS = [
        ["txt2img_prompt", "Prompt", "txt2img", "txt2img_token_counter"],
        ["txt2img_neg_prompt", "Negative prompt", "txt2img", "txt2img_negative_token_counter"],
        ["img2img_prompt", "Prompt", "img2img", "img2img_token_counter"],
        ["img2img_neg_prompt", "Negative prompt", "img2img", "img2img_negative_token_counter"]
    ];
    const NOTES_KEPT = 30;

    const S = {
        //: The sheets open now, bottom first.
        stack: [],
        keyHandler: null,
        viewportHandler: null,
        opened: 0,
        //: Requests made, by action, for a bug report and the checks.
        asked: {},
        gearWatch: null,
        //: The set the system prompt editor last showed, in this page: where it
        //: opens again when its opener has not said which.
        lastSet: "",
        last: { action: "", code: "", message: "" },
        //: What happened, in words without a prompt in them: state().notes.
        notes: []
    };

    /* ------------------------------------------------------------------ */
    /* Small helpers                                                       */
    /* ------------------------------------------------------------------ */

    function el(tag, className, text) {
        const node = document.createElement(tag);
        if (className) { node.className = className; }
        if (text !== undefined && text !== null) { node.textContent = String(text); }
        return node;
    }

    function button(className, text, title) {
        const node = el("button", className, text);
        node.type = "button";
        if (title) { node.title = title; }
        return node;
    }

    /** A drawn icon: never a character, because a theme's icon swap replaces
     *  whatever text it recognises (the Lobe theme does, ×, ⊞ and its emoji). */
    function icon(paths, circles) {
        const svg = document.createElementNS(SVG_NS, "svg");
        svg.setAttribute("viewBox", "0 0 24 24");
        svg.setAttribute("width", "18");
        svg.setAttribute("height", "18");
        svg.setAttribute("fill", "none");
        svg.setAttribute("stroke", "currentColor");
        svg.setAttribute("stroke-width", "2");
        svg.setAttribute("stroke-linecap", "round");
        svg.setAttribute("stroke-linejoin", "round");
        svg.setAttribute("aria-hidden", "true");
        svg.setAttribute("focusable", "false");
        for (const d of paths || []) {
            const path = document.createElementNS(SVG_NS, "path");
            path.setAttribute("d", d);
            svg.appendChild(path);
        }
        for (const c of circles || []) {
            const circle = document.createElementNS(SVG_NS, "circle");
            circle.setAttribute("cx", String(c[0]));
            circle.setAttribute("cy", String(c[1]));
            circle.setAttribute("r", String(c[2]));
            svg.appendChild(circle);
        }
        return svg;
    }

    const CLOSE_ICON = ["M18 6 6 18", "M6 6l12 12"];
    //: Lucide's "settings" (ISC licence).
    const GEAR_ICON = ["M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"];

    /** What happened, kept for state() and never written anywhere: the
     *  words name sheets and sets, never a prompt or an instruction. */
    function note(message) {
        S.notes.push(String(message));
        if (S.notes.length > NOTES_KEPT) { S.notes.splice(0, S.notes.length - NOTES_KEPT); }
    }

    function app() {
        try {
            if (typeof gradioApp === "function") { return gradioApp() || document; }
        } catch (e) { /* fall through */ }
        return document;
    }

    function byId(id) {
        const root = app();
        return (root.getElementById ? root.getElementById(id) : document.getElementById(id)) || document.getElementById(id) || null;
    }

    /** The page's key for LLM Studio's routes: the Forge Assistant's store
     *  when it is on the page (it renews a key a restart made stale), else
     *  the field Python wrote it into. */
    function pageKey() {
        try {
            const NS = window.forgeAssistant;
            const store = NS && typeof NS.store === "function" ? NS.store() : null;
            if (store && typeof store.key === "function") {
                const found = store.key();
                if (found) { return String(found); }
            }
        } catch (e) { /* the field below */ }
        const holder = document.getElementById(KEY_FIELD);
        if (!holder) { return ""; }
        const input = holder.tagName === "TEXTAREA" || holder.tagName === "INPUT"
            ? holder : holder.querySelector("textarea, input");
        return input ? String(input.value || "") : "";
    }

    function renewKey() {
        try {
            const NS = window.forgeAssistant;
            const store = NS && typeof NS.store === "function" ? NS.store() : null;
            if (store && typeof store.renewKey === "function") { return Promise.resolve(store.renewKey()); }
        } catch (e) { /* nothing to renew with */ }
        return Promise.resolve(false);
    }

    /** One bounded POST, answered with an object whatever happened. */
    function post(url, body, headers) {
        const action = String((body && body.action) || "");
        S.asked[action] = (S.asked[action] || 0) + 1;
        let controller = null;
        try { controller = typeof AbortController === "function" ? new AbortController() : null; } catch (e) { controller = null; }
        const cutoff = controller ? setTimeout(function () { try { controller.abort(); } catch (e) { /* settled */ } }, ASK_TIMEOUT_MS) : 0;
        const init = {
            method: "POST", credentials: "same-origin", cache: "no-store",
            headers: Object.assign({ "Content-Type": "application/json" }, headers || {}),
            body: JSON.stringify(body || {})
        };
        if (controller) { init.signal = controller.signal; }
        return fetch(url, init).then(function (response) {
            return response.json().then(function (answer) {
                const found = answer && typeof answer === "object" ? answer
                    : { ok: false, code: "failed", message: "Forge did not answer (" + response.status + ")." };
                found.status_code = response.status;
                return found;
            }, function () {
                return { ok: false, code: "failed", status_code: response.status,
                         message: "Forge did not answer (" + response.status + ")." };
            });
        }, function (error) {
            const aborted = !!(error && error.name === "AbortError");
            return { ok: false, code: "failed", unreachable: true,
                     message: aborted ? "Forge did not answer within " + Math.round(ASK_TIMEOUT_MS / 1000) + " seconds."
                                      : "Forge is not answering." };
        }).then(function (answer) {
            if (cutoff) { clearTimeout(cutoff); }
            if (answer && answer.ok === false) {
                S.last = { action: action, code: String(answer.code || ""), message: String(answer.message || "") };
            }
            return answer;
        });
    }

    /** A request to the instructions route, with the page's key; asked once
     *  more with a renewed key when a restart made this page's stale. */
    function instructionsPost(body) {
        const send = function () {
            const headers = {};
            headers[KEY_HEADER] = pageKey();
            return post(INSTRUCTIONS_ROUTE, body, headers);
        };
        return send().then(function (answer) {
            if (!answer || answer.status_code !== 401) { return answer; }
            return renewKey().then(function (renewed) { return renewed ? send() : answer; }, function () { return answer; });
        });
    }

    /** Write a host input so the host hears it: the native setter and the two
     *  events, the way Forge's `updateInput` does it. */
    function writeInput(element, text) {
        if (!element) { return false; }
        try {
            const prototype = element.tagName === "TEXTAREA" ? window.HTMLTextAreaElement.prototype : window.HTMLInputElement.prototype;
            const descriptor = Object.getOwnPropertyDescriptor(prototype, "value");
            if (descriptor && descriptor.set) { descriptor.set.call(element, String(text)); } else { element.value = String(text); }
            element.dispatchEvent(new Event("input", { bubbles: true }));
            element.dispatchEvent(new Event("change", { bubbles: true }));
            return true;
        } catch (e) { return false; }
    }

    function announce(name, open) {
        try {
            document.dispatchEvent(new CustomEvent(OVERLAY_EVENT, { detail: { name: "sheet:" + name, open: !!open, modal: true, sheet: true } }));
        } catch (e) { /* nothing here depends on being heard */ }
    }

    /* ------------------------------------------------------------------ */
    /* The sheet                                                           */
    /* ------------------------------------------------------------------ */

    /** The layer over what can be seen: the visual viewport where the browser
     *  has one (a phone's keyboard makes it shorter), else the window. */
    function fitLayer(layer) {
        const view = window.visualViewport;
        if (view && view.width > 0 && view.height > 0) {
            layer.style.left = Math.round(view.offsetLeft) + "px";
            layer.style.top = Math.round(view.offsetTop) + "px";
            layer.style.width = Math.round(view.width) + "px";
            layer.style.height = Math.round(view.height) + "px";
        } else {
            layer.style.left = "0px";
            layer.style.top = "0px";
            layer.style.width = "100%";
            layer.style.height = "100%";
        }
    }

    function fitAll() {
        for (const sheet of S.stack) { fitLayer(sheet.layer); }
    }

    function watchViewport(on) {
        const view = window.visualViewport;
        if (on && !S.viewportHandler) {
            S.viewportHandler = function () { fitAll(); };
            window.addEventListener("resize", S.viewportHandler);
            if (view && typeof view.addEventListener === "function") {
                view.addEventListener("resize", S.viewportHandler);
                view.addEventListener("scroll", S.viewportHandler);
            }
        } else if (!on && S.viewportHandler) {
            window.removeEventListener("resize", S.viewportHandler);
            if (view && typeof view.removeEventListener === "function") {
                view.removeEventListener("resize", S.viewportHandler);
                view.removeEventListener("scroll", S.viewportHandler);
            }
            S.viewportHandler = null;
        }
    }

    function topSheet() { return S.stack.length ? S.stack[S.stack.length - 1] : null; }

    function focusable(root) {
        return Array.from(root.querySelectorAll("button, [href], input, select, textarea, [tabindex]"))
            .filter(function (node) {
                if (node.disabled || node.getAttribute("tabindex") === "-1") { return false; }
                if (node.closest("[hidden]")) { return false; }
                return !!(node.getClientRects && node.getClientRects().length);
            });
    }

    /**
     * The keyboard while a sheet is open, heard on the window before anything
     * else: Escape, Tab and Ctrl+Enter are the top sheet's, and are not passed
     * on - Escape would also close what is under it, and Ctrl+Enter is Forge's
     * Generate.
     */
    function onKey(event) {
        const sheet = topSheet();
        if (!sheet) { return; }
        if (event.key === "Escape") {
            if (event.isComposing) { return; }
            event.preventDefault();
            event.stopPropagation();
            sheet.request("escape");
            return;
        }
        if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
            event.preventDefault();
            event.stopPropagation();
            if (typeof sheet.spec.submit === "function") { sheet.spec.submit(); }
            return;
        }
        if (event.key === "Tab") {
            const items = focusable(sheet.dialog);
            if (!items.length) { event.preventDefault(); return; }
            const first = items[0];
            const last = items[items.length - 1];
            const active = document.activeElement;
            if (!sheet.dialog.contains(active)) {
                event.preventDefault();
                (event.shiftKey ? last : first).focus();
            } else if (event.shiftKey && active === first) {
                event.preventDefault();
                last.focus();
            } else if (!event.shiftKey && active === last) {
                event.preventDefault();
                first.focus();
            }
        }
    }

    function watchKeys(on) {
        if (on && !S.keyHandler) {
            S.keyHandler = onKey;
            window.addEventListener("keydown", S.keyHandler, true);
        } else if (!on && S.keyHandler) {
            window.removeEventListener("keydown", S.keyHandler, true);
            S.keyHandler = null;
        }
    }

    /** No scroll of the page behind the dimmed edge: a wheel or a finger on
     *  it would otherwise move the page the sheet is over. */
    function holdStill(event) {
        if (event.cancelable) { event.preventDefault(); }
    }

    /**
     * Open a sheet. ``spec``:
     *
     *   name       which sheet, for the overlay event and the checks
     *   title      the head's title; ``subtitle`` the line under it
     *   kind       a class of its own (``CLASS + "-" + kind``)
     *   request    what Escape, the edge and the cross do (reason) - the
     *              sheet's own close when not given
     *   submit     what Ctrl+Enter does, if anything
     *   closed     told after the sheet has gone (reason)
     *
     * Returns the sheet: its ``body``, ``start`` and ``end`` (the two halves
     * of its foot), ``close(reason)``, ``request(reason)``, ``say(text,
     * failure)``, ``setTitle(text, sub)``, ``ask(text, labels)`` and
     * ``isTop()``.
     */
    function open(spec) {
        spec = spec || {};
        const depth = S.stack.length;
        const layer = el("div", CLASS + "-layer");
        layer.dataset.depth = String(depth);
        layer.dataset.sheet = String(spec.name || "sheet");
        layer.style.zIndex = "calc(var(--minipaint-dialog-layer, 2000) + " + (LAYER_STEP * (depth + 1)) + ")";
        // The layer is the dialog: what focus mode's path rule and the Forge
        // Assistant's tap outside both spare, by its role.
        layer.setAttribute("role", "dialog");
        layer.setAttribute("aria-modal", "true");
        S.opened += 1;
        const titleId = CLASS + "-title-" + S.opened;
        layer.setAttribute("aria-labelledby", titleId);
        const backdrop = el("div", CLASS + "-backdrop");
        layer.appendChild(backdrop);
        const dialog = el("div", CLASS + (spec.kind ? " " + CLASS + "-" + spec.kind : ""));

        const head = el("div", CLASS + "-head");
        const titles = el("div", CLASS + "-titles");
        const title = el("b", CLASS + "-title", spec.title || "");
        title.id = titleId;
        const subtitle = el("small", CLASS + "-sub", spec.subtitle || "");
        subtitle.hidden = !spec.subtitle;
        titles.appendChild(title);
        titles.appendChild(subtitle);
        const close = button(CLASS + "-close", "", "Close (Escape)");
        close.setAttribute("aria-label", "Close");
        close.appendChild(icon(CLOSE_ICON));
        head.appendChild(titles);
        head.appendChild(close);
        dialog.appendChild(head);

        const body = el("div", CLASS + "-body");
        dialog.appendChild(body);
        const message = el("div", CLASS + "-message");
        message.setAttribute("role", "status");
        message.hidden = true;
        dialog.appendChild(message);
        const confirm = el("div", CLASS + "-confirm");
        confirm.hidden = true;
        confirm.setAttribute("role", "alertdialog");
        dialog.appendChild(confirm);
        const foot = el("div", CLASS + "-foot");
        const start = el("div", CLASS + "-foot-start");
        const end = el("div", CLASS + "-foot-end");
        foot.appendChild(start);
        foot.appendChild(end);
        dialog.appendChild(foot);
        layer.appendChild(dialog);

        const sheet = {
            name: String(spec.name || "sheet"),
            spec: spec,
            layer: layer,
            dialog: dialog,
            head: head,
            title: title,
            subtitle: subtitle,
            body: body,
            foot: foot,
            start: start,
            end: end,
            message: message,
            confirm: confirm,
            cross: close,
            closed: false,
            before: document.activeElement,
            reason: "",
            asking: null
        };
        sheet.isTop = function () { return topSheet() === sheet; };
        sheet.say = function (text, failure) {
            if (!text) { message.hidden = true; message.textContent = ""; return; }
            message.textContent = String(text);
            message.classList.toggle(CLASS + "-message-failure", !!failure);
            message.hidden = false;
        };
        sheet.setTitle = function (text, sub) {
            title.textContent = String(text || "");
            subtitle.textContent = String(sub || "");
            subtitle.hidden = !sub;
        };
        /**
         * A question in place of the foot: an unsaved change, say. ``labels``
         * names the three answers (``keep``, ``discard``, ``save``; a missing
         * one is not offered). Resolves with the one pressed; Escape is Keep
         * editing while it is asked.
         */
        sheet.ask = function (text, labels) {
            const names = Object.assign({ keep: "Keep editing", discard: "Discard", save: "Save" }, labels || {});
            if (sheet.asking) { sheet.asking.resolve("keep"); }
            confirm.textContent = "";
            confirm.appendChild(el("span", CLASS + "-confirm-text", text));
            return new Promise(function (resolve) {
                const answer = function (choice) {
                    if (sheet.asking !== asking) { return; }
                    sheet.asking = null;
                    confirm.hidden = true;
                    foot.hidden = false;
                    resolve(choice);
                };
                const asking = { resolve: answer };
                sheet.asking = asking;
                let first = null;
                for (const choice of ["keep", "discard", "save"]) {
                    if (!names[choice]) { continue; }
                    const node = button(CLASS + "-" + choice + (choice === "save" ? "-then" : ""), names[choice]);
                    node.addEventListener("click", function () { answer(choice); });
                    confirm.appendChild(node);
                    if (!first) { first = node; }
                }
                confirm.hidden = false;
                foot.hidden = true;
                try { if (first) { first.focus(); } } catch (e) { /* fine */ }
            });
        };
        sheet.close = function (reason) {
            if (sheet.closed) { return false; }
            sheet.closed = true;
            sheet.reason = String(reason || "close");
            if (sheet.asking) { sheet.asking.resolve("keep"); }
            const index = S.stack.indexOf(sheet);
            if (index >= 0) { S.stack.splice(index, 1); }
            try { layer.remove(); } catch (e) { /* already gone */ }
            if (!S.stack.length) {
                watchKeys(false);
                watchViewport(false);
                document.documentElement.classList.remove(CLASS + "-open");
            }
            announce(sheet.name, false);
            // Back where the keyboard was: the button that opened this sheet,
            // or the sheet under it.
            const back = sheet.before;
            const under = topSheet();
            if (back && back.isConnected && (!under || under.dialog.contains(back))) {
                try { back.focus({ preventScroll: true }); } catch (e) { /* not worth an exception */ }
            } else if (under) {
                const items = focusable(under.dialog);
                if (items.length) { try { items[0].focus({ preventScroll: true }); } catch (e) { /* fine */ } }
            }
            if (typeof spec.closed === "function") {
                try { spec.closed(sheet.reason); } catch (e) { /* the sheet is gone either way */ }
            }
            return true;
        };
        sheet.request = function (reason) {
            if (sheet.closed) { return false; }
            // A question on screen is answered "keep editing" by Escape.
            if (sheet.asking && reason === "escape") { sheet.asking.resolve("keep"); return false; }
            if (typeof spec.request === "function") { return spec.request(reason); }
            return sheet.close(reason);
        };

        close.addEventListener("click", function () { sheet.request("close"); });
        backdrop.addEventListener("click", function (event) {
            if (event.target === backdrop && sheet.isTop()) { sheet.request("outside"); }
        });
        backdrop.addEventListener("wheel", holdStill, { passive: false });
        backdrop.addEventListener("touchmove", holdStill, { passive: false });

        S.stack.push(sheet);
        fitLayer(layer);
        (document.body || document.documentElement).appendChild(layer);
        document.documentElement.classList.add(CLASS + "-open");
        watchKeys(true);
        watchViewport(true);
        announce(sheet.name, true);
        note("opened " + sheet.name + " at depth " + depth);
        return sheet;
    }

    /* ------------------------------------------------------------------ */
    /* The prompt editor                                                    */
    /* ------------------------------------------------------------------ */

    /**
     * A prompt in a large box. ``options``:
     *
     *   name, title, subtitle, placeholder, text   as they read
     *   done(text, info)   told on Done - and on Escape, the edge and the
     *                      cross, which keep what was typed. ``info`` is
     *                      ``{prewritten, changed}``: ``prewritten`` is
     *                      ``{typed, written}`` when the text is a prompt the
     *                      enhancer wrote here and the user took, else null.
     *   cancel()           told on Cancel, which puts the text back
     *   prewritten         the opener's own ``{typed, written}``, when the text
     *                      it hands over is one
     *   enhance            {plan(text), start(text), status(id), cancel(id)}:
     *                      Enhance and its result, when given
     *   systemPrompt(plan) System prompt, when given: open the instructions
     *                      the plan names, stacked
     *   doneLabel          Done's word, when it is another
     *
     * Enhance writes the box's text and shows the result under it (beside it
     * on a wide sheet): Use this puts the result in the box, Roll again asks
     * once more from the same text, and the box keeps the words typed until
     * one is used. A used result is ``prewritten``: the opener sends it as it
     * is rather than enhance it a second time.
     */
    function prompt(options) {
        const o = options || {};
        const original = String(o.text || "");
        let prewritten = o.prewritten && typeof o.prewritten === "object"
            ? { typed: String(o.prewritten.typed || ""), written: String(o.prewritten.written || "") } : null;
        let plan = null;
        let finished = false;
        const E = { ticket: 0, id: "", source: "", running: false, result: "", meta: "", timer: 0, misses: 0, state: "" };

        const sheet = open({
            name: o.name || "prompt",
            title: o.title || "Prompt",
            subtitle: o.subtitle || "",
            kind: "prompt",
            request: function () { return finish(true); },
            submit: function () { finish(true); },
            closed: function () { stopPolling(true); }
        });

        const area = el("div", CLASS + "-columns");
        const editor = el("div", CLASS + "-editor");
        const box = el("textarea", CLASS + "-text");
        box.value = original;
        box.placeholder = String(o.placeholder || "");
        box.setAttribute("aria-label", o.title || "Prompt");
        box.spellcheck = true;
        editor.appendChild(box);
        const info = el("div", CLASS + "-info");
        info.hidden = true;
        editor.appendChild(info);
        area.appendChild(editor);

        const result = el("section", CLASS + "-result");
        result.hidden = true;
        result.setAttribute("aria-live", "polite");
        const resultHead = el("div", CLASS + "-result-head");
        const resultTitle = el("b", CLASS + "-result-title", "Enhanced");
        const resultMeta = el("span", CLASS + "-result-meta", "");
        resultHead.appendChild(resultTitle);
        resultHead.appendChild(resultMeta);
        const resultText = el("div", CLASS + "-result-text");
        resultText.tabIndex = 0;
        const resultActions = el("div", CLASS + "-result-actions");
        const use = button(CLASS + "-use", "Use this", "Put the enhanced prompt in the box");
        const again = button(CLASS + "-again", "Roll again", "Ask the enhancer once more, from the same words");
        const back = button(CLASS + "-back", "Back to what I typed", "Put your own words back in the box");
        const stop = button(CLASS + "-stop", "Stop", "Stop this enhancement");
        const dismiss = button(CLASS + "-dismiss", "Dismiss", "Put this result away");
        resultActions.appendChild(use);
        resultActions.appendChild(again);
        resultActions.appendChild(back);
        resultActions.appendChild(stop);
        resultActions.appendChild(dismiss);
        result.appendChild(resultHead);
        result.appendChild(resultText);
        result.appendChild(resultActions);
        area.appendChild(result);
        sheet.body.appendChild(area);

        const enhanceButton = o.enhance ? button(CLASS + "-enhance", "Enhance", "Write this prompt through the enhancer and show what it wrote") : null;
        const systemButton = typeof o.systemPrompt === "function"
            ? button(CLASS + "-system-open", "System prompt", "Edit the instructions the enhancer writes under") : null;
        if (enhanceButton) { sheet.start.appendChild(enhanceButton); }
        if (systemButton) { sheet.start.appendChild(systemButton); }
        const cancelButton = button(CLASS + "-cancel", "Cancel", "Put the text back as it was and close");
        const doneButton = button(CLASS + "-done", o.doneLabel || "Done", "Keep this text and close (Ctrl+Enter)");
        sheet.end.appendChild(cancelButton);
        sheet.end.appendChild(doneButton);

        function inUse() { return !!(prewritten && box.value === prewritten.written); }

        function render() {
            const hasResult = !!(E.running || E.result || E.state);
            result.hidden = !hasResult;
            area.classList.toggle(CLASS + "-with-result", hasResult);
            result.dataset.state = E.running ? "running" : (E.state || "");
            if (E.running) {
                resultTitle.textContent = "Enhancing…";
            } else if (E.state === "done") {
                resultTitle.textContent = inUse() && box.value === E.result ? "Enhanced - in the box" : "Enhanced";
            } else if (E.state) {
                resultTitle.textContent = E.state === "cancelled" ? "Stopped" : "Not enhanced";
            }
            resultMeta.textContent = E.meta;
            stop.hidden = !E.running;
            use.hidden = E.running || E.state !== "done";
            use.disabled = box.value === E.result;
            again.hidden = E.running || !E.source;
            back.hidden = !(prewritten && !E.running);
            dismiss.hidden = E.running;
            if (enhanceButton) {
                enhanceButton.disabled = E.running || !!(plan && plan.ok === false) || !!(plan && plan.available === false);
                enhanceButton.textContent = E.running ? "Enhancing…" : "Enhance";
            }
        }

        function setResultText(text) {
            resultText.textContent = String(text || "");
        }

        function stopPolling(cancelRunning) {
            if (E.timer) { clearTimeout(E.timer); E.timer = 0; }
            if (cancelRunning && E.running && E.id && o.enhance && typeof o.enhance.cancel === "function") {
                const id = E.id;
                try { o.enhance.cancel(id); } catch (e) { /* the writer finishes on its own */ }
                note("an enhancement still running was stopped as its editor closed");
            }
            if (cancelRunning) { E.ticket += 1; E.running = false; }
        }

        function factsLine(answer) {
            const parts = [];
            if (answer && answer.variant_label) { parts.push(String(answer.variant_label)); }
            const described = answer && Array.isArray(answer.described) ? answer.described : [];
            parts.push(described.length ? "describing " + described.join(" and ") : "from the words alone");
            return parts.join(", ");
        }

        function showPlan(answer) {
            plan = answer || null;
            if (!answer) { info.hidden = true; return; }
            if (answer.ok === false) {
                info.textContent = String(answer.message || "Enhance is not available for this picture now.");
                info.dataset.state = "blocked";
            } else if (answer.available === false) {
                info.textContent = String((answer.availability && answer.availability.text) || "The enhancer is not available now.");
                info.dataset.state = "blocked";
            } else {
                info.textContent = "Enhance writes for " + factsLine(answer) + "."
                    + (answer.left_out && answer.left_out.length ? " " + answer.left_out.join(" and ") + " left out of it." : "");
                info.dataset.state = "ready";
            }
            info.hidden = false;
            render();
        }

        function askPlan() {
            if (!o.enhance || typeof o.enhance.plan !== "function") { return Promise.resolve(null); }
            return Promise.resolve(o.enhance.plan(box.value)).then(function (answer) {
                if (finished) { return answer; }
                showPlan(answer && typeof answer === "object" ? answer : null);
                return answer;
            }, function () { return null; });
        }

        function poll(ticket) {
            E.timer = setTimeout(function () {
                E.timer = 0;
                if (finished || ticket !== E.ticket) { return; }
                Promise.resolve(o.enhance.status(E.id)).then(function (answer) {
                    if (finished || ticket !== E.ticket) { return; }
                    answer = answer && typeof answer === "object" ? answer : { ok: false };
                    if (answer.ok === false) {
                        E.misses += 1;
                        if (E.misses >= POLL_MISSES) {
                            E.running = false;
                            E.state = "lost";
                            E.meta = "";
                            setResultText(String(answer.message || "Forge stopped answering about this enhancement."));
                            render();
                            return;
                        }
                        poll(ticket);
                        return;
                    }
                    E.misses = 0;
                    const state = String(answer.state || "");
                    if (state === "done") {
                        E.running = false;
                        E.state = "done";
                        E.result = String(answer.prompt || "");
                        const seconds = Number(answer.elapsed) > 0 ? " · " + Math.round(Number(answer.elapsed)) + " s" : "";
                        E.meta = E.meta + seconds;
                        setResultText(E.result);
                        render();
                        note("an enhancement finished" + seconds);
                        return;
                    }
                    if (state === "failed" || state === "cancelled" || state === "lost") {
                        E.running = false;
                        E.state = state;
                        setResultText(String(answer.message || (state === "cancelled" ? "The enhancement was stopped." : "The enhancement failed.")));
                        render();
                        return;
                    }
                    const ahead = Number(answer.position) > 0 ? "Waiting for the LLM, " + answer.position + " ahead." : "";
                    setResultText(String(answer.stage || "") || ahead || (state === "queued" ? "Waiting for the LLM…" : "Writing…"));
                    poll(ticket);
                });
            }, POLL_MS);
        }

        function enhance(source) {
            const text = String(source == null ? box.value : source);
            if (!o.enhance || E.running) { return Promise.resolve(null); }
            if (!text.trim()) {
                sheet.say("Type a prompt to enhance first.", true);
                try { box.focus(); } catch (e) { /* fine */ }
                return Promise.resolve(null);
            }
            sheet.say("");
            E.ticket += 1;
            const ticket = E.ticket;
            E.running = true;
            E.state = "";
            E.result = "";
            E.id = "";
            E.misses = 0;
            E.source = text;
            E.meta = plan && plan.ok !== false ? factsLine(plan) : "";
            setResultText("Asking the enhancer…");
            render();
            return Promise.resolve(o.enhance.start(text)).then(function (answer) {
                answer = answer && typeof answer === "object" ? answer : { ok: false };
                if (finished || ticket !== E.ticket) {
                    // Closed or stopped while it was being asked for: a writer
                    // that did start is told to stop.
                    if (answer.ok && answer.llm_id && typeof o.enhance.cancel === "function") { o.enhance.cancel(answer.llm_id); }
                    return answer;
                }
                if (!answer.ok) {
                    E.running = false;
                    E.state = "failed";
                    E.meta = "";
                    setResultText(String(answer.message || "The prompt could not be enhanced."));
                    render();
                    return answer;
                }
                E.id = String(answer.llm_id || "");
                E.meta = factsLine(answer) + (answer.override ? ", under your system prompt" : "");
                setResultText("Waiting for the LLM…");
                render();
                poll(ticket);
                return answer;
            });
        }

        function stopEnhance() {
            if (!E.running) { return; }
            const id = E.id;
            E.ticket += 1;
            E.running = false;
            E.state = "cancelled";
            setResultText("Stopped. Nothing was written.");
            if (E.timer) { clearTimeout(E.timer); E.timer = 0; }
            if (id && typeof o.enhance.cancel === "function") { o.enhance.cancel(id); }
            render();
        }

        function useResult() {
            if (E.state !== "done" || !E.result) { return; }
            const typed = prewritten && box.value === prewritten.written ? prewritten.typed : box.value;
            prewritten = { typed: typed, written: E.result };
            box.value = E.result;
            render();
            try { box.focus(); } catch (e) { /* fine */ }
        }

        function backToTyped() {
            if (!prewritten) { return; }
            box.value = prewritten.typed;
            prewritten = null;
            render();
            try { box.focus(); } catch (e) { /* fine */ }
        }

        function finish(keep) {
            if (finished) { return false; }
            finished = true;
            stopPolling(true);
            if (keep) {
                const text = box.value;
                // A used result is still the enhancer's while it starts the
                // box - edited after, it is a refinement of it - and nothing
                // once the box is emptied.
                const written = prewritten && text.trim() ? { typed: prewritten.typed, written: prewritten.written } : null;
                if (typeof o.done === "function") {
                    try { o.done(text, { prewritten: written, changed: text !== original }); } catch (e) { /* the sheet closes */ }
                }
                return sheet.close("done");
            }
            if (typeof o.cancel === "function") {
                try { o.cancel(); } catch (e) { /* the sheet closes */ }
            }
            return sheet.close("cancel");
        }

        box.addEventListener("input", function () {
            if (prewritten && !box.value.trim()) { prewritten = null; }
            render();
        });
        if (enhanceButton) { enhanceButton.addEventListener("click", function () { enhance(); }); }
        if (systemButton) { systemButton.addEventListener("click", function () { o.systemPrompt(plan); }); }
        use.addEventListener("click", useResult);
        again.addEventListener("click", function () { enhance(E.source); });
        back.addEventListener("click", backToTyped);
        stop.addEventListener("click", stopEnhance);
        dismiss.addEventListener("click", function () {
            E.state = "";
            E.result = "";
            E.meta = "";
            render();
        });
        cancelButton.addEventListener("click", function () { finish(false); });
        doneButton.addEventListener("click", function () { finish(true); });

        render();
        // The box is what the sheet is for: the cursor goes there now, inside
        // the press that opened it, which is what lets a phone raise its
        // keyboard; at the end of the text, where typing carries on.
        try {
            box.focus({ preventScroll: true });
            const end = box.value.length;
            box.setSelectionRange(end, end);
            box.scrollTop = box.scrollHeight;
        } catch (e) { /* not worth an exception */ }
        askPlan();
        note("text editor opened (" + sheet.name + ")");

        sheet.text = function () { return box.value; };
        sheet.box = box;
        sheet.enhance = enhance;
        sheet.finish = finish;
        sheet.state = function () {
            return { running: E.running, state: E.state, id: E.id, result: E.result, meta: E.meta, source: E.source,
                     prewritten: prewritten ? { typed: prewritten.typed, written: prewritten.written } : null,
                     plan: plan ? Object.assign({}, plan) : null, info: info.hidden ? "" : info.textContent };
        };
        sheet.replan = askPlan;
        return sheet;
    }

    /* ------------------------------------------------------------------ */
    /* The system prompt editor                                             */
    /* ------------------------------------------------------------------ */

    /** A promise that gives up: an opener's choice that never answers must
     *  not hold the editor at "Reading…" for ever. */
    function bounded(promise, fallback) {
        return new Promise(function (resolve) {
            const cutoff = setTimeout(function () { resolve(fallback); }, ASK_TIMEOUT_MS);
            Promise.resolve(promise).then(function (value) { clearTimeout(cutoff); resolve(value); },
                                          function () { clearTimeout(cutoff); resolve(fallback); });
        });
    }

    /**
     * The instructions the writers write under, one set at a time, read and
     * written over this extension's route. ``options``:
     *
     *   name           the sheet's name (``system-prompt``)
     *   variant, mode  the set to open on (``fl2va``, ``ref2va``, ``ltx23`` or
     *                  ``ltx25``; ``text`` or ``image``), when the opener
     *                  knows it - else the set this page showed last
     *   choose(info)   a promise of ``{variant, mode, on}``, asked while the
     *                  sheet opens, ``info.fallback`` being the set this page
     *                  showed last: Mini Paint asks its own server which set
     *                  WanGP's model and the composer's pictures call for, and
     *                  where its switch stands (``on``)
     *   toggle         an opener's own switch, drawn above the sets:
     *                  ``{label, title, on, set(flag)}``, ``set`` answering
     *                  ``{ok, on, status, message}`` (Mini Paint's Enhance
     *                  switch, in its Clipboard tab's editor)
     *
     * Save override keeps the box for every request from then on; Restore
     * default forgets it; Reload reads it again. An unsaved change is asked
     * about before the sheet closes or moves to another set.
     */
    function systemPrompt(options) {
        const o = options || {};
        let view = null;
        let busy = false;
        const sheet = open({
            name: o.name || "system-prompt",
            title: "System prompt",
            subtitle: "Prompt writers",
            kind: "system",
            request: function (reason) { return leave(reason); },
            submit: function () { save(); }
        });

        const line = el("div", CLASS + "-availability", "Reading the writers' instructions…");
        line.dataset.state = "unknown";
        sheet.body.appendChild(line);

        let toggle = null;
        if (o.toggle && typeof o.toggle.set === "function") {
            const switchRow = el("div", CLASS + "-row");
            toggle = button(CLASS + "-switch", "", o.toggle.title || "");
            toggle.setAttribute("role", "switch");
            toggle.appendChild(el("span", CLASS + "-switch-track"));
            switchRow.appendChild(toggle);
            switchRow.appendChild(el("span", CLASS + "-switch-label", o.toggle.label || ""));
            sheet.body.appendChild(switchRow);
            setSwitch(!!o.toggle.on);
        }

        const pickRow = el("div", CLASS + "-row " + CLASS + "-picks");
        const setLabel = el("label", CLASS + "-pick");
        setLabel.appendChild(el("span", "", "Writer"));
        const setPick = el("select", CLASS + "-set");
        setLabel.appendChild(setPick);
        const modeLabel = el("label", CLASS + "-pick");
        modeLabel.appendChild(el("span", "", "Instructions used"));
        const modePick = el("select", CLASS + "-mode");
        modeLabel.appendChild(modePick);
        pickRow.appendChild(setLabel);
        pickRow.appendChild(modeLabel);
        sheet.body.appendChild(pickRow);

        const box = el("textarea", CLASS + "-text " + CLASS + "-mono");
        box.setAttribute("aria-label", "System prompt");
        box.spellcheck = false;
        box.disabled = true;
        box.placeholder = "Reading…";
        sheet.body.appendChild(box);
        const state = el("div", CLASS + "-state", "");
        sheet.body.appendChild(state);

        const restore = button(CLASS + "-restore", "Restore default", "Forget the override for this set");
        const reload = button(CLASS + "-reload", "Reload", "Read this set again, as it is saved");
        sheet.start.appendChild(restore);
        sheet.start.appendChild(reload);
        const closeButton = button(CLASS + "-close-text", "Close", "Close (Escape)");
        const saveButton = button(CLASS + "-save", "Save override", "Use these instructions for every request that brings none of its own (Ctrl+Enter)");
        sheet.end.appendChild(closeButton);
        sheet.end.appendChild(saveButton);

        function dirty() { return !!view && box.value !== view.text; }

        function fill(select, items, chosen) {
            const wanted = items.map(function (item) { return String(item.id) + "\u0000" + String(item.label); }).join("\u0001");
            if (select.dataset.items !== wanted) {
                select.textContent = "";
                for (const item of items) {
                    const option = el("option", "", item.label);
                    option.value = String(item.id);
                    select.appendChild(option);
                }
                select.dataset.items = wanted;
            }
            select.value = String(chosen || "");
        }

        function setSwitch(flag) {
            if (!toggle) { return; }
            toggle.setAttribute("aria-checked", flag ? "true" : "false");
            toggle.classList.toggle(CLASS + "-switch-on", !!flag);
        }

        function setBusy(flag) {
            busy = !!flag;
            for (const node of [reload, saveButton, toggle, setPick, modePick]) { if (node) { node.disabled = busy; } }
            restore.disabled = busy || !view || view.source !== "override";
            // Read-only while a request is out, never disabled: a disabled box
            // drops the keyboard, and the cursor would be gone after a save.
            box.disabled = !view;
            box.readOnly = busy;
            sheet.dialog.classList.toggle(CLASS + "-busy", busy);
        }

        function showView(next) {
            if (!next || typeof next !== "object") { return; }
            view = { variant: String(next.variant || ""), mode: String(next.mode || ""), text: String(next.text || ""),
                     source: String(next.source || ""), label: String(next.label || ""), line: String(next.line || "") };
            box.value = view.text;
            box.placeholder = "";
            state.textContent = view.line;
            state.dataset.source = view.source;
            setPick.value = view.variant;
            modePick.value = view.mode;
            restore.disabled = busy || view.source !== "override";
            box.disabled = false;
            S.lastSet = view.variant;
            saveButton.classList.remove(CLASS + "-unsaved");
            sheet.setTitle("System prompt", view.label);
        }

        function describe(wanted) {
            setBusy(true);
            const body = { action: "describe", variant: (wanted && wanted.variant) || "",
                           mode: (wanted && wanted.mode) || "", fallback: S.lastSet };
            return instructionsPost(body).then(function (answer) {
                if (sheet.closed) { return answer; }
                setBusy(false);
                if (!answer || !answer.ok) {
                    sheet.say((answer && answer.message) || "The writers' instructions could not be read.", true);
                    if (view) { setPick.value = view.variant; modePick.value = view.mode; }
                    return answer;
                }
                fill(setPick, Array.isArray(answer.sets) ? answer.sets : [], answer.view && answer.view.variant);
                fill(modePick, Array.isArray(answer.modes) ? answer.modes : [], answer.view && answer.view.mode);
                const available = answer.availability || {};
                line.textContent = String(available.text || "");
                line.dataset.state = String(available.state || "unknown");
                showView(answer.view);
                sheet.say("");
                return answer;
            });
        }

        function save() {
            if (busy || !view) { return Promise.resolve(false); }
            if (!box.value.trim()) {
                sheet.say("A system prompt cannot be blank. Restore default puts the shipped one back.", true);
                return Promise.resolve(false);
            }
            setBusy(true);
            const saving = { variant: view.variant, mode: view.mode };
            return instructionsPost({ action: "save", variant: saving.variant, mode: saving.mode, text: box.value }).then(function (answer) {
                if (sheet.closed) { return !!(answer && answer.ok); }
                setBusy(false);
                if (!answer || !answer.ok) {
                    sheet.say((answer && answer.message) || "The override was not saved.", true);
                    return false;
                }
                showView(answer.view);
                sheet.say(String(answer.status || "Override saved."));
                note("instructions saved (" + saving.variant + ", " + saving.mode + ")");
                return true;
            });
        }

        function restoreDefault() {
            if (busy || !view) { return; }
            setBusy(true);
            instructionsPost({ action: "restore", variant: view.variant, mode: view.mode }).then(function (answer) {
                if (sheet.closed) { return; }
                setBusy(false);
                if (!answer || !answer.ok) {
                    sheet.say((answer && answer.message) || "The default could not be restored.", true);
                    return;
                }
                showView(answer.view);
                sheet.say(String(answer.status || "Back to the default."));
            });
        }

        function flip() {
            if (busy || !toggle) { return; }
            const wanted = toggle.getAttribute("aria-checked") !== "true";
            setBusy(true);
            bounded(o.toggle.set(wanted), { ok: false, message: "The switch did not answer." }).then(function (answer) {
                if (sheet.closed) { return; }
                setBusy(false);
                if (!answer || !answer.ok) {
                    sheet.say((answer && answer.message) || "The switch could not be changed.", true);
                    return;
                }
                setSwitch(answer.on === true);
                sheet.say(String(answer.status || ""));
            });
        }

        /** An unsaved change, asked about: save it, drop it, or stay. */
        function ask(then) {
            sheet.ask("This change to the system prompt is not saved.").then(function (choice) {
                if (choice === "keep") {
                    try { box.focus(); } catch (e) { /* fine */ }
                    return;
                }
                if (choice === "discard") { then(); return; }
                save().then(function (ok) { if (ok) { then(); } });
            });
        }

        function leave(reason) {
            if (dirty()) {
                ask(function () { sheet.close(reason); });
                return false;
            }
            return sheet.close(reason);
        }

        function pick() {
            const wanted = { variant: setPick.value, mode: modePick.value };
            if (view && wanted.variant === view.variant && wanted.mode === view.mode) { return; }
            if (dirty()) {
                // Back to the set on screen while the question is asked.
                setPick.value = view.variant;
                modePick.value = view.mode;
                ask(function () { describe(wanted); });
                return;
            }
            describe(wanted);
        }

        if (toggle) { toggle.addEventListener("click", flip); }
        setPick.addEventListener("change", pick);
        modePick.addEventListener("change", pick);
        restore.addEventListener("click", restoreDefault);
        reload.addEventListener("click", function () {
            if (!view) { describe({ variant: o.variant || "", mode: o.mode || "" }); return; }
            const wanted = { variant: view.variant, mode: view.mode };
            if (dirty()) { ask(function () { describe(wanted); }); return; }
            describe(wanted);
        });
        saveButton.addEventListener("click", function () { save(); });
        closeButton.addEventListener("click", function () { leave("close"); });
        box.addEventListener("input", function () {
            saveButton.classList.toggle(CLASS + "-unsaved", dirty());
        });

        try { box.focus({ preventScroll: true }); } catch (e) { /* fine */ }
        const chosen = typeof o.choose === "function"
            ? bounded(o.choose({ fallback: S.lastSet }), null) : Promise.resolve({ variant: o.variant || "", mode: o.mode || "" });
        const first = chosen.then(function (wanted) {
            if (sheet.closed) { return null; }
            wanted = wanted && typeof wanted === "object" ? wanted : {};
            if (toggle && typeof wanted.on === "boolean") { setSwitch(wanted.on); }
            return describe({ variant: wanted.variant || "", mode: wanted.mode || "" });
        }).then(function (answer) {
            if (!sheet.closed && answer && answer.ok) {
                try { box.focus({ preventScroll: true }); box.setSelectionRange(0, 0); box.scrollTop = 0; } catch (e) { /* fine */ }
            }
            return answer;
        });
        note("instructions editor opened");

        sheet.box = box;
        sheet.ready = first;
        sheet.save = save;
        sheet.state = function () {
            return { view: view ? Object.assign({}, view) : null, dirty: dirty(), busy: busy, asking: !!sheet.asking,
                     switched: toggle ? toggle.getAttribute("aria-checked") === "true" : null,
                     availability: line.textContent };
        };
        return sheet;
    }

    /* ------------------------------------------------------------------ */
    /* The gear on Forge's prompt boxes                                     */
    /* ------------------------------------------------------------------ */

    /**
     * Where the gear sits: inside the box's top right corner, clear of a
     * classic scroll bar and of Forge's token counter, which Forge lays over
     * the box's top edge. Measured, because a theme pads the box and a
     * platform draws its scroll bar however it likes.
     */
    function placeGear(host, gear, counterId) {
        const area = host.querySelector("textarea");
        if (!area || !gear.isConnected) { return; }
        const outer = host.getBoundingClientRect();
        const inner = area.getBoundingClientRect();
        if (!outer.width || !inner.width) { return; }
        let style;
        try { style = window.getComputedStyle(area); } catch (e) { style = null; }
        const borderTop = style ? parseFloat(style.borderTopWidth) || 0 : 0;
        const borderRight = style ? parseFloat(style.borderRightWidth) || 0 : 0;
        const borderLeft = style ? parseFloat(style.borderLeftWidth) || 0 : 0;
        const scrollbar = Math.max(0, area.offsetWidth - area.clientWidth - borderLeft - borderRight);
        let top = inner.top - outer.top + borderTop + 4;
        const right = outer.right - inner.right + borderRight + scrollbar + 4;
        const counter = counterId ? byId(counterId) : null;
        if (counter) {
            const box = counter.getBoundingClientRect();
            const gearLeft = outer.right - right - (gear.offsetWidth || 28);
            const overlaps = box.height > 0 && box.width > 0 && box.right > gearLeft && box.left < outer.right - right;
            if (overlaps && box.bottom - outer.top + 2 > top) { top = box.bottom - outer.top + 2; }
        }
        const wantedTop = Math.round(top) + "px";
        const wantedRight = Math.round(right) + "px";
        if (gear.style.top !== wantedTop) { gear.style.top = wantedTop; }
        if (gear.style.right !== wantedRight) { gear.style.right = wantedRight; }
    }

    function gearCounter(id) {
        for (const target of GEAR_TARGETS) { if (target[0] === id) { return target[3]; } }
        return "";
    }

    function placeGears() {
        for (const target of GEAR_TARGETS) {
            const host = byId(target[0]);
            const gear = host ? host.querySelector("." + GEAR_CLASS) : null;
            if (gear) { placeGear(host, gear, target[3]); }
        }
    }

    /** Open the large editor on one of Forge's prompt boxes. */
    function editForgePrompt(id, label, tab) {
        const host = byId(id);
        const area = host ? host.querySelector("textarea") : null;
        if (!area) { return null; }
        const gear = host.querySelector("." + GEAR_CLASS);
        return prompt({
            name: "forge-prompt",
            title: label,
            subtitle: tab,
            text: area.value,
            placeholder: area.placeholder || "",
            done: function (text, info) {
                if (info && info.changed) { writeInput(area, text); }
                if (gear) { placeGear(host, gear, gearCounter(id)); }
            }
        });
    }

    /**
     * Put the gear on each of Forge's prompt boxes this page has, once. Asked
     * at every page load and after Forge's updates (a Reload UI builds the
     * boxes again), so a box that already has its gear is left alone and
     * nothing is written at all - an after-update callback that changed the
     * page each time would raise the next update itself.
     */
    function installGears() {
        let added = 0;
        for (const target of GEAR_TARGETS) {
            const host = byId(target[0]);
            const area = host ? host.querySelector("textarea") : null;
            if (!host || !area || host.querySelector("." + GEAR_CLASS)) { continue; }
            const label = "Edit the " + target[1].toLowerCase() + " in a large view";
            const gear = button(GEAR_CLASS, "", label);
            gear.setAttribute("aria-label", label);
            gear.appendChild(icon(GEAR_ICON, [[12, 12, 3]]));
            // The press is the gear's own: nothing under it, the box least of
            // all, is meant to take it (a tap into the box types, as always).
            gear.addEventListener("pointerdown", function (event) { event.stopPropagation(); });
            gear.addEventListener("click", function (event) {
                event.preventDefault();
                event.stopPropagation();
                editForgePrompt(target[0], target[1], target[2]);
            });
            host.classList.add(GEAR_HOST_CLASS);
            area.classList.add(GEAR_CLASS + "-room");
            host.appendChild(gear);
            placeGear(host, gear, target[3]);
            area.addEventListener("input", function () { placeGear(host, gear, target[3]); });
            host.addEventListener("pointerenter", function () { placeGear(host, gear, target[3]); });
            if (typeof ResizeObserver === "function") {
                try { new ResizeObserver(function () { placeGear(host, gear, target[3]); }).observe(area); } catch (e) { /* the events still place it */ }
            }
            added += 1;
        }
        if (!S.gearWatch) {
            S.gearWatch = function () { placeGears(); };
            window.addEventListener("resize", S.gearWatch);
        }
        if (added) { note("gear added to " + added + " of Forge's prompt boxes"); }
        return added;
    }

    function state() {
        return {
            version: VERSION,
            open: S.stack.map(function (sheet) { return sheet.name; }),
            depth: S.stack.length,
            asked: Object.assign({}, S.asked),
            gears: GEAR_TARGETS.filter(function (target) {
                const host = byId(target[0]);
                return !!(host && host.querySelector("." + GEAR_CLASS));
            }).map(function (target) { return target[0]; }),
            last: Object.assign({}, S.last),
            notes: S.notes.slice()
        };
    }

    // The gears, at the page's load and after Forge's updates (Forge's own
    // hooks; a page without them asks installGears itself).
    try {
        if (typeof onUiLoaded === "function") { onUiLoaded(installGears); }
        if (typeof onAfterUiUpdate === "function") { onAfterUiUpdate(installGears); }
    } catch (e) { /* the page asks for them itself */ }

    return {
        version: VERSION,
        open: open,
        prompt: prompt,
        systemPrompt: systemPrompt,
        installGears: installGears,
        placeGears: placeGears,
        top: topSheet,
        isOpen: function () { return S.stack.length > 0; },
        closeAll: function (reason) {
            while (S.stack.length) { topSheet().close(reason || "close"); }
        },
        state: state,
        debug: state
    };
})();
