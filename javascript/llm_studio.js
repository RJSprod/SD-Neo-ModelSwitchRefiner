// Model Chain -- LLM Studio polish.
//
// Section 5 draws the line this file stays on the right side of: "keep custom
// JavaScript focused on enhancement, not core business logic. Python should
// remain authoritative for model state, persistence, inference, and memory
// decisions." Nothing here talks to a model, stores anything, or decides
// anything. It does eight things a browser is better placed to do than a
// server round trip:
//
//   * Ctrl/Cmd+Enter submits the composer that has focus;
//   * the transcript follows a streaming reply while the reader is at the end
//     of it, and holds their place when they are not;
//   * Escape stops a run;
//   * every message in the transcript gets a row of actions, shown by a tap
//     on it: the version pager, Edit, Regenerate or Send again, Continue,
//     Branch, Copy, Send to VibeVoice, Play, Delete and Delete from here;
//   * the conversation workspace is measured against the window, so the layout
//     can fit the space it actually has rather than a space guessed in a
//     stylesheet;
//   * a status line that says something is in progress counts the seconds it
//     has been in progress for;
//   * the WebUI's footer is taken off the page, if the setting says so;
//   * Edit opens the message editor's dialog over the edit row, and hands the
//     row the words back.
//
// The counter is the one that has to justify itself, because the server could
// in principle have written the number. It could not have kept writing it: a
// status is repainted when the run yields, and the whole complaint that led to
// this was runs that yield nothing for a minute at a time while llama-server
// loads and reprocesses a prompt. A clock that stops during the wait it is
// there to measure is worse than no clock. So the sweeping bar is CSS, which
// runs without this file, and the number beside it is here, which is the only
// place a second can pass without a round trip.
//
// The measuring is the one that needs a word. The workspace is meant to fit the
// window -- the page does not scroll, the transcript does -- and how much room
// it has depends on where it starts on screen, which depends on the browser
// chrome, the host's header, the width the tabs wrapped to, the rows the top
// bar wrapped to and any theme in play. None of that is knowable from CSS. So
// the distance from the top of the workspace to the bottom of the viewport is
// measured here and published as one custom property, --mc-llm-available, and
// style.css does the layout. Every var() reading it carries a fallback that is
// a pure-CSS estimate of the same number, so the tab is laid out correctly
// without this file and exactly with it.
//
// The action row is the one that had to be here rather than in Python, and
// for a plain reason: a Gradio 4.40 Chatbot draws its own bubbles and there is
// nowhere in one to put a component. So the row is drawn here, inside the
// bubble, the way the Forge Assistant draws its own under a message -- "I
// like how in conversation mode flyout, I can access the action right in the
// message" -- and a button in it does exactly one thing when tapped: it says
// *which message* it is on, into a hidden box, and presses a hidden button
// for its action. Everything after that is Python's: loading the thread,
// branching it, streaming the reply, saving it. The browser nominates and
// decides nothing, which is the line this file stays on.
//
// Three buttons in the row never reach Python at all, because what they do is
// the browser's or the Voice Box's: Copy puts the message's words on the
// clipboard; Send to VibeVoice hands them to the Voice Box tab on this same
// page (`window.mcVoiceBox`), which renders them with whatever it is set up
// with; Play plays the render that came back. Which render belongs to which
// message is a key Python writes into the bubble, and the Voice Box files the
// render under it, so a reload finds it again.
//
// Everything is found by this extension's own element ids. Section 5 again:
// no selector below depends on a class Gradio generated, so a theme that
// replaces Gradio's internal DOM -- Lobe replaces a great deal of it -- changes
// how these panels look and cannot stop them working. If an id is missing, the
// feature it drives is skipped and the rest carry on; the tab is fully usable
// with this file absent, which is the test of whether it is really polish.
//
// The bubbles are the single exception, and they are why the paragraph above
// is worth keeping honest rather than quietly widening. A bubble is the host's
// element and carries no id of ours, so the shapes Gradio 4 and the themes
// that reskin it are known to use are tried in turn -- and when none of them
// matches, no row is drawn and nothing else changes. There is no sheet behind
// the row any more, so a theme this cannot read costs the per-message actions
// on that theme; the shapes below are the ones Gradio 4.40 and the Lobe theme
// on the user's own page draw.

(function () {
    "use strict";

    const PANELS = [
        {composer: "mc-llm-prompt-intent", submit: "mc-llm-prompt-generate", stop: "mc-llm-prompt-stop"},
        {composer: "mc-llm-chat-message", submit: "mc-llm-chat-send", stop: "mc-llm-chat-stop"},
        {composer: "mc-llm-minimax-prompt", submit: "mc-llm-minimax-enhance", stop: "mc-llm-minimax-stop"},
        {composer: "mc-llm-krea-prompt", submit: "mc-llm-krea-generate", stop: "mc-llm-krea-stop"},
    ];

    // Docked to the end of the transcript, and leaving it -- the same rule as
    // the Forge Assistant's conversation (`scrolledTranscript` there), as
    // asked for: "The moment i scroll away from the bottom of the view, i
    // should be undocked from the bottom ... If i scroll to the bottom again,
    // and reach the end of scroll, then i become docked again".
    //
    // There is no slack. This used to count the last 100 px as "the end", to
    // agree with Gradio's own Chatbot -- and both of them put a reader who had
    // scrolled up less than that back at the end with every streamed chunk.
    // Gradio's autoscroll is off now (`autoscroll=False` in
    // mc_llm_chat_panel.py) and this is the only thing that follows.
    // FOLLOW_END_PX is for the fractional pixels of a scaled page.
    const FOLLOW_END_PX = 2;
    // A finger moved this far is a scroll, not the tremble of a tap.
    const LEAVE_TOUCH_PX = 4;
    // A wheel reported in lines (Firefox does) or pages, as pixels.
    const WHEEL_LINE_PX = 16;

    function root() {
        const app = typeof gradioApp === "function" ? gradioApp() : document;
        return app || document;
    }

    function byId(id) {
        return root().querySelector("#" + id);
    }

    function clickable(id) {
        const holder = byId(id);
        if (!holder) return null;
        // Gradio wraps a Button in an element carrying the id; the button
        // itself is inside it, and sometimes *is* it.
        return holder.tagName === "BUTTON" ? holder : holder.querySelector("button");
    }

    function press(id) {
        const button = clickable(id);
        if (!button || button.disabled) return false;
        button.click();
        return true;
    }

    // -- Ctrl+Enter to submit, Escape to stop ------------------------------ //

    function wireComposer(panel) {
        const holder = byId(panel.composer);
        if (!holder || holder.dataset.mcLlmWired === "1") return;
        const field = holder.tagName === "TEXTAREA" ? holder : holder.querySelector("textarea");
        if (!field) return;
        holder.dataset.mcLlmWired = "1";

        field.addEventListener("keydown", function (event) {
            if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
                // Ctrl/Cmd+Enter submits from any composer, however tall it
                // is. Plain Enter is left to the host: Gradio submits a box
                // declared one line tall -- Conversation's -- and breaks the
                // line in the taller ones, which is what a composer somebody
                // is writing a paragraph in should do.
                if (press(panel.submit)) {
                    event.preventDefault();
                    event.stopPropagation();
                }
            } else if (event.key === "Escape") {
                if (press(panel.stop)) {
                    event.preventDefault();
                    event.stopPropagation();
                }
            }
        });
    }

    // -- follow a streaming reply ------------------------------------------ //

    function scroller(element) {
        // The scrolling element is whichever descendant actually overflows.
        // Asked for rather than assumed, because which one it is depends on
        // the theme.
        if (element.scrollHeight > element.clientHeight + 1) return element;
        const children = element.querySelectorAll("div");
        for (let index = 0; index < children.length; index += 1) {
            const child = children[index];
            if (child.scrollHeight > child.clientHeight + 1) return child;
        }
        return null;
    }

    function bottomGap(target) {
        return target.scrollHeight - target.scrollTop - target.clientHeight;
    }

    // Whether the reader is docked, decided by what they do rather than read
    // after the fact.
    //
    // The first version of this asked "are we near the bottom?" from inside
    // the MutationObserver, which by definition runs after the new content is
    // in the DOM -- so a reply that added more than the slack made the answer
    // "no" for a reader who had been at the bottom a millisecond earlier.
    // Whether to follow is a question about where you *were*. So it is
    // answered from the reader's own input -- a wheel or a finger the moment
    // it starts, before the scroll it causes (a chunk can land in between),
    // and the scroll events after it -- and the observer does what that says.
    // Moves the observer makes itself are written into `last`, so they are
    // never read as the reader's.
    function watch(target) {
        if (target.mcLlmAnchor) return target.mcLlmAnchor;
        // A scroller seen for the first time is a thread that has just been
        // opened, and a thread opens at its newest message.
        const state = {pinned: true, offset: target.scrollTop, last: target.scrollTop,
                       touchY: null};
        target.mcLlmAnchor = state;
        target.addEventListener("scroll", function () {
            scrolled(state, target);
        }, {passive: true});
        target.addEventListener("wheel", function (event) {
            let dy = Number(event && event.deltaY) || 0;
            if (event && event.deltaMode === 1) dy *= WHEEL_LINE_PX;
            else if (event && event.deltaMode === 2) dy *= target.clientHeight || 400;
            wheeled(state, target, dy);
        }, {passive: true});
        target.addEventListener("touchstart", function (event) {
            const touch = event && event.touches && event.touches[0];
            state.touchY = touch ? touch.clientY : null;
        }, {passive: true});
        target.addEventListener("touchmove", function (event) {
            const touch = event && event.touches && event.touches[0];
            if (!touch || typeof state.touchY !== "number") return;
            const moved = touch.clientY - state.touchY;
            // A finger down the glass scrolls the transcript up under it.
            if (moved >= LEAVE_TOUCH_PX) wheeled(state, target, -moved);
            else if (moved <= -LEAVE_TOUCH_PX) wheeled(state, target, -moved);
        }, {passive: true});
        target.addEventListener("keydown", function (event) {
            const key = event && event.key;
            if (key === "ArrowUp" || key === "PageUp" || key === "Home") {
                wheeled(state, target, -1);
            }
        });
        return state;
    }

    // A scroll the reader made: upward and away from the end undocks; downward
    // and reaching it docks.
    function scrolled(state, position) {
        const moved = position.scrollTop - state.last;
        state.last = position.scrollTop;
        state.offset = position.scrollTop;
        const gap = bottomGap(position);
        if (moved < 0 && gap > FOLLOW_END_PX) state.pinned = false;
        else if (moved > 0 && gap <= FOLLOW_END_PX) state.pinned = true;
    }

    // A gesture, read as it starts: any upward one undocks when there is
    // anything above to go to; a downward one at the very end (where no scroll
    // event can come) docks.
    function wheeled(state, position, dy) {
        if (dy < 0 && position.scrollTop > 0) state.pinned = false;
        else if (dy > 0 && bottomGap(position) <= FOLLOW_END_PX) state.pinned = true;
    }

    // What the scroll position should be once new content has landed, or null
    // to leave it exactly where it is. Split out from the DOM so the decision
    // can be tested without a browser.
    function anchorTo(state, position) {
        if (state.pinned) {
            // Clamped by the browser to the real maximum, which is what we
            // want: "the end", not a number.
            return position.scrollHeight;
        }
        // Not pinned, and the position has collapsed to the top. That is not
        // something a reader did -- it is what happens when a re-render empties
        // the list for an instant, because scrollTop is clamped to a
        // scrollHeight that was briefly zero. Put them back where they were
        // reading rather than at the top of an hour-old conversation.
        if (position.scrollTop === 0 && state.offset > 0) {
            return state.offset;
        }
        return null;
    }

    function wireTranscript() {
        const holder = byId("mc-llm-chat-transcript");
        if (!holder || holder.dataset.mcLlmFollow === "1") return;
        if (typeof MutationObserver !== "function") return;
        holder.dataset.mcLlmFollow = "1";

        const observer = new MutationObserver(function () {
            const target = scroller(holder);
            if (!target) return;
            const state = watch(target);
            const wanted = anchorTo(state, {
                scrollTop: target.scrollTop,
                scrollHeight: target.scrollHeight,
                clientHeight: target.clientHeight,
            });
            if (wanted !== null && wanted !== target.scrollTop) {
                target.scrollTop = wanted;
            }
            // Ours, not the reader's: the scroll event this fires changes
            // nothing.
            state.last = target.scrollTop;
        });
        observer.observe(holder, {childList: true, subtree: true, characterData: true});

        // Start watching now rather than at the first mutation, so the reader's
        // position is already being recorded when the first reply arrives.
        const target = scroller(holder);
        if (target) watch(target);
    }

    // -- a row of actions in every bubble ----------------------------------- //

    // The shapes a bubble is known to come in, most specific first. Tried in
    // turn; the first that matches anything wins, and if none does the rows
    // are not drawn. See the note at the top of this file: this is the one
    // place that reads the host's DOM.
    const REPLY_SELECTORS = [
        '[data-testid="bot"]',
        ".message-row.bot-row",
        ".bot-row",
        ".message.bot",
    ];
    const PROMPT_SELECTORS = [
        '[data-testid="user"]',
        ".message-row.user-row",
        ".user-row",
        ".message.user",
    ];

    // What the browser presses and writes. The box is one, the buttons one
    // per action, named by Python (mc_llm_chat_panel.NOMINATED_ACTIONS) and
    // compared across the two files by a test.
    const ACTION_AT = "mc-llm-chat-action-at";
    const ACTION_PREFIX = "mc-llm-chat-";
    const ACTION_SUFFIX = "-now";

    // What Python puts in every bubble for this row to read: the role, the
    // versions and the one showing, and the key the Voice Box files a render
    // under. A bubble without it is drawn from what can be seen -- its role,
    // from which list it was found in.
    const META_CLASS = "mc-llm-meta";

    const ROW_CLASS = "mc-llm-message-actions";
    const BUTTON_CLASS = "mc-llm-message-action";
    const REVEALED_CLASS = "mc-llm-revealed";

    // A waveform, because that is what the Voice Box makes and "that's vibey".
    // Drawn rather than typed: there is no glyph for it, and a picture inside a
    // button is one Lobe's icon swap never touches, since it has no text.
    const WAVEFORM = '<svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true" '
        + 'focusable="false"><g fill="currentColor"><rect x="2" y="9" width="2.5" height="6" '
        + 'rx="1"/><rect x="6.5" y="5" width="2.5" height="14" rx="1"/><rect x="11" y="2" '
        + 'width="2.5" height="20" rx="1"/><rect x="15.5" y="6" width="2.5" height="12" '
        + 'rx="1"/><rect x="20" y="10" width="2.5" height="4" rx="1"/></g></svg>';

    // The row, in the order it is drawn: the version pager where there are
    // versions to page; the actions that change the thread; the three that
    // take the words elsewhere; and last, behind a gap of their own, the two
    // that lose something. `role` limits a button to one kind of bubble;
    // `local` marks the ones this file answers itself. Every glyph is a
    // symbol and not an emoji, and none is the cross Lobe's icon swap
    // replaces (CLAUDE.md, "The host's theme rewrites the page"); U+FE0E
    // keeps the play triangle text rather than a coloured emoji.
    const ACTIONS = [
        {action: "edit", glyph: "✎", label: "Edit"},
        // "\u21bb", spelt as the Forge Assistant spells it: one glyph for one
        // action in both views of the conversation, and a test compares them.
        {action: "regenerate", glyph: "\u21bb", label: "Regenerate", role: "assistant"},
        {action: "resend", glyph: "↪", label: "Send again from here", role: "user"},
        {action: "continue", glyph: "⇢", label: "Continue", role: "assistant"},
        {action: "branch", glyph: "⎇", label: "Branch from here"},
        {action: "copy", glyph: "⧉", label: "Copy message", local: true},
        {action: "vibe", svg: WAVEFORM, label: "Send to VibeVoice", local: true},
        {action: "play", glyph: "▶︎", label: "Play VibeVoice", local: true},
        {action: "delete", glyph: "✕", label: "Delete message", destructive: true},
        {action: "delete_from", glyph: "⤓", label: "Delete from here", destructive: true},
    ];
    const PAGER = [
        {action: "back", glyph: "‹", label: "Show the previous version"},
        {action: "forward", glyph: "›", label: "Show the next version"},
        {action: "drop", glyph: "⊗", label: "Delete this version"},
    ];

    // How long a note in the row stays, and a copied tick on its button.
    const NOTE_MS = 4000;
    const COPIED_MS = 1200;
    // The Voice Box's poll runs at a second while a job of its is live, so
    // asking it more often than that reads the same answer twice.
    const WATCH_MS = 1000;
    // A job the Voice Box no longer lists (it keeps the last fifty) is given
    // this many looks before the message is told its render is lost.
    const WATCH_PATIENCE = 20;

    function bubblesIn(holder, selectors) {
        for (let index = 0; index < selectors.length; index += 1) {
            const found = holder.querySelectorAll(selectors[index]);
            if (found && found.length) {
                const bubbles = [];
                for (let at = 0; at < found.length; at += 1) {
                    const bubble = bubbleOf(found[at]);
                    if (bubbles.indexOf(bubble) < 0) bubbles.push(bubble);
                }
                return bubbles;
            }
        }
        return [];
    }

    // Gradio 4.40 draws a message as `div.message > button[data-testid]`: the
    // button is its select target and holds the words, the div is the bubble
    // the theme paints. The row belongs beside the button, inside the bubble,
    // not inside the button -- a button inside a button is a tap that presses
    // both -- so a matched button stands for its wrapper. Anything else (a
    // theme's own shape, the stand-ins the tests use) is the bubble itself.
    function bubbleOf(node) {
        if (node && node.tagName === "BUTTON" && node.parentNode
            && node.parentNode.classList && node.parentNode.classList.contains("message")) {
            return node.parentNode;
        }
        return node;
    }

    // The host's own select button inside a bubble, when the bubble has one:
    // a tap on it is a tap on the message, and Enter or Space on it already
    // becomes a click, which is why the bubble gets no tabindex of its own.
    function messageButtonOf(bubble) {
        if (!bubble) return null;
        if (bubble.tagName === "BUTTON") return bubble;
        if (typeof bubble.querySelector !== "function") return null;
        return bubble.querySelector("button[data-testid]");
    }

    function replyBubbles(holder) {
        return bubblesIn(holder, REPLY_SELECTORS);
    }

    function promptBubbles(holder) {
        return bubblesIn(holder, PROMPT_SELECTORS);
    }

    // Which message this is, counted down the transcript among its role,
    // asked at the moment of the tap rather than remembered from when the row
    // was drawn. A thread that has had a message deleted out of the middle of
    // it has renumbered every bubble below, and an ordinal captured in a
    // closure would name the wrong one -- which for a delete means deleting a
    // message the reader did not point at.
    function ordinalOf(holder, bubble, role) {
        const current = role === "user" ? promptBubbles(holder) : replyBubbles(holder);
        for (let index = 0; index < current.length; index += 1) {
            if (current[index] === bubble) return index;
        }
        return -1;
    }

    // What Python wrote into the bubble, or what can be seen of it.
    function metaOf(bubble, role) {
        const found = bubble.querySelector ? bubble.querySelector("." + META_CLASS) : null;
        const read = function (name, fallback) {
            const value = found && found.getAttribute ? found.getAttribute(name) : null;
            return value === null || value === undefined || value === "" ? fallback : value;
        };
        const versions = Math.max(1, parseInt(read("data-mc-versions", "1"), 10) || 1);
        const active = Math.max(0, Math.min(parseInt(read("data-mc-active", "0"), 10) || 0,
                                            versions - 1));
        return {role: read("data-mc-role", role), versions: versions, active: active,
                key: read("data-mc-key", "")};
    }

    // Hand a nomination to Python and let go. Nothing is decided here: which
    // message that is, whether the action applies to it, and whether doing it
    // branches the thread are all answered on the other side of the button.
    function nominate(holder, bubble, role, action) {
        const ordinal = ordinalOf(holder, bubble, role);
        if (ordinal < 0) return false;
        const box = byId(ACTION_AT);
        if (!box) return false;
        const field = box.tagName === "TEXTAREA" || box.tagName === "INPUT"
            ? box : box.querySelector("textarea, input");
        if (!field) return false;
        field.value = role + ":" + ordinal;
        // Gradio learns a value from the event, not from the property: a box
        // written to without this is a box the server still reads as empty.
        field.dispatchEvent(new Event("input", {bubbles: true}));
        // Next tick, so the value is in the host's store before the press that
        // sends it.
        window.setTimeout(function () {
            press(ACTION_PREFIX + action.replace(/_/g, "-") + ACTION_SUFFIX);
        }, 0);
        return true;
    }

    function actionButton(spec) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = BUTTON_CLASS + (spec.destructive ? " mc-llm-message-action-destructive" : "");
        if (spec.svg) button.innerHTML = spec.svg;
        else button.textContent = spec.glyph;
        button.title = spec.label;
        button.setAttribute("aria-label", spec.label);
        button.setAttribute("data-action", spec.action);
        return button;
    }

    // The words of a message, for the clipboard and the Voice Box: the
    // bubble's text without the row, the marker and anything else this file
    // put in it. A copy is taken so nothing on screen moves.
    function messageText(bubble) {
        const copy = typeof bubble.cloneNode === "function" ? bubble.cloneNode(true) : null;
        if (copy && copy.querySelectorAll) {
            const strays = copy.querySelectorAll("." + ROW_CLASS + ", ." + META_CLASS);
            for (let index = 0; index < strays.length; index += 1) {
                const stray = strays[index];
                if (stray.parentNode && stray.parentNode.removeChild) {
                    stray.parentNode.removeChild(stray);
                }
            }
            const text = typeof copy.innerText === "string" ? copy.innerText : copy.textContent;
            return String(text || "").trim();
        }
        return String(bubble.dataset && bubble.dataset.mcLlmText || bubble.textContent || "").trim();
    }

    // -- the row itself ------------------------------------------------------ //

    // Which message's row is open, by key; re-applied after every redraw, so a
    // reply arriving token by token does not close the row somebody opened.
    let revealed = "";

    function keyOf(bubble, role, holder) {
        const meta = metaOf(bubble, role);
        if (meta.key) return meta.key;
        return role + ":" + ordinalOf(holder, bubble, role);
    }

    function rowFor(holder, bubble, role) {
        const meta = metaOf(bubble, role);
        const key = keyOf(bubble, role, holder);
        const bar = document.createElement("div");
        bar.className = ROW_CLASS;
        bar.hidden = true;
        bar.setAttribute("data-key", key);
        if (meta.role === "assistant" && meta.versions > 1) {
            const cluster = document.createElement("span");
            cluster.className = "mc-llm-message-actions-versions";
            const back = actionButton(PAGER[0]);
            back.disabled = meta.active <= 0;
            const pager = document.createElement("span");
            pager.className = "mc-llm-message-actions-pager";
            pager.textContent = (meta.active + 1) + "/" + meta.versions;
            const forward = actionButton(PAGER[1]);
            forward.disabled = meta.active >= meta.versions - 1;
            const drop = actionButton(PAGER[2]);
            [back, forward, drop].forEach(function (button) {
                button.addEventListener("click", function (event) {
                    event.preventDefault();
                    event.stopPropagation();
                    nominate(holder, bubble, meta.role, button.getAttribute("data-action"));
                });
            });
            cluster.appendChild(back);
            cluster.appendChild(pager);
            cluster.appendChild(forward);
            cluster.appendChild(drop);
            bar.appendChild(cluster);
        }
        ACTIONS.forEach(function (spec) {
            if (spec.role && spec.role !== meta.role) return;
            const button = actionButton(spec);
            button.addEventListener("click", function (event) {
                // Stopped here, and deliberately: the bubble under this row
                // is what a tap toggles the row with, and a click that reached
                // it would put the row away under the finger.
                event.preventDefault();
                event.stopPropagation();
                if (spec.local) {
                    local(spec.action, holder, bubble, bar, key, button);
                    return;
                }
                // Pressed is chosen: the row goes away, and the transcript
                // that comes back is drawn afresh.
                reveal(holder, "");
                nominate(holder, bubble, meta.role, spec.action);
            });
            bar.appendChild(button);
        });
        const note = document.createElement("span");
        note.className = "mc-llm-message-actions-note";
        note.hidden = true;
        bar.appendChild(note);
        return bar;
    }

    function rowOf(bubble) {
        if (!bubble.querySelector) return null;
        const rows = bubble.querySelectorAll("." + ROW_CLASS);
        for (let index = 0; index < rows.length; index += 1) {
            if (rows[index].parentNode === bubble) return rows[index];
        }
        return null;
    }

    function noteIn(bar, text) {
        const note = bar.querySelector ? bar.querySelector(".mc-llm-message-actions-note") : null;
        if (!note) return;
        note.textContent = text || "";
        note.hidden = !text;
        if (note.mcLlmTimer) window.clearTimeout(note.mcLlmTimer);
        if (text) {
            note.mcLlmTimer = window.setTimeout(function () {
                note.textContent = "";
                note.hidden = true;
            }, NOTE_MS);
        }
    }

    // One bubble: its row drawn once, and drawn again when the marker changed
    // under it -- a version paged, a reply still arriving -- because the row
    // reads the marker when it is built. Idempotent, because this runs after
    // every update the host makes.
    function wireBubble(holder, bubble, role) {
        // A node without an element's methods -- a stand-in, or something a
        // theme put where a bubble goes -- gets no row and costs nothing else.
        if (!bubble.dataset || typeof bubble.setAttribute !== "function"
            || typeof bubble.appendChild !== "function"
            || typeof bubble.querySelectorAll !== "function") return;
        const key = keyOf(bubble, role, holder);
        const meta = metaOf(bubble, role);
        const stamp = key + "|" + meta.versions + "|" + meta.active;
        let bar = rowOf(bubble);
        if (bar && bubble.dataset.mcLlmActions === stamp) {
            applyVibe(bubble, bar, key);
            return;
        }
        if (bar && bar.parentNode && bar.parentNode.removeChild) bar.parentNode.removeChild(bar);
        bubble.dataset.mcLlmActions = stamp;
        bubble.setAttribute("data-mc-role", meta.role);
        // Focusable from the keyboard -- unless the host's own message button
        // is there to be focused, in which case a second tab stop on the same
        // message would be one too many.
        if (!bubble.getAttribute("tabindex") && !messageButtonOf(bubble)) {
            bubble.setAttribute("tabindex", "0");
        }
        bar = rowFor(holder, bubble, role);
        bubble.appendChild(bar);
        applyVibe(bubble, bar, key);
    }

    function wireBubbles() {
        const holder = byId("mc-llm-chat-transcript");
        if (!holder) return;
        const replies = replyBubbles(holder);
        for (let index = 0; index < replies.length; index += 1) {
            wireBubble(holder, replies[index], "assistant");
        }
        const prompts = promptBubbles(holder);
        for (let index = 0; index < prompts.length; index += 1) {
            wireBubble(holder, prompts[index], "user");
        }
        wireTaps(holder);
        reveal(holder, revealed);
        if (replies.length || prompts.length) recoverVibe(holder);
    }

    // -- tap to reveal ------------------------------------------------------- //

    function bubbleAt(holder, target) {
        if (!target || typeof target.closest !== "function") return null;
        // A selection inside the bubble is reading, not a tap.
        const selection = typeof window.getSelection === "function" ? window.getSelection() : null;
        if (selection && !selection.isCollapsed) return null;
        const selectors = REPLY_SELECTORS.concat(PROMPT_SELECTORS);
        let bubble = null;
        for (let index = 0; index < selectors.length && !bubble; index += 1) {
            const found = bubbleOf(target.closest(selectors[index]));
            if (found && holder.contains && holder.contains(found) && found.dataset
                && found.dataset.mcLlmActions) {
                bubble = found;
            }
        }
        if (!bubble) return null;
        // A press on a control inside the bubble is that control's -- a link
        // in the words, a button of the row -- except the host's own message
        // button, which *is* the message: Gradio 4.40 wraps every message's
        // words in one, so a tap anywhere on the words lands inside it.
        const control = target.closest("button, a, input, textarea, select, ." + ROW_CLASS);
        if (control && control !== bubble && control !== messageButtonOf(bubble)) return null;
        return bubble;
    }

    function roleOf(bubble) {
        return bubble.getAttribute && bubble.getAttribute("data-mc-role") === "user"
            ? "user" : "assistant";
    }

    // Show one message's row and nobody else's. Kept as a key rather than a
    // node and applied again after every redraw: a reply arriving token by
    // token redraws the transcript once a frame, and a row that closed
    // whenever a word arrived could not be used while a reply was arriving.
    function reveal(holder, key) {
        revealed = key || "";
        const all = Array.prototype.slice.call(replyBubbles(holder))
            .concat(Array.prototype.slice.call(promptBubbles(holder)));
        let found = null;
        all.forEach(function (bubble) {
            const bar = rowOf(bubble);
            if (!bar) return;
            const open = !!revealed && bar.getAttribute("data-key") === revealed;
            if (open) found = bubble;
            if (bar.hidden === open) bar.hidden = !open;
            if (bubble.classList) bubble.classList.toggle(REVEALED_CLASS, open);
            if (bubble.getAttribute("aria-expanded") !== String(open)) {
                bubble.setAttribute("aria-expanded", String(open));
            }
        });
        if (!found) revealed = "";
    }

    function toggle(holder, bubble) {
        const key = keyOf(bubble, roleOf(bubble), holder);
        reveal(holder, revealed === key ? "" : key);
    }

    function wireTaps(holder) {
        if (holder.dataset.mcLlmTaps === "1") return;
        holder.dataset.mcLlmTaps = "1";
        // Delegated, because bubbles are rebuilt whenever the thread moves.
        holder.addEventListener("click", function (event) {
            const bubble = bubbleAt(holder, event.target);
            if (bubble) toggle(holder, bubble);
        });
        // Enter or Space on a focused message does what a tap does. A message
        // that is a button already turns those keys into a click, which the
        // handler above takes; toggling here as well would open and close it.
        holder.addEventListener("keydown", function (event) {
            if (!event || (event.key !== "Enter" && event.key !== " ")) return;
            const bubble = event.target;
            if (!bubble || !bubble.dataset || !bubble.dataset.mcLlmActions) return;
            if (bubble.tagName === "BUTTON") return;
            event.preventDefault();
            toggle(holder, bubble);
        });
        // A press anywhere but on the open message puts its row away, in the
        // capture phase so it runs before the press lands: a tap on another
        // message closes this one here and opens that one above.
        if (!window.mcLlmDismissWired) {
            window.mcLlmDismissWired = true;
            window.addEventListener("pointerdown", function (event) {
                if (!revealed) return;
                const open = openBubble(holder);
                const target = event && event.target;
                if (open && target && typeof open.contains === "function" && open.contains(target)) {
                    return;
                }
                reveal(holder, "");
            }, true);
            // Escape puts the open row away from anywhere on the page, the
            // way it closes everything else of this tab's.
            window.addEventListener("keydown", function (event) {
                if (!revealed || !event || event.key !== "Escape") return;
                reveal(holder, "");
            }, true);
        }
    }

    function openBubble(holder) {
        const all = Array.prototype.slice.call(replyBubbles(holder))
            .concat(Array.prototype.slice.call(promptBubbles(holder)));
        for (let index = 0; index < all.length; index += 1) {
            const bar = rowOf(all[index]);
            if (bar && bar.getAttribute("data-key") === revealed) return all[index];
        }
        return null;
    }

    // -- Copy, Send to VibeVoice, Play --------------------------------------- //

    function local(action, holder, bubble, bar, key, button) {
        if (action === "copy") copyMessage(bubble, bar, button);
        else if (action === "vibe") sendToVibe(holder, bubble, bar, key);
        else if (action === "play") playVibe(holder, bubble, bar, key, button);
    }

    function copyMessage(bubble, bar, button) {
        const text = messageText(bubble);
        const done = function () {
            const was = button.textContent;
            button.textContent = "✓";
            window.setTimeout(function () { button.textContent = was; }, COPIED_MS);
        };
        const clipboard = typeof navigator !== "undefined" && navigator.clipboard;
        if (clipboard && typeof clipboard.writeText === "function") {
            clipboard.writeText(text).then(done, function () {
                if (copyByCommand(text)) done();
                else noteIn(bar, "The browser would not copy. Select the text and copy it.");
            });
            return;
        }
        if (copyByCommand(text)) done();
        else noteIn(bar, "The browser would not copy. Select the text and copy it.");
    }

    // The old way, for a page that is not a secure context: a box off screen,
    // selected and copied with the command the browser still honours.
    function copyByCommand(text) {
        try {
            const box = document.createElement("textarea");
            box.value = text;
            box.setAttribute("readonly", "");
            box.style.position = "fixed";
            box.style.left = "-9999px";
            document.body.appendChild(box);
            box.select();
            const copied = typeof document.execCommand === "function" && document.execCommand("copy");
            document.body.removeChild(box);
            return !!copied;
        } catch (error) {
            return false;
        }
    }

    // The Voice Box tab on this page, when it is there and has grown the bridge.
    function voiceBox() {
        const box = window.mcVoiceBox;
        if (!box || typeof box.renderText !== "function" || typeof box.canRender !== "function") {
            return null;
        }
        return box;
    }

    // What each message has asked of the Voice Box, by key: the job in
    // flight, the output that came back, and whether it has been played. One
    // record per message, replaced by the next press -- "it should be 1 to 1.
    // Only play the audio from the last vibe voice button press" -- so a
    // render that comes back for a press that has been superseded is left to
    // the Voice Box's own list and never offered here.
    const records = {};
    let watcher = 0;
    let recovered = "";
    let audio = null;

    function recordOf(key) {
        return records[key] || null;
    }

    function applyVibe(bubble, bar, key) {
        const record = recordOf(key);
        const state = !record ? "" : (record.pending ? "rendering"
            : (record.output ? (record.played ? "played" : "ready") : ""));
        if ((bubble.getAttribute("data-mc-vibe") || "") !== state) {
            if (state) bubble.setAttribute("data-mc-vibe", state);
            else bubble.removeAttribute("data-mc-vibe");
        }
        const buttons = bar.querySelectorAll ? bar.querySelectorAll("button") : [];
        for (let index = 0; index < buttons.length; index += 1) {
            const button = buttons[index];
            const action = button.getAttribute("data-action");
            if (action === "play") {
                const playable = !!(record && record.output && !record.pending);
                if (button.disabled !== !playable) button.disabled = !playable;
                const playing = !!(audio && audio.mcLlmKey === key && !audio.paused);
                const glyph = playing ? "‖" : "▶︎";
                if (button.textContent !== glyph) button.textContent = glyph;
                button.title = playing ? "Pause VibeVoice" : "Play VibeVoice";
                button.setAttribute("aria-label", button.title);
            } else if (action === "vibe") {
                const box = voiceBox();
                const why = box ? "" : "Voice Box is not on this page.";
                if (button.disabled !== !!why) button.disabled = !!why;
                const label = why ? "Send to VibeVoice — " + why
                    : (record && record.pending ? "Rendering with VibeVoice…"
                       : "Send to VibeVoice");
                if (button.title !== label) {
                    button.title = label;
                    button.setAttribute("aria-label", label);
                }
            }
        }
    }

    function sendToVibe(holder, bubble, bar, key) {
        const box = voiceBox();
        if (!box) {
            noteIn(bar, "Voice Box is not on this page.");
            return;
        }
        const text = messageText(bubble);
        const why = box.canRender(text);
        if (why) {
            noteIn(bar, why);
            return;
        }
        const record = {job: "", output: "", pending: true, played: false, looks: 0};
        records[key] = record;
        applyVibe(bubble, bar, key);
        noteIn(bar, "Sent to VibeVoice…");
        const opening = text.replace(/\s+/g, " ").slice(0, 40);
        let asked;
        try {
            asked = box.renderText(text, {
                name: "LLM Studio · " + opening,
                origin: {kind: "llm", key: key, label: "LLM Studio"},
            });
        } catch (error) {
            asked = Promise.reject(error);
        }
        Promise.resolve(asked).then(function (job) {
            // Superseded by a later press while the request was out: that
            // press's record is the one that counts.
            if (records[key] !== record) return;
            record.job = job && job.id ? String(job.id) : "";
            if (!record.job) {
                record.pending = false;
                noteIn(bar, "The Voice Box took the message but named no job.");
            }
            applyVibe(bubble, bar, key);
            watchRenders(holder);
        }, function (error) {
            if (records[key] !== record) return;
            record.pending = false;
            applyVibe(bubble, bar, key);
            noteIn(bar, (error && error.message) || "The Voice Box refused the message.");
        });
    }

    // The jobs in flight, looked at once a second while there is one. The
    // Voice Box's own poll keeps its job list current; this reads it.
    function watchRenders(holder) {
        if (watcher) return;
        watcher = window.setInterval(function () { lookAtRenders(holder); }, WATCH_MS);
        lookAtRenders(holder);
    }

    function lookAtRenders(holder) {
        const box = voiceBox();
        let pending = 0;
        Object.keys(records).forEach(function (key) {
            const record = records[key];
            if (!record.pending || !record.job) return;
            const job = box && typeof box.jobById === "function" ? box.jobById(record.job) : null;
            if (!job) {
                record.looks += 1;
                if (record.looks > WATCH_PATIENCE) {
                    record.pending = false;
                    settleRender(holder, key, "The Voice Box no longer lists that render.");
                } else {
                    pending += 1;
                }
                return;
            }
            record.looks = 0;
            if (job.live) {
                pending += 1;
                return;
            }
            record.pending = false;
            if (job.phase === "done" && job.output_id) {
                record.output = String(job.output_id);
                record.played = false;
                settleRender(holder, key, "");
            } else {
                settleRender(holder, key, job.warning || ("The render was " + (job.phase || "not finished") + "."));
            }
        });
        if (!pending && watcher) {
            window.clearInterval(watcher);
            watcher = 0;
        }
    }

    // A record has changed: the bubble it belongs to, if it is still on the
    // page, shows it.
    function settleRender(holder, key, note) {
        const bubble = bubbleFor(holder, key);
        if (!bubble) return;
        const bar = rowOf(bubble);
        if (!bar) return;
        applyVibe(bubble, bar, key);
        if (note) noteIn(bar, note);
    }

    function bubbleFor(holder, key) {
        const all = Array.prototype.slice.call(replyBubbles(holder))
            .concat(Array.prototype.slice.call(promptBubbles(holder)));
        for (let index = 0; index < all.length; index += 1) {
            const bar = rowOf(all[index]);
            if (bar && bar.getAttribute("data-key") === key) return all[index];
        }
        return null;
    }

    // The renders already made for this thread, read once per thread from the
    // Voice Box's outputs by their origin keys: a reload, or a thread opened
    // again, finds Play where it was. Newest first, so the latest render of a
    // message is the one offered; a record this page made is never replaced.
    function recoverVibe(holder) {
        const box = voiceBox();
        if (!box || typeof box.outputsFor !== "function") return;
        const first = replyBubbles(holder)[0] || promptBubbles(holder)[0];
        if (!first) return;
        const key = keyOf(first, roleOf(first), holder);
        const thread = key.indexOf(":") > 0 ? key.slice(0, key.lastIndexOf(":") + 1) : "";
        if (!thread || thread === recovered) return;
        recovered = thread;
        let asked;
        try {
            asked = box.outputsFor(thread);
        } catch (error) {
            asked = Promise.reject(error);
        }
        Promise.resolve(asked).then(function (outputs) {
            (outputs || []).forEach(function (output) {
                const origin = output && output.render && output.render.origin;
                const found = origin && typeof origin.key === "string" ? origin.key : "";
                if (!found || records[found]) return;
                records[found] = {job: "", output: String(output.id || ""), pending: false,
                                  played: true, looks: 0};
            });
            Object.keys(records).forEach(function (known) { settleRender(holder, known, ""); });
        }, function () {
            // The next thread asks again; this one is simply not recovered.
            recovered = "";
        });
    }

    // One player for the whole transcript, made when first needed. Playing
    // claims the page's audio focus, so Voice Chat and the Voice Box go
    // quiet, and anybody else claiming it pauses this.
    const FOCUS_EVENT = "mc:audio-focus";
    const FOCUS_OWNER = "llm-studio";

    function player(holder) {
        if (audio) return audio;
        audio = document.createElement("audio");
        audio.preload = "auto";
        audio.mcLlmKey = "";
        const refresh = function () {
            if (audio.mcLlmKey) settleRender(holder, audio.mcLlmKey, "");
        };
        ["play", "pause", "ended"].forEach(function (name) {
            audio.addEventListener(name, refresh);
        });
        if (typeof document.addEventListener === "function") {
            document.addEventListener(FOCUS_EVENT, function (event) {
                const detail = event && event.detail;
                if (!detail || detail.owner === FOCUS_OWNER) return;
                if (!audio.paused && typeof audio.pause === "function") audio.pause();
            });
        }
        return audio;
    }

    function claimFocus() {
        if (typeof CustomEvent !== "function" || typeof document.dispatchEvent !== "function") {
            return;
        }
        try {
            document.dispatchEvent(new CustomEvent(FOCUS_EVENT,
                                                   {detail: {owner: FOCUS_OWNER, kind: "playback"}}));
        } catch (error) { /* a page without the event is a page as before */ }
    }

    function playVibe(holder, bubble, bar, key, button) {
        const record = recordOf(key);
        if (!record || !record.output || record.pending) {
            noteIn(bar, record && record.pending ? "VibeVoice is still rendering this message."
                   : "Send this message to VibeVoice first.");
            return;
        }
        const box = voiceBox();
        if (!box || typeof box.outputAudioUrl !== "function") {
            noteIn(bar, "Voice Box is not on this page.");
            return;
        }
        const sound = player(holder);
        if (sound.mcLlmKey === key && !sound.paused) {
            sound.pause();
            return;
        }
        record.played = true;
        applyVibe(bubble, bar, key);
        let asked;
        try {
            asked = box.outputAudioUrl(record.output);
        } catch (error) {
            asked = Promise.reject(error);
        }
        Promise.resolve(asked).then(function (url) {
            if (records[key] !== record) return;
            claimFocus();
            if (sound.mcLlmKey !== key || sound.src !== url) {
                sound.src = url;
                sound.mcLlmKey = key;
            }
            const started = sound.play();
            return Promise.resolve(started).catch(function (error) {
                noteIn(bar, "Playback was refused by the browser" + (error && error.message
                    ? ": " + error.message : "."));
            });
        }, function (error) {
            noteIn(bar, (error && error.message) || "The render could not be fetched.");
        });
    }

    // -- a section that opens stays where it can be read -------------------- //

    // Conversation's screens are fixed-height scrolling sheets: the menu, the
    // threads, the character, the persona, in the room the workspace has. Open
    // a disclosure near the bottom of one -- the character editor, the advanced
    // sampling settings -- and what you opened is below the fold, which is a
    // thing browsers do not fix for you: the click landed on the heading, and
    // the heading was already visible.
    //
    // So the section is brought back after it has opened. Not by the browser's
    // own scrollIntoView: that scrolls every scrollable ancestor including the
    // page, and the page is not meant to move. This scrolls the sheet, by the
    // smallest amount that helps, and nothing else.

    // How long to wait for the section to have opened. A frame is not enough
    // -- Gradio re-renders on its own schedule -- and anything long enough to
    // notice would feel like the panel jumping on its own.
    const OPEN_SETTLE_MS = 80;

    // What the sheet's scroll position should become once a section has
    // opened, or null to leave it alone. Split out from the DOM so the rule
    // can be read and tested as arithmetic.
    function sectionScroll(section, view) {
        const top = section.top;
        const bottom = top + section.height;
        const seen = view.scrollTop + view.clientHeight;
        if (top >= view.scrollTop && bottom <= seen) return null;   // all of it is there
        // Taller than the sheet: show its beginning, because that is where
        // the control you just pressed is.
        if (section.height >= view.clientHeight) return top;
        // Otherwise the smallest move that brings the end of it into view.
        if (bottom > seen) return bottom - view.clientHeight;
        return top;
    }

    function keepInView(sheet, target) {
        let section = target;
        while (section && section.parentElement !== sheet) section = section.parentElement;
        if (!section) return;
        const wanted = sectionScroll(
            {top: section.offsetTop - sheet.offsetTop, height: section.offsetHeight},
            {scrollTop: sheet.scrollTop, clientHeight: sheet.clientHeight});
        if (wanted !== null) sheet.scrollTop = wanted;
    }

    // The scrolling surfaces this applies to, by this extension's own ids.
    const SHEETS = [
        "mc-llm-chat-nav",
        "mc-llm-chat-threads",
        "mc-llm-chat-character",
        "mc-llm-chat-persona",
        "mc-llm-model-sheet",
        "mc-llm-mode-sheet",
    ];

    function wireSheet(id) {
        const sheet = byId(id);
        if (!sheet || sheet.dataset.mcLlmInView === "1") return;
        sheet.dataset.mcLlmInView = "1";
        sheet.addEventListener("click", function (event) {
            const target = event.target;
            window.setTimeout(function () {
                try {
                    keepInView(sheet, target);
                } catch (error) {
                    console.error("Model Chain: could not keep the sheet in view", error);
                }
            }, OPEN_SETTLE_MS);
        });
    }

    function wireSheets() {
        SHEETS.forEach(wireSheet);
    }

    // -- how long the request in flight has been in flight ------------------ //

    // The status lines that can be busy. Named rather than searched for, for
    // the same reason everything else here is: this file may not depend on
    // Gradio's own DOM, and an id this extension chose is the only thing it
    // can rely on being there.
    const STATUSES = ["mc-llm-prompt-status", "mc-llm-chat-status", "mc-llm-minimax-status",
                      "mc-llm-krea-status"];

    // Below this the number says nothing anybody needs: every request is
    // "starting" for a moment, and a readout that flickers 0s-1s-gone on a
    // reply that arrived immediately is noise where the point was reassurance.
    const ELAPSED_QUIET_SECONDS = 2;

    function elapsedLabel(seconds) {
        const whole = Math.max(Math.floor(seconds), 0);
        if (whole < ELAPSED_QUIET_SECONDS) return "";
        if (whole < 60) return whole + "s";
        return Math.floor(whole / 60) + "m " + (whole % 60) + "s";
    }

    // One status line, at one moment. Returns the label it wrote, which is
    // what makes the rule testable without a browser: a busy line that has
    // just appeared starts the clock, a busy line that was already there keeps
    // the clock it started with, and a line that is not busy stops it.
    //
    // The start time is kept on the holder rather than on the notice, because
    // the notice is replaced wholesale every time the run says something new
    // -- "Starting…", "Replying…" are three separate elements -- and a clock
    // stored on it would restart at each of them. The holder is the component,
    // and it survives the run.
    function tickStatus(holder, now) {
        const busy = holder.querySelector(".mc-llm-busy");
        if (!busy) {
            delete holder.dataset.mcLlmSince;
            return "";
        }
        if (!holder.dataset.mcLlmSince) holder.dataset.mcLlmSince = String(now);

        const label = elapsedLabel((now - Number(holder.dataset.mcLlmSince)) / 1000);
        let readout = busy.querySelector(".mc-llm-busy-elapsed");
        if (!readout) {
            readout = document.createElement("span");
            readout.className = "mc-llm-busy-elapsed";
            busy.appendChild(readout);
        }
        if (readout.textContent !== label) readout.textContent = label;
        return label;
    }

    function tick() {
        const now = Date.now();
        STATUSES.forEach(function (id) {
            const holder = byId(id);
            if (holder) tickStatus(holder, now);
        });
    }

    function watchActivity() {
        if (window.mcLlmElapsedWired) return;
        window.mcLlmElapsedWired = true;
        // A second, because the number is in seconds. Three querySelectors a
        // second on an idle tab is not a cost worth optimising away, and an
        // observer would fire far more often for the same answer.
        window.setInterval(tick, 1000);
    }

    // -- the paperclip opens the file picker -------------------------------- //

    // Conversation's picture chip is a Gradio Image, and a Gradio Image is an
    // upload area with a file input inside it. Tapping the area opens the
    // browser's own picker; the paperclip beside it forwards a press to that
    // same input, so there is one way in and it is the ordinary one.
    //
    // The alternative was what used to be there: a full-width drop target above
    // the composer, opened by the paperclip and taking a panel's worth of room
    // to say "no picture yet". Reported as exactly that.
    //
    // Python still handles the press as well -- it makes the chip visible and
    // says whether the model running can be shown a picture at all -- so a
    // browser where this script did not run has a target to click rather than
    // nothing.
    const PICKERS = [
        {button: "mc-llm-chat-attach", picker: "mc-llm-chat-image"},
        {button: "mc-llm-chat-edit-attach", picker: "mc-llm-chat-edit-image"},
    ];

    function wirePicker(pair) {
        const button = clickable(pair.button);
        if (!button || button.dataset.mcLlmPicker) return;
        button.dataset.mcLlmPicker = "1";
        button.addEventListener("click", function () {
            // After the press has been handed to Python, which is what makes
            // the chip visible: a file input inside a display:none ancestor
            // opens nothing in some browsers, and the timeout is the cheapest
            // way to be after the update rather than racing it.
            window.setTimeout(function () {
                const holder = byId(pair.picker);
                if (!holder) return;
                const input = holder.querySelector('input[type="file"]');
                if (input) input.click();
            }, 60);
        });
    }

    // -- the footer, which is not ours and is in the way -------------------- //

    // The workspace above is built to fit the window: the page does not scroll,
    // the transcript does. The footer defeats that from outside anything this
    // extension lays out -- it sits below the fold and takes real space, so the
    // page scrolls by exactly the height of a row of links and no measurement
    // here can prevent it. Reported as that, and asked for as "can we just make
    // the footer go away".
    //
    // Nothing is removed and no style is written on the element: an attribute
    // goes on the root and style.css does the hiding, so the rule is one a
    // theme or a user stylesheet can override, and turning the setting off puts
    // the footer straight back on the next update rather than at the next
    // reload.
    const FOOTER_ATTRIBUTE = "data-mc-footer";

    function setting(name, fallback) {
        // The host publishes every registered option on a global. Read
        // defensively: it does not exist before the settings have loaded, and
        // an exception here would take the rest of this file's wiring with it.
        try {
            if (typeof opts === "undefined" || opts === null) return fallback;
            const value = opts[name];
            return value === undefined || value === null ? fallback : value;
        } catch (error) {
            return fallback;
        }
    }

    function hideFooter() {
        const html = document.documentElement;
        if (!html) return;
        const wanted = setting("model_chain_hide_footer", true) ? "hidden" : "";
        if (!wanted) {
            if (html.getAttribute(FOOTER_ATTRIBUTE)) html.removeAttribute(FOOTER_ATTRIBUTE);
            return;
        }
        // Written only when it changes: this runs after every update the host
        // makes, and setting an attribute invalidates style whether or not the
        // value moved.
        if (html.getAttribute(FOOTER_ATTRIBUTE) !== wanted) {
            html.setAttribute(FOOTER_ATTRIBUTE, wanted);
        }
    }

    // -- fit the workspace to the window ------------------------------------ //

    // Left under the workspace so a status line or a wrapped row of buttons
    // has somewhere to grow into before anything is pushed off the bottom.
    const BOTTOM_MARGIN_PX = 16;

    // Below this there is no layout worth doing, and style.css hands the page
    // back its scroll bar instead. Matching the max-height media query there.
    const MIN_AVAILABLE_PX = 260;

    // What is measured, and what the height is published on. The *workspace*
    // and not the whole tab: the mode selector, the model chooser and the
    // status line sit above it, and measuring from the top of the tab gave the
    // workspace their height as well -- which is exactly how far below the fold
    // the composer ended up.
    const FITTED = ["mc-llm-chat"];

    function documentTop(element) {
        // The distance from the top of the *document*, not of the viewport.
        //
        // This is the whole correctness argument for this function, so it is
        // worth stating. getBoundingClientRect().top alone falls as the page
        // scrolls, so "innerHeight - top" grows as the page scrolls -- and
        // since this sets the height of an element on that page, a taller
        // element means more page to scroll, which means a larger measurement
        // next time. That is a feedback loop, and what it looks like from the
        // outside is a panel that grows a little every time you click, with
        // blank space under the messages.
        //
        // Adding the scroll offset back makes the measurement scroll-
        // invariant: it is where the element sits in the document, which does
        // not depend on the height being set here, because nothing above it
        // does either.
        const scrolled = window.scrollY || document.documentElement.scrollTop || 0;
        return element.getBoundingClientRect().top + scrolled;
    }

    function publish(element, height) {
        const wanted = Math.round(height) + "px";
        // Written only when it changes: this runs on every click, and setting
        // a custom property invalidates layout whether or not the value moved.
        if (element.style.getPropertyValue("--mc-llm-available") === wanted) return false;
        element.style.setProperty("--mc-llm-available", wanted);
        return true;
    }

    function fitOne(element) {
        // An element in a tab that is not open measures as nothing at all.
        // Publishing that would hand every tab a height of zero, so it is
        // skipped and the fallback in the CSS stands until this tab is looked
        // at. offsetParent is null for a display:none ancestor, which is how
        // Gradio hides the tab that is not showing.
        if (!element.offsetParent) return;

        const available = window.innerHeight - documentTop(element) - BOTTOM_MARGIN_PX;
        if (available < MIN_AVAILABLE_PX) {
            element.style.removeProperty("--mc-llm-available");
            return;
        }
        // Never taller than the window, whatever the arithmetic said. A
        // measurement that has somehow gone wrong should cost a workspace that
        // is a little short, never one that cannot be scrolled back out of.
        // And nothing after it. There was a pass here that read the page's own
        // scrollHeight and gave back whatever still hung below the fold, on the
        // theory that a strip left behind by a hidden footer could be found by
        // its effect rather than by a selector.
        //
        // It found the wrong thing. This page's scroll overflow is not all ours
        // -- the host's layout has its own -- so what the workspace gave back
        // was somebody else's, and the result was a band of empty space above
        // the strip it was trying to remove, with the page still scrolling.
        // Reported as "your fix added space instead of removed it".
        //
        // The strip was the theme's own footer, and a footer is a thing that
        // can be named. It is named in style.css and hidden there. A measurement
        // that cannot tell whose overflow it is measuring should not be acting
        // on it.
        publish(element, Math.min(available, window.innerHeight - BOTTOM_MARGIN_PX));
    }

    function fit() {
        FITTED.forEach(function (id) {
            const element = byId(id);
            if (element) fitOne(element);
        });
    }

    function watchWindow() {
        if (window.mcLlmFitWired) return;
        window.mcLlmFitWired = true;
        window.addEventListener("resize", fit);
        // A tab switch changes where the workspace starts without resizing
        // anything, and nothing fires for it, so the click that does it is
        // what is listened to. Deliberately not scroll: the measurement above
        // does not depend on the scroll position, and re-running it on every
        // scroll event would be work for a value that cannot have changed.
        document.addEventListener("click", function () {
            window.setTimeout(fit, 0);
        }, true);
    }

    // -- editing in a dialog ------------------------------------------------ //
    //
    // Edit opens the row under the transcript -- the box, the paperclip, Save
    // and Cancel -- and that row is still what the server knows about. What
    // was asked for is a dialog: "a simple pop up, current text in an input
    // field, cancel, and a done button", the same one the Forge Assistant's
    // Edit opens. So when the row appears, its words go into the dialog; Done
    // puts the dialog's words back into the row's box and presses its Save,
    // and Cancel presses its Cancel. Python decides everything after that,
    // exactly as before, and a page without the dialog's script still has
    // the row. The browser nominates and decides nothing.
    //
    // The row is watched rather than the Edit button, because the words to
    // edit come back from the server with the row: only once it is showing
    // is there a message in the box. `offsetParent` is how this file already
    // asks whether something is showing -- it is null for an element with
    // `display: none` on it or on any ancestor, whichever way Gradio wrote it.
    const EDIT_ROW = "mc-llm-chat-edit";
    const EDIT_BOX = "mc-llm-chat-editor";
    const EDIT_SAVE = "mc-llm-chat-edit-save";
    const EDIT_CANCEL = "mc-llm-chat-edit-cancel";

    function editorLoaded() {
        return !!(window.mcMessageEditor && typeof window.mcMessageEditor.open === "function");
    }

    function wireEditor() {
        const row = byId(EDIT_ROW);
        if (!row || row.dataset.mcLlmEditor === "1") return;
        if (!editorLoaded()) return;
        row.dataset.mcLlmEditor = "1";
        let was = !!row.offsetParent;
        const observer = new MutationObserver(function () {
            const now = !!row.offsetParent;
            if (now && !was) openEditor();
            was = now;
        });
        observer.observe(row, {attributes: true, attributeFilter: ["class", "style"]});
        if (was) openEditor();
    }

    function openEditor() {
        const holder = byId(EDIT_BOX);
        const field = holder && (holder.tagName === "TEXTAREA" ? holder
            : holder.querySelector("textarea"));
        if (!field) return;
        // A frame later: the row and the words in its box arrive in the same
        // update, and the observer may run on the first of them.
        const open = function () {
            window.mcMessageEditor.open({
                title: "Edit message",
                text: field.value,
                placeholder: "The message’s new words…",
                done: function (words) {
                    field.value = words;
                    // Gradio reads the box on `input`, not on assignment.
                    field.dispatchEvent(new Event("input", {bubbles: true}));
                    if (!press(EDIT_SAVE)) {
                        return {ok: false, message: "Save is not available right now."};
                    }
                    return {ok: true};
                },
                cancel: function () {
                    press(EDIT_CANCEL);
                },
            });
        };
        if (typeof window.requestAnimationFrame === "function") {
            window.requestAnimationFrame(open);
        } else {
            open();
        }
    }

    // Each concern on its own. Polish must never be able to break the tab it is
    // polishing -- and one piece of polish must never be able to break another,
    // which a single try around all of them does not give you: these are
    // independent features, and the first of them throwing took the six after
    // it down with it. They are wired in no particular order and none of them
    // needs any other to have run.
    function attempt(what, run) {
        try {
            run();
        } catch (error) {
            console.error("Model Chain: LLM Studio could not " + what, error);
        }
    }

    function wire() {
        attempt("hide the footer", hideFooter);
        attempt("wire the attachment pickers", function () { PICKERS.forEach(wirePicker); });
        attempt("wire the composers", function () { PANELS.forEach(wireComposer); });
        attempt("follow the transcript", wireTranscript);
        attempt("draw the action rows", wireBubbles);
        attempt("keep the sheets in view", wireSheets);
        attempt("edit in a dialog", wireEditor);
        attempt("count the seconds", function () { watchActivity(); tick(); });
        attempt("fit the workspace", function () { watchWindow(); fit(); });
    }

    // The tab's contents are rebuilt on some UI updates, so the wiring is
    // re-applied rather than installed once. Each wiring is idempotent through
    // the dataset flags above, so re-running it costs a query and nothing else.
    if (typeof onUiLoaded === "function") {
        onUiLoaded(wire);
    } else if (document.readyState !== "loading") {
        wire();
    } else {
        document.addEventListener("DOMContentLoaded", wire);
    }

    if (typeof onAfterUiUpdate === "function") {
        onAfterUiUpdate(wire);
    }
})();
