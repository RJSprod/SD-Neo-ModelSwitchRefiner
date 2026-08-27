// Model Chain -- LLM Studio polish.
//
// Section 5 draws the line this file stays on the right side of: "keep custom
// JavaScript focused on enhancement, not core business logic. Python should
// remain authoritative for model state, persistence, inference, and memory
// decisions." Nothing here talks to a model, stores anything, or decides
// anything. It does seven things a browser is better placed to do than a
// server round trip:
//
//   * Ctrl/Cmd+Enter submits the composer that has focus;
//   * the transcript follows a streaming reply while the reader is at the end
//     of it, and holds their place when they are not;
//   * Escape stops a run;
//   * every reply in the transcript gets a regenerate icon;
//   * the conversation workspace is measured against the window, so the layout
//     can fit the space it actually has rather than a space guessed in a
//     stylesheet;
//   * a status line that says something is in progress counts the seconds it
//     has been in progress for;
//   * the WebUI's footer is taken off the page, if the setting says so.
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
// The regenerate icon is the one that had to be here rather than in Python, and
// for a plain reason: a Gradio 4.40 Chatbot draws its own bubbles and there is
// nowhere in one to put a component. So the icon is drawn here and does exactly
// one thing when tapped -- it says *which reply* it is on, into a hidden box,
// and presses a hidden button. Everything after that is Python's: loading the
// thread, branching it, streaming the reply, saving it. The browser nominates
// and decides nothing, which is the line this file stays on.
//
// Everything is found by this extension's own element ids. Section 5 again:
// no selector below depends on a class Gradio generated, so a theme that
// replaces Gradio's internal DOM -- Lobe replaces a great deal of it -- changes
// how these panels look and cannot stop them working. If an id is missing, the
// feature it drives is skipped and the rest carry on; the tab is fully usable
// with this file absent, which is the test of whether it is really polish.
//
// The reply bubbles are the single exception, and they are why the paragraph
// above is worth keeping honest rather than quietly widening. A bubble is the
// host's element and carries no id of ours, so the shapes Gradio 4 and the
// themes that reskin it are known to use are tried in turn -- and when none of
// them matches, no icon is drawn and nothing else changes. Regenerate is still
// on the sheet a tap on the bubble opens, which is where it was before this
// existed. A theme this cannot read costs an icon, never an action.

(function () {
    "use strict";

    const PANELS = [
        {composer: "mc-llm-prompt-intent", submit: "mc-llm-prompt-generate", stop: "mc-llm-prompt-stop"},
        {composer: "mc-llm-chat-message", submit: "mc-llm-chat-send", stop: "mc-llm-chat-stop"},
        {composer: "mc-llm-minimax-prompt", submit: "mc-llm-minimax-enhance", stop: "mc-llm-minimax-stop"},
        {composer: "mc-llm-krea-prompt", submit: "mc-llm-krea-generate", stop: "mc-llm-krea-stop"},
    ];

    // How close to the bottom still counts as "following along". A reader who
    // has scrolled up by more than this is reading, and moving the viewport
    // under them is the rudest thing a chat window can do.
    //
    // 100 and not a number of our own: it is what Gradio's own Chatbot uses,
    // and two components disagreeing about whether you are at the bottom is
    // worse than either answer.
    const FOLLOW_SLACK_PX = 100;

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

    // Where the reader put the transcript, remembered *before* anything
    // arrives.
    //
    // This is the whole of why the first version of this did not work. It
    // asked "are we near the bottom?" from inside the MutationObserver, which
    // by definition runs after the new content is in the DOM -- so a reply
    // that added more than the slack made the answer "no" for a reader who had
    // been pinned to the bottom a millisecond earlier. Whether to follow is a
    // question about where you *were*, and there is no reading of the DOM
    // after the fact that answers it.
    //
    // So it is answered from scroll events instead, which only fire when the
    // position actually changes, and the observer does what the recorded
    // answer says.
    function watch(target) {
        if (target.mcLlmAnchor) return target.mcLlmAnchor;
        // A scroller seen for the first time is a thread that has just been
        // opened, and a thread opens at its newest message.
        const state = {pinned: true, offset: target.scrollTop};
        target.mcLlmAnchor = state;
        target.addEventListener("scroll", function () {
            state.offset = target.scrollTop;
            state.pinned = bottomGap(target) <= FOLLOW_SLACK_PX;
        }, {passive: true});
        return state;
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
        });
        observer.observe(holder, {childList: true, subtree: true, characterData: true});

        // Start watching now rather than at the first mutation, so the reader's
        // position is already being recorded when the first reply arrives.
        const target = scroller(holder);
        if (target) watch(target);
    }

    // -- a regenerate icon on every reply ----------------------------------- //

    // The shapes a reply bubble is known to come in, most specific first. Tried
    // in turn; the first that matches anything wins, and if none does the icons
    // are not drawn. See the note at the top of this file: this is the one
    // place that reads the host's DOM, and the cost of failing to read it is an
    // icon, never an action.
    const REPLY_SELECTORS = [
        '[data-testid="bot"]',
        ".message-row.bot-row",
        ".bot-row",
        ".message.bot",
    ];

    const AGAIN_LABEL = "Regenerate this reply";

    function replyBubbles(holder) {
        for (let index = 0; index < REPLY_SELECTORS.length; index += 1) {
            const found = holder.querySelectorAll(REPLY_SELECTORS[index]);
            if (found && found.length) return found;
        }
        return [];
    }

    // Which reply this is, counted down the transcript, asked at the moment of
    // the tap rather than remembered from when the icon was drawn. A thread
    // that has had a message deleted out of the middle of it has renumbered
    // every bubble below, and an ordinal captured in a closure would name the
    // wrong one -- which for this particular button means rewriting a reply the
    // reader did not point at.
    function ordinalOf(holder, bubble) {
        const current = replyBubbles(holder);
        for (let index = 0; index < current.length; index += 1) {
            if (current[index] === bubble) return index;
        }
        return -1;
    }

    // Hand the ordinal to Python and let go. Nothing is decided here: which
    // message that is, whether it can be regenerated, and whether doing so
    // branches the thread are all answered on the other side of this button.
    function askAgain(ordinal) {
        const holder = byId("mc-llm-chat-regenerate-at");
        if (!holder) return false;
        const field = holder.tagName === "TEXTAREA" || holder.tagName === "INPUT"
            ? holder : holder.querySelector("textarea, input");
        if (!field) return false;
        field.value = String(ordinal);
        // Gradio learns a value from the event, not from the property: a box
        // written to without this is a box the server still reads as empty.
        field.dispatchEvent(new Event("input", {bubbles: true}));
        // Next tick, so the value is in the host's store before the press that
        // sends it.
        window.setTimeout(function () {
            press("mc-llm-chat-regenerate-now");
        }, 0);
        return true;
    }

    function againButton(holder, bubble) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "mc-llm-again";
        button.textContent = "\u21bb";
        button.title = AGAIN_LABEL;
        button.setAttribute("aria-label", AGAIN_LABEL);
        button.addEventListener("click", function (event) {
            // Stopped here, and deliberately: the bubble under this icon is
            // wired to the Chatbot's own select event, and a click that reached
            // it would open the action sheet over the reply that is about to
            // start arriving.
            event.preventDefault();
            event.stopPropagation();
            const ordinal = ordinalOf(holder, bubble);
            if (ordinal >= 0) askAgain(ordinal);
        });
        return button;
    }

    function wireReplies() {
        const holder = byId("mc-llm-chat-transcript");
        if (!holder) return;
        const bubbles = replyBubbles(holder);
        for (let index = 0; index < bubbles.length; index += 1) {
            const bubble = bubbles[index];
            // Idempotent, because this runs after every update the host makes:
            // a bubble Gradio re-rendered is a new element without the flag and
            // gets its icon back, and one it left alone is skipped.
            if (!bubble.dataset || bubble.dataset.mcLlmAgain === "1") continue;
            bubble.dataset.mcLlmAgain = "1";
            bubble.appendChild(againButton(holder, bubble));
        }
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
        const height = Math.round(Math.min(available, window.innerHeight - BOTTOM_MARGIN_PX));
        const wanted = height + "px";
        // Written only when it changes: this runs on every click, and setting
        // a custom property invalidates layout whether or not the value moved.
        if (element.style.getPropertyValue("--mc-llm-available") === wanted) return;
        element.style.setProperty("--mc-llm-available", wanted);
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
        attempt("wire the composers", function () { PANELS.forEach(wireComposer); });
        attempt("follow the transcript", wireTranscript);
        attempt("draw the reply icons", wireReplies);
        attempt("keep the sheets in view", wireSheets);
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
