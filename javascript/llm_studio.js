// Model Chain -- LLM Studio polish.
//
// Section 5 draws the line this file stays on the right side of: "keep custom
// JavaScript focused on enhancement, not core business logic. Python should
// remain authoritative for model state, persistence, inference, and memory
// decisions." Nothing here talks to a model, stores anything, or decides
// anything. It does four things a browser is better placed to do than a
// server round trip:
//
//   * Ctrl/Cmd+Enter submits the composer that has focus;
//   * the transcript follows a streaming reply, unless the reader has scrolled
//     up to read something, in which case it stays where they put it;
//   * Escape stops a run;
//   * the conversation workspace is measured against the window, so the layout
//     can fit the space it actually has rather than a space guessed in a
//     stylesheet.
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
// Everything is found by this extension's own element ids. Section 5 again:
// no selector below depends on a class Gradio generated, so a theme that
// replaces Gradio's internal DOM -- Lobe replaces a great deal of it -- changes
// how these panels look and cannot stop them working. If an id is missing, the
// feature it drives is skipped and the rest carry on; the tab is fully usable
// with this file absent, which is the test of whether it is really polish.

(function () {
    "use strict";

    const PANELS = [
        {composer: "mc-llm-prompt-intent", submit: "mc-llm-prompt-generate", stop: "mc-llm-prompt-stop"},
        {composer: "mc-llm-chat-message", submit: "mc-llm-chat-send", stop: "mc-llm-chat-stop"},
        {composer: "mc-llm-minimax-prompt", submit: "mc-llm-minimax-enhance", stop: "mc-llm-minimax-stop"},
    ];

    // How close to the bottom still counts as "following along". A reader who
    // has scrolled up by more than this is reading, and moving the viewport
    // under them is the rudest thing a chat window can do.
    const FOLLOW_SLACK_PX = 120;

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
                // Plain Enter and Shift+Enter both keep their normal meaning:
                // a newline. A composer that sends on Enter loses a paragraph
                // the first time somebody writes one.
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

    function wireTranscript() {
        const holder = byId("mc-llm-chat-transcript");
        if (!holder || holder.dataset.mcLlmFollow === "1") return;
        if (typeof MutationObserver !== "function") return;
        holder.dataset.mcLlmFollow = "1";

        const observer = new MutationObserver(function () {
            const target = scroller(holder);
            if (!target) return;
            const distance = target.scrollHeight - target.scrollTop - target.clientHeight;
            if (distance <= FOLLOW_SLACK_PX) {
                target.scrollTop = target.scrollHeight;
            }
        });
        observer.observe(holder, {childList: true, subtree: true, characterData: true});
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

    function wire() {
        try {
            PANELS.forEach(wireComposer);
            wireTranscript();
            watchWindow();
            fit();
        } catch (error) {
            // Polish must never be able to break the tab it is polishing.
            console.error("Model Chain: LLM Studio polish failed", error);
        }
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
