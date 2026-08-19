// Model Chain -- LLM Studio polish.
//
// Section 5 draws the line this file stays on the right side of: "keep custom
// JavaScript focused on enhancement, not core business logic. Python should
// remain authoritative for model state, persistence, inference, and memory
// decisions." Nothing here talks to a model, stores anything, or decides
// anything. It does three things a browser is better placed to do than a
// server round trip:
//
//   * Ctrl/Cmd+Enter submits the composer that has focus;
//   * the transcript follows a streaming reply, unless the reader has scrolled
//     up to read something, in which case it stays where they put it;
//   * Escape stops a run.
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

    function wire() {
        try {
            PANELS.forEach(wireComposer);
            wireTranscript();
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
