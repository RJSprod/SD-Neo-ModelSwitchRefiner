// Forge Assistant -- a character's system prompt, edited on a page of its own.
//
// Asked for: "I like the idea of a full page editor for system prompt, with
// option to restore default ... in LLM studio character menu, i just want a
// button that opens a full page editor like this where i can set the prompt
// for that character. Same as with flyout."
//
// One editor with two ways in: the flyout's ⋯ menu and LLM Studio's character
// screen. It is Mini Paint's system prompt editor in shape -- the box is most
// of the window, a line under it says whose prompt this is and where it came
// from, and four buttons: Apply override, Restore default, Reload, Close --
// without the model and enhancer choices, which are about video prompts. What
// it edits is the character's `system` field, the same one LLM Studio's
// override box writes: one prompt per character, two places to change it.
//
// Three things it will not do.
//
// *Freeze the default by accident.* It opens on the prompt the character is
// actually given, which for most characters is built from their Context and
// the persona. Applied untouched, that is not an override -- the server keeps
// none and says so -- because a frozen copy stops following either, and
// nothing on screen would ever say it had.
//
// *Lose an edit quietly.* Close, Escape and Reload ask first while the box
// holds something that has not been applied. Apply and Restore are not offered
// until the prompt has been read: an empty box applied is "go back to the
// default", and a read that failed must not become a press that wipes an
// override.
//
// *Vanish under focus mode.* It is a native modal <dialog> -- the top layer,
// above focus mode, the panel and every z-index on the page -- appended to the
// body, and marked role="dialog" aria-modal="true" out loud as well: focus
// mode's rule spares whatever says it is a dialog, wherever it ends up.

(function () {
    "use strict";

    const NS = (window.forgeAssistant = window.forgeAssistant || {});

    // LLM Studio's character editor: its Name box, and its override box, which
    // holds this same field. Told about a save when it is open on the same
    // character, or its next Save would write the old prompt back.
    const STUDIO_NAME = "mc-llm-chat-name";
    const STUDIO_SYSTEM = "mc-llm-chat-system";

    const HEADING_ID = "forge-assistant-system-heading";

    function make(tag, className, text) {
        const node = document.createElement(tag);
        if (className) node.className = className;
        if (text !== undefined) node.textContent = text;
        return node;
    }

    function button(className, text, title) {
        const node = make("button", className, text);
        node.type = "button";
        if (title) node.title = title;
        return node;
    }

    function messageOf(error, fallback) {
        return (error && error.message) || fallback;
    }

    function field(id) {
        const holder = document.getElementById(id);
        if (!holder) return null;
        if (holder.tagName === "TEXTAREA" || holder.tagName === "INPUT") return holder;
        return typeof holder.querySelector === "function"
            ? holder.querySelector("textarea, input") : null;
    }

    function same(first, second) {
        return String(first || "").trim().toLowerCase()
            === String(second || "").trim().toLowerCase();
    }

    /** Tell LLM Studio's override box what was just saved, when its editor is
     *  on this character. An input event, the way Forge's `updateInput` does
     *  it, because Gradio reads the box from that and not from `value`. */
    function syncStudio(character, view) {
        const name = field(STUDIO_NAME);
        const box = field(STUDIO_SYSTEM);
        if (!name || !box || !same(name.value, character)) return false;
        const wanted = view.source === "override" ? view.text : "";
        if (box.value === wanted) return false;
        box.value = wanted;
        if (typeof updateInput === "function") {
            updateInput(box);
        } else if (typeof Event === "function") {
            box.dispatchEvent(new Event("input", {bubbles: true}));
        }
        return true;
    }

    function Editor() {
        this.dialog = null;
        this.character = "";
        // What the box held when it was last read or applied, and where that
        // came from. `null` until a read has answered.
        this.loaded = null;
        this.busy = false;
        // A read or a save that answers after a newer one began is not drawn.
        this.turn = 0;
        this.controls = {};
    }

    Editor.prototype.isOpen = function () {
        return !!(this.dialog && this.dialog.open);
    };

    /** Whether the box holds something that has not been applied. */
    Editor.prototype.dirty = function () {
        const box = this.controls.text;
        return !!(this.loaded && box && box.value !== this.loaded.text);
    };

    /** True when there is nothing to lose, or the person said to lose it. */
    Editor.prototype.discardable = function () {
        if (!this.dirty()) return true;
        if (typeof window.confirm !== "function") return true;
        return window.confirm("Discard your changes to " + this.character
                              + "’s system prompt?");
    };

    Editor.prototype.open = function (character) {
        const who = String(character || "").trim();
        if (!this.dialog) this.build();
        if (this.isOpen()) {
            // Already there: an edit in progress is not thrown away by a
            // second press on the way in.
            if (same(who, this.character)) return true;
            if (!this.discardable()) return false;
        }
        this.character = who;
        this.loaded = null;
        const c = this.controls;
        this.title();
        c.text.value = "";
        this.state("", "");
        this.note("", "info");
        this.show();
        if (!who) {
            this.note("Choose a character first.", "warn");
            this.draw();
            return false;
        }
        this.read();
        return true;
    };

    /** The heading and the sentence under it, for whoever is being edited. */
    Editor.prototype.title = function () {
        const who = this.character;
        const c = this.controls;
        c.heading.textContent = who ? "System prompt — " + who : "System prompt";
        c.hint.textContent = who
            ? "What " + who + " is told before every conversation, in LLM Studio and in "
              + "the Forge Assistant alike."
            : "";
    };

    Editor.prototype.show = function () {
        const dialog = this.dialog;
        if (this.isOpen()) return;
        this.fit();
        try {
            if (typeof dialog.showModal === "function") dialog.showModal();
            else dialog.setAttribute("open", "");
        } catch (error) {
            dialog.setAttribute("open", "");
        }
        this.follow(true);
    };

    /** Fill what is actually on the screen.
     *
     * `position: fixed` -- the top layer's too -- is placed in the layout
     * viewport, and on a phone that is not the screen whenever the page is
     * wider than it is: the browser widens the layout viewport to the page.
     * Filling that, the editor ran off the right-hand edge with its buttons
     * below the fold (Chromium at 412px, a page 508px wide: a layout viewport
     * of 508 by 1129). The visual viewport is what is on the glass, and it is
     * also what shrinks when a phone's keyboard opens, so the buttons stay
     * above the keyboard while the box is being typed into.
     */
    Editor.prototype.fit = function () {
        const view = window.visualViewport;
        const style = this.dialog && this.dialog.style;
        if (!style || !view || !(view.width > 0) || !(view.height > 0)) return;
        style.left = Math.round(view.offsetLeft || 0) + "px";
        style.top = Math.round(view.offsetTop || 0) + "px";
        style.width = Math.round(view.width) + "px";
        style.height = Math.round(view.height) + "px";
        style.right = "auto";
        style.bottom = "auto";
    };

    /** Refit while open: a keyboard, a rotation, a pinch. */
    Editor.prototype.follow = function (on) {
        const view = window.visualViewport;
        if (!view || typeof view.addEventListener !== "function") return;
        if (!this._refit) this._refit = () => this.fit();
        const verb = on ? "addEventListener" : "removeEventListener";
        view[verb]("resize", this._refit);
        view[verb]("scroll", this._refit);
    };

    /** Put the editor away. `force` skips the question about unapplied text. */
    Editor.prototype.close = function (force) {
        if (!this.isOpen()) return true;
        if (!force && !this.discardable()) return false;
        // Answers still on their way are not drawn; a save still on its way
        // still happens -- the server has it -- and still reaches LLM Studio.
        this.turn += 1;
        this.loaded = null;
        this.follow(false);
        if (typeof this.dialog.close === "function") this.dialog.close();
        else this.dialog.removeAttribute("open");
        return true;
    };

    Editor.prototype.reload = function () {
        if (!this.character || !this.discardable()) return Promise.resolve(false);
        return this.read();
    };

    Editor.prototype.read = function () {
        const who = this.character;
        const turn = ++this.turn;
        const store = typeof NS.store === "function" ? NS.store() : null;
        this.busy = true;
        this.note("Reading " + who + "’s system prompt…", "info");
        this.draw();
        if (!store) return this.settle(turn, false, "The Forge Assistant is not on this page.");
        return store.systemPrompt(who).then((view) => {
            if (turn !== this.turn) return false;
            this.take(view);
            this.note("", "info");
            return this.settle(turn, true);
        }, (error) => this.settle(turn, false,
                                  messageOf(error, "The system prompt could not be read.")));
    };

    Editor.prototype.apply = function () {
        if (this.busy || !this.loaded) return Promise.resolve(false);
        return this.write({text: this.controls.text.value}, "Saving…");
    };

    Editor.prototype.restore = function () {
        if (this.busy || !this.loaded) return Promise.resolve(false);
        return this.write({restore: true}, "Restoring the default…");
    };

    Editor.prototype.write = function (change, doing) {
        const who = this.character;
        const turn = ++this.turn;
        const store = NS.store();
        this.busy = true;
        this.note(doing, "info");
        this.draw();
        return store.saveSystemPrompt(who, change).then((view) => {
            // LLM Studio hears about it even if this page has moved on.
            syncStudio(view.character || who, view);
            if (turn !== this.turn) return false;
            this.take(view);
            this.note(view.message || "Saved.", "info");
            return this.settle(turn, true);
        }, (error) => this.settle(turn, false, messageOf(error, "That could not be saved.")));
    };

    Editor.prototype.settle = function (turn, ok, failure) {
        if (turn !== this.turn) return false;
        this.busy = false;
        if (failure) this.note(failure, "error");
        this.draw();
        return ok;
    };

    Editor.prototype.take = function (view) {
        // The name as the character's file spells it, which is how the rest
        // of the page spells it too.
        this.character = String(view.character || this.character);
        this.title();
        this.loaded = {text: String(view.text || ""),
                       source: view.source === "override" ? "override" : "default"};
        this.controls.text.value = this.loaded.text;
    };

    /** The line under the box: whose prompt this is, and where it came from.
     *  Mini Paint's two sentences, with the name in them. */
    Editor.prototype.describe = function () {
        const loaded = this.loaded;
        const who = this.character;
        if (!loaded) return this.state("", "");
        if (loaded.source === "override") {
            this.state("Override saved", " for " + who + ". Restore default forgets it.");
        } else {
            this.state("Default", " for " + who + ", built from " + who + "’s Context "
                       + "and your persona. Edit and Apply override to replace it.");
        }
    };

    Editor.prototype.state = function (lead, rest) {
        this.controls.lead.textContent = lead || "";
        this.controls.rest.textContent = rest || "";
    };

    Editor.prototype.note = function (text, kind) {
        const note = this.controls.note;
        note.textContent = text || "";
        note.dataset.kind = kind || "info";
    };

    /** Everything that follows from the state: what can be pressed, and the
     *  lines that say what is going on. */
    Editor.prototype.draw = function () {
        const c = this.controls;
        const ready = !!this.loaded && !this.busy;
        c.text.readOnly = !ready;
        c.apply.disabled = !ready;
        c.restore.disabled = !ready;
        c.reload.disabled = this.busy || !this.character;
        if (this.dialog) this.dialog.dataset.dirty = String(this.dirty());
        this.describe();
    };

    const UNAPPLIED = "Not applied yet.";

    /** Typing. The Apply button is what keeps it, and the line says so. */
    Editor.prototype.edited = function () {
        const dirty = this.dirty();
        if (this.dialog) this.dialog.dataset.dirty = String(dirty);
        const note = this.controls.note;
        if (dirty && note.dataset.kind !== "error") this.note(UNAPPLIED, "info");
        else if (!dirty && note.textContent === UNAPPLIED) this.note("", "info");
    };

    Editor.prototype.build = function () {
        const c = this.controls;
        const dialog = make("dialog", "forge-assistant-system");
        // Said out loud rather than left implicit in the tag: focus mode hides
        // what sits beside the focused workspace unless it carries one of
        // these.
        dialog.setAttribute("role", "dialog");
        dialog.setAttribute("aria-modal", "true");
        dialog.setAttribute("aria-labelledby", HEADING_ID);
        const sheet = make("div", "forge-assistant-system-sheet");
        dialog.appendChild(sheet);

        // The way out first, so a keyboard -- and a phone's, which would open
        // over the text -- does not land in the box the moment it opens.
        const top = make("div", "forge-assistant-system-top");
        const heading = make("h2", "forge-assistant-system-heading", "System prompt");
        heading.id = HEADING_ID;
        const dismiss = button("forge-assistant-icon-button forge-assistant-system-dismiss",
                               "×", "Close");
        dismiss.setAttribute("aria-label", "Close");
        top.appendChild(heading);
        top.appendChild(dismiss);
        sheet.appendChild(top);

        const hint = make("p", "forge-assistant-system-hint");
        sheet.appendChild(hint);

        const text = make("textarea", "forge-assistant-system-text");
        text.setAttribute("aria-labelledby", HEADING_ID);
        text.spellcheck = true;
        text.addEventListener("input", () => this.edited());
        sheet.appendChild(text);

        // Mini Paint's line: the first words bold, the rest plain. Names go
        // in as text, never as markup -- a character is called what its file
        // says, and a file can say anything.
        const state = make("p", "forge-assistant-system-state");
        const lead = make("strong", "forge-assistant-system-lead");
        const rest = make("span", "forge-assistant-system-rest");
        state.appendChild(lead);
        state.appendChild(rest);
        sheet.appendChild(state);
        const note = make("p", "forge-assistant-system-note");
        note.setAttribute("role", "status");
        sheet.appendChild(note);

        const foot = make("div", "forge-assistant-system-foot");
        const apply = button("forge-assistant-system-action forge-assistant-system-apply",
                             "Apply override", "Save this as the character’s own prompt");
        const restore = button("forge-assistant-system-action forge-assistant-system-restore",
                               "Restore default",
                               "Forget the override and go back to the built prompt");
        const reload = button("forge-assistant-system-action forge-assistant-system-reload",
                              "Reload", "Read the saved prompt again");
        const close = button("forge-assistant-system-action forge-assistant-system-close",
                             "Close");
        apply.addEventListener("click", () => this.apply());
        restore.addEventListener("click", () => this.restore());
        reload.addEventListener("click", () => this.reload());
        close.addEventListener("click", () => this.close(false));
        dismiss.addEventListener("click", () => this.close(false));
        [apply, restore, reload, close].forEach((node) => foot.appendChild(node));
        sheet.appendChild(foot);

        // Escape is the browser's `cancel`. Taken over so it can ask first.
        dialog.addEventListener("cancel", (event) => {
            event.preventDefault();
            this.close(false);
        });

        Object.assign(c, {heading, hint, text, state, lead, rest, note, apply, restore,
                          reload, close, dismiss});
        document.body.appendChild(dialog);
        this.dialog = dialog;
    };

    let editor = null;

    NS.systemEditor = {
        Editor,
        syncStudio,
        editor() {
            if (!editor) editor = new Editor();
            return editor;
        },
        open(character) {
            return NS.systemEditor.editor().open(character);
        },
        isOpen() {
            return !!(editor && editor.isOpen());
        },
    };
})();
