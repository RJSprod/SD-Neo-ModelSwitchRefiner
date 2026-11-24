// Forge Assistant -- a character's system prompt, edited on a page of its own.
//
// Asked for: "I like the idea of a full page editor for system prompt, with
// option to restore default ... in LLM studio character menu, i just want a
// button that opens a full page editor like this where i can set the prompt
// for that character. Same as with flyout."
//
// One editor with two ways in: the flyout's ⋯ menu and LLM Studio's character
// screen. What it edits is the character's `system` field, the same one LLM
// Studio's override box writes: one prompt per character, two places to change
// it.
//
// Since 2026-11-24 it is drawn on the pop-over every large editor shares
// (javascript/mc_sheet.js): "Lets make both the system prompt editors and this
// wan2gp pop over be like a pop up that covers 90% of the screen ... these
// should look the same". So it is the writers' system prompt editor in shape --
// the box is 90% of the window with the page dimmed round it, a line under the
// box says whose prompt this is and where it came from, Restore default and
// Reload on the left, Close and Save override on the right -- without the
// writer and picture choices, which are about video prompts.
//
// Three things it will not do.
//
// *Freeze the default by accident.* It opens on the prompt the character is
// actually given, which for most characters is built from their Context and
// the persona. Saved untouched, that is not an override -- the server keeps
// none and says so -- because a frozen copy stops following either, and
// nothing on screen would ever say it had.
//
// *Lose an edit quietly.* Close, Escape, the dimmed edge and Reload ask first
// while the box holds something that has not been saved -- keep editing,
// discard or save, the question the writers' editor asks. Save and Restore are
// not offered until the prompt has been read: an empty box saved is "go back to
// the default", and a read that failed must not become a press that wipes an
// override.
//
// *Vanish under focus mode.* The sheet's layer is a dialog (role="dialog",
// aria-modal) on the body, on the dialog layer above focus mode and the panel:
// focus mode's rule spares whatever says it is a dialog, wherever it ends up.

(function () {
    "use strict";

    const NS = (window.forgeAssistant = window.forgeAssistant || {});

    // LLM Studio's character editor: its Name box, and its override box, which
    // holds this same field. Told about a save when it is open on the same
    // character, or its next Save would write the old prompt back.
    const STUDIO_NAME = "mc-llm-chat-name";
    const STUDIO_SYSTEM = "mc-llm-chat-system";

    const SHEET_NAME = "character-prompt";

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
        this.sheet = null;
        this.character = "";
        // What the box held when it was last read or saved, and where that
        // came from. `null` until a read has answered.
        this.loaded = null;
        this.busy = false;
        // A read or a save that answers after a newer one began is not drawn.
        this.turn = 0;
        this.controls = {};
    }

    /** A sheet closed from outside -- Mini Paint closes every sheet when its
     *  popup goes, say -- is closed here too: the sheet says so itself. */
    Editor.prototype.isOpen = function () {
        return !!(this.sheet && !this.sheet.closed);
    };

    /** Whether the box holds something that has not been saved. */
    Editor.prototype.dirty = function () {
        const box = this.controls.text;
        return !!(this.loaded && box && box.value !== this.loaded.text);
    };

    /**
     * Ask what to do with an unsaved edit, then do `then` unless the answer
     * was to keep editing (or a save asked for failed). Resolves with what
     * `then` answered, or false. Nothing to lose runs it at once.
     */
    Editor.prototype.settleEdit = function (then) {
        if (!this.dirty() || !this.isOpen()) return Promise.resolve(then());
        const who = this.character;
        return this.sheet.ask("Your change to " + who + "’s system prompt is not saved.",
                              {keep: "Keep editing", discard: "Discard", save: "Save"})
            .then((choice) => {
                if (choice === "keep") {
                    this.focusText(false);
                    return false;
                }
                if (choice === "discard") return then();
                return this.apply().then((ok) => (ok ? then() : false));
            });
    };

    Editor.prototype.open = function (character) {
        const who = String(character || "").trim();
        if (!this.controls.text) this.build();
        if (this.isOpen()) {
            // Already there: an edit in progress is not thrown away by a
            // second press on the way in.
            if (same(who, this.character)) return true;
            if (this.dirty()) {
                this.settleEdit(() => this.switchTo(who));
                return false;
            }
            return this.switchTo(who);
        }
        if (!this.show()) return false;
        return this.switchTo(who);
    };

    /** Onto a character, in the sheet that is open. */
    Editor.prototype.switchTo = function (who) {
        this.character = who;
        this.loaded = null;
        this.title();
        this.controls.text.value = "";
        this.state("", "");
        this.note("", "info");
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
        if (this.sheet) this.sheet.setTitle(who ? "System prompt — " + who : "System prompt", "Character");
        c.hint.textContent = who
            ? "What " + who + " is told before every conversation, in LLM Studio and in "
              + "the Forge Assistant alike."
            : "";
    };

    /** The sheet, opened, with the controls in it. */
    Editor.prototype.show = function () {
        const api = window.mcSheet;
        if (!api || typeof api.open !== "function") {
            this.note("The editor is not on this page. Reload it.", "error");
            return false;
        }
        const c = this.controls;
        const sheet = api.open({
            name: SHEET_NAME,
            title: "System prompt",
            subtitle: "Character",
            kind: "system",
            request: () => this.close(false),
            submit: () => this.apply(),
        });
        this.sheet = sheet;
        c.heading = sheet.title;
        c.dismiss = sheet.cross;
        [c.hint, c.text, c.state].forEach((node) => sheet.body.appendChild(node));
        c.note = sheet.message;
        sheet.start.appendChild(c.restore);
        sheet.start.appendChild(c.reload);
        sheet.end.appendChild(c.close);
        sheet.end.appendChild(c.apply);
        this.focusText(false);
        return true;
    };

    /** The cursor in the box, inside the press that opened it, so a phone
     *  raises its keyboard; at the start once a prompt is in it. */
    Editor.prototype.focusText = function (toStart) {
        const box = this.controls.text;
        try {
            box.focus({preventScroll: true});
            if (toStart && typeof box.setSelectionRange === "function") {
                box.setSelectionRange(0, 0);
                box.scrollTop = 0;
            }
        } catch (error) {
            // Not worth an exception.
        }
    };

    /** Let go of the sheet as it is put away: answers still on their way are
     *  not drawn; a save still on its way still happens -- the server has it
     *  -- and still reaches LLM Studio. A sheet put away by anything else
     *  reads as shut through `isOpen`, and the next open starts a read whose
     *  turn drops any older answer. */
    Editor.prototype.gone = function (sheet) {
        if (this.sheet !== sheet) return;
        this.turn += 1;
        this.loaded = null;
        this.busy = false;
        this.sheet = null;
    };

    /** Put the editor away. `force` skips the question about an unsaved edit.
     *  Answers whether it went now; with a question on screen, it is the
     *  answer that decides. */
    Editor.prototype.close = function (force) {
        if (!this.isOpen()) return true;
        if (!force && this.dirty()) {
            this.settleEdit(() => this.close(true));
            return false;
        }
        const sheet = this.sheet;
        this.gone(sheet);
        sheet.close("close");
        return true;
    };

    Editor.prototype.reload = function () {
        if (!this.character || !this.isOpen()) return Promise.resolve(false);
        return this.settleEdit(() => this.read());
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
            this.focusText(true);
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
     *  The writers' editor's two sentences, with the name in them. */
    Editor.prototype.describe = function () {
        const loaded = this.loaded;
        const who = this.character;
        if (!loaded) return this.state("", "");
        if (loaded.source === "override") {
            this.state("Override saved", " for " + who + ". Restore default forgets it.");
        } else {
            this.state("Default", " for " + who + ", built from " + who + "’s Context "
                       + "and your persona. Edit it and save an override to replace it.");
        }
    };

    Editor.prototype.state = function (lead, rest) {
        this.controls.lead.textContent = lead || "";
        this.controls.rest.textContent = rest || "";
    };

    /** The sheet's message line, or nothing while no sheet is open. `kind` is
     *  kept on it for the checks: info, warn or error. */
    Editor.prototype.note = function (text, kind) {
        const sheet = this.sheet;
        if (!sheet) return;
        sheet.say(text || "", kind === "error" || kind === "warn");
        if (sheet.message) sheet.message.dataset.kind = kind || "info";
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
        if (this.sheet) this.sheet.dialog.dataset.dirty = String(this.dirty());
        this.describe();
    };

    const UNSAVED = "Not saved yet.";

    /** Typing. Save override is what keeps it, and the line says so. */
    Editor.prototype.edited = function () {
        const dirty = this.dirty();
        if (this.sheet) this.sheet.dialog.dataset.dirty = String(dirty);
        this.controls.apply.classList.toggle("mc-sheet-unsaved", dirty);
        const note = this.controls.note;
        const kind = note && note.dataset ? note.dataset.kind : "";
        if (dirty && kind !== "error") this.note(UNSAVED, "info");
        else if (!dirty && note && note.textContent === UNSAVED) this.note("", "info");
    };

    /** The controls, once; each opening puts them in its sheet. */
    Editor.prototype.build = function () {
        const c = this.controls;
        const hint = make("div", "mc-sheet-availability");
        const text = make("textarea", "mc-sheet-text mc-sheet-mono");
        text.setAttribute("aria-label", "System prompt");
        text.spellcheck = true;
        text.addEventListener("input", () => this.edited());

        // The line under the box: the first words bold, the rest plain. Names
        // go in as text, never as markup -- a character is called what its
        // file says, and a file can say anything.
        const state = make("div", "mc-sheet-state");
        const lead = make("strong", "");
        const rest = make("span", "");
        state.appendChild(lead);
        state.appendChild(rest);

        const restore = button("mc-sheet-restore", "Restore default",
                               "Forget the override and go back to the built prompt");
        const reload = button("mc-sheet-reload", "Reload",
                              "Read the saved prompt again");
        const close = button("mc-sheet-close-text", "Close",
                             "Close (Escape)");
        const apply = button("mc-sheet-save", "Save override",
                             "Save this as the character’s own prompt (Ctrl+Enter)");
        apply.addEventListener("click", () => this.apply());
        restore.addEventListener("click", () => this.restore());
        reload.addEventListener("click", () => this.reload());
        close.addEventListener("click", () => this.close(false));

        Object.assign(c, {hint, text, state, lead, rest, apply, restore, reload, close,
                          heading: null, dismiss: null, note: null});
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
