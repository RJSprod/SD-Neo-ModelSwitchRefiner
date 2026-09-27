// Model Chain -- the message editor.
//
// One small dialog for editing a message, shared by the two places a message
// can be edited from: LLM Studio's Conversation tab and the Forge Assistant's
// panel. Asked for in one sentence: "when i choose to edit, i want a simple
// pop up. It should have the current text in an input field, cancel, and a
// done button." So that is the whole of it -- a heading saying whose message,
// the words in a box, Cancel and Done -- and whoever opened it says what Done
// means.
//
// Why a dialog of our own and not the browser's
// ---------------------------------------------
// `window.prompt` is one line of the browser's chrome, with no room to read a
// long prompt and, on a phone, a modal over everything. Editing in the
// composer's own box, which the panel did next, displaced the draft, turned
// Send into Save and needed a strip to say which message the box held -- and
// LLM Studio's edit row did the same under its transcript. A native `<dialog>`
// opened with `showModal()` is in the top layer, contains focus, takes Escape,
// and needs neither surface to change shape.
//
// Why it is not full screen
// -------------------------
// "This new pop up should not be full screen, make it reasonable, and
// responsive to the browser available space so it also looks good on mobile."
// It is sized from the *visual* viewport -- the glass, not the layout viewport,
// which on a phone is wider than the screen and does not shrink for the
// keyboard -- no wider than MAX_WIDTH, MARGIN from every edge at most, and
// placed a third of the spare room from the top so the keyboard, which takes
// the bottom, leaves it alone. The box grows with the words up to what the
// glass leaves for it and scrolls past that.
//
// Keys: Enter is Done, Shift+Enter starts a new line, Escape is Cancel. A
// composition in progress (an IME) owns its own Enter.
//
// The contract with whoever opens it
// ----------------------------------
//   window.mcMessageEditor.open({title, text, placeholder, done, cancel})
//
// `done(text)` is called with the box's words and may answer at once or with
// a promise. An answer of `{ok: false, message}` keeps the dialog open with
// the message under the box -- the server refused the edit, say -- and any
// other answer closes it. `cancel()` is called when the edit is given up:
// Cancel, Escape, or a second `open` replacing this one. Nothing here knows
// what a message is; it holds words and hands them back.

(function () {
    "use strict";

    const HEADING_ID = "mc-message-editor-heading";
    // From the glass's edges, at most; and the dialog's width, at most.
    const MARGIN = 16;
    const MAX_WIDTH = 560;
    const MIN_WIDTH = 240;
    const MIN_HEIGHT = 160;
    // The box's floor: about four lines.
    const MIN_BOX = 96;

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

    /** The glass: what is actually on the screen. */
    function glass() {
        const view = window.visualViewport;
        if (view && view.width > 0 && view.height > 0) {
            return {left: view.offsetLeft || 0, top: view.offsetTop || 0,
                    width: view.width, height: view.height};
        }
        const root = document.documentElement || {};
        return {left: 0, top: 0,
                width: root.clientWidth || window.innerWidth || 360,
                height: root.clientHeight || window.innerHeight || 640};
    }

    function Editor() {
        this.dialog = null;
        this.controls = {};
        // The edit open now: whoever asked, and what Done and Cancel mean.
        this.request = null;
        this.shown = false;
        this.busy = false;
        this.room = 0;
    }

    Editor.prototype.isOpen = function () {
        return this.shown;
    };

    Editor.prototype.open = function (options) {
        options = options || {};
        if (!this.dialog) this.build();
        // A second request while one is open replaces it, and the first is
        // told it was given up -- its Done can no longer come.
        if (this.shown && this.request) this.giveUp();
        this.request = {done: options.done, cancel: options.cancel};
        const c = this.controls;
        c.heading.textContent = options.title || "Edit message";
        c.text.value = String(options.text === undefined || options.text === null
            ? "" : options.text);
        c.text.placeholder = options.placeholder || "";
        this.note("", "info");
        this.setBusy(false);
        this.show();
        this.fit();
        c.text.focus();
        try {
            const end = c.text.value.length;
            c.text.setSelectionRange(end, end);
        } catch (error) { /* a box that will not take a caret still takes the edit */ }
        return true;
    };

    Editor.prototype.show = function () {
        const dialog = this.dialog;
        if (this.shown) return;
        this.shown = true;
        try {
            if (typeof dialog.showModal === "function") dialog.showModal();
            else dialog.setAttribute("open", "");
        } catch (error) {
            dialog.setAttribute("open", "");
        }
        this.follow(true);
    };

    /** Put the dialog away. Says nothing to whoever opened it: `done` and
     *  `cancel` decide that first. */
    Editor.prototype.close = function () {
        if (!this.shown) return false;
        this.shown = false;
        this.follow(false);
        if (typeof this.dialog.close === "function") this.dialog.close();
        else this.dialog.removeAttribute("open");
        return true;
    };

    /** Done. The words go to whoever asked, and the dialog waits for the
     *  answer with Done greyed out: `{ok: false, message}` keeps it open and
     *  says why under the box; anything else closes it. */
    Editor.prototype.done = function () {
        if (!this.shown || this.busy) return Promise.resolve(false);
        const request = this.request;
        const text = this.controls.text.value;
        this.setBusy(true);
        this.note("", "info");
        let answer;
        try {
            answer = request && typeof request.done === "function"
                ? request.done(text) : {ok: true};
        } catch (error) {
            answer = {ok: false, message: (error && error.message)
                || "That edit could not be saved."};
        }
        return Promise.resolve(answer).then((outcome) => outcome, (error) => ({
            ok: false, message: (error && error.message) || "That edit could not be saved.",
        })).then((outcome) => {
            // Replaced or given up while the answer was on its way: whatever
            // it was, it is not this dialog's to show any more.
            if (this.request !== request) return false;
            this.setBusy(false);
            if (outcome && outcome.ok === false) {
                this.note(outcome.message || "That edit was refused.", "error");
                this.controls.text.focus();
                return false;
            }
            this.request = null;
            this.close();
            return true;
        });
    };

    /** Cancel, Escape: the edit is given up and the dialog goes. Allowed
     *  while a Done is still being answered -- a save that takes a minute is
     *  not a reason to be stuck in front of it -- and whoever asked is told,
     *  so a late answer lands on nothing. */
    Editor.prototype.cancel = function () {
        if (!this.shown) return false;
        this.giveUp();
        this.close();
        return true;
    };

    Editor.prototype.giveUp = function () {
        const request = this.request;
        this.request = null;
        this.setBusy(false);
        if (request && typeof request.cancel === "function") {
            try {
                request.cancel();
            } catch (error) {
                console.error("Model Chain: the edit's cancel failed", error);
            }
        }
    };

    Editor.prototype.setBusy = function (on) {
        this.busy = !!on;
        const c = this.controls;
        if (c.done) c.done.disabled = this.busy;
        if (this.dialog) this.dialog.dataset.busy = String(this.busy);
    };

    Editor.prototype.note = function (text, kind) {
        const note = this.controls.note;
        if (!note) return;
        note.textContent = text || "";
        note.dataset.kind = kind || "info";
    };

    /** Where it goes and how big it is, from the glass.
     *
     * `position: fixed` -- the top layer's too -- is placed in the layout
     * viewport, and on a phone that is not the screen whenever the page is
     * wider than it is. The visual viewport is what is on the glass, and it
     * also shrinks when a phone's keyboard opens; the dialog is centred
     * across it, no wider than MAX_WIDTH, and placed a third of the spare
     * room down so the keyboard, which takes the bottom, leaves it alone.
     */
    Editor.prototype.fit = function () {
        const dialog = this.dialog;
        const style = dialog && dialog.style;
        if (!style) return;
        const box = glass();
        const wide = Math.max(MIN_WIDTH, Math.min(MAX_WIDTH, box.width - 2 * MARGIN));
        this.room = Math.max(MIN_HEIGHT, box.height - 2 * MARGIN);
        style.left = Math.round(box.left + (box.width - wide) / 2) + "px";
        style.width = Math.round(wide) + "px";
        style.maxHeight = Math.round(this.room) + "px";
        style.right = "auto";
        style.bottom = "auto";
        style.margin = "0";
        this.grow();
        // `tall` is at most the room, so the spare is at least two margins
        // and a third of it never reaches past the bottom one.
        const tall = Math.min(dialog.offsetHeight || 0, this.room);
        const spare = Math.max(0, box.height - tall);
        style.top = Math.round(box.top + Math.max(MARGIN, spare / 3)) + "px";
    };

    /** The box takes the words' height, up to what the glass leaves for it
     *  once the heading, the lines and the buttons have had theirs. */
    Editor.prototype.grow = function () {
        const dialog = this.dialog;
        const text = this.controls.text;
        if (!dialog || !text) return;
        text.style.height = "auto";
        const chrome = Math.max(0, (dialog.offsetHeight || 0) - (text.offsetHeight || 0));
        const most = Math.max(MIN_BOX, (this.room || 0) - chrome);
        const wanted = Math.max(MIN_BOX, text.scrollHeight || 0);
        text.style.height = Math.round(Math.min(wanted, most)) + "px";
        text.style.overflowY = wanted > most ? "auto" : "hidden";
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

    Editor.prototype.keyed = function (event) {
        if (event.isComposing || event.keyCode === 229) return;   // IME
        if (event.key === "Enter" && !event.shiftKey) {
            // Enter, and Ctrl or Cmd with it, is Done; Shift+Enter is the
            // browser's own new line.
            event.preventDefault();
            this.done();
        }
    };

    Editor.prototype.build = function () {
        const c = this.controls;
        const dialog = make("dialog", "mc-message-editor");
        // Said out loud rather than left implicit in the tag: the Forge
        // Assistant's focus mode hides what sits beside the focused workspace
        // unless it carries one of these.
        dialog.setAttribute("role", "dialog");
        dialog.setAttribute("aria-modal", "true");
        dialog.setAttribute("aria-labelledby", HEADING_ID);
        const sheet = make("div", "mc-message-editor-sheet");
        dialog.appendChild(sheet);

        const heading = make("h2", "mc-message-editor-heading", "Edit message");
        heading.id = HEADING_ID;
        sheet.appendChild(heading);

        const text = make("textarea", "mc-message-editor-text");
        text.setAttribute("aria-labelledby", HEADING_ID);
        text.rows = 4;
        text.spellcheck = true;
        text.addEventListener("input", () => this.grow());
        text.addEventListener("keydown", (event) => this.keyed(event));
        sheet.appendChild(text);

        const hint = make("p", "mc-message-editor-hint",
                          "Enter is Done. Shift+Enter starts a new line.");
        sheet.appendChild(hint);
        const note = make("p", "mc-message-editor-note");
        note.setAttribute("role", "status");
        sheet.appendChild(note);

        const foot = make("div", "mc-message-editor-foot");
        const cancel = button("mc-message-editor-action mc-message-editor-cancel",
                              "Cancel", "Keep the message as it was");
        const done = button("mc-message-editor-action mc-message-editor-done",
                            "Done", "Replace the message with these words");
        cancel.addEventListener("click", () => this.cancel());
        done.addEventListener("click", () => this.done());
        foot.appendChild(cancel);
        foot.appendChild(done);
        sheet.appendChild(foot);

        // Escape is the browser's `cancel` on a modal dialog. Taken over so
        // whoever opened the edit hears it was given up; and read from the
        // key as well, for a dialog opened without `showModal`, where the
        // browser sends no `cancel`.
        dialog.addEventListener("cancel", (event) => {
            event.preventDefault();
            this.cancel();
        });
        dialog.addEventListener("keydown", (event) => {
            if (event.key !== "Escape") return;
            event.preventDefault();
            event.stopPropagation();
            this.cancel();
        });

        Object.assign(c, {heading, text, hint, note, cancel, done});
        document.body.appendChild(dialog);
        this.dialog = dialog;
    };

    let editor = null;

    window.mcMessageEditor = {
        Editor,
        editor() {
            if (!editor) editor = new Editor();
            return editor;
        },
        open(options) {
            return window.mcMessageEditor.editor().open(options);
        },
        cancel() {
            return !!(editor && editor.cancel());
        },
        isOpen() {
            return !!(editor && editor.isOpen());
        },
    };
})();
