// Model Chain -- the Literal Prompt boxes, browser side.
//
// Three jobs, all of them presentation, none of them able to change what a
// generation produces:
//
//   1. Move the Literal Prompt row up under the native Negative Prompt.
//   2. Put its two boxes into Forge's own "last prompt box you used" family,
//      so the Extra Networks browser inserts into whichever of the four you
//      touched last.
//   3. Hide the native Negative Prompt row while CFG is 1, and give its space
//      back.
//
// If every line of this file fails: the row is still on the page, lower down,
// with two working prompt boxes whose values still travel with every press;
// the LoRA browser still inserts into the positive prompt as it always did;
// and the Negative Prompt is still there at CFG 1, doing exactly what Forge
// already makes it do. Nothing here is load-bearing, which is the only basis on
// which moving somebody else's DOM around is acceptable at all.
//
// Why the row is moved rather than built in place
// -----------------------------------------------
// Forge gives an extension no way to build a component into the prompt area.
// `ui()` runs inside the script accordion, several hundred pixels below the box
// these two belong beside, and the design intent puts them under the Negative
// Prompt because that is where they read as prompt boxes rather than as
// settings.
//
// So they are built as ordinary Gradio Textboxes -- real components, real
// handlers, real values in the payload -- and this moves the finished row. The
// component is never rebuilt, re-parented into a different Gradio tree, or
// duplicated: one `insertBefore` of an element Gradio owns, which Gradio itself
// does not care about because it addresses that element by id.
//
// What this file must never become
// --------------------------------
// A second prompt path. It does not read the boxes' values for any purpose but
// counting them, does not assemble a prompt, does not know what a payload is,
// and never writes into a prompt box -- the one place text is inserted is the
// host's own Extra Networks handler, reached by telling it which textarea is
// active and letting it do the rest.

(function () {
    "use strict";

    const TAB = "txt2img";

    const IDS = {
        row: "mc-krea-creative-literal-row",
        positive: "mc-krea-creative-literal-positive",
        negative: "mc-krea-creative-literal-negative",
        prompt: "txt2img_prompt",
        negativePrompt: "txt2img_neg_prompt",
        cfg: "txt2img_cfg_scale",
    };

    const COLLAPSED = "mc-literal-cfg-collapsed";

    function root() {
        const app = typeof gradioApp === "function" ? gradioApp() : document;
        return app || document;
    }

    function byId(id) {
        try {
            return root().querySelector("#" + id) || document.getElementById(id);
        } catch (error) {
            return null;
        }
    }

    function fieldIn(id) {
        const holder = byId(id);
        return holder ? holder.querySelector("textarea, input") : null;
    }

    // Idempotent by a dataset flag, the way every listener in this extension is:
    // onAfterUiUpdate fires often, and a listener added twice is a handler that
    // runs twice.
    function once(element, event, handler, tag) {
        if (!element || !element.dataset) return;
        const flag = "mcLiteral" + (tag || "") + event;
        if (element.dataset[flag]) return;
        element.dataset[flag] = "1";
        element.addEventListener(event, handler);
    }

    // ------------------------------------------------------------------ //
    // 1. Where the row sits
    // ------------------------------------------------------------------ //

    function place() {
        const row = byId(IDS.row);
        const negative = byId(IDS.negativePrompt);
        if (!row || !negative || !negative.parentNode) return;
        // Already immediately after the Negative Prompt: nothing to do, and
        // checking is what makes this safe to run on every UI update.
        if (row.previousElementSibling === negative) return;
        try {
            negative.parentNode.insertBefore(row, negative.nextSibling);
        } catch (error) {
            // A theme that has restructured the toprow, or a Forge build that
            // renames the negative prompt. The row stays where Gradio put it,
            // which is lower down the page and completely functional.
            console.warn("Model Chain: the Literal Prompt row could not be moved "
                         + "under the Negative Prompt", error);
        }
    }

    // ------------------------------------------------------------------ //
    // 2. The last prompt box you used
    // ------------------------------------------------------------------ //
    //
    // §7. Forge remembers which prompt box was last focused and inserts an
    // Extra Networks card into that one; this adds two more boxes to the set it
    // remembers, and does nothing else. There is no LoRA formatter here, no
    // caret arithmetic and no insertion code -- all three already exist in the
    // host, are already correct, and are already what the positive and negative
    // prompt get.
    //
    // The family is Forge's `activePromptTextarea`, a plain global keyed by tab.
    // Reached through `window` and guarded, because a build that does not have
    // it is a build where this feature simply does not apply.

    function family() {
        try {
            if (typeof window === "undefined") return null;
            return window.activePromptTextarea || null;
        } catch (error) {
            return null;
        }
    }

    function remember(textarea) {
        const known = family();
        if (!known || !textarea) return;
        known[TAB] = textarea;
    }

    function registerPrompts() {
        if (!family()) return;
        [IDS.positive, IDS.negative].forEach(function (id) {
            const field = fieldIn(id);
            if (!field) return;
            // Focus only, which is exactly how the host registers its own two.
            // The Extra Networks browser is not a prompt box, so opening it
            // fires no focus event here and the remembered target survives --
            // which is the half of §7 that is about *not* doing something.
            once(field, "focus", function () { remember(field); }, "Family");
        });
    }

    // ------------------------------------------------------------------ //
    // 3. The Negative Prompt at CFG 1
    // ------------------------------------------------------------------ //
    //
    // §9, and the word that matters is *presentation*. Forge already decides
    // what CFG 1 means for the negative prompt; this only stops the row taking
    // up space while that is true. The value is never read, never written and
    // never cleared -- turn CFG back up and the text is still there, because
    // nothing here ever touched it.

    function cfg() {
        const holder = byId(IDS.cfg);
        if (!holder) return null;
        const fields = holder.querySelectorAll("input");
        for (const field of fields) {
            const value = parseFloat(field.value);
            if (Number.isFinite(value)) return value;
        }
        return null;
    }

    function applyCfg() {
        const negative = byId(IDS.negativePrompt);
        if (!negative || !negative.classList) return;
        const value = cfg();
        // A CFG this file cannot read leaves the row alone. Hiding a prompt box
        // because a number could not be parsed is the wrong way round.
        if (value === null) {
            negative.classList.remove(COLLAPSED);
            return;
        }
        // Forge's own minimum is 1, so "at or below 1" and "exactly 1" are the
        // same set -- written as the former so a build with a lower minimum
        // behaves sensibly rather than surprisingly.
        negative.classList.toggle(COLLAPSED, value <= 1);
    }

    function watchCfg() {
        const holder = byId(IDS.cfg);
        if (!holder) return;
        holder.querySelectorAll("input").forEach(function (field) {
            once(field, "input", applyCfg, "Cfg");
            once(field, "change", applyCfg, "Cfg");
        });
        applyCfg();
    }

    // ------------------------------------------------------------------ //
    // Wiring
    // ------------------------------------------------------------------ //

    function wire() {
        try {
            place();
        } catch (error) {
            console.error("Model Chain: the Literal Prompt row placement failed", error);
        }
        try {
            registerPrompts();
        } catch (error) {
            console.error("Model Chain: the Literal Prompt boxes could not join the "
                          + "prompt family", error);
        }
        try {
            watchCfg();
        } catch (error) {
            console.error("Model Chain: the CFG negative-prompt collapse failed", error);
        }
    }

    if (typeof onUiLoaded === "function") onUiLoaded(wire);

    // Gradio rebuilds parts of the tab on some updates, which can replace any
    // of the elements above. Every listener is guarded by a dataset flag and
    // the move checks where the row already is, so re-running costs a handful
    // of queries and can never double-bind or double-move.
    if (typeof onAfterUiUpdate === "function") onAfterUiUpdate(wire);

    // Exposed for the tests, which drive this file under node against a fake
    // page. Nothing in the extension reads it.
    window.modelChainLiterals = {
        wire: wire,
        place: place,
        registerPrompts: registerPrompts,
        applyCfg: applyCfg,
        cfg: cfg,
        ids: IDS,
        collapsedClass: COLLAPSED,
    };
})();
