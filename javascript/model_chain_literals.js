// Model Chain -- the Literal Prompt boxes, browser side.
//
// Four jobs, all of them presentation, none of them able to change what a
// generation produces:
//
//   1. Move the Literal Prompt row up under the native Negative Prompt.
//   2. Put its two boxes into Forge's own "last prompt box you used" family,
//      so the Extra Networks browser inserts into whichever of the four you
//      touched last -- and offer them to Tag Autocomplete, if it is installed.
//   3. Hide the native Negative Prompt row while CFG is 1, and give its space
//      back.
//   4. Ask the prompt column to be as tall as the prompts in it.
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
    //
    // Read as a bare identifier *and* off `window`, because those are two
    // different places and this host uses the one that is not `window`.
    // Forge Neo's javascript/extraNetworks.js declares it
    //
    //     let activePromptTextarea = {};
    //
    // at the top level of a classic script, and a top-level `let` goes into the
    // global *lexical* environment: shared with every other classic script on
    // the page -- this file included -- and absent from `window`. Reading only
    // `window.activePromptTextarea` therefore found nothing on the very build
    // this extension targets, declined to register, and left every Extra
    // Networks card going to the native positive prompt no matter which box was
    // last used. Older `var` builds put it on `window`; both are read here, and
    // a build with neither is a build where this feature does not apply.

    function family() {
        try {
            // Not `window.` -- see above. `typeof` on an undeclared name is the
            // one way to ask without throwing.
            if (typeof activePromptTextarea !== "undefined" && activePromptTextarea) {
                return activePromptTextarea;
            }
        } catch (error) {
            // A build where the name exists but is not readable from here.
        }
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
        const known = family();
        [IDS.positive, IDS.negative].forEach(function (id) {
            const field = fieldIn(id);
            if (!field) return;
            if (known) {
                // Focus only, which is exactly how the host registers its own
                // two. The Extra Networks browser is not a prompt box, so
                // opening it fires no focus event here and the remembered
                // target survives -- which is the half of §7 that is about
                // *not* doing something.
                once(field, "focus", function () { remember(field); }, "Family");
            }
            joinAutocomplete(field);
        });
    }

    // ------------------------------------------------------------------ //
    // 2b. Tag Autocomplete, if it is installed
    // ------------------------------------------------------------------ //
    //
    // §8 asked for tag completion in these two boxes and got it by carrying the
    // host's `prompt` class, which is what that extension's own selector list
    // matches (`.prompt > label > textarea`, and Gradio builds these two with
    // exactly the label/textarea shape Forge's prompt boxes have). That list is
    // walked once, during its setup, so whether these two are in it depends on
    // an ordering neither extension controls -- and when they are not, the
    // boxes look like prompt boxes and complete nothing.
    //
    // So they are offered again, through the same per-textarea entry point that
    // extension uses for its own late arrivals. It is idempotent: it returns on
    // a textarea it has already claimed, and marks the ones it takes, so this
    // can run on every UI update and either extension may go first.
    //
    // Feature-detected in both directions and never awaited. No Tag
    // Autocomplete, a version without that entry point, or one whose config has
    // not loaded yet is a page where this does nothing -- and the boxes are
    // still ordinary prompt boxes, which is where they started.

    function joinAutocomplete(field) {
        if (!field || !field.classList) return;
        if (field.classList.contains("autocomplete")) return;
        if (typeof addAutocompleteToArea !== "function") return;
        if (typeof TAC_CFG === "undefined" || !TAC_CFG) return;
        addAutocompleteToArea(field);
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
    // 4. The space the prompt column was holding open
    // ------------------------------------------------------------------ //
    //
    // Forge's prompt container is `gr.Column(..., scale=6)`. In the classic top
    // row that 6 is a *width* share against the Generate column, which is what
    // it was written to be. In the Compact prompt layout the same column is
    // stacked inside the settings column instead, the settings column is a grid
    // item stretched to the height of the gallery beside it, and the 6 becomes
    // the largest claim on that leftover height: the container ends up hundreds
    // of pixels taller than the prompts in it, with the difference showing as
    // empty space underneath them. Measured in a browser at 989px of container
    // holding 168px of prompt, with no extension on the page at all.
    //
    // The Literal Prompt row lands in that container, so it inherits the gap and
    // -- being shorter side by side than stacked -- changes how much of it shows
    // when the divider moves. Hence this: the container is told to take its
    // content's height, which is the height it appeared to have anyway.
    //
    // Guarded three ways, because this is somebody else's element. Only when the
    // row is actually in it, only when its parent stacks vertically (in the
    // classic top row that flex-grow is doing real work and zeroing it would
    // squeeze the prompt boxes to their content width), and only once. It reads
    // no value, moves nothing, and a failure leaves the gap that was there
    // before.

    function reclaim() {
        const row = byId(IDS.row);
        const negative = byId(IDS.negativePrompt);
        if (!row || !negative || typeof negative.closest !== "function") return;
        const container = negative.closest('[id$="_prompt_container"]');
        if (!container || !container.parentElement || !container.dataset) return;
        if (container.dataset.mcLiteralHugged) return;
        if (typeof container.contains !== "function" || !container.contains(row)) return;
        if (typeof getComputedStyle !== "function") return;
        if (getComputedStyle(container.parentElement).flexDirection !== "column") return;
        if (!(parseFloat(getComputedStyle(container).flexGrow) > 0)) return;
        container.style.flexGrow = "0";
        container.dataset.mcLiteralHugged = "1";
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
        try {
            reclaim();
        } catch (error) {
            console.error("Model Chain: the prompt column could not be asked to fit "
                          + "its contents", error);
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
        family: family,
        reclaim: reclaim,
        ids: IDS,
        collapsedClass: COLLAPSED,
    };
})();
