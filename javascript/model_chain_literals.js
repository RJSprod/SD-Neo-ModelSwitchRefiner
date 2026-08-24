// Model Chain -- the Literal Prompt boxes, browser side.
//
// Three jobs, all of them presentation, none of them able to change what a
// generation produces:
//
//   1. Move the Literal Prompt row up under the native Negative Prompt.
//   2. Put its two boxes into Forge's own "last prompt box you used" family,
//      so the Extra Networks browser inserts into whichever of the four you
//      touched last -- and offer them to Tag Autocomplete, if it is installed.
//   3. Hide the native Negative Prompt row while CFG is 1, and give its space
//      back.
//
// ...and one line in the log saying whether 1 and 2 worked, because every fact
// about that lives in the browser and the alternative is asking somebody to
// open the developer tools.
//
// Nothing here styles, sizes or moves anything the host owns. The one exception
// is job 3, which puts a class on Forge's own Negative Prompt and takes it off
// again -- and an earlier version of this file also set `flex-grow: 0` on
// Forge's prompt column to reclaim the empty space it holds open. That space is
// the host's (`gr.Column(scale=6)` claiming the gallery's height), and so is the
// decision about it: writing to that element changed how somebody else's page
// lays out, which is not this component's business. It is gone.
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
        extendAutocompleteList();
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
            // The same event, for the report: the first time somebody puts the
            // caret in one of these boxes, everything that was going to load
            // has loaded.
            once(field, "focus", sendReport, "Report");
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

    function autocompleteInstalled() {
        return typeof addAutocompleteToArea === "function";
    }

    function joinAutocomplete(field) {
        if (!field || !field.classList) return;
        if (field.classList.contains("autocomplete")) return;
        if (!autocompleteInstalled()) return;
        // `var TAC_CFG = null` until it has read its tag files. Calling in
        // before that is calling into a half-built extension; there is always
        // another attempt.
        if (typeof TAC_CFG === "undefined" || !TAC_CFG) return;
        addAutocompleteToArea(field);
    }

    // ...and the other half of the same job, from the other direction, which is
    // what makes the ordering between the two extensions stop mattering: no
    // timer waits for that extension to finish loading, because its own setup
    // asks for the list below whenever it gets there.
    //
    // That extension decides what to attach to by calling its own
    // `getTextAreas()`, once, inside a setup that runs when its files have
    // loaded. Handing it two textareas afterwards (above) only works if this
    // file gets a turn after that setup; extending the list it asks for works
    // whichever of the two goes first, and also covers the re-scans it does for
    // accordions that open later.
    //
    // `getTextAreas` is a top-level function declaration in a classic script,
    // so it is a writable property of the global object and the call inside
    // that extension resolves through it. Wrapped once, additive only: its list
    // comes back with its own contents in its own order, and if it ever throws
    // that is its business and the error is its own.

    function extendAutocompleteList() {
        if (!autocompleteInstalled()) return;
        if (typeof getTextAreas !== "function" || getTextAreas.mcLiteral) return;
        const original = getTextAreas;
        const extended = function () {
            const found = original.apply(this, arguments);
            try {
                if (Array.isArray(found)) {
                    [IDS.positive, IDS.negative].forEach(function (id) {
                        const field = fieldIn(id);
                        if (field && found.indexOf(field) < 0) found.push(field);
                    });
                }
            } catch (error) {
                // Their list, unchanged, is a perfectly good answer.
            }
            return found;
        };
        extended.mcLiteral = true;
        window.getTextAreas = extended;
    }

    // ------------------------------------------------------------------ //
    // 2c. Saying so, once, in the log everything else is in
    // ------------------------------------------------------------------ //
    //
    // Everything above depends on two other extensions' cooperation, and every
    // fact about whether it worked lives in the browser. "Tag completion does
    // not work in these boxes" was answered, twice, by asking somebody to open
    // the developer tools and paste a snippet -- which is a fine thing to ask
    // of the person who wrote the snippet and a poor thing to ask of anybody
    // else.
    //
    // So the page says it. On the first focus of a literal box -- which is
    // exactly when tag completion is the thing somebody is expecting to happen
    // -- a small fixed-shape report goes to the extension's own route and
    // becomes one line in the WebUI's log, beside every other message this
    // extension writes. Once per page load.
    //
    // Booleans and one word, and no way for text to get in: mc_literal_report
    // reads a fixed set of keys and coerces every one of them. A diagnostic
    // that could carry what somebody typed into a prompt box would be a
    // diagnostic nobody should install.

    const REPORT_ROUTE = "/model-chain/literal-prompts/report";
    let reported = false;

    function autocompleteConfig() {
        if (typeof TAC_CFG === "undefined") return "missing";
        return TAC_CFG ? "loaded" : "null";
    }

    function thirdPartyBoxes() {
        try {
            if (typeof TAC_CFG === "undefined" || !TAC_CFG || !TAC_CFG.activeIn) {
                return null;
            }
            return !!TAC_CFG.activeIn.thirdParty;
        } catch (error) {
            return null;
        }
    }

    function inTheirList() {
        try {
            if (typeof getTextAreas !== "function") return false;
            const list = getTextAreas();
            if (!Array.isArray(list)) return false;
            return [IDS.positive, IDS.negative].every(function (id) {
                const field = fieldIn(id);
                return !!field && list.indexOf(field) >= 0;
            });
        } catch (error) {
            return false;
        }
    }

    function report() {
        if (reported) return null;
        const positive = fieldIn(IDS.positive);
        const negative = fieldIn(IDS.negative);
        const row = byId(IDS.row);
        return {
            boxesFound: !!positive && !!negative,
            claimed: claimed(),
            autocompleteInstalled: autocompleteInstalled(),
            listWrapped: typeof getTextAreas === "function" && !!getTextAreas.mcLiteral,
            inTheirList: inTheirList(),
            config: autocompleteConfig(),
            thirdPartyBoxes: thirdPartyBoxes(),
            promptFamily: !!family(),
            placed: !!row && !!byId(IDS.negativePrompt)
                    && row.previousElementSibling === byId(IDS.negativePrompt),
        };
    }

    function claimed() {
        return [IDS.positive, IDS.negative].every(function (id) {
            const field = fieldIn(id);
            return !!field && !!field.classList
                && field.classList.contains("autocomplete");
        });
    }

    // Posted and forgotten. No await, no retry, no reading of the answer: if
    // the route is not there, or the fetch fails, or this is a build without
    // `fetch` at all, the boxes carry on being boxes.
    function sendReport() {
        if (reported) return;
        const found = report();
        if (!found) return;
        reported = true;
        if (typeof fetch !== "function") return;
        try {
            fetch(REPORT_ROUTE, {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify(found),
            }).catch(function () { /* a log line, not a feature */ });
        } catch (error) {
            /* a log line, not a feature */
        }
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
        family: family,
        joinAutocomplete: joinAutocomplete,
        report: report,
        sendReport: sendReport,
        claimed: claimed,
        extendAutocompleteList: extendAutocompleteList,
        ids: IDS,
        collapsedClass: COLLAPSED,
    };
})();
