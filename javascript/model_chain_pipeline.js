// Model Chain -- the Image Pipeline panel, browser side.
//
// Two cosmetic jobs and no third one:
//
//   1. The Prompt row echoes what is in the prompt box, as it is typed.
//   2. During a generation, the row that is currently working says so.
//
// Both are *reflections*. Nothing here decides anything, stores anything, or
// takes part in a generation: if every line of this file fails, the pipeline
// still shows the right stages with the right switches, the right settings
// still travel with the next press, and the picture still arrives. The panel
// is built and wired in Python -- see mc_pipeline_panel -- and this only
// describes what is already happening.
//
// Why the prompt echo is here and not in Python
// ---------------------------------------------
// It follows every keystroke. A Gradio round trip per keystroke is not a thing
// to do to somebody's typing, and the value being echoed is one the user is
// looking at anyway -- so the cheapest correct implementation is also the only
// polite one.
//
// Why the phase animation reads the progress bar
// ----------------------------------------------
// Because that is where the truth already is. mc_progress runs one ordered
// list of phases for the whole job and writes the current phase's label into
// the progress response; the host prefixes that label to the bar's text. This
// file reads the label and lights the matching row.
//
// It is deliberately *not* a second progress calculation. There is no timer
// predicting anything, no arithmetic on percentages, and no state that could
// disagree with the bar -- when the bar says "Stage 2", this says Stage 2, and
// when the bar says nothing this says nothing. §4 of the design intent asks
// for cosmetic system status, and the way to keep it cosmetic is to give it
// nothing of its own to be wrong about.
//
// The labels are matched as text, which is a real coupling, so it is a tested
// one: tests/test_pipeline.py asserts that every phase label the extension can
// produce is matched by the table below, and fails if one is renamed here or
// there without the other.

(function () {
    "use strict";

    const P = "mc-pipeline";

    const RUNNING = P + "-running";
    const DONE = P + "-done";

    const ECHO = P + "-prompt-echo";
    const PROMPT = "txt2img_prompt";

    // The Literal Prompt boxes. Read from here rather than reported by them,
    // because the Prompt row has exactly one writer and this is it -- two
    // files writing one element is how a line ends up alternating between two
    // truths depending on which event fired last.
    const LITERALS = {
        row: "mc-krea-creative-literal-row",
        positive: "mc-krea-creative-literal-positive",
        negative: "mc-krea-creative-literal-negative",
    };

    // How long the rows keep their finished state after the bar goes away.
    // Long enough to be seen, short enough that the next press starts clean.
    const SETTLE = 2500;

    // Label -> stage, in order, first match wins. Order is load-bearing:
    // "Waiting for Stage 1 preload" has to reach the Stage 1 rule before
    // anything more general sees it.
    const PHASES = [
        {match: "stage 1", stage: "stage1"},
        {match: "stage 2", stage: "stage2"},
        {match: "loading ", stage: "stage2"},
        {match: "finishing", stage: "output"},
        {match: "reading the layout", stage: "spatial"},
        {match: "reconciling the scene", stage: "spatial"},
        {match: "reading the prompt", stage: "creative"},
        {match: "writing the krea prompt", stage: "creative"},
    ];

    // Said by both language-model passes, so on its own it names no stage.
    // Handled by keeping whatever stage was running -- the writer and the
    // composer each announce themselves on the phase either side of their
    // wait, so "still that one" is always the right answer after the first.
    const AMBIGUOUS = "waiting for the language model";

    const ORDER = ["prompt", "creative", "spatial", "stage1", "stage2", "output"];

    const live = {
        stage: "",
        seen: [],
        settling: 0,
    };

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

    function rowFor(stage) {
        return byId(P + "-stage-" + stage);
    }

    // ------------------------------------------------------------------ //
    // The Prompt row
    // ------------------------------------------------------------------ //

    function trim(text) {
        const said = String(text || "").replace(/\s+/g, " ").trim();
        if (!said) return "";
        return said.length > 120 ? said.slice(0, 119) + "…" : said;
    }

    function valueOf(id) {
        const holder = byId(id);
        const box = holder ? holder.querySelector("textarea, input") : null;
        return box ? String(box.value || "").trim() : "";
    }

    // Whether an element is not currently taking up space. `offsetParent` is
    // null for anything with `display: none` anywhere above it, which is what
    // Gradio does to a component it has been told is not visible.
    function offScreen(element) {
        if (!element) return true;
        return element.offsetParent === null;
    }

    // §3.3: hidden does not mean inactive. The Literal Prompt boxes keep
    // reaching every generation while their row is off screen, so the row being
    // off screen is exactly when somebody needs telling that they are in
    // effect. Said only then -- while the boxes are visible they speak for
    // themselves, and a count beside them would be furniture.
    function literalNote() {
        if (!offScreen(byId(LITERALS.row))) return "";
        const active = [LITERALS.positive, LITERALS.negative]
            .filter(function (id) { return valueOf(id) !== ""; }).length;
        if (!active) return "";
        return active + " literal" + (active === 1 ? "" : "s") + " active";
    }

    function echo() {
        const target = byId(ECHO);
        if (!target) return;
        const holder = byId(PROMPT);
        const box = holder ? holder.querySelector("textarea, input") : null;
        const said = trim(box ? box.value : "");
        const note = literalNote();
        // textContent, never innerHTML: this is somebody's prompt, and a prompt
        // is the one string on the page most likely to contain angle brackets.
        target.textContent = [said || "nothing typed yet", note]
            .filter(Boolean).join(" · ");
    }

    function watchPrompt() {
        const holder = byId(PROMPT);
        const box = holder ? holder.querySelector("textarea, input") : null;
        if (box && box.dataset && !box.dataset.mcPipelineEcho) {
            box.dataset.mcPipelineEcho = "1";
            box.addEventListener("input", echo);
            box.addEventListener("change", echo);
        }
        // The literal boxes drive the same line, so they are followed too. The
        // count changes when one is emptied, not only when the row is hidden.
        [LITERALS.positive, LITERALS.negative].forEach(function (id) {
            const owner = byId(id);
            const field = owner ? owner.querySelector("textarea, input") : null;
            if (!field || !field.dataset || field.dataset.mcPipelineEcho) return;
            field.dataset.mcPipelineEcho = "1";
            field.addEventListener("input", echo);
            field.addEventListener("change", echo);
        });
        echo();
    }

    // ------------------------------------------------------------------ //
    // The running row
    // ------------------------------------------------------------------ //

    function stageFor(label) {
        const said = String(label || "").trim().toLowerCase();
        if (!said) return "";
        if (said.indexOf(AMBIGUOUS) === 0) return live.stage || "creative";
        for (const entry of PHASES) {
            if (said.indexOf(entry.match) !== -1) return entry.stage;
        }
        return "";
    }

    function clear() {
        ORDER.forEach(function (stage) {
            const row = rowFor(stage);
            if (!row || !row.classList) return;
            row.classList.remove(RUNNING);
            row.classList.remove(DONE);
        });
        live.stage = "";
        live.seen = [];
    }

    function show(stage) {
        if (!stage || stage === live.stage) return;
        if (live.stage && live.seen.indexOf(live.stage) === -1) {
            live.seen.push(live.stage);
        }
        live.stage = stage;

        ORDER.forEach(function (name) {
            const row = rowFor(name);
            if (!row || !row.classList) return;
            row.classList.toggle(RUNNING, name === stage);
            // A stage that was never entered is a stage that was skipped, and
            // a skipped stage gets nothing -- which is how "bypassed" stays
            // legible while the pipeline is running.
            row.classList.toggle(DONE, live.seen.indexOf(name) !== -1);
        });
    }

    function read(bar) {
        if (!bar) return;
        const stage = stageFor(bar.textContent);
        if (stage) show(stage);
    }

    function watchBar(progressDiv) {
        if (!progressDiv || !progressDiv.querySelector) return;
        const bar = progressDiv.querySelector(".progress");
        if (!bar) return;
        if (live.settling) {
            clearTimeout(live.settling);
            live.settling = 0;
        }
        clear();
        read(bar);

        if (typeof MutationObserver !== "function") return;
        // The host rewrites the bar's text about twice a second; this reads it
        // and does nothing else, so the observer costs a string comparison per
        // update and can never write anything the host will read back.
        const watcher = new MutationObserver(function () {
            try {
                if (!bar.isConnected && bar.isConnected !== undefined) {
                    watcher.disconnect();
                    finish();
                    return;
                }
                read(bar);
            } catch (error) {
                watcher.disconnect();
            }
        });
        watcher.observe(bar, {childList: true, characterData: true, subtree: true});

        // The bar is removed when the job ends, which is the only reliable
        // "finished" signal available from out here.
        if (progressDiv.parentNode) {
            const gone = new MutationObserver(function () {
                if (progressDiv.parentNode) return;
                gone.disconnect();
                watcher.disconnect();
                finish();
            });
            gone.observe(progressDiv.parentNode, {childList: true});
        }
    }

    function finish() {
        // Everything that ran shows finished for a moment, then the panel goes
        // back to describing the next press rather than the last one.
        const row = rowFor(live.stage);
        if (row && row.classList) {
            row.classList.remove(RUNNING);
            row.classList.add(DONE);
        }
        if (live.settling) clearTimeout(live.settling);
        live.settling = setTimeout(function () {
            live.settling = 0;
            clear();
        }, SETTLE);
    }

    function watchForBars() {
        if (typeof MutationObserver !== "function" || !document.body) return;
        new MutationObserver(function (records) {
            for (const record of records) {
                for (const node of record.addedNodes) {
                    if (!node || node.nodeType !== 1) continue;
                    try {
                        if (node.classList && node.classList.contains("progressDiv")) {
                            watchBar(node);
                        } else if (node.querySelectorAll) {
                            node.querySelectorAll(".progressDiv").forEach(watchBar);
                        }
                    } catch (error) {
                        console.error("Model Chain: the pipeline phase indicator "
                                      + "could not follow this generation", error);
                    }
                }
            }
        }).observe(document.body, {childList: true, subtree: true});
    }

    // ------------------------------------------------------------------ //
    // Wiring
    // ------------------------------------------------------------------ //

    function wire() {
        try {
            watchPrompt();
        } catch (error) {
            console.error("Model Chain: the pipeline prompt echo failed", error);
        }
    }

    if (typeof onUiLoaded === "function") {
        onUiLoaded(function () {
            wire();
            try {
                watchForBars();
            } catch (error) {
                console.error("Model Chain: could not watch for the progress bar",
                              error);
            }
        });
    }

    // Gradio rebuilds parts of the tab on some updates, which can replace the
    // prompt box. Guarded by a dataset flag, so re-running costs a query.
    if (typeof onAfterUiUpdate === "function") onAfterUiUpdate(wire);

    // Exposed for the tests, which drive this file under node against a fake
    // page. Nothing in the extension reads it.
    window.modelChainPipeline = {
        literalNote: literalNote,
        echo: echo,
        stageFor: stageFor,
        phases: PHASES,
        order: ORDER,
        live: live,
        echo: echo,
        wire: wire,
        watchBar: watchBar,
        clear: clear,
    };
})();
