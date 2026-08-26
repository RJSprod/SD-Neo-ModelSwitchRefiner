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

    // Written by mc_pipeline_panel: the class on every disclosure this
    // extension builds.
    const DRAWER = P + "-drawer";

    // The stage card, its header once this file has found it, and the card
    // once that has happened. All four are mc_pipeline_panel's names -- see
    // CARD_HEAD and CARDED there, which is where they are documented.
    const STAGE = P + "-stage";
    const CARD_HEAD = P + "-card-head";
    const CARDED = P + "-carded";
    const NAME = P + "-name";
    const SUMMARY = P + "-said";

    // The one child of the header that holds the two lines. Everything else in
    // there is the theme's -- a chevron, a caret, a marker -- and is hidden,
    // because the header is this file's band now and the right-hand end of it
    // belongs to the switch.
    const LABEL = P + "-label";

    // The first line of a stage card's label. Python's mc_pipeline_panel.TITLES
    // says these, and tests/test_pipeline.py fails if the two lists drift --
    // the same coupling PHASES has, for the same reason.
    const TITLES = ["Creative", "Spatial", "Stage 2"];

    // The Literal Prompt boxes. Read from here rather than reported by them,
    // because the note has exactly one writer and this is it -- two files
    // writing one element is how a line ends up alternating between two truths
    // depending on which event fired last.
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
    //
    // `stage1` and `output` are still in here and no longer rows on the panel.
    // Recognising a phase and having nowhere to draw it is the right answer for
    // those two; not recognising a label at all is the failure this table
    // exists to prevent, and the two must not be confused for one another.
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

    // The rows the panel draws. Stage 1 and Output are still phases a run goes
    // through -- see PHASES -- and no longer rows anybody looks at, so a phase
    // that names one lights nothing and that is the whole of it.
    const ORDER = ["creative", "spatial", "stage2"];

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
    // What is in effect while its own control is off screen
    // ------------------------------------------------------------------ //

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
        // textContent, never innerHTML: what this says is counted from what
        // somebody typed, and a prompt is the string on the page most likely to
        // contain angle brackets. An empty note is an empty element, which the
        // stylesheet gives no height at all.
        target.textContent = literalNote();
    }

    function watchPrompt() {
        // The literal boxes drive the line: the count changes when one is
        // emptied, not only when the row goes off screen.
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
    // Finding a disclosure's header
    // ------------------------------------------------------------------ //
    //
    // A Gradio Accordion is two elements: the thing you press and the thing it
    // shows. That is the whole rule, and it is the only one that survives a
    // theme -- Lobe rebuilds the header as a <button>, stock Gradio renders a
    // <span>, and a future version may render something neither of them is.
    // Counting children names no class, assumes no tag, and is true of every
    // one of them.
    //
    // This used to look for a button, or for an element that said whether it
    // was expanded, and then move the switch and the summary into whatever it
    // found. It worked under Lobe and found nothing at all under stock Gradio,
    // which is how the panel ended up with its header stacked in three pieces
    // on the theme most people are using. The layout is the stylesheet's job
    // now -- it can select "the accordion's first child" without any of this --
    // and the only thing left that needs the header is remembering which
    // drawers were open, which is a press rather than a rearrangement.

    function headerOf(accordion) {
        if (!accordion || !accordion.children) return null;
        if (accordion.children.length < 2) return null;

        // The one this file dressed, if it got that far: `twoLines` walks to
        // the element Gradio actually put the label in and marks it, which is
        // the only answer here that is not an assumption about the shape of
        // somebody else's DOM.
        const marked = accordion.querySelector("." + CARD_HEAD);
        if (marked && marked.parentNode === accordion) return marked;

        // Otherwise the first child, which is true of a disclosure in general:
        // the thing you press is drawn before the thing it shows.
        //
        // It used to be first child *and exactly two children*, and the count
        // is what broke it. A build that puts anything else inside the
        // accordion -- one extra node is enough -- made this return null, and
        // then nothing here ran at all: no drawer was remembered and no header
        // was found. A guard that answers "no" to a question it could have
        // answered is worse than the risk it was avoiding.
        return accordion.children[0] || null;
    }

    // ------------------------------------------------------------------ //
    // Two lines in a card header
    // ------------------------------------------------------------------ //
    //
    // Python writes a stage card's header as one string with one newline in
    // it -- `Creative\nBypassed - prompt as-is` -- because an Accordion's own
    // label is the only place a Gradio disclosure carries text that is
    // guaranteed to be inside its header.
    //
    // Making that newline *show* was left to the stylesheet, and the
    // stylesheet lost. Three rounds of it: `white-space: pre` and
    // `::first-line` both have to reach an element whose shape and selector
    // belong to Gradio and whose styling belongs to the theme, and a rule that
    // has to win a cascade it does not control is a rule that works until
    // somebody changes their theme.
    //
    // So the newline is not styled any more, it is *removed*: the text node
    // becomes two elements with two classes of this extension's own, and the
    // stylesheet is left addressing its own elements. Nothing here guesses at
    // structure -- it finds the text Python wrote, wherever the theme has put
    // it, and works outwards from there.
    //
    // If it cannot be found, nothing is marked, and the card falls back to a
    // plain stack that is still readable and still works. That is what CARDED
    // is for: the layout that assumes a lane on the right is a stylesheet
    // state, not a hope.

    function labelNode(accordion) {
        // The text node carrying a stage card's label: two lines, and a first
        // line that is one of the three names this panel gives a stage. Both
        // halves matter -- a body full of prose has newlines in it too, and
        // this walk cannot know where the header ends.
        let walk;
        try {
            walk = document.createTreeWalker(accordion, NodeFilter.SHOW_TEXT, null);
        } catch (error) {
            return null;
        }
        let node;
        while ((node = walk.nextNode())) {
            const said = node.nodeValue || "";
            const cut = said.indexOf("\n");
            if (cut <= 0) continue;
            if (TITLES.indexOf(said.slice(0, cut).trim()) === -1) continue;
            return node;
        }
        return null;
    }

    function twoLines(accordion) {
        if (!accordion || accordion.querySelector("." + NAME)) return false;

        const node = labelNode(accordion);
        if (!node || !node.parentNode) return false;

        const said = node.nodeValue.split("\n");
        const name = said.shift().trim();
        const rest = said.join(" ").trim();
        if (!name) return false;

        // Divs, and that is not arbitrary. A theme that styles a card header
        // reaches the text through the span inside it -- that is the shape
        // every one of those rules has -- and the two lines were coming out at
        // the same size because such a rule was reaching them. Being a
        // different element is not a trick: these are two stacked blocks,
        // which is what a div is for and what a span is not.
        const holder = node.parentNode;

        // The header first, because what is marked afterwards is relative to
        // it: whichever ancestor of the text sits directly inside the
        // accordion, however many wrappers a theme put in between.
        let head = holder;
        while (head && head.parentNode && head.parentNode !== accordion) {
            head = head.parentNode;
        }
        if (!head || head.parentNode !== accordion) return false;

        const first = document.createElement("div");
        first.className = NAME;
        first.textContent = name;

        const second = document.createElement("div");
        second.className = SUMMARY;
        second.textContent = rest;
        // The line is cut to fit, so the whole of it is worth having on hover.
        if (rest) second.title = rest;

        try {
            holder.replaceChild(second, node);
            holder.insertBefore(first, second);
        } catch (error) {
            return false;
        }

        if (head.classList) head.classList.add(CARD_HEAD);

        // And the lane the two lines are in, so the stylesheet can hide the
        // header's other children without having to know what they are. A
        // theme's own marker or caret is fine on a header it owns; on this one
        // it is furniture in a band that has exactly two lines' room and a
        // switch painted over the end of it.
        [first, second].forEach(function (line) {
            let lane = line;
            while (lane && lane.parentNode && lane.parentNode !== head) {
                lane = lane.parentNode;
            }
            if (lane && lane.classList) lane.classList.add(LABEL);
        });

        const card = accordion.closest ? accordion.closest("." + STAGE) : null;
        if (card && card.classList) card.classList.add(CARDED);
        return true;
    }

    function dressCards() {
        drawers().forEach(function (accordion) {
            try {
                twoLines(accordion);
            } catch (error) {
                console.error("Model Chain: a pipeline card header could not be "
                              + "split into two lines", error);
            }
        });
    }

    // Gradio repaints a card's description by setting the accordion's label,
    // which replaces the text and takes this file's two spans with it. Watching
    // is cheaper than re-running on a timer and catches the repaint in the same
    // frame it happens in.
    function watchCards() {
        if (typeof MutationObserver !== "function") return;
        drawers().forEach(function (accordion) {
            if (!accordion.dataset || accordion.dataset.mcPipelineCards) return;
            accordion.dataset.mcPipelineCards = "1";
            new MutationObserver(function () {
                try {
                    twoLines(accordion);
                } catch (error) {
                    /* a repaint this file could not follow is not fatal */
                }
            }).observe(accordion, {childList: true, subtree: true,
                                   characterData: true});
        });
    }

    // ------------------------------------------------------------------ //
    // What was opened stays opened
    // ------------------------------------------------------------------ //
    //
    // Every drawer this extension builds starts closed, which is the right
    // default exactly once. After that, a tab that folds everything away again
    // on every reload is a tab somebody has to re-open four drawers in before
    // they can carry on.
    //
    // So the open ones are remembered, by element id, in localStorage -- a
    // per-browser preference about furniture, which is what it is. Nothing
    // about a generation is stored here and nothing here is ever sent
    // anywhere: if the whole store is lost, every drawer is closed and one
    // press opens the one that was wanted.
    //
    // Restoring is a *click*, not a class. Gradio owns whether an accordion is
    // open and re-renders it from its own state; setting the class would leave
    // the two disagreeing at the first update. Pressing the header is the same
    // thing the user would have done.

    const REMEMBERED = "modelChainOpenDrawers";

    function opened() {
        try {
            const said = window.localStorage.getItem(REMEMBERED);
            const read = said ? JSON.parse(said) : null;
            return read && typeof read === "object" ? read : {};
        } catch (error) {
            // Private browsing, a blocked store, a value somebody edited by
            // hand. Every one of them means "no preferences", which is the
            // state this shipped in.
            return {};
        }
    }

    function remember(id, open) {
        try {
            const found = opened();
            if (open) found[id] = 1;
            else delete found[id];
            window.localStorage.setItem(REMEMBERED, JSON.stringify(found));
        } catch (error) {
            // Nothing is remembered this session. The drawer still opens.
        }
    }

    // Asked of the page rather than of a class name. The body is the
    // accordion's other child and Gradio hides it rather than removing it, so
    // "is this open" is "is the body taking up space" -- which is true under
    // any theme, any Gradio version, and any class this file has never heard of.
    function isOpen(accordion, header) {
        if (!accordion) return false;
        if (header && header.getAttribute) {
            const said = header.getAttribute("aria-expanded");
            if (said === "true") return true;
            if (said === "false") return false;
        }
        const body = accordion.children[accordion.children.length - 1];
        if (!body || body === header) return false;
        return !offScreen(body);
    }

    // Every disclosure this extension built, and nothing else on the page.
    // The class is put there by mc_pipeline_panel.drawer(), which is the only
    // way this extension makes one -- so this cannot pick up a host accordion,
    // a theme's own furniture, or a Column that happens to start with a button.
    function drawers() {
        try {
            return Array.prototype.slice.call(root().querySelectorAll("." + DRAWER));
        } catch (error) {
            return [];
        }
    }

    function watchDrawers() {
        const wanted = opened();
        drawers().forEach(function (accordion) {
            const header = headerOf(accordion);
            if (!header || !header.dataset) return;
            if (!header.dataset.mcPipelineDrawer) {
                header.dataset.mcPipelineDrawer = "1";
                // After the press, not before: Gradio toggles on the same
                // event, so the state worth recording is the one that follows.
                header.addEventListener("click", function () {
                    remember(accordion.id, !isOpen(accordion, header));
                });
                if (wanted[accordion.id] && !isOpen(accordion, header)) {
                    header.click();
                }
            }
        });
    }

    // ------------------------------------------------------------------ //
    // Wiring
    // ------------------------------------------------------------------ //

    function wire() {
        // Before the drawers: headerOf() prefers the header this marks, and
        // finding it by the text Python wrote beats assuming which child it is.
        try {
            dressCards();
            watchCards();
        } catch (error) {
            console.error("Model Chain: the pipeline card headers could not be "
                          + "split into two lines", error);
        }
        try {
            watchPrompt();
        } catch (error) {
            console.error("Model Chain: the pipeline prompt echo failed", error);
        }
        try {
            watchDrawers();
        } catch (error) {
            console.error("Model Chain: which drawers were open was not restored",
                          error);
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
        watchDrawers: watchDrawers,
        headerOf: headerOf,
        twoLines: twoLines,
        dressCards: dressCards,
        watchCards: watchCards,
        labelNode: labelNode,
        titles: TITLES,
        drawers: drawers,
        opened: opened,
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
