"""The Literal Prompt boxes' browser file, executed rather than read.

Three jobs, and each one is a promise about somebody else's page: move a row,
join a family, hide a control. What makes them acceptable at all is that none
of them is load-bearing -- if every line of the file fails the feature still
works, one scroll further down, with the LoRA browser behaving exactly as it
did before. So the tests come in two kinds.

**Does it do the thing?** The row ends up under the Negative Prompt, the two
boxes become the remembered insertion target when focused, and the Negative
Prompt row collapses at CFG 1 and comes back above it.

**Does it stay harmless when it cannot?** A page with no Negative Prompt, a
build with no ``activePromptTextarea``, a CFG that will not parse -- each of
those has one right answer, which is to leave the page alone and carry on. The
wrong answers are all silent, and two of them lose somebody's text.

These run under node, which is not a Forge dependency, so they skip without it.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "javascript" / "model_chain_literals.js"
PIPELINE = ROOT / "javascript" / "model_chain_pipeline.js"

pytestmark = pytest.mark.skipif(shutil.which("node") is None,
                                reason="node is not installed")


HARNESS = """
// A txt2img toprow with the four prompt boxes and a CFG slider in it, and just
// enough DOM for the file under test to be the real file.

class El {
    constructor(tag) {
        this.tagName = String(tag).toUpperCase();
        this.id = "";
        this.children = [];
        this.parentNode = null;
        this.dataset = {};
        this.style = {};
        this.value = "";
        this.type = "";
        this._classes = new Set();
        this._listeners = {};
        // Nothing here lays anything out, so "on screen" is a property a test
        // sets rather than one the harness computes.
        this.offsetParent = {};
    }
    get className() { return Array.from(this._classes).join(" "); }
    set className(value) {
        this._classes = new Set(String(value).split(/\\s+/).filter(Boolean));
    }
    get classList() {
        const owner = this;
        return {
            add(name) { owner._classes.add(name); },
            remove(name) { owner._classes.delete(name); },
            contains(name) { return owner._classes.has(name); },
            toggle(name, force) {
                const on = force === undefined ? !owner._classes.has(name) : !!force;
                if (on) owner._classes.add(name); else owner._classes.delete(name);
            },
        };
    }
    appendChild(child) {
        if (child.parentNode) child.parentNode.removeChild(child);
        child.parentNode = this;
        this.children.push(child);
        return child;
    }
    removeChild(child) {
        this.children = this.children.filter((entry) => entry !== child);
        child.parentNode = null;
        return child;
    }
    insertBefore(child, before) {
        if (child.parentNode) child.parentNode.removeChild(child);
        child.parentNode = this;
        const at = before ? this.children.indexOf(before) : -1;
        if (at < 0) this.children.push(child);
        else this.children.splice(at, 0, child);
        return child;
    }
    get parentElement() { return this.parentNode; }
    closest(selector) {
        let at = this;
        while (at) {
            if (matches(at, selector)) return at;
            at = at.parentNode;
        }
        return null;
    }
    contains(other) {
        let at = other;
        while (at) {
            if (at === this) return true;
            at = at.parentNode;
        }
        return false;
    }
    get previousElementSibling() {
        if (!this.parentNode) return null;
        const at = this.parentNode.children.indexOf(this);
        return at > 0 ? this.parentNode.children[at - 1] : null;
    }
    get nextSibling() {
        if (!this.parentNode) return null;
        const at = this.parentNode.children.indexOf(this);
        return this.parentNode.children[at + 1] || null;
    }
    addEventListener(kind, fn) {
        (this._listeners[kind] = this._listeners[kind] || []).push(fn);
    }
    dispatchEvent(event) {
        event.target = event.target || this;
        (this._listeners[event.type] || []).forEach((fn) => fn(event));
        return true;
    }
    walk() {
        return this.children.reduce(function (found, child) {
            return found.concat([child], child.walk());
        }, []);
    }
    querySelector(selector) {
        return this.walk().filter((el) => matches(el, selector))[0] || null;
    }
    querySelectorAll(selector) {
        return this.walk().filter((el) => matches(el, selector));
    }
}

function matches(element, selector) {
    return selector.split(",").map((part) => part.trim()).filter(Boolean)
        .some(function (part) {
            if (part.startsWith("#")) return element.id === part.slice(1);
            if (part.startsWith(".")) return element._classes.has(part.slice(1));
            if (part.includes("[")) {
                const tag = part.slice(0, part.indexOf("["));
                const want = part.slice(part.indexOf("=") + 1).replace(/[\\]"']/g, "");
                // `[id$="_prompt_container"]`: how the file finds Forge's
                // prompt column without knowing which tab it is on.
                if (part.includes("id$=")) return element.id.endsWith(want);
                return element.tagName === tag.toUpperCase() && element.type === want;
            }
            return element.tagName === part.toUpperCase();
        });
}

const body = new El("body");
const page = body.appendChild(new El("div"));

function make(tag, id, into) {
    const element = new El(tag);
    element.id = id;
    (into || page).appendChild(element);
    return element;
}

// The toprow, in the nesting Forge builds: the prompt and the negative prompt
// are siblings, which is what the row has to become one of.
const toprow = make("div", "txt2img_toprow");
const prompt = make("div", "txt2img_prompt", toprow);
prompt.appendChild(new El("textarea"));
const negative = make("div", "txt2img_neg_prompt", toprow);
negative.appendChild(new El("textarea"));
negative.querySelector("textarea").value = "blurry, low quality";

// The CFG slider: Gradio renders a range and a number, and either may be read.
const cfgHolder = make("div", "txt2img_cfg_scale");
const cfgRange = new El("input");
cfgRange.type = "range";
cfgRange.value = "7";
cfgHolder.appendChild(cfgRange);

// The Image Pipeline's Prompt row, which is where the "N literals active" note
// is written when the row above is off screen.
const echo = make("span", "mc-pipeline-prompt-echo");
make("div", "mc-pipeline-stage-prompt");

// The Literal Prompt row, where Gradio put it: at the bottom of the page.
const row = make("div", "mc-krea-creative-literal-row");
const literalPositive = make("div", "mc-krea-creative-literal-positive", row);
literalPositive.appendChild(new El("textarea"));
const literalNegative = make("div", "mc-krea-creative-literal-negative", row);
literalNegative.appendChild(new El("textarea"));

globalThis.document = {
    body: body,
    readyState: "complete",
    createElement: (tag) => new El(tag),
    querySelector: (selector) => body.querySelector(selector),
    querySelectorAll: (selector) => body.querySelectorAll(selector),
    getElementById: (id) => body.querySelector("#" + id),
    addEventListener() {},
};

globalThis.window = globalThis;
globalThis.gradioApp = () => globalThis.document;
globalThis.Event = function (type) { this.type = type; };
globalThis.MutationObserver = function () {
    return {observe() {}, disconnect() {}};
};
// Nothing here lays anything out, so a computed style is whatever was set
// inline -- which is all the file asks about: an axis and a flex-grow.
globalThis.getComputedStyle = (element) => ({
    flexDirection: (element.style || {}).flexDirection || "row",
    flexGrow: String((element.style || {}).flexGrow || "0"),
});
globalThis.posted = [];
globalThis.fetch = (url, options) => {
    posted.push({url: url, body: JSON.parse((options || {}).body || "null")});
    return {catch() {}};
};
globalThis.requestAnimationFrame = (fn) => fn();
globalThis.setTimeout = () => 0;
globalThis.clearTimeout = () => {};

const loaded = [];
globalThis.onUiLoaded = (fn) => loaded.push(fn);
globalThis.onAfterUiUpdate = () => {};

FAMILY

SOURCE

loaded.forEach((fn) => fn());

const mc = globalThis.modelChainLiterals;

function el(id) { return body.querySelector("#" + id); }
function fieldIn(id) {
    const holder = el(id);
    return holder ? holder.querySelector("textarea, input") : null;
}
function focus(id) {
    const field = fieldIn(id);
    field.dispatchEvent({type: "focus", target: field});
}
function setCfg(value) {
    cfgRange.value = String(value);
    cfgRange.dispatchEvent({type: "input", target: cfgRange});
}

const report = (extra) => console.log(JSON.stringify(Object.assign({
    afterNegative: row.previousElementSibling === negative,
    rowParentIsToprow: row.parentNode === toprow,
    negativeCollapsed: negative._classes.has("mc-literal-cfg-collapsed"),
    negativeText: negative.querySelector("textarea").value,
    positionInToprow: toprow.children.indexOf(row),
}, extra || {})));

BODY
"""


PLACEHOLDERS = ("SOURCE", "FAMILY", "BODY")
"""The words this harness substitutes to build itself.

A plain substring replace over the file under test, which is fine until that
file contains one of these words -- as a constant, or inside a longer name, or
even in a comment. Then the substitution lands mid-token and node reports a
syntax error twenty lines from anything to do with the test. It has happened
twice; `test_the_files_under_test_avoid_the_placeholders` is why it will not
happen a third time.
"""


def test_the_files_under_test_avoid_the_placeholders():
    for path in (PIPELINE, SCRIPT):
        source = path.read_text(encoding="utf-8")
        for word in PLACEHOLDERS:
            assert word not in source, (path.name, word)


def run_pipeline(script: str) -> dict:
    """The same page, driving the Image Pipeline's browser file instead.

    The note about hidden-but-active literals is written by that file and not
    by this one, because the Prompt row has exactly one writer -- two files
    writing one element is how a line ends up alternating between two truths
    depending on which event fired last.
    """
    harness = (HARNESS
               .replace("SOURCE", PIPELINE.read_text(encoding="utf-8"))
               .replace("FAMILY", "")
               .replace("const mc = globalThis.modelChainLiterals;",
                        "const mc = globalThis.modelChainPipeline;")
               .replace("BODY", script))
    done = subprocess.run([shutil.which("node"), "--input-type=module"],
                          input=harness, capture_output=True, text=True, timeout=60)
    if done.returncode != 0:
        print(done.stderr, file=sys.stderr)
        raise AssertionError(done.stderr.strip() or "node exited non-zero")
    lines = [line for line in done.stdout.splitlines() if line.strip().startswith("{")]
    assert lines, done.stdout or "the harness reported nothing"
    return json.loads(lines[-1])


def run(script: str, family=True) -> dict:
    """``family`` picks how the host declares ``activePromptTextarea``.

    ``True`` is the old ``var`` form, which lands on ``window``. ``"lexical"``
    is what Forge Neo actually writes -- a top-level ``let``, which goes into
    the global lexical environment and is *not* on ``window``. ``False`` is a
    build that has neither.
    """
    if family == "lexical":
        setup = "let activePromptTextarea = {};"
    elif family:
        setup = "globalThis.activePromptTextarea = {};"
    else:
        setup = "// no activePromptTextarea on this build"
    harness = (HARNESS
               .replace("SOURCE", SCRIPT.read_text(encoding="utf-8"))
               .replace("FAMILY", setup)
               .replace("BODY", script))
    done = subprocess.run([shutil.which("node"), "--input-type=module"],
                          input=harness, capture_output=True, text=True, timeout=60)
    if done.returncode != 0:
        print(done.stderr, file=sys.stderr)
        raise AssertionError(done.stderr.strip() or "node exited non-zero")
    lines = [line for line in done.stdout.splitlines() if line.strip().startswith("{")]
    assert lines, done.stdout or "the harness reported nothing"
    return json.loads(lines[-1])


# --------------------------------------------------------------------------- #
# Where the row ends up
# --------------------------------------------------------------------------- #


class TestThePlacement:
    """One insertBefore of an element Gradio owns, and a check first.

    Gradio addresses that element by id and does not care where in the document
    it is, which is what makes moving it safe. What would not be safe is moving
    it again on every UI update, or moving it into a container that is not
    there.
    """

    def test_the_row_lands_directly_under_the_negative_prompt(self):
        found = run("report({});")

        assert found["afterNegative"] is True
        assert found["rowParentIsToprow"] is True

    def test_moving_it_twice_leaves_it_where_it_was(self):
        """`onAfterUiUpdate` fires often, so this runs many times per session."""
        found = run("""
            mc.place();
            mc.place();
            report({children: toprow.children.length});
        """)

        assert found["afterNegative"] is True
        assert found["children"] == 3

    def test_a_page_with_no_negative_prompt_is_left_alone(self):
        """A theme that restructured the toprow, or a Forge build that renamed
        it. The row stays where Gradio put it, which is lower down and
        completely functional."""
        found = run("""
            negative.parentNode.removeChild(negative);
            const before = row.parentNode;
            mc.place();
            report({unmoved: row.parentNode === before});
        """)

        assert found["unmoved"] is True

    def test_the_row_is_never_rebuilt_or_copied(self):
        """Moved, not recreated: one element with one id, or Gradio would be
        addressing a node that is no longer the one on screen."""
        found = run("""
            const before = row;
            mc.place();
            report({same: body.querySelector("#mc-krea-creative-literal-row") === before,
                    count: body.querySelectorAll(
                        "#mc-krea-creative-literal-row").length});
        """)

        assert found["same"] is True
        assert found["count"] == 1


# --------------------------------------------------------------------------- #
# The last prompt box you used
# --------------------------------------------------------------------------- #


class TestThePromptFamily:
    """§7. Two more boxes in the set Forge already remembers, and nothing else.

    There is no formatter here, no caret arithmetic and no insertion code: all
    three already exist in the host, are already correct, and are already what
    the positive and negative prompt get. Adding a second one is how the LoRA
    tag in a literal box starts differing from the one in the prompt box.
    """

    def test_focusing_a_literal_box_makes_it_the_target(self):
        found = run("""
            focus("mc-krea-creative-literal-positive");
            report({target: activePromptTextarea.txt2img
                    === fieldIn("mc-krea-creative-literal-positive")});
        """)

        assert found["target"] is True

    def test_the_other_box_takes_over_when_it_is_focused(self):
        found = run("""
            focus("mc-krea-creative-literal-positive");
            focus("mc-krea-creative-literal-negative");
            report({target: activePromptTextarea.txt2img
                    === fieldIn("mc-krea-creative-literal-negative")});
        """)

        assert found["target"] is True

    def test_the_target_survives_focus_leaving_the_textarea(self):
        """The half of §7 that is about *not* doing something. The Extra
        Networks browser is not a prompt box, so opening it fires no focus
        event here and the remembered target is still the literal box."""
        found = run("""
            focus("mc-krea-creative-literal-positive");
            const browser = make("div", "txt2img_extra_networks");
            browser.dispatchEvent({type: "focus", target: browser});
            report({target: activePromptTextarea.txt2img
                    === fieldIn("mc-krea-creative-literal-positive")});
        """)

        assert found["target"] is True

    def test_a_build_without_the_family_is_not_a_crash(self):
        """Then this feature simply does not apply, and everything else on the
        page still works."""
        found = run("""
            focus("mc-krea-creative-literal-positive");
            report({ok: true});
        """, family=False)

        assert found["ok"] is True

    def test_a_host_that_declares_the_family_with_let_is_still_joined(self):
        """The bug this was written for, and the reason no LoRA ever reached
        these boxes.

        Forge Neo's extraNetworks.js says `let activePromptTextarea = {}` at the
        top level of a classic script. A top-level `let` goes into the global
        *lexical* environment -- shared with every other classic script on the
        page, and absent from `window` -- so a file reading
        `window.activePromptTextarea` finds nothing on the one host this
        extension targets, registers nothing, and leaves every Extra Networks
        card going to the native positive prompt however many literal boxes were
        focused first.
        """
        found = run("""
            focus("mc-krea-creative-literal-positive");
            report({onWindow: typeof window.activePromptTextarea,
                    target: activePromptTextarea.txt2img
                            === fieldIn("mc-krea-creative-literal-positive")});
        """, family="lexical")

        assert found["onWindow"] == "undefined"
        assert found["target"] is True

    def test_registering_again_never_binds_a_second_time(self):
        """`onAfterUiUpdate` fires often, so this runs many times per session.

        Two focus listeners per box, both added once: one remembers the box as
        the insertion target, one sends the diagnostic line the first time
        somebody puts the caret in it. What this guards is that neither number
        grows.
        """
        found = run("""
            mc.registerPrompts();
            mc.registerPrompts();
            mc.registerPrompts();
            const field = fieldIn("mc-krea-creative-literal-positive");
            report({listeners: (field._listeners.focus || []).length});
        """)

        assert found["listeners"] == 2


# --------------------------------------------------------------------------- #
# The Negative Prompt at CFG 1
# --------------------------------------------------------------------------- #


class TestTheCfgCollapse:
    """§9, and the word that matters is *presentation*.

    Forge already decides what CFG 1 means for the negative prompt. This only
    stops the row taking up space while that is true, and the test that matters
    most is the one asserting it never touches the text.
    """

    def test_the_row_is_there_above_cfg_one(self):
        found = run("setCfg(7); report({});")

        assert found["negativeCollapsed"] is False

    def test_the_row_collapses_at_cfg_one(self):
        found = run("setCfg(1); report({});")

        assert found["negativeCollapsed"] is True

    def test_it_comes_back_when_cfg_rises(self):
        found = run("setCfg(1); setCfg(4.5); report({});")

        assert found["negativeCollapsed"] is False

    def test_hiding_it_never_clears_the_text(self):
        """The failure this rules out is the expensive one: a user drops CFG to
        1 for a distilled model, raises it again, and their negative prompt is
        gone."""
        found = run("setCfg(1); setCfg(7); report({});")

        assert found["negativeText"] == "blurry, low quality"

    def test_the_value_is_never_written_at_all(self):
        found = run("""
            const field = negative.querySelector("textarea");
            let writes = 0;
            Object.defineProperty(field, "value", {
                get() { return "blurry, low quality"; },
                set() { writes += 1; },
            });
            setCfg(1);
            setCfg(7);
            report({writes: writes});
        """)

        assert found["writes"] == 0

    def test_a_cfg_that_will_not_parse_leaves_the_row_alone(self):
        """Hiding a prompt box because a number would not parse is the wrong
        way round."""
        found = run("""
            setCfg(1);
            cfgRange.value = "";
            mc.applyCfg();
            report({});
        """)

        assert found["negativeCollapsed"] is False

    def test_a_number_field_is_read_when_the_range_is_blank(self):
        """Gradio renders both, and which one carries the value depends on how
        it was last changed."""
        found = run("""
            cfgRange.value = "";
            const number = new El("input");
            number.type = "number";
            number.value = "1";
            cfgHolder.appendChild(number);
            mc.applyCfg();
            report({});
        """)

        assert found["negativeCollapsed"] is True


# --------------------------------------------------------------------------- #
# What the file may not become
# --------------------------------------------------------------------------- #


class TestItCannotAffectAGeneration:
    """The property every browser file in this extension is held to.

    This one moves somebody else's DOM around, which is only acceptable while
    nothing depends on it having worked.
    """

    def source(self) -> str:
        return SCRIPT.read_text(encoding="utf-8")

    def code(self) -> str:
        """The file with its comments removed -- it names what it avoids."""
        import re

        return "\n".join(re.sub(r"//.*$", "", line)
                         for line in self.source().splitlines())

    def test_it_never_names_the_generate_button(self):
        assert "txt2img_generate" not in self.code()

    def test_it_arms_no_timer(self):
        code = self.code()

        assert "setInterval" not in code
        assert "setTimeout" not in code

    def test_it_writes_into_no_prompt_box(self):
        """The one place text is inserted is the host's own Extra Networks
        handler, reached by saying which textarea is active and letting it do
        the rest."""
        code = self.code()

        assert ".value =" not in code
        assert "updateInput" not in code

    def test_it_never_reads_a_payload_for_anything_but_existence(self):
        """Literal payloads stay opaque. A file that knew what a LoRA tag was
        would have an opinion about every extension the user has installed."""
        code = self.code().casefold()

        for syntax in ("lora", "wildcard", "$style", "extra_network", "__"):
            assert syntax not in code, syntax


# --------------------------------------------------------------------------- #
# Hidden does not mean inactive
# --------------------------------------------------------------------------- #


class TestTheActiveLiteralsNote:
    """§3.3, and the price of persisting the two boxes.

    Their values keep reaching every generation while the row is off screen,
    which is exactly the invisible active state this extension keeps warning
    itself about. So the Image Pipeline says how many are in effect -- and says
    it only then, because while the boxes are visible they speak for themselves
    and a count beside them would be furniture.

    It used to share a line with an echo of the prompt, on a Prompt row that no
    longer exists. What is left is the note alone, and *empty* the rest of the
    time: an empty note is an empty element, and the stylesheet gives an empty
    element no height.
    """

    def test_nothing_is_said_while_the_row_is_on_screen(self):
        found = run_pipeline("""
            fieldIn("mc-krea-creative-literal-positive").value = "<lora:x:1>";
            report({note: mc.literalNote(), echo: echo.textContent});
        """)

        assert found["note"] == ""

    def test_a_hidden_row_with_one_value_says_one(self):
        found = run_pipeline("""
            row.offsetParent = null;
            fieldIn("mc-krea-creative-literal-positive").value = "<lora:x:1>";
            report({note: mc.literalNote()});
        """)

        assert found["note"] == "1 literal active"

    def test_a_hidden_row_with_both_values_says_two(self):
        found = run_pipeline("""
            row.offsetParent = null;
            fieldIn("mc-krea-creative-literal-positive").value = "<lora:x:1>";
            fieldIn("mc-krea-creative-literal-negative").value = "grain";
            report({note: mc.literalNote()});
        """)

        assert found["note"] == "2 literals active"

    def test_a_hidden_row_with_nothing_in_it_says_nothing(self):
        """The row is hidden for most of most sessions. A note that appeared
        whenever it was would be a permanent fixture saying nothing."""
        found = run_pipeline("""
            row.offsetParent = null;
            report({note: mc.literalNote()});
        """)

        assert found["note"] == ""

    def test_whitespace_is_not_a_literal(self):
        found = run_pipeline("""
            row.offsetParent = null;
            // Built rather than written, so the escape survives the trip from
            // a Python string into node without becoming a real newline.
            fieldIn("mc-krea-creative-literal-positive").value =
                "  " + String.fromCharCode(10) + " ";
            report({note: mc.literalNote()});
        """)

        assert found["note"] == ""

    def test_the_note_reaches_the_panel(self):
        found = run_pipeline("""
            row.offsetParent = null;
            fieldIn("txt2img_prompt").value = "a quiet street";
            fieldIn("mc-krea-creative-literal-positive").value = "<lora:x:1>";
            mc.echo();
            report({echo: echo.textContent});
        """)

        assert found["echo"] == "1 literal active"

    def test_the_prompt_itself_is_not_echoed_any_more(self):
        """The prompt box is directly above the panel. Repeating the first 120
        characters of it underneath was a row of screen space spent on
        something the user could already read."""
        found = run_pipeline("""
            row.offsetParent = null;
            fieldIn("txt2img_prompt").value = "a quiet street";
            mc.echo();
            report({echo: echo.textContent});
        """)

        assert found["echo"] == ""

    def test_the_note_still_works_with_no_literals_on_the_page(self):
        """The two files are loaded independently and either may be absent from
        a page a theme rebuilt."""
        found = run_pipeline("""
            row.parentNode.removeChild(row);
            fieldIn("txt2img_prompt").value = "a quiet street";
            mc.echo();
            report({echo: echo.textContent});
        """)

        assert found["echo"] == ""


# --------------------------------------------------------------------------- #
# The ids the two sides agree on
# --------------------------------------------------------------------------- #


class TestTheIdContract:
    """Python names these elements; JavaScript finds them by name.

    Nothing enforces that at run time and nothing complains when it breaks: a
    renamed id makes `byId` return null, every guard in the browser file does
    what it is supposed to do with a missing element, and the feature quietly
    stops moving the row, joining the family or collapsing anything. Silence is
    the whole problem, so the agreement is a test.
    """

    def ids_in(self, path: Path) -> set[str]:
        import re

        return set(re.findall(r'"(mc-krea-[a-z-]+|txt2img_[a-z_]+|mc-pipeline-[a-z-]+)"',
                              path.read_text(encoding="utf-8")))

    def python_ids(self) -> set[str]:
        import model_chain_krea_creative as creative_script

        return {
            creative_script.ident("literal", "row"),
            creative_script.ident("literal", "positive"),
            creative_script.ident("literal", "negative"),
            creative_script._spatial_id("literal-prefix"),
            creative_script._spatial_id("literal-suffix"),
        }

    def test_the_global_row_and_boxes_are_named_the_same_on_both_sides(self, host):
        wanted = self.python_ids()
        found = self.ids_in(SCRIPT)

        for name in ("mc-krea-creative-literal-row",
                     "mc-krea-creative-literal-positive",
                     "mc-krea-creative-literal-negative"):
            assert name in wanted, f"Python no longer builds {name}"
            assert name in found, f"the browser file no longer looks for {name}"

    def test_the_pipeline_file_looks_for_the_same_boxes(self, host):
        found = self.ids_in(PIPELINE)

        for name in ("mc-krea-creative-literal-row",
                     "mc-krea-creative-literal-positive",
                     "mc-krea-creative-literal-negative"):
            assert name in found, f"the pipeline file no longer looks for {name}"

    def test_the_region_fields_are_named_the_same_on_both_sides(self, host):
        wanted = self.python_ids()
        editor = ROOT / "javascript" / "model_chain_spatial_krea.js"
        source = editor.read_text(encoding="utf-8")

        for name in ("mc-krea-spatial-literal-prefix",
                     "mc-krea-spatial-literal-suffix"):
            assert name in wanted, f"Python no longer builds {name}"
            # The editor composes its ids from its prefix constant, so what
            # appears in the file is the tail. Asserted in that form rather than
            # with a fallback, because a fallback that also accepted the whole
            # id would pass whichever way the file was written and catch
            # neither rename.
            tail = 'P + "' + name[len("mc-krea-spatial"):] + '"'
            assert tail in source, f"the editor no longer looks for {name}"

    def test_the_region_inputs_are_actually_in_the_markup(self, host):
        import model_chain_krea_creative as creative_script

        markup = creative_script.spatial_editor()

        assert 'id="mc-krea-spatial-literal-prefix"' in markup
        assert 'id="mc-krea-spatial-literal-suffix"' in markup

    def test_the_native_controls_it_reaches_for_are_named(self):
        """Three of Forge's own, and the only three. Each is optional at run
        time -- a build that renames one costs that job and nothing else -- but
        a typo here would cost it silently on every build."""
        found = self.ids_in(SCRIPT)

        assert {"txt2img_prompt", "txt2img_neg_prompt", "txt2img_cfg_scale"} <= found


# --------------------------------------------------------------------------- #
# Tag Autocomplete
# --------------------------------------------------------------------------- #


class TestTagAutocomplete:
    """§8, offered rather than assumed.

    These two carry the host's `prompt` class, which is the selector that
    extension matches on -- but its list is walked once during its own setup,
    and whether these boxes were in the document by then is an ordering neither
    extension controls. So they are offered again through the same per-textarea
    entry point it uses for its own late arrivals, and everything about that is
    conditional: no Tag Autocomplete, an older one, or one whose config has not
    loaded yet is a page where this does nothing.
    """

    def test_both_boxes_are_offered_to_it(self):
        found = run("""
            globalThis.TAC_CFG = {activeIn: {thirdParty: true}};
            globalThis.claimed = [];
            globalThis.addAutocompleteToArea = (area) => {
                if (area.classList.contains("autocomplete")) return;
                area.classList.add("autocomplete");
                claimed.push(area.parentNode.id);
            };
            mc.registerPrompts();
            report({claimed: claimed});
        """)

        assert found["claimed"] == ["mc-krea-creative-literal-positive",
                                    "mc-krea-creative-literal-negative"]

    def test_the_third_party_switch_being_off_is_not_the_end_of_it(self):
        """The one thing that was actually standing in the way, from a user's
        log: `[config loaded, third-party boxes no, list extended True, boxes in
        the list True]` -- everything on this side done, and that extension
        refusing the boxes because its "Active in third party textboxes" switch
        gates every textarea it does not recognise as one of the four core
        prompt boxes.

        It is read in exactly one place, the gate inside the call below. So it
        is lifted for the length of that call and put back.
        """
        found = run("""
            globalThis.TAC_CFG = {activeIn: {thirdParty: false}};
            globalThis.addAutocompleteToArea = (area) => {
                // The gate, verbatim in shape: it refuses anything it does not
                // recognise while the switch is off.
                if (!TAC_CFG.activeIn.thirdParty) return;
                area.classList.add("autocomplete");
            };
            mc.registerPrompts();
            report({claimed: mc.claimed(),
                    switchAfter: TAC_CFG.activeIn.thirdParty,
                    said: mc.report().liftedThirdParty});
        """)

        assert found["claimed"] is True
        assert found["switchAfter"] is False
        assert found["said"] is True

    def test_a_switch_that_was_already_on_is_never_written_to(self):
        """Nothing to lift, nothing to restore, and nothing to say about it."""
        found = run("""
            globalThis.writes = 0;
            const activeIn = {};
            Object.defineProperty(activeIn, "thirdParty", {
                get() { return true; },
                set() { writes += 1; },
            });
            globalThis.TAC_CFG = {activeIn: activeIn};
            globalThis.addAutocompleteToArea = (area) => {
                area.classList.add("autocomplete");
            };
            mc.registerPrompts();
            report({writes: writes, said: mc.report().liftedThirdParty});
        """)

        assert found["writes"] == 0
        assert found["said"] is False

    def test_an_option_that_does_not_exist_is_put_back_as_it_was(self):
        """A build of that extension without the setting at all. `undefined`
        goes back, not `false`: putting it back means putting it back."""
        found = run("""
            globalThis.TAC_CFG = {activeIn: {}};
            globalThis.addAutocompleteToArea = (area) => {
                area.classList.add("autocomplete");
            };
            mc.registerPrompts();
            report({present: "thirdParty" in TAC_CFG.activeIn,
                    value: TAC_CFG.activeIn.thirdParty === undefined});
        """)

        assert found["present"] is True
        assert found["value"] is True

    def test_a_call_that_throws_still_puts_the_switch_back(self):
        """Their function, their exception -- but not their setting left
        changed because of it."""
        found = run("""
            globalThis.TAC_CFG = {activeIn: {thirdParty: false}};
            globalThis.addAutocompleteToArea = () => { throw new Error("no"); };
            let raised = false;
            try { mc.registerPrompts(); } catch (error) { raised = true; }
            report({switchAfter: TAC_CFG.activeIn.thirdParty, raised: raised});
        """)

        assert found["switchAfter"] is False

    def test_a_box_it_already_claimed_is_not_offered_again(self):
        """It runs on every UI update, and that extension's own setup may go
        first. Handing it a textarea it already owns has to be a no-op."""
        found = run("""
            globalThis.TAC_CFG = {activeIn: {}};
            globalThis.calls = 0;
            globalThis.addAutocompleteToArea = (area) => {
                calls += 1;
                area.classList.add("autocomplete");
            };
            mc.registerPrompts();
            mc.registerPrompts();
            mc.registerPrompts();
            report({calls: calls});
        """)

        assert found["calls"] == 2

    def test_its_own_list_comes_back_with_both_boxes_in_it(self):
        """The half that makes the ordering stop mattering.

        That extension decides what to attach to by calling its own
        `getTextAreas()`, once, inside a setup that runs when its files have
        loaded. Handing it two textareas only works if this file gets a turn
        after that; extending the list it asks for works whichever goes first.
        """
        found = run("""
            globalThis.TAC_CFG = {activeIn: {}};
            globalThis.addAutocompleteToArea = (area) => {
                area.classList.add("autocomplete");
            };
            globalThis.getTextAreas = () => [fieldIn("txt2img_prompt")];
            mc.extendAutocompleteList();
            const list = getTextAreas();
            report({theirs: list[0] === fieldIn("txt2img_prompt"),
                    ours: list.slice(1).map((field) => field.parentNode.id),
                    length: list.length});
        """)

        assert found["theirs"] is True
        assert found["ours"] == ["mc-krea-creative-literal-positive",
                                 "mc-krea-creative-literal-negative"]

    def test_the_list_is_wrapped_once_however_often_this_runs(self):
        found = run("""
            globalThis.TAC_CFG = {activeIn: {}};
            globalThis.addAutocompleteToArea = () => {};
            globalThis.calls = 0;
            globalThis.getTextAreas = () => { calls += 1; return []; };
            mc.extendAutocompleteList();
            mc.extendAutocompleteList();
            mc.extendAutocompleteList();
            const list = getTextAreas();
            report({calls: calls, length: list.length});
        """)

        assert found["calls"] == 1
        assert found["length"] == 2

    def test_a_list_that_is_not_an_array_is_handed_back_untouched(self):
        """Their function, their answer. This adds to a list; it does not
        decide what one is."""
        found = run("""
            globalThis.TAC_CFG = {activeIn: {}};
            globalThis.addAutocompleteToArea = () => {};
            globalThis.getTextAreas = () => "not a list";
            mc.extendAutocompleteList();
            report({answer: getTextAreas()});
        """)

        assert found["answer"] == "not a list"

    def test_a_page_without_it_is_a_page_where_nothing_happens(self):
        found = run("""
            globalThis.getTextAreas = () => ["untouched"];
            mc.registerPrompts();
            report({claimed: fieldIn("mc-krea-creative-literal-positive")
                             ._classes.has("autocomplete"),
                    list: getTextAreas()});
        """)

        assert found["claimed"] is False
        assert found["list"] == ["untouched"]

    def test_a_config_that_has_not_loaded_yet_is_left_alone(self):
        """`var TAC_CFG = null` until its files are read. Calling in before
        that is calling into a half-built extension; the next UI update tries
        again, and there is always another UI update."""
        found = run("""
            globalThis.TAC_CFG = null;
            globalThis.calls = 0;
            globalThis.addAutocompleteToArea = () => { calls += 1; };
            mc.registerPrompts();
            report({calls: calls});
        """)

        assert found["calls"] == 0


# --------------------------------------------------------------------------- #
# The page around the row
# --------------------------------------------------------------------------- #


class TestItLeavesTheHostsLayoutAlone:
    """The empty space under this row belongs to Forge.

    `ui_toprow.create_prompts()` builds `gr.Column(scale=6)`, and in the Compact
    prompt layout that column claims the height the gallery beside it has -- so
    it stands hundreds of pixels taller than the prompts in it, with or without
    this extension. A version of this file wrote `flex-grow: 0` onto that column
    to take the space back. It worked, and it was still this component reaching
    out and changing how somebody else's page lays out, which is not a thing it
    gets to decide. These tests are the reason it does not come back.
    """

    SETUP = """
        const container = make("div", "txt2img_prompt_container");
        container.appendChild(prompt);
        container.appendChild(negative);
        container.style.flexGrow = "6";
        container.parentNode.style.flexDirection = "column";
    """

    def test_the_prompt_column_is_not_written_to(self):
        found = run(self.SETUP + """
            mc.wire();
            report({grow: container.style.flexGrow,
                    touched: Object.keys(container.dataset).length});
        """)

        assert found["grow"] == "6"
        assert found["touched"] == 0

    def test_the_only_element_of_the_hosts_it_touches_is_the_negative_prompt(self):
        """One class, put on and taken off again by §9's collapse. Everything
        else on the page is read and left alone."""
        found = run(self.SETUP + """
            const before = JSON.stringify([prompt, negative, container, toprow]
                .map((el) => [el.className, JSON.stringify(el.style)]));
            mc.wire();
            setCfg(7);
            const after = JSON.stringify([prompt, negative, container, toprow]
                .map((el) => [el.className, JSON.stringify(el.style)]));
            report({unchanged: before === after});
        """)

        assert found["unchanged"] is True


# --------------------------------------------------------------------------- #
# The line in the log
# --------------------------------------------------------------------------- #


class TestTheReport:
    """Everything above depends on two other extensions, and every fact about
    whether it worked lives in the browser.

    Twice now the answer to "tag completion does not work in these boxes" has
    been a console snippet somebody has to open the developer tools to paste.
    So the page says it instead, once, on the first focus of a literal box --
    which is exactly when tag completion is the thing being expected -- into the
    log every other message from this extension goes to.
    """

    def test_focusing_a_box_reports_once(self):
        found = run("""
            focus("mc-krea-creative-literal-positive");
            focus("mc-krea-creative-literal-negative");
            focus("mc-krea-creative-literal-positive");
            report({posts: posted.length, url: (posted[0] || {}).url});
        """)

        assert found["posts"] == 1
        assert found["url"] == "/model-chain/literal-prompts/report"

    def test_it_says_what_was_found(self):
        found = run("""
            globalThis.TAC_CFG = {activeIn: {thirdParty: true}};
            globalThis.addAutocompleteToArea = (area) => {
                area.classList.add("autocomplete");
            };
            globalThis.getTextAreas = () => [];
            mc.wire();
            focus("mc-krea-creative-literal-positive");
            report({sent: posted[0].body});
        """)

        assert found["sent"] == {
            "boxesFound": True, "claimed": True, "autocompleteInstalled": True,
            "listWrapped": True, "inTheirList": True, "config": "loaded",
            "thirdPartyBoxes": True, "liftedThirdParty": False,
            "promptFamily": True, "placed": True}

    def test_it_says_when_tag_autocomplete_is_not_there(self):
        found = run("""
            focus("mc-krea-creative-literal-positive");
            report({sent: posted[0].body});
        """, family=False)

        assert found["sent"]["autocompleteInstalled"] is False
        assert found["sent"]["claimed"] is False
        assert found["sent"]["config"] == "missing"
        assert found["sent"]["promptFamily"] is False

    def test_it_carries_no_text_from_any_prompt_box(self):
        """The whole payload is booleans and one word out of three. A
        diagnostic that could carry what somebody typed would be a diagnostic
        nobody should install -- and this one posts to the extension's own
        route, so "could" is the only word that matters."""
        found = run("""
            fieldIn("mc-krea-creative-literal-positive").value = "a secret";
            fieldIn("txt2img_prompt").value = "another secret";
            focus("mc-krea-creative-literal-positive");
            report({sent: JSON.stringify(posted[0].body)});
        """)

        assert "secret" not in found["sent"]
        for value in json.loads(found["sent"]).values():
            assert isinstance(value, bool) or value in ("missing", "null", "loaded") \
                or value is None

    def test_a_build_without_fetch_is_not_a_crash(self):
        found = run("""
            globalThis.fetch = undefined;
            focus("mc-krea-creative-literal-positive");
            report({ok: true});
        """)

        assert found["ok"] is True

    def test_a_route_that_is_not_there_is_not_a_crash(self):
        """It is a log line, not a feature."""
        found = run("""
            globalThis.fetch = () => { throw new Error("404"); };
            focus("mc-krea-creative-literal-positive");
            report({ok: true});
        """)

        assert found["ok"] is True
