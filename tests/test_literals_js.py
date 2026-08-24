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


def run(script: str, family: bool = True) -> dict:
    setup = ("globalThis.activePromptTextarea = {};" if family
             else "// no activePromptTextarea on this build")
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

    def test_registering_twice_adds_one_listener(self):
        found = run("""
            mc.registerPrompts();
            mc.registerPrompts();
            const field = fieldIn("mc-krea-creative-literal-positive");
            report({listeners: (field._listeners.focus || []).length});
        """)

        assert found["listeners"] == 1


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
    itself about. So the Prompt row of the Image Pipeline says how many are in
    effect -- and says it only then, because while the boxes are visible they
    speak for themselves and a count beside them would be furniture.
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

    def test_the_note_reaches_the_prompt_row(self):
        found = run_pipeline("""
            row.offsetParent = null;
            fieldIn("txt2img_prompt").value = "a quiet street";
            fieldIn("mc-krea-creative-literal-positive").value = "<lora:x:1>";
            mc.echo();
            report({echo: echo.textContent});
        """)

        assert found["echo"] == "a quiet street · 1 literal active"

    def test_the_prompt_row_still_works_with_no_literals_on_the_page(self):
        """The two files are loaded independently and either may be absent from
        a page a theme rebuilt."""
        found = run_pipeline("""
            row.parentNode.removeChild(row);
            fieldIn("txt2img_prompt").value = "a quiet street";
            mc.echo();
            report({echo: echo.textContent});
        """)

        assert found["echo"] == "a quiet street"


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
