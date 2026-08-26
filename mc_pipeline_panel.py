"""The Image Pipeline: one surface for the whole txt2img journey.

    IMAGE PIPELINE

    Creative                         ON
      C7 - 2 directions - Editorial
      |
    Spatial                          ON
      Smart - 2 regions - Studio thirds
      |
      1536x2304 pixel handoff
    Stage 2                          ON
      Klein 9B - Portrait polish - denoise .35

Three panels became one path. Creative Mode, Spatial Layout and the Stage 2
accordion were three independent top-level surfaces on txt2img, each one
truthful about itself and none of them saying what the others were going to do
to the same picture. A user could read all three and still not know what the
next press of Generate would produce, because the one thing none of them
described was the *order*.

What this module is
-------------------
A presentation shell and nothing else. It owns no setting, reads no
preference, and runs no part of a generation. It builds three rows and hands out
the empty containers -- two per row -- that the feature scripts fill with their
own existing controls. Every control on the finished panel is the same Gradio
component, wired to the same handler, writing to the same state as before this
file existed. The pipeline is where they are drawn, not what they do.

Why the shell is shared rather than owned
-----------------------------------------
Creative and Spatial live in ``scripts/model_chain_krea_creative.py``; Stage 2
lives in ``scripts/model_chain.py``. Forge builds each ``alwayson`` script's
``ui()`` inside a wrapper of its own, in an order this extension does not
choose, so neither script can contain the other.

So neither does. Whichever script's ``ui()`` runs first calls :func:`host`,
which builds the whole shell -- all three rows, in pipeline order, with empty
slots -- and remembers it against the Blocks currently being assembled. The
second script calls :func:`host`, gets the same object back, and fills its
slots by re-entering them:

    with pipeline.body("creative"):
        ...the Creative editor, built exactly as it always was...

Gradio appends to a container that is re-entered, and renders each container's
children in the order they were added. The slots are created in pipeline order
at build time, so the finished panel reads Creative -> Spatial -> Stage 2
whichever script got there first. Nothing here
depends on script order, which is the property that makes it safe: a host that
sorts its scripts differently tomorrow rearranges nothing.

Only what this extension owns
-----------------------------
Three rows, and they are the three stages Model Chain runs. The panel used to
draw Prompt, Stage 1 and Output as well -- Forge's own, muted and uneditable, so
that the path had no holes in it. What that produced in practice was three rows
restating things the page already said louder: the prompt box is directly above,
the size sliders are directly below, and the output is the picture. They are
gone.

One number survives them, because nothing else on the page states it: the pixel
size that crosses into Stage 2. Stage 1's Hires pass changes it, so a user who
has read the width and height sliders has read the wrong number -- see
:func:`handoff_note`, drawn on the edge above the Stage 2 row and true whether
Stage 2 is armed or not.

Section 2.5 still holds for everything else: no duplicate source-of-truth,
because two controls holding one value is a bug with a delay on it.

Everything is a stock component
-------------------------------
Accordion, Row, Column, Group, Checkbox, Markdown. No custom-HTML control, no
Gradio-generated class in any selector, and an extension-owned id on every
element. That is what lets a theme extension -- Lobe, or any other -- restyle
this panel along with the rest of the page instead of around it.
"""

from __future__ import annotations

import logging

import gradio as gr

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

PREFIX = "mc-pipeline"
"""Element-id and class stem for everything this module puts in the page."""

OWNED = ("creative", "spatial", "stage2")
"""The stages Model Chain controls: switchable, expandable, emphasised."""

ORDER = OWNED
"""Top to bottom, and the order a generation actually runs in.

The one ordering in the extension that a user can see. It matches
:mod:`mc_krea_pipeline`'s, which is the ordering that actually happens, and the
two are meant to be compared: a panel that drew Spatial above Creative while the
code ran them the other way round would be a diagram of a different program.

It used to run Prompt -> Creative -> Spatial -> Stage 1 -> Stage 2 -> Output.
Those three extra rows were Forge's own, drawn muted and uneditable so that the
path had no holes in it -- and a path with no holes in it turned out to be three
rows of things somebody already knew, above the prompt box that already says the
prompt and beside the sliders that already say the size. The panel is the three
stages this extension actually owns; what crosses the edge into Stage 2 is still
said, because that is a number nothing else on the page states.
"""

PHASES_WITHOUT_A_ROW = ("stage1", "output")
"""Phases a generation still runs through, and the panel no longer draws.

The progress bar's labels are the host's and did not change when three rows
left: "Stage 1", "Finishing" and the rest still arrive, and
``javascript/model_chain_pipeline.js`` still recognises every one of them. What
it does with the two named here is light nothing, which is the answer a phase
with no row on the panel should get -- and a very different thing from a label
this extension has never heard of, which is the case that would mean the table
had gone stale.
"""

TITLES = {
    "creative": "Creative",
    "spatial": "Spatial",
    "stage2": "Stage 2",
}

CARD_HEAD = f"{PREFIX}-card-head"
"""The class the browser file puts on whichever element Gradio made the header.

Everything about the card's header layout is keyed off this rather than off
Gradio's own class for it. That is the whole of rule 2 below: a theme is allowed
to rebuild Gradio's internals, and when it does the browser file either still
recognises the header -- in which case this class lands on it and the layout
holds -- or it does not, in which case no rule matches and the card falls back
to a plain stack that still works.
"""

CARDED = f"{PREFIX}-carded"
"""On the stage column, once its header has been found and furnished.

Two classes rather than one because they answer different questions. The header
needs to know it is a header; the card needs to know whether the move happened
at all, so that the fallback layout is a stylesheet state rather than a guess.
"""

PROMPT_ECHO = f"{PREFIX}-prompt-echo"
"""The one element the browser writes the hidden-but-active note into.

An id of this extension's own, inside a stock ``gr.HTML``, because what it says
follows a keystroke and a Gradio round trip per keystroke is not a thing to do
to somebody's typing. Writing into a Markdown component would mean reaching for
a Gradio-generated class, which is the one thing every selector in this
extension avoids -- a theme is allowed to rearrange Gradio's internals, and this
has to keep working when it does.

It carried an echo of the prompt as well, on a Prompt row that no longer exists.
What is left is section 3.3 and only that: ``2 literals active``, said when the
Literal Prompt boxes are off screen and carrying something, and *empty* every
other second of the day -- ``:empty`` in the stylesheet gives an empty note no
height at all, so the line costs nothing until there is something to say.
"""

PLACEHOLDERS = {
    "creative": "Bypassed — the prompt is expanded as written.",
    "spatial": "Bypassed — nothing is composed onto the scene.",
    "stage2": "Bypassed — the Stage 1 image is the final image.",
}
"""The second line before anything has been wired. Replaced on first render.

"Bypassed" and not "Off", and the same word every stage that is switched off
uses. §1 of the intent: a bypassed card stays visible and configurable, so the
summary is the only thing standing between "this setting exists" and "this
setting will run", and it has to say which without being read twice.
"""


def ident(*parts) -> str:
    """One element id, from the pieces that make it unique."""
    return "-".join((PREFIX,) + tuple(str(part) for part in parts if part))


def classes(*names: str) -> list[str]:
    """Class names for one element, all under this module's stem."""
    return [f"{PREFIX}-{name}" for name in names]


DRAWER = f"{PREFIX}-drawer"
"""The class on every disclosure this extension builds.

One class, three jobs, and all of them belong to whoever is looking at the page
rather than to whoever wrote the feature behind it: the stylesheet gives every
drawer the same outline and the same header, the browser file remembers which
ones were opened, and a test can count them. A feature that builds its own bare
``gr.Accordion`` gets none of that, which is why :func:`drawer` exists and why
there is a test that says so.
"""


def drawer(label: str, *, elem_id=None, elem_classes=None, **kwargs):
    """One disclosure, closed, and marked as belonging to this extension.

    ``open`` is not a parameter. Every drawer starts closed -- a tab that
    unfolds four levels of settings the moment it is drawn is a tab nobody can
    see the top of -- and what somebody opens is remembered per browser by
    ``javascript/model_chain_pipeline.js``, so the default is a first-visit
    answer rather than a decision imposed on every visit after it.
    """
    found = [DRAWER]
    if elem_classes:
        found.extend(elem_classes if isinstance(elem_classes, (list, tuple))
                     else [elem_classes])
    return gr.Accordion(label=label, open=False, elem_id=elem_id,
                        elem_classes=found, **kwargs)


def handoff_note(width: int = 0, height: int = 0) -> str:
    """The line on the connector between Stage 1 and Stage 2.

    The single most important number on the panel and the one that used to be
    nowhere: Stage 2 refines *finished Stage 1 pixels*, so a Hires pass changes
    what Stage 2 is handed, and a user who has read only the width and height
    sliders has read the wrong number.
    """
    if width > 0 and height > 0:
        return f"{width} × {height} pixel handoff"
    return "pixel handoff"


class Pipeline:
    """The three rows, and the empty containers the feature scripts fill.

    Built once per Gradio Blocks by :func:`host`. A script asks for the two
    slots belonging to a stage it owns, re-enters them, and builds whatever it
    was building before -- unchanged, and in most cases moved rather than
    rewritten.
    """

    def __init__(self):
        self.accordion = None
        self.heads: dict = {}
        self.bodies: dict = {}
        self.editors: dict = {}
        self.summaries: dict = {}
        self.rows: dict = {}
        self.handoff = None
        self.filled: set = set()

    # -- what a feature script asks for ------------------------------------ #

    def head(self, stage: str):
        """The container beside a stage's title, for its ON/OFF switch.

        Section 3.3: a stage is switched from the collapsed row, so the switch
        cannot live inside the drawer it would otherwise be needed to open.
        """
        return self._slot(self.heads, stage, "head")

    def body(self, stage: str):
        """The container inside a stage's disclosure, for its editor."""
        self.filled.add(stage)
        return self._slot(self.bodies, stage, "body")

    def summary(self, stage: str):
        """The live second line under a stage's title, or ``None``.

        Owned by this module and written by the feature that knows what it
        should say, which is how one row can describe a feature this file knows
        nothing about.
        """
        return self.summaries.get(stage)

    def _slot(self, where: dict, stage: str, kind: str):
        slot = where.get(stage)
        if slot is None:
            # A stage nobody built a slot for. Returning a throwaway container
            # keeps the caller's `with` block legal and puts its controls
            # wherever Gradio currently is, which is visible and wrong rather
            # than a traceback in the middle of building the tab.
            logger.warning("Model Chain: the Image Pipeline has no %s slot for %r",
                           kind, stage)
            return gr.Column(visible=True)
        return slot


# --------------------------------------------------------------------------- #
# Building it
# --------------------------------------------------------------------------- #


def _row(pipeline: Pipeline, stage: str) -> None:
    """One stage card: one disclosure, one switch, one live summary, one body.

    The card used to be four things stacked: a title row, a switch, a summary
    line, and *then* a second accordion, labelled with a restatement of the
    stage's own name, that had to be opened to reach the settings. Two rows
    saying the same word, one above the other, and only the lower one opening
    anything.

    So there is one disclosure now and it is the stage's own name. The switch
    and the summary are built as siblings of it and moved into its header by
    ``javascript/model_chain_pipeline.js``, because Gradio gives an Accordion a
    plain string for a label and neither a live line nor a checkbox can be built
    into one. The move is presentation only and is allowed to fail: if the
    browser file never runs, or a Gradio version renders a header this file
    cannot recognise, the card reads name / summary / switch stacked in ordinary
    flow -- which is the panel as it was, and every control still works.
    """
    if stage == "stage2":
        # The edge into Stage 2 says what crosses it: the pixel size Stage 2 is
        # handed, which is Stage 1's size *after* any Hires pass and so not the
        # number the width and height sliders show. Its own element and not
        # Stage 2's summary line, because it is equally true when Stage 2 is
        # off -- where it says what Stage 2 would have been given.
        pipeline.handoff = gr.Markdown(
            handoff_note(), elem_id=ident("handoff"),
            elem_classes=classes("handoff"))

    with gr.Column(elem_id=ident("stage", stage),
                   elem_classes=classes("stage", "stage-owned", f"stage-{stage}")) as row:
        pipeline.rows[stage] = row

        # Closed. Every drawer this extension builds opens on a press and not
        # before -- a tab that unfolds three stages of settings the moment it is
        # drawn is a tab nobody can see the top of.
        with drawer(TITLES[stage], elem_id=ident("editor", stage),
                    elem_classes=classes("editor")) as editor:
            pipeline.editors[stage] = editor
            with gr.Column(elem_id=ident("body", stage),
                           elem_classes=classes("body")) as body:
                pipeline.bodies[stage] = body

        # Two separate elements and not one row holding both, because they are
        # moved into two different places in the header: the summary under the
        # name, the switch out at the right where §2 of the intent asks for it
        # to be spatially separated from the surface that opens the card.
        pipeline.summaries[stage] = gr.Markdown(
            PLACEHOLDERS.get(stage, ""), elem_id=ident("summary", stage),
            elem_classes=classes("summary"))

        # Filled by the feature that owns the stage, with the switch it already
        # had. Nothing is created here: a second checkbox mirroring the real one
        # is exactly the duplicate source-of-truth section 2.5 forbids, and it
        # would be the one the user reached for first.
        with gr.Column(min_width=90, scale=0,
                       elem_id=ident("switch", stage),
                       elem_classes=classes("switch")) as head:
            pipeline.heads[stage] = head


def _build() -> Pipeline:
    """Assemble the whole shell into whatever container is currently open."""
    pipeline = Pipeline()

    # Closed, like everything else. This is one accordion among however many
    # else a user has on txt2img, and the extension does not get to decide that
    # its own is the one worth the whole screen on arrival. What is opened is
    # remembered -- see javascript/model_chain_pipeline.js -- so the second
    # visit to the tab looks like the first one was left.
    with drawer("Image Pipeline", elem_id=ident("panel"),
                elem_classes=classes("panel")) as accordion:
        pipeline.accordion = accordion

        # Section 3.3, and the only thing left of the Prompt row: what is in
        # effect while its own control is off screen. Empty the rest of the
        # time, and an empty note has no height -- see `.mc-pipeline-notes` in
        # style.css.
        gr.HTML(f'<span id="{PROMPT_ECHO}" class="{PREFIX}-echo"></span>',
                elem_id=ident("notes"), elem_classes=classes("notes"))

        for stage in ORDER:
            _row(pipeline, stage)

    return pipeline


# --------------------------------------------------------------------------- #
# One shell per page
# --------------------------------------------------------------------------- #

_BUILT: tuple = ()
"""``(build key, Pipeline)`` for the page currently being assembled."""


def _key():
    """Something that identifies this Blocks build, or ``None``.

    ``Context.root_block`` is the Blocks Gradio is assembling right now, which
    is exactly the scope one shell should live for: Forge rebuilds the whole
    tab on a UI reload, and a shell remembered across that would hand the new
    page slots belonging to a page that no longer exists.
    """
    try:
        from gradio.context import Context

        return id(Context.root_block) if Context.root_block is not None else None
    except Exception:
        return None


def host() -> Pipeline:
    """The Image Pipeline shell for this page, building it on first ask.

    Both feature scripts call this. The first one builds; the second gets the
    same object and fills the slots the first left empty. Which is which is not
    decided here and does not need to be.
    """
    global _BUILT

    key = _key()
    if _BUILT and (key is None or _BUILT[0] == key):
        # An unknown key reuses rather than rebuilds. The two ways to be wrong
        # here are not equal: reusing a stale shell puts the second script's
        # controls into a page nobody is looking at, while rebuilding puts a
        # second Image Pipeline on the page the user *is* looking at, with half
        # the stages in each. Tests that build a page more than once call
        # forget() between them, which is the same thing said explicitly.
        return _BUILT[1]

    pipeline = _build()
    _BUILT = (key, pipeline)
    return pipeline


def forget() -> None:
    """Drop the remembered shell. For tests, and for a host that rebuilds."""
    global _BUILT
    _BUILT = ()
