"""The Image Pipeline: one surface for the whole txt2img journey.

    IMAGE PIPELINE

    Prompt
      "astronaut botanist in a Martian greenhouse"
      |
    Creative                         ON
      C7 - 2 directions - Editorial
      |
    Spatial                          ON
      Smart - 2 regions - Studio thirds
      |
    Stage 1
      Krea 2 - 1024x1536 - Hires 1.5x -> 1536x2304
      ImageStitch - 2 references
      | 1536x2304 pixel handoff
    Stage 2                          ON
      Klein 9B - Portrait polish - denoise .35
      |
    Output
      1536x2304

Three panels became one path. Creative Mode, Spatial Layout and the Stage 2
accordion were three independent top-level surfaces on txt2img, each one
truthful about itself and none of them saying what the others were going to do
to the same picture. A user could read all three and still not know what the
next press of Generate would produce, because the one thing none of them
described was the *order*.

What this module is
-------------------
A presentation shell and nothing else. It owns no setting, reads no
preference, and runs no part of a generation. It builds six rows and hands out
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
which builds the whole shell -- all six rows, in pipeline order, with empty
slots -- and remembers it against the Blocks currently being assembled. The
second script calls :func:`host`, gets the same object back, and fills its
slots by re-entering them:

    with pipeline.body("creative"):
        ...the Creative editor, built exactly as it always was...

Gradio appends to a container that is re-entered, and renders each container's
children in the order they were added. The slots are created in pipeline order
at build time, so the finished panel reads Prompt -> Creative -> Spatial ->
Stage 1 -> Stage 2 -> Output whichever script got there first. Nothing here
depends on script order, which is the property that makes it safe: a host that
sorts its scripts differently tomorrow rearranges nothing.

Ownership is a visual fact, not a new rule
------------------------------------------
Creative, Spatial and Stage 2 are this extension's. Prompt, Stage 1 and Output
belong to Forge, and appear here only because a path with holes in it does not
teach anybody the path. They are drawn muted, they carry no switch, and they
open nothing -- section 2.4 of the design intent, expressed as three CSS
classes rather than as three disabled controls, because a control that is
present and refuses to work is worse than no control.

Stage 1 in particular is *read*. Forge's own width, height, Hires and
checkpoint controls stay the only place those values can be changed, and this
panel summarises what they currently say. Section 2.5: no duplicate
source-of-truth, ever, because two controls holding one value is a bug with a
delay on it.

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

CONTEXT = ("prompt", "stage1", "output")
"""The stages Forge owns. Shown so the path is whole; never editable here."""

ORDER = ("prompt", "creative", "spatial", "stage1", "stage2", "output")
"""Top to bottom, and the order a generation actually runs in.

The one ordering in the extension that a user can see. It matches
:mod:`mc_krea_pipeline`'s, which is the ordering that actually happens, and the
two are meant to be compared: a panel that drew Spatial above Creative while the
code ran them the other way round would be a diagram of a different program.
"""

TITLES = {
    "prompt": "Prompt",
    "creative": "Creative",
    "spatial": "Spatial",
    "stage1": "Stage 1",
    "stage2": "Stage 2",
    "output": "Output",
}

EDITOR_LABELS = {
    "creative": "Creative direction",
    "spatial": "Spatial layout",
    "stage2": "Stage 2 refinement",
}
"""What an owned stage's disclosure says. The stage name is already above it."""

PLACEHOLDERS = {
    "prompt": "*whatever is in the prompt box above*",
    "creative": "Off — the prompt is expanded as written.",
    "spatial": "Off — nothing is composed onto the scene.",
    "stage1": "Forge generates this stage.",
    "stage2": "Off — the Stage 1 image is the final image.",
    "output": "Stage 1 delivers the final image.",
}
"""The second line before anything has been wired. Replaced on first render."""


def ident(*parts) -> str:
    """One element id, from the pieces that make it unique."""
    return "-".join((PREFIX,) + tuple(str(part) for part in parts if part))


def classes(*names: str) -> list[str]:
    """Class names for one element, all under this module's stem."""
    return [f"{PREFIX}-{name}" for name in names]


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
    """The six rows, and the empty containers the feature scripts fill.

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
    """One pipeline item: a title, a switch slot, a live line, and a drawer.

    Owned and context stages differ in three ways and are otherwise the same
    element, which is the point -- they are steps of one path, not two kinds of
    thing that happen to be listed together.
    """
    owned = stage in OWNED
    kind = "owned" if owned else "context"

    with gr.Column(elem_id=ident("stage", stage),
                   elem_classes=classes("stage", f"stage-{kind}", f"stage-{stage}")) as row:
        pipeline.rows[stage] = row

        with gr.Row(elem_classes=classes("head")):
            gr.Markdown(f"**{TITLES[stage]}**" if owned else TITLES[stage],
                        elem_id=ident("title", stage),
                        elem_classes=classes("title"))
            if owned:
                # Filled by the feature that owns the stage, with the switch it
                # already had. Nothing is created here: a second checkbox
                # mirroring the real one is exactly the duplicate
                # source-of-truth section 2.5 forbids, and it would be the one
                # the user reached for first.
                with gr.Column(min_width=96, scale=0,
                               elem_id=ident("switch", stage),
                               elem_classes=classes("switch")) as head:
                    pipeline.heads[stage] = head

        pipeline.summaries[stage] = gr.Markdown(
            PLACEHOLDERS.get(stage, ""), elem_id=ident("summary", stage),
            elem_classes=classes("summary"))

        if owned:
            with gr.Accordion(EDITOR_LABELS[stage], open=False,
                              elem_id=ident("editor", stage),
                              elem_classes=classes("editor")) as editor:
                pipeline.editors[stage] = editor
                with gr.Column(elem_id=ident("body", stage),
                               elem_classes=classes("body")) as body:
                    pipeline.bodies[stage] = body

    if stage == "stage1":
        # The connector between Stage 1 and Stage 2 says what crosses it.
        # Written as its own element rather than into Stage 1's summary,
        # because it describes the *edge* -- it is equally true when Stage 2 is
        # off, where it says what Stage 2 would have received.
        pipeline.handoff = gr.Markdown(
            handoff_note(), elem_id=ident("handoff"),
            elem_classes=classes("handoff"))


def _build() -> Pipeline:
    """Assemble the whole shell into whatever container is currently open."""
    pipeline = Pipeline()

    with gr.Accordion("Image Pipeline", open=True, elem_id=ident("panel"),
                      elem_classes=classes("panel")) as accordion:
        pipeline.accordion = accordion
        gr.Markdown(
            "The path from your prompt to the finished image. **Creative**, "
            "**Spatial** and **Stage 2** are Model Chain's — switch them here, "
            "open one to edit it. Prompt, Stage 1 and Output are Forge's own and "
            "are shown for context.",
            elem_id=ident("intro"), elem_classes=classes("intro"))

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
