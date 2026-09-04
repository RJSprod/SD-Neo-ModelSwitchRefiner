"""The Image Pipeline: one surface for the whole txt2img journey.

    IMAGE PIPELINE

    Neutralize Prompt                OFF
      Bypassed - prompt as-is
      |
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
described was the *order*. Neutralize Prompt joined the top of the path later,
and joined it as a row, because a stage that runs first and is drawn anywhere
else would be the same hole in the diagram.

What this module is
-------------------
A presentation shell and nothing else. It owns no setting, reads no
preference, and runs no part of a generation. It builds four rows and hands out
the empty containers -- two per expandable row, one for the toggle-only row --
that the feature scripts fill with their own existing controls. Every control on
the finished panel is the same Gradio component, wired to the same handler,
writing to the same state as before this file existed. The pipeline is where
they are drawn, not what they do.

Why the shell is shared rather than owned
-----------------------------------------
Creative and Spatial live in ``scripts/model_chain_krea_creative.py``; Stage 2
lives in ``scripts/model_chain.py``. Forge builds each ``alwayson`` script's
``ui()`` inside a wrapper of its own, in an order this extension does not
choose, so neither script can contain the other.

So neither does. Whichever script's ``ui()`` runs first calls :func:`host`,
which builds the whole shell -- all four rows, in pipeline order, with empty
slots -- and remembers it against the Blocks currently being assembled. The
second script calls :func:`host`, gets the same object back, and fills its
slots by re-entering them:

    with pipeline.body("creative"):
        ...the Creative editor, built exactly as it always was...

Gradio appends to a container that is re-entered, and renders each container's
children in the order they were added. The slots are created in pipeline order
at build time, so the finished panel reads Neutralize Prompt -> Creative ->
Spatial -> Stage 2 whichever script got there first. Nothing here depends on
script order, which is the property that makes it safe: a host that sorts its
scripts differently tomorrow rearranges nothing.

Only what this extension owns
-----------------------------
Four rows, and they are the four stages Model Chain runs. The panel used to
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
Accordion, Row, Column, Group, Checkbox, Markdown, HTML. No custom-HTML
*control*, no Gradio-generated class in any selector, and an extension-owned id
on every element. That is what lets a theme extension -- Lobe, or any other --
restyle this panel along with the rest of the page instead of around it. The
one ``gr.HTML`` is the Neutralize Prompt row's two lines of text, which take no
input and carry no state; the switch beside them is the same stock Checkbox
every other stage has.
"""

from __future__ import annotations

import html
import logging

import gradio as gr

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

PREFIX = "mc-pipeline"
"""Element-id and class stem for everything this module puts in the page."""

OWNED = ("neutralize", "creative", "spatial", "stage2")
"""The stages Model Chain controls: switchable, emphasised, all but one expandable."""

PLAIN = ("neutralize",)
"""The stages with one control and nothing behind it.

Neutralize Prompt has a switch and no settings: model, device and runtime are
decided where every other role's are, in LLM Studio's Setup, and the txt2img
surface answers one question -- should this generation neutralize pose and
placement before Creative and Spatial? A row for that is a name, a description
and a switch, and not a way in. Drawing it with the card builder below would
put a caret on it that opens a drawer with nothing inside, which is the kind
of furniture the card was pared down to remove.
"""

EXPANDABLE = tuple(stage for stage in OWNED if stage not in PLAIN)
"""The stages with a disclosure: everything a feature fills a body slot in."""

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
    "neutralize": "Neutralize Prompt",
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
    "neutralize": "Bypassed — prompt as-is",
    "creative": "Bypassed — prompt as-is",
    "spatial": "Bypassed — no regions",
    "stage2": "Bypassed — Stage 1 is final",
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


CONFIRM = "Confirm delete"
"""What a Delete button says once it is armed.

§3 of the redesign intent: a destructive action should require an explicit
confirmation where the loss is irreversible. Deleting a saved profile, a Stage 2
preset or a named Spatial layout removes a file, and nothing brings it back.

The confirmation is the button itself rather than a dialog. A modal would be a
second thing to dismiss on a tab that has enough of them, and a browser
``confirm()`` is not styleable, not themeable and not touch-friendly; a button
that changes what it says is all three, and it is the same in-page idiom the
layout workspace uses for Discard.
"""


def confirmed(armed, label: str = "Delete"):
    """One press of a Delete button. ``(go, still_armed, how it now reads)``.

    The first press arms and says so; the second deletes and disarms. Nothing
    else clears it, deliberately: an arm that expired on the next unrelated
    click would be a confirmation somebody could miss by being slow, and the
    only cost of it lingering is a button that has to be read.
    """
    if armed:
        return True, False, gr.update(value=label, variant="stop")
    return False, True, gr.update(value=CONFIRM, variant="stop")


def handoff_note(width: int = 0, height: int = 0) -> str:
    """What crosses the edge into Stage 2, for Stage 2's own description.

    The single most important number on the panel and the one that used to be
    nowhere: Stage 2 refines *finished Stage 1 pixels*, so a Hires pass changes
    what Stage 2 is handed, and a user who has read only the width and height
    sliders has read the wrong number.

    It had a row of its own on the connector between the two stages. That row
    is gone: a pipeline of three cards should be three cards, and a fourth
    thing between two of them that is not a stage, cannot be opened and cannot
    be switched off is furniture. The number is Stage 2's, so it is in Stage 2's
    description now.
    """
    if width > 0 and height > 0:
        return f"{width} × {height} in"
    return ""


SAID = 30
"""How much description a card header carries before it is cut short.

The header is two lines and the second one is a *glance*: what the stage is set
to, not an explanation of it. Cut here, in Python, rather than left to
`text-overflow` -- an ellipsis is a promise the CSS has to keep on every theme,
and this panel has now been broken four separate ways by trusting one to.

Thirty is not a guess. The header keeps a lane clear on its right for the
switch and the caret, which leaves about 200px of text in Forge Neo's
generation column -- thirty characters at the size the second line is set in.
Past that the line wraps, and a wrapped third line is clipped by the band with
no ellipsis to show for it, which reads as a description that just stops.
"""


def _said(summary: str) -> str:
    """One description, flattened and cut to the header's budget."""
    said = " ".join(str(summary or "").split())
    if len(said) > SAID:
        said = said[:SAID - 1].rstrip(" ·,—-") + "…"
    return said


def card_label(stage: str, summary: str = "") -> str:
    """A stage card's whole header, as the one string Gradio gives it.

    Name on the first line, description on the second, in the Accordion's own
    label -- which is the only place a Gradio disclosure can carry text that is
    guaranteed to be *inside* its header.

    Three arrangements were tried before this one and each failed on a theme.
    Building the description beside the accordion and moving it into the header
    from JavaScript worked under Lobe and found nothing under stock Gradio.
    Painting it into a reserved band with `position: absolute` worked under
    stock Gradio and fell outside the card under Lobe, whose header is a
    different height. Leaving it in flow puts it under the *body* the moment
    the card is open.

    A label has none of those failure modes because it is not a second element:
    whatever a theme does to the header, the text is in it. The newline needs
    `white-space: pre-line` to show as a line break, and a theme that refuses
    that gets one line reading "Creative C7 · 2 directions" -- which is a worse
    layout and still the right information in the right place.
    """
    said = _said(summary)
    name = TITLES.get(stage, str(stage))
    return f"{name}\n{said}" if said else name


def plain_label(stage: str, summary: str = "") -> str:
    """A toggle-only row's whole header, as the markup its ``gr.HTML`` shows.

    The same two lines a card carries, cut to the same budget, and written as
    two elements rather than one string with a newline in it. A card has to
    put both lines in an accordion's label -- the only text a disclosure is
    guaranteed to keep inside its header -- and leave it to the browser file
    to split them. This row has no disclosure and so no such constraint: the
    elements the browser file would have made are made here, with the same
    classes, so the stylesheet's rules for a dressed card's two lines apply to
    them from the moment the page is built and whether or not any script
    ever runs. Escaped, because a description is text and not markup.
    """
    said = _said(summary)
    name = html.escape(TITLES.get(stage, str(stage)))
    return (f'<div class="{CARD_HEAD} {PREFIX}-plain-head">'
            f'<div class="{PREFIX}-label">'
            f'<div class="{PREFIX}-name">{name}</div>'
            f'<div class="{PREFIX}-said" title="{html.escape(said)}">{html.escape(said)}</div>'
            f'</div></div>')


def switch(*, elem_id=None, label="ON", **kwargs):
    """A stage's ON control: built off, and never restored from ui-config.json.

    The second half is the point, and it is why this exists rather than three
    ``gr.Checkbox`` calls. Forge keeps a ``ui-config.json`` of every component a
    script builds and writes the saved value back over the one the script asked
    for -- which is exactly right for a slider somebody has tuned, and exactly
    wrong for this. A stage left armed last week came back armed, the card's
    description was built from the settings and said *Bypassed*, and the panel
    contradicted itself in a way no amount of work on the description could fix:
    the checkbox was the half the engine reads, so Generate really did start a
    language model for a stage the panel had just called bypassed.

    ``do_not_save_to_config`` is the host's own opt-out for this, and it is the
    whole fix. The value below is then the value that ships: off, every time,
    on every stage. Being armed lasts for a session, which is one press to
    start and nothing at all to remember.
    """
    box = gr.Checkbox(value=False, label=label, container=False,
                      elem_id=elem_id, elem_classes=classes("toggle"), **kwargs)
    # An attribute rather than an argument: it is Forge's, not Gradio's, and a
    # host that does not have it simply does not read it.
    box.do_not_save_to_config = True
    return box


def card_summary(stage: str, summary: str):
    """The update that repaints one stage's description.

    A card's description is its accordion's label, so the update sets a
    label; a toggle-only row's is the value of its ``gr.HTML``, so the update
    sets a value. One function for both, because the feature scripts repaint
    a row by stage name and should not have to know which kind it is.
    """
    if stage in PLAIN:
        return gr.update(value=plain_label(stage, summary))
    return gr.update(label=card_label(stage, summary))


class Pipeline:
    """The four rows, and the empty containers the feature scripts fill.

    Built once per Gradio Blocks by :func:`host`. A script asks for the slots
    belonging to a stage it owns -- two for a card, one for a toggle-only row
    -- re-enters them, and builds whatever it was building before, unchanged
    and in most cases moved rather than rewritten.
    """

    def __init__(self):
        self.accordion = None
        # Kept as an attribute and never built. The handoff had a row of its
        # own between Spatial and Stage 2; the number is Stage 2's and is in
        # Stage 2's description now. Anything still reading this gets None
        # rather than a traceback.
        self.handoff = None
        self.heads: dict = {}
        self.bodies: dict = {}
        self.editors: dict = {}
        self.summaries: dict = {}
        self.rows: dict = {}
        self.filled: set = set()

    # -- what a feature script asks for ------------------------------------ #

    def head(self, stage: str):
        """The container beside a stage's title, for its ON/OFF switch.

        Section 3.3: a stage is switched from the collapsed row, so the switch
        cannot live inside the drawer it would otherwise be needed to open.
        For a toggle-only row the switch is the whole of what a feature fills,
        so asking for its head is what marks the stage filled.
        """
        if stage in PLAIN:
            self.filled.add(stage)
        return self._slot(self.heads, stage, "head")

    def body(self, stage: str):
        """The container inside a stage's disclosure, for its editor."""
        self.filled.add(stage)
        return self._slot(self.bodies, stage, "body")

    def summary(self, stage: str):
        """The component a stage's description is written to, or ``None``.

        For a card it is the stage's own disclosure: a feature repaints its
        description by returning ``card_summary(stage, text)`` for this
        component, which sets the accordion's label -- so the description is
        inside the header under every theme, and there is no second element to
        fall out of it. For a toggle-only row it is the ``gr.HTML`` that holds
        both lines, and the same call repaints its value.
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
    """One stage card: a name, a description, a switch, and a way in.

    That is the whole of it, and the list is exhaustive on purpose. A pipeline
    of three stages should be three rows; everything this card grew that was
    not one of those four things -- a title row above the disclosure, a rail
    with a node on it beside every card, a handoff line between two of them --
    was furniture that read as structure under one theme and as debris under
    another.

    The name and the description are one string in the accordion's own label,
    because that is the only place a Gradio disclosure can carry text that is
    guaranteed to be inside its header. The switch is the one element that has
    to be outside it: it is a sibling, painted into a lane the header reserves,
    so that arming a stage and opening it are two controls that share a line
    and never each other's presses.
    """
    with gr.Column(elem_id=ident("stage", stage),
                   elem_classes=classes("stage", "stage-owned", f"stage-{stage}")) as row:
        pipeline.rows[stage] = row

        # Closed, and the summary target. A stage's description is repainted by
        # whichever feature owns it, through `card_summary()`, which writes this
        # accordion's label rather than a second component's value.
        with drawer(card_label(stage, PLACEHOLDERS.get(stage, "")),
                    elem_id=ident("editor", stage),
                    elem_classes=classes("editor")) as editor:
            pipeline.editors[stage] = editor
            pipeline.summaries[stage] = editor
            with gr.Column(elem_id=ident("body", stage),
                           elem_classes=classes("body")) as body:
                pipeline.bodies[stage] = body

        # Filled by the feature that owns the stage, with the switch it already
        # had. Nothing is created here: a second checkbox mirroring the real one
        # is exactly the duplicate source-of-truth section 2.5 forbids, and it
        # would be the one the user reached for first.
        with gr.Column(min_width=0, scale=0,
                       elem_id=ident("switch", stage),
                       elem_classes=classes("switch")) as head:
            pipeline.heads[stage] = head


def _plain_row(pipeline: Pipeline, stage: str) -> None:
    """One toggle-only stage: a name, a description, a switch, and no way in.

    The first three of the four things a card is and not the fourth. Built as
    a stage column like every other row, with the same switch slot in the same
    lane, so the stylesheet and the browser file treat it as one of the panel's
    rows -- it lights when its phase runs, it sits in the same outline -- and
    differ from a card in exactly one respect: there is no disclosure, so
    nothing draws a caret and nothing opens.

    The two lines are written here as two elements rather than as a label the
    browser file splits, and the column carries :data:`CARDED` from the start,
    because Python furnished this header itself and there is nothing left for a
    script to find. If every line of JavaScript on the page fails, this row
    looks exactly as it does when none of it does.
    """
    with gr.Column(elem_id=ident("stage", stage),
                   elem_classes=[*classes("stage", "stage-owned", f"stage-{stage}",
                                          "stage-plain"), CARDED]) as row:
        pipeline.rows[stage] = row

        # The summary target: the same call that sets a card's label sets this
        # component's value. See card_summary().
        head = gr.HTML(plain_label(stage, PLACEHOLDERS.get(stage, "")),
                       elem_id=ident("plain", stage), elem_classes=classes("plain"))
        pipeline.summaries[stage] = head

        # Filled by the feature that owns the stage, with the one control it
        # has. Nothing is created here, for the reason nothing is created in
        # a card's switch slot: the switch is the control the generation reads,
        # and a second one mirroring it would be the duplicate source-of-truth
        # section 2.5 forbids.
        with gr.Column(min_width=0, scale=0,
                       elem_id=ident("switch", stage),
                       elem_classes=classes("switch")) as switch:
            pipeline.heads[stage] = switch


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
            if stage in PLAIN:
                _plain_row(pipeline, stage)
            else:
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
