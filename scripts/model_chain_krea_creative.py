"""Krea Creative Mode in txt2img: two controls, and a drawer of decisions made.

The default surface is one checkbox, one slider and a collapsed accordion:

    [ Creative Mode ]   Creativity [0----5----10]   ▸ Creative Controls

Opening the drawer on a fresh install shows a profile bar, the sentence "Active
direction: None", and one dropdown offering to add one. It does not show ten
axes. Ten axes with a mode and a value each is twenty controls describing
decisions nobody has made, and the old panel drew all twenty every time -- with
nine of them saying Vary, because the factory defaults said Vary, which is art
direction arriving from nowhere.

What the drawer holds now is in :mod:`mc_creative_panel`, which LLM Studio's Krea
tab builds too: one panel, one layout, one set of handlers, two surfaces.

A separate always-on script rather than another accordion inside Model Chain,
for three reasons that are all about blast radius. Model Chain's ``ui()`` returns
a long argument list that travels in presets and infotexts, and Creative Mode's
controls have no business in either. Model Chain is a large, long-settled thing
and a Creative gate that failed to build must not be able to take the two-stage
chain down with it. And Creative Mode is txt2img-only for a different reason than
Model Chain is, so the two ``show()`` methods happen to agree today without that
being one decision.

Where the model is called
-------------------------
:meth:`ScriptKreaCreative.before_process` calls it, once, at the top of the
generation the user just started -- before Forge builds ``all_prompts`` and
before the checkpoint is (re)loaded, which is the same ordering the roll always
had and the reason the writer can still size itself against a card the image
model has not taken yet.

It used to be called from a Gradio handler instead, ahead of the click, because
an LLM run waits for the host to stop generating and a roll asked for from
inside a running image job would have been waiting for the job that was waiting
on it. The browser enforced that ordering: it swallowed the Generate click, ran
the roll, then clicked Generate again itself.

The cost of that arrangement was that a generation could not finish without a
live page. The press did not start an image; a ``setInterval`` in the tab did,
once the roll came back -- and browsers throttle those to one tick a second in a
hidden tab and one a minute in a frozen one, so a Creative generation was late
if you changed windows and never happened at all if you closed the tab.

So the deadlock is now answered where it lives: :class:`mc_broker.host_job` lets
this hook say that the image job is blocked waiting for the roll, which is
precisely the case in which waiting for the image job is the wrong thing to do.
One press does everything, and nothing after the press needs a browser.

Pasting an image back
---------------------
A Creative generation records its *expanded* prompt as the image's own
``Prompt:`` line, because that is what the image model was given. So an ordinary
paste -- PNG Info, "send to txt2img", the arrow under the gallery -- restores that
paragraph and switches Creative Mode **off**, and the picture reproduces. Leaving
Creative Mode on would hand the expansion back to the writer as though it were a
short idea and expand it a second time, which reproduces nothing.

Getting back to the *workflow* is a separate, explicit action: **Restore Creative
setup** puts the recorded source phrase back in the prompt box, restores the axis
configuration the image was made under, and can arm the recorded recipe for one
generation. See :func:`_restore_setup`.

What is not here
----------------
No idle delay, no typing watcher, no repeat toggle, no reroll scheduler, no
status machine and no click gate. A roll happens because somebody pressed
Generate. Pressing it again is how you get another one.
"""

from __future__ import annotations

import gradio as gr

import mc_creative_krea
import mc_creative_panel
import mc_creative_profiles
import mc_hint
import mc_infotext
import mc_llm_sessions as sessions
import mc_literal_prompts
import mc_lora
import mc_memory
import mc_pipeline_panel
import mc_plan
import mc_krea_pipeline
import mc_profile_state
import mc_spatial
import mc_spatial_profiles
from modules import errors, scripts
from modules.ui_common import refresh_symbol
from modules.ui_components import ToolButton

logger = mc_memory.logger
"""Shared with the helper modules; mc_memory attaches the console handler."""

PREFIX = "mc-krea-creative"
"""Every id this script puts in the page starts here.

The browser gate finds its elements by these ids and by nothing else -- no
Gradio-generated class, no DOM shape. A theme that replaces Gradio's internals
can change how this looks and cannot stop it working.
"""

PROMPT_ELEM_ID = "txt2img_prompt"
"""The one native control this script writes to, and only when asked to.

"Restore Creative setup" puts the recorded *source* phrase -- the short idea, not
the expansion -- back where it was typed, because there is nowhere else for it to
go: continuing from an old image's idea means having that idea in the prompt box.
Nothing else here touches it, and nothing touches it without a button press.
"""


def ident(*parts: str) -> str:
    """A stable, extension-owned element id."""
    return "-".join((PREFIX,) + tuple(str(part) for part in parts if part))


def notice(text: str, kind: str = "info", hint: str = "") -> str:
    """One line of Creative Mode status, as scoped HTML.

    Its own classes rather than LLM Studio's, because ``style.css`` scopes those
    under ``#mc-llm-studio`` and this line is in txt2img. Same idea, same
    reliance on the host's custom properties for colour, different neighbourhood.

    ``hint`` is the half of the old line that never changed: what a mode *means*,
    as against what is true right now. It becomes an "i" at the end of the line
    -- see :mod:`mc_hint` -- so the line says "Spatial Layout: 7 regions" and
    keeps the paragraph that used to follow it a hover away.

    The text is escaped and the badge is not, which is the right way round: the
    text can contain a layout name somebody typed, and the badge is built here
    out of this extension's own words.
    """
    import html

    return (f'<div class="{PREFIX}-notice {PREFIX}-notice-{kind}">'
            f'{html.escape(str(text or ""))}{mc_hint.badge(hint)}</div>')


# --------------------------------------------------------------------------- #
# The settings one roll runs with
# --------------------------------------------------------------------------- #


SPATIAL_CONTROLS = 3
"""How many controls the Spatial block contributes: enabled, mode, layout."""

LITERAL_CONTROLS = 2
"""How many the Literal Prompt block contributes: the positive and negative box.

Inserted *before* the Spatial tail and not after it, which looks backwards and
is the only placement that works. :mod:`mc_plan` reads the Spatial controls off
the end of this tuple -- ``args[-SPATIAL_TAIL:]`` -- because the middle is a
variable number of axis controls and the two ends are the two that can be read
without counting. Appending here would have moved the end out from under it and
described every Spatial generation's plan wrongly.

So the ends stay the ends. The new block goes in the variable middle, where the
only thing that reads it is :func:`_split`, which knows how long the axis block
is because it asks the library.
"""

def _split(values) -> tuple[tuple, tuple, tuple, tuple]:
    """``before_process``'s tuple, cut into its four parts.

    ``ui()`` returns, after the enabled flag: three scalars, then three controls
    per axis, then the three Spatial controls. Two of those three lengths are
    fixed and the middle one is the library's, so the cut is made by *asking the
    library* rather than by pattern-matching a length -- both a layout with
    spatial and one without are multiples of three long, and a tuple cannot say
    which it is.

    Three scalars and not four: the Pinned LoRAs box is gone, and
    ``[[<lora:name:weight>]]`` in the prompt is what replaced it. An older API
    caller that still sends four will have its LoRA string read as the first
    axis mode, which is not a mode and is dropped by
    :func:`mc_creative_panel.axes_from` -- the axis falls back to the saved
    setting rather than to something invented.

    The second shape is the one that matters now Spatial is a peer feature. A
    creativity library that will not load takes the axis controls with it and
    Creative Mode with them -- but not Spatial, which needs no vocabulary at all
    to place a box. So ``ui()`` still emits the Spatial tail on that path, and
    this cut recognises the exact shape it emits rather than guessing from a
    length: one scalar and three Spatial controls.

    Anything else means there is no panel behind this call -- an API request
    that sent only the flag. The saved settings answer for it, which is what
    those callers already got.
    """
    values = tuple(values or ())
    try:
        from prompt_master.krea import library as library_module

        axes = len(library_module.library().axis_keys) * 3
    except Exception:
        axes = None

    if axes is not None:
        expected = 3 + axes
        if len(values) >= expected + LITERAL_CONTROLS + SPATIAL_CONTROLS:
            after = expected + LITERAL_CONTROLS
            return (values[:3], values[3:expected],
                    values[after:after + SPATIAL_CONTROLS],
                    values[expected:after])
        if len(values) >= expected + SPATIAL_CONTROLS:
            # The shape sent before the Literal Prompt boxes existed. Cut
            # exactly where it always was, and contributing no fields.
            return (values[:3], values[3:expected],
                    values[expected:expected + SPATIAL_CONTROLS], ())
        if len(values) >= expected:
            return values[:3], values[3:expected], (), ()

    # The no-panel shapes: creativity, then -- for a caller new enough to send
    # them -- the two Literal Prompt boxes, then the Spatial tail.
    if len(values) == 1 + LITERAL_CONTROLS + SPATIAL_CONTROLS:
        return ((), (), values[-SPATIAL_CONTROLS:],
                values[1:1 + LITERAL_CONTROLS])
    if len(values) == 1 + SPATIAL_CONTROLS:
        return (), (), values[-SPATIAL_CONTROLS:], ()
    if len(values) >= 3:
        return values[:3], (), (), ()
    return (), (), (), ()


def _literals_for(values) -> tuple[str, str]:
    """``(Literal Positive, Literal Negative)``, from the panel or the settings.

    Read off the panel when it sent them, for the reason every other control on
    this hook is: the box somebody just typed into is what they are looking at,
    and a generation that used the last *saved* value would silently ignore it.

    A caller with no panel -- the API, or a build whose UI could not be
    assembled -- gets the saved values, which is also what makes section 3.3
    work: the fields keep affecting generations while their row is off screen,
    and they are off screen precisely when this extension's features are off.
    """
    _, _, _, fields = _split(values)
    stored = mc_literal_prompts.settings()
    if len(fields) < LITERAL_CONTROLS:
        return stored["positive"], stored["negative"]
    positive, negative = fields
    return str(positive or ""), str(negative or "")


def _settings_for(values) -> dict:
    """This generation's Creative settings, from the panel when it sent them.

    ``values`` is what Forge handed ``before_process`` after the enabled flag,
    in the order :meth:`ui` returned it. A UI that could not build its axis
    controls sends fewer, and an API request sends none at all, so the length is
    checked rather than assumed and the saved preferences answer for anything
    absent.
    """
    scalars, axes, _, _ = _split(values)
    if not scalars:
        return mc_creative_krea.settings()
    creativity, seed, anti_repetition = scalars
    return _stored(creativity, seed, anti_repetition, axes)


def _spatial_for(values) -> dict:
    """This generation's Spatial settings, from the panel when it sent them.

    Read off the panel for the same reason the Creative settings are: the radio
    button somebody just moved is what they are looking at, and a generation
    that used the last *saved* compose mode would silently ignore it. A caller
    with no panel -- the API -- gets the saved settings, which is also how it
    gets a layout at all: the serialized canvas is persisted, so a script that
    sends only the Creative flag still composes the boxes the user last drew.
    """
    _, _, spatial, _ = _split(values)
    stored = mc_spatial.settings()
    if len(spatial) < SPATIAL_CONTROLS:
        return stored
    enabled, mode, layout = spatial
    from prompt_master.krea import spatial as spatial_module

    mode = str(mode or "").strip().casefold()
    stored["enabled"] = bool(enabled)
    stored["compose_mode"] = mode if mode in spatial_module.COMPOSE_MODES \
        else stored["compose_mode"]
    stored["layout"] = str(layout or "")
    return stored


def _stored(creativity, seed, anti_repetition, axis_values) -> dict:
    """The settings for this roll, taken from the controls rather than the file.

    Read off the panel and not out of preferences, because the panel is what the
    user is looking at. A roll that used the last *saved* value would silently
    ignore the slider somebody had just moved, which is the sort of bug that
    only shows up as "the creativity control does not seem to do anything".
    """
    from prompt_master.krea import director, variation

    stored = mc_creative_krea.settings()
    stored["creativity"] = variation.clamp(creativity)
    stored["anti_repetition"] = bool(anti_repetition)
    try:
        stored["seed"] = int(seed)
    except (TypeError, ValueError):
        stored["seed"] = director.RANDOM_SEED

    modes, fixed, excluded = mc_creative_panel.axes_from(axis_values)
    if modes:
        stored["axis_modes"] = modes
        stored["fixed_values"] = mc_creative_krea.known_fixed(fixed)
        stored["excluded_values"] = mc_creative_krea.known_excluded(excluded)
    return stored


def _recipe_view(recipe) -> str:
    """The last recipe as something a person can read and argue with.

    Deliberately the ids *and* the labels. The ids are what the metadata records
    and what a Fixed selection stores, so somebody who liked a roll needs to see
    them; the labels are what makes the list mean anything at a glance.
    """
    items = getattr(recipe, "items", ())
    notes = getattr(recipe, "notes", ())
    if not items:
        head = (f"No creative direction at Creativity {getattr(recipe, 'creativity', 0)}.\n"
                "Creativity 0 and 1 direct nothing by design; above that, add at least "
                "one direction.")
        return "\n".join([head, *notes]) if notes else head

    lines = [f"Creative seed: {recipe.creative_seed}   ·   LLM seed: {recipe.llm_seed}"
             f"   ·   library {recipe.library_version}"]
    if getattr(recipe, "replayed", False):
        lines.append("Replayed from a recorded recipe: nothing was drawn and no recent "
                     "history was consulted.")
    if recipe.locked:
        lines.append("Locked by your prompt: " + ", ".join(recipe.locked))
    for note in notes:
        lines.append(f"Note: {note}")
    lines.append("")
    for item in items:
        lines.append(f"{item.label} [{item.source}] {item.variant_id} — {item.variant_label}")
    lines.append("")
    lines.append(recipe.brief)
    return "\n".join(lines)


def _toggled(enabled, spatial_enabled, serialized, mode):
    """Arm or bypass Creative Mode, and re-describe the pipeline it sits in.

    Nothing appears or disappears any more. The slider and the drawer used to
    hide with the feature, which meant the only way to configure Creative Mode
    was to turn it on first; the stage is a row with a switch on it now, and a
    bypassed stage stays visible and reads as bypassed -- section 3.3.

    Two of the four outputs belong to the other stage, and they are here for the
    reason this whole refactor exists. Spatial no longer appears or disappears
    with Creative Mode -- it is a peer feature -- but what its pipeline *is*
    depends on both switches: the same Direct merge composes around a written
    scene with Creative on and around the typed prompt with it off. One of the
    two sentences would be wrong if this handler did not send the other stage's
    lines as well.
    """
    mc_creative_krea.remember(**{mc_creative_krea.ENABLED: bool(enabled)})
    if enabled:
        objection = mc_creative_krea.checkpoint_objection()
        stored = mc_creative_krea.settings()
        directions = mc_creative_panel.active_note(stored)
        told = notice(objection or
                      ("Creative Mode is on. Press Generate: the prompt is directed "
                       f"locally and expanded once, then Forge makes the image. "
                       f"{directions}"),
                      "warn" if objection else "info")
    else:
        told = notice("Creative Mode is off.")
    return (told,
            spatial_summary(serialized, bool(spatial_enabled),
                            creative=bool(enabled), mode=mode),
            mc_pipeline_panel.card_summary("creative", _creative_line(bool(enabled))),
            mc_pipeline_panel.card_summary(
                "spatial", _spatial_line(serialized, bool(spatial_enabled), mode)),
            _literal_row(enabled, spatial_enabled, other=spatial_enabled))


def _remember_creativity(value):
    """Keep the slider's position, and say what it will actually do.

    ``variation.describe`` alone describes the *scale*, which is only half the
    answer and reads as a lie when the other half is "nothing is directed": at
    Creativity 10 with every axis Natural it said "extreme direction on every
    eligible axis" over a brief of zero characters.
    """
    from prompt_master.krea.variation import clamp

    mc_creative_krea.remember(**{mc_creative_krea.CREATIVITY: clamp(value)})
    stored = mc_creative_krea.settings()
    told = mc_creative_panel.describe_creativity(value, stored)
    return notice(told, "warn" if "nothing to scale" in told else "info")


def _last_roll():
    """The most recent roll, for the diagnostics drawer.

    Reads :attr:`mc_creative_krea.Creative.last`, which the roll writes as its
    final act and which outlives the page: a generation started before the tab
    was closed can be inspected in the tab that opens after it.
    """
    last = mc_creative_krea.creative.last
    if last is None:
        return ("No roll has been made yet in this session.", "")
    return _recipe_view(last.recipe), last.expanded


# --------------------------------------------------------------------------- #
# The Spatial Layout editor
# --------------------------------------------------------------------------- #

SPATIAL_PREFIX = "mc-krea-spatial"
"""Every element the layout editor owns starts here.

Its own prefix rather than Creative Mode's, because the editor is a workspace
rather than a control: it is built as one static block, it is the only thing in
this extension whose interior the browser rearranges, and giving it a namespace
of its own is what lets a test say "these ids are the editor" without knowing
anything about the panel around it.
"""

HANDLES = ("nw", "n", "ne", "e", "se", "s", "sw", "w")
"""The eight resize affordances, in the order they are drawn.

Eight rather than four because a one-dimensional adjustment -- widen this, lower
that edge -- is most of what resizing a region actually is, and doing it from a
corner means fixing the other dimension afterwards. The four extra knobs cost
four elements; the visible knob stays small and the hit target does not (see
``--mc-grab`` in style.css).
"""


def _spatial_id(*parts: str) -> str:
    return "-".join((SPATIAL_PREFIX,) + tuple(str(part) for part in parts if part))


def _options(vocabulary, blank: str) -> str:
    """One ``<select>``'s options, out of the vocabulary that defines them.

    Read from :mod:`prompt_master.krea.spatial` rather than written out here, so
    that a framing offered on screen is a framing the compositor knows a phrase
    for. The alternative -- a list in the markup and a dictionary in Python --
    is two lists that agree until somebody adds to one of them, and the failure
    is a selection silently dropped at generation time.
    """
    import html

    found = []
    for value in vocabulary:
        label = value or blank
        found.append(f'<option value="{html.escape(value)}">{html.escape(label)}</option>')
    return "".join(found)


def _handles() -> str:
    """The eight resize knobs, drawn once into the selection proxy.

    They belong to the proxy and not to a region, which is the whole of §7:
    handles that lived on the region body would be buried with it the moment
    something overlapped it.
    """
    return "".join(
        f'<span class="{SPATIAL_PREFIX}-handle {SPATIAL_PREFIX}-handle-{corner}"'
        f' data-corner="{corner}" aria-hidden="true"></span>'
        for corner in HANDLES)


SHAPES = (
    ("rect", "Box"),
    ("head", "Head"),
    ("chest", "Chest"),
    ("waist", "Waist"),
    ("left_arm", "Left arm"),
    ("right_arm", "Right arm"),
    ("left_hand", "Left hand"),
    ("right_hand", "Right hand"),
    ("left_leg", "Left leg"),
    ("right_leg", "Right leg"),
    ("left_foot", "Left foot"),
    ("right_foot", "Right foot"),
)
"""Every ``ui_shape`` the editor can draw, and what it is called on screen.

Editor metadata and nothing else. §2.5: whatever silhouette is on the canvas,
the compositor is handed the same axis-aligned rectangle it has always been
handed, so this list can grow without a single downstream branch changing. The
proportions each shape starts at and the prompt it arrives with live in
``javascript/model_chain_spatial_krea.js``, next to the code that applies them;
a test asserts the two lists still name the same shapes.
"""

QUICK_SHAPES = ("rect", "head", "chest", "waist")
"""§10: the high-frequency four, reachable without opening the rail."""

PERSON_SHAPES = tuple(name for name, _label in SHAPES if name != "rect")
"""§9.2: the segmented outline, every part of it grabbable on its own."""

PANELS = (
    ("prompts", "Prompts"),
    ("person", "Person"),
    ("layers", "Layers"),
    ("inspector", "Inspector"),
    ("gallery", "Gallery"),
    ("session", "Session"),
)
"""The right rail, in the order §12.2 lists it.

One tuple, three readers: the rail builds a widget per row, the Panels popup
builds a visibility switch per row, and the browser file collapses and hides
them by the same key. A widget added here appears in all three.
"""


def _label_for(shape: str) -> str:
    for name, label in SHAPES:
        if name == shape:
            return label
    return shape


def _palette(shapes, kind: str) -> str:
    """One row of shape buttons: tap to place centred, drag to place where dropped.

    Buttons rather than decorated ``<div>``s because both interactions are on
    the same control -- §10.2 and §10.3 -- and a tap that does something is a
    button whatever else it can also do. The silhouette is a ``clip-path`` in
    the stylesheet keyed off ``data-shape``, so this markup carries the shape
    name once and no geometry at all.
    """
    import html

    found = []
    for shape in shapes:
        label = html.escape(_label_for(shape), quote=True)
        # Two classes doing two jobs. ``-shape-`` is the silhouette, and it
        # belongs to the art layer *inside* the button rather than to the
        # button -- a clip-path on the button would cut the label off with it.
        # ``-part-`` is where the button sits, which only the Person outline
        # has an opinion about, so only the Person outline is given one.
        where = f" {SPATIAL_PREFIX}-part-{shape}" if kind == "person" else ""
        found.append(
            f'<button type="button" class="{SPATIAL_PREFIX}-shape-button{where}"'
            f' data-shape="{shape}"'
            f' data-palette="{kind}" aria-label="{label}">'
            f'<span class="{SPATIAL_PREFIX}-shape-art {SPATIAL_PREFIX}-shape-{shape}"'
            f' aria-hidden="true"></span>'
            f'<span class="{SPATIAL_PREFIX}-shape-name">{label}</span></button>')
    return "".join(found)


def _panel_switches() -> str:
    """The Panels popup's contents: one switch per widget, and nothing else.

    §4.6 -- this popup exists to control workspace layout. It carries no
    editing action, so nothing in it can be pressed by mistake while reaching
    for one.
    """
    import html

    found = []
    for key, label in PANELS:
        said = html.escape(label, quote=True)
        found.append(
            f'<label class="{SPATIAL_PREFIX}-switch-row">'
            f'<input type="checkbox"'
            f' id="{_spatial_id("show", key)}" data-panel="{key}" checked />'
            f'<span>{said}</span></label>')
    return "".join(found)


def _widget(key: str, label: str, head_extra: str, body: str) -> str:
    """One collapsible rail widget.

    The header is a ``<button>``: §12.2 asks for every widget to be
    collapsible, and a header somebody can collapse is a control rather than a
    heading with a listener bolted to it. Whatever live number belongs beside
    the name -- a region count, an image position -- rides in ``head_extra``,
    which is the only thing §2.2 lets a header say beyond its own name.

    The grip is where a widget is dragged to reorder the rail, and it is a
    separate target from the rest of the header for one reason: the header is
    wide and gets pressed often, and a rail that rearranged itself whenever
    somebody's finger slid six pixels on the way to collapsing something would
    be a rail nobody trusted. Grip to move, header to collapse.
    """
    return f'''
      <section class="{SPATIAL_PREFIX}-widget" data-panel="{key}"
               id="{_spatial_id("panel", key)}">
        <button type="button" class="{SPATIAL_PREFIX}-widget-head"
                data-panel="{key}" aria-expanded="true"
                aria-controls="{_spatial_id("panel", key, "body")}">
          <span class="{SPATIAL_PREFIX}-widget-grip" aria-hidden="true">&#10287;</span>
          <span class="{SPATIAL_PREFIX}-caret" aria-hidden="true"></span>
          <span class="{SPATIAL_PREFIX}-widget-name">{label}</span>
          {head_extra}
        </button>
        <div class="{SPATIAL_PREFIX}-widget-body"
             id="{_spatial_id("panel", key, "body")}">{body}</div>
      </section>'''


def spatial_editor() -> str:
    """The layout workspace's markup: every control, none of its behaviour.

    Built in Python and handed to the page as one static block, for two reasons
    that are both about single sources of truth. The framing and camera-angle
    lists are the compositor's own vocabularies, so a value that can be chosen
    is a value that renders a phrase. And the whole editor is one element with
    stable ids, so the browser file finds what it needs by id and by nothing
    else -- no Gradio class, no DOM shape, nothing a theme can rearrange.

    Canvas first
    ------------
    §2.1. The composition frame is the work, and everything else is a tool
    beside it, so the shape of this markup is one bar, one canvas column and
    one rail -- not a stack of panels of equal weight with a picture somewhere
    in it. The canvas column is as tall as the workspace and does not move; the
    rail beside it scrolls on its own and every widget in it can be collapsed
    or switched off entirely, so no amount of panel content can push the frame
    off screen.

    A workspace, not a modal
    ------------------------
    This used to be a ``position: fixed`` overlay that JavaScript moved to
    ``document.body``, because a fixed overlay inside an accordion is one
    ``overflow: hidden`` or one ``transform`` away from being a modal nobody can
    see. Moving it solved that and bought a second problem: two copies of every
    id whenever Gradio rebuilt the tab, and a page whose scrolling belonged to
    the overlay rather than to the browser.

    It is now a block in ordinary document flow that Full Screen reveals and
    Close hides -- and when it is revealed, §3.1's takeover hides the rest of
    txt2img around it rather than this element escaping anywhere. It inherits
    the theme it is standing in, it cannot be clipped out of existence by a
    container it is no longer escaping, and on a phone it is a page rather than
    a window over one.

    No coordinates
    --------------
    §8.3 offers numeric X/Y/W/H, and they are not here. They were built once,
    removed, built again with the rail underneath them, and removed again --
    which is worth recording rather than quietly reverting a third time. The
    editor is pointer-first: a normalized coordinate is an implementation
    detail of the storage format rather than something anybody composes in, and
    four number boxes and a coordinate readout invite people to think in a unit
    the picture does not have. Boxes are still clamped, ordered and validated
    exactly as before; what is gone is the way of typing one, and the way of
    reading one off a layer row.

    Nothing here explains itself
    ----------------------------
    §2.2 and §25. There is no helper paragraph, no ``title=`` hover text, no
    instructional footer and no badge describing a control that is already
    labelled. What is left is control labels, field labels, button names, live
    state and the selected region's own data.

    The canvas is a ``<div>``, not a ``<canvas>``. Regions are elements, so a
    region can be found by id, be focused, be styled by a theme and be read by
    a test; a bitmap canvas would put all of that behind a redraw loop in order
    to draw rectangles.
    """
    from prompt_master.krea import spatial

    return f'''
<div id="{_spatial_id("workspace")}" class="{SPATIAL_PREFIX}-workspace"
     role="region" aria-label="Spatial Layout" hidden>

  <div class="{SPATIAL_PREFIX}-bar" role="toolbar" aria-label="Spatial Layout">
    <label class="{SPATIAL_PREFIX}-power">
      <input type="checkbox" id="{_spatial_id("power")}" />
      <span>Spatial</span>
    </label>

    <span class="{SPATIAL_PREFIX}-segmented" role="group" aria-label="Composition">
      <button type="button" id="{_spatial_id("mode", spatial.DIRECT)}"
              class="{SPATIAL_PREFIX}-segment" data-mode="{spatial.DIRECT}"
              aria-pressed="false">Direct BBOX</button>
      <button type="button" id="{_spatial_id("mode", spatial.SMART)}"
              class="{SPATIAL_PREFIX}-segment" data-mode="{spatial.SMART}"
              aria-pressed="false">Smart Spatial</button>
    </span>

    <span class="{SPATIAL_PREFIX}-divider" aria-hidden="true"></span>

    <span class="{SPATIAL_PREFIX}-menu">
      <button type="button" id="{_spatial_id("quick")}"
              class="{SPATIAL_PREFIX}-tool" aria-haspopup="true"
              aria-expanded="false"
              aria-controls="{_spatial_id("quick", "popup")}">Quick Add ▾</button>
      <div id="{_spatial_id("quick", "popup")}" class="{SPATIAL_PREFIX}-popup" hidden>
        <div class="{SPATIAL_PREFIX}-palette">{_palette(QUICK_SHAPES, "quick")}</div>
        <label class="{SPATIAL_PREFIX}-slider">
          <span>Size</span>
          <input type="range" id="{_spatial_id("quick", "size")}"
                 min="60" max="900" step="10" value="340" />
        </label>
      </div>
    </span>

    <button type="button" id="{_spatial_id("draw")}"
            class="{SPATIAL_PREFIX}-tool" aria-pressed="false">Draw</button>
    <button type="button" id="{_spatial_id("clear")}"
            class="{SPATIAL_PREFIX}-tool">Clear All</button>

    <span class="{SPATIAL_PREFIX}-menu">
      <button type="button" id="{_spatial_id("panels")}"
              class="{SPATIAL_PREFIX}-tool" aria-haspopup="true"
              aria-expanded="false"
              aria-controls="{_spatial_id("panels", "popup")}">Panels ▾</button>
      <div id="{_spatial_id("panels", "popup")}" class="{SPATIAL_PREFIX}-popup" hidden>
        {_panel_switches()}
        <div class="{SPATIAL_PREFIX}-popup-actions">
          <button type="button" id="{_spatial_id("collapse", "all")}"
                  class="{SPATIAL_PREFIX}-tool">Collapse All</button>
          <button type="button" id="{_spatial_id("expand", "visible")}"
                  class="{SPATIAL_PREFIX}-tool">Expand Visible</button>
        </div>
      </div>
    </span>

    <span class="{SPATIAL_PREFIX}-spacer"></span>

    <button type="button" id="{_spatial_id("undo")}"
            class="{SPATIAL_PREFIX}-tool">Undo</button>
    <button type="button" id="{_spatial_id("redo")}"
            class="{SPATIAL_PREFIX}-tool">Redo</button>
    <button type="button" id="{_spatial_id("save")}"
            class="{SPATIAL_PREFIX}-tool {SPATIAL_PREFIX}-primary">Save</button>
    <button type="button" id="{_spatial_id("cancel")}"
            class="{SPATIAL_PREFIX}-tool">Close</button>
  </div>

  <div id="{_spatial_id("confirm")}" class="{SPATIAL_PREFIX}-confirm" hidden>
    <span>Unsaved changes</span>
    <span class="{SPATIAL_PREFIX}-spacer"></span>
    <button type="button" id="{_spatial_id("keep")}"
            class="{SPATIAL_PREFIX}-tool">Cancel</button>
    <button type="button" id="{_spatial_id("discard")}"
            class="{SPATIAL_PREFIX}-tool {SPATIAL_PREFIX}-danger">Discard</button>
  </div>

  <p id="{_spatial_id("message")}" class="{SPATIAL_PREFIX}-message"
     role="status" aria-live="polite" hidden></p>

  <div class="{SPATIAL_PREFIX}-body">

    <div class="{SPATIAL_PREFIX}-stage">
      <div class="{SPATIAL_PREFIX}-stagebar">
        <span id="{_spatial_id("aspect")}" class="{SPATIAL_PREFIX}-aspect"></span>
        <span class="{SPATIAL_PREFIX}-spacer"></span>
        <button type="button" id="{_spatial_id("zoom-fit")}"
                class="{SPATIAL_PREFIX}-tool">Fit</button>
        <button type="button" id="{_spatial_id("zoom-out")}"
                class="{SPATIAL_PREFIX}-tool" aria-label="Zoom out">&#8722;</button>
        <span id="{_spatial_id("zoom-level")}"
              class="{SPATIAL_PREFIX}-zoom-level">100%</span>
        <button type="button" id="{_spatial_id("zoom-in")}"
                class="{SPATIAL_PREFIX}-tool" aria-label="Zoom in">+</button>
        <select id="{_spatial_id("grid")}" aria-label="Grid">
          <option value="none" selected>None</option>
          <option value="thirds">Thirds</option>
          <option value="center">Centre</option>
          <option value="both">Both</option>
        </select>
      </div>
      <div id="{_spatial_id("scroll")}" class="{SPATIAL_PREFIX}-scroll">
        <div id="{_spatial_id("canvas")}" class="{SPATIAL_PREFIX}-canvas" tabindex="0"
             role="application" aria-label="Composition frame">
          <div id="{_spatial_id("guides")}" class="{SPATIAL_PREFIX}-guides none"
               aria-hidden="true"></div>
          <div id="{_spatial_id("regions")}" class="{SPATIAL_PREFIX}-regions"></div>
          <div id="{_spatial_id("proxy")}" class="{SPATIAL_PREFIX}-proxy" hidden>
            <span id="{_spatial_id("proxy-label")}"
                  class="{SPATIAL_PREFIX}-proxy-label"></span>
            {_handles()}
          </div>
        </div>
      </div>
    </div>

    <div id="{_spatial_id("rail")}" class="{SPATIAL_PREFIX}-rail">
      {_widget("prompts", "Prompts", "", f"""
        <label class="{SPATIAL_PREFIX}-field {SPATIAL_PREFIX}-grow">
          <span>Prompt</span>
          <textarea id="{_spatial_id("scene")}" rows="4"></textarea>
        </label>
        <label class="{SPATIAL_PREFIX}-field">
          <span>Literal +</span>
          <textarea id="{_spatial_id("literal", "plus")}" rows="2"></textarea>
        </label>
        <label class="{SPATIAL_PREFIX}-field">
          <span>Literal &#8722;</span>
          <textarea id="{_spatial_id("literal", "minus")}" rows="2"></textarea>
        </label>
        <label class="{SPATIAL_PREFIX}-check">
          <input type="checkbox" id="{_spatial_id("creative")}" />
          <span>Creative first</span>
        </label>""")}

      {_widget("person", "Person", "", f"""
        <div id="{_spatial_id("person")}" class="{SPATIAL_PREFIX}-person">
          {_palette(PERSON_SHAPES, "person")}
        </div>""")}

      {_widget("layers", "Layers",
               f'<span id="{_spatial_id("count")}" class="{SPATIAL_PREFIX}-tally"></span>',
               f"""
        <div id="{_spatial_id("list")}" class="{SPATIAL_PREFIX}-list" role="listbox"
             aria-label="Layers" tabindex="0"></div>
        <div class="{SPATIAL_PREFIX}-rowbar" role="group" aria-label="Layer actions">
          <button type="button" id="{_spatial_id("duplicate")}"
                  class="{SPATIAL_PREFIX}-tool">Duplicate</button>
          <button type="button" id="{_spatial_id("lower")}"
                  class="{SPATIAL_PREFIX}-tool">Back</button>
          <button type="button" id="{_spatial_id("front")}"
                  class="{SPATIAL_PREFIX}-tool">Front</button>
          <button type="button" id="{_spatial_id("delete")}"
                  class="{SPATIAL_PREFIX}-tool {SPATIAL_PREFIX}-danger">Delete</button>
        </div>""")}

      {_widget("inspector", "Inspector",
               f'<span id="{_spatial_id("selected-name")}"'
               f' class="{SPATIAL_PREFIX}-tally"></span>',
               f"""
        <div class="{SPATIAL_PREFIX}-pair">
          <label class="{SPATIAL_PREFIX}-field">
            <span>Name</span>
            <input type="text" id="{_spatial_id("name")}" />
          </label>
          <label class="{SPATIAL_PREFIX}-field">
            <span>Shape</span>
            <input type="text" id="{_spatial_id("shape")}" readonly />
          </label>
        </div>
        <label class="{SPATIAL_PREFIX}-field">
          <span>Type</span>
          <select id="{_spatial_id("type")}">
            <option value="{spatial.OBJECT}">Object</option>
            <option value="{spatial.TEXT}">Text</option>
          </select>
        </label>
        <label class="{SPATIAL_PREFIX}-field" id="{_spatial_id("text-field")}" hidden>
          <span>Text</span>
          <input type="text" id="{_spatial_id("text")}" />
        </label>
        <label class="{SPATIAL_PREFIX}-field {SPATIAL_PREFIX}-grow">
          <span id="{_spatial_id("prompt-label")}">Prompt</span>
          <textarea id="{_spatial_id("prompt")}" rows="4"></textarea>
        </label>
        <div class="{SPATIAL_PREFIX}-pair">
          <label class="{SPATIAL_PREFIX}-field">
            <span>Literal +</span>
            <input type="text" id="{_spatial_id("literal-prefix")}" />
          </label>
          <label class="{SPATIAL_PREFIX}-field">
            <span>Literal &#8722;</span>
            <input type="text" id="{_spatial_id("literal-suffix")}" />
          </label>
        </div>
        <div class="{SPATIAL_PREFIX}-pair">
          <label class="{SPATIAL_PREFIX}-field" id="{_spatial_id("framing-field")}">
            <span>Framing</span>
            <select id="{_spatial_id("framing")}">{_options(spatial.FRAMINGS, "Automatic")}</select>
          </label>
          <label class="{SPATIAL_PREFIX}-field" id="{_spatial_id("angle-field")}">
            <span>Camera angle</span>
            <select id="{_spatial_id("angle")}">{_options(spatial.ANGLES, "Automatic")}</select>
          </label>
        </div>
        <label class="{SPATIAL_PREFIX}-field {SPATIAL_PREFIX}-range"
               id="{_spatial_id("rotation-field")}" hidden>
          <span>Rotation <b id="{_spatial_id("rotation-read")}">0°</b></span>
          <input type="range" id="{_spatial_id("rotation")}"
                 min="-180" max="180" step="1" value="0" />
        </label>
        <label class="{SPATIAL_PREFIX}-check">
          <input type="checkbox" id="{_spatial_id("auto-hint")}" checked />
          <span>Position and size hints</span>
        </label>""")}

      {_widget("gallery", "Gallery",
               f'<span id="{_spatial_id("shot", "at")}"'
               f' class="{SPATIAL_PREFIX}-tally"></span>',
               f"""
        <div id="{_spatial_id("shot")}" class="{SPATIAL_PREFIX}-shot"></div>
        <div class="{SPATIAL_PREFIX}-rowbar" role="group" aria-label="Results">
          <button type="button" id="{_spatial_id("shot", "previous")}"
                  class="{SPATIAL_PREFIX}-tool">Previous</button>
          <button type="button" id="{_spatial_id("shot", "next")}"
                  class="{SPATIAL_PREFIX}-tool">Next</button>
          <button type="button" id="{_spatial_id("generate")}"
                  class="{SPATIAL_PREFIX}-tool {SPATIAL_PREFIX}-primary">Generate</button>
        </div>
        <div id="{_spatial_id("progress")}" class="{SPATIAL_PREFIX}-progress" hidden>
          <span id="{_spatial_id("progress", "bar")}"
                class="{SPATIAL_PREFIX}-progress-bar"></span>
          <span id="{_spatial_id("progress", "read")}"
                class="{SPATIAL_PREFIX}-progress-read"></span>
        </div>""")}

      {_widget("session", "Session", "", f"""
        <dl class="{SPATIAL_PREFIX}-facts">
          <dt>Frame</dt><dd id="{_spatial_id("fact", "frame")}"></dd>
          <dt>Ratio</dt><dd id="{_spatial_id("fact", "ratio")}"></dd>
          <dt>Regions</dt><dd id="{_spatial_id("fact", "regions")}"></dd>
          <dt>Pipeline</dt><dd id="{_spatial_id("fact", "pipeline")}"></dd>
          <dt>State</dt><dd id="{_spatial_id("fact", "state")}"></dd>
        </dl>
        <div class="{SPATIAL_PREFIX}-pair">
          <label class="{SPATIAL_PREFIX}-field">
            <span>Width</span>
            <input type="number" id="{_spatial_id("size", "width")}"
                   min="64" max="8192" step="8" />
          </label>
          <label class="{SPATIAL_PREFIX}-field">
            <span>Height</span>
            <input type="number" id="{_spatial_id("size", "height")}"
                   min="64" max="8192" step="8" />
          </label>
        </div>""")}
    </div>
  </div>
</div>'''


def spatial_compact() -> str:
    """The compact canvas: a frame, the regions in it, and one live line.

    Section 6.2 -- *position correction only*. There is no create, no delete,
    no resize, no rename, no Object/Text switch, no framing or angle, and no
    stacking control. Every one of those still exists, in the full editor, one
    button away.

    That restraint is the feature. The compact canvas answers the question
    somebody has while looking at the pipeline -- "that subject is slightly too
    far left" -- without opening a workspace, and a compact canvas that could
    also delete a region would be a second editor competing with the first over
    the same document.

    Empty of behaviour, like :func:`spatial_editor`: the browser file draws the
    regions into it, moves them, and writes the result back into the same
    hidden state box the full editor writes to. Stable ids and nothing a theme
    can rearrange.
    """
    return f'''
<div id="{_spatial_id("compact")}" class="{SPATIAL_PREFIX}-compact">
  <div id="{_spatial_id("compact", "frame")}" class="{SPATIAL_PREFIX}-compact-frame"
       role="application" tabindex="0"
       aria-label="Compact spatial layout — drag a region to move it">
    <div id="{_spatial_id("compact", "regions")}"
         class="{SPATIAL_PREFIX}-compact-regions"></div>
    <p id="{_spatial_id("compact", "empty")}" class="{SPATIAL_PREFIX}-compact-empty">
      No regions yet. Press Full Screen to draw one.</p>
  </div>
  <p id="{_spatial_id("compact", "note")}" class="{SPATIAL_PREFIX}-compact-note"
     role="status" aria-live="polite"></p>
</div>'''


def spatial_summary(serialized, enabled: bool = True, creative=None,
                    mode=None) -> str:
    """The one line under Spatial Layout: what is drawn, and what will happen.

    Rendered by the server so that a restored workflow and a freshly built page
    both say something true before any JavaScript has run, and repainted by the
    browser as the user draws. Two writers for one line is a thing to be careful
    about; the care is that both compute it from the same serialized layout.

    It describes the *pipeline*, not just the canvas, and that is why it takes
    the other feature's state. Spatial is a peer of Creative Mode now, so "your
    regions are applied" is only half an answer: the half that matters is
    whether the scene they are applied to is the prompt as typed or a paragraph
    a language model wrote, and that is Creative Mode's checkbox. ``creative``
    and ``mode`` default to the saved preferences so that a caller who has only
    the layout -- the browser's own repaint, a restore -- still gets a true
    line.
    """
    from prompt_master.krea import spatial

    if creative is None:
        creative = bool(mc_creative_krea.settings().get("enabled"))
    if mode is None:
        mode = mc_spatial.settings().get("compose_mode", spatial.SMART)

    layout = spatial.parse(serialized)
    if layout.unreadable:
        return notice(" ".join(layout.notes) or "The saved layout could not be read.",
                      "warn")

    pipeline = mc_krea_pipeline.described(creative=bool(creative),
                                          spatial=bool(enabled), mode=mode)
    if not layout.regions:
        return notice("No regions yet. Press Full Screen to draw one.", hint=pipeline)
    said = spatial.summarise(layout)
    if not enabled:
        return notice(f"{said} — Spatial Layout is off, so they are not applied.",
                      hint=pipeline)
    # The count is what changed since the last render; the rest of what this
    # line used to say is what the selected mode has meant since it was written.
    return notice(said, hint="Region prompts are used exactly as typed. " + pipeline)


def _spatial_toggled(enabled, serialized, creative, mode):
    """Remember the Spatial toggle, and say what it now does."""
    mc_spatial.remember(**{mc_spatial.ENABLED: bool(enabled)})
    return (spatial_summary(serialized, bool(enabled), creative=bool(creative),
                            mode=mode),
            mc_pipeline_panel.card_summary(
                "spatial", _spatial_line(serialized, bool(enabled), mode)),
            _literal_row(creative, enabled, other=creative))


def _spatial_mode(mode, serialized, enabled, creative):
    """Remember Smart or Direct, and say what the difference costs.

    The sentence is built from both toggles rather than written here, because
    the difference Smart makes is not the same difference with Creative Mode on
    as with it off: with the writer on it is a second request that reconciles a
    paragraph somebody else wrote, and with it off it is the only request there
    is and it reconciles the user's own sentence.
    """
    from prompt_master.krea import spatial

    mode = str(mode or "").strip().casefold()
    if mode not in spatial.COMPOSE_MODES:
        mode = spatial.SMART
    mc_spatial.remember(**{mc_spatial.COMPOSE_MODE: mode})
    return (spatial_summary(serialized, bool(enabled), creative=bool(creative),
                            mode=mode),
            mc_pipeline_panel.card_summary(
                "spatial", _spatial_line(serialized, bool(enabled), mode)))


def _spatial_saved(serialized, enabled, creative, mode):
    """The browser committed a layout. Keep it, and repaint what describes it.

    The *working* layout, and only that. A named layout is never written here,
    which is section 8.5: dragging a box with Auto Save on commits what the next
    Generate composes and leaves ``Studio thirds`` holding exactly what it held.
    The dirty line is how that is made visible rather than surprising.
    """
    mc_spatial.remember(**{mc_spatial.LAYOUT: str(serialized or "")})
    return (spatial_summary(serialized, bool(enabled), creative=bool(creative),
                            mode=mode),
            mc_pipeline_panel.card_summary(
                "spatial", _spatial_line(serialized, bool(enabled), mode)),
            _layout_state(None, serialized))


def _spatial_scenes(record):
    mc_spatial.remember(**{mc_spatial.RECORD_SCENES: bool(record)})
    return notice("The scene before and after the composer pass will be recorded in "
                  "each spatial image's metadata."
                  if record else
                  "The intermediate scenes will not be recorded. Smart and Direct "
                  "images can still be told apart by Krea Spatial Compose Mode, but "
                  "what the composer changed will not be recoverable afterwards.")


# --------------------------------------------------------------------------- #
# The two pipeline rows this script owns
# --------------------------------------------------------------------------- #


def _literal_row(creative, spatial, other=None):
    """Whether the Literal Prompt row is on screen. Section 5, and only that.

    Visible when either owned prompt-transforming feature is on, hidden when
    neither is. It is a statement about *relevance*, not about execution: the
    boxes are the place you go when a language model is about to rewrite your
    prompt, and the row would otherwise be two more things to scroll past on a
    tab where nothing is rewriting anything.

    What it is emphatically not is a switch. The values keep travelling with
    every generation while the row is hidden -- see :func:`_literals_for` --
    which is why :func:`mc_literal_prompts.active_note` exists and why the
    Prompt row of the Image Pipeline says how many are in effect.

    ``other`` is the stage that did *not* just change, and giving it turns this
    into "say something only if the answer moved". The row is visible when
    either stage is on, so flipping one of them changes the answer only when
    the other is off -- and re-sending a visibility Gradio already has is a
    component torn down and rebuilt for nothing, in the middle of the prompt
    area, which is a whole page reflowing so that nothing can change.
    """
    if other is not None and bool(other):
        return gr.update()
    return gr.update(visible=bool(creative) or bool(spatial))


def _creative_line(enabled=None, stored=None) -> str:
    """Creative's second line on the pipeline: ``C7 · 2 directions · Editorial``.

    Short by design. It is read at a glance while the stage is collapsed, and
    the three things worth a glance are how strongly the prompt is being
    directed, how many axes are being directed at all, and which named profile
    that came from.
    """
    stored = stored or mc_creative_krea.settings()
    if enabled is None:
        enabled = bool(stored.get("enabled"))
    if not enabled:
        return "Bypassed — prompt as-is"

    parts = [f"C{int(stored.get('creativity', 5))}"]
    directing = mc_creative_krea.active_axes(stored)
    rows = list(stored.get("directions") or ())
    if directing:
        parts.append(f"{len(directing)} direction"
                     f"{'' if len(directing) == 1 else 's'}")
    elif rows:
        # A row with an empty picker is a decision somebody started. Saying
        # "no directions" about it would be true of the brief and misleading
        # about the panel.
        parts.append(f"{len(rows)} direction"
                     f"{'' if len(rows) == 1 else 's'} with no treatments chosen")
    else:
        parts.append("nothing directed")

    try:
        profile = mc_creative_profiles.selected()
    except Exception:
        profile = ""
    if profile:
        parts.append(profile)
    return " · ".join(parts)


def _spatial_line(serialized=None, enabled=None, mode=None) -> str:
    """Spatial's second line: ``Smart · 2 regions · Studio thirds``."""
    from prompt_master.krea import spatial

    settings = mc_spatial.settings()
    if serialized is None:
        serialized = settings.get("layout", "")
    if enabled is None:
        enabled = bool(settings.get("enabled"))
    if mode is None:
        mode = settings.get("compose_mode", spatial.SMART)

    layout = spatial.parse(serialized)
    if layout.unreadable:
        return "⚠️ the saved layout could not be read"

    count = len(layout.regions)
    regions = f"{count} region{'' if count == 1 else 's'}" if count else "no regions"
    if not enabled:
        return f"Bypassed — {regions}"

    named = "Smart" if str(mode or "").strip().casefold() == spatial.SMART else "Direct"
    parts = [named, regions]
    loaded = settings.get("profile") or ""
    if loaded:
        parts.append(loaded)
    return " · ".join(parts)


# --------------------------------------------------------------------------- #
# Named spatial layouts
# --------------------------------------------------------------------------- #
#
# Section 8.5, which is the one place the loaded/modified contract needs saying
# twice. Two saves live next to each other here: Auto Save commits the *working
# layout*, which is what the next Generate composes, and Save updates a *named
# layout*, which is a copy somebody asked for. "Loaded: Studio thirds ·
# Modified · not saved" is an ordinary state to sit in -- the boxes just moved
# are the boxes that will be composed, and Studio thirds still holds what it
# held.


def _layout_state(name=None, serialized=None) -> str:
    """``Loaded: Studio thirds · Modified · not saved``, or nothing."""
    settings = mc_spatial.settings()
    if name is None:
        name = settings.get("profile") or ""
    if serialized is None:
        serialized = settings.get("layout", "")

    name = str(name or "").strip()
    if not name or name == mc_spatial_profiles.NONE:
        return ""
    if mc_spatial_profiles.get(name) is None:
        return ""

    modified = not mc_spatial_profiles.matches(name, serialized)
    line = mc_profile_state.describe(name, modified)
    explained = mc_profile_state.explain(modified)
    return f"{line}  \n{explained}" if explained else line


def _layout_chosen(name, enabled, creative, mode):
    """Load a named layout into the working canvas.

    The working layout is replaced, because that is what loading one means, and
    it is persisted at once so that closing the tab does not lose it. What is
    *not* touched is any other named layout: this reads the store and writes the
    canvas, never the other way round.
    """
    name = str(name or "").strip()
    if not name or name == mc_spatial_profiles.NONE:
        mc_spatial.remember(**{mc_spatial.PROFILE: ""})
        return gr.skip(), gr.skip(), "", gr.skip()

    serialized = mc_spatial_profiles.get(name)
    if serialized is None:
        return (gr.skip(), notice(f'There is no spatial layout called "{name}" any '
                                  "more — refresh the list.", "warn"),
                "", gr.skip())

    mc_spatial.remember(**{mc_spatial.LAYOUT: serialized, mc_spatial.PROFILE: name})
    return (gr.update(value=serialized),
            spatial_summary(serialized, bool(enabled), creative=bool(creative),
                            mode=mode),
            _layout_state(name, serialized),
            mc_pipeline_panel.card_summary(
                "spatial", _spatial_line(serialized, bool(enabled), mode)))


def _layout_saved(name, serialized):
    """Create or overwrite a named layout from the working canvas."""
    try:
        remaining = mc_spatial_profiles.save(name, serialized)
    except mc_spatial_profiles.LayoutError as exc:
        return notice(str(exc), "warn"), gr.skip(), gr.skip()

    kept = str(name or "").strip()
    mc_spatial.remember(**{mc_spatial.PROFILE: kept})
    return (notice(f'Saved the spatial layout "{kept}".'),
            gr.update(choices=[mc_spatial_profiles.NONE] + remaining, value=kept),
            _layout_state(kept, serialized))


def _layout_deleted(name, serialized, armed):
    """Remove a named layout, on the second press. The canvas is left as it is.

    §3 of the pipeline intent asks for an explicit confirmation where the loss
    is irreversible, and a named layout is a file. The confirmation is the
    button: the first press arms it and says which layout is about to go.
    """
    go, now, button = mc_pipeline_panel.confirmed(armed)
    if not go:
        return (now, button,
                notice(f'Press Delete again to remove the spatial layout "{name}". '
                       "This cannot be undone.", "warn"),
                gr.skip(), gr.skip())
    try:
        remaining = mc_spatial_profiles.delete(name)
    except mc_spatial_profiles.LayoutError as exc:
        return now, button, notice(str(exc), "warn"), gr.skip(), gr.skip()

    mc_spatial.remember(**{mc_spatial.PROFILE: ""})
    return (now, button,
            notice(f'Deleted the spatial layout "{name}". The boxes on screen are '
                   "unchanged."),
            gr.update(choices=[mc_spatial_profiles.NONE] + remaining,
                      value=mc_spatial_profiles.NONE),
            "")


def _layout_refreshed(current):
    available = mc_spatial_profiles.names()
    keep = current if current in available else mc_spatial_profiles.NONE
    return gr.update(choices=[mc_spatial_profiles.NONE] + available, value=keep)


def _auto_save_changed(value):
    """Remember whether a drag commits on release, and say what that means."""
    mc_spatial.remember(**{mc_spatial.AUTO_SAVE: bool(value)})
    if value:
        return notice("Moving a box on the compact canvas commits the working "
                      "layout as soon as you let go. It does not overwrite a named "
                      "layout — Save does that.")
    return notice("Moving a box changes the layout on screen only. Press Save "
                  "working layout to commit it, or switch Auto Save back on.")


# --------------------------------------------------------------------------- #
# Continuing from a pasted image
# --------------------------------------------------------------------------- #


def _pasted_view() -> str:
    """What the last paste said about Creative Mode, as one short block."""
    setup = mc_creative_krea.pasted.setup
    if setup is None or not setup.recorded:
        return ("Nothing yet. Paste an image made with Creative Mode — PNG Info, the "
                "arrow under the gallery, or a dropped file — and what it records "
                "appears here.")

    # An image can record Literal Prompt fields and nothing else: two boxes
    # filled in with both features switched off is an ordinary way to use this
    # extension. Its record is worth showing -- the paste emptied those boxes so
    # the picture would reproduce, and this is where somebody finds out what was
    # in them.
    if not setup.present:
        lines = ["This image recorded no Creative Mode setup."]
    else:
        lines = [f"Source prompt: {setup.source}" if setup.source else
                 "Source prompt: (not recorded)"]
    if setup.creativity is not None:
        lines.append(f"Creativity: {setup.creativity}")
    if setup.seed is not None:
        lines.append(f"Creative seed: {setup.seed}")
    if setup.recipe:
        lines.append(f"Recipe: {setup.recipe}")
    if setup.library_version:
        lines.append(f"Creativity library: {setup.library_version}")
    if setup.writer:
        lines.append(f"Written by: {setup.writer}")
    if setup.loras:
        # Read from an older image and shown, never restored. The control it
        # came from is gone; what is useful about it now is that somebody
        # continuing from that picture can see which tags it used and put them
        # back as literal commands, where they belong.
        lines.append(f"Pinned LoRAs (this build has no such field — type them into "
                     f"the prompt as [[{setup.loras}]]): {setup.loras}")
    if setup.literal_positive:
        lines.append(f"Literal Positive: {setup.literal_positive}")
    if setup.literal_negative:
        lines.append(f"Literal Negative: {setup.literal_negative}")
    if setup.literals:
        lines.append("The Literal Prompt boxes were emptied by the paste so this "
                     "picture reproduces exactly. Restore Creative setup puts them "
                     "back.")
    if setup.spatial:
        from prompt_master.krea import spatial as spatial_module

        drawn = spatial_module.parse(setup.spatial_layout)
        lines.append(f"{spatial_module.summarise(drawn)}"
                     f"{f' · {setup.spatial_compose_mode} merge' if setup.spatial_compose_mode else ''}")
        for region in drawn.ordered:
            body = (region.text if region.kind == spatial_module.TEXT
                    else region.source_prompt)
            lines.append(f"  [{region.identifier}] {list(region.bbox)} "
                         f"{region.kind}: {body}")
    for warning in setup.warnings():
        lines.append(f"Warning: {warning}")
    return "\n".join(lines)


def _restore_setup(replay_exactly):
    """Put a pasted image's Creative *workflow* back, on purpose and only here.

    The ordinary paste already restored the picture: the expanded prompt is in
    the prompt box and Creative Mode is off, so pressing Generate makes the same
    image again. This is the other thing somebody might want -- the short idea
    and the configuration behind it, to carry on from.

    So it overwrites the prompt box, which nothing else in this file does, and
    it says so. With ``replay_exactly`` it also arms the recorded recipe for
    exactly one generation, which is the only way to get the recorded art
    direction back verbatim: rolling again at the recorded seed re-derives the
    same *draw*, and the draw is weighted by a recent history that is not the
    history the original roll saw.
    """
    setup = mc_creative_krea.pasted.setup
    if setup is None or not setup.recorded:
        return (gr.update(), gr.update(),
                notice("There is no Creative setup from a pasted image to restore.",
                       "warn"),
                gr.update(), gr.update(), gr.update())

    # Two Literal Prompt boxes and nothing else is a whole restorable setup, so
    # it is handled before the Creative half rather than as a footnote to it.
    # These are put back rather than merely shown -- unlike the Pinned LoRAs
    # field above, the controls they came from still exist.
    fields = (gr.update(value=setup.literal_positive),
              gr.update(value=setup.literal_negative)) if setup.literals \
        else (gr.update(), gr.update())
    if not setup.present:
        mc_literal_prompts.remember(**{
            mc_literal_prompts.POSITIVE: setup.literal_positive,
            mc_literal_prompts.NEGATIVE: setup.literal_negative})
        return (gr.update(), gr.update(),
                notice("The Literal Prompt boxes are back as this image had them. "
                       "It recorded no Creative Mode setup, so nothing else changed."),
                gr.update(value=_pasted_view()), *fields)

    stored = mc_creative_krea.settings()
    remembered = {}
    if setup.creativity is not None:
        remembered[mc_creative_krea.CREATIVITY] = setup.creativity
    if setup.seed is not None:
        remembered[mc_creative_krea.SEED] = setup.seed
    if setup.anti_repetition is not None:
        remembered[mc_creative_krea.ANTI_REPETITION] = setup.anti_repetition
    if setup.axis_modes:
        remembered[mc_creative_krea.AXIS_MODES] = setup.axis_modes
        remembered[mc_creative_krea.FIXED_VALUES] = setup.fixed_values
        remembered[mc_creative_krea.EXCLUDED_VALUES] = setup.excluded_values
    if remembered:
        mc_creative_krea.remember(**remembered)
        stored = mc_creative_krea.settings()

    # Creative Mode goes back on, and this is the one place that is right. The
    # paste turned it off so the picture would reproduce; continuing from the
    # *source* is the opposite request, and a short idea generated with the
    # writer switched off is not a smaller version of this feature -- it is a
    # bare phrase handed to Krea 2.
    mc_creative_krea.remember(**{mc_creative_krea.ENABLED: True})

    said = ["Creative setup restored: the source prompt is back in the prompt box, the "
            "axes are as this image was made, and Creative Mode is on again."]
    if replay_exactly and setup.replayable:
        mc_creative_krea.replay.arm(mc_creative_krea.ReplayPlan(
            creativity=int(setup.creativity or stored["creativity"]),
            creative_seed=int(setup.seed if setup.seed is not None else -1),
            llm_seed=int(setup.llm_seed or 0),
            recipe=setup.recipe, library_version=setup.library_version,
            source=setup.source))
        said.append("The recorded recipe is armed for the next generation only — that "
                    "generation replays it exactly instead of rolling.")
    elif replay_exactly:
        said.append("No recipe was recorded in this image, so there is nothing to "
                    "replay; the next generation rolls normally.")
    else:
        said.append("Creative Mode will roll fresh art direction — this is a new roll "
                    "from the same idea, not the original.")
    said.extend(setup.warnings())

    # The canvas is *not* restored here any more, and that is the point of the
    # split. Spatial Layout is a peer feature with its own record and its own
    # button; a Creative restore that also turned Spatial on and overwrote the
    # canvas would be one feature reaching into another's state because they
    # happened to share a PNG. An image with both records has two buttons, and
    # pressing one is a decision about one of them.
    if setup.spatial:
        said.append("This image also recorded a spatial layout — Spatial Layout → "
                    "Continue from a pasted image restores the canvas.")

    if setup.literals:
        mc_literal_prompts.remember(**{
            mc_literal_prompts.POSITIVE: setup.literal_positive,
            mc_literal_prompts.NEGATIVE: setup.literal_negative})
        said.append("The Literal Prompt boxes are back as this image had them.")

    kind = "warn" if setup.warnings() or (replay_exactly and not setup.replayable) \
        else "info"
    told = notice(" ".join(said), kind)
    return (gr.update(value=setup.source) if setup.source else gr.update(),
            gr.update(value=True), told, gr.update(value=_pasted_view()), *fields)


def _spatial_pasted_view() -> str:
    """What a pasted image recorded about its *layout*, as text somebody reads.

    The Spatial half of the same captured record the Creative drawer shows. Its
    own view because a Spatial-only image has no Creative half to show, and
    printing "no Creative setup" over a perfectly good canvas would read as
    though the paste had failed.
    """
    from prompt_master.krea import spatial as spatial_module

    setup = mc_creative_krea.pasted.setup
    if setup is None or not setup.spatial:
        return "No spatial layout has been pasted in this session."

    layout = spatial_module.parse(setup.spatial_layout)
    lines = []
    if setup.spatial_version is not None and setup.spatial_version != spatial_module.VERSION:
        lines.append(f"Recorded with Spatial Layout version {setup.spatial_version}; "
                     f"this build reads version {spatial_module.VERSION}.")
    mode = str(setup.spatial_compose_mode or "").strip().casefold()
    if mode:
        lines.append(f"Composition: {mode} merge")
    lines.append(f"Frame: {layout.width} × {layout.height}" if layout.width
                 else "Frame: not recorded")
    lines.append(spatial_module.summarise(layout))
    lines.append("")
    for region in layout.ordered:
        body = region.text if region.kind == spatial_module.TEXT else region.prompt
        lines.append(f"  [{region.identifier}] {region.kind}: {body}")
    for note in layout.notes:
        lines.append(f"Note: {note}")
    return "\n".join(lines)


def _restore_spatial():
    """Put a pasted image's *layout* back, without touching Creative Mode.

    §17. The ordinary paste already restored the picture: the finished
    structured prompt is in the prompt box and Spatial Layout is off, so
    pressing Generate makes the same image again. This is the other thing
    somebody might want -- the canvas it was drawn on, to carry on from.

    It restores three things and nothing else: the layout, the compose mode, and
    Spatial Layout's own switch. Creative Mode's checkbox, the prompt box and
    every Creative setting are left exactly as they are, because a Spatial-only
    image has nothing to say about them and an image with both records has a
    second button that does.
    """
    from prompt_master.krea import spatial as spatial_module

    setup = mc_creative_krea.pasted.setup
    if setup is None or not setup.spatial:
        return (gr.update(), gr.update(), gr.update(),
                notice("There is no spatial layout from a pasted image to restore.",
                       "warn"))

    if setup.spatial_version is not None \
            and setup.spatial_version != spatial_module.VERSION:
        # Refused, not migrated. Exact replay never depended on this record --
        # the picture is reproducible from its own Prompt line -- so the honest
        # thing to do with a layout from another build is leave the canvas
        # alone and say so.
        return (gr.update(), gr.update(), gr.update(),
                notice("The recorded spatial layout is from a different version of "
                       "this feature and was not restored; your canvas is "
                       "untouched.", "warn"))

    remembered = {mc_spatial.LAYOUT: setup.spatial_layout, mc_spatial.ENABLED: True}
    mode = str(setup.spatial_compose_mode or "").strip().casefold()
    mode_update = gr.update()
    if mode in spatial_module.COMPOSE_MODES:
        remembered[mc_spatial.COMPOSE_MODE] = mode
        mode_update = gr.update(value=mode)
    mc_spatial.remember(**remembered)

    creative = bool(mc_creative_krea.settings().get("enabled"))
    said = ["The spatial canvas is back as it was drawn and Spatial Layout is on."]
    if not creative:
        said.append("Creative Mode was left off — your regions will be composed "
                    "around the prompt exactly as typed.")
    return (gr.update(value=setup.spatial_layout), gr.update(value=True), mode_update,
            notice(" ".join(said), hint=mc_krea_pipeline.described(
                creative=creative, spatial=True,
                mode=mode or mc_spatial.settings()["compose_mode"])))


def _image_seed(p) -> int:
    """The seed Forge will use for this image, or -1 if it has not settled one.

    Only ever the Spatial Composer's seed basis, and only on the path where no
    Creative roll supplies one -- see :func:`mc_spatial.composer_seed_for`.
    ``before_process`` runs before the host resolves ``-1``, so "not settled
    yet" is a real answer here rather than an error, and it is passed through
    as one.
    """
    try:
        return int(getattr(p, "seed", -1))
    except (TypeError, ValueError):
        return -1


def _say_what_ran(outcome, written) -> None:
    """One log line naming the pipeline that actually ran.

    §19, and the reason it is a function rather than a format string: the old
    line said "Creative Mode prompt applied" for everything this hook touched,
    which is now wrong for four of the six combinations and actively misleading
    for the two where no language model ran at all.
    """
    length = f"{len(written.generation):,}"
    if outcome.ran_creative:
        roll = outcome.roll
        logger.info("Model Chain: Creative Mode prompt written — %s characters from a "
                    "%s-character source at creativity %s, creative seed %s",
                    length, f"{len(roll.source):,}", roll.creativity,
                    roll.creative_seed)
        return
    if outcome.ran_spatial:
        logger.info("Model Chain: Spatial Layout prompt applied — %s merge over the "
                    "prompt as typed, %s characters of structured prompt",
                    outcome.merge, length)
        return
    # Neither pass produced anything and a prompt was still substituted, so the
    # literal commands are the whole of what happened to it. Said rather than
    # left silent: this is the one path where the prompt changed and no feature
    # will admit to having changed it.
    from prompt_master.krea import literals

    logger.info("Model Chain: the prompt carried %s and was restored around them",
                literals.describe(getattr(written, "literals", None)))


def _disarm_replay():
    mc_creative_krea.replay.clear()
    return notice("The armed replay was cleared; the next generation rolls normally.")


# --------------------------------------------------------------------------- #
# The script
# --------------------------------------------------------------------------- #


class ScriptKreaCreative(scripts.Script):
    """The Creative Mode controls, and the hook that writes the prompt."""

    def __init__(self):
        super().__init__()
        # Filled in by ui(); the shell never reads either, and a test asking
        # what the panel is made of and what it sends has something to ask.
        self.components: dict = {}
        self.arguments: list = []
        self.panel: mc_creative_panel.Panel | None = None
        # The native prompt box, handed over by after_component. Only the
        # restore action writes to it. See PROMPT_ELEM_ID.
        self.prompt_box = None
        # True while this hook is inside its own roll. A nested process_images()
        # -- Stage 2's, or any extension's -- must not start a second one, and
        # the cost of getting that wrong is not a duplicate image but an LLM
        # request that begins while the first is still streaming.
        self._rolling = False
        # Why this generation's prompt was not expanded, if it was not. Written
        # by the hook that tried and read by postprocess, which is the only
        # place a sentence can reach the person who pressed Generate.
        self._complaint = ""
        # The same, for the layout half: a Composer that did not run, a layout
        # that could not be read, a compositor that raised. Separate from the
        # complaint above because they are separate outcomes -- a spatial
        # generation can have a perfectly good prompt and no boxes in it, and
        # saying "Creative Mode did not write this prompt" about that would be
        # false as well as unhelpful.
        self._spatial_note = ""
        self._record_scenes = True
        # Whether the layout was composed onto the raw prompt because the writer
        # failed rather than because Creative Mode was off. Read only by
        # postprocess, to get one sentence right.
        self._composed_without_creative = False
        # This generation's negative prompt with its literal commands lifted
        # out, for Stage 2 to inherit instead of the restored one. Set at the
        # top of before_process, before anything can fail.
        self._inheritable_negative = ""

    def title(self):
        return "Krea Creative Mode"

    def show(self, is_img2img):
        # txt2img only. Creative Mode's whole shape is "the positive prompt is a
        # short idea and the image is made from an expansion of it", and
        # img2img's prompt describes an edit to a picture that already exists.
        return scripts.AlwaysVisible if not is_img2img else None

    def after_component(self, component, **kwargs):
        """Keep hold of the txt2img prompt box, and nothing else.

        The restore action has to write the recorded source phrase somewhere, and
        the prompt box is where a source phrase lives. Grabbed by the host's own
        element id rather than by position or by class, so a theme that rebuilds
        the page around it changes nothing here.
        """
        if kwargs.get("elem_id") == PROMPT_ELEM_ID:
            self.prompt_box = component

    # -- UI ---------------------------------------------------------------- #

    def ui(self, is_img2img):
        from prompt_master.krea import spatial as spatial_module
        from prompt_master.krea import variation

        stored = mc_creative_krea.settings()
        spatial = mc_spatial.settings()

        # A stage is armed for a session, never inherited from one. Both
        # switches come up off and the store is told so, which is the whole of
        # this change to the engine.
        #
        # It is a fix for a state the panel could not describe. The switch was
        # built from the saved preference and the card's description from the
        # placeholder, so a Creative Mode left on last week came back as a
        # checkbox reading ON above a line reading "Bypassed" -- and the engine
        # sided with the checkbox, so pressing Generate started a language model
        # for a stage the panel had just called bypassed.
        #
        # Off is the safe half of that disagreement to settle on. Stage 2's
        # switch has always been built this way; these two now match it, so all
        # three stages of the pipeline mean the same thing on a fresh page.
        # Arming any of them is one press, and it is a press somebody makes
        # while looking at the panel rather than one made for them by a
        # preference file they last touched days ago.
        if stored.get("enabled"):
            stored["enabled"] = False
            mc_creative_krea.remember(**{mc_creative_krea.ENABLED: False})
        if spatial.get("enabled"):
            spatial["enabled"] = False
            mc_spatial.remember(**{mc_spatial.ENABLED: False})

        # The shared shell. Whichever of the two feature scripts Forge builds
        # first creates it; this one fills the two stages it owns, wherever it
        # came in the order. See mc_pipeline_panel.
        pipeline = mc_pipeline_panel.host()

        # -- Creative ------------------------------------------------------- #

        with pipeline.head("creative"):
            enabled = mc_pipeline_panel.switch(elem_id=ident("toggle"))

        with pipeline.body("creative"):
            status = gr.HTML(notice("Creative Mode is off."),
                             elem_id=ident("status"))

            # A Group and no longer an Accordion, and always visible. The
            # disclosure is the stage card's now, and the switch is on it --
            # so the drawer is already open by the time anybody is looking at
            # this, and hiding its contents when the stage is off would mean
            # the only way to configure Creative Mode was to turn it on first.
            with gr.Group(elem_id=ident("controls")) as controls:
                # Profile, Creativity, and three of the four sibling drawers §3
                # asks for. The fourth is built below rather than in there,
                # because what it holds -- the pasted-image recovery, the last
                # roll, the help -- belongs to this surface and not to the panel
                # LLM Studio's Krea tab builds from the same function.
                panel = mc_creative_panel.build(ident, notice, status,
                                                stored=stored)
                if panel is not None:
                    creativity = panel.creativity
                else:
                    # No vocabulary, so no drawers and no directions. The slider
                    # still means something without a library, and losing it
                    # with the rest would take away the one Creative control
                    # that does.
                    creativity = gr.Slider(
                        label=variation.LABEL, minimum=variation.MINIMUM,
                        maximum=variation.MAXIMUM, step=1,
                        value=stored["creativity"], info=variation.HELP,
                        elem_id=ident("creativity"))

                # §3's fourth drawer, at the same level as the other three.
                with mc_pipeline_panel.drawer(
                        "Recovery & diagnostics", elem_id=ident("recovery"),
                        elem_classes=mc_pipeline_panel.classes("drawer")):
                    with mc_pipeline_panel.drawer("Continue from a pasted image", elem_id=ident("restore")):
                        gr.Markdown(
                            "Pasting an image made with Creative Mode restores its **final "
                            "expanded prompt** and turns Creative Mode off, so the picture "
                            "reproduces. This is the other half: the short idea it was "
                            "written from, and the settings behind it.")
                        pasted = gr.Textbox(
                            label="What the pasted image records", lines=6, max_lines=8,
                            interactive=False, show_copy_button=True,
                            value=_pasted_view(), elem_id=ident("pasted"))
                        exactly = gr.Checkbox(
                            value=True, label="Replay the recorded recipe exactly",
                            elem_id=ident("replay"),
                            info="one generation only; off rolls fresh direction from the "
                                 "same idea")
                        with gr.Row():
                            restore = gr.Button("Restore Creative setup", size="sm",
                                                variant="primary",
                                                elem_id=ident("restore", "apply"))
                            disarm = gr.Button("Clear armed replay", size="sm",
                                               elem_id=ident("restore", "clear"))

                    # Filled on request rather than streamed. The roll happens
                    # inside the generation now, where there is no open Gradio
                    # event to push anything down -- and a drawer nobody opened is
                    # the wrong thing to hold a websocket open for anyway. The
                    # button reads whatever the last roll left behind, which is
                    # still there after the tab has been closed and reopened.
                    with mc_pipeline_panel.drawer("Last creative roll", elem_id=ident("diagnostics")):
                        show = gr.Button("Show the last roll", size="sm",
                                         elem_id=ident("show"))
                        recipe = gr.Textbox(
                            label="Recipe and brief", lines=12, max_lines=12,
                            interactive=False, show_copy_button=True,
                            elem_id=ident("recipe"))
                        expanded = gr.Textbox(
                            label="Expanded Krea prompt", lines=6, max_lines=6,
                            interactive=False, show_copy_button=True,
                            elem_id=ident("expanded"))

                    with mc_pipeline_panel.drawer("How Creative Mode reads your prompt", elem_id=ident("help")):
                        gr.Markdown(
                            "**Literal commands.** Anything you write inside `[[double "
                            "brackets]]` is lifted out of the prompt before any language "
                            "model sees it and put back at the end, on its way to Forge's "
                            "own prompt processing — so LoRA tags, wildcards, `$styles`, "
                            "another extension's syntax and instructions about your "
                            "ImageStitch reference images all arrive exactly as you typed "
                            "them. `[[<lora:krea2_edit:1>]]` goes in front of the written "
                            "prompt; `-[[__grain__]]` goes after it. Written inside a "
                            "region's prompt on the Spatial canvas, a command stays with "
                            "that region and reaches that element of the composition.\n\n"
                            "**Directions.** An axis with no row is left out of the brief "
                            "entirely — the model decides as it would without Creative "
                            "Mode. Add a direction and choose the treatments you are "
                            "willing to use: **one** treatment repeats every roll, "
                            "**several** let the Creative seed choose between them, and "
                            "the Creativity slider decides how strongly the choice is "
                            "expressed and how hard recent ones are pushed away. A row "
                            "with nothing chosen directs nothing.\n\n"
                            "Your own words always win. Type *oil painting of a car* and "
                            "Medium stays oil painting however Medium is set.\n\n"
                            "Creative Mode changes the positive prompt only. The negative "
                            "prompt, the checkpoint, the sampler, the size, Steps, the "
                            "image seed and every other setting stay exactly where Forge "
                            "puts them, and the image itself is generated by Forge.")

        # -- Spatial -------------------------------------------------------- #
        #
        # A peer of Creative Mode and not a child of it. It used to live inside
        # Creative's group, visible only while Creative Mode was on, which made
        # it a mode of Creative rather than a feature of its own. Two things
        # follow from it being its own stage, and both are the point: it is
        # usable with Creative Mode off, and it is built whether or not the
        # creativity library loaded. That second one is not tidiness -- the
        # deterministic compositor needs no vocabulary at all, so an
        # installation whose Creative Mode cannot run can still place boxes
        # around the prompt somebody typed.

        with pipeline.head("spatial"):
            spatial_enabled = mc_pipeline_panel.switch(
                elem_id=ident("spatial", "toggle"))

        with pipeline.body("spatial"):
            with gr.Group(elem_id=ident("spatial", "layout")):
                gr.Markdown(
                    mc_hint.beside(
                        "**Spatial Layout**",
                        "Draw regions on the canvas and each one's prompt is "
                        "placed where you drew it. Smart Spatial Compose sends the "
                        "scene to the composer first; Direct BBOX Merge applies "
                        "your regions deterministically with no language-model "
                        "request."),
                    elem_id=ident("spatial", "heading"))

                # The remembered name only if it still names something. A layout
                # deleted in another tab -- or a store replaced wholesale --
                # would otherwise leave the dropdown claiming a composition is
                # loaded when the file it came from is gone.
                layout_choices = mc_spatial_profiles.choices()
                loaded_layout = (spatial["profile"]
                                 if spatial["profile"] in layout_choices
                                 else mc_spatial_profiles.NONE)

                # §3 asks for the two composition modes as large segmented
                # targets rather than two radio dots. It is still the same
                # stock Radio -- the same component, the same handler, the same
                # value -- wearing the shape: the class turns the group into
                # two full-height halves in the stylesheet and nothing about
                # what it sends changes.
                spatial_compose = gr.Radio(
                    choices=[("Smart Spatial Compose", spatial_module.SMART),
                             ("Direct BBOX Merge", spatial_module.DIRECT)],
                    value=spatial["compose_mode"], label="Composition",
                    elem_id=ident("spatial", "compose"),
                    elem_classes=mc_pipeline_panel.classes("segmented"))

                # Position correction, in the pipeline, without opening a
                # workspace. Section 6.2: drag the topmost box under the
                # pointer and nothing else.
                gr.HTML(spatial_compact(), elem_id=_spatial_id("compact", "host"))

                with gr.Row(elem_id=ident("spatial", "actions")):
                    spatial_auto_save = gr.Checkbox(
                        value=bool(spatial["auto_save"]), label="Auto Save",
                        elem_id=_spatial_id("autosave"), scale=1,
                        info="saves every finished edit")
                    mc_hint.control(
                        "One switch for both canvases: this one and the full "
                        "Full Screen editor. On, every finished edit is saved "
                        "as it happens -- a box let go of after a move or a "
                        "resize, a region added, deleted or reordered, an undo. "
                        "A text field saves when the cursor leaves it, not on "
                        "every keystroke. Off, nothing is written until Save "
                        "working layout.",
                        label="Auto Save", elem_id=_spatial_id("autosave", "hint"))
                    spatial_undo = gr.Button("Undo", size="sm", scale=1,
                                             elem_id=_spatial_id("compact", "undo"))
                    spatial_commit = gr.Button("Save working layout", size="sm",
                                               scale=1,
                                               elem_id=_spatial_id("compact", "commit"))
                    edit = gr.Button("Full Screen", size="sm", scale=1,
                                     variant="primary", elem_id=_spatial_id("open"))

            spatial_status = gr.HTML(
                spatial_summary(spatial["layout"], bool(spatial["enabled"]),
                                creative=bool(stored["enabled"]),
                                mode=spatial["compose_mode"]),
                elem_id=ident("spatial", "status"))

            # The one component the browser writes to, and the one that travels
            # with the generation. Hidden rather than absent: the editor is a
            # page, the compositor is a hook, and a hidden textbox is the only
            # thing Gradio offers that is both.
            spatial_state = gr.Textbox(
                value=spatial["layout"], visible=False, lines=1,
                elem_id=_spatial_id("state"))

            # Saved compositions, one drawer down. They were the first thing
            # in this panel, above the canvas -- which had the same shape of
            # mistake Stage 2's checkpoint did: a layout is loaded once at the
            # start of a session and then left alone, while the canvas under it
            # is what somebody actually works in. The canvas keeps the top of
            # the panel; this is where you go to keep what is on it.
            with mc_pipeline_panel.drawer("Saved layouts", elem_id=ident("spatial", "profiles")):
                with gr.Row():
                    spatial_profile = gr.Dropdown(
                        label="Layout", value=loaded_layout,
                        choices=layout_choices, scale=3,
                        filterable=False, elem_id=ident("spatial", "profile"),
                        info="a saved composition; loading one replaces the boxes "
                             "on the canvas")
                    spatial_profile_refresh = ToolButton(
                        value=refresh_symbol,
                        elem_id=ident("spatial", "profile", "refresh"),
                        tooltip="Spatial layouts: refresh")

                spatial_profile_state = gr.Markdown(
                    _layout_state(loaded_layout, spatial["layout"]),
                    elem_id=ident("spatial", "profile", "state"))

                with gr.Row():
                    spatial_profile_name = gr.Textbox(
                        label="Layout name", scale=3, max_lines=1,
                        placeholder="Studio thirds",
                        elem_id=ident("spatial", "profile", "name"))
                    spatial_profile_save = gr.Button(
                        "Save", size="sm", scale=1,
                        elem_id=ident("spatial", "profile", "save"))
                    spatial_profile_delete = gr.Button(
                        "Delete", size="sm", scale=1, variant="stop",
                        elem_id=ident("spatial", "profile", "delete"))
                # Whether Delete is armed. A gr.State and not a module variable:
                # an arm is one person's half-finished gesture in one browser.
                arm_layout_delete = gr.State(False)

            with mc_pipeline_panel.drawer("Spatial options", elem_id=ident("spatial", "options")):
                record_scenes = gr.Checkbox(
                    value=bool(spatial["record_scenes"]),
                    label="Record the scene before and after the composer pass",
                    elem_id=ident("spatial", "scenes"),
                    info="Smart merges only; makes a Smart/Direct comparison "
                         "readable afterwards")

                # Spatial's own recovery, and not a corner of Creative's.
                # §17: a Spatial-only image has no Creative record to restore
                # from, and asking somebody to look under Creative Controls for
                # the canvas of an image that never used Creative Mode is the
                # coupling this refactor removed, left in the furniture.
                with mc_pipeline_panel.drawer("Continue from a pasted image", elem_id=ident("spatial", "restore")):
                    gr.Markdown(
                        "Pasting an image made with Spatial Layout restores its "
                        "**final structured prompt** and turns Spatial Layout off, so "
                        "the picture reproduces. This is the other half: the canvas it "
                        "was composed on.")
                    spatial_pasted = gr.Textbox(
                        label="What the pasted image recorded about its layout",
                        lines=8, max_lines=10, interactive=False,
                        show_copy_button=True, value=_spatial_pasted_view(),
                        elem_id=ident("spatial", "pasted"))
                    restore_spatial = gr.Button(
                        "Restore Spatial setup", size="sm",
                        elem_id=ident("spatial", "restore", "apply"))

        # -- the full editor, deliberately outside the pipeline -------------- #
        #
        # Section 6.5 and section 10: the full BBOX workspace keeps its
        # dedicated large area. It is a block in ordinary document flow that
        # Full Screen reveals and Close hides, and it belongs at the top level
        # rather than nested three containers deep inside an accordion -- a
        # workspace inside a drawer is one `overflow: hidden` away from being a
        # window nobody can see. It is hidden until opened, so it costs no
        # space in the pipeline and competes with nothing.
        gr.HTML(spatial_editor(), elem_id=_spatial_id("editor"))

        # -- the Literal Prompt row ------------------------------------------ #
        #
        # Two ordinary prompt boxes for text no language model may rewrite, and
        # the whole of the feature that used to require typing [[...]].
        #
        # Built here and *moved* by the browser to sit under the native Negative
        # Prompt, because Forge offers an extension no way to build a component
        # into the prompt area: `ui()` runs inside the script accordion, which
        # is several hundred pixels below the box these two belong beside. The
        # move is presentation only and is allowed to fail -- see
        # javascript/model_chain_literals.js. If it does, the row stays where
        # Gradio put it and every one of these controls still works.
        #
        # Real Textboxes with the host's own `prompt` class, so Tag
        # Autocomplete, LoRA completion and anything else that looks for a
        # prompt box finds two more of them rather than two impostors.
        #
        # One above the other, always, and a `gr.Column` because that is
        # Gradio's word for it rather than a Row talked out of laying out
        # sideways. Two boxes that shared a line while they fitted was the
        # responsive behaviour asked for first; what it produced was a row
        # whose height changed as the divider moved, in a prompt column that
        # holds its own leftover space, so the empty space under it grew and
        # shrank while somebody was dragging. Stacked, there is one width and
        # one height and nothing about the column below to think about.
        #
        # No `scale`, no `min_width`: nothing here needs a width to decide
        # anything, and a component that states a minimum is a component the
        # column around it has to be at least that wide for.
        #
        # The id still says `row`, because it is the name Python, the browser
        # file and every test address this element by. It is a stack.
        #
        # Label and field, and nothing else: no `info=` copy and no
        # instructional placeholder. This sits in the prompt area, where four
        # boxes of explanatory prose is three too many.
        literal = mc_literal_prompts.settings()
        with gr.Column(elem_id=ident("literal", "row"),
                       visible=bool(stored["enabled"]) or bool(spatial["enabled"]),
                       elem_classes=["mc-literal-row"]) as literal_row:
            literal_positive = gr.Textbox(
                label="Positive Literal", lines=2, max_lines=4,
                value=literal["positive"], elem_id=ident("literal", "positive"),
                elem_classes=["prompt", "mc-literal-box"])
            literal_negative = gr.Textbox(
                label="Negative Literal", lines=2, max_lines=4,
                value=literal["negative"], elem_id=ident("literal", "negative"),
                elem_classes=["prompt", "mc-literal-box"])

        self.panel = panel
        self.components = {
            "enabled": enabled, "creativity": creativity, "status": status,
            "controls": controls, "show": show, "recipe": recipe, "expanded": expanded,
            "pasted": pasted, "replay": exactly, "restore": restore, "disarm": disarm,
            "spatial_enabled": spatial_enabled,
            "spatial_compose": spatial_compose, "spatial_status": spatial_status,
            "spatial_state": spatial_state, "spatial_edit": edit,
            "spatial_scenes": record_scenes, "spatial_pasted": spatial_pasted,
            "spatial_restore": restore_spatial,
            "spatial_profile": spatial_profile,
            "spatial_profile_state": spatial_profile_state,
            "spatial_profile_name": spatial_profile_name,
            "spatial_auto_save": spatial_auto_save,
            "spatial_undo": spatial_undo, "spatial_commit": spatial_commit,
            "creative_line": pipeline.summary("creative"),
            "spatial_line": pipeline.summary("spatial"),
            "literal_row": literal_row,
            "literal_positive": literal_positive,
            "literal_negative": literal_negative}
        if panel is not None:
            self.components.update(panel.components())

        self._wire(enabled, creativity, status, show, recipe, expanded,
                   pasted, exactly, restore, disarm)
        self._wire_spatial(spatial_enabled, spatial_compose, spatial_status,
                           spatial_state, record_scenes, restore_spatial,
                           enabled)
        self._wire_literals(literal_positive, literal_negative)
        self._wire_layouts(spatial_profile, spatial_profile_refresh,
                           spatial_profile_state, spatial_profile_name,
                           spatial_profile_save, spatial_profile_delete,
                           arm_layout_delete,
                           spatial_auto_save, spatial_state, spatial_enabled,
                           spatial_compose, spatial_status, enabled)
        self._register_paste_fields()

        # What the two pipeline rows say before anybody has touched anything.
        # Gradio fires no event at page load, so the first render is written
        # into the components themselves, which works because the config the
        # browser is built from is generated after every ui() has run.
        try:
            if pipeline.summary("creative") is not None:
                pipeline.summary("creative").label = mc_pipeline_panel.card_label(
                    "creative", _creative_line(bool(stored["enabled"]), stored))
            if pipeline.summary("spatial") is not None:
                pipeline.summary("spatial").label = mc_pipeline_panel.card_label(
                    "spatial", _spatial_line(
                        spatial["layout"], bool(spatial["enabled"]),
                        spatial["compose_mode"]))
        except Exception:
            logger.debug("Model Chain: could not pre-render the pipeline rows",
                         exc_info=True)

        # Every control travels to before_process, because that is where the
        # roll happens and the panel is what the user is looking at. They are
        # this script's own arguments and reach neither Model Chain's preset
        # list nor its infotext.
        #
        # The Spatial controls go last and stay last. _split() cuts this tuple
        # by asking the library how long the axis block is, so the two fixed
        # ends are the two that can be read without counting.
        # The Spatial tail travels on every shape of this list, including the
        # one a missing creativity library produces. Spatial does not need the
        # library, so a page that could not build the axis controls still has to
        # be able to send a layout. _split() knows both shapes.
        # The Literal Prompt boxes go in the middle, before the Spatial tail.
        # That looks backwards and is the only placement that works: mc_plan
        # reads the Spatial controls off the *end* of this tuple, so appending
        # would have moved the end out from under it. The two ends stay the two
        # ends, and the new block joins the variable middle that only _split()
        # reads -- which knows how long the axis block is because it asks.
        spatial_controls = [spatial_enabled, spatial_compose, spatial_state]
        literal_controls = [literal_positive, literal_negative]
        if panel is None:
            self.arguments = ([enabled, creativity] + literal_controls
                              + spatial_controls)
        else:
            self.arguments = ([enabled, creativity] + list(panel.settings_controls)
                              + list(panel.axis_controls) + literal_controls
                              + spatial_controls)
        return list(self.arguments)

    def _wire_literals(self, positive, negative) -> None:
        """Keep what the two Literal Prompt boxes hold. Two handlers, no more.

        ``blur`` and not ``change``: a Textbox's change event fires as somebody
        types, and a round trip per keystroke is not a thing to do to a prompt
        box. Nothing is waiting on this write -- the value that travels with a
        generation is whatever the component holds when Generate is pressed, so
        a press before the box has been left still uses what is on screen. This
        is only what makes it survive a restart.

        And no outputs, for the same reason one level down: a handler that
        answers is a handler the browser has to apply an update for.
        """
        def keep_positive(value):
            mc_literal_prompts.remember(**{mc_literal_prompts.POSITIVE:
                                           str(value or "")})

        def keep_negative(value):
            mc_literal_prompts.remember(**{mc_literal_prompts.NEGATIVE:
                                           str(value or "")})

        # No outputs. Answering with a `gr.update()` sent the box a value it
        # already had, and every component update the browser applies is a UI
        # update every `onAfterUiUpdate` handler on the page then runs -- ours
        # and every other extension's -- for a write nothing was waiting on.
        positive.blur(fn=keep_positive, inputs=[positive], outputs=[],
                      queue=False, show_progress=False)
        negative.blur(fn=keep_negative, inputs=[negative], outputs=[],
                      queue=False, show_progress=False)

    def _wire_layouts(self, profile, refresh, state_line, name, save, delete,
                      arm_layout_delete, auto_save, spatial_state, spatial_enabled,
                      spatial_compose, spatial_status, creative_enabled) -> None:
        """Named layouts, and the Auto Save switch. Six handlers, none of them
        touching a layout the user did not name.

        The dirty line is recomputed from the two documents rather than tracked,
        for the same reason the Creative panel recomputes its own: a flag that
        is derived cannot drift out of step with the thing it describes, and the
        thing it describes here is written by a browser file that this side does
        not hear from between presses.
        """
        profile.change(
            fn=_layout_chosen,
            inputs=[profile, spatial_enabled, creative_enabled, spatial_compose],
            outputs=[spatial_state, spatial_status, state_line,
                     self.components["spatial_line"]],
            queue=False, show_progress=False)
        refresh.click(fn=_layout_refreshed, inputs=[profile], outputs=[profile],
                      queue=False, show_progress=False)
        save.click(fn=_layout_saved, inputs=[name, spatial_state],
                   outputs=[spatial_status, profile, state_line],
                   queue=False, show_progress=False)
        delete.click(fn=_layout_deleted,
                     inputs=[profile, spatial_state, arm_layout_delete],
                     outputs=[arm_layout_delete, delete,
                              spatial_status, profile, state_line],
                     queue=False, show_progress=False)
        auto_save.change(fn=_auto_save_changed, inputs=[auto_save],
                         outputs=[spatial_status], queue=False,
                         show_progress=False)

    def _wire(self, enabled, creativity, status, show, recipe, expanded,
              pasted, exactly, restore, disarm):
        """Every handler this file owns, in one place. The panel wires its own.

        All of them ``queue=False``: not one of these does any work worth
        queueing, and none of them starts, stops or waits for a generation. The
        panel is settings and nothing else now -- the roll is in
        :meth:`before_process`, which is not reachable from here.
        """
        # Creative's toggle no longer shows or hides Spatial -- that coupling is
        # the whole of what this refactor removed. What it still does is repaint
        # Spatial's status line, because the pipeline that line describes has
        # Creative Mode's state in it.
        enabled.change(fn=_toggled,
                       inputs=[enabled, self.components["spatial_enabled"],
                               self.components["spatial_state"],
                               self.components["spatial_compose"]],
                       outputs=[status, self.components["spatial_status"],
                                self.components["creative_line"],
                                self.components["spatial_line"],
                                self.components["literal_row"]],
                       queue=False, show_progress=False)

        # The slider moves what the brief costs as well as what it says, and the
        # cost line is the thing somebody looks at straight after moving it. Sent
        # together so the two cannot disagree by one action.
        if self.panel is not None:
            creativity.release(
                fn=lambda value: (_remember_creativity(value),
                                  gr.update(value=mc_creative_panel.describe_cost()),
                                  mc_pipeline_panel.card_summary(
                                      "creative", _creative_line())),
                inputs=[creativity],
                outputs=[status, self.panel.cost, self.components["creative_line"]],
                queue=False, show_progress=False)
        else:
            creativity.release(
                fn=lambda value: (_remember_creativity(value),
                                  mc_pipeline_panel.card_summary(
                                      "creative", _creative_line())),
                inputs=[creativity],
                outputs=[status, self.components["creative_line"]], queue=False, show_progress=False)
        show.click(fn=_last_roll, outputs=[recipe, expanded], queue=False,
                   show_progress=False)

        # The one handler in this extension that writes to a native control, and
        # the only one that ever should: it is a button whose entire purpose is
        # to put a recorded source phrase back where source phrases are typed.
        #
        # Without the prompt box -- a host that renders the accordion before the
        # prompt, or a test with no page at all -- the restore still restores the
        # settings and still says so; only the phrase has nowhere to go, and it
        # is in the record above for copying.
        literal_boxes = [self.components["literal_positive"],
                         self.components["literal_negative"]]
        if self.prompt_box is not None:
            restore.click(fn=_restore_setup, inputs=[exactly],
                          outputs=[self.prompt_box, enabled, status, pasted,
                                   *literal_boxes],
                          queue=False, show_progress=False)
        else:
            logger.debug("Model Chain: the txt2img prompt box was not offered to "
                         "Creative Mode; Restore Creative setup will not fill it in")
            restore.click(fn=lambda exactly: _restore_setup(exactly)[1:],
                          inputs=[exactly],
                          outputs=[enabled, status, pasted, *literal_boxes],
                          queue=False, show_progress=False)
        disarm.click(fn=_disarm_replay, outputs=[status], queue=False, show_progress=False)

    def _wire_spatial(self, spatial_enabled, spatial_compose, spatial_status,
                      spatial_state, record_scenes, restore_spatial,
                      creative_enabled):
        """The Spatial controls' handlers. Five, and none of them generates.

        The state box is the whole of the browser-to-server channel, and it is a
        channel in one direction only and only between presses: the editor writes
        the serialized document into it when somebody saves a layout, Gradio's
        own change event carries that to Python, and Python persists it and
        repaints the summary line. Nothing polls it, nothing waits for it, and no
        generation is held up by it -- press Generate and close the tab and the
        layout that was in the box when the request left is the layout that gets
        composed.

        That is the difference between this box and the one the old Creative
        gate had, which looked the same and was not: that one was *polled* by a
        setInterval that had to see a token appear in it before an image could
        start, so a throttled tab made a generation late and a closed one made it
        never happen. Here the box is an input, like the slider.

        ``change`` rather than ``input``: the value arrives from JavaScript, not
        from a keystroke. The feedback loop the Creative panel avoids by using
        ``input`` cannot form here because no handler on this box writes back to
        it -- and the server *does* write to it, on a workflow restore, where
        having the restored layout persist itself is exactly right.
        """
        line = self.components["spatial_line"]
        spatial_enabled.change(
            fn=_spatial_toggled,
            inputs=[spatial_enabled, spatial_state, creative_enabled, spatial_compose],
            outputs=[spatial_status, line, self.components["literal_row"]],
            queue=False, show_progress=False)
        spatial_compose.change(
            fn=_spatial_mode,
            inputs=[spatial_compose, spatial_state, spatial_enabled, creative_enabled],
            outputs=[spatial_status, line], queue=False, show_progress=False)
        spatial_state.change(
            fn=_spatial_saved,
            inputs=[spatial_state, spatial_enabled, creative_enabled, spatial_compose],
            outputs=[spatial_status, line,
                     self.components["spatial_profile_state"]], queue=False, show_progress=False)
        record_scenes.change(fn=_spatial_scenes, inputs=[record_scenes],
                             outputs=[spatial_status], queue=False,
                             show_progress=False)
        # Spatial's own restore, writing only to Spatial's own controls. Nothing
        # in this list is Creative Mode's, which is the whole difference between
        # this button and the one in the other section.
        restore_spatial.click(
            fn=_restore_spatial,
            outputs=[spatial_state, spatial_enabled, spatial_compose, spatial_status],
            queue=False, show_progress=False)

    def _register_paste_fields(self):
        """Make an ordinary paste reproduce the image rather than re-expand it.

        This is the fix the whole Creative infotext story turns on. The recorded
        ``Prompt:`` line of a Creative image is the *expanded* prompt -- Creative
        Mode assigned it before Forge wrote the infotext -- so restoring it with
        Creative Mode still on would send an already-written Krea paragraph back
        to the writer as though it were a short idea. The picture that came out
        would be a picture of the prompt of the picture.

        So the enabled checkbox is a paste field that answers False for any
        infotext carrying Creative Mode's own key. Everything else about the
        paste is the host's: the prompt, the seed, the checkpoint, the sampler
        and the size are restored exactly as they always were.
        """
        try:
            self.infotext_fields = mc_infotext.build_creative_paste_fields(
                self.components, notice=notice, view=_pasted_view,
                spatial_view=_spatial_pasted_view)
            self.paste_field_names = mc_infotext.creative_paste_field_names()
        except Exception:
            errors.report("Model Chain: failed to register the Creative Mode paste "
                          "fields", exc_info=True)

    # -- generation --------------------------------------------------------- #

    def before_process(self, p, enabled=False, *args, **kwargs):
        """Write this generation's prompt, then substitute it.

        ``before_process`` and not ``process``: this runs before Forge builds
        ``all_prompts`` from ``p.prompt``, so one assignment reaches the batch,
        the styles pass, the infotext and Stage 2's inherited prompt without any
        of them having to be told about Creative Mode. It also runs before the
        checkpoint is loaded, which is what lets the writer size itself against
        a card the image model has not taken yet.

        Everything one press needs happens here, on the thread the host is
        already running the job on. Nothing is waited for in a browser and
        nothing is picked up from one: close the tab after pressing Generate and
        this still finishes, and Forge writes the files.

        Failure is always "generate what the user typed". A library that will not
        load, a checkpoint that is not Krea 2, a language model that will not
        answer, an Interrupt during the roll -- none of them is a reason to
        refuse a generation the user asked for, and all of them say so in the
        log.

        Nothing is carried in from before the press except an armed replay, which
        is a list of variant ids the user explicitly asked to reuse, is visible on
        the panel while it is armed, and is spent by the first generation that
        runs.

        Two features, one gate
        ----------------------
        This used to return immediately unless Creative Mode was on, which made
        Spatial Layout unreachable without it. The gate is now "did the user
        switch on either of them, or write a literal command", and everything
        after it is :mod:`mc_krea_pipeline` deciding the order. An ordinary
        generation -- both switched off, no ``[[`` in the prompt -- still leaves
        on the first line, which is what keeps this feature free for everybody
        who does not use it.

        The third reason is the one that costs a substring search on every
        generation, and it buys the property that makes the syntax teachable:
        ``[[...]]`` means the same thing whether or not a language model was
        going to run. A user who switches Creative Mode off for one image does
        not discover that their reference instruction has started reaching the
        text encoder with its brackets on.
        """
        import mc_broker

        if self._rolling:
            # A process_images() nested inside our own run. There is nothing to
            # do for it and a great deal to get wrong -- including clearing the
            # notes the outer run has already written, which is why this is the
            # first thing here and not the third.
            logger.debug("Model Chain: the Krea pipeline is already running; the "
                         "nested generation is left alone")
            return

        from prompt_master.krea import literals

        self._complaint = ""
        self._spatial_note = ""
        self._composed_without_creative = False

        # The negative prompt first, and on its own. No language model in this
        # extension has ever seen it, so there is nothing to protect it from --
        # what it needs is the same syntax honoured in the same way, so that a
        # wildcard written ``[[__grain__]]`` in one box behaves as it does in
        # the other. It is restored here and not in the pipeline because the
        # pipeline is about the positive prompt and should stay that way.
        self._restore_negative(p)

        # One parse, one merge, and every path below reads the result. The two
        # Literal Prompt boxes become commands here and nowhere else -- section
        # 11 of the Literal Prompts intent, and the reason the fields cannot
        # develop a second prompt path of their own: by the time anything
        # downstream sees them they are LiteralCommands in the sidecar the
        # parser already produced, indistinguishable from a [[...]] somebody
        # typed and restored by the same single assembly step.
        #
        # Before the "neither feature is on" check, deliberately. A field that
        # only worked while Creative or Spatial was running would be a field
        # whose row is hidden exactly when it stops working, which is the
        # opposite of section 3.3.
        before, after = _literals_for(args)
        parsed = literals.merge(
            literals.parse(getattr(p, "prompt", "") or ""), before, after)
        self._record_literal_fields(p, before, after)
        layout = self._layout(p, args)
        creative = bool(enabled)
        if not creative and not getattr(layout, "regions", ()):
            # Neither feature is on. If the prompt carries literal commands they
            # still have to come off it; if it does not, this is the ordinary
            # generation that leaves here having done nothing at all.
            self._literals_only(p, parsed)
            return

        settings = _settings_for(args)
        self._publish_plan(p, layout, creative)

        # The Krea 2 checkpoint guard, asked once for the whole pipeline rather
        # than by whichever feature happened to own it. Creative Mode's roll
        # asks the same question again for itself; what changed is that a
        # Spatial-only generation asks it too, because Direct BBOX Merge hands
        # Krea's structured JSON to the loaded checkpoint without ever calling a
        # language model, and that prompt is no more readable by SD 1.5 for
        # having been built deterministically.
        objection = mc_krea_pipeline.objection() if not creative else ""
        if objection:
            logger.warning("Model Chain: Spatial Layout was not applied — %s", objection)
            self._spatial_note = objection
            # The layout is refused and the literals are not. They are ordinary
            # Forge prompt syntax that the user typed into a prompt box, and a
            # checkpoint this feature will not build a structured prompt for is
            # still a checkpoint that can be sent a LoRA tag.
            self._literals_only(p, parsed)
            return

        from prompt_master.krea import spatial as spatial_module

        request = mc_krea_pipeline.Request(
            source=parsed.clean_text.strip(),
            raw_source=str(getattr(p, "prompt", "") or ""),
            literals=parsed,
            creative=creative,
            creative_settings=settings,
            layout=layout,
            ratio=spatial_module.aspect_ratio(getattr(p, "width", 0),
                                              getattr(p, "height", 0)),
            image_seed=_image_seed(p),
            record_scenes=self._record_scenes)

        # Every pass, inside one declaration and under one re-entrancy flag.
        #
        # ``host_job`` is what tells ``mc_llm_sessions._Gpu.acquire`` that the
        # image job is blocked waiting for this request rather than competing
        # with it, and it has to cover the Spatial Composer as much as the
        # writer: both run at the same point in the same hook, with the same
        # generation waiting on them, and one started outside this block would
        # wait for the job that is waiting for it. It is re-entrant and
        # thread-local, so declaring it once here is the whole of it.
        self._rolling = True
        try:
            with mc_broker.host_job():
                outcome = mc_krea_pipeline.run(
                    request,
                    write=lambda source: (self._roll(source, settings, layout,
                                                     request.raw_source),
                                          self._complaint))
        finally:
            self._rolling = False

        if outcome.spatial_note and not self._spatial_note:
            self._spatial_note = outcome.spatial_note
        self._composed_without_creative = bool(
            creative and not outcome.ran_creative and outcome.ran_spatial)
        written = outcome.prepared
        if written is None:
            return

        p.prompt = written.generation
        # What Stage 2 may inherit, if there is a Stage 2. Recorded whether or
        # not this generation had literals in it: the two prompts are the same
        # string when it did not, and a Stage 2 that reads one attribute on
        # every chain is simpler than one that has to know when to.
        mc_lora.remember_inheritable(p, written.inheritable,
                                     self._inheritable_negative)
        try:
            p.extra_generation_params.update(written.metadata)
        except Exception:
            logger.debug("Model Chain: could not record the Krea metadata",
                         exc_info=True)
        _say_what_ran(outcome, written)

    def _publish_plan(self, p, layout, creative: bool = True) -> None:
        """Work out what this generation will actually do, before any of it happens.

        This is the whole point of running here. ``before_process`` is earlier
        than the checkpoint load and much earlier than Stage 2, which is
        precisely why the language model used to be sized so badly: it is
        placed against a card that nothing has taken yet, and everything that
        *will* take it is still in the future.

        Building the plan first turns that future into a number. The writer is
        then placed in what the largest phase of the generation does not need,
        and it keeps that placement through Stage 1, the handoff, Stage 2 and
        the warm-up -- because none of those is a plan boundary and none of
        them re-opens the question.

        Failure here is never a refused generation. A plan that cannot be built
        is simply not published, and every path falls back to the behaviour it
        had before plans existed.
        """
        from prompt_master.krea import spatial as spatial_module

        try:
            compose = ""
            if getattr(layout, "regions", ()):
                compose = (spatial_module.SMART
                           if layout.compose_mode == spatial_module.SMART
                           else spatial_module.DIRECT)
            # The actual Creative state, not True. A Spatial-only generation
            # that published a Creative Writer phase would have the bar
            # describing a request nobody is going to make, and the VRAM
            # arithmetic reserving room for it.
            mc_plan.publish(mc_plan.build_for(p, creative=bool(creative),
                                              spatial_compose=compose))
        except Exception:
            logger.debug("Model Chain: could not build this generation's plan",
                         exc_info=True)

    def postprocess(self, p, processed, *args):
        """Say on the result when Creative Mode did not write the prompt.

        Failure is always "generate what the user typed", and that is right: a
        language model that will not answer is not a reason to refuse a
        generation somebody asked for. But it was said only in the console, and
        what a user sees is an image made from their four words with no
        indication that anything was meant to happen -- which reads as "Creative
        Mode does nothing", not as "the writer is down".

        So the reason goes on the result, beside the image, in the place this
        extension already puts the sentence when Stage 2 fails. The roll runs in
        ``before_process`` and this is the first hook after it that is handed
        something the user will look at.

        The layout half is reported separately and can appear on its own. A
        generation whose prompt was written perfectly well and whose Spatial
        Composer timed out is not a Creative Mode failure, and describing it as
        one would send somebody to look at the wrong thing.
        """
        complaint = self._complaint
        spatial_note = self._spatial_note
        composed_raw = self._composed_without_creative
        self._complaint = ""
        self._spatial_note = ""
        self._composed_without_creative = False
        if processed is None:
            return
        try:
            if complaint:
                # "The prompt exactly as typed" is only true when nothing else
                # ran. With Spatial Layout on, a writer that failed no longer
                # takes the boxes down with it -- §21 -- so the sentence has to
                # say which of the two generations actually happened.
                after = ("Your regions were composed around the prompt as typed "
                         "instead." if composed_raw else
                         "The image was generated from the prompt exactly as typed.")
                processed.comments += (
                    f"\nModel Chain: Creative Mode did not write this prompt — "
                    f"{complaint}. {after} The console and LLM Studio → Setup say "
                    "more.")
            if spatial_note:
                processed.comments += (
                    f"\nModel Chain: Spatial Layout — {spatial_note}. Your regions "
                    "have not been changed.")
        except Exception:
            logger.debug("Model Chain: could not put the Creative Mode notice on the "
                         "result", exc_info=True)

    def _record_literal_fields(self, p, before, after) -> None:
        """Write what the two Literal Prompt boxes held into this image.

        Before anything can return, because every path out of ``before_process``
        applies them and every one of those images should be able to say so.

        The one place a literal payload is recorded under a key of its own, and
        the exception that proves the rule: a bracketed command is in the
        ``Prompt:`` line and in the recorded source with its brackets still on,
        so a third copy would repeat the file. A field's text is in neither --
        the prompt line has it restored and unbracketed, indistinguishable from
        the words around it, and the source line never had it. Without these two
        keys the authoring setup could not be reconstructed from an image.

        Absent entirely when the boxes are empty, like every other optional key:
        an ordinary image should say nothing about a feature it did not use.
        """
        recorded = {}
        if str(before or "").strip():
            recorded[mc_infotext.LITERAL_POSITIVE] = str(before).strip()
        if str(after or "").strip():
            recorded[mc_infotext.LITERAL_NEGATIVE] = str(after).strip()
        if not recorded:
            return
        try:
            p.extra_generation_params.update(recorded)
        except Exception:
            logger.debug("Model Chain: could not record the Literal Prompt fields",
                         exc_info=True)

    def _restore_negative(self, p) -> None:
        """Take the brackets off the negative prompt, if it has any.

        Cheap and unconditional, in that order: :func:`literals.parse` returns
        the string it was given, untouched, when there is no ``[[`` in it, so a
        negative prompt nobody wrote a command in is the same object it was.

        The inheritable half is kept for Stage 2 the same way the positive one
        is. A Stage 1 negative that names another extension's syntax has no more
        business in a Stage 2 pass than a Stage 1 LoRA does.
        """
        from prompt_master.krea import literals

        self._inheritable_negative = ""
        negative = getattr(p, "negative_prompt", "") or ""
        if not literals.present(negative):
            self._inheritable_negative = str(negative)
            return
        parsed = literals.parse(negative)
        for warning in parsed.warnings:
            logger.warning("Model Chain: the negative prompt — %s", warning)
        self._inheritable_negative = parsed.clean_text
        if not parsed.commands:
            return
        try:
            p.negative_prompt = literals.restore(parsed.clean_text, parsed)
        except Exception:
            logger.debug("Model Chain: could not restore the negative prompt's literal "
                         "commands", exc_info=True)
            self._inheritable_negative = str(negative)
            return
        # Recorded here and not only at the end, because this half can be the
        # only half: a generation whose positive prompt carried no command and
        # whose negative did takes none of the paths below, and Stage 2 would
        # otherwise inherit the restored negative. Whatever runs after this
        # writes the pair again with the positive filled in.
        mc_lora.remember_inheritable(p, "", self._inheritable_negative)
        logger.info("Model Chain: the negative prompt carried %s",
                    literals.describe(parsed))

    def _literals_only(self, p, parsed) -> None:
        """Substitute a prompt whose only change is that its brackets came off.

        The path an ordinary generation takes when somebody has typed a literal
        command with both features switched off, and the path a refused one
        takes on its way out. It runs no model, publishes no plan and records no
        Creative or Spatial key -- there is nothing to say about a pipeline that
        did not run.

        Does nothing at all when there are no commands, which is the case it is
        called in almost every time.
        """
        from prompt_master.krea import literals

        if not getattr(parsed, "commands", ()):
            return
        try:
            restored = literals.restore(parsed.clean_text, parsed)
        except Exception:
            logger.debug("Model Chain: could not restore the prompt's literal commands",
                         exc_info=True)
            return
        p.prompt = restored
        # The clean text and not the restored one: this is the whole of §9 in a
        # generation with no language model in it. A Stage 2 inheriting
        # ``<lora:krea2_edit:1>`` from here would be applying a Stage 1 edit LoRA
        # to a model that has never seen the reference it edits against.
        mc_lora.remember_inheritable(p, parsed.clean_text,
                                     self._inheritable_negative)
        try:
            p.extra_generation_params.update(
                mc_creative_krea.prepare(None, literals=parsed).metadata)
        except Exception:
            logger.debug("Model Chain: could not record the literal command metadata",
                         exc_info=True)
        logger.info("Model Chain: the prompt carried %s; no language model ran",
                    literals.describe(parsed))

    def _layout(self, p, values):
        """This generation's spatial layout, or an empty one.

        Parsed before the roll rather than after it, because pass 1 has to know
        whether a layout exists: the one sentence it is told about placement is
        added only when there is a layout to justify it, and adding it after the
        writer had already written would be adding it to nothing.

        An empty answer -- Spatial Layout switched off, nothing drawn, or a
        layout this build cannot read -- is the answer that makes the rest of
        this generation whatever Creative Mode alone makes of it, which for a
        generation with Creative Mode off as well is nothing at all.

        Parsed before the gate rather than after it, now that it is half of what
        the gate reads: "should this hook do anything" is "is either feature on",
        and only one of the two can be answered from a checkbox.
        """
        from prompt_master.krea import spatial

        chosen = _spatial_for(values)
        self._record_scenes = bool(chosen.get("record_scenes", True))
        if not chosen.get("enabled"):
            return spatial.Layout()
        layout = mc_spatial.layout_for(chosen.get("layout"),
                                       width=getattr(p, "width", 0),
                                       height=getattr(p, "height", 0),
                                       compose_mode=chosen.get("compose_mode"))
        for note in layout.notes:
            logger.warning("Model Chain: Spatial Layout — %s", note)
        if layout.unreadable:
            self._spatial_note = " ".join(layout.notes)
        return layout

    def _roll(self, source, settings, layout, raw_source=""):
        """One creative roll for this generation, or ``None`` to leave it alone.

        Returns the :class:`mc_creative_krea.Roll` rather than the finished
        prompt, because what the prompt is made of is settled after this: a
        layout may still be composed onto the scene it produced. The two steps
        are separate functions for the same reason they are separate passes --
        one of them talks to a model and the other one cannot.

        Called inside the :class:`mc_broker.host_job` block ``before_process``
        opens, which is how ``mc_llm_sessions._Gpu.acquire`` is told that the
        image job is blocked waiting for this request rather than competing with
        it. The bar is borrowed rather than claimed, because the host already
        started one for the generation this is the first part of.

        The events are drained rather than forwarded. There is no open Gradio
        event to forward them to -- the press became a native generation, not a
        handler with an output list -- so the progress bar carries the phase and
        the log carries the rest.

        ``source`` is handed in rather than read off ``p``, and it is the
        *transformable* text: the pipeline has already lifted the literal
        commands out of the prompt box. Reading ``p.prompt`` here would put them
        back in front of the writer, which is the one thing this feature exists
        to prevent.
        """
        source = str(source or "").strip()
        if not source:
            # A prompt that was nothing but literal commands lands here, and it
            # is not an error: there was no transformable text to write from,
            # the commands are restored around nothing, and the generation goes
            # ahead with them. §14.
            logger.info("Model Chain: Creative Mode has no source prompt to work from")
            self._complaint = ("there was no transformable text to work from — the "
                               "prompt was entirely literal commands"
                               if raw_source.strip() else
                               "there was no prompt to work from")
            return None

        session = mc_creative_krea.creative

        # Held by name and closed explicitly, the way the roll itself holds the
        # LLM run: every path out of the loop below leaves the generator
        # suspended, and it is its ``finally`` that gives the progress bar and
        # the workload lock back. Closing it is what runs that now rather than
        # whenever the interpreter next collects the frame.
        events = session.roll(source, settings, guard_checkpoint=True, own_bar=False,
                              spatial_layout=layout, raw_source=raw_source)
        written = False
        try:
            for event in events:
                if event.kind == sessions.STATUS:
                    logger.debug("Model Chain: Creative Mode — %s", event.text)
                elif event.kind == sessions.CANCELLED:
                    logger.info("Model Chain: the Creative Mode roll was stopped; "
                                "the generation continues with the typed prompt")
                    self._complaint = "the roll was stopped"
                    break
                elif event.kind == sessions.FAILED:
                    logger.warning("Model Chain: the Creative Mode roll failed (%s); "
                                   "generating from the typed prompt instead",
                                   event.text)
                    self._complaint = event.text
                    break
                elif event.kind == sessions.DONE:
                    written = True
                    break
        except Exception as exc:
            errors.report("Model Chain: the Creative Mode roll failed", exc_info=True)
            self._complaint = str(exc) or exc.__class__.__name__
            return None
        finally:
            events.close()

        if not written:
            return None

        last = session.last
        if last is None or not last.expanded.strip():
            logger.warning("Model Chain: the Creative Mode roll produced nothing; "
                           "generating from the typed prompt instead")
            self._complaint = "the writer returned nothing"
            return None
        return last
