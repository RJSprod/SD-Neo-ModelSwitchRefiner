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

import mc_arch
import mc_creative_krea
import mc_creative_panel
import mc_infotext
import mc_llm_sessions as sessions
import mc_lora
import mc_memory
import mc_plan
import mc_krea_pipeline
import mc_references
import mc_spatial
import mc_spatial_klein
from modules import errors, scripts

logger = mc_memory.logger
"""Shared with the helper modules; mc_memory attaches the console handler."""

PREFIX = "mc-krea-creative"
"""Every id this script puts in the page starts here.

The browser gate finds its elements by these ids and by nothing else -- no
Gradio-generated class, no DOM shape. A theme that replaces Gradio's internals
can change how this looks and cannot stop it working.
"""

CHECKPOINT_ELEM_ID = "setting_sd_model_checkpoint"
"""The host's own checkpoint dropdown, read and never written.

Watched for one reason: which backend consumes the Spatial canvas is decided by
the loaded architecture, and §9 asks the panel to say so as soon as it changes
rather than on the next press. Nothing here selects a checkpoint or reacts to
one beyond repainting a status line.
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


def notice(text: str, kind: str = "info") -> str:
    """One line of Creative Mode status, as scoped HTML.

    Its own classes rather than LLM Studio's, because ``style.css`` scopes those
    under ``#mc-llm-studio`` and this line is in txt2img. Same idea, same
    reliance on the host's custom properties for colour, different neighbourhood.
    """
    import html

    return (f'<div class="{PREFIX}-notice {PREFIX}-notice-{kind}">'
            f'{html.escape(str(text or ""))}</div>')


# --------------------------------------------------------------------------- #
# The settings one roll runs with
# --------------------------------------------------------------------------- #


SPATIAL_CONTROLS = 4
"""How many controls the Spatial block contributes.

Enabled, Krea's compose mode, Klein's spatial mode, and the serialized layout --
in that order, and the order is the contract :func:`_split` cuts by.

Four and not three because there is one canvas with two backends behind it. The
compose mode is Krea's question (does a language model reconcile the scene) and
the spatial mode is Klein's (what is the source image, and what happens to it),
and neither is answerable in the other's terms. They both travel on every
generation because which one is *read* depends on the checkpoint that happens to
be loaded when Generate is pressed, which is not knowable while the tuple is
being assembled.

An older API caller that still sends three lands on the pre-Klein shape and is
answered from the saved preferences, which is what :func:`_spatial_for` already
did for a caller that sent none.
"""

LEGACY_SPATIAL_CONTROLS = 3
"""The tail an API script written against the pre-Klein build still sends."""


def _split(values) -> tuple[tuple, tuple, tuple]:
    """``before_process``'s tuple, cut into its three parts.

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
        for tail in (SPATIAL_CONTROLS, LEGACY_SPATIAL_CONTROLS):
            if len(values) >= expected + tail:
                return (values[:3], values[3:expected],
                        values[expected:expected + tail])
        if len(values) >= expected:
            return values[:3], values[3:expected], ()

    for tail in (SPATIAL_CONTROLS, LEGACY_SPATIAL_CONTROLS):
        if len(values) == 1 + tail:
            # The no-panel shape: creativity, then the Spatial tail.
            return (), (), values[-tail:]
    if len(values) >= 3:
        return values[:3], (), ()
    return (), (), ()


def _settings_for(values) -> dict:
    """This generation's Creative settings, from the panel when it sent them.

    ``values`` is what Forge handed ``before_process`` after the enabled flag,
    in the order :meth:`ui` returned it. A UI that could not build its axis
    controls sends fewer, and an API request sends none at all, so the length is
    checked rather than assumed and the saved preferences answer for anything
    absent.
    """
    scalars, axes, _ = _split(values)
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
    _, _, spatial = _split(values)
    stored = mc_spatial.settings()
    if len(spatial) < LEGACY_SPATIAL_CONTROLS:
        return stored

    from prompt_master import spatial as generic
    from prompt_master.krea import spatial as spatial_module

    if len(spatial) >= SPATIAL_CONTROLS:
        enabled, mode, klein_mode, layout = spatial[:SPATIAL_CONTROLS]
        stored["klein_mode"] = generic.normalise_mode(klein_mode,
                                                      stored["klein_mode"])
    else:
        # A caller from before the Klein backend existed. Its three controls
        # still mean what they meant; the mode it does not send is the saved
        # one, which is the same answer an API caller with no panel gets.
        enabled, mode, layout = spatial[:LEGACY_SPATIAL_CONTROLS]

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
    """Show or hide Creative's own controls, and re-describe the pipeline.

    Three of the four outputs are Creative's: the slider and the drawer appear
    and disappear with the feature, because neither does anything while it is
    off, and the status line says what the feature is now doing.

    The fourth is Spatial's status, and it is here for the reason this whole
    refactor exists. Spatial no longer appears or disappears with Creative Mode
    -- it is a peer feature and stays exactly where it is -- but what its
    pipeline *is* depends on both toggles: the same Direct merge composes around
    a written scene with Creative on and around the typed prompt with it off.
    One of the two sentences would be wrong if this handler did not send the
    other section's line as well.
    """
    mc_creative_krea.remember(**{mc_creative_krea.ENABLED: bool(enabled)})
    shown = gr.update(visible=bool(enabled))
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
    return (shown, shown, gr.update(value=told, visible=bool(enabled)),
            gr.update(value=spatial_summary(serialized, bool(spatial_enabled),
                                            creative=bool(enabled), mode=mode)))


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

    They belong to the proxy and not to a region, which is the whole of §6:
    handles that lived on the region body would be buried with it the moment
    something overlapped it.
    """
    return "".join(
        f'<span class="{SPATIAL_PREFIX}-handle {SPATIAL_PREFIX}-handle-{corner}"'
        f' data-corner="{corner}" aria-hidden="true"></span>'
        for corner in HANDLES)


def spatial_editor() -> str:
    """The layout workspace's markup: every control, none of its behaviour.

    Built in Python and handed to the page as one static block, for two reasons
    that are both about single sources of truth. The framing and camera-angle
    lists are the compositor's own vocabularies, so a value that can be chosen
    is a value that renders a phrase. And the whole editor is one element with
    stable ids, so the browser file finds what it needs by id and by nothing
    else -- no Gradio class, no DOM shape, nothing a theme can rearrange.

    A workspace, not a modal
    ------------------------
    This used to be a ``position: fixed`` overlay that JavaScript moved to
    ``document.body``, because a fixed overlay inside an accordion is one
    ``overflow: hidden`` or one ``transform`` away from being a modal nobody can
    see. Moving it solved that and bought a second problem: two copies of every
    id whenever Gradio rebuilt the tab, and a page whose scrolling belonged to
    the overlay rather than to the browser.

    It is now a block in ordinary document flow that Edit Layout reveals and
    Back hides. It scrolls the way the page scrolls, it inherits the theme it is
    standing in, it cannot be clipped out of existence by a container it is no
    longer escaping, and on a phone it is a page rather than a window over one.

    The canvas is a ``<div>``, not a ``<canvas>``. Regions are elements, so a
    region can carry a title, be found by id, be focused, be styled by a theme
    and be read by a test; a bitmap canvas would put all of that behind a
    redraw loop in order to draw rectangles.
    """
    from prompt_master.krea import spatial

    return f'''
<div id="{_spatial_id("workspace")}" class="{SPATIAL_PREFIX}-workspace"
     role="region" aria-label="Spatial Layout editor" hidden>

  <div class="{SPATIAL_PREFIX}-topbar">
    <button type="button" id="{_spatial_id("cancel")}"
            class="{SPATIAL_PREFIX}-tool {SPATIAL_PREFIX}-back">&#8249;&nbsp;Back</button>
    <span class="{SPATIAL_PREFIX}-title">Spatial Layout</span>
    <span id="{_spatial_id("size")}" class="{SPATIAL_PREFIX}-dims"></span>
    <span class="{SPATIAL_PREFIX}-spacer"></span>
    <button type="button" id="{_spatial_id("full")}"
            class="{SPATIAL_PREFIX}-tool" aria-pressed="false">Full screen</button>
    <button type="button" id="{_spatial_id("save")}"
            class="{SPATIAL_PREFIX}-tool {SPATIAL_PREFIX}-primary">Save &amp; Return</button>
  </div>

  <div class="{SPATIAL_PREFIX}-toolbar" role="toolbar" aria-label="Region actions">
    <button type="button" id="{_spatial_id("add")}"
            class="{SPATIAL_PREFIX}-tool">+&nbsp;Add region</button>
    <button type="button" id="{_spatial_id("draw")}"
            class="{SPATIAL_PREFIX}-tool">Draw region</button>
    <button type="button" id="{_spatial_id("clear")}"
            class="{SPATIAL_PREFIX}-tool">Clear all</button>
    <span class="{SPATIAL_PREFIX}-spacer"></span>
    <span class="{SPATIAL_PREFIX}-group">
      <button type="button" id="{_spatial_id("undo")}"
              class="{SPATIAL_PREFIX}-tool" title="Undo (Ctrl+Z)">Undo</button>
      <button type="button" id="{_spatial_id("redo")}"
              class="{SPATIAL_PREFIX}-tool" title="Redo (Ctrl+Shift+Z)">Redo</button>
    </span>
  </div>

  <p id="{_spatial_id("warning")}" class="{SPATIAL_PREFIX}-warning" hidden></p>
  <p id="{_spatial_id("message")}" class="{SPATIAL_PREFIX}-message" hidden></p>

  <div id="{_spatial_id("confirm")}" class="{SPATIAL_PREFIX}-confirm" hidden>
    <span>This layout has unsaved changes.</span>
    <span class="{SPATIAL_PREFIX}-spacer"></span>
    <button type="button" id="{_spatial_id("keep")}"
            class="{SPATIAL_PREFIX}-tool">Keep editing</button>
    <button type="button" id="{_spatial_id("discard")}"
            class="{SPATIAL_PREFIX}-tool {SPATIAL_PREFIX}-danger">Discard changes</button>
  </div>

  <div class="{SPATIAL_PREFIX}-body">

    <div class="{SPATIAL_PREFIX}-stage">
      <div id="{_spatial_id("scroll")}" class="{SPATIAL_PREFIX}-scroll">
        <div id="{_spatial_id("canvas")}" class="{SPATIAL_PREFIX}-canvas" tabindex="0"
             role="application" aria-label="Composition frame">
          <div id="{_spatial_id("guides")}" class="{SPATIAL_PREFIX}-guides thirds"
               aria-hidden="true"></div>
          <div id="{_spatial_id("regions")}" class="{SPATIAL_PREFIX}-regions"></div>
          <div id="{_spatial_id("proxy")}" class="{SPATIAL_PREFIX}-proxy" hidden>
            <span id="{_spatial_id("proxy-label")}"
                  class="{SPATIAL_PREFIX}-proxy-label"></span>
            {_handles()}
          </div>
        </div>
      </div>

      <div class="{SPATIAL_PREFIX}-stagebar">
        <label class="{SPATIAL_PREFIX}-inline">
          <span>Grid</span>
          <select id="{_spatial_id("grid")}">
            <option value="none" selected>None</option>
            <option value="thirds">Thirds</option>
            <option value="center">Centre</option>
            <option value="both">Thirds + centre</option>
          </select>
        </label>
        <span class="{SPATIAL_PREFIX}-zoom" role="group" aria-label="Zoom">
          <button type="button" id="{_spatial_id("zoom-fit")}"
                  class="{SPATIAL_PREFIX}-tool">Fit</button>
          <button type="button" id="{_spatial_id("zoom-out")}"
                  class="{SPATIAL_PREFIX}-tool" aria-label="Zoom out">&#8722;</button>
          <span id="{_spatial_id("zoom-level")}"
                class="{SPATIAL_PREFIX}-zoom-level">100%</span>
          <button type="button" id="{_spatial_id("zoom-in")}"
                  class="{SPATIAL_PREFIX}-tool" aria-label="Zoom in">+</button>
        </span>
        <label class="{SPATIAL_PREFIX}-check">
          <input type="checkbox" id="{_spatial_id("auto-hint")}" checked />
          <span>Auto position hints</span>
        </label>
        <span class="{SPATIAL_PREFIX}-note">Scene prompt goes through the Creative
          LLM · region prompts bypass it · [[double brackets]] bypass every
          language model and stay in this box</span>
      </div>
    </div>

    <div class="{SPATIAL_PREFIX}-side">

      <section class="{SPATIAL_PREFIX}-panel {SPATIAL_PREFIX}-regions-panel">
        <header class="{SPATIAL_PREFIX}-panel-head">
          <h4>Regions</h4>
          <span id="{_spatial_id("count")}" class="{SPATIAL_PREFIX}-note"></span>
        </header>
        <div id="{_spatial_id("list")}" class="{SPATIAL_PREFIX}-list" role="listbox"
             aria-label="Regions, frontmost first" tabindex="0"></div>
        <p id="{_spatial_id("empty")}" class="{SPATIAL_PREFIX}-empty">
          No regions yet. Add one, or draw one on the frame.</p>
        <div class="{SPATIAL_PREFIX}-layerbar" role="group" aria-label="Stacking order">
          <button type="button" id="{_spatial_id("front")}"
                  class="{SPATIAL_PREFIX}-tool" title="Bring to front">To front</button>
          <button type="button" id="{_spatial_id("raise")}"
                  class="{SPATIAL_PREFIX}-tool" title="Bring forward">Forward</button>
          <button type="button" id="{_spatial_id("lower")}"
                  class="{SPATIAL_PREFIX}-tool" title="Send backward">Back</button>
          <button type="button" id="{_spatial_id("bottom")}"
                  class="{SPATIAL_PREFIX}-tool" title="Send to back">To back</button>
        </div>
      </section>

      <section class="{SPATIAL_PREFIX}-panel {SPATIAL_PREFIX}-inspector">
        <header class="{SPATIAL_PREFIX}-panel-head">
          <h4>Selected region</h4>
          <span id="{_spatial_id("selected-name")}"
                class="{SPATIAL_PREFIX}-note">nothing selected</span>
        </header>

        <label class="{SPATIAL_PREFIX}-field">
          <span>Name</span>
          <input type="text" id="{_spatial_id("name")}" />
        </label>
        <label class="{SPATIAL_PREFIX}-field">
          <span>Type</span>
          <select id="{_spatial_id("type")}">
            <option value="{spatial.OBJECT}">Object</option>
            <option value="{spatial.TEXT}">Text</option>
          </select>
        </label>
        <label class="{SPATIAL_PREFIX}-field" id="{_spatial_id("text-field")}" hidden>
          <span>Visible text</span>
          <input type="text" id="{_spatial_id("text")}"
                 placeholder="the exact words to render" />
        </label>
        <label class="{SPATIAL_PREFIX}-field {SPATIAL_PREFIX}-grow">
          <span id="{_spatial_id("prompt-label")}">Region prompt</span>
          <textarea id="{_spatial_id("prompt")}" rows="5"
                    placeholder="what is in this box, in your own words &#10;&#10;[[her shirt from image 1]] is passed straight to the image model"></textarea>
        </label>
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

        <div class="{SPATIAL_PREFIX}-destructive">
          <button type="button" id="{_spatial_id("duplicate")}"
                  class="{SPATIAL_PREFIX}-tool">Duplicate</button>
          <button type="button" id="{_spatial_id("delete")}"
                  class="{SPATIAL_PREFIX}-tool {SPATIAL_PREFIX}-danger">Delete</button>
        </div>
      </section>

    </div>
  </div>
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

    # Which backend the boxes go to decides what this line can truthfully say
    # about them. Krea composes a prompt; Klein conditions a region. Describing
    # one in the other's words is how a user ends up expecting a mask from a
    # sentence, or a sentence from a mask.
    klein = _klein_visible()
    if klein:
        pipeline = _klein_pipeline_sentence()
    else:
        pipeline = mc_krea_pipeline.described(creative=bool(creative),
                                              spatial=bool(enabled), mode=mode)
    if not layout.regions:
        return notice("No regions yet. Press Edit Layout to draw one. " + pipeline)
    said = spatial.summarise(layout)
    if not enabled:
        return notice(f"{said} — Spatial Layout is off, so they are not applied.")
    return notice(f"{said}. Region prompts are used exactly as typed. {pipeline}")


def _klein_pipeline_sentence() -> str:
    """What Spatial Layout does on a Klein checkpoint, in one sentence.

    The Krea sentence's counterpart, and deliberately not a copy of it. Krea's
    turns on how the *scene* is written; Klein's turns on the fact that the scene
    is not written at all -- the prompt reaches the model as typed, and the boxes
    reach it as conditioning geometry beside it.
    """
    return ("Your prompt is generated exactly as typed and each region's prompt is "
            "conditioned on its own part of the frame. No language-model request is "
            "made. This is spatial conditioning rather than a mask: it biases where "
            "a concept appears and does not confine it to the rectangle.")


def _spatial_toggled(enabled, serialized, creative, mode):
    """Remember the Spatial toggle, and say what it now does."""
    mc_spatial.remember(**{mc_spatial.ENABLED: bool(enabled)})
    return spatial_summary(serialized, bool(enabled), creative=bool(creative),
                           mode=mode)


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
    return spatial_summary(serialized, bool(enabled), creative=bool(creative),
                           mode=mode)


def _spatial_saved(serialized, enabled, creative, mode):
    """The browser saved a layout. Keep it, and repaint the summary."""
    mc_spatial.remember(**{mc_spatial.LAYOUT: str(serialized or "")})
    return spatial_summary(serialized, bool(enabled), creative=bool(creative),
                           mode=mode)


# --------------------------------------------------------------------------- #
# The Klein backend's own controls
# --------------------------------------------------------------------------- #
#
# One canvas, two backends, and the difference between them is what the boxes
# are *for*. Krea's question is whether a language model reconciles the scene
# with them; Klein's is what the source image is and how much of it survives.
# Neither question is answerable in the other's terms, so the panel shows
# whichever belongs to the loaded checkpoint and remembers both.
#
# §8: the existing "Smart Spatial Compose" / "Direct BBOX Merge" pair is a Krea
# prompt-composition concept and is deliberately *not* offered as a Klein mode.


def _klein_choices(has_source: bool) -> list:
    """``(label, value)`` pairs for the Klein spatial mode radio.

    Every mode is listed whether or not it can run, which is §4: a mode that
    vanished when a gallery emptied would look like an upgrade removed it, and
    somebody would go looking for a feature that is sitting right there waiting
    for an image. The unavailable ones carry their reason in the label.

    A label and not a disabled attribute, because Gradio's Radio disables all of
    its choices or none of them, and the alternative -- five checkboxes, or a
    dropdown rebuilt on every gallery change -- would trade a real control for a
    cosmetic property. That is a smaller cost than it looks: §9 is explicit that
    UI availability is advisory and generation-time revalidation is
    authoritative, so an unavailable mode selected here is refused before
    sampling by :func:`prompt_master.spatial.resolve` with the same sentence.
    """
    from prompt_master import spatial as generic

    available = generic.available_modes(bool(has_source))
    choices = []
    for mode in generic.MODES:
        label = generic.MODE_LABELS[mode]
        if mode not in available:
            label = f"{label} — needs an ImageStitch image"
        choices.append((label, mode))
    return choices


def _klein_architecture(checkpoint=None):
    """Which architecture the Spatial canvas will be consumed by.

    The *selected* checkpoint when the panel has one to look at, and the loaded
    engine otherwise. That order is the right way round for a live panel and the
    wrong way round for a generation, which is why only the panel calls it this
    way: a user who has just picked a Klein checkpoint from the dropdown has not
    loaded it yet -- the reload happens on the same change event this handler is
    racing -- and a status line that answered from the model still in VRAM would
    describe the checkpoint they just left.

    :func:`mc_spatial_klein.loaded_architecture` is the generation-time answer
    and stays that way. It asks the engine first, because by then there is
    nothing left to predict.
    """
    name = str(checkpoint or "").strip()
    if name:
        found = mc_arch.detect_from_checkpoint_name(name)
        if found is not mc_arch.UNKNOWN:
            return found
    return mc_spatial_klein.loaded_architecture()


def _klein_status(live=None, mode=None, checkpoint=None) -> str:
    """The Backend / Source line under the Klein mode radio.

    Three facts, in the order somebody reads them: which backend the loaded
    checkpoint gets, what ImageStitch is currently holding, and -- because Auto
    is the default and its whole promise is that it is predictable -- what Auto
    would resolve to right now.
    """
    from prompt_master import spatial as generic

    arch = _klein_architecture(checkpoint)
    if not mc_spatial_klein.is_klein(arch):
        return notice("Spatial Layout composes a Krea 2 structured prompt for this "
                      "checkpoint. Load a FLUX.2 Klein 9B checkpoint for regional "
                      "conditioning instead.")

    if live is None:
        live = mc_references.live_source()
    mode = generic.normalise_mode(mode, generic.AUTO)

    said = [f"Backend: {arch.label}.", f"Source: {live.describe()}"]
    if mode == generic.AUTO and not live.known:
        # Both halves, because the panel genuinely does not know which one it
        # will be. Naming one of them would be a guess dressed as a readout.
        said.append(f"Auto will use {generic.MODE_LABELS[generic.REFERENCE_REGIONS]} "
                    f"if ImageStitch holds an image and "
                    f"{generic.MODE_LABELS[generic.REGIONAL_GENERATE]} if it does "
                    f"not.")
    elif mode == generic.AUTO:
        resolved = generic.resolve(generic.AUTO, live.usable)
        said.append(f"Auto will use {generic.MODE_LABELS[resolved]}.")
    elif not generic.is_available(mode, live.usable):
        said.append(f"{generic.MODE_LABELS[mode]} {generic.NO_SOURCE_REASON} "
                    "This generation will be refused rather than run as something "
                    "else.")
    else:
        said.append(generic.MODE_HELP.get(mode, ""))
    return notice(" ".join(part for part in said if part),
                  "warn" if (mode != generic.AUTO
                             and not generic.is_available(mode, live.usable))
                  else "info")


def _klein_visible(checkpoint=None) -> bool:
    """Whether the Klein block belongs on the page right now."""
    try:
        return mc_spatial_klein.is_klein(_klein_architecture(checkpoint))
    except Exception:
        logger.debug("Model Chain: could not decide whether to show the Klein "
                     "Spatial controls", exc_info=True)
        return False


def _klein_refresh(mode, gallery=None, checkpoint=None):
    """Repaint the Klein block: the choices, the status, and nothing else.

    Called whenever any of §9's four triggers fires -- the checkpoint, the
    ImageStitch gallery, the Spatial toggle, or the mode itself. It never
    *changes* the selected mode: a gallery emptying is not somebody choosing a
    different picture, and a control that reset itself under the cursor would be
    the silent adaptation §10 reserves for Auto alone.
    """
    from prompt_master import spatial as generic

    live = mc_references.live_source(gallery=gallery)
    mode = generic.normalise_mode(mode, generic.AUTO)
    klein = _klein_visible(checkpoint)
    # Krea's compose radio and Klein's mode radio are the same question asked of
    # two backends, so exactly one of them belongs on the page at a time. Hiding
    # the other is not tidiness: "Smart Spatial Compose" offered beside a Klein
    # checkpoint would be a control that does nothing, and §8 asks for it not to
    # be shown rather than for it to be shown and ignored.
    return (gr.update(choices=_klein_choices(live.usable), visible=klein),
            gr.update(value=_klein_status(live, mode, checkpoint), visible=klein),
            gr.update(visible=not klein))


def _klein_mode_chosen(mode, gallery=None, checkpoint=None):
    """Remember the chosen Klein mode, and say what it will do."""
    from prompt_master import spatial as generic

    mode = generic.normalise_mode(mode, generic.AUTO)
    mc_spatial.remember(**{mc_spatial.KLEIN_MODE: mode})
    live = mc_references.live_source(gallery=gallery)
    return gr.update(value=_klein_status(live, mode, checkpoint))


def _spatial_scenes(record):
    mc_spatial.remember(**{mc_spatial.RECORD_SCENES: bool(record)})
    return notice("The scene before and after the composer pass will be recorded in "
                  "each spatial image's metadata."
                  if record else
                  "The intermediate scenes will not be recorded. Smart and Direct "
                  "images can still be told apart by Krea Spatial Compose Mode, but "
                  "what the composer changed will not be recoverable afterwards.")


# --------------------------------------------------------------------------- #
# Continuing from a pasted image
# --------------------------------------------------------------------------- #


def _pasted_view() -> str:
    """What the last paste said about Creative Mode, as one short block."""
    setup = mc_creative_krea.pasted.setup
    if setup is None or not setup.present:
        return ("Nothing yet. Paste an image made with Creative Mode — PNG Info, the "
                "arrow under the gallery, or a dropped file — and what it records "
                "appears here.")

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
    if setup is None or not setup.present:
        return (gr.update(), gr.update(),
                notice("There is no Creative setup from a pasted image to restore.",
                       "warn"),
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update())

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

    kind = "warn" if setup.warnings() or (replay_exactly and not setup.replayable) \
        else "info"
    told = notice(" ".join(said), kind)
    return (gr.update(value=setup.source) if setup.source else gr.update(),
            gr.update(value=True), told, gr.update(value=_pasted_view()))


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

    layout = spatial_module.parse(setup.layout)
    lines = []
    if setup.spatial_version is not None and setup.spatial_version != spatial_module.VERSION:
        lines.append(f"Recorded with Spatial Layout version {setup.spatial_version}; "
                     f"this build reads version {spatial_module.VERSION}.")
    if setup.klein:
        # A Klein record says three things a Krea one does not, and the third is
        # the one somebody needs before pressing Generate: the references are not
        # in the file and cannot be, so a mode that needs them needs them putting
        # back by hand.
        lines.append("Backend: FLUX.2 Klein regional conditioning"
                     + (f" ({setup.klein_backend})" if setup.klein_backend else ""))
        lines.append(f"Spatial mode: {setup.klein_mode or 'Auto'}"
                     + (f" → {setup.klein_resolved_mode}"
                        if setup.klein_resolved_mode else ""))
        if setup.klein_source_count:
            lines.append(f"Made with {setup.klein_source_count} ImageStitch reference "
                         f"image(s). Reference pixels are not stored in the file — "
                         f"add them back to ImageStitch to use an image-required "
                         f"mode.")
        else:
            lines.append("Made with no source image.")
    mode = str(setup.spatial_compose_mode or "").strip().casefold()
    if mode and not setup.klein:
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

    It restores three things and nothing else: the layout, the backend's own mode
    -- Krea's compose mode or Klein's spatial mode, whichever the image recorded
    -- and Spatial Layout's own switch. Creative Mode's checkbox, the prompt box and
    every Creative setting are left exactly as they are, because a Spatial-only
    image has nothing to say about them and an image with both records has a
    second button that does.
    """
    from prompt_master.krea import spatial as spatial_module

    setup = mc_creative_krea.pasted.setup
    if setup is None or not setup.spatial:
        return (gr.update(), gr.update(), gr.update(),
                notice("There is no spatial layout from a pasted image to restore.",
                       "warn"),
                gr.update(), gr.update())

    if setup.klein:
        return _restore_klein(setup)

    if setup.spatial_version is not None \
            and setup.spatial_version != spatial_module.VERSION:
        # Refused, not migrated. Exact replay never depended on this record --
        # the picture is reproducible from its own Prompt line -- so the honest
        # thing to do with a layout from another build is leave the canvas
        # alone and say so.
        return (gr.update(), gr.update(), gr.update(),
                notice("The recorded spatial layout is from a different version of "
                       "this feature and was not restored; your canvas is "
                       "untouched.", "warn"),
                gr.update(), gr.update())

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
            notice(" ".join(said) + " " + mc_krea_pipeline.described(
                creative=creative, spatial=True,
                mode=mode or mc_spatial.settings()["compose_mode"])),
            gr.update(), gr.update())


def _restore_klein(setup):
    """Put a pasted FLUX.2 Klein spatial image's workflow back.

    §39's three things: the layout, the *requested* mode, and the source
    expectation. The requested mode and not the resolved one -- the resolved mode
    is what Auto decided on the day, and restoring that would turn somebody's
    Auto into a fixed choice they never made.

    The source expectation is the half that cannot be restored, and saying so is
    the whole of what this can do about it: reference pixels are not in the file
    by policy, so an image made with references restores into whatever
    ImageStitch happens to hold now. The mode stays visible and, if its source is
    missing, unavailable -- which is the state §39 asks for, and which the next
    press refuses rather than silently reinterprets.
    """
    from prompt_master import spatial as generic

    mode = generic.mode_from_label(setup.klein_mode, generic.AUTO)
    mc_spatial.remember(**{mc_spatial.LAYOUT: setup.klein_layout,
                           mc_spatial.ENABLED: True,
                           mc_spatial.KLEIN_MODE: mode})

    live = mc_references.live_source()
    said = ["The spatial canvas is back as it was drawn and Spatial Layout is on.",
            f"Spatial mode: {generic.MODE_LABELS[mode]}."]
    kind = "info"
    if setup.klein_source_count and not live.usable:
        said.append(f"This image was made with {setup.klein_source_count} ImageStitch "
                    "reference image(s); re-add them to continue.")
        kind = "warn"
    elif mode in generic.IMAGE_REQUIRED_MODES and not live.usable:
        said.append(generic.NO_SOURCE_REASON)
        kind = "warn"

    return (gr.update(value=setup.klein_layout), gr.update(value=True), gr.update(),
            notice(" ".join(said), kind),
            gr.update(value=mode, choices=_klein_choices(live.usable), visible=True),
            gr.update(value=_klein_status(live, mode), visible=True))


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
        # ImageStitch's own gallery and the checkpoint dropdown, also handed
        # over by after_component. Both are read and never written: they are
        # somebody else's controls, and all this script does with them is
        # repaint its own line when they change.
        self.stitch_gallery = None
        self.checkpoint_box = None
        # Set by ui() once the Klein controls exist, and called by
        # after_component when a component it was waiting for arrives. Both
        # orderings happen -- ImageStitch sorts after this script and the
        # checkpoint dropdown is built before it -- so the wiring is deferred
        # from whichever side is not ready yet.
        self._wire_klein_live = None
        self._klein_wired: set = set()
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
        # This generation's Klein spatial request, when the loaded checkpoint is
        # a Klein one and Spatial Layout is on. Set in before_process, validated
        # against the resident engine in process, consumed by
        # process_before_every_sampling and dropped at both ends.
        self._klein = None
        self._klein_backend = None
        self._klein_layout = ""
        self._klein_note = ""
        # Two scopes, and they close at different times. The job stack holds the
        # reference set, which one encode serves for the whole batch; the pass
        # stack holds the regional conditioning, which is rebuilt against every
        # sampling pass because hires fix runs a second one at a different size.
        #
        # Both are closed at the *top* of before_process as well as at the
        # bottom of postprocess, which is what makes §41 hold: an exception
        # during sampling skips postprocess entirely, so the guarantee that a
        # failed generation leaves the next one ordinary cannot rest on it.
        self._klein_job = None
        self._klein_pass = None

    def title(self):
        return "Krea Creative Mode"

    def show(self, is_img2img):
        # txt2img only. Creative Mode's whole shape is "the positive prompt is a
        # short idea and the image is made from an expansion of it", and
        # img2img's prompt describes an edit to a picture that already exists.
        return scripts.AlwaysVisible if not is_img2img else None

    def after_component(self, component, **kwargs):
        """Keep hold of three of the host's own controls, and nothing else.

        The prompt box, because the restore action has to write the recorded
        source phrase somewhere and that is where a source phrase lives.

        ImageStitch's gallery and the checkpoint dropdown, because §9 asks the
        Klein block to repaint when either changes and neither is this script's.
        Both are matched by element id rather than by position or by class, so a
        theme that rebuilds the page around them changes nothing here -- and the
        ImageStitch id is matched by :mod:`mc_references`, which is the one
        module that knows what that id is.

        Every one of them is optional. A page without ImageStitch installed, or
        a host that renders the checkpoint dropdown somewhere this never sees,
        loses a live repaint and loses nothing else: the runtime check at
        generation time is the authoritative one either way.
        """
        elem_id = kwargs.get("elem_id") or ""

        if elem_id == PROMPT_ELEM_ID:
            self.prompt_box = component
        elif elem_id == CHECKPOINT_ELEM_ID:
            self.checkpoint_box = component
            self._wire_deferred_klein("checkpoint", component)
        elif mc_references.is_stitch_gallery(elem_id) and self.stitch_gallery is None:
            self.stitch_gallery = component
            self._wire_deferred_klein("gallery", component)

    # -- UI ---------------------------------------------------------------- #

    def ui(self, is_img2img):
        from prompt_master.krea import spatial as spatial_module
        from prompt_master.krea import variation

        stored = mc_creative_krea.settings()
        spatial = mc_spatial.settings()

        with gr.Group(elem_id=ident("group")):
            with gr.Row(elem_id=ident("bar")):
                enabled = gr.Checkbox(
                    value=bool(stored["enabled"]), label="Creative Mode", scale=1,
                    elem_id=ident("toggle"),
                    info="direct the prompt locally, then expand it with Krea 2")
                creativity = gr.Slider(
                    label=variation.LABEL, minimum=variation.MINIMUM,
                    maximum=variation.MAXIMUM, step=1, value=stored["creativity"],
                    scale=3, visible=bool(stored["enabled"]), info=variation.HELP,
                    elem_id=ident("creativity"))

            status = gr.HTML(notice("Creative Mode is off."),
                             visible=bool(stored["enabled"]), elem_id=ident("status"))

            with gr.Accordion("Creative Controls", open=False,
                              visible=bool(stored["enabled"]),
                              elem_id=ident("controls")) as controls:
                panel = mc_creative_panel.build(ident, notice, status, creativity,
                                                stored=stored)

                with gr.Accordion("Continue from a pasted image", open=False,
                                  elem_id=ident("restore")):
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
                with gr.Accordion("Last creative roll", open=False,
                                  elem_id=ident("diagnostics")):
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
                    "**Natural** leaves the axis out of the brief entirely — the model "
                    "decides as it would without Creative Mode, and a Natural axis has "
                    "no row above. **Vary** lets the local director choose, and the "
                    "Creativity slider decides whether the axis activates at all, how "
                    "strongly it is expressed, and how hard recent choices are pushed "
                    "away; exclude any treatments you never want. **Fixed** repeats one "
                    "chosen value every roll.\n\n"
                    "Your own words always win. Type *oil painting of a car* and Medium "
                    "stays oil painting however Medium is set.\n\n"
                    "Creative Mode changes the positive prompt only. The negative prompt, "
                    "the checkpoint, the sampler, the size, Steps, the image seed and "
                    "every other setting stay exactly where Forge puts them, and the "
                    "image itself is generated by Forge.")

        # Spatial Layout, a peer and not a child.
        #
        # It used to live inside the group above, visible only while Creative
        # Mode was on, which made it a mode of Creative rather than a feature of
        # its own. Two things follow from moving it out, and both are the point:
        # it is visible and usable with Creative Mode off, and it is built
        # whether or not the creativity library loaded. That second one is not
        # tidiness -- the deterministic compositor needs no vocabulary at all,
        # so an installation whose Creative Mode cannot run can still place
        # boxes around the prompt somebody typed.
        with gr.Group(elem_id=ident("spatial")) as spatial_group:
            with gr.Row(elem_id=ident("spatial", "bar")):
                spatial_enabled = gr.Checkbox(
                    value=bool(spatial["enabled"]), label="Spatial Layout", scale=1,
                    elem_id=ident("spatial", "toggle"),
                    info="place subjects with bounding boxes")
                spatial_compose = gr.Radio(
                    choices=[("Smart Spatial Compose", spatial_module.SMART),
                             ("Direct BBOX Merge", spatial_module.DIRECT)],
                    value=spatial["compose_mode"], label="Composition", scale=2,
                    visible=not _klein_visible(),
                    elem_id=ident("spatial", "compose"))
                edit = gr.Button("Edit Layout…", size="sm", scale=1,
                                 elem_id=_spatial_id("open"))
            spatial_status = gr.HTML(
                spatial_summary(spatial["layout"], bool(spatial["enabled"]),
                                creative=bool(stored["enabled"]),
                                mode=spatial["compose_mode"]),
                elem_id=ident("spatial", "status"))

            # The Klein backend's controls, in the same group and under the same
            # switch. One canvas, one enabled checkbox, one Edit Layout button --
            # what changes with the checkpoint is which backend's question is
            # being asked about the boxes, not whether there are boxes.
            klein_visible = _klein_visible()
            klein_live = mc_references.live_source()
            klein_mode = gr.Radio(
                choices=_klein_choices(klein_live.usable),
                value=spatial["klein_mode"], label="Spatial mode",
                visible=klein_visible, elem_id=ident("spatial", "klein", "mode"),
                info="how the regions reach FLUX.2 Klein, and what the source "
                     "image is")
            klein_status = gr.HTML(
                _klein_status(klein_live, spatial["klein_mode"]),
                visible=klein_visible, elem_id=ident("spatial", "klein", "status"))
            # The one component the browser writes to, and the one that travels
            # with the generation. Hidden rather than absent: the editor is a
            # page, the compositor is a hook, and a hidden textbox is the only
            # thing Gradio offers that is both.
            spatial_state = gr.Textbox(
                value=spatial["layout"], visible=False, lines=1,
                elem_id=_spatial_id("state"))
            gr.HTML(spatial_editor(), elem_id=_spatial_id("editor"))

            with gr.Accordion("Spatial options", open=False,
                              elem_id=ident("spatial", "options")):
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
                with gr.Accordion("Continue from a pasted image", open=False,
                                  elem_id=ident("spatial", "restore")):
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

        self.panel = panel
        self.components = {
            "enabled": enabled, "creativity": creativity, "status": status,
            "controls": controls, "show": show, "recipe": recipe, "expanded": expanded,
            "pasted": pasted, "replay": exactly, "restore": restore, "disarm": disarm,
            "spatial_group": spatial_group, "spatial_enabled": spatial_enabled,
            "spatial_compose": spatial_compose, "spatial_status": spatial_status,
            "spatial_state": spatial_state, "spatial_edit": edit,
            "spatial_scenes": record_scenes, "spatial_pasted": spatial_pasted,
            "spatial_restore": restore_spatial,
            "klein_mode": klein_mode, "klein_status": klein_status}
        if panel is not None:
            self.components.update(panel.components())

        self._wire(enabled, creativity, status, controls, show, recipe, expanded,
                   pasted, exactly, restore, disarm)
        self._wire_spatial(spatial_enabled, spatial_compose, spatial_status,
                           spatial_state, record_scenes, restore_spatial,
                           enabled)
        self._wire_klein(klein_mode, klein_status, spatial_enabled, spatial_compose)
        self._register_paste_fields()

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
        spatial_controls = [spatial_enabled, spatial_compose, klein_mode,
                            spatial_state]
        if panel is None:
            self.arguments = [enabled, creativity] + spatial_controls
        else:
            self.arguments = ([enabled, creativity] + list(panel.settings_controls)
                              + list(panel.axis_controls) + spatial_controls)
        return list(self.arguments)

    def _wire(self, enabled, creativity, status, controls, show, recipe, expanded,
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
                       outputs=[creativity, controls, status,
                                self.components["spatial_status"]],
                       queue=False)

        # The slider moves what the brief costs as well as what it says, and the
        # cost line is the thing somebody looks at straight after moving it. Sent
        # together so the two cannot disagree by one action.
        if self.panel is not None:
            creativity.release(
                fn=lambda value: (_remember_creativity(value),
                                  gr.update(value=mc_creative_panel.describe_cost())),
                inputs=[creativity], outputs=[status, self.panel.cost], queue=False)
        else:
            creativity.release(fn=_remember_creativity, inputs=[creativity],
                               outputs=[status], queue=False)
        show.click(fn=_last_roll, outputs=[recipe, expanded], queue=False)

        # The one handler in this extension that writes to a native control, and
        # the only one that ever should: it is a button whose entire purpose is
        # to put a recorded source phrase back where source phrases are typed.
        #
        # Without the prompt box -- a host that renders the accordion before the
        # prompt, or a test with no page at all -- the restore still restores the
        # settings and still says so; only the phrase has nowhere to go, and it
        # is in the record above for copying.
        if self.prompt_box is not None:
            restore.click(fn=_restore_setup, inputs=[exactly],
                          outputs=[self.prompt_box, enabled, status, pasted],
                          queue=False)
        else:
            logger.debug("Model Chain: the txt2img prompt box was not offered to "
                         "Creative Mode; Restore Creative setup will not fill it in")
            restore.click(fn=lambda exactly: _restore_setup(exactly)[1:],
                          inputs=[exactly], outputs=[enabled, status, pasted],
                          queue=False)
        disarm.click(fn=_disarm_replay, outputs=[status], queue=False)

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
        spatial_enabled.change(
            fn=_spatial_toggled,
            inputs=[spatial_enabled, spatial_state, creative_enabled, spatial_compose],
            outputs=[spatial_status], queue=False)
        spatial_compose.change(
            fn=_spatial_mode,
            inputs=[spatial_compose, spatial_state, spatial_enabled, creative_enabled],
            outputs=[spatial_status], queue=False)
        spatial_state.change(
            fn=_spatial_saved,
            inputs=[spatial_state, spatial_enabled, creative_enabled, spatial_compose],
            outputs=[spatial_status], queue=False)
        record_scenes.change(fn=_spatial_scenes, inputs=[record_scenes],
                             outputs=[spatial_status], queue=False)
        # Spatial's own restore, writing only to Spatial's own controls. Nothing
        # in this list is Creative Mode's, which is the whole difference between
        # this button and the one in the other section.
        # The four outputs this button has always had, then the two the Klein
        # backend added. Appended rather than interleaved: what the first four
        # mean is unchanged, so a Krea restore is the same call it was and only
        # the two new ones are ever ``gr.update()`` no-ops for it.
        restore_spatial.click(
            fn=_restore_spatial,
            outputs=[spatial_state, spatial_enabled, spatial_compose, spatial_status,
                     self.components["klein_mode"], self.components["klein_status"]],
            queue=False)

    def _wire_klein(self, klein_mode, klein_status, spatial_enabled,
                    spatial_compose):
        """The Klein block's handlers, and the ones that have to wait.

        Two of §9's four triggers are this script's own controls and are wired
        immediately. The other two -- ImageStitch's gallery and the checkpoint
        dropdown -- belong to other parts of the page and may not exist yet, so
        the wiring that needs them is held in ``_wire_klein_live`` and run by
        :meth:`after_component` when they arrive. If they never arrive, the
        block still works: it repaints on every interaction with a Spatial
        control, and the generation-time check is authoritative regardless.

        Nothing here selects a mode on the user's behalf. A repaint changes the
        *labels* and the sentence under them; the value stays where it was put,
        because §10 gives Auto alone the licence to adapt.
        """
        outputs = [klein_mode, klein_status, spatial_compose]

        klein_mode.change(fn=_klein_mode_chosen, inputs=[klein_mode],
                          outputs=[klein_status], queue=False)
        # Turning Spatial Layout on is a moment somebody is about to look at
        # this block, which makes it the cheapest place to re-ask a question
        # whose answer may have changed since the page was built.
        spatial_enabled.change(fn=_klein_refresh, inputs=[klein_mode],
                               outputs=outputs, queue=False)

        def wire(kind, component):
            """Repaint when ``component`` changes, reading everything reachable.

            The gallery is an *input* wherever it exists, so a repaint driven by
            the checkpoint still knows what ImageStitch holds. It is never an
            output: this script reads that gallery and has no business writing to
            it.

            Which inputs are available depends on what has been built by the time
            this runs, and the two orderings are both real -- the checkpoint
            dropdown is a quicksetting that exists before any script's ``ui()``,
            and ImageStitch sorts after this script so its gallery appears later.
            So each case gets its own small adapter rather than a placeholder
            argument, and a handler that cannot see the gallery says the contents
            are unknown instead of reporting them as empty.
            """
            gallery = self.stitch_gallery
            if gallery is not None and component is not gallery:
                component.change(
                    fn=lambda mode, images, name: _klein_refresh(
                        mode, gallery=images, checkpoint=name),
                    inputs=[klein_mode, gallery, component], outputs=outputs,
                    queue=False)
            elif gallery is not None:
                component.change(
                    fn=lambda mode, images: _klein_refresh(mode, gallery=images),
                    inputs=[klein_mode, gallery], outputs=outputs, queue=False)
            else:
                component.change(
                    fn=lambda mode, name: _klein_refresh(mode, checkpoint=name),
                    inputs=[klein_mode, component], outputs=outputs, queue=False)

        self._wire_klein_live = wire
        for kind, component in (("gallery", self.stitch_gallery),
                                ("checkpoint", self.checkpoint_box)):
            if component is not None:
                self._wire_deferred_klein(kind, component)

    def _wire_deferred_klein(self, kind: str, component) -> None:
        """Wire one of the two components §9 asks the Klein block to watch.

        Each is wired at most once, and independently of the other, because they
        arrive in either order and sometimes only one arrives at all: the
        checkpoint dropdown is a quicksetting built before any script's ``ui()``,
        and ImageStitch sorts after this script so its gallery appears
        afterwards. A single shared one-shot would have let whichever came first
        cancel the other.

        Never fatal and never noisy: a failure here costs a repaint, and a
        repaint is a convenience on top of a server-side rule.
        """
        wire = self._wire_klein_live
        if wire is None or kind in self._klein_wired or component is None:
            return
        self._klein_wired.add(kind)
        try:
            wire(kind, component)
        except Exception:
            errors.report("Model Chain: failed to wire the Klein Spatial status",
                          exc_info=True)

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
        # Before anything else: whatever a previous generation left behind.
        # An exception during sampling never reaches postprocess, so this is the
        # cleanup §41 actually depends on -- the one that runs whether or not
        # the last job finished.
        self._klein_release()

        # The negative prompt first, and on its own. No language model in this
        # extension has ever seen it, so there is nothing to protect it from --
        # what it needs is the same syntax honoured in the same way, so that a
        # wildcard written ``[[__grain__]]`` in one box behaves as it does in
        # the other. It is restored here and not in the pipeline because the
        # pipeline is about the positive prompt and should stay that way.
        self._restore_negative(p)

        parsed = literals.parse(getattr(p, "prompt", "") or "")
        layout = self._layout(p, args)
        layout = self._klein_route(p, args, layout)
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

    # -- the Klein backend -------------------------------------------------- #

    def _klein_route(self, p, values, layout):
        """Decide which backend this generation's canvas belongs to.

        Returns the layout the Krea pipeline should see. For a Krea 2 checkpoint
        that is the layout unchanged, and every line below is skipped. For a
        Klein checkpoint it is an *empty* layout, and the boxes go to
        :mod:`mc_spatial_klein` instead -- because the two backends are not two
        ways of doing the same thing. Krea's compositor rewrites ``p.prompt``
        into a structured document; Klein's leaves the prompt exactly as the user
        typed it and attaches the regions to the conditioning. Running both would
        hand a Klein model a Krea JSON prompt *and* condition it regionally, and
        the first half of that is the failure ``mc_krea_pipeline.objection``
        already exists to prevent.

        Raises when an explicitly chosen image-required mode has lost its source.
        §10 asks for a clear generation error before sampling rather than a
        silent fallback, and ``before_process`` is the earliest place it can be
        raised from -- nothing has been loaded, nothing has been sampled, and the
        message names the mode the user picked.
        """
        from prompt_master import spatial as generic
        from prompt_master.krea import spatial as spatial_module

        chosen = _spatial_for(values)
        if not chosen.get("enabled"):
            return layout

        arch = mc_spatial_klein.intended_architecture(p)
        if not mc_spatial_klein.is_klein(arch):
            return layout

        try:
            request = mc_spatial_klein.request_for(
                p, chosen.get("layout"),
                enabled=True, requested_mode=chosen.get("klein_mode"))
        except generic.ModeUnavailable as exc:
            # The one place this feature refuses a generation. Everything else in
            # this hook falls back to "generate what the user typed"; this does
            # not, because the user did not ask for a picture from noise -- they
            # asked for their source image edited, and the source is gone.
            logger.error("Model Chain: %s", exc)
            raise RuntimeError(str(exc)) from exc

        import contextlib

        self._klein = request
        self._klein_layout = chosen.get("layout") or ""

        # Whether ImageStitch's own references reach the model, decided here
        # because ``override_settings`` has to be in place before the host
        # applies it -- which happens before any script's ``process`` runs, and
        # so before ImageStitch encodes anything.
        try:
            p.override_settings.update(
                mc_spatial_klein.reference_override(request, arch))
        except Exception:
            logger.debug("Model Chain: could not scope the Klein reference toggle",
                         exc_info=True)

        # Opened here for the same timing reason: after the previous job and
        # before anything has encoded for this one.
        self._klein_job = contextlib.ExitStack()
        self._klein_job.enter_context(mc_spatial_klein.reference_scope(p))

        for note in request.notes:
            logger.warning("Model Chain: Klein Spatial Layout — %s", note)
        if request.unreadable:
            self._spatial_note = " ".join(request.notes)

        # The canvas has gone to the other backend, so the Krea pipeline is told
        # there is nothing to compose. It still runs for Creative Mode and for
        # the literal commands, both of which are prompt concerns and neither of
        # which is a backend's.
        return spatial_module.Layout(width=layout.width, height=layout.height)

    def _klein_release(self) -> None:
        """Unwind everything a Klein spatial job installed. Never raises.

        Called at the top of every generation and at the bottom of every one that
        finishes. Both, deliberately: the second is where it belongs and the
        first is where it is guaranteed to happen.
        """
        for attribute in ("_klein_pass", "_klein_job"):
            stack = getattr(self, attribute, None)
            setattr(self, attribute, None)
            if stack is None:
                continue
            try:
                stack.close()
            except Exception:
                logger.warning("Model Chain: a Klein Spatial scope did not unwind "
                               "cleanly", exc_info=True)
        self._klein = None
        self._klein_backend = None
        self._klein_layout = ""
        self._klein_note = ""

    def process(self, p, *args, **kwargs):
        """Settle the Klein spatial job against the model that actually loaded.

        The first hook after the checkpoint is resident, which makes it the first
        place two questions can be answered honestly: is this really a Klein
        engine, and does it really expose a regional-conditioning path.

        ``before_process`` guessed the first from a checkpoint header, which is a
        guess it is allowed to make because it has to redirect the prompt before
        anything is loaded. If it guessed wrong the boxes are dropped with a note
        -- not composed as Krea, because the prompt has already been assembled by
        then, and a second composition here would be a different prompt from the
        one the plan and the metadata describe.

        Nothing is encoded here. The references are ImageStitch's own and it
        registers them itself; what this generation decided about them is a
        scoped setting applied before this hook ran. See
        :func:`mc_spatial_klein.reference_override`.
        """
        request = self._klein
        if request is None:
            return

        arch = mc_spatial_klein.loaded_architecture()
        if not mc_spatial_klein.is_klein(arch):
            self._klein_note = (
                f"the layout was not applied because the checkpoint that loaded is "
                f"{getattr(arch, 'label', 'not a Klein model')}, not the FLUX.2 Klein "
                f"the Spatial panel was showing")
            logger.warning("Model Chain: Klein Spatial — %s", self._klein_note)
            self._klein = None
            return

        if request.regions:
            backend = mc_spatial_klein.select_backend()
            if backend is None:
                # §34, and the whole reason the backend is probed rather than
                # assumed. Raised rather than logged: generating globally with
                # the boxes silently discarded produces a plausible image that is
                # not the one that was asked for, and nothing about the result
                # would say so.
                message = mc_spatial_klein.compatibility_error(arch)
                logger.error("Model Chain: %s", message)
                # Released before raising: this hook's exception skips
                # postprocess entirely, and leaving the reference scope open
                # would hold a job's worth of state over a generation that never
                # happened.
                self._klein_release()
                raise RuntimeError(message)
            self._klein_backend = backend

        logger.info("Model Chain: %s",
                    mc_spatial_klein.describe(request, self._klein_backend))
        if request.references and not mc_arch.references_available(arch):
            # A pre-flight check, and the one signal a checkpoint name cannot
            # fake: the flag is set by the loader from the state dict it actually
            # read. A repacked build whose header promised a reference path its
            # engine does not implement fails here rather than silently.
            self._klein_note = ("the loaded engine does not expose Forge's reference "
                                "path, so the ImageStitch images did not reach this "
                                "generation")
            logger.warning("Model Chain: Klein Spatial — %s", self._klein_note)

        try:
            p.extra_generation_params.update(
                mc_spatial_klein.metadata(request, self._klein_backend,
                                          layout_serialized=self._klein_layout))
        except Exception:
            logger.debug("Model Chain: could not record the Klein Spatial metadata",
                         exc_info=True)

    def process_before_every_sampling(self, p, *args, **kwargs):
        """Attach the regions to the conditioning for the pass about to run.

        The narrowest hook that has both halves of what §29 needs: the positive
        conditioning, and the tensor whose shape decides where a normalized box
        lands. Per *pass* and not per generation, because hires fix samples a
        second time at a different size and a set compiled against the first grid
        would place every box against the wrong geometry.

        The previous pass's conditioning is removed before this one's is added.
        Two passes' worth of regions on one conditioning list would double every
        condition and halve nothing.
        """
        import contextlib

        request = self._klein
        if request is None or not request.regions:
            return

        if self._klein_pass is not None:
            try:
                self._klein_pass.close()
            except Exception:
                logger.warning("Model Chain: the previous Klein Spatial pass did not "
                               "unwind cleanly", exc_info=True)
            self._klein_pass = None

        conditioning = kwargs.get("c", kwargs.get("cond"))
        tensor = kwargs.get("x")
        if tensor is None:
            tensor = kwargs.get("noise")
        if conditioning is None or tensor is None:
            self._klein_note = ("the layout was not applied because this host did not "
                                "offer the conditioning and the latent for the "
                                "sampling pass")
            logger.warning("Model Chain: Klein Spatial — %s", self._klein_note)
            return

        stack = contextlib.ExitStack()
        try:
            stack.enter_context(mc_spatial_klein.regional_conditioning(
                request, conditioning, tensor=tensor,
                backend=self._klein_backend,
                model=getattr(p, "sd_model", None)))
        except Exception:
            stack.close()
            raise
        self._klein_pass = stack

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
        spatial_note = self._spatial_note or self._klein_note
        composed_raw = self._composed_without_creative
        self._complaint = ""
        self._spatial_note = ""
        self._composed_without_creative = False
        # The ordinary end of a Klein spatial job: references off the model, the
        # global reference toggle back where the user left it, the regional
        # conditioning gone. It runs before the early return below, because a
        # generation that produced nothing still has to put back what it moved.
        self._klein_release()
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
