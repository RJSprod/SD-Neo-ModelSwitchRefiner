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
import mc_infotext
import mc_llm_sessions as sessions
import mc_memory
import mc_plan
import mc_spatial
from modules import errors, scripts

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


SPATIAL_CONTROLS = 3
"""How many controls the Spatial block contributes: enabled, mode, layout."""


def _split(values) -> tuple[tuple, tuple, tuple]:
    """``before_process``'s tuple, cut into its three parts.

    ``ui()`` returns, after the enabled flag: four scalars, then three controls
    per axis, then the three Spatial controls. Two of those three lengths are
    fixed and the middle one is the library's, so the cut is made by *asking the
    library* rather than by pattern-matching a length -- both a layout with
    spatial and one without are multiples of three long, and a tuple cannot say
    which it is.

    Anything shorter than the axis block means there is no panel behind this
    call: an API request that sent only the flag, or a page built against a
    library that will not load. The saved settings answer for all of it, which
    is what those callers already got.
    """
    values = tuple(values or ())
    if len(values) < 4:
        return (), (), ()
    scalars, rest = values[:4], values[4:]
    try:
        from prompt_master.krea import library as library_module

        expected = len(library_module.library().axis_keys) * 3
    except Exception:
        return scalars, (), ()
    if len(rest) < expected:
        return scalars, rest, ()
    return scalars, rest[:expected], rest[expected:expected + SPATIAL_CONTROLS]


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
    creativity, seed, anti_repetition, loras = scalars
    return _stored(creativity, seed, anti_repetition, loras, axes)


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


def _stored(creativity, seed, anti_repetition, loras, axis_values) -> dict:
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
    stored["loras"] = mc_creative_krea.lora_suffix(loras)
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


def _toggled(enabled):
    """Show or hide the controls, and remember the toggle.

    Four outputs and three of them are the same update: the slider, the drawer
    and the Spatial block all appear and disappear with the feature, because
    none of them does anything while it is off.
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
    return shown, shown, gr.update(value=told, visible=bool(enabled)), shown


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


def _has_vocabulary() -> bool:
    """Whether the creativity library loaded, asked before the page is laid out.

    :func:`mc_creative_panel.build` asks the same question and answers it with a
    sentence on the page, but it is called from inside the drawer and the Spatial
    block is built above the drawer. The library is read once per process and
    cached, so asking twice costs a lock.
    """
    try:
        from prompt_master.krea import library as library_module

        library_module.library()
        return True
    except Exception:
        logger.debug("Model Chain: the creativity library could not be read; the "
                     "Spatial Layout controls were not built", exc_info=True)
        return False


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
            <option value="thirds">Thirds</option>
            <option value="center">Centre</option>
            <option value="both">Thirds + centre</option>
            <option value="none">None</option>
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
          LLM · region prompts bypass it</span>
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
                    placeholder="what is in this box, in your own words"></textarea>
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

        <div class="{SPATIAL_PREFIX}-numbers">
          <label class="{SPATIAL_PREFIX}-field">
            <span>X</span>
            <input type="number" id="{_spatial_id("x")}" min="0" max="{spatial.SCALE}"
                   step="1" inputmode="numeric" />
          </label>
          <label class="{SPATIAL_PREFIX}-field">
            <span>Y</span>
            <input type="number" id="{_spatial_id("y")}" min="0" max="{spatial.SCALE}"
                   step="1" inputmode="numeric" />
          </label>
          <label class="{SPATIAL_PREFIX}-field">
            <span>W</span>
            <input type="number" id="{_spatial_id("w")}" min="0" max="{spatial.SCALE}"
                   step="1" inputmode="numeric" />
          </label>
          <label class="{SPATIAL_PREFIX}-field">
            <span>H</span>
            <input type="number" id="{_spatial_id("h")}" min="0" max="{spatial.SCALE}"
                   step="1" inputmode="numeric" />
          </label>
        </div>
        <div class="{SPATIAL_PREFIX}-readout">
          <span>Box (0–{spatial.SCALE})</span>
          <code id="{_spatial_id("bbox")}">—</code>
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


def spatial_summary(serialized, enabled: bool = True) -> str:
    """The one line beside the Edit button: what is on the canvas right now.

    Rendered by the server so that a restored workflow and a freshly built page
    both say something true before any JavaScript has run, and repainted by the
    browser as the user draws. Two writers for one line is a thing to be careful
    about; the care is that both compute it from the same serialized layout.
    """
    from prompt_master.krea import spatial

    layout = spatial.parse(serialized)
    if layout.unreadable:
        return notice(" ".join(layout.notes) or "The saved layout could not be read.",
                      "warn")
    if not layout.regions:
        return notice("No regions yet. Press Edit Layout to draw one — the scene "
                      "prompt still goes through Creative Mode either way.")
    said = spatial.summarise(layout)
    if not enabled:
        return notice(f"{said} — Spatial Layout is off, so they are not applied.")
    return notice(f"{said}. Region prompts are used exactly as typed; the scene "
                  f"around them is written by Creative Mode.")


def _spatial_toggled(enabled, serialized):
    """Remember the Spatial toggle, and say what it now does."""
    mc_spatial.remember(**{mc_spatial.ENABLED: bool(enabled)})
    return spatial_summary(serialized, bool(enabled))


def _spatial_mode(mode, serialized, enabled):
    """Remember Smart or Direct, and say what the difference costs."""
    from prompt_master.krea import spatial

    mode = str(mode or "").strip().casefold()
    if mode not in spatial.COMPOSE_MODES:
        mode = spatial.SMART
    mc_spatial.remember(**{mc_spatial.COMPOSE_MODE: mode})
    if not enabled:
        return spatial_summary(serialized, bool(enabled))
    if mode == spatial.SMART:
        return notice("Smart Spatial Compose: a second, short language-model pass "
                      "rewrites the scene so it stops arguing with your boxes. It "
                      "cannot change a box, a region prompt or a visible text. It "
                      "costs one extra request per generation.")
    return notice("Direct BBOX Merge: the scene Creative Mode wrote is used as it "
                  "stands and no second request is made. Faster, and the control "
                  "half of an A/B against Smart.")


def _spatial_saved(serialized, enabled):
    """The browser saved a layout. Keep it, and repaint the summary."""
    mc_spatial.remember(**{mc_spatial.LAYOUT: str(serialized or "")})
    return spatial_summary(serialized, bool(enabled))


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
    if setup.spatial:
        from prompt_master.krea import spatial as spatial_module

        drawn = spatial_module.parse(setup.spatial_layout)
        lines.append(f"{spatial_module.summarise(drawn)}"
                     f"{f' · {setup.spatial_compose_mode} merge' if setup.spatial_compose_mode else ''}")
        for region in drawn.ordered:
            body = region.text if region.kind == spatial_module.TEXT else region.prompt
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
    if setup.loras:
        remembered[mc_creative_krea.LORAS] = setup.loras
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

    # The canvas, restored as one document. The state box is this script's own
    # component, so writing to it is nothing like writing to the prompt box --
    # and it is what makes "restore the workflow" mean the whole workflow rather
    # than the half of it that fits in a settings file.
    layout_update, spatial_on, mode_update = gr.update(), gr.update(), gr.update()
    if setup.spatial:
        from prompt_master.krea import spatial as spatial_module

        readable = (setup.spatial_version is None
                    or setup.spatial_version == spatial_module.VERSION)
        if readable:
            mc_spatial.remember(**{mc_spatial.LAYOUT: setup.spatial_layout,
                                   mc_spatial.ENABLED: True})
            layout_update = gr.update(value=setup.spatial_layout)
            spatial_on = gr.update(value=True)
            said.append("The spatial canvas is back as it was drawn, and Spatial "
                        "Layout is on.")
            mode = str(setup.spatial_compose_mode or "").strip().casefold()
            if mode in spatial_module.COMPOSE_MODES:
                mc_spatial.remember(**{mc_spatial.COMPOSE_MODE: mode})
                mode_update = gr.update(value=mode)
        else:
            # Refused, not migrated. §8.4 is explicit that exact replay never
            # depended on this record -- the picture is reproducible from its own
            # Prompt line -- so the honest thing to do with a layout from a later
            # build is leave it alone and say so.
            said.append("The recorded spatial layout is from a later version and "
                        "was not restored; the canvas is untouched.")

    kind = "warn" if setup.warnings() or (replay_exactly and not setup.replayable) \
        else "info"
    told = notice(" ".join(said), kind)
    return (gr.update(value=setup.source) if setup.source else gr.update(),
            gr.update(value=True), told, gr.update(value=_pasted_view()),
            layout_update, spatial_on, mode_update,
            gr.update(value=spatial_summary(
                setup.spatial_layout if setup.spatial else
                mc_spatial.settings()["layout"], True)))


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
        vocabulary = _has_vocabulary()

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

            # Spatial Layout sits above the drawer rather than inside it. It is
            # a decision about *this* picture -- where the subjects go -- and the
            # drawer holds decisions about how this installation does art
            # direction. It is hidden with the slider when Creative Mode is off,
            # because a composition guide with nothing to compose around is a
            # control that cannot do anything.
            #
            # Built only when the creativity library loaded, and that is not a
            # dependency this feature has -- the compositor needs no vocabulary
            # at all. It is that a Spatial Layout composes boxes around a scene
            # Creative Mode wrote, and an installation whose Creative Mode
            # cannot run has no scene for them to go around. One sentence on the
            # page beats two dead controls under it.
            with gr.Group(visible=bool(stored["enabled"]) and vocabulary,
                          elem_id=ident("spatial")) as spatial_group:
                with gr.Row(elem_id=ident("spatial", "bar")):
                    spatial_enabled = gr.Checkbox(
                        value=bool(spatial["enabled"]), label="Spatial Layout", scale=1,
                        elem_id=ident("spatial", "toggle"),
                        info="place subjects with bounding boxes")
                    spatial_compose = gr.Radio(
                        choices=[("Smart Spatial Compose", spatial_module.SMART),
                                 ("Direct BBOX Merge", spatial_module.DIRECT)],
                        value=spatial["compose_mode"], label="Composition", scale=2,
                        elem_id=ident("spatial", "compose"))
                    edit = gr.Button("Edit Layout…", size="sm", scale=1,
                                     elem_id=_spatial_id("open"))
                spatial_status = gr.HTML(
                    spatial_summary(spatial["layout"], bool(spatial["enabled"])),
                    elem_id=ident("spatial", "status"))
                # The one component the browser writes to, and the one that
                # travels with the generation. Hidden rather than absent: the
                # editor is a page, the compositor is a hook, and a hidden
                # textbox is the only thing Gradio offers that is both.
                spatial_state = gr.Textbox(
                    value=spatial["layout"], visible=False, lines=1,
                    elem_id=_spatial_id("state"))
                gr.HTML(spatial_editor(), elem_id=_spatial_id("editor"))

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

                record_scenes = gr.Checkbox(
                    value=bool(spatial["record_scenes"]),
                    label="Record the scene before and after the composer pass",
                    elem_id=ident("spatial", "scenes"),
                    info="spatial images only; makes a Smart/Direct comparison "
                         "readable afterwards")

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

        self.panel = panel
        self.components = {
            "enabled": enabled, "creativity": creativity, "status": status,
            "controls": controls, "show": show, "recipe": recipe, "expanded": expanded,
            "pasted": pasted, "replay": exactly, "restore": restore, "disarm": disarm,
            "spatial_group": spatial_group, "spatial_enabled": spatial_enabled,
            "spatial_compose": spatial_compose, "spatial_status": spatial_status,
            "spatial_state": spatial_state, "spatial_edit": edit,
            "spatial_scenes": record_scenes}
        if panel is not None:
            self.components.update(panel.components())

        self._wire(enabled, creativity, status, controls, show, recipe, expanded,
                   pasted, exactly, restore, disarm, spatial_group)
        self._wire_spatial(spatial_enabled, spatial_compose, spatial_status,
                           spatial_state, record_scenes)
        self._register_paste_fields()

        # Every control travels to before_process, because that is where the
        # roll happens and the panel is what the user is looking at. They are
        # this script's own arguments and reach neither Model Chain's preset
        # list nor its infotext.
        #
        # The Spatial controls go last and stay last. _split() cuts this tuple
        # by asking the library how long the axis block is, so the two fixed
        # ends are the two that can be read without counting.
        spatial_controls = [spatial_enabled, spatial_compose, spatial_state]
        if panel is None:
            self.arguments = [enabled, creativity]
        else:
            self.arguments = ([enabled, creativity] + list(panel.settings_controls)
                              + list(panel.axis_controls) + spatial_controls)
        return list(self.arguments)

    def _wire(self, enabled, creativity, status, controls, show, recipe, expanded,
              pasted, exactly, restore, disarm, spatial_group):
        """Every handler this file owns, in one place. The panel wires its own.

        All of them ``queue=False``: not one of these does any work worth
        queueing, and none of them starts, stops or waits for a generation. The
        panel is settings and nothing else now -- the roll is in
        :meth:`before_process`, which is not reachable from here.
        """
        enabled.change(fn=_toggled, inputs=[enabled],
                       outputs=[creativity, controls, status, spatial_group],
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
        spatial_outputs = [self.components["spatial_state"],
                           self.components["spatial_enabled"],
                           self.components["spatial_compose"],
                           self.components["spatial_status"]]
        if self.prompt_box is not None:
            restore.click(fn=_restore_setup, inputs=[exactly],
                          outputs=[self.prompt_box, enabled, status, pasted]
                                  + spatial_outputs,
                          queue=False)
        else:
            logger.debug("Model Chain: the txt2img prompt box was not offered to "
                         "Creative Mode; Restore Creative setup will not fill it in")
            restore.click(fn=lambda exactly: _restore_setup(exactly)[1:],
                          inputs=[exactly],
                          outputs=[enabled, status, pasted] + spatial_outputs,
                          queue=False)
        disarm.click(fn=_disarm_replay, outputs=[status], queue=False)

    def _wire_spatial(self, spatial_enabled, spatial_compose, spatial_status,
                      spatial_state, record_scenes):
        """The Spatial controls' handlers. Four, and none of them generates.

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
        spatial_enabled.change(fn=_spatial_toggled,
                               inputs=[spatial_enabled, spatial_state],
                               outputs=[spatial_status], queue=False)
        spatial_compose.change(fn=_spatial_mode,
                               inputs=[spatial_compose, spatial_state, spatial_enabled],
                               outputs=[spatial_status], queue=False)
        spatial_state.change(fn=_spatial_saved,
                             inputs=[spatial_state, spatial_enabled],
                             outputs=[spatial_status], queue=False)
        record_scenes.change(fn=_spatial_scenes, inputs=[record_scenes],
                             outputs=[spatial_status], queue=False)

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
                self.components, notice=notice, view=_pasted_view)
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
        """
        if not enabled:
            return
        import mc_broker

        self._complaint = ""
        if self._rolling:
            # A process_images() nested inside our own roll. There is nothing to
            # do for it and a great deal to get wrong.
            logger.debug("Model Chain: Creative Mode is already rolling; the nested "
                         "generation is left alone")
            return

        self._spatial_note = ""
        settings = _settings_for(args)
        layout = self._layout(p, args)
        self._publish_plan(p, layout)

        # Both passes, inside one declaration and under one re-entrancy flag.
        #
        # ``host_job`` is what tells ``mc_llm_sessions._Gpu.acquire`` that the
        # image job is blocked waiting for this request rather than competing
        # with it, and it has to cover the Spatial Composer as much as the
        # writer: pass 2 runs at the same point in the same hook, with the same
        # generation waiting on it, and one started outside this block would
        # wait for the job that is waiting for it. It is re-entrant and
        # thread-local, so declaring it once here is the whole of it.
        written = None
        self._rolling = True
        try:
            with mc_broker.host_job():
                rolled = self._roll(p, settings, layout)
                if rolled is not None:
                    written = self._spatially(p, rolled, settings, layout)
        finally:
            self._rolling = False
        if written is None:
            return

        p.prompt = written.generation
        try:
            p.extra_generation_params.update(written.metadata)
        except Exception:
            logger.debug("Model Chain: could not record the Creative Mode metadata",
                         exc_info=True)
        logger.info("Model Chain: Creative Mode prompt applied — %s characters from a "
                    "%s-character source at creativity %s, creative seed %s",
                    f"{len(written.generation):,}", f"{len(written.roll.source):,}",
                    written.roll.creativity, written.roll.creative_seed)

    def _publish_plan(self, p, layout) -> None:
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
            mc_plan.publish(mc_plan.build_for(p, creative=True, spatial_compose=compose))
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
        self._complaint = ""
        self._spatial_note = ""
        if processed is None:
            return
        try:
            if complaint:
                processed.comments += (
                    f"\nModel Chain: Creative Mode did not write this prompt — "
                    f"{complaint}. The image was generated from the prompt exactly as "
                    "typed. The console and LLM Studio → Setup say more.")
            if spatial_note:
                processed.comments += (
                    f"\nModel Chain: Spatial Layout — {spatial_note}. Your regions "
                    "have not been changed.")
        except Exception:
            logger.debug("Model Chain: could not put the Creative Mode notice on the "
                         "result", exc_info=True)

    def _layout(self, p, values):
        """This generation's spatial layout, or an empty one.

        Parsed before the roll rather than after it, because pass 1 has to know
        whether a layout exists: the one sentence it is told about placement is
        added only when there is a layout to justify it, and adding it after the
        writer had already written would be adding it to nothing.

        An empty answer -- Spatial Layout switched off, nothing drawn, or a
        layout this build cannot read -- is the answer that makes the rest of
        this generation exactly the Creative Mode generation it would have been
        before this feature existed.
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

    def _spatially(self, p, rolled, settings, layout):
        """The scene, the boxes, and the one structured prompt built from both.

        Everything after pass 1 and before ``p.prompt``. The order is the design
        intent's §3 exactly and each step can only fail *backwards*:

        * no regions -> the writer's paragraph, which is Creative Mode as it was;
        * Direct, or a Composer that failed -> that paragraph as the scene;
        * a compositor that raised -> that paragraph, and a sentence saying the
          layout was not applied.

        None of those refuses a generation, and none of them silently drops a
        box: every one of them says what happened, in the log and on the result.
        """
        from prompt_master.krea import spatial

        loras = settings.get("loras", "")
        if not layout.regions:
            return mc_creative_krea.prepare(rolled, loras, settings)

        enhanced = rolled.expanded
        composed = None
        scene, background = enhanced, ""
        if layout.compose_mode == spatial.SMART:
            composed = mc_spatial.compose(
                source=rolled.source, scene=enhanced, layout=layout,
                ratio=spatial.aspect_ratio(getattr(p, "width", 0),
                                           getattr(p, "height", 0)),
                seed=mc_spatial.composer_seed(rolled.creative_seed),
                reserve=mc_creative_krea.image_reserve_bytes())
            # The last LLM phase of this plan has now finished, so this is where
            # the card goes back to the image side. The roll deliberately did
            # not do it: between the writer and the Composer the only way to
            # free image VRAM is to stop the server the Composer was about to
            # use, which buys nothing and costs a GGUF load mid-generation.
            #
            # Run whether or not the Composer succeeded. A Composer that failed
            # still leaves the same card behind it, and the image pass that
            # follows needs the same room either way.
            freed = mc_creative_krea.hand_back_vram()
            if freed:
                logger.info("Model Chain: freed %.1f GB for the image generation that "
                            "follows the Spatial Composer", freed / (1024 ** 3))
            if composed.ran:
                scene, background = composed.scene, composed.background
            else:
                # Direct merge is the answer to every Composer failure. The
                # boxes are the user's and are unaffected; what is lost is the
                # de-conflicting of the global scene, and saying so is the
                # difference between a fallback and a mystery.
                logger.warning("Model Chain: the Spatial Composer did not run (%s); "
                               "merging the layout directly instead", composed.failed)
                self._spatial_note = (f"the scene was merged directly because the "
                                      f"Spatial Composer did not run — "
                                      f"{composed.failed}")

        try:
            prompt = spatial.compose(
                layout, scene=scene, background=background,
                ratio=spatial.aspect_ratio(getattr(p, "width", 0),
                                           getattr(p, "height", 0)))
        except Exception as exc:
            errors.report("Model Chain: the spatial compositor failed", exc_info=True)
            self._spatial_note = (f"the Spatial Layout was not applied because the "
                                  f"compositor failed ({exc})")
            return mc_creative_krea.prepare(rolled, loras, settings)

        metadata = mc_spatial.metadata(
            layout, compose_mode=layout.compose_mode, composed=composed,
            enhanced=enhanced, record_scenes=self._record_scenes)
        logger.info("Model Chain: Spatial Layout applied — %s element%s, %s merge, "
                    "%s characters of structured prompt",
                    len(layout.regions), "" if len(layout.regions) == 1 else "s",
                    layout.compose_mode, f"{len(prompt):,}")
        return mc_creative_krea.prepare(rolled, loras, settings, prompt=prompt,
                                        spatial=metadata)

    def _roll(self, p, settings, layout):
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
        """
        source = str(getattr(p, "prompt", "") or "").strip()
        if not source:
            logger.info("Model Chain: Creative Mode has no source prompt to work from")
            self._complaint = "there was no prompt to work from"
            return None

        session = mc_creative_krea.creative

        # Held by name and closed explicitly, the way the roll itself holds the
        # LLM run: every path out of the loop below leaves the generator
        # suspended, and it is its ``finally`` that gives the progress bar and
        # the workload lock back. Closing it is what runs that now rather than
        # whenever the interpreter next collects the frame.
        events = session.roll(source, settings, guard_checkpoint=True, own_bar=False,
                              spatial_layout=layout)
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
