"""The Creative Mode control surface, built once and used by both surfaces.

    Profile: Default                       [Save] [Save As] [Delete]
    Active directions
      Medium      Fixed · Fashion editorial                    [Edit]
      Lighting    Vary · excludes 2                            [Edit]
    [+ Add direction ▾]

That is the whole panel when two axes have been configured, and one line shorter
for every axis that has not. What replaced it was ten permanent rows of radio
buttons and ten permanent "Fixed value" dropdowns -- twenty controls describing
decisions nobody had made yet, nine of which said Vary because the factory
defaults said Vary, which is art direction arriving from nowhere.

The rule this file follows
--------------------------
**Show the decisions the user has made, not every decision the software knows
how to make.** Natural is the absence of a decision, so a Natural axis has no
row; returning an axis to Natural removes its row; and the editor for an axis
exists only while that axis is being edited.

How that is done in Gradio
--------------------------
Gradio 4 cannot create a component after the page is built, so every row and
every editor is built up front and hidden. ``visible=False`` in Gradio removes
the element from the layout rather than making it transparent, so a hidden
editor costs no space -- which is what makes "build them all, show one" the same
thing on screen as "create one on demand", and a great deal simpler than a
component pool.

Everything is a stock component: Accordion, Row, Group, Dropdown, Radio,
Checkbox, Slider, Number, Textbox, Button, Markdown. No custom HTML application,
no Gradio-generated class in any selector, and every element carries an
extension-owned id. A theme decides what all of it looks like.

One render, many handlers
-------------------------
Every handler ends in :meth:`Panel.render`, which returns an update for every
component the panel owns, computed from the stored settings. Handlers therefore
never have to work out which *other* controls their change affects -- setting an
axis to Fixed shows its value dropdown, hides its exclusions, rewrites its
summary line and removes it from the add-a-direction list, and the handler that
did it only said "the mode is now Fixed".

The alternative -- each handler updating the components it believes it touched --
is how a panel ends up showing an exclusion list for an axis that is no longer
varying. It costs one wide outputs list per handler, which is a thing to read
once, rather than a class of bug to find repeatedly.
"""

from __future__ import annotations

import logging

import gradio as gr

import mc_creative_krea
import mc_creative_profiles as profiles

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

CLASS_PREFIX = "mc-creative"
"""Class names this panel puts in the page, shared by both surfaces.

The ids differ between txt2img and LLM Studio -- they are in different tabs and
an id is unique to a page -- but the *layout* is one layout, so the classes that
style it are one set of classes. ``style.css`` scopes to these and to nothing
Gradio generated.
"""

ADD_LABEL = "+ Add direction"
NO_DIRECTIONS = ("**Active direction:** None. Creative Mode is not influencing "
                 "any axis — the prompt is expanded as written.")
DIRECTIONS_HEADING = "**Active directions**"


def classes(*names: str) -> list[str]:
    return [f"{CLASS_PREFIX}-{name}" for name in names]


# --------------------------------------------------------------------------- #
# What one axis row says about itself
# --------------------------------------------------------------------------- #


def summarise(axis, setting) -> str:
    """One axis's whole configuration as one short line.

    ``Medium · Fixed: Fashion editorial``. ``Texture · Vary · excludes glossy
    plastic, heavy impasto``. Labels rather than ids, because this is the line
    somebody reads to check they configured what they meant; the ids are in the
    diagnostics view and in the PNG metadata, where they are what matters.
    """
    from prompt_master.krea import director

    label = getattr(axis, "label", "")
    mode = setting.get("mode", director.NATURAL)
    if mode == director.FIXED:
        pinned = setting.get("fixed")
        chosen = axis.variant(pinned) if pinned else None
        if chosen is None:
            return f"**{label}** · Fixed · *no treatment chosen yet*"
        return f"**{label}** · Fixed: {chosen.label}"

    excluded = [axis.variant(identifier) for identifier in setting.get("excluded") or ()]
    named = [variant.label for variant in excluded if variant is not None]
    if not named:
        return f"**{label}** · Vary"
    return f"**{label}** · Vary · excludes {', '.join(named)}"


def _axis_setting(stored, key) -> dict:
    from prompt_master.krea import director

    return {
        "mode": (stored.get("axis_modes") or {}).get(key, director.NATURAL),
        "fixed": (stored.get("fixed_values") or {}).get(key),
        "excluded": list((stored.get("excluded_values") or {}).get(key) or ()),
    }


# --------------------------------------------------------------------------- #
# What a direction costs before the image starts
# --------------------------------------------------------------------------- #

COST_SAMPLES = 3
"""Rolls to average the brief's length over.

One would do at Creativity 10, where every eligible axis activates and the
length is settled. Lower down the activation count is itself a draw, so one
sample is one of the possibilities rather than the typical one.
"""

COST_SOURCE = "a lighthouse in a storm"
"""A stand-in prompt for the estimate. Its own length is subtracted back out."""


def brief_cost(stored=None) -> tuple[int, float]:
    """``(characters, seconds)`` this configuration adds to every press.

    The Krea instruction is the same bytes on every roll, so llama.cpp's prompt
    cache answers for it and it costs nothing after the first request to a given
    server. The brief is different every roll by construction -- that is what
    Vary *means* -- so it can never be cached, and it is therefore the whole of
    what a press pays before the writing starts.

    The seconds are this machine's own measurement, out of the same store the
    progress bar predicts from, falling back to the built-in guess until a roll
    has been timed. Both numbers are honest about being estimates: which axes
    activate is a draw, so the length varies roll to roll around this.
    """
    from prompt_master.krea import director

    stored = stored or mc_creative_krea.settings()
    try:
        settings = mc_creative_krea.axis_settings(stored)
        creativity = int(stored.get("creativity", 5))
        lengths = [len(director.roll(COST_SOURCE, creativity, seed, settings).brief)
                   for seed in range(1, COST_SAMPLES + 1)]
    except Exception:
        logger.debug("Model Chain: could not size the creative brief", exc_info=True)
        return 0, 0.0

    characters = int(sum(lengths) / len(lengths)) if lengths else 0
    if not characters:
        return 0, 0.0
    try:
        import mc_progress

        per_character = mc_progress.measured("krea:read", 0.0028)
    except Exception:
        per_character = 0.0028
    return characters, characters * float(per_character)


def describe_cost(stored=None) -> str:
    """The line under the directions: what they cost, in this machine's seconds.

    Said where the decision is made rather than only on the progress bar, which
    is the first place anybody looks and the last place they can act on it. A
    user who wants the image to start sooner has three levers -- fewer
    directions, a lower Creativity, and where the language model runs -- and two
    of the three are on this panel.
    """
    characters, seconds = brief_cost(stored)
    if not characters:
        return ("*Nothing extra to read: with no directions the writer is handed the "
                "prompt as you typed it.*")

    where = _placement_note()
    return (f"*About {characters:,} characters of brief — roughly {seconds:.0f}s of "
            f"reading before the writing starts{where}. The brief is different every "
            "roll, so it is the one part of the request that can never come out of the "
            "model's cache.*")


def _placement_note() -> str:
    """", on the card" or ", from system RAM", when that can be told.

    Where the language model runs is the largest of the three levers by a wide
    margin -- the same machine that reads at 36 tokens a second with the weights
    in RAM reads at 900 with them on the card -- and it is the one this panel
    cannot change, so it is named rather than acted on.
    """
    try:
        import mc_llm_runtime

        configuration = mc_llm_runtime.config()
    except Exception:
        return ""
    if not configuration.configured:
        return ""
    return " on the card" if configuration.on_gpu else " with the model in system RAM"


# --------------------------------------------------------------------------- #
# Writing one change back
# --------------------------------------------------------------------------- #


def set_axis(key, *, mode=None, fixed=..., excluded=...) -> dict:
    """Change one axis and hand back the settings as they now are.

    Read-modify-write against the whole stored mapping rather than against the
    panel's own components, so that two surfaces open in two tabs cannot each
    write back a stale copy of the other's axes. Ellipsis rather than ``None``
    for the two value arguments, because ``None`` is a meaningful value for both:
    it is how a Fixed pin and an exclusion list are cleared.
    """
    stored = mc_creative_krea.settings()
    modes = dict(stored.get("axis_modes") or {})
    pinned = dict(stored.get("fixed_values") or {})
    excluded_ids = {name: list(values)
                    for name, values in (stored.get("excluded_values") or {}).items()}

    if mode is not None:
        modes[key] = mode
    if fixed is not ...:
        if fixed:
            pinned[key] = str(fixed)
        else:
            pinned.pop(key, None)
    if excluded is not ...:
        chosen = [str(value) for value in (excluded or ()) if str(value)]
        if chosen:
            excluded_ids[key] = chosen
        else:
            excluded_ids.pop(key, None)

    mc_creative_krea.remember(**{mc_creative_krea.AXIS_MODES: modes,
                                 mc_creative_krea.FIXED_VALUES: pinned,
                                 mc_creative_krea.EXCLUDED_VALUES: excluded_ids})
    return mc_creative_krea.settings()


# --------------------------------------------------------------------------- #
# The panel
# --------------------------------------------------------------------------- #


class Panel:
    """Every Creative control below the toggle, and one render for all of them.

    Built by :func:`build`. The surface supplies the three controls it owns and
    the panel does not -- the status line, the Creativity slider and its own
    element-id function -- because those sit outside the drawer on both surfaces
    and each surface styles its own.
    """

    def __init__(self, ident, notice, status, creativity, keys, axes):
        self.ident = ident
        self.notice = notice
        self.status = status
        self.creativity = creativity
        self.keys = list(keys)
        self.axes = dict(axes)

        # Filled in by build(), in the order the page is assembled.
        self.profile = None
        self.name_row = None
        self.profile_name = None
        self.summary = None
        self.cost = None
        self.add = None
        self.seed = None
        self.anti = None
        self.loras = None
        self.rows: dict = {}
        self.labels: dict = {}
        self.editors: dict = {}
        self.modes: dict = {}
        self.fixed: dict = {}
        self.excluded: dict = {}
        self.buttons: list = []

    # -- what a handler answers with --------------------------------------- #

    @property
    def axis_controls(self) -> list:
        """The three controls per axis, flat, in the library's own axis order.

        This is the list that travels to the generation, and the order is the
        contract: mode, fixed value, exclusions, for each axis in turn. It is
        parsed back in exactly one place on each surface, which is what keeps a
        misalignment from becoming a wrong setting nobody notices.
        """
        return [control for key in self.keys
                for control in (self.modes[key], self.fixed[key], self.excluded[key])]

    @property
    def settings_controls(self) -> list:
        """The secondary settings, in the order a surface passes them on."""
        found = [self.seed, self.anti]
        if self.loras is not None:
            found.append(self.loras)
        return found

    def outputs(self) -> list:
        """Every component :meth:`render` returns an update for, in order."""
        found = [self.status, self.creativity, self.profile, self.name_row,
                 self.profile_name, self.summary, self.cost, self.add, self.seed,
                 self.anti]
        if self.loras is not None:
            found.append(self.loras)
        for key in self.keys:
            found.extend([self.rows[key], self.labels[key], self.editors[key],
                          self.modes[key], self.fixed[key], self.excluded[key]])
        return found

    def components(self) -> dict:
        """Everything this panel owns, for a surface to expose and a test to read."""
        found = {"profile": self.profile, "profile_name": self.profile_name,
                 "name_row": self.name_row, "summary": self.summary, "cost": self.cost,
                 "add": self.add,
                 "seed": self.seed, "anti": self.anti,
                 "axes": self.axis_controls,
                 "rows": [self.rows[key] for key in self.keys],
                 "labels": [self.labels[key] for key in self.keys],
                 "editors": [self.editors[key] for key in self.keys],
                 "buttons": list(self.buttons)}
        if self.loras is not None:
            found["loras"] = self.loras
        return found

    def render(self, stored=None, editing=None, told=None, kind="info",
               profile=None, naming=False) -> list:
        """An update for every component this panel owns.

        ``editing`` is the one axis whose editor is open, or ``None`` for none.
        ``told`` is a sentence for the status line, or ``None`` to leave whatever
        is there. ``naming`` opens the Save As name box.
        """
        from prompt_master.krea import director

        stored = stored or mc_creative_krea.settings()
        modes = stored.get("axis_modes") or {}
        active = [key for key in self.keys
                  if modes.get(key) in (director.VARY, director.FIXED)]
        natural = [key for key in self.keys if key not in active]

        updates = [
            gr.update(value=self.notice(told, kind)) if told is not None else gr.update(),
            gr.update(value=stored.get("creativity")),
            gr.update(choices=profiles.choices(),
                      value=profile) if profile is not None
            else gr.update(choices=profiles.choices()),
            gr.update(visible=bool(naming)),
            gr.update(value="") if not naming else gr.update(),
            gr.update(value=DIRECTIONS_HEADING if active else NO_DIRECTIONS),
            gr.update(value=describe_cost(stored)),
            gr.update(choices=[(self.axes[key].label, key) for key in natural],
                      value=None, visible=bool(natural)),
            gr.update(value=stored.get("seed")),
            gr.update(value=bool(stored.get("anti_repetition"))),
        ]
        if self.loras is not None:
            updates.append(gr.update(value=stored.get("loras", "")))

        for key in self.keys:
            setting = _axis_setting(stored, key)
            mode = setting["mode"]
            open_here = editing == key
            updates.extend([
                gr.update(visible=key in active),
                gr.update(value=summarise(self.axes[key], setting)),
                gr.update(visible=open_here),
                gr.update(value=mode),
                gr.update(value=setting["fixed"],
                          visible=open_here and mode == director.FIXED),
                gr.update(value=setting["excluded"],
                          visible=open_here and mode == director.VARY),
            ])
        return updates


# --------------------------------------------------------------------------- #
# Building it
# --------------------------------------------------------------------------- #


def build(ident, notice, status, creativity, *, loras=True, stored=None) -> Panel | None:
    """Assemble the panel into whatever container is open. ``None`` if it cannot.

    A creativity library that will not load leaves a sentence on the page saying
    so and returns nothing: Creative Mode has no vocabulary to direct with, and a
    panel of empty dropdowns would invite somebody to configure a feature that
    cannot run.
    """
    from prompt_master.krea import director

    try:
        from prompt_master.krea import library as library_module

        lib = library_module.library()
    except Exception as exc:
        gr.Markdown(f"The creativity library could not be read, so Creative Mode has "
                    f"no vocabulary to direct with: {exc}")
        return None

    stored = stored or mc_creative_krea.settings()
    keys = list(lib.axis_keys)
    panel = Panel(ident, notice, status, creativity, keys,
                  {key: lib.axis(key) for key in keys})

    def button(label, *parts, **kwargs):
        made = gr.Button(label, size="sm", elem_id=ident(*parts), **kwargs)
        panel.buttons.append(made)
        return made

    # -- the profile bar --------------------------------------------------- #
    # Shown, not applied. The dropdown opens on the profile the live settings
    # were last loaded from, and the settings are whatever they were left as: a
    # panel that reapplied its default every time a tab opened would silently
    # discard whatever the last tab adjusted.
    current = profiles.selected()
    complaint = profiles.default_profile()[2]
    with gr.Group(elem_id=ident("profiles"), elem_classes=classes("profiles")):
        with gr.Row(elem_classes=classes("profile-bar")):
            panel.profile = gr.Dropdown(
                label="Profile", choices=profiles.choices(), value=current, scale=3,
                elem_id=ident("profile"), filterable=False,
                info="a named set of Creative settings; Factory is the neutral one")
        with gr.Row(elem_classes=classes("profile-actions")):
            save = button("Save", "profile", "save")
            save_as = button("Save As", "profile", "save-as")
            drop = button("Delete", "profile", "delete", variant="stop")
            make_default = button("Set as default", "profile", "default")
            reset = button("Reset to default", "profile", "reset")
        with gr.Row(visible=False, elem_classes=classes("profile-name")) as name_row:
            panel.name_row = name_row
            panel.profile_name = gr.Textbox(
                label="New profile name", value="", scale=3, max_lines=1,
                placeholder="Editorial portraits", elem_id=ident("profile", "name"))
            create = button("Create", "profile", "create", variant="primary")

    # -- the active directions --------------------------------------------- #
    active_now = [key for key in keys
                  if _axis_setting(stored, key)["mode"] in (director.VARY, director.FIXED)]
    panel.summary = gr.Markdown(DIRECTIONS_HEADING if active_now else NO_DIRECTIONS,
                                elem_id=ident("summary"),
                                elem_classes=classes("summary"))

    editor_buttons: list = []

    for key in keys:
        axis = lib.axis(key)
        setting = _axis_setting(stored, key)
        active = setting["mode"] in (director.VARY, director.FIXED)

        with gr.Row(visible=active, elem_id=ident("row", key),
                    elem_classes=classes("direction")) as row:
            panel.rows[key] = row
            panel.labels[key] = gr.Markdown(
                summarise(axis, setting), elem_id=ident("row", key, "summary"),
                elem_classes=classes("direction-summary"))
            edit = button("Edit", "row", key, "edit")

        with gr.Group(visible=False, elem_id=ident("editor", key),
                      elem_classes=classes("editor")) as editor:
            panel.editors[key] = editor
            gr.Markdown(f"### {axis.label}", elem_classes=classes("editor-title"))
            panel.modes[key] = gr.Radio(
                label="How this axis behaves",
                choices=[("Natural", director.NATURAL), ("Vary", director.VARY),
                         ("Fixed", director.FIXED)],
                value=setting["mode"], elem_id=ident("editor", key, "mode"),
                info="Natural leaves it out of the brief entirely")
            panel.fixed[key] = gr.Dropdown(
                label="Always use", value=setting["fixed"],
                choices=[(variant.label, variant.identifier) for variant in axis.variants],
                visible=False, elem_id=ident("editor", key, "fixed"),
                info="repeated every roll, unless your own words say otherwise")
            panel.excluded[key] = gr.Dropdown(
                label="Exclude choices", value=setting["excluded"], multiselect=True,
                choices=[(variant.label, variant.identifier) for variant in axis.variants],
                visible=False, elem_id=ident("editor", key, "excluded"),
                info="the director may choose anything except these")
            with gr.Row(elem_classes=classes("editor-actions")):
                done = button("Done", "editor", key, "done", variant="primary")
                natural = button("Return to Natural", "editor", key, "natural")

        # Wired at the end, not here: every handler answers with an update for
        # every component the panel owns, and half of them do not exist yet.
        editor_buttons.append((key, edit, done, natural))

    panel.cost = gr.Markdown(describe_cost(stored), elem_id=ident("cost"),
                             elem_classes=classes("cost"))

    natural_now = [key for key in keys
                   if _axis_setting(stored, key)["mode"] not in (director.VARY,
                                                                 director.FIXED)]
    panel.add = gr.Dropdown(
        label=ADD_LABEL, value=None, elem_id=ident("add"),
        choices=[(lib.axis(key).label, key) for key in natural_now],
        visible=bool(natural_now), elem_classes=classes("add"), filterable=False,
        info="choose an axis to give a direction; everything else stays Natural")

    # -- the secondary settings -------------------------------------------- #
    with gr.Accordion("Settings", open=False, elem_id=ident("settings"),
                      elem_classes=classes("settings")):
        panel.seed = gr.Number(
            label="Creative seed", value=stored["seed"], precision=0,
            elem_id=ident("seed"),
            info="-1 rolls new art direction each time; a fixed value repeats it. "
                 "Not the image seed.")
        panel.anti = gr.Checkbox(
            value=bool(stored["anti_repetition"]), label="Avoid recent treatments",
            elem_id=ident("anti"),
            info="pushes the last few rolls' choices away at high Creativity")
        forget = button("Clear recent memory", "forget")
        if loras:
            panel.loras = gr.Textbox(
                label="Pinned LoRAs", value=stored["loras"],
                placeholder="<lora:name:0.8> <lora:other:0.5>", elem_id=ident("loras"),
                info="appended to the generated prompt; never sent to the language model")

    for key, edit, done, natural in editor_buttons:
        _wire_axis(panel, key, edit, done, natural)
    _wire_profiles(panel, save, save_as, create, drop, make_default, reset)
    _wire_settings(panel, forget)

    if complaint:
        logger.warning("Model Chain: %s", complaint)
    return panel


# --------------------------------------------------------------------------- #
# The handlers
# --------------------------------------------------------------------------- #


def _wire_axis(panel, key, edit, done, natural) -> None:
    """The five ways one axis changes, all ending in one full render.

    ``queue=False`` throughout: none of these does work worth queueing, and none
    of them starts, stops or waits for a generation. They read a settings file,
    write a settings file and redraw a drawer.
    """
    from prompt_master.krea import director

    label = panel.axes[key].label

    def open_editor():
        return panel.render(editing=key)

    def change_mode(mode):
        mode = str(mode or director.NATURAL).casefold()
        if mode not in director.MODES:
            mode = director.NATURAL
        stored = set_axis(key, mode=mode)
        if mode == director.NATURAL:
            told = f"{label} is Natural again — it is left out of the brief entirely."
        elif mode == director.FIXED:
            told = f"{label} is Fixed. Choose the treatment to repeat."
        else:
            told = f"{label} varies. Exclude any treatments you never want."
        return panel.render(stored, editing=key, told=told)

    def change_fixed(value):
        stored = set_axis(key, fixed=value or None)
        chosen = panel.axes[key].variant(str(value)) if value else None
        told = (f"{label} is fixed to {chosen.label}." if chosen is not None
                else f"{label} has no treatment chosen yet.")
        return panel.render(stored, editing=key, told=told)

    def change_excluded(values):
        chosen = list(values or ())
        stored = set_axis(key, excluded=chosen)
        counted = len(chosen)
        told = (f"{label} may choose anything: nothing is excluded." if not counted
                else f"{label} will never choose {counted} "
                     f"{'treatment' if counted == 1 else 'treatments'}.")
        return panel.render(stored, editing=key, told=told)

    def close_editor():
        return panel.render()

    def to_natural():
        stored = set_axis(key, mode=director.NATURAL)
        return panel.render(stored, told=f"{label} is Natural again and has left the "
                                         "active directions.")

    outputs = panel.outputs()
    edit.click(fn=open_editor, outputs=outputs, queue=False)
    # ``input`` and not ``change`` throughout, and it is not a detail: every one
    # of these handlers answers by rewriting the whole panel, including the very
    # control that fired it. ``change`` fires when the server sets a value, so
    # this would be a loop -- one that terminates only because the value it
    # writes back is the value it just read.
    panel.modes[key].input(fn=change_mode, inputs=[panel.modes[key]], outputs=outputs,
                           queue=False)
    panel.fixed[key].input(fn=change_fixed, inputs=[panel.fixed[key]], outputs=outputs,
                           queue=False)
    panel.excluded[key].input(fn=change_excluded, inputs=[panel.excluded[key]],
                              outputs=outputs, queue=False)
    done.click(fn=close_editor, outputs=outputs, queue=False)
    natural.click(fn=to_natural, outputs=outputs, queue=False)


def _wire_profiles(panel, save, save_as, create, drop, make_default, reset) -> None:
    """Load, save, name, delete, nominate, restore. One render each."""

    def choose(name):
        stored, complaint = profiles.apply(name)
        if complaint:
            return panel.render(stored, told=complaint, kind="warn",
                                profile=profiles.FACTORY)
        return panel.render(stored, told=f'Loaded the "{name}" Creative profile.',
                            profile=name)

    def save_over(name):
        try:
            profiles.save(name, profiles.from_settings())
        except profiles.ProfileError as exc:
            return panel.render(told=str(exc), kind="warn")
        profiles.remember_selection(name)
        return panel.render(told=f'Saved the current settings to "{name}".',
                            profile=name)

    def start_naming():
        return panel.render(told="Name the new profile, then press Create.",
                            naming=True)

    def create_new(name):
        try:
            profiles.save(name, profiles.from_settings())
        except profiles.ProfileError as exc:
            return panel.render(told=str(exc), kind="warn", naming=True)
        profiles.remember_selection(name.strip())
        return panel.render(told=f'Created the "{name.strip()}" Creative profile.',
                            profile=name.strip())

    def remove(name):
        try:
            profiles.delete(name)
        except profiles.ProfileError as exc:
            return panel.render(told=str(exc), kind="warn")
        # The settings on screen are left exactly as they are. Deleting a saved
        # copy of a configuration is not a request to stop using it, and a delete
        # that silently reconfigured the panel would be a destructive undo of
        # work nobody asked to undo.
        return panel.render(told=f'Deleted the "{name}" Creative profile. The settings '
                                 "on screen are unchanged.", profile=profiles.FACTORY)

    def nominate(name):
        try:
            profiles.set_default(name)
        except profiles.ProfileError as exc:
            return panel.render(told=str(exc), kind="warn")
        return panel.render(
            told=f'"{name}" is now the default: Reset to default restores it, and a '
                 "panel that has not loaded any other profile opens showing it.",
            profile=name)

    def restore_default():
        name, _values, complaint = profiles.default_profile()
        stored, _ = profiles.apply(name)
        told = complaint or f'Reset to the "{name}" Creative profile.'
        return panel.render(stored, told=told, kind="warn" if complaint else "info",
                            profile=name)

    outputs = panel.outputs()
    panel.profile.input(fn=choose, inputs=[panel.profile], outputs=outputs, queue=False)
    save.click(fn=save_over, inputs=[panel.profile], outputs=outputs, queue=False)
    save_as.click(fn=start_naming, outputs=outputs, queue=False)
    create.click(fn=create_new, inputs=[panel.profile_name], outputs=outputs, queue=False)
    drop.click(fn=remove, inputs=[panel.profile], outputs=outputs, queue=False)
    make_default.click(fn=nominate, inputs=[panel.profile], outputs=outputs, queue=False)
    reset.click(fn=restore_default, outputs=outputs, queue=False)


def _wire_settings(panel, forget) -> None:
    """The add-a-direction dropdown and the secondary settings."""
    from prompt_master.krea import director

    def add_direction(key):
        if not key:
            return panel.render()
        stored = set_axis(str(key), mode=director.VARY)
        label = panel.axes[str(key)].label if str(key) in panel.axes else str(key)
        return panel.render(stored, editing=str(key),
                            told=f"{label} now varies. Exclude anything you never want, "
                                 "or pin one treatment.")

    def remember_seed(value):
        mc_creative_krea.remember(**{mc_creative_krea.SEED: mc_creative_krea._seed(value)})
        return gr.update()

    def remember_anti(value):
        mc_creative_krea.remember(**{mc_creative_krea.ANTI_REPETITION: bool(value)})
        return gr.update()

    def forget_history():
        mc_creative_krea.forget_history()
        return panel.notice("Recent-roll memory cleared; every treatment is available "
                            "again.")

    def remember_loras(value):
        # The box is rewritten with the parsed tags rather than with what was
        # typed, which is the visible half of the rule that this field
        # contributes networks and never prompt text: prose typed here
        # disappears in front of the person who typed it, instead of quietly
        # reaching the image model.
        suffix = mc_creative_krea.lora_suffix(value)
        mc_creative_krea.remember(**{mc_creative_krea.LORAS: suffix})
        counted = len(mc_creative_krea.pinned_tags(suffix))
        return suffix, panel.notice(f"{counted} pinned LoRA{'' if counted == 1 else 's'}.")

    panel.add.input(fn=add_direction, inputs=[panel.add], outputs=panel.outputs(),
                    queue=False)
    panel.seed.input(fn=remember_seed, inputs=[panel.seed], outputs=[panel.seed],
                     queue=False)
    panel.anti.input(fn=remember_anti, inputs=[panel.anti], outputs=[panel.anti],
                     queue=False)
    forget.click(fn=forget_history, outputs=[panel.status], queue=False)
    if panel.loras is not None:
        panel.loras.blur(fn=remember_loras, inputs=[panel.loras],
                         outputs=[panel.loras, panel.status], queue=False)


# --------------------------------------------------------------------------- #
# Reading the controls back
# --------------------------------------------------------------------------- #


def axes_from(axis_values) -> tuple[dict, dict, dict]:
    """The axis controls as ``(modes, fixed ids, excluded ids)``.

    The values arrive as one flat tuple -- Gradio has no other shape for a
    variable number of inputs -- laid out mode, fixed, excluded, three per axis,
    in the library's own axis order. That grouping is the one thing that could go
    quietly wrong, so it is done in one place rather than at each of the callers
    that needs it.

    A tuple of the wrong length is refused outright rather than unpacked as far
    as it goes. A short list means the caller is an API request with no panel
    behind it, or a page built against a different library, and reading three
    controls per axis out of two would produce a *valid* configuration that
    nobody chose -- which the caller would then save over the one they did.
    """
    from prompt_master.krea import director

    if not axis_values:
        return {}, {}, {}

    try:
        from prompt_master.krea import library as library_module

        keys = library_module.library().axis_keys
    except Exception:
        return {}, {}, {}

    if len(axis_values) != len(keys) * 3:
        logger.debug("Model Chain: the Creative axis controls sent %s values for %s "
                     "axes; the saved settings answer instead",
                     len(axis_values), len(keys))
        return {}, {}, {}

    modes, fixed, excluded = {}, {}, {}
    for position, key in enumerate(keys):
        mode, pinned, dropped = axis_values[position * 3:position * 3 + 3]
        mode = str(mode or "").casefold()
        modes[key] = mode if mode in director.MODES else director.NATURAL
        if pinned:
            fixed[key] = str(pinned)
        chosen = [str(value) for value in (dropped or ()) if str(value)]
        if chosen:
            excluded[key] = chosen
    return modes, fixed, excluded


def remember_axes(axis_values) -> None:
    """Keep what the axis controls currently hold, if they hold anything."""
    modes, fixed, excluded = axes_from(axis_values)
    if not modes:
        return
    mc_creative_krea.remember(**{mc_creative_krea.AXIS_MODES: modes,
                                 mc_creative_krea.FIXED_VALUES: fixed,
                                 mc_creative_krea.EXCLUDED_VALUES: excluded})
