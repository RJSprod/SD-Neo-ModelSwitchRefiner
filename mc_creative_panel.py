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
import mc_pipeline_panel
import mc_profile_state

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


def selection(axis, setting) -> list[str]:
    """The treatments an axis is willing to use, as the picker shows them.

    The other half of :func:`apply_treatments`, and the reason the two are
    written next to each other: they are one mapping read in two directions,
    and a mapping whose halves disagree is a panel that forgets what somebody
    chose the moment they reload the tab.

    Vary with nothing excluded is *every* treatment selected, which is exactly
    what it has always meant -- "the director may choose anything" -- now said
    in the vocabulary of the picker rather than in the vocabulary of exclusion.
    """
    from prompt_master.krea import director

    mode = setting.get("mode", director.NATURAL)
    if mode == director.FIXED:
        pinned = setting.get("fixed")
        return [str(pinned)] if pinned else []
    if mode == director.VARY:
        excluded = {str(value) for value in (setting.get("excluded") or ())}
        return [variant.identifier for variant in axis.variants
                if variant.identifier not in excluded]
    return []


def apply_treatments(key, axis, chosen) -> dict:
    """One picker's selection, written back as a mode, a pin and an exclusion list.

    The whole of section 5.3 in nine lines. The user answers one question --
    *which treatments am I willing to use?* -- and the three things the Director
    has always read are derived from the answer:

    ===========  =========================================================
    0 selected   Natural. The direction has a row and no effect; the axis is
                 left out of the brief entirely, exactly as an axis nobody
                 added would be.
    1 selected   Fixed to it. Repeated every roll.
    2+ selected  Vary, with everything *not* selected excluded. The Creative
                 seed chooses from the pool, which is what it has always done.
    ===========  =========================================================

    Nothing downstream changes. Stable treatment ids, creativity eligibility,
    compatibility, anti-repetition, user-prompt precedence and the written
    expression tiers all read the same three keys they read before, and cannot
    tell that the control above them was replaced.
    """
    from prompt_master.krea import director

    known = [variant.identifier for variant in axis.variants]
    wanted = [str(value) for value in (chosen or ()) if str(value) in known]
    # Deduplicated in the library's own order, so two selections that differ
    # only in the order they were clicked produce the same settings and the
    # same brief.
    picked = [identifier for identifier in known if identifier in set(wanted)]

    if not picked:
        return set_axis(key, mode=director.NATURAL, fixed=None, excluded=None)
    if len(picked) == 1:
        return set_axis(key, mode=director.FIXED, fixed=picked[0], excluded=None)
    return set_axis(key, mode=director.VARY, fixed=None,
                    excluded=[identifier for identifier in known
                              if identifier not in set(picked)])


def summarise(axis, setting) -> str:
    """One axis's whole configuration as one short line.

    ``Medium · Fashion editorial``. ``Lighting · 4 treatments, chosen by the
    Creative seed``. Labels rather than ids, because this is the line somebody
    reads to check they configured what they meant; the ids are in the
    diagnostics view and in the PNG metadata, where they are what matters.

    A row with nothing chosen says so plainly. It is the one state that looks
    like a mistake and is not -- it is a direction somebody has started and not
    finished, and the panel's job is to make the difference between "started"
    and "doing nothing" impossible to miss.
    """
    label = getattr(axis, "label", "")
    chosen = selection(axis, setting)

    if not chosen:
        return f"**{label}** · *no treatments chosen — not directed*"
    if len(chosen) == 1:
        variant = axis.variant(chosen[0])
        named = variant.label if variant is not None else chosen[0]
        return f"**{label}** · {named}"
    return (f"**{label}** · {len(chosen)} treatments, chosen by the Creative "
            "seed")


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

        import mc_llm_progress

        per_character = mc_progress.rate_for(mc_llm_progress.writer_rates("krea:read"))
    except Exception:
        per_character = 0.0028
    return characters, characters * float(per_character or 0.0028)


def describe_creativity(value, stored=None) -> str:
    """What moving the Creativity slider will actually do, given the directions.

    Reported as "the creativity slider is not working": at Creativity 10 with
    every axis Natural the panel said *extreme direction on every eligible axis*
    and produced a brief of zero characters. The slider was not broken. It was
    describing a scale it had nothing to apply -- Creativity governs how a
    *direction* is expressed, and an axis nobody has directed has no expression
    to scale.

    Both of the ways that happens are named here rather than left to be
    inferred, because they look identical from outside: no directions at all,
    and directions that exist but sit below the position where the Director
    starts emitting any (0 and 1 add nothing, by design and by promise).
    """
    from prompt_master.krea import director, variation

    creativity = variation.clamp(value)
    stored = stored or mc_creative_krea.settings()
    modes = stored.get("axis_modes") or {}
    active = [key for key, mode in modes.items()
              if mode in (director.VARY, director.FIXED)]

    if not active:
        return (f"Creativity {creativity} has nothing to scale: every axis is Natural, "
                "so the prompt is expanded with no art direction at all. Add a "
                "direction to give the slider something to act on.")

    counted = f"{len(active)} direction{'' if len(active) == 1 else 's'}"
    varying = [key for key in active if modes.get(key) == director.VARY]
    if creativity <= variation.LEGACY and varying:
        return (f"Creativity {creativity} adds no direction by design, so your "
                f"{counted} will not be expressed. Fixed values still apply; varying "
                "axes start at 2.")
    return f"{variation.describe(creativity)} · {counted}"


def active_note(stored=None) -> str:
    """One clause naming what Creative Mode is currently directing.

    For the status line the toggle writes, which is the only Creative text on
    screen while the drawer is shut. "Creative Mode is on" is true and, on a
    fresh configuration, deeply misleading on its own: on and directing nothing
    looks exactly like on and directing everything until the image arrives.
    """
    from prompt_master.krea import director

    stored = stored or mc_creative_krea.settings()
    modes = stored.get("axis_modes") or {}
    active = [key for key, mode in modes.items()
              if mode in (director.VARY, director.FIXED)]
    if not active:
        return ("No directions are set, so nothing is being art-directed — open "
                "Creative Controls to add one.")
    counted = f"{len(active)} direction{'' if len(active) == 1 else 's'}"
    return f"{counted} set."


def profile_state(name, stored=None) -> str:
    """``Loaded: Editorial · Modified · not saved``, or nothing.

    Section 8, in the vocabulary :mod:`mc_profile_state` settles for all three
    kinds of named configuration. Computed rather than tracked: every handler
    on this panel ends in a full render, so "does the screen still match the
    profile it came from" can be answered by comparing the two, and a dirty
    flag that is derived cannot drift out of step with the thing it describes.

    The sentence beside it is not decoration. "Not saved" is one word away from
    "not applied", and the two readings are opposite in consequence -- these
    settings are the ones the next Generate uses, saved or not.
    """
    name = str(name or "").strip()
    if not name:
        return ""
    saved = profiles.get(name)
    if saved is None:
        return ""

    stored = stored or mc_creative_krea.settings()
    modified = mc_profile_state.changed(profiles.from_settings(stored),
                                        mc_profile_state.snapshot(saved))
    line = mc_profile_state.describe(name, modified)
    explained = mc_profile_state.explain(modified)
    return f"{line}  \n{explained}" if explained else line


def describe_cost(stored=None) -> str:
    """The line under the directions: what they cost, in this machine's seconds.

    Said where the decision is made rather than only on the progress bar, which
    is the first place anybody looks and the last place they can act on it. A
    user who wants the image to start sooner has three levers -- fewer
    directions, a lower Creativity, and where the language model runs -- and two
    of the three are on this panel.
    """
    characters, seconds = brief_cost(stored)
    writing = _writing_seconds()
    where = _placement_note()
    after = (f", then about {writing:.0f}s of writing" if writing else "")

    if not characters:
        return (f"*No directions: nothing extra for the model to read{after}"
                f"{where}. A press is the plain Krea expansion of what you typed.*")

    return (f"*About {characters:,} characters of brief — roughly {seconds:.0f}s of "
            f"reading{after}{where}. The brief is different every roll, so it is the "
            "one part of the request that can never come out of the model's cache.*")


def _writing_seconds() -> float:
    """How long the expansion itself takes on this machine, or 0 if unknown.

    The other half of a press, and on a model running from system RAM it is
    usually the larger one: a hundred tokens at five tokens a second is twenty
    seconds, and no configuration on this panel changes it. It is quoted so that
    a user reading "4s of reading" is not left to conclude that the other twenty
    seconds are unaccounted for.

    Both numbers come out of the progress store, which learns them from this
    installation's own rolls: seconds per character of reply, and how long a
    reply usually is.
    """
    try:
        import mc_progress

        import mc_llm_progress

        per_character = float(mc_progress.rate_for(mc_llm_progress.writer_rates("krea:write")))
        length = float(mc_progress.measured("krea:reply", 0.0))
    except Exception:
        return 0.0
    return per_character * length if per_character > 0 and length > 0 else 0.0


def _placement_note() -> str:
    """Where the writer runs and how fast it was last measured going, or "".

    The two facts a user needs to act on and the two the panel was leaving them
    to infer from a log. Placement, because it is the largest lever by a wide
    margin -- the same machine reads at 36 tokens a second with the weights in
    RAM and 900 with them on the card. And the measured rate, because it is the
    one number that says whether *this* backbone is a good one to be running in
    that placement: on the machine this was written for, in system RAM, a dense
    12B wrote at 4.9 tokens a second where a 26B mixture-of-experts wrote at
    12.8.

    Neither is something this panel can change, which is exactly why it names
    them: the levers it does have are worth a few seconds each, and this one is
    worth twenty.
    """
    try:
        import mc_llm_runtime

        configuration = mc_llm_runtime.config()
        if not configuration.configured:
            return ""
        where = (" on the card" if configuration.on_gpu
                 else " with the model in system RAM")
        _prompt, reply = mc_llm_runtime.measured_speed()
    except Exception:
        return ""
    return f"{where}, measured at {reply:.1f} tokens/s" if reply > 0 else where


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


def add_direction(key) -> dict:
    """Give an axis a row, with no treatments chosen yet.

    Deliberately not "make it vary". Adding a direction is opening a question,
    not answering it: the row appears, the picker is empty, and until something
    is chosen the axis is Natural and the brief is exactly what it would have
    been. Section 5.1 -- Natural is represented by omission, and a row nobody
    has filled in omits just as thoroughly as a row that is not there.
    """
    stored = mc_creative_krea.settings()
    directions = [str(name) for name in (stored.get("directions") or ())]
    if str(key) not in directions:
        directions.append(str(key))
    mc_creative_krea.remember(**{mc_creative_krea.DIRECTIONS: directions})
    return mc_creative_krea.settings()


def remove_direction(key) -> dict:
    """Take an axis's row away, and its treatments with it.

    One action, because on screen they are one thing. A row removed but still
    pinned to a treatment would keep directing the brief from somewhere the
    user can no longer see, which is the exact failure the visible-row rule
    exists to prevent.
    """
    stored = set_axis(str(key), mode=_natural(), fixed=None, excluded=None)
    directions = [str(name) for name in (stored.get("directions") or ())
                  if str(name) != str(key)]
    mc_creative_krea.remember(**{mc_creative_krea.DIRECTIONS: directions})
    return mc_creative_krea.settings()


def _natural() -> str:
    from prompt_master.krea import director

    return director.NATURAL


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

    def __init__(self, ident, notice, status, keys, axes):
        self.ident = ident
        self.notice = notice
        self.status = status
        self.keys = list(keys)
        self.axes = dict(axes)

        # Whether Delete is armed. A gr.State and not a module variable: an
        # arm is one person's half-finished gesture in one browser, and a flag
        # on this process would be shared by every tab open on the server.
        self.arm_delete = None

        # Filled in by build(), in the order the page is assembled.
        self.creativity = None
        self.profile = None
        self.profile_state = None
        self.create = None
        self.directions = None
        self.profile_name = None
        self.summary = None
        self.cost = None
        self.add = None
        self.seed = None
        self.anti = None
        self.rows: dict = {}
        self.labels: dict = {}
        self.treatments: dict = {}
        self.removes: dict = {}
        # The three machine-facing controls per axis. Never visible any more:
        # the treatment picker is the only thing a user touches, and these are
        # what it writes. They stay components rather than becoming plain
        # values because they are the list that travels to the generation --
        # see `axis_controls`, whose shape and order are unchanged.
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
        """The secondary settings, in the order a surface passes them on.

        Two, and it used to be three. The Pinned LoRAs box is gone: a text box
        beside a prompt box that accepted exactly one kind of syntax was a
        second, narrower prompt input, and ``[[<lora:name:weight>]]`` in the
        prompt itself does the same job for every kind. See
        :mod:`prompt_master.krea.literals`.
        """
        return [self.seed, self.anti]

    def outputs(self) -> list:
        """Every component :meth:`render` returns an update for, in order."""
        found = [self.status, self.creativity, self.profile, self.profile_state,
                 self.create, self.directions, self.profile_name, self.summary,
                 self.cost, self.add, self.seed, self.anti]
        for key in self.keys:
            found.extend([self.rows[key], self.labels[key], self.treatments[key],
                          self.modes[key], self.fixed[key], self.excluded[key]])
        return found

    def components(self) -> dict:
        """Everything this panel owns, for a surface to expose and a test to read."""
        found = {"profile": self.profile, "profile_name": self.profile_name,
                 "profile_state": self.profile_state,
                 "create": self.create, "directions": self.directions,
                 "creativity": self.creativity, "arm_delete": self.arm_delete,
                 "summary": self.summary, "cost": self.cost,
                 "add": self.add,
                 "seed": self.seed, "anti": self.anti,
                 "axes": self.axis_controls,
                 "rows": [self.rows[key] for key in self.keys],
                 "labels": [self.labels[key] for key in self.keys],
                 "treatments": [self.treatments[key] for key in self.keys],
                 "editors": [self.editors[key] for key in self.keys],
                 "buttons": list(self.buttons)}
        return found

    def render(self, stored=None, told=None, kind="info",
               profile=None, naming=False) -> list:
        """An update for every component this panel owns.

        ``told`` is a sentence for the status line, or ``None`` to leave
        whatever is there. ``naming`` opens the Save As name box.

        There is no longer an ``editing`` argument, because there is no longer
        an editor to open. An axis is configured by the picker on its own row,
        so the panel has one fewer state to be in and one fewer way for that
        state to be wrong.
        """
        stored = stored or mc_creative_krea.settings()
        active = [key for key in self.keys
                  if key in set(stored.get("directions") or ())]
        natural = [key for key in self.keys if key not in active]

        updates = [
            gr.update(value=self.notice(told, kind)) if told is not None else gr.update(),
            gr.update(value=stored.get("creativity")),
            gr.update(choices=profiles.choices(),
                      value=profile) if profile is not None
            else gr.update(choices=profiles.choices()),
            gr.update(value=profile_state(profile if profile is not None
                                          else profiles.selected(), stored)),
            # Save As opens the drawer the name box lives in, and nothing ever
            # closes it: a render that folded it away would do so while
            # somebody was typing into it, on any handler that happened to fire.
            gr.update(open=True) if naming else gr.update(),
            gr.update(label=directions_label(active)),
            gr.update(value="") if not naming else gr.update(),
            gr.update(value=DIRECTIONS_HEADING if active else NO_DIRECTIONS),
            gr.update(value=describe_cost(stored)),
            gr.update(choices=[(self.axes[key].label, key) for key in natural],
                      value=None, visible=bool(natural)),
            gr.update(value=stored.get("seed")),
            gr.update(value=bool(stored.get("anti_repetition"))),
        ]

        for key in self.keys:
            setting = _axis_setting(stored, key)
            updates.extend([
                gr.update(visible=key in active),
                gr.update(value=summarise(self.axes[key], setting)),
                gr.update(value=selection(self.axes[key], setting)),
                # The three that travel to the generation. Rewritten from the
                # settings on every render, never touched by hand, and never
                # shown -- which is what keeps the picker and the brief from
                # being two opinions about the same axis.
                gr.update(value=setting["mode"]),
                gr.update(value=setting["fixed"]),
                gr.update(value=setting["excluded"]),
            ])
        return updates


# --------------------------------------------------------------------------- #
# Building it
# --------------------------------------------------------------------------- #


def directions_label(active) -> str:
    """The Directions drawer's own title, with how many are set.

    §3 of the pipeline intent: a disclosure summary says what is behind it, so
    the count belongs in the label rather than on a line inside the drawer that
    only opening the drawer reveals. Derived from the settings on every render
    like every other summary in this file -- never written from what a control
    was just set to.
    """
    count = len(list(active or ()))
    if not count:
        return "Directions"
    return f"Directions — {count} active"


def build(ident, notice, status, *, creativity=None, stored=None) -> Panel | None:
    """Assemble the panel into whatever container is open. ``None`` if it cannot.

    A creativity library that will not load leaves a sentence on the page saying
    so and returns nothing: Creative Mode has no vocabulary to direct with, and a
    panel of empty dropdowns would invite somebody to configure a feature that
    cannot run.

    The shape it builds
    -------------------
    Profile and Creativity at the top level, then four drawers at one level:

        Profile
        Creativity
        Create a profile      ]
        Directions            ]  same left edge, same treatment,
        Advanced settings     ]  nesting only inside their contents
        Recovery & diagnostics]  (built by the owning surface)

    Directions used to be the body of this panel with everything else arranged
    around it: twenty axis rows, a heading, a cost line and an Add dropdown, all
    unfolded the moment Creative Mode was expanded, with Settings tucked in an
    accordion underneath. What that produced was a stage whose first screen was
    a list of axes nobody had asked about yet. The four drawers are peers now
    and all of them start closed, so expanding Creative shows the two settings
    somebody came for and four labelled ways in.

    The slider is built here rather than handed in, because where it goes is
    part of the shape: §3 puts Creativity beside Profile at the top level, and a
    caller that made it first would put it above them both.

    ``creativity`` is for the surface where that is not true. LLM Studio's Krea
    tab draws the slider outside this panel entirely -- it shows and hides with
    Creative Mode there, and the panel is inside a drawer that does the same --
    so it passes the one it already made and this builds none.
    """
    from prompt_master.krea import director
    from prompt_master.krea import variation

    try:
        from prompt_master.krea import library as library_module

        lib = library_module.library()
    except Exception as exc:
        gr.Markdown(f"The creativity library could not be read, so Creative Mode has "
                    f"no vocabulary to direct with: {exc}")
        return None

    stored = stored or mc_creative_krea.settings()
    keys = list(lib.axis_keys)
    panel = Panel(ident, notice, status, keys, {key: lib.axis(key) for key in keys})

    def button(label, *parts, **kwargs):
        made = gr.Button(label, size="sm", elem_id=ident(*parts), **kwargs)
        panel.buttons.append(made)
        return made

    active_now = list(stored.get("directions") or ())
    panel.arm_delete = gr.State(False)

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
        panel.profile_state = gr.Markdown(
            profile_state(current, stored), elem_id=ident("profile", "state"),
            elem_classes=classes("profile-state"))
        with gr.Row(elem_classes=classes("profile-actions")):
            save = button("Save", "profile", "save")
            save_as = button("Save As", "profile", "save-as")
            drop = button("Delete", "profile", "delete", variant="stop")
            make_default = button("Set as default", "profile", "default")
            reset = button("Reset to default", "profile", "reset")

    # -- creativity, beside the profile it belongs to ---------------------- #
    panel.creativity = creativity if creativity is not None else gr.Slider(
        label=variation.LABEL, minimum=variation.MINIMUM, maximum=variation.MAXIMUM,
        step=1, value=stored["creativity"], info=variation.HELP,
        elem_id=ident("creativity"), elem_classes=classes("creativity"))

    # -- drawer one: create a profile -------------------------------------- #
    # The Save As name box, in a drawer of its own rather than a row that
    # appears out of nowhere when a button is pressed. Save As opens it; it is
    # also just *there*, which is how somebody finds it without pressing Save As
    # first to discover what Save As does.
    with mc_pipeline_panel.drawer(
            "Create a profile", elem_id=ident("profile", "create-drawer"),
            elem_classes=classes("drawer", "profile-create")) as create:
        panel.create = create
        with gr.Row(elem_classes=classes("profile-name")):
            panel.profile_name = gr.Textbox(
                label="New profile name", value="", scale=3, max_lines=1,
                placeholder="Editorial portraits", elem_id=ident("profile", "name"))
            make = button("Create", "profile", "create", variant="primary")

    # -- drawer two: the active directions --------------------------------- #
    axis_rows: list = []

    with mc_pipeline_panel.drawer(
            directions_label(active_now), elem_id=ident("directions"),
            elem_classes=classes("drawer", "directions")) as directions:
        panel.directions = directions
        panel.summary = gr.Markdown(
            DIRECTIONS_HEADING if active_now else NO_DIRECTIONS,
            elem_id=ident("summary"), elem_classes=classes("summary"))

        for key in keys:
            axis = lib.axis(key)
            setting = _axis_setting(stored, key)
            active = key in active_now

            # One row, one question. The picker is a stock multiselect Dropdown
            # -- a compact closed field that opens a scrollable, filterable
            # popup and closes on an outside click -- which is section 5.4
            # exactly, and is the host's own component rather than an imitation
            # of it. A theme restyles it along with every other dropdown.
            with gr.Group(visible=active, elem_id=ident("row", key),
                          elem_classes=classes("direction")) as row:
                panel.rows[key] = row
                with gr.Row(elem_classes=classes("direction-bar")):
                    panel.treatments[key] = gr.Dropdown(
                        label=axis.label, multiselect=True, filterable=True,
                        value=selection(axis, setting), scale=5,
                        choices=[(variant.label, variant.identifier)
                                 for variant in axis.variants],
                        elem_id=ident("treatments", key),
                        elem_classes=classes("treatments"),
                        info="one treatment repeats it; two or more let the "
                             "Creative seed choose between them")
                    panel.removes[key] = button("Remove", "row", key, "remove")
                panel.labels[key] = gr.Markdown(
                    summarise(axis, setting), elem_id=ident("row", key, "summary"),
                    elem_classes=classes("direction-summary"))

            # The machine-facing triple, built and never shown. `visible=False`
            # in Gradio removes an element from the layout but keeps its value
            # in the payload, which is exactly what is wanted: these are what
            # travel to the generation, and `axis_controls` -- mode, fixed,
            # excluded, three per axis, in the library's own order -- is the
            # contract the hook, the profiles and `axes_from` all still read
            # unchanged.
            with gr.Group(visible=False, elem_id=ident("editor", key),
                          elem_classes=classes("editor")) as editor:
                panel.editors[key] = editor
                panel.modes[key] = gr.Radio(
                    label="How this axis behaves",
                    choices=[("Natural", director.NATURAL), ("Vary", director.VARY),
                             ("Fixed", director.FIXED)],
                    value=setting["mode"], elem_id=ident("editor", key, "mode"))
                panel.fixed[key] = gr.Dropdown(
                    label="Always use", value=setting["fixed"],
                    choices=[(variant.label, variant.identifier)
                             for variant in axis.variants],
                    elem_id=ident("editor", key, "fixed"))
                panel.excluded[key] = gr.Dropdown(
                    label="Exclude choices", value=setting["excluded"],
                    multiselect=True,
                    choices=[(variant.label, variant.identifier)
                             for variant in axis.variants],
                    elem_id=ident("editor", key, "excluded"))

            # Wired at the end, not here: every handler answers with an update
            # for every component the panel owns, and half of them do not exist
            # yet.
            axis_rows.append((key, panel.treatments[key], panel.removes[key]))

        panel.cost = gr.Markdown(describe_cost(stored), elem_id=ident("cost"),
                                 elem_classes=classes("cost"))

        natural_now = [key for key in keys if key not in set(active_now)]
        panel.add = gr.Dropdown(
            label=ADD_LABEL, value=None, elem_id=ident("add"),
            choices=[(lib.axis(key).label, key) for key in natural_now],
            visible=bool(natural_now), elem_classes=classes("add"),
            filterable=False,
            info="choose an axis to give a direction; everything else stays Natural")

    # -- drawer three: the secondary settings ------------------------------ #
    with mc_pipeline_panel.drawer("Advanced settings", elem_id=ident("settings"),
                                  elem_classes=classes("drawer", "settings")):
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

    for key, picker, remove in axis_rows:
        _wire_axis(panel, key, picker, remove)
    _wire_profiles(panel, save, save_as, make, drop, make_default, reset)
    _wire_settings(panel, forget)

    if complaint:
        logger.warning("Model Chain: %s", complaint)
    return panel


# --------------------------------------------------------------------------- #
# The handlers
# --------------------------------------------------------------------------- #


def _wire_axis(panel, key, picker, remove) -> None:
    """The two ways one axis changes, both ending in one full render.

    Two, and it used to be five. Choosing treatments and taking the row away
    are the only actions a direction has now -- the mode radio, the pinned
    value and the exclusion list were three controls asking three versions of
    the same question, and the picker asks it once.

    ``queue=False`` throughout: neither does work worth queueing, and neither
    starts, stops or waits for a generation. They read a settings file, write a
    settings file and redraw a drawer.
    """
    label = panel.axes[key].label
    axis = panel.axes[key]

    def choose(values):
        stored = apply_treatments(key, axis, values)
        chosen = selection(axis, _axis_setting(stored, key))
        if not chosen:
            told = (f"{label} has no treatments chosen, so it is left out of the "
                    "brief entirely — exactly as it would be with no row at all.")
        elif len(chosen) == 1:
            variant = axis.variant(chosen[0])
            named = variant.label if variant is not None else chosen[0]
            told = f"{label} is always {named}."
        else:
            told = (f"{label} may be any of {len(chosen)} treatments; the Creative "
                    "seed picks one each roll.")
        return panel.render(stored, told=told)

    def drop_row():
        stored = remove_direction(key)
        return panel.render(stored, told=f"{label} is Natural again and has left the "
                                         "active directions.")

    outputs = panel.outputs()
    # ``input`` and not ``change``, and it is not a detail: this handler answers
    # by rewriting the whole panel, including the very control that fired it.
    # ``change`` fires when the server sets a value, so this would be a loop --
    # one that terminates only because the value it writes back is the value it
    # just read.
    picker.input(fn=choose, inputs=[picker], outputs=outputs, queue=False)
    remove.click(fn=drop_row, outputs=outputs, queue=False)


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

    def remove(name, armed):
        """Two presses, because deleting a profile removes a file.

        The first press arms the button and says which profile is about to go;
        the second does it. §3 of the pipeline intent asks for an explicit
        confirmation where the loss is irreversible, and this one is.
        """
        go, now, button = mc_pipeline_panel.confirmed(armed)
        if not go:
            return (now, button,
                    *panel.render(told=f'Press Delete again to remove the "{name}" '
                                       "Creative profile. This cannot be undone.",
                                  kind="warn"))
        try:
            profiles.delete(name)
        except profiles.ProfileError as exc:
            return (now, button, *panel.render(told=str(exc), kind="warn"))
        # The settings on screen are left exactly as they are. Deleting a saved
        # copy of a configuration is not a request to stop using it, and a delete
        # that silently reconfigured the panel would be a destructive undo of
        # work nobody asked to undo.
        return (now, button,
                *panel.render(told=f'Deleted the "{name}" Creative profile. The '
                                   "settings on screen are unchanged.",
                              profile=profiles.FACTORY))

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
    drop.click(fn=remove, inputs=[panel.profile, panel.arm_delete],
               outputs=[panel.arm_delete, drop, *outputs], queue=False)
    make_default.click(fn=nominate, inputs=[panel.profile], outputs=outputs, queue=False)
    reset.click(fn=restore_default, outputs=outputs, queue=False)


def _wire_settings(panel, forget) -> None:
    """The add-a-direction dropdown and the secondary settings."""
    from prompt_master.krea import director

    def add(key):
        if not key:
            return panel.render()
        stored = add_direction(str(key))
        label = panel.axes[str(key)].label if str(key) in panel.axes else str(key)
        return panel.render(stored,
                            told=f"{label} has a row. Choose the treatments you are "
                                 "willing to use — one repeats it, several let the "
                                 "Creative seed choose.")

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

    panel.add.input(fn=add, inputs=[panel.add], outputs=panel.outputs(),
                    queue=False)
    panel.seed.input(fn=remember_seed, inputs=[panel.seed], outputs=[panel.seed],
                     queue=False)
    panel.anti.input(fn=remember_anti, inputs=[panel.anti], outputs=[panel.anti],
                     queue=False)
    forget.click(fn=forget_history, outputs=[panel.status], queue=False)


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
