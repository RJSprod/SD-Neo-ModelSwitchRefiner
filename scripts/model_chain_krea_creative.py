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


def _settings_for(values) -> dict:
    """This generation's Creative settings, from the panel when it sent them.

    ``values`` is what Forge handed ``before_process`` after the enabled flag:
    the four scalars and then the axis controls, in the order :meth:`ui` returned
    them. A UI that could not build its axis controls sends fewer, and an API
    request sends none at all, so the length is checked rather than assumed and
    the saved preferences answer for anything absent.
    """
    values = tuple(values or ())
    if len(values) < 4:
        return mc_creative_krea.settings()
    creativity, seed, anti_repetition, loras = values[:4]
    return _stored(creativity, seed, anti_repetition, loras, values[4:])


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
    """Show or hide the controls, and remember the toggle."""
    mc_creative_krea.remember(**{mc_creative_krea.ENABLED: bool(enabled)})
    shown = gr.update(visible=bool(enabled))
    if enabled:
        objection = mc_creative_krea.checkpoint_objection()
        told = notice(objection or
                      "Creative Mode is on. Press Generate: the prompt is directed "
                      "locally and expanded once, then Forge makes the image.",
                      "warn" if objection else "info")
    else:
        told = notice("Creative Mode is off.")
    return shown, shown, gr.update(value=told, visible=bool(enabled))


def _remember_creativity(value):
    from prompt_master.krea.variation import clamp, describe

    mc_creative_krea.remember(**{mc_creative_krea.CREATIVITY: clamp(value)})
    return notice(describe(value))


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
                gr.update())

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

    kind = "warn" if setup.warnings() or (replay_exactly and not setup.replayable) \
        else "info"
    return (gr.update(value=setup.source) if setup.source else gr.update(),
            gr.update(value=True),
            notice(" ".join(said), kind), gr.update(value=_pasted_view()))


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
        # Filled in by ui(); the shell never reads it, and a test asking what
        # the panel is made of has something to ask.
        self.components: dict = {}
        self.panel: mc_creative_panel.Panel | None = None
        # The native prompt box, handed over by after_component. Only the
        # restore action writes to it. See PROMPT_ELEM_ID.
        self.prompt_box = None
        # True while this hook is inside its own roll. A nested process_images()
        # -- Stage 2's, or any extension's -- must not start a second one, and
        # the cost of getting that wrong is not a duplicate image but an LLM
        # request that begins while the first is still streaming.
        self._rolling = False

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
        from prompt_master.krea import variation

        stored = mc_creative_krea.settings()

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
            "pasted": pasted, "replay": exactly, "restore": restore, "disarm": disarm}
        if panel is not None:
            self.components.update(panel.components())

        self._wire(enabled, creativity, status, controls, show, recipe, expanded,
                   pasted, exactly, restore, disarm)
        self._register_paste_fields()

        # Every control travels to before_process, because that is where the
        # roll happens and the panel is what the user is looking at. They are
        # this script's own arguments and reach neither Model Chain's preset
        # list nor its infotext.
        if panel is None:
            return [enabled, creativity]
        return ([enabled, creativity] + list(panel.settings_controls)
                + list(panel.axis_controls))

    def _wire(self, enabled, creativity, status, controls, show, recipe, expanded,
              pasted, exactly, restore, disarm):
        """Every handler this file owns, in one place. The panel wires its own.

        All of them ``queue=False``: not one of these does any work worth
        queueing, and none of them starts, stops or waits for a generation. The
        panel is settings and nothing else now -- the roll is in
        :meth:`before_process`, which is not reachable from here.
        """
        enabled.change(fn=_toggled, inputs=[enabled],
                       outputs=[creativity, controls, status], queue=False)
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
        if self._rolling:
            # A process_images() nested inside our own roll. There is nothing to
            # do for it and a great deal to get wrong.
            logger.debug("Model Chain: Creative Mode is already rolling; the nested "
                         "generation is left alone")
            return

        written = self._roll(p, args)
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

    def _roll(self, p, values):
        """One creative roll for this generation, or ``None`` to leave it alone.

        The whole of the deadlock fix is the two lines that matter here: the roll
        runs inside :class:`mc_broker.host_job`, which is how
        ``mc_llm_sessions._Gpu.acquire`` is told that the image job is blocked
        waiting for this request rather than competing with it, and the bar is
        borrowed rather than claimed, because the host already started one for
        the generation this is the first part of.

        The events are drained rather than forwarded. There is no open Gradio
        event to forward them to -- the press became a native generation, not a
        handler with an output list -- so the progress bar carries the phase and
        the log carries the rest.
        """
        import mc_broker

        source = str(getattr(p, "prompt", "") or "").strip()
        if not source:
            logger.info("Model Chain: Creative Mode has no source prompt to work from")
            return None

        session = mc_creative_krea.creative
        settings = _settings_for(values)
        loras = settings.get("loras", "")

        # Held by name and closed explicitly, the way the roll itself holds the
        # LLM run: every path out of the loop below leaves the generator
        # suspended, and it is its ``finally`` that gives the progress bar and
        # the workload lock back. Closing it is what runs that now rather than
        # whenever the interpreter next collects the frame.
        events = session.roll(source, settings, guard_checkpoint=True, own_bar=False)
        written = False
        self._rolling = True
        try:
            with mc_broker.host_job():
                for event in events:
                    if event.kind == sessions.STATUS:
                        logger.debug("Model Chain: Creative Mode — %s", event.text)
                    elif event.kind == sessions.CANCELLED:
                        logger.info("Model Chain: the Creative Mode roll was stopped; "
                                    "the generation continues with the typed prompt")
                        break
                    elif event.kind == sessions.FAILED:
                        logger.warning("Model Chain: the Creative Mode roll failed (%s); "
                                       "generating from the typed prompt instead",
                                       event.text)
                        break
                    elif event.kind == sessions.DONE:
                        written = True
                        break
        except Exception:
            errors.report("Model Chain: the Creative Mode roll failed", exc_info=True)
            return None
        finally:
            events.close()
            self._rolling = False

        if not written:
            return None

        last = session.last
        if last is None or not last.expanded.strip():
            logger.warning("Model Chain: the Creative Mode roll produced nothing; "
                           "generating from the typed prompt instead")
            return None
        return mc_creative_krea.prepare(last, loras, settings)
