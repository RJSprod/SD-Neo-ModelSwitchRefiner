"""LLM Studio: one native Forge tab, five distinct workspaces (section 4.1).

This module is the shell. It builds the tab, switches between the modes, and
owns the two things all of them share -- the residency status that makes memory
decisions visible (section 14), and the one control every mode needs, which is
*which model is running*.

Both of those used to be a top bar: a mode selector, a model chooser, a rescan,
Load, Unload and a status line, above every workspace, at all times. That is six
controls' worth of configuration a reader has to look past to reach the thing
they opened the tab for, and on a narrow display it wrapped to four rows and
pushed the workspace under it off the bottom of the window. So the bar is now a
menu, a title and a state chip, and everything it used to carry lives in two
sheets that open over the tab and close again -- the workspace chooser and the
model sheet. Nothing is gone; it is one tap away instead of permanently on
screen, which is the difference between chrome and content.

Where setup went, and why
-------------------------
The model, the placement and the context budget used to be a "Models, hardware
and memory" accordion sitting under whichever workspace was open. Two things
were wrong with that. Everything in it that is a plain value -- the context
sizing, the cache types, the residency policy, the folders -- describes the
installation rather than anything a mode is doing, which is precisely the test
this extension already applies to decide that a control belongs on the WebUI's
Settings page; and everything in it that is *not* a plain value -- the file
dialogs, the runtime download, the estimator, the residency table -- is a panel
in its own right and had no business being a footnote to a chat window.

So the plain values are registered as Forge settings (see
``scripts/model_chain.py``, ``mc_llm_state.HOSTED`` and
``mc_llm_paths.OPT_MODELS``) and the panel is a mode of its own, **Setup**,
reached from the same selector as the other workspaces. What is left in the model
sheet is a chooser filled from the models folder, and Load and Unload -- start
the thing with what was chosen last, or give the VRAM back -- because that is
the whole of what switching models day to day actually needs.

What this module deliberately does not do is host the modes' logic. Prompt
Studio, Conversation, MiniMax and Krea 2 are built by four separate modules and
share no state beyond the preferences file, which is section 4.1's requirement that
the modes "may reuse shared panels" but "must not be collapsed into a single
generic chat workflow" enforced at the level of the source tree.

Failure is a first-class state here. Section 18 requires that a failure to
start or load the LLM must not poison image generation, and the way that is
guaranteed is that nothing below runs at import time and every entry point is
wrapped: if the vendored package will not import, if Pillow is missing, if the
data directory is unwritable, the tab renders an explanation and the rest of
the WebUI never knows.
"""

from __future__ import annotations

import logging
from pathlib import Path

import gradio as gr

import mc_broker
import mc_llm_files
import mc_llm_paths
import mc_llm_ui as ui

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

_GB = 1024**3

TAB_LABEL = "LLM Studio"
TAB_ID = "llm_studio"

OPT_ENABLE = "model_chain_llm_studio"

MODES = (
    ("Prompt Studio", "prompt"),
    ("Conversation", "chat"),
    ("MiniMax H3", "minimax"),
    ("Krea 2", "krea"),
    ("Setup", "setup"),
)
"""The workspaces, as Gradio choices -- ``(label, value)``.

Setup is one of them rather than an accordion under the others: it is where a
model, a runtime and a placement are chosen, and none of that is a footnote to
whichever mode happened to be open when it was needed.
"""


def enabled() -> bool:
    """Whether the tab should be built at all.

    A setting rather than an assumption: an installation that only wants the
    image half of this extension should not carry an LLM tab it will never
    open, and section 18 asks for ordinary generation to be unaffected when
    LLM Studio is never used.
    """
    return bool(mc_broker.option(OPT_ENABLE, True))


def on_ui_tabs():
    """The ``on_ui_tabs`` callback. Never raises into the host."""
    if not enabled():
        return []
    try:
        return [(_build(), TAB_LABEL, TAB_ID)]
    except Exception:
        logger.warning("Model Chain: LLM Studio could not be built", exc_info=True)
        try:
            return [(_unavailable(), TAB_LABEL, TAB_ID)]
        except Exception:
            # Two failures means the tab cannot be drawn at all. Returning
            # nothing costs the feature; raising would cost the WebUI its UI.
            return []


def _unavailable():
    with gr.Blocks(analytics_enabled=False) as block:
        gr.Markdown(
            "### LLM Studio could not start\n\n"
            "The panel failed to build. The console holds the traceback.\n\n"
            "Image generation and Model Chain are unaffected."
        )
    return block


def _build():
    import mc_llm_chat_panel
    import mc_llm_krea_panel
    import mc_llm_minimax_panel
    import mc_llm_prompt_panel

    initial = _initial_mode()

    with gr.Blocks(analytics_enabled=False) as block:
        with gr.Column(elem_id=ui.ident("studio"), elem_classes=ui.classes("studio")):

            # Which sheet is open, by name. A Column has no value a handler can
            # be given, so the one control that has to *know* -- the menu, which
            # closes what it opened -- reads it from here.
            sheet_state = gr.State("")

            # -- the shell bar: a menu, a title and a state ----------------- #
            #
            # Three controls, one line, and none of them is a model filename.
            # This bar used to carry the mode selector, the chooser, a rescan,
            # Load, Unload and a status line, which is six controls' worth of
            # configuration above every workspace at all times -- and on a
            # phone it wrapped to four rows and pushed the conversation off the
            # bottom of the window. What replaced it is what application
            # chrome is for: a way in to everything, and nothing else.
            #
            # It is hidden while Conversation is open, because Conversation
            # draws the same three affordances into its own header, where they
            # sit beside the character and the thread rather than above them.
            with gr.Row(visible=(initial != "chat"),
                        elem_id=ui.ident("shellbar"),
                        elem_classes=ui.classes("shellbar")) as shellbar:
                menu = gr.Button("\u2630", size="sm", scale=0, min_width=44,
                                 elem_id=ui.ident("menu"),
                                 elem_classes=ui.classes("icon-button"))
                title = gr.HTML(_mode_title(initial), elem_id=ui.ident("mode", "title"),
                                elem_classes=ui.classes("shell-title"))
                chip = gr.Button(_chip_label(), size="sm", scale=0, min_width=0,
                                 elem_id=ui.ident("runtime"),
                                 elem_classes=ui.classes("chip-button"))

            # -- the sheets ------------------------------------------------- #
            #
            # Both are absolutely positioned inside this column by style.css,
            # so opening either costs the workspace under it nothing: no mode
            # is re-laid-out, no composer moves, and closing one puts back
            # exactly what was there. Only one is ever open, which is what
            # every handler below returns rather than toggles.

            with gr.Column(visible=False, elem_id=ui.ident("mode", "sheet"),
                           elem_classes=ui.classes("sheet", "sheet-shell")) as mode_sheet:
                with gr.Row(elem_classes=ui.classes("sheet-head")):
                    gr.Markdown("#### Workspace")
                    close_modes = gr.Button("\u2715", size="sm", scale=0, min_width=44,
                                            elem_classes=ui.classes("icon-button"))
                # The selector is still one Radio over one list of modes -- the
                # same control, in a sheet instead of a bar. Which mode is open
                # is a thing a radio says by construction, and four buttons
                # would have had to be told.
                mode = gr.Radio(
                    label=None, show_label=False, choices=list(MODES), value=initial,
                    container=False,
                    elem_id=ui.ident("mode"), elem_classes=ui.classes("modes"))
                model_from_modes = gr.Button("Model / Runtime", size="sm",
                                             elem_classes=ui.classes("nav-entry"))

            with gr.Column(visible=False, elem_id=ui.ident("model", "sheet"),
                           elem_classes=ui.classes("sheet", "sheet-shell")) as model_sheet:
                with gr.Row(elem_classes=ui.classes("sheet-head")):
                    gr.Markdown("#### Model and runtime")
                    close_model = gr.Button("\u2715", size="sm", scale=0, min_width=44,
                                            elem_classes=ui.classes("icon-button"))
                runtime_status = gr.HTML(_runtime_line(), elem_id=ui.ident("runtime", "status"),
                                         elem_classes=ui.classes("runtime-state"))
                chooser = gr.Dropdown(
                    label="Model", choices=_model_choices(),
                    value=_current_model(),
                    elem_id=ui.ident("model"),
                    elem_classes=ui.classes("model-choice"))
                with gr.Row():
                    rescan = gr.Button("\u21bb Rescan", size="sm",
                                       elem_id=ui.ident("model", "rescan"))
                    load = gr.Button("Load", variant="primary", size="sm",
                                     elem_id=ui.ident("load"))
                    unload = gr.Button("Unload", variant="stop", size="sm",
                                       elem_id=ui.ident("unload"))
                # The route out of the sheet and into the panel that has the
                # runtime, the residency and the estimator in full. A model
                # chooser can say which model; only Setup can say why it did
                # not fit.
                to_setup = gr.Button("Open Setup", size="sm",
                                     elem_classes=ui.classes("nav-entry"))

            # Keyed by mode rather than zipped against MODES: a mode added to
            # the selector without a panel behind it should fail here, where
            # on_ui_tabs turns it into the "could not start" tab, rather than
            # silently render an empty column.
            builders = {"prompt": mc_llm_prompt_panel.build,
                        "chat": mc_llm_chat_panel.build,
                        "minimax": mc_llm_minimax_panel.build,
                        "krea": mc_llm_krea_panel.build,
                        "setup": _setup_panel}
            views, settings, conversation = [], None, None
            for _, value in MODES:
                # visible= comes from the stored mode rather than being
                # hard-coded. It used to be hard-coded to Prompt Studio while
                # the selector was restored from preferences, so a tab left on
                # Conversation opened with the selector reading Conversation
                # and Prompt Studio's panel underneath it -- two controls
                # telling the truth about different things, which reads as the
                # mode selector not working.
                with gr.Column(visible=(value == initial),
                               elem_id=ui.ident("view", value),
                               elem_classes=ui.classes("mode-view")) as view:
                    built = builders[value]()
                    if value == "setup":
                        settings = built
                    if value == "chat":
                        conversation = built
                views.append(view)

        # -- wiring ------------------------------------------------------- #

        # The State first, then one visibility per sheet: the order
        # :func:`_sheet` answers in.
        sheets = [sheet_state, mode_sheet, model_sheet]
        # Both state controls say the same thing and are updated together: the
        # one in this bar, and the one Conversation draws in its own header.
        chips = [chip, conversation["chip"]]

        mode.change(fn=_switch, inputs=[mode],
                    outputs=views + [runtime_status, shellbar, title] + sheets + chips,
                    queue=False)

        # Toggles, not openers. A menu that can only open is a menu you cannot
        # dismiss from the button you opened it with, and on a desktop that
        # button is not even covered by what it opened -- so pressing it again
        # looked like a dead control.
        menu.click(fn=_toggle_sheet("mode"), inputs=[sheet_state], outputs=sheets,
                   queue=False)
        close_modes.click(fn=_sheet, outputs=sheets, queue=False)
        close_model.click(fn=_sheet, outputs=sheets, queue=False)
        # Conversation's own way in to the same two sheets. The panel has
        # already closed its own surfaces by the time this runs; all that is
        # left is to open the shell's.
        for control in conversation["model"] + [chip, model_from_modes]:
            control.click(fn=_toggle_sheet("model"), inputs=[sheet_state], outputs=sheets,
                          queue=False)
        conversation["modes"].click(fn=_toggle_sheet("mode"), inputs=[sheet_state],
                                    outputs=sheets, queue=False)
        # Setup is a workspace, so getting to it is a mode switch: the radio is
        # moved, which is what redraws the views and the bar through _switch.
        for control in (to_setup, conversation["setup"]):
            control.click(fn=lambda: (gr.update(value="setup"),) + tuple(_sheet("")),
                          outputs=[mode] + sheets, queue=False)

        rescanning = rescan.click(fn=_rescan_models, outputs=[chooser, runtime_status],
                                  queue=False)
        # ``input`` and not ``change``: this code refills the chooser on load
        # and on every rescan, and ``change`` fires on the refill -- which
        # would re-record the model, and stop the running server, every time
        # the tab was opened.
        chosen = _picked(chooser)(fn=_choose_model, inputs=[chooser],
                                  outputs=[runtime_status, settings["model"],
                                           settings["estimator"], settings["notice"]],
                                  queue=False)
        loading = load.click(
            fn=_load_model,
            outputs=[runtime_status, settings["residency"], settings["estimator"]])
        # Said before the load rather than after it: reading twenty gigabytes
        # off a disk is long enough that a control which looks inert gets
        # pressed twice, and the chip is the only thing on screen that can say
        # the press landed.
        load.click(fn=lambda: _chips(LOADING), outputs=chips, queue=False)
        unloading = unload.click(fn=_unload_model,
                                 outputs=[runtime_status, settings["residency"]], queue=False)

        # The state chips are refreshed after anything that can have changed
        # the runtime's state, rather than by every one of those handlers
        # returning two more values: what they each return is the *detail*,
        # and the chip is one word read back off the runtime itself.
        opened = block.load(fn=_on_load,
                            outputs=[runtime_status, settings["residency"], chooser],
                            queue=False)
        for dependency in (chosen, loading, unloading, opened, rescanning):
            dependency.then(fn=_chips, outputs=chips, queue=False)

    return block


def _sheet(name: str = "") -> list:
    """Which sheet is open, and one visibility per sheet.

    The name comes back first because it is what the ``sheet`` State holds, and
    that State is what makes ``\u2630`` a toggle rather than a control that can
    only ever open something -- a Column has no value a handler can be given,
    so the open sheet has to be remembered somewhere a handler can read.
    """
    wanted = name if name in SHEETS else ""
    return [wanted] + [gr.update(visible=(wanted == key)) for key in SHEETS]


def _toggle_sheet(name: str):
    """A control that opens ``name``, or closes it if it is already open."""
    def toggle(open_now):
        return _sheet("" if open_now == name else name)

    return toggle


SHEETS = ("mode", "model")
"""The shell's overlay sheets, in the order :func:`_sheet` answers in."""


def _mode_title(chosen: str) -> str:
    """Which workspace is open, for the shell bar.

    The selector's own label, looked up rather than restated: a title that has
    its own copy of the mode names is a title that can disagree with the
    control that set it.
    """
    named = dict((value, label) for label, value in MODES).get(chosen, "LLM Studio")
    return (f'<div class="{ui.PREFIX}-heading">'
            f'<span class="{ui.PREFIX}-heading-who">{ui.escape(named)}</span>'
            f'</div>')


def _chip_label(label: str | None = None) -> str:
    """The runtime's state as a button label: a mark and one word.

    The same words the status chip carries, and a mark rather than a colour in
    front of them: a button's label is text, and text is the one thing a theme
    cannot restyle into invisibility. What the mark cannot say -- loaded as
    what, on which device, at what context -- is a tap away in the model sheet,
    which is also the only place anything can be done about it.
    """
    try:
        return _chip_words(label)
    except Exception:
        logger.debug("Model Chain: could not read the runtime state", exc_info=True)
        return "\u25cc Unavailable"


def _chip_words(label: str | None = None) -> str:
    import mc_llm_runtime

    if label:
        return f"\u25cc {label}"
    state = mc_llm_runtime.runtime.status()
    if not state["configured"]:
        return "\u26a0 Not set up"
    return "\u25cf Loaded" if state["running"] else "\u25cb Unloaded"


def _chips(label: str | None = None) -> tuple:
    """The same label for every state chip on the tab."""
    return tuple(gr.update(value=_chip_label(label)) for _ in range(2))


def _picked(component):
    """The event that fires when a *user* picks, not when this code refills.

    The same helper ``mc_llm_browse`` needs and for the same reason, restated
    rather than imported: this module builds the tab and must not import the
    picker to do it.
    """
    return getattr(component, "input", None) or component.change


def _on_load():
    """What the tab shows the moment it is opened, rather than at build time."""
    return (_runtime_line(), _residency_html(),
            gr.update(choices=_model_choices(), value=_current_model()))


def _initial_mode() -> str:
    import mc_llm_state

    stored = mc_llm_state.preferences().get("mode", "prompt")
    return stored if stored in [value for _, value in MODES] else "prompt"


def _switch(chosen):
    """Show one workspace. The others are hidden, not rebuilt.

    Rebuilding would lose whatever was on screen -- a half-read reply, a prompt
    someone is editing -- every time the selector moved, which is the one thing
    a mode switch must not do.

    Everything after the views is the chrome the switch decides: the runtime
    line, whether the shell bar is drawn at all (Conversation carries its own),
    which workspace the title names, and the sheets, which close -- a mode
    chooser that stayed open over the mode it had just chosen would be asking
    the question again.
    """
    import mc_llm_state

    mc_llm_state.remember(mode=chosen)
    return ([gr.update(visible=(chosen == value)) for _, value in MODES]
            + [_runtime_line(), gr.update(visible=(chosen != "chat")), _mode_title(chosen)]
            + _sheet("") + list(_chips()))


# --------------------------------------------------------------------------- #
# The model chooser, Load and Unload
# --------------------------------------------------------------------------- #


def _library():
    """Every model under the models folder. Never raises into a panel."""
    try:
        return mc_llm_files.library(mc_llm_paths.models_root())
    except Exception:
        logger.debug("Model Chain: could not scan the models folder", exc_info=True)
        return mc_llm_files.Library(mc_llm_paths.data_root())


def _model_choices() -> list[tuple[str, str]]:
    """The chooser's choices: what is in the models folder, plus what is running.

    The second half matters more than it looks. A model may be recorded from
    anywhere on the machine -- that is the whole point of section 6b's path
    boxes -- so a chooser filled only from the scan would show *nothing
    selected* on an installation that is working perfectly, which reads as the
    model having been lost.
    """
    found = _library()
    root = found.folder
    choices: list[tuple[str, str]] = []
    seen: set[str] = set()

    def add(path, label=None):
        key = str(path)
        if key in seen:
            return
        seen.add(key)
        choices.append((label or _model_label(path, root), key))

    for path in found.models:
        add(path)

    current = _current_model()
    if current and str(current) not in seen:
        add(current, f"{Path(current).name} · {Path(current).parent}")
    return choices


def _model_label(path, root) -> str:
    """A model's name in the chooser: enough of its path to tell two apart."""
    try:
        relative = Path(path).relative_to(root)
    except (ValueError, TypeError):
        return Path(path).name
    return relative.as_posix()


def _current_model() -> str | None:
    import mc_llm_runtime

    try:
        model = mc_llm_runtime.config().model
    except Exception:
        logger.debug("Model Chain: could not read the configured model", exc_info=True)
        return None
    return str(model) if model else None


def _rescan_models():
    """Walk the models folder again, and say what came back.

    What it found goes to the console and to the chooser's own tooltip rather
    than to the state chip: "6 models under D:/models" is news for a moment and
    the chip has one job, which is to say whether the model is up.
    """
    found = _library()
    if not found.models:
        logger.info("Model Chain: LLM Studio — no .gguf files found under %s", found.folder)
        return (gr.update(choices=_model_choices(), value=_current_model()),
                ui.state("No models", "warn",
                         f"No .gguf files under {found.folder}. Set the Model Chain LLM "
                         f"models folder setting, or choose a model by path in Setup."))
    counted = f"{len(found.models)} model{'s' if len(found.models) != 1 else ''}"
    limited = (f" — the scan stopped at {mc_llm_files.MAX_LIBRARY_ENTRIES}"
               if found.truncated else "")
    logger.info("Model Chain: LLM Studio — %s found under %s%s", counted, found.folder,
                limited)
    return (gr.update(choices=_model_choices(), value=_current_model()), _runtime_line())


def _choose_model(path):
    """Record the chosen model. Loading it is a separate press.

    Separate because they cost differently: recording is a line in a file and
    switching back is free, while loading spends the time to read twenty
    gigabytes off a disk. A dropdown that started a load on every change would
    make scrolling through the list an expensive mistake.

    The projector is deliberately *not* carried over or inferred. A projector
    has to match the model it was made for and a file name does not prove that
    it does -- so one sitting beside the new model is mentioned and left for
    Setup to apply, which is the rule the rest of this extension already keeps.
    """
    import mc_llm_runtime
    from prompt_master.inference import model_choice

    unchanged = (gr.update(), gr.update(), gr.update())
    if not path:
        return (_runtime_line(),) + unchanged

    configuration = mc_llm_runtime.config()
    if configuration.model is not None and str(configuration.model) == str(path):
        return (_runtime_line(),) + unchanged
    if configuration.runtime is None:
        return (ui.state("Not set up", "warn",
                         "There is no llama.cpp runtime yet, so there is nothing to run a "
                         "model with. Set one up in Setup first."),
                gr.update(),
                gr.update(),
                ui.notice("There is no llama.cpp runtime yet, so there is nothing to run a "
                          "model with. Set one up in Setup first.", "warn"))

    try:
        chosen = mc_llm_files.resolve_model(path)
        model_choice.choose(mc_llm_paths.app_paths(), chosen.path, None)
    except mc_llm_files.PathError as exc:
        return (ui.state("Not set up", "warn", str(exc)), gr.update(), gr.update(),
                ui.notice(str(exc), "warn"))
    except Exception as exc:
        return (ui.state("Failed", "error", ui.failure(exc)), gr.update(), gr.update(),
                ui.notice(ui.failure(exc), "error"))

    # The running server holds the weights it was started with.
    mc_llm_runtime.runtime.stop()
    logger.info("Model Chain: LLM Studio — model set to %s", chosen.path.name)
    notes = list(chosen.notes) + _projector_hint(chosen.path)
    updated = mc_llm_runtime.config()
    return (_runtime_line(), str(chosen.path), _estimator_html(),
            _model_line(updated, notes))


def _load_model(progress=gr.Progress()):
    """Start llama-server on what is recorded, and report where it landed.

    A generator, so the chip can say Loading before the load and Loaded after
    it. Gradio streams a handler's yields, which is the only way a control can
    describe a state it is still in.

    The one press that makes the tab usable again after an Unload, and the one
    that answers "is it ready?" without sending a message to find out. It asks
    the runtime for a client and throws the client away: starting the server is
    the whole of the work, and every placement decision, every reduction and
    every note comes back on the report either way.
    """
    import mc_llm_runtime

    logger.info("Model Chain: LLM Studio — Load pressed")
    # Yielded before the work rather than after it: reading twenty gigabytes off
    # a disk is the one press here long enough that a button which looks like it
    # did nothing gets pressed again.
    yield _runtime_line(LOADING), gr.update(), gr.update()
    try:
        progress(0, desc="Starting llama-server…")
        mc_llm_runtime.runtime.client()
    except Exception as exc:
        logger.warning("Model Chain: llama-server could not be started: %s", ui.failure(exc))
        logger.debug("Model Chain: the llama-server start failed", exc_info=True)
        yield (ui.state("Failed", "error", ui.failure(exc)),
               _residency_html(), _estimator_html())
        return
    yield _runtime_line(), _residency_html(), _estimator_html()


def _unload_model():
    """Stop llama-server, releasing every byte of VRAM it held.

    And then stop anything else of ours still holding some. "Unload" is a
    request for the card back, and until now it could only reach the one server
    this WebUI has a handle to -- so a server that had outlived its parent went
    on holding fifteen gigabytes while this button correctly reported there was
    nothing to stop. See :func:`mc_llm_runtime.strays`. It is done here, from a
    press, rather than at startup, because two WebUIs sharing a card would each
    see the other's server as a stray.
    """
    import mc_llm_runtime

    logger.info("Model Chain: LLM Studio — Unload pressed")
    failed = ""
    if mc_llm_runtime.runtime.running():
        try:
            mc_llm_runtime.runtime.stop()
        except Exception:
            logger.warning("Model Chain: llama-server could not be stopped", exc_info=True)
            failed = "llama-server could not be stopped — see the console."

    stopped, freed = _release_strays()
    if failed:
        return ui.notice(failed, "error"), _residency_html()
    if stopped:
        return _stray_notice(stopped, freed), _residency_html()
    return _runtime_line(), _residency_html()


def _release_strays() -> tuple[int, int]:
    """Stop stray servers. A failure here never costs the Unload its answer."""
    import mc_llm_runtime

    try:
        return mc_llm_runtime.release_strays()
    except Exception:
        logger.warning("Model Chain: could not look for stray llama-server processes",
                       exc_info=True)
        return 0, 0


def _stray_notice(stopped: int, freed: int) -> str:
    """What Unload says when it found something the runtime had lost track of.

    Said out loud rather than done quietly: a process this WebUI did not know
    about was holding the card, and somebody who has just spent an afternoon
    wondering where their VRAM went is owed the sentence.
    """
    servers = "server" if stopped == 1 else "servers"
    amount = f", releasing {freed / _GB:.1f} GB" if freed else ""
    return ui.state("Unloaded", "idle",
                    f"Also stopped {stopped} stray llama-server {servers} left running by an "
                    f"earlier session{amount}.")


# --------------------------------------------------------------------------- #
# Status (section 14)
# --------------------------------------------------------------------------- #


LOADING = "Loading…"
"""What the chip says between pressing Load and the server answering."""


def _runtime_line(label: str | None = None) -> str:
    """The runtime's state, in one word.

    ``label`` overrides what it says without changing what it knows, which is
    how Load shows :data:`LOADING` while it is still loading -- the state a
    status can only report by being told, because nothing has changed yet.
    """
    try:
        return _runtime_state(label)
    except Exception:
        logger.warning("Model Chain: the LLM status line could not be drawn", exc_info=True)
        return ui.state("Unavailable", "error", "See the console for the traceback.")


def _runtime_state(label: str | None = None) -> str:
    import mc_llm_runtime

    state = mc_llm_runtime.runtime.status()
    detail = _runtime_detail(state)

    if label:
        # "busy" and not "info": the only thing that overrides the chip is a
        # state it was told about because nothing has happened yet, and every
        # one of them is a state something is happening *in*.
        return ui.state(label, "busy", detail)
    if not state["configured"]:
        # Which of the two is missing decides what to do about it, so the
        # tooltip says which -- the chip only has room to say that neither is
        # the reason the model is not running.
        missing = ("a llama.cpp runtime and a model" if not state["has_runtime"]
                   else "a model")
        return ui.state("Not set up", "warn",
                        f"LLM Studio needs {missing}. Open Setup.")
    return ui.state("Loaded" if state["running"] else "Unloaded",
                    "info" if state["running"] else "idle", detail)


def _runtime_detail(state=None) -> str:
    """Everything the chip used to say, for its tooltip and for Setup.

    Kept whole rather than trimmed: it is the answer to "loaded as what?", and
    the only reason it is not on screen is that it is not the question anybody
    asks of a status bar forty times an hour.
    """
    import mc_llm_runtime

    state = state if state is not None else mc_llm_runtime.runtime.status()
    if not state["configured"]:
        missing = ("a llama.cpp runtime and a model" if not state["has_runtime"]
                   else "a model")
        return f"LLM Studio needs {missing}. Open Setup."

    parts = [f"Model: {state['quantization'] or state['model'] or 'unknown'}",
             f"Device: {state['device'] or 'unknown'}",
             "Server: running" if state["running"] else "Server: stopped"]
    if state["running"] and state["resident_bytes"]:
        parts.append(f"VRAM: {ui.gigabytes(state['resident_bytes'])}")
    report = state["report"]
    if report is not None and report.placement is not None:
        parts.append(f"Context: {ui.tokens(report.placement.context)}")
    # What llama.cpp said, not what it was asked for. The two agreeing is the
    # answer to "is it really all on the GPU?", and that question is only ever
    # asked by somebody whose replies are slower than they should be.
    if report is not None and report.offload.known:
        parts.append(f"llama.cpp: {report.offload.describe()}")
    if not state["sees"]:
        parts.append("No vision projector")
    return " · ".join(parts)


def _measured_speed() -> str:
    """What llama.cpp timed for the most recent request, or ""."""
    import mc_llm_runtime

    try:
        return mc_llm_runtime.runtime.speed_note()
    except Exception:
        logger.debug("Model Chain: could not read llama.cpp's timings", exc_info=True)
        return ""


def _log_path() -> str:
    """Where llama-server writes. Asked for by name often enough to be a function."""
    try:
        return str(mc_llm_paths.app_paths().logs / "llama-server.log")
    except Exception:
        logger.debug("Model Chain: could not work out the LLM log path", exc_info=True)
        return "unknown"


def _residency_html() -> str:
    """The detailed residency view (section 14), kept out of the main UI."""
    try:
        return _residency_table()
    except Exception:
        logger.warning("Model Chain: the residency view could not be drawn", exc_info=True)
        return ui.notice("Residency information unavailable — see the console.", "warn")


def _residency_table() -> str:
    import mc_llm_runtime

    status = mc_broker.status()
    state = mc_llm_runtime.runtime.status()

    rows = []
    for entry in status.residencies:
        rows.append(
            f"<tr><td>{ui.escape(entry.label)}</td>"
            f"<td>{ui.escape(entry.family)}</td>"
            f"<td>{ui.gigabytes(entry.bytes)}</td>"
            f"<td>{ui.escape(mc_broker.RANK_LABELS.get(entry.effective_rank, '?'))}</td></tr>")
    if not rows:
        rows.append('<tr><td colspan="4">Nothing is registered as VRAM-resident.</td></tr>')

    running = status.active
    owners = ", ".join(status.owners) or "nothing"
    summary = [
        # First, and spelled out: the top bar's chip says "Loaded" and this is
        # where "loaded as what?" is answered. The chip's tooltip carries the
        # same sentence, but a tooltip is not somewhere anybody reads carefully.
        f"<li>LLM runtime: {ui.escape(_runtime_detail(state))}</li>",
        f"<li>Mode: <b>{ui.escape(mc_broker.label_for(mc_broker.MODES, status.mode))}</b></li>",
        f"<li>Policy: <b>{ui.escape(mc_broker.label_for(mc_broker.POLICIES, status.policy))}</b></li>",
        f"<li>VRAM: {ui.gigabytes(status.free_vram)} free of "
        f"{ui.gigabytes(status.total_vram)}, {ui.gigabytes(status.reserve)} reserved</li>",
        f"<li>VRAM owners: {ui.escape(owners)}</li>",
        f"<li>Active workload: {ui.escape(running.label) if running else 'none'}</li>",
        # Where the file is, always, whether or not anything has gone wrong
        # with it. It is the one thing in this panel that a user has to be
        # able to find on a day when the panel itself is not enough.
        f"<li>llama-server log: <code>{ui.escape(_log_path())}</code></li>",
    ]
    # The measurement, not the plan. Everything above this line is what was
    # decided; this is what it produced, and the two have differed by a factor
    # of forty while every line of the plan read "all layers on the GPU".
    measured = _measured_speed()
    if measured:
        summary.append(f"<li>Last reply: <b>{ui.escape(measured)}</b></li>")
    stray = mc_broker.unaccounted_bytes()
    if stray > 0:
        # Named here as well as in the console, because this is the panel
        # somebody opens when a placement makes no sense, and VRAM held by
        # another process is the explanation that no row in the table below
        # can ever show.
        summary.append(
            f"<li><b>{ui.gigabytes(stray)}</b> of the card is in use by something this "
            f"WebUI is not managing — another program on the same GPU, or a llama-server "
            f"left running by a previous session. Nothing here can reclaim it.</li>")
    report = state.get("report")
    if report is not None and report.placement is not None:
        summary.append(
            f"<li>LLM placement: {ui.escape(report.placement.describe())}, "
            f"{ui.tokens(report.placement.context)} token context</li>")
    for text in (report.notes if report is not None else ()):
        summary.append(f"<li>Reported change: {ui.escape(text)}</li>")

    decisions = "".join(f"<li>{ui.escape(entry.text)}</li>"
                        for entry in reversed(mc_broker.decisions(8)))
    return (
        f'<div class="{ui.PREFIX}-residency">'
        f'<ul>{"".join(summary)}</ul>'
        f'<table class="{ui.PREFIX}-table"><thead><tr><th>Resident</th><th>Family</th>'
        f'<th>VRAM</th><th>Rank</th></tr></thead><tbody>{"".join(rows)}</tbody></table>'
        f'<h4>Recent decisions</h4><ul>{decisions or "<li>No decisions yet.</li>"}</ul>'
        f'</div>'
    )


# --------------------------------------------------------------------------- #
# Setup mode
# --------------------------------------------------------------------------- #


def _setup_panel() -> dict:
    """Setup mode: the runtime, the model, what fits, and what is resident.

    Only the things that are not plain values. Context sizing, the cache types,
    the residency mode, the policy and the release behaviour were all controls
    here once and are now Forge settings, because every one of them describes
    the installation rather than the click being made -- the same test that put
    the VRAM reserve and the progress theme on the Settings page. What is left
    is the four things a Settings page cannot draw: a file dialog, a download,
    an estimate and a table.
    """
    import mc_llm_browse
    import mc_llm_runtime
    import mc_llm_setup

    configuration = mc_llm_runtime.config()

    with gr.Row(elem_classes=ui.classes("workspace")):
        with gr.Column(scale=2, min_width=320):
            # The runtime comes first because it is first: a model cannot be
            # chosen until there is something to run it with, and the panel
            # used to offer the second step without the first.
            gr.Markdown("#### llama.cpp runtime")
            # Asked once and reused: each call stats the state file and walks
            # the runtime directory, and the panel wants three answers from it.
            found = _runtime_status()
            choices = _device_choices()
            runtime_notice = gr.HTML(_runtime_setup_line(found))
            runtime_path = gr.Textbox(
                label="llama-server", value=str(found.recorded or found.found or ""),
                placeholder="Path to llama-server, or to the folder holding it",
                elem_id=ui.ident("settings", "runtime"))
            mc_llm_browse.attach(runtime_path, suffixes=(), key="runtime",
                                 allow_folders=True, title="Choose llama-server",
                                 folder_title="Choose an unpacked llama.cpp release",
                                 fallback=_folder(mc_llm_setup.RUNTIME_DIRNAME))
            device = gr.Dropdown(
                label="Device", choices=choices, value=_current_device(choices),
                elem_id=ui.ident("settings", "device"))
            with gr.Row():
                use_runtime = gr.Button("Use this runtime", variant="primary", size="sm")
                detect_runtime = gr.Button("Detect", size="sm")
                fetch_runtime = gr.Button("Download the pinned build", size="sm",
                                          interactive=found.downloadable)

            gr.Markdown("#### Which model runs")
            gr.Markdown(
                "A model can live anywhere on the machine — it is read, not started, so it "
                "does not have to be copied in. Press Browse for a file dialog, or paste a "
                "path; a folder is as good as a file when it holds one model. Anything under "
                f"the models folder ({_models_folder()}) is also offered in the chooser in the "
                "model sheet, so it can be switched to without coming back here.",
                elem_classes=ui.classes("hint"))
            model_path = gr.Textbox(
                label="GGUF model", value=str(configuration.model or ""),
                placeholder="Path to a .gguf file, or to the folder holding it",
                elem_id=ui.ident("settings", "model"))
            mc_llm_browse.attach(model_path, key="model", title="Choose a GGUF model",
                                 fallback=_folder("models"))
            mmproj_path = gr.Textbox(
                label="Vision projector (optional)", value=str(configuration.mmproj or ""),
                placeholder="Path to an mmproj .gguf, or empty for a text-only model",
                elem_id=ui.ident("settings", "mmproj"))
            mc_llm_browse.attach(mmproj_path, key="mmproj", fallback=_folder("models"),
                                 title="Choose a vision projector")
            with gr.Row():
                apply_model = gr.Button("Use this model", variant="primary", size="sm")
                suggest = gr.Button("Find the projector beside it", size="sm")
            model_notice = gr.HTML(_model_line(configuration))

        with gr.Column(scale=3, min_width=360):
            gr.Markdown("#### What fits")
            estimator = gr.HTML(_estimator_html(), elem_id=ui.ident("settings", "estimator"))
            estimate_now = gr.Button("Re-estimate", size="sm")

            gr.Markdown("#### Residency")
            residency = gr.HTML(_residency_html(), elem_id=ui.ident("settings", "residency"))
            refresh_residency = gr.Button("Refresh", size="sm")

            gr.Markdown("#### Everything else")
            gr.Markdown(_settings_pointer(), elem_classes=ui.classes("hint"))

    # -- wiring ----------------------------------------------------------- #

    use_runtime.click(fn=_apply_runtime, inputs=[runtime_path, device],
                      outputs=[runtime_notice, runtime_path, model_notice], queue=False)
    detect_runtime.click(fn=_detect_runtime, outputs=[runtime_notice, runtime_path],
                         queue=False)
    fetch_runtime.click(fn=_download_runtime, inputs=[device],
                        outputs=[runtime_notice, runtime_path])

    apply_model.click(fn=_apply_model, inputs=[model_path, mmproj_path],
                      outputs=[model_notice, estimator, model_path, mmproj_path],
                      queue=False)
    suggest.click(fn=_suggest_projector, inputs=[model_path],
                  outputs=[mmproj_path, model_notice], queue=False)

    estimate_now.click(fn=lambda: _estimator_html(), outputs=[estimator], queue=False)
    refresh_residency.click(fn=_residency_html, outputs=[residency], queue=False)

    return {"residency": residency, "estimator": estimator, "model": model_path,
            "mmproj": mmproj_path, "notice": model_notice, "runtime": runtime_path}


def _models_folder() -> str:
    try:
        return str(mc_llm_paths.models_root())
    except Exception:
        logger.debug("Model Chain: could not read the models folder", exc_info=True)
        return "the models folder"


def _settings_pointer() -> str:
    """Where the plain values went, named so nobody has to go looking.

    Written out rather than left implicit: a control that used to be on this
    panel and is not any more is indistinguishable, from the reader's side,
    from a control that was removed.
    """
    return (
        "Context sizing and buffer, the key/value cache types, the VRAM residency mode, what "
        "happens when the LLM and an image model do not both fit, what the image side gets "
        "back, the LLM data directory and the models folder are all on the WebUI's "
        "**Settings** page, under **Model Chain**. They describe this installation rather "
        "than this click, so the host stores them with the rest of its configuration and "
        "they survive a restart."
    )


def _model_line(configuration, notes=()) -> str:
    """What is set up to run, and anything that was decided on the way there.

    ``notes`` carries what :mod:`mc_llm_files` had to work out from what was
    typed — a folder resolved to the one model in it, a shard corrected to the
    first, a name matched case-insensitively. None of those is an error and all
    of them are things somebody should be told happened, because the path now
    recorded is not the path they entered.
    """
    if configuration.runtime is None:
        return ui.notice("Set up a llama.cpp runtime above before choosing a model.", "warn")
    if configuration.model is None:
        return ui.notice("No model chosen yet. Enter the path to a .gguf file, or press "
                         "Browse.", "warn")
    line = (f"{configuration.model.name} · "
            f"{'vision projector loaded' if configuration.sees else 'text only'}")
    return ui.notice(" ".join([line] + [str(note) for note in notes]))


# --------------------------------------------------------------------------- #
# Runtime setup
# --------------------------------------------------------------------------- #


def _folder(name: str):
    """A folder under the install root, for a picker to open at.

    Best-effort: the pickers fall back to the first of
    :func:`mc_llm_files.places` when the root cannot be read, which is the
    right answer on an installation that has not been set up yet.
    """
    try:
        return mc_llm_paths.data_root() / name
    except Exception:
        logger.debug("Model Chain: could not read the LLM data directory", exc_info=True)
        return None


def _runtime_status():
    import mc_llm_setup

    try:
        return mc_llm_setup.status()
    except Exception:
        logger.debug("Model Chain: could not read the runtime status", exc_info=True)
        return mc_llm_setup.RuntimeStatus(None, None, False, "")


def _runtime_setup_line(found=None) -> str:
    """What is missing, and which of the three routes will fix it."""
    found = found if found is not None else _runtime_status()
    if found.ready:
        return ui.notice(f"Runtime: {found.recorded}")
    if found.adoptable:
        return ui.notice(
            f"A llama.cpp build is already in place at {found.found} but is not recorded. "
            f"Press Detect to use it.", "warn")

    root = mc_llm_paths.data_root()
    routes = [
        f"LLM Studio keeps its runtime and models in {root}.",
        "Point the box below at a llama.cpp release you already have and press "
        "Use this runtime — it will be copied in.",
    ]
    if found.downloadable:
        routes.append("Or press Download the pinned build to fetch and verify one.")
    else:
        routes.append(found.detail)
    routes.append(
        "Already have a Prompt Master install? Set the Model Chain LLM data directory "
        "setting (or PROMPT_MASTER_ROOT) to its root and reuse it whole."
    )
    return ui.notice(" ".join(routes), "warn")


def _device_choices() -> list[tuple[str, str]]:
    import mc_llm_setup

    try:
        return [(mc_llm_setup.describe_device(found), str(found.physical_index))
                for found in mc_llm_setup.devices()]
    except Exception:
        logger.debug("Model Chain: could not list devices", exc_info=True)
        return []


def _current_device(choices=None) -> str | None:
    import mc_llm_runtime

    index = mc_llm_runtime.config().gpu_index
    choices = _device_choices() if choices is None else choices
    values = [value for _, value in choices]
    if str(index) in values:
        return str(index)
    return values[0] if values else None


def _device_for(value):
    """The detected device the dropdown's value refers to."""
    import mc_llm_setup

    try:
        wanted = int(value)
    except (TypeError, ValueError):
        return mc_llm_setup.preferred_device()
    for found in mc_llm_setup.devices():
        if found.physical_index == wanted:
            return found
    return mc_llm_setup.preferred_device()


def _apply_runtime(path, device_value):
    """Adopt the build at ``path`` and record it. The panel's first step."""
    import mc_llm_runtime
    import mc_llm_setup

    try:
        chosen = mc_llm_files.resolve_runtime(path)
    except mc_llm_files.PathError as exc:
        # Not an error in the sense the panel colours red: nothing was attempted
        # and the sentence says what to do instead.
        return ui.notice(str(exc), "warn"), gr.update(), gr.update()

    # Stopped *before* the copy, not after it. The running server holds the
    # build it was started with -- on Windows it holds every DLL beside it open,
    # and adopting a build replaces that whole folder, so a server still up made
    # the copy fail with "[WinError 5] Access is denied:
    # ...\\runtime\\cublas64_12.dll" and left the runtime half-swapped. The stop
    # was already here; it was in the wrong place, which is a thing that only
    # shows on the platform that locks open files.
    mc_llm_runtime.runtime.stop()
    try:
        executable, note = mc_llm_setup.adopt(chosen.path)
        mc_llm_setup.record(executable, _device_for(device_value))
    except Exception as exc:
        return ui.notice(ui.failure(exc), "error"), gr.update(), gr.update()

    return (ui.notice(f"{note} Runtime recorded."), str(executable),
            _model_line(mc_llm_runtime.config()))


def _detect_runtime():
    """Record a build already sitting under the install root."""
    import mc_llm_setup

    found = mc_llm_setup.detect()
    if found is None:
        return (ui.notice(f"No llama-server found under "
                          f"{mc_llm_paths.data_root() / mc_llm_setup.RUNTIME_DIRNAME}.",
                          "warn"),
                gr.update())
    try:
        mc_llm_setup.record(found)
    except Exception as exc:
        return ui.notice(ui.failure(exc), "error"), gr.update()
    return ui.notice(f"Found and recorded {found}."), str(found)


def _download_runtime(device_value, progress=gr.Progress()):
    """Fetch the pinned build. Weights are not downloaded here."""
    import mc_llm_runtime
    import mc_llm_setup

    # Before the download rather than after it, for the reason ``_apply_runtime``
    # stops first: extracting the pinned build replaces the runtime folder, and
    # a running server holds the files in it open.
    mc_llm_runtime.runtime.stop()
    try:
        chosen = _device_for(device_value)
        executable = mc_llm_setup.download(
            chosen,
            on_status=lambda text: progress(0, desc=text),
            on_progress=lambda fraction: progress(fraction))
        mc_llm_setup.record(executable, chosen)
    except Exception as exc:
        return ui.notice(ui.failure(exc), "error"), gr.update()

    return ui.notice(f"Downloaded and recorded {executable}."), str(executable)


def _apply_model(model, mmproj):
    """Point the install at a different GGUF, without re-provisioning anything.

    Four outputs rather than two: the two path boxes are written back with what
    was actually recorded. A user who pasted a folder, or a quoted path, or the
    third shard of a split model has just had it turned into something else,
    and the box they typed into is where they will look to find out what.
    """
    import mc_llm_runtime
    from prompt_master.inference import model_choice

    unchanged = (gr.update(), gr.update(), gr.update())
    if mc_llm_runtime.config().runtime is None:
        # Checked here so the answer names something in this tab. Upstream's own
        # refusal ends "Run Models and Hardware setup first", which is a Qt
        # wizard this extension does not have -- a dead end rather than an
        # instruction.
        return (ui.notice("There is no llama.cpp runtime yet, so there is nothing to run a "
                          "model with. Set one up under llama.cpp runtime above first.",
                          "warn"),) + unchanged

    try:
        chosen = mc_llm_files.resolve_model(model)
        projector = mc_llm_files.resolve_projector(mmproj, chosen.path)
    except mc_llm_files.PathError as exc:
        return (ui.notice(str(exc), "warn"),) + unchanged
    except Exception as exc:
        return (ui.notice(ui.failure(exc), "error"),) + unchanged

    try:
        model_choice.choose(mc_llm_paths.app_paths(), chosen.path,
                            projector.path if projector is not None else None)
    except Exception as exc:
        return (ui.notice(ui.failure(exc), "error"),) + unchanged

    # The running server holds the weights it was started with, so it is
    # stopped rather than left to answer as the previous model.
    mc_llm_runtime.runtime.stop()
    notes = list(chosen.notes) + list(projector.notes if projector is not None else ())
    if projector is None:
        notes.extend(_projector_hint(chosen.path))
    return (_model_line(mc_llm_runtime.config(), notes), _estimator_html(),
            str(chosen.path), str(projector.path) if projector is not None else "")


def _projector_hint(model) -> list[str]:
    """Mention a projector sitting beside a model chosen without one.

    Mentioned and not used: a projector has to match the model it was made for
    and a file name does not prove that it does, which is the vendored module's
    own reasoning for asking rather than inferring. What is unhelpful is saying
    nothing at all, because "text only" on a vision model reads as a bug.
    """
    from prompt_master.inference import model_choice

    try:
        found = model_choice.projector_beside(model)
    except Exception:
        logger.debug("Model Chain: could not look for a projector", exc_info=True)
        return []
    if found is None:
        return []
    return [f"{found.name} sits beside it and may be its vision projector — press Find the "
            f"projector beside it to use it."]


def _suggest_projector(model):
    from prompt_master.inference import model_choice

    try:
        chosen = mc_llm_files.resolve_model(model)
    except mc_llm_files.PathError as exc:
        return gr.update(), ui.notice(str(exc), "warn")
    found = model_choice.projector_beside(chosen.path)
    if found is None:
        return gr.update(), ui.notice(
            f"No projector was found beside that model. {chosen.path.parent} holds nothing "
            f"named mmproj or projector — a text-only model simply has none.", "warn")
    return str(found), ui.notice(f"Suggested {found.name} — check it belongs to this model, "
                                 f"then press Use this model.")


# --------------------------------------------------------------------------- #
# The estimator panel (section 12)
# --------------------------------------------------------------------------- #


def _estimator_html() -> str:
    """Section 12's table, plus the two hybrid answers it asks for.

    Everything shown is per model. When the header cannot be read the panel
    says so rather than filling the table with a constant.

    Wrapped, and the wrapper is the point. This is an output of *Use this
    model*, so anything it raises takes the whole handler down with it: the
    model is recorded, nothing on screen changes, and the only thing the user
    is told is the word "Error" in a toast. A panel that cannot draw a table
    has to say which panel and why, and let the rest of the click stand.
    """
    try:
        return _estimate_html()
    except Exception as exc:
        logger.warning("Model Chain: the context estimator could not be drawn", exc_info=True)
        return ui.notice(f"What fits could not be estimated: {ui.failure(exc)}", "error")


def _attention(described) -> str:
    """The shape the cache cost comes from, in one clause.

    Worth spelling out when it varies: a hybrid model's cache is a fraction of
    what its block count suggests, and a reader who sees "62 blocks" beside a
    small number has no other way to know why.
    """
    if described.uniform_attention:
        return f"{described.block_count} blocks × {described.head_count_kv} KV heads"
    return (f"{described.attending_blocks} of {described.block_count} blocks keep a cache, "
            f"up to {described.head_count_kv} KV heads each")


def _estimate_html() -> str:
    import mc_gguf
    import mc_llm_context
    import mc_llm_runtime

    configuration = mc_llm_runtime.config()
    if not configuration.model:
        return ui.notice("Choose a GGUF to see what context fits.", "warn")

    described = mc_gguf.describe(configuration.model)
    if described is None or not described.usable:
        return ui.notice(
            f"{configuration.model.name} does not describe its attention shape in its GGUF "
            f"header, so context capacity cannot be estimated for it. The context size you set "
            f"is still used.", "warn")

    # VRAM the running server is holding. It belongs on the free side of every
    # figure below, for the reason it belongs there in the negotiation itself:
    # a re-placement stops that server first, so its own footprint is not a
    # constraint on where it can go. A table drawn without this term reads a
    # loaded model as a card with no room on it, and reports that the model
    # currently answering at 7,168 tokens could not be given a context at all.
    ours = mc_llm_runtime.runtime.resident_bytes()

    try:
        # reclaim=False: this panel is drawn when the tab is built and whenever
        # the accordion opens, and drawing a table is not a reason to evict
        # anybody's checkpoint.
        negotiated = mc_llm_runtime.negotiate(configuration, described, reclaim=False,
                                              already_ours=ours)
    except Exception as exc:
        return ui.notice(ui.failure(exc), "error")

    placement = negotiated.placement
    estimate = negotiated.estimate
    per_token = estimate.kv_bytes_per_token
    reserve = mc_broker.safety_margin_bytes()
    free = mc_broker.device_free_vram_bytes() + ours
    image_resident = mc_broker.resident_bytes(mc_broker.FAMILY_IMAGE)

    rows = []
    for found in mc_llm_context.table(configuration.model, placement):
        capped = " (model ceiling)" if found.limited_by_model else ""
        rows.append(
            f"<tr><td>{ui.gigabytes(found.budget_bytes)}</td>"
            f"<td>{ui.tokens(found.theoretical)}</td>"
            f"<td>{ui.tokens(found.usable)}{capped}</td></tr>")

    # The two questions section 12 asks the same estimator to answer.
    keeping = mc_llm_context.automatic_buffer_bytes(free, estimate.weights_bytes,
                                                    reserve + estimate.compute_bytes)
    moving = mc_llm_context.automatic_buffer_bytes(free + image_resident, estimate.weights_bytes,
                                                   reserve + estimate.compute_bytes)
    with_image = mc_llm_context.capacity(configuration.model, placement, keeping, gguf=described)
    without_image = mc_llm_context.capacity(configuration.model, placement, moving, gguf=described)

    facts = [
        f"<li>Model ceiling: <b>{ui.tokens(described.context_length)}</b> tokens</li>",
        f"<li>Current context: <b>{ui.tokens(placement.context)}</b> tokens "
        f"({ui.gigabytes(estimate.kv_bytes)} of key/value cache)</li>",
        f"<li>Cost per token: {per_token:,.0f} bytes ({_attention(described)})</li>",
        f"<li>Weights on the GPU: {ui.gigabytes(estimate.weights_bytes)} "
        f"({ui.escape(placement.describe(described.block_count))})</li>",
        f"<li>Runtime reserve: {ui.megabytes(estimate.compute_bytes)} — "
        f"<b>{'calibrated from a real load' if estimate.calibrated else 'estimated'}</b></li>",
        f"<li>Keeping the current image model resident: "
        f"<b>{ui.tokens(with_image.usable)}</b> tokens</li>",
        f"<li>If the image model is demoted to system RAM: "
        f"<b>{ui.tokens(without_image.usable)}</b> tokens</li>",
    ]
    if estimate.capped:
        facts.append("<li>Context is limited by the model's own ceiling, not by VRAM.</li>")
    if estimate.detail:
        facts.append(f"<li>{ui.escape(estimate.detail)}</li>")
    for text in negotiated.notes:
        facts.append(f"<li>Would be changed to fit: {ui.escape(text)}</li>")

    return (
        f'<div class="{ui.PREFIX}-estimator">'
        f'<ul>{"".join(facts)}</ul>'
        f'<table class="{ui.PREFIX}-table"><thead><tr><th>Context buffer</th>'
        f'<th>Theoretical tokens</th><th>Recommended</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>'
        f'</div>'
    )
