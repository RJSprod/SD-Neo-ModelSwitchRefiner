"""Browse, for the three boxes that ask for a path.

The panel used to ask for three absolute paths and offer nothing but a text box
to supply them in, which is a fine interface for somebody who already has the
path on a clipboard and a poor one for everybody else -- and it is worse than
it looks, because the obvious way to get a path onto a Windows clipboard adds
quotes that make it name nothing. :mod:`mc_llm_files` undoes that; this is the
other half of the answer, which is not having to paste at all.

**Browse** opens the operating system's own dialog (:mod:`mc_llm_native`),
because that is what "browse" means to everybody who presses it and because
navigating to a models folder eight levels down a drive is one keystroke there
and eight clicks in anything drawn in a page.

The page's own picker is the fallback, and it exists because the native dialog
is not always the right thing to open. A WebUI started with ``--listen`` or
``--share`` is being looked at from another machine: a dialog opened by the
server appears on the *server's* screen, and from the browser's side the button
simply does nothing until it times out. So when the native route is
unavailable, this drawer opens instead and the notice inside it says why --
which is the one outcome a Browse button must never have, silence.

Why it is built per box rather than shared
------------------------------------------
Gradio wires outputs statically: one shared picker would have to know at build
time which box it was going to write into, and it cannot. Three pickers is a
little more markup and no ambiguity, and only one of them is ever open.

Navigation is bound to ``input`` rather than ``change`` where the host offers
it. A dropdown whose choices this module replaces would otherwise fire its own
handler on the replacement and walk a folder deeper on every click.
"""

from __future__ import annotations

import logging
import sys

import gradio as gr

import mc_llm_files as files
import mc_llm_native as native
import mc_llm_ui as ui

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

FOLDER_LABEL = "Folders"
FILE_LABEL = "Files here"


def attach(target, *, suffixes: tuple[str, ...] = (files.MODEL_SUFFIX,), key: str = "browse",
           allow_folders: bool = False, label: str = "Browse…", fallback=None,
           title: str = "Choose a file", folder_title: str = "Choose a folder"):
    """Add a Browse button and a picker that writes into ``target``.

    ``target`` is the textbox this fills, and ``title`` is what the operating
    system's dialog is called when it opens. ``allow_folders`` adds the "Use
    this folder" button the runtime box needs in the fallback drawer, because a
    llama.cpp release is adopted by naming either the executable or the
    directory around it.

    Built where it is called: the button and the panel appear at that point in
    the layout. Returns the components, for tests and for anybody who wants to
    bind something else to them.
    """
    with gr.Row():
        opener = gr.Button(label, size="sm", elem_classes=ui.classes("browse-open"))
        folder_opener = (gr.Button("Browse for a folder…", size="sm",
                                   elem_classes=ui.classes("browse-open"))
                         if allow_folders else None)

    with gr.Group(visible=False, elem_id=ui.ident("browse", key),
                  elem_classes=ui.classes("browse")) as panel:
        location = gr.Textbox(label="Folder", show_label=True, interactive=True,
                              placeholder="Type a folder and press Enter")
        with gr.Row():
            up = gr.Button("Up one level", size="sm")
            close = gr.Button("Close", size="sm")
        places = gr.Dropdown(label="Go to", choices=_places(), value=None, filterable=True)
        folders = gr.Dropdown(label=FOLDER_LABEL, choices=[], value=None, filterable=True)
        picks = gr.Dropdown(label=_file_label(suffixes), choices=[], value=None,
                            filterable=True, elem_classes=ui.classes("browse-files"))
        if allow_folders:
            use_folder = gr.Button("Use this folder", size="sm", variant="primary")
        notice = gr.HTML(ui.notice("Pick a folder, then a file."))

    outputs = [panel, location, places, folders, picks, notice, target]

    # Not queue=False: this one waits for a person to pick a file, which is as
    # long as they take. It belongs in the queue rather than on the threadpool
    # that answers everything else.
    opener.click(fn=lambda current: _open(current, suffixes, fallback, title), inputs=[target],
                 outputs=outputs)
    close.click(fn=lambda: _shut(), outputs=outputs, queue=False)
    up.click(fn=lambda where: _show(files.parent_of(where), suffixes), inputs=[location],
             outputs=outputs, queue=False)
    location.submit(fn=lambda where: _show(where, suffixes), inputs=[location],
                    outputs=outputs, queue=False)
    _picked(places)(fn=lambda where, current: _show(where or current, suffixes),
                    inputs=[places, location], outputs=outputs, queue=False)
    _picked(folders)(fn=lambda where, current: _show(where or current, suffixes),
                     inputs=[folders, location], outputs=outputs, queue=False)
    _picked(picks)(fn=lambda chosen, where: _choose(chosen, where, suffixes),
                   inputs=[picks, location], outputs=outputs, queue=False)
    if allow_folders:
        use_folder.click(fn=lambda where: _choose(where, where, suffixes), inputs=[location],
                         outputs=outputs, queue=False)
        folder_opener.click(
            fn=lambda current: _open_folder(current, suffixes, fallback, folder_title),
            inputs=[target], outputs=outputs)

    return {"open": opener, "open_folder": folder_opener, "panel": panel,
            "location": location, "places": places, "folders": folders, "files": picks,
            "notice": notice}


def _picked(component):
    """The event that fires when a *user* picks, not when this code refills.

    ``input`` is that event where the host's Gradio has it. Falling back to
    ``change`` keeps the picker working on a build that does not, at the cost
    of one extra navigation per refill -- which is why the fallback is a
    fallback.
    """
    return getattr(component, "input", None) or component.change


def _file_label(suffixes: tuple[str, ...]) -> str:
    if not suffixes:
        return FILE_LABEL
    return f"{FILE_LABEL} ({', '.join(suffixes)})"


# --------------------------------------------------------------------------- #
# Handlers
# --------------------------------------------------------------------------- #


def _open(current, suffixes, fallback, title="Choose a file"):
    """Ask the operating system first, and fall back into the page if it cannot.

    The three outcomes are: a file was picked, which goes straight into the box
    and opens nothing; the dialog was cancelled, which changes nothing at all;
    and there is no dialog to open here, which opens the drawer below with the
    reason in it.
    """
    start = files.starting_folder(current, fallback)
    try:
        chosen = native.choose_file(title, _patterns(suffixes), start)
    except native.Unavailable as exc:
        return _show(start, suffixes, note=str(exc))
    except Exception as exc:
        logger.warning("Model Chain: the native file dialog failed", exc_info=True)
        return _show(start, suffixes, note=f"A file dialog could not be opened ({exc}).")

    if chosen is None:
        # Cancelled. Opening the in-page picker here would be the panel
        # arguing with somebody who has just said no.
        return _shut()
    return [gr.update(visible=False)] + [gr.update()] * 5 + [str(chosen)]


def _open_folder(current, suffixes, fallback, title="Choose a folder"):
    """The same three outcomes, for the box that accepts a folder.

    The runtime box does: a llama.cpp release is adopted by naming either the
    server or the directory around it, and somebody who unpacked a release has
    the directory in mind rather than the executable inside it.
    """
    start = files.starting_folder(current, fallback)
    try:
        chosen = native.choose_folder(title, start)
    except native.Unavailable as exc:
        return _show(start, suffixes, note=str(exc))
    except Exception as exc:
        logger.warning("Model Chain: the native folder dialog failed", exc_info=True)
        return _show(start, suffixes, note=f"A folder dialog could not be opened ({exc}).")

    if chosen is None:
        return _shut()
    return [gr.update(visible=False)] + [gr.update()] * 5 + [str(chosen)]


def _patterns(suffixes: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    """Filters for the native dialog. Tk spells "everything" per platform."""
    everything = ("All files", "*.*" if sys.platform == "win32" else "*")
    if not suffixes:
        return (everything,)
    named = tuple((f"{suffix.lstrip('.').upper()} files", f"*{suffix}") for suffix in suffixes)
    return named + (everything,)


def _shut():
    return [gr.update(visible=False)] + [gr.update()] * 6


def _show(where, suffixes, note: str = ""):
    """List ``where`` in the drawer and leave the target box alone."""
    found = files.listing(where, suffixes=suffixes)
    detail = found.detail or (
        f"{len(found.files)} file{'' if len(found.files) == 1 else 's'} here, "
        f"{len(found.folders)} folder{'' if len(found.folders) == 1 else 's'}.")
    return [
        gr.update(visible=True),
        str(found.directory),
        gr.update(choices=_places(), value=None),
        gr.update(choices=_choices(found.folders), value=None),
        gr.update(choices=_choices(found.files, sized=True), value=None),
        ui.notice(" ".join(filter(None, (note, detail))),
                  "warn" if (note or found.detail) else "info"),
        gr.update(),
    ]


def _choose(chosen, where, suffixes):
    """Write the pick into the target box and close.

    A pick that arrives empty -- which is what a dropdown being refilled looks
    like on a host without an ``input`` event -- leaves everything as it was
    rather than emptying the box somebody just filled.
    """
    if not str(chosen or "").strip():
        return _show(where, suffixes)
    return [gr.update(visible=False)] + [gr.update()] * 5 + [str(chosen)]


def _choices(paths, sized: bool = False) -> list[tuple[str, str]]:
    return [(_name(path, sized), str(path)) for path in paths]


def _name(path, sized: bool) -> str:
    if not sized:
        return path.name
    try:
        return f"{path.name} — {ui.gigabytes(path.stat().st_size)}"
    except OSError:
        return path.name


def _places() -> list[tuple[str, str]]:
    try:
        return [(str(path), str(path)) for path in files.places()]
    except Exception:
        logger.debug("Model Chain: could not list browsing places", exc_info=True)
        return []
