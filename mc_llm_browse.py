"""A file picker for the path boxes, drawn in the page rather than by the OS.

The panel used to ask for three absolute paths and offer nothing but a text
box to supply them in, which is a fine interface for somebody who already has
the path on a clipboard and a poor one for everybody else -- and it is worse
than it looks, because the obvious way to get a path onto a Windows clipboard
adds quotes that make it name nothing. :mod:`mc_llm_files` undoes that; this is
the other half of the answer, which is not having to paste at all.

Why not a native dialog
-----------------------
Because the dialog would open on the wrong machine. A WebUI is a server: the
files are beside it and the browser is not necessarily on the same computer,
so a ``tkinter`` file dialog would either open on somebody's desktop when the
server is local, or open on a headless box where nobody can see it and hang the
worker thread until it is dismissed -- which nobody can do. A listing rendered
by the server is correct in both cases, and is the same thing every other file
picker in a web application is.

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

import gradio as gr

import mc_llm_files as files
import mc_llm_ui as ui

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

FOLDER_LABEL = "Folders"
FILE_LABEL = "Files here"


def attach(target, *, suffixes: tuple[str, ...] = (files.MODEL_SUFFIX,), key: str = "browse",
           allow_folders: bool = False, label: str = "Browse…", fallback=None):
    """Add a Browse button and a picker that writes into ``target``.

    ``target`` is the textbox this picker fills. ``allow_folders`` adds the
    "Use this folder" button the runtime box needs, because a llama.cpp release
    is adopted by naming either the executable or the directory around it.

    Built where it is called: the button and the panel appear at that point in
    the layout. Returns the components, for tests and for anybody who wants to
    bind something else to them.
    """
    opener = gr.Button(label, size="sm", elem_classes=ui.classes("browse-open"))

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
                            filterable=True)
        if allow_folders:
            use_folder = gr.Button("Use this folder", size="sm", variant="primary")
        notice = gr.HTML(ui.notice("Pick a folder, then a file."))

    outputs = [panel, location, places, folders, picks, notice, target]

    opener.click(fn=lambda current: _open(current, suffixes, fallback), inputs=[target],
                 outputs=outputs, queue=False)
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

    return {"open": opener, "panel": panel, "location": location, "places": places,
            "folders": folders, "files": picks, "notice": notice}


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


def _open(current, suffixes, fallback):
    """Open the picker where whatever is already in the box points."""
    return _show(files.starting_folder(current, fallback), suffixes, opened=True)


def _shut():
    return [gr.update(visible=False)] + [gr.update()] * 6


def _show(where, suffixes, opened: bool = True):
    """List ``where`` and leave the target box alone."""
    found = files.listing(where, suffixes=suffixes)
    detail = found.detail or (
        f"{len(found.files)} file{'' if len(found.files) == 1 else 's'} here, "
        f"{len(found.folders)} folder{'' if len(found.folders) == 1 else 's'}.")
    return [
        gr.update(visible=opened),
        str(found.directory),
        gr.update(choices=_places(), value=None),
        gr.update(choices=_choices(found.folders), value=None),
        gr.update(choices=_choices(found.files, sized=True), value=None),
        ui.notice(detail, "warn" if found.detail else "info"),
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
