"""Gallery Quadrant: five tap zones over the gallery's preview, and where the
top-left one saves to.

Carried in from the user's own ``sd-gallery-quadrant`` extension (2026-10-17),
so that one extension is installed instead of two. The script
(``javascript/model_chain_gallery_quadrant.js``) and the stylesheet section
(``style.css``, *Gallery Quadrant*) are that extension's, as they were, under
the same class names and the same ``data-gqa-*`` markers: a page where the
standalone extension is still enabled draws one overlay, because whichever
script reaches a gallery first marks it and the other skips it. The user
disables the standalone once this is in.

What the zones do, from the standalone's own description::

    ┌───────────┬───────────┬───────────┐
    │⬇ Download │  ✕ Unfocus│  💾 Save  │
    ├─────────────────┬─────────────────┤
    │     ◀ Prev      │      Next ▶     │
    └─────────────────┴─────────────────┘

Top-right is Forge's own Save (``saveImage()``, a copy into ``outdir_save``);
the bottom row is Forge's ``modalImageSwitch``; the middle leaves the preview
by pressing Gradio's Clear. It all needs the lightbox off
(``js_modal_lightbox``), as the standalone did.

The one change, at the user's word: the top-left action's **destination**.
It downloaded through the browser, into whatever folder the browser saves
to; now a setting chooses between that and **a folder on this machine**,
named in a second setting, that the server writes the picture into. The
page posts the preview's ``<img src>`` -- a Gradio file URL -- to
:data:`ROUTE`; :func:`resolve_source` turns it back into the file it names,
and takes only an existing file under one of Forge's output folders or
Gradio's temporary folder (the two places a gallery picture is served from),
so the route copies what the gallery shows and nothing else;
:func:`save_copy` writes it into the folder under its own name and never
overwrites -- the same picture saved twice is the one file, a different
picture of the same name gets a numbered suffix. The folder is made if it is
missing. Every refusal is a sentence the page shows in the zone.

All the settings live in one section, *Model Chain: Gallery Quadrant*
(:data:`SECTION`): the switch, on by default, as the standalone's was; the
destination; the folder.
"""

from __future__ import annotations

import filecmp
import logging
import os
import shutil
import tempfile
import urllib.parse
from pathlib import Path
from typing import Iterable, Mapping

logger = logging.getLogger("model_chain")

# FastAPI reads the route handler's annotations by name from the handler's
# module globals, and under ``from __future__ import annotations`` every
# annotation is a string. ``Request`` therefore has to be a *module* global:
# imported inside :func:`install`, the string "Request" resolved to nothing,
# FastAPI read ``request`` as a required query parameter, and every save from
# a browser answered HTTP 422 (found by the user from a phone). Guarded, so a
# host without FastAPI (the tests' stand-in) still imports this module.
try:
    from fastapi import Request  # type: ignore
    from fastapi.responses import JSONResponse  # type: ignore
except Exception:  # pragma: no cover - a host without FastAPI
    Request = None  # type: ignore
    JSONResponse = None  # type: ignore

SECTION = ("model_chain_gallery_quadrant", "Model Chain: Gallery Quadrant")

OPT_ENABLED = "model_chain_gallery_quadrant"
OPT_DESTINATION = "model_chain_gallery_quadrant_destination"
OPT_FOLDER = "model_chain_gallery_quadrant_folder"

DESTINATION_BROWSER = "The browser's downloads"
DESTINATION_FOLDER = "A folder on this machine"
DESTINATIONS = (DESTINATION_BROWSER, DESTINATION_FOLDER)

ROUTE = "/model-chain/gallery-quadrant/save"

# Forge's output folders, every one an option holding a path relative to the
# data directory or absolute. A gallery picture is served from one of these,
# or from Gradio's temporary folder when a path is outside what Gradio serves
# directly.
OUTPUT_OPTIONS = (
    "outdir_samples", "outdir_txt2img_samples", "outdir_img2img_samples",
    "outdir_extras_samples", "outdir_grids", "outdir_txt2img_grids",
    "outdir_img2img_grids", "outdir_save", "outdir_init_images",
)

MAX_SUFFIX = 10000


class SaveRefused(Exception):
    """A save the route will not make, with the status the answer carries."""

    def __init__(self, status: int, reason: str):
        super().__init__(reason)
        self.status = int(status)
        self.reason = reason


# --------------------------------------------------------------------------- #
# The settings
# --------------------------------------------------------------------------- #


def settings(shared) -> dict:
    """The options of the Settings section, in the order they are drawn."""
    OptionInfo = shared.OptionInfo
    import gradio as gr  # noqa: F811 -- the host's, for the component classes

    return {
        OPT_ENABLED: OptionInfo(
            True,
            "Gallery quadrant actions (needs the lightbox off)",
        ).info(
            "splits the gallery's preview into five tap zones: top-left saves the "
            "picture (to the browser's downloads, or to the folder below), top-middle "
            "leaves the preview, top-right is Forge's own Save, and the bottom row is "
            "previous and next. Carried in from the sd-gallery-quadrant extension, "
            "which can be disabled once this is on"
        ),
        OPT_DESTINATION: OptionInfo(
            DESTINATION_BROWSER,
            "Where the top-left zone saves the picture",
            gr.Radio,
            {"choices": list(DESTINATIONS)},
        ).info(
            "the browser's downloads, as before, or a folder of your own on the machine "
            "the WebUI runs on, written by the WebUI itself under the picture's own name; "
            "a picture saved twice is one file, and a different picture of the same name "
            "gets a number"
        ),
        OPT_FOLDER: OptionInfo(
            "",
            "That folder",
        ).info(
            "an absolute path, or one relative to the WebUI's data directory; made if it "
            "is missing. Empty with the folder chosen above, the zone says so instead of "
            "saving"
        ),
    }


def destination_of(opts) -> str:
    """The destination the settings choose: one of :data:`DESTINATIONS`."""
    value = str(getattr(opts, OPT_DESTINATION, "") or "")
    return DESTINATION_FOLDER if value == DESTINATION_FOLDER else DESTINATION_BROWSER


# --------------------------------------------------------------------------- #
# Where a gallery picture is, and where it may go
# --------------------------------------------------------------------------- #


def data_directory() -> Path:
    """The WebUI's data directory, which Forge's relative output paths are
    relative to; the working directory where the host's module is not there
    (the tests, a stand-in host)."""
    try:
        from modules import paths  # type: ignore
        found = getattr(paths, "data_path", None)
        if found:
            return Path(found)
    except Exception:
        pass
    return Path.cwd()


def gradio_temp_directory() -> Path:
    """Where Gradio copies a file it serves from outside its allowed paths:
    ``GRADIO_TEMP_DIR``, else ``gradio`` under the system's temporary folder
    (Gradio 4.40's own rule)."""
    named = os.environ.get("GRADIO_TEMP_DIR")
    return Path(named) if named else Path(tempfile.gettempdir()) / "gradio"


def output_roots(opts, data_dir: Path | None = None, temp_dir: Path | None = None) -> list:
    """The folders a gallery picture can be served from, resolved: Forge's
    output options that are set, and Gradio's temporary folder."""
    base = Path(data_dir) if data_dir is not None else data_directory()
    roots = []
    for name in OUTPUT_OPTIONS:
        value = getattr(opts, name, None) if opts is not None else None
        if not value or not isinstance(value, str):
            continue
        path = Path(value)
        if not path.is_absolute():
            path = base / path
        roots.append(_resolved(path))
    roots.append(_resolved(Path(temp_dir) if temp_dir is not None else gradio_temp_directory()))
    unique = []
    for root in roots:
        if root not in unique:
            unique.append(root)
    return unique


def _resolved(path: Path) -> Path:
    try:
        return Path(os.path.realpath(str(path)))
    except Exception:
        return path


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def file_of_url(src: str) -> str:
    """The path a Gradio file URL names: ``/file=<path>`` with or without a
    host, percent-encoded, with a query or a fragment after it. Empty when the
    URL is not one of those."""
    text = str(src or "").strip()
    if not text:
        return ""
    parsed = urllib.parse.urlsplit(text)
    path = parsed.path if parsed.scheme or parsed.netloc else text.split("?", 1)[0].split("#", 1)[0]
    marker = "/file="
    at = path.find(marker)
    if at < 0:
        return ""
    return urllib.parse.unquote(path[at + len(marker):])


def resolve_source(src: str, roots: Iterable[Path]) -> Path:
    """The existing file the gallery's picture is, under one of ``roots``.

    Raises :class:`SaveRefused` for a URL that is not a Gradio file URL, a
    file that is not there, or one outside every root -- the route copies what
    the gallery shows and nothing else."""
    named = file_of_url(src)
    if not named:
        raise SaveRefused(400, "That is not a picture the gallery serves.")
    # Windows paths arrive with backslashes, percent-encoded or not; on a POSIX
    # host a backslash is a character of the name, which no output folder has.
    path = _resolved(Path(named))
    allowed = [Path(root) for root in roots]
    if not any(_is_under(path, root) for root in allowed):
        raise SaveRefused(403, "That picture is not in one of Forge's output folders.")
    if not path.is_file():
        raise SaveRefused(404, "That picture is no longer on disk.")
    return path


def resolve_folder(value: str, data_dir: Path | None = None) -> Path:
    """The folder the setting names, resolved: absolute as given, relative to
    the data directory otherwise. Raises :class:`SaveRefused` for an empty
    setting."""
    text = str(value or "").strip()
    if not text:
        raise SaveRefused(400, "No folder is set: Settings → Model Chain: Gallery Quadrant → That folder.")
    path = Path(os.path.expanduser(text))
    if not path.is_absolute():
        base = Path(data_dir) if data_dir is not None else data_directory()
        path = base / path
    return _resolved(path)


def save_copy(source: Path, folder: Path) -> Path:
    """Copy ``source`` into ``folder`` under its own name, never over another
    file: the same picture already there is left as it is and answered; a
    different file of that name gets ``name-1``, ``name-2``, ... The folder is
    made if it is missing."""
    folder = Path(folder)
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise SaveRefused(500, f"That folder cannot be made: {exc.strerror or exc}.")
    if not folder.is_dir():
        raise SaveRefused(500, "That folder is a file.")
    source = Path(source)
    stem, suffix = source.stem, source.suffix
    target = folder / source.name
    for n in range(1, MAX_SUFFIX + 1):
        if not target.exists():
            break
        try:
            if target.is_file() and filecmp.cmp(str(source), str(target), shallow=False):
                return target
        except OSError:
            pass
        target = folder / f"{stem}-{n}{suffix}"
    else:
        raise SaveRefused(500, "That folder holds too many pictures of that name.")
    try:
        shutil.copy2(str(source), str(target))
    except OSError as exc:
        raise SaveRefused(500, f"The picture could not be written there: {exc.strerror or exc}.")
    return target


# --------------------------------------------------------------------------- #
# The route
# --------------------------------------------------------------------------- #


def same_origin(headers: Mapping) -> bool:
    """Did the request come from this WebUI's own page? The browser sends the
    page's origin on a cross-site POST and on a same-site one alike; a request
    with neither Origin nor Referer is not a browser's, and is refused too."""
    host = str(headers.get("host") or "").strip().lower()
    for name in ("origin", "referer"):
        value = str(headers.get(name) or "").strip()
        if not value:
            continue
        try:
            parsed = urllib.parse.urlsplit(value)
        except ValueError:
            return False
        return bool(host) and parsed.netloc.lower() == host
    return False


def save(payload: Mapping, *, opts, roots: Iterable[Path] | None = None,
         data_dir: Path | None = None) -> dict:
    """The route's work: the picture the payload names, copied into the
    folder the settings name. Raises :class:`SaveRefused`."""
    if not isinstance(payload, Mapping):
        raise SaveRefused(400, "The request is not a save.")
    if destination_of(opts) != DESTINATION_FOLDER:
        raise SaveRefused(400, "The top-left zone is set to the browser's downloads.")
    folder = resolve_folder(getattr(opts, OPT_FOLDER, ""), data_dir)
    source = resolve_source(str(payload.get("src") or ""), roots if roots is not None else output_roots(opts, data_dir))
    written = save_copy(source, folder)
    logger.info("Model Chain: Gallery Quadrant saved %s to %s", source.name, written)
    return {"ok": True, "path": str(written), "name": written.name, "folder": str(folder)}


def handle(payload, headers: Mapping, *, opts, **kwargs) -> tuple:
    """``(status, body)`` for one request, with every refusal a sentence and
    nothing raised: what the route's handler answers with."""
    if not same_origin(headers or {}):
        return 403, {"ok": False, "error": "That request did not come from this WebUI."}
    try:
        return 200, save(payload, opts=opts, **kwargs)
    except SaveRefused as exc:
        return exc.status, {"ok": False, "error": exc.reason}
    except Exception:
        logger.warning("Model Chain: a Gallery Quadrant save failed", exc_info=True)
        return 500, {"ok": False, "error": "The save failed; the log says why."}


def install(_demo=None, app=None) -> bool:
    """Register :data:`ROUTE` on the WebUI's FastAPI app. Never fatal, and
    idempotent: a UI reload calls it again."""
    if app is None or not hasattr(app, "add_api_route"):
        return False
    for existing in getattr(app, "routes", []):
        if getattr(existing, "path", None) == ROUTE:
            return True
    if Request is None or JSONResponse is None:
        logger.debug("Model Chain: Gallery Quadrant has no FastAPI to register its route on")
        return False

    async def gallery_quadrant_save(request: Request):  # type: ignore[valid-type]
        try:
            payload = await request.json()
        except Exception:
            payload = None
        from modules import shared  # type: ignore
        status, body = handle(payload, request.headers, opts=shared.opts)
        return JSONResponse(body, status_code=status)

    try:
        app.add_api_route(ROUTE, gallery_quadrant_save, methods=["POST"])
    except Exception:
        logger.debug("Model Chain: could not register the Gallery Quadrant route", exc_info=True)
        return False
    return True
