"""The LoRA details' settings, in three sections, and the one-time carry-over
from the standalone extension.

The sections, one per area (``tests/test_settings_sections.py`` holds the
rule: a sentence or two under each setting, never a paragraph):

* **Model Chain: LoRA Details** -- the dialog, the CSV library and its
  uploader, the downloaded gallery images, how a LoRA is matched to a row;
* **Model Chain: Civitai fetch** -- the API key, what a fetch collects and
  what a refresh fills in, the request pacing, the base-model gate;
* **Model Chain: Extra networks** -- the folder filter per UI Preset and the
  folder buttons.

Every key is the standalone extension's with its prefix changed
(``eld_library_dir`` is ``model_chain_lora_library_dir`` here, and so on), so
the carry-over below is a rename. Three of the standalone's settings have no
counterpart, because the dialog they described is gone: the panel at the
bottom opened automatically, opened on a match, or drawn above the fields --
it is the page now.

**The carry-over.** The standalone extension kept its library, its images, its
folder assignments, the last key and folder it fetched with, and its hash
cache at its own root, and its settings under ``eld_*`` in ``config.json``.
:func:`import_standalone` brings all of it over once, the way the theme once
read the user's Lobe settings: the files are copied into the LoRA data root
(``mc_lora_paths``) where nothing of that name is there yet, a setting still
at its default here takes the standalone's value, and the record goes into a
hidden option, which is what makes it once. The standalone's folder is left as
it is, so uninstalling it afterwards is the user's own step; nothing here
deletes anything.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from pathlib import Path

import mc_lora_last_used
import mc_lora_paths
import mc_lora_render
from mc_lora_common import log, opt, report

logger = logging.getLogger("model_chain")

SECTION = ("model_chain_lora_details", "Model Chain: LoRA Details")
FETCH_SECTION = ("model_chain_lora_fetch", "Model Chain: Civitai fetch")
NETWORKS_SECTION = ("model_chain_extra_networks", "Model Chain: Extra networks")
HIDDEN_SECTION = (None, "Model Chain: LoRA Details (hidden)")

# -- the dialog, the library, the images, the matching ----------------------
OPT_ENABLED = "model_chain_lora_enabled"
OPT_LIBRARY_DIR = mc_lora_paths.OPT_LIBRARY_DIR
OPT_UPLOAD = "model_chain_lora_upload_csv"
OPT_HASH_LOOKUP = "model_chain_lora_hash_lookup"
OPT_HASH_ON_OPEN = "model_chain_lora_hash_on_open"
OPT_PROMPT_CARDS = "model_chain_lora_prompt_list_limit"
OPT_PROMPT_LABEL_WIDTH = "model_chain_lora_prompt_label_width"
OPT_IMAGE_DIR = mc_lora_paths.OPT_IMAGE_DIR
OPT_IMAGE_MAX_MB = "model_chain_lora_image_max_mb"
OPT_IMAGE_TIMEOUT = "model_chain_lora_image_timeout"

# -- the Civitai fetch ---------------------------------------------------------
OPT_API_KEY = "model_chain_lora_civitai_api_key"
OPT_REMEMBER_KEY = "model_chain_lora_remember_api_key"
OPT_REMEMBER_FOLDER = "model_chain_lora_remember_scan_folder"
OPT_FETCH_DESCRIPTIONS = "model_chain_lora_fetch_descriptions"
OPT_BACKFILL = "model_chain_lora_backfill_text"
OPT_RETRY_UNRESOLVED = "model_chain_lora_retry_unresolved"
OPT_REQUEST_DELAY = "model_chain_lora_request_delay"
OPT_REQUEST_TIMEOUT = "model_chain_lora_request_timeout"
OPT_MAX_IMAGES = "model_chain_lora_max_images"
OPT_BASE_MODEL = "model_chain_lora_base_model_filter"
OPT_REQUIRE_LABEL = "model_chain_lora_require_explicit_label"

# -- the extra-network browsers ---------------------------------------------
OPT_FILTER = "model_chain_lora_folder_filter_enabled"
OPT_FILTER_DIRS = "model_chain_lora_folder_filter_dirs"
OPT_FILTER_CHECKPOINT_MENU = "model_chain_lora_folder_filter_checkpoint_menu"
OPT_FILTER_STRICT = "model_chain_lora_folder_filter_strict"
OPT_FILTER_AUTO_REFRESH = "model_chain_lora_folder_filter_auto_refresh"
OPT_TOP_LEVEL_BUTTONS = "model_chain_lora_folder_buttons_top_level"

OPT_IMPORT = "model_chain_lora_import_record"
"""Hidden: the record of the one-time carry-over from the standalone extension."""

STANDALONE_PREFIX = "eld_"
"""What the standalone extension's settings were called."""

RENAMED = {
    "eld_enabled": OPT_ENABLED,
    "eld_library_dir": OPT_LIBRARY_DIR,
    "eld_hash_lookup": OPT_HASH_LOOKUP,
    "eld_hash_on_open": OPT_HASH_ON_OPEN,
    "eld_prompt_list_limit": OPT_PROMPT_CARDS,
    "eld_prompt_label_width": OPT_PROMPT_LABEL_WIDTH,
    "eld_image_dir": OPT_IMAGE_DIR,
    "eld_image_max_mb": OPT_IMAGE_MAX_MB,
    "eld_image_timeout": OPT_IMAGE_TIMEOUT,
    "eld_civitai_api_key": OPT_API_KEY,
    "eld_remember_api_key": OPT_REMEMBER_KEY,
    "eld_remember_scan_folder": OPT_REMEMBER_FOLDER,
    "eld_fetch_descriptions": OPT_FETCH_DESCRIPTIONS,
    "eld_backfill_text": OPT_BACKFILL,
    "eld_retry_unresolved": OPT_RETRY_UNRESOLVED,
    "eld_request_delay": OPT_REQUEST_DELAY,
    "eld_request_timeout": OPT_REQUEST_TIMEOUT,
    "eld_max_images": OPT_MAX_IMAGES,
    "eld_base_model_filter": OPT_BASE_MODEL,
    "eld_require_explicit_label": OPT_REQUIRE_LABEL,
    "eld_folder_filter_enabled": OPT_FILTER,
    "eld_folder_filter_dirs": OPT_FILTER_DIRS,
    "eld_folder_filter_checkpoint_menu": OPT_FILTER_CHECKPOINT_MENU,
    "eld_folder_filter_strict": OPT_FILTER_STRICT,
    "eld_folder_filter_auto_refresh": OPT_FILTER_AUTO_REFRESH,
}
"""The standalone's keys and theirs here. Its ``eld_panel_first``,
``eld_open_by_default`` and ``eld_open_on_match`` describe a panel that no
longer exists and are left behind on purpose."""

IMPORT_VERSION = 1

# What is copied from the standalone's root into the data root: each a
# (source relative to that root, destination relative to the data root).
CARRIED_FILES = (
    ("preset_folders.json", mc_lora_paths.FOLDERS_FILENAME),
    ("last_used.json", mc_lora_paths.LAST_USED_FILENAME),
    (os.path.join(".cache", "hashes.json"), os.path.join(mc_lora_paths.CACHE_DIRNAME, "hashes.json")),
)
CARRIED_FOLDERS = (
    ("library", mc_lora_paths.LIBRARY_DIRNAME),
    ("images", mc_lora_paths.IMAGES_DIRNAME),
)


# --------------------------------------------------------------- the sections


def _library_status_html() -> str:
    try:
        from mc_lora_library import get_library

        return mc_lora_render.library_summary_html(get_library().stats())
    except Exception:
        report("could not read the library status")
        return mc_lora_render.notice("The library could not be read — see the console for details.", kind="error")


def _uploader_component(**kwargs):
    """Factory used as ``OptionInfo.component``; runs inside the settings page.

    A real ``gr.File`` with its status line and two buttons, wired here. The
    option is marked ``do_not_save`` so the settings machinery treats it as a
    display-only widget and never persists a temporary path into config.json.
    """
    import gradio as gr

    from mc_lora_library import get_library

    # The settings builder passes the option's stored value; a File component
    # must not be seeded with one.
    kwargs.pop("value", None)
    label = kwargs.pop("label", "") or "Add a CSV to the library"
    elem_id = kwargs.pop("elem_id", None)

    with gr.Column(elem_classes=["mc-lora-settings-block"]):
        gr.HTML(
            mc_lora_render.notice(
                "Drop a CSV here to add it to the library. Rows are matched to a LoRA by "
                "<b>SHA256</b> when the CSV has a hash column, otherwise by file path or file name.",
                kind="info",
            )
        )
        uploader = gr.File(
            label=label,
            file_count="multiple",
            file_types=[".csv", "text/csv"],
            type="filepath",
            elem_id=elem_id,
            **kwargs,
        )
        status = gr.HTML(elem_classes=["mc-lora-upload-status"])
        with gr.Row():
            refresh = gr.Button("Reload library from disk", size="sm")
            open_note = gr.Button("Show library contents", size="sm")
        library_status = gr.HTML(value=_library_status_html, elem_classes=["mc-lora-library-status"])
        sources = gr.HTML(visible=False)

    def do_upload(paths):
        if not paths:
            return "", _library_status_html(), None

        if not isinstance(paths, (list, tuple)):
            paths = [paths]

        messages = []
        for entry in paths:
            path = getattr(entry, "name", entry)
            try:
                ok, message = get_library().import_csv(path)
            except Exception as exc:  # noqa: BLE001 - reported to the user
                report("could not import a CSV file")
                ok, message = False, f"{type(exc).__name__}: {exc}"
            messages.append(mc_lora_render.notice(mc_lora_render.escape(message), kind="ok" if ok else "error"))

        return "".join(messages), _library_status_html(), None

    uploader.upload(
        fn=do_upload,
        inputs=[uploader],
        outputs=[status, library_status, uploader],
        show_progress=True,
    )

    def do_refresh():
        try:
            get_library().reload()
            message = mc_lora_render.notice("Library reloaded.", kind="ok")
        except Exception as exc:  # noqa: BLE001
            report("could not reload the library")
            message = mc_lora_render.notice(mc_lora_render.escape(f"{type(exc).__name__}: {exc}"), kind="error")
        return message, _library_status_html()

    refresh.click(fn=do_refresh, outputs=[status, library_status], show_progress=False)

    def do_list():
        try:
            html = mc_lora_render.sources_html(get_library().sources)
        except Exception:
            report("could not list the library sources")
            html = mc_lora_render.notice("The library could not be listed.", kind="error")
        return gr.update(value=html, visible=True)

    open_note.click(fn=do_list, outputs=[sources], show_progress=False)

    # The settings page collects the returned component into its inputs list.
    return uploader


def _forget_key_when_off() -> None:
    """Turning remembering off is also how the stored key is deleted."""
    if not opt(OPT_REMEMBER_KEY, True):
        mc_lora_last_used.forget_api_key()


def _slider(gr):
    return getattr(gr, "Slider", None)


def settings(shared) -> dict:
    """*Model Chain: LoRA Details*, in the order the page draws it."""
    OptionInfo = shared.OptionInfo
    import gradio as gr  # noqa: F811 -- the host's, for the component classes

    uploader = OptionInfo([], "Add a CSV to the library", component=_uploader_component)
    uploader.do_not_save = True

    return {
        OPT_ENABLED: OptionInfo(
            True,
            "Draw a LoRA card's details dialog the width of the window",
        ).info(
            "The file, what Civitai knows about the LoRA and its gallery prompts side by side, "
            "with Forge's own fields in the first column. Off leaves Forge's dialog as it is. "
            "Needs a Reload UI"
        ),
        OPT_LIBRARY_DIR: OptionInfo(
            "",
            "Library directory",
        ).info(
            "Where the CSV files are read from, recursively; takes effect at once. Empty uses "
            "a model_chain_lora/library folder in the WebUI's data directory"
        ),
        OPT_UPLOAD: uploader,
        OPT_HASH_LOOKUP: OptionInfo(
            True,
            "Match a LoRA to its CSV row by SHA256 first",
        ).info("Falls back to the file's path, then its name, when no row carries the hash"),
        OPT_HASH_ON_OPEN: OptionInfo(
            True,
            "Compute a missing SHA256 when the dialog opens",
        ).info(
            "The first open of a large LoRA takes a moment; off, use Precompute hashes on the "
            "LoRA Details tab instead"
        ),
        OPT_PROMPT_CARDS: OptionInfo(
            50,
            "Prompt cards drawn in the dialog",
            _slider(gr),
            {"minimum": 5, "maximum": 500, "step": 5},
        ).info("The selector above the cards always reaches every prompt"),
        OPT_PROMPT_LABEL_WIDTH: OptionInfo(
            90,
            "Characters of each prompt shown in the selector",
            _slider(gr),
            {"minimum": 30, "maximum": 200, "step": 5},
        ).info("A MOSS layout prompt is named by its caption whatever its length"),
        OPT_IMAGE_DIR: OptionInfo(
            "",
            "Folder for downloaded gallery images",
        ).info(
            "One flat folder, so the collection is managed in one place. Empty uses "
            "model_chain_lora/images in the WebUI's data directory"
        ),
        OPT_IMAGE_MAX_MB: OptionInfo(
            32,
            "Largest gallery image to download (MB)",
            _slider(gr),
            {"minimum": 1, "maximum": 256, "step": 1},
        ).info("A larger one is refused rather than saved in part"),
        OPT_IMAGE_TIMEOUT: OptionInfo(
            30.0,
            "Gallery image download timeout (seconds)",
            _slider(gr),
            {"minimum": 5.0, "maximum": 180.0, "step": 5.0},
        ).info("Per image; the download is offered again afterwards"),
    }


def fetch_settings(shared) -> dict:
    """*Model Chain: Civitai fetch*."""
    OptionInfo = shared.OptionInfo
    import gradio as gr  # noqa: F811

    # Assigned rather than passed to the constructor, so the option still
    # registers on a build whose OptionInfo does not take ``onchange``.
    remember_key = OptionInfo(
        True,
        "Remember the Civitai API key that last worked",
    ).info(
        "Fills the fetch tab's key field with the key of the last fetch Civitai accepted; "
        "kept in the LoRA data folder, not in config.json. Turning this off forgets the stored key"
    )
    remember_key.onchange = _forget_key_when_off

    return {
        OPT_API_KEY: OptionInfo(
            "",
            "Civitai API key",
        ).info("Optional, but improves coverage; falls back to the CIVITAI_API_KEY environment variable"),
        OPT_REMEMBER_KEY: remember_key,
        OPT_REMEMBER_FOLDER: OptionInfo(
            True,
            "Open the fetch tab on the folder that was scanned last",
        ).info("Off, it opens on the first LoRA folder"),
        OPT_FETCH_DESCRIPTIONS: OptionInfo(
            True,
            "Fetch the model description",
        ).info(
            "Costs one request per model page; the page also names the creator. Gallery images "
            "are never downloaded by a fetch"
        ),
        OPT_BACKFILL: OptionInfo(
            True,
            "On a refresh, fill in text details missing from LoRAs already in the library",
        ).info("Picks up descriptions for rows written before they were collected"),
        OPT_RETRY_UNRESOLVED: OptionInfo(
            False,
            "On a refresh, look up LoRAs Civitai had nothing for last time",
        ).info("Off by default: a refresh is for new LoRAs, not for re-asking about known misses"),
        OPT_REQUEST_DELAY: OptionInfo(
            0.15,
            "Minimum delay between Civitai requests (seconds)",
            _slider(gr),
            {"minimum": 0.0, "maximum": 5.0, "step": 0.05},
        ).info("Civitai rate-limits; a longer pause costs time and nothing else"),
        OPT_REQUEST_TIMEOUT: OptionInfo(
            30.0,
            "Civitai request timeout (seconds)",
            _slider(gr),
            {"minimum": 5.0, "maximum": 180.0, "step": 5.0},
        ).info("Per request; a request is retried a few times before the LoRA is marked unresolved"),
        OPT_MAX_IMAGES: OptionInfo(
            0,
            "Gallery images inspected per version (0 = all)",
            _slider(gr),
            {"minimum": 0, "maximum": 500, "step": 10},
        ).info("Each inspected image's prompt is kept; the images themselves are not fetched"),
        OPT_BASE_MODEL: OptionInfo(
            "",
            "Only accept these base models",
        ).info(
            'One or several, comma-separated: "Krea 2", or "Krea 2, MOSS Image". Empty accepts any '
            "exact hash match"
        ),
        OPT_REQUIRE_LABEL: OptionInfo(
            False,
            "Require an explicit base-model label",
        ).info(
            "Rejects an exact hash match that carries no label; a MOSS-Image LoRA's own tensor "
            "names count as one"
        ),
    }


def networks_settings(shared) -> dict:
    """*Model Chain: Extra networks*."""
    OptionInfo = shared.OptionInfo

    return {
        OPT_FILTER: OptionInfo(
            True,
            "Filter the extra-network browsers by the folders assigned to the active UI Preset",
        ).info(
            "Assign them on the LoRA Details tab. Networks outside those folders are not listed, "
            "and searching cannot surface them either"
        ),
        OPT_FILTER_DIRS: OptionInfo(
            True,
            "Also hide the folder buttons of folders that are not assigned",
        ).info("The row of quick-navigation buttons above the cards"),
        OPT_TOP_LEVEL_BUTTONS: OptionInfo(
            True,
            "Show only top-level folders in the folder buttons",
        ).info(
            "A button for SDXL, none for SDXL/lighting: the subfolders' models are still listed "
            "under their parent. Off draws every folder's button, as Forge does"
        ),
        OPT_FILTER_CHECKPOINT_MENU: OptionInfo(
            False,
            "Also narrow Forge's own Checkpoint quick setting to the assigned folders",
        ).info("Uses the folders assigned to Checkpoints; the loaded checkpoint always stays listed"),
        OPT_FILTER_STRICT: OptionInfo(
            False,
            "Show nothing when the active preset has no folders assigned for a network type",
        ).info("Off, a network type with no folders assigned keeps showing everything"),
        OPT_FILTER_AUTO_REFRESH: OptionInfo(
            True,
            "Rebuild the network browsers when the UI Preset changes",
        ).info("Off, press the browser's own refresh after switching preset"),
    }


def hidden_settings(shared) -> dict:
    return {
        OPT_IMPORT: shared.OptionInfo("", "The record of the one-time carry-over from the standalone extension"),
    }


# ------------------------------------------------------------ the carry-over


def import_record(opts) -> dict | None:
    data = getattr(opts, "data", None)
    if not isinstance(data, dict):
        return None
    raw = data.get(OPT_IMPORT)
    if not raw:
        return None
    try:
        record = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return record if isinstance(record, dict) else None


def _copy_tree(source: Path, destination: Path) -> tuple[int, int]:
    """Copy every file under ``source`` that ``destination`` lacks. Returns
    ``(copied, kept)``; a file already there is never overwritten."""
    copied = kept = 0
    for folder, _dirs, names in os.walk(source):
        for name in names:
            path = Path(folder) / name
            target = destination / path.relative_to(source)
            if target.exists():
                kept += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            copied += 1
    return copied, kept


def carry_files(old_root: Path, data_root: Path) -> dict:
    """The standalone's files into the data root; what was copied, by kind."""
    done: dict = {}
    for name, destination_name in CARRIED_FOLDERS:
        source = old_root / name
        if source.is_dir():
            copied, kept = _copy_tree(source, data_root / destination_name)
            done[destination_name] = {"copied": copied, "kept": kept}
    for name, destination_name in CARRIED_FILES:
        source = old_root / name
        if source.is_file():
            target = data_root / destination_name
            if target.exists():
                done[destination_name] = {"copied": 0, "kept": 1}
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            # The last-used file holds an API key: owner-only, as it was.
            if destination_name == mc_lora_paths.LAST_USED_FILENAME:
                try:
                    os.chmod(target, 0o600)
                except OSError:
                    pass
            done[destination_name] = {"copied": 1, "kept": 0}
    return done


def carry_settings(opts, *, defaults: dict) -> tuple[dict, list]:
    """The standalone's ``eld_*`` values into ours, where ours are still at
    their defaults. Returns ``(written, kept)``."""
    data = getattr(opts, "data", None)
    if not isinstance(data, dict):
        return {}, []
    written: dict = {}
    kept: list = []
    for old, new in RENAMED.items():
        if old not in data:
            continue
        value = data[old]
        current = data.get(new)
        if current is not None and current != defaults.get(new):
            kept.append(new)
            continue
        if value == defaults.get(new):
            continue
        opts.set(new, value)
        written[new] = value
    return written, kept


def import_standalone(opts, extensions_dir, *, defaults: dict, data_root: Path | None = None,
                      save=None, now=None) -> dict | None:
    """Bring the standalone extension's files and settings over, once.

    Returns the record, or None when there was nothing to do: already done,
    or no standalone folder and no ``eld_*`` setting to read.
    """
    data = getattr(opts, "data", None)
    if not isinstance(data, dict):
        return None
    if data.get(OPT_IMPORT):
        return None

    old_root = mc_lora_paths.old_extension_root(extensions_dir)
    has_settings = any(key.startswith(STANDALONE_PREFIX) for key in data)
    if old_root is None and not has_settings:
        return None

    root = data_root if data_root is not None else mc_lora_paths.data_root()
    files: dict = {}
    if old_root is not None:
        try:
            files = carry_files(old_root, root)
        except OSError:
            report("could not copy the standalone extension's files into the LoRA data folder")
            return None

    written, kept = carry_settings(opts, defaults=defaults)

    record = {
        "version": IMPORT_VERSION,
        "from": str(old_root) if old_root is not None else "",
        "when": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now() if now else time.time())),
        "files": files,
        "applied": {name: value for name, value in written.items() if name != OPT_API_KEY},
        "kept": kept,
    }
    opts.set(OPT_IMPORT, json.dumps(record, ensure_ascii=False))
    if save is not None:
        try:
            save()
        except Exception:
            report("the carry-over could not be saved")

    copied = sum(entry.get("copied", 0) for entry in files.values())
    log(
        f"carried {copied} file(s) and {len(written)} setting(s) over from the Extended LoRA Details "
        f"extension ({len(kept)} setting(s) kept as you set them); that extension can be uninstalled"
    )
    return record


def on_app_started(_demo=None, _app=None) -> None:
    """Settings are loaded by now: the carry-over, then the library."""
    try:
        from modules import shared

        labels = getattr(shared.opts, "data_labels", {}) or {}
        defaults = {name: info.default for name, info in labels.items() if name in RENAMED.values()}
        import_standalone(
            shared.opts,
            mc_lora_paths.extensions_dir(),
            defaults=defaults,
            save=lambda: shared.opts.save(shared.config_filename),
        )
    except Exception:
        report("could not carry the standalone extension's files and settings over")
    try:
        if mc_lora_paths.old_extension_root() is not None:
            log(
                "the Extended LoRA Details extension is still installed; its own dialog panel and "
                "folder filter run beside these until it is uninstalled or disabled under Extensions"
            )
    except Exception:
        pass
    try:
        from mc_lora_library import get_library

        get_library().ensure_loaded()
    except Exception:
        report("could not load the CSV library")
