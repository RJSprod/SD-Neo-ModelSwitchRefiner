"""Filter the extra-network browsers down to the active UI Preset's folders.

Forge builds every network pane server-side: ``ExtraNetworksPage.create_html``
lists the items, renders the cards, the tree and - in *Dirs* view - the row of
quick-navigation folder buttons. There is no hook for narrowing any of that, so
this module wraps two methods on the **base** page class:

* ``create_html`` - the page's own ``list_items`` is shadowed for the duration of
  the call, so cards *and* the tree view are built from the assigned folders
  only. Filtering at the item level also keeps hidden networks out of the search
  index, so a search cannot surface a LoRA from a folder the preset does not own.
* ``create_dirs_view_html`` - the folder buttons are walked straight off disk by
  Forge, so the generated buttons are filtered afterwards: only assigned folders
  (and, where the assignment includes them, their subfolders) keep a button.
  The same wrapper keeps the row to the **top-level** folders -- the user's
  word: "I made all lora folders model names, like SDXL ... I have no need to
  ever tap on a subfolder" -- a button for ``SDXL``, none for ``SDXL/lighting``,
  whose models are still listed under their parent. A setting turns that off.

Patching the base class covers every page - Lora, Checkpoints, Textual Inversion
and any page another extension registers - regardless of load order. Everything
fails soft: if anything goes wrong the stock, unfiltered HTML is returned.
"""

from __future__ import annotations

import html as html_module
import os
import re

import mc_extra_networks_folders
from mc_lora_common import log, opt, report

MARKER = "_model_chain_lora_preset_folder_filter"
DROPDOWN_MARKER = "_model_chain_lora_preset_folder_filter_dropdown"
CHECKPOINT_PAGE = "checkpoints"

OPT_FILTER = "model_chain_lora_folder_filter_enabled"
OPT_FILTER_DIRS = "model_chain_lora_folder_filter_dirs"
OPT_FILTER_STRICT = "model_chain_lora_folder_filter_strict"
OPT_CHECKPOINT_MENU = "model_chain_lora_folder_filter_checkpoint_menu"
OPT_TOP_LEVEL_BUTTONS = "model_chain_lora_folder_buttons_top_level"
"""Registered in ``mc_lora_settings``; read here by name so that this module
imports nothing of Gradio's."""

# ``create_dirs_view_html`` emits one <button> per folder; the "all" button is the
# only one carrying the ``search-all`` class.
BUTTON_RE = re.compile(r"<button\b(?P<attrs>[^>]*)>(?P<label>.*?)</button>", re.DOTALL)
SEPARATOR_RE = re.compile(r"[\\/]+")


def active_rules(page):
    """Rules to apply to ``page``, or ``None`` when it must not be filtered.

    An empty list is meaningful: it means *hide everything* (strict mode with
    nothing assigned to the active preset).
    """
    if not opt(OPT_FILTER, True):
        return None

    preset = mc_extra_networks_folders.current_preset()
    if not preset:
        return None

    rules = mc_extra_networks_folders.get_store().rules(preset, page)
    if rules:
        return rules

    # Nothing assigned for this preset and this network type: show everything,
    # unless the user asked for the strict reading.
    return [] if opt(OPT_FILTER_STRICT, False) else None


def item_allowed(item, rules) -> bool:
    if not rules:
        return False
    if not isinstance(item, dict):
        return True
    return mc_extra_networks_folders.file_allowed(item.get("filename", ""), rules)


def _folder_parts(match: re.Match) -> list[str]:
    """The folder a button names, split at its separators: ``SDXL/lighting/``
    is ``["SDXL", "lighting"]``; the "all" button and an empty label are ``[]``."""
    if "search-all" in match.group("attrs"):
        return []  # the "all" button is navigation, not a folder
    label = html_module.unescape(match.group("label")).strip()
    return [part for part in SEPARATOR_RE.split(label) if part not in ("", ".")]


def top_level_dirs_html(markup: str) -> str:
    """Keep the buttons of the folders directly under a LoRA directory.

    A LoRA in ``SDXL/lighting`` is still listed when ``SDXL`` is pressed,
    because Forge's button searches by path prefix; what goes is the button
    the user never presses. The "all" button stays.
    """
    if not markup or not BUTTON_RE.search(markup):
        return markup

    def keep(match: re.Match) -> str:
        return match.group(0) if len(_folder_parts(match)) <= 1 else ""

    return BUTTON_RE.sub(keep, markup)


def filter_dirs_html(markup: str, roots, rules) -> str:
    """Drop the quick-navigation buttons for folders the preset does not own."""
    if not markup or not BUTTON_RE.search(markup):
        return markup

    bases = [os.path.abspath(os.path.expanduser(str(root))) for root in roots or () if str(root or "").strip()]

    def keep(match: re.Match) -> str:
        parts = _folder_parts(match)
        if not parts:
            return match.group(0)

        for base in bases:
            if mc_extra_networks_folders.matches(rules, os.path.join(base, *parts)):
                return match.group(0)
        return ""

    return BUTTON_RE.sub(keep, markup)


def _wrap_create_html(original):
    def create_html(self, tabname, *args, **kwargs):
        try:
            rules = active_rules(self)
        except Exception:
            report("could not read the preset folder assignments")
            rules = None

        if rules is None:
            return original(self, tabname, *args, **kwargs)

        stock_list_items = self.list_items

        def list_items():
            for item in stock_list_items():
                if item is None:
                    continue
                try:
                    if item_allowed(item, rules):
                        yield item
                except Exception:
                    report("could not test a network against the preset folders")
                    yield item

        # Shadow the bound method for this call only; ``create_html`` reads it
        # once, so cards, tree view and metadata all come from the same list.
        self.list_items = list_items
        try:
            return original(self, tabname, *args, **kwargs)
        finally:
            try:
                del self.list_items
            except AttributeError:
                pass

    return create_html


def _wrap_dirs_view(original):
    def create_dirs_view_html(self, *args, **kwargs):
        markup = original(self, *args, **kwargs)
        try:
            if opt(OPT_TOP_LEVEL_BUTTONS, True):
                markup = top_level_dirs_html(markup)
            if not opt(OPT_FILTER_DIRS, True):
                return markup
            rules = active_rules(self)
            if rules is None:
                return markup
            roots = self.allowed_directories_for_previews()
            return filter_dirs_html(markup, roots, rules)
        except Exception:
            report("could not filter the folder buttons")
            return markup

    return create_dirs_view_html


def _page_class():
    try:
        from modules.ui_extra_networks import ExtraNetworksPage  # type: ignore
    except Exception:
        return None
    return ExtraNetworksPage


def apply() -> bool:
    """Wrap the base page class. Idempotent across ``Reload UI``."""
    page_class = _page_class()
    if page_class is None:
        log("extra-network pages were not found; the preset folder filter is disabled")
        return False

    if getattr(page_class, MARKER, False):
        return True

    page_class.create_html = _wrap_create_html(page_class.create_html)
    if hasattr(page_class, "create_dirs_view_html"):
        page_class.create_dirs_view_html = _wrap_dirs_view(page_class.create_dirs_view_html)
    setattr(page_class, MARKER, True)
    log("preset folder filter attached to the extra-network pages")

    try:
        apply_checkpoint_menu()
    except Exception:
        report("could not attach the filter to the Checkpoint menu")

    return True


# ------------------------------------------- the Checkpoint quick-setting menu


def filter_checkpoint_tiles(tiles, rules) -> list[str]:
    """Keep the checkpoint titles whose file sits in an assigned folder.

    The quick-setting dropdown lists titles, not paths, so each one is matched
    back to its ``CheckpointInfo``. A title that cannot be resolved is kept - a
    checkpoint the filter cannot place must not become unloadable - and so is
    the one currently loaded.
    """
    from modules import sd_models, shared

    use_short = bool(getattr(shared.opts, "sd_checkpoint_dropdown_use_short", False))
    active = str(getattr(shared.opts, "sd_model_checkpoint", "") or "")

    known: dict[str, bool] = {}
    for info in sd_models.checkpoints_list.values():
        title = info.short_title if use_short else info.name
        known[title] = mc_extra_networks_folders.file_allowed(getattr(info, "filename", ""), rules)

    kept = []
    for title in tiles:
        if title == active or known.get(title, True):
            kept.append(title)
    return kept


def _wrap_refresh_models(original):
    def refresh_models(*args, **kwargs):
        result = original(*args, **kwargs)
        if not opt(OPT_CHECKPOINT_MENU, False):
            return result
        try:
            checkpoints, modules_list = result
            rules = active_rules(CHECKPOINT_PAGE)
            if rules is None:
                return result
            return filter_checkpoint_tiles(checkpoints, rules), modules_list
        except Exception:
            # Whatever the shape of the result, the menu keeps working.
            report("could not filter the Checkpoint menu")
            return result

    return refresh_models


def apply_checkpoint_menu() -> bool:
    """Narrow Forge's own *Checkpoint* quick setting as well (opt-in)."""
    try:
        from modules_forge import main_entry  # type: ignore
    except Exception:
        return False

    if getattr(main_entry, DROPDOWN_MARKER, False):
        return True

    main_entry.refresh_models = _wrap_refresh_models(main_entry.refresh_models)
    setattr(main_entry, DROPDOWN_MARKER, True)
    return True


def on_before_ui() -> None:
    """The host's hook before it builds the page, which is when Forge renders
    the network pages: the wrappers go on first. Never raises into the host."""
    try:
        apply()
    except Exception:
        report("could not attach the preset folder filter")
