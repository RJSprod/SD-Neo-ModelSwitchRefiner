"""The LoRA dialog: Forge's *edit metadata* popup for a LoRA card, redrawn as
one page the width of the window.

Forge builds the dialog in ``LoraUserMetadataEditor.create_editor()``: a row
with the card's name, description, file table and preset on the left and the
preview on the right, then the training tags, the activation text, the weight,
the negative prompt, the random prompt and the notes, then
``create_default_buttons()`` -- Cancel, Replace preview, Save -- all in one
column 56 em wide. The standalone extension added an accordion of Civitai
details under the notes, and the user's word on the result was that it "still
renders everything in a narrow column in the middle of the page".

So :func:`install` wraps ``create_default_buttons`` -- the one moment Forge
gives an extension inside the dialog, with every stock component built and the
dialog's ``Group`` still open -- and :func:`build_dialog` lays the whole thing
out again, Forge's components included:

* a head row: the card's name, and when the library knows the LoRA, Civitai's
  name for it with its version, base model and creator and a link to the page;
* three columns under it -- **the file** (the preview, Forge's file table, and
  every field you edit: description, preset, training tags, activation text,
  weight, negative prompt, random prompt, notes), **what Civitai says** (the
  facts, trigger-word chips, the description and version notes, the negative
  prompt, every field of the CSV row and the raw row folded away; or, for a
  LoRA the library has nothing for, the fetch button), and **the gallery
  prompts** (a selector with stepping, the selected prompt large beside its
  image, and every prompt as a card), which is not drawn at all when there are
  none;
* Forge's own buttons, last, across the whole width.

Nothing of Forge's is rebuilt: its components are *moved* -- taken out of the
row Forge put them in and appended to a column of ours, the way Gradio itself
places a component when a ``with`` block closes -- so every one of them keeps
its id, its events and its place in Forge's ``put_values_into_components``
and save handlers. The move is a list operation on Gradio's block tree, made
before the page's layout is read; should any of it fail, the dialog is left
as Forge built it, with the details appended under the notes as before.

Population is wired as a **second** listener on the same ``button_edit``
click that Forge already uses, so Forge's outputs are untouched: if anything
in here raises, the stock dialog still works.

Carried in from the user's ``SD-ForgeNeo-ExtendedLoraDetails`` extension
(2026-10-23); ``docs/26-lora-details.md`` is the record.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import gradio as gr

import mc_lora_civitai
import mc_lora_hashing
import mc_lora_images
import mc_lora_jobs
import mc_lora_last_used
import mc_lora_layouts
import mc_lora_render
from mc_lora_common import SAFETENSORS_EXTENSIONS, log, lora_directories, normalize_path, opt, report
from mc_lora_library import get_library

OPT_ENABLED = "model_chain_lora_enabled"
"""Draw the LoRA dialog this way at all. Off, Forge's dialog is as Forge built it."""

MARKER = "_model_chain_lora_details"
DIALOG_CLASS = "mc-lora-dialog"
"""On the dialog's own ``Group``: the stylesheet widens the popup under it."""

# The stock components, by the attribute Forge keeps them under, in the order
# the file column draws them. Anything Forge adds that is not named here is
# moved after them in the order Forge made it, so nothing of the dialog's is
# lost when Forge grows a field.
FILE_INFO = ("html_preview", "html_filedata")
YOUR_SETTINGS = (
    "edit_description",
    "select_sd_version",
    "taginfo",
    "edit_activation_text",
    "slider_preferred_weight",
    "edit_negative_text",
    "edit_notes",
)


# ------------------------------------------------------------------ gathering


def _relative_candidates(filename: str) -> list[str]:
    """Every path form a CSV might have used for this file."""
    if not filename:
        return []

    absolute = os.path.abspath(filename)
    candidates: list[str] = []

    for directory in lora_directories():
        try:
            if os.path.commonpath([absolute, directory]) == directory:
                candidates.append(os.path.relpath(absolute, directory))
        except ValueError:
            continue

    parent = os.path.dirname(absolute)
    candidates.append(os.path.basename(absolute))
    if parent:
        candidates.append(os.path.join(os.path.basename(parent), os.path.basename(absolute)))
    candidates.append(absolute)

    seen: list[str] = []
    for candidate in candidates:
        key = normalize_path(candidate)
        if key and key not in seen:
            seen.append(key)
    return seen


def _short_hash_candidates(item: dict) -> list[str]:
    """Cheap hashes already known to Forge - no file reads involved."""
    metadata = item.get("metadata") or {}
    candidates = [
        item.get("shorthash") or "",
        metadata.get("sshs_model_hash") or "",
        metadata.get("sshs_legacy_hash") or "",
    ]

    filename = item.get("filename") or ""
    if filename:
        cached_addnet = mc_lora_hashing.CACHE.peek(filename, "addnet")
        if cached_addnet:
            candidates.append(cached_addnet)

    return [candidate for candidate in candidates if candidate]


def _resolve_sha256(filename: str) -> tuple[str, bool]:
    """Return ``(sha256, computed_now)``; honours the two hashing settings."""
    if not filename or not opt("model_chain_lora_hash_lookup", True):
        return "", False

    cached = mc_lora_hashing.known_hash(filename)
    if cached:
        return cached, False

    if not opt("model_chain_lora_hash_on_open", True):
        return "", False

    value = mc_lora_hashing.sha256_file(filename) or ""
    if value:
        mc_lora_hashing.flush()
    return value, bool(value)


_ARCHITECTURES: dict[tuple[str, int, int], str] = {}


def _architecture(filename: str) -> str:
    """What the file's tensor names say it is - read from its header only."""
    if not filename or not filename.lower().endswith(SAFETENSORS_EXTENSIONS):
        return ""
    try:
        info = os.stat(filename)
    except OSError:
        return ""
    key = (os.path.abspath(filename), info.st_size, info.st_mtime_ns)
    if key not in _ARCHITECTURES:
        if len(_ARCHITECTURES) > 4096:
            _ARCHITECTURES.clear()
        _metadata, names = mc_lora_civitai.read_safetensors_header(Path(filename))
        _ARCHITECTURES[key] = mc_lora_civitai.local_architecture(names)
    return _ARCHITECTURES[key]


def _fetch_plan(filename: str, records) -> dict:
    """What the dialog's fetch button would do, or ``{}`` when it has nothing to do.

    A LoRA the library has never seen is fetched with its folder; one Civitai
    could not resolve before is retried. A resolved one needs no button.
    """
    folder = os.path.dirname(os.path.abspath(filename)) if filename else ""
    if not folder or not os.path.isdir(folder):
        return {}
    if records and any(record.resolved for record in records):
        return {}

    label = os.path.basename(folder) or folder
    for root in lora_directories():
        try:
            if os.path.commonpath([folder, root]) == root:
                relative = os.path.relpath(folder, root)
                label = os.path.basename(root) if relative == "." else relative.replace(os.sep, "/")
                break
        except ValueError:
            continue
    return {"folder": folder, "label": label + "/", "retry": bool(records)}


def gather(page, name: str) -> dict:
    """Everything the dialog needs for one card, as plain data."""
    item = (page.items.get(name, {}) if page is not None else {}) or {}
    filename = item.get("filename") or ""

    sha256, computed = _resolve_sha256(filename)
    short_hashes = _short_hash_candidates(item)
    rel_paths = _relative_candidates(filename)

    library = get_library()
    records, how = library.lookup(
        sha256=sha256,
        short_hashes=short_hashes,
        filename=filename,
        rel_paths=rel_paths,
        name=name,
    )

    prompts: list[str] = []
    prompt_images: list[tuple[str, str]] = []
    prompt_sources: list[str] = []
    triggers: list[tuple[str, str]] = []
    descriptions: list[tuple[str, str, str, str]] = []
    negatives: list[str] = []
    seen_triggers: set[str] = set()
    for record in records:
        if record.has_description:
            descriptions.append((record.title, record.source, record.description, record.version_description))
        for entry in record.entries:
            if entry.text and entry.text not in prompts:
                prompts.append(entry.text)
                prompt_images.append((entry.image_url, entry.image_id))
                prompt_sources.append(record.source)
        for word in record.trigger_words:
            if word not in seen_triggers:
                seen_triggers.add(word)
                triggers.append((word, record.source))
        for negative in record.negative_prompts:
            if negative not in negatives:
                negatives.append(negative)

    identity = {
        "SHA256": sha256 or ("not computed" if filename else ""),
        "Forge hash": item.get("shorthash") or "",
    }

    user_metadata = item.get("user_metadata") or {}

    return {
        "records": records,
        "how": how,
        "descriptions": descriptions,
        "prompts": prompts,
        "prompt_images": prompt_images,
        "prompt_sources": prompt_sources,
        "triggers": triggers,
        "negatives": negatives,
        "identity": identity,
        "computed": computed,
        "filename": filename,
        "architecture": _architecture(filename),
        "activation": str(user_metadata.get("activation text", "") or ""),
        "fetch": _fetch_plan(filename, records),
    }


# ------------------------------------------------------------------- helpers


def _prompt_choices(prompts: list[str]) -> list[tuple[str, int]]:
    width = max(1, int(opt("model_chain_lora_prompt_label_width", 90)))
    return [(f"{index + 1}. {mc_lora_layouts.label(prompt, width)}", index) for index, prompt in enumerate(prompts)]


def _clamp(index, total: int) -> int:
    try:
        value = int(index)
    except (TypeError, ValueError):
        value = 0
    if total <= 0:
        return 0
    return max(0, min(total - 1, value))


def _image_state(prompt_images: list, index: int):
    """Status HTML plus the preview update for the prompt at ``index``.

    Presence is a filesystem check, so an image the user deleted goes straight
    back to being offered as a download.
    """
    url, image_id = ("", "")
    if 0 <= index < len(prompt_images):
        url, image_id = prompt_images[index]

    if not url:
        return (
            mc_lora_render.notice("This prompt has no image URL in the library. Re-run a Civitai fetch to pick one up.", kind="empty"),
            gr.update(value=None, visible=False),
            gr.update(interactive=False, value="Download this image"),
        )

    local = mc_lora_images.find_local(url, image_id)
    if local is not None:
        done = gr.update(interactive=False, value="Downloaded", variant="secondary")
        if mc_lora_images.is_video(local):
            return (
                mc_lora_render.image_status_html(downloaded=True, name=local.name, url=url),
                gr.update(value=None, visible=False),
                done,
            )
        return (
            mc_lora_render.image_status_html(downloaded=True, name=local.name, url=url),
            gr.update(value=str(local), visible=True),
            done,
        )

    return (
        mc_lora_render.image_status_html(downloaded=False, name="", url=url),
        gr.update(value=None, visible=False),
        gr.update(interactive=True, value="Download this image", variant="primary"),
    )


def _cards(prompts: list[str], prompt_images: list) -> tuple[list[dict], int]:
    """One entry per card shown, plus how many prompts have their image saved."""
    limit = max(1, int(opt("model_chain_lora_prompt_list_limit", 50)))
    cards: list[dict] = []
    saved = 0
    for index, prompt in enumerate(prompts):
        url, image_id = prompt_images[index] if index < len(prompt_images) else ("", "")
        local = mc_lora_images.find_local(url, image_id) if url else None
        if local is not None:
            saved += 1
        if index >= limit:
            continue
        video = local is not None and mc_lora_images.is_video(local)
        cards.append(
            {
                "index": index,
                "prompt": prompt,
                "url": url,
                "saved": local is not None,
                "video": video,
                "thumb": mc_lora_images.thumbnail_data_uri(local) if local is not None and not video else "",
            }
        )
    return cards, saved


def _gallery(prompts: list[str], prompt_images: list, *, tabname: str, bridge: str) -> tuple[str, str]:
    """The gallery's heading and its cards, both of which change on a download."""
    cards, saved = _cards(prompts, prompt_images)
    return (
        mc_lora_render.prompts_head_html(len(prompts), saved),
        mc_lora_render.prompt_cards_html(cards, total=len(prompts), tab=tabname, bridge=bridge),
    )


def _source_for(sources: list, index: int) -> str:
    """Which CSV a prompt came from - worth saying only when there are several."""
    if len(set(sources or [])) <= 1 or not (0 <= index < len(sources)):
        return ""
    return f"from {sources[index]}"


def _raw_json(records) -> str:
    if not records:
        return "{}"
    payload = [
        {
            "source": record.source,
            "row": record.row_number,
            "fields": record.raw,
        }
        for record in records
    ]
    try:
        return json.dumps(payload if len(payload) > 1 else payload[0], indent=2, ensure_ascii=False)
    except (TypeError, ValueError):
        return "{}"


def _activation_target(editor) -> str:
    """The element id of the dialog's activation text box, giving it one if it
    has none, so the trigger-word chips can edit it from the page."""
    activation = getattr(editor, "edit_activation_text", None)
    if activation is None:
        return ""
    if not getattr(activation, "elem_id", None):
        activation.elem_id = f"{getattr(editor, 'id_part', 'mc_lora')}_activation_text"
    return str(activation.elem_id)


def _api_key() -> str:
    """The key the dialog's fetch uses: the remembered one, the setting, the
    environment - the same order the fetch tab fills its field in."""
    return (
        mc_lora_last_used.remembered_api_key()
        or str(opt("model_chain_lora_civitai_api_key", "") or "").strip()
        or os.environ.get("CIVITAI_API_KEY", "")
    )


# ------------------------------------------------------------------ the move
#
# Gradio keeps the page's layout as a tree of ``children`` lists, read when the
# page's configuration is built (``Blocks.get_config_file``); a component's
# ``parent`` is the context it was made in. Moving one is taking it out of one
# list and putting it in another, with ``parent`` to match -- the same two
# writes Gradio's own ``with`` blocks make. A text box Gradio has already
# wrapped in a pseudo ``Form`` (``fill_expected_parents``, at the row's close)
# is moved out of that form, and a form left empty is dropped with it.


def _children(block) -> list:
    found = getattr(block, "children", None)
    return found if isinstance(found, list) else []


def _descendants(block) -> list:
    out = []
    for child in _children(block):
        out.append(child)
        out.extend(_descendants(child))
    return out


def _detach(block) -> None:
    parent = getattr(block, "parent", None)
    siblings = _children(parent)
    if block in siblings:
        siblings.remove(block)
    # A pseudo form Gradio made for this box alone is nothing without it.
    if parent is not None and not siblings and type(parent).__name__ == "Form":
        _detach(parent)


def _move(block, into) -> None:
    _detach(block)
    block.parent = into
    _children(into).append(block)


def _empty(block) -> bool:
    """A layout block with no component left anywhere under it."""
    return not any(not isinstance(child, gr.blocks.BlockContext) for child in _descendants(block))


def _stock_row(editor):
    """The row Forge drew the name, description, file table and preview in."""
    node = getattr(editor, "edit_name", None)
    box = getattr(editor, "box", None)
    while node is not None and getattr(node, "parent", None) is not box:
        node = getattr(node, "parent", None)
    return node


def _lay_out(editor, *, head, file_column, before: list, status_after: list) -> bool:
    """Put Forge's components where this layout wants them. True when done;
    False leaves the dialog exactly as Forge built it, with ours under it."""
    box = getattr(editor, "box", None)
    if not isinstance(_children(box), list):
        return False
    row = _stock_row(editor)
    if row is None or row not in _children(box):
        return False

    name = getattr(editor, "edit_name", None)
    if name is not None:
        _move(name, head)
        # The name first, before our title strip.
        head.children.remove(name)
        head.children.insert(0, name)

    for attribute in FILE_INFO:
        component = getattr(editor, attribute, None)
        if component is not None:
            _move(component, file_column)

    # Your settings: Forge's editable fields, in its order; then whatever else
    # Forge (or another extension) put in the dialog before this ran -- the
    # random prompt row among them -- in the order it came.
    named = []
    for attribute in YOUR_SETTINGS:
        component = getattr(editor, attribute, None)
        if component is not None:
            named.append(component)
    for component in named:
        _move(component, file_column)

    skip = set(status_after) | {row, head}
    for block in list(before):
        if block in skip or block in named or getattr(block, "visible", True) is False:
            continue
        if block not in _children(box):
            continue
        # The hidden name box and the hidden button Forge opens the dialog with
        # stay where they are; a form that holds nothing but them too.
        if all(getattr(leaf, "visible", True) is False for leaf in _descendants(block) + [block]
               if not isinstance(leaf, gr.blocks.BlockContext)) and _descendants(block):
            continue
        _move(block, file_column)

    if _empty(row):
        _detach(row)

    # The moved boxes get their forms back, as Gradio would have grouped them.
    for column in (head, file_column):
        try:
            column.fill_expected_parents()
        except Exception:
            report("could not regroup the dialog's fields; they are drawn without their forms")
    return True


def _tag(block, css_class: str) -> None:
    classes = getattr(block, "elem_classes", None) or []
    if isinstance(classes, str):
        classes = [classes]
    if css_class not in classes:
        block.elem_classes = [*classes, css_class]


# ---------------------------------------------------------------- the dialog


def build_dialog(editor) -> None:
    """Lay the dialog out and wire the details into ``editor.button_edit``."""
    tabname = getattr(editor, "tabname", "txt2img")
    id_part = getattr(editor, "id_part", f"{tabname}_lora_edit_user_metadata")
    bridge = f"{id_part}_mc_lora_pick"
    target = _activation_target(editor)
    box = getattr(editor, "box", None)
    before = list(_children(box))

    prompts_state = gr.State([])
    images_state = gr.State([])
    sources_state = gr.State([])
    fetch_state = gr.State({})

    with gr.Row(equal_height=False, elem_classes=["mc-lora-head-row"]) as head:
        header = gr.HTML(elem_classes=["mc-lora-header"])

    with gr.Row(equal_height=False, elem_classes=["mc-lora-body"]):
        # ------------------------------------------------------- the file
        with gr.Column(scale=3, min_width=280, elem_classes=["mc-lora-col", "mc-lora-col-file"]) as file_column:
            gr.HTML('<div class="mc-lora-col-title">File</div>', elem_classes=["mc-lora-col-head"])

        # --------------------------------------------------- what Civitai says
        with gr.Column(scale=4, min_width=320, elem_classes=["mc-lora-col", "mc-lora-col-about"]):
            about_title = gr.HTML(elem_classes=["mc-lora-col-head"])
            facts = gr.HTML(elem_classes=["mc-lora-facts-html"])
            with gr.Column(visible=False, elem_classes=["mc-lora-fetch"]) as fetch_group:
                fetch_button = gr.Button("Fetch from Civitai", size="sm", variant="primary")
                fetch_note = gr.HTML(elem_classes=["mc-lora-fetch-note"])
            side = gr.HTML(elem_classes=["mc-lora-side-html"])

        # ------------------------------------------------ the gallery prompts
        with gr.Column(scale=6, min_width=480, visible=False, elem_classes=["mc-lora-col", "mc-lora-col-gallery"]) as main:
            prompts_head = gr.HTML(elem_classes=["mc-lora-prompts-head"])
            with gr.Row(equal_height=True, elem_classes=["mc-lora-picker"]):
                prompt_selector = gr.Dropdown(
                    label="Prompt",
                    show_label=False,
                    container=False,
                    choices=[],
                    value=None,
                    interactive=True,
                    filterable=True,
                    scale=10,
                    min_width=160,
                    elem_classes=["mc-lora-selector"],
                )
                previous_button = gr.Button("‹", size="sm", scale=0, min_width=48, elem_classes=["mc-lora-step"])
                next_button = gr.Button("›", size="sm", scale=0, min_width=48, elem_classes=["mc-lora-step"])

            with gr.Row(equal_height=False, elem_classes=["mc-lora-viewer"]):
                with gr.Column(scale=4, min_width=220, elem_classes=["mc-lora-viewer-media"]):
                    image_preview = gr.Image(
                        label="Gallery image",
                        show_label=False,
                        visible=False,
                        interactive=False,
                        show_download_button=True,
                        elem_classes=["mc-lora-image-preview"],
                    )
                    image_status = gr.HTML(elem_classes=["mc-lora-image-status"])
                    with gr.Row(elem_classes=["mc-lora-actions"]):
                        download_image = gr.Button("Download this image", size="sm", variant="primary")
                        download_all_images = gr.Button("Download all images", size="sm")

                with gr.Column(scale=5, min_width=260, elem_classes=["mc-lora-viewer-text"]):
                    prompt_meta = gr.HTML(elem_classes=["mc-lora-prompt-meta"])
                    prompt_box = gr.Textbox(
                        label="Positive prompt",
                        show_label=False,
                        lines=7,
                        max_lines=18,
                        interactive=False,
                        show_copy_button=True,
                        elem_classes=["mc-lora-prompt-box"],
                    )
                    with gr.Row(elem_classes=["mc-lora-actions"]):
                        append_prompt = gr.Button("Append to prompt", size="sm")
                        replace_prompt = gr.Button("Replace prompt", size="sm", variant="primary")

            cards = gr.HTML(elem_classes=["mc-lora-cards-html"])

    # The cards' "view" and "download" reach Python through these two, the
    # same way Forge's own cards open this dialog.
    pick_text = gr.Textbox(visible=False, elem_id=bridge)
    pick_button = gr.Button(visible=False, elem_id=f"{bridge}_button")

    # ------------------------------------------------------------- populate

    outputs = [
        header,
        about_title,
        facts,
        fetch_group,
        fetch_button,
        fetch_note,
        side,
        main,
        prompts_head,
        prompt_selector,
        prompt_meta,
        prompt_box,
        image_status,
        image_preview,
        download_image,
        download_all_images,
        cards,
        prompts_state,
        images_state,
        sources_state,
        fetch_state,
    ]

    def blank(message: str = "") -> list:
        return [
            message,
            mc_lora_render.column_title_html("Civitai"),
            "",
            gr.update(visible=False),
            gr.update(),
            "",
            "",
            gr.update(visible=False),
            "",
            gr.update(choices=[], value=None),
            "",
            "",
            "",
            gr.update(value=None, visible=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            "",
            [],
            [],
            [],
            {},
        ]

    def populate(name):
        try:
            data = gather(editor.page, name)
        except Exception:
            report("could not gather the details of a LoRA")
            return blank(mc_lora_render.notice("The LoRA's details could not be loaded — see the console for details.", kind="error"))

        records = data["records"]
        prompts = data["prompts"]
        triggers = data["triggers"]
        prompt_images = data["prompt_images"]
        sources = data["prompt_sources"]
        plan = data["fetch"]

        active = {piece.strip() for piece in data["activation"].split(",") if piece.strip()}
        texts = mc_lora_render.descriptions_html(data["descriptions"])
        if not texts and any(record.resolved for record in records):
            texts = mc_lora_render.notice(
                "No description in the library yet. A fetch with <b>fill in missing text details</b> ticked "
                "picks it up.",
                kind="empty",
            )
        side_html = "".join(
            [
                mc_lora_render.triggers_html([word for word, _ in triggers], target=target, active=active),
                texts,
                mc_lora_render.negative_html(data["negatives"], tab=tabname),
                mc_lora_render.fields_html(records),
                mc_lora_render.raw_html(_raw_json(records)),
            ]
        )

        if plan:
            fetch_label = "Retry on Civitai" if plan["retry"] else "Fetch from Civitai"
            what = (
                f"Looks this LoRA up again, with the others in <code>{mc_lora_render.escape(plan['label'])}</code> "
                "Civitai could not resolve before."
                if plan["retry"]
                else f"Looks up the LoRAs in <code>{mc_lora_render.escape(plan['label'])}</code> the library does not have "
                "yet, this one included."
            )
            note = f'<div class="mc-lora-muted">{what} It runs in the background; reopen this dialog when it is done.</div>'
        else:
            fetch_label, note = "Fetch from Civitai", ""

        choices = _prompt_choices(prompts)
        if prompts:
            status, preview, download_button = _image_state(prompt_images, 0)
        else:
            status, preview, download_button = "", gr.update(value=None, visible=False), gr.update(interactive=False)
        downloadable = sum(1 for url, _ in prompt_images if url)
        bulk = gr.update(
            interactive=bool(downloadable),
            value=f"Download all {downloadable} images" if downloadable else "No images in the library",
        )
        head_html, gallery = _gallery(prompts, prompt_images, tabname=tabname, bridge=bridge)

        pieces = []
        if triggers:
            pieces.append(f"{len(triggers)} trigger word{'s' if len(triggers) != 1 else ''}")
        if prompts:
            pieces.append(f"{len(prompts)} prompt{'s' if len(prompts) != 1 else ''}")
        if data["descriptions"]:
            pieces.append("description")

        return [
            mc_lora_render.header_html(records=records, how=data["how"]),
            mc_lora_render.column_title_html("Civitai", ", ".join(pieces) if records else "nothing in the library"),
            mc_lora_render.facts_html(records, identity=data["identity"], architecture=data["architecture"]),
            gr.update(visible=bool(plan)),
            gr.update(value=fetch_label, interactive=bool(plan)),
            note,
            side_html,
            gr.update(visible=bool(prompts)),
            head_html,
            gr.update(choices=choices, value=0 if choices else None),
            mc_lora_render.prompt_meta_html(len(prompts), 0, _source_for(sources, 0), mc_lora_layouts.parse(prompts[0]) if prompts else None),
            prompts[0] if prompts else "",
            status,
            preview,
            download_button,
            bulk,
            gallery,
            prompts,
            prompt_images,
            sources,
            plan,
        ]

    # A separate listener on the existing trigger: the stock dialog keeps working
    # even if this one raises.
    editor.button_edit.click(fn=populate, inputs=[editor.edit_name_input], outputs=outputs, show_progress=False)

    # ------------------------------------------------------------ navigation

    def viewer(prompts, prompt_images, sources, position):
        status, preview, button = _image_state(prompt_images or [], position)
        text = prompts[position] if prompts else ""
        meta = mc_lora_render.prompt_meta_html(len(prompts), position, _source_for(sources, position), mc_lora_layouts.parse(text) if text else None)
        return text, meta, status, preview, button

    def select_prompt(index, prompts, prompt_images, sources):
        prompts = prompts or []
        return viewer(prompts, prompt_images, sources, _clamp(index, len(prompts)))

    navigation_outputs = [prompt_box, prompt_meta, image_status, image_preview, download_image]

    prompt_selector.change(
        fn=select_prompt,
        inputs=[prompt_selector, prompts_state, images_state, sources_state],
        outputs=navigation_outputs,
        show_progress=False,
    )

    def step(index, prompts, prompt_images, sources, delta):
        prompts = prompts or []
        if not prompts:
            return (gr.update(), "", mc_lora_render.prompt_meta_html(0, 0), "", gr.update(visible=False), gr.update(interactive=False))
        position = (_clamp(index, len(prompts)) + delta) % len(prompts)
        return (position, *viewer(prompts, prompt_images, sources, position))

    previous_button.click(
        fn=lambda index, prompts, prompt_images, sources: step(index, prompts, prompt_images, sources, -1),
        inputs=[prompt_selector, prompts_state, images_state, sources_state],
        outputs=[prompt_selector, *navigation_outputs],
        show_progress=False,
    )
    next_button.click(
        fn=lambda index, prompts, prompt_images, sources: step(index, prompts, prompt_images, sources, 1),
        inputs=[prompt_selector, prompts_state, images_state, sources_state],
        outputs=[prompt_selector, *navigation_outputs],
        show_progress=False,
    )

    # ------------------------------------------------------- image downloads

    def download_at(position, prompt_images):
        """Download one prompt's image; returns an error notice, or ``""``."""
        url, image_id = prompt_images[position]
        try:
            result = mc_lora_images.download(url, image_id)
        except Exception as exc:  # noqa: BLE001 - surfaced next to the button
            report("could not download a gallery image")
            return mc_lora_render.notice(mc_lora_render.escape(f"{type(exc).__name__}: {exc}"), kind="error")
        return "" if result.ok else mc_lora_render.notice(mc_lora_render.escape(result.message), kind="error")

    def fetch_current_image(index, prompts, prompt_images):
        prompts = prompts or []
        prompt_images = prompt_images or []
        position = _clamp(index, len(prompt_images))
        if not (0 <= position < len(prompt_images)):
            return "", gr.update(visible=False), gr.update(interactive=False), gr.update(), gr.update()

        problem = download_at(position, prompt_images)
        status, preview, button = _image_state(prompt_images, position)
        head_html, gallery = _gallery(prompts, prompt_images, tabname=tabname, bridge=bridge)
        return problem + status, preview, button, head_html, gallery

    download_image.click(
        fn=fetch_current_image,
        inputs=[prompt_selector, prompts_state, images_state],
        outputs=[image_status, image_preview, download_image, prompts_head, cards],
    )

    def fetch_all_images(prompt_images, name):
        pairs = [(url, image_id) for url, image_id in (prompt_images or []) if url]
        if not pairs:
            return mc_lora_render.notice("There are no image URLs to download.", kind="empty")

        try:
            job = mc_lora_jobs.submit_image_downloads(pairs, label=str(name or ""))
        except Exception as exc:  # noqa: BLE001
            report("could not queue the image downloads")
            return mc_lora_render.notice(mc_lora_render.escape(f"{type(exc).__name__}: {exc}"), kind="error")

        return mc_lora_render.notice(
            f"Queued job #{job.id} for {len(pairs)} image(s). "
            "Progress is on the <b>LoRA Details</b> tab; reopen this dialog to see them.",
            kind="ok",
        )

    download_all_images.click(
        fn=fetch_all_images,
        inputs=[images_state, editor.edit_name_input],
        outputs=[image_status],
        show_progress=False,
    )

    # ------------------------------------------------- the cards' actions

    def pick(value, selected, prompts, prompt_images):
        """``view:<n>`` shows prompt n above; ``download:<n>`` saves its image."""
        prompts = prompts or []
        prompt_images = prompt_images or []
        nothing = (gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update())
        action, _, raw_index = str(value or "").partition(":")
        try:
            position = int(raw_index)
        except ValueError:
            return nothing
        if not (0 <= position < len(prompts)):
            return nothing

        if action == "view":
            return (position, gr.update(), gr.update(), gr.update(), gr.update(), gr.update())

        if action == "download" and position < len(prompt_images):
            problem = download_at(position, prompt_images)
            head_html, gallery = _gallery(prompts, prompt_images, tabname=tabname, bridge=bridge)
            if _clamp(selected, len(prompts)) == position:
                status, preview, button = _image_state(prompt_images, position)
                return (gr.update(), problem + status, preview, button, head_html, gallery)
            note = problem or gr.update()
            return (gr.update(), note, gr.update(), gr.update(), head_html, gallery)

        return nothing

    pick_button.click(
        fn=pick,
        inputs=[pick_text, prompt_selector, prompts_state, images_state],
        outputs=[prompt_selector, image_status, image_preview, download_image, prompts_head, cards],
        show_progress=False,
    )

    # ------------------------------------------------------ fetch from here

    def fetch_folder(plan):
        plan = plan or {}
        folder = plan.get("folder") or ""
        if not folder or not os.path.isdir(folder):
            return mc_lora_render.notice("This LoRA's folder could not be found.", kind="error"), gr.update()

        key = _api_key()
        try:
            job = mc_lora_jobs.submit_scan(
                folder,
                api_key=key,
                recursive=False,
                incremental=True,
                retry_unresolved=bool(plan.get("retry")) or bool(opt("model_chain_lora_retry_unresolved", False)),
                backfill_text=bool(opt("model_chain_lora_backfill_text", True)),
                fetch_descriptions=bool(opt("model_chain_lora_fetch_descriptions", True)),
                base_model_filter=str(opt("model_chain_lora_base_model_filter", "") or "").strip(),
                require_explicit_label=bool(opt("model_chain_lora_require_explicit_label", False)),
                request_delay=float(opt("model_chain_lora_request_delay", 0.15)),
                timeout=float(opt("model_chain_lora_request_timeout", 30.0)),
                max_images=int(opt("model_chain_lora_max_images", 0)),
                keep_existing_rows=True,
            )
        except Exception as exc:  # noqa: BLE001
            report("could not queue the fetch")
            return mc_lora_render.notice(mc_lora_render.escape(f"{type(exc).__name__}: {exc}"), kind="error"), gr.update()

        key_note = "" if key else " No API key is set, so only public models are found."
        message = mc_lora_render.notice(
            f"Queued job #{job.id} for <code>{mc_lora_render.escape(plan.get('label') or folder)}</code>. Progress is on the "
            f"<b>LoRA Details</b> tab; reopen this dialog when it has finished.{mc_lora_render.escape(key_note)}",
            kind="ok",
        )
        return message, gr.update(interactive=False, value="Fetch queued")

    fetch_button.click(fn=fetch_folder, inputs=[fetch_state], outputs=[fetch_note, fetch_button], show_progress=False)

    # -------------------------------------------------------- prompt actions

    append_prompt.click(
        fn=None,
        _js=f"function(text){{ mcLoraAppendPrompt({json.dumps(tabname)}, text); return []; }}",
        inputs=[prompt_box],
        outputs=[],
        show_progress=False,
    )
    replace_prompt.click(
        fn=None,
        _js=f"function(text){{ mcLoraReplacePrompt({json.dumps(tabname)}, text); return []; }}",
        inputs=[prompt_box],
        outputs=[],
        show_progress=False,
    )

    # ------------------------------------------------------------- the move

    try:
        if _lay_out(editor, head=head, file_column=file_column, before=before, status_after=[pick_text, pick_button]):
            _tag(box, DIALOG_CLASS)
        else:
            log("the LoRA dialog was left as Forge built it; the details are drawn under the notes")
    except Exception:
        report("could not lay the LoRA dialog out; the details are drawn under the notes")


# ---------------------------------------------------------------- the patch


def _editor_class():
    """``LoraUserMetadataEditor`` - importable because Forge puts every
    extension root on ``sys.path`` before loading any script."""
    try:
        from ui_edit_user_metadata import LoraUserMetadataEditor  # type: ignore
    except Exception:
        return None
    return LoraUserMetadataEditor


def install() -> bool:
    """Wrap the LoRA editor's ``create_default_buttons``, once.

    Forge exposes no hook for extending an extra-networks dialog, so one
    method on the LoRA editor class is wrapped. The wrapper is idempotent
    (``Reload UI`` re-runs every script but leaves ``ui_edit_user_metadata``
    in ``sys.modules``) and fails soft: if the layout cannot be built, the
    stock dialog is created exactly as before.
    """
    editor_class = _editor_class()
    if editor_class is None:
        log("the built-in LoRA extension was not found; the LoRA dialog is Forge's own")
        return False

    if getattr(editor_class, MARKER, False):
        return True

    original = editor_class.create_default_buttons

    def create_default_buttons(self):
        if opt(OPT_ENABLED, True):
            try:
                build_dialog(self)
            except Exception:
                report("could not build the LoRA dialog; Forge's own is drawn")
        return original(self)

    create_default_buttons.__wrapped__ = original
    editor_class.create_default_buttons = create_default_buttons
    setattr(editor_class, MARKER, True)
    log("the LoRA dialog is drawn full width")
    return True


def on_before_ui() -> None:
    """The host's hook before it builds the page: the editor class exists by
    then, and the dialog is built with it."""
    try:
        install()
    except Exception:
        report("could not attach the LoRA dialog")
