"""The LoRA dialog, drawn the width of the window (mc_lora_details).

The user's word on the standalone extension's panel under Forge's fields:
"it still renders everything in a narrow column in the middle of the page".
So Forge's own components are *moved* -- out of the row Forge drew them in,
into the first of three columns under one head row -- inside the wrapper
around ``create_default_buttons``, the one moment Forge gives an extension
inside the dialog. Gradio is not installed here, so its block tree is stood
in for by one with the same contract: a block is appended to the open
context's ``children`` with ``parent`` set, a context is entered and left,
and ``fill_expected_parents`` regroups a context's fields.
"""

from __future__ import annotations

import csv
import hashlib
import json
import struct
import types
from pathlib import Path

import pytest

import mc_lora_details as details


def fake_gradio():
    """Gradio 4's block tree, as far as the layout reads and writes it."""
    stack = []

    class Block:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs
            self.value = kwargs.get("value", args[0] if args else None)
            self.elem_id = kwargs.get("elem_id")
            classes = kwargs.get("elem_classes") or []
            self.elem_classes = [classes] if isinstance(classes, str) else list(classes)
            self.visible = kwargs.get("visible", True)
            self.parent = stack[-1] if stack else None
            if self.parent is not None:
                self.parent.children.append(self)
            self.events = []

        def _on(self, kind, kwargs):
            self.events.append((kind, kwargs))
            return self

        def click(self, **kwargs):
            return self._on("click", kwargs)

        def change(self, **kwargs):
            return self._on("change", kwargs)

        def then(self, **kwargs):
            return self._on("then", kwargs)

    class BlockContext(Block):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.children = []
            self.regrouped = 0

        def __enter__(self):
            stack.append(self)
            return self

        def __exit__(self, *exc):
            stack.pop()
            return False

        def fill_expected_parents(self):
            self.regrouped += 1

    def context(name):
        return type(name, (BlockContext,), {})

    def component(name):
        return type(name, (Block,), {})

    return types.SimpleNamespace(
        Row=context("Row"), Column=context("Column"), Group=context("Group"), Form=context("Form"),
        HTML=component("HTML"), Button=component("Button"), Dropdown=component("Dropdown"),
        Image=component("Image"), Textbox=component("Textbox"), State=component("State"),
        Slider=component("Slider"), HighlightedText=component("HighlightedText"),
        update=lambda **kwargs: dict(kwargs),
        blocks=types.SimpleNamespace(BlockContext=BlockContext),
    )


def forge_editor(gr, page=None):
    """The dialog as Forge's ``create_editor`` builds it, up to the moment it
    calls ``create_default_buttons``: the box still open. The description
    sits in the pseudo form Gradio wraps a lone text box in."""
    box = gr.Group(visible=False, elem_id="txt2img_lora_edit_user_metadata", elem_classes="edit-user-metadata")
    box.__enter__()
    editor = types.SimpleNamespace(box=box, tabname="txt2img", id_part="txt2img_lora_edit_user_metadata", page=page)
    editor.edit_name_input = gr.Textbox(visible=False)
    editor.button_edit = gr.Button(visible=False)
    with gr.Row() as row:
        with gr.Column():
            editor.edit_name = gr.HTML(elem_classes="extra-network-name")
            with gr.Form() as form:
                editor.edit_description = gr.Textbox(label="Description")
            editor.html_filedata = gr.HTML()
            editor.select_sd_version = gr.Dropdown(label="Preset")
        with gr.Column():
            editor.html_preview = gr.HTML()
    editor.taginfo = gr.HighlightedText(label="Training dataset tags")
    editor.edit_activation_text = gr.Textbox(label="Activation text")
    editor.slider_preferred_weight = gr.Slider(label="Preferred weight")
    editor.edit_negative_text = gr.Textbox(label="Negative prompt")
    with gr.Row() as random_row:
        gr.Textbox(label="Random prompt")
        gr.Button("Generate")
    editor.edit_notes = gr.Textbox(label="Notes")
    editor.stock = types.SimpleNamespace(row=row, form=form, random_row=random_row)
    return editor


def finish(editor):
    editor.box.__exit__(None, None, None)
    return editor.box


@pytest.fixture
def gr(monkeypatch):
    fake = fake_gradio()
    monkeypatch.setattr(details, "gr", fake)
    return fake


def make_safetensors(path: Path, payload: bytes, metadata: dict | None = None) -> str:
    header = json.dumps({"__metadata__": metadata or {}}).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(header)) + header + payload)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_library(directory: Path, rows: list[dict]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        columns.extend(key for key in row if key not in columns)
    with (directory / "civitai-moss.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def moss_row(sha256: str) -> dict:
    return {
        "safetensor_file": "moss/mossLight_v1.safetensors",
        "sha256": sha256,
        "civitai_model_name": "Moss light",
        "civitai_version_name": "v1.0",
        "base_model": "MOSS Image",
        "civitai_creator": "kohaku",
        "civitai_url": "https://civitai.com/models/11?modelVersionId=22",
        "trigger_words": "rim light, backlit",
        "status": "ok",
        "model_description": "About it.",
        "negative_prompt": "flat",
        "positive_prompt_1": "a prompt",
        "prompt_image_url_1": "https://image.civitai.com/x/1.jpeg",
        "prompt_image_id_1": "1",
        "positive_prompt_2": "another prompt",
    }


def head_and_body(box):
    head = next(child for child in box.children if "mc-lora-head-row" in child.elem_classes)
    body = next(child for child in box.children if "mc-lora-body" in child.elem_classes)
    return head, body


def populate_of(editor):
    clicks = [kwargs for kind, kwargs in editor.button_edit.events if kind == "click"]
    assert len(clicks) == 1, "one listener of ours on Forge's own trigger"
    assert clicks[0]["inputs"] == [editor.edit_name_input]
    return clicks[0]["fn"]


# ------------------------------------------------------------------ the move


class TestTheMove:
    def test_forges_components_are_moved_into_the_columns(self, gr, lora_root, host):
        editor = forge_editor(gr)
        details.build_dialog(editor)
        box = finish(editor)

        assert "mc-lora-dialog" in box.elem_classes, "what the stylesheet widens the popup under"
        assert editor.stock.row not in box.children, "the row Forge drew is empty, and gone"
        head, body = head_and_body(box)
        assert box.children.index(head) < box.children.index(body)

        # the name first in the head row, then our title strip
        assert head.children[0] is editor.edit_name and editor.edit_name.parent is head
        assert "mc-lora-header" in head.children[1].elem_classes

        columns = body.children
        assert [column.elem_classes[-1] for column in columns] == ["mc-lora-col-file", "mc-lora-col-about", "mc-lora-col-gallery"]
        file_column = columns[0]
        assert "mc-lora-col-head" in file_column.children[0].elem_classes
        moved = file_column.children[1:]
        assert moved == [
            editor.html_preview, editor.html_filedata,
            editor.edit_description, editor.select_sd_version, editor.taginfo,
            editor.edit_activation_text, editor.slider_preferred_weight, editor.edit_negative_text, editor.edit_notes,
            editor.stock.random_row,
        ], "the file's two blocks, Forge's fields in Forge's order, then what else Forge drew"
        assert all(component.parent is file_column for component in moved)
        assert editor.stock.form not in file_column.children, "the pseudo-form that held the description alone is dropped with it"
        assert head.regrouped == 1 and file_column.regrouped == 1, "the moved boxes get their forms back"

        # the hidden pair Forge opens the dialog with stays where it was
        assert editor.edit_name_input.parent is box and editor.button_edit.parent is box
        # the gallery starts hidden; the cards' bridge is in the box
        assert columns[2].visible is False
        assert [child.elem_id for child in box.children if child.elem_id == "txt2img_lora_edit_user_metadata_mc_lora_pick"]
        # the chips edit Forge's activation text by id, which Forge gives none
        assert editor.edit_activation_text.elem_id == "txt2img_lora_edit_user_metadata_activation_text"
        populate_of(editor)

    def test_a_dialog_this_cannot_read_is_left_as_forge_built_it(self, gr, lora_root, host, caplog):
        import logging

        editor = forge_editor(gr)
        editor.edit_name = None  # nothing to find Forge's row by
        with caplog.at_level(logging.INFO, logger="model_chain"):
            details.build_dialog(editor)
        box = finish(editor)

        assert "mc-lora-dialog" not in box.elem_classes
        assert editor.stock.row in box.children and editor.edit_description.parent is editor.stock.form
        head, body = head_and_body(box)
        assert box.children.index(body) > box.children.index(editor.stock.row), "ours under Forge's, as before"
        assert any("left as Forge built it" in record.getMessage() for record in caplog.records)
        populate_of(editor), "and the details still fill in"

    def test_a_failure_in_the_move_is_reported_not_raised(self, gr, lora_root, host, monkeypatch):
        def explode(*args, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(details, "_lay_out", explode)
        editor = forge_editor(gr)
        details.build_dialog(editor)
        box = finish(editor)
        assert "mc-lora-dialog" not in box.elem_classes


# --------------------------------------------------------------- populating


class TestPopulating:
    def test_the_three_columns_fill_from_the_library(self, gr, lora_root, host, tmp_path, monkeypatch):
        loras = tmp_path / "Lora"
        sha = make_safetensors(loras / "moss" / "mossLight_v1.safetensors", b"MOSS")
        write_library(lora_root / "library", [moss_row(sha)])
        monkeypatch.setattr(details, "lora_directories", lambda: [str(loras)])
        page = types.SimpleNamespace(items={
            "mossLight_v1": {
                "filename": str(loras / "moss" / "mossLight_v1.safetensors"),
                "shorthash": sha[:12],
                "metadata": {},
                "user_metadata": {"activation text": "rim light"},
            },
        })
        editor = forge_editor(gr, page)
        details.build_dialog(editor)
        finish(editor)

        out = populate_of(editor)("mossLight_v1")
        assert len(out) == 21
        (header, about_title, facts, fetch_group, fetch_button, fetch_note, side, main, prompts_head,
         selector, meta, prompt_box, status, preview, download, bulk, cards, prompts, images, sources, plan) = out

        assert "Moss light" in header and "by kohaku" in header and "Matched by SHA256" in header
        assert "2 trigger words, 2 prompts, description" in about_title
        assert sha in facts and "MOSS Image" in facts
        assert fetch_group == {"visible": False} and plan == {}, "a resolved LoRA needs no fetch button"
        assert "mc-lora-word-on" in side and side.count("mc-lora-word") >= 2, "the chip of the word in the activation text is lit"
        assert "About it." in side and "flat" in side
        assert main == {"visible": True}
        assert "Gallery prompts" in prompts_head
        assert [value for _label, value in selector["choices"]] == [0, 1] and selector["value"] == 0
        assert "Prompt" in meta and prompt_box == "a prompt"
        assert "not downloaded yet" in status and preview == {"value": None, "visible": False}
        assert download["interactive"] is True
        assert bulk == {"interactive": True, "value": "Download all 1 images"}
        assert cards.count('class="mc-lora-card"') == 2
        assert prompts == ["a prompt", "another prompt"]
        assert images == [("https://image.civitai.com/x/1.jpeg", "1"), ("", "")]
        assert sources == ["civitai-moss.csv", "civitai-moss.csv"]

    def test_a_lora_the_library_has_nothing_for_offers_its_folders_fetch(self, gr, lora_root, host, tmp_path, monkeypatch):
        loras = tmp_path / "Lora"
        make_safetensors(loras / "krea2" / "funnyFaces_v1.safetensors", b"KREA")
        monkeypatch.setattr(details, "lora_directories", lambda: [str(loras)])
        page = types.SimpleNamespace(items={
            "funnyFaces_v1": {"filename": str(loras / "krea2" / "funnyFaces_v1.safetensors"), "shorthash": "abc", "metadata": {}, "user_metadata": {}},
        })
        editor = forge_editor(gr, page)
        details.build_dialog(editor)
        finish(editor)

        out = populate_of(editor)("funnyFaces_v1")
        header, about_title, _facts, fetch_group, fetch_button, fetch_note, _side, main = out[:8]
        assert "No CSV row matched" in header
        assert "nothing in the library" in about_title
        assert fetch_group == {"visible": True}
        assert fetch_button == {"value": "Fetch from Civitai", "interactive": True}
        assert "<code>krea2/</code>" in fetch_note
        assert main == {"visible": False}, "no gallery column without prompts"
        plan = out[-1]
        assert plan == {"folder": str(loras / "krea2"), "label": "krea2/", "retry": False}

    def test_a_failure_to_gather_shows_a_notice_and_nothing_else(self, gr, lora_root, host, monkeypatch):
        def explode(page, name):
            raise RuntimeError("boom")

        monkeypatch.setattr(details, "gather", explode)
        editor = forge_editor(gr)
        details.build_dialog(editor)
        finish(editor)
        out = populate_of(editor)("anything")
        assert "could not be loaded" in out[0] and out[7] == {"visible": False} and out[-4] == []


# ------------------------------------------------------------- the fetch plan


class TestTheFetchPlan:
    def test_the_fetch_button_appears_only_when_it_has_work(self, tmp_path, monkeypatch):
        from mc_lora_library import parse_row

        folder = tmp_path / "Lora" / "krea2"
        lora = folder / "x.safetensors"
        folder.mkdir(parents=True)
        lora.write_bytes(b"x")
        monkeypatch.setattr(details, "lora_directories", lambda: [str(tmp_path / "Lora")])

        plan = details._fetch_plan(str(lora), [])
        assert plan["folder"] == str(folder) and plan["label"] == "krea2/" and not plan["retry"]

        unresolved = parse_row({"safetensor_file": "x.safetensors", "sha256": "ab" * 32, "status": "not_found"}, source="c.csv", row_number=2)
        assert details._fetch_plan(str(lora), [unresolved])["retry"], "a miss is retried"

        resolved = parse_row(dict(moss_row("ab" * 32)), source="c.csv", row_number=2)
        assert details._fetch_plan(str(lora), [resolved]) == {}
        assert details._fetch_plan("", []) == {}

    def test_a_lora_at_the_root_of_a_lora_folder_is_labelled_by_that_folder(self, tmp_path, monkeypatch):
        root = tmp_path / "Lora"
        root.mkdir()
        (root / "x.safetensors").write_bytes(b"x")
        monkeypatch.setattr(details, "lora_directories", lambda: [str(root)])
        assert details._fetch_plan(str(root / "x.safetensors"), [])["label"] == "Lora/"


# ---------------------------------------------------------------- the patch


class TestAttachingToForge:
    def test_the_editor_class_is_wrapped_once_and_fails_soft(self, monkeypatch, host):
        calls = []

        class LoraUserMetadataEditor:
            def create_default_buttons(self):
                calls.append("stock")
                return "buttons"

        monkeypatch.setattr(details, "_editor_class", lambda: LoraUserMetadataEditor)
        assert details.install()
        wrapped = LoraUserMetadataEditor.create_default_buttons
        assert details.install() and LoraUserMetadataEditor.create_default_buttons is wrapped, "Reload UI runs every script again"
        assert wrapped.__wrapped__ is not wrapped

        built = []
        monkeypatch.setattr(details, "build_dialog", lambda editor: built.append(editor))
        editor = LoraUserMetadataEditor()
        assert editor.create_default_buttons() == "buttons"
        assert built == [editor] and calls == ["stock"], "ours first, with the box still open, then Forge's buttons"

        # off: Forge's dialog as Forge built it
        monkeypatch.setattr(host.shared.opts, details.OPT_ENABLED, False, raising=False)
        assert editor.create_default_buttons() == "buttons" and built == [editor]

        # a failure in the layout never reaches Forge
        monkeypatch.setattr(host.shared.opts, details.OPT_ENABLED, True, raising=False)

        def explode(editor):
            raise RuntimeError("boom")

        monkeypatch.setattr(details, "build_dialog", explode)
        assert editor.create_default_buttons() == "buttons"

    def test_without_the_built_in_lora_extension_nothing_is_attached(self, monkeypatch):
        monkeypatch.setattr(details, "_editor_class", lambda: None)
        assert details.install() is False
        details.on_before_ui()

    def test_the_hook_never_raises_into_the_host(self, monkeypatch):
        def explode():
            raise RuntimeError("boom")

        monkeypatch.setattr(details, "install", explode)
        details.on_before_ui()
