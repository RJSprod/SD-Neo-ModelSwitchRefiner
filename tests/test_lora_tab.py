"""The LoRA Details tab: the library, the Civitai fetch, the folders per UI
Preset and the jobs, as one top-level tab the host is told about through
``on_ui_tabs``. Built here against the faked Gradio, as the other tabs are."""

from __future__ import annotations

import mc_lora_tab


def test_the_tab_builds_and_is_announced_by_its_label_and_id(lora_root, host):
    tabs = mc_lora_tab.on_ui_tabs()
    assert len(tabs) == 1
    interface, label, ident = tabs[0]
    assert (label, ident) == (mc_lora_tab.TAB_LABEL, mc_lora_tab.TAB_ID) == ("LoRA Details", "model_chain_lora_details")
    assert interface is not None


def test_the_tab_has_one_root_for_the_stylesheet_to_scope_on(lora_root, host, monkeypatch):
    import gradio as gr

    made = []
    original = gr.Column

    class Column(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            made.append(kwargs)

    monkeypatch.setattr(gr, "Column", Column)
    mc_lora_tab.create_tab()
    roots = [kwargs for kwargs in made if kwargs.get("elem_id") == "mc-lora-details"]
    assert len(roots) == 1 and "mc-lora-tab" in roots[0]["elem_classes"]


def test_the_notice_names_the_dialog_as_it_is_now(lora_root, host, monkeypatch):
    import gradio as gr

    notices = []
    original = gr.HTML

    class HTML(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            notices.append(self.value)

    monkeypatch.setattr(gr, "HTML", HTML)
    mc_lora_tab.create_tab()
    texts = [value for value in notices if isinstance(value, str)]
    assert any("width of the window" in text for text in texts)
    assert not any("Extended details" in text or "Extended LoRA" in text for text in texts)


def test_the_hook_never_raises_into_the_host(monkeypatch):
    def explode():
        raise RuntimeError("boom")

    monkeypatch.setattr(mc_lora_tab, "create_tab", explode)
    assert mc_lora_tab.on_ui_tabs() == []
