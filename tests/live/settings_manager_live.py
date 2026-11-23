"""The settings as accordions and the Settings Manager on a real Gradio 4.40 page.

``javascript/model_chain_settings_manager.js`` moves Gradio's own blocks into
items of its own and draws accordions of Gradio's classes, read off one on the
page. ``tests/test_settings_manager_js.py`` holds the behaviour on a page in
Gradio's shape; only a real Gradio says that the shape is right -- which blocks
Gradio wraps in a form, what an accordion's generated classes are -- and that
Gradio's own events still reach a block after it moved: a checkbox that shows a
group, a dropdown that shows a chosen script's controls, a hidden dropdown a
handler fills and shows.

So this builds Txt2Img's settings column the way Forge Neo builds it
(``modules/ui.py``, ``ui_toprow.py`` for the compact prompt-box style,
``ui_components.py`` and the processing scripts placed by section, with Forge's
``gradio-*`` class patch and its ``updateable`` patch), with Forge's own
``script.js`` callback lists stood in for, serves this extension's stylesheet
and script the way Forge does (scripts before ``</head>``, stylesheets before
``</body>``), and drives it in Chromium.

It needs Gradio 4.40.0 (with the pins Mini Paint NEO's ``tests/requirements.txt``
names) and Playwright, which the suite's own interpreter does not have, so
``tests/test_settings_manager_live.py`` runs it in the interpreter
``MC_GRADIO_PYTHON`` names and skips without one. Run by hand::

    <venv>/bin/python tests/live/settings_manager_live.py

It prints one JSON object and exits 0 only when every check held.
"""
from __future__ import annotations

import glob
import json
import os
import socket
import sys
from functools import wraps
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Forge's script.js, as far as the script leans on it: gradioApp(), the
# loaded and after-update callback lists (the latter 250 ms after a change to
# a child list, as Forge's own), and the options-changed list; opts starts
# with the setting off, so the page can be measured as Forge drew it first.
HEAD = """
<script>
window.gradioApp = function () {
    const app = document.getElementsByTagName("gradio-app")[0];
    return app && app.shadowRoot ? app.shadowRoot : (app || document);
};
var opts = {model_chain_settings_manager: false};
const __loaded = [], __after = [], __options = [];
let __ran = false, __timer = null;
window.onUiLoaded = (fn) => __loaded.push(fn);
window.onAfterUiUpdate = (fn) => __after.push(fn);
window.onOptionsChanged = (fn) => __options.push(fn);
window.__setOption = (on) => { opts.model_chain_settings_manager = on; __options.forEach((fn) => fn()); };
document.addEventListener("DOMContentLoaded", () => {
    new MutationObserver(() => {
        if (!__ran && document.querySelector("#txt2img_prompt")) {
            __ran = true;
            __loaded.forEach((fn) => { try { fn(); } catch (e) { console.error(e); } });
        }
        clearTimeout(__timer);
        __timer = setTimeout(() => __after.forEach((fn) => { try { fn(); } catch (e) { console.error(e); } }), 250);
    }).observe(document.documentElement, {childList: true, subtree: true});
});
</script>
"""


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def forge_gradio():
    """Gradio with Forge Neo's own patches (modules/gradio_extensions.py)."""
    import gradio as gr
    import gradio.blocks
    import gradio.component_meta

    gradio.component_meta.updateable = lambda x: x
    # Forge's too: Gradio writes a .pyi beside the file a component class is
    # defined in -- here, the checkout -- and cannot read a class defined in a
    # function.
    gradio.component_meta.create_or_modify_pyi = lambda *args, **kwargs: None

    def add_classes(comp):
        comp.elem_classes = [f"gradio-{comp.get_block_name()}", *(getattr(comp, "elem_classes", None) or [])]

    component_init = gr.components.Component.__init__
    context_init = gradio.blocks.BlockContext.__init__

    def patched_component(self, *args, **kwargs):
        kwargs.pop("tooltip", None)
        component_init(self, *args, **kwargs)
        add_classes(self)

    def patched_context(self, *args, **kwargs):
        context_init(self, *args, **kwargs)
        add_classes(self)

    gr.components.Component.__init__ = patched_component
    gradio.blocks.BlockContext.__init__ = patched_context
    return gr


def build():
    gr = forge_gradio()

    class FormComponent:
        def get_expected_parent(self):
            return gr.components.Form

    gr.Dropdown.get_expected_parent = FormComponent.get_expected_parent

    class ToolButton(gr.Button, FormComponent):
        @wraps(gr.Button.__init__)
        def __init__(self, value="", *args, elem_classes=None, **kwargs):
            super().__init__(*args, elem_classes=["tool", *(elem_classes or [])], value=value, **kwargs)

        def get_block_name(self):
            return "button"

    class FormRow(gr.Row, FormComponent):
        def get_block_name(self):
            return "row"

    class FormGroup(gr.Group, FormComponent):
        def get_block_name(self):
            return "group"

    class InputAccordionImpl(gr.Checkbox):
        @wraps(gr.Checkbox.__init__)
        def __init__(self, value=None, setup=False, **kwargs):
            if not setup:
                super().__init__(value=value, **kwargs)
                return
            self.accordion_id = kwargs.get("elem_id")
            super().__init__(value=value, **{**kwargs, "elem_id": f"{self.accordion_id}-checkbox",
                                             "visible": False})
            self.accordion = gr.Accordion(**{**kwargs, "elem_id": self.accordion_id,
                                             "label": kwargs.get("label", "Accordion"),
                                             "elem_classes": ["input-accordion"], "open": value})

        def __enter__(self):
            self.accordion.__enter__()
            return self

        def __exit__(self, *exc):
            self.accordion.__exit__(*exc)

        def get_block_name(self):
            return "checkbox"

    with gr.Blocks(analytics_enabled=False) as demo:
        with gr.Row():
            with gr.Column(variant="compact", elem_id="txt2img_settings"):
                with gr.Column(elem_id="txt2img_prompt_container", elem_classes="prompt-container-compact", scale=6):
                    with gr.Row(elem_id="txt2img_prompt_row", elem_classes=["prompt-row"]):
                        gr.Textbox(label="Prompt", elem_id="txt2img_prompt", show_label=False, lines=3,
                                   elem_classes=["prompt"])
                    with gr.Row(elem_id="txt2img_neg_prompt_row", elem_classes=["prompt-row"]):
                        gr.Textbox(label="Negative Prompt", elem_id="txt2img_neg_prompt", show_label=False,
                                   lines=3, elem_classes=["prompt"])
                with gr.Row(elem_classes=["toprow-compact-stylerow"]):
                    with gr.Column(elem_classes=["toprow-compact-tools"]):
                        with gr.Row(elem_id="txt2img_tools"):
                            ToolButton(value="↙️", elem_id="paste")
                            ToolButton(value="\U0001f5d1️", elem_id="txt2img_clear_prompt")
                    with gr.Column():
                        with gr.Row(elem_id="txt2img_styles_row"):
                            gr.Dropdown(label="Styles", show_label=False, elem_id="txt2img_styles",
                                        choices=["a"], value=[], multiselect=True)
                with gr.Group(visible=True):
                    with FormRow(elem_id="sampler_selection_txt2img"):
                        gr.Dropdown(label="Sampling Method", elem_id="txt2img_sampling", choices=["Euler"],
                                    value="Euler")
                        gr.Slider(minimum=1, maximum=150, step=1, elem_id="txt2img_steps",
                                  label="Sampling Steps", value=20)
                with gr.Row(elem_id="txt2img_accordions", elem_classes="accordions"):
                    with InputAccordionImpl(False, setup=True, label="Hires. fix", elem_id="txt2img_hr"):
                        gr.Slider(minimum=0, maximum=150, step=1, label="Hires steps", value=0,
                                  elem_id="txt2img_hires_steps")
                with FormRow():
                    with gr.Column(elem_id="txt2img_column_size", scale=4):
                        gr.Slider(minimum=64, maximum=4096, step=16, label="Width", value=1024, elem_id="txt2img_width")
                    with gr.Column(elem_id="txt2img_column_batch"):
                        gr.Slider(minimum=1, maximum=128, step=1, label="Batch Count", value=1,
                                  elem_id="txt2img_batch_count")
                with gr.Row():
                    gr.Slider(minimum=1.0, maximum=24.0, step=0.5, label="CFG Scale", value=6.0,
                              elem_id="txt2img_cfg_scale", scale=4)
                with gr.Row(elem_id="txt2img_seed_row"):
                    gr.Number(label="Seed", value=-1, elem_id="txt2img_seed", min_width=100, precision=0)
                    extra = gr.Checkbox(label="Extra", elem_id="txt2img_subseed_show", value=False, scale=0)
                with gr.Group(visible=False, elem_id="txt2img_seed_extras") as extras:
                    gr.Number(label="Variation seed", value=-1, elem_id="txt2img_subseed", precision=0)
                extra.change(lambda x: gr.update(visible=x), inputs=[extra], outputs=[extras])
                with FormRow(elem_id="txt2img_override_settings_row"):
                    override = gr.Dropdown([], label="Override settings", visible=False,
                                           elem_id="txt2img_override_settings", multiselect=True)
                with FormGroup(elem_id="txt2img_script_container"):
                    with gr.Group(visible=True):
                        with gr.Accordion(open=False, label="ControlNet Integrated", elem_id="controlnet"):
                            gr.Textbox(label="ControlNet setting")
                    with gr.Group(visible=True):
                        with InputAccordionImpl(False, setup=True, label="ImageStitch Integrated",
                                                elem_id="input-accordion-3"):
                            gr.Slider(label="Stitch strength", value=0.5, minimum=0, maximum=1)
                    chosen = gr.Dropdown(label="Script", elem_id="script_list", choices=["None", "X/Y/Z plot"],
                                         value="None", type="index")
                    with gr.Group(visible=False) as xyz:
                        gr.Textbox(label="X/Y/Z plot values", elem_id="txt2img_xyz_values")
                    chosen.change(lambda i: gr.update(visible=i == 1), inputs=[chosen], outputs=[xyz])
            with gr.Column(elem_id="txt2img_results"):
                fill = gr.Button("Fill the override", elem_id="live_fill_override")
                fill.click(lambda: gr.update(value=["Clip skip: 2"], choices=["Clip skip: 2"], visible=True),
                           outputs=[override])
    return demo


def install_assets():
    """Forge's way (modules/ui_gradio_extensions.py): this extension's script
    before </head>, its stylesheet before </body>, served by Gradio's file
    route."""
    import gradio.routes

    script = ROOT / "javascript" / "model_chain_settings_manager.js"
    style = ROOT / "style.css"
    head = HEAD + f'<script type="text/javascript" src="file={script}"></script>\n'
    body = f'<link rel="stylesheet" property="stylesheet" href="file={style}">\n'
    original = gradio.routes.templates.TemplateResponse

    def template_response(*args, **kwargs):
        res = original(*args, **kwargs)
        res.body = res.body.replace(b"</head>", f"{head}</head>".encode("utf8"))
        res.body = res.body.replace(b"</body>", f"{body}</body>".encode("utf8"))
        res.init_headers()
        return res

    gradio.routes.templates.TemplateResponse = template_response


def launch_browser(playwright):
    try:
        return playwright.chromium.launch()
    except Exception:
        pass
    base = os.environ.get("PLAYWRIGHT_BROWSERS_PATH") or "/opt/pw-browsers"
    for pattern in ("chromium-*/chrome-linux/chrome", "chromium-*/chrome-linux64/chrome",
                    "chromium_headless_shell-*/chrome-linux/headless_shell"):
        for executable in sorted(glob.glob(str(Path(base) / pattern)), reverse=True):
            try:
                return playwright.chromium.launch(executable_path=executable)
            except Exception:
                continue
    raise RuntimeError("no Chromium to launch")


# Where every block with an id in the column sits: its parent's id (or the
# parent's own place, for a parent with none) and its place among its parent's
# children. Gradio changes what is inside a component as the page is used (a
# dropdown's chips, a group shown) and the checks below do that on purpose;
# where a block stands is what the script moves and must put back.
SIGNATURE = """() => {
    const column = document.getElementById('txt2img_settings');
    const where = (node) => {
        const parent = node.parentElement;
        const index = Array.prototype.indexOf.call(parent.children, node);
        return (parent === column ? 'column' : (parent.id || (where(parent) + '>'))) + '#' + index;
    };
    const found = {};
    column.querySelectorAll('[id]').forEach((node) => { found[node.id] = where(node); });
    return found;
}"""

KEYS = "() => (mcSettingsManager.describe('txt2img') || []).map((s) => s.key + (s.empty ? ' (empty)' : ''))"


def main() -> int:
    from playwright.sync_api import sync_playwright

    install_assets()
    demo = build()
    port = free_port()
    demo.queue().launch(server_name="127.0.0.1", server_port=port, prevent_thread_lock=True,
                        quiet=True, show_api=False, share=False, allowed_paths=[str(ROOT)])
    checks: dict = {}
    found: dict = {}
    errors: list = []
    with sync_playwright() as playwright:
        browser = launch_browser(playwright)
        page = browser.new_page(viewport={"width": 700, "height": 1000})
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(f"http://127.0.0.1:{port}/")
        page.wait_for_selector("#txt2img_hr")
        page.wait_for_function("!!window.mcSettingsManager")
        page.wait_for_timeout(600)
        before = page.evaluate(SIGNATURE)
        checks["off at load touches nothing"] = page.evaluate("!document.querySelector('.mc-sm-list')")

        page.evaluate("__setOption(true)")
        page.wait_for_selector("#mc-sm-list-txt2img")
        found["sections"] = page.evaluate(KEYS)
        checks["every block a section, in Forge's order"] = found["sections"] == [
            "tools", "sampler", "id:txt2img_hr", "size", "cfg", "seed", "override (empty)",
            "id:controlnet", "id:input-accordion-3", "script"] or found["sections"] == [
            "tools", "sampler", "id:txt2img_hr", "size", "cfg", "seed", "override (empty)",
            "id:controlnet", "label:ImageStitch Integrated", "script"]
        look = page.evaluate("""() => {
            const ours = document.querySelector('#mc-sm-list-txt2img [data-mc-sm-key=sampler] > .mc-sm-acc');
            const gradio = document.getElementById('txt2img_hr');
            const generated = (node) => Array.from(node.classList).filter((c) => c.indexOf('svelte-') === 0).sort();
            const style = (node) => { const s = getComputedStyle(node);
                return [s.paddingTop, s.paddingLeft, s.borderTopWidth, s.borderTopStyle, s.borderTopLeftRadius,
                        s.backgroundColor]; };
            const wrap = (node) => node.querySelector(':scope > .label-wrap');
            return {classes: JSON.stringify(generated(ours)) === JSON.stringify(generated(gradio))
                        && JSON.stringify(generated(wrap(ours))) === JSON.stringify(generated(wrap(gradio))),
                    style: JSON.stringify(style(ours)) === JSON.stringify(style(gradio)),
                    wrapStyle: JSON.stringify(style(wrap(ours))) === JSON.stringify(style(wrap(gradio)))};
        }""")
        found["look"] = look
        checks["an accordion of Gradio's classes, drawn as Gradio's"] = all(look.values())

        page.click("#txt2img_subseed_show input")
        page.wait_for_function("getComputedStyle(document.getElementById('txt2img_seed_extras')).display !== 'none'",
                               timeout=10000)
        checks["a checkbox still shows a group moved into a section"] = page.evaluate(
            "!!document.getElementById('txt2img_seed_extras').closest('[data-mc-sm-key=seed]')")

        page.click("#script_list input")
        page.get_by_role("option", name="X/Y/Z plot").first.click()
        page.wait_for_function("(() => { const box = document.getElementById('txt2img_xyz_values');"
                               " return !!box && box.getClientRects().length > 0; })()", timeout=10000)
        checks["the Script choice still shows the chosen script's controls in its section"] = page.evaluate(
            "!!document.getElementById('txt2img_xyz_values').closest('[data-mc-sm-key=script]')")

        page.click("#live_fill_override")
        page.wait_for_function(
            "mcSettingsManager.describe('txt2img').find((s) => s.key === 'override').empty === false",
            timeout=10000)
        checks["a section Gradio fills later is drawn"] = page.evaluate(
            "document.querySelector('#mc-sm-list-txt2img [data-mc-sm-key=override]').getClientRects().length > 0")

        checks["Hires. fix never opened by the script"] = page.evaluate(
            "!document.querySelector('#txt2img_hr > .label-wrap').classList.contains('open')")

        page.click("#mc-sm-list-txt2img [data-mc-sm-key=manager] .label-wrap")
        page.click("#mc-sm-list-txt2img .mc-sm-row[data-mc-sm-key=seed] .mc-sm-up")
        page.click("#mc-sm-list-txt2img .mc-sm-row[data-mc-sm-key=cfg] .mc-sm-eye")
        order = page.evaluate("""() => Array.from(document.querySelectorAll('#mc-sm-list-txt2img > .mc-sm-item'))
            .map((item) => item.getAttribute('data-mc-sm-key'))""")
        found["order"] = order
        checks["the manager moves and hides at once"] = order[:6] == [
            "tools", "sampler", "id:txt2img_hr", "size", "seed", "cfg"] \
            and page.evaluate("document.getElementById('txt2img_cfg_scale').getClientRects().length === 0")

        page.evaluate("__setOption(false)")
        page.wait_for_timeout(300)
        after = page.evaluate(SIGNATURE)
        moved = sorted(key for key in before if key in after and before[key] != after[key])
        found["not put back"] = moved[:10]
        checks["off again puts Forge's column back exactly"] = not moved and len(before) > 40
        page.evaluate("localStorage.clear()")
        browser.close()
    checks["no page error"] = not errors
    print(json.dumps({"checks": checks, "found": found, "errors": errors[:5]}, indent=1))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
