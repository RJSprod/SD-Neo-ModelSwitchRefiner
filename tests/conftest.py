"""Test harness that stands in for the Forge Neo host.

The extension is normally imported by the WebUI, which supplies ``modules.*``
and ``gradio``. These tests run without either, so this file installs minimal
fakes into ``sys.modules`` before the extension is imported.

The fakes are deliberately faithful in the places the extension actually
depends on -- infotext quoting and parsing mirror
``modules/infotext_utils.py`` upstream, and ``create_infotext`` reproduces the
list-indexing behaviour that lets a per-image value vary across a batch. They
are stubs everywhere else.
"""

from __future__ import annotations

import json
import re
import sys
import types
from pathlib import Path

import pytest

EXTENSION_ROOT = Path(__file__).resolve().parent.parent
# The host puts the extension root on sys.path and loads scripts/*.py by path as
# top-level modules, so both directories need to be importable here.
sys.path.insert(0, str(EXTENSION_ROOT))
sys.path.insert(0, str(EXTENSION_ROOT / "scripts"))


# --------------------------------------------------------------------------- #
# gradio
# --------------------------------------------------------------------------- #


class _Component:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.value = kwargs.get("value")
        self.label = kwargs.get("label")
        self.elem_id = kwargs.get("elem_id")
        self._callbacks = []

    def change(self, **kwargs):
        self._callbacks.append(("change", kwargs))

    def click(self, **kwargs):
        self._callbacks.append(("click", kwargs))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Update(dict):
    """Mirrors gr.update()'s dict-like payload."""


def _make_gradio() -> types.ModuleType:
    gradio = types.ModuleType("gradio")

    for name in (
        "Dropdown", "Radio", "Textbox", "Slider", "Number", "Checkbox",
        "Markdown", "Row", "Group", "Accordion", "Button", "HTML", "State",
    ):
        setattr(gradio, name, type(name, (_Component,), {}))

    gradio.update = lambda **kwargs: _Update(kwargs)
    gradio.skip = lambda: _Update({"__skip__": True})
    gradio.components = types.SimpleNamespace(Component=_Component)
    return gradio


# --------------------------------------------------------------------------- #
# modules.infotext_utils (faithful to upstream)
# --------------------------------------------------------------------------- #

RE_PARAM = re.compile(r'\s*([\w\s\-\/]+):\s*("(?:\\.|[^\\"])+"|[^,]*)(?:,|$)')
RE_IMAGESIZE = re.compile(r"^(\d+)x(\d+)$")


def quote(text):
    if "," not in str(text) and "\n" not in str(text) and ":" not in str(text):
        return text
    try:
        return json.dumps(text, ensure_ascii=False)
    except Exception:
        return text


def unquote(text):
    if not text or not (text.startswith('"') and text.endswith('"')):
        return text
    try:
        return json.loads(text)
    except Exception:
        return text


def parse_generation_parameters(x: str) -> dict:
    """Cut-down version of the host's parser, covering the last-line params."""
    *lines, lastline = x.strip().split("\n")
    if len(RE_PARAM.findall(lastline)) < 3:
        lines.append(lastline)
        lastline = ""

    res: dict = {}
    for key, value in RE_PARAM.findall(lastline):
        value = unquote(value)
        match = RE_IMAGESIZE.match(value)
        if match is not None:
            res[f"{key}-1"] = match.group(1)
            res[f"{key}-2"] = match.group(2)
        else:
            res[key] = value
    return res


class PasteField(tuple):
    def __new__(cls, component, target, *, api=None):
        return super().__new__(cls, (component, target))

    def __init__(self, component, target, *, api=None):
        self.component = component
        self.label = target if isinstance(target, str) else None
        self.function = target if callable(target) else None
        self.api = api


# --------------------------------------------------------------------------- #
# modules.*
# --------------------------------------------------------------------------- #


class FakeOptions:
    def __init__(self):
        self.samples_save = True
        self.samples_format = "png"
        self.enable_pnginfo = True
        self.save_incomplete_images = False
        self.model_chain_save_stage1 = False
        self.model_chain_ram_budget_gb = 0.0
        self.forge_additional_modules = []
        self.forge_unet_storage_dtype = "Automatic"
        self.sd_model_checkpoint = "modelA.safetensors"
        self.data = {}

    def set(self, key, value):
        setattr(self, key, value)

    def save(self, *args, **kwargs):
        pass


class FakeState:
    def __init__(self):
        self.interrupted = False
        self.skipped = False
        self.stopping_generation = False
        self.job = ""
        self.job_count = 0
        self.job_no = 0


class FakeOptionInfo:
    def __init__(self, default=None, label="", component=None, *args, **kwargs):
        self.default = default
        self.label = label
        self.component = component
        self.section = None

    def info(self, text):
        return self

    def needs_restart(self):
        return self


class FakeScript:
    AlwaysVisible = object()

    def __init__(self):
        self.args_from = None
        self.args_to = None
        self.infotext_fields = None
        self.paste_field_names = None

    def elem_id(self, item_id):
        return f"script_modelchain_{item_id}"


class FakeProcessed:
    def __init__(self, images, all_seeds, all_prompts, all_negative_prompts, infotexts, index_of_first_image=0):
        self.images = list(images)
        self.all_seeds = list(all_seeds)
        self.all_subseeds = [-1] * len(all_seeds)
        self.all_prompts = list(all_prompts)
        self.all_negative_prompts = list(all_negative_prompts)
        self.infotexts = list(infotexts)
        self.index_of_first_image = index_of_first_image
        self.comments = ""
        self.info = infotexts[0] if infotexts else ""


def create_infotext(p, all_prompts, all_seeds, all_subseeds, comments=None, iteration=0,
                    position_in_batch=0, use_main_prompt=False, index=None, all_negative_prompts=None):
    """Reproduces the host's per-image list indexing of extra_generation_params."""
    if index is None:
        index = position_in_batch + iteration * p.batch_size

    params = {
        "Steps": p.steps,
        "Seed": all_seeds[index],
        "Size": f"{p.width}x{p.height}",
        "Model": getattr(p, "sd_model_name", "modelA"),
    }
    for key, value in p.extra_generation_params.items():
        params[key] = value[index] if isinstance(value, list) else value

    text = ", ".join(f"{k}: {quote(v)}" for k, v in params.items() if v is not None)
    negative = f"\nNegative prompt: {all_negative_prompts[index]}" if all_negative_prompts and all_negative_prompts[index] else ""
    return f"{all_prompts[index]}{negative}\n{text}".strip()


def _install_modules() -> None:
    modules = types.ModuleType("modules")
    modules.__path__ = []

    errors = types.ModuleType("modules.errors")
    errors.report = lambda message, *, exc_info=False: None
    errors.display = lambda *a, **k: None

    images = types.ModuleType("modules.images")
    images.saved = []
    images.save_image = lambda image, path, basename, seed=None, prompt=None, extension="png", info=None, p=None, **kw: (
        images.saved.append({"image": image, "path": path, "seed": seed, "info": info}), (path, None)
    )[1]

    shared = types.ModuleType("modules.shared")
    shared.opts = FakeOptions()
    shared.state = FakeState()
    shared.sd_model = None
    shared.prompt_styles = None
    shared.OptionInfo = FakeOptionInfo
    shared.options_templates = {}
    shared.options_section = lambda section, options: options
    shared.cmd_opts = types.SimpleNamespace()

    scripts_mod = types.ModuleType("modules.scripts")
    scripts_mod.Script = FakeScript
    scripts_mod.AlwaysVisible = FakeScript.AlwaysVisible
    scripts_mod.ScriptBuiltinUI = FakeScript

    processing = types.ModuleType("modules.processing")
    processing.need_global_unload = False
    processing.create_infotext = create_infotext
    processing.calls = []

    class StableDiffusionProcessingImg2Img:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.extra_generation_params = {}
            self.scripts = None
            self.script_args = []
            self.index_of_first_image = 0

        def close(self):
            pass

    processing.StableDiffusionProcessingImg2Img = StableDiffusionProcessingImg2Img
    processing.process_images = lambda p: None  # replaced per-test

    infotext_utils = types.ModuleType("modules.infotext_utils")
    infotext_utils.PasteField = PasteField
    infotext_utils.quote = quote
    infotext_utils.unquote = unquote
    infotext_utils.parse_generation_parameters = parse_generation_parameters

    ui_common = types.ModuleType("modules.ui_common")
    ui_common.refresh_symbol = "\U0001f504"
    ui_common.create_refresh_button = lambda *a, **k: _Component()

    ui_components = types.ModuleType("modules.ui_components")
    ui_components.InputAccordion = type("InputAccordion", (_Component,), {})
    ui_components.ToolButton = type("ToolButton", (_Component,), {})

    sd_models = types.ModuleType("modules.sd_models")
    sd_models.get_closet_checkpoint_match = lambda name: None
    sd_models.model_data = types.SimpleNamespace(
        sd_model=None, forge_loading_parameters={}, forge_hash="",
        set_sd_model=lambda v: None, get_sd_model=lambda: None,
    )
    sd_models.forge_model_reload = lambda: (None, True)

    sd_samplers = types.ModuleType("modules.sd_samplers")
    sd_samplers.visible_samplers = lambda: [types.SimpleNamespace(name="Euler"), types.SimpleNamespace(name="DPM++ 2M")]

    script_callbacks = types.ModuleType("modules.script_callbacks")
    script_callbacks.on_script_unloaded = lambda cb, **k: None

    modules.errors = errors
    modules.images = images
    modules.shared = shared
    modules.scripts = scripts_mod
    modules.processing = processing
    modules.infotext_utils = infotext_utils
    modules.ui_common = ui_common
    modules.ui_components = ui_components
    modules.sd_models = sd_models
    modules.sd_samplers = sd_samplers
    modules.script_callbacks = script_callbacks

    modules_forge = types.ModuleType("modules_forge")
    modules_forge.__path__ = []
    main_entry = types.ModuleType("modules_forge.main_entry")
    main_entry.refresh_models = lambda: ([], [])
    main_entry.checkpoint_change = lambda *a, **k: True
    main_entry.refresh_model_loading_parameters = lambda **k: None
    main_entry.forge_unet_storage_dtype_options = {"Automatic": (None, False)}
    modules_forge.main_entry = main_entry

    backend = types.ModuleType("backend")
    backend.__path__ = []
    memory_management = types.ModuleType("backend.memory_management")
    memory_management.get_free_memory = lambda dev=None: 8 * 1024**3
    memory_management.get_torch_device = lambda: "cuda"
    memory_management.soft_empty_cache = lambda force=False: None
    backend.memory_management = memory_management

    for name, module in {
        "modules": modules,
        "modules.errors": errors,
        "modules.images": images,
        "modules.shared": shared,
        "modules.scripts": scripts_mod,
        "modules.processing": processing,
        "modules.infotext_utils": infotext_utils,
        "modules.ui_common": ui_common,
        "modules.ui_components": ui_components,
        "modules.sd_models": sd_models,
        "modules.sd_samplers": sd_samplers,
        "modules.script_callbacks": script_callbacks,
        "modules_forge": modules_forge,
        "modules_forge.main_entry": main_entry,
        "backend": backend,
        "backend.memory_management": memory_management,
    }.items():
        sys.modules[name] = module


sys.modules.setdefault("gradio", _make_gradio())
_install_modules()


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def host():
    """The faked host modules, reset between tests.

    ``opts`` and ``state`` are reset *in place* rather than replaced: the
    extension does ``from modules.shared import opts, state``, exactly as the
    host's own scripts do, so rebinding the attribute would leave the code
    under test looking at a stale object and quietly disarm these fixtures.
    """
    import modules

    modules.shared.opts.__dict__.update(FakeOptions().__dict__)
    modules.shared.state.__dict__.update(FakeState().__dict__)
    modules.images.saved.clear()
    modules.processing.need_global_unload = False
    return modules


@pytest.fixture
def style_store(host):
    """Installs a fake style database backed by a plain dict."""
    from collections import namedtuple

    PromptStyle = namedtuple("PromptStyle", "name prompt negative_prompt path")

    class FakeStyleDatabase:
        def __init__(self, styles):
            self.styles = styles
            self.no_style = PromptStyle("None", "", "", None)
            self.reloaded = 0

        def reload(self):
            self.reloaded += 1

        @staticmethod
        def _apply(prompt, style_texts):
            for style in style_texts:
                if "{prompt}" in style:
                    prompt = style.replace("{prompt}", prompt)
                else:
                    parts = filter(None, (prompt.strip(), style.strip()))
                    prompt = ", ".join(parts)
            return prompt

        def apply_styles_to_prompt(self, prompt, styles):
            return self._apply(prompt, [self.styles.get(x, self.no_style).prompt for x in styles])

        def apply_negative_styles_to_prompt(self, prompt, styles):
            return self._apply(prompt, [self.styles.get(x, self.no_style).negative_prompt for x in styles])

    store = FakeStyleDatabase(
        {
            "Cinematic": PromptStyle("Cinematic", "cinematic still of {prompt}, film grain", "cartoon, anime", None),
            "Detailed": PromptStyle("Detailed", "highly detailed", "blurry", None),
            "WithLora": PromptStyle("WithLora", "{prompt} <lora:filmgrain:0.6>", "", None),
        }
    )
    host.shared.prompt_styles = store
    return store


@pytest.fixture
def image_factory():
    from PIL import Image

    def make(width=1024, height=1024, color=(128, 128, 128)):
        return Image.new("RGB", (width, height), color)

    return make
