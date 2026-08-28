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
import os
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
    def __init__(self, *args, **kwargs):
        # Several host components take their value positionally
        # (InputAccordion(False, ...), gr.Markdown("text")).
        self.__dict__.update(kwargs)
        self.value = kwargs.get("value", args[0] if args else None)
        self.label = kwargs.get("label")
        self.elem_id = kwargs.get("elem_id")
        self._callbacks = []

    def change(self, **kwargs):
        return self._record("change", kwargs)

    def input(self, **kwargs):
        """Gradio 4's user-input-only event.

        Faithful because the distinction is load-bearing: ``change`` also fires
        when the server replaces a value, so a picker that navigates on
        ``change`` walks a folder deeper every time it refills its own dropdown.
        """
        return self._record("input", kwargs)

    def click(self, **kwargs):
        return self._record("click", kwargs)

    def blur(self, **kwargs):
        """Gradio 4's "the box lost focus" event.

        Creative Mode's pinned-LoRA field rewrites itself with the tags it kept,
        and doing that on ``input`` would move the caret while somebody was still
        typing.
        """
        return self._record("blur", kwargs)

    def release(self, **kwargs):
        """Gradio 4's slider event: fired once, when the handle is let go.

        Distinct from ``change`` for the reason the Krea panel uses it -- a
        slider dragged from 1 to 10 fires ``change`` at every step it passes
        through, and a handler that writes a preferences file would write it
        nine times for one decision.
        """
        return self._record("release", kwargs)

    def select(self, **kwargs):
        return self._record("select", kwargs)

    def submit(self, **kwargs):
        return self._record("submit", kwargs)

    def upload(self, **kwargs):
        return self._record("upload", kwargs)

    def clear(self, **kwargs):
        """Gradio 4's "the user emptied this component" event.

        An Image draws its own ✕, and that is the only way to take a picture
        off a message without replacing it -- so the panel has to hear about
        it. Distinct from ``change``, which also fires when the server puts a
        value in.
        """
        return self._record("clear", kwargs)

    def load(self, **kwargs):
        return self._record("load", kwargs)

    def _record(self, kind, kwargs):
        """Register a handler and hand back a dependency.

        Gradio 4 returns an object from every event binding, and the two things
        callers do with it are chain a ``.then()`` and pass it to another
        binding's ``cancels=``. Both are what the LLM Studio panels do, so the
        return value has to be real rather than None -- a stub returning None
        would make `.then()` an AttributeError at UI-build time and nothing
        below would ever be exercised.
        """
        self._callbacks.append((kind, kwargs))
        return _Dependency(self, kind, kwargs)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Dependency:
    """What a Gradio event binding returns: chainable, and passable to cancels."""

    def __init__(self, component, kind, kwargs):
        self.component, self.kind, self.kwargs = component, kind, kwargs
        self.chained = []

    def then(self, **kwargs):
        return self._chain("then", kwargs)

    def success(self, **kwargs):
        """Gradio 4's success-only continuation.

        Recorded under its own kind rather than folded into ``then``, because
        the difference is load-bearing and invisible from the outside: Gradio
        runs a ``then`` whether or not the event before it raised, and runs a
        ``success`` only after one that did not. Voice Chat's automatic speech
        hangs off the second, and a stub that made them the same object would
        let a regression to ``then`` pass every test.
        """
        return self._chain("success", kwargs)

    def _chain(self, kind, kwargs):
        dependency = _Dependency(self.component, kind, kwargs)
        self.chained.append(dependency)
        return dependency


class _Update(dict):
    """Mirrors gr.update()'s dict-like payload."""


def _make_gradio() -> types.ModuleType:
    gradio = types.ModuleType("gradio")

    for name in (
        "Dropdown", "Radio", "Textbox", "Slider", "Number", "Checkbox",
        "Markdown", "Row", "Column", "Group", "Accordion", "Button", "HTML",
        "State", "Gallery", "Image", "Chatbot", "File", "Blocks", "Tab",
        "DownloadButton",
    ):
        setattr(gradio, name, type(name, (_Component,), {}))

    class _Progress:
        """Stands in for gr.Progress, which Gradio passes to a long-running fn.

        Called as ``progress(fraction, desc=...)``; the real one drives the
        host's progress bar and returns nothing, so this does the same.
        """

        def __init__(self, *args, **kwargs):
            self.reports = []

        def __call__(self, fraction=0, desc=None, **kwargs):
            self.reports.append((fraction, desc))

    gradio.Progress = _Progress
    gradio.update = lambda **kwargs: _Update(kwargs)
    gradio.skip = lambda: _Update({"__skip__": True})
    gradio.components = types.SimpleNamespace(Component=_Component)
    gradio.SelectData = type("SelectData", (), {"index": -1})
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
        # Per-architecture reference/edit toggles. Note the inverted polarity of
        # the Klein one, which mirrors the host.
        self.krea2_do_reference = False
        self.anima_do_reference = False
        self.klein_no_reference = False
        self.img2img_background_color = "#ffffff"
        self.data = {}

    def set(self, key, value):
        setattr(self, key, value)

    def save(self, *args, **kwargs):
        pass

    def __getattr__(self, item):
        """Serve a registered option's default, as the host's ``Options`` does.

        Faithful because it is what makes a setting readable the moment it is
        registered: the host looks in saved data first and falls back to
        ``data_labels[key].default``, so an option the user has never saved
        still answers with its default rather than raising. Without this the
        fakes would turn every unsaved setting into an AttributeError, which
        the real object never produces -- and would hide the fact that the
        appearance settings reach the browser on a fresh install.
        """
        if item.startswith("_"):
            raise AttributeError(item)

        shared = sys.modules.get("modules.shared")
        registered = getattr(shared, "options_templates", None) or {}
        if item in registered:
            return registered[item].default

        raise AttributeError(item)


class FakeState:
    def __init__(self):
        self.interrupted = False
        self.skipped = False
        self.stopping_generation = False
        self.job = ""
        self.job_count = 0
        self.job_no = 0
        # The step counters the progress endpoint reads, and the flag the host
        # sets once it has settled job_count for a hires pass.
        self.sampling_step = 0
        self.sampling_steps = 0
        self.processing_has_refined_job_count = False
        # Stamped by state.begin() when a task starts. None until then, which
        # is what a leftover plan is checked against.
        self.time_start = None
        self.textinfo = None


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
    # Reference preprocessing goes through the host's own resize and flatten,
    # so both have to exist for that path to be exercised at all.
    images.resize_image = lambda mode, image, width, height, **kw: image.resize((width, height))
    images.flatten = lambda image, bgcolor: image.convert("RGB")

    shared = types.ModuleType("modules.shared")
    shared.opts = FakeOptions()
    shared.state = FakeState()
    shared.sd_model = None
    shared.prompt_styles = None
    shared.OptionInfo = FakeOptionInfo
    shared.options_templates = {}
    shared.device = "cuda"

    def options_section(section, options):
        """Stamps the section onto each option, as the host's own does.

        Faithful because the section identifier decides whether a setting is
        ever drawn: ui_settings skips anything whose section id is None.
        Discarding it here would let that go unnoticed.
        """
        for option in options.values():
            option.section = section
        return options

    shared.options_section = options_section
    shared.cmd_opts = types.SimpleNamespace(disable_console_progressbars=False)
    # What ``opts.save`` is given. Present because the extension writes settings
    # through the host's own store -- Voice Chat's two switches are written the
    # moment they are tapped -- and a missing name here would make every one of
    # those writes fail into a debug log rather than into a test.
    shared.config_filename = "config.json"

    class FakeTotalTqdm:
        """Enough of TotalTQDM to check the Stage 2 steps are accounted for."""

        def __init__(self):
            self._tqdm = types.SimpleNamespace(total=0)

        def updateTotal(self, new_total):
            self._tqdm.total = new_total

    shared.total_tqdm = FakeTotalTqdm()

    # modules.progress, faithful in the three functions the extension calls and
    # in the one rule that matters: a task is "active" only while it is the
    # current one. Reproduced rather than stubbed because Krea Creative Mode
    # claims a task the same way modules.call_queue does, and a stub that always
    # said yes would hide a claim that never took.
    progress_mod = types.ModuleType("modules.progress")
    progress_mod.current_task = None
    progress_mod.pending_tasks = {}
    progress_mod.finished_tasks = []

    def add_task_to_queue(id_job):
        progress_mod.pending_tasks[id_job] = 0.0

    def start_task(id_task):
        progress_mod.current_task = id_task
        progress_mod.pending_tasks.pop(id_task, None)

    def finish_task(id_task):
        if progress_mod.current_task == id_task:
            progress_mod.current_task = None
        progress_mod.finished_tasks.append(id_task)
        if len(progress_mod.finished_tasks) > 16:
            progress_mod.finished_tasks.pop(0)

    progress_mod.add_task_to_queue = add_task_to_queue
    progress_mod.start_task = start_task
    progress_mod.finish_task = finish_task

    scripts_mod = types.ModuleType("modules.scripts")
    scripts_mod.Script = FakeScript
    scripts_mod.AlwaysVisible = FakeScript.AlwaysVisible
    scripts_mod.ScriptBuiltinUI = FakeScript
    # The tab runners Model Chain reaches through to reset ImageStitch's memo.
    scripts_mod.scripts_txt2img = types.SimpleNamespace(alwayson_scripts=[])
    scripts_mod.scripts_img2img = types.SimpleNamespace(alwayson_scripts=[])

    sd_samplers_common = types.ModuleType("modules.sd_samplers_common")
    sd_samplers_common.encoded = []

    def images_tensor_to_samples(image, approximation=None, model=None):
        """Mirrors the host's route from an encoded image to a reference.

        Faithful because the whole Stage 2 reference feature rides on it: the
        reference-capable engines override ``encode_first_stage`` to divert
        anything encoded while ``dynamic_args.is_referencing`` is raised into
        their own ``ref_latents``, and to treat anything else as the img2img
        input. An engine without ``ref_latents`` simply takes neither, which is
        what an architecture with no reference path looks like from outside.
        """
        from backend.args import dynamic_args

        sd_samplers_common.encoded.append((image, approximation, model))
        if model is None:
            return image

        if dynamic_args.is_referencing:
            references = getattr(model, "ref_latents", None)
            if references is not None:
                references.append(image)
        elif hasattr(model, "ini_latent"):
            model.ini_latent = image

        return image

    sd_samplers_common.images_tensor_to_samples = images_tensor_to_samples

    api_module = types.ModuleType("modules.api")
    api_module.__path__ = []
    api_api = types.ModuleType("modules.api.api")
    api_api.decode_base64_to_image = lambda value: None
    api_module.api = api_api

    processing = types.ModuleType("modules.processing")
    processing.need_global_unload = False
    processing.create_infotext = create_infotext
    processing.calls = []
    # The host's module-level latent divisor. forge_model_reload() rewrites it
    # from the loaded VAE on every load, and process_images_inner divides the
    # requested pixel size by it to shape the noise.
    processing.opt_f = 8

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

    paths = types.ModuleType("modules.paths")
    paths.data_path = "."

    sd_models = types.ModuleType("modules.sd_models")
    sd_models.get_closet_checkpoint_match = lambda name: None
    sd_models.model_data = types.SimpleNamespace(
        sd_model=None, forge_loading_parameters={}, forge_hash="",
        set_sd_model=lambda v: None, get_sd_model=lambda: None,
    )
    sd_models.forge_model_reload = lambda: (None, True)

    sd_samplers = types.ModuleType("modules.sd_samplers")
    sd_samplers.visible_sampler_names = lambda: ["Euler", "DPM++ 2M"]
    sd_samplers.visible_samplers = lambda: [types.SimpleNamespace(name=n) for n in sd_samplers.visible_sampler_names()]

    sd_schedulers = types.ModuleType("modules.sd_schedulers")
    sd_schedulers.schedulers = [
        types.SimpleNamespace(name="automatic", label="Automatic"),
        types.SimpleNamespace(name="karras", label="Karras"),
        types.SimpleNamespace(name="beta", label="Beta"),
    ]

    script_callbacks = types.ModuleType("modules.script_callbacks")
    # Callbacks are recorded rather than discarded. The extension registers
    # them inside a try/except -- correctly, since a failure there must not stop
    # the rest of it loading -- which means a misnamed host function would
    # otherwise be swallowed and never noticed until somebody looked for a tab
    # that was not there.
    script_callbacks.registered = {"script_unloaded": [], "ui_tabs": [],
                                  "app_started": []}
    script_callbacks.on_script_unloaded = lambda cb, **k: (
        script_callbacks.registered["script_unloaded"].append(cb))
    script_callbacks.on_ui_tabs = lambda cb, **k: (
        script_callbacks.registered["ui_tabs"].append(cb))
    script_callbacks.on_app_started = lambda cb, **k: (
        script_callbacks.registered["app_started"].append(cb))

    modules.errors = errors
    modules.images = images
    modules.shared = shared
    modules.progress = progress_mod
    modules.scripts = scripts_mod
    modules.processing = processing
    modules.infotext_utils = infotext_utils
    modules.sd_samplers_common = sd_samplers_common
    modules.api = api_module
    modules.ui_common = ui_common
    modules.ui_components = ui_components
    modules.paths = paths
    modules.sd_models = sd_models
    modules.sd_samplers = sd_samplers
    modules.sd_schedulers = sd_schedulers
    modules.script_callbacks = script_callbacks

    modules_forge = types.ModuleType("modules_forge")
    modules_forge.__path__ = []
    main_entry = types.ModuleType("modules_forge.main_entry")
    main_entry.module_list = {
        "flux2_vae.safetensors": "/models/VAE/flux2_vae.safetensors",
        "qwen3_8b.safetensors": "/models/text_encoder/qwen3_8b.safetensors",
        "sdxl_vae.safetensors": "/models/VAE/sdxl_vae.safetensors",
    }
    main_entry.refresh_models = lambda: ([], sorted(main_entry.module_list))
    main_entry.checkpoint_change = lambda *a, **k: True
    main_entry.modules_change = lambda *a, **k: True
    main_entry.refresh_model_loading_parameters = lambda **k: None
    main_entry.forge_unet_storage_dtype_options = {"Automatic": (None, False)}
    modules_forge.main_entry = main_entry

    backend = types.ModuleType("backend")
    backend.__path__ = []

    backend_args = types.ModuleType("backend.args")

    class dynamic_args:
        """Mirrors the host's per-model flags and per-generation latent state."""

        kontext = False
        edit = False
        nunchaku = False
        klein = False
        wan = False
        pid = False
        anima = False
        krea2 = False
        ref_latents = []
        concat_latent = None
        lq_latent = [None, None]
        context_handler = None
        is_referencing = False
        loading_refiner = False
        resets = 0

        @classmethod
        def reset(cls):
            if cls.loading_refiner:
                return
            cls.resets += 1
            cls.ref_latents.clear()
            cls.concat_latent = None
            cls.lq_latent = [None, None]
            cls.context_handler = None

    backend_args.dynamic_args = dynamic_args
    backend.args = backend_args

    memory_management = types.ModuleType("backend.memory_management")
    memory_management.get_free_memory = lambda dev=None: 8 * 1024**3
    memory_management.get_total_memory = lambda dev=None: 24 * 1024**3
    memory_management.get_torch_device = lambda: "cuda"
    memory_management.soft_empty_cache = lambda force=False: None
    memory_management.freed = []
    memory_management.kept = []
    memory_management.loaded_to_gpu = []
    memory_management.current_loaded_models = []

    def free_memory(required, device, keep_loaded=()):
        memory_management.freed.append(required)
        memory_management.kept.append(list(keep_loaded))
        return []

    memory_management.free_memory = free_memory
    memory_management.load_models_gpu = lambda models, **kw: memory_management.loaded_to_gpu.append(
        list(models)
    )
    backend.memory_management = memory_management

    for name, module in {
        "modules": modules,
        "modules.errors": errors,
        "modules.images": images,
        "modules.shared": shared,
        "modules.progress": progress_mod,
        "modules.scripts": scripts_mod,
        "modules.processing": processing,
        "modules.infotext_utils": infotext_utils,
        "modules.sd_samplers_common": sd_samplers_common,
        "modules.api": api_module,
        "modules.api.api": api_api,
        "modules.ui_common": ui_common,
        "modules.ui_components": ui_components,
        "modules.paths": paths,
        "modules.sd_models": sd_models,
        "modules.sd_samplers": sd_samplers,
        "modules.sd_schedulers": sd_schedulers,
        "modules.script_callbacks": script_callbacks,
        "modules_forge": modules_forge,
        "modules_forge.main_entry": main_entry,
        "backend": backend,
        "backend.args": backend_args,
        "backend.memory_management": memory_management,
    }.items():
        sys.modules[name] = module


sys.modules.setdefault("gradio", _make_gradio())
_install_modules()


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def timing_store(tmp_path, monkeypatch):
    """Point the progress calibration at a throwaway file, and empty it.

    Autouse because the store is module state that outlives a test: a rate
    measured by one would silently become the next one's starting estimate, and
    the file itself would land in the working tree, since the fake host's data
    directory is the current one.
    """
    import mc_progress

    monkeypatch.setattr(mc_progress, "path", lambda: str(tmp_path / mc_progress.FILENAME))
    mc_progress.forget()
    mc_progress.abandon()
    yield
    mc_progress.abandon()


@pytest.fixture(autouse=True)
def active_plan(tmp_path, monkeypatch):
    """No execution plan in force unless the test under way publishes one.

    Autouse for the same reason as the timing store above: the plan is module
    state that outlives a test, and it is *load-bearing* module state -- a plan
    left behind by a test that ran a generation caps what the next test's
    language model may hold, so a reserve assertion that passes alone fails in
    a suite. It also mirrors what the extension does, where a plan deliberately
    survives its generation so that the VRAM freed at the end of one is not
    read as room to grow into before the next.
    """
    import mc_plan

    # Pointed at a throwaway file as well as emptied: the weight store is now on
    # disk, and a test that measured a checkpoint would otherwise write one into
    # the working tree and hand it to every test that followed.
    monkeypatch.setattr(mc_plan, "_weights_path",
                        lambda: str(tmp_path / mc_plan.WEIGHTS_FILENAME))
    mc_plan.clear()
    mc_plan.note_placement(None)
    mc_plan.forget_misses()
    mc_plan.forget_weights()
    yield
    mc_plan.clear()
    mc_plan.note_placement(None)
    mc_plan.forget_misses()
    mc_plan.forget_weights()


@pytest.fixture(autouse=True)
def _forget_runtime_roles():
    """An empty runtime registry, and a shared runtime nobody has claimed.

    Both are module state, and the registry writes to the *shared* runtime: it
    adopts it for a role whose configuration matches the installation's, which
    leaves that role recorded on an object every other test also uses. A test
    that then patched ``config`` with a no-argument double got it called with a
    role name, from a binding a completely different test had left behind.
    """
    import mc_llm_runtime

    def clean():
        mc_llm_runtime.registry.forget()
        mc_llm_runtime.runtime.roles = ()
        mc_llm_runtime.runtime._role = ""
        mc_llm_runtime.runtime._key = None
        mc_llm_runtime.runtime.residency_key = mc_llm_runtime.RESIDENCY_KEY

    clean()
    yield
    clean()


@pytest.fixture
def host():
    """The faked host modules, reset between tests.

    ``opts`` and ``state`` are reset *in place* rather than replaced: the
    extension does ``from modules.shared import opts, state``, exactly as the
    host's own scripts do, so rebinding the attribute would leave the code
    under test looking at a stale object and quietly disarm these fixtures.

    Cleared before updating, so a setting one test invents cannot survive into
    the next. Options the extension registers at runtime are absent from
    ``FakeOptions``, and an update alone would leave them set for every test
    that followed.
    """
    import modules

    modules.shared.opts.__dict__.clear()
    modules.shared.opts.__dict__.update(FakeOptions().__dict__)
    modules.shared.state.__dict__.clear()
    modules.shared.state.__dict__.update(FakeState().__dict__)
    modules.images.saved.clear()
    modules.processing.need_global_unload = False
    modules.processing.opt_f = 8
    modules.sd_samplers_common.encoded.clear()
    modules.scripts.scripts_txt2img.alwayson_scripts.clear()
    modules.scripts.scripts_img2img.alwayson_scripts.clear()
    modules.shared.sd_model = None

    from backend.args import dynamic_args

    for name in ("kontext", "edit", "nunchaku", "klein", "wan", "pid", "anima", "krea2"):
        setattr(dynamic_args, name, False)
    dynamic_args.ref_latents.clear()
    dynamic_args.resets = 0

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


@pytest.fixture
def chain(host, style_store, monkeypatch, image_factory):
    """A ScriptModelChain wired to fakes, with a recording Stage 2.

    Shared because more than one module drives full generations through it.
    """
    import types

    import mc_memory

    """A ScriptModelChain wired to fakes, with a recording Stage 2."""
    import model_chain

    monkeypatch.setattr(
        host.sd_models,
        "get_closet_checkpoint_match",
        lambda name: None
        if not name or name == "None"
        else types.SimpleNamespace(
            filename=f"/models/{name}", name_for_extra=name.split(".")[0], title=name, sha256="abc123"
        ),
    )
    monkeypatch.setattr(mc_memory, "plan", lambda name, mods=None: mc_memory.ResidencyPlan("dual", "both fit"))

    switches: list[str] = []
    monkeypatch.setattr(
        mc_memory, "ensure_resident",
        lambda name, mods=None: (switches.append((name, mods)), "cold")[1],
    )

    restores: list[str] = []
    monkeypatch.setattr(
        mc_memory, "restore_selection",
        lambda name, mods=None: restores.append((name, mods)),
    )
    monkeypatch.setattr(mc_memory, "reinstate_pending", lambda: False)

    refine_calls: list = []

    def fake_process_images(p2):
        refine_calls.append(p2)
        image = image_factory(p2.width, p2.height)
        return types.SimpleNamespace(images=[image], index_of_first_image=0)

    monkeypatch.setattr(model_chain, "process_images", fake_process_images)
    monkeypatch.setattr(
        model_chain, "create_infotext",
        lambda p, prompts, seeds, subseeds, index=None, **kw: f"infotext#{index} seed={seeds[index]}",
    )

    script = model_chain.ScriptModelChain()
    return types.SimpleNamespace(
        script=script,
        switches=switches,
        restores=restores,
        refine_calls=refine_calls,
        module=model_chain,
    )


# --------------------------------------------------------------------------- #
# Voice Chat
# --------------------------------------------------------------------------- #


FAKE_VOICE_WORKER = r'''
"""A Voice Worker that speaks the real protocol and loads no speech engine.

Deliberately a second implementation of the framing rather than an import of
``voice_worker.worker``: a test in which both ends of a wire protocol are the
same function proves that the function agrees with itself.

What it does is steered by MC_FAKE_VOICE in the environment, so one script
covers the ordinary case, a handshake the parent must refuse, a worker that
ignores a polite shutdown, and one that is inside "native" work when the parent
is killed.
"""

import json
import os
import struct
import sys
import time

PLAN = json.loads(os.environ.get("MC_FAKE_VOICE") or "{}")
LENGTH = struct.Struct(">I")


def _read(stream, count):
    out = b""
    while len(out) < count:
        block = stream.read(count - len(out))
        if not block:
            return None
        out += block
    return out


def read_frame(stream):
    raw = _read(stream, 4)
    if raw is None:
        return None
    header = _read(stream, LENGTH.unpack(raw)[0])
    if header is None:
        return None
    raw = _read(stream, 4)
    if raw is None:
        return None
    payload = _read(stream, LENGTH.unpack(raw)[0])
    if payload is None:
        return None
    return json.loads(header.decode("utf-8")), payload


def write_frame(stream, header, payload=b""):
    raw = json.dumps(header).encode("utf-8")
    stream.write(LENGTH.pack(len(raw)))
    stream.write(raw)
    stream.write(LENGTH.pack(len(payload)))
    if payload:
        stream.write(payload)
    stream.flush()


def note(text):
    sys.stderr.write("fake-voice-worker: %s\n" % text)
    sys.stderr.flush()


def touch():
    marker = PLAN.get("alive_marker")
    if marker:
        with open(marker, "a", encoding="utf-8") as handle:
            handle.write("%d\n" % os.getpid())


def containment(parent_pid):
    if sys.platform.startswith("linux"):
        try:
            import ctypes
            import signal

            libc = ctypes.CDLL("libc.so.6", use_errno=True)
            if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
                return "pipe"
        except Exception:
            return "pipe"
        if parent_pid and os.getppid() != parent_pid:
            raise SystemExit(0)
        return "pdeathsig"
    if os.name == "nt":
        return "job"
    return "pipe"


def main():
    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    touch()
    behaviour = PLAN.get("behaviour", "normal")
    while True:
        frame = read_frame(stdin)
        if frame is None:
            note("input closed")
            return 0
        header, payload = frame
        operation = header.get("op")
        request = header.get("id")

        if operation == "init":
            found = containment(int(header.get("parent_pid") or 0))
            reply = {
                "ok": True, "id": request, "protocol_version": 1,
                "runtime_version": "1.13.6", "provider": "cpu",
                "parent_death": found, "stt_model_id": "whisper-small-int8",
                "tts_model_id": "kokoro-multi-lang-v1-cpu",
                "stt_threads": 4, "tts_threads": 2,
            }
            reply.update(PLAN.get("handshake") or {})
            if behaviour == "silent_handshake":
                # Never answers. The parent's start has to time out and still
                # leave no process behind.
                while True:
                    time.sleep(0.05)
            write_frame(stdout, reply)
            continue

        if operation == "shutdown":
            if behaviour == "ignore_shutdown":
                note("ignoring shutdown")
                while True:
                    time.sleep(0.05)
            write_frame(stdout, {"ok": True, "id": request})
            return 0

        if operation == "stt":
            if behaviour == "die_on_stt":
                os._exit(9)
            if behaviour == "busy_native":
                # Stands in for a long native inference call: this process is
                # not waiting on stdin, so pipe EOF cannot be what ends it.
                touch()
                while True:
                    time.sleep(0.05)
            write_frame(stdout, {"ok": True, "id": request,
                                 "text": PLAN.get("transcript", ""),
                                 "audio_seconds": 1.0, "elapsed": 0.1})
            continue

        if operation == "tts":
            write_frame(stdout, {"ok": True, "id": request, "sample_rate": 24000,
                                 "audio_seconds": 1.0, "elapsed": 0.1},
                        bytes.fromhex(PLAN.get("audio_hex", "")))
            continue

        write_frame(stdout, {"ok": False, "id": request, "error": "unknown operation"})


if __name__ == "__main__":
    raise SystemExit(main())
'''


class VoiceHarness:
    """A voice installation that is entirely fake and entirely real-shaped."""

    def __init__(self, root, script, plan):
        self.root = root
        self.script = script
        self.plan = plan
        self.handshake = plan.setdefault("handshake", {})
        self.transcript = plan.get("transcript", "")
        self.audio = bytes.fromhex(plan.get("audio_hex", ""))
        self.wav = _silent_wav()

    def publish(self):
        """Write the plan where the fake worker will read it."""
        import os as _os

        _os.environ["MC_FAKE_VOICE"] = json.dumps(self.plan)


def _silent_wav(seconds: float = 1.0, rate: int = 16000) -> bytes:
    """A valid 16 kHz mono PCM16 WAV of the requested length."""
    import struct as _struct

    frames = int(rate * seconds)
    body = b"\0\0" * frames
    return (b"RIFF" + _struct.pack("<I", 36 + len(body)) + b"WAVE"
            + b"fmt " + _struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
            + b"data" + _struct.pack("<I", len(body)) + body)


def pytest_configure(config):
    """Register the one marker the voice fixtures read.

    Declared rather than left to pytest's "unknown mark" warning, because a
    misspelled behaviour would otherwise silently fall back to the ordinary
    worker and the test that asked for a crashing one would pass for the wrong
    reason.
    """
    config.addinivalue_line(
        "markers",
        "voice(**plan): how the fake Voice Worker should behave for this test.")


@pytest.fixture
def silent_wav():
    return _silent_wav


@pytest.fixture
def voice_root(tmp_path, monkeypatch):
    """Point every voice path at a throwaway directory.

    Autouse would have been wrong: the paths module answers from the host's data
    directory, and a test that has not asked for a voice installation should see
    exactly what a fresh machine sees.
    """
    import mc_voice_paths

    root = tmp_path / "model_chain_voice"
    monkeypatch.setattr(mc_voice_paths, "data_root", lambda: root)
    return root


@pytest.fixture
def fake_worker(tmp_path, monkeypatch, voice_root, request):
    """A started-on-demand Voice Worker that is a Python script, not a model.

    Parametrise it with ``@pytest.mark.voice(behaviour=..., handshake=...)``.
    Everything the runtime manager checks -- readiness, the interpreter, the
    worker path, the model ids -- is answered here, so the tests are about the
    manager's own behaviour rather than about having a gigabyte of ONNX.
    """
    import mc_voice_models
    import mc_voice_paths
    import mc_voice_runtime

    marker = request.node.get_closest_marker("voice")
    plan = dict(marker.kwargs) if marker else {}
    plan.setdefault("transcript", "the quick brown fox")
    plan.setdefault("audio_hex", _silent_wav(0.25, 24000).hex())
    plan.setdefault("alive_marker", str(tmp_path / "worker-alive.txt"))

    script = tmp_path / "fake_voice_worker.py"
    script.write_text(FAKE_VOICE_WORKER, encoding="utf-8")

    harness = VoiceHarness(voice_root, script, plan)
    harness.publish()

    ready = mc_voice_models.Status(
        runtime_ready=True, stt_ready=True, tts_ready=True,
        runtime_message="Installed", stt_message="Installed", tts_message="Installed",
        platform_supported=True, stt_id="whisper-small-int8",
        tts_id="kokoro-multi-lang-v1-cpu", tts_voice="af_heart")

    monkeypatch.setattr(mc_voice_models, "status", lambda: ready)
    monkeypatch.setattr(mc_voice_models, "runtime_python", lambda: Path(sys.executable))
    monkeypatch.setattr(mc_voice_paths, "worker_script", lambda: script)
    monkeypatch.setattr(mc_voice_models, "bundle_paths",
                        lambda kind: {"id": ready.stt_id if kind == "stt" else ready.tts_id})
    monkeypatch.setattr(mc_voice_runtime, "CONTAINMENT",
                        {"windows": "job", "linux": "pdeathsig"})

    yield harness

    mc_voice_runtime.stop("test finished")
    mc_voice_runtime._failures.clear()
    os.environ.pop("MC_FAKE_VOICE", None)


@pytest.fixture(autouse=True)
def _forget_voice_targets():
    """No speech target survives a test.

    Module state, and the kind that would make a later test speak an earlier
    test's sentence -- which is precisely the confusion the one-shot token
    exists to prevent in production.
    """
    yield
    try:
        import mc_voice_api

        mc_voice_api.forget_targets()
    except Exception:
        pass
