"""LoRA tags at the end of a prompt, one a line, saved and recalled (mc_saved_prompt).

The user's report: an image made here, read back in PNG Info, has every LoRA at
the front of its prompt, and they want them at the end (2026-11-03) -- when an
image is recalled into the prompt box as well, on every model ("krea, moss,
sdxl, etc"), and each tag on a line of its own (2026-11-23); then PNG Info
itself, which showed an older image's text as the file holds it ("I went
straight to png info to load the settings of a previous image. The lora all
showed up at the front of the prompt!", 2026-11-23). The front is this
extension's doing -- every bare tag is lifted out of the Txt2Img prompt and put
back in front of it before Forge sees it -- so the fix moves them in the saved
text and in the text read back, and never in what Forge generates from. What
has to hold, a class or two each:

* the move itself: the words, then each tag on its own line in the order they
  had, the words tidied as a paste of them would tidy them;
* what makes the move safe: pasting the saved text back gives Stage 1 the same
  prompt as pasting the text Forge would have saved before, for every prompt
  the hook can produce;
* the wiring: Forge's ``create_infotext`` is wrapped without being made to
  compute anything differently, and every generation's text is rewritten;
* the recall: every recalled prompt comes back in the same shape, whatever
  made the image;
* PNG Info: an image saved before the move is shown, and sent, as the same
  image saved today, with only its prompt's lines moved; an image dropped on
  the prompt box puts that text there; and the move runs after every other
  extension's paste hook.
"""

from __future__ import annotations

import html
import inspect
import random
import re
import sys
import types

import pytest

import mc_saved_prompt
from prompt_master.krea import extra_networks, literals

from test_krea_literals import generate, script, store  # noqa: F401


# --------------------------------------------------------------------------- #
# What a paste of a saved prompt gives Stage 1
# --------------------------------------------------------------------------- #


def stage_one(pasted: str) -> str:
    """The prompt Stage 1 gets when ``pasted`` is the Prompt box on Generate.

    The hook's own steps, in its order (``before_process``): every bare tag
    bracketed, the brackets lifted, the payloads put back around the words.
    """
    parsed = literals.parse(extra_networks.protect(pasted))
    return literals.restore(parsed.clean_text, parsed)


def restored(prompt: str, positive: str = "", negative: str = "") -> str:
    """What the hook makes of a typed prompt and the two Literal boxes: the
    string Forge is given, and so the ``Prompt:`` line it saves today."""
    parsed = literals.merge(literals.parse(extra_networks.protect(prompt)),
                            positive, negative)
    return literals.restore(parsed.clean_text, parsed)


class TestTheMove:
    def test_tags_go_to_the_end_in_the_order_they_had_one_a_line(self):
        line = "<lora:realfilter_v2:0.85> <lora:contrast:-6> a woman in the rain, soft light"
        assert extra_networks.last(line) == (
            "a woman in the rain, soft light\n<lora:realfilter_v2:0.85>\n<lora:contrast:-6>")

    def test_the_words_are_tidied_as_a_paste_would_tidy_them(self):
        line = "<lora:a:1>, a cat, <lora:b:0.5>, on a mat"
        assert extra_networks.last(line) == "a cat, on a mat\n<lora:a:1>\n<lora:b:0.5>"

    def test_every_kind_the_hook_lifts_moves_and_case_does_not_matter(self):
        line = "<LoRA:a:1> <lyco:b:1> <hypernet:c:1> words"
        assert extra_networks.last(line) == "words\n<LoRA:a:1>\n<lyco:b:1>\n<hypernet:c:1>"

    def test_a_line_without_tags_is_the_same_string(self):
        line = "a quiet street,  at dusk "
        assert extra_networks.last(line) is line

    def test_a_line_of_tags_alone_is_one_tag_a_line(self):
        assert extra_networks.last("<lora:a:1> <lora:b:2>") == "<lora:a:1>\n<lora:b:2>"

    def test_tags_already_at_the_end_on_one_line_are_given_a_line_each(self):
        line = "a cat, on a mat <lora:a:1> <lora:b:0.5>"
        assert extra_networks.last(line) == "a cat, on a mat\n<lora:a:1>\n<lora:b:0.5>"

    def test_a_text_already_in_shape_comes_back_equal_and_twice_is_once(self):
        shaped = "a cat, on a mat\n<lora:a:1>\n<lora:b:0.5>"
        assert extra_networks.last(shaped) == shaped
        for line in ("<lora:a:1>, a cat, <lora:b:0.5>, on a mat", "x <lora:a:1>\n\ny",
                     "<lora:only:1>"):
            once = extra_networks.last(line)
            assert extra_networks.last(once) == once

    def test_a_line_that_still_holds_brackets_is_left_as_it_is(self):
        line = "<lora:a:1> [[__wildcard__]] words"
        assert extra_networks.last(line) is line

    @pytest.mark.parametrize("sign", ["-", "+"])
    def test_a_tag_right_after_a_sign_leaves_the_line_as_it_is(self, sign):
        line = f"<lora:a:1> words {sign}<lora:b:1>"
        assert extra_networks.last(line) is line

    def test_lines_are_kept(self):
        line = "<lora:a:1> first line\nsecond line"
        assert extra_networks.last(line) == "first line\nsecond line\n<lora:a:1>"

    def test_tags_between_lines_join_the_others_at_the_end(self):
        line = "first line <lora:a:1>\n<lora:b:1>\nsecond line"
        assert extra_networks.last(line) == "first line\n\nsecond line\n<lora:a:1>\n<lora:b:1>"


class TestAPasteGeneratesWhatItDidBefore:
    """The move is safe only if pasting the saved text back is unchanged by it."""

    CASES = [
        ("a cat <lora:a:1>", "", ""),
        ("<lora:a:1> a cat, <lora:b:0.5>, on a mat", "", ""),
        ("portrait of a woman", "<lora:realfilter:1>", "blue hat"),
        ("portrait <lora:x:0.7>", "masterpiece, <lora:y:1>", "film grain"),
        ("first line\nsecond <lora:x:1>\n\n\nthird", "", ""),
        (", <lora:x:1> ,, words ,", "", ""),
        ("<lora:only:1>", "", ""),
        ("words <LORA:Up:1> more", "", ""),
        ("[[<lora:x:1>]] typed in front", "", ""),
        ("a scene <lora:x:1:0.5> with two weights", "", ""),
    ]

    @pytest.mark.parametrize("prompt,positive,negative", CASES)
    def test_both_saved_lines_paste_to_the_same_stage_one_prompt(self, prompt, positive,
                                                                 negative):
        today = restored(prompt, positive, negative)
        assert stage_one(extra_networks.last(today)) == stage_one(today)

    def test_over_a_spread_of_generated_prompts(self):
        rng = random.Random(20261103)
        words = ["a", "cat", "neon", "street,", ",", " ", "  ", "\n", "BREAK", "(soft:1.2)",
                 "[a:b:0.5]", "-", "+", "film grain", "<lora:x:0.85>", "<lora:y:-1>",
                 "<lyco:z:1>", "<lora:w:1:0.5>", "__wild__"]
        checked = 0
        for _ in range(2500):
            prompt = " ".join(rng.choice(words) for _ in range(rng.randint(0, 9)))
            positive = rng.choice(["", "<lora:p:1>", "masterpiece, <lora:q:0.5>", "best,"])
            negative = rng.choice(["", "blue hat", "film grain <lora:n:1>"])
            today = restored(prompt, positive, negative)
            moved = extra_networks.last(today)
            assert stage_one(moved) == stage_one(today), (prompt, positive, negative)
            if moved != today:
                checked += 1
                tags = extra_networks._TAG.findall(today)
                assert extra_networks._TAG.findall(moved) == tags
                assert moved.split("\n")[-len(tags):] == tags
        assert checked > 1000, "the spread moved too few lines to prove anything"


# --------------------------------------------------------------------------- #
# The wrapper around Forge's create_infotext
# --------------------------------------------------------------------------- #


def forges_create_infotext(p, all_prompts, all_seeds, all_subseeds, comments=None,
                           iteration=0, position_in_batch=0, use_main_prompt=False,
                           index=None, all_negative_prompts=None):
    """Forge Neo's ``create_infotext`` as far as this feature reads it.

    The index, the prompt, the negative prompt, callable parameters given
    ``**locals()`` (how Forge's ``Hires prompt`` compares itself with the
    prompt), and the shape of the returned text, as in ``modules/processing.py``.
    """
    if use_main_prompt:
        index = 0
    elif index is None:
        index = position_in_batch + iteration * p.batch_size
    if all_negative_prompts is None:
        all_negative_prompts = p.all_negative_prompts
    prompt_text = p.main_prompt if use_main_prompt else all_prompts[index]
    negative_prompt = p.main_negative_prompt if use_main_prompt else all_negative_prompts[index]
    generation_params = {"Steps": 20, "Seed": all_seeds[index], **p.extra_generation_params}
    for key, value in generation_params.items():
        if callable(value):
            generation_params[key] = value(**locals())
    text = ", ".join(f"{k}: {v}" for k, v in generation_params.items() if v is not None)
    negative_prompt_text = f"\nNegative prompt: {negative_prompt}" if negative_prompt else ""
    return f"{prompt_text}{negative_prompt_text}\n{text}".strip()


def get_hr_prompt(p, index, prompt_text, **kwargs):
    """Forge's: the hires prompt, only when it differs from the prompt."""
    hr_prompt = p.all_hr_prompts[index]
    return hr_prompt if hr_prompt != prompt_text else None


def processing_object(prompts, *, negatives=None, batch_size=1):
    return types.SimpleNamespace(
        batch_size=batch_size, all_negative_prompts=negatives or [""] * len(prompts),
        main_prompt=prompts[0], main_negative_prompt=(negatives or [""])[0],
        all_hr_prompts=list(prompts), extra_generation_params={"Hires prompt": get_hr_prompt})


@pytest.fixture
def host_module():
    module = types.SimpleNamespace(create_infotext=forges_create_infotext)
    assert mc_saved_prompt.install(module)
    yield module
    mc_saved_prompt.uninstall(module)


FRONT = "<lora:a:1> <lora:b:0.5> a woman in the rain"


class TestTheWrapper:
    def test_a_generations_prompt_ends_with_its_tags_one_a_line(self, host_module):
        p = processing_object([FRONT], negatives=["<lora:neg:1> blurry"])
        text = host_module.create_infotext(p, [FRONT], [7], [0])
        assert text.splitlines() == [
            "a woman in the rain",
            "<lora:a:1>",
            "<lora:b:0.5>",
            "Negative prompt: <lora:neg:1> blurry",
            "Steps: 20, Seed: 7",
        ]

    def test_forge_computes_its_text_as_it_always_did(self, host_module):
        """No ``Hires prompt`` appears: Forge compares the hires prompt with the
        prompt it was handed, which the wrapper leaves alone."""
        p = processing_object([FRONT])
        text = host_module.create_infotext(p, [FRONT], [7], [0])
        assert "Hires prompt" not in text
        assert text == extra_networks.last(FRONT) + "\nSteps: 20, Seed: 7"

    def test_every_generation_is_moved_whatever_ran_on_it(self, host_module):
        """Img2Img, a model the Krea hook stands down for, another extension's
        pass: nothing marks the processing object, and the text moves all the
        same ("regardless of the model", 2026-11-23)."""
        typed = "a portrait, <lora:moss_style:0.6> soft light"
        p = processing_object([typed])
        assert not vars(p).keys() - {"batch_size", "all_negative_prompts", "main_prompt",
                                     "main_negative_prompt", "all_hr_prompts",
                                     "extra_generation_params"}
        text = host_module.create_infotext(p, [typed], [7], [0])
        assert text == "a portrait, soft light\n<lora:moss_style:0.6>\nSteps: 20, Seed: 7"

    def test_a_prompt_without_tags_is_left_as_forge_wrote_it(self, host_module):
        plain = "a woman in the rain,  soft light"
        p = processing_object([plain])
        assert host_module.create_infotext(p, [plain], [7], [0]) == \
            forges_create_infotext(p, [plain], [7], [0])

    def test_the_saved_text_parses_back_to_the_moved_prompt(self, host_module):
        """Forge's parse reads a prompt over several lines, so the tags on their
        own lines come back with the prompt and not as parameters."""
        p = processing_object([FRONT], negatives=["blurry"])
        text = host_module.create_infotext(p, [FRONT], [7], [0])
        *lines, last_line = text.split("\n")
        prompt_lines = [line for line in lines if not line.startswith("Negative prompt:")]
        assert "\n".join(prompt_lines) == extra_networks.last(FRONT)
        assert last_line == "Steps: 20, Seed: 7"

    def test_the_index_is_forges_and_so_is_the_main_prompt(self, host_module):
        prompts = ["<lora:a:1> zero", "<lora:b:1> one", "<lora:c:1> two", "<lora:d:1> three"]
        p = processing_object(prompts, batch_size=2)
        third = host_module.create_infotext(p, prompts, [0, 1, 2, 3], [0] * 4,
                                            iteration=1, position_in_batch=0)
        assert third.splitlines()[:2] == ["two", "<lora:c:1>"]
        second = host_module.create_infotext(p, prompts, [0, 1, 2, 3], [0] * 4, index=1)
        assert second.splitlines()[:2] == ["one", "<lora:b:1>"]
        p.main_prompt = "<lora:g:1> the grid"
        grid = host_module.create_infotext(p, prompts, [0, 1, 2, 3], [0] * 4,
                                           use_main_prompt=True)
        assert grid.splitlines()[:2] == ["the grid", "<lora:g:1>"]

    def test_a_prompt_with_nothing_after_it_still_moves(self):
        """Forge strips the whole text, so a prompt ending in a space and
        followed by nothing is saved without that space."""
        module = types.SimpleNamespace(
            create_infotext=lambda p, all_prompts, *a, **k: f"{all_prompts[0]}".strip())
        assert mc_saved_prompt.install(module)
        p = processing_object([FRONT + " "])
        assert module.create_infotext(p, [FRONT + " "], [7], [0]) == \
            "a woman in the rain\n<lora:a:1>\n<lora:b:0.5>"

    def test_text_that_does_not_start_with_the_prompt_is_left_alone(self):
        module = types.SimpleNamespace(
            create_infotext=lambda p, all_prompts, *a, **k: "someone else's text")
        assert mc_saved_prompt.install(module)
        p = processing_object([FRONT])
        p.batch_size = 1
        assert module.create_infotext(p, [FRONT], [7], [0]) == "someone else's text"

    def test_a_surprise_returns_forges_text(self, host_module, monkeypatch):
        def surprise(text):
            raise RuntimeError("a surprise")

        monkeypatch.setattr(extra_networks, "last", surprise)
        p = processing_object([FRONT])
        assert host_module.create_infotext(p, [FRONT], [7], [0]) == \
            forges_create_infotext(p, [FRONT], [7], [0])

    def test_installing_twice_wraps_once_and_uninstalling_puts_forges_back(self):
        module = types.SimpleNamespace(create_infotext=forges_create_infotext)
        assert mc_saved_prompt.install(module)
        first = module.create_infotext
        assert mc_saved_prompt.install(module)
        assert module.create_infotext is first
        assert mc_saved_prompt.uninstall(module)
        assert module.create_infotext is forges_create_infotext
        assert not mc_saved_prompt.uninstall(module)

    def test_the_wrapper_keeps_forges_signature_for_readers(self, host_module):
        assert inspect.signature(host_module.create_infotext) == \
            inspect.signature(forges_create_infotext)

    def test_a_host_without_the_function_is_not_wrapped(self):
        assert not mc_saved_prompt.install(types.SimpleNamespace())


# --------------------------------------------------------------------------- #
# A Txt2Img generation through the hook saves its tags at the end
# --------------------------------------------------------------------------- #


class TestThroughTheHook:
    """Stage 1 is given the tags in front, as it always was; the text saved for
    it has them at the end. Nothing is marked on the processing object any more:
    every generation's text moves (2026-11-23)."""

    def test_stage_one_gets_them_in_front_and_the_image_saves_them_at_the_end(
            self, script, store, host, host_module):
        p = generate(script, "a woman <lora:realfilter:1>", enabled=False)
        assert p.prompt == "<lora:realfilter:1> a woman"
        saved = host_module.create_infotext(processing_object([p.prompt]), [p.prompt],
                                            [7], [0])
        assert saved.splitlines()[:2] == ["a woman", "<lora:realfilter:1>"]

    def test_a_tag_from_a_literal_box_is_saved_at_the_end_too(self, script, store, host,
                                                              host_module):
        p = generate(script, "a woman", enabled=False, literal_positive="<lora:x:1>",
                     literal_negative="blue hat")
        saved = host_module.create_infotext(processing_object([p.prompt]), [p.prompt],
                                            [7], [0])
        assert saved.splitlines()[:2] == ["a woman blue hat", "<lora:x:1>"]

    def test_the_mark_is_gone_from_both_sides(self):
        import model_chain_krea_creative as creative_script

        for name in ("mark", "lifted", "MARK", "LIFTED_KEYS"):
            assert not hasattr(mc_saved_prompt, name), name
        assert not hasattr(creative_script, "_mark_lifted")


# --------------------------------------------------------------------------- #
# A recalled prompt has its tags at the end too
# --------------------------------------------------------------------------- #


class TestARecalledPromptEndsWithItsTags:
    """The user's second half (2026-11-06): *when recalling a png info, i would
    like those lora appended as well when brought back into prompt box*; and
    (2026-11-23) *All lora loaded from info text should appear appended to the
    prompt at the end, separated by a new line*, *regardless of the model
    (krea, moss, sdxl, etc)*. Forge's paste hands the parsed parameters to
    ``on_infotext_pasted`` before it fills a field, so that is where the box's
    text is decided -- for every image, whatever made it."""

    LIFTED = {"Krea Literal Command Count": "2", "Krea Literal Syntax Version": "1"}

    def test_an_image_saved_with_its_tags_in_front_recalls_them_at_the_end(self):
        params = {"Prompt": "<lora:a:1> <lora:b:0.5> a woman, red coat", **self.LIFTED}
        mc_saved_prompt.on_infotext_pasted("text", params)
        assert params["Prompt"] == "a woman, red coat\n<lora:a:1>\n<lora:b:0.5>"

    def test_the_recalled_box_gives_stage_one_what_the_image_was_made_from(self):
        saved = restored("a woman <lora:a:1>, red coat <lora:b:0.5>")
        assert saved.startswith("<lora:a:1> <lora:b:0.5>")
        params = {"Prompt": saved, **self.LIFTED}
        mc_saved_prompt.on_infotext_pasted("text", params)
        assert params["Prompt"].split("\n")[-2:] == ["<lora:a:1>", "<lora:b:0.5>"]
        assert stage_one(params["Prompt"]) == stage_one(saved)

    @pytest.mark.parametrize("model", ["krea2_dev", "moss_image_v1", "sdxl_base_1.0",
                                       "flux2_klein_9b", None])
    def test_an_image_without_the_hooks_record_recalls_them_at_the_end_too(self, model):
        """Img2Img, another tool, any model: the record of this extension's
        lift no longer decides anything."""
        params = {"Prompt": "<lora:a:1> a woman, <lora:b:0.5>, red coat"}
        if model:
            params["Model"] = model
        mc_saved_prompt.on_infotext_pasted("text", params)
        assert params["Prompt"] == "a woman, red coat\n<lora:a:1>\n<lora:b:0.5>"

    def test_tags_typed_at_the_end_on_one_line_are_given_a_line_each(self):
        params = {"Prompt": "a woman, red coat, <lora:a:1>, <lora:b:0.5>"}
        mc_saved_prompt.on_infotext_pasted("text", params)
        assert params["Prompt"] == "a woman, red coat\n<lora:a:1>\n<lora:b:0.5>"

    def test_a_prompt_without_tags_is_left_as_it_was(self):
        params = {"Prompt": "a woman,  red coat\n\nsecond line "}
        mc_saved_prompt.on_infotext_pasted("text", params)
        assert params["Prompt"] == "a woman,  red coat\n\nsecond line "

    def test_a_prompt_a_move_could_change_is_left_alone(self):
        params = {"Prompt": "<lora:a:1> a woman [[keep]]", **self.LIFTED}
        mc_saved_prompt.on_infotext_pasted("text", params)
        assert params["Prompt"] == "<lora:a:1> a woman [[keep]]"
        params = {"Prompt": "a woman -<lora:a:1>"}
        mc_saved_prompt.on_infotext_pasted("text", params)
        assert params["Prompt"] == "a woman -<lora:a:1>"

    def test_the_negative_prompt_is_left_where_it_was_typed(self):
        params = {"Prompt": "<lora:a:1> a woman", "Negative prompt": "<lora:n:1> blurry"}
        mc_saved_prompt.on_infotext_pasted("text", params)
        assert params["Negative prompt"] == "<lora:n:1> blurry"

    def test_nothing_odd_raises(self):
        for params in (None, {}, {"Prompt": None, **self.LIFTED}, {"Prompt": 3}):
            mc_saved_prompt.on_infotext_pasted("text", params)

    def test_the_extension_registers_it_with_forges_paste(self):
        import model_chain  # noqa: F401  -- imported for its registrations
        from modules import script_callbacks

        assert mc_saved_prompt.on_infotext_pasted in \
            script_callbacks.registered["infotext_pasted"]


# --------------------------------------------------------------------------- #
# What PNG Info shows and sends, a dropped image, and the paste hooks' order
# --------------------------------------------------------------------------- #

SAVED_PROMPT = "<lora:a:1> <lora:b:0.5> a woman in the rain, soft light"
MOVED_PROMPT = "a woman in the rain, soft light\n<lora:a:1>\n<lora:b:0.5>"
PARAMETER_LINE = ("Steps: 20, Sampler: Euler, CFG scale: 7, Seed: 1, Size: 832x1216, "
                  'Lora hashes: "a: 0123abcd, b: 4567ef01", Version: f2.0.1')
SAVED = f"{SAVED_PROMPT}\nNegative prompt: <lora:n:1> blurry, low quality\n{PARAMETER_LINE}"
MOVED = f"{MOVED_PROMPT}\nNegative prompt: <lora:n:1> blurry, low quality\n{PARAMETER_LINE}"
ITEMS = {"extras": "Postprocess upscale by: 2"}

FORGES_RE_PARAM = re.compile(r'\s*([\w\s\-\/]+):\s*("(?:\\.|[^\\"])+"|[^,]*)(?:,|$)')
"""Forge Neo's ``re_param``, copied from ``modules/infotext_utils.py``."""


def forges_plaintext_to_html(text, classname=None):
    """Forge Neo's ``modules/ui_common.py``."""
    content = "<br>\n".join(html.escape(x) for x in text.split("\n"))
    return f"<p class='{classname}'>{content}</p>" if classname else f"<p>{content}</p>"


def extras_module(saved, items=None, *, names_its_writer=True):
    """Forge Neo's ``modules/extras.py`` as far as PNG Info reads an image: its
    ``run_pnginfo`` line for line, over an image whose saved text is ``saved``
    and whose other text chunks are ``items``. Forge's module imports the HTML
    writer it uses (``plaintext_to_html``), so the module names it -- unless
    ``names_its_writer`` is false, a Forge whose PNG Info writes its HTML some
    way the module does not show."""
    module = types.SimpleNamespace()
    if names_its_writer:
        module.plaintext_to_html = forges_plaintext_to_html

    def read_info_from_image(image):
        return saved, dict(items or {})

    def run_pnginfo(image):
        if image is None:
            return "", "", ""

        geninfo, found = read_info_from_image(image)
        found = {**{"parameters": geninfo}, **found}

        info = ""
        for key, text in found.items():
            info += f"""<div>
            <p><b>{forges_plaintext_to_html(str(key))}</b></p>
            <p>{forges_plaintext_to_html(str(text))}</p>
        </div>""".strip() + "\n"

        if len(info) == 0:
            message = "Nothing found in the image."
            info = f"<div><p>{message}<p></div>"

        return "", geninfo, info

    module.run_pnginfo = run_pnginfo
    return module


def forges_parse(text, strip_whitespaces=True):
    """Forge Neo's ``parse_generation_parameters`` as far as it splits a saved
    text: the prompt, the negative prompt and the parameters' line, under
    either reading of its *strip whitespaces* setting."""
    *lines, lastline = text.strip().split("\n")
    if len(FORGES_RE_PARAM.findall(lastline)) < 3:
        lines.append(lastline)
        lastline = ""
    prompts, negatives, negative = [], [], False
    for line in lines:
        if strip_whitespaces:
            line = line.strip()
            if line.startswith("Negative prompt:"):
                line = line.replace("Negative prompt:", "").strip()
                negative = True
        elif line.lstrip().startswith("Negative prompt: "):
            line = line.replace("Negative prompt: ", "")
            negative = True
        (negatives if negative else prompts).append(line)
    return "\n".join(prompts), "\n".join(negatives), lastline


def callbacks_module(*hooks):
    """Forge Neo's ``script_callbacks`` as far as a paste runs its hooks: each
    in turn, one that raises reported and the next one run."""
    module = types.SimpleNamespace(hooks=list(hooks), reported=[])

    def infotext_pasted_callback(infotext, params):
        for hook in module.hooks:
            try:
                hook(infotext, params)
            except Exception as error:
                module.reported.append(error)

    module.infotext_pasted_callback = infotext_pasted_callback
    return module


def forges_paste(callbacks, text):
    """What Forge's paste does with a saved text before it fills a single field
    (``infotext_utils.connect_paste``): the text parsed, then the hooks run
    through the runner, which it reads off the module at the call."""
    prompt, negative, _ = forges_parse(text)
    params = {"Prompt": prompt, "Negative prompt": negative}
    callbacks.infotext_pasted_callback(text, params)
    return params


def hosts(saved=SAVED, items=ITEMS, *hooks):
    return {"extras": extras_module(saved, items),
            "images": types.SimpleNamespace(image_data=lambda data: (saved, None)),
            "script_callbacks": callbacks_module(mc_saved_prompt.on_infotext_pasted, *hooks)}


def only(**named):
    """``install_readers``' arguments with every host not named standing in empty."""
    return {"extras": types.SimpleNamespace(), "images": types.SimpleNamespace(),
            "script_callbacks": types.SimpleNamespace(), **named}


@pytest.fixture
def readers():
    """Forge's three readers of an image's text, wrapped as ``on_before_ui``
    wraps them, and put back afterwards."""
    named = hosts()
    assert mc_saved_prompt.install_readers(**named) == []
    yield types.SimpleNamespace(**named)
    mc_saved_prompt.uninstall_readers(**named)


class TestWhatPNGInfoShows:
    """The user's report once the saved and recalled text were moved: "I went
    straight to png info to load the settings of a previous image. The lora all
    showed up at the front of the prompt!" (2026-11-23). PNG Info shows the text
    an image holds, and an image saved before the move holds its tags in front."""

    def test_it_shows_the_prompt_with_its_tags_at_the_end_one_a_line(self, readers):
        _, _, shown = readers.extras.run_pnginfo(object())
        assert ("<p>a woman in the rain, soft light<br>\n&lt;lora:a:1&gt;<br>\n"
                "&lt;lora:b:0.5&gt;<br>\nNegative prompt:") in shown
        assert "&lt;lora:a:1&gt; &lt;lora:b:0.5&gt; a woman" not in shown

    def test_it_shows_exactly_what_forge_shows_for_the_same_image_saved_today(self, readers):
        """The whole answer is Forge's own for the moved text: the parameters'
        paragraph, the image's other text, the layout and the hidden box."""
        today = extras_module(MOVED, ITEMS)
        assert readers.extras.run_pnginfo(object()) == today.run_pnginfo(object())

    def test_its_send_to_buttons_read_the_moved_text(self, readers):
        _, sent, _ = readers.extras.run_pnginfo(object())
        assert sent == MOVED
        for strip in (True, False):
            assert forges_parse(sent, strip) == \
                (MOVED_PROMPT, *forges_parse(SAVED, strip)[1:])
        from modules import infotext_utils

        assert infotext_utils.parse_generation_parameters(sent) == \
            infotext_utils.parse_generation_parameters(SAVED)

    def test_sent_from_png_info_the_box_gets_the_moved_prompt(self, readers):
        _, sent, _ = readers.extras.run_pnginfo(object())
        assert forges_paste(readers.script_callbacks, sent)["Prompt"] == MOVED_PROMPT

    @pytest.mark.parametrize("saved", [
        None,
        "",
        "a woman in the rain, soft light",
        "Steps: 20, Sampler: Euler, CFG scale: 7",
        "<lora:a:1> a woman [[keep]]\nSteps: 20, Sampler: Euler, CFG scale: 7",
        "a woman -<lora:a:1>\nSteps: 20, Sampler: Euler, CFG scale: 7",
        '{"sui_image_params": {"prompt": "<lora:a:1> a cat", "steps": 20}}',
        f"{MOVED_PROMPT}\nSteps: 20, Sampler: Euler, CFG scale: 7",
    ])
    def test_an_image_with_nothing_to_move_is_shown_as_forge_shows_it(self, saved):
        ours, forge = extras_module(saved, ITEMS), extras_module(saved, ITEMS)
        mc_saved_prompt.install_readers(**only(extras=ours))
        assert ours.run_pnginfo(object()) == forge.run_pnginfo(object())
        assert ours.run_pnginfo(None) == ("", "", "")

    def test_a_tab_laid_out_otherwise_shows_what_it_showed_and_sends_the_moved_text(self):
        def run_pnginfo(image):
            return "", SAVED, f"<pre>{html.escape(SAVED)}</pre>"

        module = types.SimpleNamespace(run_pnginfo=run_pnginfo,
                                       plaintext_to_html=forges_plaintext_to_html)
        mc_saved_prompt.install_readers(**only(extras=module))
        assert module.run_pnginfo(object()) == \
            ("", MOVED, f"<pre>{html.escape(SAVED)}</pre>")

    def test_a_tab_whose_module_names_no_html_writer_is_read_with_forge_neos(self):
        module = extras_module(SAVED, ITEMS, names_its_writer=False)
        mc_saved_prompt.install_readers(**only(extras=module))
        assert module.run_pnginfo(object()) == extras_module(MOVED, ITEMS).run_pnginfo(object())

    def test_a_surprise_shows_what_forge_showed(self, readers, monkeypatch):
        def surprise(text):
            raise RuntimeError("a surprise")

        monkeypatch.setattr(mc_saved_prompt, "moved_text", surprise)
        assert readers.extras.run_pnginfo(object()) == \
            extras_module(SAVED, ITEMS).run_pnginfo(object())


class TestTheSavedTextMoved:
    """:func:`mc_saved_prompt.moved_text`: the prompt found as Forge's parse finds
    it, moved, and every other character of the text kept."""

    def test_only_the_prompts_lines_move(self):
        assert mc_saved_prompt.moved_text(SAVED) == MOVED

    def test_the_whitespace_around_the_text_is_kept(self):
        assert mc_saved_prompt.moved_text(" \n" + SAVED + "\n\n") == " \n" + MOVED + "\n\n"

    def test_a_prompt_over_several_lines_moves_as_one(self):
        saved = "first <lora:a:1>\nsecond\n\nthird\nNegative prompt: x\n" + PARAMETER_LINE
        assert mc_saved_prompt.moved_text(saved) == \
            "first\nsecond\n\nthird\n<lora:a:1>\nNegative prompt: x\n" + PARAMETER_LINE

    def test_the_negative_prompt_is_found_as_forge_finds_it(self):
        saved = "<lora:a:1> a cat\n   Negative prompt: <lora:n:1> dull\n" + PARAMETER_LINE
        moved = mc_saved_prompt.moved_text(saved)
        assert moved == "a cat\n<lora:a:1>\n   Negative prompt: <lora:n:1> dull\n" + PARAMETER_LINE
        for strip in (True, False):
            assert forges_parse(moved, strip)[1:] == forges_parse(saved, strip)[1:]

    def test_a_last_line_of_fewer_than_three_parameters_is_prompt_as_forge_reads_it(self):
        saved = "<lora:a:1> a cat\nSteps: 20, Sampler: Euler"
        moved = mc_saved_prompt.moved_text(saved)
        assert moved == "a cat\nSteps: 20, Sampler: Euler\n<lora:a:1>"
        assert forges_parse(moved)[0] == extra_networks.last(forges_parse(saved)[0])

    @pytest.mark.parametrize("saved", [
        "", "   ", PARAMETER_LINE, "Negative prompt: <lora:n:1> dull\n" + PARAMETER_LINE,
        "a cat, no tags\n" + PARAMETER_LINE, MOVED,
    ])
    def test_a_text_with_nothing_to_move_is_the_same_string(self, saved):
        assert mc_saved_prompt.moved_text(saved) is saved

    def test_a_json_record_is_left_whole_and_a_wildcard_is_not_json(self):
        record = '[{"prompt": "<lora:a:1> a cat"}]'
        assert mc_saved_prompt.moved_text(record) is record
        assert mc_saved_prompt.recalled(record) is record
        # Read as Forge reads a saved text, this one's last line is parameters
        # (three "key: value" in a string), and what is left is not JSON.
        record = ('{\n  "prompt": "<lora:a:1> a cat",\n'
                  '  "note": "Steps: 20, Sampler: Euler, CFG scale: 7"}')
        assert len(FORGES_RE_PARAM.findall(record.split("\n")[-1])) >= 3
        assert mc_saved_prompt.moved_text(record) is record
        assert mc_saved_prompt.moved_text("{red|blue} <lora:a:1> coat\n" + PARAMETER_LINE) == \
            "{red|blue} coat\n<lora:a:1>\n" + PARAMETER_LINE

    def test_twice_is_once(self):
        once = mc_saved_prompt.moved_text(SAVED)
        assert mc_saved_prompt.moved_text(once) == once

    def test_what_is_not_text_is_handed_back(self):
        for value in (None, 3, b"<lora:a:1> a"):
            assert mc_saved_prompt.moved_text(value) is value

    def test_forges_own_pattern_is_read_and_the_copy_is_forges(self, monkeypatch):
        assert mc_saved_prompt.PARAMETERS.pattern == FORGES_RE_PARAM.pattern
        from modules import infotext_utils

        never = re.compile(r"(?!)")
        monkeypatch.setattr(infotext_utils, "re_param", never, raising=False)
        assert mc_saved_prompt.moved_text("<lora:a:1> a cat\n" + PARAMETER_LINE) == \
            "a cat\n" + PARAMETER_LINE + "\n<lora:a:1>"

    def test_an_old_image_reads_as_the_same_image_saved_today(self):
        """Over a spread of prompts the hook can save: PNG Info's text for an image
        saved before the move is the text the wrapper writes for it today."""
        rng = random.Random(20261123)
        words = ["a", "cat", "neon", "street,", ",", " ", "  ", "\n", "BREAK", "(soft:1.2)",
                 "[a:b:0.5]", "-", "+", "film grain", "<lora:x:0.85>", "<lora:y:-1>",
                 "<lyco:z:1>", "<lora:w:1:0.5>", "__wild__", "{a|b}"]
        checked = 0
        for _ in range(2500):
            prompt = " ".join(rng.choice(words) for _ in range(rng.randint(0, 9)))
            positive = rng.choice(["", "<lora:p:1>", "masterpiece, <lora:q:0.5>", "best,"])
            negative = rng.choice(["", "blue hat", "film grain <lora:n:1>"])
            today = restored(prompt, positive, negative)
            unwanted = rng.choice(["", "blurry", "<lora:n:1> dull, flat"])
            tail = f"\nNegative prompt: {unwanted}" if unwanted else ""
            old = f"{today}{tail}\n{PARAMETER_LINE}".strip()
            new = mc_saved_prompt.rewrite(old, today)
            assert mc_saved_prompt.moved_text(old) == new, (prompt, positive, negative)
            checked += new != old
        assert checked > 1000, "the spread moved too few texts to prove anything"


class TestADroppedImage:
    """An image dropped on the prompt box puts its whole saved text there
    (``images.image_data``), for the box's paste button to read."""

    def test_the_box_gets_the_text_with_its_tags_at_the_end(self, readers):
        assert readers.images.image_data(b"png") == (MOVED, None)

    def test_the_paste_button_then_reads_the_moved_prompt(self, readers):
        text, _ = readers.images.image_data(b"png")
        assert forges_paste(readers.script_callbacks, text)["Prompt"] == MOVED_PROMPT

    def test_what_is_not_an_images_text_is_handed_back_as_it_is(self):
        skip = object()
        answers = iter([(skip, None), ("a cat, no tags", None), None])
        images = types.SimpleNamespace(image_data=lambda data: next(answers))
        mc_saved_prompt.install_readers(**only(images=images))
        assert images.image_data(b"x") == (skip, None)
        assert images.image_data(b"x") == ("a cat, no tags", None)
        assert images.image_data(b"x") is None


class TestTheRecallRunsLast:
    """Forge runs the paste hooks in the order extensions registered them, so a
    hook of another extension that writes the prompt back from the text it read
    would put the tags in front again after this one had moved them."""

    @staticmethod
    def writes_it_back(infotext, params):
        params["Prompt"] = forges_parse(infotext)[0]

    def test_a_later_hook_that_writes_the_prompt_back_cannot_put_the_tags_in_front(self):
        named = hosts(SAVED, ITEMS, self.writes_it_back)
        mc_saved_prompt.install_readers(**named)
        assert forges_paste(named["script_callbacks"], SAVED)["Prompt"] == MOVED_PROMPT

    def test_a_later_hook_that_raises_does_not_stop_the_move(self):
        def raises(infotext, params):
            params["Prompt"] = SAVED_PROMPT
            raise RuntimeError("another extension's surprise")

        named = hosts(SAVED, ITEMS, raises)
        mc_saved_prompt.install_readers(**named)
        assert forges_paste(named["script_callbacks"], SAVED)["Prompt"] == MOVED_PROMPT
        assert len(named["script_callbacks"].reported) == 1

    def test_the_runner_called_by_keyword_moves_them_too(self):
        named = hosts(SAVED, ITEMS, self.writes_it_back)
        mc_saved_prompt.install_readers(**named)
        params = {"Prompt": SAVED_PROMPT}
        named["script_callbacks"].infotext_pasted_callback(infotext=SAVED, params=params)
        assert params["Prompt"] == MOVED_PROMPT

    def test_the_runner_answers_what_forges_answered(self):
        module = types.SimpleNamespace(infotext_pasted_callback=lambda infotext, params: "said")
        mc_saved_prompt.install_readers(**only(script_callbacks=module))
        assert module.infotext_pasted_callback("text", {"Prompt": "a"}) == "said"
        assert module.infotext_pasted_callback("text", None) == "said"


class TestTheReadersAreWrappedBeforeTheUI:
    """PNG Info and the prompt row take their functions as the UI is built, so
    the wrappers go on before it: ``on_before_ui``, every build."""

    def test_the_extension_wraps_them_before_the_ui_is_built(self):
        import model_chain  # noqa: F401  -- imported for its registrations
        from modules import script_callbacks

        assert mc_saved_prompt.on_before_ui in script_callbacks.registered["before_ui"]

    def test_wrapping_twice_wraps_once_and_unwrapping_puts_forges_back(self):
        named = hosts()
        originals = {host: getattr(named[host], name)
                     for host, name, _, _ in mc_saved_prompt.READERS}
        assert mc_saved_prompt.install_readers(**named) == []
        first = {host: getattr(named[host], name) for host, name, _, _ in mc_saved_prompt.READERS}
        assert all(first[host] is not originals[host] for host in first)
        assert mc_saved_prompt.install_readers(**named) == []
        for host, name, _, _ in mc_saved_prompt.READERS:
            assert getattr(named[host], name) is first[host]
            assert getattr(named[host], name).__wrapped__ is originals[host]
        mc_saved_prompt.uninstall_readers(**named)
        for host, name, _, _ in mc_saved_prompt.READERS:
            assert getattr(named[host], name) is originals[host]
        mc_saved_prompt.uninstall_readers(**named)
        for host, name, _, _ in mc_saved_prompt.READERS:
            assert getattr(named[host], name) is originals[host]

    def test_the_wrappers_keep_forges_signatures(self, readers):
        named = hosts()
        for host, name, _, _ in mc_saved_prompt.READERS:
            assert inspect.signature(getattr(getattr(readers, host), name)) == \
                inspect.signature(getattr(named[host], name))

    def test_a_host_without_them_is_named_once_and_nothing_raises(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="model_chain"):
            mc_saved_prompt.on_before_ui()
        lines = [record.getMessage() for record in caplog.records
                 if "LoRA tags where the image holds them" in record.getMessage()]
        assert len(lines) == 1
        for _, _, _, label in mc_saved_prompt.READERS:
            assert label in lines[0]
        from modules import images, script_callbacks

        assert not hasattr(images, "image_data")
        assert not hasattr(script_callbacks, "infotext_pasted_callback")

    def test_the_extensions_unload_puts_forges_back(self, monkeypatch):
        import model_chain
        import modules

        named = hosts()
        originals = {host: getattr(named[host], name)
                     for host, name, _, _ in mc_saved_prompt.READERS}
        for host in named:
            module = types.ModuleType(f"modules.{host}")
            vars(module).update(vars(named[host]))
            named[host] = module
            monkeypatch.setitem(sys.modules, f"modules.{host}", module)
            monkeypatch.setattr(modules, host, module, raising=False)
        monkeypatch.setattr(model_chain.mc_memory, "release_all", lambda: None)
        monkeypatch.setattr(model_chain.mc_llm_runtime, "shutdown", lambda: None)
        monkeypatch.setattr(model_chain.mc_broker, "clear", lambda: None)
        mc_saved_prompt.on_before_ui()
        for host, name, _, _ in mc_saved_prompt.READERS:
            assert getattr(named[host], name).__wrapped__ is originals[host]

        model_chain._on_script_unloaded()

        for host, name, _, _ in mc_saved_prompt.READERS:
            assert getattr(named[host], name) is originals[host]
