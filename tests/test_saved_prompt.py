"""LoRA tags at the end of an image's saved prompt (mc_saved_prompt).

The user's report: an image made here, read back in PNG Info, has every LoRA at
the front of its prompt, and they want them at the end. The front is this
extension's doing -- every bare tag is lifted out of the Txt2Img prompt and put
back in front of it before Forge sees it -- so the fix moves them in the saved
text alone. Three things have to hold, and each class below is one of them:

* the move itself: tags at the end, in the order they had, the words tidied
  as a paste of them would tidy them;
* what makes the move safe: pasting the saved text back gives Stage 1 the same
  prompt as pasting the text Forge would have saved before, for every prompt
  the hook can produce;
* the wiring: Forge's ``create_infotext`` is wrapped without being made to
  compute anything differently, only a generation the hook lifted tags in is
  rewritten, and the hook marks exactly those.
"""

from __future__ import annotations

import inspect
import random
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
    def test_tags_go_to_the_end_in_the_order_they_had(self):
        line = "<lora:realfilter_v2:0.85> <lora:contrast:-6> a woman in the rain, soft light"
        assert extra_networks.last(line) == (
            "a woman in the rain, soft light <lora:realfilter_v2:0.85> <lora:contrast:-6>")

    def test_the_words_are_tidied_as_a_paste_would_tidy_them(self):
        line = "<lora:a:1>, a cat, <lora:b:0.5>, on a mat"
        assert extra_networks.last(line) == "a cat, on a mat <lora:a:1> <lora:b:0.5>"

    def test_every_kind_the_hook_lifts_moves_and_case_does_not_matter(self):
        line = "<LoRA:a:1> <lyco:b:1> <hypernet:c:1> words"
        assert extra_networks.last(line) == "words <LoRA:a:1> <lyco:b:1> <hypernet:c:1>"

    def test_a_line_without_tags_is_the_same_string(self):
        line = "a quiet street,  at dusk "
        assert extra_networks.last(line) is line

    def test_a_line_of_tags_alone_is_left_as_it_is(self):
        assert extra_networks.last("<lora:a:1> <lora:b:2>") == "<lora:a:1> <lora:b:2>"

    def test_a_line_that_still_holds_brackets_is_left_as_it_is(self):
        line = "<lora:a:1> [[__wildcard__]] words"
        assert extra_networks.last(line) is line

    @pytest.mark.parametrize("sign", ["-", "+"])
    def test_a_tag_right_after_a_sign_leaves_the_line_as_it_is(self, sign):
        line = f"<lora:a:1> words {sign}<lora:b:1>"
        assert extra_networks.last(line) is line

    def test_lines_are_kept(self):
        line = "<lora:a:1> first line\nsecond line"
        assert extra_networks.last(line) == "first line\nsecond line <lora:a:1>"


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
                tags = extra_networks._TAG.findall
                assert tags(moved) == tags(today)
                assert moved.endswith(tags(today)[-1])
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


def processing_object(prompts, *, lifted=True, negatives=None, batch_size=1):
    p = types.SimpleNamespace(
        batch_size=batch_size, all_negative_prompts=negatives or [""] * len(prompts),
        main_prompt=prompts[0], main_negative_prompt=(negatives or [""])[0],
        all_hr_prompts=list(prompts), extra_generation_params={"Hires prompt": get_hr_prompt})
    if lifted:
        mc_saved_prompt.mark(p)
    return p


@pytest.fixture
def host_module():
    module = types.SimpleNamespace(create_infotext=forges_create_infotext)
    assert mc_saved_prompt.install(module)
    yield module
    mc_saved_prompt.uninstall(module)


FRONT = "<lora:a:1> <lora:b:0.5> a woman in the rain"


class TestTheWrapper:
    def test_a_lifted_generations_prompt_line_ends_with_its_tags(self, host_module):
        p = processing_object([FRONT], negatives=["<lora:neg:1> blurry"])
        text = host_module.create_infotext(p, [FRONT], [7], [0])
        assert text.splitlines() == [
            "a woman in the rain <lora:a:1> <lora:b:0.5>",
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

    def test_a_generation_the_hook_did_not_lift_is_left_as_forge_wrote_it(self,
                                                                          host_module):
        p = processing_object([FRONT], lifted=False)
        assert host_module.create_infotext(p, [FRONT], [7], [0]) == \
            forges_create_infotext(p, [FRONT], [7], [0])

    def test_the_index_is_forges_and_so_is_the_main_prompt(self, host_module):
        prompts = ["<lora:a:1> zero", "<lora:b:1> one", "<lora:c:1> two", "<lora:d:1> three"]
        p = processing_object(prompts, batch_size=2)
        third = host_module.create_infotext(p, prompts, [0, 1, 2, 3], [0] * 4,
                                            iteration=1, position_in_batch=0)
        assert third.splitlines()[0] == "two <lora:c:1>"
        second = host_module.create_infotext(p, prompts, [0, 1, 2, 3], [0] * 4, index=1)
        assert second.splitlines()[0] == "one <lora:b:1>"
        p.main_prompt = "<lora:g:1> the grid"
        grid = host_module.create_infotext(p, prompts, [0, 1, 2, 3], [0] * 4,
                                           use_main_prompt=True)
        assert grid.splitlines()[0] == "the grid <lora:g:1>"

    def test_a_prompt_with_nothing_after_it_still_moves(self):
        """Forge strips the whole text, so a prompt ending in a space and
        followed by nothing is saved without that space."""
        module = types.SimpleNamespace(
            create_infotext=lambda p, all_prompts, *a, **k: f"{all_prompts[0]}".strip())
        assert mc_saved_prompt.install(module)
        p = processing_object([FRONT + " "])
        assert module.create_infotext(p, [FRONT + " "], [7], [0]) == \
            "a woman in the rain <lora:a:1> <lora:b:0.5>"

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
# The hook marks exactly the generations it lifted tags in
# --------------------------------------------------------------------------- #


class TestTheHookMarks:
    def test_a_bare_tag_in_the_prompt_marks_the_generation(self, script, store, host):
        p = generate(script, "a woman <lora:realfilter:1>", enabled=False)
        assert p.prompt == "<lora:realfilter:1> a woman"
        assert mc_saved_prompt.lifted(p)

    def test_a_tag_in_a_literal_box_marks_it(self, script, store, host):
        p = generate(script, "a woman", enabled=False, literal_positive="<lora:x:1>",
                     literal_negative="blue hat")
        assert mc_saved_prompt.lifted(p)

    def test_a_prompt_with_no_tag_anywhere_is_not_marked(self, script, store, host):
        p = generate(script, "a woman [[__wildcard__]]", enabled=False,
                     literal_positive="masterpiece", literal_negative="blue hat")
        assert not mc_saved_prompt.lifted(p)
