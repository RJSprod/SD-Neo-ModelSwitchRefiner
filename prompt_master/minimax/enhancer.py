"""How WanGP asks for an H3 prompt, over the server this application runs.

Every word of instruction comes from ``prompt_enhancer.py``, vendored verbatim
beside this file. What is here is only the calling convention around it, and it
is WanGP's rather than ours — see ``UPSTREAM_SOURCE.txt`` for the call sites
each part was read from. Four of those decisions are worth stating, because
they are what "the same enhancer" actually means:

*Which instructions apply is decided by the variant and by the image, and by
nothing else.* WanGP reads ``text_prompt_enhancer_instructions`` when the
generation has no image and ``video_prompt_enhancer_instructions`` when it has
one, from the model definition of whichever H3 model is loaded — the four
combinations in ``INSTRUCTIONS``. Here the variant is a drop-down instead of a
loaded model, which is the one thing this port adds to the choice.

*The image is described, not shown.* WanGP's enhancer LLM never sees pixels: a
captioner is run first and its paragraph is handed to the enhancer as an
``image_caption:`` line. That is reproduced exactly, down to the instruction the
captioner is given, because an H3 image prompt is written *about* a caption and
a model shown the still instead would be answering a different question.

*The user turn is two labelled lines, not a chat message.* ``user_prompt:`` and
``image_caption:`` are what the enhancer was trained against; the whole of the
request is the system message plus those.

*A prompt may carry its own instructions.* ``@`` in the text splits off a suffix
that is appended to the system message with WanGP's own preamble, and ``@@``
splits off one that replaces the system message entirely. It is undocumented in
the WanGP UI and reproduced here because a prompt written for WanGP may use it
and would otherwise silently ask for a video about its own instructions.

The one departure: WanGP folds the finished prompt's newlines into spaces,
because its prompt box holds one prompt per line. An H3 prompt is three or six
blank-line-separated fields, so the newlines are kept here — WanGP keeps them
too, in the multi-prompt mode where a prompt may span lines.
"""

from __future__ import annotations

import re

from .prompt_enhancer import (FL2VA_IMAGE_SYSTEM_PROMPT, FL2VA_PROMPT_INFOS,
                              FL2VA_TEXT_SYSTEM_PROMPT, REF2VA_IMAGE_SYSTEM_PROMPT,
                              REF2VA_PROMPT_INFOS, REF2VA_TEXT_SYSTEM_PROMPT)

# The two H3 architectures the vendored instructions cover. WanGP's own keys are
# the model types minimax_h3_fl2va and minimax_h3_ref2va; the halves that name
# the variant are kept, because the pruned checkpoints differ in weights and not
# in a single word of prompting.
FL2VA = "fl2va"
REF2VA = "ref2va"

# What the drop-down offers. WanGP has no such control — the variant is whichever
# H3 model is loaded — so these labels are this application's own, and are the
# only strings on this page that are.
VARIANTS = ((FL2VA, "FL2VA — from text or a frame"),
            (REF2VA, "Ref2VA — from reference material"))

# ``prompt_enhancer_def["labels"]`` from the H3 handler: what WanGP calls each of
# the four generations this page can run. Shown under the image row, so that the
# thing about to happen is named the way the tool it came from names it.
LABELS = {
    (FL2VA, False): "Write an H3 Prompt from Text",
    (FL2VA, True): "Write an H3 Prompt from Text + Start Image",
    (REF2VA, False): "Write an H3 Reference Prompt from Text",
    (REF2VA, True): "Write an H3 Reference Prompt from Text + First Reference Image",
}

# ``prompt_enhancer_button_label``.
BUTTON_LABEL = "Write H3 Prompt"

# The four system prompts, by what WanGP would have read them from:
# text_prompt_enhancer_instructions with no image, video_prompt_enhancer_
# instructions with one, on the FL2VA or the Ref2VA model definition.
INSTRUCTIONS = {
    (FL2VA, False): FL2VA_TEXT_SYSTEM_PROMPT,
    (FL2VA, True): FL2VA_IMAGE_SYSTEM_PROMPT,
    (REF2VA, False): REF2VA_TEXT_SYSTEM_PROMPT,
    (REF2VA, True): REF2VA_IMAGE_SYSTEM_PROMPT,
}

# text/video_prompt_enhancer_max_tokens. Ref2VA is given twice the room because
# it writes six sections rather than three, one of which is a 350-500 word
# timeline.
MAX_TOKENS = {FL2VA: 1024, REF2VA: 2048}

# The prompt-structure guide WanGP shows beside the prompt box.
INFOS = {FL2VA: FL2VA_PROMPT_INFOS, REF2VA: REF2VA_PROMPT_INFOS}

# server_config prompt_enhancer_temperature / prompt_enhancer_top_p, and
# prompt_enhancer_randomize_seed, which is why the seed is drawn per run rather
# than being a setting on the page.
TEMPERATURE = 0.6
TOP_P = 0.9

# The captioner's instruction, verbatim from WanGP's Qwen3.5-VL caption pass,
# with the numbers that go with it: 128 new tokens, and no sampling at all
# (do_sample=False, which is temperature 0 here). A caption is a description of
# what is in the picture, and there is nothing for a sampler to be creative
# about.
CAPTION_INSTRUCTION = ("Describe this image accurately in one concise paragraph, focusing on "
                       "the main subject, setting, and notable objects. Output only the "
                       "description.")
CAPTION_MAX_TOKENS = 128
CAPTION_TEMPERATURE = 0.0
CAPTION_TOP_P = 1.0

# What ``@`` puts between the instructions and the user's addition to them.
SUFFIX_PREAMBLE = ("Follow these additional user instructions with higher priority if they "
                   "conflict with the guidance above:")

# A reply that leaked its own reasoning, or wrapped the answer in a fence the
# instructions twice forbid. WanGP has no equivalent — its enhancer runs a model
# it chose, where this one runs whichever model is installed — and both patterns
# put text in the box that must not be pasted into H3.
_THINKING = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
_FENCED = re.compile(r"\A```[^\n]*\n(?P<body>.*?)\n?```\Z", re.DOTALL)


def variant_of(name: str) -> str:
    """The variant key, or FL2VA — which is the H3 model WanGP lists first."""
    return name if name in (FL2VA, REF2VA) else FL2VA


def instructions(variant: str, has_image: bool) -> str:
    """The system prompt for this generation, as the model definition names it."""
    return INSTRUCTIONS[(variant_of(variant), bool(has_image))]


def label(variant: str, has_image: bool) -> str:
    """WanGP's name for this generation."""
    return LABELS[(variant_of(variant), bool(has_image))]


def max_tokens(variant: str) -> int:
    return MAX_TOKENS[variant_of(variant)]


def infos(variant: str) -> str:
    return INFOS[variant_of(variant)]


def split_system_suffix(prompt: str) -> tuple[str, str, bool]:
    """Split a prompt into what to write about and what to write it under.

    ``body @@ instructions`` replaces the H3 instructions with the suffix;
    ``body @ instructions`` adds the suffix to them. The double form is looked
    for first, because a single ``@`` matches inside a double one.
    """
    prompt = str(prompt or "").strip()
    body, separator, suffix = prompt.partition("@@")
    if separator == "@@":
        return body.strip(), suffix.strip(), True
    body, separator, suffix = prompt.partition("@")
    if separator == "":
        return prompt, "", False
    return body.strip(), suffix.strip(), False


def merge_system(system: str, suffix: str, replace: bool = False) -> str:
    """The system message: the instructions, plus whatever ``@`` added to them."""
    system = str(system or "").rstrip()
    suffix = str(suffix or "").strip()
    if not suffix:
        return system
    if replace:
        return suffix
    return f"{system}\n{SUFFIX_PREAMBLE}\n{suffix}"


def user_content(prompt: str, image_caption: str | None = None) -> str:
    """The user turn: one labelled line, or two when there is a picture."""
    if image_caption is None:
        return f"user_prompt: {prompt}"
    return f"user_prompt: {prompt}\nimage_caption: {image_caption}"


def messages(prompt: str, *, variant: str = FL2VA,
             image_caption: str | None = None) -> list[dict[str, str]]:
    """The whole request: the instructions, and the request under them."""
    body, suffix, replace = split_system_suffix(prompt)
    system = merge_system(instructions(variant, image_caption is not None), suffix, replace)
    return [{"role": "system", "content": system},
            {"role": "user", "content": user_content(body, image_caption)}]


def caption_messages(image_data_url: str) -> list[dict[str, object]]:
    """The captioner's turn: the picture, then what to say about it."""
    return [{"role": "user",
             "content": [{"type": "image_url", "image_url": {"url": image_data_url}},
                         {"type": "text", "text": CAPTION_INSTRUCTION}]}]


def clean(text: str) -> str:
    """The finished prompt, with what is not part of it taken off.

    Newlines are kept: the fields of an H3 prompt are separated by them.
    """
    text = _THINKING.sub("", str(text or "")).strip()
    fenced = _FENCED.match(text)
    return (fenced.group("body") if fenced else text).strip()
