"""The LTX prompt writer, for LTX 2.3 and 2.5: instructions and calling convention, no UI.

Another extension -- Mini Paint NEO's Clipboard tab, sending to WanGP's LTX 2.3
and LTX 2.5 Distilled models -- asks :mod:`mc_llm_api` for an LTX prompt. This
module is everything about how that prompt is asked for: the two system prompts,
the user turn, the sampler, the cleaning. The run itself is
:func:`mc_llm_sessions.ltx`, over the same llama-server and under the same
workload lock as every other mode.

Where the instructions come from
--------------------------------
Lightricks ship a prompt enhancer with LTX-2 (``github.com/Lightricks/LTX-2``,
read at commit ``9ec55f9f22798a3198d9c923856824821bc3317e``, 2026-10-02). LTX
2.3 runs it on its own text encoder, Gemma 3, under two system prompts:
``packages/ltx-core/src/ltx_core/text_encoders/gemma/encoders/prompts/
gemma3_i2v_system_prompt.txt`` (a first frame and a request) and
``gemma3_t2v_system_prompt.txt`` (a request alone). The ``gemma4_*`` pair beside
them is LTX 2.5's and is not used here, for LTX 2.5 either (see below). The
repository's README adds the
prompting guide the structure below follows: one flowing paragraph, start with
the main action, then movements, appearances, environment, camera, light and
colour, and changes, literal and precise, within about 200 words.

The two prompts here are written from those, and depart from them on purpose,
because what was asked for is a writer *without bias*:

* no default style. The text-to-video original says "Default to
  cinematic-realistic if unspecified"; here a style is written only when the
  user names one, and with a first frame the image is the style;
* no invented camera. Both originals already say so, and it is kept;
* no added mood, genre, era or aesthetic, and no promotional adjectives -- the
  originals' "restrained language" rule, extended to everything a writer can
  slip in;
* what is left open is filled with plain, ordinary detail rather than "invent
  concrete details", which in practice meant invented drama;
* structured input -- labelled fields, lists, numbered beats, dialogue lines --
  is turned into the same paragraph with nothing dropped, which the originals do
  not mention.

Everything else -- the image read first and never contradicted, only what
changes from the first frame described, present-progressive verbs, a
chronological flow, the soundtrack woven into the action, quoted dialogue kept
word for word, nothing that cannot be seen or heard, one paragraph and no
headings -- is Lightricks' own guidance.

The calling convention is Lightricks' too: the system prompt, then one user
turn. With a first frame that turn is the picture and then
``User Raw Input Prompt: <request>.``; without one it is ``user prompt:
<request>`` (``base_encoder.enhance_i2v`` and ``enhance_t2v``). The picture is
*shown* to the model -- a vision request, not a caption pass first -- which is
how Lightricks' own enhancer sees it and what the user asked for.

LTX 2.5: the same writer
------------------------
WanGP's LTX 2.5 Distilled takes what its LTX 2.3 Distilled takes -- a prompt, a
first frame, a last frame -- and Mini Paint's Clipboard sends to both. Its
prompts are written under exactly these instructions: the user asked for LTX 2.5
to keep the LTX 2.3 system prompt, so there is one pair of instructions for
both, one default, and one override on the caller's side. Lightricks'
``gemma4_*`` prompts for LTX 2.5 are not used. What the model changes is only
what a request is called -- on the console, in its status line and in Prompt
Studio's history -- so a 2.5 prompt never says it was written for 2.3
(:data:`MODELS`, :func:`model_name`).

One paragraph, always
---------------------
WanGP reads an LTX prompt one line per prompt unless told otherwise, so a reply
in two paragraphs would be two videos, or two sliding windows. :func:`clean`
folds whatever came back onto one line; the instructions ask for one paragraph
anyway, and this is what makes that a property rather than a hope.
"""

from __future__ import annotations

import re

LTX23 = "ltx23"
"""The writer's name on the external API, beside MiniMax's ``fl2va`` and ``ref2va``."""

LTX25 = "ltx25"
"""LTX 2.5, written for under the same instructions as LTX 2.3 (see above)."""

MODELS = (LTX23, LTX25)
"""The LTX models this writer writes for, by the external API's names."""

MODEL_NAMES = {LTX23: "LTX 2.3", LTX25: "LTX 2.5"}
"""What the console, the status line and Prompt Studio call each one."""

LABEL = "LTX 2.3 and 2.5 — from text or a first frame"
"""What a caller may show for it: one set of instructions for both models."""

KIND = "ltx"
"""The kind of request on the external queue."""

STATUS_TEXT = "Writing the {name} prompt"
STATUS_IMAGE = "Looking at the first frame and writing the {name} prompt"

IMAGE_SYSTEM_PROMPT = """\
You write prompts for LTX 2.3, an image-to-video model that also generates the soundtrack. You are given an image, which is the exact first frame of the video, and the user's raw input. The input may be a few loose words, a full description, or structured notes (labelled fields, lists, numbered beats, dialogue lines, timings). Write the one prompt that turns this first frame into the video the user asked for.

Guidelines:
- Look at the image first: the subjects, how they look and where they are, the setting, the light and the visual style, exactly as they are. Everything you write must stay consistent with the image.
- Follow the user's input completely. Include every requested action, motion, camera instruction, sound, line of dialogue and detail, and drop nothing. Turn labels, lists and numbered beats into flowing prose in the order the events happen; a timing becomes the order of events, not a timestamp. If the input conflicts with the image, follow the input and describe the change happening from the first frame, keeping everything else as the image shows it.
- Describe what happens, not what is already there. The first frame already shows the scene, so do not describe it again at length. Name each subject briefly and exactly as it appears ("the man in the grey coat") so it is clear who acts. A description that contradicts the image causes cuts and jumps in the video.
- Start directly with the main action in one sentence. Then give the movements and gestures in the order they happen, then any change or event the user asked for.
- Use active, present-progressive verbs ("is walking", "turns", "lifts"). If the input asks for no particular action, describe the small, natural movements the subjects in the image would make.
- Keep a strict chronological flow with plain connectors such as "as", "then", "while" and "a moment later".
- Sound: describe the soundtrack the scene produces, woven into the action where it happens rather than collected at the end: the ambient sound of the place, the sounds the actions make, and speech or music only when the user asks for them. Be specific ("footsteps on wet pavement"), not vague ("ambient sound"), and match the loudness to the action.
- Speech, only when the user asks for speech, talking, singing or a conversation: give the exact words in quotes, and say who speaks and how the voice sounds. Keep the user's own dialogue word for word, correcting only typos, in the language the user wrote it in, and name that language if it is not English. If the user asks for speech without giving the words, write a short line that fits the scene.
- Camera: describe camera movement, angles or framing only when the user asks for them. Do not add any of your own.
- Style: do not add a style; the image sets the look. Only if the user names a style, begin with "Style: <style>," and then the description.
- Describe only what can be seen and heard: no smell, taste, touch or thoughts. Do not interpret emotions or intentions; describe the visible expression or action instead.
- Write plainly. Use neutral, restrained wording, with no dramatic or promotional adjectives, and add no mood, genre, era or aesthetic that neither the user nor the image gives.
- No timestamps and no scene cuts unless the user asks for them.

Output format (strict):
- One single paragraph of natural English prose, roughly 60 to 200 words: shorter for a simple action, longer only when the user asked for more.
- No titles, headings, labels, lists, code fences or Markdown. Do not begin with "The scene opens", "The video starts", "In this video" or the like, and do not begin with a punctuation mark.
- Output only the prompt. Never ask a question or explain. If the input cannot be turned into a video description, return it unchanged.
"""

TEXT_SYSTEM_PROMPT = """\
You write prompts for LTX 2.3, a text-to-video model that also generates the soundtrack. You are given the user's raw input: a few loose words, a full description, or structured notes (labelled fields, lists, numbered beats, dialogue lines, timings). Expand it into one complete LTX 2.3 prompt.

Guidelines:
- Follow the user's input completely. Include every requested subject, action, setting, style, camera instruction, sound, line of dialogue and detail, contradict nothing, and drop nothing. Turn labels, lists and numbered beats into flowing prose in the order the events happen; a timing becomes the order of events, not a timestamp.
- Build the prompt in this order, as one flowing description: the main action in one sentence; the specific movements and gestures; the appearance of each person or object (for people: clothing, hair, build and visible expression); the setting and background; the camera, only if the user asked for it; the light and colours; then any change or event that happens.
- Where the video has to show something the input leaves open, such as what someone wears, what the place looks like or where the light comes from, fill it with plain, ordinary detail that fits what the user gave. Do not add people, objects, events, a genre, a mood, an era or a visual style the user did not ask for.
- Use active, present-progressive verbs ("is walking", "turns", "lifts"). If the input asks for no particular action, describe small, natural movements.
- Keep a strict chronological flow with plain connectors such as "as", "then", "while" and "a moment later".
- Sound: describe the soundtrack the scene produces, woven into the action where it happens rather than collected at the end: the ambient sound of the place, the sounds the actions make, and speech or music only when the user asks for them. Be specific ("footsteps on wet pavement"), not vague ("ambient sound"), and match the loudness to the action.
- Speech, only when the user asks for speech, talking, singing or a conversation: give the exact words in quotes, and say who speaks and how the voice sounds. Keep the user's own dialogue word for word, correcting only typos, in the language the user wrote it in, and name that language if it is not English. If the user asks for speech without giving the words, write a short line that fits the scene.
- Camera: describe camera movement, angles or framing only when the user asks for them. Do not add any of your own.
- Style: only if the user names a style, begin with "Style: <style>," and then the description. Otherwise write no style label and add no look of your own.
- For a first-person or point-of-view request, do not describe the person whose view it is.
- Describe only what can be seen and heard: no smell, taste, touch or thoughts. Do not interpret emotions or intentions; describe the visible expression or action instead.
- Write plainly. Use neutral, restrained wording: plain colours ("a red dress", not "a vibrant red dress"), neutral light ("soft light from a window"), and no dramatic or promotional adjectives.
- No timestamps and no scene cuts unless the user asks for them.
- If the input is already a detailed, chronological prompt of this kind, keep it and change little: put it in order and add the soundtrack if it is missing.

Output format (strict):
- One single paragraph of natural English prose, roughly 80 to 200 words: shorter for a simple scene, longer only when the user asked for more.
- No titles, headings, labels, lists, code fences or Markdown. Do not begin with "The scene opens", "The video starts", "In this video" or the like, and do not begin with a punctuation mark.
- Output only the prompt. Never ask a question or explain. If the input cannot be turned into a video description, return it unchanged.
"""

STRUCTURE = """\
**What an LTX 2.3 prompt is made of** (Lightricks' prompting guide for LTX-2)

One flowing paragraph, chronological, literal and precise, within about 200 words:

- the main action, in a single sentence
- specific movements and gestures
- character and object appearances, precisely
- background and environment
- camera angles and movements (here: only when asked for)
- lighting and colours
- any changes or sudden events

LTX 2.3 makes the soundtrack too: sounds are described where they happen, and spoken words go in quotes with who says them and how. With a first frame, the prompt describes what happens from that frame on rather than what it already shows.
"""

TEMPERATURE = 0.7
"""Lightricks' Gemma 3 enhancer samples at 0.7 (``GEMMA3_ENHANCE_GENERATION_KWARGS``)."""

TOP_P = 0.95

MAX_TOKENS = 768
"""Lightricks allow 512 new tokens. A 200-word paragraph is about 270; the
margin is for a local model that is slower to finish its sentence, because a
prompt cut off mid-sentence is worse than one that took a moment longer."""

USER_IMAGE_TEMPLATE = "User Raw Input Prompt: {prompt}."
"""The user turn beside a first frame, as ``enhance_i2v`` writes it."""

USER_TEXT_TEMPLATE = "user prompt: {prompt}"
"""The user turn with no picture, as ``enhance_t2v`` writes it."""

_THINKING = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
_FENCED = re.compile(r"\A```[^\n]*\n(?P<body>.*?)\n?```\Z", re.DOTALL)
_LEAD_LABEL = re.compile(r"\A(?:\*\*)?\s*(?:enhanced\s+prompt|ltx(?:[- ]?2(?:\.[0-9])?)?\s+prompt|prompt|output)"
                         r"\s*(?:\*\*)?\s*:\s*(?:\*\*)?\s*", re.IGNORECASE)
_SPACES = re.compile(r"\s+")


def instructions(has_image: bool) -> str:
    """The system prompt for this request: with a first frame, or with none.

    The same for every model in :data:`MODELS`: LTX 2.5 is written for under
    LTX 2.3's instructions, by the user's choice.
    """
    return IMAGE_SYSTEM_PROMPT if has_image else TEXT_SYSTEM_PROMPT


def model_name(model: str = LTX23) -> str:
    """``"LTX 2.5"`` for ``"ltx25"``; LTX 2.3 for anything else, which is what
    every LTX request was before there was a second model to name."""
    return MODEL_NAMES.get(str(model or "").strip().casefold(), MODEL_NAMES[LTX23])


def label(has_image: bool, model: str = LTX23) -> str:
    """What the run says it is doing, naming the model it writes for."""
    return (STATUS_IMAGE if has_image else STATUS_TEXT).format(name=model_name(model))


def user_turn(prompt: str, image_data_url: str | None = None):
    """Lightricks' user turn: the picture first and then the request, or the request."""
    if image_data_url is None:
        return USER_TEXT_TEMPLATE.format(prompt=prompt)
    return [{"type": "image_url", "image_url": {"url": image_data_url}},
            {"type": "text", "text": USER_IMAGE_TEMPLATE.format(prompt=prompt)}]


def messages(prompt: str, *, image_data_url: str | None = None,
             system: str | None = None) -> list[dict]:
    """The whole request: the instructions, and the request under them.

    ``system`` replaces the instructions this request would otherwise have run
    under -- the external API's override, which Mini Paint's Clipboard tab
    sends when its user saved one. WanGP's own prompt dialect applies on top,
    exactly as it does for MiniMax: ``body @ extra`` adds ``extra`` to the
    instructions and ``body @@ replacement`` replaces them, the double form
    looked for first. The parsing is the vendored MiniMax enhancer's, which is
    WanGP's.
    """
    from prompt_master.minimax import enhancer

    body, suffix, replace = enhancer.split_system_suffix(prompt)
    base = instructions(image_data_url is not None) if system is None else str(system)
    return [{"role": "system", "content": enhancer.merge_system(base, suffix, replace)},
            {"role": "user", "content": user_turn(body, image_data_url)}]


def clean(text: str) -> str:
    """The finished prompt, on one line, with what is not part of it taken off.

    Reasoning a model leaked, a fence round the answer, a leading "Prompt:"
    label, and every line break -- see the module docstring on why one line.
    """
    text = _THINKING.sub("", str(text or "")).strip()
    fenced = _FENCED.match(text)
    text = (fenced.group("body") if fenced else text).strip()
    text = _LEAD_LABEL.sub("", text, count=1)
    text = _SPACES.sub(" ", text).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'" and text.count(text[0]) == 2:
        text = text[1:-1].strip()
    return text
