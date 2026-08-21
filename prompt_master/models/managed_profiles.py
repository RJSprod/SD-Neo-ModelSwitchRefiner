"""The quality settings a managed backbone runs at, decided here rather than asked.

A user of the Managed Backbones catalogue chooses *a model*. They do not choose
a KV cache type, a top-k, a min-p or a chat-template flag, and the catalogue is
only worth having because they do not have to: every entry in it was picked with
a set of values already established for it, and shipping the model without them
would be shipping the half of the decision that is easy to make.

So these are constants in the source tree, not controls in Setup and not keys in
the preferences file. The rule the rest of this module keeps is section 5 of the
design intent: *managed profiles control model behaviour; the broker controls
where the model fits.* Nothing below names a GPU index, forces a full offload or
asks for mixed mode, because those are facts about somebody's machine and this
file is checked in. ``mc_llm_runtime`` negotiates placement exactly as it always
has, and reads the values here for everything else.

What is deliberately *not* per-model
------------------------------------
The Creative Director instruction, Krea's system prompt, its 1024-token output
budget and the reference-captioning behaviour are unchanged across every
backbone -- they are the product, and a backbone is a thing the product runs on.
Creative Mode's 0-10 curve is the same: it stays the one place temperature and
top_p come from (``prompt_master.krea.variation``), so that "Creativity 7" keeps
one meaning no matter which model is loaded. A profile may add *fixed* family
sampler constraints beside that curve -- Qwen wants top_k 20 where Gemma wants
64 -- and may not introduce a second thing called creativity.

Why 8192 context, on models that advertise far more
---------------------------------------------------
This application writes image prompts and captions references. The longest
thing it ever sends is a system instruction, four reference captions and a
brief, and 8192 tokens covers it with room left. Context is the term in the
VRAM arithmetic that scales without limit, so allocating an advertised 128K or
256K by default would spend gigabytes of a card on tokens nothing will ever
put there. Qwen's own recommendation of a very large context is tied to
preserving its thinking behaviour, and these profiles disable thinking.
"""

from __future__ import annotations

from dataclasses import dataclass, field

VERSION = "1"
"""Bumped when any value below changes.

Recorded in the setup state beside the model id, so an installation can say
which revision of these constants it was configured against -- and so a future
change to a profile is a visible migration rather than a silent one.
"""

CONTEXT_SIZE = 8192
"""See the module docstring. Common to every managed backbone."""

PARALLEL_SEQUENCES = 1
"""One sequence. Every mode here is one user asking one question at a time."""


@dataclass(frozen=True)
class ManagedProfile:
    """How one managed backbone is run, in the two vocabularies it needs.

    The fields split by *where the value has to arrive*, which is not a detail:
    ``context``, the cache types and ``jinja`` are llama-server command-line
    arguments and are fixed when the process starts, while ``top_k``, ``min_p``,
    ``repeat_penalty`` and ``presence_penalty`` travel per request in the
    completion payload. Temperature and top_p are in neither list on purpose --
    they come from Creative Mode's curve, per request, and a profile that set
    them would be overriding the user's Creativity setting.
    """

    profile_id: str
    context: int = CONTEXT_SIZE
    kv_type_k: str = "q8_0"
    kv_type_v: str = "q8_0"
    jinja: bool = True
    """llama-server's ``--jinja``: use the template baked into the GGUF.

    True for every profile here. Both families ship a chat template that their
    tuning depends on, and the difference between running it and running
    llama.cpp's built-in guess is a model that answers in the format the prompt
    engine parses versus one that does not.
    """
    thinking: bool = False
    """Reasoning traces, off everywhere.

    Not a quality preference: this application reads the model's output as a
    finished prompt, and a chain of thought in front of it is text the parser
    has to strip. ``llama_process`` already passes ``--reasoning off``; the
    request payload says it a second time because a template can re-enable it.
    """
    sampling: dict = field(default_factory=dict)
    """Fixed sampler fields, in llama.cpp's own spelling.

    Only the keys :data:`SAMPLER_FIELDS` allows survive into a request. Empty
    for a profile that wants the family default, which is not the same as a
    profile that wants the value zero -- ``{"min_p": 0.0}`` is a decision and
    ``{}`` is the absence of one.
    """


SAMPLER_FIELDS = ("top_k", "min_p", "repeat_penalty", "presence_penalty",
                  "frequency_penalty")
"""The only per-request sampler keys a profile may set.

A whitelist rather than a passthrough, and the reason is the same one that
keeps the registry closed: everything here ends up in a JSON body sent to a
local server, and "whatever the profile dict happened to contain" is not a
thing to put there. ``temperature`` and ``top_p`` are absent by design -- see
the note on :class:`ManagedProfile`.
"""


PROFILES: dict[str, ManagedProfile] = {
    # -- Gemma 4 ------------------------------------------------------------ #
    #
    # The publisher's own anchor for the QAT Q4 build, which is also close to
    # what the Krea writer was already doing before there was a catalogue: a
    # top-k wide enough to keep the prose varied, a small min-p floor to cut
    # the tail, and a light repetition penalty because image prompts legitimately
    # repeat a subject noun several times in one sentence.
    "gemma4-12b-qat-balanced": ManagedProfile(
        profile_id="gemma4-12b-qat-balanced",
        sampling={"top_k": 64, "min_p": 0.05, "repeat_penalty": 1.10},
    ),
    # The same family at a fraction of the residency. The repetition penalty is
    # neutral here rather than 1.10: this build is already terse, and penalising
    # repeats on top of that costs the deliberate ones.
    "gemma4-e4b-aggressive": ManagedProfile(
        profile_id="gemma4-e4b-aggressive",
        sampling={"top_k": 64, "repeat_penalty": 1.00},
    ),

    # -- Qwen 3.5 ----------------------------------------------------------- #
    #
    # Qwen's published sampling for non-thinking use is a much narrower top-k
    # than Gemma's, with no min-p floor at all -- 0.00 is written out rather
    # than omitted because it is a decision the publisher documented, not a
    # value nobody considered.
    "qwen35-9b-aggressive": ManagedProfile(
        profile_id="qwen35-9b-aggressive",
        sampling={"top_k": 20, "min_p": 0.00, "repeat_penalty": 1.00},
    ),
    # The creative Qwen. Its publisher suggests a presence penalty for general
    # chat; this starts at zero, because the workload here is a constrained
    # prompt-writing one where the same nouns recur by design and pushing the
    # model off them is behavioural drift rather than variety. Changing it is a
    # tested revision of this file, never a control in Setup.
    "qwen35-9b-defiant-fable": ManagedProfile(
        profile_id="qwen35-9b-defiant-fable",
        sampling={"top_k": 20, "min_p": 0.00, "repeat_penalty": 1.00,
                  "presence_penalty": 0.00},
    ),
    "qwen35-4b-aggressive": ManagedProfile(
        profile_id="qwen35-4b-aggressive",
        sampling={"top_k": 20, "min_p": 0.00, "repeat_penalty": 1.00},
    ),

    # -- The baseline ------------------------------------------------------- #
    #
    # f16 cache and no extra samplers: this entry exists so there is an
    # automated route back to the behaviour this extension is known to have,
    # and every value that differs from that behaviour makes it worse at that
    # job. It is not optimised here on purpose -- tuning the large baseline in
    # the same change that introduces model switching would leave nothing to
    # compare a switch against.
    "gemma4-26b-a4b-balanced": ManagedProfile(
        profile_id="gemma4-26b-a4b-balanced",
        kv_type_k="f16", kv_type_v="f16",
    ),
}


def profile(profile_id: str) -> ManagedProfile | None:
    """The profile ``profile_id`` names, or ``None`` when there is no such thing.

    ``None`` rather than a raise, and rather than a default profile: a state
    file naming a profile this build has never heard of is a downgrade, and the
    honest response is for the caller to fall back to manual behaviour and say
    so -- not to run somebody's model at settings invented for a different one.
    """
    return PROFILES.get(str(profile_id or "").strip())


def sampler_arguments(found: ManagedProfile | None) -> dict:
    """``found``'s per-request sampler fields, filtered to :data:`SAMPLER_FIELDS`.

    Returns a fresh dict every call so a caller that adds ``temperature`` to it
    on the way into a request cannot edit the profile constant.
    """
    if found is None:
        return {}
    return {key: found.sampling[key] for key in SAMPLER_FIELDS
            if key in found.sampling and found.sampling[key] is not None}
