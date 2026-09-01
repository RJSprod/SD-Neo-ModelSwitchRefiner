"""Where a voice component runs, and the truth about where it can.

Voice Chat has six things that execute: three speech engines, the recording
cleanup, and the two enhancement stages. Until now every one of them ran on the
processor and said so -- each closure pins a CPU wheel, and each worker's
environment sets ``CUDA_VISIBLE_DEVICES`` to the empty string so that a library
which would have found a card finds nothing. That was a deliberate Model Chain
V1 decision and it is still the default, because a voice engine that quietly
took VRAM would be a voice engine competing with the image generation nobody
asked it to compete with.

This module is where that stops being a constant and becomes a choice, for the
components whose runtime can actually honour one. It answers three questions
and keeps them apart, because collapsing them is how a settings panel ends up
lying:

    what devices does this machine have           :func:`cards`
    where has this component been asked to run    :func:`placement`
    can this component run there at all           :func:`placeable`

The third is the one that earns the module. A control whose value the engine
ignores is a control that lies about what it did, so a component that cannot
leave the processor does not get a dropdown here -- it gets a sentence naming
the wheel that would have to change, which :mod:`mc_voice_ui` prints where the
dropdown would have been.

Three namespaces, and only one of them is trustworthy
-----------------------------------------------------
A machine with two cards numbers them at least three different ways, and this
repository has already been bitten once by assuming two of them agree:

    nvidia-smi orders by PCI bus;
    CUDA orders by ``CUDA_DEVICE_ORDER``, which defaults to fastest-first;
    DirectML orders by its own DXGI adapter enumeration.

So the token stored for a card is its **UUID** -- the one identifier none of
those orderings can move, and the same string :mod:`mc_llm_setup` records for
the language model, which means a card that reads "GPU 1" in one panel and
"GPU 0" in another is still provably the same card in stable storage.

What is *not* solved by that is which DirectML adapter number reaches that card.
Nothing in the ONNX Runtime Python API enumerates DXGI adapters, so there is no
mapping to look up and this module does not invent one: :func:`adapter_for`
hands DirectML the card's nvidia-smi index and says in this docstring that it is
an assumption. The recovery is not a guess either -- the worker reports the
provider each session was actually built on, ``_log_placement`` writes it to the
log at every load, and the settings panel says that if the wrong card lights up,
choosing the other entry is the fix. (The adapter is not reported back, because
ONNX Runtime does not expose one; what is reported is enough to tell a card from
the processor, and choosing the other entry is what tells one card from the
other.) An assumption somebody can see and correct in one click is a different
thing from a mapping presented as fact.

Placement is not memory management
----------------------------------
A choice made here is meant to stick, and it does -- structurally rather than by
anybody agreeing to respect it. A placed component runs in its own process, and
``mc_broker.free_vram_bytes`` is a ``mem_get_info`` query about the whole card,
so the image side and the language model size themselves against what is
actually free and cannot evict what they cannot see. That is the same property
that makes co-residency with llama-server work at all, and the asymmetry is
deliberate: this is a hundreds-of-megabytes tenant, not a tens-of-gigabytes one,
and being evicted mid-sentence is a defect the listener hears.

What that does *not* buy is priority on a card somebody else filled first. A
stage that cannot allocate fails to load and the reply is spoken unenhanced,
with the reason in the log -- which is a graceful answer but not the one a
reservation would give. Closing that gap means the language model's planner
subtracting a known voice reservation *before* it sizes an offload rather than
discovering it in a live reading afterwards, and :func:`reserved_mb` is the seam
it would attach to. It answers zero today, honestly: nothing here yet knows how
much a loaded stage took.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

CPU = "cpu"
"""The token for the processor. Not a UUID, because the processor has none and
inventing one would put a fake identifier in somebody's config file."""

GPU_PREFIX = "gpu:"
"""What a card's token starts with, so a token is self-describing.

The remainder is the card's UUID exactly as nvidia-smi reports it. Prefixed
rather than bare so that a stored value can be recognised as a device token by
looking at it, which matters when the card it names is no longer in the machine
and there is nothing to match it against.
"""

DEVICE_CACHE_SECONDS = 60
"""How long a card list is reused before nvidia-smi is asked again.

The same number and the same reasoning as :data:`mc_llm_setup.DEVICE_CACHE_SECONDS`:
detection is a subprocess call, a settings panel asks for the list several times
while it is being drawn, and the set of cards in a machine does not change while
a WebUI is running.
"""

_cards_cache = None

PROVIDER_CPU = "CPUExecutionProvider"
PROVIDER_DIRECTML = "DmlExecutionProvider"
"""The two ONNX Runtime execution providers the pinned wheel carries.

Named here rather than in the worker because the settings surface has to be able
to say which one a choice means without importing anything from the isolated
runtime -- which it could not do anyway, that being the point of the runtime
being isolated.
"""


class Placement:
    """What one component can be told about where to run.

    Deliberately not a dataclass with a ``supported`` boolean. The two states
    are not symmetrical: a component that can move needs an execution provider
    name, and one that cannot needs a *sentence* -- and a sentence is not a
    fallback for a missing boolean, it is the whole content of the answer. A
    reason that read "GPU unsupported" would be the same as no reason at all.
    """

    __slots__ = ("component", "label", "option", "provider", "reason")

    def __init__(self, component: str, label: str, option: str = "",
                 provider: str = "", reason: str = ""):
        self.component = component
        self.label = label
        self.option = option
        self.provider = provider
        self.reason = reason

    @property
    def placeable(self) -> bool:
        """Whether this component can be asked to run anywhere but the processor."""
        return bool(self.provider and self.option)


_TORCH_REASON = (
    "This engine runs on PyTorch, and the closure installed for it pins the CPU "
    "build of Torch. Moving it to a graphics card is not a setting -- it is a "
    "different closure, roughly two and a half gigabytes of CUDA wheels, pinned "
    "per card generation because the build that serves a 30-series card is not "
    "the one that serves a 50-series card. That is a download and a decision "
    "rather than a checkbox, so it is not offered as one.")

_SHERPA_REASON = (
    "Kokoro runs on sherpa-onnx, and the closure installed for it pins the CPU "
    "wheel. Upstream publishes GPU wheels only for CUDA, one per toolkit "
    "version, so unlike the enhancement stages there is no single wheel that "
    "serves both cards in this machine. Until there is, this engine stays on "
    "the processor and says so rather than offering a dropdown that would have "
    "to guess.")

_CLEANUP_REASON = (
    "Recording cleanup runs DeepFilterNet on PyTorch, and its closure pins the "
    "CPU build of Torch for the same reason the speech engines' do -- a graphics "
    "build is a different closure of roughly two and a half gigabytes, pinned per "
    "card generation. It is also the one component here that runs while nothing "
    "is being spoken and nothing is being generated, cleaning one recording once, "
    "so the processor is where the work belongs anyway.")

OPT_DEVICE_DPDFNET = "model_chain_voice_pipeline_device_dpdfnet"
OPT_DEVICE_LAVASR = "model_chain_voice_pipeline_device_lavasr"
"""The option names, spelled out rather than reached for through
:data:`PLACEMENTS`. The settings section registers them by name, and an index
into a tuple is a registration that silently moves to a different setting the
day somebody reorders the tuple."""

PLACEMENTS = (
    Placement("voice-pipeline-dpdfnet", "DPDFNet",
              option=OPT_DEVICE_DPDFNET, provider=PROVIDER_DIRECTML),
    Placement("voice-pipeline-lavasr", "LavaSR",
              option=OPT_DEVICE_LAVASR, provider=PROVIDER_DIRECTML),
    Placement("tts-pocket", "PocketTTS", reason=_TORCH_REASON),
    Placement("tts-sopro", "Sopro V2", reason=_TORCH_REASON),
    Placement("tts-kokoro", "Kokoro", reason=_SHERPA_REASON),
    Placement("recording-cleanup", "Recording cleanup", reason=_CLEANUP_REASON),
)
"""Every voice component, and what it can be told.

The ids are :data:`mc_voice_ui.COMPONENTS`'s ids and that is not a coincidence:
the settings surface asks this module by the id it already has. The dependency
runs one way -- this module knows nothing about the UI, so a component that
gains a panel does not have to gain an entry here to keep working, it simply has
no placement control.
"""


def placement_for(component: str) -> "Placement | None":
    """One component's placement rules, or ``None`` for an id this build lacks."""
    wanted = str(component or "")
    for spec in PLACEMENTS:
        if spec.component == wanted:
            return spec
    return None


def placeable(component: str) -> bool:
    """Whether ``component`` can be asked to run anywhere but the processor."""
    spec = placement_for(component)
    return bool(spec is not None and spec.placeable)


def unplaceable_reason(component: str) -> str:
    """Why ``component`` stays on the processor, or ``""`` when it need not.

    Empty for a component that *can* move and empty for an id this build does
    not have, which are different situations that happen to want the same
    behaviour from a caller: print nothing.
    """
    spec = placement_for(component)
    if spec is None or spec.placeable:
        return ""
    return spec.reason


# --------------------------------------------------------------------------- #
# What the machine has
# --------------------------------------------------------------------------- #


def cards(refresh: bool = False) -> list:
    """The processor and every card in the machine, as plain records. Never raises.

    Deliberately *not* :func:`mc_llm_setup.devices`, which offers each card three
    or four times -- once per placement mode -- because the language model can
    be split across a card and system RAM. An enhancement session cannot: it is
    built on one execution provider and it either runs there or it does not. So
    this list has one entry per piece of hardware, which is also the only shape
    a reader can check against what is physically in the box.

    A machine with no NVIDIA driver gets the processor and nothing else, which
    is a true statement rather than an error: the panel then offers one choice,
    which is the choice that machine has.
    """
    global _cards_cache

    import time

    if not refresh and _cards_cache is not None:
        cached_at, cached = _cards_cache
        if time.monotonic() - cached_at < DEVICE_CACHE_SECONDS:
            return [dict(item) for item in cached]

    found = [_cpu_card()]
    try:
        from prompt_master.inference.device_detection import detect_gpus

        for gpu in detect_gpus():
            record = _gpu_card(gpu)
            if record is not None:
                found.append(record)
    except Exception:
        # No driver, no nvidia-smi, or a timeout. All three mean the same thing
        # to a settings panel -- there is no card to offer -- and none of them
        # is a reason to fail the panel that was being drawn.
        logger.debug("Model Chain: Voice Chat could not enumerate graphics cards",
                     exc_info=True)

    _cards_cache = (time.monotonic(), [dict(item) for item in found])
    return [dict(item) for item in found]


def forget_cards() -> None:
    """Drop the cached card list. For tests, and for a rescan."""
    global _cards_cache
    _cards_cache = None


def _cpu_card() -> dict:
    """The processor, described the way a card is, and never absent.

    Built without the detector when the detector will not import, because the
    processor is the one device whose presence is not in question: a record
    saying so with no name on it is still true, and it is what keeps the
    dropdown from being empty on a machine where everything else failed.
    """
    name, memory = "Processor", 0
    try:
        from prompt_master.inference.device_detection import detect_cpu

        found = detect_cpu()
        name = str(getattr(found, "name", "") or name)
        memory = int(getattr(found, "memory_total_mb", 0) or 0)
    except Exception:
        logger.debug("Model Chain: Voice Chat could not describe the processor",
                     exc_info=True)
    return {"token": CPU, "kind": "cpu", "index": -1, "adapter": -1,
            "uuid": "", "name": name, "memory_mb": memory,
            "label": name if not memory else f"{name} — {memory} MiB of system RAM"}


def _gpu_card(gpu) -> "dict | None":
    """One nvidia-smi row as a record, or ``None`` when it has no usable identity.

    A card with no UUID is dropped rather than given a positional token. The
    whole reason the token is a UUID is that positions move between the three
    device namespaces this machine has, so a token built from a position would
    be the exact bug this module exists to avoid, wearing the costume of a
    fallback.
    """
    uuid = str(getattr(gpu, "uuid", "") or "").strip()
    if not uuid:
        return None
    index = int(getattr(gpu, "physical_index", -1))
    name = str(getattr(gpu, "name", "") or "Graphics card").strip()
    memory = int(getattr(gpu, "memory_total_mb", 0) or 0)
    label = f"GPU {index} — {name}" if index >= 0 else name
    if memory:
        label = f"{label} ({memory} MiB)"
    return {"token": f"{GPU_PREFIX}{uuid}", "kind": "gpu", "index": index,
            "adapter": index, "uuid": uuid, "name": name, "memory_mb": memory,
            "label": label}


def card(token: str) -> "dict | None":
    """The record for one token, or ``None`` when this machine has no such device.

    ``None`` for a card that has been removed since the setting was written, and
    that is the case worth naming: the stored token stays exactly as it was
    rather than being rewritten to the processor, so putting the card back
    restores the choice instead of silently having lost it.
    """
    wanted = str(token or "")
    if not wanted:
        return None
    for item in cards():
        if item["token"] == wanted:
            return dict(item)
    return None


def adapter_for(token: str) -> int:
    """The DirectML adapter number for a card token. An assumption, on purpose.

    Nothing in the ONNX Runtime Python API enumerates DXGI adapters, so the
    number handed to the DirectML execution provider is the card's nvidia-smi
    index and there is no lookup that would make it more than that. See this
    module's own docstring: the recovery is that the worker reports what it was
    actually given and the panel says to choose the other entry if the wrong
    card lights up, which is a correction somebody can make rather than a
    mapping this code pretends to know.

    Zero for anything unrecognised, because a provider asked for adapter -1
    fails to create a session and the resulting message would be about an
    adapter index rather than about a card that is no longer in the machine.
    """
    found = card(token)
    if found is None or found["kind"] != "gpu":
        return 0
    return max(0, int(found.get("adapter", 0)))


# --------------------------------------------------------------------------- #
# What the user chose
# --------------------------------------------------------------------------- #


def placement(component: str) -> str:
    """The device token ``component`` is configured to run on. CPU on any doubt.

    Every path out of this function that is not a stored, recognised token for a
    component that can honour it answers :data:`CPU`, and the list is longer
    than it looks: no host, no option, a component with no placement rules, a
    component whose rules say it cannot move, and a token naming a card that is
    not in this machine right now. All five mean the same thing to the thing
    about to build a session -- run it where it has always run -- and none of
    them is a reason to fail a reply.

    The *stored* value is untouched by any of that. A card that is missing today
    because the machine is on a different dock is a card whose setting is still
    there tomorrow.
    """
    spec = placement_for(component)
    if spec is None or not spec.placeable:
        return CPU
    token = _text(spec.option)
    if not token or token == CPU:
        return CPU
    return CPU if card(token) is None else token


def stored_placement(component: str) -> str:
    """The token as written, without checking the card is present.

    What the settings panel needs and what :func:`placement` deliberately will
    not give it: a panel that showed the processor for a card that is merely
    unplugged would be a panel inviting somebody to lose their setting by
    touching it.
    """
    spec = placement_for(component)
    if spec is None or not spec.placeable:
        return CPU
    return _text(spec.option) or CPU


def remember(component: str, token) -> str:
    """Write one component's placement through to the host, and return the truth.

    Refuses a token this machine has no device for rather than storing it, which
    is the same posture the enhancement thread budget takes: a value silently
    replaced on the way in is a control that does not do what its own label
    says. The refusal is a raised :class:`ValueError` and not a quiet fallback,
    because the caller is a route with a browser on the other end that can say
    so.

    Returns what the store reads back afterwards rather than what it was asked
    to write, so a surface redraws from the truth even when the host refused the
    write.
    """
    spec = placement_for(component)
    if spec is None or not spec.placeable:
        raise ValueError(f"{component} does not have a device setting.")
    wanted = str(token or "").strip() or CPU
    if wanted != CPU and card(wanted) is None:
        raise ValueError("That device is not in this machine.")
    try:
        from modules import shared

        shared.opts.set(spec.option, wanted)
        shared.opts.save(shared.config_filename)
    except Exception:
        logger.debug("Model Chain: could not persist the device for %s", component,
                     exc_info=True)
    return stored_placement(component)


def provider_for(component: str) -> tuple:
    """The execution provider and adapter number ``component`` should be built on.

    A pair rather than a provider name, because the adapter number is only
    meaningful alongside the provider that reads it and returning them
    separately invites a caller to combine a CPU provider with an adapter index
    that came from somewhere else.
    """
    spec = placement_for(component)
    if spec is None or not spec.placeable:
        return (PROVIDER_CPU, 0)
    token = placement(component)
    if token == CPU:
        return (PROVIDER_CPU, 0)
    return (spec.provider, adapter_for(token))


def describe(component: str) -> dict:
    """Everything a settings surface needs about one component's placement.

    One call rather than five, because the surface draws them together and a
    panel assembled from five reads could show a dropdown built from one card
    list against a current value resolved from another.
    """
    spec = placement_for(component)
    if spec is None:
        return {"component": str(component or ""), "placeable": False, "reason": "",
                "device": CPU, "devices": [], "provider": PROVIDER_CPU, "adapter": 0}
    if not spec.placeable:
        return {"component": spec.component, "placeable": False, "reason": spec.reason,
                "device": CPU, "devices": [], "provider": PROVIDER_CPU, "adapter": 0}
    provider, adapter = provider_for(spec.component)
    return {"component": spec.component, "placeable": True, "reason": "",
            "device": stored_placement(spec.component), "devices": cards(),
            "provider": provider, "adapter": adapter}


def reserved_mb() -> int:
    """How much VRAM the voice side is holding, for a planner to subtract first.

    Zero today, and honestly so: nothing here yet knows how much a loaded stage
    took, and a number invented for the shape of the interface would be worse
    than the absence.

    The seam is here rather than in a later commit because what it is *for* is a
    decision this module has already made. A placed component is not a tenant to
    evict -- it cannot be, it is another process and the card is queried live --
    so the only thing left to fix is ordering: a planner that sized an offload
    against a card the enhancement had not loaded on yet, and then took the room
    it needed. Subtracting a known reservation before that decision is what this
    would answer.
    """
    return 0


def _text(name: str, default: str = "") -> str:
    """One string option, read live, falling back to ``default`` on any doubt.

    Read through ``shared.opts.data`` rather than ``getattr(shared.opts, ...)``
    so an option this build registered but a stale config never wrote answers the
    default instead of whatever the host puts on a missing attribute.
    """
    try:
        from modules import shared

        value = shared.opts.data.get(name, default)
    except Exception:
        return default
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip()
    return default
