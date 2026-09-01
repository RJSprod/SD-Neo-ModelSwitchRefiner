"""The Voice Pipeline: what it is, what is installed, and what a turn froze onto.

Optional, off until somebody turns it on, and made of exactly two stages in one
fixed order:

    1  DPDFNet   cleans noise and synthesis artifacts out of generated speech
    2  LavaSR    restores speech bandwidth and delivers 48 kHz

It is not a text-to-speech engine and it is not in the engine selector. It takes
the PCM a speech engine has already finished with and gives back better PCM at a
declared rate, which is why it lives beside :mod:`mc_voice_cleanup` in shape and
nowhere near :mod:`mc_voice_engines` in kind.

Where the pipeline begins
-------------------------
Not at the model. The Voice Pipeline's input is the PCM the source worker would
otherwise have handed straight to :meth:`mc_voice_turn.VoiceTurn.offer_audio` --
which is to say *after* the delivery shaper, after the trim and the internal-gap
policy, after the 8 ms unit seam and after the intentional pauses (I-VP-04).

That boundary is the single most load-bearing decision in this feature and it is
a decision the repository has already paid for. PRs #148 through #153 moved the
quiet-normalisation, gap-shortening and de-click policy into the source workers
on the strength of measurements from real machines, and every one of those
operations deliberately changes duration. A pipeline inserted before them would
enhance audio the source intends to discard, disagree with the source's own
clock, and end up with two competing seam systems. So the enhancement layer owns
what it creates at or after its own boundary -- recurrent state, analysis
windows, overlap, and the sample ledger -- and owns nothing before it.

The corollary is stated here because it is the thing somebody will be tempted to
"fix": if the source trimmed 300 ms of dead air, that 300 ms is not in the
pipeline's input and is never restored (I-VP-08). Preserving duration means
preserving the finalised clock this feature was handed, not the model's
hypothetical pre-trim one.

Three switches and no fourth
----------------------------
The order is structural, not a preference. There are no drag handles, no
move-up buttons, and no persisted order key -- ``dpdfnet`` is 100 and ``lavasr``
is 200 because cleaning a signal before asking a bandwidth-extension model to
reconstruct its missing top is the way round that does not ask the second model
to invent detail out of the first model's hiss (I-VP-01, section 2.5).

What the user gets is master on/off and one switch per stage, and every
combination of those runs: neither, either alone, or both (I-VP-02). A stage
that is off is a bypass, not a no-op wrapper, and master off is the current
delivery path with no pipeline object in it at all (I-VP-03).

Why this build cannot install anything yet
------------------------------------------
:func:`pinned` answers False and :func:`install` refuses, because
``voice/managed-pipeline-models.json`` carries no revisions, no digests and --
the one that matters -- no measured LavaSR rate contract. Upstream's README
advertises 8-48 kHz input; the reviewed upstream inference path resamples 16 kHz
to 48 kHz internally. Those are different claims, and shipping an installer
built on the friendlier one means shipping speech played at two-thirds speed to
somebody who trusted it. Section 23 calls Phase 0 blocking; this module is what
makes "blocking" a refusal rather than a note in a document.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

import mc_voice_paths as paths

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

FEATURE = "voice_pipeline"
LABEL = "Voice Pipeline"

KIND = "pipeline"
"""The key this feature's install progress is filed under.

Beside ``runtime``, ``stt``, ``tts``, ``sopro`` and ``cleanup`` in the one
progress map, so the settings row draws it with the same code and two installs
cannot run at once under the same name.
"""

SCHEMA = 1

RUNTIME_FLAVOURS = ("onnx", "torch")
"""The closures this build knows how to install, newest capability last.

``onnx`` is the light one: ONNX Runtime, librosa and their dependencies, which
is everything DPDFNet needs. ``torch`` adds PyTorch for LavaSR upstream and
keeps ONNX Runtime alongside it, so one worker still serves every enabled stage.

Which one gets installed follows from what the enabled stages need rather than
from a preference anybody sets, and :func:`required_flavour` is that derivation.
"""

RUNTIME_PROVIDERS = ("cpu", "cpu+directml", "cpu+directml+torch")
"""Which runtime closures this build knows how to execute on.

Checked rather than assumed, and it used to be a check for the single string
``cpu`` with a refusal that said this feature would never take a graphics device
from an image being made. That claim is no longer the one being made, and the
sentence went with it: a stage may now be placed on a card, deliberately, by
somebody who asked for it on its own panel.

What the check still does is the useful half. A manifest naming a closure this
code cannot execute on -- a CUDA build, say, whose wheels this installer would
happily unpack and whose sessions nothing here knows how to construct -- is a
broken extension rather than a broken installation, and it is refused before a
download starts rather than after one finishes.
"""

MAX_INTRAOP_THREADS = 16
"""The most this stage may be given, whatever is typed at it.

Not a hardware limit -- a budget one. Beyond the cores actually present, more
ONNX Runtime threads is contention rather than throughput, and every one of
them is taken from the PocketTTS generation this stage exists to improve."""


def threads() -> int:
    """The enhancement stage's intra-op budget, as configured.

    A setting rather than a constant since a user's own measurement beat the
    reasoning behind the constant. Upstream's session is built with one thread
    and this feature's own default was two, chosen so as not to crowd Pocket --
    and on a sixteen-thread machine that produced a real-time factor of 2.38,
    which is not a stage being polite, it is a stage that cannot keep up.

    Whoever is listening has the only instrument that matters here: the
    ``Voice pipeline ran`` line reports the measured factor for every reply, so
    this is a dial to turn against a number rather than a guess to get right
    first time.
    """
    try:
        from modules import shared

        found = int(shared.opts.data.get(OPT_THREADS, INTRAOP_THREADS))
    except Exception:
        return INTRAOP_THREADS
    return max(1, min(MAX_INTRAOP_THREADS, found))


def component_of(stage_id: str) -> str:
    """The component id one stage is known by outside this module.

    The settings surface, the install progress map and :mod:`mc_voice_device`
    all name a stage ``voice-pipeline-<id>``; this module names it ``<id>``.
    Written once here rather than formatted at each of the four call sites,
    because the two vocabularies meeting in four places is how they drift.
    """
    return f"voice-pipeline-{str(stage_id or '')}"


def placement(stage_id: str) -> tuple:
    """The execution provider and adapter number one stage should be built on.

    Delegated whole to :mod:`mc_voice_device` rather than read out of an option
    here, because "where does this run" is a question the speech engines and the
    cleanup ask too, and answering it in each feature's own module is how three
    features end up with three device vocabularies.

    Falls back to the processor on any failure at all, including this build not
    having that module. A stage that cannot find out where it was asked to run
    still runs, in the place it has always run.
    """
    try:
        import mc_voice_device as devices

        return devices.provider_for(component_of(stage_id))
    except Exception:
        logger.debug("Model Chain: could not resolve the device for the stage %s",
                     stage_id, exc_info=True)
        return ("CPUExecutionProvider", 0)


def devices_for(stage_id: str) -> dict:
    """What a settings surface needs to draw one stage's placement control."""
    try:
        import mc_voice_device as devices

        return devices.describe(component_of(stage_id))
    except Exception:
        logger.debug("Model Chain: could not describe the devices for the stage %s",
                     stage_id, exc_info=True)
        return {"component": component_of(stage_id), "placeable": False, "reason": "",
                "device": "cpu", "devices": [], "provider": "CPUExecutionProvider",
                "adapter": 0}


INTRAOP_THREADS = 2
INTEROP_THREADS = 1
"""The enhancement runtime's CPU budget, and it is smaller than everybody
else's on purpose.

Sopro, Kokoro and the cleanup engine each get four intra-op threads because each
of them is the only thing running when it runs. This one runs *beside* a
PocketTTS generation, on the same cores, for the whole length of a reply
(section 11.9). Four here would be four more threads contending with the model
whose output this is supposed to be improving, and an enhancement stage that
makes the speech it is enhancing arrive late has not improved anything.

Two rather than one because the measured target is a sustained real-time factor
comfortably under 1.0 with headroom (section 11.7), and one thread has none.
This is a starting budget to benchmark against, which is what section 9.8 asks
for, not a number anybody has proved.
"""


class PipelineError(RuntimeError):
    """A Voice Pipeline operation that could not be completed. Never fatal.

    Never fatal is the whole contract. Every path that raises this has a caller
    that turns it into a sentence and a reply that is still spoken -- with the
    pipeline off for that turn if it had not started, and cancelled cleanly if
    it had (section 15). Speech that is not enhanced is a disappointment;
    speech that does not happen because an optional polish failed is a bug.
    """


# --------------------------------------------------------------------------- #
# The stage registry
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class StageSpec:
    """One enhancement stage, and everything the pipeline decides from it.

    ``order`` is the fixed semantic position and it is not a default anybody may
    override. It exists as a number rather than as tuple position so that
    :func:`snapshot` can sort by it and a reader can see that sorting by
    *anything the user typed* was never possible: there is no order in the
    persisted settings to sort by (I-VP-01, section 18.1).

    ``output_rate_policy`` is how this stage answers "what rate comes out of
    you". ``"preserve"`` means the caller's rate, unchanged; a numeric string
    means that rate, whatever went in. It is a policy rather than a number
    because DPDFNet's answer genuinely depends on its caller and LavaSR's
    genuinely does not.
    """

    id: str
    order: int
    role: str
    label: str
    summary: str
    option: str
    output_rate_policy: str

    def rate_after(self, rate: int) -> int:
        """The sample rate downstream of this stage, given ``rate`` into it."""
        if self.output_rate_policy == "preserve":
            return int(rate)
        return int(self.output_rate_policy)


OPT_ENABLED = "model_chain_voice_pipeline"
OPT_DPDFNET = "model_chain_voice_pipeline_dpdfnet"
OPT_LAVASR = "model_chain_voice_pipeline_lavasr"
OPT_THREADS = "model_chain_voice_pipeline_threads"

STAGES = (
    StageSpec(
        id="dpdfnet",
        order=100,
        role="denoise",
        label="DPDFNet",
        summary="Clean noise and synthesis artifacts",
        option=OPT_DPDFNET,
        output_rate_policy="preserve",
    ),
    StageSpec(
        id="lavasr",
        order=200,
        role="bandwidth_extension",
        label="LavaSR",
        summary="Restore speech bandwidth to 48 kHz",
        option=OPT_LAVASR,
        output_rate_policy="48000",
    ),
)
"""The whole pipeline, in the only order it runs in.

A tuple rather than a list, and read-only everywhere: this is the structure the
feature promises, and a registry somebody can append to at runtime is a registry
that eventually gets appended to from a request body.
"""

STAGE_IDS = tuple(spec.id for spec in STAGES)

SUPPORTED_ENGINES = ("pocket",)
"""Which speech engines hand their finalised PCM to the pipeline today.

One, deliberately (I-VP-29, section 7). The pipeline's own contract is generic
-- PCM, a real rate, a channel count and a snapshot -- and nothing in the worker
knows what a Pocket unit is. What is Pocket-only is the *plumbing*: the handoff
lives in :mod:`mc_voice_pocket_runtime` because that is where the finalised PCM
of that engine exists, and Kokoro's and Sopro's own boundaries are theirs to
declare when somebody attaches them (section 7 of the phase plan). A tuple
rather than a bare string because the second entry is a plumbing change, not a
design one.
"""


def stage(stage_id: str) -> "StageSpec | None":
    """One stage by id, or ``None`` for a name this build does not have."""
    wanted = str(stage_id or "")
    for spec in STAGES:
        if spec.id == wanted:
            return spec
    return None


# --------------------------------------------------------------------------- #
# The switches
# --------------------------------------------------------------------------- #


def enabled() -> bool:
    """Whether the master switch is on. Default off, on any doubt at all.

    The same shape as :func:`mc_voice_state.auto_speak` and for the same reason:
    a host that will not answer, an option nobody registered and a value that is
    not a boolean all mean "the user has not asked for this", and the answer to
    that is the existing playback path.
    """
    return _flag(OPT_ENABLED)


def stage_enabled(stage_id: str) -> bool:
    """Whether one stage's switch is on, ignoring the master and installation.

    Intent only. What actually runs is decided once per turn by
    :func:`snapshot`, which also has to know whether the master is on and
    whether the stage is installed -- three questions that are deliberately not
    collapsed into one, because "off" and "not installed" are different
    sentences to put in front of somebody (section 4.2).
    """
    spec = stage(stage_id)
    if spec is None:
        return False
    return _flag(spec.option, default=True)


def desired_stages() -> tuple:
    """The stage ids the user has asked for, in pipeline order.

    Ignores installation and ignores the master switch. This is what the UI
    draws ticks against and what :func:`snapshot` starts from.
    """
    return tuple(spec.id for spec in STAGES if _flag(spec.option, default=True))


def settings() -> dict:
    """The three switches, as the settings surface and the status route see them."""
    found = {"enabled": enabled(), "threads": threads(),
             "max_threads": MAX_INTRAOP_THREADS, "devices": {}}
    for spec in STAGES:
        found[spec.id] = stage_enabled(spec.id)
        found["devices"][spec.id] = devices_for(spec.id)
    return found


def remember(enabled_value=None, stages=None, threads_value=None, devices=None) -> dict:
    """Write the switches through to the host's options store, and save once.

    One write for all three rather than one route per switch, because the three
    are read together at the top of every turn and a browser that posted them
    separately could be interrupted between two of them -- leaving a
    configuration nobody chose, which the next reply would then be spoken with
    (section 18.3).

    Best-effort against the host and never fatal, and it returns what the store
    says *afterwards* rather than what it was asked to write, so a surface
    redraws from the truth. A host that refuses the write keeps the switches it
    had and the pipeline keeps working at that setting.
    """
    wanted = {}
    if enabled_value is not None:
        wanted[OPT_ENABLED] = bool(enabled_value)
    for spec in STAGES:
        value = (stages or {}).get(spec.id)
        if value is not None:
            wanted[spec.option] = bool(value)
    if threads_value is not None:
        try:
            wanted[OPT_THREADS] = max(1, min(MAX_INTRAOP_THREADS, int(threads_value)))
        except (TypeError, ValueError):
            pass
    if wanted:
        try:
            from modules import shared

            for name, value in wanted.items():
                shared.opts.set(name, value)
            shared.opts.save(shared.config_filename)
        except Exception:
            logger.debug("Model Chain: could not persist a Voice Pipeline switch",
                         exc_info=True)
    # After the switches, and the order is the point. This is the one setting
    # here that can be *refused* -- a token naming a card this machine does not
    # have is not a value to clamp, it is a request to decline -- and a refusal
    # raises out of this function. Done first, it would take the switches in the
    # same request down with it, so a browser posting a tick alongside a stale
    # card list would lose the tick as well. Done last, the tick is already
    # stored and only the device is declined, which is the half that was wrong.
    moved = False
    for stage_id, token in (devices or {}).items():
        if token is None:
            continue
        moved = _remember_device(stage_id, token) or moved
    if moved or OPT_THREADS in wanted:
        _restart_for_execution()
    return settings()


def _restart_for_execution() -> None:
    """Drop the worker when a setting it only reads at load time has changed.

    The thread budget and the placement are both read once, in ``_send_load``,
    and ``ensure_started`` returns early when the loaded stage set already
    matches -- so before this, turning either dial changed nothing until
    something else happened to stop the worker. A control whose label says
    "takes effect on the next reply" has to be one that does.

    Only for those two. The master switch and the stage ticks are read fresh by
    :func:`snapshot` at the top of every turn and need no restart, and stopping
    a worker for them would be a stop nobody asked for.

    Never fatal. A worker that would not stop is a worker still running at the
    old setting, and the surface says as much: ``applied`` already reports
    whether a change is in force now or waiting for the reply in flight to
    finish.
    """
    try:
        import mc_voice_pipeline_runtime as runtime

        runtime.reconfigure("the Voice Pipeline execution settings changed")
    except Exception:
        logger.debug("Model Chain: could not restart the Voice Pipeline for a new "
                     "execution setting", exc_info=True)


def _remember_device(stage_id: str, token) -> bool:
    """Persist one stage's device, and say whether it actually moved.

    :class:`ValueError` reaches the caller intact -- this is the one place in
    :func:`remember` that is allowed to fail the request rather than absorb it,
    and the sentence is the module's own rather than one composed here, so the
    reason a device was declined is written where the decision was made.

    The boolean is what keeps a redundant restart from happening. A browser
    posts the select's value on every change, and a value that is already the
    one in force is a request that stores the same string -- which is not a
    reason to stop a worker mid-conversation.
    """
    if stage(stage_id) is None:
        raise ValueError(f"{stage_id} is not a stage in this build.")
    import mc_voice_device as devices

    component = component_of(stage_id)
    before = devices.stored_placement(component)
    return devices.remember(component, token) != before


def _flag(name: str, default: bool = False) -> bool:
    """One boolean option, read live, falling back to ``default`` on any doubt.

    ``default`` differs by switch and the difference is the whole of section
    4.1's recommended defaults. The master is False: a feature that turned
    itself on across an upgrade would be a feature that changed how somebody's
    WebUI sounds without being asked. The two stages are True: once the master
    *is* on, the intended chain is the one that should already be ticked, so
    turning the feature on is one gesture rather than three.

    A host that cannot be read answers the default rather than raising, because
    this is called on the path that decides how a reply is spoken.
    """
    try:
        from modules import shared

        value = getattr(shared.opts, name, None)
    except Exception:
        return default
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().casefold()
        if text in ("true", "1", "yes", "on"):
            return True
        if text in ("false", "0", "no", "off"):
            return False
    return default


# --------------------------------------------------------------------------- #
# The manifest
# --------------------------------------------------------------------------- #

_manifest_cache = None


def manifest(refresh: bool = False) -> dict:
    """The checked-in trust root for everything the pipeline may fetch.

    Cached, and the cache is this module's own rather than
    :mod:`mc_voice_models`'. Two caches is one more than ideal and fewer than
    the alternative: sharing one would mean a pipeline pin overlay clearing the
    speech manifest, which is a coupling nothing wants and a test nobody would
    think to write.
    """
    global _manifest_cache

    if _manifest_cache is not None and not refresh:
        return _manifest_cache
    path = paths.pipeline_manifest_path()
    try:
        found = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PipelineError(
            f"The Voice Pipeline manifest could not be read ({exc.__class__.__name__}). "
            f"This is a problem with the extension rather than with your installation."
        ) from None
    _manifest_cache = _read_manifest(found)
    return _manifest_cache


def _read_manifest(found) -> dict:
    """Validate the manifest, and refuse the shapes section 20.3 names.

    Every refusal here says the extension is broken rather than the
    installation, because it is: this file is committed in this repository and a
    user cannot edit it into any of these states by using the feature.
    """
    if not isinstance(found, dict):
        raise PipelineError("The Voice Pipeline manifest is not an object.")
    if int(found.get("schema") or 0) != SCHEMA:
        raise PipelineError(
            "The Voice Pipeline manifest is a newer schema than this build reads.")
    runtime = found.get("runtime")
    if not isinstance(runtime, dict):
        raise PipelineError("The Voice Pipeline manifest names no runtime.")
    if str(runtime.get("provider") or "") not in RUNTIME_PROVIDERS:
        raise PipelineError(
            "The Voice Pipeline manifest names a runtime this build does not know how to "
            "execute on. Nothing was installed.")
    entries = found.get("stages")
    if not isinstance(entries, list) or not entries:
        raise PipelineError("The Voice Pipeline manifest names no stages.")

    stages = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise PipelineError("The Voice Pipeline manifest has a stage that is not an "
                                "object.")
        identifier = str(entry.get("id") or "")
        spec = stage(identifier)
        if spec is None:
            raise PipelineError(
                f"The Voice Pipeline manifest names a stage {identifier!r} that this build "
                f"does not have.")
        if identifier in stages:
            raise PipelineError(
                f"The Voice Pipeline manifest names the stage {identifier!r} twice.")
        if int(entry.get("order") or 0) != spec.order:
            # The order is structural. A manifest that disagreed with the code
            # about it would be a manifest that could reorder the chain from a
            # file, which is exactly the thing I-VP-01 forbids from the UI.
            raise PipelineError(
                f"The Voice Pipeline manifest gives {identifier!r} an order this build "
                f"does not agree with. Stage order is structural and is not configuration.")
        stages[identifier] = _read_stage(spec, entry)

    for spec in STAGES:
        if spec.id not in stages:
            raise PipelineError(
                f"The Voice Pipeline manifest does not describe {spec.id!r}.")

    flavours = {}
    for name, block in (("onnx", runtime),) + tuple(
            (key, value) for key, value in sorted((found.get("runtimes") or {}).items())):
        if name != "onnx":
            if name not in RUNTIME_FLAVOURS:
                raise PipelineError(
                    f"The Voice Pipeline manifest names a runtime closure {name!r} that "
                    f"this build does not know how to install.")
            if not isinstance(block, dict):
                raise PipelineError(f"The Voice Pipeline manifest's {name} runtime is not "
                                    f"an object.")
            if str(block.get("provider") or "") not in RUNTIME_PROVIDERS:
                raise PipelineError(
                    "The Voice Pipeline manifest names a runtime this build does not know "
                    "how to execute on. Nothing was installed.")
        flavours[name] = _read_runtime(name, block)

    return {
        "schema": SCHEMA,
        "version": int(found.get("version") or 0),
        "pinned": bool(found.get("pinned")),
        "notes": str(found.get("notes") or ""),
        "runtime": {
            "python": str(runtime.get("python") or ""),
            # The validated value rather than the constant it used to be. It was
            # a constant while there was one closure and one answer; now it is
            # what the installed record is stamped with, and a record claiming
            # "cpu" for a DirectML closure would make every stale-install check
            # compare the wrong thing.
            "provider": str(runtime.get("provider") or ""),
            "build": int(runtime.get("build") or 0),
            "import_name": str(runtime.get("import_name") or ""),
            "license": str(runtime.get("license") or ""),
            "attribution": str(runtime.get("attribution") or ""),
            "about_bytes": int(runtime.get("about_bytes") or 0),
            "platforms": flavours["onnx"]["platforms"],
            "wanted": list(runtime.get("wanted") or ()),
        },
        # Every flavour under one key, the onnx one included, so code that asks
        # for a flavour by name never has to know which of the two shapes the
        # manifest happened to keep it in.
        "runtimes": flavours,
        "stages": stages,
    }


def _read_runtime(name: str, block: dict) -> dict:
    """One runtime closure, validated -- the platforms and their wheels.

    Split out when the torch flavour arrived, so that a second closure is read
    by the same code as the first rather than by a copy of it that could drift
    into being more permissive. The wheels here are the artifacts whose contents
    get *executed*; there is no flavour for which that is less true.
    """
    platforms = []
    seen = set()
    for entry in (block.get("platforms") or ()):
        if not isinstance(entry, dict):
            raise PipelineError(f"The Voice Pipeline manifest has a {name} runtime "
                                f"platform that is not an object.")
        identifier = str(entry.get("id") or "")
        if not identifier:
            raise PipelineError(f"The Voice Pipeline manifest has a {name} runtime "
                                f"platform with no id.")
        if identifier in seen:
            raise PipelineError(f"The Voice Pipeline manifest names the {name} runtime "
                                f"platform {identifier!r} twice.")
        seen.add(identifier)
        accelerator = str(entry.get("accelerator") or "cpu").casefold()
        if accelerator not in ("cpu", "cuda"):
            raise PipelineError(
                f"The Voice Pipeline manifest's {name} runtime names an accelerator "
                f"({accelerator!r}) this build does not know how to build for.")
        platforms.append({
            "id": identifier,
            "system": str(entry.get("system") or "").casefold(),
            "machines": [str(name_).casefold() for name_ in (entry.get("machines") or ())],
            "python": str(entry.get("python") or ""),
            "accelerator": accelerator,
            # Read through the same validator the stages' artifacts go through,
            # rather than trusted because they are the runtime's. A wheel is the
            # one artifact here whose contents get *executed*, so it is the last
            # place to relax a check (section 20.3).
            "artifacts": [_read_artifact(f"the {name} runtime closure", item)
                          for item in (entry.get("artifacts") or ())],
        })
    return {
        "python": str(block.get("python") or ""),
        "provider": str(block.get("provider") or ""),
        "build": int(block.get("build") or 0),
        "import_name": str(block.get("import_name") or ""),
        "license": str(block.get("license") or ""),
        "attribution": str(block.get("attribution") or ""),
        "about_bytes": int(block.get("about_bytes") or 0),
        "platforms": platforms,
        "wanted": list(block.get("wanted") or ()),
    }


def _read_artifact(owner: str, item) -> dict:
    """One artifact entry, validated. The shapes section 20.3 names, refused.

    Shared by the runtime closure and by every stage, because the checks are the
    same checks and a second copy is a second thing to forget to tighten.
    """
    if not isinstance(item, dict):
        raise PipelineError(f"The Voice Pipeline manifest's {owner} has an artifact that "
                            f"is not an object.")
    local = str(item.get("local_name") or item.get("filename") or "")
    url = str(item.get("url") or "")
    if not local:
        raise PipelineError(f"The Voice Pipeline manifest's {owner} has an artifact with "
                            f"no name.")
    if "/" in local or "\\" in local or local in (".", ".."):
        # Path traversal in an artifact's local name, refused here rather than
        # at the write. The name is joined to a directory this module chose;
        # anything that could leave it is a manifest bug with a filesystem
        # consequence.
        raise PipelineError(f"The Voice Pipeline manifest's {owner} names a file that "
                            f"would not stay in its own folder.")
    if url and not url.startswith("https://"):
        raise PipelineError(f"The Voice Pipeline manifest's {owner} names a file to fetch "
                            f"over something other than HTTPS.")
    digest = str(item.get("sha256") or "").casefold()
    if digest and (len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)):
        raise PipelineError(f"The Voice Pipeline manifest's {owner} names a hash that is "
                            f"not a SHA-256.")
    resolve = item.get("resolve")
    if resolve is not None:
        # An artifact resolved from a publisher's index at install time. Only
        # for wheels that cannot be pinned here at all -- PyTorch's CUDA builds
        # live on a host the manifest-writing machine cannot reach -- and it is
        # validated as strictly as a pinned one so that "resolved" never becomes
        # the loose path somebody reaches for to avoid pinning.
        if not isinstance(resolve, dict):
            raise PipelineError(f"The Voice Pipeline manifest's {owner} has a resolve that "
                                f"is not an object.")
        base = str(resolve.get("index") or "")
        package = str(resolve.get("package") or "")
        if not base.startswith("https://"):
            raise PipelineError(f"The Voice Pipeline manifest's {owner} names a package "
                                f"index that is not HTTPS.")
        if not package:
            raise PipelineError(f"The Voice Pipeline manifest's {owner} has a resolve with "
                                f"no package name.")
        if url or digest:
            raise PipelineError(
                f"The Voice Pipeline manifest's {owner} both pins {package} and resolves "
                f"it. One or the other: a pinned digest that is then overwritten by a "
                f"publisher's is a pin nobody is checking.")
        resolve = {"index": base, "package": package,
                   "version": str(resolve.get("version") or "")}
    return {
        "filename": str(item.get("filename") or local),
        "local_name": local,
        "url": url,
        "bytes": int(item.get("bytes") or 0),
        "sha256": digest,
        "resolve": resolve,
        # Whether this artifact is behind the publisher's access gate, and
        # therefore whether the shared credential may be offered for it. A fact
        # this repository commits to in a manifest, never one a response teaches
        # us at download time (I-VP-25).
        "authorized": bool(item.get("authorized")),
    }


def _read_stage(spec: StageSpec, entry: dict) -> dict:
    """One stage entry, validated against the shape the worker will be handed."""
    artifacts = entry.get("artifacts")
    if not isinstance(artifacts, list):
        raise PipelineError(f"The Voice Pipeline manifest's {spec.id!r} names no artifacts "
                            f"list.")
    seen = set()
    read = []
    for item in artifacts:
        found = _read_artifact(repr(spec.id), item)
        if found["local_name"] in seen:
            raise PipelineError(f"The Voice Pipeline manifest's {spec.id!r} names "
                                f"{found['local_name']!r} twice.")
        seen.add(found["local_name"])
        read.append(found)

    revision = str(entry.get("revision") or "")
    provisional = bool(entry.get("provisional"))
    if revision and not _immutable(revision) and not provisional:
        # Section 13.4, said out loud where it can be enforced: a release whose
        # model identity is a branch name is a release that means something
        # different next week and cannot be reproduced from this repository.
        #
        # ``provisional`` is the deliberate exception, and it is not a loophole
        # because it is not quiet: a stage in that state says so in its own
        # settings row, in its status message and in the log line that installs
        # it. It exists so that a branch nobody could reach a hub from can still
        # be tested on a machine that can, and `tools/pin_pipeline_models.py`
        # turns it into a real pin from there.
        raise PipelineError(
            f"The Voice Pipeline manifest pins {spec.id!r} to {revision!r}, which is not a "
            f"release identity, and does not mark it provisional. It must name an "
            f"immutable revision.")

    contract = entry.get("contract")
    if not isinstance(contract, dict):
        raise PipelineError(f"The Voice Pipeline manifest's {spec.id!r} declares no "
                            f"contract.")
    return {
        "id": spec.id,
        "provisional": provisional,
        "available": bool(entry.get("available", True)),
        "unavailable_reason": str(entry.get("unavailable_reason") or ""),
        "label": str(entry.get("label") or spec.label),
        "order": spec.order,
        "role": spec.role,
        "output_rate_policy": str(entry.get("output_rate_policy")
                                  or spec.output_rate_policy),
        "summary": str(entry.get("summary") or spec.summary),
        "license": str(entry.get("license") or ""),
        "upstream": str(entry.get("upstream") or ""),
        "repo": str(entry.get("repo") or ""),
        "revision": revision,
        "model_id": str(entry.get("model_id") or ""),
        "attribution": str(entry.get("attribution") or ""),
        "about_bytes": int(entry.get("about_bytes") or 0),
        "artifacts": read,
        "required_paths": [str(name) for name in (entry.get("required_paths") or ())],
        "contract": dict(contract),
    }


def _immutable(revision: str) -> bool:
    """Whether a revision names one thing forever. A 40-hex commit does."""
    text = str(revision or "")
    return len(text) == 40 and all(c in "0123456789abcdef" for c in text.casefold())


def runtime_installable(flavour: str = "onnx", accelerator: str = "cpu") -> bool:
    """Whether this build has a runtime closure it could stand behind.

    Every wheel sized and hashed, for a platform this machine matches. This is
    the artifact set whose contents get *executed*, so it is the one place with
    no provisional path at all: an unpinned wheel is a wheel nobody reviewed.

    Asked per flavour because the answer genuinely differs between them. The
    torch closure is pinned for Windows and CPU only -- its CUDA wheels live on
    a host the machine that wrote the manifest could not reach -- and a machine
    this build has no torch closure for is not a machine with no runtime.
    """
    try:
        found = manifest()
    except PipelineError:
        return False
    entry = _platform_entry_or_none(found, flavour, accelerator)
    if entry is None or not entry["artifacts"]:
        return False
    # A resolved artifact has no digest here by definition -- it gets one from
    # the publisher's index at install time, and the download is checked against
    # that. It is still installable; what it is not is *pinned*, which is a
    # different question that :func:`runtime_pinned` answers separately so that
    # a surface can say which of the two it is looking at.
    return all(item["resolve"] or (item["sha256"] and item["bytes"] > 0)
               for item in entry["artifacts"])


def runtime_pinned(flavour: str = "onnx", accelerator: str = "cpu") -> bool:
    """Whether every wheel in this closure is hashed in this repository.

    Kept apart from :func:`runtime_installable` because the difference is the
    whole security story of the CUDA path and it should be sayable in a
    sentence on a settings panel rather than inferred. Pinned means checked
    against a number this repository reviewed; not pinned means checked against
    a number its publisher states today. Both refuse a corrupted download; only
    the first refuses a publisher who changed their mind.
    """
    try:
        found = manifest()
    except PipelineError:
        return False
    entry = _platform_entry_or_none(found, flavour, accelerator)
    if entry is None or not entry["artifacts"]:
        return False
    return all(item["sha256"] and item["bytes"] > 0 for item in entry["artifacts"])


def stage_available(stage_id: str) -> bool:
    """Whether this build ships the stage at all, ignoring the machine.

    A narrower question than :func:`stage_installable`, and they are worth
    keeping apart because their answers mean different things to a surface. A
    stage that is merely not installable *here* -- no pinned runtime for this
    operating system, files not pinned yet -- is still a stage this build has,
    and its settings are still settings. A stage that is not available is one
    the repository does not ship: there is nothing to configure, and every
    control on its panel would be a control with nothing behind it.
    """
    try:
        found = manifest()
    except PipelineError:
        return False
    entry = found["stages"].get(str(stage_id or ""))
    return bool(entry is not None and entry["available"])


def stage_installable(stage_id: str) -> bool:
    """Whether one stage could be installed on this machine right now.

    Per stage rather than all-or-nothing, and that is the whole shape of this
    function. The two stages are independently installable by design (I-VP-02),
    they are pinned by different people at different times, and a build where
    one is ready and the other is not is the ordinary state of a feature being
    brought up -- not a reason to refuse the half that works.
    """
    try:
        found = manifest()
    except PipelineError:
        return False
    entry = found["stages"].get(str(stage_id or ""))
    if entry is None or not entry["available"]:
        return False
    if not runtime_installable():
        return False
    if not entry["revision"] or not entry["artifacts"]:
        return False
    if entry["provisional"]:
        # A provisional stage is allowed to arrive unhashed: what it is checked
        # against is the digest its publisher reports at download time, which
        # the installed record then keeps.
        return all(item["url"] for item in entry["artifacts"])
    return all(item["sha256"] and item["bytes"] > 0 for item in entry["artifacts"])


def stage_unavailable_reason(stage_id: str) -> str:
    """Why a stage cannot be installed, as a sentence somebody can act on."""
    try:
        found = manifest()
    except PipelineError as exc:
        return str(exc)
    entry = found["stages"].get(str(stage_id or ""))
    if entry is None:
        return "Not available in this build."
    if not entry["available"]:
        return entry["unavailable_reason"] or "Not available in this build."
    if not runtime_installable():
        return ("Not available — this build has no pinned Voice Pipeline runtime for this "
                "operating system and Python version.")
    if not entry["revision"] or not entry["artifacts"]:
        return f"Not available — this build has not pinned {entry['label']}'s files yet."
    return ""


def runtime_entry(found: dict, flavour: str = "onnx") -> dict:
    """One flavour's runtime block, whichever key the manifest keeps it under.

    ``onnx`` stays at the top-level ``runtime`` key it has always had rather
    than moving into ``runtimes`` beside its sibling. That asymmetry is on
    purpose: a manifest written before the torch flavour existed is still a
    valid manifest, and a schema change that invalidated one would have made
    every installed runtime stale for a feature its owner never asked for.
    """
    import mc_voice_paths as paths

    name = paths.pipeline_runtime_flavour(flavour)
    entry = (found.get("runtimes") or {}).get(name)
    if not isinstance(entry, dict):
        raise PipelineError(f"This build has no {name} runtime closure.")
    return entry


def _platform_entry_or_none(found: dict, flavour: str = "onnx",
                            accelerator: str = "cpu"):
    """The closure for this machine, built for the accelerator that was asked for.

    Two entries can describe the same operating system and Python and still be
    different closures: the CUDA one differs from the CPU one only in which
    wheels it carries. So the accelerator is part of the lookup rather than a
    filter applied afterwards, and a machine that asks for CUDA and has no CUDA
    entry gets nothing rather than quietly getting the CPU build -- which would
    install successfully, run on the processor, and never mention the card the
    user picked.
    """
    import mc_voice_models as models

    try:
        block = runtime_entry(found, flavour)
    except PipelineError:
        return None
    wanted = str(accelerator or "cpu").casefold()
    system, machine, python = models.current_platform()
    for entry in block["platforms"]:
        machines = tuple(str(name).casefold() for name in (entry.get("machines") or ()))
        if (str(entry.get("system") or "").casefold() == system
                and machine in machines and str(entry.get("python") or "") == python
                and str(entry.get("accelerator") or "cpu").casefold() == wanted):
            return entry
    return None


def pinned() -> bool:
    """Whether this build has a releasable manifest. False in this build.

    Four things have to be true, and section 23 makes the fourth the blocking
    one:

        1  the manifest says it is pinned;
        2  the runtime closure names a platform for this machine and every
           wheel in it is sized and hashed -- the one artifact set here whose
           contents get *executed*, so it is the last place to relax a check;
        3  every stage names an immutable revision and at least one artifact,
           each sized and hashed;
        4  LavaSR's contract carries a MEASURED backend input rate, analysis
           window and context window.

    The fourth is the one that cannot be filled in by a machine with a network
    connection alone. Upstream's README and upstream's code disagree about what
    rate LavaSR interprets its input at; only running it settles that, and until
    somebody has run it there is no honest number to put in a manifest. A build
    that installed anyway would be a build that either resampled 24 kHz speech
    to 16 kHz and played it back a third too slow, or did not resample it and
    played it back half again too fast -- and would do so silently, because
    nothing downstream of the model can tell a wrong clock from a strange voice.
    """
    try:
        found = manifest()
    except PipelineError:
        return False
    if not found["pinned"]:
        return False
    platforms = found["runtime"]["platforms"]
    if not platforms:
        return False
    for entry in platforms:
        # Every wheel, hashed and sized. This is the artifact set whose contents
        # get *executed*, so an unpinned entry here is worse than an unpinned
        # model: a model that arrives wrong makes bad audio, and a wheel that
        # arrives wrong runs.
        if not entry["artifacts"]:
            return False
        if any(not item["sha256"] or item["bytes"] <= 0 for item in entry["artifacts"]):
            return False
    for spec in STAGES:
        entry = found["stages"][spec.id]
        if not entry["revision"] or not entry["artifacts"]:
            return False
        if any(not item["sha256"] or item["bytes"] <= 0 for item in entry["artifacts"]):
            return False
    return _lava_contract_measured(found)


def _lava_contract_measured(found: dict) -> bool:
    """Whether Phase 0 has filled in the numbers LavaSR cannot be run without."""
    contract = found["stages"]["lavasr"]["contract"]
    return bool(contract.get("backend")
                and int(contract.get("backend_input_rate") or 0) > 0
                and int(contract.get("analysis_ms") or 0) > 0
                and int(contract.get("context_ms") or 0) > 0)


def unpinned_reason() -> str:
    """Why :func:`pinned` said no, as a sentence somebody can act on."""
    try:
        found = manifest()
    except PipelineError as exc:
        return str(exc)
    if not _lava_contract_measured(found):
        return ("Not installable — the LavaSR rate and window contract has not been "
                "measured yet, so this build will not download a model it cannot prove it "
                "would play at the right speed.")
    platforms = found["runtime"]["platforms"]
    if not platforms or any(
            not entry["artifacts"]
            or any(not item["sha256"] or item["bytes"] <= 0 for item in entry["artifacts"])
            for entry in platforms):
        return ("Not installable — this build has not pinned a Voice Pipeline runtime "
                "closure yet.")
    for spec in STAGES:
        entry = found["stages"][spec.id]
        if not entry["revision"]:
            return (f"Not installable — this build has not pinned a revision for "
                    f"{entry['label']} yet.")
        if not entry["artifacts"] or any(not item["sha256"]
                                         for item in entry["artifacts"]):
            return (f"Not installable — this build has not pinned {entry['label']}'s files "
                    f"yet.")
    if not found["pinned"]:
        return "Not installable — this build's Voice Pipeline manifest is not marked pinned."
    return ""


# --------------------------------------------------------------------------- #
# What is on this machine
# --------------------------------------------------------------------------- #


def supported_platform() -> bool:
    """Whether the pinned closure covers this machine.

    Answered from the manifest rather than from a hardcoded list, because the
    ONNX runtime this feature wants publishes CPU wheels for Windows, Linux and
    macOS and the reason to support fewer than that would be a measurement
    nobody has taken yet. An unpinned manifest names no platforms and this
    answers False, which is the same answer for the same reason as everything
    else in this build.
    """
    try:
        found = manifest()
    except PipelineError:
        return False
    return _platform_entry_or_none(found) is not None


def runtime_closure_id(flavour: str = "onnx", accelerator: str = "cpu") -> str:
    """The freshness fingerprint of the enhancement runtime.

    Derived from the pinned closure and never declared, in the shape
    :attr:`mc_voice_models.RuntimePlatform.closure_id` established: the platform
    id, then every artifact as ``local_name:sha256``, hashed and truncated.
    Adding, removing, reordering or re-pinning one wheel makes every installed
    runtime stale by arithmetic rather than by somebody remembering to bump a
    number (I-VP-23, section 13.2).

    One thing is hashed here that is not hashed there: the manifest's ``build``.
    The wheels are what the runtime *is*, and the build number is how this
    repository says it changed something about how they are assembled -- the
    path layout, the import check -- which no artifact digest would move.
    """
    import mc_voice_models as models

    try:
        found = manifest()
    except PipelineError:
        return ""
    try:
        block = runtime_entry(found, flavour)
    except (PipelineError, ValueError):
        return ""
    system, machine, python = models.current_platform()
    wanted = str(accelerator or "cpu").casefold()
    for entry in block["platforms"]:
        if not isinstance(entry, dict):
            continue
        machines = tuple(str(name).casefold() for name in (entry.get("machines") or ()))
        if not (str(entry.get("system") or "").casefold() == system
                and machine in machines
                and str(entry.get("python") or "") == python
                and str(entry.get("accelerator") or "cpu").casefold() == wanted):
            continue
        # The flavour leads the fingerprint for every closure except the first
        # one, whose line stays exactly what it was before a second existed.
        # That asymmetry is worth the ugliness: prefixing onnx too would change
        # every already-installed runtime's id and call it stale, which is a
        # forced 138 MB reinstall to add a flavour its owner may never want.
        # Nothing is lost by it -- each flavour keeps its installed record in
        # its own directory, so the two ids are never compared with each other.
        lines = [str(entry.get("id") or "") if flavour == "onnx"
                 else f"{flavour}:{entry.get('id') or ''}"]
        for item in (entry.get("artifacts") or ()):
            if not isinstance(item, dict):
                continue
            local = str(item.get("local_name") or item.get("filename") or "")
            lines.append(f"{local}:{str(item.get('sha256') or '').casefold()}")
        lines.append(f"build:{block['build']}")
        return hashlib.sha256("\n".join(lines).encode("ascii", "replace")).hexdigest()[:16]
    return ""


def stage_closure_id(stage_id: str) -> str:
    """One stage's model identity: its revision and every artifact's digest.

    Separate from the runtime's, because the two go stale for different reasons
    and a user who needs to re-fetch one model should not be told to rebuild a
    runtime (section 13.3).
    """
    try:
        found = manifest()
    except PipelineError:
        return ""
    entry = found["stages"].get(str(stage_id or ""))
    if entry is None:
        return ""
    lines = [entry["id"], entry["repo"], entry["revision"]]
    for item in entry["artifacts"]:
        lines.append(f"{item['local_name']}:{item['sha256']}")
    return hashlib.sha256("\n".join(lines).encode("ascii", "replace")).hexdigest()[:16]


def _record(path: Path) -> dict:
    """One installed record, or ``{}`` for anything that is not one.

    Its presence and its match are what "installed" means. The files being on
    disk is not, which is the rule :data:`mc_voice_paths.INSTALLED_FILENAME`
    states and the reason a half-finished install reads as missing rather than
    as broken.
    """
    try:
        found = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return found if isinstance(found, dict) else {}


def runtime_python() -> "Path | None":
    """The installed enhancement interpreter, or ``None``."""
    root = paths.pipeline_runtime_root() / "env"
    candidates = (root / "Scripts" / "python.exe", root / "bin" / "python3",
                  root / "bin" / "python")
    for candidate in candidates:
        try:
            if candidate.exists():
                return candidate
        except OSError:
            continue
    return None


def runtime_installed() -> dict:
    return _record(paths.pipeline_runtime_manifest())


def stage_installed(stage_id: str) -> dict:
    try:
        return _record(paths.pipeline_stage_manifest(str(stage_id or "")))
    except ValueError:
        return {}


# --------------------------------------------------------------------------- #
# Status
# --------------------------------------------------------------------------- #

INSTALL_STATES = ("not_installed", "installing", "installed", "stale", "corrupt", "error")
RUNTIME_STATES = ("unavailable", "unloaded", "loading", "loaded", "busy", "stopping",
                  "error")
PIPELINE_STATES = ("disabled", "bypassed", "waiting_install", "warming", "ready",
                   "processing", "error")
"""Three vocabularies, kept apart on purpose (section 5.6).

"Installed" is a fact about disk. "Loaded" is a fact about memory right now.
"Ready" is a fact about whether the next reply will be enhanced. Collapsing any
two of them produces the settings page this redesign exists to replace, where a
row says "Installed" and somebody reasonably concludes their speech is being
cleaned when the worker has not been started.
"""


@dataclass(frozen=True)
class StageStatus:
    id: str
    label: str
    order: int
    install_state: str
    message: str
    enabled: bool
    revision: str
    closure_id: str
    license: str
    about_bytes: int


@dataclass(frozen=True)
class Status:
    """Everything the settings surface and the status route draw, and no secret.

    Frozen, and assembled by a function with no side effects: reading status
    must never start a download or load a model, which is the rule
    :func:`mc_voice_engines.installed` states and the reason a settings poll is
    cheap enough to run on a timer.
    """

    supported: bool
    pinned: bool
    runtime_install_state: str
    runtime_message: str
    runtime_closure_id: str
    stages: tuple
    master_enabled: bool
    message: str
    download_bytes: int

    @property
    def runtime_ready(self) -> bool:
        return self.runtime_install_state == "installed"

    def stage(self, stage_id: str) -> "StageStatus | None":
        for found in self.stages:
            if found.id == stage_id:
                return found
        return None

    @property
    def ready(self) -> bool:
        """Whether at least one selected stage could actually run right now."""
        if not self.runtime_ready:
            return False
        return any(found.enabled and found.install_state == "installed"
                   for found in self.stages)


def status() -> Status:
    """What is installed, what is fresh, and what the next turn would do.

    A pure filesystem question. Nothing here starts a process, and nothing here
    is allowed to raise: a settings page that could not be drawn because an
    optional feature's manifest was unreadable would be a settings page nobody
    could use to uninstall it.
    """
    try:
        found = manifest()
    except PipelineError as exc:
        return Status(supported=False, pinned=False, runtime_install_state="error",
                      runtime_message=str(exc), runtime_closure_id="", stages=(),
                      master_enabled=enabled(), message=str(exc), download_bytes=0)

    is_pinned = pinned()
    supported = supported_platform()
    wanted_closure = runtime_closure_id()
    record = runtime_installed()

    if not runtime_installable():
        runtime_state = "not_installed"
        runtime_message = ("Not available — this build has not pinned a Voice Pipeline "
                           "runtime closure for this operating system and Python version.")
    elif not supported:
        runtime_state = "not_installed"
        runtime_message = ("Not available — this build has no pinned Voice Pipeline "
                           "runtime for this operating system and Python version.")
    elif not record:
        runtime_state, runtime_message = "not_installed", "Not installed."
    elif str(record.get("closure") or "") != wanted_closure:
        runtime_state = "stale"
        runtime_message = ("Installed by an older build and needs installing again.")
    elif runtime_python() is None:
        runtime_state = "corrupt"
        runtime_message = "Installed, but its interpreter is missing. Reinstall it."
    else:
        runtime_state, runtime_message = "installed", "Installed."

    stages = []
    for spec in STAGES:
        entry = found["stages"][spec.id]
        wanted = stage_closure_id(spec.id)
        held = stage_installed(spec.id)
        blocked = stage_unavailable_reason(spec.id)
        if blocked and not held:
            state, message = "not_installed", blocked
        elif not held:
            state, message = "not_installed", "Not installed."
        elif str(held.get("closure") or "") != wanted:
            state, message = "stale", "Installed by an older build and needs installing again."
        elif entry["provisional"]:
            # Said on the row rather than only in the manifest. Somebody testing
            # a stage whose model identity is a branch should be able to see
            # that from the page they installed it on.
            state, message = "installed", ("Installed — from the publisher's current "
                                           "branch rather than a pinned release.")
        else:
            state, message = "installed", "Installed."
        stages.append(StageStatus(
            id=spec.id,
            label=entry["label"],
            order=spec.order,
            install_state=state,
            message=message,
            enabled=stage_enabled(spec.id),
            revision=str(held.get("revision") or entry["revision"]),
            closure_id=wanted,
            license=entry["license"],
            about_bytes=entry["about_bytes"],
        ))

    total = sum(entry["about_bytes"] for entry in found["stages"].values())
    return Status(
        supported=supported,
        pinned=is_pinned,
        runtime_install_state=runtime_state,
        runtime_message=runtime_message,
        runtime_closure_id=wanted_closure,
        stages=tuple(stages),
        master_enabled=enabled(),
        message=runtime_message if runtime_state != "installed" else "Installed.",
        download_bytes=total,
    )


def pipeline_state(found: "Status | None" = None) -> str:
    """Which of :data:`PIPELINE_STATES` describes the *next* turn.

    Deliberately about the next turn rather than about this one. The turn in
    flight is frozen onto a snapshot taken before its first sample (I-VP-06),
    so "what would happen now" and "what is happening" are two different
    questions and this is the first. The runtime module answers the second,
    because it is the only thing that knows whether a worker is warm.
    """
    state = found if found is not None else status()
    if not state.master_enabled:
        return "disabled"
    wanted = desired_stages()
    if not wanted:
        return "bypassed"
    for stage_id in wanted:
        held = state.stage(stage_id)
        if held is None or held.install_state != "installed":
            return "waiting_install"
    if not state.runtime_ready:
        return "waiting_install"
    return "ready"


# --------------------------------------------------------------------------- #
# The turn snapshot
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Snapshot:
    """One turn's whole pipeline configuration, frozen before its first sample.

    Frozen is the point (I-VP-06, I-VP-07). Somebody who turns LavaSR off while
    a reply is being spoken has changed the *next* reply, not this one: the
    output sample rate was advertised in this response's headers before the
    first byte, and a stage that stopped running halfway would either change
    that rate mid-stream or quietly emit a different signal at the same rate.
    Both are worse than finishing the sentence the way it started.

    ``output_rate`` is computed here rather than asked of the worker, because
    the browser has to be told it before any audio exists (section 8.5). The
    arithmetic is the stage chain's own: each stage in order says what rate
    comes out of it given what went in.
    """

    enabled: bool
    stage_ids: tuple
    input_rate: int
    output_rate: int
    reason: str

    @property
    def active(self) -> bool:
        """Whether any inference will happen for this turn at all."""
        return bool(self.enabled and self.stage_ids)

    @property
    def path(self) -> tuple:
        """The path preview, as names rather than as a rendered string."""
        return tuple(stage(identifier).label for identifier in self.stage_ids
                     if stage(identifier) is not None)

    def describe(self, source: str = "") -> str:
        """``PocketTTS -> DPDFNet -> LavaSR -> 48 kHz output``, from the truth.

        Generated from this snapshot rather than from the settings, so it can
        never claim a stage ran that was not installed: the snapshot is what
        actually happens, and a stage the user ticked but never installed is not
        in it (section 4.5).
        """
        parts = [source or "Speech"]
        parts.extend(self.path)
        if self.output_rate and self.output_rate != self.input_rate:
            parts.append(f"{self.output_rate // 1000} kHz output")
        else:
            parts.append("Output")
        return " → ".join(parts)


def snapshot(input_rate: int, engine: str = "", found: "Status | None" = None) -> Snapshot:
    """Freeze what this turn will do, given the rate its source speaks at.

    Called once per turn, before the first sample crosses the boundary and
    before the response's headers are written. Everything after it reads the
    snapshot and nothing re-reads the switches.

    A stage the user enabled but has not installed is *left out and said so*
    rather than silently skipped (section 4.2): ``reason`` carries the sentence,
    the surface shows it, and the turn is spoken with whatever else was ready.
    The alternative -- refusing to speak at all because an optional polish is
    missing -- fails the rule that this feature is never fatal to a reply.
    """
    rate = int(input_rate or 0)
    state = found if found is not None else status()

    if engine and engine not in SUPPORTED_ENGINES:
        return Snapshot(enabled=False, stage_ids=(), input_rate=rate, output_rate=rate,
                        reason="")
    if not state.master_enabled:
        return Snapshot(enabled=False, stage_ids=(), input_rate=rate, output_rate=rate,
                        reason="")
    if not state.runtime_ready:
        return Snapshot(enabled=True, stage_ids=(), input_rate=rate, output_rate=rate,
                        reason=state.runtime_message)

    wanted, missing = [], []
    for spec in STAGES:
        held = state.stage(spec.id)
        if held is None or not held.enabled:
            continue
        if held.install_state != "installed":
            missing.append(held.label)
            continue
        wanted.append(spec)

    rate_out = rate
    for spec in wanted:
        rate_out = spec.rate_after(rate_out)

    reason = ""
    if missing:
        names = " and ".join(missing)
        reason = (f"{names} {'is' if len(missing) == 1 else 'are'} switched on but not "
                  f"installed, so {'it was' if len(missing) == 1 else 'they were'} left out "
                  f"of this reply.")
    elif not wanted:
        reason = "Bypassed — no stages enabled."

    return Snapshot(enabled=True, stage_ids=tuple(spec.id for spec in wanted),
                    input_rate=rate, output_rate=rate_out, reason=reason)


# --------------------------------------------------------------------------- #
# Installing
# --------------------------------------------------------------------------- #

COMPONENTS = ("runtime",) + STAGE_IDS
"""What :func:`install` will install, one at a time.

The runtime and the two stages are separate because they go stale for separate
reasons and because a user who needs one model re-fetched should not be told to
rebuild an interpreter (section 13.3, 13.11).
"""


def install(component: str, on_status=None, on_progress=None) -> "Status":
    """Fetch, verify, prove and promote one component. A transaction.

    One component at a time -- the runtime, or one stage -- because the three go
    stale for different reasons and somebody who needs a model re-fetched should
    not be told to rebuild an interpreter.

    Nothing outside a staging directory is touched until every declared byte has
    arrived with the digest this repository committed to and the component has
    been proved: for the runtime, that the staged interpreter starts and the
    inference library imports; for a stage, that a second of synthetic speech
    has been through the model at Pocket's real rate and come back the right
    length. A failure anywhere leaves the machine exactly as it was (13.8).

    That last check is the one this feature exists to be careful about. A
    LavaSR backend that interprets 24 kHz samples as 16 kHz loads perfectly and
    returns perfectly finite numbers; the only thing wrong with it is the
    duration, and the only way to find that out is to measure it.
    """
    import mc_voice_models as models

    wanted = str(component or "")
    if wanted not in COMPONENTS:
        raise PipelineError(f"There is no Voice Pipeline component called {wanted!r}.")
    # The component's own reason first, and the machine's second. A stage this
    # build cannot ship at all would fail the same way on every platform, so
    # answering "not on this operating system" would send somebody looking for a
    # different computer to solve a problem that is not about computers.
    if wanted == "runtime":
        if not supported_platform():
            raise PipelineError(
                "This build has no pinned Voice Pipeline runtime for this operating system "
                "and Python version.")
        if not runtime_installable():
            raise PipelineError(
                "This build has not pinned a Voice Pipeline runtime closure for this "
                "machine.")
    else:
        blocked = stage_unavailable_reason(wanted)
        if blocked:
            raise PipelineError(blocked)

    say = models._narrator(KIND, on_status)
    tick = models._ticker(KIND, on_progress)
    with models._claim(KIND, say, wanted):
        if wanted == "runtime":
            _install_runtime(say, tick)
        elif runtime_python() is None:
            # The runtime is not a second thing to install, it is the thing this
            # stage is proved inside: ``_install_stage`` ends by running the
            # model through a second of speech in the staged interpreter, and
            # there is no interpreter to run it in. Refusing here used to be the
            # answer, and it was the wrong shape -- it sent somebody who had
            # pressed the only button their panel offered to go and find a
            # different button, with the reason written to a log file they were
            # not reading. Nobody wants a stage without the runtime under it, so
            # the prerequisite is installed rather than described.
            #
            # Both halves are inside one claim, so this is still one install
            # that either finishes or leaves the machine as it was, and the two
            # progress spans below are what stops the bar from reaching the end
            # twice.
            if not supported_platform():
                raise PipelineError(
                    "This build has no pinned Voice Pipeline runtime for this operating "
                    "system and Python version, and the enhancement stages cannot run "
                    "without one.")
            if not runtime_installable():
                raise PipelineError(
                    "This build has not pinned a Voice Pipeline runtime closure for this "
                    "machine, and the enhancement stages cannot run without one.")
            say("The Voice Pipeline runtime has to be installed first \u2014 doing that now\u2026")
            _install_runtime(say, _span(tick, 0.0, RUNTIME_SHARE))
            _install_stage(wanted, say, _span(tick, RUNTIME_SHARE, 1.0))
        else:
            _install_stage(wanted, say, tick)
    return status()


RUNTIME_SHARE = 0.75
"""How much of a stage install's progress bar the runtime gets when it is being
installed underneath it.

Not a half, because the two are not the same size: the runtime is a hundred and
fourteen megabytes of wheels and an interpreter build, and the stage on top of
it is a single ONNX file and a self-test. A bar that gave them equal halves
would sit still for most of the first minute and then sprint."""


def _span(tick, low: float, high: float):
    """Report one component's own 0..1 progress inside a slice of the whole.

    Written because ``_install_runtime`` and ``_install_stage`` each believe
    they own the bar and each end by calling ``tick(1.0)``. Chaining them
    without this would show a bar that filled, reset and filled again, which
    reads as "it restarted" rather than "it is two thirds done".
    """
    width = high - low

    def scaled(fraction) -> None:
        try:
            value = float(fraction)
        except (TypeError, ValueError):
            return
        tick(low + width * max(0.0, min(1.0, value)))

    return scaled


def _platform_entry(flavour: str = "onnx") -> dict:
    entry = _platform_entry_or_none(manifest(), flavour)
    if entry is None:
        raise PipelineError(
            "This build has no pinned Voice Pipeline runtime for this operating system and "
            "Python version.")
    return entry


def _resolve_artifacts(entry: dict, say) -> list:
    """Turn any ``resolve`` entries into concrete, digested artifacts.

    Most closures have none of these: every wheel is named and hashed in the
    manifest, and this returns the list unchanged. The exception is the CUDA
    closure, whose wheels are published only on a host the manifest-writing
    machine cannot reach and so cannot be pinned in advance (I-VP-24).

    For those, the publisher's own index is read here, on the machine that *can*
    reach it, and the SHA-256 the publisher states is what the download is then
    checked against -- the same shape the stage models already use, applied to a
    closure. The weaker claim that makes is written into the installed record
    rather than left implied: see ``mc_voice_wheelindex``.
    """
    import mc_voice_wheelindex as wheelindex

    rows = list(entry["artifacts"])
    pending = [item for item in rows if item.get("resolve")]
    if not pending:
        return rows

    tags = wheelindex.platform_tags(
        str(entry.get("python") or ""), str(entry.get("system") or ""),
        (list(entry.get("machines") or ["amd64"]) or ["amd64"])[0])
    say("Asking the publisher which files this machine needs…")
    out = []
    for item in rows:
        wanted = item.get("resolve")
        if not wanted:
            out.append(item)
            continue
        package = str(wanted.get("package") or "")
        base = str(wanted.get("index") or "")
        if not base.startswith("https://"):
            raise PipelineError(
                f"The manifest points {package or 'a wheel'} at an index that is not "
                f"HTTPS. Nothing was downloaded.")
        try:
            found = wheelindex.choose(_read_index(base), base, package,
                                      str(wanted.get("version") or ""), tags)
        except wheelindex.IndexError_ as exc:
            raise PipelineError(str(exc)) from None
        except OSError as exc:
            raise PipelineError(
                f"The publisher's index for {package} could not be read "
                f"({exc.__class__.__name__}). Nothing was installed.") from None
        logger.info("Model Chain: the Voice Pipeline resolved %s from the publisher's "
                    "index — %s", package, found["filename"])
        out.append({**item, "filename": found["filename"],
                    "local_name": found["filename"], "url": found["url"],
                    "sha256": found["sha256"], "bytes": 0})
    return out


def _read_index(url: str) -> str:
    """Fetch one index page, through the one module allowed to reach the network.

    Not with ``urllib`` here, which is invariant I-4: there is a single door out
    of this feature and it is the one with the hashes behind it. A transport in
    this module would be a second way for bytes to arrive with nothing checking
    them, and the fact that these particular bytes are HTML rather than a wheel
    is exactly the argument that would erode the rule.
    """
    import mc_voice_models as models

    return models.read_index_page(url)


def _artifacts(entries) -> list:
    """Manifest entries as :class:`mc_voice_models.Artifact` objects.

    Built here rather than through :func:`mc_voice_models._read_artifact`,
    because that reader serves the speech manifest's schema and does not read an
    ``authorized`` key at all. The pipeline's model artifacts may be behind an
    access gate, and whether one is is a fact this repository commits to in a
    manifest rather than something a response teaches us at download time.
    """
    import mc_voice_models as models

    return [models.Artifact(filename=item["filename"], local_name=item["local_name"],
                            url=item["url"], size=item["bytes"] or None,
                            sha256=item["sha256"] or None,
                            authorized=bool(item.get("authorized")))
            for item in entries]


def _install_runtime(say, tick) -> None:
    import mc_voice_models as models

    entry = _platform_entry()
    staging = paths.pipeline_staging_for("runtime", uuid.uuid4().hex[:8])
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)
    try:
        say("Checking what the publishers say these files are…")
        artifacts = _artifacts(entry["artifacts"])
        expectations = models._expectations(artifacts, say)
        models._make_room(artifacts, staging, expectations)
        digests = models._fetch_all(artifacts, staging, say, tick, 0.7, expectations)

        say("Building the isolated enhancement runtime…")
        found = manifest()["runtime"]
        chosen = models.RuntimePlatform(
            identifier=str(entry.get("id") or ""),
            system=str(entry.get("system") or "").casefold(),
            machines=tuple(str(name).casefold() for name in (entry.get("machines") or ())),
            python=str(entry.get("python") or ""),
            artifacts=tuple(artifacts))
        models._build_environment(staging, staging, chosen,
                                  import_name=found["import_name"] or "onnxruntime")
        for item in artifacts:
            # The wheels are inputs, not part of the installation. Left in the
            # staging tree they would be promoted along with it and sit in the
            # user's data directory forever, a second copy of a closure that is
            # already unpacked beside them.
            try:
                (staging / item.local_name).unlink()
            except OSError:
                pass
        tick(0.9)

        say("Checking that the enhancement runtime starts…")
        _run_staged(_staged_python(staging),
                    ["-c", f"import {found['import_name']}; print('ok')"],
                    "the Voice Pipeline runtime")
        models._write_json(staging / paths.INSTALLED_FILENAME, {
            "schema": SCHEMA,
            "closure": runtime_closure_id(),
            "platform": chosen.identifier,
            "python": chosen.python,
            "provider": found["provider"],
            "build": found["build"],
            "license": found["license"],
            "artifacts": {name: digest for name, digest in digests.items()},
        })
        models._promote(staging, paths.pipeline_runtime_root())
        tick(1.0)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    say("The Voice Pipeline runtime is installed.")
    logger.info("Model Chain: the Voice Pipeline runtime is installed — closure %s",
                runtime_closure_id())


def _run_staged(interpreter: Path, arguments: list, what: str, timeout: float = 300):
    """Run something in the staged runtime, with no credential and no network.

    This feature's own rather than :func:`mc_voice_models._run_staged`, and the
    difference is the whole reason it exists: that helper composes the *speech*
    runtime's environment onto ``os.environ``, which leaves an inherited
    ``HF_TOKEN`` in place. That is right for a Kokoro smoke test and wrong here
    -- the process being started is the enhancement worker, and I-VP-21 says it
    never sees a credential, at install time as much as at inference time.

    It also *checks the exit code*, which the shared helper deliberately does
    not: there it is one step of a longer smoke test that reads the output
    afterwards, and here a non-zero exit is the answer.
    """
    import subprocess

    environ = dict(os.environ)
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        environ.pop(name, None)
    environ.update(worker_environment())
    try:
        found = subprocess.run(  # noqa: S603 - a path this module built
            [str(interpreter)] + [str(item) for item in arguments],
            capture_output=True, text=True, env=environ, timeout=timeout,
            cwd=str(paths.extension_root()))
    except Exception as exc:
        logger.warning("Model Chain: the Voice Pipeline could not run %s (%s)",
                       what, exc.__class__.__name__)
        raise PipelineError(f"The staged Voice Pipeline runtime could not be started "
                            f"({exc.__class__.__name__}). Nothing was installed.") from None
    if found.returncode != 0:
        logger.warning("Model Chain: the Voice Pipeline's %s failed with exit code %s.\n"
                       "  stderr: %s\n  stdout: %s", what, found.returncode,
                       (found.stderr or "")[-1200:] or "(nothing)",
                       (found.stdout or "")[-1200:] or "(nothing)")
        raise PipelineError(f"{what} did not run on this machine. Nothing was installed.")
    return found.stdout or ""


def _staged_python(staging: Path) -> Path:
    root = staging / "env"
    for candidate in (root / "Scripts" / "python.exe", root / "bin" / "python3",
                      root / "bin" / "python"):
        if candidate.exists():
            return candidate
    raise PipelineError("The staged Voice Pipeline runtime has no interpreter in it. "
                        "Nothing was installed.")


def _install_stage(stage_id: str, say, tick) -> None:
    import mc_voice_models as models

    if runtime_python() is None:
        raise PipelineError(
            "The Voice Pipeline runtime is not installed yet, and a model that cannot be "
            "run cannot be proved. Install the runtime first.")
    entry = manifest()["stages"][stage_id]
    staging = paths.pipeline_staging_for(stage_id, uuid.uuid4().hex[:8])
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)
    try:
        say(f"Checking what the publisher says {entry['label']}'s files are…")
        artifacts = _artifacts(entry["artifacts"])
        expectations = models._expectations(artifacts, say)
        models._make_room(artifacts, staging, expectations)
        digests = models._fetch_all(artifacts, staging, say, tick, 0.75, expectations)
        for name in entry["required_paths"]:
            if not (staging / name).exists():
                raise PipelineError(
                    f"{entry['label']} was downloaded but {name} is not in it. Nothing was "
                    f"installed.")

        say(f"Checking that {entry['label']} runs on this machine…")
        _self_test(stage_id, staging)
        tick(0.95)

        models._write_json(staging / paths.INSTALLED_FILENAME, {
            "schema": SCHEMA,
            "closure": stage_closure_id(stage_id),
            "id": stage_id,
            "repo": entry["repo"],
            "revision": entry["revision"],
            "model_id": entry["model_id"],
            "license": entry["license"],
            "attribution": entry["attribution"],
            "contract": entry["contract"],
            "provisional": entry["provisional"],
            "model_file": (entry["required_paths"][0] if entry["required_paths"]
                           else (entry["artifacts"][0]["local_name"]
                                 if entry["artifacts"] else "")),
            "artifacts": dict(digests),
        })
        models._promote(staging, paths.pipeline_stage_root(stage_id))
        tick(1.0)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    say(f"{entry['label']} is installed.")
    logger.info("Model Chain: %s is installed — revision %s", entry["label"],
                entry["revision"][:12])


def _self_test(stage_id: str, staging: Path) -> dict:
    """Run the staged model once, on the staged interpreter, before promoting it.

    Needs no sound hardware and no network (section 13.10). What it proves is
    narrow and load-bearing: the model loads from a local directory, it returns
    finite numbers, and -- for LavaSR -- one second of 24 kHz audio comes back as
    one second of 48 kHz audio. An installation that fails that last check is
    refused after everything has already been downloaded, which is the right
    place to fail: the alternative is a user discovering it as a voice that
    speaks too slowly.
    """
    import mc_voice_models as models

    config = dict(stage_config(stage_id))
    config["intraop"] = threads()
    config["interop"] = INTEROP_THREADS
    # On the processor, whatever the placement setting says, and deliberately.
    # What this proves is that the downloaded model loads and returns finite
    # numbers -- a claim about the file, not about the machine's graphics card.
    # Running it on a card would let a driver that is busy with an image
    # generation fail an install of a model that is perfectly good, and the
    # message somebody got would be about the model.
    config["provider"] = "CPUExecutionProvider"
    config["adapter"] = 0
    interpreter = runtime_python()
    if interpreter is None:
        raise PipelineError("The Voice Pipeline runtime is not installed.")
    report = _run_staged(
        interpreter,
        [str(paths.pipeline_worker_script()), "--selftest",
         "--stage", f"{stage_id}={staging}",
         "--config", json.dumps({stage_id: config, "test_rate": 24000})],
        f"the Voice Pipeline's {stage_id} model")
    # The last non-empty line of stdout, which is where the worker prints its
    # one JSON object. Anything a library wrote before it is ignored rather than
    # parsed, because a warning on stdout is not a reason to refuse an install.
    lines = [row for row in report.splitlines() if row.strip()]
    try:
        found = json.loads(lines[-1]) if lines else {}
    except ValueError:
        raise PipelineError(f"The {stage_id} self-test did not answer with a result. "
                            f"Nothing was installed.") from None
    if not isinstance(found, dict):
        raise PipelineError(f"The {stage_id} self-test answered with something this build "
                            f"does not read. Nothing was installed.")
    if not found.get("ok"):
        raise PipelineError(
            f"The {stage_id} self-test failed ({found.get('error') or 'no reason given'}). "
            f"Nothing was installed.")
    return found


def uninstall(component: str) -> None:
    """Remove one component's files, after making sure nothing has them open.

    The runtime is refused while a stage still needs it, and every path is
    checked to be inside this feature's own tree before anything is deleted:
    an uninstaller that could reach another engine's runtime is an uninstaller
    that can break a working voice (section 13.11).
    """
    import mc_voice_pipeline_runtime as runtime

    wanted = str(component or "")
    if wanted not in COMPONENTS:
        raise PipelineError(f"There is no Voice Pipeline component called {wanted!r}.")
    runtime.stop(f"{wanted} is being removed")
    if wanted == "runtime":
        found = status()
        held = [stage.label for stage in found.stages
                if stage.install_state == "installed"]
        if held:
            raise PipelineError(
                f"{' and '.join(held)} still {'needs' if len(held) == 1 else 'need'} the "
                f"Voice Pipeline runtime. Remove {'it' if len(held) == 1 else 'them'} "
                f"first.")
        target = paths.pipeline_runtime_root()
    else:
        target = paths.pipeline_stage_root(wanted)
    if not paths.pipeline_inside(target):
        raise PipelineError("That is not a Voice Pipeline folder, so nothing was removed.")
    shutil.rmtree(target, ignore_errors=True)
    logger.info("Model Chain: the Voice Pipeline's %s was removed", wanted)


# --------------------------------------------------------------------------- #
# Worker environment
# --------------------------------------------------------------------------- #


def worker_environment() -> dict:
    """What the enhancement worker is started with, and what it is not.

    The absences are the contract (I-VP-21, I-VP-26, section 13.7). There is no
    ``HF_TOKEN``, no ``HUGGING_FACE_HUB_TOKEN``, no ``HUGGINGFACE_TOKEN`` and no
    bearer header anywhere in this dictionary, and there is nothing that would
    let a model library resolve a name over the network if one were passed. The
    worker is handed verified local directories and could not fetch a missing
    file if it wanted to.

    The visible-device blanking stays, and it is worth being precise about what
    it now means. CUDA, HIP and ROCm are still blanked unconditionally: nothing
    here executes through any of them, and a library that would have found a
    card through one of those variables has no business finding one. The
    placement setting reaches a card through DirectML, which enumerates DXGI
    adapters and does not read these -- so the two are not in tension, and the
    blanking keeps meaning what it always did.

    ``ONNXRUNTIME_FORCE_CPU`` is the one that had to become conditional. It is
    set while every stage is on the processor and dropped as soon as one is not,
    because a variable that says "run on the CPU" sitting in the environment of
    a worker that has been asked for a graphics card is either a contradiction
    ONNX Runtime ignores today or one it honours tomorrow. Neither is something
    to leave in place: the second would make the placement setting silently do
    nothing, which is the exact failure the session check exists to prevent.

    The thread caps track the *setting* rather than the constant, and that is
    the same correction :func:`threads` was written for. OpenMP sizes its pool
    from this environment before any of this feature's code runs, so a cap
    pinned at the released two while the session is asked for sixteen is a pool
    that belongs to neither number.
    """
    budget = threads()
    on_a_card = any(placement(spec.id)[0] != "CPUExecutionProvider" for spec in STAGES)
    found = {
        "CUDA_VISIBLE_DEVICES": "",
        "HIP_VISIBLE_DEVICES": "",
        "ROCR_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": str(budget),
        "MKL_NUM_THREADS": str(budget),
        "OPENBLAS_NUM_THREADS": str(budget),
        "PYTHONNOUSERSITE": "1",
        "PYTHONUNBUFFERED": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "NO_PROXY": "*",
    }
    if not on_a_card:
        found["ONNXRUNTIME_FORCE_CPU"] = "1"
    return found


def stage_paths() -> dict:
    """The verified local directory for every installed stage, by id.

    What the worker is told, and the only thing it is told about where a model
    lives. The browser sends stage ids; this turns ids into paths from the
    installed records; nothing anywhere turns a browser-supplied string into a
    filesystem path (section 20.2).
    """
    found = {}
    state = status()
    for held in state.stages:
        if held.install_state != "installed":
            continue
        try:
            found[held.id] = str(paths.pipeline_stage_root(held.id))
        except ValueError:
            continue
    return found


def stage_config(stage_id: str) -> dict:
    """The release contract the worker enforces for one stage.

    Read out of the manifest rather than defaulted in the worker, because these
    are the Phase-0 measurements and a default in the worker would be a number
    somebody guessed surviving into a release nobody re-measured.
    """
    try:
        found = manifest()
    except PipelineError:
        return {}
    entry = found["stages"].get(str(stage_id or ""))
    if entry is None:
        return {}
    contract = dict(entry["contract"])
    contract["model_id"] = entry["model_id"]
    contract["revision"] = entry["revision"]
    # The file the worker opens, by the name it was installed under. Named here
    # rather than guessed there: the worker is handed a directory and must not
    # have to decide which file in it is the model (section 20.2).
    installed = stage_installed(stage_id)
    names = [str(name) for name in (entry["required_paths"] or ())]
    if not names:
        names = [item["local_name"] for item in entry["artifacts"]]
    contract["model_file"] = str((installed.get("model_file") or "")
                                 or (names[0] if names else ""))
    return contract
