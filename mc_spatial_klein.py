"""Spatial Layout on FLUX.2 Klein: the host side of regional conditioning.

The same canvas Krea 2 uses, consumed completely differently. Krea's backend
turns the boxes into *text* -- a structured prompt with coordinates written into
it -- and the model reads them as a very strong hint. Klein's backend turns them
into *conditioning geometry*, so a region's prompt is attached to the part of
the output the box covers rather than described to the whole of it.

    Spatial Layout document (one editor, one document)
          |
          +-- Krea 2      -> structured BBOX prompt      mc_spatial / krea.spatial
          |
          +-- Klein 9B    -> regional conditioning       this module

What this module is responsible for
-----------------------------------
Four things, and it is worth naming them separately because only the last one is
uncertain:

1. **What the source is.** ImageStitch, read from this generation's own script
   arguments, and nothing else (§3, §16).
2. **Which mode runs.** The availability matrix and the Auto resolver, applied
   server-side for UI and API alike (§4, §5, §10, §40).
3. **Lifecycle.** References cleared before and after, toggles restored, hooks
   removed, on success and on failure alike (§18, §41).
4. **Regional conditioning itself.** Which host mechanism actually attaches a
   prompt to a rectangle.

The first three are ordinary code with ordinary tests. The fourth is the one the
design intent puts a spike in front of, and it is treated accordingly below.

The backend is probed, never assumed
------------------------------------
§28 lists three ways a host might support this and §34 says what to do when it
supports none of them: **fail before sampling, with the detected architecture and
engine in the message.** Not generate globally and hope nobody looks.

So regional conditioning lives behind :class:`RegionalBackend`, several
candidates are registered, and :func:`select_backend` asks each in turn whether
the *loaded* engine actually exposes what it needs. A checkpoint whose header
says Klein and whose engine does not implement the path is a real case -- §34
names repacked and GGUF builds specifically -- and it is the case a capability
probe catches and a name check does not.

What is deliberately not here
-----------------------------
No natural-language position hint masquerading as regional conditioning. §28
forbids it in the strongest terms in the document, and it forbids it for a
reason worth restating: "lamp on the right side" appended to a global prompt
produces an image, the image sometimes has the lamp on the right, and nothing
about the result tells you that the feature never ran. A failure that looks like
a success is worse than a failure.
"""

from __future__ import annotations

import contextlib
import logging

import mc_arch
import mc_references
from prompt_master import spatial

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

VERSION = 1
"""The version of the Klein spatial behaviour, recorded in infotext.

Separate from the layout document's version and from the backend's own. This one
answers "what did this build do with those boxes", which changes for different
reasons than "what shape was the document".
"""

ARCHITECTURES = ("flux2_9b",)
"""Which architectures get the Klein spatial backend.

9B only, as §7 asks. Klein 4B is identical in everything :mod:`mc_arch` records
about it -- same alignment, same reference flag, same inverse edit toggle -- so
adding it is this tuple and a test, and it is deliberately not done here: §7
makes 4B a non-goal "unless it falls out safely from the same architecture
adapter *and test coverage*", and the test coverage is the half that needs a
card rather than a keystroke.
"""


BACKEND_AUTO = "auto"
BACKEND_KREA = "krea2"
BACKEND_KLEIN = "flux2_9b"
BACKENDS_OFFERED = (BACKEND_AUTO, BACKEND_KREA, BACKEND_KLEIN)
"""Which backend the panel is being asked to show, as the user may pin it.

Auto follows the checkpoint and is the default. The other two are an override,
and they exist because detection has a real blind spot in a live page: a
checkpoint *selected* and not yet loaded is not something every host announces
in a way this extension can read. Forge Neo builds its model chooser in
``modules_forge.main_entry`` rather than as an A1111 quicksetting, so the
component this panel watches may simply not be there -- and the panel then keeps
describing the checkpoint that is still resident until the next generation
loads the new one.

Pinning it is not a promise about what will be loaded, and nothing here treats
it as one: it decides what the *panel shows*, and the generation still checks
the engine that actually loaded and says so if the two disagree. An override
that could make a Krea checkpoint take the Klein path would be a worse bug than
the one it fixes.
"""

BACKEND_LABELS = {
    BACKEND_AUTO: "Auto — follow the loaded checkpoint",
    BACKEND_KREA: "Krea 2 — structured BBOX prompt",
    BACKEND_KLEIN: "FLUX.2 Klein 9B — regional conditioning",
}


def normalise_backend(value) -> str:
    """One backend preference, from whatever a control or a file supplied."""
    text = str(value or "").strip().casefold()
    return text if text in BACKENDS_OFFERED else BACKEND_AUTO


def is_klein(arch) -> bool:
    """Whether ``arch`` is an architecture this backend serves."""
    return bool(arch is not None and getattr(arch, "key", "") in ARCHITECTURES)


def chosen_architecture(preference=BACKEND_AUTO, p=None):
    """Which backend the panel should show, honouring an explicit pin.

    Auto detects. Anything else is the user telling this panel which checkpoint
    they have, and is taken at face value *for the panel only* -- see
    :data:`BACKENDS_OFFERED` for why that is a display decision and not a
    routing one.
    """
    preference = normalise_backend(preference)
    if preference != BACKEND_AUTO:
        return mc_arch.by_key(preference)
    return intended_architecture(p) if p is not None else loaded_architecture()


def loaded_architecture():
    """The architecture of the loaded Stage-1 model.

    The engine's own answer first, and the selected checkpoint's header only as a
    fallback -- the same order :func:`mc_creative_krea.checkpoint_objection` uses
    and for the same reason. §8 is explicit that this is architecture detection
    and not a filename check: a Klein checkpoint called ``final_v3_FIXED.gguf``
    is still a Klein checkpoint, and a Krea 2 checkpoint with "klein" in its name
    is still not one.
    """
    try:
        from modules import shared

        found = mc_arch.detect_loaded_engine()
        if found is mc_arch.UNKNOWN:
            found = mc_arch.detect_from_checkpoint_name(shared.opts.sd_model_checkpoint)
        return found
    except Exception:
        logger.debug("Model Chain: could not identify the image checkpoint for Klein "
                     "Spatial Layout", exc_info=True)
        return mc_arch.UNKNOWN


def intended_architecture(p=None):
    """The architecture this generation is *about to* run on.

    Asked in ``before_process``, which is the one hook early enough to redirect
    the prompt and too early to ask the model: Forge loads the checkpoint after
    it, so the resident engine at that moment can still be the previous one. So
    the selected checkpoint answers first here -- the per-generation override if
    there is one, the global setting otherwise -- and the loaded engine is the
    fallback for a build whose header cannot be read.

    Exactly the reverse of :func:`loaded_architecture`, which is asked in
    ``process`` and later, where the model is resident and there is nothing left
    to predict. Both exist because both moments are real, and the second one is
    the one that decides.
    """
    name = ""
    try:
        overrides = getattr(p, "override_settings", None) or {}
        name = str(overrides.get("sd_model_checkpoint") or "")
    except Exception:
        name = ""

    if not name:
        try:
            from modules import shared

            name = str(getattr(shared.opts, "sd_model_checkpoint", "") or "")
        except Exception:
            name = ""

    if name:
        found = mc_arch.detect_from_checkpoint_name(name)
        if found is not mc_arch.UNKNOWN:
            return found
    return loaded_architecture()


def active(p=None) -> bool:
    """Whether the Klein spatial backend is the one this generation would use."""
    return is_klein(intended_architecture(p) if p is not None
                    else loaded_architecture())


# --------------------------------------------------------------------------- #
# The source
# --------------------------------------------------------------------------- #


def source_for(p) -> spatial.SpatialSource:
    """This generation's ImageStitch images, as a :class:`SpatialSource`.

    One call into :func:`mc_references.runtime_source`, which is the authoritative
    path, and the result wrapped in the model-agnostic type the rest of the
    feature speaks. Never raises: a source that cannot be read is a generation
    with no source, which is a perfectly ordinary thing to be.
    """
    try:
        enabled, images, max_dim = mc_references.runtime_source(p)
    except Exception:
        logger.debug("Model Chain: could not read the ImageStitch source for Klein "
                     "Spatial Layout", exc_info=True)
        return spatial.NO_SOURCE
    return spatial.SpatialSource(images=tuple(images), max_dim=int(max_dim or 0),
                                 origin="imagestitch", enabled=bool(enabled))


def request_for(p, serialized, *, enabled: bool, requested_mode=spatial.AUTO,
                compose_mode: str = "") -> spatial.SpatialRequest:
    """One press of Generate, parsed and resolved.

    Raises :class:`prompt_master.spatial.ModeUnavailable` when an explicitly
    chosen image-required mode has no image behind it any more. The caller turns
    that into a refused generation rather than a different one -- see §10, and
    :func:`compatibility_error` for the sentence it gets.
    """
    request = spatial.request_from(
        serialized,
        enabled=enabled,
        requested_mode=requested_mode,
        source=source_for(p),
        width=int(getattr(p, "width", 0) or 0),
        height=int(getattr(p, "height", 0) or 0),
        compose_mode=compose_mode,
    )
    return spatial.resolved_request(request)


# --------------------------------------------------------------------------- #
# Regional conditioning backends
# --------------------------------------------------------------------------- #


class RegionalConditioningUnavailable(RuntimeError):
    """The loaded engine exposes no way to condition a rectangle.

    §34: a compatibility error before sampling, naming what was detected. It is
    an exception and not a warning because the alternative -- generating globally
    with the boxes silently discarded -- produces a plausible image that is not
    the one that was asked for, and there is no way to tell afterwards.
    """


class RegionalBackend:
    """One way of attaching a prompt to a rectangle, and how to tell if it works.

    Subclasses answer two questions. :meth:`available` asks the *loaded engine*
    whether the mechanism is there -- a probe, not a version check, because the
    thing being asked about is a code path in a host this extension does not ship
    and cannot pin. :meth:`apply` installs it for one job and yields; everything
    it changed is unwound when the block leaves, however it leaves.

    ``name`` and ``version`` reach the image's infotext. That pairing is the
    point of recording them: two images made from the same layout by two
    different mechanisms are two different experiments, and without the backend
    name in the file there is no way to know afterwards which one produced which.
    """

    name = "unknown"
    version = 0
    summary = ""

    def available(self, model) -> bool:
        raise NotImplementedError

    @contextlib.contextmanager
    def apply(self, conditioning, compiled, model=None, p=None):
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name} v{self.version}>"


def _host_sampling_module():
    """Forge Neo's sampling core, under whichever name this build uses.

    ``backend.sampling.sampling_function`` is Forge Neo's; ``ldm_patched.modules
    .samplers`` is what older Forge builds called the same ComfyUI-derived
    module. Tried in order and reported as absent rather than raised, because a
    host that has neither is a host this backend simply does not serve.
    """
    for module_name in ("backend.sampling.sampling_function",
                        "backend.modules.k_diffusion_extra",
                        "ldm_patched.modules.samplers"):
        try:
            import importlib

            return importlib.import_module(module_name)
        except Exception:
            continue
    return None


class AreaConditioningBackend(RegionalBackend):
    """§28 Path 1: the host's own area metadata on a conditioning entry.

    ComfyUI's ``ConditioningSetArea`` is the shape of this, and Forge's sampling
    core is a fork of the same file: ``get_area_and_mult`` reads ``'area'`` and
    ``'strength'`` off each conditioning entry and crops the model input to it.
    Where that function exists, a region is one more entry in the positive
    conditioning list with an area attached.

    Probed by *finding the function*, not by trusting the family name. §15 warns
    against assuming ComfyUI's geometry here, and the same caution applies to
    assuming its plumbing.
    """

    name = "host-area-conditioning"
    version = 1
    summary = ("the host sampler's own area metadata on positive conditioning "
               "entries")

    def available(self, model) -> bool:
        module = _host_sampling_module()
        if module is None:
            return False
        return callable(getattr(module, "get_area_and_mult", None))

    @contextlib.contextmanager
    def apply(self, conditioning, compiled, model=None, p=None):
        installed = _install_conditioning_regions(conditioning, compiled, model, p,
                                                  use_mask=False)
        try:
            yield installed
        finally:
            installed.remove()


class MaskConditioningBackend(RegionalBackend):
    """§28 Path 2: a spatial mask attached to positive conditioning.

    The same list, a different key. Where the host honours ``'mask'`` and
    ``'mask_strength'`` it can weight a region with soft edges rather than
    cropping to a rectangle, which is strictly better for overlapping boxes --
    §13 allows overlapping areas to receive multiple conditions, and a mask
    combines where a crop competes.

    Ranked below the area path only because the area path is the one whose
    behaviour is most widely understood; both are honest implementations of the
    same intent.
    """

    name = "host-mask-conditioning"
    version = 1
    summary = "a spatial mask attached to positive conditioning entries"

    def available(self, model) -> bool:
        module = _host_sampling_module()
        if module is None:
            return False
        function = getattr(module, "get_area_and_mult", None)
        if not callable(function):
            return False
        try:
            import inspect

            return "mask" in inspect.getsource(function)
        except Exception:
            # Source is not always readable -- a compiled or patched host, a
            # zipped install. Unreadable is not the same as absent, but this
            # backend cannot confirm itself without it, so it declines and lets
            # the area path answer instead.
            return False

    @contextlib.contextmanager
    def apply(self, conditioning, compiled, model=None, p=None):
        installed = _install_conditioning_regions(conditioning, compiled, model, p,
                                                  use_mask=True)
        try:
            yield installed
        finally:
            installed.remove()


class ComposableRegionBackend(RegionalBackend):
    """Regional conditioning built on the host's own composable diffusion.

    The backend for A1111-derived hosts, which is what Forge Neo turns out to be
    at the hook this extension can reach. Its conditioning is a
    ``MulticondLearnedConditioning`` whose batch entries are lists of
    ``ComposableScheduledPromptConditioning`` -- schedules and a weight, and no
    geometry anywhere. That is the ``AND`` composition A1111 has always had: each
    composable prompt is evaluated by the model separately and their results are
    blended by weight.

    Blended *globally* by weight, which is the one thing that makes it not
    regional. So this backend does two things:

    1. appends each region as one more composable prompt, which the host then
       evaluates for free because evaluating composable prompts is what it
       already does;
    2. replaces the blend with a spatially masked one, so a region's
       contribution lands only inside its rectangle.

    The masked blend is derived from the host's own blend rather than
    reimplemented. ``combine_denoised`` is called once for the global prompt
    alone and once per region, and the difference between them is what gets
    masked -- so whatever that function knows about CFG scale, skipped
    unconditional passes and edit models stays true, and none of it has to be
    guessed at from outside. Guessing at it is what put an ``area`` key into a
    structure that has never had one.

    **The cost is real and is the reason this is not free.** Each region is
    another conditioning the model evaluates at every step, so four regions is
    roughly five model evaluations a step instead of one. That is inherent to
    the technique rather than to this implementation: separate evaluation is
    exactly what makes a region's contribution separable enough to mask.
    """

    name = "composable-masked-regions"
    version = 1
    summary = ("the host's composable diffusion, with the blend between prompts "
               "masked to each region's rectangle")

    def available(self, model) -> bool:
        """Torch, and nothing else.

        Deliberately the least demanding probe of the three, and deliberately
        last in :data:`BACKENDS`: this one works against a structure every
        A1111-derived host has, so if it were asked first it would answer for
        hosts whose native area path is both cheaper and better.
        """
        import importlib.util

        return importlib.util.find_spec("torch") is not None

    @contextlib.contextmanager
    def apply(self, conditioning, compiled, model=None, p=None):
        installed = _install_composable_regions(
            conditioning, compiled, model, p,
            cutoff=region_cutoff(getattr(p, "steps", 0),
                                 getattr(compiled, "region_percent", 100)))
        try:
            yield installed
        finally:
            installed.remove()


BACKENDS: tuple[RegionalBackend, ...] = (
    AreaConditioningBackend(),
    MaskConditioningBackend(),
    ComposableRegionBackend(),
)
"""Every regional-conditioning mechanism this build knows, in preference order.

§28's Path 3 -- a transformer attention intervention -- is deliberately not in
this tuple yet. It is the only one of the three that cannot be implemented
without knowing the shape of the model's attention call, it is the one that has
to be written against internals rather than against an interface, and R1's
mitigation is explicit that it should be reached for "only if necessary". The
spike in ``tools/klein_regional_spike.py`` is what decides whether it is
necessary; until it has been run against a real 9B checkpoint, adding a third
candidate would mean shipping the least verifiable mechanism as a silent
fallback for the other two.
"""


def select_backend(model=None) -> RegionalBackend | None:
    """The first registered backend the loaded engine can actually support.

    ``None`` when none of them can, which the caller turns into
    :class:`RegionalConditioningUnavailable` before sampling starts.
    """
    if model is None:
        try:
            from modules import shared

            model = shared.sd_model
        except Exception:
            model = None

    for backend in BACKENDS:
        try:
            if backend.available(model):
                return backend
        except Exception:
            logger.debug("Model Chain: the %s regional backend could not be probed",
                         backend.name, exc_info=True)
    return None


def usable_backends(model=None) -> tuple:
    """Every backend whose mechanism this host exposes, in preference order.

    A pre-filter and not a decision. :func:`regional_conditioning` tries them in
    turn against the conditioning it was actually handed, because "the host has
    an area path somewhere" and "a region can be written into *this* object" are
    different questions and only the second one matters.
    """
    if model is None:
        try:
            from modules import shared

            model = shared.sd_model
        except Exception:
            model = None

    found = []
    for backend in BACKENDS:
        try:
            if backend.available(model):
                found.append(backend)
        except Exception:
            logger.debug("Model Chain: the %s regional backend could not be probed",
                         backend.name, exc_info=True)
    return tuple(found)


def supports_klein_regional_conditioning(model=None) -> bool:
    """Whether regional conditioning can run at all on the loaded engine."""
    return select_backend(model) is not None


def compatibility_error(arch=None, model=None) -> str:
    """Why this generation cannot run regionally, with what was detected in it.

    §34 asks for the architecture and the loaded engine by name, and it asks for
    them because the interesting failure is the one where they disagree: a
    repacked build whose header predicted Klein and whose engine implements
    something else is a real case, and a message that said only "regional
    conditioning is not available" would send somebody looking at their layout.
    """
    arch = arch if arch is not None else loaded_architecture()
    if model is None:
        try:
            from modules import shared

            model = shared.sd_model
        except Exception:
            model = None

    engine = type(model).__name__ if model is not None else "no model loaded"
    config = type(getattr(model, "model_config", None)).__name__ if model is not None \
        else "unknown"
    return (f"Spatial Layout cannot run on this checkpoint: the loaded engine exposes "
            f"no regional-conditioning path. Detected architecture "
            f"{getattr(arch, 'label', 'unknown')!r}, engine {engine!r}, model config "
            f"{config!r}. Turn Spatial Layout off to generate with the prompt as "
            f"typed.")


# --------------------------------------------------------------------------- #
# Compiling the layout against the tensor about to be sampled
# --------------------------------------------------------------------------- #


class CompiledRegions:
    """The layout, mapped onto one concrete conditioning grid, ready to install.

    Built per sampling pass rather than per generation, because the grid it is
    built against is a property of the tensor: hires fix runs a second pass at a
    different size, and a compiled set from the first one would place every box
    against the wrong geometry.
    """

    __slots__ = ("pairs", "grid", "notes", "backend", "request", "attached",
                 "diagnosis", "region_percent")

    def __init__(self, pairs, grid, notes=(), backend=None, request=None):
        self.pairs = tuple(pairs)
        self.grid = tuple(grid)
        self.notes = tuple(notes)
        self.backend = backend
        self.request = request
        self.attached = 0
        """How many regions actually reached the model. Set after installation.

        The number to trust, and the one recorded in infotext: it is an
        observation rather than an intention, and "three boxes drawn" and "three
        boxes conditioning the image" are not the same claim.
        """

        self.diagnosis = ""

        self.region_percent = 100
        """How much of the sample the regions apply for, as a percentage.

        Carried here rather than read from a preference inside the backend so
        that a test can set it and a spike can vary it, and so the backend keeps
        knowing nothing about where settings live.
        """

    def __bool__(self) -> bool:
        return bool(self.pairs)

    def __len__(self) -> int:
        return len(self.pairs)


def latent_grid(tensor) -> tuple[int, int] | None:
    """``(width, height)`` of the conditioning grid, from the live tensor.

    The last two dimensions of whatever is about to be sampled, which for every
    latent layout this host produces are height then width. Read from the tensor
    and not computed from ``p.width`` over a VAE ratio, because §15 and §29 both
    insist on the same thing: the runtime shape is authoritative. A build that
    changed its patchify factor, a PiD model that reshapes its latent, a hires
    second pass -- all of them are handled by measuring rather than predicting,
    and none of them is handled by a constant.
    """
    shape = getattr(tensor, "shape", None)
    if shape is None:
        return None
    try:
        dims = tuple(int(value) for value in shape)
    except (TypeError, ValueError):
        return None
    if len(dims) < 2:
        return None
    height, width = dims[-2], dims[-1]
    if width <= 0 or height <= 0:
        return None
    return width, height


def compile_regions(request, tensor=None, grid=None, backend=None,
                    region_percent: int = 100) -> CompiledRegions:
    """Every region of ``request`` on the grid ``tensor`` is about to be sampled at.

    ``grid`` may be given directly instead, which is what a test does and what a
    caller that has already measured does. One of the two is required; nothing
    here guesses a shape.

    Regions that cannot cover a single cell are dropped with a note rather than
    widened. §29's last line asks that at least one cell survive for a valid
    region, and :func:`prompt_master.spatial.to_grid` guarantees that for every
    box with area -- so a note here means the grid itself was smaller than the
    box was thin, which is worth saying out loud.
    """
    if grid is None:
        grid = latent_grid(tensor)
    if grid is None:
        raise ValueError("Klein regional conditioning needs the shape of the tensor "
                         "being sampled; none was supplied.")

    width, height = int(grid[0]), int(grid[1])
    pairs, notes = spatial.compile_grid(getattr(request, "regions", ()), width, height)
    compiled = CompiledRegions(pairs=pairs, grid=(width, height), notes=notes,
                               backend=backend, request=request)
    compiled.region_percent = max(1, min(100, int(region_percent or 100)))
    return compiled


# --------------------------------------------------------------------------- #
# Installing regional conditioning for one pass
# --------------------------------------------------------------------------- #


def describe_conditioning(conditioning, target=None) -> str:
    """What the host actually handed this hook, in enough detail to act on.

    Printed when no region could be attached, and printed *once* rather than per
    region. It exists because the alternative -- guessing again at a structure
    that lives in a host this extension does not ship and cannot pin -- has now
    been wrong twice, and each guess cost somebody a run of images that looked
    like the feature working.

    Deliberately shape and type names only. No tensor data, no prompt text: this
    goes to a log a user may paste into an issue, and what is needed from it is
    the structure.
    """
    def name(value):
        return type(value).__name__

    lines = [f"conditioning={name(conditioning)}"]

    for attribute in ("batch", "schedules", "shape"):
        found = getattr(conditioning, attribute, None)
        if found is not None:
            lines.append(f"  .{attribute}={name(found)}"
                         + (f" len={len(found)}" if hasattr(found, "__len__") else ""))

    if target is None:
        target = _conditioning_list(conditioning)
    if target is None:
        lines.append("  no list of entries could be located")
        return "; ".join(lines)

    lines.append(f"entries={name(target)} len={len(target)}")
    if target:
        first = target[0]
        lines.append(f"  [0]={name(first)}"
                     + (f" len={len(first)}" if isinstance(first, (list, tuple)) else ""))
        if isinstance(first, (list, tuple)) and first:
            inner = first[0]
            lines.append(f"  [0][0]={name(inner)}")
            fields = [key for key in dir(inner) if not key.startswith("_")][:12]
            if fields:
                lines.append(f"  [0][0] fields: {', '.join(fields)}")
        elif isinstance(first, dict):
            lines.append(f"  [0] keys: {', '.join(sorted(first)[:12])}")
        else:
            fields = [key for key in dir(first) if not key.startswith("_")][:12]
            if fields:
                lines.append(f"  [0] fields: {', '.join(fields)}")
    return "; ".join(lines)


class _InstalledRegions:
    """What one pass added to the conditioning, and how to take it away again.

    Deliberately a small object holding the *list it mutated* and the *entries it
    appended*, rather than a copy of the list as it was. Forge hands the same
    conditioning object to the sampler that it hands to this hook, and replacing
    it wholesale would drop anything another always-on script added in between;
    removing exactly what was added leaves everybody else's contributions where
    they were.
    """

    __slots__ = ("target", "added", "count", "diagnosis")

    def __init__(self, target=None, added=(), diagnosis=""):
        self.target = target
        self.added = list(added)
        self.count = len(self.added)
        self.diagnosis = str(diagnosis or "")
        """Why nothing was attached, when nothing was.

        Empty on the ordinary path. A sentence when the host's conditioning is a
        shape an area cannot be written into -- which is a fact about the host
        and not about the layout, and is exactly what a user staring at an image
        that ignored their boxes needs to be told.
        """

    def remove(self) -> None:
        if self.target is None:
            return
        for entry in self.added:
            try:
                self.target.remove(entry)
            except (ValueError, AttributeError):
                # Already gone, or never a list. Both are fine: the point of
                # this method is that nothing of ours is left behind, and
                # something else having removed it satisfies that.
                logger.debug("Model Chain: a Klein regional conditioning entry was "
                             "already gone at cleanup", exc_info=True)
        self.added = []


def _conditioning_list(conditioning):
    """The mutable list of positive conditioning entries, out of whatever wraps it.

    The host hands this hook its ``c`` for the pass about to run, and what ``c``
    *is* has three shapes across Forge's history: the list itself, a
    ``MulticondLearnedConditioning`` with a ``batch``, and a scheduled structure
    whose first element holds the list. All three are unwrapped here rather than
    at each call site, and the list that comes back is the one the sampler will
    read -- not a copy of it, which would be installed into nothing.

    ``None`` when no shape matches, which the caller reports as an unavailable
    backend rather than working around. Appending to the wrong object would
    produce a generation that looked like it worked.
    """
    found = conditioning
    for _ in range(4):
        if isinstance(found, list):
            if found and isinstance(found[0], (list, dict)):
                return found
            if len(found) == 1:
                found = found[0]
                continue
            return found or None
        batch = getattr(found, "batch", None)
        if batch is not None:
            found = batch
            continue
        schedules = getattr(found, "schedules", None)
        if schedules is not None:
            found = schedules
            continue
        return None
    return None


def _install_conditioning_regions(conditioning, compiled, model, p,
                                  use_mask: bool) -> _InstalledRegions:
    """Append one conditioning entry per region, each scoped to its rectangle.

    The shared half of the two host-metadata backends. Both encode the same
    prompts against the same grid; they differ only in whether the geometry
    travels as an ``area`` or as a ``mask``, so the encode -- which is the
    expensive and the fragile part -- is written once.
    """
    target = _conditioning_list(conditioning)
    if target is None:
        raise RegionalConditioningUnavailable(
            "the positive conditioning for this pass could not be located, so region "
            "prompts could not be attached to it")

    width, height = compiled.grid
    added = []
    for region, cell in compiled.pairs:
        text = region.qualified_prompt()
        if not text:
            continue
        encoded = _encode(model, text, p)
        if encoded is None:
            continue
        entry = _entry_like(target, encoded)
        if entry is None:
            continue
        gx0, gy0, gx1, gy1 = cell
        options = _entry_options(entry)
        if options is None:
            continue
        if use_mask:
            mask = _rect_mask(width, height, cell, encoded)
            if mask is None:
                continue
            options["mask"] = mask
            options["mask_strength"] = float(region.strength)
        else:
            # (height, width, y, x) -- the host's own ordering, which is the
            # order ComfyUI's area tuple has always been in and the order
            # get_area_and_mult unpacks.
            options["area"] = (gy1 - gy0, gx1 - gx0, gy0, gx0)
            options["strength"] = float(region.strength)
        options["klein_spatial_region"] = region.identifier
        target.append(entry)
        added.append(entry)

    diagnosis = ""
    if compiled.pairs and not added:
        # The silent failure this whole feature is arranged against, caught at
        # the one place it can be caught: regions were compiled, encoding
        # succeeded, and not one of them could be given a geometry. That is a
        # property of the host's conditioning shape, so the shape is what gets
        # reported.
        diagnosis = describe_conditioning(conditioning, target)

    return _InstalledRegions(target=target, added=added, diagnosis=diagnosis)


PATCHER_HOOKS = (
    "set_model_unet_function_wrapper",
    "set_model_attn1_patch",
    "set_model_attn2_patch",
    "set_model_attn1_replace",
    "set_model_attn2_replace",
    "set_model_attn1_output_patch",
    "set_model_attn2_output_patch",
    "set_model_sampler_cfg_function",
    "set_model_sampler_pre_cfg_function",
    "set_model_sampler_post_cfg_function",
    "set_model_patch",
    "set_model_patch_replace",
    "add_patches",
)
"""ModelPatcher hooks worth knowing about, for a cheaper implementation.

None of these are used yet. They are *reported*, once per job, because the
composable backend that works today costs one model evaluation per region per
step and the mechanisms that would not are all reached through one of these.

The cheap ones, in order of how much they would save:

``set_model_attn2_patch`` and friends
    Regional attention: one forward pass, with each region's tokens masked to
    its rectangle inside cross-attention. Roughly free, and the reason to want
    it. Needs the attention call's shape for this architecture, which is the
    thing to establish before writing any of it.

``set_model_unet_function_wrapper``
    Wraps the model call. Enough to implement a crop-and-composite pass, which
    costs the *area* of each region rather than a whole frame -- a fifth of a
    frame for a fifth-of-a-frame box, instead of another whole evaluation.

Printed rather than guessed at, because guessing at this host's internals has
been wrong twice and each time the cost was somebody's afternoon.
"""


def describe_patcher(p=None) -> str:
    """Which model-patching hooks this host offers, in one line.

    Written to the log once per Klein spatial job that attaches regions, so the
    next implementation is chosen from what is there rather than from what is
    usually there.
    """
    try:
        from modules import shared

        model = getattr(p, "sd_model", None) if p is not None else None
        model = model or shared.sd_model
        objects = getattr(model, "forge_objects", None)
        unet = getattr(objects, "unet", None)
    except Exception:
        return "no loaded model to inspect"

    if unet is None:
        return "the loaded model exposes no forge_objects.unet"

    offered = [name for name in PATCHER_HOOKS if callable(getattr(unet, name, None))]
    options = getattr(unet, "model_options", None)
    keys = sorted(options)[:16] if isinstance(options, dict) else []
    return (f"patcher={type(unet).__name__} hooks={', '.join(offered) or 'none'}"
            + (f" model_options={', '.join(keys)}" if keys else ""))


class _InstalledComposable:
    """What the composable backend added, and how to take all of it away.

    Two things to unwind rather than one: the region prompts appended to the
    host's conditioning, and the masked blend installed on its denoiser. Both are
    removed on every exit, and the denoiser is restored to whatever it was --
    including to *not having* an instance attribute, if that is what it had.
    """

    __slots__ = ("batches", "added", "count", "restore", "diagnosis")

    def __init__(self, batches=(), added=0, restore=None, diagnosis=""):
        self.batches = list(batches)
        self.added = int(added)
        self.count = int(added)
        self.restore = restore
        self.diagnosis = str(diagnosis or "")

    def retire(self) -> None:
        """Take the regions out of the conditioning, mid-sample.

        Separate from :meth:`remove` because it happens at a different time for a
        different reason: this is the step cutoff arriving, and the blend stays
        installed because it is what will notice. :meth:`remove` is the end of
        the job and takes everything.
        """
        for batch, entries in self.batches:
            for entry in entries:
                try:
                    batch.remove(entry)
                except (ValueError, AttributeError):
                    logger.debug("Model Chain: a Klein region conditioning was "
                                 "already gone", exc_info=True)
        self.batches = []

    def remove(self) -> None:
        self.retire()

        if self.restore is not None:
            try:
                self.restore()
            except Exception:
                logger.warning("Model Chain: the Klein masked blend did not unwind "
                               "cleanly", exc_info=True)
            self.restore = None


def _composable_batches(conditioning):
    """Every per-image list of composable conditionings, or ``None``.

    ``MulticondLearnedConditioning.batch`` is a list of lists: one inner list per
    image in the batch, each holding the composable prompts for that image. Both
    levels matter -- a region has to be appended to *every* image's list, or the
    second image in a batch of two would be generated without it.
    """
    batch = getattr(conditioning, "batch", None)
    if not isinstance(batch, list) or not batch:
        return None

    for entry in batch:
        if not isinstance(entry, list) or not entry:
            return None
        if not hasattr(entry[0], "schedules") or not hasattr(entry[0], "weight"):
            return None
    return batch


def _composable_like(template, encoded, weight: float):
    """One composable conditioning shaped like ``template``, holding ``encoded``.

    Built from the classes of the objects already there rather than by importing
    them by name. The host's own entry is the specification, which is the only
    approach in this file that has not had to be corrected: a name can move
    between releases and the object in front of you cannot be wrong about what it
    is.
    """
    schedules = list(getattr(template, "schedules", ()) or ())
    if not schedules:
        return None
    step = schedules[-1]

    cond = _conditioning_tensor(encoded, getattr(step, "cond", None))
    if cond is None:
        return None

    try:
        scheduled = type(step)(getattr(step, "end_at_step", 0), cond)
    except Exception:
        try:
            scheduled = type(step)(end_at_step=getattr(step, "end_at_step", 0),
                                   cond=cond)
        except Exception:
            logger.debug("Model Chain: could not build a scheduled conditioning",
                         exc_info=True)
            return None

    try:
        return type(template)([scheduled], float(weight))
    except Exception:
        try:
            return type(template)(schedules=[scheduled], weight=float(weight))
        except Exception:
            logger.debug("Model Chain: could not build a composable conditioning",
                         exc_info=True)
            return None


def _compatible_cond(candidate, template) -> bool:
    """Whether ``candidate`` can sit in the same batch as ``template``.

    The host stacks every composable conditioning into one tensor before calling
    the model, padding the token axis but not reconciling anything else. A region
    whose conditioning is a different type, or has a different feature width,
    would raise inside the sampler with nothing in the traceback naming this
    feature -- so it is refused here, where the message can say what happened.
    """
    if type(candidate) is not type(template):
        return False

    left, right = getattr(candidate, "shape", None), getattr(template, "shape", None)
    if left is None or right is None:
        return True
    try:
        return tuple(left)[1:] == tuple(right)[1:]
    except Exception:
        return True


def _install_composable_regions(conditioning, compiled, model, p, cutoff=0):
    """Append each region as a composable prompt, and mask how it is blended in.

    The order the regions are appended in is the order their masks are listed in,
    and the masked blend relies on that correspondence and on nothing else: it
    identifies the regions as *the last N entries* of each image's list rather
    than by absolute index, so anything else in the conditioning -- another
    extension's composable prompt, a batch of several images -- is left alone and
    keeps working.
    """
    batches = _composable_batches(conditioning)
    if batches is None:
        return _InstalledComposable(
            diagnosis=describe_conditioning(conditioning))

    template = batches[0][0]
    width, height = compiled.grid
    prepared = []

    for region, cell in compiled.pairs:
        text = region.qualified_prompt()
        if not text:
            continue
        encoded = _encode(model, text, p)
        if encoded is None:
            continue

        composable = _composable_like(template, encoded, region.strength)
        if composable is None:
            continue

        reference = getattr(template.schedules[-1], "cond", None)
        if not _compatible_cond(getattr(composable.schedules[-1], "cond", None),
                                reference):
            logger.warning("Model Chain: the conditioning for region %r is not the "
                           "shape this host stacks; it was left out",
                           region.identifier)
            continue

        mask = _rect_mask(width, height, cell)
        if mask is None:
            continue
        prepared.append((composable, mask))

    if not prepared:
        return _InstalledComposable(
            diagnosis=describe_conditioning(conditioning))

    added = []
    for batch in batches:
        entries = []
        for composable, _mask in prepared:
            # A separate object per image: the host may hold on to these, and two
            # images sharing one would make removing it from the first remove it
            # from the second.
            copy = _composable_like(template,
                                    getattr(composable.schedules[-1], "cond", None),
                                    composable.weight) or composable
            batch.append(copy)
            entries.append(copy)
        added.append((batch, entries))

    installed = _InstalledComposable(batches=added, added=len(prepared))
    restore = _install_masked_blend(
        p, [mask for _composable, mask in prepared], cutoff=cutoff,
        prune=installed.retire)
    if restore is None:
        # Without the masked blend a region would be composed across the whole
        # frame, which is a different picture rather than a weaker version of the
        # one asked for. Everything appended above comes straight back off.
        installed = _InstalledComposable(
            batches=added, added=0,
            diagnosis="the host's denoiser does not expose a blend this build can "
                      "mask (no sampler.model_wrap_cfg.combine_denoised)")
        installed.remove()
        return installed

    installed.restore = restore
    return installed


def region_cutoff(steps, percent) -> int:
    """The last step regions are applied on, from a percentage of ``steps``.

    Composition is decided early. The first steps of a diffusion sample settle
    where the large shapes are and the last ones settle texture, so a region that
    stops contributing part-way through has usually already done its work -- and
    every step it does not contribute on is one fewer model evaluation per
    region.

    At least one step, always: a region that never applied is not a cheaper
    version of the feature, it is the feature switched off.
    """
    try:
        steps = max(int(steps), 1)
        percent = max(1, min(100, int(percent)))
    except (TypeError, ValueError):
        return 0
    return max(1, round(steps * percent / 100.0))


def _install_masked_blend(p, masks, cutoff=0, prune=None):
    """Make the host blend each region only inside its rectangle.

    Returns a callable that puts the original blend back, or ``None`` when there
    is no blend to replace.

    The replacement calls the *original* once for the global prompt alone and
    once per region, and masks the difference. Deriving it that way rather than
    reimplementing the arithmetic is the whole point: CFG scale, a skipped
    unconditional pass at CFG 1, an edit model's own formula -- all of that stays
    exactly as the host computes it, and none of it has to be known here.

    ``cutoff`` is the last step regions apply on, and ``prune`` is what removes
    them from the host's conditioning once it passes. Pruning rather than merely
    ignoring is the part that saves anything: the host evaluates every composable
    prompt in the batch whether or not this function uses the result, so a region
    stops costing a model evaluation only when it stops being in the batch.

    The prune happens one step late by construction -- this function runs after
    the evaluation for the current step -- and that is fine. One extra evaluation
    is not worth reaching further into the denoiser to avoid.
    """
    sampler = getattr(p, "sampler", None) if p is not None else None
    denoiser = getattr(sampler, "model_wrap_cfg", None) or getattr(
        sampler, "model_wrap", None)
    original = getattr(denoiser, "combine_denoised", None)
    if denoiser is None or not callable(original):
        return None

    count = len(masks)
    owned = "combine_denoised" in getattr(denoiser, "__dict__", {})
    spent = []

    def combine(x_out, conds_list, uncond, cond_scale):
        try:
            base_list = [conds[:-count] for conds in conds_list]
            if any(not conds for conds in base_list):
                return original(x_out, conds_list, uncond, cond_scale)

            step = getattr(denoiser, "step", None)
            if cutoff and step is not None and step >= cutoff and not spent:
                # Past the cutoff: blend without the regions, and take them out
                # of the batch so the next step does not evaluate them.
                spent.append(True)
                if prune is not None:
                    try:
                        prune()
                    except Exception:
                        logger.debug("Model Chain: could not retire the Klein "
                                     "spatial regions", exc_info=True)
                return original(x_out, base_list, uncond, cond_scale)

            blended = original(x_out, base_list, uncond, cond_scale)
            for index in range(count):
                one = [list(conds[:-count]) + [conds[len(conds) - count + index]]
                       for conds in conds_list]
                with_region = original(x_out, one, uncond, cond_scale)
                mask = masks[index].to(device=blended.device, dtype=blended.dtype)
                blended = blended + (with_region - blended) * mask
            return blended
        except Exception:
            # A blend that raised would take the generation with it. The
            # unmasked answer is the host's own and is a picture; it is also
            # wrong about placement, so it is said out loud rather than returned
            # quietly.
            logger.error("Model Chain: the Klein masked blend failed; this image "
                         "was blended without region masks and its placement is "
                         "not what the layout asked for", exc_info=True)
            return original(x_out, conds_list, uncond, cond_scale)

    denoiser.combine_denoised = combine

    def restore():
        if owned:
            denoiser.combine_denoised = original
        else:
            try:
                del denoiser.combine_denoised
            except AttributeError:
                denoiser.combine_denoised = original

    return restore


def _encode(model, text: str, p=None):
    """One region prompt through the loaded model's own text encoder.

    The host's path, not a reimplementation of it: whatever Klein's Qwen3 encoder
    wants -- padding, attention mask, pooled output -- the model already knows,
    and a second encode written here would be a second thing to keep in step with
    the loader.

    What it is given matters as much as which function is called. A diffusion
    engine's ``get_learned_conditioning`` does not take a list of strings; it
    takes the host's own prompt container, and it reads attributes off it before
    it reads any text. Flux.2's asks ``prompt.is_negative_prompt`` on its first
    line, so a bare ``[text]`` raises ``AttributeError`` there -- which is not a
    Klein quirk but the ordinary calling convention, and passing a list was
    simply wrong.

    :func:`_prompt_container` builds the real thing, carrying this generation's
    width, height and distilled CFG across, because an architecture that sizes
    its conditioning from those would otherwise size a region differently from
    the global prompt beside it.

    Returns ``None`` on failure, which drops that one region with a warning
    rather than failing the generation.
    """
    try:
        if model is None:
            from modules import shared

            model = shared.sd_model
        encode = getattr(model, "get_learned_conditioning", None)
        if not callable(encode):
            return None
        return encode(_prompt_container([text], p))
    except Exception:
        logger.warning("Model Chain: a Klein spatial region prompt could not be "
                       "encoded and was left out", exc_info=True)
        return None


class _Conditioning(list):
    """A stand-in for the host's prompt container, when its own cannot be found.

    Same shape and the same three attributes an engine reads off one. Used only
    where ``modules.prompt_parser`` is missing or has been rearranged; the host's
    own class is preferred wherever it exists, because it is the one the engines
    were written against and it is free to grow a fourth attribute without
    telling this file.
    """

    def __init__(self, prompts, is_negative_prompt=False, width=None, height=None,
                 distilled_cfg_scale=None):
        super().__init__()
        self.extend(prompts)
        self.is_negative_prompt = is_negative_prompt
        self.width = width
        self.height = height
        self.distilled_cfg_scale = distilled_cfg_scale


def _prompt_container(prompts, p=None):
    """``prompts`` in whatever container this host's engines expect.

    ``modules.prompt_parser.SdConditioning`` when it is there, which it is on
    every build this extension supports, and :class:`_Conditioning` when it is
    not. Either way the geometry comes off the generation rather than being left
    unset: a region encoded at no particular size, beside a global prompt encoded
    at 768x1024, is two conditionings that need not agree about anything.

    Never a negative prompt. V1 conditions regions positively only (§32), and the
    global negative continues through the host's own path untouched.
    """
    width = int(getattr(p, "width", 0) or 0) or None
    height = int(getattr(p, "height", 0) or 0) or None
    distilled = getattr(p, "distilled_cfg_scale", None)

    try:
        from modules.prompt_parser import SdConditioning

        return SdConditioning(prompts, is_negative_prompt=False, width=width,
                              height=height, distilled_cfg_scale=distilled)
    except Exception:
        logger.debug("Model Chain: the host's prompt container was not available; "
                     "using the local stand-in", exc_info=True)
        return _Conditioning(prompts, is_negative_prompt=False, width=width,
                             height=height, distilled_cfg_scale=distilled)


def _entry_like(target, encoded):
    """A conditioning entry in whatever shape the host's existing ones use.

    Two shapes are supported and both are ComfyUI's: ``[tensor, {options}]``, and
    a dict with the conditioning under a key. Those are the shapes that carry an
    ``area``, which is the whole point of building one.

    ``None`` when the host's entries are neither -- and that is a real case
    rather than a defensive branch. A1111-derived hosts hand this hook a
    ``MulticondLearnedConditioning`` whose batch entries are lists of
    *composable* conditionings with a weight and no geometry at all; the
    ComfyUI-style cond dicts those become do not exist until later, inside the
    sampler. A region cannot be given an area there because there is nowhere to
    put one, and the honest answer is to say so rather than to append something
    that would condition the whole frame.

    :func:`describe_conditioning` is what turns that ``None`` into a sentence
    naming what was actually found.
    """
    if not target:
        return None
    template = target[0]

    if isinstance(template, list) and len(template) >= 2 and isinstance(template[1], dict):
        options = dict(template[1])
        options.pop("area", None)
        options.pop("mask", None)
        options.pop("mask_strength", None)
        options.pop("strength", None)
        return [_conditioning_tensor(encoded, template[0]), options]

    if isinstance(template, dict):
        entry = dict(template)
        for key in ("area", "mask", "mask_strength", "strength"):
            entry.pop(key, None)
        for key in ("model_conds", "cond", "crossattn", "c_crossattn"):
            if key in entry:
                entry[key] = _conditioning_tensor(encoded, entry[key])
                return entry
        return None

    return None


def _conditioning_tensor(encoded, like):
    """The tensor half of an encode, matched to how the host stores its own.

    ``get_learned_conditioning`` returns a batch, a list of one, or a scheduled
    structure depending on the host and the prompt; ``like`` is what the existing
    entries actually hold. Unwrapped towards that shape rather than assumed,
    because guessing wrong here is a shape error deep inside the sampler with
    nothing in the traceback naming this feature.
    """
    found = encoded
    for _ in range(3):
        if type(found) is type(like):
            return found
        if isinstance(found, (list, tuple)) and len(found) == 1:
            found = found[0]
            continue
        schedules = getattr(found, "schedules", None)
        if schedules:
            found = schedules
            continue
        cond = getattr(found, "cond", None)
        if cond is not None:
            found = cond
            continue
        break
    return found


def _entry_options(entry):
    """The mutable options dict of a conditioning entry, or ``None``."""
    if isinstance(entry, list) and len(entry) >= 2 and isinstance(entry[1], dict):
        return entry[1]
    if isinstance(entry, dict):
        return entry
    return None


def _rect_mask(width: int, height: int, cell, like=None):
    """A ``height`` x ``width`` mask that is 1 inside ``cell`` and 0 outside.

    Built with torch when torch is importable, which it is inside any host that
    could sample anything. ``None`` when it is not, which makes the mask backend
    decline rather than the generation fail.
    """
    try:
        import torch
    except Exception:
        return None

    gx0, gy0, gx1, gy1 = cell
    mask = torch.zeros((height, width), dtype=torch.float32)
    mask[gy0:gy1, gx0:gx1] = 1.0

    device = getattr(like, "device", None)
    if device is not None:
        try:
            mask = mask.to(device)
        except Exception:
            pass
    return mask


# --------------------------------------------------------------------------- #
# References, and putting everything back
# --------------------------------------------------------------------------- #


def clear_references(p=None) -> None:
    """Drop reference state, and make ImageStitch re-encode its own next time.

    The same pair of actions ``ScriptModelChain._clear_references`` performs, for
    the same reason: a set left resident outlives the job that encoded it, and
    ImageStitch's memo would then skip an encode it needed to do. §18 asks for
    both halves and asks for them at both ends of every job.
    """
    try:
        from modules import shared

        model = shared.sd_model
        if model is not None and getattr(model, "ref_latents", None):
            model.ref_latents = []
    except Exception:
        logger.debug("Model Chain: could not clear the model's reference latents",
                     exc_info=True)

    try:
        mc_references.invalidate_stitch_cache(p)
    except Exception:
        logger.debug("Model Chain: could not reset the ImageStitch cache", exc_info=True)

    try:
        from backend.args import dynamic_args

        dynamic_args.is_referencing = False
    except Exception:
        logger.debug("Model Chain: could not lower the referencing flag", exc_info=True)


def reference_override(request, arch=None) -> dict:
    """The settings fragment deciding whether ImageStitch's references reach Klein.

    This is where §6 and §22 are actually enforced, and it is worth being exact
    about why it is a *toggle* rather than an encode.

    ImageStitch is an always-on script and it runs for Stage 1 on its own. When
    it is enabled with images in its gallery it raises
    ``dynamic_args.is_referencing`` and pushes each one through
    ``encode_first_stage``, and Klein's engine diverts them into ``ref_latents``.
    That is not something this extension arranges; it is what the user switching
    ImageStitch on already means, and it is exactly the ordered reference set
    §17 describes.

    So Reference + Regions does not need this module to encode anything --
    the images are already on their way -- and a second encode here would
    register every one of them twice, because ``ref_latents`` is appended to
    rather than replaced. Which of the two encodes ran first would then depend on
    the order Forge happens to run two always-on scripts in, and an ordering that
    happens to be right is not an ordering.

    What the modes actually differ in is whether those references should reach
    the model at all, and Forge already has one switch for that: Klein's
    ``klein_no_reference``. Returned as a ``p.override_settings`` fragment so the
    host applies it for this generation and restores the user's global value
    afterwards -- the same mechanism, and the same reasoning, as
    :func:`mc_arch.edit_override`.

        Regional Generate      references off; ImageStitch is ignored (§6)
        Reference + Regions    references on; ImageStitch's own set is used

    The residual cost, stated rather than hidden: in Regional Generate with
    images left in the gallery, ImageStitch still performs its VAE encode and the
    engine simply does not keep the result. Avoiding that would mean writing to
    another script's arguments, which this extension does not do -- §3 is
    explicit that ImageStitch's gallery is read and never written back.
    """
    arch = arch if arch is not None else loaded_architecture()
    if not arch.supports_edit:
        return {}

    mode = mc_arch.EDIT_ENABLE if getattr(request, "uses_source", False) \
        else mc_arch.EDIT_DISABLE
    return mc_arch.edit_override(arch, mode)


@contextlib.contextmanager
def reference_scope(p=None):
    """Clear stale reference state at both ends of a Klein spatial job.

    Entered in ``before_process``, which is the one moment that is both after the
    previous job and before any script has encoded anything for this one. That
    timing is the whole point: clearing later would race ImageStitch's own encode
    and could wipe the very references this generation is meant to use, and
    clearing only at the end would leave a previous job's set resident for the
    whole of this one -- which is exactly the state that makes an empty gallery
    look source-capable.

    Dropping ImageStitch's memo is the other half. It returns early when its
    inputs are unchanged, on the assumption that what it encoded last time is
    still on the model; having just made that untrue, this says so.
    """
    clear_references(p)
    try:
        yield
    finally:
        clear_references(p)


@contextlib.contextmanager
def regional_conditioning(request, conditioning, tensor=None, grid=None,
                          backend=None, model=None, p=None,
                          region_percent: int = 100):
    """Regional conditioning for one sampling pass, installed and then removed.

    The whole of §41's unwind requirement for the conditioning half, in one
    block: whatever is installed here is removed in the ``finally``, on success,
    on an exception and on an interrupt alike, so the next ordinary Klein
    generation is an ordinary Klein generation.

    ``conditioning`` is the positive conditioning the host handed the hook for
    the pass about to run, and ``tensor`` is the latent it is about to sample.
    Both are passed in rather than fetched, because this is the one moment they
    exist and the caller is the only thing holding them.

    Yields the compiled regions, so the caller can record how many actually
    reached the model rather than how many were drawn.
    """
    candidates = [backend] if backend is not None else list(usable_backends(model))
    if not candidates:
        raise RegionalConditioningUnavailable(compatibility_error(model=model))

    compiled = compile_regions(request, tensor=tensor, grid=grid,
                               backend=candidates[0], region_percent=region_percent)
    for note in compiled.notes:
        logger.warning("Model Chain: %s", note)

    if not compiled:
        # §33: Spatial Layout on with nothing drawn is valid, and the answer is
        # an ordinary generation rather than an error. Nothing is installed, so
        # nothing needs removing.
        yield compiled
        return

    # Tried in order until one actually attaches something, and this fall-through
    # is the correction to the design error that cost several runs of images. A
    # probe can only ask whether a *mechanism* exists in the host; whether a
    # region can be written into the conditioning object this pass was handed is
    # a question only the attempt can answer. The area backend probes present on
    # this host and fails here, every time, because the cond dicts it writes into
    # are built later, inside the sampler.
    diagnosis = ""
    with contextlib.ExitStack() as stack:
        for candidate in candidates:
            compiled.backend = candidate
            try:
                installed = stack.enter_context(
                    candidate.apply(conditioning, compiled, model, p))
            except RegionalConditioningUnavailable:
                raise
            except Exception:
                logger.warning("Model Chain: the %s regional backend failed while "
                               "installing; trying the next one", candidate.name,
                               exc_info=True)
                continue

            attached = getattr(installed, "count", 0)
            if attached:
                compiled.attached = attached
                compiled.diagnosis = ""
                logger.info("Model Chain: Klein Spatial attached %d of %d "
                            "region(s) to a %dx%d conditioning grid via %s",
                            attached, len(compiled), compiled.grid[0],
                            compiled.grid[1], candidate.name)
                if candidate.name == ComposableRegionBackend.name:
                    # The working-but-expensive path. Say what this host offers
                    # that a cheaper one could use, so the next implementation is
                    # chosen from what is there.
                    logger.info("Model Chain: Klein Spatial costs one model "
                                "evaluation per region per step on this backend; "
                                "for a cheaper one this host offers %s",
                                describe_patcher(p))
                yield compiled
                return

            diagnosis = getattr(installed, "diagnosis", "") or diagnosis
            logger.debug("Model Chain: the %s regional backend attached nothing; "
                         "trying the next one", candidate.name)

        compiled.attached = 0
        compiled.diagnosis = diagnosis
        # ERROR and not INFO. "attached 0 of 3" read as a status line for several
        # runs of images that looked like the feature working, which is the exact
        # failure §28 is about: a plausible picture with nothing anywhere saying
        # the regions never arrived.
        logger.error(
            "Model Chain: Klein Spatial attached NONE of %d region(s) — no "
            "backend could write a region geometry into this host's conditioning, "
            "so your boxes did not reach the model. Tried %s. Observed %s",
            len(compiled), ", ".join(one.name for one in candidates), diagnosis)
        yield compiled


# --------------------------------------------------------------------------- #
# Smart Compose: reconciling the global prompt with the layout
# --------------------------------------------------------------------------- #
#
# The second half of what Spatial Layout means, and the half regional
# conditioning does not supply on its own.
#
# Klein reads the global prompt as written. So a user who types "a living room
# with a tall brass floor lamp on the left" and then draws the lamp on the right
# has said two things: the prompt asks for a lamp, and the regional condition
# asks for a lamp somewhere else. What comes back is usually two lamps -- and
# that is not a failure of the conditioning, it is the prompt competing with it.
#
# §31 says the same thing as a rule: do not repeat region prompts in the global
# prompt, because that encourages duplicated objects outside the target boxes.
# Direct mode leaves that to the user, which is what a control half of them will
# want. Smart mode has a language model take it out.


def compose_scene(prompt: str, layout, ratio: str = "", seed: int = 0,
                  reserve: int = 0, task_id: str = ""):
    """Reconcile ``prompt`` with ``layout``. Returns a :class:`mc_spatial.Composed`.

    The same Spatial Composer pass Krea 2 uses, and reused rather than rewritten
    for a reason worth stating, because §31 warns against running "the Krea
    composer" for Klein.

    What that warning is about is the Krea *compositor* -- the deterministic step
    that turns a scene and a layout into a structured JSON prompt with
    coordinates in it. That step is emphatically not run here and would be
    meaningless if it were: Klein reads prose, and the boxes reach it as
    conditioning geometry rather than as text.

    The Composer is a different thing. Its instruction never mentions Krea, it is
    forbidden from emitting structure, and its whole job is "rewrite this scene
    so it stops arguing with these boxes" -- which is a copy-editing task over
    two paragraphs and is exactly as true for Klein as for Krea. Sharing it means
    one instruction to maintain, one recorded instruction version, and a Smart
    against Direct comparison that means the same thing on both backends.

    The one difference is where the result goes. For Krea the scene becomes a
    field of a structured document and ``background`` becomes another; for Klein
    the composed text *is* the prompt the model reads, so the two are folded into
    one string by :func:`composed_prompt`.

    Never raises, and never cancels a generation: every failure here falls back
    to the prompt as typed, which is Direct mode, which is a perfectly good
    generation.
    """
    import mc_spatial

    return mc_spatial.compose(source=prompt, scene=prompt, layout=layout,
                              ratio=ratio, seed=seed, reserve=reserve,
                              task_id=task_id)


def composed_prompt(composed, fallback: str) -> str:
    """One prompt string out of the Composer's two fields.

    ``scene`` carries the picture and ``background`` the setting behind it, and
    Klein has one prompt rather than two fields to put them in. Joined with a
    comma, which is how every other part of this feature joins prose, and
    skipped entirely when the background is empty or already inside the scene --
    a Composer that answered both with the same sentence should not produce it
    twice.

    ``fallback`` is returned whenever the pass did not run or came back with
    nothing, so a caller can assign the result unconditionally.
    """
    if composed is None or not getattr(composed, "ran", False):
        return fallback

    scene = str(getattr(composed, "scene", "") or "").strip()
    if not scene:
        return fallback

    background = str(getattr(composed, "background", "") or "").strip()
    if background and background.casefold() not in scene.casefold():
        return f"{scene}, {background}"
    return scene


# --------------------------------------------------------------------------- #
# What one Klein spatial generation says about itself
# --------------------------------------------------------------------------- #


def metadata(request, backend=None, layout_serialized: str = "",
             composed=None) -> dict:
    """The Klein Spatial keys for one generation's infotext.

    Its own namespace, separate from Krea's, because they describe different
    things and §37 requires that an old Krea record never be read as a Klein
    regional-conditioning job.

    The serialized layout is essential here in a way it is not for Krea (§39):
    Krea's finished structured prompt carries its boxes in the ``Prompt:`` line,
    so a paste reproduces the picture without the layout. Klein's regional
    conditioning leaves no trace in the prompt at all, so the layout is the only
    record of what was drawn.

    Records what was *used*, never source pixels. §38's last line, and the same
    policy the rest of this extension already follows about reference images.
    """
    import mc_infotext

    source = getattr(request, "source", None)
    recorded = {
        mc_infotext.KLEIN_SPATIAL_MODE: spatial.label_for(request.requested_mode),
        mc_infotext.KLEIN_SPATIAL_RESOLVED: spatial.label_for(request.resolved_mode),
        mc_infotext.KLEIN_SPATIAL_VERSION: VERSION,
        mc_infotext.KLEIN_SPATIAL_LAYOUT: str(layout_serialized or ""),
    }
    if request.compose_mode:
        recorded[mc_infotext.KLEIN_SPATIAL_COMPOSE_MODE] = request.compose_mode
    if composed is not None and getattr(composed, "ran", False):
        from prompt_master.krea import composer as composer_module

        recorded[mc_infotext.KLEIN_SPATIAL_COMPOSER_SEED] = composed.seed
        recorded[mc_infotext.KLEIN_SPATIAL_COMPOSER_VERSION] = \
            composer_module.INSTRUCTION_VERSION
    if request.uses_source and source is not None and source.usable:
        recorded[mc_infotext.KLEIN_SPATIAL_SOURCE] = "ImageStitch"
        recorded[mc_infotext.KLEIN_SPATIAL_SOURCE_COUNT] = source.image_count
    else:
        recorded[mc_infotext.KLEIN_SPATIAL_SOURCE] = "None"
    if backend is not None:
        recorded[mc_infotext.KLEIN_SPATIAL_BACKEND] = backend.name
        recorded[mc_infotext.KLEIN_SPATIAL_BACKEND_VERSION] = backend.version
    return recorded


def describe(request, backend=None) -> str:
    """The one concise line §42 asks for at the start of a Klein spatial job."""
    name = getattr(backend, "name", "none")
    return f"Klein Spatial: {request.summary()} backend={name}"
