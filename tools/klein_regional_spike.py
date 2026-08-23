"""Find the narrowest host hook that can condition a rectangle on FLUX.2 Klein.

Design intent §53, PR/STEP 0, run as a script instead of read as a plan. It
answers one question and does not build anything on top of the answer:

    can this Forge Neo build attach a prompt to an x/y/w/h region of a
    txt2img generation, and if so, through which mechanism?

Not part of the extension and never imported by it. It lives in ``tools/``
rather than ``scripts/`` for the same reason ``pin_managed_models.py`` does --
Forge imports everything in ``scripts/`` at startup, and this must run only when
somebody runs it.

Why this exists as a separate thing
-----------------------------------
:mod:`mc_spatial_klein` probes for the mechanisms it knows about and refuses to
sample when it finds none, which is §34's requirement and is the right behaviour
in front of a user. What it cannot do is tell you *why* none was found, or
whether a mechanism that probed as present actually moves an object when it is
used. Those are the two questions this answers, and both need a card, a
checkpoint and a few minutes.

§28's warning is the reason the second half exists at all: a regional
implementation that silently does nothing produces a perfectly good image, and
nothing about that image says the feature never ran. So the acceptance here is
not "no exception was raised" -- it is a measurement, on pixels, that the same
seed and the same prompt put a bright object on the left when told left and on
the right when told right.

How to run it
-------------
From the WebUI's own environment, with a FLUX.2 Klein 9B checkpoint selected in
Settings so that Forge loads it::

    python tools/klein_regional_spike.py --probe
    python tools/klein_regional_spike.py --generate --out spike/

``--probe`` is cheap, needs no card, and reports which candidate mechanisms this
build exposes. ``--generate`` is the half that decides anything: it runs six
small txt2img generations -- three seeds, left and right -- through whichever
mechanism probed as available, measures where the brightness went, and prints a
verdict.

It can also be run inside the WebUI process, which is what to do when running it
standalone cannot find the host's modules::

    from tools import klein_regional_spike
    klein_regional_spike.main(["--probe"])

What to do with the answer
--------------------------
Record it. §53 asks for "the backend mechanism and host API assumptions" to be
written down, and the place they are written down is
:data:`mc_spatial_klein.BACKENDS` -- a mechanism this reports as working belongs
in that tuple, in preference order, with its ``name`` and ``version`` set so
that images made through it say so in their infotext.

And if it reports nothing: **stop**, which is §53's own instruction. Do not
reach for a broader monkey-patch to make one of these work. Reassess the host
integration first.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import sys
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# Candidate mechanisms
# --------------------------------------------------------------------------- #

SAMPLING_MODULES = (
    "backend.sampling.sampling_function",
    "backend.modules.k_diffusion_extra",
    "ldm_patched.modules.samplers",
)
"""Where a ComfyUI-derived sampling core might live in this build.

Three names for one file across Forge's history. Tried in order and reported by
name, because *which* one answered is part of the finding: an extension written
against the third has been wrong about this host since the first was introduced.
"""

PATCHER_ATTRIBUTES = (
    "set_model_attn1_patch",
    "set_model_attn2_patch",
    "set_model_attn1_replace",
    "set_model_attn2_replace",
    "set_model_unet_function_wrapper",
    "set_model_patch",
)
"""Hooks a Forge ``ModelPatcher`` may offer, for §28's Path 3.

Reported and not used. An attention intervention is the mechanism of last resort
in R1's mitigation, and the useful thing to know before writing one is which of
these the loaded patcher actually has -- writing against ``set_model_attn2_patch``
on a build that only offers the wrapper is a morning lost to an AttributeError
that arrives during sampling.
"""


@dataclass
class Finding:
    """One candidate mechanism, and what was actually observed about it."""

    name: str
    available: bool = False
    detail: str = ""
    evidence: list = field(default_factory=list)

    def report(self) -> str:
        mark = "yes" if self.available else "no "
        lines = [f"  [{mark}] {self.name}: {self.detail}"]
        lines.extend(f"        {line}" for line in self.evidence)
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# The probe
# --------------------------------------------------------------------------- #


def _sampling_module():
    for name in SAMPLING_MODULES:
        try:
            return name, importlib.import_module(name)
        except Exception:
            continue
    return "", None


def probe_area_conditioning() -> Finding:
    """§28 Path 1: does the sampler honour an ``area`` on a conditioning entry?

    Looked for by finding ``get_area_and_mult`` and reading it, rather than by
    trusting that a ComfyUI-derived core still behaves like ComfyUI. It is the
    function that crops the model input to a conditioning entry's rectangle, and
    if it is not there then nothing downstream will look at an ``area`` key
    however carefully it is set.
    """
    found = Finding("Path 1 — host area conditioning")
    name, module = _sampling_module()
    if module is None:
        found.detail = ("no ComfyUI-derived sampling core was importable under any "
                        f"of {', '.join(SAMPLING_MODULES)}")
        return found

    function = getattr(module, "get_area_and_mult", None)
    if not callable(function):
        found.detail = f"{name} has no get_area_and_mult"
        return found

    found.available = True
    found.detail = f"{name}.get_area_and_mult is present"
    try:
        source = inspect.getsource(function)
        for key in ("'area'", '"area"', "'strength'", '"strength"'):
            if key in source:
                found.evidence.append(f"reads {key} off the conditioning entry")
    except Exception:
        found.evidence.append("source not readable; presence confirmed by import only")
    return found


def probe_mask_conditioning() -> Finding:
    """§28 Path 2: does the same function honour a spatial ``mask``?"""
    found = Finding("Path 2 — host mask conditioning")
    name, module = _sampling_module()
    function = getattr(module, "get_area_and_mult", None) if module else None
    if not callable(function):
        found.detail = "no get_area_and_mult to inspect"
        return found

    try:
        source = inspect.getsource(function)
    except Exception:
        found.detail = (f"{name}.get_area_and_mult is present but its source could "
                        "not be read, so mask support could not be confirmed")
        return found

    for key in ("'mask'", '"mask"', "'mask_strength'", '"mask_strength"'):
        if key in source:
            found.available = True
            found.evidence.append(f"reads {key}")
    found.detail = (f"{name}.get_area_and_mult honours a mask" if found.available
                    else f"{name}.get_area_and_mult does not mention a mask")
    return found


def probe_attention_hooks() -> Finding:
    """§28 Path 3: which attention hooks does the loaded patcher offer?

    Reported so that a regional-attention implementation, if one turns out to be
    necessary, can be written against a hook that exists. Nothing here installs
    one: R1's mitigation asks for this path only if necessary and asks for it to
    stay scoped, and a probe that patched the model to find out would be neither.
    """
    found = Finding("Path 3 — transformer attention hooks")
    try:
        from modules import shared

        objects = getattr(shared.sd_model, "forge_objects", None)
        unet = getattr(objects, "unet", None)
    except Exception as exc:
        found.detail = f"no loaded model to inspect ({exc})"
        return found

    if unet is None:
        found.detail = "the loaded model exposes no forge_objects.unet"
        return found

    offered = [name for name in PATCHER_ATTRIBUTES if callable(getattr(unet, name, None))]
    found.available = bool(offered)
    found.detail = (f"{type(unet).__name__} offers {len(offered)} of "
                    f"{len(PATCHER_ATTRIBUTES)} known hooks")
    found.evidence = list(offered)
    return found


def probe_architecture() -> Finding:
    """Is the thing that is loaded actually a Klein 9B, as the engine reports it?

    Asked through the extension's own detector so that the answer here and the
    answer a generation gets are the same answer. A spike run against the wrong
    checkpoint is worse than no spike: it reports a mechanism as missing when
    what is missing is the model.
    """
    found = Finding("Loaded architecture")
    try:
        _ensure_importable()
        import mc_arch
        import mc_spatial_klein

        arch = mc_spatial_klein.loaded_architecture()
        found.available = mc_spatial_klein.is_klein(arch)
        found.detail = f"{arch.label} ({arch.key})"
        if arch is mc_arch.UNKNOWN:
            found.evidence.append("nothing is loaded, or the engine is unrecognised")
        elif not found.available:
            found.evidence.append("select a FLUX.2 Klein 9B checkpoint and re-run")
    except Exception as exc:
        found.detail = f"could not be determined ({exc})"
    return found


def probe_conditioning_shape() -> Finding:
    """What one positive conditioning entry actually looks like in this build.

    The one finding that is pure reconnaissance. ``mc_spatial_klein`` builds a
    region's entry by copying the shape of the entries already there, precisely
    because this has changed between Forge releases -- and when that copy goes
    wrong the symptom is a shape error deep inside the sampler with nothing in
    the traceback naming the feature. Printing the real shape here is how that
    debugging session is avoided rather than survived.
    """
    found = Finding("Conditioning entry shape")
    try:
        from modules import shared

        model = shared.sd_model
        encoded = model.get_learned_conditioning(["a red cube"])
    except Exception as exc:
        found.detail = f"could not encode a probe prompt ({exc})"
        return found

    found.available = True
    found.detail = type(encoded).__name__
    entry = encoded
    for _ in range(3):
        if isinstance(entry, (list, tuple)) and entry:
            found.evidence.append(f"{type(entry).__name__} of {len(entry)}, "
                                  f"first is {type(entry[0]).__name__}")
            entry = entry[0]
            continue
        if isinstance(entry, dict):
            found.evidence.append(f"dict keys: {sorted(entry)}")
            break
        shape = getattr(entry, "shape", None)
        if shape is not None:
            found.evidence.append(f"tensor of shape {tuple(shape)}")
        break
    return found


def _repository_root():
    from pathlib import Path

    return Path(__file__).resolve().parent.parent


def _ensure_importable() -> None:
    """Put the extension's own modules on the path, once.

    Run standalone this file sits in ``tools/`` and the modules it measures sit
    one directory up, which is nowhere Python looks. Run inside the WebUI the
    path is already there and this is a no-op -- the same call answers both, so
    neither entry point has to know which one it is.
    """
    root = str(_repository_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def probe() -> list:
    """Every candidate, asked once. Cheap, and safe to run without a card."""
    return [probe_architecture(),
            probe_area_conditioning(),
            probe_mask_conditioning(),
            probe_attention_hooks(),
            probe_conditioning_shape()]


# --------------------------------------------------------------------------- #
# The measurement
# --------------------------------------------------------------------------- #

LEFT = (0, 200, 400, 800)
RIGHT = (600, 200, 1000, 800)
"""The two boxes, in the extension's own normalized 0..1000 space.

Deliberately large, deliberately non-overlapping, and deliberately on opposite
sides. §47 asks for exactly this shape of test and gives the reason: this is
statistical spatial guidance, so the question a small box asks -- did the object
land inside these particular coordinates -- is not one a diffusion model answers
reliably even when the mechanism is working perfectly. The question a large box
on one side asks is whether the mechanism moved anything at all, and that one has
a clean answer.
"""

GLOBAL_PROMPT = "a plain grey studio wall, soft even lighting"
REGION_PROMPT = "a large bright red glowing sphere"
"""One clearly named object against a background chosen to have none of it.

The measurement is a brightness-weighted red-channel centroid, so the background
is picked to contribute nothing to it. A scene prompt with its own colour in it
would move the centroid on its own and the result would measure the prompt.
"""

SEEDS = (1, 2, 3)
STEPS = 8
SIZE = 512


def generate(out_dir: str = "", seeds=SEEDS) -> list:
    """Run the left/right pairs and report where the object actually went.

    Returns one row per seed: ``(seed, left_centroid, right_centroid, moved)``,
    where each centroid is the horizontal position of the red mass as a fraction
    of the frame, 0 at the left edge and 1 at the right.

    ``moved`` is the whole finding. It is true when the object sat further right
    under the right box than under the left one, by a margin wide enough not to
    be noise -- which is the acceptance §47 asks for, and is not "an exception
    was not raised".
    """
    rows = []
    for seed in seeds:
        centroids = {}
        for side, bbox in (("left", LEFT), ("right", RIGHT)):
            image = _one_generation(bbox, seed)
            if image is None:
                return rows
            centroids[side] = _red_centroid(image)
            if out_dir:
                _save(image, out_dir, f"seed{seed}-{side}.png")
        moved = centroids["right"] - centroids["left"] > MARGIN
        rows.append((seed, centroids["left"], centroids["right"], moved))
    return rows


MARGIN = 0.08
"""How far the object must move before the run counts as a success.

Eight per cent of the frame width. Small enough that a real bias is not missed
and large enough that seed-to-seed wobble in an unconditioned scene does not
look like one; the same generation run twice with no regional conditioning at
all should sit well inside it, which is worth confirming with ``--control``.
"""


def _one_generation(bbox, seed: int):
    """One txt2img image with one region, through the extension's own backend.

    Through it and not around it, deliberately. A spike that reimplemented the
    installation would prove that *something* can bias an object and not that the
    thing this extension ships can, which is the only question worth a card.

    The wrapping below is the one liberty this file takes, and it is a liberty a
    maintainer's tool may take where the extension may not: in a real generation
    the host calls ``process_before_every_sampling`` and hands the hook both the
    conditioning and the latent, and here there is no script stack to be called
    from. So the sampler is wrapped at the moment it is assigned, which is the
    same moment for the same two objects, and unwrapped when the job ends.
    """
    _ensure_importable()
    import mc_spatial_klein
    from prompt_master import spatial

    try:
        from modules import processing, shared
        from modules.processing import StableDiffusionProcessingTxt2Img
    except Exception as exc:
        print(f"  ! the WebUI's processing module is not importable ({exc})")
        print("    run this from inside the WebUI environment, or import it in the "
              "running process -- see the module docstring.")
        return None

    backend = mc_spatial_klein.select_backend()
    if backend is None:
        print("  ! no regional-conditioning backend is available on this build; "
              "there is nothing to measure")
        return None

    region = spatial.SpatialRegion(identifier="spike", bbox_norm=tuple(bbox),
                                   prompt=REGION_PROMPT)
    request = spatial.SpatialRequest(
        enabled=True, requested_mode=spatial.REGIONAL_GENERATE,
        resolved_mode=spatial.REGIONAL_GENERATE,
        regions=(region,), source=spatial.NO_SOURCE)

    class _Spike(StableDiffusionProcessingTxt2Img):
        """A txt2img job that installs regional conditioning as the sampler starts.

        ``__setattr__`` and not an override of ``sample`` because the latent is
        built *inside* ``sample`` and handed straight to the sampler: the
        assignment of ``self.sampler`` is the last point before that at which
        anything can be arranged, and the sampler's own call is the first at
        which both the latent and the conditioning are in one place.
        """

        def __setattr__(self, name, value):
            if name == "sampler" and value is not None:
                value = _wrap_sampler(value, self, request, backend)
            super().__setattr__(name, value)

    p = _Spike(
        sd_model=shared.sd_model, prompt=GLOBAL_PROMPT, negative_prompt="",
        seed=seed, steps=STEPS, cfg_scale=1.0, width=SIZE, height=SIZE,
        n_iter=1, batch_size=1, do_not_save_samples=True, do_not_save_grid=True)

    try:
        processed = processing.process_images(p)
    finally:
        p.close()

    return processed.images[0] if getattr(processed, "images", None) else None


def _wrap_sampler(sampler, p, request, backend):
    """Hold regional conditioning open for the length of one sampler call."""
    import mc_spatial_klein

    original = sampler.sample

    def sample(processing_object, x, conditioning, unconditional_conditioning,
               *args, **kwargs):
        with mc_spatial_klein.regional_conditioning(
                request, conditioning, tensor=x, backend=backend,
                model=getattr(p, "sd_model", None)) as compiled:
            grid = getattr(compiled, "grid", ())
            if grid:
                print(f"    conditioning grid {grid[0]}x{grid[1]}, "
                      f"{len(compiled)} region(s) attached")
            return original(processing_object, x, conditioning,
                            unconditional_conditioning, *args, **kwargs)

    sampler.sample = sample
    return sampler


def _red_centroid(image) -> float:
    """Where the red mass sits horizontally, as a fraction of the frame.

    Red minus the mean of the other two channels, so a uniformly bright image
    contributes nothing and only actual redness moves the number. A frame with no
    red at all returns 0.5, which is "nothing was measured" and is reported as a
    failure to move rather than as a centre hit.
    """
    import numpy as np

    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    redness = array[:, :, 0] - 0.5 * (array[:, :, 1] + array[:, :, 2])
    redness = np.clip(redness, 0.0, None)
    total = redness.sum()
    if total <= 1e-6:
        return 0.5

    columns = redness.sum(axis=0)
    positions = np.arange(columns.shape[0], dtype=np.float32) / max(columns.shape[0] - 1, 1)
    return float((columns * positions).sum() / total)


def _save(image, out_dir: str, name: str) -> None:
    import os

    os.makedirs(out_dir, exist_ok=True)
    image.save(os.path.join(out_dir, name))


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Probe Forge Neo for a FLUX.2 Klein regional-conditioning path.")
    parser.add_argument("--probe", action="store_true",
                        help="report which mechanisms this build exposes (cheap)")
    parser.add_argument("--generate", action="store_true",
                        help="run the left/right measurement (needs a card and a "
                             "loaded Klein checkpoint)")
    parser.add_argument("--out", default="",
                        help="directory to write the measured images into")
    parser.add_argument("--seeds", default="",
                        help="comma-separated seeds; defaults to 1,2,3")
    args = parser.parse_args(argv)

    if not args.probe and not args.generate:
        args.probe = True

    if args.probe:
        print("Regional-conditioning probe")
        print("---------------------------")
        for finding in probe():
            print(finding.report())
        print()

    if not args.generate:
        return 0

    seeds = SEEDS
    if args.seeds:
        try:
            seeds = tuple(int(value) for value in args.seeds.split(","))
        except ValueError:
            print("--seeds wants a comma-separated list of integers")
            return 2

    print("Left/right measurement")
    print("----------------------")
    rows = generate(args.out, seeds)
    if not rows:
        print("  nothing was measured; see the probe above")
        return 1

    for seed, left, right, moved in rows:
        print(f"  seed {seed}: left box -> {left:.3f}   right box -> {right:.3f}   "
              f"{'moved' if moved else 'DID NOT MOVE'}")

    succeeded = sum(1 for *_rest, moved in rows if moved)
    print()
    print(f"  {succeeded} of {len(rows)} seeds moved the object by more than "
          f"{MARGIN:.0%} of the frame.")
    if succeeded == len(rows):
        print("  VERDICT: this mechanism biases placement. Record it in "
              "mc_spatial_klein.BACKENDS and carry on with §53 STEP 1.")
        return 0
    if succeeded:
        print("  VERDICT: partial. Widen the seed set before concluding either way; "
              "this is statistical guidance, not geometry.")
        return 1
    print("  VERDICT: nothing moved. Per §53, STOP and reassess the host "
          "integration before building any UX on this mechanism. Do not ship a "
          "prompt-hint fallback as regional conditioning.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
