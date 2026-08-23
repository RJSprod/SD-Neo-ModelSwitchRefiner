"""Supplemental Stage 2 reference images for the Model Chain extension.

Forge Neo's "ImageStitch Integrated" does not stitch anything, despite the name:
it hands the loaded model an *ordered set of reference images*. Mechanically it
raises ``dynamic_args.is_referencing`` and encodes each gallery image through
``images_tensor_to_samples`` -> ``encode_first_stage``; the reference-capable
engines (``flux2``, ``krea``, ``anima``, ``qwen``, ``flux``) intercept that call
and append the result to ``sd_model.ref_latents``. At conditioning time each of
those engines assembles its reference list the same way::

    _references = [*self.ref_latents]        # ImageStitch's images, in order
    _references.insert(0, self.ini_latent)   # the img2img input image

That layout is what makes this feature work at all. Because the img2img input is
inserted at index 0, every Stage 2 image keeps its own Stage 1 result as the
primary reference and the supplemental set follows it in the order the user
arranged. ``ref_latents`` is not consumed by conditioning either, so a single
encode before the refine loop serves every image in the batch.

Model Chain runs Stage 2 through its own processing object with scripts
disabled, so ImageStitch never executes for the refine pass -- deliberately, see
the note in ``scripts/model_chain.py`` about not re-enabling the Stage 2 script
stack. Model Chain therefore owns Stage 2 reference routing itself. It does that
through the same host entry point ImageStitch uses rather than through a
parallel vision-conditioning implementation, and it never calls ImageStitch's
``process()``. The only thing it takes from ImageStitch is the contents of the
user's input gallery, read from ``p.script_args`` and never written back.
"""

from __future__ import annotations

import contextlib
import logging
import os

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

DISABLED = "Disabled"
PASS_THROUGH = "Pass Through ImageStitch"
DECOUPLED = "Decoupled"
MODES = (DISABLED, PASS_THROUGH, DECOUPLED)

DEFAULT_MAX_DIM = 1024
"""Matches ImageStitch's own "Maximum Side Length" default."""

ALIGNMENT = 64
"""Reference images are snapped to this grid, as ImageStitch snaps its own."""

STITCH_TITLE = "ImageStitch Integrated"

STITCH_GALLERY_SUFFIX = "imagestitch_integrated_ref_latent"
"""Tail of the elem_id ImageStitch gives its Reference Image(s) gallery.

``Script.elem_id`` prefixes ``script_`` and, for a script visible on both tabs,
the tab name -- so matching the tail rather than the whole id keeps this working
whichever tab the gallery was built for.
"""


def is_active(mode: str) -> bool:
    """Whether ``mode`` asks for supplemental references at all."""
    return mode in (PASS_THROUGH, DECOUPLED)


# --------------------------------------------------------------------------- #
# Reading ImageStitch's input gallery
# --------------------------------------------------------------------------- #


def find_stitch_script(p):
    """The live ImageStitch script instance for this generation, or None.

    Located by title rather than by import: ImageStitch is a builtin extension
    loaded from a path, so there is no importable module name to depend on, and
    a host that ships without it must simply degrade rather than fail.
    """
    runner = getattr(p, "scripts", None)
    for script in getattr(runner, "alwayson_scripts", None) or []:
        try:
            if script.title() == STITCH_TITLE:
                return script
        except Exception:
            continue
    return None


def stitch_arguments(p):
    """``(enabled, gallery, max_dim)`` as the user left ImageStitch's controls.

    Read from ``p.script_args`` -- the *input* gallery, which is what section 12
    permits Pass Through to take from ImageStitch. Nothing here touches
    ``ImageStitch.cached_parameters`` or any other internal state, and nothing is
    written back: the user's Stage 1 gallery is left exactly as they arranged it.

    Returns None when ImageStitch is not present or its arguments cannot be
    located, which the caller reports as a non-fatal notice.
    """
    script = find_stitch_script(p)
    if script is None:
        return None

    start, end = getattr(script, "args_from", None), getattr(script, "args_to", None)
    if start is None or end is None:
        return None

    try:
        # Sliced defensively: ``script_args`` is a list for any real generation,
        # but the host's own dataclass defaults it to an empty *dict*, and a
        # slice of that raises rather than coming back empty.
        args = list((getattr(p, "script_args", None) or [])[start:end])
    except (TypeError, KeyError):
        return None

    if len(args) < 3:
        return None

    enabled, gallery, max_dim = args[0], args[1], args[2]
    try:
        max_dim = int(max_dim)
    except (TypeError, ValueError):
        max_dim = DEFAULT_MAX_DIM
    return bool(enabled), gallery, max_dim


def invalidate_stitch_cache(p=None) -> None:
    """Make ImageStitch re-encode its own references on the next generation.

    ImageStitch memoises on (loading parameters, image hashes) and returns early
    when they are unchanged, because the references it encoded last time are
    still sitting on the model. Model Chain breaks that assumption twice over: it
    clears reference state off the Stage 2 model, which is the *same* object when
    both stages share a checkpoint, and its cache can hand back a checkpoint that
    was reloaded from disk with an empty reference list. Either way ImageStitch
    would skip an encode it needed to do, and Stage 1 would quietly run with no
    references at all.

    Dropping the memo costs one re-encode and cannot cause a wrong result, so it
    is done whenever Model Chain clears references rather than only when it can
    prove it has to.
    """
    for script in _stitch_scripts(p):
        try:
            # A class attribute: ImageStitch reads and writes it through the
            # class, so setting it on the instance would shadow rather than
            # clear it.
            type(script).cached_parameters = None
        except Exception:
            logger.debug("Model Chain: could not reset the ImageStitch cache", exc_info=True)


def stitch_is_installed() -> bool:
    """Whether ImageStitch is part of the UI currently being built.

    Asked while Model Chain's own panel is being laid out, to decide whether the
    live reference count has an ImageStitch gallery to wait for. The scripts are
    instantiated before any of their ``ui()`` methods run, so this is answerable
    then even though ImageStitch's components do not exist yet.
    """
    return bool(_stitch_scripts())


def _stitch_scripts(p=None) -> list:
    """Every reachable ImageStitch instance, from this generation and the tabs."""
    found = []

    script = find_stitch_script(p) if p is not None else None
    if script is not None:
        found.append(script)

    try:
        from modules import scripts

        runners = (
            getattr(scripts, "scripts_current", None),
            getattr(scripts, "scripts_txt2img", None),
            getattr(scripts, "scripts_img2img", None),
        )
        for runner in runners:
            for candidate in getattr(runner, "alwayson_scripts", None) or []:
                try:
                    if candidate.title() == STITCH_TITLE and candidate not in found:
                        found.append(candidate)
                except Exception:
                    continue
    except Exception:
        pass

    return found


# --------------------------------------------------------------------------- #
# Gallery values
# --------------------------------------------------------------------------- #


def extract_images(gallery) -> list:
    """PIL images from a Gradio gallery value, in the order they are shown.

    Deliberately liberal about the entry shape. A gallery reaches a script as
    ``(image, caption)`` pairs from the UI and as base64 strings from the API,
    and the exact payload has changed between Gradio releases -- so anything
    that can be resolved to an image is, and anything that cannot is skipped
    rather than failing the generation. Order is never rearranged: it is
    semantically meaningful to every architecture that consumes references.
    """
    if not gallery or isinstance(gallery, (str, bytes, dict)):
        # A bare string or mapping is a malformed gallery, not a one-image one.
        return []

    images = []
    try:
        entries = list(gallery)
    except TypeError:
        return []

    for entry in entries:
        image = _as_image(entry)
        if image is not None:
            images.append(image)
    return images


def _as_image(entry):
    try:
        from PIL import Image
    except Exception:
        return None

    if entry is None:
        return None
    if isinstance(entry, Image.Image):
        return entry
    if isinstance(entry, (tuple, list)):
        return _as_image(entry[0]) if entry else None
    if isinstance(entry, dict):
        for key in ("image", "name", "path", "data"):
            if entry.get(key) is not None:
                return _as_image(entry[key])
        return None
    if isinstance(entry, str):
        return _image_from_string(entry)
    return None


def _image_from_string(value: str):
    """A gallery entry handed over as a file path, or as base64 by the API."""
    try:
        from PIL import Image

        if os.path.exists(value):
            image = Image.open(value)
            image.load()
            return image
    except Exception:
        logger.warning("Model Chain: could not read the reference image %r", value, exc_info=True)
        return None

    try:
        from modules.api import api

        return api.decode_base64_to_image(value)
    except Exception:
        logger.warning("Model Chain: could not decode a reference image", exc_info=True)
        return None


def preprocess(image, limit: int):
    """Size a reference image the way ImageStitch sizes its own.

    The longest side is capped at ``limit`` (0 means no cap) and both sides are
    snapped to a multiple of 64. Mirrored rather than invented so a reference
    reaches the encoder in the same shape it would have through the native path;
    the engines round again to their own grid on top of this.
    """
    width, height = image.size
    limit = max(int(limit or 0), 0)

    if limit > 0 and max(width, height) > limit:
        ratio = limit / max(width, height)
        target_w, target_h = int(width * ratio), int(height * ratio)
    else:
        target_w, target_h = width, height

    if target_w % ALIGNMENT or target_h % ALIGNMENT:
        target_w = round(target_w / ALIGNMENT) * ALIGNMENT
        target_h = round(target_h / ALIGNMENT) * ALIGNMENT

    target_w, target_h = max(target_w, ALIGNMENT), max(target_h, ALIGNMENT)

    if (width, height) == (target_w, target_h):
        return image

    from modules import images

    return images.resize_image(1, image, target_w, target_h)


# --------------------------------------------------------------------------- #
# Encoding
# --------------------------------------------------------------------------- #


def reference_count(model=None) -> int:
    """How many references the loaded model is currently holding."""
    if model is None:
        try:
            from modules import shared

            model = shared.sd_model
        except Exception:
            return 0
    return len(getattr(model, "ref_latents", None) or ())


@contextlib.contextmanager
def conditioning_enabled(override: dict):
    """Hold an architecture's reference toggle open while references encode.

    ``encode_first_stage`` only stashes a reference when the model's *global*
    toggle says references are wanted -- ``opts.krea2_do_reference`` for Krea 2,
    ``not opts.klein_no_reference`` for Klein. Stage 2 gets that toggle through
    ``p.override_settings``, which the host does not apply until
    ``process_images`` starts the pass, and the references have to exist before
    then. So the same override is applied here for the length of the encode and
    unwound immediately, leaving the user's global value exactly as it was.

    Assigned directly rather than through ``opts.set``: this is a scoped
    borrow, not a settings change, and it must not run the option's onchange or
    reach the config file.
    """
    if not override:
        yield
        return

    from modules import shared

    previous = {}
    try:
        for key, value in override.items():
            previous[key] = getattr(shared.opts, key, None)
            setattr(shared.opts, key, value)
        yield
    finally:
        for key, value in previous.items():
            try:
                setattr(shared.opts, key, value)
            except Exception:
                logger.warning(
                    "Model Chain: failed to restore the %s setting after encoding references",
                    key,
                    exc_info=True,
                )


def encode(images: list, max_dim: int = DEFAULT_MAX_DIM) -> int:
    """Register ``images`` as ordered references on the loaded model.

    Returns how many the model actually took. That is the number to trust: it is
    an observation rather than a prediction, and it is the one signal that cannot
    be fooled by a checkpoint whose name promises a reference path its engine
    does not implement.

    One encode per job serves the whole Stage 2 batch, because conditioning
    reads ``ref_latents`` without consuming it. There is no cache *between* jobs
    on purpose: references are cleared at both ends of every job so a stale set
    cannot leak into the next one, and re-encoding a handful of small images
    costs far less than the risk of getting that wrong.
    """
    if not images:
        return 0

    from backend.args import dynamic_args
    from modules import shared

    model = shared.sd_model
    if model is None:
        return 0

    before = reference_count(model)

    dynamic_args.is_referencing = True
    try:
        for image in images:
            tensor = _to_tensor(preprocess(image, max_dim))
            if tensor is None:
                continue
            _encode_one(tensor, model)
    finally:
        dynamic_args.is_referencing = False

    return reference_count(model) - before


def _to_tensor(image):
    """A preprocessed reference as the ``[0, 1]`` NCHW tensor the encoder wants.

    The same conversion ImageStitch performs, including the flatten against the
    img2img background colour, so an image with alpha lands identically either
    way.
    """
    import numpy as np
    import torch

    from modules import images, shared

    flattened = images.flatten(image, shared.opts.img2img_background_color)
    array = np.array(flattened, dtype=np.float32) / 255.0
    array = np.moveaxis(array, 2, 0)
    return torch.from_numpy(array).to(device=shared.device).unsqueeze(0)


def _encode_one(tensor, model) -> None:
    """Push one reference through the host's own encode path.

    ``images_tensor_to_samples`` is what ImageStitch calls, and the ``0`` pins
    the full VAE: the TAESD approximation short-circuits before
    ``encode_first_stage`` and would silently register nothing at all.
    """
    from modules.sd_samplers_common import images_tensor_to_samples

    images_tensor_to_samples(tensor, 0, model)
