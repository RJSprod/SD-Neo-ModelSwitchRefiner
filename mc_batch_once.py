"""Count each image once in the host's VRAM request, on a pass at CFG 1.

Before a pass, Forge's ``sampling_prepare`` asks ``load_models_gpu`` for room
beside the UNet's weights for the pass's activations, and it sizes that room
from a latent with the batch doubled -- ``(2 if shared.batch_cond_uncond else
1) * batch`` -- because a pass at a CFG above one runs the positive and the
negative prompt through the model together. At CFG 1 it runs the positive
alone: ``sampling_function_inner`` drops the negative outright (``uncond_ =
None``). The request is then for a pass that never runs, twice what the pass
can allocate, and on a card with little spare room beside the weights it is
what pushes the text encoder off before every batch above one. From the
user's log, a 1024x1024 pass on a 24 GB card holding an 18.4 GB checkpoint:
4.3 GB free beside the weights, 3.5 GB asked at batch 1, 6.3 GB at batch 2,
11.9 GB at batch 4, and roughly 0.75 GB of working VRAM measured per image.

``shared.batch_cond_uncond`` is a module constant in Forge Neo
(``modules/shared.py``: ``batch_cond_uncond = True``), read in two places: the
request above, and ``can_concat_cond``, which decides whether two conditioning
entries may share one forward pass. The Settings page's *Batch Cond/Uncond*
(``opts.batch_cond_uncond``) is a different thing that neither reads, so it
cannot shrink the request.

This module sets the constant to False for the moment the host reads it, and
only when that is both safe and honest:

* the pass runs at CFG 1 with the host's CFG-1 optimisation on, so no negative
  pass will run and the second count is for nothing;
* every image's conditioning is one entry, so ``can_concat_cond`` has nothing
  to concatenate and the flag's other reading changes nothing;
* the host still reads the constant where this expects it -- the source of
  ``sampling_prepare`` names it; and
* the Forge build is the one the setting was turned on with.

The last is the user's rule: an override of a host internal is tied to the
host it was verified on, and a Forge update stands it down until the setting
is turned off and on again on the new build. The record of the build is a
hidden option, like the extra networks' sort record. The first start with the
setting on pins the build then installed.

The flag goes back as soon as the host has read it: at the first sampling
step (the denoiser callback, which runs after ``sampling_prepare`` and before
``can_concat_cond``), and failing that after the pass and before the next
generation. Nothing in the host reads it in between.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import json
import logging
import math
import os
import subprocess
import time

logger = logging.getLogger("model_chain")

OPT_ENABLED = "model_chain_batch_once"
ENABLED_DEFAULT = True

OPT_RECORD = "model_chain_batch_once_record"
"""The Forge build the override was turned on with: ``{"build": ..., "when": ...}``."""

HIDDEN_SECTION = (None, "Model Chain: batch counted once (hidden)")

UNKNOWN_BUILD = "unknown"

_build: str | None = None
"""The host's build, read once per process."""

_prior: bool | None = None
"""The flag's value before this module flipped it; None while it is not flipped."""

_said: set = set()
"""Notices said once this session, by key."""


def settings(shared) -> dict:
    """The option of the Settings page, registered by the main script in its section."""
    return {
        OPT_ENABLED: shared.OptionInfo(
            ENABLED_DEFAULT,
            "Count each image once in Forge's VRAM request at CFG 1",
        ).info(
            "Forge asks for room for every image twice, positive and negative, and at "
            "CFG 1 it never runs the negative. This asks once there, so a batch pushes "
            "less off the card. Pinned to the Forge build it was turned on with: after a "
            "Forge update it stands down until turned off and on again"
        ),
    }


def hidden_settings(shared) -> dict:
    return {
        OPT_RECORD: shared.OptionInfo(
            "", "The Forge build the batch-count override was turned on with"),
    }


# --------------------------------------------------------------------------- #
# The host: its build, and whether it still has the shape this relies on
# --------------------------------------------------------------------------- #


def host_build() -> str:
    """Forge's version and commit, e.g. ``neo 2.30 @ 1a2b3c4d5e``; read once."""
    global _build
    if _build is None:
        _build = _read_host_build()
    return _build


def _read_host_build() -> str:
    parts = []
    try:
        forge_version = importlib.import_module("modules_forge.forge_version")
        version = " ".join(
            str(getattr(forge_version, name, "") or "") for name in ("version", "release")
        ).strip()
        if version:
            parts.append(version)
    except Exception:
        pass
    commit = _git_head()
    if commit:
        parts.append(f"@ {commit}")
    return " ".join(parts) or UNKNOWN_BUILD


def _script_path() -> str:
    for name in ("modules.paths_internal", "modules.paths"):
        try:
            path = getattr(importlib.import_module(name), "script_path", "")
        except Exception:
            continue
        if path:
            return str(path)
    return ""


def _git_head() -> str:
    """The short hash of the host's checkout, or "" where there is none to read."""
    root = _script_path()
    if not root or not os.path.isdir(os.path.join(root, ".git")):
        return ""
    try:
        run = subprocess.run(
            ["git", "-C", root, "rev-parse", "--short=10", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except Exception:
        return ""
    head = (run.stdout or "").strip()
    if run.returncode != 0 or not head or any(c not in "0123456789abcdef" for c in head):
        return ""
    return head


def host_shape() -> str:
    """"" when the host reads the flag where this expects it, else why not."""
    try:
        shared = importlib.import_module("modules.shared")
    except Exception:
        return "the host's shared module cannot be imported"
    if not isinstance(getattr(shared, "batch_cond_uncond", None), bool):
        return "modules.shared has no batch_cond_uncond flag"
    try:
        sampling = importlib.import_module("backend.sampling.sampling_function")
        source = inspect.getsource(sampling.sampling_prepare)
    except Exception:
        return "the host's sampling_prepare cannot be read"
    if "shared.batch_cond_uncond" not in source:
        return "the host's sampling_prepare no longer reads shared.batch_cond_uncond"
    return ""


# --------------------------------------------------------------------------- #
# The setting and the record
# --------------------------------------------------------------------------- #


def enabled() -> bool:
    import mc_memory

    return bool(mc_memory.option(OPT_ENABLED, ENABLED_DEFAULT))


def record() -> dict:
    """The stored record, or ``{}`` for none or one that cannot be read."""
    try:
        from modules import shared

        raw = getattr(shared.opts, OPT_RECORD, "") or ""
    except Exception:
        return {}
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        found = json.loads(raw)
    except ValueError:
        return {}
    return found if isinstance(found, dict) else {}


def _write_record(found: dict | None) -> None:
    from modules import shared

    shared.opts.set(OPT_RECORD, json.dumps(found, ensure_ascii=False) if found else "")
    try:
        shared.opts.save(shared.config_filename)
    except Exception:
        logger.debug("Model Chain: the batch-count record could not be saved", exc_info=True)


def pin(build: str | None = None) -> str:
    """Record the host's build as the one the override is turned on with."""
    build = build or host_build()
    _write_record({"build": build, "when": time.strftime("%Y-%m-%d %H:%M:%S")})
    return build


def unpin() -> None:
    _write_record(None)


@dataclasses.dataclass(frozen=True)
class Status:
    active: bool
    state: str
    """``off``, ``unsupported``, ``unpinned``, ``stood_down`` or ``active``."""
    reason: str
    build: str
    pinned: str


def status() -> Status:
    build = host_build()
    if not enabled():
        return Status(False, "off", "the setting is off", build, "")
    shape = host_shape()
    if shape:
        return Status(False, "unsupported", shape, build, "")
    pinned = str(record().get("build") or "")
    if not pinned:
        return Status(False, "unpinned", "no Forge build has been recorded yet", build, "")
    if pinned != build:
        return Status(
            False, "stood_down",
            f"Forge is {build} now, not the {pinned} the setting was turned on with",
            build, pinned,
        )
    return Status(True, "active", "", build, pinned)


def _say_once(key: str, level: int, message: str) -> None:
    if key in _said:
        return
    _said.add(key)
    logger.log(level, message)


def say_state() -> Status:
    """One line on the override's state, once per state per session."""
    found = status()
    if found.state == "active":
        _say_once(
            "active:" + found.build, logging.INFO,
            f"Model Chain: Forge's VRAM request counts each image once at CFG 1 — "
            f"pinned to Forge {found.build}, the build it was turned on with",
        )
    elif found.state == "stood_down":
        _say_once(
            "stood_down:" + found.build, logging.WARNING,
            f"Model Chain: {found.reason} — counting each image once is off until the "
            "setting is turned off and on again on this build",
        )
    elif found.state == "unsupported":
        _say_once(
            "unsupported:" + found.reason, logging.WARNING,
            f"Model Chain: {found.reason} — counting each image once is off",
        )
    return found


def on_app_started(_demo=None, _app=None) -> None:
    """Pin the build on the first start with the setting on; say the state; watch the switch.

    Signature is ``script_callbacks.on_app_started``'s. Never raises: a host
    this cannot read is a host the override simply stays off on.
    """
    try:
        if enabled() and not record().get("build") and not host_shape():
            pin()
        say_state()
        _watch_switch()
    except Exception:
        logger.debug("Model Chain: could not settle the batch-count override", exc_info=True)


def _watch_switch() -> None:
    """Re-pin when the setting is turned on, clear the record when it is turned off."""
    try:
        from modules import shared

        shared.opts.onchange(OPT_ENABLED, switched, call=False)
    except Exception:
        logger.debug("Model Chain: the host offers no onchange for the batch-count setting",
                     exc_info=True)


def switched() -> None:
    """The setting was changed on the Settings page."""
    _said.clear()
    if enabled():
        pin()
        say_state()
    else:
        unpin()
        logger.info("Model Chain: counting each image once in Forge's VRAM request is off; "
                    "the record of the Forge build is cleared")


# --------------------------------------------------------------------------- #
# The pass
# --------------------------------------------------------------------------- #


def pass_cfg(p) -> float:
    """The CFG the pass about to run uses: the hires pass has one of its own."""
    if getattr(p, "is_hr_pass", False):
        return float(getattr(p, "hr_cfg", getattr(p, "cfg_scale", 0.0)) or 0.0)
    return float(getattr(p, "cfg_scale", 0.0) or 0.0)


def applies(p, unet, conditioning) -> str:
    """"" when the override may be applied to this pass, else why not."""
    try:
        cfg = pass_cfg(p)
    except (TypeError, ValueError):
        return "the pass's CFG cannot be read"
    if not math.isclose(cfg, 1.0):
        return f"CFG is {cfg:g}, so the negative pass runs and both counts are real"
    options = getattr(unet, "model_options", None) or {}
    try:
        if options.get("disable_cfg1_optimization", False):
            return "the host's CFG 1 optimisation is off, so the negative pass runs"
    except AttributeError:
        pass
    parts = getattr(conditioning, "batch", None)
    if parts is not None:
        try:
            if any(len(entry) != 1 for entry in parts):
                return "a prompt is composed of more than one part"
        except TypeError:
            return "the conditioning cannot be read"
    return ""


def apply(p, unet, conditioning) -> bool:
    """Flip the host's flag for the request the sampler is about to make.

    Called from the last hook before ``sampling_prepare``. True when the flag
    was flipped; :func:`first_step` or :func:`restore` puts it back.
    """
    found = status()
    if not found.active:
        return False
    why = applies(p, unet, conditioning)
    if why:
        logger.debug("Model Chain: counting each image twice for this pass — %s", why)
        return False
    try:
        from modules import shared

        restore()
        global _prior
        _prior = bool(shared.batch_cond_uncond)
        shared.batch_cond_uncond = False
    except Exception:
        logger.debug("Model Chain: could not flip the host's batch flag", exc_info=True)
        return False
    _say_once(
        "applied", logging.INFO,
        f"Model Chain: counting each image once in Forge's VRAM request for this CFG 1 "
        f"pass (Forge {found.build}); the request would otherwise count every image "
        "twice, for a negative pass that does not run at CFG 1",
    )
    return True


def restore() -> bool:
    """Put the host's flag back. True when there was something to put back."""
    global _prior
    if _prior is None:
        return False
    prior, _prior = _prior, None
    try:
        from modules import shared

        shared.batch_cond_uncond = prior
    except Exception:
        logger.debug("Model Chain: could not put the host's batch flag back", exc_info=True)
        return False
    return True


def first_step() -> None:
    """The first sampling step: the host has read its flag."""
    restore()


def in_force() -> bool:
    """Whether the flag is flipped right now, for the reading of the request."""
    return _prior is not None
