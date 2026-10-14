"""The Voice Box's engines: which there are, which is active, and how it changes.

The Voice Box renders with one engine at a time -- VibeVoice 7B or Breeze TTS 2
-- and this module is the only place that says which. It is the shape Voice
Chat's registry has (:mod:`mc_voice_engines`, docs/18): one frozen row per
engine naming its modules, one :func:`active` answer, and one :func:`select`
that is the only way the answer changes.

Each engine is three things in a row of :data:`SPECS`:

    an adapter    what is installed and how, the engine's settings, what a
                  render needs, and the engine's vocabulary as data the Voice
                  Box reads -- its configuration ``FIELDS``, ``clean``,
                  ``refuse``, ``infotext`` and ``capabilities``;
    a runtime     one worker process per card, rendered through and evicted
                  (``load``, ``render``, ``cancel``, ``evict``, ``shutdown``…);
    a guest name  what :mod:`mc_turns` calls the engine on a card, registered
                  from outside the voice side by :mod:`mc_turns_guests`.

Never two engines at once
-------------------------
:func:`select` refuses while anything is queued or running, ends the outgoing
engine's workers on every card -- through the turn client when it has one, so a
warm stay ends the way it always ends: the image model it parked comes back and
WanGP's card is handed back -- and then makes sure with the runtime's own
``shutdown``. It starts nothing: the incoming engine's worker starts on the next
render, as every worker always has. So at most one engine's worker is ever
resident, on one card or on two, and the turn system sees exactly what it saw
when there was one engine.

Standard library only, imports nothing from the memory side (invariant I-3),
and imports an engine's modules only when asked for them.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""


@dataclass(frozen=True)
class BoxEngineSpec:
    """One engine of the Voice Box, by the modules that make it."""

    id: str
    label: str
    guest: str
    adapter: str
    runtime: str


SPECS = (
    BoxEngineSpec("vibevoice", "VibeVoice", "VibeVoice", "mc_voice_vibevoice",
                  "mc_voice_vibevoice_runtime"),
    BoxEngineSpec("breeze", "Breeze TTS 2", "Breeze", "mc_voice_breeze",
                  "mc_voice_breeze_runtime"),
)

ENGINES = tuple(spec.id for spec in SPECS)
DEFAULT_ENGINE = "vibevoice"
"""The engine a Voice Box with no choice recorded renders with: the one it had."""


class EngineError(ValueError):
    """An engine id this build does not know, or a switch that cannot happen now."""

    status = 400


class Busy(EngineError):
    status = 409


def spec(engine_id: str = "") -> BoxEngineSpec:
    """The row for ``engine_id`` -- or the active engine's for ``""``."""
    wanted = str(engine_id or "").strip().lower() or active_id()
    for found in SPECS:
        if found.id == wanted:
            return found
    raise EngineError(f"{engine_id!r} is not an engine the Voice Box has.")


def known(engine_id) -> bool:
    return str(engine_id or "").strip().lower() in ENGINES


def active_id() -> str:
    """The engine the Voice Box's settings name, or the default for anything else."""
    import mc_voice_box

    found = str((mc_voice_box.settings() or {}).get("engine") or "").strip().lower()
    return found if found in ENGINES else DEFAULT_ENGINE


def active() -> BoxEngineSpec:
    return spec(active_id())


def adapter(engine_id: str = ""):
    """The engine's adapter module, imported on first use."""
    return importlib.import_module(spec(engine_id).adapter)


def runtime(engine_id: str = ""):
    """The engine's runtime module, imported on first use."""
    return importlib.import_module(spec(engine_id).runtime)


def label(engine_id: str = "") -> str:
    try:
        return spec(engine_id).label
    except EngineError:
        return str(engine_id or "")


def summary() -> dict:
    """``{"active", "available": [{id, label, installed, ready, supported, message}]}``.

    For the page's Engine select. Each engine's installation is read from disk
    the way its own status is, and an engine whose status cannot be read is
    listed with the reason rather than left out.
    """
    rows = []
    for found in SPECS:
        row = {"id": found.id, "label": found.label, "installed": False, "ready": False,
               "supported": False, "message": ""}
        try:
            status = importlib.import_module(found.adapter).status()
            row.update(installed=bool(status.ready), ready=bool(status.ready),
                       supported=bool(status.supported), message=str(status.message or ""))
        except Exception as exc:
            logger.debug("Model Chain: the Voice Box could not read %s's status", found.label,
                         exc_info=True)
            row["message"] = f"{found.label}'s installation could not be read ({exc})."
        rows.append(row)
    return {"active": active_id(), "available": rows}


def select(engine_id: str, client=None) -> dict:
    """Make ``engine_id`` the Voice Box's engine. The only way the engine changes.

    Refused (:class:`Busy`) while a render or a design is queued or running on
    any card -- the page says so and offers Cancel and Clear queue. Otherwise
    every worker of the outgoing engine is ended before the choice is written:
    through ``client`` (the Voice Box's turn client) for every card the
    outgoing runtime reports, then by the runtime's own ``shutdown`` in case
    one was started outside any turn. Nothing of the incoming engine is
    started here.
    """
    import mc_voice_box

    wanted = spec(engine_id)
    current = active()
    if wanted.id == current.id:
        return {"changed": False, "engine": wanted.id}
    if mc_voice_box.busy():
        raise Busy("A render is running or queued — cancel it or clear the queue before "
                   "switching engines.")
    reason = f"the Voice Box is switching to {wanted.label}"
    _release(current, client, reason)
    mc_voice_box.set_settings({"engine": wanted.id})
    logger.info("Model Chain: the Voice Box now renders with %s (was %s)", wanted.label,
                current.label)
    return {"changed": True, "engine": wanted.id}


def _release(outgoing: BoxEngineSpec, client, reason: str) -> None:
    """End every worker ``outgoing`` has, on every card. Never raises."""
    try:
        module = importlib.import_module(outgoing.runtime)
    except Exception:
        logger.debug("Model Chain: %s's runtime could not be imported to stop it",
                     outgoing.label, exc_info=True)
        return
    try:
        cards = dict((module.status() or {}).get("cards") or {})
    except Exception:
        cards = {}
    for key, card in cards.items():
        uuid = str((card or {}).get("uuid") or key)
        if client is not None:
            try:
                client.unload(uuid, reason, guest=outgoing.guest)
                continue
            except Exception:
                logger.warning("Model Chain: %s could not be unloaded from a card through "
                               "its turn; stopping it directly", outgoing.label, exc_info=True)
        try:
            module.evict(uuid, reason)
        except Exception:
            logger.debug("Model Chain: %s could not be evicted", outgoing.label, exc_info=True)
    try:
        module.shutdown()
    except Exception:
        logger.debug("Model Chain: %s's shutdown failed", outgoing.label, exc_info=True)


def shutdowns() -> list:
    """``[(label, runtime.shutdown)]`` for every engine whose runtime imports.

    For the WebUI's shutdown door, which ends every engine's workers whichever
    is active: a switch never leaves one running, but a shutdown does not rely
    on that.
    """
    found = []
    for row in SPECS:
        try:
            module = importlib.import_module(row.runtime)
        except Exception:
            logger.debug("Model Chain: %s's runtime could not be imported for the shutdown",
                         row.label, exc_info=True)
            continue
        found.append((row.label, module.shutdown))
    return found
