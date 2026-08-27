"""Vision as a capability a llama-server acquires, not a mode a request sets.

Three facts about a multimodal backbone are separate, and the regression this
module exists to close came from treating them as one:

    the registry says this backbone has a projector        -- *declared*
    the projector is a file on this disk                   -- *present*
    the running llama-server was started with ``--mmproj`` -- *loaded*

A managed multimodal backbone whose projector is declared and present but not
loaded is not a broken installation. It is the preferred cold-start state: the
lightest possible text-only server, with everything still known that a first
image-bearing request needs to upgrade it without asking the user anything.

What lives here is the middle fact and the bridge to the first. Given a
configuration that is about to be shown a picture, :func:`ensure_projector`
answers "which file do I load", repairing a managed bundle from the registry
when the file is missing, and it does that *before* the runtime takes its
process lock -- a slow download must not hold an unrelated warm server shut
(design intent section 13).

The third fact is not here. Whether a process was started with a projector is
process state, and it belongs to :class:`mc_llm_runtime.Runtime` -- see
``Runtime._projector`` and :meth:`mc_llm_runtime.Runtime.vision_loaded`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import mc_llm_paths

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""


class VisionUnavailable(RuntimeError):
    """A projector this request needs is known and could not be provided.

    Distinct from "this model has no projector at all", which is a sentence
    about the *selection* and is written by the caller. This one always names
    an artifact and a local reason, because it is the failure a user can do
    something about -- and because design intent section 17 forbids the other
    way of handling it, which would be to send the picture somewhere else.
    """


NO_PROJECTOR = (
    "This request carries an image, and the model running has no vision projector. "
    "Choose one in LLM Studio’s Setup mode, or pick a managed multimodal backbone; "
    "send the request without the image if you meant it as text. Text-only fallback "
    "is disabled, and nothing is ever sent to a cloud model."
)
"""What a request that needs eyes is told when the selection simply has none.

Spelled once, here, because three callers used to spell it three ways and the
one thing every version of it has to keep saying is the last clause: an image
request that cannot be served locally fails locally (invariants I-4 and I-10).
"""


# --------------------------------------------------------------------------- #
# What the registry declares
# --------------------------------------------------------------------------- #


def declared_projector(identifier: str):
    """The catalogue's projector artifact for ``identifier``, or ``None``.

    Never raises. A registry that will not load is a reason to fall back to
    whatever the state file already names, exactly as :func:`mc_llm_runtime.config`
    treats the same failure -- not a reason for an image request to explode
    with a JSON error.
    """
    if not identifier:
        return None
    try:
        import mc_llm_managed_models

        return mc_llm_managed_models.entry(identifier).projector
    except Exception:
        logger.debug("Model Chain: could not read the managed projector for %s", identifier,
                     exc_info=True)
        return None


def declared_path(identifier: str) -> Path | None:
    """Where ``identifier``'s projector belongs on disk, declared or not present.

    Deliberately answers about a file that may not exist. Section 10: the
    configuration records *the compatible projector to use when vision is
    needed*, which is a fact about the bundle rather than about the filesystem,
    and losing it because the file is missing is how a repairable installation
    comes to look like a text-only one.
    """
    if declared_projector(identifier) is None:
        return None
    try:
        import mc_llm_managed_models

        return mc_llm_managed_models.bundle_root(identifier) / mc_llm_managed_models.MMPROJ_FILENAME
    except Exception:
        logger.debug("Model Chain: could not place the managed projector for %s", identifier,
                     exc_info=True)
        return None


def projector_for_model(model_path) -> Path | None:
    """The trusted projector for a GGUF somebody picked out of a file list.

    The whole of invariant I-8 at the one place it can be enforced. A managed
    bundle lives under the folder the ordinary model chooser scans, so picking
    a curated multimodal backbone there is routine rather than exotic -- and
    the chooser's own rule, quite correctly, is that a projector is never
    inferred from a filename. A registry entry is not an inference. It is the
    association the bundle was downloaded with, and it survives every selection
    path by being restored here rather than guessed at there.
    """
    try:
        import mc_llm_managed_models

        identifier = mc_llm_managed_models.identify_path(model_path)
    except Exception:
        logger.debug("Model Chain: could not recognise %s as a managed bundle", model_path,
                     exc_info=True)
        return None
    return declared_path(identifier) if identifier else None


# --------------------------------------------------------------------------- #
# Resolving and repairing, before any process lock is held
# --------------------------------------------------------------------------- #


def ensure_projector(configuration, role: str = "", say=None, cancel=None) -> Path | None:
    """The projector to start this configuration's server with, repaired if need be.

    ``None`` means the selection genuinely has no eyes -- a hand-picked GGUF
    with no projector, which is a perfectly good text model and whose image
    requests are refused with :data:`NO_PROJECTOR` by the caller. It never
    means "the download failed": that raises :class:`VisionUnavailable`, so a
    repairable bundle can never be mistaken for a text-only one.

    Section 11's sequence, minus the step that is not this module's to take:

        1. the projector the configuration names is on disk      -> use it
        2. the managed bundle's declared projector is on disk     -> record and use it
        3. the managed registry declares one that is missing      -> repair it
        4. nothing is declared                                    -> ``None``

    Called *outside* :attr:`mc_llm_runtime.Runtime._lock` on purpose. Step 3
    can be a gigabyte over somebody's connection, and section 13 is explicit
    that a slow transfer must not lock a llama-server that is answering other
    requests perfectly well in the meantime.
    """
    known = configuration.mmproj
    if known is not None and Path(known).is_file():
        return Path(known)

    identifier = str(getattr(configuration, "managed_id", "") or "")
    if not identifier or getattr(configuration, "source", "manual") != "managed":
        # A manual install whose recorded projector has gone missing is a
        # broken path rather than an incomplete bundle: there is no trusted
        # association to repair it from, and guessing at one is I-8's whole
        # prohibition. The runtime says which file is missing.
        return Path(known) if known is not None else None

    declared = declared_path(identifier)
    if declared is None:
        return Path(known) if known is not None else None

    if declared.is_file():
        if known is None or Path(known) != declared:
            logger.info("Model Chain: the managed backbone %s declares a vision projector that "
                        "was not recorded; the association has been restored", identifier)
            remember(declared, role)
        return declared

    logger.info("Model Chain: %s needs vision and %s's projector is not on disk; repairing the "
                "managed bundle before anything is started", role or "this request", identifier)
    import mc_llm_managed_models

    try:
        repaired = mc_llm_managed_models.repair_projector(identifier, on_status=say,
                                                          cancel=cancel)
    except mc_llm_managed_models.Cancelled:
        # Section 24.4. A user who pressed Stop is told they pressed Stop, in
        # the downloader's own words -- "what has arrived is kept, so starting
        # it again carries on from there" -- rather than being told the bundle
        # could not be provisioned, which is a different and worrying thing.
        raise
    except Exception as exc:
        # Section 24.1: the text server that is running stays running. Nothing
        # has been stopped at this point -- that is the reason this call
        # happens before the runtime's lock is taken and not inside it.
        raise VisionUnavailable(
            f"This request needs the vision projector for {identifier}, and it could not be "
            f"put in place ({exc}). The model you are running is untouched and still answers "
            f"text; nothing was sent anywhere else."
        ) from None

    remember(repaired, role)
    return repaired


def remember(projector, role: str = "") -> None:
    """Record ``projector`` as this configuration's compatible one. Never raises.

    Written into the layer the *model* came from, which is the only way a role
    with a backbone of its own does not quietly inherit the installation's
    projector -- and the only way a role that follows the installation does not
    grow an override it never asked for.

    Best-effort by design. A state file that cannot be written is a reason for
    the next vision request to resolve the projector again, which costs a
    directory listing; it is not a reason to fail the image request that has
    just had its projector successfully verified.
    """
    try:
        import mc_llm_roles
        from prompt_master.core.config import atomic_write_json, read_json

        paths = mc_llm_paths.app_paths()
        try:
            state = read_json(paths.state_file)
        except (OSError, ValueError):
            state = {}
        recorded = paths.record(Path(projector))
        chosen = mc_llm_roles.named(role)
        own = mc_llm_roles.overrides(chosen, state, keys=("model", "mmproj")) if chosen else {}
        if "model" in own:
            if str(own.get("mmproj") or "") == recorded:
                return
            mc_llm_roles.apply(state, chosen, {"mmproj": recorded},
                               keys=mc_llm_roles.STATE_FIELDS)
        else:
            if str(state.get("mmproj") or "") == recorded:
                return
            state["mmproj"] = recorded
        paths.data.mkdir(parents=True, exist_ok=True)
        atomic_write_json(paths.state_file, state)
        logger.info("Model Chain: recorded %s as the compatible vision projector%s",
                    Path(projector).name, f" for {chosen}" if chosen and "model" in own else "")
    except Exception:
        logger.debug("Model Chain: could not record the vision projector", exc_info=True)


__all__ = ["NO_PROJECTOR", "VisionUnavailable", "declared_path", "declared_projector",
           "ensure_projector", "projector_for_model", "remember"]
