"""Named Stage 2 configurations for the Model Chain extension.

A preset captures every Stage 2 control -- checkpoint, VAE / text encoder,
prompt mode and text, styles, seed handling, sampling, size and edit mode -- so
a working pairing can be recalled in one click instead of rebuilt by hand.

Presets live in a single JSON file under the WebUI's data directory rather than
in the extension folder, so reinstalling or updating the extension does not
throw them away. Writes go through a temporary file and an atomic replace: a
crash mid-save leaves the previous file intact rather than a truncated one.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

FILENAME = "model_chain_presets.json"

SCHEMA_VERSION = 1

FIELDS = (
    "enabled",
    "target",
    "modules",
    "prompt_mode",
    "prompt",
    "negative",
    "styles",
    "seed_mode",
    "seed_offset",
    "fixed_seed",
    "cfg",
    "steps",
    "sampler",
    "scheduler",
    "denoise",
    "size_multiplier",
    "edit_mode",
    "reference_mode",
    "reference_max_dim",
)
"""Every Stage 2 control, by the name the UI uses for it.

Kept in the same order the accordion returns its controls so a preset reads
like the panel. Adding a control here without adding it to the UI (or the other
way round) is caught by the tests.

One control is deliberately absent: the Decoupled reference gallery. A preset is
a small JSON file of settings, and images are not settings -- storing them would
mean copying pixels into the data directory and owning their lifetime. The mode
is saved, the images are re-supplied by hand, which is what native ImageStitch
asks for too.
"""

EXCLUDED_FIELDS = ("reference_images",)
"""UI controls a preset intentionally does not carry. See FIELDS."""

NONE = "None"
"""Shown in the dropdown when no preset is selected."""


class PresetError(RuntimeError):
    """Raised for a save or delete that cannot be carried out."""


def path() -> str:
    """Where presets are stored."""
    try:
        from modules import paths

        base = paths.data_path
    except Exception:
        base = os.getcwd()
    return os.path.join(base, FILENAME)


# --------------------------------------------------------------------------- #
# Storage
# --------------------------------------------------------------------------- #


def _read() -> dict:
    """Load the whole store, tolerating a missing or damaged file."""
    file = path()
    if not os.path.exists(file):
        return {}

    try:
        with open(file, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        logger.warning(
            "Model Chain: could not read presets from %s; treating it as empty",
            file,
            exc_info=True,
        )
        return {}

    if not isinstance(data, dict):
        return {}

    presets = data.get("presets", {})
    if not isinstance(presets, dict):
        return {}

    # Drop anything that is not a mapping rather than letting it reach the UI.
    return {name: values for name, values in presets.items() if isinstance(values, dict)}


def _write(presets: dict) -> None:
    file = path()
    payload = {"version": SCHEMA_VERSION, "presets": presets}

    directory = os.path.dirname(file) or "."
    try:
        os.makedirs(directory, exist_ok=True)
        # Write beside the target so the replace stays on one filesystem.
        handle = tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=directory, prefix=".model_chain_presets", suffix=".tmp", delete=False
        )
        try:
            with handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False)
            os.replace(handle.name, file)
        except Exception:
            try:
                os.unlink(handle.name)
            except OSError:
                pass
            raise
    except Exception as exc:
        raise PresetError(f"Could not save presets to {file}: {exc}") from exc


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #


def names() -> list[str]:
    """Saved preset names, case-insensitively sorted for a stable dropdown."""
    return sorted(_read(), key=str.casefold)


def choices() -> list[str]:
    """Dropdown choices, with the no-selection entry first."""
    return [NONE] + names()


def get(name: str) -> dict | None:
    """Return a preset's values, or None if it does not exist."""
    if not name or name == NONE:
        return None
    return _read().get(name)


def save(name: str, values: dict) -> list[str]:
    """Create or overwrite a preset. Returns the refreshed name list."""
    name = (name or "").strip()
    if not name:
        raise PresetError("Give the preset a name before saving.")
    if name == NONE:
        raise PresetError(f'"{NONE}" is reserved and cannot be used as a preset name.')

    presets = _read()
    existing = name in presets
    presets[name] = {key: values.get(key) for key in FIELDS}
    _write(presets)

    logger.info("Model Chain: %s preset %r", "updated" if existing else "saved", name)
    return names()


def delete(name: str) -> list[str]:
    """Remove a preset. Returns the refreshed name list."""
    if not name or name == NONE:
        raise PresetError("Select a preset to delete.")

    presets = _read()
    if name not in presets:
        raise PresetError(f'No preset named "{name}".')

    presets.pop(name)
    _write(presets)

    logger.info("Model Chain: deleted preset %r", name)
    return names()


def apply_defaults(values: dict, defaults: dict) -> dict:
    """Fill in fields a preset does not carry.

    A preset saved by an older version will not have controls added since. Those
    fall back to the current defaults rather than to None, which would blank the
    control.
    """
    return {key: values.get(key, defaults.get(key)) if values.get(key) is not None else defaults.get(key) for key in FIELDS}
