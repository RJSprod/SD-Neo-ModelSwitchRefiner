"""Style-library integration helpers for the Model Chain extension.

Everything here reads from ``modules.shared.prompt_styles`` -- the live,
in-memory store that also backs ``styles.csv`` and the main prompt's style
selector -- so styles created during the session are visible without a restart
(section 5.2). ``styles.csv`` is never parsed directly.

Style *application* delegates to the host's own
``StyleDatabase.apply_styles_to_prompt`` / ``apply_negative_styles_to_prompt``,
which guarantees the ``{prompt}`` placeholder convention and multi-style
ordering behave identically to the main prompt box. No substitution logic is
reimplemented here.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""


def _store():
    from modules import shared

    return shared.prompt_styles


def available_styles() -> list[str]:
    """Names of every style currently in the live store."""
    try:
        return list(_store().styles)
    except Exception:
        return []


def reload_styles() -> list[str]:
    """Re-read the style store from disk and return the refreshed name list.

    Backs the refresh button next to the style dropdown (section 5.3).
    """
    try:
        _store().reload()
    except Exception:
        logger.warning("Model Chain: failed to reload the style library", exc_info=True)
    return available_styles()


def prune_selection(selected: list[str] | None) -> tuple[list[str], list[str]]:
    """Split a selection into (still valid, missing) against the live store.

    Used to preserve the user's selections across a refresh and to report
    styles that vanished when restoring from infotext.
    """
    if not selected:
        return [], []

    known = set(available_styles())
    kept = [name for name in selected if name in known]
    missing = [name for name in selected if name not in known]
    return kept, missing


def apply(positive: str, negative: str, selected: list[str] | None) -> tuple[str, str]:
    """Expand ``selected`` styles into the given prompt pair.

    Styles apply to both the positive and negative components, in the order
    listed, matching core WebUI semantics because it is the core WebUI code
    doing the work.
    """
    if not selected:
        return positive, negative

    store = _store()
    try:
        positive = store.apply_styles_to_prompt(positive, selected)
        negative = store.apply_negative_styles_to_prompt(negative, selected)
    except Exception:
        logger.warning("Model Chain: failed to apply styles %s", selected, exc_info=True)

    return positive, negative
