"""Loaded, modified, not saved -- said the same way about three different things.

Creative has profiles, Spatial has layouts, Stage 2 has presets. Three features,
three storage formats, three sets of handlers, and -- before this file -- three
different ways of answering the one question a user actually asks, which is
"is what I am looking at what will happen?"

The answer this module standardises
-----------------------------------
    Loaded: Portrait polish
    Loaded: Portrait polish - Modified - not saved
    (nothing at all, when no named thing has been loaded)

The critical rule, which is section 8.3 of the design intent
------------------------------------------------------------
**Modified-but-unsaved settings are the active settings.** They are what the
next Generate uses. "Not saved" is a statement about a file on disk, never about
the generation:

    not saved  ==  the named profile still holds its old values
    not saved  !=  your changes will be ignored

That distinction is the entire reason this file exists as shared code rather
than as three similar sentences. The two readings are one word apart in English
and opposite in consequence, and the failure mode of getting it wrong is a user
who presses Save before every generation because they are not sure -- or worse,
one who does not, and believes the image came from settings it did not.

Nothing here writes anything
----------------------------
No preference, no file, no component value. It formats a string and compares two
snapshots. The features keep their own storage and their own handlers; this is
the vocabulary they answer in.
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

MODIFIED = "Modified · not saved"
"""Said in full every time. An asterisk or a dot would need a legend."""


def describe(name: str, modified: bool, *, active: bool = True) -> str:
    """The one status line, for any of the three kinds of named configuration.

    ``active`` is for a stage that is switched off: its settings are still
    loaded and still modified, and will still be what runs the moment it is
    switched on, so the identity is worth keeping on screen -- but promising
    that they are in use would be false.
    """
    name = str(name or "").strip()
    if not name:
        return ""

    line = f"**Loaded:** {name}"
    if modified:
        line += f" · *{MODIFIED}*"
    if not active:
        line += " · *stage is off*"
    return line


def explain(modified: bool) -> str:
    """The sentence that stops "not saved" from being read as "not applied".

    Shown beside the status rather than in documentation, because the moment a
    user needs it is the moment they are looking at the word "unsaved" and
    deciding whether to trust the screen.
    """
    if not modified:
        return ""
    return ("*These edited settings are the ones the next Generate will use. "
            "Saving only updates the stored copy under this name.*")


def snapshot(values) -> str:
    """A comparable fingerprint of whatever a named configuration covers.

    JSON rather than a tuple or a hash: it survives the round trip through
    Gradio's own JSON transport unchanged, so a value that arrives as a list
    where it left as a tuple compares equal -- which is the difference between
    a dirty flag that means something and one that is always on.

    Anything that will not serialise falls back to ``repr``. A configuration
    holding an unserialisable value is not a reason to stop reporting on the
    rest of it.
    """
    try:
        return json.dumps(_plain(values), sort_keys=True, default=repr)
    except Exception:
        logger.debug("Model Chain: could not fingerprint a configuration", exc_info=True)
        return repr(values)


def _plain(value):
    """Lists and dicts, recursively, with tuples and sets flattened to lists."""
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def changed(current, baseline) -> bool:
    """Whether ``current`` differs from the snapshot it was loaded from.

    An empty baseline means nothing named has been loaded, so there is nothing
    to have diverged from and the answer is no -- not "everything is modified",
    which is what a naive comparison against a blank string would say about a
    freshly opened tab.
    """
    if not baseline:
        return False
    return snapshot(current) != str(baseline)
