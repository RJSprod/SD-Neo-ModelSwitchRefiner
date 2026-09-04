"""The three LLM roles, each configurable apart: Neutralizer, Creative and Spatial.

The design intent asks for first-class configurations where there was one, and
asks for the upgrade to be invisible: an installation that has never heard of
roles must keep behaving exactly as it did, with one llama-server and the same
handoff from one pass to the next.

This module gets that by *inheritance* rather than by copying.

The state file's existing top-level configuration stays exactly where it is and
keeps meaning exactly what it meant: the installation's LLM. It is what Prompt
Studio, Conversation and MiniMax use, and it is what a role uses when the role
has nothing of its own to say. A role override is a sparse mapping written only
where the user has actually chosen something different.

That is worth the paragraph, because the obvious alternative -- migrate by
copying the old configuration into ``roles.creative`` and ``roles.spatial`` --
is what section 15 of the intent describes and is worse in one specific way.
Copies go stale. Somebody who changes the model in Setup after such a migration
changes the shared configuration and neither role, and the change appears to do
nothing at all. With inheritance there is nothing to go stale: a role that has
not been split follows the installation, so the first upgraded run resolves every
role to the same identity, coalesces them onto one runtime, and reproduces the
old behaviour without a migration step having to run correctly to get there.

Splitting a role is then the only event that creates a second runtime, which is
section 15's actual requirement -- "only a user action creates the new dual-
runtime arrangement" -- reached by a road that cannot half-fail.

Three roles, and a rule about counting them
-------------------------------------------
The Pose Neutralizer arrived third, and arrived under the same inheritance: an
installation upgraded across it has no ``roles.neutralizer`` entry, so the
Neutralize Prompt stage runs on whatever the installation runs, on the same
server the writer is about to use. What the third role changed is not this
module but every helper *around* it that had been written for two -- "both
roles", "the other role", a sentence that named Creative and Spatial by hand.
Those are generalised to :data:`ROLES` now, and the rule for anything added
later is the one the Neutralizer was held to: nothing may reason about a
role by naming its partner. :func:`describe` and :func:`others` exist so that
nothing has to.
"""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

NEUTRALIZER = "neutralizer"
CREATIVE = "creative"
SPATIAL = "spatial"

ROLES = (NEUTRALIZER, CREATIVE, SPATIAL)
"""The roles that may be configured apart. Order is display order.

It is also pipeline order -- the Neutralizer runs before the writer, which runs
before the Composer -- so a menu, a register line and the Image Pipeline all
list the same three things the same way round.
"""

LABELS = {NEUTRALIZER: "Pose Neutralizer", CREATIVE: "Creative Writer",
          SPATIAL: "Spatial Composer"}
"""What each role is called in a status line, a register line and a menu.

The Neutralizer's label names what the *model* does; the txt2img stage that
asks it is called "Neutralize Prompt", which names what the *pipeline* does.
The two differ on purpose, the way "Creative Writer" and "Creative" do.
"""

SHORT = {NEUTRALIZER: "Neutralizer", CREATIVE: "Creative", SPATIAL: "Spatial"}
"""One word per role, for the front of a console line.

Spelled out rather than derived from :data:`LABELS`, because the first word of
"Pose Neutralizer" is the wrong one: a line prefixed ``[Pose]`` names nothing
anybody configured.
"""

SHARED = ""
"""The installation's own configuration: every mode that is not one of the roles.

Spelled as the empty string so that ``role or SHARED`` is the whole of "this
caller did not ask for a role", and so that a stored key can never collide with
it. Prompt Studio, Conversation, MiniMax and LLM Studio's own model loading all
run here, unchanged and unaware that roles exist.
"""

SECTION = "roles"
"""Where role overrides live, in both files.

One name rather than two, because the two files are separate documents and a
section called something different in each would be two things to remember for
no benefit. Which *fields* belong in which file is the distinction that matters,
and that is :data:`STATE_FIELDS` and :data:`PREFS_FIELDS`.
"""

# What a role may override, split by which file holds the shared value. Two
# tuples rather than one because the two files are read by different code with
# different precedence rules, and a key written into the wrong one is a setting
# that appears to save and then does nothing.
STATE_FIELDS = (
    "runtime", "runtime_id", "model", "mmproj", "mode", "gpu_index", "gpu_uuid",
    "gpu_name", "gpu_device", "gpu_device_name", "gpu_layers", "quantization",
    # Mixed Minimum shares Aggressive's ``mode``, so this is the only field that
    # distinguishes them. A role that did not carry it would inherit the
    # installation's placement and silently start from every layer.
    "expert_minimum",
    # Spelled exactly as mc_llm_managed_models writes them, because the managed
    # selection is read straight out of the layered state by that module's own
    # ``selection()`` -- a role that renamed these would be a role whose managed
    # backbone silently reverted to the installation's.
    "source", "managed_model_id", "managed_profile", "managed_profile_version",
)
"""Hardware and model choices, which live beside the installation's own."""

PREFS_FIELDS = (
    "context_mode", "context_size", "context_buffer_gb", "kv_type_k", "kv_type_v",
    # Spelled exactly as mc_llm_accel writes them, for the same reason the
    # managed-selection keys above are spelled as mc_llm_managed_models writes
    # them: that module reads its own settings straight out of the layered
    # preferences, and a role that renamed these would be a role whose
    # performance mode silently reverted to the installation's.
    "llm_performance_preset", "llm_accelerator", "llm_memory_priority",
)
"""Context, cache and performance settings, which live in the preferences file.

The performance pair is here rather than beside the hardware fields because a
role's acceleration is a question about how much room *that role's* card has,
and a machine with a 3090 and a 5090 in it can reasonably answer it twice."""

FIELDS = (*STATE_FIELDS, *PREFS_FIELDS)

_lock = threading.RLock()


def named(role: str) -> str:
    """``role`` as one of :data:`ROLES`, or :data:`SHARED` for anything else.

    Total on purpose. A role name arriving from a saved panel value, a URL or an
    older build is a reason to fall back to the installation's own configuration
    -- which is always a valid answer -- and never a reason to raise into a
    generation.
    """
    text = str(role or "").strip().casefold()
    return text if text in ROLES else SHARED


def label(role: str) -> str:
    """What to call ``role`` in a sentence somebody reads."""
    return LABELS.get(named(role), "LLM")


def prefix(role: str) -> str:
    """``"[Creative] "`` or ``""``, for the front of a console line.

    Empty for the shared configuration so that every line this extension has
    always written keeps the shape it has always had. Only a line that is really
    about one role announces one.
    """
    chosen = named(role)
    return f"[{SHORT[chosen]}] " if chosen else ""


def others(role: str) -> tuple:
    """Every role that is not ``role``, in display order.

    "The other role" stopped being a phrase with a referent when the third one
    arrived. Anything that used to reason about a role's partner asks this
    instead and gets however many there are.
    """
    chosen = named(role)
    return tuple(found for found in ROLES if found != chosen)


def describe(roles) -> str:
    """The roles named in a sentence: ``"Creative Writer and Spatial Composer"``.

    One name on its own, two joined by "and", three or more with commas before
    the last -- the register line, the Setup notice and the log all read these,
    and a sentence that said "A and B and C" would read as a list nobody
    finished. Unknown names are dropped and duplicates collapsed, so a caller
    can hand this whatever a runtime says it serves.
    """
    names: list[str] = []
    for found in roles or ():
        chosen = named(found)
        if chosen and LABELS[chosen] not in names:
            names.append(LABELS[chosen])
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #


def overrides(role: str, state: dict | None = None, prefs: dict | None = None,
              keys=None) -> dict:
    """What ``role`` has been configured to do differently, if anything.

    An empty mapping is the answer for the shared configuration and for a role
    that has never been split, and those two are deliberately the same answer:
    both mean "use the installation's settings", and a caller that had to tell
    them apart would be a caller that could get it wrong.
    """
    chosen = named(role)
    if not chosen:
        return {}
    wanted = None if keys is None else set(keys)
    stored: dict = {}
    for source, group in ((state, STATE_FIELDS), (prefs, PREFS_FIELDS)):
        if source is None:
            continue
        section = source.get(SECTION)
        if not isinstance(section, dict):
            continue
        entry = section.get(chosen)
        if not isinstance(entry, dict):
            continue
        stored.update({key: entry[key] for key in group
                       if key in entry and (wanted is None or key in wanted)})
    return stored


def split(role: str, state: dict | None = None, prefs: dict | None = None) -> bool:
    """Whether ``role`` has been given a configuration of its own.

    The question the UI asks to decide between "following the installation" and
    "configured separately", and the question :mod:`mc_llm_runtime` asks to
    decide whether resolving the role can be skipped entirely.
    """
    return bool(overrides(role, state, prefs))


def layered(role: str, shared: dict, state: dict | None = None,
            prefs: dict | None = None, keys=None) -> dict:
    """``shared``, with ``role``'s overrides on top.

    The one place inheritance happens. ``shared`` is whatever the caller has
    already resolved for the installation, so this stays ignorant of where any
    of it came from -- which is what lets the state file and the preferences
    file keep their own precedence rules without this module learning them.

    ``keys`` narrows it to one file's worth of fields. Layering a state
    document with the preferences fields would put a context size into the
    state file's namespace, where the next writer of that file would faithfully
    persist a key that belongs somewhere else.
    """
    merged = dict(shared)
    merged.update(overrides(role, state, prefs, keys))
    return merged


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #


def _section(document: dict, key: str) -> dict:
    section = document.get(key)
    if not isinstance(section, dict):
        section = {}
        document[key] = section
    return section


def apply(document: dict, role: str, values: dict, *, keys) -> dict:
    """Record ``values`` as ``role``'s overrides inside ``document``.

    Mutates and returns ``document`` so the caller can write it back with
    whatever atomic-write helper owns that file; this module reads and writes no
    files of its own, because the two it would have to write are owned by
    :mod:`mc_llm_setup` and :mod:`mc_llm_state` and having three writers for two
    files is how a setting gets lost between them.

    A value of ``None`` clears that one key, and a role left with no keys at all
    is removed rather than left as an empty mapping -- so "reset this role to
    follow the installation" is expressible, and leaves a file that looks the
    way it did before the role was ever split.
    """
    chosen = named(role)
    if not chosen:
        return document
    with _lock:
        section = _section(document, SECTION)
        entry = dict(section.get(chosen) or {})
        for field in keys:
            if field not in values:
                continue
            if values[field] is None:
                entry.pop(field, None)
            else:
                entry[field] = values[field]
        if entry:
            section[chosen] = entry
        else:
            section.pop(chosen, None)
        if not section:
            document.pop(SECTION, None)
    return document


def clear(document: dict, role: str) -> dict:
    """Drop every override ``role`` has in ``document``.

    "Follow the installation again", which is the state every role starts in and
    the one an upgraded installation is already in.
    """
    chosen = named(role)
    if not chosen:
        return document
    with _lock:
        section = document.get(SECTION)
        if isinstance(section, dict):
            section.pop(chosen, None)
            if not section:
                document.pop(SECTION, None)
    return document
