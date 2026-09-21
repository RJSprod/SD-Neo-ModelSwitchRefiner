"""One overlay at a time, decided in one place, for all seven of them.

LLM Studio has seven surfaces that pop over the workspace, and until now it had
*three* mechanisms for deciding which was open: the shell's two sheets knew
about each other, the chat panel's four screens knew about each other, and the
message action sheet knew about nothing. So opening the character editor left
the workspace chooser underneath it, and tapping a bubble opened an action
sheet over whichever of those happened to be showing. Three half-drawn panels
stacked on one phone screen, and each mechanism individually correct.

This module is the fourth mechanism that replaces all three. :func:`showing`
answers "what is visible" for every surface at once, so exclusivity is a
property of one function rather than a rule seven handlers have to remember.

How it reaches Gradio
---------------------
A Gradio handler can only write the outputs it declares, so a decision about
all seven is only worth making if every opener *can* write all seven. Both
halves are built in one ``gr.Blocks`` but by two modules, and the shell's
sheets exist before the chat panel is built — so the components are registered
here as they are created, and each module asks for the ones it did not build
when it wires its outputs. The registry is per-build and cleared at the start
of one, because a UI reload builds a second set of components and a handler
writing into the first set's would be writing into a page nobody is looking at.

What is deliberately *not* here
-------------------------------
Inline disclosures. The pipeline drawer, the creative drawers, the browse
pickers, the character voice delivery group, the reference panel, the edit row
and the rename row all push content down rather than covering it — they are
part of the layout, two of them can sensibly be open at once, and closing one
because another opened would be a rule nobody asked for. The audit is in the
specification's 15.1 and the line between the two lists is "does it cover
something".

The assistant is exempt both ways: opening a sheet does not close it, and it
does not close a sheet. It is a second window onto the conversation, not
another sheet over it.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

SHELL = ("mode", "model")
"""The shell's own sheets: the workspace chooser and the model/runtime sheet."""

CHAT = ("threads", "character", "persona", "voice")
"""Conversation's screens. The same tuple ``mc_llm_chat_panel.SCREENS`` names."""

ACTIONS = "actions"
"""The per-message action sheet. The seventh, and the one that knew nothing."""

OVERLAYS = SHELL + CHAT + (ACTIONS,)
"""Every pop surface in LLM Studio, in the order this module answers in."""

_components: dict = {}
_states: dict = {}


def reset() -> None:
    """Forget the components of a previous build. Called when one starts."""
    _components.clear()
    _states.clear()


def register_state(owner: str, component) -> None:
    """Record the ``gr.State`` one module keeps its open-surface name in.

    There are two -- the shell's and the chat panel's -- because there were two
    mechanisms, and they are kept rather than merged: each is an output of
    handlers in its own module and merging them would mean every handler in one
    module writing a component built by the other.

    What they are registered for is the stale half. When a surface in the other
    group opens, this group's surfaces are hidden, and a State still naming one
    of them would make the button that opens it close nothing instead -- the
    dead control the chat panel's own comments warn about. So a handler closes
    the other group's State along with its components.
    """
    _states[str(owner)] = component


def register(name: str, component) -> None:
    """Record one surface's component so other modules can write to it."""
    if name not in OVERLAYS:
        logger.debug("Model Chain: %s is not one of the overlay surfaces", name)
        return
    _components[name] = component


def registered(name: str):
    return _components.get(name)


def components(names) -> list:
    """The components for ``names`` that this build actually has.

    Skipping the absent ones is what lets a panel be built on its own -- in a
    test, or in a host where one mode failed to build -- without every handler
    in the other panel losing an output it was counting on.
    """
    return [_components[name] for name in names if name in _components]


def present(names) -> tuple:
    """The subset of ``names`` this build has components for, in order."""
    return tuple(name for name in names if name in _components)


def showing(name: str = "") -> dict:
    """Which of the seven is open. Exactly one, or none.

    The single decision. ``name`` is the surface being opened; everything else
    is closed, including the surfaces the caller has never heard of, which is
    the whole point.
    """
    wanted = name if name in OVERLAYS else ""
    return {surface: (surface == wanted) for surface in OVERLAYS}


def updates(name: str, order) -> list:
    """:func:`showing`, as Gradio visibility updates in ``order``."""
    import gradio as gr

    open_now = showing(name)
    return [gr.update(visible=open_now[surface]) for surface in order if surface in OVERLAYS]


def answer(name: str, own, extra=()) -> list:
    """``[name] + visibilities`` for a handler that owns ``own``.

    The shape every existing caller already returns -- the chosen name first,
    because that is what the ``surface``/``sheet`` State holds and what makes a
    menu button a toggle rather than a control that can only open. ``extra``
    is the surfaces this handler did not build and now closes anyway.
    """
    wanted = name if name in OVERLAYS else ""
    return [wanted] + updates(wanted, tuple(own) + tuple(present(extra)))


def foreign(own, own_state: str = ""):
    """``(components, values)`` for everything a handler closes but did not build.

    Handed straight into an ``outputs`` list and onto the end of a return
    value. The values are all "closed": hidden for a surface, empty for a
    State, because there is only ever one open surface and this handler is
    opening it.
    """
    import gradio as gr

    surfaces = closes(own)
    states = [name for name in sorted(_states) if name != own_state]
    parts = components(surfaces) + [_states[name] for name in states]
    values = [gr.update(visible=False)] * len(surfaces) + [""] * len(states)
    return parts, values


def closes(own) -> tuple:
    """The surfaces a handler owning ``own`` must also be able to write.

    Everything it does not own, in :data:`OVERLAYS` order, limited to what this
    build registered. Used to extend an ``outputs`` list so that the answer
    above has somewhere to land.
    """
    mine = set(own)
    return tuple(name for name in OVERLAYS if name not in mine and name in _components)
