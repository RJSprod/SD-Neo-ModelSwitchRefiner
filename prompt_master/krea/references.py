"""The ordered references, kept as structure rather than flattened into prose.

This module exists because of one boundary the design intent draws twice, in
§4 and again in §12: **the user's visible order is the semantic source of
truth.** The first slot on screen is Image 1 and stays Image 1 -- not the first
file that finished uploading, not the first tensor a backend wants, not
whichever caption sounds like a portrait.

So a reference is a record with its own index rather than a position in a list
that somebody may sort later, and a finished run hands back
:class:`KreaPromptResult` -- the prompt *and* the references it was written
about -- rather than one string. A Krea edit implementation may well need its
references in a different order than the person supplying them saw (a
subject-first UI over a LoRA trained scene-first, for instance); that
reordering belongs to a backend adapter, and an adapter that is handed this
structure can do it without ever redefining what "Image 1" meant in the prompt
it was handed alongside.

Nothing here holds image bytes for longer than a run. ``data_url`` is the
encoded picture the vision model is shown and is never persisted (§11, §14);
``path`` and ``caption`` are what history keeps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Reference:
    """One attached image, numbered the way the user sees it.

    ``ui_index`` is 1-based because every user-facing string derived from it --
    "Image 1", "Describing image 2 of 4" -- is 1-based, and a model that stores
    0 and adds 1 at each of five call sites is a model that will one day add it
    at four of them.
    """

    ui_index: int
    path: str = ""
    """Where the picture came from, when it came from a file at all.

    Empty when the UI handed over a decoded picture instead of a path, which is
    what the reference slots now ask for -- see :attr:`picture`. Kept because it
    is where :attr:`name` comes from, and a file that does have a name should
    still be called by it in a history."""
    picture: object = None
    """The picked image itself, when the slot handed one over rather than a path.

    Untyped on purpose: this package is imported by the prompt engine and by the
    tests, and neither should have to have Pillow installed to describe a
    reference. What consumes it -- ``mc_llm_krea_panel._encoded`` -- already
    depends on Pillow through the vendored preprocessor."""
    data_url: str = ""
    caption: str = ""
    semantic_role: str = ""
    """What the user said this reference is for, when they said it explicitly.

    Left empty in version 1 and never inferred: §13 rules out automatic role
    classification, and a guessed role written into the prompt is exactly the
    semantic loss this whole package is arranged to prevent. The field is here
    because a future backend needs somewhere to put a role the *user* declared.
    """

    @property
    def source(self):
        """What to read this reference's pixels from: the picture, or the path.

        One property rather than the same conditional at every call site. Which
        of the two a slot hands over is a fact about the UI component and about
        nothing else -- see ``mc_llm_krea_panel.build`` -- and no caller of this
        package should have to know it.
        """
        return self.picture if self.picture is not None else self.path

    @property
    def name(self) -> str:
        """The file's name, for history and for status lines. Never the path.

        A full path is somebody's home directory, their username and often the
        name of the project they are working on, and none of that belongs in a
        history file or a log line (§14).
        """
        return Path(self.path).name if self.path else ""

    @property
    def label(self) -> str:
        return f"Image {self.ui_index}"


@dataclass
class KreaPromptResult:
    """What a finished Krea run produced, with its inputs still separable.

    §12: "Do not reduce the entire task to one prompt string if future backend
    integration is expected." The prompt is what goes in the box; the
    references are what it was written about, still numbered, still carrying
    their captions. A backend adapter takes this, not the string.
    """

    prompt: str = ""
    references: list = field(default_factory=list)
    seed: int = 0

    @property
    def names(self) -> list[str]:
        return [reference.name for reference in self.references]

    @property
    def captions(self) -> list[str]:
        return [reference.caption for reference in self.references]
