"""Characters, in oobabooga's own format, and the persona replying to them.

A character is a YAML file in ``<install root>/characters`` holding the three
fields oobabooga's character editor writes — ``name``, ``context`` and
``greeting`` — with its picture beside it as ``<name>.png``. That is not a
format chosen for its elegance; it is chosen because a folder of characters is
something people already have, and one copied in from a text-generation-webui
install has to work here without conversion. The settings this application adds
— the sampling a character is talked to with, and a system message that
replaces the default wrapper — are written as extra keys, which oobabooga
ignores when it reads the file back.

Three shapes import: a ``.yaml`` character, a ``.json`` character, and a
``.png`` character card with its JSON in a text chunk (the V2/V3 ``chara`` key,
base64-encoded, as TavernAI and SillyTavern write it). Cards carry fields this
application has no separate box for — personality, scenario, example dialogue —
so importing folds them into the context under headings rather than dropping
them, which is what oobabooga does with the same fields when it builds a prompt.

The persona is the other half: a name and a description for whoever is typing.
It is optional by design. Undefined, the model is answering an unnamed "You" —
which is what a chat with no persona has always been — and defined, the name
and description go into the system prompt so the character can use them.
"""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from pathlib import Path

from prompt_master.core.config import atomic_write_json, read_json
from prompt_master.core.models import RANDOM_SEED

from . import yamlish

# What a character file may be called, and what may be imported as one.
CHARACTER_SUFFIXES = (".yaml", ".yml")
IMPORTABLE_SUFFIXES = CHARACTER_SUFFIXES + (".json", ".png")
AVATAR_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")

# Where the persona lives. JSON beside the other application state rather than
# YAML beside the characters: it is a setting, not a document to be shared.
PERSONA_FILE = "persona.json"

# The card fields, in every spelling the three formats use. First key present
# wins, so the modern name leads and the legacy TavernAI one follows.
_NAME_KEYS = ("name", "char_name", "character")
_CONTEXT_KEYS = ("context", "description", "char_persona", "persona")
_GREETING_KEYS = ("greeting", "first_mes", "char_greeting")
_PERSONALITY_KEYS = ("personality",)
_SCENARIO_KEYS = ("scenario", "world_scenario")
_EXAMPLE_KEYS = ("example_dialogue", "mes_example", "example_dialog")

# Defaults for the settings this application adds. The sampling is the writer's
# own — ``LlamaClient.stream_chat``'s 0.85 / 0.95 — so a character created and
# never adjusted is talked to the way everything else in this app is.
DEFAULT_TEMPERATURE = 0.85
DEFAULT_TOP_P = 0.95
DEFAULT_MAX_REPLY_TOKENS = 512

# Illegal in a Windows file name, and in a name that is about to become one.
_ILLEGAL = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


@dataclass
class Character:
    """One character, and how this application talks to it."""

    name: str
    # oobabooga's "Context": who the character is, everything the model is told
    # about them before the first line of dialogue.
    context: str = ""
    greeting: str = ""
    # Ours, and unknown to oobabooga, which ignores keys it does not read.
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    max_reply_tokens: int = DEFAULT_MAX_REPLY_TOKENS
    # RANDOM_SEED — the same -1 prompt mode uses — draws a fresh seed per reply,
    # which is what makes a regenerate come back different.
    seed: int = RANDOM_SEED
    # Replaces the built system prompt entirely when set. The escape hatch for
    # a character whose framing this application's wrapper gets wrong.
    system: str = ""

    def to_mapping(self) -> dict[str, object]:
        """The file, in the order it is written."""
        return {"name": self.name, "context": self.context, "greeting": self.greeting,
                "temperature": self.temperature, "top_p": self.top_p,
                "max_reply_tokens": self.max_reply_tokens, "seed": self.seed,
                "system": self.system}

    @classmethod
    def from_mapping(cls, data: dict) -> "Character":
        """A character from any of the three formats, or from our own file."""
        values = {str(key).casefold(): value for key, value in data.items()}
        context = _first(values, _CONTEXT_KEYS)
        # A card's other prose has no box of its own here, so it goes into the
        # context under a heading rather than being dropped on import.
        for label, keys in (("Personality", _PERSONALITY_KEYS),
                            ("Scenario", _SCENARIO_KEYS),
                            ("Example dialogue", _EXAMPLE_KEYS)):
            extra = _first(values, keys)
            if extra and extra not in context:
                context = f"{context}\n\n{label}: {extra}".strip()
        return cls(
            name=_first(values, _NAME_KEYS).strip() or "Unnamed",
            context=context,
            greeting=_first(values, _GREETING_KEYS),
            temperature=_number(values.get("temperature"), DEFAULT_TEMPERATURE),
            top_p=_number(values.get("top_p"), DEFAULT_TOP_P),
            max_reply_tokens=int(_number(values.get("max_reply_tokens"), DEFAULT_MAX_REPLY_TOKENS)),
            seed=int(_number(values.get("seed"), RANDOM_SEED)),
            system=_first(values, ("system", "system_message")),
        )


@dataclass
class Persona:
    """Whoever is typing — optional, and empty until it is filled in."""

    name: str = ""
    description: str = ""

    @property
    def display(self) -> str:
        """What the character calls you. Unnamed, that is "You"."""
        return self.name.strip() or "You"

    @property
    def defined(self) -> bool:
        return bool(self.name.strip() or self.description.strip())


class CharacterStore:
    """The characters folder, and every operation the UI performs on it."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)

    @classmethod
    def from_paths(cls, paths) -> "CharacterStore":
        return cls(paths.characters)

    # ── reading ──────────────────────────────────────────────────────────────

    def names(self) -> list[str]:
        """Every character, by the name inside the file rather than its path."""
        found = []
        for path in sorted(self.directory.glob("*")):
            if path.suffix.casefold() not in CHARACTER_SUFFIXES:
                continue
            try:
                found.append(self._read(path).name)
            except (OSError, ValueError):
                continue          # a broken file must not empty the list
        return sorted(found, key=str.casefold)

    def load(self, name: str) -> Character:
        path = self.resolve(name)
        if path is None:
            raise FileNotFoundError(f"No character named {name}")
        return self._read(path)

    def path_for(self, name: str) -> Path:
        """Where a character of this name would be written."""
        return self.directory / f"{safe_stem(name)}.yaml"

    def resolve(self, name: str) -> Path | None:
        """The file holding this character, whatever it happens to be called.

        A file's stem and the name inside it are not the same thing, and in a
        folder copied from oobabooga they frequently disagree — ``Chiharu.yaml``
        holding ``name: Chiharu Yamada`` is the format's own example file. The
        name in the file is the character's identity everywhere in this
        application, so finding one means looking inside rather than guessing a
        path from it.
        """
        direct = self.path_for(name)
        if direct.is_file():
            return direct
        wanted = name.strip().casefold()
        for path in sorted(self.directory.glob("*")):
            if path.suffix.casefold() not in CHARACTER_SUFFIXES:
                continue
            try:
                if self._read(path).name.strip().casefold() == wanted:
                    return path
            except (OSError, ValueError):
                continue
        return None

    def exists(self, name: str) -> bool:
        return self.resolve(name) is not None

    def avatar_for(self, name: str) -> Path | None:
        """The picture beside the character file, whatever it is called."""
        path = self.resolve(name)
        return self._avatar_beside(path.stem if path is not None else safe_stem(name))

    def _avatar_beside(self, stem: str) -> Path | None:
        for suffix in AVATAR_SUFFIXES:
            candidate = self.directory / f"{stem}{suffix}"
            if candidate.is_file():
                return candidate
        return None

    def _read(self, path: Path) -> Character:
        character = Character.from_mapping(yamlish.loads(path.read_text(encoding="utf-8")))
        # The file name is the fallback identity: a character file written by
        # hand may have no name field at all.
        if character.name == "Unnamed":
            character.name = path.stem
        return character

    # ── writing ──────────────────────────────────────────────────────────────

    def save(self, character: Character, previous_name: str | None = None) -> Path:
        """Write ``character``, in place if it is already on disk.

        In place matters: a character read out of ``Chiharu.yaml`` and saved
        again has to go back into ``Chiharu.yaml``, not into a second file
        named after the name inside it.
        """
        if not character.name.strip():
            raise ValueError("A character needs a name")
        self.directory.mkdir(parents=True, exist_ok=True)
        renaming = bool(previous_name) and previous_name != character.name
        old = self.resolve(previous_name) if renaming else None
        path = self.path_for(character.name) if renaming \
            else (self.resolve(previous_name or character.name) or self.path_for(character.name))
        path.write_text(yamlish.dumps(character.to_mapping()), encoding="utf-8")
        if old is not None and old != path:
            # A rename moves the picture with the file, or the character loses
            # its face the moment it is renamed.
            avatar = self._avatar_beside(old.stem)
            if avatar is not None:
                avatar.replace(self.directory / f"{path.stem}{avatar.suffix}")
            old.unlink(missing_ok=True)
        return path

    def delete(self, name: str) -> None:
        path = self.resolve(name)
        avatar = self.avatar_for(name)
        if path is not None:
            path.unlink(missing_ok=True)
        if avatar is not None:
            avatar.unlink(missing_ok=True)

    def set_avatar(self, name: str, source: Path | None) -> Path | None:
        """Put ``source`` beside the character as its picture, or remove one.

        Saved as PNG under the character's own stem, which is where oobabooga
        looks for it, and bounded in size because a character list showing
        thirty faces should not be reading thirty full-resolution photographs.
        """
        existing = self.avatar_for(name)
        if existing is not None:
            existing.unlink(missing_ok=True)
        if source is None:
            return None
        target = self.directory / f"{safe_stem(name)}.png"
        self.directory.mkdir(parents=True, exist_ok=True)
        write_avatar(source, target)
        return target

    # ── importing ────────────────────────────────────────────────────────────

    def import_file(self, source: Path) -> Character:
        """Read a character out of a ``.yaml``, ``.json`` or ``.png`` card."""
        source = Path(source)
        suffix = source.suffix.casefold()
        if suffix not in IMPORTABLE_SUFFIXES:
            raise ValueError("Import a .yaml, .json or .png character card")
        if suffix == ".png":
            data = read_card(source)
        elif suffix == ".json":
            data = _unwrap(json.loads(source.read_text(encoding="utf-8")))
        else:
            data = yamlish.loads(source.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not data:
            raise ValueError(f"{source.name} holds no character")
        character = Character.from_mapping(data)
        if character.name == "Unnamed":
            character.name = source.stem
        character.name = self.unique_name(character.name)
        self.save(character)
        if suffix == ".png":
            # The card is the picture: the JSON was carried inside the image
            # that is meant to be the character's face.
            self.set_avatar(character.name, source)
        return character

    def unique_name(self, name: str) -> str:
        """``name``, or the first numbered variant of it that is free."""
        candidate, index = name, 2
        while self.exists(candidate):
            candidate, index = f"{name} ({index})", index + 1
        return candidate


def read_card(path: Path) -> dict:
    """The character JSON embedded in a PNG card.

    V2 and V3 cards store base64-encoded JSON in a ``tEXt`` chunk — ``chara``,
    or ``ccv3`` for the newer spec — which Pillow hands back in ``Image.info``.
    """
    from PIL import Image

    with Image.open(path) as image:
        info = {str(key).casefold(): value for key, value in (image.info or {}).items()}
    raw = info.get("ccv3") or info.get("chara")
    if not isinstance(raw, str):
        raise ValueError(f"{path.name} is an image, not a character card")
    try:
        decoded = base64.b64decode(raw, validate=True).decode("utf-8")
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError(f"{path.name} carries a character card this cannot read") from exc
    return _unwrap(json.loads(decoded))


def _unwrap(data: object) -> dict:
    """A V2/V3 card nests everything under ``data``; a V1 card is flat."""
    if isinstance(data, dict) and isinstance(data.get("data"), dict):
        return data["data"]
    return data if isinstance(data, dict) else {}


PERSONA_AVATAR = "persona.png"
"""Your own picture, beside the persona it belongs to.

Not in the characters folder: that folder is a set of character files and their
pictures, in the layout oobabooga reads, and a face in it with no character
behind it is a file that list has to know to skip.
"""


def write_avatar(source, target: Path) -> None:
    """One picture, sized down and written as PNG.

    ``source`` is a path or an already-decoded picture, because the UI has both:
    an imported card is a file, and the picker in the panel hands over the image
    itself -- Gradio's filepath preprocess is unusable in this host, which is
    why nothing in this application asks it for one.

    Bounded in size because a transcript showing a face beside every message
    should not be reading a full-resolution photograph to do it.
    """
    from PIL import Image, ImageOps

    opened = source if hasattr(source, "convert") else Image.open(source)
    try:
        image = ImageOps.exif_transpose(opened).convert("RGBA")
        image.thumbnail((512, 512), Image.Resampling.LANCZOS)
        target.parent.mkdir(parents=True, exist_ok=True)
        image.save(target, "PNG")
    finally:
        if opened is not source:
            opened.close()


def persona_avatar(paths) -> Path | None:
    """Your picture, or ``None`` when you have not chosen one."""
    found = Path(paths.data) / PERSONA_AVATAR
    return found if found.is_file() else None


def set_persona_avatar(paths, source) -> Path | None:
    """Give yourself a face, or take the one you had away."""
    target = Path(paths.data) / PERSONA_AVATAR
    if source is None:
        target.unlink(missing_ok=True)
        return None
    write_avatar(source, target)
    return target


def safe_stem(name: str) -> str:
    """``name`` as a file name Windows will accept."""
    stem = _ILLEGAL.sub("_", name).strip().rstrip(".")
    return stem or "character"


def load_persona(paths) -> Persona:
    try:
        data = read_json(paths.data / PERSONA_FILE)
    except (OSError, ValueError):
        return Persona()
    return Persona(name=str(data.get("name", "")), description=str(data.get("description", "")))


def save_persona(paths, persona: Persona) -> None:
    atomic_write_json(paths.data / PERSONA_FILE,
                      {"name": persona.name, "description": persona.description})


def _first(values: dict, keys: tuple[str, ...]) -> str:
    for key in keys:
        value = values.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list) and value:
            # Some cards write example dialogue as a list of lines.
            return "\n".join(str(item) for item in value).strip()
    return ""


def _number(value: object, default: float) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
