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
    # Which voice reads this character's replies aloud, as the stable id
    # ``mc_voice_registry`` mints — ``official:af_nicole`` or ``clone:<uuid>``,
    # never a speaker number, for the reason that module gives at length: a
    # number is an address in a block of floats and moves when a bank is
    # rebuilt. Empty means "whatever the default voice is", which is what every
    # character written before voices were per-character has.
    voice: str = ""
    # How that voice delivers it. ``None`` is not a missing value here, it is
    # the value: it means "follow the default voice's setting", so a character
    # nobody has configured tracks Settings → Voice Chat instead of freezing
    # today's numbers into its own file. See mc_voice_profile.resolve.
    voice_speed: float | None = None
    voice_pitch: float | None = None
    voice_gain: float | None = None
    voice_pause: float | None = None
    # Sopro's half of the same idea, and flat for the same reason the four
    # Kokoro fields are: this file is written by :mod:`yamlish`, which is a
    # deliberately small scalar-only writer shared with oobabooga's format, and
    # a nested mapping in it would be a mapping nothing could read back.
    #
    # Voice Chat has more than one text-to-speech engine now, and a character
    # may have a voice on each -- but the engine itself is *global*, never a
    # character property. So these are not "which engine does Ada use"; they are
    # "what has Ada been given on Sopro", so that switching the global engine
    # restores the right one instead of erasing the other.
    #
    # Absence is the ordinary state and is not a missing value: a character with
    # nothing here follows Sopro's current defaults, which is what every
    # character written before this existed does. See
    # ``mc_voice_engines.character_voice``.
    sopro_voice: str = ""
    sopro_speed: float | None = None
    sopro_pitch: float | None = None
    sopro_gain: float | None = None
    sopro_pause: float | None = None
    sopro_temperature: float | None = None
    sopro_top_p: float | None = None
    sopro_top_k: float | None = None
    sopro_language: str = ""
    # Pocket's, on exactly the same rule and for exactly the same reason. Flat
    # scalars because that is what :mod:`yamlish` can round-trip, and absence
    # rather than a written default because a character with nothing here
    # follows PocketTTS's current defaults (I-PKT-4).
    #
    # What is deliberately *not* here is the compute half. Precision, sampler
    # steps and the model choice are engine-global (I-PKT-23): a character that
    # carried them would be a character whose turn to speak silently restarted
    # the Pocket worker, which is a setting pretending to be a personality
    # trait (section 13, section 49.11).
    pocket_voice: str = ""
    pocket_speed: float | None = None
    pocket_pitch: float | None = None
    pocket_gain: float | None = None
    pocket_pause: float | None = None
    pocket_temperature: float | None = None

    @property
    def voice_profile(self) -> dict[str, object]:
        """This character's Kokoro delivery overrides, in the profile spelling.

        A dictionary rather than four attributes because that is what crosses
        into the voice modules, and because ``None`` has to survive the trip:
        the whole point of the four fields is which of them are unset.

        Kokoro's specifically. :attr:`voice_profiles` is the engine-aware view,
        and ``mc_voice_engines.character_profile`` is what reads the right one
        for the engine that is selected.
        """
        return {"speed": self.voice_speed, "pitch": self.voice_pitch,
                "gain": self.voice_gain, "pause": self.voice_pause}

    @property
    def voices(self) -> dict[str, str]:
        """The engine-aware view of which voice this character has, per engine.

        Derived rather than stored, so there is exactly one place a voice lives
        and no second copy to keep in step. Only engines this character has
        actually been given something on appear, because absence is what
        inheritance is made of (I-4).
        """
        found = {}
        if self.voice:
            found["kokoro"] = self.voice
        if self.sopro_voice:
            found["sopro"] = self.sopro_voice
        if self.pocket_voice:
            found["pocket"] = self.pocket_voice
        return found

    @property
    def voice_profiles(self) -> dict[str, dict]:
        """The engine-aware view of this character's delivery overrides.

        An engine whose fields are all unset is absent rather than present and
        empty -- the difference between "follows the default" and "has been
        configured to nothing", which is the difference the whole inheritance
        model rests on.
        """
        found = {}
        kokoro = self.voice_profile
        if any(value is not None for value in kokoro.values()):
            found["kokoro"] = kokoro
        sopro = {"speed": self.sopro_speed, "pitch": self.sopro_pitch,
                 "gain": self.sopro_gain, "pause": self.sopro_pause,
                 "temperature": self.sopro_temperature, "top_p": self.sopro_top_p,
                 "top_k": self.sopro_top_k,
                 "language": self.sopro_language or None}
        if any(value is not None for value in sopro.values()):
            found["sopro"] = sopro
        pocket = {"speed": self.pocket_speed, "pitch": self.pocket_pitch,
                  "gain": self.pocket_gain, "pause": self.pocket_pause,
                  "temperature": self.pocket_temperature}
        if any(value is not None for value in pocket.values()):
            found["pocket"] = pocket
        return found

    @staticmethod
    def voice_fields(engine: str, voice_id: str, profile: dict) -> dict:
        """One engine's voice and delivery as this dataclass's own field names.

        The translation between the logical engine-aware shape
        ``mc_voice_engines`` speaks and the flat fields this file stores. Here
        rather than in the panel because it is a fact about the *format*, and a
        second engine added later should have to touch this function and no
        caller.
        """
        offered = dict(profile or {})
        wanted = str(engine or "kokoro")
        if wanted == "kokoro":
            return {"voice": str(voice_id or "").strip(),
                    "voice_speed": offered.get("speed"),
                    "voice_pitch": offered.get("pitch"),
                    "voice_gain": offered.get("gain"),
                    "voice_pause": offered.get("pause")}
        if wanted == "sopro":
            found = {"sopro_voice": str(voice_id or "").strip(),
                     "sopro_language": str(offered.get("language") or "")}
            for name in ("speed", "pitch", "gain", "pause", "temperature", "top_p",
                         "top_k"):
                found[f"sopro_{name}"] = offered.get(name)
            return found
        if wanted == "pocket":
            found = {"pocket_voice": str(voice_id or "").strip()}
            for name in ("speed", "pitch", "gain", "pause", "temperature"):
                found[f"pocket_{name}"] = offered.get(name)
            return found
        # Named rather than defaulted. This used to end in a Kokoro ``return``
        # that any unrecognised engine fell into, which was harmless while
        # "not Sopro" could only mean Kokoro and became a silent data loss the
        # moment a third engine existed: a Pocket save landing in the Kokoro
        # branch would write a Pocket voice id into ``voice`` and quietly
        # replace the character's Kokoro voice (section 13).
        raise ValueError(f"{engine!r} is not a text-to-speech engine this build stores "
                         f"character voice fields for.")

    def to_mapping(self) -> dict[str, object]:
        """The file, in the order it is written.

        The voice keys are written only when they hold something. oobabooga
        ignores keys it does not read either way, but a character with four
        ``null``s in it reads as a character that has been configured, and the
        difference between "unset" and "set to the default" is the whole of how
        inheritance works here.
        """
        found: dict[str, object] = {
            "name": self.name, "context": self.context, "greeting": self.greeting,
            "temperature": self.temperature, "top_p": self.top_p,
            "max_reply_tokens": self.max_reply_tokens, "seed": self.seed,
            "system": self.system}
        if self.voice:
            found["voice"] = self.voice
        for key, value in (("voice_speed", self.voice_speed),
                           ("voice_pitch", self.voice_pitch),
                           ("voice_gain", self.voice_gain),
                           ("voice_pause", self.voice_pause)):
            if value is not None:
                found[key] = value
        # Sopro's, on the same rule: written only when they hold something, so
        # a character nobody has configured for Sopro reads as one that has not
        # been rather than as one configured to today's defaults.
        if self.sopro_voice:
            found["sopro_voice"] = self.sopro_voice
        if self.sopro_language:
            found["sopro_language"] = self.sopro_language
        for key, value in (("sopro_speed", self.sopro_speed),
                           ("sopro_pitch", self.sopro_pitch),
                           ("sopro_gain", self.sopro_gain),
                           ("sopro_pause", self.sopro_pause),
                           ("sopro_temperature", self.sopro_temperature),
                           ("sopro_top_p", self.sopro_top_p),
                           ("sopro_top_k", self.sopro_top_k)):
            if value is not None:
                found[key] = value
        # Pocket's, on the same rule again. A character edited on Kokoro keeps
        # its Pocket keys because they are written from this object's own
        # fields, and this object was loaded with them (I-PKT-3, section 44).
        if self.pocket_voice:
            found["pocket_voice"] = self.pocket_voice
        for key, value in (("pocket_speed", self.pocket_speed),
                           ("pocket_pitch", self.pocket_pitch),
                           ("pocket_gain", self.pocket_gain),
                           ("pocket_pause", self.pocket_pause),
                           ("pocket_temperature", self.pocket_temperature)):
            if value is not None:
                found[key] = value
        return found

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
            voice=_first(values, ("voice", "voice_id")),
            voice_speed=_optional_number(values.get("voice_speed")),
            voice_pitch=_optional_number(values.get("voice_pitch")),
            voice_gain=_optional_number(values.get("voice_gain")),
            voice_pause=_optional_number(values.get("voice_pause")),
            sopro_voice=_first(values, ("sopro_voice",)),
            sopro_speed=_optional_number(values.get("sopro_speed")),
            sopro_pitch=_optional_number(values.get("sopro_pitch")),
            sopro_gain=_optional_number(values.get("sopro_gain")),
            sopro_pause=_optional_number(values.get("sopro_pause")),
            sopro_temperature=_optional_number(values.get("sopro_temperature")),
            sopro_top_p=_optional_number(values.get("sopro_top_p")),
            sopro_top_k=_optional_number(values.get("sopro_top_k")),
            sopro_language=_first(values, ("sopro_language",)),
            pocket_voice=_first(values, ("pocket_voice",)),
            pocket_speed=_optional_number(values.get("pocket_speed")),
            pocket_pitch=_optional_number(values.get("pocket_pitch")),
            pocket_gain=_optional_number(values.get("pocket_gain")),
            pocket_pause=_optional_number(values.get("pocket_pause")),
            pocket_temperature=_optional_number(values.get("pocket_temperature")),
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


def _optional_number(value: object) -> float | None:
    """A number, or ``None`` for a key that is absent or unreadable.

    Distinct from :func:`_number` because here ``None`` is a meaning rather
    than a failure: an unset delivery field follows the default voice, and a
    field this could not read is unset. A hand-edited character file with
    ``voice_pitch: fast`` in it is a character with no pitch override, not a
    character that fails to load.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        found = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return None if found != found else found
