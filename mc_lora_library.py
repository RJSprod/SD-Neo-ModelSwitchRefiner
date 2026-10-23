"""The CSV library: everything the extension knows about LoRAs from user CSVs.

A "library" is simply a directory of CSV files. Each row describes one LoRA. The
reader is deliberately forgiving about column naming so CSVs from the bundled
Civitai tool *and* from unrelated tools both index correctly:

===================  ==========================================================
Identity             ``sha256`` / ``hash`` / ``model_hash`` / ``addnet_hash``,
                     and ``safetensor_file`` / ``filename`` / ``path``
Displayed as fields  ``civitai_model_name``, ``trigger_words``, ``notes``, ...
Displayed as prompts ``positive_prompt_1..N`` / ``prompt_1..N`` / ``prompt``
Displayed as text    ``model_description`` / ``description``, ``version_description``
===================  ==========================================================

Anything that is not recognised is still shown, under "Other fields", so no
information from a user's CSV is silently dropped.
"""

from __future__ import annotations

import csv
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from mc_lora_common import (
    NETWORK_EXTENSIONS,
    clean_cell,
    log,
    normalize_hash,
    normalize_key,
    normalize_path,
    path_stem,
    report,
)
import mc_lora_paths

# CSV cells can be very long (a gallery prompt easily exceeds the 128 KiB default).
try:
    csv.field_size_limit(sys.maxsize)
except (OverflowError, ValueError):  # pragma: no cover - platform dependent
    csv.field_size_limit(2**31 - 1)


FULL_HASH_COLUMNS = ("sha256", "sha_256", "sha256_hash", "file_sha256", "hash", "file_hash", "checksum")
SHORT_HASH_COLUMNS = ("addnet_hash", "shorthash", "short_hash", "model_hash", "sshs_model_hash", "autov2")
PATH_COLUMNS = ("safetensor_file", "safetensors_file", "file", "filename", "file_name", "path", "filepath", "file_path", "relative_path", "lora", "lora_file", "model_file")
NAME_COLUMNS = ("civitai_model_name", "model_name", "name", "title", "lora_name")
TRIGGER_COLUMNS = ("trigger_words", "trained_words", "trainedwords", "activation_text", "activation_words", "keywords", "triggers", "tags")
NEGATIVE_COLUMNS = ("negative_prompt", "negative_text", "negative_prompts")
DESCRIPTION_COLUMNS = ("model_description", "civitai_model_description", "description", "about_this_model", "model_about")
VERSION_DESCRIPTION_COLUMNS = ("version_description", "civitai_version_description", "about_this_version", "version_notes")
URL_COLUMNS = ("civitai_url", "url", "model_url", "link")

# Read for the panel's header without being taken out of the row's other fields,
# so the full field table still lists every one of them.
VERSION_NAME_COLUMNS = ("civitai_version_name", "version_name", "model_version", "version")
BASE_MODEL_COLUMNS = ("base_model", "civitai_base_model", "basemodel", "base_model_name")
CREATOR_COLUMNS = ("civitai_creator", "creator", "creator_username", "author", "uploader")
MODEL_ID_COLUMNS = ("civitai_model_id", "model_id")
VERSION_ID_COLUMNS = ("civitai_version_id", "version_id", "model_version_id")
STATUS_COLUMNS = ("status",)
NOTE_COLUMNS = ("note",)

PROMPT_COLUMN_RE = re.compile(r"^(?:positive_)?prompt(?:_?(\d+))?$")
# Written beside each prompt by the bundled fetcher; `image_url_N` is accepted
# too so a hand-made CSV can supply images without matching our exact naming.
PROMPT_IMAGE_URL_RE = re.compile(r"^(?:prompt_)?image_url_?(\d+)$")
PROMPT_IMAGE_ID_RE = re.compile(r"^(?:prompt_)?image_id_?(\d+)$")

# Separators used inside a single "trigger words" cell.
TRIGGER_SPLIT_RE = re.compile(r"\s*[,;\n]\s*")


def resolve_library_dir() -> Path:
    """The library directory from the setting, made if it is missing; the
    default under the LoRA data root when the configured one cannot be."""
    path = mc_lora_paths.library_dir()
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        log(f"could not create the configured library directory ({exc.strerror or type(exc).__name__}); using the default")
        path = mc_lora_paths.data_root() / mc_lora_paths.LIBRARY_DIRNAME
        path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass(frozen=True)
class PromptEntry:
    """A prompt from the library, with the gallery image it came from."""

    text: str
    image_url: str = ""
    image_id: str = ""


@dataclass
class LibraryRecord:
    """One CSV row, decoded into the pieces the details panel renders."""

    source: str = ""
    row_number: int = 0
    sha256: str = ""
    short_hashes: tuple[str, ...] = ()
    rel_path: str = ""
    stem: str = ""
    file_name: str = ""
    model_name: str = ""
    url: str = ""
    description: str = ""
    version_description: str = ""
    version_name: str = ""
    base_model: str = ""
    creator: str = ""
    model_id: str = ""
    version_id: str = ""
    status: str = ""
    note: str = ""
    trigger_words: tuple[str, ...] = ()
    entries: tuple[PromptEntry, ...] = ()
    negative_prompts: tuple[str, ...] = ()
    extras: dict[str, str] = field(default_factory=dict)
    raw: dict[str, str] = field(default_factory=dict)

    @property
    def title(self) -> str:
        return self.model_name or self.file_name or self.stem or self.rel_path or "(unnamed)"

    @property
    def prompts(self) -> tuple[str, ...]:
        return tuple(entry.text for entry in self.entries)

    @property
    def has_description(self) -> bool:
        return bool(self.description or self.version_description)

    @property
    def resolved(self) -> bool:
        """Did the row get any content? A fetch writes a row for a LoRA Civitai
        had nothing for too - with a status, and nothing else."""
        return bool(self.model_name or self.trigger_words or self.prompts)


def _file_name(raw_path: str) -> str:
    """The file's name as the CSV spells it, extension dropped - for display."""
    base = clean_cell(raw_path).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    for extension in NETWORK_EXTENSIONS:
        if base.lower().endswith(extension):
            return base[: -len(extension)]
    return base


def _split_triggers(value: str) -> list[str]:
    return [part for part in (piece.strip() for piece in TRIGGER_SPLIT_RE.split(value)) if part]



def parse_row(row: dict[str, str], *, source: str, row_number: int) -> LibraryRecord | None:
    """Decode one CSV row. Returns ``None`` for rows with nothing usable."""
    normalized: dict[str, str] = {}
    display: dict[str, str] = {}
    for key, value in row.items():
        if key is None:
            continue
        column = normalize_key(key)
        if not column:
            continue
        text = clean_cell(value if not isinstance(value, list) else " ".join(str(v) for v in value))
        # Later duplicate columns must not blank out an earlier populated one.
        if column in normalized and not text:
            continue
        normalized[column] = text
        display[str(key).strip() or column] = text

    if not any(normalized.values()):
        return None

    consumed: set[str] = set()

    def take(candidates) -> str:
        for candidate in candidates:
            value = normalized.get(candidate, "")
            if value:
                consumed.add(candidate)
                return value
            if candidate in normalized:
                consumed.add(candidate)
        return ""

    sha256 = ""
    for column in FULL_HASH_COLUMNS:
        candidate = normalize_hash(normalized.get(column, ""))
        if column in normalized:
            consumed.add(column)
        if not sha256 and len(candidate) == 64:
            sha256 = candidate

    short_hashes: list[str] = []
    for column in (*SHORT_HASH_COLUMNS, *FULL_HASH_COLUMNS):
        candidate = normalize_hash(normalized.get(column, ""))
        if column in normalized:
            consumed.add(column)
        if candidate and len(candidate) != 64 and candidate not in short_hashes:
            short_hashes.append(candidate)

    def peek(candidates) -> str:
        for candidate in candidates:
            value = normalized.get(candidate, "")
            if value:
                return value
        return ""

    raw_path = take(PATH_COLUMNS)
    model_name = take(NAME_COLUMNS)
    url = take(URL_COLUMNS)
    description = take(DESCRIPTION_COLUMNS)
    version_description = take(VERSION_DESCRIPTION_COLUMNS)

    triggers: list[str] = []
    for column in TRIGGER_COLUMNS:
        value = normalized.get(column, "")
        if column in normalized:
            consumed.add(column)
        for word in _split_triggers(value):
            if word not in triggers:
                triggers.append(word)

    # Prompts and their images are paired by the trailing index, so an empty
    # prompt cell cannot shift the images onto the wrong rows.
    prompt_by_index: dict[int, str] = {}
    urls_by_index: dict[int, str] = {}
    ids_by_index: dict[int, str] = {}
    unnumbered: list[str] = []

    for column, value in normalized.items():
        match = PROMPT_COLUMN_RE.match(column)
        if match:
            consumed.add(column)
            if not value:
                continue
            if match.group(1):
                prompt_by_index[int(match.group(1))] = value
            else:
                unnumbered.append(value)
            continue

        match = PROMPT_IMAGE_URL_RE.match(column)
        if match:
            consumed.add(column)
            if value:
                urls_by_index[int(match.group(1))] = value
            continue

        match = PROMPT_IMAGE_ID_RE.match(column)
        if match:
            consumed.add(column)
            if value:
                ids_by_index[int(match.group(1))] = value

    entries: list[PromptEntry] = [PromptEntry(text) for text in unnumbered]
    entries.extend(
        PromptEntry(prompt_by_index[index], urls_by_index.get(index, ""), ids_by_index.get(index, ""))
        for index in sorted(prompt_by_index)
    )

    negatives: list[str] = []
    for column in NEGATIVE_COLUMNS:
        value = normalized.get(column, "")
        if column in normalized:
            consumed.add(column)
        if value:
            negatives.append(value)

    extras: dict[str, str] = {}
    for label, value in display.items():
        if normalize_key(label) in consumed:
            continue
        if value:
            extras[label] = value

    return LibraryRecord(
        source=source,
        row_number=row_number,
        sha256=sha256,
        short_hashes=tuple(short_hashes),
        rel_path=normalize_path(raw_path),
        stem=path_stem(raw_path) or normalize_key(model_name),
        file_name=_file_name(raw_path),
        model_name=model_name,
        url=url,
        description=description,
        version_description=version_description,
        version_name=peek(VERSION_NAME_COLUMNS),
        base_model=peek(BASE_MODEL_COLUMNS),
        creator=peek(CREATOR_COLUMNS),
        model_id=peek(MODEL_ID_COLUMNS),
        version_id=peek(VERSION_ID_COLUMNS),
        status=peek(STATUS_COLUMNS),
        note=peek(NOTE_COLUMNS),
        trigger_words=tuple(triggers),
        entries=tuple(entries),
        negative_prompts=tuple(negatives),
        extras=extras,
        raw=display,
    )


def read_csv(path: Path) -> tuple[list[LibraryRecord], str]:
    """Parse one CSV file. Returns ``(records, error_message)``."""
    records: list[LibraryRecord] = []
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            with path.open("r", encoding=encoding, newline="") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames:
                    return [], "file has no header row"
                for number, row in enumerate(reader, start=2):
                    record = parse_row(row, source=path.name, row_number=number)
                    if record is not None:
                        records.append(record)
            return records, ""
        except UnicodeDecodeError:
            records = []
            continue
        except (OSError, csv.Error) as exc:
            return [], (getattr(exc, "strerror", None) or str(exc))
    return [], "could not decode the file as text"


@dataclass
class SourceInfo:
    name: str
    path: str
    rows: int
    size: int
    mtime: float
    error: str = ""


class Library:
    """Thread-safe index over every CSV in the library directory."""

    def __init__(self, directory: Path | None = None):
        self._lock = threading.RLock()
        self._directory: Path | None = Path(directory) if directory else None
        self._records: list[LibraryRecord] = []
        self._by_sha256: dict[str, list[LibraryRecord]] = {}
        self._by_short: dict[str, list[LibraryRecord]] = {}
        self._by_path: dict[str, list[LibraryRecord]] = {}
        self._by_stem: dict[str, list[LibraryRecord]] = {}
        self._sources: list[SourceInfo] = []
        self._signature: tuple = ()
        self._loaded_at: float = 0.0

    # ------------------------------------------------------------ directory

    @property
    def directory(self) -> Path:
        with self._lock:
            if self._directory is not None:
                return self._directory
        return resolve_library_dir()

    def set_directory(self, directory: Path | None) -> None:
        with self._lock:
            self._directory = Path(directory) if directory else None
            self._signature = ()

    def csv_files(self) -> list[Path]:
        directory = self.directory
        if not directory.is_dir():
            return []
        try:
            return sorted(
                (path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() == ".csv"),
                key=lambda path: path.name.casefold(),
            )
        except OSError as exc:
            log(f"could not list the library directory ({exc.strerror or type(exc).__name__})")
            return []

    def _current_signature(self) -> tuple:
        signature = []
        for path in self.csv_files():
            try:
                info = path.stat()
            except OSError:
                continue
            signature.append((str(path), info.st_size, int(info.st_mtime_ns)))
        return tuple(signature)

    # --------------------------------------------------------------- (re)load

    def ensure_loaded(self) -> None:
        """Reload when a CSV was added, removed or modified since last time."""
        signature = self._current_signature()
        with self._lock:
            if signature == self._signature and self._loaded_at:
                return
        self.reload()

    def reload(self) -> list[SourceInfo]:
        directory = self.directory
        signature = self._current_signature()

        records: list[LibraryRecord] = []
        sources: list[SourceInfo] = []

        for path in self.csv_files():
            try:
                info = path.stat()
            except OSError:
                continue
            parsed, error = read_csv(path)
            if error:
                log(f"skipping a CSV file that could not be read: {error}")
            records.extend(parsed)
            sources.append(
                SourceInfo(
                    name=str(path.relative_to(directory)) if path.is_relative_to(directory) else path.name,
                    path=str(path),
                    rows=len(parsed),
                    size=info.st_size,
                    mtime=info.st_mtime,
                    error=error,
                )
            )

        by_sha256: dict[str, list[LibraryRecord]] = {}
        by_short: dict[str, list[LibraryRecord]] = {}
        by_path: dict[str, list[LibraryRecord]] = {}
        by_stem: dict[str, list[LibraryRecord]] = {}

        def add(index: dict[str, list[LibraryRecord]], key: str, record: LibraryRecord) -> None:
            if not key:
                return
            bucket = index.setdefault(key, [])
            if not any(existing is record for existing in bucket):
                bucket.append(record)

        for record in records:
            add(by_sha256, record.sha256, record)
            if record.sha256:
                add(by_short, record.sha256[:12], record)
                add(by_short, record.sha256[:10], record)
            for short in record.short_hashes:
                add(by_short, short[:12], record)
                add(by_short, short[:10], record)
            if record.rel_path:
                add(by_path, record.rel_path, record)
                add(by_path, record.rel_path.rsplit("/", 1)[-1], record)
            add(by_stem, record.stem, record)

        with self._lock:
            self._records = records
            self._by_sha256 = by_sha256
            self._by_short = by_short
            self._by_path = by_path
            self._by_stem = by_stem
            self._sources = sources
            self._signature = signature
            self._loaded_at = time.time()

        log(f"library loaded: {len(records)} row(s) from {len(sources)} CSV file(s)")
        return sources

    # ---------------------------------------------------------------- lookup

    def lookup(
        self,
        *,
        sha256: str = "",
        short_hashes=(),
        filename: str = "",
        rel_paths=(),
        name: str = "",
    ) -> tuple[list[LibraryRecord], str]:
        """Find rows for one LoRA. Returns ``(records, how_it_matched)``."""
        self.ensure_loaded()

        with self._lock:
            by_sha256 = self._by_sha256
            by_short = self._by_short
            by_path = self._by_path
            by_stem = self._by_stem

        sha256 = normalize_hash(sha256)
        if sha256 and sha256 in by_sha256:
            return list(by_sha256[sha256]), "SHA256"

        for short in short_hashes:
            candidate = normalize_hash(short)
            for width in (12, 10):
                key = candidate[:width]
                if key and key in by_short:
                    return list(by_short[key]), "short hash"

        for candidate in rel_paths:
            key = normalize_path(candidate)
            if key and key in by_path:
                return list(by_path[key]), "file path"

        base = normalize_path(filename).rsplit("/", 1)[-1]
        if base and base in by_path:
            return list(by_path[base]), "file name"

        stem = path_stem(filename) or normalize_key(name)
        if stem and stem in by_stem:
            return list(by_stem[stem]), "file name"

        return [], ""

    # ----------------------------------------------------------------- stats

    @property
    def sources(self) -> list[SourceInfo]:
        self.ensure_loaded()
        with self._lock:
            return list(self._sources)

    @property
    def records(self) -> list[LibraryRecord]:
        self.ensure_loaded()
        with self._lock:
            return list(self._records)

    def known_sha256(self, *, resolved_only: bool = False, require_description: bool = False) -> set[str]:
        """SHA256s an incremental scan can skip.

        Every LoRA the library has already recorded counts, including ones
        Civitai had nothing for: a refresh is for picking up *new* LoRAs, and
        re-asking about known misses on every run wastes the whole scan.

        Two options narrow that down:

        ``resolved_only``
            drops rows that carry no content at all, so the LoRAs Civitai could
            not resolve are looked at again. This is what the "retry unresolved"
            checkbox passes.
        ``require_description``
            drops rows that *did* resolve but hold no description - rows written
            before descriptions were collected, say - so a refresh fills in the
            text they are missing. Rows that never resolved are deliberately not
            affected: whether those are retried stays ``resolved_only``'s call,
            and no miss should be re-queried on every single run.
        """
        self.ensure_loaded()
        with self._lock:
            records = list(self._records)

        known: set[str] = set()
        for record in records:
            if len(record.sha256) != 64:
                continue
            resolved = record.resolved
            if resolved_only and not resolved:
                continue
            if require_description and resolved and not record.has_description:
                continue
            known.add(record.sha256)
        return known

    def stats(self) -> dict:
        self.ensure_loaded()
        with self._lock:
            return {
                "directory": str(self.directory),
                "files": len(self._sources),
                "rows": len(self._records),
                "hashed_rows": sum(1 for record in self._records if record.sha256),
                "prompts": sum(len(record.prompts) for record in self._records),
                "described_rows": sum(1 for record in self._records if record.has_description),
                "errors": [source for source in self._sources if source.error],
            }

    # ------------------------------------------------------------ management

    def import_csv(self, source_path, *, new_name: str = "", overwrite: bool = True) -> tuple[bool, str]:
        """Copy a CSV into the library directory and reindex."""
        import shutil

        if not source_path:
            return False, "No file was provided."

        source = Path(source_path)
        if not source.is_file():
            return False, f"File not found: {source}"

        name = (new_name or source.name).strip()
        name = Path(name).name  # never let a path escape the library directory
        if not name:
            return False, "The file needs a name."
        if not name.lower().endswith(".csv"):
            name += ".csv"

        records, error = read_csv(source)
        if error:
            return False, f"Could not read the CSV: {error}"
        if not records:
            return False, "The CSV parsed cleanly but contains no usable rows."

        destination = self.directory / name
        if destination.exists() and not overwrite:
            stem, suffix = destination.stem, destination.suffix
            for index in range(2, 1000):
                candidate = destination.with_name(f"{stem}-{index}{suffix}")
                if not candidate.exists():
                    destination = candidate
                    break

        try:
            same_file = destination.exists() and source.resolve() == destination.resolve()
            if not same_file:
                shutil.copyfile(source, destination)
        except OSError as exc:
            return False, f"Could not copy the file into the library: {exc}"

        self.reload()
        hashed = sum(1 for record in records if record.sha256)
        note = "" if hashed else " — no SHA256 column found, rows will match by file name only"
        return True, f'Imported "{destination.name}": {len(records)} row(s), {hashed} with a SHA256{note}.'

    def remove_csv(self, name: str) -> tuple[bool, str]:
        if not name:
            return False, "No file selected."
        target = (self.directory / name).resolve()
        directory = self.directory.resolve()
        if not target.is_relative_to(directory):
            return False, "Refusing to delete a file outside the library directory."
        if not target.is_file():
            return False, f'"{name}" is not in the library.'
        try:
            target.unlink()
        except OSError as exc:
            return False, f"Could not delete the file: {exc}"
        self.reload()
        return True, f'Removed "{name}" from the library.'


LIBRARY = Library()


def get_library() -> Library:
    """The process-wide library, kept in sync with the configured directory."""
    try:
        configured = resolve_library_dir()
    except Exception:
        report("could not resolve the library directory")
        return LIBRARY

    if LIBRARY._directory is None or Path(LIBRARY._directory) != configured:
        LIBRARY.set_directory(configured)
    return LIBRARY
