"""Civitai lookup and CSV generation.

Adapted from the user's ``civitai_krea2_lora_csv.py`` so it can run inside the
WebUI process (cancellable, with progress callbacks) as well as from a CLI. The
identity rules of the original are kept intact:

* every local file is SHA256-hashed and looked up via ``/model-versions/by-hash``
* when the matched version exposes SHA256 values, one of them must equal the
  local digest - a disagreement is rejected rather than trusted
* the returned *version id* is authoritative for trigger words and for filtering
  the gallery, so sibling versions on the same model page cannot bleed in

The Krea 2 specific gate is generalised into an optional ``base_model_filter`` -
one base model, or several ("Krea 2, MOSS Image") - so the extension is useful
for any LoRA collection; leaving it empty accepts any exact hash match, which is
the behaviour of the original script's default mode. Where Civitai's label is
generic, a LoRA's own tensor names can stand in for it (MOSS-Image's layout is
unmistakable).

What a scan collects is *text*: model name, base model, trigger words, gallery
prompts and - fetched by default - the model's description, converted from
Civitai's HTML by :mod:`.descriptions`. Gallery images are never downloaded here;
only the URL of the image behind each prompt is recorded, so downloading one
stays a separate, deliberate request.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import struct
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import urljoin, urlparse

import mc_lora_layouts
from mc_lora_common import SAFETENSORS_EXTENSIONS, clean_cell, normalize_key
from mc_lora_descriptions import html_to_text
from mc_lora_hashing import addnet_sha256_file, sha256_file

# Gallery prompts routinely exceed csv's default 128 KiB field limit.
try:
    csv.field_size_limit(sys.maxsize)
except (OverflowError, ValueError):  # pragma: no cover - platform dependent
    csv.field_size_limit(2**31 - 1)

BASE_URL = "https://civitai.com/api/v1"
GALLERY_PAGE_SIZE = 200
USER_AGENT = "SD-Neo-ModelSwitchRefiner/1.0"
MAX_HEADER_BYTES = 100_000_000

LOCAL_BASE_KEYS = {
    "ss_base_model_version",
    "base_model_version",
    "base_model",
    "model_type",
    "model_family",
    "architecture",
    "base_model_name",
    "modelspec_architecture",
    "modelspec_implementation",
    "ss_architecture",
    "ss_base_checkpoint",
}

NEUTRAL_BASE_LABELS = {
    "",
    "other",
    "unknown",
    "custom",
    "n_a",
    "na",
    "none",
    "not_specified",
    "unspecified",
}

# Civitai's ``baseModelType`` names the *variant* of a version, never its family,
# so it can neither satisfy a base-model filter nor contradict one.
VARIANT_TYPE_LABELS = {"standard", "inpainting", "refiner", "pix2pix"}

# Several base models can be accepted at once: "Krea 2, MOSS Image".
BASE_MODEL_SPLIT_RE = re.compile(r"\s*[,;|]\s*")

# Architectures recognisable from a LoRA's own tensor names, for when Civitai's
# label is missing or generic ("Other"). A pattern is only listed when it is
# specific to that architecture's module layout: MOSS-Image is the one DiT whose
# blocks keep their projections in a named ``proj`` dict, MLP included.
LOCAL_ARCHITECTURES = (
    ("MOSS Image", re.compile(r"(?:^|[._])layers[._]\d+[._]proj[._]ffn_(?:up_gate|down)(?:[._]|$)")),
)

@dataclass(frozen=True)
class PromptImage:
    """A gallery prompt together with the image it was taken from."""

    prompt: str
    image_url: str = ""
    image_id: str = ""


BASE_COLUMNS = [
    "safetensor_file",
    "sha256",
    "addnet_hash",
    "civitai_model_name",
    "civitai_model_id",
    "civitai_version_id",
    "civitai_version_name",
    "base_model",
    "civitai_creator",
    "civitai_url",
    "trigger_words",
    "status",
    "note",
    # Text details, kept at the end of the fixed columns: they are long, and a
    # spreadsheet is far easier to read when the identity columns come first.
    "model_description",
    "version_description",
]


class CivitaiError(RuntimeError):
    """A Civitai/network response that prevents safe extraction."""


class Cancelled(RuntimeError):
    """The operator asked for the scan to stop."""


# --------------------------------------------------------------- base model


def base_model_choices(wanted: Any) -> list[str]:
    """The base models a filter accepts: ``"Krea 2, MOSS Image"`` is two."""
    if isinstance(wanted, (list, tuple, set)):
        pieces = [str(item) for item in wanted]
    else:
        pieces = BASE_MODEL_SPLIT_RE.split(str(wanted or ""))
    return [piece.strip() for piece in pieces if normalize_key(piece)]


def matches_base_model(value: Any, wanted: Any) -> bool:
    """Loose comparison so "Krea 2" matches "Krea2", "Krea-2-Turbo", ...

    ``wanted`` may name several base models; matching any one of them is enough.
    """
    choices = base_model_choices(wanted)
    if not choices:
        return True
    return any(_matches_one_base_model(value, choice) for choice in choices)


def _matches_one_base_model(value: Any, wanted: str) -> bool:
    wanted_key = normalize_key(wanted)
    text = normalize_key(value)
    if not text:
        return False
    if wanted_key in text or text in wanted_key:
        return True
    if wanted_key.replace("_", "") in text.replace("_", ""):
        return True
    wanted_tokens = [token for token in wanted_key.split("_") if token]
    tokens = set(text.split("_"))
    return bool(wanted_tokens) and all(token in tokens for token in wanted_tokens)


@dataclass(frozen=True)
class BaseModelAssessment:
    allowed: bool
    explicit: bool
    reason: str


def read_safetensors_header(path: Path) -> tuple[dict[str, Any], list[str]]:
    """``(metadata, tensor names)`` from the JSON header only. Never fatal."""
    try:
        path = Path(path)
        file_size = path.stat().st_size
        if file_size < 10:
            return {}, []
        with path.open("rb") as handle:
            raw_len = handle.read(8)
            if len(raw_len) != 8:
                return {}, []
            header_len = struct.unpack("<Q", raw_len)[0]
            if header_len < 2 or header_len > file_size - 8 or header_len > MAX_HEADER_BYTES:
                return {}, []
            raw_header = handle.read(header_len)
            if len(raw_header) != header_len:
                return {}, []
        header = json.loads(raw_header.decode("utf-8"))
        if not isinstance(header, dict):
            return {}, []
        metadata = header.get("__metadata__", {})
        names = [str(key) for key in header if key != "__metadata__"]
        return (metadata if isinstance(metadata, dict) else {}), names
    except (OSError, ValueError, UnicodeDecodeError, struct.error):
        return {}, []


def read_safetensors_metadata(path: Path) -> dict[str, Any]:
    """Read only the safetensors JSON header. Failures are not fatal."""
    return read_safetensors_header(path)[0]


def local_architecture(tensor_names: Iterable[str]) -> str:
    """The architecture a LoRA's tensor names identify, or ``""``."""
    names = list(tensor_names or ())
    for label, pattern in LOCAL_ARCHITECTURES:
        if any(pattern.search(name) for name in names):
            return label
    return ""


def local_base_model_evidence(metadata: dict[str, Any], wanted: str, tensor_names: Iterable[str] = ()) -> str | None:
    for key, value in metadata.items():
        if normalize_key(key) in LOCAL_BASE_KEYS and matches_base_model(value, wanted):
            return f"safetensors metadata {key}={value}"
    architecture = local_architecture(tensor_names)
    if architecture and matches_base_model(architecture, wanted):
        return f"safetensors tensor names are {architecture}'s"
    return None


def family_label(version: dict[str, Any]) -> str:
    """The base model a version names: ``baseModel``, or a ``baseModelType``
    that is more than a variant such as "Standard"."""
    label = clean_cell(version.get("baseModel"))
    if label:
        return label
    fallback = clean_cell(version.get("baseModelType"))
    return "" if normalize_key(fallback) in VARIANT_TYPE_LABELS else fallback


def assess_base_model(
    version: dict[str, Any],
    local_metadata: dict[str, Any],
    *,
    wanted: str,
    require_explicit: bool,
    tensor_names: Iterable[str] = (),
) -> BaseModelAssessment:
    """Validate the hash-matched version against the requested base model(s)."""
    if not base_model_choices(wanted):
        label = family_label(version)
        return BaseModelAssessment(True, bool(label), f"exact hash match (baseModel={label or 'unspecified'})")

    fields = [("baseModel", version.get("baseModel")), ("baseModelType", version.get("baseModelType"))]

    conflicting: list[str] = []
    saw_neutral = False

    for field_name, value in fields:
        if value is None or not str(value).strip():
            saw_neutral = True
            continue
        key = normalize_key(value)
        if field_name == "baseModelType" and key in VARIANT_TYPE_LABELS:
            saw_neutral = True
            continue
        if matches_base_model(value, wanted):
            return BaseModelAssessment(True, True, f"Civitai {field_name}={value}")
        if key in NEUTRAL_BASE_LABELS:
            saw_neutral = True
        else:
            conflicting.append(f"{field_name}={value}")

    local_evidence = local_base_model_evidence(local_metadata, wanted, tensor_names)
    if local_evidence:
        return BaseModelAssessment(True, True, local_evidence)

    if conflicting:
        return BaseModelAssessment(False, False, "Civitai labels a different base model: " + "; ".join(conflicting))

    weak: list[tuple[str, Any]] = [("version name", version.get("name"))]
    model = version.get("model")
    if isinstance(model, dict):
        weak.append(("model name", model.get("name")))
    files = version.get("files")
    if isinstance(files, list):
        weak.extend(("Civitai file name", item.get("name")) for item in files if isinstance(item, dict))

    for field_name, value in weak:
        if value and matches_base_model(value, wanted):
            return BaseModelAssessment(True, False, f"exact hash match; {wanted} indicated by {field_name}={value}")

    if require_explicit:
        return BaseModelAssessment(False, False, f"exact hash matched, but no explicit {wanted} label was available")

    if saw_neutral or not conflicting:
        return BaseModelAssessment(True, False, "exact hash matched; Civitai supplied no conflicting base-model label")

    return BaseModelAssessment(False, False, "unable to validate the base model")


def version_file_hash_state(version: dict[str, Any], local_sha256: str) -> tuple[bool, str]:
    """Cross-check the digests inside the version response against the file."""
    files = version.get("files")
    if not isinstance(files, list):
        return True, "version response has no file list; exact by-hash lookup used"

    observed: list[str] = []
    for file_info in files:
        if not isinstance(file_info, dict):
            continue
        hashes = file_info.get("hashes")
        if not isinstance(hashes, dict):
            continue
        for key, value in hashes.items():
            if normalize_key(key) == "sha256" and isinstance(value, str):
                candidate = value.strip().lower()
                if candidate:
                    observed.append(candidate)
                    if candidate == local_sha256:
                        name = clean_cell(file_info.get("name")) or "unnamed file"
                        return True, f"Civitai file SHA256 matches ({name})"

    if observed:
        return False, "by-hash response is inconsistent: the version exposes SHA256 values, but none equals the local file"

    return True, "version exposes no SHA256 field; exact by-hash lookup used"


# ------------------------------------------------------------------ prompts


def coerce_meta(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                return None
            return parsed if isinstance(parsed, dict) else None
    return None


def scalar_prompt(value: Any) -> str:
    if isinstance(value, str):
        return clean_cell(value)
    if isinstance(value, list):
        pieces = [clean_cell(item) for item in value]
        return ", ".join(piece for piece in pieces if piece)
    return ""


DESIRED_PROMPT_KEYS = {"prompt", "positive_prompt", "positiveprompt", "positive_text_prompt", "positivetextprompt"}


def extract_positive_prompt(image: dict[str, Any]) -> str:
    """Pull the positive prompt out of an image record, never the negative one."""
    containers: list[dict[str, Any]] = []
    meta = coerce_meta(image.get("meta"))
    if meta:
        containers.append(meta)
    containers.append(image)

    for container in containers:
        for key, value in container.items():
            if normalize_key(key) in DESIRED_PROMPT_KEYS:
                text = scalar_prompt(value)
                if text:
                    return text
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if normalize_key(subkey) in {"positive", "prompt", "text"}:
                            text = scalar_prompt(subvalue)
                            if text:
                                return text
                    # A MOSS-Image layout the API passed on as an object rather
                    # than as the compiled line it was written as.
                    text = mc_lora_layouts.compile_object(value)
                    if text:
                        return text

    for container in containers:
        for nested_key in ("parameters", "generation", "generation_data", "generationData"):
            nested = container.get(nested_key)
            if not isinstance(nested, dict):
                continue
            for key, value in nested.items():
                if normalize_key(key) in DESIRED_PROMPT_KEYS:
                    text = scalar_prompt(value)
                    if text:
                        return text

    return ""


def image_identity(image: dict[str, Any]) -> str:
    image_id = image.get("id")
    if image_id is not None:
        return f"id:{image_id}"
    url = clean_cell(image.get("url"))
    if url:
        return f"url:{url}"
    try:
        payload = json.dumps(image, sort_keys=True, ensure_ascii=False, default=str)
        return "json:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()
    except (TypeError, ValueError):
        return f"object:{id(image)}"


def image_url(image: dict[str, Any]) -> str:
    """The image's own URL, if the record exposes a usable http(s) one."""
    url = clean_cell(image.get("url"))
    if url.startswith(("http://", "https://")):
        return url
    return ""


def image_key(image: dict[str, Any]) -> str:
    """A stable id for the image, used to name its file on disk."""
    raw = image.get("id")
    if raw is not None and str(raw).strip():
        return str(raw).strip()
    return ""


def image_sort_key(image: dict[str, Any]) -> tuple[str, int, str]:
    created = clean_cell(image.get("createdAt"))
    try:
        numeric_id = int(image.get("id"))
    except (TypeError, ValueError):
        numeric_id = 0
    return (created, numeric_id, image_identity(image))


# ------------------------------------------------------------------- client


class CivitaiClient:
    def __init__(self, *, api_key: str | None = None, timeout: float = 30.0, request_delay: float = 0.15, cancel: threading.Event | None = None):
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        self.timeout = timeout
        self.request_delay = max(0.0, request_delay)
        self.cancel = cancel
        self._last_request_at = 0.0
        self._requests = requests

        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json", "User-Agent": USER_AGENT})
        if api_key:
            self.session.headers["Authorization"] = f"Bearer {api_key}"

        retry = Retry(
            total=5,
            connect=5,
            read=5,
            status=5,
            backoff_factor=1.0,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
            respect_retry_after_header=True,
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=4)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        self.model_cache: dict[int, dict[str, Any]] = {}
        self.gallery_cache: dict[int, tuple[PromptImage, ...]] = {}

    def close(self) -> None:
        try:
            self.session.close()
        except Exception:
            pass

    def _check_cancel(self) -> None:
        if self.cancel is not None and self.cancel.is_set():
            raise Cancelled()

    def _pace(self) -> None:
        if self.request_delay <= 0:
            return
        deadline = self._last_request_at + self.request_delay
        while True:
            self._check_cancel()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(remaining, 0.1))

    def _get_json(self, url: str, *, params: dict[str, Any] | None = None, allow_404: bool = False) -> dict[str, Any] | None:
        self._check_cancel()
        self._pace()
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            self._last_request_at = time.monotonic()
        except self._requests.RequestException as exc:
            raise CivitaiError(f"network error: {exc}") from exc

        if allow_404 and response.status_code == 404:
            return None
        if response.status_code == 401:
            raise CivitaiError("Civitai returned HTTP 401; check the API key")
        if response.status_code == 403:
            raise CivitaiError("Civitai returned HTTP 403; authentication or content access may be required")
        if response.status_code != 200:
            raise CivitaiError(f"Civitai returned HTTP {response.status_code} for {response.url}")

        try:
            payload = response.json()
        except ValueError as exc:
            raise CivitaiError(f"Civitai returned invalid JSON for {response.url}: {exc}") from exc
        if not isinstance(payload, dict):
            raise CivitaiError(f"Civitai returned non-object JSON for {response.url}")
        return payload

    def version_by_hash(self, sha256: str) -> dict[str, Any] | None:
        return self._get_json(f"{BASE_URL}/model-versions/by-hash/{sha256}", allow_404=True)

    def model(self, model_id: int) -> dict[str, Any]:
        cached = self.model_cache.get(model_id)
        if cached is not None:
            return cached
        payload = self._get_json(f"{BASE_URL}/models/{model_id}") or {}
        self.model_cache[model_id] = payload
        return payload

    def model_info(self, version: dict[str, Any], *, with_description: bool = False) -> dict[str, Any]:
        """The parent model record, fetching the model page when needed.

        The version response embeds a small model record that carries the name
        and type but never the description, so asking for a description means
        the model page has to be fetched. ``self.model`` caches by model id, so
        several versions of one model still cost a single request.
        """
        nested_model = version.get("model")
        nested = dict(nested_model) if isinstance(nested_model, dict) else {}

        try:
            model_id = int(version.get("modelId"))
        except (TypeError, ValueError):
            return nested

        sufficient = bool(clean_cell(nested.get("name")) and clean_cell(nested.get("type")))
        if with_description and not clean_cell(nested.get("description")):
            sufficient = False
        if sufficient:
            return nested

        try:
            parent = self.model(model_id)
        except CivitaiError:
            return nested

        merged = dict(parent)
        for key, value in nested.items():
            if value is None:
                continue
            # A blank field on the embedded record must not blank out the one
            # the model page actually filled in.
            if isinstance(value, str) and not value.strip() and clean_cell(merged.get(key)):
                continue
            merged[key] = value
        return merged

    def gallery_prompts(self, version_id: int, *, embedded_images: Any, max_images: int) -> tuple[tuple[PromptImage, ...], str | None]:
        cached = self.gallery_cache.get(version_id)
        if cached is not None and max_images == 0:
            return cached, None

        images_by_identity: dict[str, dict[str, Any]] = {}

        # Seed with the version's own showcase images so a transient /images
        # failure does not discard prompt metadata we already hold.
        if isinstance(embedded_images, list):
            for image in embedded_images:
                if isinstance(image, dict):
                    images_by_identity.setdefault(image_identity(image), image)

        fetched_count = 0
        seen_pages: set[tuple[str, ...]] = set()
        warning: str | None = None

        initial_limit = min(GALLERY_PAGE_SIZE, max_images) if max_images > 0 else GALLERY_PAGE_SIZE
        next_url: str | None = f"{BASE_URL}/images"
        next_params: dict[str, Any] | None = {
            "modelVersionId": version_id,
            "limit": initial_limit,
            "sort": "Newest",
            "period": "AllTime",
        }

        while next_url:
            if max_images > 0 and fetched_count >= max_images:
                break
            try:
                payload = self._get_json(next_url, params=next_params)
            except CivitaiError as exc:
                warning = str(exc)
                break
            if payload is None:
                break

            # Only the first request carries explicit params: nextPage may switch
            # to cursor pagination and must be followed verbatim.
            next_params = None

            raw_items = payload.get("items", [])
            items = [item for item in raw_items if isinstance(item, dict)] if isinstance(raw_items, list) else []

            signature = tuple(image_identity(item) for item in items)
            if signature and signature in seen_pages:
                warning = "Civitai gallery pagination repeated a previously seen page; stopped defensively"
                break
            if signature:
                seen_pages.add(signature)

            for image in items:
                if max_images > 0 and fetched_count >= max_images:
                    break
                images_by_identity.setdefault(image_identity(image), image)
                fetched_count += 1

            if (max_images > 0 and fetched_count >= max_images) or not items:
                break

            metadata = payload.get("metadata")
            candidate_next: str | None = None
            if isinstance(metadata, dict):
                raw_next = metadata.get("nextPage")
                if isinstance(raw_next, str) and raw_next.strip():
                    candidate_next = urljoin(f"{BASE_URL}/", raw_next.strip())

            if candidate_next:
                host = (urlparse(candidate_next).hostname or "").lower()
                if host != "civitai.com" and not host.endswith(".civitai.com"):
                    warning = "Civitai returned a gallery nextPage URL outside civitai.com; stopped rather than forwarding credentials"
                    break
                next_url = candidate_next
                continue

            # Fallback for responses that omit nextPage but still paginate.
            if isinstance(metadata, dict):
                try:
                    total_pages = int(metadata.get("totalPages"))
                    current_page = int(metadata.get("currentPage"))
                except (TypeError, ValueError):
                    total_pages = current_page = 0
                if 0 < current_page < total_pages:
                    next_url = f"{BASE_URL}/images"
                    next_params = {
                        "modelVersionId": version_id,
                        "limit": initial_limit,
                        "page": current_page + 1,
                        "sort": "Newest",
                        "period": "AllTime",
                    }
                    continue

            break

        ordered = sorted(images_by_identity.values(), key=image_sort_key)
        if max_images > 0:
            ordered = ordered[:max_images]

        entries: list[PromptImage] = []
        for image in ordered:
            prompt = extract_positive_prompt(image)
            if prompt:
                entries.append(PromptImage(prompt, image_url(image), image_key(image)))

        result = tuple(entries)
        if max_images == 0 and warning is None:
            self.gallery_cache[version_id] = result
        return result, warning


# ---------------------------------------------------------------- resolution


@dataclass(frozen=True)
class Resolution:
    status: str
    model_name: str = ""
    trigger_words: tuple[str, ...] = ()
    prompts: tuple[PromptImage, ...] = ()
    model_id: int | None = None
    version_id: int | None = None
    version_name: str = ""
    base_model: str = ""
    creator: str = ""
    url: str = ""
    model_description: str = ""
    version_description: str = ""
    reason: str = ""

    @property
    def resolved(self) -> bool:
        return self.status in {"ok", "ok_unlabeled", "partial"}


def clean_words(values: Any) -> tuple[str, ...]:
    if not isinstance(values, list):
        return ()
    return tuple(text for text in (clean_cell(value) for value in values) if text)


def resolve_hash(
    sha256: str,
    local_metadata: dict[str, Any],
    client: CivitaiClient,
    *,
    base_model_filter: str = "",
    require_explicit_label: bool = False,
    max_images: int = 0,
    fetch_descriptions: bool = True,
    allowed_types: Iterable[str] = ("lora", "locon", "loha", "lycoris", "dora"),
    tensor_names: Iterable[str] = (),
) -> Resolution:
    """Resolve one SHA256 into the CSV payload for that LoRA.

    ``tensor_names`` are the file's own, from its header: a MOSS-Image LoRA is
    recognisable by them, which counts as base-model evidence when Civitai's
    label is generic.
    """
    try:
        version = client.version_by_hash(sha256)
    except CivitaiError as exc:
        return Resolution("error", reason=str(exc))

    if version is None:
        return Resolution("not_found", reason="SHA256 not found on Civitai")

    try:
        version_id = int(version.get("id"))
    except (TypeError, ValueError):
        return Resolution("invalid_response", reason="the hash lookup returned no valid model version id")

    hashes_ok, hash_reason = version_file_hash_state(version, sha256)
    if not hashes_ok:
        return Resolution("hash_mismatch", version_id=version_id, reason=hash_reason)

    parent_model = client.model_info(version, with_description=fetch_descriptions)
    model_type_raw = parent_model.get("type")
    model_type = normalize_key(model_type_raw)
    allowed = {normalize_key(item) for item in allowed_types}
    if model_type and allowed and model_type not in allowed:
        return Resolution("not_lora", version_id=version_id, reason=f"the Civitai model type is {model_type_raw}")

    assessment = assess_base_model(
        version,
        local_metadata,
        wanted=base_model_filter,
        require_explicit=require_explicit_label,
        tensor_names=tensor_names,
    )
    if not assessment.allowed:
        return Resolution("base_model_mismatch", version_id=version_id, reason=assessment.reason)

    try:
        model_id = int(version.get("modelId"))
    except (TypeError, ValueError):
        model_id = None

    model_name = clean_cell(parent_model.get("name"))
    base_model = family_label(version)
    # Only the model page names its creator, so this is filled in whenever the
    # page is fetched - which collecting descriptions always does.
    creator = parent_model.get("creator")
    creator_name = clean_cell(creator.get("username")) if isinstance(creator, dict) else ""
    url = f"https://civitai.com/models/{model_id}?modelVersionId={version_id}" if model_id else ""

    prompts, warning = client.gallery_prompts(version_id, embedded_images=version.get("images"), max_images=max_images)

    common = {
        "model_name": model_name,
        "trigger_words": clean_words(version.get("trainedWords")),
        "prompts": prompts,
        "model_id": model_id,
        "version_id": version_id,
        "version_name": clean_cell(version.get("name")),
        "base_model": base_model,
        "creator": creator_name,
        "url": url,
        "model_description": html_to_text(parent_model.get("description")) if fetch_descriptions else "",
        "version_description": html_to_text(version.get("description")) if fetch_descriptions else "",
    }

    if warning:
        return Resolution("partial", reason=f"{hash_reason}; {assessment.reason}; gallery warning: {warning}", **common)

    status = "ok" if assessment.explicit else "ok_unlabeled"
    return Resolution(status, reason=f"{hash_reason}; {assessment.reason}", **common)


# ------------------------------------------------------------------ scanning


def find_networks(root: Path, *, recursive: bool = True, extensions: Iterable[str] = SAFETENSORS_EXTENSIONS) -> list[Path]:
    suffixes = {suffix.lower() for suffix in extensions}
    iterator = root.rglob("*") if recursive else root.glob("*")
    try:
        paths = [path for path in iterator if path.is_file() and path.suffix.lower() in suffixes]
    except OSError:
        return []
    return sorted(paths, key=lambda path: path.relative_to(root).as_posix().casefold())


@dataclass
class ScanReport:
    root: str = ""
    output: str = ""
    total: int = 0
    processed: int = 0
    skipped: int = 0
    resolved: int = 0
    failed: int = 0
    cancelled: bool = False
    # True once Civitai has answered a lookup, which is what tells the caller the
    # credentials this run used were accepted rather than rejected.
    api_answered: bool = False
    counts: dict[str, int] = field(default_factory=dict)
    rows: list[dict[str, str]] = field(default_factory=list)


def read_existing_rows(path: Path) -> dict[str, dict[str, str]]:
    """Existing CSV rows, keyed by their ``safetensor_file`` value."""
    if not path.is_file():
        return {}

    rows: dict[str, dict[str, str]] = {}
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            with path.open("r", encoding=encoding, newline="") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames:
                    return {}
                for row in reader:
                    cleaned = {key: clean_cell(value) for key, value in row.items() if key}
                    key = cleaned.get("safetensor_file", "")
                    if key:
                        rows[key] = cleaned
            return rows
        except UnicodeDecodeError:
            rows = {}
            continue
        except (OSError, csv.Error):
            return {}
    return rows


# Per-prompt columns, emitted adjacent to each other so the CSV stays readable:
# positive_prompt_1, prompt_image_url_1, prompt_image_id_1, positive_prompt_2, ...
PROMPT_COLUMN_PREFIXES = ("positive_prompt_", "prompt_image_url_", "prompt_image_id_")
PROMPT_COLUMN_RE = re.compile(r"^(?:positive_prompt|prompt_image_url|prompt_image_id)_(\d+)$")


def write_csv(output_path: Path, rows: list[dict[str, str]]) -> None:
    """Atomically write the CSV: the destination is replaced only on success."""
    prompt_count = 0
    for row in rows:
        for key in row:
            match = PROMPT_COLUMN_RE.match(key)
            if match:
                prompt_count = max(prompt_count, int(match.group(1)))
    prompt_count = max(1, prompt_count)

    extra_columns: list[str] = []
    for row in rows:
        for key in row:
            if key in BASE_COLUMNS or PROMPT_COLUMN_RE.match(key):
                continue
            if key not in extra_columns:
                extra_columns.append(key)

    prompt_columns = [f"{prefix}{index}" for index in range(1, prompt_count + 1) for prefix in PROMPT_COLUMN_PREFIXES]
    headers = [*BASE_COLUMNS, *extra_columns, *prompt_columns]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=output_path.parent,
            prefix=output_path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_name = handle.name
            writer = csv.DictWriter(handle, fieldnames=headers, extrasaction="ignore", quoting=csv.QUOTE_MINIMAL)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in headers})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, output_path)
        temp_name = None
    finally:
        if temp_name:
            try:
                os.unlink(temp_name)
            except OSError:
                pass


def row_for(relative_name: str, sha256: str, addnet: str, resolution: Resolution) -> dict[str, str]:
    row: dict[str, str] = {
        "safetensor_file": relative_name,
        "sha256": sha256,
        "addnet_hash": addnet or "",
        "civitai_model_name": resolution.model_name,
        "civitai_model_id": str(resolution.model_id) if resolution.model_id else "",
        "civitai_version_id": str(resolution.version_id) if resolution.version_id else "",
        "civitai_version_name": resolution.version_name,
        "base_model": resolution.base_model,
        "civitai_creator": resolution.creator,
        "civitai_url": resolution.url,
        "trigger_words": ", ".join(resolution.trigger_words),
        "status": resolution.status,
        "note": resolution.reason,
        "model_description": resolution.model_description,
        "version_description": resolution.version_description,
    }
    for index, entry in enumerate(resolution.prompts, start=1):
        row[f"positive_prompt_{index}"] = entry.prompt
        if entry.image_url:
            row[f"prompt_image_url_{index}"] = entry.image_url
        if entry.image_id:
            row[f"prompt_image_id_{index}"] = entry.image_id
    return row


def merge_row(existing: dict[str, str] | None, fresh: dict[str, str]) -> dict[str, str]:
    """Overlay a freshly fetched row on the one already in the CSV.

    Columns this tool owns are always taken from the fresh row - a re-fetch that
    returned fewer prompts must not leave stale ones behind. Columns it does not
    own (anything a user added by hand) are carried across, so re-running a fetch
    to pick up missing text details never costs someone their own notes.
    """
    if not existing:
        return fresh

    merged = dict(fresh)
    for key, value in existing.items():
        if not value or key in BASE_COLUMNS or PROMPT_COLUMN_RE.match(key):
            continue
        merged.setdefault(key, value)
    return merged


def scan_folder(
    root,
    output_path,
    *,
    api_key: str | None = None,
    timeout: float = 30.0,
    request_delay: float = 0.15,
    max_images: int = 0,
    recursive: bool = True,
    base_model_filter: str = "",
    require_explicit_label: bool = False,
    fetch_descriptions: bool = True,
    skip_hashes: set[str] | None = None,
    keep_existing_rows: bool = True,
    cancel: threading.Event | None = None,
    log: Callable[[str], None] | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> ScanReport:
    """Hash every network under ``root``, resolve it on Civitai, write a CSV."""
    root = Path(root).expanduser().resolve()
    output_path = Path(output_path).expanduser()

    def emit(message: str) -> None:
        if log is not None:
            log(message)

    report = ScanReport(root=str(root), output=str(output_path))

    if not root.is_dir():
        emit(f"Not a directory: {root}")
        report.failed = 1
        return report

    files = find_networks(root, recursive=recursive)
    report.total = len(files)
    emit(f"Found {len(files)} safetensors file(s) under {root}")
    emit(
        "Collecting trigger words, gallery prompts and model descriptions (text only)"
        if fetch_descriptions
        else "Collecting trigger words and gallery prompts; descriptions are turned off"
    )

    existing = read_existing_rows(output_path) if keep_existing_rows else {}
    if existing:
        emit(f"Merging with {len(existing)} row(s) already in {output_path.name}")

    skip_hashes = skip_hashes or set()
    if skip_hashes and not keep_existing_rows:
        emit("Rewriting the CSV from scratch, so nothing is skipped.")
        skip_hashes = set()
    rows_by_file: dict[str, dict[str, str]] = dict(existing)
    resolution_cache: dict[str, Resolution] = {}

    client = CivitaiClient(api_key=api_key, timeout=timeout, request_delay=request_delay, cancel=cancel)

    try:
        for index, path in enumerate(files, start=1):
            if cancel is not None and cancel.is_set():
                report.cancelled = True
                emit("Cancelled by the operator.")
                break

            relative_name = path.relative_to(root).as_posix()
            if progress is not None:
                progress(index, len(files), relative_name)

            sha256 = sha256_file(path, cancel=cancel)
            if not sha256:
                if cancel is not None and cancel.is_set():
                    report.cancelled = True
                    break
                report.failed += 1
                report.counts["hash_error"] = report.counts.get("hash_error", 0) + 1
                emit(f"[{index}/{len(files)}] {relative_name}: could not hash the file")
                continue

            addnet = addnet_sha256_file(path, cancel=cancel) or ""

            if sha256 in skip_hashes:
                report.skipped += 1
                continue

            resolution = resolution_cache.get(sha256)
            if resolution is None:
                local_metadata, tensor_names = read_safetensors_header(path)
                try:
                    resolution = resolve_hash(
                        sha256,
                        local_metadata,
                        client,
                        base_model_filter=base_model_filter,
                        require_explicit_label=require_explicit_label,
                        max_images=max_images,
                        fetch_descriptions=fetch_descriptions,
                        tensor_names=tensor_names,
                    )
                except Cancelled:
                    report.cancelled = True
                    emit("Cancelled by the operator.")
                    break
                resolution_cache[sha256] = resolution

            report.processed += 1
            report.counts[resolution.status] = report.counts.get(resolution.status, 0) + 1
            # "error" is the one status that means the request itself did not go
            # through - a rejected API key lands there - so any other status is
            # proof that Civitai answered the key this run was given.
            if resolution.status != "error":
                report.api_answered = True
            rows_by_file[relative_name] = merge_row(
                existing.get(relative_name),
                row_for(relative_name, sha256, addnet, resolution),
            )

            if resolution.resolved:
                report.resolved += 1
                emit(f"[{index}/{len(files)}] {relative_name}: {resolution.model_name or '(unnamed)'} — {len(resolution.trigger_words)} trigger word(s), {len(resolution.prompts)} prompt(s)")
            else:
                report.failed += 1
                emit(f"[{index}/{len(files)}] {relative_name}: {resolution.status} — {resolution.reason}")
    finally:
        client.close()
        import mc_lora_hashing

        mc_lora_hashing.flush()

    report.rows = [rows_by_file[key] for key in sorted(rows_by_file, key=str.casefold)]
    write_csv(output_path, report.rows)
    emit(f"Wrote {len(report.rows)} row(s) to {output_path}")
    return report
