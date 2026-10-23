"""Downloading gallery images for a prompt.

Everything lands in one flat folder (``images/`` under the LoRA data root by
default) so the collection can be managed in one place. A file's name is derived from the image's
Civitai id, falling back to a digest of its URL, which makes "have I already got
this one?" a plain filesystem question — delete a file and the UI offers to
download it again, exactly like it had never been fetched.
"""

from __future__ import annotations

import base64
import hashlib
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

import mc_lora_paths
from mc_lora_common import clean_cell, human_bytes, log, opt

DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_BYTES = 32 * 1024 * 1024
CHUNK_SIZE = 64 * 1024

# The gallery cards show a small copy of each downloaded image, embedded in the
# page: the image folder can live anywhere, and Gradio only serves files from
# the paths it was told about.
THUMB_SIZE = 384
THUMB_SOURCE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".avif"}
VIDEO_SUFFIXES = {".mp4", ".webm"}

# Extensions we are willing to write. Anything else is stored as .img rather
# than trusting a remote-supplied suffix.
KNOWN_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".avif", ".bmp", ".mp4", ".webm"}

CONTENT_TYPE_SUFFIXES = {
    "image/jpeg": ".jpeg",
    "image/jpg": ".jpeg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/avif": ".avif",
    "image/bmp": ".bmp",
    "video/mp4": ".mp4",
    "video/webm": ".webm",
}

_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9._-]+")


def default_image_dir() -> Path:
    return mc_lora_paths.data_root() / mc_lora_paths.IMAGES_DIRNAME


def image_dir() -> Path:
    """The configured image folder, created on demand."""
    path = mc_lora_paths.image_dir()
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        log(f"could not create the configured image directory ({exc.strerror or type(exc).__name__}); using the default")
        path = default_image_dir()
        path.mkdir(parents=True, exist_ok=True)
    return path


def _suffix_from_url(url: str) -> str:
    name = unquote(urlparse(url).path).rsplit("/", 1)[-1]
    suffix = os.path.splitext(name)[1].lower()
    return suffix if suffix in KNOWN_SUFFIXES else ""


def _suffix_from_content_type(content_type: str) -> str:
    return CONTENT_TYPE_SUFFIXES.get(content_type.split(";", 1)[0].strip().lower(), "")


def stem_for(url: str, image_id: str = "") -> str:
    """A stable, filesystem-safe base name for one gallery image."""
    identifier = _SAFE_ID_RE.sub("", clean_cell(image_id))
    if identifier:
        return f"civitai-{identifier}"
    digest = hashlib.sha256(clean_cell(url).encode("utf-8")).hexdigest()[:16]
    return f"url-{digest}"


def find_local(url: str, image_id: str = "", *, directory: Path | None = None) -> Path | None:
    """The downloaded file for this image, or ``None`` if it is not on disk.

    Answered from the filesystem every time, never from a manifest, so a file
    the user deleted simply reads as "not downloaded".
    """
    if not clean_cell(url):
        return None

    folder = directory or image_dir()
    stem = stem_for(url, image_id)

    preferred = _suffix_from_url(url)
    candidates = [preferred, *KNOWN_SUFFIXES, ".img"] if preferred else [*KNOWN_SUFFIXES, ".img"]
    for suffix in candidates:
        candidate = folder / f"{stem}{suffix}"
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None


@dataclass(frozen=True)
class DownloadResult:
    ok: bool
    path: Path | None
    message: str
    skipped: bool = False


def download(url: str, image_id: str = "", *, directory: Path | None = None, timeout: float | None = None, max_bytes: int | None = None, cancel=None) -> DownloadResult:
    """Fetch one gallery image into the image folder.

    Already-present files are reported as skipped rather than re-fetched.
    """
    url = clean_cell(url)
    if not url:
        return DownloadResult(False, None, "This prompt has no image URL in the library.")

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return DownloadResult(False, None, f"Refusing to fetch a non-HTTP URL: {url[:120]}")

    folder = directory or image_dir()
    existing = find_local(url, image_id, directory=folder)
    if existing is not None:
        return DownloadResult(True, existing, f"Already downloaded: {existing.name}", skipped=True)

    timeout = float(timeout if timeout is not None else opt("model_chain_lora_image_timeout", DEFAULT_TIMEOUT))
    max_bytes = int(max_bytes if max_bytes is not None else opt("model_chain_lora_image_max_mb", 32) * 1024 * 1024)
    max_bytes = max(1, max_bytes)

    try:
        import requests
    except ImportError:  # pragma: no cover - requests ships with the WebUI
        return DownloadResult(False, None, "The requests package is not installed.")

    headers = {"User-Agent": "SD-Neo-ModelSwitchRefiner/1.0", "Accept": "image/*,video/*;q=0.8,*/*;q=0.5"}
    api_key = str(opt("model_chain_lora_civitai_api_key", "") or "").strip()
    if api_key and parsed.hostname and parsed.hostname.lower().endswith("civitai.com"):
        headers["Authorization"] = f"Bearer {api_key}"

    temp_name: str | None = None
    try:
        with requests.get(url, stream=True, timeout=timeout, headers=headers) as response:
            if response.status_code != 200:
                return DownloadResult(False, None, f"HTTP {response.status_code} from {parsed.netloc}")

            content_type = response.headers.get("Content-Type", "")
            suffix = _suffix_from_content_type(content_type) or _suffix_from_url(url) or ".img"

            declared = response.headers.get("Content-Length")
            if declared and declared.isdigit() and int(declared) > max_bytes:
                return DownloadResult(False, None, f"Image is larger than the {human_bytes(max_bytes)} limit.")

            folder.mkdir(parents=True, exist_ok=True)
            written = 0
            with tempfile.NamedTemporaryFile(dir=folder, prefix=".mc-lora-", suffix=".part", delete=False) as handle:
                temp_name = handle.name
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if cancel is not None and cancel.is_set():
                        return DownloadResult(False, None, "Cancelled.")
                    if not chunk:
                        continue
                    written += len(chunk)
                    if written > max_bytes:
                        return DownloadResult(False, None, f"Image exceeded the {human_bytes(max_bytes)} limit.")
                    handle.write(chunk)

            if written == 0:
                return DownloadResult(False, None, "The server returned an empty response.")

            destination = folder / f"{stem_for(url, image_id)}{suffix}"
            os.replace(temp_name, destination)
            temp_name = None
            return DownloadResult(True, destination, f"Saved {destination.name} ({human_bytes(written)})")

    except requests.RequestException as exc:
        return DownloadResult(False, None, f"Download failed: {exc}")
    except OSError as exc:
        return DownloadResult(False, None, f"Could not write the file: {exc}")
    finally:
        if temp_name:
            try:
                os.unlink(temp_name)
            except OSError:
                pass


def is_video(path) -> bool:
    return Path(path).suffix.lower() in VIDEO_SUFFIXES


def thumbnail_dir() -> Path:
    return mc_lora_paths.cache_dir() / "thumbs"


def thumbnail(path, *, size: int = THUMB_SIZE, directory: Path | None = None) -> Path | None:
    """A small WebP copy of a downloaded image, made once and cached.

    The cache key covers the file's path, size and mtime, so a replaced image
    gets a fresh thumbnail. ``None`` for videos, unreadable files, or when
    Pillow is not available.
    """
    source = Path(path)
    if source.suffix.lower() not in THUMB_SOURCE_SUFFIXES:
        return None
    try:
        info = source.stat()
    except OSError:
        return None

    key = hashlib.sha256(f"{source.resolve()}|{info.st_size}|{info.st_mtime_ns}|{size}".encode("utf-8")).hexdigest()[:16]
    folder = directory or thumbnail_dir()
    target = folder / f"{source.stem}-{key}.webp"
    if target.is_file():
        return target

    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - Pillow ships with the WebUI
        return None

    temp_name: str | None = None
    try:
        folder.mkdir(parents=True, exist_ok=True)
        with Image.open(source) as image:
            image.thumbnail((size, size))
            if image.mode not in ("RGB", "RGBA"):
                image = image.convert("RGBA" if "A" in image.getbands() else "RGB")
            with tempfile.NamedTemporaryFile(dir=folder, prefix=".mc-lora-", suffix=".webp", delete=False) as handle:
                temp_name = handle.name
                image.save(handle, "WEBP", quality=78)
        os.replace(temp_name, target)
        temp_name = None
        return target
    except Exception as exc:  # noqa: BLE001 - a card without a picture is fine
        log(f"could not make a thumbnail of a gallery image ({type(exc).__name__})")
        return None
    finally:
        if temp_name:
            try:
                os.unlink(temp_name)
            except OSError:
                pass


def thumbnail_data_uri(path, *, size: int = THUMB_SIZE, directory: Path | None = None) -> str:
    """The thumbnail as a ``data:`` URI, or ``""`` when there is none."""
    thumb = thumbnail(path, size=size, directory=directory)
    if thumb is None:
        return ""
    try:
        payload = thumb.read_bytes()
    except OSError:
        return ""
    return "data:image/webp;base64," + base64.b64encode(payload).decode("ascii")


def stats(directory: Path | None = None) -> dict:
    """Count and total size of the image folder."""
    folder = directory or image_dir()
    count = 0
    total = 0
    try:
        for entry in os.scandir(folder):
            if entry.is_file() and not entry.name.startswith(".mc-lora-"):
                count += 1
                total += entry.stat().st_size
    except OSError:
        pass
    return {"directory": str(folder), "files": count, "bytes": total}


def purge(directory: Path | None = None) -> tuple[int, str]:
    """Delete every downloaded image. Returns ``(removed, message)``."""
    folder = directory or image_dir()
    removed = 0
    failed = 0
    try:
        entries = list(os.scandir(folder))
    except OSError as exc:
        return 0, f"Could not read the image folder: {exc}"

    for entry in entries:
        if not entry.is_file():
            continue
        try:
            os.unlink(entry.path)
            removed += 1
        except OSError:
            failed += 1

    # The cached thumbnails are copies of what was just deleted.
    if directory is None:
        try:
            for entry in os.scandir(thumbnail_dir()):
                if entry.is_file():
                    try:
                        os.unlink(entry.path)
                    except OSError:
                        pass
        except OSError:
            pass

    message = f"Deleted {removed} file(s) from {folder}."
    if failed:
        message += f" {failed} could not be removed."
    return removed, message
