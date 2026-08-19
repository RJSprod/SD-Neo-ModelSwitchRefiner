from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import os
import time
import httpx

from .manifest import Component
from .verifier import verify

# The model is 16-27 GiB in one file, which is long enough on any connection
# that a dropped transfer is ordinary rather than exceptional: an HTTP/2 reset,
# a CDN closing a stream, a Wi-Fi handover. Every attempt resumes from the bytes
# already on disk, so the budget below is spent only on attempts that transfer
# nothing at all — one that moves any bytes earns a fresh budget, and a download
# that keeps inching forward is never abandoned.
RETRIES_WITHOUT_PROGRESS = 5
BACKOFF_SECONDS = (2, 4, 8, 16, 30)
RETRYABLE_STATUS = frozenset({408, 416, 425, 429, 500, 502, 503, 504})
# Also the resume granularity: a drop loses at most the block being filled.
CHUNK_BYTES = 1024 * 1024

Progress = Callable[[int, int], None]
Notice = Callable[[str], None]


def _client() -> httpx.Client:
    # The read timeout is per block, not for the whole transfer, so it is the
    # length of silence that counts as a stall. A minute is generous for a CDN
    # that is still sending; ten would be ten minutes of a progress bar that
    # has stopped moving before the retry that fixes it.
    return httpx.Client(follow_redirects=True, timeout=httpx.Timeout(60, connect=30))


def _retryable(error: httpx.HTTPError) -> bool:
    # TransportError covers the timeouts, resets and incomplete bodies that a
    # long transfer runs into; a 5xx or 429 is the same thing said politely.
    if isinstance(error, httpx.TransportError): return True
    return isinstance(error, httpx.HTTPStatusError) and error.response.status_code in RETRYABLE_STATUS


def _range_rejected(error: httpx.HTTPError) -> bool:
    return isinstance(error, httpx.HTTPStatusError) and error.response.status_code == 416


def _attempt(component: Component, partial: Path, progress: Progress | None) -> None:
    """One transfer, continuing ``partial`` from wherever it stopped."""
    existing = partial.stat().st_size if partial.is_file() else 0
    if component.size is not None and existing >= component.size:
        # A part file at or past the full size cannot be resumed — asking for a
        # range beyond the end is a 416, and every later attempt would repeat it.
        partial.unlink(); existing = 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    with _client() as client:
        with client.stream("GET", component.url, headers=headers) as response:
            response.raise_for_status()
            append = existing > 0 and response.status_code == 206
            if existing and not append: existing = 0  # the server ignored the range
            with partial.open("ab" if append else "wb") as stream:
                done = existing
                for chunk in response.iter_bytes(CHUNK_BYTES):
                    stream.write(chunk); done += len(chunk)
                    if progress: progress(done, component.size or max(done, 1))
                stream.flush(); os.fsync(stream.fileno())


def download(component: Component, destination: Path, progress: Progress | None = None,
             notice: Notice | None = None) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True); partial = destination.with_name(destination.name + ".part")
    if destination.is_file():
        try:
            verify(destination, component.size, component.sha256)
            if progress: progress(component.size or destination.stat().st_size, component.size or destination.stat().st_size)
            return destination
        except (OSError, ValueError):
            destination.unlink(missing_ok=True)

    retries = 0
    while True:
        before = partial.stat().st_size if partial.is_file() else 0
        try:
            _attempt(component, partial, progress); break
        except httpx.HTTPError as error:
            if not _retryable(error): raise
            # A rejected range means the part file is longer than the file the
            # server will send — a re-uploaded release — and every resume from
            # it repeats the 416. It can only be started again.
            if _range_rejected(error): partial.unlink(missing_ok=True)
            after = partial.stat().st_size if partial.is_file() else 0
            retries = 1 if after > before else retries + 1
            if retries > RETRIES_WITHOUT_PROGRESS: raise
            pause = BACKOFF_SECONDS[min(retries, len(BACKOFF_SECONDS)) - 1]
            if notice: notice(f"interrupted after {after / 2 ** 20:.0f} MiB, retrying in {pause}s…")
            time.sleep(pause)

    try:
        verify(partial, component.size, component.sha256)
    except (OSError, ValueError):
        # Bytes that failed verification are never repaired by appending more,
        # and a full-size part file poisons every later resume. Drop it, so
        # re-running setup starts this component again instead of repeating.
        partial.unlink(missing_ok=True); raise
    os.replace(partial, destination); return destination
