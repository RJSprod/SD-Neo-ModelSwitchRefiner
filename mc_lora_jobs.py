"""Background jobs.

Scans run on a worker thread so the WebUI stays responsive; the UI only ever
reads a snapshot of the job state, never the job objects themselves.
"""

from __future__ import annotations

import itertools
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from mc_lora_common import log, report

MAX_LOG_LINES = 400

QUEUED = "queued"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
CANCELLED = "cancelled"

FINISHED_STATES = (DONE, FAILED, CANCELLED)


@dataclass
class JobSnapshot:
    id: int
    kind: str
    title: str
    state: str
    created_at: float
    started_at: float
    finished_at: float
    current: int
    total: int
    message: str
    summary: str
    error: str
    log: list[str] = field(default_factory=list)

    @property
    def active(self) -> bool:
        return self.state in (QUEUED, RUNNING)

    @property
    def percent(self) -> int:
        if self.total <= 0:
            return 0
        return max(0, min(100, int(round(self.current * 100 / self.total))))

    @property
    def elapsed(self) -> float:
        if not self.started_at:
            return 0.0
        end = self.finished_at or time.time()
        return max(0.0, end - self.started_at)


class Job:
    _ids = itertools.count(1)

    def __init__(self, kind: str, title: str, target):
        self.id = next(Job._ids)
        self.kind = kind
        self.title = title
        self.target = target

        self.state = QUEUED
        self.created_at = time.time()
        self.started_at = 0.0
        self.finished_at = 0.0

        self.current = 0
        self.total = 0
        self.message = "Waiting to start…"
        self.summary = ""
        self.error = ""

        self.cancel_event = threading.Event()
        self._log: deque[str] = deque(maxlen=MAX_LOG_LINES)
        self._lock = threading.RLock()

    # ------------------------------------------------------- worker-side api

    def log_line(self, text: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        with self._lock:
            for line in str(text).splitlines() or [""]:
                self._log.append(f"[{stamp}] {line}")

    def set_progress(self, current: int, total: int, message: str = "") -> None:
        with self._lock:
            self.current = int(current)
            self.total = int(total)
            if message:
                self.message = message

    def set_message(self, message: str) -> None:
        with self._lock:
            self.message = message

    @property
    def cancelled(self) -> bool:
        return self.cancel_event.is_set()

    # --------------------------------------------------------------- control

    def cancel(self) -> None:
        self.cancel_event.set()
        with self._lock:
            if self.state == QUEUED:
                self.state = CANCELLED
                self.finished_at = time.time()
                self.message = "Cancelled before it started."
            elif self.state == RUNNING:
                self.message = "Stopping…"

    def snapshot(self) -> JobSnapshot:
        with self._lock:
            return JobSnapshot(
                id=self.id,
                kind=self.kind,
                title=self.title,
                state=self.state,
                created_at=self.created_at,
                started_at=self.started_at,
                finished_at=self.finished_at,
                current=self.current,
                total=self.total,
                message=self.message,
                summary=self.summary,
                error=self.error,
                log=list(self._log),
            )


class JobManager:
    """One worker thread, one queue - scans never run concurrently."""

    def __init__(self, history: int = 20):
        self._lock = threading.RLock()
        self._queue: deque[Job] = deque()
        self._jobs: deque[Job] = deque(maxlen=history)
        self._current: Job | None = None
        self._worker: threading.Thread | None = None
        self._wake = threading.Event()

    # ---------------------------------------------------------------- submit

    def submit(self, kind: str, title: str, target) -> Job:
        job = Job(kind, title, target)
        with self._lock:
            self._queue.append(job)
            self._jobs.append(job)
            self._ensure_worker()
        self._wake.set()
        log(f"queued job #{job.id} ({kind})")
        return job

    def _ensure_worker(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._worker = threading.Thread(target=self._run, name="ExtendedLoraDetails", daemon=True)
        self._worker.start()

    def _run(self) -> None:
        while True:
            with self._lock:
                job = self._queue.popleft() if self._queue else None
                self._current = job
                if job is None:
                    self._worker = None
                    return

            if job.cancelled:
                continue

            with job._lock:
                job.state = RUNNING
                job.started_at = time.time()
                job.message = "Starting…"

            try:
                summary = job.target(job)
                with job._lock:
                    if job.cancelled:
                        job.state = CANCELLED
                        job.message = "Cancelled."
                    else:
                        job.state = DONE
                        job.message = "Finished."
                    job.summary = str(summary or "")
            except Exception as exc:  # noqa: BLE001 - surfaced in the UI
                report(f"job #{job.id} ({job.kind}) failed")
                with job._lock:
                    job.state = FAILED
                    job.error = f"{type(exc).__name__}: {exc}"
                    job.message = "Failed."
                job.log_line(f"ERROR {type(exc).__name__}: {exc}")
            finally:
                with job._lock:
                    job.finished_at = time.time()
                with self._lock:
                    self._current = None

    # ----------------------------------------------------------------- query

    @property
    def current(self) -> JobSnapshot | None:
        with self._lock:
            job = self._current
        return job.snapshot() if job is not None else None

    def snapshots(self) -> list[JobSnapshot]:
        with self._lock:
            jobs = list(self._jobs)
        return [job.snapshot() for job in reversed(jobs)]

    def active_snapshots(self) -> list[JobSnapshot]:
        return [snapshot for snapshot in self.snapshots() if snapshot.active]

    def get(self, job_id: int) -> Job | None:
        with self._lock:
            for job in self._jobs:
                if job.id == job_id:
                    return job
        return None

    def has_active(self) -> bool:
        return bool(self.active_snapshots())

    def cancel(self, job_id: int) -> bool:
        job = self.get(job_id)
        if job is None:
            return False
        job.cancel()
        return True

    def cancel_all(self) -> int:
        count = 0
        for snapshot in self.active_snapshots():
            if self.cancel(snapshot.id):
                count += 1
        return count

    def clear_finished(self) -> int:
        with self._lock:
            keep = [job for job in self._jobs if job.state not in FINISHED_STATES]
            removed = len(self._jobs) - len(keep)
            self._jobs.clear()
            self._jobs.extend(keep)
        return removed


MANAGER = JobManager()


# ------------------------------------------------------------- job factories


def submit_scan(
    folder,
    *,
    output_name: str = "",
    api_key: str = "",
    request_delay: float = 0.15,
    timeout: float = 30.0,
    max_images: int = 0,
    recursive: bool = True,
    base_model_filter: str = "",
    require_explicit_label: bool = False,
    incremental: bool = True,
    retry_unresolved: bool = False,
    backfill_text: bool = True,
    fetch_descriptions: bool = True,
    keep_existing_rows: bool = True,
) -> Job:
    """Queue a Civitai fetch for one folder, writing into the library."""
    import mc_lora_civitai
    import mc_lora_last_used
    from mc_lora_library import get_library

    folder_path = Path(folder).expanduser()
    library = get_library()

    # Remembered when the job is queued rather than when it finishes: the tab
    # should reopen on the folder you last pointed it at, cancelled run or not.
    mc_lora_last_used.remember_scan_folder(folder_path)

    name = (output_name or "").strip()
    if not name:
        stem = folder_path.name or "loras"
        name = f"civitai-{stem}"
    name = Path(name).name
    if not name.lower().endswith(".csv"):
        name += ".csv"
    output_path = library.directory / name

    def target(job: Job) -> str:
        job.log_line(f"Scanning {folder_path}")
        job.log_line(f"Writing to {output_path}")

        # Backfilling only makes sense for details this run would actually
        # collect, so it follows the description switch.
        backfill = backfill_text and fetch_descriptions
        skip_hashes = (
            library.known_sha256(resolved_only=retry_unresolved, require_description=backfill)
            if incremental
            else set()
        )
        if incremental:
            detail = "resolved LoRAs only" if retry_unresolved else "every LoRA already recorded, resolved or not"
            if backfill:
                detail += ", except rows still missing their text details"
            job.log_line(f"Incremental: skipping {len(skip_hashes)} known hash(es) — {detail}")

        def progress(current: int, total: int, name: str) -> None:
            job.set_progress(current, total, f"{current}/{total} · {name}")

        result = mc_lora_civitai.scan_folder(
            folder_path,
            output_path,
            api_key=api_key or None,
            timeout=timeout,
            request_delay=request_delay,
            max_images=max_images,
            recursive=recursive,
            base_model_filter=base_model_filter,
            require_explicit_label=require_explicit_label,
            fetch_descriptions=fetch_descriptions,
            skip_hashes=skip_hashes,
            keep_existing_rows=keep_existing_rows,
            cancel=job.cancel_event,
            log=job.log_line,
            progress=progress,
        )

        # Only a key Civitai actually answered is worth keeping: a run that
        # never got past a rejected one must not replace the key that worked.
        if api_key and result.api_answered:
            mc_lora_last_used.remember_api_key(api_key)

        library.reload()

        parts = [
            f"{result.total} file(s) found",
            f"{result.processed} looked up",
            f"{result.resolved} resolved",
            f"{result.skipped} skipped",
        ]
        if result.failed:
            parts.append(f"{result.failed} unresolved")
        if result.cancelled:
            parts.append("cancelled early")
        return f"{output_path.name}: " + ", ".join(parts)

    return MANAGER.submit("scan", f"Civitai fetch · {folder_path.name or folder_path}", target)


def submit_image_downloads(entries, *, label: str = "") -> Job:
    """Queue gallery-image downloads for a list of ``(url, image_id)`` pairs."""
    import mc_lora_images

    pairs = [(str(url), str(image_id or "")) for url, image_id in entries if str(url or "").strip()]
    title = f"Download images · {label}" if label else "Download images"

    def target(job: Job) -> str:
        folder = mc_lora_images.image_dir()
        job.log_line(f"{len(pairs)} image(s) into {folder}")

        saved = skipped = failed = 0
        for index, (url, image_id) in enumerate(pairs, start=1):
            if job.cancelled:
                break
            job.set_progress(index, len(pairs), f"{index}/{len(pairs)}")
            result = mc_lora_images.download(url, image_id, directory=folder, cancel=job.cancel_event)
            if result.skipped:
                skipped += 1
            elif result.ok:
                saved += 1
                job.log_line(result.message)
            else:
                failed += 1
                job.log_line(f"{url[:90]}: {result.message}")

        parts = [f"{saved} downloaded", f"{skipped} already present"]
        if failed:
            parts.append(f"{failed} failed")
        return ", ".join(parts)

    return MANAGER.submit("images", title, target)


def submit_prehash(folders=None) -> Job:
    """Queue SHA256 precomputation so the details panel never blocks on it."""
    import mc_lora_hashing
    from mc_lora_common import NETWORK_EXTENSIONS, lora_directories

    directories = [Path(entry) for entry in (folders or lora_directories())]

    def target(job: Job) -> str:
        files: list[Path] = []
        for directory in directories:
            if not directory.is_dir():
                job.log_line(f"Skipping missing directory {directory}")
                continue
            for path in sorted(directory.rglob("*")):
                if path.is_file() and path.suffix.lower() in NETWORK_EXTENSIONS:
                    files.append(path)

        job.log_line(f"{len(files)} network file(s) to hash")
        hashed = cached = 0

        for index, path in enumerate(files, start=1):
            if job.cancelled:
                break
            job.set_progress(index, len(files), f"{index}/{len(files)} · {path.name}")
            if mc_lora_hashing.known_hash(path):
                cached += 1
                continue
            if mc_lora_hashing.sha256_file(path, cancel=job.cancel_event):
                hashed += 1
            if index % 25 == 0:
                mc_lora_hashing.flush()

        mc_lora_hashing.flush()
        return f"{hashed} newly hashed, {cached} already cached, {len(files)} total"

    return MANAGER.submit("prehash", "Precompute LoRA hashes", target)
