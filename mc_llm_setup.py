"""Getting a llama.cpp runtime in place, from inside Forge.

The vendored provisioning pipeline can install one, and its console front end
(``prompt_master/setup_cli.py``) still drives it. What was missing was a way to
do it from the tab -- which left the panel able to reach a state whose only
recovery instruction, inherited from the standalone application, was "Run
Models and Hardware setup first". There is no such thing in Forge, so that
sentence was a dead end. This module is the answer to it.

Three routes in, because installations differ:

1. **Detect.** A build already sitting under ``<root>/runtime/`` -- extracted
   there by hand, or left behind by a standalone install this extension was
   pointed at -- is found and recorded. Costs nothing and is tried first.
2. **Adopt.** A llama.cpp build already on the machine is placed under the
   install root and recorded. This is the route that works everywhere.
3. **Download.** The pinned build from ``release-manifest.json``, verified by
   SHA-256 and extracted atomically, exactly as the standalone installer does
   it. The manifest pins Windows x64 archives only, so this route is offered
   where it exists and explained where it does not.

Why adopting copies rather than linking
---------------------------------------
``AppPaths.contained`` requires the runtime to resolve to a path inside the
install root, and it is right to: the runtime is a *program this extension
starts*, unlike the weights, which are a file it reads. That is upstream's
distinction and this module keeps it. ``resolve()`` follows symlinks, so a
symlink into the root would resolve back outside it and be refused -- which
leaves copying as the only honest way to satisfy the rule.

What gets copied depends on what was pointed at. A llama.cpp release directory
holds the server *and* the shared libraries it loads, so the whole directory
comes across. A distribution-packaged ``llama-server`` on the system path finds
its libraries through the system loader and does not have a directory worth
copying, so only the executable does -- and the difference is reported, because
the second case works only as long as those system libraries stay installed.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import mc_llm_files
import mc_llm_paths

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

_MB = 1024**2

SERVER_NAMES = ("llama-server.exe", "llama-server")
"""What llama.cpp calls its server, in the two spellings a release uses."""

RUNTIME_DIRNAME = "runtime"
"""Where a runtime lives under the install root. The vendored installer's own
choice, matched so a build placed by either route is found by both."""

MAX_ADOPT_BYTES = 6 * 1024**3
MAX_ADOPT_ENTRIES = 4000
"""Ceilings on a directory adopt.

A llama.cpp release with CUDA libraries runs to a couple of gigabytes and a few
hundred files. These are not that limit -- they are the guard against being
pointed at something that is not a release directory at all. Somebody whose
``llama-server`` lives in ``/usr/bin`` should get the single-file route and a
sentence about it, not a copy of ``/usr/bin``.
"""

# Files that mark a directory as a llama.cpp build rather than a system
# location that merely happens to contain the server. Matched case-insensitively
# against the whole name, as a prefix, so llama.dll, libllama.so, libllama.dylib
# and the sibling llama-cli all count.
BUILD_MARKERS = ("llama", "ggml")


class SetupError(RuntimeError):
    """Something a user can act on, phrased as a sentence for the panel."""


# --------------------------------------------------------------------------- #
# What is present
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RuntimeStatus:
    """What this installation has, and what it can do about what it has not."""

    recorded: Path | None
    """The executable the state file names, if it exists and is contained."""
    found: Path | None
    """A server under ``<root>/runtime/``, recorded or not."""
    downloadable: bool
    """Whether the manifest pins a build for this platform."""
    detail: str = ""

    @property
    def ready(self) -> bool:
        return self.recorded is not None

    @property
    def adoptable(self) -> bool:
        """There is a build in place that is simply not recorded yet."""
        return self.recorded is None and self.found is not None


def status() -> RuntimeStatus:
    """What the runtime situation is. Never raises; the panel renders this."""
    can_download = downloadable()
    return RuntimeStatus(
        recorded=recorded_runtime(),
        found=detect(),
        downloadable=can_download,
        detail="" if can_download else _no_download_reason(),
    )


def recorded_runtime() -> Path | None:
    """The executable the state file names, if it resolves and still exists."""
    from prompt_master.core.config import read_json

    paths = mc_llm_paths.app_paths()
    try:
        recorded = str(read_json(paths.state_file).get("runtime") or "")
    except (OSError, ValueError):
        return None
    if not recorded:
        return None
    try:
        candidate = paths.contained(recorded)
    except ValueError:
        # A state file naming a runtime outside the root. Treated as absent
        # rather than repaired: it is the one path here that would start a
        # program from somewhere this extension does not own.
        logger.warning("Model Chain: the recorded llama.cpp runtime is outside the LLM data "
                       "directory and will not be used: %s", recorded)
        return None
    return candidate if candidate.is_file() else None


def detect() -> Path | None:
    """A llama-server under ``<root>/runtime/``, whether or not it is recorded."""
    directory = mc_llm_paths.data_root() / RUNTIME_DIRNAME
    if not directory.is_dir():
        return None
    for name in SERVER_NAMES:
        matches = sorted(directory.rglob(name))
        if matches:
            return matches[0]
    return None


def downloadable() -> bool:
    """Whether ``release-manifest.json`` pins a build this machine can run."""
    return sys.platform == "win32"


def _no_download_reason() -> str:
    return (
        "The bundled release manifest pins Windows x64 llama.cpp builds only, so there is "
        "nothing to download for this platform. Install or unpack a llama.cpp release "
        "yourself and point the box above at its llama-server binary — that route works "
        "everywhere and installs exactly the same thing."
    )


# --------------------------------------------------------------------------- #
# Adopting a build that is already on the machine
# --------------------------------------------------------------------------- #


def adopt(source: str | Path) -> tuple[Path, str]:
    """Place an existing llama.cpp build under the install root.

    ``source`` may be the ``llama-server`` executable or the directory holding
    it. Returns the executable's new path and a sentence describing what was
    done, which the panel shows -- the single-file case in particular has a
    caveat worth reading.
    """
    # Cleaned rather than trusted: this is reached from a text box, and the
    # obvious way to get a path into one on Windows adds quotes around it. See
    # mc_llm_files.
    chosen = mc_llm_files.to_path(source)
    if chosen is None:
        raise SetupError("Enter the path to llama-server, or to the folder holding it.")
    if not chosen.exists():
        raise SetupError(f"There is nothing at {chosen}")

    executable = _server_in(chosen) if chosen.is_dir() else chosen
    if executable is None:
        raise SetupError(f"{chosen} contains no llama-server executable")
    if executable.name not in SERVER_NAMES:
        raise SetupError(
            f"{executable.name} is not a llama.cpp server. Point this at llama-server"
            f"{'.exe' if sys.platform == 'win32' else ''}, or at the folder holding it."
        )

    paths = mc_llm_paths.app_paths()
    root = paths.root.resolve()
    resolved = executable.resolve()

    if root == resolved or root in resolved.parents:
        # Already inside the install root -- nothing to copy, and copying would
        # duplicate a build the user deliberately put there.
        return resolved, f"Using the llama.cpp build already in place at {resolved}."

    destination = paths.root / RUNTIME_DIRNAME
    directory = resolved.parent
    if _is_build_directory(directory):
        _check_copyable(directory)
        placed = _copy_tree(directory, destination)
        return (_server_in(placed) or placed / resolved.name,
                f"Copied the llama.cpp build from {directory} into {destination}.")

    destination.mkdir(parents=True, exist_ok=True)
    target = destination / resolved.name
    shutil.copy2(resolved, target)
    _make_executable(target)
    return target, (
        f"Copied {resolved.name} into {destination}. Its folder did not look like a "
        f"llama.cpp release, so only the executable was taken — it will load its shared "
        f"libraries from the system, and will stop working if that package is removed."
    )


def _server_in(directory: Path) -> Path | None:
    for name in SERVER_NAMES:
        matches = sorted(directory.rglob(name))
        if matches:
            return matches[0]
    return None


def _is_build_directory(directory: Path) -> bool:
    """Whether ``directory`` looks like a llama.cpp release rather than /usr/bin.

    The test is for siblings that belong to the same build -- other ``llama-*``
    tools, ``libllama``/``llama.dll``, the ggml libraries. A system binary
    directory holds the server and nothing else that matches.
    """
    try:
        siblings = list(os.scandir(directory))
    except OSError:
        return False

    related = 0
    for entry in siblings:
        name = entry.name.casefold()
        if name in {n.casefold() for n in SERVER_NAMES}:
            continue
        stem = name[3:] if name.startswith("lib") else name
        if stem.startswith(BUILD_MARKERS):
            related += 1
    # More than one related file, and not a directory so large it cannot be a
    # release: a llama.cpp build ships the server beside its libraries and its
    # sibling tools, and /usr/bin ships thousands of unrelated things.
    return related >= 2 and len(siblings) <= MAX_ADOPT_ENTRIES


def _check_copyable(directory: Path) -> None:
    total = 0
    count = 0
    for root, _dirs, files in os.walk(directory):
        for name in files:
            count += 1
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                continue
            if total > MAX_ADOPT_BYTES or count > MAX_ADOPT_ENTRIES:
                raise SetupError(
                    f"{directory} is larger than a llama.cpp release should be "
                    f"({total / 1024**3:.1f} GB in {count}+ files). Point this at the "
                    f"release's own folder rather than at a folder containing it."
                )


def in_use_error(destination: Path, exc: BaseException) -> SetupError:
    """The sentence for a runtime folder that cannot be replaced.

    Windows will not let a file be renamed or deleted while a process has it
    open, and a running llama-server holds every DLL beside it -- so replacing
    the runtime under a server that is still up fails with "[WinError 5] Access
    is denied: ...\\cublas64_12.dll", which names a CUDA library and reads like
    a driver problem rather than like the file lock it is. Callers stop the
    server first; this is what is said when something else is holding it.
    """
    return SetupError(
        f"{destination} is in use and could not be replaced ({exc}). Something still has "
        f"the runtime open — press Unload in LLM Studio, close any other copy of the WebUI, "
        f"and try again."
    )


def _copy_tree(source: Path, destination: Path) -> Path:
    """Copy ``source`` into ``destination``, replacing what was there.

    Through a staging directory and a rename, for the reason
    ``extract_zips_atomic`` uses one: a half-copied runtime that a later launch
    finds and tries to start is worse than no runtime at all.
    """
    import tempfile

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.",
                                    dir=str(destination.parent)))
    try:
        target = staging / "build"
        shutil.copytree(source, target)
        previous = destination.with_name(destination.name + ".previous")
        shutil.rmtree(previous, ignore_errors=True)
        if destination.exists():
            destination.rename(previous)
        target.rename(destination)
        shutil.rmtree(previous, ignore_errors=True)
    except PermissionError as exc:
        raise in_use_error(destination, exc) from None
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    for name in SERVER_NAMES:
        for found in destination.rglob(name):
            _make_executable(found)
    return destination


def _make_executable(path: Path) -> None:
    """Restore the executable bit, which a copy across filesystems can drop."""
    if sys.platform == "win32":
        return
    try:
        mode = path.stat().st_mode
        path.chmod(mode | 0o111)
    except OSError:
        logger.debug("Model Chain: could not mark %s executable", path, exc_info=True)


# --------------------------------------------------------------------------- #
# Downloading the pinned build
# --------------------------------------------------------------------------- #


def download(device=None, on_status=None, on_progress=None) -> Path:
    """Fetch and extract the pinned llama.cpp runtime. Weights are not touched.

    Deliberately not ``installer.provision``, which also downloads the pinned
    16-27 GiB model and its projector. Somebody who already has a GGUF wants the
    runtime and only the runtime, so this reuses the same verified download and
    the same atomic extract for the runtime components alone.
    """
    from prompt_master.provisioning.downloader import download as fetch
    from prompt_master.provisioning.extractor import extract_zips_atomic
    from prompt_master.provisioning.installer import load_components, runtime_component_ids

    if not downloadable():
        raise SetupError(_no_download_reason())

    say = on_status or (lambda _text: None)
    tick = on_progress or (lambda _fraction: None)
    gpu = device or preferred_device()
    paths = mc_llm_paths.app_paths()
    paths.create_managed_dirs()

    components = load_components()
    ids = runtime_component_ids(gpu)
    missing = [key for key in ids if key not in components]
    if missing:
        raise SetupError(f"The release manifest has no {', '.join(missing)}")

    archives = []
    share = 1.0 / (len(ids) + 1)
    for number, key in enumerate(ids):
        component = components[key]
        say(f"Downloading {key}…")
        report = (lambda done, total, n=number: tick(share * (n + done / max(total, 1))))
        archives.append(fetch(component, paths.contained(component.destination), report))

    say("Extracting the runtime…")
    tick(share * len(ids))
    destination = paths.root / RUNTIME_DIRNAME
    try:
        extract_zips_atomic(archives, destination)
    except PermissionError as exc:
        raise in_use_error(destination, exc) from None
    executable = _server_in(destination)
    if executable is None:
        raise SetupError("The downloaded archives contain no llama-server executable")
    tick(1.0)
    return executable


# --------------------------------------------------------------------------- #
# Recording it
# --------------------------------------------------------------------------- #


_devices_cache: tuple[float, list] | None = None
DEVICE_CACHE_SECONDS = 60.0
"""How long a device list is reused before nvidia-smi is asked again.

Detection is a subprocess call with a fifteen-second timeout, and the panel
wants the list several times while it is being built -- for the dropdown, for
its current value, and again on every button press. Cached because the set of
cards in a machine does not change while a WebUI is running, and short-lived
because on the rare occasion it does, a minute is not long to wait.
"""


def devices(refresh: bool = False) -> list:
    """Devices the LLM can run on, for the panel's dropdown. Never raises."""
    global _devices_cache

    import time

    from prompt_master.core.models import GpuInfo
    from prompt_master.inference.device_detection import detect_cpu, detect_devices

    if not refresh and _devices_cache is not None:
        cached_at, cached = _devices_cache
        if time.monotonic() - cached_at < DEVICE_CACHE_SECONDS:
            return list(cached)

    try:
        found = list(detect_devices())
    except Exception:
        logger.debug("Model Chain: device detection failed", exc_info=True)
        found = []
    if not found:
        try:
            found = [detect_cpu()]
        except Exception:
            # Detection failed and so did the fallback. An entry describing the
            # processor is still a true statement about the machine, and it is
            # what lets the panel offer CPU execution rather than nothing.
            found = [GpuInfo(physical_index=-1, uuid="", name="CPU (system RAM)",
                             memory_total_mb=0, memory_free_mb=0, driver_version="")]

    _devices_cache = (time.monotonic(), list(found))
    return list(found)


def forget_devices() -> None:
    """Drop the cached device list. For tests, and for a rescan."""
    global _devices_cache

    _devices_cache = None


def preferred_device():
    """The device to use when the user has not chosen one.

    The first card detected, or the processor when there is none. A default
    rather than a decision: the panel offers the list and this only fills it in.
    """
    found = devices()
    for device in found:
        if not device.is_cpu and not device.is_mixed:
            return device
    return found[0]


def describe_device(device) -> str:
    from prompt_master.inference.device_detection import describe

    try:
        return describe(device)
    except Exception:
        return getattr(device, "name", "unknown device")


def record(executable: str | Path, device=None) -> dict:
    """Write ``executable`` into the state file as this install's runtime.

    Merges rather than replaces. A state file that already names a model and a
    projector keeps them: fixing a missing runtime must not cost somebody the
    model they had already chosen.

    The llama.cpp device token is probed by running the executable, exactly as
    ``installer.write_state`` does, because the mapping from a physical GPU
    index to llama.cpp's own ``CUDA0`` naming is not something to guess at. A
    probe that fails falls back to the conventional token and says so in the
    log rather than refusing to record a runtime that is otherwise fine.
    """
    from prompt_master.core.config import atomic_write_json, read_json
    from prompt_master.inference.device_detection import (CPU_DEVICE, NO_OFFLOAD,
                                                          list_llama_devices,
                                                          runtime_component_id)
    from prompt_master.provisioning.installer import DEFAULT_CONTEXT_SIZE, FULL_OFFLOAD

    paths = mc_llm_paths.app_paths()
    path = (mc_llm_files.to_path(executable) or Path("")).resolve()
    if not path.is_file():
        raise SetupError(f"There is no llama-server at {path}")
    try:
        relative = paths.contained(paths.record(path))
    except ValueError:
        raise SetupError(
            f"{path} is outside the LLM data directory ({paths.root}). The runtime is a "
            f"program this extension starts, so it has to live inside that directory — use "
            f"the button above to copy it in."
        ) from None
    if relative != path:
        raise SetupError(f"{path} could not be recorded relative to {paths.root}")

    chosen = device or preferred_device()
    try:
        state = read_json(paths.state_file)
    except (OSError, ValueError):
        state = {}

    if chosen.is_cpu:
        token, token_name, layers = CPU_DEVICE, chosen.name, NO_OFFLOAD
    else:
        try:
            token, token_name = list_llama_devices(path, chosen.physical_index)
        except Exception:
            token, token_name = "CUDA0", chosen.name
            logger.warning("Model Chain: could not ask llama-server which devices it sees; "
                           "assuming CUDA0", exc_info=True)
        layers = NO_OFFLOAD if chosen.is_mixed else FULL_OFFLOAD

    state.update({
        "runtime": paths.record(path),
        "runtime_id": runtime_component_id(chosen),
        "mode": chosen.mode,
        "gpu_index": chosen.physical_index,
        "gpu_uuid": chosen.uuid,
        "gpu_name": chosen.name,
        "gpu_device": token,
        "gpu_device_name": token_name,
        "gpu_layers": str(layers),
    })
    state.setdefault("model", "")
    state.setdefault("mmproj", "")
    state.setdefault("context_size", DEFAULT_CONTEXT_SIZE)

    paths.data.mkdir(parents=True, exist_ok=True)
    atomic_write_json(paths.state_file, state)
    return state
