"""Intel graphics through llama.cpp's SYCL backend, as an LLM runtime target.

An Intel Arc GPU -- the integrated one in a Meteor Lake processor, or a
discrete card -- runs llama.cpp through the SYCL / oneAPI Level Zero backend,
and this module is everything this extension knows about that which the
vendored device model does not. The vendored detector enumerates NVIDIA cards
through ``nvidia-smi`` and nothing else; the vendored launcher selects a card by
writing an NVIDIA index into ``CUDA_VISIBLE_DEVICES``; the vendored installer
pairs every non-CPU runtime with a cudart archive. None of that is true of
Intel, and ``prompt_master/VENDORED_FROM.txt`` forbids teaching the vendored
tree otherwise, so the Intel half lives here and the modules above the vendored
tree consult it.

Two facts about the target that everything below keeps apart
------------------------------------------------------------

**The execution resource is the Intel GPU.** A request on it competes with
nothing on an NVIDIA card and nothing on the processor: the broker's
execution-domain vocabulary gains a SYCL kind so an image generation on the
3090 and a language model on the Arc run at the same time, and so two requests
for the same Arc still take turns.

**The memory resource is system RAM.** An integrated Arc has no VRAM bank of
its own. Windows reports a "shared GPU memory" *limit* -- on the validation
machine, 54.4 GB -- and that number is a ceiling the OS may let graphics use,
not memory set aside for it and not memory that is free. The weights a SYCL
server holds are host RAM, charged to the host-RAM domain the rest of this
extension already brokers, and never declared as VRAM on any card. So a SYCL
language model can never evict an NVIDIA checkpoint to make itself fit, and an
image generation short of system RAM may stop an idle SYCL server exactly as it
may stop an idle processor one.

Three memory numbers, never collapsed into one
----------------------------------------------

The OS shared-GPU limit (A), the host RAM available right now (B), and what is
*safe* for a language model after this extension's host-RAM reserve and what
its other servers already hold (C). The UI may show A and C; placement is
admitted against C alone::

    shared_remaining = max(limit - committed, 0)          # only when A is known
    host_safe        = max(available - reserve, 0)
    safe             = min(shared_remaining, host_safe)   # host_safe when A is unknown

The Windows "Adapter RAM" field -- 2 GB on the validation machine -- is
recorded for the label and used for nothing. A model far larger than it loads
when the numbers above permit, and proving that is part of accepting this
feature.

Backend identity
----------------

``CUDA0`` and ``SYCL0`` are different namespaces and can both exist on one
machine. Every token this module hands the menu is backend-qualified
(``sycl:0``), the state file records ``compute_backend``, and an older state
file without it resolves through :func:`resolve_backend` to exactly what it
meant before this module existed.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from prompt_master.core.models import GpuInfo as _GpuInfo

logger = logging.getLogger("model_chain")
"""Handler is attached once, in mc_memory."""

_GB = 1024**3
_MB = 1024**2

BACKEND_CUDA = "cuda"
BACKEND_SYCL = "sycl"
BACKEND_CPU = "cpu"
BACKENDS = (BACKEND_CUDA, BACKEND_SYCL, BACKEND_CPU)
"""What ``compute_backend`` in the state file may say."""

TOKEN_PREFIX = "sycl"
"""The menu token namespace: ``sycl:0`` never collides with ``gpu:0``."""

DEVICE_PREFIX = "SYCL"
"""llama.cpp's own device naming: ``SYCL0``, ``SYCL1``..."""

MEMORY_KIND = "uma"
"""The memory domain of a SYCL placement: unified, shared with the processor."""

INTEL_VENDOR_ID = 0x8086

LIST_DEVICES_TIMEOUT = 30.0
DISCOVERY_TIMEOUT = 15.0
CACHE_SECONDS = 60.0

BACKEND_LABEL = "SYCL / Level Zero"
MEMORY_LABEL = "shared system memory"

# The sentences section 13 of the design intent asks for, spelled once.
HARDWARE_WITHOUT_RUNTIME = (
    "Intel Arc is available. Install the SYCL llama.cpp runtime: press Download with the "
    "Intel device selected — it is fetched into a runtime directory of its own, so the build "
    "you already have is kept and switching between them costs nothing after that."
)
NO_DEVICE = (
    "The SYCL llama-server could not see Intel Arc. It started and enumerated no SYCL "
    "device — check the Intel graphics driver and the oneAPI Level Zero runtime, then rescan "
    "the devices in LLM Studio → Setup."
)
CANNOT_ASK = (
    "The SYCL llama-server could not be asked which devices it sees, so the Intel device "
    "was not recorded. Check that the runtime's own DLLs are beside llama-server.exe and "
    "that the Intel graphics driver is installed."
)
MISSING_TOKEN = (
    "The recorded SYCL device is no longer enumerated. Rescan the devices in LLM Studio → "
    "Setup and choose the Intel device again."
)


class SyclError(RuntimeError):
    """Something a person can act on, phrased as a sentence for the panel."""


# --------------------------------------------------------------------------- #
# The device, as the menu and the state file see it
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True, slots=True)
class SyclGpu(_GpuInfo):
    """An Intel GPU reached through SYCL, in the vendored device's own shape.

    A ``GpuInfo`` subclass for the reason ``mc_llm_setup.MinimumGpu`` is one:
    every lifecycle rule that asks ``isinstance(device, GpuInfo)`` keeps
    working, ``mode`` answers ``gpu`` -- the whole model on the device, which
    is the one placement this target offers -- and nothing has to be taught a
    second device type. The fields added are the ones the vendored shape has no
    room for: which backend, which SYCL ordinal, and what the OS says the
    shared limit is.

    ``physical_index`` carries the SYCL ordinal. It is never compared with an
    NVIDIA index: the token and the state file both carry the backend beside
    it, and every reader that used to build ``mode:index`` asks
    :func:`token` instead.
    """

    backend: str = BACKEND_SYCL
    sycl_index: int = 0
    shared_limit_mb: int = 0
    """Windows' shared-GPU-memory limit for this adapter, in MiB; 0 unknown."""
    pci_id: str = ""
    """``VEN_8086&DEV_7D55`` where the OS could say; the label's business only."""
    adapter_ram_mb: int = 0
    """The Adapter RAM field. Recorded because it exists; a ceiling for nothing."""
    memory_kind: str = MEMORY_KIND


def is_sycl_device(device) -> bool:
    """Whether ``device`` is an Intel GPU chosen through SYCL."""
    return getattr(device, "backend", "") == BACKEND_SYCL


def token(device) -> str:
    """``sycl:<ordinal>`` -- the backend-qualified menu value."""
    return f"{TOKEN_PREFIX}:{int(getattr(device, 'sycl_index', 0))}"


def parse_token(value) -> int | None:
    """The SYCL ordinal a menu value names, or ``None`` for any other token."""
    text = str(value or "").strip().casefold()
    if not text.startswith(f"{TOKEN_PREFIX}:"):
        return None
    try:
        return int(text.split(":", 1)[1])
    except (TypeError, ValueError):
        return None


def device_token(index: int) -> str:
    """llama.cpp's own name for SYCL ordinal ``index``: ``SYCL0``."""
    return f"{DEVICE_PREFIX}{int(index)}"


def describe(device) -> str:
    """The menu line: the name, the backend, the memory it really uses.

    Never "54.4 GB VRAM". The suffix names the shared limit as a limit, and
    only when the OS reported one.
    """
    name = getattr(device, "name", "Intel graphics")
    limit = int(getattr(device, "shared_limit_mb", 0) or 0)
    suffix = f" — up to {limit * _MB / _GB:.1f} GB shared" if limit > 0 else ""
    return f"{name} — {BACKEND_LABEL.split(' /')[0]} / {MEMORY_LABEL}{suffix}"


def detail_lines(device=None, found: "Budget | None" = None) -> list[str]:
    """Section 11.3's capacity detail: the limit, the safe figure, the backend, the memory."""
    found = found if found is not None else budget(device)
    limit = ("not reported by the OS" if not found.shared_limit
             else f"{found.shared_limit / _GB:.1f} GB")
    safe = ("unknown — system RAM could not be read" if not found.known
            else f"{found.safe / _GB:.1f} GB")
    return [f"Shared GPU limit: {limit}",
            f"Safe for LLM now: {safe}",
            f"Backend: {BACKEND_LABEL}",
            f"Memory: {MEMORY_LABEL} (unified with the processor), not dedicated VRAM"]


# --------------------------------------------------------------------------- #
# Backend identity, old state files included
# --------------------------------------------------------------------------- #


def resolve_backend(recorded: str = "", device: str = "", mode: str = "",
                    runtime_id: str = "") -> str:
    """Which backend a configuration names, whether or not it was written down.

    The order is section 5.3's, and its point is that no working CPU or CUDA
    installation changes behaviour because a field was introduced: an explicit
    ``compute_backend`` wins; otherwise ``none`` or CPU mode is the processor,
    a ``SYCL`` device token or a SYCL runtime family is SYCL, and everything
    else is what every GPU installation has always been -- CUDA.
    """
    from prompt_master.core.models import CPU_MODE, normalise_mode

    named = str(recorded or "").strip().casefold()
    if named in BACKENDS:
        return named
    token_text = str(device or "").strip()
    if token_text.casefold() == "none" or normalise_mode(mode) == CPU_MODE:
        return BACKEND_CPU
    if token_text.upper().startswith(DEVICE_PREFIX):
        return BACKEND_SYCL
    if "sycl" in str(runtime_id or "").casefold():
        return BACKEND_SYCL
    return BACKEND_CUDA


def backend_of_state(state: dict | None) -> str:
    """:func:`resolve_backend` read off a state document (or a role's layered one)."""
    found = state or {}
    return resolve_backend(found.get("compute_backend", ""), found.get("gpu_device", ""),
                           found.get("mode", ""), found.get("runtime_id", ""))


# --------------------------------------------------------------------------- #
# Hardware discovery
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class IntelGraphics:
    """One Intel display adapter the operating system reports."""

    name: str
    pci_id: str = ""
    adapter_ram_mb: int = 0
    driver_version: str = ""


_PCI_ID = re.compile(r"VEN_8086&DEV_([0-9A-F]{4})", re.I)


def detect_intel_graphics(timeout: float = DISCOVERY_TIMEOUT) -> list[IntelGraphics]:
    """Every Intel display adapter the OS reports. Never raises; empty when it cannot say."""
    try:
        if sys.platform == "win32":
            return _windows_intel_graphics(timeout)
        if sys.platform.startswith("linux"):
            return _linux_intel_graphics()
    except Exception:
        logger.debug("Model Chain: Intel graphics discovery failed", exc_info=True)
    return []


def _windows_intel_graphics(timeout: float) -> list[IntelGraphics]:
    """Windows' own adapter list, filtered to Intel by PCI vendor id.

    Through CIM rather than a registry walk, because it is the same source
    Device Manager reads and it names the adapter the way the user knows it.
    The ``AdapterRAM`` field comes along and is recorded for the label; it is
    not the usable ceiling and the module docstring says why.
    """
    command = [
        "powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
        "-Command",
        "Get-CimInstance Win32_VideoController | "
        "Select-Object Name,PNPDeviceID,AdapterRAM,DriverVersion | ConvertTo-Json -Compress",
    ]
    finished = subprocess.run(  # noqa: S603 - a fixed query of the OS
        command, capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=timeout, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if finished.returncode != 0 or not (finished.stdout or "").strip():
        return []
    return parse_adapter_listing(finished.stdout)


def parse_adapter_listing(text: str) -> list[IntelGraphics]:
    """The Intel adapters in a ``Win32_VideoController`` JSON listing."""
    try:
        loaded = json.loads(text)
    except ValueError:
        return []
    entries = loaded if isinstance(loaded, list) else [loaded]
    found = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        pnp = str(entry.get("PNPDeviceID") or "").upper()
        if not pnp.startswith("PCI\\VEN_8086"):
            continue
        match = _PCI_ID.search(pnp)
        try:
            adapter_ram = int(entry.get("AdapterRAM") or 0) // _MB
        except (TypeError, ValueError):
            adapter_ram = 0
        found.append(IntelGraphics(
            name=str(entry.get("Name") or "Intel(R) Graphics").strip(),
            pci_id=f"VEN_8086&DEV_{match.group(1).upper()}" if match else "VEN_8086",
            adapter_ram_mb=max(adapter_ram, 0),
            driver_version=str(entry.get("DriverVersion") or "").strip()))
    return found


def _linux_intel_graphics() -> list[IntelGraphics]:
    """Intel display-class PCI devices under sysfs."""
    found = []
    for device in sorted(Path("/sys/bus/pci/devices").glob("*")):
        try:
            vendor = (device / "vendor").read_text(encoding="utf-8").strip()
            klass = (device / "class").read_text(encoding="utf-8").strip()
            product = (device / "device").read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if vendor.casefold() != "0x8086" or not klass.casefold().startswith("0x03"):
            continue
        code = product.replace("0x", "").upper()
        found.append(IntelGraphics(name=f"Intel graphics (PCI 8086:{code})",
                                   pci_id=f"VEN_8086&DEV_{code}"))
    return found


def shared_limit_bytes(pci_id: str = "") -> int:
    """Windows' shared-system-memory limit for the Intel adapter, or 0 when unknown.

    Read from the DirectX adapter records the OS keeps under
    ``HKLM\\SOFTWARE\\Microsoft\\DirectX``, where ``SharedSystemMemory`` is
    the same figure Task Manager shows as the shared GPU memory limit. Not
    from WMI's ``AdapterRAM``, which is the dedicated field and is exactly
    the number the design intent says must not be mistaken for a ceiling.
    Zero -- unknown -- whenever the records cannot be read, so the budget falls
    back to host RAM alone rather than to an invented limit.
    """
    if sys.platform != "win32":
        return 0
    wanted_device = None
    match = _PCI_ID.search(str(pci_id or ""))
    if match:
        wanted_device = int(match.group(1), 16)
    try:
        import winreg
    except ImportError:
        return 0
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\DirectX") as root:
            index = 0
            best = 0
            while True:
                try:
                    name = winreg.EnumKey(root, index)
                except OSError:
                    break
                index += 1
                try:
                    with winreg.OpenKey(root, name) as adapter:
                        vendor = _registry_int(adapter, "VendorId")
                        if vendor != INTEL_VENDOR_ID:
                            continue
                        device = _registry_int(adapter, "DeviceId")
                        shared = _registry_int(adapter, "SharedSystemMemory")
                        if shared <= 0:
                            continue
                        if wanted_device is not None and device == wanted_device:
                            return shared
                        best = max(best, shared)
                except OSError:
                    continue
            return best
    except OSError:
        return 0


def _registry_int(key, name: str) -> int:
    try:
        import winreg

        value, _kind = winreg.QueryValueEx(key, name)
        return int(value)
    except (OSError, TypeError, ValueError):
        return -1


# --------------------------------------------------------------------------- #
# What the runtime itself enumerates
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ListedDevice:
    """One SYCL device as ``llama-server --list-devices`` printed it."""

    token: str
    index: int
    name: str
    total_mb: int = 0
    free_mb: int = 0


_LISTED = re.compile(r"\b(SYCL(\d+))\b\s*[:\-]?\s*([^\r\n]*)")
_MEMORY = re.compile(r"\(\s*(\d+)\s*MiB,\s*(\d+)\s*MiB free\s*\)")


def parse_device_listing(text: str) -> list[ListedDevice]:
    """The SYCL devices in a ``--list-devices`` output, whatever else is in it."""
    found: dict[int, ListedDevice] = {}
    for token_text, index, rest in _LISTED.findall(text or ""):
        memory = _MEMORY.search(rest)
        name = _MEMORY.sub("", rest).strip(" -:[]\t")
        ordinal = int(index)
        if ordinal in found:
            continue
        found[ordinal] = ListedDevice(
            token=token_text.upper(), index=ordinal, name=name or token_text.upper(),
            total_mb=int(memory.group(1)) if memory else 0,
            free_mb=int(memory.group(2)) if memory else 0)
    return [found[key] for key in sorted(found)]


def enumerate_devices(executable, timeout: float = LIST_DEVICES_TIMEOUT
                      ) -> list[ListedDevice] | None:
    """Ask a llama-server build which SYCL devices it sees.

    ``None`` when it could not be asked at all -- no executable, a crash, a
    timeout -- which is a different answer from an empty list, and the two
    are acted on differently: an empty list is evidence about the machine,
    ``None`` is not evidence of anything.

    The environment is inherited unchanged. ``ONEAPI_DEVICE_SELECTOR`` in the
    shell that started the WebUI is part of what the server will see when it
    starts, so it is part of the answer.
    """
    if executable is None:
        return None
    try:
        finished = subprocess.run(  # noqa: S603 - the runtime this extension starts
            [str(executable), "--list-devices"], capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except Exception:
        logger.debug("Model Chain: could not ask %s which SYCL devices it sees", executable,
                     exc_info=True)
        return None
    if finished.returncode != 0:
        return None
    return parse_device_listing("\n".join((finished.stdout or "", finished.stderr or "")))


def installed_runtime():
    """The SYCL llama-server this installation holds, or ``None``."""
    try:
        import mc_llm_runtime_components
        import mc_llm_setup

        families = mc_llm_setup.runtime_families()
        return families.get(mc_llm_runtime_components.SYCL_RUNTIME)
    except Exception:
        logger.debug("Model Chain: could not look for an installed SYCL runtime",
                     exc_info=True)
        return None


def validate_runtime(executable, device) -> tuple[str, str]:
    """The device token and name to record, or a refusal (section 7.4).

    The build is run and asked. A SYCL token is never written down on the
    strength of the hardware being present: a CUDA build recorded beside an
    Intel device refuses ``--device SYCL0`` at every start, and the place to
    find that out is here, with the sentence that says what to do.
    """
    listed = enumerate_devices(executable)
    if listed is None:
        raise SyclError(CANNOT_ASK)
    if not listed:
        raise SyclError(NO_DEVICE)
    wanted = int(getattr(device, "sycl_index", 0))
    found = next((entry for entry in listed if entry.index == wanted), None)
    if found is None:
        raise SyclError(MISSING_TOKEN)
    chosen = str(getattr(device, "name", "") or "")
    if not _consistent(found.name, chosen):
        raise SyclError(
            f"{Path(str(executable)).name} enumerates {found.token} as {found.name}, which "
            f"is not the {chosen} that was chosen. Rescan the devices and choose again."
        )
    return found.token, found.name or chosen


def _consistent(enumerated: str, chosen: str) -> bool:
    """Whether the runtime's name for a device can be the one the OS gave it.

    Lenient on purpose: the OS says "Intel(R) Arc(TM) Graphics" and llama.cpp
    may say "Intel(R) Arc(TM) Graphics" or "Intel Arc Graphics" or add a
    driver string. What must not pass is a device that is plainly something
    else -- an NVIDIA card enumerated by a build with more than one backend.
    """
    left, right = enumerated.casefold(), chosen.casefold()
    if not left or not right:
        return True
    if "intel" in right:
        return "intel" in left
    return left.split()[0] == right.split()[0]


# --------------------------------------------------------------------------- #
# The device list the Setup menu merges in
# --------------------------------------------------------------------------- #

_cache: tuple[float, list] | None = None


def devices(refresh: bool = False) -> list[SyclGpu]:
    """Every Intel GPU this machine offers through SYCL. Never raises.

    The runtime's own enumeration is the launch identity when a SYCL build is
    installed; the OS's adapter list is what offers the target before one is
    (section 6.2, step 6), so the Download button has something to be pressed
    for. Empty on a machine with no Intel graphics at all, which leaves every
    existing menu exactly as it was.
    """
    global _cache

    if not refresh and _cache is not None:
        cached_at, cached = _cache
        if time.monotonic() - cached_at < CACHE_SECONDS:
            return list(cached)
    found = _discover()
    _cache = (time.monotonic(), list(found))
    return list(found)


def forget() -> None:
    """Drop the cached device list. For tests, and for a rescan."""
    global _cache

    _cache = None


def _discover() -> list[SyclGpu]:
    hardware = detect_intel_graphics()
    listed = enumerate_devices(installed_runtime()) if hardware or _sycl_runtime_present() \
        else None
    entries: list[SyclGpu] = []
    if listed:
        for position, entry in enumerate(listed):
            adapter = hardware[position] if position < len(hardware) else (
                hardware[0] if hardware else None)
            entries.append(_entry(entry.index, entry.name, adapter))
    elif hardware:
        for position, adapter in enumerate(hardware):
            entries.append(_entry(position, adapter.name, adapter))
    return entries


def _sycl_runtime_present() -> bool:
    return installed_runtime() is not None


def _entry(index: int, name: str, adapter: IntelGraphics | None) -> SyclGpu:
    pci = adapter.pci_id if adapter is not None else ""
    limit = shared_limit_bytes(pci)
    return SyclGpu(
        physical_index=int(index), uuid="", name=name or (adapter.name if adapter else "Intel graphics"),
        memory_total_mb=int(limit // _MB), memory_free_mb=0,
        driver_version=adapter.driver_version if adapter is not None else "",
        compute_capability=None, mixed=False, conservative=False,
        backend=BACKEND_SYCL, sycl_index=int(index), shared_limit_mb=int(limit // _MB),
        pci_id=pci, adapter_ram_mb=adapter.adapter_ram_mb if adapter is not None else 0)


def device_for_token(value, offered=None):
    """The SYCL device a menu token names, or ``None``."""
    wanted = parse_token(value)
    if wanted is None:
        return None
    for device in (devices() if offered is None else offered):
        if is_sycl_device(device) and int(device.sycl_index) == wanted:
            return device
    return None


# --------------------------------------------------------------------------- #
# The memory budget (design intent section 4)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Budget:
    """The three numbers, kept apart, and the one placement is admitted against."""

    shared_limit: int = 0
    """The OS shared-GPU limit in bytes; 0 when it could not be read."""
    committed: int = 0
    """What this extension's own SYCL servers already hold, in bytes."""
    host_available: int = 0
    host_reserve: int = 0
    known: bool = True
    """False when available host RAM could not be read at all."""

    @property
    def shared_remaining(self) -> int | None:
        if self.shared_limit <= 0:
            return None
        return max(self.shared_limit - self.committed, 0)

    @property
    def host_safe(self) -> int:
        return max(self.host_available - self.host_reserve, 0)

    @property
    def safe(self) -> int:
        """What a new SYCL placement may take: the smaller of the two ceilings."""
        remaining = self.shared_remaining
        if remaining is None:
            return self.host_safe
        return min(remaining, self.host_safe)

    def describe(self) -> str:
        if not self.known:
            return "system RAM could not be read"
        parts = [f"{self.safe / _GB:.1f} GB of {MEMORY_LABEL} safe to use "
                 f"({self.host_available / _GB:.1f} GB available above a "
                 f"{self.host_reserve / _GB:.1f} GB reserve"]
        if self.shared_limit > 0:
            parts.append(f", {self.shared_limit / _GB:.1f} GB shared GPU limit")
        return "".join(parts) + ")"


def committed_bytes(excluding=None) -> int:
    """Host RAM this extension's running SYCL servers hold, less ``excluding``'s."""
    try:
        import mc_llm_runtime

        total = 0
        for found in mc_llm_runtime.registry.all():
            if found is excluding:
                continue
            try:
                if not found.configuration().uses_sycl_compute or not found.running():
                    continue
                total += max(int(found.host_ram_bytes() or 0), 0)
            except Exception:
                continue
        return total
    except Exception:
        logger.debug("Model Chain: could not count the SYCL servers' memory", exc_info=True)
        return 0


def budget(device=None, *, excluding=None) -> Budget:
    """The budget right now, for ``device`` (or the first Intel GPU found)."""
    import mc_broker

    available = int(mc_broker.free_ram_bytes() or 0)
    reserve = int(mc_broker.ram_reserve_bytes() or 0)
    limit = int(getattr(device, "shared_limit_mb", 0) or 0) * _MB
    if limit <= 0:
        limit = _first_known_limit()
    return Budget(shared_limit=limit, committed=committed_bytes(excluding),
                  host_available=available, host_reserve=reserve, known=available > 0)


def _first_known_limit() -> int:
    for found in devices():
        if int(getattr(found, "shared_limit_mb", 0) or 0) > 0:
            return int(found.shared_limit_mb) * _MB
    return 0


def shortfall_sentence(needed: int, found: Budget) -> str:
    """Section 13's admission failure, with both numbers in it."""
    limit = (f", within the {found.shared_limit / _GB:.1f} GB shared GPU limit"
             if found.shared_limit > 0 else "")
    return (f"The requested model and context need {needed / _GB:.1f} GB of "
            f"{MEMORY_LABEL}; {found.safe / _GB:.1f} GB is currently safe after the "
            f"{found.host_reserve / _GB:.1f} GB host-RAM reserve{limit}. Free some system "
            f"RAM, choose a smaller context or a smaller quantisation, or choose another "
            f"device in LLM Studio → Setup.")


# --------------------------------------------------------------------------- #
# The launch environment
# --------------------------------------------------------------------------- #


def launch_environment(environ: dict | None) -> dict | None:
    """``environ`` with no NVIDIA card visible behind the SYCL selection.

    The vendored launcher writes ``CUDA_VISIBLE_DEVICES=<gpu_index>`` for every
    device that is not ``none``, and for SYCL that number is a SYCL ordinal
    being read as an NVIDIA slot. It selects nothing for a SYCL build -- there
    is no CUDA backend in it to read the variable -- and it must not become the
    identity of anything, so it is emptied exactly as a processor placement
    empties it: no card that happens to be in the machine is picked up behind
    ``--device SYCL0``, which stays the whole of the selection.

    ``ONEAPI_DEVICE_SELECTOR`` is left as the user's shell had it. Section 8.3
    allows process-level pinning only after the exact selector has been
    validated against the installed runtime, and nothing here has done that.
    """
    if not isinstance(environ, dict):
        return environ
    found = dict(environ)
    found["CUDA_VISIBLE_DEVICES"] = ""
    return found
