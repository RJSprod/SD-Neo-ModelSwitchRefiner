from __future__ import annotations

import csv
import dataclasses
import os
import platform
import re
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from prompt_master.core.models import CPU_INDEX, GpuInfo

# The two cards the upstream Windows build shipped with, and the runtime and
# quantization it pinned for each. These stay hard-coded rather than derived, so
# a 3090 and a 5090 keep provisioning byte-for-byte what they provisioned before
# this file learned about any other card.
PINNED: dict[str, tuple[str, str]] = {
    "NVIDIA GeForce RTX 3090": ("llama-runtime-cuda12", "Q4_K_M"),
    "NVIDIA GeForce RTX 5090": ("llama-runtime-cuda13", "Q6_K_P"),
}

# The CPU build of llama.cpp. It is one self-contained archive — no CUDA DLLs to
# pair it with — and it ships a ggml-cpu-*.dll per instruction set, choosing the
# one this processor supports when the server starts.
CPU_RUNTIME = "llama-runtime-cpu"

# llama.cpp's own token for "offload to nothing" — what CPU mode passes to
# --device, so no backend but the CPU one is even looked at.
CPU_DEVICE = "none"

# llama.cpp's --n-gpu-layers when the weights are to stay in system RAM. It is
# what CPU mode and mixed mode both record. In mixed mode the card is still
# named on --device, so llama.cpp keeps using it for the work it can take off
# the CPU — large-batch matrix multiplies during prompt processing, and the
# vision projector — copying what it needs across as it goes. That is the whole
# of the difference: same CUDA runtime, same install, no layers resident.
NO_OFFLOAD = "0"

# What CPU and mixed mode install unless the user chooses otherwise.
# Deliberately not "the largest that fits", which is how a card in GPU mode is
# sized: with the weights in system RAM every byte of them crosses the memory
# bus, so the smallest pinned build is the one that moves least — and it is
# also the shortest download. All three quantizations stay selectable.
SYSTEM_RAM_DEFAULT_QUANT = "Q4_K_M"

# Shown when the processor will not name itself.
GENERIC_CPU_NAME = "CPU (system RAM)"

# Blackwell is sm_100/sm_120 and the CUDA 12.4 build cannot target it, so
# compute capability — not marketing name — is what selects the runtime.
CUDA13_MIN_COMPUTE = 10.0

# Minimum total VRAM for each quantization, largest first: the weights, plus the
# f16 vision projector, plus room for a 16K KV cache and the compute buffers,
# rounded to a whole number of GiB. They sit deliberately above the raw file
# sizes — a 24 GiB card physically fits the 21.2 GiB Q6_K_P weights and then
# fails the moment the projector and context are added, which is exactly why the
# 3090 is a Q4_K_M card.
QUANT_MIN_VRAM_MB: tuple[tuple[str, int], ...] = (
    ("Q8_K_P", 40 * 1024),
    ("Q6_K_P", 30 * 1024),
    ("Q4_K_M", 22 * 1024),
)

QUANTIZATIONS: tuple[str, ...] = tuple(quant for quant, _ in reversed(QUANT_MIN_VRAM_MB))

SMALLEST_QUANT = QUANT_MIN_VRAM_MB[-1][0]

_QUERY = "index,uuid,name,memory.total,memory.free,driver_version"


def detect_gpus(timeout: float = 15) -> list[GpuInfo]:
    """Every NVIDIA GPU nvidia-smi reports, with compute capability when known.

    ``compute_cap`` is a relatively recent addition to nvidia-smi, so the query
    is attempted with it and retried without. A driver too old to report the
    field still yields usable rows; only the runtime choice falls back to the
    model-number heuristic.
    """
    rows = _query(f"{_QUERY},compute_cap", timeout)
    if rows is None:
        rows = _query(_QUERY, timeout)
    if rows is None:
        raise RuntimeError(
            "nvidia-smi is not available. Install the NVIDIA driver, or check that "
            "nvidia-smi is on PATH."
        )
    output = []
    for row in rows:
        if len(row) < 6:
            continue
        compute = None
        if len(row) >= 7:
            try:
                compute = float(row[6].strip())
            except ValueError:
                compute = None
        output.append(GpuInfo(int(row[0]), row[1].strip(), row[2].strip(), int(row[3]),
                              int(row[4]), row[5].strip(), compute))
    return output


def detect_cpu() -> GpuInfo:
    """The processor and system RAM, described the way a card is.

    Nothing is detected in the nvidia-smi sense — the processor is always there.
    What is looked up is its name and how much RAM the machine has, so the
    device list can say "Intel(R) Core(TM) i7-13700K — 65413 MiB of system RAM"
    rather than "CPU".
    """
    total, free = system_memory_mb()
    return GpuInfo(CPU_INDEX, "CPU", cpu_name(), total, free, platform_tag(), None)


def mixed_device(gpu: GpuInfo) -> GpuInfo:
    """The same card, chosen for Mixed Aggressive instead of a full offload."""
    if gpu.is_cpu:
        raise ValueError("Mixed mode needs a CUDA GPU to hand work to")
    return dataclasses.replace(gpu, mixed=True, conservative=False)


def conservative_device(gpu: GpuInfo) -> GpuInfo:
    """The same card, chosen for Mixed Conservative.

    The third entry a card gets. It is the same hardware as the other two and
    differs only in what may be put on it, which is why it is a variant of the
    card rather than a device of its own.
    """
    if gpu.is_cpu:
        raise ValueError("Mixed mode needs a CUDA GPU to hand work to")
    return dataclasses.replace(gpu, mixed=True, conservative=True)


def detect_devices(timeout: float = 15) -> list[GpuInfo]:
    """Every choice an install can be pinned to, in the order they are offered.

    Each CUDA GPU appears three times — once holding the model, once for each
    mixed mode — and the CPU is always last and always present, so a machine
    with no NVIDIA driver is one with a single option rather than one setup
    refuses. A failed scan is therefore not an error here; it only shortens the
    list. Two cards and a processor is seven options.

    A card's own entry stays first throughout, so the first option is what it
    has always been.
    """
    try:
        gpus = detect_gpus(timeout)
    except RuntimeError:
        gpus = []
    offered: list[GpuInfo] = []
    for gpu in gpus:
        offered += [gpu, mixed_device(gpu), conservative_device(gpu)]
    return [*offered, detect_cpu()]


def recorded_mode(mode: str = "", device: str = "", gpu_layers: str = "") -> str:
    """Which of the three ways a recorded install is using its device.

    The state file's own ``mode`` when it has one, and otherwise read off the
    two settings that say the same thing in older files: no CUDA device at all
    is CPU mode, and a card with no resident layers is mixed mode. Written once
    because the three callers that need it — the menu, the runtime and the
    setup writer — disagreeing about which mode an install is in is the same
    bug as not recording the mode at all.
    """
    from prompt_master.core.models import (
        CPU_MODE, GPU_MODE, MIXED_AGGRESSIVE_MODE, normalise_mode)

    named = normalise_mode(mode)
    if named:
        return named
    if str(device or "").strip().casefold() == CPU_DEVICE:
        return CPU_MODE
    # An old file with no mode at all. Zero layers meant Mixed when it was
    # written, and Mixed then did what Aggressive does now -- see
    # ``models.LEGACY_MIXED_MODE``.
    return MIXED_AGGRESSIVE_MODE if str(gpu_layers) == NO_OFFLOAD else GPU_MODE


def device_token(device: GpuInfo) -> str:
    """A device's identity as one string, for a dropdown value or a saved choice.

    The physical index alone is not an identity: a CUDA card is offered twice,
    once holding the weights and once for each mixed mode, and all three entries
    carry the same index. Keyed on the index alone the options collapse into
    one, and the one that survives is whichever was listed first — which is how choosing
    "mixed" in a menu records a full offload and fills the card it was meant to
    keep free. The mode is the half that tells them apart, so it is in here.
    """
    return f"{device.mode}:{device.physical_index}"


def device_for_token(token: str, offered: Sequence[GpuInfo]) -> GpuInfo | None:
    """The device in ``offered`` that ``token`` names, or None.

    A bare index is still understood, and still means exactly what it meant
    when it was all the menu wrote: the first device with that index, which is
    the card itself for a card and the processor for -1. A menu value saved by
    an earlier version therefore selects the same device it selected then.

    So does ``mixed:0``, which is what every installation configured before the
    two mixed modes were split has saved. It resolves to Mixed Aggressive, for
    the reason ``models.LEGACY_MIXED_MODE`` gives: that is the behaviour the
    single Mixed mode had. Without this the saved token matches no offered
    device, the menu falls back to the first entry, and an install that had been
    told to keep the card free silently starts filling it.
    """
    from prompt_master.core.models import normalise_mode

    text = str(token or "").strip().casefold()
    if not text:
        return None
    mode, separator, index = text.partition(":")
    try:
        wanted = int(index if separator else text)
    except ValueError:
        return None
    mode = normalise_mode(mode) if separator else ""
    for device in offered:
        if device.physical_index == wanted and (not separator or device.mode == mode):
            return device
    return None


def describe(device: GpuInfo) -> str:
    """One line naming a device and what it will do with the model.

    Written once and read in three places — the console questions, the setup
    wizard's list and the runtime menu — because a device that is described
    differently in two of them reads as two different devices.
    """
    if device.is_cpu:
        return f"{device.name} — {device.memory_total_mb} MiB of system RAM — no GPU used"
    if device.is_conservative:
        return (f"{device.name} — mixed conservative: no model layers in VRAM, "
                "card used for processing")
    if device.is_mixed:
        return f"{device.name} — mixed: model in system RAM, card used for processing"
    return f"{device.name} — {device.memory_total_mb} MiB — {device.uuid}"


def cpu_name() -> str:
    """The processor's marketing name, e.g. "Intel(R) Core(TM) i7-13700K".

    Every source below is best-effort and platform-specific; naming the chip is
    a nicety, so anything that does not answer falls through to the generic
    label rather than failing setup.
    """
    name = ""
    if sys.platform == "win32":
        # PROCESSOR_IDENTIFIER is always set but coarse ("Intel64 Family 6 ...").
        # The registry holds the full name the vendor programmed into the chip.
        name = os.environ.get("PROCESSOR_IDENTIFIER", "")
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                name = winreg.QueryValueEx(key, "ProcessorNameString")[0] or name
        except (ImportError, OSError):
            pass
    elif sys.platform == "darwin":
        try:
            result = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                    capture_output=True, text=True, timeout=5, check=True)
            name = result.stdout
        except (OSError, subprocess.SubprocessError):
            pass
    else:
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace").splitlines():
                if line.split(":", 1)[0].strip() in ("model name", "Model"):
                    name = line.split(":", 1)[1]
                    break
        except OSError:
            pass
    return " ".join(name.split()) or GENERIC_CPU_NAME


def platform_tag() -> str:
    """What the processor records where a card records its driver version."""
    return platform.machine() or "cpu"


def system_memory_mb() -> tuple[int, int]:
    """``(total, available)`` system RAM in MiB, or ``(0, 0)`` if unreadable.

    Unreadable is survivable: the CPU path reports memory, it is not sized by
    it, so an unanswered query costs a number in a label and nothing more.
    """
    try:
        import psutil

        memory = psutil.virtual_memory()
        return memory.total // 2 ** 20, memory.available // 2 ** 20
    except Exception:
        return 0, 0


def _query(fields: str, timeout: float) -> list[list[str]] | None:
    command = ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                                errors="replace", timeout=timeout, check=True,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.SubprocessError):
        return None
    return list(csv.reader(result.stdout.splitlines(), skipinitialspace=True))


def recommended_quantization(gpu: GpuInfo) -> str:
    """The largest quantization this card has the memory to run.

    A card with less memory than even the smallest quantization needs still gets
    that quantization back: the choice remains legitimate with a partial offload,
    so whether to go ahead is the caller's decision, and ``vram_shortfall_mb``
    is what it warns with.

    A device that keeps the weights in system RAM is not sized this way — see
    ``SYSTEM_RAM_DEFAULT_QUANT``.
    """
    if gpu.weights_in_system_ram:
        return SYSTEM_RAM_DEFAULT_QUANT
    pinned = PINNED.get(gpu.name)
    if pinned is not None:
        return pinned[1]
    for quant, minimum in QUANT_MIN_VRAM_MB:
        if gpu.memory_total_mb >= minimum:
            return quant
    return SMALLEST_QUANT


def vram_shortfall_mb(gpu: GpuInfo, quantization: str) -> int:
    """How far short of a full GPU offload this pairing is; 0 when it fits.

    Always 0 in mixed and CPU mode. A shortfall measures what would have to
    spill out of VRAM and into system RAM; those two put the weights in system
    RAM to begin with, so there is nothing to fall short of and nothing to warn
    about — a card too small to hold its quantization is what mixed mode is for.
    """
    minimum = dict(QUANT_MIN_VRAM_MB).get(quantization)
    if minimum is None:
        raise ValueError(f"Unknown quantization: {quantization}")
    if gpu.weights_in_system_ram:
        return 0
    return max(0, minimum - gpu.memory_total_mb)


def runtime_component_id(gpu: GpuInfo) -> str:
    """Return the independently pinned runtime this device requires."""
    if gpu.is_cpu:
        return CPU_RUNTIME
    pinned = PINNED.get(gpu.name)
    if pinned is not None:
        return pinned[0]
    if gpu.compute_capability is not None:
        return ("llama-runtime-cuda13" if gpu.compute_capability >= CUDA13_MIN_COMPUTE
                else "llama-runtime-cuda12")
    # No compute capability from the driver. RTX 50-series is the only Blackwell
    # consumer line, so its model number is the last usable signal.
    return ("llama-runtime-cuda13" if re.search(r"\bRTX\s*50\d\d\b", gpu.name, re.I)
            else "llama-runtime-cuda12")


def list_llama_devices(executable: Path, physical_index: int, timeout: float = 30) -> tuple[str, str]:
    """Ask llama.cpp for the device identifier/name after restricting visibility.

    llama.cpp has used both ``CUDA0`` and ``CUDA0: <name>``-style output over
    time, so parsing deliberately accepts the identifier wherever it occurs.
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(physical_index)
    result = subprocess.run(
        [str(executable), "--list-devices"], env=env, capture_output=True,
        text=True, encoding="utf-8", errors="replace", timeout=timeout,
        check=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    output = "\n".join((result.stdout, result.stderr))
    match = re.search(r"\b(CUDA\d+)\b\s*[:\-]?\s*([^\r\n]*)", output, re.I)
    if not match:
        raise RuntimeError(f"llama-server --list-devices returned no CUDA device:\n{output.strip()}")
    device = match.group(1).upper()
    name = match.group(2).strip(" -:[]") or device
    return device, name
