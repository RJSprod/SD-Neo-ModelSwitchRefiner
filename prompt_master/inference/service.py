from __future__ import annotations

from pathlib import Path

from prompt_master.core.config import read_json
from prompt_master.core.paths import AppPaths
from .device_detection import CPU_DEVICE, NO_OFFLOAD
from .llama_client import LlamaClient
from .llama_process import LlamaProcess

# How long llama-server is given to load the model and answer /health. Reading
# 17-27 GiB into system RAM takes longer than filling VRAM does, so an install
# that keeps the weights there — CPU mode, and mixed mode with it — is given the
# room to do it rather than being declared dead at three minutes. Neither number
# is a limit on generation, only on start-up.
GPU_READY_TIMEOUT = 180
CPU_READY_TIMEOUT = 1200


class InferenceService:
    """Owns the single managed llama-server process for the application."""

    def __init__(self, paths: AppPaths):
        self.paths = paths
        self.process = LlamaProcess()
        self.signature: tuple | None = None

    def vision_ready(self) -> bool:
        """Whether the model this install runs can be shown a picture.

        Asked before an image is attached rather than after it has been sent,
        so "this model has no projector" is something the window can say while
        there is still a choice to make about it.
        """
        try:
            return self.projector() is not None
        except (OSError, ValueError):
            return False

    def projector(self) -> Path | None:
        """The configured vision projector, or ``None`` when there is none.

        Optional because a model chosen by hand may not have one: the pinned
        model always ships beside its projector, and an arbitrary GGUF does
        not. Recorded-but-absent stays an error — that is a broken install
        rather than a choice — and it is ``client`` that raises it.
        """
        recorded = str(read_json(self.paths.state_file).get("mmproj") or "")
        if not recorded:
            return None
        found = self.paths.locate(recorded)
        return found if found.is_file() else None

    def client(self, needs_vision: bool = False) -> LlamaClient:
        state = read_json(self.paths.state_file)
        required = ("runtime", "model", "gpu_index")
        missing = [key for key in required if key not in state]
        if missing:
            raise RuntimeError("Setup is incomplete (missing " + ", ".join(missing) + "). Run `python app.py --setup`, or open Models and Hardware setup.")
        # The runtime is contained and the weights are only located: one of
        # these is a program this application starts, and the other is a file it
        # reads. See AppPaths.locate.
        runtime, model = self.paths.contained(state["runtime"]), self.paths.locate(state["model"])
        recorded = str(state.get("mmproj") or "")
        mmproj = self.paths.locate(recorded) if recorded else None
        for label, path in (("llama-server", runtime), ("model", model), ("vision projector", mmproj)):
            if path is not None and not path.is_file():
                raise RuntimeError(f"Configured {label} is missing: {path}")
        if needs_vision and mmproj is None:
            raise RuntimeError("This request carries an image, and the model running has no vision projector. Choose one under Settings → Which model runs, or send the request without the image; text-only fallback is disabled.")
        signature = (runtime, model, mmproj, int(state["gpu_index"]), state.get("gpu_device", "CUDA0"), int(state.get("context_size", 8192)), str(state.get("gpu_layers", "all")))
        # "No layers offloaded" is what both system-RAM modes record, and it is
        # the physical fact the load time follows from, so it is what is read
        # here rather than the mode name beside it.
        from_system_ram = signature[4].casefold() == CPU_DEVICE or signature[6] == NO_OFFLOAD
        if not self.process.running or signature != self.signature:
            self.process.start(runtime, model, mmproj, signature[3], signature[4], signature[5], self.paths.logs / "llama-server.log", gpu_layers=signature[6])
            self.process.wait_ready(CPU_READY_TIMEOUT if from_system_ram else GPU_READY_TIMEOUT)
            self.signature = signature
        return LlamaClient(f"http://127.0.0.1:{self.process.port}", self.process.api_key)

    def stop(self) -> None:
        self.process.stop()
        self.signature = None
