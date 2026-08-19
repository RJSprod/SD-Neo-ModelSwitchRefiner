from __future__ import annotations

import os, secrets, socket, subprocess, time
from pathlib import Path
import httpx


class LlamaProcess:
    def __init__(self): self.process: subprocess.Popen | None = None; self.port = 0; self.api_key = ""; self._log = None

    def start(self, executable: Path, model: Path, mmproj: Path | None, gpu_index: int, device: str, context_size: int, log_path: Path, gpu_layers: str = "all") -> None:
        # gpu_layers is llama.cpp's --n-gpu-layers. It stays "all" for every card
        # that fits its quantization, which is what the 3090 and 5090 pinned
        # builds always used; a smaller card can record a layer count at setup
        # and run a partial offload instead of being refused.
        #
        # device is "none" for a CPU install — llama.cpp's own token for "offload
        # to nothing", which leaves every layer where the CPU backend reads it.
        # CUDA_VISIBLE_DEVICES is then emptied rather than set to an index, so a
        # card that happens to be in the machine is not picked up behind it.
        #
        # mmproj is None when the model running has no vision projector — a
        # model chosen by hand may simply not have one. --mmproj is left off the
        # command entirely in that case rather than passed something empty,
        # which llama-server would refuse to start on. What it costs is images:
        # InferenceService is what turns that into a sentence, before anything
        # here is reached.
        self.stop(); self.port = self._free_port(); self.api_key = secrets.token_urlsafe(32)
        command = [str(executable),"--model",str(model)]
        if mmproj is not None: command += ["--mmproj",str(mmproj)]
        command += ["--alias","prompt-master","--host","127.0.0.1","--port",str(self.port),"--api-key",self.api_key,"--no-webui","--device",device,"--split-mode","none","--main-gpu","0","--n-gpu-layers",str(gpu_layers),"--ctx-size",str(context_size),"--parallel","1","--reasoning","off","--reasoning-budget","0","--timeout","600"]
        env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = "" if device.casefold() == "none" else str(gpu_index)
        log_path.parent.mkdir(parents=True, exist_ok=True); self._log = log_path.open("a", encoding="utf-8")
        self.process = subprocess.Popen(command, env=env, stdout=self._log, stderr=subprocess.STDOUT, creationflags=getattr(subprocess,"CREATE_NEW_PROCESS_GROUP",0)|getattr(subprocess,"CREATE_NO_WINDOW",0))

    @property
    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def wait_ready(self, timeout: float = 180) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self.process or self.process.poll() is not None: raise RuntimeError("llama-server exited before becoming ready")
            try:
                response = httpx.get(f"http://127.0.0.1:{self.port}/health", timeout=2)
                if response.status_code == 200 and response.json().get("status") == "ok": return
            except (httpx.HTTPError, ValueError): pass
            time.sleep(.5)
        raise TimeoutError(f"llama-server did not become ready within {timeout:.0f} seconds")

    def stop(self) -> None:
        process, self.process = self.process, None
        if process and process.poll() is None:
            process.terminate()
            try: process.wait(10)
            except subprocess.TimeoutExpired: process.kill(); process.wait(5)
        if self._log: self._log.close(); self._log = None

    @staticmethod
    def _free_port() -> int:
        with socket.socket() as sock: sock.bind(("127.0.0.1", 0)); return sock.getsockname()[1]
