"""Invariant I-2 and I-3, asserted against the import graph rather than hoped for.

Voice Chat's whole architectural claim is that it runs *beside* Forge instead of
negotiating with it: no residency published to the broker, no VRAM made room
for, no waiting for an image job to finish. That claim is only worth anything if
it cannot quietly stop being true, and the way it stops being true is somebody
importing ``mc_memory`` in a voice module to answer a question that looked
related.

So the graph is read. Every ``import`` in every voice module -- including the
ones inside functions, which is where a coupling would actually be added --
is collected and checked against a list of modules voice is not allowed to know
about. The worker gets a stricter rule again: nothing but the standard library
at module level, because the parent has to be able to import it under Forge's
interpreter to read a struct format, and the day that starts requiring
``sherpa_onnx`` is the day the WebUI stops starting.
"""

from __future__ import annotations

import ast
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

VOICE_MODULES = sorted(ROOT.glob("mc_voice_*.py"))

FORBIDDEN = ("mc_memory", "mc_broker", "mc_plan", "mc_llm_runtime")
"""The memory planner, the GPU broker, the execution plan, and the language
model runtime. Voice may not import any of them at any depth."""

WORKER = ROOT / "voice_worker" / "worker.py"


def imported_names(path: Path) -> set[str]:
    """Every module named by an import anywhere in ``path``, top level or not."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                found.add(node.module.split(".")[0])
    return found


def dotted_imports(path: Path) -> set[str]:
    """Every import as it was written, so ``urllib.request`` is distinguishable
    from ``urllib.parse``."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
    return found


def module_level_imports(path: Path) -> set[str]:
    found: set[str] = set()
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module.split(".")[0])
    return found


class TestTheImportGraph:
    def test_there_are_voice_modules_to_check(self):
        """A guard on the guard: a glob that stopped matching would make every
        assertion below vacuously true."""
        assert len(VOICE_MODULES) >= 5

    @pytest.mark.parametrize("path", VOICE_MODULES, ids=lambda p: p.name)
    def test_no_voice_module_imports_the_memory_side(self, path):
        offending = imported_names(path) & set(FORBIDDEN)
        assert not offending, (
            f"{path.name} imports {sorted(offending)}. Voice Chat runs beside Forge and does "
            f"not participate in its memory decisions — see invariant I-3.")

    def test_the_worker_imports_only_the_standard_library_at_module_level(self):
        """The parent imports this file to read the frame format. It must be
        importable under Forge's interpreter, where the speech engine is not
        installed and never will be."""
        found = module_level_imports(WORKER)
        assert "sherpa_onnx" not in found
        for name in found:
            assert name in sys.stdlib_module_names, (
                f"voice_worker/worker.py imports {name} at module level; it has to be "
                f"importable by the parent process, which has no voice runtime.")

    def test_the_worker_does_not_import_the_extension(self):
        for name in imported_names(WORKER):
            assert not name.startswith("mc_"), (
                f"the voice worker imports {name}; it runs under a different interpreter "
                f"which cannot see this extension's modules.")

    def test_only_the_model_manager_reaches_the_network(self):
        """One door out, and it is the one with the hashes behind it.

        A transport in the runtime manager, in the API routes or in the UI would
        be a second way for bytes to arrive and nothing would be checking them
        against the manifest. Named exactly -- ``urllib.parse`` is string
        handling and is used by the Origin check, and banning it would be
        banning the wrong thing.
        """
        transports = {"urllib.request", "urllib.error", "http.client", "requests",
                      "httpx", "socket", "ftplib", "aiohttp"}
        for path in VOICE_MODULES:
            names = dotted_imports(path)
            reaching = names & transports
            if path.name == "mc_voice_models.py":
                assert "urllib.request" in names, "the downloader has stopped downloading"
                continue
            assert not reaching, (
                f"{path.name} imports {sorted(reaching)}; only mc_voice_models may make "
                f"network requests (invariant I-4).")


class TestNothingWaitsOnTheBroker:
    def test_a_transcription_does_not_touch_the_broker_or_the_planner(self, monkeypatch,
                                                                     fake_worker):
        """T-INDEP-1. The broker and the planner are made to explode; speech
        still works, because it never asks either of them anything."""
        import mc_broker
        import mc_memory
        import mc_voice_runtime

        def detonate(*args, **kwargs):
            raise AssertionError("Voice Chat asked the memory side for something")

        for module, name in ((mc_broker, "workload"), (mc_broker, "await_idle"),
                             (mc_broker, "clear"), (mc_memory, "make_vram_room"),
                             (mc_memory, "release_all")):
            if hasattr(module, name):
                monkeypatch.setattr(module, name, detonate)

        found = mc_voice_runtime.transcribe(fake_worker.wav)
        assert found["text"] == fake_worker.transcript
        assert mc_voice_runtime.synthesize("hello") == fake_worker.audio

    def test_a_busy_image_job_does_not_hold_speech_up(self, host, fake_worker):
        """T-INDEP-3. ``shared.state`` says a generation is running from start to
        finish, and both speech operations complete inside it."""
        state = host.shared.state
        state.job = "txt2img"
        state.job_count = 20
        state.sampling_step = 3

        import mc_voice_runtime

        assert mc_voice_runtime.transcribe(fake_worker.wav)["text"] == fake_worker.transcript
        assert mc_voice_runtime.synthesize("hello")
        assert state.job == "txt2img", "voice changed the host's generation state"


class TestTheWorkerAsksForTheCpu:
    def test_the_environment_hides_every_gpu(self):
        """T-INDEP-4 and I-1, at the layer that is enforceable from Python."""
        import mc_voice_models

        environ = mc_voice_models.worker_environment()
        assert environ["CUDA_VISIBLE_DEVICES"] == ""
        assert environ["HIP_VISIBLE_DEVICES"] == ""

    @pytest.mark.voice(handshake={"provider": "cuda"})
    def test_a_worker_reporting_a_gpu_provider_is_refused(self, fake_worker):
        """Fail closed. A runtime that came back saying "cuda" is a runtime that
        would quietly take VRAM from the image being generated."""
        import mc_voice_runtime

        with pytest.raises(mc_voice_runtime.VoiceRuntimeError, match="CPU only"):
            mc_voice_runtime.ensure_started()
        assert not mc_voice_runtime.status()["running"], "the refused worker is still running"

    def test_a_manifest_that_asks_for_a_gpu_provider_is_refused(self, tmp_path, monkeypatch):
        import json

        import mc_voice_models
        import mc_voice_paths

        spec = json.loads(mc_voice_paths.manifest_path().read_text(encoding="utf-8"))
        spec["runtime"]["provider"] = "cuda"
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps(spec), encoding="utf-8")
        monkeypatch.setattr(mc_voice_paths, "manifest_path", lambda: path)
        with pytest.raises(mc_voice_models.VoiceError, match="only 'cpu'"):
            mc_voice_models.manifest(refresh=True)


class TestTheExtensionStartsWithoutVoice:
    def test_the_conversation_panel_does_not_need_a_voice_runtime(self, monkeypatch):
        """I-8. Every voice call the panel makes is behind a handler that
        answers rather than raises, so a broken voice installation costs the
        Voice chip and nothing else."""
        import mc_voice_models
        import mc_voice_ui

        def broken():
            raise RuntimeError("no voice here")

        monkeypatch.setattr(mc_voice_models, "status", broken)
        marker = mc_voice_ui.speech_marker(lambda: "a completed reply")
        assert marker() == ""

    def test_the_voice_modules_import_without_a_host(self):
        """They are imported at extension import time, before Forge has finished
        starting, and on installations that will never open LLM Studio."""
        for name in ("mc_voice_paths", "mc_voice_state", "mc_voice_models",
                     "mc_voice_runtime", "mc_voice_api", "mc_voice_ui"):
            assert isinstance(sys.modules.get(name), types.ModuleType) or __import__(name)
