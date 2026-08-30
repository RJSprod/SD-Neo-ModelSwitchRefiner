"""The recording-cleanup engine: what it costs, what it refuses, and when it stops.

DeepFilterNet on an interpreter of its own, because its Rust library publishes
no wheel for the Python Forge runs on. None of this can be *executed* in this
repository -- there is no Windows, no cp311 Torch and no model here -- so what
is asserted is everything up to the point where the model would load: the
manifest is complete and pinned, the platform gate is closed, the install
transaction refuses before it touches anything, the protocol round-trips, and
the runtime stops.
"""

from __future__ import annotations

import json

import pytest

import mc_voice_cleanup as cleanup
import mc_voice_cleanup_runtime as runtime
import mc_voice_paths as paths
from cleanup_worker import worker


class TestTheManifest:
    def test_it_parses_and_names_one_platform(self):
        found = cleanup.manifest()
        assert found["runtime"]["python"] == "3.11"
        assert found["runtime"]["platform"] == "windows-x86_64-cp311"

    def test_every_wheel_is_pinned_and_served_over_https(self):
        """The interpreter is the one exception and is named as one below."""
        for item in cleanup.manifest()["runtime"]["wheels"]:
            assert item["url"].startswith("https://"), item["filename"]
            assert len(item["sha256"]) == 64, item["filename"]
            assert item["bytes"] > 0, item["filename"]

    def test_the_closure_holds_what_deepfilternet_actually_imports(self):
        """Read off ``df/enhance.py`` and ``df/io.py`` rather than off a guess.

        ``deepfilterlib`` is the one that matters: it is the Rust extension, it
        is the reason for the second interpreter, and a closure without it
        installs cleanly and fails at the first import.
        """
        names = {item["filename"].split("-")[0].lower().replace("_", "-")
                 for item in cleanup.manifest()["runtime"]["wheels"]}
        for wanted in ("deepfilternet", "deepfilterlib", "torch", "torchaudio", "numpy",
                       "loguru", "appdirs", "requests"):
            assert wanted in names, f"{wanted} is missing from the closure"

    def test_the_torchaudio_pin_is_one_deepfilternet_can_use(self):
        """``df/io.py`` imports ``torchaudio.backend.common.AudioMetaData``, which
        torchaudio removed after 2.2. Pinning the newest pair that exists would
        install cleanly and fail on import, so the pin is the newest pair that
        still has the API DeepFilterNet was written against."""
        found = {item["filename"].split("-")[0].lower(): item["filename"]
                 for item in cleanup.manifest()["runtime"]["wheels"]}
        version = found["torchaudio"].split("-")[1]
        major, minor = (int(part) for part in version.split(".")[:2])
        assert (major, minor) <= (2, 2), f"torchaudio {version} has no backend.common"

    def test_the_interpreter_is_the_one_unpinned_thing_and_says_so(self):
        """Agreed to explicitly rather than slipped in: python.org is not
        reachable from the workspace this manifest is generated in."""
        interpreter = cleanup.manifest()["runtime"]["interpreter"]
        assert interpreter["url"].startswith("https://www.python.org/")
        assert "sha256" not in interpreter
        assert "3.11" in interpreter["filename"]

    def test_the_price_is_a_number_this_build_can_state(self):
        size = cleanup.download_bytes()
        assert 200e6 < size < 400e6, size

    def test_the_interpreter_is_fetched_like_any_other_artifact(self):
        """Last in the list and carrying no hash, which is what makes the
        installer ask the publisher about it and record what arrived."""
        artifacts = cleanup._artifacts()
        assert len(artifacts) == len(cleanup.manifest()["runtime"]["wheels"]) + 1
        assert artifacts[-1].sha256 is None
        assert not artifacts[-1].pinned
        assert all(item.pinned for item in artifacts[:-1])


class TestWhereItWillNotRun:
    def test_it_is_windows_only_and_says_why_it_is_not_here(self, host):
        """Linux Torch on PyPI pulls the entire CUDA closure, which this engine
        neither claims nor consumes."""
        assert cleanup.supported_platform() is False
        found = cleanup.status()
        assert found.platform_supported is False
        assert found.ready is False
        assert "in-page cleanup" in found.message

    def test_installing_here_refuses_before_it_downloads_anything(self, host, voice_root):
        with pytest.raises(cleanup.CleanupError, match="no tested CPU runtime"):
            cleanup.install()
        assert not paths.cleanup_runtime_root().exists()

    def test_starting_here_refuses_rather_than_leaving_a_process(self, host, voice_root):
        with pytest.raises(runtime.CleanupRuntimeError):
            runtime.ensure_started()
        assert runtime.running() is False


class TestTheProtocol:
    def test_a_frame_round_trips(self):
        import io

        buffer = io.BytesIO()
        worker.write_frame(buffer, {"op": "clean", "rate": 24000}, b"\x01\x02\x03\x04")
        buffer.seek(0)
        header, payload = worker.read_frame(buffer)
        assert header == {"op": "clean", "rate": 24000}
        assert payload == b"\x01\x02\x03\x04"

    def test_end_of_input_is_not_an_error(self):
        """Door D: the parent's death arrives as end-of-input while the worker
        is waiting for work, and the loop ends with 0 rather than a traceback."""
        import io

        assert worker.read_frame(io.BytesIO(b"")) is None

    def test_an_oversized_header_is_refused(self):
        import io
        import struct

        buffer = io.BytesIO(struct.pack(">I", worker.MAX_HEADER + 1))
        with pytest.raises(ValueError, match="header too large"):
            worker.read_frame(buffer)

    def test_it_speaks_the_same_framing_as_the_other_two_workers(self):
        """By agreement rather than by import: the three run on three different
        interpreters, so sharing a module would mean sharing a Python."""
        import io

        from sopro_worker import worker as sopro

        buffer = io.BytesIO()
        worker.write_frame(buffer, {"op": "ping"}, b"abc")
        buffer.seek(0)
        assert sopro.read_frame(buffer) == ({"op": "ping"}, b"abc")


class TestItStopsWhenItIsNotBeingUsed:
    def test_the_worker_names_an_idle_timeout(self):
        """The engine is meant to run only while it is being used. A worker
        holding a Torch runtime all afternoon because somebody cleaned one clip
        is exactly what that promise is about."""
        assert 30.0 <= worker.IDLE_SECONDS <= 600.0

    def test_the_runtime_reports_it_without_starting_anything(self):
        found = runtime.status()
        assert found["running"] is False
        assert found["stops_after"] == worker.IDLE_SECONDS

    def test_shutdown_is_safe_when_nothing_is_running(self):
        """Doors A, B and C all land here and none of them may raise."""
        runtime.shutdown()
        runtime.shutdown()
        assert runtime.running() is False

    def test_the_idle_timer_is_a_courtesy_and_not_the_guarantee(self):
        """``scripts/model_chain.py`` calls ``shutdown`` unconditionally on
        unload, whatever the timer did or did not do."""
        source = (paths.extension_root() / "scripts" / "model_chain.py").read_text(
            encoding="utf-8")
        assert "mc_voice_cleanup_runtime.shutdown()" in source


class TestTheCpuOnlyRule:
    def test_the_worker_environment_hides_every_graphics_device(self):
        found = cleanup.worker_environment()
        assert found["CUDA_VISIBLE_DEVICES"] == ""
        assert found["HIP_VISIBLE_DEVICES"] == ""
        assert found["ROCR_VISIBLE_DEVICES"] == ""

    def test_it_caps_its_threads_the_way_the_other_engines_do(self):
        found = cleanup.worker_environment()
        assert found["OMP_NUM_THREADS"] == str(cleanup.INTRAOP_THREADS)
        assert found["MKL_NUM_THREADS"] == str(cleanup.INTRAOP_THREADS)

    def test_the_handshake_refuses_a_worker_that_is_not_on_the_cpu(self, monkeypatch):
        """Fail closed, the same way Sopro's does: a runtime that came back
        saying "cuda" is one that would take VRAM from an image being made."""
        monkeypatch.setattr(runtime, "_request",
                            lambda header, payload, timeout: (
                                {"protocol_version": worker.PROTOCOL_VERSION,
                                 "backend": "deepfilternet", "device": "cuda:0",
                                 "parent_death": "pdeathsig"}, b""))
        with pytest.raises(runtime.CleanupRuntimeError, match="CPU only"):
            runtime._handshake_with()

    def test_the_handshake_refuses_another_backend(self, monkeypatch):
        monkeypatch.setattr(runtime, "_request",
                            lambda header, payload, timeout: (
                                {"protocol_version": worker.PROTOCOL_VERSION,
                                 "backend": "sopro", "device": "cpu",
                                 "parent_death": "pdeathsig"}, b""))
        with pytest.raises(runtime.CleanupRuntimeError, match="backend"):
            runtime._handshake_with()

    def test_the_handshake_refuses_a_worker_it_cannot_speak_to(self, monkeypatch):
        monkeypatch.setattr(runtime, "_request",
                            lambda header, payload, timeout: (
                                {"protocol_version": 99, "backend": "deepfilternet",
                                 "device": "cpu", "parent_death": "pdeathsig"}, b""))
        with pytest.raises(runtime.CleanupRuntimeError, match="protocol"):
            runtime._handshake_with()

    def test_on_linux_the_workers_word_on_containment_is_the_only_evidence(self,
                                                                          monkeypatch):
        """``PR_SET_PDEATHSIG`` can only be set inside the child, so there the
        child's word has to be good. Windows is the other way round and is
        proved at the parent -- see ``_die_with_us``."""
        monkeypatch.setattr(runtime, "_request",
                            lambda header, payload, timeout: (
                                {"protocol_version": worker.PROTOCOL_VERSION,
                                 "backend": "deepfilternet", "device": "cpu",
                                 "parent_death": "pipe"}, b""))
        with pytest.raises(runtime.CleanupRuntimeError, match="lifetime"):
            runtime._handshake_with()


class TestItIsNotATextToSpeechEngine:
    def test_it_is_not_in_the_engine_selector(self):
        """I-1: one text-to-speech engine is selected for the whole WebUI, and
        cleaning a recording is not one of them. A row in the selector would be
        a third thing that could be "the engine"."""
        import mc_voice_engines as engines

        assert cleanup.ENGINE not in engines.ENGINES

    def test_its_status_does_not_depend_on_which_engine_speaks(self, host, voice_root):
        import mc_voice_engines as engines

        engines.select("kokoro")
        first = cleanup.status()
        engines.select("sopro")
        assert cleanup.status() == first


class TestTheInstalledRecord:
    def test_a_closure_that_changed_reads_as_stale(self, host, voice_root, monkeypatch):
        """The same rule the other runtimes follow: an installation from an
        older build is not silently used, because the closure is the identity."""
        paths.cleanup_runtime_manifest().parent.mkdir(parents=True, exist_ok=True)
        paths.cleanup_runtime_manifest().write_text(
            json.dumps({"closure": "not-the-current-one", "torch": "2.2.2"}),
            encoding="utf-8")
        monkeypatch.setattr(cleanup, "supported_platform", lambda: True)

        found = cleanup.status()
        assert found.runtime_ready is False
        assert "older build" in found.runtime_message

    def test_a_matching_closure_is_ready(self, host, voice_root, monkeypatch):
        paths.cleanup_runtime_manifest().parent.mkdir(parents=True, exist_ok=True)
        paths.cleanup_runtime_manifest().write_text(
            json.dumps({"closure": cleanup._closure_id(), "torch": "2.2.2",
                        "deepfilternet": "0.5.6", "python": "3.11.9"}),
            encoding="utf-8")
        monkeypatch.setattr(cleanup, "supported_platform", lambda: True)

        found = cleanup.status()
        assert found.runtime_ready is True
        assert "Torch 2.2.2" in found.runtime_message

    def test_the_interpreter_is_not_in_the_closure_fingerprint(self):
        """It has no committed digest to hash, and folding a
        recorded-on-this-machine value in would make the fingerprint mean
        something different on every installation."""
        first = cleanup._closure_id()
        assert len(first) == 16
        assert first == cleanup._closure_id()
