"""Storytime, borrowed for one job and never left running.

Cloning is the only part of Voice Chat that runs somebody else's program, and
the release-blocking requirement is not that it produces a good voice -- it is
that it cannot outlive the WebUI, cannot be discovered to be impossible after
four hours of optimization, cannot reach a graphics device, and cannot register
a voice that has not been proved through the ordinary speech runtime.

The Storytime here is a Python script with the real CLI shape (``conftest
.FAKE_STORYTIME``). It prints the progress lines the parser reads, writes the
checkpoint files Abort deletes, stops on SIGINT after its current step, and
forks a child of its own -- because a grandchild holding the output pipe is the
specific thing that turns "read until end of output" into a supervisor that
never notices its job finished.
"""

from __future__ import annotations

import struct
import time
from pathlib import Path

import pytest

import mc_voice_clone as cloning
import mc_voice_paths as paths


def settle(seconds: float = 30.0) -> dict:
    """Wait for the job to stop being active. Returns its final state."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        found = cloning.state()
        if not found["active"]:
            return found
        time.sleep(0.05)
    return cloning.state()


def running(pid: int) -> bool:
    """Running, not merely un-reaped: a killed orphan whose parent has gone is
    a zombie until something reaps it, and a zombie answers ``kill(pid, 0)``."""
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except OSError:
        return False
    return state != "Z"


class TestInstallation:
    def test_t_clone_1_a_missing_bundle_reports_not_installed(self, voice_root):
        found = cloning.installation()
        assert found["state"] in ("not_installed", "unsupported")

    def test_t_clone_2_every_missing_part_is_named_separately(self, tmp_path, voice_root,
                                                             monkeypatch):
        """Section 61. "Cloning is not installed" sends somebody back to the
        beginning; "the speaker encoder is missing" names the one file."""
        base = tmp_path / "half"
        (base / "bin").mkdir(parents=True)
        (base / "assets").mkdir(parents=True)
        (base / "assets" / "kokoro.onnx").write_bytes(b"x")
        found = cloning.validate(base)
        failed = {item["item"] for item in found["checks"] if not item["ok"]}
        assert "assets/spk_encoder.onnx" in failed
        assert "assets/kokoro.onnx" not in failed
        assert found["ok"] is False

    def test_t_clone_3_a_bundle_that_does_not_answer_is_not_marked_usable(self, storytime,
                                                                         voice_root):
        (storytime / "bin" / "storytime").write_text("#!/bin/false\n", encoding="utf-8")
        (storytime / "bin" / "storytime").chmod(0o755)
        with pytest.raises(cloning.CloneError, match="version check"):
            cloning.adopt(str(storytime))

    def test_a_validated_bundle_is_adopted_and_records_its_version(self, storytime,
                                                                   voice_root):
        found = cloning.adopt(str(storytime))
        assert found["state"] == "installed"
        assert found["cpu_validated"] is True
        assert "0.0.0-test" in cloning._record()["version"]

    def test_one_click_says_what_is_missing_rather_than_resolving_a_package(self,
                                                                           storytime,
                                                                           voice_root):
        """Section 78. A version, a URL, a size and a checksum before anything
        is downloaded -- and inventing one of those would be the trust model the
        rest of this feature refuses."""
        with pytest.raises(cloning.CloneError, match="no pinned cloning bundle"):
            cloning.install()

    def test_an_unsupported_platform_says_so_and_leaves_voice_alone(self, voice_root,
                                                                    monkeypatch):
        """Section 59. macOS runs Storytime's ONNX path on CoreML, so
        ``--backend onnx`` is not a CPU-only claim to make there."""
        monkeypatch.setattr(cloning, "supported", lambda: False)
        found = cloning.installation()
        assert found["state"] == "unsupported"
        assert found["supported"] is False
        assert "unaffected" in found["message"]

    def test_the_platform_is_named_the_way_it_names_itself(self, voice_root, monkeypatch):
        """``platform.system()`` is lowercased on the way through to a directory
        name, which is right there and wrong in a sentence: "not offered on
        windows" reads like a typo rather than like a fact about this PC."""
        monkeypatch.setattr(cloning, "supported", lambda: False)
        for reported, wanted in (("windows", "Windows"), ("darwin", "macOS"),
                                 ("linux", "Linux")):
            monkeypatch.setattr(cloning.models, "current_platform",
                                lambda reported=reported: (reported, "amd64", "cp313"))
            assert wanted in cloning.installation()["message"]

    def test_a_platform_nobody_has_heard_of_is_still_said_plainly(self, voice_root,
                                                                  monkeypatch):
        monkeypatch.setattr(cloning, "supported", lambda: False)
        monkeypatch.setattr(cloning.models, "current_platform",
                            lambda: ("haiku", "amd64", "cp313"))
        assert "haiku" in cloning.installation()["message"]


class TestTheReferenceRecording:
    def test_t_clone_5_a_recording_is_normalized_to_mono_24_khz(self, reference_wav):
        """Storytime's own instructions are ``ffmpeg -ar 24000 -ac 1``, and
        asking somebody to run ffmpeg before pressing a button is the difference
        between a feature and a procedure."""
        found, seconds = cloning.normalize_wav(reference_wav(12.0, 48000, 2))
        channels, rate = struct.unpack_from("<HI", found, 22)
        assert channels == 1
        assert rate == 24000
        assert seconds == pytest.approx(12.0, abs=0.1)

    def test_a_recording_that_is_already_right_is_left_alone(self, reference_wav):
        found, _seconds = cloning.normalize_wav(reference_wav(12.0, 24000, 1))
        assert struct.unpack_from("<I", found, 24)[0] == 24000

    def test_a_recording_that_is_too_short_or_too_long_is_refused(self, reference_wav):
        with pytest.raises(cloning.CloneError, match="at least"):
            cloning.normalize_wav(reference_wav(1.0))
        with pytest.raises(cloning.CloneError, match="limit"):
            cloning.normalize_wav(reference_wav(200.0, 8000, 1))

    def test_something_that_is_not_a_wav_is_refused(self):
        for data in (b"", b"not a wav at all", b"RIFF" + b"\x00" * 40):
            with pytest.raises(cloning.CloneError):
                cloning.normalize_wav(data)

    def test_compressed_audio_is_refused_rather_than_guessed_at(self, reference_wav):
        raw = bytearray(reference_wav(12.0, 24000, 1))
        struct.pack_into("<H", raw, 20, 3)  # IEEE float, not PCM
        with pytest.raises(cloning.CloneError, match="16-bit PCM"):
            cloning.normalize_wav(bytes(raw))


class TestOneJob:
    def test_t_clone_9_a_finished_clone_becomes_a_registered_voice(self, storytime,
                                                                   voice_registry,
                                                                   kokoro_bundle,
                                                                   reference_wav):
        cloning.adopt(str(storytime))
        cloning.start("Alice", "en-GB", reference_wav())
        found = settle()
        assert found["status"] == "complete", found
        entry = voice_registry.custom()[0]
        assert entry["display_name"] == "Alice"
        assert entry["language"] == "en-GB"
        assert entry["sid"] == 53
        assert voice_registry.spoken[-1] == 53, "the clone was registered without being spoken"

    def test_t_clone_11_a_finished_clone_is_shown_with_an_asterisk(self, storytime,
                                                                   voice_registry,
                                                                   kokoro_bundle,
                                                                   reference_wav):
        cloning.adopt(str(storytime))
        cloning.start("Alice", "en-US", reference_wav())
        settle()
        assert voice_registry.custom()[0]["label"] == "* Alice"

    def test_t_clone_6_progress_is_parsed_while_the_job_runs(self, storytime,
                                                             voice_registry,
                                                             kokoro_bundle,
                                                             reference_wav, monkeypatch):
        monkeypatch.setenv("MC_FAKE_CLONE_DELAY", "0.05")
        cloning.adopt(str(storytime))
        cloning.start("Alice", "en-US", reference_wav())
        seen = []
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            found = cloning.state()
            if found.get("step"):
                seen.append((found["step"], found["percent"], found["score"]))
            if not found["active"]:
                break
            time.sleep(0.05)
        assert seen, "no progress was ever reported"
        assert seen[-1][1] > 0
        assert seen[-1][2] is not None, "the score line was never parsed"

    def test_t_clone_4_a_second_job_is_refused_while_one_is_running(self, storytime,
                                                                    voice_registry,
                                                                    kokoro_bundle,
                                                                    reference_wav,
                                                                    monkeypatch):
        monkeypatch.setenv("MC_FAKE_CLONE_DELAY", "0.1")
        cloning.adopt(str(storytime))
        cloning.start("Alice", "en-US", reference_wav())
        try:
            with pytest.raises(cloning.CloneError, match="already being cloned"):
                cloning.start("Bob", "en-US", reference_wav())
        finally:
            cloning.abort()

    def test_t_clone_10_a_failed_run_registers_nothing(self, storytime, voice_registry,
                                                       kokoro_bundle, reference_wav,
                                                       monkeypatch):
        monkeypatch.setenv("MC_FAKE_CLONE_FAIL", "1")
        cloning.adopt(str(storytime))
        cloning.start("Alice", "en-US", reference_wav())
        found = settle()
        assert found["status"] == "failed"
        assert "exit code" in found["error"]
        assert voice_registry.custom() == []

    def test_the_state_a_browser_sees_carries_no_paths(self, storytime, voice_registry,
                                                       kokoro_bundle, reference_wav):
        """Section 81: no filesystem paths in browser-visible messages."""
        cloning.adopt(str(storytime))
        cloning.start("Alice", "en-US", reference_wav())
        found = cloning.state()
        cloning.abort()
        assert "reference" not in found and "output" not in found
        assert str(paths.data_root()) not in repr(found)

    def test_t_clone_17_no_graphics_device_is_visible_to_the_worker(self, storytime,
                                                                    voice_registry,
                                                                    kokoro_bundle,
                                                                    reference_wav):
        """The fake exits non-zero if it can see one, so a completed clone is
        the assertion. Section 67: the claim does not rest on trusting somebody
        else's build flags."""
        assert cloning._environment()["CUDA_VISIBLE_DEVICES"] == ""
        cloning.adopt(str(storytime))
        cloning.start("Alice", "en-US", reference_wav())
        assert settle()["status"] == "complete"

    def test_t_clone_12_no_cloning_process_is_left_after_a_completed_job(self, storytime,
                                                                        voice_registry,
                                                                        kokoro_bundle,
                                                                        reference_wav):
        cloning.adopt(str(storytime))
        cloning.start("Alice", "en-US", reference_wav())
        settle()
        assert cloning._process is None
        assert "worker_pid" not in cloning.state()


class TestAbort:
    def test_t_clone_7_abort_ends_the_whole_process_tree(self, storytime, voice_registry,
                                                         kokoro_bundle, reference_wav,
                                                         monkeypatch):
        """Section 69, and the reason it signals a process *group*: terminating
        the one pid Storytime was given leaves anything it started running."""
        monkeypatch.setenv("MC_FAKE_CLONE_DELAY", "0.2")
        monkeypatch.setenv("MC_FAKE_CLONE_CHILD", "1")
        cloning.adopt(str(storytime))
        marker = storytime / "assets" / "child.pid"
        cloning.start("Alice", "en-US", reference_wav())
        deadline = time.monotonic() + 15
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert marker.exists(), "the fake never forked a child to be killed"
        child = int(marker.read_text())
        cloning.abort()
        deadline = time.monotonic() + 10
        while running(child) and time.monotonic() < deadline:
            time.sleep(0.1)
        assert not running(child), "a grandchild survived the abort"
        assert cloning.state()["status"] == "aborted"

    def test_t_clone_8_abort_removes_the_reference_and_the_checkpoints(self, storytime,
                                                                       voice_registry,
                                                                       kokoro_bundle,
                                                                       reference_wav,
                                                                       monkeypatch):
        monkeypatch.setenv("MC_FAKE_CLONE_DELAY", "0.2")
        cloning.adopt(str(storytime))
        cloning.start("Alice", "en-US", reference_wav())
        assert list(paths.reference_root().glob("*.wav"))
        time.sleep(0.4)
        cloning.abort()
        settle()
        assert not list(paths.reference_root().glob("*.wav"))
        assert not list((storytime / "assets" / "voices").glob("*.bin.temp*"))
        assert voice_registry.custom() == []

    def test_aborting_when_nothing_is_running_is_harmless(self, storytime, voice_root):
        assert cloning.abort()["status"] in ("idle", "aborted", "complete", "failed")

    def test_a_successful_clone_leaves_no_reference_recording_either(self, storytime,
                                                                     voice_registry,
                                                                     kokoro_bundle,
                                                                     reference_wav):
        """Section 79: the default policy is deletion, on both endings."""
        cloning.adopt(str(storytime))
        cloning.start("Alice", "en-US", reference_wav())
        settle()
        assert not list(paths.reference_root().glob("*.wav"))


class TestShutdown:
    def test_t_clone_13_the_webui_closing_ends_the_process_tree(self, storytime,
                                                                voice_registry,
                                                                kokoro_bundle,
                                                                reference_wav,
                                                                monkeypatch):
        """Release blocker nine. Storytime's own default run is hours, so this
        is not a theoretical concern."""
        monkeypatch.setenv("MC_FAKE_CLONE_DELAY", "0.2")
        monkeypatch.setenv("MC_FAKE_CLONE_CHILD", "1")
        cloning.adopt(str(storytime))
        marker = storytime / "assets" / "child.pid"
        cloning.start("Alice", "en-US", reference_wav())
        deadline = time.monotonic() + 15
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        child = int(marker.read_text())
        cloning.shutdown()
        deadline = time.monotonic() + 10
        while running(child) and time.monotonic() < deadline:
            time.sleep(0.1)
        assert not running(child)

    def test_a_clone_the_webui_interrupted_is_not_called_a_failure(self, storytime,
                                                                   voice_registry,
                                                                   kokoro_bundle,
                                                                   reference_wav,
                                                                   monkeypatch):
        """Section 74. Telling somebody their clone failed when what happened is
        that they closed the WebUI is a lie about their own action."""
        monkeypatch.setenv("MC_FAKE_CLONE_DELAY", "0.2")
        cloning.adopt(str(storytime))
        cloning.start("Alice", "en-US", reference_wav())
        time.sleep(0.3)
        cloning.shutdown()
        assert cloning.state()["status"] == "interrupted"

    def test_shutdown_with_no_job_is_harmless(self, voice_root):
        cloning.shutdown()
        assert cloning.state()["status"] == "idle"


class TestIndependence:
    def test_nothing_here_imports_the_model_or_image_managers(self):
        """Section 67 and 91: cloning must not register with, be unloaded by, or
        wait for anything that manages VRAM."""
        import ast

        source = Path(cloning.__file__).read_text(encoding="utf-8")
        banned = {"mc_memory", "mc_broker", "mc_plan", "mc_llm_runtime", "torch",
                  "mc_llm_managed_models"}
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                assert all(alias.name.split(".")[0] not in banned for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in banned, node.module

    def test_the_command_is_an_argument_array_and_never_a_shell_string(self):
        """Section 77: never build a command from a display name."""
        import ast

        source = Path(cloning.__file__).read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in ("Popen", "run", "call", "check_output"):
                    for keyword in node.keywords:
                        assert keyword.arg != "shell", "a cloning subprocess used a shell"
