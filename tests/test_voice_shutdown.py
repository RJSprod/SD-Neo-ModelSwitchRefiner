"""No Voice Worker outlives the WebUI. This file is a release gate.

The repository has already shipped the other version of this bug once: *"if I
kill the webui process, there tends to be llama-server.exe running on my
system."* The fix needed four separate doors because there are four separate
ways a process ends, and only one of them runs any Python you wrote. Voice Chat
has a fifth -- the worker's own pipe -- and still needs all of the others.

Every test below starts a *real* parent process, which starts a *real* worker,
and then ends the parent in one particular way. Nothing is mocked on either
side of the pipe, because the entire question is what the operating system does
when nothing is left to ask.

    T-SHUT-1  the extension unload callback
    T-SHUT-2  ordinary interpreter exit
    T-SHUT-3  SIGINT
    T-SHUT-4  SIGTERM
    T-SHUT-5  a hard kill *while the worker is inside native work*
    T-SHUT-6  a worker that ignores the polite request (in test_voice_runtime)
    T-SHUT-7  a start that fails after Popen (in test_voice_runtime)
    T-SHUT-8  unload and rebuild
    T-SHUT-9  nothing carrying the worker marker is left in the process list

T-SHUT-5 is the one the others cannot stand in for, and the reason the design
intent makes OS containment a support requirement rather than a nicety. A worker
that is blocked reading its input notices EOF; a worker that is inside a
four-second ONNX call does not, and that is precisely the moment somebody
force-quits a WebUI that "has hung".
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from conftest import FAKE_VOICE_WORKER

ROOT = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.skipif(os.name == "nt",
                                reason="the POSIX signal doors; Windows has its own gate")


PARENT = r'''
"""A minimal WebUI: it starts a voice worker and then ends in a chosen way."""

import json
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, PLAN_ROOT)

import mc_voice_models
import mc_voice_paths
import mc_voice_runtime

plan = json.loads(os.environ["MC_PARENT_PLAN"])

ready = mc_voice_models.Status(
    runtime_ready=True, stt_ready=True, tts_ready=True,
    runtime_message="i", stt_message="i", tts_message="i", platform_supported=True,
    stt_id="whisper-small-int8", tts_id="kokoro-multi-lang-v1-cpu", tts_voice="af_heart")

mc_voice_models.status = lambda: ready
mc_voice_models.runtime_python = lambda: Path(sys.executable)
mc_voice_paths.worker_script = lambda: Path(plan["script"])
mc_voice_models.bundle_paths = lambda kind: {"id": ready.stt_id if kind == "stt"
                                             else ready.tts_id}

mc_voice_runtime.ensure_started()
sys.stdout.write(json.dumps({"worker": mc_voice_runtime.status()["pid"],
                             "parent": os.getpid()}) + "\n")
sys.stdout.flush()

mode = plan["mode"]

if mode == "exit":
    # Nothing else. The atexit hook is the whole test.
    raise SystemExit(0)

if mode == "unload":
    # What Forge calls when an extension is reloaded, twice, with a fresh start
    # in between -- so a second UI build cannot inherit a stale handle.
    first = mc_voice_runtime.status()["pid"]
    mc_voice_runtime.shutdown()
    mc_voice_runtime.ensure_started()
    second = mc_voice_runtime.status()["pid"]
    sys.stdout.write(json.dumps({"first": first, "second": second}) + "\n")
    sys.stdout.flush()
    mc_voice_runtime.shutdown()
    raise SystemExit(0)

if mode == "busy":
    # A request the fake worker never answers, standing in for a long native
    # inference call. Fired on a thread so this process is still killable.
    threading.Thread(target=lambda: mc_voice_runtime.transcribe(b"x" * 64),
                     daemon=True).start()
    time.sleep(0.6)

while True:
    time.sleep(0.05)
'''


def build(tmp_path, mode: str, behaviour: str = "normal") -> tuple[Path, dict]:
    script = tmp_path / "fake_voice_worker.py"
    script.write_text(FAKE_VOICE_WORKER, encoding="utf-8")
    parent = tmp_path / "parent_harness.py"
    parent.write_text(PARENT.replace("PLAN_ROOT", json.dumps(str(ROOT))), encoding="utf-8")

    environ = dict(os.environ)
    environ["MC_PARENT_PLAN"] = json.dumps({"script": str(script), "mode": mode})
    environ["MC_FAKE_VOICE"] = json.dumps({
        "behaviour": behaviour,
        "transcript": "unused",
        "audio_hex": "",
        "alive_marker": str(tmp_path / "alive.txt"),
    })
    return parent, environ


def launch(tmp_path, mode: str, behaviour: str = "normal"):
    parent, environ = build(tmp_path, mode, behaviour)
    process = subprocess.Popen([sys.executable, str(parent)], env=environ,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    line = process.stdout.readline()
    assert line, f"the harness never reported a worker: {process.stderr.read()}"
    return process, json.loads(line)


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def gone(pid: int, seconds: float = 10.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not alive(pid):
            return True
        time.sleep(0.02)
    return not alive(pid)


def reap(process) -> None:
    if process.poll() is None:
        process.kill()
    try:
        process.communicate(timeout=10)
    except Exception:
        pass


class TestTheDoors:
    def test_t_shut_2_an_ordinary_exit_leaves_no_worker(self, tmp_path):
        """Door B. The interpreter finishes and ``atexit`` runs."""
        process, found = launch(tmp_path, "exit")
        try:
            process.wait(timeout=30)
            assert gone(found["worker"]), "a worker survived a normal parent exit"
        finally:
            reap(process)

    def test_t_shut_3_sigint_leaves_no_worker(self, tmp_path):
        """Door C, and the one somebody performs several times a day: Ctrl+C."""
        process, found = launch(tmp_path, "wait")
        try:
            process.send_signal(signal.SIGINT)
            process.wait(timeout=30)
            assert gone(found["worker"]), "a worker survived Ctrl+C"
        finally:
            reap(process)

    def test_t_shut_4_sigterm_leaves_no_worker(self, tmp_path):
        process, found = launch(tmp_path, "wait")
        try:
            process.terminate()
            process.wait(timeout=30)
            assert gone(found["worker"]), "a worker survived SIGTERM"
        finally:
            reap(process)

    def test_t_shut_3_sigint_still_ends_the_parent(self, tmp_path):
        """The other half of chaining rather than replacing a handler. A
        handler of ours that swallowed SIGINT would leave a window nobody can
        close, which is a worse bug than the one it was added to fix."""
        process, _found = launch(tmp_path, "wait")
        try:
            process.send_signal(signal.SIGINT)
            process.wait(timeout=30)
            assert process.poll() is not None
        finally:
            reap(process)


class TestTheHardKill:
    def test_t_shut_5_a_hard_kill_during_native_work_leaves_no_worker(self, tmp_path):
        """The test the whole containment design exists for.

        The worker is deliberately *not* waiting on stdin: it is inside a loop
        standing in for a native ONNX call, so the closing pipe is not something
        it will look at. The parent is then killed with SIGKILL, which runs no
        atexit hook, no signal handler and no Python at all. What is left is the
        operating system, and on Linux that is PR_SET_PDEATHSIG."""
        process, found = launch(tmp_path, "busy", behaviour="busy_native")
        try:
            os.kill(process.pid, signal.SIGKILL)
            process.wait(timeout=30)
            assert gone(found["worker"], seconds=15.0), (
                "a voice worker survived a hard kill of its parent while it was inside "
                "native work — pipe EOF is not enough and the OS mechanism did not fire")
        finally:
            reap(process)

    def test_t_shut_9_no_worker_command_is_left_in_the_process_list(self, tmp_path):
        """T-SHUT-9, read from the process list rather than from a pid we kept.

        A pid can be reused; a command line carrying this feature's own marker
        cannot be anything but one of ours."""
        process, found = launch(tmp_path, "busy", behaviour="busy_native")
        try:
            os.kill(process.pid, signal.SIGKILL)
            process.wait(timeout=30)
            assert gone(found["worker"], seconds=15.0)
        finally:
            reap(process)

        marker = "--model-chain-voice-worker"
        leftover = []
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                command = (entry / "cmdline").read_bytes().decode("utf-8", "replace")
            except OSError:
                continue
            if marker in command and str(tmp_path) in command:
                leftover.append(entry.name)
        assert not leftover, f"voice workers left running: {leftover}"


class TestReload:
    def test_t_shut_8_unload_and_rebuild_leaves_one_worker_and_then_none(self, tmp_path):
        """Forge can call an extension's callbacks more than once. Two starts
        must not mean two workers, and the first handle must not survive."""
        process, found = launch(tmp_path, "unload")
        try:
            reloaded = json.loads(process.stdout.readline())
            process.wait(timeout=30)
        finally:
            reap(process)

        assert reloaded["first"] != reloaded["second"], "the reload reused a dead handle"
        assert gone(reloaded["first"]), "the first worker survived the unload"
        assert gone(reloaded["second"]), "the second worker survived the exit"


class TestTheUnloadCallback:
    def test_t_shut_1_the_extension_unload_callback_stops_the_worker(self, fake_worker):
        """Door A, in process. What Forge calls when an extension is reloaded."""
        import mc_voice_runtime

        mc_voice_runtime.ensure_started()
        pid = mc_voice_runtime.status()["pid"]
        mc_voice_runtime.shutdown()
        assert gone(pid)

    def test_the_extension_registers_that_callback(self, monkeypatch):
        """And that ``scripts/model_chain.py`` actually calls it -- a shutdown
        function nobody invokes is the exact shape of the bug this file is
        about."""
        sys.path.insert(0, str(ROOT / "scripts"))
        import model_chain

        called = []
        monkeypatch.setattr(model_chain.mc_voice_runtime, "shutdown",
                            lambda: called.append(True))
        monkeypatch.setattr(model_chain.mc_memory, "release_all", lambda: None)
        monkeypatch.setattr(model_chain.mc_llm_runtime, "shutdown", lambda: None)
        monkeypatch.setattr(model_chain.mc_broker, "clear", lambda: None)

        model_chain._on_script_unloaded()
        assert called == [True]

    def test_a_voice_failure_does_not_stop_the_rest_of_the_unload(self, monkeypatch):
        """I-3 is not only about imports. A speech process that survived because
        releasing an image model raised would be exactly the coupling this
        feature was designed without -- so each side is in its own try."""
        sys.path.insert(0, str(ROOT / "scripts"))
        import model_chain

        released = []

        def explode():
            raise RuntimeError("voice is broken")

        monkeypatch.setattr(model_chain.mc_voice_runtime, "shutdown", explode)
        monkeypatch.setattr(model_chain.mc_memory, "release_all",
                            lambda: released.append("memory"))
        monkeypatch.setattr(model_chain.mc_llm_runtime, "shutdown",
                            lambda: released.append("llm"))
        monkeypatch.setattr(model_chain.mc_broker, "clear", lambda: released.append("broker"))

        model_chain._on_script_unloaded()
        assert released == ["memory", "llm", "broker"]
