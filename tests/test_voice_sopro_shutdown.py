"""No engine outlives the WebUI. Not Kokoro, not Sopro, not both at once.

``tests/test_voice_shutdown.py`` is this file's older sibling and proves the
same thing for the Kokoro worker. Sopro gets its own because it is a *second
process, with a second lifecycle, out of a second dependency closure* -- and the
one thing a release gate must never do is assume that a guarantee proved for one
process holds for another that shares none of its code.

Every test below starts a real parent, which starts one or both real workers,
and then ends the parent in one particular way. Nothing is mocked on either side
of a pipe, because the entire question is what the operating system does when
nothing is left to ask.

    T-SOPRO-2   a hard kill while the worker is busy, which is the case
                pipe EOF cannot cover
    Gate S-4    parent death always kills the worker
    section 19  exactly one TTS worker is left running after an engine switch
    section 58  the inactive engine consumes no RAM

The last two are the ones an optional second engine makes possible to get wrong:
a switch that persisted the new choice and left the old worker resident would be
a WebUI holding a hundred and forty megabytes of Torch and four hundred of ONNX
at the same time, and every symptom of it would look like something else.
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

from conftest import FAKE_SOPRO_WORKER, FAKE_VOICE_WORKER

ROOT = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.skipif(os.name == "nt",
                                reason="the POSIX signal doors; Windows has its own gate")


PARENT = r'''
"""A minimal WebUI: it starts one or both voice workers and then ends."""

import json
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, PLAN_ROOT)

import mc_voice_engines
import mc_voice_models
import mc_voice_paths
import mc_voice_runtime
import mc_voice_sopro
import mc_voice_sopro_runtime

plan = json.loads(os.environ["MC_PARENT_PLAN"])

# Kokoro, answered exactly as the shutdown harness beside this one answers it.
ready = mc_voice_models.Status(
    runtime_ready=True, stt_ready=True, tts_ready=True,
    runtime_message="i", stt_message="i", tts_message="i", platform_supported=True,
    stt_id="whisper-small-int8", tts_id="kokoro-multi-lang-v1-cpu", tts_voice="af_heart")
mc_voice_models.status = lambda: ready
mc_voice_models.runtime_python = lambda: Path(sys.executable)
mc_voice_paths.worker_script = lambda: Path(plan["kokoro_script"])
mc_voice_models.bundle_paths = lambda kind: {"id": ready.stt_id if kind == "stt"
                                             else ready.tts_id}

# Sopro, answered the same way: installed, this interpreter, that script.
sopro_ready = mc_voice_sopro.Status(
    platform_supported=True, runtime_ready=True, model_ready=True,
    runtime_message="i", model_message="i", fingerprint="fingerprint01234")
mc_voice_sopro.status = lambda: sopro_ready
mc_voice_sopro.runtime_python = lambda: Path(sys.executable)
mc_voice_paths.sopro_worker_script = lambda: Path(plan["sopro_script"])
mc_voice_sopro.worker_environment = lambda: {
    "MC_FAKE_SOPRO": json.dumps(plan.get("sopro_plan") or {})}
mc_voice_sopro.worker_config = lambda: {
    "model_root": plan["root"], "model_id": "sopro-v2-turbo-cpu", "precision": "full",
    "steps": 2, "chunk_frames": 64, "fingerprint": "fingerprint01234", "voices": {}}

# The host's option store is a plain object here, so selecting an engine is a
# write to it and nothing else. This harness is about processes.
class _Options:
    def __init__(self):
        self.values = {}

    def set(self, name, value):
        self.values[name] = value

    def save(self, *args):
        pass

    def __getattr__(self, name):
        return self.values.get(name, "")

import types

shared = types.ModuleType("modules.shared")
shared.opts = _Options()
shared.config_filename = "config.json"
modules = types.ModuleType("modules")
modules.shared = shared
sys.modules.setdefault("modules", modules)
sys.modules["modules.shared"] = shared

mode = plan["mode"]
started = {}

if plan.get("start_kokoro"):
    mc_voice_runtime.ensure_started()
    started["kokoro"] = mc_voice_runtime.status()["pid"]

if plan.get("start_sopro"):
    shared.opts.set("model_chain_tts_engine", "sopro")
    mc_voice_sopro_runtime.ensure_started()
    started["sopro"] = mc_voice_sopro_runtime._process.pid

sys.stdout.write(json.dumps({"started": started, "parent": os.getpid()}) + "\n")
sys.stdout.flush()

if mode == "exit":
    # Nothing else. The atexit hooks are the whole test.
    raise SystemExit(0)

if mode == "switch":
    # Kokoro is running; select Sopro. Section 19: exactly one TTS worker after
    # this, and the one still alive is the newly selected engine's.
    shared.opts.set("model_chain_tts_engine", "kokoro")
    mc_voice_engines.select("sopro")
    mc_voice_sopro_runtime.ensure_started()
    sys.stdout.write(json.dumps({"after": mc_voice_sopro_runtime._process.pid}) + "\n")
    sys.stdout.flush()

if mode == "busy":
    # A request the fake worker never answers, standing in for a long solver
    # call. Fired on a thread so this process is still killable.
    def work():
        try:
            mc_voice_sopro_runtime.synthesize("hello", "sopro:clone:abc")
        except Exception:
            pass

    threading.Thread(target=work, daemon=True).start()
    time.sleep(0.6)

while True:
    time.sleep(0.05)
'''


def build(tmp_path, mode: str, *, kokoro=False, sopro=True, sopro_plan=None):
    kokoro_script = tmp_path / "fake_voice_worker.py"
    kokoro_script.write_text(FAKE_VOICE_WORKER, encoding="utf-8")
    sopro_script = tmp_path / "fake_sopro_worker.py"
    sopro_script.write_text(FAKE_SOPRO_WORKER, encoding="utf-8")
    parent = tmp_path / "sopro_parent_harness.py"
    parent.write_text(PARENT.replace("PLAN_ROOT", json.dumps(str(ROOT))), encoding="utf-8")

    environ = dict(os.environ)
    environ["MC_PARENT_PLAN"] = json.dumps({
        "mode": mode,
        "root": str(tmp_path),
        "kokoro_script": str(kokoro_script),
        "sopro_script": str(sopro_script),
        "start_kokoro": bool(kokoro),
        "start_sopro": bool(sopro),
        "sopro_plan": dict(sopro_plan or {}),
    })
    environ["MC_FAKE_VOICE"] = json.dumps({
        "behaviour": "normal", "transcript": "unused", "audio_hex": "",
        "alive_marker": str(tmp_path / "kokoro-alive.txt")})
    return parent, environ


def launch(tmp_path, mode: str, **kwargs):
    parent, environ = build(tmp_path, mode, **kwargs)
    process = subprocess.Popen([sys.executable, str(parent)], env=environ,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    line = process.stdout.readline()
    if not line:
        process.kill()
        raise AssertionError(f"the harness never reported a worker: {process.stderr.read()}")
    return process, json.loads(line)


def alive(pid) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except (OSError, ProcessLookupError, ValueError, TypeError):
        return False
    return True


def gone(pid, seconds: float = 15.0) -> bool:
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
    def test_an_ordinary_exit_leaves_no_sopro_worker(self, tmp_path):
        """Door B. The interpreter finishes and ``atexit`` runs.

        Sopro arms its own hook from its own first start, rather than relying on
        Kokoro's: an installation that selects Sopro and never touches Kokoro
        has never run a line of the other module.
        """
        process, found = launch(tmp_path, "exit")
        try:
            process.wait(timeout=30)
            assert gone(found["started"]["sopro"]), "a Sopro worker survived a normal exit"
        finally:
            reap(process)

    def test_sigint_leaves_no_sopro_worker(self, tmp_path):
        """Door C, and the one somebody performs several times a day: Ctrl+C."""
        process, found = launch(tmp_path, "wait")
        try:
            process.send_signal(signal.SIGINT)
            process.wait(timeout=30)
            assert gone(found["started"]["sopro"]), "a Sopro worker survived Ctrl+C"
        finally:
            reap(process)

    def test_sigterm_leaves_no_sopro_worker(self, tmp_path):
        process, found = launch(tmp_path, "wait")
        try:
            process.terminate()
            process.wait(timeout=30)
            assert gone(found["started"]["sopro"]), "a Sopro worker survived SIGTERM"
        finally:
            reap(process)

    def test_sigint_still_ends_the_parent(self, tmp_path):
        """The other half of chaining rather than replacing a handler. Two
        engines now chain onto the same signal, and a handler that swallowed
        SIGINT would leave a terminal whose Ctrl+C does nothing."""
        process, _found = launch(tmp_path, "wait")
        try:
            process.send_signal(signal.SIGINT)
            process.wait(timeout=30)
            assert process.poll() is not None
        finally:
            reap(process)


class TestBothEnginesAtOnce:
    def test_neither_engine_survives_an_ordinary_exit(self, tmp_path):
        """The case an optional second engine makes possible to get wrong.

        Both workers are started -- which is not a state the product allows, and
        is exactly why it is worth testing: if a stale page, a bug or a half-done
        switch ever left both resident, the exit path still has to end both.
        """
        process, found = launch(tmp_path, "exit", kokoro=True, sopro=True)
        try:
            process.wait(timeout=30)
            assert gone(found["started"]["kokoro"]), "a Kokoro worker survived the exit"
            assert gone(found["started"]["sopro"]), "a Sopro worker survived the exit"
        finally:
            reap(process)

    def test_neither_engine_survives_a_hard_kill(self, tmp_path):
        """SIGKILL runs no atexit hook, no signal handler and no Python at all.
        What is left is the operating system, and on Linux that is
        ``PR_SET_PDEATHSIG`` -- arranged inside each child by its own worker."""
        process, found = launch(tmp_path, "wait", kokoro=True, sopro=True)
        try:
            os.kill(process.pid, signal.SIGKILL)
            process.wait(timeout=30)
            assert gone(found["started"]["kokoro"], seconds=20.0), (
                "a Kokoro worker survived a hard kill of its parent")
            assert gone(found["started"]["sopro"], seconds=20.0), (
                "a Sopro worker survived a hard kill of its parent")
        finally:
            reap(process)

    def test_switching_engines_leaves_exactly_one_worker(self, tmp_path):
        """Section 19 and section 58. Kokoro is running, Sopro is selected: the
        Kokoro worker is stopped before the new choice is persisted, so the
        inactive engine consumes no RAM."""
        process, found = launch(tmp_path, "switch", kokoro=True, sopro=False)
        after = {}
        try:
            after = json.loads(process.stdout.readline())
            assert gone(found["started"]["kokoro"], seconds=20.0), (
                "the Kokoro worker was still resident after switching to Sopro")
            assert alive(after["after"]), "the newly selected engine did not start"
        finally:
            reap(process)
            gone(after.get("after"), seconds=20.0)


class TestTheHardKillDuringWork:
    def test_t_sopro_2_a_hard_kill_during_inference_leaves_no_worker(self, tmp_path):
        """The test the whole containment design exists for.

        The worker is deliberately *not* waiting on stdin: it is inside a loop
        standing in for a solver call, so the closing pipe is not something it
        will look at. Gate S-4's last clause -- "parent death always kills the
        worker" -- is only true because of the OS mechanism, and this is the only
        way to find out whether that mechanism actually fired.

        The wait for ``busy_marker`` is load-bearing rather than tidy. Without
        it the kill lands while the worker is still blocked on ``read_frame``,
        where EOF ends it perfectly well -- so the test passed with the OS
        mechanism removed, which is the worst possible state for a release gate
        to be in.
        """
        busy = tmp_path / "sopro-busy.txt"
        process, found = launch(tmp_path, "busy",
                                sopro_plan={"busy": True, "busy_marker": str(busy)})
        try:
            deadline = time.monotonic() + 20.0
            while not busy.exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            assert busy.exists(), "the worker never reached the busy loop"
            os.kill(process.pid, signal.SIGKILL)
            process.wait(timeout=30)
            assert gone(found["started"]["sopro"], seconds=20.0), (
                "a Sopro worker survived a hard kill of its parent while it was busy — "
                "pipe EOF is not enough and the OS mechanism did not fire")
        finally:
            reap(process)

    def test_no_sopro_command_is_left_in_the_process_list(self, tmp_path):
        """Read from the process list rather than from a pid we kept, because
        the failure this guards against is a worker whose parent forgot it."""
        from sopro_worker import worker as protocol

        process, _found = launch(tmp_path, "wait")
        try:
            os.kill(process.pid, signal.SIGKILL)
            process.wait(timeout=30)
        finally:
            reap(process)
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            found = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True)
            if protocol.MARKER not in (found.stdout or ""):
                return
            time.sleep(0.2)
        pytest.fail(f"a process carrying {protocol.MARKER} was left running")
