"""No engine outlives the WebUI, and now there are three of them.

``tests/test_voice_shutdown.py`` proves it for Kokoro and
``tests/test_voice_sopro_shutdown.py`` for Sopro. PocketTTS gets its own for the
reason Sopro got its own: it is a *third process, with a third lifecycle, out of
a third dependency closure*, and the one thing a release gate must never do is
assume a guarantee proved for one process holds for another that shares none of
its code.

Every test below starts a real parent, which starts one or more real workers, and
then ends the parent in one particular way. Nothing is mocked on either side of a
pipe, because the entire question is what the operating system does when nothing
is left to ask.

    T-SHUT-P1..P4   ordinary exit, SIGINT, SIGTERM, and the hard kill that runs
                    no Python at all
    T-SHUT-P6       an engine switch leaves exactly one worker
    GATE P-9        a hard kill *during a drain*, which is the state this engine
                    spends time in that neither of the others does

That last one is Pocket's alone and is the reason this file exists rather than a
parametrisation of Sopro's. A Pocket worker that has been asked to stop is a
worker that is deliberately still computing, and "the WebUI closed while an
abandoned unit was finishing" is an ordinary Tuesday here rather than a corner
case. If containment did not cover it, the symptom would be a PyTorch process
left running after every conversation that ended with a Stop.
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

from conftest import FAKE_POCKET_WORKER, FAKE_SOPRO_WORKER, FAKE_VOICE_WORKER

ROOT = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.skipif(os.name == "nt",
                                reason="the POSIX signal doors; Windows has its own gate")


PARENT = r'''
"""A minimal WebUI: it starts one or more voice workers and then ends."""

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
import mc_voice_pocket
import mc_voice_pocket_runtime
import mc_voice_runtime
import mc_voice_sopro
import mc_voice_sopro_runtime

plan = json.loads(os.environ["MC_PARENT_PLAN"])

# Kokoro, answered exactly as the two shutdown harnesses beside this one answer it.
ready = mc_voice_models.Status(
    runtime_ready=True, stt_ready=True, tts_ready=True,
    runtime_message="i", stt_message="i", tts_message="i", platform_supported=True,
    stt_id="whisper-small-int8", tts_id="kokoro-multi-lang-v1-cpu", tts_voice="af_heart")
mc_voice_models.status = lambda: ready
mc_voice_models.runtime_python = lambda: Path(sys.executable)
mc_voice_paths.worker_script = lambda: Path(plan["kokoro_script"])
mc_voice_models.bundle_paths = lambda kind: {"id": ready.stt_id if kind == "stt"
                                             else ready.tts_id}

sopro_ready = mc_voice_sopro.Status(
    platform_supported=True, runtime_ready=True, model_ready=True,
    runtime_message="i", model_message="i", fingerprint="fingerprint01234")
mc_voice_sopro.status = lambda: sopro_ready
mc_voice_sopro.runtime_python = lambda: Path(sys.executable)
mc_voice_paths.sopro_worker_script = lambda: Path(plan["sopro_script"])
mc_voice_sopro.worker_environment = lambda: {"MC_FAKE_SOPRO": json.dumps({})}
mc_voice_sopro.worker_config = lambda: {
    "model_root": plan["root"], "model_id": "sopro-v2-turbo-cpu", "precision": "full",
    "steps": 2, "chunk_frames": 64, "fingerprint": "fingerprint01234", "voices": {}}

# PocketTTS, answered the same way: installed, this interpreter, that script.
pocket_ready = mc_voice_pocket.Status(
    platform_supported=True, runtime_ready=True, speech_model_ready=True,
    official_voices_ready=True, cloning_ready=True,
    runtime_message="i", model_message="i", cloning_message="i",
    model_id="english", fingerprint="pocketprint01234")
mc_voice_pocket.status = lambda: pocket_ready
mc_voice_pocket.runtime_python = lambda: Path(sys.executable)
mc_voice_paths.pocket_worker_script = lambda: Path(plan["pocket_script"])
mc_voice_pocket.worker_environment = lambda: {
    "MC_FAKE_POCKET": json.dumps(plan.get("pocket_plan") or {})}
mc_voice_pocket.worker_config = lambda: {
    "model_root": plan["root"], "config_path": plan["root"] + "/model.json",
    "model_id": "english", "precision": "full", "sampler_steps": 1,
    "fingerprint": "pocketprint01234", "state_schema": 1, "sample_rate": 24000,
    "cloning_ready": True, "voices": {}}


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

if plan.get("start_pocket"):
    shared.opts.set("model_chain_tts_engine", "pocket")
    mc_voice_pocket_runtime.ensure_started()
    started["pocket"] = mc_voice_pocket_runtime._process.pid


class _Turn:
    """Only what the runtime touches. This harness is about processes."""

    def __init__(self, identifier):
        self.id = identifier
        self.sample_rate = 0
        self.streaming = ""
        self.cancelled = threading.Event()
        self.draining = False

    def offer_audio(self, pcm, rate=0):
        return not self.cancelled.is_set()

    def note_segment(self, **values):
        pass

    def audio_finished(self):
        pass

    def audio_failed(self, reason):
        pass

    def cancel(self, reason="user"):
        self.cancelled.set()
        return True

    def drain_audio(self):
        pass

    def interrupting(self, chars=None, audio_ms=None):
        self.draining = True

    def interrupted(self):
        self.draining = False


if mode == "draining":
    # A turn is interrupted and its unit never finishes, so this parent is left
    # holding a worker that is deliberately still computing. That is the state
    # Pocket spends real time in and neither other engine ever reaches.
    turn = _Turn("drain")
    mc_voice_pocket_runtime.begin_turn(turn, "pocket:official:alba", None)
    mc_voice_pocket_runtime.send_segment(turn, "A long sentence.")
    time.sleep(0.4)
    turn.cancel("user")
    mc_voice_pocket_runtime.interrupt_turn(turn)
    time.sleep(0.3)

sys.stdout.write(json.dumps({"started": started, "parent": os.getpid()}) + "\n")
sys.stdout.flush()

if mode == "exit":
    raise SystemExit(0)

if mode == "switch":
    shared.opts.set("model_chain_tts_engine", "sopro")
    mc_voice_engines.select("pocket")
    mc_voice_pocket_runtime.ensure_started()
    sys.stdout.write(json.dumps({"after": mc_voice_pocket_runtime._process.pid}) + "\n")
    sys.stdout.flush()

if mode == "busy":
    def work():
        try:
            mc_voice_pocket_runtime.synthesize("hello", "pocket:official:alba")
        except Exception:
            pass

    threading.Thread(target=work, daemon=True).start()
    time.sleep(0.6)

while True:
    time.sleep(0.05)
'''


def build(tmp_path, mode: str, *, kokoro=False, sopro=False, pocket=True, pocket_plan=None):
    kokoro_script = tmp_path / "fake_voice_worker.py"
    kokoro_script.write_text(FAKE_VOICE_WORKER, encoding="utf-8")
    sopro_script = tmp_path / "fake_sopro_worker.py"
    sopro_script.write_text(FAKE_SOPRO_WORKER, encoding="utf-8")
    pocket_script = tmp_path / "fake_pocket_worker.py"
    pocket_script.write_text(FAKE_POCKET_WORKER, encoding="utf-8")
    parent = tmp_path / "pocket_parent_harness.py"
    parent.write_text(PARENT.replace("PLAN_ROOT", json.dumps(str(ROOT))), encoding="utf-8")

    environ = dict(os.environ)
    environ["MC_PARENT_PLAN"] = json.dumps({
        "mode": mode,
        "root": str(tmp_path),
        "kokoro_script": str(kokoro_script),
        "sopro_script": str(sopro_script),
        "pocket_script": str(pocket_script),
        "start_kokoro": bool(kokoro),
        "start_sopro": bool(sopro),
        "start_pocket": bool(pocket),
        "pocket_plan": dict(pocket_plan or {}),
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
    def test_t_shut_p1_an_ordinary_exit_leaves_no_pocket_worker(self, tmp_path):
        """Door B. The interpreter finishes and ``atexit`` runs.

        PocketTTS arms its own hook from its own first start rather than relying
        on either other engine's: an installation that selects Pocket and never
        touches the others has never run a line of their modules.
        """
        process, found = launch(tmp_path, "exit")
        try:
            process.wait(timeout=30)
            assert gone(found["started"]["pocket"]), (
                "a PocketTTS worker survived a normal exit")
        finally:
            reap(process)

    def test_t_shut_p2_sigint_leaves_no_pocket_worker(self, tmp_path):
        """Door C, and the one somebody performs several times a day: Ctrl+C."""
        process, found = launch(tmp_path, "wait")
        try:
            process.send_signal(signal.SIGINT)
            process.wait(timeout=30)
            assert gone(found["started"]["pocket"]), (
                "a PocketTTS worker survived Ctrl+C")
        finally:
            reap(process)

    def test_t_shut_p3_sigterm_leaves_no_pocket_worker(self, tmp_path):
        process, found = launch(tmp_path, "wait")
        try:
            process.terminate()
            process.wait(timeout=30)
            assert gone(found["started"]["pocket"]), (
                "a PocketTTS worker survived SIGTERM")
        finally:
            reap(process)

    def test_sigint_still_ends_the_parent(self, tmp_path):
        """The other half of chaining rather than replacing a handler. Three
        engines now chain onto the same signal, and a handler that swallowed
        SIGINT would leave a terminal whose Ctrl+C does nothing."""
        process, _found = launch(tmp_path, "wait")
        try:
            process.send_signal(signal.SIGINT)
            process.wait(timeout=30)
            assert process.poll() is not None
        finally:
            reap(process)


class TestAllThreeEnginesAtOnce:
    def test_none_of_them_survives_an_ordinary_exit(self, tmp_path):
        """Not a state the product allows, and exactly why it is worth testing:
        if a stale page, a bug or a half-done switch ever left all three
        resident, the exit path still has to end all three."""
        process, found = launch(tmp_path, "exit", kokoro=True, sopro=True, pocket=True)
        try:
            process.wait(timeout=30)
            for name in ("kokoro", "sopro", "pocket"):
                assert gone(found["started"][name]), f"a {name} worker survived the exit"
        finally:
            reap(process)

    def test_none_of_them_survives_a_hard_kill(self, tmp_path):
        """SIGKILL runs no atexit hook, no signal handler and no Python at all.
        What is left is the operating system, and on Linux that is
        ``PR_SET_PDEATHSIG`` -- arranged inside each child by its own worker."""
        process, found = launch(tmp_path, "wait", kokoro=True, sopro=True, pocket=True)
        try:
            os.kill(process.pid, signal.SIGKILL)
            process.wait(timeout=30)
            for name in ("kokoro", "sopro", "pocket"):
                assert gone(found["started"][name], seconds=20.0), (
                    f"a {name} worker survived a hard kill of its parent")
        finally:
            reap(process)

    def test_t_shut_p6_switching_to_pocket_leaves_exactly_one_worker(self, tmp_path):
        """Sopro is running, Pocket is selected: the Sopro worker is stopped
        before the new choice is persisted, so the inactive engine consumes no
        RAM. An engine switch is a lifecycle boundary and does not wait for
        Pocket's drain contract (section 21.6)."""
        process, found = launch(tmp_path, "switch", sopro=True, pocket=False)
        after = {}
        try:
            after = json.loads(process.stdout.readline())
            assert gone(found["started"]["sopro"], seconds=20.0), (
                "the Sopro worker was still resident after switching to PocketTTS")
            assert alive(after["after"]), "the newly selected engine did not start"
        finally:
            reap(process)
            gone(after.get("after"), seconds=20.0)


class TestTheHardKillDuringWork:
    def test_t_shut_p4_a_hard_kill_during_inference_leaves_no_worker(self, tmp_path):
        """The test the whole containment design exists for.

        The worker is deliberately *not* waiting on stdin: it is inside a loop
        standing in for a generation call, so the closing pipe is not something
        it will look at. Parent death always killing the worker is only true
        because of the OS mechanism, and this is the only way to find out
        whether that mechanism actually fired.

        The wait for ``busy_marker`` is load-bearing rather than tidy. Without
        it the kill lands while the worker is still blocked on ``read_frame``,
        where EOF ends it perfectly well -- so the test would pass with the OS
        mechanism removed, which is the worst state a release gate can be in.
        """
        busy = tmp_path / "pocket-busy.txt"
        process, found = launch(tmp_path, "busy",
                                pocket_plan={"busy": True, "busy_marker": str(busy)})
        try:
            deadline = time.monotonic() + 20.0
            while not busy.exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            assert busy.exists(), "the worker never reached the busy loop"
            os.kill(process.pid, signal.SIGKILL)
            process.wait(timeout=30)
            assert gone(found["started"]["pocket"], seconds=20.0), (
                "a PocketTTS worker survived a hard kill of its parent while it was busy "
                "— pipe EOF is not enough and the OS mechanism did not fire")
        finally:
            reap(process)

    def test_gate_p_9_a_hard_kill_during_a_drain_leaves_no_worker(self, tmp_path):
        """The state this engine spends real time in and neither other reaches.

        A Pocket worker that has been asked to stop is a worker that is
        deliberately still computing. "The WebUI closed while an abandoned unit
        was finishing" is an ordinary Tuesday here rather than a corner case, and
        if containment did not cover it the symptom would be a PyTorch process
        left behind after every conversation that ended with a Stop.
        """
        process, found = launch(tmp_path, "draining",
                                pocket_plan={"never_finishes": True})
        try:
            os.kill(process.pid, signal.SIGKILL)
            process.wait(timeout=30)
            assert gone(found["started"]["pocket"], seconds=20.0), (
                "a PocketTTS worker survived a hard kill of its parent while it was "
                "draining an abandoned unit")
        finally:
            reap(process)

    def test_an_ordinary_exit_during_a_drain_does_not_wait_for_it(self, tmp_path):
        """Section 21.6. A shutdown that waited for quiescence would be a WebUI
        that hangs on this feature's own workaround."""
        process, found = launch(tmp_path, "draining",
                                pocket_plan={"never_finishes": True})
        try:
            began = time.monotonic()
            process.terminate()
            process.wait(timeout=30)
            assert time.monotonic() - began < 20.0
            assert gone(found["started"]["pocket"], seconds=20.0), (
                "a PocketTTS worker survived a shutdown that happened during a drain")
        finally:
            reap(process)

    def test_no_pocket_command_is_left_in_the_process_list(self, tmp_path):
        """Read from the process list rather than from a pid we kept, because
        the failure this guards against is a worker whose parent forgot it."""
        from pocket_worker import worker as protocol

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
