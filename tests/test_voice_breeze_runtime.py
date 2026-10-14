"""The Breeze TTS 2 runtime: one proven worker per card, rendered through and evicted.

Driven against a real subprocess that speaks the real framing and holds no
tensors, the way VibeVoice's runtime tests are. What is asserted here is the
half of the card contract the worker cannot assert on its own, and the half of
the render that is Breeze's alone:

    a worker that came up on the wrong card, on no provable card, on no CUDA
        device, speaking another protocol, claiming another name or a codec
        at another rate is refused and stopped, and no process is left;
    one process per card whichever way the card is spelt, and a stop that
        reaches only the card it names;
    the load names the settings' model and fast path and the render's
        attention, and nothing VibeVoice's names (no solver, no steps);
    the Voice Box's engine-neutral job is translated exactly -- every line
        with its direction, every voice with its transcript, description and
        standing direction, a voice that is only a description holding no
        samples -- and the worker reads exactly that, byte for byte;
    the takes come back cut by their sample counts, and a reply that does not
        add up is refused rather than played;
    progress frames reach the caller as the worker sent them, and the peak is
        noted with the takes and the fast path of the model the worker holds;
    a render marks the card rendering for exactly its duration, a cancel
        reaches it, an eviction during it is refused while one after it stops
        the process and hands back the figure it held;
    the resident figure is a cached read, refreshed at most every two seconds
        and never during a render, because the turns read it every hundred
        milliseconds;
    the crash-loop guard, the shutdown door, and invariant three.

Nothing here imports Torch or Breeze's code. The installer module the runtime
reaches for lazily (``mc_voice_breeze``) is stood in by a namespace through
``_installer``, so these tests do not depend on it existing.
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import os
import struct
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import pytest

import mc_voice_breeze_runtime as runtime
import mc_voice_box_render as render_types
import mc_voice_paths as paths

ROOT = Path(__file__).resolve().parent.parent

CARD = "GPU-0123abcd-4567-89ef-0123-456789abcdef"
CARD_KEY = "0123abcd456789ef0123456789abcdef"
CARD_TORCH = "0123abcd-4567-89ef-0123-456789abcdef"
OTHER = "GPU-fedcba98-7654-3210-fedc-ba9876543210"
OTHER_KEY = "fedcba9876543210fedcba9876543210"
RATE = 24000
HELD = 9_000_000_000
PEAK = 11_000_000_000

FAKE_BREEZE_WORKER = r'''#!/usr/bin/env python3
# A Breeze worker that speaks the real framing and holds no tensors.
#
# Faithful in the places the parent depends on -- the framing, the handshake
# fields, the containment it arranges for itself, one render at a time, a
# cancel read while a render is in flight, progress frames naming the line and
# the take, one entry per take and every take's samples in one payload -- and
# a stub everywhere else. Steered by MC_FAKE_BREEZE in the environment. It
# copies the framing rather than importing the real worker, as a worker in
# another interpreter would have to.
import hashlib
import json
import os
import struct
import sys
import threading
import time

PLAN = json.loads(os.environ.get("MC_FAKE_BREEZE", "{}"))
CARD = os.environ.get("CUDA_VISIBLE_DEVICES", "")
LENGTH = struct.Struct(">I")
LOCK = threading.Lock()
STATE = {"loaded": False, "job": "", "cancel": None, "resident": 0, "peak": 0,
         "model_dir": "", "attention": "", "fast_path": False, "loads": 0}


def read_frame(stream):
    head = stream.read(4)
    if len(head) < 4:
        return None
    (size,) = LENGTH.unpack(head)
    header = json.loads(stream.read(size).decode("utf-8"))
    (size,) = LENGTH.unpack(stream.read(4))
    return header, (stream.read(size) if size else b"")


def write_frame(stream, header, payload=b""):
    raw = json.dumps(header).encode("utf-8")
    with LOCK:
        stream.write(LENGTH.pack(len(raw)))
        stream.write(raw)
        stream.write(LENGTH.pack(len(payload)))
        if payload:
            stream.write(payload)
        stream.flush()


def containment(parent_pid):
    """The same arrangement the real worker makes, in the same place."""
    if sys.platform.startswith("linux"):
        try:
            import ctypes
            import signal

            libc = ctypes.CDLL("libc.so.6", use_errno=True)
            if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
                return "none"
        except Exception:
            return "none"
        if parent_pid and os.getppid() != parent_pid:
            raise SystemExit(0)
        return "pdeathsig"
    if os.name == "nt":
        return "job"
    return "none"


def takes(header, cancelled):
    """Each take's 32-bit float samples, and what the reply says about them.

    Take k lasts (k + 1) quarter-seconds and every sample of it is (k + 1) / 8,
    so a parent that cut the payload anywhere but at the counts would be seen
    to. A cancelled render made a little of its first take and none of the
    rest, and still answers one entry per take, as the real worker does.
    """
    count = max(1, int(header.get("takes") or 1))
    seed = header.get("seed")
    entries, chunks = [], []
    for index in range(count):
        if cancelled:
            samples = 2400 if index == 0 else 0
        else:
            samples = (index + 1) * 6000
        chunks.append(struct.pack("<f", (index + 1) / 8.0) * samples)
        entries.append({"seed": None if seed is None else int(seed) + index,
                        "samples": samples, "seconds": samples / 24000.0,
                        "tokens": 10 * (index + 1),
                        "capped": index == PLAN.get("capped_take", -1)})
    answered = PLAN.get("takes_answered")
    if answered is not None:
        entries, chunks = entries[:int(answered)], chunks[:int(answered)]
    body = b"".join(chunks)
    if PLAN.get("short_payload"):
        body = body[:-4]
    return entries, body


def mark(name, line):
    path = PLAN.get(name)
    if path:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def main():
    mark("alive_marker", str(os.getpid()))
    sys.stderr.write("Breeze worker: the fake worker is up\n")
    sys.stderr.flush()
    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    while True:
        frame = read_frame(stdin)
        if frame is None:
            return 0
        header, payload = frame
        operation = header.get("op")
        rid = header.get("id")
        if operation == "shutdown":
            write_frame(stdout, {"id": rid, "ok": True})
            return 0
        if operation == "init":
            mark("init_log", json.dumps(header))
            if PLAN.get("no_cuda"):
                write_frame(stdout, {"id": rid, "ok": False, "refusal": True,
                                     "error": "no CUDA device is visible to the worker"})
                continue
            if PLAN.get("init_error"):
                write_frame(stdout, {"id": rid, "ok": False, "refusal": False,
                                     "error": PLAN["init_error"]})
                continue
            found = containment(int(header.get("parent_pid") or 0))
            reply = {"id": rid, "ok": True,
                     "protocol": PLAN.get("protocol", 1),
                     "worker": PLAN.get("worker", "breeze"),
                     "python": "3.12.10", "torch": "2.9.1+cu128",
                     "cuda": PLAN.get("cuda", True),
                     "device_name": PLAN.get("device_name", "Fake GPU 32GB"),
                     "device_uuid": PLAN.get("device_uuid", CARD),
                     "device_index": 0,
                     "total_vram_bytes": 32 << 30, "free_vram_bytes": 31 << 30,
                     "containment": PLAN.get("containment", found),
                     "breeze": "89abcdef0123456789abcdef0123456789abcdef",
                     "transformers": "4.57.3", "qwen_tts": "0.0.4",
                     "sample_rate": PLAN.get("sample_rate", 24000),
                     "attention": ["flash_attention_2", "sdpa", "eager"],
                     "compile": PLAN.get("compile", False), "max_takes": 4}
            write_frame(stdout, reply)
            continue
        if operation == "status":
            mark("status_marker", "status")
            write_frame(stdout, {"id": rid, "ok": True, "loaded": STATE["loaded"],
                                 "rendering": bool(STATE["job"]), "job": STATE["job"],
                                 "resident_bytes": STATE["resident"],
                                 "peak_bytes": STATE["peak"],
                                 "model_dir": STATE["model_dir"],
                                 "attention": STATE["attention"],
                                 "fast_path": STATE["fast_path"]})
            continue
        if operation == "load":
            mark("load_log", json.dumps(header))
            fails = PLAN.get("load_fails")
            if fails:
                write_frame(stdout, {"id": rid, "ok": False, "error": fails,
                                     "refusal": " " in fails})
                continue
            STATE["loaded"] = True
            STATE["resident"] = int(PLAN.get("resident_bytes", 9000000000))
            STATE["model_dir"] = header.get("model_dir", "")
            STATE["attention"] = header.get("attention", "")
            STATE["fast_path"] = header.get("fast_path") is True
            STATE["loads"] += 1
            write_frame(stdout, {"id": rid, "ok": True, "resident_bytes": STATE["resident"],
                                 "weights_bytes": 8500000000, "load_seconds": 0.5,
                                 "already_loaded": STATE["loads"] > 1,
                                 "attention": STATE["attention"],
                                 "fast_path": STATE["fast_path"],
                                 "compiled": PLAN.get("compile", False)})
            continue
        if operation == "unload":
            STATE["loaded"] = False
            STATE["resident"] = int(PLAN.get("empty_bytes", 400000000))
            STATE["model_dir"] = ""
            STATE["attention"] = ""
            STATE["fast_path"] = False
            write_frame(stdout, {"id": rid, "ok": True, "resident_bytes": STATE["resident"]})
            continue
        if operation == "cancel":
            mark("cancel_log", header.get("job") or "-")
            event = STATE["cancel"]
            wanted = header.get("job") or ""
            ok = event is not None and (not wanted or wanted == STATE["job"])
            if ok:
                event.set()
            write_frame(stdout, {"id": rid, "ok": True, "cancelled": bool(ok)})
            continue
        if operation == "render":
            mark("render_requests", header.get("job") or "-")
            if PLAN.get("die_on_render"):
                os._exit(3)
            if STATE["job"]:
                write_frame(stdout, {"id": rid, "ok": False, "refusal": True,
                                     "error": "one render at a time"})
                continue
            if not STATE["loaded"]:
                write_frame(stdout, {"id": rid, "ok": False, "refusal": True,
                                     "error": "no model is loaded"})
                continue
            mark("render_log", json.dumps({
                "header": header, "payload_bytes": len(payload),
                "payload_sha256": hashlib.sha256(payload).hexdigest()}))
            STATE["job"] = header.get("job") or "-"
            event = threading.Event()
            STATE["cancel"] = event

            def run(rid=rid, event=event, head=header):
                began = time.monotonic()
                count = max(1, int(head.get("takes") or 1))
                lines = len(head.get("lines") or ())
                # A frame at the start of every line of every take, as the
                # real worker sends one, each written down as it is sent.
                for take in range(count):
                    for line in range(lines):
                        progress = {"seconds": round(0.75 * line, 2), "line": line + 1,
                                    "lines": lines, "take": take + 1, "takes": count}
                        mark("progress_log", json.dumps(progress))
                        write_frame(stdout, {"id": rid, "progress": progress})
                seconds = float(PLAN.get("render_seconds", 0.05))
                while time.monotonic() - began < seconds:
                    if event.is_set():
                        break
                    time.sleep(0.02)
                if PLAN.get("never_answers_render"):
                    while True:
                        time.sleep(0.05)
                STATE["peak"] = int(PLAN.get("peak_bytes", 11000000000))
                entries, body = takes(head, event.is_set())
                reply = {"id": rid, "ok": True, "format": PLAN.get("format", "f32le"),
                         "sample_rate": 24000, "takes": entries,
                         "seconds": entries[0]["seconds"] if entries else 0.0,
                         "render_seconds": time.monotonic() - began,
                         "peak_bytes": STATE["peak"], "resident_bytes": STATE["resident"],
                         "cancelled": event.is_set(),
                         "tokens": sum(entry["tokens"] for entry in entries),
                         "capped": any(entry["capped"] for entry in entries)}
                STATE["job"] = ""
                STATE["cancel"] = None
                write_frame(stdout, reply, body)

            threading.Thread(target=run, daemon=True).start()
            continue
        write_frame(stdout, {"id": rid, "ok": False, "refusal": True,
                             "error": "unknown operation"})


if __name__ == "__main__":
    raise SystemExit(main())
'''


def wait_until(condition, timeout=5.0, what="the condition"):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    raise AssertionError(f"{what} never became true")


def pcm(seconds: float, value: float = 0.25) -> bytes:
    return struct.pack("<f", value) * int(seconds * RATE)


def take_pcm(index: int, samples: int) -> bytes:
    """What the fake worker makes of take ``index``: every sample ``(index + 1) / 8``."""
    return struct.pack("<f", (index + 1) / 8.0) * samples


def job_for(identifier="job-1", **values):
    """A two-speaker section cloned from two samples, as the Voice Box builds one."""
    found = dict(id=identifier,
                 lines=[runtime.Line(1, "Hello there."), runtime.Line(2, "Hi.")],
                 voices={1: runtime.Voice(pcm=pcm(0.5), rate=RATE,
                                          transcript="This is how I sound."),
                         2: runtime.Voice(pcm=pcm(0.25), rate=RATE,
                                          transcript="And this is how I do.")},
                 seed=7, takes=1, options={})
    found.update(values)
    return runtime.RenderJob(**found)


def described_job(**values):
    """Two speakers: one designed from a description and one cloned from a sample.

    Handed in speaker order 2, 1 so that the frame's ascending order is the
    frame's own doing; the sample is three bytes past a whole sample so that
    the cut is too; and directions both on lines and on a slot.
    """
    found = dict(
        id="job-1",
        lines=[runtime.Line(2, "Hello there.", "whispering"),
               runtime.Line(1, "Hi.", ""),
               runtime.Line(2, "Goodbye.")],
        voices={2: runtime.Voice(pcm=pcm(0.5) + b"\x01\x02\x03", rate=RATE, sample_id="a" * 16,
                                 title="Ada", transcript="This is how I sound."),
                1: runtime.Voice(description="A low, unhurried voice.",
                                 instruction="Speak slowly.")},
        seed=41, takes=2,
        options={"cfg_scale": 5.5, "attention": "sdpa", "sampling": True, "temperature": 0.7,
                 "top_p": 0.8, "max_new_tokens": 600, "line_gap_ms": 0})
    found.update(values)
    return runtime.RenderJob(**found)


class Stub:
    """The stood-in installer module and the fake worker's plan, per test."""

    def __init__(self, root: Path, script: Path, models: Path, plan: dict, settings: dict,
                 peaks: list, refusal: list, module):
        self.root = root
        self.script = script
        self.models = models
        self.plan = plan
        self.settings = settings
        self.peaks = peaks
        self.refusal = refusal
        self.module = module

    def lines(self, name: str) -> list:
        path = Path(self.plan[name])
        if not path.exists():
            return []
        return [line for line in path.read_text(encoding="utf-8").splitlines() if line]

    def pids(self) -> list:
        return [int(line) for line in self.lines("alive_marker")]

    def status_ops(self) -> int:
        return len(self.lines("status_marker"))

    def renders(self) -> list:
        return [json.loads(line) for line in self.lines("render_log")]

    def loads(self) -> list:
        return [json.loads(line) for line in self.lines("load_log")]

    def inits(self) -> list:
        return [json.loads(line) for line in self.lines("init_log")]

    def progress(self) -> list:
        return [json.loads(line) for line in self.lines("progress_log")]


@pytest.fixture
def stub(tmp_path, monkeypatch, voice_root):
    script = tmp_path / "fake_breeze_worker.py"
    script.write_text(FAKE_BREEZE_WORKER, encoding="utf-8")
    models = tmp_path / "models"
    (models / "breeze-tts-2").mkdir(parents=True)
    plan = {name: str(tmp_path / f"{name}.txt")
            for name in ("alive_marker", "status_marker", "render_log", "render_requests",
                         "load_log", "init_log", "progress_log", "cancel_log")}
    settings = {"model_id": "breeze-tts-2", "fast_path": False}
    peaks = []
    refusal = [""]
    module = types.SimpleNamespace(
        LABEL="Breeze TTS 2",
        refusal=lambda: refusal[0],
        runtime_python=lambda: Path(sys.executable),
        worker_script=lambda: script,
        worker_environment=lambda card: {"MC_FAKE_BREEZE": json.dumps(plan),
                                         "CUDA_VISIBLE_DEVICES": str(card)},
        settings=lambda: dict(settings),
        model_dir=lambda identifier: models / identifier,
        note_peak=lambda identifier, peak_bytes, takes=1, fast_path=False:
        peaks.append((identifier, int(peak_bytes), int(takes), bool(fast_path))))
    monkeypatch.setattr(runtime, "_installer", lambda: module)
    monkeypatch.setattr(runtime, "STOP_GRACE", 3.0)
    monkeypatch.setattr(runtime, "TERMINATE_GRACE", 1.5)
    found = Stub(tmp_path, script, models, plan, settings, peaks, refusal, module)
    yield found
    runtime.stop("", "test finished")
    with runtime._table_lock:
        runtime._failures.clear()
        runtime._peaks.clear()
        runtime._last_error = ""


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def render_in_background(stub, job=None, options=None):
    """A render that lasts until it is cancelled, on its own thread.

    Long on purpose: a test that asserts something *during* a render must not
    find the render already over because its own thread was slow to look.
    """
    stub.plan["render_seconds"] = 30.0
    runtime.load(CARD, options)
    results, errors = [], []

    def run():
        try:
            results.append(runtime.render(CARD, job or job_for()))
        except Exception as exc:  # noqa: BLE001 - the test reads it
            errors.append(exc)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    wait_until(lambda: runtime.rendering(CARD), what="the card rendering")
    return thread, results, errors


def finish(thread, job_id="job-1"):
    runtime.cancel(CARD, job_id)
    thread.join(timeout=10.0)
    assert not thread.is_alive(), "the render never ended"


# --------------------------------------------------------------------------- #
# Card keys
# --------------------------------------------------------------------------- #


class TestCardKeysAreTheHexDigitsBothSpellingsShare:
    def test_nvidia_smi_and_torch_spell_one_card(self):
        assert runtime.card_key(CARD) == CARD_KEY
        assert runtime.card_key(CARD_TORCH) == CARD_KEY
        assert runtime.card_key(CARD_KEY) == CARD_KEY
        assert runtime.card_key(CARD.upper()) == CARD_KEY

    def test_nothing_is_the_empty_key(self):
        assert runtime.card_key("") == ""
        assert runtime.card_key(None) == ""
        assert runtime.card_key("GPU-") == ""


# --------------------------------------------------------------------------- #
# The handshake
# --------------------------------------------------------------------------- #


REFUSALS = [
    pytest.param({"protocol": 99}, ("protocol 99", "Reinstall Breeze in the Voice Box tab"),
                 id="protocol"),
    pytest.param({"worker": "vibevoice"}, ("did not identify itself as Breeze",),
                 id="worker-name"),
    pytest.param({"cuda": False}, ("found no CUDA device",), id="no-cuda"),
    pytest.param({"device_uuid": ""}, ("cannot prove which card",), id="no-uuid"),
    pytest.param({"device_uuid": OTHER}, ("not the card asked for", OTHER), id="other-card"),
    pytest.param({"sample_rate": 22050}, ("22050", "24000"), id="sample-rate"),
    pytest.param({"containment": "none"}, ("could not confirm",), id="containment",
                 marks=pytest.mark.skipif(os.name == "nt",
                                          reason="the parent arranges the job on Windows")),
]


class TestTheHandshakeRefusesWhatThisBuildCannotAccept:
    def test_a_good_worker_starts_and_reports_what_it_is(self, stub):
        found = runtime.ensure_started(CARD)
        assert isinstance(found, runtime.Handshake)
        assert found.cuda is True and found.device_uuid == CARD
        assert found.device_name == "Fake GPU 32GB"
        assert found.protocol == 1 == runtime.protocol.PROTOCOL_VERSION
        assert found.worker == "breeze" == runtime.protocol.WORKER_NAME
        assert found.breeze == "89abcdef0123456789abcdef0123456789abcdef"
        assert (found.torch, found.transformers, found.qwen_tts) == \
            ("2.9.1+cu128", "4.57.3", "0.0.4")
        assert found.attention == ["flash_attention_2", "sdpa", "eager"]
        assert found.compile is False and found.sample_rate == 24000
        card = runtime.status()["cards"][CARD_KEY]
        assert card["running"] is True and card["loaded"] is False
        assert card["device_name"] == "Fake GPU 32GB" and card["uuid"] == CARD
        assert len(stub.pids()) == 1
        init = stub.inits()[0]
        assert init["expect_uuid"] == CARD_KEY
        assert init["parent_pid"] == os.getpid()
        names = {thread.name for thread in threading.enumerate()}
        assert {f"mc-breeze-reader-{CARD_KEY[:8]}", f"mc-breeze-stderr-{CARD_KEY[:8]}"} <= names

    def test_the_worker_may_spell_the_uuid_the_way_torch_does(self, stub):
        stub.plan["device_uuid"] = CARD_TORCH
        found = runtime.ensure_started(CARD)
        assert found.device_uuid == CARD_TORCH

    @pytest.mark.parametrize("answer, words", REFUSALS)
    def test_each_refusal_is_its_own_sentence_and_leaves_no_process(self, stub, answer, words):
        stub.plan.update(answer)
        with pytest.raises(runtime.BreezeRuntimeError) as raised:
            runtime.ensure_started(CARD)
        for word in words:
            assert word in str(raised.value)
        assert len(stub.pids()) == 1, "the worker was started, so the refusal is its answer"
        assert not alive(stub.pids()[0]), "a refused worker is stopped, not dropped"
        assert runtime.status()["cards"] == {}
        assert runtime.status()["last_error"] == str(raised.value)
        assert runtime.resident_bytes(CARD) == 0 and runtime.rendering(CARD) is False

    def test_a_worker_that_saw_no_cuda_device_answers_with_the_sentence(self, stub):
        stub.plan["no_cuda"] = True
        with pytest.raises(runtime.BreezeRuntimeError) as raised:
            runtime.ensure_started(CARD)
        assert str(raised.value) == "no CUDA device is visible to the worker"
        assert not alive(stub.pids()[0])

    def test_a_runtime_missing_a_part_says_to_reinstall_rather_than_a_class_name(self, stub):
        """Breeze's code sits in a folder beside the runtime, not in a wheel, so
        an import that fails is a runtime missing a part."""
        stub.plan["init_error"] = "ModuleNotFoundError"
        with pytest.raises(runtime.BreezeRuntimeError) as raised:
            runtime.ensure_started(CARD)
        assert str(raised.value) == ("Part of Breeze TTS 2's runtime is missing. Reinstall it "
                                     "in the Voice Box tab.")
        assert not alive(stub.pids()[0])
        assert runtime.status()["cards"] == {}

    def test_no_runtime_installed_starts_nothing(self, stub):
        stub.module.runtime_python = lambda: None
        with pytest.raises(runtime.BreezeRuntimeError) as raised:
            runtime.ensure_started(CARD)
        assert "not installed" in str(raised.value)
        assert stub.pids() == []

    def test_the_installers_refusal_is_the_answer_and_starts_nothing(self, stub):
        stub.refusal[0] = "Breeze TTS 2's model is not installed yet."
        with pytest.raises(runtime.BreezeRuntimeError) as raised:
            runtime.ensure_started(CARD)
        assert str(raised.value) == "Breeze TTS 2's model is not installed yet."
        assert stub.pids() == []

    def test_no_card_named_is_refused_before_anything_is_looked_at(self, stub):
        with pytest.raises(runtime.BreezeRuntimeError):
            runtime.ensure_started("")
        assert stub.pids() == []

    def test_the_worker_is_given_exactly_one_card_and_no_credential(self, stub, monkeypatch):
        for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
            monkeypatch.setenv(name, "secret")
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
        seen = {}
        real = runtime.subprocess.Popen

        def watching(command, **kwargs):
            seen["command"] = list(command)
            seen["env"] = dict(kwargs["env"])
            seen["cwd"] = kwargs.get("cwd")
            return real(command, **kwargs)

        monkeypatch.setattr(runtime.subprocess, "Popen", watching)
        runtime.ensure_started(CARD)
        assert seen["env"]["CUDA_VISIBLE_DEVICES"] == CARD, "the worker's own environment wins"
        for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
            assert name not in seen["env"]
        assert seen["command"][1] == str(stub.script)
        assert seen["command"][2:] == [runtime.protocol.MARKER, "--parent-pid",
                                       str(os.getpid())]
        assert seen["cwd"] == str(paths.extension_root())

    def test_without_an_answer_from_the_installer_the_worker_beside_this_module_runs(self):
        assert runtime._worker_script(types.SimpleNamespace()) == \
            ROOT / "breeze_worker" / "worker.py"

        def broken():
            raise OSError("no such folder")

        assert runtime._worker_script(types.SimpleNamespace(worker_script=broken)) == \
            ROOT / "breeze_worker" / "worker.py"

    def test_the_workers_own_diagnostics_reach_the_log(self, stub, caplog):
        caplog.set_level(logging.INFO, logger="model_chain")
        runtime.ensure_started(CARD)
        wait_until(lambda: any("Breeze worker: the fake worker is up" in record.getMessage()
                               for record in caplog.records),
                   what="the worker's stderr line in the log")


class TestStatusIsAPureRead:
    def test_status_before_any_start_is_empty_and_asks_nothing(self, stub):
        assert runtime.status() == {"cards": {}, "last_error": ""}
        assert runtime.resident_bytes(CARD) == 0
        assert runtime.rendering(CARD) is False
        assert runtime.peak_observed(CARD) == 0
        assert stub.pids() == []

    def test_status_never_sends_a_request(self, stub):
        runtime.load(CARD)
        before = stub.status_ops()
        for _turn in range(20):
            runtime.status()
            runtime.status(CARD)
            runtime.rendering(CARD)
        assert stub.status_ops() == before

    def test_status_for_one_card_shows_that_card_only(self, stub):
        runtime.ensure_started(CARD)
        runtime.ensure_started(OTHER)
        assert set(runtime.status(CARD)["cards"]) == {CARD_KEY}
        assert set(runtime.status()["cards"]) == {CARD_KEY, OTHER_KEY}


# --------------------------------------------------------------------------- #
# One process per card
# --------------------------------------------------------------------------- #


class TestOneProcessPerCard:
    def test_every_spelling_of_a_card_is_one_process(self, stub):
        first = runtime.ensure_started(CARD)
        second = runtime.ensure_started(CARD_KEY)
        third = runtime.ensure_started(CARD_TORCH)
        assert first is second is third
        assert len(stub.pids()) == 1
        assert set(runtime.status()["cards"]) == {CARD_KEY}
        runtime.load(CARD_KEY)
        assert runtime.resident_bytes(CARD) == HELD, "loaded by one spelling, read by another"

    def test_two_cards_are_two_processes_and_a_stop_names_its_card(self, stub):
        runtime.ensure_started(CARD)
        runtime.ensure_started(OTHER)
        pids = stub.pids()
        assert len(pids) == 2 and pids[0] != pids[1]
        runtime.stop(OTHER, "test")
        wait_until(lambda: not alive(pids[1]), what="the other card's worker exiting")
        assert alive(pids[0])
        assert set(runtime.status()["cards"]) == {CARD_KEY}
        runtime.stop("", "test")
        wait_until(lambda: not alive(pids[0]), what="the first card's worker exiting")
        assert runtime.status()["cards"] == {}

    def test_each_card_holds_its_own_figures(self, stub):
        runtime.ensure_started(CARD)
        runtime.ensure_started(OTHER)
        runtime.load(CARD)
        assert runtime.resident_bytes(CARD) == HELD
        assert runtime.resident_bytes(OTHER) == 0
        assert runtime.status()["cards"][CARD_KEY]["loaded"] is True
        assert runtime.status()["cards"][OTHER_KEY]["loaded"] is False


# --------------------------------------------------------------------------- #
# The render frame: the Voice Box's job as Breeze's worker reads it
# --------------------------------------------------------------------------- #


class TestTheRenderFrame:
    def test_the_engine_neutral_types_are_the_voice_boxs_own(self):
        for name in ("RenderJob", "Take", "RenderResult", "Voice", "Line"):
            assert getattr(runtime, name) is getattr(render_types, name), name

    def test_a_described_voice_and_directions_are_translated_exactly(self):
        header, payload = runtime._render_frame(described_job())
        assert header == {
            "op": "render",
            "job": "job-1",
            "lines": [{"speaker": 2, "text": "Hello there.", "instruction": "whispering"},
                      {"speaker": 1, "text": "Hi.", "instruction": ""},
                      {"speaker": 2, "text": "Goodbye.", "instruction": ""}],
            "voices": [{"speaker": 1, "offset": 0, "count": 0, "rate": RATE,
                        "transcript": "", "description": "A low, unhurried voice.",
                        "instruction": "Speak slowly."},
                       {"speaker": 2, "offset": 0, "count": 12000, "rate": RATE,
                        "transcript": "This is how I sound.", "description": "",
                        "instruction": ""}],
            "cfg_scale": 5.5,
            "attention": "sdpa",
            "sampling": True,
            "temperature": 0.7,
            "top_p": 0.8,
            "max_new_tokens": 600,
            "line_gap_ms": 0,
            "seed": 41,
            "takes": 2,
        }
        assert payload == pcm(0.5), "the sample alone, cut to whole samples"

    def test_a_described_voice_between_two_samples_sits_at_the_running_offset(self):
        job = runtime.RenderJob(
            id="job-3", lines=[(1, "One."), (2, "Two."), (3, "Three.")],
            voices={3: runtime.Voice(pcm=pcm(0.125, 0.5), transcript="Third."),
                    1: runtime.Voice(pcm=pcm(0.25), transcript="First."),
                    2: runtime.Voice(description="A bright voice.")})
        header, payload = runtime._render_frame(job)
        assert [(voice["speaker"], voice["offset"], voice["count"])
                for voice in header["voices"]] == [(1, 0, 6000), (2, 6000, 0), (3, 6000, 3000)]
        assert payload == pcm(0.25) + pcm(0.125, 0.5)

    def test_a_job_of_pairs_is_translated_with_breezes_own_defaults(self):
        job = runtime.RenderJob(id="job-2", lines=[(1, "Hello there."), (2, "Hi.")],
                                voices={2: (pcm(0.25), RATE), 1: (pcm(0.5), RATE)})
        header, payload = runtime._render_frame(job)
        assert header == {
            "op": "render",
            "job": "job-2",
            "lines": [{"speaker": 1, "text": "Hello there.", "instruction": ""},
                      {"speaker": 2, "text": "Hi.", "instruction": ""}],
            "voices": [{"speaker": 1, "offset": 0, "count": 12000, "rate": RATE,
                        "transcript": "", "description": "", "instruction": ""},
                       {"speaker": 2, "offset": 12000, "count": 6000, "rate": RATE,
                        "transcript": "", "description": "", "instruction": ""}],
            "cfg_scale": 4.0,
            "attention": "flash_attention_2",
            "sampling": True,
            "temperature": None,
            "top_p": None,
            "max_new_tokens": None,
            "line_gap_ms": 150,
            "seed": None,
            "takes": 1,
        }
        assert payload == pcm(0.5) + pcm(0.25)
        assert header["cfg_scale"] == runtime.protocol.CFG_DEFAULT
        assert header["line_gap_ms"] == runtime.protocol.GAP_DEFAULT_MS
        assert header["attention"] == runtime.protocol.ATTENTION_DEFAULT

    def test_temperature_and_top_p_go_only_while_sampling(self):
        header, _ = runtime._render_frame(described_job(
            options={"sampling": False, "temperature": 0.7, "top_p": 0.8}))
        assert (header["sampling"], header["temperature"], header["top_p"]) == \
            (False, None, None)
        header, _ = runtime._render_frame(described_job(
            options={"temperature": 0.7, "top_p": 0.8}))
        assert (header["sampling"], header["temperature"], header["top_p"]) == \
            (True, 0.7, 0.8), "Breeze samples unless the options turn it off"

    def test_an_option_left_empty_is_the_workers_default(self):
        header, _ = runtime._render_frame(described_job(
            takes=0, options={"cfg_scale": None, "attention": "", "sampling": None,
                              "max_new_tokens": 0, "line_gap_ms": None}))
        assert (header["cfg_scale"], header["attention"], header["sampling"]) == \
            (4.0, "flash_attention_2", True)
        assert (header["max_new_tokens"], header["line_gap_ms"], header["takes"]) == \
            (None, 150, 1)

    def test_breezes_own_worker_reads_the_frame_as_the_voice_box_meant_it(self):
        """The fake worker proves the pipe; this proves the words. The frame goes
        through the real worker's own parser, which refuses whatever it does
        not understand -- a key spelt otherwise, a voice with neither a sample
        nor a description, a sample at the wrong rate."""
        pytest.importorskip("numpy")
        header, payload = runtime._render_frame(described_job())
        request = runtime.protocol.RenderRequest.parse(header, payload)
        assert request.job == "job-1"
        assert request.lines == [(2, "Hello there.", "whispering"), (1, "Hi.", ""),
                                 (2, "Goodbye.", "")]
        designed, cloned = request.voices[1], request.voices[2]
        assert designed.samples is None
        assert (designed.description, designed.instruction) == \
            ("A low, unhurried voice.", "Speak slowly.")
        assert cloned.transcript == "This is how I sound."
        assert cloned.samples.tolist() == [0.25] * 12000
        assert (request.seed, request.takes, request.cfg_scale, request.attention) == \
            (41, 2, 5.5, "sdpa")
        assert (request.sampling, request.temperature, request.top_p) == (True, 0.7, 0.8)
        assert (request.max_new_tokens, request.gap_ms) == (600, 0)

        request = runtime.protocol.RenderRequest.parse(*runtime._render_frame(job_for(seed=None)))
        assert (request.cfg_scale, request.attention, request.sampling) == \
            (4.0, "flash_attention_2", True)
        assert (request.temperature, request.top_p, request.max_new_tokens, request.gap_ms) == \
            (0.9, 1.0, None, 150), "what the job left unsaid is the worker's own default"

    def test_nothing_of_vibevoices_vocabulary_is_sent(self):
        header, _ = runtime._render_frame(described_job(
            options={"steps": 12, "solver": "dpmpp_2m", "cfg_scale": 2.0}))
        assert not {"steps", "solver", "script", "dtype"} & set(header)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


class TestLoading:
    def test_load_starts_the_worker_and_asks_for_the_settings_model(self, stub):
        found = runtime.load(CARD)
        assert found == {"loaded": True, "resident_bytes": HELD,
                         "weights_bytes": 8_500_000_000, "load_seconds": 0.5,
                         "attention": "flash_attention_2", "fast_path": False}
        header = stub.loads()[0]
        assert set(header) == {"op", "id", "model_dir", "attention", "fast_path"}, \
            "no step count, no precision, no solver"
        assert header["model_dir"] == str(stub.models / "breeze-tts-2")
        assert header["attention"] == runtime.protocol.ATTENTION_DEFAULT
        assert header["fast_path"] is False
        assert runtime.status()["cards"][CARD_KEY]["loaded"] is True
        assert runtime.resident_bytes(CARD) == HELD, "from the load reply"

    def test_load_names_the_renders_attention_and_the_settings_fast_path(self, stub):
        stub.settings["fast_path"] = True
        found = runtime.load(CARD, {"attention": "sdpa", "cfg_scale": 5.0})
        header = stub.loads()[0]
        assert (header["attention"], header["fast_path"]) == ("sdpa", True)
        assert (found["attention"], found["fast_path"]) == ("sdpa", True)
        runtime.load(CARD, {"attention": ""})
        assert stub.loads()[1]["attention"] == "flash_attention_2", \
            "an empty choice is the default"

    def test_the_load_is_logged_once_and_not_for_a_model_already_held(self, stub, caplog):
        caplog.set_level(logging.INFO, logger="model_chain")
        stub.settings["fast_path"] = True
        runtime.load(CARD, {"attention": "eager"})
        runtime.load(CARD, {"attention": "eager"})
        found = [record.getMessage() for record in caplog.records
                 if "loaded its model" in record.getMessage()]
        assert len(found) == 1
        assert "Eager" in found[0] and "eager" not in found[0]
        assert ", fast path (CUDA graphs only)" in found[0], "the fake runtime has no compiler"
        assert str(stub.models) not in found[0]

    def test_load_is_refused_while_rendering(self, stub):
        thread, _results, _errors = render_in_background(stub)
        loads = len(stub.loads())
        with pytest.raises(runtime.BreezeRuntimeError) as raised:
            runtime.load(CARD, {"attention": "eager"})
        assert "rendering" in str(raised.value)
        assert len(stub.loads()) == loads, "the load never reached the worker"
        finish(thread)

    def test_the_workers_refusal_is_the_sentence_the_user_sees(self, stub):
        stub.plan["load_fails"] = "the model directory does not exist"
        with pytest.raises(runtime.BreezeRuntimeError) as raised:
            runtime.load(CARD)
        assert str(raised.value) == "the model directory does not exist"
        assert runtime.status()["cards"][CARD_KEY]["running"] is True, "a refusal is not a crash"

    def test_a_library_failure_becomes_a_sentence_rather_than_a_class_name(self, stub):
        stub.plan["load_fails"] = "OutOfMemoryError"
        with pytest.raises(runtime.BreezeRuntimeError) as raised:
            runtime.load(CARD)
        assert "ran out of memory" in str(raised.value)
        assert "OutOfMemoryError" not in str(raised.value)


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


class TestRendering:
    def test_the_worker_reads_the_translated_job_byte_for_byte(self, stub):
        runtime.load(CARD, {"attention": "sdpa"})
        job = described_job()
        header, payload = runtime._render_frame(job)
        runtime.render(CARD, job)
        sent = stub.renders()[0]
        received = dict(sent["header"])
        assert received.pop("id")
        assert received == header
        assert sent["payload_bytes"] == len(payload) == 4 * 12000
        assert sent["payload_sha256"] == hashlib.sha256(payload).hexdigest()

    def test_the_takes_are_cut_from_the_payload_by_their_sample_counts(self, stub):
        runtime.load(CARD)
        result = runtime.render(CARD, job_for(seed=9990, takes=4))
        assert isinstance(result, runtime.RenderResult)
        assert [take.seed for take in result.takes] == [9990, 9991, 9992, 9993]
        for index, take in enumerate(result.takes):
            samples = (index + 1) * 6000
            assert take.pcm == take_pcm(index, samples), f"take {index + 1}"
            assert take.seconds == samples / RATE
            assert take.tokens == 10 * (index + 1)
            assert take.capped is False
        assert result.seconds == 0.25 and result.tokens == 100
        assert result.sample_rate == RATE and result.cancelled is False
        assert result.peak_bytes == PEAK and result.capped is False
        assert result.render_seconds >= 0.0

    def test_a_reply_whose_takes_do_not_add_up_is_refused_not_played(self, stub):
        stub.plan["short_payload"] = True
        runtime.load(CARD)
        with pytest.raises(runtime.BreezeRuntimeError) as raised:
            runtime.render(CARD, job_for(seed=1, takes=2))
        assert "did not add up" in str(raised.value)
        assert runtime.rendering(CARD) is False

    def test_a_reply_with_fewer_takes_than_were_asked_for_is_refused(self, stub):
        stub.plan["takes_answered"] = 1
        runtime.load(CARD)
        with pytest.raises(runtime.BreezeRuntimeError) as raised:
            runtime.render(CARD, job_for(seed=1, takes=2))
        assert "did not add up" in str(raised.value)

    def test_a_reply_in_another_format_is_refused_with_a_sentence(self, stub):
        stub.plan["format"] = "pcm16-wav"
        runtime.load(CARD)
        with pytest.raises(runtime.BreezeRuntimeError) as raised:
            runtime.render(CARD, job_for(seed=1))
        assert "format" in str(raised.value)

    def test_the_reply_is_read_with_the_ceiling_for_replies(self):
        """Four long takes of 32-bit float are more than a request may carry."""
        source = Path(runtime.__file__).read_text(encoding="utf-8")
        assert "protocol.read_frame(stream, protocol.MAX_REPLY_PAYLOAD)" in source

    def test_progress_frames_reach_the_caller_as_the_worker_sent_them(self, stub):
        runtime.load(CARD)
        progress = []
        runtime.render(CARD, described_job(), progress.append)
        sent = stub.progress()
        assert len(sent) == 2 * 3, "a frame per line of each take"
        assert progress == sent
        assert progress[-1] == {"seconds": 1.5, "line": 3, "lines": 3, "take": 2, "takes": 2}

    def test_a_progress_callback_that_raises_does_not_end_the_render(self, stub):
        runtime.load(CARD)

        def broken(progress):
            raise RuntimeError("the page went away")

        result = runtime.render(CARD, job_for(), broken)
        assert result.cancelled is False and result.seconds == 0.25

    def test_the_peak_is_noted_with_the_takes_and_the_fast_path(self, stub):
        stub.settings["fast_path"] = True
        runtime.load(CARD)
        assert runtime.peak_observed(CARD) == 0
        runtime.render(CARD, job_for())
        assert stub.peaks == [("breeze-tts-2", PEAK, 1, True)]
        runtime.render(CARD, job_for("job-2", seed=1, takes=3))
        assert stub.peaks[-1] == ("breeze-tts-2", PEAK, 3, True), \
            "a batch's peak is noted with how many takes made it"
        assert runtime.peak_observed(CARD) == PEAK
        assert runtime.status()["cards"][CARD_KEY]["peak_bytes"] == PEAK

    def test_the_peak_belongs_to_the_model_the_worker_loaded(self, stub):
        """The settings may move on while a section is rendering; the model the
        worker holds is the one the last load named."""
        runtime.load(CARD)
        stub.settings.update({"model_id": "another-model", "fast_path": True})
        runtime.render(CARD, job_for())
        assert stub.peaks == [("breeze-tts-2", PEAK, 1, False)]

    def test_an_installer_that_cannot_note_the_peak_does_not_fail_the_render(self, stub):
        def broken(identifier, peak_bytes, takes=1, fast_path=False):
            raise OSError("the calibration file is read-only")

        stub.module.note_peak = broken
        runtime.load(CARD)
        result = runtime.render(CARD, job_for())
        assert result.peak_bytes == PEAK

    def test_the_render_line_names_the_attention_and_never_the_words(self, stub, caplog):
        caplog.set_level(logging.DEBUG, logger="model_chain")
        stub.plan["capped_take"] = 1
        runtime.load(CARD, {"attention": "sdpa"})
        secrets = ("The vault opens at dawn.", "My grandmother's maiden name.",
                   "A gravelly voice from far away.", "conspiratorially", "with a sneer")
        job = runtime.RenderJob(
            id="job-9", seed=3, takes=2, options={"attention": "sdpa"},
            lines=[runtime.Line(1, secrets[0], secrets[3]), runtime.Line(2, "Fine.")],
            voices={1: runtime.Voice(pcm=pcm(0.5), transcript=secrets[1]),
                    2: runtime.Voice(description=secrets[2], instruction=secrets[4])})
        result = runtime.render(CARD, job)
        assert result.capped is True
        rendered = [record.getMessage() for record in caplog.records
                    if "rendered" in record.getMessage()]
        assert len(rendered) == 1
        assert "SDPA" in rendered[0] and "sdpa" not in rendered[0]
        assert "2 take(s)" in rendered[0] and ", a line ran out of room" in rendered[0]
        assert ", cancelled" not in rendered[0]
        everything = "\n".join(record.getMessage() for record in caplog.records)
        for secret in secrets + (str(stub.models), str(stub.root)):
            assert secret not in everything

    def test_a_render_marks_the_card_rendering_for_its_duration(self, stub):
        thread, results, errors = render_in_background(stub)
        assert runtime.status()["cards"][CARD_KEY]["rendering"] is True
        assert runtime.status()["cards"][CARD_KEY]["job"] == "job-1"
        finish(thread)
        assert results and not errors
        assert runtime.rendering(CARD) is False
        assert runtime.status()["cards"][CARD_KEY]["job"] == ""

    def test_a_second_render_on_a_busy_card_is_refused_here(self, stub):
        thread, _results, _errors = render_in_background(stub, job_for("one"))
        with pytest.raises(runtime.BreezeRuntimeError) as raised:
            runtime.render(CARD, job_for("two"))
        assert "one render at a time" in str(raised.value)
        assert runtime.status()["cards"][CARD_KEY]["job"] == "one", "the first still owns the card"
        finish(thread, "one")
        assert stub.lines("render_requests") == ["one"], "the second never reached the worker"

    def test_cancel_reaches_a_render_in_flight(self, stub, caplog):
        caplog.set_level(logging.INFO, logger="model_chain")
        thread, results, errors = render_in_background(stub, job_for(seed=5, takes=2))
        began = time.monotonic()
        assert runtime.cancel(CARD, "job-1") is True
        thread.join(timeout=10.0)
        assert time.monotonic() - began < 5.0
        assert stub.lines("cancel_log") == ["job-1"]
        assert results and not errors and results[0].cancelled is True
        assert [len(take.pcm) // 4 for take in results[0].takes] == [2400, 0], \
            "the audio there was, and an empty entry for the take never reached"
        assert runtime.rendering(CARD) is False
        assert any("rendered" in record.getMessage() and ", cancelled" in record.getMessage()
                   for record in caplog.records)

    def test_a_cancel_never_overtakes_its_render_on_the_pipe(self, stub, monkeypatch):
        """The card is marked rendering a moment before the render's frame is
        written. A cancel made in that moment waits for the frame instead of
        reaching a worker with nothing to stop -- which would say no, and then
        run the whole render it was meant to stop."""
        built = runtime._render_frame

        def slowly(job):
            time.sleep(0.5)
            return built(job)

        monkeypatch.setattr(runtime, "_render_frame", slowly)
        stub.plan["render_seconds"] = 30.0
        runtime.load(CARD)
        results = []
        thread = threading.Thread(target=lambda: results.append(runtime.render(CARD, job_for())),
                                  daemon=True)
        thread.start()
        wait_until(lambda: runtime.rendering(CARD), what="the card rendering")
        assert stub.lines("render_requests") == [], "the cancel comes before the frame is out"
        began = time.monotonic()
        assert runtime.cancel(CARD, "job-1") is True
        thread.join(timeout=10.0)
        assert not thread.is_alive() and time.monotonic() - began < 5.0
        assert results and results[0].cancelled is True

    def test_a_cancel_waiting_for_a_render_that_never_went_out_is_let_go(self, stub,
                                                                        monkeypatch):
        def broken(job):
            time.sleep(0.5)
            raise ValueError("a voice that is not a voice")

        monkeypatch.setattr(runtime, "_render_frame", broken)
        runtime.load(CARD)
        errors = []

        def run():
            try:
                runtime.render(CARD, job_for())
            except ValueError as exc:
                errors.append(exc)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        wait_until(lambda: runtime.rendering(CARD), what="the card rendering")
        began = time.monotonic()
        assert runtime.cancel(CARD, "job-1") is False
        assert time.monotonic() - began < runtime.CANCEL_TIMEOUT / 2, "let go, not timed out"
        thread.join(timeout=5.0)
        assert errors and runtime.rendering(CARD) is False

    def test_cancel_with_nothing_rendering_is_false(self, stub):
        assert runtime.cancel(CARD, "job-1") is False
        runtime.load(CARD)
        assert runtime.cancel(CARD, "job-1") is False
        assert stub.lines("cancel_log") == [], "nothing rendering is not worth asking about"

    def test_a_worker_that_dies_mid_render_is_a_sentence_and_a_reset(self, stub):
        stub.plan["die_on_render"] = True
        runtime.load(CARD)
        with pytest.raises(runtime.BreezeRuntimeError) as raised:
            runtime.render(CARD, job_for())
        assert str(raised.value) == "the Breeze worker exited during the render"
        assert runtime.rendering(CARD) is False
        assert runtime.status()["cards"] == {}
        assert runtime.resident_bytes(CARD) == 0
        assert "stopped unexpectedly" in runtime.status()["last_error"]

    def test_a_render_that_never_answers_is_stopped_at_the_deadline(self, stub, monkeypatch):
        stub.plan["never_answers_render"] = True
        stub.plan["render_seconds"] = 0.0
        monkeypatch.setattr(runtime, "RENDER_TIMEOUT", 0.5)
        runtime.load(CARD)
        pid = stub.pids()[0]
        with pytest.raises(runtime.BreezeRuntimeError) as raised:
            runtime.render(CARD, job_for())
        assert "did not finish the render in time" in str(raised.value)
        assert runtime.status()["cards"] == {}
        assert not alive(pid)

    def test_rendering_on_a_card_with_no_worker_is_refused_with_a_sentence(self, stub):
        with pytest.raises(runtime.BreezeRuntimeError) as raised:
            runtime.render(CARD, job_for())
        assert "not loaded" in str(raised.value)
        assert stub.pids() == []

    def test_unload_keeps_the_process_and_drops_the_model(self, stub):
        runtime.load(CARD)
        pid = stub.pids()[0]
        found = runtime.unload(CARD, "test")
        assert found["cards"][CARD_KEY]["loaded"] is False
        assert found["cards"][CARD_KEY]["running"] is True
        assert alive(pid)
        assert runtime.resident_bytes(CARD) == 400_000_000
        assert runtime.evict(CARD, "after unload") == 400_000_000

    def test_unload_is_refused_while_rendering(self, stub):
        thread, _results, _errors = render_in_background(stub)
        with pytest.raises(runtime.BreezeRuntimeError):
            runtime.unload(CARD, "test")
        assert runtime.status()["cards"][CARD_KEY]["loaded"] is True
        finish(thread)


# --------------------------------------------------------------------------- #
# Eviction and the resident figure
# --------------------------------------------------------------------------- #


class TestEvictionGivesTheCardBack:
    def test_evict_stops_the_process_and_returns_what_it_held(self, stub):
        runtime.load(CARD)
        pid = stub.pids()[0]
        assert runtime.evict(CARD, "an image job needs the card") == HELD
        assert not alive(pid)
        assert runtime.resident_bytes(CARD) == 0
        assert runtime.status()["cards"] == {}
        assert runtime.rendering(CARD) is False

    def test_evict_while_rendering_does_nothing_and_says_zero(self, stub):
        thread, results, _errors = render_in_background(stub)
        pid = stub.pids()[0]
        assert runtime.evict(CARD, "an image job needs the card") == 0
        assert alive(pid)
        assert runtime.rendering(CARD) is True, "the render was never cut off"
        assert runtime.resident_bytes(CARD) == HELD
        finish(thread)
        assert results and results[0].takes
        assert runtime.evict(CARD, "now it is idle") == HELD
        assert not alive(pid)

    def test_evict_with_no_process_is_zero(self, stub):
        assert runtime.evict(CARD, "nothing there") == 0
        assert stub.pids() == []

    def test_evict_before_a_load_returns_the_figure_the_worker_reported(self, stub):
        runtime.ensure_started(CARD)
        assert runtime.evict(CARD, "test") == 0

    def test_the_resident_figure_is_refreshed_at_most_every_interval(self, stub, monkeypatch):
        runtime.load(CARD)
        assert stub.status_ops() == 0
        for _turn in range(5):
            assert runtime.resident_bytes(CARD) == HELD
        assert stub.status_ops() == 0, "the load reply is fresh enough"
        monkeypatch.setattr(runtime, "RESIDENT_REFRESH", 0.2)
        time.sleep(0.3)
        assert runtime.resident_bytes(CARD) == HELD
        wait_until(lambda: stub.status_ops() == 1, what="one status request")
        for _turn in range(5):
            runtime.resident_bytes(CARD)
        assert stub.status_ops() == 1
        time.sleep(0.3)
        runtime.resident_bytes(CARD)
        wait_until(lambda: stub.status_ops() == 2, what="a second status request")

    def test_the_resident_figure_is_never_asked_for_during_a_render(self, stub, monkeypatch):
        thread, _results, _errors = render_in_background(stub)
        monkeypatch.setattr(runtime, "RESIDENT_REFRESH", 0.05)
        before = stub.status_ops()
        time.sleep(0.2)
        for _turn in range(5):
            assert runtime.resident_bytes(CARD) == HELD
            time.sleep(0.06)
        assert runtime.rendering(CARD) is True, "every read was made during the render"
        assert stub.status_ops() == before
        finish(thread)

    def test_the_render_reply_refreshes_the_figure(self, stub):
        stub.plan["render_seconds"] = 0.0
        runtime.load(CARD)
        runtime.render(CARD, job_for())
        assert runtime.resident_bytes(CARD) == HELD
        assert stub.status_ops() == 0


# --------------------------------------------------------------------------- #
# The crash loop, shutdown, and the import graph
# --------------------------------------------------------------------------- #


class TestTheCrashLoopIsBounded:
    def test_three_failed_starts_on_a_card_stop_the_fourth(self, stub):
        stub.plan["protocol"] = 99
        for _attempt in range(runtime.CRASH_LIMIT):
            with pytest.raises(runtime.BreezeRuntimeError) as raised:
                runtime.ensure_started(CARD)
            assert "protocol" in str(raised.value)
        with pytest.raises(runtime.BreezeRuntimeError) as raised:
            runtime.ensure_started(CARD)
        assert "several times in a row" in str(raised.value)
        assert len(stub.pids()) == runtime.CRASH_LIMIT

    def test_the_guard_is_per_card(self, stub):
        stub.plan["protocol"] = 99
        for _attempt in range(runtime.CRASH_LIMIT):
            with pytest.raises(runtime.BreezeRuntimeError):
                runtime.ensure_started(CARD)
        with pytest.raises(runtime.BreezeRuntimeError) as raised:
            runtime.ensure_started(OTHER)
        assert "protocol" in str(raised.value), "the other card gets its own attempts"


class TestShutdownIsIdempotentAndArrangedOnce:
    def test_shutdown_stops_every_worker_and_may_be_called_again(self, stub):
        runtime.load(CARD)
        runtime.ensure_started(OTHER)
        pids = stub.pids()
        assert len(pids) == 2
        runtime.shutdown()
        assert runtime.status()["cards"] == {}
        for pid in pids:
            assert not alive(pid)
        runtime.shutdown()
        assert runtime.status()["cards"] == {}
        assert runtime._exit_registered is True

    def test_starting_during_shutdown_is_refused(self, stub, monkeypatch):
        monkeypatch.setattr(runtime, "_closing", True)
        with pytest.raises(runtime.BreezeRuntimeError) as raised:
            runtime.ensure_started(CARD)
        assert "shutting down" in str(raised.value)
        assert stub.pids() == []

    def test_stop_with_nothing_running_is_quiet(self, stub):
        runtime.stop("", "nothing")
        runtime.stop(CARD, "nothing")


FORBIDDEN = ("mc_memory", "mc_broker", "mc_plan", "mc_llm_runtime", "mc_turns",
             "mc_turns_guests", "torch", "numpy", "qwen_tts", "breeze_infer", "models",
             "transformers")
"""The memory side, which no voice module may reach (invariant I-3), and the
engine, which only the worker's own interpreter has."""

NETWORK = ("socket", "ssl", "http", "urllib", "requests", "httpx", "aiohttp", "ftplib")


class TestTheModuleKeepsInvariantThree:
    def test_the_runtime_imports_none_of_the_memory_side_or_the_engine_at_any_depth(self):
        path = Path(runtime.__file__)
        source = path.read_text(encoding="utf-8")
        found = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                found.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                found.add(node.module.split(".")[0])
        assert "breeze_worker" in found, "the walk reads this module's imports"
        for forbidden in FORBIDDEN + NETWORK + ("tempfile", "importlib"):
            assert forbidden not in found, forbidden
        assert "__import__(" not in source, "an import the walk cannot see"

    def test_importing_the_runtime_alone_loads_neither_the_engine_nor_the_memory_side(self):
        """In an interpreter of its own, with the installer module unavailable:
        it imports on its own, and what it brings with it is the standard
        library, the worker's framing and the Voice Box's own types."""
        code = ("import json, sys\n"
                "sys.modules['mc_voice_breeze'] = None\n"
                "import mc_voice_breeze_runtime as runtime\n"
                "assert runtime.status() == {'cards': {}, 'last_error': ''}\n"
                "print(json.dumps(sorted(name for name in sys.modules if sys.modules[name])))\n")
        done = subprocess.run([sys.executable, "-B", "-c", code], cwd=str(ROOT),
                              capture_output=True, text=True, timeout=60)
        assert done.returncode == 0, done.stderr
        loaded = set(json.loads(done.stdout.strip().splitlines()[-1]))
        assert "mc_voice_breeze_runtime" in loaded and "breeze_worker.worker" in loaded
        for forbidden in FORBIDDEN:
            assert not any(name == forbidden or name.startswith(forbidden + ".")
                           for name in loaded), forbidden
