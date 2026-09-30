"""The VibeVoice runtime: one proven worker per card, rendered through and evicted.

Driven against a real subprocess that speaks the real protocol and holds no
tensors, the way the Pocket runtime's tests are. What is asserted here is the
half of the card contract the worker cannot assert on its own:

    a worker that came up on the wrong card, on no provable card, or on no
        CUDA device at all is refused and stopped;
    one process per card, and a stop that reaches only the card it names;
    a render marks the card rendering for exactly its duration, and an
        eviction during it is refused while one after it stops the process
        and leaves the card's figure at zero;
    the resident figure is a cached read, refreshed at most every two
        seconds and never during a render, because the turns read it every
        hundred milliseconds;
    a worker that dies mid-render is a sentence and a reset, not a hang;
    the crash-loop guard and the shutdown door;
    a load that names its identity -- model, precision, LoRA, strength --
        resolved through the installer before anything starts, with the
        settings filling only what the caller left to them, and ``loaded``
        saying what is on the card;
    a streamed render whose frames reach ``on_audio`` in order on the reader
        thread, all of them before ``render`` returns, and whose ``False``
        cancels the render rather than deadlocking the reader.

Nothing here imports Torch or vibevoice. The installer module the runtime
reaches for lazily (``mc_voice_vibevoice``) is stood in by a namespace, so
these tests do not depend on it existing.
"""

from __future__ import annotations

import ast
import json
import os
import struct
import sys
import threading
import time
import types
from pathlib import Path

import pytest

import mc_voice_vibevoice_runtime as runtime

CARD = "GPU-0123abcd-4567-89ef-0123-456789abcdef"
CARD_KEY = "0123abcd456789ef0123456789abcdef"
CARD_TORCH = "0123abcd-4567-89ef-0123-456789abcdef"
OTHER = "GPU-fedcba98-7654-3210-fedc-ba9876543210"
OTHER_KEY = "fedcba9876543210fedcba9876543210"
RATE = 24000

FAKE_VIBEVOICE_WORKER = r'''#!/usr/bin/env python3
# A VibeVoice worker that speaks the real protocol and holds no tensors.
#
# Faithful in the places the parent depends on -- the framing, the handshake
# fields, the containment it arranges for itself, one render at a time, a
# cancel read while a render is in flight, progress frames -- and a stub
# everywhere else. Steered by MC_FAKE_VIBEVOICE in the environment.
import json
import os
import struct
import sys
import threading
import time

PLAN = json.loads(os.environ.get("MC_FAKE_VIBEVOICE", "{}"))
CARD = os.environ.get("CUDA_VISIBLE_DEVICES", "")
LENGTH = struct.Struct(">I")
LOCK = threading.Lock()
STATE = {"loaded": False, "job": "", "cancel": None, "resident": 0, "peak": 0,
         "model_dir": "", "loads": 0, "identity": None, "kind": "", "precision": "",
         "lora_dir": ""}


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


def wav(seconds, rate=24000):
    body = b"\x00\x00" * int(seconds * rate)
    return (b"RIFF" + struct.pack("<I", 36 + len(body)) + b"WAVEfmt "
            + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
            + b"data" + struct.pack("<I", len(body)) + body)


def mark(name, line):
    path = PLAN.get(name)
    if path:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def main():
    mark("alive_marker", str(os.getpid()))
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
            found = containment(int(header.get("parent_pid") or 0))
            reply = {"id": rid, "ok": True,
                     "protocol": PLAN.get("protocol", 1),
                     "worker": PLAN.get("worker", "vibevoice"),
                     "python": "3.13.0", "torch": "2.8.0+cu128",
                     "cuda": PLAN.get("cuda", True),
                     "device_name": PLAN.get("device_name", "Fake GPU 24GB"),
                     "device_uuid": PLAN.get("device_uuid", CARD),
                     "device_index": 0,
                     "total_vram_bytes": 24 << 30, "free_vram_bytes": 23 << 30,
                     "containment": PLAN.get("containment", found),
                     "vibevoice": "0.0.1", "transformers": "4.51.3",
                     "sample_rate": PLAN.get("sample_rate", 24000)}
            write_frame(stdout, reply)
            continue
        if operation == "status":
            mark("status_marker", "status")
            write_frame(stdout, {"id": rid, "ok": True, "loaded": STATE["loaded"],
                                 "rendering": bool(STATE["job"]), "job": STATE["job"],
                                 "resident_bytes": STATE["resident"],
                                 "peak_bytes": STATE["peak"],
                                 "model_dir": STATE["model_dir"],
                                 "kind": STATE["kind"], "precision": STATE["precision"],
                                 "lora": bool(STATE["lora_dir"])})
            continue
        if operation == "load":
            fails = PLAN.get("load_fails")
            if header.get("lora_dir") and PLAN.get("load_fails_with_lora"):
                # The worker unloads before it joins a LoRA, so a LoRA that
                # does not fit leaves nothing loaded.
                STATE["loaded"] = False
                STATE["identity"] = None
                fails = PLAN["load_fails_with_lora"]
            if fails:
                write_frame(stdout, {"id": rid, "ok": False, "error": fails,
                                     "refusal": " " in fails})
                continue
            time.sleep(float(PLAN.get("load_seconds", 0)))
            identity = [header.get(name) for name in ("model_dir", "kind", "precision",
                                                      "lora_dir", "lora_scale", "device")]
            already = STATE["loaded"] and identity == STATE["identity"]
            STATE["loaded"] = True
            STATE["identity"] = identity
            STATE["resident"] = int(PLAN.get("resident_bytes", 18000000000))
            STATE["model_dir"] = header.get("model_dir", "")
            STATE["kind"] = header.get("kind") or "longform"
            STATE["precision"] = header.get("precision") or header.get("dtype") or "bf16"
            STATE["lora_dir"] = header.get("lora_dir") or ""
            STATE["loads"] += 1
            mark("load_log", json.dumps(header))
            write_frame(stdout, {"id": rid, "ok": True, "resident_bytes": STATE["resident"],
                                 "weights_bytes": 17900000000, "load_seconds": 0.5,
                                 "already_loaded": already})
            continue
        if operation == "unload":
            STATE["loaded"] = False
            STATE["identity"] = None
            STATE["resident"] = int(PLAN.get("empty_bytes", 400000000))
            STATE["model_dir"] = ""
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
            mark("render_log", json.dumps({"header": header, "payload_bytes": len(payload)}))
            STATE["job"] = header.get("job") or "-"
            event = threading.Event()
            STATE["cancel"] = event

            def run(rid=rid, event=event, stream=header.get("stream") is True):
                seconds = float(PLAN.get("render_seconds", 0.05))
                began = time.monotonic()
                if stream:
                    # One frame per chunk, PCM16 whose first sample is the
                    # frame's own number, then a reply with no WAV.
                    sent = 0
                    for number in range(int(PLAN.get("stream_frames", 4))):
                        if event.is_set():
                            break
                        time.sleep(float(PLAN.get("frame_gap", 0.01)))
                        sent += 1
                        write_frame(stdout, {"id": rid, "audio": {"seq": sent, "samples": 2400,
                                                                  "rate": 24000}},
                                    struct.pack("<h", sent) * 2400)
                    mark("streamed_log", str(sent))
                    STATE["peak"] = int(PLAN.get("peak_bytes", 20000000000))
                    reply = {"id": rid, "ok": True, "seconds": sent * 2400 / 24000.0,
                             "sample_rate": 24000, "render_seconds": time.monotonic() - began,
                             "peak_bytes": STATE["peak"], "resident_bytes": STATE["resident"],
                             "cancelled": event.is_set(), "tokens": sent, "capped": False,
                             "streamed": True, "first_audio_ms": 9, "frames": sent}
                    STATE["job"] = ""
                    STATE["cancel"] = None
                    write_frame(stdout, reply)
                    return
                while time.monotonic() - began < seconds:
                    if event.is_set():
                        break
                    time.sleep(0.02)
                    write_frame(stdout, {"id": rid, "progress":
                                         {"seconds": (time.monotonic() - began) * 10.0}})
                if PLAN.get("never_answers_render"):
                    while True:
                        time.sleep(0.05)
                if PLAN.get("stray_frame"):
                    # A frame for this request that is neither progress, audio
                    # nor the reply: a newer worker's notice, say.
                    write_frame(stdout, {"id": rid, "notice": "something new"})
                STATE["peak"] = int(PLAN.get("peak_bytes", 20000000000))
                audio_seconds = 0.5 if event.is_set() else 1.0
                reply = {"id": rid, "ok": True, "seconds": audio_seconds,
                         "sample_rate": 24000,
                         "render_seconds": time.monotonic() - began,
                         "peak_bytes": STATE["peak"], "resident_bytes": STATE["resident"],
                         "cancelled": event.is_set(), "tokens": 42, "capped": False}
                STATE["job"] = ""
                STATE["cancel"] = None
                write_frame(stdout, reply, wav(audio_seconds))

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


def pcm(seconds: float) -> bytes:
    return struct.pack("<f", 0.25) * int(seconds * RATE)


def job_for(identifier="job-1", **values):
    found = dict(id=identifier,
                 script=[(1, "Hello there."), (2, "Hi.")],
                 voices={1: (pcm(0.5), RATE), 2: (pcm(0.25), RATE)},
                 cfg_scale=1.3, steps=10, seed=7, max_new_tokens=None)
    found.update(values)
    return runtime.RenderJob(**found)


MODEL_7B = "vibevoice-7b"
MODEL_REALTIME = "vibevoice-realtime-0.5b"

MODELS = {
    MODEL_7B: {"id": MODEL_7B, "label": "VibeVoice 7B", "kind": "longform",
               "precisions": ["bf16", "int8", "nf4"], "lora": True, "max_speakers": 4,
               "voices": "samples", "defaults": {"steps": 10, "cfg_scale": 1.3}},
    MODEL_REALTIME: {"id": MODEL_REALTIME, "label": "VibeVoice Realtime 0.5B",
                     "kind": "realtime", "precisions": ["bf16"], "lora": False,
                     "max_speakers": 1, "voices": "presets",
                     "defaults": {"steps": 5, "cfg_scale": 1.5}},
}
"""The two models as the installer's ``model_info`` describes them (the
contract's names; the installer is stood in, so these tests do not depend on
its manifest)."""


class InstallerError(RuntimeError):
    """The installer's ``VibeVoiceError``: a sentence for the user."""


class Stub:
    """The stood-in installer module and the fake worker's plan, per test."""

    def __init__(self, root: Path, plan: dict, settings: dict, peaks: list, refusal: list,
                 module, asked: list, loras: dict):
        self.root = root
        self.plan = plan
        self.settings = settings
        self.peaks = peaks
        self.refusal = refusal
        self.module = module
        self.asked = asked
        self.loras = loras

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

    def cancels(self) -> list:
        return self.lines("cancel_log")

    def streamed(self) -> list:
        return [int(line) for line in self.lines("streamed_log")]


@pytest.fixture
def stub(tmp_path, monkeypatch):
    script = tmp_path / "fake_vibevoice_worker.py"
    script.write_text(FAKE_VIBEVOICE_WORKER, encoding="utf-8")
    model = tmp_path / "model"
    model.mkdir()
    realtime = tmp_path / "realtime"
    realtime.mkdir()
    plan = {name: str(tmp_path / f"{name}.txt")
            for name in ("alive_marker", "status_marker", "render_log", "render_requests",
                         "load_log", "init_log", "cancel_log", "streamed_log")}
    settings = {"card_uuid": CARD, "model_id": MODEL_7B, "steps": 10,
                "cfg_scale": 1.3, "seed": None, "max_new_tokens": None, "keep_warm": True,
                "precision": "bf16", "lora_id": "", "lora_scale": 1.0}
    peaks = []
    refusal = [""]
    asked = []
    loras = {"warm-lora": str(tmp_path / "loras" / "warm-lora")}

    def model_info(identifier=""):
        name = identifier or settings["model_id"]
        if name not in MODELS:
            raise InstallerError(f"{name!r} is not a VibeVoice model this build has recorded.")
        return dict(MODELS[name])

    def lora_dir(identifier):
        if identifier not in loras:
            raise InstallerError("That LoRA is not in the library.")
        return Path(loras[identifier])

    def refuse(manual=False, model_id=""):
        asked.append(model_id)
        return refusal[0]

    module = types.SimpleNamespace(
        GUEST="VibeVoice", LABEL="VibeVoice", MODEL_DEFAULT=MODEL_7B, MODEL_7B=MODEL_7B,
        MODEL_REALTIME=MODEL_REALTIME, VibeVoiceError=InstallerError,
        runtime_python=lambda: Path(sys.executable),
        worker_environment=lambda card: {"MC_FAKE_VIBEVOICE": json.dumps(plan),
                                         "CUDA_VISIBLE_DEVICES": str(card)},
        worker_script=lambda: script,
        model_dir=lambda identifier="": realtime if identifier == MODEL_REALTIME else model,
        model_info=model_info,
        voices_dir=lambda identifier=MODEL_REALTIME: realtime / "voices",
        lora_dir=lora_dir,
        settings=lambda: dict(settings),
        note_peak=lambda identifier, peak_bytes, rss_bytes=0, precision="":
        peaks.append((identifier, int(peak_bytes), int(rss_bytes), precision)),
        refusal=refuse)
    monkeypatch.setitem(sys.modules, "mc_voice_vibevoice", module)
    monkeypatch.setattr(runtime, "STOP_GRACE", 3.0)
    monkeypatch.setattr(runtime, "TERMINATE_GRACE", 1.5)
    found = Stub(tmp_path, plan, settings, peaks, refusal, module, asked, loras)
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


# --------------------------------------------------------------------------- #
# Card keys
# --------------------------------------------------------------------------- #


class TestCardKeysAreTheHexDigitsBothSpellingsShare:
    def test_nvidia_smi_and_torch_spell_one_card(self):
        assert runtime.card_key(CARD) == CARD_KEY
        assert runtime.card_key(CARD_TORCH) == CARD_KEY
        assert runtime.card_key(CARD.upper()) == CARD_KEY

    def test_nothing_is_the_empty_key(self):
        assert runtime.card_key("") == ""
        assert runtime.card_key(None) == ""
        assert runtime.card_key("GPU-") == ""


# --------------------------------------------------------------------------- #
# The handshake
# --------------------------------------------------------------------------- #


class TestTheHandshakeRefusesWhatThisBuildCannotAccept:
    def test_a_good_worker_starts_and_reports_what_it_is(self, stub):
        found = runtime.ensure_started(CARD)
        assert isinstance(found, runtime.Handshake)
        assert found.cuda is True and found.device_uuid == CARD
        assert found.device_name == "Fake GPU 24GB"
        assert found.protocol == 1 and found.worker == "vibevoice"
        assert found.sample_rate == 24000
        card = runtime.status()["cards"][CARD_KEY]
        assert card["running"] is True and card["loaded"] is False
        assert card["device_name"] == "Fake GPU 24GB" and card["uuid"] == CARD
        assert len(stub.pids()) == 1
        init = stub.inits()[0]
        assert init["expect_uuid"] == CARD_KEY
        assert init["parent_pid"] == os.getpid()

    def test_the_worker_may_spell_the_uuid_the_way_torch_does(self, stub):
        stub.plan["device_uuid"] = CARD_TORCH
        found = runtime.ensure_started(CARD)
        assert found.device_uuid == CARD_TORCH

    def test_a_worker_on_another_card_is_refused_and_stopped(self, stub):
        stub.plan["device_uuid"] = OTHER
        with pytest.raises(runtime.VibeVoiceRuntimeError) as raised:
            runtime.ensure_started(CARD)
        assert "not the card asked for" in str(raised.value)
        assert OTHER in str(raised.value)
        assert runtime.status()["cards"] == {}
        assert not alive(stub.pids()[0])
        assert runtime.status()["last_error"] == str(raised.value)

    def test_a_worker_that_cannot_prove_its_card_is_refused(self, stub):
        stub.plan["device_uuid"] = ""
        with pytest.raises(runtime.VibeVoiceRuntimeError) as raised:
            runtime.ensure_started(CARD)
        assert "cannot prove which card" in str(raised.value)
        assert runtime.status()["cards"] == {}

    def test_a_worker_without_cuda_is_refused(self, stub):
        stub.plan["cuda"] = False
        with pytest.raises(runtime.VibeVoiceRuntimeError) as raised:
            runtime.ensure_started(CARD)
        assert "no CUDA device" in str(raised.value)
        assert runtime.status()["cards"] == {}

    def test_a_worker_that_saw_no_cuda_device_answers_with_the_sentence(self, stub):
        stub.plan["no_cuda"] = True
        with pytest.raises(runtime.VibeVoiceRuntimeError) as raised:
            runtime.ensure_started(CARD)
        assert str(raised.value) == "no CUDA device is visible to the worker"

    def test_a_protocol_this_build_does_not_speak_is_refused(self, stub):
        stub.plan["protocol"] = 99
        with pytest.raises(runtime.VibeVoiceRuntimeError) as raised:
            runtime.ensure_started(CARD)
        assert "protocol 99" in str(raised.value)

    def test_a_worker_that_is_not_vibevoice_is_refused(self, stub):
        stub.plan["worker"] = "pocket"
        with pytest.raises(runtime.VibeVoiceRuntimeError) as raised:
            runtime.ensure_started(CARD)
        assert "identify itself" in str(raised.value)

    @pytest.mark.skipif(os.name == "nt", reason="the parent arranges the job on Windows")
    def test_a_worker_without_containment_is_refused(self, stub):
        stub.plan["containment"] = "none"
        with pytest.raises(runtime.VibeVoiceRuntimeError) as raised:
            runtime.ensure_started(CARD)
        assert "could not confirm" in str(raised.value)

    def test_a_sample_rate_that_is_not_the_models_is_refused(self, stub):
        stub.plan["sample_rate"] = 22050
        with pytest.raises(runtime.VibeVoiceRuntimeError) as raised:
            runtime.ensure_started(CARD)
        assert "22050" in str(raised.value) and "24000" in str(raised.value)

    def test_no_runtime_installed_starts_nothing(self, stub):
        stub.module.runtime_python = lambda: None
        with pytest.raises(runtime.VibeVoiceRuntimeError) as raised:
            runtime.ensure_started(CARD)
        assert "not installed" in str(raised.value)
        assert stub.pids() == []

    def test_the_installers_refusal_is_the_answer_and_starts_nothing(self, stub):
        stub.refusal[0] = "VibeVoice's model is not installed yet."
        with pytest.raises(runtime.VibeVoiceRuntimeError) as raised:
            runtime.ensure_started(CARD)
        assert str(raised.value) == "VibeVoice's model is not installed yet."
        assert stub.pids() == []

    def test_no_card_named_is_refused_before_anything_is_looked_at(self, stub):
        with pytest.raises(runtime.VibeVoiceRuntimeError):
            runtime.ensure_started("")
        assert stub.pids() == []

    def test_a_second_start_on_the_same_card_is_the_same_process(self, stub):
        first = runtime.ensure_started(CARD)
        second = runtime.ensure_started(CARD_TORCH)
        assert first is second
        assert len(stub.pids()) == 1

    def test_the_worker_is_given_exactly_one_card_and_no_credential(self, stub, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "secret")
        seen = {}
        real = runtime.subprocess.Popen

        def watching(command, **kwargs):
            seen["command"] = list(command)
            seen["env"] = dict(kwargs["env"])
            return real(command, **kwargs)

        monkeypatch.setattr(runtime.subprocess, "Popen", watching)
        runtime.ensure_started(CARD)
        assert seen["env"]["CUDA_VISIBLE_DEVICES"] == CARD
        assert "HF_TOKEN" not in seen["env"]
        assert runtime.protocol.MARKER in seen["command"]
        assert "--parent-pid" in seen["command"]


class TestStatusIsAPureRead:
    def test_status_before_any_start_is_empty_and_asks_nothing(self, stub):
        assert runtime.status() == {"cards": {}, "last_error": ""}
        assert runtime.resident_bytes(CARD) == 0
        assert runtime.rendering(CARD) is False
        assert runtime.peak_observed(CARD) == 0
        assert stub.pids() == []

    def test_status_never_sends_a_request(self, stub):
        runtime.ensure_started(CARD)
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
        assert runtime.resident_bytes(CARD) == 18_000_000_000
        assert runtime.resident_bytes(OTHER) == 0
        assert runtime.status()["cards"][CARD_KEY]["loaded"] is True
        assert runtime.status()["cards"][OTHER_KEY]["loaded"] is False


# --------------------------------------------------------------------------- #
# Loading and rendering
# --------------------------------------------------------------------------- #


class TestLoadingAndRendering:
    def test_load_starts_the_worker_and_asks_for_the_settings_model(self, stub):
        found = runtime.load(CARD)
        assert found == {"loaded": True, "resident_bytes": 18_000_000_000,
                         "weights_bytes": 17_900_000_000, "load_seconds": 0.5}
        header = stub.loads()[0]
        assert header["model_dir"] == str(stub.root / "model")
        assert header["steps"] == 10
        assert header["dtype"] == "bf16" and header["attention"] == "sdpa"
        assert runtime.status()["cards"][CARD_KEY]["loaded"] is True
        assert runtime.resident_bytes(CARD) == 18_000_000_000, "from the load reply"

    def test_a_render_builds_the_frame_the_worker_reads(self, stub):
        runtime.load(CARD)
        progress = []
        result = runtime.render(CARD, job_for(seed=7, cfg_scale=1.5, steps=12,
                                              max_new_tokens=None), progress.append)
        assert isinstance(result, runtime.RenderResult)
        assert result.wav[:4] == b"RIFF"
        assert result.seconds == 1.0 and result.sample_rate == 24000
        assert result.cancelled is False and result.tokens == 42
        assert result.peak_bytes == 20_000_000_000 and result.capped is False
        assert result.render_seconds >= 0.0
        sent = stub.renders()[0]
        header = sent["header"]
        assert header["job"] == "job-1"
        assert header["script"] == [{"speaker": 1, "text": "Hello there."},
                                    {"speaker": 2, "text": "Hi."}]
        assert header["voices"] == [{"speaker": 1, "offset": 0, "count": 12000, "rate": RATE},
                                    {"speaker": 2, "offset": 12000, "count": 6000,
                                     "rate": RATE}]
        assert sent["payload_bytes"] == 4 * 18000
        assert header["seed"] == 7 and header["cfg_scale"] == 1.5 and header["steps"] == 12
        assert header["max_new_tokens"] is None
        assert progress and all("seconds" in one for one in progress)

    def test_a_render_marks_the_card_rendering_for_its_duration(self, stub):
        stub.plan["render_seconds"] = 1.0
        runtime.load(CARD)
        assert runtime.rendering(CARD) is False
        results = []
        thread = threading.Thread(target=lambda: results.append(runtime.render(CARD, job_for())))
        thread.start()
        wait_until(lambda: runtime.rendering(CARD), what="the card rendering")
        assert runtime.status()["cards"][CARD_KEY]["rendering"] is True
        assert runtime.status()["cards"][CARD_KEY]["job"] == "job-1"
        thread.join(timeout=10.0)
        assert results and results[0].cancelled is False
        assert runtime.rendering(CARD) is False
        assert runtime.status()["cards"][CARD_KEY]["job"] == ""

    def test_a_second_render_on_a_busy_card_is_refused_here(self, stub):
        stub.plan["render_seconds"] = 1.0
        runtime.load(CARD)
        thread = threading.Thread(target=lambda: runtime.render(CARD, job_for("one")))
        thread.start()
        wait_until(lambda: runtime.rendering(CARD), what="the card rendering")
        with pytest.raises(runtime.VibeVoiceRuntimeError) as raised:
            runtime.render(CARD, job_for("two"))
        assert "one render at a time" in str(raised.value)
        assert runtime.status()["cards"][CARD_KEY]["job"] == "one", "the first still owns the card"
        thread.join(timeout=10.0)
        assert stub.lines("render_requests") == ["one"], "the second never reached the worker"

    def test_cancel_reaches_a_render_in_flight(self, stub):
        stub.plan["render_seconds"] = 20.0
        runtime.load(CARD)
        results = []
        thread = threading.Thread(target=lambda: results.append(runtime.render(CARD, job_for())))
        thread.start()
        wait_until(lambda: runtime.rendering(CARD), what="the card rendering")
        began = time.monotonic()
        assert runtime.cancel(CARD, "job-1") is True
        thread.join(timeout=10.0)
        assert time.monotonic() - began < 5.0
        assert results and results[0].cancelled is True
        assert results[0].seconds == 0.5, "the audio there was"
        assert runtime.rendering(CARD) is False

    def test_cancel_with_nothing_rendering_is_false(self, stub):
        assert runtime.cancel(CARD, "job-1") is False
        runtime.load(CARD)
        assert runtime.cancel(CARD, "job-1") is False

    def test_a_worker_that_dies_mid_render_is_a_sentence_and_a_reset(self, stub):
        stub.plan["die_on_render"] = True
        runtime.load(CARD)
        with pytest.raises(runtime.VibeVoiceRuntimeError) as raised:
            runtime.render(CARD, job_for())
        assert str(raised.value) == "the VibeVoice worker exited during the render"
        assert runtime.rendering(CARD) is False
        assert runtime.status()["cards"] == {}
        assert runtime.resident_bytes(CARD) == 0
        assert "stopped unexpectedly" in runtime.status()["last_error"]

    def test_a_render_that_never_answers_is_stopped_at_the_deadline(self, stub, monkeypatch):
        stub.plan["never_answers_render"] = True
        stub.plan["render_seconds"] = 0.0
        monkeypatch.setattr(runtime, "RENDER_TIMEOUT", 0.5)
        runtime.load(CARD)
        with pytest.raises(runtime.VibeVoiceRuntimeError) as raised:
            runtime.render(CARD, job_for())
        assert "did not finish the render in time" in str(raised.value)
        assert runtime.status()["cards"] == {}

    def test_rendering_on_a_card_with_no_worker_is_refused_with_a_sentence(self, stub):
        with pytest.raises(runtime.VibeVoiceRuntimeError) as raised:
            runtime.render(CARD, job_for())
        assert "not loaded" in str(raised.value)
        assert stub.pids() == []

    def test_the_workers_refusal_is_the_sentence_the_user_sees(self, stub):
        stub.plan["load_fails"] = "the model directory does not exist"
        with pytest.raises(runtime.VibeVoiceRuntimeError) as raised:
            runtime.load(CARD)
        assert str(raised.value) == "the model directory does not exist"
        assert runtime.status()["cards"][CARD_KEY]["running"] is True, "a refusal is not a crash"

    def test_a_library_failure_becomes_a_sentence_rather_than_a_class_name(self, stub):
        stub.plan["load_fails"] = "OutOfMemoryError"
        with pytest.raises(runtime.VibeVoiceRuntimeError) as raised:
            runtime.load(CARD)
        assert "ran out of memory" in str(raised.value)
        assert "OutOfMemoryError" not in str(raised.value)

    def test_the_peak_is_recorded_with_the_installer_and_remembered(self, stub):
        runtime.load(CARD)
        assert runtime.peak_observed(CARD) == 0
        runtime.render(CARD, job_for())
        assert stub.peaks == [("vibevoice-7b", 20_000_000_000, 0, "bf16")]
        assert runtime.peak_observed(CARD) == 20_000_000_000
        assert runtime.status()["cards"][CARD_KEY]["peak_bytes"] == 20_000_000_000

    def test_a_progress_callback_that_raises_does_not_end_the_render(self, stub):
        stub.plan["render_seconds"] = 0.2
        runtime.load(CARD)

        def broken(progress):
            raise RuntimeError("the page went away")

        result = runtime.render(CARD, job_for(), broken)
        assert result.cancelled is False and result.seconds == 1.0

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
        stub.plan["render_seconds"] = 1.0
        runtime.load(CARD)
        thread = threading.Thread(target=lambda: runtime.render(CARD, job_for()))
        thread.start()
        wait_until(lambda: runtime.rendering(CARD), what="the card rendering")
        with pytest.raises(runtime.VibeVoiceRuntimeError):
            runtime.unload(CARD, "test")
        thread.join(timeout=10.0)


# --------------------------------------------------------------------------- #
# Eviction and the resident figure
# --------------------------------------------------------------------------- #


class TestEvictionGivesTheCardBack:
    def test_evict_stops_the_process_and_returns_what_it_held(self, stub):
        runtime.load(CARD)
        pid = stub.pids()[0]
        assert runtime.evict(CARD, "an image job needs the card") == 18_000_000_000
        assert not alive(pid)
        assert runtime.resident_bytes(CARD) == 0
        assert runtime.status()["cards"] == {}
        assert runtime.rendering(CARD) is False

    def test_evict_while_rendering_does_nothing_and_says_zero(self, stub):
        stub.plan["render_seconds"] = 1.0
        runtime.load(CARD)
        pid = stub.pids()[0]
        results = []
        thread = threading.Thread(target=lambda: results.append(runtime.render(CARD, job_for())))
        thread.start()
        wait_until(lambda: runtime.rendering(CARD), what="the card rendering")
        assert runtime.evict(CARD, "an image job needs the card") == 0
        assert alive(pid)
        assert runtime.resident_bytes(CARD) == 18_000_000_000
        thread.join(timeout=10.0)
        assert results and results[0].cancelled is False
        assert runtime.evict(CARD, "now it is idle") == 18_000_000_000

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
            assert runtime.resident_bytes(CARD) == 18_000_000_000
        assert stub.status_ops() == 0, "the load reply is fresh enough"
        monkeypatch.setattr(runtime, "RESIDENT_REFRESH", 0.2)
        time.sleep(0.3)
        assert runtime.resident_bytes(CARD) == 18_000_000_000
        wait_until(lambda: stub.status_ops() == 1, what="one status request")
        for _turn in range(5):
            runtime.resident_bytes(CARD)
        assert stub.status_ops() == 1
        time.sleep(0.3)
        runtime.resident_bytes(CARD)
        wait_until(lambda: stub.status_ops() == 2, what="a second status request")

    def test_the_resident_figure_is_never_asked_for_during_a_render(self, stub, monkeypatch):
        stub.plan["render_seconds"] = 1.0
        runtime.load(CARD)
        monkeypatch.setattr(runtime, "RESIDENT_REFRESH", 0.05)
        thread = threading.Thread(target=lambda: runtime.render(CARD, job_for()))
        thread.start()
        wait_until(lambda: runtime.rendering(CARD), what="the card rendering")
        before = stub.status_ops()
        time.sleep(0.2)
        for _turn in range(5):
            assert runtime.resident_bytes(CARD) == 18_000_000_000
            time.sleep(0.06)
        assert stub.status_ops() == before
        thread.join(timeout=10.0)

    def test_the_render_reply_refreshes_the_figure(self, stub):
        stub.plan["render_seconds"] = 0.0
        runtime.load(CARD)
        runtime.render(CARD, job_for())
        assert runtime.resident_bytes(CARD) == 18_000_000_000
        assert stub.status_ops() == 0


# --------------------------------------------------------------------------- #
# A load names its identity
# --------------------------------------------------------------------------- #


class TestALoadNamesItsIdentity:
    def test_no_arguments_take_the_whole_identity_from_the_settings(self, stub):
        stub.settings.update({"precision": "int8", "lora_id": "warm-lora", "lora_scale": 0.6})
        runtime.load(CARD)
        header = stub.loads()[0]
        assert header["model_dir"] == str(stub.root / "model")
        assert header["kind"] == "longform" and header["precision"] == "int8"
        assert header["lora_dir"] == stub.loras["warm-lora"] and header["lora_scale"] == 0.6
        assert header["voices_dir"] == "" and header["device"] == "cuda"
        assert header["steps"] == 10 and header["attention"] == "sdpa"
        assert runtime.loaded(CARD) == {"model_id": MODEL_7B, "kind": "longform",
                                        "precision": "int8", "lora_id": "warm-lora",
                                        "lora_scale": 0.6}

    def test_a_named_model_and_no_lora_is_no_lora_whatever_the_settings_hold(self, stub):
        """The render service names its configuration's model and passes an
        empty LoRA for a configuration without one; the Voice Box settings'
        LoRA must not ride in on that."""
        stub.settings.update({"lora_id": "warm-lora", "lora_scale": 0.6})
        runtime.load(CARD, model_id=MODEL_7B, precision="bf16", lora_id="", lora_scale=1.0)
        header = stub.loads()[0]
        assert header["lora_dir"] is None and header["lora_scale"] == 1.0
        assert runtime.loaded(CARD)["lora_id"] == ""

    def test_an_explicit_identity_is_sent_as_named(self, stub):
        runtime.load(CARD, model_id=MODEL_7B, precision="nf4", lora_id="warm-lora",
                     lora_scale=1.5)
        header = stub.loads()[0]
        assert (header["precision"], header["lora_dir"], header["lora_scale"]) == \
            ("nf4", stub.loras["warm-lora"], 1.5)
        assert runtime.loaded(CARD) == {"model_id": MODEL_7B, "kind": "longform",
                                        "precision": "nf4", "lora_id": "warm-lora",
                                        "lora_scale": 1.5}

    def test_the_realtime_model_gets_its_voices_its_steps_and_full_precision(self, stub):
        stub.settings["precision"] = "int8"
        runtime.load(CARD, model_id=MODEL_REALTIME)
        header = stub.loads()[0]
        assert header["kind"] == "realtime" and header["precision"] == "bf16"
        assert header["model_dir"] == str(stub.root / "realtime")
        assert header["voices_dir"] == str(stub.root / "realtime" / "voices")
        assert header["lora_dir"] is None
        assert header["steps"] == 5, "its own default: the settings' steps are the 7B's"
        assert runtime.loaded(CARD)["kind"] == "realtime"

    @pytest.mark.parametrize("values, sentence", [
        ({"model_id": MODEL_REALTIME, "lora_id": "warm-lora"},
         "VibeVoice Realtime 0.5B does not take a LoRA."),
        ({"model_id": MODEL_REALTIME, "precision": "int8"},
         "VibeVoice Realtime 0.5B does not load at that precision."),
        ({"model_id": MODEL_7B, "precision": "fp8"},
         "VibeVoice 7B does not load at that precision."),
        ({"model_id": MODEL_7B, "lora_id": "warm-lora", "lora_scale": 3.0},
         "A LoRA's strength is from 0 to 2."),
        ({"model_id": MODEL_7B, "lora_id": "nobody-made-this"},
         "That LoRA is not in the library."),
        ({"model_id": "vibevoice-9000"},
         "'vibevoice-9000' is not a VibeVoice model this build has recorded."),
    ])
    def test_what_the_model_cannot_take_is_refused_before_anything_starts(self, stub, values,
                                                                           sentence):
        with pytest.raises(runtime.VibeVoiceRuntimeError) as raised:
            runtime.load(CARD, **values)
        assert str(raised.value) == sentence
        assert stub.pids() == [] and stub.loads() == []

    def test_the_installer_is_asked_about_the_model_being_loaded(self, stub):
        runtime.load(CARD, model_id=MODEL_REALTIME)
        assert stub.asked and set(stub.asked) == {MODEL_REALTIME}
        stub.refusal[0] = "The Realtime model is not installed yet."
        runtime.evict(CARD, "test")
        with pytest.raises(runtime.VibeVoiceRuntimeError) as raised:
            runtime.load(CARD, model_id=MODEL_REALTIME)
        assert str(raised.value) == "The Realtime model is not installed yet."

    def test_a_worker_already_running_is_still_asked_nothing_for_a_model_not_installed(
            self, stub):
        runtime.load(CARD)
        stub.refusal[0] = "The Realtime model is not installed yet."
        with pytest.raises(runtime.VibeVoiceRuntimeError):
            runtime.load(CARD, model_id=MODEL_REALTIME)
        assert len(stub.loads()) == 1, "the running worker was not sent a load it would fail"

    def test_another_identity_reaches_the_worker_and_loaded_follows_it(self, stub):
        runtime.load(CARD)
        runtime.load(CARD, model_id=MODEL_REALTIME)
        assert [header["kind"] for header in stub.loads()] == ["longform", "realtime"]
        assert runtime.loaded(CARD)["model_id"] == MODEL_REALTIME
        assert runtime.status()["cards"][CARD_KEY]["model"]["kind"] == "realtime"
        assert len(stub.pids()) == 1, "one worker per card, whatever it holds"

    def test_loaded_is_none_whenever_nothing_is_known_to_be_there(self, stub):
        assert runtime.loaded(CARD) is None
        runtime.load(CARD)
        assert runtime.loaded(CARD)["model_id"] == MODEL_7B
        runtime.unload(CARD, "test")
        assert runtime.loaded(CARD) is None
        assert runtime.status()["cards"][CARD_KEY]["model"] is None
        runtime.load(CARD)
        runtime.evict(CARD, "test")
        assert runtime.loaded(CARD) is None

    def test_a_failed_load_forgets_what_was_loaded(self, stub):
        stub.plan["load_fails_with_lora"] = ("the LoRA's language-model adapter does not fit "
                                             "this model")
        runtime.load(CARD)
        assert runtime.loaded(CARD) is not None
        with pytest.raises(runtime.VibeVoiceRuntimeError) as raised:
            runtime.load(CARD, model_id=MODEL_7B, lora_id="warm-lora")
        assert "does not fit" in str(raised.value)
        assert runtime.loaded(CARD) is None
        assert runtime.status()["cards"][CARD_KEY]["loaded"] is False

    def test_a_worker_that_says_it_holds_nothing_clears_what_this_side_thought(self, stub,
                                                                            monkeypatch):
        runtime.load(CARD)
        runtime.unload(CARD, "test")
        runtime.load(CARD)
        # The worker's own status reply is the authority on "loaded".
        monkeypatch.setattr(runtime, "RESIDENT_REFRESH", 0.0)
        stub.plan["resident_bytes"] = 0
        record = runtime._live(CARD_KEY)
        with record.lock:
            runtime._absorb(record, {"loaded": False, "rendering": False})
        assert runtime.loaded(CARD) is None


# --------------------------------------------------------------------------- #
# Streamed renders, presets, and what a render is noted as
# --------------------------------------------------------------------------- #


class TestAStreamedRender:
    def test_frames_reach_on_audio_in_order_on_the_reader_thread(self, stub):
        runtime.load(CARD)
        heard = []
        result = runtime.render(CARD, job_for(),
                                on_audio=lambda pcm, rate: heard.append(
                                    (pcm, rate, threading.current_thread().name)))
        assert [struct.unpack("<h", pcm[:2])[0] for pcm, _rate, _name in heard] == [1, 2, 3, 4]
        assert all(len(pcm) == 2400 * 2 and rate == RATE for pcm, rate, _name in heard)
        assert all(name.startswith("mc-vibevoice-reader") for _pcm, _rate, name in heard)
        assert result.streamed is True and result.wav == b""
        assert result.first_audio_ms >= 1 and result.cancelled is False
        assert result.seconds == pytest.approx(4 * 2400 / RATE)
        assert stub.renders()[0]["header"]["stream"] is True

    def test_the_first_frame_is_timed_from_this_side(self, stub):
        """What a listener waits: from the request to the first frame here,
        not the figure the worker reports about itself (9 ms in the fake)."""
        stub.plan["frame_gap"] = 0.25
        runtime.load(CARD)
        result = runtime.render(CARD, job_for(), on_audio=lambda pcm, rate: True)
        assert result.first_audio_ms >= 250

    def test_every_frame_is_delivered_before_render_returns(self, stub):
        stub.plan["stream_frames"] = 30
        stub.plan["frame_gap"] = 0.0
        runtime.load(CARD)
        heard = []
        runtime.render(CARD, job_for(), on_audio=lambda pcm, rate: heard.append(pcm))
        assert len(heard) == 30

    def test_a_false_return_cancels_the_render_and_stops_delivery(self, stub):
        stub.plan["stream_frames"] = 200
        stub.plan["frame_gap"] = 0.02
        runtime.load(CARD)
        heard = []

        def on_audio(pcm, rate):
            heard.append(pcm)
            return len(heard) < 2

        began = time.monotonic()
        result = runtime.render(CARD, job_for("spoken"), on_audio=on_audio)
        assert time.monotonic() - began < 3.0, "not the whole two hundred frames"
        assert len(heard) == 2, "nothing is delivered after the False"
        assert result.cancelled is True
        wait_until(lambda: stub.cancels() == ["spoken"], what="the cancel reaching the worker")
        assert stub.streamed()[0] < 200
        assert runtime.rendering(CARD) is False

    def test_an_on_audio_that_raises_is_taken_as_a_refusal(self, stub):
        stub.plan["stream_frames"] = 200
        stub.plan["frame_gap"] = 0.02
        runtime.load(CARD)
        calls = []

        def on_audio(pcm, rate):
            calls.append(pcm)
            raise RuntimeError("the turn went away")

        result = runtime.render(CARD, job_for(), on_audio=on_audio)
        assert len(calls) == 1 and result.cancelled is True
        wait_until(lambda: stub.cancels() == ["job-1"], what="the cancel reaching the worker")

    def test_a_job_that_streams_with_no_on_audio_gets_its_frames_as_a_wav(self, stub):
        runtime.load(CARD)
        result = runtime.render(CARD, job_for(stream=True))
        assert result.streamed is True and result.wav[:4] == b"RIFF"
        assert len(result.wav) == 44 + 4 * 2400 * 2
        assert stub.renders()[0]["header"]["stream"] is True

    def test_a_frame_that_is_not_the_reply_is_never_taken_for_it(self, stub):
        stub.plan["stray_frame"] = True
        runtime.load(CARD)
        result = runtime.render(CARD, job_for())
        assert result.seconds == 1.0 and result.wav[:4] == b"RIFF" and result.tokens == 42

    def test_a_render_that_does_not_stream_is_as_it_was(self, stub):
        runtime.load(CARD)
        result = runtime.render(CARD, job_for())
        assert result.streamed is False and result.first_audio_ms == 0
        assert result.wav[:4] == b"RIFF" and result.seconds == 1.0
        assert stub.renders()[0]["header"]["stream"] is False

    def test_a_clone_preview_is_one_speaker_speaking_with_a_recording(self, stub):
        """Voice Chat's clone preview renders one line for Speaker 1 with a
        recording, through exactly the path a Voice Box render takes."""
        runtime.load(CARD)
        voice = pcm(1.3)
        result = runtime.render(CARD, runtime.RenderJob(
            id="preview", script=[(1, "This is how I sound.")],
            voices={1: (voice, RATE)}, cfg_scale=1.3, steps=10, seed=None))
        sent = stub.renders()[0]
        assert sent["header"]["voices"] == [{"speaker": 1, "offset": 0,
                                             "count": len(voice) // 4, "rate": RATE}]
        assert sent["payload_bytes"] == len(voice)
        assert result.wav[:4] == b"RIFF"

    def test_preset_voices_are_named_and_carry_no_audio(self, stub):
        runtime.load(CARD, model_id=MODEL_REALTIME)
        runtime.render(CARD, runtime.RenderJob(
            id="realtime", script=[(1, "Hello there, how are you today?")],
            voices={1: {"preset": "en-Carter_man"}}, cfg_scale=None, steps=0))
        sent = stub.renders()[0]
        assert sent["header"]["voices"] == [{"speaker": 1, "preset": "en-Carter_man"}]
        assert sent["payload_bytes"] == 0
        assert sent["header"]["cfg_scale"] is None, "the model's own guidance"
        assert sent["header"]["steps"] == 0, "the steps it was loaded with"

    def test_a_voice_that_is_neither_a_recording_nor_a_preset_is_refused_here(self, stub):
        runtime.load(CARD)
        with pytest.raises(runtime.VibeVoiceRuntimeError) as raised:
            runtime.render(CARD, runtime.RenderJob(id="x", script=[(1, "Hi.")],
                                                   voices={1: {"sample": "abc"}}))
        assert "neither a recording nor a preset" in str(raised.value)
        assert stub.renders() == [] and runtime.rendering(CARD) is False

    def test_the_peak_is_noted_for_the_model_and_precision_that_are_loaded(self, stub):
        runtime.load(CARD, model_id=MODEL_7B, precision="nf4")
        runtime.render(CARD, job_for())
        runtime.load(CARD, model_id=MODEL_REALTIME)
        runtime.render(CARD, runtime.RenderJob(id="r", script=[(1, "Hello there, friend.")],
                                               voices={1: {"preset": "en-Carter_man"}}))
        assert stub.peaks == [(MODEL_7B, 20_000_000_000, 0, "nf4"),
                              (MODEL_REALTIME, 20_000_000_000, 0, "bf16")]


# --------------------------------------------------------------------------- #
# The crash loop, shutdown, and the import graph
# --------------------------------------------------------------------------- #


class TestTheCrashLoopIsBounded:
    def test_three_failed_starts_on_a_card_stop_the_fourth(self, stub):
        stub.plan["protocol"] = 99
        for _attempt in range(runtime.CRASH_LIMIT):
            with pytest.raises(runtime.VibeVoiceRuntimeError) as raised:
                runtime.ensure_started(CARD)
            assert "protocol" in str(raised.value)
        with pytest.raises(runtime.VibeVoiceRuntimeError) as raised:
            runtime.ensure_started(CARD)
        assert "several times in a row" in str(raised.value)
        assert len(stub.pids()) == runtime.CRASH_LIMIT

    def test_the_guard_is_per_card(self, stub):
        stub.plan["protocol"] = 99
        for _attempt in range(runtime.CRASH_LIMIT):
            with pytest.raises(runtime.VibeVoiceRuntimeError):
                runtime.ensure_started(CARD)
        with pytest.raises(runtime.VibeVoiceRuntimeError) as raised:
            runtime.ensure_started(OTHER)
        assert "protocol" in str(raised.value), "the other card gets its own attempts"


class TestShutdownIsIdempotentAndArrangedOnce:
    def test_shutdown_stops_every_worker_and_may_be_called_again(self, stub):
        runtime.load(CARD)
        runtime.ensure_started(OTHER)
        pids = stub.pids()
        runtime.shutdown()
        assert runtime.status()["cards"] == {}
        for pid in pids:
            assert not alive(pid)
        runtime.shutdown()
        assert runtime._exit_registered is True

    def test_starting_during_shutdown_is_refused(self, stub, monkeypatch):
        monkeypatch.setattr(runtime, "_closing", True)
        with pytest.raises(runtime.VibeVoiceRuntimeError) as raised:
            runtime.ensure_started(CARD)
        assert "shutting down" in str(raised.value)
        assert stub.pids() == []

    def test_stop_with_nothing_running_is_quiet(self, stub):
        runtime.stop("", "nothing")
        runtime.stop(CARD, "nothing")


class TestTheModuleKeepsInvariantThree:
    def test_the_runtime_imports_none_of_the_memory_side_at_any_depth(self):
        path = Path(runtime.__file__)
        found = set()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                found.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                found.add(node.module.split(".")[0])
        for forbidden in ("mc_memory", "mc_broker", "mc_plan", "mc_llm_runtime", "mc_turns",
                          "torch", "vibevoice", "numpy", "transformers"):
            assert forbidden not in found, forbidden

    def test_the_runtime_imports_without_the_installer_module(self, monkeypatch):
        """Part B's module is reached lazily, so this one loads on its own."""
        import importlib

        monkeypatch.setitem(sys.modules, "mc_voice_vibevoice", None)
        module = importlib.import_module("mc_voice_vibevoice_runtime")
        assert module.status() == {"cards": {}, "last_error": ""} or "cards" in module.status()
