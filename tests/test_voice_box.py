"""The Voice Box's files and its render service, with no card and no worker.

Samples, prompts, configurations, pipelines and outputs are files under the
voice data root; the render service asks a turn client for a card and a runtime
for the sound. Both are doubles here -- the turn client records what was asked
for and answers the phase a test chooses; the runtime returns a tone -- so every
rule of the service is proved without a GPU: what is refused before a card is
asked for, what the turn is asked for, what happens when it is granted, blocked
or withdrawn, and how the card is handed back.
"""

from __future__ import annotations

import json
import struct
import threading
import time
import wave
import io

import pytest

import mc_voice_box as box
from conftest import _spoken_wav, _wav

_GB = 1024**3
CARD = "GPU-11112222-3333-4444-5555-666677778888"


def spoken(seconds: float = 4.0, rate: int = 24000) -> bytes:
    return _spoken_wav(seconds, rate)


def tone_pcm(seconds: float = 1.0, rate: int = 24000) -> bytes:
    data = spoken(seconds, rate)
    with wave.open(io.BytesIO(data), "rb") as handle:
        return handle.readframes(handle.getnframes())


# --------------------------------------------------------------------------- #
# Doubles
# --------------------------------------------------------------------------- #


class FakeTurn:
    def __init__(self, outcome: str = "granted", looks: int = 1, warning: str = ""):
        self.outcome = outcome
        self.looks = looks
        self.seen = 0
        self.phase = "queued"
        self.reason = "waiting for the work already on the card"
        self.warning = warning
        self.finished: list = []
        self.cancelled = False

    def wait(self, timeout=None, cancelled=None):
        if cancelled is not None and cancelled.is_set():
            self.cancel()
            return "cancelled"
        self.seen += 1
        if self.seen >= self.looks:
            self.phase = self.outcome
            return self.outcome
        return "clearing"

    def finish(self, keep_warm=None):
        self.finished.append(keep_warm)

    def cancel(self):
        self.cancelled = True
        self.phase = "cancelled"


class FakeTurns:
    def __init__(self):
        self.turn = FakeTurn()
        self.requests: list = []
        self.unloaded: list = []

    def request(self, card, *, need_vram, need_ram=0, label=""):
        self.requests.append({"card": card, "need_vram": need_vram, "need_ram": need_ram,
                              "label": label})
        return self.turn

    def snapshot(self):
        return [{"card": "the card", "uuid": CARD}]

    def cards(self):
        return [{"uuid": CARD, "key": "1111222233334444555566667777888", "index": 1,
                 "name": "NVIDIA GeForce RTX 5090", "image_card": False, "wangp_card": True}]

    def unload(self, card, reason):
        self.unloaded.append((card, reason))
        return 18 * _GB


class Result:
    def __init__(self, wav: bytes, seconds: float, cancelled: bool = False):
        self.wav = wav
        self.seconds = seconds
        self.sample_rate = 24000
        self.render_seconds = 2.5
        self.peak_bytes = 19 * _GB
        self.cancelled = cancelled
        self.tokens = 120


class RenderJob:
    def __init__(self, **fields):
        self.__dict__.update(fields)


class FakeRuntime:
    RenderJob = RenderJob

    def __init__(self, seconds: float = 1.0):
        self.seconds = seconds
        self.loaded: list = []
        self.renders: list = []
        self.cancelled: list = []
        self.fail = None
        self.cancel_on_render = False
        self.hold = None

    def load(self, card):
        self.loaded.append(card)
        return {"resident_bytes": 18 * _GB}

    def render(self, card, job, on_progress=None):
        self.renders.append(job)
        if self.hold is not None:
            self.hold.wait(5.0)
        if self.fail is not None:
            raise self.fail
        if on_progress is not None:
            on_progress({"seconds": self.seconds / 2})
        pcm = tone_pcm(self.seconds)
        return Result(box.wav_bytes(pcm), self.seconds, cancelled=self.cancel_on_render)

    def cancel(self, card, job_id):
        self.cancelled.append(job_id)
        return True

    def status(self, card=""):
        return {"cards": {}, "last_error": ""}

    def stop(self, card="", reason=""):
        return None


class FakeEngine:
    LABEL = "VibeVoice 7B"
    GUEST = "VibeVoice"

    def __init__(self):
        self.refused = ""
        self.peaks: list = []
        self.values = {"steps": 10, "cfg_scale": 1.3, "seed": None, "max_new_tokens": None}

    def refusal(self, manual=False):
        return self.refused

    def need_vram_bytes(self, identifier=""):
        return 20 * _GB

    def need_ram_bytes(self, identifier=""):
        return 3 * _GB

    def note_peak(self, identifier, peak_bytes, rss_bytes=0):
        self.peaks.append((identifier, peak_bytes))

    def settings(self):
        return dict(self.values)

    def set_settings(self, values):
        self.values.update(values)
        return dict(self.values)

    def public_status(self):
        return {"ready": True, "message": "", "parts": []}

    def progress(self):
        return {}

    def install(self, part=""):
        return None

    def install_from(self, part, folder):
        return None


REAL_MP3 = box._encode_mp3
"""The real encoder, before any test puts its stand-in there."""


def no_mp3(pcm16, rate, tags):
    return None


@pytest.fixture(autouse=True)
def _fresh(voice_root, monkeypatch):
    # Whether this machine can make an MP3 is PyAV's business, not a test's:
    # every test keeps WAVs unless it puts an encoder in place itself.
    monkeypatch.setattr(box, "_encode_mp3", no_mp3)
    box.forget()
    yield
    box.forget()


@pytest.fixture
def turns():
    found = FakeTurns()
    box.use_turns(found)
    return found


@pytest.fixture
def runtime():
    found = FakeRuntime()
    box.use_runtime(found)
    return found


@pytest.fixture
def engine():
    found = FakeEngine()
    box.use_engine(found)
    return found


@pytest.fixture
def ready(turns, runtime, engine):
    """A pipeline, two samples on speakers 1 and 2, a configuration on the card."""
    first = box.add_sample(spoken(), "Ada")
    second = box.add_sample(spoken(5.0), "Brook")
    configuration = box.save_configuration({"name": "Studio", "card_uuid": CARD,
                                            "model_id": "vibevoice-7b",
                                            "speakers": {"1": first["id"], "2": second["id"]}})
    found = box.new_pipeline("Trailer")
    return {"pipeline": found, "configuration": configuration, "samples": (first, second)}


def settled(identifier: str, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        found = box.job(identifier)
        if not found["live"]:
            return found
        time.sleep(0.01)
    raise AssertionError(f"the job never settled: {box.job(identifier)}")


def phase_reached(identifier: str, phase: str, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        found = box.job(identifier)
        if found["phase"] == phase or not found["live"]:
            return found
        time.sleep(0.005)
    raise AssertionError(f"the job never reached {phase}: {box.job(identifier)}")


# --------------------------------------------------------------------------- #
# Samples
# --------------------------------------------------------------------------- #


class TestSamples:
    def test_a_sample_is_kept_canonical_with_a_waveform_to_draw(self):
        entry = box.add_sample(spoken(4.0, 48000), "  Ada   Lovelace ", "file")

        assert entry["title"] == "Ada Lovelace" and entry["rate"] == 24000
        assert entry["seconds"] == pytest.approx(4.0, abs=0.05)
        assert len(entry["peaks"]) == box.PEAKS and max(entry["peaks"]) > 0.3
        pcm, rate = box.sample_pcm(entry["id"])
        assert rate == 24000 and len(pcm) == pytest.approx(4.0 * 24000 * 2, rel=0.01)
        assert box.samples() == [entry]

    def test_a_recording_too_short_or_silent_is_refused_in_a_sentence(self):
        with pytest.raises(box.VoiceBoxError, match="1.0 seconds long"):
            box.add_sample(spoken(1.0), "Blip")
        with pytest.raises(box.VoiceBoxError):
            box.add_sample(_wav(b"\0\0" * (24000 * 4), 24000), "Nothing")
        with pytest.raises(box.VoiceBoxError, match="WAV"):
            box.add_sample(b"not audio at all", "Junk")
        assert box.samples() == []

    def test_a_deleted_sample_leaves_the_speaker_slots_that_used_it(self):
        entry = box.add_sample(spoken(), "Ada")
        other = box.add_sample(spoken(), "Brook")
        configuration = box.save_configuration({"speakers": {"1": entry["id"], "2": other["id"]}})

        box.delete_sample(entry["id"])

        assert [s["id"] for s in box.samples()] == [other["id"]]
        assert box.configuration(configuration["id"])["speakers"] == {"2": other["id"]}
        with pytest.raises(box.NotFound):
            box.sample(entry["id"])

    def test_renaming_keeps_everything_but_the_title(self):
        entry = box.add_sample(spoken(), "Ada")
        renamed = box.rename_sample(entry["id"], "Ada, take two")
        assert renamed["title"] == "Ada, take two" and renamed["peaks"] == entry["peaks"]
        assert box.rename_sample(entry["id"], "   ")["title"] == "Ada, take two"

    def test_an_identifier_that_is_not_one_is_not_found_rather_than_a_path(self):
        with pytest.raises(box.NotFound, match="not in the Voice Box"):
            box.sample("../../etc/passwd")
        with pytest.raises(box.NotFound, match="not in the Voice Box"):
            box.output_audio("")


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #


class TestPrompts:
    def test_the_same_words_are_one_entry_at_the_top(self):
        box.remember_prompt("Speaker 1: Hello.")
        box.remember_prompt("Speaker 1: Goodbye.")
        again = box.remember_prompt("Speaker 1: Hello.")

        found = box.prompts()
        assert [entry["text"] for entry in found["history"]] == ["Speaker 1: Hello.",
                                                                  "Speaker 1: Goodbye."]
        assert found["history"][0]["id"] == again["id"] and found["favourites"] == []

    def test_the_history_is_capped_and_favourites_are_never_dropped(self):
        first = box.remember_prompt("the very first")
        box.favourite_prompt(first["id"], True)
        for number in range(box.HISTORY_LIMIT + 5):
            box.remember_prompt(f"prompt number {number}")

        found = box.prompts()
        texts = [entry["text"] for entry in found["history"]]
        assert "the very first" in texts and "prompt number 0" not in texts
        assert len(texts) == box.HISTORY_LIMIT + 1
        assert [entry["text"] for entry in found["favourites"]] == ["the very first"]

    def test_a_favourite_can_be_unmarked_and_a_prompt_deleted(self):
        entry = box.remember_prompt("keep me")
        assert box.favourite_prompt(entry["id"], True)["favourite"] is True
        assert box.favourite_prompt(entry["id"], False)["favourite"] is False
        box.delete_prompt(entry["id"])
        assert box.prompts()["history"] == []
        with pytest.raises(box.NotFound):
            box.delete_prompt(entry["id"])

    def test_an_empty_or_enormous_prompt_is_not_remembered(self):
        with pytest.raises(box.VoiceBoxError):
            box.remember_prompt("   ")
        with pytest.raises(box.VoiceBoxError, match="at most"):
            box.remember_prompt("x" * (box.MAX_PROMPT_CHARS + 1))


# --------------------------------------------------------------------------- #
# Configurations and pipelines
# --------------------------------------------------------------------------- #


class TestConfigurations:
    def test_a_configuration_is_validated_field_by_field(self):
        entry = box.add_sample(spoken(), "Ada")
        found = box.save_configuration({"name": " Late night ", "steps": "12", "cfg_scale": 1.5,
                                        "seed": "", "max_new_tokens": None, "card_uuid": CARD,
                                        "speakers": {"1": entry["id"], "2": "nope", "9": entry["id"]}})
        assert found["name"] == "Late night" and found["steps"] == 12
        assert found["cfg_scale"] == 1.5 and found["seed"] is None
        assert found["speakers"] == {"1": entry["id"]}
        assert found["card_uuid"] == CARD and found["id"]

        for bad in ({"steps": 0}, {"steps": 51}, {"cfg_scale": 0.5}, {"cfg_scale": "x"},
                    {"seed": -1}, {"max_new_tokens": 0}):
            with pytest.raises(box.VoiceBoxError):
                box.save_configuration(dict(bad, id=found["id"]))
        assert box.configuration(found["id"])["steps"] == 12

    def test_an_update_keeps_the_identity_and_the_rest_of_the_fields(self):
        found = box.save_configuration({"name": "A", "steps": 7})
        time.sleep(0.01)
        updated = box.save_configuration({"id": found["id"], "cfg_scale": 2.0})
        assert updated["id"] == found["id"] and updated["steps"] == 7
        assert updated["cfg_scale"] == 2.0 and updated["created"] == found["created"]
        assert updated["updated"] > found["updated"]
        box.delete_configuration(found["id"])
        assert box.configurations() == []

    def test_a_pipeline_is_named_kept_and_its_outputs_survive_it(self):
        first = box.new_pipeline("")
        assert first["name"] == "Pipeline 1" and first["outputs"] == []
        configuration = box.save_configuration({"name": "A"})
        saved = box.save_pipeline({"id": first["id"], "prompt": "Speaker 1: Hi",
                                   "configuration_id": configuration["id"], "name": "Intro"})
        assert saved["prompt"] == "Speaker 1: Hi" and saved["name"] == "Intro"
        with pytest.raises(box.NotFound):
            box.save_pipeline({"id": first["id"], "configuration_id": "0" * 16})
        with pytest.raises(box.VoiceBoxError, match="at most"):
            box.save_pipeline({"id": first["id"], "prompt": "y" * (box.MAX_PROMPT_CHARS + 1)})

        entry = box.add_output(tone_pcm(0.5), 24000, "Take 1", first["id"], {"seed": 3})
        assert box.pipeline(first["id"])["outputs"] == [entry["id"]]
        box.delete_pipeline(first["id"])
        assert box.pipelines() == []
        assert box.output(entry["id"])["pipeline_id"] == ""


# --------------------------------------------------------------------------- #
# Outputs and the save folder
# --------------------------------------------------------------------------- #


class TestOutputs:
    def test_an_output_carries_its_waveform_and_can_loop(self):
        owner = box.new_pipeline("P")
        entry = box.add_output(tone_pcm(2.0), 24000, "Take", owner["id"], {"steps": 10})
        assert entry["seconds"] == pytest.approx(2.0, abs=0.01)
        assert len(entry["peaks"]) == box.PEAKS and entry["loop"] is False
        assert box.set_loop(entry["id"], True)["loop"] is True
        assert box.rename_output(entry["id"], "Final")["name"] == "Final"
        assert box.outputs(owner["id"])[0]["name"] == "Final"
        data, kind = box.output_audio(entry["id"])
        pcm, rate = box.pcm_of(data)
        assert kind == "wav" and rate == 24000 and len(pcm) == len(tone_pcm(2.0))

        box.delete_output(entry["id"])
        assert box.outputs() == [] and box.pipeline(owner["id"])["outputs"] == []

    def test_saving_needs_a_folder_once_and_never_overwrites(self, tmp_path):
        entry = box.add_output(tone_pcm(0.5), 24000, 'Take: "one"?', "", {"seed": 1})
        with pytest.raises(box.VoiceBoxError, match="Choose a folder"):
            box.save_output(entry["id"])

        box.set_settings({"save_folder": str(tmp_path / "renders")})
        first = box.save_output(entry["id"])
        second = box.save_output(entry["id"])

        assert first.endswith("Take one.wav") and second.endswith("Take one (2).wav")
        assert (tmp_path / "renders" / "Take one.json").exists()
        sidecar = json.loads((tmp_path / "renders" / "Take one.json").read_text("utf-8"))
        assert sidecar["render"] == {"seed": 1} and "peaks" not in sidecar
        with wave.open(first, "rb") as handle:
            assert handle.getframerate() == 24000 and handle.getnchannels() == 1

    def test_the_folder_dialog_remembers_what_it_is_told(self, monkeypatch, tmp_path):
        import mc_llm_native

        asked = []
        monkeypatch.setattr(mc_llm_native, "choose_folder",
                            lambda title, initial=None: asked.append((title, initial))
                            or str(tmp_path / "picked"))
        assert box.choose_save_folder() == str(tmp_path / "picked")
        assert box.settings()["save_folder"] == str(tmp_path / "picked")
        assert asked[0][1] is None

        monkeypatch.setattr(mc_llm_native, "choose_folder", lambda title, initial=None: None)
        assert box.choose_save_folder() == ""
        assert box.settings()["save_folder"] == str(tmp_path / "picked")

    def test_settings_hold_the_card_the_model_and_the_warm_stay(self):
        found = box.set_settings({"card_uuid": CARD, "model_id": "vibevoice-7b",
                                  "keep_warm": 0, "save_folder": ""})
        assert found == {"save_folder": "", "card_uuid": CARD, "model_id": "vibevoice-7b",
                         "keep_warm": False}
        assert box.settings() == found


# --------------------------------------------------------------------------- #
# The script
# --------------------------------------------------------------------------- #


class TestTheScript:
    def test_speakers_continuations_and_pauses(self):
        sections = box.parse_script(
            "Speaker 1: Hello there.\n  and welcome.\nSPEAKER 2: Thanks!\n[pause]\n"
            "[2]: Second part. [pause:1500] Speaker 1: Last words.")

        assert [section.pause_ms for section in sections] == [None, 700, 1500]
        assert sections[0].lines == [(1, "Hello there. and welcome."), (2, "Thanks!")]
        assert sections[1].lines == [(2, "Second part.")]
        assert sections[2].lines == [(1, "Last words.")]
        assert sections[0].speakers == [1, 2]

    def test_text_without_a_speaker_is_speaker_one(self):
        sections = box.parse_script("Just read this.\nAnd this.")
        assert sections[0].lines == [(1, "Just read this. And this.")]

    def test_pauses_are_clamped_and_adjacent_ones_add_up(self):
        sections = box.parse_script("a [pause:5] b [pause:99999] c [pause] [pause:300] d")
        assert [section.pause_ms for section in sections] == [None, box.PAUSE_MIN_MS,
                                                              box.PAUSE_MAX_MS, 1000]
        assert box.parse_script("[pause] a [pause]")[0].pause_ms is None

    def test_a_fifth_speaker_and_an_empty_prompt_are_refused(self):
        with pytest.raises(box.VoiceBoxError, match="1 to 4"):
            box.parse_script("Speaker 5: no")
        with pytest.raises(box.VoiceBoxError, match="no words"):
            box.parse_script("[pause] [pause:200]")
        summary = box.script_summary("Speaker 1: one two\n[pause]\nSpeaker 3: three")
        assert summary == {"ok": True, "speakers": [1, 3], "pauses": 1, "words": 3,
                           "sections": 2}
        assert box.script_summary("")["ok"] is False


# --------------------------------------------------------------------------- #
# The render service
# --------------------------------------------------------------------------- #


class TestRendering:
    def test_a_render_asks_for_its_card_and_makes_an_output_of_its_sections(self, ready):
        turns, runtime, engine = box.turns(), box._runtime(), box._engine()
        found = box.render(ready["pipeline"]["id"], "Speaker 1: Hi.\n[pause:1000]\nSpeaker 2: Yo.",
                           ready["configuration"]["id"], "Two lines")

        job = settled(found["id"])
        assert job["phase"] == "done" and job["output_id"], job
        assert turns.requests == [{"card": CARD, "need_vram": 20 * _GB, "need_ram": 3 * _GB,
                                   "label": "VibeVoice 7B — Two lines"}]
        assert turns.turn.finished == [None]
        assert runtime.loaded == [CARD] and len(runtime.renders) == 2
        assert runtime.renders[0].script == [(1, "Hi.")] and runtime.renders[1].script == [(2, "Yo.")]
        assert list(runtime.renders[0].voices) == [1] and list(runtime.renders[1].voices) == [2]
        assert runtime.renders[0].cfg_scale == 1.3 and runtime.renders[0].steps == 10
        entry = box.output(job["output_id"])
        assert entry["seconds"] == pytest.approx(3.0, abs=0.02)
        assert entry["pipeline_id"] == ready["pipeline"]["id"]
        assert entry["render"]["sections"] == 2 and entry["render"]["peak_bytes"] == 19 * _GB
        assert [s["title"] for s in entry["render"]["speakers"]] == ["Ada", "Brook"]
        assert box.pipeline(ready["pipeline"]["id"])["outputs"] == [entry["id"]]
        assert engine.peaks == [], ("the runtime notes every render's peak itself; a note "
                                    "here would count the render twice")
        assert engine.values["model_id"] == "vibevoice-7b", "the engine loads what it is told"
        assert box.prompts()["history"][0]["text"].startswith("Speaker 1: Hi.")
        assert job["progress"]["sections"] == 2

    def test_what_is_wrong_is_said_before_any_card_is_asked_for(self, ready):
        pipeline, configuration = ready["pipeline"]["id"], ready["configuration"]["id"]
        with pytest.raises(box.VoiceBoxError, match="Speaker 3 has no sample"):
            box.render(pipeline, "Speaker 3: Who?", configuration)
        with pytest.raises(box.VoiceBoxError, match="no words"):
            box.render(pipeline, "[pause]", configuration)
        box.save_configuration({"id": configuration, "card_uuid": ""})
        with pytest.raises(box.VoiceBoxError, match="Choose the card"):
            box.render(pipeline, "Speaker 1: Hi", configuration)
        box.set_settings({"card_uuid": CARD})
        box._engine().refused = "VibeVoice's runtime is not installed."
        with pytest.raises(box.VoiceBoxError, match="not installed"):
            box.render(pipeline, "Speaker 1: Hi", configuration)
        assert box.turns().requests == [] and box.jobs() == []

    def test_a_blocked_turn_fails_the_job_with_the_turns_warning(self, ready):
        turns = box.turns()
        turns.turn = FakeTurn("blocked", warning="VibeVoice 7B needs 20.0 GB on the card")
        found = box.render(ready["pipeline"]["id"], "Speaker 1: Hi", ready["configuration"]["id"])

        job = settled(found["id"])
        assert job["phase"] == "failed" and job["warning"] == "VibeVoice 7B needs 20.0 GB on the card"
        assert box._runtime().loaded == [] and turns.turn.finished == [None]
        assert box.outputs() == []

    def test_the_warm_stay_can_be_declined_but_never_forced(self, ready):
        box.set_settings({"keep_warm": False})
        found = box.render(ready["pipeline"]["id"], "Speaker 1: Hi", ready["configuration"]["id"])
        settled(found["id"])
        assert box.turns().turn.finished == [False]

    def test_a_job_waiting_for_its_turn_reports_what_the_turn_is_waiting_for(self, ready):
        turns = box.turns()
        turns.turn = FakeTurn("granted", looks=3)
        found = box.render(ready["pipeline"]["id"], "Speaker 1: Hi", ready["configuration"]["id"])
        job = settled(found["id"])
        assert job["phase"] == "done"
        assert turns.turn.seen == 3

    def test_a_queued_job_can_be_withdrawn_and_a_waiting_one_cancelled(self, ready, monkeypatch):
        turns = box.turns()
        gate = threading.Event()

        class Held(FakeTurn):
            def wait(self, timeout=None, cancelled=None):
                gate.set()
                if cancelled is not None and cancelled.is_set():
                    self.cancel()
                    return "cancelled"
                time.sleep(0.005)
                return "clearing"

        turns.turn = Held()
        first = box.render(ready["pipeline"]["id"], "Speaker 1: One", ready["configuration"]["id"])
        second = box.render(ready["pipeline"]["id"], "Speaker 1: Two", ready["configuration"]["id"])
        assert gate.wait(2.0)
        assert box.job(second["id"])["phase"] == "queued"

        assert box.cancel_job(second["id"])["phase"] == "cancelled"
        assert box.cancel_job(first["id"])["phase"] in ("waiting", "cancelled")
        assert settled(first["id"])["phase"] == "cancelled"
        assert turns.turn.cancelled and turns.turn.finished == [None]
        assert box._runtime().loaded == []

    def test_a_rendering_job_is_told_to_stop_and_the_card_handed_back(self, ready):
        runtime = box._runtime()
        runtime.hold = threading.Event()
        runtime.cancel_on_render = True
        found = box.render(ready["pipeline"]["id"], "Speaker 1: Long", ready["configuration"]["id"])
        job = phase_reached(found["id"], "rendering")
        assert job["phase"] == "rendering"

        box.cancel_job(found["id"])
        runtime.hold.set()

        job = settled(found["id"])
        assert job["phase"] == "cancelled" and box.outputs() == []
        assert runtime.cancelled == [f"{found['id']}:1"]
        assert box.turns().turn.finished == [None]

    def test_a_worker_failure_fails_the_job_and_still_hands_the_card_back(self, ready):
        runtime = box._runtime()
        runtime.fail = RuntimeError("the VibeVoice worker exited during the render")
        found = box.render(ready["pipeline"]["id"], "Speaker 1: Hi", ready["configuration"]["id"])

        job = settled(found["id"])
        assert job["phase"] == "failed" and "exited during the render" in job["warning"]
        assert box.turns().turn.finished == [None] and box.outputs() == []

    def test_renders_on_one_card_run_in_order_and_are_listed_newest_first(self, ready):
        pipeline, configuration = ready["pipeline"]["id"], ready["configuration"]["id"]
        first = box.render(pipeline, "Speaker 1: One", configuration)
        second = box.render(pipeline, "Speaker 1: Two", configuration)
        settled(first["id"])
        settled(second["id"])

        listed = box.jobs()
        assert [job["id"] for job in listed] == [second["id"], first["id"]]
        assert [job["name"] for job in reversed(listed)] == ["Trailer 1", "Trailer 2"]
        outputs = box.outputs(pipeline)
        assert [entry["name"] for entry in outputs] == ["Trailer 2", "Trailer 1"]

    def test_without_a_turn_client_nothing_is_queued(self, ready):
        box.use_turns(None)
        with pytest.raises(box.VoiceBoxError, match="not connected"):
            box.render(ready["pipeline"]["id"], "Speaker 1: Hi", ready["configuration"]["id"])

    def test_stop_cancels_what_is_queued_and_running(self, ready):
        runtime = box._runtime()
        runtime.hold = threading.Event()
        runtime.cancel_on_render = True
        first = box.render(ready["pipeline"]["id"], "Speaker 1: One", ready["configuration"]["id"])
        second = box.render(ready["pipeline"]["id"], "Speaker 1: Two", ready["configuration"]["id"])
        phase_reached(first["id"], "rendering")

        box.stop()
        # A render asked to stop after the door closed would finish normally:
        # the queued job is cancelled by the stop itself, not by the worker.
        runtime.cancel_on_render = False
        runtime.hold.set()

        assert settled(first["id"])["phase"] == "cancelled"
        assert settled(second["id"])["phase"] == "cancelled"
        assert runtime.cancelled == [f"{first['id']}:1"]
        assert runtime.loaded == [CARD], "the queued job never asked for the card"


class TestSound:
    def test_float_samples_are_the_same_sound(self):
        pcm = struct.pack("<4h", 0, 16384, -32768, 32767)
        floats = struct.unpack("<4f", box.float32_of(pcm))
        assert floats == pytest.approx((0.0, 0.5, -1.0, 32767 / 32768))

    def test_peaks_follow_the_loudness_across_the_sound(self):
        quiet = struct.pack("<h", 300) * 1000
        loud = struct.pack("<h", 30000) * 1000
        peaks = box.peaks_of(quiet + loud, count=4)
        assert peaks[0] == pytest.approx(300 / 32768) and peaks[3] == pytest.approx(30000 / 32768)
        assert box.peaks_of(b"", count=3) == [0.0, 0.0, 0.0]
        assert len(box.silence(700)) == 2 * int(24000 * 0.7)


# --------------------------------------------------------------------------- #
# Seeds, the recorded configuration and the infotext
# --------------------------------------------------------------------------- #


class TestSeeds:
    def test_a_blank_seed_is_drawn_once_and_every_section_renders_with_it(self, ready):
        assert ready["configuration"]["seed"] is None
        found = box.render(ready["pipeline"]["id"], "Speaker 1: Hi.\n[pause]\nSpeaker 2: Yo.",
                           ready["configuration"]["id"])
        assert isinstance(found["seed"], int) and 0 <= found["seed"] <= box.SEED_MAX
        assert found["seed_drawn"] is True

        job = settled(found["id"])
        runtime = box._runtime()
        assert [request.seed for request in runtime.renders] == [found["seed"]] * 2
        render = box.output(job["output_id"])["render"]
        assert render["seed"] == found["seed"] and render["seed_drawn"] is True
        assert render["configuration"]["seed"] == found["seed"], \
            "Reuse settings makes the same render again"

    def test_a_configured_seed_is_the_one_used_and_is_not_drawn(self, ready):
        box.save_configuration({"id": ready["configuration"]["id"], "seed": 1234})
        found = box.render(ready["pipeline"]["id"], "Speaker 1: Hi.",
                           ready["configuration"]["id"])
        assert found["seed"] == 1234 and found["seed_drawn"] is False

        job = settled(found["id"])
        assert box._runtime().renders[0].seed == 1234
        render = box.output(job["output_id"])["render"]
        assert render["seed"] == 1234 and render["seed_drawn"] is False

    def test_two_blank_renders_draw_their_own_seeds(self, ready):
        seeds = {box.render(ready["pipeline"]["id"], "Speaker 1: Hi.",
                            ready["configuration"]["id"])["seed"] for _ in range(4)}
        assert len(seeds) > 1


class TestTheRecordedConfiguration:
    def test_an_output_records_the_configuration_it_was_made_with(self, ready):
        first, second = ready["samples"]
        found = box.render(ready["pipeline"]["id"], "Speaker 1: Hi.",
                           ready["configuration"]["id"])
        record = box.output(settled(found["id"])["output_id"])["render"]["configuration"]

        assert record == {"id": ready["configuration"]["id"], "name": "Studio",
                          "model_id": "vibevoice-7b", "card_uuid": CARD, "steps": 10,
                          "cfg_scale": 1.3, "seed": found["seed"], "max_new_tokens": None,
                          "speakers": {"1": first["id"], "2": second["id"]}}, \
            "every slot is kept, used by this script or not"

    def test_unsaved_changes_are_recorded_with_the_configuration_they_changed(self, ready):
        inline = dict(ready["configuration"], steps=20, cfg_scale=2.0, name="Studio")
        found = box.render(ready["pipeline"]["id"], "Speaker 1: Hi.",
                           ready["configuration"]["id"], inline=inline)
        record = box.output(settled(found["id"])["output_id"])["render"]["configuration"]
        assert record["id"] == ready["configuration"]["id"]
        assert record["steps"] == 20 and record["cfg_scale"] == 2.0
        assert box._runtime().renders[0].steps == 20

    def test_changes_to_a_deleted_configuration_still_render_and_name_none(self, ready):
        box.delete_configuration(ready["configuration"]["id"])
        inline = dict(ready["configuration"], steps=12)
        found = box.render(ready["pipeline"]["id"], "Speaker 1: Hi.",
                           ready["configuration"]["id"], inline=inline)
        record = box.output(settled(found["id"])["output_id"])["render"]["configuration"]
        assert record["id"] == "" and record["steps"] == 12
        with pytest.raises(box.NotFound):
            box.render(ready["pipeline"]["id"], "Speaker 1: Hi.", ready["configuration"]["id"])

    def test_a_blank_model_is_recorded_as_the_model_the_engine_loads(self, ready, monkeypatch):
        box.save_configuration({"id": ready["configuration"]["id"], "model_id": ""})
        monkeypatch.setattr(box._engine(), "model_id", lambda: "vibevoice-7b", raising=False)
        found = box.render(ready["pipeline"]["id"], "Speaker 1: Hi.",
                           ready["configuration"]["id"])
        render = box.output(settled(found["id"])["output_id"])["render"]
        assert render["model_id"] == "vibevoice-7b"
        assert render["configuration"]["model_id"] == "", "the configuration's own choice"


class TestInfotext:
    RENDER = {"prompt": "Speaker 1: Hello there.\n[pause]\nSpeaker 2: Hi.", "steps": 10,
              "cfg_scale": 1.3, "seed": 1234, "seed_drawn": True, "model_id": "vibevoice-7b",
              "speakers": [{"n": 1, "sample_id": "a" * 16, "title": "Ada"},
                           {"n": 2, "sample_id": "b" * 16, "title": "Brook, take two"}],
              "max_new_tokens": None, "sections": 2, "render_seconds": 45.25}

    def test_the_prompt_then_one_line_of_parameters(self):
        entry = box.add_output(tone_pcm(2.0), 24000, "Take", "", dict(self.RENDER))
        assert entry["infotext"] == (
            "Speaker 1: Hello there.\n[pause]\nSpeaker 2: Hi.\n"
            'Steps: 10, CFG scale: 1.3, Seed: 1234, Model: vibevoice-7b, Speaker 1: Ada, '
            'Speaker 2: "Brook, take two", Sections: 2, Length: 2.0 s, Render time: 45.2 s')

    def test_an_output_from_before_seeds_were_recorded_says_nothing_of_one(self):
        entry = box.add_output(tone_pcm(0.5), 24000, "Old", "",
                               {"steps": 10, "cfg_scale": 1.3, "seed": None,
                                "max_new_tokens": 900, "prompt": "Hi"})
        assert entry["infotext"] == ("Hi\nSteps: 10, CFG scale: 1.3, Max new tokens: 900, "
                                     "Length: 0.5 s")

    def test_every_output_handed_out_carries_its_format_and_infotext_and_none_is_stored(self):
        entry = box.add_output(tone_pcm(0.5), 24000, "Take", "", dict(self.RENDER))
        assert entry["format"] == "wav"
        for found in (box.output(entry["id"]), box.outputs()[0],
                      box.rename_output(entry["id"], "Final"), box.set_loop(entry["id"], True)):
            assert found["format"] == "wav" and found["infotext"] == entry["infotext"]
        stored = json.loads((box._outputs_root() / entry["id"] / box.META_FILENAME)
                            .read_text("utf-8"))
        assert "infotext" not in stored and "format" not in stored
        assert stored["name"] == "Final" and stored["loop"] is True


def _info_chunk(data: bytes) -> dict:
    """The ``INFO`` fields of a RIFF file, read chunk by chunk."""
    found, at = {}, 12
    while at + 8 <= len(data):
        kind, size = data[at:at + 4], struct.unpack("<I", data[at + 4:at + 8])[0]
        body = data[at + 8:at + 8 + size]
        if kind == b"LIST" and body[:4] == b"INFO":
            inner = 4
            while inner + 8 <= len(body):
                key = body[inner:inner + 4].decode("ascii")
                length = struct.unpack("<I", body[inner + 4:inner + 8])[0]
                found[key] = body[inner + 8:inner + 8 + length].rstrip(b"\x00").decode("utf-8")
                inner += 8 + length + (length % 2)
        at += 8 + size + (size % 2)
    assert at == len(data), "the RIFF size and the chunks agree"
    return found


class TestTheFiles:
    def test_an_output_is_an_mp3_carrying_its_infotext_when_one_can_be_made(self, monkeypatch,
                                                                             tmp_path):
        asked = []

        def encoder(pcm16, rate, tags):
            asked.append((len(pcm16), rate, dict(tags)))
            return b"ID3-the-encoded-render"

        monkeypatch.setattr(box, "_encode_mp3", encoder)
        entry = box.add_output(tone_pcm(1.0), 24000, "Take", "", dict(TestInfotext.RENDER))

        assert entry["format"] == "mp3"
        assert box.output_audio(entry["id"]) == (b"ID3-the-encoded-render", "mp3")
        assert not (box._outputs_root() / entry["id"] / box.AUDIO_FILENAME).exists()
        (size, rate, tags), = asked
        assert size == len(tone_pcm(1.0)) and rate == 24000
        assert tags["comment"] == entry["infotext"] and "title" not in tags, \
            "a rename would leave a title saying the old name"
        assert json.loads(tags["voicebox"]) == {"id": entry["id"], "render": entry["render"]}

        box.set_settings({"save_folder": str(tmp_path)})
        saved = box.save_output(entry["id"])
        assert saved.endswith("Take.mp3")
        assert (tmp_path / "Take.mp3").read_bytes() == b"ID3-the-encoded-render"
        sidecar = json.loads((tmp_path / "Take.json").read_text("utf-8"))
        assert sidecar["infotext"] == entry["infotext"] and sidecar["format"] == "mp3"

    def test_a_wav_carries_the_infotext_in_its_info_chunk_and_reads_as_before(self):
        entry = box.add_output(tone_pcm(1.0), 24000, "Take", "", dict(TestInfotext.RENDER))
        data, kind = box.output_audio(entry["id"])

        assert kind == "wav" and data[:4] == b"RIFF" and data[8:12] == b"WAVE"
        assert struct.unpack("<I", data[4:8])[0] == len(data) - 8
        info = _info_chunk(data)
        assert info == {"ICMT": entry["infotext"], "ISFT": box.SOFTWARE}
        pcm, rate = box.pcm_of(data)
        assert rate == 24000 and pcm == tone_pcm(1.0)

    def test_a_render_from_before_mp3s_is_listed_played_and_saved_as_a_wav(self, tmp_path):
        identifier = "0123456789abcdef"
        folder = box._outputs_root() / identifier
        folder.mkdir(parents=True)
        (folder / box.AUDIO_FILENAME).write_bytes(box.wav_bytes(tone_pcm(0.5)))
        (folder / box.META_FILENAME).write_text(json.dumps({
            "id": identifier, "name": "Old take", "pipeline_id": "", "seconds": 0.5,
            "rate": 24000, "peaks": [0.1], "loop": False, "created": 1.0,
            "render": {"steps": 10, "cfg_scale": 1.3, "seed": None, "prompt": "Hi"}}), "utf-8")

        listed = box.outputs()
        assert [entry["id"] for entry in listed] == [identifier]
        assert listed[0]["format"] == "wav" and listed[0]["infotext"].startswith("Hi\nSteps: 10")
        assert box.output_audio(identifier)[1] == "wav"
        box.set_settings({"save_folder": str(tmp_path)})
        assert box.save_output(identifier).endswith("Old take.wav")

    def test_a_name_taken_by_another_formats_sidecar_is_not_overwritten(self, monkeypatch,
                                                                         tmp_path):
        box.set_settings({"save_folder": str(tmp_path)})
        monkeypatch.setattr(box, "_encode_mp3", lambda pcm16, rate, tags: b"ID3")
        first = box.save_output(box.add_output(tone_pcm(0.5), 24000, "Take", "", {})["id"])
        monkeypatch.setattr(box, "_encode_mp3", no_mp3)
        second = box.save_output(box.add_output(tone_pcm(0.5), 24000, "Take", "", {})["id"])
        assert first.endswith("Take.mp3") and second.endswith("Take (2).wav")
        assert json.loads((tmp_path / "Take.json").read_text("utf-8"))["format"] == "mp3"

    def test_the_card_is_handed_back_before_the_file_is_made(self, ready, monkeypatch):
        turns = box.turns()
        seen = []

        def encoder(pcm16, rate, tags):
            seen.append(list(turns.turn.finished))
            return None

        monkeypatch.setattr(box, "_encode_mp3", encoder)
        found = box.render(ready["pipeline"]["id"], "Speaker 1: Hi.",
                           ready["configuration"]["id"])
        assert settled(found["id"])["phase"] == "done"
        assert seen == [[None]], "the turn was finished before the encoder ran"

    @pytest.mark.skipif(not __import__("importlib").util.find_spec("av"),
                        reason="PyAV is not installed here (Forge Neo ships it)")
    def test_the_real_encoder_makes_an_mp3_that_decodes_to_every_sample(self):
        import av
        import numpy

        pcm = tone_pcm(3.0)
        text = "Speaker 1: Hi, there.\nSteps: 10, Seed: 7"
        encoded = REAL_MP3(pcm, 24000, {"comment": text, "encoded_by": box.SOFTWARE,
                                        "voicebox": json.dumps({"seed": 7})})
        assert encoded is not None and encoded[:3] == b"ID3"
        assert len(encoded) < len(box.wav_bytes(pcm)) / 2.5

        with av.open(io.BytesIO(encoded)) as container:
            stream = container.streams.audio[0]
            assert (container.format.name, stream.rate) == ("mp3", 24000)
            assert stream.codec_context.layout.name == "mono"
            assert container.metadata["comment"] == text
            assert json.loads(container.metadata["voicebox"]) == {"seed": 7}
            decoded = numpy.concatenate([frame.to_ndarray().reshape(-1)
                                         for frame in container.decode(stream)])
        assert decoded.size == len(pcm) // 2, "the encoder's delay and padding are undone"


class TestTheQueue:
    def test_clear_withdraws_every_queued_job_and_leaves_the_running_one(self, ready):
        turns = box.turns()
        gate = threading.Event()

        class Held(FakeTurn):
            def wait(self, timeout=None, cancelled=None):
                gate.set()
                if cancelled is not None and cancelled.is_set():
                    self.cancel()
                    return "cancelled"
                time.sleep(0.005)
                return "clearing"

        turns.turn = Held()
        pipeline, configuration = ready["pipeline"]["id"], ready["configuration"]["id"]
        first = box.render(pipeline, "Speaker 1: One", configuration)
        second = box.render(pipeline, "Speaker 1: Two", configuration)
        third = box.render(pipeline, "Speaker 1: Three", configuration)
        assert gate.wait(2.0)

        box.cancel_job(third["id"])
        assert box.clear_queue() == 1, "a job withdrawn already is not counted again"
        assert box.job(second["id"])["phase"] == "cancelled"
        assert box.job(third["id"])["phase"] == "cancelled"
        assert box.job(first["id"])["phase"] == "waiting", "Cancel is for the running one"
        assert box.clear_queue() == 0

        box.cancel_job(first["id"])
        assert settled(first["id"])["phase"] == "cancelled"
        assert len(turns.requests) == 1, "a withdrawn job never asks for the card"

    def test_a_job_says_how_long_it_has_run_by_the_servers_clock(self, ready, monkeypatch):
        clock = [1000.0]
        monkeypatch.setattr(box, "_now", lambda: clock[0])
        runtime = box._runtime()
        runtime.hold = threading.Event()
        found = box.render(ready["pipeline"]["id"], "Speaker 1: Hi.",
                           ready["configuration"]["id"])
        assert found["elapsed"] is None, "not started"

        phase_reached(found["id"], "rendering")
        clock[0] = 1042.5
        assert box.job(found["id"])["elapsed"] == 42.5
        runtime.hold.set()
        settled(found["id"])
        clock[0] = 2000.0
        assert box.job(found["id"])["elapsed"] == 42.5, "an ended job's time stops"

    def test_a_render_ending_while_the_next_is_named_leaves_no_two_one_name(self, ready,
                                                                             monkeypatch):
        runtime = box._runtime()
        runtime.hold = threading.Event()
        pipeline, configuration = ready["pipeline"]["id"], ready["configuration"]["id"]
        first = box.render(pipeline, "Speaker 1: One", configuration)
        phase_reached(first["id"], "rendering")
        real = box._service.pending

        def pending(pipeline_id):
            # The first render ends exactly while the second is being named.
            runtime.hold.set()
            settled(first["id"])
            return real(pipeline_id)

        monkeypatch.setattr(box._service, "pending", pending)
        second = box.render(pipeline, "Speaker 1: Two", configuration)
        assert (first["name"], second["name"]) == ("Trailer 1", "Trailer 2")
