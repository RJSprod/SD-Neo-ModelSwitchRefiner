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
        self.identities: list = []

    def load(self, card, *, model_id="", precision="", lora_id="", lora_scale=1.0):
        self.loaded.append(card)
        self.identities.append({"model_id": model_id, "precision": precision,
                                "lora_id": lora_id, "lora_scale": lora_scale})
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


MODEL_7B = "vibevoice-7b"
MODEL_REALTIME = "vibevoice-realtime-0.5b"
PRESETS = [{"id": "en-Carter_man", "name": "Carter", "language": "en",
            "language_label": "English", "gender": "man", "experimental": False,
            "installed": True},
           {"id": "de-Spk0_man", "name": "Spk0", "language": "de", "language_label": "German",
            "gender": "man", "experimental": True, "installed": True}]


class VibeVoiceError(RuntimeError):
    pass


class FakeEngine:
    LABEL = "VibeVoice 7B"
    GUEST = "VibeVoice"
    MODEL_DEFAULT = MODEL_7B
    VibeVoiceError = VibeVoiceError

    def __init__(self):
        self.refused = ""
        self.refused_for: list = []
        self.peaks: list = []
        self.values = {"steps": 10, "cfg_scale": 1.3, "seed": None, "max_new_tokens": None}
        self.library = {"0123456789abcdef": {"id": "0123456789abcdef", "name": "Narrator",
                                             "base": MODEL_7B, "bytes": 1024,
                                             "parts": ["llm"], "created": 1.0}}
        self.installed: list = []

    def model_info(self, identifier=""):
        if identifier == MODEL_REALTIME:
            return {"id": MODEL_REALTIME, "label": "VibeVoice Realtime 0.5B",
                    "kind": "realtime", "installed": True, "precisions": ["bf16"],
                    "lora": False, "max_speakers": 1, "voices": "presets",
                    "defaults": {"steps": 5, "cfg_scale": 1.5}, "presets": list(PRESETS)}
        if identifier not in ("", MODEL_7B):
            raise VibeVoiceError("That model is not one this build has.")
        return {"id": MODEL_7B, "label": "VibeVoice 7B", "kind": "longform",
                "installed": True, "precisions": ["bf16", "int8", "nf4"], "lora": True,
                "max_speakers": 4, "voices": "samples",
                "defaults": {"steps": 10, "cfg_scale": 1.3}, "presets": []}

    def models_info(self):
        return [self.model_info(MODEL_7B), self.model_info(MODEL_REALTIME)]

    def loras(self):
        return list(self.library.values())

    def add_lora(self, folder, name=""):
        if "bad" in folder:
            raise VibeVoiceError("That folder holds no LoRA adapter for the language model.")
        entry = {"id": "fedcba9876543210", "name": name or "Added", "base": MODEL_7B,
                 "bytes": 2048, "parts": ["llm"], "created": 2.0}
        self.library[entry["id"]] = entry
        return entry

    def rename_lora(self, identifier, name):
        if identifier not in self.library:
            raise VibeVoiceError("That LoRA is no longer in the library.")
        self.library[identifier]["name"] = name
        return self.library[identifier]

    def delete_lora(self, identifier):
        if identifier not in self.library:
            raise VibeVoiceError("That LoRA is no longer in the library.")
        return {"deleted": self.library.pop(identifier)["id"]}

    def refusal(self, manual=False, model_id=""):
        self.refused_for.append(model_id)
        return self.refused

    def need_vram_bytes(self, identifier="", precision=""):
        if identifier == MODEL_REALTIME:
            return 3 * _GB
        return {"int8": 13, "nf4": 9}.get(precision, 20) * _GB

    def need_ram_bytes(self, identifier="", precision=""):
        return 3 * _GB

    def note_peak(self, identifier, peak_bytes, rss_bytes=0, precision=""):
        self.peaks.append((identifier, peak_bytes, precision))

    def settings(self):
        return dict(self.values)

    def set_settings(self, values):
        self.values.update(values)
        return dict(self.values)

    def public_status(self):
        return {"ready": True, "message": "", "parts": []}

    def progress(self):
        return {}

    def install(self, part="", model_id=""):
        self.installed.append((part, model_id))
        return None

    def install_from(self, part, folder, model_id=""):
        self.installed.append((part, folder, model_id))
        return None


@pytest.fixture(autouse=True)
def _fresh(voice_root):
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
            box.output_wav("")


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
        pcm, rate = box.pcm_of(box.output_wav(entry["id"]))
        assert rate == 24000 and len(pcm) == len(tone_pcm(2.0))

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
        assert engine.peaks == [], "the runtime notes every render's peak itself; a second note " \
                                   "from the Voice Box counted each render twice"
        assert runtime.identities == [{"model_id": "vibevoice-7b", "precision": "bf16",
                                       "lora_id": "", "lora_scale": 1.0}]
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


class TestModelsPrecisionAndLoRA:
    """Phase 4 in a configuration: which model, how its weights are held, which
    fine-tune on top, and -- for the Realtime model -- which of its own voices."""

    def test_a_configuration_keeps_the_precision_and_the_lora_the_model_takes(self, engine):
        found = box.save_configuration({"model_id": MODEL_7B, "precision": "nf4",
                                        "lora_id": "0123456789abcdef", "lora_scale": 0.75})
        assert (found["precision"], found["lora_id"], found["lora_scale"]) == \
            ("nf4", "0123456789abcdef", 0.75)

        for bad, words in (({"precision": "fp8"}, "cannot run at that precision"),
                           ({"lora_id": "ffffffffffffffff"}, "no longer in the library"),
                           ({"lora_scale": 2.5}, "strength"),
                           ({"model_id": "vibevoice-9b"}, "not one this Voice Box knows")):
            with pytest.raises(box.VoiceBoxError, match=words):
                box.save_configuration(dict(bad, id=found["id"]))

    def test_the_realtime_model_takes_neither_a_precision_nor_a_lora(self, engine):
        with pytest.raises(box.VoiceBoxError, match="cannot run at that precision"):
            box.save_configuration({"model_id": MODEL_REALTIME, "precision": "nf4"})
        with pytest.raises(box.VoiceBoxError, match="does not take a LoRA"):
            box.save_configuration({"model_id": MODEL_REALTIME,
                                    "lora_id": "0123456789abcdef"})

    def test_speaker_slots_follow_the_model(self, engine):
        sample = box.add_sample(spoken(), "Ada")
        mixed = {"1": "preset:en-Carter_man", "2": sample["id"], "3": "preset:nope"}

        realtime = box.save_configuration({"model_id": MODEL_REALTIME, "speakers": mixed})
        longform = box.save_configuration({"model_id": MODEL_7B, "speakers": mixed})

        assert realtime["speakers"] == {"1": "preset:en-Carter_man"}
        assert longform["speakers"] == {"2": sample["id"]}
        unknown = box.save_configuration({"model_id": MODEL_REALTIME,
                                          "speakers": {"1": "preset:en-Nobody_man"}})
        assert unknown["speakers"] == {}, "a preset the model does not have is dropped"

    def test_a_realtime_render_speaks_with_its_preset_and_asks_for_its_own_room(
            self, turns, runtime, engine):
        owner = box.new_pipeline("Realtime")
        chosen = box.save_configuration({"model_id": MODEL_REALTIME, "card_uuid": CARD,
                                         "steps": 5, "cfg_scale": 1.5,
                                         "speakers": {"1": "preset:en-Carter_man"}})
        found = box.render(owner["id"], "Hello there.\n[pause]\nAnd again.", chosen["id"])

        job = settled(found["id"])
        assert job["phase"] == "done", job
        assert turns.requests[0]["need_vram"] == 3 * _GB
        assert runtime.identities[0]["model_id"] == MODEL_REALTIME
        assert [request.voices for request in runtime.renders] == [
            {1: {"preset": "en-Carter_man"}}, {1: {"preset": "en-Carter_man"}}]
        meta = box.output(job["output_id"])["render"]
        assert meta["model"] == "VibeVoice Realtime 0.5B"
        assert meta["speakers"] == [{"n": 1, "sample_id": "", "title": "Carter",
                                     "preset": "en-Carter_man"}]
        assert engine.refused_for[-1] == MODEL_REALTIME

    def test_a_realtime_script_with_two_speakers_is_refused_before_any_card(
            self, turns, runtime, engine):
        owner = box.new_pipeline("Realtime")
        chosen = box.save_configuration({"model_id": MODEL_REALTIME, "card_uuid": CARD,
                                         "speakers": {"1": "preset:en-Carter_man"}})
        with pytest.raises(box.VoiceBoxError, match="speaks with one voice.*Speaker 2"):
            box.render(owner["id"], "Speaker 1: Hi.\nSpeaker 2: Hello.", chosen["id"])
        empty = box.save_configuration({"model_id": MODEL_REALTIME, "card_uuid": CARD})
        with pytest.raises(box.VoiceBoxError, match="Choose a voice for Speaker 1"):
            box.render(owner["id"], "Hi there, friend.", empty["id"])
        assert turns.requests == []

    def test_a_quantised_render_with_a_lora_loads_that_identity_and_sizes_its_turn(
            self, turns, runtime, engine):
        sample = box.add_sample(spoken(), "Ada")
        owner = box.new_pipeline("Quantised")
        chosen = box.save_configuration({"model_id": MODEL_7B, "card_uuid": CARD,
                                         "precision": "nf4", "lora_id": "0123456789abcdef",
                                         "lora_scale": 0.5, "speakers": {"1": sample["id"]}})
        job = settled(box.render(owner["id"], "Speaker 1: Hello.", chosen["id"])["id"])

        assert job["phase"] == "done"
        assert turns.requests[0]["need_vram"] == 9 * _GB
        assert runtime.identities == [{"model_id": MODEL_7B, "precision": "nf4",
                                       "lora_id": "0123456789abcdef", "lora_scale": 0.5}]
        meta = box.output(job["output_id"])["render"]
        assert meta["precision"] == "nf4"
        assert meta["lora"] == {"id": "0123456789abcdef", "name": "Narrator", "scale": 0.5}
        assert meta["peak_bytes"] == 19 * _GB and engine.peaks == []

    def test_deleting_a_lora_takes_it_out_of_the_configurations_that_used_it(self, engine):
        using = box.save_configuration({"name": "With", "model_id": MODEL_7B, "precision": "int8",
                                        "lora_id": "0123456789abcdef", "lora_scale": 0.5})
        added = engine.add_lora("C:/loras/warm", "Warm")
        other = box.save_configuration({"name": "Another", "model_id": MODEL_7B,
                                        "lora_id": added["id"], "lora_scale": 1.5})
        engine.delete_lora("0123456789abcdef")
        with pytest.raises(box.VoiceBoxError, match="no longer in the library"):
            box.save_configuration({"id": using["id"], "name": "With"})

        assert box.forget_lora("0123456789abcdef") == [using["id"]]

        kept = box.configuration(using["id"])
        assert (kept["lora_id"], kept["lora_scale"], kept["precision"]) == ("", 1.0, "int8")
        assert box.configuration(other["id"]) == other
        assert box.save_configuration({"id": using["id"], "name": "With"})["lora_id"] == ""
        assert box.forget_lora("") == []

    def test_an_engine_that_cannot_describe_its_models_renders_as_before(self, turns, runtime):
        """A configuration saved before models could be described: the 7B's shape."""
        class Plain:
            LABEL = "VibeVoice"

            def refusal(self, manual=False):
                return ""

            def need_vram_bytes(self, identifier="", precision=""):
                return 20 * _GB

            def need_ram_bytes(self, identifier="", precision=""):
                return 3 * _GB

            def note_peak(self, *args):
                pass

        box.use_engine(Plain())
        sample = box.add_sample(spoken(), "Ada")
        owner = box.new_pipeline("Plain")
        chosen = box.save_configuration({"card_uuid": CARD, "speakers": {"1": sample["id"]}})
        assert chosen["precision"] == "bf16" and chosen["lora_id"] == ""
        assert settled(box.render(owner["id"], "Hi there.", chosen["id"])["id"])["phase"] == "done"


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
