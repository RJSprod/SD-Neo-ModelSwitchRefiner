"""The Voice Box with Breeze TTS 2 as its engine, with no card and no worker.

The render service and the files are the ones ``test_voice_box`` proves for
VibeVoice; what is asserted here is what a second engine adds or changes, with
the same doubles and a Breeze-shaped engine:

* what Breeze refuses at the press -- a speaker whose sample has no
  transcript, a slot that only describes a voice not yet designed;
* what a render hands the runtime -- each line's direction, each voice's
  transcript and standing direction, Breeze's own options;
* what an output records and how its infotext reads;
* voice design: a description becomes one take, which is kept as a sample with
  the sentence it spoke as its transcript;
* a sample's transcript, typed or heard by Whisper;
* configurations that keep each engine's options, and the engine switch.
"""

from __future__ import annotations

import io
import threading
import wave

import pytest

import mc_voice_box as box
import mc_voice_box_engines as engines
import mc_voice_breeze
from test_voice_box import (CARD, FakeRuntime, FakeTurns, Result, Take, _GB, no_mp3,
                            phase_reached, settled, spoken, tone_f32)


class FakeBreeze:
    """Breeze's adapter as the service reads it, with the real vocabulary."""

    ID = "breeze"
    LABEL = "Breeze TTS 2"
    GUEST = "Breeze"
    MAX_TAKES = 4
    FIELDS = mc_voice_breeze.FIELDS

    def __init__(self):
        self.refused = ""
        self.flash = True
        self.values = {"card_uuid": "", "model_id": "breeze-tts-2", "fast_path": False,
                       "keep_warm": True}

    def capabilities(self):
        return mc_voice_breeze.capabilities()

    def refuse(self, chosen):
        if str((chosen or {}).get("attention") or "") == "flash_attention_2" \
                and not self.flash:
            return "Flash attention 2 is not in Breeze's runtime yet."
        return ""

    def refusal(self, manual=False):
        return self.refused

    def model_ids(self):
        return ("breeze-tts-2",)

    def model_id(self):
        return "breeze-tts-2"

    def need_vram_bytes(self, identifier="", takes=1):
        return 9 * _GB

    def need_ram_bytes(self, identifier=""):
        return 8 * _GB

    def settings(self):
        return dict(self.values)

    def set_settings(self, values):
        self.values.update(values)
        return dict(self.values)

    def public_status(self):
        return {"id": "breeze", "ready": True, "message": "", "parts": []}

    def progress(self):
        return {}


class LineRuntime(FakeRuntime):
    """A runtime that reports Breeze's progress -- the line and the take -- and can hold."""

    def __init__(self, seconds: float = 1.0):
        super().__init__(seconds)
        self.started = threading.Event()

    def render(self, card, job, on_progress=None):
        self.renders.append(job)
        if on_progress is not None:
            on_progress({"seconds": 0.5, "line": 2, "lines": len(job.line_list()), "take": 1,
                         "takes": int(job.takes or 1)})
        self.started.set()
        if self.hold is not None:
            self.hold.wait(5.0)
        count = int(job.takes or 1)
        return Result([Take(tone_f32(self.seconds), self.seconds,
                            None if job.seed is None else job.seed + take)
                       for take in range(count)])


@pytest.fixture(autouse=True)
def _fresh(voice_root, monkeypatch):
    monkeypatch.setattr(box, "_encode_mp3", no_mp3)
    box.forget()
    yield
    box.forget()


@pytest.fixture
def breeze():
    turns, runtime, engine = FakeTurns(), LineRuntime(), FakeBreeze()
    box.use_turns(turns)
    box.use_runtime(runtime)
    box.use_engine(engine)
    box.set_settings({"engine": "breeze"})
    first = box.add_sample(spoken(), "Ada", transcript="  This is   Ada speaking. ")
    second = box.add_sample(spoken(5.0), "Brook", transcript="And this is Brook.")
    configuration = box.save_configuration({
        "name": "Studio", "card_uuid": CARD, "model_id": "breeze-tts-2",
        "speakers": {"1": first["id"], "2": second["id"]},
        "slots": {"2": {"instruction": "calm and slow"}},
        "options": {"breeze": {"cfg_scale": 3.5}}})
    found = box.new_pipeline("Drama")
    return {"turns": turns, "runtime": runtime, "engine": engine, "pipeline": found,
            "configuration": configuration, "samples": (first, second)}


class TestWhatBreezeRefusesAtThePress:
    def test_a_sample_without_a_transcript_is_refused_with_where_to_fix_it(self, breeze):
        silent = box.add_sample(spoken(), "Cal")
        box.save_configuration({"id": breeze["configuration"]["id"],
                                "speakers": {"1": silent["id"]}})
        with pytest.raises(box.VoiceBoxError) as raised:
            box.render(breeze["pipeline"]["id"], "Speaker 1: Hi.",
                       breeze["configuration"]["id"])
        assert str(raised.value) == ("Speaker 1's sample “Cal” has no transcript — Breeze TTS 2 "
                                     "clones a voice only from its exact words. Add one in the "
                                     "sample's row.")
        assert breeze["turns"].requests == [] and box.jobs() == []

    def test_vibevoice_needs_no_transcript(self, breeze):
        silent = box.add_sample(spoken(), "Cal")
        box.save_configuration({"id": breeze["configuration"]["id"],
                                "speakers": {"1": silent["id"]}})
        box.set_settings({"engine": "vibevoice"})

        class VibeVoice(FakeBreeze):
            ID, LABEL, GUEST = "vibevoice", "VibeVoice", "VibeVoice"

            def capabilities(self):
                import mc_voice_vibevoice

                return mc_voice_vibevoice.capabilities()

            def model_ids(self):
                return ("vibevoice-7b",)

        box.use_engine(VibeVoice())
        found = box.render(breeze["pipeline"]["id"], "Speaker 1: Hi.",
                           breeze["configuration"]["id"])
        assert settled(found["id"])["phase"] == "done"

    def test_a_described_slot_with_no_voice_says_design_it_first(self, breeze):
        box.save_configuration({"id": breeze["configuration"]["id"],
                                "speakers": {"1": breeze["samples"][0]["id"]},
                                "slots": {"3": {"description": "An old sailor."}}})
        with pytest.raises(box.VoiceBoxError, match="Speaker 3 has a description but no voice "
                                                    "yet — Design it and Keep it first."):
            box.render(breeze["pipeline"]["id"], "Speaker 3: Ahoy.",
                       breeze["configuration"]["id"])

    def test_flash_attention_is_refused_until_the_runtime_has_it(self, breeze):
        breeze["engine"].flash = False
        with pytest.raises(box.VoiceBoxError, match="Flash attention 2"):
            box.render(breeze["pipeline"]["id"], "Speaker 1: Hi.",
                       breeze["configuration"]["id"])
        box.save_configuration({"id": breeze["configuration"]["id"],
                                "options": {"breeze": {"attention": "sdpa"}}})
        found = box.render(breeze["pipeline"]["id"], "Speaker 1: Hi.",
                           breeze["configuration"]["id"])
        assert settled(found["id"])["phase"] == "done"


class TestWhatARenderHandsTheRuntime:
    def test_lines_carry_their_directions_and_voices_their_words(self, breeze):
        found = box.render(breeze["pipeline"]["id"],
                           "Speaker 1 (whispering): Come closer.\n"
                           "Speaker 2: I am here.\n"
                           "Speaker 1 ( shouting ,  angry ): Now!",
                           breeze["configuration"]["id"])
        assert settled(found["id"])["phase"] == "done"
        job, = breeze["runtime"].renders
        assert [(line.speaker, line.text, line.instruction) for line in job.line_list()] == \
            [(1, "Come closer.", "whispering"), (2, "I am here.", ""),
             (1, "Now!", "shouting , angry")]
        voices = job.voice_map()
        assert (voices[1].transcript, voices[1].instruction, voices[1].title) == \
            ("This is Ada speaking.", "", "Ada")
        assert (voices[2].transcript, voices[2].instruction) == \
            ("And this is Brook.", "calm and slow")
        assert job.options == {"cfg_scale": 3.5, "attention": "flash_attention_2",
                               "max_new_tokens": None, "line_gap_ms": 150, "sampling": True,
                               "temperature": 0.9, "top_p": 1.0}
        assert breeze["runtime"].load_options == [job.options], \
            "the load is told the attention the render uses"
        request, = breeze["turns"].requests
        assert (request["guest"], request["need_vram"], request["label"]) == \
            ("Breeze", 9 * _GB, "Breeze TTS 2 — Drama 1")

    def test_vibevoice_is_handed_the_same_lines_and_drops_the_directions(self):
        import mc_voice_vibevoice_runtime as runtime

        job = runtime.RenderJob(id="j", lines=[box.render_types.Line(1, "Hi.", "whispering")],
                                voices={1: (b"\x00" * 8, 24000)}, options={"steps": 12})
        header, _payload = runtime._render_frame(job)
        assert header["script"] == [{"speaker": 1, "text": "Hi."}]

    def test_the_jobs_progress_names_the_line_and_the_take(self, breeze):
        breeze["runtime"].hold = threading.Event()
        found = box.render(breeze["pipeline"]["id"], "Speaker 1: One.\nSpeaker 2: Two.",
                           breeze["configuration"]["id"])
        assert breeze["runtime"].started.wait(5.0)
        progress = phase_reached(found["id"], "rendering")["progress"]
        assert (progress["line"], progress["lines"], progress["take"], progress["takes"]) == \
            (2, 2, 1, 1)
        breeze["runtime"].hold.set()
        assert settled(found["id"])["phase"] == "done"


class TestWhatAnOutputRecords:
    def test_the_record_names_the_engine_and_its_options_and_nothing_of_vibevoices(
            self, breeze):
        found = box.render(breeze["pipeline"]["id"], "Speaker 2: Hello.",
                           breeze["configuration"]["id"])
        entry = box.output(settled(found["id"])["output_id"])
        render = entry["render"]
        assert render["engine"] == "breeze" and render["options"]["cfg_scale"] == 3.5
        assert not set(box._flat_keys()) & set(render), "VibeVoice's flat keys are its own"
        configured = render["configuration"]
        assert configured["engine"] == "breeze"
        assert configured["options"] == {"breeze": render["options"]}
        assert configured["slots"] == {"2": {"instruction": "calm and slow",
                                             "description": ""}}
        assert entry["infotext"] == (
            f"Speaker 2: Hello.\nEngine: Breeze TTS 2, CFG scale: 3.5, Seed: {found['seed']}, "
            f"Temperature: 0.9, Top-p: 1, Attention: Flash attention 2, Model: breeze-tts-2, "
            f"Speaker 2: Brook, Speaker 2 direction: calm and slow, Line gap: 150 ms, "
            f"Length: 1.0 s, Render time: 2.5 s")

    def test_greedy_says_so_in_the_infotext(self, breeze):
        box.save_configuration({"id": breeze["configuration"]["id"],
                                "options": {"breeze": {"sampling": False}}})
        found = box.render(breeze["pipeline"]["id"], "Speaker 1: Hi.",
                           breeze["configuration"]["id"])
        entry = box.output(settled(found["id"])["output_id"])
        assert ", Sampling: greedy, " in entry["infotext"]
        assert "Temperature" not in entry["infotext"]
        job, = breeze["runtime"].renders
        assert (job.options["sampling"], job.options["temperature"], job.options["top_p"]) == \
            (False, None, None)


class TestVoiceDesign:
    def test_a_design_is_one_take_of_a_described_voice_kept_as_a_sample(self, breeze):
        breeze["runtime"].seconds = 4.0
        queued = box.design("A warm, low voice with a slight rasp.", seed=41,
                            configuration_id=breeze["configuration"]["id"])
        assert queued["kind"] == "design" and queued["engine"] == "breeze"
        job = settled(queued["id"])
        assert job["phase"] == "done" and job["design"]["seed"] == 41
        request, = breeze["runtime"].renders
        assert [(line.speaker, line.text) for line in request.line_list()] == \
            [(1, box.DESIGN_TEXT)]
        voice = request.voice_map()[1]
        assert (voice.pcm, voice.description) == (b"", "A warm, low voice with a slight rasp.")
        assert request.seed == 41 and request.takes == 1
        assert box.pipeline(breeze["pipeline"]["id"])["outputs"] == [], "a design is no output"

        audio = box.design_audio(queued["id"])
        with wave.open(io.BytesIO(audio), "rb") as handle:
            assert (handle.getframerate(), handle.getnchannels()) == (24000, 1)

        kept = box.keep_design(queued["id"], "Rasp")
        assert (kept["title"], kept["source"], kept["transcript"],
                kept["transcript_source"]) == ("Rasp", "design", box.DESIGN_TEXT, "design")
        assert kept["design"] == {"description": "A warm, low voice with a slight rasp.",
                                  "seed": 41, "cfg_scale": 3.5, "text": box.DESIGN_TEXT}
        assert box.sample(kept["id"])["transcript"] == box.DESIGN_TEXT

    def test_a_design_in_its_own_words_and_in_chinese(self, breeze):
        assert box.design_text("一位溫柔的老奶奶，說話很慢") == box.DESIGN_TEXT_ZH
        assert box.design_text("A gentle grandmother 一位") == box.DESIGN_TEXT
        queued = box.design("A pirate.", text="Ahoy there, sailor.",
                            configuration_id=breeze["configuration"]["id"])
        settled(queued["id"])
        assert breeze["runtime"].renders[0].line_list()[0].text == "Ahoy there, sailor."

    def test_a_designed_take_outside_the_sample_envelope_is_not_kept(self, breeze):
        queued = box.design("A pirate.", configuration_id=breeze["configuration"]["id"])
        assert settled(queued["id"])["phase"] == "done"
        with pytest.raises(box.VoiceBoxError, match="1.0 seconds long"):
            box.keep_design(queued["id"])
        assert box.samples() == list(breeze["samples"]) or len(box.samples()) == 2

    def test_a_design_that_has_not_finished_cannot_be_kept_or_heard(self, breeze):
        breeze["runtime"].hold = threading.Event()
        queued = box.design("A pirate.", configuration_id=breeze["configuration"]["id"])
        assert breeze["runtime"].started.wait(5.0)
        with pytest.raises(box.NotFound):
            box.keep_design(queued["id"])
        with pytest.raises(box.NotFound):
            box.design_audio(queued["id"])
        breeze["runtime"].hold.set()
        settled(queued["id"])
        with pytest.raises(box.NotFound):
            box.keep_design("0" * 16)

    def test_an_empty_or_long_description_and_an_engine_that_cannot_design(self, breeze):
        with pytest.raises(box.VoiceBoxError, match="Describe the voice"):
            box.design("  ", configuration_id=breeze["configuration"]["id"])
        with pytest.raises(box.VoiceBoxError, match="at most 1000"):
            box.design("x" * 1001, configuration_id=breeze["configuration"]["id"])

        class VibeVoice(FakeBreeze):
            ID, LABEL = "vibevoice", "VibeVoice"

            def capabilities(self):
                import mc_voice_vibevoice

                return mc_voice_vibevoice.capabilities()

        box.use_engine(VibeVoice())
        with pytest.raises(box.VoiceBoxError, match="VibeVoice does not design voices"):
            box.design("A pirate.", configuration_id=breeze["configuration"]["id"])


class TestTranscripts:
    def test_a_transcript_is_kept_cleaned_and_cleared(self, breeze):
        first = breeze["samples"][0]
        assert first["transcript"] == "This is Ada speaking."
        assert first["transcript_source"] == "typed"
        found = box.set_transcript(first["id"], "  Hello\n  there. ", source="whisper")
        assert (found["transcript"], found["transcript_source"]) == ("Hello there.", "whisper")
        assert box.sample(first["id"])["transcript"] == "Hello there."
        cleared = box.set_transcript(first["id"], "")
        assert (cleared["transcript"], cleared["transcript_source"]) == ("", "")
        with pytest.raises(box.VoiceBoxError, match="at most 2000"):
            box.set_transcript(first["id"], "word " * 500)

    def test_whisper_hears_the_sample_at_sixteen_kilohertz_and_nothing_is_saved(
            self, breeze, monkeypatch):
        import mc_voice_models
        import mc_voice_runtime

        heard = []
        monkeypatch.setattr(mc_voice_models, "status",
                            lambda: type("Ready", (), {"ready": True})())

        def transcribe(wav):
            with wave.open(io.BytesIO(wav), "rb") as handle:
                heard.append((handle.getframerate(), handle.getnframes()))
            return {"text": "  Hello   there. "}

        monkeypatch.setattr(mc_voice_runtime, "transcribe", transcribe)
        first = breeze["samples"][0]
        found = box.transcribe(first["id"])
        assert found == {"text": "Hello there.", "seconds": first["seconds"]}
        (rate, frames), = heard
        assert rate == 16000 and abs(frames - first["seconds"] * 16000) <= 2
        assert box.sample(first["id"])["transcript"] == "This is Ada speaking.", \
            "a reading is a draft for the person to check, never saved by itself"

    def test_nothing_heard_and_no_whisper_are_sentences(self, breeze, monkeypatch):
        import mc_voice_models
        import mc_voice_runtime

        monkeypatch.setattr(mc_voice_models, "status",
                            lambda: type("Missing", (), {"ready": False})())
        assert box.stt_status()["available"] is False
        with pytest.raises(box.Unavailable, match="Settings → Voice Chat"):
            box.transcribe(breeze["samples"][0]["id"])
        monkeypatch.setattr(mc_voice_models, "status",
                            lambda: type("Ready", (), {"ready": True})())
        monkeypatch.setattr(mc_voice_runtime, "transcribe", lambda wav: {"text": "[BLANK_AUDIO]"})
        with pytest.raises(box.VoiceBoxError, match="heard no words"):
            box.transcribe(breeze["samples"][0]["id"])


class TestConfigurationsKeepEveryEnginesOptions:
    def test_each_engine_keeps_its_own_and_a_save_changes_only_what_it_names(self, breeze):
        found = box.configuration(breeze["configuration"]["id"])
        assert found["engines_saved"] == ["breeze"] and found["engine"] == "breeze"
        assert found["options"]["breeze"]["cfg_scale"] == 3.5
        assert found["options"]["vibevoice"]["steps"] == 12, "VibeVoice's at its defaults"
        saved = box.save_configuration({"id": found["id"],
                                        "options": {"vibevoice": {"steps": 20}}})
        assert saved["options"]["vibevoice"]["steps"] == 20 and saved["steps"] == 20
        assert saved["options"]["breeze"]["cfg_scale"] == 3.5, "Breeze's kept"
        assert saved["engines_saved"] == ["vibevoice", "breeze"]

    def test_a_configuration_from_before_engines_is_vibevoices(self, breeze):
        import json

        identifier = "0123456789abcdef"
        path = box._configuration_path(identifier)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"id": identifier, "name": "Old", "steps": 30,
                                    "cfg_scale": 2.0, "speakers": {}}), encoding="utf-8")
        found = box.configuration(identifier)
        assert found["engine"] == "vibevoice" and found["engines_saved"] == ["vibevoice"]
        assert found["options"]["vibevoice"]["steps"] == 30
        assert found["options"]["breeze"] == mc_voice_breeze.defaults()

    def test_an_option_breeze_does_not_take_is_refused_in_its_sentence(self, breeze):
        with pytest.raises(box.VoiceBoxError, match="CFG"):
            box.save_configuration({"id": breeze["configuration"]["id"],
                                    "options": {"breeze": {"cfg_scale": 12}}})
        with pytest.raises(box.VoiceBoxError, match="Line gap"):
            box.save_configuration({"id": breeze["configuration"]["id"],
                                    "options": {"breeze": {"line_gap_ms": 5000}}})


class TestSwitchingEngines:
    class Client:
        def __init__(self):
            self.unloaded = []

        def unload(self, card, reason, guest=""):
            self.unloaded.append((card, guest))
            return 0

    class Runtime:
        def __init__(self, cards):
            self.cards = cards
            self.shut = 0
            self.evicted = []

        def status(self, card=""):
            return {"cards": {key: {"uuid": uuid} for key, uuid in self.cards.items()}}

        def evict(self, card, reason):
            self.evicted.append(card)
            return 0

        def shutdown(self):
            self.shut += 1

    def test_a_switch_ends_the_outgoing_engines_workers_through_their_turns(self, monkeypatch):
        outgoing = self.Runtime({"aaaa": CARD})
        monkeypatch.setattr(engines, "runtime", lambda engine_id="": outgoing)
        modules = {"mc_voice_vibevoice_runtime": outgoing}
        real = engines.importlib.import_module
        monkeypatch.setattr(engines.importlib, "import_module",
                            lambda name: modules.get(name) or real(name))
        client = self.Client()
        assert engines.active_id() == "vibevoice"
        found = engines.select("breeze", client=client)
        assert found == {"changed": True, "engine": "breeze"}
        assert client.unloaded == [(CARD, "VibeVoice")]
        assert outgoing.shut == 1 and box.settings()["engine"] == "breeze"
        assert engines.select("breeze", client=client) == {"changed": False,
                                                            "engine": "breeze"}

    def test_a_switch_is_refused_while_anything_renders_and_an_unknown_engine_is_400(
            self, breeze):
        breeze["runtime"].hold = threading.Event()
        found = box.render(breeze["pipeline"]["id"], "Speaker 1: Hi.",
                           breeze["configuration"]["id"])
        assert breeze["runtime"].started.wait(5.0)
        with pytest.raises(engines.Busy) as raised:
            engines.select("vibevoice")
        assert raised.value.status == 409 and box.settings()["engine"] == "breeze"
        breeze["runtime"].hold.set()
        settled(found["id"])
        with pytest.raises(engines.EngineError) as unknown:
            engines.select("kokoro")
        assert unknown.value.status == 400

    def test_every_engine_is_listed_with_its_installation(self):
        found = engines.summary()
        assert found["active"] == "vibevoice"
        assert [row["id"] for row in found["available"]] == ["vibevoice", "breeze"]
        assert all({"label", "installed", "ready", "supported", "message"} <= set(row)
                   for row in found["available"])
        assert [label for label, _stop in engines.shutdowns()] == ["VibeVoice", "Breeze TTS 2"]

    def test_each_adapter_spells_its_rows_guest_and_label(self):
        for row in engines.SPECS:
            adapter = engines.adapter(row.id)
            assert (adapter.ID, adapter.GUEST, adapter.LABEL) == (row.id, row.guest, row.label)
