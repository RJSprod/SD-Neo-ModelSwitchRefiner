"""The Voice Box's routes: the page token, the shapes, the refusals, the audio.

The engine and the runtime are doubles (``test_voice_box``'s), the turn client
too, so every route is exercised end to end through FastAPI's test client
without a card: what a page receives, with what status, and that nothing leaves
the WebUI without the token.
"""

from __future__ import annotations

import json
import urllib.parse

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import mc_voice_api  # noqa: E402
import mc_voice_box as box  # noqa: E402
import mc_voice_box_api as api  # noqa: E402
import mc_voice_vibevoice  # noqa: E402
from test_voice_box import (CARD, FakeEngine, FakeRuntime, FakeTurn, FakeTurns, no_mp3,  # noqa: E402
                            spoken, settled)


@pytest.fixture(autouse=True)
def _fresh(voice_root, monkeypatch):
    monkeypatch.setattr(box, "_encode_mp3", no_mp3)
    box.forget()
    engine, runtime, turns = FakeEngine(), FakeRuntime(), FakeTurns()
    box.use_engine(engine)
    box.use_runtime(runtime)
    box.use_turns(turns)
    monkeypatch.setattr(api, "_engine", lambda: engine)
    monkeypatch.setattr(api, "_runtime", lambda: runtime)
    mc_voice_api.forget_repeats()
    yield
    box.forget()
    mc_voice_api.forget_repeats()


@pytest.fixture
def app():
    built = FastAPI()
    assert api.install(None, built) is True
    return built


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture
def key():
    return {mc_voice_api.TOKEN_HEADER: mc_voice_api.session_token()}


def post(client, route, key, payload=None):
    return client.post(route, headers=key, content=json.dumps(payload or {}))


class TestRegistration:
    def test_every_route_is_registered_once(self, app):
        paths = [route.path for route in app.routes]
        for route in api.ROUTES:
            assert route in paths
        assert api.install(None, app) is True
        assert len([p for p in [r.path for r in app.routes] if p == api.STATUS_ROUTE]) == 1

    def test_the_audio_routes_are_gets_and_the_rest_posts(self, app):
        methods = {route.path: set(getattr(route, "methods", ())) for route in app.routes}
        assert methods[api.SAMPLE_AUDIO_ROUTE] == {"GET"}
        assert methods[api.OUTPUT_AUDIO_ROUTE] == {"GET"}
        assert methods[api.DESIGN_AUDIO_ROUTE] == {"GET"}
        assert {route for route in api.ROUTES if methods[route] == {"GET"}} \
            == set(api.AUDIO_ROUTES)
        assert methods[api.RENDER_ROUTE] == {"POST"}

    def test_no_app_means_no_routes_and_no_error(self):
        assert api.install(None, None) is False


class TestThePageToken:
    def test_every_route_refuses_without_the_token(self, client):
        for route in api.ROUTES:
            if route in api.AUDIO_ROUTES:
                answer = client.get(route, params={"id": "0" * 16})
            else:
                answer = client.post(route, content=b"{}")
            assert answer.status_code == 403, route
            assert "Reload" in answer.json()["error"]

    def test_a_foreign_origin_is_refused(self, client, key):
        answer = client.post(api.STATUS_ROUTE, content=b"{}",
                             headers=dict(key, origin="https://evil.example"))
        assert answer.status_code == 403


class TestStatus:
    def test_the_status_carries_every_part_the_page_draws(self, client, key):
        box.set_settings({"card_uuid": CARD})
        answer = post(client, api.STATUS_ROUTE, key)

        assert answer.status_code == 200
        found = answer.json()
        assert found["ok"] is True and found["engine"]["ready"] is True
        assert found["settings"]["card_uuid"] == CARD
        # The engine's own default, which handoff 38 moved from ten to twelve.
        assert found["engine_settings"]["steps"] == mc_voice_vibevoice.STEPS_DEFAULT == 12
        assert found["cards"][0]["uuid"] == CARD and found["cards"][0]["wangp_card"] is True
        assert found["turns"] == [{"card": "the card", "uuid": CARD}]
        assert found["runtime"] == {"cards": {}, "last_error": ""} and found["jobs"] == []

    def test_a_part_that_fails_is_reported_and_the_rest_still_answers(self, client, key,
                                                                       monkeypatch):
        class Broken(FakeEngine):
            def public_status(self):
                raise RuntimeError("no manifest")

        monkeypatch.setattr(api, "_engine", lambda: Broken())
        found = post(client, api.STATUS_ROUTE, key).json()
        assert found["ok"] is True and found["engine"]["ready"] is False
        assert "no manifest" in found["engine"]["error"] and found["settings"]

    def test_settings_go_to_both_stores(self, client, key, tmp_path):
        answer = post(client, api.SETTINGS_ROUTE, key,
                      {"card_uuid": CARD, "steps": 12, "save_folder": str(tmp_path / "out")})
        found = answer.json()
        assert found["settings"]["card_uuid"] == CARD
        assert found["settings"]["save_folder"] == str(tmp_path / "out")
        assert found["engine_settings"]["steps"] == 12
        assert (tmp_path / "out").is_dir()

    def test_the_folder_dialog_says_when_it_cannot_open(self, client, key, monkeypatch):
        import mc_llm_native

        def refuse(title, initial=None):
            raise mc_llm_native.Unavailable("This WebUI is being served to other machines")

        monkeypatch.setattr(mc_llm_native, "choose_folder", refuse)
        answer = post(client, api.FOLDER_ROUTE, key)
        assert answer.status_code == 409 and "other machines" in answer.json()["error"]

    def test_an_install_starts_once(self, client, key, monkeypatch):
        started = []
        engine = api._engine()
        monkeypatch.setattr(engine, "install", lambda part="": started.append(part))
        assert post(client, api.INSTALL_ROUTE, key, {"part": "runtime"}).json() == {
            "ok": True, "already": False}
        import time

        deadline = time.monotonic() + 2.0
        while not started and time.monotonic() < deadline:
            time.sleep(0.01)
        assert started == ["runtime"]
        monkeypatch.setattr(engine, "progress", lambda: {"running": True})
        assert post(client, api.INSTALL_ROUTE, key, {}).json()["already"] is True


class TestSamplesOverTheWire:
    def test_a_selection_uploads_raw_with_its_title_in_a_header(self, client, key):
        title = urllib.parse.quote("Ada – take 1")
        answer = client.post(api.SAMPLE_UPLOAD_ROUTE, content=spoken(),
                             headers=dict(key, **{api.TITLE_HEADER: title,
                                                  api.SOURCE_HEADER: "microphone"}))

        assert answer.status_code == 200, answer.text
        sample = answer.json()["sample"]
        assert sample["title"] == "Ada – take 1" and sample["source"] == "microphone"
        listed = post(client, api.SAMPLES_ROUTE, key).json()["samples"]
        assert [entry["id"] for entry in listed] == [sample["id"]]

        audio = client.get(api.SAMPLE_AUDIO_ROUTE, params={"id": sample["id"]}, headers=key)
        assert audio.status_code == 200 and audio.headers["content-type"].startswith("audio/wav")
        assert audio.content[:4] == b"RIFF" and "attachment" not in audio.headers.get(
            "content-disposition", "")

    def test_a_bad_recording_is_refused_with_a_sentence_and_a_status(self, client, key):
        short = client.post(api.SAMPLE_UPLOAD_ROUTE, content=spoken(1.0), headers=key)
        assert short.status_code == 400 and "seconds long" in short.json()["error"]
        huge = client.post(api.SAMPLE_UPLOAD_ROUTE, content=b"x" * (api.MAX_UPLOAD_BYTES + 1),
                           headers=key)
        assert huge.status_code == 413

    def test_renaming_and_deleting_and_the_unknown(self, client, key):
        sample = client.post(api.SAMPLE_UPLOAD_ROUTE, content=spoken(), headers=key).json()["sample"]
        renamed = post(client, api.SAMPLE_RENAME_ROUTE, key, {"id": sample["id"], "title": "New"})
        assert renamed.json()["sample"]["title"] == "New"
        assert post(client, api.SAMPLE_DELETE_ROUTE, key, {"id": sample["id"]}).json()["deleted"]
        gone = client.get(api.SAMPLE_AUDIO_ROUTE, params={"id": sample["id"]}, headers=key)
        assert gone.status_code == 404 and "library" in gone.json()["error"]


class TestRenderingOverTheWire:
    @pytest.fixture
    def prepared(self, client, key):
        sample = client.post(api.SAMPLE_UPLOAD_ROUTE, content=spoken(), headers=key).json()["sample"]
        configuration = post(client, api.CONFIGURATION_SAVE_ROUTE, key, {
            "configuration": {"name": "Studio", "card_uuid": CARD,
                              "speakers": {"1": sample["id"]}}}).json()["configuration"]
        pipeline = post(client, api.PIPELINE_NEW_ROUTE, key, {"name": "Trailer"}).json()["pipeline"]
        return {"sample": sample, "configuration": configuration, "pipeline": pipeline}

    def test_a_render_is_queued_followed_and_its_output_served(self, client, key, prepared):
        answer = post(client, api.RENDER_ROUTE, key, {
            "pipeline_id": prepared["pipeline"]["id"], "prompt": "Speaker 1: Hello.",
            "configuration_id": prepared["configuration"]["id"], "name": "Hello"})
        assert answer.status_code == 200, answer.text
        job = answer.json()["job"]
        assert job["phase"] in ("queued", "waiting", "loading", "rendering", "done")

        done = settled(job["id"])
        listed = post(client, api.JOBS_ROUTE, key).json()["jobs"]
        assert listed[0]["id"] == job["id"] and listed[0]["phase"] == "done"
        outputs = post(client, api.OUTPUTS_ROUTE, key,
                       {"pipeline_id": prepared["pipeline"]["id"]}).json()["outputs"]
        assert [entry["id"] for entry in outputs] == [done["output_id"]]
        assert post(client, api.OUTPUT_LOOP_ROUTE, key,
                    {"id": done["output_id"], "loop": True}).json()["output"]["loop"] is True

        audio = client.get(api.OUTPUT_AUDIO_ROUTE, params={"id": done["output_id"], "download": 1},
                           headers=key)
        assert audio.status_code == 200 and audio.content[:4] == b"RIFF"
        assert audio.headers["content-disposition"].startswith('attachment; filename="render.wav"')
        assert "Hello.wav" in audio.headers["content-disposition"]

    def test_a_render_sent_from_another_tab_carries_its_origin_both_ways(self, client, key,
                                                                         prepared):
        """What LLM Studio sends, and what it reads back: the origin on the
        job the route answers with, and on the output listed afterwards."""
        origin = {"kind": "llm", "key": "thread-1:3:abcd1234", "label": "Message 4"}
        job = post(client, api.RENDER_ROUTE, key, {
            "pipeline_id": prepared["pipeline"]["id"], "prompt": "Speaker 1: Hello.",
            "configuration_id": prepared["configuration"]["id"], "name": "Message 4",
            "origin": dict(origin, junk="dropped")}).json()["job"]
        assert job["origin"] == dict(origin, engine="vibevoice"), "and which engine answered"
        done = settled(job["id"])
        entry, = post(client, api.OUTPUTS_ROUTE, key, {"pipeline_id": ""}).json()["outputs"]
        assert entry["id"] == done["output_id"] and entry["render"]["origin"] == job["origin"]

    def test_an_mp3_render_is_served_and_downloaded_as_one(self, client, key, prepared,
                                                           monkeypatch):
        monkeypatch.setattr(box, "_encode_mp3", lambda pcm16, rate, tags: b"ID3-an-mp3")
        job = post(client, api.RENDER_ROUTE, key, {
            "pipeline_id": prepared["pipeline"]["id"], "prompt": "Speaker 1: Hello.",
            "configuration_id": prepared["configuration"]["id"], "name": "Hello"}).json()["job"]
        assert isinstance(job["seed"], int) and job["seed_drawn"] is True
        done = settled(job["id"])
        entry, = post(client, api.OUTPUTS_ROUTE, key,
                      {"pipeline_id": prepared["pipeline"]["id"]}).json()["outputs"]
        assert entry["format"] == "mp3" and entry["render"]["seed"] == job["seed"]
        # The WebUI-shaped infotext as handoff 38 left it: the engine's twelve
        # steps and its solver before the CFG scale, the attention after the seed.
        assert entry["infotext"].startswith(
            f"Speaker 1: Hello.\nSteps: {mc_voice_vibevoice.STEPS_DEFAULT}, Solver: DPM++ 2M, "
            f"CFG scale: 1.3, Seed: {job['seed']}, Attention: SDPA")

        played = client.get(api.OUTPUT_AUDIO_ROUTE, params={"id": done["output_id"]}, headers=key)
        assert played.headers["content-type"] == "audio/mpeg" and played.content == b"ID3-an-mp3"
        audio = client.get(api.OUTPUT_AUDIO_ROUTE, params={"id": done["output_id"], "download": 1},
                           headers=key)
        assert audio.headers["content-disposition"].startswith(
            'attachment; filename="render.mp3"')
        assert audio.headers["content-disposition"].endswith("Hello.mp3")

    def test_clearing_the_queue_withdraws_what_waits_and_answers_the_jobs(self, client, key,
                                                                         prepared):
        class Held(FakeTurn):
            def wait(self, timeout=None, cancelled=None):
                if cancelled is not None and cancelled.is_set():
                    self.cancel()
                    return "cancelled"
                return "clearing"

        box.turns().turn = Held()
        body = {"pipeline_id": prepared["pipeline"]["id"], "prompt": "Speaker 1: Hi",
                "configuration_id": prepared["configuration"]["id"]}
        running = post(client, api.RENDER_ROUTE, key, body).json()["job"]
        queued = post(client, api.RENDER_ROUTE, key, body).json()["job"]
        import time

        deadline = time.monotonic() + 2.0
        while box.job(running["id"])["phase"] == "queued" and time.monotonic() < deadline:
            time.sleep(0.01)

        answer = post(client, api.JOBS_CLEAR_ROUTE, key).json()
        assert answer["ok"] is True and answer["cleared"] == 1
        phases = {entry["id"]: entry["phase"] for entry in answer["jobs"]}
        assert phases == {running["id"]: "waiting", queued["id"]: "cancelled"}
        post(client, api.JOB_CANCEL_ROUTE, key, {"id": running["id"]})
        assert settled(running["id"])["phase"] == "cancelled"

    def test_a_render_that_cannot_start_is_a_400_with_the_reason(self, client, key, prepared):
        answer = post(client, api.RENDER_ROUTE, key, {
            "pipeline_id": prepared["pipeline"]["id"], "prompt": "Speaker 2: Who?",
            "configuration_id": prepared["configuration"]["id"]})
        assert answer.status_code == 400 and "Speaker 2 has no sample" in answer.json()["error"]
        missing = post(client, api.RENDER_ROUTE, key, {"pipeline_id": "0" * 16, "prompt": "x"})
        assert missing.status_code == 404

    def test_saving_without_a_folder_is_the_409_the_page_acts_on(self, client, key, prepared,
                                                                 tmp_path):
        job = post(client, api.RENDER_ROUTE, key, {
            "pipeline_id": prepared["pipeline"]["id"], "prompt": "Speaker 1: Hi",
            "configuration_id": prepared["configuration"]["id"]}).json()["job"]
        done = settled(job["id"])
        refused = post(client, api.OUTPUT_SAVE_ROUTE, key, {"id": done["output_id"]})
        assert refused.status_code == 409 and "Choose a folder" in refused.json()["error"]

        post(client, api.SETTINGS_ROUTE, key, {"save_folder": str(tmp_path / "renders")})
        saved = post(client, api.OUTPUT_SAVE_ROUTE, key, {"id": done["output_id"]}).json()
        assert saved["path"].endswith(".wav") and saved["save_folder"] == str(tmp_path / "renders")

    def test_cancel_and_the_runtime_actions(self, client, key, prepared):
        job = post(client, api.RENDER_ROUTE, key, {
            "pipeline_id": prepared["pipeline"]["id"], "prompt": "Speaker 1: Hi",
            "configuration_id": prepared["configuration"]["id"]}).json()["job"]
        settled(job["id"])
        assert post(client, api.JOB_CANCEL_ROUTE, key, {"id": job["id"]}).json()["job"]["phase"] == "done"

        unloaded = post(client, api.RUNTIME_ROUTE, key, {"action": "unload", "card_uuid": CARD})
        assert unloaded.json()["freed"] == 18 * 1024**3
        assert box.turns().unloaded == [(CARD, "unloaded from the Voice Box")]
        assert post(client, api.RUNTIME_ROUTE, key, {"action": "stop"}).json()["ok"] is True
        assert post(client, api.RUNTIME_ROUTE, key, {"action": "dance"}).status_code == 400


class TestPromptsAndPipelinesOverTheWire:
    def test_prompts_round_trip(self, client, key):
        added = post(client, api.PROMPT_ADD_ROUTE, key, {"text": "Speaker 1: Hi"}).json()
        assert added["prompt"]["text"] == "Speaker 1: Hi" and len(added["history"]) == 1
        starred = post(client, api.PROMPT_FAVOURITE_ROUTE, key,
                       {"id": added["prompt"]["id"], "favourite": True}).json()
        assert len(starred["favourites"]) == 1
        assert post(client, api.PROMPTS_ROUTE, key).json()["favourites"][0]["favourite"] is True
        gone = post(client, api.PROMPT_DELETE_ROUTE, key, {"id": added["prompt"]["id"]}).json()
        assert gone["history"] == []

    def test_pipelines_round_trip(self, client, key):
        made = post(client, api.PIPELINE_NEW_ROUTE, key, {}).json()
        assert made["pipeline"]["name"] == "Pipeline 1"
        saved = post(client, api.PIPELINE_SAVE_ROUTE, key,
                     {"id": made["pipeline"]["id"], "prompt": "Speaker 1: x", "name": "Named"}).json()
        assert saved["pipeline"]["prompt"] == "Speaker 1: x" and saved["pipeline"]["name"] == "Named"
        assert post(client, api.PIPELINES_ROUTE, key).json()["pipelines"][0]["name"] == "Named"
        assert post(client, api.PIPELINE_DELETE_ROUTE, key,
                    {"id": made["pipeline"]["id"]}).json()["pipelines"] == []
        bad = post(client, api.CONFIGURATION_SAVE_ROUTE, key, {"configuration": {"steps": 99}})
        assert bad.status_code == 400 and "between 1 and 50" in bad.json()["error"]


class TestTheSecondEngineOverTheWire:
    """The routes Breeze brought: the engine switch, transcripts, and voice design."""

    def test_the_status_lists_the_engines_and_whisper(self, client, key, monkeypatch):
        import mc_voice_models

        monkeypatch.setattr(mc_voice_models, "status",
                            lambda: type("Missing", (), {"ready": False})())
        found = post(client, api.STATUS_ROUTE, key).json()
        assert found["engines"]["active"] == "vibevoice"
        assert [row["id"] for row in found["engines"]["available"]] == ["vibevoice", "breeze"]
        assert found["stt"]["available"] is False and "Voice Chat" in found["stt"]["message"]
        assert found["settings"]["engine"] == "vibevoice"

    def test_the_engine_is_switched_by_its_own_route_and_not_by_settings(self, client, key,
                                                                         monkeypatch):
        import mc_voice_box_engines as engines

        class Idle:
            def status(self, card=""):
                return {"cards": {}}

            def shutdown(self):
                pass

        real = engines.importlib.import_module
        monkeypatch.setattr(engines.importlib, "import_module",
                            lambda name: Idle() if name.endswith("_runtime") else real(name))
        refused = post(client, api.SETTINGS_ROUTE, key, {"engine": "breeze"}).json()
        assert refused["settings"]["engine"] == "vibevoice", "settings never switch the engine"
        answer = post(client, api.ENGINE_ROUTE, key, {"engine": "breeze"})
        assert answer.status_code == 200
        assert answer.json()["changed"] is True and answer.json()["settings"]["engine"] == \
            "breeze"
        unknown = post(client, api.ENGINE_ROUTE, key, {"engine": "kokoro"})
        assert unknown.status_code == 400 and "not an engine" in unknown.json()["error"]

    def test_a_switch_while_rendering_is_the_409_the_page_acts_on(self, client, key):
        first = box.add_sample(spoken(), "Ada")
        configuration = box.save_configuration({"name": "S", "card_uuid": CARD,
                                                "speakers": {"1": first["id"]}})
        pipeline = box.new_pipeline("P")
        box._runtime().hold = __import__("threading").Event()
        job = box.render(pipeline["id"], "Speaker 1: Hi.", configuration["id"])
        try:
            answer = post(client, api.ENGINE_ROUTE, key, {"engine": "breeze"})
            assert answer.status_code == 409
            assert "cancel it or clear the queue" in answer.json()["error"]
        finally:
            box._runtime().hold.set()
            settled(job["id"])

    def test_a_transcript_is_saved_and_whisper_refused_where_it_is_not_installed(
            self, client, key, monkeypatch):
        import mc_voice_models

        found = box.add_sample(spoken(), "Ada")
        saved = post(client, api.SAMPLE_TRANSCRIPT_ROUTE, key,
                     {"id": found["id"], "transcript": " Hello  there. "}).json()
        assert saved["sample"]["transcript"] == "Hello there."
        monkeypatch.setattr(mc_voice_models, "status",
                            lambda: type("Missing", (), {"ready": False})())
        answer = post(client, api.SAMPLE_TRANSCRIBE_ROUTE, key, {"id": found["id"]})
        assert answer.status_code == 409 and "Voice Chat" in answer.json()["error"]

    def test_a_design_is_queued_heard_and_kept_over_the_wire(self, client, key):
        import mc_voice_breeze

        class Breeze(FakeEngine):
            ID, LABEL, GUEST = "breeze", "Breeze TTS 2", "Breeze"

            def capabilities(self):
                return mc_voice_breeze.capabilities()

            def refuse(self, chosen):
                return ""

        box.use_engine(Breeze())
        box._runtime().seconds = 4.0
        box.set_settings({"engine": "breeze", "card_uuid": CARD})
        answer = post(client, api.DESIGN_ROUTE, key, {"description": "A warm voice.",
                                                      "seed": 9})
        assert answer.status_code == 200
        job = answer.json()["job"]
        assert job["kind"] == "design"
        settled(job["id"])
        audio = client.get(api.DESIGN_AUDIO_ROUTE, params={"id": job["id"]}, headers=key)
        assert audio.status_code == 200 and audio.content[:4] == b"RIFF"
        assert audio.headers["content-type"].startswith("audio/wav")
        kept = post(client, api.DESIGN_KEEP_ROUTE, key, {"id": job["id"], "title": "Warm"})
        assert kept.status_code == 200 and kept.json()["sample"]["source"] == "design"
        missing = client.get(api.DESIGN_AUDIO_ROUTE, params={"id": "0" * 16}, headers=key)
        assert missing.status_code == 404


class TestTheSharedModelSetting:
    def test_a_model_of_the_other_engines_is_not_told_to_this_one(self, client, key,
                                                                  monkeypatch):
        """The model is one Voice Box setting shared by both engines; a page that has
        not caught up with a switch sends the other engine's, which is no refusal."""
        told = []

        class Breeze(FakeEngine):
            SETTINGS_KEYS = ("card_uuid", "model_id", "fast_path", "keep_warm")

            def model_ids(self):
                return ("breeze-tts-2",)

            def set_settings(self, values):
                if values.get("model_id") not in (None, "", "breeze-tts-2"):
                    raise ValueError("not a Breeze model")
                told.append(dict(values))
                return dict(values)

        engine = Breeze()
        monkeypatch.setattr(api, "_engine", lambda: engine)
        answer = post(client, api.SETTINGS_ROUTE, key, {"model_id": "vibevoice-7b",
                                                         "card_uuid": CARD})
        assert answer.status_code == 200 and told == [{"card_uuid": CARD}]
        assert answer.json()["settings"]["model_id"] == "vibevoice-7b", \
            "the Voice Box's own setting keeps it for the other engine"
        post(client, api.SETTINGS_ROUTE, key, {"model_id": "breeze-tts-2"})
        assert told[-1] == {"model_id": "breeze-tts-2"}
