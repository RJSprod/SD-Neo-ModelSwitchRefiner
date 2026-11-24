"""The writers' instructions, kept here and used by every request that brings none.

Asked on 2026-11-24 about the system prompt editor Mini Paint NEO had drawn:
"those are related LLM studio IMO ... should it be removed from mini paint and
put in modelswitchrefiner?" Until then an edit to a writer's instructions lived
in Mini Paint and reached Mini Paint's presses alone -- LLM Studio's MiniMax H3
workspace went on writing under the defaults. These tests hold the move: the
store (:mod:`mc_llm_instructions`), the run that reads it
(:func:`mc_llm_sessions.minimax` and :func:`mc_llm_sessions.ltx`, every door's),
the external API's half of it, Mini Paint's one-time hand-over, and the
editor's route.
"""

from __future__ import annotations

import json
import logging

import pytest

import mc_llm_api as api
import mc_llm_instructions as store
import mc_llm_instructions_api as route
import mc_llm_jobs as jobs
import mc_llm_ltx
import mc_llm_sessions as sessions
from prompt_master.minimax import enhancer
from test_llm_external_api import FakeClient, client, enhancement, sighted  # noqa: F401 (fixtures)

HAIKU = "Write every prompt as a haiku."
TERSE = "Write it in one short line."


@pytest.fixture(autouse=True)
def data(tmp_path, monkeypatch, host):
    import mc_llm_paths

    monkeypatch.setattr(mc_llm_paths, "data_root", lambda: tmp_path)
    monkeypatch.setattr(store, "_warned", False)
    return tmp_path


def kept(data):
    return json.loads((data / store.FILE_NAME).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# The store
# --------------------------------------------------------------------------- #


class TestTheStore:
    def test_nothing_saved_is_the_default_and_no_file(self, data):
        assert store.saved("fl2va", has_image=False) is None
        found = store.view("fl2va", "text")
        assert found["source"] == store.DEFAULT
        assert found["text"] == found["default"] == enhancer.instructions("fl2va", False)
        assert not (data / store.FILE_NAME).exists()

    def test_a_saved_set_is_what_its_requests_run_under(self, data):
        found = store.save("fl2va", "image", HAIKU)

        assert found["source"] == store.OVERRIDE and found["text"] == HAIKU
        assert store.saved("fl2va", has_image=True) == HAIKU
        assert store.saved("fl2va", has_image=False) is None, "the other mode is its own set"
        assert store.saved("ref2va", has_image=True) is None, "and so is the other writer"
        assert kept(data)["sets"] == {"fl2va": {"image": HAIKU}}

    def test_the_six_sets_are_kept_apart(self, data):
        for name in store.SETS:
            for mode in store.MODES:
                store.save(name, mode, f"{name} {mode}")
        for name in store.SETS:
            for mode in store.MODES:
                assert store.saved(name, has_image=mode == "image") == f"{name} {mode}"

    def test_ltx_25_is_written_under_ltx_23s_set(self, data):
        store.save("ltx25", "text", TERSE)

        assert store.saved("ltx23", has_image=False) == TERSE
        assert store.view("ltx25", "text")["variant"] == "ltx23"
        assert set(kept(data)["sets"]) == {"ltx23"}

    def test_line_endings_are_made_one_kind_and_nothing_else_is_touched(self, data):
        store.save("ref2va", "text", "  First.\r\nSecond.\rThird.  ")
        assert store.saved("ref2va", has_image=False) == "  First.\nSecond.\nThird.  "

    @pytest.mark.parametrize("text", ["", "   ", "\n\n"])
    def test_blank_is_refused_and_nothing_written(self, data, text):
        with pytest.raises(store.Refused) as refusal:
            store.save("fl2va", "text", text)
        assert refusal.value.code == "empty_system_prompt"
        assert not (data / store.FILE_NAME).exists()

    def test_too_long_is_refused_never_cut(self, data):
        with pytest.raises(store.Refused) as refusal:
            store.save("fl2va", "text", "x" * (store.MAX_LENGTH + 1))
        assert refusal.value.code == "system_prompt_too_long"
        assert store.saved("fl2va", has_image=False) is None

    @pytest.mark.parametrize("variant, mode, code", [
        ("wan22", "text", "unknown_variant"), ("", "text", "unknown_variant"),
        ("fl2va", "video", "unknown_mode")])
    def test_a_set_that_is_none_of_the_writers_is_refused(self, data, variant, mode, code):
        with pytest.raises(store.Refused) as refusal:
            store.save(variant, mode, HAIKU)
        assert refusal.value.code == code

    def test_reading_an_unknown_set_is_the_default_never_an_error(self, data):
        assert store.saved("wan22", has_image=False) is None

    def test_the_default_saved_unchanged_is_no_override(self, data):
        shipped = enhancer.instructions("fl2va", False)
        found = store.save("fl2va", "text", shipped + "\n")

        assert found["source"] == store.DEFAULT
        assert store.saved("fl2va", has_image=False) is None

    def test_the_default_saved_over_an_override_forgets_it(self, data):
        store.save("fl2va", "text", HAIKU)
        store.save("fl2va", "text", enhancer.instructions("fl2va", False))
        assert store.saved("fl2va", has_image=False) is None

    def test_restore_forgets_one_set_and_says_whether_there_was_one(self, data):
        store.save("fl2va", "text", HAIKU)
        store.save("ref2va", "text", TERSE)

        assert store.restore("fl2va", "text") is True
        assert store.restore("fl2va", "text") is False
        assert store.saved("fl2va", has_image=False) is None
        assert store.saved("ref2va", has_image=False) == TERSE

    def test_saved_sets_says_which_without_their_text(self, data):
        store.save("ltx25", "image", HAIKU)
        found = store.saved_sets()
        assert found["ltx23"] == {"text": False, "image": True}
        assert found["fl2va"] == {"text": False, "image": False}
        assert HAIKU not in json.dumps(found)

    def test_a_file_that_cannot_be_read_is_the_defaults_and_said_once(self, data, caplog):
        (data / store.FILE_NAME).write_text("{not json", encoding="utf-8")
        with caplog.at_level(logging.WARNING, logger="model_chain"):
            assert store.saved("fl2va", has_image=False) is None
            assert store.saved("fl2va", has_image=True) is None
            assert store.view("fl2va", "text")["source"] == store.DEFAULT
        warned = [record for record in caplog.records if "writer instructions" in record.getMessage()]
        assert len(warned) == 1

    def test_no_instructions_are_ever_logged(self, data, caplog):
        with caplog.at_level(logging.DEBUG, logger="model_chain"):
            store.save("fl2va", "text", HAIKU)
            store.restore("fl2va", "text")
            store.adopt({"ref2va": {"image": TERSE}}, origin="minipaint-clipboard")
        said = "\n".join(record.getMessage() for record in caplog.records)
        assert "fl2va" in said, "the set is named"
        assert HAIKU not in said and TERSE not in said


class TestTheHandOver:
    """Mini Paint kept its own until this extension did, and hands them over once."""

    def test_sets_this_store_lacks_are_taken(self, data):
        found = store.adopt({"fl2va": {"text": HAIKU, "image": TERSE}, "ltx25": {"text": TERSE}},
                            origin="minipaint-clipboard")

        assert found == {"adopted": 3, "kept": 0, "refused": 0}
        assert store.saved("fl2va", has_image=False) == HAIKU
        assert store.saved("fl2va", has_image=True) == TERSE
        assert store.saved("ltx23", has_image=False) == TERSE

    def test_an_edit_made_here_is_never_overwritten(self, data):
        store.save("fl2va", "text", "Mine, made here.")
        found = store.adopt({"fl2va": {"text": HAIKU}}, origin="minipaint-clipboard")

        assert found == {"adopted": 0, "kept": 1, "refused": 0}
        assert store.saved("fl2va", has_image=False) == "Mine, made here."

    def test_the_default_handed_over_is_no_override(self, data):
        found = store.adopt({"ref2va": {"image": enhancer.instructions("ref2va", True)}})
        assert found["adopted"] == 0
        assert store.saved("ref2va", has_image=True) is None

    def test_what_cannot_be_kept_is_counted_and_left(self, data):
        found = store.adopt({"wan22": {"text": HAIKU}, "fl2va": {"video": HAIKU, "text": "  ",
                                                                  "image": "x" * (store.MAX_LENGTH + 1)}})
        assert found == {"adopted": 0, "kept": 0, "refused": 4}

    def test_it_is_recorded_by_who_handed_over(self, data):
        store.adopt({"fl2va": {"text": HAIKU}}, origin="minipaint-clipboard")
        store.save("ref2va", "text", TERSE)

        handed = kept(data)["handed_over"]["minipaint-clipboard"]
        assert handed["adopted"] == 1 and handed["kept"] == 0
        assert "at" in handed

    def test_a_second_hand_over_takes_nothing_twice(self, data):
        store.adopt({"fl2va": {"text": HAIKU}}, origin="minipaint-clipboard")
        assert store.adopt({"fl2va": {"text": HAIKU}}, origin="minipaint-clipboard")["adopted"] == 0


# --------------------------------------------------------------------------- #
# Every request that brings none runs under them
# --------------------------------------------------------------------------- #


class TestTheRunReadsThem:
    def test_a_minimax_run_without_instructions_runs_under_the_saved_ones(self, client):
        store.save("fl2va", "text", HAIKU)

        list(sessions.minimax("a car chase", "fl2va", None, 7, sessions.Cancellation()))

        assert client.system == HAIKU

    def test_with_nothing_saved_the_run_is_the_default(self, client):
        list(sessions.minimax("a car chase", "ref2va", None, 7, sessions.Cancellation()))
        assert client.system == enhancer.messages("a car chase", variant="ref2va")[0]["content"]

    def test_a_picture_runs_under_the_with_a_picture_set(self, client, sighted):
        store.save("fl2va", "text", TERSE)
        store.save("fl2va", "image", HAIKU)
        client.answers = ["A red car on a road.", "An H3 prompt."]

        list(sessions.minimax("a car chase", "fl2va", "data:image/png;base64,AA", 7,
                              sessions.Cancellation()))

        assert client.system == HAIKU

    def test_an_unknown_variant_is_fl2vas_run_and_fl2vas_instructions(self, client):
        store.save("fl2va", "text", HAIKU)
        list(sessions.minimax("a car chase", "whatever", None, 7, sessions.Cancellation()))
        assert client.system == HAIKU

    def test_a_callers_own_instructions_still_win_for_its_request(self, client):
        store.save("fl2va", "text", HAIKU)
        list(sessions.minimax("a car chase", "fl2va", None, 7, sessions.Cancellation(),
                              system="The caller's."))
        assert client.system == "The caller's."

    def test_wangps_at_still_appends_to_the_saved_ones(self, client):
        store.save("fl2va", "text", HAIKU)
        list(sessions.minimax("a car chase @ under ten seconds", "fl2va", None, 7,
                              sessions.Cancellation()))
        assert client.system.startswith(HAIKU)
        assert "under ten seconds" in client.system

    @pytest.mark.parametrize("model", ["ltx23", "ltx25"])
    def test_an_ltx_run_reads_the_one_ltx_set(self, client, model):
        store.save("ltx23", "text", TERSE)
        list(sessions.ltx("she turns and waves", None, 7, sessions.Cancellation(), model=model))
        assert client.system == TERSE

    def test_an_ltx_run_with_a_frame_reads_its_with_a_picture_set(self, client, sighted):
        store.save("ltx23", "text", TERSE)
        list(sessions.ltx("she turns", "data:image/png;base64,AA", 7, sessions.Cancellation()))
        assert client.system == mc_llm_ltx.messages(
            "she turns", image_data_url="data:image/png;base64,AA")[0]["content"]

    def test_the_console_says_saved_instructions_and_never_their_text(self, client, caplog):
        store.save("fl2va", "text", HAIKU)
        with caplog.at_level(logging.DEBUG, logger="model_chain"):
            list(sessions.minimax("a car chase", "fl2va", None, 7, sessions.Cancellation()))
        said = "\n".join(record.getMessage() for record in caplog.records)
        assert "under saved instructions" in said
        assert HAIKU not in said

    def test_they_are_read_when_the_run_starts(self, client):
        """A request queued before an edit is written under the edit."""
        run = sessions.minimax("a car chase", "fl2va", None, 7, sessions.Cancellation())
        store.save("fl2va", "text", HAIKU)
        list(run)
        assert client.system == HAIKU

    def test_llm_studios_own_minimax_workspace_runs_under_them(self, client, monkeypatch):
        """The door they never reached while they lived in Mini Paint."""
        import mc_llm_minimax_panel

        monkeypatch.setattr(mc_llm_minimax_panel, "jobs_active", lambda: False)
        monkeypatch.setattr(mc_llm_minimax_panel, "_remember", lambda *args: None)
        store.save("ref2va", "text", HAIKU)

        list(mc_llm_minimax_panel._enhance("a car chase", "ref2va", None, 7))

        assert client.system == HAIKU

    def test_an_external_request_runs_under_them(self, client):
        store.save("fl2va", "text", HAIKU)
        identifier = api.submit_minimax("a car chase", origin="test")
        jobs.drain_once()

        assert api.status(identifier)["state"] == jobs.DONE
        assert client.system == HAIKU


# --------------------------------------------------------------------------- #
# The external API
# --------------------------------------------------------------------------- #


class TestTheApi:
    def test_it_says_it_keeps_them(self):
        assert api.capabilities()["keeps_instructions"] is True

    def test_instructions_answers_with_the_saved_ones_or_the_default(self):
        assert api.instructions("fl2va")["source"] == "default"
        api.save_instructions("fl2va", HAIKU)
        found = api.instructions("fl2va")
        assert found == {"variant": "fl2va", "mode": "text", "text": HAIKU, "source": "override",
                         "default": enhancer.instructions("fl2va", False),
                         "label": "MiniMax H3 FL2VA, without a picture"}

    def test_either_ltx_name_is_the_ltx_set(self):
        api.save_instructions("ltx25", TERSE, has_image=True)
        assert api.instructions("ltx23", has_image=True)["text"] == TERSE
        assert api.instructions("ltx23", has_image=False)["source"] == "default"

    def test_anything_else_is_a_minimax_variant_as_system_prompt_resolves_it(self):
        assert api.instructions("")["variant"] == "fl2va"
        assert api.instructions("REF2VA")["variant"] == "ref2va"

    def test_a_blank_or_long_save_is_refused_with_a_code(self):
        with pytest.raises(api.Rejected) as blank:
            api.save_instructions("fl2va", "  ")
        assert blank.value.code == "empty_system_prompt"
        with pytest.raises(api.Rejected) as long:
            api.save_instructions("fl2va", "x" * (store.MAX_LENGTH + 1))
        assert long.value.code == "system_prompt_too_long"

    def test_restore_says_whether_there_was_one(self):
        api.save_instructions("ref2va", HAIKU, has_image=True)
        assert api.restore_instructions("ref2va", has_image=True) is True
        assert api.restore_instructions("ref2va", has_image=True) is False

    def test_the_hand_over_is_the_stores(self):
        found = api.adopt_instructions({"fl2va": {"text": HAIKU}}, origin="minipaint-clipboard")
        assert found == {"adopted": 1, "kept": 0, "refused": 0}
        assert api.instructions("fl2va")["text"] == HAIKU

    def test_the_defaults_stay_readable_as_they_were(self):
        api.save_instructions("fl2va", HAIKU)
        assert api.system_prompt("fl2va") == enhancer.instructions("fl2va", False)
        assert api.system_prompts()["fl2va"]["text"] == enhancer.instructions("fl2va", False)


# --------------------------------------------------------------------------- #
# The editor's route
# --------------------------------------------------------------------------- #


class TestTheRoute:
    def test_describe_offers_the_three_writers_and_two_modes(self):
        status, found = route.answer({"action": "describe"})

        assert status == 200 and found["ok"] is True
        assert [item["id"] for item in found["sets"]] == ["fl2va", "ref2va", "ltx23"]
        assert [item["id"] for item in found["modes"]] == ["text", "image"]
        assert found["view"]["variant"] == "fl2va" and found["view"]["mode"] == "text"
        assert set(found["availability"]) == {"state", "text"}

    def test_describe_opens_on_the_set_named_else_the_fallback(self):
        assert route.answer({"action": "describe", "variant": "ltx25", "mode": "image"})[1]["view"][
            "variant"] == "ltx23"
        found = route.answer({"action": "describe", "fallback": "ref2va"})[1]["view"]
        assert (found["variant"], found["mode"]) == ("ref2va", "text")
        found = route.answer({"action": "describe", "variant": "nonsense", "mode": "nonsense"})[1]["view"]
        assert (found["variant"], found["mode"]) == ("fl2va", "text")

    def test_save_answers_with_the_set_as_it_now_stands(self):
        status, found = route.answer({"action": "save", "variant": "ref2va", "mode": "image",
                                      "text": HAIKU})
        assert status == 200 and found["view"]["source"] == "override"
        assert found["view"]["text"] == HAIKU
        assert "every request" in found["status"].lower()
        assert store.saved("ref2va", has_image=True) == HAIKU

    def test_saving_the_default_says_no_override_is_kept(self):
        shipped = enhancer.instructions("fl2va", True)
        status, found = route.answer({"action": "save", "variant": "fl2va", "mode": "image",
                                      "text": shipped})
        assert status == 200 and found["view"]["source"] == "default"
        assert "no override" in found["status"]

    def test_restore_answers_with_the_default(self):
        store.save("fl2va", "text", HAIKU)
        status, found = route.answer({"action": "restore", "variant": "fl2va", "mode": "text"})
        assert status == 200 and found["view"]["source"] == "default"
        assert found["status"] == "Back to the default."
        assert route.answer({"action": "restore", "variant": "fl2va", "mode": "text"})[1][
            "status"].startswith("There was no override")

    def test_a_refusal_is_a_400_with_the_stores_code_and_sentence(self):
        status, found = route.answer({"action": "save", "variant": "fl2va", "mode": "text",
                                      "text": " "})
        assert status == 400 and found["ok"] is False
        assert found["code"] == "empty_system_prompt"
        assert "blank" in found["message"]

    @pytest.mark.parametrize("body", [{}, {"action": "toggle"}, None, "describe"])
    def test_anything_else_is_a_400(self, body):
        status, found = route.answer(body)
        assert status == 400 and found["ok"] is False and found["code"] == "unknown_action"

    def test_a_failure_says_so_and_logs_no_text(self, monkeypatch, caplog):
        def broken(*args):
            raise OSError("disk full")

        monkeypatch.setattr(store, "_write", broken)
        with caplog.at_level(logging.WARNING, logger="model_chain"):
            status, found = route.answer({"action": "save", "variant": "fl2va", "mode": "text",
                                          "text": HAIKU})
        assert status == 500 and found["ok"] is False
        assert HAIKU not in "\n".join(record.getMessage() for record in caplog.records)

    @pytest.mark.parametrize("found, state", [
        ({"enabled": False, "reason": "LLM Studio is switched off."}, "off"),
        ({"enabled": True, "configured": False, "reason": "No language model is set up."},
         "unconfigured"),
        ({"enabled": True, "configured": True, "vision": False, "model": "m.gguf"}, "novision"),
        ({"enabled": True, "configured": True, "vision": True, "model": "m.gguf"}, "ready"),
    ])
    def test_the_line_about_the_llm_side(self, monkeypatch, found, state):
        monkeypatch.setattr(api, "capabilities", lambda: found)
        answered = route.availability()
        assert answered["state"] == state
        if state in ("novision", "ready"):
            assert "m.gguf" in answered["text"]

    def test_it_is_registered_where_the_other_routes_are(self):
        from pathlib import Path

        script = (Path(__file__).resolve().parents[1] / "scripts" / "model_chain.py").read_text(
            encoding="utf-8")
        assert "script_callbacks.on_app_started(mc_llm_instructions_api.install)" in script

    def test_with_no_app_or_no_fastapi_nothing_is_registered(self, monkeypatch):
        assert route.install(None, None) is False
        monkeypatch.setattr(route, "Request", None)

        class App:
            routes: list = []

            def add_api_route(self, *args, **kwargs):
                raise AssertionError("registered without FastAPI")

        assert route.install(None, App()) is False


class TestTheRouteOnFastAPI:
    """The gate and the answer through a real FastAPI app, where there is one."""

    @pytest.fixture
    def http(self):
        fastapi = pytest.importorskip("fastapi")
        testclient = pytest.importorskip("fastapi.testclient")
        import mc_llm_conversation_api as conversation

        app = fastapi.FastAPI()
        assert route.install(None, app) is True
        assert route.install(None, app) is True, "a second install is the first"
        assert sum(1 for item in app.routes if getattr(item, "path", "") == route.ROUTE) == 1
        return testclient.TestClient(app), conversation

    def test_without_the_pages_key_it_is_refused(self, http):
        client_, _ = http
        answered = client_.post(route.ROUTE, json={"action": "describe"})
        assert answered.status_code == 401

    def test_from_another_origin_it_is_refused(self, http):
        client_, conversation = http
        answered = client_.post(route.ROUTE, json={"action": "describe"},
                                headers={conversation.HEADER: conversation.session_token(),
                                         "origin": "https://elsewhere.example"})
        assert answered.status_code == 403

    def test_with_the_key_it_answers_and_saves(self, http):
        client_, conversation = http
        key = {conversation.HEADER: conversation.session_token()}
        answered = client_.post(route.ROUTE, headers=key,
                                json={"action": "save", "variant": "fl2va", "mode": "text",
                                      "text": HAIKU})
        assert answered.status_code == 200 and answered.json()["view"]["text"] == HAIKU
        assert answered.headers["cache-control"] == "no-store"
        assert store.saved("fl2va", has_image=False) == HAIKU


# --------------------------------------------------------------------------- #
# LLM Studio's MiniMax workspace opens the editor
# --------------------------------------------------------------------------- #


class TestTheMiniMaxWorkspaceOpensTheEditor:
    def _built(self, monkeypatch):
        import gradio as gr

        import mc_llm_minimax_panel

        seen = {}

        def recording(original):
            class Recorded(original):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    if kwargs.get("elem_id"):
                        seen[kwargs["elem_id"]] = self
            return Recorded

        for name in ("Button", "Radio", "Image"):
            monkeypatch.setattr(gr, name, recording(getattr(gr, name)))
        mc_llm_minimax_panel.build()
        return seen

    def test_the_inspector_has_the_button_and_it_is_the_browsers_alone(self, monkeypatch):
        import mc_llm_minimax_panel
        import mc_llm_ui as ui

        seen = self._built(monkeypatch)
        button = seen[ui.ident("minimax", "system-prompt")]
        clicks = [kwargs for kind, kwargs in button._callbacks if kind == "click"]

        assert button.value == "System prompt"
        assert len(clicks) == 1
        assert clicks[0]["fn"] is None, "no round trip through the queue"
        assert clicks[0]["js"] == mc_llm_minimax_panel.OPEN_SYSTEM_PROMPT
        assert clicks[0]["inputs"] == [seen[ui.ident("minimax", "variant")],
                                       seen[ui.ident("minimax", "image")]], "the variant, and the frame"

    def test_the_press_names_what_the_sheet_script_defines(self):
        from pathlib import Path

        import mc_llm_minimax_panel

        script = (Path(mc_llm_minimax_panel.__file__).resolve().parent / "javascript"
                  / "mc_sheet.js").read_text(encoding="utf-8")
        assert "window.mcSheet" in mc_llm_minimax_panel.OPEN_SYSTEM_PROMPT
        assert "window.mcSheet = window.mcSheet || (function () {" in script
        assert "        systemPrompt: systemPrompt," in script

    @pytest.mark.skipif(__import__("shutil").which("node") is None, reason="node is not installed")
    @pytest.mark.parametrize("sheet, image, wanted", [
        (True, None, [{"name": "system-prompt", "variant": "ref2va", "mode": "text"}]),
        (True, {"url": "/file=frame.png"}, [{"name": "system-prompt", "variant": "ref2va", "mode": "image"}]),
        (False, None, []),
    ])
    def test_the_press_opens_the_variant_with_a_frame_or_without_and_never_throws(self, sheet, image,
                                                                                    wanted):
        import subprocess

        import mc_llm_minimax_panel

        scenario = f"""
            globalThis.window = globalThis;
            const opened = [];
            const warned = [];
            console.warn = (text) => warned.push(text);
            if ({json.dumps(sheet)}) window.mcSheet = {{systemPrompt: (o) => opened.push(o)}};
            const press = eval({json.dumps(mc_llm_minimax_panel.OPEN_SYSTEM_PROMPT)});
            const out = press("ref2va", {json.dumps(image)});
            process.stdout.write(JSON.stringify({{opened, out, warned: warned.length}}));
        """
        result = subprocess.run(["node", "-e", scenario], capture_output=True, text=True, timeout=60)

        assert result.returncode == 0, result.stderr
        found = json.loads(result.stdout)
        assert found == {"opened": wanted, "out": [], "warned": 0 if sheet else 1}
