"""The full-page system prompt editor: what it opens on, and what its buttons do.

One system prompt per character, two places to change it -- LLM Studio's
character editor (the override box) and this editor, opened from the tab's
character screen and from the flyout's ⋯ menu. Both write the character's
``system`` field, the one ``prompt_master.chat.prompt.system_text`` reads, so
these tests hold the editor to that field rather than to anything of its own.

The rule the editor adds is the one worth pinning: it opens on the *built*
prompt, so Apply pressed on a page nobody edited must not turn the built prompt
into an override. A frozen copy would stop following the character's Context
and the persona the moment it was saved, and nothing on screen would say so.
"""

from __future__ import annotations

import pytest

import mc_llm_conversation_api as api
import mc_llm_conversation_service as service
import mc_llm_paths


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch, host):
    monkeypatch.setattr(mc_llm_paths, "data_root", lambda: tmp_path)
    from prompt_master.chat.characters import Character, CharacterStore

    CharacterStore(tmp_path / "characters").save(Character(
        name="Ada", context="A reader of maps.", greeting="Hello.", temperature=0.4,
        voice="official:af_nicole", pocket_voice="alba"))
    yield tmp_path


def characters(store):
    from prompt_master.chat.characters import CharacterStore

    return CharacterStore(store / "characters")


def built(store, name="Ada"):
    """The prompt a reply is given, built the way a reply builds it."""
    from dataclasses import replace

    from prompt_master.chat.characters import load_persona
    from prompt_master.chat.prompt import build

    loaded = replace(characters(store).load(name), system="")
    return build(loaded, load_persona(mc_llm_paths.app_paths()), [])[0]["content"]


def given(store, name="Ada"):
    """What a reply is actually given now, override and all."""
    from prompt_master.chat.characters import load_persona
    from prompt_master.chat.prompt import build

    return build(characters(store).load(name), load_persona(mc_llm_paths.app_paths()),
                 [])[0]["content"]


class TestWhatTheEditorOpensOn:
    def test_a_character_with_no_override_opens_on_the_built_prompt(self, store):
        found = service.system_prompt("Ada")

        assert found["ok"] is True
        assert found["character"] == "Ada"
        assert found["source"] == "default"
        assert found["text"] == found["default"] == built(store)
        assert "A reader of maps." in found["text"]

    def test_an_override_opens_as_stored_with_its_placeholders(self, store):
        """``{{char}}`` is substituted when the prompt is *used*; the editor is
        where it is written, so it has to show what is written."""
        from dataclasses import replace

        loaded = characters(store).load("Ada")
        characters(store).save(replace(loaded, system="You are {{char}}. Be brief."))

        found = service.system_prompt("Ada")

        assert found["source"] == "override"
        assert found["text"] == "You are {{char}}. Be brief."
        assert found["default"] == built(store)

    def test_the_default_follows_the_persona(self, store):
        from prompt_master.chat.characters import Persona, save_persona

        save_persona(mc_llm_paths.app_paths(), Persona(name="Rob", description="A sailor."))

        found = service.system_prompt("Ada")

        assert "Rob" in found["default"] and "A sailor." in found["default"]
        assert found["default"] == built(store)

    def test_a_name_in_another_case_opens_the_character_under_its_own_name(self):
        assert service.system_prompt("ada")["character"] == "Ada"


class TestApplyOverride:
    def test_it_becomes_what_every_reply_is_given(self, store):
        found = service.set_system_prompt("Ada", "You are {{char}}. Answer in one line.")

        assert found["saved"] is True
        assert found["source"] == "override"
        assert found["text"] == "You are {{char}}. Answer in one line."
        assert given(store) == "You are Ada. Answer in one line."
        assert "Override saved for Ada" in found["message"]

    def test_the_rest_of_the_character_is_left_as_it_was(self, store):
        service.set_system_prompt("Ada", "Be brief.")

        loaded = characters(store).load("Ada")
        assert (loaded.context, loaded.greeting, loaded.temperature) == \
            ("A reader of maps.", "Hello.", 0.4)
        assert (loaded.voice, loaded.pocket_voice) == ("official:af_nicole", "alba")

    def test_it_is_written_into_the_character_s_own_file(self, store):
        """A character read out of ``Chiharu.yaml`` goes back into it, not into
        a second file named after the name inside it."""
        (store / "characters" / "Chiharu.yaml").write_text(
            "name: Chiharu Yamada\ncontext: A tech enthusiast.\n", encoding="utf-8")

        service.set_system_prompt("Chiharu Yamada", "Be brief.")

        assert sorted(path.name for path in (store / "characters").glob("*.yaml")) == \
            ["Ada.yaml", "Chiharu.yaml"]
        assert characters(store).load("Chiharu Yamada").system == "Be brief."

    def test_the_built_prompt_applied_untouched_is_not_an_override(self, store):
        """The editor opens on the built prompt. Apply on an untouched page
        must not freeze it: the character would stop following its Context."""
        opened = service.system_prompt("Ada")["text"]

        found = service.set_system_prompt("Ada", opened)

        assert found["saved"] is False
        assert found["source"] == "default"
        assert characters(store).load("Ada").system == ""
        assert "already the default" in found["message"]

    def test_the_built_prompt_with_only_its_edges_changed_is_still_not_one(self, store):
        found = service.set_system_prompt("Ada", "\n  " + built(store) + "  \n")

        assert found["source"] == "default"
        assert characters(store).load("Ada").system == ""

    def test_editing_an_override_back_to_the_built_prompt_removes_it(self, store):
        service.set_system_prompt("Ada", "Be brief.")

        found = service.set_system_prompt("Ada", built(store))

        assert found["saved"] is True
        assert found["source"] == "default"
        assert characters(store).load("Ada").system == ""
        assert "override is gone" in found["message"]

    def test_an_empty_box_is_the_default(self, store):
        service.set_system_prompt("Ada", "Be brief.")

        found = service.set_system_prompt("Ada", "   ")

        assert found["source"] == "default"
        assert characters(store).load("Ada").system == ""
        assert "empty prompt is no override" in found["message"]

    def test_the_same_override_twice_writes_once(self, store):
        service.set_system_prompt("Ada", "Be brief.")

        assert service.set_system_prompt("Ada", "Be brief.")["saved"] is False


class TestRestoreDefault:
    def test_it_forgets_the_override(self, store):
        service.set_system_prompt("Ada", "Be brief.")

        found = service.set_system_prompt("Ada", restore=True)

        assert found["saved"] is True
        assert found["source"] == "default"
        assert found["text"] == built(store)
        assert given(store) == built(store)
        assert found["message"] == "Back to the default for Ada."

    def test_with_no_override_it_writes_nothing_and_says_so(self, store):
        found = service.set_system_prompt("Ada", restore=True)

        assert found["saved"] is False
        assert "There was no override" in found["message"]

    def test_the_text_sent_with_it_is_ignored(self, store):
        service.set_system_prompt("Ada", "Be brief.")

        found = service.set_system_prompt("Ada", "Something else.", restore=True)

        assert found["source"] == "default"
        assert characters(store).load("Ada").system == ""


class TestWhatIsRefused:
    def test_a_character_that_is_not_there_is_a_404(self, store):
        payload, status = api.system_prompt("Nobody")

        assert status == 404
        assert payload["ok"] is False
        assert payload["error"]["code"] == service.NOT_FOUND
        assert "Nobody" in payload["error"]["message"]

    @pytest.mark.parametrize("who", ["", "   ", None])
    def test_no_character_at_all_is_a_400(self, who):
        payload, status = api.system_prompt(who)

        assert status == 400
        assert payload["error"]["code"] == service.INVALID_INPUT

    @pytest.mark.parametrize("text", [None, 5, ["Be brief."]])
    def test_an_apply_without_text_writes_nothing(self, store, text):
        """Absent text is not "empty text": an empty box is a choice the page
        makes, a missing field is a page that sent the wrong thing."""
        service.set_system_prompt("Ada", "Be brief.")

        payload, status = api.set_system_prompt({"character": "Ada", "text": text})

        assert status == 400
        assert payload["error"]["code"] == service.INVALID_INPUT
        assert characters(store).load("Ada").system == "Be brief."

    @pytest.mark.parametrize("restore", ["true", "false", 1, 0, None])
    def test_restore_is_true_or_false_and_nothing_else(self, store, restore):
        """"false" is truthy. A route that took bool() of it would throw the
        override away when asked to keep it."""
        service.set_system_prompt("Ada", "Be brief.")

        payload, status = api.set_system_prompt({"character": "Ada", "text": "Other.",
                                                 "restore": restore})

        assert status == 400
        assert payload["error"]["code"] == service.INVALID_INPUT
        assert characters(store).load("Ada").system == "Be brief."

    def test_a_prompt_past_the_limit_is_refused_and_never_cut(self, store):
        payload, status = api.set_system_prompt(
            {"character": "Ada", "text": "x" * (service.MAX_TEXT_BYTES + 1)})

        assert status == 400
        assert payload["error"]["code"] == service.INVALID_INPUT
        assert characters(store).load("Ada").system == ""

    def test_a_save_that_fails_says_so_and_is_worth_retrying(self, store, monkeypatch):
        from prompt_master.chat.characters import CharacterStore

        def refuse(self, character, previous_name=None):
            raise OSError("disk full")

        monkeypatch.setattr(CharacterStore, "save", refuse)

        payload, status = api.set_system_prompt({"character": "Ada", "text": "Be brief."})

        assert status == 500
        assert payload["error"]["code"] == service.SAVE_FAILED
        assert payload["error"]["retryable"] is True


class TestTheRoute:
    def test_the_api_s_apply_and_restore_are_the_service_s(self, store):
        payload, status = api.set_system_prompt({"character": "Ada", "text": "Be brief."})
        assert (status, payload["source"]) == (200, "override")

        payload, status = api.set_system_prompt({"character": "Ada", "restore": True})
        assert (status, payload["source"]) == (200, "default")

        payload, status = api.system_prompt("Ada")
        assert (status, payload["text"]) == (200, built(store))

    def test_it_is_one_of_the_routes_this_extension_registers(self):
        assert api.SYSTEM_PROMPT_ROUTE in api.ROUTES
        assert api.SYSTEM_PROMPT_ROUTE == api.PREFIX + "/system-prompt"

    def test_it_answers_both_reading_and_writing(self, monkeypatch):
        """One path, two methods: GET opens the editor, POST is its buttons.
        Registered as one route, because a reload's "already registered"
        check looks at paths, and a second registration of the same path would
        be skipped by it."""

        class App:
            routes = []

            def __init__(self):
                self.added = {}

            def add_api_route(self, path, handler, methods):
                self.added[path] = list(methods)

        monkeypatch.setattr(api, "Request", object)
        app = App()

        assert api.install(app=app) is True
        assert sorted(app.added[api.SYSTEM_PROMPT_ROUTE]) == ["GET", "POST"]


class TestLLMStudioOpensIt:
    """"In LLM studio character menu, i just want a button that opens a full
    page editor like this where i can set the prompt for that character.\""""

    def _built(self, monkeypatch):
        import gradio as gr

        import mc_llm_chat_panel

        seen = {}

        def recording(original):
            class Recorded(original):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    if kwargs.get("elem_id"):
                        seen[kwargs["elem_id"]] = self
            return Recorded

        for name in ("Button", "Textbox", "Dropdown"):
            monkeypatch.setattr(gr, name, recording(getattr(gr, name)))
        mc_llm_chat_panel.build()
        return seen

    def test_the_character_screen_has_the_button(self, monkeypatch):
        import mc_llm_chat_panel
        import mc_llm_ui as ui

        seen = self._built(monkeypatch)
        button = seen[ui.ident("chat", "system-open")]
        clicks = [kwargs for kind, kwargs in button._callbacks if kind == "click"]

        assert button.value == "⤢ System prompt"
        assert len(clicks) == 1
        assert clicks[0]["fn"] is None, "the browser's alone: no round trip through the queue"
        assert clicks[0]["js"] == mc_llm_chat_panel.OPEN_SYSTEM_EDITOR
        assert clicks[0]["inputs"] == [seen[ui.ident("chat", "who")]], "Talking to"

    def test_the_override_box_is_named_for_the_editor_to_find(self, monkeypatch):
        import mc_llm_ui as ui

        seen = self._built(monkeypatch)

        assert seen[ui.ident("chat", "system")].label == "System prompt override"
        assert seen[ui.ident("chat", "name")].label == "Name"

    def test_the_press_names_what_the_editor_script_defines(self):
        """Two halves in two languages with nothing between them but these
        names, so they are compared rather than trusted."""
        from pathlib import Path

        import mc_llm_chat_panel

        script = (Path(mc_llm_chat_panel.__file__).resolve().parent / "javascript"
                  / "forge_assistant_system.js").read_text(encoding="utf-8")

        assert "window.forgeAssistant.systemEditor" in mc_llm_chat_panel.OPEN_SYSTEM_EDITOR
        assert "NS.systemEditor = {" in script
        assert "        open(character) {" in script

    @pytest.mark.skipif(__import__("shutil").which("node") is None,
                        reason="node is not installed")
    @pytest.mark.parametrize("editor", [True, False])
    def test_the_press_hands_the_editor_the_character_and_never_throws(self, editor):
        import json
        import subprocess

        import mc_llm_chat_panel

        scenario = f"""
            globalThis.window = globalThis;
            const opened = [];
            const warned = [];
            console.warn = (text) => warned.push(text);
            if ({json.dumps(editor)}) {{
                window.forgeAssistant = {{systemEditor: {{open: (who) => opened.push(who)}}}};
            }}
            const press = eval({json.dumps(mc_llm_chat_panel.OPEN_SYSTEM_EDITOR)});
            const out = press("Ada");
            process.stdout.write(JSON.stringify({{opened, out, warned: warned.length}}));
        """
        result = subprocess.run(["node", "-e", scenario], capture_output=True, text=True,
                                timeout=60)

        assert result.returncode == 0, result.stderr
        found = json.loads(result.stdout)
        assert found["out"] == []
        if editor:
            assert found == {"opened": ["Ada"], "out": [], "warned": 0}
        else:
            assert found == {"opened": [], "out": [], "warned": 1}
