"""The LoRA viewer's route (mc_lora_viewer): a picture and trigger words per LoRA.

The viewer reads the tags in the browser; this answers, for every name at once,
what the row needs that only the server has. The rules held here are the ones
the user would notice going wrong: a tag Forge would load reads as found and a
typo does not, the picture is the one the LoRA browser shows, the trigger words
are Civitai's and the user's own activation text without repeats, and opening
the viewer never reads a whole LoRA file to hash it.
"""

from __future__ import annotations

import base64
import importlib
import inspect
import sys
import types
import typing

import pytest

import mc_lora_details
import mc_lora_hashing
import mc_lora_images
import mc_lora_viewer
from mc_lora_library import LibraryRecord, PromptEntry

# A 1x1 PNG, for a downloaded Civitai example picture.
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADUlEQVR4nGP4z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg==")


class Network:
    def __init__(self, name, alias=None, filename=None):
        self.name = name
        self.alias = alias or name
        self.filename = filename or f"/loras/{name}.safetensors"


def registry(*entries, forbidden=(), later=()):
    """Forge's ``networks`` module as the viewer reads it: two maps, the forbidden
    aliases, and a rescan that finds ``later`` the way a file added since the
    last refresh would be found."""
    module = types.SimpleNamespace(available_networks={}, available_network_aliases={},
                                   forbidden_network_aliases=set(forbidden), rescans=[])

    def add(entry):
        module.available_networks[entry.name] = entry
        module.available_network_aliases[entry.name] = entry
        module.available_network_aliases[entry.alias] = entry

    def update_available_networks_by_names(names):
        module.rescans.append(list(names))
        for entry in later:
            if entry.name in names:
                add(entry)

    for entry in entries:
        add(entry)
    module.update_available_networks_by_names = update_available_networks_by_names
    return module


class Page:
    """Forge's LoRA page: ``create_item`` reads a card off the disk."""

    def __init__(self, cards):
        self.cards = cards
        self.asked = []
        self.items = {}

    def create_item(self, name, index=None, enable_filter=True):
        self.asked.append((name, enable_filter))
        card = self.cards.get(name)
        if isinstance(card, Exception):
            raise card
        return dict(card) if card is not None else None


class Library:
    def __init__(self, records=()):
        self.records = list(records)
        self.asked = []

    def lookup(self, **kwargs):
        self.asked.append(kwargs)
        return self.records, "sha256" if self.records else ""


@pytest.fixture
def library(monkeypatch, lora_root):
    found = Library()
    monkeypatch.setattr(mc_lora_details, "get_library", lambda: found)

    def never(path, *a, **k):
        raise AssertionError("the viewer hashed a LoRA file")

    monkeypatch.setattr(mc_lora_hashing, "sha256_file", never)
    return found


def card(name, preview="./sd_extra_networks/thumb?filename=x.png", activation=""):
    return {"name": name, "filename": f"/loras/{name}.safetensors", "preview": preview,
            "user_metadata": {"activation text": activation}, "shorthash": ""}


class TestFound:
    def test_a_tag_is_found_by_its_alias_as_forge_finds_it(self, library):
        networks = registry(Network("realfilter_v2_final", alias="realfilter_v2"))
        page = Page({"realfilter_v2_final": card("realfilter_v2_final")})
        answer = mc_lora_viewer.details(["realfilter_v2"], networks=networks, page=page)
        (row,) = answer["loras"]
        assert row["name"] == "realfilter_v2" and row["found"] is True
        assert page.asked == [("realfilter_v2_final", False)], \
            "the card is read by the file's name, with Forge's preset filter off"

    def test_an_ambiguous_alias_is_read_as_a_file_name(self, library):
        """Two files with one alias: Forge forbids the alias and loads by file
        name, so the viewer does too."""
        first, second = Network("one", alias="shared"), Network("shared", alias="other")
        networks = registry(first, second, forbidden={"shared"})
        networks.available_network_aliases["shared"] = first
        page = Page({"one": card("one"), "shared": card("shared")})
        mc_lora_viewer.details(["shared"], networks=networks, page=page)
        assert page.asked == [("shared", False)]

    def test_a_name_forge_does_not_know_yet_is_looked_for_once(self, library):
        networks = registry(later=[Network("new_today")])
        page = Page({"new_today": card("new_today")})
        answer = mc_lora_viewer.details(["new_today", "typo"], networks=networks, page=page)
        assert [row["found"] for row in answer["loras"]] == [True, False]
        assert networks.rescans == [["new_today", "typo"]]

    def test_a_typo_is_not_found_and_carries_nothing(self, library):
        answer = mc_lora_viewer.details(["typo"], networks=registry(), page=Page({}))
        assert answer == {"loras": [{"name": "typo", "found": False, "picture": "",
                                     "title": "", "triggers": []}]}

    def test_names_are_answered_once_each_in_the_order_asked(self, library):
        networks = registry(Network("a"), Network("b"))
        page = Page({"a": card("a"), "b": card("b")})
        answer = mc_lora_viewer.details(["b", " a ", "b", ""], networks=networks, page=page)
        assert [row["name"] for row in answer["loras"]] == ["b", "a"]

    def test_the_request_is_bounded(self, library):
        names = [f"n{i}" for i in range(mc_lora_viewer.MAX_NAMES + 10)] + ["x" * 5000]
        answer = mc_lora_viewer.details(names, networks=registry(), page=Page({}))
        assert len(answer["loras"]) == mc_lora_viewer.MAX_NAMES

    def test_no_forge_registry_reads_as_not_found(self, library, monkeypatch):
        monkeypatch.setattr(mc_lora_viewer, "forge_networks", lambda: None)
        answer = mc_lora_viewer.details(["a"], page=Page({}))
        assert answer["loras"][0]["found"] is False


class TestTriggers:
    def test_civitai_words_then_the_activation_text_none_twice(self, library):
        library.records = [LibraryRecord(model_name="RealFilter",
                                         trigger_words=("realfilter", "film still"))]
        networks = registry(Network("rf"))
        page = Page({"rf": card("rf", activation="Film Still, 35mm photograph;\nmuted")})
        (row,) = mc_lora_viewer.details(["rf"], networks=networks, page=page)["loras"]
        assert row["triggers"] == ["realfilter", "film still", "35mm photograph", "muted"]
        assert row["title"] == "RealFilter"

    def test_the_library_is_asked_with_the_cached_hash_and_never_hashes(self, library,
                                                                       monkeypatch):
        monkeypatch.setattr(mc_lora_hashing, "known_hash", lambda path: "ab" * 32)
        networks = registry(Network("rf"))
        mc_lora_viewer.details(["rf"], networks=networks, page=Page({"rf": card("rf")}))
        assert library.asked[0]["sha256"] == "ab" * 32
        assert library.asked[0]["filename"] == "/loras/rf.safetensors"

    def test_an_unhashed_lora_is_still_looked_up_by_its_path(self, library):
        networks = registry(Network("rf"))
        mc_lora_viewer.details(["rf"], networks=networks, page=Page({"rf": card("rf")}))
        assert library.asked[0]["sha256"] == ""
        assert library.asked[0]["name"] == "rf"

    def test_a_card_that_cannot_be_read_still_answers_found(self, library):
        library.records = [LibraryRecord(trigger_words=("word",))]
        networks = registry(Network("rf"))
        (row,) = mc_lora_viewer.details(["rf"], networks=networks,
                                        page=Page({"rf": OSError("gone")}))["loras"]
        assert row["found"] is True and row["triggers"] == ["word"] and row["picture"] == ""


class TestPicture:
    def test_the_lora_browsers_own_picture_comes_first(self, library):
        networks = registry(Network("rf"))
        page = Page({"rf": card("rf", preview="./sd_extra_networks/thumb?filename=rf.png")})
        (row,) = mc_lora_viewer.details(["rf"], networks=networks, page=page)["loras"]
        assert row["picture"] == "./sd_extra_networks/thumb?filename=rf.png"

    def test_without_one_the_first_downloaded_example_picture(self, library, tmp_path,
                                                             monkeypatch):
        folder = tmp_path / "examples"
        folder.mkdir()
        (folder / "clip.mp4").write_bytes(b"not a picture")
        (folder / "photo.png").write_bytes(PNG)
        files = {"https://civitai/clip.mp4": folder / "clip.mp4",
                 "https://civitai/missing.jpeg": None,
                 "https://civitai/photo.png": folder / "photo.png"}
        monkeypatch.setattr(mc_lora_images, "find_local", lambda url, image_id="": files[url])
        monkeypatch.setattr(mc_lora_images, "thumbnail_dir", lambda: tmp_path / "thumbs")
        library.records = [LibraryRecord(entries=tuple(
            PromptEntry(text=f"prompt {i}", image_url=url, image_id=str(i))
            for i, url in enumerate(files)))]
        networks = registry(Network("rf"))
        page = Page({"rf": card("rf", preview="")})
        (row,) = mc_lora_viewer.details(["rf"], networks=networks, page=page)["loras"]
        assert row["picture"].startswith("data:image/webp;base64,")

    def test_without_either_there_is_no_picture(self, library):
        networks = registry(Network("rf"))
        (row,) = mc_lora_viewer.details(["rf"], networks=networks,
                                        page=Page({"rf": card("rf", preview=None)}))["loras"]
        assert row["picture"] == ""


class TestTheRoute:
    class App:
        def __init__(self, paths=()):
            self.routes = [types.SimpleNamespace(path=p) for p in paths]
            self.handlers = []

        def add_api_route(self, path, handler, methods):
            self.handlers.append((path, handler, methods))
            self.routes.append(types.SimpleNamespace(path=path))

    @pytest.fixture
    def with_fastapi(self):
        """FastAPI, or a stand-in with the two names where this container has none."""
        stubbed = []
        try:
            import fastapi  # noqa: F401
        except Exception:
            fastapi = types.ModuleType("fastapi")
            fastapi.Request = type("Request", (), {})
            responses = types.ModuleType("fastapi.responses")

            class JSONResponse:
                def __init__(self, content, status_code=200, headers=None):
                    self.content, self.status_code, self.headers = content, status_code, headers

            responses.JSONResponse = JSONResponse
            fastapi.responses = responses
            for name, module in (("fastapi", fastapi), ("fastapi.responses", responses)):
                stubbed.append(name)
                sys.modules[name] = module
            importlib.reload(mc_lora_viewer)
        yield sys.modules["fastapi"]
        if stubbed:
            for name in stubbed:
                sys.modules.pop(name, None)
            importlib.reload(mc_lora_viewer)

    def test_registered_once_as_a_get(self, with_fastapi):
        app = self.App()
        assert mc_lora_viewer.install(None, app) is True
        assert mc_lora_viewer.install(None, app) is True
        assert [(path, methods) for path, _handler, methods in app.handlers] == \
            [(mc_lora_viewer.ROUTE, ["GET"])]
        assert mc_lora_viewer.install(None, object()) is False

    def test_the_handler_is_a_plain_function_so_it_runs_on_the_thread_pool(self,
                                                                          with_fastapi):
        """A coroutine reading files holds every other request on the event loop;
        Starlette runs a plain ``def`` on its thread pool. The signature is the
        behaviour here."""
        app = self.App()
        mc_lora_viewer.install(None, app)
        handler = app.handlers[0][1]
        assert not inspect.iscoroutinefunction(handler)
        hints = typing.get_type_hints(handler)
        assert hints["request"] is sys.modules["fastapi"].Request

    def test_the_handler_answers_the_names_in_the_query(self, with_fastapi, monkeypatch):
        seen = []
        monkeypatch.setattr(mc_lora_viewer, "details",
                            lambda names: seen.append(list(names)) or {"loras": []})
        app = self.App()
        mc_lora_viewer.install(None, app)
        handler = app.handlers[0][1]
        query = types.SimpleNamespace(getlist=lambda key: ["a", "b"] if key == "name" else [])
        response = handler(types.SimpleNamespace(query_params=query))
        assert seen == [["a", "b"]]
        assert getattr(response, "status_code", 200) == 200

    def test_a_failure_is_a_sentence_not_a_traceback(self, with_fastapi, monkeypatch):
        def boom(names):
            raise RuntimeError("no")

        monkeypatch.setattr(mc_lora_viewer, "details", boom)
        app = self.App()
        mc_lora_viewer.install(None, app)
        query = types.SimpleNamespace(getlist=lambda key: ["a"])
        response = app.handlers[0][1](types.SimpleNamespace(query_params=query))
        assert response.status_code == 500

    def test_a_real_fastapi_app_answers_a_get(self, library):
        fastapi = pytest.importorskip("fastapi")
        testclient = pytest.importorskip("fastapi.testclient")
        app = fastapi.FastAPI()
        assert mc_lora_viewer.install(None, app) is True
        networks = registry(Network("rf"))
        page = Page({"rf": card("rf")})
        original = mc_lora_viewer.details
        mc_lora_viewer.details = lambda names: original(names, networks=networks, page=page)
        try:
            answer = testclient.TestClient(app).get(mc_lora_viewer.ROUTE,
                                                    params=[("name", "rf"), ("name", "typo")])
        finally:
            mc_lora_viewer.details = original
        assert answer.status_code == 200, answer.text
        assert [row["found"] for row in answer.json()["loras"]] == [True, False]
        assert answer.headers["cache-control"] == "no-store"
