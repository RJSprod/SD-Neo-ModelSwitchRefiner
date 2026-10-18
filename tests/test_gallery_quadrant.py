"""Gallery Quadrant (mc_gallery_quadrant.py): the settings section, the file a
gallery's picture URL names, the copy into the folder, and the route's answers.

The script's side -- the zones, the clicks, the overlay, the request the
folder destination makes -- is tests/test_gallery_quadrant_js.py, in Chromium.
"""

from __future__ import annotations

import importlib
import os
import sys
import types
import typing
from pathlib import Path

import pytest

import mc_gallery_quadrant as quadrant

ROOT = Path(__file__).resolve().parent.parent


def host_shared():
    """The stand-in host's ``modules.shared``, with its OptionInfo."""
    return sys.modules["modules.shared"]


def options(**values):
    """An ``opts`` of the given settings and nothing else."""
    return types.SimpleNamespace(**values)


def picture(folder: Path, name: str = "00001-42.png", content: bytes = b"\x89PNG one") -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_bytes(content)
    return path


def file_url(path: Path, host: str = "http://127.0.0.1:7860") -> str:
    return host + "/file=" + str(path)


# --------------------------------------------------------------------------- #
# The settings
# --------------------------------------------------------------------------- #


class TestTheSettings:
    def test_the_section_holds_the_switch_the_destination_and_the_folder(self):
        found = quadrant.settings(host_shared())

        assert list(found) == [quadrant.OPT_ENABLED, quadrant.OPT_DESTINATION, quadrant.OPT_FOLDER]
        assert found[quadrant.OPT_ENABLED].default is True, "on by default, as the standalone was"
        assert found[quadrant.OPT_DESTINATION].default == quadrant.DESTINATION_BROWSER
        assert found[quadrant.OPT_FOLDER].default == ""
        assert quadrant.SECTION == ("model_chain_gallery_quadrant", "Model Chain: Gallery Quadrant")
        assert quadrant.DESTINATIONS == (quadrant.DESTINATION_BROWSER, quadrant.DESTINATION_FOLDER)

    def test_the_main_script_registers_the_section_and_the_route(self):
        script = (ROOT / "scripts" / "model_chain.py").read_text(encoding="utf-8")

        assert "import mc_gallery_quadrant" in script
        assert "shared.options_section(mc_gallery_quadrant.SECTION, mc_gallery_quadrant.settings(shared))" in script
        assert "script_callbacks.on_app_started(mc_gallery_quadrant.install)" in script

    def test_the_destination_reads_as_the_browser_unless_the_folder_is_chosen(self):
        assert quadrant.destination_of(options()) == quadrant.DESTINATION_BROWSER
        assert quadrant.destination_of(options(**{quadrant.OPT_DESTINATION: "anything else"})) \
            == quadrant.DESTINATION_BROWSER
        assert quadrant.destination_of(options(**{quadrant.OPT_DESTINATION: quadrant.DESTINATION_FOLDER})) \
            == quadrant.DESTINATION_FOLDER


# --------------------------------------------------------------------------- #
# What a gallery URL names
# --------------------------------------------------------------------------- #


class TestWhatAGalleryUrlNames:
    @pytest.mark.parametrize("src, expected", [
        ("/file=/data/outputs/txt2img-images/2026-10-17/00001-1.png",
         "/data/outputs/txt2img-images/2026-10-17/00001-1.png"),
        ("http://127.0.0.1:7860/file=/data/outputs/a.png?1234", "/data/outputs/a.png"),
        ("https://forge.local:7860/file=C%3A%5CForge%5Coutputs%5Ca%20b.png", "C:\\Forge\\outputs\\a b.png"),
        ("/file=/data/outputs/a.png#x", "/data/outputs/a.png"),
        ("  /file=/data/outputs/a.png  ", "/data/outputs/a.png"),
        ("blob:http://127.0.0.1:7860/abc-def", ""),
        ("data:image/png;base64,iVBORw0KGgo=", ""),
        ("/outputs/a.png", ""),
        ("", ""),
        (None, ""),
    ])
    def test_a_gradio_file_url_names_its_file_and_nothing_else_does(self, src, expected):
        assert quadrant.file_of_url(src) == expected


class TestThePictureTheGalleryShows:
    def test_an_existing_file_under_an_output_folder_is_resolved(self, tmp_path):
        outputs = tmp_path / "outputs"
        found = picture(outputs / "txt2img-images" / "2026-10-17")

        resolved = quadrant.resolve_source(file_url(found), [outputs])

        assert resolved == Path(os.path.realpath(str(found)))

    def test_outside_every_root_is_refused(self, tmp_path):
        outputs = tmp_path / "outputs"
        outputs.mkdir()
        elsewhere = picture(tmp_path / "elsewhere")
        with pytest.raises(quadrant.SaveRefused) as refused:
            quadrant.resolve_source(file_url(elsewhere), [outputs])
        assert refused.value.status == 403 and "output folders" in refused.value.reason

        # A path that climbs out of a root is read for where it lands.
        climbing = outputs / ".." / "elsewhere" / elsewhere.name
        with pytest.raises(quadrant.SaveRefused) as climbed:
            quadrant.resolve_source(file_url(climbing), [outputs])
        assert climbed.value.status == 403

    def test_a_file_that_is_gone_is_404(self, tmp_path):
        outputs = tmp_path / "outputs"
        outputs.mkdir()
        with pytest.raises(quadrant.SaveRefused) as refused:
            quadrant.resolve_source(file_url(outputs / "gone.png"), [outputs])
        assert refused.value.status == 404

    def test_a_folder_under_a_root_is_not_a_picture(self, tmp_path):
        outputs = tmp_path / "outputs"
        (outputs / "a-folder").mkdir(parents=True)
        with pytest.raises(quadrant.SaveRefused) as refused:
            quadrant.resolve_source(file_url(outputs / "a-folder"), [outputs])
        assert refused.value.status == 404

    def test_not_a_file_url_is_400(self, tmp_path):
        with pytest.raises(quadrant.SaveRefused) as refused:
            quadrant.resolve_source("blob:http://127.0.0.1:7860/abc", [tmp_path])
        assert refused.value.status == 400

    def test_the_roots_are_forges_output_options_and_gradios_temporary_folder(self, tmp_path):
        data = tmp_path / "data"
        temp = tmp_path / "temp" / "gradio"
        opts = options(outdir_samples="", outdir_txt2img_samples="outputs/txt2img-images",
                       outdir_img2img_samples=str(tmp_path / "abs" / "img2img"),
                       outdir_save="log/images", outdir_grids=None)

        roots = quadrant.output_roots(opts, data_dir=data, temp_dir=temp)

        realpath = lambda p: Path(os.path.realpath(str(p)))
        assert roots == [realpath(data / "outputs/txt2img-images"), realpath(tmp_path / "abs" / "img2img"),
                         realpath(data / "log/images"), realpath(temp)]

    def test_gradios_temporary_folder_is_its_environment_variable_or_the_systems(self, monkeypatch, tmp_path):
        monkeypatch.setenv("GRADIO_TEMP_DIR", str(tmp_path / "named"))
        assert quadrant.gradio_temp_directory() == tmp_path / "named"
        monkeypatch.delenv("GRADIO_TEMP_DIR")
        assert quadrant.gradio_temp_directory().name == "gradio"


# --------------------------------------------------------------------------- #
# The copy
# --------------------------------------------------------------------------- #


class TestTheCopy:
    def test_copies_under_its_own_name_and_makes_the_folder(self, tmp_path):
        source = picture(tmp_path / "outputs")
        folder = tmp_path / "keep" / "best"

        written = quadrant.save_copy(source, folder)

        assert written == folder / source.name
        assert written.read_bytes() == source.read_bytes()

    def test_the_same_picture_twice_is_one_file(self, tmp_path):
        source = picture(tmp_path / "outputs")
        folder = tmp_path / "keep"

        first = quadrant.save_copy(source, folder)
        second = quadrant.save_copy(source, folder)

        assert first == second
        assert sorted(p.name for p in folder.iterdir()) == [source.name]

    def test_a_different_picture_of_the_same_name_gets_a_number(self, tmp_path):
        folder = tmp_path / "keep"
        first = quadrant.save_copy(picture(tmp_path / "one", content=b"one"), folder)
        second = quadrant.save_copy(picture(tmp_path / "two", content=b"two"), folder)
        third = quadrant.save_copy(picture(tmp_path / "three", content=b"three"), folder)

        assert [p.name for p in (first, second, third)] == ["00001-42.png", "00001-42-1.png", "00001-42-2.png"]
        assert second.read_bytes() == b"two" and third.read_bytes() == b"three"
        assert first.read_bytes() == b"one", "never overwritten"

    def test_a_folder_that_is_a_file_is_refused(self, tmp_path):
        source = picture(tmp_path / "outputs")
        blocker = tmp_path / "keep"
        blocker.write_text("not a folder")
        with pytest.raises(quadrant.SaveRefused) as refused:
            quadrant.save_copy(source, blocker)
        assert refused.value.status == 500


class TestTheFolderSetting:
    def test_an_absolute_folder_is_taken_as_given_and_a_relative_one_under_the_data_directory(self, tmp_path):
        assert quadrant.resolve_folder(str(tmp_path / "keep"), tmp_path / "data") \
            == Path(os.path.realpath(str(tmp_path / "keep")))
        assert quadrant.resolve_folder("keep/best", tmp_path / "data") \
            == Path(os.path.realpath(str(tmp_path / "data" / "keep" / "best")))

    def test_an_empty_setting_is_refused_with_where_to_set_it(self):
        with pytest.raises(quadrant.SaveRefused) as refused:
            quadrant.resolve_folder("   ")
        assert refused.value.status == 400
        assert "Model Chain: Gallery Quadrant" in refused.value.reason


# --------------------------------------------------------------------------- #
# The route
# --------------------------------------------------------------------------- #


def folder_opts(tmp_path, folder="keep", destination=quadrant.DESTINATION_FOLDER):
    return options(**{quadrant.OPT_DESTINATION: destination, quadrant.OPT_FOLDER: folder,
                      "outdir_txt2img_samples": "outputs/txt2img-images"})


SAME_ORIGIN = {"host": "127.0.0.1:7860", "origin": "http://127.0.0.1:7860"}


class TestTheRoute:
    def test_a_save_lands_in_the_folder_and_answers_its_path(self, tmp_path):
        data = tmp_path / "data"
        source = picture(data / "outputs" / "txt2img-images" / "2026-10-17")

        status, body = quadrant.handle({"src": file_url(source)}, SAME_ORIGIN,
                                       opts=folder_opts(tmp_path), data_dir=data)

        assert status == 200 and body["ok"] is True, body
        written = Path(body["path"])
        assert written == Path(os.path.realpath(str(data / "keep"))) / source.name
        assert written.read_bytes() == source.read_bytes()
        assert body["name"] == source.name and Path(body["folder"]) == written.parent

    def test_the_browser_destination_is_refused(self, tmp_path):
        data = tmp_path / "data"
        source = picture(data / "outputs" / "txt2img-images")

        status, body = quadrant.handle({"src": file_url(source)}, SAME_ORIGIN,
                                       opts=folder_opts(tmp_path, destination=quadrant.DESTINATION_BROWSER),
                                       data_dir=data)

        assert status == 400 and "browser" in body["error"]
        assert not (data / "keep").exists()

    def test_an_empty_folder_setting_is_refused_and_nothing_is_written(self, tmp_path):
        data = tmp_path / "data"
        source = picture(data / "outputs" / "txt2img-images")

        status, body = quadrant.handle({"src": file_url(source)}, SAME_ORIGIN,
                                       opts=folder_opts(tmp_path, folder=""), data_dir=data)

        assert status == 400 and "That folder" in body["error"]

    def test_a_picture_outside_the_output_folders_is_refused(self, tmp_path):
        data = tmp_path / "data"
        elsewhere = picture(tmp_path / "elsewhere")

        status, body = quadrant.handle({"src": file_url(elsewhere)}, SAME_ORIGIN,
                                       opts=folder_opts(tmp_path), data_dir=data)

        assert status == 403 and not (data / "keep").exists()

    def test_a_request_that_is_not_a_save_is_refused(self, tmp_path):
        status, body = quadrant.handle(None, SAME_ORIGIN, opts=folder_opts(tmp_path), data_dir=tmp_path)
        assert status == 400
        status, body = quadrant.handle({"src": ""}, SAME_ORIGIN, opts=folder_opts(tmp_path), data_dir=tmp_path)
        assert status == 400

    def test_a_request_from_elsewhere_is_refused_before_anything_is_read(self, tmp_path):
        data = tmp_path / "data"
        source = picture(data / "outputs" / "txt2img-images")
        for headers in ({"host": "127.0.0.1:7860"},
                        {"host": "127.0.0.1:7860", "origin": "http://evil.example"},
                        {"host": "127.0.0.1:7860", "referer": "http://evil.example/page"},
                        {"host": "127.0.0.1:7860", "origin": "null"}):
            status, body = quadrant.handle({"src": file_url(source)}, headers,
                                           opts=folder_opts(tmp_path), data_dir=data)
            assert status == 403, headers
        assert not (data / "keep").exists()
        # A referer of this WebUI's is a browser's too.
        status, _ = quadrant.handle({"src": file_url(source)},
                                    {"host": "127.0.0.1:7860", "referer": "http://127.0.0.1:7860/?__theme=dark"},
                                    opts=folder_opts(tmp_path), data_dir=data)
        assert status == 200

    def test_install_registers_the_route_once_and_only_on_an_app(self):
        assert quadrant.install(None, None) is False
        assert quadrant.install(None, object()) is False

        class App:
            def __init__(self, paths=()):
                self.routes = [types.SimpleNamespace(path=p) for p in paths]
                self.added = []

            def add_api_route(self, path, handler, methods):
                self.added.append((path, methods))
                self.routes.append(types.SimpleNamespace(path=path))

        already = App([quadrant.ROUTE])
        assert quadrant.install(None, already) is True and already.added == []

        app = App()
        installed = quadrant.install(None, app)
        try:
            import fastapi  # noqa: F401
        except Exception:
            assert installed is False and app.added == [], "no FastAPI, no route, nothing raised"
        else:
            assert installed is True and app.added == [(quadrant.ROUTE, ["POST"])]
            assert quadrant.install(None, app) is True and len(app.added) == 1

    def test_the_handlers_request_annotation_resolves_where_fastapi_looks(self):
        """FastAPI reads a postponed annotation as a string and evaluates it
        against the handler's module globals. With ``Request`` imported inside
        ``install()`` the string resolved to nothing, FastAPI took ``request``
        for a required query parameter, and every save answered HTTP 422 --
        the user's report from a phone. Where this container has no FastAPI, a
        stand-in with the two names stands for it."""

        class App:
            def __init__(self):
                self.routes, self.handlers = [], []

            def add_api_route(self, path, handler, methods):
                self.handlers.append(handler)
                self.routes.append(types.SimpleNamespace(path=path))

        stubbed = []
        try:
            import fastapi  # noqa: F401
        except Exception:
            fastapi = types.ModuleType("fastapi")
            fastapi.Request = type("Request", (), {})
            responses = types.ModuleType("fastapi.responses")
            responses.JSONResponse = type("JSONResponse", (), {})
            fastapi.responses = responses
            for name, module in (("fastapi", fastapi), ("fastapi.responses", responses)):
                stubbed.append(name)
                sys.modules[name] = module
            importlib.reload(quadrant)
        try:
            app = App()
            assert quadrant.install(None, app) is True
            (handler,) = app.handlers
            # typing.get_type_hints evaluates the strings against handler.__globals__,
            # exactly as FastAPI's get_typed_signature does.
            hints = typing.get_type_hints(handler)
            assert hints["request"] is sys.modules["fastapi"].Request
            assert quadrant.Request is sys.modules["fastapi"].Request
        finally:
            if stubbed:
                for name in stubbed:
                    sys.modules.pop(name, None)
                importlib.reload(quadrant)
                assert quadrant.Request is None

    def test_a_browsers_post_reaches_the_handler_on_a_real_fastapi_app(self, tmp_path, monkeypatch):
        """The whole route under FastAPI's own request parsing, where it is
        installed: the body is read, the picture is copied byte for byte (its
        PNG info with it), never a 422 about a missing ``request`` query
        parameter."""
        fastapi = pytest.importorskip("fastapi")
        testclient = pytest.importorskip("fastapi.testclient")
        data = tmp_path / "data"
        source = picture(data / "outputs" / "txt2img-images" / "2026-10-17")
        folder = tmp_path / "saved"
        monkeypatch.setattr(quadrant, "data_directory", lambda: data)
        host_shared().opts = folder_opts(tmp_path, folder=str(folder))
        try:
            app = fastapi.FastAPI()
            assert quadrant.install(None, app) is True
            client = testclient.TestClient(app)
            answer = client.post(quadrant.ROUTE, json={"src": file_url(source, "http://testserver")},
                                 headers={"Origin": "http://testserver"})
        finally:
            del host_shared().opts
        assert answer.status_code == 200, answer.text
        assert answer.json()["ok"] is True
        assert (folder / source.name).read_bytes() == source.read_bytes()
