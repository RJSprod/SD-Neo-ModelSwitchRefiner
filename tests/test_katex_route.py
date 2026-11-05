"""The vendored KaTeX, and the one route that serves it to the assistant's panel.

``vendor/katex`` is KaTeX 0.16.10 as its package ships it, the files listed in
``manifest.json`` with their digests; ``mc_katex`` serves exactly the script,
the stylesheet and the fonts among them, and nothing else from anywhere.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

import mc_katex

ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / "vendor" / "katex"
MANIFEST = json.loads((VENDOR / "manifest.json").read_text(encoding="utf-8"))


class TestTheVendoredFiles:
    def test_every_file_is_the_one_the_manifest_names(self):
        """An edit to a vendored file -- or a file dropped in beside them --
        fails here: what is served is what the package shipped."""
        listed = MANIFEST["sha256"]
        present = {path.relative_to(VENDOR).as_posix() for path in VENDOR.rglob("*") if path.is_file()}

        assert present - {"manifest.json", "README.md"} == set(listed)
        for name, digest in listed.items():
            assert hashlib.sha256((VENDOR / name).read_bytes()).hexdigest() == digest, name

    def test_it_is_the_version_gradio_bundles_and_the_panel_asks_for(self):
        script = (ROOT / "javascript" / "forge_assistant_markdown.js").read_text(encoding="utf-8")

        assert MANIFEST["version"] == mc_katex.VERSION == "0.16.10"
        assert f'const KATEX_VERSION = "{mc_katex.VERSION}";' in script
        assert f'"{mc_katex.VERSION}"' in (VENDOR / "katex.min.js").read_text(encoding="utf-8")

    def test_the_licence_travels_with_it(self):
        assert MANIFEST["license"] == "MIT"
        assert (VENDOR / "LICENSE").read_text(encoding="utf-8").startswith("The MIT License")

    def test_every_font_the_stylesheet_asks_for_first_is_served(self):
        """The stylesheet lists a woff2, a woff and a ttf of each font; a
        browser takes the first it can use, and every one it can is here."""
        css = (VENDOR / "katex.min.css").read_text(encoding="utf-8")
        first = re.findall(r"src:url\((fonts/[^)]+?\.woff2)\)", css)

        assert len(first) == 20
        assert set(first) <= set(mc_katex.served())


class TestTheRoute:
    def test_it_serves_the_script_the_stylesheet_and_the_fonts(self):
        script = mc_katex.file_for("katex.min.js")
        style = mc_katex.file_for("katex.min.css")
        font = mc_katex.file_for("fonts/KaTeX_Main-Regular.woff2")

        assert script[1].startswith("text/javascript") and script[0].startswith(b"!function")
        assert style[1].startswith("text/css")
        assert font[1] == "font/woff2" and font[0][:4] == b"wOF2"

    @pytest.mark.parametrize("name", [
        "LICENSE", "manifest.json", "README.md", "../mc_katex.py", "fonts/../../mc_katex.py",
        "/etc/passwd", "..\\mc_katex.py", "fonts/KaTeX_Main-Regular.woff", "katex.js", "",
        "fonts", "fonts/", "%2e%2e/mc_katex.py",
    ])
    def test_it_serves_nothing_else(self, name):
        assert mc_katex.file_for(name) is None

    def test_it_is_registered_when_the_app_starts(self):
        source = (ROOT / "scripts" / "model_chain.py").read_text(encoding="utf-8")

        assert "script_callbacks.on_app_started(mc_katex.install)" in source

    def test_installing_twice_registers_once_and_a_host_without_an_app_is_no_error(self):
        class App:
            def __init__(self):
                self.routes = []

            def add_api_route(self, path, handler, methods):
                self.routes.append(type("Route", (), {"path": path, "handler": handler})())

        app = App()
        if mc_katex.Response is None:
            assert mc_katex.install(None, app) is False
            return
        assert mc_katex.install(None, app) is True
        assert mc_katex.install(None, app) is True
        assert [route.path for route in app.routes] == [mc_katex.ROUTE]
        assert mc_katex.install(None, None) is False

    def test_a_real_app_answers_with_the_bytes_the_type_and_a_years_cache(self):
        fastapi = pytest.importorskip("fastapi")
        testclient = pytest.importorskip("fastapi.testclient")

        app = fastapi.FastAPI()
        assert mc_katex.install(None, app) is True
        client = testclient.TestClient(app)

        answer = client.get(mc_katex.PREFIX + "katex.min.css")
        assert answer.status_code == 200
        assert answer.headers["content-type"].startswith("text/css")
        assert answer.headers["cache-control"] == "public, max-age=31536000, immutable"
        assert answer.content == (VENDOR / "katex.min.css").read_bytes()

        font = client.get(mc_katex.PREFIX + "fonts/KaTeX_Math-Italic.woff2")
        assert font.status_code == 200 and font.headers["content-type"] == "font/woff2"

        for refused in ("LICENSE", "../mc_katex.py", "fonts/..%2F..%2Fmc_katex.py", "manifest.json"):
            assert client.get(mc_katex.PREFIX + refused).status_code == 404, refused
        # Another version's address is not this route's at all.
        assert client.get("/model-chain/katex/0.16.9/katex.min.js").status_code == 404
