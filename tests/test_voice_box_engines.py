"""The Voice Box's engines: the registry, the two contracts, and one engine at a time.

``mc_voice_box_engines`` is the only place that says which engine the Voice Box
renders with, the way ``mc_voice_engines`` is for Voice Chat (docs/18). Each
row names an adapter and a runtime, and the Voice Box reaches an engine only
through them -- so the contracts are held here for every row: what an adapter
offers the service and the page, what a runtime offers the render service and
the turn system, and that each row's names agree with its modules' own.

And the service itself names no engine: ``mc_voice_box.py`` holds no
"vibevoice" and imports no engine's module (the brief's own proof for the
seam), so a third engine is a row and two modules, nothing else.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

import mc_voice_box as box
import mc_voice_box_engines as engines
import mc_voice_box_fields as box_fields

ROOT = Path(__file__).resolve().parent.parent

ADAPTER = ("ID", "LABEL", "GUEST", "KIND", "MAX_TAKES", "FIELDS", "SETTINGS_DEFAULTS",
           "fields", "defaults", "clean", "refuse", "infotext", "capabilities", "options",
           "status", "public_status", "settings", "set_settings", "model_ids", "model_id",
           "need_vram_bytes", "need_ram_bytes", "note_peak", "refusal", "install",
           "install_from", "uninstall", "progress", "runtime_python", "worker_script",
           "worker_environment", "model_dir", "flash_attention_installed")
"""What the Voice Box, its routes and the engine's runtime read off an adapter."""

RUNTIME = ("card_key", "status", "rendering", "resident_bytes", "peak_observed",
           "ensure_started", "load", "render", "cancel", "unload", "evict", "stop",
           "shutdown", "stop_on_exit", "RenderJob", "Take", "RenderResult", "Voice", "Line")
"""What the render service, the turn system's guest and the WebUI's shutdown
read off a runtime."""


@pytest.fixture(autouse=True)
def _fresh(voice_root):
    box.forget()
    yield
    box.forget()


class TestTheRegistry:
    def test_two_engines_vibevoice_first_and_the_default(self):
        assert engines.ENGINES == ("vibevoice", "breeze")
        assert engines.DEFAULT_ENGINE == "vibevoice" == engines.active_id()
        assert [(row.label, row.guest) for row in engines.SPECS] == \
            [("VibeVoice", "VibeVoice"), ("Breeze TTS 2", "Breeze")]

    @pytest.mark.parametrize("row", engines.SPECS, ids=lambda row: row.id)
    def test_each_adapter_offers_the_contract_and_spells_its_row(self, row):
        adapter = importlib.import_module(row.adapter)
        missing = [name for name in ADAPTER if not hasattr(adapter, name)]
        assert missing == [], f"{row.adapter} lacks {missing}"
        assert (adapter.ID, adapter.LABEL, adapter.GUEST) == (row.id, row.label, row.guest)
        assert box_fields.check(adapter.FIELDS) == []
        assert box_fields.check(adapter.fields()) == []
        assert adapter.defaults() == box_fields.defaults(adapter.FIELDS)
        assert set(adapter.SETTINGS_DEFAULTS) >= {"card_uuid", "model_id", "keep_warm"}
        found = adapter.capabilities()
        assert set(found) >= {"speakers", "takes", "design", "directions", "voice", "events",
                              "languages"}
        assert set(found["voice"]) == {"sample", "transcript", "description", "instruction"}

    @pytest.mark.parametrize("row", engines.SPECS, ids=lambda row: row.id)
    def test_each_runtime_offers_the_contract_and_uses_the_neutral_types(self, row):
        runtime = importlib.import_module(row.runtime)
        missing = [name for name in RUNTIME if not hasattr(runtime, name)]
        assert missing == [], f"{row.runtime} lacks {missing}"
        assert runtime.RenderJob is box.render_types.RenderJob
        assert runtime.RenderResult is box.render_types.RenderResult

    def test_an_unknown_engine_is_a_refusal_and_never_a_fallback_for_a_named_one(self):
        with pytest.raises(engines.EngineError):
            engines.spec("kokoro")
        assert engines.known("Breeze") and not engines.known("kokoro")
        assert engines.label("kokoro") == "kokoro"


class TestTheServiceNamesNoEngine:
    def test_no_engine_is_named_in_the_voice_box(self):
        """The brief's proof for the seam: ``grep -c vibevoice mc_voice_box.py``."""
        text = (ROOT / "mc_voice_box.py").read_text(encoding="utf-8")
        assert text.count("vibevoice") == 0
        assert text.count("breeze") == 0

    def test_it_imports_no_engines_module(self):
        tree = ast.parse((ROOT / "mc_voice_box.py").read_text(encoding="utf-8"))
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module)
        engine_modules = {name for row in engines.SPECS for name in (row.adapter, row.runtime)}
        assert not names & engine_modules
        assert not any(name.split(".")[0] in ("vibevoice_worker", "breeze_worker")
                       for name in names)


class TestOneEngineAtATime:
    def test_after_a_switch_neither_runtime_holds_a_card(self):
        """Nothing of the incoming engine is started by a switch, and the outgoing
        engine has let go of every card before the choice is written."""
        vibevoice = engines.runtime("vibevoice")
        breeze = engines.runtime("breeze")
        assert engines.select("breeze") == {"changed": True, "engine": "breeze"}
        assert vibevoice.status()["cards"] == {} and breeze.status()["cards"] == {}
        assert box.settings()["engine"] == "breeze" and engines.active().id == "breeze"
        assert engines.select("vibevoice")["changed"] is True
        assert breeze.status()["cards"] == {}

    def test_a_switch_is_refused_while_anything_is_queued(self, monkeypatch):
        monkeypatch.setattr(box, "busy", lambda: True)
        with pytest.raises(engines.Busy) as raised:
            engines.select("breeze")
        assert raised.value.status == 409 and engines.active_id() == "vibevoice"

    def test_the_settings_route_cannot_switch_it(self):
        box.set_settings({"engine": "breeze"})
        assert engines.active_id() == "breeze", "the service's own store is where it lives"
        with pytest.raises(box.VoiceBoxError):
            box.set_settings({"engine": "kokoro"})

    def test_the_shutdown_door_reaches_every_engine(self):
        assert [label for label, _stop in engines.shutdowns()] == ["VibeVoice", "Breeze TTS 2"]
