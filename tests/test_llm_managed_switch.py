"""Switching backbones, under the rule that exactly one model may be resident.

Section 8 of the design intent is one sentence long in effect -- *at no point
may two llama-server model processes intentionally coexist* -- and everything in
this file is a way of failing to keep it. Two models on a 24 GB card is not a
degraded experience, it is an out-of-memory error in whatever the user was doing
at the time, so the order (refuse if busy, stop, *observe* the stop, start,
prove it answers) is asserted step by step rather than by its outcome.

The other half is the rollback. A user who presses Use on a new backbone and
gets a model that will not start must end up back on the one they had, running,
with the download they paid for still on disk.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import pytest

import mc_broker
import mc_llm_managed_models as managed
import mc_llm_paths
import mc_llm_runtime


MODEL_SHA = "a" * 64
MMPROJ_SHA = "b" * 64


@pytest.fixture(autouse=True)
def root(tmp_path, monkeypatch, host):
    install = tmp_path / "install"
    (install / "data").mkdir(parents=True)
    monkeypatch.setattr(mc_llm_paths, "data_root", lambda: install)
    managed._registry_cache = None
    mc_broker.clear()
    yield install
    mc_broker.clear()
    managed._registry_cache = None


@pytest.fixture
def registry(tmp_path, monkeypatch):
    """Two catalogue entries, so a switch has somewhere to switch *to*."""
    def row(identifier, label, profile):
        return {
            "id": identifier, "label": label, "role": "Recommended",
            "group": "Recommended", "family": "Test", "profile": profile,
            "multimodal": True,
            "source_url": "https://huggingface.co/example/test",
            "repo_id": "example/test", "revision": "main",
            "model": {"filename": f"{identifier}-Q4_K_M.gguf", "sha256": MODEL_SHA,
                      "bytes": None, "display_size": "~7.4 GB"},
            "projector": {"filename": f"mmproj-{identifier}.gguf", "sha256": MMPROJ_SHA,
                          "bytes": None, "display_size": "175 MB"},
        }

    document = {"version": 1, "registry_version": "test-1", "models": [
        row("first-model", "First Model", "gemma4-12b-qat-balanced"),
        row("second-model", "Second Model", "qwen35-9b-aggressive"),
    ]}
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.setattr(managed, "REGISTRY_PATH", path)
    managed._registry_cache = None
    return document


def install_bundle(root: Path, identifier: str, profile: str) -> Path:
    """A verified bundle, put on disk directly.

    The switch does not care how a bundle arrived -- that is
    ``test_llm_managed_download``'s subject -- and building one here without a
    fake HTTP server keeps these tests about the one thing they are for.
    """
    bundle = root / "models" / "managed" / identifier
    bundle.mkdir(parents=True, exist_ok=True)
    (bundle / "model.gguf").write_bytes(b"weights")
    (bundle / "mmproj.gguf").write_bytes(b"projector")
    (bundle / "installed.json").write_text(json.dumps({
        "schema": 1, "model_id": identifier, "registry_version": "test-1",
        "revision": "main", "profile": profile, "profile_version": "1",
        "artifacts": {"model": {"filename": f"{identifier}-Q4_K_M.gguf",
                                "stored_as": "model.gguf", "sha256": MODEL_SHA},
                      "projector": {"filename": f"mmproj-{identifier}.gguf",
                                    "stored_as": "mmproj.gguf", "sha256": MMPROJ_SHA}},
        "installed_at": 0,
    }), encoding="utf-8")
    return bundle


def state_of(root: Path) -> dict:
    path = root / "data" / "setup-state.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def write_state(root: Path, **values) -> None:
    path = root / "data" / "setup-state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values), encoding="utf-8")


class FakeRuntime:
    """One llama-server, and a log of everything asked of it in order.

    ``started`` never exceeding one at a time is the invariant the whole
    feature exists to keep, so it is asserted by the fake rather than by each
    test remembering to.
    """

    def __init__(self, *, fails=False, exits=True, up=True):
        self.events: list[str] = []
        self.fails = fails
        self.exits = exits
        self.up = up
        self.servers = 0
        self.loaded = ""

    def running(self):
        return self.up

    def stop(self):
        self.events.append("stop")
        if self.exits:
            self.up = False
            self.servers = 0

    def client(self, needs_vision=False, reserve=0):
        # The real signature, so a caller that passes what the modes pass --
        # ``mc_llm_sessions._client`` passes both -- is exercised rather than
        # accommodated.
        #
        # A warm server for the model that is selected is handed back, exactly
        # as the real runtime does. Starting a *different* one while that server
        # is still up is the thing that must never happen, so it is the thing
        # this fake refuses.
        wanted = managed.selection().identifier
        if self.servers and self.loaded == wanted:
            self.events.append("reuse")
            return FakeClient(self)
        assert self.servers == 0, "a second llama-server was started before the first stopped"
        self.events.append("start")
        if self.fails:
            raise RuntimeError("llama-server exited before becoming ready")
        self.servers = 1
        self.up = True
        self.loaded = wanted
        return FakeClient(self)


class FakeClient:
    def __init__(self, owner, answers=True):
        self.owner = owner
        self.answers = answers

    def stream_chat(self, messages, max_tokens, seed, on_text, **_kwargs):
        self.owner.events.append(f"smoke:{max_tokens}")
        if not self.answers:
            raise RuntimeError("the model will not answer")
        on_text("ready")
        return "ready"


@pytest.fixture
def runtime(monkeypatch):
    fake = FakeRuntime()
    monkeypatch.setattr(mc_llm_runtime, "runtime", fake)
    return fake


@contextmanager
def _no_gpu(*_args, **_kwargs):
    """A workload that cannot be taken: something else has the card."""
    yield None


class TestASuccessfulSwitch:
    def test_it_records_the_backbone_its_files_and_its_profile(self, root, registry, runtime):
        install_bundle(root, "first-model", "gemma4-12b-qat-balanced")
        write_state(root, runtime="runtime/llama-server", model="", mmproj="")

        managed.use("first-model")
        state = state_of(root)

        assert state["source"] == "managed"
        assert state["managed_model_id"] == "first-model"
        assert state["managed_profile"] == "gemma4-12b-qat-balanced"
        assert state["managed_profile_version"] == "1"
        assert state["model"] == "models/managed/first-model/model.gguf"
        assert state["mmproj"] == "models/managed/first-model/mmproj.gguf"

    def test_the_status_line_shows_the_catalogue_name_and_not_model_gguf(
            self, root, registry, runtime):
        """Every bundle's weights are called ``model.gguf`` on disk, so the
        filename alone would tell a reader nothing about which one is running."""
        install_bundle(root, "first-model", "gemma4-12b-qat-balanced")
        write_state(root, runtime="runtime/llama-server")

        managed.use("first-model")

        assert state_of(root)["quantization"] == "Q4_K_M"

    def test_it_keeps_the_hardware_the_installation_already_learned(self, root, registry,
                                                                    runtime):
        """A profile controls model behaviour; the broker controls where the
        model fits. Nothing about this machine is a decision the catalogue gets
        to make."""
        install_bundle(root, "first-model", "gemma4-12b-qat-balanced")
        write_state(root, runtime="runtime/llama-server", gpu_index=1, gpu_device="CUDA1",
                    mode="mixed", gpu_layers="0", gpu_name="A card")

        managed.use("first-model")
        state = state_of(root)

        assert (state["gpu_index"], state["gpu_device"]) == (1, "CUDA1")
        assert (state["mode"], state["gpu_layers"]) == ("mixed", "0")

    def test_the_old_server_is_stopped_before_the_new_one_starts(self, root, registry,
                                                                 runtime):
        install_bundle(root, "first-model", "gemma4-12b-qat-balanced")
        write_state(root, runtime="runtime/llama-server")

        managed.use("first-model")

        assert runtime.events == ["stop", "start", "smoke:8"]

    def test_it_proves_the_new_backbone_answers_before_calling_it_active(
            self, root, registry, runtime):
        """A server that reaches /health has loaded a file. It has not
        necessarily loaded a chat model or applied a template."""
        install_bundle(root, "first-model", "gemma4-12b-qat-balanced")
        write_state(root, runtime="runtime/llama-server")

        managed.use("first-model")

        assert f"smoke:{managed.SMOKE_TOKENS}" in runtime.events
        assert managed.status(managed.entry("first-model")).state == managed.ACTIVE

    def test_switching_between_two_managed_backbones_leaves_one_resident(
            self, root, registry, runtime):
        install_bundle(root, "first-model", "gemma4-12b-qat-balanced")
        install_bundle(root, "second-model", "qwen35-9b-aggressive")
        write_state(root, runtime="runtime/llama-server")

        managed.use("first-model")
        managed.use("second-model")

        assert runtime.servers == 1
        assert managed.selection().identifier == "second-model"
        assert managed.status(managed.entry("first-model")).state == managed.INSTALLED

    def test_every_mode_resolves_it_because_they_all_read_one_config(self, root, registry,
                                                                     runtime):
        """Section 11's global-backbone criterion, checked where it is actually
        decided: there is one ``config()`` and one ``runtime.client``, so a
        switch reaching them reaches Prompt Studio, Conversation, MiniMax,
        Krea 2 and Creative Mode at once."""
        import mc_llm_sessions

        install_bundle(root, "first-model", "gemma4-12b-qat-balanced")
        write_state(root, runtime="runtime/llama-server")

        managed.use("first-model")
        configuration = mc_llm_runtime.config()

        assert configuration.model == root / "models" / "managed" / "first-model" / "model.gguf"
        assert configuration.managed_id == "first-model"
        assert mc_llm_sessions._client(False) is not None


class TestItRefusesRatherThanTearingWeightsOut:
    def test_an_llm_generation_in_progress_stops_the_switch(self, root, registry, runtime,
                                                            monkeypatch):
        install_bundle(root, "first-model", "gemma4-12b-qat-balanced")
        write_state(root, runtime="runtime/llama-server", model="/elsewhere/mine.gguf")
        monkeypatch.setattr(mc_broker, "active",
                            lambda: mc_broker.Active(mc_broker.FAMILY_LLM, "a Krea prompt", 0))

        with pytest.raises(managed.Busy) as raised:
            managed.use("first-model")

        assert "a Krea prompt" in str(raised.value)
        assert runtime.events == []
        assert state_of(root)["model"] == "/elsewhere/mine.gguf"

    def test_an_image_generation_holding_the_card_stops_the_switch(self, root, registry,
                                                                   runtime, monkeypatch):
        install_bundle(root, "first-model", "gemma4-12b-qat-balanced")
        write_state(root, runtime="runtime/llama-server", model="/elsewhere/mine.gguf")
        monkeypatch.setattr(mc_broker, "workload", _no_gpu)

        with pytest.raises(managed.Busy):
            managed.use("first-model")

        assert runtime.events == []
        assert state_of(root)["model"] == "/elsewhere/mine.gguf"

    def test_a_backbone_that_is_not_downloaded_cannot_be_applied(self, root, registry,
                                                                 runtime):
        write_state(root, runtime="runtime/llama-server", model="/elsewhere/mine.gguf")

        with pytest.raises(managed.ManagedError):
            managed.use("first-model")

        assert runtime.events == []
        assert "managed_model_id" not in state_of(root)

    def test_an_id_that_is_not_in_the_catalogue_cannot_be_applied(self, root, registry,
                                                                  runtime):
        with pytest.raises(managed.ManagedError):
            managed.use("../../etc")

        assert runtime.events == []

    def test_a_server_that_will_not_exit_stops_the_switch(self, root, registry, monkeypatch):
        """Never start the second one on a guess. If the first is still there,
        the switch has not happened and says so."""
        fake = FakeRuntime(exits=False)
        monkeypatch.setattr(mc_llm_runtime, "runtime", fake)
        monkeypatch.setattr(managed, "RESIDENT_STOP_TIMEOUT", 0.05)
        install_bundle(root, "first-model", "gemma4-12b-qat-balanced")
        write_state(root, runtime="runtime/llama-server", model="/elsewhere/mine.gguf")

        with pytest.raises(managed.ManagedError) as raised:
            managed.use("first-model")

        assert "has not exited" in str(raised.value)
        assert "start" not in fake.events
        assert state_of(root)["model"] == "/elsewhere/mine.gguf"


class TestRollback:
    def test_a_backbone_that_will_not_start_restores_the_previous_selection(
            self, root, registry, monkeypatch):
        fake = FakeRuntime(fails=True)
        monkeypatch.setattr(mc_llm_runtime, "runtime", fake)
        install_bundle(root, "first-model", "gemma4-12b-qat-balanced")
        install_bundle(root, "second-model", "qwen35-9b-aggressive")
        write_state(root, runtime="runtime/llama-server",
                    model="models/managed/first-model/model.gguf",
                    mmproj="models/managed/first-model/mmproj.gguf",
                    source="managed", managed_model_id="first-model",
                    managed_profile="gemma4-12b-qat-balanced", managed_profile_version="1")

        with pytest.raises(managed.ManagedError) as raised:
            managed.use("second-model")

        assert "rolled back to First Model" in str(raised.value).replace("Rolled", "rolled")
        assert managed.selection().identifier == "first-model"
        assert state_of(root)["model"] == "models/managed/first-model/model.gguf"

    def test_a_failed_switch_tries_to_put_the_previous_model_back_on_the_card(
            self, root, registry, monkeypatch):
        attempts = []

        class Flaky(FakeRuntime):
            def client(self, needs_vision=False, reserve=0):
                attempts.append(managed.selection().identifier)
                if len(attempts) == 1:
                    raise RuntimeError("out of memory")
                return FakeClient(self)

        fake = Flaky()
        monkeypatch.setattr(mc_llm_runtime, "runtime", fake)
        install_bundle(root, "first-model", "gemma4-12b-qat-balanced")
        install_bundle(root, "second-model", "qwen35-9b-aggressive")
        write_state(root, runtime="runtime/llama-server",
                    model="models/managed/first-model/model.gguf",
                    source="managed", managed_model_id="first-model",
                    managed_profile="gemma4-12b-qat-balanced")

        with pytest.raises(managed.ManagedError):
            managed.use("second-model")

        assert attempts == ["second-model", "first-model"]

    def test_a_backbone_that_starts_but_will_not_answer_is_also_rolled_back(
            self, root, registry, monkeypatch):
        class Mute(FakeRuntime):
            def client(self, needs_vision=False, reserve=0):
                super().client(needs_vision, reserve)
                return FakeClient(self, answers=False)

        fake = Mute()
        monkeypatch.setattr(mc_llm_runtime, "runtime", fake)
        install_bundle(root, "second-model", "qwen35-9b-aggressive")
        write_state(root, runtime="runtime/llama-server", model="/elsewhere/mine.gguf")

        with pytest.raises(managed.ManagedError):
            managed.use("second-model")

        assert state_of(root)["model"] == "/elsewhere/mine.gguf"
        assert managed.selection().source == "manual"

    def test_a_failed_switch_keeps_the_downloaded_files(self, root, registry, monkeypatch):
        """Twenty minutes of somebody's connection is not thrown away because a
        start failed. The bundle stays; Use can be pressed again."""
        monkeypatch.setattr(mc_llm_runtime, "runtime", FakeRuntime(fails=True))
        bundle = install_bundle(root, "second-model", "qwen35-9b-aggressive")
        write_state(root, runtime="runtime/llama-server", model="/elsewhere/mine.gguf")

        with pytest.raises(managed.ManagedError):
            managed.use("second-model")

        assert (bundle / "model.gguf").is_file()
        assert managed.status(managed.entry("second-model")).state == managed.INSTALLED


class TestFollowingAHandPickedPath:
    def test_choosing_a_managed_bundles_own_weights_keeps_its_profile(self, root, registry):
        """The managed root is under the folder the ordinary chooser scans, so
        this happens the moment a backbone is downloaded."""
        bundle = install_bundle(root, "first-model", "gemma4-12b-qat-balanced")
        write_state(root, runtime="runtime/llama-server")

        chosen = managed.follow_path(bundle / "model.gguf")

        assert chosen.managed
        assert chosen.identifier == "first-model"
        assert chosen.profile_id == "gemma4-12b-qat-balanced"

    def test_choosing_a_stranger_gguf_clears_the_managed_selection(self, root, registry,
                                                                   tmp_path):
        install_bundle(root, "first-model", "gemma4-12b-qat-balanced")
        write_state(root, runtime="runtime/llama-server", source="managed",
                    managed_model_id="first-model",
                    managed_profile="gemma4-12b-qat-balanced")
        mine = tmp_path / "mine.gguf"
        mine.write_bytes(b"my own weights")

        chosen = managed.follow_path(mine)

        assert not chosen.managed
        assert chosen.source == "manual"
        assert state_of(root)["managed_model_id"] == ""

    def test_a_deleted_bundle_stops_being_recognised(self, root, registry):
        import shutil

        bundle = install_bundle(root, "first-model", "gemma4-12b-qat-balanced")
        path = bundle / "model.gguf"
        shutil.rmtree(bundle)

        assert managed.identify_path(path) is None


class TestTheSetupPanel:
    """The buttons, driven the way Gradio drives them.

    Worth doing at this level rather than only under it: the catalogue is a
    dropdown, a line and one button whose label is the entire explanation of
    what pressing it costs, and every one of those is computed by a handler
    that can be wrong on its own.
    """

    def last(self, generated):
        """A Gradio handler's final yield -- what the user is left looking at."""
        outputs = list(generated)
        assert outputs, "the handler yielded nothing"
        return outputs[-1]

    def test_the_dropdown_lists_the_catalogue_grouped(self, root, registry, runtime):
        import mc_llm_studio

        assert mc_llm_studio._managed_choices() == [
            ("Recommended · First Model", "first-model"),
            ("Recommended · Second Model", "second-model"),
        ]

    def test_it_opens_on_the_active_backbone_and_otherwise_on_the_first(self, root, registry,
                                                                       runtime):
        import mc_llm_studio

        assert mc_llm_studio._managed_current() == "first-model"

        install_bundle(root, "second-model", "qwen35-9b-aggressive")
        write_state(root, runtime="runtime/llama-server")
        managed.use("second-model")

        assert mc_llm_studio._managed_current() == "second-model"

    def test_the_button_says_what_pressing_it_will_cost(self, root, registry, runtime):
        import mc_llm_studio

        assert mc_llm_studio._managed_action("first-model") == mc_llm_studio.DOWNLOAD_AND_USE

        install_bundle(root, "first-model", "gemma4-12b-qat-balanced")

        assert mc_llm_studio._managed_action("first-model") == mc_llm_studio.USE

    def test_the_line_says_what_it_is_where_it_stands_and_where_it_came_from(
            self, root, registry, runtime):
        import mc_llm_studio

        line = mc_llm_studio._managed_line("first-model")

        assert "Recommended · ~7.4 GB + 175 MB vision · Test" in line
        assert managed.NOT_DOWNLOADED in line
        assert "https://huggingface.co/example/test" in line

    def test_pressing_use_on_an_installed_backbone_applies_it(self, root, registry, runtime):
        import mc_llm_studio

        install_bundle(root, "first-model", "gemma4-12b-qat-balanced")
        write_state(root, runtime="runtime/llama-server")

        notice, button, model_box, mmproj_box, model_line, _estimate, _residency = self.last(
            mc_llm_studio._use_managed("first-model"))

        assert managed.ACTIVE in notice
        assert button["value"] == mc_llm_studio.USE and button["interactive"]
        assert model_box.endswith("models/managed/first-model/model.gguf")
        assert mmproj_box.endswith("models/managed/first-model/mmproj.gguf")
        assert "First Model (managed)" in model_line

    def test_a_busy_gpu_leaves_the_button_pressable_and_the_boxes_alone(
            self, root, registry, runtime, monkeypatch):
        import mc_llm_studio

        install_bundle(root, "first-model", "gemma4-12b-qat-balanced")
        write_state(root, runtime="runtime/llama-server", model="/elsewhere/mine.gguf")
        monkeypatch.setattr(mc_broker, "workload", _no_gpu)

        notice, button, *boxes = self.last(mc_llm_studio._use_managed("first-model"))

        assert "not changed" in notice
        assert button["interactive"]
        assert all(isinstance(box, dict) for box in boxes), "the path boxes were rewritten"
        assert state_of(root)["model"] == "/elsewhere/mine.gguf"

    def test_pressing_use_with_nothing_chosen_says_so_and_starts_nothing(self, root, registry,
                                                                         runtime):
        import mc_llm_studio

        notice, *_rest = self.last(mc_llm_studio._use_managed(""))

        assert "Choose a backbone" in notice
        assert runtime.events == []

    def test_a_backbone_that_will_not_start_reports_the_rollback(self, root, registry,
                                                                 monkeypatch):
        import mc_llm_studio

        monkeypatch.setattr(mc_llm_runtime, "runtime", FakeRuntime(fails=True))
        install_bundle(root, "first-model", "gemma4-12b-qat-balanced")
        write_state(root, runtime="runtime/llama-server", model="/elsewhere/mine.gguf")

        notice, button, *_rest = self.last(mc_llm_studio._use_managed("first-model"))

        assert "olled back to" in notice
        assert button["interactive"]

    def test_refresh_re_reads_the_registry_without_downloading(self, root, registry, runtime):
        import mc_llm_studio

        dropdown, notice, button = mc_llm_studio._refresh_managed("second-model")

        assert dropdown["value"] == "second-model"
        assert len(dropdown["choices"]) == 2
        assert managed.NOT_DOWNLOADED in notice
        assert button["value"] == mc_llm_studio.DOWNLOAD_AND_USE

    def test_a_broken_catalogue_costs_the_section_and_not_the_tab(self, root, monkeypatch,
                                                                  tmp_path, runtime):
        """Section 18's rule, applied to the newest thing that can fail. An
        extension whose registry will not load still has to let somebody point
        at a GGUF of their own."""
        import mc_llm_studio

        monkeypatch.setattr(managed, "REGISTRY_PATH", tmp_path / "not-here.json")
        managed._registry_cache = None

        built = mc_llm_studio._setup_panel()

        assert mc_llm_studio._managed_choices() == []
        assert mc_llm_studio._managed_current() is None
        assert "could not be read" in mc_llm_studio._managed_line("first-model")
        assert built["model"] is not None and built["mmproj"] is not None
