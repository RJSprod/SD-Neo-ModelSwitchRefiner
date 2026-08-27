"""Vision as a capability a llama-server acquires, and never gives back.

The regression this file exists to keep fixed was a lifecycle, not a bug in one
function. A request's ``needs_vision`` flag was read as a complete description
of the server that ought to be running, so the flag going false was an
instruction to rebuild a smaller server -- and a conversation with pictures in
it thrashed:

    text  -> text server
    image -> restart with the projector
    text  -> restart without it
    image -> restart with it again

Every one of those restarts costs a model load, a CUDA context, and llama.cpp's
prompt cache, which is thirteen seconds of prompt evaluation before a single
token appears. So the assertions here are mostly about *process identity and
counts* rather than about return values: a test that only checked each request
eventually succeeded would have passed against the broken lifecycle.

The intended shape is monotonic. Capability moves one way while the same server
stays valid -- OFF -> TEXT_ONLY -> VISION_LOADED -- and a normal request never
moves it back.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import mc_llm_managed_models as managed
import mc_llm_paths
import mc_llm_roles
import mc_llm_runtime as runtime
import mc_llm_vision as vision
from test_llm_context import build_model
from test_llm_managed_download import (FakeHub, artifacts, hub, registry,  # noqa: F401
                                       root, state_of, write_state)
from test_llm_runtime import FakeProcess, placed, set_free  # noqa: F401

_GB = 1024**3


# --------------------------------------------------------------------------- #
# A configured install that has eyes
# --------------------------------------------------------------------------- #


def configure(monkeypatch, tmp_path, *, projector=True, context=8192, blocks=32,
              size_mb=4, roles=None):
    """An install whose backbone is multimodal, and whose projector is on disk.

    ``projector=False`` is the other half of the same fixture: a backbone with
    no eyes at all, whose image requests have to be refused in a sentence rather
    than by a server that will not start.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    model = build_model(tmp_path, blocks=blocks, size_mb=size_mb, context=131072)
    server_binary = tmp_path / "llama-server"
    server_binary.write_bytes(b"")
    mmproj = None
    if projector:
        mmproj = build_model(tmp_path, "mmproj.gguf", size_mb=1, blocks=4)

    def build(role=""):
        return runtime.Config(
            runtime=server_binary, model=model, mmproj=mmproj, gpu_index=0,
            device="CUDA0", gpu_layers="all", context_size=context, context_mode="fixed",
            context_buffer_gb=4.0, kv_type_k="f16", kv_type_v="f16", mode="gpu")

    lookup = roles or {}
    monkeypatch.setattr(runtime, "config", lambda role="": lookup.get(role) or build(role))
    return build()


@pytest.fixture
def eyes(monkeypatch, tmp_path):
    """A runtime with fake processes, and the log of what each start was given."""
    monkeypatch.setattr(mc_llm_paths, "data_root", lambda: tmp_path / "data")
    monkeypatch.setattr(runtime, "OFFLOAD_WAIT_SECONDS", 0.0)
    started: list = []
    managed_runtime = runtime.Runtime()
    monkeypatch.setattr(managed_runtime, "_new_process", lambda: FakeProcess(started))
    yield managed_runtime, started
    managed_runtime.stop()


def _elsewhere(tmp_path) -> Path:
    """A GGUF that is nobody's managed bundle."""
    folder = tmp_path / "elsewhere"
    folder.mkdir(parents=True, exist_ok=True)
    return build_model(folder, "stranger.gguf", size_mb=1)


def projectors(started: list) -> list:
    """The ``--mmproj`` argument of every start, in order. ``None`` for none."""
    return [call[0][2] for call in started]


# --------------------------------------------------------------------------- #
# AT-1 to AT-5: the capability lifecycle
# --------------------------------------------------------------------------- #


class TestTheCapabilityLifecycle:
    def test_a_cold_text_start_omits_the_projector(self, placed, eyes, tmp_path, monkeypatch):
        """AT-1. The projector is a gigabyte and a third of a card the model is
        already filling, and a conversation that never attaches a picture must
        never pay for it."""
        configure(monkeypatch, tmp_path)
        set_free(monkeypatch, 40)
        server, started = eyes

        server.client()

        assert projectors(started) == [None]
        assert not server.vision_loaded()

    def test_a_cold_vision_start_loads_it_once(self, placed, eyes, tmp_path, monkeypatch):
        """AT-2, cold. There is no text-only start followed immediately by a
        second restart when the very first request already carries an image."""
        configuration = configure(monkeypatch, tmp_path)
        set_free(monkeypatch, 40)
        server, started = eyes

        server.client(needs_vision=True)

        assert projectors(started) == [configuration.mmproj]
        assert server.vision_loaded()

    def test_the_first_image_upgrades_a_warm_server_once(self, placed, eyes, tmp_path,
                                                         monkeypatch):
        """AT-2, warm. One restart, and the exact compatible projector."""
        configuration = configure(monkeypatch, tmp_path)
        set_free(monkeypatch, 40)
        server, started = eyes

        server.client()
        server.client(needs_vision=True)

        assert projectors(started) == [None, configuration.mmproj]
        assert server.vision_loaded()

    def test_text_after_an_image_reuses_the_same_process(self, placed, eyes, tmp_path,
                                                         monkeypatch):
        """AT-3, and the core latency rule. Vision capability is a superset of
        text capability, so a text request is fully satisfied by the server that
        is already up -- the same process, with its prompt cache intact."""
        configure(monkeypatch, tmp_path)
        set_free(monkeypatch, 40)
        server, started = eyes

        server.client(needs_vision=True)
        process = server._process
        server.client()

        assert len(started) == 1
        assert server._process is process
        assert server.vision_loaded()

    def test_a_second_image_reuses_it_too(self, placed, eyes, tmp_path, monkeypatch):
        """AT-4."""
        configure(monkeypatch, tmp_path)
        set_free(monkeypatch, 40)
        server, started = eyes

        server.client(needs_vision=True)
        process = server._process
        server.client(needs_vision=True)

        assert len(started) == 1
        assert server._process is process

    def test_a_long_mixed_session_has_exactly_one_capability_restart(
            self, placed, eyes, tmp_path, monkeypatch):
        """AT-5 and section 30's performance regression test.

        text, text, image, text, text, image, text -- one start, one upgrade,
        and nothing else. A change that reintroduced the thrash would still
        answer every one of these requests correctly, which is exactly why the
        count is what is asserted."""
        configuration = configure(monkeypatch, tmp_path)
        set_free(monkeypatch, 40)
        server, started = eyes

        for needs_vision in (False, False, True, False, False, True, False):
            server.client(needs_vision=needs_vision)

        assert projectors(started) == [None, configuration.mmproj]

    def test_the_process_survives_every_text_turn_after_the_upgrade(
            self, placed, eyes, tmp_path, monkeypatch):
        """Process-ID continuity, which is what section 30 asks to be recorded.
        A restart that happened to produce an identical command line would still
        have thrown the prompt cache away."""
        configure(monkeypatch, tmp_path)
        set_free(monkeypatch, 40)
        server, started = eyes

        server.client(needs_vision=True)
        seen = {id(server._process)}
        for _turn in range(4):
            server.client()
            seen.add(id(server._process))

        assert len(seen) == 1


# --------------------------------------------------------------------------- #
# Section 7: when stickiness legitimately ends
# --------------------------------------------------------------------------- #


class TestWhenVisionEnds:
    def test_an_explicit_unload_drops_it(self, placed, eyes, tmp_path, monkeypatch):
        configure(monkeypatch, tmp_path)
        set_free(monkeypatch, 40)
        server, started = eyes

        server.client(needs_vision=True)
        server.stop()

        assert not server.vision_loaded()
        assert server.loaded_projector() is None

    def test_the_next_request_after_an_unload_starts_text_only(self, placed, eyes, tmp_path,
                                                               monkeypatch):
        """Sticky is per *running process*. A server that has been stopped takes
        the capability with it, so the lightest start is available again."""
        configure(monkeypatch, tmp_path)
        set_free(monkeypatch, 40)
        server, started = eyes

        server.client(needs_vision=True)
        server.stop()
        server.client()

        assert projectors(started)[-1] is None

    def test_changing_the_model_drops_it(self, placed, eyes, tmp_path, monkeypatch):
        """A projector belongs to the weights it was made for. New weights are a
        new server, and inheriting the old projector into it would be I-8's
        prohibition arriving by the back door."""
        configure(monkeypatch, tmp_path)
        set_free(monkeypatch, 40)
        server, started = eyes

        server.client(needs_vision=True)
        configure(monkeypatch, tmp_path / "other", projector=False)
        server.client()

        assert projectors(started)[-1] is None
        assert not server.vision_loaded()

    def test_a_projector_deleted_underneath_a_live_server_is_not_passed_on(
            self, placed, eyes, tmp_path, monkeypatch):
        """The one case where a text request legitimately loses vision: the file
        is gone, so it cannot be an argument to the next start. It is reported
        as the replacement it is rather than crashing a text turn."""
        configuration = configure(monkeypatch, tmp_path)
        set_free(monkeypatch, 40)
        server, started = eyes

        server.client(needs_vision=True)
        Path(configuration.mmproj).unlink()
        server._identity = None  # force the slow path, as a settings change would

        server.client()

        assert projectors(started)[-1] is None


# --------------------------------------------------------------------------- #
# Section 21: a text request must not re-negotiate the projector away
# --------------------------------------------------------------------------- #


class TestPlacementDoesNotUndoIt:
    def test_a_text_request_does_not_re_place_a_vision_loaded_server(
            self, placed, eyes, tmp_path, monkeypatch):
        """The subtle way the thrash could come back. A vision-loaded server is
        holding the projector's VRAM, so a placement previewed *without* it
        looks roomier -- and "it now fits better" is a restart. The preview has
        to be told what the running process is actually holding."""
        configure(monkeypatch, tmp_path, size_mb=64)
        set_free(monkeypatch, 40)
        server, started = eyes

        server.client(needs_vision=True)
        set_free(monkeypatch, 2)  # the card fills up underneath it
        server.client()

        assert len(started) == 1


# --------------------------------------------------------------------------- #
# AT-9 and AT-10: isolation and sharing
# --------------------------------------------------------------------------- #


class TestRuntimesAreIsolated:
    def test_upgrading_one_runtime_leaves_the_others_alone(self, placed, tmp_path, monkeypatch):
        """AT-9. Vision state is not global (I-6): one llama-server acquiring a
        projector must not restart, reload, or rewrite the state of another."""
        monkeypatch.setattr(mc_llm_paths, "data_root", lambda: tmp_path / "data")
        monkeypatch.setattr(runtime, "OFFLOAD_WAIT_SECONDS", 0.0)
        configure(monkeypatch, tmp_path)
        set_free(monkeypatch, 40)

        started_a, started_b = [], []
        conversation, creative = runtime.Runtime(), runtime.Runtime()
        monkeypatch.setattr(conversation, "_new_process", lambda: FakeProcess(started_a))
        monkeypatch.setattr(creative, "_new_process", lambda: FakeProcess(started_b))
        try:
            conversation.client()
            creative.client()
            untouched = creative._process

            conversation.client(needs_vision=True)

            assert conversation.vision_loaded()
            assert not creative.vision_loaded()
            assert creative._process is untouched
            assert len(started_b) == 1
            assert projectors(started_b) == [None]
        finally:
            conversation.stop()
            creative.stop()

    def test_two_roles_sharing_a_runtime_share_its_capability(self, placed, eyes, tmp_path,
                                                              monkeypatch):
        """AT-10. Roles coalesce onto one process when their configurations
        match, and a capability that process has is a capability both of them
        get -- Krea pays for the upgrade, Creative's next text pass reuses it."""
        configure(monkeypatch, tmp_path)
        set_free(monkeypatch, 40)
        server, started = eyes
        server.roles = (mc_llm_roles.CREATIVE, mc_llm_roles.SPATIAL)

        server.client(needs_vision=True)   # Krea captions a reference
        process = server._process
        server.client()                    # Creative writes without references
        server.client()                    # the Composer composes

        assert len(started) == 1
        assert server._process is process


# --------------------------------------------------------------------------- #
# A backbone with no eyes at all
# --------------------------------------------------------------------------- #


class TestABackboneWithNoProjector:
    def test_text_runs_perfectly_well(self, placed, eyes, tmp_path, monkeypatch):
        configure(monkeypatch, tmp_path, projector=False)
        set_free(monkeypatch, 40)
        server, started = eyes

        server.client()

        assert projectors(started) == [None]

    def test_an_image_request_is_refused_locally_and_says_why(self, placed, eyes, tmp_path,
                                                              monkeypatch):
        """I-4 and I-10. The image is never silently dropped, and the refusal is
        a local sentence rather than a fallback to somebody's API."""
        configure(monkeypatch, tmp_path, projector=False)
        set_free(monkeypatch, 40)
        server, started = eyes

        with pytest.raises(RuntimeError) as refusal:
            server.client(needs_vision=True)

        assert "vision projector" in str(refusal.value)
        assert "cloud" in str(refusal.value)
        assert not started

    def test_a_refused_image_does_not_stop_the_text_server(self, placed, eyes, tmp_path,
                                                           monkeypatch):
        """Section 24.1: do not destroy a usable text model because an optional
        vision upgrade could not be provisioned."""
        configure(monkeypatch, tmp_path, projector=False)
        set_free(monkeypatch, 40)
        server, started = eyes

        server.client()
        process = server._process
        with pytest.raises(RuntimeError):
            server.client(needs_vision=True)

        assert server.running()
        assert server._process is process


# --------------------------------------------------------------------------- #
# AT-6: selection must not destroy the association
# --------------------------------------------------------------------------- #


class TestSelectionKeepsTheProjector:
    def test_the_ordinary_chooser_restores_a_managed_projector(self, root, hub, registry):
        """The regression's root cause, as one test. The model chooser records a
        model with no projector -- correctly, because a filename proves nothing
        -- and managed-path recognition used to put back the identity without
        the association, leaving a state file that knew a multimodal backbone
        was selected and reported that it had no eyes."""
        from prompt_master.inference import model_choice

        bundle = managed.download("test-model")
        write_state(root, runtime="llama-server", model="x.gguf", mmproj="was-here.gguf")

        model_choice.choose(mc_llm_paths.app_paths(), bundle.model, None)
        assert state_of(root)["mmproj"] == ""           # the chooser cleared it

        managed.follow_path(bundle.model)

        recorded = mc_llm_paths.app_paths().locate(state_of(root)["mmproj"])
        assert recorded == bundle.mmproj

    def test_it_is_restored_even_when_the_selection_was_already_right(self, root, hub, registry):
        """The early return had to learn about the projector too: source and id
        already saying ``managed`` is exactly the state the chooser leaves
        behind after it has blanked ``mmproj``."""
        from prompt_master.inference import model_choice

        bundle = managed.download("test-model")
        write_state(root, runtime="llama-server", model="x.gguf",
                    source="managed", managed_model_id="test-model")
        model_choice.choose(mc_llm_paths.app_paths(), bundle.model, None)

        managed.follow_path(bundle.model)

        assert state_of(root)["mmproj"]

    def test_selecting_it_loads_nothing(self, root, hub, registry):
        """AT-6's last clause. Knowing which projector belongs to the backbone is
        not the same as loading it, and selection must stay free."""
        bundle = managed.download("test-model")
        managed.follow_path(bundle.model)

        assert managed.selection().managed
        assert state_of(root)["mmproj"]

    def test_a_hand_picked_gguf_still_clears_it(self, root, hub, registry, tmp_path):
        """The other direction, which must keep working: a stranger's model has
        no business inheriting a managed backbone's projector."""
        from prompt_master.inference import model_choice

        managed.download("test-model")
        stranger = _elsewhere(tmp_path)
        write_state(root, runtime="llama-server", model="x.gguf",
                    source="managed", managed_model_id="test-model")

        model_choice.choose(mc_llm_paths.app_paths(), stranger, None)
        managed.follow_path(stranger)

        assert not managed.selection().managed
        assert state_of(root)["mmproj"] == ""

    def test_switching_backbones_records_the_declared_projector(self, root, hub, registry):
        """A switch whose bundle has lost its projector file still records the
        association, because that is what the first image request repairs from."""
        bundle = managed.download("test-model")
        bundle.mmproj.unlink()

        recorded = managed._recorded_projector(managed.entry("test-model"))

        assert recorded and mc_llm_paths.app_paths().locate(recorded) == \
            bundle.root / managed.MMPROJ_FILENAME

    def test_a_text_only_entry_gets_no_invented_path(self, root, hub, registry, monkeypatch):
        """An entry the catalogue declares no projector for must not be given a
        path to a file that will never exist -- every later image request would
        then fail on a missing artifact instead of saying plainly that this
        backbone does not see."""
        import dataclasses

        text_only = dataclasses.replace(managed.entry("test-model"), projector=None)

        assert managed._recorded_projector(text_only) is None

    def test_the_trusted_projector_is_found_from_the_bundle_path(self, root, hub, registry):
        """What the Setup boxes use, and it is not a filename guess (I-8): the
        path is recognised as a managed bundle and the registry is asked."""
        bundle = managed.download("test-model")

        assert vision.projector_for_model(bundle.model) == bundle.mmproj

    def test_a_stranger_has_no_trusted_projector(self, root, hub, registry, tmp_path):
        stranger = _elsewhere(tmp_path)

        assert vision.projector_for_model(stranger) is None


# --------------------------------------------------------------------------- #
# AT-7 and AT-8: repair
# --------------------------------------------------------------------------- #


class TestRepairingAMissingProjector:
    def test_a_missing_projector_is_downloaded_and_verified(self, root, hub, registry):
        """AT-7. The user is not asked to go and find a file this extension
        knows the exact name, revision and SHA-256 of."""
        bundle = managed.download("test-model")
        expected = bundle.mmproj.read_bytes()
        bundle.mmproj.unlink()

        repaired = managed.repair_projector("test-model")

        assert repaired == bundle.root / managed.MMPROJ_FILENAME
        assert repaired.read_bytes() == expected

    def test_the_weights_are_not_touched(self, root, hub, registry):
        """The invariant that makes an automatic repair safe: a repair that went
        wrong must leave an installation that still runs, without eyes."""
        bundle = managed.download("test-model")
        before = bundle.model.read_bytes()
        bundle.mmproj.unlink()
        hub.requests.clear()

        managed.repair_projector("test-model")

        assert bundle.model.read_bytes() == before
        assert all("weights" not in url for url, _headers in hub.requests)

    def test_only_the_projector_is_fetched(self, root, hub, registry):
        """Not ``download()``. Re-running the whole transaction to obtain one
        missing sidecar would re-verify -- and possibly re-fetch -- weights that
        are already on the disk and quite possibly mmapped by a running
        server."""
        bundle = managed.download("test-model")
        bundle.mmproj.unlink()
        hub.requests.clear()

        managed.repair_projector("test-model")

        assert len(hub.requests) == 1
        assert hub.requests[0][0].endswith("mmproj-test.gguf")

    def test_a_corrupt_projector_is_replaced(self, root, hub, registry):
        """Section 12: a bundle is not valid merely because a manifest remembers
        that a projector once existed. Wrong bytes are an incomplete bundle."""
        bundle = managed.download("test-model")
        expected = bundle.mmproj.read_bytes()
        bundle.mmproj.write_bytes(b"not a projector")

        managed.repair_projector("test-model")

        assert bundle.mmproj.read_bytes() == expected

    def test_a_projector_that_is_already_right_is_not_downloaded_again(self, root, hub, registry):
        bundle = managed.download("test-model")
        hub.requests.clear()

        assert managed.repair_projector("test-model") == bundle.mmproj
        assert hub.requests == []

    def test_the_manifest_learns_about_the_repair(self, root, hub, registry):
        """Otherwise the panel goes on offering to re-download a backbone whose
        files are now complete."""
        bundle = managed.download("test-model")
        bundle.mmproj.unlink()
        document = json.loads((bundle.root / managed.INSTALLED_FILENAME).read_text())
        document["artifacts"].pop("projector")
        (bundle.root / managed.INSTALLED_FILENAME).write_text(json.dumps(document))

        managed.repair_projector("test-model")

        assert managed.installed("test-model").matches(managed.entry("test-model"))

    def test_ensure_repairs_and_records_it(self, root, hub, registry):
        """AT-7 end to end, at the seam the runtime actually calls."""
        bundle = managed.download("test-model")
        bundle.mmproj.unlink()
        write_state(root, runtime="llama-server", model=str(bundle.model),
                    source="managed", managed_model_id="test-model")
        configuration = runtime.config()

        found = vision.ensure_projector(configuration)

        assert found == bundle.root / managed.MMPROJ_FILENAME
        assert found.is_file()
        assert mc_llm_paths.app_paths().locate(state_of(root)["mmproj"]) == found

    def test_a_failed_repair_says_so_rather_than_looking_text_only(self, root, hub, registry):
        """The distinction that matters: "the download failed" must never be
        returned as "this backbone has no projector", or an image request would
        be refused as unsupported instead of reported as unprovisioned."""
        bundle = managed.download("test-model")
        bundle.mmproj.unlink()
        hub.files.clear()
        write_state(root, runtime="llama-server", model=str(bundle.model),
                    source="managed", managed_model_id="test-model")

        with pytest.raises(vision.VisionUnavailable) as refusal:
            vision.ensure_projector(runtime.config())

        assert "test-model" in str(refusal.value)
        assert "still answers text" in str(refusal.value)

    def test_a_text_request_never_reaches_the_repair(self, root, hub, registry, tmp_path,
                                                     monkeypatch, placed):
        """AT-8. A missing projector is not a reason to download anything, or to
        refuse anything, when nobody has attached a picture."""
        bundle = managed.download("test-model")
        bundle.mmproj.unlink()
        write_state(root, runtime="llama-server", model=str(bundle.model),
                    source="managed", managed_model_id="test-model")
        hub.requests.clear()

        asked = []
        monkeypatch.setattr(vision, "ensure_projector",
                            lambda *args, **kwargs: asked.append(args) or None)
        monkeypatch.setattr(runtime, "OFFLOAD_WAIT_SECONDS", 0.0)
        started: list = []
        server = runtime.Runtime()
        monkeypatch.setattr(server, "_new_process", lambda: FakeProcess(started))
        monkeypatch.setattr(runtime, "config", lambda role="": runtime.Config(
            runtime=tmp_path / "llama-server", model=bundle.model, mmproj=None, gpu_index=0,
            device="CUDA0", gpu_layers="all", context_size=8192, context_mode="fixed",
            context_buffer_gb=4.0, kv_type_k="f16", kv_type_v="f16", mode="gpu"))
        (tmp_path / "llama-server").write_bytes(b"")
        set_free(monkeypatch, 40)

        try:
            server.client()
        finally:
            server.stop()

        assert asked == []
        assert hub.requests == []

    def test_a_manual_install_is_never_guessed_at(self, root, hub, registry, tmp_path):
        """I-8 from the other side. There is no trusted association for a
        hand-picked GGUF, so there is nothing to repair from and nothing is
        invented."""
        stranger = _elsewhere(tmp_path)
        write_state(root, runtime="llama-server", model=str(stranger))

        assert vision.ensure_projector(runtime.config()) is None


# --------------------------------------------------------------------------- #
# Section 24: what each failure costs
# --------------------------------------------------------------------------- #


class TestWhenTheUpgradeFails:
    def test_a_start_that_fails_with_the_projector_does_not_serve_the_image_blind(
            self, placed, eyes, tmp_path, monkeypatch):
        """24.2 and 24.3. The request that needed vision fails. What must never
        happen is the quiet retry without ``--mmproj``: a reply written about a
        picture the model was never shown reads as a working feature and is
        the worst possible outcome."""
        configure(monkeypatch, tmp_path)
        set_free(monkeypatch, 40)
        server, started = eyes

        server.client()
        def refuse(*args, **kwargs):
            raise runtime._StartFailed("llama.cpp could not load the projector")
        monkeypatch.setattr(server, "_launch", refuse)

        with pytest.raises(RuntimeError):
            server.client(needs_vision=True)

        assert not server.vision_loaded()
        assert projectors(started) == [None]     # nothing was started blind

    def test_the_runtime_is_left_in_a_state_the_next_request_can_use(
            self, placed, eyes, tmp_path, monkeypatch):
        """24.5. A failed or cancelled upgrade must not leave a half-recorded
        server that the next request reuses or trips over."""
        configure(monkeypatch, tmp_path)
        set_free(monkeypatch, 40)
        server, started = eyes

        working = server._launch

        def refuse(*args, **kwargs):
            raise runtime._StartFailed("llama.cpp could not load the projector")

        server._launch = refuse
        with pytest.raises(RuntimeError):
            server.client(needs_vision=True)
        server._launch = working

        server.client()

        assert server.running()
        assert projectors(started)[-1] is None

    def test_the_projector_is_resolved_before_the_warm_server_is_stopped(
            self, placed, eyes, tmp_path, monkeypatch):
        """Section 13. A projector download is a gigabyte over somebody's
        connection, and it must not happen while holding the process lock of a
        llama-server that is answering other requests perfectly well. The order
        is the observable part of that: resolve, then stop."""
        configure(monkeypatch, tmp_path)
        set_free(monkeypatch, 40)
        server, started = eyes
        order: list = []

        server.client()
        monkeypatch.setattr(vision, "ensure_projector",
                            lambda *args, **kwargs: order.append("resolved"))
        original = server._stop_locked
        monkeypatch.setattr(server, "_stop_locked",
                            lambda reason: (order.append("stopped"), original(reason))[1])

        server.client(needs_vision=True)

        assert order == ["resolved", "stopped"]

    def test_a_cancelled_repair_keeps_the_server_that_was_running(self, root, hub, registry):
        """24.4. Stopping the download stops the download. What has arrived is
        kept, so the next attempt resumes -- and the model that was answering
        text is still the model that is answering text."""
        import threading

        bundle = managed.download("test-model")
        bundle.mmproj.unlink()
        cancel = threading.Event()
        cancel.set()

        with pytest.raises(managed.Cancelled):
            managed.repair_projector("test-model", cancel=cancel)

        assert bundle.model.is_file()
        assert not bundle.mmproj.is_file()

    def test_a_cancelled_repair_reads_as_a_cancellation(self, root, hub, registry):
        """Not as "the bundle could not be provisioned", which is a different
        and much more worrying thing to read after pressing Stop."""
        import threading

        bundle = managed.download("test-model")
        bundle.mmproj.unlink()
        write_state(root, runtime="llama-server", model=str(bundle.model),
                    source="managed", managed_model_id="test-model")
        cancel = threading.Event()
        cancel.set()

        with pytest.raises(managed.Cancelled):
            vision.ensure_projector(runtime.config(), cancel=cancel)

    def test_a_repair_that_cannot_finish_leaves_the_weights_alone(self, root, hub, registry):
        """The bundle is incomplete for vision and complete for everything else,
        which is exactly what it was before the attempt."""
        bundle = managed.download("test-model")
        bundle.mmproj.unlink()
        hub.files.clear()

        with pytest.raises(managed.ManagedError):
            managed.repair_projector("test-model")

        assert bundle.model.is_file()


# --------------------------------------------------------------------------- #
# OT-2 to OT-5: the offline and privacy contract
# --------------------------------------------------------------------------- #


class TestInferenceStaysOnThisMachine:
    def client(self, url="http://127.0.0.1:8080"):
        from prompt_master.inference.llama_client import LlamaClient

        return LlamaClient(url, "key")

    @pytest.mark.parametrize("url", [
        "http://127.0.0.1:8080", "http://localhost:8080", "http://[::1]:8080",
        "https://127.0.0.1:8443",
    ])
    def test_loopback_endpoints_are_accepted(self, url):
        assert self.client(url).base_url == url

    @pytest.mark.parametrize("url", [
        "https://api.openai.com/v1", "https://api.anthropic.com",
        "http://192.168.1.40:8080", "http://10.0.0.5:8080",
        "https://generativelanguage.googleapis.com",
        "https://api-inference.huggingface.co/models/x",
        "http://llm.internal:8080",
    ])
    def test_every_other_endpoint_is_refused_before_a_prompt_exists(self, url):
        """OT-2. Checked at construction, which is the last moment before a
        prompt, a character card or a Creative brief can be attached to it."""
        from prompt_master.inference.local_only import NotLocal

        with pytest.raises(NotLocal):
            self.client(url)

    def test_the_refusal_says_there_is_no_cloud_fallback(self):
        from prompt_master.inference.local_only import NotLocal

        with pytest.raises(NotLocal) as refusal:
            self.client("https://api.openai.com/v1")
        assert "no cloud fallback" in str(refusal.value)

    def test_credentials_in_an_endpoint_are_refused(self):
        from prompt_master.inference.local_only import NotLocal

        with pytest.raises(NotLocal):
            self.client("http://user:secret@127.0.0.1:8080")

    def test_a_remote_image_url_is_refused_locally(self):
        """OT-3. llama.cpp will fetch a remote ``image_url`` if it is given one,
        which would make the *server* perform a data-dependent network request.
        It is never given one."""
        from prompt_master.inference.local_only import NotLocal, check_messages

        with pytest.raises(NotLocal):
            check_messages([{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "https://remote.example/image.png"}},
                {"type": "text", "text": "what is this"}]}])

    def test_an_embedded_image_is_accepted(self):
        """OT-4. The representation every caller in this project already
        builds."""
        from prompt_master.inference.local_only import check_messages

        check_messages([{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            {"type": "text", "text": "what is this"}]}])

    def test_plain_text_messages_pass_through(self):
        from prompt_master.inference.local_only import check_messages

        check_messages([{"role": "system", "content": "be helpful"},
                        {"role": "user", "content": "hello"}])

    def test_the_guard_runs_before_the_request_is_built(self, monkeypatch):
        """OT-3's second half: no remote fetch, by anybody. The refusal has to
        happen before httpx is reached, not after llama-server has been asked."""
        import httpx

        from prompt_master.inference.local_only import NotLocal

        def refuse(*_args, **_kwargs):
            raise AssertionError("a request was made for a rejected payload")

        monkeypatch.setattr(httpx, "Client", refuse)
        with pytest.raises(NotLocal):
            self.client().stream_chat(
                [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": "http://example.com/x.jpg"}}]}],
                64, 1, lambda _text: None)


class TestNothingRoutesInferenceElsewhere:
    """I-10, asserted about the repository rather than about one call.

    ``prompt_master`` was vendored from a standalone application that could also
    talk to LM Studio and Ollama over HTTP, and that module is still in the tree
    as reference material. Nothing in this extension may reach it: a single
    import would put a configurable inference host back into the product, and
    an import is a very easy line to add by accident.
    """

    def modules(self):
        root = Path(__file__).resolve().parent.parent
        for path in root.rglob("*.py"):
            parts = path.relative_to(root).parts
            if parts[0] in ("tests", "tools") or "upstream" in parts:
                continue
            yield path

    def test_the_connect_only_backends_are_never_imported(self):
        wired = ("upstream.backend", "upstream import backend",
                 "upstream.routes", "upstream import routes",
                 "upstream.node", "upstream import node")
        offenders = [(str(path), name) for path in self.modules()
                     for name in wired if name in path.read_text(encoding="utf-8")]

        assert offenders == []

    def test_no_hosted_model_endpoint_appears_anywhere_in_the_extension(self):
        """A URL that is not there cannot be requested by mistake."""
        hosted = ("api.openai.com", "api.anthropic.com", "generativelanguage.googleapis.com",
                  "api-inference.huggingface.co", "api.cohere", "api.mistral.ai")
        offenders = [(str(path), host) for path in self.modules()
                     for host in hosted
                     if host in path.read_text(encoding="utf-8")
                     and path.name != "local_only.py"]

        assert offenders == []


class TestTheRequestGoesToTheLocalServer:
    def test_the_runtime_builds_a_loopback_client(self, placed, eyes, tmp_path, monkeypatch):
        """OT-5's precondition and I-9's: the only client this module can build
        addresses a port on this machine that it opened itself."""
        configure(monkeypatch, tmp_path)
        set_free(monkeypatch, 40)
        server, _started = eyes

        client = server.client()

        assert client.base_url.startswith("http://127.0.0.1:")


# --------------------------------------------------------------------------- #
# Which workflows ask for vision (section 9, invariants I-1 and I-2)
# --------------------------------------------------------------------------- #


class TestWhoAsksForVision:
    def message(self, text="hello", image=""):
        from prompt_master.chat.history import Message

        return Message(role="user", versions=[text], image=image)

    def test_a_conversation_with_an_attached_image_asks_for_it(self):
        from prompt_master.chat.prompt import needs_vision

        assert needs_vision([{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}},
            {"type": "text", "text": "hello"}]}])

    def test_ordinary_conversation_text_does_not(self):
        from prompt_master.chat.prompt import needs_vision

        assert not needs_vision([{"role": "system", "content": "be helpful"},
                                 {"role": "user", "content": "hello"}])

    def test_an_image_trimmed_out_of_the_window_does_not(self):
        """The question is about the request that is actually sent. A picture
        that scrolled out of the context window is not on the wire, and paying
        for a projector upgrade to send a request with nothing for it to look at
        is the cost this asks about the built payload to avoid."""
        from prompt_master.chat.prompt import build, has_image, needs_vision
        from prompt_master.chat.characters import Character, Persona

        history = [self.message("turn 0 " + "word " * 200,
                                image="data:image/png;base64,AA"),
                   *(self.message(f"turn {index} " + "word " * 200)
                     for index in range(1, 200))]

        wire = build(Character(name="C"), Persona(name="P"), history,
                     context_size=2048, reply_tokens=256)

        assert has_image(history)          # the history still holds one
        assert not needs_vision(wire)      # the request does not
