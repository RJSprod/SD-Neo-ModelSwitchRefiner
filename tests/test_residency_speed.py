"""Residency behaviour that exists purely to make the switch faster.

Three mechanisms, all measured against real console logs from a Krea 2 ->
Flux.2 chain on a 22 GB card:

* the VRAM estimate is widened so the host stops partially unloading mid-load,
* Stage 1's encoders stay resident through the Stage 2 switch,
* Stage 1 is warmed back into VRAM in the background between generations.

None of them may change what comes out of the pipeline, so the tests here are
as interested in the fallbacks as in the happy paths.
"""

from __future__ import annotations

import types

import pytest

import mc_memory

GB = 1024**3


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class FakePatcher:
    """Stands in for a ``ModelPatcher``."""

    def __init__(self, label, size):
        self.label = label
        self.size = size

    def model_size(self):
        return self.size

    def __repr__(self):  # keeps assertion output readable
        return f"<{self.label}>"


class FakeClip:
    """``forge_objects.clip`` wraps its patcher rather than being one."""

    def __init__(self, patcher):
        self.patcher = patcher


class FakeLoadedModel:
    """Stands in for an entry of ``memory_management.current_loaded_models``."""

    def __init__(self, patcher):
        self.model = patcher

    def model_loaded_memory(self):
        return self.model.size


def make_model(name="A", unet=8 * GB, clip=4 * GB, vae=1 * GB):
    unet_patcher = FakePatcher(f"{name}-unet", unet)
    clip_patcher = FakePatcher(f"{name}-clip", clip)
    vae_patcher = FakePatcher(f"{name}-vae", vae)
    model = types.SimpleNamespace(
        name=name,
        sd_checkpoint_info=types.SimpleNamespace(
            filename=f"/models/{name}", name_for_extra=name, sha256=f"sha-{name}"
        ),
        forge_objects=types.SimpleNamespace(
            unet=unet_patcher, clip=FakeClip(clip_patcher), vae=vae_patcher
        ),
    )
    model.patchers = types.SimpleNamespace(
        unet=unet_patcher, clip=clip_patcher, vae=vae_patcher
    )
    return model


@pytest.fixture
def memory(host, monkeypatch):
    """The faked ``backend.memory_management``, reset and with a loaded model."""
    from backend import memory_management

    memory_management.freed.clear()
    memory_management.kept.clear()
    memory_management.loaded_to_gpu.clear()
    memory_management.current_loaded_models.clear()

    model = make_model("A")
    host.sd_models.model_data.sd_model = model
    memory_management.current_loaded_models.extend(
        FakeLoadedModel(p) for p in (model.patchers.unet, model.patchers.clip, model.patchers.vae)
    )

    monkeypatch.setattr(mc_memory, "file_size_bytes", lambda name, mods=None: 10 * GB)
    monkeypatch.setattr(mc_memory, "resolve_modules", lambda mods=None: None)
    monkeypatch.setattr(mc_memory, "current_modules", lambda: [])
    monkeypatch.setattr(
        host.sd_models, "get_closet_checkpoint_match",
        lambda name: types.SimpleNamespace(
            filename=f"/models/{name}", name_for_extra=name, sha256=f"sha-{name}"
        ),
    )
    mc_memory.clear_pinned_encoders()

    yield types.SimpleNamespace(mm=memory_management, model=model)

    mc_memory.clear_pinned_encoders()


# --------------------------------------------------------------------------- #
# The VRAM estimate
# --------------------------------------------------------------------------- #


class TestVramEstimate:
    """A model is bigger in VRAM than it is on disk.

    Under-counting is what produced the ``Unloaded partially`` lines in the log
    immediately before each UNet load: free_memory() hit our target and stopped,
    the target was short, and the host then unloaded in chunks *while* loading.

    Every test here isolates the *model* term by subtracting the activation
    headroom. Left in, the headroom is large enough to keep these assertions
    true even with the overhead removed entirely -- which would make the whole
    class look like coverage while testing nothing.
    """

    @staticmethod
    def model_term(file_size=10 * GB, width=0, height=0):
        return (
            mc_memory.vram_required_bytes("A", None, width, height)
            - mc_memory.vram_headroom_bytes(width, height)
            - file_size
        )

    def test_a_model_is_counted_as_larger_than_its_file(self, memory):
        assert self.model_term() > 0

    def test_the_overhead_is_proportional_to_the_model(self, memory, monkeypatch):
        def overhead(file_size):
            monkeypatch.setattr(mc_memory, "file_size_bytes", lambda name, mods=None: file_size)
            return self.model_term(file_size)

        assert overhead(20 * GB) == pytest.approx(overhead(10 * GB) * 2, rel=1e-6)

    def test_the_overhead_covers_the_shortfalls_measured_in_practice(self):
        """Two logged chains under-ran by 11% and 14% of model size."""
        assert mc_memory.VRAM_MODEL_OVERHEAD_FRACTION >= 0.15

    def test_activation_headroom_still_scales_with_pass_size(self, memory):
        small = mc_memory.vram_required_bytes("A", None, 1024, 1024)
        large = mc_memory.vram_required_bytes("A", None, 2048, 2048)
        assert large > small

    def test_make_vram_room_frees_past_the_file_size(self, memory, monkeypatch):
        """"B" rather than "A": the estimate only applies to a model not yet loaded."""
        monkeypatch.setattr(mc_memory, "free_vram_bytes", lambda: 1 * GB)
        mc_memory.make_vram_room("B", None, 1024, 1024)

        assert memory.mm.freed[0] > 10 * GB + mc_memory.vram_headroom_bytes(1024, 1024)

    def test_the_preflight_plan_predicts_against_the_same_figure(self, memory, monkeypatch):
        """Otherwise the accordion promises dual residency the switch then breaks."""
        from modules import sd_models

        monkeypatch.setattr(mc_memory, "loaded_size_bytes", lambda model: 7 * GB)
        monkeypatch.setattr(
            sd_models, "get_closet_checkpoint_match",
            lambda name: types.SimpleNamespace(
                filename=f"/models/{name}", name_for_extra=name, sha256=f"sha-{name}"
            ),
        )
        # Exactly enough for the file plus its activations, and nothing more --
        # so only the overhead can decide this.
        monkeypatch.setattr(
            mc_memory, "free_vram_bytes", lambda: 10 * GB + mc_memory.vram_headroom_bytes()
        )

        assert mc_memory.plan("B").kind != "dual"


# --------------------------------------------------------------------------- #
# Never evicting the model the room is being made for
# --------------------------------------------------------------------------- #


class TestTargetIsSpared:
    """Regression: the preload's work was being thrown away and redone.

    Observed in a real log. The preload moved Krea 2's VAE, UNet and text
    encoder into VRAM (8.5s), and the very next generation opened with
    ``freed 13.8 GB VRAM for Stage 1 ... offloaded UnetPatcher, ModelPatcher,
    ModelPatcher`` and then re-loaded the same three. Asking to free the whole
    requirement while the target is already resident evicts the target, because
    the target's own weights are the largest evictable thing on the card.
    """

    def test_a_fully_resident_target_needs_no_eviction(self, memory, monkeypatch):
        """This is the preload case: everything is already where it belongs."""
        # 13 GB of model resident, leaving only the activation headroom to find.
        monkeypatch.setattr(mc_memory, "free_vram_bytes", lambda: 3 * GB)

        mc_memory.make_vram_room("A", None, 1024, 1024, stage=mc_memory.STAGE_1)

        assert memory.mm.freed == [], "the preloaded model must not be evicted"

    def test_the_target_is_never_offered_for_eviction(self, memory, monkeypatch):
        """Even when the pass genuinely needs more room than is free."""
        monkeypatch.setattr(mc_memory, "free_vram_bytes", lambda: 0.1 * GB)

        mc_memory.make_vram_room("A", None, 4096, 4096, stage=mc_memory.STAGE_1)

        spared = {entry.model.label for entry in memory.mm.kept[0]}
        assert spared == {"A-unet", "A-clip", "A-vae"}

    def test_only_the_shortfall_is_requested(self, memory, monkeypatch):
        """Not the whole requirement: 13 GB of it is already on the card."""
        monkeypatch.setattr(mc_memory, "free_vram_bytes", lambda: 0.5 * GB)

        mc_memory.make_vram_room("A", None, 1024, 1024, stage=mc_memory.STAGE_1)

        resident = 13 * GB
        requirement = resident + mc_memory.vram_headroom_bytes(1024, 1024)
        assert memory.mm.freed == [requirement - resident]

    def test_a_growing_pass_still_frees_the_difference(self, memory, monkeypatch):
        """Raising the resolution between generations must still make room."""
        monkeypatch.setattr(mc_memory, "free_vram_bytes", lambda: 0.5 * GB)

        mc_memory.make_vram_room("A", None, 2048, 2048, stage=mc_memory.STAGE_1)

        assert memory.mm.freed and memory.mm.freed[0] > 0

    def test_a_target_that_is_not_loaded_falls_back_to_the_disk_estimate(self, memory, monkeypatch):
        """A cold Stage 2: nothing of it is resident, so nothing is subtracted."""
        monkeypatch.setattr(mc_memory, "free_vram_bytes", lambda: 1 * GB)

        mc_memory.make_vram_room("B", None, 1024, 1024, stage=mc_memory.STAGE_2)

        expected = 10 * GB * (1 + mc_memory.VRAM_MODEL_OVERHEAD_FRACTION)
        assert memory.mm.freed[0] == pytest.approx(
            expected + mc_memory.vram_headroom_bytes(1024, 1024)
        )

    def test_the_real_model_size_beats_the_disk_estimate(self, memory, monkeypatch):
        """The patchers know their true size; the file is only a proxy."""
        monkeypatch.setattr(mc_memory, "free_vram_bytes", lambda: 0.5 * GB)
        monkeypatch.setattr(mc_memory, "file_size_bytes", lambda name, mods=None: 99 * GB)

        mc_memory.make_vram_room("A", None, 1024, 1024, stage=mc_memory.STAGE_1)

        # Sized from the 13 GB of resident patchers, not the bogus 99 GB file.
        assert memory.mm.freed == [mc_memory.vram_headroom_bytes(1024, 1024)]


# --------------------------------------------------------------------------- #
# Pinning Stage 1's encoders
# --------------------------------------------------------------------------- #


class TestPinnedEncoders:
    def test_capture_takes_the_encoders_but_not_the_unet(self, memory):
        assert mc_memory.capture_stage_1_encoders() == 2

        labels = {p.label for p in mc_memory._pinned_patchers}
        assert labels == {"A-clip", "A-vae"}

    def test_the_clip_wrapper_is_unwrapped_to_its_patcher(self, memory):
        mc_memory.capture_stage_1_encoders()
        assert memory.model.patchers.clip in mc_memory._pinned_patchers

    def test_stage_2_spares_the_pinned_encoders(self, memory, monkeypatch):
        monkeypatch.setattr(mc_memory, "free_vram_bytes", lambda: 1 * GB)
        mc_memory.capture_stage_1_encoders()

        mc_memory.make_vram_room("B", None, 1024, 1024, stage=mc_memory.STAGE_2)

        kept = {entry.model.label for entry in memory.mm.kept[0]}
        assert kept == {"A-clip", "A-vae"}

    def test_stage_1_spares_nothing(self, memory, monkeypatch):
        """The asymmetry is the point: Stage 2's encoders stay evictable."""
        monkeypatch.setattr(mc_memory, "free_vram_bytes", lambda: 1 * GB)
        mc_memory.capture_stage_1_encoders()

        mc_memory.make_vram_room("B", None, 1024, 1024, stage=mc_memory.STAGE_1)

        assert memory.mm.kept == [[]]

    def test_the_pin_is_dropped_when_stage_2_would_not_fit_alongside(self, memory, monkeypatch):
        """Pinning more than the card can spare is worse than not pinning.

        free_memory() would fall short of its target and load_models_gpu() would
        make up the difference the slow way -- the exact thrash being avoided.
        """
        monkeypatch.setattr(mc_memory, "free_vram_bytes", lambda: 1 * GB)
        monkeypatch.setattr(mc_memory, "total_vram_bytes", lambda: 12 * GB)
        mc_memory.capture_stage_1_encoders()

        mc_memory.make_vram_room("B", None, 1024, 1024, stage=mc_memory.STAGE_2)

        assert memory.mm.kept == [[]]

    def test_a_host_without_keep_loaded_still_gets_its_vram_freed(self, memory, monkeypatch):
        """Losing the pin is acceptable; losing the eviction is not."""
        calls = []

        def free_memory(required, device):
            calls.append(required)
            return []

        monkeypatch.setattr(memory.mm, "free_memory", free_memory)
        monkeypatch.setattr(mc_memory, "free_vram_bytes", lambda: 1 * GB)
        mc_memory.capture_stage_1_encoders()

        mc_memory.make_vram_room("B", None, 1024, 1024, stage=mc_memory.STAGE_2)

        assert calls, "the eviction must still happen without keep_loaded support"

    def test_a_cold_stage_2_load_leaves_nothing_to_pin(self, memory, monkeypatch):
        """forge_model_reload() unloads everything, so the pin cannot survive it.

        Only the warm path can hold encoders across the switch. This must
        degrade quietly rather than pass stale entries to the host.
        """
        monkeypatch.setattr(mc_memory, "free_vram_bytes", lambda: 1 * GB)
        mc_memory.capture_stage_1_encoders()
        memory.mm.current_loaded_models.clear()  # what unload_all_models() leaves

        mc_memory.make_vram_room("B", None, 1024, 1024, stage=mc_memory.STAGE_2)

        assert memory.mm.kept == [[]]

    def test_encoders_already_evicted_are_not_pinned(self, memory, monkeypatch):
        monkeypatch.setattr(mc_memory, "free_vram_bytes", lambda: 1 * GB)
        mc_memory.capture_stage_1_encoders()
        memory.mm.current_loaded_models.clear()

        mc_memory.make_vram_room("B", None, 1024, 1024, stage=mc_memory.STAGE_2)

        assert memory.mm.kept == [[]]

    def test_the_setting_disables_capture(self, memory, host):
        host.shared.opts.model_chain_pin_stage1_encoders = False
        assert mc_memory.capture_stage_1_encoders() == 0

    def test_a_missing_setting_keeps_the_documented_default(self, memory, host):
        """Settings do not exist until the UI builds them, and in tests never."""
        assert not hasattr(host.shared.opts, mc_memory.OPT_PIN_ENCODERS)
        assert mc_memory.capture_stage_1_encoders() == 2

    def test_capture_survives_a_model_without_forge_objects(self, memory, host):
        host.sd_models.model_data.sd_model = types.SimpleNamespace(name="bare")
        assert mc_memory.capture_stage_1_encoders() == 0


# --------------------------------------------------------------------------- #
# The background Stage 1 preload
# --------------------------------------------------------------------------- #


@pytest.fixture
def preload(memory, monkeypatch):
    """Records what the preload worker did, with the swap itself faked."""
    calls = types.SimpleNamespace(reinstated=0, rooms=[], gpu_loads=0)

    def reinstate_pending():
        calls.reinstated += 1
        return True

    monkeypatch.setattr(mc_memory, "reinstate_pending", reinstate_pending)
    monkeypatch.setattr(
        mc_memory, "make_vram_room",
        lambda name, mods=None, w=0, h=0, stage=mc_memory.STAGE_2: calls.rooms.append((name, w, h, stage)),
    )
    monkeypatch.setattr(
        mc_memory, "_load_current_to_gpu",
        lambda: (calls.__setattr__("gpu_loads", calls.gpu_loads + 1), 4 * GB)[1],
    )
    monkeypatch.setattr(mc_memory, "_pending_restore", "A", raising=False)

    yield calls

    mc_memory.join_preload(timeout=5)
    mc_memory.consume_preload()


class TestPreload:
    def test_nothing_pending_means_no_preload(self, preload, monkeypatch):
        monkeypatch.setattr(mc_memory, "_pending_restore", None)
        assert mc_memory.preload_async(1024, 1024) is False
        assert preload.reinstated == 0

    def test_the_preload_swaps_stage_1_in_and_moves_it_to_the_gpu(self, preload):
        assert mc_memory.preload_async(1024, 1024) is True
        mc_memory.join_preload(timeout=5)

        assert preload.reinstated == 1
        assert preload.gpu_loads == 1

    def test_the_preload_makes_room_as_stage_1(self, preload):
        mc_memory.preload_async(1024, 1024)
        mc_memory.join_preload(timeout=5)

        assert [room[3] for room in preload.rooms] == [mc_memory.STAGE_1]

    def test_the_next_generation_learns_the_swap_already_happened(self, preload):
        mc_memory.preload_async(1024, 1024)
        mc_memory.join_preload(timeout=5)

        assert mc_memory.consume_preload() is True

    def test_consuming_is_one_shot(self, preload):
        mc_memory.preload_async(1024, 1024)
        mc_memory.join_preload(timeout=5)

        mc_memory.consume_preload()
        assert mc_memory.consume_preload() is False

    def test_a_preload_that_found_nothing_to_do_reports_nothing(self, preload, monkeypatch):
        monkeypatch.setattr(mc_memory, "reinstate_pending", lambda: False)
        mc_memory.preload_async(1024, 1024)
        mc_memory.join_preload(timeout=5)

        assert mc_memory.consume_preload() is False

    def test_a_failing_preload_does_not_escape(self, preload, monkeypatch):
        def boom():
            raise RuntimeError("driver said no")

        monkeypatch.setattr(mc_memory, "reinstate_pending", boom)
        mc_memory.preload_async(1024, 1024)
        mc_memory.join_preload(timeout=5)  # must not raise

        assert preload.gpu_loads == 0

    def test_starting_a_second_preload_waits_for_the_first(self, preload):
        mc_memory.preload_async(1024, 1024)
        mc_memory.preload_async(1024, 1024)
        mc_memory.join_preload(timeout=5)

        assert preload.reinstated == 2

    def test_the_setting_disables_it(self, preload, host):
        host.shared.opts.model_chain_preload_stage1 = False
        assert mc_memory.preload_async(1024, 1024) is False
        assert preload.reinstated == 0

    def test_joining_when_nothing_runs_is_a_no_op(self, preload):
        mc_memory.join_preload(timeout=5)

    def test_ensure_resident_waits_for_an_in_flight_preload(self, preload, monkeypatch):
        """Two threads must never move weights at once."""
        joins = []
        monkeypatch.setattr(mc_memory, "join_preload", lambda timeout=None: joins.append(timeout))
        monkeypatch.setattr(mc_memory, "checkpoint_info", lambda name: None)

        with pytest.raises(mc_memory.ModelChainError):
            mc_memory.ensure_resident("missing")

        assert joins, "ensure_resident must join the preload before touching models"


class TestLoadCurrentToGpu:
    def test_every_patcher_is_handed_to_the_host(self, memory):
        mc_memory._load_current_to_gpu()

        handed = {p.label for p in memory.mm.loaded_to_gpu[0]}
        assert handed == {"A-unet", "A-clip", "A-vae"}

    def test_a_model_without_patchers_is_skipped(self, memory, host):
        host.sd_models.model_data.sd_model = types.SimpleNamespace(name="bare")
        assert mc_memory._load_current_to_gpu() == 0
        assert memory.mm.loaded_to_gpu == []


# --------------------------------------------------------------------------- #
# How the orchestration drives all of the above
# --------------------------------------------------------------------------- #


@pytest.fixture
def wired(chain, host, monkeypatch, image_factory):
    """A full generation with the residency calls recorded rather than performed."""
    from test_orchestration import make_p, make_processed, run_chain

    calls = types.SimpleNamespace(captures=0, preloads=[], joins=0, rooms=[], consumed=False)

    monkeypatch.setattr(
        mc_memory, "capture_stage_1_encoders",
        lambda: calls.__setattr__("captures", calls.captures + 1) or 2,
    )
    monkeypatch.setattr(
        mc_memory, "preload_async",
        lambda w=0, h=0: calls.preloads.append((w, h)) or True,
    )
    monkeypatch.setattr(
        mc_memory, "join_preload",
        lambda timeout=None: calls.__setattr__("joins", calls.joins + 1),
    )
    monkeypatch.setattr(
        mc_memory, "make_vram_room",
        lambda name, mods=None, w=0, h=0, stage=mc_memory.STAGE_2: calls.rooms.append((w, h, stage)),
    )
    monkeypatch.setattr(mc_memory, "current_modules", lambda: [])

    def run(**overrides):
        p = make_p(host, **{k: overrides.pop(k) for k in ("width", "height") if k in overrides})
        return run_chain(chain, host, p, make_processed(host, p, image_factory), **overrides)

    return types.SimpleNamespace(run=run, calls=calls, chain=chain)


class TestOrchestrationWiring:
    def test_stage_2_captures_the_encoders_before_switching(self, wired):
        wired.run()
        assert wired.calls.captures == 1

    def test_a_preload_starts_once_stage_2_is_done(self, wired):
        wired.run()
        assert wired.calls.preloads == [(1024, 1024)]

    def test_the_preload_is_sized_from_the_stage_1_pass(self, wired):
        wired.run(width=768, height=512)
        assert wired.calls.preloads == [(768, 512)]

    def test_the_next_generation_joins_before_touching_models(self, wired):
        wired.run()
        assert wired.calls.joins >= 1

    def test_a_preloaded_swap_still_gets_its_vram_budget_rechecked(self, wired, monkeypatch):
        """The preload sized itself from the *previous* generation's resolution.

        Growing the image between generations would otherwise leave Stage 1
        running in a budget freed for a smaller pass.
        """
        monkeypatch.setattr(mc_memory, "reinstate_pending", lambda: False)
        monkeypatch.setattr(mc_memory, "consume_preload", lambda: True)

        wired.run(width=1536, height=1536)

        assert (1536, 1536, mc_memory.STAGE_1) in wired.calls.rooms

    def test_no_swap_and_no_preload_means_no_vram_work(self, wired, monkeypatch):
        monkeypatch.setattr(mc_memory, "reinstate_pending", lambda: False)
        monkeypatch.setattr(mc_memory, "consume_preload", lambda: False)

        wired.run()

        assert not [room for room in wired.calls.rooms if room[2] == mc_memory.STAGE_1]

    def test_a_failing_preload_does_not_break_the_generation(self, wired, monkeypatch):
        def boom(w=0, h=0):
            raise RuntimeError("thread refused to start")

        monkeypatch.setattr(mc_memory, "preload_async", boom)

        processed = wired.run()

        assert processed.images, "images must still be returned"


class TestSettingsRegistration:
    """The registered default and the code's fallback have to agree.

    ``option()`` falls back to the documented behaviour when a setting is
    missing, so a registration defaulting the other way would leave the feature
    off in the UI while the fallback reported it on.
    """

    @pytest.mark.parametrize(
        "name", [mc_memory.OPT_PRELOAD, mc_memory.OPT_PIN_ENCODERS]
    )
    def test_registered_on_by_default(self, host, name):
        import model_chain  # noqa: F401  (registration happens on import)

        assert host.shared.options_templates[name].default is True
        assert mc_memory.option(name, True) is True


class TestModelPatchers:
    def test_a_patcher_shared_between_slots_is_listed_once(self, memory):
        shared_patcher = FakePatcher("shared", 1 * GB)
        model = types.SimpleNamespace(
            forge_objects=types.SimpleNamespace(
                unet=shared_patcher, clip=FakeClip(shared_patcher), vae=shared_patcher
            )
        )
        assert mc_memory.model_patchers(model) == [shared_patcher]

    def test_the_placeholder_model_has_none(self, memory):
        class FakeInitialModel:
            forge_objects = None

        assert mc_memory.model_patchers(FakeInitialModel()) == []
