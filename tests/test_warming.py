"""Speed-first VRAM warming, and the reserve that keeps it safe.

The feature is an asymmetric bet. Warming a component that turns out to be
needed saves seconds of PCIe traffic; warming one that turns out to be in the
way costs an out-of-memory error, or -- far more likely and far harder to
diagnose -- the driver silently spilling into system RAM and sampling dropping
from sub-second to tens of seconds per step with nothing in the log.

So the tests are lopsided on purpose. A handful cover warming happening; most
cover it *not* happening, and specifically not happening at the expense of the
reserve, of the user's manual floor, of Forge's own reservation, or of Stage 1.
"""

from __future__ import annotations

import types

import pytest

import mc_memory
from test_residency_speed import FakeLoadedModel, make_model

GB = 1024**3


def static(width, height):
    """The a-priori estimate alone, for tests asserting nothing raised it.

    Not ``VRAM_HEADROOM_BYTES``: 1024x1024 is 1.049 megapixels, so even the
    baseline pass carries a little of the per-megapixel term.
    """
    return mc_memory._static_headroom_bytes(width, height)


@pytest.fixture
def warming(host, monkeypatch):
    """A card with Stage 1 resident and Stage 2's components captured."""
    from backend import memory_management

    memory_management.freed.clear()
    memory_management.kept.clear()
    memory_management.loaded_to_gpu.clear()
    memory_management.current_loaded_models.clear()

    stage_1 = make_model("A")
    stage_2 = make_model("B", unet=6 * GB, clip=2 * GB, vae=1 * GB)

    host.sd_models.model_data.sd_model = stage_1
    memory_management.current_loaded_models.extend(
        FakeLoadedModel(p)
        for p in (stage_1.patchers.unet, stage_1.patchers.clip, stage_1.patchers.vae)
    )

    monkeypatch.setattr(mc_memory, "file_size_bytes", lambda name, mods=None: 10 * GB)
    monkeypatch.setattr(mc_memory, "current_modules", lambda: [])

    # Learned peaks are module state and would otherwise leak between tests.
    monkeypatch.setattr(mc_memory, "_peak_bytes_per_megapixel", 0.0, raising=False)
    monkeypatch.setattr(mc_memory, "_peak_observations", 0, raising=False)
    mc_memory.clear_stage_2_components()

    yield types.SimpleNamespace(mm=memory_management, stage_1=stage_1, stage_2=stage_2)

    mc_memory.clear_stage_2_components()


def capture(warming, host):
    """Capture Stage 2's patchers the way the swap back to Stage 1 does.

    Opts the preload in, because warming only ever runs on its thread -- see
    TestWarmingIsConfinedToThePreload for why, and for the test that holds that
    boundary in place.
    """
    host.shared.opts.model_chain_preload_stage1 = True
    host.sd_models.model_data.sd_model = warming.stage_2
    count = mc_memory.capture_stage_2_components()
    host.sd_models.model_data.sd_model = warming.stage_1
    return count


# --------------------------------------------------------------------------- #
# The reserve
# --------------------------------------------------------------------------- #


class TestReserve:
    """Four floors, largest wins. Each is an answer to the same question."""

    def test_the_static_estimate_still_applies(self, warming):
        assert mc_memory.vram_headroom_bytes(1024, 1024) == static(1024, 1024)

    def test_it_still_scales_with_the_pass(self, warming):
        assert mc_memory.vram_headroom_bytes(2048, 2048) > mc_memory.vram_headroom_bytes(1024, 1024)

    def test_a_manual_reserve_raises_it(self, warming, host):
        host.shared.opts.model_chain_vram_reserve_gb = 6.0
        assert mc_memory.vram_headroom_bytes(1024, 1024) == 6 * GB

    def test_a_manual_reserve_below_the_estimate_does_not_lower_it(self, warming, host):
        """A floor, not an override. Undercutting the estimate is not on offer."""
        host.shared.opts.model_chain_vram_reserve_gb = 0.25
        assert mc_memory.vram_headroom_bytes(2048, 2048) > 0.25 * GB

    def test_zero_means_automatic(self, warming, host):
        host.shared.opts.model_chain_vram_reserve_gb = 0.0
        assert mc_memory.vram_headroom_bytes(1024, 1024) == static(1024, 1024)

    def test_a_nonsense_setting_is_ignored_rather_than_crashing(self, warming, host):
        host.shared.opts.model_chain_vram_reserve_gb = "lots"
        assert mc_memory.vram_headroom_bytes(1024, 1024) == static(1024, 1024)

    def test_forge_s_own_reservation_is_never_undercut(self, warming, monkeypatch):
        """Cancelling a reservation the user made in the host's settings would
        be this extension quietly overruling the host."""
        monkeypatch.setattr(
            warming.mm, "minimum_inference_memory", lambda: 5 * GB, raising=False
        )
        assert mc_memory.vram_headroom_bytes(1024, 1024) >= 5 * GB

    def test_the_largest_host_figure_wins(self, warming, monkeypatch):
        monkeypatch.setattr(warming.mm, "minimum_inference_memory", lambda: 2 * GB, raising=False)
        monkeypatch.setattr(warming.mm, "current_inference_memory", 7 * GB, raising=False)
        assert mc_memory.host_reserved_bytes() == 7 * GB

    def test_a_host_that_exposes_none_of_them_reports_nothing(self, warming):
        assert mc_memory.host_reserved_bytes() == 0

    def test_a_host_figure_that_raises_is_skipped(self, warming, monkeypatch):
        def boom():
            raise RuntimeError("no such device")

        monkeypatch.setattr(warming.mm, "minimum_inference_memory", boom, raising=False)
        assert mc_memory.host_reserved_bytes() == 0


class TestObservedPeaks:
    """Automatic mode is meant to get better at this, not stay a guess."""

    @pytest.fixture(autouse=True)
    def torch(self, warming, monkeypatch):
        """A fake ``torch.cuda`` whose peak reading the test controls."""
        import sys

        torch = types.ModuleType("torch")
        torch.cuda = types.SimpleNamespace(
            max_memory_allocated=lambda device=None: torch.peak,
            reset_peak_memory_stats=lambda device=None: setattr(torch, "resets", torch.resets + 1),
        )
        torch.peak = 0
        torch.resets = 0
        monkeypatch.setitem(sys.modules, "torch", torch)
        return torch

    def test_an_observed_peak_raises_the_reserve(self, warming, torch):
        baseline = mc_memory.vram_headroom_bytes(1024, 1024)
        # 13 GB of weights are resident, so this reads as 4 GB of activations.
        torch.peak = 17 * GB
        mc_memory.observe_activation_peak(1024, 1024)

        assert mc_memory.vram_headroom_bytes(1024, 1024) > baseline

    def test_the_observation_carries_a_margin(self, warming, torch):
        """A peak is the largest thing that happened, not the largest possible."""
        torch.peak = 15 * GB  # 2 GB of activations over 1 megapixel
        mc_memory.observe_activation_peak(1024, 1024)

        assert mc_memory.vram_headroom_bytes(1024, 1024) == pytest.approx(
            2 * GB * mc_memory.PEAK_MARGIN, rel=0.01
        )

    def test_it_scales_the_learned_figure_by_pass_size(self, warming, torch):
        torch.peak = 15 * GB
        mc_memory.observe_activation_peak(1024, 1024)

        small = mc_memory.vram_headroom_bytes(1024, 1024)
        large = mc_memory.vram_headroom_bytes(2048, 2048)
        assert large == pytest.approx(small * 4, rel=0.01)

    def test_a_cheaper_pass_never_lowers_it(self, warming, torch):
        """Evidence that one pass was cheap is not evidence the next will be."""
        torch.peak = 16 * GB
        mc_memory.observe_activation_peak(1024, 1024)
        high = mc_memory.vram_headroom_bytes(1024, 1024)

        torch.peak = 13 * GB
        mc_memory.observe_activation_peak(1024, 1024)

        assert mc_memory.vram_headroom_bytes(1024, 1024) == high

    def test_a_wild_reading_is_capped(self, warming, torch):
        """A pass that evicted a model mid-flight attributes it to activations."""
        torch.peak = 400 * GB
        mc_memory.observe_activation_peak(1024, 1024)

        ceiling = mc_memory.PEAK_CEILING * static(1024, 1024) * mc_memory.PEAK_MARGIN
        assert mc_memory.vram_headroom_bytes(1024, 1024) <= ceiling

    def test_a_peak_below_the_static_estimate_changes_nothing(self, warming, torch):
        torch.peak = 13 * GB + 1  # essentially no activations
        mc_memory.observe_activation_peak(1024, 1024)

        assert mc_memory.vram_headroom_bytes(1024, 1024) == static(1024, 1024)

    def test_the_window_is_reset_before_a_pass(self, warming, torch):
        mc_memory.begin_pass_observation()
        assert torch.resets == 1

    def test_observations_are_counted(self, warming, torch):
        torch.peak = 15 * GB
        mc_memory.observe_activation_peak(1024, 1024)
        mc_memory.observe_activation_peak(1024, 1024)

        assert mc_memory.observed_peaks()[1] == 2

    def test_a_pass_with_no_size_is_not_measured(self, warming, torch):
        torch.peak = 99 * GB
        assert mc_memory.observe_activation_peak(0, 0) == 0

    def test_a_host_without_cuda_stats_degrades_quietly(self, warming, monkeypatch):
        import sys

        monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
        assert mc_memory.observe_activation_peak(1024, 1024) == 0
        mc_memory.begin_pass_observation()  # must not raise


# --------------------------------------------------------------------------- #
# Warming Stage 2 with what is left
# --------------------------------------------------------------------------- #


class TestSecondaryWarming:
    def test_stage_2_is_warmed_when_there_is_room(self, warming, host, monkeypatch):
        capture(warming, host)
        monkeypatch.setattr(mc_memory, "free_vram_bytes", lambda: 12 * GB)

        mc_memory.warm_secondary(1024, 1024)

        warmed = {p.label for p in warming.mm.loaded_to_gpu[0]}
        assert warmed == {"B-unet", "B-clip", "B-vae"}

    def test_the_reserve_is_handed_to_the_host(self, warming, host, monkeypatch):
        """So the host keeps it free while it loads, not merely afterwards."""
        calls = []
        monkeypatch.setattr(
            warming.mm, "load_models_gpu",
            lambda models, **kw: calls.append(kw.get("memory_required")),
        )
        capture(warming, host)
        monkeypatch.setattr(mc_memory, "free_vram_bytes", lambda: 12 * GB)

        mc_memory.warm_secondary(1024, 1024)

        assert calls == [mc_memory.vram_headroom_bytes(1024, 1024)]

    def test_nothing_is_warmed_into_the_reserve(self, warming, host, monkeypatch):
        """9 GB of Stage 2 against 9 GB free, 1 GB of which is reserved."""
        capture(warming, host)
        monkeypatch.setattr(mc_memory, "free_vram_bytes", lambda: 9 * GB)

        mc_memory.warm_secondary(1024, 1024)

        warmed = {p.label for p in warming.mm.loaded_to_gpu[0]}
        assert "B-unet" not in warmed

    def test_a_manual_reserve_is_never_consumed(self, warming, host, monkeypatch):
        capture(warming, host)
        host.shared.opts.model_chain_vram_reserve_gb = 11.0
        monkeypatch.setattr(mc_memory, "free_vram_bytes", lambda: 12 * GB)

        mc_memory.warm_secondary(1024, 1024)

        assert warming.mm.loaded_to_gpu == []

    def test_the_largest_component_is_dropped_first(self, warming, host, monkeypatch):
        """Encoders are small and a disproportionate share of a switch's cost,
        so "the UNet did not fit" must not mean "nothing was warmed"."""
        capture(warming, host)
        monkeypatch.setattr(mc_memory, "free_vram_bytes", lambda: 5 * GB)

        mc_memory.warm_secondary(1024, 1024)

        warmed = {p.label for p in warming.mm.loaded_to_gpu[0]}
        assert warmed == {"B-clip", "B-vae"}

    def test_nothing_fits_means_nothing_is_moved(self, warming, host, monkeypatch):
        capture(warming, host)
        monkeypatch.setattr(mc_memory, "free_vram_bytes", lambda: 1.4 * GB)

        assert mc_memory.warm_secondary(1024, 1024) == 0
        assert warming.mm.loaded_to_gpu == []

    def test_a_full_card_warms_nothing(self, warming, host, monkeypatch):
        capture(warming, host)
        monkeypatch.setattr(mc_memory, "free_vram_bytes", lambda: 0.1 * GB)

        assert mc_memory.warm_secondary(1024, 1024) == 0

    def test_a_bigger_pass_leaves_less_to_warm_with(self, warming, host, monkeypatch):
        """The reserve grows with the pass, so the spare shrinks with it."""
        capture(warming, host)
        monkeypatch.setattr(mc_memory, "free_vram_bytes", lambda: 10 * GB)

        mc_memory.warm_secondary(4096, 4096)

        assert warming.mm.loaded_to_gpu in ([], [[]]) or all(
            p.label != "B-unet" for p in warming.mm.loaded_to_gpu[0]
        )

    def test_components_already_resident_cost_nothing_to_keep(self, warming, host, monkeypatch):
        """Warming is sized on what still has to *move*, not on total size."""
        capture(warming, host)
        warming.mm.current_loaded_models.extend(
            FakeLoadedModel(p) for p in (warming.stage_2.patchers.unet,)
        )
        monkeypatch.setattr(mc_memory, "free_vram_bytes", lambda: 5 * GB)

        mc_memory.warm_secondary(1024, 1024)

        warmed = {p.label for p in warming.mm.loaded_to_gpu[0]}
        assert warmed == {"B-unet", "B-clip", "B-vae"}

    def test_the_setting_disables_capture(self, warming, host):
        host.shared.opts.model_chain_warm_stage_2 = False
        assert capture(warming, host) == 0

    def test_the_setting_disables_warming(self, warming, host, monkeypatch):
        capture(warming, host)
        host.shared.opts.model_chain_warm_stage_2 = False
        monkeypatch.setattr(mc_memory, "free_vram_bytes", lambda: 20 * GB)

        assert mc_memory.warm_secondary(1024, 1024) == 0

    def test_nothing_captured_means_nothing_to_do(self, warming, monkeypatch):
        monkeypatch.setattr(mc_memory, "free_vram_bytes", lambda: 20 * GB)
        assert mc_memory.warm_secondary(1024, 1024) == 0

    def test_a_host_that_rejects_memory_required_still_warms(self, warming, host, monkeypatch):
        """Losing the hint is acceptable; the components were sized to fit anyway."""
        calls = []

        def load_models_gpu(models):
            calls.append(list(models))

        monkeypatch.setattr(warming.mm, "load_models_gpu", load_models_gpu)
        capture(warming, host)
        monkeypatch.setattr(mc_memory, "free_vram_bytes", lambda: 12 * GB)

        mc_memory.warm_secondary(1024, 1024)

        assert calls, "the warm must still happen without memory_required support"

    def test_a_failing_warm_does_not_escape(self, warming, host, monkeypatch):
        def boom(models, **kwargs):
            raise RuntimeError("driver said no")

        monkeypatch.setattr(warming.mm, "load_models_gpu", boom)
        capture(warming, host)
        monkeypatch.setattr(mc_memory, "free_vram_bytes", lambda: 12 * GB)

        assert mc_memory.warm_secondary(1024, 1024) == 0

    def test_a_capture_from_a_bare_model_is_empty(self, warming, host):
        host.sd_models.model_data.sd_model = types.SimpleNamespace(name="bare")
        assert mc_memory.capture_stage_2_components() == 0


class TestStage1KeepsPriority:
    """The whole ordering rests on this: Stage 2 gets the leftovers, only."""

    def test_warm_components_are_evicted_before_stage_1_runs_short(self, warming, host, monkeypatch):
        """Warm Stage 2 components are in no keep list, so the next pass takes
        their VRAM back without ceremony."""
        capture(warming, host)
        warming.mm.current_loaded_models.extend(
            FakeLoadedModel(p) for p in (warming.stage_2.patchers.unet,)
        )
        monkeypatch.setattr(mc_memory, "free_vram_bytes", lambda: 0.5 * GB)
        monkeypatch.setattr(
            host.sd_models, "get_closet_checkpoint_match",
            lambda name: types.SimpleNamespace(
                filename=f"/models/{name}", name_for_extra=name, sha256=f"sha-{name}"
            ),
        )

        mc_memory.make_vram_room("A", None, 1024, 1024, stage=mc_memory.STAGE_1)

        spared = {entry.model.label for entry in warming.mm.kept[0]}
        assert "B-unet" not in spared
        assert spared == {"A-unet", "A-clip", "A-vae"}

    def test_the_load_tells_the_host_what_to_keep_free(self, warming, host, monkeypatch):
        """Filling the card would be undone moments later by a partial unload."""
        calls = []
        monkeypatch.setattr(
            warming.mm, "load_models_gpu",
            lambda models, **kw: calls.append(kw.get("memory_required")),
        )

        mc_memory._load_current_to_gpu(1024, 1024)

        assert calls == [mc_memory.vram_headroom_bytes(1024, 1024)]

    def test_a_host_without_memory_required_still_gets_its_load(self, warming, monkeypatch):
        calls = []
        monkeypatch.setattr(warming.mm, "load_models_gpu", lambda models: calls.append(list(models)))

        mc_memory._load_current_to_gpu(1024, 1024)

        assert calls, "losing the hint must not lose the load"


class TestWarmingIsConfinedToThePreload:
    """Regression, from a real crash: warming may only run on the preload thread.

    Loading weights *in* rewrites and re-patches them. Done from a script hook
    rather than from inside the sampler, that left the model in a state the very
    next sampling step rejected outright:

        RuntimeError: Inference tensors do not track version counter

    -- the same failure the preload's own notes describe, reached on the path
    that had opted *out* of the preload. The preload has a deliberate answer to
    it, an opt-in switch and a circuit breaker; a hook on the generation thread
    has none of the three. So the boundary is: this module frees VRAM from
    anywhere and fills it from one place only.
    """

    def test_before_process_never_initiates_a_gpu_load(self, warming, chain, host, monkeypatch):
        """The crash, as a test. Freeing is fine; loading is not."""
        from test_orchestration import make_p

        monkeypatch.setattr(mc_memory, "reinstate_pending", lambda: True)
        monkeypatch.setattr(mc_memory, "make_vram_room", lambda *a, **k: 0)

        chain.script.before_process(make_p(host), False, "None")

        assert warming.mm.loaded_to_gpu == [], "before_process must not move weights in"

    def test_nothing_is_captured_when_the_preload_is_off(self, warming, host):
        """Otherwise gigabytes stay alive for a warm-up that never runs."""
        host.shared.opts.model_chain_preload_stage1 = False
        host.sd_models.model_data.sd_model = warming.stage_2

        assert mc_memory.capture_stage_2_components() == 0
        assert mc_memory._stage_2_patchers == []

    @pytest.fixture
    def swapped_back(self, warming, host, monkeypatch):
        """A cache holding Stage 1, with Stage 2 the model currently loaded."""
        mc_memory._cache.clear()
        monkeypatch.setattr(mc_memory, "_loading_parameters_key", lambda: "key-A")
        monkeypatch.setattr(mc_memory, "_loaded_model_key", lambda: "key-B")
        monkeypatch.setattr(mc_memory, "loaded_size_bytes", lambda model: 8 * GB)
        monkeypatch.setattr(mc_memory, "free_ram_bytes", lambda: 64 * GB)
        monkeypatch.setattr(mc_memory, "total_ram_bytes", lambda: 96 * GB)
        monkeypatch.setattr(mc_memory, "_pending_restore", "A", raising=False)

        mc_memory._cache.admit(
            mc_memory._Entry(
                key="key-A", checkpoint_name="A", sd_model=warming.stage_1, size_bytes=8 * GB
            ),
            64 * GB,
        )
        host.sd_models.model_data.sd_model = warming.stage_2

        yield warming

        mc_memory._cache.clear()

    def test_the_swap_back_captures_stage_2(self, swapped_back, host):
        host.shared.opts.model_chain_preload_stage1 = True

        assert mc_memory.reinstate_pending() is True
        assert {p.label for p in mc_memory._stage_2_patchers} == {"B-unet", "B-clip", "B-vae"}

    def test_the_swap_back_captures_nothing_with_the_preload_off(self, swapped_back, host):
        host.shared.opts.model_chain_preload_stage1 = False

        mc_memory.reinstate_pending()

        assert mc_memory._stage_2_patchers == []

    def test_the_setting_still_governs_it(self, swapped_back, host):
        host.shared.opts.model_chain_preload_stage1 = True
        host.shared.opts.model_chain_warm_stage_2 = False

        mc_memory.reinstate_pending()

        assert mc_memory._stage_2_patchers == []

    def test_the_preload_warms_stage_2_only_after_stage_1(self, warming, host, monkeypatch):
        """Ordering inside the worker, asserted rather than assumed."""
        order = []

        monkeypatch.setattr(mc_memory, "reinstate_pending", lambda: True)
        monkeypatch.setattr(mc_memory, "make_vram_room", lambda *a, **k: 0)
        monkeypatch.setattr(
            mc_memory, "_load_current_to_gpu", lambda w=0, h=0: order.append("stage 1") or 4 * GB
        )
        monkeypatch.setattr(
            mc_memory, "warm_secondary", lambda w=0, h=0: order.append("stage 2") or 0
        )
        monkeypatch.setattr(mc_memory, "_pending_restore", "A", raising=False)
        host.shared.opts.model_chain_preload_stage1 = True

        mc_memory.preload_async(1024, 1024)
        mc_memory.join_preload(timeout=5)
        mc_memory.consume_preload()

        assert order == ["stage 1", "stage 2"]
