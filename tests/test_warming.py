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

import math
import sys
import types

import pytest

import mc_memory
from test_residency_speed import FakeLoadedModel, FakePatcher, make_model

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


class TestTheRequirementStaysAttainable:
    """Asking to free more VRAM than the card holds fails in the worst way.

    ``free_memory`` cannot reach an impossible target, so it evicts everything
    it is allowed to and still reports a shortfall -- the pass then runs having
    thrown away models it could have kept, and the console warns about spilling
    on every single generation. A real log did exactly this: a 13.9 GB model
    asking for 24.3 GB on a 24 GB card, every cycle, because a learned reserve
    had grown to 10.4 GB.
    """

    def test_the_requirement_never_exceeds_the_card(self, warming, monkeypatch):
        monkeypatch.setattr(mc_memory, "total_vram_bytes", lambda: 24 * GB)
        monkeypatch.setattr(mc_memory, "vram_headroom_bytes", lambda w=0, h=0, batch=1: 20 * GB)

        required = mc_memory._pass_requirement("A", None, 1024, 1024, [])

        assert required <= 24 * GB

    def test_the_model_is_kept_and_the_margin_is_what_gives(self, warming, monkeypatch):
        """The model has to be resident to sample at all; the reserve is a
        margin, and one that cannot be honoured is better spent than pretended."""
        monkeypatch.setattr(mc_memory, "total_vram_bytes", lambda: 16 * GB)
        monkeypatch.setattr(mc_memory, "vram_headroom_bytes", lambda w=0, h=0, batch=1: 20 * GB)
        monkeypatch.setattr(mc_memory, "file_size_bytes", lambda name, mods=None: 12 * GB)

        model = 12 * GB * (1 + mc_memory.VRAM_MODEL_OVERHEAD_FRACTION)
        assert mc_memory._pass_requirement("A", None, 1024, 1024, []) >= model

    def test_an_unknowable_card_size_changes_nothing(self, warming, monkeypatch):
        monkeypatch.setattr(mc_memory, "total_vram_bytes", lambda: 0)
        monkeypatch.setattr(mc_memory, "vram_headroom_bytes", lambda w=0, h=0, batch=1: 20 * GB)

        assert mc_memory._pass_requirement("A", None, 1024, 1024, []) > 20 * GB

    def test_an_estimated_requirement_is_trimmed_the_same_way(self, warming, monkeypatch):
        """``vram_required_bytes`` is the file-size twin of ``_pass_requirement``
        and the plan reaches for whichever one has an answer. A phase must not
        be described as fitting or not fitting according to whether the
        checkpoint behind it happens to have been loaded once already."""
        monkeypatch.setattr(mc_memory, "total_vram_bytes", lambda: 24 * GB)
        monkeypatch.setattr(mc_memory, "vram_headroom_bytes", lambda w=0, h=0, batch=1: 20 * GB)
        monkeypatch.setattr(mc_memory, "file_size_bytes", lambda name, mods=None: 18 * GB)

        assert mc_memory.vram_required_bytes("A", None, 1024, 1024) <= 24 * GB

    def test_a_pass_that_fits_is_left_alone(self, warming, monkeypatch):
        monkeypatch.setattr(mc_memory, "total_vram_bytes", lambda: 24 * GB)
        monkeypatch.setattr(mc_memory, "vram_headroom_bytes", lambda w=0, h=0, batch=1: 2 * GB)

        patchers = [warming.stage_1.patchers.unet]  # 8 GB
        assert mc_memory._pass_requirement("A", None, 1024, 1024, patchers) == 10 * GB


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
        torch.peak = 14.5 * GB  # 1.5 GB of activations over ~1 megapixel
        mc_memory.observe_activation_peak(1024, 1024)

        assert mc_memory.vram_headroom_bytes(1024, 1024) == pytest.approx(
            1.5 * GB * mc_memory.PEAK_MARGIN, rel=0.01
        )

    def test_it_scales_the_learned_figure_by_pass_size(self, warming, torch, monkeypatch):
        # A card large enough that the fraction cap cannot be what is being
        # measured here; scaling is.
        monkeypatch.setattr(mc_memory, "total_vram_bytes", lambda: 80 * GB)
        torch.peak = 14.5 * GB
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

        assert mc_memory.observed_peaks()[0] == mc_memory.MAX_LEARNED_BYTES_PER_MEGAPIXEL

    def test_a_small_pass_cannot_authorise_a_huge_rate_for_a_large_one(self, warming, torch):
        """Regression, from a real log. The ceiling used to be a multiple of the
        static estimate for the pass that produced the reading -- and that
        estimate is mostly a flat 1 GB, so dividing it by a small pass's
        megapixels let 512x512 authorise 15 GB per megapixel. The session then
        carried that rate into every larger pass, and reserved 10.4 GB for a
        1280x960 one: more than the 13.9 GB model it was protecting."""
        torch.peak = 60 * GB
        mc_memory.observe_activation_peak(512, 512)

        learned = mc_memory.observed_peaks()[0]
        assert learned <= mc_memory.MAX_LEARNED_BYTES_PER_MEGAPIXEL
        assert mc_memory.vram_headroom_bytes(1280, 960) < 4 * GB

    def test_a_learned_reserve_never_annexes_the_card(self, warming, torch):
        """The learned term specifically. The a-priori formula is not clamped
        here -- a genuinely huge pass does need a lot of headroom, and trimming
        that to something the card can actually give is _attainable_headroom's
        job, not this one's."""
        torch.peak = 400 * GB
        mc_memory.observe_activation_peak(2048, 2048)

        total = mc_memory.total_vram_bytes()
        assert mc_memory._observed_headroom_bytes(4096, 4096) <= total * mc_memory.MAX_RESERVE_FRACTION

    def test_a_manual_reserve_is_still_not_clamped(self, warming, host):
        host.shared.opts.model_chain_vram_reserve_gb = 20.0
        assert mc_memory.vram_headroom_bytes(1024, 1024) == 20 * GB

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


# --------------------------------------------------------------------------- #
# What the host's sampler is about to ask for
# --------------------------------------------------------------------------- #


class FakeUnet(FakePatcher):
    """A UNet patcher carrying the fields ``sampling_prepare`` reads."""

    def __init__(self, size, *, activations, preserved=0, loaded=None, on_card=True):
        super().__init__("unet", size)
        # The torch module, shared by every clone of the patcher exactly as
        # ``ModelPatcher.clone`` shares it, and stamped by the host's ``load``
        # with the merge it carries.
        self.model = type("KModel", (), {})()
        self.model.current_weight_patches_uuid = "merge-0"
        self.model.model_loaded_weight_memory = size if loaded is None else loaded
        self.patches_uuid = "merge-0"
        self.patches = {}
        self.extra_preserved_memory_during_sampling = preserved
        self.controlnet_linked_list = None
        self.weight_wrapper_patches = {}
        self.load_device = "cuda"
        self.current_device = "cuda" if on_card else "cpu"
        self._activations = activations

    def memory_required(self, shape):
        self.asked_shape = list(shape)
        return self._activations

    def loaded_size(self):
        return self.model.model_loaded_weight_memory


def with_a_new_merge(unet, patches=228):
    """The clone the host's LoRA loader hands the sampler: same weights, new merge."""
    clone = FakeUnet(unet.size, activations=unet._activations,
                     preserved=unet.extra_preserved_memory_during_sampling)
    clone.model = unet.model
    clone.patches_uuid = "merge-1"
    clone.patches = {f"key-{i}": [] for i in range(patches)}
    return clone

    def has_online_lora(self):
        return False


def loaded(name, size):
    """Another model on the card, named the way the host names it in its log."""
    patcher = FakePatcher(name, size)
    patcher.model = type(name, (), {})()
    return FakeLoadedModel(patcher)


@pytest.fixture
def card(warming, monkeypatch):
    """The reporting user's card: 4.4 GB free beside an 18.3 GB resident set."""
    warming.mm.current_loaded_models.clear()
    unet = FakeUnet(12.6 * GB, activations=1.8 * GB)
    warming.mm.current_loaded_models.extend([
        FakeLoadedModel(unet),
        loaded("JointTextEncoder", 5.6 * GB),
        loaded("Qwen2DVAE", 0.1 * GB),
    ])
    monkeypatch.setattr(warming.mm, "get_free_memory", lambda dev=None: 4.4 * GB)
    monkeypatch.setattr(warming.mm, "minimum_inference_memory", lambda: 0.8 * GB, raising=False)
    monkeypatch.setattr(warming.mm, "extra_reserved_memory", lambda: 0, raising=False)
    monkeypatch.setattr(mc_memory, "_last_sampler_demand", None)
    return types.SimpleNamespace(unet=unet, mm=warming.mm)


class TestWhatTheSamplerAsksFor:
    """The host frees against its own request with nothing protected, and says
    nothing about why. The reading is taken where the host takes it."""

    def test_a_pass_that_fits_says_nothing(self, card, caplog):
        with caplog.at_level("INFO"):
            demand = mc_memory.note_sampler_demand(card.unet, (1, 16, 128, 128))

        assert demand.request == pytest.approx(1.8 * GB)
        assert demand.evicts == 0
        assert card.unet.asked_shape == [2, 16, 128, 128], "the host doubles the batch"
        assert not [m for m in caplog.messages if "about to ask for" in m]

    def test_a_reservation_that_forces_an_eviction_is_named(self, card, caplog):
        # The reporting user's log: 4.4 GB free, the same prompt every time, and
        # ``Moving model(s) has taken 4.55 seconds`` before every pass -- the
        # 5.6 GB text encoder leaving to satisfy a request this extension had
        # no part in. The only line that could say so is the one before it.
        card.unet.extra_preserved_memory_during_sampling = 8.6 * GB

        with caplog.at_level("INFO"):
            demand = mc_memory.note_sampler_demand(card.unet, (1, 16, 128, 128))

        assert demand.request == pytest.approx(10.4 * GB)
        assert demand.shortfall == pytest.approx(6.0 * GB)
        assert demand.evicts == pytest.approx(5.7 * GB)
        assert [name for name, _held in demand.leaving()] == ["Qwen2DVAE", "JointTextEncoder"]
        [line] = [m for m in caplog.messages if "about to ask for" in m]
        assert "10.4 GB of free VRAM" in line and "the card has 4.4 GB" in line
        assert "the 0.1 GB Qwen2DVAE and the 5.6 GB JointTextEncoder" in line
        assert "8.6 GB reserved during sampling on the model by another extension" in line
        assert "this extension makes no such reservation" in line
        assert "1.8 GB for this pass's activations (a 1x16x128x128 latent)" in line

    def test_a_clone_the_host_never_loaded_is_charged_its_whole_size(self, card, caplog):
        # ``Requested to load KModel`` followed by a move with no ``loaded``
        # line: every weight is on the card, but the patcher's device tag says
        # it was never loaded, so the host asks for room to load all of it.
        card.unet.current_device = "cpu"

        with caplog.at_level("INFO"):
            demand = mc_memory.note_sampler_demand(card.unet, (1, 16, 128, 128))

        assert demand.stale_tag
        assert demand.off_card == pytest.approx(12.6 * GB)
        assert demand.request == pytest.approx(12.6 * GB * 1.1 + 1.8 * GB)
        [line] = [m for m in caplog.messages if "about to ask for" in m]
        assert "counts as still to load although they are resident" in line
        assert "stale device tag" in line

    def test_forges_own_reserve_is_counted_as_the_hosts(self, card, monkeypatch, caplog):
        monkeypatch.setattr(card.mm, "extra_reserved_memory", lambda: 6 * GB, raising=False)

        with caplog.at_level("INFO"):
            demand = mc_memory.note_sampler_demand(card.unet, (1, 16, 128, 128))

        assert demand.request == pytest.approx(7.8 * GB)
        [line] = [m for m in caplog.messages if "about to ask for" in m]
        assert "6.0 GB reserved in Forge's own settings" in line

    def test_forges_minimum_is_a_floor_under_the_request(self, card, monkeypatch, caplog):
        # ``minimum_inference_memory`` is the host's floor: a small pass asks
        # for the whole of it, and it evicts exactly as a reservation would.
        monkeypatch.setattr(card.mm, "minimum_inference_memory", lambda: 6 * GB, raising=False)

        with caplog.at_level("INFO"):
            demand = mc_memory.note_sampler_demand(card.unet, (1, 16, 128, 128))

        assert demand.request == pytest.approx(6.0 * GB)
        assert demand.shortfall == pytest.approx(1.6 * GB)
        [line] = [m for m in caplog.messages if "about to ask for" in m]
        assert "Forge's minimum of 6.0 GB, which is what counts here" in line

    def test_the_warm_up_ties_its_move_to_the_eviction(self, card, caplog):
        card.unet.extra_preserved_memory_during_sampling = 8.6 * GB
        mc_memory.note_sampler_demand(card.unet, (1, 16, 128, 128))
        result = mc_memory.PreloadResult(state="ready", checkpoint="A", moved_bytes=int(5.6 * GB),
                                         resident_bytes=int(18.3 * GB), model_bytes=int(18.3 * GB),
                                         seconds=4.8, moved_seconds=4.8)

        with caplog.at_level("INFO"):
            mc_memory._log_preload_result(result)

        [line] = [m for m in caplog.messages if "left the card during the last sampling" in m]
        assert "5.6 GB just moved back" in line and "asked for 10.4 GB" in line

    def test_a_move_larger_than_the_eviction_is_not_the_samplers(self, card, caplog):
        # The sampler took 5.7 GB; a warm-up that moved a whole checkpoint back
        # is a checkpoint change, and the line would blame the wrong thing.
        card.unet.extra_preserved_memory_during_sampling = 8.6 * GB
        mc_memory.note_sampler_demand(card.unet, (1, 16, 128, 128))
        result = mc_memory.PreloadResult(state="ready", checkpoint="B", moved_bytes=int(18.3 * GB),
                                         resident_bytes=int(18.3 * GB), model_bytes=int(18.3 * GB),
                                         seconds=15.0, moved_seconds=15.0)

        with caplog.at_level("INFO"):
            mc_memory._log_preload_result(result)

        assert not [m for m in caplog.messages if "left the card during the last sampling" in m]

    def test_the_reading_is_spent_by_the_warm_up_that_uses_it(self, card, caplog):
        card.unet.extra_preserved_memory_during_sampling = 8.6 * GB
        mc_memory.note_sampler_demand(card.unet, (1, 16, 128, 128))
        result = mc_memory.PreloadResult(state="ready", checkpoint="A", moved_bytes=int(5.6 * GB),
                                         resident_bytes=int(18.3 * GB), model_bytes=int(18.3 * GB),
                                         seconds=4.8, moved_seconds=4.8)

        with caplog.at_level("INFO"):
            mc_memory._log_preload_result(result)
            mc_memory._log_preload_result(result)

        assert len([m for m in caplog.messages if "left the card during the last sampling" in m]) == 1

    def test_a_warm_up_after_a_pass_that_fit_ties_nothing(self, card, caplog):
        mc_memory.note_sampler_demand(card.unet, (1, 16, 128, 128))
        result = mc_memory.PreloadResult(state="ready", checkpoint="A", moved_bytes=int(0.6 * GB),
                                         resident_bytes=int(18.3 * GB), model_bytes=int(18.3 * GB),
                                         seconds=0.8, moved_seconds=0.8)

        with caplog.at_level("INFO"):
            mc_memory._log_preload_result(result)

        assert not [m for m in caplog.messages if "left the card during the last sampling" in m]

    def test_a_host_without_the_fields_is_left_alone(self, card, caplog):
        with caplog.at_level("INFO"):
            assert mc_memory.note_sampler_demand(FakePatcher("unet", 8 * GB), (1, 16, 128, 128)) is None
        assert mc_memory.note_sampler_demand(card.unet, None) is None
        assert not [m for m in caplog.messages if "about to ask for" in m]


class TestAChangeOfMergeIsAnEviction:
    """A LoRA added, removed or reweighted is a change of model: the weights the
    host is about to take off the card to re-merge leave it first, cleanly,
    and nothing else on the card is touched."""

    def test_a_changed_merge_takes_the_weights_off_before_the_sampler_does(self, card, caplog):
        # The reporting user's log: several generations with no LoRA, then one
        # with -- ``Requested to load KModel``, ``Moving model(s) has taken
        # 101.95 seconds``, a first step that never finished.
        clone = with_a_new_merge(card.unet)

        with caplog.at_level("INFO"):
            mc_memory.evict_for_rebake(clone)

        assert math.isinf(card.mm.freed[-1]), "all of the model, never part of it"
        [line] = [m for m in caplog.messages if "left the card before this pass" in m]
        assert "Stage 1's KModel left the card" in line and "12.6 GB" in line
        assert "A change of LoRA is a change of model" in line

    def test_everything_else_on_the_card_is_kept(self, card):
        clone = with_a_new_merge(card.unet)

        mc_memory.evict_for_rebake(clone)

        others = [e for e in card.mm.current_loaded_models if e.model is not card.unet]
        assert card.mm.kept[-1] == others and len(others) == 2

    def test_an_unchanged_merge_leaves_the_card_alone(self, card):
        assert mc_memory.evict_for_rebake(card.unet) == 0
        assert card.mm.freed == []

    def test_a_model_the_host_never_stamped_is_left_alone(self, card):
        # ``current_weight_patches_uuid`` is None until the host's own load
        # sets it, and None is the single-pass branch the host takes itself.
        clone = with_a_new_merge(card.unet)
        card.unet.model.current_weight_patches_uuid = None

        assert mc_memory.evict_for_rebake(clone) == 0
        assert card.mm.freed == []

    def test_weights_already_off_the_card_are_not_moved_again(self, card):
        clone = with_a_new_merge(card.unet)
        card.unet.model.model_loaded_weight_memory = 0

        assert mc_memory.evict_for_rebake(clone) == 0
        assert card.mm.freed == []

    def test_a_model_the_host_is_not_holding_is_left_alone(self, card):
        card.mm.current_loaded_models[:] = [
            e for e in card.mm.current_loaded_models if e.model is not card.unet]
        clone = with_a_new_merge(card.unet)

        assert mc_memory.evict_for_rebake(clone) == 0
        assert card.mm.freed == []

    def test_a_host_that_cannot_be_told_what_to_keep_is_left_to_itself(self, card, monkeypatch):
        # Without ``keep_loaded`` the only figure that means "all of this one"
        # would take the text encoder too. The host makes its own move then.
        monkeypatch.setattr(card.mm, "free_memory",
                            lambda required, device: card.mm.freed.append(required))

        assert mc_memory.evict_for_rebake(with_a_new_merge(card.unet)) == 0
        assert card.mm.freed == []

    def test_the_language_model_is_not_asked_for_anything(self, card, monkeypatch):
        import mc_broker

        asked = []
        monkeypatch.setattr(mc_memory, "_reclaim_foreign", lambda *a, **k: asked.append(a))
        monkeypatch.setattr(mc_broker, "request_vram", lambda *a, **k: asked.append(a))

        mc_memory.evict_for_rebake(with_a_new_merge(card.unet))

        assert asked == []

    def test_an_over_committed_card_is_said_on_the_way_out(self, card, caplog, monkeypatch):
        monkeypatch.setattr(mc_memory, "spilled_vram_bytes", lambda: int(1.5 * GB))

        with caplog.at_level("INFO"):
            mc_memory.evict_for_rebake(with_a_new_merge(card.unet))

        [line] = [m for m in caplog.messages if "left the card before this pass" in m]
        assert "over-committed: at least 1.5 GB" in line


class TestOverCommit:
    """What the task manager shows as "VRAM full and everything crawls",
    measured from inside the process."""

    @staticmethod
    def torch_with(reserved, free, total):
        return types.SimpleNamespace(cuda=types.SimpleNamespace(
            mem_get_info=lambda dev: (free, total), memory_reserved=lambda dev: reserved))

    def test_it_is_what_torch_holds_beyond_what_the_card_gave_out(self, card, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", self.torch_with(25 * GB, 1 * GB, 24 * GB))
        assert mc_memory.spilled_vram_bytes() == 2 * GB

    def test_a_card_holding_everything_reports_none(self, card, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", self.torch_with(20 * GB, 1 * GB, 24 * GB))
        assert mc_memory.spilled_vram_bytes() == 0

    def test_without_torch_the_reading_is_zero(self, card, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", None)
        assert mc_memory.spilled_vram_bytes() == 0

    def test_an_over_committed_card_is_said_even_when_the_pass_fits(self, card, caplog, monkeypatch):
        monkeypatch.setattr(mc_memory, "spilled_vram_bytes", lambda: int(2 * GB))

        with caplog.at_level("INFO"):
            demand = mc_memory.note_sampler_demand(card.unet, (1, 16, 128, 128))

        assert demand.evicts == 0
        [line] = [m for m in caplog.messages if "about to ask for" in m]
        assert "and that fits" in line and "over-committed: at least 2.0 GB" in line
