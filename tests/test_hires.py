"""Interaction with the host's Hires. fix.

Hires fix runs *inside* Stage 1's sampling: the first pass is generated, then
upscaled and re-sampled, and only the upscaled result reaches postprocess. So
Stage 2 must refine that result exactly once, at its real size.

Two things make that easy to get wrong. ``p.width``/``p.height`` keep
describing the *first pass* when hires is on -- the upscaled size lives in
``hr_upscale_to_x``/``hr_upscale_to_y`` -- and a second call into postprocess
would re-refine images that have already been through Stage 2.
"""

from __future__ import annotations

import types

import pytest

import mc_arch
import mc_infotext

from test_orchestration import DEFAULTS, UI_ORDER, make_p, make_processed, run_chain  # noqa: F401


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #


class TestHiresTargetSize:
    def test_scale_factor_multiplies_both_axes(self):
        assert mc_arch.hires_target_size(1024, 1024, scale=2.0, step=64) == (2048, 2048)

    def test_scale_is_snapped_to_the_resolution_step(self):
        width, height = mc_arch.hires_target_size(1024, 1024, scale=1.5, step=64)
        assert width % 64 == 0 and height % 64 == 0

    def test_a_width_only_resize_drives_height_by_aspect_ratio(self):
        width, height = mc_arch.hires_target_size(1024, 512, resize_x=2048, resize_y=0, step=64)
        assert width == 2048
        assert height == pytest.approx(1024, abs=64)

    def test_a_height_only_resize_drives_width_by_aspect_ratio(self):
        width, height = mc_arch.hires_target_size(1024, 512, resize_x=0, resize_y=1024, step=64)
        assert height == 1024
        assert width == pytest.approx(2048, abs=64)

    def test_both_resizes_are_used_verbatim(self):
        assert mc_arch.hires_target_size(1024, 1024, resize_x=1536, resize_y=768, step=64) == (1536, 768)

    def test_an_explicit_resize_wins_over_the_scale(self):
        scaled = mc_arch.hires_target_size(1024, 1024, scale=4.0, step=64)
        resized = mc_arch.hires_target_size(1024, 1024, scale=4.0, resize_x=1536, resize_y=1536, step=64)
        assert scaled != resized
        assert resized == (1536, 1536)


class TestStage1Size:
    def make(self, **kwargs):
        base = dict(width=1024, height=1024, enable_hr=False)
        base.update(kwargs)
        return types.SimpleNamespace(**base)

    def test_without_hires_it_is_the_requested_size(self):
        assert mc_arch.stage1_size(self.make()) == (1024, 1024)

    def test_with_hires_it_is_the_upscaled_size(self):
        """p.width/p.height still describe the first pass here."""
        p = self.make(enable_hr=True, hr_upscale_to_x=2048, hr_upscale_to_y=2048)
        assert mc_arch.stage1_size(p) == (2048, 2048)

    def test_the_hosts_computed_target_is_preferred(self):
        """init() already accounts for the old-hires-fix compatibility option."""
        p = self.make(enable_hr=True, hr_scale=4.0, hr_upscale_to_x=1536, hr_upscale_to_y=1536)
        assert mc_arch.stage1_size(p) == (1536, 1536)

    def test_it_falls_back_to_computing_the_target(self):
        p = self.make(enable_hr=True, hr_scale=2.0, hr_resize_x=0, hr_resize_y=0)
        assert mc_arch.stage1_size(p) == (2048, 2048)

    def test_a_processing_object_without_hires_attributes(self):
        assert mc_arch.stage1_size(types.SimpleNamespace(width=832, height=1216)) == (832, 1216)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def make_hires_p(host, **kwargs):
    """A Stage 1 processing object with hires fix enabled.

    Mirrors the host: width/height stay at the first pass, and the upscaled
    size lives in hr_upscale_to_x/y.
    """
    p = make_p(host, **kwargs)
    p.enable_hr = True
    p.hr_scale = 2.0
    p.hr_resize_x = 0
    p.hr_resize_y = 0
    p.hr_upscale_to_x = p.width * 2
    p.hr_upscale_to_y = p.height * 2
    return p


class TestStageTwoRunsOnce:
    def test_one_refine_per_image_with_hires_enabled(self, chain, host, image_factory):
        p = make_hires_p(host, batch_size=1)
        # postprocess sees the upscaled images, not the first pass.
        processed = make_processed(host, p, image_factory)
        processed.images = [image_factory(2048, 2048)]

        run_chain(chain, host, p, processed)

        assert len(chain.refine_calls) == 1
        assert len(processed.images) == 1

    @pytest.mark.parametrize("batch_size, n_iter", [(1, 1), (2, 1), (2, 2)])
    def test_refine_count_matches_the_batch_not_the_pass_count(
        self, chain, host, image_factory, batch_size, n_iter
    ):
        """Hires adds a second *sampling* pass, not a second image."""
        p = make_hires_p(host, batch_size=batch_size, n_iter=n_iter)
        processed = make_processed(host, p, image_factory)

        run_chain(chain, host, p, processed)

        assert len(chain.refine_calls) == batch_size * n_iter

    def test_still_exactly_one_checkpoint_switch(self, chain, host, image_factory):
        p = make_hires_p(host, batch_size=2)
        processed = make_processed(host, p, image_factory)

        run_chain(chain, host, p, processed)

        assert len(chain.switches) == 1

    def test_a_second_postprocess_does_not_refine_again(self, chain, host, image_factory):
        """Guards against a wrapper extension retrying the generation."""
        p = make_hires_p(host, batch_size=1)
        processed = make_processed(host, p, image_factory)

        settings = {**DEFAULTS}
        args = [settings[name] for name in UI_ORDER]

        chain.script.before_process(p, *args)
        chain.script.process(p, *args)
        chain.script.postprocess(p, processed, *args)
        first = list(processed.images)

        # Second invocation over the same result set.
        chain.script.before_process(p, *args)
        chain.script.process(p, *args)
        chain.script.postprocess(p, processed, *args)

        assert len(chain.refine_calls) == 1
        assert processed.images == first

    def test_a_fresh_result_set_is_still_refined(self, chain, host, image_factory):
        """The guard is per result set, not a one-shot for the session."""
        for _ in range(2):
            p = make_hires_p(host, batch_size=1)
            processed = make_processed(host, p, image_factory)
            run_chain(chain, host, p, processed)

        assert len(chain.refine_calls) == 2


class TestStageTwoIgnoresHiresSettings:
    def test_stage_2_never_runs_its_own_hires_pass(self, chain, host, image_factory):
        p = make_hires_p(host, batch_size=1)
        processed = make_processed(host, p, image_factory)

        run_chain(chain, host, p, processed)

        assert getattr(chain.refine_calls[0], "enable_hr", False) is False

    def test_stage_2_uses_its_own_steps_not_the_hires_steps(self, chain, host, image_factory):
        p = make_hires_p(host, batch_size=1)
        p.hr_second_pass_steps = 40
        processed = make_processed(host, p, image_factory)

        run_chain(chain, host, p, processed, steps=4)

        assert chain.refine_calls[0].steps == 4

    def test_stage_2_uses_its_own_denoise_not_the_hires_denoise(self, chain, host, image_factory):
        p = make_hires_p(host, batch_size=1)
        p.denoising_strength = 0.4  # what hires fix used
        processed = make_processed(host, p, image_factory)

        run_chain(chain, host, p, processed, denoise=1.0)

        assert chain.refine_calls[0].denoising_strength == 1.0

    def test_stage_2_sizes_from_the_upscaled_image(self, chain, host, image_factory):
        """The multiplier applies to what Stage 2 receives, not the first pass."""
        p = make_hires_p(host, batch_size=1, width=1024, height=1024)
        processed = make_processed(host, p, image_factory)
        processed.images = [image_factory(2048, 2048)]

        run_chain(chain, host, p, processed, size_multiplier=1.0)

        call = chain.refine_calls[0]
        assert (call.width, call.height) == (2048, 2048)


class TestStageOneSizeInfotext:
    def test_records_the_upscaled_size_with_hires(self, chain, host, image_factory):
        """Stage 1's recorded size must name the image Stage 2 received."""
        p = make_hires_p(host, batch_size=1, width=1024, height=1024)
        processed = make_processed(host, p, image_factory)

        run_chain(chain, host, p, processed)

        assert p.extra_generation_params[mc_infotext.STAGE1_SIZE] == "2048x2048"

    def test_records_the_plain_size_without_hires(self, chain, host, image_factory):
        p = make_p(host, batch_size=1, width=1216, height=832)
        processed = make_processed(host, p, image_factory)

        run_chain(chain, host, p, processed)

        assert p.extra_generation_params[mc_infotext.STAGE1_SIZE] == "1216x832"


class TestProgressAccounting:
    """Stage 1 sizes the Total progress bar for its own passes only.

    With hires fix that total is resized again for the second pass, so Stage
    2's steps would overflow it and the bar would wrap -- reading as though the
    work were repeating.
    """

    def test_stage_2_steps_are_added_to_the_total(self, chain, host, image_factory):
        host.shared.total_tqdm._tqdm.total = 40  # 20 first pass + 20 hires
        p = make_hires_p(host, batch_size=1)
        processed = make_processed(host, p, image_factory)

        run_chain(chain, host, p, processed, steps=4)

        assert host.shared.total_tqdm._tqdm.total == 44

    def test_the_whole_batch_is_accounted_for(self, chain, host, image_factory):
        host.shared.total_tqdm._tqdm.total = 40
        p = make_hires_p(host, batch_size=4)
        processed = make_processed(host, p, image_factory)

        run_chain(chain, host, p, processed, steps=4)

        assert host.shared.total_tqdm._tqdm.total == 40 + 4 * 4

    def test_an_unstarted_bar_is_left_alone(self, chain, host, image_factory):
        host.shared.total_tqdm._tqdm.total = 0
        p = make_p(host, batch_size=1)
        processed = make_processed(host, p, image_factory)

        run_chain(chain, host, p, processed, steps=4)

        assert host.shared.total_tqdm._tqdm.total == 0


class TestVramHeadroom:
    """Activation headroom has to follow the output size.

    A flat allowance under-estimates a large pass badly, and on Windows the
    overflow is spilled to system RAM rather than raising -- sampling drops
    from sub-second to tens of seconds per step with nothing in the log to
    explain it.
    """

    def test_a_one_megapixel_pass_uses_the_baseline(self):
        import mc_memory

        assert mc_memory.vram_headroom_bytes(1024, 1024) == pytest.approx(
            mc_memory.VRAM_HEADROOM_BYTES, rel=0.05
        )

    def test_headroom_grows_with_the_pass_size(self):
        import mc_memory

        small = mc_memory.vram_headroom_bytes(1024, 1024)
        large = mc_memory.vram_headroom_bytes(2048, 2048)
        assert large > small * 2

    def test_an_unknown_size_falls_back_to_the_baseline(self):
        import mc_memory

        assert mc_memory.vram_headroom_bytes(0, 0) == mc_memory.VRAM_HEADROOM_BYTES


def memory_management_fake():
    from backend import memory_management

    return memory_management


class TestMakeVramRoom:
    @pytest.fixture(autouse=True)
    def sized(self, host, monkeypatch):
        import mc_memory

        monkeypatch.setattr(mc_memory, "file_size_bytes", lambda name, mods=None: 14 * 1024**3)
        memory_management_fake().freed.clear()

    def test_ample_vram_leaves_both_models_resident(self, host, monkeypatch):
        """VRAM-first: nothing is evicted when the pass fits alongside."""
        import mc_memory

        monkeypatch.setattr(mc_memory, "free_vram_bytes", lambda: 22 * 1024**3)

        assert mc_memory.make_vram_room("klein", None, 1024, 1024) == 0
        assert memory_management_fake().freed == []

    def test_tight_vram_evicts_to_make_the_pass_fit(self, host, monkeypatch):
        import mc_memory

        monkeypatch.setattr(mc_memory, "free_vram_bytes", lambda: 8 * 1024**3)

        mc_memory.make_vram_room("klein", None, 2048, 2048)

        assert memory_management_fake().freed, "nothing was evicted for a pass that does not fit"

    def test_the_request_accounts_for_the_pass_size(self, host, monkeypatch):
        """A hires-sized refine must ask for more than a 1 MP one."""
        import mc_memory

        monkeypatch.setattr(mc_memory, "free_vram_bytes", lambda: 1 * 1024**3)

        mc_memory.make_vram_room("klein", None, 1024, 1024)
        small = memory_management_fake().freed[-1]

        mc_memory.make_vram_room("klein", None, 2048, 2048)
        large = memory_management_fake().freed[-1]

        assert large > small

    def test_an_unqueryable_card_is_left_to_the_host(self, host, monkeypatch):
        import mc_memory

        monkeypatch.setattr(mc_memory, "free_vram_bytes", lambda: 0)

        assert mc_memory.make_vram_room("klein", None, 2048, 2048) == 0
        assert memory_management_fake().freed == []


class TestStageTwoMakesRoom:
    def test_room_is_made_once_before_the_loop(self, chain, host, image_factory, monkeypatch):
        import mc_memory

        calls = []
        monkeypatch.setattr(
            mc_memory, "make_vram_room",
            lambda name, mods=None, w=0, h=0: calls.append((name, w, h)) or 0,
        )

        p = make_hires_p(host, batch_size=4)
        processed = make_processed(host, p, image_factory)

        run_chain(chain, host, p, processed)

        assert len(calls) == 1, "VRAM should be sized once per switch, not per image"

    def test_the_size_matches_what_stage_2_will_run(self, chain, host, image_factory, monkeypatch):
        import mc_memory

        calls = []
        monkeypatch.setattr(
            mc_memory, "make_vram_room",
            lambda name, mods=None, w=0, h=0: calls.append((name, w, h)) or 0,
        )

        p = make_hires_p(host, batch_size=1)
        processed = make_processed(host, p, image_factory)
        processed.images = [image_factory(2048, 2048)]

        run_chain(chain, host, p, processed, size_multiplier=1.0)

        _, width, height = calls[0]
        call = chain.refine_calls[0]
        assert (width, height) == (call.width, call.height)
