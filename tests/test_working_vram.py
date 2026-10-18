"""What a pass really allocates, measured from its first step and said beside the host's request.

The host asks for a fixed allowance of free VRAM beside the weights before a
pass -- 2.8 GB an image on the user's card -- and frees other models to get it.
Whether that is a good bet is a question only a measurement answers, so every
Stage 1 pass reads PyTorch's peak from its first sampling step to its end and
says it, with the request, and with the cost of each extra image when the same
size has been seen at another batch size.
"""

from __future__ import annotations

import sys
import types

import pytest

import mc_memory
import mc_working_vram
from test_orchestration import DEFAULTS, UI_ORDER, make_p

GB = 1024**3


@pytest.fixture
def torch(monkeypatch):
    """A fake ``torch.cuda`` whose allocation and peak the test controls."""
    fake = types.ModuleType("torch")
    fake.allocated = 13 * GB
    fake.peak = 13 * GB
    fake.resets = 0

    def reset(device=None):
        fake.resets += 1
        fake.peak = fake.allocated

    fake.cuda = types.SimpleNamespace(
        memory_allocated=lambda device=None: fake.allocated,
        max_memory_allocated=lambda device=None: fake.peak,
        reset_peak_memory_stats=reset,
    )
    monkeypatch.setitem(sys.modules, "torch", fake)
    mc_working_vram.forget()
    yield fake
    mc_working_vram.forget()


def demand(request=11.9 * GB, activations=11.2 * GB, latent="4x16x1x128x128", counted_once=False):
    return types.SimpleNamespace(request=request, activations=activations, latent=latent,
                                 counted_once=counted_once)


def measure(torch, batch, peak_gb, asked=None, width=1024, height=1024, stage="Stage 1"):
    """One whole pass: armed, first step, the peak, closed."""
    mc_working_vram.arm(asked, width, height, batch, stage, f"{batch}x16x1x128x128")
    mc_working_vram.first_step()
    torch.peak = torch.allocated + peak_gb * GB
    return mc_working_vram.close()


class TestTheWindow:
    def test_opens_at_the_first_step_and_closes_at_the_end(self, torch, host):
        mc_working_vram.arm(demand(), 1024, 1024, 4, "Stage 1", "4x16x1x128x128")
        assert torch.resets == 0, "nothing is reset before the host has made its room"

        assert mc_working_vram.first_step() is True
        assert torch.resets == 1
        torch.peak = torch.allocated + 3.2 * GB

        found = mc_working_vram.close()
        assert found.measured == pytest.approx(3.2 * GB)
        assert found.baseline == 13 * GB

    def test_a_step_without_an_armed_pass_resets_nothing(self, torch, host):
        assert mc_working_vram.first_step() is False
        assert torch.resets == 0

    def test_only_the_first_step_opens_it(self, torch, host):
        mc_working_vram.arm(demand(), 1024, 1024, 4, "Stage 1")
        mc_working_vram.first_step()
        mc_working_vram.first_step()
        assert torch.resets == 1

    def test_closing_nothing_is_nothing(self, torch, host):
        assert mc_working_vram.close() is None

    def test_a_pass_that_never_stepped_is_not_measured(self, torch, host, caplog):
        mc_working_vram.arm(demand(), 1024, 1024, 4, "Stage 1")
        with caplog.at_level("INFO"):
            assert mc_working_vram.close() is None
        assert not [m for m in caplog.messages if "working VRAM" in m]
        assert mc_working_vram.first_step() is False, "and it is disarmed"

    def test_arming_the_next_pass_closes_an_open_one(self, torch, host, caplog):
        """The hires pass follows the first without a post_sample between them."""
        mc_working_vram.arm(demand(), 1024, 1024, 1, "Stage 1")
        mc_working_vram.first_step()
        torch.peak = torch.allocated + 0.8 * GB
        with caplog.at_level("INFO"):
            mc_working_vram.arm(demand(), 2048, 2048, 1, "Stage 1 hires")
        [line] = [m for m in caplog.messages if "working VRAM" in m]
        assert "the Stage 1 pass used 0.8 GB" in line

    def test_a_host_without_torch_measures_nothing_quietly(self, host, monkeypatch):
        monkeypatch.delitem(sys.modules, "torch", raising=False)
        mc_working_vram.forget()
        mc_working_vram.arm(demand(), 1024, 1024, 4, "Stage 1")
        assert mc_working_vram.first_step() is False
        assert mc_working_vram.close() is None


class TestTheLine:
    def test_says_the_peak_the_request_and_the_ratio(self, torch, host, caplog):
        with caplog.at_level("INFO"):
            measure(torch, 4, 3.2, demand())

        [line] = [m for m in caplog.messages if "working VRAM" in m]
        assert ("the Stage 1 pass used 3.2 GB of working VRAM at its peak — a batch "
                "of 4 at 1024x1024 (0.80 GB per image)") in line
        assert ("Forge's sampler had asked for 11.9 GB free beside the weights, 11.2 GB of it "
                "for activations (2.80 GB per image) — 3.5x what the pass used") in line
        assert line.endswith("PyTorch's own allocations only, so a floor on what the driver had to find")

    def test_says_when_each_image_was_counted_once(self, torch, host, caplog):
        with caplog.at_level("INFO"):
            measure(torch, 4, 3.2, demand(request=6.3 * GB, activations=5.6 * GB, counted_once=True))
        [line] = [m for m in caplog.messages if "working VRAM" in m]
        assert "(1.40 GB per image, each counted once)" in line

    def test_without_a_request_says_the_measurement_alone(self, torch, host, caplog):
        with caplog.at_level("INFO"):
            measure(torch, 2, 1.5, None)
        [line] = [m for m in caplog.messages if "working VRAM" in m]
        assert "a batch of 2 at 1024x1024 (0.75 GB per image)" in line
        assert "asked for" not in line and "x what the pass used" not in line

    def test_the_increment_against_the_batch_below(self, torch, host, caplog):
        measure(torch, 1, 0.8, demand(request=3.5 * GB, activations=2.8 * GB, latent="1x16x1x128x128"))
        with caplog.at_level("INFO"):
            measure(torch, 4, 3.2, demand())

        [line] = [m for m in caplog.messages if "a batch of 4" in m]
        assert ("Against the batch of 1 at this size: +0.80 GB used per extra image against "
                "2.80 GB charged for each") in line

    def test_the_nearest_batch_below_is_the_one_compared_with(self, torch, host, caplog):
        measure(torch, 1, 0.8, demand())
        measure(torch, 2, 1.6, demand())
        with caplog.at_level("INFO"):
            measure(torch, 3, 2.4, demand())
        [line] = [m for m in caplog.messages if "a batch of 3" in m]
        assert "Against the batch of 2 at this size" in line

    def test_a_smaller_batch_is_compared_with_the_one_above_it(self, torch, host, caplog):
        measure(torch, 4, 3.2, demand())
        with caplog.at_level("INFO"):
            measure(torch, 2, 1.6, demand())
        [line] = [m for m in caplog.messages if "a batch of 2" in m]
        assert "Against the batch of 4 at this size: +0.80 GB used per extra image" in line

    def test_another_size_is_not_compared(self, torch, host, caplog):
        measure(torch, 1, 0.8, demand())
        with caplog.at_level("INFO"):
            measure(torch, 4, 3.2, demand(), width=768, height=768)
        [line] = [m for m in caplog.messages if "a batch of 4" in m]
        assert "Against the batch" not in line

    def test_the_same_batch_again_replaces_its_record(self, torch, host):
        measure(torch, 2, 1.6, demand())
        measure(torch, 2, 1.9, demand())
        assert mc_working_vram._seen[("Stage 1", 1024, 1024)] == {2: pytest.approx(1.9 * GB)}

    def test_an_unknown_size_is_named_by_its_latent(self, torch, host, caplog):
        with caplog.at_level("INFO"):
            measure(torch, 2, 1.6, demand(), width=0, height=0)
        [line] = [m for m in caplog.messages if "working VRAM" in m]
        assert "a batch of 2 at 2x16x1x128x128" in line

    def test_a_peak_below_the_baseline_reads_as_nothing(self, torch, host):
        mc_working_vram.arm(demand(), 1024, 1024, 1, "Stage 1")
        mc_working_vram.first_step()
        torch.peak = torch.allocated - 1 * GB
        assert mc_working_vram.close().measured == 0


class TestTheHooks:
    """The script arms the pass with what the host asked, opens it at the first
    step and closes it when the sampler returns."""

    def test_the_pre_sampling_hook_arms_the_pass(self, chain, host, torch, monkeypatch):
        asked = demand()
        monkeypatch.setattr(mc_memory, "note_sampler_demand", lambda unet, shape: asked)
        p = make_p(host, batch_size=4)
        args = [DEFAULTS[name] for name in UI_ORDER]

        chain.script.process_before_every_sampling(
            p, *args, noise=types.SimpleNamespace(shape=(4, 16, 1, 128, 128)))

        armed = mc_working_vram._armed
        assert (armed.stage, armed.width, armed.height, armed.batch) == ("Stage 1", 1024, 1024, 4)
        assert armed.latent == "4x16x1x128x128"
        assert (armed.asked, armed.asked_activations) == (pytest.approx(11.9 * GB), pytest.approx(11.2 * GB))

    def test_the_hires_pass_is_armed_at_its_own_size(self, chain, host, torch, monkeypatch):
        monkeypatch.setattr(mc_memory, "note_sampler_demand", lambda unet, shape: None)
        p = make_p(host, batch_size=1)
        p.is_hr_pass, p.hr_upscale_to_x, p.hr_upscale_to_y = True, 2048, 1536
        args = [DEFAULTS[name] for name in UI_ORDER]

        chain.script.process_before_every_sampling(
            p, *args, noise=types.SimpleNamespace(shape=(1, 16, 1, 192, 256)))

        armed = mc_working_vram._armed
        assert (armed.stage, armed.width, armed.height) == ("Stage 1 hires", 2048, 1536)
        assert armed.asked == 0

    def test_the_callback_opens_it_and_post_sample_closes_it(self, chain, host, torch, caplog,
                                                              monkeypatch):
        monkeypatch.setattr(mc_memory, "note_sampler_demand", lambda unet, shape: demand())
        p = make_p(host, batch_size=4)
        args = [DEFAULTS[name] for name in UI_ORDER]
        chain.script.process_before_every_sampling(
            p, *args, noise=types.SimpleNamespace(shape=(4, 16, 1, 128, 128)))

        chain.module._on_cfg_denoiser(types.SimpleNamespace(sampling_step=0))
        assert torch.resets == 1
        torch.peak = torch.allocated + 3.2 * GB
        with caplog.at_level("INFO"):
            chain.script.post_sample(p, types.SimpleNamespace(samples=None))

        [line] = [m for m in caplog.messages if "working VRAM" in m]
        assert "the Stage 1 pass used 3.2 GB" in line

    def test_stage_2_touches_nothing(self, chain, host, torch, monkeypatch):
        mc_working_vram.arm(demand(), 1024, 1024, 4, "Stage 1")
        mc_working_vram.first_step()
        chain.script._in_stage_2 = True
        try:
            chain.script.post_sample(make_p(host), types.SimpleNamespace(samples=None))
        finally:
            chain.script._in_stage_2 = False
        assert mc_working_vram._open is not None, "Stage 2's hooks are not Stage 1's pass"
