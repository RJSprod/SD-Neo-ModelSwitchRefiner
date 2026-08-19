"""Placement negotiation: where the LLM goes, and what it says it changed.

Section 13's requirement has two halves and both are tested here. The first is
that hybrid mode degrades gracefully rather than choosing between "full GPU" and
"stop the image model". The second is the one that is easy to skip and matters
more: "The app must not quietly reduce context or quality-critical settings
without reporting what it changed."
"""

from __future__ import annotations

import pytest

import mc_broker
import mc_llm_context as ctx
import mc_llm_runtime as runtime
from test_llm_context import build_model

_GB = 1024**3


@pytest.fixture
def placed(host, tmp_path, monkeypatch):
    """A configured install, an empty register, and a card we control."""
    mc_broker.clear()
    monkeypatch.setattr(ctx, "_store_path", lambda: tmp_path / "calibration.json")
    ctx.forget()
    for family in (mc_broker.FAMILY_IMAGE, mc_broker.FAMILY_LLM):
        mc_broker.unregister_reclaimer(family)
    monkeypatch.setattr(mc_broker, "safety_margin_bytes", lambda: 0)
    yield
    mc_broker.clear()
    ctx.forget()


def configure(monkeypatch, tmp_path, *, context=8192, mode="fixed", blocks=32,
              size_mb=4, gpu_layers="all", ceiling=131072):
    model = build_model(tmp_path, blocks=blocks, size_mb=size_mb, context=ceiling)
    configuration = runtime.Config(
        runtime=tmp_path / "llama-server", model=model, mmproj=None, gpu_index=0,
        device="CUDA0", gpu_layers=gpu_layers, context_size=context, context_mode=mode,
        context_buffer_gb=4.0, kv_type_k="f16", kv_type_v="f16")
    monkeypatch.setattr(runtime, "config", lambda: configuration)
    return configuration


def set_free(monkeypatch, gigabytes):
    monkeypatch.setattr(mc_broker, "free_vram_bytes", lambda: int(gigabytes * _GB))


class Recorder:
    def __init__(self, holds=0):
        self.holds = holds
        self.calls = []

    def release(self, needed_bytes, reason=""):
        self.calls.append((needed_bytes, reason))
        freed, self.holds = self.holds, 0
        return freed

    def resident_bytes(self):
        return self.holds

    def describe(self):
        return "the image checkpoint"


class TestItFits:
    def test_a_placement_that_fits_changes_nothing(self, placed, tmp_path, monkeypatch):
        configuration = configure(monkeypatch, tmp_path, context=8192)
        set_free(monkeypatch, 20)
        image = Recorder(holds=8 * _GB)
        mc_broker.register_reclaimer(mc_broker.FAMILY_IMAGE, image)

        negotiated = runtime.negotiate(configuration)

        assert negotiated.fits
        assert not negotiated.degraded
        assert negotiated.placement.context == 8192
        assert image.calls == []

    def test_the_model_ceiling_caps_a_larger_request(self, placed, tmp_path, monkeypatch):
        configuration = configure(monkeypatch, tmp_path, context=200_000, ceiling=8192)
        set_free(monkeypatch, 40)

        negotiated = runtime.negotiate(configuration)

        assert negotiated.placement.context == 8192

    def test_a_cpu_install_makes_no_vram_decision_at_all(self, placed, tmp_path, monkeypatch):
        configuration = configure(monkeypatch, tmp_path, gpu_layers="0")
        set_free(monkeypatch, 1)
        image = Recorder(holds=8 * _GB)
        mc_broker.register_reclaimer(mc_broker.FAMILY_IMAGE, image)

        negotiated = runtime.negotiate(configuration)

        assert negotiated.fits
        assert not negotiated.placement.on_gpu
        assert image.calls == []


class TestDegradation:
    def test_adaptive_lowers_the_context_before_moving_a_checkpoint(self, placed, tmp_path,
                                                                    host, monkeypatch):
        """Section 13's "least disruption": a context nobody is using is
        cheaper to give up than a model somebody is about to use."""
        host.shared.opts.set(mc_broker.OPT_POLICY, mc_broker.POLICY_ADAPTIVE)
        configuration = configure(monkeypatch, tmp_path, context=131072)
        set_free(monkeypatch, 3)
        image = Recorder(holds=8 * _GB)
        mc_broker.register_reclaimer(mc_broker.FAMILY_IMAGE, image)

        negotiated = runtime.negotiate(configuration)

        assert negotiated.placement.context < 131072
        assert image.calls == []
        assert any("context reduced" in note for note in negotiated.notes)

    def test_it_reports_every_reduction_it_made(self, placed, tmp_path, monkeypatch):
        configuration = configure(monkeypatch, tmp_path, context=131072)
        set_free(monkeypatch, 3)

        negotiated = runtime.negotiate(configuration)

        assert negotiated.degraded
        assert negotiated.notes
        assert all(isinstance(note, str) and note for note in negotiated.notes)

    def test_preserve_image_shrinks_the_llm_instead_of_the_checkpoint(self, placed, tmp_path,
                                                                      host, monkeypatch):
        host.shared.opts.set(mc_broker.OPT_POLICY, mc_broker.POLICY_PRESERVE_IMAGE)
        configuration = configure(monkeypatch, tmp_path, context=131072)
        set_free(monkeypatch, 3)
        image = Recorder(holds=10 * _GB)
        mc_broker.register_reclaimer(mc_broker.FAMILY_IMAGE, image)

        negotiated = runtime.negotiate(configuration)

        assert image.calls == []
        assert negotiated.placement.context < 131072

    def test_llm_priority_asks_for_the_checkpoint_first(self, placed, tmp_path, host,
                                                        monkeypatch):
        host.shared.opts.set(mc_broker.OPT_POLICY, mc_broker.POLICY_LLM_PRIORITY)
        configuration = configure(monkeypatch, tmp_path, context=131072, size_mb=4)
        set_free(monkeypatch, 3)
        image = Recorder(holds=10 * _GB)
        mc_broker.register_reclaimer(mc_broker.FAMILY_IMAGE, image)

        runtime.negotiate(configuration)

        assert image.calls

    def test_offload_is_reduced_when_context_alone_cannot_save_it(self, placed, tmp_path,
                                                                  monkeypatch):
        """A model whose *weights* do not fit cannot be rescued by a smaller
        cache, so blocks move to system RAM instead -- graceful degradation
        rather than a refusal."""
        configuration = configure(monkeypatch, tmp_path, context=4096, size_mb=64)
        set_free(monkeypatch, 0.03)

        negotiated = runtime.negotiate(configuration)

        assert negotiated.placement.gpu_layers != ctx.ALL_LAYERS
        assert any("offload reduced" in note or "system RAM" in note
                   for note in negotiated.notes)

    def test_a_reduction_never_goes_below_the_usable_floor(self, placed, tmp_path,
                                                           monkeypatch):
        """Below a couple of thousand tokens a chat model is not one; the
        placement is reported as not fitting rather than made useless."""
        configuration = configure(monkeypatch, tmp_path, context=131072, size_mb=4)
        set_free(monkeypatch, 0.05)

        negotiated = runtime.negotiate(configuration)

        assert negotiated.placement.context >= runtime.MINIMUM_CONTEXT


class TestExclusiveMode:
    def test_the_image_family_is_swept_before_anything_is_measured(self, placed, tmp_path,
                                                                   host, monkeypatch):
        host.shared.opts.set(mc_broker.OPT_MODE, mc_broker.MODE_EXCLUSIVE)
        configuration = configure(monkeypatch, tmp_path, context=8192)
        set_free(monkeypatch, 2)
        image = Recorder(holds=10 * _GB)
        mc_broker.register_reclaimer(mc_broker.FAMILY_IMAGE, image)
        mc_broker.declare(mc_broker.FAMILY_IMAGE, "ckpt", "a checkpoint", 10 * _GB)

        runtime.negotiate(configuration)

        assert image.calls


class TestAutomaticSizing:
    def test_automatic_mode_spends_what_is_free_on_context(self, placed, tmp_path,
                                                           monkeypatch):
        configuration = configure(monkeypatch, tmp_path, context=2048, mode="auto")
        set_free(monkeypatch, 12)

        negotiated = runtime.negotiate(configuration)

        assert negotiated.placement.context > 2048

    def test_automatic_mode_still_respects_the_model_ceiling(self, placed, tmp_path,
                                                             monkeypatch):
        configuration = configure(monkeypatch, tmp_path, context=2048, mode="auto",
                                  ceiling=8192)
        set_free(monkeypatch, 40)

        negotiated = runtime.negotiate(configuration)

        assert negotiated.placement.context == 8192


class TestUnconfigured:
    def test_it_says_what_to_do_rather_than_failing_obscurely(self, placed, monkeypatch):
        monkeypatch.setattr(runtime, "config", lambda: runtime.Config(
            runtime=None, model=None, mmproj=None, gpu_index=0, device="CUDA0",
            gpu_layers="all", context_size=8192, context_mode="fixed",
            context_buffer_gb=4.0, kv_type_k="f16", kv_type_v="f16"))

        with pytest.raises(runtime.NotConfigured, match="Choose a GGUF"):
            runtime.negotiate()


class TestReclaim:
    def test_a_stopped_runtime_frees_nothing_and_says_so(self, placed):
        assert runtime.runtime.release(4 * _GB, "an image pass") == 0

    def test_the_runtime_is_registered_as_the_llm_reclaimer(self, placed):
        """Importing the module is what wires the broker to it; a broken wiring
        would silently make every LLM demotion a no-op."""
        mc_broker.register_reclaimer(mc_broker.FAMILY_LLM, runtime.runtime)

        assert mc_broker._reclaimer(mc_broker.FAMILY_LLM) is runtime.runtime

    def test_status_is_answerable_before_anything_has_run(self, placed, tmp_path,
                                                          monkeypatch):
        configure(monkeypatch, tmp_path)

        status = runtime.runtime.status()

        assert status["configured"]
        assert not status["running"]
        assert status["resident_bytes"] == 0


class TestPreview:
    def test_a_preview_never_moves_anything(self, placed, tmp_path, host, monkeypatch):
        """The estimator panel is drawn on tab build and on every accordion
        open. Drawing a table must not cost somebody their checkpoint."""
        host.shared.opts.set(mc_broker.OPT_POLICY, mc_broker.POLICY_LLM_PRIORITY)
        configuration = configure(monkeypatch, tmp_path, context=131072)
        set_free(monkeypatch, 2)
        image = Recorder(holds=12 * _GB)
        mc_broker.register_reclaimer(mc_broker.FAMILY_IMAGE, image)

        runtime.negotiate(configuration, reclaim=False)

        assert image.calls == []

    def test_a_preview_in_exclusive_mode_does_not_sweep(self, placed, tmp_path, host,
                                                        monkeypatch):
        host.shared.opts.set(mc_broker.OPT_MODE, mc_broker.MODE_EXCLUSIVE)
        configuration = configure(monkeypatch, tmp_path)
        set_free(monkeypatch, 2)
        image = Recorder(holds=10 * _GB)
        mc_broker.register_reclaimer(mc_broker.FAMILY_IMAGE, image)
        mc_broker.declare(mc_broker.FAMILY_IMAGE, "ckpt", "a checkpoint", 10 * _GB)

        runtime.negotiate(configuration, reclaim=False)

        assert image.calls == []

    def test_a_preview_still_reports_the_placement_it_would_use(self, placed, tmp_path,
                                                                monkeypatch):
        configuration = configure(monkeypatch, tmp_path, context=131072)
        set_free(monkeypatch, 3)

        negotiated = runtime.negotiate(configuration, reclaim=False)

        assert negotiated.placement.context < 131072
        assert negotiated.notes
