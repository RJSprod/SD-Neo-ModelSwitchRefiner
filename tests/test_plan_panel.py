"""The txt2img memory section: what it says, and what it must never say.

Two rules do most of the work here. A phase that will not run is shown as
inactive and never as ``0.0 GB``, because a zero that looks like a measurement
invites the reader to conclude that Stage 2 is free rather than absent. And the
three LLM figures -- the calculated allowance, the observed residency, and the
configured cap -- are kept apart, because they are routinely three different
numbers and a user looking at one cannot tell which they are looking at.

The third rule has no assertion because it is structural: nothing in this module
negotiates a placement. A status display that decided one in order to show it
would restart llama-server every time somebody opened an accordion, on the same
tab as the Generate button.
"""

from __future__ import annotations

import types

import pytest

import mc_broker
import mc_plan
import mc_plan_panel as panel

GB = 1024**3


@pytest.fixture
def budget(host, monkeypatch):
    monkeypatch.setattr(mc_plan, "usable_vram_bytes", lambda: 24 * GB)
    monkeypatch.setattr(mc_broker, "safety_margin_bytes", lambda: 1 * GB)
    monkeypatch.setattr(mc_broker, "held_bytes", lambda family: 0)
    return monkeypatch


def chain(*, stage_2=True):
    phases = [
        mc_plan.Phase(mc_plan.CREATIVE_WRITER, mc_plan.KIND_PREPARATION, "Creative Writer"),
        mc_plan.Phase(mc_plan.STAGE_1, mc_plan.KIND_IMAGE, "Stage 1", 10 * GB,
                      detail="krea2"),
    ]
    if stage_2:
        phases += [
            mc_plan.Phase(mc_plan.HANDOFF, mc_plan.KIND_TRANSITION, "Handoff", 17 * GB,
                          detail="Stage 1 encoders kept"),
            mc_plan.Phase(mc_plan.STAGE_2, mc_plan.KIND_IMAGE, "Stage 2", 15 * GB,
                          detail="klein9b"),
        ]
    return mc_plan.publish(mc_plan.Plan(tuple(phases), 1024, 1024))


class TestThePlanHalf:
    def test_it_names_the_plan_in_order(self, budget):
        chain()

        assert "Creative Writer -> Stage 1 (krea2) -> Handoff" in panel.plan_view()

    def test_it_names_the_phase_that_sets_the_peak(self, budget):
        chain()
        view = panel.plan_view()

        assert "sets the protected peak" in view
        assert "| Handoff | 17.0 GB" in view

    def test_the_peak_is_the_largest_phase_and_says_so(self, budget):
        chain()
        view = panel.plan_view()

        assert "| Image working peak | 17.0 GB | the largest phase, not the sum" in view

    def test_a_preparation_phase_shows_no_image_residency(self, budget):
        """Its memory is llama-server's, which is the thing being budgeted
        around. A zero there would read as a measurement of nothing."""
        chain()

        assert "| Creative Writer | — | no image residency |" in panel.plan_view()

    def test_a_stage_2_that_will_not_run_is_absent_rather_than_zero(self, budget):
        chain(stage_2=False)
        view = panel.plan_view()

        assert "Stage 2" not in view
        assert "Not in this plan: Spatial Composer, Stage 2." in panel.report()

    def test_before_the_first_generation_it_says_so_plainly(self, budget):
        mc_plan.clear()

        assert "none yet" in panel.plan_view()

    def test_the_global_margin_is_shown_as_already_included(self, budget):
        """Because it is, and a reader adding it to the peak by hand would get
        a protected budget a gigabyte larger than the one in force."""
        chain()

        assert "already inside each phase peak above" in panel.plan_view()


@pytest.fixture
def server(monkeypatch):
    """A configured install with a server up, holding 6.2 GB.

    Faked at ``status()`` rather than by starting anything: this file is about
    what the section prints, and ``tests/test_llm_runtime.py`` is about what the
    runtime does.
    """
    import mc_llm_context as ctx
    import mc_llm_runtime

    report = mc_llm_runtime.Report(
        placement=ctx.Placement(gpu_layers=30, context=8192, kv_type_k="f16",
                                kv_type_v="f16", on_gpu=True),
        estimate=None, notes=(), observed_bytes=int(6.2 * GB),
        started_at=0.0, model="gemma.gguf", fits=True,
    )
    monkeypatch.setattr(mc_llm_runtime.runtime, "status", lambda: {
        "configured": True, "has_runtime": True, "has_model": True, "running": True,
        "model": "gemma.gguf", "quantization": "Q4_K_M", "device": "CUDA0",
        "mode": "gpu", "sees": True, "placement": report.placement,
        "report": report, "resident_bytes": int(6.2 * GB),
    })
    monkeypatch.setattr(mc_llm_runtime.runtime, "speed", lambda: (73.0, 31.0))
    return report


class TestTheLlmHalf:
    def test_off_says_the_arena_belongs_to_the_plan(self, budget, host):
        host.shared.opts.model_chain_llm_cap_mode = mc_plan.CAP_OFF
        chain()

        assert "Off" in panel.llm_view()

    def test_the_allowance_is_what_the_plan_leaves_over(self, budget, server):
        chain()

        assert "| Calculated allowance | 7.0 GB" in panel.llm_view()

    def test_the_observed_residency_is_not_the_allowance(self, budget, server):
        """Three different numbers, and a user looking at one cannot tell which
        they are looking at unless all three are named."""
        chain()
        view = panel.llm_view()

        assert "| Calculated allowance | 7.0 GB" in view
        assert "| Observed residency | 6.2 GB" in view

    def test_the_measured_speed_is_shown_both_ways_round(self, budget, server):
        chain()

        assert "73 / 31 tok/s" in panel.llm_view()

    def test_it_says_whether_the_placement_matches_the_active_plan(self, budget, server):
        chain()
        mc_plan.note_placement(mc_plan.current())

        assert "| Matches the active plan | yes |" in panel.llm_view()

    def test_a_placement_made_for_a_different_plan_says_so(self, budget, server):
        chain()
        mc_plan.note_placement(mc_plan.current())
        chain(stage_2=False)

        assert "not yet" in panel.llm_view()

    def test_a_custom_cap_is_shown_beside_the_allowance_not_instead_of_it(
            self, budget, server, host):
        host.shared.opts.model_chain_llm_cap_mode = mc_plan.CAP_CUSTOM
        host.shared.opts.model_chain_llm_cap_gb = 4.0
        chain()
        view = panel.llm_view()

        assert "Calculated allowance" in view and "Your cap" in view

    def test_a_learned_ceiling_says_it_will_not_be_promoted_back(self, budget, server):
        chain()
        mc_plan.record_miss("Stage 2", 1 * GB, llm_bytes=6 * GB, evicted=True)

        assert "not promoted back automatically" in panel.llm_view()

    def test_a_model_that_never_reached_the_card_is_not_shown_as_zero(
            self, budget, server, monkeypatch):
        import mc_llm_runtime

        status = mc_llm_runtime.runtime.status()
        status["report"] = mc_llm_runtime.Report()
        monkeypatch.setattr(mc_llm_runtime.runtime, "status", lambda: status)
        chain()

        assert panel.INACTIVE in panel.llm_view()

    def test_no_model_configured_says_where_to_choose_one(self, budget):
        assert "LLM Studio" in panel.llm_view()


class TestTheMissHalf:
    def test_silence_when_nothing_has_gone_wrong(self, budget):
        assert "No reserve miss" in panel.miss_view()

    def test_a_miss_names_the_phase_the_amount_and_the_eviction(self, budget):
        chain()
        mc_plan.record_miss("Stage 2", 1 * GB, llm_bytes=6 * GB, evicted=True)
        view = panel.miss_view()

        assert "Stage 2 exceeded" in view
        assert "1.0 GB" in view
        assert "emergency-evicted" in view

    def test_it_suggests_a_lower_custom_cap(self, budget):
        chain()
        mc_plan.record_miss("Stage 2", 1 * GB, llm_bytes=6 * GB, evicted=True)

        assert "Set **Custom** to about" in panel.miss_view()


class TestTheWholeSection:
    def test_it_never_raises_however_the_runtime_answers(self, budget, monkeypatch):
        """It is on the same tab as the Generate button. A status display that
        could throw would take the generation controls with it."""
        import mc_llm_runtime

        def explode(*args, **kwargs):
            raise RuntimeError("no runtime")

        monkeypatch.setattr(mc_llm_runtime.runtime, "status", explode)
        chain()

        assert panel.report()

    def test_it_never_raises_with_no_plan_and_no_llm(self, budget):
        mc_plan.clear()

        assert panel.report()

    def test_a_broken_budget_still_produces_a_sentence(self, budget, monkeypatch):
        def explode(*args, **kwargs):
            raise RuntimeError("no")

        monkeypatch.setattr(mc_plan, "budget", explode)

        assert "could not be built" in panel.report()


class TestItIsActuallyOnTheTab:
    """Built outside the Model Chain accordion, and built without raising.

    The section is wrapped in a ``try`` in ``ui()`` so that a panel that cannot
    be built costs a log line rather than the whole txt2img tab. That guard is
    also a way for the section to go missing silently, which is what this
    checks it has not.
    """

    def test_the_model_chain_ui_builds_it(self, host, monkeypatch):
        import mc_plan_panel
        import scripts.model_chain as model_chain

        built = []
        original = mc_plan_panel.build
        monkeypatch.setattr(mc_plan_panel, "build",
                            lambda elem_id: built.append(elem_id) or original(elem_id))
        monkeypatch.setattr(model_chain, "_model_choices",
                            lambda: (["None", "klein9b"], ["Inherit"]))

        model_chain.ScriptModelChain().ui(False)

        assert built, "the memory section was not built"

    def test_it_contributes_no_script_arguments(self, host, monkeypatch):
        """It is a readout, not a control. An argument added here would shift
        every positional index ``mc_plan`` reads out of ``p.script_args``."""
        import mc_plan
        import scripts.model_chain as model_chain

        monkeypatch.setattr(model_chain, "_model_choices",
                            lambda: (["None", "klein9b"], ["Inherit"]))
        components = model_chain.ScriptModelChain().ui(False)

        assert len(components) == 20
        assert max(mc_plan.STAGE_2_ARGUMENTS.values()) < len(components)
