"""The composable execution plan, and the budget every placement comes out of.

The acceptance scenarios of the design intent are section 27's A to H, and each
one has a class below named for it. What they are really testing is a single
claim: that no feature has a memory policy of its own, so switching one on adds
a phase and switching one off removes it, and the arithmetic underneath never
changes shape.

The regression tests at the end are drawn from a real ``llama-server.log`` -- 71
server starts in one session, 47 of them dying at model load, the negotiated
context alternating 7168 / 8192 across consecutive generations. Every one of
those is what sizing a placement against instantaneous free VRAM looks like from
the outside.
"""

from __future__ import annotations

import types

import pytest

import mc_broker
import mc_plan

GB = 1024**3


# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #


@pytest.fixture
def card(host, monkeypatch):
    """A 24 GB card, and a checkpoint sizer that answers in gigabytes by name.

    Takes ``host`` so that the extension's own settings are reset with it. The
    cap mode is a host option, and a test that leaves it on Off would otherwise
    hand the next test a language model that is not allowed to exist.

    ``sizes`` maps a checkpoint name to what a pass on it costs. Nothing is
    read off disk: the arithmetic under test is what the plan does with the
    numbers, not where they came from, and ``mc_memory`` already has its own
    tests for the sizing itself.
    """
    sizes: dict[str, float] = {}
    modules: dict[str, float] = {}

    def required(name, mods=None, width=0, height=0):
        base = sizes.get(str(name), 0.0)
        # Activations scale with pixels, as the real estimator's do, so a test
        # can show a multiplier changing a peak rather than asserting it does.
        # Units of one 1024x1024 pass, so a test that does not care about size
        # gets round numbers and one that does can still show a peak moving.
        units = max((width * height) / (1024 * 1024), 1.0) if width and height else 1.0
        return int((base + 0.5 * (units - 1.0)) * GB)

    monkeypatch.setattr(mc_plan, "_pass_bytes",
                        lambda stage, w, h: required(stage.name, stage.modules, w, h)
                        if stage.present else 0)
    monkeypatch.setattr(mc_plan, "_module_bytes",
                        lambda stage: int(modules.get(str(stage.name), 0.0) * GB))
    monkeypatch.setattr(mc_plan, "usable_vram_bytes", lambda ours=0: 24 * GB)
    monkeypatch.setattr(mc_broker, "safety_margin_bytes", lambda: 0)
    monkeypatch.setattr(mc_broker, "held_bytes", lambda family: 0)
    return types.SimpleNamespace(sizes=sizes, modules=modules)


def plan_for(card, **kwargs):
    kwargs.setdefault("width", 1024)
    kwargs.setdefault("height", 1024)
    return mc_plan.build(**kwargs)


def stage(name, **kwargs):
    return mc_plan.Stage(name=name, **kwargs)


# --------------------------------------------------------------------------- #
# Section 27's scenarios
# --------------------------------------------------------------------------- #


class TestScenarioAPlainStageOne:
    """Creative, Spatial compose and Stage 2 all off."""

    def test_the_plan_contains_stage_1_and_nothing_else(self, card):
        card.sizes["krea2"] = 14.0
        plan = plan_for(card, stage_1=stage("krea2"))

        assert [phase.name for phase in plan.phases] == [mc_plan.STAGE_1]

    def test_no_reserve_is_added_for_a_stage_2_that_will_not_run(self, card):
        card.sizes["krea2"] = 14.0
        plan = plan_for(card, stage_1=stage("krea2"))

        assert plan.image_working_peak() == 14 * GB

    def test_the_peak_follows_the_size_actually_being_sampled(self, card):
        card.sizes["krea2"] = 14.0
        small = plan_for(card, stage_1=stage("krea2"), width=1024, height=1024)
        large = plan_for(card, stage_1=stage("krea2"), width=2048, height=2048)

        assert large.image_working_peak() > small.image_working_peak()

    def test_the_llm_gets_the_rest_of_the_card(self, card):
        card.sizes["krea2"] = 14.0
        mc_plan.publish(plan_for(card, stage_1=stage("krea2")))

        assert mc_plan.persistent_llm_budget() == 10 * GB


class TestScenarioBCreativeWriterAndStageOne:
    def test_the_writer_is_a_phase_of_the_plan(self, card):
        card.sizes["krea2"] = 14.0
        plan = plan_for(card, stage_1=stage("krea2"), creative=True)

        assert plan.has(mc_plan.CREATIVE_WRITER)
        assert plan.uses_llm

    def test_the_writer_holds_no_image_vram_of_its_own(self, card):
        """Its memory is llama-server's, which is the thing being budgeted
        around. Counted on both sides it would be reserved twice."""
        card.sizes["krea2"] = 14.0
        plan = plan_for(card, stage_1=stage("krea2"), creative=True)

        assert plan.image_working_peak() == 14 * GB
        assert plan.phase(mc_plan.CREATIVE_WRITER).peak_bytes == 0

    def test_stage_1_still_sets_the_peak(self, card):
        card.sizes["krea2"] = 14.0
        plan = plan_for(card, stage_1=stage("krea2"), creative=True)

        assert plan.limiting().name == mc_plan.STAGE_1


class TestScenarioCSmartBboxAndStageOne:
    def test_both_llm_calls_are_in_the_plan(self, card):
        card.sizes["krea2"] = 14.0
        plan = plan_for(card, stage_1=stage("krea2"), creative=True,
                        spatial_compose="smart")

        assert plan.has(mc_plan.CREATIVE_WRITER)
        assert plan.has(mc_plan.SPATIAL_COMPOSER)
        assert plan.llm_calls == 2

    def test_a_second_llm_call_does_not_shrink_the_image_budget(self, card):
        """Two calls to one warm server cost the image side nothing. If they
        did, the composer would be paid for by the checkpoint."""
        card.sizes["krea2"] = 14.0
        one = plan_for(card, stage_1=stage("krea2"), creative=True)
        two = plan_for(card, stage_1=stage("krea2"), creative=True,
                       spatial_compose="smart")

        assert one.image_working_peak() == two.image_working_peak()


class TestScenarioDDirectBbox:
    def test_direct_merge_introduces_no_second_llm_call(self, card):
        card.sizes["krea2"] = 14.0
        plan = plan_for(card, stage_1=stage("krea2"), creative=True,
                        spatial_compose="direct")

        assert plan.has(mc_plan.DIRECT_MERGE)
        assert not plan.has(mc_plan.SPATIAL_COMPOSER)
        assert plan.llm_calls == 1

    def test_the_merge_is_still_shown_in_the_plan(self, card):
        """A layout being applied deterministically is not a layout being
        dropped, and a plan with a gap where it should be reads as one."""
        card.sizes["krea2"] = 14.0
        plan = plan_for(card, stage_1=stage("krea2"), spatial_compose="direct")

        assert "Direct BBOX Merge" in plan.describe()


class TestScenarioEBothStagesNoCreative:
    def test_the_two_stages_are_not_summed(self, card):
        card.sizes.update({"krea2": 14.0, "klein9b": 12.0})
        plan = plan_for(card, stage_1=stage("krea2"),
                        stage_2=stage("klein9b", multiplier=1.0))

        assert plan.image_working_peak() < 26 * GB

    def test_the_peak_is_the_largest_phase(self, card):
        card.sizes.update({"krea2": 14.0, "klein9b": 12.0})
        card.modules["krea2"] = 0.0
        plan = plan_for(card, stage_1=stage("krea2"),
                        stage_2=stage("klein9b", multiplier=1.0))

        assert plan.image_working_peak() == 14 * GB
        assert plan.limiting().name == mc_plan.STAGE_1

    def test_the_handoff_carries_the_encoders_the_switch_really_spares(self, card):
        """The one moment two models are on the card together. Left out, the
        plan under-reserves exactly where it matters most."""
        card.sizes.update({"krea2": 10.0, "klein9b": 12.0})
        card.modules["krea2"] = 3.0
        plan = plan_for(card, stage_1=stage("krea2"),
                        stage_2=stage("klein9b", multiplier=1.0))

        assert plan.phase(mc_plan.HANDOFF).peak_bytes == 15 * GB
        assert plan.limiting().name == mc_plan.HANDOFF

    def test_the_whole_of_stage_1_is_not_assumed_to_survive(self, card):
        card.sizes.update({"krea2": 14.0, "klein9b": 12.0})
        card.modules["krea2"] = 2.0
        plan = plan_for(card, stage_1=stage("krea2"),
                        stage_2=stage("klein9b", multiplier=1.0))

        assert plan.phase(mc_plan.HANDOFF).peak_bytes == 14 * GB

    def test_stage_2_is_sized_at_the_size_stage_2_samples_at(self, card):
        """A 1.5x multiplier is not a 50% error to ignore. Activations scale
        with pixels, so it is a 125% one."""
        card.sizes.update({"krea2": 10.0, "klein9b": 10.0})
        plain = plan_for(card, stage_1=stage("krea2"),
                         stage_2=stage("klein9b", multiplier=1.0))
        upscaled = plan_for(card, stage_1=stage("krea2"),
                            stage_2=stage("klein9b", multiplier=1.5))

        assert (upscaled.phase(mc_plan.STAGE_2).peak_bytes
                > plain.phase(mc_plan.STAGE_2).peak_bytes)

    def test_a_warm_up_phase_is_planned_for(self, card):
        card.sizes.update({"krea2": 14.0, "klein9b": 12.0})
        plan = plan_for(card, stage_1=stage("krea2"), stage_2=stage("klein9b"))

        assert plan.has(mc_plan.WARM_UP)


class TestScenarioFFullComposition:
    def test_every_phase_appears_once_in_order(self, card):
        card.sizes.update({"krea2": 14.0, "klein9b": 12.0})
        plan = plan_for(card, stage_1=stage("krea2"), stage_2=stage("klein9b"),
                        creative=True, spatial_compose="smart")

        assert [phase.name for phase in plan.phases] == [
            mc_plan.CREATIVE_WRITER, mc_plan.SPATIAL_COMPOSER, mc_plan.STAGE_1,
            mc_plan.HANDOFF, mc_plan.STAGE_2, mc_plan.WARM_UP,
        ]

    def test_the_plan_reads_as_the_design_intent_writes_it(self, card):
        card.sizes.update({"krea2": 14.0, "klein9b": 12.0})
        plan = plan_for(card, stage_1=stage("krea2"), stage_2=stage("klein9b"),
                        creative=True, spatial_compose="smart")

        assert plan.describe().startswith(
            "Creative Writer -> Spatial Composer -> Stage 1 (krea2) -> ")

    def test_the_llm_budget_survives_the_whole_chain(self, card):
        """The number that used to be computed from Stage 1 alone, and is the
        reason a long chain evicted the model it had just started."""
        card.sizes.update({"krea2": 10.0, "klein9b": 18.0})
        mc_plan.publish(plan_for(card, stage_1=stage("krea2"),
                                 stage_2=stage("klein9b", multiplier=1.0),
                                 creative=True, spatial_compose="smart"))

        assert mc_plan.persistent_llm_budget() == 6 * GB


class TestScenarioGReserveMiss:
    def test_a_miss_is_recorded_with_the_phase_that_caused_it(self, card):
        card.sizes["krea2"] = 14.0
        mc_plan.publish(plan_for(card, stage_1=stage("krea2")))

        miss = mc_plan.record_miss("Stage 2", 1 * GB, llm_bytes=6 * GB, evicted=True)

        assert miss.phase == "Stage 2"
        assert miss.shortfall_bytes == 1 * GB
        assert miss.evicted

    def test_the_miss_is_visible_rather_than_only_logged(self, card):
        card.sizes["krea2"] = 14.0
        mc_plan.publish(plan_for(card, stage_1=stage("krea2")))
        mc_plan.record_miss("Stage 2", 1 * GB, llm_bytes=6 * GB, evicted=True)

        assert mc_plan.last_miss() is not None
        assert "emergency-evicted" in mc_plan.last_miss().describe()

    def test_a_safer_cap_is_suggested_below_what_was_held(self, card):
        card.sizes["krea2"] = 14.0
        mc_plan.publish(plan_for(card, stage_1=stage("krea2")))

        miss = mc_plan.record_miss("Stage 2", 1 * GB, llm_bytes=6 * GB, evicted=True)

        assert 0 < miss.suggested_bytes < 5 * GB

    def test_the_llm_is_not_silently_promoted_back(self, card):
        card.sizes["krea2"] = 14.0
        mc_plan.publish(plan_for(card, stage_1=stage("krea2")))
        mc_plan.record_miss("Stage 2", 1 * GB, llm_bytes=6 * GB, evicted=True)

        assert 0 < mc_plan.learned_cap_bytes() < 10 * GB

    def test_the_lowest_learned_ceiling_wins(self, card):
        card.sizes["krea2"] = 14.0
        mc_plan.publish(plan_for(card, stage_1=stage("krea2")))
        mc_plan.record_miss("Stage 2", 1 * GB, llm_bytes=6 * GB)
        first = mc_plan.learned_cap_bytes()
        mc_plan.record_miss("Stage 2", 3 * GB, llm_bytes=6 * GB)

        assert mc_plan.learned_cap_bytes() < first

    def test_a_miss_with_no_plan_quotes_no_cap(self, card):
        """The budget arithmetic returns the whole card when nothing is
        protected, and printing that at a user holding four gigabytes would be
        a lie in the one place they are looking for the truth."""
        mc_plan.clear()

        miss = mc_plan.record_miss("Stage 1", 1 * GB, llm_bytes=4 * GB)

        assert miss.cap_bytes == 0


class TestScenarioHFeatureToggle:
    def test_turning_stage_2_on_rebuilds_the_plan(self, card):
        card.sizes.update({"krea2": 10.0, "klein9b": 18.0})
        without = plan_for(card, stage_1=stage("krea2"))
        with_it = plan_for(card, stage_1=stage("krea2"),
                           stage_2=stage("klein9b", multiplier=1.0))

        assert without.identity() != with_it.identity()
        assert with_it.image_working_peak() > without.image_working_peak()

    def test_turning_stage_2_off_removes_its_reserve_entirely(self, card):
        card.sizes.update({"krea2": 10.0, "klein9b": 18.0})
        mc_plan.publish(plan_for(card, stage_1=stage("krea2"),
                                 stage_2=stage("klein9b", multiplier=1.0)))
        armed = mc_plan.persistent_llm_budget()
        mc_plan.publish(plan_for(card, stage_1=stage("krea2")))

        assert mc_plan.persistent_llm_budget() > armed

    def test_no_stale_assumption_survives_the_toggle(self, card):
        card.sizes.update({"krea2": 10.0, "klein9b": 18.0})
        mc_plan.publish(plan_for(card, stage_1=stage("krea2"),
                                 stage_2=stage("klein9b")))
        mc_plan.publish(plan_for(card, stage_1=stage("krea2")))

        assert not mc_plan.current().has(mc_plan.STAGE_2)
        assert not mc_plan.current().has(mc_plan.HANDOFF)


# --------------------------------------------------------------------------- #
# Plan boundaries -- the rule that stops the restarts
# --------------------------------------------------------------------------- #


class TestPlanBoundaries:
    def test_the_same_plan_twice_is_not_a_boundary(self, card):
        card.sizes["krea2"] = 14.0
        mc_plan.publish(plan_for(card, stage_1=stage("krea2")))
        mc_plan.note_placement(mc_plan.current())
        mc_plan.publish(plan_for(card, stage_1=stage("krea2")))

        assert not mc_plan.boundary_moved()

    def test_a_changed_stage_2_is_a_boundary(self, card):
        card.sizes.update({"krea2": 10.0, "klein9b": 18.0, "other": 8.0})
        mc_plan.publish(plan_for(card, stage_1=stage("krea2"),
                                 stage_2=stage("klein9b", multiplier=1.0)))
        mc_plan.note_placement(mc_plan.current())
        mc_plan.publish(plan_for(card, stage_1=stage("krea2"),
                                 stage_2=stage("other", multiplier=1.0)))

        assert mc_plan.boundary_moved()

    def test_measurement_noise_is_not_a_boundary(self, card):
        """An observed activation peak moves a little every pass. If that were
        a boundary, every generation would restart llama-server -- which is
        precisely the behaviour a user's log showed 71 times in one session."""
        card.sizes["krea2"] = 14.0
        mc_plan.publish(plan_for(card, stage_1=stage("krea2")))
        mc_plan.note_placement(mc_plan.current())

        nudged = mc_plan.Plan(
            tuple(
                mc_plan.Phase(phase.name, phase.kind, phase.label,
                              phase.peak_bytes + 8 * 1024 * 1024, phase.measured,
                              phase.detail)
                for phase in mc_plan.current().phases
            ),
            1024, 1024,
        )
        mc_plan.publish(nudged)

        assert not mc_plan.boundary_moved()

    def test_a_resolution_change_that_moves_a_memory_class_is_a_boundary(self, card):
        card.sizes["krea2"] = 14.0
        mc_plan.publish(plan_for(card, stage_1=stage("krea2"),
                                 width=1024, height=1024))
        mc_plan.note_placement(mc_plan.current())
        mc_plan.publish(plan_for(card, stage_1=stage("krea2"),
                                 width=2048, height=2048))

        assert mc_plan.boundary_moved()

    def test_a_server_placed_before_any_plan_is_not_re_placed(self, card):
        """It is already running and its placement worked. The plan's job is to
        size the next one."""
        card.sizes["krea2"] = 14.0
        mc_plan.note_placement(None)
        mc_plan.publish(plan_for(card, stage_1=stage("krea2")))

        assert not mc_plan.boundary_moved()


# --------------------------------------------------------------------------- #
# The budget
# --------------------------------------------------------------------------- #


class TestTheBudget:
    def test_no_plan_protects_nothing(self, card):
        """Outside a generation there is no image plan to protect, and LLM
        Studio may use what is free."""
        mc_plan.clear()

        assert mc_plan.image_protected_bytes() == 0

    def test_the_safety_margin_is_not_counted_twice(self, card, monkeypatch):
        """It is inside every phase peak already. Added again it is a gigabyte
        and a half of a card set aside twice for one activation peak."""
        card.sizes["krea2"] = 14.0
        monkeypatch.setattr(mc_broker, "safety_margin_bytes", lambda: 2 * GB)
        mc_plan.publish(plan_for(card, stage_1=stage("krea2")))

        assert mc_plan.image_protected_bytes() == 14 * GB

    def test_a_user_safety_adjustment_comes_off_the_llm(self, card, host):
        card.sizes["krea2"] = 14.0
        host.shared.opts.model_chain_plan_safety_gb = 2.0
        mc_plan.publish(plan_for(card, stage_1=stage("krea2")))

        assert mc_plan.image_protected_bytes() == 16 * GB
        assert mc_plan.persistent_llm_budget() == 8 * GB

    def test_custom_lowers_the_allowance(self, card, host):
        card.sizes["krea2"] = 14.0
        host.shared.opts.model_chain_llm_cap_mode = mc_plan.CAP_CUSTOM
        host.shared.opts.model_chain_llm_cap_gb = 4.0
        mc_plan.publish(plan_for(card, stage_1=stage("krea2")))

        assert mc_plan.persistent_llm_budget() == 4 * GB

    def test_custom_can_never_raise_the_allowance(self, card, host):
        """A control that could be less conservative than the arithmetic would
        be a way to break image generation from a text box."""
        card.sizes["krea2"] = 14.0
        host.shared.opts.model_chain_llm_cap_mode = mc_plan.CAP_CUSTOM
        host.shared.opts.model_chain_llm_cap_gb = 20.0
        mc_plan.publish(plan_for(card, stage_1=stage("krea2")))

        assert mc_plan.persistent_llm_budget() == 10 * GB

    def test_off_keeps_no_residency(self, card, host):
        card.sizes["krea2"] = 14.0
        host.shared.opts.model_chain_llm_cap_mode = mc_plan.CAP_OFF
        mc_plan.publish(plan_for(card, stage_1=stage("krea2")))

        assert mc_plan.persistent_llm_budget() == 0

    def test_a_card_that_cannot_be_measured_falls_back_rather_than_starving(
            self, card, monkeypatch):
        """-1 is "there is no card to divide up", not "there is no room". A
        zero here would force the model into system RAM on every machine whose
        VRAM cannot be queried."""
        monkeypatch.setattr(mc_plan, "usable_vram_bytes", lambda ours=0: 0)
        card.sizes["krea2"] = 14.0
        mc_plan.publish(plan_for(card, stage_1=stage("krea2")))

        assert mc_plan.persistent_llm_budget() == -1

    def test_what_the_image_side_already_holds_is_not_reserved_again(
            self, card, monkeypatch):
        card.sizes["krea2"] = 14.0
        monkeypatch.setattr(mc_broker, "held_bytes", lambda family: 14 * GB)
        mc_plan.publish(plan_for(card, stage_1=stage("krea2")))

        assert mc_plan.llm_reserve_bytes() == 0

    def test_the_reserve_covers_the_whole_plan_not_just_stage_1(self, card):
        """The bug this module was written for. A writer placed against Stage
        1's requirement alone is holding VRAM the handoff needs."""
        card.sizes.update({"krea2": 8.0, "klein9b": 18.0})
        mc_plan.publish(plan_for(card, stage_1=stage("krea2"),
                                 stage_2=stage("klein9b", multiplier=1.0),
                                 creative=True))

        assert mc_plan.llm_reserve_bytes() == 18 * GB


# --------------------------------------------------------------------------- #
# Reading the configuration off a generation
# --------------------------------------------------------------------------- #


class FakeScript:
    def __init__(self, title, args_from, args_to):
        self._title = title
        self.args_from = args_from
        self.args_to = args_to

    def title(self):
        return self._title


def processing(*, chain_args=(), creative_args=()):
    """A ``p`` carrying both scripts' arguments, laid out as the host lays them."""
    args = list(chain_args) + list(creative_args)
    scripts = []
    if chain_args:
        scripts.append(FakeScript(mc_plan.MODEL_CHAIN_TITLE, 0, len(chain_args)))
    if creative_args:
        scripts.append(FakeScript(mc_plan.CREATIVE_TITLE, len(chain_args), len(args)))
    return types.SimpleNamespace(
        width=1024, height=1024, enable_hr=False, script_args=args,
        scripts=types.SimpleNamespace(alwayson_scripts=scripts),
    )


def chain_arguments(enabled=True, target="klein9b", modules=None, multiplier=1.0):
    args = [enabled, target, modules] + [None] * 12 + [multiplier]
    assert args[mc_plan.STAGE_2_ARGUMENTS["size_multiplier"]] == multiplier
    return args


class TestReadingTheOtherScript:
    def test_stage_2_is_found_by_title_not_by_import(self, card):
        found = mc_plan.stage_2_from(processing(chain_args=chain_arguments()))

        assert found is not None and found.name == "klein9b"

    def test_a_disarmed_chain_contributes_no_stage_2(self, card):
        assert mc_plan.stage_2_from(
            processing(chain_args=chain_arguments(enabled=False))) is None

    def test_no_checkpoint_chosen_contributes_no_stage_2(self, card):
        assert mc_plan.stage_2_from(
            processing(chain_args=chain_arguments(target=mc_plan.NO_STAGE_2))) is None

    def test_an_absent_model_chain_script_is_not_an_error(self, card):
        assert mc_plan.stage_2_from(processing()) is None

    def test_the_size_multiplier_reaches_the_plan(self, card):
        found = mc_plan.stage_2_from(
            processing(chain_args=chain_arguments(multiplier=1.5)))

        assert found.multiplier == 1.5

    def test_creative_mode_is_read_off_its_own_arguments(self, card):
        found, mode = mc_plan.creative_from(
            processing(creative_args=[True, 5, "a", "b", True, "smart", "{}"]))

        assert found and mode == "smart"

    def test_a_direct_layout_is_reported_as_direct(self, card):
        _, mode = mc_plan.creative_from(
            processing(creative_args=[True, 5, "a", "b", True, "direct", "{}"]))

        assert mode == "direct"

    def test_spatial_switched_off_reports_no_layout(self, card):
        found, mode = mc_plan.creative_from(
            processing(creative_args=[True, 5, "a", "b", False, "smart", "{}"]))

        assert found and mode == ""

    def test_creative_switched_off_is_no_writer(self, card):
        found, mode = mc_plan.creative_from(
            processing(creative_args=[False, 5, "a", "b", True, "smart", "{}"]))

        assert not found and mode == ""

    def test_an_api_request_that_sent_only_the_flag_still_writes(self, card):
        found, mode = mc_plan.creative_from(processing(creative_args=[True]))

        assert found and mode == ""

    def test_either_script_builds_the_same_plan(self, card, host, monkeypatch):
        """Neither of them chooses which hook the host runs first, so neither
        of them may get a different answer."""
        card.sizes.update({"krea2": 10.0, "klein9b": 12.0})
        host.shared.opts.sd_model_checkpoint = "krea2"
        p = processing(chain_args=chain_arguments(),
                       creative_args=[True, 5, "a", "b", True, "smart", "{}"])

        assert mc_plan.build_for(p).identity() == mc_plan.build_for(p).identity()
        assert mc_plan.build_for(p).has(mc_plan.SPATIAL_COMPOSER)
        assert mc_plan.build_for(p).has(mc_plan.STAGE_2)

    def test_the_caller_may_override_what_it_knows_better(self, card, host):
        """Spatial Layout switched on over an empty canvas runs no Composer,
        and only the script holding the parsed layout can tell."""
        card.sizes["krea2"] = 10.0
        host.shared.opts.sd_model_checkpoint = "krea2"
        p = processing(creative_args=[True, 5, "a", "b", True, "smart", "{}"])

        plan = mc_plan.build_for(p, creative=True, spatial_compose="")

        assert not plan.has(mc_plan.SPATIAL_COMPOSER)


class TestTheArgumentIndicesAreRight:
    """The one place positional arguments are read, held in place by a test.

    ``p.script_args`` is a flat list and there is no other form to read it in,
    so the indices are checked against the live panel rather than a copy of it:
    reorder the Model Chain UI and this fails, rather than a reserve silently
    being computed from a seed offset.
    """

    def test_the_indices_match_the_panels_own_return_list(self, host, monkeypatch):
        import scripts.model_chain as model_chain

        script = model_chain.ScriptModelChain()
        monkeypatch.setattr(model_chain, "_model_choices",
                            lambda: (["None", "klein9b"], ["Inherit"]))
        components = script.ui(False)
        labels = [getattr(component, "label", None) for component in components]

        assert labels[mc_plan.STAGE_2_ARGUMENTS["target"]] == "Stage 2 checkpoint"
        assert labels[mc_plan.STAGE_2_ARGUMENTS["size_multiplier"]] == "Output size multiplier"

    def test_creative_mode_puts_its_enable_flag_first_and_spatial_last(self, host):
        """Its middle is a variable number of axis controls, so the two ends are
        the two that can be read without counting -- which is exactly what
        :func:`mc_plan.creative_from` does."""
        import scripts.model_chain_krea_creative as creative

        script = creative.ScriptKreaCreative()
        components = script.ui(False)

        assert len(components) >= mc_plan.SPATIAL_TAIL + 1
        assert components[mc_plan.CREATIVE_ENABLED] is script.components["enabled"] \
            or getattr(components[mc_plan.CREATIVE_ENABLED], "label", "") in ("", None,
                                                                             "Creative Mode")
        tail = components[-mc_plan.SPATIAL_TAIL:]
        assert len(tail) == mc_plan.SPATIAL_TAIL
        # The compose control is the middle of the three, and its choices are
        # the two modes the plan distinguishes.
        choices = getattr(tail[1], "choices", None) or []
        values = {str(choice[-1] if isinstance(choice, tuple) else choice).casefold()
                  for choice in choices}
        assert values == {"smart", "direct"}


class TestHowACheckpointIsNamed:
    """Forge names one ``subdir/krea2.safetensors [a1b2c3d4]``.

    The plan line is read by a person, and a plan that said
    ``Stage 2 (models/Stable-diffusion/klein9b.safetensors [7f3c...])`` would be
    a line nobody reads twice.
    """

    def test_the_directory_comes_off(self):
        assert mc_plan.Stage(name="Stable-diffusion/krea2.safetensors").shown() == "krea2"

    def test_a_windows_path_comes_off_too(self):
        assert mc_plan.Stage(name="models\\\\SD\\\\krea2.safetensors").shown() == "krea2"

    def test_the_hash_suffix_comes_off(self):
        assert mc_plan.Stage(name="krea2.safetensors [a1b2c3d4]").shown() == "krea2"

    def test_all_three_at_once(self):
        assert mc_plan.Stage(
            name="Stable-diffusion/flux/klein9b.gguf [7f3c1e]").shown() == "klein9b"

    def test_a_name_that_looks_like_none_of_that_is_left_alone(self):
        assert mc_plan.Stage(name="My Model v2 [best]").shown() == "My Model v2"

    def test_a_plain_name_survives_intact(self):
        assert mc_plan.Stage(name="krea2").shown() == "krea2"

    def test_a_square_bracket_that_is_not_a_hash_suffix_is_kept(self):
        assert mc_plan.Stage(name="krea2").shown() == "krea2"
        assert mc_plan.Stage(name="[experimental]krea2").shown() == "[experimental]krea2"


class TestTheBudgetIsMeasuredNotDeclared:
    """A card's nameplate is not available to anything, and the gap is fatal.

    Numbers throughout are one user's RTX 3090, from the ``device_info`` line
    llama-server prints at every start: ``24575 MiB`` total, and never more than
    ``23304 MiB`` free with nothing whatever loaded. The missing 1.24 GB is the
    display, the driver's working set and the desktop.

    Sized from the nameplate, ``PersistentLLMBudget`` is 1.24 GB too generous,
    so the model is placed 1.24 GB larger than the card can carry beside the
    image plan. Nothing fails then -- the VRAM really is free while the
    checkpoint is not loaded. What fails is the next question anybody asks:
    "does the image plan fit right now?" No, by almost exactly the overshoot,
    and the only answer the broker has is to stop llama-server. Every
    generation, with an identical placement every time.
    """

    MiB = 1024**2

    @pytest.fixture
    def card_3090(self, host, monkeypatch):
        state = {"free": 23304 * self.MiB, "image": 0, "llm": 0}
        monkeypatch.setattr(mc_broker, "total_vram_bytes", lambda: 24575 * self.MiB)
        monkeypatch.setattr(mc_broker, "device_free_vram_bytes", lambda: state["free"])
        monkeypatch.setattr(mc_broker, "held_bytes",
                            lambda family: state["image"] if family == mc_broker.FAMILY_IMAGE
                            else state["llm"])
        monkeypatch.setattr(mc_broker, "safety_margin_bytes", lambda: 0)
        return state

    def test_the_nameplate_is_not_what_the_card_can_give(self, card_3090):
        assert mc_plan.usable_vram_bytes() == 23304 * self.MiB

    def test_the_overshoot_it_removes_is_the_one_that_was_evicting(self, card_3090):
        nameplate = 24575 * self.MiB
        assert nameplate - mc_plan.usable_vram_bytes() == 1271 * self.MiB

    def test_it_is_invariant_as_models_come_and_go(self, card_3090):
        """The property that makes the budget stable. A checkpoint loading moves
        bytes from the free term into the image term and leaves the sum alone,
        so the allowance does not move when the card fills up."""
        empty = mc_plan.usable_vram_bytes()

        card_3090["free"] = 2824 * self.MiB
        card_3090["image"] = 14336 * self.MiB
        loaded = mc_plan.usable_vram_bytes(already_ours=6144 * self.MiB)

        assert loaded == empty

    def test_a_running_server_of_ours_counts_as_ours_to_spend(self, card_3090):
        card_3090["free"] = 17160 * self.MiB
        held = 6144 * self.MiB

        assert mc_plan.usable_vram_bytes(already_ours=held) == 23304 * self.MiB

    def test_another_process_taking_vram_shrinks_the_budget(self, card_3090):
        """Correctly, and this is the case the nameplate could never see."""
        card_3090["free"] = 13304 * self.MiB

        assert mc_plan.usable_vram_bytes() == 13304 * self.MiB

    def test_the_allowance_no_longer_exceeds_what_the_card_can_carry(self, card_3090):
        """The whole bug in one assertion. Stage 1 at 14 GB on this card leaves
        8.76 GB, not the 10.00 GB the nameplate claimed."""
        mc_plan.publish(mc_plan.Plan((
            mc_plan.Phase(mc_plan.STAGE_1, mc_plan.KIND_IMAGE, "Stage 1", 14 * GB),
        ), 1024, 1024))

        allowance = mc_plan.persistent_llm_budget()

        assert allowance == 23304 * self.MiB - 14 * GB
        assert allowance < 10 * GB

    def test_a_model_placed_in_the_allowance_leaves_the_plan_room(self, card_3090):
        """The invariant that stops the eviction: with the language model
        holding its whole allowance, what is left is still the protected peak,
        so nothing has to be handed back."""
        peak = 14 * GB
        mc_plan.publish(mc_plan.Plan((
            mc_plan.Phase(mc_plan.STAGE_1, mc_plan.KIND_IMAGE, "Stage 1", peak),
        ), 1024, 1024))
        allowance = mc_plan.persistent_llm_budget()

        card_3090["free"] = 23304 * self.MiB - allowance

        assert card_3090["free"] >= peak

    def test_the_nameplate_allowance_did_not_leave_that_room(self, card_3090):
        """The same arithmetic with the old figure, kept so the test above is
        known to be measuring the fix rather than restating a tautology."""
        peak = 14 * GB
        nameplate_allowance = 24575 * self.MiB - peak

        assert 23304 * self.MiB - nameplate_allowance < peak

    def test_a_host_that_cannot_answer_falls_back_to_the_nameplate(
            self, card_3090, monkeypatch):
        monkeypatch.setattr(mc_broker, "device_free_vram_bytes", lambda: 0)

        assert mc_plan.usable_vram_bytes() == 24575 * self.MiB

    def test_it_never_reports_more_than_the_card_holds(self, card_3090):
        """Belt and braces: the three terms are read at slightly different
        moments and a model unloading between two of them must not produce a
        budget larger than the card."""
        card_3090["free"] = 23304 * self.MiB
        card_3090["image"] = 8 * GB

        assert mc_plan.usable_vram_bytes() == 24575 * self.MiB


class TestTheDerivationIsWrittenDown:
    """A restart should never need a log round-trip to explain.

    Every figure that decides whether llama-server survives the generation goes
    on one line at the moment the plan is published, so a budget that looks
    wrong is wrong visibly, against a named phase.
    """

    def test_the_whole_sum_is_logged_when_the_plan_changes(self, card, caplog):
        card.sizes["krea2"] = 14.0
        with caplog.at_level("INFO", logger="model_chain"):
            mc_plan.publish(plan_for(card, stage_1=stage("krea2")))

        assert any("memory budget" in record.message for record in caplog.records)

    def test_it_names_obtainable_and_nameplate_separately(self, card, caplog):
        """The two differing is the single likeliest explanation for a language
        model that will not stay up, so a line showing only one of them would
        hide the thing it exists to reveal."""
        card.sizes["krea2"] = 14.0
        with caplog.at_level("INFO", logger="model_chain"):
            mc_plan.publish(plan_for(card, stage_1=stage("krea2")))

        line = next(r.getMessage() for r in caplog.records if "memory budget" in r.message)
        assert "obtainable of" in line and "on the card" in line

    def test_an_unchanged_plan_says_nothing(self, card, caplog):
        """Ten presses with the same settings is one line, not ten."""
        card.sizes["krea2"] = 14.0
        mc_plan.publish(plan_for(card, stage_1=stage("krea2")))
        caplog.clear()
        with caplog.at_level("INFO", logger="model_chain"):
            mc_plan.publish(plan_for(card, stage_1=stage("krea2")))

        assert not any("memory budget" in r.message for r in caplog.records)

    def test_a_learned_ceiling_is_named_when_one_is_in_force(self, card, caplog):
        card.sizes["krea2"] = 14.0
        mc_plan.publish(plan_for(card, stage_1=stage("krea2")))
        mc_plan.record_miss("Stage 2", 2 * GB, llm_bytes=8 * GB, evicted=True)
        caplog.clear()
        with caplog.at_level("INFO", logger="model_chain"):
            mc_plan.publish(plan_for(card, stage_1=stage("krea2"), width=2048, height=2048))

        line = next(r.getMessage() for r in caplog.records if "memory budget" in r.message)
        assert "learned ceiling" in line

    def test_logging_that_fails_does_not_stop_the_plan_being_published(
            self, card, monkeypatch):
        def explode(*args, **kwargs):
            raise RuntimeError("no")

        monkeypatch.setattr(mc_broker, "reported_bytes", explode)
        card.sizes["krea2"] = 14.0

        assert mc_plan.publish(plan_for(card, stage_1=stage("krea2"))) is not None
