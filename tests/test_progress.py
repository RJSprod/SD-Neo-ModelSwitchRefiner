"""Whole-job progress and ETA.

The host's own calculation counts sampler jobs and steps, which describes one
sampling loop well and a two-stage chain badly: it reaches 100% when Stage 1
ends, drops back when Stage 2 is added to the count, and never accounts for the
seconds spent moving a model between them. The tests here are about the three
properties that replace it -- the bar only ever moves forwards, time spent
outside the sampler is part of it, and the prediction is measured on the
machine it runs on rather than guessed.

Time is faked throughout. A test that measured real durations would be a test of
the machine running it.
"""

from __future__ import annotations

import json
import re
import types
from pathlib import Path

import pytest

import mc_progress

EXTENSION_ROOT = Path(__file__).resolve().parent.parent
STYLESHEET = EXTENSION_ROOT / "style.css"
STYLE_SCRIPT = EXTENSION_ROOT / "javascript" / "model_chain_progress.js"

from test_orchestration import DEFAULTS, UI_ORDER, make_p, make_processed, run_chain  # noqa: F401


GB = 1024**3


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


class Clock:
    """A perf_counter that only moves when a test says so."""

    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds
        return self.now


@pytest.fixture
def clock(monkeypatch):
    fake = Clock()
    monkeypatch.setattr(mc_progress.time, "perf_counter", fake)
    return fake


def build_job(**overrides):
    """The spec's worked example: batch of two, 20 steps then 8."""
    facts = dict(
        stage1_arch="SDXL",
        stage1_passes=[(20, 1.0)],
        batch_size=2,
        stage2_arch="Flux.2 Klein 9B",
        stage2_passes=[(8, 1.0), (8, 1.0)],
        transition="warm",
        move_gigabytes=8.0,
        free_gigabytes=2.0,
        target_label="klein",
    )
    facts.update(overrides)
    return mc_progress.build(**facts)


# --------------------------------------------------------------------------- #
# The calibration store
# --------------------------------------------------------------------------- #


class TestMeasurement:
    def test_the_first_observation_is_taken_whole(self):
        """Nothing is gained by averaging a real measurement with a guess."""
        mc_progress._record(("move:warm",), seconds=8.0, units=4.0)

        assert mc_progress.rates()["move:warm"] == pytest.approx(2.0)

    def test_later_observations_are_smoothed_towards_the_newest(self):
        mc_progress._record(("move:warm",), seconds=8.0, units=4.0)
        mc_progress._record(("move:warm",), seconds=4.0, units=4.0)

        rate = mc_progress.rates()["move:warm"]
        assert 1.0 < rate < 2.0
        # Recent evidence has to outweigh the older figure, or a machine that
        # changed -- a new drive, a different card -- would take a dozen jobs to
        # be believed.
        assert rate < 1.7

    def test_every_key_a_phase_answers_to_learns_from_it(self):
        mc_progress._record(("sample:SDXL:b4", "sample:SDXL", "sample"), seconds=10.0, units=5.0)

        rates = mc_progress.rates()
        assert rates["sample:SDXL:b4"] == pytest.approx(2.0)
        assert rates["sample:SDXL"] == pytest.approx(2.0)
        assert rates["sample"] == pytest.approx(2.0)

    def test_a_phase_with_nothing_to_measure_teaches_nothing(self):
        mc_progress._record(("free",), seconds=3.0, units=0.0)

        assert "free" not in mc_progress.rates()

    def test_a_zero_duration_teaches_nothing(self):
        """A phase that was skipped is not evidence that it is free."""
        mc_progress._record(("free",), seconds=0.0, units=4.0)

        assert "free" not in mc_progress.rates()

    def test_the_most_specific_key_wins(self):
        mc_progress._record(("sample",), seconds=10.0, units=1.0)
        mc_progress._record(("sample:SDXL:b4",), seconds=2.0, units=1.0)

        assert mc_progress.rate_for(("sample:SDXL:b4", "sample:SDXL", "sample")) == pytest.approx(2.0)

    def test_an_unmeasured_key_falls_back_to_a_broader_one(self):
        mc_progress._record(("sample:SDXL",), seconds=3.0, units=1.0)

        assert mc_progress.rate_for(("sample:SDXL:b7", "sample:SDXL", "sample")) == pytest.approx(3.0)

    def test_with_nothing_measured_the_baseline_stands_in(self):
        assert mc_progress.rate_for(("move:warm",)) == pytest.approx(mc_progress.BASELINES["move:warm"])

    def test_an_unknown_transition_is_costed_as_the_slowest_one(self):
        """An under-estimate reads as a hang; an over-estimate reads as slack."""
        assert mc_progress.rate_for(("move:something_new",)) == pytest.approx(
            mc_progress.BASELINES["move:disk"]
        )


class TestFirstRunBaseline:
    """The very first job has no measurements and still has to be plausible."""

    def test_the_sampling_guess_follows_the_card(self, monkeypatch):
        import mc_memory

        monkeypatch.setattr(mc_memory, "total_vram_bytes", lambda: 8 * GB)
        small = mc_progress.rate_for(("sample",))

        monkeypatch.setattr(mc_memory, "total_vram_bytes", lambda: 48 * GB)
        large = mc_progress.rate_for(("sample",))

        assert small > large

    def test_the_guess_is_clamped_at_both_ends(self, monkeypatch):
        import mc_memory

        monkeypatch.setattr(mc_memory, "total_vram_bytes", lambda: 1 * GB)
        assert mc_progress.rate_for(("sample",)) <= mc_progress._MAX_SAMPLE_BASELINE

        monkeypatch.setattr(mc_memory, "total_vram_bytes", lambda: 4096 * GB)
        assert mc_progress.rate_for(("sample",)) >= mc_progress._MIN_SAMPLE_BASELINE

    def test_hardware_that_cannot_be_read_still_gives_an_estimate(self, monkeypatch):
        import mc_memory

        def boom():
            raise RuntimeError("no device")

        monkeypatch.setattr(mc_memory, "total_vram_bytes", boom)

        assert mc_progress.rate_for(("sample",)) > 0


class TestPersistence:
    def test_measurements_survive_a_restart(self, monkeypatch, tmp_path):
        mc_progress._record(("move:warm",), seconds=8.0, units=4.0)
        mc_progress.save()

        # A fresh session reads the file rather than starting from the table.
        monkeypatch.setattr(mc_progress, "_rates", None)

        assert mc_progress.rates()["move:warm"] == pytest.approx(2.0)

    def test_a_damaged_file_is_treated_as_no_measurements(self, monkeypatch):
        with open(mc_progress.path(), "w", encoding="utf-8") as handle:
            handle.write("{not json")
        monkeypatch.setattr(mc_progress, "_rates", None)

        assert mc_progress.rates() == {}

    def test_a_nonsense_rate_is_dropped_rather_than_believed(self, monkeypatch):
        """A stored zero would make a phase predict instantly."""
        with open(mc_progress.path(), "w", encoding="utf-8") as handle:
            json.dump({"version": 1, "rates": {"free": 0, "move:warm": "fast", "decode": -1}}, handle)
        monkeypatch.setattr(mc_progress, "_rates", None)

        assert mc_progress.rates() == {}

    def test_a_store_that_cannot_be_written_is_not_fatal(self, monkeypatch):
        monkeypatch.setattr(mc_progress, "path", lambda: "/does/not/exist/nowhere.json")
        mc_progress._record(("free",), seconds=4.0, units=2.0)

        mc_progress.save()  # must not raise


# --------------------------------------------------------------------------- #
# The phase model
# --------------------------------------------------------------------------- #


class TestMonotonicProgress:
    """The complaint this module exists to answer.

    Today the bar reaches 100% when Stage 1 finishes and drops to a third when
    Stage 2's images are added to the job count.
    """

    def test_the_bar_never_moves_backwards_across_the_whole_job(self, clock):
        job = build_job()
        mc_progress.begin(job)

        seen = []

        def sample(sampling=None):
            progress, _, _ = job.snapshot(sampling)
            seen.append(progress)

        mc_progress.enter(mc_progress.PHASE_STAGE1)
        mc_progress.note_pass()
        for step in range(0, 21, 5):
            clock.advance(5)
            sample((step, 20))

        mc_progress.enter(mc_progress.PHASE_SWITCH)
        for _ in range(4):
            clock.advance(2)
            sample()

        mc_progress.enter(mc_progress.PHASE_STAGE2)
        for image in range(2):
            mc_progress.note_pass()
            for step in range(0, 9, 4):
                clock.advance(1)
                sample((step, 8))

        mc_progress.enter(mc_progress.PHASE_FINALIZE)
        clock.advance(1)
        sample()

        assert seen == sorted(seen), "progress went backwards"
        assert len(set(seen)) > 5, "the bar barely moved"

    def test_stage_1_finishing_does_not_fill_the_bar(self, clock):
        """The acceptance example: 20 steps, then 8, over a batch of two."""
        job = build_job()
        mc_progress.begin(job)

        mc_progress.enter(mc_progress.PHASE_STAGE1)
        mc_progress.note_pass()
        clock.advance(job.phases[2].estimate)

        at_stage_1_end, _, _ = job.snapshot((20, 20))

        assert at_stage_1_end < 0.9

    def test_the_bar_never_reports_completion(self, clock):
        """The host owns completion -- it removes the bar when the task ends."""
        job = build_job()
        mc_progress.begin(job)
        mc_progress.enter(mc_progress.PHASE_FINALIZE)
        clock.advance(3600)

        progress, _, _ = job.snapshot(None)

        assert progress < 1.0


class TestModelMovementIsVisible:
    def test_the_switch_is_part_of_the_prediction(self):
        with_move = build_job(move_gigabytes=16.0, transition="disk")
        without = build_job(move_gigabytes=0.0, transition="unchanged")

        assert with_move.estimate > without.estimate + 10

    def test_the_residency_kind_decides_the_cost(self):
        """A warm swap and a disk read differ by an order of magnitude."""
        warm = build_job(transition="warm")
        disk = build_job(transition="disk")

        warm_phase = warm.phases[warm._index[mc_progress.PHASE_SWITCH]]
        disk_phase = disk.phases[disk._index[mc_progress.PHASE_SWITCH]]

        assert disk_phase.estimate > warm_phase.estimate * 3

    def test_the_bar_keeps_moving_while_weights_move(self, clock):
        """Nothing reports partial completion of a model load."""
        job = build_job()
        mc_progress.begin(job)
        mc_progress.enter(mc_progress.PHASE_SWITCH)

        clock.advance(1)
        first, _, _ = job.snapshot(None)
        clock.advance(1)
        second, _, _ = job.snapshot(None)

        assert second > first

    def test_an_overrunning_phase_still_counts_down(self, clock):
        """A stalled ETA at zero is indistinguishable from a hang."""
        job = build_job()
        mc_progress.begin(job)
        mc_progress.enter(mc_progress.PHASE_SWITCH)
        clock.advance(job.phases[job._index[mc_progress.PHASE_SWITCH]].estimate * 4)

        _, eta, _ = job.snapshot(None)

        assert eta > 0


class TestPhaseAccounting:
    def test_a_phase_that_never_ran_shortens_what_is_left(self, clock):
        job = build_job()
        mc_progress.begin(job)

        clock.advance(1)
        _, before, _ = job.snapshot(None)

        # Straight to Stage 1: nothing had to be freed for it.
        mc_progress.enter(mc_progress.PHASE_STAGE1)
        _, after, _ = job.snapshot(None)

        assert after < before

    def test_a_skipped_phase_teaches_the_store_nothing(self, clock):
        job = build_job()
        mc_progress.begin(job)
        mc_progress.enter(mc_progress.PHASE_STAGE2)
        mc_progress.end()

        assert "free" not in mc_progress.rates()

    def test_a_measured_phase_teaches_the_store(self, clock):
        job = build_job(move_gigabytes=8.0)
        mc_progress.begin(job)
        mc_progress.enter(mc_progress.PHASE_SWITCH)
        clock.advance(16)
        mc_progress.enter(mc_progress.PHASE_STAGE2)
        mc_progress.end()

        assert mc_progress.rates()["move:warm"] == pytest.approx(2.0)

    def test_an_abandoned_job_teaches_the_store_nothing(self, clock):
        """An interrupted job measures the interruption, not the work."""
        job = build_job()
        mc_progress.begin(job)
        mc_progress.enter(mc_progress.PHASE_SWITCH)
        clock.advance(0.2)
        mc_progress.abandon()

        assert mc_progress.rates() == {}

    def test_phases_measured_before_the_plan_existed_still_count(self, clock):
        """The preload wait and Stage 1 preparation are over by then."""
        job = build_job()
        job.record(mc_progress.PHASE_JOIN, 4.0)
        job.record(mc_progress.PHASE_STAGE1_PREPARE, 6.0, units=3.0)
        mc_progress.begin(job, since=clock.now - 10.0)

        assert mc_progress.rates()["free"] == pytest.approx(2.0)
        # The ten seconds already gone are part of the job, not free time.
        progress, _, _ = job.snapshot(None)
        assert progress > 0

    def test_the_label_names_the_phase_that_is_running(self, clock):
        job = build_job()
        mc_progress.begin(job)
        mc_progress.enter(mc_progress.PHASE_SWITCH)

        _, _, label = job.snapshot(None)

        assert "klein" in label

    def test_a_label_can_count_through_the_batch(self, clock):
        job = build_job()
        mc_progress.begin(job)
        mc_progress.enter(mc_progress.PHASE_STAGE2)
        mc_progress.relabel("Stage 2 1/2")

        before, _, label = job.snapshot(None)
        mc_progress.relabel("Stage 2 2/2")
        after, _, _ = job.snapshot(None)

        assert label == "Stage 2 1/2"
        # A relabel is not progress; the percentage must not react to it.
        assert after >= before


class TestSamplingInterpolation:
    def test_uneven_passes_are_weighted_by_their_cost(self, clock):
        """Hires fix pairs a long first pass with a shorter second one."""
        job = mc_progress.build(
            stage1_arch="SDXL",
            stage1_passes=[(30, 1.0), (10, 4.0)],
            batch_size=1,
            stage2_arch="SDXL",
            stage2_passes=[(8, 1.0)],
            transition="warm",
            move_gigabytes=0.0,
            free_gigabytes=0.0,
        )
        phase = job.phases[job._index[mc_progress.PHASE_STAGE1]]

        # 30 steps at 1 MP against 10 steps at 4 MP: the second pass is the
        # larger share, so finishing the first must not read as halfway.
        assert phase.weights == (30.0, 40.0)

    def test_the_bar_creeps_on_after_the_last_step(self, clock):
        """Sampling ending is not the pass ending -- the latents still decode."""
        job = build_job()
        mc_progress.begin(job)
        mc_progress.enter(mc_progress.PHASE_STAGE1)
        mc_progress.note_pass()
        clock.advance(20)

        first, _, _ = job.snapshot((20, 20))
        clock.advance(2)
        second, _, _ = job.snapshot((20, 20))

        assert second > first

    def test_a_slow_pass_pushes_the_estimate_out(self, clock):
        """Real steps beat the prediction the moment there are any."""
        job = build_job()
        mc_progress.begin(job)
        mc_progress.enter(mc_progress.PHASE_STAGE1)
        mc_progress.note_pass()

        clock.advance(10)
        _, quick, _ = job.snapshot((10, 20))

        clock.advance(60)
        _, slow, _ = job.snapshot((11, 20))

        assert slow > quick

    def test_a_pass_the_host_never_reports_does_not_stall_the_bar(self, clock):
        job = build_job()
        mc_progress.begin(job)
        mc_progress.enter(mc_progress.PHASE_STAGE1)

        clock.advance(5)
        first, _, _ = job.snapshot(None)
        clock.advance(5)
        second, _, _ = job.snapshot(None)

        assert second > first


class TestBatching:
    def test_stage_2_gets_none_of_stage_1_s_batching_gain(self):
        """Every refine is its own batch-of-one pass through process_images."""
        one = mc_progress.build(
            stage1_arch="SDXL", stage1_passes=[(20, 1.0)], batch_size=1,
            stage2_arch="SDXL", stage2_passes=[(8, 1.0)],
            transition="warm", move_gigabytes=0.0, free_gigabytes=0.0,
        )
        four = mc_progress.build(
            stage1_arch="SDXL", stage1_passes=[(20, 1.0)], batch_size=4,
            stage2_arch="SDXL", stage2_passes=[(8, 1.0)] * 4,
            transition="warm", move_gigabytes=0.0, free_gigabytes=0.0,
        )

        one_stage_2 = one.phases[one._index[mc_progress.PHASE_STAGE2]]
        four_stage_2 = four.phases[four._index[mc_progress.PHASE_STAGE2]]

        assert four_stage_2.estimate == pytest.approx(one_stage_2.estimate * 4)

    def test_batch_count_multiplies_the_stage_1_passes(self):
        one = mc_progress.build(
            stage1_arch="SDXL", stage1_passes=[(20, 1.0)], batch_size=1,
            stage2_arch="SDXL", stage2_passes=[], transition="warm",
            move_gigabytes=0.0, free_gigabytes=0.0,
        )
        three = mc_progress.build(
            stage1_arch="SDXL", stage1_passes=[(20, 1.0)] * 3, batch_size=1,
            stage2_arch="SDXL", stage2_passes=[], transition="warm",
            move_gigabytes=0.0, free_gigabytes=0.0,
        )

        assert three.estimate > one.estimate

    def test_a_rate_learned_at_one_batch_size_is_kept_apart_from_another(self, clock):
        job = mc_progress.build(
            stage1_arch="SDXL", stage1_passes=[(20, 1.0)], batch_size=4,
            stage2_arch="SDXL", stage2_passes=[], transition="warm",
            move_gigabytes=0.0, free_gigabytes=0.0,
        )
        mc_progress.begin(job)
        mc_progress.enter(mc_progress.PHASE_STAGE1)
        clock.advance(40)
        mc_progress.end()

        rates = mc_progress.rates()
        assert "sample:SDXL:b4" in rates
        # Batching is sublinear, so a batch of one must not inherit the batch of
        # four's per-image rate as though it were exact.
        assert "sample:SDXL:b1" not in rates


class TestJobLifecycle:
    def test_a_new_job_replaces_the_last(self, clock):
        mc_progress.begin(build_job())
        mc_progress.enter(mc_progress.PHASE_STAGE2)
        mc_progress.begin(build_job())

        assert mc_progress.snapshot()[0] < 0.5

    def test_there_is_no_snapshot_without_a_job(self):
        mc_progress.abandon()

        assert mc_progress.snapshot() is None

    def test_a_finished_job_stops_reporting(self, clock):
        mc_progress.begin(build_job())
        mc_progress.end()

        assert mc_progress.snapshot() is None

    def test_calls_without_a_job_are_harmless(self):
        mc_progress.abandon()

        mc_progress.enter(mc_progress.PHASE_STAGE2)
        mc_progress.note_pass()
        mc_progress.relabel("nothing")
        mc_progress.end()

    def test_the_preload_is_not_a_phase(self):
        """It runs after the images are delivered and the bar is already gone."""
        assert not any("preload" in key.lower() for key in mc_progress.ORDER)


# --------------------------------------------------------------------------- #
# The host's progress endpoint
# --------------------------------------------------------------------------- #


class FakeResponse:
    def __init__(self, active=True):
        self.active = active
        self.progress = 0.25
        self.eta = 99.0
        self.textinfo = None
        self.live_preview = "data:image/png;base64,zzz"


class FakeRequest:
    pass


@pytest.fixture
def host_progress(monkeypatch):
    """A stand-in for modules.progress, wired as the host wires it.

    Faithful in the one way that matters: setup_progress_api resolves
    ``progressapi`` from the module globals when it registers the route, and
    that happens long after extensions are imported. Rebinding the global is
    only picked up because of that.
    """
    import sys

    module = types.ModuleType("modules.progress")
    module.ProgressRequest = FakeRequest
    module.responses = []

    def progressapi(req):
        response = FakeResponse()
        module.responses.append(response)
        return response

    module.progressapi = progressapi
    module.original = progressapi

    def setup_progress_api(app):
        """Registers whatever the global names at call time, as the host does."""
        app.append(module.progressapi)

    module.setup_progress_api = setup_progress_api

    import modules

    monkeypatch.setattr(modules, "progress", module, raising=False)
    monkeypatch.setitem(sys.modules, "modules.progress", module)
    monkeypatch.setattr(mc_progress, "_installed", False)
    return module


class TestEndpointWrapping:
    def test_the_route_registered_later_is_the_wrapped_one(self, host_progress):
        assert mc_progress.install() is True

        routes = []
        host_progress.setup_progress_api(routes)

        assert routes[0] is not host_progress.original

    def test_installing_twice_does_not_stack_wrappers(self, host_progress):
        mc_progress.install()
        wrapped = host_progress.progressapi
        mc_progress._installed = False
        mc_progress.install()

        assert host_progress.progressapi is wrapped

    def test_the_request_annotation_is_the_real_class(self, host_progress):
        """FastAPI builds the request model from it, and cannot resolve a name
        that only exists inside install()."""
        mc_progress.install()

        assert host_progress.progressapi.__annotations__["req"] is FakeRequest

    def test_a_host_without_the_endpoint_is_not_fatal(self, host_progress):
        del host_progress.progressapi

        assert mc_progress.install() is False


class TestEndpointOutput:
    def test_a_live_job_supplies_its_own_progress_and_eta(self, host, host_progress, clock):
        mc_progress.install()
        mc_progress.begin(build_job())
        mc_progress.enter(mc_progress.PHASE_STAGE2)
        clock.advance(5)

        response = host_progress.progressapi(FakeRequest())

        assert response.progress != 0.25
        assert response.eta != 99.0
        assert response.textinfo == "Stage 2"

    def test_the_host_keeps_everything_else(self, host, host_progress, clock):
        """Live previews, queueing and task bookkeeping stay the host's."""
        mc_progress.install()
        mc_progress.begin(build_job())

        response = host_progress.progressapi(FakeRequest())

        assert response.live_preview == "data:image/png;base64,zzz"

    def test_an_ordinary_generation_is_left_alone(self, host, host_progress):
        mc_progress.install()
        mc_progress.abandon()

        response = host_progress.progressapi(FakeRequest())

        assert response.progress == 0.25
        assert response.eta == 99.0
        assert response.textinfo is None

    def test_an_inactive_task_is_left_alone(self, host, host_progress, clock, monkeypatch):
        mc_progress.install()
        mc_progress.begin(build_job())

        def inactive(req):
            return FakeResponse(active=False)

        monkeypatch.setattr(host_progress, "original", inactive)
        mc_progress._installed = False
        monkeypatch.setattr(host_progress, "progressapi", inactive)
        mc_progress.install()

        response = host_progress.progressapi(FakeRequest())

        assert response.progress == 0.25

    def test_the_setting_switches_it_off(self, host_progress, host, clock):
        host.shared.opts.model_chain_progress = False
        mc_progress.install()
        mc_progress.begin(build_job())

        response = host_progress.progressapi(FakeRequest())

        assert response.progress == 0.25

    def test_a_label_is_kept_to_one_line(self, host, host_progress, clock):
        """The host's JS drops a textinfo containing a newline."""
        mc_progress.install()
        job = build_job()
        mc_progress.begin(job)
        mc_progress.enter(mc_progress.PHASE_STAGE2)
        mc_progress.relabel("Stage 2\n1/2")

        response = host_progress.progressapi(FakeRequest())

        assert "\n" not in response.textinfo

    def test_a_plan_left_over_from_an_earlier_task_is_dropped(self, host, host_progress, clock):
        """A generation that raised part-way leaves its phases behind.

        Describing the next task with the last one's plan is worse than not
        describing it: the host's own numbers at least belong to this job.
        """
        import time as real_time

        mc_progress.install()
        mc_progress.begin(build_job())
        host.shared.state.time_start = real_time.time() + 60

        response = host_progress.progressapi(FakeRequest())

        assert response.progress == 0.25
        assert mc_progress.active() is False

    def test_a_broken_snapshot_leaves_the_response_untouched(self, host, host_progress, monkeypatch):
        mc_progress.install()
        mc_progress.begin(build_job())

        def boom():
            raise RuntimeError("no")

        monkeypatch.setattr(mc_progress, "snapshot", boom)

        response = host_progress.progressapi(FakeRequest())

        assert response.progress == 0.25


# --------------------------------------------------------------------------- #
# Appearance
# --------------------------------------------------------------------------- #


def javascript_list(name):
    """The named array literal from the styling script."""
    match = re.search(rf"const {name} = \[(.*?)\]", STYLE_SCRIPT.read_text(), re.S)
    assert match, f"{name} not found in {STYLE_SCRIPT.name}"
    return re.findall(r'"([^"]+)"', match.group(1))


def javascript_themes():
    """The theme names the styling script knows how to render."""
    body = STYLE_SCRIPT.read_text()
    block = re.search(r"const THEMES = \{(.*?)\n    \};", body, re.S)
    assert block, "THEMES not found in the styling script"
    return re.findall(r"^\s{8}(\w+):", block.group(1), re.M)


def stylesheet():
    """The stylesheet with its comments removed.

    The comments discuss the very properties and selectors these tests forbid,
    which is the point of them -- but it means scanning the raw file would find
    the explanation rather than a violation.
    """
    return re.sub(r"/\*.*?\*/", "", STYLESHEET.read_text(), flags=re.S)


def styled_bar_rules():
    """Every rule that styles the host's own bar, as (selector, body) pairs."""
    css = stylesheet()
    return [
        (selector.strip(), body)
        for selector, body in re.findall(r"([^{}]+)\{([^{}]*)\}", css)
        if ".mc-progress-styled" in selector
    ]


class TestThemeRegistration:
    """The dropdown and the code that renders it must not drift apart.

    The themes are defined in the JS, because a theme is a choice of effects and
    the effects are CSS. Python only has to offer the names.
    """

    def test_the_dropdown_offers_exactly_the_themes_the_script_renders(self):
        import model_chain

        assert list(model_chain.STYLE_THEMES) == javascript_themes() + [model_chain.STYLE_CUSTOM]

    def test_the_default_theme_is_one_of_them(self):
        import model_chain

        assert model_chain.STYLE_THEME_DEFAULT in model_chain.STYLE_THEMES

    def test_every_effect_the_script_toggles_is_implemented(self):
        css = stylesheet()

        for effect in javascript_list("EFFECTS"):
            assert f".mc-fx-{effect}" in css, f"no rule for the {effect} effect"

    def test_the_appearance_settings_are_registered(self, host):
        import model_chain

        for name in (
            model_chain.OPT_STYLE_ENABLE,
            model_chain.OPT_STYLE_THEME,
            model_chain.OPT_STYLE_COLOR,
            model_chain.OPT_STYLE_GRADIENT,
            model_chain.OPT_STYLE_SHEEN,
            model_chain.OPT_STYLE_GLOW,
            model_chain.OPT_STYLE_COMPLETE,
        ):
            assert name in host.shared.options_templates


class TestStylesheetConstraints:
    """The three findings that shaped the stylesheet, guarded as tests.

    Each of these is a rule that reads as an arbitrary restriction until it is
    broken, at which point it is a bug in somebody else's theme.
    """

    def test_it_claims_no_pseudo_elements_on_the_bar(self):
        """Lobe already uses ::before and ::after on .progress.

        Claiming either replaces the theme's effect rather than layering with
        it, so every effect is expressed on the element itself.
        """
        css = stylesheet()

        assert "::before" not in css
        assert "::after" not in css

    def test_it_sets_no_geometry_on_the_bar(self):
        """Themes move this element; Lobe takes it out of the host's overlay.

        Colour and animation travel between themes. Height, offset and radius
        do not.
        """
        forbidden = ("height:", "top:", "left:", "right:", "bottom:", "position:", "border-radius:")

        for selector, body in styled_bar_rules():
            for property_name in forbidden:
                assert property_name not in body, f"{selector} sets {property_name}"

    def test_a_solid_fill_is_declared_before_any_gradient(self):
        """color-mix is recent; a browser without it must still get a bar."""
        css = stylesheet()

        assert css.index("background-color: var(--mc-progress-fill)") < css.index("color-mix")

    def test_every_effect_derives_from_the_one_colour(self):
        """Which is what lets a custom colour recolour every theme."""
        for selector, body in styled_bar_rules():
            if "color-mix" not in body:
                continue
            assert "--mc-progress-fill" in body, f"{selector} mixes a colour from nowhere"

    def test_motion_can_be_turned_off_by_the_system(self):
        css = stylesheet()
        reduced = css[css.index("prefers-reduced-motion") :]

        for effect in ("sheen", "pulse"):
            assert f"mc-fx-{effect}" in reduced, f"{effect} keeps animating under reduced motion"


# --------------------------------------------------------------------------- #
# Wiring into a generation
# --------------------------------------------------------------------------- #


class TestGenerationWiring:
    def test_a_chained_generation_plans_its_phases(self, chain, host, image_factory, monkeypatch):
        plans = []
        monkeypatch.setattr(
            mc_progress, "begin", lambda job, since=None: plans.append(job)
        )

        p = make_p(host, batch_size=2)
        run_chain(chain, host, p, make_processed(host, p, image_factory), steps=8)

        assert len(plans) == 1
        assert [phase.key for phase in plans[0].phases] == list(mc_progress.ORDER)

    def test_the_plan_follows_the_batch(self, chain, host, image_factory, monkeypatch):
        plans = []
        monkeypatch.setattr(mc_progress, "begin", lambda job, since=None: plans.append(job))

        p = make_p(host, batch_size=2, n_iter=3)
        run_chain(chain, host, p, make_processed(host, p, image_factory), steps=8)

        job = plans[0]
        stage_1 = job.phases[job._index[mc_progress.PHASE_STAGE1]]
        stage_2 = job.phases[job._index[mc_progress.PHASE_STAGE2]]

        # Three Stage 1 passes of a batch of two; six separate Stage 2 refines.
        assert len(stage_1.weights) == 3
        assert len(stage_2.weights) == 6

    def test_the_residency_prediction_decides_the_switch_cost(
        self, chain, host, image_factory, monkeypatch
    ):
        import mc_memory

        plans = []
        monkeypatch.setattr(mc_progress, "begin", lambda job, since=None: plans.append(job))
        monkeypatch.setattr(
            mc_memory, "plan", lambda name, mods=None: mc_memory.ResidencyPlan("disk", "cold")
        )
        monkeypatch.setattr(mc_memory, "file_size_bytes", lambda name, mods=None: 12 * GB)

        p = make_p(host)
        run_chain(chain, host, p, make_processed(host, p, image_factory), steps=8)

        switch = plans[0].phases[plans[0]._index[mc_progress.PHASE_SWITCH]]
        assert switch.units == pytest.approx(12.0)
        assert switch.rate_keys == ("move:disk",)

    def test_the_setting_switches_planning_off(self, chain, host, image_factory, monkeypatch):
        host.shared.opts.model_chain_progress = False
        plans = []
        monkeypatch.setattr(mc_progress, "begin", lambda job, since=None: plans.append(job))

        p = make_p(host)
        run_chain(chain, host, p, make_processed(host, p, image_factory))

        assert plans == []

    def test_a_plain_generation_plans_nothing(self, chain, host, image_factory, monkeypatch):
        plans = []
        monkeypatch.setattr(mc_progress, "begin", lambda job, since=None: plans.append(job))

        p = make_p(host)
        run_chain(chain, host, p, make_processed(host, p, image_factory), enabled=False)

        assert plans == []

    def test_a_chained_generation_leaves_no_job_behind(self, chain, host, image_factory):
        p = make_p(host)
        run_chain(chain, host, p, make_processed(host, p, image_factory))

        assert mc_progress.active() is False

    def test_a_failed_stage_2_abandons_the_job(self, chain, host, image_factory, monkeypatch):
        """Stage 1 really did run, so it is measured; the failed switch is not."""
        import mc_memory

        def boom(name, mods=None):
            raise RuntimeError("switch failed")

        monkeypatch.setattr(mc_memory, "ensure_resident", boom)

        p = make_p(host)
        run_chain(chain, host, p, make_processed(host, p, image_factory))

        rates = mc_progress.rates()
        assert mc_progress.active() is False
        assert not any(key.startswith("move:") for key in rates)
        assert "finalize" not in rates

    def test_an_interrupted_batch_is_not_measured(self, chain, host, image_factory, monkeypatch):
        original = chain.module.process_images

        def interrupt(p2):
            host.shared.state.interrupted = True
            return original(p2)

        monkeypatch.setattr(chain.module, "process_images", interrupt)

        p = make_p(host, batch_size=3)
        run_chain(chain, host, p, make_processed(host, p, image_factory))

        assert mc_progress.active() is False

    def test_a_planning_failure_does_not_break_the_generation(
        self, chain, host, image_factory, monkeypatch
    ):
        def boom(**kwargs):
            raise RuntimeError("no plan")

        monkeypatch.setattr(mc_progress, "build", boom)

        p = make_p(host, batch_size=2)
        processed = run_chain(chain, host, p, make_processed(host, p, image_factory))

        assert len(processed.images) == 2
        assert mc_progress.active() is False
