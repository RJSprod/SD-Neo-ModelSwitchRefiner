"""Cross-workload residency policy.

The invariant under test throughout is the one section 8 states as a sentence:
"Never unload merely because another workload started. Demote only because the
incoming workload actually needs the memory." Most of what follows is a way of
asking that question from a different angle.
"""

from __future__ import annotations

import threading
import time

import pytest

import mc_broker

_GB = 1024**3


class Recorder:
    """A reclaimer that records what it was asked for and frees what it is told."""

    def __init__(self, holds=0, frees=None, label="the recorded workload"):
        self.holds = holds
        self.frees = holds if frees is None else frees
        self.label = label
        self.calls: list[tuple[int, str]] = []

    def release(self, needed_bytes, reason=""):
        self.calls.append((needed_bytes, reason))
        freed = min(self.frees, needed_bytes) if self.frees else 0
        self.holds = max(self.holds - freed, 0)
        return freed

    def resident_bytes(self):
        return self.holds

    def describe(self):
        return self.label


@pytest.fixture
def broker(host, monkeypatch):
    """A broker with an empty register, a known VRAM figure and no reserve."""
    mc_broker.clear()
    monkeypatch.setattr(mc_broker, "safety_margin_bytes", lambda: 0)
    for family in (mc_broker.FAMILY_IMAGE, mc_broker.FAMILY_LLM):
        mc_broker.unregister_reclaimer(family)
    yield mc_broker
    mc_broker.clear()
    for family in (mc_broker.FAMILY_IMAGE, mc_broker.FAMILY_LLM):
        mc_broker.unregister_reclaimer(family)


def set_free(monkeypatch, gigabytes):
    """Fix what the card reports as free."""
    monkeypatch.setattr(mc_broker, "free_vram_bytes", lambda: int(gigabytes * _GB))


class TestFitFirst:
    def test_nothing_moves_when_the_incoming_workload_already_fits(self, broker, monkeypatch):
        """The co-residency case, and the one the whole design exists for."""
        set_free(monkeypatch, 10)
        llm = Recorder(holds=4 * _GB)
        broker.register_reclaimer(broker.FAMILY_LLM, llm)

        result = broker.request_vram(broker.FAMILY_IMAGE, 6 * _GB, reason="a Stage 2 pass")

        assert result.satisfied
        assert not result.moved_anything
        assert llm.calls == []

    def test_only_the_deficit_is_freed_not_everything(self, broker, monkeypatch):
        """Section 8: "free only enough residency to satisfy the deficit"."""
        set_free(monkeypatch, 2)
        llm = Recorder(holds=12 * _GB)
        broker.register_reclaimer(broker.FAMILY_LLM, llm)

        broker.request_vram(broker.FAMILY_IMAGE, 5 * _GB)

        asked, _reason = llm.calls[0]
        assert asked == 3 * _GB

    def test_the_reserve_is_part_of_what_has_to_fit(self, broker, monkeypatch):
        set_free(monkeypatch, 6)
        llm = Recorder(holds=8 * _GB)
        broker.register_reclaimer(broker.FAMILY_LLM, llm)

        broker.request_vram(broker.FAMILY_IMAGE, 5 * _GB, margin=2 * _GB)

        assert llm.calls[0][0] == 1 * _GB

    def test_an_unreadable_card_causes_no_eviction(self, broker, monkeypatch):
        """Guessing at a deficit on no evidence is exactly what this must not do."""
        set_free(monkeypatch, 0)
        llm = Recorder(holds=8 * _GB)
        broker.register_reclaimer(broker.FAMILY_LLM, llm)

        result = broker.request_vram(broker.FAMILY_IMAGE, 20 * _GB)

        assert llm.calls == []
        assert not result.moved_anything


class TestExclusiveMode:
    def test_the_other_family_leaves_entirely(self, broker, host, monkeypatch):
        """Section 10: Exclusive mode is a promise about ownership, so the
        sweep is not limited to the arithmetic deficit."""
        host.shared.opts.set(broker.OPT_MODE, broker.MODE_EXCLUSIVE)
        set_free(monkeypatch, 4)
        llm = Recorder(holds=10 * _GB)
        broker.register_reclaimer(broker.FAMILY_LLM, llm)
        broker.declare(broker.FAMILY_LLM, "llm", "the LLM", 10 * _GB)

        broker.request_vram(broker.FAMILY_IMAGE, 5 * _GB)

        assert llm.calls[0][0] == 10 * _GB

    def test_a_sweep_ignores_a_pin(self, broker, host, monkeypatch):
        """Pinning is a hybrid-mode preference. Exclusive mode's answer to
        "which family owns VRAM" is not negotiable by a pin."""
        host.shared.opts.set(broker.OPT_MODE, broker.MODE_EXCLUSIVE)
        set_free(monkeypatch, 1)
        llm = Recorder(holds=8 * _GB)
        broker.register_reclaimer(broker.FAMILY_LLM, llm)
        broker.declare(broker.FAMILY_LLM, "llm", "the LLM", 8 * _GB, pinned=True)

        broker.request_vram(broker.FAMILY_IMAGE, 6 * _GB)

        assert llm.calls


class TestPolicy:
    def test_an_image_pass_outranks_an_idle_llm_under_every_policy(self, broker, host,
                                                                   monkeypatch):
        """Section 18's regression requirement: ordinary txt2img keeps working,
        whatever the LLM was promised."""
        for chosen in (broker.POLICY_ADAPTIVE, broker.POLICY_PRESERVE_IMAGE,
                       broker.POLICY_LLM_PRIORITY):
            host.shared.opts.set(broker.OPT_POLICY, chosen)
            set_free(monkeypatch, 1)
            llm = Recorder(holds=8 * _GB)
            broker.register_reclaimer(broker.FAMILY_LLM, llm)

            broker.request_vram(broker.FAMILY_IMAGE, 5 * _GB)

            assert llm.calls, f"{chosen} refused to yield to an image pass"

    def test_preserve_image_never_asks_the_image_side_for_anything(self, broker, host,
                                                                   monkeypatch):
        host.shared.opts.set(broker.OPT_POLICY, broker.POLICY_PRESERVE_IMAGE)
        set_free(monkeypatch, 1)
        image = Recorder(holds=12 * _GB)
        broker.register_reclaimer(broker.FAMILY_IMAGE, image)

        result = broker.request_vram(broker.FAMILY_LLM, 8 * _GB)

        assert image.calls == []
        assert not result.satisfied

    def test_llm_priority_demotes_the_image_model(self, broker, host, monkeypatch):
        host.shared.opts.set(broker.OPT_POLICY, broker.POLICY_LLM_PRIORITY)
        set_free(monkeypatch, 1)
        image = Recorder(holds=12 * _GB)
        broker.register_reclaimer(broker.FAMILY_IMAGE, image)

        broker.request_vram(broker.FAMILY_LLM, 8 * _GB)

        assert image.calls[0][0] == 7 * _GB

    def test_a_family_is_never_asked_to_make_room_for_itself(self, broker, monkeypatch):
        set_free(monkeypatch, 1)
        llm = Recorder(holds=8 * _GB)
        broker.register_reclaimer(broker.FAMILY_LLM, llm)

        broker.request_vram(broker.FAMILY_LLM, 6 * _GB)

        assert llm.calls == []


class TestRanking:
    def test_a_pinned_residency_is_spared(self, broker, monkeypatch):
        set_free(monkeypatch, 1)
        llm = Recorder(holds=8 * _GB)
        broker.register_reclaimer(broker.FAMILY_LLM, llm)
        broker.declare(broker.FAMILY_LLM, "llm", "the LLM", 8 * _GB, pinned=True)

        broker.request_vram(broker.FAMILY_IMAGE, 5 * _GB)

        assert llm.calls == []

    def test_an_active_residency_is_spared(self, broker, monkeypatch):
        """Sections 9 and 15: a model executing right now is never evicted."""
        set_free(monkeypatch, 1)
        llm = Recorder(holds=8 * _GB)
        broker.register_reclaimer(broker.FAMILY_LLM, llm)
        broker.declare(broker.FAMILY_LLM, "llm", "the LLM", 8 * _GB,
                       rank=broker.RANK_ACTIVE)

        broker.request_vram(broker.FAMILY_IMAGE, 5 * _GB)

        assert llm.calls == []

    def test_a_hot_residency_goes_stale_with_time(self, broker):
        entry = broker.declare(broker.FAMILY_LLM, "llm", "the LLM", _GB)
        assert entry.effective_rank == broker.RANK_HOT

        entry.last_used = time.monotonic() - broker.STALE_AFTER_SECONDS - 1

        assert entry.effective_rank == broker.RANK_STALE

    def test_pinning_outlives_staleness(self, broker):
        entry = broker.declare(broker.FAMILY_LLM, "llm", "the LLM", _GB, pinned=True)
        entry.last_used = time.monotonic() - broker.STALE_AFTER_SECONDS * 10

        assert entry.effective_rank == broker.RANK_PINNED
        assert not entry.evictable

    def test_partly_protected_residency_still_gives_up_the_rest(self, broker, monkeypatch):
        """One pinned model does not make the whole family untouchable."""
        set_free(monkeypatch, 1)
        llm = Recorder(holds=12 * _GB)
        broker.register_reclaimer(broker.FAMILY_LLM, llm)
        broker.declare(broker.FAMILY_LLM, "pinned", "pinned weights", 4 * _GB, pinned=True)
        broker.declare(broker.FAMILY_LLM, "spare", "the KV cache", 8 * _GB)

        broker.request_vram(broker.FAMILY_IMAGE, 10 * _GB)

        # Nine short, but only eight are unpinned, so eight is what is asked for.
        assert llm.calls[0][0] == 8 * _GB


class TestSerialization:
    def test_two_workloads_do_not_overlap(self, broker):
        order: list[str] = []
        entered = threading.Event()

        def second():
            entered.wait(2)
            with broker.workload(broker.FAMILY_IMAGE, "an image pass"):
                order.append("image")

        thread = threading.Thread(target=second)
        with broker.workload(broker.FAMILY_LLM, "an LLM turn"):
            thread.start()
            entered.set()
            time.sleep(0.1)
            order.append("llm")
        thread.join(2)

        assert order == ["llm", "image"]

    def test_the_same_thread_may_nest(self, broker):
        """A chained generation is one workload with two stages, not two
        workloads, and Stage 2 must not deadlock on Stage 1's lock."""
        with broker.workload(broker.FAMILY_IMAGE, "stage 1"):
            with broker.workload(broker.FAMILY_IMAGE, "stage 2") as held:
                assert held
                assert broker.active().label == "stage 2"
        assert broker.active() is None

    def test_background_work_stands_aside_for_a_foreground_request(self, broker):
        """Section 15: background warming never has priority over a foreground
        request."""
        holding = threading.Event()
        release = threading.Event()

        def foreground():
            with broker.workload(broker.FAMILY_IMAGE, "a generation"):
                holding.set()
                release.wait(2)

        thread = threading.Thread(target=foreground)
        thread.start()
        holding.wait(2)

        with broker.workload(broker.FAMILY_IMAGE, "a preload", background=True) as held:
            assert not held

        release.set()
        thread.join(2)

    def test_an_optional_workload_gives_up_rather_than_blocking(self, broker):
        holding = threading.Event()
        release = threading.Event()

        def occupy():
            with broker.workload(broker.FAMILY_LLM, "an LLM turn"):
                holding.set()
                release.wait(2)

        thread = threading.Thread(target=occupy)
        thread.start()
        holding.wait(2)

        with broker.workload(broker.FAMILY_IMAGE, "a pass", timeout=0.05,
                             required=False) as held:
            assert not held

        release.set()
        thread.join(2)

    def test_a_required_workload_that_times_out_says_who_has_the_gpu(self, broker):
        holding = threading.Event()
        release = threading.Event()

        def occupy():
            with broker.workload(broker.FAMILY_LLM, "an LLM turn"):
                holding.set()
                release.wait(2)

        thread = threading.Thread(target=occupy)
        thread.start()
        holding.wait(2)

        with pytest.raises(broker.Busy, match="an LLM turn"):
            with broker.workload(broker.FAMILY_IMAGE, "a pass", timeout=0.05):
                pass

        release.set()
        thread.join(2)


class TestHostExclusion:
    def test_a_running_host_job_is_visible(self, broker, host):
        host.shared.state.job = "task(abc)"
        assert broker.host_busy()

        host.shared.state.job = ""
        host.shared.state.job_count = 0
        assert not broker.host_busy()

    def test_await_idle_returns_at_once_when_no_llm_is_running(self, broker):
        assert broker.await_idle(timeout=0.1)

    def test_await_idle_gives_up_rather_than_blocking_a_generation_forever(self, broker):
        holding = threading.Event()
        release = threading.Event()

        def occupy():
            with broker.workload(broker.FAMILY_LLM, "a stuck turn"):
                holding.set()
                release.wait(3)

        thread = threading.Thread(target=occupy)
        thread.start()
        holding.wait(2)

        assert not broker.await_idle(timeout=0.1)

        release.set()
        thread.join(3)


class TestExplainability:
    def test_a_demotion_is_recorded_with_its_reason(self, broker, monkeypatch):
        """Section 14: the panel has to be able to say why a model was demoted."""
        set_free(monkeypatch, 1)
        broker.register_reclaimer(broker.FAMILY_LLM,
                                  Recorder(holds=8 * _GB, label="the LLM (Q6_K)"))

        broker.request_vram(broker.FAMILY_IMAGE, 5 * _GB, reason="a Stage 2 pass")

        text = " ".join(entry.text for entry in broker.decisions())
        assert "the LLM (Q6_K)" in text
        assert "a Stage 2 pass" in text

    def test_a_shortfall_with_nothing_to_evict_is_recorded_too(self, broker, monkeypatch):
        set_free(monkeypatch, 1)

        broker.request_vram(broker.FAMILY_IMAGE, 8 * _GB, reason="a Stage 2 pass")

        assert any("short" in entry.text for entry in broker.decisions())

    def test_status_reports_both_families_and_the_active_workload(self, broker, monkeypatch):
        set_free(monkeypatch, 4)
        broker.declare(broker.FAMILY_IMAGE, "ckpt", "a checkpoint", 6 * _GB)
        broker.declare(broker.FAMILY_LLM, "llm", "the LLM", 8 * _GB)

        with broker.workload(broker.FAMILY_LLM, "an LLM turn"):
            status = broker.status()

        assert status.image_bytes == 6 * _GB
        assert status.llm_bytes == 8 * _GB
        assert status.owners == ("image", "LLM")
        assert status.active.label == "an LLM turn"


class TestSettings:
    def test_a_label_chosen_in_settings_resolves_to_its_value(self, broker, host):
        """The Settings page stores what its radio displayed, which is a label."""
        host.shared.opts.set(broker.OPT_MODE,
                             broker.label_for(broker.MODES, broker.MODE_EXCLUSIVE))

        assert broker.mode() == broker.MODE_EXCLUSIVE

    def test_a_bare_value_resolves_too(self, broker, host):
        host.shared.opts.set(broker.OPT_POLICY, broker.POLICY_LLM_PRIORITY)

        assert broker.policy() == broker.POLICY_LLM_PRIORITY

    def test_an_unrecognised_setting_falls_back_to_the_default(self, broker, host):
        host.shared.opts.set(broker.OPT_MODE, "something else entirely")

        assert broker.mode() == broker.MODE_HYBRID


class TestStatusSeesTheImageSide:
    def test_a_loaded_checkpoint_appears_even_though_nothing_declared_it(self, broker,
                                                                        monkeypatch):
        """Image checkpoints are loaded and moved by Forge; mc_memory cooperates
        with that rather than announcing every load to the register. A panel
        that only listed declared residency would show the LLM and pretend the
        checkpoint was not there."""
        set_free(monkeypatch, 4)
        broker.register_reclaimer(broker.FAMILY_IMAGE,
                                  Recorder(holds=6 * _GB, label="the image checkpoint"))

        status = broker.status()

        assert status.image_bytes == 6 * _GB
        assert any("checkpoint" in entry.label for entry in status.residencies)

    def test_a_declared_residency_is_not_double_counted(self, broker, monkeypatch):
        set_free(monkeypatch, 4)
        broker.register_reclaimer(broker.FAMILY_LLM, Recorder(holds=8 * _GB))
        broker.declare(broker.FAMILY_LLM, "llm", "the LLM", 8 * _GB)

        status = broker.status()

        assert status.llm_bytes == 8 * _GB
        assert len([e for e in status.residencies if e.family == broker.FAMILY_LLM]) == 1

    def test_a_reclaimer_that_raises_does_not_break_the_panel(self, broker, monkeypatch):
        class Broken:
            def release(self, needed_bytes, reason=""):
                return 0

            def resident_bytes(self):
                raise RuntimeError("cannot tell")

        set_free(monkeypatch, 4)
        broker.register_reclaimer(broker.FAMILY_IMAGE, Broken())

        assert broker.status().image_bytes == 0


class TestVramNobodyAdmitsTo:
    """"Nothing evictable was found" is true and, on its own, misleading: it
    reads as though the card were full of things this extension chose not to
    move, when what it usually means is that the card is full of something it
    cannot see at all — another program, or a llama-server left running by a
    WebUI that was killed rather than closed. Two very different problems.
    """

    def test_a_card_held_by_nothing_here_is_reported_as_such(self, broker, monkeypatch):
        monkeypatch.setattr(mc_broker, "total_vram_bytes", lambda: 24 * _GB)
        monkeypatch.setattr(mc_broker, "free_vram_bytes", lambda: 4 * _GB)

        assert round(mc_broker.unaccounted_bytes() / _GB) == 19

    def test_what_a_family_holds_is_not_unaccounted_for(self, broker, monkeypatch):
        monkeypatch.setattr(mc_broker, "total_vram_bytes", lambda: 24 * _GB)
        monkeypatch.setattr(mc_broker, "free_vram_bytes", lambda: 4 * _GB)
        mc_broker.declare(mc_broker.FAMILY_LLM, "llm:x", "the LLM", 19 * _GB)

        assert mc_broker.unaccounted_bytes() == 0

    def test_the_driver_s_own_share_is_not_worth_reporting(self, broker, monkeypatch):
        """A card in a desktop is never entirely free: a CUDA context is
        hundreds of megabytes before a single weight is loaded."""
        monkeypatch.setattr(mc_broker, "total_vram_bytes", lambda: 24 * _GB)
        monkeypatch.setattr(mc_broker, "free_vram_bytes", lambda: 23.5 * _GB)

        assert mc_broker.unaccounted_bytes() == 0

    def test_a_shortfall_says_where_the_card_went(self, broker, monkeypatch):
        monkeypatch.setattr(mc_broker, "total_vram_bytes", lambda: 24 * _GB)
        monkeypatch.setattr(mc_broker, "free_vram_bytes", lambda: 4 * _GB)

        mc_broker.request_vram(mc_broker.FAMILY_LLM, 18 * _GB, reason="an LLM request")

        said = [entry.text for entry in mc_broker.decisions()]
        assert any("not managing" in text for text in said), said
