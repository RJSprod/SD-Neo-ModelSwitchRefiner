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


class Server:
    """An LLM reclaimer that can say whether its process is up.

    A llama-server placed in system RAM is running and holding nothing here,
    which is the whole of the case worth a fake: the register has no entry for
    it, and only asking the runtime distinguishes it from no server at all.
    """

    def __init__(self, running=False, holds=0):
        self.up, self.holds = running, holds

    def running(self):
        return self.up

    def resident_bytes(self):
        return self.holds

    def release(self, needed_bytes, reason=""):
        return 0

    def describe(self):
        return "llama-server"


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
    """Fix what the card reports as free, to both of the questions that asks.

    The host counts its allocator's cached blocks as free and another process
    cannot have them, so the broker asks a different question on the LLM's
    behalf than on the image side's. A test that set only one of the two would
    be describing a machine that does not exist.
    """
    monkeypatch.setattr(mc_broker, "free_vram_bytes", lambda: int(gigabytes * _GB))
    monkeypatch.setattr(mc_broker, "device_free_vram_bytes", lambda: int(gigabytes * _GB))


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


class TestWhatTheNoteSays:
    def test_a_sweep_reports_the_card_as_it_was_before_it_moved_anything(
            self, broker, host, monkeypatch):
        """The reading is re-taken after the sweep, and quoting it on both sides
        of the arrow printed "22.5 GB -> 22.5 GB free" on a call that had just
        recovered fourteen gigabytes."""
        host.shared.opts.set(broker.OPT_MODE, broker.MODE_EXCLUSIVE)
        card = [4 * _GB]
        monkeypatch.setattr(broker, "free_vram_bytes", lambda: card[0])
        monkeypatch.setattr(broker, "device_free_vram_bytes", lambda: card[0])

        class Mover(Recorder):
            def release(self, needed_bytes, reason=""):
                freed = super().release(needed_bytes, reason)
                card[0] += freed
                return freed

        broker.register_reclaimer(broker.FAMILY_LLM, Mover(holds=10 * _GB))

        broker.request_vram(broker.FAMILY_IMAGE, 20 * _GB)

        said = [entry.text for entry in broker.decisions()]
        assert any("4.0 GB -> 14.0 GB free" in text for text in said), said

    def test_an_exclusive_sweep_says_what_hybrid_would_have_done_instead(
            self, broker, host, monkeypatch):
        """The stop is the mode's promise, not a fault -- but it is a model load
        per image, and the setting that avoids it is worth naming once."""
        host.shared.opts.set(broker.OPT_MODE, broker.MODE_EXCLUSIVE)
        set_free(monkeypatch, 4)
        broker.register_reclaimer(broker.FAMILY_LLM, Recorder(holds=6 * _GB))

        broker.request_vram(broker.FAMILY_IMAGE, 2 * _GB)

        said = " ".join(entry.text for entry in broker.decisions())
        assert "Hybrid would have kept it warm" in said

    def test_a_request_that_moved_nothing_says_nothing_about_modes(
            self, broker, host, monkeypatch):
        host.shared.opts.set(broker.OPT_MODE, broker.MODE_EXCLUSIVE)
        set_free(monkeypatch, 20)

        broker.request_vram(broker.FAMILY_IMAGE, 2 * _GB)

        said = " ".join(entry.text for entry in broker.decisions())
        assert "Hybrid" not in said


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


class TestTheImageModelKeepsItsVram:
    """The one asymmetry: an image residency is never demoted for the LLM.

    The LLM is a helper that writes a prompt for the image model. A helper that
    throws the checkpoint off the card has made the job slower, because the
    checkpoint is wanted again seconds later and every byte borrowed is paid
    for twice -- once moving the weights out, once moving them back.
    """

    def test_an_image_pass_outranks_an_idle_llm(self, broker, host, monkeypatch):
        """Section 18's regression requirement: ordinary txt2img keeps working."""
        set_free(monkeypatch, 1)
        llm = Recorder(holds=8 * _GB)
        broker.register_reclaimer(broker.FAMILY_LLM, llm)

        broker.request_vram(broker.FAMILY_IMAGE, 5 * _GB)

        assert llm.calls

    def test_the_llm_never_asks_the_image_side_for_anything(self, broker, host,
                                                            monkeypatch):
        set_free(monkeypatch, 1)
        image = Recorder(holds=12 * _GB)
        broker.register_reclaimer(broker.FAMILY_IMAGE, image)

        result = broker.request_vram(broker.FAMILY_LLM, 8 * _GB)

        assert image.calls == []
        assert not result.satisfied

    def test_not_even_in_exclusive_mode(self, broker, host, monkeypatch):
        """The sweep is directional. Exclusive mode used to run it both ways,
        and a Krea roll on a 24 GB card evicted a 13.9 GB checkpoint it then
        reserved room for, so the generation two seconds later spent thirteen
        seconds moving the same weights back -- every press.
        """
        host.shared.opts.set(broker.OPT_MODE, broker.MODE_EXCLUSIVE)
        set_free(monkeypatch, 1)
        image = Recorder(holds=14 * _GB)
        broker.register_reclaimer(broker.FAMILY_IMAGE, image)

        broker.request_vram(broker.FAMILY_LLM, 8 * _GB)

        assert image.calls == []

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


def _taken_by_another_thread(broker, label) -> bool:
    """Whether a workload on a thread that is not this one gets the GPU.

    Asked from another thread on purpose: the lock is reentrant, so a hold this
    thread failed to give back is a hold this thread can take again -- which is
    precisely how a stranded lock hid until a run landed on a different Gradio
    worker.
    """
    taken: list[bool] = []

    def contend():
        with broker.workload(broker.FAMILY_IMAGE, label, timeout=0.2,
                             required=False) as held:
            taken.append(bool(held))

    thread = threading.Thread(target=contend)
    thread.start()
    thread.join(2)
    return taken == [True]


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

    def test_a_workload_may_be_given_back_from_another_thread(self, broker):
        """The failure this exists to stop: a Krea run whose generator was
        finalized by the garbage collector, on whichever thread happened to
        trigger the collection, could not give the card back at all -- and
        every later run waited for a job that had finished minutes before."""
        job = broker.workload(broker.FAMILY_LLM, "a Krea prompt")
        assert job.__enter__()

        elsewhere = threading.Thread(target=lambda: job.__exit__(None, None, None))
        elsewhere.start()
        elsewhere.join(2)

        assert broker.active() is None
        assert _taken_by_another_thread(broker, "a later pass")

    def test_giving_the_same_workload_back_twice_releases_once(self, broker):
        """A run that releases explicitly and is then unwound -- or closed after
        its finally already ran -- must not hand the card to somebody else while
        the job that took it is still on it."""
        job = broker.workload(broker.FAMILY_LLM, "a Krea prompt")
        job.__enter__()
        job.__exit__(None, None, None)

        with broker.workload(broker.FAMILY_IMAGE, "a pass"):
            job.__exit__(None, None, None)
            assert not _taken_by_another_thread(broker, "an intruding turn")

    def test_a_late_release_does_not_take_the_running_job_off_the_list(self, broker):
        """``active()`` is what a waiting run is told it is waiting for. An exit
        arriving after a later workload started used to pop that later job's
        name, leaving a wait with nothing to name."""
        stale = broker.workload(broker.FAMILY_LLM, "an abandoned turn")
        stale.__enter__()

        with broker.workload(broker.FAMILY_LLM, "the running turn") as held:
            assert held
            stale.__exit__(None, None, None)
            assert broker.active().label == "the running turn"

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
        host.shared.opts.set(broker.OPT_MODE, broker.MODE_EXCLUSIVE)

        assert broker.mode() == broker.MODE_EXCLUSIVE

    def test_a_label_whose_explanation_was_reworded_still_resolves(self, broker, host):
        """What a radio stores is the whole string. Rewriting the half after the
        dash must not silently reset everybody's residency mode to the default.
        """
        host.shared.opts.set(broker.OPT_MODE,
                             "Exclusive — one family owns VRAM at a time")

        assert broker.mode() == broker.MODE_EXCLUSIVE

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


class TestHeldBytes:
    """The one function to ask when the question is "is it already there".

    Callers that asked the register instead reserved room for a checkpoint that
    was already resident -- see ``mc_creative_krea.image_reserve_bytes``.
    """

    def test_it_reads_the_register_when_something_declared_itself(self, broker,
                                                                  monkeypatch):
        set_free(monkeypatch, 4)
        broker.declare(broker.FAMILY_LLM, "llm", "the LLM", 8 * _GB)

        assert broker.held_bytes(broker.FAMILY_LLM) == 8 * _GB

    def test_it_asks_the_family_when_nothing_declared_itself(self, broker, monkeypatch):
        set_free(monkeypatch, 4)
        broker.register_reclaimer(broker.FAMILY_IMAGE, Recorder(holds=13 * _GB))

        assert broker.resident_bytes(broker.FAMILY_IMAGE) == 0
        assert broker.held_bytes(broker.FAMILY_IMAGE) == 13 * _GB

    def test_it_never_double_counts(self, broker, monkeypatch):
        set_free(monkeypatch, 4)
        broker.register_reclaimer(broker.FAMILY_LLM, Recorder(holds=8 * _GB))
        broker.declare(broker.FAMILY_LLM, "llm", "the LLM", 8 * _GB)

        assert broker.held_bytes(broker.FAMILY_LLM) == 8 * _GB

    def test_it_answers_zero_for_a_family_holding_nothing(self, broker, monkeypatch):
        set_free(monkeypatch, 4)

        assert broker.held_bytes(broker.FAMILY_IMAGE) == 0


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

    def test_our_own_server_s_cuda_context_is_not_somebody_else_s_stray(
            self, broker, monkeypatch):
        """A llama-server placed in system RAM holds no weights on the card and
        declares none -- but its process is still there, and a CUDA context is
        hundreds of megabytes before a single weight is loaded. Counting that
        as VRAM nobody admits to sent the user hunting for an orphan process
        that does not exist."""
        monkeypatch.setattr(mc_broker, "total_vram_bytes", lambda: 24 * _GB)
        monkeypatch.setattr(mc_broker, "free_vram_bytes", lambda: 22.5 * _GB)
        broker.register_reclaimer(broker.FAMILY_LLM, Server(running=True))

        assert mc_broker.unaccounted_bytes() == 0

    def test_a_server_that_is_on_the_card_gets_no_second_allowance(
            self, broker, monkeypatch):
        """A placement on the card is measured as a change in free VRAM, so the
        context is already inside the declared figure. Subtracting it twice
        would hide a gigabyte of real residency."""
        monkeypatch.setattr(mc_broker, "total_vram_bytes", lambda: 24 * _GB)
        monkeypatch.setattr(mc_broker, "free_vram_bytes", lambda: 4 * _GB)
        broker.register_reclaimer(broker.FAMILY_LLM, Server(running=True, holds=5 * _GB))

        assert round(mc_broker.unaccounted_bytes() / _GB) == 14

    def test_a_stray_is_blamed_on_our_own_server_only_when_we_have_one(
            self, broker, monkeypatch):
        monkeypatch.setattr(mc_broker, "total_vram_bytes", lambda: 24 * _GB)
        monkeypatch.setattr(mc_broker, "free_vram_bytes", lambda: 4 * _GB)

        assert "previous session" in mc_broker.stray_explanation()

        broker.register_reclaimer(broker.FAMILY_LLM, Server(running=True))

        explained = mc_broker.stray_explanation()
        assert "previous session" not in explained
        assert "Unload" in explained


class TestTheReasonReadsAsASentence:
    """Every message built from a ``reason`` reads it as a noun phrase -- "X is
    short 2 GB", "freed 2 GB for X", "released 2 GB of image VRAM for X". A
    reason written as a clause turns all three into nonsense, which is how "a
    Krea image generation follows is short 18.5 GB" reached a user's console.
    """

    def test_a_request_that_falls_short_names_what_fell_short(self, broker, monkeypatch):
        monkeypatch.setattr(mc_broker, "total_vram_bytes", lambda: 24 * _GB)
        monkeypatch.setattr(mc_broker, "free_vram_bytes", lambda: 4 * _GB)

        mc_broker.request_vram(mc_broker.FAMILY_IMAGE, 18 * _GB,
                               reason="the image generation that follows a Krea roll")

        said = [entry.text for entry in mc_broker.decisions()]
        assert any(text.startswith("the image generation that follows a Krea roll is short")
                   for text in said), said

    def test_a_request_that_did_not_say_why_still_reads_as_a_sentence(
            self, broker, monkeypatch):
        monkeypatch.setattr(mc_broker, "total_vram_bytes", lambda: 24 * _GB)
        monkeypatch.setattr(mc_broker, "free_vram_bytes", lambda: 4 * _GB)

        mc_broker.request_vram(mc_broker.FAMILY_LLM, 18 * _GB)

        said = [entry.text for entry in mc_broker.decisions()]
        assert any(text.startswith("the LLM workload is short") for text in said), said

    def test_the_reason_an_exclusive_sweep_gives_is_a_noun_phrase(self, broker, monkeypatch):
        monkeypatch.setattr(mc_broker, "mode", lambda: mc_broker.MODE_EXCLUSIVE)
        monkeypatch.setattr(mc_broker, "total_vram_bytes", lambda: 24 * _GB)
        set_free(monkeypatch, 4)
        llm = Recorder(holds=6 * _GB, label="the LLM")
        broker.register_reclaimer(broker.FAMILY_LLM, llm)

        mc_broker.request_vram(mc_broker.FAMILY_IMAGE, 8 * _GB)

        assert llm.calls[0][1] == "the image workload taking VRAM ownership"
