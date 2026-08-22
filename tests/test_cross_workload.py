"""The two halves meeting: image residency and the LLM, on one card.

Everything else tests one side. This tests the seams -- that ``mc_memory`` can
reach the LLM when Forge's own eviction has fallen short, that the broker can
reach the image side when the LLM needs room, and that neither can reach the
other when nothing actually needs the memory.
"""

from __future__ import annotations

import pytest

import mc_broker
import mc_memory

_GB = 1024**3


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
        return "the LLM"


@pytest.fixture
def wired(host, monkeypatch):
    """Both families registered, an empty register, no reserve."""
    mc_broker.clear()
    monkeypatch.setattr(mc_broker, "safety_margin_bytes", lambda: 0)
    llm = Recorder(holds=8 * _GB)
    mc_broker.register_reclaimer(mc_broker.FAMILY_LLM, llm)
    yield llm
    mc_broker.clear()
    mc_broker.unregister_reclaimer(mc_broker.FAMILY_LLM)


def set_free(monkeypatch, gigabytes):
    value = int(gigabytes * _GB)
    monkeypatch.setattr(mc_memory, "free_vram_bytes", lambda: value)
    monkeypatch.setattr(mc_broker, "free_vram_bytes", lambda: value)


class TestTheHook:
    def test_the_broker_installs_itself_into_mc_memory_at_import(self):
        """Without this the LLM would never give VRAM back for an image pass,
        and nothing else in the system would notice."""
        assert mc_memory._foreign_reclaim is not None

    def test_mc_memory_works_with_no_hook_installed(self, monkeypatch):
        """The image half has to stay importable, testable and correct on an
        installation that never loads the LLM half."""
        monkeypatch.setattr(mc_memory, "_foreign_reclaim", None)

        assert mc_memory._reclaim_foreign(4 * _GB, "a pass") == 0

    def test_a_hook_that_raises_costs_vram_not_the_generation(self, monkeypatch):
        def broken(needed, reason):
            raise RuntimeError("the runtime is wedged")

        monkeypatch.setattr(mc_memory, "_foreign_reclaim", broken)

        assert mc_memory._reclaim_foreign(4 * _GB, "a pass") == 0

    def test_the_hook_asks_for_the_whole_shortfall(self, wired, monkeypatch):
        """mc_memory passes a deficit; request_vram takes a requirement and
        subtracts free VRAM itself. Getting the conversion wrong evicts
        something and still leaves the pass short, which is the quiet kind of
        wrong -- so it is asserted in bytes."""
        set_free(monkeypatch, 2)

        mc_broker._reclaim_for_image(3 * _GB, "the Stage 2 pass")

        assert wired.calls[0][0] == 3 * _GB

    def test_a_zero_shortfall_asks_for_nothing(self, wired, monkeypatch):
        set_free(monkeypatch, 2)

        assert mc_broker._reclaim_for_image(0, "the Stage 2 pass") == 0
        assert wired.calls == []


class TestImagePassNeedsRoom:
    def test_forge_gets_the_first_chance_and_the_llm_the_second(self, wired, host,
                                                                monkeypatch):
        """Moving an image model to RAM is cheap and keeps it warm; ending a
        llama-server process is neither. So the LLM is the second answer."""
        from backend import memory_management

        memory_management.freed.clear()
        set_free(monkeypatch, 1)
        monkeypatch.setattr(mc_memory, "_pass_requirement", lambda *a, **k: 6 * _GB)
        monkeypatch.setattr(mc_memory, "_loaded_target_patchers", lambda name: [])
        monkeypatch.setattr(mc_memory, "_resident_bytes", lambda patchers: 0)

        mc_memory.make_vram_room("Model B.safetensors")

        assert memory_management.freed, "Forge's own eviction was skipped"
        assert wired.calls, "the LLM was never asked once Forge fell short"

    def test_the_llm_is_left_alone_when_forge_freed_enough(self, wired, host, monkeypatch):
        from backend import memory_management

        memory_management.freed.clear()
        free = {"value": 1 * _GB}
        monkeypatch.setattr(mc_memory, "free_vram_bytes", lambda: free["value"])
        monkeypatch.setattr(mc_broker, "free_vram_bytes", lambda: free["value"])
        monkeypatch.setattr(mc_memory, "_pass_requirement", lambda *a, **k: 6 * _GB)
        monkeypatch.setattr(mc_memory, "_loaded_target_patchers", lambda name: [])
        monkeypatch.setattr(mc_memory, "_resident_bytes", lambda patchers: 0)

        def free_memory(required, device, keep_loaded=()):
            memory_management.freed.append(required)
            free["value"] = 8 * _GB
            return []

        monkeypatch.setattr(memory_management, "free_memory", free_memory)

        mc_memory.make_vram_room("Model B.safetensors")

        assert memory_management.freed
        assert wired.calls == []


class TestGenerationStart:
    def test_hybrid_moves_nothing_merely_because_a_generation_started(self, wired,
                                                                      monkeypatch):
        """Section 8, stated as a sentence and tested as one."""
        set_free(monkeypatch, 4)

        mc_broker.request_vram(mc_broker.FAMILY_IMAGE, 0,
                               reason="an image generation started", margin=0)

        assert wired.calls == []

    def test_exclusive_hands_ownership_over_at_the_start(self, wired, host, monkeypatch):
        """Section 10: ownership is a promise, not an optimisation, so it does
        not wait for the generation to turn out to be a large one."""
        host.shared.opts.set(mc_broker.OPT_MODE, mc_broker.MODE_EXCLUSIVE)
        set_free(monkeypatch, 20)
        mc_broker.declare(mc_broker.FAMILY_LLM, "llm", "the LLM", 8 * _GB)

        mc_broker.request_vram(mc_broker.FAMILY_IMAGE, 0,
                               reason="an image generation started", margin=0)

        assert wired.calls

    def test_exclusive_with_no_llm_running_does_nothing(self, host, monkeypatch):
        """A sweep with nothing to sweep must not be reported as one."""
        mc_broker.clear()
        host.shared.opts.set(mc_broker.OPT_MODE, mc_broker.MODE_EXCLUSIVE)
        set_free(monkeypatch, 20)
        idle = Recorder(holds=0)
        mc_broker.register_reclaimer(mc_broker.FAMILY_LLM, idle)
        try:
            result = mc_broker.request_vram(mc_broker.FAMILY_IMAGE, 0, margin=0)

            assert idle.calls == []
            assert not result.moved_anything
        finally:
            mc_broker.unregister_reclaimer(mc_broker.FAMILY_LLM)


class TestLlmNeedsRoom:
    def test_the_image_reclaimer_goes_through_mc_memory(self, host, monkeypatch):
        """Section 8 and 17: the broker decides, mc_memory moves. A broker that
        moved tensors itself would be a second memory manager."""
        called = {}

        def release_vram(needed_bytes, reason=""):
            called["needed"] = needed_bytes
            called["reason"] = reason
            return needed_bytes

        monkeypatch.setattr(mc_memory, "release_vram", release_vram)
        reclaimer = mc_broker._ImageReclaimer()

        assert reclaimer.release(3 * _GB, "an LLM request") == 3 * _GB
        assert called["needed"] == 3 * _GB

    def test_release_vram_moves_weights_rather_than_discarding_them(self, host, monkeypatch):
        """free_memory moves to the offload device, so a checkpoint demoted for
        an LLM stays in the RAM cache and coming back is still a warm swap."""
        from backend import memory_management

        memory_management.freed.clear()
        free = {"value": 1 * _GB}
        monkeypatch.setattr(mc_memory, "free_vram_bytes", lambda: free["value"])

        def free_memory(required, device, keep_loaded=()):
            memory_management.freed.append(required)
            free["value"] = 6 * _GB
            return []

        monkeypatch.setattr(memory_management, "free_memory", free_memory)

        freed = mc_memory.release_vram(5 * _GB, reason="an LLM request")

        assert memory_management.freed == [5 * _GB]
        assert freed == 5 * _GB

    def test_release_vram_does_nothing_when_there_is_already_room(self, host, monkeypatch):
        from backend import memory_management

        memory_management.freed.clear()
        monkeypatch.setattr(mc_memory, "free_vram_bytes", lambda: 10 * _GB)

        assert mc_memory.release_vram(4 * _GB) == 0
        assert memory_management.freed == []

    def test_release_vram_leaves_the_host_alone_when_vram_cannot_be_read(self, host,
                                                                         monkeypatch):
        from backend import memory_management

        memory_management.freed.clear()
        monkeypatch.setattr(mc_memory, "free_vram_bytes", lambda: 0)

        assert mc_memory.release_vram(4 * _GB) == 0
        assert memory_management.freed == []


class TestWhatCountsAsResident:
    def test_resident_means_on_the_card_not_merely_loaded(self, host, monkeypatch):
        """A checkpoint offloaded to system RAM still has a model_size(); it
        does not still have VRAM. Confusing the two makes Exclusive mode sweep
        a family that already left, every single request."""
        from backend import memory_management

        class Entry:
            def __init__(self, held):
                self._held = held

            def model_loaded_memory(self):
                return self._held

        monkeypatch.setattr(memory_management, "current_loaded_models",
                            [Entry(6 * _GB), Entry(0)], raising=False)

        assert mc_memory.resident_vram_bytes() == 6 * _GB

    def test_an_empty_registry_reads_as_nothing_resident(self, host, monkeypatch):
        from backend import memory_management

        monkeypatch.setattr(memory_management, "current_loaded_models", [], raising=False)

        assert mc_memory.resident_vram_bytes() == 0
        assert mc_broker._ImageReclaimer().resident_bytes() == 0

    def test_a_sweep_with_nothing_on_the_card_asks_for_nothing(self, host, monkeypatch):
        """The follow-on: with residency measured properly, a second LLM
        request in Exclusive mode does not re-sweep an image family that has
        already gone."""
        from backend import memory_management

        mc_broker.clear()
        monkeypatch.setattr(memory_management, "current_loaded_models", [], raising=False)
        monkeypatch.setattr(mc_broker, "safety_margin_bytes", lambda: 0)
        set_free(monkeypatch, 2)
        host.shared.opts.set(mc_broker.OPT_MODE, mc_broker.MODE_EXCLUSIVE)

        image = Recorder(holds=0)
        mc_broker.register_reclaimer(mc_broker.FAMILY_IMAGE, image)
        try:
            mc_broker.request_vram(mc_broker.FAMILY_LLM, 1 * _GB)

            assert image.calls == []
        finally:
            mc_broker.unregister_reclaimer(mc_broker.FAMILY_IMAGE)
            mc_broker.register_reclaimer(mc_broker.FAMILY_IMAGE, mc_broker._ImageReclaimer())


class TestTheCheckpointStaysOnTheCard:
    """The cycle a user reported, and the seam that produced it.

    Their console, on a 24 GB card with a 13.9 GB checkpoint loaded, said the
    same three things on every press of Generate:

        released 13.9 GB of image VRAM for the LLM taking VRAM ownership
        Moving model(s) has taken 5.92 seconds
        Moving model(s) has taken 8.07 seconds

    The language model swept the checkpoint off the card, reserved room for it
    again, placed itself in what was left, and the generation two seconds later
    paid thirteen seconds to move the very same weights back. Both halves of
    that are tested here, because either one alone would have hidden it.
    """

    def test_a_krea_roll_does_not_sweep_a_resident_checkpoint(self, host, monkeypatch):
        mc_broker.clear()
        monkeypatch.setattr(mc_broker, "safety_margin_bytes", lambda: 0)
        set_free(monkeypatch, 8)
        host.shared.opts.set(mc_broker.OPT_MODE, mc_broker.MODE_EXCLUSIVE)

        image = Recorder(holds=14 * _GB)
        mc_broker.register_reclaimer(mc_broker.FAMILY_IMAGE, image)
        try:
            mc_broker.request_vram(mc_broker.FAMILY_LLM, 12 * _GB,
                                   reason="a Krea prompt")

            assert image.calls == []
        finally:
            mc_broker.unregister_reclaimer(mc_broker.FAMILY_IMAGE)
            mc_broker.register_reclaimer(mc_broker.FAMILY_IMAGE, mc_broker._ImageReclaimer())

    def test_the_roll_reserves_nothing_for_a_checkpoint_already_on_the_card(
            self, host, monkeypatch):
        """The other half. The reserve is sized from what the image family
        holds, and the register does not know about checkpoints -- so it has to
        be asked, not read."""
        import mc_creative_krea

        mc_broker.clear()
        monkeypatch.setattr(mc_memory, "vram_required_bytes", lambda name, *a, **k: 14 * _GB)
        monkeypatch.setattr(mc_broker, "safety_margin_bytes", lambda: 0)
        host.shared.opts.sd_model_checkpoint = "kroma.safetensors"

        image = Recorder(holds=14 * _GB)
        mc_broker.register_reclaimer(mc_broker.FAMILY_IMAGE, image)
        try:
            assert mc_broker.resident_bytes(mc_broker.FAMILY_IMAGE) == 0
            assert mc_creative_krea.image_reserve_bytes() == 0

            # And the same call still reserves the full amount when the card is
            # genuinely empty -- otherwise this would pass on a function that
            # had simply stopped working.
            image.holds = 0
            assert mc_creative_krea.image_reserve_bytes() == 14 * _GB
        finally:
            mc_broker.unregister_reclaimer(mc_broker.FAMILY_IMAGE)
            mc_broker.register_reclaimer(mc_broker.FAMILY_IMAGE, mc_broker._ImageReclaimer())

    def test_handing_vram_back_afterwards_is_a_no_op_when_it_never_left(
            self, host, monkeypatch):
        """And so llama-server is not stopped and restarted for every image."""
        import mc_creative_krea

        mc_broker.clear()
        monkeypatch.setattr(mc_memory, "vram_required_bytes", lambda name, *a, **k: 14 * _GB)
        host.shared.opts.sd_model_checkpoint = "kroma.safetensors"
        host.shared.opts.set(mc_broker.OPT_MODE, mc_broker.MODE_EXCLUSIVE)

        image = Recorder(holds=14 * _GB)
        llm = Recorder(holds=6 * _GB)
        mc_broker.register_reclaimer(mc_broker.FAMILY_IMAGE, image)
        mc_broker.register_reclaimer(mc_broker.FAMILY_LLM, llm)
        try:
            assert mc_creative_krea.hand_back_vram() == 0
            assert llm.calls == []
        finally:
            mc_broker.unregister_reclaimer(mc_broker.FAMILY_LLM)
            mc_broker.unregister_reclaimer(mc_broker.FAMILY_IMAGE)
            mc_broker.register_reclaimer(mc_broker.FAMILY_IMAGE, mc_broker._ImageReclaimer())


class TestAReserveMissIsRecordedNotJustSurvived:
    """Section 17, and the sentence at the end of it: emergency eviction is
    recovery, not ordinary scheduling.

    Reaching the cross-workload reclaim at all means the plan said this phase
    would fit inside the protected image budget and it did not. That is worth
    recording whether or not the recovery worked, because a recovery nobody is
    told about is indistinguishable from the policy working -- the user sees a
    language model that got slower and has nothing to connect it to.
    """

    @pytest.fixture
    def short(self, wired, host, monkeypatch):
        """A pass that does not fit, and a Forge eviction that cannot close it."""
        import mc_plan
        from backend import memory_management

        memory_management.freed.clear()
        set_free(monkeypatch, 1)
        monkeypatch.setattr(mc_memory, "_pass_requirement", lambda *a, **k: 6 * _GB)
        monkeypatch.setattr(mc_memory, "_loaded_target_patchers", lambda name: [])
        monkeypatch.setattr(mc_memory, "_resident_bytes", lambda patchers: 0)
        monkeypatch.setattr(mc_plan, "usable_vram_bytes", lambda: 24 * _GB)
        mc_plan.publish(mc_plan.Plan((
            mc_plan.Phase(mc_plan.STAGE_1, mc_plan.KIND_IMAGE, "Stage 1", 14 * _GB),
        ), 1024, 1024))
        return wired

    def test_the_miss_is_filed_with_the_phase_that_overran(self, short):
        import mc_plan

        mc_memory.make_vram_room("Model B.safetensors", stage="Stage 2")

        assert mc_plan.last_miss() is not None
        assert mc_plan.last_miss().phase == "Stage 2"

    def test_the_shortfall_is_the_amount_the_estimate_was_out_by(self, short):
        import mc_plan

        mc_memory.make_vram_room("Model B.safetensors", stage="Stage 2")

        assert mc_plan.last_miss().shortfall_bytes > 0

    def test_the_llm_residency_is_read_before_it_is_taken(self, short):
        """After the eviction the answer is zero for every miss, and the panel
        could not tell a model that gave up eight gigabytes from one that was
        never on the card."""
        import mc_plan

        mc_broker.declare(mc_broker.FAMILY_LLM, "llm", "the LLM", 8 * _GB)

        mc_memory.make_vram_room("Model B.safetensors", stage="Stage 2")

        assert mc_plan.last_miss().llm_bytes == 8 * _GB

    def test_the_eviction_is_recorded_as_having_happened(self, short):
        import mc_plan

        mc_memory.make_vram_room("Model B.safetensors", stage="Stage 2")

        assert mc_plan.last_miss().evicted

    def test_a_safer_cap_is_learned_and_not_promoted_back(self, short):
        import mc_plan

        mc_broker.declare(mc_broker.FAMILY_LLM, "llm", "the LLM", 8 * _GB)

        mc_memory.make_vram_room("Model B.safetensors", stage="Stage 2")

        assert 0 < mc_plan.learned_cap_bytes() < 8 * _GB

    def test_a_pass_that_fits_files_nothing(self, wired, host, monkeypatch):
        """A miss log that fires on healthy generations is a miss log nobody
        reads by the third one."""
        import mc_plan

        set_free(monkeypatch, 20)
        monkeypatch.setattr(mc_memory, "_pass_requirement", lambda *a, **k: 6 * _GB)
        monkeypatch.setattr(mc_memory, "_loaded_target_patchers", lambda name: [])
        monkeypatch.setattr(mc_memory, "_resident_bytes", lambda patchers: 0)

        mc_memory.make_vram_room("Model B.safetensors", stage="Stage 2")

        assert mc_plan.last_miss() is None

    def test_bookkeeping_that_fails_does_not_fail_the_generation(
            self, short, monkeypatch):
        import mc_plan

        def explode(*args, **kwargs):
            raise RuntimeError("no")

        monkeypatch.setattr(mc_plan, "record_miss", explode)

        mc_memory.make_vram_room("Model B.safetensors", stage="Stage 2")
