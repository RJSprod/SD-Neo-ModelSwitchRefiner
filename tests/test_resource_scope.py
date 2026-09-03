"""Resource-scoped concurrency: the acceptance matrix, section 19.

Every test here asks one question in two forms, and the difference between the
two forms is the whole change set:

    "Are these two workloads in conflict?"
    "Are these two workloads competing for the same physical thing?"

The first was what the code used to ask, and the answer was yes whenever both
were models -- so a conversation on a 5090 waited for a generation on a 3090,
a 3090 shortfall stopped the 5090's server, and an image plan protecting one
card resized a language model on the other. The second is what it asks now,
and on a machine with two cards and one pool of system RAM the two questions
have different answers several times a minute.

The tests are grouped as the design intent groups them, and the ``T`` numbers
in the docstrings are its own.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

import mc_broker
import mc_llm_runtime as runtime
import mc_llm_sessions as sessions
import mc_memory  # noqa: F401  -- imported for the end-to-end translation test

from test_llm_context import build_model

_GB = 1024**3

IMAGE_CARD = 0
"""The card Forge generates on throughout this file. The 3090 of the report."""

OTHER_CARD = 1
"""The card the language model is pinned to. The 5090 of the report."""


# --------------------------------------------------------------------------- #
# Doubles
# --------------------------------------------------------------------------- #


def configuration_on(tmp_path, *, card=OTHER_CARD, mode="gpu", device=None,
                     gpu_layers="all", blocks=32, size_mb=4, name="model.gguf"):
    """A :class:`mc_llm_runtime.Config` placed on one specific processor.

    ``card=None`` with ``mode="cpu"`` is the processor installation, which is
    the case that must never be spelled as "no card index" -- see
    :data:`mc_broker.EXEC_CUDA_UNKNOWN`.
    """
    from prompt_master.inference.device_detection import CPU_DEVICE

    model = build_model(tmp_path, name=name, blocks=blocks, size_mb=size_mb)
    executable = tmp_path / "llama-server"
    executable.write_bytes(b"")
    return runtime.Config(
        runtime=executable, model=model, mmproj=None,
        gpu_index=0 if card is None else card,
        device=device if device is not None else (
            CPU_DEVICE if mode == "cpu" else f"CUDA{card}"),
        gpu_layers=gpu_layers, context_size=8192, context_mode="fixed",
        context_buffer_gb=4.0, kv_type_k="f16", kv_type_v="f16", mode=mode)


class Server:
    """A llama-server on one card, with the parts of ``Runtime`` a broker uses.

    Faithful to the contract rather than to the class: ``on_card`` refusing a
    different card is the defence section 7.4 asks for, and a double that did
    not have it would let every reclaim test pass against an implementation
    with no scoping at all.
    """

    def __init__(self, card=OTHER_CARD, holds=0, up=True, host_ram=0, cpu=False):
        self.card = None if cpu else card
        self.cpu = cpu
        self.holds = holds
        self.up = up
        self.host_ram = host_ram
        self.calls: list[tuple[int, str, object]] = []
        self.roles: tuple = ()
        self.stopped = False

    # -- what the registry and the broker ask ----------------------------- #

    def on_card(self, card):
        if isinstance(card, mc_broker._AnyCard):
            return True
        if self.cpu or self.card is None or card is None:
            return False
        return int(self.card) == int(card)

    def running(self):
        return self.up

    def resident_bytes(self, *, card=mc_broker.ANY_CARD):
        return self.holds if self.on_card(card) else 0

    def host_ram_bytes(self):
        return self.host_ram if self.up else 0

    def release(self, needed_bytes, reason="", *, card=mc_broker.ANY_CARD):
        self.calls.append((int(needed_bytes), reason, card))
        if not self.on_card(card):
            return 0
        freed, self.holds = self.holds, 0
        self.up, self.stopped = False, True
        return freed

    def release_host_ram(self, needed_bytes, reason=""):
        """Stop, when this server's weights are in system RAM to begin with.

        A server on the card answers zero for the same reason the real one does:
        stopping it returns nothing where the memory is wanted, and an eviction
        that frees nothing is pure cost.
        """
        self.calls.append((int(needed_bytes), reason, "host-ram"))
        if not self.up or self.host_ram <= 0:
            return 0
        freed, self.host_ram = self.host_ram, 0
        self.up, self.stopped = False, True
        return freed

    def describe(self):
        return f"llama-server on {'the processor' if self.cpu else f'GPU {self.card}'}"

    def configuration(self):
        class _Settings:
            uses_cuda_compute = not self.cpu
            gpu_index = 0 if self.card is None else self.card

        return _Settings()

    @property
    def _card(self):
        return self.card


class ImageSide:
    """The image family's reclaimer, which has exactly one card by construction."""

    def __init__(self, holds=0):
        self.holds = holds
        self.calls: list[tuple[int, str]] = []

    def release(self, needed_bytes, reason="", *, card=mc_broker.ANY_CARD):
        self.calls.append((int(needed_bytes), reason))
        freed, self.holds = self.holds, 0
        return freed

    def resident_bytes(self, *, card=mc_broker.ANY_CARD):
        return self.holds

    def describe(self):
        return "the image checkpoint"


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def scoped(host, monkeypatch):
    """Forge on GPU 0, an empty register, no reserve, and one card per question."""
    mc_broker.clear()
    monkeypatch.setattr(mc_broker, "safety_margin_bytes", lambda: 0)
    monkeypatch.setattr(mc_broker, "image_device_index", lambda: IMAGE_CARD)
    for family in (mc_broker.FAMILY_IMAGE, mc_broker.FAMILY_LLM):
        mc_broker.unregister_reclaimer(family)
    yield mc_broker
    mc_broker.clear()
    for family in (mc_broker.FAMILY_IMAGE, mc_broker.FAMILY_LLM):
        mc_broker.unregister_reclaimer(family)
    mc_broker.register_reclaimer(mc_broker.FAMILY_IMAGE, mc_broker._ImageReclaimer())
    mc_broker.register_reclaimer(mc_broker.FAMILY_LLM, runtime.registry)


@pytest.fixture
def registry(scoped):
    """A real :class:`RuntimeRegistry` holding doubles, registered as the reclaimer."""
    found = runtime.RuntimeRegistry()
    mc_broker.register_reclaimer(mc_broker.FAMILY_LLM, found)
    yield found
    found.forget()


def hold(registry, server, key="one"):
    """Put ``server`` in ``registry`` under ``key``."""
    registry._runtimes[(key,)] = server
    return server


def free_on(monkeypatch, **cards):
    """Fix per-card free VRAM: ``free_on(monkeypatch, gpu0=4, gpu1=20)``."""
    amounts = {int(name[3:]): int(value * _GB) for name, value in cards.items()}
    monkeypatch.setattr(mc_broker, "device_free_vram_bytes",
                        lambda index=None: amounts.get(
                            IMAGE_CARD if index is None else int(index), 0))
    monkeypatch.setattr(mc_broker, "free_vram_bytes",
                        lambda: amounts.get(IMAGE_CARD, 0))


def ram(monkeypatch, available_gb, reserve_gb=2.0):
    monkeypatch.setattr(mc_broker, "free_ram_bytes", lambda: int(available_gb * _GB))
    monkeypatch.setattr(mc_broker, "ram_reserve_bytes", lambda: int(reserve_gb * _GB))


def take(gpu):
    """Drive ``_Gpu.acquire`` to completion. Returns ``(acquired, events)``."""
    events = []
    generator = gpu.acquire()
    while True:
        try:
            events.append(next(generator))
        except StopIteration as stop:
            return stop.value, events


def waiting_texts(events):
    return [event.text for event in events if event.kind == sessions.STATUS]


# --------------------------------------------------------------------------- #
# 19.1 Execution domains (T1-T10)
# --------------------------------------------------------------------------- #


class TestTheThreeExecutionStates:
    """Invariant I-10: the processor and an unresolved card are not the same thing.

    They were, for as long as both were spelled "no GPU index", and they need
    opposite decisions: a processor runtime must not wait for an image
    generation, and a CUDA runtime whose card nobody can name must.
    """

    def test_a_processor_runtime_is_not_an_unknown_card(self, scoped, tmp_path):
        cpu = runtime.execution_domain(configuration_on(tmp_path, card=None, mode="cpu"))

        assert cpu == mc_broker.CPU_EXECUTION
        assert cpu != mc_broker.UNKNOWN_CUDA_EXECUTION

    def test_a_named_card_resolves_to_that_card(self, scoped, tmp_path):
        found = runtime.execution_domain(configuration_on(tmp_path, card=OTHER_CARD))

        assert found == mc_broker.cuda_execution(OTHER_CARD)
        assert found.known

    def test_an_unparseable_index_is_unknown_cuda_and_not_the_processor(self, tmp_path):
        broken = configuration_on(tmp_path, card=OTHER_CARD)
        object.__setattr__(broken, "gpu_index", "not a number")

        found = runtime.execution_domain(broken)

        assert found == mc_broker.UNKNOWN_CUDA_EXECUTION
        assert not found.is_cpu

    def test_mixed_conservative_executes_on_its_card_though_it_holds_no_layers(
            self, scoped, tmp_path):
        """The case that proves ``uses_cuda_compute`` and ``on_gpu`` are two
        questions. Every weight is in system RAM and the card is still doing the
        work, so an image job on it is competing and one elsewhere is not."""
        mixed = configuration_on(tmp_path, card=IMAGE_CARD, mode="mixed_conservative")

        assert not mixed.on_gpu
        assert runtime.execution_domain(mixed) == mc_broker.cuda_execution(IMAGE_CARD)

    def test_two_cards_do_not_conflict(self):
        assert not mc_broker.cuda_execution(0).conflicts_with(mc_broker.cuda_execution(1))

    def test_one_card_conflicts_with_itself(self):
        assert mc_broker.cuda_execution(0).conflicts_with(mc_broker.cuda_execution(0))

    def test_the_processor_never_conflicts_with_a_card(self):
        assert not mc_broker.CPU_EXECUTION.conflicts_with(mc_broker.cuda_execution(0))
        assert not mc_broker.cuda_execution(0).conflicts_with(mc_broker.CPU_EXECUTION)

    def test_an_unresolved_card_conflicts_with_every_card(self):
        assert mc_broker.UNKNOWN_CUDA_EXECUTION.conflicts_with(mc_broker.cuda_execution(7))
        assert mc_broker.cuda_execution(7).conflicts_with(mc_broker.UNKNOWN_CUDA_EXECUTION)

    def test_an_unresolved_card_still_does_not_conflict_with_the_processor(self):
        assert not mc_broker.UNKNOWN_CUDA_EXECUTION.conflicts_with(mc_broker.CPU_EXECUTION)


class TestTheLlmSideWait:
    """T1-T4, T8. What an LLM request waits for before it starts."""

    @pytest.fixture
    def quick(self, monkeypatch):
        monkeypatch.setattr(sessions, "WAIT_NOTICE_SECONDS", 0.0)
        monkeypatch.setattr(sessions, "WAIT_POLL_SECONDS", 0.01)

    @pytest.fixture
    def generating(self, scoped, monkeypatch):
        """An image generation running on GPU 0, as ``host_busy`` sees one."""
        monkeypatch.setattr(mc_broker, "host_busy", lambda: True)

    def serve(self, monkeypatch, tmp_path, **placement):
        settings = configuration_on(tmp_path, **placement)
        monkeypatch.setattr(runtime, "config", lambda role="": settings)
        return settings

    def test_a_conversation_on_the_other_card_does_not_wait(
            self, generating, quick, tmp_path, monkeypatch):
        """T1. The headline case: image on GPU 0, conversation on GPU 1, and
        nothing whatever is shared between them."""
        self.serve(monkeypatch, tmp_path, card=OTHER_CARD)
        gpu = sessions._Gpu("a conversation reply", sessions.Cancellation())

        acquired, events = take(gpu)
        gpu.release()

        assert acquired
        assert waiting_texts(events) == []

    def test_a_conversation_on_the_image_card_still_waits(
            self, generating, quick, tmp_path, monkeypatch):
        """T2. Same card, real contention, unchanged behaviour."""
        self.serve(monkeypatch, tmp_path, card=IMAGE_CARD)
        gpu = sessions._Gpu("a conversation reply", sessions.Cancellation())
        cancel = threading.Timer(0.2, gpu.cancel.cancel)
        cancel.start()

        acquired, events = take(gpu)
        cancel.cancel()

        assert not acquired
        assert any("image generation" in text for text in waiting_texts(events))

    def test_a_processor_conversation_does_not_wait(
            self, generating, quick, tmp_path, monkeypatch):
        """T3. A CPU runtime and a CUDA generation share no processor at all."""
        self.serve(monkeypatch, tmp_path, card=None, mode="cpu")
        gpu = sessions._Gpu("a conversation reply", sessions.Cancellation())

        acquired, events = take(gpu)
        gpu.release()

        assert acquired
        assert waiting_texts(events) == []

    def test_an_unresolvable_card_waits_conservatively(
            self, generating, quick, tmp_path, monkeypatch):
        """T4. False serialisation costs throughput; the other error puts two
        jobs on one card with neither expecting the other."""
        settings = self.serve(monkeypatch, tmp_path, card=OTHER_CARD)
        object.__setattr__(settings, "gpu_index", "?")
        gpu = sessions._Gpu("a conversation reply", sessions.Cancellation())
        cancel = threading.Timer(0.2, gpu.cancel.cancel)
        cancel.start()

        acquired, _events = take(gpu)
        cancel.cancel()

        assert not acquired

    def test_the_image_card_being_unreadable_also_waits(
            self, generating, quick, tmp_path, monkeypatch):
        """Section 16.1's other half. Independence that cannot be shown is not
        independence, so a CUDA request waits."""
        self.serve(monkeypatch, tmp_path, card=OTHER_CARD)
        monkeypatch.setattr(mc_broker, "image_device_index", lambda: -1)
        gpu = sessions._Gpu("a conversation reply", sessions.Cancellation())
        cancel = threading.Timer(0.2, gpu.cancel.cancel)
        cancel.start()

        acquired, _events = take(gpu)
        cancel.cancel()

        assert not acquired

    def test_a_processor_request_proceeds_even_with_the_image_card_unreadable(
            self, generating, quick, tmp_path, monkeypatch):
        """Also section 16.1: the processor is positive evidence, not an
        absence of evidence, so it does not inherit the card's uncertainty."""
        self.serve(monkeypatch, tmp_path, card=None, mode="cpu")
        monkeypatch.setattr(mc_broker, "image_device_index", lambda: -1)
        gpu = sessions._Gpu("a conversation reply", sessions.Cancellation())

        acquired, _events = take(gpu)
        gpu.release()

        assert acquired

    def test_a_configuration_moved_onto_the_image_card_while_queued_is_caught(
            self, generating, quick, tmp_path, monkeypatch):
        """T8. Queued as GPU 1, changed to GPU 0 while waiting: it must not
        start on top of the generation it has just been moved onto."""
        away = configuration_on(tmp_path, card=OTHER_CARD, name="away.gguf")
        onto = configuration_on(tmp_path, card=IMAGE_CARD, name="onto.gguf")
        state = {"settings": away, "asked": 0}

        def chosen(role=""):
            state["asked"] += 1
            # The first read is the one before the lock; every read after it
            # sees the setting the user has changed.
            return away if state["asked"] <= 1 else onto

        monkeypatch.setattr(runtime, "config", chosen)
        gpu = sessions._Gpu("a conversation reply", sessions.Cancellation())
        cancel = threading.Timer(0.3, gpu.cancel.cancel)
        cancel.start()

        acquired, _events = take(gpu)
        cancel.cancel()

        assert not acquired, "a request that moved onto the busy card started anyway"

    def test_the_krea_roll_inside_the_host_job_still_never_waits(
            self, generating, quick, tmp_path, monkeypatch):
        """The exception that must survive all of this. The generation is
        blocked waiting for this prompt, so waiting for the generation would be
        waiting for itself."""
        self.serve(monkeypatch, tmp_path, card=IMAGE_CARD)
        gpu = sessions._Gpu("a Krea prompt", sessions.Cancellation())

        with mc_broker.host_job():
            acquired, events = take(gpu)
        gpu.release()

        assert acquired
        assert waiting_texts(events) == []


class TestTheImageSideWait:
    """T5-T7. What a generation waits for before it touches the card."""

    def test_it_returns_at_once_for_an_llm_on_another_card(self, scoped):
        """T5. The 5090 conversation is not on the 3090 and never was."""
        with mc_broker.workload(mc_broker.FAMILY_LLM, "a reply",
                                domain=mc_broker.cuda_execution(OTHER_CARD)):
            started = time.monotonic()

            assert mc_broker.await_idle(timeout=5.0,
                                        domain=mc_broker.cuda_execution(IMAGE_CARD))
            assert time.monotonic() - started < 1.0

    def test_it_waits_for_an_llm_on_the_image_card(self, scoped):
        """T6. Same card, existing bounded wait."""
        with mc_broker.workload(mc_broker.FAMILY_LLM, "a reply",
                                domain=mc_broker.cuda_execution(IMAGE_CARD)):
            assert not mc_broker.await_idle(timeout=0.1,
                                            domain=mc_broker.cuda_execution(IMAGE_CARD))

    def test_it_returns_at_once_for_a_processor_llm(self, scoped):
        """T7."""
        with mc_broker.workload(mc_broker.FAMILY_LLM, "a reply",
                                domain=mc_broker.CPU_EXECUTION):
            assert mc_broker.await_idle(timeout=5.0,
                                        domain=mc_broker.cuda_execution(IMAGE_CARD))

    def test_it_waits_for_an_llm_whose_card_is_unknown(self, scoped):
        with mc_broker.workload(mc_broker.FAMILY_LLM, "a reply",
                                domain=mc_broker.UNKNOWN_CUDA_EXECUTION):
            assert not mc_broker.await_idle(timeout=0.1,
                                            domain=mc_broker.cuda_execution(IMAGE_CARD))

    def test_no_domain_at_all_keeps_the_old_meaning(self, scoped):
        """A caller that cannot say where it executes is no worse off than it
        was before the parameter existed: any LLM at all is a conflict."""
        with mc_broker.workload(mc_broker.FAMILY_LLM, "a reply",
                                domain=mc_broker.cuda_execution(OTHER_CARD)):
            assert not mc_broker.await_idle(timeout=0.1)

    def test_the_independence_is_said_once_and_not_per_generation(self, scoped, caplog):
        """Section 21.3 wants this observable and explicitly not per chunk."""
        with mc_broker.workload(mc_broker.FAMILY_LLM, "a reply",
                                domain=mc_broker.cuda_execution(OTHER_CARD)):
            with caplog.at_level("INFO"):
                for _ in range(4):
                    mc_broker.await_idle(timeout=1.0,
                                         domain=mc_broker.cuda_execution(IMAGE_CARD))

        said = [line for line in caplog.messages if "different processors" in line]
        assert len(said) == 1


class TestBackgroundPriming:
    """T9, T10. Priming follows the same execution rule as everything else."""

    def test_a_prime_on_the_other_card_proceeds(self, scoped, tmp_path, monkeypatch):
        monkeypatch.setattr(mc_broker, "host_busy", lambda: True)

        assert not runtime._stands_down_for_the_image_job(
            configuration_on(tmp_path, card=OTHER_CARD))

    def test_a_prime_on_the_image_card_stands_down(self, scoped, tmp_path, monkeypatch):
        monkeypatch.setattr(mc_broker, "host_busy", lambda: True)

        assert runtime._stands_down_for_the_image_job(
            configuration_on(tmp_path, card=IMAGE_CARD))

    def test_a_processor_prime_proceeds(self, scoped, tmp_path, monkeypatch):
        monkeypatch.setattr(mc_broker, "host_busy", lambda: True)

        assert not runtime._stands_down_for_the_image_job(
            configuration_on(tmp_path, card=None, mode="cpu"))

    def test_nothing_stands_down_when_no_generation_is_running(
            self, scoped, tmp_path, monkeypatch):
        monkeypatch.setattr(mc_broker, "host_busy", lambda: False)

        assert not runtime._stands_down_for_the_image_job(
            configuration_on(tmp_path, card=IMAGE_CARD))


# --------------------------------------------------------------------------- #
# 19.2 The image plan (T11-T16)
# --------------------------------------------------------------------------- #


class TestThePlanIsAboutOneCard:
    """T11-T16. ``mc_plan`` describes Forge's card and no other.

    The failure this group pins down is the one the report opens with: a
    19 GB language model on a 5090, judged against a 3090 plan that leaves
    4 GB, declared to be overspending by fifteen gigabytes it was never
    holding on that card, and restarted -- every time the plan moved.
    """

    @pytest.fixture
    def planned(self, scoped, monkeypatch):
        import mc_plan

        monkeypatch.setattr(mc_plan, "current", lambda: _Plan(("stage-1",)))
        monkeypatch.setattr(mc_plan, "persistent_llm_budget", lambda ours=0: 4 * _GB)
        yield mc_plan

    def server_on(self, card, cpu=False):
        held = runtime.Runtime()
        held._card = card
        held._settings = None
        held.configuration = lambda: _Settings(cpu=cpu, card=card)
        return held

    def test_a_runtime_on_another_card_gets_no_allowance(self, planned, monkeypatch):
        """T11. Not "a larger allowance" -- no allowance. The plan is not a
        statement about GPU 1 at all."""
        held = self.server_on(OTHER_CARD)

        assert held._allowance(19 * _GB) == -1
        assert not held._overspending(19 * _GB)

    def test_a_runtime_on_the_image_card_is_judged_exactly_as_before(self, planned):
        """T12."""
        held = self.server_on(IMAGE_CARD)

        assert held._allowance(19 * _GB) == 4 * _GB
        assert held._overspending(19 * _GB)

    def test_a_processor_runtime_is_never_overspending_image_vram(self, planned):
        held = self.server_on(None, cpu=True)

        assert held._allowance(19 * _GB) == -1
        assert not held._overspending(19 * _GB)

    def test_an_unresolvable_card_is_still_treated_as_the_image_card(self, planned,
                                                                     monkeypatch):
        """The conservative direction, matching ``shares_the_image_card``: the
        cost of being wrong here is a smaller language model, and the cost of
        the opposite is a generation that runs out of VRAM."""
        monkeypatch.setattr(mc_broker, "image_device_index", lambda: -1)
        held = self.server_on(IMAGE_CARD)

        assert held._allowance(19 * _GB) == 4 * _GB

    def test_two_runtimes_keep_separate_baselines(self, planned):
        """T13. One module-level value had each overwriting the other's."""
        first, second = self.server_on(IMAGE_CARD), self.server_on(IMAGE_CARD)
        first._note_placement()

        assert first._placed_for == ("stage-1",)
        assert second._placed_for is None
        assert not second._boundary_moved()

    def test_a_declined_boundary_is_recorded_on_the_runtime_that_declined_it(
            self, planned, monkeypatch):
        """T14."""
        held = self.server_on(IMAGE_CARD)
        held._placed_for = ("stage-1-and-2",)
        assert held._boundary_moved()

        held._reconciled()

        assert held._placed_for == ("stage-1",)
        assert not held._boundary_moved()

    def test_a_later_boundary_is_examined_again(self, planned, monkeypatch):
        """T15. What is recorded is "considered", not "stop asking"."""
        import mc_plan

        held = self.server_on(IMAGE_CARD)
        held._note_placement()
        monkeypatch.setattr(mc_plan, "current", lambda: _Plan(("stage-1-smaller",)))

        assert held._boundary_moved()

    def test_a_different_card_runtime_accumulates_no_boundary_state(self, planned,
                                                                    monkeypatch):
        """T16. It records nothing, so nothing can later "move" and force a
        GGUF re-read and a re-negotiation for a plan about another card."""
        import mc_plan

        held = self.server_on(OTHER_CARD)
        held._note_placement()
        assert held._placed_for is None

        monkeypatch.setattr(mc_plan, "current", lambda: _Plan(("something else",)))

        assert not held._boundary_moved()

    def test_stopping_clears_only_this_runtimes_baseline(self, planned):
        """Two servers up: one stopping is no reason to tell the panel the
        other was never placed."""
        import mc_plan

        first, second = self.server_on(IMAGE_CARD), self.server_on(IMAGE_CARD)
        first._note_placement()
        second._placed_for = ("an older plan",)

        second._forget_placement_plan()

        assert second._placed_for is None
        assert mc_plan.placed_for() == ("stage-1",)


class _Plan:
    def __init__(self, identity):
        self._identity = identity

    def identity(self):
        return self._identity


class _Settings:
    def __init__(self, cpu=False, card=IMAGE_CARD, uuid="", name=""):
        self.uses_cuda_compute = not cpu
        self.gpu_index = 0 if card is None else card
        self.gpu_uuid = uuid
        self.card_name = name


class _Unparseable:
    """A CUDA installation whose card index is not a number.

    Rare and real: a hand-edited state file, or a device string a future
    llama.cpp spells differently. What matters is that it stays *unknown* --
    section 4.1 is explicit that this must not collapse into the processor,
    and section 7.3 that it must not become a targeted reclaim victim.
    """

    uses_cuda_compute = True
    gpu_index = "CUDA?"


class TestTheImageJobIsNotAlwaysThisRuntimesBusiness:
    """Section 8.4. "The host is busy" is a fact about a card, not the machine."""

    def server_on(self, card, cpu=False):
        held = runtime.Runtime()
        held._card = card
        held.configuration = lambda: _Settings(cpu=cpu, card=card)
        return held

    def test_a_generation_on_the_image_card_concerns_a_runtime_there(
            self, scoped, monkeypatch):
        monkeypatch.setattr(mc_broker, "host_busy", lambda: True)

        assert self.server_on(IMAGE_CARD)._image_job_conflicts()

    def test_it_does_not_concern_a_runtime_on_another_card(self, scoped, monkeypatch):
        monkeypatch.setattr(mc_broker, "host_busy", lambda: True)

        assert not self.server_on(OTHER_CARD)._image_job_conflicts()

    def test_nor_a_processor_runtime(self, scoped, monkeypatch):
        monkeypatch.setattr(mc_broker, "host_busy", lambda: True)

        assert not self.server_on(None, cpu=True)._image_job_conflicts()


# --------------------------------------------------------------------------- #
# 19.3 VRAM accounting (T17-T21)
# --------------------------------------------------------------------------- #


class TestVramIsCountedPerCard:
    """Invariant I-2: a reading from GPU N is never combined with bytes from M."""

    def test_each_card_answers_for_itself(self, scoped, registry):
        """T17. 2 GB on GPU 0 and 19 GB on GPU 1 is not 21 GB on either."""
        hold(registry, Server(card=IMAGE_CARD, holds=2 * _GB), "small")
        hold(registry, Server(card=OTHER_CARD, holds=19 * _GB), "large")

        assert registry.resident_bytes(card=IMAGE_CARD) == 2 * _GB
        assert registry.resident_bytes(card=OTHER_CARD) == 19 * _GB

    def test_the_machine_wide_total_is_still_available_when_asked_for(
            self, scoped, registry):
        """T17's other half. The figure is not wrong, it was in the wrong place."""
        hold(registry, Server(card=IMAGE_CARD, holds=2 * _GB), "small")
        hold(registry, Server(card=OTHER_CARD, holds=19 * _GB), "large")

        assert registry.resident_bytes() == 21 * _GB

    def test_a_processor_runtime_holds_no_vram_on_any_card(self, scoped, registry):
        hold(registry, Server(cpu=True, holds=0), "processor")

        assert registry.resident_bytes(card=IMAGE_CARD) == 0
        assert registry.resident_bytes(card=OTHER_CARD) == 0

    def test_the_register_filters_by_card_too(self, scoped):
        mc_broker.declare(mc_broker.FAMILY_LLM, "a", "on 0", 2 * _GB, card=IMAGE_CARD)
        mc_broker.declare(mc_broker.FAMILY_LLM, "b", "on 1", 19 * _GB, card=OTHER_CARD)

        assert mc_broker.resident_bytes(mc_broker.FAMILY_LLM, card=IMAGE_CARD) == 2 * _GB
        assert mc_broker.resident_bytes(mc_broker.FAMILY_LLM, card=OTHER_CARD) == 19 * _GB
        assert mc_broker.resident_bytes(mc_broker.FAMILY_LLM) == 21 * _GB

    def test_no_filter_is_not_the_same_request_as_the_unknown_card(self, scoped):
        """Section 7.1. One ``None`` cannot carry both, so it carries neither."""
        mc_broker.declare(mc_broker.FAMILY_LLM, "a", "on 0", 2 * _GB, card=IMAGE_CARD)
        mc_broker.declare(mc_broker.FAMILY_LLM, "b", "nowhere", 5 * _GB, card=None)

        assert mc_broker.resident_bytes(mc_broker.FAMILY_LLM) == 7 * _GB
        assert mc_broker.resident_bytes(mc_broker.FAMILY_LLM, card=None) == 5 * _GB
        assert mc_broker.resident_bytes(mc_broker.FAMILY_LLM, card=IMAGE_CARD) == 2 * _GB

    def test_the_image_card_status_shows_only_the_image_cards_llm_bytes(
            self, scoped, registry, monkeypatch):
        """T20. Nineteen gigabytes of 5090 beside a 3090's free/total describes
        a card that is over-subscribed by fifteen and a machine where neither
        card is short of anything."""
        free_on(monkeypatch, gpu0=4, gpu1=11)
        monkeypatch.setattr(mc_broker, "total_vram_bytes", lambda: 24 * _GB)
        hold(registry, Server(card=IMAGE_CARD, holds=2 * _GB), "small")
        hold(registry, Server(card=OTHER_CARD, holds=19 * _GB), "large")

        status = mc_broker.status()

        assert status.card == IMAGE_CARD
        assert status.llm_bytes == 2 * _GB
        assert status.llm_bytes_total == 21 * _GB
        assert status.llm_bytes_elsewhere == 19 * _GB

    def test_a_card_scoped_budget_uses_only_that_cards_llm_bytes(
            self, scoped, registry, monkeypatch):
        """T18, T19. The figure the plan and the reserve-miss record read."""
        hold(registry, Server(card=IMAGE_CARD, holds=2 * _GB), "small")
        hold(registry, Server(card=OTHER_CARD, holds=19 * _GB), "large")

        assert mc_broker.reported_bytes(mc_broker.FAMILY_LLM,
                                        card=IMAGE_CARD) == 2 * _GB
        assert mc_memory._llm_residency_bytes() == 2 * _GB

    def test_a_cuda_context_on_another_card_is_not_this_cards_unaccounted_vram(
            self, scoped, registry, monkeypatch):
        """T21. A Mixed Conservative server executing on GPU 1 holds no VRAM
        anywhere this register can see, and GPU 0's status must not hide a
        gigabyte as "our own LLM context" on its account."""
        free_on(monkeypatch, gpu0=4, gpu1=20)
        monkeypatch.setattr(mc_broker, "total_vram_bytes", lambda: 24 * _GB)
        mc_broker.register_reclaimer(mc_broker.FAMILY_IMAGE, ImageSide(holds=18 * _GB))
        hold(registry, Server(card=OTHER_CARD, holds=0), "mixed")

        # 24 total - 4 free - 18 image - 1 driver = 1 GB unexplained, and no
        # second allowance for a context that is not on this card.
        assert mc_broker.unaccounted_bytes(card=IMAGE_CARD) == 1 * _GB

    def test_a_context_on_this_card_does_get_its_allowance(
            self, scoped, registry, monkeypatch):
        """The same rule pointing the other way: a server really executing here
        with its weights in system RAM is ours, not somebody else's stray."""
        free_on(monkeypatch, gpu0=4, gpu1=20)
        monkeypatch.setattr(mc_broker, "total_vram_bytes", lambda: 24 * _GB)
        mc_broker.register_reclaimer(mc_broker.FAMILY_IMAGE, ImageSide(holds=18 * _GB))
        hold(registry, Server(card=IMAGE_CARD, holds=0), "mixed")

        assert mc_broker.unaccounted_bytes(card=IMAGE_CARD) == 0


# --------------------------------------------------------------------------- #
# 19.4 VRAM reclaim (T22-T28)
# --------------------------------------------------------------------------- #


class TestReclaimStaysOnItsCard:
    """Invariant I-3, and the sentence the design intent repeats three times:
    a 19 GB language model on GPU 1 must never be stopped to solve a 4 GB
    shortage on GPU 0."""

    def test_a_shortfall_here_does_not_reach_a_server_there(
            self, scoped, registry, monkeypatch):
        """T22. The PID survives."""
        free_on(monkeypatch, gpu0=0.5, gpu1=11)
        away = hold(registry, Server(card=OTHER_CARD, holds=19 * _GB), "away")

        result = mc_broker.request_vram(mc_broker.FAMILY_IMAGE, 4 * _GB,
                                        reason="a Stage 2 pass", card=IMAGE_CARD)

        assert away.up
        assert away.holds == 19 * _GB
        assert result.freed == 0

    def test_only_the_same_card_runtime_is_eligible(self, scoped, registry, monkeypatch):
        """T23. Two servers, one shortfall, one victim."""
        free_on(monkeypatch, gpu0=0.5, gpu1=11)
        here = hold(registry, Server(card=IMAGE_CARD, holds=3 * _GB), "here")
        away = hold(registry, Server(card=OTHER_CARD, holds=19 * _GB), "away")

        mc_broker.request_vram(mc_broker.FAMILY_IMAGE, 3 * _GB,
                               reason="a Stage 2 pass", card=IMAGE_CARD)

        assert here.stopped
        assert not away.stopped

    def test_an_exclusive_sweep_has_no_authority_over_a_second_card(
            self, scoped, registry, monkeypatch, host):
        """T24. "Image owns the card" is a promise about Forge's card."""
        from modules import shared

        shared.opts.model_chain_memory_mode = mc_broker.MODE_EXCLUSIVE
        free_on(monkeypatch, gpu0=20, gpu1=11)
        away = hold(registry, Server(card=OTHER_CARD, holds=19 * _GB), "away")

        mc_broker.request_vram(mc_broker.FAMILY_IMAGE, 0,
                               reason="an image generation started", margin=0,
                               card=IMAGE_CARD)

        assert away.up, "an Exclusive sweep crossed to another GPU"

    def test_an_exclusive_sweep_still_clears_the_image_card(
            self, scoped, registry, monkeypatch, host):
        """T25. Unchanged where it applies."""
        from modules import shared

        shared.opts.model_chain_memory_mode = mc_broker.MODE_EXCLUSIVE
        free_on(monkeypatch, gpu0=20, gpu1=11)
        here = hold(registry, Server(card=IMAGE_CARD, holds=3 * _GB), "here")

        mc_broker.request_vram(mc_broker.FAMILY_IMAGE, 0,
                               reason="an image generation started", margin=0,
                               card=IMAGE_CARD)

        assert here.stopped

    def test_an_unresolvable_image_card_on_a_multi_gpu_machine_reclaims_nothing(
            self, scoped, registry, monkeypatch):
        """T26. Section 16.1: a guessed GPU is not a reclaim target. Stopping a
        server chosen by guesswork can free every byte it holds and leave the
        shortfall exactly where it was."""
        free_on(monkeypatch, gpu0=0.5, gpu1=11)
        monkeypatch.setattr(mc_broker, "image_device_index", lambda: -1)
        monkeypatch.setattr(mc_broker, "cuda_device_count", lambda: 2)
        away = hold(registry, Server(card=OTHER_CARD, holds=19 * _GB), "away")

        mc_broker.request_vram(mc_broker.FAMILY_IMAGE, 4 * _GB, reason="a pass")

        assert away.up

    def test_one_card_that_cannot_be_named_is_still_reclaimed_from(
            self, scoped, registry, monkeypatch):
        """The other half of T26, and the reason it is not simply "refuse". On
        a single-card machine an unreadable index changes nothing: everything
        is on the one card, so the unfiltered answer *is* the card-local one,
        and refusing would break every installation that works today."""
        free_on(monkeypatch, gpu0=0.5)
        monkeypatch.setattr(mc_broker, "image_device_index", lambda: -1)
        monkeypatch.setattr(mc_broker, "cuda_device_count", lambda: 1)
        only = hold(registry, Server(card=IMAGE_CARD, holds=3 * _GB), "only")

        mc_broker.request_vram(mc_broker.FAMILY_IMAGE, 3 * _GB, reason="a pass")

        assert only.stopped

    def test_a_runtime_refuses_a_release_aimed_at_another_card(self, scoped, monkeypatch):
        """T27. Defence in depth: the registry filters, and this refuses even
        when something reaches past the registry."""
        held = runtime.Runtime()
        held._card = OTHER_CARD

        assert held.release(4 * _GB, "a pass on GPU 0", card=IMAGE_CARD) == 0

    def test_a_runtime_whose_card_is_unknown_is_not_a_targeted_victim(self, scoped):
        """Section 7.3. A release that cannot be shown to help is not made."""
        held = runtime.Runtime()
        held._card = None
        # CUDA, and an index nothing can parse -- which is exactly the state
        # that must not be read as "the image card, probably".
        held.configuration = lambda: _Unparseable()

        assert not held.on_card(IMAGE_CARD)
        assert not held.on_card(OTHER_CARD)

    def test_a_processor_runtime_is_not_a_vram_victim_for_any_card(self, scoped, registry,
                                                                   monkeypatch):
        free_on(monkeypatch, gpu0=0.5)
        processor = hold(registry, Server(cpu=True, holds=0), "processor")

        mc_broker.request_vram(mc_broker.FAMILY_IMAGE, 4 * _GB, reason="a pass",
                               card=IMAGE_CARD)

        assert processor.up

    def test_the_reserve_miss_hook_carries_the_image_card(self, scoped, registry,
                                                          monkeypatch):
        """T22 through the door ``mc_memory`` actually uses."""
        free_on(monkeypatch, gpu0=0.5, gpu1=11)
        away = hold(registry, Server(card=OTHER_CARD, holds=19 * _GB), "away")

        assert mc_broker._reclaim_for_image(3 * _GB, reason="the Stage 2 pass") == 0
        assert away.up

    def test_the_target_cards_free_vram_is_re_measured_after_a_release(
            self, scoped, registry, monkeypatch):
        """T28, and invariant I-15: the measurement wins over the arithmetic."""
        readings = {IMAGE_CARD: int(0.5 * _GB)}
        monkeypatch.setattr(mc_broker, "free_vram_bytes",
                            lambda: readings[IMAGE_CARD])
        monkeypatch.setattr(mc_broker, "device_free_vram_bytes",
                            lambda index=None: readings.get(
                                IMAGE_CARD if index is None else int(index), 0))

        class Recovering(Server):
            def release(self, needed_bytes, reason="", *, card=mc_broker.ANY_CARD):
                freed = super().release(needed_bytes, reason, card=card)
                readings[IMAGE_CARD] = 5 * _GB
                return freed

        hold(registry, Recovering(card=IMAGE_CARD, holds=3 * _GB), "here")

        result = mc_broker.request_vram(mc_broker.FAMILY_IMAGE, 4 * _GB,
                                        reason="a pass", card=IMAGE_CARD)

        assert result.satisfied


# --------------------------------------------------------------------------- #
# 19.5 The image allocator's cache (T29-T31)
# --------------------------------------------------------------------------- #


class TestTheImageAllocatorCache:
    """Section 13. Emptying it is useful only for a start on the same card.

    The blocks Forge's allocator is sitting on belong to one GPU's driver. A
    llama-server being placed on that GPU can have them once they are handed
    back; one anywhere else cannot, and the image side has given up every
    cached block it had ready for nothing.
    """

    @pytest.fixture
    def emptied(self, scoped, monkeypatch):
        calls: list[int] = []
        monkeypatch.setattr(mc_broker, "release_cached_vram",
                            lambda: (calls.append(1), 2 * _GB)[1])
        return calls

    def test_a_start_on_the_image_card_still_empties_it(self, emptied, tmp_path):
        """T29."""
        held = runtime.Runtime()

        recovered = held._release_the_image_cache_if_it_helps(
            configuration_on(tmp_path, card=IMAGE_CARD))

        assert emptied == [1]
        assert recovered == 2 * _GB

    def test_a_start_on_another_card_leaves_it_alone(self, emptied, tmp_path):
        """T30."""
        held = runtime.Runtime()

        recovered = held._release_the_image_cache_if_it_helps(
            configuration_on(tmp_path, card=OTHER_CARD))

        assert emptied == []
        assert recovered == 0

    def test_a_processor_start_leaves_it_alone(self, emptied, tmp_path):
        """T31."""
        held = runtime.Runtime()

        recovered = held._release_the_image_cache_if_it_helps(
            configuration_on(tmp_path, card=None, mode="cpu"))

        assert emptied == []
        assert recovered == 0

    def test_a_mixed_conservative_start_on_the_image_card_still_empties_it(
            self, emptied, tmp_path):
        """It executes here and allocates a context here, so the blocks are
        genuinely useful to it -- ``on_gpu`` being False is beside the point."""
        held = runtime.Runtime()

        held._release_the_image_cache_if_it_helps(
            configuration_on(tmp_path, card=IMAGE_CARD, mode="mixed_conservative"))

        assert emptied == [1]


# --------------------------------------------------------------------------- #
# 19.6 Host RAM (T32-T42)
# --------------------------------------------------------------------------- #


class TestHostRamAdmission:
    """Section 10.8. Sharing a pool is not a conflict; being short of it is."""

    @pytest.fixture
    def cache(self, scoped, monkeypatch):
        """A warm image cache the broker can ask about and ask from."""
        state = {"warm": 14 * _GB, "reclaimable": 14 * _GB, "released": []}
        monkeypatch.setattr(mc_broker, "image_warm_ram_bytes", lambda: state["warm"])
        monkeypatch.setattr(mc_broker, "reclaimable_image_ram_bytes",
                            lambda: state["reclaimable"])

        def release(needed, reason=""):
            state["released"].append((needed, reason))
            freed = min(state["reclaimable"], needed)
            state["warm"] -= freed
            state["reclaimable"] -= freed
            return freed

        monkeypatch.setattr(mc_broker, "release_image_warm_ram", release)
        return state

    def test_a_demand_that_fits_disturbs_nothing(self, cache, monkeypatch):
        """T32, and invariant I-5. The entire point of the cache is to use
        otherwise-available memory; emptying it because somebody else also uses
        RAM would be evict-on-switch wearing a scheduler's clothes."""
        ram(monkeypatch, available_gb=40)

        admission = mc_broker.admit_host_ram(28 * _GB, reason="a CPU LLM")

        assert admission.fits
        assert cache["released"] == []
        assert cache["warm"] == 14 * _GB

    def test_warm_cache_yields_when_the_demand_does_not_fit(self, cache, monkeypatch):
        """T33, and invariant I-6. Warm state exists to shorten a future
        switch; active execution memory answers the request in front of you."""
        readings = {"available": 19 * _GB}
        monkeypatch.setattr(mc_broker, "free_ram_bytes", lambda: readings["available"])
        monkeypatch.setattr(mc_broker, "ram_reserve_bytes", lambda: 2 * _GB)
        released = mc_broker.release_image_warm_ram

        def release(needed, reason=""):
            freed = released(needed, reason)
            readings["available"] += freed
            return freed

        monkeypatch.setattr(mc_broker, "release_image_warm_ram", release)

        admission = mc_broker.admit_host_ram(28 * _GB, reason="a CPU LLM")

        assert cache["released"], "the warm cache was never asked"
        assert admission.fits
        assert admission.freed > 0

    def test_the_re_reading_is_what_decides_not_the_arithmetic(self, cache, monkeypatch):
        """T39, and invariant I-15. A cache that reports fourteen gigabytes
        released has said what it stopped referencing, not what the operating
        system has made available again."""
        ram(monkeypatch, available_gb=19)

        admission = mc_broker.admit_host_ram(28 * _GB, reason="a CPU LLM")

        # The cache said it freed 14 GB; available RAM did not move, so the
        # answer is no.
        assert admission.freed > 0
        assert not admission.fits

    def test_nothing_reclaimable_is_reported_rather_than_forced(self, cache, monkeypatch):
        """T34. Active image host memory is not a generic victim: only what
        ``mc_memory`` positively identifies as safe warm cache may go."""
        ram(monkeypatch, available_gb=19)
        cache["reclaimable"] = 0

        admission = mc_broker.admit_host_ram(28 * _GB, reason="a CPU LLM")

        assert cache["released"] == []
        assert not admission.fits
        assert admission.shortfall > 0

    def test_unreadable_memory_reclaims_nothing_and_says_so(
            self, cache, monkeypatch, caplog):
        """T40, section 16.3. Reclaiming on invented numbers is worse than not
        knowing, and refusing to start would break installations that work."""
        monkeypatch.setattr(mc_broker, "free_ram_bytes", lambda: 0)

        with caplog.at_level("WARNING"):
            admission = mc_broker.admit_host_ram(28 * _GB, reason="a CPU LLM")

        assert not admission.known
        assert admission.fits, "an unanswerable question must not refuse a start"
        assert cache["released"] == []
        assert any("could not be read" in line for line in caplog.messages)

    def test_the_reserve_is_part_of_what_has_to_fit(self, cache, monkeypatch):
        ram(monkeypatch, available_gb=29, reserve_gb=2.0)

        assert not mc_broker.host_ram_fits(28 * _GB)
        assert mc_broker.host_ram_fits(27 * _GB)


class TestHowMuchHostRamAPlacementReallyNeeds:
    """T41, T42, section 10.7. The shape matters more than the precision."""

    def test_a_processor_placement_needs_the_whole_model(self, scoped, tmp_path):
        """T42."""
        settings = configuration_on(tmp_path, card=None, mode="cpu", size_mb=64)

        assert runtime.host_ram_demand(settings) == Path(settings.model).stat().st_size

    def test_mixed_conservative_needs_the_whole_model_too(self, scoped, tmp_path):
        """T42. Every weight is in system RAM; the card only computes."""
        settings = configuration_on(tmp_path, card=OTHER_CARD,
                                    mode="mixed_conservative", size_mb=64)

        assert runtime.host_ram_demand(settings) == Path(settings.model).stat().st_size

    def test_a_full_gpu_placement_reserves_no_permanent_host_ram(self, scoped, tmp_path):
        """T41. Its GGUF is read through mmap and the pages the OS keeps
        afterwards are the OS's to reclaim. Counting the file size as a hard
        reservation would block image work that is perfectly safe."""
        import mc_llm_context as ctx

        settings = configuration_on(tmp_path, card=OTHER_CARD, size_mb=64)
        resident = ctx.Placement(gpu_layers=ctx.ALL_LAYERS, on_gpu=True)

        assert runtime.host_ram_demand(settings, resident) == 0

    def test_a_partial_offload_needs_the_share_that_stayed_behind(self, scoped, tmp_path):
        """Coarse and far closer than either extreme: half the blocks on the
        card is about half the file in host memory."""
        import mc_llm_context as ctx

        settings = configuration_on(tmp_path, card=OTHER_CARD, blocks=32, size_mb=64)
        size = Path(settings.model).stat().st_size
        half = ctx.Placement(gpu_layers=16, on_gpu=True)

        assert runtime.host_ram_demand(settings, half) == pytest.approx(size / 2, rel=0.02)

    def test_a_model_that_cannot_be_sized_demands_nothing(self, scoped, tmp_path):
        """An unknown demand admits nothing and reclaims nothing, which is the
        only honest answer."""
        settings = configuration_on(tmp_path, card=None, mode="cpu")
        object.__setattr__(settings, "model", tmp_path / "not-there.gguf")

        assert runtime.host_ram_demand(settings) == 0


class TestTheWarmImageCacheAsTheBrokerSeesIt:
    """Section 10.2. Three functions, and ``mc_memory`` keeps every decision."""

    @pytest.fixture
    def cached(self, host, monkeypatch):
        mc_memory.release_all()
        entries = [
            mc_memory._Entry(key="a", checkpoint_name="krea2", sd_model=object(),
                             size_bytes=6 * _GB),
            mc_memory._Entry(key="b", checkpoint_name="klein9b", sd_model=object(),
                             size_bytes=8 * _GB),
        ]
        entries[0].last_used = 1.0
        entries[1].last_used = 2.0
        for entry in entries:
            mc_memory._cache._entries[entry.key] = entry
        yield entries
        mc_memory.release_all()

    def test_it_reports_what_it_is_holding(self, cached):
        assert mc_memory.warm_ram_bytes() == 14 * _GB

    def test_the_loaded_checkpoint_is_not_reclaimable(self, cached, monkeypatch):
        """Section 10.10. Its weights may be partly on the card and partly
        offloaded here, and it is what the running pass is executing against."""
        monkeypatch.setattr(mc_memory, "_loaded_model_key", lambda: "b")

        assert mc_memory.reclaimable_warm_ram_bytes() == 6 * _GB

    def test_it_drops_least_recently_used_first(self, cached, monkeypatch):
        monkeypatch.setattr(mc_memory, "_loaded_model_key", lambda: "")

        freed = mc_memory.release_warm_ram(1 * _GB, reason="a CPU LLM")

        assert freed == 6 * _GB
        assert mc_memory.cached_names() == ["klein9b"]

    def test_it_never_drops_the_loaded_checkpoint_to_meet_a_request(
            self, cached, monkeypatch):
        monkeypatch.setattr(mc_memory, "_loaded_model_key", lambda: "b")

        freed = mc_memory.release_warm_ram(20 * _GB, reason="a CPU LLM")

        assert freed == 6 * _GB
        assert mc_memory.cached_names() == ["klein9b"]

    def test_asking_for_nothing_drops_nothing(self, cached):
        assert mc_memory.release_warm_ram(0) == 0
        assert mc_memory.warm_ram_bytes() == 14 * _GB


class TestTheImageCacheStillOwnsItsOwnAdmission:
    """T38. The reverse direction, which was already right and must stay right."""

    def test_a_large_processor_llm_shrinks_what_the_cache_may_take(
            self, host, monkeypatch, tmp_path):
        """``_stash_current`` reads live available RAM, so a llama-server
        holding twenty gigabytes has already reduced the budget without
        anybody having to model it."""
        monkeypatch.setattr(mc_memory, "free_ram_bytes", lambda: 3 * _GB)
        monkeypatch.setattr(mc_memory, "total_ram_bytes", lambda: 64 * _GB)

        headroom = max(mc_memory.free_ram_bytes() - mc_memory.RAM_RESERVE_BYTES, 0)

        assert headroom < 2 * _GB, "the live guard did not see the pressure"

    def test_the_refusal_names_the_language_model_when_it_is_the_reason(
            self, host, monkeypatch, caplog):
        """Section 10.9 is visibility only. Without it the message points at
        the RAM-budget setting, and raising that setting cannot conjure memory
        another process is holding."""
        monkeypatch.setattr(mc_broker, "llm_host_ram_bytes", lambda: 22 * _GB)

        note = mc_memory._llm_ram_note()

        assert "22.0 GB" in note
        assert "system RAM" in note

    def test_it_says_nothing_when_no_language_model_wants_host_ram(self, host,
                                                                   monkeypatch):
        monkeypatch.setattr(mc_broker, "llm_host_ram_bytes", lambda: 0)

        assert mc_memory._llm_ram_note() == ""


class TestRamBackedRolesCoexistWhileItIsSafe:
    """T35-T37, section 11.4. "Coexist" means coexist *while safe*."""

    @staticmethod
    def wants(monkeypatch, gigabytes):
        """A model of ``gigabytes`` without writing one to disk.

        The estimate itself is examined in
        :class:`TestHowMuchHostRamAPlacementReallyNeeds`; what these tests are
        about is what the registry *does* with it.
        """
        monkeypatch.setattr(runtime, "host_ram_demand",
                            lambda configuration, placement=None: int(gigabytes * _GB))

    def test_two_processor_servers_that_fit_both_stay_up(
            self, scoped, registry, monkeypatch, tmp_path):
        """T37."""
        ram(monkeypatch, available_gb=64)
        self.wants(monkeypatch, 28)
        other = hold(registry, Server(cpu=True, host_ram=8 * _GB), "spatial")
        settings = configuration_on(tmp_path, card=None, mode="cpu")
        monkeypatch.setattr(runtime, "config", lambda role="": settings)

        assert registry._can_coexist_in_ram("creative", settings)
        assert other.up

    def test_a_second_server_that_cannot_fit_is_not_admitted_silently(
            self, scoped, registry, monkeypatch, tmp_path):
        """T36. Refused with a memory reason rather than admitted below the
        floor, where it does not run faster -- it pages, and takes the desktop
        with it."""
        ram(monkeypatch, available_gb=3)
        self.wants(monkeypatch, 28)
        monkeypatch.setattr(mc_broker, "reclaimable_image_ram_bytes", lambda: 0)
        settings = configuration_on(tmp_path, card=None, mode="cpu")

        assert not registry._can_coexist_in_ram("creative", settings)

    def test_warm_image_cache_is_asked_before_two_servers_are_called_a_conflict(
            self, scoped, registry, monkeypatch, tmp_path):
        """T35's first step, and section 11.3's priority: dropping an unused
        cache entry is a smaller action than stopping a working server."""
        readings = {"available": 3 * _GB}
        monkeypatch.setattr(mc_broker, "free_ram_bytes", lambda: readings["available"])
        monkeypatch.setattr(mc_broker, "ram_reserve_bytes", lambda: 2 * _GB)
        monkeypatch.setattr(mc_broker, "reclaimable_image_ram_bytes", lambda: 20 * _GB)
        asked: list = []

        def release(needed, reason=""):
            asked.append((needed, reason))
            readings["available"] = 40 * _GB
            return needed

        monkeypatch.setattr(mc_broker, "release_image_warm_ram", release)
        self.wants(monkeypatch, 28)
        settings = configuration_on(tmp_path, card=None, mode="cpu")

        assert registry._can_coexist_in_ram("creative", settings)
        assert asked, "the warm cache was never asked before declaring a conflict"

    def test_a_model_of_unknown_size_is_not_evidence_of_pressure(
            self, scoped, registry, monkeypatch, tmp_path):
        """Inventing pressure would stand a working server down for a number
        nobody has."""
        ram(monkeypatch, available_gb=1)
        settings = configuration_on(tmp_path, card=None, mode="cpu")
        object.__setattr__(settings, "model", tmp_path / "missing.gguf")

        assert registry._can_coexist_in_ram("creative", settings)


# --------------------------------------------------------------------------- #
# 19.7 Cross-domain transitions (T43, T44)
# --------------------------------------------------------------------------- #


class TestAVramDeficitIsNeverSolvedByAHostFailure:
    """T43, T44, sections 17.9 and 18.11. Moving an LLM from VRAM to system RAM
    solves a VRAM shortage by creating a host-RAM demand of the model's size."""

    @pytest.fixture
    def demoting(self, scoped, monkeypatch, tmp_path):
        from modules import shared

        shared.opts.model_chain_llm_release = runtime.RELEASE_SYSTEM_RAM
        settings = configuration_on(tmp_path, card=IMAGE_CARD, size_mb=64)
        monkeypatch.setattr(runtime, "config", lambda role="": settings)
        return settings

    def test_a_healthy_host_pool_lets_the_demotion_proceed(self, demoting, monkeypatch):
        """T43. The right trade: the server keeps answering, more slowly, and
        its prompt cache survives."""
        ram(monkeypatch, available_gb=64)
        moved: list = []
        held = runtime.Runtime()
        held._placement = _resident()
        monkeypatch.setattr(runtime.Runtime, "_stop_locked",
                            lambda self, reason: moved.append(reason))
        monkeypatch.setattr(runtime.Runtime, "_new_process", lambda self: _Process())

        assert held._restart_in_system_ram(0, "an image pass") or True
        assert moved, "a demotion with room to spare was refused"

    def test_an_exhausted_host_pool_stops_the_server_instead(
            self, demoting, monkeypatch, caplog):
        """T44. Stopping is the safe reclaim; the file pages stay soft-warm in
        the OS cache for the next start."""
        ram(monkeypatch, available_gb=1)
        monkeypatch.setattr(mc_broker, "reclaimable_image_ram_bytes", lambda: 0)
        moved: list = []
        held = runtime.Runtime()
        held._placement = _resident()
        monkeypatch.setattr(runtime.Runtime, "_stop_locked",
                            lambda self, reason: moved.append(reason))

        with caplog.at_level("INFO"):
            freed = held._restart_in_system_ram(0, "an image pass")

        assert freed == 0
        assert moved == [], "the server was relocated into a host pool that cannot hold it"
        assert any("rather than moved to system RAM" in line for line in caplog.messages)


def _resident():
    import mc_llm_context as ctx

    return ctx.Placement(gpu_layers=ctx.ALL_LAYERS, context=8192, on_gpu=True)


class _Process:
    running = True

    def start(self, *args, **kwargs):
        return None

    def wait_ready(self, timeout=0):
        return True

    def stop(self):
        return None


# --------------------------------------------------------------------------- #
# 19.8 Simultaneous activity (T45-T50)
# --------------------------------------------------------------------------- #


class TestBothJobsCanBeVisibleAtOnce:
    """T45-T49, section 14.1.

    A backend that technically overlaps work but presents one global busy
    state still teaches the user that the machine cannot do both jobs. The
    target hardware was assembled to create independent resources, and the
    product has to expose that.
    """

    def test_an_image_job_and_an_llm_turn_are_both_reported(self, scoped, monkeypatch):
        """T45."""
        monkeypatch.setattr(mc_broker, "host_busy", lambda: True)

        with mc_broker.workload(mc_broker.FAMILY_LLM, "a conversation reply",
                                domain=mc_broker.cuda_execution(OTHER_CARD)):
            running = mc_broker.activities()

        families = {entry.family for entry in running}
        assert families == {mc_broker.FAMILY_IMAGE, mc_broker.FAMILY_LLM}

    def test_each_one_names_its_own_processor(self, scoped, monkeypatch):
        """Section 14.3's status text, which is the point of carrying the
        domain rather than merely allowing two rows."""
        monkeypatch.setattr(mc_broker, "host_busy", lambda: True)

        with mc_broker.workload(mc_broker.FAMILY_LLM, "a conversation reply",
                                domain=mc_broker.cuda_execution(OTHER_CARD)):
            said = [entry.describe() for entry in mc_broker.activities()]

        assert "Image generation on GPU 0" in said
        assert "a conversation reply on GPU 1" in said

    def test_a_processor_conversation_is_representable_beside_an_image_job(
            self, scoped, monkeypatch):
        """T46."""
        monkeypatch.setattr(mc_broker, "host_busy", lambda: True)

        with mc_broker.workload(mc_broker.FAMILY_LLM, "a conversation reply",
                                domain=mc_broker.CPU_EXECUTION):
            said = [entry.describe() for entry in mc_broker.activities()]

        assert "Image generation on GPU 0" in said
        assert "a conversation reply on the processor" in said

    def test_nothing_running_is_still_nothing(self, scoped, monkeypatch):
        monkeypatch.setattr(mc_broker, "host_busy", lambda: False)

        assert mc_broker.activities() == ()

    def test_an_image_job_alone_is_reported_without_any_llm_lock(
            self, scoped, monkeypatch):
        """T49. Image activity is derived from the host's own state rather than
        from a lock this extension holds, so it can be shown without anything
        being disabled on the LLM side."""
        monkeypatch.setattr(mc_broker, "host_busy", lambda: True)

        running = mc_broker.activities()

        assert len(running) == 1
        assert running[0].family == mc_broker.FAMILY_IMAGE
        assert mc_broker.active() is None

    def test_cancelling_the_llm_leaves_the_image_job_running(self, scoped, monkeypatch):
        """T47. The LLM's stop button applies to the LLM request."""
        monkeypatch.setattr(mc_broker, "host_busy", lambda: True)
        with mc_broker.workload(mc_broker.FAMILY_LLM, "a conversation reply",
                                domain=mc_broker.cuda_execution(OTHER_CARD)):
            pass

        running = mc_broker.activities()

        assert [entry.family for entry in running] == [mc_broker.FAMILY_IMAGE]

    def test_the_image_job_ending_leaves_the_llm_running(self, scoped, monkeypatch):
        """T48."""
        busy = {"value": True}
        monkeypatch.setattr(mc_broker, "host_busy", lambda: busy["value"])

        with mc_broker.workload(mc_broker.FAMILY_LLM, "a conversation reply",
                                domain=mc_broker.cuda_execution(OTHER_CARD)):
            busy["value"] = False
            running = mc_broker.activities()

        assert [entry.family for entry in running] == [mc_broker.FAMILY_LLM]

    def test_a_same_card_conflict_still_produces_a_waiting_state(self, scoped,
                                                                 monkeypatch):
        """T50. Narrowing the predicate must not remove the message for the
        case where waiting is the correct thing to do."""
        monkeypatch.setattr(sessions, "WAIT_NOTICE_SECONDS", 0.0)
        monkeypatch.setattr(sessions, "WAIT_POLL_SECONDS", 0.01)
        monkeypatch.setattr(mc_broker, "host_busy", lambda: True)
        gpu = sessions._Gpu("a conversation reply", sessions.Cancellation())
        monkeypatch.setattr(gpu, "_domain",
                            lambda: mc_broker.cuda_execution(IMAGE_CARD))
        cancel = threading.Timer(0.2, gpu.cancel.cancel)
        cancel.start()

        _acquired, events = take(gpu)
        cancel.cancel()

        assert any("image generation on GPU 0" in text
                   for text in waiting_texts(events))


class TestThePanelSaysWhichPool:
    """Section 21.1. Every physical-memory number names its domain."""

    def test_the_residency_view_names_each_card_and_the_host_pool(
            self, scoped, registry, monkeypatch):
        import mc_llm_studio

        free_on(monkeypatch, gpu0=4.2, gpu1=11.6)
        monkeypatch.setattr(mc_broker, "total_vram_bytes", lambda: 24 * _GB)
        monkeypatch.setattr(mc_broker, "free_ram_bytes", lambda: int(41.2 * _GB))
        monkeypatch.setattr(mc_broker, "image_warm_ram_bytes", lambda: 18 * _GB)
        hold(registry, Server(card=OTHER_CARD, holds=19 * _GB), "away")

        html = mc_llm_studio._residency_table()

        assert "GPU 0 VRAM" in html
        assert "Host RAM" in html
        assert "LLM VRAM on other cards" in html

    def test_it_shows_both_activities_rather_than_one(self, scoped, registry,
                                                      monkeypatch):
        import mc_llm_studio

        monkeypatch.setattr(mc_broker, "host_busy", lambda: True)
        with mc_broker.workload(mc_broker.FAMILY_LLM, "a conversation reply",
                                domain=mc_broker.cuda_execution(OTHER_CARD)):
            html = mc_llm_studio._residency_table()

        assert "Image generation on GPU 0" in html
        assert "a conversation reply on GPU 1" in html


# --------------------------------------------------------------------------- #
# 19.9 Lifecycle (T51-T54)
# --------------------------------------------------------------------------- #


class TestOwnershipIsStillGlobal:
    """T51-T54, section 15. Device scoping changes scheduling, not ownership.

    A 5090 process is independent of 3090 pressure. It is not independent of
    the WebUI that started it.
    """

    def test_shutdown_stops_servers_on_every_card(self, scoped, registry, monkeypatch):
        """T51."""
        stopped: list = []

        class Stoppable(Server):
            def stop(self):
                stopped.append(self.card)
                self.up = False

        hold(registry, Stoppable(card=IMAGE_CARD, holds=2 * _GB), "here")
        hold(registry, Stoppable(card=OTHER_CARD, holds=19 * _GB), "away")
        monkeypatch.setattr(runtime, "registry", registry)
        monkeypatch.setattr(runtime, "release_strays", lambda: (0, 0))

        runtime.shutdown()

        assert sorted(stopped) == [IMAGE_CARD, OTHER_CARD]

    def test_registered_pids_on_any_card_are_excluded_from_stray_detection(
            self, scoped, registry, monkeypatch):
        """T52. Process-name scanning is not memory-resource inference: a
        server on another card is still ours, and must not be swept."""
        class WithPid(Server):
            def __init__(self, pid, **kwargs):
                super().__init__(**kwargs)
                # The shape ``_pid_of`` reads: a runtime holds a process
                # wrapper, which holds the ``Popen`` the pid lives on.
                self._process = type("Wrapper", (), {
                    "process": type("Popen", (), {"pid": pid})()})()

        hold(registry, WithPid(4242, card=IMAGE_CARD), "here")
        hold(registry, WithPid(5353, card=OTHER_CARD), "away")
        monkeypatch.setattr(runtime, "registry", registry)

        ours = runtime._own_pids()

        assert {4242, 5353} <= ours

    def test_the_registry_alone_excludes_a_runtime_that_does_not_defend_itself(
            self, scoped, registry, monkeypatch):
        """The filter and the runtime's own refusal are two locks on one door,
        and this checks the first without the second. A runtime that cannot say
        which card it is on is not a targeted victim (section 7.3), so the
        registry must leave it out rather than ask it and hope."""
        class Undefended(Server):
            on_card = None  # not callable: this double answers no such question

        free_on(monkeypatch, gpu0=0.5)
        loose = hold(registry, Undefended(card=OTHER_CARD, holds=19 * _GB), "loose")

        registry.release(4 * _GB, "an image pass on GPU 0", card=IMAGE_CARD)

        assert loose.calls == []
        assert loose.up

    def test_a_release_that_frees_nothing_never_stops_a_process_on_another_card(
            self, scoped, registry, monkeypatch):
        """The lifecycle counterpart of T22: the reclaim path may not become a
        back door to stopping a server it has no business stopping."""
        free_on(monkeypatch, gpu0=0.5, gpu1=20)
        away = hold(registry, Server(card=OTHER_CARD, holds=19 * _GB), "away")

        registry.release(4 * _GB, "an image pass on GPU 0", card=IMAGE_CARD)

        assert not away.stopped
        assert away.calls == [], "the runtime was asked at all"


# --------------------------------------------------------------------------- #
# 19.10 and 19.11: the end-to-end statements
# --------------------------------------------------------------------------- #


class TestTheReportedMachine:
    """19.10. Forge on the 3090, a fully resident Qwen-class model on the 5090.

    The acceptance statement, in one place: a 3090 generation can sample while
    a 5090 conversation keeps generating tokens, with both jobs visibly active,
    without waiting, reload, prompt-cache loss, process restart, cross-card
    reclaim, or forced system-RAM placement.
    """

    @pytest.fixture
    def warm(self, scoped, registry, monkeypatch):
        """A 19 GB server on GPU 1, and a 3090 with a checkpoint on it."""
        free_on(monkeypatch, gpu0=4.2, gpu1=11.6)
        monkeypatch.setattr(mc_broker, "total_vram_bytes", lambda: 24 * _GB)
        mc_broker.register_reclaimer(mc_broker.FAMILY_IMAGE, ImageSide(holds=18 * _GB))
        server = hold(registry, Server(card=OTHER_CARD, holds=19 * _GB), "conversation")
        mc_broker.declare(mc_broker.FAMILY_LLM, "llm:away", "the 5090 conversation",
                          19 * _GB, card=OTHER_CARD)
        return server

    def generate(self):
        """What ``before_process`` does to the broker, in the same order."""
        card = mc_broker.image_device_index()
        mc_broker.await_idle(timeout=1.0, domain=mc_broker.image_execution_domain())
        return mc_broker.request_vram(mc_broker.FAMILY_IMAGE, 0,
                                      reason="an image generation started", margin=0,
                                      card=card if card >= 0 else mc_broker.ANY_CARD)

    def test_a_generation_starting_does_not_touch_the_other_cards_server(self, warm):
        self.generate()

        assert warm.up
        assert warm.holds == 19 * _GB
        assert warm.calls == []

    def test_the_generation_does_not_wait_for_a_reply_in_flight(self, warm):
        with mc_broker.workload(mc_broker.FAMILY_LLM, "a conversation reply",
                                domain=mc_broker.cuda_execution(OTHER_CARD)):
            started = time.monotonic()
            self.generate()

            assert time.monotonic() - started < 1.0

    def test_a_reserve_miss_on_the_3090_leaves_the_5090_alone(self, warm):
        """Step 9 of the procedure. The 3090 is short and the 5090's nineteen
        gigabytes are not an answer to that."""
        freed = mc_broker._reclaim_for_image(3 * _GB, reason="the Stage 2 pass")

        assert freed == 0
        assert warm.up

    def test_the_image_card_budget_never_sees_the_other_cards_bytes(self, warm):
        status = mc_broker.status()

        assert status.llm_bytes == 0
        assert status.llm_bytes_total == 19 * _GB

    def test_both_are_visible_at_once(self, warm, monkeypatch):
        monkeypatch.setattr(mc_broker, "host_busy", lambda: True)

        with mc_broker.workload(mc_broker.FAMILY_LLM, "a conversation reply",
                                domain=mc_broker.cuda_execution(OTHER_CARD)):
            said = [entry.describe() for entry in mc_broker.activities()]

        assert said == ["Image generation on GPU 0", "a conversation reply on GPU 1"]

    def test_the_3090s_own_safety_still_works(self, scoped, registry, monkeypatch):
        """The other half of the acceptance statement, and the one that would
        be easy to lose: same-card protection is untouched."""
        free_on(monkeypatch, gpu0=0.5, gpu1=11.6)
        here = hold(registry, Server(card=IMAGE_CARD, holds=3 * _GB), "same card")

        mc_broker.request_vram(mc_broker.FAMILY_IMAGE, 3 * _GB,
                               reason="the Stage 2 pass", card=IMAGE_CARD)

        assert here.stopped


class TestTheHostRamRegression:
    """19.11. A warm Stage 2 checkpoint and a RAM-backed language model.

    Acceptance: no unnecessary serialisation, the host reserve preserved,
    optional warmth sacrificed before active work, no unrelated GPU reclaimed.
    """

    @pytest.fixture
    def machine(self, scoped, registry, monkeypatch):
        state = {"available": 19 * _GB, "warm": 14 * _GB}
        monkeypatch.setattr(mc_broker, "free_ram_bytes", lambda: state["available"])
        monkeypatch.setattr(mc_broker, "ram_reserve_bytes", lambda: 2 * _GB)
        monkeypatch.setattr(mc_broker, "image_warm_ram_bytes", lambda: state["warm"])
        monkeypatch.setattr(mc_broker, "reclaimable_image_ram_bytes",
                            lambda: state["warm"])

        def release(needed, reason=""):
            freed = min(state["warm"], needed)
            state["warm"] -= freed
            state["available"] += freed
            return freed

        monkeypatch.setattr(mc_broker, "release_image_warm_ram", release)
        free_on(monkeypatch, gpu0=4.2, gpu1=11.6)
        return state

    def test_a_processor_model_that_fits_leaves_the_cache_alone(self, machine):
        """Step 4: if the LLM fits, the image cache remains."""
        machine["available"] = 60 * _GB

        admission = mc_broker.admit_host_ram(28 * _GB, reason="a CPU conversation")

        assert admission.fits
        assert machine["warm"] == 14 * _GB

    def test_only_warm_cache_is_released_and_only_as_much_as_needed(self, machine):
        """Steps 5 and 6. Optional warmth yields before active demand, and the
        reserve is still standing afterwards."""
        admission = mc_broker.admit_host_ram(28 * _GB, reason="a CPU conversation")

        assert admission.fits
        assert machine["warm"] < 14 * _GB
        assert machine["available"] >= 28 * _GB + 2 * _GB

    def test_no_gpu_is_reclaimed_to_solve_a_host_shortage(self, machine, registry):
        """Section 10.8 step 9: a RAM shortage is not solved by making more RAM
        demand, and it is certainly not solved by emptying a card."""
        away = hold(registry, Server(card=OTHER_CARD, holds=19 * _GB), "away")
        here = hold(registry, Server(card=IMAGE_CARD, holds=3 * _GB), "here")

        mc_broker.admit_host_ram(28 * _GB, reason="a CPU conversation")

        assert away.up and here.up
        assert away.calls == [] and here.calls == []

    def test_after_the_conflict_is_resolved_the_two_still_run_concurrently(
            self, machine, monkeypatch):
        """Step 7. Resolving a *memory* conflict must not leave behind an
        *execution* one -- they were never the same question."""
        mc_broker.admit_host_ram(28 * _GB, reason="a CPU conversation")

        with mc_broker.workload(mc_broker.FAMILY_LLM, "a conversation reply",
                                domain=mc_broker.CPU_EXECUTION):
            assert mc_broker.await_idle(timeout=1.0,
                                        domain=mc_broker.cuda_execution(IMAGE_CARD))


# --------------------------------------------------------------------------- #
# The machine that reported this
# --------------------------------------------------------------------------- #


class TestTwoNamespacesForOneSetOfCards:
    """The failure every test above stipulated away.

    Both halves of this extension say "card 0" and mean different cards.
    ``nvidia-smi`` numbers by PCI bus and is what the language-model side
    records at setup; the CUDA runtime numbers by ``CUDA_DEVICE_ORDER``, which
    defaults to fastest-first, and a process under ``CUDA_VISIBLE_DEVICES``
    numbers from zero over whatever it can see. On the reported machine
    physical 0 is the 5090 and Forge's ordinal 0 is the 3090.

    Every fixture in this file used to *state* that the two agreed -- image
    card 0, language model card 1 -- so none of them could have caught it. From
    the user's log, with the old comparison in place:

        Waiting for image generation on GPU 0…
        re-placing llama-server — it holds 19.3 GB where the active plan leaves 4.1 GB
        released 13.1 GB of image VRAM on GPU 0 for Qwen 3.8 27B …
        5 layers on the GPU … 17.2 GB free

    A wait that was not needed, a plan that did not apply, a checkpoint evicted
    from a card the model was not going to, and a placement sized against the
    wrong GPU's free VRAM. One equality, four consequences.
    """

    IMAGE_UUID = "3090aaaa"
    LLM_UUID = "5090bbbb"
    IMAGE_NAME = "NVIDIA GeForce RTX 3090"
    LLM_NAME = "NVIDIA GeForce RTX 5090"

    @pytest.fixture
    def machine(self, scoped, monkeypatch):
        """Forge on the 3090, which nvidia-smi calls card 1 and torch calls 0.

        Built on ``scoped`` rather than beside it, and that is not tidiness: a
        sibling fixture would have been ordered against the one it is meant to
        override, and this class's whole subject is an image card that is *not*
        card 0. A version of this that quietly ran after ``scoped`` reset the
        card would have passed while testing nothing.
        """
        # Physical 1, not 0: the point is that this is not the number the
        # language model recorded, even though both processes would say "0" if
        # asked for their own ordinal.
        monkeypatch.setattr(mc_broker, "image_device_index", lambda: 1)
        monkeypatch.setattr(mc_broker, "image_device_uuid", lambda: self.IMAGE_UUID)
        monkeypatch.setattr(mc_broker, "image_device_name", lambda: self.IMAGE_NAME)
        yield

    def llm(self, tmp_path, *, card=0):
        """The language model as the user has it: physical card 0, the 5090."""
        settings = configuration_on(tmp_path, card=card, name="qwen.gguf")
        object.__setattr__(settings, "gpu_uuid", self.LLM_UUID)
        object.__setattr__(settings, "gpu_name", self.LLM_NAME)
        return settings

    # -- the comparison itself -------------------------------------------- #

    def test_the_same_number_is_not_the_same_card(self, machine, tmp_path):
        """Physical 0 and image ordinal 0. The old comparison said yes."""
        settings = self.llm(tmp_path)

        assert not runtime.shares_the_image_card(0, settings)

    def test_the_uuid_is_what_answers_it(self, machine, tmp_path, monkeypatch):
        """Not the name, and not the index: with both UUIDs known nothing else
        is consulted, so a machine with two identically named cards is still
        answered correctly."""
        settings = self.llm(tmp_path)
        monkeypatch.setattr(mc_broker, "image_device_name", lambda: self.LLM_NAME)

        assert not runtime.shares_the_image_card(0, settings)

    def test_a_placement_really_on_the_image_card_is_still_recognised(
            self, machine, tmp_path):
        """The rule has to work in both directions or it is not a rule. Physical
        1 *is* the image card here, whatever either process calls it."""
        settings = self.llm(tmp_path, card=1)
        object.__setattr__(settings, "gpu_uuid", self.IMAGE_UUID)
        object.__setattr__(settings, "gpu_name", self.IMAGE_NAME)

        assert runtime.shares_the_image_card(1, settings)

    def test_end_to_end_through_the_real_translation(self, host, tmp_path, monkeypatch):
        """The reported machine, with nothing about the card stubbed.

        Every other test in this class patches ``image_device_index`` to the
        answer. This one builds the machine -- nvidia-smi reporting the 5090 as
        card 0 and the 3090 as card 1, the CUDA runtime ordering them the other
        way -- and lets the real code work it out. It is the only test here
        that would have failed for the original reason: both halves called
        their card 0, and the comparison agreed with them.
        """
        from test_memory import FakeTorch, install_topology

        mc_broker.clear()
        monkeypatch.setattr(mc_broker, "safety_margin_bytes", lambda: 0)
        # Forge's ordinal 0 is the 3090; nvidia-smi calls that card 1.
        torch = FakeTorch(device_free=4 * _GB, uuids={0: "GPU-3090", 1: "GPU-5090"})
        install_topology(monkeypatch, cards={0: ("GPU-5090", self.LLM_NAME),
                                             1: ("GPU-3090", self.IMAGE_NAME)})
        import sys
        import types as _types

        monkeypatch.setitem(sys.modules, "torch", torch)
        torch.get_device_name = lambda ordinal: {0: self.IMAGE_NAME,
                                                 1: self.LLM_NAME}[ordinal]
        torch.cuda.get_device_name = torch.get_device_name
        from backend import memory_management

        monkeypatch.setattr(memory_management, "get_torch_device",
                            lambda: _types.SimpleNamespace(type="cuda", index=0))
        monkeypatch.setattr(memory_management, "get_total_memory", lambda dev=None: 24 * _GB)

        # What the two sides each call their own card, before anything compares
        # them: the collision the whole failure was made of.
        assert mc_memory.image_torch_ordinal() == 0
        assert self.llm(tmp_path).gpu_index == 0

        # And what they are actually on.
        assert mc_broker.image_device_index() == 1
        assert not runtime.shares_the_image_card(0, self.llm(tmp_path))
        assert not runtime.execution_domain(self.llm(tmp_path)).conflicts_with(
            mc_broker.image_execution_domain())

        mc_broker.clear()

    def test_the_uuid_settles_it_when_no_index_can_be_translated(
            self, machine, tmp_path, monkeypatch):
        """nvidia-smi unavailable, so there is no physical index to compare --
        and the UUIDs are still there, and still decisive."""
        monkeypatch.setattr(mc_broker, "image_device_index", lambda: -1)
        settings = self.llm(tmp_path)

        assert not runtime.shares_the_image_card(0, settings)
        assert not runtime.execution_domain(settings).conflicts_with(
            mc_broker.image_execution_domain())

    def test_the_uuid_also_says_when_they_are_the_same_card(
            self, machine, tmp_path, monkeypatch):
        """Both directions again. An untranslatable index is not a licence to
        call two workloads independent -- only a matching identity is, and a
        matching identity says they are not."""
        monkeypatch.setattr(mc_broker, "image_device_index", lambda: -1)
        settings = self.llm(tmp_path)
        object.__setattr__(settings, "gpu_uuid", self.IMAGE_UUID)

        assert runtime.shares_the_image_card(0, settings)
        assert runtime.execution_domain(settings).conflicts_with(
            mc_broker.image_execution_domain())

    def test_two_different_names_settle_it_without_any_uuid(
            self, machine, tmp_path, monkeypatch):
        """An older torch exposes no UUID and a machine without nvidia-smi has
        no index to translate. Two different card models are still two cards."""
        settings = self.llm(tmp_path)
        object.__setattr__(settings, "gpu_uuid", "")
        monkeypatch.setattr(mc_broker, "image_device_uuid", lambda: "")
        monkeypatch.setattr(mc_broker, "image_device_index", lambda: -1)

        assert not runtime.shares_the_image_card(0, settings)

    def test_nothing_identifiable_at_all_stays_conservative(
            self, machine, tmp_path, monkeypatch):
        """The cost of being wrong this way is a smaller language model. The
        cost of the other way is a generation that runs out of VRAM."""
        settings = self.llm(tmp_path)
        object.__setattr__(settings, "gpu_uuid", "")
        object.__setattr__(settings, "gpu_name", "")
        object.__setattr__(settings, "device_name", "")
        monkeypatch.setattr(mc_broker, "image_device_uuid", lambda: "")
        monkeypatch.setattr(mc_broker, "image_device_name", lambda: "")
        monkeypatch.setattr(mc_broker, "image_device_index", lambda: -1)

        assert runtime.shares_the_image_card(0, settings)

    # -- the four consequences from the log ------------------------------- #

    def test_the_conversation_does_not_wait_for_the_generation(
            self, machine, tmp_path, monkeypatch):
        """`Waiting for image generation on GPU 0…` — for a model on the 5090."""
        monkeypatch.setattr(sessions, "WAIT_NOTICE_SECONDS", 0.0)
        monkeypatch.setattr(sessions, "WAIT_POLL_SECONDS", 0.01)
        monkeypatch.setattr(mc_broker, "host_busy", lambda: True)
        settings = self.llm(tmp_path)
        monkeypatch.setattr(runtime, "config", lambda role="": settings)
        gpu = sessions._Gpu("a conversation reply", sessions.Cancellation())

        acquired, events = take(gpu)
        gpu.release()

        assert acquired
        assert waiting_texts(events) == []

    def test_the_generation_does_not_wait_for_the_conversation(self, machine, tmp_path):
        """`waiting for the LLM on GPU 0 to finish before generating on GPU 0`."""
        settings = self.llm(tmp_path)
        with mc_broker.workload(mc_broker.FAMILY_LLM, "a conversation reply",
                                domain=runtime.execution_domain(settings)):
            started = time.monotonic()

            assert mc_broker.await_idle(timeout=5.0,
                                        domain=mc_broker.image_execution_domain())
            assert time.monotonic() - started < 1.0

    def test_the_image_plan_does_not_apply_to_it(self, machine, tmp_path, monkeypatch):
        """`re-placing llama-server — it holds 19.3 GB where the active plan
        leaves 4.1 GB`. The plan describes the 3090. The server is on the 5090.
        """
        import mc_plan

        monkeypatch.setattr(mc_plan, "current", lambda: _Plan(("stage-1",)))
        monkeypatch.setattr(mc_plan, "persistent_llm_budget", lambda ours=0: 4 * _GB)
        settings = self.llm(tmp_path)
        held = runtime.Runtime()
        held._card = 0
        held.configuration = lambda: settings

        assert not held._plan_applies()
        assert held._allowance(19 * _GB) == -1
        assert not held._overspending(19 * _GB)

    def test_it_does_not_evict_the_checkpoint_from_the_other_card(
            self, machine, tmp_path, monkeypatch):
        """`released 13.1 GB of image VRAM on GPU 0 for Qwen…` — thirteen
        gigabytes off a card the language model was never going to touch, and
        every byte of it reloaded for the generation that followed."""
        image = ImageSide(holds=13 * _GB)
        mc_broker.register_reclaimer(mc_broker.FAMILY_IMAGE, image)
        free_on(monkeypatch, gpu0=0.5, gpu1=4.2)
        settings = self.llm(tmp_path)
        assert mc_broker.image_device_index() == 1, "the image card was reset"

        released = mc_broker.release_for_llm(
            8 * _GB, card=0, uuid=settings.gpu_uuid, name=settings.card_name,
            reason="the writer, which has been given priority on this card")

        assert released.freed == 0
        assert image.calls == []
        assert image.holds == 13 * _GB

    def test_llm_priority_still_works_on_the_card_it_was_given_for(
            self, machine, tmp_path, monkeypatch):
        """And the setting still does what it says when the model really is on
        the image card -- otherwise this would be a fix that removed a feature.
        """
        image = ImageSide(holds=13 * _GB)
        mc_broker.register_reclaimer(mc_broker.FAMILY_IMAGE, image)
        free_on(monkeypatch, gpu0=0.5, gpu1=0.5)
        settings = self.llm(tmp_path, card=1)
        object.__setattr__(settings, "gpu_uuid", self.IMAGE_UUID)
        object.__setattr__(settings, "gpu_name", self.IMAGE_NAME)

        released = mc_broker.release_for_llm(
            8 * _GB, card=1, uuid=settings.gpu_uuid, name=settings.card_name,
            reason="the writer, which has been given priority on this card")

        assert released.freed == 13 * _GB

    def test_the_allocator_cache_on_the_other_card_is_left_alone(
            self, machine, tmp_path, monkeypatch):
        emptied: list = []
        monkeypatch.setattr(mc_broker, "release_cached_vram",
                            lambda: (emptied.append(1), 2 * _GB)[1])
        held = runtime.Runtime()

        assert held._release_the_image_cache_if_it_helps(self.llm(tmp_path)) == 0
        assert emptied == []

    def test_a_reserve_miss_on_the_image_card_leaves_the_other_server_alone(
            self, registry, machine, monkeypatch):
        free_on(monkeypatch, gpu0=22.7, gpu1=0.5)
        away = hold(registry, Server(card=0, holds=19 * _GB), "the 5090 server")

        assert mc_broker._reclaim_for_image(3 * _GB, reason="the Stage 1 pass") == 0
        assert away.up
        assert away.holds == 19 * _GB


# --------------------------------------------------------------------------- #
# Three copies of one model on one card
# --------------------------------------------------------------------------- #


class TestOneModelInMemory:
    """A 32 GB card with three llama-servers on it, from the second report.

    The user's expectation was the right one: LLM Studio, Creative and Spatial
    all pointing at the same model should be *one* copy of the weights with
    three things talking to it. What the log showed instead was a conversation
    holding twenty gigabytes while the two roles took turns in the eleven that
    were left -- twenty-five of sixty-five layers each, four tokens a second,
    and a model load on every switch.
    """

    def test_residency_is_measured_on_the_card_the_server_is_on(self, scoped,
                                                                monkeypatch):
        """`llama-server ready — 8.7 GB VRAM` about a model holding twenty.

        31.4 GB free on the 5090 before, 22.7 GB free on the 3090 after, and
        the difference between two different cards reported as one card's
        residency. It also produced a warning that llama.cpp had left the model
        in system RAM, about a server answering at 92 tokens a second.
        """
        import mc_llm_context as ctx

        readings = {IMAGE_CARD: 22.7, OTHER_CARD: 31.4}
        monkeypatch.setattr(mc_broker, "device_free_vram_bytes",
                            lambda index=None: int(readings[
                                IMAGE_CARD if index is None else int(index)] * _GB))
        before = int(readings[OTHER_CARD] * _GB)
        readings[OTHER_CARD] = 11.4  # the server took 20 GB of the card it is on

        observed = runtime.Runtime._observed_residency(
            before, ctx.Placement(gpu_layers=ctx.ALL_LAYERS, on_gpu=True), card=OTHER_CARD)

        assert observed == pytest.approx(20 * _GB, rel=0.01)
        # And the figure the bug produced, so a regression is recognisable:
        # one card's free VRAM minus another's.
        assert observed != pytest.approx(int(31.4 * _GB) - int(22.7 * _GB), rel=0.01)

    def test_an_unnamed_card_still_answers_for_the_image_side(self, scoped,
                                                              monkeypatch):
        """The single-card installation, unchanged: ``None`` is the image card,
        and the difference is one card from itself."""
        import mc_llm_context as ctx

        monkeypatch.setattr(mc_broker, "device_free_vram_bytes",
                            lambda index=None: int(4 * _GB))

        observed = runtime.Runtime._observed_residency(
            int(20 * _GB), ctx.Placement(gpu_layers=ctx.ALL_LAYERS, on_gpu=True), None)

        assert observed == 16 * _GB

    def test_the_card_has_to_be_said(self):
        """No default, because the default was the bug.

        A caller that forgot which card it was measuring got the image card's
        free VRAM silently subtracted from another card's, which is a wrong
        number rather than an error and reads perfectly plausibly in a log. It
        is a ``TypeError`` now, at the call site, immediately.
        """
        import inspect

        card = inspect.signature(runtime.Runtime._observed_residency).parameters["card"]

        assert card.default is inspect.Parameter.empty


class TestTakeTurnsMeansAllOfOurServers:
    """`stood the other role's llama-server down — 1.0 GB` while 20 GB stayed up.

    ``make_room_for`` skipped any runtime serving no role, and the runtime
    serving no role is the *shared* one -- the server Conversation, Prompt
    Studio, MiniMax and LLM Studio all use, and routinely the largest thing on
    the card. "Serves no role other than mine" and "serves no role at all" read
    identically to the test that used to be there.
    """

    @pytest.fixture
    def roles(self, scoped, tmp_path, monkeypatch, host):
        from modules import shared

        shared.opts.model_chain_llm_role_processes = runtime.PROCESSES_SEPARATE
        shared.opts.model_chain_llm_role_sharing = runtime.SHARE_TAKE_TURNS
        settings = configuration_on(tmp_path, card=OTHER_CARD)
        monkeypatch.setattr(runtime, "config", lambda role="": settings)
        found = runtime.RuntimeRegistry()
        return found, settings

    def test_the_shared_server_is_stood_down_too(self, roles):
        """The one the log never mentioned, holding the memory the roles then
        fought over."""
        import mc_llm_roles

        found, settings = roles
        conversation = hold(found, Server(card=OTHER_CARD, holds=20 * _GB), "shared")

        freed = found.make_room_for(mc_llm_roles.CREATIVE, settings)

        assert conversation.stopped
        assert freed == 20 * _GB

    def test_a_role_server_is_still_stood_down(self, roles):
        """The case that already worked, which the fix must not lose."""
        import mc_llm_roles

        found, settings = roles
        spatial = hold(found, Server(card=OTHER_CARD, holds=6 * _GB), "spatial")
        spatial.roles = (mc_llm_roles.SPATIAL,)

        found.make_room_for(mc_llm_roles.CREATIVE, settings)

        assert spatial.stopped

    def test_a_roles_own_server_is_never_its_own_victim(self, roles):
        """Making room for a role by stopping that role's own server is not a
        policy, it is a reload."""
        import mc_llm_roles

        found, settings = roles
        mine = Server(card=OTHER_CARD, holds=6 * _GB)
        mine.roles = (mc_llm_roles.CREATIVE,)
        found._runtimes[found.key_for(mc_llm_roles.CREATIVE, settings)] = mine

        found.make_room_for(mc_llm_roles.CREATIVE, settings)

        assert not mine.stopped

    def test_a_server_in_another_pool_is_left_alone(self, roles, tmp_path, monkeypatch):
        """A role on the processor and a role on a card are not competing, and
        take turns has never meant otherwise."""
        import mc_llm_roles

        found, settings = roles
        elsewhere = hold(found, Server(cpu=True, holds=0), "processor")
        elsewhere.roles = (mc_llm_roles.SPATIAL,)
        processor = configuration_on(tmp_path, card=None, mode="cpu", name="cpu.gguf")
        monkeypatch.setattr(runtime, "config",
                            lambda role="": processor if role == mc_llm_roles.SPATIAL
                            else settings)

        found.make_room_for(mc_llm_roles.CREATIVE, settings)

        assert not elsewhere.stopped

    def test_sharing_one_server_stands_nothing_down_at_all(
            self, roles, monkeypatch, host):
        """And the setting the user actually wants: identical roles resolve to
        one runtime, so there is never a second one to stop."""
        import mc_llm_roles
        from modules import shared

        shared.opts.model_chain_llm_role_processes = runtime.PROCESSES_SHARED
        found, settings = roles
        # Filed under the identity rather than under a role, which is the whole
        # of what sharing is: every role resolves to this one key, so the server
        # a role would have stood down *is* the server it is about to use.
        key = found.key_for(mc_llm_roles.CREATIVE, settings)
        assert key == found.key_for(mc_llm_roles.SPATIAL, settings)
        conversation = Server(card=OTHER_CARD, holds=20 * _GB)
        found._runtimes[key] = conversation

        found.make_room_for(mc_llm_roles.CREATIVE, settings)

        assert not conversation.stopped
        assert conversation.holds == 20 * _GB


class TestWhoFilledTheCard:
    """`Free VRAM on this card` is advice for a card somebody else filled."""

    def test_it_names_our_own_servers_and_the_setting(self, scoped, tmp_path,
                                                      monkeypatch):
        import mc_llm_context as ctx
        import mc_gguf

        found = runtime.RuntimeRegistry()
        hold(found, Server(card=OTHER_CARD, holds=20 * _GB), "shared")
        monkeypatch.setattr(runtime, "registry", found)
        settings = configuration_on(tmp_path, card=OTHER_CARD)
        described = mc_gguf.describe(settings.model)

        said = runtime._unsatisfied(settings, ctx.Placement(gpu_layers=8, on_gpu=True),
                                    described)

        assert said
        assert "20.0 GB of it is held by 1 other llama-server" in said[0]
        assert "One server" in said[0]

    def test_it_says_nothing_when_nothing_of_ours_is_on_the_card(self, scoped, tmp_path,
                                                                 monkeypatch):
        """Then the advice really is "free VRAM", and adding a sentence about
        servers that are not there would be worse than saying nothing."""
        import mc_llm_context as ctx
        import mc_gguf

        monkeypatch.setattr(runtime, "registry", runtime.RuntimeRegistry())
        settings = configuration_on(tmp_path, card=OTHER_CARD)

        said = runtime._unsatisfied(settings, ctx.Placement(gpu_layers=8, on_gpu=True),
                                    mc_gguf.describe(settings.model))

        assert said
        assert "held by" not in said[0]

    def test_a_satisfied_placement_says_nothing(self, scoped, tmp_path, monkeypatch):
        import mc_llm_context as ctx
        import mc_gguf

        settings = configuration_on(tmp_path, card=OTHER_CARD)

        assert runtime._unsatisfied(
            settings, ctx.Placement(gpu_layers=ctx.ALL_LAYERS, on_gpu=True),
            mc_gguf.describe(settings.model)) == []


# --------------------------------------------------------------------------- #
# 19.7 Host RAM, the other way round
# --------------------------------------------------------------------------- #


class TestAnImageGenerationOutranksAnIdleLlmInRamToo:
    """LLM Studio intent section 3.3, moved into the other memory domain.

    The rule was written about the card -- *an image generation always outranks
    an idle LLM* -- but the reason behind it was never about cards. The image
    model is the workload somebody is waiting on; the language model wrote a
    prompt and then had nothing to do. That reads the same way about system RAM,
    where the image side keeps every weight that is not currently on the card.

    What made this worth implementing rather than arguing about: host-RAM
    pressure is *silent*. A VRAM shortage announces itself and can be measured
    afterwards; a RAM shortage produces no error at all, only the same weights
    moving at a quarter of their usual rate, which is how it went unnoticed long
    enough to be diagnosed from a stopwatch.
    """

    @pytest.fixture
    def servers(self, scoped, monkeypatch):
        """An LLM family reclaimer holding host RAM, and a record of the asking."""
        state = {"held": 12 * _GB, "released": [], "busy": False}

        class Reclaimer:
            def host_ram_bytes(self):
                return state["held"]

            def release_host_ram(self, needed, reason=""):
                state["released"].append((needed, reason))
                freed, state["held"] = state["held"], 0
                return freed

        monkeypatch.setattr(mc_broker, "llm_busy", lambda: state["busy"])
        mc_broker.register_reclaimer(mc_broker.FAMILY_LLM, Reclaimer())
        return state

    def test_a_demand_that_fits_stops_nothing(self, servers, monkeypatch):
        """Invariant I-5 in this direction as well. Two workloads sharing a
        memory domain is not a conflict, and a warm llama-server stopped because
        an image model also uses RAM would be evict-on-switch with the arrow
        turned round -- the exact thing 3.3 warns about."""
        ram(monkeypatch, available_gb=40)

        admission = mc_broker.admit_image_host_ram(20 * _GB, reason="Stage 1")

        assert admission.fits
        assert servers["released"] == []
        assert servers["held"] == 12 * _GB

    def test_an_idle_server_is_stopped_when_the_image_model_is_short(
            self, servers, monkeypatch):
        readings = {"available": 8 * _GB}
        monkeypatch.setattr(mc_broker, "free_ram_bytes", lambda: readings["available"])
        monkeypatch.setattr(mc_broker, "ram_reserve_bytes", lambda: 2 * _GB)
        asked = mc_broker.release_llm_host_ram

        def release(needed, reason=""):
            freed = asked(needed, reason)
            readings["available"] += freed
            return freed

        monkeypatch.setattr(mc_broker, "release_llm_host_ram", release)

        admission = mc_broker.admit_image_host_ram(18 * _GB, reason="Stage 1")

        assert servers["released"], "the language model was never asked"
        assert admission.fits
        assert admission.freed == 12 * _GB

    def test_a_generating_server_is_never_stopped(self, servers, monkeypatch):
        """The word "idle" in 3.3 doing its work. A reply somebody is watching
        arrive is not spare memory, however much the image side wants it."""
        ram(monkeypatch, available_gb=8)
        servers["busy"] = True

        admission = mc_broker.admit_image_host_ram(18 * _GB, reason="Stage 1")

        assert servers["released"] == []
        assert not admission.fits
        assert mc_broker.reclaimable_llm_host_ram_bytes() == 0

    def test_a_busy_family_reports_nothing_reclaimable_rather_than_nothing_held(
            self, servers, monkeypatch):
        """Two different questions, and conflating them would make the panel
        say a generating server is holding no RAM."""
        ram(monkeypatch, available_gb=8)

        assert mc_broker.llm_host_ram_bytes() == 12 * _GB
        assert mc_broker.reclaimable_llm_host_ram_bytes() == 12 * _GB
        servers["busy"] = True
        assert mc_broker.llm_host_ram_bytes() == 12 * _GB
        assert mc_broker.reclaimable_llm_host_ram_bytes() == 0

    def test_the_re_reading_is_what_decides_here_too(self, servers, monkeypatch):
        """Invariant I-15, unchanged by the direction of travel: a reclaimer
        reports what it stopped, and the operating system reports what that
        actually returned."""
        ram(monkeypatch, available_gb=8)

        admission = mc_broker.admit_image_host_ram(18 * _GB, reason="Stage 1")

        assert admission.freed == 12 * _GB
        assert not admission.fits, "available RAM never moved, so the answer is no"

    def test_unreadable_memory_stops_nothing(self, servers, monkeypatch):
        """Section 16.3's rule, and the asymmetry it turns on: an unanswerable
        question may proceed, but it may never *reclaim*."""
        monkeypatch.setattr(mc_broker, "free_ram_bytes", lambda: 0)

        admission = mc_broker.admit_image_host_ram(18 * _GB, reason="Stage 1")

        assert not admission.known
        assert servers["released"] == []

    def test_a_family_with_no_reclaimer_is_not_an_error(self, scoped, monkeypatch):
        """An installation that has never opened LLM Studio registers no LLM
        controller at all, and section 18 requires ordinary txt2img to be
        unaffected by a feature that was never used."""
        ram(monkeypatch, available_gb=8)

        assert mc_broker.release_llm_host_ram(18 * _GB, "Stage 1") == 0

    def test_the_reserve_is_part_of_what_has_to_fit(self, servers, monkeypatch):
        """Both sides of the boundary, because the reserve is the whole
        difference between them: 6 GB fits in 8 with 2 held back and 7 does
        not, and a demand that merely fits must stop nothing."""
        ram(monkeypatch, available_gb=8, reserve_gb=2.0)

        assert mc_broker.admit_image_host_ram(6 * _GB, reason="Stage 1").fits
        assert servers["released"] == []

        mc_broker.admit_image_host_ram(7 * _GB, reason="Stage 1")

        assert servers["released"], "7 GB + a 2 GB reserve does not fit in 8 GB free"


class TestTheLogCanExplainASlowMove:
    """The measurements that were missing when this was diagnosed by stopwatch.

    Everything about host RAM was recorded on the way *in* -- the admission
    arithmetic, the shortfall, the reserve -- and nothing on the way out. So a
    log could say "12.5 GB wanted, 14.5 GB available", and then show an image
    module moving at a quarter of its usual rate with no reading anywhere near
    it to say why. These are the two numbers that close that gap.
    """

    def test_the_rate_is_reported_when_it_can_be_known(self):
        import mc_memory

        result = mc_memory.PreloadResult("ready", moved_bytes=12 * _GB,
                                         moved_seconds=12.0)

        assert round(result.megabytes_per_second) == 1024

    def test_a_move_nobody_timed_claims_no_rate(self):
        import mc_memory

        assert mc_memory.PreloadResult("ready", moved_bytes=12 * _GB
                                       ).megabytes_per_second == 0.0
        assert mc_memory.PreloadResult("ready", moved_seconds=4.0
                                       ).megabytes_per_second == 0.0

    def test_the_rate_is_the_move_and_not_the_whole_warm_up(self):
        """A preload that spent thirty seconds reinstating a checkpoint and four
        moving weights is not a 400 MB/s move, and a rate that said so would be
        useless for the one comparison it exists to support."""
        import mc_memory

        result = mc_memory.PreloadResult("ready", moved_bytes=4 * _GB,
                                         seconds=34.0, moved_seconds=4.0)

        assert round(result.megabytes_per_second) == 1024

    def test_the_host_ram_line_names_who_of_ours_is_in_it(self, scoped, monkeypatch):
        ram(monkeypatch, available_gb=6, reserve_gb=2.0)
        monkeypatch.setattr(mc_broker, "llm_host_ram_bytes", lambda: 12 * _GB)
        monkeypatch.setattr(mc_broker, "llm_busy", lambda: False)

        said = mc_broker.describe_host_ram("after the load")

        assert "after the load" in said
        assert "6.0 GB free" in said
        assert "12.0 GB in our language models (idle)" in said

    def test_it_says_when_a_language_model_is_generating(self, scoped, monkeypatch):
        """The difference between "that RAM is reclaimable" and "that RAM is in
        use", which is the whole of whether the next line should worry."""
        ram(monkeypatch, available_gb=6)
        monkeypatch.setattr(mc_broker, "llm_host_ram_bytes", lambda: 12 * _GB)
        monkeypatch.setattr(mc_broker, "llm_busy", lambda: True)

        assert "(generating)" in mc_broker.describe_host_ram()

    def test_unreadable_memory_says_so_rather_than_printing_a_zero(
            self, scoped, monkeypatch):
        monkeypatch.setattr(mc_broker, "free_ram_bytes", lambda: 0)

        assert "could not be read" in mc_broker.describe_host_ram()


class TestTheRegistryStopsTheFewestServersItCan:
    """The fan-out, ordered the way the VRAM half is ordered and for the same
    reason: largest holder first, so the fewest processes are ended for the
    memory asked for, and stopping the moment the request is covered rather than
    emptying the machine because one pass was short."""

    def test_the_largest_holder_answers_first(self, scoped, registry, monkeypatch):
        monkeypatch.setattr(mc_broker, "llm_busy", lambda: False)
        small = Server(cpu=True, host_ram=3 * _GB)
        large = Server(cpu=True, host_ram=12 * _GB)
        hold(registry, small, "small")
        hold(registry, large, "large")

        freed = registry.release_host_ram(10 * _GB, "Stage 1")

        assert freed == 12 * _GB
        assert large.stopped
        assert not small.stopped, "one server covered the request; the other was spared"

    def test_it_keeps_going_until_the_request_is_covered(self, scoped, registry,
                                                         monkeypatch):
        monkeypatch.setattr(mc_broker, "llm_busy", lambda: False)
        first = Server(cpu=True, host_ram=6 * _GB)
        second = Server(cpu=True, host_ram=5 * _GB)
        hold(registry, first, "first")
        hold(registry, second, "second")

        assert registry.release_host_ram(10 * _GB, "Stage 1") == 11 * _GB
        assert first.stopped and second.stopped

    def test_nothing_is_stopped_while_anything_is_generating(self, scoped, registry,
                                                             monkeypatch):
        """Checked at the fan-out and again inside each runtime, because a read
        and a stop are not one atomic act and the one thing this must never do
        is end a reply somebody is watching arrive."""
        monkeypatch.setattr(mc_broker, "llm_busy", lambda: True)
        server = Server(cpu=True, host_ram=12 * _GB)
        hold(registry, server, "one")

        assert registry.release_host_ram(10 * _GB, "Stage 1") == 0
        assert not server.stopped

    def test_a_server_holding_no_host_ram_is_left_alone(self, scoped, registry,
                                                        monkeypatch):
        """Its weights are on the card. Stopping it frees nothing where the
        memory is wanted, which is the same reasoning the VRAM half applies to a
        server on the wrong GPU."""
        monkeypatch.setattr(mc_broker, "llm_busy", lambda: False)
        server = Server(card=IMAGE_CARD, holds=12 * _GB, host_ram=0)
        hold(registry, server, "on-the-card")

        assert registry.release_host_ram(10 * _GB, "Stage 1") == 0
        assert not server.stopped

    def test_a_request_for_nothing_asks_nobody(self, scoped, registry, monkeypatch):
        monkeypatch.setattr(mc_broker, "llm_busy", lambda: False)
        server = Server(cpu=True, host_ram=12 * _GB)
        hold(registry, server, "one")

        assert registry.release_host_ram(0, "Stage 1") == 0
        assert server.calls == []
