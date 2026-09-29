"""Whose turn it is on each card: every rule :mod:`mc_turns` adds, with the case it exists for.

The user's rules, which the whole file is arranged around (docs/23-voice-box.md,
section 4.1):

    1. A running job is never cut off.
    2. A VibeVoice request goes next, ahead of every job on its card that has
       not started.
    3. All VibeVoice requests first.
    4. A render runs to its end in one generation.
    5. One render per card.
    6. Between requests VibeVoice stays warm, and leaves -- only while idle --
       when an image job, WanGP or the language model needs the card, or when
       system RAM runs low under a parked image model.
    7. What made room comes back, from system RAM or from its recipe.
    8. Never the pagefile.

Nothing here has a GPU. A card is a number of free bytes that a load takes and
an unload gives back, system RAM is another, and the speech guest is a double
with the three methods the contract gives a guest. The card's driver is stepped
by hand (``tick``) everywhere except the last class, which runs the real thread
end to end: a turn system whose rules held only when a test stepped it would be
one nobody had seen work.
"""

from __future__ import annotations

import threading
import time
import types

import pytest

import mc_arm
import mc_broker
import mc_memory
import mc_plan
import mc_turns
import mc_wangp
import mc_llm_runtime as runtime
import mc_llm_sessions as sessions
from test_orchestration import DEFAULTS, UI_ORDER, make_p
from test_resource_scope import configuration_on

_GB = 1024**3

IMAGE_CARD = 0
"""Forge's card, the 3090 of the report."""

OTHER_CARD = 1
"""WanGP's card, the 5090 of the report."""

IMAGE_UUID = "GPU-99998888-7777-6666-5555-444433332222"
WANGP_UUID = "GPU-11112222-3333-4444-5555-666677778888"
NAMES = {IMAGE_CARD: "NVIDIA GeForce RTX 3090", OTHER_CARD: "NVIDIA GeForce RTX 5090"}

GUEST = "VibeVoice"

_DRIVER = mc_turns._ensure_driver
"""The real driver start, kept before any fixture replaces it."""


def key(uuid: str) -> str:
    return mc_memory._uuid_key(uuid)


# --------------------------------------------------------------------------- #
# Doubles
# --------------------------------------------------------------------------- #


class Machine:
    """Two cards and one pool of system RAM, in bytes, every figure mutable.

    ``log`` is one ordered record across every double, which is what lets a test
    say "evicted, *then* released" rather than only "both happened".
    """

    def __init__(self):
        self.free = {IMAGE_CARD: 4 * _GB, OTHER_CARD: 30 * _GB}
        self.ram = 40 * _GB
        self.image = int(18.4 * _GB)
        """What the image model holds on Forge's card."""
        self.can_drop = True
        self.moved: list[tuple[str, str]] = []
        self.armed: list[str] = []
        self.loadable = 0
        self.joined = 0
        self.fresh: list[bool] = []
        self.sleeps = 0
        self.log: list[str] = []

    def index(self, uuid: str) -> int:
        return {key(IMAGE_UUID): IMAGE_CARD, key(WANGP_UUID): OTHER_CARD}[key(uuid)]

    def take(self, uuid: str, amount: int) -> None:
        self.free[self.index(uuid)] -= amount

    def give(self, uuid: str, amount: int) -> None:
        self.free[self.index(uuid)] += amount

    def park(self, reason: str = "") -> str:
        """Forge's own unload: the same model, moved to system RAM."""
        self.moved.append((mc_memory.PARKED_RAM, reason))
        self.log.append("park")
        self.free[IMAGE_CARD] += self.image
        self.ram -= self.image
        self.image = 0
        return mc_memory.PARKED_RAM

    def drop(self, reason: str = "") -> str:
        """Forge's Unload button: the model forgotten, its recipe kept."""
        self.moved.append((mc_memory.PARKED_RECIPE, reason))
        self.log.append("drop")
        self.free[IMAGE_CARD] += self.image
        self.image = 0
        return mc_memory.PARKED_RECIPE

    def loaded(self) -> bool:
        self.loadable += 1
        return False

    def join(self, timeout=None) -> None:
        self.joined += 1

    def arm(self, width=0, height=0, *, reason=""):
        self.armed.append(reason)
        self.log.append("arm")

    def no_sleep(self, seconds) -> None:
        self.sleeps += 1
        if self.sleeps > 2000:
            raise AssertionError("waited for ever: nothing moved the turn on")


class Guest:
    """A speech engine as the turn system sees it: memory per card, working or not.

    ``lingers`` is how many looks its memory outlives an eviction by -- a worker
    process letting go of eighteen gigabytes does not do it in the same instant
    it is asked to.
    """

    def __init__(self, machine: Machine, size_gb: float = 18.0, lingers: int = 0):
        self.machine = machine
        self.size = int(size_gb * _GB)
        self.lingers = lingers
        self.held: dict[str, int] = {}
        self.leaving: dict[str, int] = {}
        self.working = False
        self.evictions: list[str] = []

    def load(self, uuid: str) -> None:
        if not self.held.get(key(uuid)):
            self.held[key(uuid)] = self.size
            self.machine.take(uuid, self.size)
            self.machine.log.append("load")

    def resident_bytes(self, uuid: str) -> int:
        found = key(uuid)
        if found in self.leaving:
            if self.leaving[found] > 0:
                self.leaving[found] -= 1
            else:
                del self.leaving[found]
                self.machine.give(uuid, self.held.pop(found, 0))
                self.machine.log.append("gone")
        return self.held.get(found, 0)

    def rendering(self, uuid: str) -> bool:
        return self.working

    def evict(self, uuid: str, reason: str) -> int:
        self.evictions.append(reason)
        self.machine.log.append("evict")
        self.leaving[key(uuid)] = self.lingers
        return self.held.get(key(uuid), 0)


class MiniPaint:
    """Mini Paint NEO's card lease as the contract describes it (docs/23 section 6.1).

    The phase each call answers is the test's to set, which is the point: the
    turn system must do the right thing for every phase Mini Paint can report,
    in whatever order it reports them. A flush is asynchronous, as Mini Paint's
    is: ``flush`` answers ``holding`` at once, and the flush is done -- the
    memory given back, ``held`` again -- only after ``settle_after`` more looks,
    the way its executor does it on its own next pass.
    """

    def __init__(self, machine: Machine, presence: dict):
        self.machine = machine
        self.presence = presence
        """What Mini Paint's presence report says, shared with the presence seam."""
        self.phase = "pending"
        self.reason = "waiting for WanGP to finish its job"
        self.wanted = False
        self.bridge_hold = "holding"
        self.lease = ""
        self.calls: list[tuple] = []
        self.flushes = {"soft": 0.0, "hard": 0.0}
        """Gigabytes each flush level gives back on WanGP's card."""
        self.settle_after = 1
        self.flush_word = ""
        """What Mini Paint says when a flush settles; its own sentence when empty."""
        self._flushing = ""
        self._looks = 0

    @property
    def running(self) -> bool:
        return bool(self.presence.get("running"))

    @running.setter
    def running(self, value: bool) -> None:
        self.presence["running"] = bool(value)
        mc_wangp.forget_presence()

    def _record(self) -> dict:
        return {"version": 1, "lease": self.lease, "owner": mc_turns.LEASE_OWNER,
                "phase": self.phase, "reason": self.reason, "card_uuid": WANGP_UUID,
                "wanted": self.wanted, "bridge_hold": self.bridge_hold, "expires_in_s": 20.0,
                "wangp": {"running": self.running, "task_running": False, "queue_length": 0,
                          "jobs_waiting": 0, "vram_free_bytes": None,
                          "vram_total_bytes": None}}

    def request(self, owner, *, purpose="", need_bytes=0):
        self.calls.append(("request", owner, purpose, need_bytes))
        self.lease = self.lease or "lease-1"
        return self._record()

    def state(self, lease):
        self.calls.append(("state", lease))
        if self._flushing:
            if self._looks < self.settle_after:
                self._looks += 1
                return self._record()
            level, self._flushing = self._flushing, ""
            self.machine.free[OTHER_CARD] += int(self.flushes.get(level, 0.0) * _GB)
            self.machine.log.append(f"flushed {level}")
            self.phase = "held"
            self.reason = self.flush_word or f"{level.capitalize()} flush done."
        return self._record()

    def flush(self, lease, level):
        self.calls.append(("flush", level))
        self.machine.log.append(f"flush {level}")
        self._flushing, self._looks = level, 0
        self.phase = "holding"
        self.reason = f"WanGP's weights are being moved off its card ({level} flush)."
        return self._record()

    def release(self, lease, *, reason=""):
        self.calls.append(("release", lease))
        self.machine.log.append("release")
        self.phase = "released"
        return self._record()

    def report(self):
        return self._record()["wangp"]

    def said(self, name: str) -> list[tuple]:
        return [call for call in self.calls if call[0] == name]


# --------------------------------------------------------------------------- #
# Fixtures and steps
# --------------------------------------------------------------------------- #


@pytest.fixture
def machine(host, monkeypatch):
    """Forge on the 3090 with an 18.4 GB checkpoint on it, the 5090 empty, no Mini Paint.

    The card's driver is stepped by hand: ``request`` queues a turn and nothing
    moves it until a test calls :func:`step`.
    """
    mc_broker.clear()
    mc_turns.forget()
    mc_wangp.forget()
    found = Machine()
    topology = {"by_uuid": {key(IMAGE_UUID): IMAGE_CARD, key(WANGP_UUID): OTHER_CARD},
                "ordinals": {IMAGE_CARD: 0, OTHER_CARD: 1}, "names": dict(NAMES), "count": 2}
    monkeypatch.setattr(mc_memory, "_cards", lambda: topology)
    # What mc_memory reports for Forge's card: torch's bare digits.
    monkeypatch.setattr(mc_broker, "image_device_uuid", lambda: key(IMAGE_UUID))
    monkeypatch.setattr(mc_broker, "image_device_index", lambda: IMAGE_CARD)
    monkeypatch.setattr(mc_broker, "image_device_name", lambda: NAMES[IMAGE_CARD])
    monkeypatch.setattr(mc_broker, "safety_margin_bytes", lambda: 0)

    def free(index=None):
        return found.free[IMAGE_CARD if index is None else int(index)]

    def physical(index, fresh=False):
        found.fresh.append(bool(fresh))
        return found.free[int(index)]

    monkeypatch.setattr(mc_broker, "device_free_vram_bytes", free)
    monkeypatch.setattr(mc_broker, "free_vram_bytes", lambda: found.free[IMAGE_CARD])
    monkeypatch.setattr(mc_memory, "device_free_vram_bytes", free)
    monkeypatch.setattr(mc_memory, "physical_free_vram_bytes", physical)
    monkeypatch.setattr(mc_memory, "free_ram_bytes", lambda: found.ram)
    monkeypatch.setattr(mc_memory, "resident_vram_bytes", lambda: found.image)
    monkeypatch.setattr(mc_memory, "release_cached_vram", lambda: 0)
    monkeypatch.setattr(mc_memory, "park_image_model", found.park)
    monkeypatch.setattr(mc_memory, "drop_image_model", found.drop)
    monkeypatch.setattr(mc_memory, "can_drop_image_model", lambda: found.can_drop)
    monkeypatch.setattr(mc_memory, "ensure_model_loadable", found.loaded)
    monkeypatch.setattr(mc_memory, "join_preload", found.join)
    monkeypatch.setattr(mc_arm, "arm_later", found.arm)
    monkeypatch.setattr(mc_turns, "_sleep", found.no_sleep)
    monkeypatch.setattr(mc_turns, "_ensure_driver", lambda target: None)
    mc_wangp.use_source(lambda: None)
    yield found
    mc_turns.forget()
    mc_wangp.use_lease(None)
    mc_wangp.use_source(None)
    mc_wangp.forget()
    mc_broker.clear()


@pytest.fixture
def wangp(machine):
    """Mini Paint present, WanGP running on the 5090, and a lease to ask it for."""
    report = {"version": 1, "available": True, "configured": True, "gpu_uuid": WANGP_UUID,
              "state": "READY", "running": True, "instance_id": "abc123", "generating": None}
    mc_wangp.use_source(lambda: dict(report))
    mini = MiniPaint(machine, report)
    mc_wangp.use_lease(mini)
    return mini


def speaker(machine, **kw) -> Guest:
    found = Guest(machine, **kw)
    mc_turns.register_guest(GUEST, found)
    return found


def ask(uuid: str = IMAGE_UUID, need: float = 18.0, ram: float = 2.0,
        label: str = "VibeVoice 7B") -> mc_turns.Turn:
    return mc_turns.request(GUEST, uuid, need_vram=int(need * _GB), need_ram=int(ram * _GB),
                            label=label)


def card_of(turn: mc_turns.Turn) -> mc_turns._CardState:
    return mc_turns._state(turn.card.uuid)


def step(turn: mc_turns.Turn, times: int = 1) -> str:
    for _ in range(times):
        mc_turns.tick(card_of(turn))
    return turn.phase


def until(turn: mc_turns.Turn, *phases: str, limit: int = 12) -> str:
    for _ in range(limit):
        if turn.phase in phases:
            return turn.phase
        step(turn)
    assert turn.phase in phases, turn.describe()
    return turn.phase


def render(turn: mc_turns.Turn, guest: Guest, *, keep_warm=None) -> None:
    """Granted, loaded, finished and concluded: one whole request."""
    assert until(turn, mc_turns.GRANTED, mc_turns.BLOCKED) == mc_turns.GRANTED, turn.describe()
    guest.load(turn.card.uuid)
    turn.finish(keep_warm)
    step(turn)
    assert turn.phase == mc_turns.DONE, turn.describe()


def choose(host, name: str, value: str) -> None:
    host.shared.opts.set(name, value)


def queue_lock():
    from modules import call_queue

    return call_queue.queue_lock


def lock_is_free() -> bool:
    lock = queue_lock()
    if lock.acquire(False):
        lock.release()
        return True
    return False


# --------------------------------------------------------------------------- #
# Rule 2: the gate holds the image job that has not started
# --------------------------------------------------------------------------- #


class TestTheImageGate:
    def test_a_generation_with_no_turn_anywhere_costs_one_dictionary_check(
            self, machine, monkeypatch):
        """Phase 1a has no guest in it: every generation passes the gate, and
        the gate must be nothing at all until a guest has asked for a card."""
        asked = []
        monkeypatch.setattr(mc_broker, "image_device_uuid", lambda: asked.append(1) or "x")

        assert mc_turns.image_gate(object()) is False
        assert asked == []
        assert machine.joined == 0

    def test_a_generation_waits_for_the_turn_that_is_next_on_its_card(
            self, machine, host, monkeypatch):
        """Rule 2. The job is already inside Forge's queue lock, so the turn
        goes ahead of it without taking the lock -- and the job starts only
        once the render is over."""
        guest = speaker(machine)
        turn = ask()
        monkeypatch.setattr(host.progress, "current_task", "task(1)")
        host.shared.state.job = "txt2img"
        seen, notices = [], []

        def driver(seconds):
            notices.append(host.shared.state.textinfo)
            step(turn)
            seen.append(turn.phase)
            if turn.phase == mc_turns.GRANTED:
                guest.load(IMAGE_UUID)
                turn.finish(keep_warm=False)

        monkeypatch.setattr(mc_turns, "_sleep", driver)

        assert mc_turns.image_gate(object()) is True
        assert turn.phase == mc_turns.DONE
        assert mc_turns.GRANTED in seen
        assert card_of(turn).queue_lock is None, "the held job already holds the lock"
        assert notices[0] == mc_turns.GATE_NOTICE.format(card=NAMES[IMAGE_CARD])
        assert host.shared.state.textinfo is None
        assert machine.loadable == 1

    def test_the_gate_is_called_twice_a_job_and_holds_it_once(self, machine, host, monkeypatch):
        """Model Chain's hook and the gate script both call it on txt2img. The
        second call of the same job must not wait for a turn queued in between."""
        guest = speaker(machine)
        monkeypatch.setattr(host.progress, "current_task", "task(7)")
        first = ask()

        def driver(seconds):
            step(first)
            if first.phase == mc_turns.GRANTED:
                guest.load(IMAGE_UUID)
                first.finish(keep_warm=False)

        monkeypatch.setattr(mc_turns, "_sleep", driver)
        assert mc_turns.image_gate(object()) is True

        ask()
        monkeypatch.setattr(mc_turns, "_sleep", machine.no_sleep)
        before = machine.sleeps
        assert mc_turns.image_gate(object()) is False
        assert machine.sleeps == before

    def test_interrupt_is_honoured_after_the_render_and_never_cuts_the_wait(
            self, machine, host, monkeypatch):
        """Let go early, the job would still load its checkpoint -- and encode
        its img2img input -- on top of a render rule 1 says may not be cut off."""
        guest = speaker(machine)
        turn = ask()
        host.shared.state.interrupted = True
        notices = []

        def driver(seconds):
            notices.append(host.shared.state.textinfo)
            step(turn)
            if turn.phase == mc_turns.GRANTED:
                guest.load(IMAGE_UUID)
                turn.finish(keep_warm=False)

        monkeypatch.setattr(mc_turns, "_sleep", driver)
        mc_turns.image_gate(object())

        assert turn.phase == mc_turns.DONE
        assert notices and all(text.startswith("Interrupted") for text in notices)

    def test_the_gate_never_marks_the_host_busy_itself(self, machine, host, monkeypatch):
        """A hook that sets ``state.job`` or ``job_count`` blocks every request
        that waits for the host to go idle -- the turn's own clearing step among
        them, which would then wait for the job that is waiting for it."""
        guest = speaker(machine)
        turn = ask()

        def driver(seconds):
            assert host.shared.state.job == "" and host.shared.state.job_count == 0
            step(turn)
            if turn.phase == mc_turns.GRANTED:
                guest.load(IMAGE_UUID)
                turn.finish(keep_warm=False)

        monkeypatch.setattr(mc_turns, "_sleep", driver)

        assert mc_turns.image_gate(object()) is True
        assert turn.phase == mc_turns.DONE

    def test_a_warm_guest_leaves_for_a_generation_which_loads_its_own_model(
            self, machine, host):
        """Rule 6: the image job evicts the warm guest. The model is not warmed
        back beside it -- the job loads it, and two loads of one model at once
        is what the return must never cause."""
        guest = speaker(machine)
        turn = ask()
        render(turn, guest)
        assert card_of(turn).warm == GUEST

        assert mc_turns.image_gate(object()) is True
        assert guest.evictions == ["an image generation needs the card"]
        assert machine.armed == []
        assert card_of(turn).warm == "" and card_of(turn).parked == ""
        assert machine.loadable == 1

    def test_a_generation_waits_out_a_warm_up_still_loading_the_model(self, machine):
        """A turn that ended with nobody waiting brings the model back on a
        thread of its own; a job arriving meanwhile joins it rather than
        starting a second load beside it, which img2img would otherwise do."""
        guest = speaker(machine)
        render(ask(), guest, keep_warm=False)
        assert machine.armed

        mc_turns.image_gate(object())
        assert machine.joined == 1

    def test_a_turn_on_another_card_holds_no_generation(self, machine, host):
        speaker(machine)
        ask(WANGP_UUID)

        assert mc_turns.image_gate(object()) is False
        assert machine.sleeps == 0


# --------------------------------------------------------------------------- #
# Rule 1: the job that has started finishes
# --------------------------------------------------------------------------- #


class TestClearing:
    def test_a_turn_waits_for_the_generation_already_running(self, machine, host, monkeypatch):
        """A batch of ten is one job, and it finishes first."""
        speaker(machine)
        monkeypatch.setattr(host.progress, "current_task", "task(3)")
        host.shared.state.job = "txt2img"
        turn = ask()

        assert step(turn, 3) == mc_turns.CLEARING
        assert "image generation" in turn.reason
        assert machine.moved == []

        monkeypatch.setattr(host.progress, "current_task", None)
        host.shared.state.job = ""
        assert step(turn) == mc_turns.MAKING_ROOM

    def test_a_generation_between_two_calls_still_counts_as_running(
            self, machine, host, monkeypatch):
        """``shared.state`` is cleared in the ``finally`` of any wrapped call
        and can read idle mid-job; the progress module's task is the other half."""
        speaker(machine)
        monkeypatch.setattr(host.progress, "current_task", "task(4)")
        turn = ask()

        assert step(turn, 2) == mc_turns.CLEARING

    def test_the_queue_lock_is_taken_without_queueing_for_it(self, machine, host):
        """Queueing on a FIFO lock would put VibeVoice behind every job already
        waiting on it, which is rule 2 broken. So it is taken only when free,
        and a step never blocks on it."""
        speaker(machine)
        lock = queue_lock()
        lock.acquire()  # Extras, say: host work the gate never sees
        turn = ask()
        stepped = threading.Thread(target=step, args=(turn, 2), daemon=True)
        stepped.start()
        stepped.join(2.0)

        assert not stepped.is_alive(), "a step blocked on the WebUI's queue lock"
        assert turn.phase == mc_turns.CLEARING
        assert "current job" in turn.reason

        lock.release()
        assert step(turn) == mc_turns.MAKING_ROOM
        assert card_of(turn).queue_lock is lock
        assert not lock_is_free()

    def test_the_queue_lock_goes_back_when_the_turn_ends(self, machine, host):
        guest = speaker(machine)
        render(ask(), guest, keep_warm=False)

        assert lock_is_free()

    def test_the_queue_lock_goes_back_when_the_guest_stays_warm(self, machine, host):
        """Warm is idle. Host work may run beside an idle guest, and anything
        that needs the card evicts it on its way in."""
        guest = speaker(machine)
        turn = ask()
        render(turn, guest)

        assert card_of(turn).warm == GUEST
        assert lock_is_free()

    def test_a_running_llm_turn_on_the_card_finishes_first(self, machine):
        speaker(machine)
        turn = ask(WANGP_UUID)
        with mc_broker.workload(mc_broker.FAMILY_LLM, "a conversation reply",
                                domain=mc_broker.cuda_execution(OTHER_CARD)):
            assert step(turn, 2) == mc_turns.CLEARING
            assert "a conversation reply" in turn.reason
        assert step(turn) == mc_turns.MAKING_ROOM

    def test_an_llm_turn_on_the_other_card_is_not_waited_for(self, machine):
        speaker(machine)
        turn = ask(WANGP_UUID)
        with mc_broker.workload(mc_broker.FAMILY_LLM, "a conversation reply",
                                domain=mc_broker.cuda_execution(IMAGE_CARD)):
            assert until(turn, mc_turns.GRANTED) == mc_turns.GRANTED

    def test_an_llm_turn_that_slips_in_while_clearing_is_waited_for_before_anything_stops(
            self, machine, monkeypatch):
        """Krea's writer, inside the host's own job, does not wait for a turn that
        is still clearing -- so one can begin in the gap. Making room waits for it
        rather than stopping a llama-server under a reply."""
        speaker(machine)
        stopped = []
        monkeypatch.setattr(mc_turns, "_stop_idle_llm", lambda target: stopped.append(1))
        turn = ask(WANGP_UUID)
        assert step(turn) == mc_turns.MAKING_ROOM
        with mc_broker.workload(mc_broker.FAMILY_LLM, "Krea's writer",
                                domain=mc_broker.cuda_execution(OTHER_CARD)):
            assert step(turn, 2) == mc_turns.MAKING_ROOM
            assert "Krea's writer" in turn.reason
            assert stopped == []
        assert step(turn) == mc_turns.GRANTED
        assert stopped == [1]

    def test_the_parked_job_is_the_one_job_the_turn_goes_ahead_of(self, machine, host,
                                                                 monkeypatch):
        """Threads, as the host has them: the generation is inside the queue lock
        and blocked at the gate on its own thread, and the card's driver runs on
        another. The turn is granted while the job waits, and the job goes on
        afterwards with nothing held by the turn."""
        guest = speaker(machine)
        monkeypatch.setattr(mc_turns, "_sleep", lambda seconds: time.sleep(0.005))
        monkeypatch.setattr(host.progress, "current_task", "task(9)")
        host.shared.state.job = "txt2img"
        turn = ask()
        lock = queue_lock()
        passed = []

        def generation():
            with lock:
                passed.append(mc_turns.image_gate(object()))
                passed.append(turn.phase)

        job = threading.Thread(target=generation, daemon=True)
        job.start()
        deadline = time.monotonic() + 5.0
        while card_of(turn).parked_job is None:
            assert time.monotonic() < deadline, "the gate never held the job"
            time.sleep(0.005)

        assert until(turn, mc_turns.GRANTED) == mc_turns.GRANTED
        assert passed == [], "the job went on while the turn held its card"
        guest.load(IMAGE_UUID)
        turn.finish(keep_warm=False)
        step(turn)
        job.join(5.0)

        assert passed == [True, mc_turns.DONE]
        assert card_of(turn).queue_lock is None
        assert lock_is_free()


# --------------------------------------------------------------------------- #
# Rule 3: all VibeVoice first
# --------------------------------------------------------------------------- #


class TestAllVibeVoiceFirst:
    def test_consecutive_requests_run_before_a_waiting_generation(
            self, machine, host, monkeypatch):
        guest = speaker(machine)
        first, second = ask(label="line one"), ask(label="line two")
        order = []

        def driver(seconds):
            for turn in (first, second):
                if turn.phase not in mc_turns.FINAL:
                    step(turn)
                    break
            for turn in (first, second):
                if turn.phase == mc_turns.GRANTED and turn.label not in order:
                    order.append(turn.label)
                    guest.load(IMAGE_UUID)
                    turn.finish()

        monkeypatch.setattr(mc_turns, "_sleep", driver)
        mc_turns.image_gate(object())

        assert order == ["line one", "line two"]
        assert first.phase == second.phase == mc_turns.DONE

    def test_the_card_stays_held_between_them(self, machine, host):
        """Nothing is handed back in the gap: no model warmed up, no queue lock
        released, and the guest still loaded for the second request."""
        guest = speaker(machine)
        first, second = ask(), ask()
        render(first, guest)

        state = card_of(first)
        assert state.queue_lock is not None
        assert machine.armed == [] and guest.evictions == []
        assert state.parked == mc_memory.PARKED_RAM
        assert until(second, mc_turns.GRANTED) == mc_turns.GRANTED
        assert machine.moved == [(mc_memory.PARKED_RAM, "VibeVoice 7B's turn")]

    def test_one_render_per_card_and_one_on_each(self, machine, host):
        """Rule 5: the second request on a card waits; one on the other card does not."""
        speaker(machine)
        here, queued, there = ask(), ask(), ask(WANGP_UUID)

        until(here, mc_turns.GRANTED)
        until(there, mc_turns.GRANTED)
        assert queued.phase == mc_turns.QUEUED


# --------------------------------------------------------------------------- #
# Rule 6: warm between requests, and evicted only while idle
# --------------------------------------------------------------------------- #


class TestWarm:
    def test_the_guest_stays_warm_when_nothing_else_wants_the_card(self, machine, host):
        guest = speaker(machine)
        turn = ask()
        render(turn, guest)

        state = card_of(turn)
        assert state.warm == GUEST
        assert guest.evictions == []
        assert machine.armed == [], "the model came back under a warm guest"
        held = mc_broker.residencies(mc_broker.FAMILY_VOICE, card=IMAGE_CARD)
        assert [(entry.bytes, entry.rank) for entry in held] == [(guest.size,
                                                                  mc_broker.RANK_HOT)]

    def test_off_unloads_after_every_request_and_brings_the_model_back(self, machine, host):
        guest = speaker(machine)
        choose(host, mc_turns.OPT_KEEP_WARM, mc_turns.WARM_OFF)
        turn = ask()
        render(turn, guest)

        assert guest.evictions and card_of(turn).warm == ""
        assert machine.armed == ["VibeVoice gave the card back"]
        assert mc_broker.residencies(mc_broker.FAMILY_VOICE) == []

    def test_the_setting_is_read_from_its_label_as_the_settings_page_stores_it(
            self, machine, host):
        guest = speaker(machine)
        label = dict(mc_turns.WARM_MODES)[mc_turns.WARM_OFF]
        choose(host, mc_turns.OPT_KEEP_WARM, label)
        render(ask(), guest)

        assert guest.evictions

    def test_not_on_the_image_card_means_exactly_that(self, machine, host):
        guest = speaker(machine)
        choose(host, mc_turns.OPT_KEEP_WARM, mc_turns.WARM_NOT_IMAGE)
        on_image = ask()
        render(on_image, guest)
        assert card_of(on_image).warm == "" and machine.armed

        elsewhere = ask(WANGP_UUID)
        render(elsewhere, guest)
        assert card_of(elsewhere).warm == GUEST

    def test_a_finish_can_decline_the_warm_stay(self, machine, host):
        guest = speaker(machine)
        turn = ask()
        render(turn, guest, keep_warm=False)

        assert guest.evictions and card_of(turn).warm == ""

    def test_the_language_model_needing_the_card_evicts_a_warm_guest(self, machine):
        """The broker's own path: the LLM asks for room, and the one family it
        may take from on that card is a warm guest."""
        guest = speaker(machine)
        render(ask(WANGP_UUID), guest)
        machine.free[OTHER_CARD] = 1 * _GB

        released = mc_broker.request_vram(mc_broker.FAMILY_LLM, 8 * _GB, card=OTHER_CARD,
                                          reason="a conversation reply")

        assert guest.evictions == ["a conversation reply"]
        assert released.freed == guest.size
        assert machine.armed == [], "the image model raced the LLM for the room"

    def test_the_language_model_start_asks_only_when_a_guest_is_on_its_card(
            self, machine, tmp_path, monkeypatch):
        """``_make_room_from_a_warm_guest``, end to end: the configured placement is
        asked for on its own card, the warm guest there goes, and a start on a card
        with no guest asks nothing at all."""
        guest = speaker(machine)
        configuration = configuration_on(tmp_path, card=OTHER_CARD)
        asked = []
        real = mc_broker.request_vram
        monkeypatch.setattr(mc_broker, "request_vram",
                            lambda *a, **k: asked.append(k.get("card")) or real(*a, **k))

        assert runtime._make_room_from_a_warm_guest(configuration) == 0
        assert asked == []

        render(ask(WANGP_UUID), guest)
        machine.free[OTHER_CARD] = 1024**2  # 0 would read as "could not be measured"
        freed = runtime._make_room_from_a_warm_guest(configuration)

        assert asked == [OTHER_CARD]
        assert freed == guest.size and guest.evictions

    def test_a_working_guest_is_never_evicted(self, machine):
        """Rule 1 holds for the guest too: while granted it is declared active,
        and the broker's release refuses before it ever asks."""
        guest = speaker(machine)
        turn = ask(WANGP_UUID)
        until(turn, mc_turns.GRANTED)
        guest.load(WANGP_UUID)
        guest.working = True
        machine.free[OTHER_CARD] = 1 * _GB

        released = mc_broker.request_vram(mc_broker.FAMILY_LLM, 8 * _GB, card=OTHER_CARD)

        assert released.freed == 0 and guest.evictions == []
        held = mc_broker.residencies(mc_broker.FAMILY_VOICE, card=OTHER_CARD)
        assert [entry.rank for entry in held] == [mc_broker.RANK_ACTIVE]

    def test_the_room_a_warm_guest_leaves_is_for_whoever_asked_for_it(self, machine, host):
        """An image pass evicting the guest from Forge's card loads its own model
        into the room; a warm-up of the parked model beside it would be a third
        tenant racing the second."""
        guest = speaker(machine)
        turn = ask()
        render(turn, guest)
        assert card_of(turn).parked == mc_memory.PARKED_RAM

        released = mc_broker.request_vram(mc_broker.FAMILY_IMAGE, 10 * _GB, card=IMAGE_CARD,
                                          reason="Stage 1")

        assert guest.evictions == ["Stage 1"] and released.freed == guest.size
        assert machine.armed == []

    def test_a_warm_guest_that_says_it_is_working_keeps_the_card(self, machine):
        guest = speaker(machine)
        turn = ask(WANGP_UUID)
        render(turn, guest)
        guest.working = True

        assert mc_turns.end_warm(card_of(turn), "an image generation") == 0
        assert guest.evictions == []

    def test_a_guest_that_cannot_answer_is_taken_to_be_working(self, machine):
        """Unanswerable is answered yes: wrong that way, a warm guest stays a
        little longer; wrong the other way, a render is cut off."""
        guest = speaker(machine)
        turn = ask(WANGP_UUID)
        render(turn, guest)

        def broken(uuid):
            raise RuntimeError("the worker is gone")

        guest.rendering = broken
        assert mc_turns.end_warm(card_of(turn), "the LLM") == 0

    def test_ram_running_low_under_a_parked_model_ends_the_warm_stay(self, machine, host):
        guest = speaker(machine)
        turn = ask()
        render(turn, guest)
        assert card_of(turn).parked == mc_memory.PARKED_RAM

        machine.ram = 3 * _GB
        mc_turns.tick(card_of(turn))

        assert guest.evictions == ["system RAM is running low"]
        assert machine.armed == ["VibeVoice gave the card back"]

    def test_ram_under_a_dropped_model_is_nobodys_business(self, machine, host):
        """A model dropped to its recipe holds no RAM, so low RAM is no reason
        to end the stay: ending it would free nothing."""
        guest = speaker(machine)
        choose(host, mc_turns.OPT_IMAGE_RETURN, mc_turns.RETURN_RECIPE)
        turn = ask()
        render(turn, guest)

        machine.ram = 3 * _GB
        mc_turns.tick(card_of(turn))
        assert guest.evictions == []

    def test_a_guest_unloaded_by_its_own_hand_gives_the_card_back(self, machine, host):
        guest = speaker(machine)
        turn = ask()
        render(turn, guest)
        guest.held.clear()

        mc_turns.tick(card_of(turn))

        assert card_of(turn).warm == ""
        assert machine.armed == ["VibeVoice gave the card back"]
        assert mc_broker.residencies(mc_broker.FAMILY_VOICE) == []

    def test_an_eviction_waits_until_the_guest_has_let_go(self, machine, host):
        """Whoever needed the card goes next and sizes itself against what the
        card reports free; handing it over while the worker is still letting go
        is how the next tenant runs out of memory."""
        guest = speaker(machine, lingers=3)
        turn = ask(WANGP_UUID)
        render(turn, guest)

        mc_turns.end_warm(card_of(turn), "WanGP needs its card")

        assert machine.sleeps == 3
        assert machine.log[-2:] == ["evict", "gone"]
        assert machine.free[OTHER_CARD] == 30 * _GB

    def test_a_guest_that_never_lets_go_does_not_hold_the_card_for_ever(
            self, machine, host, monkeypatch):
        guest = speaker(machine, lingers=10**9)
        turn = ask(WANGP_UUID)
        render(turn, guest)
        monkeypatch.setattr(mc_turns, "UNLOAD_WAIT_SECONDS", 0.05)
        monkeypatch.setattr(mc_turns, "_sleep", lambda seconds: time.sleep(0.01))

        mc_turns.end_warm(card_of(turn), "WanGP needs its card")

        assert card_of(turn).warm == ""
        assert mc_broker.residencies(mc_broker.FAMILY_VOICE) == []


# --------------------------------------------------------------------------- #
# Rule 7: the image model makes room, and comes back
# --------------------------------------------------------------------------- #


class TestParking:
    def test_automatic_keeps_the_model_in_ram_when_it_fits_above_the_margin(
            self, machine, host):
        speaker(machine)
        turn = ask()
        until(turn, mc_turns.GRANTED)

        assert machine.moved == [(mc_memory.PARKED_RAM, "VibeVoice 7B's turn")]
        assert card_of(turn).parked == mc_memory.PARKED_RAM

    def test_automatic_uses_the_recipe_when_ram_is_short(self, machine, host):
        """18.4 GB of model, 2 GB for the guest and the 4 GB margin do not fit
        in 20 GB, so the model is dropped rather than paged."""
        speaker(machine)
        machine.ram = 20 * _GB
        turn = ask()
        until(turn, mc_turns.GRANTED)

        assert machine.moved == [(mc_memory.PARKED_RECIPE, "VibeVoice 7B's turn")]

    def test_from_ram_is_refused_rather_than_paged_and_names_the_setting(self, machine, host):
        speaker(machine)
        choose(host, mc_turns.OPT_IMAGE_RETURN, mc_turns.RETURN_RAM)
        machine.ram = 20 * _GB
        turn = ask()

        assert until(turn, mc_turns.BLOCKED) == mc_turns.BLOCKED
        assert machine.moved == []
        assert "Settings → Model Chain" in turn.warning
        assert "18.4 GB" in turn.warning and "20.0 GB is available" in turn.warning
        assert lock_is_free()

    def test_from_its_recipe_drops_even_when_ram_would_hold_it(self, machine, host):
        speaker(machine)
        choose(host, mc_turns.OPT_IMAGE_RETURN, mc_turns.RETURN_RECIPE)
        until(ask(), mc_turns.GRANTED)

        assert [how for how, _ in machine.moved] == [mc_memory.PARKED_RECIPE]

    def test_a_forge_that_cannot_drop_refuses_before_anything_moves(self, machine, host):
        """Dropping would fall back to parking in RAM -- the one thing RAM was
        just measured too short for."""
        speaker(machine)
        machine.ram = 20 * _GB
        machine.can_drop = False
        turn = ask()

        assert until(turn, mc_turns.BLOCKED) == mc_turns.BLOCKED
        assert machine.moved == []

    def test_nothing_is_parked_when_nothing_is_on_the_card(self, machine, host):
        speaker(machine)
        machine.image = 0
        machine.free[IMAGE_CARD] = 22 * _GB
        until(ask(), mc_turns.GRANTED)

        assert machine.moved == []

    def test_a_refusal_after_parking_puts_the_model_back(self, machine, host):
        """Blocked leaves nothing moved: the model that made room for a render
        that then did not fit is warmed straight back."""
        speaker(machine)
        turn = ask(need=24.0)

        assert until(turn, mc_turns.BLOCKED) == mc_turns.BLOCKED
        assert "with the image model out of the way" in turn.warning
        assert machine.armed == ["VibeVoice gave the card back"]
        assert lock_is_free()

    def test_a_generation_waiting_at_the_gate_is_not_raced_by_a_warm_up(
            self, machine, host, monkeypatch):
        """The job the gate held loads its own model when it goes on; a warm-up
        started beside it would be a second load of the same model."""
        guest = speaker(machine)
        choose(host, mc_turns.OPT_KEEP_WARM, mc_turns.WARM_OFF)
        turn = ask()

        def driver(seconds):
            step(turn)
            if turn.phase == mc_turns.GRANTED:
                guest.load(IMAGE_UUID)
                turn.finish()

        monkeypatch.setattr(mc_turns, "_sleep", driver)
        mc_turns.image_gate(object())

        assert turn.phase == mc_turns.DONE and guest.evictions
        assert machine.armed == []


# --------------------------------------------------------------------------- #
# Rule 8: never the pagefile
# --------------------------------------------------------------------------- #


class TestNeverThePagefile:
    def test_a_request_that_would_reach_into_the_pagefile_is_refused_with_the_numbers(
            self, machine):
        guest = speaker(machine)
        machine.ram = 5 * _GB
        turn = ask(WANGP_UUID, ram=3.0)

        assert until(turn, mc_turns.BLOCKED) == mc_turns.BLOCKED
        assert "3.0 GB of system RAM" in turn.warning and "5.0 GB is available" in turn.warning
        assert "Nothing was loaded" in turn.warning
        assert guest.held == {}

    def test_a_request_the_card_cannot_hold_is_refused_with_the_numbers(self, machine):
        speaker(machine)
        machine.free[OTHER_CARD] = 10 * _GB
        turn = ask(WANGP_UUID)

        assert until(turn, mc_turns.BLOCKED) == mc_turns.BLOCKED
        assert turn.warning == ("VibeVoice 7B needs 18.0 GB on NVIDIA GeForce RTX 5090 plus "
                                "1 GB of margin, and 10.0 GB can be freed there")

    def test_a_guest_already_on_the_card_counts_what_it_holds(self, machine):
        """A warm guest's second request needs no more than it already has."""
        guest = speaker(machine)
        render(ask(WANGP_UUID), guest)
        assert machine.free[OTHER_CARD] == 12 * _GB

        assert until(ask(WANGP_UUID), mc_turns.GRANTED) == mc_turns.GRANTED

    def test_the_card_is_read_through_the_driver_and_fresh(self, machine):
        """Not through torch, which would make a context on the card being
        emptied, and not from a cached reading taken before the room was made."""
        speaker(machine)
        until(ask(WANGP_UUID), mc_turns.GRANTED)

        assert machine.fresh and all(machine.fresh)


# --------------------------------------------------------------------------- #
# WanGP's card: the lease
# --------------------------------------------------------------------------- #


class TestTheLease:
    def test_mini_paint_is_asked_and_the_turn_waits_until_the_card_is_held(
            self, machine, wangp):
        speaker(machine)
        turn = ask(WANGP_UUID)

        assert step(turn) == mc_turns.CLEARING
        assert wangp.said("request") == [("request", mc_turns.LEASE_OWNER, "VibeVoice 7B",
                                          int(18 * _GB))]
        assert turn.reason == "waiting for WanGP to finish its job"

        wangp.phase = "holding"
        assert step(turn) == mc_turns.CLEARING
        wangp.phase, wangp.bridge_hold = "held", "held"
        assert step(turn) == mc_turns.MAKING_ROOM
        assert step(turn) == mc_turns.GRANTED

    def test_the_lease_is_renewed_while_the_render_runs(self, machine, wangp):
        """Mini Paint expires a lease nobody renews in twenty seconds."""
        speaker(machine)
        wangp.phase, wangp.bridge_hold = "held", "held"
        turn = ask(WANGP_UUID)
        until(turn, mc_turns.GRANTED)
        before = len(wangp.said("state"))

        step(turn, 3)
        assert len(wangp.said("state")) == before + 3

    def test_the_lease_is_released_when_the_guest_leaves(self, machine, host, wangp):
        guest = speaker(machine)
        wangp.phase, wangp.bridge_hold = "held", "held"
        choose(host, mc_turns.OPT_KEEP_WARM, mc_turns.WARM_OFF)
        turn = ask(WANGP_UUID)
        render(turn, guest)

        assert wangp.said("release") == [("release", "lease-1")]
        assert machine.log.index("gone") < machine.log.index("release")

    def test_a_warm_stay_keeps_the_lease_and_renews_it(self, machine, wangp):
        guest = speaker(machine)
        wangp.phase, wangp.bridge_hold = "held", "held"
        turn = ask(WANGP_UUID)
        render(turn, guest)
        before = len(wangp.said("state"))

        mc_turns.tick(card_of(turn))
        assert wangp.said("release") == []
        assert len(wangp.said("state")) == before + 1

    def test_wanted_ends_the_warm_stay_and_the_memory_goes_before_the_card(
            self, machine, wangp):
        """WanGP sizes itself against what the card reports free and has no
        fallback, so it is not handed the card until the guest has let go."""
        guest = speaker(machine, lingers=2)
        wangp.phase, wangp.bridge_hold = "held", "held"
        turn = ask(WANGP_UUID)
        render(turn, guest)

        wangp.wanted = True
        mc_turns.tick(card_of(turn))

        assert guest.evictions == ["WanGP needs its card"]
        assert machine.log[-3:] == ["evict", "gone", "release"]

    def test_a_lease_that_stops_being_held_ends_the_warm_stay(self, machine, wangp):
        guest = speaker(machine)
        wangp.phase, wangp.bridge_hold = "held", "held"
        turn = ask(WANGP_UUID)
        render(turn, guest)

        wangp.phase = "expired"
        mc_turns.tick(card_of(turn))
        assert guest.evictions == ["WanGP's card is no longer held for it"]

    def test_a_flush_is_asked_for_only_when_the_render_does_not_fit_soft_first(
            self, machine, wangp):
        speaker(machine)
        wangp.phase, wangp.bridge_hold = "held", "held"
        machine.free[OTHER_CARD] = 16 * _GB
        wangp.flushes = {"soft": 1.5, "hard": 8.0}
        turn = ask(WANGP_UUID)

        assert until(turn, mc_turns.GRANTED) == mc_turns.GRANTED
        assert wangp.said("flush") == [("flush", "soft"), ("flush", "hard")]
        assert machine.log.index("flushed soft") < machine.log.index("flush hard")

    def test_a_flush_is_waited_for_and_the_card_read_only_once_it_is_done(
            self, machine, wangp):
        """Mini Paint's executor does a flush on its own next pass. Until then the
        lease says holding, and a reading of the card says only that WanGP has not
        finished letting go -- measured then, the render would be refused for
        room the flush was about to make."""
        speaker(machine)
        wangp.phase, wangp.bridge_hold = "held", "held"
        wangp.settle_after = 3
        machine.free[OTHER_CARD] = 16 * _GB
        wangp.flushes = {"soft": 4.0}
        turn = ask(WANGP_UUID)
        step(turn, 2)
        assert wangp.said("flush") == [("flush", "soft")]
        reads = len(machine.fresh)

        step(turn, 3)
        assert turn.phase == mc_turns.MAKING_ROOM
        assert "being moved off its card" in turn.reason
        assert len(machine.fresh) == reads, "the card was read while WanGP was letting go"

        assert until(turn, mc_turns.GRANTED) == mc_turns.GRANTED
        assert wangp.said("flush") == [("flush", "soft")]

    def test_a_lease_lost_while_wangp_is_flushing_refuses_the_turn(self, machine, wangp):
        speaker(machine)
        wangp.phase, wangp.bridge_hold = "held", "held"
        wangp.settle_after = 50
        machine.free[OTHER_CARD] = 16 * _GB
        turn = ask(WANGP_UUID)
        step(turn, 2)
        wangp._flushing = ""
        wangp.phase, wangp.reason = "expired", "Not renewed for 20 s"

        assert until(turn, mc_turns.BLOCKED) == mc_turns.BLOCKED
        assert "stopped being held" in turn.warning and "Not renewed" in turn.warning

    def test_a_soft_flush_that_is_enough_is_the_only_one(self, machine, wangp):
        speaker(machine)
        wangp.phase, wangp.bridge_hold = "held", "held"
        machine.free[OTHER_CARD] = 16 * _GB
        wangp.flushes = {"soft": 4.0, "hard": 8.0}
        turn = ask(WANGP_UUID)
        until(turn, mc_turns.GRANTED)

        assert wangp.said("flush") == [("flush", "soft")]
        # Granted is the flush finished with: the next request under the same
        # lease starts from nothing rather than from this one's soft flush.
        assert card_of(turn).flushing == "" and card_of(turn).flush_said == ""

    def test_a_render_that_fits_asks_for_no_flush(self, machine, wangp):
        speaker(machine)
        wangp.phase, wangp.bridge_hold = "held", "held"
        until(ask(WANGP_UUID), mc_turns.GRANTED)

        assert wangp.said("flush") == []

    def test_still_short_after_both_flushes_is_refused_and_the_card_given_back(
            self, machine, wangp):
        """With Mini Paint's own sentence in the warning: a WanGP run paused
        between tasks refuses a hard flush, and that is the whole explanation."""
        speaker(machine)
        wangp.phase, wangp.bridge_hold = "held", "held"
        wangp.flush_word = "A WanGP run paused between tasks refuses a hard flush."
        machine.free[OTHER_CARD] = 10 * _GB
        turn = ask(WANGP_UUID)

        assert until(turn, mc_turns.BLOCKED) == mc_turns.BLOCKED
        assert wangp.said("flush") == [("flush", "soft"), ("flush", "hard")]
        assert "refuses a hard flush" in turn.warning
        assert wangp.said("release") == [("release", "lease-1")]

    def test_a_request_after_a_refused_flush_starts_from_a_soft_one(self, machine, wangp):
        """The card forgets what the last turn's flush was doing when its lease
        ends; carried over, the next turn would skip straight past the soft
        flush that might have been enough."""
        speaker(machine)
        wangp.phase, wangp.bridge_hold = "held", "held"
        machine.free[OTHER_CARD] = 10 * _GB
        first = ask(WANGP_UUID)
        assert until(first, mc_turns.BLOCKED) == mc_turns.BLOCKED

        wangp.phase, wangp.lease = "held", ""
        wangp.flushes = {"soft": 12.0, "hard": 0.0}
        again = ask(WANGP_UUID)

        assert until(again, mc_turns.GRANTED) == mc_turns.GRANTED
        assert wangp.said("flush")[-1] == ("flush", "soft")

    def test_a_wangp_that_cannot_be_held_is_refused_rather_than_waited_on(self, machine, wangp):
        """Mini Paint keeps such a lease holding until it is released; a turn
        waiting on it would wait for as long as WanGP stayed up."""
        speaker(machine)
        wangp.phase, wangp.bridge_hold = "holding", "unsupported"
        wangp.reason = "This WanGP build does not expose what holding it between tasks needs."
        turn = ask(WANGP_UUID)

        assert until(turn, mc_turns.BLOCKED) == mc_turns.BLOCKED
        assert "does not expose" in turn.warning
        assert wangp.said("release") == [("release", "lease-1")]

    def test_a_card_lost_under_a_render_is_said_once_and_the_render_goes_on(
            self, machine, wangp, caplog):
        """A task that slipped past the hold takes held back to holding. Rule 1
        leaves nothing to do about it mid-render but say so, once."""
        speaker(machine)
        wangp.phase, wangp.bridge_hold = "held", "held"
        turn = ask(WANGP_UUID)
        until(turn, mc_turns.GRANTED)
        wangp.phase, wangp.reason = "holding", "WanGP is finishing the task it is running."

        with caplog.at_level("WARNING", logger="model_chain"):
            step(turn, 3)

        assert turn.phase == mc_turns.GRANTED
        assert len([line for line in caplog.messages if "no longer held" in line]) == 1

    def test_a_refused_lease_blocks_while_wangp_runs(self, machine, wangp):
        speaker(machine)
        wangp.phase, wangp.reason = "refused", "the WanGP integration is off"
        turn = ask(WANGP_UUID)

        assert until(turn, mc_turns.BLOCKED) == mc_turns.BLOCKED
        assert "the WanGP integration is off" in turn.warning

    def test_a_refused_lease_is_no_obstacle_while_wangp_is_down(self, machine, wangp):
        speaker(machine)
        wangp.phase = "refused"
        wangp.running = False
        assert until(ask(WANGP_UUID), mc_turns.GRANTED) == mc_turns.GRANTED

    def test_an_old_bridge_holds_the_clipboard_only_so_it_is_used_only_while_wangp_is_down(
            self, machine, wangp):
        speaker(machine)
        wangp.phase, wangp.bridge_hold = "held", "unsupported"
        turn = ask(WANGP_UUID)

        assert until(turn, mc_turns.BLOCKED) == mc_turns.BLOCKED
        assert "bridge 1.12.0" in turn.warning
        assert wangp.said("release") == [("release", "lease-1")]

        wangp.running = False
        assert until(ask(WANGP_UUID), mc_turns.GRANTED) == mc_turns.GRANTED

    def test_under_an_old_bridge_the_warm_stay_ends_when_wangp_starts(self, machine, wangp):
        guest = speaker(machine)
        wangp.phase, wangp.bridge_hold = "held", "unsupported"
        wangp.running = False
        turn = ask(WANGP_UUID)
        render(turn, guest)
        assert card_of(turn).warm == GUEST

        wangp.running = True
        mc_turns.tick(card_of(turn))
        assert guest.evictions == ["WanGP started, and its bridge cannot hold it"]

    def test_a_mini_paint_without_the_lease_is_no_licence_to_render_under_wangp(
            self, machine, wangp):
        """Mini Paint says WanGP lives on this card and offers no lease: an older
        Mini Paint. Nothing can hold WanGP, so its card is used only while it is
        down -- and no warm stay is kept there, because nothing would say when
        WanGP wanted the card back."""
        guest = speaker(machine)
        mc_wangp.use_lease(None)
        turn = ask(WANGP_UUID)

        assert until(turn, mc_turns.BLOCKED) == mc_turns.BLOCKED
        assert "cannot hold it" in turn.warning

        wangp.running = False
        again = ask(WANGP_UUID)
        render(again, guest)
        assert card_of(again).warm == "" and guest.evictions

    def test_a_lease_that_stops_answering_is_asked_for_again(self, machine, wangp,
                                                             monkeypatch):
        speaker(machine)
        turn = ask(WANGP_UUID)
        step(turn)
        silent = {"on": True}
        answer = wangp.state
        monkeypatch.setattr(wangp, "state",
                            lambda lease: None if silent["on"] else answer(lease))

        assert step(turn) == mc_turns.CLEARING
        assert turn.reason == "waiting for Mini Paint to answer"
        silent["on"] = False
        wangp.phase, wangp.bridge_hold = "held", "held"
        step(turn)

        assert len(wangp.said("request")) == 2, "the lease was never asked for again"
        assert until(turn, mc_turns.GRANTED) == mc_turns.GRANTED

    def test_a_lease_record_outside_the_contract_is_no_answer(self, machine, wangp,
                                                              monkeypatch):
        """A record missing a key is not half-trusted: a Mini Paint that does not
        speak the contract is one that cannot hold WanGP."""
        speaker(machine)
        answer = wangp.request

        def partial(owner, **kw):
            record = answer(owner, **kw)
            record.pop("wanted")
            return record

        monkeypatch.setattr(wangp, "request", partial)
        turn = ask(WANGP_UUID)

        assert until(turn, mc_turns.BLOCKED) == mc_turns.BLOCKED
        assert "cannot hold it" in turn.warning

    def test_no_mini_paint_at_all_is_a_card_like_any_other(self, machine):
        speaker(machine)
        assert until(ask(WANGP_UUID), mc_turns.GRANTED) == mc_turns.GRANTED


class TestWanGPsPeak:
    def test_a_guest_on_wangps_card_is_not_filed_as_wangps_peak(self, machine, wangp,
                                                                monkeypatch):
        """Counted as WanGP's, eighteen gigabytes of guest would become WanGP's
        learned peak and keep the language model's ceiling there that much lower
        for the rest of the session."""
        guest = speaker(machine)
        wangp.phase, wangp.bridge_hold = "held", "held"
        monkeypatch.setattr(mc_memory, "physical_total_vram_bytes", lambda index: 32 * _GB)
        render(ask(WANGP_UUID), guest)

        seen = mc_wangp.observe(OTHER_CARD)

        assert seen.others == 32 * _GB - machine.free[OTHER_CARD] - guest.size
        assert seen.peak < guest.size


# --------------------------------------------------------------------------- #
# The language model's side
# --------------------------------------------------------------------------- #


class TestTheLanguageModelsSide:
    def test_a_turn_holding_the_card_is_waited_for_by_every_llm_turn_on_it(self, machine):
        speaker(machine)
        turn = ask(WANGP_UUID)
        until(turn, mc_turns.GRANTED)
        there = mc_broker.cuda_execution(OTHER_CARD)

        assert mc_turns.llm_wait_reason(there) == "VibeVoice 7B on NVIDIA GeForce RTX 5090"
        assert mc_turns.llm_wait_reason(there, inside_host_job=True)
        assert mc_turns.llm_wait_reason(mc_broker.cuda_execution(IMAGE_CARD)) == ""
        assert mc_turns.llm_wait_reason(mc_broker.CPU_EXECUTION) == ""

    def test_a_turn_inside_the_host_job_does_not_wait_for_a_clearing_turn(
            self, machine, host, monkeypatch):
        """Krea's writer runs inside ``before_process``; the clearing turn is
        waiting for that very job. Making the writer wait would be a deadlock
        with a progress bar on it."""
        speaker(machine)
        monkeypatch.setattr(host.progress, "current_task", "task(2)")
        turn = ask()
        step(turn)
        here = mc_broker.cuda_execution(IMAGE_CARD)

        assert turn.phase == mc_turns.CLEARING
        assert mc_turns.llm_wait_reason(here, inside_host_job=True) == ""
        assert mc_turns.llm_wait_reason(here) != ""

    def test_a_warm_guest_is_no_reason_to_wait(self, machine):
        guest = speaker(machine)
        render(ask(WANGP_UUID), guest)

        assert mc_turns.llm_wait_reason(mc_broker.cuda_execution(OTHER_CARD)) == ""

    def test_one_card_in_both_spellings_of_its_uuid_is_one_card(self, machine):
        """The language model's setup records nvidia-smi's ``GPU-…``; the turn
        here was asked for with the bare digits torch reports. Compared as
        written they were two cards, and the conversation would not have waited."""
        speaker(machine)
        turn = mc_turns.request(GUEST, key(WANGP_UUID), need_vram=18 * _GB)
        until(turn, mc_turns.GRANTED)
        configured = mc_broker.cuda_execution(None, uuid=WANGP_UUID)

        assert mc_turns.llm_wait_reason(configured) != ""

    def test_a_conversation_on_the_guests_card_waits_and_says_what_for(
            self, machine, monkeypatch):
        guest = speaker(machine)
        turn = ask(WANGP_UUID)
        until(turn, mc_turns.GRANTED)
        monkeypatch.setattr(sessions._Gpu, "_domain",
                            lambda self: mc_broker.cuda_execution(OTHER_CARD))
        monkeypatch.setattr(sessions, "WAIT_POLL_SECONDS", 0.001)
        monkeypatch.setattr(sessions, "WAIT_NOTICE_SECONDS", -1.0)
        gpu = sessions._Gpu("a conversation reply", _Never())
        waiting = gpu.acquire()

        said = next(waiting)
        assert said.text == "Waiting for VibeVoice 7B on NVIDIA GeForce RTX 5090…"

        guest.load(WANGP_UUID)
        turn.finish(keep_warm=False)
        step(turn)
        with pytest.raises(StopIteration) as done:
            next(waiting)
        assert done.value.value is True
        gpu.release()

    def test_a_server_start_asks_a_warm_guest_for_room_before_anything_else(
            self, host, tmp_path, monkeypatch):
        """From ``client``'s own start path, ahead of the image side's reclaim and
        the negotiation that places against whatever is then free."""
        import mc_llm_context as ctx
        import mc_llm_paths
        from test_llm_runtime import FakeProcess, configure, set_free

        order = []
        monkeypatch.setattr(runtime, "_make_room_from_a_warm_guest",
                            lambda *a, **k: order.append("guest") or 0)
        monkeypatch.setattr(runtime, "_make_room_for_the_llm",
                            lambda *a, **k: order.append("image") or 0)
        mc_wangp.use_source(lambda: None)
        monkeypatch.setattr(mc_llm_paths, "data_root", lambda: tmp_path / "data")
        monkeypatch.setattr(runtime, "OFFLOAD_WAIT_SECONDS", 0.0)
        monkeypatch.setattr(runtime, "RESIDENCY_SETTLE_SECONDS", 0.0)
        monkeypatch.setattr(runtime, "_prime_prompt_cache", lambda client: None)
        monkeypatch.setattr(ctx, "_store_path", lambda: tmp_path / "calibration.json")
        ctx.forget()
        configure(monkeypatch, tmp_path, gpu_layers="all")
        set_free(monkeypatch, 20)
        managed = runtime.Runtime()
        monkeypatch.setattr(managed, "_new_process", lambda: FakeProcess([]))
        try:
            managed.client()
            assert order == ["guest", "image"]
        finally:
            managed.stop()
            ctx.forget()
            mc_wangp.use_source(None)

    def test_the_guest_is_checked_again_once_the_lock_is_held(self, machine, monkeypatch):
        """The check and the lock are not atomic: a turn queued in between is
        found by the second look, and the lock is given back to wait for it."""
        answers = iter(["", "VibeVoice 7B on GPU 1", "", ""])
        looks = []

        def blocked(domain, ours):
            looks.append(1)
            return next(answers)

        monkeypatch.setattr(sessions._Gpu, "_blocked_by_a_guest", staticmethod(blocked))
        monkeypatch.setattr(sessions._Gpu, "_domain",
                            lambda self: mc_broker.cuda_execution(OTHER_CARD))
        monkeypatch.setattr(sessions, "WAIT_POLL_SECONDS", 0.001)
        gpu = sessions._Gpu("a conversation reply", _Never())

        assert list(gpu.acquire()) == []
        assert len(looks) == 4
        gpu.release()


class _Never:
    def is_set(self):
        return False


# --------------------------------------------------------------------------- #
# The broker's view of a guest
# --------------------------------------------------------------------------- #


class TestTheBroker:
    def test_a_warm_guest_gives_ground_first_to_an_image_pass_and_to_the_llm(self):
        assert mc_broker._victim_order(mc_broker.FAMILY_IMAGE) == (mc_broker.FAMILY_VOICE,
                                                                   mc_broker.FAMILY_LLM)
        assert mc_broker._victim_order(mc_broker.FAMILY_LLM) == (mc_broker.FAMILY_VOICE,)

    def test_nothing_is_ever_released_for_a_guest(self):
        """A guest takes a card through a turn -- a request the user made --
        and never through the broker's reclaim."""
        assert mc_broker._victim_order(mc_broker.FAMILY_VOICE) == ()

    def test_the_family_is_named_in_a_sentence(self):
        assert mc_broker._named(mc_broker.FAMILY_VOICE) == "VibeVoice"
        assert mc_broker._named(mc_broker.FAMILY_IMAGE) == "image"
        assert mc_broker._named(mc_broker.FAMILY_LLM) == "LLM"

    def test_a_guest_on_the_image_card_is_an_owner_there(self, machine):
        guest = speaker(machine)
        render(ask(), guest)

        found = mc_broker.status()
        assert found.voice_bytes == guest.size
        assert "VibeVoice" in found.owners

    def test_a_guests_bytes_are_never_reported_as_stray(self, machine, monkeypatch):
        guest = speaker(machine)
        render(ask(), guest)
        monkeypatch.setattr(mc_broker, "_DRIVER_OVERHEAD", 0)
        monkeypatch.setattr(mc_broker, "_own_llm_context_bytes", lambda scope: 0)
        monkeypatch.setattr(mc_broker, "total_vram_bytes", lambda: 24 * _GB)

        machine.free[IMAGE_CARD] = 24 * _GB - guest.size
        assert mc_broker.unaccounted_bytes(card=IMAGE_CARD) == 0


# --------------------------------------------------------------------------- #
# Model Chain's own hooks
# --------------------------------------------------------------------------- #


class TestModelChainsHooks:
    @staticmethod
    def cold(monkeypatch):
        """A pipeline with nothing loaded, whatever an earlier test left behind:
        an armed one returns before it would warm anything, and the check below
        would then pass for the wrong reason."""
        monkeypatch.setattr(mc_arm, "readiness",
                            lambda: types.SimpleNamespace(armed=False, describe=lambda: "cold",
                                                          explain=lambda: "cold"))

    def test_the_warm_up_moves_nothing_onto_a_card_a_guest_is_next_for(self, machine,
                                                                        monkeypatch):
        speaker(machine)
        warmed = []
        monkeypatch.setattr(mc_arm, "_arm_image", lambda *a, **k: warmed.append(1))
        self.cold(monkeypatch)
        ask()

        mc_arm.arm(reason="this generation")
        assert warmed == []

    def test_the_warm_up_runs_once_the_card_is_free_again(self, machine, host, monkeypatch):
        guest = speaker(machine)
        warmed = []
        monkeypatch.setattr(mc_arm, "_arm_image", lambda *a, **k: warmed.append(1))
        self.cold(monkeypatch)
        choose(host, mc_turns.OPT_KEEP_WARM, mc_turns.WARM_OFF)
        render(ask(), guest)

        mc_arm.arm(reason="VibeVoice gave the card back")
        assert warmed == [1]

    def test_the_background_preloads_leave_a_held_card_alone(self, machine, host,
                                                            monkeypatch):
        import model_chain

        speaker(machine)
        started = []
        monkeypatch.setattr(mc_memory, "preload_async", lambda *a, **k: started.append(1))
        ask()

        model_chain.ScriptModelChain._preload_stage_1(make_p(host))
        model_chain.ScriptModelChain._keep_stage_1_warm(make_p(host))
        assert started == []

        mc_turns.forget()
        model_chain.ScriptModelChain._preload_stage_1(make_p(host))
        assert started == [1]

    def test_model_chains_hook_asks_whose_turn_it_is_before_anything_else(
            self, chain, host, monkeypatch):
        """Even with the chain off: the gate is about the card, not the chain."""
        order = []
        monkeypatch.setattr(mc_turns, "image_gate", lambda p: order.append("gate"))
        monkeypatch.setattr(mc_plan, "publish", lambda plan: order.append("plan"))
        settings = {**DEFAULTS, "enabled": False}

        chain.script.before_process(make_p(host), *[settings[name] for name in UI_ORDER])
        assert order[:2] == ["gate", "plan"]

    def test_the_gate_script_is_on_both_tabs_and_draws_nothing(self, host):
        import model_chain_turns

        script = model_chain_turns.ScriptModelChainTurns()
        assert script.show(False) is host.scripts.AlwaysVisible
        assert script.show(True) is host.scripts.AlwaysVisible
        assert script.ui(True) == [] and script.ui(False) == []

    def test_the_gate_script_calls_the_gate_and_never_stops_a_generation(
            self, host, monkeypatch):
        """The host swallows what a hook raises anyway; this reports it and goes on."""
        import model_chain_turns

        seen, reported = [], []
        monkeypatch.setattr(mc_turns, "image_gate", lambda p: seen.append(p))
        monkeypatch.setattr(host.errors, "report",
                            lambda message, *, exc_info=False: reported.append(message))
        script = model_chain_turns.ScriptModelChainTurns()
        p = object()
        script.before_process(p)
        assert seen == [p]

        monkeypatch.setattr(mc_turns, "image_gate",
                            lambda p: (_ for _ in ()).throw(RuntimeError("boom")))
        script.before_process(p)
        assert reported and "generating anyway" in reported[0]

    def test_both_settings_are_registered_with_their_labels(self, host):
        import model_chain  # noqa: F401 -- registers the settings

        templates = host.shared.options_templates
        back = templates[mc_turns.OPT_IMAGE_RETURN]
        warm = templates[mc_turns.OPT_KEEP_WARM]
        assert back.default == mc_turns.RETURN_AUTOMATIC
        assert warm.default == mc_turns.WARM_EVERY
        assert back.component_args["choices"] == [label for _, label in mc_turns.RETURN_MODES]
        assert warm.component_args["choices"] == [label for _, label in mc_turns.WARM_MODES]


# --------------------------------------------------------------------------- #
# Making room in Forge: the two ways the image model leaves the card
# --------------------------------------------------------------------------- #


class TestMakingRoomInForge:
    """:func:`mc_memory.park_image_model` and :func:`mc_memory.drop_image_model`,
    through the host's own functions rather than stand-ins for them."""

    @pytest.fixture
    def loaded(self, host, monkeypatch):
        from backend import memory_management

        monkeypatch.setattr(mc_memory, "_cache", mc_memory._Cache())
        data = host.sd_models.model_data
        monkeypatch.setattr(data, "sd_model", object())
        monkeypatch.setattr(data, "forge_hash", "{'checkpoint': 'flux.safetensors'}")
        monkeypatch.setattr(data, "forge_loading_parameters",
                            {"checkpoint": "flux.safetensors"})
        memory_management.unloaded_all.clear()
        host.sd_models.unloaded.clear()
        return data

    def test_parking_moves_every_weight_and_keeps_the_very_same_model(self, loaded, host):
        """What comes back from RAM is exactly what left -- LoRA patches included --
        because nothing but the weights' device changed."""
        from backend import memory_management

        model = loaded.sd_model
        assert mc_memory.park_image_model("a test") == mc_memory.PARKED_RAM

        assert memory_management.unloaded_all == [True]
        assert loaded.sd_model is model
        assert loaded.forge_hash == "{'checkpoint': 'flux.safetensors'}"

    def test_dropping_keeps_only_the_recipe(self, loaded, host, monkeypatch):
        """The model object goes and the loading hash with it, so the next
        generation rebuilds from the loading parameters -- which stay."""
        cleared = []
        monkeypatch.setattr(mc_memory, "clear_pinned_encoders", lambda: cleared.append("pins"))
        monkeypatch.setattr(mc_memory, "clear_stage_2_components",
                            lambda: cleared.append("stage 2"))

        assert mc_memory.drop_image_model("a test") == mc_memory.PARKED_RECIPE
        assert host.sd_models.unloaded == [True]
        assert type(loaded.sd_model).__name__ == "FakeInitialModel"
        assert loaded.forge_hash == ""
        assert loaded.forge_loading_parameters == {"checkpoint": "flux.safetensors"}
        assert cleared == ["pins", "stage 2"], "a reference this module kept would pin it"

    def test_dropping_lets_go_of_a_warm_copy_of_the_same_checkpoint_only(self, loaded):
        """Another cached checkpoint is a different model, already in RAM
        before this; the same one would keep every byte the drop was for."""
        same = loaded.forge_hash
        for name, key_ in (("flux", same), ("sdxl", "{'checkpoint': 'sdxl.safetensors'}")):
            mc_memory._cache._entries[key_] = mc_memory._Entry(
                key=key_, checkpoint_name=name, sd_model=object(), size_bytes=_GB)

        mc_memory.drop_image_model("a test")
        assert mc_memory._cache.names() == ["sdxl"]

    def test_a_forge_without_the_unload_is_parked_instead_and_says_so(
            self, loaded, host, monkeypatch):
        monkeypatch.delattr(host.sd_models, "unload_model_weights")

        assert not mc_memory.can_drop_image_model()
        assert mc_memory.drop_image_model("a test") == mc_memory.PARKED_RAM

    def test_a_fresh_reading_of_another_card_is_not_the_cached_one(self, monkeypatch):
        """A turn that has just unloaded a guest or flushed WanGP decides on the
        card as it is now, not as nvidia-smi saw it seconds ago."""
        monkeypatch.setattr(mc_memory, "_smi_readings", (time.monotonic(), {1: 5 * _GB}))
        monkeypatch.setattr(mc_memory, "_smi_read",
                            lambda index, what: mc_memory._smi_readings[1] or {1: 25 * _GB})

        assert mc_memory.physical_free_vram_bytes(1) == 5 * _GB
        assert mc_memory.physical_free_vram_bytes(1, fresh=True) == 25 * _GB


# --------------------------------------------------------------------------- #
# Asking, cancelling, and a guest that goes away
# --------------------------------------------------------------------------- #


class TestRequests:
    def test_an_unregistered_guest_is_refused_without_touching_any_card(self, machine):
        turn = mc_turns.request("Nobody", IMAGE_UUID, need_vram=_GB)

        assert turn.phase == mc_turns.BLOCKED
        assert mc_turns._cards == {}, "the gate's fast path is gone for the session"

    def test_a_request_withdrawn_before_its_grant_leaves_nothing_held(self, machine, host):
        speaker(machine)
        turn = ask()
        step(turn)
        assert card_of(turn).queue_lock is not None

        stop = threading.Event()
        stop.set()
        assert turn.wait(timeout=1.0, cancelled=stop) == mc_turns.CANCELLED
        step(turn)

        assert turn.phase == mc_turns.CANCELLED
        assert lock_is_free()
        assert not card_of(turn).busy

    def test_a_withdrawn_request_leaves_a_warm_guest_warm(self, machine, host):
        guest = speaker(machine)
        render(ask(WANGP_UUID), guest)
        turn = ask(WANGP_UUID)
        step(turn)
        turn.cancel()
        step(turn)

        assert card_of(turn).warm == GUEST and guest.evictions == []

    def test_a_guest_that_vanishes_after_its_grant_gives_the_card_back(self, machine, host):
        speaker(machine)
        turn = ask()
        until(turn, mc_turns.GRANTED)
        mc_turns.unregister_guest(GUEST)

        step(turn)
        assert turn.phase == mc_turns.DONE
        assert lock_is_free() and machine.armed

    def test_a_grant_nobody_uses_is_taken_back(self, machine, host, monkeypatch):
        speaker(machine)
        monkeypatch.setattr(mc_turns, "GRANT_IDLE_SECONDS", 0.0)
        turn = ask()
        until(turn, mc_turns.GRANTED)

        step(turn)
        assert turn.phase == mc_turns.DONE

    def test_the_snapshot_says_what_each_card_is_doing(self, machine, host):
        guest = speaker(machine)
        render(ask(WANGP_UUID), guest)
        ask()

        found = {row["uuid"]: row for row in mc_turns.snapshot()}
        assert found[WANGP_UUID]["warm"] == GUEST
        assert found[IMAGE_UUID]["queued"] == ["VibeVoice 7B on NVIDIA GeForce RTX 3090: queued"]


# --------------------------------------------------------------------------- #
# The real driver
# --------------------------------------------------------------------------- #


@pytest.fixture
def driven(machine, monkeypatch):
    """The card's own thread instead of a test's hand."""
    monkeypatch.setattr(mc_turns, "_ensure_driver", _DRIVER)
    monkeypatch.setattr(mc_turns, "TICK_SECONDS", 0.005)
    monkeypatch.setattr(mc_turns, "WARM_WATCH_SECONDS", 0.005)
    monkeypatch.setattr(mc_turns, "_sleep", lambda seconds: time.sleep(0.001))
    return machine


def stopped(turn, seconds=5.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        state = card_of(turn)
        if state is None or state.thread is None:
            return True
        time.sleep(0.005)
    return False


class TestTheRealDriver:
    def test_a_request_is_granted_used_and_given_back_by_the_cards_own_thread(self, driven):
        guest = speaker(driven)
        turn = ask(WANGP_UUID)

        assert turn.wait(timeout=5.0) == mc_turns.GRANTED
        guest.load(WANGP_UUID)
        turn.finish(keep_warm=False)

        assert stopped(turn), "the card's thread never ended"
        assert turn.phase == mc_turns.DONE and guest.evictions

    def test_a_warm_guest_keeps_its_thread_until_something_ends_the_stay(self, driven):
        guest = speaker(driven)
        turn = ask(WANGP_UUID)
        assert turn.wait(timeout=5.0) == mc_turns.GRANTED
        guest.load(WANGP_UUID)
        turn.finish()
        time.sleep(0.05)
        assert card_of(turn).thread is not None

        guest.held.clear()  # the engine's own Unload
        assert stopped(turn)
        assert card_of(turn).warm == ""

    def test_a_caller_that_gives_up_is_withdrawn(self, driven, host):
        speaker(driven)
        host.shared.state.job = "txt2img"
        turn = ask()
        stop = threading.Event()
        threading.Timer(0.05, stop.set).start()

        assert turn.wait(timeout=5.0, cancelled=stop) == mc_turns.CANCELLED
        assert stopped(turn)
        assert lock_is_free()
