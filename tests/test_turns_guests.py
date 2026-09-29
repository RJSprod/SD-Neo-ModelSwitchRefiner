"""The join between the turn system and VibeVoice: the guest, and the Voice Box's client.

Everything here runs against the real :mod:`mc_turns` on the two-card machine
double of ``test_turns``, with the VibeVoice runtime replaced by a stub module:
the point is that the guest's three methods reach that runtime per card, that a
request from the client is a turn like any other, and that the two modules
agree on their words -- the guest's name and the turn phases -- without either
importing the other.
"""

from __future__ import annotations

import sys
import types

import pytest

import mc_broker
import mc_memory
import mc_turns
import mc_turns_guests as guests
import mc_voice_box as box
from test_turns import (IMAGE_CARD, IMAGE_UUID, OTHER_CARD, WANGP_UUID, machine, key,  # noqa: F401
                        step, until)

_GB = 1024**3


class StubRuntime:
    """VibeVoice's runtime as the guest sees it: figures per card key."""

    def __init__(self):
        self.held: dict[str, int] = {}
        self.working: set = set()
        self.evictions: list = []

    def resident_bytes(self, card_uuid):
        return self.held.get(key(card_uuid), 0)

    def rendering(self, card_uuid):
        return key(card_uuid) in self.working

    def evict(self, card_uuid, reason):
        self.evictions.append((key(card_uuid), reason))
        return self.held.pop(key(card_uuid), 0)


@pytest.fixture
def stub(monkeypatch):
    found = StubRuntime()
    monkeypatch.setitem(sys.modules, "mc_voice_vibevoice_runtime", found)
    yield found
    box.forget()


class TestTheWords:
    def test_the_guest_is_named_the_same_on_both_sides(self):
        vibevoice = pytest.importorskip("mc_voice_vibevoice")
        assert guests.GUEST == vibevoice.GUEST == "VibeVoice"

    def test_the_phases_the_voice_box_acts_on_are_the_turn_systems(self):
        assert guests.PHASES["granted"] == mc_turns.GRANTED == box.TURN_GRANTED
        assert guests.PHASES["blocked"] == mc_turns.BLOCKED == box.TURN_BLOCKED
        assert guests.PHASES["cancelled"] == mc_turns.CANCELLED == box.TURN_CANCELLED
        assert guests.PHASES["done"] == mc_turns.DONE

    def test_the_voice_side_never_imports_the_turn_system(self):
        import ast
        from pathlib import Path

        for name in ("mc_voice_box.py", "mc_voice_box_api.py"):
            tree = ast.parse((Path(mc_turns.__file__).parent / name).read_text("utf-8"))
            imported = {node.names[0].name.split(".")[0] for node in ast.walk(tree)
                        if isinstance(node, ast.Import)}
            imported |= {(node.module or "").split(".")[0] for node in ast.walk(tree)
                         if isinstance(node, ast.ImportFrom)}
            assert not imported & {"mc_turns", "mc_turns_guests", "mc_memory", "mc_broker"}, name


class TestTheGuest:
    def test_the_three_methods_reach_the_runtime_per_card(self, stub):
        guest = guests.VibeVoiceGuest()
        stub.held[key(WANGP_UUID)] = 18 * _GB
        stub.working.add(key(WANGP_UUID))

        assert guest.resident_bytes(WANGP_UUID) == 18 * _GB
        assert guest.resident_bytes(key(WANGP_UUID)) == 18 * _GB, "both spellings are one card"
        assert guest.resident_bytes(IMAGE_UUID) == 0
        assert guest.rendering(WANGP_UUID) is True and guest.rendering(IMAGE_UUID) is False
        assert guest.evict(WANGP_UUID, "an image generation needs the card") == 18 * _GB
        assert stub.evictions == [(key(WANGP_UUID), "an image generation needs the card")]
        assert guest.resident_bytes(WANGP_UUID) == 0

    def test_install_registers_the_guest_and_hands_the_box_its_client(self, machine, stub):
        guests.install()
        guests.install()

        assert mc_turns._guest(guests.GUEST) is not None
        assert isinstance(box.turns(), guests.TurnClient)


class TestTheClient:
    def test_a_request_is_a_turn_for_the_guest_and_is_granted_on_a_free_card(self, machine, stub):
        guests.install()
        client = box.turns()
        turn = client.request(WANGP_UUID, need_vram=18 * _GB, need_ram=2 * _GB,
                              label="VibeVoice 7B — Take 1")

        assert turn.guest == guests.GUEST and turn.label == "VibeVoice 7B — Take 1"
        assert until(turn, mc_turns.GRANTED) == mc_turns.GRANTED
        assert client.snapshot()[0]["active"].startswith("VibeVoice 7B — Take 1")
        stub.held[key(WANGP_UUID)] = 18 * _GB
        turn.finish(keep_warm=None)
        step(turn)
        assert turn.phase == mc_turns.DONE and mc_turns._state(WANGP_UUID).warm == guests.GUEST

    def test_the_cards_come_with_their_roles_and_the_drivers_spelling(self, machine, stub, wangp):
        client = guests.TurnClient()
        found = client.cards()

        assert [row["index"] for row in found] == [IMAGE_CARD, OTHER_CARD]
        assert found[0]["uuid"] == IMAGE_UUID and found[1]["uuid"] == WANGP_UUID
        assert found[0]["image_card"] is True and found[0]["wangp_card"] is False
        assert found[1]["image_card"] is False and found[1]["wangp_card"] is True
        assert found[1]["name"] == "NVIDIA GeForce RTX 5090"
        assert found[0]["key"] == key(IMAGE_UUID)

    def test_the_spelling_is_nvidia_smis(self):
        assert guests.spell_uuid(key(WANGP_UUID)) == WANGP_UUID
        assert guests.spell_uuid(WANGP_UUID) == WANGP_UUID
        assert guests.spell_uuid("gpu-abc") == "GPU-abc"
        assert guests.spell_uuid("abc") == "GPU-abc" and guests.spell_uuid("") == ""

    def test_unloading_by_hand_goes_through_the_turn_system_when_it_has_the_card(
            self, machine, stub, host):
        guests.install()
        client = box.turns()
        turn = client.request(IMAGE_UUID, need_vram=18 * _GB)
        until(turn, mc_turns.GRANTED)
        stub.held[key(IMAGE_UUID)] = 18 * _GB
        turn.finish()
        step(turn)
        state = mc_turns._state(IMAGE_UUID)
        assert state.warm == guests.GUEST and state.parked == mc_memory.PARKED_RAM

        assert client.unload(IMAGE_UUID, "unloaded from the Voice Box") == 18 * _GB

        assert state.warm == "" and stub.evictions[-1][1] == "unloaded from the Voice Box"
        assert machine.armed == ["VibeVoice gave the card back"], "the parked model came back"
        assert mc_broker.residencies(mc_broker.FAMILY_VOICE) == []

    def test_unloading_a_card_the_turn_system_does_not_know_asks_the_runtime(self, machine, stub):
        stub.held[key(WANGP_UUID)] = 5 * _GB
        assert guests.TurnClient().unload(WANGP_UUID, "by hand") == 5 * _GB
        assert stub.evictions == [(key(WANGP_UUID), "by hand")]


@pytest.fixture
def wangp(machine):
    from test_turns import MiniPaint
    import mc_wangp

    report = {"version": 1, "available": True, "configured": True, "gpu_uuid": WANGP_UUID,
              "state": "READY", "running": False, "instance_id": "abc123", "generating": None}
    mc_wangp.use_source(lambda: dict(report))
    mini = MiniPaint(machine, report)
    mc_wangp.use_lease(mini)
    return mini
