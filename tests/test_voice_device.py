"""Where a voice component runs, and the three namespaces that number cards.

This file is mostly about refusals. The device catalogue itself is a short
function; what earns a test file is everything around it -- a token that has to
survive a machine renumbering its cards, a control that must not appear for a
component that would ignore it, a stored choice that must not be quietly
rewritten when a card is unplugged, and a session that must not be reported as
running on a card it fell back off.

The card list is faked rather than detected. A test that called nvidia-smi
would pass on one machine and be skipped everywhere else, which is the same as
not having been written.
"""

from __future__ import annotations

import sys
import types

import pytest

import mc_voice_device as devices


class FakeGpu:
    """One nvidia-smi row, with the attribute names ``GpuInfo`` uses.

    Hand-written rather than the real frozen dataclass because the point of
    these tests is what this module does with an *arbitrary* row -- including
    rows the real detector would never emit, like one with no UUID, which is
    exactly the row the token rule exists to reject.
    """

    def __init__(self, index, uuid, name="NVIDIA GeForce RTX 3090", memory=24576):
        self.physical_index = index
        self.uuid = uuid
        self.name = name
        self.memory_total_mb = memory


THREE_NINETY = FakeGpu(0, "GPU-1111aaaa-2222-3333-4444-555566667777",
                       "NVIDIA GeForce RTX 3090", 24576)
FIFTY_NINETY = FakeGpu(1, "GPU-8888bbbb-9999-cccc-dddd-eeeeffff0000",
                       "NVIDIA GeForce RTX 5090", 32607)


@pytest.fixture(autouse=True)
def forget():
    """No card list survives from one test into the next.

    The cache is a minute long and these tests install different machines in
    consecutive functions, so without this the second one would be asserting
    against the first one's hardware.
    """
    devices.forget_cards()
    yield
    devices.forget_cards()


@pytest.fixture
def machine(monkeypatch):
    """Install a set of cards, and a processor that is always there."""
    def install(cards):
        detection = types.ModuleType("prompt_master.inference.device_detection")
        detection.detect_gpus = lambda *a, **k: list(cards)
        detection.detect_cpu = lambda: types.SimpleNamespace(
            name="AMD Ryzen 9 7950X", memory_total_mb=65413)
        monkeypatch.setitem(sys.modules,
                            "prompt_master.inference.device_detection", detection)
        devices.forget_cards()
    return install


@pytest.fixture
def stored(monkeypatch):
    """A host whose options store can be read and written."""
    held = {}

    class Opts:
        def __init__(self):
            self.data = held

        def set(self, key, value):
            held[key] = value

        def save(self, *a, **k):
            pass

    monkeypatch.setitem(sys.modules, "modules", types.SimpleNamespace(
        shared=types.SimpleNamespace(opts=Opts(), config_filename="config.json")))
    return held


class TestTheCardListDescribesTheMachine:
    def test_the_processor_is_always_offered(self, machine):
        machine([])

        found = devices.cards()

        assert [item["token"] for item in found] == [devices.CPU]
        assert "7950X" in found[0]["label"], found[0]["label"]

    def test_a_machine_with_no_driver_still_answers(self, monkeypatch):
        """Detection failing is not a reason to fail the panel being drawn."""
        detection = types.ModuleType("prompt_master.inference.device_detection")

        def refuse(*a, **k):
            raise RuntimeError("nvidia-smi is not available")

        detection.detect_gpus = refuse
        detection.detect_cpu = lambda: types.SimpleNamespace(
            name="Some CPU", memory_total_mb=16384)
        monkeypatch.setitem(sys.modules,
                            "prompt_master.inference.device_detection", detection)
        devices.forget_cards()

        found = devices.cards()

        assert [item["kind"] for item in found] == ["cpu"]

    def test_every_card_is_offered_once(self, machine):
        """Once each, unlike the language model's list.

        That one offers a card three or four times because the model can be
        split across it and system RAM. An inference session cannot be, so a
        second entry for the same card would be a second way to say one thing.
        """
        machine([THREE_NINETY, FIFTY_NINETY])

        found = devices.cards()

        assert len(found) == 3
        assert [item["kind"] for item in found] == ["cpu", "gpu", "gpu"]
        assert "RTX 3090" in found[1]["label"]
        assert "RTX 5090" in found[2]["label"]

    def test_a_card_with_no_uuid_is_dropped_rather_than_numbered(self, machine):
        """The one fallback this module refuses to have.

        A positional token is precisely the bug the UUID exists to avoid, so a
        row with no identity is left out rather than given a position for a
        name.
        """
        machine([FakeGpu(0, ""), FIFTY_NINETY])

        found = devices.cards()

        assert len(found) == 2
        assert found[1]["uuid"] == FIFTY_NINETY.uuid

    def test_the_token_is_the_uuid_and_not_the_index(self, machine):
        machine([THREE_NINETY, FIFTY_NINETY])

        tokens = [item["token"] for item in devices.cards() if item["kind"] == "gpu"]

        assert tokens == [f"{devices.GPU_PREFIX}{THREE_NINETY.uuid}",
                          f"{devices.GPU_PREFIX}{FIFTY_NINETY.uuid}"]
        # Not a position, in either spelling. The stored string has to name the
        # card rather than where it happened to be in one program's list.
        assert f"{devices.GPU_PREFIX}0" not in tokens
        assert f"{devices.GPU_PREFIX}1" not in tokens

    def test_the_same_card_keeps_its_token_when_the_machine_renumbers_it(self, machine):
        """The whole reason the token is a UUID.

        nvidia-smi orders by PCI bus, CUDA defaults to fastest-first, and
        DirectML enumerates DXGI adapters. A setting written under one ordering
        has to still name the same card under another.
        """
        machine([THREE_NINETY, FIFTY_NINETY])
        first = devices.card(f"{devices.GPU_PREFIX}{FIFTY_NINETY.uuid}")

        swapped = FakeGpu(0, FIFTY_NINETY.uuid, FIFTY_NINETY.name,
                          FIFTY_NINETY.memory_total_mb)
        machine([swapped, FakeGpu(1, THREE_NINETY.uuid, THREE_NINETY.name)])
        second = devices.card(f"{devices.GPU_PREFIX}{FIFTY_NINETY.uuid}")

        assert first is not None and second is not None
        assert second["name"] == first["name"]
        assert second["index"] != first["index"], "so this proves nothing"


class TestOnlyWhatCanMoveIsOffered:
    """A control whose value the engine ignores is a control that lies."""

    def test_the_enhancement_stages_can_be_placed(self):
        assert devices.placeable("voice-pipeline-dpdfnet")
        assert devices.placeable("voice-pipeline-lavasr")

    @pytest.mark.parametrize("component", ["tts-pocket", "tts-sopro", "tts-kokoro",
                                           "recording-cleanup"])
    def test_the_cpu_only_components_are_not_offered_a_choice(self, component):
        assert not devices.placeable(component)

    @pytest.mark.parametrize("component", ["tts-pocket", "tts-sopro", "tts-kokoro",
                                           "recording-cleanup"])
    def test_each_of_them_says_why_rather_than_saying_nothing(self, component):
        reason = devices.unplaceable_reason(component)

        assert len(reason) > 80, reason
        assert not reason.lower().startswith("unsupported"), reason

    def test_a_component_that_can_move_has_no_reason_to_print(self):
        assert devices.unplaceable_reason("voice-pipeline-dpdfnet") == ""

    def test_an_unknown_component_is_answered_rather_than_raised(self):
        assert devices.placeable("tts-nothing") is False
        assert devices.unplaceable_reason("tts-nothing") == ""
        assert devices.describe("tts-nothing")["placeable"] is False


class TestTheStoredChoiceIsRespectedAndRefused:
    def test_the_default_is_the_processor(self, machine, stored):
        machine([THREE_NINETY])

        assert devices.placement("voice-pipeline-dpdfnet") == devices.CPU

    def test_a_card_can_be_chosen_and_reads_back(self, machine, stored):
        machine([THREE_NINETY, FIFTY_NINETY])
        token = f"{devices.GPU_PREFIX}{FIFTY_NINETY.uuid}"

        devices.remember("voice-pipeline-dpdfnet", token)

        assert devices.placement("voice-pipeline-dpdfnet") == token
        assert stored[devices.OPT_DEVICE_DPDFNET] == token

    def test_a_card_this_machine_does_not_have_is_refused(self, machine, stored):
        machine([THREE_NINETY])

        with pytest.raises(ValueError):
            devices.remember("voice-pipeline-dpdfnet", f"{devices.GPU_PREFIX}nope")

        assert devices.OPT_DEVICE_DPDFNET not in stored

    def test_a_component_with_no_device_setting_is_refused(self, machine, stored):
        machine([THREE_NINETY])

        with pytest.raises(ValueError):
            devices.remember("tts-pocket", devices.CPU)

    def test_a_missing_card_does_not_erase_the_setting(self, machine, stored):
        """Unplugged is not unchosen.

        The stage runs on the processor while the card is away -- there is
        nowhere else for it to run -- but the setting is untouched, so putting
        the card back restores the choice rather than having lost it.
        """
        machine([THREE_NINETY, FIFTY_NINETY])
        token = f"{devices.GPU_PREFIX}{FIFTY_NINETY.uuid}"
        devices.remember("voice-pipeline-dpdfnet", token)

        machine([THREE_NINETY])

        assert devices.placement("voice-pipeline-dpdfnet") == devices.CPU
        assert devices.stored_placement("voice-pipeline-dpdfnet") == token
        assert stored[devices.OPT_DEVICE_DPDFNET] == token

    def test_the_panel_is_told_the_card_is_missing(self, machine, stored):
        machine([THREE_NINETY, FIFTY_NINETY])
        token = f"{devices.GPU_PREFIX}{FIFTY_NINETY.uuid}"
        devices.remember("voice-pipeline-dpdfnet", token)
        machine([THREE_NINETY])

        found = devices.describe("voice-pipeline-dpdfnet")

        assert found["device"] == token
        assert token not in [item["token"] for item in found["devices"]]
        assert found["provider"] == devices.PROVIDER_CPU


class TestTheProviderFollowsTheChoice:
    def test_the_processor_means_the_cpu_provider(self, machine, stored):
        machine([THREE_NINETY])

        assert devices.provider_for("voice-pipeline-dpdfnet") == (
            devices.PROVIDER_CPU, 0)

    def test_a_card_means_directml_and_that_card_s_adapter(self, machine, stored):
        machine([THREE_NINETY, FIFTY_NINETY])
        devices.remember("voice-pipeline-dpdfnet",
                         f"{devices.GPU_PREFIX}{FIFTY_NINETY.uuid}")

        provider, adapter = devices.provider_for("voice-pipeline-dpdfnet")

        assert provider == devices.PROVIDER_DIRECTML
        assert adapter == FIFTY_NINETY.physical_index

    def test_a_component_that_cannot_move_is_answered_with_the_cpu(self, machine, stored):
        machine([THREE_NINETY])

        assert devices.provider_for("tts-pocket") == (devices.PROVIDER_CPU, 0)

    def test_an_absent_card_falls_back_to_the_cpu_rather_than_a_bad_adapter(
            self, machine, stored):
        machine([THREE_NINETY, FIFTY_NINETY])
        devices.remember("voice-pipeline-dpdfnet",
                         f"{devices.GPU_PREFIX}{FIFTY_NINETY.uuid}")
        machine([THREE_NINETY])

        assert devices.provider_for("voice-pipeline-dpdfnet") == (
            devices.PROVIDER_CPU, 0)


class TestTheWorkerAndThisModuleAgreeOnTheProviderNames:
    """Two files spell these strings, and only one of them can import the other.

    The worker runs inside the isolated runtime and cannot import this module,
    so the names are written twice on purpose. This is the test that keeps the
    two copies the same -- a typo in either would make every placement fall
    back to the processor while the panel said otherwise.
    """

    def test_they_are_the_same_strings(self):
        from pipeline_worker import worker

        assert worker.PROVIDER_CPU == devices.PROVIDER_CPU
        assert worker.PROVIDER_DIRECTML == devices.PROVIDER_DIRECTML


class TestVoiceVramIsNotTheBrokerSToTake:
    def test_nothing_is_reserved_until_something_is_placed(self):
        assert devices.reserved_mb() == 0
