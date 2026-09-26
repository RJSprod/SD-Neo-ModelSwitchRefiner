"""WanGP on the other card: every rule :mod:`mc_wangp` adds, with the case it exists for.

The user's rule, which the whole file is arranged around:

    WanGP's VRAM is priority one and is never taken from it. The language
    model may use what WanGP is not using, is stopped when WanGP grows into
    the room kept for it, and is never a reason WanGP runs short. A card WanGP
    is not running on is the language model's to claim in full, and a language
    model on some other card costs WanGP nothing.

Every test here is one clause of that, and every one has been run against the
change reverted: a ceiling that is not applied, a watch that never releases, a
thread cap that is not added, each fails the test that names it.
"""

from __future__ import annotations

import sys
import threading
import types

import pytest

import mc_broker
import mc_llm_context as ctx
import mc_llm_runtime as runtime
import mc_memory
import mc_wangp
from test_resource_scope import IMAGE_CARD, OTHER_CARD, Server, configuration_on

_GB = 1024**3

WANGP_UUID = "GPU-11112222-3333-4444-5555-666677778888"
"""The card Mini Paint records for WanGP. The 5090 of the report."""

IMAGE_UUID = "GPU-99998888-7777-6666-5555-444433332222"
"""Forge's card. The 3090 of the report."""

TOTAL_GB = 32.0


def readings_on(monkeypatch, **cards):
    """Fix per-card ``(free, total)``: ``readings_on(monkeypatch, gpu0=(4, 24), gpu1=(20, 32))``.

    Returns the mutable table, so a test can move a reading between two ticks
    the way a WanGP generation moves one.
    """
    amounts = {int(name[3:]): (int(free * _GB), int(total * _GB))
               for name, (free, total) in cards.items()}
    monkeypatch.setattr(mc_broker, "device_free_vram_bytes",
                        lambda index=None: amounts.get(
                            IMAGE_CARD if index is None else int(index), (0, 0))[0])
    monkeypatch.setattr(mc_broker, "free_vram_bytes", lambda: amounts.get(IMAGE_CARD, (0, 0))[0])
    monkeypatch.setattr(mc_memory, "physical_total_vram_bytes",
                        lambda index: amounts.get(int(index), (0, 0))[1])
    return amounts


def move(amounts, card, *, free):
    amounts[card] = (int(free * _GB), amounts[card][1])


class Scene:
    """One machine: Forge on GPU 0, WanGP on GPU 1, a registry of our servers."""

    def __init__(self, report, registry, readings):
        self.report = report
        self.registry = registry
        self.readings = readings

    def wangp(self, **changes):
        self.report.update(changes)
        mc_wangp.forget_presence()

    def hold(self, server, key="one"):
        self.registry._runtimes[(key,)] = server
        return server


@pytest.fixture
def scene(host, monkeypatch):
    """Mini Paint present, WanGP running on GPU 1, nothing of ours anywhere yet."""
    mc_broker.clear()
    mc_wangp.forget()
    monkeypatch.setattr(mc_broker, "safety_margin_bytes", lambda: 0)
    monkeypatch.setattr(mc_broker, "image_device_index", lambda: IMAGE_CARD)
    for family in (mc_broker.FAMILY_IMAGE, mc_broker.FAMILY_LLM):
        mc_broker.unregister_reclaimer(family)
    registry = runtime.RuntimeRegistry()
    mc_broker.register_reclaimer(mc_broker.FAMILY_LLM, registry)
    topology = {
        "by_uuid": {mc_memory._uuid_key(WANGP_UUID): OTHER_CARD,
                    mc_memory._uuid_key(IMAGE_UUID): IMAGE_CARD},
        "ordinals": {IMAGE_CARD: 0, OTHER_CARD: 1},
        "names": {IMAGE_CARD: "NVIDIA GeForce RTX 3090", OTHER_CARD: "NVIDIA GeForce RTX 5090"},
        "count": 2,
    }
    monkeypatch.setattr(mc_memory, "_cards", lambda: topology)
    report = {"version": 1, "available": True, "configured": True, "gpu_uuid": WANGP_UUID,
              "state": "READY", "running": True, "instance_id": "abc123", "generating": None}
    mc_wangp.use_source(lambda: dict(report))
    readings = readings_on(monkeypatch, gpu0=(4, 24), gpu1=(20, TOTAL_GB))
    monkeypatch.setattr(mc_wangp, "_physical_cores", lambda: 8)
    yield Scene(report, registry, readings)
    mc_wangp.use_source(None)
    mc_wangp.forget()
    registry.forget()
    mc_broker.clear()
    for family in (mc_broker.FAMILY_IMAGE, mc_broker.FAMILY_LLM):
        mc_broker.unregister_reclaimer(family)
    mc_broker.register_reclaimer(mc_broker.FAMILY_IMAGE, mc_broker._ImageReclaimer())
    mc_broker.register_reclaimer(mc_broker.FAMILY_LLM, runtime.registry)


def on_wangps_card(tmp_path, uuid=WANGP_UUID, **keywords):
    """A configuration for the LLM on WanGP's card, with the UUID setup records."""
    found = configuration_on(tmp_path, card=OTHER_CARD, **keywords)
    return runtime._replaced(found, gpu_uuid=uuid) if uuid else found


# --------------------------------------------------------------------------- #
# Presence: what Mini Paint says
# --------------------------------------------------------------------------- #


class TestPresence:
    def test_the_report_is_read_into_this_extensions_terms(self, scene):
        found = mc_wangp.presence()

        assert found.available and found.configured and found.running
        assert found.uuid == WANGP_UUID
        assert found.card == OTHER_CARD  # the UUID, placed in the topology
        assert found.generating is None
        assert found.state == "READY"

    def test_generating_is_a_tristate_and_never_a_guess(self, scene):
        scene.wangp(generating="yes")
        assert mc_wangp.presence().generating is None
        scene.wangp(generating=True)
        assert mc_wangp.presence().generating is True
        scene.wangp(generating=False)
        assert mc_wangp.presence().generating is False

    def test_no_mini_paint_is_not_applicable_everywhere(self, scene, tmp_path):
        """The installation without Mini Paint NEO, which is most of them:
        nothing here may change a single placement on it."""
        mc_wangp.use_source(lambda: None)

        assert not mc_wangp.presence().available
        assert mc_wangp.cap_bytes(OTHER_CARD, on_wangps_card(tmp_path)) == -1
        assert mc_wangp.thread_flags(on_wangps_card(tmp_path),
                                     ctx.Placement(gpu_layers=20), lambda flag: True) == []
        assert mc_wangp.describe() == ""
        assert mc_wangp.Watch().arm() is False
        assert runtime._spendable(0, OTHER_CARD, configuration=on_wangps_card(tmp_path)) == 20 * _GB

    def test_a_package_that_was_never_imported_is_not_looked_for(self, scene, monkeypatch,
                                                                tmp_path):
        """By name and only once the host has loaded it. An extension that is
        on disk but not in the process -- disabled, or not yet loaded -- is
        not imported from here, however importable it is."""
        package = tmp_path / "minipaint_neo"
        (package / "wangp").mkdir(parents=True)
        (package / "__init__.py").write_text("", encoding="utf-8")
        (package / "wangp" / "__init__.py").write_text("", encoding="utf-8")
        (package / "wangp" / "presence.py").write_text(
            "def report():\n    return {'available': True, 'configured': True, 'running': True,"
            " 'gpu_uuid': 'GPU-1', 'state': 'READY'}\n", encoding="utf-8")
        monkeypatch.syspath_prepend(str(tmp_path))
        for name in list(sys.modules):
            if name == mc_wangp.PACKAGE or name.startswith(mc_wangp.PACKAGE + "."):
                monkeypatch.delitem(sys.modules, name)
        mc_wangp.use_source(None)

        assert mc_wangp.presence(fresh=True) == mc_wangp.Presence()
        assert mc_wangp.PACKAGE not in sys.modules

    def test_the_presence_module_is_preferred_when_mini_paint_has_one(self, scene, monkeypatch):
        mc_wangp.use_source(None)
        package = types.ModuleType(mc_wangp.PACKAGE)
        presence = types.ModuleType(mc_wangp.PRESENCE_MODULE)
        presence.report = lambda: {"available": True, "configured": True, "gpu_uuid": WANGP_UUID,
                                   "state": "READY", "running": True, "instance_id": "x",
                                   "generating": True, "source": "presence"}
        monkeypatch.setitem(sys.modules, mc_wangp.PACKAGE, package)
        monkeypatch.setitem(sys.modules, mc_wangp.PRESENCE_MODULE, presence)

        found = mc_wangp.presence(fresh=True)

        assert found.running and found.generating is True and found.source == "presence"

    def test_an_older_mini_paint_is_read_through_its_runtime(self, scene, monkeypatch):
        """No ``presence`` module: the snapshot, the saved setup and the cached
        hello it has always had, in the report's shape, without guessing."""
        mc_wangp.use_source(None)
        package = types.ModuleType(mc_wangp.PACKAGE)
        wangp = types.ModuleType("minipaint_neo.wangp")
        runtime_module = types.ModuleType(mc_wangp.RUNTIME_MODULE)
        runtime_module.snapshot = lambda: {"state": "READY", "running": True, "instance_id": "r1",
                                           "port_bound": True}
        config_module = types.ModuleType(mc_wangp.CONFIG_MODULE)
        config_module.load = lambda: types.SimpleNamespace(initialized=True,
                                                           gpu={"uuid": WANGP_UUID})
        control_module = types.ModuleType(mc_wangp.CONTROL_MODULE)
        control_module.last_hello = lambda: {"generation_running": False}
        for name, module in ((mc_wangp.PACKAGE, package), ("minipaint_neo.wangp", wangp),
                             (mc_wangp.RUNTIME_MODULE, runtime_module),
                             (mc_wangp.CONFIG_MODULE, config_module),
                             (mc_wangp.CONTROL_MODULE, control_module)):
            monkeypatch.setitem(sys.modules, name, module)
        monkeypatch.delitem(sys.modules, mc_wangp.PRESENCE_MODULE, raising=False)

        found = mc_wangp.presence(fresh=True)

        assert found.available and found.configured and found.running
        assert found.uuid == WANGP_UUID and found.card == OTHER_CARD
        assert found.generating is False
        assert found.source == "runtime"

    def test_a_reading_is_reused_within_the_second_and_refreshed_on_request(self, scene):
        assert mc_wangp.presence().running
        scene.report["running"] = False  # without forgetting the cache

        assert mc_wangp.presence().running  # a second-old answer, by design
        assert not mc_wangp.presence(fresh=True).running

    def test_a_report_that_raises_is_no_mini_paint(self, scene):
        def broken():
            raise RuntimeError("the runtime is wedged")

        mc_wangp.use_source(broken)

        assert mc_wangp.presence(fresh=True) == mc_wangp.Presence()


class TestWhichCardIsWanGPs:
    def test_a_matching_uuid_settles_it(self, scene, tmp_path):
        assert mc_wangp.is_wangps_card(OTHER_CARD, on_wangps_card(tmp_path))

    def test_a_different_uuid_settles_it_the_other_way_whatever_the_index_says(
            self, scene, tmp_path):
        """Two cards can both be "card 1" in two namespaces. The UUID is what
        both sides agree on, and it wins over an index that happens to match."""
        assert not mc_wangp.is_wangps_card(OTHER_CARD, on_wangps_card(tmp_path, uuid=IMAGE_UUID))

    def test_without_a_uuid_of_ours_the_physical_index_decides(self, scene, tmp_path):
        """An older state file recorded no UUID. Its index is nvidia-smi's, and
        so is the one WanGP's UUID translates to."""
        assert mc_wangp.is_wangps_card(OTHER_CARD, on_wangps_card(tmp_path, uuid=""))
        assert not mc_wangp.is_wangps_card(IMAGE_CARD, configuration_on(tmp_path, card=IMAGE_CARD))

    def test_a_card_nobody_can_place_is_not_wangps(self, scene, tmp_path, monkeypatch):
        """Unanswerable is no: the cost of that mistake is the old behaviour on
        one card, the cost of the opposite is a language model shrunk for a
        WanGP it never shared a card with."""
        monkeypatch.setattr(mc_memory, "_cards", lambda: {"by_uuid": {}, "ordinals": {},
                                                          "names": {}, "count": 0})
        mc_wangp.forget_presence()

        assert not mc_wangp.is_wangps_card(OTHER_CARD, on_wangps_card(tmp_path, uuid=""))
        # ...while a UUID of ours still settles it without any topology at all.
        assert mc_wangp.is_wangps_card(OTHER_CARD, on_wangps_card(tmp_path))

    def test_a_wangp_that_is_not_running_owns_no_card(self, scene, tmp_path):
        scene.wangp(running=False)

        assert not mc_wangp.is_wangps_card(OTHER_CARD, on_wangps_card(tmp_path))

    def test_the_setting_off_owns_no_card(self, scene, host, tmp_path):
        host.shared.opts.set(mc_wangp.OPT_MODE, mc_wangp.MODE_OFF)

        assert not mc_wangp.is_wangps_card(OTHER_CARD, on_wangps_card(tmp_path))


# --------------------------------------------------------------------------- #
# The ceiling
# --------------------------------------------------------------------------- #


class TestTheCeiling:
    def test_the_card_less_what_wangp_holds_less_the_reserve(self, scene, tmp_path):
        """32 GB card, 20 GB free, nothing of ours: WanGP and the driver hold
        12, four stay free for WanGP's next allocation, 16 are the LLM's."""
        assert mc_wangp.cap_bytes(OTHER_CARD, on_wangps_card(tmp_path)) == 16 * _GB

    def test_our_own_server_is_not_mistaken_for_wangp(self, scene, tmp_path):
        """The same card with 6 GB of ours on it and 14 free: WanGP still holds
        12, and the ceiling is still 16 -- the 6 are ours to spend again."""
        scene.hold(Server(card=OTHER_CARD, holds=6 * _GB))
        move(scene.readings, OTHER_CARD, free=14)

        assert mc_wangp.cap_bytes(OTHER_CARD, on_wangps_card(tmp_path)) == 16 * _GB

    def test_the_peak_is_remembered_for_the_session(self, scene, tmp_path):
        """A video model is loaded and unloaded around every generation. The
        idle reading is the one number a placement must not be sized against."""
        move(scene.readings, OTHER_CARD, free=6)   # generating: 26 GB taken
        assert mc_wangp.cap_bytes(OTHER_CARD, on_wangps_card(tmp_path)) == 2 * _GB
        move(scene.readings, OTHER_CARD, free=30)  # idle again: 2 GB taken

        assert mc_wangp.cap_bytes(OTHER_CARD, on_wangps_card(tmp_path)) == 2 * _GB

    def test_the_reserve_is_the_setting(self, scene, host, tmp_path):
        host.shared.opts.set(mc_wangp.OPT_RESERVE_GB, 8)

        assert mc_wangp.cap_bytes(OTHER_CARD, on_wangps_card(tmp_path)) == 12 * _GB

    def test_a_card_that_cannot_be_sized_has_no_ceiling(self, scene, tmp_path):
        scene.readings[OTHER_CARD] = (20 * _GB, 0)

        assert mc_wangp.cap_bytes(OTHER_CARD, on_wangps_card(tmp_path)) == -1

    def test_the_ceiling_is_said_once_per_change_not_once_per_rung(self, scene, tmp_path,
                                                                    caplog):
        import logging

        with caplog.at_level(logging.INFO, logger="model_chain"):
            for _ in range(4):
                mc_wangp.cap_bytes(OTHER_CARD, on_wangps_card(tmp_path))
            move(scene.readings, OTHER_CARD, free=10)
            mc_wangp.cap_bytes(OTHER_CARD, on_wangps_card(tmp_path))

        said = [record for record in caplog.records if "WanGP is running on GPU 1" in record.message]
        assert len(said) == 2
        assert "up to 16.0 GB" in said[0].message and "up to 6.0 GB" in said[1].message


class TestWhatAPlacementMaySpend:
    """The ceiling reaching the ladder: every rung reads ``_spendable``."""

    def test_on_wangps_card_the_ceiling_is_what_may_be_spent(self, scene, tmp_path):
        assert runtime._spendable(0, OTHER_CARD, configuration=on_wangps_card(tmp_path)) == 16 * _GB

    def test_a_server_of_ours_being_replaced_spends_its_own_bytes_again(self, scene, tmp_path):
        scene.hold(Server(card=OTHER_CARD, holds=6 * _GB))
        move(scene.readings, OTHER_CARD, free=14)

        assert runtime._spendable(6 * _GB, OTHER_CARD,
                                  configuration=on_wangps_card(tmp_path)) == 16 * _GB

    def test_the_reserve_comes_off_what_wangp_holds_now_as_much_as_off_its_peak(
            self, scene, tmp_path):
        """A learned peak of 12 says nothing about a WanGP that is holding 27
        this instant: 5 GB free is not 5 GB to spend, it is 1 -- the four the
        reserve keeps come off whichever figure is larger."""
        mc_wangp.cap_bytes(OTHER_CARD, on_wangps_card(tmp_path))  # peak 12 on record
        move(scene.readings, OTHER_CARD, free=5)

        assert runtime._spendable(0, OTHER_CARD, configuration=on_wangps_card(tmp_path)) == 1 * _GB

    def test_the_other_card_is_untouched(self, scene, tmp_path):
        """An LLM on any other card costs WanGP nothing, and WanGP costs it
        nothing: Forge's card is placed against exactly as before."""
        assert runtime._spendable(0, IMAGE_CARD,
                                  configuration=configuration_on(tmp_path, card=IMAGE_CARD)) == 4 * _GB

    def test_a_wangp_that_is_not_running_leaves_the_whole_card(self, scene, tmp_path):
        scene.wangp(running=False)

        assert runtime._spendable(0, OTHER_CARD, configuration=on_wangps_card(tmp_path)) == 20 * _GB

    def test_the_setting_off_leaves_the_whole_card(self, scene, host, tmp_path):
        host.shared.opts.set(mc_wangp.OPT_MODE, mc_wangp.MODE_OFF)

        assert runtime._spendable(0, OTHER_CARD, configuration=on_wangps_card(tmp_path)) == 20 * _GB

    def test_llm_priority_does_not_override_wangp(self, scene, tmp_path):
        """LLM priority releases *image* residency for the language model.
        WanGP's is not image residency and is not anybody's to release."""
        assert runtime._spendable(0, OTHER_CARD, image_budget=False,
                                  configuration=on_wangps_card(tmp_path)) == 16 * _GB

    def test_the_ladder_places_under_the_ceiling(self, scene, tmp_path, monkeypatch):
        """End to end through negotiate: a 20 GB-free card with WanGP's
        12 GB on it and a 4 GB reserve places a model that would have fit in
        20 as one that has to fit in 16."""
        from test_llm_context import build_model

        model = build_model(tmp_path, blocks=32, size_mb=4)
        server = tmp_path / "llama-server"
        server.write_bytes(b"")
        configuration = runtime._replaced(
            runtime.Config(runtime=server, model=model, mmproj=None, gpu_index=OTHER_CARD,
                           device="CUDA1", gpu_layers="all", context_size=8192,
                           context_mode="fixed", context_buffer_gb=4.0, kv_type_k="f16",
                           kv_type_v="f16"),
            gpu_uuid=WANGP_UUID)
        monkeypatch.setattr(ctx, "estimate", lambda model, placement, header: ctx.Estimate(
            model=model, context=placement.context, ceiling=0,
            weights_bytes=18 * _GB if placement.gpu_layers == ctx.ALL_LAYERS else 8 * _GB,
            kv_bytes=int(0.5 * _GB), compute_bytes=int(0.5 * _GB), kv_bytes_per_token=0.0,
            calibrated=False, placement=placement))

        negotiated = runtime.negotiate(configuration)

        assert negotiated.placement.gpu_layers != ctx.ALL_LAYERS  # 19 GB does not fit in 16
        scene.wangp(running=False)
        assert runtime.negotiate(configuration).placement.gpu_layers == ctx.ALL_LAYERS


# --------------------------------------------------------------------------- #
# The watch
# --------------------------------------------------------------------------- #


@pytest.fixture
def watched(scene):
    """Ours on both cards: 8 GB on WanGP's, 4 GB on Forge's; WanGP idle at 4 GB."""
    ours = scene.hold(Server(card=OTHER_CARD, holds=8 * _GB), "wangp-card")
    theirs = scene.hold(Server(card=IMAGE_CARD, holds=4 * _GB), "image-card")
    move(scene.readings, OTHER_CARD, free=20)  # 32 - 8 ours - 4 WanGP
    return scene, ours, theirs, mc_wangp.Watch()


class TestTheWatch:
    def test_growing_into_the_reserve_stops_the_server_on_that_card_only(self, watched):
        scene, ours, theirs, watch = watched
        move(scene.readings, OTHER_CARD, free=3)  # WanGP is loading into the reserve

        assert watch.tick() == mc_wangp.RELEASED
        assert ours.stopped and not ours.up
        assert not theirs.stopped  # Forge's card is not WanGP's business
        assert any("grown into the 4.0 GB kept for it" in decision.text
                   for decision in mc_broker.decisions())
        assert mc_wangp.learned(WANGP_UUID)

    def test_a_server_that_fits_beside_wangp_stays(self, watched):
        scene, ours, theirs, watch = watched

        assert watch.tick() == mc_wangp.KEPT
        assert not ours.stopped and ours.up

    def test_a_known_peak_that_leaves_no_room_stops_it(self, watched, tmp_path):
        """Placed while WanGP was off, and WanGP came back: its peak from
        earlier in the session -- seen by a placement that found it at 26 GB
        and went to system RAM -- is on record and leaves 2 GB where the
        server now holds 8."""
        scene, ours, theirs, watch = watched
        ours.holds = 0
        move(scene.readings, OTHER_CARD, free=6)
        assert mc_wangp.cap_bytes(OTHER_CARD, on_wangps_card(tmp_path)) == 2 * _GB
        scene.wangp(running=False)
        ours.holds = 8 * _GB  # placed with WanGP off: the whole card was its
        move(scene.readings, OTHER_CARD, free=22)
        assert watch.tick() == mc_wangp.KEPT
        scene.wangp(running=True)  # back, idle at 2 GB

        assert watch.tick() == mc_wangp.RELEASED
        assert ours.stopped
        assert any("peak on GPU 1" in decision.text for decision in mc_broker.decisions())

    def test_a_first_generation_on_an_unmeasured_card_gets_the_whole_card(self, watched):
        scene, ours, theirs, watch = watched
        scene.wangp(generating=False)
        assert watch.tick() == mc_wangp.KEPT
        scene.wangp(generating=True)

        assert watch.tick() == mc_wangp.RELEASED
        assert ours.stopped
        assert any("has not been measured" in decision.text for decision in mc_broker.decisions())

    def test_a_generation_seen_through_makes_the_peak_trusted(self, watched):
        """Idle, generating, idle: the peak is measured now, and a server that
        fits under it is not sent away by the next generation."""
        scene, ours, theirs, watch = watched
        ours.holds = 0
        scene.wangp(generating=False)
        watch.tick()
        scene.wangp(generating=True)
        move(scene.readings, OTHER_CARD, free=12)  # 20 GB at its peak
        watch.tick()
        scene.wangp(generating=False)
        move(scene.readings, OTHER_CARD, free=28)
        watch.tick()
        assert mc_wangp.learned(WANGP_UUID)
        ours.holds = 6 * _GB  # under the 8 GB the peak leaves
        move(scene.readings, OTHER_CARD, free=22)
        scene.wangp(generating=True)

        assert watch.tick() == mc_wangp.KEPT
        assert not ours.stopped

    def test_a_card_first_seen_mid_generation_is_not_sent_away_for_it(self, watched):
        """A server placed while a generation was running was sized against
        that generation. Unknown-to-generating is not idle-to-generating."""
        scene, ours, theirs, watch = watched
        scene.wangp(generating=True)

        assert watch.tick() == mc_wangp.KEPT
        assert not ours.stopped

    def test_wangp_off_is_nothing_to_protect_but_worth_waiting_for(self, watched):
        scene, ours, theirs, watch = watched
        scene.wangp(running=False)
        move(scene.readings, OTHER_CARD, free=1)  # somebody else's, not WanGP's

        assert watch.tick() == mc_wangp.KEPT
        assert not ours.stopped

    def test_nothing_of_ours_on_any_card_ends_the_watch(self, scene):
        assert mc_wangp.Watch().tick() == mc_wangp.STOP

    def test_the_setting_off_ends_the_watch(self, watched, host):
        scene, ours, theirs, watch = watched
        host.shared.opts.set(mc_wangp.OPT_MODE, mc_wangp.MODE_OFF)
        move(scene.readings, OTHER_CARD, free=1)

        assert watch.tick() == mc_wangp.STOP
        assert not ours.stopped

    def test_no_mini_paint_ends_the_watch(self, watched):
        scene, ours, theirs, watch = watched
        mc_wangp.use_source(lambda: None)

        assert watch.tick() == mc_wangp.STOP

    def test_a_processor_server_is_never_a_reason_to_read_a_card(self, scene):
        scene.hold(Server(cpu=True, holds=0))

        assert mc_wangp.Watch().tick() == mc_wangp.STOP

    def test_a_server_in_system_ram_on_wangps_card_holds_nothing_to_give(self, scene):
        """Placed to zero layers, it has a CUDA context on WanGP's card and no
        VRAM to surrender: stopping it would buy WanGP nothing, so there is
        nothing to watch and nothing is stopped, however full the card is."""
        held = scene.hold(Server(card=OTHER_CARD, holds=0))
        move(scene.readings, OTHER_CARD, free=1)

        assert mc_wangp.Watch().tick() == mc_wangp.STOP
        assert not held.stopped and held.up

    def test_one_thread_that_ends_itself(self, watched):
        """Armed twice is one thread; a server gone is a thread gone."""
        scene, ours, theirs, watch = watched
        held = threading.Event()
        watch._sleep = lambda seconds: held.wait(2.0)

        assert watch.arm() is True
        first = watch._thread
        assert watch.arm() is True
        assert watch._thread is first and watch.watching
        scene.registry.forget()
        held.set()
        first.join(3.0)

        assert not watch.watching
        assert watch.ticks >= 1

    def test_the_watch_is_armed_when_a_server_is_handed_out(self, scene, tmp_path, monkeypatch):
        """From both doors of ``client``: a start, and a reuse a message later."""
        from test_llm_runtime import FakeProcess, configure, set_free

        armed = []
        monkeypatch.setattr(mc_wangp, "watch", lambda: armed.append(1))
        mc_wangp.use_source(lambda: None)  # keep the fake start off WanGP's rules
        import mc_llm_paths

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
            assert armed == [1]
            managed.client()
            assert armed == [1, 1]
        finally:
            managed.stop()
            ctx.forget()


# --------------------------------------------------------------------------- #
# Threads
# --------------------------------------------------------------------------- #


class TestTheThreadCap:
    def everything(self, flag):
        return True

    def test_a_partial_offload_is_held_to_half_the_cores_while_wangp_runs(self, scene, tmp_path):
        flags = mc_wangp.thread_flags(on_wangps_card(tmp_path), ctx.Placement(gpu_layers=20),
                                      self.everything)

        assert flags == ["--threads", "4", "--threads-batch", "4"]

    def test_the_setting_names_the_count(self, scene, host, tmp_path):
        host.shared.opts.set(mc_wangp.OPT_THREADS, 3)

        assert mc_wangp.thread_flags(on_wangps_card(tmp_path), ctx.Placement(gpu_layers=20),
                                     self.everything) == ["--threads", "3", "--threads-batch", "3"]

    def test_a_full_offload_touches_no_core_and_is_left_alone(self, scene, tmp_path):
        assert mc_wangp.thread_flags(on_wangps_card(tmp_path), ctx.Placement(),
                                     self.everything) == []

    def test_a_processor_placement_and_an_expert_spill_both_count(self, scene, tmp_path):
        found = on_wangps_card(tmp_path)
        assert mc_wangp.thread_flags(found, ctx.Placement(on_gpu=False), self.everything)
        assert mc_wangp.thread_flags(found, ctx.Placement(cpu_expert_layers=6), self.everything)
        assert mc_wangp.thread_flags(found, ctx.Placement(uma=True), self.everything) == []

    def test_the_processor_is_the_machines_so_the_other_card_is_capped_too(self, scene, tmp_path):
        """WanGP needs cores whichever card the LLM is on."""
        flags = mc_wangp.thread_flags(configuration_on(tmp_path, card=IMAGE_CARD),
                                      ctx.Placement(gpu_layers=20), self.everything)

        assert flags == ["--threads", "4", "--threads-batch", "4"]

    def test_wangp_off_or_the_setting_off_caps_nothing(self, scene, host, tmp_path):
        scene.wangp(running=False)
        assert mc_wangp.thread_flags(on_wangps_card(tmp_path), ctx.Placement(gpu_layers=20),
                                     self.everything) == []
        scene.wangp(running=True)
        host.shared.opts.set(mc_wangp.OPT_MODE, mc_wangp.MODE_OFF)
        assert mc_wangp.thread_flags(on_wangps_card(tmp_path), ctx.Placement(gpu_layers=20),
                                     self.everything) == []

    def test_only_the_spelling_the_build_advertises(self, scene, tmp_path):
        flags = mc_wangp.thread_flags(on_wangps_card(tmp_path), ctx.Placement(gpu_layers=20),
                                      lambda flag: flag == mc_wangp.THREADS_FLAG)

        assert flags == ["--threads", "4"]

    def test_the_cap_reaches_the_command_line(self, scene, tmp_path, monkeypatch):
        monkeypatch.setattr(runtime, "runtime_supports",
                            lambda flag, configuration=None: flag in (
                                mc_wangp.THREADS_FLAG, mc_wangp.THREADS_BATCH_FLAG))

        flags = runtime._launch_flags(on_wangps_card(tmp_path), ctx.Placement(gpu_layers=20), None)

        assert flags[-4:] == ["--threads", "4", "--threads-batch", "4"]
        scene.wangp(running=False)
        assert "--threads" not in runtime._launch_flags(on_wangps_card(tmp_path),
                                                        ctx.Placement(gpu_layers=20), None)


# --------------------------------------------------------------------------- #
# Saying so
# --------------------------------------------------------------------------- #


class TestSayingSo:
    def test_the_stray_on_wangps_card_is_named(self, scene, monkeypatch):
        assert "WanGP" in mc_broker.stray_explanation(OTHER_CARD)
        assert "WanGP" not in mc_broker.stray_explanation(IMAGE_CARD)
        assert "WanGP" not in mc_broker.stray_explanation()
        monkeypatch.setattr(mc_broker, "unaccounted_bytes", lambda card=mc_broker.ANY_CARD: 12 * _GB)
        assert "WanGP" in mc_broker._unaccounted_note(OTHER_CARD)
        assert "nvidia-smi" in mc_broker._unaccounted_note(IMAGE_CARD)

    def test_the_panel_line(self, scene, host, monkeypatch):
        reads = []
        original = mc_broker.device_free_vram_bytes
        monkeypatch.setattr(mc_broker, "device_free_vram_bytes",
                            lambda index=None: (reads.append(index), original(index))[1])
        said = mc_wangp.describe()
        # Nothing of ours on WanGP's card and no configuration for it: the
        # panel must not be what first reads that card from this process.
        assert "running on GPU 1" in said and "nothing here reads that card" in said
        assert reads == []
        scene.hold(Server(card=OTHER_CARD, holds=6 * _GB))
        move(scene.readings, OTHER_CARD, free=14)
        said = mc_wangp.describe()
        assert "peak 12.0 GB" in said and "up to 16.0 GB" in said and "holds 6.0 GB" in said
        assert reads == [OTHER_CARD]
        scene.wangp(generating=True)
        assert "and generating" in mc_wangp.describe()
        scene.wangp(running=False)
        assert "not running" in mc_wangp.describe()
        scene.wangp(configured=False)
        assert "not set up" in mc_wangp.describe()
        host.shared.opts.set(mc_wangp.OPT_MODE, mc_wangp.MODE_OFF)
        assert "off" in mc_wangp.describe()

    def test_the_residency_panel_carries_it(self, scene, monkeypatch):
        import mc_llm_studio

        monkeypatch.setattr(mc_broker, "total_vram_bytes", lambda: 24 * _GB)
        scene.hold(Server(card=OTHER_CARD, holds=6 * _GB))

        assert "WanGP: running on GPU 1" in mc_llm_studio._residency_table()
        mc_wangp.use_source(lambda: None)
        assert "WanGP" not in mc_llm_studio._residency_table()


class TestSettings:
    def test_the_mode_resolves_either_half_of_the_pair(self, scene, host):
        host.shared.opts.set(mc_wangp.OPT_MODE, mc_wangp.MODES[1][1])
        assert mc_wangp.mode() == mc_wangp.MODE_OFF
        host.shared.opts.set(mc_wangp.OPT_MODE, "nonsense")
        assert mc_wangp.mode() == mc_wangp.MODE_SHARE

    def test_the_reserve_and_the_threads_read_as_numbers(self, scene, host):
        host.shared.opts.set(mc_wangp.OPT_RESERVE_GB, "2.5")
        assert mc_wangp.reserve_bytes() == int(2.5 * _GB)
        host.shared.opts.set(mc_wangp.OPT_RESERVE_GB, -3)
        assert mc_wangp.reserve_bytes() == 0
        host.shared.opts.set(mc_wangp.OPT_THREADS, "6")
        assert mc_wangp.thread_count() == 6
        host.shared.opts.set(mc_wangp.OPT_THREADS, 0)
        assert mc_wangp.thread_count() == 4

    def test_the_settings_page_offers_all_three(self, host):
        import importlib

        import model_chain  # noqa: F401  -- registers the options on import

        registered = host.shared.options_templates
        for name in (mc_wangp.OPT_MODE, mc_wangp.OPT_RESERVE_GB, mc_wangp.OPT_THREADS):
            assert name in registered, name
