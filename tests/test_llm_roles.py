"""Creative and Spatial as two independently configurable LLM roles.

Structured around the design intent's own acceptance tests (section 22), plus
the two things the specification got wrong about this repository and the one
thing the user added to it afterwards:

* section 5 asks for legacy ``mixed`` to migrate to Mixed *Conservative*. In
  this branch Mixed asks the ladder for every layer and takes what it is given,
  which is what the new vocabulary calls Aggressive -- so migrating the label
  to Conservative would change what an installed machine does. The spec's own
  behavioural rule ("preserve what the old saved option DID") is what decides
  it, and these tests pin the behaviour rather than the name;
* section 15 asks for the old configuration to be *copied* into both roles.
  Here a role that has not been split inherits it instead, which reaches the
  same end state -- one identity, one server, unchanged behaviour -- without a
  migration that has to run correctly to get there;
* the sharing choice applies whenever two differently-configured roles land in
  the same memory, not only when their identities are equal, because two
  servers in one pool is exactly the case somebody wants to decide about.
"""

from __future__ import annotations

import pathlib

import pytest

import mc_broker
import mc_llm_context as ctx
import mc_llm_roles as roles
import mc_llm_runtime as runtime
import mc_llm_setup as setup
from prompt_master.core.models import GpuInfo
from prompt_master.inference import device_detection as detection

_GB = 1024**3


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    """A registry and a register that no other test has written into."""
    mc_broker.clear()
    runtime.registry.forget()
    yield
    runtime.registry.forget()
    mc_broker.clear()


@pytest.fixture
def two_cards(monkeypatch):
    """Two cards, and no device list cached from another test.

    ``mc_llm_setup.devices`` keeps its answer for a while -- a scan costs a
    subprocess -- so patching the detection boundary alone leaves whatever the
    previous test saw in front of it.
    """
    cards = [GpuInfo(0, "uuid-a", "NVIDIA GeForce RTX 3090", 24576, 23000, "570"),
             GpuInfo(1, "uuid-b", "NVIDIA GeForce RTX 5090", 32768, 32000, "570")]
    monkeypatch.setattr(detection, "detect_gpus", lambda timeout=15: list(cards))
    setup.forget_devices()
    yield cards
    setup.forget_devices()


def configured(tmp_path, **over):
    """One role's resolved configuration, with sensible defaults for the rest."""
    server = tmp_path / "llama-server"
    if not server.exists():
        server.write_bytes(b"")
    model = tmp_path / str(over.pop("model_name", "A.gguf"))
    if not model.exists():
        model.write_bytes(b"")
    base = dict(runtime=server, model=model, mmproj=None, gpu_index=0, device="CUDA0",
                gpu_layers="all", context_size=8192, context_mode="fixed",
                context_buffer_gb=4.0, kv_type_k="f16", kv_type_v="f16", mode="gpu")
    base.update(over)
    return runtime.Config(**base)


def pair(monkeypatch, creative, spatial, shared=None):
    """Make ``config(role)`` answer with these two, and register a fresh registry."""
    monkeypatch.setattr(runtime, "config",
                        lambda role="": {roles.CREATIVE: creative,
                                         roles.SPATIAL: spatial}.get(role, shared or creative))
    return runtime.RuntimeRegistry()


# --------------------------------------------------------------------------- #
# A. Device enumeration (section 22 A, section 4)
# --------------------------------------------------------------------------- #


class TestEveryCardIsOfferedThreeWays:
    def test_two_cards_and_a_processor_is_seven_options(self, two_cards):
        offered = detection.detect_devices()

        assert len(offered) == 7
        assert [device.mode for device in offered] == [
            "gpu", "mixed_aggressive", "mixed_conservative",
            "gpu", "mixed_aggressive", "mixed_conservative",
            "cpu",
        ]

    def test_the_three_variants_of_one_card_have_three_tokens(self, two_cards):
        tokens = [detection.device_token(found) for found in detection.detect_devices()]

        assert len(tokens) == len(set(tokens))
        assert {"gpu:0", "mixed_aggressive:0", "mixed_conservative:0"} <= set(tokens)

    def test_nothing_is_hard_coded_to_a_3090_or_a_5090(self, monkeypatch):
        """Section 4: "must not hard-code 3090 and 5090 as the only eligible
        cards." Any card the existing detection returns gets the same three."""
        monkeypatch.setattr(detection, "detect_gpus",
                            lambda timeout=15: [GpuInfo(0, "u", "NVIDIA T400", 4096, 4000, "570")])
        offered = detection.detect_devices()

        assert [found.mode for found in offered] == [
            "gpu", "mixed_aggressive", "mixed_conservative", "cpu"]


# --------------------------------------------------------------------------- #
# Legacy migration (sections 5 and 22 I)
# --------------------------------------------------------------------------- #


class TestTheOldMixedModeKeepsDoingWhatItDid:
    """The specification says map ``mixed`` to Conservative. This branch's Mixed
    asks for every layer and lets the ladder take it apart, which is Aggressive.

    Mixed *was* pinned at zero layers once, and was deliberately changed when a
    machine with a 3090 in it turned out to be running every matrix multiply on
    the processor at 4.2 tokens a second. Migrating the label rather than the
    behaviour would restore that quietly, on a working installation, which is
    the one thing section 5's own compatibility rule forbids.
    """

    def test_the_stored_word_migrates_to_aggressive(self):
        from prompt_master.core.models import normalise_mode

        assert normalise_mode("mixed") == "mixed_aggressive"

    def test_a_saved_menu_value_still_names_a_device(self, two_cards):
        """Every installation configured before the split has "mixed:0" saved.
        Resolving it to nothing drops the menu to its first entry -- the full
        offload -- so a card that was being kept free would start filling."""
        found = detection.device_for_token("mixed:0", detection.detect_devices())

        assert found is not None and found.mode == "mixed_aggressive"

    def test_a_file_with_no_mode_at_all_is_read_the_same_way(self):
        assert detection.recorded_mode("", "CUDA0", "0") == "mixed_aggressive"
        assert detection.recorded_mode("", "CUDA0", "all") == "gpu"
        assert detection.recorded_mode("", "none", "0") == "cpu"

    def test_an_aggressive_config_still_asks_the_ladder_for_everything(self, tmp_path):
        """The behaviour the label is being preserved for."""
        configuration = configured(tmp_path, mode="mixed_aggressive")

        assert runtime.is_mixed(configuration)
        assert not runtime.is_conservative(configuration)
        assert runtime._requested_placement(configuration, None).gpu_layers == ctx.ALL_LAYERS


# --------------------------------------------------------------------------- #
# D/E/F. The placement each mode asks for (section 22 D, E, F)
# --------------------------------------------------------------------------- #


class TestWhatEachModeAsksFor:
    def test_conservative_asks_for_no_layers_and_is_never_promoted(self, tmp_path):
        """Section 12.2: "automatic placement must not promote model layers into
        VRAM simply because free VRAM is visible." The user answered that
        question in the menu; the ladder does not get to answer it again."""
        configuration = configured(tmp_path, mode="mixed_conservative")
        placement = runtime._requested_placement(configuration, None)

        assert runtime.is_conservative(configuration)
        assert not runtime.is_mixed(configuration)
        assert placement.gpu_layers == ctx.NO_LAYERS
        assert not placement.on_gpu

    def test_conservative_still_counts_as_a_cuda_installation(self, tmp_path):
        """The distinction section 12.2 asks for. One boolean cannot mean both
        "uses the card" and "has weights on the card", and Conservative is the
        mode that makes the difference visible."""
        configuration = configured(tmp_path, mode="mixed_conservative")

        assert configuration.uses_cuda_compute
        assert not configuration.on_gpu

    def test_cpu_mode_uses_no_card_at_all(self, tmp_path):
        configuration = configured(tmp_path, mode="cpu", device="none")

        assert not configuration.uses_cuda_compute
        assert not configuration.on_gpu
        assert runtime.pool(configuration) == runtime.POOL_SYSTEM_RAM

    def test_gpu_mode_asks_for_the_whole_model(self, tmp_path):
        configuration = configured(tmp_path, mode="gpu")

        assert configuration.on_gpu
        assert runtime._requested_placement(configuration, None).gpu_layers == ctx.ALL_LAYERS


class TestConservativeCommandLine:
    @pytest.fixture
    def build(self, monkeypatch):
        def announce(*flags):
            monkeypatch.setattr(runtime, "runtime_supports",
                                lambda flag, config=None, offered=flags: flag in offered)
        return announce

    def test_it_keeps_the_cache_off_the_card(self, tmp_path, build):
        build(runtime.NO_KV_OFFLOAD_FLAG, runtime.OP_OFFLOAD_FLAG)
        configuration = configured(tmp_path, mode="mixed_conservative")

        assert runtime.conservative_flags(configuration) == [
            runtime.NO_KV_OFFLOAD_FLAG, runtime.OP_OFFLOAD_FLAG]

    def test_no_other_mode_gets_them(self, tmp_path, build):
        build(runtime.NO_KV_OFFLOAD_FLAG, runtime.OP_OFFLOAD_FLAG)

        for mode in ("gpu", "mixed_aggressive"):
            assert runtime.conservative_flags(configured(tmp_path, mode=mode)) == [], mode
        assert runtime.conservative_flags(
            configured(tmp_path, mode="cpu", device="none")) == []

    def test_a_build_without_them_keeps_the_zero_layer_promise(self, tmp_path, build):
        """Section 21: an unsupported optional flag is left off and the reduced
        capability reported -- never a reason to abandon the mode itself."""
        build()
        configuration = configured(tmp_path, mode="mixed_conservative")

        assert runtime.conservative_flags(configuration) == []
        assert runtime._requested_placement(configuration, None).gpu_layers == ctx.NO_LAYERS

    def test_they_reach_the_command_line(self, tmp_path, build):
        build(runtime.NO_KV_OFFLOAD_FLAG, runtime.FULL_ATTENTION_WINDOW_FLAG)
        configuration = configured(tmp_path, mode="mixed_conservative")
        placement = runtime._requested_placement(configuration, None)

        assert runtime.NO_KV_OFFLOAD_FLAG in runtime.accelerator_flags(configuration, placement)


# --------------------------------------------------------------------------- #
# G/H. One runtime or two (sections 10, 20, 22 G and H)
# --------------------------------------------------------------------------- #


class TestWhenTwoRolesShareAServer:
    """Section 10.1: "Do NOT share a runtime merely because both roles say CPU."

    Each case below is one of section 20's worked examples, and the expected
    number of runtimes is the number that section states.
    """

    def test_E_identical_configurations_share_one(self, tmp_path, monkeypatch):
        same = dict(mode="mixed_conservative", context_size=8192)
        registry = pair(monkeypatch, configured(tmp_path, **same), configured(tmp_path, **same))

        assert (registry.for_role(roles.CREATIVE)
                is registry.for_role(roles.SPATIAL))
        assert registry.shared()

    def test_F_the_same_mode_with_a_different_model_does_not(self, tmp_path, monkeypatch):
        registry = pair(monkeypatch,
                        configured(tmp_path, mode="cpu", device="none"),
                        configured(tmp_path, mode="cpu", device="none", model_name="B.gguf"))

        assert (registry.for_role(roles.CREATIVE)
                is not registry.for_role(roles.SPATIAL))
        assert not registry.shared()

    def test_D_the_same_card_with_a_different_placement_does_not(self, tmp_path, monkeypatch):
        """The case that needs the mode in the identity: both record zero
        layers on the same device, and only the mode tells them apart."""
        registry = pair(monkeypatch,
                        configured(tmp_path, mode="mixed_conservative"),
                        configured(tmp_path, mode="mixed_aggressive"))

        assert (registry.for_role(roles.CREATIVE)
                is not registry.for_role(roles.SPATIAL))

    def test_C_different_cards_do_not(self, tmp_path, monkeypatch):
        registry = pair(monkeypatch,
                        configured(tmp_path, mode="mixed_aggressive", gpu_index=0),
                        configured(tmp_path, mode="mixed_aggressive", gpu_index=1,
                                   device="CUDA1"))

        assert (registry.for_role(roles.CREATIVE)
                is not registry.for_role(roles.SPATIAL))

    def test_a_different_context_does_not(self, tmp_path, monkeypatch):
        """Section 10.1's fourth example. A context size is a start-time
        property, so it is a property two roles cannot share a process across."""
        registry = pair(monkeypatch,
                        configured(tmp_path, mode="cpu", device="none"),
                        configured(tmp_path, mode="cpu", device="none", context_size=4096))

        assert (registry.for_role(roles.CREATIVE)
                is not registry.for_role(roles.SPATIAL))

    def test_two_servers_get_two_lines_in_the_register(self, tmp_path, monkeypatch):
        """Section 17: "residency keys must be runtime-specific rather than one
        global LLM key." One key between two servers has each declaration
        overwrite the other's, so the broker sees one server holding whatever
        the last one to start happened to hold."""
        registry = pair(monkeypatch,
                        configured(tmp_path, mode="cpu", device="none"),
                        configured(tmp_path, mode="cpu", device="none", model_name="B.gguf"))
        one = registry.for_role(roles.CREATIVE)
        two = registry.for_role(roles.SPATIAL)

        assert one.residency_key != two.residency_key

    def test_a_shared_server_has_one_line_and_names_both_roles(self, tmp_path, monkeypatch):
        same = dict(mode="cpu", device="none")
        registry = pair(monkeypatch, configured(tmp_path, **same), configured(tmp_path, **same))
        registry.for_role(roles.CREATIVE)
        found = registry.for_role(roles.SPATIAL)

        assert found.roles == (roles.CREATIVE, roles.SPATIAL)
        assert "Creative Writer and Spatial Composer" in found._label(
            configured(tmp_path, **same))


# --------------------------------------------------------------------------- #
# I. The upgrade (sections 15 and 22 I)
# --------------------------------------------------------------------------- #


class TestAnInstallationThatHasNeverHeardOfRoles:
    def test_a_role_with_no_overrides_follows_the_installation(self):
        assert roles.overrides(roles.CREATIVE, {}, {}) == {}
        assert not roles.split(roles.CREATIVE, {}, {})

    def test_both_roles_resolve_to_the_shared_configuration(self, tmp_path, monkeypatch):
        """Section 22 I: behaviour remains old-style until the user changes one
        role. Reached by inheritance rather than by a copy, so there is no
        migration step that can half-run."""
        shared = configured(tmp_path, mode="mixed_aggressive")
        registry = pair(monkeypatch, shared, shared, shared)

        assert registry.for_role(roles.CREATIVE) is registry.for_role(roles.SPATIAL)
        assert registry.shared()

    def test_the_shared_singleton_is_the_one_they_get(self, tmp_path, monkeypatch):
        """Otherwise an installation with nothing split runs three servers: the
        one every other mode uses, plus one per role pointing at the same
        model with the same settings."""
        shared = configured(tmp_path, mode="cpu", device="none")
        monkeypatch.setattr(runtime, "config", lambda role="": shared)
        registry = runtime.RuntimeRegistry()

        assert registry.for_role(roles.CREATIVE) is runtime.runtime
        assert registry.for_role(roles.SPATIAL) is runtime.runtime
        assert registry.for_role() is runtime.runtime

    def test_splitting_one_role_leaves_the_other_where_it_was(self, tmp_path, monkeypatch):
        shared = configured(tmp_path, mode="cpu", device="none")
        split = configured(tmp_path, mode="cpu", device="none", model_name="B.gguf")
        registry = pair(monkeypatch, shared, split, shared)

        assert registry.for_role(roles.CREATIVE) is runtime.runtime
        assert registry.for_role(roles.SPATIAL) is not runtime.runtime


class TestRoleOverrides:
    def test_an_override_layers_over_the_installation(self):
        state = {"model": "A.gguf", "mode": "gpu",
                 "roles": {"spatial": {"mode": "cpu"}}}

        assert roles.layered(roles.SPATIAL, state, state, {}) == {
            **state, "mode": "cpu"}
        assert roles.layered(roles.CREATIVE, state, state, {})["mode"] == "gpu"

    def test_state_and_preference_fields_stay_in_their_own_files(self):
        """A context size written into the state file's namespace is a key the
        next writer of that file faithfully persists somewhere it is not read."""
        state = {"roles": {"creative": {"mode": "cpu"}}}
        prefs = {"roles": {"creative": {"context_size": 4096}}}

        layered_state = roles.layered(roles.CREATIVE, {}, state, prefs,
                                      keys=roles.STATE_FIELDS)
        layered_prefs = roles.layered(roles.CREATIVE, {}, state, prefs,
                                      keys=roles.PREFS_FIELDS)

        assert layered_state == {"mode": "cpu"}
        assert layered_prefs == {"context_size": 4096}

    def test_applying_and_clearing_round_trips(self):
        document: dict = {}
        roles.apply(document, roles.CREATIVE, {"mode": "cpu"}, keys=roles.STATE_FIELDS)

        assert document == {"roles": {"creative": {"mode": "cpu"}}}

        roles.clear(document, roles.CREATIVE)

        assert document == {}

    def test_clearing_a_role_leaves_the_other_alone(self):
        document = {"roles": {"creative": {"mode": "cpu"}, "spatial": {"mode": "gpu"}}}
        roles.clear(document, roles.CREATIVE)

        assert document == {"roles": {"spatial": {"mode": "gpu"}}}

    def test_an_unknown_role_name_is_the_installation(self):
        """Total on purpose: a role arriving from a saved panel value or an
        older build falls back to a configuration that is always valid."""
        assert roles.named("nonsense") == roles.SHARED
        assert roles.overrides("nonsense", {"roles": {"nonsense": {"mode": "cpu"}}}) == {}


# --------------------------------------------------------------------------- #
# The same-pool choice (the clarification the specification did not carry)
# --------------------------------------------------------------------------- #


class TestWhenBothRolesWantTheSameMemory:
    def test_conservative_and_cpu_are_both_system_ram(self, tmp_path):
        """Conservative belongs with the processor and not with the card it
        names: its promise is that no layer is resident, so what it competes
        for is system RAM."""
        assert runtime.pool(configured(tmp_path, mode="mixed_conservative")) == \
            runtime.POOL_SYSTEM_RAM
        assert runtime.pool(configured(tmp_path, mode="cpu", device="none")) == \
            runtime.POOL_SYSTEM_RAM

    def test_a_card_is_its_own_pool_per_index(self, tmp_path):
        assert runtime.pool(configured(tmp_path, mode="gpu", gpu_index=0)) == "cuda:0"
        assert runtime.pool(configured(tmp_path, mode="mixed_aggressive",
                                       gpu_index=1, device="CUDA1")) == "cuda:1"

    def test_automatic_coexists_in_ram_and_takes_turns_on_a_card(self):
        """Two servers in system RAM on a machine with the RAM for them cost
        nothing by coexisting, and taking turns there would buy a model reload
        per role switch for no reason. Two on one card is the case the user
        described as fighting over what is left."""
        assert runtime.resolved_sharing(runtime.POOL_SYSTEM_RAM) == runtime.SHARE_COEXIST
        assert runtime.resolved_sharing("cuda:0") == runtime.SHARE_TAKE_TURNS

    def test_either_can_be_forced(self):
        assert runtime.resolved_sharing("cuda:0", runtime.SHARE_COEXIST) == \
            runtime.SHARE_COEXIST
        assert runtime.resolved_sharing(runtime.POOL_SYSTEM_RAM,
                                        runtime.SHARE_TAKE_TURNS) == runtime.SHARE_TAKE_TURNS

    def test_roles_in_different_pools_are_never_contending(self, tmp_path, monkeypatch):
        """The arrangement the scenarios companion recommends -- Creative on a
        card, Spatial on the processor -- needs no policy at all."""
        registry = pair(monkeypatch,
                        configured(tmp_path, mode="mixed_aggressive"),
                        configured(tmp_path, mode="cpu", device="none", model_name="B.gguf"))

        assert registry.contending() == ""

    def test_a_shared_runtime_is_never_contending(self, tmp_path, monkeypatch):
        same = dict(mode="cpu", device="none")
        registry = pair(monkeypatch, configured(tmp_path, **same), configured(tmp_path, **same))

        assert registry.contending() == ""

    def test_two_servers_in_one_pool_are(self, tmp_path, monkeypatch):
        registry = pair(monkeypatch,
                        configured(tmp_path, mode="cpu", device="none"),
                        configured(tmp_path, mode="cpu", device="none", model_name="B.gguf"))

        assert registry.contending() == runtime.POOL_SYSTEM_RAM

    def test_taking_turns_stands_the_other_role_down(self, tmp_path, monkeypatch):
        creative = configured(tmp_path, mode="gpu", gpu_index=0)
        spatial = configured(tmp_path, mode="gpu", gpu_index=0, model_name="B.gguf")
        registry = pair(monkeypatch, creative, spatial)
        monkeypatch.setattr(runtime, "_sharing_mode", lambda: runtime.SHARE_TAKE_TURNS)

        theirs = registry.for_role(roles.CREATIVE)
        released: list = []
        monkeypatch.setattr(theirs, "running", lambda: True)
        monkeypatch.setattr(theirs, "release",
                            lambda needed, reason="": released.append(reason) or _GB)

        freed = registry.make_room_for(roles.SPATIAL, spatial)

        assert freed == _GB
        assert released and "Spatial Composer" in released[0]

    def test_coexisting_stands_nobody_down(self, tmp_path, monkeypatch):
        creative = configured(tmp_path, mode="gpu", gpu_index=0)
        spatial = configured(tmp_path, mode="gpu", gpu_index=0, model_name="B.gguf")
        registry = pair(monkeypatch, creative, spatial)
        monkeypatch.setattr(runtime, "_sharing_mode", lambda: runtime.SHARE_COEXIST)

        theirs = registry.for_role(roles.CREATIVE)
        monkeypatch.setattr(theirs, "running", lambda: True)
        monkeypatch.setattr(theirs, "release",
                            lambda needed, reason="": pytest.fail("should not release"))

        assert registry.make_room_for(roles.SPATIAL, spatial) == 0

    def test_a_shared_runtime_is_never_stood_down_for_itself(self, tmp_path, monkeypatch):
        """The role about to run and the role holding the server are the same
        server. Stopping it to make room for itself would be a reload per pass."""
        same = dict(mode="gpu", gpu_index=0)
        creative = configured(tmp_path, **same)
        registry = pair(monkeypatch, creative, configured(tmp_path, **same))
        monkeypatch.setattr(runtime, "_sharing_mode", lambda: runtime.SHARE_TAKE_TURNS)

        found = registry.for_role(roles.CREATIVE)
        registry.for_role(roles.SPATIAL)
        monkeypatch.setattr(found, "running", lambda: True)
        monkeypatch.setattr(found, "release",
                            lambda needed, reason="": pytest.fail("should not release"))

        assert registry.make_room_for(roles.SPATIAL, creative) == 0


# --------------------------------------------------------------------------- #
# The broker sees every server (section 17)
# --------------------------------------------------------------------------- #


class TestTheRegistryAnswersForEveryRuntime:
    def test_resident_bytes_is_the_sum(self, tmp_path, monkeypatch):
        registry = pair(monkeypatch,
                        configured(tmp_path, mode="gpu"),
                        configured(tmp_path, mode="gpu", model_name="B.gguf"))
        one = registry.for_role(roles.CREATIVE)
        two = registry.for_role(roles.SPATIAL)
        monkeypatch.setattr(one, "resident_bytes", lambda: 2 * _GB)
        monkeypatch.setattr(two, "resident_bytes", lambda: 3 * _GB)

        assert registry.resident_bytes() == 5 * _GB

    def test_releasing_stops_the_largest_first_and_no_more_than_needed(
            self, tmp_path, monkeypatch):
        registry = pair(monkeypatch,
                        configured(tmp_path, mode="gpu"),
                        configured(tmp_path, mode="gpu", model_name="B.gguf"))
        big = registry.for_role(roles.CREATIVE)
        small = registry.for_role(roles.SPATIAL)
        for found, held in ((big, 8 * _GB), (small, 1 * _GB)):
            monkeypatch.setattr(found, "running", lambda: True)
            monkeypatch.setattr(found, "resident_bytes", lambda held=held: held)
        stopped: list = []
        monkeypatch.setattr(big, "release",
                            lambda needed, reason="": stopped.append("big") or 8 * _GB)
        monkeypatch.setattr(small, "release",
                            lambda needed, reason="": stopped.append("small") or _GB)

        freed = registry.release(4 * _GB, "an image generation")

        assert freed == 8 * _GB
        assert stopped == ["big"]

    def test_a_request_for_everything_reaches_them_all(self, tmp_path, monkeypatch):
        registry = pair(monkeypatch,
                        configured(tmp_path, mode="gpu"),
                        configured(tmp_path, mode="gpu", model_name="B.gguf"))
        stopped: list = []
        for role in roles.ROLES:
            found = registry.for_role(role)
            monkeypatch.setattr(found, "running", lambda: True)
            monkeypatch.setattr(found, "resident_bytes", lambda: _GB)
            monkeypatch.setattr(found, "release",
                                lambda needed, reason="", name=role:
                                    stopped.append(name) or _GB)

        registry.release(0, "shutting down")

        assert sorted(stopped) == sorted(roles.ROLES)

    def test_the_shared_singleton_is_counted_before_any_role_asks(self, monkeypatch):
        """The broker registers the registry at import and may ask what is
        resident long before Creative Mode runs. Answering "nothing" then tells
        the image side there is VRAM to take that a server is holding."""
        registry = runtime.RuntimeRegistry()
        monkeypatch.setattr(runtime.runtime, "resident_bytes", lambda: 7 * _GB)

        assert runtime.runtime in registry.all()
        assert registry.resident_bytes() == 7 * _GB


# --------------------------------------------------------------------------- #
# Two llama.cpp families on one machine (section 14)
# --------------------------------------------------------------------------- #


class TestTwoRuntimeFamiliesCanCoexist:
    def test_the_first_family_installs_where_it_always_did(self, tmp_path):
        assert setup.runtime_directory("llama-runtime-cuda12", tmp_path) == \
            tmp_path / setup.RUNTIME_DIRNAME

    def test_a_second_family_gets_its_own_directory(self, tmp_path):
        """Section 14: Creative on a 5090 wants the CUDA 13 build and Spatial on
        a 3090 wants CUDA 12. ``install_runtime`` swaps its destination
        directory wholesale, so without this the second install takes the first
        role's llama.cpp away with it."""
        first = tmp_path / setup.RUNTIME_DIRNAME
        first.mkdir()
        (first / setup.RUNTIME_MARKER).write_text("llama-runtime-cuda12", encoding="utf-8")

        assert setup.runtime_directory("llama-runtime-cuda13", tmp_path) == \
            tmp_path / "runtime-llama-runtime-cuda13"

    def test_reinstalling_the_same_family_reuses_its_directory(self, tmp_path):
        first = tmp_path / setup.RUNTIME_DIRNAME
        first.mkdir()
        (first / setup.RUNTIME_MARKER).write_text("llama-runtime-cuda12", encoding="utf-8")

        assert setup.runtime_directory("llama-runtime-cuda12", tmp_path) == first

    def test_an_unmarked_directory_is_upgraded_in_place(self, tmp_path):
        """An installation made before the marker existed. Treating it as a
        foreign family would move the runtime somebody's state file names."""
        (tmp_path / setup.RUNTIME_DIRNAME).mkdir()

        assert setup.runtime_directory("llama-runtime-cuda12", tmp_path) == \
            tmp_path / setup.RUNTIME_DIRNAME

    def test_installed_families_are_listed_by_id(self, tmp_path):
        for name, family in (("runtime", "llama-runtime-cuda12"),
                             ("runtime-llama-runtime-cuda13", "llama-runtime-cuda13")):
            directory = tmp_path / name
            directory.mkdir()
            (directory / setup.RUNTIME_MARKER).write_text(family, encoding="utf-8")
            (directory / "llama-server").write_bytes(b"")

        found = setup.runtime_families(tmp_path)

        assert set(found) == {"llama-runtime-cuda12", "llama-runtime-cuda13"}

    def test_a_half_written_directory_is_not_offered(self, tmp_path):
        for name in (".runtime.incoming", "runtime.previous"):
            directory = tmp_path / name
            directory.mkdir()
            (directory / "llama-server").write_bytes(b"")

        assert setup.runtime_families(tmp_path) == {}


# --------------------------------------------------------------------------- #
# The pipeline actually reaches the two servers (sections 11.1 and 22 G)
# --------------------------------------------------------------------------- #


class TestTheWriterAndTheComposerAskForTheirOwnRuntime:
    """The integration point the whole feature exists for.

    Everything above can be right and this can still be wrong: two runtimes
    resolved correctly and both passes sent to whichever one the module
    singleton happens to be is exactly the bug that would look like the feature
    working until somebody timed it.
    """

    @pytest.fixture
    def asked(self, monkeypatch, tmp_path):
        import mc_llm_sessions as sessions

        creative = configured(tmp_path, mode="mixed_conservative")
        spatial = configured(tmp_path, mode="cpu", device="none", model_name="B.gguf")
        registry = pair(monkeypatch, creative, spatial)
        monkeypatch.setattr(runtime, "registry", registry)

        seen: list = []

        class Fake:
            def __init__(self, role):
                self.role = role

            def client(self, needs_vision=False, reserve=0, cancel=None):
                seen.append(self.role)
                return object()

        wanted = {registry.for_role(roles.CREATIVE): roles.CREATIVE,
                  registry.for_role(roles.SPATIAL): roles.SPATIAL}
        for found, role in wanted.items():
            monkeypatch.setattr(found, "client", Fake(role).client)
        return sessions, seen

    def test_the_writer_asks_the_creative_runtime(self, asked):
        sessions, seen = asked
        sessions._client(False, 0, roles.CREATIVE)

        assert seen == [roles.CREATIVE]

    def test_the_composer_asks_the_spatial_runtime(self, asked):
        sessions, seen = asked
        sessions._client(False, 0, roles.SPATIAL)

        assert seen == [roles.SPATIAL]

    def test_a_mode_that_names_no_role_asks_the_installation(self, asked, monkeypatch):
        """Prompt Studio, Conversation and MiniMax, unchanged."""
        sessions, seen = asked
        called: list = []
        monkeypatch.setattr(runtime.runtime, "client",
                            lambda needs_vision=False, reserve=0, cancel=None: called.append(1))

        sessions._client(False, 0)

        assert called == [1] and seen == []

    def test_the_status_line_is_about_the_role_s_own_server(self, asked, monkeypatch):
        """With two up, asking the module singleton is how a Composer running on
        the processor comes to report the writer's card."""
        sessions, _seen = asked
        registry = runtime.registry
        monkeypatch.setattr(registry.for_role(roles.CREATIVE), "running", lambda: True)
        monkeypatch.setattr(registry.for_role(roles.SPATIAL), "running", lambda: False)

        assert sessions._preparing(roles.CREATIVE) == "Preparing…"
        assert sessions._preparing(roles.SPATIAL) == "Starting llama-server…"

    def test_the_console_says_which_role_a_run_was(self):
        assert roles.prefix(roles.CREATIVE) == "[Creative] "
        assert roles.prefix(roles.SPATIAL) == "[Spatial] "
        assert roles.prefix(roles.SHARED) == ""


# --------------------------------------------------------------------------- #
# What a split actually writes (sections 8 and 14)
# --------------------------------------------------------------------------- #


class TestSplittingARoleWritesOnlyThatRole:
    def test_a_role_s_model_does_not_move_the_installation_s(self, tmp_path, monkeypatch):
        """Section 8: "Selecting B must not overwrite A." The design intent's own
        example is a large backbone for the writer and a small instruction
        follower for the Composer, and it needs this to be true."""
        import mc_llm_paths
        import mc_llm_setup
        from prompt_master.core.config import atomic_write_json, read_json

        root = tmp_path / "llm"
        (root / "data").mkdir(parents=True)
        monkeypatch.setattr(mc_llm_paths, "data_root", lambda: root)
        paths = mc_llm_paths.app_paths()
        big = root / "big.gguf"
        big.write_bytes(b"")
        small = root / "small.gguf"
        small.write_bytes(b"")
        atomic_write_json(paths.state_file, {"model": paths.record(big), "mode": "gpu"})

        mc_llm_setup.record_model(small, None, role=roles.SPATIAL)

        state = read_json(paths.state_file)
        assert state["model"] == paths.record(big)
        assert state["roles"]["spatial"]["model"] == paths.record(small)
        assert "creative" not in state.get("roles", {})

    def test_a_role_s_manual_model_loses_the_managed_profile(self, tmp_path, monkeypatch):
        """The managed selection lives in the same file. A role that overrode
        only the path would still resolve the installation's profile, so a
        hand-picked 4B GGUF would run with a 26B backbone's context."""
        import mc_llm_paths
        import mc_llm_setup
        from prompt_master.core.config import atomic_write_json, read_json

        root = tmp_path / "llm"
        (root / "data").mkdir(parents=True)
        monkeypatch.setattr(mc_llm_paths, "data_root", lambda: root)
        paths = mc_llm_paths.app_paths()
        small = root / "small.gguf"
        small.write_bytes(b"")
        atomic_write_json(paths.state_file,
                          {"source": "managed", "managed_model_id": "gemma4-26b"})

        mc_llm_setup.record_model(small, None, role=roles.SPATIAL)

        entry = read_json(paths.state_file)["roles"]["spatial"]
        assert entry["source"] == "manual"
        assert "managed_model_id" not in entry

    def test_recording_a_device_for_a_role_leaves_the_installation_alone(
            self, tmp_path, monkeypatch):
        state = {"mode": "gpu", "gpu_index": 0, "gpu_device": "CUDA0"}
        roles.apply(state, roles.SPATIAL, {"mode": "cpu", "gpu_device": "none"},
                    keys=roles.STATE_FIELDS)

        assert state["mode"] == "gpu" and state["gpu_device"] == "CUDA0"
        assert state["roles"]["spatial"] == {"mode": "cpu", "gpu_device": "none"}


class TestEachServerHasItsOwnLineInTheRegister:
    def test_a_runtime_keeps_the_key_it_was_built_with(self):
        """Section 17. Two servers sharing one key have each declaration
        overwrite the other's, so the broker believes there is one server
        holding whatever the last one to start happened to hold."""
        one = runtime.Runtime(residency_key="llm:llama.cpp:aaaa")
        two = runtime.Runtime(residency_key="llm:llama.cpp:bbbb")

        assert one.residency_key == "llm:llama.cpp:aaaa"
        assert two.residency_key == "llm:llama.cpp:bbbb"
        assert one.residency_key != two.residency_key

    def test_a_runtime_built_without_one_falls_back_to_the_module_key(self):
        assert runtime.Runtime().residency_key == runtime.RESIDENCY_KEY

    def test_two_identities_produce_two_keys(self, tmp_path):
        one = runtime._residency_key(
            runtime._identity(configured(tmp_path, mode="gpu"), None))
        two = runtime._residency_key(
            runtime._identity(configured(tmp_path, mode="cpu", device="none"), None))

        assert one != two
        assert one.startswith(runtime.RESIDENCY_KEY)

    def test_the_same_identity_produces_the_same_key(self, tmp_path):
        """Two roles that coalesce share a server, so they must share its line."""
        same = configured(tmp_path, mode="gpu")

        assert (runtime._residency_key(runtime._identity(same, None))
                == runtime._residency_key(runtime._identity(same, None)))


# --------------------------------------------------------------------------- #
# A second card is a second pool (reported from a two-card machine)
# --------------------------------------------------------------------------- #


class TestARoleOnAnotherCardIsNotTheImageCardsProblem:
    """From a 5090 + 3090 log: the plan said "22.7 GB obtainable of 24.0 GB on
    the card" while llama.cpp reported CUDA0 with 30.2 GB free and CUDA1 with
    22.8 GB. Every VRAM figure came from whichever card Forge was generating on,
    so a role pinned to the idle one was sized against, and capped by, a card it
    was never going to touch.
    """

    def test_the_free_reading_is_asked_of_the_role_s_own_card(self, monkeypatch):
        asked: list = []
        monkeypatch.setattr(mc_broker, "device_free_vram_bytes",
                            lambda index=None: asked.append(index) or (8 * _GB))

        runtime._free_vram(0, 1)

        assert asked == [1]

    def test_a_role_on_the_image_card_is_still_capped_by_the_plan(self, monkeypatch):
        monkeypatch.setattr(mc_broker, "image_device_index", lambda: 0)

        assert runtime.shares_the_image_card(0)

    def test_a_role_on_another_card_is_not(self, monkeypatch):
        monkeypatch.setattr(mc_broker, "image_device_index", lambda: 0)

        assert not runtime.shares_the_image_card(1)

    def test_an_unknown_card_is_treated_as_the_image_card(self, monkeypatch):
        """Conservative on purpose: being wrong this way costs a smaller
        language model, and being wrong the other way costs a generation that
        runs out of memory."""
        monkeypatch.setattr(mc_broker, "image_device_index", lambda: -1)

        assert runtime.shares_the_image_card(1)
        assert runtime.shares_the_image_card(None)

    def test_the_plan_stops_capping_a_placement_on_another_card(self, monkeypatch):
        import mc_plan

        monkeypatch.setattr(mc_broker, "device_free_vram_bytes",
                            lambda index=None: 30 * _GB)
        monkeypatch.setattr(mc_broker, "image_device_index", lambda: 0)
        monkeypatch.setattr(mc_plan, "current", lambda: object())
        monkeypatch.setattr(mc_plan, "persistent_llm_budget", lambda ours=0: 3 * _GB)
        monkeypatch.setattr(mc_plan, "learned_cap_bytes", lambda: 0)

        assert runtime._spendable(0, 0) == 3 * _GB
        assert runtime._spendable(0, 1) == 30 * _GB

    def test_a_processor_placement_has_no_card_to_ask_about(self, tmp_path):
        assert runtime.card_of(configured(tmp_path, mode="cpu", device="none")) is None
        assert runtime.card_of(configured(tmp_path, mode="gpu", gpu_index=1)) == 1
        assert runtime.card_of(
            configured(tmp_path, mode="mixed_conservative", gpu_index=1)) == 1


# --------------------------------------------------------------------------- #
# The menu has to be able to express the choice (reported from the UI)
# --------------------------------------------------------------------------- #


class TestEveryDeviceOptionReadsDifferently:
    def test_the_three_entries_for_one_card_are_three_sentences(self, two_cards):
        labels = [setup.describe_device(found) for found in setup.devices()]

        assert len(labels) == len(set(labels))

    def test_the_two_mixed_modes_are_named(self, two_cards):
        labels = " | ".join(setup.describe_device(found) for found in setup.devices())

        assert "Mixed Aggressive" in labels
        assert "Mixed Conservative" in labels

    def test_conservative_says_it_puts_nothing_on_the_card(self, two_cards):
        conservative = [found for found in setup.devices() if found.is_conservative]

        assert conservative
        for found in conservative:
            assert "no model layers in VRAM" in setup.describe_device(found)


# --------------------------------------------------------------------------- #
# A role's server is not a stray (reported: "stopped 3 stray processes")
# --------------------------------------------------------------------------- #


class TestARunningRoleServerIsNeverAStray:
    def test_every_runtime_s_pid_counts_as_ours(self, tmp_path, monkeypatch):
        """A stray is a server nothing here has a handle to. Asking only the
        shared runtime made two live role servers answer to nobody, so Unload
        reported them as strays and killed them."""
        import types

        registry = pair(monkeypatch,
                        configured(tmp_path, mode="cpu", device="none"),
                        configured(tmp_path, mode="cpu", device="none", model_name="B.gguf"))
        monkeypatch.setattr(runtime, "registry", registry)
        for role, pid in ((roles.CREATIVE, 4242), (roles.SPATIAL, 4343)):
            found = registry.for_role(role)
            # Through monkeypatch, not by assignment: one of these may be the
            # module singleton, and a process left on it outlives the test.
            monkeypatch.setattr(found, "_process", types.SimpleNamespace(
                process=types.SimpleNamespace(pid=pid)))

        assert {4242, 4343} <= runtime._own_pids()

    def test_a_runtime_with_no_server_contributes_nothing(self, monkeypatch):
        registry = runtime.RuntimeRegistry()
        monkeypatch.setattr(runtime, "registry", registry)
        monkeypatch.setattr(runtime.runtime, "_process", None)

        assert runtime._own_pids() == set()


# --------------------------------------------------------------------------- #
# A role's server is started from that role's settings
# --------------------------------------------------------------------------- #


class TestTheServerIsStartedFromTheRoleSConfiguration:
    """The gap the first version of this feature left, and what it cost.

    ``registry.for_role`` resolved *which* runtime serves a role, correctly, and
    then ``Runtime.client`` asked ``config()`` -- with no role -- for what to
    start. So a Creative role pinned to a 5090 got a server of its own and that
    server was launched from the installation's settings.

    From the log that found it: two llama-servers, each with its own prompt
    cache, both reporting "system RAM (no GPU offload)" on the processor, and
    llama.cpp's own fit line saying "projected to use 14157 MiB of host memory"
    while it listed a 5090 with 30.9 GB free and a 3090 with 23.2 GB free. Every
    role-specific thing worked except the one that makes the feature worth
    having.
    """

    def test_a_split_role_starts_from_its_own_settings(self, tmp_path, monkeypatch):
        creative = configured(tmp_path, mode="mixed_aggressive", gpu_index=0,
                              device="CUDA0")
        shared = configured(tmp_path, mode="cpu", device="none")
        registry = pair(monkeypatch, creative, shared, shared)

        found = registry.for_role(roles.CREATIVE)

        assert found.configuration().mode == "mixed_aggressive"
        assert found.configuration().device == "CUDA0"

    def test_the_other_role_starts_from_its_own(self, tmp_path, monkeypatch):
        creative = configured(tmp_path, mode="mixed_aggressive", gpu_index=0,
                              device="CUDA0")
        spatial = configured(tmp_path, mode="cpu", device="none", model_name="B.gguf")
        registry = pair(monkeypatch, creative, spatial, creative)

        assert registry.for_role(roles.CREATIVE).configuration().device == "CUDA0"
        assert registry.for_role(roles.SPATIAL).configuration().device == "none"

    def test_a_runtime_nobody_claimed_starts_from_the_installation(self, tmp_path,
                                                                   monkeypatch):
        shared = configured(tmp_path, mode="cpu", device="none")
        monkeypatch.setattr(runtime, "config", lambda role="": shared)

        assert runtime.Runtime().configuration() is shared

    def test_a_role_that_moved_away_no_longer_speaks_for_the_shared_runtime(
            self, tmp_path, monkeypatch):
        """The check that makes this safe rather than merely correct. The shared
        runtime is every non-role mode's server too, so a Creative role that was
        mapped onto it and has since been given a card of its own must not go on
        deciding what Prompt Studio starts."""
        shared = configured(tmp_path, mode="cpu", device="none")
        registry = pair(monkeypatch, shared, shared, shared)
        found = registry.for_role(roles.CREATIVE)
        assert found is runtime.runtime
        assert found.configuration().device == "none"

        # Creative is reconfigured onto a card; its identity no longer matches
        # the runtime it was adopted by.
        moved = configured(tmp_path, mode="mixed_aggressive", gpu_index=0, device="CUDA0")
        monkeypatch.setattr(runtime, "config",
                            lambda role="": moved if role == roles.CREATIVE else shared)

        assert found.configuration().device == "none"

    def test_client_launches_the_server_from_the_role_s_settings(self, tmp_path,
                                                                 monkeypatch):
        """Through ``client``, not through ``configuration``.

        The version that shipped got the second one right and the first one
        wrong, so every test about roles passed while the servers all came up on
        the installation's placement. What has to be asserted is what reaches
        the launch.
        """
        creative = configured(tmp_path, mode="mixed_aggressive", gpu_index=0,
                              device="CUDA0")
        shared = configured(tmp_path, mode="cpu", device="none")
        registry = pair(monkeypatch, creative, shared, shared)
        monkeypatch.setattr(mc_broker, "device_free_vram_bytes",
                            lambda index=None: 30 * _GB)
        monkeypatch.setattr(runtime, "_prime_prompt_cache", lambda client: None)
        found = registry.for_role(roles.CREATIVE)

        launched: list = []

        class Reached(RuntimeError):
            """Far enough: the configuration is decided by now."""

        def capture(configuration, placement, projector, plan=None):
            launched.append(configuration)
            raise Reached()

        monkeypatch.setattr(found, "_launch", capture)
        with pytest.raises(Reached):
            found.client()

        assert launched, "the launch was never reached"
        assert launched[0].mode == "mixed_aggressive"
        assert launched[0].device == "CUDA0"

    def test_the_shared_runtime_still_launches_from_the_installation(self, tmp_path,
                                                                     monkeypatch):
        shared = configured(tmp_path, mode="cpu", device="none")
        monkeypatch.setattr(runtime, "config", lambda role="": shared)
        monkeypatch.setattr(mc_broker, "device_free_vram_bytes",
                            lambda index=None: 30 * _GB)
        monkeypatch.setattr(runtime, "_prime_prompt_cache", lambda client: None)
        found = runtime.Runtime()

        launched: list = []

        class Reached(RuntimeError):
            pass

        def capture(configuration, placement, projector, plan=None):
            launched.append(configuration)
            raise Reached()

        monkeypatch.setattr(found, "_launch", capture)
        with pytest.raises(Reached):
            found.client()

        assert launched[0].device == "none"

    def test_coalesced_roles_agree_about_what_to_start(self, tmp_path, monkeypatch):
        """Either role answers, because roles only share a runtime when their
        complete resolved identity is equal."""
        same = configured(tmp_path, mode="mixed_conservative", gpu_index=1, device="CUDA1")
        registry = pair(monkeypatch, same, same, same)
        registry.for_role(roles.CREATIVE)
        found = registry.for_role(roles.SPATIAL)

        assert found.configuration().mode == "mixed_conservative"
        assert found.configuration().gpu_index == 1


class TestThePlacementLinesSayWhichConfigurationTheyCameFrom:
    """With two servers up, "on Intel(R) Core(TM) Ultra 9" names a processor and
    not a role -- and which role it was is the fact somebody needs when one of
    the two is on the wrong card."""

    def test_a_role_s_runtime_prefixes_its_own_lines(self, tmp_path, monkeypatch):
        creative = configured(tmp_path, mode="mixed_aggressive", gpu_index=0,
                              device="CUDA0")
        shared = configured(tmp_path, mode="cpu", device="none")
        registry = pair(monkeypatch, creative, shared, shared)

        assert registry.for_role(roles.CREATIVE)._said_for() == "[Creative] "

    def test_the_shared_runtime_says_nothing_extra(self):
        """Every line this extension has always written keeps its shape."""
        assert runtime.Runtime()._said_for() == ""


# --------------------------------------------------------------------------- #
# A server each, even when the two roles are identical (section 10.3)
# --------------------------------------------------------------------------- #


class TestGivingIdenticalRolesAServerEach:
    """The memory decision the design intent left optional.

    Sharing is right when memory is tight and wrong when it is not: two servers
    on a 32 GB card both stay warm and neither pass re-reads the other's system
    prompt, which is exactly the handoff cost section 10.2 says sharing buys.
    """

    def test_identical_roles_share_by_default(self, tmp_path, monkeypatch):
        same = dict(mode="gpu", gpu_index=0)
        registry = pair(monkeypatch, configured(tmp_path, **same), configured(tmp_path, **same))

        assert registry.for_role(roles.CREATIVE) is registry.for_role(roles.SPATIAL)
        assert registry.shared()

    def test_asking_for_one_each_gives_two_servers(self, tmp_path, monkeypatch):
        same = dict(mode="gpu", gpu_index=0)
        registry = pair(monkeypatch, configured(tmp_path, **same), configured(tmp_path, **same))
        monkeypatch.setattr(runtime, "_process_mode", lambda: runtime.PROCESSES_SEPARATE)

        assert registry.for_role(roles.CREATIVE) is not registry.for_role(roles.SPATIAL)
        assert not registry.shared()

    def test_each_gets_its_own_line_in_the_register(self, tmp_path, monkeypatch):
        same = dict(mode="gpu", gpu_index=0)
        registry = pair(monkeypatch, configured(tmp_path, **same), configured(tmp_path, **same))
        monkeypatch.setattr(runtime, "_process_mode", lambda: runtime.PROCESSES_SEPARATE)

        assert (registry.for_role(roles.CREATIVE).residency_key
                != registry.for_role(roles.SPATIAL).residency_key)

    def test_a_role_still_starts_from_its_own_settings(self, tmp_path, monkeypatch):
        same = dict(mode="mixed_conservative", gpu_index=1, device="CUDA1")
        registry = pair(monkeypatch, configured(tmp_path, **same), configured(tmp_path, **same))
        monkeypatch.setattr(runtime, "_process_mode", lambda: runtime.PROCESSES_SEPARATE)

        found = registry.for_role(roles.SPATIAL)

        assert found.configuration().mode == "mixed_conservative"
        assert found.configuration().gpu_index == 1

    def test_the_shared_configuration_is_untouched_by_the_setting(self, tmp_path, monkeypatch):
        """Prompt Studio, Conversation and MiniMax are not roles and have no
        second server to be given."""
        shared = configured(tmp_path, mode="gpu")
        monkeypatch.setattr(runtime, "config", lambda role="": shared)
        monkeypatch.setattr(runtime, "_process_mode", lambda: runtime.PROCESSES_SEPARATE)
        registry = runtime.RuntimeRegistry()

        assert registry.for_role() is runtime.runtime


# --------------------------------------------------------------------------- #
# GPU / VRAM Only means what it says (sections 12.4 and 21)
# --------------------------------------------------------------------------- #


class _BigHeader:
    """A dense 30-block model too large for an empty card."""

    file_bytes = 16 * _GB
    block_count = 30
    usable = True
    context_length = 262144
    embedding_length = 3584
    expert_count = 0
    expert_used_count = 0
    mixture_of_experts = False
    expert_share = 0.0
    head_counts_kv = (8,) * 30
    key_lengths = (128,) * 30
    value_lengths = (128,) * 30
    attending_blocks = 30
    path = pathlib.Path("big.gguf")


class TestAModeThatCouldNotBeCarriedOut:
    """From a 5090 log: the mode was GPU / VRAM Only and the model went to
    system RAM without a word about it. Section 21 asks for the opposite --
    report the inability to satisfy the selected mode rather than silently
    converting it to something else."""

    def test_gpu_only_says_so_when_nothing_fit(self, tmp_path):
        configuration = configured(tmp_path, mode="gpu")
        said = runtime._unsatisfied(configuration,
                                    ctx.Placement(gpu_layers=ctx.NO_LAYERS), None)

        assert said and "GPU / VRAM Only was chosen and could not be satisfied" in said[0]
        assert "none of it is on the card" in said[0]

    def test_it_says_how_much_did_fit(self, tmp_path):
        class Header:
            block_count = 30

        configuration = configured(tmp_path, mode="gpu")
        said = runtime._unsatisfied(configuration, ctx.Placement(gpu_layers=2), Header())

        assert "only 2 of 30 layers" in said[0]

    def test_a_satisfied_gpu_placement_says_nothing(self, tmp_path):
        configuration = configured(tmp_path, mode="gpu")

        assert runtime._unsatisfied(configuration,
                                    ctx.Placement(gpu_layers=ctx.ALL_LAYERS), None) == []

    def test_the_mixed_modes_are_defined_as_taking_what_fits(self, tmp_path):
        """Aggressive degrading is Aggressive working, so it says nothing here.
        Conservative asks for nothing and gets it."""
        for mode in ("mixed_aggressive", "mixed_conservative", "cpu"):
            configuration = configured(tmp_path, mode=mode,
                                       device="none" if mode == "cpu" else "CUDA0")
            assert runtime._unsatisfied(
                configuration, ctx.Placement(gpu_layers=ctx.NO_LAYERS), None) == [], mode

    def test_it_reaches_the_negotiation_notes(self, tmp_path, monkeypatch):
        """Through ``negotiate``, so the sentence really is on the report the
        panel and the console print rather than only on a helper."""
        monkeypatch.setattr(mc_broker, "safety_margin_bytes", lambda: 0)
        monkeypatch.setattr(runtime, "_free_vram", lambda ours=0, card=None: 0)
        monkeypatch.setattr(runtime, "_spendable", lambda ours=0, card=None, **_: 0)
        configuration = configured(tmp_path, mode="gpu")

        negotiated = runtime.negotiate(configuration, _BigHeader(), reclaim=False)

        assert negotiated.placement.gpu_layers == ctx.NO_LAYERS
        assert any("could not be satisfied" in note for note in negotiated.notes), \
            negotiated.notes

    def test_a_mixed_placement_that_degrades_says_nothing_extra(self, tmp_path,
                                                                monkeypatch):
        monkeypatch.setattr(mc_broker, "safety_margin_bytes", lambda: 0)
        monkeypatch.setattr(runtime, "_free_vram", lambda ours=0, card=None: 0)
        monkeypatch.setattr(runtime, "_spendable", lambda ours=0, card=None, **_: 0)
        configuration = configured(tmp_path, mode="mixed_aggressive")

        negotiated = runtime.negotiate(configuration, _BigHeader(), reclaim=False)

        assert not any("could not be satisfied" in note for note in negotiated.notes)


class TestTheStartReadsItsOwnCard:
    def test_the_free_figure_is_the_card_the_server_is_going_on(self, tmp_path, monkeypatch):
        """It is both the number the start line prints and the baseline the
        residency is measured against, so a reading of another card makes the
        second one a subtraction of two unrelated numbers."""
        asked: list = []
        monkeypatch.setattr(mc_broker, "device_free_vram_bytes",
                            lambda index=None: asked.append(index) or (30 * _GB))
        creative = configured(tmp_path, mode="gpu", gpu_index=0, device="CUDA0")
        registry = pair(monkeypatch, creative, creative)
        monkeypatch.setattr(runtime, "_prime_prompt_cache", lambda client: None)
        found = registry.for_role(roles.CREATIVE)

        class Reached(RuntimeError):
            pass

        def stop_inside_launch(placement, gguf):
            raise Reached()

        # Inside ``_launch`` rather than instead of it. The reading under test
        # is taken there, so replacing the whole method would mean this passed
        # whichever card it asked about -- which it did.
        monkeypatch.setattr(runtime, "_layers_argument", stop_inside_launch)
        asked.clear()
        with pytest.raises(Reached):
            found.client()

        assert asked, "nothing asked about free VRAM at all"
        assert 0 in asked, f"the placement never asked about card 0: {asked}"
        # None is "whichever card the image side is on", which is the question
        # this placement must never ask: it is going on card 0.
        assert None not in asked, f"something asked about the image card: {asked}"


class TestTheIdleCardWarning:
    """It fired on Mixed Conservative, where zero layers is the setting working.

    From the log that found it, on a 3090 in Conservative: "will do no work in
    this placement — nothing was free to offload into". Both halves were wrong.
    Nothing failed to be free; the user asked for zero layers. And the card was
    not idle -- ``--op-offload`` gave it the arithmetic, which was 82 tokens a
    second on prompts against 50 on the processor alone.
    """

    def test_conservative_is_not_warned_about(self, tmp_path, caplog):
        configuration = configured(tmp_path, mode="mixed_conservative")
        with caplog.at_level("INFO"):
            runtime._warn_about_an_idle_card(configuration, "0")

        assert "will do no work" not in caplog.text

    def test_the_processor_is_not_warned_about(self, tmp_path, caplog):
        configuration = configured(tmp_path, mode="cpu", device="none")
        with caplog.at_level("INFO"):
            runtime._warn_about_an_idle_card(configuration, "0")

        assert "will do no work" not in caplog.text

    def test_a_mode_that_asked_for_layers_and_got_none_still_is(self, tmp_path, caplog):
        """Aggressive and GPU / VRAM Only both ask for layers, and a card
        holding none of them really is doing nothing."""
        for mode in ("mixed_aggressive", "gpu"):
            caplog.clear()
            configuration = configured(tmp_path, mode=mode)
            with caplog.at_level("INFO"):
                runtime._warn_about_an_idle_card(configuration, "0")

            assert "will do no work" in caplog.text, mode

    def test_a_card_holding_layers_is_never_warned_about(self, tmp_path, caplog):
        configuration = configured(tmp_path, mode="mixed_aggressive")
        with caplog.at_level("INFO"):
            runtime._warn_about_an_idle_card(configuration, "12")

        assert "will do no work" not in caplog.text


# --------------------------------------------------------------------------- #
# Reading the prompt is not generating slowly
# --------------------------------------------------------------------------- #


class TestTheProgressLinesSeparateReadingFromWriting:
    """Reported as "the warm Spatial server starts off slow before hitting top
    speed", and it never did either.

    llama.cpp reads the whole prompt before emitting anything, so a pass with
    230 new tokens to read spends the first 4.6 seconds producing nothing and
    then runs at its full rate. The old line measured from the *request* and
    called all of it "generating", which turned that into "generating, 3
    characters in 5s" -- a sentence that describes a server crawling, about a
    server that had not started writing yet.
    """

    def _events(self, sessions, gap, chunks):
        """A run whose first token arrives ``gap`` seconds after the request."""
        clock = {"now": 1000.0}

        def emitted():
            for text in chunks:
                clock["now"] += gap if text is chunks[0] else 1.0
                yield sessions.Event(sessions.CHUNK, text)
            yield sessions.Event(sessions.DONE, "".join(chunks))

        return clock, emitted()

    def test_the_wait_is_reported_as_reading(self, monkeypatch, caplog):
        import mc_llm_sessions as sessions

        clock, events = self._events(sessions, 4.6, ["a", "b", "c"])
        monkeypatch.setattr(sessions.time, "monotonic", lambda: clock["now"])

        with caplog.at_level("INFO"):
            list(sessions._traced("a spatial composition", events))

        assert "prompt read in 4.6s, writing now" in caplog.text
        assert "generating, 3 characters in 5s" not in caplog.text

    def test_the_generating_clock_starts_at_the_first_token(self, monkeypatch, caplog):
        """Otherwise the first rate reported is the prompt's length divided by
        the writing speed, which is not a speed of anything."""
        import mc_llm_sessions as sessions

        clock = {"now": 1000.0}

        def emitted():
            clock["now"] += 10.0          # ten seconds of reading
            yield sessions.Event(sessions.CHUNK, "x")
            clock["now"] += 6.0           # then six seconds of writing
            yield sessions.Event(sessions.CHUNK, "y" * 60)
            yield sessions.Event(sessions.DONE, "x" + "y" * 60)

        monkeypatch.setattr(sessions.time, "monotonic", lambda: clock["now"])
        with caplog.at_level("INFO"):
            list(sessions._traced("a spatial composition", emitted()))

        assert "generating, 61 characters in 6s" in caplog.text

    def test_the_finished_line_splits_the_two(self, monkeypatch, caplog):
        import mc_llm_sessions as sessions

        clock = {"now": 1000.0}

        def emitted():
            clock["now"] += 4.6
            yield sessions.Event(sessions.CHUNK, "abc")
            clock["now"] += 7.5
            yield sessions.Event(sessions.DONE, "abc")

        monkeypatch.setattr(sessions.time, "monotonic", lambda: clock["now"])
        monkeypatch.setattr(sessions, "_measured", lambda role="": "")
        with caplog.at_level("INFO"):
            list(sessions._traced("a spatial composition", emitted()))

        assert "12.1s (4.6s reading, 7.5s writing)" in caplog.text

    def test_a_run_that_never_produced_a_token_still_reports(self, monkeypatch, caplog):
        import mc_llm_sessions as sessions

        clock = {"now": 1000.0}

        def emitted():
            clock["now"] += 3.0
            yield sessions.Event(sessions.DONE, "")

        monkeypatch.setattr(sessions.time, "monotonic", lambda: clock["now"])
        monkeypatch.setattr(sessions, "_measured", lambda role="": "")
        with caplog.at_level("INFO"):
            list(sessions._traced("a spatial composition", emitted()))

        assert "LLM run finished" in caplog.text
        assert "reading," not in caplog.text
