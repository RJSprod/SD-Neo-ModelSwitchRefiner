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
    cards = [GpuInfo(0, "uuid-a", "NVIDIA GeForce RTX 3090", 24576, 23000, "570"),
             GpuInfo(1, "uuid-b", "NVIDIA GeForce RTX 5090", 32768, 32000, "570")]
    monkeypatch.setattr(detection, "detect_gpus", lambda timeout=15: list(cards))
    return cards


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

            def client(self, needs_vision=False, reserve=0):
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
                            lambda needs_vision=False, reserve=0: called.append(1))

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
