"""Acceleration and VRAM priority, which are two settings and must stay two.

The design intent for Qwen 3.8 and DFlash2 turns on one claim, and nearly every
test here is a way of breaking it: *a DFlash2 run and an emptied card are
independent*. Fold them together and a machine whose backbone and draft already
fit has to evict a checkpoint to get a decoder it could have had for free -- so
the first thing asserted below is a reclaimer that is never called, and the
second is a different card that is never touched.

The other half is the failure contract. A forced accelerator that cannot run
has to say so: section 18 lists six conditions and forbids a silent fallback in
every one of them, because a Lightning run that quietly became a partial offload
is slower than the Normal run it replaced and says the opposite on screen.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import mc_broker
import mc_gguf
import mc_llm_accel as accel
import mc_llm_context as ctx
import mc_llm_dflash as dflash
import mc_llm_managed_models as managed
import mc_llm_paths
import mc_llm_roles as roles
import mc_llm_runtime as runtime
import mc_llm_state
from prompt_master.models import managed_profiles
from test_gguf import _text, _u32, _u32s, write_gguf

_GB = 1024**3
_MB = 1024**2

MODEL_SHA = "a" * 64
MMPROJ_SHA = "b" * 64
DRAFT_SHA = "c" * 64

FAMILY = "qwen38-dflash2"
COMPONENT = "llama-runtime-cuda13-dflash2-qwen38"

DFLASH_FLAGS = frozenset({
    "--spec-draft-model", "--spec-type", "--spec-draft-n-max", "--spec-draft-n-min",
    "--spec-draft-p-min", "--spec-draft-type-k", "--spec-draft-type-v",
    "--n-gpu-layers-draft", "--flash-attn",
})
"""What a build that really has the DFlash2 branch advertises.

A set rather than a real ``--help``: what is under test here is the planner's
gating, and a subprocess would be testing llama.cpp.
"""


# --------------------------------------------------------------------------- #
# A machine, assembled piece by piece
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def install(tmp_path, monkeypatch, host):
    """An LLM data root of our own, and no state carried between tests."""
    root = tmp_path / "install"
    (root / "data").mkdir(parents=True)
    monkeypatch.setattr(mc_llm_paths, "data_root", lambda: root)
    monkeypatch.setattr(ctx, "_store_path", lambda: tmp_path / "calibration.json")
    monkeypatch.setattr(mc_broker, "safety_margin_bytes", lambda: 0)
    managed._registry_cache = None
    mc_gguf.forget()
    ctx.forget()
    mc_broker.clear()
    yield root
    mc_broker.clear()
    ctx.forget()
    mc_gguf.forget()
    managed._registry_cache = None


def hybrid_model(path: Path, *, blocks=8, attending=2, size_mb=4, recurrent=True) -> Path:
    """A Qwen-3.8-shaped header: some blocks attend, the rest keep state.

    The interleave is the point. llama.cpp writes ``head_count_kv`` as an array
    with a zero for every block that keeps no key/value cache, and a planner
    that read the first entry and multiplied would size this model's cache four
    times too large -- and then charge nothing at all for the recurrent state
    the other blocks do hold.
    """
    per_block = [8 if index % (blocks // attending) == 0 else 0 for index in range(blocks)]
    metadata = [
        _text("general.architecture", "qwen3next"),
        _u32("qwen3next.block_count", blocks),
        _u32("qwen3next.context_length", 262144),
        _u32("qwen3next.embedding_length", 2048),
        _u32("qwen3next.attention.head_count", 16),
        _u32s("qwen3next.attention.head_count_kv", per_block),
    ]
    if recurrent:
        metadata += [
            _u32("qwen3next.ssm.state_size", 128),
            _u32("qwen3next.ssm.conv_kernel", 4),
            _u32("qwen3next.ssm.inner_size", 4096),
            _u32("qwen3next.ssm.group_count", 8),
        ]
    return write_gguf(path, b"".join(metadata), len(metadata), padding=size_mb * _MB)


def registry_document(**dflash) -> dict:
    """One Qwen 3.8-shaped catalogue row, with whatever is being tested replaced."""
    speculator = {
        "filename": "dflash-test-BF16.gguf", "sha256": DRAFT_SHA, "bytes": None,
        "display_size": "3.86 GB", "runtime_family": FAMILY,
        "requires_full_target_gpu": True, "requires_full_draft_gpu": True,
        "same_gpu_as_target": True, "draft_tokens": 3, "draft_min_tokens": 0,
        "draft_p_min": 0.0, "draft_kv_type_k": "f16", "draft_kv_type_v": "f16",
    }
    speculator.update(dflash)
    return {"version": 1, "registry_version": "test-1", "models": [{
        "id": "qwen38-test", "label": "Qwen 3.8 Test", "role": "Recommended 27B",
        "group": "Qwen 3.8 27B", "family": "Qwen 3.8",
        "profile": "qwen38-27b-prompt-author", "multimodal": True,
        "source_url": "https://huggingface.co/example/test",
        "repo_id": "example/test", "revision": "e" * 40,
        "model": {"filename": "qwen38-test-Q4_K_M.gguf", "sha256": MODEL_SHA,
                  "bytes": None, "display_size": "~16.8 GB"},
        "projector": {"filename": "mmproj-qwen38-test-F16.gguf", "sha256": MMPROJ_SHA,
                      "bytes": None, "display_size": "928 MB"},
        "accelerators": {"mtp": {"embedded": True, "draft_tokens": 3},
                         "dflash2": speculator},
    }]}


@pytest.fixture
def registry(tmp_path, monkeypatch):
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(registry_document()), encoding="utf-8")
    monkeypatch.setattr(managed, "REGISTRY_PATH", path)
    managed._registry_cache = None
    return path


def install_bundle(root: Path, tmp_path: Path, *, draft=True, projector=True,
                   size_mb=4, draft_mb=1) -> Path:
    """A verified bundle on disk, optionally with its speculative sidecar."""
    bundle = root / "models" / "managed" / "qwen38-test"
    bundle.mkdir(parents=True, exist_ok=True)
    hybrid_model(bundle / "model.gguf", size_mb=size_mb)
    artifacts = {"model": {"filename": "qwen38-test-Q4_K_M.gguf",
                           "stored_as": "model.gguf", "sha256": MODEL_SHA}}
    if projector:
        (bundle / "mmproj.gguf").write_bytes(b"\x00" * _MB)
        artifacts["projector"] = {"filename": "mmproj-qwen38-test-F16.gguf",
                                  "stored_as": "mmproj.gguf", "sha256": MMPROJ_SHA}
    if draft:
        hybrid_model(bundle / "draft.gguf", blocks=2, attending=2, size_mb=draft_mb,
                     recurrent=False)
        artifacts["draft"] = {"filename": "dflash-test-BF16.gguf",
                              "stored_as": "draft.gguf", "sha256": DRAFT_SHA}
    (bundle / "installed.json").write_text(json.dumps({
        "schema": 1, "model_id": "qwen38-test", "registry_version": "test-1",
        "revision": "e" * 40, "profile": "qwen38-27b-prompt-author",
        "profile_version": managed_profiles.VERSION, "artifacts": artifacts,
        "installed_at": 0,
    }), encoding="utf-8")
    return bundle


def install_dflash_runtime(root: Path, *, text=True, vision=False) -> Path:
    """A DFlash2 family directory with both markers and a capability record."""
    import mc_llm_setup

    directory = root / mc_llm_setup.family_dirname(COMPONENT)
    directory.mkdir(parents=True, exist_ok=True)
    server = directory / "llama-server"
    server.write_bytes(b"")
    (directory / mc_llm_setup.RUNTIME_MARKER).write_text(COMPONENT, encoding="utf-8")
    (directory / dflash.PROVENANCE_MARKER).write_text(json.dumps({
        "component_id": COMPONENT, "family": FAMILY,
        "commit": "1deefcca395743049c3820ab8f9b15043f3e9446",
        "source": "a local build", "installed_at": 0,
    }), encoding="utf-8")
    record = root / "data" / dflash.CAPABILITY_FILENAME
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps({COMPONENT: {
        "dflash2_text": text, "dflash2_vision": vision,
        "commit": "1deefcca395743049c3820ab8f9b15043f3e9446",
        "fingerprint": dflash._fingerprint(server), "checked_at": 1.0,
        "detail": "verified in a test",
    }}), encoding="utf-8")
    return server


def configure(monkeypatch, install, tmp_path, *, accelerator=accel.ACCEL_AUTO,
              memory_priority=accel.PRIORITY_COOPERATIVE, gpu_index=0,
              gpu_layers="all", mode="gpu", vision=False, context=8192):
    """The configuration ``mc_llm_runtime`` would resolve for this machine."""
    bundle = install / "models" / "managed" / "qwen38-test"
    server = tmp_path / "llama-server"
    server.write_bytes(b"")
    configuration = runtime.Config(
        runtime=server, model=bundle / "model.gguf",
        mmproj=(bundle / "mmproj.gguf") if vision else None,
        gpu_index=gpu_index, device=f"CUDA{gpu_index}", gpu_layers=gpu_layers,
        context_size=context, context_mode="fixed", context_buffer_gb=0.0,
        kv_type_k="q8_0", kv_type_v="q8_0", mode=mode, source=managed.SOURCE_MANAGED,
        managed_id="qwen38-test", accelerator=accelerator,
        memory_priority=memory_priority,
        profile=managed_profiles.profile("qwen38-27b-prompt-author"))
    monkeypatch.setattr(runtime, "config", lambda role="": configuration)
    return configuration


def set_free(monkeypatch, gigabytes, *, card=None):
    """A card with this much free, to both of the questions that asks."""
    def device_free(index=None):
        if card is not None and index is not None and int(index) != card:
            return 0
        return int(gigabytes * _GB)

    monkeypatch.setattr(mc_broker, "free_vram_bytes", lambda: int(gigabytes * _GB))
    monkeypatch.setattr(mc_broker, "device_free_vram_bytes", device_free)


def with_flags(monkeypatch, flags=DFLASH_FLAGS):
    """Every build in this test advertises ``flags``, without a subprocess."""
    monkeypatch.setattr(runtime, "runtime_capabilities",
                        lambda configuration=None: frozenset(flags))
    monkeypatch.setattr(runtime, "_flash_attention_takes_a_value",
                        lambda configuration=None: True)


class Counting:
    """An image reclaimer that records being asked, which is the whole assertion.

    Section 3's cooperative rule is not "little was released", it is *nothing
    was asked for*. A test that checked free VRAM afterwards would pass against
    an implementation that asked and was refused.
    """

    def __init__(self, holds=0):
        self.calls: list[tuple[int, str]] = []
        self.holds = holds

    def release(self, needed_bytes, reason=""):
        self.calls.append((int(needed_bytes), reason))
        freed, self.holds = min(self.holds, int(needed_bytes)), max(
            self.holds - int(needed_bytes), 0)
        return freed

    def resident_bytes(self):
        return self.holds

    def describe(self):
        return "the image checkpoint"


@pytest.fixture
def image(monkeypatch):
    """An image family that counts what it is asked for, on card 0 by default."""
    counter = Counting()
    mc_broker.register_reclaimer(mc_broker.FAMILY_IMAGE, counter)
    monkeypatch.setattr(mc_broker, "image_device_index", lambda: 0)
    yield counter
    mc_broker.unregister_reclaimer(mc_broker.FAMILY_IMAGE)
    mc_broker.register_reclaimer(mc_broker.FAMILY_IMAGE, mc_broker._ImageReclaimer())


# --------------------------------------------------------------------------- #
# The two axes, and the presets over them (test plan E)
# --------------------------------------------------------------------------- #


class TestTheAxes:
    def test_the_defaults_are_what_every_build_before_this_one_did(self):
        """An upgrade must not begin releasing image VRAM because a feature exists."""
        found = accel.settings()

        assert found.accelerator == accel.ACCEL_AUTO
        assert found.memory_priority == accel.PRIORITY_COOPERATIVE
        assert found.cooperative
        assert not found.forced

    @pytest.mark.parametrize("preset, axes", [
        (accel.PRESET_NORMAL, (accel.ACCEL_AUTO, accel.PRIORITY_COOPERATIVE)),
        (accel.PRESET_FAST, (accel.ACCEL_MTP, accel.PRIORITY_LLM)),
        (accel.PRESET_LIGHTNING, (accel.ACCEL_DFLASH2, accel.PRIORITY_LLM)),
    ])
    def test_each_preset_is_exactly_two_settings(self, preset, axes):
        assert accel.preset_axes(preset) == axes
        assert accel.remember(preset=preset) == accel.Settings(*axes)

    def test_the_advanced_combination_no_preset_offers_is_accepted(self):
        """DFlash2 with cooperative memory: the fast decoder on a card that
        already has room, with nothing evicted to get it. Section 3 calls this
        combination mandatory, and no preset names it."""
        found = accel.remember(accelerator=accel.ACCEL_DFLASH2,
                               memory_priority=accel.PRIORITY_COOPERATIVE)

        assert found.accelerator == accel.ACCEL_DFLASH2
        assert found.cooperative
        assert found.preset == accel.PRESET_CUSTOM

    def test_an_axis_wins_over_the_preset_it_arrives_with(self):
        """Which is the direction a user moves in: choose Lightning, then open
        Advanced and turn the eviction back off."""
        found = accel.remember(preset=accel.PRESET_LIGHTNING,
                               memory_priority=accel.PRIORITY_COOPERATIVE)

        assert found.accelerator == accel.ACCEL_DFLASH2
        assert found.memory_priority == accel.PRIORITY_COOPERATIVE

    def test_a_role_can_answer_differently_from_the_installation(self):
        """A 3090 Creative Writer and a 5090 Spatial Composer have different
        amounts of room, so section 15 asks for these to be role-aware."""
        accel.remember(preset=accel.PRESET_NORMAL)
        accel.remember(role=roles.CREATIVE, preset=accel.PRESET_LIGHTNING)

        assert accel.settings(roles.CREATIVE).preset == accel.PRESET_LIGHTNING
        assert accel.settings().preset == accel.PRESET_NORMAL
        assert accel.settings(roles.SPATIAL).preset == accel.PRESET_NORMAL

    def test_a_role_that_follows_the_installation_is_not_a_copy_of_it(self):
        accel.remember(preset=accel.PRESET_FAST)

        assert accel.follows_installation(roles.SPATIAL)
        assert accel.settings(roles.SPATIAL).preset == accel.PRESET_FAST

    def test_the_role_section_survives_being_read_back(self):
        """The preferences reader keeps only keys it knows, and the role
        overrides live under one it did not: they were written and dropped."""
        accel.remember(role=roles.CREATIVE, preset=accel.PRESET_LIGHTNING)

        assert roles.SECTION in mc_llm_state.preferences()

    def test_the_two_modules_agree_on_where_role_overrides_live(self):
        assert mc_llm_state.ROLES_SECTION == roles.SECTION


# --------------------------------------------------------------------------- #
# The catalogue entry (test plan A and B)
# --------------------------------------------------------------------------- #


class TestTheShippedQwen:
    def test_all_three_tiers_share_one_profile_and_one_projector(self):
        tiers = [managed.entry(f"qwen38-27b-abliterated-{name}")
                 for name in ("q6k", "q5km", "q4km")]

        assert {model.profile_id for model in tiers} == {"qwen38-27b-prompt-author"}
        assert len({model.projector.sha256 for model in tiers}) == 1
        assert {model.model.display_size for model in tiers} == {
            "~22.4 GB", "~19.5 GB", "~16.8 GB"}

    def test_the_high_tier_is_q6_and_not_q8(self):
        """Q8_0 is about 29 GB, which leaves a 32 GB card too little for a
        draft model, 8K of state, compute buffers and a projector."""
        high = managed.entry("qwen38-27b-abliterated-q6k")

        assert high.model.filename.endswith("-Q6_K.gguf")
        assert not any("Q8" in model.model.filename for model in managed.catalogue()
                       if model.family == "Qwen 3.8")

    def test_every_tier_declares_both_accelerators(self):
        for name in ("q6k", "q5km", "q4km"):
            found = managed.entry(f"qwen38-27b-abliterated-{name}").accelerators

            assert found.mtp is not None and found.mtp.embedded
            assert found.dflash2 is not None
            assert found.dflash2.runtime_family == FAMILY
            assert found.dflash2.requires_full_target_gpu
            assert found.dflash2.requires_full_draft_gpu
            assert found.dflash2.same_gpu_as_target

    def test_the_draft_is_not_part_of_an_ordinary_download(self):
        """3.86 GB that only a machine choosing Lightning will ever load."""
        model = managed.entry("qwen38-27b-abliterated-q4km")

        assert [artifact.local_name for artifact in model.artifacts] == [
            managed.MODEL_FILENAME, managed.MMPROJ_FILENAME]
        assert model.draft is not None
        assert model.draft.local_name == managed.DRAFT_FILENAME

    def test_the_registry_never_names_a_local_path_of_its_own(self):
        """The publisher's filename builds the URL and is recorded; the file on
        disk is always one of the five names this module fixed."""
        for model in managed.catalogue():
            for artifact in (*model.artifacts, model.draft):
                if artifact is None:
                    continue
                assert artifact.local_name in (managed.MODEL_FILENAME,
                                               managed.MMPROJ_FILENAME,
                                               managed.DRAFT_FILENAME)

    def test_no_entry_carries_an_executable_command_string(self):
        """Section 15: declarative metadata, never free-form command lines."""
        raw = json.loads(managed.REGISTRY_PATH.read_text(encoding="utf-8"))
        text = json.dumps(raw)

        assert "--spec" not in text
        assert "llama-server" not in text


class TestTheQwenProfile:
    def test_it_is_the_prompt_authoring_profile_the_intent_specifies(self):
        found = managed_profiles.profile("qwen38-27b-prompt-author")

        assert found.context == 8192
        assert (found.kv_type_k, found.kv_type_v) == ("q8_0", "q8_0")
        assert found.jinja
        assert not found.thinking

    def test_the_samplers_are_qwens_anchors_with_no_novelty_penalty(self):
        """An image prompt repeats its subject, its colours and its materials on
        purpose, so a presence penalty is behavioural drift rather than variety."""
        found = managed_profiles.sampler_arguments(
            managed_profiles.profile("qwen38-27b-prompt-author"))

        assert found == {"top_k": 20, "min_p": 0.0, "repeat_penalty": 1.0,
                         "presence_penalty": 0.0, "frequency_penalty": 0.0}

    def test_temperature_and_top_p_stay_where_creativity_owns_them(self):
        found = managed_profiles.profile("qwen38-27b-prompt-author")

        assert "temperature" not in found.sampling
        assert "top_p" not in found.sampling
        assert not hasattr(found, "temperature")


# --------------------------------------------------------------------------- #
# A hybrid model's memory (design intent section 9)
# --------------------------------------------------------------------------- #


class TestHybridState:
    def test_only_the_attending_blocks_grow_a_cache_with_the_context(self, tmp_path):
        model = mc_gguf.read(hybrid_model(tmp_path / "hybrid.gguf"))
        placement = ctx.Placement(context=8192, kv_type_k="f16", kv_type_v="f16")

        # Two of eight blocks attend, at 8 KV heads and 128 wide each way.
        assert model.attending_blocks == 2
        assert ctx.kv_bytes_per_token(model, placement) == 2 * 8 * (128 + 128) * 2.0

    def test_the_recurrent_blocks_are_charged_for_their_state(self, tmp_path):
        model = mc_gguf.read(hybrid_model(tmp_path / "hybrid.gguf"))
        placement = ctx.Placement(context=8192)

        # (4 - 1) x (4096 + 2 x 8 x 128) convolution, plus 128 x 4096 of state,
        # in f32, for each of the six blocks that keep one.
        per_block = (3 * (4096 + 2 * 8 * 128) + 128 * 4096) * 4
        assert model.recurrent_blocks == 6
        assert ctx.recurrent_bytes(model, placement) == 6 * per_block

    def test_the_state_does_not_grow_with_the_context(self, tmp_path):
        """One slot per sequence, written in place. Folding it into the cache
        term would make it scale with a number it does not follow."""
        model = mc_gguf.read(hybrid_model(tmp_path / "hybrid.gguf"))

        assert (ctx.recurrent_bytes(model, ctx.Placement(context=2048))
                == ctx.recurrent_bytes(model, ctx.Placement(context=8192)))

    def test_a_header_that_will_not_say_gets_a_conservative_allowance(self, tmp_path):
        """Never zero. Under-charging ends in an allocation failure at load;
        over-charging ends in a slightly smaller context."""
        model = mc_gguf.read(hybrid_model(tmp_path / "quiet.gguf", recurrent=False))
        charged = ctx.recurrent_bytes(model, ctx.Placement())

        assert not model.recurrent_state_described
        assert charged == model.recurrent_blocks * ctx.RECURRENT_FALLBACK_BYTES

    def test_an_ordinary_transformer_is_charged_nothing(self, tmp_path):
        from test_llm_context import build_model

        model = mc_gguf.read(build_model(tmp_path))

        assert not model.recurrent
        assert ctx.recurrent_bytes(model, ctx.Placement()) == 0

    def test_the_estimate_carries_it_as_its_own_term(self, tmp_path):
        model = hybrid_model(tmp_path / "hybrid.gguf")
        found = ctx.estimate(model, ctx.Placement(context=8192))

        assert found.state_bytes > 0
        assert found.total_bytes == (found.weights_bytes + found.kv_bytes
                                     + found.state_bytes + found.compute_bytes)
        assert found.resident_bytes == (found.weights_bytes + found.kv_bytes
                                        + found.state_bytes)


# --------------------------------------------------------------------------- #
# Cooperative DFlash2 (test plan F)
# --------------------------------------------------------------------------- #


class TestCooperativeDFlash2:
    def test_it_runs_without_releasing_anything_when_the_plan_already_fits(
            self, install, tmp_path, monkeypatch, registry, image):
        """The case the whole two-axis design exists for."""
        install_bundle(install, tmp_path)
        install_dflash_runtime(install)
        with_flags(monkeypatch)
        set_free(monkeypatch, 24)
        configuration = configure(monkeypatch, install, tmp_path,
                                  accelerator=accel.ACCEL_DFLASH2,
                                  memory_priority=accel.PRIORITY_COOPERATIVE)
        image.holds = 20 * _GB

        plan = runtime.accelerator_plan(configuration)

        assert plan.accelerator == accel.ACCEL_DFLASH2
        assert not plan.refused
        assert plan.reclaimed_bytes == 0
        assert image.calls == []

    def test_a_plan_that_does_not_fit_releases_nothing_and_says_why(
            self, install, tmp_path, monkeypatch, registry, image):
        install_bundle(install, tmp_path)
        install_dflash_runtime(install)
        with_flags(monkeypatch)
        set_free(monkeypatch, 0.001)
        configuration = configure(monkeypatch, install, tmp_path,
                                  accelerator=accel.ACCEL_DFLASH2,
                                  memory_priority=accel.PRIORITY_COOPERATIVE)
        image.holds = 20 * _GB

        plan = runtime.accelerator_plan(configuration)

        assert plan.refused
        assert image.calls == []
        assert "No image VRAM was released." in plan.refusal
        assert "Required estimate" in plan.refusal
        assert "Spendable now" in plan.refusal

    def test_it_carries_the_publishers_start_contract(
            self, install, tmp_path, monkeypatch, registry, image):
        install_bundle(install, tmp_path)
        install_dflash_runtime(install)
        with_flags(monkeypatch)
        set_free(monkeypatch, 24)
        configuration = configure(monkeypatch, install, tmp_path,
                                  accelerator=accel.ACCEL_DFLASH2)

        flags = runtime.accelerator_plan(configuration).flags

        assert flags[:2] == ("--spec-draft-model", str(install / "models" / "managed"
                                                       / "qwen38-test" / "draft.gguf"))
        for flag, value in (("--spec-type", "draft-dflash"), ("--spec-draft-n-max", "3"),
                            ("--spec-draft-n-min", "0"), ("--spec-draft-p-min", "0"),
                            ("--spec-draft-type-k", "f16"), ("--spec-draft-type-v", "f16")):
            assert flags[flags.index(flag) + 1] == value
        assert "--n-gpu-layers-draft" in flags

    def test_a_build_without_the_options_is_refused_rather_than_started(
            self, install, tmp_path, monkeypatch, registry, image):
        """A flag passed to a build that does not know it is a server that
        exits at startup, not a slower one."""
        install_bundle(install, tmp_path)
        install_dflash_runtime(install)
        with_flags(monkeypatch, frozenset({"--flash-attn"}))
        set_free(monkeypatch, 24)
        configuration = configure(monkeypatch, install, tmp_path,
                                  accelerator=accel.ACCEL_DFLASH2)

        plan = runtime.accelerator_plan(configuration)

        assert plan.refused
        assert "does not accept the speculative draft options" in plan.refusal

    def test_a_partial_offload_is_refused_rather_than_labelled_lightning(
            self, install, tmp_path, monkeypatch, registry, image):
        """Section 6: no silent degradation. The target has to be whole."""
        install_bundle(install, tmp_path)
        install_dflash_runtime(install)
        with_flags(monkeypatch)
        set_free(monkeypatch, 24)
        configuration = configure(monkeypatch, install, tmp_path,
                                  accelerator=accel.ACCEL_DFLASH2, gpu_layers="4")

        plan = runtime.accelerator_plan(configuration)

        assert plan.refused
        assert "every layer of the backbone resident" in plan.refusal
        assert image.calls == []

    def test_the_processor_is_told_it_is_the_wrong_kind_of_machine(
            self, install, tmp_path, monkeypatch, registry, image):
        install_bundle(install, tmp_path)
        install_dflash_runtime(install)
        with_flags(monkeypatch)
        set_free(monkeypatch, 24)
        configuration = configure(monkeypatch, install, tmp_path,
                                  accelerator=accel.ACCEL_DFLASH2, mode="cpu")

        plan = runtime.accelerator_plan(configuration)

        assert plan.refused
        assert "DFlash2 is a CUDA path" in plan.refusal

    def test_the_draft_is_charged_from_its_own_header_and_cache_types(
            self, install, tmp_path, monkeypatch, registry, image):
        """A BF16 sidecar beside a quantised 27B: charging it the target's
        bytes-per-token would be arithmetic about the wrong file."""
        install_bundle(install, tmp_path, draft_mb=6)
        install_dflash_runtime(install)
        with_flags(monkeypatch)
        set_free(monkeypatch, 24)
        configuration = configure(monkeypatch, install, tmp_path,
                                  accelerator=accel.ACCEL_DFLASH2)

        plan = runtime.accelerator_plan(configuration)
        draft = install / "models" / "managed" / "qwen38-test" / "draft.gguf"

        assert plan.draft_bytes >= draft.stat().st_size
        assert plan.required_bytes > plan.draft_bytes


# --------------------------------------------------------------------------- #
# LLM priority, and the card it is scoped to (test plan G and J)
# --------------------------------------------------------------------------- #


class TestLLMPriority:
    def test_a_plan_that_already_fits_reclaims_nothing(
            self, install, tmp_path, monkeypatch, registry, image):
        install_bundle(install, tmp_path)
        install_dflash_runtime(install)
        with_flags(monkeypatch)
        set_free(monkeypatch, 24)
        configuration = configure(monkeypatch, install, tmp_path,
                                  accelerator=accel.ACCEL_DFLASH2,
                                  memory_priority=accel.PRIORITY_LLM)
        image.holds = 20 * _GB

        plan = runtime.accelerator_plan(configuration)

        assert plan.accelerator == accel.ACCEL_DFLASH2
        assert image.calls == []

    def test_it_asks_the_same_card_for_the_deficit_and_then_measures_again(
            self, install, tmp_path, monkeypatch, registry, image):
        install_bundle(install, tmp_path)
        install_dflash_runtime(install)
        with_flags(monkeypatch)
        configuration = configure(monkeypatch, install, tmp_path,
                                  accelerator=accel.ACCEL_DFLASH2,
                                  memory_priority=accel.PRIORITY_LLM)
        image.holds = 20 * _GB
        readings = iter([0.001, 0.001, 24, 24, 24, 24])
        monkeypatch.setattr(mc_broker, "device_free_vram_bytes",
                            lambda index=None: int(next(readings, 24) * _GB))
        monkeypatch.setattr(mc_broker, "free_vram_bytes", lambda: 24 * _GB)

        plan = runtime.accelerator_plan(configuration)

        assert image.calls, "the deficit was never asked for"
        assert plan.accelerator == accel.ACCEL_DFLASH2
        assert plan.reclaimed_bytes > 0

    def test_a_creative_writer_on_one_card_never_empties_the_other(
            self, install, tmp_path, monkeypatch, registry, image):
        """Section 10, example A: the image side is on the 5090 and the writer
        is on the 3090. Reclaiming the 5090 cannot free a byte of the 3090."""
        install_bundle(install, tmp_path)
        install_dflash_runtime(install)
        with_flags(monkeypatch)
        set_free(monkeypatch, 0.001)
        monkeypatch.setattr(mc_broker, "image_device_index", lambda: 1)
        configuration = configure(monkeypatch, install, tmp_path, gpu_index=0,
                                  accelerator=accel.ACCEL_DFLASH2,
                                  memory_priority=accel.PRIORITY_LLM)
        image.holds = 30 * _GB

        plan = runtime.accelerator_plan(configuration)

        assert plan.refused
        assert image.calls == []

    def test_the_broker_refuses_a_cross_card_release_on_its_own_terms(self, monkeypatch,
                                                                      image):
        """Asserted at the broker as well as through the planner, because this
        is the rule and the planner is only one of its callers."""
        set_free(monkeypatch, 1)
        image.holds = 30 * _GB

        result = mc_broker.release_for_llm(50 * _GB, card=1, reason="a test")

        assert image.calls == []
        assert result.freed == 0
        assert not result.satisfied

    def test_an_unknown_card_releases_nothing(self, monkeypatch, image):
        """The caution runs the other way here from the placement side: an
        unanswerable question costs a smaller model there and an evicted
        checkpoint here, so this one answers no."""
        set_free(monkeypatch, 1)
        image.holds = 30 * _GB

        assert mc_broker.release_for_llm(50 * _GB, card=None).freed == 0
        assert image.calls == []

    def test_an_ordinary_llm_request_still_cannot_evict_the_image_model(self, monkeypatch,
                                                                         image):
        """The default rule is untouched: only the explicit path opens the door."""
        set_free(monkeypatch, 1)
        image.holds = 30 * _GB

        mc_broker.request_vram(mc_broker.FAMILY_LLM, 50 * _GB)

        assert image.calls == []


# --------------------------------------------------------------------------- #
# What runs when nobody asked for anything in particular (test plan E)
# --------------------------------------------------------------------------- #


class TestAuto:
    def test_it_prefers_dflash2_when_everything_is_in_place(
            self, install, tmp_path, monkeypatch, registry, image):
        install_bundle(install, tmp_path)
        install_dflash_runtime(install)
        with_flags(monkeypatch)
        set_free(monkeypatch, 24)
        configuration = configure(monkeypatch, install, tmp_path)

        assert runtime.accelerator_plan(configuration).accelerator == accel.ACCEL_DFLASH2

    def test_it_steps_down_to_the_backbones_own_heads_without_a_draft(
            self, install, tmp_path, monkeypatch, registry, image):
        install_bundle(install, tmp_path, draft=False)
        with_flags(monkeypatch)
        set_free(monkeypatch, 24)
        configuration = configure(monkeypatch, install, tmp_path)

        plan = runtime.accelerator_plan(configuration)

        assert plan.accelerator == accel.ACCEL_MTP
        assert not plan.refused
        assert plan.flags[:2] == ("--spec-type", "mtp")

    def test_it_steps_down_to_ordinary_decoding_on_an_older_runtime(
            self, install, tmp_path, monkeypatch, registry, image):
        install_bundle(install, tmp_path, draft=False)
        with_flags(monkeypatch, frozenset())
        set_free(monkeypatch, 24)
        configuration = configure(monkeypatch, install, tmp_path)

        plan = runtime.accelerator_plan(configuration)

        assert plan.accelerator == accel.ACCEL_NONE
        assert not plan.refused
        assert plan.flags == ()

    def test_it_never_refuses(self, install, tmp_path, monkeypatch, registry, image):
        """Auto is a policy rather than a request, so there is nothing to
        refuse: it names what it used and carries on."""
        install_bundle(install, tmp_path, draft=False)
        with_flags(monkeypatch)
        set_free(monkeypatch, 0.001)
        configuration = configure(monkeypatch, install, tmp_path)

        assert not runtime.accelerator_plan(configuration).refused

    def test_a_manual_gguf_gets_the_decoding_it_has_always_had(
            self, install, tmp_path, monkeypatch, registry, image):
        """An accelerator claim is a statement about a specific pinned file."""
        install_bundle(install, tmp_path)
        install_dflash_runtime(install)
        with_flags(monkeypatch)
        set_free(monkeypatch, 24)
        configuration = configure(monkeypatch, install, tmp_path)
        configuration = dataclasses_replace(configuration, managed_id="",
                                            source=managed.SOURCE_MANUAL)
        monkeypatch.setattr(runtime, "config", lambda role="": configuration)

        assert runtime.accelerator_plan(configuration).accelerator == accel.ACCEL_NONE


def dataclasses_replace(value, **changes):
    from dataclasses import replace

    return replace(value, **changes)


# --------------------------------------------------------------------------- #
# The failure contract (test plan D, K and M)
# --------------------------------------------------------------------------- #


class TestForcedRequestsThatCannotBeMet:
    def test_a_missing_runtime_names_the_thing_to_install(
            self, install, tmp_path, monkeypatch, registry, image):
        install_bundle(install, tmp_path)
        with_flags(monkeypatch)
        set_free(monkeypatch, 24)
        configuration = configure(monkeypatch, install, tmp_path,
                                  accelerator=accel.ACCEL_DFLASH2)

        plan = runtime.accelerator_choice(configuration)

        assert plan.refused
        assert "Install the DFlash2 runtime in Setup" in plan.refusal

    def test_a_missing_sidecar_names_the_thing_to_install(
            self, install, tmp_path, monkeypatch, registry, image):
        install_bundle(install, tmp_path, draft=False)
        install_dflash_runtime(install)
        with_flags(monkeypatch)
        configuration = configure(monkeypatch, install, tmp_path,
                                  accelerator=accel.ACCEL_DFLASH2)

        plan = runtime.accelerator_choice(configuration)

        assert plan.refused
        assert "draft model, which has not been downloaded" in plan.refusal

    def test_help_text_alone_does_not_make_a_runtime_usable(
            self, install, tmp_path, monkeypatch, registry, image):
        """Upstream llama.cpp already carries DFlash terminology. The gate is a
        real load of the real target with the real sidecar."""
        install_bundle(install, tmp_path)
        install_dflash_runtime(install, text=False)
        with_flags(monkeypatch)
        configuration = configure(monkeypatch, install, tmp_path,
                                  accelerator=accel.ACCEL_DFLASH2)

        plan = runtime.accelerator_choice(configuration)

        assert plan.refused
        assert "has not passed the text smoke test" in plan.refusal

    def test_vision_is_gated_apart_from_text(
            self, install, tmp_path, monkeypatch, registry, image):
        """The pull request's multimodal work moved separately from its text
        path, so text passing says nothing about an image request."""
        install_bundle(install, tmp_path)
        install_dflash_runtime(install, text=True, vision=False)
        with_flags(monkeypatch)
        set_free(monkeypatch, 24)
        configuration = configure(monkeypatch, install, tmp_path, vision=True,
                                  accelerator=accel.ACCEL_DFLASH2)

        text_request = runtime.accelerator_choice(configuration, needs_vision=False)
        image_request = runtime.accelerator_choice(configuration, needs_vision=True)

        assert text_request.accelerator == accel.ACCEL_DFLASH2
        assert image_request.refused
        assert "DFlash2 vision is not validated" in image_request.refusal

    def test_a_verified_vision_runtime_takes_the_image_request(
            self, install, tmp_path, monkeypatch, registry, image):
        install_bundle(install, tmp_path)
        install_dflash_runtime(install, text=True, vision=True)
        with_flags(monkeypatch)
        set_free(monkeypatch, 24)
        configuration = configure(monkeypatch, install, tmp_path, vision=True,
                                  accelerator=accel.ACCEL_DFLASH2)

        assert runtime.accelerator_choice(
            configuration, needs_vision=True).accelerator == accel.ACCEL_DFLASH2

    def test_forced_mtp_on_a_backbone_without_heads_is_not_a_refusal(
            self, install, tmp_path, monkeypatch, registry, image):
        """Section 3 defines Fast LLM as MTP *when supported*, otherwise
        ordinary decoding -- with the memory priority the preset also carries."""
        install_bundle(install, tmp_path, draft=False)
        with_flags(monkeypatch)
        set_free(monkeypatch, 24)
        configuration = configure(monkeypatch, install, tmp_path,
                                  accelerator=accel.ACCEL_MTP,
                                  memory_priority=accel.PRIORITY_LLM)
        monkeypatch.setattr(runtime, "_advertised_accelerators",
                            lambda _configuration: managed.Accelerators())

        plan = runtime.accelerator_plan(configuration)

        assert not plan.refused
        assert plan.accelerator == accel.ACCEL_NONE
        assert plan.memory_priority == accel.PRIORITY_LLM
        assert any("no multi-token prediction heads" in note for note in plan.notes)

    def test_forced_mtp_on_an_older_runtime_says_which_half_is_missing(
            self, install, tmp_path, monkeypatch, registry, image):
        """The backbone has the heads and the build cannot be told to use them,
        which is a different sentence and a different thing to do about it."""
        install_bundle(install, tmp_path, draft=False)
        with_flags(monkeypatch, frozenset())
        set_free(monkeypatch, 24)
        configuration = configure(monkeypatch, install, tmp_path,
                                  accelerator=accel.ACCEL_MTP)

        plan = runtime.accelerator_plan(configuration)

        assert not plan.refused
        assert plan.accelerator == accel.ACCEL_NONE
        assert any("Update the runtime" in note for note in plan.notes)


# --------------------------------------------------------------------------- #
# Warm runtime identity (test plan L)
# --------------------------------------------------------------------------- #


class TestWarmIdentity:
    def _configured(self, install, tmp_path, monkeypatch, **changes):
        return configure(monkeypatch, install, tmp_path, **changes)

    def test_changing_the_accelerator_invalidates_a_warm_server(
            self, install, tmp_path, monkeypatch, registry):
        install_bundle(install, tmp_path)
        first = self._configured(install, tmp_path, monkeypatch,
                                 accelerator=accel.ACCEL_MTP)
        second = self._configured(install, tmp_path, monkeypatch,
                                  accelerator=accel.ACCEL_DFLASH2)

        assert runtime._identity(first) != runtime._identity(second)

    def test_changing_the_memory_priority_invalidates_a_warm_server(
            self, install, tmp_path, monkeypatch, registry):
        install_bundle(install, tmp_path)
        first = self._configured(install, tmp_path, monkeypatch,
                                 memory_priority=accel.PRIORITY_COOPERATIVE)
        second = self._configured(install, tmp_path, monkeypatch,
                                  memory_priority=accel.PRIORITY_LLM)

        assert runtime._identity(first) != runtime._identity(second)

    def test_the_plan_identity_distinguishes_everything_section_16_lists(self):
        ordinary = accel.Plan()
        speculative = accel.Plan(accelerator=accel.ACCEL_DFLASH2,
                                 runtime=Path("special/llama-server"),
                                 draft=Path("bundle/draft.gguf"), runtime_family=FAMILY)
        other_draft = dataclasses_replace(speculative, draft=Path("other/draft.gguf"))
        other_runtime = dataclasses_replace(speculative,
                                            runtime=Path("ordinary/llama-server"))

        assert len({ordinary.identity, speculative.identity, other_draft.identity,
                    other_runtime.identity}) == 4

    def test_a_signature_tells_two_runtimes_apart(self, install, tmp_path, monkeypatch,
                                                  registry):
        install_bundle(install, tmp_path)
        configuration = configure(monkeypatch, install, tmp_path)
        placement = ctx.Placement()
        speculative = accel.Plan(accelerator=accel.ACCEL_DFLASH2,
                                 runtime=Path("special/llama-server"),
                                 draft=Path("bundle/draft.gguf"), runtime_family=FAMILY)

        assert (runtime._signature_of(configuration, None, placement)
                != runtime._signature_of(configuration, None, placement, speculative))


# --------------------------------------------------------------------------- #
# The launch boundary (test plan M)
# --------------------------------------------------------------------------- #


class TestFlagsAtTheBoundary:
    def test_speculative_flags_reach_the_command_line_once(self):
        runtime._arm_flags(["--spec-draft-model", "draft.gguf", "--spec-type",
                            "draft-dflash"])
        command = ["llama-server", "--model", "m.gguf", "--ctx-size", "8192"]

        first = runtime.with_extra_flags(command)
        second = runtime.with_extra_flags(command)

        assert "--spec-draft-model" in first
        assert "--spec-draft-model" not in second, "an armed flag was used twice"

    def test_they_do_not_leak_into_a_command_that_is_not_a_server(self):
        runtime._arm_flags(["--spec-type", "draft-dflash"])

        assert runtime.with_extra_flags(["llama-server", "--help"]) == [
            "llama-server", "--help"]

    def test_flash_attention_is_not_passed_twice(self, install, tmp_path, monkeypatch,
                                                 registry):
        """The placement adds it for a resident model and the DFlash contract
        asks for it too; a switch-style build passed it twice is a build passed
        an argument it will try to parse as a value."""
        install_bundle(install, tmp_path)
        with_flags(monkeypatch)
        configuration = configure(monkeypatch, install, tmp_path)
        plan = accel.Plan(accelerator=accel.ACCEL_DFLASH2,
                          flags=("--spec-type", "draft-dflash", "--flash-attn", "on"))

        flags = runtime._launch_flags(configuration, ctx.Placement(), plan)

        assert flags.count("--flash-attn") == 1
        assert "--spec-type" in flags


# --------------------------------------------------------------------------- #
# Measured speed, kept apart (design intent section 17)
# --------------------------------------------------------------------------- #


class TestMeasurement:
    def test_a_dflash_rate_is_not_averaged_into_an_ordinary_one(self):
        resident = ctx.Placement()

        ordinary = runtime.measurement_token(resident, accel.ACCEL_NONE, 0)
        speculative = runtime.measurement_token(resident, accel.ACCEL_DFLASH2, 0)

        assert ordinary != speculative
        assert runtime.speed_key(runtime.WRITE_RATE, "qwen38", ordinary) != \
            runtime.speed_key(runtime.WRITE_RATE, "qwen38", speculative)

    def test_the_physical_card_is_part_of_the_key(self):
        assert (runtime.measurement_token(ctx.Placement(), accel.ACCEL_DFLASH2, 0)
                != runtime.measurement_token(ctx.Placement(), accel.ACCEL_DFLASH2, 1))

    def test_an_ordinary_run_keeps_the_key_it_always_had(self):
        """Every rate this machine has already measured has to keep answering."""
        assert runtime.measurement_token(ctx.Placement(), accel.ACCEL_NONE, None) == "gpu"

    def test_the_token_reads_back_as_a_sentence(self):
        said = runtime.describe_placement_token(
            runtime.measurement_token(ctx.Placement(), accel.ACCEL_DFLASH2, 0))

        assert "DFlash2" in said
        assert "GPU 0" in said

    @pytest.mark.parametrize("line, drafted, accepted", [
        ("n_draft = 30, n_accept = 21", 30, 21),
        ("draft acceptance rate: 21 / 30", 30, 21),
    ])
    def test_acceptance_is_read_out_of_the_log_where_it_is_printed(self, line, drafted,
                                                                   accepted):
        found = runtime.read_speculation(line)

        assert found.known
        assert (found.drafted, found.accepted) == (drafted, accepted)
        assert found.acceptance == pytest.approx(accepted / drafted)

    def test_a_build_that_reports_nothing_is_not_reported_as_zero(self):
        """"No drafted tokens were accepted" and "this build does not say" are
        different news."""
        found = runtime.read_speculation("nothing about speculation here")

        assert not found.known
        assert found.describe() == ""


# --------------------------------------------------------------------------- #
# The sidecar, installed on its own (test plan A)
# --------------------------------------------------------------------------- #


@pytest.fixture
def fetched(monkeypatch, tmp_path):
    """``_fetch``, writing the file it was asked for instead of downloading it.

    The transfer itself is the vendored downloader and has its own tests; what
    is under test here is the transaction around it -- what exists on disk after
    each way of failing.
    """
    written: list[Path] = []

    def fetch(model, artifact, destination, report, say):
        destination.parent.mkdir(parents=True, exist_ok=True)
        hybrid_model(destination, blocks=2, attending=2, size_mb=1, recurrent=False)
        written.append(destination)
        report(1, 1)

    monkeypatch.setattr(managed, "_fetch", fetch)
    return written


class TestInstallingTheDraft:
    def test_it_refuses_before_the_backbone_is_downloaded(self, install, tmp_path,
                                                          registry, fetched):
        with pytest.raises(managed.ManagedError, match="has to be downloaded"):
            managed.install_draft("qwen38-test")

        assert fetched == []

    def test_it_lands_in_the_bundle_and_in_the_manifest(self, install, tmp_path,
                                                        registry, fetched):
        install_bundle(install, tmp_path, draft=False)

        bundle = managed.install_draft("qwen38-test")

        assert bundle.draft is not None and bundle.draft.is_file()
        assert bundle.drafts(managed.entry("qwen38-test"))
        recorded = json.loads((bundle.root / managed.INSTALLED_FILENAME)
                              .read_text(encoding="utf-8"))
        assert recorded["artifacts"]["draft"]["sha256"] == DRAFT_SHA
        assert recorded["artifacts"]["draft"]["filename"] == "dflash-test-BF16.gguf"
        assert recorded["artifacts"]["draft"]["revision"] == "e" * 40

    def test_a_second_install_downloads_nothing(self, install, tmp_path, registry,
                                                fetched):
        install_bundle(install, tmp_path)

        managed.install_draft("qwen38-test")

        assert fetched == []

    def test_a_bundle_without_a_sidecar_is_still_current(self, install, tmp_path,
                                                         registry):
        """An absent draft is the normal state of a complete install, so asking
        for it in ``matches`` would report every ordinary bundle as superseded."""
        install_bundle(install, tmp_path, draft=False)
        model = managed.entry("qwen38-test")

        found = managed.installed("qwen38-test")

        assert found.matches(model)
        assert not found.drafts(model)
        assert managed.status(model).ready

    def test_a_file_whose_hash_the_catalogue_has_moved_off_is_not_used(
            self, install, tmp_path, registry, monkeypatch):
        """A draft nobody tested with these weights is worse than no draft."""
        install_bundle(install, tmp_path)
        moved = tmp_path / "moved.json"
        moved.write_text(json.dumps(registry_document(sha256="d" * 64)), encoding="utf-8")
        monkeypatch.setattr(managed, "REGISTRY_PATH", moved)
        managed._registry_cache = None

        assert not managed.installed("qwen38-test").drafts(managed.entry("qwen38-test"))

    def test_a_manifest_that_cannot_be_written_takes_the_file_back_out(
            self, install, tmp_path, registry, fetched, monkeypatch):
        """The one ordering that must not exist is a bundle claiming a draft it
        does not have."""
        install_bundle(install, tmp_path, draft=False)
        real = managed._write_json

        def refuse(path, document):
            if Path(path).name == managed.INSTALLED_FILENAME:
                raise OSError("the disk is full")
            return real(path, document)

        monkeypatch.setattr(managed, "_write_json", refuse)

        with pytest.raises(managed.ManagedError, match="manifest could not be updated"):
            managed.install_draft("qwen38-test")

        bundle = managed.installed("qwen38-test")
        assert bundle is not None
        assert bundle.draft is None
        assert not (bundle.root / managed.DRAFT_FILENAME).exists()

    def test_removing_it_leaves_a_working_bundle(self, install, tmp_path, registry,
                                                 fetched):
        install_bundle(install, tmp_path)

        left = managed.remove_draft("qwen38-test")

        assert left is not None
        assert left.draft is None
        assert left.matches(managed.entry("qwen38-test"))
        assert not (left.root / managed.DRAFT_FILENAME).exists()

    def test_its_staging_directory_cannot_collide_with_a_bundle(self, install):
        """``~`` is a character a model id may not contain, which is the whole
        of why a sidecar's staging directory can be named after one."""
        staged = managed.draft_staging_root("qwen38-test")

        assert staged != managed.staging_root("qwen38-test")
        assert managed.DRAFT_STAGING_SUFFIX in staged.name
        assert not managed._ID.match(staged.name)


# --------------------------------------------------------------------------- #
# The special runtime family (test plan D)
# --------------------------------------------------------------------------- #


class TestTheDFlashRuntimeFamily:
    def test_it_never_appears_as_an_ordinary_runtime(self, install):
        """A machine with no ordinary llama.cpp must not silently adopt an
        unmerged pull request as the runtime for every model and every mode."""
        import mc_llm_setup

        install_dflash_runtime(install)

        assert mc_llm_setup.runtime_families(install) == {}
        assert mc_llm_setup.detect() is None
        assert dflash.installed() is not None

    def test_an_ordinary_runtime_beside_it_is_still_found(self, install):
        import mc_llm_setup

        ordinary = install / mc_llm_setup.RUNTIME_DIRNAME
        ordinary.mkdir(parents=True)
        (ordinary / "llama-server").write_bytes(b"")
        install_dflash_runtime(install)

        assert mc_llm_setup.detect() == ordinary / "llama-server"

    def test_it_is_installed_into_a_family_of_its_own(self, install, tmp_path):
        """Never ``runtime/``, even on a machine that has no ordinary build --
        that directory is where every ordinary lookup falls back to."""
        import mc_llm_setup

        build = tmp_path / "build" / "bin"
        build.mkdir(parents=True)
        (build / "llama-server").write_bytes(b"")
        (build / "ggml.dll").write_bytes(b"")

        placed, said = dflash.adopt(build)

        assert placed.parent.name == mc_llm_setup.family_dirname(COMPONENT)
        assert not (install / mc_llm_setup.RUNTIME_DIRNAME).exists()
        assert "verify it" in said
        assert (placed.parent / "ggml.dll").is_file(), "only the executable was taken"

    def test_adopting_a_build_records_which_commit_it_claims_to_be(self, install,
                                                                   tmp_path):
        build = tmp_path / "build"
        build.mkdir()
        (build / "llama-server").write_bytes(b"")
        (build / "ggml.dll").write_bytes(b"")

        dflash.adopt(build)
        source = dflash.provenance(COMPONENT)

        assert source.commit == "1deefcca395743049c3820ab8f9b15043f3e9446"
        assert source.source.endswith("build")

    def test_a_new_build_inherits_nothing_from_the_one_it_replaced(self, install,
                                                                   tmp_path):
        install_dflash_runtime(install, text=True, vision=True)
        build = tmp_path / "build"
        build.mkdir()
        (build / "llama-server").write_bytes(b"")
        (build / "ggml.dll").write_bytes(b"")

        dflash.adopt(build)

        assert not dflash.capability(COMPONENT).known

    def test_an_unpublished_build_is_refused_rather_than_guessed_at(self, install):
        """No route through this module fetches something whose bytes are not
        named in this repository."""
        with pytest.raises(dflash.DFlashError, match="no published archive"):
            dflash.download(COMPONENT)

    def test_the_shipped_manifest_pins_the_commit_blackfrost_tested(self):
        found = dflash.builds()

        assert found
        assert {build.commit for build in found} == {
            "1deefcca395743049c3820ab8f9b15043f3e9446"}
        assert {build.family for build in found} == {FAMILY}
        for build in found:
            assert build.requires_component.startswith("llama-runtime-cuda")
            assert not build.published

    def test_the_two_results_are_recorded_independently(self, install):
        """Text passing says nothing about an image request, so one boolean
        would have to choose which of two lies to tell."""
        install_dflash_runtime(install, text=False)

        dflash.record_capability(COMPONENT, text=True, detail="text passed")
        assert dflash.capability(COMPONENT).text
        assert not dflash.capability(COMPONENT).vision

        dflash.record_capability(COMPONENT, vision=True, vision_detail="image passed")
        found = dflash.capability(COMPONENT)
        assert found.text and found.vision
        assert found.detail == "text passed"
        assert found.vision_detail == "image passed"

    def test_verifying_the_text_path_again_drops_the_image_result(self, install):
        install_dflash_runtime(install, text=True, vision=True)

        dflash.record_capability(COMPONENT, text=True, detail="text passed again")

        assert not dflash.capability(COMPONENT).vision

    def test_a_changed_executable_is_not_covered_by_the_old_proof(self, install):
        install_dflash_runtime(install, text=True, vision=True)
        server = dflash.executable(COMPONENT)
        assert dflash.capability(COMPONENT).text

        server.write_bytes(b"a different build entirely")

        assert not dflash.capability(COMPONENT).known

    def test_losing_it_leaves_the_ordinary_runtime_alone(self, install):
        import mc_llm_setup

        ordinary = install / mc_llm_setup.RUNTIME_DIRNAME
        ordinary.mkdir(parents=True)
        (ordinary / "llama-server").write_bytes(b"")
        install_dflash_runtime(install)

        dflash.remove(COMPONENT)

        assert dflash.installed() is None
        assert not dflash.capability(COMPONENT).known
        assert mc_llm_setup.detect() == ordinary / "llama-server"

    def test_the_registry_and_the_manifest_agree_on_the_family_name(self):
        """A registry entry naming a family nothing installs is a Lightning
        option that can never light up."""
        families = {build.family for build in dflash.builds()}

        for name in ("q6k", "q5km", "q4km"):
            found = managed.entry(f"qwen38-27b-abliterated-{name}")
            assert found.accelerators.dflash2.runtime_family in families
