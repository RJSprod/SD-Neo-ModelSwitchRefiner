"""Acceleration and VRAM priority, which are two settings and must stay two.

The obvious way to ship a "go faster" control is one switch meaning *use the
fast decoder and empty the card for it*. Fold the two together and a machine
that already has room has to evict a checkpoint to get a decoder it could have
had for free, and a Creative Writer on a 3090 empties a 5090 it was never going
to touch. So the first thing asserted below is a reclaimer that is never
called, and the second is a different card that is never touched.

The accelerator these tests were first written for -- DFlash2, a draft model on
a separately built llama.cpp -- is gone: it could not run on the machine it was
built for. What is left is the shape it argued for, which was never a fact
about DFlash2 in the first place, and multi-token prediction, which is in the
weights already.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import mc_broker
import mc_gguf
import mc_llm_accel as accel
import mc_llm_context as ctx
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

MTP_FLAGS = frozenset({"--spec-type", "--spec-draft-n-max", "--flash-attn"})
"""What a build with the speculative framework advertises."""

SPEC_TYPES = ("none", "draft-simple", "draft-eagle3", "draft-mtp", "draft-dflash",
              "draft-dspark", "ngram-simple", "ngram-map-k", "ngram-mod", "ngram-cache")
"""The speculative types llama.cpp b10621 lists, copied from a real usage line.

One of them is load-bearing and the rest are here because leaving them out
would make the list a restatement of the assertion rather than a sample of what
a build really prints."""

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


def registry_document(**mtp) -> dict:
    """One Qwen 3.8-shaped catalogue row, with whatever is being tested replaced."""
    heads = {"embedded": True, "draft_tokens": 3}
    heads.update(mtp)
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
        "accelerators": {"mtp": heads},
    }]}


@pytest.fixture
def registry(tmp_path, monkeypatch):
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(registry_document()), encoding="utf-8")
    monkeypatch.setattr(managed, "REGISTRY_PATH", path)
    managed._registry_cache = None
    return path


def install_bundle(root: Path, tmp_path: Path, *, projector=True, size_mb=4) -> Path:
    """A verified bundle on disk."""
    bundle = root / "models" / "managed" / "qwen38-test"
    bundle.mkdir(parents=True, exist_ok=True)
    hybrid_model(bundle / "model.gguf", size_mb=size_mb)
    artifacts = {"model": {"filename": "qwen38-test-Q4_K_M.gguf",
                           "stored_as": "model.gguf", "sha256": MODEL_SHA}}
    if projector:
        (bundle / "mmproj.gguf").write_bytes(b"\x00" * _MB)
        artifacts["projector"] = {"filename": "mmproj-qwen38-test-F16.gguf",
                                  "stored_as": "mmproj.gguf", "sha256": MMPROJ_SHA}
    (bundle / "installed.json").write_text(json.dumps({
        "schema": 1, "model_id": "qwen38-test", "registry_version": "test-1",
        "revision": "e" * 40, "profile": "qwen38-27b-prompt-author",
        "profile_version": managed_profiles.VERSION, "artifacts": artifacts,
        "installed_at": 0,
    }), encoding="utf-8")
    return bundle


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


def with_flags(monkeypatch, flags=MTP_FLAGS, types=SPEC_TYPES):
    """A fake ``llama-server --help``, in the shape a real one has.

    The *text* rather than a parsed flag set, because the difference between
    those two is the bug this helper exists to be able to reproduce: every
    build in the wild advertises ``--spec-type``, and which types it takes is
    an enumeration printed in its usage line. A test that faked only the flag
    set could not tell a build that accepts ``draft-mtp`` from one that has
    never heard of it, which is exactly the pair that has to be told apart.
    """
    lines = [f"  {flag} VALUE" for flag in sorted(flags)]
    if "--spec-type" in flags:
        lines.append("--spec-type " + ",".join(types))
        lines.append("       comma-separated list of types of speculative decoding to use")
    help_text = "\n".join(lines)
    monkeypatch.setattr(runtime, "_capability_text", lambda configuration=None: help_text)
    monkeypatch.setattr(runtime, "_flash_attention_takes_a_value",
                        lambda configuration=None: True)
    runtime._rejected.clear()
    return help_text


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
    ])
    def test_each_preset_is_exactly_two_settings(self, preset, axes):
        assert accel.preset_axes(preset) == axes
        assert accel.remember(preset=preset) == accel.Settings(*axes)

    def test_the_advanced_combination_no_preset_offers_is_accepted(self):
        """MTP with cooperative memory: the faster decoder without giving the
        language model priority over the image side. No preset names it, and
        the axes being two settings is what makes it reachable at all."""
        found = accel.remember(accelerator=accel.ACCEL_MTP,
                               memory_priority=accel.PRIORITY_COOPERATIVE)

        assert found.accelerator == accel.ACCEL_MTP
        assert found.cooperative
        assert found.preset == accel.PRESET_CUSTOM

    def test_an_axis_wins_over_the_preset_it_arrives_with(self):
        """Which is the direction a user moves in: choose Lightning, then open
        Advanced and turn the eviction back off."""
        found = accel.remember(preset=accel.PRESET_FAST,
                               memory_priority=accel.PRIORITY_COOPERATIVE)

        assert found.accelerator == accel.ACCEL_MTP
        assert found.memory_priority == accel.PRIORITY_COOPERATIVE

    def test_a_role_can_answer_differently_from_the_installation(self):
        """A 3090 Creative Writer and a 5090 Spatial Composer have different
        amounts of room, so section 15 asks for these to be role-aware."""
        accel.remember(preset=accel.PRESET_NORMAL)
        accel.remember(role=roles.CREATIVE, preset=accel.PRESET_FAST)

        assert accel.settings(roles.CREATIVE).preset == accel.PRESET_FAST
        assert accel.settings().preset == accel.PRESET_NORMAL
        assert accel.settings(roles.SPATIAL).preset == accel.PRESET_NORMAL

    def test_a_role_that_follows_the_installation_is_not_a_copy_of_it(self):
        accel.remember(preset=accel.PRESET_FAST)

        assert accel.follows_installation(roles.SPATIAL)
        assert accel.settings(roles.SPATIAL).preset == accel.PRESET_FAST

    def test_the_role_section_survives_being_read_back(self):
        """The preferences reader keeps only keys it knows, and the role
        overrides live under one it did not: they were written and dropped."""
        accel.remember(role=roles.CREATIVE, preset=accel.PRESET_FAST)

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
        a draft model, 8K of state, compute buffers and a projector."""
        high = managed.entry("qwen38-27b-abliterated-q6k")

        assert high.model.filename.endswith("-Q6_K.gguf")
        assert not any("Q8" in model.model.filename for model in managed.catalogue()
                       if model.family == "Qwen 3.8")

    def test_every_tier_declares_its_own_heads(self):
        for name in ("q6k", "q5km", "q4km"):
            found = managed.entry(f"qwen38-27b-abliterated-{name}").accelerators

            assert found.mtp is not None and found.mtp.embedded
            assert found.mtp.draft_tokens == 3

    def test_a_bundle_is_the_weights_and_the_projector(self):
        """And nothing else. An accelerator here describes the file already
        being downloaded rather than naming another one to fetch."""
        model = managed.entry("qwen38-27b-abliterated-q4km")

        assert [artifact.local_name for artifact in model.artifacts] == [
            managed.MODEL_FILENAME, managed.MMPROJ_FILENAME]

    def test_the_registry_never_names_a_local_path_of_its_own(self):
        """The publisher's filename builds the URL and is recorded; the file on
        disk is always one of the five names this module fixed."""
        for model in managed.catalogue():
            for artifact in model.artifacts:
                assert artifact.local_name in (managed.MODEL_FILENAME,
                                               managed.MMPROJ_FILENAME)

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
# LLM priority, and the card it is scoped to (test plan G and J)
# --------------------------------------------------------------------------- #


class TestLLMPriority:
    """The one door out of "an image residency is never demoted for the LLM".

    The default rule is right and stays: the image model is the workload and
    the language model is a helper writing a prompt for it. LLM priority is a
    user saying, on one card, that they would rather have the language model --
    and it is the whole of what Fast LLM's second half means, so every one of
    its edges is worth a test.
    """

    def _configured(self, install, tmp_path, monkeypatch, **changes):
        install_bundle(install, tmp_path)
        with_flags(monkeypatch)
        return configure(monkeypatch, install, tmp_path, **changes)

    def test_cooperative_memory_asks_for_nothing_at_all(
            self, install, tmp_path, monkeypatch, registry, image):
        """Not "little was released" -- *nothing was asked for*. A test that
        checked free VRAM afterwards would pass against an implementation that
        asked and was refused."""
        set_free(monkeypatch, 0.001)
        configuration = self._configured(install, tmp_path, monkeypatch,
                                         memory_priority=accel.PRIORITY_COOPERATIVE)
        image.holds = 20 * _GB

        runtime._make_room_for_the_llm(configuration)

        assert image.calls == []

    def test_a_placement_that_already_fits_reclaims_nothing(
            self, install, tmp_path, monkeypatch, registry, image):
        set_free(monkeypatch, 24)
        configuration = self._configured(install, tmp_path, monkeypatch,
                                         memory_priority=accel.PRIORITY_LLM)
        image.holds = 20 * _GB

        assert runtime._make_room_for_the_llm(configuration) == 0
        assert image.calls == []

    def test_it_asks_the_same_card_for_the_deficit(
            self, install, tmp_path, monkeypatch, registry, image):
        set_free(monkeypatch, 0.001)
        configuration = self._configured(install, tmp_path, monkeypatch,
                                         memory_priority=accel.PRIORITY_LLM)
        image.holds = 20 * _GB

        freed = runtime._make_room_for_the_llm(configuration)

        assert image.calls, "the deficit was never asked for"
        assert freed > 0

    def test_a_creative_writer_on_one_card_never_empties_the_other(
            self, install, tmp_path, monkeypatch, registry, image):
        """The image side is on the 5090 and the writer is on the 3090.
        Reclaiming the 5090 cannot free a byte of the 3090."""
        set_free(monkeypatch, 0.001)
        monkeypatch.setattr(mc_broker, "image_device_index", lambda: 1)
        configuration = self._configured(install, tmp_path, monkeypatch, gpu_index=0,
                                         memory_priority=accel.PRIORITY_LLM)
        image.holds = 30 * _GB

        assert runtime._make_room_for_the_llm(configuration) == 0
        assert image.calls == []

    def test_a_processor_placement_asks_for_nothing(
            self, install, tmp_path, monkeypatch, registry, image):
        set_free(monkeypatch, 0.001)
        configuration = self._configured(install, tmp_path, monkeypatch, mode="cpu",
                                         memory_priority=accel.PRIORITY_LLM)
        image.holds = 30 * _GB

        assert runtime._make_room_for_the_llm(configuration) == 0
        assert image.calls == []

    def test_the_broker_refuses_a_cross_card_release_on_its_own_terms(self, monkeypatch,
                                                                      image):
        """Asserted at the broker as well as through the placement, because
        this is the rule and the placement is only one of its callers."""
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
# What runs when nobody asked for anything in particular
# --------------------------------------------------------------------------- #


class TestAuto:
    """Auto is a policy rather than a request, so it never refuses.

    It uses what it can prove is available and names the mechanism it landed
    on, which is the whole difference between it and asking for one by name.
    """

    def test_it_uses_the_backbones_own_heads_when_the_build_takes_them(
            self, install, tmp_path, monkeypatch, registry, image):
        install_bundle(install, tmp_path)
        with_flags(monkeypatch)
        set_free(monkeypatch, 24)
        configuration = configure(monkeypatch, install, tmp_path)

        plan = runtime.accelerator_plan(configuration)

        assert plan.accelerator == accel.ACCEL_MTP
        assert not plan.refused
        # llama.cpp's own spelling. ``mtp`` is not one of the types it takes,
        # and a build handed it prints "unknown speculative type" and exits
        # before it loads a tensor -- which arrives as "this backbone would not
        # start" and rolls the user back to whatever they were on.
        assert plan.flags == ("--spec-type", "draft-mtp", "--spec-draft-n-max", "3")

    def test_a_build_that_lists_the_option_but_not_the_type_gets_no_flags(
            self, install, tmp_path, monkeypatch, registry, image):
        """Advertising an option is not accepting a value for it.

        Every llama.cpp since the speculative framework landed advertises
        ``--spec-type``; which types it takes is an enumeration that grew
        release by release. Gating on the option alone is how a start dies at
        argument parsing, and the model switch that asked for it rolls back.
        """
        install_bundle(install, tmp_path)
        with_flags(monkeypatch, types=("none", "draft-simple", "ngram-cache"))
        set_free(monkeypatch, 24)
        configuration = configure(monkeypatch, install, tmp_path)

        plan = runtime.accelerator_plan(configuration)

        assert runtime.runtime_supports("--spec-type", configuration)
        assert not runtime.runtime_accepts("--spec-type", "draft-mtp", configuration)
        assert plan.accelerator == accel.ACCEL_NONE
        assert plan.flags == ()
        assert any("Update the runtime" in note for note in plan.notes)

    def test_a_value_the_build_refused_at_startup_is_not_asked_for_again(
            self, install, tmp_path, monkeypatch, registry, image):
        """The help text can be wrong by omission, and one failed start is the
        only way to find out. Paying for it twice is not."""
        install_bundle(install, tmp_path)
        with_flags(monkeypatch)
        set_free(monkeypatch, 24)
        configuration = configure(monkeypatch, install, tmp_path)
        assert runtime.accelerator_plan(configuration).accelerator == accel.ACCEL_MTP

        runtime.note_rejected_value("--spec-type", "draft-mtp", configuration)

        assert not runtime.runtime_accepts("--spec-type", "draft-mtp", configuration)
        assert runtime.accelerator_plan(configuration).accelerator == accel.ACCEL_NONE
        assert ("--spec-type", "draft-mtp") in runtime.rejected_values(configuration)

    def test_it_steps_down_to_ordinary_decoding_on_an_older_runtime(
            self, install, tmp_path, monkeypatch, registry, image):
        install_bundle(install, tmp_path)
        with_flags(monkeypatch, frozenset())
        set_free(monkeypatch, 24)
        configuration = configure(monkeypatch, install, tmp_path)

        plan = runtime.accelerator_plan(configuration)

        assert plan.accelerator == accel.ACCEL_NONE
        assert not plan.refused
        assert plan.flags == ()

    def test_it_never_refuses(self, install, tmp_path, monkeypatch, registry, image):
        install_bundle(install, tmp_path)
        with_flags(monkeypatch, frozenset())
        set_free(monkeypatch, 0.001)
        configuration = configure(monkeypatch, install, tmp_path)

        assert not runtime.accelerator_plan(configuration).refused

    def test_a_backbone_with_no_heads_says_so_rather_than_refusing(
            self, install, tmp_path, monkeypatch, registry, image):
        """Fast LLM is "MTP when supported, otherwise ordinary decoding" -- with
        the memory priority the preset also carries."""
        install_bundle(install, tmp_path)
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

    def test_none_asks_for_nothing(self, install, tmp_path, monkeypatch, registry, image):
        install_bundle(install, tmp_path)
        with_flags(monkeypatch)
        set_free(monkeypatch, 24)
        configuration = configure(monkeypatch, install, tmp_path,
                                  accelerator=accel.ACCEL_NONE)

        plan = runtime.accelerator_plan(configuration)

        assert plan.accelerator == accel.ACCEL_NONE
        assert plan.flags == ()

    def test_a_manual_gguf_gets_the_decoding_it_has_always_had(
            self, install, tmp_path, monkeypatch, registry, image):
        """An accelerator claim is a statement about a specific pinned file."""
        from dataclasses import replace

        install_bundle(install, tmp_path)
        with_flags(monkeypatch)
        set_free(monkeypatch, 24)
        configuration = replace(configure(monkeypatch, install, tmp_path),
                                managed_id="", source=managed.SOURCE_MANUAL)
        monkeypatch.setattr(runtime, "config", lambda role="": configuration)

        assert runtime.accelerator_plan(configuration).accelerator == accel.ACCEL_NONE


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
                                  accelerator=accel.ACCEL_NONE)

        assert runtime._identity(first) != runtime._identity(second)

    def test_changing_the_memory_priority_invalidates_a_warm_server(
            self, install, tmp_path, monkeypatch, registry):
        install_bundle(install, tmp_path)
        first = self._configured(install, tmp_path, monkeypatch,
                                 memory_priority=accel.PRIORITY_COOPERATIVE)
        second = self._configured(install, tmp_path, monkeypatch,
                                  memory_priority=accel.PRIORITY_LLM)

        assert runtime._identity(first) != runtime._identity(second)

    def test_the_plan_identity_distinguishes_the_mechanism_and_the_binary(self):
        """Both are start-time facts: a server started with ``--spec-type``
        cannot be told to stop, and a different binary is a different program."""
        from dataclasses import replace

        ordinary = accel.Plan()
        accelerated = accel.Plan(accelerator=accel.ACCEL_MTP)
        elsewhere = replace(accelerated, runtime=Path("another/llama-server"))

        assert len({ordinary.identity, accelerated.identity, elsewhere.identity}) == 3

    def test_a_signature_tells_two_runtimes_apart(self, install, tmp_path, monkeypatch,
                                                  registry):
        install_bundle(install, tmp_path)
        configuration = configure(monkeypatch, install, tmp_path)
        placement = ctx.Placement()
        accelerated = accel.Plan(accelerator=accel.ACCEL_MTP,
                                 runtime=Path("another/llama-server"))

        assert (runtime._signature_of(configuration, None, placement)
                != runtime._signature_of(configuration, None, placement, accelerated))


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
        plan = accel.Plan(accelerator=accel.ACCEL_MTP,
                          flags=("--spec-type", "draft-mtp", "--flash-attn", "on"))

        flags = runtime._launch_flags(configuration, ctx.Placement(), plan)

        assert flags.count("--flash-attn") == 1
        assert "--spec-type" in flags


# --------------------------------------------------------------------------- #
# Measured speed, kept apart (design intent section 17)
# --------------------------------------------------------------------------- #


class TestMeasurement:
    def test_an_accelerated_rate_is_not_averaged_into_an_ordinary_one(self):
        resident = ctx.Placement()

        ordinary = runtime.measurement_token(resident, accel.ACCEL_NONE, 0)
        speculative = runtime.measurement_token(resident, accel.ACCEL_MTP, 0)

        assert ordinary != speculative
        assert runtime.speed_key(runtime.WRITE_RATE, "qwen38", ordinary) != \
            runtime.speed_key(runtime.WRITE_RATE, "qwen38", speculative)

    def test_the_physical_card_is_part_of_the_key(self):
        assert (runtime.measurement_token(ctx.Placement(), accel.ACCEL_MTP, 0)
                != runtime.measurement_token(ctx.Placement(), accel.ACCEL_MTP, 1))

    def test_an_ordinary_run_keeps_the_key_it_always_had(self):
        """Every rate this machine has already measured has to keep answering."""
        assert runtime.measurement_token(ctx.Placement(), accel.ACCEL_NONE, None) == "gpu"

    def test_the_token_reads_back_as_a_sentence(self):
        said = runtime.describe_placement_token(
            runtime.measurement_token(ctx.Placement(), accel.ACCEL_MTP, 0))

        assert "MTP" in said
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
# A flag llama.cpp will not take (27 August 2026)
# --------------------------------------------------------------------------- #


class TestAnOptionTheBuildRefuses:
    """The model is not what pays for a flag this extension got wrong.

    Reported from a real machine: ``--spec-type mtp`` on llama.cpp b10621,
    whose speculative types are all ``draft-``-prefixed. The server printed
    ``unknown speculative type`` and exited before it loaded a tensor, the
    switch that asked for it rolled back, and what the user read was "Qwen 3.8
    27B Abliterated — Medium was downloaded but would not start".
    """

    REAL = ('error while handling argument "--spec-type": unknown speculative type: mtp\n'
            '\nusage:\n--spec-type none,draft-simple,draft-eagle3,draft-mtp,draft-dflash\n')

    def test_the_refusal_is_read_as_a_flag_problem_and_not_a_model_problem(self):
        failure = runtime.read_failure(self.REAL)

        assert failure.bad_argument == "--spec-type"
        assert failure.bad_value == "mtp"
        assert not failure.out_of_memory
        assert "not anything about the model" in failure.text

    def test_a_flag_the_launcher_must_pass_is_not_treated_this_way(self):
        """``--device`` is not optional, a refusal of it is a real
        misconfiguration, and it already has a diagnosis worth far more."""
        failure = runtime.read_failure(
            'error while handling argument "--device": invalid device: CUDA0\n')

        assert failure.bad_argument == ""
        assert failure.out_of_memory

    @pytest.fixture
    def quiet(self, monkeypatch):
        monkeypatch.setattr(runtime, "_prime_prompt_cache", lambda client: None)
        monkeypatch.setattr(runtime, "RESIDENCY_SETTLE_SECONDS", 0.0)
        for family in (mc_broker.FAMILY_IMAGE, mc_broker.FAMILY_LLM):
            mc_broker.unregister_reclaimer(family)
        yield runtime.Runtime()

    def test_the_start_is_retried_without_the_accelerator(self, install, tmp_path,
                                                          monkeypatch, registry, quiet):
        install_bundle(install, tmp_path)
        with_flags(monkeypatch)
        set_free(monkeypatch, 24)
        configure(monkeypatch, install, tmp_path)
        attempts: list = []

        class Reached(RuntimeError):
            pass

        def launch(configuration, placement, projector=None, plan=None):
            attempts.append(plan)
            if len(attempts) == 1:
                raise runtime._StartFailed(
                    "llama-server would not take --spec-type: unknown speculative type: mtp",
                    bad_argument="--spec-type", bad_value="mtp")
            raise Reached()

        monkeypatch.setattr(quiet, "_launch", launch)
        with pytest.raises(Reached):
            quiet.client()

        assert len(attempts) == 2, "the start was not retried"
        assert attempts[0].accelerator == accel.ACCEL_MTP
        assert attempts[1].accelerator == accel.ACCEL_NONE
        assert attempts[1].flags == ()
        assert any("MTP was not used" in note for note in attempts[1].notes)

    def test_the_value_is_not_asked_for_again_afterwards(self, install, tmp_path,
                                                         monkeypatch, registry, quiet):
        install_bundle(install, tmp_path)
        with_flags(monkeypatch)
        set_free(monkeypatch, 24)
        configuration = configure(monkeypatch, install, tmp_path)

        def launch(configuration, placement, projector=None, plan=None):
            if plan.flags:
                raise runtime._StartFailed("refused", bad_argument="--spec-type",
                                           bad_value="draft-mtp")
            raise RuntimeError("far enough")

        monkeypatch.setattr(quiet, "_launch", launch)
        with pytest.raises(RuntimeError, match="far enough"):
            quiet.client()

        assert ("--spec-type", "draft-mtp") in runtime.rejected_values(configuration)
        assert runtime.accelerator_plan(configuration).accelerator == accel.ACCEL_NONE

    def test_it_is_retried_once_and_not_for_ever(self, install, tmp_path, monkeypatch,
                                                 registry, quiet):
        """A second argument error is a real failure. Retrying past it would
        hide whatever is actually wrong behind an endless loop."""
        install_bundle(install, tmp_path)
        with_flags(monkeypatch)
        set_free(monkeypatch, 24)
        configure(monkeypatch, install, tmp_path)
        attempts: list = []

        def launch(configuration, placement, projector=None, plan=None):
            attempts.append(plan)
            raise runtime._StartFailed("refused", bad_argument="--swa-full")

        monkeypatch.setattr(quiet, "_launch", launch)
        with pytest.raises(RuntimeError, match="refused"):
            quiet.client()

        assert len(attempts) == 2
