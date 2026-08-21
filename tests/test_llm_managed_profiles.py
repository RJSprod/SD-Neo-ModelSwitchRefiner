"""Hidden quality profiles: applied to a managed backbone, invisible in Setup.

Two properties, and they pull against each other, which is why they are tested
together. A managed backbone has to *run at* the values chosen for it -- an
8192 context, a q8_0 cache, its own chat template, its family's top-k -- and a
user has to never *see* any of that, because the whole point of a curated list
is that choosing a model is the entire decision.

The third property is the one that protects everybody who is not using the
catalogue: a manual GGUF must behave exactly as it did before this existed. The
Settings page stays authoritative for it, no sampler fields are added to its
requests, and its llama-server command line is unchanged.
"""

from __future__ import annotations

import json

import pytest

import mc_llm_context as ctx
import mc_llm_managed_models as managed
import mc_llm_paths
import mc_llm_runtime as runtime
from prompt_master.models import managed_profiles
from test_llm_managed_switch import MMPROJ_SHA, MODEL_SHA, install_bundle, write_state


@pytest.fixture(autouse=True)
def root(tmp_path, monkeypatch, host):
    install = tmp_path / "install"
    (install / "data").mkdir(parents=True)
    monkeypatch.setattr(mc_llm_paths, "data_root", lambda: install)
    managed._registry_cache = None
    yield install
    managed._registry_cache = None


@pytest.fixture
def registry(tmp_path, monkeypatch):
    document = {"version": 1, "registry_version": "test-1", "models": [{
        "id": "gemma-test", "label": "Gemma Test", "role": "Recommended",
        "group": "Recommended", "family": "Gemma 4",
        "profile": "gemma4-12b-qat-balanced", "multimodal": True,
        "source_url": "https://huggingface.co/example/test",
        "repo_id": "example/test", "revision": "main",
        "model": {"filename": "gemma-test-Q4_K_M.gguf", "sha256": MODEL_SHA,
                  "bytes": None, "display_size": "~7.4 GB"},
        "projector": {"filename": "mmproj-gemma-test.gguf", "sha256": MMPROJ_SHA,
                      "bytes": None, "display_size": "175 MB"},
    }, {
        "id": "qwen-test", "label": "Qwen Test", "role": "Modern alternative",
        "group": "Alternatives", "family": "Qwen 3.5",
        "profile": "qwen35-9b-aggressive", "multimodal": True,
        "source_url": "https://huggingface.co/example/test",
        "repo_id": "example/test", "revision": "main",
        "model": {"filename": "qwen-test-Q6_K.gguf", "sha256": MODEL_SHA,
                  "bytes": None, "display_size": "~7.4 GB"},
        "projector": {"filename": "mmproj-qwen-test.gguf", "sha256": MMPROJ_SHA,
                      "bytes": None, "display_size": "922 MB"},
    }]}
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.setattr(managed, "REGISTRY_PATH", path)
    managed._registry_cache = None
    return document


def select(root, identifier, profile):
    """Put a bundle on disk and record it as selected, without a switch."""
    install_bundle(root, identifier, profile)
    write_state(root, runtime="runtime/llama-server",
                model=f"models/managed/{identifier}/model.gguf",
                mmproj=f"models/managed/{identifier}/mmproj.gguf",
                source="managed", managed_model_id=identifier,
                managed_profile=profile, managed_profile_version=managed_profiles.VERSION)


class TestTheProfileReachesTheRuntime:
    def test_a_managed_backbone_runs_at_its_own_context_and_cache(self, root, registry):
        select(root, "gemma-test", "gemma4-12b-qat-balanced")

        configuration = runtime.config()

        assert configuration.context_size == 8192
        assert (configuration.kv_type_k, configuration.kv_type_v) == ("q8_0", "q8_0")

    def test_the_profile_wins_over_the_settings_page(self, root, registry, host):
        """Somebody who chose a backbone chose its context size with it. Leaving
        the Settings page authoritative would run a curated model at whatever
        the previous one happened to need."""
        host.shared.opts.model_chain_llm_context_size = 65536
        host.shared.opts.model_chain_llm_kv_type_k = "f16"
        host.shared.opts.model_chain_llm_kv_type_v = "f16"
        select(root, "gemma-test", "gemma4-12b-qat-balanced")

        configuration = runtime.config()

        assert configuration.context_size == 8192
        assert configuration.kv_type_k == "q8_0"

    def test_a_managed_backbone_never_grows_into_free_vram(self, root, registry, host):
        """Automatic sizing spends whatever is free on context. 8192 is a
        decision about this workload, not a floor to grow from."""
        host.shared.opts.model_chain_llm_context_mode = "Automatic — fill what is free"
        select(root, "gemma-test", "gemma4-12b-qat-balanced")

        assert runtime.config().context_mode == "fixed"

    def test_the_two_families_get_their_own_samplers(self, root, registry):
        select(root, "gemma-test", "gemma4-12b-qat-balanced")
        gemma = managed_profiles.sampler_arguments(runtime.config().profile)
        select(root, "qwen-test", "qwen35-9b-aggressive")
        qwen = managed_profiles.sampler_arguments(runtime.config().profile)

        assert gemma == {"top_k": 64, "min_p": 0.05, "repeat_penalty": 1.10}
        assert qwen == {"top_k": 20, "min_p": 0.00, "repeat_penalty": 1.00}

    def test_the_chat_template_and_cache_reach_the_command_line(self, root, registry):
        """``--jinja`` and the cache types are start-time arguments. A profile
        that only lived in a dataclass would change nothing about the server."""
        select(root, "gemma-test", "gemma4-12b-qat-balanced")
        configuration = runtime.config()
        placement = ctx.Placement(context=8192, kv_type_k="q8_0", kv_type_v="q8_0")

        arguments = runtime._profile_arguments(configuration, placement)

        assert arguments == {"cache_type_k": "q8_0", "cache_type_v": "q8_0", "jinja": True}

    def test_changing_backbone_restarts_the_server_even_at_the_same_context(
            self, root, registry):
        """Two profiles can agree about context and cache and still want
        different templates and samplers, and a running server cannot be told
        about either."""
        select(root, "gemma-test", "gemma4-12b-qat-balanced")
        before = runtime._identity(runtime.config())
        select(root, "qwen-test", "qwen35-9b-aggressive")
        after = runtime._identity(runtime.config())

        assert before != after

    def test_a_profile_this_build_has_never_heard_of_falls_back_to_manual(self, root,
                                                                          registry):
        """A state file from a newer extension. Running somebody's weights at
        settings invented for a different model is worse than running them at
        the installation's own."""
        select(root, "gemma-test", "gemma4-12b-qat-balanced")
        write_state(root, runtime="runtime/llama-server",
                    model="models/managed/gemma-test/model.gguf",
                    source="managed", managed_model_id="gemma-test",
                    managed_profile="a-profile-from-the-future")

        assert runtime.config().profile is None


class TestAManualInstallIsUntouched:
    def test_it_keeps_the_settings_pages_context_and_cache(self, root, host, registry,
                                                           tmp_path):
        host.shared.opts.model_chain_llm_context_size = 16384
        host.shared.opts.model_chain_llm_kv_type_k = "q8_0"
        mine = tmp_path / "mine.gguf"
        mine.write_bytes(b"my own weights")
        write_state(root, runtime="runtime/llama-server", model=str(mine))

        configuration = runtime.config()

        assert configuration.profile is None
        assert configuration.context_size == 16384
        assert configuration.kv_type_k == "q8_0"

    def test_its_command_line_gains_nothing(self, root, tmp_path):
        """No ``--jinja``, no cache-type flags: byte for byte the command this
        extension has always started."""
        mine = tmp_path / "mine.gguf"
        mine.write_bytes(b"my own weights")
        write_state(root, runtime="runtime/llama-server", model=str(mine))

        assert runtime._profile_arguments(runtime.config(), ctx.Placement()) == {}

    def test_its_requests_gain_no_sampler_fields(self):
        from prompt_master.inference.llama_client import LlamaClient

        assert LlamaClient("http://127.0.0.1:1", "key").sampling == {}
        assert managed_profiles.sampler_arguments(None) == {}


class TestNothingLeaksIntoTheUI:
    def test_the_catalogue_lines_name_no_low_level_setting(self, registry):
        """Section 10: do NOT show temperature, top_p, top_k, min_p,
        presence/repeat penalties, KV cache type, or template flags."""
        forbidden = ("temperature", "top_p", "top-p", "top_k", "top-k", "min_p", "min-p",
                     "presence", "repeat", "penalt", "q8_0", "f16", "jinja", "template",
                     "kv ")
        for model in managed.catalogue():
            text = f"{model.label} {model.describe()} {model.role} {model.group}".casefold()
            assert not any(word in text for word in forbidden), text

    def test_setup_offers_no_control_for_any_profile_value(self, host, registry):
        """Checked against the panel Setup actually builds, so a control added
        later fails here rather than being noticed by a user."""
        import mc_llm_studio

        built = mc_llm_studio._setup_panel()
        labels = " ".join(str(getattr(component, "label", "") or "")
                          for component in built.values()).casefold()

        assert "temperature" not in labels
        assert "top" not in labels
        assert "penalty" not in labels
        assert "cache" not in labels

    def test_the_only_sampler_fields_a_profile_can_set_are_whitelisted(self):
        """A profile is checked-in data that ends up in a JSON request body.
        "Whatever the dict contained" is not a thing to put in one."""
        from prompt_master.inference.llama_client import LlamaClient

        rogue = managed_profiles.ManagedProfile(
            "rogue", sampling={"top_k": 20, "model": "something-else",
                               "messages": [], "temperature": 2.0, "stream": False})

        assert managed_profiles.sampler_arguments(rogue) == {"top_k": 20}
        assert LlamaClient("http://127.0.0.1:1", "key", rogue.sampling).sampling == {"top_k": 20}

    def test_temperature_and_top_p_are_left_to_creative_modes_own_curve(self):
        """Creativity 0-10 keeps one meaning across every backbone. A profile
        that set them would be a checked-in file overriding the user's slider."""
        assert "temperature" not in managed_profiles.SAMPLER_FIELDS
        assert "top_p" not in managed_profiles.SAMPLER_FIELDS
        for profile in managed_profiles.PROFILES.values():
            assert "temperature" not in profile.sampling
            assert "top_p" not in profile.sampling


class TestTheProfilesThemselves:
    def test_every_backbone_gets_the_same_short_context(self):
        """8192 is a decision about this application's workload -- a system
        instruction, four reference captions and a brief -- not about what each
        model advertises."""
        for profile in managed_profiles.PROFILES.values():
            assert profile.context == managed_profiles.CONTEXT_SIZE == 8192

    def test_thinking_is_off_everywhere(self):
        """This application reads the model's output as a finished prompt. A
        chain of thought in front of it is text the parser has to strip."""
        for profile in managed_profiles.PROFILES.values():
            assert profile.thinking is False

    def test_no_profile_makes_a_claim_about_somebody_elses_hardware(self):
        """Section 5.8. Profiles control model behaviour; the broker decides
        where the model fits."""
        fields = set(managed_profiles.ManagedProfile.__dataclass_fields__)

        assert not fields & {"gpu_index", "gpu_layers", "device", "mode", "offload"}

    def test_the_large_baseline_is_left_exactly_as_it_was(self):
        """It exists to provide the known behaviour. Every value that differs
        from that behaviour makes it worse at its one job."""
        baseline = managed_profiles.profile("gemma4-26b-a4b-balanced")

        assert (baseline.kv_type_k, baseline.kv_type_v) == ("f16", "f16")
        assert baseline.sampling == {}

    def test_the_quant_tiers_each_get_a_profile_of_their_own(self):
        """Three quantisations of one backbone, three profile ids. Sharing one
        would make the cache choice below impossible to express."""
        for name in ("gemma4-26b-a4b-q4kp", "gemma4-26b-a4b-q3kp",
                     "gemma4-26b-a4b-q2kp"):
            found = managed_profiles.profile(name)
            assert found is not None
            assert found.profile_id == name

    def test_the_quality_tier_keeps_the_baselines_full_precision_cache(self):
        """Section 5: the quality tier exists to be as close to the known Q4
        behaviour as this file can make it, and the cache is half of that."""
        found = managed_profiles.profile("gemma4-26b-a4b-q4kp")
        baseline = managed_profiles.profile("gemma4-26b-a4b-balanced")

        assert (found.kv_type_k, found.kv_type_v) == (baseline.kv_type_k,
                                                      baseline.kv_type_v) == ("f16", "f16")

    def test_the_smaller_tiers_buy_their_cache_back_with_q8_0(self):
        for name in ("gemma4-26b-a4b-q3kp", "gemma4-26b-a4b-q2kp"):
            found = managed_profiles.profile(name)
            assert (found.kv_type_k, found.kv_type_v) == ("q8_0", "q8_0")

    def test_a_different_quant_does_not_move_the_samplers(self):
        """Section 5: "Do not change temperature/top_p because a different quant
        was selected." Nothing this file can set would only move those two, so
        the rule is kept by setting nothing at all."""
        for name in ("gemma4-26b-a4b-q4kp", "gemma4-26b-a4b-q3kp",
                     "gemma4-26b-a4b-q2kp"):
            assert managed_profiles.profile(name).sampling == {}
            assert managed_profiles.sampler_arguments(managed_profiles.profile(name)) == {}

    def test_the_quant_tiers_keep_the_family_template(self):
        for name in ("gemma4-26b-a4b-q4kp", "gemma4-26b-a4b-q3kp",
                     "gemma4-26b-a4b-q2kp"):
            found = managed_profiles.profile(name)
            assert found.jinja is True
            assert found.thinking is False
            assert found.context == 8192

    def test_an_unknown_profile_id_is_none_rather_than_a_default(self):
        assert managed_profiles.profile("nothing-like-this") is None
        assert managed_profiles.profile("") is None

    def test_a_caller_cannot_edit_a_profile_by_editing_what_it_was_given(self):
        found = managed_profiles.profile("gemma4-12b-qat-balanced")

        managed_profiles.sampler_arguments(found)["top_k"] = 1

        assert managed_profiles.profile("gemma4-12b-qat-balanced").sampling["top_k"] == 64
