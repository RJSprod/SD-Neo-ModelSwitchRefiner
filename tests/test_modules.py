"""Stage 2 VAE / text encoder selection.

Flux-family and Krea 2 checkpoints keep their VAE and text encoder in separate
files, selected in the host as "additional modules". Without a Stage 2
selection those files are inherited from Stage 1, which is wrong for any
cross-architecture chain -- SDXL's encoder stack cannot serve Flux.2 Klein.

The host folds the module list into ``forge_loading_parameters``, so a
checkpoint paired with its own encoders is a distinct cache key, and both
pairings stay resident independently.
"""

from __future__ import annotations

import types

import pytest

import mc_infotext
import mc_memory

FLUX_VAE = "/models/VAE/flux2_vae.safetensors"
QWEN_TE = "/models/text_encoder/qwen3_8b.safetensors"
SDXL_VAE = "/models/VAE/sdxl_vae.safetensors"


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #


class TestResolveModules:
    def test_sentinel_means_inherit(self, host):
        assert mc_memory.resolve_modules([mc_memory.INHERIT_MODULES]) is None

    def test_sentinel_wins_even_when_mixed_with_files(self, host):
        """Matches the host: the sentinel's presence is what is checked."""
        assert mc_memory.resolve_modules([mc_memory.INHERIT_MODULES, "flux2_vae.safetensors"]) is None

    def test_none_means_inherit(self, host):
        assert mc_memory.resolve_modules(None) is None

    def test_empty_list_means_built_in_not_inherit(self, host):
        """An empty selection is meaningful: use the checkpoint's own encoders."""
        assert mc_memory.resolve_modules([]) == []

    def test_names_resolve_to_full_paths(self, host):
        assert mc_memory.resolve_modules(["flux2_vae.safetensors"]) == [FLUX_VAE]

    def test_full_paths_are_accepted_too(self, host):
        assert mc_memory.resolve_modules([FLUX_VAE]) == [FLUX_VAE]

    def test_result_is_sorted_to_match_the_host(self, host):
        """main_entry.modules_change sorts, so an unsorted selection would
        never compare equal to what the host actually stores."""
        forward = mc_memory.resolve_modules(["qwen3_8b.safetensors", "flux2_vae.safetensors"])
        backward = mc_memory.resolve_modules(["flux2_vae.safetensors", "qwen3_8b.safetensors"])
        assert forward == backward == sorted([FLUX_VAE, QWEN_TE])

    def test_unknown_module_is_dropped_rather_than_fatal(self, host):
        assert mc_memory.resolve_modules(["nonexistent.safetensors"]) == []

    def test_a_single_string_is_accepted(self, host):
        assert mc_memory.resolve_modules("flux2_vae.safetensors") == [FLUX_VAE]


class TestFootprint:
    def test_selection_sizes_the_named_modules(self, host, tmp_path, monkeypatch):
        """A Flux.2 text encoder is several GB; ignoring it would badly
        under-count the Stage 2 footprint."""
        vae = tmp_path / "vae.safetensors"
        vae.write_bytes(b"\0" * 2048)
        encoder = tmp_path / "te.safetensors"
        encoder.write_bytes(b"\0" * 4096)

        monkeypatch.setattr(
            host.sd_models, "get_closet_checkpoint_match", lambda name: None
        )
        monkeypatch.setattr(
            mc_memory, "resolve_modules", lambda mods: [str(vae), str(encoder)]
        )

        assert mc_memory.file_size_bytes("model.safetensors", ["vae", "te"]) == 6144

    def test_inheriting_falls_back_to_the_current_selection(self, host, tmp_path, monkeypatch):
        current = tmp_path / "current.safetensors"
        current.write_bytes(b"\0" * 1024)
        host.shared.opts.forge_additional_modules = [str(current)]

        monkeypatch.setattr(host.sd_models, "get_closet_checkpoint_match", lambda name: None)

        assert mc_memory.file_size_bytes("model.safetensors", None) == 1024


# --------------------------------------------------------------------------- #
# Transitions
# --------------------------------------------------------------------------- #


@pytest.fixture
def residency(host, monkeypatch):
    """Fakes the host's single-model slot, tracking checkpoint and modules."""
    state = types.SimpleNamespace(reloads=0, modules=[], checkpoint="A")

    class FakeModel:
        def __init__(self, name, modules):
            self.name = name
            self.modules = list(modules)
            self.sd_checkpoint_info = types.SimpleNamespace(
                filename=f"/models/{name}", name_for_extra=name, sha256=f"sha-{name}"
            )

    model_data = host.sd_models.model_data
    model_data.sd_model = FakeModel("A", [])
    model_data.forge_loading_parameters = {"checkpoint_info": "A", "modules": []}
    model_data.forge_hash = str(model_data.forge_loading_parameters)
    model_data.set_sd_model = lambda v: setattr(model_data, "sd_model", v)

    monkeypatch.setattr(
        host.sd_models, "get_closet_checkpoint_match",
        lambda name: types.SimpleNamespace(
            filename=f"/models/{name}", name_for_extra=name, sha256=f"sha-{name}"
        ),
    )

    import modules_forge.main_entry as main_entry

    def checkpoint_change(name, preset, save=True, refresh=True):
        state.checkpoint = name
        return True

    def modules_change(values, preset, save=True, refresh=True):
        resolved = sorted(values)
        if resolved == host.shared.opts.forge_additional_modules:
            return False
        host.shared.opts.forge_additional_modules = resolved
        state.modules = resolved
        return True

    def refresh_model_loading_parameters(**kwargs):
        model_data.forge_loading_parameters = {
            "checkpoint_info": state.checkpoint,
            "modules": list(host.shared.opts.forge_additional_modules),
        }

    def forge_model_reload():
        state.reloads += 1
        key = str(model_data.forge_loading_parameters)
        model_data.sd_model = FakeModel(
            model_data.forge_loading_parameters["checkpoint_info"],
            model_data.forge_loading_parameters["modules"],
        )
        model_data.forge_hash = key
        return model_data.sd_model, True

    monkeypatch.setattr(main_entry, "checkpoint_change", checkpoint_change)
    monkeypatch.setattr(main_entry, "modules_change", modules_change)
    monkeypatch.setattr(main_entry, "refresh_model_loading_parameters", refresh_model_loading_parameters)
    monkeypatch.setattr(host.sd_models, "forge_model_reload", forge_model_reload)

    monkeypatch.setattr(mc_memory, "loaded_size_bytes", lambda model: 7 * 1024**3)
    monkeypatch.setattr(mc_memory, "file_size_bytes", lambda name, mods=None: 7 * 1024**3)
    monkeypatch.setattr(mc_memory, "free_ram_bytes", lambda: 128 * 1024**3)
    monkeypatch.setattr(mc_memory, "cache_budget_bytes", lambda: 64 * 1024**3)

    mc_memory._cache = mc_memory._Cache()
    mc_memory._pending_restore = None
    yield types.SimpleNamespace(state=state, model_data=model_data)
    mc_memory._cache = mc_memory._Cache()
    mc_memory._pending_restore = None


class TestEnsureResident:
    def test_stage_2_modules_are_applied_with_the_checkpoint(self, residency, host):
        mc_memory.ensure_resident("B", ["flux2_vae.safetensors", "qwen3_8b.safetensors"])

        assert host.shared.opts.forge_additional_modules == sorted([FLUX_VAE, QWEN_TE])
        assert residency.model_data.sd_model.name == "B"

    def test_inheriting_leaves_stage_1_modules_untouched(self, residency, host):
        host.shared.opts.forge_additional_modules = [SDXL_VAE]

        mc_memory.ensure_resident("B", [mc_memory.INHERIT_MODULES])

        assert host.shared.opts.forge_additional_modules == [SDXL_VAE]

    def test_empty_selection_clears_the_modules(self, residency, host):
        host.shared.opts.forge_additional_modules = [SDXL_VAE]

        mc_memory.ensure_resident("B", [])

        assert host.shared.opts.forge_additional_modules == []

    def test_changing_only_the_modules_still_counts_as_a_change(self, residency, host):
        """Same checkpoint, different encoders, is a different model to load."""
        mc_memory.ensure_resident("B", ["flux2_vae.safetensors"])
        reloads = residency.state.reloads

        result = mc_memory.ensure_resident("B", ["qwen3_8b.safetensors"])

        assert result != "unchanged"
        assert residency.state.reloads > reloads

    def test_the_same_pair_is_unchanged(self, residency, host):
        mc_memory.ensure_resident("B", ["flux2_vae.safetensors"])
        reloads = residency.state.reloads

        assert mc_memory.ensure_resident("B", ["flux2_vae.safetensors"]) == "unchanged"
        assert residency.state.reloads == reloads


class TestResidencyWithModules:
    def test_each_checkpoint_keeps_its_own_encoders_in_memory(self, residency, host):
        """The user's requirement: Stage 2's VAE/TE object stays resident."""
        host.shared.opts.forge_additional_modules = [SDXL_VAE]
        residency.model_data.forge_loading_parameters = {
            "checkpoint_info": "A", "modules": [SDXL_VAE]
        }
        residency.model_data.forge_hash = str(residency.model_data.forge_loading_parameters)
        model_a = residency.model_data.sd_model

        mc_memory.ensure_resident("B", ["flux2_vae.safetensors", "qwen3_8b.safetensors"])
        model_b = residency.model_data.sd_model
        assert model_b.modules == sorted([FLUX_VAE, QWEN_TE])

        # Back to A with its own VAE: warm, and A's modules come back too.
        assert mc_memory.ensure_resident("A", [SDXL_VAE]) == "warm"
        assert residency.model_data.sd_model is model_a
        assert host.shared.opts.forge_additional_modules == [SDXL_VAE]

        # Forward to B again: still warm, encoders intact, no disk read.
        reloads = residency.state.reloads
        assert mc_memory.ensure_resident("B", ["flux2_vae.safetensors", "qwen3_8b.safetensors"]) == "warm"
        assert residency.model_data.sd_model is model_b
        assert residency.state.reloads == reloads

    def test_a_checkpoint_with_two_encoder_sets_is_two_cache_entries(self, residency, host):
        mc_memory.ensure_resident("B", ["flux2_vae.safetensors"])
        mc_memory.ensure_resident("B", ["qwen3_8b.safetensors"])

        assert len(mc_memory.cached_names()) == 2


class TestRestoreSelection:
    def test_restores_both_halves_of_the_stage_1_pair(self, residency, host):
        host.shared.opts.forge_additional_modules = [SDXL_VAE]
        stage1_modules = mc_memory.current_modules()

        mc_memory.ensure_resident("B", ["flux2_vae.safetensors"])
        assert host.shared.opts.forge_additional_modules == [FLUX_VAE]

        mc_memory.restore_selection("A", stage1_modules)

        assert host.shared.opts.forge_additional_modules == [SDXL_VAE]
        assert residency.state.checkpoint == "A"

    def test_restoring_still_does_not_reload(self, residency, host):
        mc_memory.ensure_resident("B", ["flux2_vae.safetensors"])
        reloads = residency.state.reloads

        mc_memory.restore_selection("A", [])

        assert residency.state.reloads == reloads


# --------------------------------------------------------------------------- #
# Infotext
# --------------------------------------------------------------------------- #


class TestModuleInfotext:
    def test_inheriting_records_nothing(self):
        assert mc_infotext.build_module_params([mc_infotext.INHERIT_MODULES]) == {}
        assert mc_infotext.build_module_params(None) == {}

    def test_empty_selection_records_the_builtin_sentinel(self):
        params = mc_infotext.build_module_params([])
        assert params == {"Model Chain Module 1": "Built-in"}

    def test_modules_are_recorded_by_basename(self):
        params = mc_infotext.build_module_params([FLUX_VAE, QWEN_TE])
        assert params == {
            "Model Chain Module 1": "flux2_vae",
            "Model Chain Module 2": "qwen3_8b",
        }

    def test_round_trip_restores_the_selection(self, host):
        params = mc_infotext.build_module_params([FLUX_VAE, QWEN_TE])
        assert mc_infotext.parse_modules(params) == [
            "flux2_vae.safetensors",
            "qwen3_8b.safetensors",
        ]

    def test_round_trip_preserves_order(self, host):
        params = {"Model Chain Module 2": "qwen3_8b", "Model Chain Module 1": "flux2_vae"}
        assert mc_infotext.parse_modules(params) == [
            "flux2_vae.safetensors",
            "qwen3_8b.safetensors",
        ]

    def test_builtin_sentinel_round_trips(self, host):
        assert mc_infotext.parse_modules({"Model Chain Module 1": "Built-in"}) == []

    def test_inherit_sentinel_round_trips(self, host):
        params = {"Model Chain Module 1": mc_infotext.INHERIT_MODULES}
        assert mc_infotext.parse_modules(params) == [mc_infotext.INHERIT_MODULES]

    def test_absent_keys_restore_the_inherit_default(self, host):
        assert mc_infotext.parse_modules({"Model Chain CFG": "1.0"}) is None

    def test_an_uninstalled_module_is_dropped_not_fatal(self, host):
        params = {"Model Chain Module 1": "flux2_vae", "Model Chain Module 2": "gone"}
        assert mc_infotext.parse_modules(params) == ["flux2_vae.safetensors"]

    def test_non_numeric_suffixes_are_ignored(self, host):
        params = {"Model Chain Module X": "flux2_vae"}
        assert mc_infotext.parse_modules(params) is None

    def test_keys_are_declared_for_pasting(self):
        declared = set(mc_infotext.paste_field_names())
        assert "Model Chain Module 1" in declared
        assert "Model Chain Module 2" in declared

    def test_build_params_includes_the_selection(self):
        params = mc_infotext.build_params(
            target="krea2.safetensors", prompt_mode="Replace", prompt="a lake",
            negative="", styles=[], seed_mode="Inherit", seed_offset=0, fixed_seed=-1,
            cfg=1.0, steps=8, sampler=mc_infotext.INHERIT_SAMPLER, denoise=1.0,
            size_multiplier=1.0, stage1_size="1024x1024", modules=[FLUX_VAE],
        )
        assert params["Model Chain Module 1"] == "flux2_vae"
