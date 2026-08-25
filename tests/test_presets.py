"""Named Stage 2 configurations."""

from __future__ import annotations

import json

import pytest

import mc_presets


@pytest.fixture(autouse=True)
def store(host, tmp_path, monkeypatch):
    """Point the preset store at a scratch file."""
    file = tmp_path / "model_chain_presets.json"
    monkeypatch.setattr(mc_presets, "path", lambda: str(file))
    return file


def config(**overrides):
    values = {field: None for field in mc_presets.FIELDS}
    values.update(
        {
            "enabled": True,
            "target": "fluxKlein9B.safetensors",
            "modules": ["flux2_vae.safetensors", "qwen3_8b.safetensors"],
            "prompt_mode": "Replace",
            "prompt": "a serene lake <lora:detail:0.8>",
            "negative": "blurry",
            "styles": ["Cinematic"],
            "seed_mode": "Inherit",
            "seed_offset": 0,
            "fixed_seed": -1,
            "cfg": 1.0,
            "steps": 4,
            "sampler": "Euler",
            "scheduler": "Beta",
            "denoise": 1.0,
            "size_multiplier": 1.5,
            "edit_mode": "Enable",
        }
    )
    values.update(overrides)
    return values


class TestRoundTrip:
    def test_save_then_get_returns_every_field(self):
        mc_presets.save("Klein refine", config())

        loaded = mc_presets.get("Klein refine")
        assert loaded == config()

    def test_every_stage_2_control_is_captured(self):
        """"Everything configured in it" -- no control silently dropped."""
        mc_presets.save("All", config())
        loaded = mc_presets.get("All")
        assert set(loaded) == set(mc_presets.FIELDS)

    def test_list_types_survive_the_round_trip(self):
        mc_presets.save("Lists", config())
        loaded = mc_presets.get("Lists")
        assert loaded["modules"] == ["flux2_vae.safetensors", "qwen3_8b.safetensors"]
        assert loaded["styles"] == ["Cinematic"]

    def test_lora_tags_in_a_saved_prompt_survive(self):
        mc_presets.save("Lora", config(prompt="a lake <lora:krea2_edit:1.0>"))
        assert mc_presets.get("Lora")["prompt"] == "a lake <lora:krea2_edit:1.0>"

    def test_saving_twice_overwrites_rather_than_duplicating(self):
        mc_presets.save("Same", config(steps=4))
        mc_presets.save("Same", config(steps=20))

        assert mc_presets.names() == ["Same"]
        assert mc_presets.get("Same")["steps"] == 20

    def test_several_presets_coexist(self):
        mc_presets.save("Klein", config())
        mc_presets.save("Krea", config(target="krea2.safetensors"))

        assert mc_presets.names() == ["Klein", "Krea"]
        assert mc_presets.get("Krea")["target"] == "krea2.safetensors"


class TestNaming:
    def test_names_are_sorted_case_insensitively(self):
        for name in ("zebra", "Apple", "mango"):
            mc_presets.save(name, config())

        assert mc_presets.names() == ["Apple", "mango", "zebra"]

    def test_choices_lead_with_the_no_selection_entry(self):
        mc_presets.save("Klein", config())
        assert mc_presets.choices() == [mc_presets.NONE, "Klein"]

    def test_an_empty_name_is_refused(self):
        with pytest.raises(mc_presets.PresetError):
            mc_presets.save("   ", config())

    def test_the_reserved_name_is_refused(self):
        with pytest.raises(mc_presets.PresetError):
            mc_presets.save(mc_presets.NONE, config())

    def test_surrounding_whitespace_is_trimmed(self):
        mc_presets.save("  Klein  ", config())
        assert mc_presets.names() == ["Klein"]


class TestDelete:
    def test_removes_only_the_named_preset(self):
        mc_presets.save("Klein", config())
        mc_presets.save("Krea", config())

        remaining = mc_presets.delete("Klein")

        assert remaining == ["Krea"]
        assert mc_presets.get("Klein") is None

    def test_deleting_an_unknown_preset_is_refused(self):
        with pytest.raises(mc_presets.PresetError):
            mc_presets.delete("nope")

    def test_deleting_nothing_is_refused(self):
        with pytest.raises(mc_presets.PresetError):
            mc_presets.delete(mc_presets.NONE)


class TestMissing:
    def test_no_file_yet_reads_as_empty(self, store):
        assert not store.exists()
        assert mc_presets.names() == []
        assert mc_presets.get("anything") is None

    def test_get_on_the_no_selection_entry(self):
        assert mc_presets.get(mc_presets.NONE) is None
        assert mc_presets.get("") is None


class TestDamagedFile:
    def test_unparseable_json_is_treated_as_empty(self, store):
        store.write_text("{not json at all")
        assert mc_presets.names() == []

    def test_a_damaged_file_can_still_be_written_over(self, store):
        store.write_text("{not json at all")
        mc_presets.save("Fresh", config())
        assert mc_presets.names() == ["Fresh"]

    def test_unexpected_shapes_are_ignored(self, store):
        store.write_text(json.dumps({"version": 1, "presets": {"ok": {"target": "x"}, "bad": "a string"}}))
        assert mc_presets.names() == ["ok"]

    def test_a_top_level_list_is_ignored(self, store):
        store.write_text(json.dumps(["not", "a", "store"]))
        assert mc_presets.names() == []


class TestFileFormat:
    def test_written_file_is_versioned_json(self, store):
        mc_presets.save("Klein", config())

        data = json.loads(store.read_text())
        assert data["version"] == mc_presets.SCHEMA_VERSION
        assert "Klein" in data["presets"]

    def test_no_temporary_files_are_left_behind(self, store, tmp_path):
        mc_presets.save("Klein", config())
        leftovers = [p.name for p in tmp_path.iterdir() if p.name != store.name]
        assert leftovers == []

    def test_an_unwritable_location_raises_rather_than_failing_silently(self, tmp_path, monkeypatch):
        # A regular file standing where a directory would have to be: makedirs
        # cannot succeed, whatever the process's privileges.
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory")
        monkeypatch.setattr(mc_presets, "path", lambda: str(blocker / "presets.json"))

        with pytest.raises(mc_presets.PresetError):
            mc_presets.save("Klein", config())


class TestForwardCompatibility:
    def test_fields_added_since_a_preset_was_saved_fall_back_to_defaults(self):
        """A preset from an older version must not blank newer controls."""
        old = {"target": "klein.safetensors", "steps": 4}
        defaults = {field: f"default-{field}" for field in mc_presets.FIELDS}

        resolved = mc_presets.apply_defaults(old, defaults)

        assert resolved["target"] == "klein.safetensors"
        assert resolved["steps"] == 4
        assert resolved["scheduler"] == "default-scheduler"
        assert set(resolved) == set(mc_presets.FIELDS)

    def test_null_values_fall_back_to_defaults(self):
        defaults = {field: f"default-{field}" for field in mc_presets.FIELDS}
        resolved = mc_presets.apply_defaults({"cfg": None}, defaults)
        assert resolved["cfg"] == "default-cfg"

    def test_unknown_fields_in_a_stored_preset_are_dropped(self):
        defaults = {field: None for field in mc_presets.FIELDS}
        resolved = mc_presets.apply_defaults({"from_the_future": 1}, defaults)
        assert "from_the_future" not in resolved


class TestWhereThePresetControlsSit:
    """A preset is the whole of Stage 2, so it is the first thing in Stage 2.

    Checkpoint, modules, sampling, seed policy, denoise -- one saved choice
    decides all of them. The panel used to open on the checkpoint chooser with
    presets in the last of six accordions, which put the decisions in the wrong
    order: the checkpoint is one of the things a preset has already decided.

    Read from the source rather than from a built page because the fake Gradio
    these tests run against records components, not the containers they were
    built inside -- and what is being asserted here is an ordering of the
    building, which the source is the honest record of.
    """

    def source(self) -> str:
        from pathlib import Path

        return (Path(__file__).resolve().parent.parent
                / "scripts" / "model_chain.py").read_text(encoding="utf-8")

    def stage2(self) -> str:
        text = self.source()
        return text[text.index("# -- 9.1 "):text.index("# -- the memory contract")]

    def test_the_preset_chooser_comes_before_the_checkpoint(self):
        block = self.stage2()

        assert block.index('elem_id=self.elem_id("preset")') < \
            block.index('elem_id=self.elem_id("target")')

    def test_the_preset_chooser_is_not_behind_an_accordion(self):
        """The first section of Stage 2 is not a section anybody has to open."""
        block = self.stage2()
        before = block[:block.index('elem_id=self.elem_id("preset")')]

        assert "gr.Accordion" not in before

    def test_the_checkpoint_is_behind_one(self):
        block = self.stage2()
        opened = block[:block.index('elem_id=self.elem_id("target")')]

        assert opened.rindex("gr.Accordion") > opened.rindex("gr.Group")

    def test_the_modules_are_beside_the_checkpoint_they_describe(self):
        """They used to be five sections apart, describing the same model."""
        block = self.stage2()
        between = block[block.index('elem_id=self.elem_id("target")'):
                        block.index('elem_id=self.elem_id("modules")')]

        assert "gr.Accordion" not in between
