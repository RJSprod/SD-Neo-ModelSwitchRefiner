"""The extra-network browsers' folders: the folder assignments per UI Preset,
the filter that narrows the pages to them, and the folder buttons -- the
assigned ones only, and only the top-level ones.

Carried in with the LoRA details (docs/26), and the third of the user's three
asks on 2026-10-23: "I noticed that show / hide folders doesn't work in the
extra networks drawer ... I'm okay with hiding all subfolders in general from
the folder buttons. I made all lora folders model names, like SDXL ... I have
no need to ever tap on a subfolder." The drawer's toggle is the stylesheet's
(``test_lora_css.py``); the buttons are here.
"""

from __future__ import annotations

import json
import os
import types
from pathlib import Path

import pytest

import mc_extra_networks_filter as network_filter
import mc_extra_networks_folders as preset_folders


def make_network_tree(root: Path) -> dict:
    """A models folder with two top-level folders and one nested folder."""
    layout = {
        "root": root,
        "zit": root / "zit",
        "zit_faces": root / "zit" / "faces",
        "xl": root / "xl",
    }
    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)
    for name, path in layout.items():
        (path / f"{name}.safetensors").write_bytes(b"x")
    return layout


def buttons_html(*subdirs: str) -> str:
    """What Forge's ``create_dirs_view_html`` writes: one button per folder
    and the "all" button first, the only one with ``search-all``."""
    return "".join(
        f"""<button class='lg secondary gradio-button custom-button{" search-all" if subdir == "" else ""}'
        onclick='extraNetworksSearchButton("txt2img", "lora", event)'>
        {subdir if subdir else "all"}
        </button>"""
        for subdir in subdirs
    )


@pytest.fixture
def chosen(monkeypatch, lora_root):
    """A store under the data root, and the preset the filter sees as active."""
    store = preset_folders.PresetFolderStore(lora_root / "preset_folders.json")

    def choose(preset: str):
        monkeypatch.setattr(preset_folders, "get_store", lambda: store)
        monkeypatch.setattr(preset_folders, "current_preset", lambda: preset)
        return store

    return choose


# ------------------------------------------------------------------ the rules


class TestTheRules:
    def test_a_folder_without_subfolders_covers_only_itself(self, tmp_path):
        tree = make_network_tree(tmp_path / "Lora")
        rules = [preset_folders.Rule(str(tree["zit"]), subfolders=False)]

        assert preset_folders.matches(rules, tree["zit"])
        assert not preset_folders.matches(rules, tree["zit_faces"])
        assert not preset_folders.matches(rules, tree["xl"])
        assert not preset_folders.matches(rules, tree["root"])

        assert preset_folders.file_allowed(tree["zit"] / "zit.safetensors", rules)
        assert not preset_folders.file_allowed(tree["zit_faces"] / "zit_faces.safetensors", rules)

    def test_including_subfolders_covers_the_whole_subtree(self, tmp_path):
        tree = make_network_tree(tmp_path / "Lora")
        rules = [preset_folders.Rule(str(tree["zit"]), subfolders=True)]

        assert preset_folders.matches(rules, tree["zit"])
        assert preset_folders.matches(rules, tree["zit_faces"])
        assert not preset_folders.matches(rules, tree["xl"])

        # a sibling whose name merely starts the same way is not "inside"
        assert not preset_folders.matches(rules, str(tree["zit"]) + "-extra")

    def test_assignments_survive_a_reload_and_drop_duplicates(self, tmp_path):
        tree = make_network_tree(tmp_path / "Lora")
        config = tmp_path / "preset_folders.json"
        store = preset_folders.PresetFolderStore(config)

        store.set_rules(
            "zit",
            "lora",
            [
                preset_folders.Rule(str(tree["zit"]), subfolders=True),
                preset_folders.Rule(str(tree["zit"]) + os.sep, subfolders=False),  # same folder
                preset_folders.Rule(str(tree["xl"]), subfolders=False),
            ],
        )

        reopened = preset_folders.PresetFolderStore(config)
        rules = reopened.rules("zit", "lora")
        assert [rule.subfolders for rule in rules] == [True, False]
        assert preset_folders.normalize_dir(rules[0].path) == preset_folders.normalize_dir(tree["zit"])
        assert reopened.rules("xl", "lora") == []
        assert reopened.presets_in_use() == ["zit"]

        # an empty assignment removes the page, and the preset with it
        reopened.set_rules("zit", "lora", [])
        assert preset_folders.PresetFolderStore(config).presets_in_use() == []

    def test_a_bare_string_entry_is_read_as_folder_only(self, tmp_path):
        config = tmp_path / "preset_folders.json"
        config.write_text(
            json.dumps({"version": 1, "presets": {"zit": {"lora": ["/models/Lora/zit"]}}}),
            encoding="utf-8",
        )
        rules = preset_folders.PresetFolderStore(config).rules("zit", "lora")
        assert len(rules) == 1
        assert rules[0].subfolders is False

    def test_the_assignments_file_lives_under_the_data_root(self, lora_root):
        store = preset_folders.get_store()
        store.set_rules("zit", "lora", [preset_folders.Rule("/models/Lora/zit", subfolders=False)])
        assert (lora_root / "preset_folders.json").is_file()
        assert preset_folders.get_store() is store

    def test_folder_choices_list_the_root_and_every_subfolder(self, tmp_path):
        tree = make_network_tree(tmp_path / "Lora")
        (tree["root"] / ".hidden").mkdir()

        choices = preset_folders.folder_choices([str(tree["root"])])
        labels = [label for label, _ in choices]
        values = [preset_folders.normalize_dir(value) for _, value in choices]

        assert "Lora/" in labels
        assert "Lora/zit/" in labels
        assert "Lora/zit/faces/" in labels
        assert "Lora/.hidden/" not in labels
        assert preset_folders.normalize_dir(tree["zit_faces"]) in values
        assert len(values) == len(set(values))


# ----------------------------------------------------------------- the filter


class TestTheFilter:
    def test_the_filter_only_runs_for_presets_with_folders_assigned(self, tmp_path, chosen, host):
        tree = make_network_tree(tmp_path / "Lora")
        store = chosen("zit")
        store.set_rules("zit", "lora", [preset_folders.Rule(str(tree["zit"]), subfolders=False)])

        class Page:
            extra_networks_tabname = "lora"

        rules = network_filter.active_rules(Page())
        assert rules and rules[0].subfolders is False

        # a preset with nothing assigned is left alone
        chosen("xl")
        assert network_filter.active_rules(Page()) is None

        # ... and so is a page with nothing assigned
        class Checkpoints:
            extra_networks_tabname = "checkpoints"

        chosen("zit")
        assert network_filter.active_rules(Checkpoints()) is None

    def test_the_filter_can_be_turned_off_and_made_strict(self, tmp_path, chosen, host, monkeypatch):
        tree = make_network_tree(tmp_path / "Lora")
        store = chosen("zit")
        store.set_rules("zit", "lora", [preset_folders.Rule(str(tree["zit"]), subfolders=False)])

        class Page:
            extra_networks_tabname = "lora"

        monkeypatch.setattr(host.shared.opts, network_filter.OPT_FILTER, False, raising=False)
        assert network_filter.active_rules(Page()) is None

        monkeypatch.setattr(host.shared.opts, network_filter.OPT_FILTER, True, raising=False)
        monkeypatch.setattr(host.shared.opts, network_filter.OPT_FILTER_STRICT, True, raising=False)
        chosen("xl")
        assert network_filter.active_rules(Page()) == [], "strict: nothing assigned means nothing shown"

    def test_items_outside_the_assigned_folders_are_dropped(self, tmp_path):
        tree = make_network_tree(tmp_path / "Lora")
        rules = [preset_folders.Rule(str(tree["zit"]), subfolders=False)]

        inside = {"filename": str(tree["zit"] / "zit.safetensors")}
        nested = {"filename": str(tree["zit_faces"] / "zit_faces.safetensors")}
        elsewhere = {"filename": str(tree["xl"] / "xl.safetensors")}

        assert network_filter.item_allowed(inside, rules)
        assert not network_filter.item_allowed(nested, rules)
        assert not network_filter.item_allowed(elsewhere, rules)

        # an empty rule list means "the preset owns nothing here": hide it all
        assert not network_filter.item_allowed(inside, [])

    def test_the_wrapped_page_lists_only_the_assigned_folders_items(self, tmp_path, chosen, host):
        tree = make_network_tree(tmp_path / "Lora")
        store = chosen("zit")
        store.set_rules("zit", "lora", [preset_folders.Rule(str(tree["zit"]), subfolders=False)])
        seen = []

        class Page:
            extra_networks_tabname = "lora"

            def list_items(self):
                yield {"filename": str(tree["zit"] / "zit.safetensors")}
                yield {"filename": str(tree["xl"] / "xl.safetensors")}
                yield None

            def create_html(self, tabname):
                seen.extend(item for item in self.list_items())
                return "cards"

        create_html = network_filter._wrap_create_html(Page.create_html)
        page = Page()
        assert create_html(page, "txt2img") == "cards"
        assert [os.path.basename(item["filename"]) for item in seen] == ["zit.safetensors"]
        assert "list_items" not in vars(page), "the shadow lasts for the call only"

    def test_checkpoint_titles_outside_the_assigned_folders_are_dropped(self, tmp_path, host, monkeypatch):
        tree = make_network_tree(tmp_path / "Stable-diffusion")
        rules = [preset_folders.Rule(str(tree["zit"]), subfolders=False)]

        def info(name, path):
            return types.SimpleNamespace(name=name, short_title=name, filename=str(path))

        checkpoints = {
            "zit.safetensors": info("zit.safetensors", tree["zit"] / "zit.safetensors"),
            "nested.safetensors": info("nested.safetensors", tree["zit_faces"] / "nested.safetensors"),
            "xl.safetensors": info("xl.safetensors", tree["xl"] / "xl.safetensors"),
        }
        monkeypatch.setattr(host.sd_models, "checkpoints_list", checkpoints, raising=False)
        monkeypatch.setattr(host.shared.opts, "sd_checkpoint_dropdown_use_short", False, raising=False)
        monkeypatch.setattr(host.shared.opts, "sd_model_checkpoint", "xl.safetensors", raising=False)

        kept = network_filter.filter_checkpoint_tiles(
            ["zit.safetensors", "nested.safetensors", "xl.safetensors", "mystery.safetensors"],
            rules,
        )
        # inside the folder: kept. nested: dropped. loaded checkpoint: always kept.
        # unknown title: kept, so nothing becomes unloadable.
        assert kept == ["zit.safetensors", "xl.safetensors", "mystery.safetensors"]


# ---------------------------------------------------------- the folder buttons


class TestTheFolderButtons:
    def test_folder_buttons_are_kept_only_for_assigned_folders(self, tmp_path):
        tree = make_network_tree(tmp_path / "Lora")
        markup = buttons_html("", "zit/", "zit/faces/", "xl/")
        roots = [str(tree["root"])]

        kept = network_filter.filter_dirs_html(markup, roots, [preset_folders.Rule(str(tree["zit"]), False)])
        assert "all" in kept
        assert "zit/" in kept
        assert "zit/faces/" not in kept
        assert "xl/" not in kept

        deep = network_filter.filter_dirs_html(markup, roots, [preset_folders.Rule(str(tree["zit"]), True)])
        assert "zit/faces/" in deep
        assert "xl/" not in deep

        # backslash-separated labels (Windows) resolve to the same folders
        windows = markup.replace("zit/faces/", "\\zit\\faces\\")
        assert "zit\\faces" in network_filter.filter_dirs_html(windows, roots, [preset_folders.Rule(str(tree["zit"]), True)])

        # markup without buttons is handed back untouched
        assert network_filter.filter_dirs_html("<div>none</div>", roots, []) == "<div>none</div>"

    def test_only_the_top_level_folders_keep_a_button(self):
        """A button for SDXL, none for SDXL/LIGHTING; the "all" button stays,
        and so does every folder directly under a LoRA directory, whichever
        way Forge spells it."""
        markup = buttons_html("", "SDXL/", "SDXL/LIGHTING/", "/Flux/", "/Flux/nsfw/deep/", "Krea 2\\", "Krea 2\\styles\\", "empty")
        kept = network_filter.top_level_dirs_html(markup)
        assert "all" in kept
        assert "SDXL/" in kept and "LIGHTING" not in kept
        assert "/Flux/" in kept and "deep" not in kept and "nsfw" not in kept
        assert "Krea 2\\" in kept and "styles" not in kept
        assert "empty" in kept, "a folder with nothing in it has no trailing separator, and is still top-level"
        assert kept.count("<button") == 5

        assert network_filter.top_level_dirs_html("<div>none</div>") == "<div>none</div>"
        assert network_filter.top_level_dirs_html("") == ""

    def test_the_wrapped_page_draws_top_level_buttons_unless_told_otherwise(self, tmp_path, chosen, host, monkeypatch):
        tree = make_network_tree(tmp_path / "Lora")
        chosen("")  # no preset: the assignment filter stays out of it
        markup = buttons_html("", "zit/", "zit/faces/", "xl/")

        class Page:
            extra_networks_tabname = "lora"

            def allowed_directories_for_previews(self):
                return [str(tree["root"])]

            def create_dirs_view_html(self, tabname):
                return markup

        draw = network_filter._wrap_dirs_view(Page.create_dirs_view_html)

        drawn = draw(Page(), "txt2img")
        assert "zit/" in drawn and "xl/" in drawn and "zit/faces/" not in drawn, "on by default"

        monkeypatch.setattr(host.shared.opts, network_filter.OPT_TOP_LEVEL_BUTTONS, False, raising=False)
        assert draw(Page(), "txt2img") == markup, "off: every folder's button, as Forge draws them"

    def test_the_assignment_filter_still_applies_after_the_top_level_cut(self, tmp_path, chosen, host, monkeypatch):
        tree = make_network_tree(tmp_path / "Lora")
        store = chosen("zit")
        store.set_rules("zit", "lora", [preset_folders.Rule(str(tree["zit"]), subfolders=True)])
        markup = buttons_html("", "zit/", "zit/faces/", "xl/")

        class Page:
            extra_networks_tabname = "lora"

            def allowed_directories_for_previews(self):
                return [str(tree["root"])]

            def create_dirs_view_html(self, tabname):
                return markup

        draw = network_filter._wrap_dirs_view(Page.create_dirs_view_html)
        drawn = draw(Page(), "txt2img")
        assert "zit/" in drawn and "xl/" not in drawn, "the preset owns zit only"
        assert "zit/faces/" not in drawn, "assigned with its subfolders, and still no button for one"

        # the buttons' own filter off: the preset's buttons, every level
        monkeypatch.setattr(host.shared.opts, network_filter.OPT_TOP_LEVEL_BUTTONS, False, raising=False)
        assert "zit/faces/" in draw(Page(), "txt2img")

        # the assignment filter off for the buttons: every top-level button
        monkeypatch.setattr(host.shared.opts, network_filter.OPT_FILTER_DIRS, False, raising=False)
        monkeypatch.setattr(host.shared.opts, network_filter.OPT_TOP_LEVEL_BUTTONS, True, raising=False)
        drawn = draw(Page(), "txt2img")
        assert "xl/" in drawn and "zit/faces/" not in drawn

    def test_a_failure_inside_the_wrapper_hands_forges_markup_back(self, chosen, host):
        chosen("zit")
        markup = buttons_html("", "zit/", "zit/faces/")

        class Page:
            extra_networks_tabname = "lora"

            def allowed_directories_for_previews(self):
                raise RuntimeError("no roots today")

            def create_dirs_view_html(self, tabname):
                return markup

        chosen("zit").set_rules("zit", "lora", [preset_folders.Rule("/models/Lora/zit", subfolders=False)])
        draw = network_filter._wrap_dirs_view(Page.create_dirs_view_html)
        drawn = draw(Page(), "txt2img")
        assert "zit/" in drawn and "zit/faces/" not in drawn, "the top-level cut had already happened"


# ------------------------------------------------------------ the attachment


class TestAttachingToForge:
    def test_the_page_class_is_wrapped_once(self, monkeypatch):
        class ExtraNetworksPage:
            def create_html(self, tabname):
                return "stock"

            def create_dirs_view_html(self, tabname):
                return "<button class='search-all'>all</button>"

        stock_html = ExtraNetworksPage.create_html
        monkeypatch.setattr(network_filter, "_page_class", lambda: ExtraNetworksPage)
        monkeypatch.setattr(network_filter, "apply_checkpoint_menu", lambda: True)

        assert network_filter.apply()
        assert ExtraNetworksPage.create_html is not stock_html
        wrapped = ExtraNetworksPage.create_html
        assert network_filter.apply(), "Reload UI runs every script again"
        assert ExtraNetworksPage.create_html is wrapped, "and the wrapper is not wrapped"

    def test_without_the_page_class_nothing_is_attached_and_nothing_raises(self, monkeypatch):
        monkeypatch.setattr(network_filter, "_page_class", lambda: None)
        assert network_filter.apply() is False
        network_filter.on_before_ui()

    def test_the_hook_never_raises_into_the_host(self, monkeypatch):
        def explode():
            raise RuntimeError("boom")

        monkeypatch.setattr(network_filter, "apply", explode)
        network_filter.on_before_ui()
