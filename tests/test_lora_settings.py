"""The LoRA details' settings -- three sections and one hidden option -- and
the one-time carry-over from the user's standalone extension.

The standalone kept its library, its images, its folder assignments, the last
key and folder it fetched with and its hash cache at its own root, and its
settings under ``eld_*``; the user's ask was to move all of it here "so I can
uninstall the lora extender repo". So the first start copies what is not here
yet into the LoRA data root, takes a setting still at its default here from
the standalone's value, and records that it did, which is what makes it once.
"""

from __future__ import annotations

import json
import logging
import os
import re
import stat
from pathlib import Path

import pytest

import mc_lora_paths
import mc_lora_settings as settings

LONGEST_DESCRIPTION = 400


@pytest.fixture
def registered(host):
    import model_chain  # noqa: F401 -- registers every section

    return host.shared.options_templates


def ours(templates):
    sections = {settings.SECTION, settings.FETCH_SECTION, settings.NETWORKS_SECTION}
    return {key: info for key, info in templates.items() if info.section in sections}


def described(info) -> str:
    return re.sub(r"<[^>]+>", "", info.comment_after or "").strip()


# -------------------------------------------------------------- the sections


class TestTheSections:
    def test_three_sections_are_drawn_and_the_record_is_hidden(self, registered):
        drawn = {info.section for info in ours(registered).values()}
        assert drawn == {settings.SECTION, settings.FETCH_SECTION, settings.NETWORKS_SECTION}
        assert all(section[0] and section[1].startswith("Model Chain: ") for section in drawn)
        assert registered[settings.OPT_IMPORT].section == settings.HIDDEN_SECTION
        assert settings.HIDDEN_SECTION[0] is None, "the host draws no control for a section without an id"

    def test_each_setting_is_in_its_areas_section(self, registered):
        assert registered[settings.OPT_ENABLED].section == settings.SECTION
        assert registered[settings.OPT_LIBRARY_DIR].section == settings.SECTION
        assert registered[settings.OPT_IMAGE_DIR].section == settings.SECTION
        assert registered[settings.OPT_API_KEY].section == settings.FETCH_SECTION
        assert registered[settings.OPT_BASE_MODEL].section == settings.FETCH_SECTION
        assert registered[settings.OPT_FILTER].section == settings.NETWORKS_SECTION
        assert registered[settings.OPT_TOP_LEVEL_BUTTONS].section == settings.NETWORKS_SECTION

    def test_every_setting_says_a_sentence_or_two_about_itself(self, registered):
        for key, info in ours(registered).items():
            if key == settings.OPT_UPLOAD:
                continue  # a widget, not a value: its notice is drawn inside it
            text = described(info)
            assert text, f"{key} says nothing about itself"
            assert len(text) <= LONGEST_DESCRIPTION, (key, len(text))
            assert text.count(". ") <= 3, f"{key}: {text}"

    def test_the_uploader_is_a_widget_the_host_never_saves(self, registered):
        uploader = registered[settings.OPT_UPLOAD]
        assert uploader.do_not_save is True
        assert callable(uploader.component)

    def test_every_standalone_setting_has_its_counterpart_here(self, registered):
        for old, new in settings.RENAMED.items():
            assert old.startswith(settings.STANDALONE_PREFIX), old
            assert new.startswith("model_chain_lora_"), new
            assert new in registered, new
        for gone in ("eld_panel_first", "eld_open_by_default", "eld_open_on_match"):
            assert gone not in settings.RENAMED
            assert gone.replace("eld_", "model_chain_lora_") not in registered

    def test_the_defaults_are_the_standalones(self, registered):
        assert registered[settings.OPT_ENABLED].default is True
        assert registered[settings.OPT_PROMPT_CARDS].default == 50
        assert registered[settings.OPT_PROMPT_LABEL_WIDTH].default == 90
        assert registered[settings.OPT_REQUEST_DELAY].default == 0.15
        assert registered[settings.OPT_RETRY_UNRESOLVED].default is False
        assert registered[settings.OPT_FILTER].default is True
        assert registered[settings.OPT_FILTER_CHECKPOINT_MENU].default is False
        assert registered[settings.OPT_TOP_LEVEL_BUTTONS].default is True, "the one new setting, and on"

    def test_the_filter_module_reads_the_names_this_module_registers(self):
        import mc_extra_networks_filter as network_filter

        assert network_filter.OPT_FILTER == settings.OPT_FILTER
        assert network_filter.OPT_FILTER_DIRS == settings.OPT_FILTER_DIRS
        assert network_filter.OPT_FILTER_STRICT == settings.OPT_FILTER_STRICT
        assert network_filter.OPT_CHECKPOINT_MENU == settings.OPT_FILTER_CHECKPOINT_MENU
        assert network_filter.OPT_TOP_LEVEL_BUTTONS == settings.OPT_TOP_LEVEL_BUTTONS


# ------------------------------------------------------------ the carry-over


@pytest.fixture
def standalone(tmp_path):
    """The standalone extension's folder as it is left after a year of use."""
    extensions = tmp_path / "extensions"
    root = extensions / mc_lora_paths.OLD_EXTENSION_DIRNAME
    (root / "library").mkdir(parents=True)
    (root / "library" / "civitai-moss.csv").write_text("sha256,civitai_model_name\nab,Moss\n", encoding="utf-8")
    (root / "images").mkdir()
    (root / "images" / "civitai-1.jpeg").write_bytes(b"\xff\xd8")
    (root / ".cache").mkdir()
    (root / ".cache" / "hashes.json").write_text("{}", encoding="utf-8")
    (root / "preset_folders.json").write_text(json.dumps({"version": 1, "presets": {}}), encoding="utf-8")
    (root / "last_used.json").write_text(json.dumps({"civitai_api_key": "secret"}), encoding="utf-8")
    (root / "scripts").mkdir()
    (root / "scripts" / "extended_lora_details.py").write_text("# code, not data\n", encoding="utf-8")
    return extensions


DEFAULTS = {
    settings.OPT_ENABLED: True,
    settings.OPT_PROMPT_CARDS: 50,
    settings.OPT_HASH_ON_OPEN: True,
    settings.OPT_API_KEY: "",
    settings.OPT_LIBRARY_DIR: "",
}


class TestTheCarryOver:
    def test_files_and_settings_come_over_once(self, standalone, lora_root, host, caplog):
        opts = host.shared.opts
        opts.data.update({
            "eld_enabled": True,                  # the default here too: nothing to write
            "eld_prompt_list_limit": 120,         # ours at its default: taken
            "eld_hash_on_open": True,             # ours already set here: kept as set
            "eld_civitai_api_key": "secret",      # taken, and kept out of the record
            "eld_panel_first": True,              # no counterpart: left behind
            settings.OPT_HASH_ON_OPEN: False,
        })
        saved = []
        # a library file already here is never overwritten
        (lora_root / "library").mkdir(parents=True)
        (lora_root / "library" / "civitai-moss.csv").write_text("mine\n", encoding="utf-8")

        with caplog.at_level(logging.INFO, logger="model_chain"):
            record = settings.import_standalone(
                opts, standalone, defaults=DEFAULTS, data_root=lora_root,
                save=lambda: saved.append(1), now=lambda: 1_800_000_000)

        assert record is not None and saved == [1]
        assert record["version"] == settings.IMPORT_VERSION
        assert record["from"] == str(standalone / mc_lora_paths.OLD_EXTENSION_DIRNAME)
        assert re.match(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d", record["when"])

        # the files, each where mc_lora_paths puts it now
        assert (lora_root / "library" / "civitai-moss.csv").read_text(encoding="utf-8") == "mine\n"
        assert record["files"]["library"] == {"copied": 0, "kept": 1}
        assert (lora_root / "images" / "civitai-1.jpeg").read_bytes() == b"\xff\xd8"
        assert record["files"]["images"] == {"copied": 1, "kept": 0}
        assert (lora_root / "cache" / "hashes.json").is_file()
        assert (lora_root / "preset_folders.json").is_file()
        assert json.loads((lora_root / "last_used.json").read_text(encoding="utf-8")) == {"civitai_api_key": "secret"}
        if os.name == "posix":
            assert stat.S_IMODE((lora_root / "last_used.json").stat().st_mode) & 0o077 == 0, "it holds an API key"
        assert not (lora_root / "scripts").exists(), "data, never the standalone's code"

        # the settings
        assert opts.data[settings.OPT_PROMPT_CARDS] == 120
        assert opts.data[settings.OPT_API_KEY] == "secret"
        assert opts.data[settings.OPT_HASH_ON_OPEN] is False, "set here already: kept as set"
        assert settings.OPT_ENABLED not in opts.data, "equal to the default: nothing to write"
        assert record["applied"] == {settings.OPT_PROMPT_CARDS: 120}, "the key is applied and never recorded"
        assert record["kept"] == [settings.OPT_HASH_ON_OPEN]
        assert json.loads(opts.data[settings.OPT_IMPORT]) == record
        assert opts.data["eld_panel_first"] is True, "the standalone's own settings are left as they are"

        lines = [r.getMessage() for r in caplog.records if r.name == "model_chain"]
        assert any("can be uninstalled" in line for line in lines), lines
        assert not any(str(standalone) in line or str(lora_root) in line for line in lines), "counts, never a path"

        # once
        again = settings.import_standalone(opts, standalone, defaults=DEFAULTS, data_root=lora_root, save=lambda: saved.append(2))
        assert again is None and saved == [1]

    def test_nothing_to_do_leaves_no_record_and_makes_no_folder(self, tmp_path, lora_root, host):
        opts = host.shared.opts
        record = settings.import_standalone(opts, tmp_path / "extensions", defaults=DEFAULTS, data_root=lora_root)
        assert record is None
        assert settings.OPT_IMPORT not in opts.data
        assert not lora_root.exists()

    def test_the_settings_alone_come_over_when_the_folder_is_already_gone(self, tmp_path, lora_root, host):
        opts = host.shared.opts
        opts.data["eld_library_dir"] = "D:/loras/library"
        record = settings.import_standalone(opts, tmp_path / "extensions", defaults=DEFAULTS, data_root=lora_root)
        assert record is not None and record["from"] == "" and record["files"] == {}
        assert opts.data[settings.OPT_LIBRARY_DIR] == "D:/loras/library"

    def test_the_standalones_folder_is_found_whatever_its_case(self, tmp_path):
        extensions = tmp_path / "extensions"
        (extensions / "sd-forgeneo-extendedloradetails").mkdir(parents=True)
        assert mc_lora_paths.old_extension_root(extensions) == extensions / "sd-forgeneo-extendedloradetails"
        assert mc_lora_paths.old_extension_root(tmp_path / "elsewhere") is None
        (extensions / "SD-ForgeNeo-ExtendedLoraDetails.txt").write_text("a file, not the folder", encoding="utf-8")
        assert mc_lora_paths.old_extension_root(extensions) == extensions / "sd-forgeneo-extendedloradetails"

    def test_a_copy_that_fails_is_reported_and_tried_again_next_start(self, standalone, lora_root, host, monkeypatch):
        def refuse(*args, **kwargs):
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(settings.shutil, "copy2", refuse)
        opts = host.shared.opts
        assert settings.import_standalone(opts, standalone, defaults=DEFAULTS, data_root=lora_root) is None
        assert settings.OPT_IMPORT not in opts.data, "no record, so the next start tries again"

    def test_on_app_started_runs_under_the_host(self, lora_root, host, caplog):
        with caplog.at_level(logging.INFO, logger="model_chain"):
            settings.on_app_started()
        from mc_lora_library import get_library

        assert get_library().directory == lora_root / "library"
        assert not any(record.levelno >= logging.WARNING for record in caplog.records), [
            record.getMessage() for record in caplog.records if record.levelno >= logging.WARNING]


# ---------------------------------------------------------------- the paths


class TestThePaths:
    def test_everything_lives_under_one_data_root(self, lora_root):
        assert mc_lora_paths.data_root() == lora_root
        assert mc_lora_paths.library_dir() == lora_root / "library"
        assert mc_lora_paths.image_dir() == lora_root / "images"
        assert mc_lora_paths.cache_dir() == lora_root / "cache"
        assert mc_lora_paths.folders_config_path() == lora_root / "preset_folders.json"
        assert mc_lora_paths.last_used_path() == lora_root / "last_used.json"
        assert not lora_root.exists(), "asking makes nothing"

    def test_the_two_folders_a_setting_can_move(self, lora_root, host, monkeypatch):
        monkeypatch.setattr(host.shared.opts, settings.OPT_LIBRARY_DIR, "~/csv", raising=False)
        monkeypatch.setattr(host.shared.opts, settings.OPT_IMAGE_DIR, "  ", raising=False)
        assert mc_lora_paths.library_dir() == Path("~/csv").expanduser()
        assert mc_lora_paths.image_dir() == lora_root / "images", "blank is the default"
