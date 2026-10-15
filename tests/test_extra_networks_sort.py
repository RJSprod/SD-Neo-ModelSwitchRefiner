"""Extra networks: newest created first, and "created" meaning created.

The user's word: "default sort to newest created to oldest". Forge's own two
settings -- *Default sorting method* and *Default sorting order* for Extra
Networks cards -- are set to Date Created and Descending once, while both are
still at Forge's defaults. And Forge's sort keys, which put a file's
modification time under ``date_created`` and its ``st_ctime`` (on Windows the
creation time) under ``date_modified``, are corrected wherever the platform
keeps a creation time.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import types

import pytest

import mc_extra_networks_sort as sort


class Stat:
    """An ``os.stat`` result as far as the sort keys read one."""

    def __init__(self, mtime, ctime, birthtime=None):
        self.st_mtime = mtime
        self.st_ctime = ctime
        if birthtime is not None:
            self.st_birthtime = birthtime


def forge_module(times):
    """``modules.ui_extra_networks`` as far as the sort keys go: Forge's own
    ``get_sort_keys``, which reads its lister's (mtime, ctime) and puts the
    first under ``date_created`` and the second under ``date_modified``."""

    class ExtraNetworksPage:
        def get_sort_keys(self, path):
            mtime, ctime = times.get(path, (0, 0))
            return {"date_created": int(mtime), "date_modified": int(ctime),
                    "name": os.path.basename(path).lower(), "path": str(path).lower()}

    return types.SimpleNamespace(ExtraNetworksPage=ExtraNetworksPage)


@pytest.fixture
def files(monkeypatch):
    """Two LoRAs: one downloaded (written and created at 2000) and one copied
    in from another drive later (created at 3000, but keeping the time it was
    last written elsewhere, 1000). What the platform keeps is set per test."""
    stats = {}
    times = {}

    def fake_stat(path, follow_symlinks=True):
        assert follow_symlinks is False, "not following a link, as Forge's lister does not"
        if path not in stats:
            raise FileNotFoundError(path)
        return stats[path]

    monkeypatch.setattr(sort, "_stat", fake_stat)

    def keep(birthtimes: bool, windows: bool):
        monkeypatch.setattr(sort, "_windows", lambda: windows)
        for path, (mtime, created) in {"downloaded.safetensors": (2000, 2000),
                                       "copied.safetensors": (1000, 3000)}.items():
            # Windows' ctime is the creation time; elsewhere it is not one.
            ctime = created if windows else 4000
            stats[path] = Stat(mtime, ctime, birthtime=created if birthtimes else None)
            times[path] = (mtime, ctime)
        return times

    return keep


def newest_created_first(module, paths):
    page = module.ExtraNetworksPage()
    return sorted(paths, key=lambda path: page.get_sort_keys(path)["date_created"], reverse=True)


class TestCreatedMeansCreated:
    def test_forge_alone_puts_a_copied_file_among_the_oldest(self, files):
        """What the user's Windows Forge did: *date created* is the last write."""
        module = forge_module(files(birthtimes=False, windows=True))

        assert newest_created_first(module, ["downloaded.safetensors", "copied.safetensors"]) == [
            "downloaded.safetensors", "copied.safetensors"]

    def test_where_the_platform_keeps_a_creation_time_date_created_is_it(self, files):
        module = forge_module(files(birthtimes=True, windows=True))
        assert sort.install(module) is True
        page = module.ExtraNetworksPage()

        keys = page.get_sort_keys("copied.safetensors")
        assert keys == {"date_created": 3000, "date_modified": 1000,
                        "name": "copied.safetensors", "path": "copied.safetensors"}
        assert newest_created_first(module, ["downloaded.safetensors", "copied.safetensors"]) == [
            "copied.safetensors", "downloaded.safetensors"]

    def test_on_windows_without_a_birth_time_the_ctime_is_the_creation_time(self, files):
        """Python before 3.12 has no ``st_birthtime`` on Windows; its
        ``st_ctime`` is the creation time there."""
        module = forge_module(files(birthtimes=False, windows=True))
        sort.install(module)

        assert module.ExtraNetworksPage().get_sort_keys("copied.safetensors")["date_created"] == 3000
        assert module.ExtraNetworksPage().get_sort_keys("copied.safetensors")["date_modified"] == 1000

    def test_where_no_creation_time_is_kept_forges_keys_stand(self, files):
        """Linux keeps no creation time, and its ctime is the last change to
        the file's metadata: nothing better to put there."""
        module = forge_module(files(birthtimes=False, windows=False))
        sort.install(module)

        assert module.ExtraNetworksPage().get_sort_keys("copied.safetensors") == {
            "date_created": 1000, "date_modified": 4000,
            "name": "copied.safetensors", "path": "copied.safetensors"}

    def test_a_file_that_cannot_be_read_and_keys_without_dates_are_left_as_forge_made_them(self, files):
        module = forge_module(files(birthtimes=True, windows=True))
        sort.install(module)

        assert module.ExtraNetworksPage().get_sort_keys("gone.safetensors") == {
            "date_created": 0, "date_modified": 0, "name": "gone.safetensors", "path": "gone.safetensors"}
        assert sort.corrected({"name": "x"}, "copied.safetensors") == {"name": "x"}
        assert sort.corrected(None, "copied.safetensors") is None

    def test_installed_once_on_the_class_every_page_inherits(self, files):
        module = forge_module(files(birthtimes=True, windows=True))
        original = module.ExtraNetworksPage.get_sort_keys

        assert sort.install(module) is True
        wrapped = module.ExtraNetworksPage.get_sort_keys
        assert sort.install(module) is True
        assert module.ExtraNetworksPage.get_sort_keys is wrapped and wrapped.__wrapped__ is original

        class LoraPage(module.ExtraNetworksPage):
            pass

        assert LoraPage().get_sort_keys("copied.safetensors")["date_created"] == 3000
        assert sort.install(types.SimpleNamespace()) is False

    def test_a_real_file_on_this_machine(self, tmp_path):
        """The wrapper against the platform's own ``os.stat``: a creation time
        where it keeps one, Forge's keys where it does not."""
        path = tmp_path / "real.safetensors"
        path.write_bytes(b"x")
        os.utime(path, (1_000_000, 1_000_000))
        stat = os.stat(path)
        module = forge_module({str(path): (stat.st_mtime, stat.st_ctime)})
        sort.install(module)

        keys = module.ExtraNetworksPage().get_sort_keys(str(path))
        born = getattr(stat, "st_birthtime", None)
        if born is None and os.name == "nt":
            born = stat.st_ctime
        if born is None:
            assert keys["date_created"] == 1_000_000 and keys["date_modified"] == int(stat.st_ctime)
        else:
            assert keys["date_created"] == int(born) and keys["date_modified"] == 1_000_000


class Opts:
    """Forge's options store as far as the default touches it."""

    class Label:
        def __init__(self, default):
            self.default = default

    def __init__(self, saved=None, *, forge=True):
        self.data = dict(saved or {})
        self.data_labels = ({sort.OPT_FIELD: Opts.Label("Path"), sort.OPT_ORDER: Opts.Label("Ascending")}
                            if forge else {})
        self.saves = 0

    def set(self, key, value):
        self.data[key] = value

    def save(self):
        self.saves += 1


class TestTheDefaultSort:
    def test_forges_defaults_become_newest_created_first_once(self):
        opts = Opts()

        record = sort.default_newest(opts, save=opts.save, now=lambda: 0)
        assert opts.data[sort.OPT_FIELD] == "Date Created" and opts.data[sort.OPT_ORDER] == "Descending"
        assert record["set"] is True and record["was"] == {sort.OPT_FIELD: "Path", sort.OPT_ORDER: "Ascending"}
        assert json.loads(opts.data[sort.OPT_RECORD]) == record and opts.saves == 1

        assert sort.default_newest(opts, save=opts.save) is None
        assert opts.saves == 1

    def test_forges_defaults_written_out_in_the_config_are_still_its_defaults(self):
        """The host writes every setting into its file at the first Apply, so
        a value equal to the default is no choice of the user's."""
        opts = Opts({sort.OPT_FIELD: "Path", sort.OPT_ORDER: "Ascending"})

        assert sort.default_newest(opts, save=opts.save)["set"] is True
        assert opts.data[sort.OPT_FIELD] == "Date Created"

    def test_a_choice_made_in_forges_settings_stands(self):
        opts = Opts({sort.OPT_FIELD: "Name", sort.OPT_ORDER: "Ascending"})

        record = sort.default_newest(opts, save=opts.save)
        assert record["set"] is False
        assert opts.data[sort.OPT_FIELD] == "Name" and opts.data[sort.OPT_ORDER] == "Ascending"
        assert sort.default_newest(opts, save=opts.save) is None

    def test_a_choice_made_afterwards_is_never_undone(self):
        opts = Opts()
        sort.default_newest(opts, save=opts.save)
        opts.set(sort.OPT_FIELD, "Path")
        opts.set(sort.OPT_ORDER, "Ascending")

        assert sort.default_newest(opts, save=opts.save) is None
        assert opts.data[sort.OPT_FIELD] == "Path" and opts.data[sort.OPT_ORDER] == "Ascending"

    def test_a_forge_without_the_settings_gets_nothing_and_no_record(self):
        opts = Opts(forge=False)

        assert sort.default_newest(opts, save=opts.save) is None
        assert opts.data == {} and opts.saves == 0

    def test_a_save_that_fails_is_logged_and_the_settings_still_set(self, caplog):
        opts = Opts()

        def broken():
            raise OSError("read-only")

        with caplog.at_level(logging.WARNING):
            assert sort.default_newest(opts, save=broken)["set"] is True
        assert opts.data[sort.OPT_ORDER] == "Descending"


class TestTheHost:
    def test_the_record_is_a_hidden_option(self, host):
        import model_chain  # noqa: F401

        info = host.shared.options_templates[sort.OPT_RECORD]
        assert info.section[0] is None and info.default == ""

    def test_the_hook_runs_before_the_page_is_built(self, host):
        """Forge renders the extra network pages while it builds the page."""
        import model_chain  # noqa: F401

        assert sort.on_before_ui in host.script_callbacks.registered["before_ui"]

    def test_the_hook_corrects_the_keys_and_sets_the_default(self, host, monkeypatch, files):
        module = forge_module(files(birthtimes=True, windows=True))
        monkeypatch.setitem(sys.modules, "modules.ui_extra_networks", module)
        monkeypatch.setattr(sys.modules["modules"], "ui_extra_networks", module, raising=False)
        opts = host.shared.opts
        monkeypatch.setattr(opts, "data_labels", Opts().data_labels, raising=False)
        saved = []
        monkeypatch.setattr(opts, "save", lambda filename: saved.append(filename), raising=False)

        sort.on_before_ui()

        assert module.ExtraNetworksPage().get_sort_keys("copied.safetensors")["date_created"] == 3000
        assert opts.data[sort.OPT_FIELD] == "Date Created" and opts.data[sort.OPT_ORDER] == "Descending"
        assert saved == [host.shared.config_filename]
