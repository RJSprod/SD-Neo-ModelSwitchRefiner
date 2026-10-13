"""The theme's foundation: its settings, the one-time import of the user's
Lobe settings, the Gradio theme object, and the hook that installs it.

The palette is a Gradio theme object because that is the host's own mechanism
for a theme (``gr.Blocks(theme=shared.gradio_theme)``): every attribute
becomes a CSS variable in both modes, with nothing fought over in the cascade.
Gradio is not installed here, so its ``themes.Base`` is stood in for by a
class with the same contract -- every variable an attribute after
construction, ``set()`` by keyword, an unknown name refused -- and the builder
is held to the names Gradio 4 knows.

Nothing in the theme comes from sd-webui-lobe-theme; what *is* read from its
folder is the user's own settings file, once, and that reading is the whole of
``TestImportingTheUsersLobeSettings``: the file the user sent, with its stray
``detail: "Not Found"`` (a 404 body the theme once saved as a setting), has to
start our switches where theirs were and touch nothing they have set here.
"""

from __future__ import annotations

import json
import os
import types

import pytest

import mc_theme

# The user's lobe_theme_config.json, as sent on 2026-10-13.
USER_CONFIG = {
    "confirmPageUnload": False,
    "enableExtraNetworkSidebar": True,
    "enableHighlight": False,
    "enableImageInfo": True,
    "enableSidebar": True,
    "enableWebFont": True,
    "extraNetworkCardSize": 86,
    "extraNetworkFixedMode": "fixed",
    "extraNetworkSidebarExpand": False,
    "extraNetworkSidebarWidth": 340,
    "i18n": "en_US",
    "layoutHideFooter": True,
    "layoutSplitPreview": True,
    "liteAnimation": True,
    "logoCustomTitle": "",
    "logoCustomUrl": "",
    "logoType": "lobe",
    "neutralColor": "",
    "primaryColor": "",
    "promptEditor": False,
    "promptTextareaType": "resizable",
    "sidebarExpand": False,
    "sidebarFixedMode": "fixed",
    "sidebarWidth": 280,
    "svgIcon": True,
    "detail": "Not Found",
}

DEFAULTS = {
    mc_theme.OPT_ENABLED: False,
    mc_theme.OPT_SPLIT: True,
    mc_theme.OPT_QUICK: "collapsed",
    mc_theme.OPT_QUICK_WIDTH: 280,
    mc_theme.OPT_PROMPTS: True,
    mc_theme.OPT_ICONS: True,
    mc_theme.OPT_TITLE: "Forge",
    mc_theme.OPT_TAB_LABELS: "as_forge",
    mc_theme.OPT_CONFIRM_UNLOAD: False,
    mc_theme.OPT_ACCENT: "",
    mc_theme.OPT_FONT: "",
    mc_theme.OPT_NETWORKS: "collapsed",
    mc_theme.OPT_NETWORKS_WIDTH: 340,
    mc_theme.OPT_NETWORKS_CARD: 120,
    mc_theme.OPT_IMAGE_INFO: True,
    mc_theme.OPT_HIDE_FOOTER: True,
}


# --------------------------------------------------------------------------- #
# The settings
# --------------------------------------------------------------------------- #


class TestTheSettings:
    def test_the_section_holds_the_switches_the_user_had_with_their_defaults(self, host):
        import model_chain  # noqa: F401 -- registers the sections

        registered = host.shared.options_templates
        for name, default in DEFAULTS.items():
            if name == mc_theme.OPT_HIDE_FOOTER:
                continue
            assert name in registered, name
            assert registered[name].default == default, name
            assert registered[name].section == mc_theme.SECTION, name

    def test_the_theme_is_off_until_asked_for(self, host):
        """An LLM and voice extension must not bring a theme with it."""
        import model_chain  # noqa: F401

        assert host.shared.options_templates[mc_theme.OPT_ENABLED].default is False

    def test_the_import_record_is_a_hidden_option(self, host):
        """The host draws no control for an option whose section id is None;
        the record exists, can be read and saved, and is never ticked."""
        import model_chain  # noqa: F401

        info = host.shared.options_templates[mc_theme.OPT_IMPORT]
        assert info.section[0] is None
        assert info.default == ""

    def test_the_footer_switch_is_named_as_model_chain_names_it(self, host):
        import model_chain

        assert mc_theme.OPT_HIDE_FOOTER == model_chain.OPT_HIDE_FOOTER

    def test_the_hook_runs_before_the_page_is_built(self, host):
        """The theme object is read when the host builds the page, so the only
        hook that can put ours in place is the one before that."""
        import model_chain  # noqa: F401

        assert mc_theme.on_before_ui in host.script_callbacks.registered["before_ui"]


# --------------------------------------------------------------------------- #
# The palette
# --------------------------------------------------------------------------- #


class TestThePalette:
    def test_night_is_a_black_page_with_surfaces_a_step_above_it(self):
        values = mc_theme.palette()

        assert values["body_background_fill_dark"] == "#000000"
        assert values["background_fill_primary_dark"] == "#000000"
        assert values["block_background_fill_dark"] not in ("#000000", "#ffffff")
        assert values["block_border_color_dark"] not in ("#000000", "#ffffff")
        # And day is a page, not a black one.
        assert values["body_background_fill"] != "#000000"
        assert values["block_background_fill"] == "#ffffff"

    def test_the_accent_is_white_at_night_and_black_by_day_unless_chosen(self):
        """The neutral primary the user's page had: the Generate button that
        was a white block, which the quiet night mode exists for."""
        values = mc_theme.palette()

        assert values["button_primary_background_fill_dark"] == "#ffffff"
        assert values["button_primary_text_color_dark"] == "#000000"
        assert values["button_primary_background_fill"] == "#000000"
        assert values["button_primary_text_color"] == "#ffffff"
        assert values["color_accent"] == "#ffffff"
        assert mc_theme.accents() == ("#000000", "#ffffff")

    def test_a_chosen_accent_reaches_every_place_the_accent_goes_in_both_modes(self):
        values = mc_theme.palette("#2F6FED")

        for name in ("button_primary_background_fill", "button_primary_border_color",
                     "slider_color", "checkbox_background_color_selected",
                     "checkbox_border_color_selected", "loader_color"):
            assert values[name] == "#2f6fed", name
            assert values[name + "_dark"] == "#2f6fed", name
        assert values["color_accent"] == "#2f6fed"
        assert values["button_primary_text_color"] == "#ffffff"
        assert values["button_primary_text_color_dark"] == "#ffffff"
        assert mc_theme.accents("#2F6FED") == ("#2f6fed", "#2f6fed")

    def test_the_text_on_the_accent_is_whichever_reads(self):
        assert mc_theme.contrast_text("#ffffff") == "#000000"
        assert mc_theme.contrast_text("#000000") == "#ffffff"
        assert mc_theme.contrast_text("#f5d90a") == "#000000"
        assert mc_theme.contrast_text("#2f6fed") == "#ffffff"

    def test_a_colour_that_is_not_six_hex_digits_is_the_automatic_one(self):
        assert mc_theme.valid_accent("blue") == ""
        assert mc_theme.valid_accent("#abc") == ""
        assert mc_theme.valid_accent("2f6fed") == ""
        assert mc_theme.valid_accent(None) == ""
        assert mc_theme.valid_accent(" #ABCDEF ") == "#abcdef"

    def test_every_name_is_one_gradio_knows(self):
        """A name outside the set never reaches set(): Gradio refuses an
        unknown keyword outright, and a theme that raises before the page is
        built is a page with no theme at all."""
        assert set(mc_theme.palette()) <= mc_theme.GRADIO_THEME_VARIABLES

    def test_every_dark_value_has_a_light_one(self):
        values = mc_theme.palette()

        for name in values:
            if name.endswith("_dark"):
                assert name[:-5] in values, name

    def test_nothing_casts_a_shadow_but_a_list_that_comes_off_the_page(self):
        values = mc_theme.palette()

        for name, value in values.items():
            if "shadow" in name and name not in ("shadow_drop_lg", "shadow_drop_lg_dark",
                                                 "shadow_spread"):
                assert value == "none", name
        assert values["shadow_drop_lg"] != "none"


# --------------------------------------------------------------------------- #
# The theme object
# --------------------------------------------------------------------------- #


class FakeBase:
    """Gradio 4's ``themes.Base``, as far as the builder touches it: every
    variable is an attribute after construction, ``set()`` takes them by
    keyword, and a name it does not know is refused."""

    KNOWN = mc_theme.GRADIO_THEME_VARIABLES

    def __init__(self, *, font=(), font_mono=()):
        self.font = list(font)
        self.font_mono = list(font_mono)
        for name in self.KNOWN:
            setattr(self, name, None)

    def set(self, **kwargs):
        for name, value in kwargs.items():
            if name not in self.KNOWN:
                raise TypeError(f"set() got an unexpected keyword argument '{name}'")
            setattr(self, name, value)


def fake_gr(base=FakeBase):
    return types.SimpleNamespace(themes=types.SimpleNamespace(Base=base))


class TestTheThemeObject:
    def test_it_carries_the_palette_and_is_marked_as_ours(self):
        theme, dropped = mc_theme.build_theme(fake_gr())

        assert dropped == ()
        assert theme.body_background_fill_dark == "#000000"
        assert theme.button_primary_background_fill_dark == "#ffffff"
        assert mc_theme.is_ours(theme)
        assert not mc_theme.is_ours(object())

    def test_both_accents_ride_along_as_attributes_for_the_stylesheet(self):
        """Gradio turns any attribute into a CSS variable; the stylesheet
        reads the day accent off these under the light theme."""
        theme, _ = mc_theme.build_theme(fake_gr(), accent="#2f6fed")

        assert theme.mc_theme_accent_day == "#2f6fed"
        assert theme.mc_theme_accent_night == "#2f6fed"
        auto, _ = mc_theme.build_theme(fake_gr())
        assert (auto.mc_theme_accent_day, auto.mc_theme_accent_night) == ("#000000", "#ffffff")

    def test_a_name_this_gradio_does_not_know_is_dropped_not_raised(self):
        class OlderBase(FakeBase):
            KNOWN = mc_theme.GRADIO_THEME_VARIABLES - {"shadow_spread", "chatbot_code_background_color_dark"}

        theme, dropped = mc_theme.build_theme(fake_gr(OlderBase))

        assert dropped == ("chatbot_code_background_color_dark", "shadow_spread")
        assert theme.body_background_fill_dark == "#000000"

    def test_the_typeface_goes_first_and_the_system_stack_follows(self):
        theme, _ = mc_theme.build_theme(fake_gr(), font=" Inter ")

        assert theme.font[0] == "Inter"
        assert theme.font[1:] == list(mc_theme.FONT_STACK)
        plain, _ = mc_theme.build_theme(fake_gr())
        assert plain.font == list(mc_theme.FONT_STACK)
        assert plain.font_mono == list(mc_theme.MONO_STACK)


def shared_with(enabled: bool, accent: str = "", font: str = ""):
    opts = types.SimpleNamespace(**{mc_theme.OPT_ENABLED: enabled, mc_theme.OPT_ACCENT: accent,
                                    mc_theme.OPT_FONT: font})
    host_theme = types.SimpleNamespace(name="forge", sd_webui_modal_lightbox_toolbar_opacity=0.9,
                                       sd_webui_modal_lightbox_icon_opacity=1.0)
    return types.SimpleNamespace(opts=opts, gradio_theme=host_theme, config_filename="config.json"), host_theme


class TestInstalling:
    def test_on_puts_ours_in_the_hosts_place_and_keeps_the_lightbox_values(self):
        shared, host_theme = shared_with(True)

        assert mc_theme.install(shared, fake_gr()) is True
        assert mc_theme.is_ours(shared.gradio_theme)
        assert shared.gradio_theme.sd_webui_modal_lightbox_toolbar_opacity == 0.9
        assert shared.gradio_theme.sd_webui_modal_lightbox_icon_opacity == 1.0

    def test_off_puts_the_hosts_own_back(self):
        """The host builds its theme object once; without this, turning the
        theme off and reloading the UI would keep our palette until a restart."""
        shared, host_theme = shared_with(True)
        mc_theme.install(shared, fake_gr())
        shared.opts.model_chain_theme = False

        assert mc_theme.install(shared, fake_gr()) is False
        assert shared.gradio_theme is host_theme

    def test_off_with_the_hosts_theme_in_place_leaves_it_alone(self):
        shared, host_theme = shared_with(False)

        assert mc_theme.install(shared, fake_gr()) is False
        assert shared.gradio_theme is host_theme

    def test_a_reinstall_remembers_the_latest_host_theme(self):
        """The host's own setting can change between reloads; what comes back
        when ours goes off is the one it replaced last."""
        shared, first = shared_with(True)
        mc_theme.install(shared, fake_gr())
        second = types.SimpleNamespace(name="hub")
        shared.gradio_theme = second
        mc_theme.install(shared, fake_gr())
        shared.opts.model_chain_theme = False
        mc_theme.install(shared, fake_gr())

        assert shared.gradio_theme is second

    def test_the_hook_installs_through_the_hosts_modules(self, host, monkeypatch):
        import sys

        import model_chain  # noqa: F401

        gradio = sys.modules["gradio"]
        monkeypatch.setattr(gradio, "themes", types.SimpleNamespace(Base=FakeBase), raising=False)
        host.shared.gradio_theme = types.SimpleNamespace(name="forge")
        host.shared.opts.set(mc_theme.OPT_ENABLED, True)

        mc_theme.on_before_ui()

        assert mc_theme.is_ours(host.shared.gradio_theme)

    def test_the_hook_survives_a_gradio_without_themes(self, host, monkeypatch):
        """A hook that raises before the page is built is a page that is
        never built."""
        import sys

        import model_chain  # noqa: F401

        gradio = sys.modules["gradio"]
        monkeypatch.delattr(gradio, "themes", raising=False)
        host.shared.gradio_theme = types.SimpleNamespace(name="forge")
        host.shared.opts.set(mc_theme.OPT_ENABLED, True)

        mc_theme.on_before_ui()

        assert host.shared.gradio_theme.name == "forge"


# --------------------------------------------------------------------------- #
# Importing the user's Lobe settings, once
# --------------------------------------------------------------------------- #


class Opts:
    """The host's options store, as far as the import touches it: ``data``
    holds what has been saved, ``set`` writes, ``save`` is counted."""

    def __init__(self, saved=None):
        self.data = dict(saved or {})
        self.saves = 0

    def set(self, key, value):
        self.data[key] = value
        setattr(self, key, value)

    def save(self, filename=None):
        self.saves += 1


def lobe_folder(tmp_path, name="sd-webui-lobe-theme", config=USER_CONFIG, mtime=None):
    folder = tmp_path / "extensions" / name
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / mc_theme.LOBE_CONFIG_NAME
    path.write_text(json.dumps(config), encoding="utf-8")
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def run_import(tmp_path, opts=None):
    opts = opts or Opts()
    record = mc_theme.import_lobe_settings(opts, tmp_path / "extensions", defaults=DEFAULTS,
                                           save=opts.save, now=lambda: 1_700_000_000)
    return opts, record


class TestImportingTheUsersLobeSettings:
    def test_the_users_file_starts_the_switches_where_theirs_were(self, tmp_path):
        path = lobe_folder(tmp_path)
        opts, record = run_import(tmp_path)

        assert record["applied"] == {
            mc_theme.OPT_SPLIT: True,
            mc_theme.OPT_QUICK: "collapsed",
            mc_theme.OPT_QUICK_WIDTH: 280,
            mc_theme.OPT_NETWORKS: "collapsed",
            mc_theme.OPT_NETWORKS_WIDTH: 340,
            mc_theme.OPT_NETWORKS_CARD: 86,
            mc_theme.OPT_PROMPTS: True,
            mc_theme.OPT_ICONS: True,
            mc_theme.OPT_CONFIRM_UNLOAD: False,
            mc_theme.OPT_HIDE_FOOTER: True,
            mc_theme.OPT_IMAGE_INFO: True,
        }
        assert opts.data[mc_theme.OPT_QUICK] == "collapsed"
        assert opts.data[mc_theme.OPT_QUICK_WIDTH] == 280
        assert opts.data[mc_theme.OPT_NETWORKS_CARD] == 86
        assert record["from"] == str(path)
        assert record["kept"] == []
        assert record["version"] == mc_theme.IMPORT_VERSION
        assert opts.saves == 1

    def test_the_theme_itself_is_not_switched_on_by_the_import(self, tmp_path):
        """The user turns it on; a theme that appears because a file was
        found would surprise everybody else."""
        lobe_folder(tmp_path)
        opts, record = run_import(tmp_path)

        assert mc_theme.OPT_ENABLED not in record["applied"]
        assert mc_theme.OPT_ENABLED not in opts.data

    def test_what_is_not_built_is_said_so_and_the_file_rides_in_the_record(self, tmp_path):
        """Lobe's floating panels, its prompt editor and the rest are not
        built; the raw file rides in ``raw`` so a later round needs no file."""
        lobe_folder(tmp_path)
        _, record = run_import(tmp_path)

        for key in ("extraNetworkFixedMode", "sidebarFixedMode", "promptEditor", "enableHighlight",
                    "i18n", "liteAnimation", "logoType", "enableWebFont"):
            assert key in record["ignored"], key
        for key in ("enableExtraNetworkSidebar", "extraNetworkSidebarExpand",
                    "extraNetworkSidebarWidth", "extraNetworkCardSize", "enableImageInfo"):
            assert key not in record["ignored"], key
        assert record["raw"]["extraNetworkSidebarWidth"] == 340
        assert record["raw"]["enableImageInfo"] is True

    def test_a_record_from_the_first_build_is_read_again_for_the_drawer_and_the_panel(self, tmp_path):
        """The first build kept the drawer's and the panel's values in the
        record's ``raw``; this one reads them from there, without the file,
        writes the switches still at their defaults, and moves the record to
        its version. A switch the user has set here since is theirs."""
        first = {
            "from": str(tmp_path / "gone" / mc_theme.LOBE_CONFIG_NAME),
            "modified": 1.0, "when": "2026-10-13 10:00:00",
            "applied": {mc_theme.OPT_SPLIT: True, mc_theme.OPT_QUICK: "collapsed",
                        mc_theme.OPT_QUICK_WIDTH: 280, mc_theme.OPT_PROMPTS: True,
                        mc_theme.OPT_ICONS: True, mc_theme.OPT_CONFIRM_UNLOAD: False,
                        mc_theme.OPT_HIDE_FOOTER: True},
            "kept": [mc_theme.OPT_TITLE],
            "ignored": ["enableExtraNetworkSidebar", "extraNetworkSidebarExpand",
                        "extraNetworkSidebarWidth", "extraNetworkCardSize", "enableImageInfo", "detail"],
            "raw": dict(USER_CONFIG, extraNetworkSidebarExpand=True),
        }
        opts = Opts({mc_theme.OPT_IMPORT: json.dumps(first), mc_theme.OPT_TITLE: "SD",
                     mc_theme.OPT_NETWORKS_WIDTH: 500})
        opts, record = run_import(tmp_path, opts)   # no Lobe folder anywhere

        assert record is not None
        assert record["version"] == mc_theme.IMPORT_VERSION
        assert opts.data[mc_theme.OPT_NETWORKS] == "open"
        assert opts.data[mc_theme.OPT_NETWORKS_CARD] == 86
        assert opts.data[mc_theme.OPT_IMAGE_INFO] is True
        assert opts.data[mc_theme.OPT_NETWORKS_WIDTH] == 500
        assert record["kept"] == [mc_theme.OPT_TITLE, mc_theme.OPT_NETWORKS_WIDTH]
        assert record["applied"][mc_theme.OPT_NETWORKS] == "open"
        assert record["applied"][mc_theme.OPT_QUICK] == "collapsed"
        assert "enableImageInfo" not in record["ignored"]
        assert "again" in record
        assert opts.saves == 1
        # The earlier switches are not written again.
        assert mc_theme.OPT_SPLIT not in opts.data

        again = mc_theme.import_lobe_settings(opts, tmp_path / "extensions", defaults=DEFAULTS,
                                              save=opts.save)
        assert again is None
        assert opts.saves == 1

    def test_a_record_with_nothing_to_read_again_is_left_as_it_is(self, tmp_path):
        first = {"from": "x", "applied": {}, "kept": [], "ignored": [], "raw": "not a dict"}
        opts = Opts({mc_theme.OPT_IMPORT: json.dumps(first)})
        opts, record = run_import(tmp_path, opts)

        assert record is None
        assert set(opts.data) == {mc_theme.OPT_IMPORT}
        assert opts.saves == 0

    def test_the_stray_404_body_falls_through(self, tmp_path):
        lobe_folder(tmp_path)
        opts, record = run_import(tmp_path)

        assert "detail" in record["ignored"]
        assert "detail" not in opts.data

    def test_it_runs_once(self, tmp_path):
        lobe_folder(tmp_path)
        opts, first = run_import(tmp_path)
        again = mc_theme.import_lobe_settings(opts, tmp_path / "extensions", defaults=DEFAULTS,
                                              save=opts.save)

        assert first is not None
        assert again is None
        assert opts.saves == 1
        assert mc_theme.import_record(opts)["from"] == first["from"]

    def test_a_switch_the_user_already_set_here_is_theirs(self, tmp_path):
        lobe_folder(tmp_path)
        opts, record = run_import(tmp_path, Opts({mc_theme.OPT_QUICK: "open",
                                                  mc_theme.OPT_ICONS: True}))

        assert opts.data[mc_theme.OPT_QUICK] == "open"
        assert record["kept"] == [mc_theme.OPT_QUICK]
        # Saved at its default counts as never touched, and is written.
        assert mc_theme.OPT_ICONS in record["applied"]

    def test_nothing_to_import_without_a_lobe_folder(self, tmp_path):
        (tmp_path / "extensions" / "something-else").mkdir(parents=True)
        opts, record = run_import(tmp_path)

        assert record is None
        assert opts.data == {}
        assert opts.saves == 0

    def test_nothing_to_import_without_an_extensions_folder(self, tmp_path):
        opts, record = run_import(tmp_path)

        assert record is None
        assert mc_theme.import_lobe_settings(opts, None, defaults=DEFAULTS) is None

    def test_the_newest_file_wins(self, tmp_path):
        old = lobe_folder(tmp_path, "lobe-old", dict(USER_CONFIG, sidebarWidth=300), mtime=1_000)
        new = lobe_folder(tmp_path, "lobe-new", dict(USER_CONFIG, sidebarWidth=400), mtime=2_000)
        opts, record = run_import(tmp_path)

        assert record["from"] == str(new)
        assert opts.data[mc_theme.OPT_QUICK_WIDTH] == 400
        assert str(old) != record["from"]

    def test_a_broken_file_is_left_alone_and_nothing_is_marked(self, tmp_path):
        path = lobe_folder(tmp_path)
        path.write_text("{not json", encoding="utf-8")
        opts, record = run_import(tmp_path)

        assert record is None
        assert opts.data == {}

        path.write_text("[1, 2]", encoding="utf-8")
        assert mc_theme.import_lobe_settings(opts, tmp_path / "extensions", defaults=DEFAULTS) is None

    def test_a_title_and_a_named_colour_translate(self, tmp_path):
        lobe_folder(tmp_path, config=dict(USER_CONFIG, logoCustomTitle=" SD ", primaryColor="blue"))
        opts, record = run_import(tmp_path)

        assert opts.data[mc_theme.OPT_TITLE] == "SD"
        assert opts.data[mc_theme.OPT_ACCENT] == mc_theme.NAMED_ACCENTS["blue"]
        assert "primaryColor" not in record["ignored"]

    def test_a_colour_with_no_name_here_is_ignored_and_said_so(self, tmp_path):
        lobe_folder(tmp_path, config=dict(USER_CONFIG, primaryColor="chartreuse"))
        opts, record = run_import(tmp_path)

        assert mc_theme.OPT_ACCENT not in opts.data
        assert "primaryColor" in record["ignored"]

    def test_the_sidebar_switches_become_the_drawer_mode(self):
        applied, _ = mc_theme.translate({"enableSidebar": False, "sidebarExpand": True})
        assert applied[mc_theme.OPT_QUICK] == "off"
        applied, _ = mc_theme.translate({"enableSidebar": True, "sidebarExpand": True})
        assert applied[mc_theme.OPT_QUICK] == "open"
        applied, _ = mc_theme.translate({"enableSidebar": True, "sidebarExpand": False})
        assert applied[mc_theme.OPT_QUICK] == "collapsed"
        applied, _ = mc_theme.translate({"sidebarWidth": 50})
        assert applied[mc_theme.OPT_QUICK_WIDTH] == mc_theme.QUICK_WIDTH_MIN
        applied, _ = mc_theme.translate({"sidebarWidth": 9000})
        assert applied[mc_theme.OPT_QUICK_WIDTH] == mc_theme.QUICK_WIDTH_MAX

    def test_the_extra_network_switches_become_the_drawer_mode_and_its_sizes(self):
        applied, _ = mc_theme.translate({"enableExtraNetworkSidebar": False, "extraNetworkSidebarExpand": True})
        assert applied[mc_theme.OPT_NETWORKS] == "off"
        applied, _ = mc_theme.translate({"enableExtraNetworkSidebar": True, "extraNetworkSidebarExpand": True})
        assert applied[mc_theme.OPT_NETWORKS] == "open"
        applied, _ = mc_theme.translate({"enableExtraNetworkSidebar": True, "extraNetworkSidebarExpand": False})
        assert applied[mc_theme.OPT_NETWORKS] == "collapsed"
        applied, _ = mc_theme.translate({"enableExtraNetworkSidebar": True})
        assert applied[mc_theme.OPT_NETWORKS] == "collapsed"
        applied, ignored = mc_theme.translate({"extraNetworkSidebarExpand": True})
        assert mc_theme.OPT_NETWORKS not in applied and "extraNetworkSidebarExpand" in ignored
        applied, _ = mc_theme.translate({"extraNetworkSidebarWidth": 50, "extraNetworkCardSize": 9000})
        assert applied[mc_theme.OPT_NETWORKS_WIDTH] == mc_theme.NETWORKS_WIDTH_MIN
        assert applied[mc_theme.OPT_NETWORKS_CARD] == mc_theme.NETWORKS_CARD_MAX
        applied, _ = mc_theme.translate({"enableImageInfo": False})
        assert applied[mc_theme.OPT_IMAGE_INFO] is False
        applied, ignored = mc_theme.translate({"extraNetworkSidebarWidth": True, "enableImageInfo": "yes"})
        assert applied == {} and set(ignored) == {"extraNetworkSidebarWidth", "enableImageInfo"}

    def test_a_key_of_the_wrong_shape_is_ignored_like_an_unknown_one(self):
        applied, ignored = mc_theme.translate({"layoutSplitPreview": "yes", "sidebarWidth": "wide",
                                               "promptTextareaType": "huge", "svgIcon": 1})

        assert applied == {}
        assert set(ignored) == {"layoutSplitPreview", "sidebarWidth", "promptTextareaType", "svgIcon"}

    def test_a_scrolling_prompt_box_translates_to_the_switch_off(self):
        applied, _ = mc_theme.translate({"promptTextareaType": "scroll"})

        assert applied[mc_theme.OPT_PROMPTS] is False

    def test_the_hook_imports_through_the_hosts_modules(self, host, monkeypatch, tmp_path):
        """Through the real path: the record lands in the host's options and
        the switches are set there."""
        import sys

        import model_chain  # noqa: F401

        lobe_folder(tmp_path, config=dict(USER_CONFIG, sidebarExpand=True))
        monkeypatch.setattr(mc_theme, "extensions_dir", lambda: tmp_path / "extensions")
        gradio = sys.modules["gradio"]
        monkeypatch.setattr(gradio, "themes", types.SimpleNamespace(Base=FakeBase), raising=False)
        host.shared.gradio_theme = types.SimpleNamespace(name="forge")

        mc_theme.on_before_ui()

        assert host.shared.opts.data[mc_theme.OPT_QUICK] == "open"
        assert mc_theme.import_record(host.shared.opts)["applied"][mc_theme.OPT_QUICK] == "open"
        # Off by default, so the host's theme stays.
        assert host.shared.gradio_theme.name == "forge"
