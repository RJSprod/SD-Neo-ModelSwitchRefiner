"""Stage 2 edit / reference conditioning (Krea 2, Anima, Flux.2 Klein).

Forge Neo lets some architectures take the input image as an *edit reference*
-- vision-conditioning the text encoder and concatenating reference latents --
instead of using it as an img2img starting point. Each is gated behind a global
Settings toggle, and the polarity is not consistent between them.
"""

from __future__ import annotations

import pytest

import mc_arch
import mc_infotext


class TestArchitectureTable:
    def test_krea2_supports_edit_and_needs_a_lora(self):
        arch = mc_arch.by_key("krea2")
        assert arch.supports_edit
        assert arch.edit_option == "krea2_do_reference"
        assert arch.edit_on_value is True
        assert arch.edit_needs_lora is True

    def test_anima_matches_the_krea2_opt_in_pattern(self):
        arch = mc_arch.by_key("anima")
        assert arch.edit_option == "anima_do_reference"
        assert arch.edit_on_value is True
        assert arch.edit_needs_lora is True

    def test_klein_opts_in_under_the_name_forge_gave_its_toggle(self):
        """Klein is reference-conditioned by default. Its option enables it and
        is on by default: Forge Neo renamed the opt-out ``klein_no_reference``
        to ``klein_do_reference`` on 2026-09-01, and the table carries the
        current name."""
        arch = mc_arch.by_key("flux2_9b")
        assert arch.edit_option == "klein_do_reference"
        assert arch.edit_on_value is True
        assert arch.edit_needs_lora is False
        assert arch.edit_is_default is True

    def test_plain_architectures_have_no_edit_mode(self):
        for key in ("sdxl", "sd15", "flux", "qwen", "chroma"):
            assert not mc_arch.by_key(key).supports_edit

    def test_unknown_has_no_edit_mode(self):
        assert not mc_arch.UNKNOWN.supports_edit

    def test_krea2_geometry(self):
        """VAE downscales 8x, transformer patchifies 2x2 -> 16px alignment."""
        arch = mc_arch.by_key("krea2")
        assert arch.alignment == 16
        assert arch.cfg == pytest.approx(1.0)


class TestEditOverride:
    def test_auto_leaves_the_global_setting_alone(self):
        for key in ("krea2", "anima", "flux2_9b"):
            assert mc_arch.edit_override(mc_arch.by_key(key), mc_arch.EDIT_AUTO) == {}

    def test_enable_turns_the_opt_in_toggle_on(self):
        override = mc_arch.edit_override(mc_arch.by_key("krea2"), mc_arch.EDIT_ENABLE)
        assert override == {"krea2_do_reference": True}

    def test_disable_turns_the_opt_in_toggle_off(self):
        override = mc_arch.edit_override(mc_arch.by_key("krea2"), mc_arch.EDIT_DISABLE)
        assert override == {"krea2_do_reference": False}

    def test_enable_turns_kleins_toggle_on(self, host):
        override = mc_arch.edit_override(mc_arch.by_key("flux2_9b"), mc_arch.EDIT_ENABLE)
        assert override == {"klein_do_reference": True}

    def test_disable_turns_kleins_toggle_off(self, host):
        override = mc_arch.edit_override(mc_arch.by_key("flux2_9b"), mc_arch.EDIT_DISABLE)
        assert override == {"klein_do_reference": False}

    def test_fibo2_is_overridden_through_its_own_toggle(self, host):
        fibo2 = mc_arch.by_key("fibo2")

        assert mc_arch.edit_override(fibo2, mc_arch.EDIT_ENABLE) == {"fibo2_do_reference": True}
        assert mc_arch.edit_override(fibo2, mc_arch.EDIT_DISABLE) == {"fibo2_do_reference": False}
        assert mc_arch.edit_override(fibo2, mc_arch.EDIT_AUTO) == {}

    def test_unsupported_architecture_produces_no_override(self):
        assert mc_arch.edit_override(mc_arch.by_key("sdxl"), mc_arch.EDIT_ENABLE) == {}
        assert mc_arch.edit_override(mc_arch.UNKNOWN, mc_arch.EDIT_ENABLE) == {}


class TestEditIsActive:
    def test_explicit_modes_do_not_consult_the_global_setting(self, host):
        arch = mc_arch.by_key("krea2")
        host.shared.opts.krea2_do_reference = False
        assert mc_arch.edit_is_active(arch, mc_arch.EDIT_ENABLE) is True

        host.shared.opts.krea2_do_reference = True
        assert mc_arch.edit_is_active(arch, mc_arch.EDIT_DISABLE) is False

    def test_auto_follows_the_global_setting(self, host):
        arch = mc_arch.by_key("krea2")
        host.shared.opts.krea2_do_reference = True
        assert mc_arch.edit_is_active(arch, mc_arch.EDIT_AUTO) is True

        host.shared.opts.krea2_do_reference = False
        assert mc_arch.edit_is_active(arch, mc_arch.EDIT_AUTO) is False

    def test_auto_follows_kleins_toggle(self, host):
        arch = mc_arch.by_key("flux2_9b")
        host.shared.opts.klein_do_reference = True
        assert mc_arch.edit_is_active(arch, mc_arch.EDIT_AUTO) is True

        host.shared.opts.klein_do_reference = False
        assert mc_arch.edit_is_active(arch, mc_arch.EDIT_AUTO) is False

    def test_unsupported_architecture_is_never_active(self, host):
        assert mc_arch.edit_is_active(mc_arch.by_key("sdxl"), mc_arch.EDIT_ENABLE) is False

    def test_auto_follows_fibo2s_toggle(self, host):
        """On by default in the FIBO 2 extension, so an Auto Stage 2 on FIBO 2 is
        an edit unless the user turned it off."""
        fibo2 = mc_arch.by_key("fibo2")
        host.shared.opts.fibo2_do_reference = True
        assert mc_arch.edit_is_active(fibo2, mc_arch.EDIT_AUTO) is True

        host.shared.opts.fibo2_do_reference = False
        assert mc_arch.edit_is_active(fibo2, mc_arch.EDIT_AUTO) is False


class TestTheRenamedKleinToggle:
    """Forge Neo's "streamline" commit of 2026-09-01 renamed Klein's toggle from
    ``klein_no_reference`` (True disabled references) to ``klein_do_reference``
    (True enables them). A stale name is not harmless: the host applies a Stage 2
    override through ``Options.set``, which raises ``KeyError`` on a key it does
    not register, so a Klein Stage 2 with Enable or Disable, or with reference
    images, never started. The table names the current option, and the host
    that registers only the old one is answered in the old name and polarity."""

    def older_host(self, host):
        del host.shared.opts.klein_do_reference
        host.shared.opts.klein_no_reference = False

    def test_the_current_host_is_overridden_by_its_current_name(self, host):
        assert mc_arch.edit_toggle(mc_arch.by_key("flux2_9b")) == ("klein_do_reference", True)
        assert mc_arch.edit_override(mc_arch.by_key("flux2_4b"), mc_arch.EDIT_ENABLE) == {
            "klein_do_reference": True}

    def test_an_older_host_is_overridden_by_the_name_it_registers(self, host):
        self.older_host(host)
        klein = mc_arch.by_key("flux2_9b")

        assert mc_arch.edit_toggle(klein) == ("klein_no_reference", False)
        assert mc_arch.edit_override(klein, mc_arch.EDIT_ENABLE) == {"klein_no_reference": False}
        assert mc_arch.edit_override(klein, mc_arch.EDIT_DISABLE) == {"klein_no_reference": True}

    def test_auto_reads_the_older_hosts_toggle_with_its_polarity(self, host):
        self.older_host(host)
        klein = mc_arch.by_key("flux2_9b")

        host.shared.opts.klein_no_reference = False
        assert mc_arch.edit_is_active(klein, mc_arch.EDIT_AUTO) is True
        host.shared.opts.klein_no_reference = True
        assert mc_arch.edit_is_active(klein, mc_arch.EDIT_AUTO) is False

    def test_the_hosts_registry_decides_not_a_stray_attribute(self, host):
        """The real host answers ``Options.set`` from ``data_labels``; an
        attribute written on the object is not a registered option."""
        host.shared.opts.data_labels = {"klein_no_reference": object()}

        assert mc_arch.edit_toggle(mc_arch.by_key("flux2_9b")) == ("klein_no_reference", False)

        host.shared.opts.data_labels = {"klein_do_reference": object()}
        assert mc_arch.edit_toggle(mc_arch.by_key("flux2_9b")) == ("klein_do_reference", True)

    def test_a_host_with_neither_name_gets_the_current_one(self, host):
        """So the error it raises, if it raises, names the option that exists
        on every Forge Neo from the rename on."""
        del host.shared.opts.klein_do_reference

        assert mc_arch.edit_toggle(mc_arch.by_key("flux2_9b")) == ("klein_do_reference", True)

    def test_an_option_that_was_never_renamed_is_left_alone(self, host):
        del host.shared.opts.krea2_do_reference

        assert mc_arch.edit_toggle(mc_arch.by_key("krea2")) == ("krea2_do_reference", True)
        assert mc_arch.edit_toggle(mc_arch.by_key("sdxl")) == (None, True)


class TestInfotext:
    def base(self, **overrides):
        params = dict(
            target="krea2.safetensors", prompt_mode="Replace", prompt="a lake",
            negative="", styles=[], seed_mode="Inherit", seed_offset=0, fixed_seed=-1,
            cfg=1.0, steps=8, sampler=mc_infotext.INHERIT,
            scheduler=mc_infotext.INHERIT, denoise=1.0,
            size_multiplier=1.0, stage1_size="1024x1024",
        )
        params.update(overrides)
        return mc_infotext.build_params(**params)

    def test_auto_is_omitted_so_old_infotext_is_unchanged(self):
        assert mc_infotext.EDIT_MODE not in self.base(edit_mode="Auto")
        assert mc_infotext.EDIT_MODE not in self.base()

    def test_explicit_modes_are_recorded(self):
        assert self.base(edit_mode="Enable")[mc_infotext.EDIT_MODE] == "Enable"
        assert self.base(edit_mode="Disable")[mc_infotext.EDIT_MODE] == "Disable"

    def test_the_key_is_declared_for_pasting(self):
        assert mc_infotext.EDIT_MODE in mc_infotext.paste_field_names()
