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

    def test_klein_opts_out_instead_of_in(self):
        """Klein is reference-conditioned by default; its option disables it."""
        arch = mc_arch.by_key("flux2_9b")
        assert arch.edit_option == "klein_no_reference"
        assert arch.edit_on_value is False
        assert arch.edit_needs_lora is False

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

    def test_enable_clears_the_opt_out_toggle(self):
        """Enabling Klein references means setting klein_no_reference to False."""
        override = mc_arch.edit_override(mc_arch.by_key("flux2_9b"), mc_arch.EDIT_ENABLE)
        assert override == {"klein_no_reference": False}

    def test_disable_sets_the_opt_out_toggle(self):
        override = mc_arch.edit_override(mc_arch.by_key("flux2_9b"), mc_arch.EDIT_DISABLE)
        assert override == {"klein_no_reference": True}

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

    def test_auto_respects_the_inverted_klein_polarity(self, host):
        arch = mc_arch.by_key("flux2_9b")
        host.shared.opts.klein_no_reference = False
        assert mc_arch.edit_is_active(arch, mc_arch.EDIT_AUTO) is True

        host.shared.opts.klein_no_reference = True
        assert mc_arch.edit_is_active(arch, mc_arch.EDIT_AUTO) is False

    def test_unsupported_architecture_is_never_active(self, host):
        assert mc_arch.edit_is_active(mc_arch.by_key("sdxl"), mc_arch.EDIT_ENABLE) is False


class TestInfotext:
    def base(self, **overrides):
        params = dict(
            target="krea2.safetensors", prompt_mode="Replace", prompt="a lake",
            negative="", styles=[], seed_mode="Inherit", seed_offset=0, fixed_seed=-1,
            cfg=1.0, steps=8, sampler=mc_infotext.INHERIT_SAMPLER, denoise=1.0,
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
