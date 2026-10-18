"""Counting each image once in Forge's VRAM request at CFG 1, pinned to the Forge build.

Forge's ``sampling_prepare`` sizes the room it asks for beside the weights from
a latent with the batch doubled, for a negative pass that never runs at CFG 1,
and reads a module constant (``shared.batch_cond_uncond``) to do it. The
override flips that constant for the moment the host reads it, only on a CFG 1
pass with single-part conditioning, only while the host still reads it there,
and only on the Forge build the setting was turned on with.
"""

from __future__ import annotations

import json
import logging
import sys
import types

import pytest

import mc_batch_once
import mc_memory
from test_orchestration import DEFAULTS, UI_ORDER, make_p

BUILD = "neo 2.30 @ 1a2b3c4d5e"


def _module_reading_the_flag():
    module = types.ModuleType("backend.sampling.sampling_function")

    def sampling_prepare(unet, x):
        shape = list(x.shape)
        return [(2 if shared.batch_cond_uncond else 1) * shape[0]] + shape[1:]  # noqa: F821

    module.sampling_prepare = sampling_prepare
    return module


def _module_not_reading_it():
    module = types.ModuleType("backend.sampling.sampling_function")

    def sampling_prepare(unet, x):
        return list(x.shape)

    module.sampling_prepare = sampling_prepare
    return module


@pytest.fixture
def forge(host, monkeypatch):
    """A host with the flag, a sampling_prepare that reads it, and a known build."""
    import model_chain  # noqa: F401  (registers the settings)

    monkeypatch.setattr(host.shared, "batch_cond_uncond", True, raising=False)
    monkeypatch.setitem(sys.modules, "backend.sampling.sampling_function", _module_reading_the_flag())
    monkeypatch.setattr(mc_batch_once, "_build", BUILD)
    monkeypatch.setattr(mc_batch_once, "_prior", None)
    monkeypatch.setattr(mc_batch_once, "_said", set())
    host.shared.opts.set(mc_batch_once.OPT_ENABLED, True)
    host.shared.opts.set(mc_batch_once.OPT_RECORD, "")
    return host


def pinned(host, build=BUILD):
    host.shared.opts.set(mc_batch_once.OPT_RECORD, json.dumps({"build": build, "when": "now"}))


def single(batch=1):
    """Conditioning with one part per image, as a plain prompt gives."""
    return types.SimpleNamespace(batch=[[object()] for _ in range(batch)])


def composed():
    """A prompt with two parts (``a AND b``)."""
    return types.SimpleNamespace(batch=[[object(), object()]])


def a_pass(cfg=1.0, **more):
    return types.SimpleNamespace(cfg_scale=cfg, is_hr_pass=False, hr_cfg=1.0, **more)


def unet(**options):
    return types.SimpleNamespace(model_options=dict(options))


class TestRegistration:
    def test_the_setting_is_on_by_default_and_drawn(self, forge):
        info = forge.shared.options_templates[mc_batch_once.OPT_ENABLED]
        assert info.default is True
        assert info.section[0] is not None

    def test_the_record_is_hidden(self, forge):
        info = forge.shared.options_templates[mc_batch_once.OPT_RECORD]
        assert info.section[0] is None
        assert info.default == ""


class TestTheHostBuild:
    def test_version_and_commit(self, monkeypatch):
        version = types.ModuleType("modules_forge.forge_version")
        version.version, version.release = "neo", "2.30"
        monkeypatch.setitem(sys.modules, "modules_forge.forge_version", version)
        monkeypatch.setattr(mc_batch_once, "_git_head", lambda: "1a2b3c4d5e")
        assert mc_batch_once._read_host_build() == "neo 2.30 @ 1a2b3c4d5e"

    def test_without_git_the_version_alone(self, monkeypatch):
        version = types.ModuleType("modules_forge.forge_version")
        version.version, version.release = "neo", "2.30"
        monkeypatch.setitem(sys.modules, "modules_forge.forge_version", version)
        monkeypatch.setattr(mc_batch_once, "_git_head", lambda: "")
        assert mc_batch_once._read_host_build() == "neo 2.30"

    def test_a_host_with_neither_is_unknown(self, monkeypatch):
        monkeypatch.delitem(sys.modules, "modules_forge.forge_version", raising=False)
        monkeypatch.setattr(mc_batch_once, "_git_head", lambda: "")
        assert mc_batch_once._read_host_build() == mc_batch_once.UNKNOWN_BUILD

    def test_a_checkout_is_read_with_git(self, tmp_path, monkeypatch):
        import subprocess

        subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
        subprocess.run(["git", "-C", str(tmp_path), "-c", "user.email=t@t", "-c", "user.name=t",
                        "commit", "-q", "--allow-empty", "-m", "x"], check=True)
        monkeypatch.setattr(mc_batch_once, "_script_path", lambda: str(tmp_path))
        head = mc_batch_once._git_head()
        assert len(head) == 10 and all(c in "0123456789abcdef" for c in head)

    def test_a_folder_without_a_checkout_has_no_commit(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mc_batch_once, "_script_path", lambda: str(tmp_path))
        assert mc_batch_once._git_head() == ""

    def test_it_is_read_once(self, monkeypatch):
        reads = []
        monkeypatch.setattr(mc_batch_once, "_build", None)
        monkeypatch.setattr(mc_batch_once, "_read_host_build", lambda: reads.append(1) or "b")
        assert mc_batch_once.host_build() == "b"
        assert mc_batch_once.host_build() == "b"
        assert reads == [1]


class TestTheHostsShape:
    def test_a_host_that_reads_the_flag_in_sampling_prepare_passes(self, forge):
        assert mc_batch_once.host_shape() == ""

    def test_a_host_without_the_flag_is_refused(self, forge, monkeypatch):
        monkeypatch.delattr(forge.shared, "batch_cond_uncond")
        assert "no batch_cond_uncond flag" in mc_batch_once.host_shape()

    def test_a_sampling_prepare_that_no_longer_reads_it_is_refused(self, forge, monkeypatch):
        monkeypatch.setitem(sys.modules, "backend.sampling.sampling_function", _module_not_reading_it())
        assert "no longer reads shared.batch_cond_uncond" in mc_batch_once.host_shape()

    def test_a_sampling_prepare_that_cannot_be_read_is_refused(self, forge, monkeypatch):
        monkeypatch.delitem(sys.modules, "backend.sampling.sampling_function")
        assert "cannot be read" in mc_batch_once.host_shape()


class TestStatus:
    def test_off_when_the_setting_is_off(self, forge):
        forge.shared.opts.set(mc_batch_once.OPT_ENABLED, False)
        pinned(forge)
        assert mc_batch_once.status().state == "off"

    def test_unpinned_before_the_first_start(self, forge):
        found = mc_batch_once.status()
        assert (found.state, found.active) == ("unpinned", False)

    def test_active_on_the_build_it_was_turned_on_with(self, forge):
        pinned(forge)
        found = mc_batch_once.status()
        assert (found.state, found.active, found.pinned) == ("active", True, BUILD)

    def test_stood_down_on_another_build(self, forge):
        pinned(forge, "neo 2.29 @ 0000000000")
        found = mc_batch_once.status()
        assert (found.state, found.active) == ("stood_down", False)
        assert found.reason == (
            f"Forge is {BUILD} now, not the neo 2.29 @ 0000000000 the setting was turned on with")

    def test_unsupported_outranks_the_record(self, forge, monkeypatch):
        pinned(forge)
        monkeypatch.setitem(sys.modules, "backend.sampling.sampling_function", _module_not_reading_it())
        assert mc_batch_once.status().state == "unsupported"

    def test_an_unreadable_record_is_no_record(self, forge):
        forge.shared.opts.set(mc_batch_once.OPT_RECORD, "{not json")
        assert mc_batch_once.record() == {}
        assert mc_batch_once.status().state == "unpinned"


class TestAppStart:
    def test_the_first_start_pins_the_installed_build(self, forge, caplog):
        with caplog.at_level("INFO"):
            mc_batch_once.on_app_started()

        assert mc_batch_once.record()["build"] == BUILD
        assert mc_batch_once.status().active is True
        [line] = [m for m in caplog.messages if "counts each image once" in m]
        assert f"pinned to Forge {BUILD}" in line

    def test_a_later_start_keeps_the_pin(self, forge):
        pinned(forge, "neo 2.29 @ 0000000000")
        mc_batch_once.on_app_started()
        assert mc_batch_once.record()["build"] == "neo 2.29 @ 0000000000"

    def test_a_forge_update_stands_it_down_and_says_so(self, forge, caplog):
        pinned(forge, "neo 2.29 @ 0000000000")
        with caplog.at_level("WARNING"):
            mc_batch_once.on_app_started()

        assert mc_batch_once.status().active is False
        [line] = [r for r in caplog.records if "turned off and on again" in r.getMessage()]
        assert line.levelno == logging.WARNING
        assert f"Forge is {BUILD} now" in line.getMessage()

    def test_a_host_without_the_shape_is_said_once(self, forge, monkeypatch, caplog):
        monkeypatch.setitem(sys.modules, "backend.sampling.sampling_function", _module_not_reading_it())
        with caplog.at_level("WARNING"):
            mc_batch_once.on_app_started()
            mc_batch_once.on_app_started()

        assert mc_batch_once.record() == {}, "nothing is pinned on a host it cannot apply to"
        assert len([m for m in caplog.messages if "counting each image once is off" in m]) == 1

    def test_the_setting_off_pins_nothing(self, forge):
        forge.shared.opts.set(mc_batch_once.OPT_ENABLED, False)
        mc_batch_once.on_app_started()
        assert mc_batch_once.record() == {}

    def test_the_switch_is_watched(self, forge, monkeypatch):
        watched = {}
        monkeypatch.setattr(forge.shared.opts, "onchange",
                            lambda key, func, call=True: watched.update({key: (func, call)}),
                            raising=False)
        mc_batch_once.on_app_started()
        assert watched[mc_batch_once.OPT_ENABLED] == (mc_batch_once.switched, False)


class TestTheSwitch:
    def test_turning_it_on_pins_the_build_now_installed(self, forge, monkeypatch):
        pinned(forge, "neo 2.29 @ 0000000000")
        monkeypatch.setattr(mc_batch_once, "_build", "neo 2.31 @ ffffffffff")
        mc_batch_once.switched()
        assert mc_batch_once.record()["build"] == "neo 2.31 @ ffffffffff"
        assert mc_batch_once.status().active is True

    def test_turning_it_off_clears_the_record(self, forge, caplog):
        pinned(forge)
        forge.shared.opts.set(mc_batch_once.OPT_ENABLED, False)
        with caplog.at_level("INFO"):
            mc_batch_once.switched()
        assert mc_batch_once.record() == {}
        assert any("the record of the Forge build is cleared" in m for m in caplog.messages)


class TestApplying:
    def test_a_cfg_1_pass_counts_once_until_the_first_step(self, forge, caplog):
        pinned(forge)
        with caplog.at_level("INFO"):
            assert mc_batch_once.apply(a_pass(1.0), unet(), single()) is True

        assert forge.shared.batch_cond_uncond is False
        assert mc_batch_once.in_force() is True
        mc_batch_once.first_step()
        assert forge.shared.batch_cond_uncond is True
        assert mc_batch_once.in_force() is False
        [line] = [m for m in caplog.messages if "counting each image once" in m]
        assert "for this CFG 1 pass" in line and BUILD in line

    def test_the_line_is_said_once_a_session(self, forge, caplog):
        pinned(forge)
        with caplog.at_level("INFO"):
            for _ in range(3):
                mc_batch_once.apply(a_pass(1.0), unet(), single())
                mc_batch_once.first_step()
        assert len([m for m in caplog.messages if "counting each image once" in m]) == 1

    def test_restore_puts_back_what_it_found(self, forge):
        pinned(forge)
        forge.shared.batch_cond_uncond = False
        mc_batch_once.apply(a_pass(1.0), unet(), single())
        assert mc_batch_once.restore() is True
        assert forge.shared.batch_cond_uncond is False

    def test_restore_with_nothing_flipped_is_a_no_op(self, forge):
        assert mc_batch_once.restore() is False
        assert forge.shared.batch_cond_uncond is True

    def test_a_second_pass_without_a_step_between_restores_first(self, forge):
        pinned(forge)
        mc_batch_once.apply(a_pass(1.0), unet(), single())
        mc_batch_once.apply(a_pass(1.0), unet(), single())
        mc_batch_once.restore()
        assert forge.shared.batch_cond_uncond is True, "the prior is never the flipped value"

    def test_cfg_above_1_counts_twice(self, forge):
        pinned(forge)
        assert mc_batch_once.apply(a_pass(7.0), unet(), single()) is False
        assert forge.shared.batch_cond_uncond is True
        assert "CFG is 7" in mc_batch_once.applies(a_pass(7.0), unet(), single())

    def test_the_hires_pass_has_a_cfg_of_its_own(self, forge):
        pinned(forge)
        hires = types.SimpleNamespace(cfg_scale=1.0, is_hr_pass=True, hr_cfg=4.0)
        assert mc_batch_once.apply(hires, unet(), single()) is False
        hires.hr_cfg = 1.0
        assert mc_batch_once.apply(hires, unet(), single()) is True
        mc_batch_once.restore()

    def test_the_hosts_cfg_1_optimisation_off_counts_twice(self, forge):
        pinned(forge)
        assert mc_batch_once.apply(a_pass(1.0), unet(disable_cfg1_optimization=True), single()) is False
        assert forge.shared.batch_cond_uncond is True

    def test_a_composed_prompt_counts_twice(self, forge):
        pinned(forge)
        assert mc_batch_once.apply(a_pass(1.0), unet(), composed()) is False
        assert "more than one part" in mc_batch_once.applies(a_pass(1.0), unet(), composed())

    def test_conditioning_without_parts_is_taken_as_simple(self, forge):
        pinned(forge)
        assert mc_batch_once.apply(a_pass(1.0), None, None) is True
        mc_batch_once.restore()

    def test_a_batch_of_several_single_parts_counts_once(self, forge):
        pinned(forge)
        assert mc_batch_once.apply(a_pass(1.0), unet(), single(batch=4)) is True
        mc_batch_once.restore()

    def test_nothing_happens_while_stood_down(self, forge):
        pinned(forge, "neo 2.29 @ 0000000000")
        assert mc_batch_once.apply(a_pass(1.0), unet(), single()) is False
        assert forge.shared.batch_cond_uncond is True

    def test_nothing_happens_before_the_first_start_pins(self, forge):
        assert mc_batch_once.apply(a_pass(1.0), unet(), single()) is False

    def test_nothing_happens_with_the_setting_off(self, forge):
        pinned(forge)
        forge.shared.opts.set(mc_batch_once.OPT_ENABLED, False)
        assert mc_batch_once.apply(a_pass(1.0), unet(), single()) is False


class TestTheDemandReading:
    """The extension's reading of the request follows the host's flag."""

    @pytest.fixture
    def card(self, forge, monkeypatch):
        from test_warming import FakeUnet, loaded, GB

        from backend import memory_management

        memory_management.current_loaded_models.clear()
        sampled = FakeUnet(12.6 * GB, activations=5.6 * GB)
        memory_management.current_loaded_models.extend([
            loaded("JointTextEncoder", 5.6 * GB), loaded("WanVAE", 0.2 * GB),
        ])
        monkeypatch.setattr(memory_management, "get_free_memory", lambda dev=None: 4.3 * GB)
        monkeypatch.setattr(memory_management, "minimum_inference_memory", lambda: 0.8 * GB, raising=False)
        monkeypatch.setattr(memory_management, "extra_reserved_memory", lambda: 0.7 * GB, raising=False)
        monkeypatch.setattr(mc_memory, "_last_sampler_demand", None)
        return sampled

    def test_counts_twice_as_the_host_does(self, card):
        demand = mc_memory.note_sampler_demand(card, (2, 16, 1, 128, 128))
        assert card.asked_shape == [4, 16, 1, 128, 128]
        assert demand.counted_once is False
        assert "counted once" not in mc_memory.describe_sampler_demand(demand)

    def test_counts_once_while_the_override_is_in_force(self, forge, card, caplog):
        pinned(forge)
        mc_batch_once.apply(a_pass(1.0), card, single(batch=2))
        with caplog.at_level("INFO"):
            demand = mc_memory.note_sampler_demand(card, (2, 16, 1, 128, 128))
        mc_batch_once.restore()

        assert card.asked_shape == [2, 16, 1, 128, 128]
        assert demand.counted_once is True
        [line] = [m for m in caplog.messages if "about to ask for" in m]
        assert "(a 2x16x1x128x128 latent, each image counted once)" in line

    def test_a_host_without_the_flag_is_read_as_counting_twice(self, card, monkeypatch, forge):
        monkeypatch.delattr(forge.shared, "batch_cond_uncond")
        mc_memory.note_sampler_demand(card, (1, 16, 1, 128, 128))
        assert card.asked_shape == [2, 16, 1, 128, 128]


class TestTheHooks:
    """The script applies the override last before the host's request and restores it."""

    def test_the_pre_sampling_hook_applies_it_with_the_passes_conditioning(self, chain, host,
                                                                            monkeypatch):
        seen = []
        monkeypatch.setattr(mc_batch_once, "apply", lambda p, unet, c: seen.append((p, c)) or True)
        readings = []
        monkeypatch.setattr(mc_memory, "note_sampler_demand",
                            lambda unet, shape: readings.append(len(seen)) or None)
        p = make_p(host)
        args = [DEFAULTS[name] for name in UI_ORDER]
        conditioning = single()

        chain.script.process_before_every_sampling(
            p, *args, noise=types.SimpleNamespace(shape=(1, 16, 1, 128, 128)), c=conditioning)

        assert seen == [(p, conditioning)]
        assert readings == [1], "the reading is taken after the override, so it says what the host will ask"

    def test_the_denoiser_callback_restores_at_the_first_step(self, chain, host, forge):
        pinned(forge)
        mc_batch_once.apply(a_pass(1.0), unet(), single())
        chain.module._on_cfg_denoiser(types.SimpleNamespace(sampling_step=0))
        assert forge.shared.batch_cond_uncond is True

    def test_post_sample_restores_a_pass_that_never_stepped(self, chain, host, forge):
        pinned(forge)
        mc_batch_once.apply(a_pass(1.0), unet(), single())
        chain.script.post_sample(make_p(host), types.SimpleNamespace(samples=None))
        assert forge.shared.batch_cond_uncond is True

    def test_before_process_restores_what_an_aborted_pass_left(self, chain, host, forge):
        pinned(forge)
        mc_batch_once.apply(a_pass(1.0), unet(), single())
        p = make_p(host)
        args = [DEFAULTS[name] for name in UI_ORDER]
        chain.script.before_process(p, *args)
        assert forge.shared.batch_cond_uncond is True

    def test_the_callback_and_the_start_hook_are_registered(self, chain, host):
        registered = host.script_callbacks.registered
        assert mc_batch_once.on_app_started in registered["app_started"]
        assert chain.module._on_cfg_denoiser in registered["cfg_denoiser"]
