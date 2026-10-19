"""The text encoder and VAE after a generation: left where the host put them, or moved back.

The host's sampler pushes Stage 1's text encoder and VAE off the card to make
room for a batch's activations (mc_memory, "What the host's sampler is about to
ask for"), and the warm-up after the generation used to move them straight
back -- a move the next pass at the same batch size undid before its first
step, so a generation paid it twice. Off by default now: the warm-up moves the
diffusion model and leaves the encoders wherever the host put them, and the
host loads them when a new prompt or a decode needs them. On is the old
behaviour.
"""

from __future__ import annotations

import types

import pytest

import mc_memory
from test_residency_speed import memory, preload  # noqa: F401  (fixtures)

GB = 1024**3


def restoring(host, value: bool) -> None:
    host.shared.opts.model_chain_restore_encoders = value


@pytest.fixture
def card(memory, host):
    """Model A loaded with its text encoder in system RAM, the way a batch leaves it."""
    model = host.sd_models.model_data.sd_model
    clip = model.patchers.clip
    memory.mm.current_loaded_models[:] = [
        entry for entry in memory.mm.current_loaded_models if entry.model is not clip
    ]
    return types.SimpleNamespace(
        model=model, mm=memory.mm, unet=model.patchers.unet, clip=clip, vae=model.patchers.vae,
    )


class TestTheSetting:
    def test_it_is_registered_off(self, host):
        import model_chain  # noqa: F401  (registration happens on import)

        assert host.shared.options_templates[mc_memory.OPT_RESTORE_ENCODERS].default is False
        assert mc_memory.RESTORE_ENCODERS_DEFAULT is False

    def test_off_means_the_diffusion_model_alone(self, host):
        restoring(host, False)
        assert mc_memory.warm_attrs() == ("unet",)

    def test_on_means_everything(self, host):
        restoring(host, True)
        assert mc_memory.warm_attrs() == ("unet", "clip", "vae")

    def test_a_missing_setting_leaves_them(self, host, monkeypatch):
        """A host that has not registered the setting gets the documented default."""
        import model_chain  # noqa: F401

        monkeypatch.delitem(host.shared.options_templates, mc_memory.OPT_RESTORE_ENCODERS)
        assert mc_memory.warm_attrs() == ("unet",)


class TestTheWarmUp:
    def test_leaves_the_encoders_where_the_host_put_them(self, card, host):
        restoring(host, False)
        mc_memory._load_current_to_gpu(1024, 1024)
        assert card.mm.loaded_to_gpu[-1] == [card.unet]

    def test_on_moves_them_back(self, card, host):
        restoring(host, True)
        mc_memory._load_current_to_gpu(1024, 1024)
        assert card.mm.loaded_to_gpu[-1] == [card.unet, card.clip, card.vae]

    def test_the_reserve_is_still_the_whole_passes(self, card, host, monkeypatch):
        """The headroom handed to the host is trimmed against the whole checkpoint,
        because the pass runs with the encoder on the card whenever a prompt had
        to be read first."""
        seen = {}
        monkeypatch.setattr(card.mm, "load_models_gpu", lambda models, **kw: seen.update(kw))
        monkeypatch.setattr(mc_memory, "total_vram_bytes", lambda: 14 * GB)
        restoring(host, False)

        mc_memory._load_current_to_gpu(1024, 1024)

        assert seen["memory_required"] == mc_memory._attainable_headroom(13 * GB, 1024, 1024)
        assert seen["memory_required"] == 1 * GB, "trimmed to what the card has beside all 13 GB"

    def test_residency_counts_what_the_warm_up_moves(self, card, host):
        restoring(host, False)
        assert mc_memory._loaded_residency() == (8 * GB, 8 * GB)
        restoring(host, True)
        assert mc_memory._loaded_residency() == (9 * GB, 13 * GB)

    def test_nothing_to_do_when_only_the_encoders_are_off(self, card, host, monkeypatch):
        monkeypatch.setattr(mc_memory, "_pending_restore", None, raising=False)
        restoring(host, False)
        assert mc_memory._preload_task() is None
        restoring(host, True)
        assert mc_memory._preload_task() == mc_memory.RESIDENT

    def test_the_worker_budgets_room_for_what_it_moves(self, card, host, preload):
        restoring(host, False)
        mc_memory._preload_worker(1024, 1024, task=mc_memory.RESIDENT)

        (_name, _w, _h, stage, kwargs) = preload.rooms[-1]
        assert stage == mc_memory.STAGE_1
        assert kwargs["moving"] == [card.unet]
        assert preload.gpu_loads == 1

    def test_the_worker_budgets_the_whole_checkpoint_when_restoring(self, card, host, preload):
        restoring(host, True)
        mc_memory._preload_worker(1024, 1024, task=mc_memory.RESIDENT)

        assert preload.rooms[-1][4]["moving"] is None

    def test_the_ready_line_says_what_stays_in_ram(self, card, host, preload, caplog):
        restoring(host, False)
        with caplog.at_level("INFO"):
            mc_memory._preload_worker(1024, 1024, task=mc_memory.RESIDENT)

        result = mc_memory.preload_result()
        assert result.state == "ready"
        assert result.kept_off == (
            "its text encoder and VAE stay in system RAM (4.0 GB) until a prompt or a "
            "decode needs them")
        [line] = [m for m in caplog.messages if "is in VRAM" in m]
        assert line.endswith("sampling immediately; " + result.kept_off)

    def test_the_ready_line_says_nothing_when_they_are_moved_back(self, memory, host, preload,
                                                                   caplog):
        restoring(host, True)
        with caplog.at_level("INFO"):
            mc_memory._preload_worker(1024, 1024, task=mc_memory.RESIDENT)

        assert mc_memory.preload_result().kept_off == ""
        [line] = [m for m in caplog.messages if "is in VRAM" in m]
        assert line.endswith("sampling immediately")

    def test_encoders_on_the_card_are_not_reported_as_left(self, memory, host):
        restoring(host, False)
        assert mc_memory.encoders_off_card() == (0, 5 * GB)
        assert mc_memory.describe_encoders_left() == ""


class TestMakingRoom:
    """``moving`` budgets a warm-up for the part it moves and the pass's headroom."""

    @pytest.fixture
    def tight(self, card, monkeypatch):
        monkeypatch.setattr(card.mm, "get_free_memory", lambda dev=None: 0.5 * GB)
        monkeypatch.setattr(mc_memory, "total_vram_bytes", lambda: 24 * GB)
        monkeypatch.setattr(mc_memory, "_llm_residency_bytes", lambda: 0, raising=False)
        return card

    def test_a_partial_warm_up_asks_only_for_its_headroom(self, tight):
        headroom = mc_memory._attainable_headroom(13 * GB, 1024, 1024)
        mc_memory.make_vram_room("A", [], 1024, 1024, stage=mc_memory.STAGE_1,
                                 reclaim_foreign=False, moving=[tight.unet])

        # The diffusion model is resident, so only the headroom is still needed.
        assert tight.mm.freed[-1] == pytest.approx(headroom)

    def test_the_default_asks_for_the_whole_checkpoint(self, tight):
        headroom = mc_memory._attainable_headroom(13 * GB, 1024, 1024)
        mc_memory.make_vram_room("A", [], 1024, 1024, stage=mc_memory.STAGE_1,
                                 reclaim_foreign=False)

        # 13 GB plus headroom, less the 9 GB resident: the encoder's room too.
        assert tight.mm.freed[-1] == pytest.approx(13 * GB + headroom - 9 * GB)

    def test_a_partial_warm_up_that_fits_says_what_it_moves(self, card, monkeypatch, caplog):
        monkeypatch.setattr(card.mm, "get_free_memory", lambda dev=None: 8 * GB)
        monkeypatch.setattr(mc_memory, "total_vram_bytes", lambda: 24 * GB)
        with caplog.at_level("INFO"):
            freed = mc_memory.make_vram_room("A", [], 1024, 1024, stage=mc_memory.STAGE_1,
                                             reclaim_foreign=False, moving=[card.unet])

        assert freed == 0
        [line] = [m for m in caplog.messages if "no eviction needed" in m]
        assert "Stage 1 needs 8.0 GB for its diffusion model plus" in line
        assert "8.0 GB of it already resident" in line

    def test_the_default_line_is_what_it_was(self, card, monkeypatch, caplog):
        monkeypatch.setattr(card.mm, "get_free_memory", lambda dev=None: 8 * GB)
        monkeypatch.setattr(mc_memory, "total_vram_bytes", lambda: 24 * GB)
        with caplog.at_level("INFO"):
            mc_memory.make_vram_room("A", [], 1024, 1024, stage=mc_memory.STAGE_1,
                                     reclaim_foreign=False)

        [line] = [m for m in caplog.messages if "no eviction needed" in m]
        assert "Stage 1 needs 14.0 GB, has 8.0 GB free and 9.0 GB already resident" in line


class TestReadiness:
    def test_says_where_the_encoders_are(self, card, host):
        restoring(host, False)
        state, line = mc_memory.stage_1_readiness()

        assert state == mc_memory.WARM
        assert "8.0 GB already in VRAM" in line
        assert line.endswith(
            "; its text encoder and VAE stay in system RAM (4.0 GB) until a prompt or a "
            "decode needs them.")

    def test_counts_them_when_they_are_moved_back(self, card, host):
        restoring(host, True)
        state, line = mc_memory.stage_1_readiness()

        assert state == mc_memory.PARTIAL
        assert "9.0 GB of 13.0 GB in VRAM" in line

    def test_says_nothing_about_encoders_that_are_on_the_card(self, memory, host):
        restoring(host, False)
        state, line = mc_memory.stage_1_readiness()

        assert state == mc_memory.WARM
        assert "stay in system RAM" not in line
