"""Residency behaviour that exists purely to make the switch faster.

Three mechanisms, all measured against real console logs from a Krea 2 ->
Flux.2 chain on a 22 GB card:

* the VRAM estimate is widened so the host stops partially unloading mid-load,
* Stage 1's encoders stay resident through the Stage 2 switch,
* Stage 1 is warmed back into VRAM in the background between generations.

None of them may change what comes out of the pipeline, so the tests here are
as interested in the fallbacks as in the happy paths.
"""

from __future__ import annotations

import types

import pytest

import mc_arm
import mc_memory

GB = 1024**3


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class FakePatcher:
    """Stands in for a ``ModelPatcher``."""

    def __init__(self, label, size):
        self.label = label
        self.size = size

    def model_size(self):
        return self.size

    def __repr__(self):  # keeps assertion output readable
        return f"<{self.label}>"


class FakeClip:
    """``forge_objects.clip`` wraps its patcher rather than being one."""

    def __init__(self, patcher):
        self.patcher = patcher


class FakeLoadedModel:
    """Stands in for an entry of ``memory_management.current_loaded_models``."""

    def __init__(self, patcher):
        self.model = patcher

    def model_loaded_memory(self):
        return self.model.size


def make_model(name="A", unet=8 * GB, clip=4 * GB, vae=1 * GB):
    unet_patcher = FakePatcher(f"{name}-unet", unet)
    clip_patcher = FakePatcher(f"{name}-clip", clip)
    vae_patcher = FakePatcher(f"{name}-vae", vae)
    model = types.SimpleNamespace(
        name=name,
        sd_checkpoint_info=types.SimpleNamespace(
            filename=f"/models/{name}", name_for_extra=name, sha256=f"sha-{name}"
        ),
        forge_objects=types.SimpleNamespace(
            unet=unet_patcher, clip=FakeClip(clip_patcher), vae=vae_patcher
        ),
    )
    model.patchers = types.SimpleNamespace(
        unet=unet_patcher, clip=clip_patcher, vae=vae_patcher
    )
    return model


@pytest.fixture
def memory(host, monkeypatch):
    """The faked ``backend.memory_management``, reset and with a loaded model."""
    from backend import memory_management

    memory_management.freed.clear()
    memory_management.kept.clear()
    memory_management.loaded_to_gpu.clear()
    memory_management.current_loaded_models.clear()

    model = make_model("A")
    host.sd_models.model_data.sd_model = model
    # model_data is a module-level singleton in the fake host, so a test that
    # points the selection somewhere would otherwise leak into the next one.
    host.sd_models.model_data.forge_hash = ""
    host.sd_models.model_data.forge_loading_parameters = {}
    memory_management.current_loaded_models.extend(
        FakeLoadedModel(p) for p in (model.patchers.unet, model.patchers.clip, model.patchers.vae)
    )

    monkeypatch.setattr(mc_memory, "file_size_bytes", lambda name, mods=None: 10 * GB)
    monkeypatch.setattr(mc_memory, "resolve_modules", lambda mods=None: None)
    monkeypatch.setattr(mc_memory, "current_modules", lambda: [])
    monkeypatch.setattr(
        host.sd_models, "get_closet_checkpoint_match",
        lambda name: types.SimpleNamespace(
            filename=f"/models/{name}", name_for_extra=name, sha256=f"sha-{name}"
        ),
    )
    mc_memory.clear_pinned_encoders()

    yield types.SimpleNamespace(mm=memory_management, model=model)

    mc_memory.clear_pinned_encoders()


# --------------------------------------------------------------------------- #
# The VRAM estimate
# --------------------------------------------------------------------------- #


class TestVramEstimate:
    """A model is bigger in VRAM than it is on disk.

    Under-counting is what produced the ``Unloaded partially`` lines in the log
    immediately before each UNet load: free_memory() hit our target and stopped,
    the target was short, and the host then unloaded in chunks *while* loading.

    Every test here isolates the *model* term by subtracting the activation
    headroom. Left in, the headroom is large enough to keep these assertions
    true even with the overhead removed entirely -- which would make the whole
    class look like coverage while testing nothing.
    """

    @staticmethod
    def model_term(file_size=10 * GB, width=0, height=0):
        return (
            mc_memory.vram_required_bytes("A", None, width, height)
            - mc_memory.vram_headroom_bytes(width, height)
            - file_size
        )

    def test_a_model_is_counted_as_larger_than_its_file(self, memory):
        assert self.model_term() > 0

    def test_the_overhead_is_proportional_to_the_model(self, memory, monkeypatch):
        def overhead(file_size):
            monkeypatch.setattr(mc_memory, "file_size_bytes", lambda name, mods=None: file_size)
            return self.model_term(file_size)

        assert overhead(20 * GB) == pytest.approx(overhead(10 * GB) * 2, rel=1e-6)

    def test_the_overhead_covers_the_shortfalls_measured_in_practice(self):
        """Two logged chains under-ran by 11% and 14% of model size."""
        assert mc_memory.VRAM_MODEL_OVERHEAD_FRACTION >= 0.15

    def test_activation_headroom_still_scales_with_pass_size(self, memory):
        small = mc_memory.vram_required_bytes("A", None, 1024, 1024)
        large = mc_memory.vram_required_bytes("A", None, 2048, 2048)
        assert large > small

    def test_make_vram_room_frees_past_the_file_size(self, memory, monkeypatch):
        """"B" rather than "A": the estimate only applies to a model not yet loaded."""
        monkeypatch.setattr(mc_memory, "free_vram_bytes", lambda: 1 * GB)
        mc_memory.make_vram_room("B", None, 1024, 1024)

        assert memory.mm.freed[0] > 10 * GB + mc_memory.vram_headroom_bytes(1024, 1024)

    def test_the_preflight_plan_predicts_against_the_same_figure(self, memory, monkeypatch):
        """Otherwise the accordion promises dual residency the switch then breaks."""
        from modules import sd_models

        monkeypatch.setattr(mc_memory, "loaded_size_bytes", lambda model: 7 * GB)
        monkeypatch.setattr(
            sd_models, "get_closet_checkpoint_match",
            lambda name: types.SimpleNamespace(
                filename=f"/models/{name}", name_for_extra=name, sha256=f"sha-{name}"
            ),
        )
        # Exactly enough for the file plus its activations, and nothing more --
        # so only the overhead can decide this.
        monkeypatch.setattr(
            mc_memory, "free_vram_bytes", lambda: 10 * GB + mc_memory.vram_headroom_bytes()
        )

        assert mc_memory.plan("B").kind != "dual"


# --------------------------------------------------------------------------- #
# Never evicting the model the room is being made for
# --------------------------------------------------------------------------- #


class TestTargetIsSpared:
    """Regression: the preload's work was being thrown away and redone.

    Observed in a real log. The preload moved Krea 2's VAE, UNet and text
    encoder into VRAM (8.5s), and the very next generation opened with
    ``freed 13.8 GB VRAM for Stage 1 ... offloaded UnetPatcher, ModelPatcher,
    ModelPatcher`` and then re-loaded the same three. Asking to free the whole
    requirement while the target is already resident evicts the target, because
    the target's own weights are the largest evictable thing on the card.
    """

    def test_a_fully_resident_target_needs_no_eviction(self, memory, monkeypatch):
        """This is the preload case: everything is already where it belongs."""
        # 13 GB of model resident, leaving only the activation headroom to find.
        monkeypatch.setattr(mc_memory, "free_vram_bytes", lambda: 3 * GB)

        mc_memory.make_vram_room("A", None, 1024, 1024, stage=mc_memory.STAGE_1)

        assert memory.mm.freed == [], "the preloaded model must not be evicted"

    def test_the_target_is_never_offered_for_eviction(self, memory, monkeypatch):
        """Even when the pass genuinely needs more room than is free."""
        monkeypatch.setattr(mc_memory, "free_vram_bytes", lambda: 0.1 * GB)

        mc_memory.make_vram_room("A", None, 4096, 4096, stage=mc_memory.STAGE_1)

        spared = {entry.model.label for entry in memory.mm.kept[0]}
        assert spared == {"A-unet", "A-clip", "A-vae"}

    def test_only_the_shortfall_is_requested(self, memory, monkeypatch):
        """Not the whole requirement: 13 GB of it is already on the card."""
        monkeypatch.setattr(mc_memory, "free_vram_bytes", lambda: 0.5 * GB)

        mc_memory.make_vram_room("A", None, 1024, 1024, stage=mc_memory.STAGE_1)

        resident = 13 * GB
        requirement = resident + mc_memory.vram_headroom_bytes(1024, 1024)
        assert memory.mm.freed == [requirement - resident]

    def test_a_growing_pass_still_frees_the_difference(self, memory, monkeypatch):
        """Raising the resolution between generations must still make room."""
        monkeypatch.setattr(mc_memory, "free_vram_bytes", lambda: 0.5 * GB)

        mc_memory.make_vram_room("A", None, 2048, 2048, stage=mc_memory.STAGE_1)

        assert memory.mm.freed and memory.mm.freed[0] > 0

    def test_a_target_that_is_not_loaded_falls_back_to_the_disk_estimate(self, memory, monkeypatch):
        """A cold Stage 2: nothing of it is resident, so nothing is subtracted."""
        monkeypatch.setattr(mc_memory, "free_vram_bytes", lambda: 1 * GB)

        mc_memory.make_vram_room("B", None, 1024, 1024, stage=mc_memory.STAGE_2)

        expected = 10 * GB * (1 + mc_memory.VRAM_MODEL_OVERHEAD_FRACTION)
        assert memory.mm.freed[0] == pytest.approx(
            expected + mc_memory.vram_headroom_bytes(1024, 1024)
        )

    def test_the_real_model_size_beats_the_disk_estimate(self, memory, monkeypatch):
        """The patchers know their true size; the file is only a proxy."""
        monkeypatch.setattr(mc_memory, "free_vram_bytes", lambda: 0.5 * GB)
        monkeypatch.setattr(mc_memory, "file_size_bytes", lambda name, mods=None: 99 * GB)

        mc_memory.make_vram_room("A", None, 1024, 1024, stage=mc_memory.STAGE_1)

        # Sized from the 13 GB of resident patchers, not the bogus 99 GB file.
        assert memory.mm.freed == [mc_memory.vram_headroom_bytes(1024, 1024)]


# --------------------------------------------------------------------------- #
# Pinning Stage 1's encoders
# --------------------------------------------------------------------------- #


class TestPinnedEncoders:
    def test_capture_takes_the_encoders_but_not_the_unet(self, memory):
        assert mc_memory.capture_stage_1_encoders() == 2

        labels = {p.label for p in mc_memory._pinned_patchers}
        assert labels == {"A-clip", "A-vae"}

    def test_the_clip_wrapper_is_unwrapped_to_its_patcher(self, memory):
        mc_memory.capture_stage_1_encoders()
        assert memory.model.patchers.clip in mc_memory._pinned_patchers

    def test_stage_2_spares_the_pinned_encoders(self, memory, monkeypatch):
        monkeypatch.setattr(mc_memory, "free_vram_bytes", lambda: 1 * GB)
        mc_memory.capture_stage_1_encoders()

        mc_memory.make_vram_room("B", None, 1024, 1024, stage=mc_memory.STAGE_2)

        kept = {entry.model.label for entry in memory.mm.kept[0]}
        assert kept == {"A-clip", "A-vae"}

    def test_stage_1_spares_nothing(self, memory, monkeypatch):
        """The asymmetry is the point: Stage 2's encoders stay evictable."""
        monkeypatch.setattr(mc_memory, "free_vram_bytes", lambda: 1 * GB)
        mc_memory.capture_stage_1_encoders()

        mc_memory.make_vram_room("B", None, 1024, 1024, stage=mc_memory.STAGE_1)

        assert memory.mm.kept == [[]]

    def test_the_pin_is_dropped_when_stage_2_would_not_fit_alongside(self, memory, monkeypatch):
        """Pinning more than the card can spare is worse than not pinning.

        free_memory() would fall short of its target and load_models_gpu() would
        make up the difference the slow way -- the exact thrash being avoided.
        """
        monkeypatch.setattr(mc_memory, "free_vram_bytes", lambda: 1 * GB)
        monkeypatch.setattr(mc_memory, "total_vram_bytes", lambda: 12 * GB)
        mc_memory.capture_stage_1_encoders()

        mc_memory.make_vram_room("B", None, 1024, 1024, stage=mc_memory.STAGE_2)

        assert memory.mm.kept == [[]]

    def test_a_host_without_keep_loaded_still_gets_its_vram_freed(self, memory, monkeypatch):
        """Losing the pin is acceptable; losing the eviction is not."""
        calls = []

        def free_memory(required, device):
            calls.append(required)
            return []

        monkeypatch.setattr(memory.mm, "free_memory", free_memory)
        monkeypatch.setattr(mc_memory, "free_vram_bytes", lambda: 1 * GB)
        mc_memory.capture_stage_1_encoders()

        mc_memory.make_vram_room("B", None, 1024, 1024, stage=mc_memory.STAGE_2)

        assert calls, "the eviction must still happen without keep_loaded support"

    def test_a_cold_stage_2_load_leaves_nothing_to_pin(self, memory, monkeypatch):
        """forge_model_reload() unloads everything, so the pin cannot survive it.

        Only the warm path can hold encoders across the switch. This must
        degrade quietly rather than pass stale entries to the host.
        """
        monkeypatch.setattr(mc_memory, "free_vram_bytes", lambda: 1 * GB)
        mc_memory.capture_stage_1_encoders()
        memory.mm.current_loaded_models.clear()  # what unload_all_models() leaves

        mc_memory.make_vram_room("B", None, 1024, 1024, stage=mc_memory.STAGE_2)

        assert memory.mm.kept == [[]]

    def test_encoders_already_evicted_are_not_pinned(self, memory, monkeypatch):
        monkeypatch.setattr(mc_memory, "free_vram_bytes", lambda: 1 * GB)
        mc_memory.capture_stage_1_encoders()
        memory.mm.current_loaded_models.clear()

        mc_memory.make_vram_room("B", None, 1024, 1024, stage=mc_memory.STAGE_2)

        assert memory.mm.kept == [[]]

    def test_the_setting_disables_capture(self, memory, host):
        host.shared.opts.model_chain_pin_stage1_encoders = False
        assert mc_memory.capture_stage_1_encoders() == 0

    def test_a_missing_setting_keeps_the_documented_default(self, memory, host, monkeypatch):
        """A host that never registered the option still gets the default.

        Which is the case for a config saved by a version that predates it, and
        for any code path that reads the option before the settings page has
        been built.
        """
        monkeypatch.delitem(host.shared.options_templates, mc_memory.OPT_PIN_ENCODERS)
        assert not hasattr(host.shared.opts, mc_memory.OPT_PIN_ENCODERS)
        assert mc_memory.capture_stage_1_encoders() == 2

    def test_capture_survives_a_model_without_forge_objects(self, memory, host):
        host.sd_models.model_data.sd_model = types.SimpleNamespace(name="bare")
        assert mc_memory.capture_stage_1_encoders() == 0


# --------------------------------------------------------------------------- #
# The background Stage 1 preload
# --------------------------------------------------------------------------- #


@pytest.fixture
def preload(memory, monkeypatch, host):
    """Records what the preload worker did, with the swap itself faked.

    Opts the setting in: the preload ships off, so every test of what it *does*
    has to enable it first.
    """
    host.shared.opts.model_chain_preload_stage1 = True
    calls = types.SimpleNamespace(reinstated=0, rooms=[], gpu_loads=0, contexts=0, warms=0)

    import contextlib

    @contextlib.contextmanager
    def fake_context():
        calls.contexts += 1
        yield

    monkeypatch.setattr(mc_memory, "_host_torch_context", fake_context)

    # Session state the circuit breaker and the readiness report both live in.
    # Left over from one test it would silently disarm the next.
    for name, value in (
        ("_preload_failures", 0),
        ("_preload_disabled_reason", None),
        ("_preload_option_seen", None),
        ("_preload_result", None),
        ("_preload_reinstated", False),
    ):
        monkeypatch.setattr(mc_memory, name, value, raising=False)

    monkeypatch.setattr(
        mc_memory, "warm_secondary",
        lambda w=0, h=0: calls.__setattr__("warms", calls.warms + 1) or 0,
    )

    def reinstate_pending():
        calls.reinstated += 1
        return True

    monkeypatch.setattr(mc_memory, "reinstate_pending", reinstate_pending)
    monkeypatch.setattr(
        mc_memory, "make_vram_room",
        lambda name, mods=None, w=0, h=0, stage=mc_memory.STAGE_2, **kwargs:
            calls.rooms.append((name, w, h, stage, kwargs)),
    )
    monkeypatch.setattr(
        mc_memory, "_load_current_to_gpu",
        lambda w=0, h=0: (calls.__setattr__("gpu_loads", calls.gpu_loads + 1), 4 * GB)[1],
    )
    monkeypatch.setattr(mc_memory, "_pending_restore", "A", raising=False)

    yield calls

    mc_memory.join_preload(timeout=5)
    mc_memory.consume_preload()


class TestPreloadIsOptIn:
    """The preload is the only mechanism here that leaves the generation thread.

    It also saves no work -- it only moves the same work earlier -- so a machine
    where it misbehaves loses far more than a machine where it is off gains.
    """

    def test_it_does_not_run_unless_asked(self, memory, monkeypatch):
        monkeypatch.setattr(mc_memory, "_pending_restore", "A", raising=False)
        assert mc_memory.preload_async(1024, 1024) is False

    def test_the_registered_default_matches_the_code_fallback(self, host):
        import model_chain  # noqa: F401

        registered = host.shared.options_templates[mc_memory.OPT_PRELOAD].default
        assert registered is mc_memory.PRELOAD_DEFAULT
        assert mc_memory.option(mc_memory.OPT_PRELOAD, mc_memory.PRELOAD_DEFAULT) is False


class TestPreload:
    def test_the_worker_enters_the_hosts_torch_context(self, preload):
        """Weights loaded outside inference mode break the next sampling step.

        The host holds torch.inference_mode() for a whole generation, so every
        model it loads yields inference tensors. A thread starting with grad
        enabled produces the other kind, and mixing them raises
        "Inference tensors do not track version counter" on the first step.
        """
        mc_memory.preload_async(1024, 1024)
        mc_memory.join_preload(timeout=5)

        assert preload.contexts == 1

    def test_nothing_pending_means_no_preload(self, preload, monkeypatch):
        monkeypatch.setattr(mc_memory, "_pending_restore", None)
        assert mc_memory.preload_async(1024, 1024) is False
        assert preload.reinstated == 0

    def test_the_preload_swaps_stage_1_in_and_moves_it_to_the_gpu(self, preload):
        assert mc_memory.preload_async(1024, 1024) is True
        mc_memory.join_preload(timeout=5)

        assert preload.reinstated == 1
        assert preload.gpu_loads == 1

    def test_the_preload_makes_room_as_stage_1(self, preload):
        mc_memory.preload_async(1024, 1024)
        mc_memory.join_preload(timeout=5)

        assert [room[3] for room in preload.rooms] == [mc_memory.STAGE_1]

    def test_the_next_generation_learns_the_swap_already_happened(self, preload):
        mc_memory.preload_async(1024, 1024)
        mc_memory.join_preload(timeout=5)

        assert mc_memory.consume_preload() is True

    def test_consuming_is_one_shot(self, preload):
        mc_memory.preload_async(1024, 1024)
        mc_memory.join_preload(timeout=5)

        mc_memory.consume_preload()
        assert mc_memory.consume_preload() is False

    def test_a_preload_that_found_nothing_to_do_reports_nothing(self, preload, monkeypatch):
        monkeypatch.setattr(mc_memory, "reinstate_pending", lambda: False)
        mc_memory.preload_async(1024, 1024)
        mc_memory.join_preload(timeout=5)

        assert mc_memory.consume_preload() is False

    def test_a_failing_preload_does_not_escape(self, preload, monkeypatch):
        def boom():
            raise RuntimeError("driver said no")

        monkeypatch.setattr(mc_memory, "reinstate_pending", boom)
        mc_memory.preload_async(1024, 1024)
        mc_memory.join_preload(timeout=5)  # must not raise

        assert preload.gpu_loads == 0

    def test_starting_a_second_preload_waits_for_the_first(self, preload):
        mc_memory.preload_async(1024, 1024)
        mc_memory.preload_async(1024, 1024)
        mc_memory.join_preload(timeout=5)

        assert preload.reinstated == 2

    def test_the_setting_disables_it(self, preload, host):
        host.shared.opts.model_chain_preload_stage1 = False
        assert mc_memory.preload_async(1024, 1024) is False
        assert preload.reinstated == 0

    def test_joining_when_nothing_runs_is_a_no_op(self, preload):
        mc_memory.join_preload(timeout=5)

    def test_ensure_resident_waits_for_an_in_flight_preload(self, preload, monkeypatch):
        """Two threads must never move weights at once."""
        joins = []
        monkeypatch.setattr(mc_memory, "join_preload", lambda timeout=None: joins.append(timeout))
        monkeypatch.setattr(mc_memory, "checkpoint_info", lambda name: None)

        with pytest.raises(mc_memory.ModelChainError):
            mc_memory.ensure_resident("missing")

        assert joins, "ensure_resident must join the preload before touching models"


class TestPreloadIsGenerationReady:
    """"Preloaded" has to mean ready to sample, not "a thread ran".

    A preload that swapped the pointer and moved nothing looks identical from
    the outside to one that moved 13 GB, and the difference is the entire point
    of the feature. The worker therefore measures what the host reports resident
    and says which of the two happened.
    """

    def test_a_full_move_is_reported_ready(self, preload, monkeypatch):
        monkeypatch.setattr(mc_memory, "_loaded_residency", lambda: (13 * GB, 13 * GB))

        mc_memory.preload_async(1024, 1024)
        mc_memory.join_preload(timeout=5)

        assert mc_memory.preload_result().state == "ready"

    def test_a_partial_move_is_not_dressed_up_as_ready(self, preload, monkeypatch):
        monkeypatch.setattr(mc_memory, "_loaded_residency", lambda: (7 * GB, 13 * GB))

        mc_memory.preload_async(1024, 1024)
        mc_memory.join_preload(timeout=5)

        assert mc_memory.preload_result().state == "partial"

    def test_a_rounding_difference_does_not_demote_a_finished_preload(self, preload, monkeypatch):
        """model_size() and the host's loaded figure are computed differently."""
        monkeypatch.setattr(mc_memory, "_loaded_residency", lambda: (13 * GB - 1, 13 * GB))

        mc_memory.preload_async(1024, 1024)
        mc_memory.join_preload(timeout=5)

        assert mc_memory.preload_result().state == "ready"

    def test_nothing_to_do_is_its_own_answer(self, preload, monkeypatch):
        monkeypatch.setattr(mc_memory, "reinstate_pending", lambda: False)

        mc_memory.preload_async(1024, 1024)
        mc_memory.join_preload(timeout=5)

        assert mc_memory.preload_result().state == "nothing"

    def test_the_result_records_what_it_warmed(self, preload, monkeypatch):
        """So a later checkpoint change can be recognised as superseding it."""
        monkeypatch.setattr(mc_memory, "_loading_parameters_key", lambda: "key-A")

        mc_memory.preload_async(1024, 1024)
        mc_memory.join_preload(timeout=5)

        assert mc_memory.preload_result().key == "key-A"


class TestASingleStagePlanIsWarmedToo:
    """The preload's original trigger only ever fires on a chain.

    ``restore_selection`` is the last thing a Stage 2 pass does, and setting
    ``_pending_restore`` is what it leaves behind. A plan with no Stage 2 never
    swaps anything out, so it never had anything to swap back -- and "nothing to
    swap back" was read as "nothing to do". From a user's log, on a Stage-1-only
    plan with the warm-up on:

        warming up for this generation — cold — image model is cold
        warm-up finished in 0.0s — cold — image model is cold

    Twice. The model sat in system RAM between generations, every generation
    paid to move it back, and every LoRA was applied by walking weights across
    the bus first.
    """

    @pytest.fixture
    def single_stage(self, preload, memory, monkeypatch):
        """No pending restore, and the loaded model only partly on the card."""
        monkeypatch.setattr(mc_memory, "_pending_restore", None, raising=False)
        memory.mm.current_loaded_models[:] = [
            entry for entry in memory.mm.current_loaded_models
            if entry.model is not memory.model.patchers.unet
        ]
        return preload

    def test_the_task_is_to_finish_moving_what_is_already_selected(self, single_stage):
        assert mc_memory._preload_task() == mc_memory.RESIDENT

    def test_a_model_already_on_the_card_is_left_alone(self, single_stage, monkeypatch):
        monkeypatch.setattr(mc_memory, "_loaded_residency", lambda: (13 * GB, 13 * GB))
        assert mc_memory._preload_task() is None

    def test_it_starts_without_anything_having_been_swapped_out(self, single_stage):
        assert mc_memory.preload_async(1024, 1024) is True

    def test_it_moves_the_weights_without_swapping_the_pointer(self, single_stage):
        mc_memory.preload_async(1024, 1024)
        mc_memory.join_preload(timeout=5)

        assert single_stage.gpu_loads == 1
        assert single_stage.reinstated == 0, "there was nothing to reinstate"

    def test_it_makes_room_as_stage_1(self, single_stage):
        mc_memory.preload_async(1024, 1024)
        mc_memory.join_preload(timeout=5)

        assert [room[3] for room in single_stage.rooms] == [mc_memory.STAGE_1]

    def test_the_next_generation_budgets_against_its_own_size(self, single_stage):
        """Weights moved, so the budget the last generation worked out is stale."""
        mc_memory.preload_async(1024, 1024)
        mc_memory.join_preload(timeout=5)

        assert mc_memory.consume_preload() is True


class TestARoundingDifferenceIsNotAShortfall:
    """From a user's log, the whole of what an emergency eviction was for:

        freed 2.8 GB for the Stage 1 pass (2.2 GB -> 4.9 GB free): demoted the LLM
        reserve miss — Stage 1 exceeded the protected image budget by 0.0 GB;
            llama-server was emergency-evicted

    Zero point zero. A language-model process stopped, its prompt cache thrown
    away and a twenty-second restart bought, because a requirement estimated
    from a file size, an overhead fraction and the largest of four activation
    floors came out a hair above a driver reading taken a moment later.
    """

    def test_a_hairs_breadth_counts_as_fitting(self, memory, monkeypatch):
        reclaimed: list = []
        monkeypatch.setattr(mc_memory, "_reclaim_foreign",
                            lambda needed, reason="": reclaimed.append(needed) or 0)
        required = mc_memory.vram_required_bytes("B", None, 1024, 1024)
        monkeypatch.setattr(mc_memory, "free_vram_bytes", lambda: required - 1)

        assert mc_memory.make_vram_room("B", None, 1024, 1024) == 0
        assert not memory.mm.freed, "nothing should have been evicted"
        assert reclaimed == [], "and certainly not a language model"

    def test_a_real_shortfall_is_still_one(self, memory, monkeypatch):
        required = mc_memory.vram_required_bytes("B", None, 1024, 1024)
        short = required - mc_memory.FIT_TOLERANCE_BYTES * 4
        monkeypatch.setattr(mc_memory, "free_vram_bytes", lambda: short)

        mc_memory.make_vram_room("B", None, 1024, 1024)

        assert memory.mm.freed, "a genuine deficit still evicts"

    def test_the_tolerance_is_far_below_anything_worth_evicting_for(self):
        """Large enough to swallow any rounding difference, small enough that
        spilling it into the driver's fallback path costs nothing measurable."""
        assert 0 < mc_memory.FIT_TOLERANCE_BYTES <= GB // 4


class TestAWarmUpNeverTakesTheLlmsVram:
    """The user's rule, stated in their words:

        free VRAM for the image model when the image model needs it; the moment
        there is not enough VRAM to make an image, that is when the LLM goes

    A background warm-up is not making an image. Nobody has pressed anything,
    and stopping llama-server there costs a model load, a thrown-away prompt
    cache and possibly a conversation for a generation that may never be
    requested. It also buys nothing: the generation that does arrive runs
    _make_room_for_stage_1, asks the same question, and reclaims then -- when
    the need is real.
    """

    def test_the_warm_up_asks_for_no_foreign_reclaim(self, preload):
        mc_memory.preload_async(1024, 1024)
        mc_memory.join_preload(timeout=5)

        assert [room[4]["reclaim_foreign"] for room in preload.rooms] == [False]

    def test_a_shortfall_leaves_the_language_model_alone(self, memory, monkeypatch):
        reclaimed: list = []
        monkeypatch.setattr(mc_memory, "_reclaim_foreign",
                            lambda needed, reason="": reclaimed.append(needed) or 0)
        monkeypatch.setattr(mc_memory, "_llm_residency_bytes", lambda: 6 * GB)
        monkeypatch.setattr(mc_memory, "free_vram_bytes", lambda: 1 * GB)

        mc_memory.make_vram_room("B", None, 1024, 1024, stage=mc_memory.STAGE_1,
                                 reclaim_foreign=False)

        assert reclaimed == []

    def test_a_real_pass_still_takes_it(self, memory, monkeypatch):
        reclaimed: list = []
        monkeypatch.setattr(mc_memory, "_reclaim_foreign",
                            lambda needed, reason="": reclaimed.append(needed) or 0)
        monkeypatch.setattr(mc_memory, "free_vram_bytes", lambda: 1 * GB)

        mc_memory.make_vram_room("B", None, 1024, 1024, stage=mc_memory.STAGE_1)

        assert reclaimed, "a generation that does not fit is exactly when the LLM goes"

    def test_a_warm_up_shortfall_is_not_filed_as_a_reserve_miss(self, memory, monkeypatch):
        """No pass ran, so the plan has not been shown to be wrong about
        anything. Auto's learned cap is taught by generations, not warm-ups."""
        missed: list = []
        monkeypatch.setattr(mc_memory, "_record_reserve_miss",
                            lambda *a, **k: missed.append(a))
        monkeypatch.setattr(mc_memory, "_llm_residency_bytes", lambda: 6 * GB)
        monkeypatch.setattr(mc_memory, "free_vram_bytes", lambda: 1 * GB)

        mc_memory.make_vram_room("B", None, 1024, 1024, stage=mc_memory.STAGE_1,
                                 reclaim_foreign=False)

        assert missed == []

    def test_it_says_it_stood_down(self, memory, monkeypatch, caplog):
        monkeypatch.setattr(mc_memory, "_llm_residency_bytes", lambda: 6 * GB)
        monkeypatch.setattr(mc_memory, "free_vram_bytes", lambda: 1 * GB)

        with caplog.at_level("INFO", logger="model_chain"):
            mc_memory.make_vram_room("B", None, 1024, 1024, stage=mc_memory.STAGE_1,
                                     reclaim_foreign=False)

        assert any("nothing is being generated yet" in record.getMessage()
                   for record in caplog.records)

    def test_image_side_eviction_is_untouched(self, memory, monkeypatch):
        """The rule is about the LLM. Moving our own weights to system RAM is
        cheap, keeps them cached, and is what a warm swap has always done."""
        monkeypatch.setattr(mc_memory, "free_vram_bytes", lambda: 1 * GB)

        mc_memory.make_vram_room("B", None, 1024, 1024, stage=mc_memory.STAGE_1,
                                 reclaim_foreign=False)

        assert memory.mm.freed, "Forge's own eviction still runs"


class TestWarmingFromDisk:
    """The coldest run of a session is its first, and nothing was loaded then.

    Offered only to a caller that asks for it: reading a checkpoint off disk is
    the one job here that is expensive whether or not anybody is waiting for
    it, which makes it right for an explicit warm-up and wrong for the
    background pass that follows every generation.
    """

    @pytest.fixture
    def nothing_loaded(self, preload, host, monkeypatch):
        monkeypatch.setattr(mc_memory, "_pending_restore", None, raising=False)
        host.sd_models.model_data.sd_model = None
        return preload

    def test_the_background_pass_does_not_read_from_disk(self, nothing_loaded):
        assert mc_memory._preload_task() is None

    def test_an_explicit_warm_up_does(self, nothing_loaded):
        assert mc_memory._preload_task(allow_disk_load=True) == mc_memory.FROM_DISK

    def test_it_asks_the_host_to_load_and_then_moves_the_weights(self, nothing_loaded, host,
                                                                 memory, monkeypatch):
        reloads = []

        def forge_model_reload():
            reloads.append(True)
            host.sd_models.model_data.sd_model = memory.model
            return None, True

        monkeypatch.setattr(host.sd_models, "forge_model_reload", forge_model_reload)

        mc_memory.preload_async(1024, 1024, allow_disk_load=True)
        mc_memory.join_preload(timeout=5)

        assert reloads == [True]
        assert nothing_loaded.gpu_loads == 1

    def test_nothing_selected_is_not_a_failure(self, nothing_loaded, monkeypatch):
        monkeypatch.setattr(mc_memory, "checkpoint_info", lambda name: None)

        mc_memory.preload_async(1024, 1024, allow_disk_load=True)
        mc_memory.join_preload(timeout=5)

        assert mc_memory.preload_result().state == "nothing"
        assert mc_memory.preload_disabled_reason() is None


class TestTheWarmUpDoesNotNeedTwoSettings:
    """Turning the warm-up on is an answer about warmth.

    The preload setting permits a background thread after every generation.
    The warm-up setting asks for the pipeline to be loaded before somebody
    waits on it. Making the second depend on the first -- off by default, and
    named after the other half of the extension -- is how a user got a
    twenty-second llama-server start and an image model still in system RAM.
    """

    def test_an_explicit_warm_up_runs_with_the_preload_setting_off(self, preload, host):
        host.shared.opts.model_chain_preload_stage1 = False

        assert mc_memory.preload_async(1024, 1024) is False
        assert mc_memory.preload_async(1024, 1024, force=True) is True

    def test_forcing_does_not_overrule_the_circuit_breaker(self, preload, host, monkeypatch):
        """A machine where this does not work is one where it does not work."""
        host.shared.opts.model_chain_preload_stage1 = False
        monkeypatch.setattr(mc_memory, "_preload_disabled_reason", "it failed twice")

        assert mc_memory.preload_async(1024, 1024, force=True) is False

    def test_stage_2_is_still_captured_for_a_warm_up_that_will_consume_it(
            self, preload, host, monkeypatch):
        host.shared.opts.model_chain_preload_stage1 = False
        host.shared.opts.model_chain_warm_up = mc_arm.WARM_BEFORE
        monkeypatch.setattr(mc_memory, "model_data_sd_model", lambda: object())
        monkeypatch.setattr(mc_memory, "model_patchers", lambda model, attrs=None: ["a", "b"])

        assert mc_memory.capture_stage_2_components() == 2


class TestPreloadFailureIsNotALoop:
    """Enabling the preload may cost one generation. It may not cost every one.

    The acceptance criterion is blunt: turning this on cannot make all
    subsequent generations fail. Two mechanisms answer it -- a single failure
    leaves the model exactly where the host's synchronous path expects it, and
    repeated failures retire the feature.
    """

    @pytest.fixture
    def failing(self, preload, monkeypatch):
        def boom():
            raise RuntimeError("driver said no")

        monkeypatch.setattr(mc_memory, "reinstate_pending", boom)
        return preload

    def test_a_failure_is_recorded_rather_than_raised(self, failing):
        mc_memory.preload_async(1024, 1024)
        mc_memory.join_preload(timeout=5)

        assert mc_memory.preload_result().state == "failed"

    def test_one_failure_does_not_retire_the_feature(self, failing):
        mc_memory.preload_async(1024, 1024)
        mc_memory.join_preload(timeout=5)

        assert mc_memory.preload_enabled() is True

    def test_repeated_failures_do(self, failing):
        for _ in range(mc_memory.PRELOAD_FAILURE_LIMIT):
            mc_memory.preload_async(1024, 1024)
            mc_memory.join_preload(timeout=5)

        assert mc_memory.preload_enabled() is False
        assert mc_memory.preload_async(1024, 1024) is False

    def test_a_success_clears_the_count(self, preload, monkeypatch):
        """Otherwise one bad moment a session apart would eventually retire it."""
        calls = types.SimpleNamespace(n=0)

        def sometimes():
            calls.n += 1
            if calls.n == 1:
                raise RuntimeError("driver said no")
            return True

        monkeypatch.setattr(mc_memory, "reinstate_pending", sometimes)

        for _ in range(3):
            mc_memory.preload_async(1024, 1024)
            mc_memory.join_preload(timeout=5)

        assert mc_memory.preload_enabled() is True

    def test_toggling_the_setting_gives_it_another_chance(self, failing, host):
        for _ in range(mc_memory.PRELOAD_FAILURE_LIMIT):
            mc_memory.preload_async(1024, 1024)
            mc_memory.join_preload(timeout=5)

        host.shared.opts.model_chain_preload_stage1 = False
        assert mc_memory.preload_enabled() is False

        host.shared.opts.model_chain_preload_stage1 = True
        assert mc_memory.preload_enabled() is True

    def test_a_failure_after_the_swap_still_budgets_vram_next_time(self, preload, monkeypatch):
        """The swap happened, so Stage 1 *is* the loaded model -- with its
        weights in RAM. Forgetting that would leave the next generation loading
        it into whatever VRAM Stage 2 left behind, which is the slow path."""
        def boom():
            raise RuntimeError("driver said no")

        monkeypatch.setattr(mc_memory, "_load_current_to_gpu", boom)

        mc_memory.preload_async(1024, 1024)
        mc_memory.join_preload(timeout=5)

        assert mc_memory.consume_preload() is True

    def test_a_failure_drops_claims_that_may_no_longer_hold(self, preload, monkeypatch):
        """Pinned encoders and a prepared LoRA state are both beliefs about a
        load that did not finish."""
        monkeypatch.setattr(
            mc_memory, "_load_current_to_gpu",
            lambda: (_ for _ in ()).throw(RuntimeError("half way")),
        )
        mc_memory.capture_stage_1_encoders()

        mc_memory.preload_async(1024, 1024)
        mc_memory.join_preload(timeout=5)

        assert mc_memory._pinned_patchers == []


class TestStalePreloadIsSuperseded:
    """A preload warms what was selected when it ran. That can change."""

    def test_a_checkpoint_change_supersedes_the_preload(self, preload, monkeypatch):
        monkeypatch.setattr(mc_memory, "_loading_parameters_key", lambda: "key-A")
        mc_memory.preload_async(1024, 1024)
        mc_memory.join_preload(timeout=5)

        monkeypatch.setattr(mc_memory, "_loading_parameters_key", lambda: "key-B")

        assert mc_memory.consume_preload() is False

    def test_an_unchanged_selection_is_still_claimed(self, preload, monkeypatch):
        monkeypatch.setattr(mc_memory, "_loading_parameters_key", lambda: "key-A")

        mc_memory.preload_async(1024, 1024)
        mc_memory.join_preload(timeout=5)

        assert mc_memory.consume_preload() is True

    def test_a_superseded_result_is_not_left_lying_around(self, preload, monkeypatch):
        monkeypatch.setattr(mc_memory, "_loading_parameters_key", lambda: "key-A")
        mc_memory.preload_async(1024, 1024)
        mc_memory.join_preload(timeout=5)

        monkeypatch.setattr(mc_memory, "_loading_parameters_key", lambda: "key-B")
        mc_memory.consume_preload()

        assert mc_memory.preload_result() is None


class TestReadiness:
    """One line per generation saying how much of Stage 1 is already in VRAM.

    This is what makes the rest of the residency machinery checkable from the
    console rather than inferred from how long a generation felt.
    """

    def test_a_fully_resident_model_reports_warm(self, memory):
        state, message = mc_memory.stage_1_readiness()

        assert state == mc_memory.WARM
        assert "warm" in message

    def test_a_partly_resident_model_says_so(self, memory, monkeypatch):
        monkeypatch.setattr(mc_memory, "_loaded_residency", lambda: (5 * GB, 13 * GB))

        state, _ = mc_memory.stage_1_readiness()
        assert state == mc_memory.PARTIAL

    def test_a_model_in_system_ram_reports_cold(self, memory):
        memory.mm.current_loaded_models.clear()

        state, message = mc_memory.stage_1_readiness()
        assert state == mc_memory.COLD
        assert "system RAM" in message

    def test_a_selection_that_is_not_the_loaded_model_reports_cold(self, memory, host):
        host.sd_models.model_data.forge_hash = "key-B"
        host.sd_models.model_data.forge_loading_parameters = {"checkpoint": "A"}

        state, _ = mc_memory.stage_1_readiness()
        assert state == mc_memory.COLD

    def test_nothing_loaded_at_all_reports_cold(self, memory, host):
        host.sd_models.model_data.sd_model = None

        state, message = mc_memory.stage_1_readiness()
        assert state == mc_memory.COLD
        assert "disk" in message

    def test_a_successful_preload_is_credited(self, preload):
        mc_memory.preload_async(1024, 1024)
        mc_memory.join_preload(timeout=5)

        assert "preloaded in" in mc_memory.stage_1_readiness()[1]

    def test_a_failed_preload_is_explained(self, preload, monkeypatch):
        monkeypatch.setattr(
            mc_memory, "reinstate_pending",
            lambda: (_ for _ in ()).throw(RuntimeError("driver said no")),
        )
        mc_memory.preload_async(1024, 1024)
        mc_memory.join_preload(timeout=5)

        assert "driver said no" in mc_memory.stage_1_readiness()[1]

    def test_a_retired_preload_is_explained(self, preload, monkeypatch):
        monkeypatch.setattr(
            mc_memory, "reinstate_pending",
            lambda: (_ for _ in ()).throw(RuntimeError("driver said no")),
        )
        for _ in range(mc_memory.PRELOAD_FAILURE_LIMIT):
            mc_memory.preload_async(1024, 1024)
            mc_memory.join_preload(timeout=5)

        assert "off for this session" in mc_memory.stage_1_readiness()[1]

    def test_it_says_when_no_warm_up_is_enabled_at_all(self, memory):
        assert "no warm-up is enabled" in mc_memory.stage_1_readiness()[1]

    def test_it_says_nothing_on_a_generation_that_has_nothing_to_do_with_us(
        self, chain, host, image_factory, monkeypatch
    ):
        """Someone who never enables Model Chain must never see a line from it."""
        from test_orchestration import make_p, make_processed, run_chain

        reports = []
        monkeypatch.setattr(
            mc_memory, "stage_1_readiness", lambda: reports.append(1) or ("warm", "warm")
        )
        monkeypatch.setattr(mc_memory, "reinstate_pending", lambda: False)
        monkeypatch.setattr(mc_memory, "consume_preload", lambda: False)

        p = make_p(host)
        run_chain(chain, host, p, make_processed(host, p, image_factory), enabled=False)

        assert reports == []

    def test_it_reports_on_the_generation_after_a_chain(
        self, chain, host, image_factory, monkeypatch
    ):
        """The plain Generate that follows a chain is the interesting one --
        it is the generation the whole preload exists for."""
        from test_orchestration import make_p, make_processed, run_chain

        reports = []
        monkeypatch.setattr(
            mc_memory, "stage_1_readiness", lambda: reports.append(1) or ("warm", "warm")
        )
        monkeypatch.setattr(mc_memory, "reinstate_pending", lambda: True)
        monkeypatch.setattr(mc_memory, "make_vram_room", lambda *a, **k: 0)

        p = make_p(host)
        run_chain(chain, host, p, make_processed(host, p, image_factory), enabled=False)

        assert reports == [1]

    def test_a_readiness_report_that_fails_does_not_break_the_generation(self, chain, monkeypatch):
        """It is a log line. Nothing may depend on it, including itself."""
        monkeypatch.setattr(
            mc_memory, "stage_1_readiness",
            lambda: (_ for _ in ()).throw(RuntimeError("no")),
        )

        chain.script._report_readiness()  # must not raise


class TestLoadCurrentToGpu:
    def test_every_patcher_is_handed_to_the_host(self, memory):
        mc_memory._load_current_to_gpu()

        handed = {p.label for p in memory.mm.loaded_to_gpu[0]}
        assert handed == {"A-unet", "A-clip", "A-vae"}

    def test_a_model_without_patchers_is_skipped(self, memory, host):
        host.sd_models.model_data.sd_model = types.SimpleNamespace(name="bare")
        assert mc_memory._load_current_to_gpu() == 0
        assert memory.mm.loaded_to_gpu == []


# --------------------------------------------------------------------------- #
# How the orchestration drives all of the above
# --------------------------------------------------------------------------- #


@pytest.fixture
def wired(chain, host, monkeypatch, image_factory):
    """A full generation with the residency calls recorded rather than performed."""
    from test_orchestration import make_p, make_processed, run_chain

    calls = types.SimpleNamespace(captures=0, preloads=[], joins=0, rooms=[], consumed=False)

    monkeypatch.setattr(
        mc_memory, "capture_stage_1_encoders",
        lambda: calls.__setattr__("captures", calls.captures + 1) or 2,
    )
    monkeypatch.setattr(
        mc_memory, "preload_async",
        lambda w=0, h=0, **kwargs: calls.preloads.append((w, h)) or True,
    )
    monkeypatch.setattr(
        mc_memory, "join_preload",
        lambda timeout=None: calls.__setattr__("joins", calls.joins + 1),
    )
    monkeypatch.setattr(
        mc_memory, "make_vram_room",
        lambda name, mods=None, w=0, h=0, stage=mc_memory.STAGE_2, **kwargs:
            calls.rooms.append((w, h, stage)),
    )
    monkeypatch.setattr(mc_memory, "current_modules", lambda: [])

    def run(**overrides):
        p = make_p(host, **{k: overrides.pop(k) for k in ("width", "height") if k in overrides})
        return run_chain(chain, host, p, make_processed(host, p, image_factory), **overrides)

    return types.SimpleNamespace(run=run, calls=calls, chain=chain)


class TestOrchestrationWiring:
    def test_stage_2_captures_the_encoders_before_switching(self, wired):
        wired.run()
        assert wired.calls.captures == 1

    def test_a_preload_starts_once_stage_2_is_done(self, wired):
        wired.run()
        assert wired.calls.preloads == [(1024, 1024)]

    def test_the_preload_is_sized_from_the_stage_1_pass(self, wired):
        wired.run(width=768, height=512)
        assert wired.calls.preloads == [(768, 512)]

    def test_the_next_generation_joins_before_touching_models(self, wired):
        wired.run()
        assert wired.calls.joins >= 1

    def test_a_preloaded_swap_still_gets_its_vram_budget_rechecked(self, wired, monkeypatch):
        """The preload sized itself from the *previous* generation's resolution.

        Growing the image between generations would otherwise leave Stage 1
        running in a budget freed for a smaller pass.
        """
        monkeypatch.setattr(mc_memory, "reinstate_pending", lambda: False)
        monkeypatch.setattr(mc_memory, "consume_preload", lambda: True)

        wired.run(width=1536, height=1536)

        assert (1536, 1536, mc_memory.STAGE_1) in wired.calls.rooms

    def test_no_swap_and_no_preload_means_no_vram_work(self, wired, monkeypatch):
        monkeypatch.setattr(mc_memory, "reinstate_pending", lambda: False)
        monkeypatch.setattr(mc_memory, "consume_preload", lambda: False)

        wired.run()

        assert not [room for room in wired.calls.rooms if room[2] == mc_memory.STAGE_1]

    def test_a_generation_with_no_stage_2_warms_the_model_as_well(self, wired):
        """The whole of the user-visible bug, at the level it was missed.

        Everything above this drives a chain, and a chain is the only shape the
        preload was ever wired into. A plan with one stage ran the same models
        on the same card and got nothing.
        """
        wired.run(enabled=False)

        assert wired.calls.preloads == [(1024, 1024)]

    def test_the_chained_path_still_warms_exactly_once(self, wired):
        """Warming Stage 1 at the top of postprocess would move its weights onto
        the card in the moment Stage 2 is about to need the room."""
        wired.run()

        assert wired.calls.preloads == [(1024, 1024)]

    def test_a_language_model_on_the_card_gets_stage_1_a_budget(self, wired, monkeypatch):
        """The only route Stage 1 has to the cross-workload reclaim.

        make_vram_room is where _reclaim_foreign lives, and Stage 1 reached it
        only after a swap -- so on a plan that never swaps, llama-server was
        never asked to give ground for a pass that did not fit. Forge cannot
        ask on its own: those bytes are in another process.
        """
        monkeypatch.setattr(mc_memory, "reinstate_pending", lambda: False)
        monkeypatch.setattr(mc_memory, "consume_preload", lambda: False)
        monkeypatch.setattr(mc_memory, "llm_vram_on_the_image_card", lambda: 6 * GB)

        wired.run(enabled=False)

        assert (1024, 1024, mc_memory.STAGE_1) in wired.calls.rooms

    def test_a_card_with_no_language_model_on_it_is_left_alone(self, wired, monkeypatch):
        """Nothing to reclaim, so the host's own management is the whole answer
        -- and somebody who does not run a language model sees no line."""
        monkeypatch.setattr(mc_memory, "reinstate_pending", lambda: False)
        monkeypatch.setattr(mc_memory, "consume_preload", lambda: False)
        monkeypatch.setattr(mc_memory, "llm_vram_on_the_image_card", lambda: 0)

        wired.run(enabled=False)

        assert not [room for room in wired.calls.rooms if room[2] == mc_memory.STAGE_1]

    def test_a_failing_preload_does_not_break_the_generation(self, wired, monkeypatch):
        def boom(w=0, h=0, **kwargs):
            raise RuntimeError("thread refused to start")

        monkeypatch.setattr(mc_memory, "preload_async", boom)

        processed = wired.run()

        assert processed.images, "images must still be returned"


class TestSettingsRegistration:
    """The registered default and the code's fallback have to agree.

    ``option()`` falls back to a hardcoded default when a setting is missing, so
    a registration defaulting the other way would leave a feature off in the UI
    while the fallback reported it on.
    """

    def test_pinning_is_registered_on(self, host):
        import model_chain  # noqa: F401  (registration happens on import)

        name = mc_memory.OPT_PIN_ENCODERS
        assert host.shared.options_templates[name].default is True
        assert mc_memory.option(name, True) is True

    def test_every_setting_lands_in_a_section_that_is_drawn(self, host):
        """A None section id registers the setting but builds no control for it.

        That is the host's idiom for options it stores and never shows -- the
        same mechanism that keeps sd_checkpoint_hash off the Settings page. Used
        by mistake it produces a setting that exists, reads, and saves, but that
        the user can never find or change.
        """
        import model_chain

        assert model_chain.SETTINGS_SECTION[0] is not None

        ours = [
            option
            for name, option in host.shared.options_templates.items()
            if name.startswith("model_chain_")
        ]
        assert ours, "no Model Chain settings were registered at all"
        # Two sections, deliberately. Voice Chat is an install-level capability
        # with two downloads behind it, and burying it under a heading about
        # image model chaining would make it findable only by somebody who
        # already knew it was there.
        known = {model_chain.SETTINGS_SECTION, model_chain.VOICE_SECTION}
        for option in ours:
            assert option.section in known, (
                f"{option.label} is in {option.section}, which is neither of this "
                f"extension's settings sections")
            assert option.section[0] is not None, f"{option.label} would never be drawn"

    def test_the_two_sections_are_distinct_and_both_drawn(self, host):
        import model_chain  # noqa: F401

        assert model_chain.SETTINGS_SECTION != model_chain.VOICE_SECTION
        assert model_chain.VOICE_SECTION[0] is not None

    def test_the_host_value_wins_over_the_fallback(self, host):
        """Otherwise opting in to the preload could not turn it on."""
        host.shared.opts.model_chain_preload_stage1 = True
        assert mc_memory.option(mc_memory.OPT_PRELOAD, mc_memory.PRELOAD_DEFAULT) is True


class TestModelPatchers:
    def test_a_patcher_shared_between_slots_is_listed_once(self, memory):
        shared_patcher = FakePatcher("shared", 1 * GB)
        model = types.SimpleNamespace(
            forge_objects=types.SimpleNamespace(
                unet=shared_patcher, clip=FakeClip(shared_patcher), vae=shared_patcher
            )
        )
        assert mc_memory.model_patchers(model) == [shared_patcher]

    def test_the_placeholder_model_has_none(self, memory):
        class FakeInitialModel:
            forge_objects = None

        assert mc_memory.model_patchers(FakeInitialModel()) == []
