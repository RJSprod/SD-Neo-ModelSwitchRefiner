"""Placement negotiation: where the LLM goes, and what it says it changed.

Section 13's requirement has two halves and both are tested here. The first is
that hybrid mode degrades gracefully rather than choosing between "full GPU" and
"stop the image model". The second is the one that is easy to skip and matters
more: "The app must not quietly reduce context or quality-critical settings
without reporting what it changed."
"""

from __future__ import annotations

import dataclasses
import threading
import types
from pathlib import Path

import pytest

import mc_broker
import mc_gguf
import mc_llm_context as ctx
import mc_llm_runtime as runtime
from test_llm_context import build_model

_GB = 1024**3


@pytest.fixture
def placed(host, tmp_path, monkeypatch):
    """A configured install, an empty register, and a card we control."""
    mc_broker.clear()
    monkeypatch.setattr(ctx, "_store_path", lambda: tmp_path / "calibration.json")
    ctx.forget()
    for family in (mc_broker.FAMILY_IMAGE, mc_broker.FAMILY_LLM):
        mc_broker.unregister_reclaimer(family)
    monkeypatch.setattr(mc_broker, "safety_margin_bytes", lambda: 0)
    # Starting a server schedules a prompt-cache prime on a background thread,
    # which in a test would take the workload lock and then sit in httpx
    # waiting for a port nothing is listening on. Its own tests call the body
    # directly; everything else here is about placement.
    monkeypatch.setattr(runtime, "_prime_prompt_cache", lambda client: None)
    yield
    mc_broker.clear()
    ctx.forget()


def configure(monkeypatch, tmp_path, *, context=8192, mode="fixed", blocks=32,
              size_mb=4, gpu_layers="all", ceiling=131072, device_mode="gpu"):
    model = build_model(tmp_path, blocks=blocks, size_mb=size_mb, context=ceiling)
    # Written rather than merely named: ``Runtime.client`` refuses to start a
    # server whose executable is not there, which is the check that turns a
    # half-finished setup into a sentence instead of a traceback.
    server = tmp_path / "llama-server"
    server.write_bytes(b"")
    configuration = runtime.Config(
        runtime=server, model=model, mmproj=None, gpu_index=0,
        device="CUDA0", gpu_layers=gpu_layers, context_size=context, context_mode=mode,
        context_buffer_gb=4.0, kv_type_k="f16", kv_type_v="f16", mode=device_mode)
    monkeypatch.setattr(runtime, "config", lambda: configuration)
    return configuration


def set_free(monkeypatch, gigabytes):
    """A card with this much free, to both of the questions that asks.

    Both, because the two answers differ in life and the difference is a bug
    this file exists to keep fixed: the host counts its allocator's cached
    blocks as free and another process cannot have them. A test that set only
    one of them would be describing a machine that does not exist.
    """
    monkeypatch.setattr(mc_broker, "free_vram_bytes", lambda: int(gigabytes * _GB))
    monkeypatch.setattr(mc_broker, "device_free_vram_bytes", lambda: int(gigabytes * _GB))


class Recorder:
    def __init__(self, holds=0):
        self.holds = holds
        self.calls = []

    def release(self, needed_bytes, reason=""):
        self.calls.append((needed_bytes, reason))
        freed, self.holds = self.holds, 0
        return freed

    def resident_bytes(self):
        return self.holds

    def describe(self):
        return "the image checkpoint"


class TestItFits:
    def test_a_placement_that_fits_changes_nothing(self, placed, tmp_path, monkeypatch):
        configuration = configure(monkeypatch, tmp_path, context=8192)
        set_free(monkeypatch, 20)
        image = Recorder(holds=8 * _GB)
        mc_broker.register_reclaimer(mc_broker.FAMILY_IMAGE, image)

        negotiated = runtime.negotiate(configuration)

        assert negotiated.fits
        assert not negotiated.degraded
        assert negotiated.placement.context == 8192
        assert image.calls == []

    def test_the_model_ceiling_caps_a_larger_request(self, placed, tmp_path, monkeypatch):
        configuration = configure(monkeypatch, tmp_path, context=200_000, ceiling=8192)
        set_free(monkeypatch, 40)

        negotiated = runtime.negotiate(configuration)

        assert negotiated.placement.context == 8192

    def test_a_cpu_install_makes_no_vram_decision_at_all(self, placed, tmp_path, monkeypatch):
        configuration = configure(monkeypatch, tmp_path, gpu_layers="0")
        set_free(monkeypatch, 1)
        image = Recorder(holds=8 * _GB)
        mc_broker.register_reclaimer(mc_broker.FAMILY_IMAGE, image)

        negotiated = runtime.negotiate(configuration)

        assert negotiated.fits
        assert not negotiated.placement.on_gpu
        assert image.calls == []


class TestDegradation:
    def test_it_lowers_the_context_before_moving_a_checkpoint(self, placed, tmp_path,
                                                              host, monkeypatch):
        """Section 13's "least disruption": a context nobody is using is
        cheaper to give up than a model somebody is about to use -- and the
        checkpoint is not on the table at all."""
        configuration = configure(monkeypatch, tmp_path, context=131072)
        set_free(monkeypatch, 3)
        image = Recorder(holds=8 * _GB)
        mc_broker.register_reclaimer(mc_broker.FAMILY_IMAGE, image)

        negotiated = runtime.negotiate(configuration)

        assert negotiated.placement.context < 131072
        assert image.calls == []
        assert any("context reduced" in note for note in negotiated.notes)

    def test_it_reports_every_reduction_it_made(self, placed, tmp_path, monkeypatch):
        configuration = configure(monkeypatch, tmp_path, context=131072)
        set_free(monkeypatch, 3)

        negotiated = runtime.negotiate(configuration)

        assert negotiated.degraded
        assert negotiated.notes
        assert all(isinstance(note, str) and note for note in negotiated.notes)

    def test_it_shrinks_the_llm_instead_of_the_checkpoint(self, placed, tmp_path,
                                                          host, monkeypatch):
        configuration = configure(monkeypatch, tmp_path, context=131072)
        set_free(monkeypatch, 3)
        image = Recorder(holds=10 * _GB)
        mc_broker.register_reclaimer(mc_broker.FAMILY_IMAGE, image)

        negotiated = runtime.negotiate(configuration)

        assert image.calls == []
        assert negotiated.placement.context < 131072

    def test_a_model_that_cannot_fit_at_all_runs_from_system_ram(self, placed, tmp_path,
                                                                 host, monkeypatch):
        """The floor of the ladder, and the last thing the user asked for: when
        nothing is spare, the answer is system RAM, never the checkpoint."""
        configuration = configure(monkeypatch, tmp_path, context=131072, size_mb=64)
        set_free(monkeypatch, 0.03)
        image = Recorder(holds=20 * _GB)
        mc_broker.register_reclaimer(mc_broker.FAMILY_IMAGE, image)
        mc_broker.declare(mc_broker.FAMILY_IMAGE, "ckpt", "a checkpoint", 20 * _GB)

        negotiated = runtime.negotiate(configuration)

        assert image.calls == []
        assert negotiated.placement.gpu_layers == 0

    def test_offload_is_reduced_when_context_alone_cannot_save_it(self, placed, tmp_path,
                                                                  monkeypatch):
        """A model whose *weights* do not fit cannot be rescued by a smaller
        cache, so blocks move to system RAM instead -- graceful degradation
        rather than a refusal."""
        configuration = configure(monkeypatch, tmp_path, context=4096, size_mb=64)
        set_free(monkeypatch, 0.03)

        negotiated = runtime.negotiate(configuration)

        assert negotiated.placement.gpu_layers != ctx.ALL_LAYERS
        assert any("offload reduced" in note or "system RAM" in note
                   for note in negotiated.notes)

    def test_a_reduction_never_goes_below_the_usable_floor(self, placed, tmp_path,
                                                           monkeypatch):
        """Below a couple of thousand tokens a chat model is not one; the
        placement is reported as not fitting rather than made useless."""
        configuration = configure(monkeypatch, tmp_path, context=131072, size_mb=4)
        set_free(monkeypatch, 0.05)

        negotiated = runtime.negotiate(configuration)

        assert negotiated.placement.context >= runtime.MINIMUM_CONTEXT


class TestExclusiveMode:
    def test_the_image_family_is_not_swept_for_the_llm(self, placed, tmp_path,
                                                       host, monkeypatch):
        """Exclusive mode is the image family's ownership, not the LLM's. The
        sweep used to run both ways, and the reverse direction is what evicted
        a user's checkpoint on every Krea roll -- twice in one log -- so that
        the generation that followed spent thirteen seconds moving the same
        13.9 GB back onto the card.
        """
        host.shared.opts.set(mc_broker.OPT_MODE, mc_broker.MODE_EXCLUSIVE)
        configuration = configure(monkeypatch, tmp_path, context=8192)
        set_free(monkeypatch, 2)
        image = Recorder(holds=10 * _GB)
        mc_broker.register_reclaimer(mc_broker.FAMILY_IMAGE, image)
        mc_broker.declare(mc_broker.FAMILY_IMAGE, "ckpt", "a checkpoint", 10 * _GB)

        runtime.negotiate(configuration)

        assert image.calls == []

    def test_the_llm_is_placed_in_what_the_checkpoint_left(self, placed, tmp_path,
                                                           host, monkeypatch):
        """What replaces the sweep: the spare VRAM, and no more than that."""
        host.shared.opts.set(mc_broker.OPT_MODE, mc_broker.MODE_EXCLUSIVE)
        configuration = configure(monkeypatch, tmp_path, context=4096, size_mb=64,
                                  blocks=30)
        set_free(monkeypatch, 0.03)
        image = Recorder(holds=14 * _GB)
        mc_broker.register_reclaimer(mc_broker.FAMILY_IMAGE, image)

        negotiated = runtime.negotiate(configuration)

        assert image.calls == []
        assert 0 <= negotiated.placement.gpu_layers < 30


class TestAutomaticSizing:
    def test_automatic_mode_spends_what_is_free_on_context(self, placed, tmp_path,
                                                           monkeypatch):
        configuration = configure(monkeypatch, tmp_path, context=2048, mode="auto")
        set_free(monkeypatch, 12)

        negotiated = runtime.negotiate(configuration)

        assert negotiated.placement.context > 2048

    def test_automatic_mode_still_respects_the_model_ceiling(self, placed, tmp_path,
                                                             monkeypatch):
        configuration = configure(monkeypatch, tmp_path, context=2048, mode="auto",
                                  ceiling=8192)
        set_free(monkeypatch, 40)

        negotiated = runtime.negotiate(configuration)

        assert negotiated.placement.context == 8192


class TestUnconfigured:
    def test_it_says_what_to_do_rather_than_failing_obscurely(self, placed, monkeypatch):
        monkeypatch.setattr(runtime, "config", lambda: runtime.Config(
            runtime=None, model=None, mmproj=None, gpu_index=0, device="CUDA0",
            gpu_layers="all", context_size=8192, context_mode="fixed",
            context_buffer_gb=4.0, kv_type_k="f16", kv_type_v="f16"))

        with pytest.raises(runtime.NotConfigured, match="Choose a GGUF"):
            runtime.negotiate()


class TestReclaim:
    def test_a_stopped_runtime_frees_nothing_and_says_so(self, placed):
        assert runtime.runtime.release(4 * _GB, "an image pass") == 0

    def test_the_runtime_is_registered_as_the_llm_reclaimer(self, placed):
        """Importing the module is what wires the broker to it; a broken wiring
        would silently make every LLM demotion a no-op."""
        mc_broker.register_reclaimer(mc_broker.FAMILY_LLM, runtime.runtime)

        assert mc_broker._reclaimer(mc_broker.FAMILY_LLM) is runtime.runtime

    def test_status_is_answerable_before_anything_has_run(self, placed, tmp_path,
                                                          monkeypatch):
        configure(monkeypatch, tmp_path)

        status = runtime.runtime.status()

        assert status["configured"]
        assert not status["running"]
        assert status["resident_bytes"] == 0


class TestPreview:
    def test_a_preview_never_moves_anything(self, placed, tmp_path, host, monkeypatch):
        """The estimator panel is drawn on tab build and on every accordion
        open. Drawing a table must not cost somebody their checkpoint."""
        configuration = configure(monkeypatch, tmp_path, context=131072)
        set_free(monkeypatch, 2)
        image = Recorder(holds=12 * _GB)
        mc_broker.register_reclaimer(mc_broker.FAMILY_IMAGE, image)

        runtime.negotiate(configuration, reclaim=False)

        assert image.calls == []

    def test_a_preview_in_exclusive_mode_does_not_sweep(self, placed, tmp_path, host,
                                                        monkeypatch):
        host.shared.opts.set(mc_broker.OPT_MODE, mc_broker.MODE_EXCLUSIVE)
        configuration = configure(monkeypatch, tmp_path)
        set_free(monkeypatch, 2)
        image = Recorder(holds=10 * _GB)
        mc_broker.register_reclaimer(mc_broker.FAMILY_IMAGE, image)
        mc_broker.declare(mc_broker.FAMILY_IMAGE, "ckpt", "a checkpoint", 10 * _GB)

        runtime.negotiate(configuration, reclaim=False)

        assert image.calls == []

    def test_a_preview_still_reports_the_placement_it_would_use(self, placed, tmp_path,
                                                                monkeypatch):
        configuration = configure(monkeypatch, tmp_path, context=131072)
        set_free(monkeypatch, 3)

        negotiated = runtime.negotiate(configuration, reclaim=False)

        assert negotiated.placement.context < 131072
        assert negotiated.notes


# --------------------------------------------------------------------------- #
# A server that is already running
# --------------------------------------------------------------------------- #
#
# Everything below is one bug, seen from four sides. Placement is decided from
# free VRAM; a running llama-server is *why* free VRAM is low; so re-deciding a
# placement before every message read the model's own footprint as somebody
# else's and placed the next server in the gap it had just left. Each answer
# differed from the last, each difference stopped the server and started
# another one, and a card that had been holding all thirty layers ended up
# running two of them with the rest in system RAM -- while every restart also
# threw away llama.cpp's prompt cache, so each reply reprocessed the whole
# conversation before it wrote a word.


class FakeProcess:
    """A llama-server that starts instantly and holds nothing."""

    def __init__(self, started: list):
        self._started = started
        self.port = 8080
        self.api_key = "test"
        self.alive = False

    def start(self, *args, **kwargs):
        self._started.append((args, kwargs))
        self.alive = True

    def wait_ready(self, timeout=0):
        return None

    def stop(self):
        self.alive = False

    @property
    def running(self) -> bool:
        return self.alive


@pytest.fixture
def server(monkeypatch, tmp_path):
    """A runtime whose processes are fakes, and the log of what it started."""
    import mc_llm_paths

    monkeypatch.setattr(mc_llm_paths, "data_root", lambda: tmp_path / "data")
    # A fake server writes no log, and waiting five seconds for one it will
    # never write is the whole of what that wait costs here.
    monkeypatch.setattr(runtime, "OFFLOAD_WAIT_SECONDS", 0.0)
    started: list = []
    managed = runtime.Runtime()
    monkeypatch.setattr(managed, "_new_process", lambda: FakeProcess(started))
    yield managed, started
    managed.stop()


class TestAServerThatIsAlreadyRunning:
    def test_its_own_vram_is_not_read_as_somebody_else_s(self, placed, tmp_path, monkeypatch):
        """The whole bug, as arithmetic. 17 GB of the card is this model; a
        negotiation told only about the 5 GB beside it demotes the model that
        is already there to a corner of the card it is holding all of."""
        configuration = configure(monkeypatch, tmp_path, context=8192, size_mb=64)
        set_free(monkeypatch, 0.2)

        blind = runtime.negotiate(configuration)
        knowing = runtime.negotiate(configuration, already_ours=8 * _GB)

        assert blind.placement.gpu_layers != ctx.ALL_LAYERS  # degraded, as it must be
        assert knowing.placement.gpu_layers == ctx.ALL_LAYERS
        assert not knowing.degraded

    def test_a_second_request_reuses_the_server_rather_than_replacing_it(self, placed, server,
                                                                        tmp_path, monkeypatch):
        managed, started = server
        configure(monkeypatch, tmp_path, gpu_layers="all")
        set_free(monkeypatch, 20)

        managed.client()
        first = managed._process
        managed.client()

        assert len(started) == 1
        assert managed._process is first

    def test_the_card_filling_up_underneath_it_does_not_restart_it(self, placed, server,
                                                                   tmp_path, monkeypatch):
        """The reading that used to cause the thrash, made harmless. Free VRAM
        collapses between two messages -- which is what it looks like from the
        outside when the model itself is what is holding the card -- and the
        server that is running is left exactly where it is."""
        managed, started = server
        configure(monkeypatch, tmp_path, gpu_layers="all")
        set_free(monkeypatch, 20)
        managed.client()

        set_free(monkeypatch, 0.5)
        managed.client()

        assert len(started) == 1

    def test_a_server_that_could_now_hold_more_is_replaced(self, placed, server, tmp_path,
                                                           monkeypatch):
        """The other half of the same rule. A placement that was degraded when
        it was made is not permanent: once the card has room, one restart buys
        back every layer, and that restart is worth what it costs."""
        managed, started = server
        configure(monkeypatch, tmp_path, gpu_layers="all", size_mb=64)
        set_free(monkeypatch, 0.2)
        managed.client()
        assert managed._placement.gpu_layers != ctx.ALL_LAYERS

        set_free(monkeypatch, 20)
        managed.client()

        assert len(started) == 2
        assert managed._placement.gpu_layers == ctx.ALL_LAYERS

    def test_changing_the_model_still_replaces_it(self, placed, server, tmp_path, monkeypatch):
        """Settings are not placements. What the user chose is always honoured
        at once; only what the arithmetic chose is held on to."""
        managed, started = server
        configure(monkeypatch, tmp_path, gpu_layers="all")
        set_free(monkeypatch, 20)
        managed.client()

        configure(monkeypatch, tmp_path, gpu_layers="all", context=4096, mode="fixed")
        managed.client()

        assert len(started) == 2

    def test_a_warm_turn_asks_the_image_side_for_nothing(self, placed, server, tmp_path, host,
                                                         monkeypatch):
        """Exclusive mode is a promise about who owns the card, and it is kept
        when the server is placed. Re-sweeping before every message afterwards
        evicts a checkpoint the LLM already has room beside, and buys nothing:
        the VRAM this turn needs is VRAM this turn is already holding."""
        host.shared.opts.set(mc_broker.OPT_MODE, mc_broker.MODE_EXCLUSIVE)
        managed, _ = server
        configure(monkeypatch, tmp_path, gpu_layers="all")
        set_free(monkeypatch, 20)
        managed.client()

        image = Recorder(holds=8 * _GB)
        mc_broker.register_reclaimer(mc_broker.FAMILY_IMAGE, image)
        managed.client()

        assert image.calls == []


class TestWorthRestarting:
    def test_more_layers_is_worth_it(self):
        assert runtime._worth_restarting(ctx.Placement(gpu_layers=6, context=8192),
                                         ctx.Placement(gpu_layers=ctx.ALL_LAYERS, context=8192),
                                         30)

    def test_fewer_layers_never_is(self):
        """A running server holds its VRAM whether or not it is using all of
        it, so placing it smaller frees nothing anybody asked for -- the image
        side asks through ``release``, which stops the process outright."""
        assert not runtime._worth_restarting(ctx.Placement(gpu_layers=ctx.ALL_LAYERS),
                                             ctx.Placement(gpu_layers=2), 30)

    def test_a_little_more_context_is_not_worth_a_reload(self):
        assert not runtime._worth_restarting(ctx.Placement(context=7168),
                                             ctx.Placement(context=8192))

    def test_a_quarter_more_context_is(self):
        assert runtime._worth_restarting(ctx.Placement(context=8192),
                                         ctx.Placement(context=32768))

    def test_coming_back_off_system_ram_is(self):
        assert runtime._worth_restarting(ctx.Placement(on_gpu=False),
                                         ctx.Placement(gpu_layers=ctx.ALL_LAYERS), 30)


# --------------------------------------------------------------------------- #
# What llama.cpp says it did
# --------------------------------------------------------------------------- #
#
# Everything else in this file is about a decision. This is about evidence: a
# placement reported as "all layers on the GPU" that generates at a fifth of
# the speed a resident model generates at is a sentence and a fact that
# disagree, and the only place the fact was written down is llama-server's own
# log. Parsed leniently and on purpose -- it is somebody else's format, it has
# changed before, and a version this cannot read has to come back as "could not
# tell" rather than as "nothing was offloaded".

FULL_LOAD = """
load_tensors: loading model tensors, this can take a while... (mmap = true)
load_tensors: offloading 30 repeating layers to GPU
load_tensors: offloading output layer to GPU
load_tensors: offloaded 31/31 layers to GPU
load_tensors:        CUDA0 model buffer size = 17000.00 MiB
load_tensors:   CPU_Mapped model buffer size =   300.00 MiB
llama_kv_cache_unified:      CUDA0 KV buffer size =   896.00 MiB
"""

SPILLED_LOAD = """
load_tensors: offloaded 31/31 layers to GPU
load_tensors:        CUDA0 model buffer size = 11000.00 MiB
load_tensors:   CPU_Mapped model buffer size =  6000.00 MiB
"""


class TestReadingLlamaCppsOwnReport:
    def test_it_reads_the_layers_and_the_buffers(self):
        offload = runtime.read_offload(FULL_LOAD)

        assert (offload.layers, offload.total_layers) == (31, 31)
        assert offload.known
        assert round(offload.device_bytes / _GB, 1) == 16.6
        assert not offload.spilled

    def test_a_token_embedding_left_on_the_host_is_not_a_spill(self):
        """A full offload still leaves a small buffer on the CPU for many
        models. A warning that fires on every load is a warning nobody reads."""
        assert not runtime.read_offload(FULL_LOAD).spilled

    def test_weights_in_system_ram_are_reported_as_such(self):
        offload = runtime.read_offload(SPILLED_LOAD)

        assert offload.spilled
        assert round(offload.system_share, 2) == 0.35

    def test_pinned_host_memory_counts_as_system_memory(self):
        """CUDA_Host is pinned *system* memory. Reading its first word would
        put six gigabytes of system RAM on the card."""
        offload = runtime.read_offload(
            "load_tensors: CUDA0 model buffer size = 1000.00 MiB\n"
            "load_tensors: CUDA_Host model buffer size = 6000.00 MiB\n")

        assert offload.spilled

    def test_a_format_it_cannot_read_says_so_rather_than_guessing(self):
        offload = runtime.read_offload("llama_model_loader: loaded meta data\n")

        assert not offload.known
        assert not offload.spilled

    def test_only_this_start_is_read(self, tmp_path):
        """The log is appended to across runs, and a load that went well an
        hour ago must not answer for the one that just happened."""
        log = tmp_path / "llama-server.log"
        log.write_text(FULL_LOAD)
        offset = log.stat().st_size
        with log.open("a") as handle:
            handle.write(SPILLED_LOAD)

        assert runtime._offload_since(log, offset).spilled

    def test_a_missing_log_is_not_a_failed_load(self, tmp_path):
        assert not runtime._offload_since(tmp_path / "nothing.log", 0).known


class TestTheOffloadArgument:
    def test_all_layers_is_asked_for_as_a_number(self, tmp_path):
        """llama.cpp's own argument is an integer and every invocation of it in
        the wild passes one. "all" is this project's word."""
        model = mc_gguf.read(build_model(tmp_path, blocks=30))

        assert runtime._layers_argument(ctx.Placement(gpu_layers=ctx.ALL_LAYERS), model) == "31"

    def test_a_header_that_could_not_be_read_asks_for_far_too_many(self, tmp_path):
        """Clamped by llama.cpp, where a guess that came out too small would
        silently run half the model on the processor."""
        assert runtime._layers_argument(ctx.Placement(gpu_layers=ctx.ALL_LAYERS), None) == "999"

    def test_a_partial_offload_is_passed_as_it_was_negotiated(self, tmp_path):
        model = mc_gguf.read(build_model(tmp_path, blocks=30))

        assert runtime._layers_argument(ctx.Placement(gpu_layers=6), model) == "6"

    def test_no_offload_is_llama_cpps_own_token_for_it(self, tmp_path):
        model = mc_gguf.read(build_model(tmp_path, blocks=30))

        assert runtime._layers_argument(ctx.Placement(gpu_layers=ctx.NO_LAYERS), model) == "0"

    def test_the_server_is_started_with_the_number(self, placed, server, tmp_path, monkeypatch):
        managed, started = server
        configure(monkeypatch, tmp_path, gpu_layers="all", blocks=30)
        set_free(monkeypatch, 20)

        managed.client()

        assert started[0][1]["gpu_layers"] == "31"

    def test_a_mixed_install_fills_the_room_that_is_spare(
            self, placed, server, tmp_path, monkeypatch):
        """Mixed used to be pinned at zero layers, so a machine with a 3090 in
        it ran every matrix multiply on the processor while the card sat idle --
        which is the one thing the mode's own description promised it would not
        do. It now takes what is genuinely free."""
        managed, started = server
        configure(monkeypatch, tmp_path, gpu_layers="all", blocks=30, device_mode="mixed")
        set_free(monkeypatch, 20)

        managed.client()

        assert started[0][1]["gpu_layers"] != "0"

    def test_a_mixed_install_takes_nothing_when_nothing_is_free(
            self, placed, server, tmp_path, monkeypatch):
        """The old behaviour, kept as the floor: a full card means system RAM,
        which is slow and is still an answer."""
        managed, started = server
        configure(monkeypatch, tmp_path, gpu_layers="all", blocks=30, device_mode="mixed")
        set_free(monkeypatch, 0.2)

        managed.client()

        assert started[0][1]["gpu_layers"] == "0"

    def test_a_mixed_install_never_asks_the_image_side_to_move(
            self, placed, tmp_path, monkeypatch):
        """Somebody who picked the middle option did not ask for their
        checkpoint to be evicted so a prompt could be written faster -- and
        neither did anybody else. No placement asks; the negotiation spends
        spare VRAM and shrinks when there is none."""
        asked: list = []
        monkeypatch.setattr(runtime.mc_broker, "request_vram",
                            lambda *args, **kwargs: asked.append(args) or _NoRoom())

        configuration = configure(monkeypatch, tmp_path, gpu_layers="all",
                                  device_mode="mixed")
        set_free(monkeypatch, 1)
        runtime.negotiate(configuration)

        assert asked == []

    def test_a_gpu_install_never_asks_the_image_side_to_move_either(
            self, placed, tmp_path, monkeypatch):
        asked: list = []
        monkeypatch.setattr(runtime.mc_broker, "request_vram",
                            lambda *args, **kwargs: asked.append(args) or _NoRoom())

        configuration = configure(monkeypatch, tmp_path, gpu_layers="all", context=131072)
        set_free(monkeypatch, 1)
        runtime.negotiate(configuration)

        assert asked == []


class _NoRoom:
    """What ``request_vram`` answers with when nothing could be freed."""

    freed = 0
    note = ""

    def __bool__(self):
        return False


class TestTheLoadReportIsWaitedFor:
    def test_a_report_that_arrives_late_is_still_read(self, tmp_path, monkeypatch):
        """llama-server answers /health the moment the model is loaded, and what
        it wrote while loading is on the other side of somebody else's output
        buffer -- a block buffer, not a line one, when the output is a file.
        Reading once, immediately, found nothing at all on Windows."""
        log = tmp_path / "llama-server.log"
        log.write_text("")
        monkeypatch.setattr(runtime, "OFFLOAD_POLL_SECONDS", 0.01)
        monkeypatch.setattr(runtime, "OFFLOAD_WAIT_SECONDS", 2.0)

        reads = {"count": 0}
        real = runtime._offload_since

        def flushing(path, offset):
            reads["count"] += 1
            if reads["count"] == 3:
                log.write_text(FULL_LOAD)
            return real(path, offset)

        monkeypatch.setattr(runtime, "_offload_since", flushing)

        assert runtime._await_offload(log, 0).known

    def test_it_gives_up_rather_than_holding_the_load_open(self, tmp_path, monkeypatch):
        monkeypatch.setattr(runtime, "OFFLOAD_POLL_SECONDS", 0.01)
        monkeypatch.setattr(runtime, "OFFLOAD_WAIT_SECONDS", 0.05)

        assert not runtime._await_offload(tmp_path / "nothing.log", 0).known

    def test_the_log_path_is_named_on_every_start(self, placed, server, tmp_path, monkeypatch,
                                                  caplog):
        """A log nobody can find is a log nobody reads."""
        managed, _ = server
        configure(monkeypatch, tmp_path)
        set_free(monkeypatch, 20)

        with caplog.at_level("INFO", logger="model_chain"):
            managed.client()

        assert any("llama-server log" in record.getMessage() for record in caplog.records)

    def test_a_run_with_no_report_says_that_rather_than_nothing(self, placed, server, tmp_path,
                                                               monkeypatch, caplog):
        """With nothing after the placement line there is no way to tell a
        report that said everything was fine from one that was never read."""
        managed, _ = server
        configure(monkeypatch, tmp_path)
        set_free(monkeypatch, 20)

        with caplog.at_level("INFO", logger="model_chain"):
            managed.client()

        assert any("no load report" in record.getMessage() for record in caplog.records)


# --------------------------------------------------------------------------- #
# A second log format, and a start that did not come up
# --------------------------------------------------------------------------- #
#
# The parser above was written against the format llama.cpp had when it was
# written. A 2025 build logs none of those lines: it fits the model to the
# device itself, says so, and leaves the per-buffer accounting out of the
# server log entirely. What it does say is the context it settled on, what it
# saw free on the card, why a load failed, and -- after every request -- how
# fast that request actually ran. The fixture beside this file is one such
# build's own output, kept because a format nobody can reproduce from memory is
# a format that quietly stops being parsed.

OOM_LOG = (Path(__file__).resolve().parent / "data"
           / "llama-server-out-of-memory.log").read_text(encoding="utf-8")


class TestTheOtherLogFormat:
    def test_the_context_it_settled_on_is_read(self):
        """A build with its own fitter adjusts the context, and then the number
        this extension is reasoning about is not the one being run."""
        assert runtime.read_offload(
            "llama_context: n_ctx_seq (6912) < n_ctx_train (262144)").granted_context == 6912

    def test_what_llama_cpp_saw_free_is_read(self):
        offload = runtime.read_offload(OOM_LOG)

        assert offload.known
        assert offload.device_free[0][0] == "CUDA0"
        assert round(offload.device_free[0][2] / 1024) == 23

    def test_a_request_s_own_timings_are_read(self):
        prompt, reply = runtime.read_speed(
            "slot print_timing: prompt eval time = 3409.96 ms / 159 tokens "
            "( 21.45 ms per token, 46.63 tokens per second)\n"
            "slot print_timing: eval time = 5584.40 ms / 53 tokens "
            "( 105.37 ms per token, 9.49 tokens per second)\n")

        assert (round(prompt, 2), round(reply, 2)) == (46.63, 9.49)

    def test_the_last_request_wins(self):
        prompt, reply = runtime.read_speed(OOM_LOG + """
slot print_timing: eval time = 1 ms / 1 tokens ( 1 ms per token, 106.07 tokens per second)
slot print_timing: eval time = 1 ms / 1 tokens ( 1 ms per token, 2.68 tokens per second)
""")

        assert reply == 2.68


class TestAStartThatDidNotComeUp:
    def test_the_reason_is_llama_cpp_s_own(self):
        """"llama-server exited before becoming ready" is true of every failed
        start and useful for none of them."""
        failure = runtime.read_failure(OOM_LOG)

        assert failure.out_of_memory
        assert "17.8 GB" in failure.text and "22.8 GB reported free" in failure.text

    def test_a_card_with_room_can_still_refuse_one_allocation(self):
        """The whole reason this is retried rather than predicted: what a
        driver will hand out in one piece is not what it has left."""
        failure = runtime.read_failure(OOM_LOG)

        assert "in one piece" in failure.text

    def test_a_start_that_failed_for_another_reason_is_not_retried(self):
        failure = runtime.read_failure(
            "E llama_model_load: error loading model: tensor 'x' has wrong shape")

        assert failure and not failure.out_of_memory

    def test_nothing_to_say_is_said_as_nothing(self):
        assert not runtime.read_failure("srv update_slots: all slots are idle")


class TestTheCommandThatStartsIt:
    """The command line, as the vendored launcher actually assembles it.

    Reported from a user's ``llama-server.log``: every start of the language
    model died at load with ``invalid value for main_gpu: 0 (available devices:
    0)``, so LLM Studio could not answer and Creative Mode generated the prompt
    as typed on every press. The installation was on CPU placement, where the
    vendored launcher writes ``--device none`` -- no devices -- and
    ``--split-mode none --main-gpu 0`` -- use device 0 -- in the same line.
    """

    @pytest.fixture
    def launched(self, monkeypatch, tmp_path):
        """One real ``LlamaProcess.start``, with the OS boundary faked."""
        from prompt_master.inference import llama_process

        runtime.runtime._new_process()  # installs the repair, as a start does
        seen: list[list[str]] = []

        class FakePopen:
            def __init__(self, command, *args, **kwargs):
                seen.append([str(part) for part in command])
                self.args = command

            def poll(self):
                return None

            def terminate(self):
                pass

            def wait(self, timeout=None):
                return 0

        # Patched on the real module rather than on the stand-in, so the
        # stand-in's own Popen -- the thing under test -- still runs.
        import subprocess

        monkeypatch.setattr(subprocess, "Popen", FakePopen)

        def start(device, gpu_layers="all"):
            process = llama_process.LlamaProcess()
            process.start(tmp_path / "llama-server", tmp_path / "model.gguf", None,
                          0, device, 8192, tmp_path / "log.txt",
                          gpu_layers=gpu_layers)
            process.process = None
            return seen[-1]

        return start

    def test_a_gpu_placement_keeps_the_command_it_has_always_had(self, launched):
        command = launched("CUDA0")

        assert "--device" in command and command[command.index("--device") + 1] == "CUDA0"
        assert "--main-gpu" in command
        assert "--split-mode" in command

    def test_a_cpu_placement_does_not_select_a_gpu(self, launched):
        """The fix. ``--device none`` and ``--main-gpu 0`` cannot both be true,
        and llama.cpp refuses the pair rather than picking one."""
        command = launched("none", gpu_layers="0")

        assert command[command.index("--device") + 1] == "none"
        assert "--main-gpu" not in command
        assert "--split-mode" not in command

    def test_nothing_else_about_the_command_moves(self, launched):
        """Only the two flags come off. The model, the context, the offload and
        every other flag are the vendored launcher's and stay its.

        Compared as flags rather than as whole lines: the port and the API key
        are drawn fresh for every start, so two identical launches differ there
        by design."""
        def flags(command):
            return {part for part in command if part.startswith("--")}

        with_gpu = launched("CUDA0", gpu_layers="0")
        without = launched("none", gpu_layers="0")

        assert flags(with_gpu) - flags(without) == {"--split-mode", "--main-gpu"}
        assert flags(without) - flags(with_gpu) == set()
        for flag, value in (("--ctx-size", "8192"), ("--n-gpu-layers", "0")):
            assert without[without.index(flag) + 1] == value

    def test_the_repair_leaves_the_vendored_file_alone(self):
        """``prompt_master/`` is a byte-identical vendored tree, and its own
        VENDORED_FROM.txt says changes belong in the modules on top of it."""
        from pathlib import Path

        source = (Path(__file__).resolve().parent.parent / "prompt_master" /
                  "inference" / "llama_process.py").read_text(encoding="utf-8")

        assert '"--split-mode","none","--main-gpu","0"' in source

    def test_it_is_installed_once_however_many_servers_are_started(self):
        from prompt_master.inference import llama_process

        runtime.runtime._new_process()
        first = llama_process.subprocess
        runtime.runtime._new_process()

        assert llama_process.subprocess is first

    def test_everything_else_reaches_the_real_subprocess(self):
        """It stands in for a module, so it has to answer like one."""
        import subprocess

        from prompt_master.inference import llama_process

        runtime.runtime._new_process()

        assert llama_process.subprocess.TimeoutExpired is subprocess.TimeoutExpired
        assert llama_process.subprocess.PIPE == subprocess.PIPE

    def test_a_command_that_is_not_llama_server_is_not_touched(self):
        """The rewrite is keyed on the contradiction, not on the program: a
        device probe spawned while a server is starting passes through."""
        probe = ["nvidia-smi", "--query-gpu=index,name", "--format=csv"]

        assert runtime.without_gpu_selection(probe) == probe

    def test_the_failure_it_prevented_is_explained_if_it_happens_anyway(self):
        """An old state file, a CUDA_VISIBLE_DEVICES in the environment, or a
        runtime build with no CUDA backend beside it can still produce this, and
        llama.cpp's own sentence names the symptom and none of the cause."""
        failure = runtime.read_failure(
            "E llama_prepare_model_devices: invalid value for main_gpu: 0 "
            "(available devices: 0)\n"
            "E llama_model_load_from_file_impl: failed to load model\n"
            "E srv  llama_server: exiting due to model loading error")

        assert failure and not failure.out_of_memory
        assert "no GPU visible" in failure.text
        assert "CUDA_VISIBLE_DEVICES" in failure.text


class TestMovingTheExpertsRatherThanTheBlocks:
    """For a mixture-of-experts model, "offload less" should not mean "fewer
    layers". Experts are the great majority of the weights and are consulted a
    couple at a time; attention is small and every token touches it."""

    @pytest.fixture
    def moe(self):
        class Header:
            file_bytes = int(16.8 * 1024 ** 3)
            block_count = 40
            usable = True
            context_length = 262144
            embedding_length = 3584
            expert_count = 8
            expert_used_count = 2
            mixture_of_experts = True
            expert_share = 0.85

        return Header()

    @pytest.fixture
    def dense(self, moe):
        class Header(type(moe)):
            file_bytes = int(7.4 * 1024 ** 3)
            expert_count = 0
            expert_used_count = 0
            mixture_of_experts = False
            expert_share = 0.0

        return Header()

    @pytest.fixture
    def shrink(self, monkeypatch, tmp_path, host):
        monkeypatch.setattr(runtime, "runtime_supports", lambda flag, config=None: True)
        monkeypatch.setattr(ctx, "estimate", lambda model, placement, header: ctx.Estimate(
            model=model, context=placement.context, ceiling=0,
            weights_bytes=ctx.weights_bytes(header, placement),
            kv_bytes=int(0.8 * 1024 ** 3), compute_bytes=int(0.4 * 1024 ** 3),
            kv_bytes_per_token=96.0, calibrated=False, placement=placement))
        configuration = runtime.Config(
            runtime=tmp_path / "llama-server", model=tmp_path / "m.gguf", mmproj=None,
            gpu_index=0, device="CUDA0", gpu_layers="all", context_size=8192,
            context_mode="fixed", context_buffer_gb=4.0, kv_type_k="f16", kv_type_v="f16")

        def run(header, free_gb):
            monkeypatch.setattr(runtime, "_free_vram",
                                lambda ours=0, gigabytes=free_gb: int(gigabytes * 1024 ** 3))
            placement = ctx.Placement(gpu_layers=ctx.ALL_LAYERS, context=8192)
            return runtime._shrink_offload(configuration, placement, header, 0)

        return run

    def test_a_moe_keeps_every_block_on_the_card(self, shrink, moe):
        """6.4 GB free against a 16.8 GB model, and all forty blocks stay
        resident because only the experts left."""
        placement, _estimate, note = shrink(moe, 6.4)

        assert placement.cpu_experts is True
        assert placement.gpu_layers == ctx.ALL_LAYERS
        assert "stay in system RAM" in note

    def test_a_dense_model_still_drops_blocks(self, shrink, dense):
        placement, _estimate, note = shrink(dense, 6.4)

        assert placement.cpu_experts is False
        assert 0 < placement.gpu_layers < 40
        assert "of 40 layers" in note

    def test_the_experts_come_out_before_any_block_does(self, shrink, moe):
        """Order matters: moving experts costs far less speed than moving
        attention, so it is tried first and blocks only if it is not enough."""
        placement, _estimate, _note = shrink(moe, 3)

        assert placement.cpu_experts is True
        assert 0 < placement.gpu_layers < 40

    def test_nothing_free_is_still_system_RAM(self, shrink, moe):
        placement, _estimate, note = shrink(moe, 0.5)

        assert placement.gpu_layers == ctx.NO_LAYERS
        assert "system RAM" in note

    def test_a_build_without_the_flag_is_not_given_the_placement(self, shrink, moe,
                                                                 monkeypatch):
        """The placement means ``--cpu-moe``, so a runtime that has never heard
        of it must not be handed one."""
        monkeypatch.setattr(runtime, "runtime_supports", lambda flag, config=None: False)
        placement, _estimate, _note = shrink(moe, 6.4)

        assert placement.cpu_experts is False

    def test_the_estimate_knows_the_experts_are_elsewhere(self, moe):
        whole = ctx.weights_bytes(moe, ctx.Placement(gpu_layers=ctx.ALL_LAYERS))
        without = ctx.weights_bytes(
            moe, ctx.Placement(gpu_layers=ctx.ALL_LAYERS, cpu_expert_layers=ctx.ALL_EXPERTS))

        assert without < whole / 4

    def test_the_placement_says_so_out_loud(self):
        said = ctx.Placement(gpu_layers=ctx.ALL_LAYERS, cpu_expert_layers=ctx.ALL_EXPERTS).describe(40)

        assert "experts in system RAM" in said

    def test_calibration_tells_the_two_apart(self):
        """They are different footprints for the same layer count, so a measured
        one must not be filed under the other."""
        assert ctx.Placement(gpu_layers=-1).key != ctx.Placement(
            gpu_layers=-1, cpu_expert_layers=ctx.ALL_EXPERTS).key
        assert ctx.Placement(gpu_layers=-1, cpu_expert_layers=8).key != ctx.Placement(
            gpu_layers=-1, cpu_expert_layers=10).key


class TestProgressiveExpertOffload:
    """Move the experts of as few blocks as the shortfall needs, not all of them.

    ``--cpu-moe`` is a cliff. A 16.8 GB model two gigabytes too large for a card
    used to answer that by reading thirty-four blocks of experts over the bus
    for the rest of the session, when six blocks' worth covered the gap. The
    ladder now has rungs: ``--n-cpu-moe N``, stepped up if the arithmetic was
    optimistic, then ``--cpu-moe``, and only then whole blocks.
    """

    @pytest.fixture
    def moe(self):
        class Header:
            file_bytes = int(16.8 * 1024 ** 3)
            block_count = 40
            usable = True
            context_length = 262144
            embedding_length = 3584
            expert_count = 8
            expert_used_count = 2
            mixture_of_experts = True
            expert_share = 0.85

        return Header()

    @pytest.fixture
    def shrink(self, monkeypatch, tmp_path, host):
        """``_shrink_offload`` against a card of a given size and a given build.

        ``flags`` is what this llama-server advertises, which is the whole of
        what decides which rung is reachable -- a flag an older build has never
        heard of is not a slower server, it is one that exits at startup.
        """
        monkeypatch.setattr(ctx, "estimate", lambda model, placement, header: ctx.Estimate(
            model=model, context=placement.context, ceiling=0,
            weights_bytes=ctx.weights_bytes(header, placement),
            kv_bytes=int(0.8 * 1024 ** 3), compute_bytes=int(0.4 * 1024 ** 3),
            kv_bytes_per_token=96.0, calibrated=False, placement=placement))
        configuration = runtime.Config(
            runtime=tmp_path / "llama-server", model=tmp_path / "m.gguf", mmproj=None,
            gpu_index=0, device="CUDA0", gpu_layers="all", context_size=8192,
            context_mode="fixed", context_buffer_gb=4.0, kv_type_k="f16", kv_type_v="f16")

        def run(header, free_gb, flags=(runtime.N_CPU_MOE_FLAG, runtime.CPU_MOE_FLAG),
                floor=ctx.NO_EXPERTS):
            monkeypatch.setattr(runtime, "runtime_supports",
                                lambda flag, config=None, offered=flags: flag in offered)
            monkeypatch.setattr(runtime, "_free_vram",
                                lambda ours=0, gigabytes=free_gb: int(gigabytes * 1024 ** 3))
            placement = ctx.Placement(gpu_layers=ctx.ALL_LAYERS, context=8192)
            return runtime._shrink_offload(configuration, placement, header, 0, 0, floor)

        return run

    def test_it_moves_the_fewest_blocks_that_cover_the_shortfall(self, shrink, moe):
        """18.0 GB wanted against 16.0 GB free. Each block's experts are worth
        about 0.36 GB, so six of forty is the answer and thirty-four blocks keep
        theirs on the card."""
        placement, _estimate, note = shrink(moe, 16.0)

        assert placement.cpu_expert_layers == 6
        assert placement.gpu_layers == ctx.ALL_LAYERS
        assert "6 of 40 layers" in note

    def test_a_bigger_shortfall_moves_more_of_them(self, shrink, moe):
        smaller, _e, _n = shrink(moe, 16.0)
        bigger, _e, _n = shrink(moe, 12.0)

        assert bigger.cpu_expert_layers > smaller.cpu_expert_layers
        assert bigger.gpu_layers == ctx.ALL_LAYERS

    def test_a_placement_that_already_fits_moves_nothing(self, shrink, moe):
        placement, _estimate, note = shrink(moe, 24.0)

        assert placement.cpu_expert_layers == ctx.NO_EXPERTS
        assert note == ""

    def test_when_every_block_must_give_them_up_it_says_cpu_moe(self, shrink, moe):
        """Design intent section 6.6. ``--cpu-moe`` is the spelling every build
        that has either flag understands, so the end of the ladder uses it."""
        placement, _estimate, _note = shrink(moe, 4.0)

        assert placement.cpu_expert_layers == ctx.ALL_EXPERTS

    def test_a_build_with_only_the_old_flag_keeps_the_old_behaviour(self, shrink, moe):
        placement, _estimate, _note = shrink(moe, 16.0, flags=(runtime.CPU_MOE_FLAG,))

        assert placement.cpu_expert_layers == ctx.ALL_EXPERTS
        assert placement.gpu_layers == ctx.ALL_LAYERS

    def test_a_build_with_only_the_new_flag_can_still_move_them_all(self, shrink, moe):
        """``--n-cpu-moe 40`` says what ``--cpu-moe`` says, and is the only
        thing such a build would accept."""
        placement, _estimate, _note = shrink(moe, 4.0, flags=(runtime.N_CPU_MOE_FLAG,))

        assert placement.cpu_expert_layers == 40

    def test_a_build_with_neither_flag_is_given_neither(self, shrink, moe):
        placement, _estimate, note = shrink(moe, 16.0, flags=())

        assert placement.cpu_expert_layers == ctx.NO_EXPERTS
        assert "experts" not in note

    def test_blocks_leave_only_after_the_experts_have(self, shrink, moe):
        """Section 6.7. Attention is small and every token touches it, so it is
        the last thing to go."""
        placement, _estimate, note = shrink(moe, 2.0)

        assert placement.cpu_expert_layers == ctx.ALL_EXPERTS
        assert placement.gpu_layers != ctx.ALL_LAYERS
        assert "stay in system RAM" in note
        assert "layers on the GPU" in note

    def test_a_floor_carried_in_from_a_failed_start_wins(self, shrink, moe):
        """The arithmetic says six and a real load that ran out of memory says
        the arithmetic was short. Evidence beats planning."""
        placement, _estimate, _note = shrink(moe, 16.0, floor=12)

        assert placement.cpu_expert_layers == 12

    def test_a_floor_below_what_is_needed_changes_nothing(self, shrink, moe):
        placement, _estimate, _note = shrink(moe, 16.0, floor=2)

        assert placement.cpu_expert_layers == 6

    def test_a_start_that_failed_with_every_expert_out_does_not_get_fewer_back(
            self, shrink, moe):
        """Arithmetic that has just been shown to be optimistic does not get to
        place the next server somewhere worse than the one that failed."""
        placement, _estimate, _note = shrink(moe, 16.0, floor=ctx.ALL_EXPERTS)

        assert placement.cpu_expert_layers == ctx.ALL_EXPERTS

    def test_the_next_floor_is_two_more_blocks(self):
        assert runtime._next_expert_floor(ctx.Placement(cpu_expert_layers=6)) == 8
        assert runtime._next_expert_floor(ctx.Placement(cpu_expert_layers=8)) == 10

    def test_a_placement_with_no_experts_moved_yet_gets_no_floor(self):
        """The added headroom will have the ladder compute a real number for
        itself, and a floor invented here would only get in its way."""
        assert runtime._next_expert_floor(ctx.Placement()) == ctx.NO_EXPERTS

    def test_a_placement_that_moved_them_all_has_nothing_left_at_this_rung(self):
        assert runtime._next_expert_floor(
            ctx.Placement(cpu_expert_layers=ctx.ALL_EXPERTS)) == ctx.ALL_EXPERTS

    def test_a_changed_split_is_a_different_server(self, tmp_path):
        """Start-time arguments, both of them: a signature that could not tell
        the two apart would hand back the server that had just run out of
        memory."""
        configuration = runtime.Config(
            runtime=tmp_path / "llama-server", model=tmp_path / "m.gguf", mmproj=None,
            gpu_index=0, device="CUDA0", gpu_layers="all", context_size=8192,
            context_mode="fixed", context_buffer_gb=4.0, kv_type_k="f16", kv_type_v="f16")

        six = runtime._signature_of(configuration, None, ctx.Placement(cpu_expert_layers=6))
        eight = runtime._signature_of(configuration, None, ctx.Placement(cpu_expert_layers=8))

        assert six != eight

    def test_getting_the_experts_back_is_worth_a_restart(self):
        """The finer steps are only useful if the ladder can also be climbed:
        the same blocks on the card with fewer of their experts elsewhere is
        more of the model resident."""
        assert runtime._worth_restarting(ctx.Placement(cpu_expert_layers=12),
                                         ctx.Placement(cpu_expert_layers=4), 40)
        assert not runtime._worth_restarting(ctx.Placement(cpu_expert_layers=4),
                                             ctx.Placement(cpu_expert_layers=12), 40)
        assert runtime._worth_restarting(ctx.Placement(cpu_expert_layers=ctx.ALL_EXPERTS),
                                         ctx.Placement(cpu_expert_layers=8), 40)


class TestAskingTheBuildWhatItSupports:
    """Two flags are worth adding and neither may be guessed at.

    The runtime is whatever build the user copied in; a flag it does not know
    is not a slower server but a server that exits at startup, which is a
    failure this extension has already spent a week on.
    """

    @pytest.fixture(autouse=True)
    def forget(self):
        runtime._capabilities.clear()
        runtime._arm_flags([])
        yield
        runtime._capabilities.clear()
        runtime._arm_flags([])

    @pytest.fixture
    def build(self, tmp_path, monkeypatch):
        """A fake llama-server whose --help says what we tell it to."""
        executable = tmp_path / "llama-server"
        executable.write_text("")

        def announce(text):
            monkeypatch.setattr(
                runtime.subprocess, "run",
                lambda *args, **kwargs: types.SimpleNamespace(stdout=text, stderr=""))
            return runtime.Config(
                runtime=executable, model=tmp_path / "model.gguf", mmproj=None,
                gpu_index=0, device="CUDA0", gpu_layers="all", context_size=8192,
                context_mode="fixed", context_buffer_gb=4.0, kv_type_k="f16",
                kv_type_v="f16")

        return announce

    def test_a_build_that_lists_a_flag_may_be_given_it(self, build):
        configuration = build("  -fa, --flash-attn        enable Flash Attention\n"
                              "      --cpu-moe            keep all MoE weights in RAM\n")

        assert runtime.runtime_supports(runtime.CPU_MOE_FLAG, configuration)
        assert runtime.runtime_supports(runtime.FLASH_ATTENTION_FLAG, configuration)

    def test_a_build_that_does_not_is_given_neither(self, build):
        configuration = build("  -m, --model FNAME\n  -c, --ctx-size N\n")
        placement = ctx.Placement(gpu_layers=20, cpu_expert_layers=ctx.ALL_EXPERTS)

        assert runtime.accelerator_flags(configuration, placement) == []

    def test_flash_attention_is_not_added_when_nothing_is_offloaded(self, build):
        """It is a CUDA kernel. On a placement with no resident layers it would
        be a flag that changes nothing, and a flag that changes nothing is one
        somebody will later believe changed something."""
        configuration = build("      --flash-attn\n      --cpu-moe\n")
        placement = ctx.Placement(gpu_layers=ctx.NO_LAYERS)

        assert runtime.FLASH_ATTENTION_FLAG not in runtime.accelerator_flags(
            configuration, placement)

    def test_no_gpu_flag_reaches_a_cpu_placement(self, build):
        configuration = build("      --flash-attn\n      --cpu-moe\n")
        placement = ctx.Placement(gpu_layers=20, on_gpu=False,
                                  cpu_expert_layers=ctx.ALL_EXPERTS)

        assert runtime.accelerator_flags(configuration, placement) == []

    def test_the_full_attention_window_reaches_every_placement(self, build):
        """The one flag that is not about the card. What it buys is prompt
        *reuse*: a sliding-window model can only resume a cached prompt at a
        context checkpoint, so a warm turn that shared 668 tokens with the last
        one resumed at 460 and processed the other 208 again -- seven seconds,
        on the processor-only placement in the log this came from."""
        configuration = build("      --swa-full\n")

        for placement in (ctx.Placement(gpu_layers=20),
                          ctx.Placement(gpu_layers=ctx.NO_LAYERS),
                          ctx.Placement(gpu_layers=20, on_gpu=False)):
            assert runtime.accelerator_flags(configuration, placement) == [
                runtime.FULL_ATTENTION_WINDOW_FLAG], placement

    def test_a_build_without_the_full_window_is_not_given_it(self, build):
        configuration = build("  -m, --model FNAME\n")

        assert runtime.accelerator_flags(configuration, ctx.Placement(gpu_layers=20)) == []

    def test_it_comes_before_the_card_flags(self, build):
        configuration = build("      --swa-full\n      --flash-attn\n      --cpu-moe\n")
        placement = ctx.Placement(gpu_layers=20, cpu_expert_layers=ctx.ALL_EXPERTS)

        assert runtime.accelerator_flags(configuration, placement) == [
            runtime.FULL_ATTENTION_WINDOW_FLAG,
            runtime.CPU_MOE_FLAG,
            runtime.FLASH_ATTENTION_FLAG,
        ]

    def test_the_three_state_spelling_is_given_its_value(self, build):
        """llama.cpp changed --flash-attn from a switch to on/off/auto and both
        spellings are in the wild. The wrong one is a server that will not
        start, so the help text decides."""
        configuration = build("  -fa, --flash-attn {on,off,auto}   (default: auto)\n")
        placement = ctx.Placement(gpu_layers=20)

        assert runtime.accelerator_flags(configuration, placement) == [
            runtime.FLASH_ATTENTION_FLAG, "on"]

    def test_the_answer_is_cached_per_binary(self, build, monkeypatch):
        configuration = build("      --cpu-moe\n")
        runtime.runtime_capabilities(configuration)

        calls: list = []
        monkeypatch.setattr(runtime.subprocess, "run",
                            lambda *a, **k: calls.append(a) or types.SimpleNamespace(
                                stdout="", stderr=""))
        runtime.runtime_capabilities(configuration)

        assert calls == []

    def test_a_build_that_will_not_answer_is_given_nothing(self, tmp_path, monkeypatch):
        executable = tmp_path / "llama-server"
        executable.write_text("")

        def explode(*args, **kwargs):
            raise OSError("not executable here")

        monkeypatch.setattr(runtime.subprocess, "run", explode)
        configuration = runtime.Config(
            runtime=executable, model=tmp_path / "model.gguf", mmproj=None, gpu_index=0,
            device="CUDA0", gpu_layers="all", context_size=8192, context_mode="fixed",
            context_buffer_gb=4.0, kv_type_k="f16", kv_type_v="f16")

        assert runtime.runtime_capabilities(configuration) == frozenset()


class TestFlagsReachTheCommand:
    @pytest.fixture(autouse=True)
    def clean(self):
        runtime._arm_flags([])
        yield
        runtime._arm_flags([])

    def _command(self):
        return ["llama-server", "--model", "m.gguf", "--ctx-size", "8192",
                "--device", "CUDA0"]

    def test_an_armed_flag_is_appended_once(self):
        runtime._arm_flags(["--cpu-moe"])

        assert runtime.with_extra_flags(self._command())[-1] == "--cpu-moe"
        assert "--cpu-moe" not in runtime.with_extra_flags(self._command())

    def test_nothing_armed_changes_nothing(self):
        assert runtime.with_extra_flags(self._command()) == self._command()

    def test_a_command_that_is_not_llama_server_is_left_alone(self):
        """A device probe spawned while a start is in flight must not collect
        the flags meant for the server."""
        runtime._arm_flags(["--cpu-moe"])
        probe = ["nvidia-smi", "--query-gpu=index", "--format=csv"]

        assert runtime.with_extra_flags(probe) == probe


class TestWhatThisMachineMeasured:
    """llama.cpp measures both rates for every request it serves. Until now the
    extension printed them in a log line and threw them away, while estimating
    the same quantities from character counts."""

    @pytest.fixture(autouse=True)
    def clean(self, timing_store):
        import mc_progress

        mc_progress.forget()
        return timing_store

    def test_a_measurement_is_kept_against_the_backbone_that_made_it(self):
        runtime.remember_speed(30.0, 12.8, identity="gemma4-26b-a4b")
        runtime.remember_speed(24.0, 4.9, identity="gemma4-12b-qat")

        assert runtime.measured_speed("gemma4-26b-a4b")[1] == pytest.approx(12.8)
        assert runtime.measured_speed("gemma4-12b-qat")[1] == pytest.approx(4.9)

    def test_the_two_backbones_do_not_share_one_number(self):
        """The reason this is keyed at all. Measured on one machine in system
        RAM: a dense 12B wrote at 4.9 tokens a second, a 26B mixture-of-experts
        at 12.8 -- the larger file two and a half times faster, because
        generation from RAM is bandwidth-bound and an MoE activates a fraction
        of its weights per token."""
        runtime.remember_speed(30.0, 12.8, identity="gemma4-26b-a4b")

        assert runtime.measured_speed("gemma4-12b-qat") == (0.0, 0.0)

    def test_an_id_and_a_filename_reach_the_same_key(self):
        """The running configuration and a catalogue entry come by different
        routes and have to agree, or the catalogue shows nothing."""
        assert runtime.speed_key(runtime.WRITE_RATE, "Gemma4 12B/QAT") == \
            runtime.speed_key(runtime.WRITE_RATE, "gemma4-12b-qat")

    def test_nothing_measured_is_reported_as_nothing(self):
        assert runtime.measured_speed("never-run-here") == (0.0, 0.0)

    def test_the_specific_key_is_asked_first(self):
        keys = runtime.speed_keys("krea:write", "gemma4-12b-qat")

        assert keys[0] == "krea:write:gemma4-12b-qat"
        assert keys[-1] == "krea:write"

    def test_the_catalogue_says_what_this_machine_measured(self, monkeypatch):
        """Size is a bad proxy for speed and the catalogue was implying
        otherwise."""
        import mc_llm_managed_models as managed

        runtime.remember_speed(30.0, 12.8, identity="gemma4-26b-a4b-balanced")
        entry = managed.catalogue()[-1]

        assert entry.identifier == "gemma4-26b-a4b-balanced"
        assert "measured here: 12.8 tokens/s" in entry.describe()

    def test_a_backbone_nobody_has_run_claims_nothing(self):
        import mc_llm_managed_models as managed

        entry = managed.catalogue()[0]

        assert "measured here" not in entry.describe()

    def test_a_rate_is_kept_against_the_placement_that_produced_it(self):
        """Section 9. The same backbone writes at forty tokens a second resident
        and at five from system RAM, and one number covering both is a number
        that is wrong for each of them."""
        resident = ctx.Placement()
        spilled = ctx.Placement(cpu_expert_layers=8)

        runtime.remember_speed(300.0, 41.0, identity="gemma-26b", placement=resident)
        runtime.remember_speed(90.0, 12.0, identity="gemma-26b", placement=spilled)

        assert runtime.measured_speed("gemma-26b", resident)[1] == pytest.approx(41.0)
        assert runtime.measured_speed("gemma-26b", spilled)[1] == pytest.approx(12.0)

    def test_the_placement_is_in_the_key_the_design_intent_names(self):
        assert runtime.speed_key(runtime.WRITE_RATE, "gemma-26b", ctx.Placement()) == \
            "llm:write:gemma-26b:gpu"
        assert runtime.speed_key(runtime.WRITE_RATE, "gemma-26b",
                                 ctx.Placement(cpu_expert_layers=8)) == \
            "llm:write:gemma-26b:ncmoe-8"
        assert runtime.speed_key(runtime.WRITE_RATE, "gemma-26b",
                                 ctx.Placement(cpu_expert_layers=ctx.ALL_EXPERTS)) == \
            "llm:write:gemma-26b:cpu-moe"
        assert runtime.speed_key(runtime.WRITE_RATE, "gemma-26b",
                                 ctx.Placement(on_gpu=False)) == "llm:write:gemma-26b:cpu"

    def test_placements_are_never_averaged_into_one_number(self):
        """"Do not merge different placement speeds into one number." Nothing
        writes the backbone-wide key any more, so nothing can."""
        runtime.remember_speed(300.0, 41.0, identity="gemma-26b", placement=ctx.Placement())

        import mc_progress

        assert "llm:write:gemma-26b" not in mc_progress.rates()

    def test_the_specific_placement_is_asked_for_first_and_the_backbone_after(self):
        keys = runtime.speed_keys("krea:write", "gemma-26b",
                                  ctx.Placement(cpu_expert_layers=8))

        assert keys == ("krea:write:gemma-26b:ncmoe-8", "krea:write:gemma-26b", "krea:write")

    def test_a_pre_placement_measurement_is_still_read_back(self):
        """A store written before placements were keyed holds the backbone-wide
        key. It is a stale approximation and a better first guess than none."""
        import mc_progress

        mc_progress.learn("llm:write:gemma-26b", 9.5)

        assert runtime.best_measured("gemma-26b")[1] == pytest.approx(9.5)
        assert runtime.best_measured("gemma-26b")[2] == ""

    def test_the_catalogue_is_told_where_the_number_came_from(self):
        """"measured here: 5.0 tokens/s" with no placement beside it reads as a
        fact about the model."""
        runtime.remember_speed(90.0, 12.0, identity="gemma-26b",
                               placement=ctx.Placement(cpu_expert_layers=8))
        prompt, reply, where = runtime.best_measured("gemma-26b")

        assert (prompt, reply) == (pytest.approx(90.0), pytest.approx(12.0))
        assert where == "ncmoe-8"
        assert runtime.describe_placement_token(where) == "8 expert layers in RAM"

    def test_the_best_placement_on_record_is_the_one_reported(self):
        runtime.remember_speed(90.0, 12.0, identity="gemma-26b",
                               placement=ctx.Placement(cpu_expert_layers=8))
        runtime.remember_speed(300.0, 41.0, identity="gemma-26b", placement=ctx.Placement())

        assert runtime.best_measured("gemma-26b")[1] == pytest.approx(41.0)
        assert runtime.best_measured("gemma-26b")[2] == "gpu"

    def test_a_placement_token_reads_as_a_clause_a_person_can_read(self):
        assert runtime.describe_placement_token("gpu") == "all layers on the GPU"
        assert runtime.describe_placement_token("cpu") == "in system RAM"
        assert runtime.describe_placement_token("cpu-moe") == "experts in system RAM"
        assert runtime.describe_placement_token("ncmoe-1") == "1 expert layer in RAM"
        assert runtime.describe_placement_token("cpu-moe-l20") == (
            "experts in system RAM, 20 layers on the GPU")
        assert runtime.describe_placement_token("") == ""


class TestTheExpertFlagOnTheCommandLine:
    """Which of the two spellings reaches llama-server, and when neither does."""

    @pytest.fixture(autouse=True)
    def forget(self):
        runtime._capabilities.clear()
        yield
        runtime._capabilities.clear()

    @pytest.fixture
    def build(self, tmp_path, monkeypatch, host):
        def announce(*flags):
            configuration = runtime.Config(
                runtime=tmp_path / "llama-server", model=tmp_path / "m.gguf", mmproj=None,
                gpu_index=0, device="CUDA0", gpu_layers="all", context_size=8192,
                context_mode="fixed", context_buffer_gb=4.0, kv_type_k="f16",
                kv_type_v="f16")
            monkeypatch.setattr(runtime, "runtime_supports",
                                lambda flag, config=None, offered=flags: flag in offered)
            return configuration

        return announce

    def test_a_partial_split_is_passed_as_a_count(self, build):
        configuration = build(runtime.N_CPU_MOE_FLAG)

        assert runtime.expert_flags(configuration, ctx.Placement(cpu_expert_layers=8)) == [
            "--n-cpu-moe", "8"]

    def test_every_expert_is_passed_as_the_switch(self, build):
        configuration = build(runtime.N_CPU_MOE_FLAG, runtime.CPU_MOE_FLAG)

        assert runtime.expert_flags(
            configuration, ctx.Placement(cpu_expert_layers=ctx.ALL_EXPERTS)) == ["--cpu-moe"]

    def test_nothing_is_passed_when_nothing_moved(self, build):
        configuration = build(runtime.N_CPU_MOE_FLAG, runtime.CPU_MOE_FLAG)

        assert runtime.expert_flags(configuration, ctx.Placement()) == []

    def test_a_partial_split_is_never_promoted_to_the_older_flag(self, build):
        """The placement was negotiated against an estimate of N layers. Moving
        every expert instead is a different footprint and a different speed than
        the one that was planned, so a build with only the old flag gets
        nothing -- and the ladder never hands it such a placement anyway."""
        configuration = build(runtime.CPU_MOE_FLAG)

        assert runtime.expert_flags(configuration, ctx.Placement(cpu_expert_layers=8)) == []

    def test_an_older_build_is_given_no_expert_flag_at_all(self, build):
        configuration = build()

        for placement in (ctx.Placement(cpu_expert_layers=8),
                          ctx.Placement(cpu_expert_layers=ctx.ALL_EXPERTS)):
            assert runtime.expert_flags(configuration, placement) == []

    def test_it_reaches_the_accelerator_flags_in_front_of_the_card_flags(self, build):
        configuration = build(runtime.N_CPU_MOE_FLAG, runtime.FLASH_ATTENTION_FLAG,
                              runtime.FULL_ATTENTION_WINDOW_FLAG)
        monkeyed = runtime.accelerator_flags(
            configuration, ctx.Placement(gpu_layers=20, cpu_expert_layers=8))

        assert monkeyed.index("--n-cpu-moe") < monkeyed.index("--flash-attn")
        assert monkeyed[monkeyed.index("--n-cpu-moe") + 1] == "8"


class TestRetryingASmallerPlacement:
    """A start that ran out of VRAM is tried again with more headroom, because
    nothing this module knows could have predicted the refusal: the card said
    22.8 GB free and the driver would not give out 17.8 GB of it in one piece.
    """

    def _failing(self, managed, monkeypatch, failures: int):
        """A runtime whose first ``failures`` starts run out of VRAM."""
        attempts: list = []
        real = managed._launch

        def launch(configuration, placement, projector=None):
            attempts.append(placement)
            if len(attempts) <= failures:
                raise runtime._StartFailed("out of memory", out_of_memory=True)
            return real(configuration, placement, projector)

        monkeypatch.setattr(managed, "_launch", launch)
        return attempts

    def test_the_second_attempt_asks_for_less(self, placed, server, tmp_path, monkeypatch):
        managed, _ = server
        # Sized so the extra headroom actually bites: a card this model fits on
        # once and does not fit on with three gigabytes held back.
        configure(monkeypatch, tmp_path, mode="auto", blocks=30)
        set_free(monkeypatch, 17)
        attempts = self._failing(managed, monkeypatch, failures=1)

        managed.client()

        assert len(attempts) == 2
        first, second = attempts
        assert (second.context < first.context
                or runtime._offloaded_layers(second, 30) < runtime._offloaded_layers(first, 30))

    def test_it_gives_up_rather_than_retrying_for_ever(self, placed, server, tmp_path,
                                                       monkeypatch):
        managed, _ = server
        configure(monkeypatch, tmp_path)
        set_free(monkeypatch, 20)
        attempts = self._failing(managed, monkeypatch, failures=99)

        with pytest.raises(RuntimeError):
            managed.client()

        assert len(attempts) == runtime.START_ATTEMPTS

    def test_a_failure_that_is_not_about_memory_is_not_retried(self, placed, server, tmp_path,
                                                               monkeypatch):
        managed, _ = server
        configure(monkeypatch, tmp_path)
        set_free(monkeypatch, 20)
        attempts: list = []

        def launch(configuration, placement, projector=None):
            attempts.append(placement)
            raise runtime._StartFailed("the model file is corrupt")

        monkeypatch.setattr(managed, "_launch", launch)

        with pytest.raises(RuntimeError, match="corrupt"):
            managed.client()

        assert len(attempts) == 1


class TestPlacingAgainstWhatTheDriverHas:
    """The bug that produced four hundred lines of failed starts.

    The host's free-VRAM figure counts the blocks its allocator is holding
    cached; llama.cpp is another process and cannot have them. Placed against
    the host's number, the server is asked to allocate memory that exists only
    inside the WebUI's own address space, and llama.cpp says so and exits.
    """

    def test_the_llm_is_placed_against_the_driver_s_figure(self, placed, tmp_path, monkeypatch):
        configuration = configure(monkeypatch, tmp_path, size_mb=64, blocks=30)
        # Twenty free by the host's accounting, four of them really on offer.
        monkeypatch.setattr(mc_broker, "free_vram_bytes", lambda: 20 * _GB)
        monkeypatch.setattr(mc_broker, "device_free_vram_bytes", lambda: 0.2 * _GB)

        negotiated = runtime.negotiate(configuration)

        assert negotiated.placement.gpu_layers != ctx.ALL_LAYERS

    def test_the_cache_is_handed_back_before_a_server_is_placed(self, placed, server, tmp_path,
                                                               monkeypatch, caplog):
        """Nothing is unloaded and no model moves: what is given up is the
        empty space between them, which is worth doing exactly once."""
        managed, _ = server
        configure(monkeypatch, tmp_path)
        set_free(monkeypatch, 20)
        released = []
        monkeypatch.setattr(mc_broker, "release_cached_vram",
                            lambda: (released.append(True), 4 * _GB)[1])

        with caplog.at_level("INFO", logger="model_chain"):
            managed.client()

        assert released
        assert any("cached VRAM" in record.getMessage() for record in caplog.records)

    def test_a_warm_turn_does_not_empty_anybody_s_cache(self, placed, server, tmp_path,
                                                        monkeypatch):
        """Reuse is the common case and touches nothing at all."""
        managed, _ = server
        configure(monkeypatch, tmp_path)
        set_free(monkeypatch, 20)
        managed.client()

        released = []
        monkeypatch.setattr(mc_broker, "release_cached_vram",
                            lambda: (released.append(True), 0)[1])
        managed.client()

        assert not released


class TestSayingWhatIsWrongWithTheMachine:
    """Two warnings, both about memory this extension does not control.

    It cannot make another application give back system RAM, and it cannot
    stop llama.cpp's own fitter deciding that the way to fit a context is to
    move weights off the card. What it can do is refuse to be silent about
    either, because both look identical from the outside: a load that seems to
    hang and a reply that arrives at five tokens a second.
    """

    def test_a_model_larger_than_free_ram_is_called_out(self, placed, server, tmp_path,
                                                        monkeypatch, caplog):
        import mc_memory

        managed, _ = server
        configure(monkeypatch, tmp_path, size_mb=64)
        set_free(monkeypatch, 20)
        monkeypatch.setattr(mc_memory, "free_ram_bytes", lambda: 8 * 1024**2)

        with caplog.at_level("WARNING", logger="model_chain"):
            managed.client()

        assert any("system RAM is free" in record.getMessage() for record in caplog.records)

    def test_room_to_read_it_is_not_worth_a_word(self, placed, server, tmp_path, monkeypatch,
                                                 caplog):
        import mc_memory

        managed, _ = server
        configure(monkeypatch, tmp_path, size_mb=64)
        set_free(monkeypatch, 20)
        monkeypatch.setattr(mc_memory, "free_ram_bytes", lambda: 64 * 1024**3)

        with caplog.at_level("WARNING", logger="model_chain"):
            managed.client()

        assert not any("system RAM is free" in record.getMessage() for record in caplog.records)

    def test_weights_that_never_reached_the_card_are_called_out(self, placed, tmp_path,
                                                                monkeypatch, caplog):
        """The one check that works on every build, because it reads nothing
        llama.cpp wrote: seventeen gigabytes of weights, and the card's free
        memory fell by four."""
        configuration = configure(monkeypatch, tmp_path, size_mb=64)
        set_free(monkeypatch, 20)
        negotiated = runtime.negotiate(configuration)
        managed = runtime.Runtime()

        with caplog.at_level("WARNING", logger="model_chain"):
            managed._record(configuration, negotiated,
                            int(negotiated.estimate.weights_bytes * 0.2))

        assert any("read it over PCIe" in record.getMessage() for record in caplog.records)

    def test_a_placement_that_landed_says_nothing(self, placed, tmp_path, monkeypatch, caplog):
        configuration = configure(monkeypatch, tmp_path, size_mb=64)
        set_free(monkeypatch, 20)
        negotiated = runtime.negotiate(configuration)
        managed = runtime.Runtime()

        with caplog.at_level("WARNING", logger="model_chain"):
            managed._record(configuration, negotiated, negotiated.estimate.weights_bytes)

        assert not any("PCIe" in record.getMessage() for record in caplog.records)


class TestTheProjectorIsNotFree:
    """A gigabyte and a third of a card the model is already filling, paid on
    every text-only turn — and where llama.cpp finds a gigabyte and a third it
    was not told about is by leaving part of the model in system RAM.
    """

    def _with_projector(self, monkeypatch, tmp_path, configuration):
        projector = tmp_path / "mmproj.gguf"
        projector.write_bytes(b"x" * (1024 * 1024))
        replaced = dataclasses.replace(configuration, mmproj=projector)
        monkeypatch.setattr(runtime, "config", lambda: replaced)
        return replaced

    def test_it_is_counted_against_the_card_when_it_is_loaded(self, placed, tmp_path,
                                                              monkeypatch):
        configuration = self._with_projector(monkeypatch, tmp_path,
                                             configure(monkeypatch, tmp_path))

        assert runtime.projector_bytes(configuration, vision=True) > 1024 * 1024
        assert runtime.projector_bytes(configuration, vision=False) == 0

    def test_a_text_only_request_does_not_load_it(self, placed, server, tmp_path, monkeypatch):
        managed, started = server
        self._with_projector(monkeypatch, tmp_path, configure(monkeypatch, tmp_path))
        set_free(monkeypatch, 20)

        managed.client(needs_vision=False)

        assert started[0][0][2] is None

    def test_a_request_carrying_an_image_replaces_the_server_with_one_that_sees(
            self, placed, server, tmp_path, monkeypatch):
        managed, started = server
        configuration = self._with_projector(monkeypatch, tmp_path,
                                             configure(monkeypatch, tmp_path))
        set_free(monkeypatch, 20)
        managed.client(needs_vision=False)

        managed.client(needs_vision=True)

        assert len(started) == 2
        assert started[1][0][2] == configuration.mmproj


class TestPrimingThePromptCache:
    """The instruction above a Krea brief is the same every roll.

    llama.cpp resumes a cached prompt at its common prefix, so the second roll
    on a server is far cheaper than the first -- from the log this came from,
    on a processor-only placement at 30 tokens a second:

        task 5   | prompt eval time = 35337 ms / 1065 tokens
        task 199 | prompt eval time = 21335 ms /  601 tokens

    Thirty-five seconds against twenty, and the difference is text that was
    known before anybody pressed anything.
    """

    @pytest.fixture(autouse=True)
    def idle(self):
        mc_broker.clear()
        runtime._priming = None
        yield
        runtime._priming = None
        mc_broker.clear()

    class Client:
        def __init__(self):
            self.sent = []

        def stream_chat(self, messages, max_tokens, seed, on_text, *args, **kwargs):
            self.sent.append((messages, max_tokens))
            return "."

    def prime(self, client):
        runtime._prime(client)

    def test_it_sends_the_writer_instruction_and_asks_for_nothing_back(self, host):
        client = self.Client()

        self.prime(client)

        assert len(client.sent) == 1
        messages, tokens = client.sent[0]
        assert tokens == runtime.PRIME_TOKENS
        assert messages[0]["role"] == "system" and messages[0]["content"]
        assert messages[-1]["content"] == ""

    def test_it_does_nothing_while_a_job_holds_the_gpu(self, host):
        """The case that makes this safe to fire from any start. A server
        restarted in the middle of a roll must not be handed a prefill the roll
        would then queue behind -- and the roll holds the workload lock, so
        asking for it as background work is the whole guard."""
        client = self.Client()

        with mc_broker.workload(mc_broker.FAMILY_LLM, "a Krea prompt"):
            self.prime(client)

        assert client.sent == []

    def test_it_does_nothing_while_the_host_is_generating(self, host, monkeypatch):
        client = self.Client()
        monkeypatch.setattr(mc_broker, "host_busy", lambda: True)

        self.prime(client)

        assert client.sent == []

    def test_a_client_that_raises_is_not_a_failed_start(self, host):
        class Broken:
            def stream_chat(self, *args, **kwargs):
                raise RuntimeError("llama-server went away")

        self.prime(Broken())  # must not raise

    def test_it_releases_the_gpu_afterwards(self, host):
        self.prime(self.Client())

        assert mc_broker.active() is None

    def test_only_one_runs_at_a_time(self, host, monkeypatch):
        """A prime holds the workload lock while llama.cpp prefills. Two of
        them queued behind each other would hold it twice as long, for a cache
        the first one already filled."""
        started = threading.Event()
        finish = threading.Event()
        monkeypatch.setattr(runtime, "_prime",
                            lambda client: (started.set(), finish.wait(5)))

        runtime._prime_prompt_cache(object())
        assert started.wait(5)
        first = runtime._priming

        runtime._prime_prompt_cache(object())
        assert runtime._priming is first

        finish.set()
        first.join(timeout=5)

    def test_a_start_schedules_it(self, placed, tmp_path, monkeypatch, server):
        """The scheduling itself, since every other test here stubs it out."""
        managed, _started = server
        scheduled: list = []
        monkeypatch.setattr(runtime, "_prime_prompt_cache", scheduled.append)
        configure(monkeypatch, tmp_path)

        managed.client()

        assert len(scheduled) == 1
