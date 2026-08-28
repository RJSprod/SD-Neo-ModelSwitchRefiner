"""Warm prompt caches: llama.cpp's ``--parallel``, and what it costs.

Every mode here opens with a different system prompt -- Conversation, Prompt
Studio, MiniMax, the Krea writer, the Spatial Composer -- and with one cache
between them each switch re-reads the prefix the previous one had just cached.
llama.cpp will keep a cache per slot and route an incoming prompt to the one
whose prefix matches best, so the fix is a number on the command line rather
than anything this extension has to schedule.

The two things that make it safe are tested hardest:

* ``--ctx-size`` is a **total** that llama.cpp divides among the slots, and
  passing the per-slot number would truncate every prompt to a fraction of what
  was asked for, silently;
* a cache is bought only with VRAM left over once the model already fits, and
  is the first thing sold when it does not -- so switching this on can never be
  the reason a context shrank or a layer left the card.
"""

from __future__ import annotations

import types

import pytest

import mc_broker
import mc_gguf
import mc_llm_context as ctx
import mc_llm_runtime as runtime

from test_llm_context import build_model

_GB = 1024**3


# --------------------------------------------------------------------------- #
# The arithmetic on the command line
# --------------------------------------------------------------------------- #


class Recording:
    """A stand-in for ``subprocess.Popen`` that keeps the command it was given."""

    commands: list[list[str]] = []

    def __init__(self, command, *args, **kwargs):
        Recording.commands.append(list(command))
        self.args = command

    def poll(self):
        return None

    def terminate(self):
        return None

    def wait(self, timeout=None):
        return 0

    def kill(self):
        return None


@pytest.fixture
def launched(monkeypatch, tmp_path):
    """A real ``LlamaProcess.start``, with the process itself recorded."""
    import subprocess

    from prompt_master.inference.llama_process import LlamaProcess

    Recording.commands = []
    monkeypatch.setattr(subprocess, "Popen", Recording)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"")

    def start(slots=1, context=8192):
        process = LlamaProcess()
        process.start(tmp_path / "llama-server", model, None, 0, "CUDA0", context,
                      tmp_path / "llama-server.log", slots=slots)
        return Recording.commands[-1]

    return start


def flag(command, name):
    return command[command.index(name) + 1]


class TestTheContextIsPerSlot:
    """The one way to get this wrong is silent, so it is the most tested."""

    def test_one_slot_is_byte_for_byte_what_it_always_was(self, launched):
        command = launched(slots=1, context=8192)

        assert flag(command, "--parallel") == "1"
        assert flag(command, "--ctx-size") == "8192"

    def test_the_total_is_multiplied_by_the_slots(self, launched):
        """8192 means "each conversation may get 8192". llama.cpp divides
        ``--ctx-size`` by the slot count, so the total is what it is told."""
        command = launched(slots=3, context=8192)

        assert flag(command, "--parallel") == "3"
        assert flag(command, "--ctx-size") == str(8192 * 3)

    def test_it_is_never_the_per_slot_number(self, launched):
        """Passing 8192 with --parallel 3 gives every slot 2730 tokens and
        truncates prompts to it without an error. That is the regression this
        test exists for."""
        command = launched(slots=3, context=8192)

        assert flag(command, "--ctx-size") != "8192"

    def test_a_nonsense_slot_count_is_one(self, launched):
        assert flag(launched(slots=0), "--parallel") == "1"
        assert flag(launched(slots=-4), "--parallel") == "1"


class TestTheEstimateAgrees:
    def test_the_cache_scales_with_the_slots(self, tmp_path):
        model = build_model(tmp_path, blocks=32)
        one = ctx.estimate(model, ctx.Placement(context=8192, slots=1))
        three = ctx.estimate(model, ctx.Placement(context=8192, slots=3))

        assert three.kv_bytes == one.kv_bytes * 3

    def test_the_weights_do_not(self, tmp_path):
        """One copy of the model, whatever the caches. That is the point."""
        model = build_model(tmp_path, blocks=32)
        one = ctx.estimate(model, ctx.Placement(context=8192, slots=1))
        three = ctx.estimate(model, ctx.Placement(context=8192, slots=3))

        assert three.weights_bytes == one.weights_bytes

    def test_total_context_is_the_only_place_the_multiplication_happens(self):
        assert ctx.Placement(context=8192, slots=3).total_context == 24576
        assert ctx.Placement(context=8192, slots=1).total_context == 8192

    def test_capacity_answers_per_slot(self, tmp_path):
        """A buffer that buys 24K across three caches buys each of them 8K, and
        saying 24K would answer a different question from the one asked."""
        model = build_model(tmp_path, blocks=32)
        budget = 6 * _GB
        one = ctx.capacity(model, ctx.Placement(slots=1), budget)
        three = ctx.capacity(model, ctx.Placement(slots=3), budget)

        assert three.theoretical == pytest.approx(one.theoretical / 3, rel=0.01)

    def test_the_ready_line_says_how_many(self):
        said = ctx.Placement(gpu_layers=ctx.ALL_LAYERS, slots=3).describe()

        assert "3 warm prompt caches" in said

    def test_and_says_nothing_at_one(self):
        """Every installation that has not asked for more reads the line it
        always read."""
        said = ctx.Placement(gpu_layers=ctx.ALL_LAYERS, slots=1).describe()

        assert "cache" not in said


# --------------------------------------------------------------------------- #
# What Automatic may and may not do
# --------------------------------------------------------------------------- #


def configure(monkeypatch, tmp_path, *, blocks=32, size_mb=64, context=8192):
    model = build_model(tmp_path, blocks=blocks, size_mb=size_mb, context=131072)
    executable = tmp_path / "llama-server"
    executable.write_bytes(b"")
    configuration = runtime.Config(
        runtime=executable, model=model, mmproj=None, gpu_index=0, device="CUDA0",
        gpu_layers="all", context_size=context, context_mode="fixed",
        context_buffer_gb=4.0, kv_type_k="f16", kv_type_v="f16", mode="gpu")
    monkeypatch.setattr(runtime, "config", lambda role="": configuration)
    return configuration


def set_free(monkeypatch, gigabytes):
    monkeypatch.setattr(mc_broker, "free_vram_bytes", lambda: int(gigabytes * _GB))
    monkeypatch.setattr(mc_broker, "device_free_vram_bytes",
                        lambda index=None: int(gigabytes * _GB))


@pytest.fixture
def card(host, monkeypatch, tmp_path):
    """A card with no reserve, a build that takes --parallel, and no plan."""
    import mc_plan

    mc_broker.clear()
    monkeypatch.setattr(mc_broker, "safety_margin_bytes", lambda: 0)
    monkeypatch.setattr(runtime, "runtime_supports",
                        lambda flag, configuration=None: True)
    monkeypatch.setattr(runtime, "runtime_accepts",
                        lambda flag, value, configuration=None: True)
    mc_plan.clear()
    yield
    mc_broker.clear()
    mc_plan.clear()


class TestAutomaticBuysOnlyTheSpare:
    """The invariant that makes it safe to leave on.

    A warm cache is bought with VRAM left over once the placement already fits.
    It can never be why a context shrank, an expert moved, or a layer left the
    card -- which is the failure this setting would otherwise reintroduce, and
    the one the user reported in exactly those words.
    """

    def test_a_roomy_card_gets_several(self, card, monkeypatch, tmp_path):
        configuration = configure(monkeypatch, tmp_path)
        set_free(monkeypatch, 24)

        negotiated = runtime.negotiate(configuration, reclaim=False)

        assert negotiated.fits
        assert negotiated.placement.slots > 1

    def test_a_card_that_only_just_fits_gets_one(self, card, monkeypatch, tmp_path):
        """No spare, no cache. Not "a smaller context so a cache fits"."""
        configuration = configure(monkeypatch, tmp_path)
        described = mc_gguf.describe(configuration.model)
        exact = ctx.estimate(configuration.model, ctx.Placement(
            gpu_layers=ctx.ALL_LAYERS, context=8192), described)
        set_free(monkeypatch, exact.total_bytes / _GB)

        negotiated = runtime.negotiate(configuration, described, reclaim=False)

        assert negotiated.placement.slots == 1

    def test_a_degraded_placement_never_gains_one(self, card, monkeypatch, tmp_path):
        """If the ladder had to move anything, there was no spare by
        definition, and a cache bought here would have been bought out of the
        thing that was just given up."""
        configuration = configure(monkeypatch, tmp_path)
        set_free(monkeypatch, 0.05)

        negotiated = runtime.negotiate(configuration, reclaim=False)

        assert negotiated.placement.slots == 1

    def test_the_context_is_never_shrunk_to_pay_for_a_cache(self, card, monkeypatch,
                                                            tmp_path):
        configuration = configure(monkeypatch, tmp_path)
        set_free(monkeypatch, 24)

        negotiated = runtime.negotiate(configuration, reclaim=False)

        assert negotiated.placement.context == 8192

    def test_a_processor_placement_asks_for_none(self, card, monkeypatch, tmp_path):
        """There is no VRAM decision to make, so there is nothing to buy with."""
        configuration = configure(monkeypatch, tmp_path)
        object.__setattr__(configuration, "mode", "cpu")
        object.__setattr__(configuration, "device", "none")

        negotiated = runtime.negotiate(configuration, reclaim=False)

        assert negotiated.placement.slots == 1


class TestCachesAreSoldFirst:
    """Cheapest thing on the ladder to give up, so the first thing given up.

    A cache that is not there costs one prompt re-read. A context that is not
    there costs conversation length; an expert in system RAM costs speed on
    every token that consults it; a layer that is not on the card costs speed on
    every token full stop.
    """

    def test_an_explicit_count_that_does_not_fit_is_reduced(self, card, monkeypatch,
                                                            tmp_path, host):
        from modules import shared

        shared.opts.model_chain_llm_prompt_caches = "6"
        configuration = configure(monkeypatch, tmp_path)
        described = mc_gguf.describe(configuration.model)
        exact = ctx.estimate(configuration.model, ctx.Placement(
            gpu_layers=ctx.ALL_LAYERS, context=8192, slots=2), described)
        set_free(monkeypatch, exact.total_bytes / _GB)

        negotiated = runtime.negotiate(configuration, described, reclaim=False)

        assert 1 <= negotiated.placement.slots <= 2
        assert negotiated.fits

    def test_the_context_survives_when_a_cache_can_be_sold_instead(
            self, card, monkeypatch, tmp_path, host):
        """The ordering, stated directly: given a choice between one fewer
        cache and a shorter context, the cache goes."""
        from modules import shared

        shared.opts.model_chain_llm_prompt_caches = "3"
        configuration = configure(monkeypatch, tmp_path)
        described = mc_gguf.describe(configuration.model)
        exact = ctx.estimate(configuration.model, ctx.Placement(
            gpu_layers=ctx.ALL_LAYERS, context=8192, slots=1), described)
        set_free(monkeypatch, exact.total_bytes / _GB)

        negotiated = runtime.negotiate(configuration, described, reclaim=False)

        assert negotiated.placement.slots == 1
        assert negotiated.placement.context == 8192

    def test_it_says_what_it_gave_up(self, card, monkeypatch, tmp_path, host):
        from modules import shared

        shared.opts.model_chain_llm_prompt_caches = "4"
        configuration = configure(monkeypatch, tmp_path)
        described = mc_gguf.describe(configuration.model)
        exact = ctx.estimate(configuration.model, ctx.Placement(
            gpu_layers=ctx.ALL_LAYERS, context=8192, slots=1), described)
        set_free(monkeypatch, exact.total_bytes / _GB)

        negotiated = runtime.negotiate(configuration, described, reclaim=False)

        assert any("warm prompt caches reduced" in note for note in negotiated.notes)


class TestABuildThatWillNotTakeThem:
    def test_a_runtime_without_the_flag_keeps_one_cache(self, card, monkeypatch,
                                                        tmp_path, host):
        from modules import shared

        shared.opts.model_chain_llm_prompt_caches = "3"
        configuration = configure(monkeypatch, tmp_path)
        set_free(monkeypatch, 24)
        monkeypatch.setattr(runtime, "runtime_supports",
                            lambda flag, configuration=None: flag != runtime.PARALLEL_FLAG)

        negotiated = runtime.negotiate(configuration, reclaim=False)

        assert negotiated.placement.slots == 1

    def test_a_value_refused_at_startup_is_not_asked_for_again(self, card, monkeypatch,
                                                               tmp_path, host):
        """The accelerator wins that trade: it is a setting the user chose and
        it pays back on every token, where a cache is worth one prompt."""
        from modules import shared

        shared.opts.model_chain_llm_prompt_caches = "3"
        configuration = configure(monkeypatch, tmp_path)
        set_free(monkeypatch, 24)
        monkeypatch.setattr(
            runtime, "runtime_accepts",
            lambda flag, value, configuration=None: flag != runtime.PARALLEL_FLAG)

        negotiated = runtime.negotiate(configuration, reclaim=False)

        assert negotiated.placement.slots == 1

    def test_the_placement_and_the_command_line_agree(self, card, monkeypatch,
                                                      tmp_path, host):
        """A footprint priced for three caches and a server started with one
        would be two descriptions of one placement."""
        from modules import shared

        shared.opts.model_chain_llm_prompt_caches = "3"
        configuration = configure(monkeypatch, tmp_path)
        set_free(monkeypatch, 24)
        monkeypatch.setattr(runtime, "runtime_supports",
                            lambda flag, configuration=None: False)

        negotiated = runtime.negotiate(configuration, reclaim=False)

        assert negotiated.placement.slots == runtime._slots_for(
            configuration, negotiated.placement)


class TestTheServerIsAskedWhatItBuilt:
    """``--ctx-size`` is a total llama.cpp divides, and getting it wrong is
    silent -- prompts are truncated to ``n_ctx_slot`` without an error. So what
    the server says it built is read back and compared."""

    def report(self, tmp_path, text):
        log = tmp_path / "llama-server.log"
        log.write_text(text, encoding="utf-8")
        return log

    def test_a_matching_report_says_nothing(self, tmp_path, caplog):
        log = self.report(tmp_path, "srv init: initializing, n_slots = 3, "
                                    "n_ctx_slot = 8192, kv_unified = 'false'\n")

        with caplog.at_level("WARNING"):
            runtime._check_slots(log, 0, ctx.Placement(context=8192, slots=3))

        assert not caplog.messages

    def test_a_short_slot_is_reported(self, tmp_path, caplog):
        """The failure the readback exists for: three slots of 2730 tokens
        where 8192 each was asked for."""
        log = self.report(tmp_path, "srv init: initializing, n_slots = 3, "
                                    "n_ctx_slot = 2730, kv_unified = 'false'\n")

        with caplog.at_level("WARNING"):
            runtime._check_slots(log, 0, ctx.Placement(context=8192, slots=3))

        assert any("will be truncated" in line for line in caplog.messages)

    def test_a_different_slot_count_is_reported(self, tmp_path, caplog):
        log = self.report(tmp_path, "srv init: initializing, n_slots = 1, "
                                    "n_ctx_slot = 8192, kv_unified = 'false'\n")

        with caplog.at_level("WARNING"):
            runtime._check_slots(log, 0, ctx.Placement(context=8192, slots=3))

        assert caplog.messages

    def test_a_log_that_says_nothing_is_not_an_error(self, tmp_path, caplog):
        log = self.report(tmp_path, "nothing useful here\n")

        with caplog.at_level("WARNING"):
            runtime._check_slots(log, 0, ctx.Placement(context=8192, slots=3))

        assert not caplog.messages


class TestChangingItRestartsTheServer:
    def test_the_slot_count_is_part_of_the_signature(self, tmp_path, monkeypatch, host):
        configuration = configure(monkeypatch, tmp_path)
        one = runtime._signature_of(configuration, None,
                                    ctx.Placement(context=8192, slots=1))
        three = runtime._signature_of(configuration, None,
                                      ctx.Placement(context=8192, slots=3))

        assert one != three

    def test_an_unchanged_count_is_the_same_signature(self, tmp_path, monkeypatch, host):
        configuration = configure(monkeypatch, tmp_path)

        assert runtime._signature_of(configuration, None,
                                     ctx.Placement(context=8192, slots=3)) == \
            runtime._signature_of(configuration, None,
                                  ctx.Placement(context=8192, slots=3))
