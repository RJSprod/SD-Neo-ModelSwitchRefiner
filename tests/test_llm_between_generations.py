"""Between generations a language model on Forge's card takes what is free.

The user's report, from their own logs: the language model switched to an RTX
3090 with 23,336 MiB free -- the image model unloaded from it by hand -- and a
10 GB model placed with no layers on the card at all. The plan the last image
generation had published, nearly three hours earlier, still protected 20.9 GB
of the card for a batch of four nobody was generating, because nothing ever took
a plan down. Their rule, which this file holds: "There should be no blocks if
the vram is free. If i try to run another image generation, thats when the LLM
could get kicked out of my 3090, but not while there is free space!"

So a plan is in force only while a generation runs (``mc_plan.in_force``);
between generations a placement on the image card is sized against what is
free; and the next generation's start takes back what its plan needs
(``mc_plan.image_claim_bytes``), stopping a language model that is in the way
and leaving one that fits beside the plan where it is. Within a generation
nothing changed: the plan classes in ``tests/test_llm_runtime.py``, which now
say a generation is running, hold that.
"""

from __future__ import annotations

import dataclasses

import pytest

import mc_broker
import mc_gguf
import mc_llm_context as ctx
import mc_llm_runtime as runtime
import mc_memory
import mc_plan
from test_llm_runtime import configure, placed, publish_plan, set_free  # noqa: F401
from test_orchestration import DEFAULTS, UI_ORDER, make_p

MiB = 1024 ** 2
_GB = 1024 ** 3


class Gemma4Q2KP:
    """What llama.cpp printed about the user's model in their llama-server.log."""

    file_bytes = int(10186.10 * MiB)  # "CPU_Mapped model buffer size = 10186.10 MiB"
    block_count = 30                  # "gemma4.block_count u32 = 30"
    usable = True
    context_length = 262144
    embedding_length = 2816
    expert_count = 128
    expert_used_count = 8
    mixture_of_experts = True
    expert_share = 0.85               # 22.8 of 25.2 B parameters are experts


KV_ONE_SLOT = (85 + 850) * MiB        # the two KV buffers at 8,192 tokens, one slot
COMPUTE = 114 * MiB                   # the full offload's compute buffer, on their Arc
MMPROJ_WORST = 1295.16 * MiB          # llama.cpp's own worst case for the projector


def measured(model, placement, header=None):
    """The extension's estimate, with the user's model's measured sizes in it.

    Checked against llama.cpp: six slots of the whole model come to 15.54 GiB,
    and llama.cpp projected 15,910 MiB for the same placement on their Arc.
    """
    header = Gemma4Q2KP()
    on_card = placement.on_gpu and placement.gpu_layers != ctx.NO_LAYERS
    if not on_card:
        share = 0.0
    elif placement.gpu_layers == ctx.ALL_LAYERS:
        share = 1.0
    else:
        share = min(int(placement.gpu_layers), header.block_count) / header.block_count
    return ctx.Estimate(
        model=model, context=placement.context, ceiling=header.context_length,
        weights_bytes=ctx.weights_bytes(header, placement),
        kv_bytes=int(KV_ONE_SLOT * max(int(placement.slots), 1) * share),
        compute_bytes=COMPUTE if on_card else 0,
        kv_bytes_per_token=KV_ONE_SLOT * share / 8192.0, calibrated=False,
        placement=placement)


@pytest.fixture
def the_3090(placed, tmp_path, monkeypatch):
    """The user's card at 20:40: 22.5 GB free with the image model unloaded,
    the last generation's plan (20.9 GB) still published, GPU / VRAM Only, the
    vision projector wanted."""
    configuration = configure(monkeypatch, tmp_path, context=8192, mode="fixed",
                              device_mode="gpu")
    mmproj = tmp_path / "mmproj.gguf"
    with open(mmproj, "wb") as handle:  # sized, not written
        handle.truncate(int(MMPROJ_WORST / 1.25))
    configuration = dataclasses.replace(configuration, mmproj=mmproj)
    monkeypatch.setattr(runtime, "config", lambda: configuration)
    monkeypatch.setattr(ctx, "estimate", measured)
    monkeypatch.setattr(mc_gguf, "describe", lambda path: Gemma4Q2KP())
    monkeypatch.setattr(runtime, "runtime_supports", lambda flag, config=None: True)
    # vram_headroom_bytes() asked with no pass size has a floor of 1 GB.
    monkeypatch.setattr(mc_broker, "safety_margin_bytes", lambda: 1 * _GB)
    set_free(monkeypatch, 22.5)
    publish_plan(stage_1_gb=20.9, total_gb=22.5, monkeypatch=monkeypatch)
    assert runtime.shares_the_image_card(runtime.card_of(configuration), configuration)
    return configuration


def generation(monkeypatch, running: bool) -> None:
    """Whether Forge is running a job, through the reading the code itself uses."""
    monkeypatch.setattr(mc_broker, "host_busy", lambda: running)


def notes(result) -> str:
    return " ".join(result.notes)


# --------------------------------------------------------------------------- #
# When a plan is in force
# --------------------------------------------------------------------------- #


class TestAPlanIsInForceOnlyWhileAGenerationRuns:
    def test_with_no_plan_none_is(self, host):
        mc_plan.clear()

        assert mc_plan.in_force() is None

    def test_a_published_plan_is_while_forge_runs_a_job(self, host, monkeypatch):
        plan = publish_plan(stage_1_gb=14.0, monkeypatch=monkeypatch)
        host.shared.state.job = "txt2img"

        assert mc_plan.generating()
        assert mc_plan.in_force() is plan

    def test_and_is_not_once_the_job_is_over_though_it_is_still_the_last_plan(
            self, host, monkeypatch):
        """Kept, not cleared: the panel still shows it, and the next
        generation's plan is compared with it."""
        plan = publish_plan(stage_1_gb=14.0, monkeypatch=monkeypatch)
        host.shared.state.job = ""
        host.shared.state.job_count = 0

        assert not mc_plan.generating()
        assert mc_plan.in_force() is None
        assert mc_plan.current() is plan

    def test_a_question_that_cannot_be_asked_keeps_the_protection_on(self, host,
                                                                     monkeypatch):
        """Wrong one way costs a smaller language model; the other way, an image
        pass that runs out of memory."""
        def unreadable():
            raise RuntimeError("no state to read")

        monkeypatch.setattr(mc_broker, "host_busy", unreadable)
        plan = publish_plan(stage_1_gb=14.0, monkeypatch=monkeypatch)

        assert mc_plan.generating()
        assert mc_plan.in_force() is plan


# --------------------------------------------------------------------------- #
# The user's card
# --------------------------------------------------------------------------- #


class TestTheUsersCard:
    def test_between_generations_the_whole_model_goes_on_it(self, the_3090, monkeypatch):
        generation(monkeypatch, running=False)

        result = runtime.negotiate(the_3090, Gemma4Q2KP(), vision=True)

        assert result.placement.gpu_layers == ctx.ALL_LAYERS
        assert not result.placement.cpu_experts
        assert result.fits
        assert result.notes == ()

    def test_while_a_generation_runs_the_same_plan_still_holds_the_card(
            self, the_3090, monkeypatch):
        """Unchanged inside a generation, and it is the user's log line for
        line: the plan leaves 1.6 GB, less than the 1.0 GB margin and the
        1.26 GB projector together, so not one layer fits."""
        generation(monkeypatch, running=True)

        result = runtime.negotiate(the_3090, Gemma4Q2KP(), vision=True)

        assert result.placement.gpu_layers == ctx.NO_LAYERS
        assert "the whole model was placed in system RAM" in notes(result)
        assert "GPU / VRAM Only was chosen and could not be satisfied" in notes(result)

    def test_what_a_placement_may_spend_between_generations_is_what_is_free(
            self, the_3090, monkeypatch):
        card = runtime.card_of(the_3090)

        generation(monkeypatch, running=False)
        between = runtime._spendable(0, card, configuration=the_3090)
        generation(monkeypatch, running=True)
        during = runtime._spendable(0, card, configuration=the_3090)

        assert between == int(22.5 * _GB)
        assert during == int(22.5 * _GB) - int(20.9 * _GB)

    def test_a_learned_cap_goes_with_the_plan(self, the_3090, monkeypatch):
        """A ceiling on the language model *beside* the plan, so not one on
        it beside nothing -- with LLM priority's spelling of the question too."""
        mc_plan.record_miss("Stage 1", 1 * _GB, llm_bytes=3 * _GB, evicted=True)
        learned = mc_plan.learned_cap_bytes()
        assert 0 < learned < int(1.6 * _GB)
        card = runtime.card_of(the_3090)

        generation(monkeypatch, running=False)
        assert runtime._spendable(0, card, configuration=the_3090) == int(22.5 * _GB)
        assert runtime._spendable(0, card, image_budget=False,
                                  configuration=the_3090) == int(22.5 * _GB)
        generation(monkeypatch, running=True)
        assert runtime._spendable(0, card, configuration=the_3090) == learned
        assert runtime._spendable(0, card, image_budget=False,
                                  configuration=the_3090) == learned


# --------------------------------------------------------------------------- #
# A server that is already running
# --------------------------------------------------------------------------- #


class TestARunningServer:
    def server(self, layers):
        held = runtime.Runtime()
        held._placement = ctx.Placement(gpu_layers=layers, context=8192, kv_type_k="f16",
                                        kv_type_v="f16", on_gpu=True)
        return held

    def test_one_placed_in_free_vram_is_not_pushed_down_by_the_last_plan(
            self, the_3090, monkeypatch):
        generation(monkeypatch, running=False)

        assert not self.server(ctx.ALL_LAYERS)._overspending(15 * _GB)

    def test_until_a_generation_runs_and_needs_the_room(self, the_3090, monkeypatch):
        generation(monkeypatch, running=True)

        assert self.server(ctx.ALL_LAYERS)._overspending(15 * _GB)

    def test_one_squeezed_by_a_generation_grows_once_it_is_over(self, the_3090,
                                                               monkeypatch):
        """Placed with nothing on the card while a generation ran; afterwards
        the card is emptied by hand and the next request grows into it."""
        generation(monkeypatch, running=True)
        server = self.server(ctx.NO_LAYERS)
        server._note_placement()

        assert not server._outgrown(the_3090, 0), "inside the generation the plan holds it"

        generation(monkeypatch, running=False)

        assert server._outgrown(the_3090, 0)

    def test_a_placement_between_generations_answers_no_plan(self, the_3090, monkeypatch):
        server = self.server(ctx.ALL_LAYERS)

        generation(monkeypatch, running=False)
        server._note_placement()
        assert server._placed_for is None

        generation(monkeypatch, running=True)
        server._note_placement()
        assert server._placed_for == mc_plan.current().identity()

    def test_between_generations_no_boundary_can_have_moved(self, the_3090, monkeypatch):
        generation(monkeypatch, running=True)
        server = self.server(ctx.ALL_LAYERS)
        server._note_placement()
        publish_plan(stage_1_gb=8.0, total_gb=22.5, monkeypatch=monkeypatch)

        assert server._boundary_moved(), "a different plan, while it is in force, is one"
        generation(monkeypatch, running=False)
        assert not server._boundary_moved()


# --------------------------------------------------------------------------- #
# The next generation takes its room back
# --------------------------------------------------------------------------- #


def image_holds(monkeypatch, gigabytes):
    def held(family, card=mc_broker.ANY_CARD):
        return int(gigabytes * _GB) if family == mc_broker.FAMILY_IMAGE else 0

    monkeypatch.setattr(mc_broker, "held_bytes", held)


class TestWhatTheNextGenerationClaims:
    def test_with_no_plan_nothing(self, host):
        mc_plan.clear()

        assert mc_plan.image_claim_bytes(15 * _GB) == 0

    def test_the_plans_room_less_what_the_image_side_already_holds(self, host, monkeypatch):
        """22.5 obtainable, 1.6 of it left to a language model, 12 already the
        image side's: it needs 8.9 more, which is the protected 20.9 less 12."""
        publish_plan(stage_1_gb=20.9, total_gb=22.5, monkeypatch=monkeypatch)
        image_holds(monkeypatch, 12)

        assert mc_plan.image_claim_bytes(5 * _GB) == int(20.9 * _GB) - 12 * _GB

    def test_a_learned_cap_claims_more(self, host, monkeypatch):
        publish_plan(stage_1_gb=20.9, total_gb=22.5, monkeypatch=monkeypatch)
        image_holds(monkeypatch, 0)
        before = mc_plan.image_claim_bytes(5 * _GB)
        mc_plan.record_miss("Stage 1", 1 * _GB, llm_bytes=3 * _GB, evicted=True)

        assert mc_plan.image_claim_bytes(5 * _GB) == (int(22.5 * _GB)
                                                      - mc_plan.learned_cap_bytes())
        assert mc_plan.image_claim_bytes(5 * _GB) > before

    def test_off_claims_all_of_it_but_the_image_sides_own(self, host, monkeypatch):
        host.shared.opts.model_chain_llm_cap_mode = mc_plan.CAP_OFF
        publish_plan(stage_1_gb=20.9, total_gb=22.5, monkeypatch=monkeypatch)
        image_holds(monkeypatch, 2)

        assert mc_plan.image_claim_bytes(5 * _GB) == int(22.5 * _GB) - 2 * _GB

    def test_a_card_that_cannot_be_measured_claims_nothing(self, host, monkeypatch):
        publish_plan(stage_1_gb=20.9, monkeypatch=None)
        monkeypatch.setattr(mc_plan, "usable_vram_bytes", lambda ours=0: 0)

        assert mc_plan.image_claim_bytes(5 * _GB) == 0


class Holder:
    """A language model on the image card, as the broker's LLM reclaimer."""

    def __init__(self, holds):
        self.holds = int(holds)
        self.reasons: list[str] = []

    def release(self, needed_bytes, reason="", card=mc_broker.ANY_CARD):
        self.reasons.append(reason)
        freed, self.holds = self.holds, 0
        return freed

    def resident_bytes(self, card=mc_broker.ANY_CARD):
        return self.holds

    def describe(self):
        return "the LLM"


@pytest.fixture
def at_the_start(chain, host, monkeypatch):
    """A generation starting on the user's card, its plan 20.9 GB, with a
    language model already there; what is free is what that model leaves."""
    mc_broker.clear()
    monkeypatch.setattr(mc_broker, "safety_margin_bytes", lambda: 0)
    plan = mc_plan.Plan((mc_plan.Phase(mc_plan.STAGE_1, mc_plan.KIND_IMAGE, "Stage 1",
                                       int(20.9 * _GB), detail="krea2"),), 1024, 1024)
    monkeypatch.setattr(mc_plan, "build_for", lambda p, **kwargs: plan)
    monkeypatch.setattr(mc_plan, "usable_vram_bytes", lambda ours=0: int(22.5 * _GB))
    image_holds(monkeypatch, 0)
    made_room: list = []
    monkeypatch.setattr(mc_memory, "make_vram_room",
                        lambda *args, **kwargs: made_room.append(args) or 0)
    card = max(mc_broker.image_device_index(), 0)

    def with_a_model_holding(gigabytes):
        holder = Holder(gigabytes * _GB)
        mc_broker.register_reclaimer(mc_broker.FAMILY_LLM, holder)
        mc_broker.declare(mc_broker.FAMILY_LLM, "llm", "the LLM", int(gigabytes * _GB),
                          card=card)
        free = int(22.5 * _GB) - int(gigabytes * _GB)
        monkeypatch.setattr(mc_broker, "free_vram_bytes", lambda: free)
        monkeypatch.setattr(mc_broker, "device_free_vram_bytes", lambda index=None: free)
        chain.script.before_process(make_p(host), *[DEFAULTS[name] for name in UI_ORDER])
        return holder

    yield with_a_model_holding
    mc_broker.clear()
    mc_broker.unregister_reclaimer(mc_broker.FAMILY_LLM)


class TestTheGenerationStart:
    def test_a_language_model_in_the_way_is_stopped_before_anything_loads(
            self, at_the_start):
        """The whole card's worth, placed while it was free: 15.5 GB where the
        plan leaves 1.6. The stop is the claim's, named as such."""
        holder = at_the_start(15.5)

        assert holder.reasons == ["the image generation starting on this card"]

    def test_one_that_fits_beside_the_plan_is_left_where_it_is(self, at_the_start):
        """'but not while there is free space'"""
        holder = at_the_start(1.0)

        assert holder.reasons == []

    def test_with_no_language_model_on_the_card_the_claim_is_not_even_worked_out(
            self, chain, host, monkeypatch):
        """Nearly every generation of somebody who does not use that half of
        the extension: the call is exactly the one it always was. Recorded
        rather than raised, because the hook swallows what it raises."""
        asked: list = []
        monkeypatch.setattr(mc_memory, "llm_vram_on_the_image_card", lambda: 0)
        monkeypatch.setattr(mc_plan, "image_claim_bytes",
                            lambda llm_bytes=0: asked.append(llm_bytes) or 5 * _GB)

        assert chain.module.ScriptModelChain._room_to_claim() == 0
        assert asked == []

    def test_with_one_it_is_the_plans_claim(self, chain, host, monkeypatch):
        asked: list = []
        monkeypatch.setattr(mc_memory, "llm_vram_on_the_image_card", lambda: 7 * _GB)
        monkeypatch.setattr(mc_plan, "image_claim_bytes",
                            lambda llm_bytes=0: asked.append(llm_bytes) or 3 * _GB)

        assert chain.module.ScriptModelChain._room_to_claim() == 3 * _GB
        assert asked == [7 * _GB]


# --------------------------------------------------------------------------- #
# What the log says
# --------------------------------------------------------------------------- #


THE_3090_START = """
load_tensors: offloading 0 repeating layers to GPU
load_tensors: offloaded 0/31 layers to GPU
load_tensors:   CPU_Mapped model buffer size = 10186.10 MiB
llama_kv_cache:        CPU KV buffer size =    85.00 MiB
llama_kv_cache:        CPU KV buffer size =   850.00 MiB
"""

A_SHORT_CHANGED_START = """
load_tensors: offloaded 31/31 layers to GPU
load_tensors:        CUDA0 model buffer size = 11000.00 MiB
load_tensors:   CPU_Mapped model buffer size =  6000.00 MiB
"""


class TestTheStartSaysWhatTheNextGenerationWillDo:
    def placement(self, configuration):
        return runtime.negotiate(configuration, Gemma4Q2KP(), vision=True).placement

    def test_a_start_bigger_than_the_last_plan_allows_says_it_will_be_stopped(
            self, the_3090, monkeypatch):
        generation(monkeypatch, running=False)

        said = runtime._between_generations(the_3090, self.placement(the_3090),
                                            Gemma4Q2KP(), vision=True)

        assert said.startswith("no image generation is running")
        assert "about 16.8 GB" in said and "1.6 GB" in said
        assert "the next image generation on this card stops it" in said

    def test_one_that_fits_beside_the_last_plan_is_not_worth_a_line(self, the_3090,
                                                                    monkeypatch):
        publish_plan(stage_1_gb=4.0, total_gb=22.5, monkeypatch=monkeypatch)
        generation(monkeypatch, running=False)

        assert runtime._between_generations(the_3090, self.placement(the_3090),
                                            Gemma4Q2KP(), vision=True) == ""

    def test_nor_one_made_during_a_generation(self, the_3090, monkeypatch):
        generation(monkeypatch, running=False)
        placement = self.placement(the_3090)
        generation(monkeypatch, running=True)

        assert runtime._between_generations(the_3090, placement, Gemma4Q2KP(),
                                            vision=True) == ""

    def test_nor_one_with_no_plan_to_compare_it_with(self, the_3090, monkeypatch):
        generation(monkeypatch, running=False)
        placement = self.placement(the_3090)
        mc_plan.clear()

        assert runtime._between_generations(the_3090, placement, Gemma4Q2KP(),
                                            vision=True) == ""

    def test_nor_one_on_another_card(self, the_3090, monkeypatch):
        generation(monkeypatch, running=False)
        placement = self.placement(the_3090)
        # Forge's card named, or every card would count as it: an unknown card
        # is treated as the image card on purpose (shares_the_image_card).
        monkeypatch.setattr(mc_broker, "image_device_index", lambda: 0)
        elsewhere = dataclasses.replace(the_3090, gpu_index=1, device="CUDA1")
        assert not runtime.shares_the_image_card(runtime.card_of(elsewhere), elsewhere)

        assert runtime._between_generations(elsewhere, placement, Gemma4Q2KP(),
                                            vision=True) == ""


class TestTheIdleCardLine:
    def test_it_names_the_generation_when_the_generations_plan_was_the_limit(
            self, the_3090, monkeypatch, caplog):
        generation(monkeypatch, running=True)

        with caplog.at_level("INFO"):
            runtime._warn_about_an_idle_card(the_3090, "0")

        assert "will do no work" in caplog.text
        assert "the image generation running on it keeps 20.9 GB" in caplog.text
        assert "leaves a language model 1.6 GB" in caplog.text
        assert "nothing was free" not in caplog.text

    def test_it_says_nothing_was_free_only_when_nothing_was(self, the_3090, monkeypatch,
                                                            caplog):
        generation(monkeypatch, running=False)
        set_free(monkeypatch, 0.5)

        with caplog.at_level("INFO"):
            runtime._warn_about_an_idle_card(the_3090, "0")

        assert "nothing was free to offload into" in caplog.text

    def test_the_card_is_named_without_the_free_vram_recorded_at_setup(
            self, the_3090, monkeypatch, caplog):
        """From the user's log: "NVIDIA GeForce RTX 3090 (24575 MiB, 23336 MiB
        free) will do no work in this placement -- nothing was free"."""
        named = dataclasses.replace(
            the_3090, device_name="NVIDIA GeForce RTX 3090 (24575 MiB, 23336 MiB free)")
        generation(monkeypatch, running=False)
        set_free(monkeypatch, 0.5)

        with caplog.at_level("INFO"):
            runtime._warn_about_an_idle_card(named, "0")

        assert "NVIDIA GeForce RTX 3090 will do no work" in caplog.text
        assert "23336 MiB free" not in caplog.text


class TestTheSpillWarning:
    def report(self, server, monkeypatch, layers, text, caplog):
        placement = ctx.Placement(gpu_layers=layers, context=8192, on_gpu=True)
        monkeypatch.setattr(server, "_report_residency", lambda negotiated: None)
        server.report = runtime.Report(placement=placement, offload=runtime.read_offload(text))
        negotiated = runtime.Negotiation(placement, measured(None, placement), (), True)
        with caplog.at_level("INFO"):
            server._report_offload(negotiated)
        return caplog.text

    def test_a_placement_that_asked_for_no_layers_is_not_blamed_on_the_card(
            self, placed, monkeypatch, caplog):
        """The user's start: this extension asked for no layers, llama.cpp
        placed exactly that, and the warning sent them to nvidia-smi."""
        text = self.report(runtime.Runtime(), monkeypatch, ctx.NO_LAYERS, THE_3090_START,
                           caplog)

        assert "llama.cpp reports 0/31 layers on the GPU" in text
        assert "less room than this extension can see" not in text

    def test_one_that_asked_for_everything_and_did_not_get_it_still_is(
            self, placed, monkeypatch, caplog):
        text = self.report(runtime.Runtime(), monkeypatch, ctx.ALL_LAYERS,
                           A_SHORT_CHANGED_START, caplog)

        assert "less room than this extension can see" in text
